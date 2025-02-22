use super::error::AppError;
use crate::audio::opus::OpusEncoder;
use crate::state::AppState;
use anyhow::{Context, Result};
use axum::{body::Body, extract::State, http::StatusCode, response::Response, Json};
use bytes::Bytes;
use candle_core::{IndexOp, Tensor, D};
use fish_speech_core::audio::{functional::resample, wav::write_pcm_as_wav};
use fish_speech_core::config::{WhichFishVersion, WhichLM, WhichModel};
use fish_speech_core::lm::generate::{generate_blocking_with_hidden, generate_static_batch};
use fish_speech_core::lm::sampling::SamplingArgs;
use fish_speech_core::text::{clean::preprocess_text, prompt::PromptEncoder};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

// Blocking token generation
pub async fn server_lm_generate_blocking(
    state: Arc<AppState>,
    encoded_input: &Tensor,
    sampling_args: &SamplingArgs,
    n_conditioning_tokens: usize,
    collect_hidden_states: bool,
) -> Result<(Tensor, Option<Tensor>), anyhow::Error> {
    let mut model = state.lm.model.lock().await;
    let (tokens, hidden_states) = generate_blocking_with_hidden(
        &mut model,
        &encoded_input,
        state.lm.max_new_tokens,
        sampling_args,
        collect_hidden_states,
        true,
    )
    .context("Failed to generate tokens")?;

    let mut tokens = tokens;
    let mut hidden_states = hidden_states;
    // It's the caller's responsibility to do final clear
    model.clear_slow_caches_until(n_conditioning_tokens)?;
    if tokens.dim(D::Minus1)? == state.lm.max_new_tokens {
        println!("Failed generation suspected. Rerolling once");
        let (new_tokens, new_hidden_states) = generate_blocking_with_hidden(
            &mut model,
            &encoded_input,
            state.lm.max_new_tokens,
            sampling_args,
            collect_hidden_states,
            true,
        )
        .context("Failed to generate tokens")?;
        if new_tokens.dim(D::Minus1)? != state.lm.max_new_tokens {
            tokens = new_tokens;
            hidden_states = new_hidden_states;
        } else {
            anyhow::bail!(
                "Encoded input failed for second time. Bailing: {:?}",
                encoded_input
            );
        }
    }

    let tokens = match state.lm.model_type {
        WhichLM::DualAR | WhichLM::Fish(WhichFishVersion::Fish1_5) => tokens,
        _ => tokens
            .broadcast_sub(&Tensor::ones_like(&tokens).context("Failed to create ones tensor")?)
            .context("Failed to broadcast subtract")?,
    };

    Ok((tokens, hidden_states))
}

pub async fn generate_pcm_batched(
    state: Arc<AppState>,
    encoded_input: &[Tensor],
) -> anyhow::Result<Tensor> {
    let mut model = state.lm.model.lock().await;

    // No sampling configurable for now
    let (sequences, _) = generate_static_batch(
        &mut model,
        encoded_input,
        state.lm.max_new_tokens,
        true,
        state.lm.default_sampling_args.clone(),
    )?;

    model.clear_slow_layer_caches();
    // By invariant, batch items are returned in order
    let semantic_tokens = Tensor::cat(&sequences, D::Minus1)?;
    vocode_semantic_tokens(state.clone(), &semantic_tokens).await
}

pub async fn vocode_semantic_tokens(
    state: Arc<AppState>,
    semantic_tokens: &Tensor,
) -> anyhow::Result<Tensor> {
    let vocoder_start = Instant::now();
    let (_, seqlen) = semantic_tokens.dims2()?;
    println!("Shape: {:?}, seqlen: {}", semantic_tokens.shape(), seqlen);
    let tokens = match state.lm.model_type {
        WhichLM::DualAR => semantic_tokens.i((.., ..seqlen - 1))?,
        _ => semantic_tokens.clone(),
    };
    // println!("Tokens shape: {:?},s eqlen: {}", tokens.shape(), seqlen);

    let out = state.codec.decode_batch(&tokens).await?;
    let duration = vocoder_start.elapsed();
    println!("Vocoding took: {} ms", duration.as_millis());

    let out = out.squeeze(0)?.squeeze(0)?;
    Ok(out)
}

async fn generate_pcm_chunk(
    state: Arc<AppState>,
    encoded_input: &Tensor,
    n_conditioning_tokens: usize,
) -> anyhow::Result<Tensor> {
    let (semantic_tokens, _) = server_lm_generate_blocking(
        state.clone(),
        encoded_input,
        &state.lm.default_sampling_args,
        n_conditioning_tokens,
        false,
    )
    .await
    .context("Failed to generate semantic tokens")?;
    vocode_semantic_tokens(state.clone(), &semantic_tokens).await
}

async fn generate_speech_blocking(
    state: Arc<AppState>,
    prompts: (usize, Vec<Tensor>),
    maybe_bsz: Option<usize>,
) -> Result<Response<Body>, AppError> {
    let mut all_pcm = Vec::new();
    let (n_conditioning_tokens, prompts) = prompts;

    match maybe_bsz {
        Some(batch_size) => {
            // Opt-in internal batching
            for (i, batch) in prompts.chunks(batch_size).enumerate() {
                println!("Processing batch {} ({}) prompts", i, batch.len());
                let pcm = generate_pcm_batched(state.clone(), batch)
                    .await?
                    .to_vec1::<f32>()?;
                all_pcm.extend(pcm);
            }
        }
        None => {
            // Single batch
            for (i, prompt) in prompts.iter().enumerate() {
                println!("Beginning chunk {} of {}", i, prompts.len());
                let pcm = generate_pcm_chunk(state.clone(), prompt, n_conditioning_tokens)
                    .await?
                    .to_vec1::<f32>()?;
                all_pcm.extend(pcm);
            }
        }
    }
    // Initial prefill
    println!("Generation complete");
    let mut model = state.lm.model.lock().await;
    // Final cache eviction
    model.clear_slow_layer_caches();
    println!("Final cache cleared");

    let mut audio_buf = Vec::new();
    write_pcm_as_wav(&mut audio_buf, &all_pcm, state.sample_rate)
        .context("Failed to write PCM as WAV")?;

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "audio/wav")
        .body(Body::from(audio_buf))
        .context("Failed to build WAV response")?)
}

async fn generate_speech_streaming(
    state: Arc<AppState>,
    prompts: (usize, Vec<Tensor>),
) -> Result<Response<Body>, AppError> {
    let src_rate: u32 = state.sample_rate;
    const DST_RATE: u32 = 24000;
    let (n_conditioning_tokens, prompts) = prompts;

    // Move all stream setup before the stream definition
    let encoder = Arc::new(Mutex::new(
        OpusEncoder::new().context("Failed to create Opus encoder")?,
    ));

    // Create a single clone of everything the stream will need
    let stream_state = state.clone();

    let stream = async_stream::stream! {
        for (i, prompt) in prompts.iter().enumerate() {
            println!("Generating chunk {} of {}", i, prompts.len());
            match generate_pcm_chunk(stream_state.clone(), prompt, n_conditioning_tokens).await {
                Ok(pcm_data) => {
                    if i == prompts.len() - 1 {
                        let mut model = stream_state.lm.model.lock().await;
                        model.clear_slow_layer_caches();
                    };
                    let resample_start = std::time::Instant::now();
                    let resampled_pcm: Tensor = resample(&pcm_data.unsqueeze(0).unwrap(), src_rate, DST_RATE)
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
                        .flatten_all()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
                    let resampled_pcm = resampled_pcm
                        .to_vec1::<f32>()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("PCM generation failed: {}", e)))?;
                    let duration = resample_start.elapsed();
                    println!("CPU resampling took: {:?}", duration);
                    let mut encoder = encoder.lock().await;
                    match encoder.encode_pcm(&resampled_pcm) {
                        Ok(encoded) => {
                            for chunk in encoded.chunks(1024) {
                                yield Ok(Bytes::copy_from_slice(chunk));
                            }
                        }
                        Err(e) => yield Err(std::io::Error::new(std::io::ErrorKind::Other, format!("PCM generation failed: {}", e)))
                    }
                }
                Err(e) => yield Err(std::io::Error::new(std::io::ErrorKind::Other, format!("PCM generation failed: {}", e)))
            }
        }
    };

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "audio/ogg")
        .header("Transfer-Encoding", "chunked")
        .body(Body::from_stream(stream))
        .context("Failed to build streaming response")?)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub model: String, // Ignored for now
    pub voice: String,
    pub input: String,
    /// Default: WAV
    pub response_format: Option<String>,
    pub batch_size: Option<usize>,
    pub speaker_prompt: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateResponse {
    audio: Vec<f32>,
}

pub async fn generate_speech(
    State(state): State<Arc<AppState>>,
    Json(request): Json<GenerateRequest>,
) -> Result<Response<Body>, AppError> {
    let voice_embedding = match &*request.voice {
        "unconditioned" => None,
        _ => Some(
            state
                .lm
                .voices
                .lock()
                .await
                .get(&request.voice)
                .unwrap_or(&state.lm.default_voice)
                .clone(),
        ),
    };

    let state = state.clone();
    let num_codebooks = state.lm.config.num_codebooks;
    let chunks = preprocess_text(&request.input);
    let prompt_encoder = PromptEncoder::new(
        &state.lm.tokenizer,
        &state.device,
        num_codebooks,
        state.lm.model_type,
    );
    let sysprompt_text = if request.speaker_prompt.is_some() {
        request.speaker_prompt
    } else {
        match state.model_type {
            WhichModel::Fish1_5 => Some("Speak out the provided text.".to_string()),
            _ => None,
        }
    };

    let prompts = prompt_encoder.encode_sequence(chunks, sysprompt_text, voice_embedding, true)?;

    if request.response_format == Some("opus".into()) {
        generate_speech_streaming(state, prompts).await
    } else {
        generate_speech_blocking(state, prompts, request.batch_size).await
    }
}
