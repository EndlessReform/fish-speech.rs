use super::error::AppError;
use crate::opus::OpusEncoder;
use crate::state::AppState;
use anyhow::{Context, Result};
use axum::{body::Body, extract::State, http::StatusCode, response::Response, Json};
use bytes::Bytes;
use candle_core::{DType, Tensor, D};
use fish_speech_core::audio::{functional::resample, wav::write_pcm_as_wav};
use fish_speech_core::models::text2semantic::utils::encode::EncodedChunks;
use fish_speech_core::models::text2semantic::utils::{
    encode::encode_chunks, generate::generate_blocking_with_hidden, sample::SamplingArgs,
    text::preprocess_text,
};
use fish_speech_core::models::vqgan::config::{WhichFishVersion, WhichLM};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

// Blocking token generation
pub async fn server_lm_generate_blocking(
    state: &Arc<AppState>,
    encoded_input: &Tensor,
    sampling_args: &SamplingArgs,
    n_conditioning_tokens: usize,
    collect_hidden_states: bool,
) -> Result<(Tensor, Option<Tensor>), anyhow::Error> {
    // Arbitrary number
    let max_tokens: usize = 768;

    let mut model = state.lm.model.lock().await;
    let (tokens, hidden_states) = generate_blocking_with_hidden(
        &mut model,
        &encoded_input,
        max_tokens,
        sampling_args,
        collect_hidden_states,
    )
    .context("Failed to generate tokens")?;

    let mut tokens = tokens;
    let mut hidden_states = hidden_states;
    // It's the caller's responsibility to do final clear
    model.clear_slow_caches_until(n_conditioning_tokens)?;
    if tokens.dim(D::Minus1)? == max_tokens {
        println!("Failed generation suspected. Rerolling once");
        let (new_tokens, new_hidden_states) = generate_blocking_with_hidden(
            &mut model,
            &encoded_input,
            1024,
            sampling_args,
            collect_hidden_states,
        )
        .context("Failed to generate tokens")?;
        if new_tokens.dim(D::Minus1)? != max_tokens {
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

pub async fn vocode_semantic_tokens(
    state: &Arc<AppState>,
    semantic_tokens: &Tensor,
) -> anyhow::Result<Tensor> {
    let vocoder_start = Instant::now();
    let feature_lengths =
        Tensor::from_slice(&[semantic_tokens.dim(D::Minus1)? as u32], 1, &state.device)?;

    let out = state
        .vocoder_model
        .decode(&semantic_tokens.unsqueeze(0)?, &feature_lengths)?
        .to_dtype(DType::F32)?;

    let duration = vocoder_start.elapsed();
    println!("Vocoding took: {} ms", duration.as_millis());

    let out = out.squeeze(0)?.squeeze(0)?;
    Ok(out)
}

async fn generate_pcm_chunk(
    state: &Arc<AppState>,
    encoded_input: &Tensor,
    n_conditioning_tokens: usize,
) -> anyhow::Result<Tensor> {
    let sampling_args = SamplingArgs {
        temp: state.lm.temp,
        top_p: state.lm.top_p,
        top_k: 256,
        repetition_penalty: 1.3,
    };

    let (semantic_tokens, _) = server_lm_generate_blocking(
        state,
        encoded_input,
        &sampling_args,
        n_conditioning_tokens,
        false,
    )
    .await
    .context("Failed to generate semantic tokens")?;
    vocode_semantic_tokens(state, &semantic_tokens).await
}

async fn generate_speech_blocking(
    state: Arc<AppState>,
    prompts: EncodedChunks,
) -> Result<Response<Body>, AppError> {
    let mut all_pcm = Vec::new();

    // Initial prefill
    for (i, prompt) in prompts.chunks.iter().enumerate() {
        println!("Beginning chunk {} of {}", i, prompts.chunks.len());
        let pcm = generate_pcm_chunk(&state, prompt, prompts.n_conditioning_tokens)
            .await?
            .to_vec1::<f32>()?;
        all_pcm.extend(pcm);
    }
    println!("Generation complete");
    let mut model = state.lm.model.lock().await;
    // Final cache eviction
    model.clear_slow_layer_caches();
    println!("Final cache cleared");

    let mut audio_buf = Vec::new();
    write_pcm_as_wav(&mut audio_buf, &all_pcm, 44100).context("Failed to write PCM as WAV")?;

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "audio/wav")
        .body(Body::from(audio_buf))
        .context("Failed to build WAV response")?)
}

async fn generate_speech_streaming(
    state: Arc<AppState>,
    prompts: EncodedChunks,
) -> Result<Response<Body>, AppError> {
    const SRC_RATE: u32 = 44100;
    const DST_RATE: u32 = 24000;

    // Move all stream setup before the stream definition
    let encoder = Arc::new(Mutex::new(
        OpusEncoder::new().context("Failed to create Opus encoder")?,
    ));

    // Create a single clone of everything the stream will need
    let stream_state = state.clone();

    let stream = async_stream::stream! {
        for (i, prompt) in prompts.chunks.iter().enumerate() {
            println!("Generating chunk {} of {}", i, prompts.chunks.len());
            match generate_pcm_chunk(&stream_state, prompt, prompts.n_conditioning_tokens).await {
                Ok(pcm_data) => {
                    if i == prompts.chunks.len() - 1 {
                        let mut model = stream_state.lm.model.lock().await;
                        model.clear_slow_layer_caches();
                    };
                    let resample_start = std::time::Instant::now();
                    let resampled_pcm: Tensor = resample(&pcm_data.unsqueeze(0).unwrap(), SRC_RATE, DST_RATE)
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

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    model: String, // Ignored for now
    voice: String,
    input: String,
    response_format: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    audio: Vec<f32>,
}
pub async fn generate_speech(
    State(state): State<Arc<AppState>>,
    Json(request): Json<GenerateRequest>,
) -> Result<Response<Body>, AppError> {
    let voice_embedding = state
        .lm
        .voices
        .lock()
        .await
        .get(&request.voice)
        .unwrap_or(&state.lm.default_voice)
        .clone();

    let state = state.clone();
    let num_codebooks = state.lm.model.lock().await.cfg.num_codebooks;
    let chunks = preprocess_text(&request.input);
    let prompts = encode_chunks(
        &state.lm.tokenizer,
        chunks,
        &state.device,
        Some(&voice_embedding),
        num_codebooks,
        state.lm.model_type,
    )?;

    if request.response_format == Some("opus".into()) {
        generate_speech_streaming(state, prompts).await
    } else {
        generate_speech_blocking(state, prompts).await
    }
}
