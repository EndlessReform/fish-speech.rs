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
    encode::encode_chunks, generate::generate, sample::SamplingArgs, text::preprocess_text,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

async fn generate_pcm_chunk(
    state: &Arc<AppState>,
    encoded_input: &Tensor,
    n_conditioning_tokens: usize,
) -> Result<Tensor> {
    let sampling_args = SamplingArgs {
        temp: state.temp,
        top_p: state.top_p,
        top_k: 256,
        repetition_penalty: 1.2,
    };

    let semantic_tokens = {
        let mut model = state.semantic_model.lock().await;
        let maybe_tokens = generate(
            &mut model,
            &encoded_input,
            1024,
            state.tokenizer.token_to_id("<|im_end|>").unwrap_or(4),
            state.tokenizer.token_to_id("<|semantic|>").unwrap_or(5),
            &sampling_args,
        );

        let tokens = maybe_tokens.map_err(|e| {
            println!("Error in semantic generation: {:?}", e);
            anyhow::Error::from(e)
        })?;

        // It's the caller's responsibility to do final clear
        model.clear_slow_caches_until(n_conditioning_tokens)?;

        match state.model_type {
            fish_speech_core::models::vqgan::config::WhichModel::Fish1_5 => tokens,
            _ => tokens
                .broadcast_sub(&Tensor::ones_like(&tokens).context("Failed to create ones tensor")?)
                .context("Failed to broadcast subtract")?,
        }
    };

    let vocoder = state.vocoder_model.clone();
    let feature_lengths = Tensor::from_slice(
        &[semantic_tokens
            .dim(D::Minus1)
            .context("Failed to get semantic tokens dimension")? as u32],
        1,
        &state.device,
    )
    .context("Failed to create feature lengths tensor")?;

    vocoder
        .decode(
            &semantic_tokens
                .unsqueeze(0)
                .context("Failed to unsqueeze semantic tokens")?,
            &feature_lengths,
        )
        .context("Failed to decode audio")?
        .to_dtype(DType::F32)
        .context("Failed to convert to F32")?
        .squeeze(0)
        .context("Failed first squeeze")?
        .squeeze(0)
        .context("Failed second squeeze")
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
    let mut model = state.semantic_model.lock().await;
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
            match generate_pcm_chunk(&stream_state, prompt, prompts.n_conditioning_tokens).await {
                Ok(pcm_data) => {
                    if i == prompts.chunks.len() - 1 {
                        let mut model = stream_state.semantic_model.lock().await;
                        model.clear_slow_layer_caches();
                    };
                    let resampled_pcm: Vec<f32> = resample(&pcm_data, SRC_RATE, DST_RATE)
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
                        .to_vec1::<f32>()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("PCM generation failed: {}", e)))?;
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
        .voices
        .get(&request.voice)
        .unwrap_or(&state.default_voice)
        .clone();

    let state = state.clone();
    let num_codebooks = state.semantic_model.lock().await.cfg.num_codebooks;
    let chunks = preprocess_text(&request.input);
    let prompts = encode_chunks(
        &state.tokenizer,
        chunks,
        &state.device,
        Some(&voice_embedding),
        num_codebooks,
        state.model_type,
    )?;

    if request.response_format == Some("opus".into()) {
        generate_speech_streaming(state, prompts).await
    } else {
        generate_speech_blocking(state, prompts).await
    }
}
