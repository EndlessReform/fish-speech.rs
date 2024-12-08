use super::error::AppError;
use crate::opus::OpusEncoder;
use crate::state::AppState;
use anyhow::{Context, Result};
use axum::{body::Body, extract::State, http::StatusCode, response::Response, Json};
use bytes::Bytes;
use candle_core::{DType, Tensor, D};
use fish_speech_core::audio::wav::write_pcm_as_wav;
use fish_speech_core::models::text2semantic::utils::{
    encode::encode_chunks, generate::generate, sample::SamplingArgs, text::preprocess_text,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

async fn generate_pcm_chunk(state: &Arc<AppState>, encoded_input: &Tensor) -> Result<Vec<f32>> {
    let sampling_args = SamplingArgs {
        temp: state.temp,
        top_p: state.top_p,
        top_k: 256,
        repetition_penalty: 1.2,
    };

    let semantic_tokens = {
        let mut model = state.semantic_model.lock().await;
        let tokens = generate(
            &mut model,
            &encoded_input,
            1024,
            state.tokenizer.token_to_id("<|im_end|>").unwrap_or(4),
            state.tokenizer.token_to_id("<|semantic|>").unwrap_or(5),
            &sampling_args,
        )
        .context("Failed to generate tokens")?;

        model.clear_slow_layer_caches();
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

    let audio = vocoder
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
        .context("Failed second squeeze")?
        .to_vec1()
        .context("Failed to convert to vec")?;

    Ok(audio)
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
        const SRC_RATE: f32 = 44100.0;
        const DST_RATE: f32 = 24000.0;

        // Move all stream setup before the stream definition
        let encoder = Arc::new(Mutex::new(
            OpusEncoder::new().context("Failed to create Opus encoder")?,
        ));

        // Create a single clone of everything the stream will need
        let stream_state = state.clone();

        let stream = async_stream::stream! {
            for prompt in prompts.iter() {
                match generate_pcm_chunk(&stream_state, prompt).await {
                    Ok(pcm_data) => {
                        let ratio = SRC_RATE / DST_RATE;
                        let resampled_pcm: Vec<f32> = (0..((pcm_data.len() as f32 / ratio) as usize))
                            .map(|i| {
                                let src_idx = i as f32 * ratio;
                                let src_idx_floor = src_idx.floor() as usize;
                                let src_idx_ceil = src_idx.ceil() as usize;
                                if src_idx_ceil >= pcm_data.len() {
                                    pcm_data[src_idx_floor]
                                } else {
                                    let t = src_idx - src_idx_floor as f32;
                                    pcm_data[src_idx_floor] * (1.0 - t) + pcm_data[src_idx_ceil] * t
                                }
                            })
                            .collect();

                        let mut encoder = encoder.lock().await;
                        match encoder.encode_pcm(&resampled_pcm) {
                            Ok(encoded) => {
                                for chunk in encoded.chunks(1024) {
                                    yield Ok(Bytes::copy_from_slice(chunk));
                                }
                            }
                            Err(e) => yield Err(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
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
    } else {
        let mut all_pcm = Vec::new();
        for prompt in prompts.iter() {
            let pcm = generate_pcm_chunk(&state, prompt).await?;
            all_pcm.extend(pcm);
        }

        let mut audio_buf = Vec::new();
        write_pcm_as_wav(&mut audio_buf, &all_pcm, 44100).context("Failed to write PCM as WAV")?;

        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "audio/wav")
            .body(Body::from(audio_buf))
            .context("Failed to build WAV response")?)
    }
}
