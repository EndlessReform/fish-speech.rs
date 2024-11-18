use super::error::AppError;
use crate::opus::OpusEncoder;
use crate::state::AppState;
use anyhow::{Context, Result};
use axum::{body::Body, extract::State, http::StatusCode, response::Response, Json};
use bytes::Bytes;
use candle_core::{DType, Tensor, D};
use fish_speech_core::audio::wav::{resample_pcm, write_pcm_as_wav};
use fish_speech_core::models::text2semantic::utils::{
    encode::encode_chunks,
    generate::{generate, NeuralChunkedGenerator},
    sample::SamplingArgs,
    text::preprocess_text,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

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

// Blocking token generation
pub async fn generate_semantic_tokens(
    state: &Arc<AppState>,
    encoded_input: &Tensor,
    sampling_args: &SamplingArgs,
    collect_hidden_states: bool,
) -> Result<(Tensor, Option<Tensor>), AppError> {
    let mut model = state.semantic_model.lock().await;
    let (tokens, hidden_states) = generate(
        &mut model,
        &encoded_input,
        1024,
        state.tokenizer.token_to_id("<|im_end|>").unwrap_or(4),
        state.tokenizer.token_to_id("<|semantic|>").unwrap_or(5),
        sampling_args,
        collect_hidden_states,
    )
    .context("Failed to generate tokens")?;

    model.clear_slow_layer_caches();
    let tokens = tokens
        .broadcast_sub(&Tensor::ones_like(&tokens).context("Failed to create ones tensor")?)
        .context("Failed to broadcast subtract")?;

    Ok((tokens, hidden_states))
}

pub async fn vocode_semantic_tokens(
    state: &Arc<AppState>,
    semantic_tokens: &Tensor,
) -> Result<Vec<f32>, AppError> {
    let feature_lengths =
        Tensor::from_slice(&[semantic_tokens.dim(D::Minus1)? as u32], 1, &state.device)?;

    state
        .vocoder_model
        .decode(&semantic_tokens.unsqueeze(0)?, &feature_lengths)?
        .to_dtype(DType::F32)?
        .squeeze(0)?
        .squeeze(0)?
        .to_vec1()
        .map_err(|e| AppError(e.into()))
}

fn stream_audio_response(
    state: Arc<AppState>,
    prompts: Vec<Tensor>,
    encoder: Arc<Mutex<OpusEncoder>>,
) -> impl futures_util::Stream<Item = Result<Bytes, std::io::Error>> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    println!("SERVER START: {}ms", now.as_millis());
    const SRC_RATE: f32 = 44100.0;
    const DST_RATE: f32 = 24000.0;

    let sampling_args = SamplingArgs {
        temp: state.temp,
        top_p: state.top_p,
        top_k: 256,
        repetition_penalty: 1.2,
    };

    let start_time = std::time::Instant::now();

    async_stream::stream! {
        for (i, prompt) in prompts.into_iter().enumerate() {
            println!("Starting chunk {} at {:?}", i, start_time.elapsed());

            let mut model = state.semantic_model.lock().await;

            let gen_start = std::time::Instant::now();
            // Create chunked generator with optional probe
            let chunked_gen = NeuralChunkedGenerator::prefill(
                &mut model,
                state.silence_probe.as_deref(),
                prompt,
                4096,
                state.tokenizer.token_to_id("<|im_end|>").unwrap_or(4),
                state.tokenizer.token_to_id("<|semantic|>").unwrap_or(5),
                &sampling_args,
            ).map_err(AppError::from)?;
            println!("Generator prefill took {:?}", gen_start.elapsed());


            // Iterate over the neural chunks
            for chunk_result in chunked_gen {
                let chunk_start = std::time::Instant::now();
                let chunk = chunk_result.map_err(AppError::from)?;
                let tokens = chunk
                    .broadcast_sub(&Tensor::ones_like(&chunk).context("Failed to create ones tensor").map_err(AppError::from)?)
                    .context("Failed to broadcast subtract").map_err(AppError::from)?;
                // let vocode_start = std::time::Instant::now();

                let pcm_data = vocode_semantic_tokens(&state, &tokens).await.map_err(AppError::from)?;
                // println!("Vocoding took {:?}", vocode_start.elapsed());

                let resampled_pcm = resample_pcm(&pcm_data, SRC_RATE, DST_RATE);


                let mut encoder = encoder.lock().await;
                let encoded = encoder.encode_pcm(&resampled_pcm).map_err(AppError::from)?;


                // let now = std::time::SystemTime::now()
                //     .duration_since(std::time::UNIX_EPOCH)
                //     .unwrap();
                // println!("SERVER YIELD: {}ms", now.as_millis());
                for opus_chunk in encoded.chunks(1024) {
                    yield Ok(Bytes::copy_from_slice(opus_chunk));
                    // Force flush immediately
                    tokio::task::yield_now().await;
                }
                println!("Total neural chunk processing took {:?}", chunk_start.elapsed());
            }

            // Release the lock
            drop(model);
            println!("Completed text chunk {} after {:?}", i, start_time.elapsed());

        }
        println!("Total generation time: {:?}", start_time.elapsed());

    }
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
    )?;

    if request.response_format == Some("opus".into()) {
        let encoder = Arc::new(Mutex::new(
            OpusEncoder::new().context("Failed to create Opus encoder")?,
        ));

        let stream = stream_audio_response(state, prompts, encoder);

        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "audio/ogg")
            .header("Transfer-Encoding", "chunked")
            .body(Body::from_stream(stream))
            .context("Failed to build streaming response")?)
    } else {
        // Non-streaming path stays relatively simple
        let sampling_args = SamplingArgs {
            temp: state.temp,
            top_p: state.top_p,
            top_k: 256,
            repetition_penalty: 1.2,
        };

        let mut all_pcm = Vec::new();
        for prompt in prompts.iter() {
            let (semantic_tokens, _) =
                generate_semantic_tokens(&state, prompt, &sampling_args, false).await?;
            let pcm = vocode_semantic_tokens(&state, &semantic_tokens).await?;
            all_pcm.extend(pcm);
        }

        let mut audio_buf = Vec::new();
        write_pcm_as_wav(&mut audio_buf, &all_pcm, 44100)?;

        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "audio/wav")
            .body(Body::from(audio_buf))
            .context("Failed to build WAV response")?)
    }
}
