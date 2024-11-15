use super::opus::OpusEncoder;
use super::state::AppState;
use anyhow::{anyhow, Context, Result};
use axum::{body::Body, extract::State, http::StatusCode, response::Response, Json};
use bytes::Bytes;
use candle_core::{DType, Tensor, D};
use fish_speech_core::audio::wav::write_pcm_as_wav;
use fish_speech_core::models::text2semantic::utils::{
    encode::encode_tokens_batch,
    generate::TokenGenerator,
    sample::SamplingArgs,
    text::{preprocess_text, TextChunk},
};
use futures_util::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;

// Custom error wrapper that holds anyhow::Error
pub struct AppError(pub anyhow::Error);

// Convert anyhow::Error into AppError
impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

// Implement IntoResponse for AppError to convert errors into HTTP responses
impl axum::response::IntoResponse for AppError {
    fn into_response(self) -> Response {
        // Log the error with its full chain of causes
        tracing::error!("Application error: {:#}", self.0);

        // You can match on specific error types and return different status codes
        let status = if self.0.downcast_ref::<std::io::Error>().is_some() {
            StatusCode::INTERNAL_SERVER_ERROR
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };

        // Return the error response
        (status, format!("Something went wrong: {}", self.0)).into_response()
    }
}

async fn generate_pcm_chunk(
    state: &Arc<AppState>,
    chunk: &TextChunk,
    voice_embedding: &Tensor,
    num_codebooks: usize,
) -> Result<Vec<f32>> {
    let encoded_input = encode_tokens_batch(
        &state.tokenizer,
        vec![chunk.clone()],
        &state.device,
        None,
        num_codebooks,
    )
    .context("Failed to encode tokens batch")?
    .pop()
    .ok_or_else(|| anyhow!("No encoded input generated"))?;

    let final_prompt = Tensor::cat(&[voice_embedding, &encoded_input], D::Minus1)
        .context("Failed to concatenate tensors")?;

    let sampling_args = SamplingArgs {
        temp: state.temp,
        top_p: state.top_p,
        top_k: 256,
        repetition_penalty: 1.2,
    };

    let semantic_tokens = {
        let mut model = state.semantic_model.lock().await;
        // TODO: Actually take advantage of the iterator! for chunking
        let (generator, first_token) = TokenGenerator::prefill(
            &mut model,
            &final_prompt,
            1024,
            state.tokenizer.token_to_id("<|im_end|>").unwrap_or(4),
            state.tokenizer.token_to_id("<|semantic|>").unwrap_or(5),
            &sampling_args,
        )?;
        println!("Prefill complete");

        let subsequent_tokens: candle_core::Result<Vec<Tensor>> = generator.collect();
        let mut all_tokens = vec![first_token];
        all_tokens.extend(subsequent_tokens?);
        println!("Tokens generated, length: {}", all_tokens.len());
        let tokens = Tensor::cat(&all_tokens, D::Minus1)?;
        model.clear_slow_layer_caches();

        tokens
            .broadcast_sub(&Tensor::ones_like(&tokens).context("Failed to create ones tensor")?)
            .context("Failed to broadcast subtract")?
    };

    let vocoder = state.vocoder_model.lock().await;
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

async fn generate_speech_wav(
    state: Arc<AppState>,
    chunks: Vec<TextChunk>,
    voice_embedding: &Tensor,
    num_codebooks: usize,
) -> Result<Vec<u8>, AppError> {
    let mut all_pcm = Vec::new();
    for chunk in chunks.iter() {
        let pcm = generate_pcm_chunk(&state, chunk, &voice_embedding, num_codebooks).await?;
        all_pcm.extend(pcm);
    }

    let mut audio_buf = Vec::new();
    write_pcm_as_wav(&mut audio_buf, &all_pcm, 44100).context("Failed to write PCM as WAV")?;
    Ok(audio_buf)
}

fn generate_speech_opus(
    state: Arc<AppState>,
    voice_embedding: &Tensor,
    chunks: Vec<TextChunk>,
    num_codebooks: usize,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> {
    // BOXED AND PINNED
    const SRC_RATE: f32 = 44100.0;
    const DST_RATE: f32 = 24000.0;

    let encoder = match OpusEncoder::new() {
        Ok(enc) => Arc::new(Mutex::new(enc)),
        Err(e) => {
            return Box::pin(futures_util::stream::once(async move {
                Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                ))
            }))
        }
    };

    let stream_state = state;
    let stream_embedding = voice_embedding.clone();

    Box::pin(async_stream::stream! {  // BOX PIN THIS SHIT
        for chunk in chunks.iter() {
            match generate_pcm_chunk(&stream_state, chunk, &stream_embedding, num_codebooks).await {
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
    })
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

    if request.response_format == Some("opus".into()) {
        let stream = generate_speech_opus(state, &voice_embedding, chunks, num_codebooks);
        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "audio/ogg")
            .header("Transfer-Encoding", "chunked")
            .body(Body::from_stream(stream))
            .context("Failed to build streaming response")?)
    } else {
        let audio_buf = generate_speech_wav(state, chunks, &voice_embedding, num_codebooks).await?;
        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "audio/wav")
            .body(Body::from(audio_buf))
            .context("Failed to build WAV response")?)
    }
}