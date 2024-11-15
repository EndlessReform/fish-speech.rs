use super::error::AppError;
use super::generate::process_token_stream;
use super::opus::{resample_pcm, OpusEncoder};
use super::state::AppState;
use anyhow::{Context, Result};
use axum::{body::Body, extract::State, http::StatusCode, response::Response, Json};
use bytes::Bytes;
use candle_core::{DType, Tensor, D};
use fish_speech_core::audio::wav::write_pcm_as_wav;
use fish_speech_core::models::text2semantic::utils::{
    encode::encode_tokens_batch, generate::TokenGenerator, sample::SamplingArgs,
    text::preprocess_text,
};
use futures_util::pin_mut;
use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;

const MAX_CONTEXT: usize = 1024;

async fn generate_pcm_chunk(
    state: &Arc<AppState>,
    prompt: &Tensor,
    voice_embedding: &Tensor,
    sampling_args: &SamplingArgs,
) -> Result<Vec<f32>> {
    let final_prompt = Tensor::cat(&[voice_embedding, &prompt], D::Minus1)
        .context("Failed to concatenate tensors")?;

    let semantic_tokens = {
        let mut model = state.semantic_model.lock().await;
        let im_end_id = state.tokenizer.token_to_id("<|im_end|>").unwrap_or(4);
        let pad_id = state.tokenizer.token_to_id("<|semantic|>").unwrap_or(5);
        // TODO: Actually take advantage of the iterator! for chunking
        let (generator, first_token) = TokenGenerator::prefill(
            &mut model,
            &final_prompt,
            MAX_CONTEXT,
            im_end_id,
            pad_id,
            sampling_args,
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
    println!(
        "Final window : {:?}, feature lengths: {:?}",
        semantic_tokens.to_vec2::<u32>().unwrap(),
        feature_lengths
    );

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

async fn generate_speech_blocking(
    state: Arc<AppState>,
    prompts: Vec<Tensor>,
    voice_embedding: &Tensor,
    sampling_args: &SamplingArgs,
) -> Result<Vec<u8>, AppError> {
    let mut all_pcm = Vec::new();
    for prompt in prompts.iter() {
        let pcm = generate_pcm_chunk(&state, prompt, &voice_embedding, sampling_args).await?;
        all_pcm.extend(pcm);
    }

    let mut audio_buf = Vec::new();
    write_pcm_as_wav(&mut audio_buf, &all_pcm, 44100).context("Failed to write PCM as WAV")?;
    Ok(audio_buf)
}

/// BOXED AND PINNED
fn generate_speech_streaming(
    state: Arc<AppState>,
    prompts: Vec<Tensor>,
    voice_embedding: &Tensor,
    sampling_args: SamplingArgs,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> {
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

    let stream_state = state.clone();
    let stream_embedding = voice_embedding.clone();
    let im_end_id = stream_state
        .tokenizer
        .token_to_id("<|im_end|>")
        .unwrap_or(4);
    let pad_id = stream_state
        .tokenizer
        .token_to_id("<|semantic|>")
        .unwrap_or(5);

    Box::pin(async_stream::stream! {
        for prompt in prompts.into_iter() {
            // TODO: fix this
            let final_prompt = Tensor::cat(&[stream_embedding.clone(), prompt.clone()], D::Minus1)
                .context("Failed to concatenate tensors").unwrap();

            let mut model = stream_state.semantic_model.lock().await;
            let (generator, first_token) = TokenGenerator::prefill(
                &mut model,
                &final_prompt,
                MAX_CONTEXT,
                im_end_id,
                pad_id,
                &sampling_args,
            ).map_err(|e| AppError(e.into()))?;
            println!("Prefill complete");

            // Set up windowed token stream
            let token_stream = process_token_stream(
                std::iter::once(Ok(first_token)).chain(generator),
                stream_state.device.clone(),
                stream_state.vocoder_model.clone()
            );
            pin_mut!(token_stream);
            while let Some(audio_result) = token_stream.next().await {
                match audio_result {
                    Err(e) => yield Err(AppError(e.into()).into()),
                    Ok(pcm_data) => {
                        println!("Got here");
                        let ratio = SRC_RATE / DST_RATE;
                        let resampled_pcm = resample_pcm(&pcm_data, ratio);
                        let mut encoder = encoder.lock().await;
                        match encoder.encode_pcm(&resampled_pcm) {
                            Err(e) => {
                                println!("Failing chunk");
                                yield Err(AppError(e.into()).into());
                            },
                            Ok(encoded) => {
                                println!("Sending OGG chunk");
                                for chunk in encoded.chunks(1024) {
                                    yield Ok(Bytes::copy_from_slice(chunk));
                                }
                            }
                        }
                    }
                }
            }
            println!("Done with text chunk");
        }
        println!("Done with core loop. Deadlock?");
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
    let encoded_input =
        encode_tokens_batch(&state.tokenizer, chunks, &state.device, None, num_codebooks)?;
    let sampling_args = SamplingArgs {
        temp: state.temp,
        top_p: state.top_p,
        top_k: 256,
        repetition_penalty: 1.2,
    };

    if request.response_format == Some("opus".into()) {
        let stream = generate_speech_streaming(
            state,
            encoded_input,
            &voice_embedding,
            sampling_args.clone(),
        );
        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "audio/ogg")
            .header("Transfer-Encoding", "chunked")
            .body(Body::from_stream(stream))
            .context("Failed to build streaming response")?)
    } else {
        let audio_buf =
            generate_speech_blocking(state, encoded_input, &voice_embedding, &sampling_args)
                .await?;
        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "audio/wav")
            .body(Body::from(audio_buf))
            .context("Failed to build WAV response")?)
    }
}
