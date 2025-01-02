use super::error::AppError;
use super::speech::{server_lm_generate_blocking, vocode_semantic_tokens};
use crate::state::AppState;
use anyhow::Context;
use axum::body::Body;
use axum::{extract::State, http::StatusCode, response::Response, Json};
use candle_core::Tensor;
use fish_speech_core::models::text2semantic::utils::{
    encode::encode_chunks, sample::SamplingArgs, text::preprocess_text,
};
use serde::Deserialize;
use std::io::{Cursor, Write};
use std::sync::Arc;
use zip::{write::FileOptions, ZipWriter};

#[derive(Debug, Deserialize)]
pub struct GenerateHiddenStatesRequest {
    text: String,
    speaker_id: String,
    return_audio: bool,
}

pub async fn generate_hidden_states(
    State(state): State<Arc<AppState>>,
    Json(request): Json<GenerateHiddenStatesRequest>,
) -> Result<Response<Body>, AppError> {
    let voice_embedding = state
        .lm
        .voices
        .lock()
        .await
        .get(&request.speaker_id)
        .unwrap_or(&state.lm.default_voice)
        .clone();

    let state = state.clone();
    let num_codebooks = state.lm.model.lock().await.cfg.num_codebooks;
    let chunks = preprocess_text(&request.text);
    let prompts = encode_chunks(
        &state.lm.tokenizer,
        chunks,
        &state.device,
        Some(&voice_embedding),
        num_codebooks,
        state.lm.model_type,
    )?;

    let mut all_hidden_states = Vec::new();
    let mut all_pcm: Vec<f32> = Vec::new();

    // Non-streaming path stays relatively simple
    let sampling_args = SamplingArgs {
        temp: state.lm.default_temp,
        top_p: state.lm.default_top_p,
        top_k: 256,
        repetition_penalty: 1.2,
    };
    for prompt in prompts.chunks.iter() {
        let (semantic_tokens, maybe_hidden) = server_lm_generate_blocking(
            &state,
            prompt,
            &sampling_args,
            prompts.n_conditioning_tokens,
            true,
        )
        .await?;
        let pcm = vocode_semantic_tokens(&state, &semantic_tokens).await?;
        if let Some(hidden) = maybe_hidden {
            // println!("{:?} maybe?", hidden.squeeze(1)?.shape());
            all_hidden_states.push(hidden.squeeze(1)?);
        }
        if request.return_audio {
            all_pcm.extend(pcm.to_vec1::<f32>()?);
        }
    }

    let mut model = state.lm.model.lock().await;
    // Final cache eviction
    model.clear_slow_layer_caches();

    let buf = Cursor::new(Vec::new());
    let mut zip = ZipWriter::new(buf);

    let hidden_states =
        Tensor::cat(&all_hidden_states, 0).context("Failed to concatenate hidden states")?;

    let temp_path = std::env::temp_dir().join("temp_hidden_states.npy");
    hidden_states
        .to_dtype(candle_core::DType::F32)?
        .write_npy(&temp_path)?;
    let npy_contents = std::fs::read(&temp_path)?;
    std::fs::remove_file(temp_path)?;

    zip.start_file::<_, ()>(
        "hidden_states.npy",
        FileOptions::default().compression_method(zip::CompressionMethod::Stored),
    )?;
    zip.write_all(&npy_contents)?;

    if request.return_audio {
        zip.start_file::<_, ()>(
            "audio.wav",
            FileOptions::default().compression_method(zip::CompressionMethod::Stored),
        )?;
        fish_speech_core::audio::wav::write_pcm_as_wav(&mut zip, &all_pcm, 44100)?;
    }

    zip.start_file::<_, ()>(
        "metadata.json",
        FileOptions::default().compression_method(zip::CompressionMethod::Stored),
    )?;
    serde_json::to_writer(
        &mut zip,
        &serde_json::json!({
            "frame_count": hidden_states.dim(0)?,
            "frame_rate": 21.535,
            "hidden_dim": hidden_states.dim(1)?
        }),
    )?;

    let buf = zip.finish()?.into_inner();

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/zip")
        .body(Body::from(buf))
        .context("Failed to build ZIP response")?)
}
