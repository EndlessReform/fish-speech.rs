use super::error::AppError;
use super::speech::generate_pcm_chunk;
use crate::state::AppState;
use anyhow::Context;
use axum::body::Body;
use axum::{extract::State, http::StatusCode, response::Response, Json};
use candle_core::Tensor;
use fish_speech_core::models::text2semantic::utils::{
    encode::encode_chunks, text::preprocess_text,
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
        .voices
        .get(&request.speaker_id)
        .unwrap_or(&state.default_voice)
        .clone();

    let state = state.clone();
    let num_codebooks = state.semantic_model.lock().await.cfg.num_codebooks;
    let chunks = preprocess_text(&request.text);
    let prompts = encode_chunks(
        &state.tokenizer,
        chunks,
        &state.device,
        Some(&voice_embedding),
        num_codebooks,
    )?;

    let mut all_hidden_states = Vec::new();
    let mut all_pcm = Vec::new();

    for prompt in prompts.iter() {
        let (pcm, maybe_hidden) = generate_pcm_chunk(&state, prompt, true).await?;
        if let Some(hidden) = maybe_hidden {
            // println!("{:?} maybe?", hidden.squeeze(1)?.shape());
            all_hidden_states.push(hidden.squeeze(1)?);
        }
        if request.return_audio {
            all_pcm.extend(pcm);
        }
    }

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
