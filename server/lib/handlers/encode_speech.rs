use super::error::AppError;
use crate::state::AppState;
use anyhow::anyhow;
use axum::{
    extract::{Multipart, Query, State},
    http::StatusCode,
    response::Response,
};
use candle_core::{Tensor, D};
use fish_speech_core::audio as torchaudio;
use fish_speech_core::audio::functional;
use fish_speech_core::text::prompt::PromptEncoder;
use std::sync::Arc;
use std::time::Instant;

use std::io::{Read, Seek};
use tempfile::NamedTempFile;

pub fn tensor_to_npy_bytes(tensor: &Tensor) -> anyhow::Result<Vec<u8>> {
    // This gives us both a Path for write_npy AND Seek for rewind
    let mut temp = NamedTempFile::new()?;

    // Now we can use write_npy with the path
    tensor.write_npy(temp.path())?;

    // And we can seek because NamedTempFile implements Seek
    temp.rewind()?;

    // Read it back
    let mut bytes = Vec::new();
    temp.read_to_end(&mut bytes)?;

    Ok(bytes)
}

pub async fn encode_speaker(
    State(state): State<Arc<AppState>>,
    Query(params): Query<std::collections::HashMap<String, String>>,
    mut multipart: Multipart,
) -> Result<Response, AppError> {
    let start_total = Instant::now();

    let field = multipart
        .next_field()
        .await?
        .ok_or_else(|| anyhow::anyhow!("No file provided"))?;

    let data = field.bytes().await?.to_vec();

    let (mut audio, sr) = torchaudio::load_from_memory(data, &state.device)?;
    if audio.dim(0)? > 1 {
        audio = audio.mean_keepdim(0)?;
    }
    let audio = functional::resample(&audio, sr, state.sample_rate as u32)?;
    // TODO handle batched audio
    let result = state
        .codec
        .encode_batch(&audio.unsqueeze(0)?)
        .await?
        .squeeze(0)?;

    let start_encode = Instant::now();
    let encode_time = start_encode.elapsed().as_secs_f32();
    if let (Some(id), Some(prompt)) = (params.get("id"), params.get("prompt")) {
        println!("Adding id: {}", id);
        let mut speaker_map = state.lm.voices.lock().await;
        let prompt_encoder = PromptEncoder::new(
            &state.lm.tokenizer,
            &state.device,
            state.lm.config.num_codebooks,
            state.lm.model_type,
        );
        if speaker_map.contains_key(id) {
            return Err(AppError(anyhow!("ID already exists on server: {}", id)));
        }
        let new_prompt = prompt_encoder
            .encode_conditioning_prompt(prompt, &result.to_dtype(candle_core::DType::U32)?)?;
        speaker_map.insert(id.to_owned(), new_prompt);
    }

    let npy_bytes = tensor_to_npy_bytes(&result)?;

    let audio_duration = audio.dim(D::Minus1)? as f32 / state.sample_rate as f32;
    println!("Encoding RTF: {:.1}x", audio_duration / encode_time);
    println!(
        "Total RTF: {:.1}x",
        audio_duration / start_total.elapsed().as_secs_f32()
    );

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/x-npy")
        .body(npy_bytes.into())?)
}
