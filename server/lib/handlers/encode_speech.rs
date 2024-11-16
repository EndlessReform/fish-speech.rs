use super::error::AppError;
use crate::state::AppState;
use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::Response,
};
use candle_core::Tensor;
use fish_speech_core::audio as torchaudio;
use fish_speech_core::audio::functional;
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

    let audio = functional::resample(
        &audio,
        sr,
        state.firefly_config.spec_transform.sample_rate as u32,
    )?
    .unsqueeze(0)?;

    let mels = state.spec_transform.forward(&audio)?;

    let start_encode = Instant::now();
    let result = state.encoder_model.encode(&mels)?.squeeze(0)?;
    let encode_time = start_encode.elapsed().as_secs_f32();

    let npy_bytes = tensor_to_npy_bytes(&result)?;

    let audio_duration =
        audio.dim(2)? as f32 / state.firefly_config.spec_transform.sample_rate as f32;
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
