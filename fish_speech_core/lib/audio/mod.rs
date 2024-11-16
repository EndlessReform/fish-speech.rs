pub mod functional;
pub mod pcm_decode;
pub mod spectrogram;
mod stft;
pub mod wav;

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::path::Path;

/// Replicates `torchaudio.load` with the following limitations:
/// - Only takes the first channel of audio instead of mean or returning all
pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<(Tensor, u32)> {
    // Load audio file and convert to PCM
    let (pcm_data, sample_rate, num_frames, num_channels) = pcm_decode::decode_audio_file(path)?;
    let tensor = Tensor::from_slice(&pcm_data, (num_channels, num_frames), device)?;

    Ok((tensor, sample_rate))
}

/// Replicates `torchaudio.load` with the following limitations:
/// - Only takes the first channel of audio instead of mean or returning all
pub fn load_from_memory(bytes: Vec<u8>, device: &Device) -> Result<(Tensor, u32)> {
    // Load audio file and convert to PCM
    let (pcm_data, sample_rate, num_frames, num_channels) = pcm_decode::decode_audio_bytes(bytes)?;
    let tensor = Tensor::from_slice(&pcm_data, (num_channels, num_frames), device)?;

    Ok((tensor, sample_rate))
}
