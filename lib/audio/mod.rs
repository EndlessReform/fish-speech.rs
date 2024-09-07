pub mod functional;
pub mod pcm_decode;
pub mod spectrogram;

use anyhow::{bail, Result};
use byteorder::{ByteOrder, LittleEndian};
use candle_core::{Device, Tensor};
use candle_transformers::models::whisper::{audio::pcm_to_mel, Config as WhisperConfig};
use std::path::Path;

pub struct AudioConfig {
    pub num_mel_bins: usize,
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
}

impl Default for AudioConfig {
    fn default() -> Self {
        AudioConfig {
            // Number of mel bins from the fish 1.2 config
            num_mel_bins: 160,
            // Sample rate in Hz
            sample_rate: 44100,
            // Size of the FFT window
            n_fft: 2048,
            // Number of samples between successive frames
            hop_length: 512,
        }
    }
}

/// Replicates `torchaudio.load` with the following limitations:
/// - Only takes the first channel of audio instead of mean or returning all
pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<(Tensor, u32)> {
    // Load audio file and convert to PCM
    let (pcm_data, sample_rate, num_frames, num_channels) = pcm_decode::pcm_decode(path)?;
    let tensor = Tensor::from_slice(&pcm_data, (num_channels, num_frames), device)?;

    Ok((tensor, sample_rate))
}

pub fn load_audio<P: AsRef<Path>>(
    path: P,
    config: &AudioConfig,
    device: &Device,
) -> Result<Tensor> {
    // Load audio file and convert to PCM
    let (pcm_data, original_sample_rate, _, _) = pcm_decode::pcm_decode(path)?;

    // Handle stereo if necessary
    let mono_pcm = if pcm_data.len() % 2 == 0 {
        handle_stereo_pcm(&pcm_data)
    } else {
        pcm_data
    };

    // Resample if necessary
    let resampled_pcm = if original_sample_rate != config.sample_rate {
        resample(&mono_pcm, original_sample_rate, config.sample_rate)
    } else {
        mono_pcm
    };

    // Load mel filters
    let mel_filters = load_mel_filters(config.num_mel_bins)?;

    // Yes, this is terrible,
    // but `pcm_to_mel`'s API only needs the mel bins but forces me to pass an entire config
    let dummy_config = WhisperConfig {
        num_mel_bins: config.num_mel_bins,
        max_source_positions: 0,
        d_model: 0,
        encoder_layers: 0,
        decoder_layers: 0,
        encoder_attention_heads: 0,
        vocab_size: 0,
        max_target_positions: 0,
        decoder_attention_heads: 0,
        suppress_tokens: Vec::new(),
    };

    // Convert PCM to MEL spectrogram
    let mel = pcm_to_mel(&dummy_config, &resampled_pcm, &mel_filters);
    let mel_len = mel.len();

    // Convert to Tensor
    let mel_tensor = Tensor::from_vec(
        mel,
        (1, config.num_mel_bins, mel_len / config.num_mel_bins),
        device,
    )?;

    Ok(mel_tensor)
}

fn handle_stereo_pcm(pcm_data: &[f32]) -> Vec<f32> {
    let samples = pcm_data.len() / 2;
    let mut mono = Vec::with_capacity(samples);

    for i in 0..samples {
        let left = pcm_data[2 * i];
        let right = pcm_data[2 * i + 1];
        mono.push((left + right) / 2.0);
    }

    mono
}

pub fn resample(pcm_data: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    // Use f64 for ratio and length calculations for better precision with large numbers
    let resample_ratio = to_rate as f64 / from_rate as f64;
    let output_len_f64 = (pcm_data.len() as f64 * resample_ratio).ceil();
    let output_len = output_len_f64 as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let input_index = i as f64 / resample_ratio;
        let input_index_floor = input_index.floor() as usize;
        let input_index_ceil = (input_index.ceil() as usize).min(pcm_data.len() - 1);

        if input_index_floor >= pcm_data.len() - 1 {
            output.push(pcm_data[pcm_data.len() - 1]);
        } else {
            // Convert t to f32 for consistency with pcm_data, which is f32
            let t = (input_index - input_index_floor as f64) as f32;
            let interpolated =
                pcm_data[input_index_floor] * (1.0 - t) + pcm_data[input_index_ceil] * t;
            output.push(interpolated);
        }
    }

    output
}

/// Load or generate mel filters
fn load_mel_filters(num_mel_bins: usize) -> Result<Vec<f32>> {
    // Load mel filters from file or generate them
    // This should return the mel filters as a vector
    let mel_bytes = match num_mel_bins {
        80 => include_bytes!("melfilters.bytes").as_slice(),
        128 => include_bytes!("melfilters128.bytes").as_slice(),
        160 => include_bytes!("melfilters160.bytes").as_slice(),
        _ => bail!("Unexpected num_mel_bins {}", num_mel_bins),
    };

    // Convert bytes to f32 values
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    LittleEndian::read_f32_into(mel_bytes, &mut mel_filters);

    Ok(mel_filters)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::path::PathBuf;

    fn get_test_audio_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("resources")
            .join("ja-1.wav") // Using the file we know exists
    }

    #[test]
    fn test_load_audio() {
        let config = AudioConfig {
            sample_rate: 16000,
            num_mel_bins: 80,
            n_fft: 400,
            hop_length: 160,
        };
        let device = Device::Cpu;

        let result = load_audio(get_test_audio_path(), &config, &device);
        assert!(result.is_ok(), "Failed to load audio");

        if let Ok(tensor) = result {
            let dims = tensor.shape().dims();
            assert_eq!(dims[0], 1, "Batch size should be 1");
            assert_eq!(
                dims[1], config.num_mel_bins,
                "Number of mel bins should match the config"
            );
            assert!(dims[2] > 0, "Number of time steps should be greater than 0");
        }
    }

    #[test]
    fn test_handle_stereo_pcm() {
        let stereo_pcm = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mono_pcm = handle_stereo_pcm(&stereo_pcm);

        assert_eq!(mono_pcm, vec![1.5, 3.5, 5.5]);
    }

    #[test]
    fn test_resample() {
        let original_pcm = vec![1.0, 2.0, 3.0, 4.0];

        // Upsample
        let upsampled = resample(&original_pcm, 8000, 16000);
        assert_eq!(upsampled.len(), 8);

        // Downsample
        let downsampled = resample(&original_pcm, 16000, 8000);
        assert_eq!(downsampled.len(), 2);
    }

    #[test]
    fn test_load_mel_filters() {
        let result_80 = load_mel_filters(80);
        assert!(result_80.is_ok());
        assert_eq!(result_80.unwrap().len(), 80 * 201);

        let result_128 = load_mel_filters(128);
        assert!(result_128.is_ok());
        assert_eq!(result_128.unwrap().len(), 128 * 201);

        let result_invalid = load_mel_filters(64);
        assert!(result_invalid.is_err());
    }
}
