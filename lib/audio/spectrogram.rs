use anyhow::{bail, Result};
use byteorder::{ByteOrder, LittleEndian};
use candle_core::{Result as CandleResult, Tensor};
use candle_transformers::models::whisper::audio::log_mel_spectrogram_;

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

#[derive(Clone, Debug)]
pub struct LogMelSpectrogramConfig {
    pub sample_rate: usize,
    pub n_mels: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub f_min: f32,
    pub f_max: Option<f32>,
}

impl Default for LogMelSpectrogramConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            n_mels: 160,
            n_fft: 2048,
            hop_length: 512,
            win_length: 2048,
            f_min: 0.0,
            f_max: None,
        }
    }
}

pub struct LogMelSpectrogram {
    pub sample_rate: usize,
    pub hop_length: usize,
    n_mels: usize,
    n_fft: usize,
    mel_filters: Vec<f32>,
}

impl LogMelSpectrogram {
    pub fn load(config: LogMelSpectrogramConfig) -> Result<Self> {
        let mel_filters = load_mel_filters(*&config.n_mels)?;

        Ok(Self {
            sample_rate: config.sample_rate,
            hop_length: config.hop_length,
            n_mels: config.n_mels,
            n_fft: config.n_fft,
            mel_filters,
        })
    }

    pub fn forward_f32(&self, x: &[f32]) -> Vec<f32> {
        log_mel_spectrogram_(
            x,
            &self.mel_filters,
            self.n_fft,
            self.hop_length,
            self.n_mels,
            false, // speed_up parameter, set to false for accuracy
        )
    }

    // This method works with Tensors (for Fish Speech compatibility)
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Convert Tensor to f32 slice
        let x_flat = x.flatten_all()?.to_vec1()?;

        // Process using the f32 method
        let mel = self.forward_f32(&x_flat);

        // Reshape and convert back to Tensor
        // Calculate the expected number of frames based on the input length
        let expected_frames = x_flat.len() / self.hop_length;

        // Reshape and convert back to Tensor
        let mel_len = mel.len();
        let frames = mel_len / self.n_mels;
        let truncated_frames = expected_frames.min(frames);

        // Truncate the mel spectrogram
        let truncated_mel: Vec<f32> = mel
            .into_iter()
            .take(self.n_mels * truncated_frames)
            .collect();

        Tensor::from_vec(
            truncated_mel,
            (1, self.n_mels, truncated_frames),
            x.device(),
        )
    }
}
