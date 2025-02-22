use super::stft::Spectrogram;
use byteorder::{ByteOrder, LittleEndian};
use candle_core::{Result as CandleResult, Tensor, D};
use num::complex::Complex;

fn complex_to_magnitude(spec: Vec<Complex<f64>>, num_freq_bins: usize) -> Vec<f32> {
    // Truncate the FFT output to keep only the first `num_freq_bins` (1025)
    spec.iter()
        .take(num_freq_bins) // Only take the first `1025` bins
        .map(|c| (c.re.powi(2) + c.im.powi(2)).sqrt() as f32) // Magnitude = sqrt(real^2 + imag^2)
        .collect()
}

fn reflect_pad(signal: &[f32], pad_size: usize) -> Vec<f32> {
    let mut padded = Vec::with_capacity(signal.len() + 2 * pad_size);

    // Reflect left padding
    padded.extend(signal[0..pad_size].iter().rev());

    // Original signal
    padded.extend(signal);

    // Reflect right padding
    padded.extend(signal[(signal.len() - pad_size)..].iter().rev());

    padded
}

fn linear_spectrogram(samples: &Tensor, fft_size: usize, hop_size: usize) -> CandleResult<Tensor> {
    let mut spectrogram = Spectrogram::new(fft_size, hop_size);

    // Flatten the tensor to a 1D vector of floats
    let x_flat = samples.flatten_all()?.to_vec1()?;

    // Padding the signal with reflect padding
    let pad_size = (fft_size - hop_size) / 2;
    let padded_signal = reflect_pad(&x_flat, pad_size);

    // Process the signal in hop_size chunks
    let mut spectrogram_frames = Vec::new();

    // Get the correct number of frequency bins (1025 for n_fft = 2048)
    let num_freq_bins = fft_size / 2 + 1;

    let mut start = 0;
    while start + hop_size <= padded_signal.len() {
        let chunk = &padded_signal[start..start + hop_size];

        // Add the chunk to the spectrogram, which returns a complex-valued result
        if let Some(spec) = spectrogram.add(chunk) {
            // Convert complex spectrogram to magnitude and truncate to 1025 bins
            let magnitude_frame = complex_to_magnitude(spec, num_freq_bins);
            spectrogram_frames.push(magnitude_frame);
        }

        start += hop_size;
    }

    // Handle any remaining data that's less than hop_size
    if start < padded_signal.len() {
        let chunk = &padded_signal[start..];
        if let Some(spec) = spectrogram.add(chunk) {
            let magnitude_frame = complex_to_magnitude(spec, num_freq_bins);
            spectrogram_frames.push(magnitude_frame);
        }
    }

    // Now, we have all spectrogram frames in magnitude, and we need to stack them into a tensor
    let num_frames = spectrogram_frames.len(); // Number of time steps

    // Flatten the frames properly
    let flat_spectrogram: Vec<f32> = spectrogram_frames
        .into_iter()
        .flat_map(|frame| frame) // Flatten each frame but maintain frequency count
        .collect();

    // Ensure the flattened spectrogram has the correct shape: [num_frames, num_freq_bins]
    Tensor::from_vec(
        flat_spectrogram,
        (num_frames, num_freq_bins),
        samples.device(),
    )? + 1e-6 // This is the epsilon added for numerical stability
}

fn load_mel_buffer(n_freqs: usize, num_mel_bins: usize) -> candle_core::Result<Tensor> {
    let mel_bytes = include_bytes!("melfilters160.bytes").as_slice();
    // Convert bytes to f32 values
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    LittleEndian::read_f32_into(mel_bytes, &mut mel_filters);
    Tensor::from_vec(
        mel_filters,
        (n_freqs, num_mel_bins),
        &candle_core::Device::Cpu,
    )
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
    n_fft: usize,
    mel_buffer: Tensor,
}

impl LogMelSpectrogram {
    pub fn load(config: LogMelSpectrogramConfig) -> CandleResult<Self> {
        let mel_buffer = load_mel_buffer((config.n_fft / 2) + 1, *&config.n_mels)?;

        Ok(Self {
            sample_rate: config.sample_rate,
            hop_length: config.hop_length,
            n_fft: config.n_fft,
            mel_buffer,
        })
    }

    fn apply_mel_scale(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x = x
            .transpose(D::Minus1, D::Minus1)?
            .matmul(&self.mel_buffer.to_device(x.device())?)?;
        x.transpose(D::Minus1, D::Minus2)
    }

    fn compress(&self, x: &Tensor) -> CandleResult<Tensor> {
        // TODO: What is max? Setting arbitrarily high
        x.clamp(1e-5, 100.0)?.log()
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Assume someone else took care of resampling
        let linear = linear_spectrogram(x, self.n_fft, self.hop_length)?;
        let x = self.apply_mel_scale(&linear)?;
        self.compress(&x)?.unsqueeze(0)
    }
}
