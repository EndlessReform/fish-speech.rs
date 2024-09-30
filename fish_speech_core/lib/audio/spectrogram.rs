use anyhow::{bail, Result};
use byteorder::{ByteOrder, LittleEndian};
use candle_core::{utils::get_num_threads, Result as CandleResult, Tensor};
use std::sync::Arc;
use std::thread;

pub trait Float:
    num_traits::Float + num_traits::FloatConst + num_traits::NumAssign + Send + Sync
{
}

impl Float for f32 {}
impl Float for f64 {}

/// Vendored from [candle-transformers::models::whisper::audio](https://github.com/huggingface/candle/blob/2f49e1b5349f4e56677ec0d3dc3fe98f9cbb87c7/candle-transformers/src/models/whisper/audio.rs#L17).
///
/// Originally from [whisper.cpp](https://github.com/ggerganov/whisper.cpp/blob/4774d2feb01a772a15de81ffc34b34a1f294f020/whisper.cpp#L2337)
fn dft<T: Float>(inp: &[T]) -> Vec<T> {
    let zero = T::zero();
    let n = inp.len();
    let two_pi = T::PI() + T::PI();

    let mut out = Vec::with_capacity(2 * n);
    let n_t = T::from(n).unwrap();
    for k in 0..n {
        let k_t = T::from(k).unwrap();
        let mut re = zero;
        let mut im = zero;

        for (j, &inp) in inp.iter().enumerate() {
            let j_t = T::from(j).unwrap();
            let angle = two_pi * k_t * j_t / n_t;
            re += inp * angle.cos();
            im -= inp * angle.sin();
        }

        out.push(re);
        out.push(im);
    }
    out
}

/// Vendored from [candle_transformers/src/models/whisper/audio.rs](https://github.com/huggingface/candle/blob/2f49e1b5349f4e56677ec0d3dc3fe98f9cbb87c7/candle-transformers/src/models/whisper/audio.rs#L17) because it doesn't expose this directly.
///
/// Originally from [whisper.cpp](https://github.com/huggingface/candle/blob/2f49e1b5349f4e56677ec0d3dc3fe98f9cbb87c7/candle-transformers/src/models/whisper/audio.rs#L17)
fn fft<T: Float>(inp: &[T]) -> Vec<T> {
    let n = inp.len();
    let zero = T::zero();
    if n == 1 {
        return vec![inp[0], zero];
    }
    if n % 2 == 1 {
        return dft(inp);
    }
    let mut out = vec![zero; n * 2];

    let mut even = Vec::with_capacity(n / 2);
    let mut odd = Vec::with_capacity(n / 2);

    for (i, &inp) in inp.iter().enumerate() {
        if i % 2 == 0 {
            even.push(inp)
        } else {
            odd.push(inp);
        }
    }

    let even_fft = fft(&even);
    let odd_fft = fft(&odd);

    let two_pi = T::PI() + T::PI();
    let n_t = T::from(n).unwrap();
    for k in 0..n / 2 {
        let k_t = T::from(k).unwrap();
        let theta = two_pi * k_t / n_t;
        let re = theta.cos();
        let im = -theta.sin();

        let re_odd = odd_fft[2 * k];
        let im_odd = odd_fft[2 * k + 1];

        out[2 * k] = even_fft[2 * k] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

        out[2 * (k + n / 2)] = even_fft[2 * k] - re * re_odd + im * im_odd;
        out[2 * (k + n / 2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
    out
}

fn reflect_pad<T: Float>(samples: &[T], pad_amount: usize) -> Vec<T> {
    let mut padded = Vec::with_capacity(samples.len() + 2 * pad_amount);

    // Reflective pre-padding
    for i in 0..pad_amount {
        // Ensure we don't go out of bounds
        let idx = pad_amount - i - 1;
        padded.push(samples[idx]);
    }

    // Original samples
    padded.extend_from_slice(samples);

    // Reflective post-padding
    let len = samples.len();
    for i in 0..pad_amount {
        // Ensure we don't go out of bounds
        let idx = len - i - 1;
        padded.push(samples[idx]);
    }

    padded
}

/// Lightly adapted from candle_transformers
#[allow(clippy::too_many_arguments)]
// https://github.com/ggerganov/whisper.cpp/blob/4774d2feb01a772a15de81ffc34b34a1f294f020/whisper.cpp#L2414
fn log_mel_spectrogram_w<T: Float>(
    ith: usize,
    hann: &[T],
    samples: &[T],
    filters: &[T],
    n_fft: usize,
    hop_length: usize,
    n_len: usize,
    n_mel: usize,
    n_threads: usize,
) -> Vec<T> {
    let epsilon = T::from(1e-6).unwrap();

    // Number of FFT bins
    let n_fft_bins = 1 + n_fft / 2;

    let zero = T::zero();
    let mut fft_in = vec![zero; n_fft];
    let mut mel = vec![zero; n_len * n_mel];
    let n_samples = samples.len();
    let end = std::cmp::min(n_samples / hop_length, n_len);

    for i in (ith..end).step_by(n_threads) {
        let offset = i * hop_length;

        // Apply Hann window
        for j in 0..n_fft {
            if offset + j < samples.len() {
                fft_in[j] = hann[j] * samples[offset + j];
            } else {
                fft_in[j] = zero;
            }
        }

        // FFT
        let fft_out = fft(&fft_in);

        // Calculate magnitude spectrum with epsilon
        let mut magnitude = vec![zero; n_fft_bins];
        for j in 0..n_fft_bins {
            let re = fft_out[2 * j];
            let im = fft_out[2 * j + 1];
            magnitude[j] = (re * re + im * im).sqrt() + epsilon;
        }

        // Apply Mel filter bank
        for j in 0..n_mel {
            let mut sum = zero;
            for k in 0..n_fft_bins {
                sum += magnitude[k] * filters[j * n_fft_bins + k];
            }
            // Natural logarithm with clamping
            mel[j * n_len + i] = T::max(sum, T::from(1e-5).unwrap()).ln();
        }
    }

    mel
}

pub fn log_mel_spectrogram_<T: Float>(
    samples: &[T],
    filters: &[T],
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    n_mel: usize,
) -> Vec<T> {
    let zero = T::zero();
    let two_pi = T::PI() + T::PI();
    let half = T::from(0.5).unwrap();
    let one = T::from(1.0).unwrap();
    let n_fft_t = T::from(n_fft).unwrap();

    // Generate Hann window
    let hann: Vec<T> = (0..win_length)
        .map(|i| half * (one - ((two_pi * T::from(i).unwrap()) / n_fft_t).cos()))
        .collect();

    // Calculate the number of frames
    // let n_len = (samples.len() + hop_length - 1) / hop_length;

    // Calculate padding amount based on win_length and hop_length
    let pad_amount = if win_length > hop_length {
        (win_length - hop_length) / 2
    } else {
        0
    };

    // Apply reflective padding
    let padded_samples = if pad_amount > 0 {
        reflect_pad(samples, pad_amount)
    } else {
        samples.to_vec()
    };

    // Recalculate the number of frames after padding
    let n_len = (padded_samples.len() + hop_length - 1) / hop_length;

    // Ensure that the number of threads is even and less than 12
    let n_threads = std::cmp::min(get_num_threads() - (get_num_threads() % 2), 12);

    let hann = Arc::new(hann);
    let samples = Arc::new(padded_samples);
    let filters = Arc::new(filters.to_vec());

    // Use thread scope for safe concurrent execution
    let all_outputs = thread::scope(|s| {
        (0..n_threads)
            .map(|thread_id| {
                let hann = Arc::clone(&hann);
                let samples = Arc::clone(&samples);
                let filters = Arc::clone(&filters);
                s.spawn(move || {
                    log_mel_spectrogram_w(
                        thread_id, &hann, &samples, &filters, n_fft, hop_length, n_len, n_mel,
                        n_threads,
                    )
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().expect("Thread failed"))
            .collect::<Vec<_>>()
    });

    // Aggregate the results from all threads
    let l = all_outputs[0].len();
    let mut mel = vec![zero; l];

    for thread_output in all_outputs.iter() {
        for (i, &val) in thread_output.iter().enumerate() {
            mel[i] += val;
        }
    }

    // No additional dynamic range compression or normalization is applied
    mel
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
    win_length: usize,
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
            win_length: config.win_length,
        })
    }

    pub fn forward_f32(&self, x: &[f32]) -> Vec<f32> {
        log_mel_spectrogram_(
            x,
            &self.mel_filters,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.n_mels,
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
