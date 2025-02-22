pub mod rep_pen;
use candle_core::{DType, Result, Tensor, D};
use candle_nn::ops::softmax_last_dim;
use rand::{distributions::Distribution, Rng, SeedableRng};
use rayon::prelude::*;

/// Extremely stripped-down CPU softmax for slow model out
pub fn legacy_softmax_sample(pad_prob: f32, eos_prob: f32, pad_id: u32, eos_id: u32) -> u32 {
    // Compute softmax
    let exp_pad = (pad_prob - pad_prob.max(eos_prob)).exp();
    let exp_eos = (eos_prob - pad_prob.max(eos_prob)).exp();
    let sum = exp_pad + exp_eos;

    let softmax_pad = exp_pad / sum;

    // Generate a random number
    let mut rng = rand::thread_rng();
    let rand_val: f32 = rng.gen(); // Generates a float between 0.0 and 1.0

    // Sample according to softmax probabilities
    if rand_val < softmax_pad {
        pad_id // pad_id
    } else {
        eos_id // im_end_id
    }
}

#[derive(Clone, Debug)]
pub struct SamplingArgs {
    pub temp: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub repetition_penalty: f32,
}

/// Simplified for my use case.
/// Drawn heavily from [Candle Transformers](https://docs.rs/candle-transformers/latest/src/candle_transformers/generation/mod.rs.html#18-21)
///
/// Thanks again, Kyutai
pub struct BatchedLogitsProcessor {
    rng: rand::rngs::StdRng,
    sampling_args: SamplingArgs,
}

impl BatchedLogitsProcessor {
    pub fn new(seed: u64, sampling_args: SamplingArgs) -> Self {
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self { rng, sampling_args }
    }

    fn sample_single_top_p_k(
        rng: &mut rand::rngs::StdRng,
        mut probs: Vec<f32>,
        top_k: usize,
        top_p: f64,
    ) -> u32 {
        if top_k >= probs.len() {
            return sample_topp(rng, &mut probs, top_p as f32);
        }

        let mut indices: Vec<_> = (0..probs.len()).collect();
        let (topk_indices, _, _) =
            indices.select_nth_unstable_by(top_k, |&i, &j| probs[j].total_cmp(&probs[i]));

        let mut topk_probs: Vec<_> = topk_indices.iter().map(|&i| probs[i]).collect();
        let sum_p = topk_probs.iter().sum::<f32>();

        let index = if top_p <= 0.0 || top_p >= sum_p as f64 {
            sample_multinomial(rng, &topk_probs)
        } else {
            sample_topp(rng, &mut topk_probs, top_p as f32)
        };

        topk_indices[index as usize] as u32
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<Vec<u32>> {
        let logits = logits.to_dtype(DType::F32)?;

        if self.sampling_args.temp <= 1e-7 {
            // Argmax sampling! yay
            return logits
                .argmax(D::Minus1)?
                .to_device(&candle_core::Device::Cpu)?
                .flatten_all()?
                .to_vec1::<u32>();
        }
        let probs = softmax_last_dim(&(&logits / self.sampling_args.temp)?)?
            .squeeze(1)?
            .to_vec2::<f32>()?;

        // Split RNG into independent streams
        let rngs = (0..probs.len())
            .map(|_| rand::rngs::StdRng::seed_from_u64(self.rng.gen()))
            .collect::<Vec<_>>();

        Ok(probs
            .into_par_iter()
            .zip(rngs)
            .map(|(prob_row, mut rng)| {
                Self::sample_single_top_p_k(
                    &mut rng,
                    prob_row,
                    self.sampling_args.top_k,
                    self.sampling_args.top_p,
                )
            })
            .collect())
    }
}

// Helper functions stripped from original code
fn sample_multinomial(rng: &mut rand::rngs::StdRng, probs: &[f32]) -> u32 {
    rand::distributions::WeightedIndex::new(probs)
        .map(|dist| dist.sample(rng) as u32)
        .unwrap_or(0)
}

fn sample_topp(rng: &mut rand::rngs::StdRng, probs: &mut [f32], top_p: f32) -> u32 {
    let mut indices: Vec<_> = (0..probs.len()).collect();
    indices.sort_by(|&i, &j| probs[j].total_cmp(&probs[i]));

    let mut cumsum = 0.0;
    for &idx in &indices {
        if cumsum >= top_p {
            probs[idx] = 0.0;
        }
        cumsum += probs[idx];
    }

    sample_multinomial(rng, probs)
}
