use candle_core::{DType, Device, Result, Tensor};
use rand::Rng;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;

pub struct RepPenProcessor {
    penalty_mask: Tensor,
    one: Tensor,
    penalty_amt: Tensor,
    context: VecDeque<usize>,
    tokens_seen: HashMap<usize, usize>,
    max_ctxt_size: usize,
    vocab_size: usize,
}

impl RepPenProcessor {
    pub fn new(
        vocab_size: usize,
        max_ctxt_size: usize,
        penalty_amt: f32,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let penalty_mask = Tensor::ones(vocab_size, dtype, device)?;
        // Yes, this is inelegant, but there's no scalar interface to slice_set
        let one = Tensor::ones(1, dtype, device)?;
        let penalty_amt = Tensor::from_vec(vec![penalty_amt], 1, device)?.to_dtype(dtype)?;
        Ok(Self {
            penalty_mask,
            one,
            penalty_amt,
            context: VecDeque::new(),
            tokens_seen: HashMap::new(),
            max_ctxt_size,
            vocab_size,
        })
    }

    pub fn apply(&mut self, logits: &Tensor, last_token: usize) -> Result<Tensor> {
        if last_token >= self.vocab_size {
            candle_core::bail!("Token must be within vocab size");
        }

        // Add latest token to penalties if it's not there already
        let count = self.tokens_seen.entry(last_token).or_insert(1);
        if *count == 1 {
            // This is the first time we're penalizing the token in this window, so add to mask
            self.penalty_mask
                .slice_set(&self.penalty_amt, 0, last_token)?;
        }
        self.context.push_front(last_token);

        if self.context.len() > self.max_ctxt_size {
            // If the token falling out of the window is the last of its kind, un-penalize it
            if let Some(dropped_token) = self.context.pop_back() {
                if let Some(count) = self.tokens_seen.get_mut(&dropped_token) {
                    *count -= 1;
                    if *count == 0 {
                        self.tokens_seen.remove(&dropped_token);
                        self.penalty_mask.slice_set(&self.one, 0, dropped_token)?;
                    }
                }
            }
        }

        logits.broadcast_div(&self.penalty_mask)
    }

    pub fn clear_cache(&mut self) -> Result<()> {
        self.penalty_mask = self.penalty_mask.ones_like()?;
        self.context.clear();
        Ok(())
    }
}

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

pub fn load_prompt_text(
    prompt_path: &PathBuf,
    device: &Device,
    num_codebooks: usize,
) -> Result<Tensor> {
    let prompt_tokens = Tensor::read_npy(prompt_path)?.to_dtype(DType::U32)?;
    let prompt_tokens = prompt_tokens.to_dtype(DType::U32)?.to_device(device)?;
    match (prompt_tokens.rank(), prompt_tokens.dim(0)) {
        (2, Ok(n_actual_codebooks)) => {
            // Fine
            if n_actual_codebooks == num_codebooks {
                return Ok(prompt_tokens);
            } else {
                candle_core::bail!(
                    "Expected {} codebooks but got {}",
                    num_codebooks,
                    n_actual_codebooks
                )
            }
        }
        (3, Ok(1)) => {
            // Ghost third dimension OK
            if prompt_tokens.dim(1)? == num_codebooks {
                prompt_tokens.squeeze(0)
            } else {
                candle_core::bail!(
                    "Expected {} codebooks but got {}",
                    num_codebooks,
                    prompt_tokens.dim(1)?
                )
            }
        }
        (d, _) => {
            candle_core::bail!(
                "Incorrect prompt token dimensions for {:?}: {d}",
                prompt_path
            )
        }
    }
}

pub struct SamplingArgs {
    pub temp: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub repetition_penalty: f32,
}
