use candle_core::{DType, Device, Result, Tensor};
use std::collections::{HashMap, VecDeque};

pub mod encode;

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
                        self.penalty_mask
                            .slice_set(&self.one, 0, dropped_token as usize)?;
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
