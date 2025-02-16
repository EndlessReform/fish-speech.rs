use candle_core::{DType, Device, Result, Tensor};
use std::collections::{HashMap, VecDeque};

pub struct SingleBatchedRepPenProcessor {
    penalty_mask: Tensor,
    one: Tensor,
    penalty_amt: Tensor,
    context: VecDeque<usize>,
    tokens_seen: HashMap<usize, usize>,
    max_ctxt_size: usize,
    vocab_size: usize,
}

impl SingleBatchedRepPenProcessor {
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

struct SequencePenalty {
    penalty_mask: Vec<f32>,
    context: VecDeque<u32>,
    tokens_seen: HashMap<u32, u32>,
    penalty_amt: f32,
    max_ctxt_size: usize,
}

impl SequencePenalty {
    pub fn new(max_ctxt_size: usize, vocab_size: usize, penalty_amt: f32) -> Self {
        Self {
            context: VecDeque::with_capacity(max_ctxt_size),
            tokens_seen: HashMap::new(),
            penalty_mask: vec![1.0; vocab_size],
            penalty_amt,
            max_ctxt_size,
        }
    }

    pub fn update_mask(&mut self, token: u32) -> Result<()> {
        if token as usize >= self.penalty_mask.len() {
            candle_core::bail!(
                "Rep pen: Token {} exceeds vocab size {}",
                token,
                self.penalty_mask.len()
            )
        }

        let count = self.tokens_seen.entry(token).or_insert(1);
        if *count == 1 {
            // First time we see token: add it to penalty
            // This is fine due to bounds checking earlier
            self.penalty_mask[token as usize] = self.penalty_amt;
        }
        self.context.push_front(token);

        if self.context.len() > self.max_ctxt_size {
            if let Some(dropped_token) = self.context.pop_back() {
                if let Some(count) = self.tokens_seen.get_mut(&dropped_token) {
                    *count -= 1;
                    if *count == 0 {
                        self.tokens_seen.remove(&dropped_token);
                        // This is OK since it passed the bounds check earlier
                        self.penalty_mask[dropped_token as usize] = 1.0;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn get_mask(&self) -> Vec<f32> {
        self.penalty_mask.clone()
    }
}

pub struct BatchedRepPenProcessor {
    sequence_pens: Vec<SequencePenalty>,
    bsz: usize,
}

impl BatchedRepPenProcessor {
    pub fn new(vocab_size: usize, max_ctxt_size: usize, penalty_amt: f32, bsz: usize) -> Self {
        let sequence_pens = (0..bsz)
            .map(|_| SequencePenalty::new(max_ctxt_size, vocab_size, penalty_amt))
            .collect();
        Self { sequence_pens, bsz }
    }

    pub fn update_mask(&mut self, last_tokens: Vec<u32>) -> Result<()> {
        if last_tokens.len() != self.bsz {
            candle_core::bail!(
                "Expected bsz {} tokens in rep pen but got last tokens: {}",
                self.bsz,
                last_tokens.len()
            );
        };
        for (seq, token) in self.sequence_pens.iter_mut().zip(last_tokens) {
            seq.update_mask(token)?;
        }
        Ok(())
    }

    pub fn apply_mask(&self, logits: &Tensor) -> Result<Tensor> {
        let (logit_bsz, seqlen, _) = logits.dims3()?;
        if seqlen != 1 {
            candle_core::bail!("Seqlen 1 only supported for rep pen but got {:?}", seqlen);
        }
        if logit_bsz != self.bsz {
            candle_core::bail!(
                "Expected bsz {} tokens in rep pen but got logits: {}",
                self.bsz,
                logit_bsz
            );
        };

        // TODO: parallelize this if it takes a while, I doubt it
        let masks: Vec<Vec<f32>> = self
            .sequence_pens
            .iter()
            .map(|seq| seq.get_mask())
            .collect();
        let penalty_mask = Tensor::new(masks, logits.device())?.unsqueeze(1)?;

        logits.div(&penalty_mask)
    }
}
