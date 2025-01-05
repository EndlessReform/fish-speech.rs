use super::utils::{constrain_probs_to_audio, rescale_semantic_tokens};
use crate::models::{text2semantic::DualARTransformer, vqgan::config::WhichLM};
use candle_core::{Device, IndexOp, Module, Result, Tensor, D};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct BatchPosition {
    pub codes: Tensor,
    pub is_audio: bool,
    // false if audio-only and the sequence is done
    pub is_active: bool,
}

pub struct BatchGenerator<'a> {
    model: &'a mut DualARTransformer,
    pub input_pos: usize,
    max_new_tokens: usize,
    prompt: Option<Tensor>,
    pad_mask: Option<Tensor>,
    audio_only: bool,
    batch_item_is_dead: Vec<bool>,
    bsz: usize,
}

impl<'a> BatchGenerator<'a> {
    // Expects prompts (num_codebooks + 1, seqlen); will do batch packing for you
    pub fn new(
        model: &'a mut DualARTransformer,
        prompts: &[Tensor],
        max_new_tokens: usize,
        audio_only: bool,
    ) -> Result<Self> {
        let (prompt, pad_mask) = BatchGenerator::pad_prompts(prompts, model, max_new_tokens)?;

        // Deliberately not implementing any sampling stuff for now
        // will just argmax to prove correctness before proceeding
        // Yes, this sucks
        Ok(Self {
            model,
            prompt: Some(prompt),
            pad_mask: Some(pad_mask),
            // Will clear KV cache before proceeding
            input_pos: 0,
            max_new_tokens,
            audio_only,
            batch_item_is_dead: vec![false; prompts.len()],
            bsz: prompts.len(),
        })
    }

    // returns (prompt, mask) concatenated
    fn pad_prompts(
        prompts: &[Tensor],
        model: &DualARTransformer,
        max_new_tokens: usize,
    ) -> Result<(Tensor, Tensor)> {
        if prompts.len() == 0 {
            candle_core::bail!("Must have at least one prompt")
        }
        let prompt_lengths: Vec<usize> = prompts
            .iter()
            .map(|p| p.dim(D::Minus1))
            .collect::<Result<_>>()?;
        let max_prefill_length = *prompt_lengths.iter().max().unwrap();
        if max_prefill_length > max_new_tokens {
            candle_core::bail!(
                "At least one prompt is longer than max seq len of {} tokens",
                max_new_tokens
            );
        }

        let codebook_pad_tensor = Tensor::zeros(
            (model.cfg.num_codebooks, max_prefill_length),
            prompts[0].dtype(),
            prompts[0].device(),
        )?;
        // This is how it happens in Fish Speech and my DualAR.
        // Hey, even if it's wrong, it's masked anyway so whatever
        let im_end_tensor = Tensor::full(
            model.token_config.im_end_id,
            (1, max_prefill_length),
            prompts[0].device(),
        )?;
        let pad_tensor = Tensor::cat(&[im_end_tensor, codebook_pad_tensor], 0)?;

        let mut padded_prompts = Vec::with_capacity(prompts.len());
        let mut pad_masks = Vec::with_capacity(prompts.len());
        let range = Tensor::arange(0u32, max_prefill_length as u32, prompts[0].device())?;

        for (p, &l) in prompts.iter().zip(&prompt_lengths) {
            padded_prompts.push(Tensor::cat(
                &[pad_tensor.i((.., ..(max_prefill_length - l)))?, p.clone()],
                1,
            )?);
            pad_masks.push(range.gt((max_prefill_length - l) as u32)?);
        }
        Ok((
            Tensor::stack(&padded_prompts, 0)?,
            Tensor::stack(&pad_masks, 0)?,
        ))
    }
}

impl<'a> Iterator for BatchGenerator<'a> {
    type Item = Result<Vec<BatchPosition>>;

    fn next(&mut self) -> Option<Result<Vec<BatchPosition>>> {
        if self.input_pos == 0 {
            // Prefill; we have no guarantees that previous size is correct
            self.model.clear_slow_layer_caches();
        }
        if self.prompt.is_none() || self.input_pos > self.max_new_tokens {
            // All generations are done
            return None;
        }
        let prompt = self.prompt.as_ref().unwrap().clone();
        let result = (|| {
            let (slow_logits, hidden_states) =
                self.model
                    .forward_generate(&prompt, self.input_pos, self.pad_mask.clone())?;

            let slow_logits = if self.audio_only {
                constrain_probs_to_audio(
                    &slow_logits,
                    &self.model.model_type,
                    &self.model.token_config,
                )?
            } else {
                slow_logits
            };
            // TODO add fancier sampling for batch (temp is low-hanging fruit)
            // Argmax sampling for now to prove correctness of rest of harness
            let raw_slow_ids = slow_logits
                .argmax(D::Minus1)?
                .to_device(&Device::Cpu)?
                .to_vec1::<u32>()?;
            let slow_ids = if self.audio_only {
                rescale_semantic_tokens(
                    raw_slow_ids,
                    &self.model.model_type,
                    &self.model.token_config,
                )
            } else {
                raw_slow_ids
            };

            // Bookkeeping: Mark newly dead positions as dead
            if slow_ids.len() != self.bsz {
                candle_core::bail!(
                    "Expected {} tokens output from slow forward pass but got {}",
                    self.bsz,
                    slow_ids.len()
                )
            }
            let position_is_newly_dead = if self.audio_only {
                slow_ids
                    .iter()
                    .map(|id| *id == self.model.token_config.im_end_id)
                    .collect()
            } else {
                vec![false; self.bsz]
            };
            self.batch_item_is_dead = self
                .batch_item_is_dead
                .iter()
                .zip(position_is_newly_dead.iter())
                .map(|(a, b)| *a || *b)
                .collect();

            let mut x = hidden_states.clone();
            self.model.clear_fast_layer_caches();

            let mut codebook_ids: Vec<Vec<u32>> = slow_ids
                .iter()
                .map(|id| {
                    let mut arr = vec![*id];
                    arr.reserve_exact(self.model.cfg.num_codebooks);
                    arr
                })
                .collect();

            for codebook_idx in 0..self.model.cfg.num_codebooks {
                let fast_logits = self.model.forward_generate_fast(&x, codebook_idx)?;
                // TODO: put sampling here too
                let ids = fast_logits
                    .argmax(D::Minus1)?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<u32>()?;
                // Prime hidden state WTE for next round
                x = self.model.fast_embeddings.forward(&Tensor::from_slice(
                    &ids,
                    ids.len(),
                    x.device(),
                )?)?;
                for (idx_in_batch, id) in ids.iter().enumerate() {
                    codebook_ids[idx_in_batch].push(*id);
                }
            }

            let (batch_positions, next_prompt_codes): (Vec<_>, Vec<_>) = codebook_ids
                .iter()
                .zip(self.batch_item_is_dead.iter())
                .map(|(ids, &is_dead)| {
                    let slow_id = ids[0];
                    let is_audio = slow_id >= self.model.token_config.semantic_start_id;
                    let vq_codes = if !is_audio {
                        let mut arr = vec![0; self.model.cfg.num_codebooks + 1];
                        arr[0] = slow_id;
                        arr
                    } else {
                        ids.clone()
                    };

                    let codes_tensor =
                        Tensor::from_vec(vq_codes, self.model.cfg.num_codebooks + 1, x.device())?;

                    Ok((
                        BatchPosition {
                            codes: codes_tensor.clone(),
                            is_audio,
                            is_active: !is_dead,
                        },
                        codes_tensor,
                    ))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip();

            // Final bookkeeping
            self.prompt = if self.audio_only && self.batch_item_is_dead.iter().all(|i| *i) {
                // All prompts are dead, end generation
                None
            } else {
                Some(Tensor::stack(&next_prompt_codes, 0)?)
            };
            if self.input_pos == 0 {
                // Prefill
                self.input_pos += prompt.dim(D::Minus1)?;
            } else {
                self.input_pos += 1;
            }
            self.pad_mask = None;

            Ok(batch_positions)
        })();

        Some(result)
    }
}

struct GeneratedSequence {
    tokens: Vec<Tensor>, // each [num_codebooks + 1, 1]
    is_audio_steps: Vec<bool>,
}

pub fn generate_static_batch(
    model: &mut DualARTransformer,
    prompts: &[Tensor],
    max_new_tokens: usize,
    audio_only: bool,
) -> Result<(Vec<Tensor>, Vec<Vec<bool>>)> {
    let mut generator = BatchGenerator::new(model, prompts, max_new_tokens, audio_only)?;

    let start_pp = Instant::now();
    let first_position: Vec<BatchPosition> = generator.next().ok_or(candle_core::Error::Msg(
        "Prefill mistakenly thought generation ended. Please check max tokens".into(),
    ))??;
    let dt = start_pp.elapsed();
    println!(
        "{:.2}ms prompt processing: {} seqlen at bs={} ({:.2} tokens/s)",
        dt.as_secs_f64() * 1000.0,
        generator.input_pos,
        prompts.len(),
        generator.input_pos as f64 / dt.as_secs_f64()
    );
    let mut sequences: Vec<GeneratedSequence> = first_position
        .into_iter()
        .map(|pos| GeneratedSequence {
            tokens: vec![pos.codes],
            is_audio_steps: vec![pos.is_audio],
        })
        .collect();

    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg} [{elapsed_precise}] {per_sec} iterations/s")
            .unwrap()
            .tick_chars("/|\\- "),
    );
    spinner.enable_steady_tick(Duration::from_millis(100));

    let start_decode = Instant::now();
    for (i, maybe_batch_pos) in generator.into_iter().enumerate() {
        let vq_token = maybe_batch_pos?;
        for (seq, pos) in sequences.iter_mut().zip(vq_token.into_iter()) {
            if !(audio_only && !pos.is_active) {
                seq.tokens.push(pos.codes);
                seq.is_audio_steps.push(pos.is_audio);
            }
        }

        spinner.inc(1);
        spinner.set_message(format!("Tokens: {}", i));
    }
    let dt = start_decode.elapsed();
    // this is fine since we ruled out bs=0 earlier
    let out_len = sequences[0].is_audio_steps.len() as f64;

    let (output_seqs, is_audio): (Vec<_>, Vec<_>) = sequences
        .into_iter()
        .map(|seq| {
            let tokens = Tensor::cat(&seq.tokens, D::Minus1)?;
            Ok((tokens, seq.is_audio_steps))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .unzip();

    let full_output = output_seqs
        .into_iter()
        .map(|t| if audio_only { t.i((1.., ..)) } else { Ok(t) })
        .collect::<Result<_>>()?;

    let frame_rate = match model.model_type {
        WhichLM::DualAR => 12.5,
        _ => 21.535,
    };
    println!(
        "{} tokens generated in {:.3}s ({:.2} tokens/s, {:.3}ms / token, RTF: {:.3})",
        out_len,
        dt.as_secs_f64(),
        out_len / dt.as_secs_f64(),
        (dt.as_secs_f64() * 1e3) / (out_len - 1f64),
        (out_len / frame_rate) / dt.as_secs_f64()
    );

    Ok((full_output, is_audio))
}
