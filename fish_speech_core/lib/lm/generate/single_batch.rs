use super::utils::{constrain_probs_to_audio, rescale_semantic_tokens};
use crate::config::{WhichFishVersion, WhichLM};
use crate::lm::sampling::{
    legacy_softmax_sample, rep_pen::SingleBatchedRepPenProcessor, SamplingArgs,
};
use crate::lm::DualARTransformer;
use candle_core::{DType, IndexOp, Module, Result, Tensor, D};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::{Duration, Instant};

pub struct VQToken {
    /// Tensor version of tokens
    pub codes: Tensor,
    pub tokens: Vec<u32>,
    pub hidden_state: Tensor,
}

pub struct SingleBatchGenerator<'a> {
    model: &'a mut DualARTransformer,
    logits_processor: LogitsProcessor,
    rep_pen_processors: Vec<SingleBatchedRepPenProcessor>,
    pub input_pos: usize,
    max_new_tokens: usize,
    prompt: Option<Tensor>,
    previous_codes: Option<Vec<u32>>,
    audio_only: bool,
}

impl<'a> SingleBatchGenerator<'a> {
    pub fn new(
        model: &'a mut DualARTransformer,
        prompt: &Tensor,
        max_new_tokens: usize,
        sampling_args: &SamplingArgs,
        audio_only: bool,
    ) -> Result<Self> {
        let sampling = match sampling_args.temp {
            0.0 => Sampling::ArgMax,
            temp => Sampling::TopKThenTopP {
                temperature: temp,
                p: sampling_args.top_p,
                k: sampling_args.top_k,
            },
        };
        let logits_processor = LogitsProcessor::from_sampling(rand::random::<u64>(), sampling);
        let rep_pen_processors: Vec<SingleBatchedRepPenProcessor> = (0..model.cfg.num_codebooks)
            .map(|_| {
                SingleBatchedRepPenProcessor::new(
                    model.cfg.codebook_size,
                    16,
                    sampling_args.repetition_penalty,
                    model.fast_embeddings.embeddings().dtype(),
                    model.fast_embeddings.embeddings().device(),
                )
            })
            .collect::<Result<_>>()?;
        let input_pos = model.curr_kv_size()?;

        Ok(Self {
            max_new_tokens: max_new_tokens + model.curr_kv_size()?,
            model,
            prompt: Some(prompt.clone()),
            logits_processor,
            rep_pen_processors,
            input_pos,
            audio_only,
            previous_codes: None,
        })
    }
}

impl<'a> Iterator for SingleBatchGenerator<'a> {
    type Item = Result<VQToken>;

    fn next(&mut self) -> Option<Result<VQToken>> {
        if self.input_pos > self.max_new_tokens {
            println!(
                "Terminating early; input pos: {:?}, max: {:?}",
                self.input_pos, self.max_new_tokens
            );
            return None;
        }

        // Audio only, <|im_end|> was reached last time
        if self.prompt.is_none() {
            return None;
        }

        let x = self.prompt.as_ref().unwrap().clone();
        let prompt_length = x.dim(D::Minus1).unwrap();
        // This is to allow using ? operator for result inside iterator block
        // Sorry
        let result = (|| {
            let x = if x.rank() == 2 {
                x.unsqueeze(0)?
            } else {
                x.clone()
            };
            let (logits, hidden_states) = self.model.forward_generate(&x, self.input_pos, None)?;

            let semantic_token = if self.audio_only {
                match self.model.model_type {
                    WhichLM::Fish(WhichFishVersion::Fish1_2)
                    | WhichLM::Fish(WhichFishVersion::Fish1_4) => {
                        let slow_logits = logits.flatten_all()?;
                        // Fish 1.2 and 1.4: semantic backbone only samples PAD/<|im_end|>
                        // Ah the halcyon days where we're not forced to do a giant softmax to double up the first semantic codes
                        let pad_prob = slow_logits
                            .i(self.model.token_config.pad_id as usize)?
                            .to_dtype(DType::F32)?
                            .to_scalar::<f32>()?;
                        let eos_prob = slow_logits
                            .i(self.model.token_config.im_end_id as usize)?
                            .to_dtype(DType::F32)?
                            .to_scalar::<f32>()?;

                        legacy_softmax_sample(
                            pad_prob,
                            eos_prob,
                            self.model.token_config.pad_id,
                            self.model.token_config.im_end_id,
                        )
                    }
                    _ => {
                        let slow_logits = constrain_probs_to_audio(
                            &logits,
                            &self.model.model_type,
                            &self.model.token_config,
                        )?
                        .flatten_all()?;

                        let shifted_token = self.logits_processor.sample(&slow_logits)?;
                        rescale_semantic_tokens(
                            vec![shifted_token],
                            &self.model.model_type,
                            &self.model.token_config,
                        )[0]
                    }
                }
            } else {
                // Unconstrained generation: accept the huge vocab size and just sample
                self.logits_processor.sample(&logits)?
            };
            let mut codebooks = vec![semantic_token];
            self.model.clear_fast_layer_caches();

            // Generate token
            let mut x = hidden_states.clone();
            // TODO: Skip this and short-circuit when we handle generating text only
            for codebook_idx in 0..self.model.cfg.num_codebooks {
                // Skip final generation step
                if self.audio_only && semantic_token == self.model.token_config.im_end_id {
                    codebooks.push(0);
                    continue;
                }
                let logits = self
                    .model
                    .forward_generate_fast(&x, codebook_idx)?
                    .flatten_all()?;

                let logits_adj = match (&self.previous_codes, self.model.cfg.depthwise_wte) {
                    (None, _) => logits.clone(),
                    // Turning off rep pen for smoltts for now
                    (_, Some(true)) => logits.clone(),
                    (Some(t), _) => self.rep_pen_processors[codebook_idx]
                        .apply(&logits, t[codebook_idx + 1] as usize)?,
                };
                let a = self.logits_processor.sample(&logits_adj.flatten_all()?)?;
                let a_tensor = Tensor::from_slice(&[a], 1, x.device())?;
                let a_tensor = if let Some(true) = self.model.cfg.depthwise_wte {
                    (a_tensor + 0.max(codebook_idx * self.model.cfg.codebook_size) as f64)?
                } else {
                    a_tensor
                };
                if codebook_idx != self.model.cfg.num_codebooks - 1 {
                    x = self
                        .model
                        .fast_embeddings
                        .forward(&a_tensor)?
                        .unsqueeze(0)?;
                }
                codebooks.push(a);
            }
            let codes_tensor = Tensor::from_vec(
                codebooks.clone(),
                self.model.cfg.num_codebooks + 1,
                x.device(),
            )?
            .unsqueeze(D::Minus1)?;

            // Internal state bookkeeping
            if self.previous_codes.is_none() {
                self.input_pos += prompt_length;
            } else {
                self.input_pos += 1;
            }
            self.previous_codes = Some(codebooks.clone());
            self.prompt = if self.audio_only && semantic_token == self.model.token_config.im_end_id
            {
                None
            } else {
                Some(codes_tensor.clone())
            };

            Ok(VQToken {
                tokens: codebooks,
                codes: codes_tensor,
                hidden_state: hidden_states,
            })
        })();

        Some(result)
    }
}

pub fn generate_blocking_with_hidden(
    model: &mut DualARTransformer,
    prompt: &Tensor,
    max_new_tokens: usize,
    sampling_args: &SamplingArgs,
    collect_hidden_states: bool,
    show_progress: bool,
) -> Result<(Tensor, Option<Tensor>)> {
    // TODO: Handle text output
    let audio_only = true;
    let n_cached = model.curr_kv_size()?;
    let im_end_id = model.token_config.im_end_id;
    let prompt_size = prompt.dim(D::Minus1)?;
    let mut generator =
        SingleBatchGenerator::new(model, prompt, max_new_tokens, sampling_args, audio_only)?;

    let start_pp = Instant::now();
    let first_vq_token = generator.next().ok_or(candle_core::Error::Msg(
        "Prefill mistakenly thought generation ended. Please check max tokens".into(),
    ))??;
    let dt = start_pp.elapsed();
    if show_progress {
        println!(
            "{:.2}ms prompt processing: {} tokens ({} new, {} cached, {:.2} tokens/s)",
            dt.as_secs_f64() * 1000.0,
            generator.input_pos,
            generator.input_pos - n_cached,
            n_cached,
            prompt_size as f64 / dt.as_secs_f64()
        );
    }

    // Set up concatenation batch
    let mut previous_tokens: Vec<Tensor> = vec![first_vq_token.codes];
    let mut hidden_states: Vec<Tensor> = vec![first_vq_token.hidden_state];

    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg} [{elapsed_precise}] {per_sec} iterations/s")
            .unwrap()
            .tick_chars("/|\\- "),
    );
    spinner.enable_steady_tick(Duration::from_millis(100));
    let start_decode = Instant::now();
    for (i, maybe_vq_token) in generator.into_iter().enumerate() {
        let vq_token = maybe_vq_token?;
        if vq_token.tokens[0] != im_end_id {
            previous_tokens.push(vq_token.codes);
        }

        if collect_hidden_states {
            hidden_states.push(vq_token.hidden_state);
        }

        if show_progress {
            spinner.inc(1);
            spinner.set_message(format!("Tokens: {}", i));
        }
    }
    let dt = start_decode.elapsed();

    let full_output = Tensor::cat(&previous_tokens, 1)?;
    let out_tokens = if audio_only {
        full_output.i((1.., ..))?
    } else {
        full_output
    };
    let out_len = previous_tokens.len() as f64;
    let hidden_states = match collect_hidden_states {
        true => Tensor::cat(&hidden_states, 0).ok(),
        false => None,
    };

    if show_progress {
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
    }
    Ok((out_tokens, hidden_states))
}

pub fn generate_blocking(
    model: &mut DualARTransformer,
    prompt: &Tensor,
    max_new_tokens: usize,
    sampling_args: &SamplingArgs,
    show_progress: bool,
) -> Result<Tensor> {
    let (out, _) = generate_blocking_with_hidden(
        model,
        prompt,
        max_new_tokens,
        sampling_args,
        false,
        show_progress,
    )?;
    Ok(out)
}
