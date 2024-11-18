use super::sample::{softmax_sample, RepPenProcessor, SamplingArgs};
use crate::models::text2semantic::silence_detector::PauseTypeProbe;
use crate::models::text2semantic::DualARTransformer;
use candle_core::{DType, IndexOp, Module, Result, Tensor, D};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::{Duration, Instant};

/// (codes, codebook tensor, hidden states)
fn decode_one_token_ar(
    model: &mut DualARTransformer,
    fast_logits_processor: &mut LogitsProcessor,
    x: &Tensor,
    input_pos: usize,
    im_end_id: u32,
    pad_id: u32,
    previous_token: Option<Vec<u32>>,
    rep_pens: &mut [RepPenProcessor],
) -> Result<(Vec<u32>, Tensor, Tensor)> {
    let (logits, hidden_states) = model.forward_generate(&x, input_pos)?;
    let slow_logits = logits.flatten_all()?;

    let pad_prob = slow_logits
        .i(pad_id as usize)?
        .to_dtype(DType::F32)?
        .to_scalar::<f32>()?;
    let eos_prob = slow_logits
        .i(im_end_id as usize)?
        .to_dtype(DType::F32)?
        .to_scalar::<f32>()?;

    let semantic_token = softmax_sample(pad_prob, eos_prob, pad_id, im_end_id);
    let mut codebooks = vec![semantic_token];
    model.clear_fast_layer_caches();

    let mut x = hidden_states.clone();
    for codebook_idx in 0..model.cfg.num_codebooks {
        let logits = model
            .forward_generate_fast(&x, codebook_idx)?
            .flatten_all()?;

        let logits_adj = match &previous_token {
            None => logits.clone(),
            Some(t) => rep_pens[codebook_idx].apply(&logits, t[codebook_idx + 1] as usize)?,
        };
        let a = fast_logits_processor.sample(&logits_adj.flatten_all()?)?;
        let a_tensor = Tensor::from_slice(&[a], 1, x.device())?;
        x = model.fast_embeddings.forward(&a_tensor)?.unsqueeze(0)?;
        codebooks.push(a);
    }
    let codes_tensor =
        Tensor::from_vec(codebooks.clone(), model.cfg.num_codebooks + 1, x.device())?
            .unsqueeze(D::Minus1)?;
    Ok((codebooks, codes_tensor, hidden_states))
}

/// Takes a conditioning sequence as input and generates as many tokens as requested
pub fn generate(
    model: &mut DualARTransformer,
    prompt: &Tensor,
    max_new_tokens: usize,
    im_end_id: u32,
    pad_id: u32,
    sampling_args: &SamplingArgs,
    collect_hidden_states: bool,
) -> Result<(Tensor, Option<Tensor>)> {
    let sampling = match sampling_args.temp {
        0.0 => Sampling::ArgMax,
        temp => Sampling::TopKThenTopP {
            temperature: temp,
            p: sampling_args.top_p,
            k: sampling_args.top_k,
        },
    };
    let mut fast_logits_processor = LogitsProcessor::from_sampling(42, sampling);
    let maybe_fast_rep_pens: Result<Vec<RepPenProcessor>> = (0..model.cfg.num_codebooks)
        .map(|_| {
            RepPenProcessor::new(
                model.cfg.codebook_size,
                16,
                sampling_args.repetition_penalty,
                model.fast_embeddings.embeddings().dtype(),
                model.fast_embeddings.embeddings().device(),
            )
        })
        .collect();
    let mut fast_rep_pens = maybe_fast_rep_pens?;

    let start_pp = Instant::now();
    let (mut previous_token, mut cur_token, hidden_state_first) = decode_one_token_ar(
        model,
        &mut fast_logits_processor,
        prompt,
        0,
        im_end_id,
        pad_id,
        None,
        &mut fast_rep_pens,
    )?;
    let dt = start_pp.elapsed();
    let mut input_pos = prompt.dim(D::Minus1)?;
    println!(
        "{} prompt processing timesteps ({:.2} tokens/s)",
        input_pos,
        input_pos as f64 / dt.as_secs_f64()
    );

    let mut previous_tokens = cur_token.clone();

    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg} [{elapsed_precise}] {per_sec} iterations/s")
            .unwrap()
            .tick_chars("/|\\- "),
    );
    spinner.enable_steady_tick(Duration::from_millis(100));
    spinner.set_message("Generating features");
    let mut hidden_states: Vec<Tensor> = vec![hidden_state_first];

    let start_decode = Instant::now();
    for i in 1..max_new_tokens {
        let (next_indices, next_token, hidden_state) = decode_one_token_ar(
            model,
            &mut fast_logits_processor,
            &cur_token,
            input_pos,
            im_end_id,
            pad_id,
            Some(previous_token),
            &mut fast_rep_pens,
        )?;
        if collect_hidden_states {
            hidden_states.push(hidden_state);
        }
        previous_tokens = Tensor::cat(&[previous_tokens, next_token.clone()], D::Minus1)?;
        spinner.inc(1);
        spinner.set_message(format!("Tokens: {}", i));
        if next_indices[0] == im_end_id {
            break;
        }
        input_pos += 1;
        cur_token = next_token;
        previous_token = next_indices;
    }
    let dt = start_decode.elapsed();
    let out_len = previous_tokens.dim(1)? as f64;
    let hidden_states = match collect_hidden_states {
        true => Tensor::cat(&hidden_states, 0).ok(),
        false => None,
    };
    println!(
        "{} tokens generated ({:.2} tokens/s, {:.3}ms / token, RTF: {:.3})",
        out_len,
        out_len / dt.as_secs_f64(),
        (dt.as_secs_f64() * 1e3) / (out_len - 1f64),
        (out_len / 21.535) / dt.as_secs_f64()
    );
    Ok((previous_tokens.i((1.., ..))?, hidden_states))
}

#[derive(Debug)]
enum ChunkedTokenState {
    FlushBuffer,
    Continue,
    EOS,
}

pub struct NeuralChunkedGenerator<'a> {
    model: &'a mut DualARTransformer,
    silence_probe: Option<&'a PauseTypeProbe>,
    max_new_tokens: usize,
    im_end_id: u32,
    pad_id: u32,
    token_buffer: Vec<Tensor>,
    prev_codes_tensor: Tensor,
    prev_codes: Vec<u32>,
    initial_silence: bool,
    consecutive_silence: usize,
    fast_logits_processor: LogitsProcessor,
    fast_rep_pens: Vec<RepPenProcessor>,
    input_pos: usize,
    is_eos: bool,
}

impl<'a> NeuralChunkedGenerator<'a> {
    pub fn prefill(
        model: &'a mut DualARTransformer,
        silence_probe: Option<&'a PauseTypeProbe>,
        prompt: Tensor,
        max_new_tokens: usize,
        im_end_id: u32,
        pad_id: u32,
        sampling_args: &'a SamplingArgs,
    ) -> Result<Self> {
        let sampling = match sampling_args.temp {
            0.0 => Sampling::ArgMax,
            temp => Sampling::TopKThenTopP {
                temperature: temp,
                p: sampling_args.top_p,
                k: sampling_args.top_k,
            },
        };
        let mut fast_logits_processor = LogitsProcessor::from_sampling(42, sampling);
        let maybe_fast_rep_pens: Result<Vec<RepPenProcessor>> = (0..model.cfg.num_codebooks)
            .map(|_| {
                RepPenProcessor::new(
                    model.cfg.codebook_size,
                    16,
                    sampling_args.repetition_penalty,
                    model.fast_embeddings.embeddings().dtype(),
                    model.fast_embeddings.embeddings().device(),
                )
            })
            .collect();
        let mut fast_rep_pens = maybe_fast_rep_pens?;
        let start_pp = Instant::now();

        let (cur_codes, cur_token, hidden_state_first) = decode_one_token_ar(
            model,
            &mut fast_logits_processor,
            &prompt,
            0,
            im_end_id,
            pad_id,
            None,
            &mut fast_rep_pens,
        )?;
        let dt = start_pp.elapsed();
        let input_pos = prompt.dim(D::Minus1)?;
        let initial_silence = match silence_probe {
            Some(p) => p.forward(&hidden_state_first)?,
            None => false,
        };
        let token_buffer = vec![cur_token.clone()];
        println!(
            "{} prompt processing timesteps ({:.2} tokens/s)",
            input_pos,
            input_pos as f64 / dt.as_secs_f64()
        );

        Ok(Self {
            model,
            silence_probe,
            max_new_tokens,
            im_end_id,
            pad_id,
            token_buffer,
            initial_silence,
            consecutive_silence: 0,
            prev_codes: cur_codes,
            prev_codes_tensor: cur_token,
            input_pos,
            fast_rep_pens,
            fast_logits_processor,
            is_eos: false,
        })
    }

    fn generate_token(&mut self) -> Result<ChunkedTokenState> {
        if self.input_pos > self.max_new_tokens {
            println!(
                "OVER MAX TOKENS: {:?}, {}",
                self.input_pos, self.max_new_tokens
            );
            return Ok(ChunkedTokenState::EOS);
        }
        let (next_indices, next_token, hidden_state) = decode_one_token_ar(
            self.model,
            &mut self.fast_logits_processor,
            &self.prev_codes_tensor,
            self.input_pos,
            self.im_end_id,
            self.pad_id,
            Some(self.prev_codes.clone()),
            &mut self.fast_rep_pens,
        )?;
        self.prev_codes_tensor = next_token.clone();
        self.token_buffer.push(next_token);
        self.input_pos += 1;
        if next_indices[0] == self.im_end_id {
            println!("EOS! Flushing everything");
            return Ok(ChunkedTokenState::EOS);
        }
        self.prev_codes = next_indices;
        let frame_is_silent = match self.silence_probe {
            Some(p) => p.forward(&hidden_state)?,
            None => false,
        };

        if frame_is_silent {
            self.consecutive_silence += 1;
            if !self.initial_silence && self.consecutive_silence >= 3 {
                println!(
                    "Three consecutive silence tokens! Flushing {} tokens",
                    self.token_buffer.len()
                );
                // Hysteresis for after the flush
                self.consecutive_silence = 0;
                self.initial_silence = true;
                return Ok(ChunkedTokenState::FlushBuffer);
            } else {
                return Ok(ChunkedTokenState::Continue);
            }
        } else {
            self.consecutive_silence = 0;
            self.initial_silence = false;
            return Ok(ChunkedTokenState::Continue);
        }
    }

    fn flush_buffer(&mut self) -> Result<Tensor> {
        let codes_chunk = Tensor::cat(&self.token_buffer, D::Minus1)?;
        self.token_buffer.clear();
        codes_chunk.i((1.., ..))
    }
}

impl<'a> Iterator for NeuralChunkedGenerator<'a> {
    type Item = Result<Tensor>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.is_eos {
            return None;
        }
        loop {
            let maybe_code = self.generate_token();
            match maybe_code {
                Err(e) => return Some(Err(e)),
                Ok(ChunkedTokenState::Continue) => continue,
                Ok(ChunkedTokenState::FlushBuffer) => return Some(self.flush_buffer()),
                Ok(ChunkedTokenState::EOS) => {
                    self.is_eos = true;
                    // TODO: Configure caching speaker prompts
                    self.model.clear_slow_layer_caches();
                    return Some(self.flush_buffer());
                }
            }
        }
    }
}
