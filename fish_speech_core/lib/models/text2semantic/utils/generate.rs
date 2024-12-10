use super::sample::{legacy_softmax_sample, RepPenProcessor, SamplingArgs};
use crate::models::text2semantic::DualARTransformer;
use candle_core::{DType, IndexOp, Module, Result, Tensor, D};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::{Duration, Instant};

fn decode_one_token_ar(
    model: &mut DualARTransformer,
    fast_logits_processor: &mut LogitsProcessor,
    x: &Tensor,
    input_pos: usize,
    im_end_id: u32,
    pad_id: u32,
    previous_token: Option<Vec<u32>>,
    rep_pens: &mut [RepPenProcessor],
) -> Result<(Vec<u32>, Tensor)> {
    let (logits, hidden_states) = model.forward_generate(&x, input_pos)?;
    let slow_logits = logits.flatten_all()?;

    let semantic_token = if model.semantic_end_id.is_none() {
        let pad_prob = slow_logits
            .i(pad_id as usize)?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?;
        let eos_prob = slow_logits
            .i(im_end_id as usize)?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?;

        // Ah the halcyon days where we're not forced to do a giant softmax to double up the first semantic codes
        legacy_softmax_sample(pad_prob, eos_prob, pad_id, im_end_id)
    } else {
        // TODO DO NOT MERGE: add the split again. I can't be bothered
        fast_logits_processor.sample(&slow_logits)?
    };
    let mut codebooks = vec![semantic_token];
    model.clear_fast_layer_caches();

    let mut x = hidden_states;
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
    Ok((codebooks, codes_tensor))
}

/// Takes a conditioning sequence as input and generates as many tokens as requested
pub fn generate(
    model: &mut DualARTransformer,
    prompt: &Tensor,
    max_new_tokens: usize,
    im_end_id: u32,
    pad_id: u32,
    sampling_args: &SamplingArgs,
) -> Result<Tensor> {
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
    let mut input_pos = model.curr_kv_size()?;
    let prompt_size = prompt.dim(D::Minus1)?;
    let (mut previous_token, mut cur_token) = decode_one_token_ar(
        model,
        &mut fast_logits_processor,
        prompt,
        input_pos,
        im_end_id,
        pad_id,
        None,
        &mut fast_rep_pens,
    )?;
    let dt = start_pp.elapsed();
    input_pos += prompt.dim(D::Minus1)?;
    println!(
        "{:.2}s prompt processing: {} tokens ({} new, {} cached, {:.2} tokens/s)",
        dt.as_secs_f64(),
        input_pos,
        prompt_size,
        input_pos - prompt_size,
        prompt_size as f64 / dt.as_secs_f64()
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

    let start_decode = Instant::now();
    for i in 1..max_new_tokens {
        let (next_indices, next_token) = decode_one_token_ar(
            model,
            &mut fast_logits_processor,
            &cur_token,
            input_pos,
            im_end_id,
            pad_id,
            Some(previous_token),
            &mut fast_rep_pens,
        )?;
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
    println!(
        "{} tokens generated ({:.2} tokens/s, {:.3}ms / token, RTF: {:.3})",
        out_len,
        out_len / dt.as_secs_f64(),
        (dt.as_secs_f64() * 1e3) / (out_len - 1f64),
        (out_len / 21.535) / dt.as_secs_f64()
    );
    previous_tokens.i((1.., ..))
}
