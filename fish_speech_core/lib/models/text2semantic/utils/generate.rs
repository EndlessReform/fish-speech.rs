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
) -> Result<(Vec<u32>, Tensor, Tensor)> {
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
        // TODO DO NOT MERGE! Hard-coded hack
        // let im_end_id = 49152;
        // Assumes im_end_id will always come after start. Since it's just 1.5 this is fine
        let special_token_range = slow_logits.i(im_end_id as usize..)?.contiguous()?;
        let shifted_token = fast_logits_processor.sample(&special_token_range)?;
        shifted_token + im_end_id
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
    Ok((codebooks, codes_tensor, x))
}

/// Takes a conditioning sequence as input and generates as many tokens as requested
pub fn generate_blocking_with_hidden(
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
    let mut fast_logits_processor = LogitsProcessor::from_sampling(rand::random::<u64>(), sampling);
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
    let (mut previous_token, mut cur_token, hidden_state_first) = decode_one_token_ar(
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
        "{:.2}ms prompt processing: {} tokens ({} new, {} cached, {:.2} tokens/s)",
        dt.as_secs_f64() * 1000.0,
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
        "{} tokens generated in {:.3}s ({:.2} tokens/s, {:.3}ms / token, RTF: {:.3})",
        out_len,
        dt.as_secs_f64(),
        out_len / dt.as_secs_f64(),
        (dt.as_secs_f64() * 1e3) / (out_len - 1f64),
        (out_len / 21.535) / dt.as_secs_f64()
    );
    Ok((previous_tokens.i((1.., ..))?, hidden_states))
}

pub fn generate_blocking(
    model: &mut DualARTransformer,
    prompt: &Tensor,
    max_new_tokens: usize,
    im_end_id: u32,
    pad_id: u32,
    sampling_args: &SamplingArgs,
) -> Result<Tensor> {
    let (out, _) = generate_blocking_with_hidden(
        model,
        prompt,
        max_new_tokens,
        im_end_id,
        pad_id,
        sampling_args,
        false,
    )?;
    Ok(out)
}
