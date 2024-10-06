use anyhow::Error;
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Module, VarBuilder};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use clap::Parser;
use fish_speech_core::models::text2semantic::utils::{encode::encode_tokens, RepPenProcessor};
use fish_speech_core::models::text2semantic::{BaseModelArgs, DualARTransformer};
use fish_speech_core::models::vqgan::config::WhichModel;
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

/// For debugging purposes
// fn print_logprobs(logits: &Tensor) -> Result<()> {
//     let mut lp: Vec<(usize, f32)> = softmax_last_dim(&logits.flatten_all()?)?
//         .to_vec1::<f32>()?
//         .iter()
//         .enumerate()
//         .map(|(idx, p)| (idx, *p))
//         .collect();
//     lp.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
//     lp.reverse();
//     // println!("Top logprobs: {:?}", lp.iter().take(10));
//     println!("Top logprobs: {:?}", &lp[..5]);
//     Ok(())
// }

/// Extremely stripped-down softmax for two tokens
fn softmax_sample(pad_prob: f32, eos_prob: f32, pad_id: u32, eos_id: u32) -> u32 {
    // Compute softmax
    let exp_pad = (pad_prob - pad_prob.max(eos_prob)).exp();
    let exp_eos = (eos_prob - pad_prob.max(eos_prob)).exp();
    let sum = exp_pad + exp_eos;

    let softmax_pad = exp_pad / sum;
    // let softmax_eos = exp_eos / sum;

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
fn generate(
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
    let (mut previous_token, mut cur_token) = decode_one_token_ar(
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
        (out_len / 43.07) / dt.as_secs_f64()
    );
    previous_tokens.i((1.., ..))
}

fn generate_long(
    model: &mut DualARTransformer,
    tokenizer: &Tokenizer,
    args: &Args,
    device: &Device,
) -> anyhow::Result<()> {
    let sampling_args = SamplingArgs {
        temp: args.temp,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: args.repetition_penalty,
    };

    let conditioning_prompts =
        load_prompt_texts(&args.prompt_tokens, args.prompt_text.clone(), device)?;

    let encoded_prompts: Result<Tensor> = conditioning_prompts
        .iter()
        .map(|(t, c)| encode_tokens(tokenizer, t, device, Some(c), model.cfg.num_codebooks))
        .try_fold(
            Tensor::from_slice(
                &(vec![] as Vec<u32>),
                (model.cfg.num_codebooks + 1, 0),
                &device,
            )?,
            |acc, e| e.and_then(|tensor| Tensor::cat(&[&acc, &tensor], D::Minus1)),
        );
    let encoded_prompts = encoded_prompts?;
    let encoded = vec![encode_tokens(
        &tokenizer,
        &args.text,
        device,
        None,
        model.cfg.num_codebooks,
    )?];
    // TODO: this is terrible; do more intelligent splitting as per upstream
    // let final_prompt = encoded_prompts.into_iter().unwrap();
    let final_prompt = Tensor::cat(&[encoded_prompts, encoded[0].clone()], D::Minus1)?;

    println!("Loaded prompt with shape {:?}", final_prompt.shape());
    let im_end_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(4);
    let pad_id = tokenizer.token_to_id("<|semantic|>").unwrap_or(5);

    let res = generate(
        model,
        &final_prompt,
        args.max_new_tokens,
        im_end_id,
        pad_id,
        &sampling_args,
    )?;
    model.clear_slow_layer_caches();
    let res = res.broadcast_sub(&Tensor::ones_like(&res)?)?;
    res.write_npy(&args.out_path)?;

    Ok(())
}

struct SamplingArgs {
    pub temp: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub repetition_penalty: f32,
}

fn load_prompt_texts(
    prompt_tokens: &[PathBuf],
    prompt_texts: Vec<String>,
    device: &Device,
) -> anyhow::Result<Vec<(String, Tensor)>> {
    if prompt_tokens.len() != prompt_texts.len() {
        Err(Error::msg(format!(
            "Prompt token length {:?} does not match prompt text length {:?}",
            prompt_tokens.len(),
            prompt_texts.len()
        )))?
    }

    let codes: Result<Vec<Tensor>> = prompt_tokens.iter().map(Tensor::read_npy).collect();
    let codes: Result<Vec<Tensor>> = codes?.into_iter().map(|c| c.to_device(device)).collect();

    Ok(prompt_texts.into_iter().zip(codes?).collect())
}

#[derive(Parser, Debug)]
#[command(
    author = "Jacob Keisling <jacob@keisling.me>",
    version = "0.1",
    about = "Generates codebooks for Fish Speech"
)]
struct Args {
    /// Temperature for sampling
    #[arg(long, default_value_t = 0.7)]
    temp: f64,

    /// Top-p sampling parameter
    #[arg(long, default_value_t = 0.7)]
    top_p: f64,

    /// Top-k sampling parameter. Set high by default
    #[arg(long, default_value_t = 256)]
    top_k: usize,

    /// Penalty for repetition
    #[arg(long, default_value_t = 1.2)]
    repetition_penalty: f32,

    /// Text to process (required)
    #[arg(long)]
    text: String,

    /// Output file path
    #[arg(short, long, default_value_t = 1024)]
    max_new_tokens: usize,

    /// Output file path
    #[arg(short, long, default_value = "out.npy")]
    out_path: PathBuf,

    /// Checkpoint file path (default: "checkpoints/fish-1.4", canonicalized)
    #[arg(long, default_value = "checkpoints/fish-speech-1.4")]
    checkpoint: PathBuf,

    /// Optional multiple prompt token files
    #[arg(long, num_args=1..)]
    prompt_tokens: Vec<PathBuf>,

    /// Optional multiple prompt text strings
    #[arg(long, num_args=1..)]
    prompt_text: Vec<String>,

    #[arg(short, long, default_value = "1.4")]
    fish_version: WhichModel,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    // TODO: Control this by feature flag
    #[cfg(feature = "cuda")]
    let device = Device::cuda_if_available(0)?;

    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;

    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = Device::Cpu;

    let checkpoint_dir = args.checkpoint.canonicalize().map_err(|_| {
        Error::msg(format!(
            "Could not find checkpoint path {:?} relative to current directory",
            args.checkpoint
        ))
    })?;
    let config = BaseModelArgs::from_json_file(checkpoint_dir.join("config.json"))?;
    let tokenizer = Tokenizer::from_file(checkpoint_dir.join("tokenizer.json")).unwrap();
    // TODO: Figure out why BF16 is breaking on Metal
    #[cfg(feature = "cuda")]
    let dtype = DType::BF16;
    #[cfg(not(feature = "cuda"))]
    let dtype = DType::F32;

    let vb = match args.fish_version {
        WhichModel::Fish1_4 => unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[checkpoint_dir.join("model.safetensors")],
                dtype,
                &device,
            )?
        },
        _ => VarBuilder::from_pth(checkpoint_dir.join("model.pth"), dtype, &device)?,
    };
    let semantic_token_id = tokenizer.token_to_id("<|semantic|>").unwrap_or(5);
    let mut model = DualARTransformer::load(&vb, &config, semantic_token_id as i64).unwrap();
    println!("Model loaded to {:?}", device);
    generate_long(&mut model, &tokenizer, &args, &device)?;

    Ok(())
}
