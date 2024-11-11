use anyhow::Error;
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::VarBuilder;
use clap::Parser;
use fish_speech_core::models::text2semantic::utils::{
    encode::encode_tokens,
    generate::generate,
    sample::{load_prompt_texts, SamplingArgs},
};
use fish_speech_core::models::text2semantic::{BaseModelArgs, DualARTransformer};
use fish_speech_core::models::vqgan::config::WhichModel;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

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
