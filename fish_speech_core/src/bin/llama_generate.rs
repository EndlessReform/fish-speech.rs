use anyhow::Error;
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::VarBuilder;
use clap::Parser;
use fish_speech_core::models::text2semantic::utils::{
    encode::PromptEncoder,
    generate::generate,
    sample::{load_prompt_text, SamplingArgs},
};
use fish_speech_core::models::text2semantic::{BaseModelArgs, DualARTransformer};
use fish_speech_core::models::vqgan::config::WhichModel;
use std::path::PathBuf;
use tokenizers::Tokenizer;

fn generate_long(
    model: &mut DualARTransformer,
    tokenizer: &Tokenizer,
    args: &Args,
    device: &Device,
    model_type: WhichModel,
) -> anyhow::Result<()> {
    let sampling_args = SamplingArgs {
        temp: args.temp,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: args.repetition_penalty,
    };

    if &args.prompt_tokens.len() != &args.prompt_text.len() {
        Err(anyhow::anyhow!(
            "Prompt token length {:?} does not match prompt text length {:?}",
            &args.prompt_tokens.len(),
            &args.prompt_text.len()
        ))?
    }

    let conditioning_tensors: Result<Vec<Tensor>> = args
        .prompt_tokens
        .iter()
        .map(|path| load_prompt_text(path, device, model.cfg.num_codebooks))
        .collect();
    let prompt_encoder = PromptEncoder::new(tokenizer, device, model.cfg.num_codebooks, model_type);

    let conditioning_prompts: Result<Vec<Tensor>> = args
        .prompt_text
        .iter()
        .zip(conditioning_tensors?.iter())
        .map(|(t, c)| prompt_encoder.encode_conditioning_prompt(t, c))
        .collect();
    let final_conditioning = match model_type {
        WhichModel::Fish1_5 => {
            // The upstream hard-codes a system prompt:
            // https://github.com/fishaudio/fish-speech/blame/b11bcf834a97a75073b535838e5f0765f169eb94/tools/llama/generate.py#L756
            // I'm also hard-coding for now to match upstream since I don't know if this actually affects output quality.
            // TODO: make this configurable, experiment!
            let system_prompt =
                prompt_encoder.encode_text("system", Some("Speak out the provided text"))?;

            // This is disgusting but whatever, it's not in the hot loop
            let mut all_tensors = vec![system_prompt];
            all_tensors.extend(conditioning_prompts?.iter().cloned());
            Tensor::cat(&all_tensors, D::Minus1)?
        }
        _ => Tensor::cat(&conditioning_prompts?, D::Minus1)?,
    };

    // TODO: use splitting code from the server
    let text_to_generate = prompt_encoder.encode_text("user", Some(&args.text))?;
    println!("Text: {:?}", &args.text);
    let assistant_preprompt = prompt_encoder.encode_vq(None)?;

    println!(
        "Speaker conditioning size: {:?}",
        final_conditioning.shape()
    );
    let final_prompt = Tensor::cat(
        &[final_conditioning, text_to_generate, assistant_preprompt],
        D::Minus1,
    )?;

    println!("Loaded prompt with shape {:?}", final_prompt.shape());
    let im_end_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(4);
    let pad_id = tokenizer.token_to_id("<|semantic|>").unwrap_or(5);
    let speaker_tokens = final_prompt
        .i((0, ..))?
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1::<u32>()?;
    println!("Input trokens:\n{:?}", &speaker_tokens);
    println!(
        "Input prompt:\n{}",
        tokenizer.decode(&speaker_tokens, false).unwrap()
    );

    let res = generate(
        model,
        &final_prompt,
        args.max_new_tokens,
        im_end_id,
        pad_id,
        &sampling_args,
    )?;
    model.clear_slow_layer_caches();
    let res = match model_type {
        WhichModel::Fish1_5 => res,
        _ => res.broadcast_sub(&Tensor::ones_like(&res)?)?,
    };
    // let res = res.broadcast_sub(&Tensor::ones_like(&res)?)?;
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

    /// Checkpoint file path (default: "checkpoints/fish-1.5", canonicalized)
    #[arg(long, default_value = "checkpoints/fish-speech-1.5")]
    checkpoint: PathBuf,

    /// Optional multiple prompt token files
    #[arg(long, num_args=1..)]
    prompt_tokens: Vec<PathBuf>,

    /// Optional multiple prompt text strings
    #[arg(long, num_args=1..)]
    prompt_text: Vec<String>,

    #[arg(short, long, default_value = "1.5")]
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
        WhichModel::Fish1_2 => {
            VarBuilder::from_pth(checkpoint_dir.join("model.pth"), dtype, &device)?
        }
        _ => unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[checkpoint_dir.join("model.safetensors")],
                dtype,
                &device,
            )?
        },
    };

    let semantic_start_id = match args.fish_version {
        WhichModel::Fish1_5 => tokenizer.token_to_id("<|semantic:0|>").unwrap_or(100012),
        _ => tokenizer.token_to_id("<|semantic|>").unwrap_or(5),
    } as i64;
    let semantic_end_id = match args.fish_version {
        WhichModel::Fish1_5 => tokenizer
            .token_to_id(&format!("<|semantic:{}|>", config.codebook_size - 1))
            .map(|id| id as i64),
        _ => None,
    };
    let mut model =
        DualARTransformer::load(&vb, &config, semantic_start_id, semantic_end_id).unwrap();
    println!("Model loaded to {:?}", device);
    generate_long(
        &mut model,
        &tokenizer,
        &args,
        &device,
        args.fish_version.clone(),
    )?;

    Ok(())
}
