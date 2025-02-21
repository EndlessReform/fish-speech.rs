use crate::audio::{codec::Codec, mimi};
use crate::state::LMState;
use crate::utils::load_speaker_prompts;
pub use bytes::Bytes;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use clap::Parser;
use fish_speech_core::codec::{FireflyCodec, FireflyConfig};
use fish_speech_core::{
    config::{WhichCodec, WhichFishVersion, WhichLM, WhichModel},
    lm::{
        dual_ar::{BaseModelArgs, TokenConfig},
        sampling::SamplingArgs,
        DualARTransformer,
    },
};
pub use futures_util::Stream;
use hf_hub::api::sync::{Api, ApiRepo};
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

#[derive(Parser)]
pub struct Args {
    /// Checkpoint file path (default: "checkpoints/fish-1.5", canonicalized)
    #[arg(long)]
    pub checkpoint: Option<PathBuf>,

    #[arg(short, long, default_value = "1.5")]
    pub fish_version: WhichModel,

    /// Directory containing voice embeddings
    #[arg(long, default_value = "voices")]
    pub voice_dir: PathBuf,

    /// Port to listen on
    #[arg(short, long, default_value = "3000")]
    pub port: u16,

    /// Temperature for sampling (higher = more random)
    #[arg(long, default_value = "0.7")]
    pub temp: f64,

    /// Top-p (nucleus) sampling threshold
    #[arg(long, default_value = "0.8")]
    pub top_p: f64,
}

pub fn get_model_repo(model_type: WhichModel) -> anyhow::Result<ApiRepo> {
    // PathBuf::from("checkpoints/fish-1.5")
    let api = Api::new()?;
    let repo_name = match model_type {
        WhichModel::Fish1_5 => "jkeisling/fish-speech-1.5",
        WhichModel::Fish1_4 => "jkeisling/fish-speech-1.4",
        WhichModel::Fish1_2 => "fishaudio/fish-speech-1.2-sft",
        WhichModel::DualAR => "jkeisling/smoltts_v0",
    };
    Ok(api.model(repo_name.to_owned()))
}

pub fn load_lm(
    args: &Args,
    checkpoint_dir: Option<PathBuf>,
    dtype: DType,
    device: &Device,
) -> anyhow::Result<LMState> {
    let repo = get_model_repo(args.fish_version)?;
    let config_path = match checkpoint_dir.as_ref() {
        Some(dir) => dir.join("config.json"),
        None => repo.get("config.json")?,
    };
    let weight_str = match args.fish_version {
        WhichModel::Fish1_2 => "model.pth",
        _ => "model.safetensors",
    };
    let weight_path = match checkpoint_dir.as_ref() {
        Some(dir) => dir.join(weight_str),
        None => repo.get(weight_str)?,
    };
    let tokenizer_path = match checkpoint_dir {
        Some(dir) => dir.join("tokenizer.json"),
        None => {
            let _special_tokens = repo.get("special_tokens_map.json")?;
            let _tokenizer_config = repo.get("tokenizer_config.json")?;
            repo.get("tokenizer.json")?
        }
    };

    let semantic_config = BaseModelArgs::from_file(config_path)?;
    let tokenizer = Arc::new(Tokenizer::from_file(tokenizer_path).unwrap());
    let lm_version = WhichLM::from_model(args.fish_version);
    let vb_lm = match lm_version {
        WhichLM::Fish(WhichFishVersion::Fish1_2) => {
            VarBuilder::from_pth(weight_path, dtype, &device)?
        }
        _ => unsafe { VarBuilder::from_mmaped_safetensors(&[weight_path], dtype, &device)? },
    };
    let lm_version = WhichLM::from_model(args.fish_version.clone());
    let semantic_token_config = TokenConfig::new(lm_version.clone(), &tokenizer, &semantic_config)?;
    let semantic_model = Arc::new(Mutex::new(DualARTransformer::load(
        &vb_lm,
        &semantic_config,
        &semantic_token_config,
        lm_version.clone(),
    )?));
    // Load all voices into memory
    let (speakers, default_speaker) = load_speaker_prompts(
        &args.voice_dir,
        &tokenizer,
        &device,
        semantic_config.num_codebooks,
        lm_version.clone(),
    )?;
    println!("Loaded {} voices", speakers.len());
    let default_sampling_args = SamplingArgs {
        temp: args.temp,
        top_p: args.top_p,
        // TODO make this configurable
        top_k: 256,
        repetition_penalty: match lm_version {
            WhichLM::DualAR | WhichLM::Fish(WhichFishVersion::Fish1_5) => 1.4,
            _ => 1.2,
        },
    };

    Ok(LMState {
        model: semantic_model,
        model_type: lm_version,
        config: Arc::new(semantic_config),
        tokenizer,
        voices: Arc::new(Mutex::new(speakers)),
        default_voice: Arc::new(default_speaker),
        default_sampling_args,
        // TODO Totally arbitrary value, make this configurable from CLI
        max_new_tokens: 1792,
    })
}

/// (codec, sample_rate)
pub fn load_codec(
    args: &Args,
    dtype: DType,
    device: &Device,
    num_codebooks: usize,
) -> anyhow::Result<(Codec, u32)> {
    let repo = get_model_repo(args.fish_version)?;

    let codec_type = WhichCodec::from_model(args.fish_version.clone());
    match codec_type {
        WhichCodec::Fish(version) => {
            let weight_name = match args.fish_version {
                WhichModel::Fish1_2 => "firefly-gan-vq-fsq-4x1024-42hz-generator-merged.pth",
                _ => "firefly-gan-vq-fsq-8x1024-21hz-generator.safetensors",
            };
            let vb_path = match args.checkpoint.as_ref() {
                Some(dir) => dir.join(weight_name),
                None => repo.get(weight_name)?,
            };
            let vb = match version {
                WhichFishVersion::Fish1_2 => VarBuilder::from_pth(vb_path, DType::F32, &device)?,
                _ => unsafe {
                    VarBuilder::from_mmaped_safetensors(&[vb_path], DType::F32, &device)?
                },
            };
            let firefly_config = FireflyConfig::get_config_for(version);

            let firefly_codec = Arc::new(FireflyCodec::load(
                firefly_config.clone(),
                vb.clone(),
                version,
            )?);
            let sample_rate = firefly_codec.sample_rate;
            Ok((Codec::Firefly(firefly_codec), sample_rate))
        }
        WhichCodec::Mimi => {
            let api = Api::new()?;
            // Yes, this is terrible, but it's literally what their MLX client does
            // TODO make this configurable, I don't really care right now
            let repo = api.model("kyutai/moshiko-mlx-bf16".to_string());
            let mimi_path = repo.get("tokenizer-e351c8d8-checkpoint125.safetensors")?;
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[mimi_path], dtype, device) }?;
            // Yes, this is hard-coded. If this ever changes I will care
            let (model, sr) = mimi::Tokenizer::load(vb, num_codebooks)?;
            Ok((Codec::Mimi(Arc::new(Mutex::new(model))), sr))
        }
    }
}
