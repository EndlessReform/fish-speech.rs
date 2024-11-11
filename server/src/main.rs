use anyhow;
use axum::{extract::State, routing::post, Json, Router};
use candle_core::{DType, Device, Error, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use fish_speech_core::models::{
    text2semantic::{BaseModelArgs, DualARTransformer},
    vqgan::{config::FireflyConfig, config::WhichModel, decoder::FireflyDecoder},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

// Load all voices from a directory
fn load_voices(
    voice_dir: &PathBuf,
    device: &Device,
) -> anyhow::Result<(HashMap<String, Tensor>, Tensor)> {
    let mut voices = HashMap::new();
    let mut default_voice = None;

    for entry in std::fs::read_dir(voice_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "npy") {
            let voice_name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| anyhow::anyhow!("Invalid voice filename"))?
                .to_string();

            let tensor = Tensor::read_npy(&path)?.to_device(device)?;

            if voice_name == "default" {
                default_voice = Some(tensor.clone());
            }

            voices.insert(voice_name, tensor);
        }
    }

    let default_voice = default_voice
        .ok_or_else(|| anyhow::anyhow!("No default.npy voice found in voices directory"))?;

    Ok((voices, default_voice))
}

#[derive(Parser, Debug)]
struct Args {
    /// Checkpoint file path (default: "checkpoints/fish-1.4", canonicalized)
    #[arg(long, default_value = "checkpoints/fish-speech-1.4")]
    checkpoint: PathBuf,

    #[arg(short, long, default_value = "1.4")]
    fish_version: WhichModel,

    /// Directory containing voice embeddings
    #[arg(long, default_value = "voices")]
    voice_dir: PathBuf,

    /// Port to listen on
    #[arg(short, long, default_value = "3000")]
    port: u16,
}

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    model: String, // Ignored for now
    voice: String,
    input: String,
}

#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    audio: Vec<f32>, // Or whatever your audio format is
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    #[cfg(feature = "cuda")]
    let device = Device::cuda_if_available(0)?;

    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;

    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = Device::Cpu;

    let checkpoint_dir = args.checkpoint.canonicalize().unwrap();
    let config = BaseModelArgs::from_json_file(checkpoint_dir.join("config.json"))?;
    let tokenizer = Tokenizer::from_file(checkpoint_dir.join("tokenizer.json")).unwrap();
    // TODO: Figure out why BF16 is breaking on Metal
    #[cfg(feature = "cuda")]
    let dtype = DType::BF16;
    #[cfg(not(feature = "cuda"))]
    let dtype = DType::F32;

    println!("Loading {:?} model on {:?}", args.fish_version, device);
    let start_load = Instant::now();
    let vb_lm = match args.fish_version {
        WhichModel::Fish1_4 => unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[checkpoint_dir.join("model.safetensors")],
                dtype,
                &device,
            )?
        },
        _ => VarBuilder::from_pth(checkpoint_dir.join("model.pth"), dtype, &device)?,
    };
    let vb_vocoder = match args.fish_version {
        WhichModel::Fish1_4 => unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[args
                    .checkpoint
                    .join("firefly-gan-vq-fsq-8x1024-21hz-generator.safetensors")],
                dtype,
                &device,
            )?
        },
        WhichModel::Fish1_2 => VarBuilder::from_pth(
            args.checkpoint
                .join("firefly-gan-vq-fsq-4x1024-42hz-generator-merged.pth"),
            dtype,
            &device,
        )?,
    };
    let vocoder_config = match args.fish_version {
        WhichModel::Fish1_2 => FireflyConfig::fish_speech_1_2(),
        _ => FireflyConfig::fish_speech_1_4(),
    };

    let semantic_token_id = tokenizer.token_to_id("<|semantic|>").unwrap_or(5);
    let semantic_model = Arc::new(Mutex::new(DualARTransformer::load(
        &vb_lm,
        &config,
        semantic_token_id as i64,
    )?));
    let vocoder_model = Arc::new(Mutex::new(FireflyDecoder::load(
        &vb_vocoder,
        &vocoder_config,
        &args.fish_version,
    )?));
    let dt = start_load.elapsed();
    println!("Models loaded in {:.2}s", dt.as_secs_f64());

    Ok(())
}
