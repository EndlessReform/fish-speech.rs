use axum::{extract::DefaultBodyLimit, routing::post, Router};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use clap::Parser;
use fish_speech_core::{
    audio::spectrogram::{LogMelSpectrogram, LogMelSpectrogramConfig},
    models::{
        text2semantic::{BaseModelArgs, DualARTransformer},
        vqgan::{
            config::{FireflyConfig, WhichModel},
            decoder::FireflyDecoder,
            encoder::FireflyEncoder,
        },
    },
};
use server::handlers::{encode_speech::encode_speaker, speech::generate_speech};
use server::load_speaker_prompts;
use server::state::AppState;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
// Re-export the key types
pub use bytes::Bytes;
pub use futures_util::Stream;

#[derive(Parser)]
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

    /// Temperature for sampling (higher = more random)
    #[arg(long, default_value = "0.7")]
    temp: f64,

    /// Top-p (nucleus) sampling threshold
    #[arg(long, default_value = "0.8")]
    top_p: f64,
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
    let semantic_config = BaseModelArgs::from_json_file(checkpoint_dir.join("config.json"))?;
    let tokenizer = Arc::new(Tokenizer::from_file(checkpoint_dir.join("tokenizer.json")).unwrap());
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
    let vb_firefly = match args.fish_version {
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
    let vb_encoder = match args.fish_version {
        WhichModel::Fish1_4 => unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[args
                    .checkpoint
                    .join("firefly-gan-vq-fsq-8x1024-21hz-generator.safetensors")],
                DType::F32,
                &device,
            )?
        },
        WhichModel::Fish1_2 => VarBuilder::from_pth(
            args.checkpoint
                .join("firefly-gan-vq-fsq-4x1024-42hz-generator-merged.pth"),
            DType::F32,
            &device,
        )?,
    };
    let firefly_config = match args.fish_version {
        WhichModel::Fish1_2 => FireflyConfig::fish_speech_1_2(),
        _ => FireflyConfig::fish_speech_1_4(),
    };

    let semantic_token_id = tokenizer.token_to_id("<|semantic|>").unwrap_or(5);
    let semantic_model = Arc::new(Mutex::new(DualARTransformer::load(
        &vb_lm,
        &semantic_config,
        semantic_token_id as i64,
    )?));
    let vocoder_model = Arc::new(FireflyDecoder::load(
        &vb_firefly,
        &firefly_config,
        &args.fish_version,
    )?);
    let encoder_model = Arc::new(FireflyEncoder::load(
        vb_encoder.clone(),
        &firefly_config,
        &args.fish_version,
    )?);
    let dt = start_load.elapsed();
    let spec_transform = Arc::new(LogMelSpectrogram::load(LogMelSpectrogramConfig::default())?);
    println!("Models loaded in {:.2}s", dt.as_secs_f64());
    // Load all voices into memory
    let (speakers, default_speaker) = load_speaker_prompts(
        &args.voice_dir,
        &tokenizer,
        &device,
        semantic_config.num_codebooks,
    )?;
    println!("Loaded {} voices", speakers.len());

    let state = Arc::new(AppState {
        semantic_model,
        vocoder_model,
        encoder_model,
        semantic_config: Arc::new(semantic_config),
        firefly_config: Arc::new(firefly_config),
        spec_transform,
        tokenizer,
        device,
        voices: Arc::new(speakers),
        default_voice: Arc::new(default_speaker),
        temp: args.temp,
        top_p: args.top_p,
    });

    // Create router
    let app = Router::new()
        .route("/v1/audio/speech", post(generate_speech))
        .route("/v1/audio/encoding", post(encode_speaker))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .with_state(state);

    // Run server
    let addr = format!("0.0.0.0:{}", args.port);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();

    Ok(())
}
