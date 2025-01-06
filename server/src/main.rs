use axum::{
    extract::DefaultBodyLimit,
    routing::{get, post},
    Router,
};
pub use bytes::Bytes;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use clap::Parser;
use fish_speech_core::{
    audio::spectrogram::{LogMelSpectrogram, LogMelSpectrogramConfig},
    config::{WhichCodec, WhichFishVersion, WhichLM, WhichModel},
    models::{
        lm::{dual_ar::BaseModelArgs, dual_ar::TokenConfig, DualARTransformer},
        vqgan::{config::FireflyConfig, decoder::FireflyDecoder, encoder::FireflyEncoder},
    },
};
pub use futures_util::Stream;
use hf_hub::api::sync::Api;
use server::audio::{
    codec::{Codec, HiFiGANState},
    mimi,
};
use server::handlers::{
    encode_speech::encode_speaker, speech::generate_speech, supported_voices::get_supported_voices,
};
use server::load_speaker_prompts;
use server::state::{AppState, LMState};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
// Re-export the key types
use tower_http::cors::{Any, CorsLayer};

#[derive(Parser)]
struct Args {
    /// Checkpoint file path (default: "checkpoints/fish-1.5", canonicalized)
    #[arg(long, default_value = "checkpoints/fish-speech-1.5")]
    checkpoint: PathBuf,

    #[arg(short, long, default_value = "1.5")]
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

fn load_lm(
    args: &Args,
    checkpoint_dir: PathBuf,
    dtype: DType,
    device: &Device,
) -> anyhow::Result<LMState> {
    let semantic_config = BaseModelArgs::from_json_file(checkpoint_dir.join("config.json"))?;
    let tokenizer = Arc::new(Tokenizer::from_file(checkpoint_dir.join("tokenizer.json")).unwrap());
    let lm_version = WhichLM::from_model(args.fish_version);
    let vb_lm = match lm_version {
        WhichLM::Fish(WhichFishVersion::Fish1_2) => {
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

    Ok(LMState {
        model: semantic_model,
        model_type: lm_version,
        config: Arc::new(semantic_config),
        tokenizer,
        voices: Arc::new(Mutex::new(speakers)),
        default_voice: Arc::new(default_speaker),
        default_temp: args.temp,
        default_top_p: args.top_p,
        // TODO Totally arbitrary value, make this configurable from CLI
        max_new_tokens: 1024,
    })
}

/// (codec, sample_rate)
fn load_codec(
    args: &Args,
    dtype: DType,
    device: &Device,
    num_codebooks: usize,
) -> anyhow::Result<(Codec, u32)> {
    let codec_type = WhichCodec::from_model(args.fish_version.clone());
    match codec_type {
        WhichCodec::Fish(version) => {
            let vb_path = match version {
                WhichFishVersion::Fish1_2 => args
                    .checkpoint
                    .join("firefly-gan-vq-fsq-4x1024-42hz-generator-merged.pth"),
                _ => args
                    .checkpoint
                    .join("firefly-gan-vq-fsq-8x1024-21hz-generator.safetensors"),
            };
            let vb_decoder = match version {
                WhichFishVersion::Fish1_2 => VarBuilder::from_pth(vb_path.clone(), dtype, &device)?,
                _ => unsafe {
                    VarBuilder::from_mmaped_safetensors(&[vb_path.clone()], dtype, &device)?
                },
            };
            let vb_encoder = match version {
                WhichFishVersion::Fish1_2 => VarBuilder::from_pth(vb_path, DType::F32, &device)?,
                _ => unsafe {
                    VarBuilder::from_mmaped_safetensors(&[vb_path], DType::F32, &device)?
                },
            };
            let firefly_config = match args.fish_version {
                WhichModel::Fish1_2 => FireflyConfig::fish_speech_1_2(),
                _ => FireflyConfig::fish_speech_1_4(),
            };

            let vocoder_model = Arc::new(FireflyDecoder::load(
                &vb_decoder.clone(),
                &firefly_config,
                &version,
            )?);
            let encoder_model = Arc::new(FireflyEncoder::load(
                vb_encoder.clone(),
                &firefly_config,
                &version,
            )?);
            let spec_transform =
                Arc::new(LogMelSpectrogram::load(LogMelSpectrogramConfig::default())?);
            let sample_rate = spec_transform.sample_rate as u32;
            Ok((
                Codec::HiFiGAN(HiFiGANState {
                    decoder_model: vocoder_model,
                    encoder_model,
                    firefly_config: Arc::new(firefly_config),
                    spec_transform,
                }),
                sample_rate,
            ))
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
    // TODO: Figure out why BF16 is breaking on Metal
    #[cfg(feature = "cuda")]
    let dtype = DType::BF16;
    #[cfg(not(feature = "cuda"))]
    let dtype = DType::F32;

    println!("Loading {:?} model on {:?}", args.fish_version, device);
    let start_load = Instant::now();
    let lm_state = load_lm(&args, checkpoint_dir, dtype, &device)?;
    let (codec_state, sample_rate) =
        load_codec(&args, dtype, &device, lm_state.config.num_codebooks)?;
    let dt = start_load.elapsed();
    println!("Models loaded in {:.2}s", dt.as_secs_f64());

    let state = Arc::new(AppState {
        lm: Arc::new(lm_state),
        codec: Arc::new(codec_state),
        device,
        model_type: args.fish_version,
        sample_rate,
    });

    // Create router
    let app = Router::new()
        .route("/v1/audio/speech", post(generate_speech))
        .route("/v1/audio/encoding", post(encode_speaker))
        .route("/v1/voices", get(get_supported_voices))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state);

    // Run server
    let addr = format!("0.0.0.0:{}", args.port);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).tcp_nodelay(true).await?;

    Ok(())
}
