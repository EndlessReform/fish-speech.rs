use axum::{body::Body, http::StatusCode, response::Response};
use axum::{extract::State, routing::post, Json, Router};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use clap::Parser;
use fish_speech_core::audio::wav::write_pcm_as_wav;
use fish_speech_core::models::{
    text2semantic::{
        utils::{encode::encode_tokens, generate::generate, sample::SamplingArgs},
        BaseModelArgs, DualARTransformer,
    },
    vqgan::{config::FireflyConfig, config::WhichModel, decoder::FireflyDecoder},
};
use serde::{Deserialize, Serialize};
use server::load_speaker_prompts;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

/// Shared state between requests
pub struct AppState {
    semantic_model: Arc<Mutex<DualARTransformer>>,
    vocoder_model: Arc<Mutex<FireflyDecoder>>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    voices: Arc<HashMap<String, Tensor>>,
    default_voice: Arc<Tensor>,
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

async fn generate_speech(
    State(state): State<Arc<AppState>>,
    Json(request): Json<GenerateRequest>,
) -> Result<Response<Body>, StatusCode> {
    let voice_embedding = state
        .voices
        .get(&request.voice)
        .unwrap_or(&state.default_voice);

    let num_codebooks = state.semantic_model.lock().await.cfg.num_codebooks;

    let encoded_input = encode_tokens(
        &state.tokenizer,
        &request.input,
        &state.device,
        None,
        num_codebooks,
    )
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let final_prompt = Tensor::cat(&[voice_embedding, &encoded_input], D::Minus1)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let sampling_args = SamplingArgs {
        temp: 0.7,
        top_p: 0.7,
        top_k: 256,
        repetition_penalty: 1.2,
    };

    let semantic_tokens = {
        let mut model = state.semantic_model.lock().await;
        let tokens = generate(
            &mut model,
            &final_prompt,
            1024,
            state.tokenizer.token_to_id("<|im_end|>").unwrap_or(4),
            state.tokenizer.token_to_id("<|semantic|>").unwrap_or(5),
            &sampling_args,
        )
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        model.clear_slow_layer_caches();
        tokens
            .broadcast_sub(
                &Tensor::ones_like(&tokens).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
            )
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    };

    let pcm: Vec<f32> = {
        let vocoder = state.vocoder_model.lock().await;
        let feature_lengths = Tensor::from_slice(
            &[semantic_tokens
                .dim(D::Minus1)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)? as u32],
            1,
            &state.device,
        )
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        vocoder
            .decode(
                &semantic_tokens
                    .unsqueeze(0)
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
                &feature_lengths,
            )
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .to_dtype(DType::F32)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .squeeze(0)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .squeeze(0)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .to_vec1()
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    };

    let mut audio_buf = Vec::new();
    write_pcm_as_wav(&mut audio_buf, &pcm, 44100).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "audio/wav")
        .body(Body::from(audio_buf))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
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
    // Load all voices into memory
    let (speakers, default_speaker) =
        load_speaker_prompts(&args.voice_dir, &tokenizer, &device, config.num_codebooks)?;
    println!("Loaded {} voices", speakers.len());

    let state = Arc::new(AppState {
        semantic_model,
        vocoder_model,
        tokenizer,
        device,
        voices: Arc::new(speakers),
        default_voice: Arc::new(default_speaker),
    });

    // Create router
    let app = Router::new()
        .route("/v1/audio/speech", post(generate_speech))
        .with_state(state);

    // Run server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:5000").await.unwrap();
    axum::serve(listener, app).await.unwrap();

    Ok(())
}
