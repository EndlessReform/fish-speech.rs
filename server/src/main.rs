use axum::{body::Body, http::StatusCode, response::Response};
use axum::{extract::State, routing::post, Json, Router};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use clap::Parser;
use fish_speech_core::audio::wav::write_pcm_as_wav;
use fish_speech_core::models::{
    text2semantic::{
        utils::{
            encode::encode_tokens_batch,
            generate::generate,
            sample::SamplingArgs,
            text::{preprocess_text, TextChunk},
        },
        BaseModelArgs, DualARTransformer,
    },
    vqgan::{config::FireflyConfig, config::WhichModel, decoder::FireflyDecoder},
};
use serde::{Deserialize, Serialize};
use server::load_speaker_prompts;
use server::opus::OpusEncoder;
use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
// Re-export the key types
pub use bytes::Bytes;
pub use futures_util::Stream;

/// Shared state between requests
pub struct AppState {
    semantic_model: Arc<Mutex<DualARTransformer>>,
    vocoder_model: Arc<Mutex<FireflyDecoder>>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    voices: Arc<HashMap<String, Tensor>>,
    default_voice: Arc<Tensor>,
    temp: f64,
    top_p: f64,
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

    /// Temperature for sampling (higher = more random)
    #[arg(long, default_value = "0.7")]
    temp: f64,

    /// Top-p (nucleus) sampling threshold
    #[arg(long, default_value = "0.8")]
    top_p: f64,
}

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    model: String, // Ignored for now
    voice: String,
    input: String,
    response_format: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    audio: Vec<f32>, // Or whatever your audio format is
}

async fn generate_pcm_chunk(
    state: &Arc<AppState>,
    chunk: &TextChunk,
    voice_embedding: &Tensor,
    num_codebooks: usize,
) -> Result<Vec<f32>, StatusCode> {
    let encoded_input = encode_tokens_batch(
        &state.tokenizer,
        vec![chunk.clone()],
        &state.device,
        None,
        num_codebooks,
    )
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    .pop()
    .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;

    let final_prompt = Tensor::cat(&[voice_embedding, &encoded_input], D::Minus1)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let sampling_args = SamplingArgs {
        temp: state.temp,
        top_p: state.top_p,
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
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

type BoxError = Box<dyn Error + Send + Sync>; // This is what axum expects!

async fn generate_speech(
    State(state): State<Arc<AppState>>,
    Json(request): Json<GenerateRequest>,
) -> Result<Response<Body>, StatusCode> {
    let voice_embedding = state
        .voices
        .get(&request.voice)
        .unwrap_or(&state.default_voice);

    let num_codebooks = state.semantic_model.lock().await.cfg.num_codebooks;
    let chunks = preprocess_text(&request.input);

    if request.response_format == Some("opus".into()) {
        // Clone everything we need for the stream
        let state = state.clone();
        let voice_embedding = voice_embedding.clone();
        let num_codebooks = num_codebooks;

        // Create a SINGLE encoder outside the stream
        let encoder = OpusEncoder::new().map_err(|e| {
            tracing::error!("Failed to create Opus encoder: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
        let encoder = Arc::new(Mutex::new(encoder));
        let encoder_clone = encoder.clone();

        let stream = async_stream::stream! {
            for chunk in chunks.iter() {
                match generate_pcm_chunk(&state, chunk, &voice_embedding, num_codebooks).await {
                    Ok(pcm_data) => {
                        // Use the shared encoder instead of creating a new OpusStream
                        let src_rate = 44100.0;
                        let dst_rate = 24000.0;
                        let ratio = src_rate / dst_rate;

                        let resampled_pcm: Vec<f32> = (0..((pcm_data.len() as f32 / ratio) as usize))
                            .map(|i| {
                                let src_idx = i as f32 * ratio;
                                let src_idx_floor = src_idx.floor() as usize;
                                let src_idx_ceil = src_idx.ceil() as usize;
                                if src_idx_ceil >= pcm_data.len() {
                                    pcm_data[src_idx_floor]
                                } else {
                                    let t = src_idx - src_idx_floor as f32;
                                    pcm_data[src_idx_floor] * (1.0 - t) + pcm_data[src_idx_ceil] * t
                                }
                            })
                            .collect();

                        let mut encoder = encoder_clone.lock().await;
                        match encoder.encode_pcm(&resampled_pcm) {
                            Ok(encoded) => {
                                // Split encoded data into chunks if needed
                                for chunk in encoded.chunks(1024) {
                                    yield Ok(Bytes::copy_from_slice(chunk));
                                }
                            }
                            Err(e) => yield Err(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
                        }
                    }
                    Err(e) => yield Err(std::io::Error::new(std::io::ErrorKind::Other, "PCM generation failed"))
                }
            }
        };

        Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "audio/ogg")
            .header("Transfer-Encoding", "chunked")
            .body(Body::from_stream(stream))
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
    } else {
        let mut all_pcm = Vec::new();
        for chunk in chunks.iter() {
            let pcm = generate_pcm_chunk(&state, chunk, voice_embedding, num_codebooks).await?;
            all_pcm.extend(pcm);
        }

        let mut audio_buf = Vec::new();
        write_pcm_as_wav(&mut audio_buf, &all_pcm, 44100)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "audio/wav")
            .body(Body::from(audio_buf))
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
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
        temp: args.temp,
        top_p: args.top_p,
    });

    // Create router
    let app = Router::new()
        .route("/v1/audio/speech", post(generate_speech))
        .with_state(state);

    // Run server
    let addr = format!("0.0.0.0:{}", args.port);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();

    Ok(())
}
