use axum::{extract::State, Json};
use candle_core::{DType, Device};
use clap::Parser;
use server::handlers::speech::{generate_speech, GenerateRequest};
use server::state::AppState;
use server::utils::load::{load_codec, load_lm, Args};
use std::sync::Arc;
use std::time::Instant;

async fn run_speech_test() -> anyhow::Result<()> {
    let args = Args::parse();

    #[cfg(feature = "cuda")]
    let device = Device::cuda_if_available(0)?;
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = Device::Cpu;

    let checkpoint_dir = match args.checkpoint.as_ref() {
        Some(dir) => Some(dir.canonicalize().unwrap()),
        _ => None,
    };

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

    // First request: batch_size = 1
    let request1 = GenerateRequest {
        model: "fish".to_string(),
        voice: "default".to_string(), // adjust as needed
        input: "The quick brown fox jumped over the lazy dog".to_string(),
        response_format: Some("wav".to_string()),
        batch_size: None,
        speaker_prompt: None,
    };

    println!("Creating first request with batch_size=1");
    let start = Instant::now();

    println!("Running batch_size=1 request");
    let _response1 = generate_speech(State(state.clone()), Json(request1)).await;
    println!(
        "Native single batch completed in {:.2}s",
        start.elapsed().as_secs_f64()
    );
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    //
    // First request: batch_size = 1
    let request1_batched = GenerateRequest {
        model: "fish".to_string(),
        voice: "default".to_string(), // adjust as needed
        input: "The quick brown fox jumped over the lazy dog".to_string(),
        response_format: Some("wav".to_string()),
        batch_size: Some(1),
        speaker_prompt: None,
    };

    println!("Creating first request with batch_size=1");
    let start = Instant::now();

    println!("Running batch_size=1 request");
    let _response1_batched = generate_speech(State(state.clone()), Json(request1_batched)).await;
    println!(
        "Batch size 1 completed in {:.2}s",
        start.elapsed().as_secs_f64()
    );
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Second request: batch_size = 4, repeated text
    let long_text = "The quick brown fox jumped over the lazy dog. ".repeat(8);
    let request2 = GenerateRequest {
        model: "fish".to_string(),
        voice: "default".to_string(),
        input: long_text,
        response_format: Some("wav".to_string()),
        batch_size: Some(4),
        speaker_prompt: None,
    };

    println!("Creating second request with batch_size=4");
    let start = Instant::now();

    println!("Running batch_size=4 request");
    let _response2 = generate_speech(State(state), Json(request2)).await;
    println!(
        "Batch size 4 completed in {:.2}s",
        start.elapsed().as_secs_f64()
    );

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_speech_test())
}
