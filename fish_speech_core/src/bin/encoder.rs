use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use clap::{Parser, ValueHint};
use fish_speech_core::audio as torchaudio;
use fish_speech_core::models::vqgan::config::FireflyConfig;
use fish_speech_core::models::vqgan::encoder::FireflyEncoder;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input source audio file path
    #[arg(short = 'i', long = "input", value_hint = ValueHint::FilePath)]
    src_audio: PathBuf,

    /// Output audio file path
    #[arg(short = 'o', long = "output-path", value_hint = ValueHint::FilePath)]
    dest_audio: PathBuf,

    /// Path to the model checkpoint
    #[arg(long, value_name = "CHECKPOINT_PATH", value_hint = ValueHint::FilePath, default_value = "checkpoints/fish-speech-1.2-sft/firefly-gan-vq-fsq-4x1024-42hz-generator.pth")]
    checkpoint_path: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Processing in-place reconstruction of {:?}", args.src_audio);
    println!(
        "Warning: Loading precomputed audio for debugging. Please don't use this for production"
    );

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0)?;

    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    let device = Device::Cpu;

    println!("Using device {:?}", device);

    // CPU preprocessing for now
    let (mut audio, sr) = torchaudio::load(args.src_audio, &Device::Cpu)?;
    if audio.dim(0)? > 1 {
        audio = audio.mean_keepdim(0)?;
    }

    // TODO: Add Fish 1.4 support
    let config = FireflyConfig::fish_speech_1_2();
    // Add spurious batch dimension for consistency
    audio = torchaudio::functional::resample(&audio, sr, config.spec_transform.sample_rate as u32)?
        .unsqueeze(0)?;
    let audio_duration_sec =
        audio.shape().dims3()?.2 as f64 / (config.spec_transform.sample_rate as f64);
    println!("Encoding {:.2}s of audio", audio_duration_sec);

    let dtype = DType::F32;
    let vb = VarBuilder::from_pth(args.checkpoint_path, dtype, &device)?;

    // NOTE: Not using audio MEL conversion for now
    let override_mel = Tensor::read_npy("spec_transform_fish_c_order.npy")?.to_device(&device)?;
    let encoder = FireflyEncoder::load(
        vb,
        &config,
        &fish_speech_core::models::vqgan::config::WhichModel::Fish1_2,
    )?;
    println!("Model loaded");
    let start_decode = Instant::now();
    // Temporarily skipping our own preprocessing code and hard-coding to isolate numerical accuracy elsewhere
    let result = encoder.encode(&override_mel)?.squeeze(0)?;
    let dt = start_decode.elapsed();
    println!(
        "Processed audio in: {:.2}s (RTF: {:.3})",
        dt.as_secs_f64(),
        (result.dim(D::Minus1)? as f64 / 43.07) / dt.as_secs_f64()
    );
    println!("Generated indices of shape {:?}", result.shape());
    result.write_npy(args.dest_audio)?;

    Ok(())
}
