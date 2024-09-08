use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::{Parser, ValueHint};
use fish_speech_lib::audio as torchaudio;
use fish_speech_lib::models::vqgan::FireflyArchitecture;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input source audio file path
    #[arg(short = 'i', long = "input", value_hint = ValueHint::FilePath)]
    src_audio: String,

    /// Output audio file path
    #[arg(short = 'o', long = "output-path", value_hint = ValueHint::FilePath)]
    dest_audio: String,

    /// Path to the model checkpoint
    #[arg(long, value_name = "CHECKPOINT_PATH", value_hint = ValueHint::FilePath)]
    checkpoint_path: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Processing in-place reconstruction of {}", args.src_audio);
    println!(
        "Warning: Loading precomputed audio for debugging. Please don't use this for production"
    );

    let device = Device::Cpu;
    let (mut audio, sr) = torchaudio::load(args.src_audio, &device)?;
    if audio.dim(0)? > 1 {
        audio = audio.mean_keepdim(0)?;
    }
    // TODO: Stop hard-coding sample rate
    const SAMPLE_RATE: u32 = 44100;
    // Add spurious batch dimension for consistency
    audio = torchaudio::functional::resample(&audio, sr, SAMPLE_RATE)?.unsqueeze(0)?;
    println!("Audio shape: {:?}", audio.shape());
    println!(
        "Loaded audio with {} seconds",
        audio.shape().dims3()?.2 as f64 / (SAMPLE_RATE as f64)
    );
    // audio.write_npy("./tests/resources/candle_fish_preprocessed_audio.npy")?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[args.checkpoint_path], DType::F32, &device)?
    };

    // NOTE: Not using audio for now
    let override_mel = Tensor::read_npy("spec_transform_fish_c_order.npy")?;
    let encoder = FireflyArchitecture::load(vb)?;
    // Temporarily skipping our own preprocessing code and hard-coding to isolate numerical accuracy elsewhere
    let result = encoder.encode(&override_mel)?.squeeze(0)?;
    println!("Generated indices of shape {:?}", result.shape());
    result.write_npy(args.dest_audio)?;

    Ok(())
}
