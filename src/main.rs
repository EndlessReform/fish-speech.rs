use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use clap::{Parser, ValueHint};
use fish_speech_lib::audio::{load_audio, AudioConfig};
use fish_speech_lib::models::vqgan::FireflyArchitecture;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input source audio file path
    #[arg(short = 'i', long = "input", value_hint = ValueHint::FilePath)]
    src_audio: String,

    /// Path to the model checkpoint
    #[arg(long, value_name = "CHECKPOINT_PATH", value_hint = ValueHint::FilePath)]
    checkpoint_path: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Input audio: {}", args.src_audio);
    println!("Checkpoint path: {}", args.checkpoint_path);

    let device = Device::Cpu;
    let audio = load_audio(
        args.src_audio,
        &AudioConfig {
            ..Default::default()
        },
        &device,
    )?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[args.checkpoint_path], DType::F32, &device)?
    };

    let encoder = FireflyArchitecture::load(vb)?;
    let result = encoder.encode(&audio)?;
    println!("{:?}", result);

    Ok(())
}
