use anyhow::Result;
use candle_core::{DType, Device, D};
use candle_nn::VarBuilder;
use clap::{Parser, ValueHint};
use fish_speech_core::audio as torchaudio;
use fish_speech_core::codec::{FireflyCodec, FireflyConfig};
use fish_speech_core::config::{WhichCodec, WhichFishVersion, WhichModel};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input source audio file path
    #[arg(short = 'i', long = "input", value_hint = ValueHint::FilePath)]
    src_audio: PathBuf,

    /// Output audio file path
    #[arg(short = 'o', long = "output-path", value_hint = ValueHint::FilePath, default_value = "fake.npy")]
    dest_audio: PathBuf,

    /// Path to the model checkpoint
    #[arg(long, default_value = "checkpoints/fish-speech-1.5")]
    checkpoint: PathBuf,

    #[arg(long, default_value = "1.5")]
    fish_version: WhichModel,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Processing in-place reconstruction of {:?}", args.src_audio);

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0)?;

    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    let device = Device::Cpu;

    let encoder_version = WhichCodec::from_model(args.fish_version.clone());
    let fish_version = match encoder_version {
        WhichCodec::Mimi => anyhow::bail!("Only official Fish HiFiGAN supported"),
        WhichCodec::Fish(v) => v,
    };
    // CPU preprocessing for now
    let (mut audio, sr) = torchaudio::load(args.src_audio, &Device::Cpu)?;
    if audio.dim(0)? > 1 {
        audio = audio.mean_keepdim(0)?;
    }

    let dtype = DType::F32;
    let model_path = match fish_version {
        WhichFishVersion::Fish1_2 => "firefly-gan-vq-fsq-4x1024-42hz-generator.pth",
        _ => "firefly-gan-vq-fsq-8x1024-21hz-generator.safetensors",
    };

    let config = FireflyConfig::get_config_for(fish_version.clone());
    // Add spurious batch dimension for consistency
    audio = torchaudio::functional::resample(&audio, sr, config.spec_transform.sample_rate as u32)?
        .unsqueeze(0)?;
    let audio_duration_sec =
        audio.shape().dims3()?.2 as f64 / (config.spec_transform.sample_rate as f64);
    println!("Encoding {:.2}s of audio", audio_duration_sec);

    println!("Audio preprocessing complete");

    println!("Using device {:?}", device);
    let vb = match fish_version {
        WhichFishVersion::Fish1_2 => {
            VarBuilder::from_pth(args.checkpoint.join(model_path), dtype, &device)?
        }
        _ => unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[args.checkpoint.join(model_path)],
                dtype,
                &device,
            )?
        },
    };
    let codec = FireflyCodec::load(config.clone(), vb, fish_version)?;
    println!("Model {:?} loaded", args.fish_version);

    let start_decode = Instant::now();
    let result = codec.encode(&audio)?.squeeze(0)?;

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
