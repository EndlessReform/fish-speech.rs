use anyhow::Result;
use candle_core::{DType, Device, D};
use candle_nn::VarBuilder;
use clap::{Parser, ValueHint};
use fish_speech_core::audio as torchaudio;
use fish_speech_core::audio::spectrogram::{LogMelSpectrogram, LogMelSpectrogramConfig};
use fish_speech_core::models::vqgan::config::{FireflyConfig, WhichModel};
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
    #[arg(short = 'o', long = "output-path", value_hint = ValueHint::FilePath, default_value = "fake.npy")]
    dest_audio: PathBuf,

    /// Path to the model checkpoint
    #[arg(long, default_value = "checkpoints/fish-speech-1.4")]
    checkpoint: PathBuf,

    #[arg(long, default_value = "1.4")]
    fish_version: WhichModel,
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

    // CPU preprocessing for now
    let (mut audio, sr) = torchaudio::load(args.src_audio, &Device::Cpu)?;
    if audio.dim(0)? > 1 {
        audio = audio.mean_keepdim(0)?;
    }


    let config = FireflyConfig::get_config_for(args.fish_version);
    // Add spurious batch dimension for consistency
    audio = torchaudio::functional::resample(&audio, sr, config.spec_transform.sample_rate as u32)?
        .unsqueeze(0)?;
    let audio_duration_sec =
        audio.shape().dims3()?.2 as f64 / (config.spec_transform.sample_rate as f64);
    println!("Encoding {:.2}s of audio", audio_duration_sec);

    let spec_transform = LogMelSpectrogram::load(LogMelSpectrogramConfig::default())?;
    let mels = spec_transform.forward(&audio)?.to_device(&device)?;
    println!("Audio preprocessing complete");

    let dtype = DType::F32;
    let model_path = match args.fish_version {
        WhichModel::Fish1_2 => "firefly-gan-vq-fsq-4x1024-42hz-generator.pth",
        _ => "firefly-gan-vq-fsq-8x1024-21hz-generator.safetensors",
    };

    println!("Using device {:?}", device);
    let vb = match args.fish_version {
        WhichModel::Fish1_2 => {
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
    let encoder = FireflyEncoder::load(vb, &config, &args.fish_version)?;
    println!("Model {:?} loaded", args.fish_version);

    let start_decode = Instant::now();
    let result = encoder.encode(&mels)?.squeeze(0)?;

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
