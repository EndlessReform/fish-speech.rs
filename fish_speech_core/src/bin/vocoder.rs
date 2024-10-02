use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use clap::Parser;
use fish_speech_core::audio::wav::write_pcm_as_wav;
use fish_speech_core::models::vqgan::config::{FireflyConfig, WhichModel};
use fish_speech_core::models::vqgan::decoder::FireflyDecoder;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    author = "Jacob Keisling <jacob@keisling.me>",
    version = "0.1",
    about = "Decodes codebooks for Fish Speech"
)]
struct Args {
    #[arg(short, long, default_value = "out.npy")]
    input_path: PathBuf,

    #[arg(short, long, default_value = "fake.wav")]
    output_path: PathBuf,

    #[arg(long, default_value = "checkpoints/fish-speech-1.4/")]
    checkpoint: PathBuf,

    #[arg(long, default_value = "1.4")]
    fish_version: WhichModel,
}

fn main() -> Result<()> {
    let args = Args::parse();

    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0)?;

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    let device = Device::Cpu;

    // TODO: BF16 on Metal is mostly unsupported
    #[cfg(feature = "cuda")]
    let dtype = DType::BF16;

    #[cfg(not(feature = "cuda"))]
    let dtype = DType::F32;

    let config = match args.fish_version {
        WhichModel::Fish1_2 => FireflyConfig::fish_speech_1_2(),
        _ => FireflyConfig::fish_speech_1_4(),
    };
    let vb = match args.fish_version {
        // NOTE: Requires weights to have weight norm merged!
        WhichModel::Fish1_2 => VarBuilder::from_pth(
            args.checkpoint
                .join("firefly-gan-vq-fsq-4x1024-42hz-generator-merged.pth"),
            dtype,
            &device,
        )?,
        _ => unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[args
                    .checkpoint
                    .join("firefly-gan-vq-fsq-8x1024-21hz-generator.safetensors")],
                dtype,
                &device,
            )?
        },
    };

    println!("Loading {:?} model on {:?}", args.fish_version, device);
    let start_load = Instant::now();
    // TODO: Make this configurable from CLI
    let model = FireflyDecoder::load(&vb, &config, &args.fish_version)?;
    let dt = start_load.elapsed();
    println!("Model loaded in {:.2}s", dt.as_secs_f64());

    let input = Tensor::read_npy(args.input_path)?.to_device(&device)?;
    let feature_lengths = Tensor::from_slice(&[input.dim(D::Minus1)? as u32], 1, &device)?;

    let start_decode = Instant::now();
    let fake_audios = model
        .decode(&input.unsqueeze(0)?, &feature_lengths)?
        .to_dtype(DType::F32)?;

    let dt = start_decode.elapsed();
    println!(
        "Generated {:.2}s of audio",
        (fake_audios.dim(D::Minus1)? as f64 / config.spec_transform.sample_rate as f64)
    );
    println!(
        "Time elapsed: {:.2}s (RTF: {:.3})",
        dt.as_secs_f64(),
        (fake_audios.dim(D::Minus1)? as f64 / config.spec_transform.sample_rate as f64)
            / dt.as_secs_f64()
    );
    let pcm = fake_audios
        .squeeze(0)?
        .squeeze(0)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let mut output = std::fs::File::create(args.output_path)?;
    write_pcm_as_wav(&mut output, &pcm, config.spec_transform.sample_rate as u32)?;

    Ok(())
}
