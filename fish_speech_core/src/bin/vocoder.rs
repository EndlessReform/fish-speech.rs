use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use fish_speech_core::models::vqgan::decoder::FireflyDecoder;
use fish_speech_core::models::vqgan::utils::config::FireflyConfig;
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    let input = Tensor::read_npy("out.npy")?;
    // TODO: Support BF16
    let dtype = DType::F32;
    // TODO: Support hardware acceleration;
    let device = Device::Cpu;
    let vb = VarBuilder::from_pth(
        "checkpoints/fish-speech-1.2-sft/firefly-gan-vq-fsq-4x1024-42hz-generator-merged.pth",
        dtype,
        &device,
    )?;

    // TODO: Support Fish 1.4
    println!("Loading model");
    let model = FireflyDecoder::load(&vb, &FireflyConfig::fish_speech_1_2())?;
    println!("Model loaded");
    let feature_lengths = Tensor::from_slice(&[input.dim(D::Minus1)? as u32], 1, &device)?;
    let start_decode = Instant::now();
    let fake_audios = model.decode(&input.unsqueeze(0)?, &feature_lengths)?;
    let dt = start_decode.elapsed();
    println!(
        "Time to decode: {:.2}s (RTF: {:.3})",
        dt.as_secs_f64(),
        (fake_audios.dim(D::Minus1)? as f64 / 44100 as f64) / dt.as_secs_f64()
    );
    // fake_audios.write_npy("vocoder_decode_rust.npy")?;

    Ok(())
}
