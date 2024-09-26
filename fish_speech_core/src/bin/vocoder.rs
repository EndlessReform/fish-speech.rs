use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

fn main() -> Result<()> {
    let input = Tensor::read_npy("out.npy")?;
    // TODO: Support BF16
    let dtype = DType::F32;
    // TODO: Support hardware acceleration;
    let device = Device::Cpu;
    let vb = VarBuilder::from_pth("checkpoints/fish-speech-1.2-sft", dtype, &device)?;

    Ok(())
}
