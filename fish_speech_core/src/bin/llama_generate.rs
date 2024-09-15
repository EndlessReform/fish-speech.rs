use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use fish_speech_core::models::text2semantic::{BaseModelArgs, DualARTransformer};

fn main() {
    println!("Running single token decode to test numerical stability");
    // TODO:

    // Plain vanilla CPU f32 inference for debugging purposes
    let vb = VarBuilder::from_pth(
        "./checkpoints/fish-speech-1.2-sft/model.pth",
        DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    // TODO: Read config from checkpoint folder w/ serde
    let config = BaseModelArgs {
        ..Default::default()
    };

    // TODO: Tokenization and preprocessing
    let example_input = Tensor::read_npy("array_fixed_standard.npy")
        .unwrap()
        .to_dtype(DType::U32)
        .unwrap();
    let mut model = DualARTransformer::load(&vb, &config, 5).unwrap();
    println!("Model loaded");
    let slow_tokens = model.forward_generate(&example_input).unwrap();
    println!("Single token generated!");
    slow_tokens.1.write_npy("first_token_out_rs.npy").unwrap();
    // TODO: Checking fast tokens
}
