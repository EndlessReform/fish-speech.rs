use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Module, VarBuilder};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use fish_speech_core::models::text2semantic::{BaseModelArgs, DualARTransformer};

fn decode_one_token_ar(
    model: &mut DualARTransformer,
    logits_processor: &mut LogitsProcessor,
    x: &Tensor,
    input_pos: usize,
    previous_tokens: &Tensor,
    write_debug_output: bool,
) -> Result<Tensor> {
    let (logits, hidden_states) = model.forward_generate(&x, input_pos)?;
    if write_debug_output {
        logits.write_npy("first_token_out_rs.npy").unwrap();
        hidden_states.write_npy("first_hidden_rs.npy").unwrap();
    }

    let mut codebooks = vec![logits_processor.sample(&logits.flatten_all()?)?];
    println!("Codes: {:?}", codebooks);
    model.clear_fast_layer_caches();

    let mut x = hidden_states;
    println!("Generating fast codebooks");
    for codebook_idx in 0..model.cfg.num_codebooks {
        // TODO: Figure out what the heck input_pos is
        let logits = model.forward_generate_fast(&x, codebook_idx)?;
        if write_debug_output {
            logits.write_npy(format!("fast_codebook_{}_logits_rust.npy", codebook_idx))?;
        }
        // TODO: Handle previous_tokens!
        let a = logits.flatten_all()?.argmax(D::Minus1)?.to_vec0::<u32>()?;
        println!("a: {:?}", a);
        let a = logits_processor.sample(&logits.flatten_all()?)?;
        println!("Code at layer {}: {:?}", codebook_idx + 1, a);
        // println!("Codebook shape: {:?}", prev_codes[codebook_idx + 1].shape());
        let a_tensor = Tensor::from_slice(&[a], 1, x.device())?;
        x = model.fast_embeddings.forward(&a_tensor)?.unsqueeze(0)?;
        println!(
            "Generated for codebook {} with shape {:?}",
            codebook_idx + 1,
            x.shape()
        );
        codebooks.push(a);
    }
    Tensor::from_vec(codebooks, model.cfg.num_codebooks + 1, x.device())
}

fn main() -> anyhow::Result<()> {
    println!("Running single token decode to test numerical stability");
    // TODO: Read config from checkpoint folder w/ serde; Fish Speech 1.4 support
    let config = BaseModelArgs::fish_speech_1_2();
    // TODO: Tokenization and preprocessing
    let example_input = Tensor::read_npy("final_prompt.npy")?.to_dtype(DType::U32)?;
    assert!(example_input.dim(0)? == config.num_codebooks + 1);

    // Plain vanilla CPU f32 inference for debugging purposes
    // TODO: Hardware acceleration, bf16
    let vb = VarBuilder::from_pth(
        "./checkpoints/fish-speech-1.2-sft/model.pth",
        DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    let mut model = DualARTransformer::load(&vb, &config, 5).unwrap();
    println!("Model loaded");

    // Using deterministic sampling for debugging purposes.
    // TODO: Make this configurable in API
    let mut logits_processor = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    println!("Single token generated!");
    decode_one_token_ar(
        &mut model,
        &mut logits_processor,
        &example_input,
        0,
        &example_input,
        true,
    )?;
    Ok(())
}
