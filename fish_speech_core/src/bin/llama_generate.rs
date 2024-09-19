use candle_core::{DType, Device, IndexOp, NdArray, Result, Tensor, D};
use candle_nn::{Module, VarBuilder};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use fish_speech_core::models::text2semantic::{BaseModelArgs, DualARTransformer};
use std::time::Instant;

fn decode_one_token_ar(
    model: &mut DualARTransformer,
    logits_processor: &mut LogitsProcessor,
    x: &Tensor,
    input_pos: usize,
    previous_tokens: &Tensor,
    write_debug_output: bool,
) -> Result<Vec<u32>> {
    let (logits, hidden_states) = model.forward_generate(&x, input_pos)?;
    if write_debug_output {
        logits.write_npy("first_token_out_rs.npy").unwrap();
        hidden_states.write_npy("first_hidden_rs.npy").unwrap();
    }

    let mut codebooks: Vec<u32> = vec![logits_processor.sample(&logits.flatten_all()?)?];
    model.clear_fast_layer_caches();

    let mut x = hidden_states;
    // println!("x shape: {:?}", x.shape());
    for codebook_idx in 0..model.cfg.num_codebooks {
        // TODO: Figure out what the heck input_pos is
        let logits = model.forward_generate_fast(&x, codebook_idx)?;
        if write_debug_output {
            logits.write_npy(format!("fast_codebook_{}_logits_rust.npy", codebook_idx))?;
        }
        // TODO: Handle previous_tokens!
        let a = logits_processor.sample(&logits.flatten_all()?)?;
        // println!("Codebook shape: {:?}", prev_codes[codebook_idx + 1].shape());
        let a_tensor = Tensor::from_slice(&[a], 1, x.device())?;
        x = model.fast_embeddings.forward(&a_tensor)?.unsqueeze(0)?;
        codebooks.push(a);
    }
    // Tensor::from_vec(codebooks, model.cfg.num_codebooks + 1, x.device())?.unsqueeze(D::Minus1)
    Ok(codebooks)
}

/// Takes a conditioning sequence as input and generates as many tokens as requested
fn generate(
    model: &mut DualARTransformer,
    prompt: &Tensor,
    max_new_tokens: usize,
    im_end_id: Option<u32>,
) -> Result<Tensor> {
    // Using deterministic sampling for debugging purposes.
    // TODO: Make this configurable in API
    let mut logits_processor = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let start_pp = Instant::now();
    let mut cur_token =
        decode_one_token_ar(model, &mut logits_processor, prompt, 0, prompt, false)?;
    let dt = start_pp.elapsed();
    let mut input_pos = prompt.dim(D::Minus1)?;
    println!(
        "{} prompt processing timesteps ({:.2} tokens/s)",
        input_pos,
        input_pos as f64 / dt.as_secs_f64()
    );

    let mut previous_tokens = vec![Tensor::from_vec(cur_token.clone(), 5, prompt.device())?];
    println!("First token: {:?}", cur_token);

    let start_decode = Instant::now();
    for i in 1..max_new_tokens {
        // TODO: handle window in inference code.
        // WTF does it do?!
        // Thank god for KV cache
        let input_token = Tensor::from_vec(cur_token, (5, 1), prompt.device())?.contiguous()?;
        let next_token = decode_one_token_ar(
            model,
            &mut logits_processor,
            &input_token,
            input_pos,
            &input_token,
            false,
        )?;
        println!("Token {}: {:?}", i + 1, next_token);
        previous_tokens.push(Tensor::from_vec(next_token.clone(), 5, prompt.device())?);
        input_pos += 1;
        cur_token = next_token;
    }
    let dt = start_decode.elapsed();
    println!(
        "{} tokens generated ({:.2} tokens/s)",
        previous_tokens.len(),
        previous_tokens.len() as f64 / dt.as_secs_f64()
    );
    Tensor::cat(&previous_tokens, D::Minus1)
}

fn main() -> anyhow::Result<()> {
    // TODO: Read config from checkpoint folder w/ serde; Fish Speech 1.4 support
    let config = BaseModelArgs::fish_speech_1_2();
    // TODO: Tokenization and preprocessing
    let example_input = Tensor::read_npy("final_prompt.npy")?.to_dtype(DType::U32)?;
    assert!(example_input.dim(0)? == config.num_codebooks + 1);
    println!("Loaded prompt with shape {:?}", example_input.shape());

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

    let res = generate(&mut model, &example_input, 100, Some(4))?;
    res.write_npy("out.npy")?;
    Ok(())
}
