use candle_core::{DType, Device, Result, Tensor, D};
use tokenizers::Tokenizer;

pub fn encode_tokens(
    tokenizer: &Tokenizer,
    input_string: &str,
    device: &Device,
    prompt_tokens: Option<&Tensor>,
    num_codebooks: usize,
) -> Result<Tensor> {
    // let cleaned_string = clean_text(input_string);
    let cleaned_string = input_string.to_owned();
    let turn_string = format!(
        "<|im_start|>user\n{}<|im_end|><|im_start|>assistant\n",
        cleaned_string
    );
    // TODO Graceful error handling
    let encodings = tokenizer.encode(turn_string, false).unwrap();
    let new_tokens = encodings.get_ids();
    let tokens = Tensor::from_slice(new_tokens, (1, new_tokens.len()), device)?;

    // Pad codebooks for text prompt
    let zeros = Tensor::zeros((num_codebooks, new_tokens.len()), DType::U32, device)?;
    let prompt = Tensor::cat(&[tokens, zeros], 0)?;

    if let None = prompt_tokens {
        return Ok(prompt);
    }
    let prompt_tokens = prompt_tokens.unwrap();
    let prompt_tokens = match prompt_tokens.dim(0) {
        Ok(3) => {
            assert_eq!(
                prompt_tokens.dim(0)?,
                1,
                "3 dim prompt tokens should have shape (1, num_codebooks, seq_len)"
            );
            &prompt_tokens.squeeze(0)?
        }
        Ok(2) => prompt_tokens,
        _ => Err(candle_core::Error::Msg(
            "Prompt tokens must have 2 or 3 dimensions".into(),
        ))?,
    };
    assert_eq!(prompt_tokens.dim(0)?, num_codebooks);
    // Yes, this is inefficient, but +1 actually fails (if you can believe it)
    let data = prompt_tokens.broadcast_add(&Tensor::ones_like(prompt_tokens)?)?;

    // Add pad token for each codebook
    let data = Tensor::cat(
        &[data, Tensor::zeros((num_codebooks, 1), DType::U32, device)?],
        1,
    )?;

    // Fill in the speaker line
    let s0_token_id = tokenizer.encode("<|semantic|>", false).unwrap().get_ids()[0];
    let end_token_id = tokenizer.encode("<|im_end|>", false).unwrap().get_ids()[0];
    let mut main_token_ids = vec![s0_token_id, (data.dim(1)? - 1) as u32];
    main_token_ids.push(end_token_id);
    let main_token_ids = Tensor::from_vec(main_token_ids, (1, data.dim(1)?), device)?;

    let data = Tensor::cat(&[main_token_ids, data], 0)?;
    Tensor::cat(&[prompt, data], 1)
}
