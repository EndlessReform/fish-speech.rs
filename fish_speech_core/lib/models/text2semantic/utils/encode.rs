use super::text::TextChunk;
use crate::models::vqgan::config::WhichModel;
use candle_core::{DType, Device, Error, IndexOp, Result, Tensor};
use tokenizers::Tokenizer;

pub struct PromptEncoder<'a> {
    device: Device,
    tokenizer: &'a Tokenizer,
    num_codebooks: usize,
    model_type: WhichModel,
}

impl<'a> PromptEncoder<'a> {
    pub fn new(
        tokenizer: &'a Tokenizer,
        device: &Device,
        num_codebooks: usize,
        model_type: WhichModel,
    ) -> Self {
        Self {
            device: device.clone(),
            tokenizer,
            num_codebooks,
            model_type,
        }
    }

    fn tokenize_text(&self, text: String) -> Result<Tensor> {
        let turn_codes = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Error::Msg(format!("Could not tokenize: {:?}", e)))?;
        let new_tokens = turn_codes.get_ids();

        let tokens = Tensor::from_slice(new_tokens, (1, new_tokens.len()), &self.device)?;
        let zeros = Tensor::zeros(
            (self.num_codebooks, new_tokens.len()),
            DType::U32,
            &self.device,
        )?;
        Tensor::cat(&[tokens, zeros], 0)
    }

    pub fn encode_text(&self, role: &str, content: Option<&str>) -> Result<Tensor> {
        let turn_string = match content {
            Some(content) => format!("<|im_start|>{}\n{}<|im_end|>", role, content),
            None => format!("<|im_start|>{}\n", role),
        };
        // Fill in zeros below
        self.tokenize_text(turn_string)
    }

    pub fn encode_vq(&self, maybe_prompt_tokens: Option<&Tensor>) -> Result<Tensor> {
        let prefix_string = format!(
            "<|im_start|>assistant\n{}",
            match self.model_type {
                WhichModel::Fish1_5 => "<|voice|>",
                _ => "",
            }
        );
        let prefix_tokens = self.tokenize_text(prefix_string)?;
        let suffix_tokens = self.tokenize_text("<|im_end|>".to_owned());
        if maybe_prompt_tokens.is_none() {
            return Ok(prefix_tokens);
        }
        let prompt_tokens = maybe_prompt_tokens.unwrap();

        let (_, seqlen) = prompt_tokens.dims2()?;

        let semantic_tokens = if self.model_type == WhichModel::Fish1_5 {
            // Fish 1.5: get semantic IDs
            let semantic_start = self.tokenizer.token_to_id("<|semantic:0|>").unwrap();
            let semantic_end = self
                .tokenizer
                .token_to_id(&format!("<|semantic:{}|>", self.num_codebooks - 1))
                .unwrap();
            let semantic_offset = (semantic_end - semantic_start) as f64;
            semantic_offset + prompt_tokens.i((0, ..))?
        } else {
            // Fish 1.4 and below: Just use semantic
            let semantic_id = self.tokenizer.token_to_id("<|semantic|>").unwrap_or(5);
            Tensor::from_vec(vec![semantic_id; seqlen], seqlen, &self.device)
        };
        let semantic_tokens = semantic_tokens?.unsqueeze(0)?;
        println!("Starting the hell block");
        let vq_span = if self.model_type == WhichModel::Fish1_5 {
            // TODO: are tokens shifted here?
            println!("Semantic size: {:?}", semantic_tokens.shape());
            Tensor::cat(&[semantic_tokens, prompt_tokens.clone()], 0)
        } else {
            println!("In the hell block");
            let data = prompt_tokens.broadcast_add(&Tensor::ones_like(&prompt_tokens)?)?;
            Tensor::cat(&[semantic_tokens, data], 0)
        };
        Tensor::cat(&[prefix_tokens, vq_span?, suffix_tokens?], 1)
    }

    /// Convenience function to replace `encode_texts`
    pub fn encode_conditioning_prompt(
        &self,
        prompt_text: &str,
        prompt_tensor: &Tensor,
    ) -> Result<Tensor> {
        let user_prompt = self.encode_text("user", Some(prompt_text))?;
        let assistant_prompt = self.encode_vq(Some(prompt_tensor))?;
        Tensor::cat(&[user_prompt, assistant_prompt], 1)
    }
}

// pub fn encode_tokens(
//     tokenizer: &Tokenizer,
//     input_string: &str,
//     device: &Device,
//     prompt_tokens: Option<&Tensor>,
//     num_codebooks: usize,
// ) -> Result<Tensor> {
//     // let cleaned_string = clean_text(input_string);
//     let cleaned_string = input_string.to_owned();
//     let turn_string = format!(
//         "<|im_start|>user\n{}<|im_end|><|im_start|>assistant\n",
//         cleaned_string
//     );
//     // TODO Graceful error handling
//     let encodings = tokenizer.encode(turn_string, false).unwrap();
//     let new_tokens = encodings.get_ids();
//     let tokens = Tensor::from_slice(new_tokens, (1, new_tokens.len()), device)?;

//     // Pad codebooks for text prompt
//     let zeros = Tensor::zeros((num_codebooks, new_tokens.len()), DType::U32, device)?;
//     let prompt = Tensor::cat(&[tokens, zeros], 0)?;

//     if prompt_tokens.is_none() {
//         return Ok(prompt);
//     }
//     let prompt_tokens = prompt_tokens.unwrap().to_dtype(DType::U32)?;
//     let prompt_tokens = match prompt_tokens.shape().rank() {
//         3 => {
//             assert_eq!(
//                 prompt_tokens.dim(0)?,
//                 1,
//                 "3 dim prompt tokens should have shape (1, num_codebooks, seq_len)"
//             );
//             prompt_tokens.squeeze(0)?
//         }
//         2 => prompt_tokens,
//         _ => Err(candle_core::Error::Msg(
//             "Prompt tokens must have 2 or 3 dimensions".into(),
//         ))?,
//     };
//     assert_eq!(prompt_tokens.dim(0)?, num_codebooks);
//     // Yes, this is inefficient, but +1 actually fails (if you can believe it)
//     let data = prompt_tokens.broadcast_add(&Tensor::ones_like(&prompt_tokens)?)?;

//     // Add pad token for each codebook
//     let data = Tensor::cat(
//         &[data, Tensor::zeros((num_codebooks, 1), DType::U32, device)?],
//         1,
//     )?;

//     // Fill in the speaker line
//     let s0_token_id = tokenizer.token_to_id("<|semantic|>").unwrap_or(5);
//     let end_token_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(4);
//     let mut main_token_ids = vec![s0_token_id; data.dim(D::Minus1)? - 1];
//     main_token_ids.push(end_token_id);
//     let main_token_ids = Tensor::from_vec(main_token_ids, (1, data.dim(1)?), device)?;

//     let data = Tensor::cat(&[main_token_ids, data], 0)?;
//     Tensor::cat(&[prompt, data], 1)
// }

pub fn encode_chunks(
    tokenizer: &Tokenizer,
    chunks: Vec<TextChunk>,
    device: &Device,
    cached_speaker: Option<&Tensor>,
    num_codebooks: usize,
) -> Result<Vec<Tensor>> {
    let mut encoded_chunks = Vec::new();

    for chunk in chunks {
        // Format each chunk with the dialogue markers
        let turn_string = format!(
            "<|im_start|>user\n{}<|im_end|><|im_start|>assistant\n",
            chunk.text
        );

        let encodings = tokenizer
            .encode(turn_string, false)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenization failed: {}", e)))?;
        let new_tokens = encodings.get_ids();
        let tokens = Tensor::from_slice(new_tokens, (1, new_tokens.len()), device)?;

        // Pad codebooks for text prompt
        let zeros = Tensor::zeros((num_codebooks, new_tokens.len()), DType::U32, device)?;
        let prompt = Tensor::cat(&[tokens, zeros], 0)?;

        let encoded = if let Some(prompt_tokens) = cached_speaker {
            let prompt_tokens = prompt_tokens.to_dtype(DType::U32)?;
            let prompt_tokens = match prompt_tokens.shape().rank() {
                3 => {
                    assert_eq!(
                        prompt_tokens.dim(0)?,
                        1,
                        "3 dim prompt tokens should have shape (1, num_codebooks, seq_len)"
                    );
                    prompt_tokens.squeeze(0)?
                }
                2 => prompt_tokens.clone(),
                _ => {
                    return Err(candle_core::Error::Msg(
                        "Prompt tokens must have 2 or 3 dimensions".into(),
                    ))
                }
            };
            println!("Prompt token shape: {:?}", prompt_tokens.shape());
            assert_eq!(prompt_tokens.dim(0)?, num_codebooks + 1);

            Tensor::cat(&[prompt_tokens, prompt], 1)?
        } else {
            prompt
        };

        encoded_chunks.push(encoded);
    }

    Ok(encoded_chunks)
}
