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
            semantic_start as f64 + prompt_tokens.i((0, ..))?
        } else {
            // Fish 1.4 and below: Just use semantic
            let semantic_id = self.tokenizer.token_to_id("<|semantic|>").unwrap_or(5);
            Tensor::from_vec(vec![semantic_id; seqlen], seqlen, &self.device)
        };
        let semantic_tokens = semantic_tokens?.unsqueeze(0)?;
        let vq_span = if self.model_type == WhichModel::Fish1_5 {
            Tensor::cat(&[semantic_tokens, prompt_tokens.clone()], 0)
        } else {
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

pub fn encode_chunks(
    tokenizer: &Tokenizer,
    chunks: Vec<TextChunk>,
    device: &Device,
    cached_speaker: Option<&Tensor>,
    num_codebooks: usize,
    model_type: WhichModel,
) -> Result<Vec<Tensor>> {
    let mut encoded_chunks = Vec::new();

    let prompt_encoder = PromptEncoder::new(tokenizer, device, num_codebooks, model_type);
    // TODO: make this configurable if lenguye says this works
    let system_prompt =
        prompt_encoder.encode_text("system", Some("Speak out the provided text"))?;
    let assistant_start = prompt_encoder.encode_vq(None)?;

    for chunk in chunks {
        // Format each chunk with the dialogue markers
        let user_request = prompt_encoder.encode_text("user", Some(&chunk.text))?;

        let encoded = if let Some(conditioning_tokens) = cached_speaker {
            // Assume the preprocessing code from earlier worked fine
            Tensor::cat(
                &[
                    system_prompt.clone(),
                    conditioning_tokens.clone(),
                    user_request,
                    assistant_start.clone(),
                ],
                1,
            )?
        } else {
            Tensor::cat(
                &[system_prompt.clone(), user_request, assistant_start.clone()],
                1,
            )?
        };

        encoded_chunks.push(encoded);
    }

    Ok(encoded_chunks)
}
