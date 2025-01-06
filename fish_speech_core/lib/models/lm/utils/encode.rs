use super::text::TextChunk;
use crate::config::{WhichFishVersion, WhichLM};
use candle_core::{DType, Device, Error, IndexOp, Result, Tensor};
use std::path::PathBuf;
use tokenizers::Tokenizer;

pub struct PromptEncoder<'a> {
    device: Device,
    tokenizer: &'a Tokenizer,
    num_codebooks: usize,
    model_type: WhichLM,
}

impl<'a> PromptEncoder<'a> {
    pub fn new(
        tokenizer: &'a Tokenizer,
        device: &Device,
        num_codebooks: usize,
        model_type: WhichLM,
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
                WhichLM::Fish(WhichFishVersion::Fish1_5) => "<|voice|>",
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

        let semantic_tokens = match self.model_type {
            WhichLM::DualAR | WhichLM::Fish(WhichFishVersion::Fish1_5) => {
                // Fish 1.5: get semantic IDs
                let semantic_start = self.tokenizer.token_to_id("<|semantic:0|>").unwrap();
                semantic_start as f64 + prompt_tokens.i((0, ..))?
            }
            _ => {
                // Fish 1.4 and below: Just use semantic
                let semantic_id = self.tokenizer.token_to_id("<|semantic|>").unwrap_or(5);
                Tensor::from_vec(vec![semantic_id; seqlen], seqlen, &self.device)
            }
        };
        let semantic_tokens = semantic_tokens?.unsqueeze(0)?;
        let vq_span = match self.model_type {
            WhichLM::DualAR | WhichLM::Fish(WhichFishVersion::Fish1_5) => {
                Tensor::cat(&[semantic_tokens, prompt_tokens.clone()], 0)
            }
            _ => {
                let data = prompt_tokens.broadcast_add(&Tensor::ones_like(&prompt_tokens)?)?;
                Tensor::cat(&[semantic_tokens, data], 0)
            }
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

pub struct EncodedChunks {
    pub n_conditioning_tokens: usize,
    pub chunks: Vec<Tensor>,
}

pub fn encode_chunks(
    tokenizer: &Tokenizer,
    chunks: Vec<TextChunk>,
    device: &Device,
    cached_speaker: Option<Tensor>,
    num_codebooks: usize,
    model_type: WhichLM,
    assume_kv_cache: bool,
) -> Result<EncodedChunks> {
    let mut encoded_chunks = Vec::new();

    let prompt_encoder = PromptEncoder::new(tokenizer, device, num_codebooks, model_type);
    // TODO: make this configurable if lenguye says this works
    let system_prompt =
        prompt_encoder.encode_text("system", Some("Speak out the provided text"))?;
    let assistant_start = prompt_encoder.encode_vq(None)?;
    let n_conditioning_tokens = match &cached_speaker {
        Some(t) => system_prompt.dim(1)? + t.dim(1)?,
        _ => 0,
    };

    if chunks.len() == 0 {
        candle_core::bail!("Input text cannot be empty");
    }
    for (i, chunk) in chunks.iter().enumerate() {
        // Format each chunk with the dialogue markers
        let user_request = prompt_encoder.encode_text("user", Some(&chunk.text))?;

        let encoded = if let Some(conditioning_tokens) = cached_speaker.as_ref() {
            // Assume the preprocessing code from earlier worked fine
            if i == 0 || !assume_kv_cache {
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
                // Assume system prompt and conditioning are already in KV cache
                Tensor::cat(&[user_request, assistant_start.clone()], 1)?
            }
        } else {
            Tensor::cat(
                &[system_prompt.clone(), user_request, assistant_start.clone()],
                1,
            )?
        };

        encoded_chunks.push(encoded);
    }

    Ok(EncodedChunks {
        n_conditioning_tokens,
        // This is fine, by invariant it will complete
        chunks: encoded_chunks,
    })
}

pub fn load_prompt_text(
    prompt_path: &PathBuf,
    device: &Device,
    num_codebooks: usize,
) -> Result<Tensor> {
    let prompt_tokens = Tensor::read_npy(prompt_path)?.to_dtype(DType::U32)?;
    let prompt_tokens = prompt_tokens.to_dtype(DType::U32)?.to_device(device)?;
    match (prompt_tokens.rank(), prompt_tokens.dim(0)) {
        (2, Ok(n_actual_codebooks)) => {
            // Fine
            if n_actual_codebooks == num_codebooks {
                return Ok(prompt_tokens);
            } else {
                candle_core::bail!(
                    "Expected {} codebooks but got {}",
                    num_codebooks,
                    n_actual_codebooks
                )
            }
        }
        (3, Ok(1)) => {
            // Ghost third dimension OK
            if prompt_tokens.dim(1)? == num_codebooks {
                prompt_tokens.squeeze(0)
            } else {
                candle_core::bail!(
                    "Expected {} codebooks but got {}",
                    num_codebooks,
                    prompt_tokens.dim(1)?
                )
            }
        }
        (d, _) => {
            candle_core::bail!(
                "Incorrect prompt token dimensions for {:?}: {d}",
                prompt_path
            )
        }
    }
}

pub struct SamplingArgs {
    pub temp: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub repetition_penalty: f32,
}
