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

    /// Returns (num_conditioning_tokens, encoded sequence)
    pub fn encode_sequence(
        self,
        chunks: Vec<String>,
        sysprompt_text: Option<String>,
        cached_speaker: Option<Tensor>,
        assume_kv_cache: bool,
    ) -> Result<(usize, Vec<Tensor>)> {
        if chunks.len() == 0 {
            candle_core::bail!("Input text cannot be empty");
        }

        let mut encoded_chunks = Vec::new();
        let sysprompt = sysprompt_text
            .map(|sysprompt_text| self.encode_text("system", Some(&sysprompt_text)))
            .transpose()?;

        let sysprompt_length = sysprompt
            .as_ref()
            .map(|sysprompt| sysprompt.dim(1).unwrap())
            .unwrap_or(0);
        let speaker_length = cached_speaker
            .as_ref()
            .map(|cached_speaker| cached_speaker.dim(1).unwrap())
            .unwrap_or(0);
        let num_conditioning_tokens = sysprompt_length + speaker_length;

        let conditioning_tokens = match (sysprompt, cached_speaker) {
            (Some(sysprompt), Some(cached_speaker)) => {
                Some(Tensor::cat(&[sysprompt, cached_speaker], 1)?)
            }
            (Some(sysprompt), None) => Some(sysprompt),
            (None, Some(cached_speaker)) => Some(cached_speaker),
            (None, None) => None,
        };
        let assistant_start = self.encode_vq(None)?;

        for (i, chunk) in chunks.iter().enumerate() {
            let mut prompt: Vec<Tensor> = Vec::new();

            if conditioning_tokens.is_some() && (i == 0 || !assume_kv_cache) {
                prompt.push(conditioning_tokens.clone().unwrap());
            }
            prompt.push(self.encode_text("user", Some(&chunk))?);
            prompt.push(assistant_start.clone());

            let encoded = Tensor::cat(&prompt, 1)?;
            encoded_chunks.push(encoded);
        }
        Ok((num_conditioning_tokens, encoded_chunks))
    }
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
