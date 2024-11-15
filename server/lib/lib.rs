pub mod error;
pub mod generate;
pub mod handlers;
pub mod opus;
pub mod state;

use candle_core::{Device, Tensor};
use fish_speech_core::models::text2semantic::utils::encode::encode_tokens;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Deserialize)]
struct SpeakerIndex {
    // filename (without .npy) -> prompt text
    speakers: HashMap<String, String>,
}

pub fn load_speaker_prompts(
    voice_dir: &Path,
    tokenizer: &Tokenizer,
    device: &Device,
    num_codebooks: usize,
) -> anyhow::Result<(HashMap<String, Tensor>, Tensor)> {
    // Load the index file
    let index_path = voice_dir.join("index.json");
    let index: SpeakerIndex = serde_json::from_reader(
        std::fs::File::open(&index_path)
            .map_err(|_| anyhow::anyhow!("Failed to open speaker index.json"))?,
    )?;

    let mut speakers = HashMap::new();
    let mut default_prompt = None;

    for (name, prompt_text) in index.speakers {
        let npy_path = voice_dir.join(format!("{}.npy", name));
        let prompt_tensor = Tensor::read_npy(&npy_path)?.to_device(device)?;

        // Pre-process the full prompt once
        let encoded = encode_tokens(
            tokenizer,
            &prompt_text,
            device,
            Some(&prompt_tensor),
            num_codebooks,
        )?;

        let prompt = encoded;

        if name == "default" {
            default_prompt = Some(prompt.clone());
        }

        speakers.insert(name, prompt);
    }

    let default_prompt = default_prompt.ok_or_else(|| {
        anyhow::anyhow!("No default speaker found in index.json and voices directory")
    })?;

    Ok((speakers, default_prompt))
}
