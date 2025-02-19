pub mod load;

use candle_core::{Device, Tensor};
use fish_speech_core::config::WhichLM;
use fish_speech_core::text::prompt::{load_prompt_text, PromptEncoder};
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
    model_type: WhichLM,
) -> anyhow::Result<(HashMap<String, Tensor>, Tensor)> {
    // Load the index file
    let index_path = voice_dir.join("index.json");
    let index: SpeakerIndex = serde_json::from_reader(
        std::fs::File::open(&index_path)
            .map_err(|_| anyhow::anyhow!("Failed to open speaker index.json"))?,
    )?;

    // TODO: pass model down?
    let prompt_encoder = PromptEncoder::new(tokenizer, device, num_codebooks, model_type);
    let mut speakers = HashMap::new();
    let mut default_prompt = None;

    for (name, prompt_text) in index.speakers {
        let npy_path = voice_dir.join(format!("{}.npy", name));
        let prompt_tensor = load_prompt_text(&npy_path, device, num_codebooks)?;

        // Pre-process the full prompt once
        let prompt = prompt_encoder.encode_conditioning_prompt(&prompt_text, &prompt_tensor)?;

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
