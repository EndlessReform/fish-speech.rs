use crate::audio::mimi;
use candle_core::{Result, Tensor};
use fish_speech_core::codec::FireflyCodec;
use std::sync::Arc;
use tokio::sync::Mutex;

pub enum Codec {
    Mimi(Arc<Mutex<mimi::Tokenizer>>),
    Firefly(Arc<FireflyCodec>),
}

impl Codec {
    /// Expects mono PCM b * t at target sample rate
    pub async fn encode_batch(&self, audio: &Tensor) -> Result<Tensor> {
        match self {
            Codec::Mimi(model_mutex) => {
                let audio = if audio.rank() == 2 {
                    &audio.unsqueeze(0)?
                } else {
                    audio
                };
                let mut model = model_mutex.lock().await;
                let out = model.encode_batch(audio);
                println!("out: {:?}", out);
                out
            }
            Codec::Firefly(state) => state.encode(&audio),
        }
    }

    pub async fn decode_batch(&self, semantic_tokens: &Tensor) -> Result<Tensor> {
        let semantic_tokens = if semantic_tokens.rank() == 2 {
            &semantic_tokens.unsqueeze(0)?
        } else {
            semantic_tokens
        };
        match self {
            Codec::Mimi(model_mutex) => {
                let mut model = model_mutex.lock().await;
                let tokens = model.decode_batch(semantic_tokens);
                model.reset();
                tokens
            }
            Codec::Firefly(state) => state.decode(&semantic_tokens),
        }
    }
}
