use crate::audio::mimi;
use candle_core::{DType, Result, Tensor, D};
use fish_speech_core::audio::spectrogram::LogMelSpectrogram;
use fish_speech_core::models::vqgan::config::FireflyConfig;
use fish_speech_core::models::{vqgan::decoder::FireflyDecoder, vqgan::encoder::FireflyEncoder};
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct HiFiGANState {
    pub decoder_model: Arc<FireflyDecoder>,
    pub encoder_model: Arc<FireflyEncoder>,
    pub firefly_config: Arc<FireflyConfig>,
    pub spec_transform: Arc<LogMelSpectrogram>,
}

pub enum Codec {
    Mimi(Arc<Mutex<mimi::Tokenizer>>),
    HiFiGAN(HiFiGANState),
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
            Codec::HiFiGAN(state) => {
                let mels = state.spec_transform.forward(&audio)?;
                state.encoder_model.encode(&mels)
            }
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
            Codec::HiFiGAN(state) => {
                let feature_lengths = Tensor::from_slice(
                    &[semantic_tokens.dim(D::Minus1)? as u32],
                    1,
                    &state.decoder_model.device,
                )?;

                state
                    .decoder_model
                    .decode(&semantic_tokens, &feature_lengths)?
                    .to_dtype(DType::F32)
            }
        }
    }
}
