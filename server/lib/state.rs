use candle_core::{Device, Tensor};
use fish_speech_core::audio::spectrogram::LogMelSpectrogram;
use fish_speech_core::models::text2semantic::BaseModelArgs;
use fish_speech_core::models::vqgan::config::{FireflyConfig, WhichLM, WhichModel};
use fish_speech_core::models::{
    text2semantic::DualARTransformer, vqgan::decoder::FireflyDecoder,
    vqgan::encoder::FireflyEncoder,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

pub struct LMState {
    pub model: Arc<Mutex<DualARTransformer>>,
    pub model_type: WhichLM,
    pub tokenizer: Arc<Tokenizer>,
    pub voices: Arc<Mutex<HashMap<String, Tensor>>>,
    pub default_voice: Arc<Tensor>,
    pub temp: f64,
    pub top_p: f64,
    pub config: Arc<BaseModelArgs>,
}

/// Shared state between requests
pub struct AppState {
    pub vocoder_model: Arc<FireflyDecoder>,
    pub encoder_model: Arc<FireflyEncoder>,
    pub firefly_config: Arc<FireflyConfig>,
    pub spec_transform: Arc<LogMelSpectrogram>,
    pub model_type: WhichModel,
    pub device: Device,
    pub lm: Arc<LMState>,
}
