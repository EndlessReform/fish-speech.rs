use candle_core::{Device, Tensor};
use fish_speech_core::audio::spectrogram::LogMelSpectrogram;
use fish_speech_core::models::text2semantic::BaseModelArgs;
use fish_speech_core::models::vqgan::config::FireflyConfig;
use fish_speech_core::models::vqgan::config::WhichModel;
use fish_speech_core::models::{
    text2semantic::DualARTransformer, vqgan::decoder::FireflyDecoder,
    vqgan::encoder::FireflyEncoder,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

/// Shared state between requests
pub struct AppState {
    pub semantic_model: Arc<Mutex<DualARTransformer>>,
    pub vocoder_model: Arc<FireflyDecoder>,
    pub encoder_model: Arc<FireflyEncoder>,
    pub semantic_config: Arc<BaseModelArgs>,
    pub firefly_config: Arc<FireflyConfig>,
    pub spec_transform: Arc<LogMelSpectrogram>,
    pub tokenizer: Arc<Tokenizer>,
    pub device: Device,
    pub voices: Arc<Mutex<HashMap<String, Tensor>>>,
    pub default_voice: Arc<Tensor>,
    pub temp: f64,
    pub top_p: f64,
    pub model_type: WhichModel,
}
