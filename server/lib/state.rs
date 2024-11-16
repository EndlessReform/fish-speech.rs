use candle_core::{Device, Tensor};
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
    pub tokenizer: Arc<Tokenizer>,
    pub device: Device,
    pub voices: Arc<HashMap<String, Tensor>>,
    pub default_voice: Arc<Tensor>,
    pub temp: f64,
    pub top_p: f64,
}
