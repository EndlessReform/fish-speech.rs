use super::audio::codec::Codec;
use candle_core::{Device, Tensor};
use fish_speech_core::models::lm::BaseModelArgs;
use fish_speech_core::models::lm::DualARTransformer;
use fish_speech_core::models::vqgan::config::{WhichLM, WhichModel};
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

pub struct LMState {
    pub model: Arc<Mutex<DualARTransformer>>,
    pub model_type: WhichLM,
    pub config: Arc<BaseModelArgs>,
    pub tokenizer: Arc<Tokenizer>,
    pub voices: Arc<Mutex<HashMap<String, Tensor>>>,
    pub default_voice: Arc<Tensor>,
    pub default_temp: f64,
    pub default_top_p: f64,
}

pub struct AppState {
    pub lm: Arc<LMState>,
    pub codec: Arc<Codec>,
    pub model_type: WhichModel,
    pub device: Device,
    pub sample_rate: u32,
}
