use super::utils::config::HiFiGANConfig;
use candle_core::Result;
use candle_nn::{Conv1d, VarBuilder};

pub struct HiFiGAN {}

impl HiFiGAN {
    pub fn load(vb: VarBuilder, cfg: &HiFiGANConfig) -> Result<Self> {
        Err(candle_core::Error::Msg("Not yet implemented".into()))
    }
}
