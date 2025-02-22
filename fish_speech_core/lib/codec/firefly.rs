use candle_core::{DType, Result, Tensor, D};
use candle_nn::VarBuilder;

use super::config::FireflyConfig;
use super::decoder::FireflyDecoder;
use super::encoder::FireflyEncoder;
use crate::audio::spectrogram::{LogMelSpectrogram, LogMelSpectrogramConfig};
use crate::config::WhichFishVersion;

pub struct FireflyCodec {
    // Fields and methods go here
    pub cfg: FireflyConfig,
    pub sample_rate: u32,
    spec_transform: LogMelSpectrogram,
    decoder: FireflyDecoder,
    encoder: FireflyEncoder,
}

impl FireflyCodec {
    pub fn load(cfg: FireflyConfig, vb: VarBuilder, version: WhichFishVersion) -> Result<Self> {
        let decoder = FireflyDecoder::load(vb.clone(), cfg.clone(), &version)?;
        let encoder = FireflyEncoder::load(vb.clone(), cfg.clone(), &version)?;

        let spec_transform = LogMelSpectrogram::load(LogMelSpectrogramConfig::default())?;
        let sample_rate = spec_transform.sample_rate as u32;

        Ok(FireflyCodec {
            cfg,
            sample_rate,
            decoder,
            encoder,
            spec_transform,
        })
    }

    pub fn encode(&self, input: &Tensor) -> Result<Tensor> {
        let mels = self.spec_transform.forward(input)?;
        self.encoder.encode(&mels)
    }

    /// Decodes Firefly codes into 44.1KHz mono F32 PCM audio
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let feature_lengths =
            Tensor::from_slice(&[codes.dim(D::Minus1)? as u32], 1, &self.decoder.device)?;
        self.decoder
            .decode(codes, &feature_lengths)?
            .to_dtype(DType::F32)
    }
}
