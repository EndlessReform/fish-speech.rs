use super::config::FireflyConfig;
use super::convnext::{ConvNeXtEncoder, ConvNeXtEncoderConfig};
use super::quantizer::DownsampleFiniteScalarQuantizer;
use crate::config::WhichFishVersion;
use candle_core::{DType, Result, Tensor};
use candle_nn::{Module, VarBuilder};

pub struct FireflyEncoder {
    backbone: ConvNeXtEncoder,
    quantizer: DownsampleFiniteScalarQuantizer,
    dtype: DType,
}

impl FireflyEncoder {
    pub fn load(vb: VarBuilder, cfg: FireflyConfig, model: &WhichFishVersion) -> Result<Self> {
        // TODO: This will have to be fixed w/ rest of config
        let backbone = ConvNeXtEncoder::load(
            vb.pp("backbone"),
            &ConvNeXtEncoderConfig {
                input_channels: cfg.backbone.input_channels,
                dims: cfg.backbone.dims.to_vec(),
                depths: cfg.backbone.depths.to_vec(),
                kernel_size: cfg.backbone.kernel_size,
                ..Default::default()
            },
            model,
        )?;
        let quantizer =
            DownsampleFiniteScalarQuantizer::load(vb.pp("quantizer"), &cfg.quantizer, model)?;
        Ok(Self {
            backbone,
            quantizer,
            dtype: vb.dtype(),
        })
    }

    /// Unlike upstream implementation, requires MEL binning beforehand
    pub fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        let mel = mel.to_dtype(self.dtype)?;
        let encoded_features = self.backbone.forward(&mel)?;
        self.quantizer.encode(&encoded_features)
    }
}
