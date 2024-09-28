use super::config::FireflyConfig;
use super::convnext::{ConvNeXtEncoder, ConvNeXtEncoderConfig};
use super::quantizer::DownsampleFiniteScalarQuantizer;
use anyhow::Result as AnyhowResult;
use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

pub struct FireflyEncoder {
    backbone: ConvNeXtEncoder,
    quantizer: DownsampleFiniteScalarQuantizer,
}

impl FireflyEncoder {
    pub fn load(vb: VarBuilder, cfg: &FireflyConfig) -> AnyhowResult<Self> {
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
        )?;
        let quantizer = DownsampleFiniteScalarQuantizer::load(vb.pp("quantizer"), &cfg.quantizer)?;
        Ok(Self {
            backbone,
            quantizer,
        })
    }

    /// Unlike upstream implementation, requires MEL binning beforehand
    pub fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        // mel.write_npy("spec_transform.npy")?;
        let encoded_features = self.backbone.forward(&mel)?;
        // encoded_features.write_npy("backbone.npy")?;
        self.quantizer.encode(&encoded_features)
    }
}
