pub mod convnext;
mod fsq;
mod grouped_residual_fsq;
pub mod quantizer;

use anyhow::Result as AnyhowResult;
use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};
use convnext::{ConvNeXtEncoder, ConvNeXtEncoderConfig};
use quantizer::{DownsampleFSQConfig, DownsampleFiniteScalarQuantizer};

#[derive(Clone)]
/// Incomplete. Will go back and fix it soon
pub struct Config {
    input_channels: usize,
    depths: [usize; 4],
    dims: [usize; 4],
    kernel_size: usize,
}

impl Config {
    /// Config from Fish Speech 1.2 SFT
    pub fn fish_1_2() -> Self {
        Self {
            input_channels: 160,
            depths: [3, 3, 9, 3],
            dims: [128, 256, 384, 512],
            // drop_path_rate: 0.2,
            kernel_size: 7,
        }
    }
}

pub struct FireflyArchitecture {
    backbone: ConvNeXtEncoder,
    quantizer: DownsampleFiniteScalarQuantizer,
}

impl FireflyArchitecture {
    pub fn load(vb: VarBuilder) -> AnyhowResult<Self> {
        let config = Config::fish_1_2();
        // TODO: This will have to be fixed w/ rest of config
        let backbone = ConvNeXtEncoder::load(
            vb.pp("backbone"),
            &ConvNeXtEncoderConfig {
                input_channels: config.input_channels,
                dims: config.dims.to_vec(),
                depths: config.depths.to_vec(),
                kernel_size: config.kernel_size,
                ..Default::default()
            },
        )?;
        let quantizer = DownsampleFiniteScalarQuantizer::load(
            vb.pp("quantizer"),
            // TODO: Parameterize this
            DownsampleFSQConfig::firefly_1_2(),
        )?;
        Ok(Self {
            backbone,
            quantizer,
        })
    }

    /// Unlike upstream implementation, requires MEL binning beforehand
    pub fn encode(self, mel: &Tensor) -> Result<Tensor> {
        // mel.write_npy("spec_transform.npy")?;
        let encoded_features = self.backbone.forward(&mel)?;
        // encoded_features.write_npy("backbone.npy")?;
        self.quantizer.encode(&encoded_features)
    }
}
