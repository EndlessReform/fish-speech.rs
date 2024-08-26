pub mod convnext;
use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};
use convnext::{ConvNeXtEncoder, ConvNeXtEncoderConfig};

#[derive(Clone)]
/// Incomplete. Will go back and fix it soon
pub struct Config {
    input_channels: usize,
    depths: [usize; 4],
    dims: [usize; 4],
    kernel_size: usize,
    drop_path_rate: f64,
}

impl Config {
    /// Config from Fish Speech 1.2 SFT
    pub fn fish_1_2() -> Self {
        Self {
            input_channels: 160,
            depths: [3, 3, 9, 3],
            dims: [128, 256, 384, 512],
            drop_path_rate: 0.2,
            kernel_size: 7,
        }
    }
}

pub struct FireflyArchitecture {
    backbone: ConvNeXtEncoder,
}

impl FireflyArchitecture {
    pub fn load(vb: VarBuilder) -> Result<Self> {
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
        Ok(Self { backbone })
    }

    pub fn encode(self, audios: &Tensor) -> Result<Tensor> {
        // TODO: everything basically
        self.backbone.forward(audios)
    }
}