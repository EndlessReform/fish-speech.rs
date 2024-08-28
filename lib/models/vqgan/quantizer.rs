use super::convnext::{ConvNeXtBlock, ConvNeXtBlockConfig};
use candle_core::{Result, Tensor};
use candle_nn::{
    seq, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, Sequential,
    VarBuilder,
};

pub struct DownsampleFSQConfig {
    input_dim: usize,
    n_codebooks: usize,
    n_groups: usize,
    levels: Vec<usize>,
    downsample_factor: Vec<usize>,
    downsample_dims: Option<Vec<usize>>,
}

impl DownsampleFSQConfig {
    pub fn firefly_1_2() -> Self {
        Self {
            input_dim: 512,
            n_groups: 4,
            n_codebooks: 1,
            levels: vec![8, 5, 5, 5],
            downsample_factor: vec![2],
            downsample_dims: None,
        }
    }
}

pub struct DownsampleFiniteScalarQuantizer {
    downsample: Sequential,
    upsample: Sequential,
}

impl DownsampleFiniteScalarQuantizer {
    pub fn load(vb: VarBuilder, config: DownsampleFSQConfig) -> Result<Self> {
        let all_dims: Vec<usize> = if let Some(downsample_dims) = config.downsample_dims {
            std::iter::once(config.input_dim)
                .chain(downsample_dims.into_iter())
                .collect()
        } else {
            std::iter::repeat(config.input_dim)
                .take(config.downsample_factor.len() + 1)
                .collect()
        };

        let mut downsample = seq();
        let vb_ds = vb.pp("downsample");
        for (idx, factor) in config.downsample_factor.iter().enumerate() {
            let in_channels = all_dims[idx];
            let out_channels = all_dims[idx + 1];

            let mut layer = seq();
            layer = layer.add(Conv1d::new(
                vb_ds.get(
                    (out_channels, in_channels, *factor),
                    &format!("{}.0.weight", idx),
                )?,
                vb_ds.get(out_channels, &format!("{}.0.bias", idx)).ok(),
                Conv1dConfig {
                    stride: *factor,
                    ..Default::default()
                },
            ));
            layer = layer.add(ConvNeXtBlock::load(
                vb_ds.pp(&format!("{}.1", idx)),
                &ConvNeXtBlockConfig::with_dim(out_channels),
            )?);
            downsample = downsample.add(layer);
        }

        let mut upsample = seq();
        let vb_us = vb.pp("upsample");
        for (idx, factor) in config.downsample_factor.iter().enumerate().rev() {
            let in_channels = all_dims[idx + 1];
            let out_channels = all_dims[idx];

            let mut layer = seq();
            layer = layer.add(ConvTranspose1d::new(
                vb_us.get(
                    (out_channels, in_channels, *factor),
                    &format!("{}.0.weight", idx),
                )?,
                vb_us.get(out_channels, &format!("{}.0.bias", idx)).ok(),
                ConvTranspose1dConfig {
                    stride: *factor,
                    ..Default::default()
                },
            ));
            layer = layer.add(ConvNeXtBlock::load(
                vb_us.pp(&format!("{}.1", idx)),
                &ConvNeXtBlockConfig::with_dim(in_channels),
            )?);
            upsample = upsample.add(layer);
        }

        Ok(Self {
            downsample,
            upsample,
        })
    }

    pub fn downsample(self, z: &Tensor) -> Result<Tensor> {
        // TODO: Residual_FSQ
        self.downsample.forward(z)
    }

    pub fn upsample(self, z: &Tensor) -> Result<Tensor> {
        // TODO: Residual_FSQ
        self.upsample.forward(z)
    }
}
