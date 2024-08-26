// Original convnext implementation: https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/convnext.rs
// convnext implementation from fish speech: https://github.com/fishaudio/fish-speech/blob/main/fish_speech/models/vqgan/modules/firefly.py
use candle_core::{Error, Result, Tensor, Var};
use candle_nn::{seq, Conv1d, Conv1dConfig, LayerNorm, Linear, Module, Sequential, VarBuilder};

#[derive(Clone)]
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

struct ConvNeXtBlockConfig {
    pub dim: usize,
    pub drop_path: f64,
    pub layer_scale_init_value: f64,
    pub mlp_ratio: usize,
    pub kernel_size: usize,
    pub dilation: usize,
}

impl Default for ConvNeXtBlockConfig {
    fn default() -> Self {
        ConvNeXtBlockConfig {
            dim: 0, // Must be set explicitly
            drop_path: 0.0,
            layer_scale_init_value: 1e-6,
            mlp_ratio: 4,
            kernel_size: 7,
            dilation: 1,
        }
    }
}

impl ConvNeXtBlockConfig {
    pub fn with_dim(dim: usize) -> Self {
        let mut config = Self::default();
        config.dim = dim;
        config
    }
}

#[derive(Clone)]
struct ConvNeXtBlock {
    dwconv: Conv1d,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Option<Tensor>,
}

impl ConvNeXtBlock {
    fn load(vb: VarBuilder, config: &ConvNeXtBlockConfig) -> Result<ConvNeXtBlock> {
        let dwconv = Conv1d::new(
            vb.get(
                (config.dim, 1 as usize, config.kernel_size),
                "dwconv.weight",
            )?,
            Some(vb.get(config.dim, "dwconv.bias")?),
            Conv1dConfig {
                padding: (config.dilation as f64 * (config.kernel_size as f64 - 1.0) / 2.0).round()
                    as usize,
                groups: config.dim,
                dilation: config.dilation,
                stride: 1,
            },
        );
        let norm = LayerNorm::new(
            vb.get(config.dim, "norm.weight")?,
            vb.get(config.dim, "norm.bias")?,
            1e-6,
        );
        let pwconv1 = Linear::new(
            vb.get(
                (config.dim, config.dim * config.mlp_ratio),
                "pwconv1.weight",
            )?,
            Some(vb.get(config.dim * config.mlp_ratio, "pwconv1.bias")?),
        );
        // Note to self: you can just call `.gelu()` on a tensor
        let pwconv2 = Linear::new(
            vb.get(
                (config.dim * config.mlp_ratio, config.dim),
                "pwconv2.weight",
            )?,
            Some(vb.get(config.dim, "pwconv2.bias")?),
        );
        let gamma: Option<Tensor> = vb.get(config.dim, "gamma").ok();

        // Ignoring DropPath until training is implemented
        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
        })
    }
}

impl Module for ConvNeXtBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input = x;
        let x = self.dwconv.forward(x)?;
        let x = x.permute((0, 2, 1))?;
        let x = self.norm.forward(&x)?;
        let x = self.pwconv1.forward(&x)?.gelu()?;
        let mut x = self.pwconv2.forward(&x)?;

        if let Some(gamma) = &self.gamma {
            x = gamma.mul(&x)?;
        }
        x = x.permute((0, 2, 1))?;
        // I will come to regret forcing applying residual...
        x = input.add(&x)?;

        Ok(x)
    }
}

struct LayerNormChannelsFirst {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNormChannelsFirst {
    fn load(vb: VarBuilder, len: usize) -> Result<Self> {
        let eps = 1e-6;
        let weight = vb.get(len, "weight")?;
        let bias = vb.get(len, "bias")?;
        Ok(Self { eps, weight, bias })
    }
}

impl Module for LayerNormChannelsFirst {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let u = xs.mean_keepdim(1)?;
        let s = (xs.sub(&u)?).powf(2.0)?.mean_keepdim(1)?;
        let x = xs.sub(&u)?.div(&(&s + self.eps)?.sqrt()?)?;
        let x = x.broadcast_mul(&self.weight)?;
        x.broadcast_add(&self.bias)
    }
}

pub struct ConvNeXtEncoderConfig {
    input_channels: usize,
    depths: Vec<usize>,
    dims: Vec<usize>,
    drop_path_rate: f64,
    layer_scale_init_value: f64,
    kernel_size: usize,
}

impl Default for ConvNeXtEncoderConfig {
    fn default() -> Self {
        Self {
            input_channels: 3,
            depths: vec![3, 3, 9, 3],
            dims: vec![96, 192, 384, 768],
            drop_path_rate: 0.0,
            layer_scale_init_value: 1e-6,
            kernel_size: 7,
        }
    }
}

pub struct ConvNeXtEncoder {
    downsample_layers: Vec<Box<dyn Module>>,
    stages: Vec<Box<dyn Module>>,
    norm: LayerNormChannelsFirst,
}

impl ConvNeXtEncoder {
    fn load(vb: VarBuilder, config: &ConvNeXtEncoderConfig) -> Result<ConvNeXtEncoder> {
        if config.depths.len() != config.dims.len() {
            return Err(Error::debug(format!(
                "ConvNeXtEncoder depths and dims do not match: {}, {}",
                config.depths.len(),
                config.dims.len()
            )));
        } else if config.depths.len() == 0 {
            return Err(Error::debug("ConvNeXtEncoder depth cannot be 0"));
        };
        let vb_ds = vb.pp("downsample_layers");

        let stem = seq();
        let stem = stem.add(Conv1d::new(
            vb_ds.get(
                (config.input_channels, config.dims[0], config.kernel_size),
                "0.0.weight",
            )?,
            vb_ds.get(config.dims[0], "0.0.bias").ok(),
            Conv1dConfig {
                padding: config.kernel_size / 2,
                stride: 1,
                groups: 1,
                dilation: 1,
            },
        ));
        let stem = stem.add(LayerNormChannelsFirst::load(vb.pp("0.1"), config.dims[0])?);
        let mut downsample_layers: Vec<Box<dyn Module>> = vec![Box::new(stem)];

        for idx in 1..config.depths.len() {
            let vb_stem = vb.pp(format!("{}", idx));
            let mid_layer = seq();
            let mid_layer = mid_layer.add(LayerNormChannelsFirst::load(
                vb_stem.pp("0"),
                config.dims[idx - 1],
            )?);

            let mid_layer = mid_layer.add(Conv1d::new(
                vb.get((config.dims[idx - 1], config.dims[idx]), "1.weight")?,
                Some(vb.get(config.dims[idx], "1.bias")?),
                Conv1dConfig {
                    stride: 1,
                    groups: 1,
                    dilation: 1,
                    padding: 0,
                },
            ));
            downsample_layers.push(Box::new(mid_layer));
        }

        let mut stages: Vec<Box<dyn Module>> = Vec::new();
        for idx in 0..config.depths.len() {
            let mut stage = seq();
            let blocks: Result<Vec<ConvNeXtBlock>> = (0..config.depths[idx])
                .map(|j| {
                    ConvNeXtBlock::load(
                        vb.pp(format!("stages.{}.{}", idx, j)),
                        &ConvNeXtBlockConfig::with_dim(config.dims[idx]),
                    )
                })
                .collect();
            for block in blocks? {
                stage = stage.add(block);
            }
            stages.push(Box::new(stage));
        }

        // We can unwrap since we already did the bounds check
        let norm = LayerNormChannelsFirst::load(vb.pp("norm"), *config.dims.last().unwrap())?;

        Ok(Self {
            downsample_layers,
            stages,
            norm,
        })
    }
}

impl Module for ConvNeXtEncoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.to_owned();
        for (downsampler, block) in self.downsample_layers.iter().zip(self.stages.iter()) {
            x = downsampler.forward(&x)?;
            x = block.forward(&x)?;
        }
        x = self.norm.forward(&x)?;
        Ok(x)
    }
}
