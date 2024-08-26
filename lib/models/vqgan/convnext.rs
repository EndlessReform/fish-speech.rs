// Original convnext implementation: https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/convnext.rs
// convnext implementation from fish speech: https://github.com/fishaudio/fish-speech/blob/main/fish_speech/models/vqgan/modules/firefly.py
use candle_core::{Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, Module, VarBuilder};

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

    fn forward(&self, x: &Tensor, apply_residual: bool) -> Result<Tensor> {
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
        if apply_residual {
            x = input.add(&x)?;
        }

        Ok(x)
    }
}
