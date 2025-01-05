// Original convnext implementation: https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/convnext.rs
// convnext implementation from fish speech: https://github.com/fishaudio/fish-speech/blob/main/fish_speech/models/vqgan/modules/firefly.py
use super::utils::FishConvNet;
use crate::config::WhichFishVersion;
use candle_core::{Error, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, Module, VarBuilder};

pub struct ConvNeXtBlockConfig {
    pub dim: usize,
    /// Unused except for training
    pub _drop_path: f64,
    /// Unused except for training
    pub _layer_scale_init_value: f64,
    pub mlp_ratio: usize,
    pub kernel_size: usize,
    pub dilation: usize,
}

impl Default for ConvNeXtBlockConfig {
    fn default() -> Self {
        ConvNeXtBlockConfig {
            dim: 0, // Must be set explicitly
            _drop_path: 0.0,
            _layer_scale_init_value: 1e-6,
            mlp_ratio: 4,
            kernel_size: 7,
            dilation: 1,
        }
    }
}

impl ConvNeXtBlockConfig {
    pub fn with_dim(dim: usize) -> Self {
        Self {
            dim,
            ..Default::default()
        }
    }
}

#[derive(Clone)]
pub struct ConvNeXtBlock {
    dwconv: FishConvNet,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Option<Tensor>,
}

impl ConvNeXtBlock {
    pub fn load(
        vb: VarBuilder,
        config: &ConvNeXtBlockConfig,
        model: &WhichFishVersion,
    ) -> Result<ConvNeXtBlock> {
        let dwconv = FishConvNet::load(
            vb.pp("dwconv"),
            1_usize,
            config.dim,
            config.kernel_size,
            Conv1dConfig {
                padding: match model {
                    WhichFishVersion::Fish1_2 => {
                        (config.dilation as f64 * (config.kernel_size as f64 - 1.0) / 2.0).round()
                            as usize
                    }
                    _ => 0,
                },
                groups: config.dim,
                dilation: config.dilation,
                stride: 1,
            },
            &model,
        )?;
        // println!("In: {}, out: {}, kernel: {}, dilation")
        let norm = LayerNorm::new(
            vb.get(config.dim, "norm.weight")?,
            vb.get(config.dim, "norm.bias")?,
            1e-6,
        );
        // These are saved in the weight dict as 1d convolutions for some stupid reason
        let pwconv1 = Linear::new(
            vb.get(
                (config.dim * config.mlp_ratio, config.dim),
                "pwconv1.weight",
            )?,
            Some(vb.get(config.dim * config.mlp_ratio, "pwconv1.bias")?),
        );
        let pwconv2 = Linear::new(
            vb.get(
                (config.dim, config.dim * config.mlp_ratio),
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
            x = gamma.broadcast_mul(&x)?;
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
        let s = (xs.broadcast_sub(&u)?).powf(2.0)?.mean_keepdim(1)?;
        let x = xs
            .broadcast_sub(&u)?
            .broadcast_div(&(&s + self.eps)?.sqrt()?)?;
        let x = x.broadcast_mul(&self.weight.unsqueeze(1)?)?;
        x.broadcast_add(&self.bias.unsqueeze(1)?)
    }
}

pub struct ConvNeXtEncoderConfig {
    pub input_channels: usize,
    pub depths: Vec<usize>,
    pub dims: Vec<usize>,
    pub kernel_size: usize,
    /// Unused except for training
    pub _drop_path_rate: f64,
    /// Unused except for training
    pub _layer_scale_init_value: f64,
}

impl Default for ConvNeXtEncoderConfig {
    fn default() -> Self {
        Self {
            input_channels: 3,
            depths: vec![3, 3, 9, 3],
            dims: vec![96, 192, 384, 768],
            _drop_path_rate: 0.0,
            _layer_scale_init_value: 1e-6,
            kernel_size: 7,
        }
    }
}

pub struct StemLayer {
    conv: FishConvNet,
    norm: LayerNormChannelsFirst,
    blocks: Vec<ConvNeXtBlock>,
}

impl StemLayer {
    pub fn load(
        vb: VarBuilder,
        config: &ConvNeXtEncoderConfig,
        model: &WhichFishVersion,
    ) -> Result<Self> {
        let vb_ds = vb.pp("downsample_layers");
        let conv = FishConvNet::load(
            vb_ds.pp("0.0"),
            config.input_channels,
            config.dims[0],
            config.kernel_size,
            Conv1dConfig {
                padding: match model {
                    WhichFishVersion::Fish1_2 => config.kernel_size / 2,
                    _ => 0,
                },
                ..Default::default()
            },
            model,
        )?;
        let norm = LayerNormChannelsFirst::load(vb_ds.pp("0.1"), config.dims[0])?;
        let blocks: Result<Vec<ConvNeXtBlock>> = (0..config.depths[0])
            .map(|j| {
                ConvNeXtBlock::load(
                    vb.pp(format!("stages.{}.{}", 0, j)),
                    &ConvNeXtBlockConfig::with_dim(config.dims[0]),
                    model,
                )
            })
            .collect();
        Ok(Self {
            conv,
            norm,
            blocks: blocks?,
        })
    }
}
impl Module for StemLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(xs)?;
        let norm = self.norm.forward(&x);
        let out = self
            .blocks
            .iter()
            .try_fold(norm, |acc, block| acc.map(|x| block.forward(&x)))?;
        out
    }
}

pub struct MidLayer {
    norm: LayerNormChannelsFirst,
    conv: Conv1d,
    blocks: Vec<ConvNeXtBlock>,
}

impl MidLayer {
    pub fn load(
        vb: VarBuilder,
        config: &ConvNeXtEncoderConfig,
        model: &WhichFishVersion,
        idx: usize,
    ) -> Result<Self> {
        let vb_stem = vb.pp(format!("downsample_layers.{}", idx));
        let norm = LayerNormChannelsFirst::load(vb_stem.pp("0"), config.dims[idx - 1])?;
        let conv = Conv1d::new(
            vb_stem.get((config.dims[idx], config.dims[idx - 1], 1), "1.weight")?,
            Some(vb_stem.get(config.dims[idx], "1.bias")?),
            Conv1dConfig {
                ..Default::default()
            },
        );
        let blocks: Result<Vec<ConvNeXtBlock>> = (0..config.depths[idx])
            .map(|j| {
                ConvNeXtBlock::load(
                    vb.pp(format!("stages.{}.{}", idx, j)),
                    &ConvNeXtBlockConfig::with_dim(config.dims[idx]),
                    model,
                )
            })
            .collect();
        Ok(Self {
            norm,
            conv,
            blocks: blocks?,
        })
    }
}

impl Module for MidLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let norm = self.norm.forward(&xs)?;
        let maybe_conv = self.conv.forward(&norm);
        let out = self
            .blocks
            .iter()
            .try_fold(maybe_conv, |acc, block| acc.map(|acc| block.forward(&acc)))?;
        out
    }
}

pub struct ConvNeXtEncoder {
    stem_layer: StemLayer,
    mid_layers: Vec<MidLayer>,
    norm: LayerNormChannelsFirst,
}

impl ConvNeXtEncoder {
    pub fn load(
        vb: VarBuilder,
        config: &ConvNeXtEncoderConfig,
        model: &WhichFishVersion,
    ) -> Result<ConvNeXtEncoder> {
        if config.depths.len() != config.dims.len() {
            return Err(Error::debug(format!(
                "ConvNeXtEncoder depths and dims do not match: {}, {}",
                config.depths.len(),
                config.dims.len()
            )));
        } else if config.depths.is_empty() {
            return Err(Error::debug("ConvNeXtEncoder depth cannot be 0"));
        };
        let stem_layer = StemLayer::load(vb.clone(), config, model)?;
        let mid_layers: Result<Vec<MidLayer>> = (1..config.depths.len())
            .map(|idx| MidLayer::load(vb.clone(), config, model, idx))
            .collect();

        let mid_layers = mid_layers?;
        // We can unwrap since we already did the bounds check
        let norm = LayerNormChannelsFirst::load(vb.pp("norm"), *config.dims.last().unwrap())?;

        Ok(Self {
            stem_layer,
            mid_layers,
            norm,
        })
    }
}

impl Module for ConvNeXtEncoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.stem_layer.forward(xs);
        let x = self
            .mid_layers
            .iter()
            .try_fold(x, |acc, layer| acc.map(|x| layer.forward(&x)))?;
        let x = self.norm.forward(&(x?));
        x
    }
}
