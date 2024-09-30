// Original convnext implementation: https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/convnext.rs
// convnext implementation from fish speech: https://github.com/fishaudio/fish-speech/blob/main/fish_speech/models/vqgan/modules/firefly.py
use super::{config::WhichModel, utils::FishConvNet};
use candle_core::{Error, Result, Tensor};
use candle_nn::{seq, Conv1d, Conv1dConfig, LayerNorm, Linear, Module, VarBuilder};

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
        model: &WhichModel,
    ) -> Result<ConvNeXtBlock> {
        let dwconv = FishConvNet::load(
            vb.pp("dwconv"),
            1_usize,
            config.dim,
            config.kernel_size,
            Conv1dConfig {
                padding: match model {
                    WhichModel::Fish1_2 => {
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

pub struct ConvNeXtEncoder {
    downsample_layers: Vec<Box<dyn Module>>,
    stages: Vec<Box<dyn Module>>,
    norm: LayerNormChannelsFirst,
}

impl ConvNeXtEncoder {
    pub fn load(
        vb: VarBuilder,
        config: &ConvNeXtEncoderConfig,
        model: &WhichModel,
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
        let vb_ds = vb.pp("downsample_layers");

        let stem = seq();
        let stem = stem.add(FishConvNet::load(
            vb_ds.pp("0.0"),
            config.input_channels,
            config.dims[0],
            config.kernel_size,
            Conv1dConfig {
                padding: match model {
                    WhichModel::Fish1_2 => config.kernel_size / 2,
                    _ => 0,
                },
                ..Default::default()
            },
            model,
        )?);
        let stem = stem.add(LayerNormChannelsFirst::load(
            vb_ds.pp("0.1"),
            config.dims[0],
        )?);
        let mut downsample_layers: Vec<Box<dyn Module>> = vec![Box::new(stem)];

        for idx in 1..config.depths.len() {
            let vb_stem = vb_ds.pp(format!("{}", idx));
            let mid_layer = seq();
            let mid_layer = mid_layer.add(LayerNormChannelsFirst::load(
                vb_stem.pp("0"),
                config.dims[idx - 1],
            )?);

            let mid_layer = mid_layer.add(Conv1d::new(
                vb_stem.get((config.dims[idx], config.dims[idx - 1], 1), "1.weight")?,
                Some(vb_stem.get(config.dims[idx], "1.bias")?),
                Conv1dConfig {
                    ..Default::default()
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
                        model,
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
