use super::config::DownsampleFSQConfig;
use super::convnext::{ConvNeXtBlock, ConvNeXtBlockConfig};
use super::grouped_residual_fsq::{GroupedResidualFSQ, GroupedResidualFSQConfig};
use super::utils::{FishConvNet, FishTransConvNet};
use crate::config::WhichFishVersion;
use candle_core::{Result, Tensor};
use candle_nn::{Conv1dConfig, ConvTranspose1dConfig, Module, VarBuilder};

pub struct DownsampleFiniteScalarQuantizer {
    downsample_layers: Vec<(FishConvNet, ConvNeXtBlock)>,
    upsample_layers: Vec<(FishTransConvNet, ConvNeXtBlock)>,
    residual_fsq: GroupedResidualFSQ,
    pub downsample_factor: Vec<usize>,
}

impl DownsampleFiniteScalarQuantizer {
    pub fn load(
        vb: VarBuilder,
        config: &DownsampleFSQConfig,
        model: &WhichFishVersion,
    ) -> Result<Self> {
        let all_dims: Vec<usize> = if let Some(downsample_dims) = config.downsample_dims.clone() {
            std::iter::once(config.input_dim)
                .chain(downsample_dims.into_iter())
                .collect()
        } else {
            std::iter::repeat(config.input_dim)
                .take(config.downsample_factor.len() + 1)
                .collect()
        };

        let residual_fsq = GroupedResidualFSQ::load(
            vb.pp("residual_fsq"),
            &GroupedResidualFSQConfig {
                dim: *all_dims.last().unwrap(),
                levels: config.levels.clone(),
                num_quantizers: config.n_codebooks,
                groups: config.n_groups,
            },
        )?;

        let mut downsample_layers = Vec::new();
        let vb_ds = vb.pp("downsample");
        for (idx, factor) in config.downsample_factor.iter().enumerate() {
            let in_channels = all_dims[idx];
            let out_channels = all_dims[idx + 1];

            let conv = FishConvNet::load(
                vb_ds.pp(format!("{}.0", idx)),
                in_channels,
                out_channels,
                *factor,
                Conv1dConfig {
                    stride: *factor,
                    ..Default::default()
                },
                model,
            )?;

            let block = ConvNeXtBlock::load(
                vb_ds.pp(&format!("{}.1", idx)),
                &ConvNeXtBlockConfig::with_dim(out_channels),
                &model,
            )?;

            downsample_layers.push((conv, block));
        }

        let mut upsample_layers = Vec::new();
        let vb_us = vb.pp("upsample");
        for (idx, factor) in config.downsample_factor.iter().enumerate().rev() {
            let in_channels = all_dims[idx + 1];
            let out_channels = all_dims[idx];

            let conv = FishTransConvNet::load(
                vb_us.pp(format!("{}.0", idx)),
                in_channels,
                out_channels,
                *factor,
                ConvTranspose1dConfig {
                    stride: *factor,
                    ..Default::default()
                },
                model,
            )?;

            let block = ConvNeXtBlock::load(
                vb_us.pp(&format!("{}.1", idx)),
                &ConvNeXtBlockConfig::with_dim(in_channels),
                &model,
            )?;

            upsample_layers.push((conv, block));
        }

        Ok(Self {
            residual_fsq,
            downsample_layers,
            upsample_layers,
            downsample_factor: config.downsample_factor.clone(),
        })
    }

    pub fn encode(&self, z: &Tensor) -> Result<Tensor> {
        let mut z = z.clone();
        // Apply each downsample layer pair in sequence
        for (conv, block) in &self.downsample_layers {
            z = conv.forward(&z)?;
            z = block.forward(&z)?;
        }

        // Transpose z (equivalent to .mT in Python)
        let z_t = z.transpose(1, 2)?;

        // Apply residual_fsq
        let (_codes, indices) = self.residual_fsq.forward(&z_t)?;

        // Rearrange indices
        let (g, b, l, r) = indices.dims4()?;
        let indices = indices.permute((1, 0, 3, 2))?; // b g r l
        let indices = indices.reshape((b, g * r, l))?;

        Ok(indices)
    }

    pub fn upsample(&self, z: &Tensor) -> Result<Tensor> {
        let mut z = z.clone();
        for (conv, block) in self.upsample_layers.iter().rev() {
            z = conv.forward(&z)?;
            z = block.forward(&z)?;
        }
        Ok(z)
    }

    pub fn decode(&self, indices: &Tensor) -> Result<Tensor> {
        // b (gr) l -> g b l r
        let (b, gr, l) = indices.dims3()?;
        let indices = indices.reshape((
            self.residual_fsq.groups,
            b,
            l,
            gr / self.residual_fsq.groups,
        ))?;
        let z_q = self.residual_fsq.get_output_from_indices(&indices)?;
        self.upsample(&z_q.transpose(1, 2)?)
    }
}
