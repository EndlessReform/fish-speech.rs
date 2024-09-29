use super::config::{DownsampleFSQConfig, WhichModel};
use super::convnext::{ConvNeXtBlock, ConvNeXtBlockConfig};
use super::grouped_residual_fsq::{GroupedResidualFSQ, GroupedResidualFSQConfig};
use super::utils::{FishConvNet, FishTransConvNet};
use candle_core::{Result, Tensor};
use candle_nn::{seq, Conv1dConfig, ConvTranspose1dConfig, Module, Sequential, VarBuilder};

pub struct DownsampleFiniteScalarQuantizer {
    downsample: Sequential,
    _upsample: Sequential,
    residual_fsq: GroupedResidualFSQ,
    pub downsample_factor: Vec<usize>,
}

impl DownsampleFiniteScalarQuantizer {
    pub fn load(vb: VarBuilder, config: &DownsampleFSQConfig, model: &WhichModel) -> Result<Self> {
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

        let mut downsample = seq();
        let vb_ds = vb.pp("downsample");
        for (idx, factor) in config.downsample_factor.iter().enumerate() {
            let in_channels = all_dims[idx];
            let out_channels = all_dims[idx + 1];

            let mut layer = seq();
            layer = layer.add(FishConvNet::load(
                vb_ds.pp(format!("{}.0", idx)),
                in_channels,
                out_channels,
                *factor,
                Conv1dConfig {
                    stride: *factor,
                    ..Default::default()
                },
                model,
            )?);
            layer = layer.add(ConvNeXtBlock::load(
                vb_ds.pp(&format!("{}.1", idx)),
                &ConvNeXtBlockConfig::with_dim(out_channels),
                &model,
            )?);
            downsample = downsample.add(layer);
        }

        let mut upsample_layers: Vec<Sequential> = vec![];
        let vb_us = vb.pp("upsample");
        for (idx, factor) in config.downsample_factor.iter().enumerate().rev() {
            let in_channels = all_dims[idx + 1];
            let out_channels = all_dims[idx];

            let mut layer = seq();
            layer = layer.add(FishTransConvNet::load(
                vb_us.pp(format!("{}.0", idx)),
                in_channels,
                out_channels,
                *factor,
                ConvTranspose1dConfig {
                    stride: *factor,
                    ..Default::default()
                },
                model,
            )?);
            layer = layer.add(ConvNeXtBlock::load(
                vb_us.pp(&format!("{}.1", idx)),
                &ConvNeXtBlockConfig::with_dim(in_channels),
                &model,
            )?);
            upsample_layers.push(layer);
        }
        let upsample = upsample_layers
            .into_iter()
            .rev()
            .fold(seq(), |acc, layer| acc.add(layer));

        Ok(Self {
            residual_fsq,
            downsample,
            downsample_factor: config.downsample_factor.clone(),
            _upsample: upsample,
        })
    }

    pub fn encode(&self, z: &Tensor) -> Result<Tensor> {
        let z = self.downsample.forward(z)?;

        // Transpose z (equivalent to .mT in Python)
        let z_t = z.transpose(1, 2)?;

        // Apply residual_fsq
        let (_codes, indices) = self.residual_fsq.forward(&z_t)?;

        // Rearrange indices
        // Original: indices = rearrange(indices, "g b l r -> b (g r) l")
        // We need to do this manually in Rust
        let (g, b, l, r) = indices.dims4()?;
        let indices = indices.permute((1, 0, 3, 2))?; // b g r l
        let indices = indices.reshape((b, g * r, l))?;

        Ok(indices)
    }

    pub fn upsample(&self, z: &Tensor) -> Result<Tensor> {
        // TODO: Residual_FSQ
        self._upsample.forward(z)
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
