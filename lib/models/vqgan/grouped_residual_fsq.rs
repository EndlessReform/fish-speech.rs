use super::fsq::{FSQConfig, FSQ};
use candle_core::{Module, Result, Tensor, D};
use candle_nn::VarBuilder;

#[derive(Debug, Clone)]
pub struct ResidualFSQConfig {
    dim: usize,
    levels: Vec<u32>,
    num_quantizers: usize,
    groups: usize,
}

pub struct ResidualFSQ {
    layers: Vec<FSQ>,
    scales: Vec<Tensor>,
    num_quantizers: usize,
}

impl ResidualFSQ {
    pub fn load(vb: VarBuilder, config: &ResidualFSQConfig) -> Result<Self> {
        let device = vb.device();
        let num_quantizers = config.num_quantizers;
        let mut layers = Vec::with_capacity(num_quantizers);
        let levels_tensor = Tensor::new(config.levels.as_slice(), device)?;

        // Calculate scales
        let mut scales = Vec::with_capacity(num_quantizers);
        for ind in 0..num_quantizers {
            let scale = (&levels_tensor.sub(&Tensor::new(&[1.0], device)?)?)
                .pow(&Tensor::new(&[-(ind as i64)], vb.device())?)?;
            scales.push(scale);
        }

        for ind in 0..num_quantizers {
            layers.push(FSQ::load(
                vb.pp(format!("rvqs.{}.", ind)),
                &FSQConfig {
                    levels: config.levels.clone(),
                    input_dim: config.dim,
                    n_codebooks: 1,
                },
            )?);
        }

        Ok(ResidualFSQ {
            layers,
            scales,
            num_quantizers,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut quantized_out = Tensor::zeros_like(x)?;
        let mut residual = x.clone();
        let mut all_indices = Vec::with_capacity(self.num_quantizers);

        for (layer, scale) in self.layers.iter().zip(self.scales.iter()) {
            let scale = scale.squeeze(0)?;
            let (quantized, indices) = layer.forward(&(&residual / &scale)?)?;
            let quantized = (&quantized * &scale)?;
            residual = (&residual - quantized.detach())?;
            quantized_out = (&quantized_out + &quantized)?;
            all_indices.push(indices);
        }

        let all_indices = Tensor::stack(&all_indices, D::Minus1)?;
        Ok((quantized_out, all_indices))
    }
}

// Assuming FSQ is already implemented
impl Module for ResidualFSQ {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.forward(x)?.0)
    }
}

pub struct GroupedResidualFSQ {
    rvqs: Vec<ResidualFSQ>,
    dim: usize,
    groups: usize,
}

impl GroupedResidualFSQ {
    pub fn load(vb: VarBuilder, config: &GroupedResidualFSQConfig) -> Result<Self> {
        let dim = config.dim;
        let groups = config.groups;
        assert!(
            dim % groups == 0,
            "Dimension must be divisible by the number of groups"
        );
        let dim_per_group = dim / groups;

        let mut rvqs = Vec::with_capacity(groups);
        for i in 0..groups {
            let rvq_config = ResidualFSQConfig {
                dim: dim_per_group,
                levels: config.levels.clone(),
                num_quantizers: config.num_quantizers,
                groups: 1, // Each RVQ handles one group
            };
            rvqs.push(ResidualFSQ::load(
                vb.pp(&format!("rvqs.{}", i)),
                &rvq_config,
            )?);
        }

        Ok(Self { rvqs, dim, groups })
    }

    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        // Split the input tensor into groups
        let chunks = x.chunk(self.groups, D::Minus1)?;

        let mut quantized_chunks = Vec::with_capacity(self.groups);
        let mut all_indices_chunks = Vec::with_capacity(self.groups);

        // Apply ResidualFSQ to each group
        for (chunk, rvq) in chunks.iter().zip(self.rvqs.iter()) {
            let (quantized, indices) = rvq.forward(chunk)?;
            quantized_chunks.push(quantized);
            all_indices_chunks.push(indices);
        }

        // Combine the results
        let quantized = Tensor::cat(&quantized_chunks, D::Minus1)?;
        let all_indices = Tensor::stack(&all_indices_chunks, D::Minus1)?;

        Ok((quantized, all_indices))
    }
}

#[derive(Debug, Clone)]
pub struct GroupedResidualFSQConfig {
    pub dim: usize,
    pub levels: Vec<u32>,
    pub num_quantizers: usize,
    pub groups: usize,
}

impl Module for GroupedResidualFSQ {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.forward(x)?.0)
    }
}
