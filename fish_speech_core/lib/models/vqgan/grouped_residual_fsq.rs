use super::fsq::{FSQConfig, FSQ};
use candle_core::{Module, Result, Tensor, D};
use candle_nn::{Linear, VarBuilder};

#[derive(Debug, Clone)]
pub struct ResidualFSQConfig {
    dim: usize,
    levels: Vec<u32>,
    num_quantizers: usize,
}

pub struct ResidualFSQ {
    layers: Vec<FSQ>,
    scales: Vec<f32>,
    scales_tensor: Tensor,
    codebooks: Tensor,
    num_quantizers: usize,
    project_in: Linear,
    project_out: Linear,
}

impl ResidualFSQ {
    pub fn load(vb: VarBuilder, config: &ResidualFSQConfig) -> Result<Self> {
        let num_quantizers = config.num_quantizers;
        let mut layers = Vec::with_capacity(num_quantizers);
        let codebook_dim = config.levels.len();

        // Calculate scales
        let scales: Vec<f32> = (0..num_quantizers)
            .map(|ind| {
                let base: f32 = (config.levels[0] - 1) as f32;
                base.powi(-(ind as i32))
            })
            .collect();

        let scales_tensor = Tensor::from_vec(scales.clone(), scales.len(), vb.device())?;

        // Step 2: Reshape it to broadcast correctly: (quantize_dim, 1, 1, 1)
        let scales_tensor = scales_tensor.unsqueeze(1)?.unsqueeze(1)?.unsqueeze(1)?;

        for _ in 0..num_quantizers {
            layers.push(FSQ::load(
                vb.clone(),
                &FSQConfig {
                    levels: config.levels.clone(),
                    input_dim: config.dim,
                    n_codebooks: 1,
                },
            )?);
        }
        let codebooks: Result<Vec<Tensor>> = layers.iter().map(|l| l.implicit_codebook()).collect();
        let codebooks = Tensor::stack(&codebooks?, 0)?;
        let project_in = Linear::new(
            vb.pp("project_in")
                .get((codebook_dim, config.dim), "weight")?,
            vb.pp("project_in").get(codebook_dim, "bias").ok(),
        );
        let project_out = Linear::new(
            vb.pp("project_out")
                .get((config.dim, codebook_dim), "weight")?,
            vb.pp("project_out").get(config.dim, "bias").ok(),
        );

        Ok(ResidualFSQ {
            layers,
            scales,
            scales_tensor,
            codebooks,
            num_quantizers,
            project_in,
            project_out,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = self.project_in.forward(x)?;
        let mut quantized_out = Tensor::zeros_like(&x)?;
        // Implicit first-step
        let mut residual = self.layers[0].bound(&x)?;
        let mut all_indices = Vec::with_capacity(self.num_quantizers);

        for (layer, scale) in self.layers.iter().zip(self.scales.iter()) {
            let (quantized, indices) = layer.forward(&(residual.clone() / *scale as f64)?)?;
            let quantized = (quantized * *scale as f64)?;
            residual = residual.broadcast_sub(&quantized)?;
            quantized_out = quantized_out.broadcast_add(&quantized)?;
            all_indices.push(indices);
        }
        let quantized_out = self.project_out.forward(&quantized_out)?;

        let all_indices = Tensor::stack(&all_indices, D::Minus1)?;
        Ok((quantized_out, all_indices))
    }

    fn get_codes_from_indices(&self, indices: &Tensor) -> Result<Tensor> {
        let (b, t, _) = indices.dims3()?;
        let all_codes = self.codebooks.gather(
            &indices
                .expand((b, t, self.codebooks.dim(D::Minus1)?))?
                .contiguous()?,
            1,
        )?;
        // let all_codes = self.codebooks.i(indices)?;
        all_codes.broadcast_mul(&self.scales_tensor)
    }

    pub fn get_output_from_indices(&self, indices: &Tensor) -> Result<Tensor> {
        let codes = self.get_codes_from_indices(&indices.squeeze(0)?)?;
        let codes_summed = codes.sum(0)?;
        let out = self
            .project_out
            .forward(&codes_summed.to_dtype(self.project_out.weight().dtype())?)?;
        Ok(out)
    }
}

impl Module for ResidualFSQ {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.forward(x)?.0)
    }
}

pub struct GroupedResidualFSQ {
    rvqs: Vec<ResidualFSQ>,
    pub groups: usize,
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
            };
            rvqs.push(ResidualFSQ::load(
                vb.pp(&format!("rvqs.{}", i)),
                &rvq_config,
            )?);
        }

        Ok(Self { rvqs, groups })
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
        let all_indices = Tensor::stack(&all_indices_chunks, 0)?;

        Ok((quantized, all_indices))
    }

    pub fn get_output_from_indices(&self, indices: &Tensor) -> Result<Tensor> {
        let indices = indices.chunk(self.groups, 0)?;

        let out: Result<Vec<Tensor>> = self
            .rvqs
            .iter()
            .zip(indices.iter())
            .map(|(rvq, chunk_indices)| rvq.get_output_from_indices(&chunk_indices))
            .collect();
        Tensor::cat(&out?, D::Minus1)
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
