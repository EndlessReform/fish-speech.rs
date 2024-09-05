use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::{Linear, VarBuilder};

#[derive(Debug, Clone)]
pub struct FSQConfig {
    pub levels: Vec<u32>,
    pub n_codebooks: usize,
    pub input_dim: usize,
}

#[derive(Debug, Clone)]
pub struct FSQ {
    levels: Tensor,
    basis: Tensor,
    project_in: Linear,
    project_out: Linear,
    codebook_dim: usize,
    effective_codebook_dim: usize,
}

// This is not implemented as a unary op in Candle, for some reason
fn tan(x: &Tensor) -> Result<Tensor> {
    let sin_x = x.sin()?;
    let cos_x = x.cos()?;
    sin_x.div(&cos_x)
}

// This is also not implemented as a unary op
fn atanh(x: &Tensor) -> Result<Tensor> {
    // atanh(x) = 0.5 * ln((1 + x) / (1 - x))
    let one = Tensor::ones_like(x)?;
    let numerator = one.add(x)?;
    let denominator = one.sub(x)?;
    numerator
        .div(&denominator)?
        .log()?
        .mul(&Tensor::full(0.5, x.shape(), x.device())?)
}

// This is also not implemented as a binary op. Don't ask me why
fn remainder(x: &Tensor, y: &Tensor) -> Result<Tensor> {
    let quotient = x.div(y)?.floor()?;
    x.sub(&quotient.mul(y)?)
}

impl FSQ {
    pub fn load(vb: VarBuilder, config: &FSQConfig) -> Result<Self> {
        let device = vb.device();
        let FSQConfig {
            levels,
            n_codebooks,
            input_dim,
        } = config;
        let codebook_dim = levels.len();
        let effective_codebook_dim = codebook_dim * n_codebooks;

        let levels = Tensor::from_slice(levels.as_slice(), 1, device)?;
        let mut basis = Vec::with_capacity(codebook_dim);
        basis.push(1);
        for i in 1..codebook_dim {
            basis.push(basis[i - 1] * config.levels[i - 1]);
        }
        let basis = Tensor::new(basis.as_slice(), device)?;

        let project_in = Linear::new(
            vb.pp("project_in")
                .get((effective_codebook_dim, *input_dim), "weight")?,
            vb.pp("project_in").get(effective_codebook_dim, "bias").ok(),
        );
        let project_out = Linear::new(
            vb.pp("project_out")
                .get((*input_dim, effective_codebook_dim), "weight")?,
            vb.pp("project_out").get(*input_dim, "bias").ok(),
        );
        Ok(Self {
            levels,
            basis,
            project_in,
            project_out,
            codebook_dim,
            effective_codebook_dim,
        })
    }

    pub fn bound(&self, z: &Tensor) -> Result<Tensor> {
        let levels_sub_1 = self.levels.sub(&Tensor::ones_like(&self.levels)?)?;
        let half_l = levels_sub_1
            .broadcast_mul(&Tensor::full(
                0.999,
                self.levels.shape(),
                self.levels.device(),
            )?)?
            .div(&Tensor::full(
                2.0,
                self.levels.shape(),
                self.levels.device(),
            )?)?;

        let two = Tensor::full(2.0, self.levels.shape(), self.levels.device())?;
        let remainder = remainder(&self.levels, &two)?;

        let offset = remainder
            .eq(&Tensor::zeros_like(&self.levels)?)?
            .to_dtype(self.levels.dtype())?
            .mul(&Tensor::full(
                0.5,
                self.levels.shape(),
                self.levels.device(),
            )?)?;

        let shift = tan(&offset.div(&half_l)?)?;
        z.add(&shift)?.tanh()?.mul(&half_l)?.sub(&offset)
    }

    pub fn quantize(&self, z: &Tensor) -> Result<Tensor> {
        let quantized = self.bound(z)?.round()?;
        let half_width = self.levels.div(&Tensor::full(
            2.0,
            self.levels.shape(),
            self.levels.device(),
        )?)?;
        quantized.div(&half_width)
    }

    pub fn forward(&self, z: &Tensor) -> Result<(Tensor, Tensor)> {
        let z = self.project_in.forward(z)?;
        let z = z.reshape((
            (),
            self.effective_codebook_dim / self.codebook_dim,
            self.codebook_dim,
        ))?;

        let codes = self.quantize(&z)?;
        let indices = codes
            .mul(&self.basis.unsqueeze(0)?.unsqueeze(0)?)?
            .sum(2)?
            .to_dtype(DType::I64)?;

        let out = codes.reshape(((), self.effective_codebook_dim))?;
        let out = self.project_out.forward(&out)?;

        Ok((out, indices))
    }

    pub fn indices_to_codes(&self, indices: &Tensor) -> Result<Tensor> {
        let level_indices = self.indices_to_level_indices(indices)?;
        self._scale_and_shift_inverse(&level_indices)
    }

    pub fn indices_to_level_indices(&self, indices: &Tensor) -> Result<Tensor> {
        let indices = indices.unsqueeze(D::Minus1)?;
        let codes_non_centered = indices
            .div(&self.basis.unsqueeze(0)?.unsqueeze(0)?)?
            .floor()?;

        let codes_non_centered = remainder(
            &codes_non_centered,
            &self.levels.unsqueeze(0)?.unsqueeze(0)?,
        )?;
        Ok(codes_non_centered)
    }

    pub fn get_output_from_indices(&self, indices: &Tensor) -> Result<Tensor> {
        let codes = self.indices_to_codes(indices)?;
        let out = self.project_out.forward(&codes)?;
        Ok(out)
    }
    fn _scale_and_shift_inverse(&self, zhat: &Tensor) -> Result<Tensor> {
        let half_width = self.levels.div(&Tensor::full(
            2.0,
            self.levels.shape(),
            self.levels.device(),
        )?)?;
        zhat.sub(&half_width)?.div(&half_width)
    }

    pub fn codebook_size(&self) -> usize {
        self.levels
            .to_vec1::<u32>()
            .unwrap()
            .iter()
            .product::<u32>() as usize
    }

    pub fn implicit_codebook(&self) -> Result<Tensor> {
        let indices = Tensor::arange(0, self.codebook_size() as i64, self.levels.device())?;
        self.indices_to_codes(&indices)
    }
}
