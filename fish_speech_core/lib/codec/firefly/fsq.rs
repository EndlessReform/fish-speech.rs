use candle_core::{DType, Result, Tensor, D};
use candle_nn::VarBuilder;

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
    codebook_dim: usize,
    n_codebooks: usize,
}

// This is also not implemented as a unary op
fn atanh(x: &Tensor) -> Result<Tensor> {
    // atanh(x) = 0.5 * ln((1 + x) / (1 - x))
    let numerator = (1.0 as f64 + x)?;
    let denominator = (1.0 as f64 - x)?;
    numerator.div(&denominator)?.log()? * 0.5 as f64
}

// This is also not implemented as a binary op. Don't ask me why
fn remainder(x: &Tensor, y: &Tensor) -> Result<Tensor> {
    let quotient = x.broadcast_div(y)?.floor()?;
    x.broadcast_sub(&quotient.broadcast_mul(y)?)
}

// This is also not implemented as a binary op. Don't ask me why
fn remainder_float(x: &Tensor, y: f64) -> Result<Tensor> {
    let quotient = (x.clone() / y)?.floor()?;
    x.sub(&(quotient.clone() * y)?)
}

impl FSQ {
    pub fn load(vb: VarBuilder, config: &FSQConfig) -> Result<Self> {
        let device = vb.device();
        let FSQConfig {
            levels,
            n_codebooks: _n_codebooks,
            input_dim: _input_dim,
        } = config;
        let codebook_dim = levels.len();

        // Convert u32 to f32 here
        let levels_f32: Vec<f32> = levels.iter().map(|&x| x as f32).collect();
        let levels = Tensor::from_slice(&levels_f32, levels.len(), device)?;

        let mut basis = Vec::with_capacity(codebook_dim);
        basis.push(1.0);
        for i in 1..codebook_dim {
            basis.push(basis[i - 1] * levels_f32[i - 1]);
        }
        let basis = Tensor::new(basis.as_slice(), device)?;

        Ok(Self {
            levels,
            basis,
            codebook_dim,
            n_codebooks: config.n_codebooks,
        })
    }

    pub fn bound(&self, z: &Tensor) -> Result<Tensor> {
        // let levels_sub_1 = self.levels.sub(&Tensor::ones_like(&self.levels)?)?;
        let levels_sub_1 = (self.levels.clone() - 1f64)?;
        let half_l = ((levels_sub_1 * 1.001 as f64)? / 2.0 as f64)?;

        let remainder = remainder_float(&self.levels, 2.0)?;

        let offset = (remainder
            .eq(&Tensor::zeros_like(&self.levels)?)?
            .to_dtype(self.levels.dtype())?
            * 0.5 as f64)?;

        let shift = atanh(&offset.div(&half_l)?)?;
        z.broadcast_add(&shift)?
            .tanh()?
            .broadcast_mul(&half_l)?
            .broadcast_sub(&offset)
    }

    pub fn quantize(&self, z: &Tensor) -> Result<Tensor> {
        let quantized = self.bound(z)?.round()?;

        let half_width = (self.levels.clone() / 2.0 as f64)?.floor()?;
        quantized.broadcast_div(&half_width)
    }

    pub fn forward(&self, z: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch_size, seq_len, _) = z.dims3()?;

        let z = z.reshape((batch_size, seq_len, self.n_codebooks, self.codebook_dim))?;

        let codes = self.quantize(&z)?;
        let indices = self.codes_to_indices(&codes)?;
        // TODO: revisit this if keep_num_codebooks_dim changes or codebooks > 1
        let indices = indices.squeeze(2)?;

        let (batch_size, seq_len, _, _) = codes.dims4()?;

        let codes_reshaped =
            codes.reshape((batch_size, seq_len, self.n_codebooks * self.codebook_dim))?;

        Ok((codes_reshaped, indices))
    }

    fn _scale_and_shift(&self, zhat_normalized: &Tensor) -> Result<Tensor> {
        let half_width = (self.levels.clone() / 2.0 as f64)?.floor()?;
        zhat_normalized
            .broadcast_mul(&half_width)?
            .broadcast_add(&half_width)
    }

    fn _scale_and_shift_inverse(&self, zhat: &Tensor) -> Result<Tensor> {
        let half_width = (self.levels.clone() / 2.0 as f64)?.floor()?;
        zhat.broadcast_sub(&half_width)?.broadcast_div(&half_width)
    }

    pub fn codes_to_indices(&self, zhat: &Tensor) -> Result<Tensor> {
        let zhat = self._scale_and_shift(&zhat)?;

        zhat.broadcast_mul(&self.basis)?
            .sum(D::Minus1)?
            .to_dtype(DType::I64)
    }

    pub fn indices_to_codes(&self, indices: &Tensor) -> Result<Tensor> {
        let level_indices = self.indices_to_level_indices(indices)?;
        self._scale_and_shift_inverse(&level_indices)
    }

    pub fn indices_to_level_indices(&self, indices: &Tensor) -> Result<Tensor> {
        let indices = indices.unsqueeze(D::Minus1)?;
        let codes_non_centered = indices.broadcast_div(&self.basis)?.floor()?;

        let codes_non_centered = remainder(&codes_non_centered, &self.levels)?;
        Ok(codes_non_centered)
    }

    // Modify codebook_size to work with f32
    pub fn codebook_size(&self) -> usize {
        self.levels
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|&x| x.round() as usize)
            .product()
    }

    pub fn implicit_codebook(&self) -> Result<Tensor> {
        let indices = Tensor::arange(0, self.codebook_size() as i64, self.levels.device())?
            .to_dtype(DType::F32)?;
        self.indices_to_codes(&indices)
    }
}
