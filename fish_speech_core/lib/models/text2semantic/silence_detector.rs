use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

pub struct PauseTypeProbe {
    layer: Linear,
    threshold: f32,
}

impl PauseTypeProbe {
    pub fn load(vb: &VarBuilder, hidden_dim: usize, threshold: f32) -> Result<Self> {
        let layer = Linear::new(
            vb.get((1, hidden_dim), "linear.weight")?,
            Some(vb.get(1, "linear.bias")?),
        );
        Ok(Self { layer, threshold })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<bool> {
        let x = self.layer.forward(xs)?;
        let p = x.to_dtype(candle_core::DType::F32)?.flatten_all()?;
        let p = p.to_device(&candle_core::Device::Cpu)?;
        let p = p.to_vec1::<f32>()?[0];
        Ok(p >= self.threshold)
    }
}
