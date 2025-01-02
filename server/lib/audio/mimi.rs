use candle_core::{bail, DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use mm::streaming::StreamTensor;
use moshi as mm;

pub struct Tokenizer {
    model: mm::encodec::Encodec,
    device: Device,
    dtype: DType,
}

impl Tokenizer {
    /// Returns self, sample_rate
    pub fn load(vb: VarBuilder, num_codebooks: usize) -> Result<(Self, u32)> {
        let cfg = mm::encodec::Config::v0_1(Some(num_codebooks));
        let sample_rate = cfg.sample_rate as u32;
        let model = mm::encodec::Encodec::new(cfg, vb.clone())?;
        Ok((
            Self {
                model,
                device: vb.device().clone(),
                dtype: vb.dtype(),
            },
            sample_rate,
        ))
    }

    /// Expects shape [batch, time]
    pub fn encode_batch(&mut self, pcm: &Tensor) -> Result<Tensor> {
        let data = pcm.to_device(&self.device)?.to_dtype(self.dtype)?;
        self.model.encode(&data)
    }

    /// Expects shape [batch, n_codebooks, time]
    pub fn decode_batch(&mut self, codes: &Tensor) -> Result<Tensor> {
        self.model.decode(&codes)?.to_dtype(DType::F32)
    }

    pub fn decode_step(&mut self, codes: &Tensor) -> Result<Tensor> {
        match self
            .model
            .decode_step(&StreamTensor::from_tensor(codes.clone()))?
            .as_option()
        {
            Some(pcm) => pcm.to_dtype(DType::F32),
            None => bail!("TODO figure out what this does"),
        }
    }

    pub fn reset(&mut self) {
        self.model.reset_state()
    }
}
