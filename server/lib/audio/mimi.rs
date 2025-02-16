#![allow(unused)]
use candle_core::{bail, DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

#[cfg(feature = "mimi")]
use moshi as mm;

pub struct Tokenizer {
    #[cfg(feature = "mimi")]
    model: mm::mimi::Mimi,
    #[cfg(not(feature = "mimi"))]
    _unused: (), // Placeholder for no-moshi version
    device: Device,
    dtype: DType,
}

impl Tokenizer {
    /// Returns self, sample_rate
    pub fn load(vb: VarBuilder, num_codebooks: usize) -> Result<(Self, u32)> {
        #[cfg(feature = "mimi")]
        {
            let cfg = mm::mimi::Config::v0_1(Some(num_codebooks));
            let sample_rate = cfg.sample_rate as u32;
            let model = mm::mimi::Mimi::new(cfg, vb.clone())?;
            Ok((
                Self {
                    model,
                    device: vb.device().clone(),
                    dtype: vb.dtype(),
                },
                sample_rate,
            ))
        }

        #[cfg(not(feature = "mimi"))]
        {
            bail!("Tokenizer requires the 'moshi' feature flag")
        }
    }

    /// Expects shape [batch, time]
    pub fn encode_batch(&mut self, pcm: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "mimi")]
        {
            let data = pcm.to_device(&self.device)?.to_dtype(self.dtype)?;
            self.model.encode(&data)
        }

        #[cfg(not(feature = "mimi"))]
        bail!("Tokenizer requires the 'moshi' feature flag")
    }

    /// Expects shape [batch, n_codebooks, time]
    pub fn decode_batch(&mut self, codes: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "mimi")]
        {
            self.model.decode(&codes)?.to_dtype(DType::F32)
        }

        #[cfg(not(feature = "mimi"))]
        bail!("Tokenizer requires the 'moshi' feature flag")
    }

    pub fn decode_step(&mut self, codes: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "mimi")]
        {
            match self
                .model
                .decode_step(&mm::streaming::StreamTensor::from_tensor(codes.clone()))?
                .as_option()
            {
                Some(pcm) => pcm.to_dtype(DType::F32),
                None => bail!("TODO figure out what this does"),
            }
        }

        #[cfg(not(feature = "mimi"))]
        bail!("Tokenizer requires the 'moshi' feature flag")
    }

    pub fn reset(&mut self) {
        #[cfg(feature = "mimi")]
        self.model.reset_state();

        #[cfg(not(feature = "mimi"))]
        {} // No-op when moshi is disabled
    }
}
