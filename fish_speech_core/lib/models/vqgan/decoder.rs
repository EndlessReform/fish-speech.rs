use candle_core::{Device, Result, Tensor, D};
use candle_nn::VarBuilder;

use super::hifi_gan::HiFiGAN;
use super::quantizer::DownsampleFiniteScalarQuantizer;
use super::utils::config::FireflyConfig;

fn sequence_mask(lengths: &Tensor, max_length: Option<u32>, device: &Device) -> Result<Tensor> {
    // If lengths is empty, we have bigger problems
    let max_length = max_length.unwrap_or(lengths.max_keepdim(D::Minus1)?.to_vec1::<u32>()?[0]);
    let x = Tensor::arange(0 as u32, max_length as u32, device)?.unsqueeze(0)?;
    x.broadcast_lt(&lengths.unsqueeze(1)?)
}

pub struct FireflyDecoder {
    quantizer: DownsampleFiniteScalarQuantizer,
    head: HiFiGAN,
    cfg: FireflyConfig,
}

impl FireflyDecoder {
    /// TODO: make this configurable!
    pub fn load(vb: &VarBuilder, cfg: &FireflyConfig) -> Result<Self> {
        let quantizer = DownsampleFiniteScalarQuantizer::load(vb.pp("quantizer"), &cfg.quantizer)?;
        let head = HiFiGAN::load(vb.pp("head"), &cfg.head)?;

        Ok(Self {
            quantizer,
            head,
            cfg: cfg.clone(),
        })
    }

    pub fn decode(&self, indices: &Tensor, feature_lengths: &Tensor) -> Result<Tensor> {
        let factor = self
            .quantizer
            .downsample_factor
            .clone()
            .into_iter()
            .reduce(|acc, e| acc * e)
            .unwrap_or(1);

        let mel_masks = sequence_mask(
            &(feature_lengths * (factor as f64))?,
            Some((indices.dim(2)? * factor) as u32),
            indices.device(),
        )?;
        // TODO: Figure out what dtype it needs to be

        let audio_masks = sequence_mask(
            &((feature_lengths * (factor as f64))? * self.cfg.spec_transform.hop_length as f64)?,
            Some((indices.dim(2)? * factor * self.cfg.spec_transform.hop_length) as u32),
            indices.device(),
        )?;
        let audio_masks_float_conv = audio_masks.unsqueeze(1)?;

        let z = self.quantizer.decode(indices)?;
        let mel_masks_float_conv = mel_masks.unsqueeze(1)?.to_dtype(z.dtype())?;
        println!(
            "z: {:?}, mel masks: {:?}",
            z.shape(),
            mel_masks_float_conv.shape()
        );

        z.broadcast_mul(&mel_masks_float_conv)
    }
}
