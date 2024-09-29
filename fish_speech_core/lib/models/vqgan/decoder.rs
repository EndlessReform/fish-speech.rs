use candle_core::{Device, Result, Tensor, D};
use candle_nn::{Module, VarBuilder};

use super::config::{FireflyConfig, WhichModel};
use super::hifi_gan::HiFiGAN;
use super::quantizer::DownsampleFiniteScalarQuantizer;

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
    pub fn load(vb: &VarBuilder, cfg: &FireflyConfig, model: &WhichModel) -> Result<Self> {
        let quantizer =
            DownsampleFiniteScalarQuantizer::load(vb.pp("quantizer"), &cfg.quantizer, model)?;
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

        let z = self.quantizer.decode(indices)?;
        println!("Tokens dequantized!");
        let mel_masks_float_conv = mel_masks.unsqueeze(1)?.to_dtype(z.dtype())?;
        let audio_masks_float_conv = audio_masks.unsqueeze(1)?.to_dtype(z.dtype())?;

        let z = z.broadcast_mul(&mel_masks_float_conv)?;
        // z.write_npy("./fish_1_4_quantize_rs.npy")?;
        let out = self
            .head
            .forward(&z)?
            .broadcast_mul(&audio_masks_float_conv)?;
        out.write_npy("fish_1_4_decode_rs.npy")?;
        Ok(out)
    }
}
