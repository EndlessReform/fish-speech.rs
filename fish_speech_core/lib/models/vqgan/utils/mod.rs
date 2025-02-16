use crate::config::WhichFishVersion;
use candle_core::{IndexOp, Result, D};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder};

#[derive(Clone)]
pub struct FishConvNet {
    conv: Conv1d,
    kernel_size_adj: usize,
    stride: usize,
    model: WhichFishVersion,
}

impl FishConvNet {
    /// Yes, this is terrible,
    /// but the authors decided to change every single conv in the VQGAN...
    /// for some fucking reason
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        config: Conv1dConfig,
        model: &WhichFishVersion,
    ) -> Result<Self> {
        let conv = Conv1d::new(
            vb.get(
                (out_channels, in_channels, kernel_size),
                match model {
                    WhichFishVersion::Fish1_2 => "weight",
                    _ => "conv.weight",
                },
            )?,
            Some(vb.get(
                out_channels,
                match model {
                    WhichFishVersion::Fish1_2 => "bias",
                    _ => "conv.bias",
                },
            )?),
            config.clone(),
        );

        let kernel_size_adj = (kernel_size - 1) * config.dilation + 1;
        Ok(Self {
            conv,
            kernel_size_adj,
            stride: config.stride,
            model: model.clone(),
        })
    }
}

impl Module for FishConvNet {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let pad = self.kernel_size_adj - self.stride;

        let x = match self.model {
            WhichFishVersion::Fish1_2 => xs,
            _ => &xs.pad_with_zeros(D::Minus1, pad, 0)?,
        };
        self.conv.forward(x)?.contiguous()
    }
}

pub struct FishTransConvNet {
    conv: ConvTranspose1d,
    stride: usize,
    kernel_size: usize,
    model: WhichFishVersion,
}

impl FishTransConvNet {
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        config: ConvTranspose1dConfig,
        model: &WhichFishVersion,
    ) -> Result<Self> {
        let conv = ConvTranspose1d::new(
            vb.get(
                (in_channels, out_channels, kernel_size),
                match model {
                    WhichFishVersion::Fish1_2 => "weight",
                    _ => "conv.weight",
                },
            )?,
            Some(vb.get(
                out_channels,
                match model {
                    WhichFishVersion::Fish1_2 => "bias",
                    _ => "conv.bias",
                },
            )?),
            config,
        );
        // println!("Config at idx {idx}: {:?}", config);
        // conv.weight()
        //     .write_npy(format!("conv_weights_{}_rs.npy", idx))?;
        Ok(Self {
            conv,
            stride: config.stride,
            kernel_size,
            model: model.clone(),
        })
    }
}

impl Module for FishTransConvNet {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let x = self.conv.forward(xs)?;
        let pad = self.kernel_size.saturating_sub(self.stride);
        let padding_right = pad;
        let padding_left = pad - padding_right;
        let x = match self.model {
            WhichFishVersion::Fish1_2 => x,
            _ => x.i((.., .., padding_left..x.dim(D::Minus1)? - padding_right))?,
        };
        let x = x.contiguous()?;
        Ok(x)
    }
}
