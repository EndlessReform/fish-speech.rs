use super::config::HiFiGANConfig;
use super::utils::{FishConvNet, FishTransConvNet};
use crate::config::WhichFishVersion;
use candle_core::{Result, Tensor};
use candle_nn::{Conv1dConfig, ConvTranspose1dConfig, Module, VarBuilder};

fn get_padding(kernel_size: usize, dilation: Option<usize>) -> usize {
    let dilation = dilation.unwrap_or(1);
    (kernel_size * dilation - dilation) / 2
}

struct ResBlock1 {
    convs1: Vec<FishConvNet>,
    convs2: Vec<FishConvNet>,
}

impl ResBlock1 {
    pub fn load(
        vb: &VarBuilder,
        channels: usize,
        kernel_size: usize,
        dilation: &Vec<usize>,
        model: &WhichFishVersion,
    ) -> Result<Self> {
        let mut convs1: Vec<FishConvNet> = vec![];
        let mut convs2: Vec<FishConvNet> = vec![];

        for (i, d) in dilation.iter().enumerate() {
            let conv = FishConvNet::load(
                vb.pp(format!("convs1.{}", i)),
                channels,
                channels,
                kernel_size,
                Conv1dConfig {
                    dilation: *d,
                    padding: match model {
                        WhichFishVersion::Fish1_2 => get_padding(kernel_size, Some(*d)),
                        _ => 0,
                    },
                    ..Default::default()
                },
                &model,
            )?;
            convs1.push(conv)
        }

        for (i, d) in dilation.iter().enumerate() {
            let conv = FishConvNet::load(
                vb.pp(format!("convs2.{}", i)),
                channels,
                channels,
                kernel_size,
                Conv1dConfig {
                    padding: match model {
                        WhichFishVersion::Fish1_2 => get_padding(kernel_size, Some(1)),
                        _ => 0,
                    },
                    dilation: match model {
                        WhichFishVersion::Fish1_2 => 1,
                        _ => *d,
                    },
                    ..Default::default()
                },
                model,
            )?;
            convs2.push(conv);
        }

        Ok(Self { convs1, convs2 })
    }
}

impl Module for ResBlock1 {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();
        for (c1, c2) in self.convs1.iter().zip(self.convs2.iter()) {
            let xt = x.silu()?;
            let xt = c1.forward(&xt)?.silu()?;
            let xt = c2.forward(&xt)?;
            x = x.add(&xt)?;
        }
        // x.write_npy("res0_rs.npy")?;
        // panic!("First res");
        Ok(x)
    }
}

struct ParallelBlock {
    blocks: Vec<ResBlock1>,
}

impl ParallelBlock {
    pub fn load(
        vb: &VarBuilder,
        channels: usize,
        kernel_sizes: &Vec<usize>,
        dilation_sizes: &Vec<Vec<usize>>,
        model: &WhichFishVersion,
    ) -> Result<Self> {
        let blocks: Result<Vec<ResBlock1>> = kernel_sizes
            .iter()
            .zip(dilation_sizes.iter())
            .enumerate()
            .map(|(i, (k, d))| {
                ResBlock1::load(&vb.pp(format!("blocks.{}", i)), channels, *k, d, model)
            })
            .collect();

        Ok(Self { blocks: blocks? })
    }
}

impl Module for ParallelBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let results: Result<Vec<Tensor>> = self.blocks.iter().map(|b| b.forward(xs)).collect();
        Tensor::stack(&results?, 0)?.mean(0)
    }
}

pub struct HiFiGAN {
    conv_pre: FishConvNet,
    ups: Vec<FishTransConvNet>,
    resblocks: Vec<ParallelBlock>,
    conv_post: FishConvNet,
}

impl HiFiGAN {
    pub fn load(vb: VarBuilder, cfg: &HiFiGANConfig, model: &WhichFishVersion) -> Result<Self> {
        let conv_pre = FishConvNet::load(
            vb.pp("conv_pre"),
            cfg.num_mels,
            cfg.upsample_initial_channel,
            cfg.pre_conv_kernel_size,
            Conv1dConfig {
                padding: match model {
                    WhichFishVersion::Fish1_2 => get_padding(cfg.pre_conv_kernel_size, None),
                    _ => 0,
                },
                ..Default::default()
            },
            model,
        )?;
        let mut ups: Vec<FishTransConvNet> = vec![];

        for (i, (u, k)) in cfg
            .upsample_rates
            .iter()
            .zip(cfg.upsample_kernel_sizes.iter())
            .enumerate()
        {
            // Ignoring noise_convs for inference
            ups.push(FishTransConvNet::load(
                vb.pp(format!("ups.{}", i)),
                cfg.upsample_initial_channel / 2_usize.pow(i as u32),
                cfg.upsample_initial_channel / 2_usize.pow(i as u32 + 1),
                *k,
                ConvTranspose1dConfig {
                    stride: *u,
                    padding: match model {
                        WhichFishVersion::Fish1_2 => (k - u) / 2,
                        _ => 0,
                    },
                    ..Default::default()
                },
                model,
            )?)
        }

        let resblocks: Result<Vec<ParallelBlock>> = (0..ups.len())
            .map(|i| {
                let ch = cfg.upsample_initial_channel / 2_usize.pow(i as u32 + 1);
                ParallelBlock::load(
                    &vb.pp(format!("resblocks.{}", i)),
                    ch,
                    &cfg.resblock_kernel_sizes,
                    &cfg.resblock_dilation_sizes,
                    model,
                )
            })
            .collect();

        let ch_final = cfg.upsample_initial_channel / 2_usize.pow(ups.len() as u32);
        let conv_post = FishConvNet::load(
            vb.pp("conv_post"),
            ch_final,
            1,
            cfg.post_conv_kernel_size,
            Conv1dConfig {
                padding: match model {
                    WhichFishVersion::Fish1_2 => get_padding(cfg.post_conv_kernel_size, None),
                    _ => 0,
                },
                ..Default::default()
            },
            model,
        )?;

        Ok(Self {
            conv_pre,
            ups,
            resblocks: resblocks?,
            conv_post,
        })
    }
}

impl Module for HiFiGAN {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.conv_pre.forward(xs)?;
        for (u, r) in self.ups.iter().zip(self.resblocks.iter()) {
            let x = u.forward(&xs.silu()?)?;
            xs = r.forward(&x)?;
        }
        let x = self.conv_post.forward(&xs.silu()?)?;
        x.tanh()
    }
}
