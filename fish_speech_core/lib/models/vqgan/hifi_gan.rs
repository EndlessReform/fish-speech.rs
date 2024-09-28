use super::utils::config::HiFiGANConfig;
use candle_core::{Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder};

fn get_padding(kernel_size: usize, dilation: Option<usize>) -> usize {
    let dilation = dilation.unwrap_or(1);
    (kernel_size * dilation - dilation) / 2
}

struct ResBlock1 {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
}

impl ResBlock1 {
    pub fn load(
        vb: &VarBuilder,
        channels: usize,
        kernel_size: usize,
        dilation: &Vec<usize>,
    ) -> Result<Self> {
        let mut convs1: Vec<Conv1d> = vec![];
        let mut convs2: Vec<Conv1d> = vec![];

        for (i, d) in dilation.iter().enumerate() {
            let conv = Conv1d::new(
                vb.get(
                    (channels, channels, kernel_size),
                    &format!("convs1.{}.weight", i),
                )?,
                Some(vb.get(channels, &format!("convs1.{}.bias", i))?),
                Conv1dConfig {
                    stride: 1,
                    dilation: *d,
                    padding: get_padding(kernel_size, Some(*d)),
                    groups: 1,
                },
            );
            convs1.push(conv)
        }

        for i in 0..dilation.len() {
            let conv = Conv1d::new(
                vb.get(
                    (channels, channels, kernel_size),
                    &format!("convs2.{}.weight", i),
                )?,
                Some(vb.get(channels, &format!("convs2.{}.bias", i))?),
                Conv1dConfig {
                    stride: 1,
                    dilation: 1,
                    padding: get_padding(kernel_size, Some(1)),
                    groups: 1,
                },
            );
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
    ) -> Result<Self> {
        let blocks: Result<Vec<ResBlock1>> = kernel_sizes
            .iter()
            .zip(dilation_sizes.iter())
            .enumerate()
            .map(|(i, (k, d))| ResBlock1::load(&vb.pp(format!("blocks.{}", i)), channels, *k, d))
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
    conv_pre: Conv1d,
    ups: Vec<ConvTranspose1d>,
    resblocks: Vec<ParallelBlock>,
    conv_post: Conv1d,
}

impl HiFiGAN {
    pub fn load(vb: VarBuilder, cfg: &HiFiGANConfig) -> Result<Self> {
        let conv_pre = Conv1d::new(
            vb.get(
                (
                    cfg.num_mels,
                    cfg.upsample_initial_channel,
                    cfg.pre_conv_kernel_size,
                ),
                "conv_pre.weight",
            )?,
            Some(vb.get(cfg.upsample_initial_channel, "conv_pre.bias")?),
            Conv1dConfig {
                stride: 1,
                padding: get_padding(cfg.pre_conv_kernel_size, None),
                dilation: 1,
                groups: 1,
            },
        );
        let mut ups: Vec<ConvTranspose1d> = vec![];

        for (i, (u, k)) in cfg
            .upsample_rates
            .iter()
            .zip(cfg.upsample_kernel_sizes.iter())
            .enumerate()
        {
            // Ignoring noise_convs for inference
            ups.push(ConvTranspose1d::new(
                vb.get(
                    (
                        cfg.upsample_initial_channel / 2_usize.pow(i as u32),
                        cfg.upsample_initial_channel / 2_usize.pow(i as u32 + 1),
                        *k,
                    ),
                    &format!("ups.{}.weight", i),
                )?,
                Some(vb.get(
                    cfg.upsample_initial_channel / 2_usize.pow(i as u32 + 1),
                    &format!("ups.{}.bias", i),
                )?),
                ConvTranspose1dConfig {
                    stride: *u,
                    padding: (k - u) / 2,
                    output_padding: 0,
                    groups: 1,
                    dilation: 1,
                },
            ));
        }

        let resblocks: Result<Vec<ParallelBlock>> = (0..ups.len())
            .map(|i| {
                let ch = cfg.upsample_initial_channel / 2_usize.pow(i as u32 + 1);
                ParallelBlock::load(
                    &vb.pp(format!("resblocks.{}", i)),
                    ch,
                    &cfg.resblock_kernel_sizes,
                    &cfg.resblock_dilation_sizes,
                )
            })
            .collect();

        let ch_final = cfg.upsample_initial_channel / 2_usize.pow(ups.len() as u32);
        let conv_post = Conv1d::new(
            vb.get((1, ch_final, cfg.post_conv_kernel_size), "conv_post.weight")?,
            Some(vb.get(1, "conv_post.bias")?),
            Conv1dConfig {
                stride: 1,
                padding: get_padding(cfg.post_conv_kernel_size, None),
                groups: 1,
                dilation: 1,
            },
        );

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
