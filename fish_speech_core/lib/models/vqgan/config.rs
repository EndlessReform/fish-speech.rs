use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
pub struct SpecTransformConfig {
    pub sample_rate: usize,
    pub n_mels: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
}

impl SpecTransformConfig {
    pub fn fish_1_2() -> Self {
        Self {
            sample_rate: 44100,
            n_mels: 160,
            n_fft: 2048,
            hop_length: 512,
            win_length: 2048,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct BackboneConfig {
    pub input_channels: usize,
    pub depths: [usize; 4],
    pub dims: [usize; 4],
    pub kernel_size: usize,
}

impl BackboneConfig {
    /// Config from Fish Speech 1.2 SFT
    pub fn fish_1_2() -> Self {
        Self {
            input_channels: 160,
            depths: [3, 3, 9, 3],
            dims: [128, 256, 384, 512],
            // drop_path_rate: 0.2,
            kernel_size: 7,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct HiFiGANConfig {
    pub hop_length: usize,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub num_mels: usize,
    pub upsample_initial_channel: usize,
    pub use_template: bool,
    pub pre_conv_kernel_size: usize,
    pub post_conv_kernel_size: usize,
}

impl HiFiGANConfig {
    pub fn fish_1_2() -> Self {
        Self {
            hop_length: 512,
            upsample_rates: vec![8, 8, 2, 2, 2],
            upsample_kernel_sizes: vec![16, 16, 4, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            num_mels: 512,
            upsample_initial_channel: 512,
            use_template: false,
            pre_conv_kernel_size: 13,
            post_conv_kernel_size: 13,
        }
    }
}

// https://github.com/fishaudio/fish-speech/blob/9e2f5e6b3a382849b8ee54da10d6a68bbd913f4d/fish_speech/models/vqgan/modules/fsq.py#L4

#[derive(Debug, Clone, Deserialize)]
pub struct DownsampleFSQConfig {
    pub input_dim: usize,
    pub n_codebooks: usize,
    pub n_groups: usize,
    pub levels: Vec<u32>,
    pub downsample_factor: Vec<usize>,
    pub downsample_dims: Option<Vec<usize>>,
}

impl DownsampleFSQConfig {
    pub fn firefly_1_2() -> Self {
        Self {
            input_dim: 512,
            n_groups: 4,
            n_codebooks: 1,
            levels: vec![8, 5, 5, 5],
            downsample_factor: vec![2],
            downsample_dims: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FireflyConfig {
    pub spec_transform: SpecTransformConfig,
    pub backbone: BackboneConfig,
    pub head: HiFiGANConfig,
    pub quantizer: DownsampleFSQConfig,
}

impl FireflyConfig {
    pub fn fish_speech_1_2() -> Self {
        Self {
            spec_transform: SpecTransformConfig::fish_1_2(),
            backbone: BackboneConfig::fish_1_2(),
            head: HiFiGANConfig::fish_1_2(),
            quantizer: DownsampleFSQConfig::firefly_1_2(),
        }
    }
}
