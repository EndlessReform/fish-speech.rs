use crate::config::WhichFishVersion;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct SpecTransformConfig {
    pub sample_rate: usize,
    pub n_mels: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
}

impl Default for SpecTransformConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            n_mels: 160,
            n_fft: 2048,
            hop_length: 512,
            win_length: 2048,
        }
    }
}
impl SpecTransformConfig {
    pub fn fish_1_2() -> Self {
        Self {
            ..Default::default()
        }
    }
    // Identical from 1.2 -> 1.4
    pub fn fish_1_4() -> Self {
        Self {
            ..Default::default()
        }
    }
    // Identical from 1.2 -> 1.5
    pub fn fish_1_5() -> Self {
        Self {
            ..Default::default()
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

impl Default for BackboneConfig {
    fn default() -> Self {
        Self {
            input_channels: 160,
            depths: [3, 3, 9, 3],
            dims: [128, 256, 384, 512],
            // drop_path_rate: 0.2,
            kernel_size: 7,
        }
    }
}
impl BackboneConfig {
    /// Config from Fish Speech 1.2 SFT
    pub fn fish_1_2() -> Self {
        Self {
            ..Default::default()
        }
    }
    // Identical from 1.2 -> 1.4
    pub fn fish_1_4() -> Self {
        Self {
            ..Default::default()
        }
    }
    // Identical from 1.2 -> 1.5
    pub fn fish_1_5() -> Self {
        Self {
            ..Default::default()
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

impl Default for HiFiGANConfig {
    fn default() -> Self {
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
impl HiFiGANConfig {
    pub fn fish_1_2() -> Self {
        Self {
            ..Default::default()
        }
    }
    pub fn fish_1_4() -> Self {
        Self {
            ..Default::default()
        }
    }
    pub fn fish_1_5() -> Self {
        Self {
            ..Default::default()
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
    pub fn firefly_1_4() -> Self {
        Self {
            input_dim: 512,
            n_groups: 8,
            n_codebooks: 1,
            levels: vec![8, 5, 5, 5],
            downsample_dims: None,
            downsample_factor: vec![2, 2],
        }
    }
    pub fn firefly_1_5() -> Self {
        DownsampleFSQConfig::firefly_1_4()
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
    pub fn fish_speech_1_4() -> Self {
        Self {
            spec_transform: SpecTransformConfig::fish_1_4(),
            backbone: BackboneConfig::fish_1_4(),
            head: HiFiGANConfig::fish_1_4(),
            quantizer: DownsampleFSQConfig::firefly_1_4(),
        }
    }

    pub fn get_config_for(model: WhichFishVersion) -> Self {
        match model {
            WhichFishVersion::Fish1_2 => Self::fish_speech_1_2(),
            WhichFishVersion::Fish1_4 => Self::fish_speech_1_4(),
            WhichFishVersion::Fish1_5 => Self::fish_speech_1_4(),
        }
    }
}
