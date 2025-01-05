use clap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhichFishVersion {
    Fish1_2,
    Fish1_4,
    Fish1_5,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhichModel {
    #[value(name = "1.2")]
    Fish1_2,

    #[value(name = "1.4")]
    Fish1_4,

    #[value(name = "1.5")]
    Fish1_5,

    #[value(name = "dual_ar")]
    DualAR,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhichCodec {
    Fish(WhichFishVersion),
    Mimi,
}

impl WhichCodec {
    pub fn from_model(model: WhichModel) -> Self {
        match model {
            WhichModel::DualAR => Self::Mimi,
            WhichModel::Fish1_2 => Self::Fish(WhichFishVersion::Fish1_2),
            WhichModel::Fish1_4 => Self::Fish(WhichFishVersion::Fish1_4),
            WhichModel::Fish1_5 => Self::Fish(WhichFishVersion::Fish1_5),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhichLM {
    Fish(WhichFishVersion),
    DualAR,
}

impl WhichLM {
    pub fn from_model(model: WhichModel) -> Self {
        match model {
            WhichModel::DualAR => Self::DualAR,
            WhichModel::Fish1_2 => Self::Fish(WhichFishVersion::Fish1_2),
            WhichModel::Fish1_4 => Self::Fish(WhichFishVersion::Fish1_4),
            WhichModel::Fish1_5 => Self::Fish(WhichFishVersion::Fish1_5),
        }
    }
}
