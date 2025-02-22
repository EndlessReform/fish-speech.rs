use crate::config::{WhichFishVersion, WhichLM};
use crate::lm::dual_ar::TokenConfig;
use candle_core::{IndexOp, Result, Tensor, D};

/// Constrains Fish 1.5+ models to <|im_end|> plus <|semantic:n|> range, leaves others untouched
pub fn constrain_probs_to_audio(
    x: &Tensor,
    model_type: &WhichLM,
    token_config: &TokenConfig,
) -> Result<Tensor> {
    match model_type {
        WhichLM::DualAR | WhichLM::Fish(WhichFishVersion::Fish1_5) => {
            if token_config.im_end_id == token_config.semantic_start_id - 1 {
                // Fish 1.5: <|im_end|> is right before the semantic range, saving us an indexop and a cat
                x.i((.., .., token_config.im_end_id as usize..))?
                    .contiguous()
            } else {
                // Generic DualAR model using Fish 1.5 speaker line
                // TODO: will still break for inner monologue-style speaker lines
                let im_end_prob = x
                    .i((.., .., token_config.im_end_id as usize))?
                    .contiguous()?
                    .unsqueeze(1)?;
                // Inefficient but functional if control tokens AFTER semantic range
                let semantic_token_range = x
                    .i((.., .., token_config.semantic_start_id as usize..))?
                    .contiguous()?;
                Tensor::cat(&[im_end_prob, semantic_token_range], D::Minus1)
            }
        }
        _ => Ok(x.clone()),
    }
}

/// Put back tokens after constrained generation sampling
pub fn rescale_semantic_tokens(
    tokens: Vec<u32>,
    model_type: &WhichLM,
    token_config: &TokenConfig,
) -> Vec<u32> {
    match model_type {
        WhichLM::DualAR | WhichLM::Fish(WhichFishVersion::Fish1_5) => tokens
            .iter()
            .map(|token| {
                if token_config.im_end_id == token_config.semantic_start_id - 1 {
                    token + token_config.im_end_id
                } else if *token == 0 {
                    token_config.im_end_id
                } else {
                    token - 1 + token_config.semantic_start_id
                }
            })
            .collect(),
        _ => tokens,
    }
}
