pub mod rep_pen;
use rand::Rng;

/// Extremely stripped-down CPU softmax for slow model out
pub fn legacy_softmax_sample(pad_prob: f32, eos_prob: f32, pad_id: u32, eos_id: u32) -> u32 {
    // Compute softmax
    let exp_pad = (pad_prob - pad_prob.max(eos_prob)).exp();
    let exp_eos = (eos_prob - pad_prob.max(eos_prob)).exp();
    let sum = exp_pad + exp_eos;

    let softmax_pad = exp_pad / sum;

    // Generate a random number
    let mut rng = rand::thread_rng();
    let rand_val: f32 = rng.gen(); // Generates a float between 0.0 and 1.0

    // Sample according to softmax probabilities
    if rand_val < softmax_pad {
        pad_id // pad_id
    } else {
        eos_id // im_end_id
    }
}

pub struct SamplingArgs {
    pub temp: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub repetition_penalty: f32,
}
