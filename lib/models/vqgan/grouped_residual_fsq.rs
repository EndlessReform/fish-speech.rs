use candle_nn::{Linear, Module, VarBuilder};
// Ported from https://github.com/lucidrains/vector-quantize-pytorch/blob/14479985c1ffbf86182c3f647197986f9f46e5d7/vector_quantize_pytorch/residual_fsq.py#L208

struct ResidualFSQConfig {
    input_dim: usize,
    levels: Vec<usize>,
}

impl Default for ResidualFSQConfig {
    fn default() -> Self {
        Self {
            input_dim: 512,
            levels: vec![8, 5, 5, 5],
        }
    }
}

struct ResidualFSQ {
    project_in: Linear,
    project_out: Linear,
}
