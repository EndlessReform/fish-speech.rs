use anyhow::Result;
use candle_core::{Device, Module, Tensor, D};
use candle_nn::{kv_cache::KvCache, linear_no_bias, Linear, VarBuilder, VarMap};
use fish_speech_core::models::text2semantic::BaseModelArgs;

struct Attention {
    wqkv: Linear,
    cache: KvCache,
    config: BaseModelArgs,
}

impl Attention {
    pub fn load(vb: VarBuilder, config: &BaseModelArgs) -> Result<Self> {
        let total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim;
        let wqkv = linear_no_bias(config.dim, total_head_dim, vb.pp("wqkv"))?;
        let cache = KvCache::new(2, 1024);

        Ok(Self {
            wqkv,
            cache,
            config: config.clone(),
        })
    }
    pub fn forward(&mut self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (bsz, seqlen, _) = x.dims3()?;

        let qkv = self.wqkv.forward(x)?;
        let query_pos = self.config.n_head * self.config.head_dim;
        let query_states = qkv.narrow(D::Minus1, 0, query_pos)?;
        let key_states = qkv.narrow(
            D::Minus1,
            query_pos,
            self.config.n_local_heads * self.config.head_dim,
        )?;
        let value_states = qkv.narrow(
            D::Minus1,
            query_pos + self.config.n_local_heads * self.config.head_dim,
            self.config.n_local_heads * self.config.head_dim,
        )?;

        let query_states = query_states
            .reshape((bsz, seqlen, self.config.n_head, self.config.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((bsz, seqlen, self.config.n_local_heads, self.config.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((bsz, seqlen, self.config.n_local_heads, self.config.head_dim))?
            .transpose(1, 2)?;

        let (key_states, value_states) = self
            .cache
            .append(&key_states.contiguous()?, &value_states.contiguous()?)?;
        Ok((key_states, value_states))
    }
}

fn main() -> Result<()> {
    println!("Foo!");
    let vm = VarMap::new();
    let device = Device::cuda_if_available(0)?;
    let vb = VarBuilder::from_varmap(&vm, candle_core::DType::BF16, &device);
    let mut faux_attn = Attention::load(vb, &BaseModelArgs::fish_speech_1_2())?;
    println!("Loaded");

    let faux_x = Tensor::ones((1, 1, 1024), candle_core::DType::BF16, &device)?;

    for i in 0..32 {
        let p = faux_attn.forward(&faux_x)?;
        println!("{}: {:?}, {:?}", i, p.0.shape(), p.1.shape())
    }

    Ok(())
}
