use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{kv_cache::KvCache, linear_no_bias, Linear, VarBuilder, VarMap};
use candle_transformers::utils::repeat_kv;
use fish_speech_core::models::text2semantic::BaseModelArgs;

fn precompute_freqs_cis(
    dtype: DType,
    cfg: &BaseModelArgs,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let rope_theta = cfg.rope_base as f32;
    let dim = cfg.head_dim;
    let max_seq_len = cfg.max_seq_len;
    let inv_freq: Vec<_> = (0..dim)
        .step_by(2)
        .map(|i| 1f32 / rope_theta.powf(i as f32 / dim as f32))
        .collect();
    let inv_freq_len = inv_freq.len();
    let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
    let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
        .to_dtype(dtype)?
        .reshape((max_seq_len, 1))?;
    let freqs = t.matmul(&inv_freq)?;
    let sin = freqs.sin()?;
    let cos = freqs.cos()?;
    Ok((sin, cos))
}

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
    pub fn forward(
        &mut self,
        x: &Tensor,
        freqs_cis: (&Tensor, &Tensor),
    ) -> Result<(Tensor, Tensor, Tensor)> {
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

        let cos = freqs_cis.0.narrow(0, 0, seqlen)?;
        let sin = freqs_cis.1.narrow(0, 0, seqlen)?;
        let q_embed = candle_nn::rotary_emb::rope(&query_states, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&key_states, &cos, &sin)?;
        let (key_states, value_states) = self
            .cache
            .append(&k_embed.contiguous()?, &value_states.contiguous()?)?;
        let key_states =
            repeat_kv(key_states, self.config.n_head / self.config.n_local_heads)?.contiguous()?;
        let value_states = repeat_kv(value_states, self.config.n_head / self.config.n_local_heads)?
            .contiguous()?;
        Ok((q_embed, key_states, value_states))
    }
}

fn main() -> Result<()> {
    println!("Foo!");
    let vm = VarMap::new();
    let device = Device::cuda_if_available(0)?;
    let vb = VarBuilder::from_varmap(&vm, candle_core::DType::BF16, &device);
    let cfg = BaseModelArgs::fish_speech_1_2();
    let mut faux_attn = Attention::load(vb.pp("attn"), &cfg)?;
    println!("Loaded");

    let faux_x = Tensor::ones((1, 1, 1024), vb.dtype(), &device)?;
    let freqs_cis = precompute_freqs_cis(vb.dtype(), &cfg, &device)?;

    for i in 0..32 {
        let p = faux_attn.forward(&faux_x, (&freqs_cis.0, &freqs_cis.1))?;
        println!("{}: {:?}, {:?}", i, p.0.shape(), p.1.shape())
    }

    Ok(())
}
