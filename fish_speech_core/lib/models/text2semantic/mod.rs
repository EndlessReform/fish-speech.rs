use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{ops::softmax_last_dim, Embedding, Linear, Module, RmsNorm, VarBuilder};

#[derive(Debug, Clone)]
pub struct BaseModelArgs {
    pub model_type: String,
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_fast_layer: usize,
    pub n_head: usize,
    pub dim: usize,
    pub intermediate_size: Option<usize>,
    pub n_local_heads: usize,
    pub head_dim: usize,
    pub rope_base: f32,
    pub norm_eps: f64,
    pub max_seq_len: usize,
    pub dropout: f32,
    pub tie_word_embeddings: bool,
    pub attention_qkv_bias: bool,
    pub codebook_size: usize,
    pub num_codebooks: usize,
}

impl Default for BaseModelArgs {
    fn default() -> Self {
        Self {
            model_type: "base".to_string(),
            vocab_size: 32000,
            n_layer: 32,
            n_fast_layer: 4,
            n_head: 32,
            dim: 4096,
            intermediate_size: Some(4096),
            n_local_heads: 2,
            head_dim: 64,
            rope_base: 10000.0,
            norm_eps: 1e-5,
            max_seq_len: 2048,
            dropout: 0.0,
            tie_word_embeddings: true,
            attention_qkv_bias: false,
            codebook_size: 160,
            num_codebooks: 4,
        }
    }
}

pub struct FeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl FeedForward {
    pub fn load(vb: &VarBuilder, config: &BaseModelArgs) -> Result<Self> {
        let w1 = Linear::new(
            vb.get(
                (
                    config.intermediate_size.unwrap_or(config.dim * 4),
                    config.dim,
                ),
                "w1.weight",
            )?,
            None,
        );
        let w2 = Linear::new(
            vb.get(
                (
                    config.intermediate_size.unwrap_or(config.dim * 4),
                    config.dim,
                ),
                "w2.weight",
            )?,
            None,
        );
        let w3 = Linear::new(
            vb.get(
                (
                    config.dim,
                    config.intermediate_size.unwrap_or(config.dim * 4),
                ),
                "w3.weight",
            )?,
            None,
        );
        Ok(Self { w1, w2, w3 })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.w2
            .forward(&Tensor::silu(&self.w1.forward(&xs)?)?.broadcast_mul(&self.w3.forward(&xs)?)?)
    }
}

/// Mostly copied from candle-transformers/phi3.rs
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dtype: DType, cfg: &BaseModelArgs, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_seq_len;
        let rope_theta = 1f64 / (cfg.rope_base as f64);

        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    pub fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

/// Copied from falcon.rs
fn make_causal_mask(t: usize) -> Result<Tensor> {
    let mask: Vec<_> = (0..t)
        .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
        .collect();
    let mask = Tensor::from_slice(&mask, (t, t), &Device::Cpu)?;
    Ok(mask)
}

fn prepare_attn_mask(b_sz: usize, seq_len: usize) -> Result<Tensor> {
    // let mask = Tensor::ones((b_sz, seq_len), DType::U32, &Device::Cpu)?;
    let mask = make_causal_mask(seq_len)?;
    let mask = mask.broadcast_as((b_sz, 1, seq_len, seq_len))?;
    Ok(mask)
}

pub struct Attention {
    dropout: f32,
    n_head: usize,
    head_dim: usize,
    n_local_heads: usize,
    dim: usize,
    wqkv: Linear,
    wo: Linear,
    kv_cache: Option<(Tensor, Tensor)>,
    use_cache: bool,
    rope_emb: RotaryEmbedding,
}

impl Attention {
    pub fn load(vb: &VarBuilder, config: &BaseModelArgs) -> Result<Self> {
        let total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim;
        // KQV for all heads, but in a batch
        let wqkv = Linear::new(vb.get((total_head_dim, config.dim), "wqkv.weight")?, None);
        let wo = Linear::new(vb.get((config.dim, config.dim), "wo.weight")?, None);

        let rope_emb = RotaryEmbedding::new(vb.dtype(), config, vb.device())?;

        Ok(Self {
            dropout: config.dropout,
            n_head: config.n_head,
            head_dim: config.head_dim,
            n_local_heads: config.n_local_heads,
            dim: config.dim,
            wqkv,
            wo,
            // TODO configure this, improve cache handling
            use_cache: true,
            kv_cache: None,
            rope_emb,
        })
    }

    /// Standard inefficient SDPA
    fn scaled_dot_product_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (l, s) = (query.dim(D::Minus2)?, key.dim(D::Minus2)?);
        let scale_factor = 1f64 / (query.dim(D::Minus1)? as f64).sqrt();
        let mut attn_bias = Tensor::zeros((1, 1, l, s), query.dtype(), query.device())?;

        if let Some(mask) = attn_mask {
            attn_bias = attn_bias.broadcast_add(mask)?;
        }

        let attn_weight = query.matmul(&(key.transpose(D::Minus2, D::Minus1)? * scale_factor)?)?;
        let attn_weight = attn_weight.broadcast_add(&attn_bias)?;
        let attn_weight = softmax_last_dim(&attn_weight)?;
        // Ignoring dropout until we implement training
        attn_weight.broadcast_matmul(value)
    }

    pub fn forward(&mut self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b_sz, q_len, _) = x.dims3()?;

        let qkv = self.wqkv.forward(x)?;
        let query_pos = self.n_head * self.head_dim;
        let query_states = qkv.narrow(D::Minus1, 0, query_pos)?;
        let key_states = qkv.narrow(D::Minus1, query_pos, self.n_local_heads * self.head_dim)?;
        let value_states = qkv.narrow(
            D::Minus1,
            query_pos + self.n_local_heads * self.head_dim,
            self.n_local_heads * self.head_dim,
        )?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.n_local_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.n_local_heads, self.head_dim))?
            .transpose(1, 2)?;

        // TODO: Is this offset right?
        let (query_states, key_states) =
            self.rope_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, 0)?;

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        // TODO: Add optional flash attention
        let y = self.scaled_dot_product_attention(
            &query_states,
            &key_states,
            &value_states,
            Some(mask),
        )?;
        let y = y
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_sz, q_len, self.dim))?;
        self.wo.forward(&y)
    }
}

pub struct TransformerBlock {
    attention: Attention,
    feed_forward: FeedForward,
    ffn_norm: RmsNorm,
    attention_norm: RmsNorm,
}

impl TransformerBlock {
    pub fn load(vb: &VarBuilder, cfg: &BaseModelArgs) -> Result<Self> {
        let attention = Attention::load(&vb.pp("attention"), cfg)?;
        let feed_forward = FeedForward::load(&vb.pp("feed_forward"), cfg)?;
        let ffn_norm = RmsNorm::new(vb.get(cfg.dim, "ffn_norm")?, cfg.norm_eps);
        let attention_norm = RmsNorm::new(vb.get(cfg.dim, "attention_norm")?, cfg.norm_eps);

        Ok(Self {
            attention,
            feed_forward,
            ffn_norm,
            attention_norm,
        })
    }

    pub fn forward(&mut self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let h = (x + self
            .attention
            .forward(&self.attention_norm.forward(x)?, mask)?)?;
        self.feed_forward.forward(&self.ffn_norm.forward(&h)?) + h
    }
}

pub struct DualARTransformer {
    embeddings: Embedding,
    codebook_embeddings: Embedding,
    fast_layers: Vec<TransformerBlock>,
    fast_embeddings: Embedding,
    layers: Vec<TransformerBlock>,
    output: Linear,
    fast_output: Linear,
    norm: RmsNorm,
    fast_norm: RmsNorm,
    semantic_token_id: i64,
    cfg: BaseModelArgs,
}

impl DualARTransformer {
    pub fn load(vb: &VarBuilder, cfg: &BaseModelArgs, semantic_token_id: i64) -> Result<Self> {
        let embeddings = Embedding::new(vb.get((cfg.vocab_size, cfg.dim), "embeddings")?, cfg.dim);
        let codebook_embeddings = Embedding::new(
            vb.get(
                (cfg.codebook_size * cfg.num_codebooks, cfg.dim),
                "codebook_embeddings",
            )?,
            cfg.dim,
        );
        let layers: Result<Vec<TransformerBlock>> = (0..cfg.n_layer)
            .map(|l| TransformerBlock::load(&vb.pp(format!("layers.{}", l)), cfg))
            .collect();
        let layers = layers?;
        let norm = RmsNorm::new(vb.get(cfg.dim, "norm")?, cfg.norm_eps);
        let output = Linear::new(vb.get((cfg.vocab_size, cfg.dim), "output")?, None);
        let fast_embeddings = Embedding::new(
            vb.get((cfg.codebook_size, cfg.dim), "fast_embeddings")?,
            cfg.dim,
        );
        let fast_layers: Result<Vec<TransformerBlock>> = (0..cfg.n_fast_layer)
            .map(|l| TransformerBlock::load(&vb.pp(format!("fast_layers.{}", l)), cfg))
            .collect();
        let fast_layers = fast_layers?;
        let fast_norm = RmsNorm::new(vb.get(cfg.dim, "fast_norm")?, cfg.norm_eps);
        let fast_output = Linear::new(vb.get((cfg.dim, cfg.codebook_size), "fast_output")?, None);

        Ok(Self {
            embeddings,
            codebook_embeddings,
            fast_embeddings,
            layers,
            fast_layers,
            output,
            fast_output,
            fast_norm,
            norm,
            semantic_token_id,
            cfg: cfg.clone(),
        })
    }

    fn embed(&self, x: &Tensor) -> Result<Tensor> {
        // Embed the initial semantic tokens
        let semantic_tokens = x.index_select(&Tensor::from_vec(vec![1u32], 1, x.device())?, 1)?;
        let mut vocab_embeds: Vec<Tensor> = vec![self.embeddings.forward(&semantic_tokens)?];

        for i in 0..self.cfg.num_codebooks {
            let indices_at_codebook = x
                .index_select(&Tensor::from_vec(vec![(i + 1) as u32], 1, x.device())?, 1)?
                + (i * self.cfg.codebook_size) as f64;
            let emb = self.codebook_embeddings.forward(&indices_at_codebook?)?;
            // Zero out tokens where the semantic token is not the designated ID
            let emb = semantic_tokens
                .eq(self.semantic_token_id)?
                .where_cond(&emb.zeros_like()?, &emb)?;
            vocab_embeds.push(emb)
        }
        let x = Tensor::stack(&vocab_embeds, 3)?;
        x.sum(3)
    }

    /// Returns (logits, hidden_states)
    pub fn forward_generate(&mut self, inp: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = self.embed(inp)?;

        // TODO: See if making masks on-the-fly is a performance bottleneck
        let mask = prepare_attn_mask(1, inp.dim(D::Minus1)?)?;

        let x = self.layers.iter_mut().fold(Ok(x), |maybe_x, layer| {
            maybe_x.and_then(|x| layer.forward(&x, &mask)? + x)
        })?;
        let slow_out = self.norm.forward(&x)?;
        let token_logits = self.output.forward(&slow_out)?;

        Ok((token_logits, x))
    }

    /// Returns codebook_logits
    ///
    /// TODO: Handle `input_pos` and figure out what that's about
    pub fn forward_generate_fast(&mut self, x: &Tensor) -> Result<Tensor> {
        // let (token_logits, x) = self.parent_forward(inp, key_padding_mask)?;
        let x = x.reshape((1, 1, ()));
        // TODO: the shape is probably wrong, let's see how it comes out
        let fast_mask = prepare_attn_mask(1, self.cfg.num_codebooks)?;

        let x = self.fast_layers.iter_mut().fold(x, |maybe_x, layer| {
            maybe_x.and_then(|x| layer.forward(&x, &fast_mask) + x)
        })?;
        let fast_out = self.fast_norm.forward(&x)?;
        self.fast_output.forward(&fast_out)
    }
}

// struct KVCacheConfig {
//     max_batch_size: usize,
//     max_seq_len: usize,
//     n_heads: usize,
//     head_dim: usize,
//     dtype: DType,
//     device: Device,
// }
// struct KVCache {
//     k_cache: Tensor,
//     v_cache: Tensor,
// }

// impl KVCache {
//     pub fn load(config: &KVCacheConfig) -> Result<Self> {
//         // B * H * N_ctxt * D/H
//         let cache_shape = (
//             config.max_batch_size,
//             config.n_heads,
//             config.max_seq_len,
//             config.head_dim,
//         );
//         let k_cache = Tensor::zeros(cache_shape, config.dtype.clone(), &config.device)?;
//         let v_cache = Tensor::zeros(cache_shape, config.dtype.clone(), &config.device)?;
//         Ok(Self { k_cache, v_cache })
//     }

//     pub fn update(&mut self, input_pos: &Tensor, k_val: &Tensor) -> Result<(Tensor, Tensor)> {
//         assert!(&input_pos.dims1()? == &k_val.dims4()?.2);

//         let k_out = self.k_cache;
//         let v_out = self.v_cache;
//         let k_out = // Would use slice_assign or equivalent on the tensors for this

//     }
// }

// pub struct RMSNorm {
//     eps: f64,
//     weight: Linear,
// }

// impl RMSNorm {
//     pub fn load(vb: &VarBuilder, dim: usize, eps: Option<f64>) -> Result<Self> {
//         let eps = eps.unwrap_or(1e-5);
//         let weight = Linear::new(vb.get(dim, "weight")?, None);
//         Ok(Self { eps, weight })
//     }

//     fn _norm(&self, x: &Tensor) -> Result<Tensor> {
//         let mu = x.powf(2.0)?.mean_keepdim(D::Minus1)?;
//         // rsqrt isn't directly implemented in Candle
//         x.mul(&Tensor::ones_like(&x)?.div(&(mu + self.eps)?)?)
//     }
// }

// impl Module for RMSNorm {
//     fn forward(&self, xs: &Tensor) -> Result<Tensor> {
//         let output = self._norm(&xs)?;
//         self.weight.forward(&output)
//     }
// }
