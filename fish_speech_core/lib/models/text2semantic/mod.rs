use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{
    embedding, ops::silu, ops::softmax_last_dim, Embedding, Linear, Module, RmsNorm, VarBuilder,
};
use candle_transformers::utils::repeat_kv;

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

impl BaseModelArgs {
    pub fn fish_speech_1_2() -> Self {
        Self {
            model_type: "base".to_string(),
            vocab_size: 32000,
            n_layer: 24,
            n_fast_layer: 4,
            n_head: 16,
            dim: 1024,
            intermediate_size: Some(4096),
            n_local_heads: 2,
            head_dim: 64,
            rope_base: 1000000.0,
            norm_eps: 1e-6,
            max_seq_len: 4096,
            dropout: 0.0,
            tie_word_embeddings: false,
            attention_qkv_bias: false,
            codebook_size: 1024,
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
                    config.dim,
                    config.intermediate_size.unwrap_or(config.dim * 4),
                ),
                "w2.weight",
            )?,
            None,
        );
        let w3 = Linear::new(
            vb.get(
                (
                    config.intermediate_size.unwrap_or(config.dim * 4),
                    config.dim,
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
        let x = silu(&self.w1.forward(xs)?)? * self.w3.forward(xs)?;
        self.w2.forward(&x?)
    }
}

pub struct Cache {
    /// TODO: Does this require Arc<Mutex>>?
    kvs: Option<(Tensor, Tensor)>,
}

impl Cache {
    pub fn new() -> Result<Self> {
        // Precompute freqs_cis
        Ok(Self { kvs: None })
    }
}

/// Returns (cos, sin) for the full possible batch size
fn precompute_freqs_cis(
    config: &BaseModelArgs,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let n_elem = config.dim / config.n_head;
    let theta: Vec<_> = (0..n_elem)
        .step_by(2)
        .map(|i| 1f32 / config.rope_base.powf(i as f32 / n_elem as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, config.max_seq_len as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((config.max_seq_len, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?.to_dtype(dtype)?;
    let sin = idx_theta.sin()?.to_dtype(dtype)?;
    Ok((cos, sin))
}

/// Copied from phi3.rs
fn get_mask(size: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (size, size), device)
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

pub struct Attention {
    n_head: usize,
    head_dim: usize,
    n_local_heads: usize,
    dim: usize,
    wqkv: Linear,
    wo: Linear,
    cache: Cache,
}

impl Attention {
    pub fn load(vb: &VarBuilder, config: &BaseModelArgs) -> Result<Self> {
        let total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim;
        // KQV for all heads, but in a batch
        let wqkv = Linear::new(vb.get((total_head_dim, config.dim), "wqkv.weight")?, None);
        let wo = Linear::new(vb.get((config.dim, config.dim), "wo.weight")?, None);

        let cache = Cache::new()?;
        // let freqs_cis = Tensor::stack(&[&cache.cos, &cache.sin], D::Minus1)?;
        // freqs_cis.write_npy("freqs_cis_rust.npy")?;

        Ok(Self {
            n_head: config.n_head,
            head_dim: config.head_dim,
            n_local_heads: config.n_local_heads,
            dim: config.dim,
            wqkv,
            wo,
            // TODO configure this, improve cache handling
            cache,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let q_embed = candle_nn::rotary_emb::rope_i(&q, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope_i(&k, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }

    /// Standard inefficient SDPA
    fn scaled_dot_product_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_mask: &Tensor,
    ) -> Result<Tensor> {
        let scale_factor = (self.head_dim as f64).sqrt();

        let attn_weight = query.matmul(&(key.t()? / scale_factor)?)?;
        let attn_weight = masked_fill(
            &attn_weight,
            &attn_mask.broadcast_left((1, self.n_head))?,
            f32::NEG_INFINITY,
        )?;
        let attn_weight = softmax_last_dim(&attn_weight)?;
        // Ignoring dropout until we implement training
        attn_weight.matmul(&value.contiguous()?)
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        mask: &Tensor,
        freqs_cis: (&Tensor, &Tensor),
    ) -> Result<Tensor> {
        let (bsz, seqlen, _) = x.dims3()?;

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
            .reshape((bsz, seqlen, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        // query_states.write_npy("q1_before_rope_rust.npy")?;
        let key_states = key_states
            .reshape((bsz, seqlen, self.n_local_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let value_states = value_states
            .reshape((bsz, seqlen, self.n_local_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Logic copied from phi3.rs
        let seqlen_offset = match &self.cache.kvs {
            None => 0,
            Some((prev_k, _)) => prev_k.dim(2)?,
        };
        let (query_states, key_states) =
            self.apply_rotary_emb_qkv(&query_states, &key_states, freqs_cis.0, freqs_cis.1)?;
        // query_states.write_npy("q1_after_rope_rust.npy")?;
        let (key_states, value_states) = match &self.cache.kvs {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.cache.kvs = Some((key_states.clone(), value_states.clone()));

        // Repeat KV cache
        let key_states = repeat_kv(key_states, self.n_head / self.n_local_heads)?.contiguous()?;
        let value_states =
            repeat_kv(value_states, self.n_head / self.n_local_heads)?.contiguous()?;

        // TODO: Add optional flash attention
        let y =
            self.scaled_dot_product_attention(&query_states, &key_states, &value_states, mask)?;
        let y = y.transpose(1, 2)?.reshape((bsz, seqlen, self.dim))?;
        // y.write_npy("y1_after_sdpa_rust.npy")?;

        self.wo.forward(&y)
    }

    pub fn clear_cache(&mut self) {
        self.cache.kvs = None;
    }
}

pub struct TransformerBlock {
    pub attention: Attention,
    feed_forward: FeedForward,
    ffn_norm: RmsNorm,
    attention_norm: RmsNorm,
}

impl TransformerBlock {
    pub fn load(vb: &VarBuilder, cfg: &BaseModelArgs) -> Result<Self> {
        let attention = Attention::load(&vb.pp("attention"), cfg)?;
        let feed_forward = FeedForward::load(&vb.pp("feed_forward"), cfg)?;
        let ffn_norm = RmsNorm::new(vb.get(cfg.dim, "ffn_norm.weight")?, cfg.norm_eps);
        let attention_norm = RmsNorm::new(vb.get(cfg.dim, "attention_norm.weight")?, cfg.norm_eps);

        Ok(Self {
            attention,
            feed_forward,
            ffn_norm,
            attention_norm,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        mask: &Tensor,
        freqs_cis: (&Tensor, &Tensor),
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.attention_norm.forward(x)?;
        let x = (self.attention.forward(&x, mask, freqs_cis)? + residual)?;
        // x.write_npy("first_h_out_rust.npy")?;
        let residual = &x;
        let x = residual + self.feed_forward.forward(&self.ffn_norm.forward(&x)?);
        // x?.write_npy("block_1_out_rust.npy")?;
        // panic!("Prematurely bailing after first attention");
        x
    }
}

pub struct DualARTransformer {
    embeddings: Embedding,
    codebook_embeddings: Embedding,
    fast_layers: Vec<TransformerBlock>,
    pub fast_embeddings: Embedding,
    layers: Vec<TransformerBlock>,
    output: Linear,
    fast_output: Linear,
    norm: RmsNorm,
    fast_norm: RmsNorm,
    semantic_token_id: i64,
    freqs_cis: (Tensor, Tensor),
    pub cfg: BaseModelArgs,
}

impl DualARTransformer {
    pub fn load(vb: &VarBuilder, cfg: &BaseModelArgs, semantic_token_id: i64) -> Result<Self> {
        let embeddings = embedding(cfg.vocab_size, cfg.dim, vb.pp("embeddings"))?;
        let codebook_embeddings = Embedding::new(
            vb.get(
                (cfg.codebook_size * cfg.num_codebooks, cfg.dim),
                "codebook_embeddings.weight",
            )?,
            cfg.dim,
        );
        let layers: Result<Vec<TransformerBlock>> = (0..cfg.n_layer)
            .map(|l| TransformerBlock::load(&vb.pp(format!("layers.{}", l)), cfg))
            .collect();
        let layers = layers?;
        let norm = RmsNorm::new(vb.get(cfg.dim, "norm.weight")?, cfg.norm_eps);
        let output = Linear::new(vb.get((cfg.vocab_size, cfg.dim), "output.weight")?, None);
        let fast_embeddings = Embedding::new(
            vb.get((cfg.codebook_size, cfg.dim), "fast_embeddings.weight")?,
            cfg.dim,
        );
        let fast_layers: Result<Vec<TransformerBlock>> = (0..cfg.n_fast_layer)
            .map(|l| TransformerBlock::load(&vb.pp(format!("fast_layers.{}", l)), cfg))
            .collect();
        let fast_layers = fast_layers?;
        let fast_norm = RmsNorm::new(vb.get(cfg.dim, "fast_norm.weight")?, cfg.norm_eps);
        let fast_output = Linear::new(
            vb.get((cfg.dim, cfg.codebook_size), "fast_output.weight")?,
            None,
        );
        let freqs_cis = precompute_freqs_cis(cfg, vb.device(), vb.dtype())?;

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
            freqs_cis,
        })
    }

    fn embed(&self, x: &Tensor) -> Result<Tensor> {
        // Embed the initial semantic tokens
        let token_codebooks = x.chunk(self.cfg.num_codebooks + 1, 0)?;
        assert!(
            token_codebooks.len() == self.cfg.num_codebooks + 1,
            "Input tokens must have num_codebooks + 1 codebooks!"
        );
        let semantic_tokens = &token_codebooks[0];

        let mut vocab_embeds: Vec<Tensor> = vec![self.embeddings.forward(semantic_tokens)?];

        for i in 0..(self.cfg.num_codebooks) {
            let shifted_indices = &token_codebooks[i + 1] + (i * self.cfg.codebook_size) as f64;
            let emb = self.codebook_embeddings.forward(&shifted_indices?)?;
            // Zero out tokens where the semantic token is not the designated ID
            let emb_mask = &token_codebooks[0]
                .eq(self.semantic_token_id)?
                .unsqueeze(D::Minus1)?
                .to_dtype(emb.dtype())?;
            let emb = emb.broadcast_mul(emb_mask)?;
            vocab_embeds.push(emb)
        }
        let x = Tensor::stack(&vocab_embeds, 3)?;
        x.sum(3)
    }

    /// Returns (logits, hidden_states)
    pub fn forward_generate(&mut self, inp: &Tensor, input_pos: usize) -> Result<(Tensor, Tensor)> {
        let (__size, seq_len) = inp.dims2()?;
        let mut x = self.embed(inp)?;
        x.write_npy("final_prompt_emb_rust.npy")?;

        // TODO: See if making masks on-the-fly is a performance bottleneck
        let mask = get_mask(seq_len - input_pos, x.device())?;

        let (cos_full, sin_full) = &self.freqs_cis;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            x = layer.forward(
                &x,
                &mask,
                (
                    &cos_full.i(input_pos..input_pos + seq_len)?,
                    &sin_full.i(input_pos..input_pos + seq_len)?,
                ),
            )?;
        }
        x.write_npy("x_end_rs.npy")?;

        let x = x.narrow(1, seq_len - 1, 1)?;
        let slow_out = self.norm.forward(&x)?;
        let token_logits = self.output.forward(&slow_out)?;

        // Only calculate the logits of last_token
        Ok((token_logits, x))
    }

    /// Returns codebook_logits
    ///
    /// TODO: Handle `input_pos` and figure out what that's about
    pub fn forward_generate_fast(&mut self, x: &Tensor, input_pos: usize) -> Result<Tensor> {
        let (bsz, seq_len, _) = x.dims3()?;
        // This is a dirty hack but it will work for now
        let fast_mask = Tensor::from_vec(
            vec![u8::from(true); input_pos + 1],
            input_pos + 1,
            x.device(),
        )?
        .unsqueeze(0)?
        .repeat(bsz)?;

        let x = x.reshape((1, 1, ()));

        let (cos_full, sin_full) = &self.freqs_cis;
        let freqs_cis = (
            &cos_full.i((input_pos..(input_pos + seq_len), ..))?,
            &sin_full.i((input_pos..input_pos + seq_len, ..))?,
        );
        let x = self.fast_layers.iter_mut().fold(x, |maybe_x, layer| {
            maybe_x.and_then(|x| layer.forward(&x, &fast_mask, freqs_cis))
        })?;
        let fast_out = self.fast_norm.forward(&x)?;
        self.fast_output.forward(&fast_out)
    }

    pub fn clear_fast_layer_caches(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.attention.clear_cache();
        }
    }
}
