pub mod utils;

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{
    embedding, kv_cache::KvCache, ops::silu, ops::softmax_last_dim, Embedding, Linear, Module,
    RmsNorm, VarBuilder,
};
use serde::Deserialize;
use serde_json;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
pub struct BaseModelArgs {
    pub attention_qkv_bias: bool,
    pub codebook_size: usize,
    pub dim: usize,
    pub dropout: f32,
    pub head_dim: usize,
    pub initializer_range: f32,
    pub intermediate_size: Option<usize>,
    pub max_seq_len: usize,
    pub model_type: String,
    pub n_fast_layer: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub n_local_heads: usize,
    pub norm_eps: f64,
    pub num_codebooks: usize,
    pub rope_base: f32,
    pub tie_word_embeddings: bool,
    pub use_gradient_checkpointing: bool,
    pub vocab_size: usize,
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
            initializer_range: 0.02,
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
            use_gradient_checkpointing: true,
        }
    }

    pub fn from_json_file(path: PathBuf) -> serde_json::Result<Self> {
        let file = File::open(path).expect("Could not open the file");
        let reader = BufReader::new(file);
        let config: Self = serde_json::from_reader(reader)?;
        Ok(config)
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
    let on_true = Tensor::new(on_true, on_false.device())?
        .to_dtype(on_false.dtype())?
        .broadcast_as(shape.dims())?;
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
    cache: KvCache,
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    softmax_scale: f32,
    attn_mask: Option<&Tensor>,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(query, key, value, softmax_scale, attn_mask.is_some())
}

impl Attention {
    pub fn load(vb: &VarBuilder, config: &BaseModelArgs, is_fast: bool) -> Result<Self> {
        let total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim;
        // KQV for all heads, but in a batch
        let wqkv = Linear::new(vb.get((total_head_dim, config.dim), "wqkv.weight")?, None);
        let wo = Linear::new(vb.get((config.dim, config.dim), "wo.weight")?, None);

        let cache = KvCache::new(2, if is_fast { config.num_codebooks } else { 1024 });

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
        let q_embed = candle_nn::rotary_emb::rope_i(q, cos, sin)?;
        let k_embed = candle_nn::rotary_emb::rope_i(k, cos, sin)?;
        Ok((q_embed, k_embed))
    }

    /// Standard inefficient SDPA
    fn scaled_dot_product_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        softmax_scale: f32,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let attn_weight = query.matmul(&(key.t()? * softmax_scale as f64)?)?;
        // Masking w/ KV cache is redundant
        let attn_weight = match attn_mask {
            None => attn_weight,
            Some(attn_mask) => masked_fill(
                &attn_weight,
                &attn_mask.broadcast_left((1, self.n_head))?,
                f32::NEG_INFINITY,
            )?,
        };
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
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((bsz, seqlen, self.n_local_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((bsz, seqlen, self.n_local_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) = self.apply_rotary_emb_qkv(
            &query_states.contiguous()?,
            &key_states.contiguous()?,
            freqs_cis.0,
            freqs_cis.1,
        )?;

        let (key_states, value_states) = self
            .cache
            .append(&key_states.contiguous()?, &value_states.contiguous()?)?;

        // Length changes after pulling
        let kv_seqlen = key_states.dim(2)?;
        let n_rep = self.n_head / self.n_local_heads;
        // TODO: Consider whether there's a better way to do this with 2 copy2ds instead of ucopy
        // https://github.com/huggingface/candle/pull/2043 got the duplication wrong but there might still be something there
        let key_states = key_states
            .unsqueeze(2)?
            .expand((bsz, self.n_local_heads, n_rep, kv_seqlen, self.head_dim))?
            .reshape((bsz, self.n_local_heads * n_rep, kv_seqlen, self.head_dim))?;
        let value_states = value_states
            .unsqueeze(2)?
            .expand((bsz, self.n_local_heads, n_rep, kv_seqlen, self.head_dim))?
            .reshape((bsz, self.n_local_heads * n_rep, kv_seqlen, self.head_dim))?;

        let scale_factor = 1f32 / (self.head_dim as f32).sqrt();
        let mask = if seqlen > 1 { Some(mask) } else { None };
        #[cfg(feature = "flash-attn")]
        let y = {
            let q = query_states.transpose(1, 2)?;
            let k = key_states.transpose(1, 2)?;
            let v = value_states.transpose(1, 2)?;
            flash_attn(&q, &k, &v, scale_factor, mask)?.transpose(1, 2)?
        };

        #[cfg(not(feature = "flash-attn"))]
        let y = self.scaled_dot_product_attention(
            &query_states,
            &key_states,
            &value_states,
            scale_factor,
            mask,
        )?;

        let y = y.transpose(1, 2)?.reshape((bsz, seqlen, self.dim))?;

        self.wo.forward(&y)
    }

    pub fn clear_cache(&mut self) {
        self.cache.reset();
    }
}

pub struct TransformerBlock {
    pub attention: Attention,
    feed_forward: FeedForward,
    ffn_norm: RmsNorm,
    attention_norm: RmsNorm,
}

impl TransformerBlock {
    pub fn load(vb: &VarBuilder, cfg: &BaseModelArgs, is_fast: bool) -> Result<Self> {
        let attention = Attention::load(&vb.pp("attention"), cfg, is_fast)?;
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
        let x = (residual + self.attention.forward(&x, mask, freqs_cis)?)?;
        let residual = &x;
        residual + self.feed_forward.forward(&self.ffn_norm.forward(&x)?)
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
            .map(|l| TransformerBlock::load(&vb.pp(format!("layers.{}", l)), cfg, false))
            .collect();
        let layers = layers?;
        let norm = RmsNorm::new(vb.get(cfg.dim, "norm.weight")?, cfg.norm_eps);
        let output = Linear::new(vb.get((cfg.vocab_size, cfg.dim), "output.weight")?, None);
        let fast_embeddings = Embedding::new(
            vb.get((cfg.codebook_size, cfg.dim), "fast_embeddings.weight")?,
            cfg.dim,
        );
        let fast_layers: Result<Vec<TransformerBlock>> = (0..cfg.n_fast_layer)
            .map(|l| TransformerBlock::load(&vb.pp(format!("fast_layers.{}", l)), cfg, true))
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
        let semantic_tokens = x.i((0, ..))?;
        let codebook_tokens = x.i((1.., ..))?;
        assert!(
            x.dim(D::Minus2)? == self.cfg.num_codebooks + 1,
            "Input tokens must have num_codebooks + 1 codebooks!"
        );
        // Embed semantic tokens, re-add batch dim if single-batched
        let semantic_embeds = self.embeddings.forward(&semantic_tokens)?;
        let semantic_embeds = match semantic_embeds.rank() {
            2 => semantic_embeds.unsqueeze(0)?,
            _ => semantic_embeds,
        };

        // Offset the ranges for each codebook so they don't overlap
        let codebook_tokens_shifted = codebook_tokens.broadcast_add(
            &Tensor::arange_step(
                0,
                (self.cfg.num_codebooks * self.cfg.codebook_size) as u32,
                self.cfg.codebook_size as u32,
                x.device(),
            )?
            .unsqueeze(1)?,
        )?;
        let codebook_emb = self.codebook_embeddings.forward(&codebook_tokens_shifted)?;
        let emb_mask = semantic_tokens
            .eq(self.semantic_token_id)?
            .unsqueeze(D::Minus1)?
            .to_dtype(codebook_emb.dtype())?;
        let codebook_embeds = codebook_emb.broadcast_mul(&emb_mask)?;
        let x = Tensor::cat(&[semantic_embeds, codebook_embeds], 0)?;
        x.sum_keepdim(0)
    }

    /// Returns (logits, hidden_states)
    pub fn forward_generate(&mut self, inp: &Tensor, input_pos: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = inp.dim(1)?;
        let mut x = self.embed(inp)?;

        // TODO: See if making masks on-the-fly is a performance bottleneck
        // TODO: Handle input_pos. I'm not convinced it matters at bs=1
        let mask = get_mask(seq_len, x.device())?;

        let (cos_full, sin_full) = &self.freqs_cis;
        for layer in self.layers.iter_mut() {
            x = layer.forward(
                &x,
                &mask,
                (
                    &cos_full.i(input_pos..input_pos + seq_len)?,
                    &sin_full.i(input_pos..input_pos + seq_len)?,
                ),
            )?;
        }

        let x = x.narrow(1, seq_len - 1, 1)?;
        let slow_out = self.norm.forward(&x)?;
        let token_logits = self.output.forward(&slow_out)?;

        // Only calculate the logits of last_token
        Ok((token_logits, x))
    }

    /// Returns codebook_logits only
    pub fn forward_generate_fast(&mut self, x: &Tensor, input_pos: usize) -> Result<Tensor> {
        let (bsz, seq_len, _) = x.dims3()?;
        // This is a dirty hack but it will work for now
        let fast_mask = Tensor::from_vec(
            vec![u8::from(false); input_pos + 1],
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
        for layer in self.fast_layers.iter_mut() {
            layer.attention.clear_cache();
        }
    }

    pub fn clear_slow_layer_caches(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.attention.clear_cache();
        }
    }
}
