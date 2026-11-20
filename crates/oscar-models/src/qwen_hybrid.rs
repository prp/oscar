//! What the interval hybrids, qwen35_moe and qwen4exp, share beyond the GDN layer: every Nth block full attention.

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_core::gguf::{ArchDeclaration, Content, LayerAttention};
use mistralrs_core::kv_cache::RecurrentLayerConfig;
use mistralrs_core::layers::Qwen3VLRotaryEmbedding;

use crate::attention::{load_gated_attention_layer, FullAttentionLayer};
use crate::declare::Meta;
use crate::gdn::{GatedDeltaNet, GdnDims};

/// The header keys the family reads the same way.
#[derive(Debug, Clone)]
pub struct GdnProps {
    pub head_count: usize,
    pub head_count_kv: usize,
    /// The causal forward's blocks, the trailing draft blocks out.
    pub n_layer: usize,
    pub embedding_length: usize,
    /// `attention.key_length`, equal to `attention.value_length` here.
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_freq_base: f32,
    /// The rotated prefix of each head.
    pub rope_dim_count: usize,
    pub rope_dimension_sections: Vec<u32>,
    pub full_attention_interval: usize,
    pub expert_used_count: usize,
    pub ssm_conv_kernel: usize,
    pub ssm_group_count: usize,
    pub ssm_state_size: usize,
    pub ssm_time_step_rank: usize,
}

/// A block's mixer: Gated DeltaNet, or every `full_attention_interval`th block's gated full attention.
pub(crate) enum Mixer {
    Linear(GatedDeltaNet),
    Full(FullAttentionLayer),
}

impl GdnProps {
    /// The family's keys, and four it reads nowhere that a family file states.
    pub(crate) fn parse(m: &Meta<'_>) -> Result<Self> {
        for k in ["expert_count", "expert_feed_forward_length", "expert_shared_feed_forward_length", "ssm.inner_size"] {
            m.get(k)?;
        }
        let rope_dim_count = m.usize("rope.dimension_count")?;
        Ok(Self {
            head_count: m.usize("attention.head_count")?,
            head_count_kv: m.usize("attention.head_count_kv")?,
            n_layer: crate::declare::trunk(m)?,
            embedding_length: m.usize("embedding_length")?,
            head_dim: m.usize("attention.key_length")?,
            rms_norm_eps: m.f64("attention.layer_norm_rms_epsilon")? as f32,
            max_seq_len: m.int_or("context_length", 4096) as usize,
            rope_freq_base: m.f64("rope.freq_base")? as f32,
            rope_dim_count,
            rope_dimension_sections: crate::declare::rope_sections(m, rope_dim_count),
            full_attention_interval: m.usize("full_attention_interval")?,
            expert_used_count: m.usize("expert_used_count")?,
            ssm_conv_kernel: m.usize("ssm.conv_kernel")?,
            ssm_group_count: m.usize("ssm.group_count")?,
            ssm_state_size: m.usize("ssm.state_size")?,
            ssm_time_step_rank: m.usize("ssm.time_step_rank")?,
        })
    }

    pub fn dims(&self) -> GdnDims {
        GdnDims {
            head_dim: self.ssm_state_size,
            num_k_heads: self.ssm_group_count,
            num_v_heads: self.ssm_time_step_rank,
            conv_kernel: self.ssm_conv_kernel,
        }
    }

    pub fn is_full_attn(&self, i: usize) -> bool {
        (i + 1) % self.full_attention_interval == 0
    }

    /// Block `i`'s mixer; `v_reordered` is the GDN writer's v-head order, `sigmoid_gate` qwen4exp's norm gate.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mixer<R: std::io::Seek + std::io::Read>(
        &self, ct: &mut Content<'_, R>, prefix: &str, i: usize, device: &Device, rope: &Arc<Qwen3VLRotaryEmbedding>,
        dtype: DType, v_reordered: bool, sigmoid_gate: bool,
    ) -> Result<Mixer> {
        Ok(if self.is_full_attn(i) {
            Mixer::Full(load_gated_attention_layer(ct, prefix, device, self, rope.clone(), dtype)?)
        } else {
            Mixer::Linear(crate::gdn::load_gdn_layer(ct, prefix, device, self, dtype, v_reordered, sigmoid_gate)?)
        })
    }

    /// Layer kinds, the GDN state's layout with an architecture's per-layer overrides, and widths: the one source.
    pub(crate) fn declare(
        &self, m: &Meta<'_>, overrides: Vec<(usize, RecurrentLayerConfig)>,
    ) -> Result<ArchDeclaration> {
        let kv_heads = self.head_count_kv as u64;
        let nextn = crate::declare::nextn(m);
        crate::declare::build(
            m,
            crate::declare::Decl {
                blocks: self.n_layer + nextn,
                nextn,
                leading_dense: crate::declare::leading_dense(m),
                attention_of: |i| {
                    if self.is_full_attn(i) {
                        LayerAttention::Kv { heads: kv_heads }
                    } else {
                        LayerAttention::Recurrent
                    }
                },
                head_count: self.head_count,
                key_len: Some(self.head_dim),
                value_len: None,
                kv_lora_rank: None,
                per_token: self.expert_used_count,
                recurrent: Some((self.dims().recurrent(self.ssm_conv_kernel), overrides)),
                hidden: self.embedding_length,
                max_seq_len: self.max_seq_len,
            },
        )
    }

    /// The full-attention layers' MRoPE (GDN uses none).
    pub fn mrope(&self, device: &Device) -> Result<Arc<Qwen3VLRotaryEmbedding>> {
        let sections = self.rope_dimension_sections.iter().map(|x| *x as usize).collect();
        Ok(Arc::new(Qwen3VLRotaryEmbedding::new(self.rope_freq_base, self.rope_dim_count, device, sections)?))
    }
}

/// Text-only MRoPE positions `(3, batch, seq_len)`, the three dims sharing each sequence's `[start, start + seq_len)`.
fn text_position_ids(start_offsets: &[usize], seq_len: usize, device: &Device) -> Result<Tensor> {
    let batch = start_offsets.len();
    let mut data = Vec::with_capacity(batch * seq_len);
    for &start in start_offsets {
        for i in 0..seq_len {
            data.push((start + i) as i64);
        }
    }
    let positions = Tensor::from_vec(data, (batch, seq_len), device)?;
    positions.unsqueeze(0)?.expand((3, batch, seq_len))
}

/// The MRoPE's F32 cos/sin at the forward's positions, `x` the `(batch, seq_len)` token ids.
pub fn mrope_cos_sin(rope: &Qwen3VLRotaryEmbedding, offsets: &[usize], x: &Tensor) -> Result<(Tensor, Tensor)> {
    rope.compute_cos_sin(&text_position_ids(offsets, x.dim(1)?, x.device())?, DType::F32)
}
