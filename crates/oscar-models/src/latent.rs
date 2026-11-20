//! What the absorbed-MLA architectures (GLM-5.2, GLM-5.3-Flash, K3) read alike, and the two latent hybrids' blocks.

use std::sync::Arc;

use candle_core::{DType, Device, Result};
use mistralrs_core::gguf::{ArchDeclaration, Content, LayerAttention};
use mistralrs_core::layers::RotaryEmbedding;

use crate::attention::{LatentMixer, MlaAttention, MlaDims};
use crate::declare::Meta;
use crate::kda::KdaWeights;
use crate::moe::Router;

/// Below the shipped 1M the rope tables and the KV gate scale with, admitting the (131072, 16384) serve row.
const MAX_SEQ_LEN_CAP: usize = 262_144;

/// The header keys every MLA architecture reads, each refusal naming its key.
pub(crate) struct MlaProps {
    /// The causal forward's blocks: the MTP block(s) carry experts but are no part of it.
    pub n_layer: usize,
    pub hidden: usize,
    pub head_count: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub kv_lora_rank: usize,
    pub key_length_mla: usize,
    pub value_length_mla: usize,
    /// The width of the un-absorbed key part: rotated on GLM-5.2, carried unrotated on K3, none on GLM-5.3-Flash.
    pub rope_dim: usize,
    pub leading_dense: usize,
    pub expert_used_count: usize,
    pub expert_weights_norm: bool,
    pub expert_weights_scale: f64,
}

impl MlaProps {
    /// The sigmoid, ungrouped router is structural, not a default: another gating would load into garbage.
    pub fn read(m: &Meta<'_>) -> Result<Self> {
        let gating = m.int_or("expert_gating_func", 2);
        if gating != 2 {
            candle_core::bail!("{}: expected sigmoid expert gating (2), got {gating}", m.arch);
        }
        let groups = m.int_or("expert_group_count", 1);
        if groups != 1 {
            candle_core::bail!("{}: expected 1 expert group, got {groups}", m.arch);
        }
        Ok(Self {
            n_layer: crate::declare::trunk(m)?,
            hidden: m.usize("embedding_length")?,
            head_count: m.usize("attention.head_count")?,
            rms_norm_eps: m.f64("attention.layer_norm_rms_epsilon")? as f32,
            max_seq_len: m.usize("context_length")?.min(MAX_SEQ_LEN_CAP),
            kv_lora_rank: m.usize("attention.kv_lora_rank")?,
            key_length_mla: m.usize("attention.key_length_mla")?,
            value_length_mla: m.usize("attention.value_length_mla")?,
            rope_dim: m.usize("rope.dimension_count")?,
            leading_dense: m.usize("leading_dense_block_count")?,
            expert_used_count: m.usize("expert_used_count")?,
            expert_weights_norm: m.bool_or("expert_weights_norm", true),
            expert_weights_scale: m.f64("expert_weights_scale")?,
        })
    }

    /// Absorbed logits equal the per-head ones: the scale is over the original qk dim, not the latent row.
    pub fn mla_dims(&self, dtype: DType, rope: Option<Arc<RotaryEmbedding>>) -> MlaDims {
        MlaDims {
            n_head: self.head_count,
            nope: self.key_length_mla - self.rope_dim,
            extra: self.rope_dim,
            kv_lora_rank: self.kv_lora_rank,
            v_head_dim: self.value_length_mla,
            softmax_scale: 1.0 / (self.key_length_mla as f64).sqrt(),
            dtype,
            rope,
        }
    }

    /// The `noaux_tc` router, `sum_eps` the reference's guard in its renormalising denominator.
    pub fn router<R: std::io::Seek + std::io::Read>(
        &self, ct: &mut Content<'_, R>, prefix: &str, device: &Device, sum_eps: f64,
    ) -> Result<Router> {
        let (norm, scale) = (self.expert_weights_norm, self.expert_weights_scale);
        Router::biased(ct, prefix, device, self.expert_used_count, norm, scale, sum_eps)
    }
}

/// The latent hybrids' keys over the MLA ones: `head_count_kv` a type marker per block, 1 MLA and 0 KDA (MLA is MQA
/// over the latent, so it is never a head count), and the KDA layers' widths.
pub(crate) struct LatentHybridProps {
    pub mla: MlaProps,
    pub head_count_kv: Vec<u64>,
    pub kda_head_dim: usize,
    pub ssm_conv_kernel: usize,
    pub kda_gate_lower_bound: f64,
}

impl std::ops::Deref for LatentHybridProps {
    type Target = MlaProps;
    fn deref(&self) -> &MlaProps {
        &self.mla
    }
}

impl LatentHybridProps {
    pub fn read(m: &Meta<'_>) -> Result<Self> {
        Ok(Self {
            mla: MlaProps::read(m)?,
            head_count_kv: m.kv_heads(m.usize("block_count")?, true)?,
            kda_head_dim: m.usize("kda.head_dim")?,
            ssm_conv_kernel: m.usize("ssm.conv_kernel")?,
            kda_gate_lower_bound: m.f64("kda.gate_lower_bound")?,
        })
    }

    /// Which blocks hold an MLA latent row and which KDA state, and that state's layout: `HybridCache`'s dimensions.
    /// `n_layer` is the trunk; the MTP blocks are blocks of their own, whose kind no count reads.
    pub fn declare(&self, m: &Meta<'_>) -> Result<ArchDeclaration> {
        let nextn = crate::declare::nextn(m);
        let kda = crate::kda::recurrent_config(self.head_count, self.kda_head_dim, self.ssm_conv_kernel);
        crate::declare::build(
            m,
            crate::declare::Decl {
                blocks: self.n_layer + nextn,
                nextn,
                leading_dense: self.leading_dense,
                attention_of: |i| match self.head_count_kv[i] {
                    0 => LayerAttention::Recurrent,
                    heads => LayerAttention::Kv { heads },
                },
                head_count: self.head_count,
                key_len: None,
                value_len: None,
                kv_lora_rank: Some(self.kv_lora_rank),
                per_token: self.expert_used_count,
                recurrent: Some((kda, vec![])),
                hidden: self.hidden,
                max_seq_len: self.max_seq_len,
            },
        )
    }

    /// Block `i`'s mixer: absorbed MLA (its `attn_gate` where `mla_gated`) under its marker, else KDA with `kda_gates`.
    pub fn mixer<R: std::io::Seek + std::io::Read>(
        &self, ct: &mut Content<'_, R>, prefix: &str, i: usize, device: &Device, mla_gated: bool, kda_gates: &[&str],
    ) -> Result<LatentMixer> {
        Ok(if self.head_count_kv[i] != 0 {
            LatentMixer::Mla(Box::new(MlaAttention::load(ct, prefix, device, self.rms_norm_eps, mla_gated)?))
        } else {
            LatentMixer::Kda(Box::new(KdaWeights::load(ct, prefix, device, self, kda_gates)?))
        })
    }
}
