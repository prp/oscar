#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! GLM-5.2 (`glm-dsa`): absorbed MLA and a routed MoE; the DSA indexer ignored, the dense-MLA fallback as llama.cpp's.

use std::sync::Arc;

use crate::embedding::QuantizedEmbedding;
use candle_core::{DType, Device, Result, Tensor};
use mistralrs_core::device_map::DeviceMapper;
use mistralrs_core::gguf::Content;
use mistralrs_core::gguf::{ArchDeclaration, LayerAttention};
use mistralrs_core::kv_cache::{EitherCache, NormalCache};
use mistralrs_core::layers::{QRmsNorm, RotaryEmbedding};
use mistralrs_core::paged_attention::AttentionImplementation;
use mistralrs_core::text_models_inputs_processor::PagedAttentionInputMetadata;
use mistralrs_core::utils::model_config as ModelConfig;

use crate::attention::{MlaAttention, MlaDims, PagedLatentRun};
use crate::declare::Meta;
use crate::draft::DraftPagedView;
use crate::ffn::GatedMlp;
use crate::latent::MlaProps;
use crate::moe::{Ffn, RoutedLayer};
use crate::plain::{PlainAttention, PlainLayer, PlainTrunk};
use oscar_kernels::ExpertActivation;

const ARCHS: &[&str] = &["glm-dsa"];

pub type ModelWeights = PlainTrunk<MlaAttention>;

impl PlainAttention for MlaAttention {
    type Shared = MlaDims;
    type Run<'a> = PagedLatentRun;
    const LATE_TAP: bool = true;
    const DRAFTS: bool = false;
    fn engine_run<'a>(
        x: &Tensor, metadata: Option<(Vec<(Tensor, Tensor)>, &'a PagedAttentionInputMetadata)>,
    ) -> Result<PagedLatentRun> {
        PagedLatentRun::new(x, metadata)
    }
    fn draft_run<'a>(_view: &'a DraftPagedView, _meta: &'a PagedAttentionInputMetadata) -> Result<PagedLatentRun> {
        candle_core::bail!("no MLA architecture drafts")
    }
    fn forward(
        &self, d: &MlaDims, x: &Tensor, start_offsets: &[usize], run: &PagedLatentRun, i: usize,
    ) -> Result<Tensor> {
        self.forward_attn(d, x, start_offsets, run, i)
    }
}

/// The header keys GLM-5.2 reads over the MLA family's.
pub(crate) struct PropsGGUF {
    pub mla: MlaProps,
    pub rope_freq_base: f32,
}

impl PropsGGUF {
    fn read(m: &Meta<'_>) -> Result<Self> {
        m.usize("expert_count")?;
        Ok(Self { mla: MlaProps::read(m)?, rope_freq_base: m.f64_or("rope.freq_base", 10_000.0) as f32 })
    }
}

/// Every trunk block holds an absorbed-MLA latent row; the dense lead and the MTP block as the file states.
pub(crate) fn declare<R: std::io::Seek + std::io::Read>(ct: &Content<'_, R>) -> Result<ArchDeclaration> {
    let m = Meta::of(ct.get_metadata(), ARCHS)?;
    let props = PropsGGUF::read(&m)?.mla;
    // `props.n_layer` is the trunk; the MTP block is a block of its own.
    let nextn = crate::declare::nextn(&m);
    let blocks = props.n_layer + nextn;
    let heads = m.kv_heads(blocks, false)?;
    crate::declare::build(
        &m,
        crate::declare::Decl {
            blocks,
            nextn,
            leading_dense: props.leading_dense,
            attention_of: |i| LayerAttention::Kv { heads: heads[i] },
            head_count: props.head_count,
            key_len: None,
            value_len: None,
            kv_lora_rank: Some(props.kv_lora_rank),
            per_token: props.expert_used_count,
            recurrent: None,
            hidden: props.hidden,
            max_seq_len: props.max_seq_len,
        },
    )
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>, device: &Device, mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation, dtype: DType,
    ) -> Result<Self> {
        crate::load::admit(&attention_mechanism, &*mapper, device)?;
        let PropsGGUF { mla: props, rope_freq_base } = PropsGGUF::read(&Meta::of(ct.get_metadata(), ARCHS)?)?;

        let tok_embeddings = QuantizedEmbedding::load(&mut ct, device, props.hidden, dtype)?;
        let norm = QRmsNorm::new(ct.tensor("output_norm.weight", device)?, props.rms_norm_eps)?;
        let output = crate::load::output_head(&mut ct, device)?;

        // rope_interleave ⇒ ROPE_TYPE_NORM ⇒ adjacent-pair rotation (rope_i).
        let rotary = Arc::new(RotaryEmbedding::new(
            rope_freq_base,
            props.rope_dim,
            props.max_seq_len,
            device,
            false,
            DType::F32,
        )?);

        let mut layers = Vec::with_capacity(props.n_layer);
        for layer_idx in crate::load::layers(props.n_layer) {
            let prefix = format!("blk.{layer_idx}");

            let attn = MlaAttention::load(&mut ct, &prefix, device, props.rms_norm_eps, false)?;
            let mlp = if layer_idx >= props.leading_dense {
                let router = props.router(&mut ct, &prefix, device, 0.0)?;
                let (silu, ordinal) = (ExpertActivation::Silu, layer_idx - props.leading_dense);
                Ffn::Moe(RoutedLayer::load(&mut ct, &prefix, ordinal, device, router, silu, Some(silu))?)
            } else {
                Ffn::Dense(GatedMlp::load(&mut ct, &prefix, "", device, ExpertActivation::Silu)?)
            };

            layers.push(PlainLayer {
                attn_norm: crate::load::norm(&mut ct, &prefix, "attn_norm.weight", device, props.rms_norm_eps)?,
                attn,
                mlp,
                ffn_norm: crate::load::norm(&mut ct, &prefix, "ffn_norm.weight", device, props.rms_norm_eps)?,
            })
        }

        Ok(Self {
            tok_embeddings,
            layers,
            shared: props.mla_dims(dtype, Some(rotary)),
            norm,
            output: crate::load::quant(output)?,
            device: device.clone(),
            cache: EitherCache::Normal(NormalCache::new(props.n_layer, props.max_seq_len)),
            max_seq_len: props.max_seq_len,
            leading_dense: props.leading_dense,
        })
    }
}
