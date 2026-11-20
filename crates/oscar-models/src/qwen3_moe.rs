#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use crate::draft::DraftPagedView;
use crate::embedding::QuantizedEmbedding;
use crate::ffn::GatedMlp;
use crate::moe::{Ffn, RoutedLayer, Router};
use crate::plain::{PlainAttention, PlainLayer, PlainTrunk};
use candle_core::{DType, Device, Result, Tensor};
use mistralrs_core::attention::SdpaParams;
use mistralrs_core::device_map::DeviceMapper;
use mistralrs_core::gguf::Content;
use mistralrs_core::gguf::{ArchDeclaration, LayerAttention};
use mistralrs_core::kv_cache::{EitherCache, NormalCache};
use mistralrs_core::layers::{MatMul, QRmsNorm, RotaryEmbedding};
use mistralrs_core::paged_attention::AttentionImplementation;
use mistralrs_core::text_models_inputs_processor::PagedAttentionInputMetadata;
use mistralrs_core::utils::model_config as ModelConfig;
use mistralrs_quant::QuantMethod;
use oscar_kernels::ExpertActivation;

use crate::declare::Meta;

const ARCHS: &[&str] = &["qwen3", "qwen3moe"];

pub type ModelWeights = PlainTrunk<Qwen3Attention>;

/// What every layer's GQA shares.
pub struct Qwen3Dims {
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
    sdpa_params: SdpaParams,
    dtype: DType,
}

pub struct Qwen3Attention {
    attention_wq: Arc<dyn QuantMethod>,
    attention_wk: Arc<dyn QuantMethod>,
    attention_wv: Arc<dyn QuantMethod>,
    attention_wo: Arc<dyn QuantMethod>,
    q_norm: QRmsNorm,
    k_norm: QRmsNorm,
}

/// The block pools a forward attends over: the engine's, or a drafter's paged view of them.
pub enum Pools<'a> {
    Engine(Vec<(Tensor, Tensor)>),
    Draft(&'a DraftPagedView),
}

impl PlainAttention for Qwen3Attention {
    type Shared = Qwen3Dims;
    type Run<'a> = (Pools<'a>, &'a PagedAttentionInputMetadata);
    const LATE_TAP: bool = false;
    const DRAFTS: bool = true;
    fn engine_run<'a>(
        _x: &Tensor, metadata: Option<(Vec<(Tensor, Tensor)>, &'a PagedAttentionInputMetadata)>,
    ) -> Result<Self::Run<'a>> {
        let (pools, meta) =
            metadata.ok_or_else(|| candle_core::Error::Msg("qwen3moe: paged KV is the only path".into()))?;
        Ok((Pools::Engine(pools), meta))
    }
    fn draft_run<'a>(view: &'a DraftPagedView, meta: &'a PagedAttentionInputMetadata) -> Result<Self::Run<'a>> {
        Ok((Pools::Draft(view), meta))
    }

    fn forward(
        &self, d: &Qwen3Dims, x: &Tensor, start_offsets: &[usize], (pools, input_metadata): &Self::Run<'_>, i: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = MatMul.qmethod_matmul(x, &*self.attention_wq)?;
        let k = MatMul.qmethod_matmul(x, &*self.attention_wk)?;
        let v = MatMul.qmethod_matmul(x, &*self.attention_wv)?;

        let q = q.reshape((b_sz, seq_len, d.n_head, d.head_dim))?;
        let dims = (b_sz, seq_len, d.n_head, d.n_kv_head, d.head_dim);
        let (q, k, v) = crate::attention::normed_heads(q, &k, &v, dims, &self.q_norm, &self.k_norm)?;

        let (q, k) = d.rotary.forward(&q, &k, start_offsets)?;

        let (q, k, v) = (q.to_dtype(d.dtype)?, k.to_dtype(d.dtype)?, v.to_dtype(d.dtype)?);

        // The block pools are the engine's, or a drafter's paged view of them.
        let (key_cache, value_cache) = match pools {
            Pools::Engine(kv) => kv[i].clone(),
            Pools::Draft(view) => view.layer_pools(i)?,
        };
        let pools = (&key_cache, &value_cache);
        let y = crate::attention::paged_attention(&q, &k, &v, pools, input_metadata, &d.sdpa_params)?;
        let y = y.reshape((b_sz, seq_len, ()))?;

        let y = MatMul.qmethod_matmul(&y.to_dtype(x.dtype())?, &*self.attention_wo)?;
        Ok(y)
    }
}

/// The header keys the architecture reads, each refusal naming its key.
pub(crate) struct PropsGGUF {
    pub head_count: usize,
    pub head_count_kv: usize,
    pub n_layer: usize,
    pub embedding_length: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_freq_base: f32,
    pub key_length: usize,
    pub value_length: usize,
    pub expert_count: usize,
    pub experts_per_tok: usize,
}

impl PropsGGUF {
    fn read(m: &Meta<'_>) -> Result<Self> {
        let (embed, heads) = (m.usize("embedding_length")?, m.usize("attention.head_count")?);
        m.usize("expert_feed_forward_length")?;
        Ok(Self {
            head_count: heads,
            head_count_kv: m.usize("attention.head_count_kv")?,
            n_layer: crate::declare::trunk(m)?,
            embedding_length: embed,
            rms_norm_eps: m.f64("attention.layer_norm_rms_epsilon")? as f32,
            max_seq_len: m.int_or("context_length", 4096) as usize,
            rope_freq_base: m.f64_or("rope.freq_base", 10_000.0) as f32,
            key_length: m.int_or("attention.key_length", (embed / heads) as i64) as usize,
            value_length: m.int_or("attention.value_length", (embed / heads) as i64) as usize,
            expert_count: m.usize("expert_count")?,
            experts_per_tok: m.usize("expert_used_count")?,
        })
    }
}

/// Every trunk block holds KV; the dense lead, draft blocks and widths as the header states.
pub(crate) fn declare<R: std::io::Seek + std::io::Read>(ct: &Content<'_, R>) -> Result<ArchDeclaration> {
    let m = Meta::of(ct.get_metadata(), ARCHS)?;
    let props = PropsGGUF::read(&m)?;
    let nextn = crate::declare::nextn(&m);
    let heads = m.kv_heads(props.n_layer + nextn, false)?;
    crate::declare::build(
        &m,
        crate::declare::Decl {
            blocks: props.n_layer + nextn,
            nextn,
            leading_dense: crate::declare::leading_dense(&m),
            attention_of: |i| LayerAttention::Kv { heads: heads[i] },
            head_count: props.head_count,
            key_len: Some(props.key_length),
            value_len: Some(props.value_length),
            kv_lora_rank: None,
            per_token: props.experts_per_tok,
            recurrent: None,
            hidden: props.embedding_length,
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
        let m = Meta::of(ct.get_metadata(), ARCHS)?;
        let PropsGGUF {
            head_count,
            head_count_kv,
            n_layer,
            embedding_length,
            rms_norm_eps,
            max_seq_len,
            rope_freq_base,
            key_length,
            value_length,
            expert_count,
            experts_per_tok,
        } = PropsGGUF::read(&m)?;
        let leading_dense = crate::declare::leading_dense(&m);

        let tok_embeddings = QuantizedEmbedding::load(&mut ct, device, embedding_length, DType::F32)?;
        let norm = QRmsNorm::new(ct.tensor("output_norm.weight", device)?, rms_norm_eps)?;
        let output = crate::load::output_head(&mut ct, device)?;
        let mut layers = Vec::with_capacity(n_layer);

        let head_dim = key_length;
        if key_length != value_length {
            candle_core::bail!("expected key_length == value_length, got {key_length} != {value_length}");
        }

        let rotary = Arc::new(RotaryEmbedding::new(rope_freq_base, head_dim, max_seq_len, device, true, DType::F32)?);

        let mut routed = 0;
        for layer_idx in crate::load::layers(n_layer) {
            let prefix = format!("blk.{layer_idx}");

            let attention_wq = crate::load::qmm(&mut ct, &prefix, "attn_q.weight", device)?;
            let attention_wk = crate::load::qmm(&mut ct, &prefix, "attn_k.weight", device)?;
            let attention_wv = crate::load::qmm(&mut ct, &prefix, "attn_v.weight", device)?;
            let attention_wo = crate::load::qmm(&mut ct, &prefix, "attn_output.weight", device)?;

            let mlp = if expert_count > 0 {
                let router = Router::softmax(&mut ct, &prefix, device, experts_per_tok, None)?;
                let moe = RoutedLayer::load(&mut ct, &prefix, routed, device, router, ExpertActivation::Silu, None)?;
                routed += 1;
                Ffn::Moe(moe)
            } else {
                Ffn::Dense(GatedMlp::load(&mut ct, &prefix, "", device, ExpertActivation::Silu)?)
            };

            let q_norm = crate::load::norm(&mut ct, &prefix, "attn_q_norm.weight", device, rms_norm_eps)?;
            let k_norm = crate::load::norm(&mut ct, &prefix, "attn_k_norm.weight", device, rms_norm_eps)?;
            let attention_norm = crate::load::norm(&mut ct, &prefix, "attn_norm.weight", device, rms_norm_eps)?;
            let ffn_norm = crate::load::norm(&mut ct, &prefix, "ffn_norm.weight", device, rms_norm_eps)?;
            let attn = Qwen3Attention { attention_wq, attention_wk, attention_wv, attention_wo, q_norm, k_norm };
            layers.push(PlainLayer { attn_norm: attention_norm, attn, ffn_norm, mlp })
        }
        Ok(Self {
            tok_embeddings,
            layers,
            shared: Qwen3Dims {
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                rotary,
                sdpa_params: SdpaParams {
                    n_kv_groups: head_count / head_count_kv,
                    softcap: None,
                    softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                    sliding_window: None,
                    sinks: None,
                },
                dtype,
            },
            norm,
            output: crate::load::quant(output)?,
            device: device.clone(),
            cache: EitherCache::Normal(NormalCache::new(n_layer, max_seq_len)),
            max_seq_len,
            leading_dense,
        })
    }
}
