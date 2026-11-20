#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! GLM-5.3-Flash (`glm5next`): KDA and NoPE MLA layers under mHC, the DSA indexer ignored as in GLM-5.2.

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_core::device_map::DeviceMapper;
use mistralrs_core::extract_logits;
use mistralrs_core::gguf::{ArchDeclaration, Content};
use mistralrs_core::kv_cache::{EitherCache, HybridCache};
use mistralrs_core::layers::{MatMul, QRmsNorm};
use mistralrs_core::paged_attention::AttentionImplementation;
use mistralrs_core::text_models_inputs_processor::PagedAttentionInputMetadata;
use mistralrs_core::utils::model_config as ModelConfig;
use mistralrs_quant::QuantMethod;
use oscar_kernels::ExpertActivation;

use crate::attention::{LatentMixer, MlaDims, PagedLatentRun};
use crate::declare::Meta;
use crate::embedding::QuantizedEmbedding;
use crate::ffn::GatedMlp;
use crate::latent::LatentHybridProps;
use crate::moe::{Ffn, RoutedLayer};

use crate::hc::{stream_update, HyperConnection};

struct LayerWeights {
    attn_norm: QRmsNorm,
    ffn_norm: QRmsNorm,
    attn: LatentMixer,
    mlp: Ffn,
    hc_attn: HyperConnection,
    hc_ffn: HyperConnection,
}

pub struct ModelWeights {
    tok_embeddings: QuantizedEmbedding,
    layers: Vec<LayerWeights>,
    mla: MlaDims,
    norm: QRmsNorm,
    output: Arc<dyn QuantMethod>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    hc_mult: usize,
    hc_eps: f64,
    sinkhorn_iters: usize,
    rms_norm_eps: f64,
    leading_dense: usize,
}

struct PropsGGUF {
    latent: LatentHybridProps,
    /// Per-layer swiglu clamps, indexed by GGUF block.
    clamp_exp: Vec<f32>,
    clamp_shexp: Vec<f32>,
    hc_mult: usize,
    hc_eps: f64,
    sinkhorn_iters: usize,
}

impl std::ops::Deref for PropsGGUF {
    type Target = LatentHybridProps;
    fn deref(&self) -> &LatentHybridProps {
        &self.latent
    }
}

const ARCH_PREFIX: &str = "glm5next";

pub(crate) fn declare<R: std::io::Seek + std::io::Read>(ct: &Content<'_, R>) -> Result<ArchDeclaration> {
    let m = Meta::of(ct.get_metadata(), &[ARCH_PREFIX])?;
    PropsGGUF::read(&m)?.declare(&m)
}

impl PropsGGUF {
    fn read(m: &Meta<'_>) -> Result<Self> {
        // NoPE is structural to the MLA below, not a default.
        let rope_dim = m.usize("rope.dimension_count")?;
        if rope_dim != 0 {
            candle_core::bail!(
                "glm5next: rope.dimension_count {rope_dim}, port implements NoPE (mla_use_nope; rope width 0)"
            );
        }

        let block_count = m.usize("block_count")?;

        // Clamps cover every block or the MoE ones: normalised per block.
        let per_block = |mut v: Vec<f32>, what: &str| -> Result<Vec<f32>> {
            let leading = m.usize("leading_dense_block_count")?;
            if v.len() == block_count {
                Ok(v)
            } else if v.len() + leading == block_count {
                let mut out = vec![v.first().copied().unwrap_or(0.0); leading];
                out.append(&mut v);
                Ok(out)
            } else {
                candle_core::bail!("glm5next: {what} has {} entries for {block_count} blocks", v.len())
            }
        };
        let clamp_exp = per_block(m.f32_array("swiglu_clamp_exp")?, "swiglu_clamp_exp")?;
        let clamp_shexp = per_block(m.f32_array("swiglu_clamp_shexp")?, "swiglu_clamp_shexp")?;

        Ok(Self {
            latent: LatentHybridProps::read(m)?,
            clamp_exp,
            clamp_shexp,
            hc_mult: m.usize("hyper_connection.count")?,
            hc_eps: m.f64("hyper_connection.epsilon")?,
            sinkhorn_iters: m.usize("hyper_connection.sinkhorn_iterations")?,
        })
    }
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>, device: &Device, mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation, dtype: DType,
    ) -> Result<Self> {
        crate::load::admit(&attention_mechanism, &*mapper, device)?;
        let props = PropsGGUF::read(&Meta::of(ct.get_metadata(), &[ARCH_PREFIX])?)?;

        let tok_embeddings = QuantizedEmbedding::load(&mut ct, device, props.hidden, dtype)?;
        let norm = QRmsNorm::new(ct.tensor("output_norm.weight", device)?, props.rms_norm_eps)?;
        let output = crate::load::output_head(&mut ct, device)?;

        let mut layers = Vec::with_capacity(props.n_layer);
        for layer_idx in crate::load::layers(props.n_layer) {
            let prefix = format!("blk.{layer_idx}");
            // No decoupled part: a cache row is the bare latent.
            let attn =
                props.mixer(&mut ct, &prefix, layer_idx, device, false, &["ssm_g_a.weight", "ssm_g_b.weight"])?;

            let clamp = |limit| ExpertActivation::SiluClamp { limit };
            let (exp, shexp) = (clamp(props.clamp_exp[layer_idx]), clamp(props.clamp_shexp[layer_idx]));
            let mlp = if layer_idx >= props.leading_dense {
                let router = props.router(&mut ct, &prefix, device, 1e-20)?;
                let ordinal = layer_idx - props.leading_dense;
                Ffn::Moe(RoutedLayer::load(&mut ct, &prefix, ordinal, device, router, exp, Some(shexp))?)
            } else {
                Ffn::Dense(GatedMlp::load(&mut ct, &prefix, "", device, exp)?)
            };

            layers.push(LayerWeights {
                attn_norm: crate::load::norm(&mut ct, &prefix, "attn_norm.weight", device, props.rms_norm_eps)?,
                ffn_norm: crate::load::norm(&mut ct, &prefix, "ffn_norm.weight", device, props.rms_norm_eps)?,
                attn,
                mlp,
                hc_attn: HyperConnection::load(&mut ct, &prefix, "attn", device, props.hc_mult, props.hidden)?,
                hc_ffn: HyperConnection::load(&mut ct, &prefix, "ffn", device, props.hc_mult, props.hidden)?,
            });
        }

        let cache = crate::declare::hybrid_cache(&declare(&ct)?, props.max_seq_len, dtype, device)?;

        Ok(Self {
            tok_embeddings,
            layers,
            mla: props.mla_dims(dtype, None),
            norm,
            output: crate::load::quant(output)?,
            device: device.clone(),
            cache,
            max_seq_len: props.max_seq_len,
            hc_mult: props.hc_mult,
            hc_eps: props.hc_eps,
            sinkhorn_iters: props.sinkhorn_iters,
            rms_norm_eps: f64::from(props.rms_norm_eps),
            leading_dense: props.leading_dense,
        })
    }
}

struct ForwardCtx<'a> {
    indices: std::sync::Arc<[u32]>,
    run: &'a PagedLatentRun,
}

impl ModelWeights {
    pub fn forward(
        &self,
        x: &Tensor,
        _start_offsets: &[usize], // NoPE: no position is read
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len) = x.dims2()?;
        let run = PagedLatentRun::new(x, metadata)?;
        let mut cache_guard = self.cache.hybrid();
        let indices = crate::draft::state_indices(&cache_guard)?;
        self.provide_router_tables(x);
        let streams = self
            .tok_embeddings
            .forward(x)?
            .to_dtype(DType::F32)?
            .unsqueeze(2)?
            .broadcast_as((b_sz, seq_len, self.hc_mult, self.tok_embeddings.hidden))?
            .contiguous()?;

        let ctx = ForwardCtx { indices, run: &run };
        let streams = self.run_layers(streams, &mut cache_guard, &ctx)?;
        drop(cache_guard);

        // The final stream collapse is an unweighted mean.
        let hidden = streams.mean(2)?;
        let normed = self.norm.forward(&hidden)?;

        let xs = extract_logits(&normed, context_lens)?;
        MatMul.qmethod_matmul(&xs.contiguous()?, &*self.output)
    }

    fn run_layers(&self, mut streams: Tensor, cache: &mut HybridCache, ctx: &ForwardCtx<'_>) -> Result<Tensor> {
        for (i, layer) in self.layers.iter().enumerate() {
            // The late tap's residual on the mHC trunk: the stream mean, made only where the dispatcher reads it.
            if i >= self.leading_dense {
                crate::moe::offer_late_tap(i - self.leading_dense, &streams, &|s| s.mean(2));
            }

            let residual = streams;
            let (post, comb, collapsed) =
                layer.hc_attn.forward(&residual, self.rms_norm_eps, self.hc_eps, self.sinkhorn_iters)?;
            let h = layer.attn_norm.forward(&collapsed)?;
            let attn_out = layer.attn.forward(&self.mla, &h, i, cache, &ctx.indices, ctx.run)?.to_dtype(DType::F32)?;
            let streams_after_attn = stream_update(&residual, &post, &comb, &attn_out)?;

            let residual = streams_after_attn;
            let (post, comb, collapsed) =
                layer.hc_ffn.forward(&residual, self.rms_norm_eps, self.hc_eps, self.sinkhorn_iters)?;
            let hf = layer.ffn_norm.forward(&collapsed)?;
            // The hyper-connection scatters the output: no residual to fold.
            let ffn_out = layer.mlp.forward(&hf, None)?.to_dtype(DType::F32)?;
            streams = stream_update(&residual, &post, &comb, &ffn_out)?;
        }
        Ok(streams)
    }

    /// The router tables, as glm-dsa's: each MoE layer's `(ffn_norm, gate, exp_probs_b)`.
    fn provide_router_tables(&self, x: &Tensor) {
        let rows = self.layers.iter().skip(self.leading_dense).map(|l| (&l.ffn_norm, l.mlp.moe()));
        crate::moe::provide_router_rows(x, rows);
    }
}

impl crate::architecture::Architecture for ModelWeights {
    const ROUTER_TABLES: bool = true;
    const LATE_TAP: bool = true;
}
