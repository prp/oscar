//! Kimi K3 (`kimi-k3`) and K3.1 (`kimi-k31`: a sparse-attention indexer on the MLA layers, never loaded, the layers
//! run dense as GLM-5.2's), transcribed from the public references: its own parts only, so it comes and goes as a unit.
//! K3's model: KDA and gated NoPE MLA layers, KDA carrying position, and an [`AttnResState`] for the residual stream.

mod attn_res;

use std::sync::Arc;

use crate::embedding::QuantizedEmbedding;
use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::QuantMethod;

use crate::attention::{LatentMixer, MlaDims, PagedLatentRun};
use crate::declare::Meta;
use crate::ffn::GatedMlp;
use crate::latent::LatentHybridProps;
use crate::moe::{Ffn, RoutedLayer};
use crate::situ::SituBetas;
use attn_res::AttnResState;
use mistralrs_core::device_map::DeviceMapper;
use mistralrs_core::extract_logits;
use mistralrs_core::gguf::{ArchDeclaration, Content};
use mistralrs_core::kv_cache::EitherCache;
use mistralrs_core::layers::{MatMul, QRmsNorm};
use mistralrs_core::paged_attention::AttentionImplementation;
use mistralrs_core::text_models_inputs_processor::PagedAttentionInputMetadata;
use mistralrs_core::utils::model_config as ModelConfig;
use oscar_kernels::ExpertActivation;

/// Required, not defaulted: `situ_linear_beta` = 0 would drop the up branch's tanh clamp and load into garbage.
struct PropsGGUF {
    latent: LatentHybridProps,
    attn_res_block_size: usize,
    betas: SituBetas,
}

impl std::ops::Deref for PropsGGUF {
    type Target = LatentHybridProps;
    fn deref(&self) -> &LatentHybridProps {
        &self.latent
    }
}

const ARCHS: &[&str] = &["kimi-k3", "kimi-k31"];

pub(crate) fn declare<R: std::io::Seek + std::io::Read>(ct: &Content<'_, R>) -> Result<ArchDeclaration> {
    let m = Meta::of(ct.get_metadata(), ARCHS)?;
    PropsGGUF::read(&m)?.declare(&m)
}

impl PropsGGUF {
    fn read(m: &Meta<'_>) -> Result<Self> {
        // Required as K3's own and read nowhere: a default loads into garbage.
        for k in [
            "attention.q_lora_rank",
            "feed_forward_length",
            "expert_count",
            "expert_feed_forward_length",
            "expert_latent_length",
            "expert_shared_count",
        ] {
            m.usize(k)?;
        }

        Ok(Self {
            latent: LatentHybridProps::read(m)?,
            attn_res_block_size: m.usize("attn_res.block_size")?,
            betas: SituBetas { gate: m.f64("activation.situ_beta")?, up: m.f64("activation.situ_linear_beta")? },
        })
    }
}

struct LayerWeights {
    attn_norm: QRmsNorm,
    attn: LatentMixer,
    attn_res_score: Tensor,
    ffn_norm: QRmsNorm,
    ffn: Ffn,
    ffn_res_score: Tensor,
}

pub struct ModelWeights {
    tok_embeddings: QuantizedEmbedding,
    layers: Vec<LayerWeights>,
    /// Gated and NoPE: `rope.dimension_count` is the width of the key part carried unrotated, no rope switch.
    mla: MlaDims,
    norm: QRmsNorm,
    output: Arc<dyn QuantMethod>,
    output_res_score: Tensor,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    attn_res_block_size: usize,
    rms_norm_eps: f64,
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
        let hidden = self.tok_embeddings.hidden;

        // F32 throughout: the AttnRes sources' magnitudes span a wide range.
        let embed = self.tok_embeddings.forward(x)?.to_dtype(DType::F32)?.reshape((b_sz * seq_len, hidden))?;
        let mut state = AttnResState::new(embed, self.attn_res_block_size, self.rms_norm_eps)?;

        let mut cache_guard = self.cache.hybrid();
        let indices = crate::draft::state_indices(&cache_guard)?;
        let to_3d = |t: &Tensor| t.reshape((b_sz, seq_len, hidden));

        for (i, layer) in self.layers.iter().enumerate() {
            let x_attn = if state.has_completed_blocks() {
                state.retrieve(&layer.attn_res_score)?
            } else {
                state.passthrough()?
            };
            // The promotion sits here: after the retrieval, before the attention.
            state.maybe_close_block(i)?;

            let h = layer.attn_norm.forward(&to_3d(&x_attn)?)?;
            let attn_out = layer.attn.forward(&self.mla, &h, i, &mut cache_guard, &indices, &run)?;
            state.accumulate(&attn_out.reshape((b_sz * seq_len, hidden))?)?;

            let x_ffn = state.retrieve(&layer.ffn_res_score)?;
            let h = layer.ffn_norm.forward(&to_3d(&x_ffn)?)?;
            let ffn_out = layer.ffn.forward(&h, None)?;
            state.accumulate(&ffn_out.reshape((b_sz * seq_len, hidden))?)?;
        }
        drop(cache_guard);

        let final_hidden = state.retrieve(&self.output_res_score)?;
        self.decode_logits(&to_3d(&final_hidden)?, context_lens)
    }

    fn decode_logits(&self, hidden: &Tensor, context_lens: Vec<(usize, usize)>) -> Result<Tensor> {
        let x = extract_logits(hidden, context_lens)?;
        let x = self.norm.forward(&x)?;
        // `x` stays F32: the quantized matvec reads its input as F32, so a BF16 buffer is silently misread.
        let logits = MatMul.qmethod_matmul(&x.contiguous()?, &*self.output)?;
        Ok(logits)
    }
}

impl crate::architecture::Architecture for ModelWeights {
    const ROUTER_TABLES: bool = false;
    const LATE_TAP: bool = false;
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>, device: &Device, mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation, dtype: DType,
    ) -> Result<Self> {
        crate::load::admit(&attention_mechanism, &*mapper, device)?;
        let props = PropsGGUF::read(&Meta::of(ct.get_metadata(), ARCHS)?)?;

        // The dense lead's and the shared experts' activation, the routed ones'.
        let situ = ExpertActivation::SituGlu { beta_gate: props.betas.gate as f32, beta_up: props.betas.up as f32 };

        let tok_embeddings = QuantizedEmbedding::load(&mut ct, device, props.hidden, dtype)?;
        let norm = QRmsNorm::new(ct.tensor("output_norm.weight", device)?, props.rms_norm_eps)?;
        let output_res_score = crate::load::f32_dequant(&mut ct, "output_res_score.weight", device)?.flatten_all()?;
        let output = crate::load::output_head(&mut ct, device)?;

        let mut layers = Vec::with_capacity(props.n_layer);
        for layer_idx in crate::load::layers(props.n_layer) {
            let prefix = format!("blk.{layer_idx}");

            let qmm = |name: &str, ct: &mut Content<'_, R>| crate::load::qmm(ct, &prefix, name, device);
            let eps = props.rms_norm_eps;
            let attn_norm = crate::load::norm(&mut ct, &prefix, "attn_norm.weight", device, eps)?;
            let ffn_norm = crate::load::norm(&mut ct, &prefix, "ffn_norm.weight", device, eps)?;
            // Each retrieval's pseudo-query, `(hidden,)` F32.
            let mut score_of =
                |name: &str| crate::load::f32_dequant(&mut ct, &format!("{prefix}.{name}"), device)?.flatten_all();
            let attn_res_score = score_of("attn_res_score.weight")?;
            let ffn_res_score = score_of("ffn_res_score.weight")?;

            let attn = props.mixer(&mut ct, &prefix, layer_idx, device, true, &["ssm_g.weight"])?;

            let ffn = if layer_idx < props.leading_dense {
                Ffn::Dense(GatedMlp::load(&mut ct, &prefix, "", device, situ)?)
            } else {
                let router = props.router(&mut ct, &prefix, device, 1e-20)?;
                let ordinal = layer_idx - props.leading_dense;
                let mut moe = RoutedLayer::load(&mut ct, &prefix, ordinal, device, router, situ, None)?;
                let down = qmm("ffn_routed_down.weight", &mut ct)?;
                let latent_norm = crate::load::norm(&mut ct, &prefix, "ffn_routed_norm.weight", device, eps)?;
                moe.latent = Some((down, latent_norm, qmm("ffn_routed_up.weight", &mut ct)?));
                // Both shared experts are one ungated SiTU-GLU on the full width.
                moe.shared = Some(GatedMlp::load(&mut ct, &prefix, "_shexp", device, situ)?);
                Ffn::Moe(moe)
            };

            layers.push(LayerWeights { attn_norm, attn, attn_res_score, ffn_norm, ffn, ffn_res_score });
        }

        let cache = crate::declare::hybrid_cache(&declare(&ct)?, props.max_seq_len, dtype, device)?;

        Ok(Self {
            tok_embeddings,
            layers,
            mla: props.mla_dims(dtype, None),
            norm,
            output: crate::load::quant(output)?,
            output_res_score,
            device: device.clone(),
            cache,
            max_seq_len: props.max_seq_len,
            attn_res_block_size: props.attn_res_block_size,
            rms_norm_eps: f64::from(props.rms_norm_eps),
        })
    }
}
