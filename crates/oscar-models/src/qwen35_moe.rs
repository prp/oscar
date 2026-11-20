#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! `qwen35moe` (Qwen 3.5/3.6 MoE) and `qwen3next`: a Gated DeltaNet hybrid with gated full attention and an MoE.

use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::QuantMethod;

use crate::draft::Source;
use crate::embedding::QuantizedEmbedding;
use crate::ffn::Stream;
use crate::load::qmatvec_or_matmul;
use crate::moe::{load_moe_with_shared, RoutedLayer};
use crate::qwen_hybrid::{mrope_cos_sin, GdnProps, Mixer};
use crate::state_slots::RecurrentState;
use mistralrs_core::device_map::DeviceMapper;
use mistralrs_core::extract_logits;
use mistralrs_core::gguf::ArchDeclaration;
use mistralrs_core::gguf::Content;
use mistralrs_core::kv_cache::EitherCache;
use mistralrs_core::layers::{QRmsNorm, Qwen3VLRotaryEmbedding};
use mistralrs_core::paged_attention::AttentionImplementation;
use mistralrs_core::text_models_inputs_processor::PagedAttentionInputMetadata;
use mistralrs_core::utils::model_config as ModelConfig;

use crate::declare::Meta;

/// `(sum, rmsnorm(sum))`, on Metal F32 by the finalize's row kernel, which is candle's ops to the bit.
fn add_then_rmsnorm(a: &Tensor, b: &Tensor, post_norm: &QRmsNorm) -> Result<(Tensor, Tensor)> {
    if let &[bs, s, d] = a.dims() {
        if a.device().is_metal()
            && a.dtype() == DType::F32
            && b.dtype() == DType::F32
            && a.dims() == b.dims()
            && d >= 128
        {
            let w = post_norm.weight_for_dtype(DType::F32)?;
            let stream = Stream { residual: b, next_norm: Some((&w, post_norm.eps_f32())) };
            let f = crate::ffn::moe_finalize(None, a, None, Some(stream), (bs, s, d), DType::F32)?;
            return Ok((f.stream, f.normed.expect("the fold writes the normed input")));
        }
    }
    let sum = a.add(b)?;
    let normed = post_norm.forward(&sum)?;
    Ok((sum, normed))
}

const ARCHS: &[&str] = &["qwen35moe", "qwen3next"];

pub(crate) fn declare<R: std::io::Seek + std::io::Read>(ct: &Content<'_, R>) -> Result<ArchDeclaration> {
    let m = Meta::of(ct.get_metadata(), ARCHS)?;
    GdnProps::parse(&m)?.declare(&m, vec![])
}

struct LayerWeights {
    attn_norm: QRmsNorm,
    post_norm: QRmsNorm,
    attn: Mixer,
    moe: RoutedLayer,
}

pub struct ModelWeights {
    tok_embeddings: QuantizedEmbedding,
    layers: Vec<LayerWeights>,
    norm: QRmsNorm,
    output: Arc<dyn QuantMethod>,
    draft_output: Option<Arc<dyn QuantMethod>>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    rope: Arc<Qwen3VLRotaryEmbedding>,
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>, device: &Device, mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation, dtype: DType,
    ) -> Result<Self> {
        crate::load::admit(&attention_mechanism, &*mapper, device)?;
        let m = Meta::of(ct.get_metadata(), ARCHS)?;
        let (props, arch) = (GdnProps::parse(&m)?, m.arch);

        let tok_embeddings = QuantizedEmbedding::load(&mut ct, device, props.embedding_length, dtype)?;
        let final_norm = QRmsNorm::new_with_dtype(
            ct.tensor("output_norm.weight", &Device::Cpu)?,
            props.rms_norm_eps,
            dtype,
            device,
        )?;
        let output_qtensor = crate::load::output_head(&mut ct, device)?;
        // A Q4_K head for the drafter: it reads less a forward than the Q6_K one, at no loss of hint recall.
        let draft_output = if crate::draft::self_draft() {
            // The artifact's copy (written at the download), else requantised here.
            let q_weight = if ct.has_tensor("output_draft.weight") {
                ct.tensor("output_draft.weight", device)?
            } else {
                requantize_rows(&output_qtensor, GgmlDType::Q4K, device)?
            };
            Some(crate::load::quant(q_weight)?)
        } else {
            None
        };

        let rope = props.mrope(device)?;

        let mut layers = Vec::with_capacity(props.n_layer);
        for i in crate::load::layers(props.n_layer) {
            let prefix = format!("blk.{i}");

            let norm = |name: &str, ct: &mut Content<'_, R>| {
                let scale = ct.tensor(&format!("{prefix}.{name}.weight"), &Device::Cpu)?;
                QRmsNorm::new_with_dtype(scale, props.rms_norm_eps, dtype, device)
            };
            let attn_norm = norm("attn_norm", &mut ct)?;
            let post_norm = norm("post_attention_norm", &mut ct)?;
            let moe =
                load_moe_with_shared(&mut ct, &prefix, i, device, props.embedding_length, props.expert_used_count)?;
            // The qwen35moe writer reorders the v heads; the qwen3next writer does not.
            let attn = props.mixer(&mut ct, &prefix, i, device, &rope, dtype, arch == "qwen35moe", false)?;
            layers.push(LayerWeights { attn_norm, post_norm, attn, moe });
        }

        let cache = crate::declare::hybrid_cache(&declare(&ct)?, props.max_seq_len, dtype, device)?;

        Ok(Self {
            tok_embeddings,
            layers,
            norm: final_norm,
            output: crate::load::quant(output_qtensor)?,
            draft_output,
            device: device.clone(),
            cache,
            max_seq_len: props.max_seq_len,
            rope,
        })
    }
}

impl ModelWeights {
    pub fn forward(
        &self, x: &Tensor, start_offsets: &[usize], context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        // An F32 residual makes the per-layer casts no-ops, as the quantised matvec outputs F32.
        let layer_in = self.tok_embeddings.forward(x)?.to_dtype(DType::F32)?;
        let mut cache = self.cache.hybrid();
        let mut source = Source::engine(&mut cache, metadata)?;
        let cos_sin = mrope_cos_sin(&self.rope, start_offsets, x)?;
        self.provide_router_tables(x);
        let (layer_in, normed) = self.run_layers(layer_in, &mut source, &cos_sin)?;
        self.head(layer_in, normed, context_lens, &self.output)
    }

    fn head(
        &self, layer_in: Tensor, normed: Option<Tensor>, context_lens: Vec<(usize, usize)>, head: &Arc<dyn QuantMethod>,
    ) -> Result<Tensor> {
        // The final norm: the one the last finalize carried, or applied here.
        let xs = match normed {
            Some(n) => n,
            None => self.norm.forward(&layer_in)?,
        };
        let xs = extract_logits(&xs, context_lens)?;
        qmatvec_or_matmul(&xs.contiguous()?, &**head)
    }

    /// Each layer's `post_norm`, the norm before its router, and the gate.
    fn provide_router_tables(&self, x: &Tensor) {
        // No drafter reads these: the drafter's hints are its own routings.
        crate::moe::provide_router_rows(x, self.layers.iter().map(|l| (&l.post_norm, Some(&l.moe))));
    }

    /// The norm after layer `i`, the next layer's input norm or the final one, as the finalize folds it.
    fn next_norm(&self, i: usize) -> Result<(Tensor, f32)> {
        let norm = self.layers.get(i + 1).map_or(&self.norm, |l| &l.attn_norm);
        Ok((norm.weight_for_dtype(DType::F32)?, norm.eps_f32()))
    }

    /// Each finalize folds the next layer's input norm; returns the stream and, if folded, its final norm.
    fn run_layers(
        &self, mut layer_in: Tensor, source: &mut Source<'_>, cos_sin: &(Tensor, Tensor),
    ) -> Result<(Tensor, Option<Tensor>)> {
        let mut normed: Option<Tensor> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let (next_w, next_eps) = self.next_norm(i)?;
            let next_norm = Some((&next_w, next_eps));
            let residual = layer_in.clone();
            // Its own `attn_norm` unless the last finalize carried it.
            let normed_in = |normed: &mut Option<Tensor>| match normed.take() {
                Some(x) => Ok(x),
                None => layer.attn_norm.forward(&layer_in),
            };
            let mut state = None;
            let attn_out = match &layer.attn {
                Mixer::Full(attn) => {
                    let meta = source.attention(i)?;
                    attn.forward(&normed_in(&mut normed)?, cos_sin, meta)?
                }
                Mixer::Linear(attn) => {
                    let state = state.insert(source.recurrent(i, None)?);
                    attn.forward(&normed_in(&mut normed)?, state.cache())?
                }
            };
            let (after_attn, normed_ffn) = add_then_rmsnorm(&attn_out, &residual, &layer.post_norm)?;
            let out = layer.moe.forward(&normed_ffn, Some(Stream { residual: &after_attn, next_norm }))?;
            (layer_in, normed) = (out.stream, out.normed);
            match &mut state {
                Some(state) => state.commit(&[])?,
                // Wait out the attention layer's GPU work: a CPU running ahead would hold more layers' activations.
                None => {
                    if layer_in.dim(1)? >= 256 {
                        if let Device::Metal(md) = layer_in.device() {
                            let _ = md.wait_until_completed();
                        }
                    }
                }
            }
        }
        Ok((layer_in, normed))
    }

    /// The drafter's one-token decode beside the engine; the GDN kernels mutate its private state in place.
    pub fn forward_with_cache(
        &self, x: &Tensor, start_offsets: &[usize], context_lens: Vec<(usize, usize)>,
        view: &crate::draft::DraftPagedView, cache: &mut [Option<RecurrentState>],
    ) -> Result<Tensor> {
        let layer_in = self.tok_embeddings.forward(x)?.to_dtype(DType::F32)?;
        let cos_sin = mrope_cos_sin(&self.rope, start_offsets, x)?;
        // No mask, as one query attends to all the past.
        let meta = view.decode_metadata(start_offsets[0], &self.device)?;
        let mut source = Source::Draft { view, meta: &meta, private: cache };
        let (layer_in, normed) = self.run_layers(layer_in, &mut source, &cos_sin)?;
        self.head(layer_in, normed, context_lens, self.draft_output.as_ref().unwrap_or(&self.output))
    }
}

/// By row chunks on the CPU, as the whole head at F32 is a 4 GB transient: the rows read in place from the head's
/// buffer (a copy of it would hold the load's heap peak), the Q4_K bytes uploaded once.
fn requantize_rows(src: &QTensor, dtype: GgmlDType, device: &Device) -> Result<QTensor> {
    use candle_core::quantized::ggml_file::qtensor_from_ggml;
    let dims = src.shape().dims().to_vec();
    let [rows, cols] = dims[..] else {
        candle_core::bail!("requantize_rows: expected (rows, cols) matrix, got {dims:?}");
    };
    let src_dtype = src.dtype();
    let Some(storage) = src.metal_storage() else { candle_core::bail!("requantize_rows: head on Metal") };
    // SAFETY: a shared buffer the load filled from the host, read here before any GPU work is enqueued on it.
    let src_bytes =
        unsafe { std::slice::from_raw_parts(storage.buffer().contents() as *const u8, src.storage_size_in_bytes()) };
    let src_row = cols / src_dtype.block_size() * src_dtype.type_size();
    let chunk = 4096.min(rows);
    let mut out = Vec::with_capacity(rows * cols / dtype.block_size() * dtype.type_size());
    for r0 in (0..rows).step_by(chunk) {
        let n = chunk.min(rows - r0);
        let q =
            qtensor_from_ggml(src_dtype, &src_bytes[r0 * src_row..(r0 + n) * src_row], vec![n, cols], &Device::Cpu)?;
        let f = q.dequantize(&Device::Cpu)?;
        out.extend_from_slice(&QTensor::quantize(&f, dtype)?.data()?);
    }
    qtensor_from_ggml(dtype, &out, dims, device)
}

impl crate::architecture::Architecture for ModelWeights {
    const ROUTER_TABLES: bool = true;
    const LATE_TAP: bool = false;
    fn draftable(model: Arc<Self>) -> Option<Arc<dyn crate::draft::DraftableModel>> {
        Some(model)
    }
}

crate::draftable!(ModelWeights);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_bitwise, fill};

    /// The architecture's widths over thousands of rows.
    #[test]
    fn add_then_rmsnorm_matches_the_chain_to_the_bit() -> Result<()> {
        let dev = oscar_offload::pools::test_device()?;
        for d in [2048usize, 4096] {
            let a = Tensor::from_vec(fill(4096 * d, 1), (1, 4096, d), &dev)?;
            let r = Tensor::from_vec(fill(4096 * d, 2), (1, 4096, d), &dev)?;
            let w = Tensor::from_vec(fill(d, 3).iter().map(|v| 1.0 + v / 4.0).collect(), d, &Device::Cpu)?;
            let norm = QRmsNorm::new_with_dtype(QTensor::quantize(&w, GgmlDType::F32)?, 1e-6, DType::F32, &dev)?;
            let (sum, normed) = add_then_rmsnorm(&a, &r, &norm)?;
            let want_sum = a.add(&r)?;
            assert_bitwise(&sum, &want_sum, &format!("d={d} sum"))?;
            assert_bitwise(&normed, &norm.forward(&want_sum)?, &format!("d={d} normed"))?;
        }
        Ok(())
    }
}
