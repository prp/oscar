//! What every architecture's loader does alike, and the quantised matvec its projections run on.

use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};
use mistralrs_core::device_map::DeviceMapper;
use mistralrs_core::gguf::Content;
use mistralrs_core::layers::QRmsNorm;
use mistralrs_core::paged_attention::AttentionImplementation;
use mistralrs_core::utils::progress::{new_multi_progress, NiceProgressBar};
use mistralrs_quant::{GgufMatMul, MatMul, QuantMethod, QuantMethodConfig};
use oscar_kernels::{dense_matvec_fits, metal_dense_matvec, metal_router_gemv_f32, router_gemv_fits};

/// Paged KV the only path, one device the only map (`config::device_mapping` states the dummy one).
pub fn admit(attn: &AttentionImplementation, mapper: &dyn DeviceMapper, device: &Device) -> Result<()> {
    if !matches!(attn, AttentionImplementation::PagedAttention) {
        candle_core::bail!("paged KV only; pipeline has no paged attention");
    }
    if mapper.get_unique_devices().iter().any(|d| !d.same_device(device)) {
        candle_core::bail!("one device only; pipeline places layer elsewhere");
    }
    Ok(())
}

pub fn quant(t: QTensor) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf { q_weight: Arc::new(t), b: None })?))
}

/// `{prefix}.{name}` as a quantised projection on `device`. An F16/BF16 weight runs dense: read on the host and
/// kept as F32, whose bytes `QMatMul` uses in place, where a device copy would sit in working memory for good.
pub fn qmm<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>, prefix: &str, name: &str, device: &Device,
) -> Result<Arc<dyn QuantMethod>> {
    let name = format!("{prefix}.{name}");
    quant(match ct.tensor_info(&name)?.ggml_dtype {
        GgmlDType::F16 | GgmlDType::BF16 => {
            let host = ct.tensor(&name, &Device::Cpu)?.dequantize(&Device::Cpu)?;
            QTensor::quantize_onto(&host, GgmlDType::F32, device)?
        }
        _ => ct.tensor(&name, device)?,
    })
}

/// `{prefix}.{name}` as an RMS norm at `eps`.
pub fn norm<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>, prefix: &str, name: &str, device: &Device, eps: f32,
) -> Result<QRmsNorm> {
    QRmsNorm::new(ct.tensor(&format!("{prefix}.{name}"), device)?, eps)
}

pub fn f32_dequant<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>, name: &str, device: &Device,
) -> Result<Tensor> {
    ct.tensor(name, &Device::Cpu)?.dequantize(&Device::Cpu)?.to_dtype(DType::F32)?.to_device_kept(device)
}

/// `output.weight`, or the tied embedding where none ships.
pub fn output_head<R: std::io::Seek + std::io::Read>(ct: &mut Content<'_, R>, device: &Device) -> Result<QTensor> {
    if ct.has_tensor("output.weight") {
        ct.tensor("output.weight", device)
    } else {
        ct.tensor("token_embd.weight", device)
    }
}

/// The blocks, with the load's progress bar.
pub fn layers(block_count: usize) -> impl Iterator<Item = usize> {
    let multi = new_multi_progress();
    NiceProgressBar::<_, 'b'>(0..block_count, "Loading repeating layers", &multi).into_iter()
}

/// The quantised matvec for one F32 row and a Metal weight it takes, else `fallback`: one place picks a shape's kernel.
fn qmatvec_or(
    x: &Tensor, view: Option<mistralrs_quant::MetalQuantWeightView<'_>>, fallback: impl FnOnce() -> Result<Tensor>,
) -> Result<Tensor> {
    if x.device().is_metal() && x.dtype() == DType::F32 {
        if let Some(view) = view {
            if x.elem_count() == view.k && dense_matvec_fits(view.storage.dtype(), 1, view.n, view.k) {
                let row = x.reshape((1usize, view.k))?;
                let out = metal_dense_matvec(&[view.storage], view.n, view.k, &row)?;
                let mut dims = x.dims().to_vec();
                *dims.last_mut().expect("rank >= 1") = view.n;
                return out.reshape(dims);
            }
        }
    }
    fallback()
}

pub fn qmatvec_or_matmul(x: &Tensor, w: &dyn QuantMethod) -> Result<Tensor> {
    qmatvec_or(x, w.metal_quant_weight_view(), || MatMul.qmethod_matmul(x, w))
}

/// A router gate's logits by the quantised matvec where it takes the gate, else [`gate_gemv_or_forward`].
pub fn gate_matvec_or_forward(gate: &QMatMul, xs: &Tensor) -> Result<Tensor> {
    qmatvec_or(xs, qmatmul_view(gate), || gate_gemv_or_forward(gate, xs))
}

/// An F32 gate's one row by the router gemv, whose sum order is not the matmul's; a prefill's rows and a quantised gate
/// take the gate's own forward.
fn gate_gemv_or_forward(gate: &QMatMul, xs: &Tensor) -> Result<Tensor> {
    match gate {
        QMatMul::Tensor(w) if router_gemv_fits(w, Some(xs)) => metal_router_gemv_f32(w, xs),
        _ => gate.forward(xs),
    }
}

/// The view of a bare `QMatMul` (the router gate) on Metal.
fn qmatmul_view(w: &QMatMul) -> Option<mistralrs_quant::MetalQuantWeightView<'_>> {
    match w {
        QMatMul::QTensor(q) => {
            let storage = q.metal_storage()?;
            let [n, k] = q.shape().dims() else { return None };
            Some(mistralrs_quant::MetalQuantWeightView { storage, n: *n, k: *k })
        }
        _ => None,
    }
}
