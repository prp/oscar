//! The feed-forward parts: the gated MLP, the routed experts' stacks and routers' math, the finalize into the stream.

use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QMatMul};
use candle_core::{DType, Device, Result, Tensor};

use mistralrs_core::gguf::Content;
use mistralrs_quant::QuantMethod;
use oscar_kernels::{all_experts, dense_matvec_fits, metal_dense_matvec, ExpertActivation, QkkMoeDType};

use crate::load::qmatvec_or_matmul;

/// One of a layer's three routed stacks as the dispatcher reads it; its bytes are the dispatcher's pool's.
pub struct RoutedExperts {
    pub all_experts: Tensor,
    pub dtype: QkkMoeDType,
    pub num_experts: usize,
    pub n: usize,
    pub k: usize,
    pub dispatch_ordinal: usize,
}

impl RoutedExperts {
    /// A K-quant or i-quant stack on Metal; any other is refused, as no kernel reads it.
    pub(crate) fn from_qtensor<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>, prefix: &str, suffix: &str, dispatch_ordinal: usize, device: &Device,
    ) -> Result<Self> {
        let key = format!("{prefix}.{suffix}");
        let info = ct.tensor_info(&key)?;
        let (dims, ggml) = (info.shape.dims().to_vec(), info.ggml_dtype);
        // The kernel dtype table is the one authority on what maps.
        let (true, &[num_experts, n, k], Some(dtype)) = (device.is_metal(), &dims[..], QkkMoeDType::from_ggml(ggml))
        else {
            candle_core::bail!(
                "{key}: routed experts must be stacked 3-D K-quant or i-quant tensor on Metal, not {ggml:?} {dims:?}"
            );
        };
        // Kept for the model's life, so built on the host.
        let all_experts = all_experts(num_experts, &Device::Cpu)?.to_device_kept(device)?;
        Ok(Self { all_experts, dtype, num_experts, n, k, dispatch_ordinal })
    }
}

/// A dense block's (`ffn_*`) or a shared expert's (`ffn_*_shexp`), its activation the routed experts' own.
pub struct GatedMlp {
    pub gate: Arc<dyn QuantMethod>,
    pub up: Arc<dyn QuantMethod>,
    pub down: Arc<dyn QuantMethod>,
    pub act: ExpertActivation,
}

impl GatedMlp {
    /// `suffix` is `""` for a dense block, `"_shexp"` for a shared expert.
    pub fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>, prefix: &str, suffix: &str, device: &Device, act: ExpertActivation,
    ) -> Result<Self> {
        let mut qmm = |p: &str| crate::load::qmm(ct, prefix, &format!("ffn_{p}{suffix}.weight"), device);
        Ok(Self { gate: qmm("gate")?, up: qmm("up")?, down: qmm("down")?, act })
    }

    /// The unfused form, for any activation.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (g, u) = (self.gate.forward(xs)?, self.up.forward(xs)?);
        self.down.forward(&apply_expert_activation(&g, &u, self.act)?)
    }

    pub fn swiglu(&self, xs_for_moe: &Tensor) -> Result<Tensor> {
        shared_swiglu(&*self.gate, &*self.up, &*self.down, xs_for_moe)
    }
}

/// The unfused path's activation: the definition `moe_fused.metal`'s activation policies must agree with.
pub fn apply_expert_activation(gate: &Tensor, up: &Tensor, activation: ExpertActivation) -> Result<Tensor> {
    match activation {
        ExpertActivation::Silu => mistralrs_core::ops::mul_and_act(gate, up, mistralrs_core::layers::Activation::Silu),
        ExpertActivation::SituGlu { beta_gate, beta_up } => {
            crate::situ::situ_glu(gate, up, crate::situ::SituBetas { gate: beta_gate as f64, up: beta_up as f64 })
        }
        ExpertActivation::SiluClamp { limit } => {
            let l = f64::from(limit);
            let gate = gate.minimum(l)?;
            let up = up.clamp(-l, l)?;
            mistralrs_core::ops::mul_and_act(&gate, &up, mistralrs_core::layers::Activation::Silu)
        }
    }
}

/// `(tokens, hidden)` F32: at one token the fused gate-up-silu matvec where a kernel takes the weights, else the chain.
pub fn shared_swiglu(
    gate_proj: &dyn QuantMethod, up_proj: &dyn QuantMethod, down_proj: &dyn QuantMethod, xs_for_moe: &Tensor,
) -> Result<Tensor> {
    let fused = |g: &mistralrs_quant::MetalQuantWeightView<'_>, u: &mistralrs_quant::MetalQuantWeightView<'_>| {
        let dtype = g.storage.dtype();
        dtype == u.storage.dtype() && (g.n, g.k) == (u.n, u.k) && dense_matvec_fits(dtype, 2, g.n, g.k)
    };
    let activated = match (gate_proj.metal_quant_weight_view(), up_proj.metal_quant_weight_view()) {
        (Some(g), Some(u)) if fused(&g, &u) && xs_for_moe.dtype() == DType::F32 && xs_for_moe.elem_count() == g.k => {
            metal_dense_matvec(&[g.storage, u.storage], g.n, g.k, xs_for_moe)?
        }
        _ => mistralrs_core::ops::mul_and_act(
            &gate_proj.forward(xs_for_moe)?,
            &up_proj.forward(xs_for_moe)?,
            mistralrs_core::layers::Activation::Silu,
        )?,
    };
    qmatvec_or_matmul(&activated, down_proj)
}

#[derive(Clone, Copy)]
pub struct ForwardShape {
    pub b_size: usize,
    pub seq_len: usize,
    pub hidden_dim: usize,
    /// Matches `xs.dtype()` at entry.
    pub out_dtype: DType,
    pub num_experts: usize,
}

impl ForwardShape {
    pub fn dims(self) -> (usize, usize, usize) {
        (self.b_size, self.seq_len, self.hidden_dim)
    }
}

/// The residual the output is added to and, if given, the next layer's norm `(F32 weight, epsilon)` to fold.
#[derive(Clone, Copy)]
pub struct Stream<'a> {
    pub residual: &'a Tensor,
    pub next_norm: Option<(&'a Tensor, f32)>,
}

impl<'a> Stream<'a> {
    pub fn residual(residual: &'a Tensor) -> Self {
        Self { residual, next_norm: None }
    }
}

/// The stream, and the next layer's normed input if a norm was folded.
pub struct Finalized {
    pub stream: Tensor,
    pub normed: Option<Tensor>,
}

impl Finalized {
    pub fn stream(stream: Tensor) -> Self {
        Self { stream, normed: None }
    }
}

/// The layer's stream in one F32 Metal row kernel, folding any next norm in `stream`: candle's ops to the bit.
pub fn moe_finalize(
    routed: Option<&Tensor>, shared: &Tensor, gate: Option<(&Tensor, &Tensor)>, stream: Option<Stream<'_>>,
    dims: (usize, usize, usize), out_dtype: DType,
) -> Result<Finalized> {
    let (b, s_len, d) = dims;
    let rows = b * s_len;
    if let Some(r) = routed {
        if r.elem_count() != rows * d {
            candle_core::bail!(
                "routed experts produced {} elements for {rows} tokens of width {d}; latent MoE projects routed \
                 output up first (`RoutedLayer::latent`)",
                r.elem_count()
            );
        }
    }
    let residual = stream.map(|s| s.residual);
    let next_norm = stream.and_then(|s| s.next_norm);
    let mut terms: Vec<(&'static str, &Tensor)> = Vec::with_capacity(6);
    let f32_metal = |t: &Tensor| t.dtype() == DType::F32 && t.device().is_metal();
    if !(d >= 128
        && f32_metal(shared)
        && routed.is_none_or(f32_metal)
        && residual.is_none_or(f32_metal)
        && gate.is_none_or(|(x, g)| f32_metal(x) && f32_metal(g))
        && next_norm.is_none_or(|(w, _)| f32_metal(w)))
    {
        candle_core::bail!("MoE finalize runs on F32 Metal tensors at width 128 or more (width {d})");
    }
    let mut body = String::new();
    if let Some((x, g)) = gate {
        terms.push(("x", x));
        terms.push(("g", g));
        body.push_str(
            "  float acc = 0.0f;\n  for (int i = tid; i < D; i += block_dim) acc = acc + x[i] * g[i];\n  \
             const float gate = 1.0f / (1.0f + exp(-row_sum(acc, shm, tid, block_dim)));\n",
        );
    }
    terms.push(("shared", shared));
    let mut expr = if gate.is_some() { "shared[i] * gate".to_string() } else { "shared[i]".to_string() };
    if let Some(r) = routed {
        terms.push(("routed", r));
        expr = format!("(routed[i] + {expr})");
    }
    if let Some(r) = residual {
        terms.push(("residual", r));
        expr = format!("({expr} + residual[i])");
    }
    let mut outputs = vec!["out"];
    match next_norm {
        None => body.push_str(&format!("  for (int i = tid; i < D; i += block_dim) out[i] = {expr};\n")),
        Some((w, eps)) => {
            // candle's rmsnorm in its kernel's order, with its fast-math `rsqrt`: the prelude's flags are one ULP off.
            terms.push(("nw", w));
            outputs.push("normed");
            body.push_str(&format!(
                "  float ss = 0.0f;\n  for (int i = tid; i < D; i += block_dim) {{ const float v = {expr}; out[i] = v; \
                 ss = fma(v, v, ss); }}\n  \
                 const float sum = row_sum_tree(ss, shm, tid, block_dim);\n  float inv;\n  {{\n#pragma clang fp reassociate(on)\n#pragma clang fp contract(fast)\n    inv = rsqrt(sum / float(D) + {eps:e}f);\n  }}\n  \
                 for (int i = tid; i < D; i += block_dim) {{ float val = out[i] * inv; val *= nw[i]; normed[i] = val; }}\n"
            ));
        }
    }
    let names: Vec<&'static str> = terms.iter().map(|(n, _)| *n).collect();
    let tensors: Vec<&Tensor> = terms.iter().map(|(_, t)| *t).collect();
    let kernel = oscar_kernels::metal_kernel::RowKernel {
        name: if next_norm.is_some() { "moe_finalize_norm" } else { "moe_finalize" },
        inputs: &names,
        outputs: &outputs,
        body: &body,
    };
    let mut outs = kernel.run(&tensors, rows, d)?;
    let normed = match next_norm {
        Some(_) => Some(outs.pop().expect("the normed output").reshape(dims)?.to_dtype(out_dtype)?),
        None => None,
    };
    let stream = outs.pop().expect("the stream").reshape(dims)?.to_dtype(out_dtype)?;
    Ok(Finalized { stream, normed })
}

/// A converted gate is read on the host, only the result placed: the file tensor read on the device would strand.
pub fn kept_router_gate<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>, prefix: &str, device: &Device, requant_to: Option<GgmlDType>,
) -> Result<QMatMul> {
    let name = format!("{prefix}.ffn_gate_inp.weight");
    // Kept as read, straight onto the device: the matvec takes a Q4_K or Q6_K gate, candle's matmul a Q5_K or Q8_0.
    let gate = if matches!(
        ct.tensor_info(&name)?.ggml_dtype,
        GgmlDType::Q4K | GgmlDType::Q6K | GgmlDType::Q5K | GgmlDType::Q8_0
    ) {
        QMatMul::from_qtensor(ct.tensor(&name, device)?)?
    } else {
        let host = Device::Cpu;
        let dense = ct.tensor(&name, &host)?.dequantize(&host)?;
        match requant_to {
            // Q4_K needs n%4==0, k%256==0, which hold at every router shape here.
            Some(t) => QMatMul::from_qtensor(candle_core::quantized::QTensor::quantize_onto(&dense, t, device)?)?,
            None => QMatMul::Tensor(dense.to_device_kept(device)?),
        }
    };
    // An F32 gate's one row rides the router gemv; candle's `CANDLE_DEQUANTIZE_ALL` makes any gate one.
    if let QMatMul::Tensor(w) = &gate {
        let fits = !device.is_metal() || oscar_kernels::router_gemv_fits(w, None);
        let why = "the router gemv, its predictions' one kernel, reads rows of a multiple of 4 floats";
        crate::architecture::ensure(fits, &format!("{name}: an F32 router gate of {:?}; {why}", w.dims()))?;
    }
    Ok(gate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_bitwise, fill};
    use candle_core::D;

    /// `[routed, shared, x, g, residual]`, 4096 rows of width `d`: a per-row rounding event shows once in many rows.
    fn terms(d: usize, dev: &Device) -> Result<[Tensor; 5]> {
        let rows = 4096;
        Ok([
            Tensor::from_vec(fill(rows * d, 1), (1, rows, d), dev)?,
            Tensor::from_vec(fill(rows * d, 2), (rows, d), dev)?,
            Tensor::from_vec(fill(rows * d, 3), (rows, d), dev)?,
            Tensor::from_vec(fill(d, 4), (1, d), dev)?,
            Tensor::from_vec(fill(rows * d, 5), (1, rows, d), dev)?,
        ])
    }

    /// Every combination of the optional terms, at the 35B's width and one not a power of two; then the fold of the
    /// next norm against candle's `rms_norm` of the stream, at the Qwen family's widths.
    #[test]
    fn finalize_matches_the_tensor_chain_and_folds_the_next_norm_to_the_bit() -> Result<()> {
        let dev = oscar_offload::pools::test_device()?;
        let (b, s) = (1usize, 4096usize);
        for d in [2048usize, 3584] {
            let [routed, shared, x, g, residual] = terms(d, &dev)?;
            let terms = (0..8).map(|m| (m & 4 != 0, m & 2 != 0, m & 1 != 0)).filter(|(r, g, _)| *r || *g);
            for (with_routed, with_gate, with_residual) in terms {
                let mut want = shared.reshape((b, s, d))?;
                if with_gate {
                    let gate = candle_nn::ops::sigmoid(&x.broadcast_mul(&g)?.sum_keepdim(D::Minus1)?)?;
                    want = want.broadcast_mul(&gate.reshape((b, s, 1))?)?;
                }
                if with_routed {
                    want = (routed.clone() + want)?;
                }
                if with_residual {
                    want = (want + residual.clone())?;
                }
                let got = moe_finalize(
                    with_routed.then_some(&routed),
                    &shared,
                    with_gate.then_some((&x, &g)),
                    with_residual.then(|| Stream::residual(&residual)),
                    (b, s, d),
                    DType::F32,
                )?;
                let what = format!("d={d} routed={with_routed} gate={with_gate} residual={with_residual}");
                assert_bitwise(&got.stream, &want, &what)?;
            }
        }
        for d in [2048usize, 2560, 3072, 4096] {
            let [routed, shared, x, g, residual] = terms(d, &dev)?;
            let w = Tensor::from_vec(fill(d, 6).iter().map(|v| 1.0 + v / 4.0).collect(), d, &dev)?;
            let finalize = |next_norm: Option<(&Tensor, f32)>| {
                let stream = Stream { residual: &residual, next_norm };
                moe_finalize(Some(&routed), &shared, Some((&x, &g)), Some(stream), (1, 4096, d), DType::F32)
            };
            let (plain, folded) = (finalize(None)?, finalize(Some((&w, 1e-6)))?);
            assert_bitwise(&folded.stream, &plain.stream, &format!("d={d} stream"))?;
            let normed = folded.normed.expect("the fold writes the normed input");
            assert_bitwise(&normed, &candle_nn::ops::rms_norm(&plain.stream, &w, 1e-6)?, &format!("d={d} normed"))?;
        }
        Ok(())
    }
}
