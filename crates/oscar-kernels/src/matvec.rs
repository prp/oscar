use candle_core::quantized::{metal::QMetalStorage, GgmlDType};
use candle_core::{DType, Result, Tensor};

use crate::loader::{candle_pipeline, launch, out, pipeline, set_params, Family, Metal};

/// Each head's weights of a Q8_0 `(H, N, K)` stack against its own rows of x `(B·H, M, K)`, B outer: `(B·H, M, N)`.
pub fn metal_batched_qmatvec(
    weight_storage: &QMetalStorage, h: usize, n: usize, k: usize, x: &Tensor,
) -> Result<Tensor> {
    refuse_if!(
        weight_storage.dtype() != GgmlDType::Q8_0,
        "metal_batched_qmatvec: unsupported dtype {:?} (only Q8_0)",
        weight_storage.dtype()
    );
    // Candle's Q8_0 mv geometry (`call_quantized_matmul_mv_t`): 8 rows a threadgroup.
    const ALIGN: usize = 8;
    refuse_if!(k % 32 != 0, "metal_batched_qmatvec: K ({k}) must be divisible by 32");
    refuse_if!(n % ALIGN != 0, "metal_batched_qmatvec: N ({n}) must be divisible by {ALIGN}");
    let (bh, m, k_in) = match x.dims() {
        &[bh, m, k_] => (bh, m, k_),
        d => candle_core::bail!("metal_batched_qmatvec: input must be rank-3 (B,H, M, K), got {d:?}"),
    };
    refuse_if!(k_in != k, "metal_batched_qmatvec: input K={k_in} != weight K={k}");
    refuse_if!(bh % h != 0, "metal_batched_qmatvec: input batch {bh} not a multiple of H={h}");
    let b = bh / h;
    refuse_if!(x.dtype() != DType::F32, "metal_batched_qmatvec: input must be F32, got {:?}", x.dtype());

    let x_c = x.contiguous()?;
    let xm = Metal::of(&x_c, "metal_batched_qmatvec: input")?;
    let p = candle_pipeline(&xm.dev, "kernel_mul_mv_q8_0_f32")?;
    let (y, o) = out(&xm.dev, (bh, m, n), DType::F32, "batched-qmatvec-output")?;
    let w = weight_storage.buffer();
    launch(&xm.dev, &p, "batched-qmatvec", [n.div_ceil(ALIGN), m, bh], [8, 16, 1], |e| {
        // Weight batch `i12 % ne12` (`r2 = 1`); the input batch collapses as `i13 / r3 = 0` (`r3 = B`).
        set_params!(
            e,
            (
                w, &xm, &o, k as i64, // ne00
                n as i64, // ne01
                h as i64, // ne02
                0i64,     // nb00
                0i64,     // nb01
                0i64,     // nb02
                k as i64, // ne10
                m as i64, // ne11
                h as i64, // ne12
                0i64,     // nb10
                0i64,     // nb11
                0i64,     // nb12
                n as i64, // ne0
                m as i64, // ne1
                1u32,     // r2
                b as u32  // r3
            )
        );
        Ok(())
    })?;
    Ok(y)
}

/// (kernel, rows per 64-thread threadgroup, K block) for one weight's `w · x`, or two's `silu(gate · x) * (up · x)`.
fn dense_kernel(dtype: GgmlDType, weights: usize) -> Option<(&'static str, usize, usize)> {
    match (dtype, weights) {
        (GgmlDType::Q4K, 1) => Some(("mistralrs_qmatvec_q4k_r2s2_f32", 4, 256)),
        (GgmlDType::Q6K, 1) => Some(("mistralrs_qmatvec_q6k_r1s2_f32", 2, 256)),
        (GgmlDType::Q4K, 2) => Some(("mistralrs_qmatvec_gateup_silu_q4k_r2s2_f32", 4, 256)),
        (GgmlDType::Q8_0, 2) => Some(("mistralrs_qmatvec_gateup_silu_q8_0_r4s2_f32", 8, 32)),
        _ => None,
    }
}

/// Whether [`metal_dense_matvec`] takes `weights` `(n, k)` weights of `dtype`.
pub fn dense_matvec_fits(dtype: GgmlDType, weights: usize, n: usize, k: usize) -> bool {
    dense_kernel(dtype, weights).is_some_and(|(_, rows, kb)| n.is_multiple_of(rows) && k.is_multiple_of(kb))
}

/// Decode only, as a multi-row input takes the method's matmul: F32 `(num_tokens, k)` to `(num_tokens, n)`.
pub fn metal_dense_matvec(ws: &[&QMetalStorage], n: usize, k: usize, x: &Tensor) -> Result<Tensor> {
    let kernel = ws
        .first()
        .map(|w| w.dtype())
        .filter(|&d| ws.iter().all(|w| w.dtype() == d) && dense_matvec_fits(d, ws.len(), n, k))
        .and_then(|d| dense_kernel(d, ws.len()));
    let Some((kernel, rows, _)) = kernel else {
        let dtypes: Vec<GgmlDType> = ws.iter().map(|w| w.dtype()).collect();
        candle_core::bail!("dense matvec: no kernel for {dtypes:?} weights of ({n}, {k})")
    };
    let num_tokens = match x.dims() {
        &[t, k_in] if k_in == k => t,
        d => candle_core::bail!("dense matvec: input must be (num_tokens, {k}), got {d:?}"),
    };
    refuse_if!(x.dtype() != DType::F32, "dense matvec: input must be F32, got {:?}", x.dtype());
    let x_c = x.contiguous()?;
    let xm = Metal::of(&x_c, "dense matvec: input")?;
    let p = pipeline(xm.dev, Family::Moe, kernel)?;
    let (label, out_label) = match ws.len() {
        1 => ("qmatvec", "qmatvec-output"),
        _ => ("qmatvec-gateup-silu", "qmatvec-gateup-silu-output"),
    };
    let (y, o) = out(xm.dev, (num_tokens, n), DType::F32, out_label)?;
    // One weight is bound in both slots; the kernel reads the second only as a gate-up pair.
    let (g, u) = (ws[0], ws[ws.len() - 1]);
    launch(xm.dev, &p, label, [n.div_ceil(rows), 1, num_tokens], [64, 1, 1], |e| {
        set_params!(e, (g.buffer(), u.buffer(), &xm, &o, k as i64, n as i64, num_tokens as i64));
        Ok(())
    })?;
    Ok(y)
}

/// Whether [`metal_router_gemv_f32`] takes `w · x`: F32 on Metal, `w` `(n, k)` with `k % 4 == 0`, `x` one row of `k`,
/// each starting float4-aligned. `x` `None` asks of the gate alone, as its load does.
pub fn router_gemv_fits(w: &Tensor, x: Option<&Tensor>) -> bool {
    // The launch copies a strided input, and a copy starts at 0.
    let float4 = |t: &Tensor| {
        t.dtype() == DType::F32 && t.device().is_metal() && (!t.is_contiguous() || t.layout().start_offset() % 4 == 0)
    };
    let one_row = |x: &Tensor, k: usize| matches!(x.dims(), &[1, xk] | &[xk] if xk == k) && float4(x);
    matches!(w.dims(), &[_, k] if k % 4 == 0 && float4(w) && x.is_none_or(|x| one_row(x, k)))
}

/// Its `simd_sum` order is not the matmul's: it moves the engine's output bits, so only predictions take it.
pub fn metal_router_gemv_f32(w: &Tensor, x: &Tensor) -> Result<Tensor> {
    let (w, x) = (w.contiguous()?, x.contiguous()?);
    refuse_if!(!router_gemv_fits(&w, Some(&x)),
        "router_gemv: takes F32 on Metal, w (n, k) with k % 4 == 0 and x one row of k, each float4-aligned; got w {:?} \
         {:?} at {}, x {:?} {:?} at {}", w.dtype(), w.dims(), w.layout().start_offset(), x.dtype(), x.dims(),
        x.layout().start_offset());
    let (n, k) = w.dims2()?;
    let (wm, xm) = (Metal::of(&w, "router_gemv: weight")?, Metal::of(&x, "router_gemv: input")?);
    let p = pipeline(xm.dev, Family::Moe, "oscar_router_gemv_f32")?;
    let (y, o) = out(xm.dev, (1usize, n), DType::F32, "router-gemv-output")?;
    // A simdgroup per output row, four a threadgroup.
    launch(xm.dev, &p, "oscar-router-gemv", [n.div_ceil(4), 1, 1], [128, 1, 1], |e| {
        set_params!(e, (&wm, &xm, &o, n as i64, k as i64));
        Ok(())
    })?;
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::sine;

    /// Within a tolerance, as its simd_sum order is not the reference's; router shapes and a k tail off 128.
    #[test]
    fn router_gemv_matches_a_host_reference() -> Result<()> {
        let device = oscar_offload::pools::test_device()?;
        for (n, k) in [(128usize, 4096usize), (512, 4096), (10, 132), (7, 4)] {
            let (w, x) = (sine(n * k, 0.0013, 0.17), sine(k, 0.0029, 0.71));
            let y = metal_router_gemv_f32(
                &Tensor::from_vec(w.clone(), (n, k), &device)?,
                &Tensor::from_vec(x.clone(), (1, k), &device)?,
            )?;
            assert_eq!(y.dims(), &[1, n]);
            for (row, got) in y.flatten_all()?.to_vec1::<f32>()?.into_iter().enumerate() {
                let want: f64 = w[row * k..][..k].iter().zip(&x).map(|(&w, &x)| w as f64 * x as f64).sum();
                let tol = 1e-4 * want.abs().max(1.0);
                assert!((got as f64 - want).abs() <= tol, "n={n} k={k} row={row}: gemv {got} vs ref {want}");
            }
        }
        Ok(())
    }
}
