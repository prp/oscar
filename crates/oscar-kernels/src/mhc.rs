//! The mHC mix shaping (`glm5next` hyper-connections) in one dispatch, where tensor ops take dozens per site.

use candle_core::{DType, Result, Tensor};

use crate::loader::{launch, out, pipeline, set_params, Family, Metal};

/// `pre | post | comb`, `comb` doubly stochastic; `scales` in that order on the device, as a host one syncs each site.
pub fn metal_mhc_mix(
    mix: &Tensor, base: &Tensor, scales: &Tensor, hc_mult: usize, sinkhorn_iters: usize, eps: f32,
) -> Result<Tensor> {
    refuse_if!(
        mix.dtype() != DType::F32 || base.dtype() != DType::F32 || scales.dtype() != DType::F32,
        "metal_mhc_mix: F32 only"
    );
    refuse_if!(scales.elem_count() != 3, "metal_mhc_mix: scales must hold 3 entries");
    refuse_if!(
        hc_mult == 0 || hc_mult > 5,
        "metal_mhc_mix: hc_mult ({hc_mult}) must satisfy 0 < h <= 5, a row's h^2 lanes one simdgroup"
    );
    let width = (2 + hc_mult) * hc_mult;
    let (rows, w) = mix.dims2()?;
    refuse_if!(w != width, "metal_mhc_mix: mix width {w} != (2+h)*h = {width}");
    refuse_if!(base.elem_count() != width, "metal_mhc_mix: base has {} entries, want {width}", base.elem_count());
    refuse_if!(rows == 0, "metal_mhc_mix: rows must be > 0");

    let mix_c = mix.contiguous()?;
    let base_c = base.contiguous()?;
    let mixm = Metal::of(&mix_c, "metal_mhc_mix: mix")?;
    let basem = Metal::of(&base_c, "metal_mhc_mix: base")?;
    let p = pipeline(&mixm.dev, Family::Moe, "mistralrs_mhc_mix_f32")?;
    let scales_c = scales.contiguous()?;
    let scalesm = Metal::of(&scales_c, "metal_mhc_mix: scales")?;
    let (y, o) = out(&mixm.dev, (rows, width), DType::F32, "mhc-mix-out")?;
    launch(&mixm.dev, &p, "mhc-mix", [rows, 1, 1], [hc_mult * hc_mult, 1, 1], |e| {
        set_params!(e, (&mixm, &o, &basem, &scalesm, hc_mult as i64, sinkhorn_iters as i64, eps));
        Ok(())
    })?;
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{fill, host, staged};

    /// The vendor's schedule on the host in f64: the 2h gates, a row softmax, a column pass, (iters - 1) row/col pairs.
    fn reference(row: &[f32], base: &[f32], scales: &[f32], h: usize, iters: usize, eps: f64) -> Vec<f64> {
        let v = |e: usize, s: usize| row[e] as f64 * scales[s] as f64 + base[e] as f64;
        let mut out = vec![0.0; (2 + h) * h];
        for i in 0..h {
            out[i] = 1.0 / (1.0 + (-v(i, 0)).exp()) + eps;
            out[h + i] = 2.0 / (1.0 + (-v(h + i, 1)).exp());
        }
        let mut comb: Vec<f64> = (0..h * h).map(|e| v(2 * h + e, 2)).collect();
        for r in comb.chunks_mut(h) {
            let mx = r.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let denom: f64 = r.iter().map(|&c| (c - mx).exp()).sum();
            r.iter_mut().for_each(|c| *c = (*c - mx).exp() / denom + eps);
        }
        let norm = |comb: &mut [f64], by_row: bool| {
            for a in 0..h {
                let at = |b: usize| if by_row { a * h + b } else { b * h + a };
                let s: f64 = (0..h).map(|b| comb[at(b)]).sum::<f64>() + eps;
                (0..h).for_each(|b| comb[at(b)] /= s);
            }
        };
        norm(&mut comb, false);
        for _ in 1..iters {
            norm(&mut comb, true);
            norm(&mut comb, false);
        }
        out[2 * h..].copy_from_slice(&comb);
        out
    }

    /// Thousands of rows at every h a simdgroup holds, within a tolerance of the f64 reference (the kernel's `exp` and
    /// sum order are its own).
    #[test]
    fn mhc_mix_matches_a_host_reference() -> Result<()> {
        let device = oscar_offload::pools::test_device()?;
        let (rows, iters, eps) = (4096usize, 20usize, 1e-6f32);
        let scales = vec![0.9f32, 1.1, 0.7];
        for h in 1..=5usize {
            let width = (2 + h) * h;
            let (mix, base) = (fill(rows * width, 1), fill(width, 2));
            let got = host(&metal_mhc_mix(
                &staged(mix.clone(), (rows, width), &device)?,
                &staged(base.clone(), width, &device)?,
                &staged(scales.clone(), 3, &device)?,
                h,
                iters,
                eps,
            )?)?;
            for r in 0..rows {
                let want = reference(&mix[r * width..][..width], &base, &scales, h, iters, eps as f64);
                for (col, (&g, &w)) in got[r * width..][..width].iter().zip(&want).enumerate() {
                    let tol = 1e-4 * w.abs().max(1.0);
                    assert!((g as f64 - w).abs() <= tol, "h={h} row={r} col={col}: kernel {g} vs ref {w}");
                }
            }
        }
        Ok(())
    }
}
