//! SiTU-GLU, a soft-capped SwiGLU: `β·tanh(x/β)` caps the Swish gate's linear factor and the up branch.

use candle_core::{Result, Tensor};

#[derive(Debug, Clone, Copy)]
pub struct SituBetas {
    /// `situ_beta`, capping the Swish gate's linear factor.
    pub gate: f64,
    /// `situ_linear_beta`, capping the up branch.
    pub up: f64,
}

fn softcap(x: &Tensor, beta: f64) -> Result<Tensor> {
    ((x / beta)?.tanh()? * beta)?.to_dtype(x.dtype())
}

/// Over same-shaped projections: the reference form, and the oracle any fused kernel must match.
pub fn situ_glu(gate: &Tensor, up: &Tensor, betas: SituBetas) -> Result<Tensor> {
    let gated = (softcap(gate, betas.gate)? * candle_nn::ops::sigmoid(gate)?)?;
    gated * softcap(up, betas.up)?
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    const K3: SituBetas = SituBetas { gate: 4.0, up: 25.0 };

    fn scalar(g: f64, u: f64, betas: SituBetas) -> f64 {
        let t = |v: f64| Tensor::from_vec(vec![v as f32], 1, &Device::Cpu).unwrap();
        situ_glu(&t(g), &t(u), betas).unwrap().to_vec1::<f32>().unwrap()[0] as f64
    }

    fn swiglu(g: f64, u: f64) -> f64 {
        g * (1.0 / (1.0 + (-g).exp())) * u
    }

    /// Against the equation evaluated independently in f64 at K3's betas, which swapped would fail it, over one
    /// `(2, 3)` tensor whose shape and dtype the op keeps; bounded by β₁β₂ where SwiGLU is not; SwiGLU near the
    /// origin, the relative error shrinking to under 1e-3 by s = 0.05, and at large betas.
    #[test]
    fn matches_the_closed_form_caps_swiglu_and_agrees_with_it_where_the_caps_do_not_bind() {
        let points = [(0.0, 0.0), (1.0, 2.0), (-3.0, 5.0), (0.5, -0.25), (37.0, 91.0), (-60.0, -80.0)];
        let t = |f: fn(&(f64, f64)) -> f64| {
            Tensor::from_vec(points.iter().map(|p| f(p) as f32).collect::<Vec<_>>(), (2, 3), &Device::Cpu).unwrap()
        };
        let out = situ_glu(&t(|p| p.0), &t(|p| p.1), K3).unwrap();
        assert_eq!((out.dims(), out.dtype()), (&[2usize, 3][..], DType::F32));
        for (&(g, u), got) in points.iter().zip(out.flatten_all().unwrap().to_vec1::<f32>().unwrap()) {
            let want = (K3.gate * (g / K3.gate).tanh() * (1.0 / (1.0 + (-g).exp()))) * (K3.up * (u / K3.up).tanh());
            let got = got as f64;
            assert!((got - want).abs() < 1e-4 * want.abs().max(1.0), "g={g} u={u}: got {got}, want {want}");
        }
        let extreme = scalar(600.0, 800.0, K3);
        assert!(extreme.abs() <= K3.gate * K3.up + 1e-3, "||SiTU-GLU||inf <= beta1beta2 = 100, got {extreme}");
        assert!(swiglu(600.0, 800.0) > 4.0e5, "guard: SwiGLU really is unbounded here");
        let mut prev_rel = f64::INFINITY;
        for s in [0.4f64, 0.2, 0.1, 0.05] {
            let rel = ((scalar(s, s, K3) - swiglu(s, s)) / swiglu(s, s)).abs();
            assert!(rel < prev_rel, "relative error must shrink: {s} -> {rel}");
            prev_rel = rel;
        }
        assert!(prev_rel < 1e-3, "and be small by s=0.05: {prev_rel}");
        let big = SituBetas { gate: 1.0e6, up: 1.0e6 };
        assert!((scalar(2.0, 3.0, big) - swiglu(2.0, 3.0)).abs() < 1e-3);
    }
}
