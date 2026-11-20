//! Kimi Delta Attention for K3 and GLM-5.3-Flash: a per-channel forget gate, which GDN's per-head kernels cannot serve.

use std::sync::Arc;

use candle_core::{DType, Result, Tensor, D};
use mistralrs_quant::QuantMethod;

use mistralrs_core::kv_cache::RecurrentLayerConfig;
use mistralrs_core::layers::MatMul;

use crate::latent::LatentHybridProps;
use crate::state_slots::{emit_conv_taps, Emit, RecurrentState};

/// Bounds a prefill's `(b, 3·h·d, S)` F32 transients; the carried states make slabs token-exact.
const PREFILL_SLAB: usize = 4096;

/// q|k|v packed on one channel axis for one depthwise pass, and a carried window of `conv_kernel - 1` taps.
pub(crate) fn recurrent_config(heads: usize, head_dim: usize, conv_kernel: usize) -> RecurrentLayerConfig {
    RecurrentLayerConfig {
        conv_dim: 3 * heads * head_dim,
        conv_width: conv_kernel - 1,
        state_dims: vec![heads, head_dim, head_dim],
    }
}

/// The q/k L2-norm epsilon (the reference kernel's), not the RMSNorm eps.
const QK_L2_EPS: f64 = 1e-6;

/// The depthwise causal conv over the packed q|k|v, then SiLU, carrying the state's window.
struct Conv {
    /// `(3 * value_dim, conv_kernel)`, for the fused conv.
    packed: Tensor,
    /// `(1, 3 * value_dim, 1)` contiguous per tap, tap 0 the oldest position.
    taps: Vec<Tensor>,
    kw: usize,
}

impl Conv {
    fn forward(&self, x: &Tensor, cache: &mut RecurrentState) -> Result<Tensor> {
        if x.dim(1)? > 1 && self.kw <= 4 && matches!(x.device(), candle_core::Device::Metal(_)) {
            return self.metal(x, cache);
        }
        self.tap_loop(x, cache)
    }

    /// A continuing chunk prepends the carried tail (newest last), narrowed off the output; a first chunk reads zeros.
    fn metal(&self, x: &Tensor, cache: &mut RecurrentState) -> Result<Tensor> {
        let s = x.dim(1)?;
        let kw = self.kw;
        let carried = if cache.seqlen_offset > 0 { kw - 1 } else { 0 };
        // (b, c, s) for the channel-major window; the transposed copy dies in the block, before the conv runs.
        let xt = {
            let xt = x.transpose(1, 2)?.contiguous()?;
            if carried > 0 {
                Tensor::cat(&[&cache.conv_state, &xt], 2)?
            } else {
                xt
            }
        };
        emit_conv_taps(&xt, carried, kw - 1, &cache.emits)?;
        let (y, state_kw) = oscar_kernels::gdn::causal_conv1d(&xt, &self.packed, kw)?;
        // The kernel returns the last `kw` positions oldest first (zero-padded); the cache keeps `kw - 1`.
        cache.conv_state = state_kw.narrow(2, 1, kw - 1)?.contiguous()?;
        y.narrow(2, carried, s)?.transpose(1, 2)?.contiguous()
    }

    /// The oracle, CPU fallback and decode: its `(b, c, s)` temporaries churn the buffer pool, so a prefill avoids it.
    fn tap_loop(&self, x: &Tensor, cache: &mut RecurrentState) -> Result<Tensor> {
        let (_b, s, c) = x.dims3()?;
        let kw = self.kw;
        let xt = x.transpose(1, 2)?.contiguous()?;
        // The carried tail prepended, so position t sees taps t-(kw-1)..t.
        let padded = Tensor::cat(&[&cache.conv_state, &xt], 2)?;
        let total = padded.dim(2)?;
        emit_conv_taps(&padded, kw - 1, kw - 1, &cache.emits)?;
        cache.conv_state = padded.narrow(2, total - (kw - 1), kw - 1)?.contiguous()?;

        debug_assert_eq!(self.taps[0].dim(1)?, c);
        let mut acc: Option<Tensor> = None;
        for tap in 0..kw {
            let slice = padded.narrow(2, tap, s)?;
            let term = slice.broadcast_mul(&self.taps[tap])?;
            acc = Some(match acc {
                Some(a) => (a + term)?,
                None => term,
            });
        }
        let y = acc.expect("conv kernel width is non-zero");
        candle_nn::ops::silu(&y.transpose(1, 2)?.contiguous()?)
    }
}

pub struct KdaWeights {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    conv: Conv,
    /// The decay logits' low-rank pair, `ssm_f_a` then `ssm_f_b`.
    f_a: Arc<dyn QuantMethod>,
    f_b: Arc<dyn QuantMethod>,
    /// F32 per channel, not per head, added to the decay logits.
    dt_bias: Tensor,
    /// `exp(A_log)` per head, i.e. `-ssm_a`, F32.
    a_exp: Tensor,
    beta_proj: Arc<dyn QuantMethod>,
    /// The output gate's projections applied in turn: K3's full-rank `ssm_g`, GLM-5.3-Flash's `ssm_g_a` then `ssm_g_b`.
    g_proj: Vec<Arc<dyn QuantMethod>>,
    o_norm: Tensor,
    o_proj: Arc<dyn QuantMethod>,

    /// Gains making the RMSNorm kernel an L2 norm (`l2_norm_heads`); q's also carries the attention scale.
    q_l2_gain: Tensor,
    k_l2_gain: Tensor,
    /// `QK_L2_EPS / head_dim`: the epsilon in the kernel's mean of squares.
    qk_l2_eps: f32,

    heads: usize,
    head_dim: usize,
    /// `kda.gate_lower_bound`, negative.
    gate_lower_bound: f64,
    rms_norm_eps: f64,
}

impl KdaWeights {
    /// `gates` are `ssm_g*`, not `attn_gate`: one HF parameter split by layer type, so the wrong one loads garbage.
    pub fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut mistralrs_core::gguf::Content<'_, R>, prefix: &str, device: &candle_core::Device,
        p: &LatentHybridProps, gates: &[&str],
    ) -> Result<Self> {
        let (heads, head_dim, conv_kernel) = (p.head_count, p.kda_head_dim, p.ssm_conv_kernel);
        if head_dim != 128 {
            candle_core::bail!("kda {prefix}: delta-rule kernel takes kda.head_dim 128, not {head_dim}");
        }
        let value_dim = heads * head_dim;
        let qmm =
            |name: &str, ct: &mut mistralrs_core::gguf::Content<'_, R>| crate::load::qmm(ct, &prefix, name, device);
        let f32_of = |name: &str, ct: &mut mistralrs_core::gguf::Content<'_, R>| {
            crate::load::f32_dequant(ct, &format!("{prefix}.{name}"), device)
        };

        let q_proj = qmm("attn_q.weight", ct)?;
        let k_proj = qmm("attn_k.weight", ct)?;
        let v_proj = qmm("attn_v.weight", ct)?;

        // candle reads GGUF's ne as `(conv_dim, 1, kernel)`, tap 0 the oldest.
        let host = candle_core::Device::Cpu;
        let mut taps = Vec::with_capacity(3);
        for branch in ["q", "k", "v"] {
            let w = ct.tensor(&format!("{prefix}.ssm_conv1d_{branch}.weight"), &host)?.dequantize(&host)?;
            taps.push(w.reshape((value_dim, conv_kernel))?);
        }
        let conv_weight = Tensor::cat(&taps, 0)?.contiguous()?;
        // Split per tap at load: a tap slice is strided, so a per-token slice would copy constant weight every token.
        let conv_taps = (0..conv_kernel)
            .map(|tap| {
                conv_weight.narrow(1, tap, 1)?.contiguous()?.reshape((1, 3 * value_dim, 1))?.to_device_kept(device)
            })
            .collect::<Result<Vec<_>>>()?;
        let conv_weight = conv_weight.to_device_kept(device)?;

        let dt_bias = f32_of("ssm_dt.bias", ct)?.flatten_all()?;
        if dt_bias.elem_count() != value_dim {
            candle_core::bail!(
                "kda {prefix}: ssm_dt.bias has {} entries, expected {value_dim} \
                 (it is per-CHANNEL, not per-head)",
                dt_bias.elem_count()
            );
        }

        // `ssm_a` stores -exp(A_log), its sign asserted: a raw A_log would decay into fluent nonsense, not a failure.
        let ssm_a = ct.tensor(&format!("{prefix}.ssm_a"), &host)?.dequantize(&host)?.flatten_all()?;
        if ssm_a.elem_count() != heads {
            candle_core::bail!("kda {prefix}: ssm_a has {} entries, expected {heads}", ssm_a.elem_count());
        }
        let a_max = ssm_a.max(0)?.to_vec0::<f32>()?;
        if a_max >= 0.0 {
            candle_core::bail!(
                "kda {prefix}: ssm_a must be strictly negative (stores -exp(A_log)); largest {a_max}; converter \
                 storing raw A_log lands here"
            );
        }
        let a_exp = ssm_a.neg()?.to_device_kept(device)?;

        let o_norm = f32_of("ssm_norm.weight", ct)?.flatten_all()?;
        if o_norm.elem_count() != head_dim {
            candle_core::bail!("kda {prefix}: ssm_norm has {} entries, expected {head_dim}", o_norm.elem_count());
        }

        // `l2_norm_heads`' gains: k's `d^-1/2`, q's `d^-1` with its attention scale.
        let inv_sqrt_d = (head_dim as f64).powf(-0.5);
        let k_l2_gain = Tensor::full(inv_sqrt_d as f32, head_dim, &host)?;
        let q_l2_gain = (&k_l2_gain * inv_sqrt_d)?.to_device_kept(device)?;
        let k_l2_gain = k_l2_gain.to_device_kept(device)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            conv: Conv { packed: conv_weight, taps: conv_taps, kw: conv_kernel },
            f_a: qmm("ssm_f_a.weight", ct)?,
            f_b: qmm("ssm_f_b.weight", ct)?,
            dt_bias,
            a_exp,
            beta_proj: qmm("ssm_beta.weight", ct)?,
            g_proj: gates.iter().map(|g| qmm(g, ct)).collect::<Result<_>>()?,
            o_norm,
            o_proj: qmm("attn_output.weight", ct)?,
            q_l2_gain,
            k_l2_gain,
            qk_l2_eps: (QK_L2_EPS / head_dim as f64) as f32,
            heads,
            head_dim,
            gate_lower_bound: p.kda_gate_lower_bound,
            rms_norm_eps: f64::from(p.rms_norm_eps),
        })
    }

    pub fn value_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    /// `x` is the pre-normed `(b, s, hidden)`, read by every projection; a prefill past [`PREFILL_SLAB`] runs in slabs.
    pub fn forward(&self, x: &Tensor, cache: &mut RecurrentState) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        if s > PREFILL_SLAB {
            // Each slab takes the emits inside it, at its own positions.
            let emits = std::mem::take(&mut cache.emits);
            let mut slabs = Vec::new();
            for s0 in (0..s).step_by(PREFILL_SLAB) {
                let len = PREFILL_SLAB.min(s - s0);
                let inside = emits.iter().filter(|e| e.at > s0 && e.at <= s0 + len);
                cache.emits = inside.map(|e| Emit { at: e.at - s0, ..e.clone() }).collect();
                slabs.push(self.forward(&x.narrow(1, s0, len)?.contiguous()?, cache)?);
            }
            cache.emits = emits;
            return Tensor::cat(&slabs, 1);
        }
        let vd = self.value_dim();
        let (h, d) = (self.heads, self.head_dim);

        let q = MatMul.qmethod_matmul(x, &*self.q_proj)?;
        let k = MatMul.qmethod_matmul(x, &*self.k_proj)?;
        let v = MatMul.qmethod_matmul(x, &*self.v_proj)?;
        let qkv = Tensor::cat(&[&q, &k, &v], D::Minus1)?.to_dtype(DType::F32)?;

        let qkv = self.conv.forward(&qkv, cache)?;
        let q = qkv.narrow(D::Minus1, 0, vd)?;
        let k = qkv.narrow(D::Minus1, vd, vd)?;
        let v = qkv.narrow(D::Minus1, 2 * vd, vd)?;

        // Per-head L2 norm on q and k, not v; q's 1/sqrt(d) is in its gain.
        let q = l2_norm_heads(&q.reshape((b, s, h, d))?, &self.q_l2_gain, self.qk_l2_eps)?;
        let k = l2_norm_heads(&k.reshape((b, s, h, d))?, &self.k_l2_gain, self.qk_l2_eps)?;
        let v = v.reshape((b, s, h, d))?;

        // `g` is per (head, channel), in log space.
        let z = MatMul.qmethod_matmul(x, &*self.f_a)?;
        let z = MatMul.qmethod_matmul(&z, &*self.f_b)?.to_dtype(DType::F32)?;
        let z = z.broadcast_add(&self.dt_bias)?.reshape((b, s, h, d))?;
        // K3's decay `g_min · sigmoid(exp(A_log) · z)`, bounded below, replaces Kimi Linear's softplus outright.
        let scaled = z.broadcast_mul(&self.a_exp.reshape((1, 1, h, 1))?)?;
        let g = (candle_nn::ops::sigmoid(&scaled)? * self.gate_lower_bound)?;

        // β has no factor of two: K3 has no negative-eigenvalue variant.
        let beta = MatMul.qmethod_matmul(x, &*self.beta_proj)?.to_dtype(DType::F32)?;
        let beta = candle_nn::ops::sigmoid(&beta.reshape((b, s, h))?)?;

        let o = recurrence(&q, &k, &v, &g, &beta, cache)?;
        cache.seqlen_offset += s;

        // The gate multiplies after the per-head norm, never entering the variance.
        let normed = candle_nn::ops::rms_norm(&o, &self.o_norm, self.rms_norm_eps as f32)?;
        let gate = self.g_proj.iter().try_fold(x.clone(), |g, p| MatMul.qmethod_matmul(&g, &**p))?;
        let y = (normed.reshape((b, s, vd))? * candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?)?)?;

        MatMul.qmethod_matmul(&y.to_dtype(x.dtype())?, &*self.o_proj)
    }
}

/// The delta rule over F32 `q`, `k`, `v`, `g` `(b, s, h, d)` and `beta` `(b, s, h)`, to `(b, s, h, d)`, in one scan a
/// layer; `g` stays in log space, the kernel exponentiating per channel.
fn recurrence(
    q: &Tensor, k: &Tensor, v: &Tensor, g: &Tensor, beta: &Tensor, cache: &mut RecurrentState,
) -> Result<Tensor> {
    let (b, s, h, d) = q.dims4()?;
    let bh = b * h;
    let to_bh = |t: &Tensor| -> Result<Tensor> { t.permute((0, 2, 1, 3))?.contiguous()?.reshape((bh, s, d)) };
    let (qs, ks, vs, gs) = (to_bh(q)?, to_bh(k)?, to_bh(v)?, to_bh(g)?);
    let betas = beta.permute((0, 2, 1))?.contiguous()?.reshape((bh, s))?;

    // The reshape views the cache's F32 buffer, which the kernel mutates in place: that is the update.
    let mut state = cache.recurrent_state.reshape((bh, d, d))?.to_dtype(DType::F32)?.contiguous()?;
    let emits: Vec<(usize, Tensor)> =
        cache.emits.iter().map(|e| Ok((e.at, e.recurrent.reshape((bh, d, d))?))).collect::<Result<_>>()?;
    let out = oscar_kernels::gdn::kda_delta_rule_recurrence(&qs, &ks, &vs, &gs, &betas, &mut state, &emits)?;
    cache.recurrent_state = state.reshape((b, h, d, d))?;

    out.reshape((b, h, s, d))?.permute((0, 2, 1, 3))?.contiguous()
}

/// `x / sqrt(Σ x² + eps)` per head as one RMSNorm, the gain carrying `d^(-1/2)`, the epsilon `eps/d`; `x` contiguous.
fn l2_norm_heads(x: &Tensor, gain: &Tensor, eps: f32) -> Result<Tensor> {
    candle_nn::ops::rms_norm(&x.to_dtype(DType::F32)?, gain, eps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_bitwise, assert_cuts_match_whole, fresh_state, host};
    use candle_core::Device;

    /// The kernel against the delta rule in f64, the written-back state too, within its FMA and fast-`exp` rounding;
    /// `b = 2`, as at `b = 1` every spelling of the `b·h` merge agrees, and from a non-zero state. With `g = 0`,
    /// `β = 1` and orthonormal keys the delta rule is an exact associative memory: `q = k_t` reads `v_t`.
    #[test]
    fn recurrence_matches_a_scalar_evaluation_and_recalls_what_it_wrote() -> Result<()> {
        let dev = oscar_offload::pools::test_device()?;
        let (b, h, d) = (2usize, 2usize, 128usize);
        for s in [7usize, 1] {
            let n = b * s * h * d;
            let wave = |seed: f32| -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.9).collect() };
            // Distinct per channel, so a wrong decay axis cannot cancel.
            let (q, k, v) = (wave(0.1), wave(1.3), wave(2.7));
            let g: Vec<f32> = wave(3.9).iter().map(|x| -x.abs() * 0.8).collect();
            let beta: Vec<f32> = (0..b * s * h).map(|i| 0.3 + 0.02 * i as f32).collect();
            let init: Vec<f32> = (0..b * h * d * d).map(|i| ((i as f32 * 0.11).cos()) * 0.05).collect();
            let t = |x: &[f32]| Tensor::from_vec(x.to_vec(), (b, s, h, d), &dev);
            let mut cache = fresh_state(&dev, (b, 3 * h * d, 3), Tensor::from_vec(init.clone(), (b, h, d, d), &dev)?)?;
            let beta_t = Tensor::from_vec(beta.clone(), (b, s, h), &dev)?;
            let got = recurrence(&t(&q)?, &t(&k)?, &t(&v)?, &t(&g)?, &beta_t, &mut cache)?;
            let (want, want_state) = delta_rule([&q[..], &k[..], &v[..], &g[..]], &beta, init, (b, s, h, d));
            for (what, got, want) in
                [("output", host(&got)?, want), ("state", host(&cache.recurrent_state)?, want_state)]
            {
                for (i, (&x, &y)) in got.iter().zip(&want).enumerate() {
                    let (y, tol) = (y as f32, 1e-3f32.max(1e-3 * x.abs().max(y.abs() as f32)));
                    assert!((x - y).abs() <= tol, "{what}[{i}]: kernel {x} vs {y}");
                }
            }
        }
        let (s, d) = (2usize, 128usize);
        // Two orthonormal keys, e0 then e1, read by the same queries.
        let keys: Vec<f32> = (0..s * d).map(|i| if i == 0 || i == d + 1 { 1.0 } else { 0.0 }).collect();
        let v: Vec<f32> = (1..=s * d).map(|x| x as f32).collect();
        let t = |x: &[f32]| Tensor::from_vec(x.to_vec(), (1, s, 1, d), &dev);
        let (g, beta) = (Tensor::zeros((1, s, 1, d), DType::F32, &dev)?, Tensor::ones((1, s, 1), DType::F32, &dev)?);
        let mut cache = fresh_state(&dev, (1, 3 * d, 3), Tensor::zeros((1, 1, d, d), DType::F32, &dev)?)?;
        let got = host(&recurrence(&t(&keys)?, &t(&keys)?, &t(&v)?, &g, &beta, &mut cache)?)?;
        for (i, (got, want)) in got.iter().zip(&v).enumerate() {
            assert!((got - want).abs() < 1e-5, "t{} recall {got} != {want}", i / d);
        }
        Ok(())
    }

    /// Per batch and head in f64: the decay on the key rows, the write `k ⊗ β(v − Sᵀk)`, then the read `Sᵀq`.
    fn delta_rule(
        [q, k, v, g]: [&[f32]; 4], beta: &[f32], state: Vec<f32>, (b, s, h, d): (usize, usize, usize, usize),
    ) -> (Vec<f64>, Vec<f64>) {
        let mut state: Vec<f64> = state.into_iter().map(f64::from).collect();
        let mut out = vec![0f64; b * s * h * d];
        for (bh, st) in state.chunks_mut(d * d).enumerate() {
            for t in 0..s {
                let bth = (bh / h * s + t) * h + bh % h;
                let (row, b_t) = (bth * d, f64::from(beta[bth]));
                let at = |x: &[f32], i: usize| f64::from(x[row + i]);
                for i in 0..d {
                    st[i * d..][..d].iter_mut().for_each(|e| *e *= at(g, i).exp());
                }
                let u: Vec<f64> =
                    (0..d).map(|j| (at(v, j) - (0..d).map(|i| st[i * d + j] * at(k, i)).sum::<f64>()) * b_t).collect();
                for i in 0..d {
                    st[i * d..][..d].iter_mut().zip(&u).for_each(|(e, u)| *e += at(k, i) * u);
                }
                for j in 0..d {
                    out[row + j] = (0..d).map(|i| st[i * d + j] * at(q, i)).sum();
                }
            }
        }
        (out, state)
    }

    /// On a first chunk the kernel's output within one `fma` chain's and fast SiLU's rounding of the tap loop, the
    /// carried state bitwise; then bitwise on both devices at cuts inside the carried window, with no one-token
    /// chunk, which would run as a decode.
    #[test]
    fn conv_metal_matches_the_tap_loop_and_a_prompt_in_chunks_convolves_as_one_forward() -> Result<()> {
        let metal = oscar_offload::pools::test_device()?;
        let (x, w) = conv_layer(&metal)?;
        let (mut kernel, mut tap_loop) = (conv_state(&metal)?, conv_state(&metal)?);
        let got = host(&w.metal(&x, &mut kernel)?)?;
        let want = host(&w.tap_loop(&x, &mut tap_loop)?)?;
        // One rounding per tap of an accumulator bounded at 1.4: 6.7e-7.
        let tol = 4.0 * f32::EPSILON * (4.0 * 0.7 * 0.5);
        assert_eq!(got.len(), want.len(), "conv output: length mismatch");
        for (i, (p, q)) in got.iter().zip(&want).enumerate() {
            let (token, channel) = (i / 384 % 3001, i % 384);
            assert!((p - q).abs() <= tol, "token {token} channel {channel}: kernel {p} against loop {q}, over {tol}");
        }
        assert_bitwise(&kernel.conv_state, &tap_loop.conv_state, "the carried conv state")?;
        for dev in [Device::Cpu, metal] {
            let (x, w) = conv_layer(&dev)?;
            let cuts = [&[1500][..], &[1000, 2998], &[2, 5, 1700]];
            assert_cuts_match_whole(&x, &cuts, || conv_state(&dev), |x, st| w.forward(x, st))?;
        }
        Ok(())
    }

    /// Both gains, as swapping q's and k's would leave every shape correct.
    #[test]
    fn l2_norm_via_rms_norm_matches_the_spelled_out_form() -> Result<()> {
        let dev = Device::Cpu;
        let (b, s, h, d) = (2usize, 3usize, 4usize, 128usize);
        let n = b * s * h * d;
        // Three magnitudes: the eps rescaling shows only near zero.
        let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017).sin() * 10f32.powi((i % 3) as i32 - 1)).collect();
        let x = Tensor::from_vec(v, (b, s, h, d), &dev)?;

        let denom = (x.sqr()?.sum_keepdim(D::Minus1)? + QK_L2_EPS)?.sqrt()?;
        let want_k = x.broadcast_div(&denom)?;
        let want_q = (&want_k * (d as f64).powf(-0.5))?;

        let inv_sqrt_d = (d as f64).powf(-0.5);
        let k_gain = Tensor::full(inv_sqrt_d as f32, d, &dev)?;
        let q_gain = (&k_gain * inv_sqrt_d)?;
        let eps = (QK_L2_EPS / d as f64) as f32;

        for (got, want, which) in
            [(l2_norm_heads(&x, &k_gain, eps)?, want_k, "k"), (l2_norm_heads(&x, &q_gain, eps)?, want_q, "q")]
        {
            for (a, e) in host(&got)?.iter().zip(host(&want)?.iter()) {
                assert!((a - e).abs() <= 1e-6 * e.abs().max(1e-3), "{which}: {a} != {e}");
            }
        }
        Ok(())
    }

    /// `(2, 3001, 384)` inputs (`s` divides by neither the kernel width nor POS_PER_THREAD = 8, `c = 384` is one and a
    /// half threadgroups) and a four-tap conv's weights, `h = 1` and `d = 128`.
    fn conv_layer(dev: &Device) -> Result<(Tensor, Conv)> {
        let (b, s, d, kw) = (2usize, 3001usize, 128usize, 4usize);
        let c = 3 * d;
        let x: Vec<f32> = (0..b * s * c).map(|i| ((i as f32 * 0.29 + 0.4).sin()) * 0.7).collect();
        let packed: Vec<f32> = (0..c * kw).map(|i| ((i as f32 * 0.13 + 1.1).cos()) * 0.5).collect();
        let packed = Tensor::from_vec(packed, (c, kw), dev)?;
        let taps =
            (0..kw).map(|t| packed.narrow(1, t, 1)?.contiguous()?.reshape((1, c, 1))).collect::<Result<Vec<_>>>()?;
        Ok((Tensor::from_vec(x, (b, s, c), dev)?, Conv { packed, taps, kw }))
    }

    /// A first chunk's state over [`conv_layer`]'s window, the recurrent state unread.
    fn conv_state(dev: &Device) -> Result<RecurrentState> {
        fresh_state(dev, (2, 384, 3), Tensor::zeros((2, 1, 128, 128), DType::F32, dev)?)
    }
}
