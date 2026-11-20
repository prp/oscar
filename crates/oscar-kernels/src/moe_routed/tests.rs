//! Each routed kernel near a dequantised reference, and bitwise over scattered rows: address resolution moves no bytes.

use candle_core::quantized::QTensor;
use half::f16;

use super::*;
use crate::test_support::{assert_bitwise, host, sine, staged};

/// The K3 GGUF's `situ_beta` and `situ_linear_beta`.
const SITU: ExpertActivation = ExpertActivation::SituGlu { beta_gate: 4.0, beta_up: 25.0 };
/// K3's `UD-IQ1_M` formats: gate and up, then down.
const K3_GATE_UP: [QkkMoeDType; 2] = [QkkMoeDType::Iq1S, QkkMoeDType::Iq2XXS];
const K3_DOWN: [QkkMoeDType; 2] = [QkkMoeDType::Iq1S, QkkMoeDType::Iq3XXS];
/// Loose enough for the simd_sum order and Metal's fast `tanh` and `exp`, far tighter than any error a control rejects.
const TOL: f32 = 2e-4;
/// The scale of every synthetic K3 block, as half bits.
const PINNED_SCALE_BITS: u16 = 0x2266;
const PINNED_SCALE: f32 = 0.012496948;

#[test]
fn the_format_tables_answer_every_production_pairing() {
    use ExpertActivation::Silu;
    use QkkMoeDType as Q;
    for (ggml, qkk) in [
        (GgmlDType::Q3K, Q::Q3K),
        (GgmlDType::Q4K, Q::Q4K),
        (GgmlDType::Q6K, Q::Q6K),
        (GgmlDType::Iq1S, Q::Iq1S),
        (GgmlDType::Iq2XXS, Q::Iq2XXS),
        (GgmlDType::Iq3XXS, Q::Iq3XXS),
    ] {
        assert_eq!(QkkMoeDType::from_ggml(ggml), Some(qkk));
    }
    assert_eq!(QkkMoeDType::from_ggml(GgmlDType::Q8_0), None);
    // The k-quants' SiLU and K3's SiTU-GLU: the activation is half the key, and two dtypes the banks map no body reads.
    let pairs = [Q::Q3K, Q::Q4K, Q::Q6K, Q::Iq1S, Q::Iq2XXS, Q::Iq3XXS].map(|d| [(d, Silu), (d, SITU)]).concat();
    let fused: Vec<_> = pairs.into_iter().filter(|&(dtype, act)| has_fused_gateup_kernel(dtype, act)).collect();
    assert_eq!(fused, [(Q::Q3K, Silu), (Q::Q4K, Silu), (Q::Iq1S, SITU), (Q::Iq2XXS, SITU)]);
}

/// Through f16 as the matmul stages x and weights, at real weights' magnitude (block scales ~5e-5, a Q4_K super-block's
/// halved scale a half subnormal); two waves fill one buffer, each (token, slot) once, so it is the stacked layer's.
#[test]
fn gemm_matches_the_reference_and_moves_no_bytes() -> Result<()> {
    let device = oscar_offload::pools::test_device()?;
    let (experts, n, k, tokens, topk) = (16usize, 192usize, 512usize, 128usize, 8usize);
    for ggml in [GgmlDType::Q4K, GgmlDType::Q6K, GgmlDType::Q3K] {
        let qkk = QkkMoeDType::from_ggml(ggml).expect("a routed format");
        let stack = Stack::quantised(ggml, (experts, n, k), (0.0017, 0.31, 0.05), &device)?;
        let x_cpu = Tensor::from_vec(sine(tokens * k, 0.0029, 1.7), (tokens, k), &Device::Cpu)?;
        let x = x_cpu.to_device(&device)?;
        let (ids_v, ids) = routing(tokens, topk, distinct(experts), &device)?;
        let all = all_experts(experts, &device)?;
        let gemm = |row: &Row, list: &Tensor, into: Option<&Tensor>| {
            moe_gemm(row.get(), qkk, list, experts, n, k, &x, &ids, into)
        };
        let stacked = gemm(&stack.row()?, &all, None)?;
        let out = host(&stacked)?;
        let x_v = host(&x_cpu.to_dtype(DType::F16)?.to_dtype(DType::F32)?)?;
        let dq: Vec<f32> = stack.dq.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
        let (want, abs_terms) = matmul_reference(&dots(&dq, &x_v, topk, &ids_v, n, k));
        assert_within_bound(&out, &want, &abs_terms, k, 2f64.powi(-10), &format!("gemm {qkk:?}"));

        let scattered = stack.scattered(40, &device)?;
        let full = scattered.row(|_| true)?;
        assert_bitwise(&host(&gemm(&full, &all, None)?)?, &out, &format!("gemm {qkk:?} scattered row"));
        let mut union: Option<Tensor> = None;
        for wave in [0..experts / 2, experts / 2..experts] {
            let list = Tensor::from_vec(wave.clone().map(|e| e as u32).collect::<Vec<_>>(), wave.len(), &device)?;
            let part = gemm(&scattered.row(|e| wave.contains(&e))?, &list, union.as_ref())?;
            union.get_or_insert(part);
        }
        let union = union.expect("two waves");
        assert_bitwise(&host(&union)?, &out, &format!("gemm {qkk:?} wave union"));
        let w = staged(sine(tokens * topk, 0.037, 0.9), (tokens, topk), &device)?;
        let summed = |y: &Tensor| host(&moe_topk_weighted_sum(y, &w, full.get(), &ids)?);
        assert_bitwise(&summed(&union)?, &summed(&stacked)?, &format!("gemm {qkk:?} one weighted sum over the waves"));
    }
    Ok(())
}

/// Two waves' rows: each (token, slot) written by exactly one wave.
#[test]
fn matvec_matches_the_reference_and_moves_no_bytes() -> Result<()> {
    let device = oscar_offload::pools::test_device()?;
    let (experts, n, k, tokens, topk) = (10usize, 1536usize, 512usize, 7usize, 4usize);
    for ggml in [GgmlDType::Q4K, GgmlDType::Q6K, GgmlDType::Q3K] {
        let qkk = QkkMoeDType::from_ggml(ggml).expect("a routed format");
        let stack = Stack::quantised(ggml, (experts, n, k), (0.0017, 0.31, 1.0), &device)?;
        let x_v = sine(tokens * k, 0.0029, 1.7);
        let x = staged(x_v.clone(), (tokens, k), &device)?;
        let (ids_v, ids) = routing(tokens, topk, distinct(experts), &device)?;
        let matvec = |row: &Row| host(&moe_matvec(row.get(), qkk, experts, n, k, &x, &ids, None)?);
        let out = matvec(&stack.row()?)?;
        let (want, abs_terms) = matmul_reference(&dots(&stack.dq, &x_v, topk, &ids_v, n, k));
        assert_within_bound(&out, &want, &abs_terms, k, 0.0, &format!("matvec {qkk:?}"));

        let scattered = stack.scattered(43, &device)?;
        assert_bitwise(&matvec(&scattered.row(|_| true)?)?, &out, &format!("matvec {qkk:?} scattered row"));
        let mut union = vec![0f32; out.len()];
        for wave in [0..experts / 2, experts / 2..experts] {
            let part = matvec(&scattered.row(|e| wave.contains(&e))?)?;
            for (i, _) in ids_v.iter().enumerate().filter(|(_, e)| wave.contains(&(**e as usize))) {
                union[i * n..][..n].copy_from_slice(&part[i * n..][..n]);
            }
        }
        assert_bitwise(&union, &out, &format!("matvec {qkk:?} wave union"));
    }
    Ok(())
}

/// The oracle is ggml's `dequantize_iq*`: it catches pairing, element order and geometry, not a shared misreading.
#[test]
fn iq_matvec_and_gemm_match_the_oracle_and_move_no_bytes() -> Result<()> {
    let device = oscar_offload::pools::test_device()?;
    let formats = [QkkMoeDType::Iq1S, QkkMoeDType::Iq2XXS, QkkMoeDType::Iq3XXS];
    for (qkk, gemm) in formats.into_iter().flat_map(|q| [(q, false), (q, true)]) {
        let (experts, n, k, tokens, topk, capacity) =
            if gemm { (16usize, 128usize, 256usize, 64usize, 8usize, 40usize) } else { (3, 128, 512, 4, 2, 11) };
        let stack = Stack::iq(qkk, scrambled_blocks(qkk, experts * n * (k / 256), 0x51C3_A17E), experts, &device)?;
        // The matmul stages x and weights through f16.
        let staging = |v: &f32| if gemm { f16::from_f32(*v).to_f32() } else { *v };
        let w: Vec<f32> = stack.dq.iter().map(staging).collect();
        let x_host = sine(tokens * k, 0.0029, 1.7);
        let x_v: Vec<f32> = x_host.iter().map(staging).collect();
        let x = staged(x_host, (tokens, k), &device)?;
        let expert = |t: usize, s: usize| if gemm { (t + s) % experts } else { (t * topk + s) % experts };
        let (ids_v, ids) = routing(tokens, topk, expert, &device)?;
        let all = all_experts(experts, &device)?;
        let routed = |row: &Row| -> Result<Vec<f32>> {
            host(&if gemm {
                moe_gemm(row.get(), qkk, &all, experts, n, k, &x, &ids, None)?
            } else {
                moe_matvec(row.get(), qkk, experts, n, k, &x, &ids, None)?
            })
        };
        let out = routed(&stack.row()?)?;
        let (want, _) = matmul_reference(&dots(&w, &x_v, topk, &ids_v, n, k));
        let worst = out.iter().zip(&want).map(|(g, w)| (g - w).abs() / w.abs().max(1.0)).fold(0f32, f32::max);
        assert!(worst < 1e-3, "{qkk:?} gemm={gemm} against the oracle: worst relative error {worst}");
        let scattered = routed(&stack.scattered(capacity, &device)?.row(|_| true)?)?;
        assert_bitwise(&scattered, &out, &format!("{qkk:?} gemm={gemm} scattered row"));
    }
    Ok(())
}

/// `n` in the thousands and `k` at a live shape, the last the 397B's, so a one-in-many rounding event cannot hide.
#[test]
fn down_matches_the_reference_and_moves_no_bytes() -> Result<()> {
    let device = oscar_offload::pools::test_device()?;
    let (experts, tokens, topk) = (12usize, 17usize, 8usize);
    for (ggml, n, k) in [
        (GgmlDType::Q4K, 2048usize, 512usize),
        (GgmlDType::Q6K, 2048, 512),
        (GgmlDType::Q5_0, 2560, 640),
        (GgmlDType::Q3K, 4096, 1024),
    ] {
        let qkk = QkkMoeDType::from_ggml(ggml).expect("a routed format");
        let stack = Stack::quantised(ggml, (experts, n, k), (0.0013, 0.47, 1.0), &device)?;
        let act_v = sine(tokens * topk * k, 0.0021, 2.3);
        let act = staged(act_v.clone(), (tokens, topk, k), &device)?;
        let w_v = sine(tokens * topk, 0.037, 0.9);
        let w = staged(w_v.clone(), (tokens, topk), &device)?;
        let (ids_v, ids) = routing(tokens, topk, distinct(experts), &device)?;
        let down = |row: &Row| host(&moe_down(row.get(), qkk, experts, n, k, &ids, &w, &act)?);
        let out = down(&stack.row()?)?;
        let (want, abs_terms) = down_reference(&stack.dq, &act_v, &w_v, &ids_v, topk, n, k);
        assert_within_bound(&out, &want, &abs_terms, topk * k, 0.0, &format!("down {qkk:?}"));
        let scattered = down(&stack.scattered(178, &device)?.row(|_| true)?)?;
        assert_bitwise(&scattered, &out, &format!("down {qkk:?} scattered row"));
        if ggml == GgmlDType::Q4K {
            // A null expert's slots weigh nothing, as if weighted zero.
            let null = Row::over(&stack.on_device, (0..experts).map(|e| (e != 2).then_some(e)), stack.bpe)?;
            let zeroed = w_v.iter().zip(&ids_v).map(|(&w, &e)| if e == 2 { 0.0 } else { w }).collect();
            let zeroed = staged(zeroed, (tokens, topk), &device)?;
            let want = host(&moe_down(stack.row()?.get(), qkk, experts, n, k, &ids, &zeroed, &act)?)?;
            assert_bitwise(&down(&null)?, &want, "down with a null expert");
        }
    }
    Ok(())
}

/// Under the pinned half: a wrong `d` would scale reference and kernel alike and hide itself; a NaN or infinite
/// one, a non-finite weight. K3's down projections over their own input, so a mismatch is the down kernel's, not
/// inherited from gate+up.
#[test]
fn down_iq_matches_the_oracle_under_the_pinned_half() -> Result<()> {
    let device = oscar_offload::pools::test_device()?;
    assert_eq!(f16::from_bits(PINNED_SCALE_BITS).to_f32(), PINNED_SCALE, "the scale is not that half");
    for qkk in [QkkMoeDType::Iq1S, QkkMoeDType::Iq2XXS, QkkMoeDType::Iq3XXS] {
        let dq = dequantize(qkk, &k3_blocks(qkk, 1, 7), &device)?;
        assert!(dq.iter().all(|v| v.is_finite()), "{qkk:?} dequantised to non-finite");
        assert!(dq.iter().any(|v| *v != 0.0), "{qkk:?} dequantised to all zeros");
    }
    let (experts, tokens, topk) = (5usize, 7usize, 4usize);
    for (n, k) in [(256usize, 512usize), (768, 2048)] {
        for qkk in K3_DOWN {
            let stack = Stack::iq(qkk, k3_blocks(qkk, experts * n * (k / 256), 0xd0c7), experts, &device)?;
            let act_v = sine(tokens * topk * k, 0.0023, 0.41);
            let act = staged(act_v.clone(), (tokens, topk, k), &device)?;
            let (ids_v, ids) = routing(tokens, topk, |t, s| ((t * topk + s) * 3 + 1) % experts, &device)?;
            let w_v = sine(tokens * topk, 0.11, 0.9);
            let w = staged(w_v.clone(), (tokens, topk), &device)?;
            let out = host(&moe_down(stack.row()?.get(), qkk, experts, n, k, &ids, &w, &act)?)?;
            let worst = max_rel(&out, &down_reference(&stack.dq, &act_v, &w_v, &ids_v, topk, n, k).0);
            assert!(worst < TOL, "{qkk:?} down at (n={n}, k={k}): max relative error {worst:e}");
        }
    }
    Ok(())
}

/// The clamp engages on both factors where it is the activation; no miss on a full row; a wave's row, null outside
/// its experts, zeroes an unresolved slot and counts it once.
#[test]
fn gateup_matches_the_reference_moves_no_bytes_and_zeroes_an_unresolved_slot() -> Result<()> {
    let device = oscar_offload::pools::test_device()?;
    let (experts, tokens, topk) = (12usize, 9usize, 8usize);
    for (ggml, activation, n, k) in [
        (GgmlDType::Q4K, ExpertActivation::Silu, 768usize, 2048usize),
        (GgmlDType::Q3K, ExpertActivation::Silu, 1024, 4096),
        (GgmlDType::Q3K, ExpertActivation::SiluClamp { limit: 1.5 }, 512, 1024),
    ] {
        let qkk = QkkMoeDType::from_ggml(ggml).expect("a routed format");
        let gate = Stack::quantised(ggml, (experts, n, k), (0.0017, 0.31, 1.0), &device)?;
        let up = Stack::quantised(ggml, (experts, n, k), (0.0019, 0.71, 1.0), &device)?;
        let x_v = sine(tokens * k, 0.0029, 1.7);
        let x = staged(x_v.clone(), (tokens, k), &device)?;
        let (ids_v, ids) = routing(tokens, topk, distinct(experts), &device)?;
        let miss = Tensor::zeros(1, DType::U32, &device)?;
        let gateup =
            |g: &Row, u: &Row| host(&moe_gateup(g.get(), u.get(), qkk, activation, experts, n, k, &ids, &x, &miss)?);
        let out = gateup(&gate.row()?, &up.row()?)?;
        assert_eq!(miss.to_vec1::<u32>()?[0], 0, "a full row must produce zero misses");
        let (want, abs_terms, clamped) = gateup_reference(&gate.dq, &up.dq, &x_v, &ids_v, topk, n, k, activation);
        if matches!(activation, ExpertActivation::SiluClamp { .. }) {
            assert!(clamped[0] > 0 && clamped[1] > 0, "the clamp never engaged - the test is vacuous");
        }
        assert_within_bound(&out, &want, &abs_terms, 2 * k, 0.0, &format!("gateup {qkk:?} {activation:?}"));
        if ggml == GgmlDType::Q4K {
            // Experts 3 and 5 null on the gate row only: the pair is unresolved.
            let null = Row::over(&gate.on_device, (0..experts).map(|e| (!matches!(e, 3 | 5)).then_some(e)), gate.bpe)?;
            let nulled = gateup(&null, &up.row()?)?;
            for (i, &e) in ids_v.iter().enumerate() {
                let slot = i * n..(i + 1) * n;
                if matches!(e, 3 | 5) {
                    assert!(nulled[slot].iter().all(|v| *v == 0.0), "an unresolved slot must read zero");
                } else {
                    assert_bitwise(&nulled[slot.clone()], &out[slot], "a resolved slot");
                }
            }
            let expected = ids_v.iter().filter(|e| matches!(e, 3 | 5)).count() as u32;
            assert!(expected > 0, "the routing never named a null expert - vacuous");
            assert_eq!(miss.to_vec1::<u32>()?[0], expected, "one miss per unresolved (token, slot)");
        }
        let (gate, up) = (gate.scattered(61, &device)?, up.scattered(61, &device)?);
        let scattered = gateup(&gate.row(|_| true)?, &up.row(|_| true)?)?;
        assert_bitwise(&scattered, &out, &format!("gateup {qkk:?} scattered rows"));
    }
    Ok(())
}

/// The small shape leaves half the simdgroup no loop trip, the large two or more; a policy not reaching the kernel, or
/// bytes decoded one block along, moves the output far past the gate; the swapped betas need gate and up stacks that
/// differ, as at `gate == up` the swap is a no-op.
#[test]
fn situ_gateup_iq_matches_the_oracle_and_its_controls_do_not() -> Result<()> {
    let device = oscar_offload::pools::test_device()?;
    let swiglu = ExpertActivation::SituGlu { beta_gate: 1.0e6, beta_up: 1.0e6 };
    let swapped = ExpertActivation::SituGlu { beta_gate: 25.0, beta_up: 4.0 };
    for qkk in K3_GATE_UP {
        let small = |activation, shift| situ_gateup(&device, qkk, activation, (256, 512), shift);
        let (worst, reference) = small(SITU, false)?;
        let (large, _) = situ_gateup(&device, qkk, SITU, (768, 2048), false)?;
        assert!(worst < TOL && large < TOL, "{qkk:?}: max relative errors {worst:e} and {large:e} against {TOL:e}");
        for (control, moved) in [
            ("SwiGLU (beta->inf)", max_rel(&small(swiglu, false)?.1, &reference)),
            ("swapped betas", max_rel(&small(swapped, false)?.1, &reference)),
            ("a one-block shift", small(SITU, true)?.0),
        ] {
            assert!(moved > 100.0 * TOL, "{qkk:?}: {control} moved the output by only {moved:e}: the gate is vacuous");
        }
    }
    Ok(())
}

/// Unresolved slots hold NaN, as the matmul never writes them; the rest is the same `fma` chain, bitwise.
#[test]
fn weighted_sum_skips_unresolved_slots() -> Result<()> {
    let device = oscar_offload::pools::test_device()?;
    let (experts, tokens, topk, hidden) = (10usize, 64usize, 8usize, 1024usize);
    let (ids_v, ids) = routing(tokens, topk, distinct(experts), &device)?;
    let resolved = |slot: usize| ids_v[slot] % 3 != 0;
    let pool = Tensor::zeros(experts * 64, DType::U8, &device)?;
    let row = Row::over(&pool, (0..experts).map(|e| (e % 3 != 0).then_some(e)), 64)?;
    let mut y_v = sine(tokens * topk * hidden, 0.0013, 0.7);
    for (_, y) in y_v.chunks_mut(hidden).enumerate().filter(|(slot, _)| !resolved(*slot)) {
        y.fill(f32::NAN);
    }
    let w_v: Vec<f32> = sine(tokens * topk, 0.027, 1.3).into_iter().map(|v| v + 0.5).collect();
    let y = staged(y_v.clone(), (tokens, topk, hidden), &device)?;
    let w = staged(w_v.clone(), (tokens, topk), &device)?;
    let out = host(&moe_topk_weighted_sum(&y, &w, row.get(), &ids)?)?;
    let want: Vec<f32> = (0..tokens * hidden)
        .map(|i| {
            let (t, h) = (i / hidden, i % hidden);
            let slots = (t * topk..(t + 1) * topk).filter(|&slot| resolved(slot));
            slots.fold(0f32, |acc, slot| y_v[slot * hidden + h].mul_add(w_v[slot], acc))
        })
        .collect();
    assert_bitwise(&out, &want, "weighted sum");
    Ok(())
}

/// An expert stack on the device, its bytes and their dequantisation on the host, `bpe` bytes an expert.
struct Stack {
    on_device: Tensor,
    bytes: Vec<u8>,
    dq: Vec<f32>,
    bpe: usize,
}

impl Stack {
    /// `(experts, n, k)` of [`sine`] at `(step, phase)` times `magnitude`, quantised by candle.
    fn quantised(
        ggml: GgmlDType, (experts, n, k): (usize, usize, usize), (step, phase, magnitude): (f32, f32, f32),
        device: &Device,
    ) -> Result<Self> {
        let floats: Vec<f32> = sine(experts * n * k, step, phase).into_iter().map(|v| v * magnitude).collect();
        let q = QTensor::quantize(&Tensor::from_vec(floats, (experts, n, k), &Device::Cpu)?, ggml)?;
        Self::new(q.data()?.into_owned(), q.dequantize(&Device::Cpu)?.flatten_all()?.to_vec1()?, experts, device)
    }

    fn iq(qkk: QkkMoeDType, bytes: Vec<u8>, experts: usize, device: &Device) -> Result<Self> {
        let dq = dequantize(qkk, &bytes, device)?;
        Self::new(bytes, dq, experts, device)
    }

    fn new(bytes: Vec<u8>, dq: Vec<f32>, experts: usize, device: &Device) -> Result<Self> {
        let on_device = Tensor::from_vec(bytes.clone(), bytes.len(), &Device::Cpu)?.to_device(device)?;
        Ok(Self { on_device, bpe: bytes.len() / experts, bytes, dq })
    }

    /// Expert `e` at `e·bpe` of the stack.
    fn row(&self) -> Result<Row> {
        Row::new(stack_addresses(&self.on_device, self.bytes.len() / self.bpe, self.bpe)?, &self.on_device)
    }

    /// The experts at scattered slots of `capacity` among bytes that decode to nonzero weights.
    fn scattered(&self, capacity: usize, device: &Device) -> Result<Scattered> {
        let (experts, bpe) = (self.bytes.len() / self.bpe, self.bpe);
        assert!(capacity >= experts);
        let mut pool: Vec<u8> = (0..capacity * bpe).map(|i| ((i * 131 + 7) % 251) as u8).collect();
        let slots: Vec<usize> = (0..experts).map(|e| (37 * e + 11) % capacity).collect();
        for (e, &s) in slots.iter().enumerate() {
            pool[s * bpe..][..bpe].copy_from_slice(&self.bytes[e * bpe..][..bpe]);
        }
        Ok(Scattered { pool: Tensor::from_vec(pool, capacity * bpe, &Device::Cpu)?.to_device(device)?, slots, bpe })
    }
}

struct Scattered {
    pool: Tensor,
    slots: Vec<usize>,
    bpe: usize,
}

impl Scattered {
    /// The row resolving the experts `resolved` admits, null elsewhere.
    fn row(&self, resolved: impl Fn(usize) -> bool) -> Result<Row> {
        Row::over(&self.pool, self.slots.iter().enumerate().map(|(e, &s)| resolved(e).then_some(s)), self.bpe)
    }
}

/// An address row and the one buffer it points into, declared as a bank's row declares the bank's segments.
struct Row {
    row: Tensor,
    segment: [Retained<ProtocolObject<dyn MTLBuffer>>; 1],
}

impl Row {
    fn new(row: Tensor, backing: &Tensor) -> Result<Self> {
        Ok(Self { segment: [(&Metal::of(backing, "backing")?.buf).into()], row })
    }

    /// Over `pool`: entry `i` at slot `slots[i]`, null where `None`.
    fn over(pool: &Tensor, slots: impl IntoIterator<Item = Option<usize>>, bpe: usize) -> Result<Self> {
        let m = Metal::of(pool, "pool")?;
        let base = m.buf.gpu_address() + m.off as u64;
        let row: Vec<i64> = slots.into_iter().map(|s| s.map_or(0, |s| (base + (s * bpe) as u64) as i64)).collect();
        Self::new(Tensor::new(row.as_slice(), pool.device())?, pool)
    }

    fn get(&self) -> ExpertRow<'_> {
        ExpertRow { row: &self.row, segments: &self.segment }
    }
}

/// `(tokens, topk)` ids, `expert(t, s)` at token `t`'s slot `s`, and their host copy.
fn routing(
    tokens: usize, topk: usize, expert: impl Fn(usize, usize) -> usize, device: &Device,
) -> Result<(Vec<u32>, Tensor)> {
    let ids: Vec<u32> = (0..tokens * topk).map(|i| expert(i / topk, i % topk) as u32).collect();
    Ok((ids.clone(), Tensor::from_vec(ids, (tokens, topk), device)?))
}

/// Every token's slots name distinct experts, as top-k routing does.
fn distinct(experts: usize) -> impl Fn(usize, usize) -> usize {
    move |t, s| (t * 7 + s + 1) % experts
}

/// The f64 dot and the sum of its terms' magnitudes, the accumulation bound's scale.
fn dot(w: &[f32], x: &[f32]) -> (f64, f64) {
    w.iter().zip(x).fold((0.0, 0.0), |(acc, mag), (&w, &x)| {
        let term = w as f64 * x as f64;
        (acc + term, mag + term.abs())
    })
}

/// Per routed slot `i` and row `r` of its expert's `(n, k)` in `w`: [`dot`] with `x`'s row `i / per`, `per` the slots
/// sharing an input row.
fn dots(w: &[f32], x: &[f32], per: usize, ids: &[u32], n: usize, k: usize) -> Vec<(f64, f64)> {
    let row = |i: usize, r: usize| dot(&w[(ids[i] as usize * n + r) * k..][..k], &x[i / per * k..][..k]);
    (0..ids.len() * n).map(|j| row(j / n, j % n)).collect()
}

/// [`dots`] as a matmul's outputs and its bound's scales.
fn matmul_reference(dots: &[(f64, f64)]) -> (Vec<f32>, Vec<f32>) {
    dots.iter().map(|&(d, mag)| (d as f32, mag as f32)).unzip()
}

/// Per token and row, `Σ_s w[t,s]·(down[e]·act[t,s])` in f64, and its terms' magnitudes.
fn down_reference(
    dq: &[f32], act: &[f32], w: &[f32], ids: &[u32], topk: usize, n: usize, k: usize,
) -> (Vec<f32>, Vec<f32>) {
    let (mut want, mut abs_terms) = (vec![0f64; ids.len() / topk * n], vec![0f32; ids.len() / topk * n]);
    for (j, (d, mag)) in dots(dq, act, 1, ids, n, k).into_iter().enumerate() {
        let (wt, at) = (w[j / n] as f64, j / n / topk * n + j % n);
        want[at] += wt * d;
        abs_terms[at] += (wt.abs() * mag) as f32;
    }
    (want.into_iter().map(|v| v as f32).collect(), abs_terms)
}

/// Per routed slot and row, `act(gate·x, up·x)` in f64; the product's terms (each factor's magnitude plus its terms'
/// sum); and how often the gate and the up factor were clamped.
#[allow(clippy::too_many_arguments)]
fn gateup_reference(
    gate: &[f32], up: &[f32], x: &[f32], ids: &[u32], topk: usize, n: usize, k: usize, activation: ExpertActivation,
) -> (Vec<f32>, Vec<f32>, [usize; 2]) {
    let silu = |g: f64| g / (1.0 + (-g).exp());
    let mut clamped = [0usize; 2];
    let (want, abs_terms) = dots(gate, x, topk, ids, n, k)
        .into_iter()
        .zip(dots(up, x, topk, ids, n, k))
        .map(|((g, ag), (u, au))| {
            let v = match activation {
                ExpertActivation::Silu => silu(g) * u,
                ExpertActivation::SiluClamp { limit } => {
                    let l = f64::from(limit);
                    clamped[0] += usize::from(g > l);
                    clamped[1] += usize::from(u.abs() > l);
                    silu(g.min(l)) * u.clamp(-l, l)
                }
                ExpertActivation::SituGlu { beta_gate, beta_up } => situ_glu(g, u, beta_gate.into(), beta_up.into()),
            };
            (v as f32, ((g.abs() + ag) * (u.abs() + au)) as f32)
        })
        .unzip();
    (want, abs_terms, clamped)
}

/// SiTU-GLU of scalars, transcribed from the oracle, `oscar_models::situ`.
fn situ_glu(g: f64, u: f64, beta_gate: f64, beta_up: f64) -> f64 {
    let softcap = |x: f64, b: f64| b * (x / b).tanh();
    softcap(g, beta_gate) / (1.0 + (-g).exp()) * softcap(u, beta_up)
}

/// The f32 accumulation bound `2·K·2^-24·Σ|terms|` (Higham's (K−1)u, doubled for rounded products), plus `per_term`.
fn assert_within_bound(got: &[f32], want: &[f32], abs_terms: &[f32], terms: usize, per_term: f64, what: &str) {
    let u = 2.0 * terms as f64 * 2f64.powi(-24) + per_term;
    let ratio = |i: usize| (got[i] as f64 - want[i] as f64).abs() / (u * abs_terms[i] as f64);
    let outside: Vec<usize> = (0..got.len()).filter(|&i| ratio(i) > 1.0).collect();
    let (worst, of, first) =
        ((0..got.len()).map(ratio).fold(0f64, f64::max), (outside.len(), got.len()), outside.first());
    assert!(outside.is_empty(), "{what}: {of:?} outside the bound, the worst at {worst:.2} of it, first {first:?}");
}

/// The worst `|got − want| / (1 + |want|)`.
fn max_rel(got: &[f32], want: &[f32]) -> f32 {
    got.iter().zip(want).map(|(g, w)| (g - w).abs() / (1.0 + w.abs())).fold(0f32, f32::max)
}

/// K3's gate+up over 5 experts, 7 tokens at top-4: its worst [`max_rel`] against the oracle, and the output. With
/// `shift`, the kernel reads the gate stack one block along, scored against the unshifted reference.
fn situ_gateup(
    device: &Device, qkk: QkkMoeDType, activation: ExpertActivation, (n, k): (usize, usize), shift: bool,
) -> Result<(f32, Vec<f32>)> {
    let (experts, tokens, topk) = (5usize, 7usize, 4usize);
    let nblocks = experts * n * (k / 256);
    let (full, at) = (k3_blocks(qkk, nblocks + usize::from(shift), 0x51ed), usize::from(shift) * qkk.block_bytes());
    let dq = dequantize(qkk, &full[..nblocks * qkk.block_bytes()], device)?;
    let gate = Stack::new(full[at..].to_vec(), dq, experts, device)?;
    let up = Stack::iq(qkk, k3_blocks(qkk, nblocks, 0xa733), experts, device)?;
    let x_v = sine(tokens * k, 0.0029, 1.7);
    let x = staged(x_v.clone(), (tokens, k), device)?;
    let (ids_v, ids) = routing(tokens, topk, |t, s| ((t * topk + s) * 3 + 1) % experts, device)?;
    let miss = Tensor::zeros(1, DType::U32, device)?;
    let out = host(&moe_gateup(gate.row()?.get(), up.row()?.get(), qkk, activation, experts, n, k, &ids, &x, &miss)?)?;
    assert_eq!(miss.to_vec1::<u32>()?[0], 0, "a full row must produce zero misses");
    // A relative-error gate over a degenerate output is vacuous, every control passing with it.
    let spread = out.iter().fold(f32::MIN, |a, &b| a.max(b)) - out.iter().fold(f32::MAX, |a, &b| a.min(b));
    assert!(spread > 1e-3, "kernel output is degenerate (spread {spread:e}) - the gate would be vacuous");
    let (want, _, _) = gateup_reference(&gate.dq, &up.dq, &x_v, &ids_v, topk, n, k, activation);
    Ok((max_rel(&out, &want), out))
}

/// `nblocks` i-quant blocks of `byte(i)` at byte `i`, each but its leading f16 scale `scale(b)`: any bits make a legal
/// block, each index spanning its table, but the scale, pinned so tolerances hold (a random half can be NaN).
fn iq_blocks(qkk: QkkMoeDType, nblocks: usize, scale: impl Fn(usize) -> u16, byte: impl FnMut(usize) -> u8) -> Vec<u8> {
    let mut out: Vec<u8> = (0..nblocks * qkk.block_bytes()).map(byte).collect();
    for (b, block) in out.chunks_exact_mut(qkk.block_bytes()).enumerate() {
        block[..2].copy_from_slice(&scale(b).to_le_bytes());
    }
    out
}

/// xorshift64*, so the fill is reproducible without a rand dependency; the scales 0.03125 .. 0.25.
fn scrambled_blocks(qkk: QkkMoeDType, nblocks: usize, seed: u64) -> Vec<u8> {
    const F16_SCALES: [u16; 7] = [0x2800, 0x2A00, 0x2C00, 0x2E00, 0x3000, 0x3200, 0x3400];
    let mut s = seed | 1;
    let next = move |_| {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as u8
    };
    iq_blocks(qkk, nblocks, |b| F16_SCALES[b % F16_SCALES.len()], next)
}

/// K3's blocks: a hash of each byte's index, `seed` decorrelating the gate, up and down stacks so a swapped pool cannot
/// pass, under the pinned scale.
fn k3_blocks(qkk: QkkMoeDType, nblocks: usize, seed: u64) -> Vec<u8> {
    let byte = |i: usize| {
        let x = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(seed);
        ((x ^ (x >> 33)).wrapping_mul(0xff51afd7ed558ccd) >> 29) as u8
    };
    iq_blocks(qkk, nblocks, |_| PINNED_SCALE_BITS, byte)
}

/// candle's `kernel_get_rows_iq*`, ggml's own `dequantize_iq*` over every block in its tile order (16-element tiles
/// of a column-major `float4x4`): not oscar's dot, so a shared misreading is all it cannot catch.
fn dequantize(qkk: QkkMoeDType, bytes: &[u8], device: &Device) -> Result<Vec<f32>> {
    let name = match qkk {
        QkkMoeDType::Iq1S => "kernel_get_rows_iq1_s",
        QkkMoeDType::Iq2XXS => "kernel_get_rows_iq2_xxs",
        QkkMoeDType::Iq3XXS => "kernel_get_rows_iq3_xxs",
        other => panic!("not an i-quant: {other:?}"),
    };
    assert_eq!(bytes.len() % qkk.block_bytes(), 0, "{qkk:?}: {} bytes is not whole blocks", bytes.len());
    let elems = bytes.len() / qkk.block_bytes() * qkk.block_elems();
    let blocks = Tensor::from_vec(bytes.to_vec(), bytes.len(), &Device::Cpu)?.to_device(device)?;
    // The blocks as one row of `elems`, at id 0 of a one-entry table.
    let id = Tensor::zeros(1, DType::U32, device)?;
    let (bm, idm) = (Metal::of(&blocks, "blocks")?, Metal::of(&id, "id")?);
    let (rows, o) = out(bm.dev, elems, DType::F32, "iq-dequantised")?;
    let p = candle_pipeline(bm.dev, name)?;
    let tg = p.max_total_threads_per_threadgroup() as usize;
    launch(bm.dev, &p, "iq-dequantise", [1, 1, 1], [tg, 1, 1], |e| {
        set_params!(e, (&bm, &idm, &o, elems as i64, 0u64, 0u64, 1i64, 4u64, 4u64, 0u64, 0u64));
        Ok(())
    })?;
    host(&rows)
}
