//! The paged attention kernels against a scalar f32 reference, to the dtype's noise floor as the reduction is online.
//! A `q` row is one query token, its `context_lens` entry its visible keys (position + 1, 0 = padding).

use candle_core::{DType, Device, Tensor};
use oscar_offload::pools::test_device;

use super::{paged_attention, reshape_and_cache};

/// Each block's row address in one `[num_blocks, ...row]` buffer, as the block pool's tables give them.
fn address_table(pool: &Tensor) -> Tensor {
    let block_bytes = pool.dims()[1..].iter().product::<usize>() * pool.dtype().size_in_bytes();
    crate::stack_addresses(pool, pool.dims()[0], block_bytes).unwrap()
}

/// Deterministic values in [-1, 1).
fn synth(i: usize) -> f32 {
    let h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(31).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    ((h >> 40) as f32 / (1u64 << 24) as f32) - 1.0
}

fn to_f32(t: &Tensor) -> Vec<Vec<Vec<f32>>> {
    t.to_dtype(DType::F32).unwrap().to_vec3().unwrap()
}

#[derive(Debug)]
struct Scenario {
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    /// Per sequence: (cached prefix tokens, new query tokens).
    seqs: Vec<(usize, usize)>,
    dtype: DType,
    /// Shifts the synthetic q stream, so a chunked scenario queries bit-identically to a whole one.
    q_pos_offset: usize,
}

impl Scenario {
    /// BF16 with the query stream unshifted.
    fn new(q_heads: usize, kv_heads: usize, head_dim: usize, block_size: usize, seqs: Vec<(usize, usize)>) -> Self {
        Self { q_heads, kv_heads, head_dim, block_size, seqs, dtype: DType::BF16, q_pos_offset: 0 }
    }
}

/// The kernel's output and its dtype-rounded inputs: `q_rows` [row][q_head][hd], `k` and `v` [seq][pos][kv_head][hd].
struct Run {
    q_rows: Vec<Vec<Vec<f32>>>,
    k: Vec<Vec<Vec<Vec<f32>>>>,
    v: Vec<Vec<Vec<Vec<f32>>>>,
    ctx: Vec<u32>,
    max_q_len: usize,
    out: Vec<f32>,
}

/// Every position cached through `reshape_and_cache`, as prompt chunks write before attending, then the kernel run.
fn run(s: &Scenario) -> Run {
    let dev = test_device().expect("metal device");
    let (hd, x, bs) = (s.head_dim, 16 / s.dtype.size_in_bytes(), s.block_size);
    let max_q_len = s.seqs.iter().map(|&(_, n)| n).max().unwrap();
    let max_ctx = s.seqs.iter().map(|&(c, n)| c + n).max().unwrap();
    let blocks_per_seq = max_ctx.div_ceil(bs);
    // Block 0 stays unused.
    let num_blocks = blocks_per_seq * s.seqs.len() + 1;
    let k_cache = Tensor::zeros((num_blocks, s.kv_heads, hd / x, bs, x), s.dtype, &dev).unwrap();
    let v_cache = Tensor::zeros((num_blocks, s.kv_heads, hd, bs), s.dtype, &dev).unwrap();
    let (k_addrs, v_addrs) = (address_table(&k_cache), address_table(&v_cache));
    let rounded = |host: Vec<f32>, shape: (usize, usize, usize)| {
        Tensor::from_vec(host, shape, &dev).unwrap().to_dtype(s.dtype).unwrap()
    };
    let rows = s.seqs.len() * max_q_len;
    let (mut k, mut v, mut tables, mut ctx) = (Vec::new(), Vec::new(), Vec::new(), vec![0u32; rows]);
    let mut q_host = vec![0f32; rows * s.q_heads * hd];
    for (si, &(cached, new)) in s.seqs.iter().enumerate() {
        let len = cached + new;
        let table: Vec<u32> = (0..blocks_per_seq).map(|b| (1 + si * blocks_per_seq + b) as u32).collect();
        let synth_from = |base: usize| (0..len * s.kv_heads * hd).map(|i| synth(base + i)).collect::<Vec<f32>>();
        let k_t = rounded(synth_from(si * 7_000_003), (len, s.kv_heads, hd));
        let v_t = rounded(synth_from(si * 7_000_003 + 3_000_017), (len, s.kv_heads, hd));
        let slots: Vec<i64> = (0..len).map(|p| (table[p / bs] as usize * bs + p % bs) as i64).collect();
        let slots = Tensor::from_vec(slots, (len,), &dev).unwrap();
        reshape_and_cache(&k_t, &v_t, &k_addrs, &v_addrs, bs, &slots).unwrap();
        k.push(to_f32(&k_t));
        v.push(to_f32(&v_t));
        tables.push(table);
        for i in 0..new {
            let row = si * max_q_len + i;
            ctx[row] = (cached + i + 1) as u32;
            let q_row = &mut q_host[row * s.q_heads * hd..][..s.q_heads * hd];
            let base = si * 11_000_027 + 5_000_011 + (i + s.q_pos_offset) * s.q_heads * hd;
            q_row.iter_mut().enumerate().for_each(|(e, q)| *q = synth(base + e));
        }
    }
    let q = rounded(q_host, (rows, s.q_heads, hd));
    let q_rows = to_f32(&q);
    let tables = Tensor::from_vec(tables.concat(), (s.seqs.len(), blocks_per_seq), &dev).unwrap();
    let lens = Tensor::from_vec(ctx.clone(), (rows,), &dev).unwrap();
    let scale = 1.0 / (hd as f32).sqrt();
    let out: Vec<f32> = paged_attention(&q, &k_addrs, &v_addrs, s.kv_heads, bs, &tables, &lens, max_ctx, scale, 1.0)
        .and_then(|o| o.to_dtype(DType::F32)?.flatten_all()?.to_vec1())
        .unwrap();
    Run { q_rows, k, v, ctx, max_q_len, out }
}

/// Scalar f32 attention per row over the same rounded inputs; a padding row stays zero.
fn reference(s: &Scenario, r: &Run) -> Vec<f32> {
    let (hd, gqa) = (s.head_dim, s.q_heads / s.kv_heads);
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0f32; r.ctx.len() * s.q_heads * hd];
    for (row, &visible) in r.ctx.iter().enumerate() {
        let seq = row / r.max_q_len;
        for h in 0..s.q_heads {
            let q = &r.q_rows[row][h];
            let scores: Vec<f32> = (0..visible as usize)
                .map(|p| scale * q.iter().zip(&r.k[seq][p][h / gqa]).map(|(a, b)| a * b).sum::<f32>())
                .collect();
            let m = scores.iter().cloned().fold(f32::MIN, f32::max);
            let weights: Vec<f32> = scores.iter().map(|sc| (sc - m).exp()).collect();
            let sum: f32 = weights.iter().sum();
            let dst = &mut out[(row * s.q_heads + h) * hd..][..hd];
            for (w, v) in weights.iter().zip(&r.v[seq]) {
                dst.iter_mut().zip(&v[h / gqa]).for_each(|(d, val)| *d += w / sum * val);
            }
        }
    }
    out
}

/// The largest |Δ| and where it falls; a NaN ranks above every number.
fn worst(got: &[f32], want: &[f32]) -> (usize, f32) {
    assert_eq!(got.len(), want.len());
    got.iter().zip(want).map(|(g, w)| (g - w).abs()).enumerate().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap()
}

/// The kernel within `tol` of the reference; the run comes back for more checks.
fn assert_close(s: &Scenario, tol: f32) -> Run {
    let r = run(s);
    let want = reference(s, &r);
    let (i, d) = worst(&r.out, &want);
    assert!(d <= tol, "max |Δ| {d} at flat index {i} exceeds tol {tol} (got {}, want {}) on {s:?}", r.out[i], want[i]);
    r
}

/// Decode and prompt match the reference; a ragged batch's pads write zeros; a later chunk sees the earlier ones as
/// cached prefix through its context lengths alone; staging normalises the layout before any math, so a prompt's
/// output bytes are invariant across block sizes (decode's reduction keys on the block size, so it is not).
#[test]
fn decode_and_prompt_match_the_reference() {
    let mut cases = Vec::new();
    // Every fleet decode geometry and block size, one split and many, in a ragged batch leaving splits past a context.
    for (q_heads, kv_heads, head_dim) in [(16, 2, 256), (24, 2, 256), (32, 2, 256), (64, 4, 128), (8, 2, 128)] {
        for (cached, bs) in [(4, 32), (39, 32), (699, 32), (699, 16), (699, 8)] {
            let seqs = vec![(cached, 1), (cached / 2, 1), (0, 1)];
            cases.push((Scenario::new(q_heads, kv_heads, head_dim, bs, seqs), 2.5e-2));
        }
    }
    cases.extend([
        // Past 8,192 tokens two blocks share a split: the in-split online step.
        (Scenario::new(16, 2, 256, 32, vec![(8300, 1), (40, 1)]), 2.5e-2),
        // Few tiles over several 512-token windows: the multi-window online reduction and the cross-split reduce.
        (Scenario::new(64, 4, 128, 32, vec![(2900, 64)]), 2.5e-2),
        // A short continuation on a long prefix, the follow-up turn: the per-window split path.
        (Scenario::new(16, 2, 256, 32, vec![(1490, 9)]), 2.5e-2),
        // An order of magnitude tighter: every accumulator is fp32 either way.
        (Scenario { dtype: DType::F32, ..Scenario::new(8, 2, 128, 32, vec![(600, 21)]) }, 2e-3),
    ]);
    cases.iter().for_each(|(s, tol)| _ = assert_close(s, *tol));
    // Rows 5..13 of the second sequence are padding.
    let s = Scenario::new(8, 2, 64, 16, vec![(50, 13), (0, 5)]);
    let (out, row) = (assert_close(&s, 2.5e-2).out, s.q_heads * s.head_dim);
    assert_eq!(out[(13 + 5) * row..(13 + 13) * row].iter().position(|&x| x != 0.0), None, "a pad row holds a nonzero");
    // The last 17 of 48 queried after a 31-token prefix; `q_pos_offset` aligns the queries.
    let whole = run(&Scenario::new(16, 2, 256, 32, vec![(0, 48)])).out;
    let tail = run(&Scenario { q_pos_offset: 31, ..Scenario::new(16, 2, 256, 32, vec![(31, 17)]) }).out;
    let row = 16 * 256;
    let diverged = (0..17).find(|i| whole[(31 + i) * row..][..row] != tail[i * row..][..row]);
    assert_eq!(diverged, None, "a chunked row diverged from the whole-prompt row");
    for (q_heads, kv_heads, head_dim, seq) in [(32, 2, 256, (96, 37)), (64, 4, 128, (700, 64))] {
        let outs: Vec<Vec<f32>> = [8, 16, 32]
            .iter()
            .map(|&bs| assert_close(&Scenario::new(q_heads, kv_heads, head_dim, bs, vec![seq]), 2.5e-2).out)
            .collect();
        assert!(outs[0] == outs[1] && outs[1] == outs[2], "hd{head_dim}: the block sizes' outputs diverged");
    }
}

/// How serve's gather hands K/V to the causal `sdpa`, and what the queue and buffer pool hold then.
#[derive(Default)]
struct Gather {
    /// Rows padded to whole blocks, NaN past the visible length, narrowed for the kernel.
    padded: bool,
    /// Narrow before adding the batch axis, as serve does: batch stride = head stride.
    narrow_first: bool,
    /// NaN-fill the buffer pool's power-of-two buckets first: a tensor's slack holds garbage in a long-lived serve.
    poison_pool: bool,
    /// NaN-heavy results left unread on the queue just before the dispatch.
    dirty_queue: bool,
}

/// The gather arm's `sdpa` on a one-sequence scenario, against the reference.
fn assert_sdpa_close(dev: &Device, s: &Scenario, g: &Gather, what: &str) {
    let r = run(s);
    let want = reference(s, &r);
    let (kv_heads, hd) = (s.kv_heads, s.head_dim);
    let (cached, new) = s.seqs[0];
    let total = cached + new;
    let rows = if g.padded { total.div_ceil(s.block_size) * s.block_size } else { total };
    let pack = |src: &[Vec<Vec<f32>>]| {
        let mut host = vec![f32::NAN; kv_heads * rows * hd];
        for (p, heads) in src.iter().enumerate() {
            heads.iter().enumerate().for_each(|(h, row)| host[(h * rows + p) * hd..][..hd].copy_from_slice(row));
        }
        host
    };
    let (k_host, v_host) = (pack(&r.k[0]), pack(&r.v[0]));
    if g.poison_pool {
        [1usize << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21].iter().for_each(|b| _ = Tensor::full(f32::NAN, b / 4, dev));
    }
    let q = Tensor::from_vec(r.q_rows.concat().concat(), (new, s.q_heads, hd), dev)
        .and_then(|t| t.transpose(0, 1)?.contiguous()?.unsqueeze(0)?.to_dtype(DType::BF16))
        .unwrap();
    let kv = |host: Vec<f32>| {
        let t = Tensor::from_vec(host, (kv_heads, rows, hd), dev)?;
        if g.narrow_first {
            t.to_dtype(DType::BF16)?.narrow(1, 0, total)?.unsqueeze(0)
        } else {
            t.unsqueeze(0)?.to_dtype(DType::BF16)?.narrow(2, 0, total)
        }
    };
    let (k, v) = (kv(k_host).unwrap(), kv(v_host).unwrap());
    if g.narrow_first {
        assert_eq!(v.stride()[0], v.stride()[1], "{what}: wants batch stride == head stride");
    }
    let _residue = g.dirty_queue.then(|| {
        let nan = Tensor::full(f32::NAN, (64, 512), dev).unwrap().to_dtype(DType::BF16).unwrap();
        let prod = nan.matmul(&nan.t().unwrap()).unwrap();
        let soft = candle_nn::ops::softmax_last_dim(&prod).unwrap();
        (prod, soft)
    });
    let scale = 1.0 / (hd as f32).sqrt();
    let out: Vec<f32> = candle_nn::ops::sdpa(&q, &k, &v, None, true, scale, 1.0)
        .and_then(|o| o.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?.flatten_all()?.to_vec1())
        .unwrap();
    let nans = out.iter().filter(|x| x.is_nan()).count();
    assert_eq!(nans, 0, "{what}: {nans} NaNs in the SDPA output");
    let (_, d) = worst(&out, &want);
    assert!(d <= 2.5e-2, "{what}: SDPA vs reference max |Δ| {d}");
}

#[test]
fn sdpa_matches_the_reference_as_serve_gathers_kv() {
    let dev = test_device().expect("metal device");
    // A chunk's rows sit at the end of the visible keys, so a causal `sdpa` must let row i see keys 0..=cached+i.
    for (q_heads, kv_heads, head_dim, cached, new) in
        [(16, 2, 256, 0, 96), (16, 2, 256, 300, 64), (32, 2, 256, 128, 48), (64, 4, 128, 200, 32)]
    {
        let s = Scenario::new(q_heads, kv_heads, head_dim, 32, vec![(cached, new)]);
        let what = format!("predicate SDPA, q{q_heads}/kv{kv_heads}/hd{head_dim}, cached {cached}, new {new}");
        assert_sdpa_close(&dev, &s, &Gather::default(), &what);
    }
    // A template runt, the short tail a deposit cut leaves, on the 397B's attention.
    let runt = Scenario::new(32, 2, 256, 32, vec![(566, 5)]);
    // Repeated, quiet and after NaN-laden dispatches: NaN from clean inputs is position-dependent.
    for dirty_queue in [false, true] {
        for round in 0..10 {
            let g = Gather { dirty_queue, ..Gather::default() };
            assert_sdpa_close(&dev, &runt, &g, &format!("runt chunk, dirty {dirty_queue}, round {round}"));
        }
    }
    // A final tile relying on zero weights instead of masked loads turns NaN slack into `0 * NaN = NaN`.
    assert_sdpa_close(&dev, &runt, &Gather { padded: true, ..Gather::default() }, "NaN tail rows past the narrow");
    // The batch dim takes the head dim's stride; misread, the second kv head's query group reads the NaN pad rows.
    let g = Gather { padded: true, narrow_first: true, poison_pool: true, ..Gather::default() };
    assert_sdpa_close(&dev, &runt, &g, "narrow-then-unsqueeze K/V strides");
}
