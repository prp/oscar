//! Absorbed MLA: the paged latent rows' write and gather, and the causal prefill over the latent as blocked GEMMs
//! under a running softmax, F32 throughout.

use std::cell::Cell;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use candle_core::{DType, MetalDevice, Result, Tensor};
use objc2_metal::MTLSharedEvent as _;

use crate::loader::{launch, out, pipeline, set_params, Family, Metal};

/// Keys a block of scores spans: `rows × MLA_PREFILL_BLOCK` F32 is the prefill's largest transient.
pub const MLA_PREFILL_BLOCK: usize = 1024;

/// The prefill's working set, one a forward: a block's scores and products in two slots taken in turn once the GPU is
/// past each slot's last block, so the host encodes one block ahead and the plan charges two blocks, not a context.
pub struct PrefillRing {
    scores: [Tensor; 2],
    products: [Tensor; 2],
    gens: [Cell<u64>; 2],
    next: Cell<usize>,
}

impl PrefillRing {
    pub fn new(dev: &MetalDevice, rows_max: usize, dv: usize) -> Result<Self> {
        let fresh = |n: usize, label: &'static str| -> Result<Tensor> { Ok(out(dev, n, DType::F32, label)?.0) };
        let (s, p) = (rows_max * MLA_PREFILL_BLOCK, rows_max * dv);
        Ok(Self {
            scores: [fresh(s, "mla-scores")?, fresh(s, "mla-scores")?],
            products: [fresh(p, "mla-products")?, fresh(p, "mla-products")?],
            gens: [Cell::new(0), Cell::new(0)],
            next: Cell::new(0),
        })
    }

    /// The next slot once the GPU is past its last block; a slot held past five seconds is a stalled GPU.
    fn claim(&self, dev: &MetalDevice) -> Result<usize> {
        let i = self.next.get();
        self.next.set((i + 1) % 2);
        let need = self.gens[i].get();
        if need == 0 {
            return Ok(i);
        }
        let event = dev.completion_event()?;
        let deadline = Instant::now() + Duration::from_secs(5);
        while event.signaledValue() < need {
            refuse_if!(
                Instant::now() >= deadline,
                "the prefill ring's slot {i} is held past 5 s at generation {need}, the GPU at {}",
                event.signaledValue()
            );
            std::thread::sleep(Duration::from_micros(50));
        }
        Ok(i)
    }
}

/// Causal attention of `q` `(h, q_len, dk)` over `k` `(kv_len, dk)` and `v` `(kv_len, dv)`, F32 with contiguous rows,
/// the last `q_len` keys the queries' own, to `(h, q_len, dv)`. A key block's scores are one GEMM into a ring slot, the
/// softmax runs in place against the rows' `(m, l)`, a second GEMM adds its products; a row's result is launch-invariant.
pub fn attend_prefill(q: &Tensor, k: &Tensor, v: &Tensor, softmax_scale: f64, ring: &PrefillRing) -> Result<Tensor> {
    let (h, q_len, dk) = q.dims3()?;
    let (kv_len, dv) = (k.dim(0)?, v.dim(1)?);
    refuse_if!(k.dim(1)? != dk, "attend_prefill: D_q ({dk}) != D_k ({})", k.dim(1)?);
    refuse_if!(v.dim(0)? != kv_len, "attend_prefill: K len ({kv_len}) != V len ({})", v.dim(0)?);
    refuse_if!(q_len > kv_len, "attend_prefill: q_len ({q_len}) > kv_len ({kv_len}) - causal offset undefined");
    refuse_if!(
        k.dtype() != DType::F32 || v.dtype() != DType::F32 || !k.is_contiguous() || !v.is_contiguous(),
        "attend_prefill: k and v must be contiguous F32"
    );
    let rows = h * q_len;
    refuse_if!(rows * MLA_PREFILL_BLOCK > ring.scores[0].elem_count(), "attend_prefill: {rows} rows outgrow the ring");
    let q = (q.to_dtype(DType::F32)?.reshape((rows, dk))? * softmax_scale)?;
    let qm = Metal::of(&q, "attend_prefill: q")?;
    let dev = qm.dev;
    let (o, om) = out(dev, (rows, dv), DType::F32, "mla-o")?;
    let (_m, mm) = out(dev, rows, DType::F32, "mla-m")?;
    let (_l, lm) = out(dev, rows, DType::F32, "mla-l")?;
    let (_alpha, am) = out(dev, rows, DType::F32, "mla-alpha")?;
    let softmax = pipeline(dev, Family::Mla, "mla_online_softmax")?;
    let accumulate = pipeline(dev, Family::Mla, "mla_accumulate")?;
    let mut b0 = 0;
    while b0 < kv_len {
        let n = MLA_PREFILL_BLOCK.min(kv_len - b0);
        let (first, last) = (b0 == 0, b0 + n == kv_len);
        let slot = ring.claim(dev)?;
        let (s, p) = (&ring.scores[slot], &ring.products[slot]);
        let (sm, pm) = (Metal::of(s, "attend_prefill: scores")?, Metal::of(p, "attend_prefill: products")?);
        gemm_into(&q, &k.narrow(0, b0, n)?.t()?, &sm, (rows, n, dk))?;
        launch(dev, &softmax, "mla-online-softmax", [rows.div_ceil(8), 1, 1], [256, 1, 1], |e| {
            set_params!(
                e,
                (
                    &sm,
                    &mm,
                    &lm,
                    &am,
                    rows as i32,
                    n as i32,
                    q_len as i32,
                    (kv_len - q_len) as i32,
                    b0 as i32,
                    first as i32
                )
            );
            Ok(())
        })?;
        gemm_into(&s.narrow(0, 0, rows * n)?.reshape((rows, n))?, &v.narrow(0, b0, n)?, &pm, (rows, dv, n))?;
        launch(dev, &accumulate, "mla-accumulate", [dv.div_ceil(64), rows, 1], [64, 1, 1], |e| {
            set_params!(e, (&om, &pm, &am, &lm, rows as i32, dv as i32, first as i32, last as i32));
            Ok(())
        })?;
        dev.flush_commands()?;
        ring.gens[slot].set(dev.completion_counter()?.load(Ordering::SeqCst));
        b0 += n;
    }
    o.reshape((h, q_len, dv))
}

/// `lhs (m, k) · rhs (k, n)`, either operand transposed in its strides, into `out`'s first `m × n` floats.
fn gemm_into(lhs: &Tensor, rhs: &Tensor, out: &Metal, (m, n, k): (usize, usize, usize)) -> Result<()> {
    let (lm, rm) = (Metal::of(lhs, "gemm: lhs")?, Metal::of(rhs, "gemm: rhs")?);
    let dev = lm.dev;
    let encoder = dev.command_encoder()?;
    encoder.set_label("mla-prefill-gemm");
    candle_metal_kernels::call_mlx_gemm(
        dev.device(),
        &encoder,
        dev.kernels(),
        candle_metal_kernels::GemmDType::F32,
        (1, m, n, k),
        lhs.layout().stride(),
        lm.off,
        &lm.buf,
        rhs.layout().stride(),
        rm.off,
        &rm.buf,
        &out.buf,
    )
    .map_err(|e| candle_core::Error::Msg(format!("mla prefill gemm: {e:?}")))
}

fn rows_of<'a>(t: &'a Tensor, what: &str) -> Result<Metal<'a>> {
    refuse_if!(!t.is_contiguous(), "paged latent rows: {what} must be contiguous");
    Metal::of(t, &format!("paged latent rows: {what}"))
}

fn check_address_table(t: &Tensor, what: &str) -> Result<()> {
    refuse_if!(
        t.dtype() != DType::I64 || t.rank() != 1,
        "paged latent rows: {what} must be a rank-1 I64 address table, got {:?} {:?}",
        t.dtype(),
        t.shape()
    );
    Ok(())
}

/// BF16 `ckv` and `kpe` rows into the paged blocks in place, by `slot_mapping` (negative = padding), after their
/// producers.
pub fn write_latent_rows(
    ckv: &Tensor, kpe: Option<(&Tensor, &Tensor)>, ckv_addrs: &Tensor, block_size: usize, slot_mapping: &Tensor,
) -> Result<()> {
    let (n, rank) = ckv.dims2()?;
    refuse_if!(ckv.dtype() != DType::BF16, "write_latent_rows: the pool is BF16, got {:?}", ckv.dtype());
    check_address_table(ckv_addrs, "latent address table")?;
    refuse_if!(slot_mapping.dims1()? != n, "write_latent_rows: {} slots for {n} tokens", slot_mapping.dims1()?);
    refuse_if!(slot_mapping.dtype() != DType::I64, "write_latent_rows: slot mapping must be I64");
    // Without rope keys the latent rows and table stand in, unread at a rope width of 0.
    let (k, addrs) = kpe.unwrap_or((ckv, ckv_addrs));
    let (n_kpe, rope) = k.dims2()?;
    refuse_if!(n != n_kpe, "write_latent_rows: rows [{n}, {rank}] vs [{n_kpe}, {rope}]");
    check_address_table(addrs, "rope address table")?;
    refuse_if!(k.dtype() != ckv.dtype(), "write_latent_rows: mixed dtypes");
    let rope = if kpe.is_some() { rope } else { 0 };
    if n == 0 {
        return Ok(());
    }
    let ckvm = rows_of(ckv, "ckv rows")?;
    let (kpem, kpe_addrsm) = (rows_of(k, "kpe rows")?, rows_of(addrs, "rope address table")?);
    let addrsm = rows_of(ckv_addrs, "latent address table")?;
    let slotsm = rows_of(slot_mapping, "slot mapping")?;
    let dev = ckvm.dev;
    let p = pipeline(dev, Family::Mla, "concat_and_cache_mla")?;
    launch(dev, &p, "write-latent-rows", [(rank + rope).div_ceil(64), n, 1], [64, 1, 1], |e| {
        set_params!(e, (&ckvm, &kpem, &addrsm, &kpe_addrsm, &slotsm, rank as i32, rope as i32, block_size as i32));
        Ok(())
    })
}

/// The BF16 rows at `slots` (U32, token order) from `addrs`' blocks: the read side of [`write_latent_rows`].
pub fn gather_latent_rows(addrs: &Tensor, slots: &Tensor, width: usize, block_size: usize) -> Result<Tensor> {
    check_address_table(addrs, "address table")?;
    let n = slots.dims1()?;
    refuse_if!(slots.dtype() != DType::U32, "gather_latent_rows: slots must be U32");
    let addrsm = rows_of(addrs, "address table")?;
    let device = addrsm.dev;
    if n == 0 || width == 0 {
        return Tensor::zeros((n, width), DType::BF16, addrs.device());
    }
    let (rows, o) = out(device, (n, width), DType::BF16, "gather-latent-rows")?;
    let slotsm = rows_of(slots, "slots")?;
    let p = pipeline(device, Family::Mla, "gather_latent_rows")?;
    launch(device, &p, "gather-latent-rows", [width.div_ceil(64), n, 1], [64, 1, 1], |e| {
        set_params!(e, (&addrsm, &slotsm, &o, width as i32, block_size as i32));
        Ok(())
    })?;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::D;

    /// Uniform in [-1, 1) from a 64-bit LCG.
    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    /// Against an F32 matmul + softmax reference: a causal offset, lengths off the block, heads and widths varied.
    #[test]
    fn the_prefill_matches_the_naive_path() -> Result<()> {
        let device = oscar_offload::pools::test_device()?;
        let candle_core::Device::Metal(dev) = &device else { unreachable!("a Metal test device") };
        let ring = PrefillRing::new(dev, 34 * 1030, 512)?;
        let scale = 1.0f64 / (192.0f64).sqrt();
        for (dk, dv, n_head, q_len, k_len, dtype, tol) in [
            (576usize, 512usize, 4usize, 13usize, 13usize, DType::F32, 2e-4f32),
            (576, 512, 4, 13, 18, DType::F32, 2e-4),
            (576, 512, 4, 32, 32, DType::BF16, 2e-2),
            (576, 512, 34, 19, 40, DType::BF16, 2e-2),
            (512, 512, 34, 19, 40, DType::F32, 2e-4),
            // Past one block and off its edge, with and without a causal offset.
            (576, 512, 3, 1030, 1030, DType::BF16, 2e-2),
            (576, 512, 3, 700, 2100, DType::BF16, 2e-2),
            (512, 512, 2, 1024, 2048, DType::F32, 2e-4),
        ] {
            let q = Tensor::from_vec(fill(n_head * q_len * dk, 7), (n_head, q_len, dk), &device)?.to_dtype(dtype)?;
            let k = Tensor::from_vec(fill(k_len * dk, 11), (k_len, dk), &device)?.to_dtype(dtype)?;
            let v = Tensor::from_vec(fill(k_len * dv, 13), (k_len, dv), &device)?.to_dtype(dtype)?;
            let (kf, vf) = (k.to_dtype(DType::F32)?, v.to_dtype(DType::F32)?);
            let got = attend_prefill(&q, &kf, &vf, scale, &ring)?;

            let qf = q.to_dtype(DType::F32)?;
            let att = qf.reshape((n_head * q_len, dk))?.matmul(&kf.t()?.contiguous()?)?;
            let off = k_len - q_len;
            let mask: Vec<f32> = (0..q_len)
                .flat_map(|i| (0..k_len).map(move |j| if j <= i + off { 0.0 } else { f32::NEG_INFINITY }))
                .collect();
            let mask = Tensor::from_vec(mask, (q_len, k_len), &device)?;
            let att = (att * scale)?.reshape((n_head, q_len, k_len))?.broadcast_add(&mask)?;
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            let reference = att.reshape((n_head * q_len, k_len))?.matmul(&vf)?.reshape((n_head, q_len, dv))?;

            let diff = (got - reference)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            assert!(diff < tol, "max abs diff {diff} >= {tol} at dk={dk} heads={n_head} q={q_len} k={k_len} {dtype:?}");
        }
        Ok(())
    }

    /// The address table of a pool laid out as one `[blocks, block_size, width]` tensor.
    fn address_table(pool: &Tensor) -> Result<Tensor> {
        let block_bytes = pool.dims()[1..].iter().product::<usize>() * pool.dtype().size_in_bytes();
        crate::stack_addresses(pool, pool.dims()[0], block_bytes)
    }

    /// Byte for byte, with out-of-order blocks, a partial tail block and a padding slot that leaves its row untouched.
    #[test]
    fn latent_rows_round_trip() -> Result<()> {
        let device = oscar_offload::pools::test_device()?;
        let (rank, rope, block_size, num_blocks) = (512usize, 64usize, 4usize, 5usize);
        // Ten tokens over three blocks out of order, two full and a partial tail; then a padding row, written nowhere.
        let (n, block_table) = (10usize, [2u32, 0, 3]);
        let slots: Vec<u32> =
            (0..n).map(|t| block_table[t / block_size] * block_size as u32 + (t % block_size) as u32).collect();
        let mapping = Tensor::new(slots.iter().map(|&s| s as i64).chain([-1]).collect::<Vec<_>>().as_slice(), &device)?;
        let rows = |width: usize, seed: u64| -> Result<Tensor> {
            Tensor::from_vec(fill((n + 1) * width, seed), (n + 1, width), &device)?.to_dtype(DType::BF16)
        };
        let (ckv, kpe) = (rows(rank, 7)?, rows(rope, 11)?);
        let ckv_cache = Tensor::zeros((num_blocks, block_size, rank), DType::BF16, &device)?;
        let kpe_cache = Tensor::zeros((num_blocks, block_size, rope), DType::BF16, &device)?;
        let (ckv_addrs, kpe_addrs) = (address_table(&ckv_cache)?, address_table(&kpe_cache)?);
        write_latent_rows(&ckv, Some((&kpe, &kpe_addrs)), &ckv_addrs, block_size, &mapping)?;

        // In token order the gathered rows equal the cache's index_select and the source rows.
        let gather = Tensor::new(slots.as_slice(), &device)?;
        let ckv_rows = gather_latent_rows(&ckv_addrs, &gather, rank, block_size)?;
        let kpe_rows = gather_latent_rows(&kpe_addrs, &gather, rope, block_size)?;
        let f32s = |t: &Tensor| -> Result<Vec<Vec<f32>>> { t.to_dtype(DType::F32)?.to_vec2() };
        let flat = ckv_cache.reshape(((), rank))?;
        assert_eq!(f32s(&ckv_rows)?, f32s(&flat.index_select(&gather, 0)?)?, "the gather differs from index_select");
        let got = Tensor::cat(&[&ckv_rows, &kpe_rows], D::Minus1)?;
        let want = Tensor::cat(&[&ckv.narrow(0, 0, n)?, &kpe.narrow(0, 0, n)?], D::Minus1)?;
        assert_eq!(f32s(&got)?, f32s(&want)?, "gathered latents differ from written rows");
        for (row, vals) in f32s(&flat)?.iter().enumerate().filter(|(row, _)| !slots.contains(&(*row as u32))) {
            assert!(vals.iter().all(|v| *v == 0.0), "untouched slot {row} was written");
        }
        Ok(())
    }
}
