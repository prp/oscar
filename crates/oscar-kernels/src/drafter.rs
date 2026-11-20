use candle_core::{DType, Result, Tensor};
use candle_metal_kernels::metal::ComputeCommandEncoder;

use crate::loader::{launch, out, pipeline, Family, Metal};

fn metals<'a>(ts: impl IntoIterator<Item = &'a Tensor>, what: &str) -> Result<Vec<Metal<'a>>> {
    ts.into_iter().map(|t| Metal::of(t, what)).collect()
}

fn bind(e: &ComputeCommandEncoder, first: usize, ms: &[Metal<'_>]) {
    for (i, m) in ms.iter().enumerate() {
        e.set_buffer(first + i, Some(&m.buf), m.off);
    }
}

/// Each expert's snapshot slot into the row the matvecs read, null where the slot's live epoch is not `expected`.
/// `snap` and `expected` U32 `[num_experts]` per projection; `epochs` U32 and `slots` I64, one entry a slot, per bank.
pub fn oscar_draft_row_validate(
    snap: [&Tensor; 3], expected: [&Tensor; 3], epochs: [&Tensor; 3], slots: [&Tensor; 3], out: [&Tensor; 3],
    num_experts: usize, slot_counts: [usize; 3],
) -> Result<()> {
    let ms = metals(snap.into_iter().chain(expected).chain(epochs).chain(slots).chain(out), "draft row tensor")?;
    let dev = ms[0].dev;
    let p = pipeline(dev, Family::Moe, "oscar_draft_row_validate")?;
    let tpg = 128usize.min(p.max_total_threads_per_threadgroup() as usize);
    launch(dev, &p, "oscar-draft-row-validate", [num_experts.div_ceil(tpg), 1, 1], [tpg, 1, 1], |e| {
        bind(e, 0, &ms);
        for (i, v) in [num_experts, slot_counts[0], slot_counts[1], slot_counts[2]].into_iter().enumerate() {
            e.set_bytes(15 + i, &(v as i64));
        }
        Ok(())
    })
}

/// Cold selections below θ (all at θ = 2) become the best-scoring residents, renormalised: `(ids, weights)` as `in_*`.
pub fn oscar_draft_theta_substitute(
    scr_gate: &Tensor, logits: &Tensor, in_ids: &Tensor, in_w: &Tensor, k: usize, num_experts: usize, theta: f32,
) -> Result<(Tensor, Tensor)> {
    let ms = metals([scr_gate, logits, in_ids, in_w], "draft theta tensor")?;
    let dev = ms[0].dev;
    let (ids, ids_m) = out(dev, in_ids.shape().clone(), DType::U32, "draft-theta-ids")?;
    let (w, w_m) = out(dev, in_w.shape().clone(), DType::F32, "draft-theta-weights")?;
    let p = pipeline(dev, Family::Moe, "oscar_draft_theta_substitute")?;
    launch(dev, &p, "oscar-draft-theta-substitute", [1, 1, 1], [1, 1, 1], |e| {
        bind(e, 0, &ms);
        bind(e, 4, &[ids_m, w_m]);
        e.set_bytes(6, &(k as i64));
        e.set_bytes(7, &(num_experts as i64));
        e.set_bytes(8, &theta);
        Ok(())
    })?;
    Ok((ids, w))
}

/// `dirty` (atomic U32) bumps where a routed entry's slot epoch moved while the matvecs read `rows`.
/// `wm_val` (`layer + 1`) into `wm_bufs` retires the layer's draft-read fence.
#[allow(clippy::too_many_arguments)]
pub fn oscar_draft_epoch_recheck(
    topk_ids: &Tensor, snap: [&Tensor; 3], rows: [&Tensor; 3], expected: [&Tensor; 3], epochs: [&Tensor; 3],
    dirty: &Tensor, num_experts: usize, slot_counts: [usize; 3],
    (wm_bufs, wm_val): (&[candle_metal_kernels::metal::Buffer; 3], i64),
) -> Result<()> {
    let count = topk_ids.elem_count();
    if count == 0 {
        return Ok(());
    }
    let ids_c = topk_ids.contiguous()?;
    let ids = Metal::of(&ids_c, "topk_ids")?;
    let dev = ids.dev;
    let p = pipeline(dev, Family::Moe, "oscar_draft_epoch_recheck")?;
    let ms = metals(snap.into_iter().chain(rows).chain(expected).chain(epochs), "draft row tensor")?;
    let dirty = Metal::of(dirty, "dirty")?;
    let tpg = 64usize.min(p.max_total_threads_per_threadgroup() as usize);
    launch(dev, &p, "oscar-draft-epoch-recheck", [count.div_ceil(tpg), 1, 1], [tpg, 1, 1], |e| {
        bind(e, 0, std::slice::from_ref(&ids));
        bind(e, 1, &ms);
        bind(e, 13, std::slice::from_ref(&dirty));
        for (i, v) in [count, num_experts, slot_counts[0], slot_counts[1], slot_counts[2]].into_iter().enumerate() {
            e.set_bytes(14 + i, &(v as i64));
        }
        for (i, b) in wm_bufs.iter().enumerate() {
            e.set_buffer(19 + i, Some(b), 0);
        }
        e.set_bytes(22, &wm_val);
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// E = 8, k = 4, θ = 0.10: the resident 5 stays, the cold 6 becomes the best resident, 1, the weights renormalised;
    /// with every selection resident nothing changes.
    #[test]
    fn theta_substitute_matches_the_host_rule() -> Result<()> {
        let device = oscar_offload::pools::test_device()?;
        let logits = Tensor::new(&[2.0f32, 1.5, 1.0, 0.5, 0.2, 0.1, -0.5, -1.0], &device)?;
        let in_w = [0.6251824f32, 0.22999177, 0.09350768, 0.0513181];
        let fired = [0.47081451f32, 0.17320298, 0.070419076, 0.28556344];
        for (in_ids, resident, want_ids, want_w, tol) in [
            ([0u32, 2, 5, 6], &[1usize, 3, 5][..], [0u32, 2, 5, 1], fired, 1e-6f32),
            ([0, 1, 2, 3], &[0, 1, 2, 3][..], [0, 1, 2, 3], in_w, 0.0),
        ] {
            // The kernel tests the gate row only for null, so any address is resident.
            let row: Vec<i64> = (0..8).map(|e| if resident.contains(&e) { 4096 } else { 0 }).collect();
            let row = Tensor::new(row.as_slice(), &device)?;
            let (ids, w) = (Tensor::new(&in_ids, &device)?, Tensor::new(&in_w, &device)?);
            let (ids, w) = oscar_draft_theta_substitute(&row, &logits, &ids, &w, 4, 8, 0.10)?;
            device.synchronize()?;
            assert_eq!(ids.to_vec1::<u32>()?, want_ids);
            for (i, (got, want)) in w.to_vec1::<f32>()?.into_iter().zip(want_w).enumerate() {
                assert!((got - want).abs() <= tol, "weight {i}: {got} vs {want}");
            }
        }
        Ok(())
    }
}
