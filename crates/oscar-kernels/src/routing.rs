use candle_core::{DType, Result, Tensor};

use crate::loader::{launch, out, pipeline, set_params, Family, Metal};

/// With `norm` the top-k's renormalised softmax, else `exp(logit - max_topk)`; neither bitwise the tensor chain's.
pub fn metal_topk_softmax(logits: &Tensor, topk: usize, norm: bool) -> Result<(Tensor, Tensor)> {
    launch_topk(logits, topk, norm, None, 1.0, 0.0)
}

/// Winners by `sigmoid(logit) + bias`; weights the unbiased scores (over their sum + `sum_eps` with `norm`) × `scale`.
/// Its sigmoid and K-sum are its own, so the bytes are not the tensor chain's.
pub fn metal_topk_biased_sigmoid(
    logits: &Tensor, bias: &Tensor, topk: usize, norm: bool, scale: f32, sum_eps: f32,
) -> Result<(Tensor, Tensor)> {
    launch_topk(logits, topk, norm, Some(bias), scale, sum_eps)
}

/// The top-k kernels' caps: experts in a row, and those picked.
pub const MAX_NCOLS: usize = 1024;
pub const MAX_K: usize = 16;

pub fn topk_fits(ncols: usize, topk: usize) -> bool {
    ncols > 0 && ncols <= MAX_NCOLS && topk > 0 && topk <= MAX_K.min(ncols)
}

fn launch_topk(
    logits: &Tensor, topk: usize, norm: bool, bias: Option<&Tensor>, scale: f32, sum_eps: f32,
) -> Result<(Tensor, Tensor)> {
    refuse_if!(logits.dtype() != DType::F32, "topk router: only F32 logits supported (got {:?})", logits.dtype());
    let dims = logits.dims();
    refuse_if!(dims.is_empty(), "topk router: logits must have at least 1 dim");
    let ncols = dims[dims.len() - 1];
    refuse_if!(!topk_fits(ncols, topk),
        "topk router: top-{topk} of {ncols} past the kernel's caps (0 < top-k <= {MAX_K}, top-k <= experts <= {MAX_NCOLS})");
    let nrows: usize = dims[..dims.len() - 1].iter().product();
    refuse_if!(nrows == 0, "topk router: nrows must be > 0");
    if let Some(b) = bias {
        refuse_if!(
            b.dtype() != DType::F32 || b.elem_count() != ncols,
            "metal_topk_biased_sigmoid: bias must be F32 ({ncols},), got {:?} {:?}",
            b.dtype(),
            b.dims()
        );
    }

    let logits_c = logits.contiguous()?;
    let lm = Metal::of(&logits_c, "topk router: logits")?;
    let p = pipeline(&lm.dev, Family::Moe, "mistralrs_topk_softmax_f32")?;
    let mut out_dims: Vec<usize> = dims[..dims.len() - 1].to_vec();
    out_dims.push(topk);
    let (weights, wm) = out(&lm.dev, out_dims.clone(), DType::F32, "topk-softmax-weights")?;
    let (indices, im) = out(&lm.dev, out_dims, DType::U32, "topk-softmax-indices")?;
    // Mode 1 binds the bias row; mode 0 binds the logits in its slot, unread.
    let bias_c = bias.map(|b| b.contiguous()).transpose()?;
    let bm = match &bias_c {
        Some(b) => Metal::of(b, "metal_topk_biased_sigmoid: bias")?,
        None => Metal::of(&logits_c, "topk router: logits")?,
    };
    let tg = 128.min(p.max_total_threads_per_threadgroup() as usize);
    launch(&lm.dev, &p, "topk-softmax", [nrows, 1, 1], [tg, 1, 1], |e| {
        set_params!(
            e,
            (&lm, &wm, &im, ncols as i32, topk as i32, i32::from(norm), &bm, i32::from(bias.is_some()), scale, sum_eps)
        );
        Ok(())
    })?;
    Ok((weights, indices))
}

/// The pred half's top-b cap: its picked list sizes threadgroup state; the layout clamps the width it plans to this.
pub const EP_PRED_MAX_B: usize = 16;

/// The true block (top-k ids, gen word) and, unless `None`, the tap block (`xs`) and the next layer's pred block
/// (`scores`' top-`look_b`); the caller's commit publishes them, the returned tensors alive until then.
#[allow(clippy::too_many_arguments)]
pub fn metal_ep_publish(
    topk_ids: &Tensor, ids_word_offset: usize, ids_gen_word: u32, scores: Option<&Tensor>, xs: Option<&Tensor>,
    mailbox: &candle_metal_kernels::metal::Buffer, num_experts: usize, look_b: usize, pred_word_offset: usize,
    pred_gen_word: u32, tap_word_offset: usize, tap_gen_word: u32,
) -> Result<Vec<Tensor>> {
    refuse_if!(
        scores.is_some() && (look_b == 0 || look_b > num_experts || look_b > EP_PRED_MAX_B),
        "metal_ep_publish: need 0 < look_b ({look_b}) <= \
             min(num_experts ({num_experts}), EP_PRED_MAX_B ({EP_PRED_MAX_B}))"
    );
    let ids_c = topk_ids.contiguous()?;
    let k = ids_c.elem_count();
    let scores_c = scores.map(|s| s.to_dtype(DType::F32)?.contiguous()).transpose()?;
    if let Some(c) = &scores_c {
        refuse_if!(
            c.elem_count() != num_experts,
            "metal_ep_publish: scores elems {} != num_experts {num_experts}",
            c.elem_count()
        );
    }
    let xs_c = xs.map(Tensor::contiguous).transpose()?;
    refuse_if!(xs_c.as_ref().is_some_and(|c| c.dtype() != DType::F32), "metal_ep_publish: tap source must be F32");
    let ids = Metal::of(&ids_c, "metal_ep_publish: topk_ids")?;
    let p = pipeline(&ids.dev, Family::Moe, "oscar_ep_publish")?;
    // A disarmed half binds the ids, never read at width 0.
    let sm = Metal::of(scores_c.as_ref().unwrap_or(&ids_c), "metal_ep_publish: tensors")?;
    let xm = Metal::of(xs_c.as_ref().unwrap_or(&ids_c), "metal_ep_publish: tensors")?;
    let tg = 256.min(p.max_total_threads_per_threadgroup() as usize);
    refuse_if!(k > tg, "metal_ep_publish: {k} ids exceed the threadgroup ({tg})");
    let look_b = if scores_c.is_some() { look_b as i64 } else { 0 };
    let tap_words = xs_c.as_ref().map_or(0, |c| c.elem_count() as i64);
    launch(&ids.dev, &p, "oscar-ep-publish", [1, 1, 1], [tg, 1, 1], |e| {
        set_params!(
            e,
            (
                &ids,
                &sm,
                &xm,
                mailbox,
                k as i64,
                ids_word_offset as i64,
                ids_gen_word,
                num_experts as i64,
                look_b,
                pred_word_offset as i64,
                pred_gen_word,
                tap_words,
                tap_word_offset as i64,
                tap_gen_word
            )
        );
        Ok(())
    })?;
    Ok([Some(ids_c), scores_c, xs_c].into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{host, sine};

    /// The host's softmax over each row of `ncols`, its top-k renormalised: `(weights, indices)`.
    fn host_topk_softmax(logits: &[f32], ncols: usize, topk: usize) -> (Vec<f32>, Vec<u32>) {
        let mut out = (Vec::new(), Vec::new());
        for row in logits.chunks(ncols) {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
            let denom: f32 = exps.iter().sum();
            let smax: Vec<f32> = exps.iter().map(|e| e / denom).collect();
            let mut idx: Vec<usize> = (0..ncols).collect();
            idx.sort_by(|&a, &b| smax[b].partial_cmp(&smax[a]).unwrap_or(std::cmp::Ordering::Equal));
            let kept: f32 = idx[..topk].iter().map(|&i| smax[i]).sum();
            out.0.extend(idx[..topk].iter().map(|&i| smax[i] / kept));
            out.1.extend(idx[..topk].iter().map(|&i| i as u32));
        }
        out
    }

    /// Weights within 1e-5 of the host's; every index where the deterministic logits hold no near-tie, the top-1 where
    /// negative logits and an outlier exercise the max-shift; a decode's single row as a prefill's 73. Then the biased
    /// sigmoid: the host's selection exactly (`sigmoid(logit) + bias`, as candle's chain computes it) and its weights
    /// within rounding, the kernel's exp precise and its K-sum a simd one, at GLM-5.3-Flash's shape and K3's.
    #[test]
    fn topk_softmax_and_biased_sigmoid_match_the_host() -> Result<()> {
        let device = oscar_offload::pools::test_device()?;
        let (ncols, topk) = (128usize, 8usize);
        let mut spiked: Vec<f32> = (0..4 * ncols).map(|i| -((i as f32) * 0.013 + 0.7).cos() * 5.0).collect();
        for r in 0..4 {
            spiked[r * ncols + (r * 17) % ncols] = 12.5;
        }
        // (logits, rows, every index compared, the tolerance of each row's sum to 1)
        for (logits, rows, every, sum_tol) in [
            (sine(73 * ncols, 0.0017, 0.31), 73usize, true, Some(1e-4f32)),
            (spiked, 4, false, None),
            (sine(ncols, 0.041, 0.13), 1, true, Some(1e-5)),
        ] {
            let (w, i) = metal_topk_softmax(&Tensor::from_vec(logits.clone(), (rows, ncols), &device)?, topk, true)?;
            let (w, i): (Vec<f32>, Vec<u32>) = (host(&w)?, i.flatten_all()?.to_vec1()?);
            let (want_w, want_i) = host_topk_softmax(&logits, ncols, topk);
            for r in 0..rows {
                let slots = r * topk..(r + 1) * topk;
                let compared = if every { slots.clone() } else { r * topk..r * topk + 1 };
                assert_eq!(i[compared.clone()], want_i[compared], "row {r}: indices");
                for s in slots.clone() {
                    assert!((w[s] - want_w[s]).abs() < 1e-5, "row {r} slot {s}: {} vs {}", w[s], want_w[s]);
                }
                let sum: f32 = w[slots].iter().sum();
                assert!(sum_tol.is_none_or(|tol| (sum - 1.0).abs() < tol), "row {r}: the weights sum to {sum}");
            }
        }
        let rows = 64usize;
        for (n, k) in [(288usize, 8usize), (896, 16)] {
            let logits: Vec<f32> = (0..rows * n).map(|i| (((i * 7919) % 10007) as f32 - 5003.0) / 900.0).collect();
            let bias: Vec<f32> = (0..n).map(|i| (((i * 31) % 97) as f32 - 48.0) / 300.0).collect();
            let logits_t = Tensor::from_vec(logits.clone(), (rows, n), &device)?;
            let bias_t = Tensor::new(&bias[..], &device)?;
            let (w, ids) = metal_topk_biased_sigmoid(&logits_t, &bias_t, k, true, 2.5, 1e-20)?;
            let (w, ids): (Vec<f32>, Vec<u32>) = (host(&w)?, ids.flatten_all()?.to_vec1()?);
            for (r, row) in logits.chunks(n).enumerate() {
                let s: Vec<f32> = row.iter().map(|&x| ((-x).exp() + 1.0).recip()).collect();
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|&a, &b| (s[b] + bias[b]).partial_cmp(&(s[a] + bias[a])).expect("finite scores"));
                let denom = idx[..k].iter().map(|&e| s[e]).sum::<f32>() + 1e-20;
                for (slot, &e) in idx[..k].iter().enumerate() {
                    let (got, want) = (w[r * k + slot], s[e] / denom * 2.5);
                    assert_eq!(ids[r * k + slot] as usize, e, "n={n} k={k} row {r}: the selection must be the host's");
                    assert!(((got - want) / want).abs() < 1e-5, "n={n} k={k} row {r}: weight {got} vs {want}");
                }
            }
        }
        Ok(())
    }

    /// The true block (top-k ids, gen word), then the pred block (look_b ids, look_b score bits, gen word): descending,
    /// lowest index first on ties; 896 experts (K3) as 128 do.
    #[test]
    fn ep_publish_writes_ids_and_selects_topb_at_any_expert_width() -> Result<()> {
        let device = oscar_offload::pools::test_device()?;
        let (topk, ids_gen, gen_word) = (10usize, 0xFFFF_1B02u32, 0xFFFF_2A01u32);
        for (ncols, look_b) in [(128usize, 8usize), (896, 16), (896, 1)] {
            let scores_v = sine(ncols, 0.0913, 0.47);
            let ids_v: Vec<u32> = (0..topk as u32).map(|i| (i * 37 + 5) % ncols as u32).collect();
            let (scores, ids) = (Tensor::new(&scores_v[..], &device)?, Tensor::new(&ids_v[..], &device)?);
            let pred = topk + 1;
            let mailbox = Tensor::zeros(pred + 2 * look_b + 1, DType::U32, &device)?;
            let buffer = Metal::of(&mailbox, "mailbox")?.buf;
            let _keep =
                metal_ep_publish(&ids, 0, ids_gen, Some(&scores), None, &buffer, ncols, look_b, pred, gen_word, 0, 0)?;
            device.synchronize()?;
            let out = mailbox.to_vec1::<u32>()?;
            assert_eq!(&out[..topk], &ids_v[..], "ncols {ncols}: the true ids");
            assert_eq!(out[topk], ids_gen, "the true gen word");
            let mut idx: Vec<usize> = (0..ncols).collect();
            let by_score = |a: &usize, b: &usize| scores_v[*b].partial_cmp(&scores_v[*a]).expect("finite scores");
            idx.sort_by(|a, b| by_score(a, b).then(a.cmp(b)));
            for (b, &want) in idx[..look_b].iter().enumerate() {
                let what = format!("ncols {ncols} look_b {look_b}: at {b}");
                assert_eq!(out[pred + b] as usize, want, "{what}: the id");
                assert_eq!(out[pred + look_b + b], scores_v[want].to_bits(), "{what}: the score's bits");
            }
            assert_eq!(out[pred + 2 * look_b], gen_word, "the pred gen word");
        }
        Ok(())
    }
}
