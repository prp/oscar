//! Approximation routing (`OSCAR_APPROX_ROUTING=θ`); its math is fixed, as every supported θ was screened at it.

use std::sync::atomic::{AtomicU64, Ordering};

/// Selections an armed θ pass examined (k per call), those replaced, and cold ones kept exact for want of a resident
/// candidate; (token, layer) calls with a replacement; dispatches whose θ block missed the bounded poll.
pub static SELECTIONS: AtomicU64 = AtomicU64::new(0);
pub static REPLACED: AtomicU64 = AtomicU64::new(0);
pub static NO_CANDIDATE: AtomicU64 = AtomicU64::new(0);
pub static CALLS_TOUCHED: AtomicU64 = AtomicU64::new(0);
pub static GEN_SKIPS: AtomicU64 = AtomicU64::new(0);

/// Per routed position: the expert executed and its renormalised weight.
pub(crate) struct ThetaSubstitution {
    pub exec: Vec<u16>,
    pub weights: Vec<f32>,
}

/// `is_resident` true only where every projection bank holds the expert; `None` leaves the engine's routing bitwise.
pub(crate) fn theta_substitute(
    theta: f32, ids: &[u16], weights: &[f32], scores: &[f32], is_resident: &dyn Fn(u16) -> bool,
) -> Option<ThetaSubstitution> {
    let k = ids.len();
    if k == 0 || weights.len() != k {
        return None;
    }
    let tot: f32 = weights.iter().sum();
    if !(tot > 0.0) {
        return None;
    }
    // Heaviest first, so the best candidates go to the heaviest selections.
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| weights[b].partial_cmp(&weights[a]).unwrap_or(std::cmp::Ordering::Equal));

    // Built lazily: the resident non-selected experts, best first.
    let mut candidates: Option<Vec<u16>> = None;
    let mut cand_next: usize = 0;

    let mut exec: Vec<u16> = ids.to_vec();
    let mut replaced = 0usize;
    for &pos in &order {
        let e = ids[pos];
        if weights[pos] / tot >= theta {
            continue;
        }
        if is_resident(e) {
            continue; // no fetch to save
        }
        let cands = candidates.get_or_insert_with(|| {
            let mut v: Vec<u16> = (0..scores.len() as u16).filter(|c| !ids.contains(c) && is_resident(*c)).collect();
            v.sort_by(|&a, &b| {
                scores[b as usize].partial_cmp(&scores[a as usize]).unwrap_or(std::cmp::Ordering::Equal)
            });
            v
        });
        if cand_next < cands.len() {
            exec[pos] = cands[cand_next];
            cand_next += 1;
            replaced += 1;
        } else {
            NO_CANDIDATE.fetch_add(1, Ordering::Relaxed);
        }
    }
    SELECTIONS.fetch_add(k as u64, Ordering::Relaxed);
    if replaced == 0 {
        return None;
    }
    REPLACED.fetch_add(replaced as u64, Ordering::Relaxed);
    CALLS_TOUCHED.fetch_add(1, Ordering::Relaxed);

    // Renormalise in routing space: softmax over the final set's raw logits.
    let mut new_w: Vec<f32> = exec.iter().map(|&e| scores[e as usize]).collect();
    let m = new_w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for w in new_w.iter_mut() {
        *w = (*w - m).exp();
        sum += *w;
    }
    for w in new_w.iter_mut() {
        *w /= sum;
    }
    Some(ThetaSubstitution { exec, weights: new_w })
}

/// θ is calibrated for the softmax + `norm_topk_prob` family; any other router armed with it fails at first decode.
pub(crate) fn assert_softmax_family(has_selection_bias: bool, norm_topk_prob: bool) {
    if has_selection_bias || !norm_topk_prob {
        panic!(
            "OSCAR_APPROX_ROUTING: theta-form supports softmax(norm_topk_prob) routers only; this model routes with \
             selection bias / unnormalised weights (margin-form router)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// θ 0.10: expert 5 is under it but resident, kept; 6 is under it and cold, replaced by 1, the best resident.
    #[test]
    fn theta_replaces_a_cold_selection_below_theta() {
        let scores = [2.0f32, 1.5, 1.0, 0.5, 0.2, 0.1, -0.5, -1.0];
        let weights = [0.6251824f32, 0.22999177, 0.09350768, 0.0513181];
        let resident = |e: u16| [1, 3, 5].contains(&e);
        let sub = theta_substitute(0.10, &[0, 2, 5, 6], &weights, &scores, &resident).expect("one replacement");
        assert_eq!(sub.exec, vec![0, 2, 5, 1]);
        let want = [0.47081451f32, 0.17320298, 0.070419076, 0.28556344];
        assert!(sub.weights.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-6), "{:?} vs {want:?}", sub.weights);
        // Nothing under θ; under θ and cold, but no resident expert outside the set: the routing stays exact.
        assert!(theta_substitute(0.10, &[0, 1], &[0.62, 0.38], &scores[..4], &|_: u16| false).is_none());
        assert!(theta_substitute(0.10, &[0, 3], &[0.95, 0.05], &scores[..4], &|e: u16| e == 0).is_none());
    }
}
