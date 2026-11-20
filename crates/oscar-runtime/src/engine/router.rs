//! Router helpers: the arch's selection scores, a batch's demand ranking and the routed-id dedup.

use std::collections::HashSet;

use candle_core::quantized::QMatMul;
use candle_core::{Result, Tensor};

/// The arch's selection space, stated once: a wrong one fails silently, as a different expert set. A prediction's gate
/// takes the cheapest kernel: its float order need not be the engine's.
pub(crate) fn selection_scores(gate: &QMatMul, bias: Option<&Tensor>, xs: &Tensor) -> Result<Tensor> {
    let raw = oscar_models::gate_matvec_or_forward(gate, xs)?;
    match bias {
        Some(b) => candle_nn::ops::sigmoid(&raw)?.broadcast_add(b),
        None => Ok(raw),
    }
}

/// The `keep` experts most tokens chose, the work each does per extent fetched; ties keep the lower id.
pub(crate) fn rank_by_token_demand(flat_ids: &[u32], num_experts: usize, keep: usize) -> Vec<u16> {
    let mut count = vec![0u32; num_experts];
    for &e in flat_ids {
        if let Some(c) = count.get_mut(e as usize) {
            *c += 1;
        }
    }
    let mut ids: Vec<u16> = (0..num_experts).filter(|&e| count[e] > 0).map(|e| e as u16).collect();
    // `sort_by_key` is stable, so equal counts stay in ascending id order.
    ids.sort_by_key(|&e| std::cmp::Reverse(count[e as usize]));
    ids.truncate(keep);
    ids
}

/// In first-seen order; expert ids fit `u16`.
pub(crate) fn dedup_u32_to_u16(flat_ids: &[u32], cap_hint: usize) -> Vec<u16> {
    let mut seen: HashSet<u32> = HashSet::with_capacity(cap_hint);
    let mut out: Vec<u16> = Vec::with_capacity(cap_hint);
    for &e in flat_ids {
        if seen.insert(e) {
            out.push(e as u16);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::rank_by_token_demand;

    /// The experts most tokens chose first, ties by the lower id, the head kept; an id out of range is a bad gate's.
    #[test]
    fn experts_rank_by_the_tokens_choosing_them() {
        let rows: [(&[u32], usize, usize, &[u16]); 4] = [
            (&[1, 7, 7, 3, 7, 3], 8, 8, &[7, 3, 1]), // three tokens at top-2: 7 chosen by all, 3 by two, 1 by one
            (&[5, 2, 5, 2, 9], 16, 2, &[2, 5]),
            (&[5, 2, 5, 2, 9], 16, 0, &[]),
            (&[99, 2, 2], 4, 4, &[2]),
        ];
        for (ids, experts, keep, want) in rows {
            assert_eq!(rank_by_token_demand(ids, experts, keep), want, "{ids:?}, {experts} experts, keep {keep}");
        }
    }
}
