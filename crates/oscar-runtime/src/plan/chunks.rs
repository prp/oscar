//! Prefill's chunks at a recorded width with its region, set once at the load's tail: neither follows the budget.

use std::sync::{Arc, OnceLock};

use mistralrs_core::{set_prefill_chunk_policy, PrefillChunkPolicy};

use super::{chosen_width, PREFILL_CHUNK_MIN_TOKENS};

/// The fewest chunks of at most `widest`, as equal as whole tokens make them, so no runt is left.
fn split(widest: usize, total: usize) -> Vec<usize> {
    if total <= widest {
        return vec![total];
    }
    let k = total.div_ceil(widest);
    // Every chunk but the last holds the floor, whatever the prompt.
    let w = total.div_ceil(k).max(PREFILL_CHUNK_MIN_TOKENS as usize);
    let mut widths = vec![w; k - 1];
    widths.push(total - w * (k - 1));
    widths
}

/// Interior chunk ends in order (the prompt step forwards the last chunk).
fn ends_for(offset: usize, end: usize, widest: usize) -> Vec<usize> {
    let widths = split(widest, end.saturating_sub(offset));
    widths
        .iter()
        .scan(offset, |at, w| {
            *at += w;
            Some(*at)
        })
        .take_while(|at| *at < end)
        .collect()
}

pub struct EqualChunks;

static CHUNK_WIDTH: OnceLock<usize> = OnceLock::new();

impl PrefillChunkPolicy for EqualChunks {
    fn chunk_ends(&self, offset: usize, end: usize) -> Vec<usize> {
        ends_for(offset, end, *CHUNK_WIDTH.get().expect("the chunk is set before the policy"))
    }
}

/// Once, at the load's tail: the plan's widest prefill width, the eval's override outranking it, and its region.
pub fn install() {
    // A load with no record refused already (`census::run_at_load`), so `None` is no width the plan solves at.
    let Some(recorded) = chosen_width() else {
        panic!(
            "memory plan REFUSED: no recorded prefill width this machine hosts - prefill under \
             {PREFILL_CHUNK_MIN_TOKENS} tokens re-reads every routed expert per token"
        );
    };
    let (live, region) = match crate::config::params().misc.prefill_chunk {
        Some((w, region)) => (w, region.or_else(|| crate::load::census::region_at(w).map(|b| b as usize))),
        None => (recorded.width, None),
    };
    // The record's replay at the chunk's width, drawn here so a request's chunk and its region are one decision.
    let region = region.unwrap_or(recorded.region as usize);

    assert!(CHUNK_WIDTH.set(live).is_ok(), "the chunk is installed once, at the load's tail");
    if let Some(pools) = oscar_offload::pools::pools() {
        pools
            .plan_region(region)
            .unwrap_or_else(|e| panic!("memory plan REFUSED: working region cannot follow plan - {e}"));
    }
    set_prefill_chunk_policy(Arc::new(EqualChunks));
}

#[cfg(test)]
mod tests {
    use super::{ends_for, split};

    /// It can fail: a full-width cut keeping the remainder leaves a runt; a width rounded up runs a chunk over.
    #[test]
    fn splits_into_the_fewest_equal_chunks_none_over_the_width() {
        for widest in [5397usize, 4331] {
            for total in [1usize, 255, 4542, 4543, 5397, 5398, 10822, 22704, 131072, 1_000_000] {
                let widths = split(widest, total);
                assert_eq!(widths.iter().sum::<usize>(), total, "split must be exact");
                assert_eq!(widths.len(), total.div_ceil(widest), "fewest chunks: {widths:?}");
                let (lo, hi) = (*widths.iter().min().unwrap(), *widths.iter().max().unwrap());
                assert!(hi - lo < widths.len(), "total {total} leaves a runt: {widths:?}");
                assert!(hi <= widest, "total {total}: a {hi}-token chunk over the {widest}-token recorded width");
            }
        }
        // The widths the banked 35B rows' two long prompts run at.
        assert_eq!(split(5397, 10822), vec![3608, 3608, 3606]);
        assert_eq!(split(5397, 22704), vec![4541, 4541, 4541, 4541, 4540]);
        assert_eq!(ends_for(0, 11149, 1 << 20), Vec::<usize>::new());
        assert_eq!(ends_for(100, 11249, 5397), vec![3817, 7534]);
    }
}
