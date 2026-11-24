// The pipeline drafts K tokens ahead on its own thread, its routings the prefetch hints; the engine's steps stay exact.

mod overlap;
mod round;
pub mod sampler;
pub(crate) mod thread;

/// The drafter's counters, read by the eval's lines; `turn_*` follow a session from its send to its first result.
#[derive(Default)]
pub struct DraftCounters {
    pub sessions: AtomicU64,
    pub ground_deferred: AtomicU64,
    pub groundings: AtomicU64,
    pub handoff_us: AtomicU64,
    pub handoff_sync_us: AtomicU64,
    pub handoff_pooled: AtomicU64,
    pub diverges: AtomicU64,
    pub late: AtomicU64,
    pub depth_sum: AtomicU64,
    pub skip_handoff: AtomicU64,
    pub fwds: AtomicU64,
    pub abort_stale: AtomicU64,
    pub ground_us: AtomicU64,
    pub fwd_us: AtomicU64,
    pub enc_us: AtomicU64,
    pub torn: AtomicU64,
    pub turn_runs: AtomicU64,
    pub turn_wake_us: AtomicU64,
    pub turn_fwd1_us: AtomicU64,
    pub turn_first: AtomicU64,
    pub turn_total_us: AtomicU64,
    pub turn_total_max_us: AtomicU64,
    pub turn_ewma_us: AtomicU64,
    pub boundaries: AtomicU64,
    /// Of `boundaries`, those with the drafter off.
    pub off: AtomicU64,
    pub verdicts: AtomicU64,
    pub switches: AtomicU64,
}

pub static DRAFT: std::sync::LazyLock<DraftCounters> = std::sync::LazyLock::new(DraftCounters::default);

/// Before the plan is solved, wherever armed, so the plan books its region; a launch sealing every layer books none.
pub fn spawn_for_record() {
    if crate::config::drafter_on() {
        let _ = overlap::spawn_once();
    }
}

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use mistralrs_core::kv_cache::EitherCache;
use mistralrs_core::sequence::{Sequence, SequenceRecognizer};
use mistralrs_core::Pipeline;

/// The round as the engine's step; ineligible batches take the pipeline's.
pub fn register_round() {
    mistralrs_core::set_step_round(Arc::new(round::DraftRound));
}

pub use overlap::{drafting, in_session, note_pool_moved, quiesce, set_held};

/// Whether this batch takes the round; others take `Pipeline::step`.
fn eligible(this: &mut dyn Pipeline, seqs: &mut [&mut Sequence], return_raw_logits: bool) -> bool {
    if !drafting() {
        return false;
    }
    let reason = ineligible_reason(this, seqs, return_raw_logits);
    if let Some(reason) = reason {
        static WARNED: AtomicU64 = AtomicU64::new(0);
        if WARNED.fetch_add(1, Ordering::Relaxed) == 0 {
            crate::detail!(target: "oscar::drafter", "configured (K={}) but ineligible: {reason}", crate::config::params().draft_k);
        }
        return false;
    }
    true
}

fn ineligible_reason(
    this: &mut dyn Pipeline, seqs: &mut [&mut Sequence], return_raw_logits: bool,
) -> Option<&'static str> {
    if return_raw_logits {
        return Some("return_raw_logits");
    }
    if seqs.len() != 1 {
        return Some("batch>1");
    }
    let meta = this.get_metadata();
    assert!(!meta.no_kv_cache && !meta.is_xlora, "oscar builds neither a cacheless nor an X-LoRA pipeline");
    if meta.tok_env().is_none() {
        return Some("no tok_env");
    }
    match this.cache() {
        EitherCache::Normal(_) => {}
        // Hybrid is fine: the drafter never rolls the engine's cache back.
        EitherCache::Hybrid(_) => {}
        _ => return Some("unsupported cache"),
    }
    let seq = &mut *seqs[0];
    // Grammars are out, as drafting would consume recognizer state; sampling is in, a rejected draft costing its hint.
    if !seq.is_running() {
        return Some("seq not running");
    }
    if !matches!(seq.recognizer, SequenceRecognizer::None) {
        return Some("grammar recognizer");
    }
    if seq.get_toks().is_empty() {
        return Some("empty seq");
    }
    None
}
