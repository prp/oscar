//! The KV and sequence-state charges: the guaranteed sequence, its blocks, the snapshot ladder and the store's staging.

use super::*;

pub fn arm_draft_reserve(reserve_tokens: u64, block_size: usize) {
    mistralrs_core::paged_attention::set_drafter_reserved_blocks(reserve_tokens as usize / block_size);
}

/// The KV block size in tokens (`OSCAR_PAGED_BLOCK`).
pub fn paged_block_size() -> u64 {
    crate::config::params().paged_block
}

pub fn padded_block_bytes(block_bytes: u64) -> u64 {
    block_bytes.next_multiple_of(16384)
}

/// The blocks `bytes` holds, counted over its whole mebibytes.
pub fn kv_blocks_of(bytes: u64, per_token: u64, block_size: u64) -> u64 {
    const MIB: u64 = 1 << 20;
    (bytes / MIB) * MIB / per_token.max(1) / block_size.max(1)
}

/// The block pool's launch count: the guaranteed sequence, a chunk's headroom at `width` and the draft reserve.
pub(crate) fn charge_blocks(stats: &ModelLayoutStats, charge: KvCharge, width: Option<usize>) -> u64 {
    let per_token = stats.kv_bytes_per_token;
    let tokens =
        charge.total_tokens().saturating_add(charge.headroom(width)).saturating_add(charge.draft_reserve_tokens);
    kv_blocks_of(tokens.saturating_mul(per_token), per_token, paged_block_size())
}

/// The tip (the next append), the turn behind (a re-render rewriting the reply) and a deep anchor, each a full copy.
pub(crate) const SNAPSHOT_RUNGS_PER_CONVERSATION: u64 = 3;

/// X in + Y out: the charge within the header's bound, the context the census records every width at.
pub(crate) fn guarantee_of(charge: KvCharge, context_bound: u64) -> (u64, u64) {
    let total = match context_bound {
        0 => charge.total_tokens(),
        bound => charge.total_tokens().min(bound),
    };
    let y = charge.max_output_tokens.min(total);
    (total - y, y)
}

/// A turn's records until the writer drains them: the guarantee's KV up to the cap, at least a chunk and three states.
pub fn store_staging_bytes(stats: &ModelLayoutStats, charge: KvCharge, chunk_tokens: u64) -> u64 {
    let armed = crate::kv_store::configured();
    if armed && !charge.bench {
        let turn = stats.kv_bytes_per_token.saturating_mul(chunk_tokens).saturating_add(3 * sequence_state_unit(stats));
        stats.kv_bytes_per_token.saturating_mul(charge.total_tokens()).min(crate::kv_store::STAGING_CAP_BYTES).max(turn)
    } else {
        0
    }
}

/// The scheduler's cap on every entry: serve admits serially and queues the rest; chat and the bench run one at a time.
pub const LIVE_SEQUENCES: u64 = 1;

/// As the pool holds it: the census's first take once recorded, else the header's bound; zero with no recurrent layer.
pub fn sequence_state_unit(stats: &ModelLayoutStats) -> u64 {
    if stats.sequence_state_bound == 0 {
        return 0;
    }
    match crate::load::census::census().map_or(0, |c| c.state_unit) {
        0 => stats.sequence_state_bound,
        recorded => recorded,
    }
}

/// (every state, the retained of them): each live sequence's state, prompt-boundary snapshot and three rungs, the
/// bench charge's none, and the drafter's hand-off while its memory is booked; nothing yields to the budget.
pub(crate) fn states(charge: KvCharge) -> (u64, u64) {
    let retained = if charge.bench { 0 } else { LIVE_SEQUENCES * SNAPSHOT_RUNGS_PER_CONVERSATION };
    (LIVE_SEQUENCES * 2 + u64::from(charge.drafter_booked) + retained, retained)
}

/// Every id a state may have, the hand-off's kept wherever a drafter is armed: host table bytes, a segment mapped when
/// taken.
pub const STATE_SLOTS_CEILING: usize = (LIVE_SEQUENCES * 2 + 1 + SNAPSHOT_RUNGS_PER_CONVERSATION) as usize;

pub fn kv_block_object_bytes(stats: &ModelLayoutStats) -> u64 {
    padded_block_bytes(stats.kv_bytes_per_token.saturating_mul(paged_block_size()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::KV_CHARGE_BENCH;
    use crate::plan::tests::routed;

    /// 397B-shaped: 15 of 60 layers bear KV, 45 hold a 32×128×128 F32 state and an 8192×4 BF16 conv tail.
    fn hybrid() -> ModelLayoutStats {
        let state = 45 * (32 * 128 * 128 * 4 + 8192 * 4 * 2);
        ModelLayoutStats { kv_bearing_layers: 15, sequence_state_bound: state, ..routed(60, 128, 1 << 20) }
    }

    /// The interactive charge guarantees one sequence plus headroom; a resident plan's remainder is the KV pool's line.
    /// The states are the live pair, a booked drafter's hand-off and, on the bench charge alone, one conversation's
    /// ladder however many the pool holds; no recurrent state, nothing.
    #[test]
    fn the_charge_guarantees_one_sequence_plus_headroom_and_the_states_are_the_pair_the_hand_off_and_a_ladder() {
        let c = KvCharge::for_interactive(6144, 2048, 0);
        assert_eq!((c.total_tokens(), c.bench), (8192, false), "an interactive charge is never the bench charge");
        let stats = routed(60, 128, 1 << 20);
        let per_token = stats.kv_bytes_per_token;
        let (one_seq, headroom) = (c.total_tokens(), c.headroom(None));
        let blocks = kv_blocks_of(per_token * (one_seq + headroom), per_token, paged_block_size());
        assert_eq!(charge_blocks(&stats, c, None), blocks, "one sequence plus headroom, nothing else");
        // The headroom is one chunk (before a record, the largest prompt), never another copy.
        assert_eq!(headroom, c.max_input_tokens);
        assert_eq!((c.headroom(Some(4096)), c.headroom(Some(8192))), (4096, 6144));
        assert!(headroom < one_seq, "headroom is a chunk, not another copy: {headroom} vs {one_seq}");
        assert_eq!(charge_blocks(&stats, KV_CHARGE_BENCH, None) * paged_block_size(), 4096, "the bench charge, 4096x1");
        // An offloaded pool: the charge and its headroom in block objects.
        let offloaded = routed(60, 128, 1 << 30);
        let plan = PlanInputs::live(54.0, c).solve(&offloaded).expect("plans");
        let (blocks, bytes) = (charge_blocks(&offloaded, c, None), plan.kv_blocks * kv_block_object_bytes(&offloaded));
        assert_eq!((plan.offloading, plan.kv_blocks, plan.kv_cache_bytes), (true, blocks, bytes));
        assert_eq!(plan.guarantee(), (6144, 2048), "the pool is sized from the charge, so the guarantee is the charge");
        // The resident remainder.
        let stats = ModelLayoutStats { kv_bearing_layers: 60, context_length: 262_144, ..hybrid() };
        let charge = KvCharge::for_interactive(8192, 8192, 0);
        let plan = PlanInputs::live(54.0, charge).solve(&stats).expect("plans");
        let (object, budget) = (kv_block_object_bytes(&stats), (54.0 * GB) as u64);
        // The remainder is what the budget leaves past everything else: the slack under it is under a block.
        assert!(!plan.offloading && budget - plan.total_allocated_bytes() < object, "the total is the budget");
        assert_eq!(plan.kv_cache_bytes, plan.kv_blocks * object);
        assert!(plan.kv_blocks * paged_block_size() > 1_000_000, "the pool holds the remainder, not the charge");
        assert_eq!(plan.kv_charge.total_tokens(), 16_384, "the charge stays what was asked");
        // The guarantee is the charge whatever the pool holds, so a session past it meets the window's refusal;
        // a charge past the header's bound promises the bound.
        assert_eq!(plan.guarantee(), (8192, 8192));
        assert_eq!(guarantee_of(KvCharge::for_interactive(262_144, 8192, 0), 262_144), (262_144 - 8192, 8192));
        assert_eq!(guarantee_of(KvCharge::for_interactive(262_144, 8192, 0), 0), (262_144, 8192));
        // The bench charge: the same line, the charge kept as provenance, the ladder's bytes the pool's.
        let bench = PlanInputs::live(54.0, KV_CHARGE_BENCH).solve(&stats).expect("plans");
        assert!(bench.kv_charge.bench);
        let snapshots = plan.paged_snapshot_entries * sequence_state_unit(&stats) / object;
        assert!(bench.kv_blocks - (plan.kv_blocks + snapshots) <= 1, "the bench pool is the ladder's blocks more");
        // A budget that cannot seal the stack plans at the charge.
        let tight = PlanInputs::live(1.65, charge).solve(&stats).expect("plans");
        assert!(tight.offloading, "a 1.65 GB budget offloads this shape");
        assert_eq!(tight.guarantee(), (8192, 8192));
        // The states.
        let mut stats = hybrid();
        let interactive = KvCharge::for_interactive(24576, 8192, 0);
        let drafted = KvCharge { draft_reserve_tokens: 64, drafter_booked: true, ..KV_CHARGE_BENCH };
        let rungs = SNAPSHOT_RUNGS_PER_CONVERSATION;
        let each = [states(interactive), states(KV_CHARGE_BENCH), states(drafted)];
        assert_eq!(each, [(2 + rungs, rungs), (2, 0), (3, 0)]);
        let unit = sequence_state_unit(&stats);
        for charge in [interactive, KV_CHARGE_BENCH] {
            let plan = PlanInputs::live(54.0, charge).solve(&stats).expect("plans");
            assert_eq!((plan.sequence_state_slots, plan.paged_snapshot_entries), states(charge));
            assert_eq!((plan.sequence_state_unit, plan.sequence_state_bytes), (unit, states(charge).0 * unit));
            assert!(plan.locked_bytes() >= plan.sequence_state_bytes, "the pool's states are wired");
        }
        stats.sequence_state_bound = 0;
        let plan = PlanInputs::live(54.0, interactive).solve(&stats).expect("plans");
        assert_eq!((plan.sequence_state_slots, plan.sequence_state_bytes, plan.paged_snapshot_entries), (0, 0, 0));
    }
}
