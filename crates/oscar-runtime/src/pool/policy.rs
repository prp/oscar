//! The policy: the transition oracle the walk scores by, the decode pacer's ticks and the stage rules a forward is
//! cut by.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, RwLock};

use super::bank::{LayerExpertKey, Oracle};
use crate::config::EvictionChannels;

/// Rollout depth in tokens.
const HORIZON: u32 = 8;

/// Rollout width per layer per step: wider lets noisy far predictions override recency.
pub const HORIZON_K: usize = 32;

pub struct NextUseOracle {
    block_count: usize,
    num_experts: usize,
    state: RwLock<TransState>,
    /// The reuse-distance prior: a key used `age` ticks ago is expected back `age` from now; `u32::MAX`: never used.
    last_access_tick: Option<Vec<AtomicU32>>,
    /// Next-use distance in tokens per `(layer, expert)`, `u32::MAX` beyond the horizon; rolled once per decode token.
    pred_dist: Vec<AtomicU32>,
    /// Turns "d tokens ahead" into the visit's tick, so `eviction_score` compares absolute deadlines.
    pacer: DecodePacer,
    /// `(window depth, deepest routing)`: where the next roll starts.
    draft_seed: std::sync::Mutex<Option<(u32, Vec<Vec<u16>>)>>,
    /// Keys the last window stamped, un-claimed at the next feed so no claim outlives its evidence.
    draft_claimed: std::sync::Mutex<Vec<u32>>,

    near_claim_tick: Vec<AtomicU32>,
    draft_claim_token: Vec<AtomicU32>,
    channels: EvictionChannels,
}

#[derive(Debug)]
struct TransState {
    /// Transition counts, indexed `(layer * ne + prev) * ne + next`.
    counts: Vec<u32>,
    latest_routing: Vec<Vec<u16>>,
    /// A non-increasing layer starts a new decode token; `usize::MAX` before the first call.
    last_layer: usize,
}

impl NextUseOracle {
    /// For the host line before the oracle exists: transition counts, four per-key tables and the latest routings.
    pub fn host_bytes(block_count: usize, num_experts: usize, experts_per_token: usize) -> usize {
        use std::mem::size_of;
        let keys = block_count * num_experts;
        keys * num_experts * size_of::<u32>()
            + 4 * keys * size_of::<AtomicU32>()
            + block_count * (size_of::<Vec<u16>>() + experts_per_token * size_of::<u16>())
    }

    pub fn new(pacer: DecodePacer, block_count: usize, num_experts: usize, channels: EvictionChannels) -> Self {
        Self {
            block_count,
            num_experts,
            state: RwLock::new(TransState {
                counts: vec![0u32; block_count * num_experts * num_experts],
                latest_routing: vec![Vec::new(); block_count],
                last_layer: usize::MAX,
            }),
            last_access_tick: None,
            pred_dist: (0..block_count * num_experts).map(|_| AtomicU32::new(u32::MAX)).collect(),
            near_claim_tick: (0..block_count * num_experts).map(|_| AtomicU32::new(u32::MAX)).collect(),
            draft_claim_token: (0..block_count * num_experts).map(|_| AtomicU32::new(u32::MAX)).collect(),
            channels,
            pacer,
            draft_seed: std::sync::Mutex::new(None),
            draft_claimed: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// From the last complete token, `k` experts a layer a step, each key's soonest predicted use into `pred_dist`.
    fn roll_horizon(&self, st: &TransState, horizon: u32, k: usize) {
        for cell in &self.pred_dist {
            cell.store(u32::MAX, Ordering::Relaxed);
        }
        // `cur[layer]`: the set the next step looks up (observed at step 0).
        let mut cur: Vec<Vec<u16>> = st.latest_routing.clone();
        // From the drafted frontier: the rollout and draft legs disjoint by construction.
        let mut start_h = 0u32;
        if let Some((depth, frontier)) = self.draft_seed.lock().expect("NextUseOracle poisoned").take() {
            for (layer, experts) in frontier.into_iter().enumerate() {
                if layer < cur.len() && !experts.is_empty() {
                    cur[layer] = experts;
                }
            }
            start_h = depth;
        }
        let (mut score, mut scratch) = (vec![0u32; self.num_experts], Vec::with_capacity(self.num_experts));
        for h in start_h..start_h + horizon {
            for layer in 0..self.block_count {
                if cur[layer].is_empty() {
                    continue;
                }
                self.step(&st.counts, layer, &cur[layer], k, &mut score, &mut scratch);
                cur[layer].clear();
                for &(_, e) in &scratch {
                    self.pred_dist[layer * self.num_experts + e as usize].fetch_min(h, Ordering::Relaxed);
                    cur[layer].push(e);
                }
            }
        }
    }

    /// The `k` experts the transitions out of `set` score highest, positive only, ties by id, into `scratch`.
    fn step(
        &self, counts: &[u32], layer: usize, set: &[u16], k: usize, score: &mut [u32], scratch: &mut Vec<(u32, u16)>,
    ) {
        score.fill(0);
        for &a in set {
            let base = self.row_base(layer, a as usize);
            for (s, &c) in score.iter_mut().zip(&counts[base..base + self.num_experts]) {
                *s = s.saturating_add(c);
            }
        }
        scratch.clear();
        scratch.extend(score.iter().enumerate().filter(|(_, &s)| s > 0).map(|(e, &s)| (s, e as u16)));
        rank_top_k(scratch, k);
    }

    #[inline]
    fn row_base(&self, layer: usize, prev: usize) -> usize {
        (layer * self.num_experts + prev) * self.num_experts
    }

    /// Arms the reuse-distance prior (`OSCAR_ORACLE=recency`).
    pub fn with_recency_prior(mut self) -> Self {
        self.last_access_tick =
            Some((0..self.block_count * self.num_experts).map(|_| AtomicU32::new(u32::MAX)).collect());
        self
    }

    /// `OSCAR_ORACLE=lru`: the reuse-distance prior alone, every feed off, so a key ranks by its last use.
    pub fn recency_alone(self) -> Self {
        Self { channels: EvictionChannels::default(), ..self }.with_recency_prior()
    }

    /// The top-`n` next-token experts, the rollout's first step, for offline quality measurement.
    pub fn predict_layer(&self, layer: usize, n: usize) -> Vec<u16> {
        let st = self.state.read().expect("NextUseOracle poisoned");
        let Some(set) = st.latest_routing.get(layer).filter(|s| !s.is_empty()) else { return Vec::new() };
        let (mut score, mut scratch) = (vec![0u32; self.num_experts], Vec::new());
        self.step(&st.counts, layer, set, n, &mut score, &mut scratch);
        scratch.into_iter().map(|(_, e)| e).collect()
    }
}

impl Oracle for NextUseOracle {
    fn eviction_score(&self, key: LayerExpertKey, current_tick: u32) -> u32 {
        let idx = key.0 as usize * self.num_experts + key.1 as usize;
        if idx >= self.pred_dist.len() {
            return u32::MAX;
        }
        // The first leg with a live claim decides, near → draft → rollout.
        if self.channels.lookahead {
            let c = self.near_claim_tick[idx].load(Ordering::Relaxed);
            if c != u32::MAX && c >= current_tick {
                // A past claim expires by the comparison.
                return c;
            }
        }
        if self.channels.draft {
            let t = self.draft_claim_token[idx].load(Ordering::Relaxed);
            let cur_tok = current_tick / self.pacer.ticks_per_token().max(1);
            // Liveness is inclusive: evictions run mid-forward, so a claim survives its own use forward.
            if t != u32::MAX && t >= cur_tok {
                return self.pacer.visit_tick(key.0 as usize, current_tick, t - cur_tok);
            }
        }
        // The rollout's visit, `u32::MAX` beyond the horizon, `min`ed with the reuse-distance prior when armed.
        let d = if self.channels.rollout { self.pred_dist[idx].load(Ordering::Relaxed) } else { u32::MAX };
        let rollout = if d == u32::MAX { u32::MAX } else { self.pacer.visit_tick(key.0 as usize, current_tick, d) };
        if let Some(lac) = &self.last_access_tick {
            let la = lac[idx].load(Ordering::Relaxed);
            if la != u32::MAX && la <= current_tick {
                let age = current_tick - la;
                let prior = current_tick.saturating_add(age.max(1));
                return rollout.min(prior);
            }
        }
        rollout
    }

    /// The latest claim wins: the chain planes re-predict every target from each newer state.
    fn note_near_claim(&self, key: LayerExpertKey, target_tick: u32) {
        if !self.channels.lookahead {
            return;
        }
        let idx = key.0 as usize * self.num_experts + key.1 as usize;
        if let Some(cell) = self.near_claim_tick.get(idx) {
            cell.store(target_tick, Ordering::Relaxed);
        }
    }

    /// Rewrites the draft leg whole and seeds the next roll; an empty window clears every claim.
    fn note_draft_window(&self, current_tick: u32, window: &[Vec<Vec<u16>>]) {
        let mut claimed = self.draft_claimed.lock().expect("NextUseOracle poisoned");
        for &idx in claimed.iter() {
            self.draft_claim_token[idx as usize].store(u32::MAX, Ordering::Relaxed);
        }
        claimed.clear();
        // Without the draft channel, no seed either: a seeded rollout carries the drafter's effect on eviction too.
        if !self.channels.draft {
            return;
        }
        *self.draft_seed.lock().expect("NextUseOracle poisoned") =
            window.last().map(|frontier| (window.len() as u32, frontier.clone()));
        // Deep to shallow, so the nearest claim wins: experts recur across drafted tokens.
        let next_tok = current_tick / self.pacer.ticks_per_token().max(1) + 1;
        for (d, tok) in window.iter().enumerate().rev() {
            for (layer, experts) in tok.iter().enumerate().take(self.block_count) {
                let base = layer * self.num_experts;
                for &e in experts {
                    if let Some(cell) = self.draft_claim_token.get(base + e as usize) {
                        cell.store(next_tok + d as u32, Ordering::Relaxed);
                        claimed.push((base + e as usize) as u32);
                    }
                }
            }
        }
    }

    fn feeds_near_claims(&self) -> bool {
        self.channels.lookahead
    }

    /// Monotonic, so a scan-style tick 0 or an out-of-order stamp never erases a reused key's prior.
    fn note_access(&self, key: LayerExpertKey, current_tick: u32) {
        let Some(lac) = &self.last_access_tick else {
            return;
        };
        let idx = key.0 as usize * self.num_experts + key.1 as usize;
        let Some(cell) = lac.get(idx) else {
            return;
        };
        let la = cell.load(Ordering::Relaxed);
        if la == u32::MAX {
            cell.store(current_tick, Ordering::Relaxed);
        } else {
            cell.fetch_max(current_tick, Ordering::Relaxed);
        }
    }

    fn tracks_access(&self) -> bool {
        self.last_access_tick.is_some()
    }

    fn observe_routing(&self, layer: usize, experts: &[u16]) {
        // What it learns is read by the rollout alone.
        if layer >= self.block_count || experts.is_empty() || !self.channels.rollout {
            return;
        }
        let mut st = self.state.write().expect("NextUseOracle poisoned");
        // A new decode token: roll from the previous token's complete routing before this one overwrites it.
        if layer <= st.last_layer {
            self.roll_horizon(&st, HORIZON, HORIZON_K);
        }
        st.last_layer = layer;
        // C[layer][prev][cur] += 1.
        let prev = std::mem::take(&mut st.latest_routing[layer]);
        for &a in &prev {
            let base = self.row_base(layer, a as usize);
            for &b in experts {
                if (b as usize) < self.num_experts {
                    let i = base + b as usize;
                    st.counts[i] = st.counts[i].saturating_add(1);
                }
            }
        }
        st.latest_routing[layer] = experts.to_vec();
    }
}

/// Highest score first, ties by ascending id; the order is total (no NaN), so selecting first equals sorting all.
fn rank_top_k<S: PartialOrd + Copy>(v: &mut Vec<(S, u16)>, k: usize) {
    let cmp = |a: &(S, u16), b: &(S, u16)| b.0.partial_cmp(&a.0).expect("NaN score").then(a.1.cmp(&b.1));
    if k == 0 {
        v.clear();
        return;
    }
    if v.len() > k {
        v.select_nth_unstable_by(k, cmp);
        v.truncate(k);
    }
    v.sort_unstable_by(cmp);
}

/// The tick schedule for fetch deadlines and next-use placement; O(1).
#[derive(Debug, Clone, Copy)]
pub struct DecodePacer {
    ticks_per_token: u32,
    layers_per_group: u32,
}

impl DecodePacer {
    /// `layers_per_group`: decode's layers a tick ([`decode_layers`]).
    pub fn new(block_count: usize, layers_per_group: u32) -> Self {
        let layers_per_group = layers_per_group.max(1);
        Self { ticks_per_token: (block_count as u32).div_ceil(layers_per_group), layers_per_group }
    }

    pub fn ticks_per_token(&self) -> u32 {
        self.ticks_per_token
    }

    /// The tick of `(layer, call)`, monotone in `call`.
    pub fn target_tick(&self, layer_idx: usize, call_count: u32) -> u32 {
        let base = layer_idx as u32 / self.layers_per_group;
        call_count.saturating_mul(self.ticks_per_token).saturating_add(base)
    }

    /// The tick of `layer`'s visit strictly after `after_tick`, skipping `tokens_ahead` visits.
    pub fn visit_tick(&self, layer_idx: usize, after_tick: u32, tokens_ahead: u32) -> u32 {
        let base = (layer_idx as u32 / self.layers_per_group) as u64;
        let per_token = self.ticks_per_token as u64;
        let after = after_tick as u64;
        // The first token visiting this layer after `after_tick`.
        let tok = if after < base { 0 } else { (after - base) / per_token + 1 };
        let tick = (tok + tokens_ahead as u64) * per_token + base;
        tick.min(u32::MAX as u64 - 1) as u32
    }
}

/// A recorded tick schedule, also the clairvoyant eviction oracle over its trace.
pub trait RecordedSchedule: Oracle {
    /// Wrapped modulo the schedule's length.
    fn target_tick(&self, layer_idx: usize, call_count: u32) -> u32;
    fn tick_count(&self) -> usize;
}

/// The ticks every fetch deadline and eviction take as "now"; an evaluation's recorded schedule carries its routings.
pub enum SchedulePacer {
    Procedural(DecodePacer),
    Recorded(Arc<dyn RecordedSchedule>),
}

impl SchedulePacer {
    pub fn target_tick(&self, layer_idx: usize, call_count: u32) -> u32 {
        match self {
            Self::Procedural(p) => p.target_tick(layer_idx, call_count),
            Self::Recorded(o) => o.target_tick(layer_idx, call_count),
        }
    }
}

/// Waves per command buffer, their slots pinned until it commits; one, as a commit's cost hides under overlapping IO.
pub const PREFILL_CB_WAVES: usize = 1;

/// A kernel threshold, not a memory decision: under ~256 tokens each routed expert's weights are re-read per token.
pub const PREFILL_BATCH_TOKENS: u32 = 2048;

/// The wave width: the widest even split fitting `PREFILL_CB_WAVES + 1` waves and a slot in the smallest bank, so
/// compute overlaps reads; 0 is one wave of the whole layer. A batch's down outputs are summed once after its last
/// wave, in slot order, so the partition moves no bit.
pub fn prefill_batch(layer_width: u32, wave_capacity: u32) -> u32 {
    let in_flight = PREFILL_CB_WAVES as u32 + 1;
    let widest = wave_capacity.saturating_sub(1) / in_flight;
    if wave_capacity > 0 && widest < layer_width {
        layer_width.div_ceil(layer_width.div_ceil(widest.max(1)))
    } else {
        0
    }
}

/// Decode's layers a tick; the /2 leaves the bank headroom for each layer's unique experts.
pub fn decode_layers(hot_budget: u32, topk: u32) -> u32 {
    (hot_budget / (topk.max(1) * 2)).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One layer a tick: tick = token × block_count + layer.
    fn oracle(block_count: usize, num_experts: usize) -> NextUseOracle {
        let every = EvictionChannels { lookahead: true, draft: true, rollout: true };
        NextUseOracle::new(DecodePacer::new(block_count, 1), block_count, num_experts, every)
    }

    /// On a table with ties, saturated cells, an empty layer and a drafted seed. The cycle {1, 2} ⇄ {5, 6} on layer 0:
    /// each call rolls from the token before, then learns the transition into it; a draft window's claims protect
    /// their use forward, the nearest depth winning; a key used `age` ticks ago is expected back `age` from now.
    #[test]
    fn rollout_matches_the_reference_loop_bitwise_and_scores_a_keys_next_use() {
        let (bc, ne, h, k) = (7usize, 41usize, 5usize, 6usize);
        let o = oracle(bc, ne);
        // A deterministic table: small values (ties everywhere), a few cells near saturation, one layer never routed.
        let lcg = |x: &u32| Some(x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223));
        let counts: Vec<u32> = std::iter::successors(lcg(&0x9E37_79B9), lcg)
            .take(bc * ne * ne)
            .map(|r| match r % 17 {
                0 => u32::MAX - (r >> 8) % 3,
                1..=4 => 0,
                _ => (r >> 16) % 4,
            })
            .collect();
        let mut last: Vec<Vec<u16>> = (0..bc)
            .map(|l| if l == 3 { Vec::new() } else { vec![(l * 5 % ne) as u16, ((l * 7 + 2) % ne) as u16, 4] })
            .collect();
        last[6] = vec![1, 1, 39];
        let seed = (2u32, vec![vec![], vec![9, 10], vec![], vec![11], vec![], vec![], vec![]]);
        let mut st = o.state.write().unwrap();
        st.counts.copy_from_slice(&counts);
        st.latest_routing = last.clone();
        drop(st);
        *o.draft_seed.lock().unwrap() = Some(seed.clone());
        o.roll_horizon(&o.state.read().unwrap(), h as u32, k);
        let got: Vec<u32> = o.pred_dist.iter().map(|c| c.load(Ordering::Relaxed)).collect();

        // The reference loop, written out.
        let mut want = vec![u32::MAX; bc * ne];
        let mut cur = last;
        seed.1.into_iter().enumerate().filter(|(_, e)| !e.is_empty()).for_each(|(l, e)| cur[l] = e);
        for step in seed.0..seed.0 + h as u32 {
            for layer in (0..bc).filter(|&l| !cur[l].is_empty()).collect::<Vec<_>>() {
                let sum = |e: usize| {
                    cur[layer].iter().fold(0u32, |s, &a| s.saturating_add(counts[(layer * ne + a as usize) * ne + e]))
                };
                let mut scratch: Vec<(u32, u16)> = (0..ne).map(|e| (sum(e), e as u16)).filter(|s| s.0 > 0).collect();
                scratch.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
                scratch.truncate(k);
                let next: Vec<u16> = scratch.iter().map(|&(_, e)| e).collect();
                next.iter().for_each(|&e| want[layer * ne + e as usize] = want[layer * ne + e as usize].min(step));
                cur[layer] = next;
            }
        }
        assert_eq!(got, want);
        assert!(got.iter().any(|&d| d != u32::MAX), "the table must predict something");
        // The learnt transitions.
        let o = oracle(1, 16);
        assert!(!o.tracks_access(), "the horizon stamps no access");
        [[1u16, 2], [5, 6], [1, 2]].iter().for_each(|set| o.observe_routing(0, set));
        let next = o.predict_layer(0, 8);
        assert!(next.contains(&5) && next.contains(&6), "predicts 5 and 6 after 1 and 2: {next:?}");
        [[5u16, 6], [1, 2], [5, 6]].iter().for_each(|set| o.observe_routing(0, set));
        // The last roll ran from {1, 2}: {5, 6} at distance 0, {1, 2} at 1; layer 0 is visited once a token.
        assert_eq!(o.eviction_score((0, 5), 100), 101, "distance 0: scored at its next visit's tick");
        assert_eq!(o.eviction_score((0, 1), 100), 102, "distance 1: a visit later");
        assert_eq!(o.eviction_score((0, 7), 100), u32::MAX, "never predicted: evicted first");
        // One layer a tick, so one tick a token; after forward 100, window[d] is forward 101 + d. Expert 3, drafted at
        // both depths, takes the nearer: during forward 101 its claim is live and sorts ahead of 4's.
        let o = oracle(1, 16);
        o.note_draft_window(100, &[vec![vec![3]], vec![vec![4, 3]]]);
        let (k3, k4) = ((0u16, 3u16), (0u16, 4u16));
        let (s3, s4) = (o.eviction_score(k3, 101), o.eviction_score(k4, 101));
        assert!(s3 < s4 && s4 < u32::MAX, "nearest live claim wins: {s3} vs {s4}");
        // 3's claim expires after its use forward; 4's lives through 102.
        assert!(o.eviction_score(k3, 103) == u32::MAX && o.eviction_score(k4, 102) < u32::MAX);
        // A new feed un-claims keys absent from it; an empty one, all.
        o.note_draft_window(100, &[vec![vec![9]]]);
        assert!(o.eviction_score(k4, 101) == u32::MAX && o.eviction_score((0, 9), 101) < u32::MAX);
        o.note_draft_window(100, &[]);
        assert_eq!(o.eviction_score((0, 9), 101), u32::MAX);
        // The recency prior; alone, as `OSCAR_ORACLE=lru`, it ranks by last use.
        let o = oracle(2, 4).with_recency_prior();
        let (key, fresh, rescanned) = ((1, 2), (1, 3), (0, 0));
        assert_eq!(o.eviction_score(key, 100), u32::MAX, "never used, no rollout: evicted first");
        o.note_access(key, 100);
        assert_eq!((o.eviction_score(key, 105), o.eviction_score(key, 150)), (110, 200), "ages 5 and 50");
        o.note_access(fresh, 140);
        assert!(o.eviction_score(fresh, 150) < o.eviction_score(key, 150), "the fresher key kept longer");
        o.note_access(rescanned, 50);
        assert_eq!(o.eviction_score(rescanned, 150), 250, "age 100");
        o.note_access(rescanned, 150);
        assert_eq!(o.eviction_score(rescanned, 151), 152, "a use refreshes the stamp");
        o.note_access(rescanned, 0);
        assert_eq!(o.eviction_score(rescanned, 151), 152, "a scan's tick 0 does not clobber it");
        let lru = oracle(1, 4).recency_alone();
        lru.observe_routing(0, &[1]);
        lru.note_access((0, 1), 10);
        lru.note_access((0, 2), 12);
        assert!(lru.eviction_score((0, 1), 13) > lru.eviction_score((0, 2), 13), "the older use first");
        assert_eq!(lru.eviction_score((0, 3), 13), u32::MAX, "an unseen key before all");
    }

    /// A bank that never binds the wave.
    const ROOMY: u32 = u32::MAX;

    /// 94 layers two a tick: 47 ticks a token, layer 4 at tick 2 of each; 4 layers eight a tick: one tick a token. The
    /// committing waves and the one in flight leave a slot to spare, or a wave is one expert; the fewest, cut even.
    #[test]
    fn the_pacer_ticks_each_layer_group_once_a_token_and_the_wave_rule_fits_the_fewest_even_waves() {
        // 128 / (8 × 2) = 8, 46 / 16 = 2; a top-k of 0 clamps rather than divide by zero.
        [(1, 8, 1), (128, 8, 8), (46, 8, 2), (64, 0, 32)]
            .iter()
            .for_each(|&(hb, k, l)| assert_eq!(decode_layers(hb, k), l, "hot_budget={hb} topk={k}"));
        let (p, one) = (DecodePacer::new(94, 2), DecodePacer::new(4, 8));
        assert_eq!((p.ticks_per_token(), one.ticks_per_token()), (47, 1));
        let targets = [((0, 0), 0), ((1, 0), 0), ((2, 0), 1), ((93, 0), 46), ((0, 1), 47), ((4, 10), 472)];
        targets.iter().for_each(|&((l, c), at)| assert_eq!(p.target_tick(l, c), at, "layer {l} call {c}"));
        assert!([0, 3, 93].into_iter().all(|l| p.target_tick(l, 5) < p.target_tick(l, 6)), "monotone in the call");
        // Strictly after `after`: from before the layer's first visit, at a visit, tokens ahead, far along.
        let visits = [((0, 0), 2), ((1, 0), 2), ((2, 0), 49), ((2, 3), 190), ((4702, 0), 4749)];
        visits.iter().for_each(|&((a, n), tick)| assert_eq!(p.visit_tick(4, a, n), tick, "after {a}, {n} ahead"));
        assert_eq!((one.target_tick(3, 7), one.visit_tick(2, 7, 0)), (7, 8));
        // (layer_width, smallest bank, experts), at the recorded banks of the shipped models and of K3.
        for (lw, cap, want) in [
            (128u32, ROOMY, 0u32),
            (128, 0, 0),    // no bank bounds the wave (a census)
            (128, 257, 0),  // two layers and the evictable slot: exactly holds
            (128, 256, 64), // one short: two even waves
            (256, 1950, 0), // GLM-5.2 at 52 GB
            (256, 600, 0),  // GLM-5.2 at 32 GB
            (256, 300, 128),
            (512, 4320, 0), // the 397B
            (896, 77, 38),  // K3 at its recorded 77-slot bank: 24 waves
            (896, 99, 48),
            (896, 132, 64),
            (896, 176, 82), // 11 waves, the last of 76
            (64, 3, 1),     // a bank of three: the narrowest wave
            (64, 1, 1),     // a bank that cannot hold two: still the narrowest
            (0, 64, 0),     // one wave
            (64, 0, 0),     // no bank bounds the wave (a census)
        ] {
            assert_eq!(prefill_batch(lw, cap), want, "lw={lw} cap={cap}");
        }
        let in_flight = PREFILL_CB_WAVES as u32 + 1;
        for lw in [64u32, 128, 160, 256, 384, 512, 896, 1024] {
            for cap in 1..=(in_flight * lw + 2) {
                let width = prefill_batch(lw, cap);
                let k = if width == 0 { lw } else { width };
                assert!(k >= 1 && k <= lw, "lw={lw} cap={cap}: k={k}");
                if cap >= in_flight * lw + 1 {
                    assert_eq!(width, 0, "lw={lw} cap={cap}: the bank holds two layers");
                    continue;
                }
                assert!(in_flight * k + 1 <= cap || k == 1, "lw={lw} cap={cap}: k={k} does not fit twice");
                let widest = ((cap - 1) / in_flight).max(1);
                let waves = lw.div_ceil(k);
                assert_eq!(waves, lw.div_ceil(widest), "lw={lw} cap={cap}: not the fewest waves");
                let last = lw - (waves - 1) * k;
                assert!(last >= 1 && k - last < waves, "lw={lw} cap={cap}: runt wave of {last} under k={k}");
            }
        }
    }
}
