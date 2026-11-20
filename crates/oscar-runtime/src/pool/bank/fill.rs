//! The bank's fills: the walk, the claims (demand, staged), the demand plan, publish and abort, the waits, the pins.

use super::*;
use crate::pool::reads::{FetchClass, FetchQueue, FetchReq, Lane};

/// What a demand plan found for a key, under one lock.
#[derive(Debug, Clone, Copy)]
pub enum Plan {
    Resident(usize),
    /// Another caller's read is in flight: wait in [`Bank::wait`].
    Waiting,
    /// `key` in flight to `slot`: the caller must push its read or call [`Bank::abort`].
    Reserved {
        slot: usize,
        extent: ExpertExtent,
    },
}

/// A free slot, or a victim with its score so the stager can decline a swap that does not pay.
#[derive(Debug, Clone, Copy)]
pub(super) enum Pick {
    Free(usize),
    Victim { slot: usize, score: u32 },
}

/// The three banks' nesting when pinned blocking; each path its own, as the resolver's key-outer order costs prefill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PinOrder {
    /// One expert across all three projections, then the next.
    KeyOuter,
    /// One projection across all keys, then the next.
    BankOuter,
}

impl Bank {
    /// A free slot first, the top one whose fence has passed (an aborted reservation leaves its fence behind); else
    /// one walk from the cursor, whose runners-up a demand claim keeps in `keep`.
    pub(super) fn pick(&self, st: &State, current_tick: u32, keep: Option<&mut Vec<(u32, u32)>>) -> Option<Pick> {
        let reached = self.reached();
        if let Some(n) = st.free.iter().rev().position(|&s| self.passed(s as usize, reached)) {
            return Some(Pick::Free(st.free[st.free.len() - 1 - n] as usize));
        }
        const PROMISE_FAR: u32 = 8;
        let (cap, per) = (self.index, st.per_segment);
        let oracle = self.oracle.get().map(|(o, _)| o);
        let early_exit = st.free.is_empty();
        let start = self.cursor.load(Ordering::Relaxed) % cap;
        let reached = self.reached();
        // A fence ranks, never excludes, as a bulk prefill fences every recent slot.
        type Rank = (bool, bool, u32, std::cmp::Reverse<u64>);
        // A segment's worth for a demand claim: the pool's own quantum.
        let want = if keep.is_some() { per.max(1) } else { 1 };
        let mut best: Vec<(Rank, usize)> = Vec::with_capacity(want);
        let mut i = 0;
        while i < cap {
            let s = (start + i) % cap;
            let seg = &st.segs[s / per];
            // An absent, retiring or wholly sealed segment is skipped whole.
            if !seg.standing() || seg.sealed as usize >= per {
                i += per - s % per;
                continue;
            }
            i += 1;
            let slot = &self.slots[s];
            // An owner-less slot is on the free list, already tried against this fence.
            let Some(owner) = st.owners[s].key.filter(|_| slot.pins.load(Ordering::Acquire) == 0) else { continue };
            // Never a fill in flight here; a duplicate (published elsewhere, as a seal leaves copies) goes first.
            let published = self.word_of(owner);
            if published == filling(s) {
                continue;
            }
            let fence_ok = self.passed(s, reached);
            let duplicate = published >= 0 && published != s as i32;
            if duplicate && fence_ok {
                self.cursor.store(s + 1, Ordering::Relaxed);
                return Some(Pick::Victim { slot: s, score: u32::MAX });
            }
            // The call's own fill, read by no launch yet, is no candidate: a claim that took it would eat the wave
            // about to pin it, and the wave's refill the next.
            let routed = st.routed.contains(&owner);
            if routed && !duplicate && slot.fence.load(Ordering::Acquire) == 0 {
                continue;
            }
            let (mut score, mut far_promise) = (u32::MAX, false);
            if !duplicate {
                score = oracle.map_or(u32::MAX, |o| o.eviction_score(owner, current_tick));
                // What the oracle cannot see: a staged fill's promise, not yet due, and the call's routed set.
                let promise = slot.promise.load(Ordering::Acquire);
                if promise != u32::MAX && promise >= current_tick && slot.prefetched.load(Ordering::Acquire) {
                    score = score.min(promise);
                    far_promise = promise > current_tick.saturating_add(PROMISE_FAR);
                }
                if routed {
                    score = score.min(current_tick);
                }
            }
            if early_exit && fence_ok && !far_promise && score == u32::MAX {
                self.cursor.store(s + 1, Ordering::Relaxed);
                return Some(Pick::Victim { slot: s, score });
            }
            let rank: Rank = (fence_ok, !far_promise, score, std::cmp::Reverse(st.owners[s].seq));
            // Ordered insert: one compare against the worst kept rejects the common case.
            if best.len() == want && rank <= best[want - 1].0 {
                continue;
            }
            let at = best.partition_point(|(b, _)| *b > rank);
            best.insert(at, (rank, s));
            best.truncate(want);
        }
        let &((_, _, score, _), slot) = best.first()?;
        if let Some(keep) = keep {
            keep.clear();
            keep.extend(best[1..].iter().rev().map(|&((_, _, sc, _), sl)| (sl as u32, sc)));
        }
        Some(Pick::Victim { slot, score })
    }

    /// A kept runner-up, not a claim: the lock was dropped since the walk, so every term is read again as the walk did.
    pub(super) fn take_runner_up(
        &self, st: &State, kept: &mut Vec<(u32, u32)>, current_tick: u32,
    ) -> Option<(usize, u32)> {
        let reached = self.reached();
        while let Some((s, score)) = kept.pop() {
            let (s, slot) = (s as usize, &self.slots[s as usize]);
            let promise = slot.promise.load(Ordering::Acquire);
            let promised = promise != u32::MAX && promise >= current_tick && slot.prefetched.load(Ordering::Acquire);
            let takeable = st.segs[s / st.per_segment].standing()
                && st.owners[s].key.is_some_and(|o| !st.routed.contains(&o) && self.word_of(o) != filling(s))
                && slot.pins.load(Ordering::Acquire) == 0
                && self.passed(s, reached)
                && !promised;
            if takeable {
                return Some((s, score));
            }
        }
        None
    }

    /// The pick's slot, its owner evicted, or its evicted layer cleared where it was free.
    fn take(&self, st: &mut State, pick: Pick, current_tick: u32) -> usize {
        match pick {
            Pick::Free(slot) => {
                self.slots[slot].evicted_layer.store(u16::MAX, Ordering::Release);
                slot
            }
            Pick::Victim { slot, .. } => {
                self.evict(st, slot, current_tick);
                slot
            }
        }
    }

    /// A demand claim's slot, its owner evicted unconditionally; `Exhausted` with no candidate.
    pub(super) fn acquire(&self, st: &mut State, current_tick: u32) -> Result<usize> {
        let mut kept = std::mem::take(&mut st.runners_up);
        let pick = match self.take_runner_up(st, &mut kept, current_tick) {
            Some((slot, score)) => Some(Pick::Victim { slot, score }),
            None => self.pick(st, current_tick, Some(&mut kept)),
        };
        st.runners_up = kept;
        Ok(self.take(st, pick.ok_or_else(|| Error::Exhausted { hot: self.capacity() })?, current_tick))
    }

    /// A slot for `key`, due at `deadline`: a free one, or a victim needed strictly later; refused, nothing changes.
    pub fn stage(&self, key: LayerExpertKey, current_tick: u32, deadline: u32) -> Option<usize> {
        let mut st = self.lock();
        // Read again under the lock: a fetch may have landed since the caller's filter.
        if self.word_of(key) != ABSENT {
            return None;
        }
        let pick = self.pick(&st, current_tick, None)?;
        // A victim needed no later than `key`: the swap does not pay.
        if matches!(pick, Pick::Victim { score, .. } if score <= deadline) {
            return None;
        }
        let slot = self.take(&mut st, pick, current_tick);
        // The promise, for the walk and the waste split, live while prefetched.
        self.claim(&mut st, slot, key, current_tick, deadline);
        self.mark(key, filling(slot));
        Some(slot)
    }

    /// The staged read `key` needs here, due at `deadline`; none when resident or not this class's.
    pub fn staged_read(self: &Arc<Self>, key: LayerExpertKey, deadline: u32) -> Option<FetchReq> {
        let extent = self.extent(key).ok().filter(|_| !self.is_resident(key))?;
        Some(FetchReq::new(self, key, extent, usize::MAX, FetchClass::Staged, deadline))
    }

    /// Plan `keys`, each reservation's read pushed at `class` as it is claimed, due at `deadline`: a claim stands, so
    /// no waiter on it is stranded and no fenced slot is thrown back to the free list.
    pub fn push_demand(
        self: &Arc<Self>, queue: &FetchQueue, keys: &[LayerExpertKey], deadline: u32, class: FetchClass,
    ) -> Result<Vec<Plan>> {
        self.plan_pushing(keys, deadline, &mut |slot, key, extent| {
            queue.push_many([FetchReq::new(self, key, extent, slot, class, deadline)]);
        })
    }

    /// A plan whose reservations push no read: a test's, which reads or aborts them itself.
    pub fn plan(&self, keys: &[LayerExpertKey], current_tick: u32) -> Result<Vec<Plan>> {
        self.plan_pushing(keys, current_tick, &mut |_, _, _| {})
    }

    /// The lock once per key: a claim sweeps the bank, and a batch-long hold would starve the readers' completions.
    /// An exhausted bank is a stall, not a failure: the claims made stand, their reads landing, until a slot frees.
    fn plan_pushing(
        &self, keys: &[LayerExpertKey], current_tick: u32, push: &mut dyn FnMut(usize, LayerExpertKey, ExpertExtent),
    ) -> Result<Vec<Plan>> {
        // Extents first: fail before touching state.
        let extents = keys.iter().map(|&k| self.extent(k)).collect::<Result<Vec<_>>>()?;
        let mut plans: Vec<Option<Plan>> = keys.iter().map(|&k| self.resident_slot(k).map(Plan::Resident)).collect();
        let stall_until = Instant::now() + self.stall_cap();
        while let Err(e) = self.plan_pass(keys, &extents, &mut plans, current_tick, push) {
            if !matches!(e, Error::Exhausted { .. }) || Instant::now() >= stall_until {
                return Err(e);
            }
            drop(self.stall(self.lock(), current_tick));
        }
        Ok(plans.into_iter().map(|p| p.expect("every key produced a plan")).collect())
    }

    /// The keys not yet planned, a lock each, a reservation's read pushed outside it; at a failure the earlier
    /// reservations stand, their reads in flight.
    fn plan_pass(
        &self, keys: &[LayerExpertKey], extents: &[ExpertExtent], plans: &mut [Option<Plan>], current_tick: u32,
        push: &mut dyn FnMut(usize, LayerExpertKey, ExpertExtent),
    ) -> Result<()> {
        // The runners-up are this pass's; an older ranking is stale.
        let mut fresh = true;
        for ((plan, &key), &extent) in plans.iter_mut().zip(keys).zip(extents).filter(|((p, _), _)| p.is_none()) {
            let mut st = self.lock();
            if std::mem::take(&mut fresh) {
                st.runners_up.clear();
            }
            *plan = Some(match self.word_of(key) {
                s if s >= 0 => Plan::Resident(s as usize),
                ABSENT => {
                    let slot = self.acquire(&mut st, current_tick)?;
                    self.claim(&mut st, slot, key, current_tick, u32::MAX);
                    self.mark(key, filling(slot));
                    drop(st);
                    self.note_cold_miss(key);
                    push(slot, key, extent);
                    Plan::Reserved { slot, extent }
                }
                _ => Plan::Waiting,
            });
        }
        Ok(())
    }

    /// A demand resolve's keys, each once: a hit if resident, a miss if reserved, a wait on another's read.
    pub fn note_resolve(plans: &[Plan]) {
        for plan in plans {
            let counter = match plan {
                Plan::Resident(_) => &COUNTERS.hits,
                Plan::Reserved { .. } => &COUNTERS.misses,
                Plan::Waiting => &COUNTERS.waits,
            };
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A fresh miss counted as a first touch or a refetch, the key marked filled.
    fn note_cold_miss(&self, key: LayerExpertKey) {
        if let Some(i) = self.key_index(key) {
            let decode = COUNTERS.decode_armed.load(Ordering::Acquire);
            let (all, in_decode) = if self.ever_filled[i].swap(true, Ordering::Relaxed) {
                (&COUNTERS.refetch, &COUNTERS.refetch_decode)
            } else {
                (&COUNTERS.first_touch, &COUNTERS.first_touch_decode)
            };
            all.fetch_add(1, Ordering::Relaxed);
            if decode {
                in_decode.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// `key`'s fill into `slot` is the one in flight, or the mis-sequence.
    fn check_fill(&self, st: &State, op: &str, slot: usize, key: LayerExpertKey) -> Result<()> {
        let why = if !self.present(st, slot) {
            "the slot is absent".to_string()
        } else if st.owners[slot].key != Some(key) {
            format!("the slot is owned by {:?}", st.owners[slot].key)
        } else if self.word_of(key) != filling(slot) {
            "the key's fill is not in flight there".to_string()
        } else {
            return Ok(());
        };
        Err(Error::Lifecycle(format!("{op}: slot {slot}, key {key:?}: {why}")))
    }

    /// Publish `key` and wake its waiters; a mis-sequenced call changes nothing.
    pub fn publish(&self, slot: usize, key: LayerExpertKey, prefetched: bool) -> Result<()> {
        let st = self.lock();
        self.check_fill(&st, "complete_fetch", slot, key)?;
        // Stamp and tag, then in flight to published in one store, under the lock; the reader waited on the old
        // owner's fence before writing, and no launch has read this fill.
        let s = &self.slots[slot];
        s.stamp.store(stamp_of(key), Ordering::Release);
        s.fence.store(0, Ordering::Release);
        s.prefetched.store(prefetched, Ordering::Release);
        self.mark(key, slot as i32);
        drop(st);
        // Stamp a demand fill's recency now, or a recency policy sees a never-accessed slot, its favourite victim.
        let claimed_at = s.claimed_at.load(Ordering::Acquire);
        if !prefetched && claimed_at != u32::MAX {
            self.note_access(key, claimed_at);
        }
        self.cv.notify_all();
        Ok(())
    }

    /// Roll back a fill whose read failed: the slot free again.
    pub fn abort(&self, slot: usize, key: LayerExpertKey) -> Result<()> {
        let mut st = self.lock();
        self.check_fill(&st, "abort_fetch", slot, key)?;
        self.mark(key, ABSENT);
        st.assign(slot, None);
        drop(st);
        self.cv.notify_all();
        Ok(())
    }

    /// Lock-free when none of `keys` is in flight; the barrier's counters count the engine's lane alone.
    pub fn wait(&self, keys: &[LayerExpertKey], lane: Lane) {
        let landing = |k: &&LayerExpertKey| self.landing(**k);
        if !keys.iter().any(|k| landing(&k)) {
            return;
        }
        let q0 = oscar_offload::io::now_raw();
        let mut st = self.lock();
        // The barrier releases on the last key; the first arrival prices what an arrival-order consumer could start.
        let missing = keys.iter().filter(landing).count();
        let mut first_arrival: u64 = 0;
        while keys.iter().any(|k| landing(&k)) {
            st = self.cv.wait(st).expect("HotPoolBank state lock poisoned");
            if first_arrival == 0 && keys.iter().filter(landing).count() < missing {
                first_arrival = oscar_offload::io::now_raw();
            }
        }
        drop(st);
        let q1 = oscar_offload::io::now_raw();
        // A drafter wait blocks the drafter, not the token, and would push `barrier_wait_share` past 1.
        if lane == Lane::Engine {
            let c = &COUNTERS;
            for (counter, n) in [
                (&c.barrier_blocked, 1),
                (&c.barrier_missing, missing as u64),
                (&c.barrier_multi, (missing >= 2) as u64),
                (&c.barrier_keys, keys.len() as u64),
                (&c.barrier_wait_raw, q1.saturating_sub(q0)),
            ] {
                counter.fetch_add(n, Ordering::Relaxed);
            }
            if first_arrival != 0 {
                c.barrier_spread_raw.fetch_add(q1.saturating_sub(first_arrival), Ordering::Relaxed);
            }
        }
    }

    /// Pin first, then read the word and the stamp again (the slot-reuse race); `Some` holds the pin, the access noted.
    pub(super) fn try_pin(&self, key: LayerExpertKey, current_tick: u32) -> Option<usize> {
        let word = self.word(key)?;
        for _ in 0..4 {
            let at = word.load(Ordering::Acquire);
            let slot = usize::try_from(at).ok()?;
            let s = &self.slots[slot];
            s.pins.fetch_add(1, Ordering::AcqRel);
            if word.load(Ordering::Acquire) == at && s.stamp.load(Ordering::Acquire) == stamp_of(key) {
                self.note_access(key, current_tick);
                // The save is counted once per staging.
                if s.prefetched.swap(false, Ordering::AcqRel) {
                    COUNTERS.prefetch_saves.fetch_add(1, Ordering::Relaxed);
                }
                return Some(slot);
            }
            s.pins.fetch_sub(1, Ordering::AcqRel);
        }
        None
    }

    /// After its fetch was waited on: evicted since or its read failed, `key` is read again as demand, twice at most.
    pub(super) fn pin_or_refill(
        self: &Arc<Self>, queue: &FetchQueue, key: LayerExpertKey, current_tick: u32, lane: Lane,
    ) -> Result<usize> {
        if let Some(slot) = self.try_pin(key, current_tick) {
            return Ok(slot);
        }
        for _ in 0..2 {
            self.push_demand(queue, std::slice::from_ref(&key), current_tick, FetchClass::Demand)?;
            self.wait(std::slice::from_ref(&key), lane);
            if let Some(slot) = self.try_pin(key, current_tick) {
                return Ok(slot);
            }
        }
        Err(Error::Lifecycle(format!("pin: key (L={}, E={}) did not stay resident through two reads", key.0, key.1)))
    }

    /// Release a pin per slot; unpinning a slot no pin holds stops oscar.
    pub fn unpin(&self, slots: &[usize]) {
        for &s in slots {
            let prev = self.slots[s].pins.fetch_sub(1, Ordering::AcqRel);
            assert!(prev > 0, "unpin of slot {s}, which no pin holds");
        }
    }
}

/// Every key on the gate, up and down banks, or none (`None` at the first miss): a pin left behind is a slot never
/// reclaimed. Whole, it is a resolve that waited on no read: a hit for each key, counted only then.
pub fn pin_all(banks: [&Arc<Bank>; 3], keys: &[LayerExpertKey], target_tick: u32) -> Option<[Vec<usize>; 3]> {
    let pinned =
        pin_each(banks, keys, PinOrder::BankOuter, |bank, key| bank.try_pin(key, target_tick).ok_or(())).ok()?;
    COUNTERS.hits.fetch_add(3 * keys.len() as u64, Ordering::Relaxed);
    Some(pinned)
}

/// [`pin_all`] after the fetches were waited on, a miss refilled.
pub fn pin_all_blocking(
    banks: [&Arc<Bank>; 3], queue: &FetchQueue, keys: &[LayerExpertKey], target_tick: u32, order: PinOrder, lane: Lane,
) -> Result<[Vec<usize>; 3]> {
    pin_each(banks, keys, order, |bank, key| bank.pin_or_refill(queue, key, target_tick, lane))
}

/// One walk for both nestings: at the first refusal every pin taken is released and the refusal returned.
fn pin_each<E>(
    banks: [&Arc<Bank>; 3], keys: &[LayerExpertKey], order: PinOrder,
    mut pin: impl FnMut(&Arc<Bank>, LayerExpertKey) -> std::result::Result<usize, E>,
) -> std::result::Result<[Vec<usize>; 3], E> {
    let mut out: [Vec<usize>; 3] = std::array::from_fn(|_| Vec::with_capacity(keys.len()));
    let n = keys.len();
    let (outer, inner) = if order == PinOrder::KeyOuter { (n, 3) } else { (3, n) };
    for o in 0..outer {
        for i in 0..inner {
            let (b, k) = if order == PinOrder::KeyOuter { (i, o) } else { (o, i) };
            match pin(banks[b], keys[k]) {
                Ok(slot) => out[b].push(slot),
                Err(e) => {
                    unpin_all(banks, &out);
                    return Err(e);
                }
            }
        }
    }
    Ok(out)
}

pub fn unpin_all(banks: [&Arc<Bank>; 3], pins: &[Vec<usize>; 3]) {
    for (bank, slots) in banks.iter().zip(pins) {
        bank.unpin(slots);
    }
}
