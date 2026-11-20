//! One class's slots across every layer: each key's word, each slot's pins and fences, the segments, the counters.

mod fill;
mod seal;
pub(crate) use seal::SEAL_READERS;

pub use fill::{pin_all, pin_all_blocking, PinOrder, Plan};

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU16, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLSharedEvent};
use oscar_offload::fixed::Fixed;
use oscar_offload::io::TieredFile;
use oscar_offload::machine::hash_map_bytes;
use oscar_offload::segments::{Addressing, Lender, SegmentPool};

use super::counters::COUNTERS;
use super::{Error, Result};
use crate::gpu::ExpertExtent;

/// `(layer, expert)`.
pub type LayerExpertKey = (u16, u16);

/// A score per key, its next-use tick: a clairvoyant oracle's actual one, the transition oracle's expected one.
pub trait Oracle: Send + Sync {
    /// Larger is more evictable; the bank `min`s in its slots' own evidence and picks the highest unpinned.
    fn eviction_score(&self, key: LayerExpertKey, current_tick: u32) -> u32;
    /// `key` was just used; stateless oracles ignore it.
    fn note_access(&self, _key: LayerExpertKey, _current_tick: u32) {}
    /// One decode call's routed set at `layer`, after the live router.
    fn observe_routing(&self, _layer: usize, _experts: &[u16]) {}
    /// Whether the bank calls [`Oracle::note_access`], a hot-path hook.
    fn tracks_access(&self) -> bool {
        false
    }
    /// A same-token prediction of `key` at `target_tick`, so eviction never runs a token stale; past claims expire.
    fn note_near_claim(&self, _key: LayerExpertKey, _target_tick: u32) {}
    /// Whether near claims are consumed, so callers can skip work for them.
    fn feeds_near_claims(&self) -> bool {
        false
    }
    /// `window[d]` is the drafted routing of decode forward d+1 after `current_tick`'s, fed at every window update.
    fn note_draft_window(&self, _current_tick: u32, _window: &[Vec<Vec<u16>>]) {}
}

/// A key's word: its published slot (`>= 0`), `ABSENT`, or a fill in flight ([`filling`]), which a demand waits on.
const ABSENT: i32 = -1;
fn filling(slot: usize) -> i32 {
    -2 - slot as i32
}

/// A slot's stamp of `key`; `-1` marks an empty slot.
fn stamp_of(key: LayerExpertKey) -> i32 {
    ((key.0 as i32) << 16) | (key.1 as i32)
}

/// The derived `Ord` fixes the order of the model's banks, which must be reproducible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BankDtype {
    Q3K,
    Q4K,
    Q6K,
    // Appended: a variant inserted among the others would reorder existing models' banks.
    Iq1S,
    Iq2XXS,
    Iq3XXS,
    /// The 32-block form, for the 640-wide down rows no 256-superblock quant holds.
    Q5_0,
}

/// The locked half of a slot: `seq` orders fills for the walk's tie-break; `free_at` indexes the free list, `MAX` none.
#[derive(Debug, Clone, Copy, Default)]
struct Owner {
    key: Option<LayerExpertKey>,
    seq: u64,
    free_at: u32,
}

struct Slot {
    pins: AtomicU32,
    /// The generation a launch reading the slot signals, which an overwrite waits for; `0` is none.
    fence: AtomicU64,
    /// The key whose bytes the slot holds (`-1` empty): a pin finding another lost the slot-reuse race.
    stamp: AtomicI32,
    /// A staged fill no demand has used yet.
    prefetched: AtomicBool,
    /// The tick the last staging admission promised (`u32::MAX`: none), a live promise only while prefetched.
    promise: AtomicU32,
    claimed_at: AtomicU32,
    /// The layer of the owner this claim evicted (`u16::MAX`: none), which the drafter may still read the slot as.
    evicted_layer: AtomicU16,
}

/// A segment's state, as a move reads it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Seg {
    pub present: bool,
    /// Slots sealed layers pin: a segment with one is never retired, a wholly sealed one never walked.
    pub sealed: u32,
    /// No claim lands here; the mover frees it once nothing can read it.
    pub retiring: bool,
    /// Inside a lender's buffer object: carrier slots only.
    pub borrowed: bool,
}

impl Seg {
    /// Present and not retiring: what the plan's shape counts, borrowed ones included.
    pub fn standing(&self) -> bool {
        self.present && !self.retiring
    }
}

/// The bank's locked half; the hot path reads only the atomics.
struct State {
    owners: Vec<Owner>,
    /// The current call's routed keys, scored as needed now; a `Vec`, cheaper than hashing at every probe.
    routed: Vec<LayerExpertKey>,
    next_seq: u64,
    /// Owner-less slots of standing segments, last freed on top; only an empty list arms the walk's early exit.
    free: Vec<u32>,
    segs: Vec<Seg>,
    per_segment: usize,
    /// This pass's full-ring runners-up as (slot, score), best last, for the demand claims after.
    runners_up: Vec<(u32, u32)>,
}

impl State {
    /// Stamps `seq` on `None → Some` and keeps the free list in step.
    fn assign(&mut self, slot: usize, key: Option<LayerExpertKey>) {
        let was_none = self.owners[slot].key.is_none();
        self.owners[slot].key = key;
        match (was_none, key.is_some()) {
            (true, true) => {
                self.leave_free(slot);
                self.owners[slot].seq = self.next_seq;
                self.next_seq += 1;
            }
            (false, false) => self.enter_free(slot),
            _ => {}
        }
    }

    /// A retiring segment's slot never re-enters the list: the pop checks no segment.
    fn enter_free(&mut self, slot: usize) {
        if self.segs[slot / self.per_segment].retiring {
            return;
        }
        self.owners[slot].free_at = self.free.len() as u32;
        self.free.push(slot as u32);
    }

    /// Swap-remove; a slot off the list (one a seal took) is a no-op.
    fn leave_free(&mut self, slot: usize) {
        let at = std::mem::replace(&mut self.owners[slot].free_at, u32::MAX) as usize;
        if at == u32::MAX as usize {
            return;
        }
        let top = self.free.pop().expect("a free slot is on the list");
        if top as usize != slot {
            self.free[at] = top;
            self.owners[top as usize].free_at = at as u32;
        }
    }
}

/// A key's slot and its epoch when read: what a consumer rechecks before trusting bytes it did not pin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resident {
    pub slot: usize,
    pub epoch: u32,
}

/// One class's expert slots across every layer, keyed by `(layer, expert)`.
pub struct Bank {
    file: Arc<TieredFile>,
    /// The slots' bytes, in segments over a fixed index space, and the address table this bank's rows read.
    segments: Arc<SegmentPool>,
    bytes_per_expert: usize,
    /// The index space, in slots.
    index: usize,
    /// Every key's extent; a missing key cannot load.
    layout: HashMap<LayerExpertKey, ExpertExtent>,
    /// Each key's word (see [`ABSENT`]) at `layer * experts + expert`, read lock-free.
    words: Vec<AtomicI32>,
    /// Set at a key's first demand miss: a later one is a refetch after eviction.
    ever_filled: Vec<AtomicBool>,
    experts: usize,
    slots: Vec<Slot>,
    /// Bumped under the lock at a retire and each claim, before any overwrite: the Draft lane nulls a stale row.
    epochs: Fixed<u32>,
    /// The draft-read fence: the highest layer encoded + 1 (0: none), and in the watermark the highest executed + 1;
    /// a committed drafter command buffer has checked its epochs, so a fill evicting a layer in between waits.
    draft_encoded_hi: AtomicU32,
    draft_watermark: Fixed<u32>,
    state: Mutex<State>,
    cv: Condvar,
    /// The next walk starts here, so the early exit round-robins; written under the state lock only.
    cursor: AtomicUsize,
    /// One held-up report per pool: a repeated stall is the same constraint.
    wedge_reported: AtomicBool,
    /// Far above a pin's life (a wave's pins drop at its commit), far below a run's timeout.
    stall_cap_ms: AtomicU64,
    /// Unset (tests), every score is `u32::MAX`; with it whether the oracle tracks access, so the rest skip the hook.
    oracle: OnceLock<(Arc<dyn Oracle>, bool)>,
    /// Candle's queue-wide completion event, which the slots' fences wait on; set on first use.
    event: OnceLock<Retained<ProtocolObject<dyn MTLSharedEvent>>>,
    dtype: BankDtype,
}

// SAFETY: the segments outlive the bank; the `state` mutex and the pin counts gate concurrency.
unsafe impl Send for Bank {}
unsafe impl Sync for Bank {}

impl Bank {
    /// One class over `segments` × `per_segment` slots, each absent until [`Self::admit`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>, file: Arc<TieredFile>, segments: usize, per_segment: usize,
        bytes_per_expert: usize, layout: HashMap<LayerExpertKey, ExpertExtent>, dtype: BankDtype, block_count: u16,
        num_experts: u16,
    ) -> Result<Self> {
        let refuse = |why: String| Err(Error::Offload(oscar_offload::Error::Layout(why)));
        if segments == 0 || per_segment == 0 || bytes_per_expert == 0 || num_experts == 0 {
            return refuse(format!(
                "HotPoolBank: segments={segments} per_segment={per_segment} bytes_per_expert={bytes_per_expert} \
                 num_experts={num_experts}"
            ));
        }
        for ((l, e), ext) in &layout {
            if ext.byte_count > bytes_per_expert {
                return refuse(format!(
                    "expert (L={l}, E={e}) byte_count {} exceeds bytes_per_expert {bytes_per_expert}",
                    ext.byte_count
                ));
            }
            if *l >= block_count || *e >= num_experts {
                return refuse(format!(
                    "key (L={l}, E={e}) out of range (block_count={block_count}, num_experts={num_experts})"
                ));
            }
        }
        let capacity = segments * per_segment;
        // A launch declares every present segment, so a segment is many slots.
        let pool = SegmentPool::new(device, bytes_per_expert, segments, per_segment, Addressing::Table)?;
        let keys = block_count as usize * num_experts as usize;
        let slot = || Slot {
            pins: AtomicU32::new(0),
            fence: AtomicU64::new(0),
            stamp: AtomicI32::new(-1),
            prefetched: AtomicBool::new(false),
            promise: AtomicU32::new(u32::MAX),
            claimed_at: AtomicU32::new(u32::MAX),
            evicted_layer: AtomicU16::new(u16::MAX),
        };
        let state = State {
            owners: vec![Owner { key: None, seq: 0, free_at: u32::MAX }; capacity],
            routed: Vec::new(),
            next_seq: 0,
            free: Vec::new(),
            segs: vec![Seg::default(); segments],
            per_segment,
            runners_up: Vec::new(),
        };
        Ok(Self {
            file,
            segments: pool,
            bytes_per_expert,
            index: capacity,
            layout,
            words: (0..keys).map(|_| AtomicI32::new(ABSENT)).collect(),
            ever_filled: (0..keys).map(|_| AtomicBool::new(false)).collect(),
            experts: num_experts as usize,
            slots: (0..capacity).map(|_| slot()).collect(),
            // Zeroed: an epoch of 0 is "never filled", a watermark of 0 "no draft past here".
            epochs: Fixed::new(device, capacity, 0)?,
            draft_encoded_hi: AtomicU32::new(0),
            draft_watermark: Fixed::new(device, 1, 0)?,
            state: Mutex::new(state),
            cv: Condvar::new(),
            cursor: AtomicUsize::new(0),
            wedge_reported: AtomicBool::new(false),
            stall_cap_ms: AtomicU64::new(30_000),
            oracle: OnceLock::new(),
            event: OnceLock::new(),
            dtype,
        })
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("HotPoolBank state lock poisoned")
    }

    /// How long a claim waits on a pool with nothing claimable before it fails.
    pub fn stall_cap(&self) -> Duration {
        Duration::from_millis(self.stall_cap_ms.load(Ordering::Relaxed))
    }

    #[cfg(test)]
    pub(crate) fn set_stall_cap_ms(&self, ms: u64) {
        self.stall_cap_ms.store(ms, Ordering::Relaxed);
    }

    /// `(segments, slots per segment)`: twice the most held, as a retired segment keeps its index until freed.
    pub fn index_space(layers: usize, experts: usize, segment_slots: usize) -> (usize, usize) {
        let segment_slots = segment_slots.max(1);
        (2 * experts.div_ceil(segment_slots), segment_slots * layers.max(1))
    }

    /// For the host line before the bank exists: each key's word and first-fill flag, the extents, each slot's state,
    /// its free-list entry and remembered resident, each segment's state, the walk's reserve, the pool's own.
    pub fn host_bytes(layers: usize, experts: usize, segment_slots: usize) -> usize {
        use std::mem::size_of;
        let keys = layers * experts;
        let (segments, per_segment) = Self::index_space(layers, experts, segment_slots);
        let per_slot = size_of::<Slot>() + size_of::<Owner>() + size_of::<u32>() + size_of::<(u32, LayerExpertKey)>();
        keys * (size_of::<AtomicI32>() + size_of::<AtomicBool>())
            + hash_map_bytes::<LayerExpertKey, ExpertExtent>(keys)
            + segments * per_segment * per_slot
            + segments * (size_of::<Seg>() + size_of::<Vec<(u32, LayerExpertKey)>>())
            + per_segment * size_of::<(u32, u32)>()
            + SegmentPool::host_bytes(segments)
    }

    /// Bytes a slot holds: the unit the run-ahead bound counts in.
    pub fn bytes_per_expert(&self) -> usize {
        self.bytes_per_expert
    }

    pub fn dtype(&self) -> BankDtype {
        self.dtype
    }

    /// The slots' segments: their bytes, buffers, addresses and the queues they are declared on.
    pub fn pool(&self) -> &SegmentPool {
        &self.segments
    }

    /// The present slots.
    pub fn capacity(&self) -> usize {
        self.segments.live_units()
    }

    /// The index space: the bound of the per-slot tables a kernel indexes.
    pub fn slot_count(&self) -> usize {
        self.index
    }

    pub fn units_per_segment(&self) -> usize {
        self.segments.per_segment()
    }

    /// Slots a claim may take: present, neither sealed nor retiring, borrowed ones only `with_loans`.
    pub fn claimable(&self, with_loans: bool) -> usize {
        let st = self.lock();
        let takes = |s: &&Seg| s.standing() && (with_loans || !s.borrowed);
        st.segs.iter().filter(takes).map(|s| st.per_segment - s.sealed as usize).sum()
    }

    /// Every segment's state, a snapshot.
    pub fn segments(&self) -> Vec<Seg> {
        self.lock().segs.clone()
    }

    pub fn residents_in(&self, seg: usize) -> usize {
        let st = self.lock();
        st.owners[seg * st.per_segment..(seg + 1) * st.per_segment].iter().filter(|o| o.key.is_some()).count()
    }

    /// Segment `seg`, its slots free, lowest on top.
    pub fn admit(&self, device: &ProtocolObject<dyn MTLDevice>, seg: usize) -> Result<()> {
        let mut st = self.lock();
        if st.segs[seg].present {
            return Err(oscar_offload::Error::Layout(format!("admit: segment {seg} is present")).into());
        }
        self.segments.take(device, seg, st.per_segment, true)?;
        self.enter(&mut st, seg, false);
        Ok(())
    }

    /// Segment `seg` inside a lender's buffer object at `offset`: free carrier slots, no OS call.
    pub fn adopt(&self, seg: usize, lender: &Lender, offset: usize) -> Result<()> {
        let mut st = self.lock();
        if st.segs[seg].present {
            return Err(oscar_offload::Error::Layout(format!("adopt: segment {seg} is present")).into());
        }
        self.segments.adopt(seg, st.per_segment, lender, offset)?;
        self.enter(&mut st, seg, true);
        Ok(())
    }

    /// Segment `seg` is present: slot state cleared, every slot free.
    fn enter(&self, st: &mut State, seg: usize, borrowed: bool) {
        let slots = seg * st.per_segment..(seg + 1) * st.per_segment;
        st.segs[seg] = Seg { present: true, borrowed, ..Seg::default() };
        for s in &self.slots[slots.clone()] {
            s.pins.store(0, Ordering::Release);
            s.fence.store(0, Ordering::Release);
            s.stamp.store(-1, Ordering::Release);
            s.evicted_layer.store(u16::MAX, Ordering::Release);
            s.prefetched.store(false, Ordering::Release);
            s.promise.store(u32::MAX, Ordering::Release);
        }
        for s in slots.rev() {
            if st.owners[s].key.is_none() {
                st.enter_free(s);
            }
        }
    }

    /// `seg` (no sealed slot) takes no claim and its unpinned residents go; [`Self::free`] finishes.
    pub fn retire(&self, seg: usize, current_tick: u32) {
        let mut st = self.lock();
        debug_assert!(
            st.segs[seg].standing() && st.segs[seg].sealed == 0,
            "retire: segment {seg} is absent, sealed or retiring"
        );
        st.segs[seg].retiring = true;
        for slot in seg * st.per_segment..(seg + 1) * st.per_segment {
            // The content is going: the drafter's next validation nulls any row naming the slot.
            self.bump_epoch(slot);
            match st.owners[slot].key {
                None => st.leave_free(slot),
                Some(o) if self.slots[slot].pins.load(Ordering::Acquire) == 0 && !self.landing(o) => {
                    self.evict(&mut st, slot, current_tick);
                }
                Some(_) => {}
            }
        }
    }

    /// Finish the retire once no pin, fill, fence or drafter frontier can read `seg`; `Err` names what holds it.
    pub fn free(&self, seg: usize, current_tick: u32, drafter_idle: bool) -> std::result::Result<(), String> {
        let mut st = self.lock();
        debug_assert!(st.segs[seg].retiring, "free: segment {seg} is not retiring");
        let fence = self.reached();
        for slot in seg * st.per_segment..(seg + 1) * st.per_segment {
            if let Some(o) = st.owners[slot].key {
                let pins = self.slots[slot].pins.load(Ordering::Acquire);
                if pins > 0 || self.landing(o) {
                    return Err(format!("slot {slot} owner {o:?} pinned {pins}, read in flight {}", self.landing(o)));
                }
                self.evict(&mut st, slot, current_tick);
            }
            if !self.passed(slot, fence) {
                let pending = self.slots[slot].fence.load(Ordering::Acquire);
                return Err(format!("slot {slot} fenced at generation {pending}, the device at {fence:?}"));
            }
            if !drafter_idle && !self.draft_frontier_clear(slot) {
                return Err(format!(
                    "slot {slot} evicted from layer {} inside the drafter's frontier at {}",
                    self.slots[slot].evicted_layer.load(Ordering::Acquire),
                    self.draft_watermark()
                ));
            }
        }
        st.segs[seg] = Seg::default();
        drop(st);
        self.segments.release(seg).expect("a retiring segment is present");
        Ok(())
    }

    /// A fence above `committed` (the pool's committed count, read once the queue has drained) names work no buffer
    /// will carry; cleared, so a retire does not wait on it. The count cleared.
    pub fn clear_orphan_fences(&self, committed: u64) -> usize {
        let orphans = self.slots.iter().filter(|s| s.fence.load(Ordering::Acquire) > committed);
        orphans.map(|s| s.fence.store(0, Ordering::Release)).count()
    }

    /// The keys' slots unpin in place as carriers, fenced at `gen` (the next completion): no sealed launch tagged them.
    /// A sealed key is pinned, so never evicted: one not resident is a broken invariant, and oscar stops.
    pub fn unseal(&self, keys: &[LayerExpertKey], gen: u64) {
        let mut st = self.lock();
        for &key in keys {
            let Some(slot) = self.resident_slot(key) else {
                panic!("unseal: (L={}, E={}) is not resident, though sealed", key.0, key.1);
            };
            let prev = self.slots[slot].pins.fetch_sub(1, Ordering::AcqRel);
            debug_assert!(prev > 0, "unseal: slot {slot} was not pinned");
            self.slots[slot].fence.store(gen, Ordering::Release);
            let seg = slot / st.per_segment;
            debug_assert!(st.segs[seg].sealed > 0, "unseal: segment {seg} holds no sealed slot");
            st.segs[seg].sealed = st.segs[seg].sealed.saturating_sub(1);
        }
    }

    fn present(&self, st: &State, slot: usize) -> bool {
        slot < self.index && st.segs[slot / st.per_segment].present
    }

    /// The one read for demand and staged fills; `device` picks a replica, a matter of bandwidth, never correctness.
    pub fn read_into(&self, device: u8, slot: usize, extent: ExpertExtent) -> Result<()> {
        let ptr = self.segments.ptr(slot);
        // SAFETY: the slot is this fill's, its GPU and drafter readers waited out; an extent fits a slot.
        Ok(unsafe { self.file.pread(device, extent.shard_id, ptr, extent.byte_count, extent.file_offset) }?)
    }

    /// `slot_count()` u32s, bound by the Draft lane's validation.
    pub fn epoch_buffer(&self) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        self.epochs.buffer()
    }

    /// The Draft lane's recheck stores `layer + 1` here as each layer's matvecs retire.
    pub fn draft_watermark_buffer(&self) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        self.draft_watermark.buffer()
    }

    fn draft_watermark(&self) -> u32 {
        // SAFETY: a UMA u32 the recheck pass stores; a lagging read only defers longer.
        unsafe { self.draft_watermark.ptr().read_volatile() }
    }

    /// The draft-read fence's encode side: layer `hi - 1`'s matvecs encoded, its rows written; on the drafter thread.
    pub fn note_draft_encoded(&self, hi: u32) {
        self.draft_encoded_hi.store(hi, Ordering::Release);
    }

    /// The draft-read fence's retire side: the forward has executed, on the drafter thread after its stream sync.
    pub fn end_draft_forward(&self) {
        self.draft_encoded_hi.store(0, Ordering::Release);
        // SAFETY: a UMA u32; the drafter stream is synced, so no GPU store races this.
        unsafe { self.draft_watermark.ptr().write_volatile(0) };
    }

    /// Racy by design: the kernel-side validation is the authority.
    fn epoch(&self, slot: usize) -> u32 {
        debug_assert!(slot < self.index);
        // SAFETY: the buffer outlives self; slot bounds asserted above.
        unsafe { self.epochs.ptr().add(slot).read_volatile() }
    }

    /// The caller holds the state lock and has not yet written the slot.
    fn bump_epoch(&self, slot: usize) {
        debug_assert!(slot < self.index);
        // SAFETY: the buffer outlives self and every writer holds the state lock; GPU readers only validate.
        unsafe {
            let p = self.epochs.ptr().add(slot);
            p.write_volatile(p.read_volatile().wrapping_add(1));
        }
    }

    /// Idempotent: the first oracle stays.
    pub fn attach_oracle(&self, oracle: Arc<dyn Oracle>) {
        let tracks = oracle.tracks_access();
        let _ = self.oracle.set((oracle, tracks));
    }

    /// For recency-tracking oracles (LRU); stateless ones skip it.
    fn note_access(&self, key: LayerExpertKey, current_tick: u32) {
        if let Some((oracle, true)) = self.oracle.get() {
            oracle.note_access(key, current_tick);
        }
    }

    /// The call's routed keys, which the walk scores as needed now.
    pub fn set_routed(&self, keys: impl IntoIterator<Item = LayerExpertKey>) {
        let mut st = self.lock();
        st.routed.clear();
        st.routed.extend(keys);
    }

    pub fn clear_routed(&self) {
        self.lock().routed.clear();
    }

    /// `UnknownKey` for a key this class does not hold (a mixed-quant model's other layers).
    pub fn extent(&self, key: LayerExpertKey) -> Result<ExpertExtent> {
        self.layout.get(&key).copied().ok_or_else(|| Error::UnknownKey(format!("(L={}, E={})", key.0, key.1)))
    }

    pub fn is_resident(&self, key: LayerExpertKey) -> bool {
        self.word_of(key) >= 0
    }

    /// Unpinned: the slot may be claimed the instant this returns.
    pub fn resident_slot(&self, key: LayerExpertKey) -> Option<usize> {
        usize::try_from(self.word_of(key)).ok()
    }

    /// Unpinned; the word read again after the epoch keeps a claim in between from pairing the slot with its old epoch.
    pub fn observe_resident(&self, key: LayerExpertKey) -> Option<Resident> {
        let slot = self.resident_slot(key)?;
        let epoch = self.epoch(slot);
        (self.resident_slot(key) == Some(slot)).then_some(Resident { slot, epoch })
    }

    fn key_index(&self, key: LayerExpertKey) -> Option<usize> {
        let (l, e) = (key.0 as usize, key.1 as usize);
        (e < self.experts && l * self.experts < self.words.len()).then_some(l * self.experts + e)
    }

    fn word(&self, key: LayerExpertKey) -> Option<&AtomicI32> {
        self.key_index(key).map(|i| &self.words[i])
    }

    /// `ABSENT` for a key outside the tables.
    fn word_of(&self, key: LayerExpertKey) -> i32 {
        self.word(key).map_or(ABSENT, |w| w.load(Ordering::Acquire))
    }

    /// A fill of `key` is in flight: a demand on it waits.
    fn landing(&self, key: LayerExpertKey) -> bool {
        self.word_of(key) < ABSENT
    }

    /// Set `key`'s word, under the lock; a key with an extent is in the table.
    fn mark(&self, key: LayerExpertKey, w: i32) {
        self.word(key).expect("a key with an extent is in the table").store(w, Ordering::Release);
    }

    /// Idempotent: the first caller wins.
    pub fn set_completion_event(&self, event: &Retained<ProtocolObject<dyn MTLSharedEvent>>) {
        let _ = self.event.set(Retained::clone(event));
    }

    /// Fence `slots` at their launch's generation `gen`; tag before unpinning, so no eviction overwrites one first.
    pub fn fence(&self, slots: &[usize], gen: u64) {
        for &s in slots {
            self.slots[s].fence.store(gen, Ordering::Release);
        }
    }

    /// The device's signalled generation; `None` before the first fence.
    fn reached(&self) -> Option<u64> {
        self.event.get().map(|ev| ev.signaledValue())
    }

    /// The device is past `slot`'s fence (an unfenced slot has none), against a `reached` read once per walk.
    fn passed(&self, slot: usize, reached: Option<u64>) -> bool {
        let pending = self.slots[slot].fence.load(Ordering::Acquire);
        pending == 0 || reached.is_none_or(|g| g >= pending)
    }

    /// A staged fill asks and never waits: the GPU may wait on the resolver, whose demand reads need this reader.
    pub(super) fn gpu_safe_now(&self, slot: usize) -> bool {
        self.passed(slot, self.reached())
    }

    /// Waits while the evicted layer is in the drafter's frontier, capped in time so a wedged drafter parks no reader.
    pub(super) fn wait_draft_safe(&self, slot: usize) {
        if self.slots[slot].evicted_layer.load(Ordering::Acquire) == u16::MAX {
            return;
        }
        for _ in 0..6 {
            if self.draft_encoded_hi.load(Ordering::Acquire) == 0 || self.draft_frontier_clear(slot) {
                break;
            }
            std::thread::sleep(Duration::from_micros(500));
        }
    }

    /// The evicted owner's layer is outside what a committed but unexecuted drafter command buffer may read.
    fn draft_frontier_clear(&self, slot: usize) -> bool {
        const FRONTIER: u32 = 6;
        let l = self.slots[slot].evicted_layer.load(Ordering::Acquire);
        let lo = self.draft_watermark();
        l == u16::MAX || (l as u32) < lo || l as u32 >= lo + FRONTIER
    }

    /// Required of a reader between the slot's claim and its read: logged past 1.5 s, past 30 s oscar stops.
    pub(super) fn wait_gpu_safe(&self, slot: usize) {
        let pending = self.slots[slot].fence.load(Ordering::Acquire);
        let Some(ev) = self.event.get().filter(|ev| pending > 0 && ev.signaledValue() < pending) else {
            return;
        };
        let t = Instant::now();
        let short = || {
            let (signaled, ms) = (ev.signaledValue(), t.elapsed().as_millis());
            format!("wait_slot_gpu_safe timed out: slot {slot}, pending {pending}, signalled {signaled}, after {ms} ms")
        };
        if !ev.waitUntilSignaledValue_timeoutMS(pending, 1_500) {
            tracing::error!(target: "oscar::experts", "{}", short());
            assert!(ev.waitUntilSignaledValue_timeoutMS(pending, 28_500), "{}", short());
        }
    }

    /// The present slots `(free, pinned per layer, reading)`; the rest are evictable.
    fn occupancy_locked(&self, st: &State) -> (usize, HashMap<u16, usize>, usize) {
        let (mut free, mut reading) = (0usize, 0usize);
        let mut layers: HashMap<u16, usize> = HashMap::new();
        let present = st.segs.iter().enumerate().filter(|(_, s)| s.present);
        for s in present.flat_map(|(seg, _)| seg * st.per_segment..(seg + 1) * st.per_segment) {
            match st.owners[s].key {
                None => free += 1,
                Some(owner) if self.landing(owner) => reading += 1,
                Some(owner) if self.slots[s].pins.load(Ordering::Acquire) > 0 => {
                    *layers.entry(owner.0).or_default() += 1
                }
                Some(_) => {}
            }
        }
        (free, layers, reading)
    }

    /// `[present, free, pinned, reading]`, for the memory trace.
    pub fn occupancy(&self) -> [usize; 4] {
        let st = self.lock();
        let (free, layers, reading) = self.occupancy_locked(&st);
        [self.capacity(), free, layers.values().sum(), reading]
    }

    /// Once per pool: slots held by running work and slots waiting on their read want different fixes.
    fn report_wedge(&self, st: &State, current_tick: u32) {
        if self.wedge_reported.swap(true, Ordering::Relaxed) {
            return;
        }
        let (free, layers, reading) = self.occupancy_locked(st);
        let running: usize = layers.values().sum();
        tracing::info!(target: "oscar::experts",
            "held up - {} slots ({:?}, {} layers x {} experts) at step {current_tick}: \
             {free} free, {running} in use over {} layers (widest {}), {reading} waiting on their own \
             read; re-checking as slots are released",
            self.capacity(),
            self.dtype,
            self.layout.len() / self.experts.max(1),
            self.experts,
            layers.len(),
            layers.values().copied().max().unwrap_or(0),
        );
    }

    /// A timed wait, as an unpin is lock-free and never notifies; callers cap it, so a failure, never a hang.
    fn stall<'a>(&self, st: MutexGuard<'a, State>, current_tick: u32) -> MutexGuard<'a, State> {
        self.report_wedge(&st, current_tick);
        self.cv.notify_all();
        self.cv.wait_timeout(st, Duration::from_millis(5)).expect("HotPoolBank state lock poisoned").0
    }

    /// Every per-claim field set, so nothing of an earlier owner's (a staged promise, a prefetched tag) carries over.
    fn claim(&self, st: &mut State, slot: usize, key: LayerExpertKey, current_tick: u32, promise: u32) {
        self.bump_epoch(slot);
        let s = &self.slots[slot];
        s.claimed_at.store(current_tick, Ordering::Release);
        s.promise.store(promise, Ordering::Release);
        s.prefetched.store(false, Ordering::Release);
        st.assign(slot, Some(key));
    }

    /// The one eviction path; the word cleared only if it points here (a duplicate's points elsewhere).
    fn evict(&self, st: &mut State, slot: usize, current_tick: u32) {
        let Some(key) = st.owners[slot].key else { return };
        st.assign(slot, None);
        let s = &self.slots[slot];
        COUNTERS.evicted.fetch_add(1, Ordering::Relaxed);
        s.evicted_layer.store(key.0, Ordering::Release);
        let _ = self.word(key).map(|w| w.compare_exchange(slot as i32, ABSENT, Ordering::AcqRel, Ordering::Relaxed));
        s.stamp.store(-1, Ordering::Release);
        // An unconsumed staged fill is wasted, churn if evicted by its promised tick.
        if s.prefetched.load(Ordering::Acquire) {
            COUNTERS.wasted_prefetch.fetch_add(1, Ordering::Relaxed);
            let promise = s.promise.load(Ordering::Acquire);
            if promise != u32::MAX && current_tick <= promise {
                COUNTERS.wasted_churn.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
