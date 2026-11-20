//! The bank's lifecycle, a test per behaviour.

use super::fill::{pin_all, pin_all_blocking, unpin_all, Pick, PinOrder, Plan};
use super::*;
use crate::pool::reads::{FetchQueue, Lane, Readers};
use crate::pool::COUNTERS;
use std::collections::HashMap;
use std::path::PathBuf;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;
use oscar_offload::io::TieredFile;

use crate::gpu::ExpertExtent;
use crate::pool::{BankDtype, LayerExpertKey};

/// Every segment's `(present, sealed slots, retiring)`.
fn states(bank: &Bank) -> Vec<(bool, usize, bool)> {
    bank.segments().iter().map(|s| (s.present, s.sealed as usize, s.retiring)).collect()
}

/// The free `slot` claimed for the absent `key` as a demand or a stage claims it: the key in flight there.
fn begin_fetch(bank: &Bank, slot: usize, key: LayerExpertKey) {
    let mut st = bank.lock();
    assert!(bank.present(&st, slot) && st.owners[slot].key.is_none() && bank.word_of(key) == ABSENT, "{key:?}");
    bank.slots[slot].evicted_layer.store(u16::MAX, Ordering::Release);
    bank.claim(&mut st, slot, key, 0, u32::MAX);
    bank.mark(key, filling(slot));
}

/// `key` filled into the free `slot` and published, the slot's bytes as they stand.
fn resident(bank: &Bank, slot: usize, key: LayerExpertKey) {
    begin_fetch(bank, slot, key);
    bank.publish(slot, key, false).expect("the publish");
}

/// As [`resident`], every byte of the slot `byte`, as a reader's read writes it.
fn resident_with(bank: &Bank, slot: usize, key: LayerExpertKey, byte: u8) {
    begin_fetch(bank, slot, key);
    // SAFETY: the fill owns the slot and this thread writes it alone.
    unsafe { bank.pool().ptr(slot).write_bytes(byte, bank.bytes_per_expert()) };
    bank.publish(slot, key, false).expect("the publish");
}

fn slot_holds(bank: &Bank, slot: usize, byte: u8) -> bool {
    // SAFETY: the slot is present and this thread reads it alone.
    let bytes = unsafe { std::slice::from_raw_parts(bank.pool().ptr(slot), bank.bytes_per_expert()) };
    bytes.iter().all(|&b| b == byte)
}

fn plan_one(bank: &Bank, key: LayerExpertKey) -> Result<Plan> {
    Ok(bank.plan(&[key], 0)?.remove(0))
}

/// A static next-use map: deterministic Belady.
struct StaticOracle(HashMap<LayerExpertKey, u32>);
impl Oracle for StaticOracle {
    fn eviction_score(&self, key: LayerExpertKey, _tick: u32) -> u32 {
        self.0.get(&key).copied().unwrap_or(u32::MAX)
    }
}

/// Sealed slots are pinned, filled and published from the low end; a seal takes no slot the device has not passed,
/// nor does a claim, which never takes a routed fill no launch has read (the wave about to pin it would refill it
/// under the same walk, twice at most, then stop oscar): with no other candidate the claim is a stall.
#[test]
fn a_seal_pins_fills_and_publishes_and_takes_no_slot_the_device_has_not_passed() {
    let _g = test_lock();
    let c0 = COUNTERS.now();
    // Two segments of three slots; layers 1 and 2 of two experts seal.
    let (device, bank, keys) = bank_over_file(2, 3, 3, 2, 0xA0, 8);
    let keys = &keys[2..];
    assert_eq!((bank.slot_count(), bank.capacity()), (6, 0), "an index space, nothing present");
    bank.admit(&device, 0).unwrap();
    bank.admit(&device, 1).unwrap();
    assert!(bank.admit(&device, 1).is_err(), "a present segment is not admitted twice");
    assert_eq!((bank.capacity(), bank.lock().free.len()), (6, 6), "every slot of an admitted segment is free");
    bank.seal(&keys[..2], 0).expect("seal layer 1");
    bank.seal(&keys[2..], 0).expect("seal layer 2");
    let pins = |s: usize| bank.slots[s].pins.load(Ordering::Acquire);
    for (i, &key) in keys.iter().enumerate() {
        let slot = bank.resident_slot(key).expect("a sealed key is resident");
        assert_eq!(slot, i, "the sealed slots are taken from the low end, in key order");
        assert!(pins(slot) >= 1 && slot_holds(&bank, slot, 0xA2 + i as u8), "slot {slot}: pinned, its extent's bytes");
    }
    let spanned = vec![(true, 3, false), (true, 1, false)];
    assert_eq!(states(&bank), spanned, "layer 2 spans the segments: one every slot sealed, one with a sealed slot");
    let (bytes, declared) = (bank.pool().bytes(), bank.pool().declared().len());
    assert_eq!((bytes, declared), (6 * oscar_offload::segments::padded_unit(8), 2), "present slots x the padded unit");
    // Unsealed experts in the low region seal again in place: nothing read or evicted.
    bank.unseal(&keys[..2], 3);
    assert_eq!(states(&bank)[0], (true, 1, false));
    bank.seal(&keys[..2], 0).expect("seal layer 1 again");
    assert_eq!((bank.resident_slot(keys[0]), bank.resident_slot(keys[1]), pins(0)), (Some(0), Some(1), 1));
    assert_eq!((states(&bank)[0], COUNTERS.now().evicted), ((true, 3, false), c0.evicted), "in place, nothing evicted");
    let mut st = bank.lock();
    assert_eq!(st.free.len(), 2);
    for _ in 0..2 {
        let slot = bank.acquire(&mut st, 0).expect("a free carrier slot");
        assert!(slot >= 4, "slot {slot}");
        st.assign(slot, Some((0, slot as u16)));
        bank.slots[slot].pins.fetch_add(1, Ordering::AcqRel);
    }
    assert!(matches!(bank.acquire(&mut st, 0), Err(Error::Exhausted { hot: 6 })), "sealed slots are never victims");
    drop(st);
    let (device, all_sealed, keys) = bank_over_file(1, 2, 1, 2, 0xC0, 8);
    all_sealed.admit(&device, 0).unwrap();
    all_sealed.seal(&keys, 0).expect("seal all");
    let one = all_sealed.resident_slot((0, 1)).is_some();
    assert_eq!((one, all_sealed.capacity(), states(&all_sealed)), (true, 2, vec![(true, 2, false)]));
    // After the queue drained, a fence above the committed count names unsubmitted work: cleared; one at it stays.
    let (device, bank, keys) = bank_over_file(1, 3, 3, 2, 0xA0, 8);
    bank.fence(&[0], 5);
    bank.fence(&[1], 7);
    assert_eq!(bank.clear_orphan_fences(5), 1, "one fence above the count");
    let fence = |s: usize| bank.slots[s].fence.load(Ordering::Acquire);
    assert_eq!((fence(0), fence(1), fence(2)), (5, 0, 0), "the one at the count stays, the orphan is cleared");
    // Layer 0's experts in slots 0 and 1, slot 0 fenced ahead of the event.
    let event = objc2_metal::MTLDevice::newSharedEvent(&*device).expect("a shared event");
    bank.set_completion_event(&event);
    bank.admit(&device, 0).unwrap();
    resident(&bank, 0, keys[0]);
    resident(&bank, 1, keys[1]);
    bank.fence(&[0], 1);
    bank.seal(&keys[2..4], 0).expect("seal layer 1");
    let sealed = (bank.resident_slot(keys[2]), bank.resident_slot(keys[3]));
    assert_eq!(sealed, (Some(1), Some(2)), "the fenced slot passed over");
    assert!(bank.is_resident(keys[0]), "its resident kept");
    // Every takeable slot fenced ahead: the seal stalls to its cap.
    bank.unseal(&keys[2..4], 0);
    bank.fence(&[1, 2], 1);
    bank.set_stall_cap_ms(30);
    let stalled = bank.seal(&keys[4..6], 0);
    assert!(matches!(stalled, Err(Error::Exhausted { .. })), "no wait on a fence the seal's own keys may hold");
    event.setSignaledValue(1);
    bank.seal(&keys[4..6], 0).expect("seal layer 2 once the device has passed the slots");
    assert!(bank.is_resident(keys[4]) && bank.is_resident(keys[5]));
    // Slot 0 a wave the GPU still reads (fenced ahead of the event), slot 1 the next wave's landed key, slot 2 a stranger.
    let (device, bank, keys) = bank_over_file(1, 3, 3, 2, 0xA0, 8);
    let event = objc2_metal::MTLDevice::newSharedEvent(&*device).expect("a shared event");
    bank.set_completion_event(&event);
    bank.admit(&device, 0).unwrap();
    keys[..3].iter().enumerate().for_each(|(slot, &key)| resident(&bank, slot, key));
    bank.fence(&[0], 1);
    bank.set_routed(keys[..2].iter().copied());
    assert_eq!(bank.acquire(&mut bank.lock(), 0).expect("a victim"), 2, "the stranger first");
    resident(&bank, 2, keys[3]);
    bank.set_routed([keys[0], keys[1], keys[3]]);
    assert_eq!(bank.acquire(&mut bank.lock(), 0).expect("a victim"), 0, "the fenced slot before the landed fill");
    assert!(bank.is_resident(keys[1]) && bank.is_resident(keys[3]), "the call's fills stand");
    // The freed slot refilled: a publish clears the fence the slot carried, so the fill is the call's own.
    resident(&bank, 0, keys[4]);
    assert_eq!(bank.slots[0].fence.load(Ordering::Acquire), 0, "no launch has read the new fill");
    bank.set_routed([keys[1], keys[3], keys[4]]);
    let err = bank.acquire(&mut bank.lock(), 0).unwrap_err();
    assert!(matches!(err, Error::Exhausted { .. }), "nothing but the call's own fills: a stall, got {err:?}");
    bank.clear_routed();
    assert!(bank.acquire(&mut bank.lock(), 0).is_ok_and(|s| s < 3), "no longer the call's: a victim again");
}

/// A borrowed segment holds carriers only and goes back released; a seal takes an expert resident elsewhere from
/// its copy, reading nothing, and keeps a key only where it is published: a stale copy's slot takes a copy, and an
/// unpublished owner is no duplicate until its key is published.
#[test]
fn segments_retire_free_unseal_seal_and_borrow_at_runtime() {
    let _g = test_lock();
    let c0 = COUNTERS.now();
    let evicted = || COUNTERS.now().evicted - c0.evicted;
    // Three segments of two slots; three layers of two experts.
    let (device, bank, keys) = bank_over_file(3, 2, 3, 2, 0xB0, 8);
    assert!((0..3).all(|seg| bank.admit(&device, seg).is_ok()), "every segment admitted");
    bank.seal(&keys[..2], 0).expect("seal layer 0");
    assert_eq!(states(&bank), vec![(true, 2, false), (true, 0, false), (true, 0, false)]);
    resident(&bank, 2, (1, 0));
    resident(&bank, 3, (1, 1));
    bank.slots[3].pins.fetch_add(1, Ordering::AcqRel);
    // Retire segment 1: slot 2 evicts now, the pinned slot 3 stays.
    bank.retire(1, 7);
    let holds = |l, e| bank.is_resident((l, e));
    assert_eq!((holds(1, 0), holds(1, 1), states(&bank)[1]), (false, true, (true, 0, true)), "pinned stays");
    let mut st = bank.lock();
    assert!(st.free.iter().all(|&s| s >= 4), "a retiring segment's free slots left the list");
    let slot = bank.acquire(&mut st, 7).expect("a claim");
    assert!(slot >= 4, "a claim never lands in a retiring segment, got {slot}");
    st.assign(slot, Some((0, 0)));
    drop(st);
    assert!(bank.free(1, 7, true).is_err(), "not free while a slot is pinned");
    bank.slots[3].pins.fetch_sub(1, Ordering::AcqRel);
    // The evicted layer is in the drafter's frontier: held unless idle.
    bank.note_draft_encoded(2);
    assert!(bank.free(1, 7, false).is_err(), "held by the drafter's frontier");
    assert!(bank.free(1, 7, true).is_ok(), "free once the pins are gone and the drafter idle");
    let shape = (states(&bank)[1], bank.capacity(), bank.pool().declared().len());
    assert_eq!((holds(1, 1), shape), (false, ((false, 0, false), 4, 2)));
    bank.end_draft_forward();
    // Unseal layer 0: unpinned in place, fenced, resident, victims.
    bank.unseal(&keys[..2], 5);
    let (pins, fence) = (bank.slots[0].pins.load(Ordering::Acquire), bank.slots[0].fence.load(Ordering::Acquire));
    assert_eq!((pins, fence, holds(0, 0) && holds(0, 1), states(&bank)[0]), (0, 5, true, (true, 0, false)));
    // Seal layer 1 with (1,0) a carrier copy in slot 4 and (1,1) in flight to slot 5: it waits, evicts, copies.
    bank.lock().assign(4, None);
    resident_with(&bank, 4, (1, 0), 0xB2);
    begin_fetch(&bank, 5, (1, 1));
    let sealer = std::thread::scope(|s| {
        let h = s.spawn(|| bank.seal(&keys[2..4], 9));
        std::thread::sleep(Duration::from_millis(30));
        assert!(!h.is_finished(), "the seal waits for the fetch in flight");
        // SAFETY: the fetch owns the slot and this thread writes it alone; the seal reads nothing of it yet.
        unsafe { bank.pool().ptr(5).write_bytes(0xB3, 8) };
        bank.publish(5, (1, 1), false).unwrap();
        h.join().unwrap()
    });
    assert_eq!(sealer.expect("seal layer 1"), (2, 0), "both copied from the carrier copies, none read");
    assert_eq!((bank.resident_slot((1, 0)), bank.resident_slot((1, 1))), (Some(0), Some(1)));
    assert!(!holds(0, 0) && !holds(0, 1) && slot_holds(&bank, 1, 0xB3), "evicted for the seal, its bytes held");
    assert_eq!(states(&bank), vec![(true, 2, false), (false, 0, false), (true, 0, false)]);
    assert_eq!(evicted(), 2 + 2, "the retire's and the free's one each, the seal's two");
    let owners: Vec<_> = bank.lock().owners[4..].iter().map(|o| o.key).collect();
    assert_eq!(owners, [Some((1, 0)), Some((1, 1))], "the copies stay owned");
    // Every key needed soon: only the duplicate rule makes the copies victims.
    let soon = [((0, 0), 1), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1), ((2, 0), 1), ((2, 1), 1)].into();
    bank.attach_oracle(Arc::new(StaticOracle(soon)));
    let st = bank.lock();
    assert!(st.free.is_empty());
    let victim = bank.pick(&st, 9, None);
    assert!(matches!(victim, Some(Pick::Victim { slot: 4 | 5, score: u32::MAX })), "a duplicate goes first");
    drop(st);
    // Seal layer 2 into the duplicates' slots 4 and 5.
    bank.seal(&keys[4..], 11).expect("seal layer 2");
    assert_eq!((bank.resident_slot((2, 0)), bank.resident_slot((2, 1))), (Some(4), Some(5)));
    assert!(holds(1, 0) && holds(1, 1) && slot_holds(&bank, 4, 0xB4), "the sealed copies stay published");
    assert_eq!(states(&bank), vec![(true, 2, false), (false, 0, false), (true, 2, false)]);
    assert_eq!(evicted(), 4 + 2, "and the duplicates for the second seal");
    let mut st = bank.lock();
    assert!(matches!(bank.acquire(&mut st, 9), Err(Error::Exhausted { hot: 4 })), "every present slot is sealed");
    drop(st);
    use oscar_offload::segments::{Addressing, Lender, SegmentPool, PAGE};
    // Three segments of two slots; three layers of two experts.
    let (device, bank, keys) = bank_over_file(3, 2, 3, 2, 0xC0, 8);
    bank.set_stall_cap_ms(50);
    let region = SegmentPool::new(&device, PAGE, 1, 8, Addressing::Whole).expect("the lender");
    region.take(&device, 0, 8, true).unwrap();
    let lender = Lender {
        buffer: region.buffer(0).expect("the lender's buffer"),
        base: std::ptr::NonNull::new(region.ptr(0).cast()).expect("the lender's mapping"),
        bytes: 8 * PAGE,
        queue: 5,
    };
    bank.admit(&device, 0).unwrap();
    bank.adopt(2, &lender, 4 * PAGE).expect("the top segment, inside the lender");
    assert!(bank.adopt(2, &lender, 6 * PAGE).is_err(), "never over a present segment");
    assert!(bank.admit(&device, 2).is_err(), "nor admitted over");
    assert_eq!((bank.capacity(), borrowed(&bank), states(&bank)[2]), (4, vec![2], (true, 0, false)));
    assert_eq!(bank.pool().address(4), lender.buffer.gpuAddress() + (4 * PAGE) as u64);
    let declared_on = |queue| bank.pool().declared_on(queue).len();
    assert_eq!((bank.pool().declared().len(), declared_on(5), declared_on(0)), (1, 2, 1));
    // A carrier's fetch lands in the lender's bytes.
    resident_with(&bank, 4, (1, 0), 0xC7);
    // SAFETY: the lender's segment is present.
    let lent = || unsafe { std::slice::from_raw_parts(region.ptr(4), 8) }.iter().all(|&b| b == 0xC7);
    assert!(lent(), "the slot is a range of the lender's mapping");
    // Layer 0 seals into the pool's own slots; layer 2 finds them pinned, refused beside a free borrowed slot.
    bank.seal(&keys[..2], 0).expect("seal layer 0");
    assert_eq!((bank.resident_slot((0, 0)), bank.resident_slot((0, 1))), (Some(0), Some(1)));
    let refused = bank.seal(&keys[4..], 1);
    assert!(matches!(refused, Err(Error::Exhausted { .. })), "a seal never lands in a borrowed segment");
    assert_eq!(states(&bank)[2], (true, 0, false));
    bank.retire(2, 3);
    assert!(!bank.is_resident((1, 0)), "its carrier evicted at the retire");
    assert!(bank.free(2, 3, true).is_ok());
    assert_eq!((bank.capacity(), borrowed(&bank), states(&bank)[2]), (2, vec![], (false, 0, false)));
    assert_eq!((bank.pool().address(4), declared_on(5), lent()), (0, 1, true), "the lender's bytes stand");
    bank.adopt(2, &lender, 4 * PAGE).expect("and adopted again, fresh");
    assert!(!bank.is_resident((1, 0)));
    // Two segments of three slots; two layers of three experts; segment 1 admitted first, so the later admit's lowest
    // slot is taken first: layer 1's experts land high, layer 0's holding the low segment.
    let (device, bank, keys) = bank_over_file(2, 3, 2, 3, 0xE0, 8);
    bank.admit(&device, 1).unwrap();
    bank.admit(&device, 0).unwrap();
    keys[..3].iter().for_each(|&key| _ = fill(&bank, key, 1));
    assert!(keys[3..].iter().all(|&key| fill(&bank, key, 1) >= 3), "layer 1's experts land high");
    std::fs::write(bank.file.path(), b"").expect("empty the model file");
    let sealed = bank.seal(&keys[3..], 2).expect("every expert resident: nothing read");
    assert_eq!(sealed, (3, 0), "three copied, none read");
    for (i, &key) in keys[3..].iter().enumerate() {
        let slot = bank.resident_slot(key).expect("sealed");
        assert!(slot < 3 && slot_holds(&bank, slot, 0xE3 + i as u8), "slot {slot}: the low end, its expert's bytes");
    }
    let _ = std::fs::remove_file(bank.file.path());
    let (device, bank, keys) = bank_over_file(1, 3, 1, 2, 0xC0, 8);
    bank.admit(&device, 0).unwrap();
    // (0,0) published in slot 2 and a stale copy owned in slot 0; (0,1) in slot 1.
    resident(&bank, 2, keys[0]);
    resident(&bank, 1, keys[1]);
    bank.lock().assign(0, Some(keys[0]));
    assert_eq!(bank.seal(&keys, 0).expect("seal layer 0"), (1, 0), "the stale copy's slot takes a copy");
    let slot = bank.resident_slot(keys[0]).expect("published");
    assert!(bank.slots[slot].pins.load(Ordering::Acquire) > 0, "the published slot is the sealed one");
    let bank = bank_of(3, 3, []);
    resident(&bank, 0, (0, 0));
    resident(&bank, 1, (0, 1));
    bank.lock().assign(2, Some((0, 2)));
    bank.attach_oracle(Arc::new(StaticOracle([((0, 0), 1), ((0, 1), 1), ((0, 2), 1)].into())));
    let pick = || bank.pick(&bank.lock(), 5, None);
    let victim = pick();
    assert!(matches!(victim, Some(Pick::Victim { score: 1, .. })), "the oracle's score, not a duplicate's: {victim:?}");
    // Published elsewhere: slot 2's copy is a duplicate.
    let mut st = bank.lock();
    st.assign(2, None);
    st.assign(2, Some((0, 0)));
    drop(st);
    assert!(matches!(pick(), Some(Pick::Victim { slot: 2, score: u32::MAX })));
}

/// Every score finite, so the walk keeps its runners-up (without the keep the reserve holds 0, not 6); a reserved
/// victim is revalidated before use: a live fill, a pinned slot, a promise before its tick is skipped; equal scores
/// take the oldest fill; a fill is in flight till its publish or abort; a stage swaps only for a sooner need.
#[test]
fn a_full_ring_walk_serves_the_claims_that_follow_each_victim_revalidated() {
    let _g = test_lock();
    // Ten keys, eight slots, two claims that must evict.
    let bank = bank_of(8, 10, 0..10);
    bank.attach_oracle(Arc::new(StaticOracle((0..10).map(|e| ((0, e), 10 * (e as u32 + 1))).collect())));
    (0..8).for_each(|e| resident(&bank, e as usize, (0, e)));
    let st = bank.lock();
    assert!(st.free.is_empty(), "a full pool: every claim must sweep");
    assert!(st.runners_up.is_empty(), "nothing ranked yet");
    drop(st);
    let plans = bank.plan(&[(0, 8), (0, 9)], 0).expect("batch");
    let top_two = matches!(plans[..], [Plan::Reserved { slot: 7, .. }, Plan::Reserved { slot: 6, .. }]);
    assert!(top_two, "the walk's top two, in rank order, as a fresh walk answers: {plans:?}");
    assert_eq!(bank.lock().runners_up.len(), 6, "eight ranked: one taken by the walk, one popped by the second claim");
    let bank = bank_of(4, 5, []);
    bank.attach_oracle(Arc::new(StaticOracle((0..5).map(|e| ((0, e), if e == 4 { 90 } else { 10 })).collect())));
    [(2, 2), (0, 0), (1, 1), (3, 3)].into_iter().for_each(|(slot, e)| resident(&bank, slot, (0, e)));
    let st = bank.lock();
    let take = |mut kept: Vec<(u32, u32)>, tick| bank.take_runner_up(&st, &mut kept, tick);
    // Best last: the pop takes it, and only it.
    let mut kept = vec![(0, 10), (3, 40)];
    assert_eq!(bank.take_runner_up(&st, &mut kept, 0), Some((3, 40)));
    assert_eq!(kept.len(), 1, "only the served entry leaves the reserve");
    assert_eq!(take(vec![], 0), None);
    bank.mark((0, 1), filling(1));
    assert_eq!(take(vec![(0, 10), (1, 20)], 0), Some((0, 10)), "skips the live fill");
    bank.mark((0, 1), 1);
    bank.slots[2].pins.fetch_add(1, Ordering::Release);
    assert_eq!(take(vec![(0, 10), (2, 30)], 0), Some((0, 10)), "skips the pinned slot");
    bank.slots[2].pins.fetch_sub(1, Ordering::Release);
    // A promise ahead is never served, one past is ordinary, and a deadline without the tag is no promise.
    bank.slots[2].promise.store(20, Ordering::Release);
    bank.slots[2].prefetched.store(true, Ordering::Release);
    assert_eq!(take(vec![(0, 10), (2, 30)], 5), Some((0, 10)), "skips the live promise");
    assert_eq!(take(vec![(0, 10), (2, 30)], 25), Some((2, 30)), "past the promise it is ordinary");
    bank.slots[2].prefetched.store(false, Ordering::Release);
    let consumed = take(vec![(0, 10), (2, 30)], 5);
    assert_eq!(consumed, Some((2, 30)), "a deadline left behind by a consumed stage is not a promise");
    drop(st);
    assert_eq!(bank.acquire(&mut bank.lock(), 0).expect("a victim"), 2, "the oldest fill wins");
    assert!(!bank.is_resident((0, 2)), "and its owner evicted");
    // The farthest needed, staged into the freed slot with a promise at tick 20: the victim only once it passes.
    let staged = bank.stage((0, 4), 0, 20).expect("staged");
    bank.publish(staged, (0, 4), true).expect("the publish");
    let victim = |tick| match bank.pick(&bank.lock(), tick, None) {
        Some(Pick::Victim { slot, .. }) => slot,
        other => panic!("expected a victim, got {other:?}"),
    };
    assert_ne!(victim(5), staged, "a live promise must not be the victim");
    assert_eq!(victim(25), staged, "evictable once the promise has passed");
    // Without the in-flight mark the walk takes a slot a read will write; a mis-sequenced publish or abort is refused.
    let bank = bank_of(4, 5, []);
    bank.attach_oracle(Arc::new(StaticOracle([((0, 0), 1), ((0, 1), 1), ((0, 2), 1)].into())));
    let refused = |r: Result<()>| matches!(r, Err(Error::Lifecycle(_)));
    begin_fetch(&bank, 0, (0, 4));
    assert!(bank.landing((0, 4)), "the fill marks its key");
    bank.abort(0, (0, 4)).expect("the abort");
    let freed = bank.lock().owners[0].key.is_none() && !bank.landing((0, 4)) && !bank.is_resident((0, 4));
    assert!(freed && refused(bank.publish(0, (0, 4), false)), "an abort frees the slot, a publish on it is refused");
    // A full pool, its last slot in flight.
    (0..3).for_each(|e| resident(&bank, e as usize, (0, e)));
    begin_fetch(&bank, 3, (0, 3));
    let in_flight = || (bank.lock().owners[3].key, bank.landing((0, 3)), bank.resident_slot((0, 3)));
    assert_eq!(in_flight(), (Some((0, 3)), true, None), "a fill in flight is not published");
    assert!(bank.lock().free.is_empty(), "no free slot, so the walk must sweep");
    let pick = || bank.pick(&bank.lock(), 0, None);
    assert!(matches!(pick(), Some(Pick::Victim { slot: 0..=2, .. })), "a live fill is never the victim");
    let wrong_key = refused(bank.publish(3, (0, 4), false)) && refused(bank.abort(3, (0, 4)));
    assert!(wrong_key, "a publish and an abort with the wrong key");
    assert_eq!(in_flight(), (Some((0, 3)), true, None), "the fill stands");
    bank.publish(3, (0, 3), false).expect("the publish");
    assert_eq!((bank.landing((0, 3)), bank.resident_slot((0, 3))), (false, Some(3)), "a completed fill publishes");
    let victim = pick();
    assert!(matches!(victim, Some(Pick::Victim { slot: 3, .. })), "the highest score, wherever it sits: {victim:?}");
    let c0 = COUNTERS.now();
    let evicted = || COUNTERS.now().evicted - c0.evicted;
    let bank = bank_of(4, 8, [1, 2, 3]);
    resident(&bank, 0, (0, 1));
    let slot = bank.stage((0, 3), 0, 5).expect("staged");
    assert!(slot != 0 && bank.landing((0, 3)) && !bank.is_resident((0, 3)), "a free slot, in flight till the publish");
    assert_eq!((evicted(), bank.lock().owners[slot].key), (0, Some((0, 3))), "a free slot: no eviction");
    assert_eq!((bank.stage((0, 1), 0, 5), bank.stage((0, 3), 0, 5)), (None, None), "resident, in flight");
    // A full pool: (0,1) next used at 10, (0,2) at 50.
    let bank = bank_of(2, 8, [1, 2, 3]);
    bank.attach_oracle(Arc::new(StaticOracle([((0, 1), 10), ((0, 2), 50)].into())));
    resident(&bank, 0, (0, 1));
    resident(&bank, 1, (0, 2));
    assert_eq!(bank.stage((0, 3), 0, 80), None, "no victim is needed later than the deadline");
    assert!(bank.is_resident((0, 2)) && evicted() == 0, "the sooner-needed victim stays resident");
    assert_eq!(bank.stage((0, 3), 0, 20), Some(1), "the later-needed victim's slot");
    assert!(!bank.is_resident((0, 2)) && bank.landing((0, 3)));
    assert_eq!(evicted(), 1);
}

/// A wait blocks on its own keys alone until each publishes or aborts; a pin holds its key's slot, counts a staged fill
/// saved once and refills a missing key through the readers; a failed resolve rolls back every pin and counts no hit.
#[test]
fn a_demand_plan_answers_resident_waiting_unknown_and_reserved_and_a_batch_waits_for_its_keys_and_pins() {
    let _g = test_lock();
    let bank = bank_of(4, 4, [1, 2, 3]);
    resident(&bank, 2, (0, 1));
    assert!(matches!(plan_one(&bank, (0, 1)), Ok(Plan::Resident(2))));
    begin_fetch(&bank, 0, (0, 2));
    assert!(matches!(plan_one(&bank, (0, 2)), Ok(Plan::Waiting)));
    bank.abort(0, (0, 2)).expect("the abort");
    assert!(matches!(plan_one(&bank, (0, 9)), Err(Error::UnknownKey(_))));
    let Ok(Plan::Reserved { slot, .. }) = plan_one(&bank, (0, 3)) else { panic!("expected Reserved") };
    bank.publish(slot, (0, 3), true).expect("the publish");
    assert!(bank.is_resident((0, 3)));
    let bank = bank_of(1, 4, [1]);
    resident(&bank, 0, (0, 0));
    assert!(matches!(plan_one(&bank, (0, 1)), Ok(Plan::Reserved { slot: 0, .. })), "the one slot's owner evicted");
    assert!(!bank.is_resident((0, 0)));
    // Running work releases its pins at its commit, so an exhausted pool is a stall, not a failure, until the cap;
    // the claims a pass made stand through it, as their reads are in flight.
    let bank = bank_of(2, 8, [1, 2]);
    resident(&bank, 0, (0, 0));
    bank.slots[0].pins.fetch_add(1, Ordering::AcqRel);
    // Slot 0 pinned, slot 1 free, no time to stall: the second cold key finds no victim, the first's claim stands.
    bank.set_stall_cap_ms(0);
    let err = bank.plan(&[(0, 1), (0, 2)], 0).unwrap_err();
    assert!(matches!(err, Error::Exhausted { .. }), "got {err:?}");
    assert!(bank.landing((0, 1)), "the first key's reservation stands, its read in flight");
    assert_eq!(bank.lock().owners[1].key, Some((0, 1)));
    assert!(matches!(plan_one(&bank, (0, 1)), Ok(Plan::Waiting)));
    bank.abort(1, (0, 1)).expect("the abort");
    // Both slots pinned: the batch waits for the pin a commit drops.
    resident(&bank, 1, (0, 3));
    bank.slots[1].pins.fetch_add(1, Ordering::AcqRel);
    bank.set_stall_cap_ms(10_000);
    let plans = std::thread::scope(|s| {
        s.spawn(|| {
            std::thread::sleep(Duration::from_millis(30));
            bank.slots[1].pins.fetch_sub(1, Ordering::AcqRel);
        });
        bank.plan(&[(0, 1)], 0).expect("stall must resolve")
    });
    assert!(matches!(plans[..], [Plan::Reserved { slot: 1, .. }]), "got {plans:?}");
    assert!(!bank.is_resident((0, 3)), "the unpinned slot is the one taken");
    // The wait.
    let bank = bank_of(4, 4, 0..4);
    resident(&bank, 3, (0, 2));
    let reserve = |key| match plan_one(&bank, key) {
        Ok(Plan::Reserved { slot, .. }) => slot,
        other => panic!("expected Reserved, got {other:?}"),
    };
    let waited = |keys: &[LayerExpertKey]| {
        let t0 = Instant::now();
        bank.wait(keys, Lane::Engine);
        t0.elapsed()
    };
    let fast = waited(&[(0, 0), (0, 1), (0, 2)]);
    let (slot0, slot1) = (reserve((0, 0)), reserve((0, 1)));
    assert!(bank.landing((0, 1)), "another key is in flight");
    let lock_free = waited(&[(0, 2)]);
    assert!(fast.max(lock_free) < Duration::from_millis(5), "the fast path took {fast:?}, lock-free {lock_free:?}");
    let waiter = |keys: &'static [LayerExpertKey]| {
        let bank = Arc::clone(&bank);
        std::thread::spawn(move || bank.wait(keys, Lane::Engine))
    };
    let w = waiter(&[(0, 0), (0, 1)]);
    std::thread::sleep(Duration::from_millis(20));
    assert!(!w.is_finished(), "the waiter returns only once its fills complete");
    bank.publish(slot0, (0, 0), true).expect("the publish");
    std::thread::sleep(Duration::from_millis(10));
    assert!(!w.is_finished(), "still one key in flight");
    bank.publish(slot1, (0, 1), true).expect("the publish");
    w.join().expect("the waiter");
    assert!(bank.is_resident((0, 0)) && bank.is_resident((0, 1)));
    let slot = reserve((0, 3));
    let w = waiter(&[(0, 3)]);
    std::thread::sleep(Duration::from_millis(20));
    assert!(!w.is_finished());
    bank.abort(slot, (0, 3)).expect("the abort");
    w.join().expect("the waiter");
    assert!(!bank.is_resident((0, 3)));
    // The pins.
    let bank = bank_of(8, 4, []);
    (0..3).for_each(|e| resident(&bank, 5 + e as usize, (0, e)));
    let pins = |slot: usize| bank.slots[slot].pins.load(Ordering::Acquire);
    let pinned: Vec<usize> = [0, 1, 2, 1].iter().map(|&e| bank.try_pin((0, e), 0).expect("resident")).collect();
    assert_eq!(pinned, [5, 6, 7, 6]);
    assert_eq!(pins(6), 2, "a repeated key is pinned once per input");
    bank.unpin(&pinned);
    assert!([5, 6, 7].into_iter().all(|s| pins(s) == 0), "every pin round-trips to zero");
    // The slot-reuse race: (0,3)'s word points at slot 5, whose stamp says (0,0).
    bank.mark((0, 3), 5);
    assert_eq!((bank.try_pin((0, 3), 0), pins(5)), (None, 0), "the stamp guard rejects the stale pointer");
    let c0 = COUNTERS.now();
    let saves = || COUNTERS.now().prefetch_saves - c0.prefetch_saves;
    let pin = || bank.try_pin((0, 0), 0).inspect(|&slot| bank.unpin(&[slot]));
    assert_eq!((pin(), saves()), (Some(5), 0));
    bank.slots[5].prefetched.store(true, Ordering::Release);
    assert_eq!((pin(), saves(), bank.slots[5].prefetched.load(Ordering::Relaxed)), (Some(5), 1, false));
    assert_eq!((pin(), saves()), (Some(5), 1), "once per staging");
    // A pin refills a missing key through the readers.
    let bank = bank_of(4, 4, [0, 2, 3]);
    resident(&bank, 0, (0, 0));
    resident(&bank, 1, (0, 2));
    let queue = Arc::new(FetchQueue::default());
    let _readers = Readers::new(Arc::clone(&queue), &[1]);
    let pin = |e| bank.pin_or_refill(&queue, (0, e), 0, Lane::Engine).expect("pinned");
    let pinned = [pin(0), pin(3), pin(2)];
    assert_eq!((pinned[0], pinned[2]), (0, 1));
    assert!(bank.is_resident((0, 3)));
    assert!(pinned.iter().all(|&s| bank.slots[s].pins.load(Ordering::Acquire) == 1), "each pinned once");
    // The third bank lacks the second key, and its empty layout fails the refill with `UnknownKey`, IO-free.
    let banks: [Arc<Bank>; 3] = std::array::from_fn(|_| bank_of(4, 8, []));
    let keys: [LayerExpertKey; 2] = [(0, 0), (0, 1)];
    for (b, bank) in banks.iter().enumerate() {
        let held = if b == 2 { &keys[..1] } else { &keys[..] };
        held.iter().enumerate().for_each(|(slot, &key)| resident(bank, slot, key));
    }
    let trio = [&banks[0], &banks[1], &banks[2]];
    let c0 = COUNTERS.now();
    let err = pin_all_blocking(trio, &FetchQueue::default(), &keys, 0, PinOrder::BankOuter, Lane::Engine)
        .expect_err("the third bank cannot resolve the second key");
    assert!(matches!(err, Error::UnknownKey(_)), "unexpected error: {err}");
    // 2 + 2 + 1 pins taken before the failure, all back.
    for (b, bank) in banks.iter().enumerate() {
        let pinned: Vec<usize> = (0..4).filter(|&s| bank.slots[s].pins.load(Ordering::Acquire) > 0).collect();
        assert!(pinned.is_empty(), "bank {b} slots {pinned:?} still pinned after a failed resolve");
    }
    assert!(pin_all(trio, &keys, 0).is_none(), "the third bank lacks the second key");
    let pinned = pin_all(trio, &keys[..1], 0).expect("every bank holds the first key");
    assert_eq!(COUNTERS.now().hits - c0.hits, 3, "a whole resolve counts each key once a bank, a failed one none");
    unpin_all(trio, &pinned);
}

pub(crate) type Device = Retained<ProtocolObject<dyn MTLDevice>>;

/// Held by a test that counts or takes memory: the counters are process-wide.
pub(crate) fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

pub(crate) fn device() -> Device {
    objc2_metal::MTLCreateSystemDefaultDevice().expect("a Metal device")
}

/// Named by the process and a count, not the clock: two tests starting in one microsecond would share a name.
pub(crate) fn temp_path(what: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("oscar-pool-{what}-{}-{n}.bin", std::process::id()))
}

/// Over a file of `content`, no segment admitted.
fn bank_over(
    content: &[u8], layout: HashMap<LayerExpertKey, ExpertExtent>,
    (segments, per_segment, slot_bytes): (usize, usize, usize), block_count: u16, num_experts: u16,
) -> (Device, Arc<Bank>) {
    let (device, path) = (device(), temp_path("bank"));
    std::fs::write(&path, content).expect("write tempfile");
    let file = Arc::new(TieredFile::open_many([&path]).expect("open tempfile"));
    let bank =
        Bank::new(&device, file, segments, per_segment, slot_bytes, layout, BankDtype::Q4K, block_count, num_experts);
    (device, Arc::new(bank.expect("a bank")))
}

/// One admitted segment of `slots` page-sized slots over a page of zeros: a layer of `experts`, `holds` its extents.
pub(crate) fn bank_of(slots: usize, experts: u16, holds: impl IntoIterator<Item = u16>) -> Arc<Bank> {
    let page = ExpertExtent { shard_id: 0, file_offset: 0, byte_count: 4096 };
    let layout = holds.into_iter().map(|e| ((0, e), page)).collect();
    let (device, bank) = bank_over(&[0; 4096], layout, (1, slots, 4096), 1, experts);
    bank.admit(&device, 0).expect("admit");
    assert_eq!(bank.capacity(), slots);
    bank
}

/// Expert `i` holds `bpe` bytes of `first + i`, keyed `(i / experts_per_layer, i % experts_per_layer)`.
pub(crate) fn bank_over_file(
    segments: usize, per_segment: usize, block_count: u16, experts_per_layer: u16, first: u8, bpe: usize,
) -> (Device, Arc<Bank>, Vec<LayerExpertKey>) {
    let (n, per) = (block_count as usize * experts_per_layer as usize, experts_per_layer as usize);
    let content: Vec<u8> = (0..n).flat_map(|i| std::iter::repeat_n(first + i as u8, bpe)).collect();
    let keys: Vec<LayerExpertKey> = (0..n).map(|i| ((i / per) as u16, (i % per) as u16)).collect();
    let layout = keys
        .iter()
        .enumerate()
        .map(|(i, &k)| (k, ExpertExtent { shard_id: 0, file_offset: (i * bpe) as u64, byte_count: bpe }))
        .collect();
    let (device, bank) = bank_over(&content, layout, (segments, per_segment, bpe), block_count, experts_per_layer);
    (device, bank, keys)
}

/// Whether `done` holds within two seconds, polled every 10 ms.
pub(crate) fn wait_until(done: impl Fn() -> bool) -> bool {
    let nap = || std::thread::sleep(std::time::Duration::from_millis(10));
    (0..200).any(|_| done() || (nap(), false).1)
}

/// Reserve, read and publish `key` on this thread; a resident key answers its slot.
pub(crate) fn fill(bank: &Bank, key: LayerExpertKey, tick: u32) -> usize {
    match bank.plan(&[key], tick).expect("a demand plan")[0] {
        Plan::Reserved { slot, extent } => {
            bank.read_into(0, slot, extent).expect("the read");
            bank.publish(slot, key, false).expect("the publish");
            slot
        }
        Plan::Resident(slot) => slot,
        Plan::Waiting => panic!("{key:?} is in flight"),
    }
}

pub(crate) fn borrowed(bank: &Bank) -> Vec<usize> {
    (0..).zip(bank.segments()).filter(|(_, s)| s.borrowed).map(|(i, _)| i).collect()
}
