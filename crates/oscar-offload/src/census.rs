//! Every buffer a forward asks for, on one clock of takes and frees, and the record replayed through a region's own
//! placement, so a width's peak is the run's.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;

use candle_metal_kernels::metal::Use;

use crate::machine::PAGE;
use crate::region::{Peak, Placement};

pub const UNRECORDED: u32 = u32::MAX;

/// The routed arm a recorded width ran: every layer sealed, or every layer a carrier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    Sealed,
    Carrier,
}

/// One buffer a forward asked for, as the pools saw it.
#[derive(Debug, Clone, Copy)]
pub struct Asked {
    /// Bytes: as asked, and padded to the whole pages cut.
    pub asked: usize,
    pub padded: usize,
    pub r#use: Use,
    pub queue: usize,
    /// The caller's label: who held the region at a peak, never part of the arithmetic.
    pub name: &'static str,
    /// In tokens: without it an entry could overlap another width's, a peak no forward ran.
    pub width: usize,
    /// One width's two arms are two forwards.
    pub arm: Arm,
    /// On the record's clock; `freed_at` is [`UNRECORDED`] while held.
    pub taken_at: u32,
    pub freed_at: u32,
}

/// Every take and free is one tick.
static CLOCK: AtomicU32 = AtomicU32::new(0);

static RECORD: Mutex<Vec<Asked>> = Mutex::new(Vec::new());

/// The width and arm the next entries are stamped with ([`note_width`]).
static WIDTH: AtomicUsize = AtomicUsize::new(0);
static CARRIER: AtomicBool = AtomicBool::new(false);

/// Candle's census window: while it runs no mark moves, as a census observes and never advances.
#[inline]
pub fn armed() -> bool {
    candle_metal_kernels::metal::pool::census()
}

pub fn note_width(width: usize, arm: Arm) {
    WIDTH.store(width, Ordering::Relaxed);
    CARRIER.store(arm == Arm::Carrier, Ordering::Relaxed);
}

#[inline]
pub fn arm_of() -> Arm {
    if CARRIER.load(Ordering::Relaxed) {
        Arm::Carrier
    } else {
        Arm::Sealed
    }
}

/// Its entry, which the holder hands back at the free, or [`UNRECORDED`] while disarmed.
#[inline]
pub fn note_take(asked: usize, padded: usize, r#use: Use, queue: usize, name: &'static str) -> u32 {
    if !armed() {
        return UNRECORDED;
    }
    let taken_at = CLOCK.fetch_add(1, Ordering::Relaxed);
    let mut record = RECORD.lock().unwrap_or_else(|e| e.into_inner());
    let at = record.len();
    record.push(Asked {
        asked,
        padded,
        r#use,
        queue,
        name,
        width: WIDTH.load(Ordering::Relaxed),
        arm: arm_of(),
        taken_at,
        freed_at: UNRECORDED,
    });
    u32::try_from(at).unwrap_or(UNRECORDED)
}

#[inline]
pub fn note_free(at: u32) {
    if at == UNRECORDED {
        return;
    }
    let freed_at = CLOCK.fetch_add(1, Ordering::Relaxed);
    let mut record = RECORD.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(entry) = record.get_mut(at as usize) {
        entry.freed_at = freed_at;
    }
}

/// The record, emptied; the clock runs on, so entries never collide across a reset.
pub fn take_record() -> Vec<Asked> {
    let mut record = std::mem::take(&mut *RECORD.lock().unwrap_or_else(|e| e.into_inner()));
    record.shrink_to_fit();
    record
}

/// The record under its lock, between two widths, still armed.
pub fn peek<T>(f: impl FnOnce(&[Asked]) -> T) -> T {
    f(&RECORD.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Each of `passes` replays `queue`'s takes and frees from `snapshot`, the placement at arming: the last pass's peak.
/// The queue passes nothing: with the GPU a forward behind, two generations are two passes.
pub fn region_peak(record: &[Asked], snapshot: &Placement, queue: usize, passes: &[(usize, Arm)]) -> Peak {
    let mut place = snapshot.unbounded();
    let mut peak = Peak::default();
    for &(width, arm) in passes {
        peak = Peak::default();
        peak.note(&place);
        let mut events: Vec<(u32, bool, usize)> = Vec::new();
        for (i, e) in record.iter().enumerate() {
            if e.queue != queue || e.width != width || e.arm != arm || e.r#use == Use::Fixed {
                continue;
            }
            events.push((e.taken_at, true, i));
            if e.freed_at != UNRECORDED {
                events.push((e.freed_at, false, i));
            }
        }
        events.sort_unstable();
        let mut placed: HashMap<usize, usize> = HashMap::new();
        for (_, taken, i) in events {
            let e = &record[i];
            if taken {
                let pages = e.padded.div_ceil(PAGE).max(1);
                let (first, _) = place
                    .cut(pages, e.asked, e.name, e.r#use == Use::Staged, 0)
                    .expect("an unbounded placement cuts anything");
                placed.insert(i, first);
                peak.note(&place);
            } else if let Some(first) = placed.remove(&i) {
                place.put_back(first, 1);
            }
        }
    }
    peak
}

/// A peak with its holders as the bench prints a measured region's, so the two read object by object.
pub fn peak_row(peak: &Peak) -> String {
    let held: Vec<String> =
        peak.by_holder.iter().map(|&(n, b)| format!("{n}:{:.1}M", b as f64 / (1u64 << 20) as f64)).collect();
    let kib = |pages: usize| pages * (PAGE >> 10);
    format!(
        "{}K ({}K from the bottom and {}K from the top){{{}}}",
        kib(peak.high),
        kib(peak.ends.0),
        kib(peak.ends.1),
        held.join(",")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{fresh, record};
    use candle_metal_kernels::metal::pool::set_census;

    /// `run`'s takes and frees, armed under the lock, as the record holds them.
    fn recorded(run: impl FnOnce()) -> Vec<Asked> {
        let _g = fresh();
        set_census(true);
        run();
        set_census(false);
        record()
    }

    /// The replay of `passes` over a 64-page region.
    fn peak(record: &[Asked], passes: &[(usize, Arm)]) -> Peak {
        region_peak(record, &Placement::bounded(64), 0, passes)
    }

    /// It can fail: drop the arm from the replay's filter and both arms read three pages; let a CPU-written cut take
    /// any run and the two passes read 3; let the decode's GPU-written object take fresh pages and the pair reads 11.
    #[test]
    fn a_widths_two_arms_are_two_forwards_a_pass_charges_a_staged_run_twice_and_a_later_pass_opens_at_the_top() {
        let r = recorded(|| {
            note_width(512, Arm::Sealed);
            let a = note_take(1000, 16384, Use::Scratch, 0, "test");
            note_width(512, Arm::Carrier);
            let b = note_take(2000, 32768, Use::Scratch, 0, "test");
            note_free(b);
            note_free(a);
        });
        assert_eq!((peak(&r, &[(512, Arm::Sealed)]).high, peak(&r, &[(512, Arm::Carrier)]).high), (1, 2));
        // Two passes charge a staged run twice and a GPU-written one once.
        let r = recorded(|| {
            note_width(512, Arm::Sealed);
            note_free(note_take(3 * PAGE, 3 * PAGE, Use::Scratch, 0, "gpu"));
            note_free(note_take(3 * PAGE, 3 * PAGE, Use::Staged, 0, "cpu"));
        });
        assert_eq!(peak(&r, &[(512, Arm::Sealed)]).high, 6, "the staged object takes fresh pages");
        let two = peak(&r, &[(512, Arm::Sealed), (512, Arm::Sealed)]);
        assert_eq!(two.high, 9, "the second pass's staged object goes above the first's");
        assert_eq!(two.holders, ["cpu"], "the holder at the moment the mark moved");
        // A later pass opens at the top the earlier one left.
        let r = recorded(|| {
            note_width(512, Arm::Sealed);
            note_free(note_take(8 * PAGE, 8 * PAGE, Use::Scratch, 0, "prefill"));
            note_free(note_take(2 * PAGE, 2 * PAGE, Use::Staged, 0, "cpu"));
            note_width(1, Arm::Sealed);
            note_free(note_take(PAGE, PAGE, Use::Scratch, 0, "decode"));
        });
        let passes = [&[(1, Arm::Sealed)][..], &[(512, Arm::Sealed)], &[(512, Arm::Sealed), (1, Arm::Sealed)]];
        // Decode alone; the prefill, eight pages up and two down; the decode after it takes the prefill's pages back.
        assert_eq!(passes.map(|p| peak(&r, p).high), [1, 10, 10]);
    }

    /// It can fail: name every entry's holder and `early` comes back.
    #[test]
    fn the_replay_places_what_the_run_placed() {
        let mut run = Placement::bounded(64);
        let r = recorded(|| {
            note_width(512, Arm::Sealed);
            let a = note_take(PAGE, PAGE, Use::Scratch, 0, "early");
            let (a_at, _) = run.cut(1, PAGE, "early", false, 0).unwrap();
            note_free(a);
            run.put_back(a_at, 1);
            let b = note_take(3 * PAGE, 3 * PAGE, Use::Scratch, 0, "late");
            let (b_at, _) = run.cut(3, 3 * PAGE, "late", false, 0).unwrap();
            let c = note_take(2 * PAGE, 2 * PAGE, Use::Scratch, 0, "other");
            let (c_at, _) = run.cut(2, 2 * PAGE, "other", false, 0).unwrap();
            note_free(b);
            run.put_back(b_at, 1);
            note_free(c);
            run.put_back(c_at, 1);
        });
        let p = peak(&r, &[(512, Arm::Sealed)]);
        assert_eq!(p.high, 5, "the run's high-water: three pages and two above them");
        assert_eq!(p.holders, ["late", "other"], "the holders at the peak; a caller gone before it is not in it");
        assert_eq!(peak(&r, &[(256, Arm::Sealed)]).high, 0, "a width never run demands nothing");
    }

    /// It can fail: drop either guard and a disarmed record takes, or the mark the census covers moves.
    #[test]
    fn nothing_is_recorded_until_armed_and_an_armed_census_moves_no_mark() {
        let _g = fresh();
        assert_eq!(note_take(1024, 16384, Use::Scratch, 0, "test"), UNRECORDED);
        assert!(record().is_empty());
        note_free(UNRECORDED);
        let marks = || (crate::machine::PLAN.phase_peaks(), crate::machine::refusals());
        let refuse = || crate::machine::refuse(crate::machine::Refused::Region, 1 << 30, 0, 0);
        let before = marks();
        set_census(true);
        assert_ne!(note_take(1024, 16384, Use::Scratch, 0, "test"), UNRECORDED);
        assert_eq!(record().len(), 1);
        assert!(crate::machine::PLAN.claim(1 << 40).is_none());
        crate::machine::PLAN.release(1 << 40);
        let _ = refuse();
        set_census(false);
        assert_eq!(marks(), before, "armed, a claim and a refusal move nothing");
        assert_eq!(note_take(1024, 16384, Use::Scratch, 0, "test"), UNRECORDED);
        assert_eq!(record().len(), 1, "a disarmed record takes nothing");
        let _ = refuse();
        assert_ne!(marks().1, before.1, "disarmed, the same refusal is counted");
    }
}
