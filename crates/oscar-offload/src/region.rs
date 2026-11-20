//! One command queue's working memory: one buffer object over one mapping, cut into objects at page offsets.
//! One object makes reuse safe: Metal's hazard tracking orders one resource, never separate objects over shared pages.

use crate::machine::{self, PAGE, PHASES};
use crate::segments::{padded_unit, Addressing, Lender, SegmentPool};
use crate::{Error, Result};
use candle_metal_kernels::metal::{Buffer, Fences, Held, Use};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLDevice};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Which pages of a region are whose, with no buffer under it: one arithmetic places the run and replays the record.
/// GPU-written objects cut from the bottom, CPU-written from the top, so host fills never split the middle.
#[derive(Clone, Debug, Default)]
pub struct Placement {
    pub(crate) size: usize,
    /// Pages each end has taken; neither cuts into the other's, so their sum is the region a state needs at any size.
    pub(crate) bottom: usize,
    pub(crate) top: usize,
    /// `(first, pages, stamp)`, sorted; the stamp a CPU write waits for the queue to pass, `None` once passed.
    free: Vec<(usize, usize, Option<u64>)>,
    /// Live runs by first page: pages, bytes asked, holder.
    pub(crate) live: BTreeMap<usize, (usize, usize, &'static str)>,
    pub(crate) in_hand: usize,
    pub(crate) asked: usize,
}

impl Placement {
    pub fn host_bytes(&self) -> usize {
        self.free.capacity() * std::mem::size_of::<(usize, usize, Option<u64>)>()
            + self.live.len()
                * (std::mem::size_of::<usize>()
                    + std::mem::size_of::<(usize, usize, &'static str)>()
                    + 3 * std::mem::size_of::<usize>())
    }

    pub(crate) fn bounded(size: usize) -> Self {
        let free = if size > 0 { vec![(0, size, None)] } else { Vec::new() };
        Self { size, free, ..Self::default() }
    }

    /// The replay's start: every run passed and the region grown in the middle, so a demand past it reads as pages.
    pub fn unbounded(&self) -> Self {
        let mut p = self.clone();
        // Passed first, so no free run straddles the ceiling.
        p.coalesce(u64::MAX);
        let far = usize::MAX / 4;
        let (by, ceiling) = (far - p.size, p.size - p.top);
        for run in &mut p.free {
            if run.0 >= ceiling {
                run.0 += by;
            }
        }
        p.live = p.live.iter().map(|(&f, &v)| (if f >= ceiling { f + by } else { f }, v)).collect();
        p.size = far;
        p.merge(ceiling, by, None);
        p.coalesce(u64::MAX);
        p
    }

    /// First fit from its own end, `cpu` from the top over passed runs only; the first page and the latest stamp taken.
    pub fn cut(
        &mut self, pages: usize, asked: usize, name: &'static str, cpu: bool, pass: u64,
    ) -> Option<(usize, Option<u64>)> {
        self.coalesce(pass);
        // Neither end cuts into what the other holds.
        let room = self.size - if cpu { self.bottom } else { self.top };
        if cpu {
            self.flip();
        }
        let taken = self.take(pages, room, cpu, pass);
        if cpu {
            self.flip();
        }
        let (at, stamp) = taken?;
        let first = if cpu { self.size - at - pages } else { at };
        self.live.insert(first, (pages, asked, name));
        self.in_hand += pages;
        self.asked += asked;
        if cpu {
            self.top = self.top.max(self.size - first);
        } else {
            self.bottom = self.bottom.max(first + pages);
        }
        Some((first, stamp))
    }

    /// The lowest contiguous free runs holding `pages` inside `room` (for `cpu`, passed ones only), taken out.
    fn take(&mut self, pages: usize, room: usize, cpu: bool, pass: u64) -> Option<(usize, Option<u64>)> {
        let usable = |stamp: Option<u64>| !cpu || stamp.is_none_or(|g| g <= pass);
        let mut i = 0;
        while i < self.free.len() {
            let first = self.free[i].0;
            if first + pages > room {
                // In address order: nothing above ends inside the room either.
                return None;
            }
            let (mut total, mut stamp, mut j) = (0, None, i);
            while j < self.free.len() && self.free[j].0 == first + total && usable(self.free[j].2) {
                total += self.free[j].1;
                stamp = stamp.max(self.free[j].2);
                if total >= pages {
                    break;
                }
                j += 1;
            }
            if total < pages {
                i += 1;
                continue;
            }
            let mut need = pages;
            while need > 0 {
                let (f, n, s) = self.free[i];
                if n <= need {
                    need -= n;
                    self.free.remove(i);
                } else {
                    self.free[i] = (f + need, n - need, s);
                    need = 0;
                }
            }
            return Some((first, stamp));
        }
        None
    }

    /// The free runs seen from the other end; its own inverse.
    fn flip(&mut self) {
        let size = self.size;
        for run in &mut self.free {
            run.0 = size - run.0 - run.1;
        }
        self.free.reverse();
    }

    /// The run at `first` comes back, stamped with the generation covering every kernel encoded against it.
    pub(crate) fn put_back(&mut self, first: usize, stamp: u64) {
        let Some((pages, asked, _)) = self.live.remove(&first) else {
            panic!("run put back at page {first} not live: region placement broken");
        };
        self.in_hand -= pages;
        self.asked -= asked;
        self.merge(first, pages, Some(stamp));
    }

    /// Free a run, merged only with a neighbour of its stamp (one command buffer's frees share one).
    fn merge(&mut self, mut first: usize, mut pages: usize, stamp: Option<u64>) {
        let mut at = self.free.partition_point(|&(f, _, _)| f < first);
        if at > 0 && self.free[at - 1].0 + self.free[at - 1].1 == first && self.free[at - 1].2 == stamp {
            let (f, n, _) = self.free.remove(at - 1);
            first = f;
            pages += n;
            at -= 1;
        }
        if at < self.free.len() && first + pages == self.free[at].0 && self.free[at].2 == stamp {
            let (_, n, _) = self.free.remove(at);
            pages += n;
        }
        self.free.insert(at, (first, pages, stamp));
    }

    /// A passed run is fresh for good, as the queue never goes back: its stamp comes off and fresh neighbours merge.
    fn coalesce(&mut self, pass: u64) {
        for run in &mut self.free {
            if run.2.is_some_and(|g| g <= pass) {
                run.2 = None;
            }
        }
        let mut i = 0;
        while i + 1 < self.free.len() {
            let (f, n, s) = self.free[i];
            let (f2, n2, s2) = self.free[i + 1];
            if f + n == f2 && s.is_none() && s2.is_none() {
                self.free[i] = (f, n + n2, None);
                self.free.remove(i + 1);
            } else {
                i += 1;
            }
        }
        // An end retreats to a free run at its edge: its extent is a level, not a high-water.
        if let Some(&(f, n, _)) =
            self.free.iter().find(|&&(f, n, s)| s.is_none() && f <= self.bottom && f + n >= self.size - self.top)
        {
            self.bottom = f;
            self.top = self.size - f - n;
        }
    }

    /// The region this state needs, in pages: at this size its runs land where they are, so the replay's is the run's.
    pub fn span(&self) -> usize {
        self.bottom + self.top
    }

    /// Freed pages the queue has not passed: a GPU-written cut's, not yet a CPU-written one's.
    pub(crate) fn fenced_pages(&self, pass: u64) -> usize {
        self.free.iter().filter(|&&(_, _, s)| s.is_some_and(|g| g > pass)).map(|&(_, n, _)| n).sum()
    }

    /// Widest first, `true` where the queue has not passed them: a refusal is about what the next cut can use.
    pub(crate) fn free_runs(&self, pass: u64) -> Vec<(usize, bool)> {
        let mut runs: Vec<(usize, bool)> =
            self.free.iter().map(|&(_, n, s)| (n, s.is_some_and(|g| g > pass))).collect();
        runs.sort_unstable_by_key(|&(n, _)| std::cmp::Reverse(n));
        runs
    }

    pub fn is_idle(&self, pass: u64) -> bool {
        self.live.is_empty() && self.free.iter().all(|&(_, _, s)| s.is_none_or(|g| g <= pass))
    }

    /// Only while idle, as a live run names pages from its end and a fenced one may still be read.
    pub(crate) fn rebound(&mut self, size: usize, pass: u64) -> bool {
        if !self.is_idle(pass) {
            return false;
        }
        *self = Self::bounded(size);
        true
    }

    pub(crate) fn holders(&self) -> Vec<&'static str> {
        self.live.values().map(|&(_, _, name)| name).collect()
    }

    /// The live runs' bytes by holder, largest first.
    pub fn by_holder(&self) -> Vec<(&'static str, usize)> {
        let mut rows: Vec<(&'static str, usize)> = Vec::new();
        for &(pages, _, name) in self.live.values() {
            match rows.iter_mut().find(|r| r.0 == name) {
                Some(row) => row.1 += pages * PAGE,
                None => rows.push((name, pages * PAGE)),
            }
        }
        rows.sort_by_key(|r| std::cmp::Reverse(r.1));
        rows
    }
}

/// Holders, ends and holes (free runs widest first, `f` where fenced): a claim is given a run, never the total.
pub(crate) fn layout_of(place: &Placement, queue: usize, pass: u64) -> String {
    let mib = |pages: usize| pages as f64 / 64.0;
    let held: Vec<String> =
        place.by_holder().iter().take(8).map(|&(n, b)| format!("{n}:{:.1}M", b as f64 / (1u64 << 20) as f64)).collect();
    let runs = place.free_runs(pass);
    let widest: Vec<String> =
        runs.iter().take(8).map(|&(n, f)| format!("{:.1}M{}", mib(n), if f { "f" } else { "" })).collect();
    let spans: Vec<String> = place
        .live
        .iter()
        .take(10)
        .map(|(&at, &(n, _, name))| format!("{name}@{:.1}M:{:.1}M", mib(at), mib(n)))
        .collect();
    format!(
        "{queue}: holders {} ({:.1}M), bottom {:.1}M, top {:.1}M of {:.1}M {{{}}} at {:?}; free \
         runs {}, widest {}",
        place.live.len(),
        mib(place.in_hand),
        mib(place.bottom),
        mib(place.top),
        mib(place.size),
        held.join(","),
        spans.join(","),
        runs.len(),
        widest.join(","),
    )
}

/// A high-water and the holders when it was set, per phase (the region) or pass (the replay): what a plan charges.
#[derive(Clone, Debug, Default)]
pub struct Peak {
    /// In pages.
    pub high: usize,
    pub holders: Vec<&'static str>,
    pub by_holder: Vec<(&'static str, usize)>,
    /// `(bottom, top)` at the mark, summing to `high`: which end grew is where a region a page short is short.
    pub ends: (usize, usize),
}

impl Peak {
    /// A cut, or a phase or pass beginning: mark where the span passes the high.
    pub fn note(&mut self, place: &Placement) {
        let span = place.span();
        if span > self.high {
            self.high = span;
            self.holders = place.holders();
            self.by_holder = place.by_holder();
            self.ends = (place.bottom, place.top);
        }
    }
}

/// High-waters per phase, a phase opening at what is present; none moves while the census is armed.
#[derive(Clone, Debug)]
pub struct Marks {
    pub peak: [Peak; PHASES],
    /// Bytes in hand at most, and asked at most.
    pub out: [usize; PHASES],
    pub asked: [usize; PHASES],
    last_phase: usize,
}

impl Default for Marks {
    fn default() -> Self {
        Self {
            peak: std::array::from_fn(|_| Peak::default()),
            out: [0; PHASES],
            asked: [0; PHASES],
            last_phase: usize::MAX,
        }
    }
}

impl Marks {
    fn before_cut(&mut self, phase: usize, place: &Placement) {
        if self.last_phase != phase {
            self.last_phase = phase;
            self.peak[phase].note(place);
        }
    }

    fn after_cut(&mut self, phase: usize, place: &Placement) {
        self.peak[phase].note(place);
        self.out[phase] = self.out[phase].max(place.in_hand * PAGE);
        self.asked[phase] = self.asked[phase].max(place.asked);
    }
}

/// Where the region stands, for the memory ledger.
#[derive(Clone, Copy, Debug, Default)]
pub struct Levels {
    /// Bytes mapped.
    pub size: usize,
    /// Of them, above the bound: lent.
    pub lent: usize,
    /// Bytes in hand, and asked for.
    pub live: usize,
    pub asked: usize,
    /// Freed bytes the queue has not passed.
    pub fenced: usize,
}

struct Inner {
    /// `(object, pages)` over the mapping's first pages; the objects come before `segments`, so are dropped first.
    first_object: Option<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)>,
    /// The object over the whole mapping, which a borrowing pool declares; `None` while nothing is mapped.
    buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// One segment for the region's life, so one residency set for its queues: a queue takes at most 32.
    segments: Arc<SegmentPool>,
    /// Pages mapped, 0 while none are.
    mapped: usize,
    place: Placement,
    marks: Marks,
    /// Locked: the census's mapping is not, until [`Region::retake`].
    wired: bool,
}

pub struct Region {
    queue: usize,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    fences: Fences,
    inner: Mutex<Inner>,
}

// SAFETY: the device and buffer are thread-safe Metal objects; the rest is locked.
unsafe impl Send for Region {}
unsafe impl Sync for Region {}

impl Region {
    /// Wired against the plan's ceiling and the machine's floor, or for the census only mapped.
    pub fn take(
        device: &ProtocolObject<dyn MTLDevice>, queue: usize, bytes: usize, fences: Fences, wired: bool,
    ) -> Result<Arc<Self>> {
        // Its one segment holds up to one buffer object's pages, whatever a re-take asks.
        let most = padded_unit(largest_bytes(device)) / PAGE;
        let segments = SegmentPool::new(device, PAGE, 1, most, Addressing::Whole)?;
        let (place, marks) = (Placement::default(), Marks::default());
        let inner = Inner { first_object: None, buffer: None, segments, mapped: 0, place, marks, wired };
        let region = Self { queue, device: objc2::Message::retain(device), fences, inner: Mutex::new(inner) };
        region.map(&mut region.lock(), bytes, wired)?;
        Ok(Arc::new(region))
    }

    /// `bytes` in whole pages over the region's segment, the placement bounded at them.
    fn map(&self, inner: &mut Inner, bytes: usize, wired: bool) -> Result<()> {
        // Past the device's limit the wrap answers nil only after the mapping and the lock.
        if bytes > largest_bytes(&self.device) {
            return Err(Error::Layout(format!(
                "a working region of {bytes} bytes is past the {} one Metal buffer object may hold",
                largest_bytes(&self.device)
            )));
        }
        let pages = padded_unit(bytes.max(1)) / PAGE;
        inner.segments.take(&self.device, 0, pages, wired)?;
        inner.buffer = Some(inner.segments.buffer(0).expect("the region's segment was just taken"));
        (inner.mapped, inner.place, inner.wired) = (pages, Placement::bounded(pages), wired);
        Ok(())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().expect("the region is poisoned")
    }

    pub fn queue(&self) -> usize {
        self.queue
    }

    /// Bytes mapped.
    pub fn size(&self) -> usize {
        self.lock().mapped * PAGE
    }

    /// What the region cuts from, in bytes; the mapping above it is lent.
    pub fn bound(&self) -> usize {
        self.lock().place.size * PAGE
    }

    pub fn lender(&self) -> Lender {
        let inner = self.lock();
        let base = std::ptr::NonNull::new(inner.segments.ptr(0).cast()).expect("the region's segment is present");
        let buffer = inner.buffer.clone().expect("a region holds its object outside a re-take");
        Lender { buffer, base, bytes: inner.mapped * PAGE, queue: self.queue }
    }

    /// The first `bytes` as their own object for decode: binding one of 4 GB or more costs every next encoder.
    /// The device counts it beside the whole mapping, so the plan books it on the GPU driver's ceiling.
    pub fn wrap_first(&self, bytes: usize) -> Result<()> {
        let pages = padded_unit(bytes.max(1)) / PAGE;
        let mut inner = self.lock();
        if pages >= inner.mapped || inner.first_object.as_ref().is_some_and(|&(_, n)| n == pages) {
            return Ok(());
        }
        let at = std::ptr::NonNull::new(inner.segments.ptr(0).cast()).expect("the region's segment is present");
        // SAFETY: the mapping is page-aligned, longer than `pages`, and outlives the object, which drops first.
        let object = unsafe {
            self.device.newBufferWithBytesNoCopy_length_options_deallocator(
                at,
                pages * PAGE,
                objc2_metal::MTLResourceOptions::StorageModeShared,
                None,
            )
        };
        inner.first_object = Some((
            object.ok_or_else(|| {
                Error::Metal("newBufferWithBytesNoCopy over the region's first pages returned nil".into())
            })?,
            pages,
        ));
        Ok(())
    }

    /// Above the bound is lent; only while idle, the caller leaving no adopted segment under the new bound.
    pub fn move_bound(&self, bytes: usize) -> Result<()> {
        let pages = padded_unit(bytes.max(1)) / PAGE;
        let mut inner = self.lock();
        let mapped = inner.mapped;
        if pages > mapped {
            return Err(Error::Layout(format!(
                "queue {}'s working region maps {mapped} pages and cannot be bound at {pages}",
                self.queue
            )));
        }
        let pass = self.fences.pass();
        if !inner.place.rebound(pages, pass) {
            return Err(Error::Layout(format!(
                "queue {}'s working region is not idle and keeps its bound -{}",
                self.queue,
                layout_of(&inner.place, self.queue, pass)
            )));
        }
        Ok(())
    }

    /// `staged` where the CPU writes it; the run's first page, or the refusal.
    fn cut(&self, bytes: usize, staged: bool, name: &'static str) -> Result<usize> {
        let pages = padded_unit(bytes.max(1)) / PAGE;
        let armed = crate::census::armed();
        let phase = machine::phase();
        let mut inner = self.lock();
        if !armed {
            let Inner { place, marks, .. } = &mut *inner;
            marks.before_cut(phase, place);
        }

        // The queue's progress, read once for every stamp the cut compares.
        let pass = self.fences.pass();
        let Some((first, stamp)) = inner.place.cut(pages, bytes, name, staged, pass) else {
            return Err(self.refuse(&inner, bytes, name, pass));
        };
        // The reuse count: one completed-generation read.
        if !staged {
            REUSE.fetch_add(1, Ordering::Relaxed);
            if stamp.is_some_and(|g| g > self.fences.completed()) {
                REUSE_PAST_FENCE.fetch_add(1, Ordering::Relaxed);
            }
        }
        if !armed {
            let Inner { place, marks, .. } = &mut *inner;
            marks.after_cut(phase, place);
        }
        Ok(first)
    }

    /// The region's buffer at the run's offset, given back at the last drop: an object's identity is buffer and run.
    pub fn object(self: &Arc<Self>, bytes: usize, r#use: Use, name: &'static str) -> Result<Buffer> {
        let staged = r#use == Use::Staged;
        let first = self.cut(bytes, staged, name)?;
        let len = padded_unit(bytes.max(1));
        let census = crate::census::note_take(bytes, len, r#use, self.queue, name);
        let buffer = {
            let inner = self.lock();
            match &inner.first_object {
                Some((first, pages)) if inner.place.size <= *pages => first.clone(),
                _ => inner.buffer.clone().expect("a region holds its object outside a re-take"),
            }
        };
        let hold = Arc::new(Hold { region: Arc::clone(self), first, census });
        Ok(Buffer::pooled(buffer, first * PAGE, len, hold))
    }

    /// Counted and said with the arithmetic and the layout: an object no free run fits is the plan's defect.
    fn refuse(&self, inner: &Inner, bytes: usize, name: &'static str, pass: u64) -> Error {
        let p = &inner.place;
        let fenced = p.fenced_pages(pass);
        let held = ((p.in_hand + fenced) * PAGE) as u64;
        let refused = machine::refuse(machine::Refused::Region, bytes as u64, held, (p.size * PAGE) as u64).into();
        // A census's refusal ends its record at this width; the layout is a live refusal's evidence.
        if crate::census::armed() {
            return refused;
        }
        let mib = |pages: usize| pages as f64 / 64.0;
        tracing::info!(target: "oscar::region",
            "refused - `{name}` asked queue {}; live {:.1}, fenced {:.1}, ends {:.1}/{:.1} MB, \
             widest free run {:.1} MB",
            self.queue,
            mib(p.in_hand),
            mib(fenced),
            mib(p.bottom),
            mib(p.top),
            mib(p.free_runs(pass).first().map_or(0, |&(n, _)| n)),
        );
        // A refusal for want of a run is about the layout: who holds the region and where the holes are.
        tracing::info!(target: "oscar::region", "layout -{}", layout_of(p, self.queue, pass));
        refused
    }

    fn give_back(&self, first: usize) {
        let stamp = self.fences.fence();
        self.lock().place.put_back(first, stamp);
    }

    /// The plan's draw: the census's mapping re-taken locked, a locked one only at a new size.
    /// Refused while an object or a loan names the mapping; one the bounds refuse leaves no mapping and cuts nothing.
    pub(crate) fn retake(&self, bytes: usize) -> Result<()> {
        let want = padded_unit(bytes.max(1)) / PAGE;
        let mut inner = self.lock();
        let (mapped, held) = (inner.mapped, inner.mapped > 0);
        if held && inner.wired && want == mapped {
            return Ok(());
        }
        if held && (inner.place.size != mapped || !inner.place.is_idle(self.fences.pass())) {
            let gib = |pages: usize| pages as f64 * PAGE as f64 / crate::machine::GB;
            return Err(Error::Layout(format!(
                "the region holds {} object(s) and lends {:.2} GB, so it keeps its {:.2} GB mapping where the plan draws {:.2} GB; layout {}",
                inner.place.live.len(),
                gib(mapped - inner.place.size),
                gib(mapped),
                gib(want),
                layout_of(&inner.place, self.queue, self.fences.pass()),
            )));
        }
        // The old mapping goes first, so the ceiling sees one region; the segment keeps the queues' residency set.
        // No object may outlive its pages: the GPU driver keeps an unmapped range for the object's life.
        (inner.first_object, inner.buffer, inner.mapped, inner.place) = (None, None, 0, Placement::bounded(0));
        if held {
            inner.segments.release(0)?;
        }
        self.map(&mut inner, bytes, true)
    }

    #[cfg(test)]
    pub(crate) fn is_locked(&self) -> bool {
        self.lock().wired
    }

    pub fn attach(&self, queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>) {
        self.lock().segments.attach(queue);
    }

    /// Idle: the one state in which the bound may move.
    pub fn is_idle(&self) -> bool {
        self.lock().place.is_idle(self.fences.pass())
    }

    /// The placement, the census's starting state.
    pub fn snapshot(&self) -> Placement {
        self.lock().place.clone()
    }

    pub fn marks(&self) -> Marks {
        self.lock().marks.clone()
    }

    pub fn levels(&self) -> Levels {
        let inner = self.lock();
        let p = &inner.place;
        let mapped = inner.mapped;
        Levels {
            size: mapped * PAGE,
            lent: (mapped - p.size) * PAGE,
            live: p.in_hand * PAGE,
            asked: p.asked,
            fenced: p.fenced_pages(self.fences.pass()) * PAGE,
        }
    }

    /// What holds the region and where, for the memory trace.
    pub fn describe(&self) -> String {
        let inner = self.lock();
        layout_of(&inner.place, self.queue, self.fences.pass())
    }
}

/// A region is one buffer object, so the device's buffer limit bounds it.
pub fn largest_bytes(device: &ProtocolObject<dyn MTLDevice>) -> usize {
    device.maxBufferLength()
}

/// A region object's run, given back at the last handle's drop.
#[derive(Debug)]
struct Hold {
    region: Arc<Region>,
    first: usize,
    /// Its census entry (or [`crate::census::UNRECORDED`]), which its give-back names.
    census: u32,
}

impl std::fmt::Debug for Region {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "region of queue {}", self.queue)
    }
}

impl Held for Hold {}

impl Drop for Hold {
    fn drop(&mut self) {
        crate::census::note_free(self.census);
        self.region.give_back(self.first);
    }
}

/// GPU-written cuts, and those of a run the GPU had not completed: the reuse hazard tracking must order.
static REUSE: AtomicUsize = AtomicUsize::new(0);
static REUSE_PAST_FENCE: AtomicUsize = AtomicUsize::new(0);

pub fn reuse() -> (usize, usize) {
    (REUSE.load(Ordering::Relaxed), REUSE_PAST_FENCE.load(Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cut's first page, `None` refused.
    fn at(cut: Option<(usize, Option<u64>)>) -> Option<usize> {
        cut.map(|c| c.0)
    }

    fn free_pages(p: &Placement) -> usize {
        p.free.iter().map(|r| r.1).sum()
    }

    #[test]
    fn a_placement_cuts_lowest_first_merges_at_the_fence_and_continues_unbounded_where_the_region_ends() {
        let mut p = Placement::bounded(16);
        let firsts = [2, 3, 4].map(|pages| at(p.cut(pages, 1, "a", false, 0)));
        assert_eq!((firsts, p.in_hand, p.span()), ([Some(0), Some(2), Some(5)], 9, 9));
        p.put_back(2, 1);
        assert_eq!(at(p.cut(5, 1, "d", false, 0)), Some(9), "a run wider than the hole goes above");
        assert_eq!(at(p.cut(3, 1, "e", false, 0)), Some(2), "one that fits takes the hole first");
        assert_eq!(p.span(), 14);
        p.put_back(0, 2);
        p.put_back(2, 2);
        assert_eq!(p.free, vec![(0, 5, Some(2)), (14, 2, None)], "one stamp, one run");
        p.put_back(5, 4);
        p.put_back(9, 5);
        let apart = vec![(0, 5, Some(2)), (5, 4, Some(4)), (9, 5, Some(5)), (14, 2, None)];
        assert_eq!(p.free, apart, "different stamps stay apart");
        assert_eq!(p.cut(15, 1, "f", false, 2), Some((0, Some(5))), "a GPU-written cut spans them all");
        assert_eq!(p.free, vec![(15, 1, None)]);
        p.put_back(0, 5);
        p.cut(1, 1, "g", false, 5);
        assert_eq!(p.free, vec![(1, 15, None)], "passed, the stamped run is fresh again and one with the tail");
        p.put_back(0, 5);
        assert_eq!((p.is_idle(5), p.is_idle(4)), (true, false));
        assert_eq!(p.cut(17, 1, "h", false, 5), None, "wider than what is free is refused");
        // The unbounded form continues where the region ends.
        let mut p = Placement::bounded(4);
        p.cut(3, 1, "a", false, 0);
        let mut u = p.unbounded();
        assert_eq!(at(u.cut(2, 1, "b", false, 0)), Some(3), "the run at the region's end continues past it");
        assert_eq!(u.span(), 5);
        let mut full = Placement::bounded(2);
        full.cut(2, 1, "a", false, 0);
        assert_eq!(at(full.unbounded().cut(1, 1, "b", false, 0)), Some(2), "a full region continues at its end");
        // A run freed before the replay is passed: a CPU-written cut takes the top end back.
        let mut freed = Placement::bounded(2);
        let (host, _) = freed.cut(1, 1, "host", true, 0).unwrap();
        freed.put_back(host, 3);
        let mut u = freed.unbounded();
        assert_eq!(at(u.cut(1, 1, "host", true, 0)), Some(u.size - 1), "the ceiling, not the page under it");
        assert_eq!(u.span(), 1);
    }

    /// It can fail: host fills cut from the bottom split the middle, refuse the claim with 47 pages free, and answer
    /// different depths at the two sizes.
    #[test]
    fn a_cpu_written_cut_takes_the_top_end_only_once_passed_and_leaves_the_region_in_one_run() {
        let mut p = Placement::bounded(12);
        assert_eq!(at(p.cut(4, 1, "gpu", false, 0)), Some(0), "what a kernel writes takes the bottom");
        assert_eq!(at(p.cut(2, 1, "host", true, 0)), Some(10), "what the host fills takes the top");
        assert_eq!((p.span(), p.bottom, p.top), (6, 4, 2), "the two ends; what is free between them is neither's");
        p.put_back(10, 7);
        assert_eq!(at(p.cut(2, 1, "host", true, 5)), Some(8), "the run given back is not passed: the next goes deeper");
        assert_eq!(p.fenced_pages(5), 2);
        p.put_back(0, 9);
        assert_eq!(p.cut(4, 1, "kernel", false, 5), Some((0, Some(9))), "a GPU-written cut takes what came back");
        p.put_back(8, 11);
        assert_eq!(at(p.cut(2, 1, "host", true, 11)), Some(10), "passed, the ceiling is the top end's again");
        assert_eq!((p.fenced_pages(11), p.bottom, p.top), (0, 4, 2), "and the end retreats to what it holds");
        // A host fill leaves the region in one run.
        let mut p = Placement::bounded(64);
        p.cut(16, 1, "kept", false, 0);
        let (gone, _) = p.cut(8, 1, "gone", false, 0).unwrap();
        p.put_back(gone, 1);
        assert_eq!((at(p.cut(1, 1, "host", true, 0)), free_pages(&p)), (Some(63), 47), "under the ceiling");
        assert_eq!(at(p.cut(40, 1, "wide", false, 0)), Some(16), "the claim spans the fenced hole and the free middle");
        // What a placement needs does not depend on the region it is in.
        let sequence = |size: usize| {
            let mut p = Placement::bounded(size);
            let mut from_its_end = Vec::new();
            for (pages, cpu) in [(4, false), (1, true), (8, false), (2, true), (16, false)] {
                let (first, _) = p.cut(pages, 1, "x", cpu, 0).expect("the region holds it");
                from_its_end.push(if cpu { size - first } else { first });
            }
            (from_its_end, p.span())
        };
        assert_eq!(sequence(64), sequence(4096));
        assert_eq!(sequence(64).1, 31, "28 pages from the bottom and 3 from the top");
    }

    /// It can fail: rebound without the idle test and a GPU-written cut lands on a live run.
    #[test]
    fn the_bound_moves_only_while_the_region_is_idle_and_leaves_one_fresh_run() {
        let mut p = Placement::bounded(64);
        let (kernel, _) = p.cut(4, 1, "kernel", false, 0).unwrap();
        let (host, _) = p.cut(2, 1, "host", true, 0).unwrap();
        assert!(!p.rebound(16, 0), "a live run holds the bound");
        p.put_back(kernel, 3);
        p.put_back(host, 3);
        assert!(!p.rebound(16, 2), "and so does a run the queue has not passed");
        assert_eq!((p.size, free_pages(&p)), (64, 64), "a refused move moves nothing");
        assert!(p.rebound(16, 3));
        assert_eq!((p.size, p.free.clone(), (p.bottom, p.top), p.in_hand), (16, vec![(0, 16, None)], (0, 0), 0));
        assert_eq!(at(p.cut(2, 1, "host", true, 3)), Some(14), "a host fill lands under the new bound");
        assert_eq!(at(p.cut(15, 1, "wide", false, 3)), None, "what is above the bound is not the region's to cut");
        p.put_back(14, 4);
        assert!(p.rebound(64, 4), "back again, once idle");
        assert_eq!(at(p.cut(60, 1, "wide", false, 4)), Some(0));
    }
}
