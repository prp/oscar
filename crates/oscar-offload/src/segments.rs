//! Segments of units over a fixed index space; a taken segment is a no-copy buffer over a mapping of its own, locked
//! but for the census's, or a lender's range.

use crate::fixed::Fixed;
use crate::{Error, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLDevice, MTLResidencySet, MTLResidencySetDescriptor,
    MTLResourceOptions,
};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// Every mapping and unit is whole pages, so the pools' users name the page here.
pub use crate::machine::PAGE;

fn map_anon(i: usize, len: usize) -> Result<*mut std::ffi::c_void> {
    // SAFETY: a fresh anonymous mapping; nothing else names it.
    let raw = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_ANON | libc::MAP_PRIVATE,
            -1,
            0,
        )
    };
    if raw == libc::MAP_FAILED {
        let why = std::io::Error::last_os_error();
        return Err(Error::Metal(format!("mmap of {len} bytes for segment {i} failed: {why}")));
    }
    Ok(raw)
}

pub fn padded_unit(bytes: usize) -> usize {
    bytes.next_multiple_of(PAGE)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Addressing {
    /// Kernels reach each unit (an expert slice, a KV block) through the table, written at every take.
    Table,
    /// A segment handed out whole as a buffer: at page-granular units a table would be large and never read.
    Whole,
}

struct Segment {
    ptr: NonNull<std::ffi::c_void>,
    units: usize,
    /// Locked and charged; a pageable or borrowed segment is neither.
    wired: bool,
    /// `None` once the emergency let it go: nothing may reference it.
    buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// A range of a lender's buffer object, whose lock, residency and charge are the lender's.
    borrowed: bool,
}

/// Per group of pools moving together: committing a set that running command buffers use is not safe.
pub(crate) struct Residency {
    inner: Mutex<ResidencyInner>,
}

struct ResidencyInner {
    set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    queues: Vec<Retained<ProtocolObject<dyn MTLCommandQueue>>>,
}

// SAFETY: the set is mutated only under its own lock or Metal's.
unsafe impl Send for Residency {}
unsafe impl Sync for Residency {}

/// Every group, weakly, for [`declare_pools`].
static GROUPS: Mutex<Vec<std::sync::Weak<Residency>>> = Mutex::new(Vec::new());

/// Every pool, weakly, for [`release_all`].
static REGISTRY: Mutex<Vec<std::sync::Weak<SegmentPool>>> = Mutex::new(Vec::new());

/// The kernel's counts around the give-up, not `munlock`'s bytes: it succeeds on unlocked pages too.
#[derive(Clone, Copy, Debug, Default)]
pub struct GivenUp {
    /// The kernel's per-task lock count, before and after.
    pub locked: (u64, u64),
    /// The machine's wired total before and after the drain: the only reading of the GPU driver's wire.
    pub wired: (u64, u64),
    /// Pools whose lock was held, left as they were.
    pub missed: usize,
}

impl std::fmt::Display for GivenUp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gib = |b: u64| b as f64 / crate::machine::GB;
        write!(
            f,
            "oscar's locked memory {:.1} -> {:.1} GB, the machine's wired memory {:.1} -> {:.1} GB",
            gib(self.locked.0),
            gib(self.locked.1),
            gib(self.wired.0),
            gib(self.wired.1),
        )?;
        if self.missed > 0 {
            write!(f, ", {} pool(s) out of reach", self.missed)?;
        }
        Ok(())
    }
}

/// Two seconds of touches, twice the GPU driver's longest grace on a retired buffer (a touch does not restart it).
const TOUCH_EVERY: std::time::Duration = std::time::Duration::from_millis(250);
const TOUCHES: u32 = 8;

static GIVING_UP: AtomicBool = AtomicBool::new(false);

/// Whether [`release_all`] has begun, so a pool move would move what is gone; only the exit follows.
pub fn giving_up() -> bool {
    GIVING_UP.load(Ordering::Acquire)
}

/// The emergency's first act, so the process dies with nothing wired: every segment and the GPU driver's wire on it.
/// Every lock a `try_lock` and the drain's wait bounded, so the backstop cannot block.
pub fn release_all() -> GivenUp {
    GIVING_UP.store(true, Ordering::Release);
    // One give-up a process: a second caller waits for the first and reads its result rather than drain past it.
    static GIVEN_UP: Mutex<Option<GivenUp>> = Mutex::new(None);
    let mut once = GIVEN_UP.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(given_up) = *once {
        return given_up;
    }
    let wired = || crate::machine::vm_pages().map_or(0, |vm| vm.wired);
    let (locked_before, wired_before) = (crate::machine::own_rusage().0, wired());
    let missed = retire_and_unlock();
    let locked_after = crate::machine::own_rusage().0;
    let (done, drained) = std::sync::mpsc::channel();
    let spawned = std::thread::Builder::new().name("oscar-drain".into()).spawn(move || {
        drain();
        let _ = done.send(());
    });
    if spawned.is_ok() {
        let _ = drained.recv_timeout(2 * (TOUCHES + 1) * TOUCH_EVERY);
    }
    let given_up = GivenUp { locked: (locked_before, locked_after), wired: (wired_before, wired()), missed };
    // One that missed a pool may be tried again.
    if missed == 0 {
        *once = Some(given_up);
    }
    given_up
}

/// Every wired segment out of its set and unlocked, a lender's object out too, the mappings left; the pools missed.
fn retire_and_unlock() -> usize {
    let Ok(pools) = REGISTRY.try_lock() else {
        return 1;
    };
    let mut missed = 0usize;
    for pool in pools.iter().filter_map(std::sync::Weak::upgrade) {
        let Ok(mut live) = pool.live.try_lock() else {
            missed += 1;
            continue;
        };
        let own = live.segments.iter().flatten().filter(|s| s.wired).filter_map(|s| s.buffer.as_deref());
        if !pool.set.retire(own.chain(live.lender.as_ref().map(|l| &*l.buffer)), false) {
            missed += 1;
            continue;
        }
        for seg in live.segments.iter_mut().flatten().filter(|s| s.wired) {
            drop(seg.buffer.take());
            // SAFETY: the mapping is present and this long, under the pool's lock.
            unsafe { libc::munlock(seg.ptr.as_ptr(), seg.units * pool.unit_bytes) };
        }
    }
    missed
}

/// The GPU driver drops its wire on a retired buffer only past a grace, at the next admit and retire of a locked page.
fn drain() {
    let Some(device) = objc2_metal::MTLCreateSystemDefaultDevice() else {
        return;
    };
    for _ in 0..TOUCHES {
        std::thread::sleep(TOUCH_EVERY);
        touch(&device);
    }
    std::thread::sleep(TOUCH_EVERY);
}

/// One locked page admitted and retired: what moves the GPU driver past its grace on the buffers it was told to drop.
fn touch(device: &ProtocolObject<dyn MTLDevice>) {
    // SAFETY: one anonymous page, this block's alone, unmapped after its buffer.
    unsafe {
        let page = libc::mmap(
            std::ptr::null_mut(),
            PAGE,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_ANON | libc::MAP_PRIVATE,
            -1,
            0,
        );
        let Some(at) = NonNull::new(page).filter(|_| page != libc::MAP_FAILED) else {
            return;
        };
        if libc::mlock(page, PAGE) == 0 {
            let buffer = device.newBufferWithBytesNoCopy_length_options_deallocator(
                at,
                PAGE,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            let set = device.newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new()).ok();
            if let (Some(buffer), Some(set)) = (&buffer, &set) {
                set.addAllocation(ProtocolObject::from_ref(&**buffer));
                set.commit();
                set.requestResidency();
                set.removeAllocation(ProtocolObject::from_ref(&**buffer));
                set.commit();
            }
            drop(buffer);
            libc::munlock(page, PAGE);
        }
        libc::munmap(page, PAGE);
    }
}

/// A set on a queue only requests residency, perhaps too late; declared on a command buffer it holds for its execution.
pub(crate) fn declare_pools(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>, queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>,
) {
    let mut groups = GROUPS.lock().expect("the residency groups are poisoned");
    groups.retain(|g| g.strong_count() > 0);
    for group in groups.iter().filter_map(std::sync::Weak::upgrade) {
        let inner = group.lock();
        if inner.queues.iter().any(|q| Retained::as_ptr(q) == Retained::as_ptr(queue)) {
            command_buffer.useResidencySet(&inner.set);
        }
    }
}

impl Residency {
    fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Arc<Self>> {
        let desc = MTLResidencySetDescriptor::new();
        let set = device
            .newResidencySetWithDescriptor_error(&desc)
            .map_err(|e| Error::Metal(format!("newResidencySet failed: {e}")))?;
        let group = Arc::new(Self { inner: Mutex::new(ResidencyInner { set, queues: Vec::new() }) });
        GROUPS.lock().expect("the residency groups are poisoned").push(Arc::downgrade(&group));
        Ok(group)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ResidencyInner> {
        self.inner.lock().expect("the residency set is poisoned")
    }

    /// Add and commit under one lock, so a commit never races an add.
    fn admit<'a>(&self, buffers: impl Iterator<Item = &'a ProtocolObject<dyn MTLBuffer>>) {
        let inner = self.lock();
        for buffer in buffers {
            inner.set.addAllocation(ProtocolObject::from_ref(buffer));
        }
        inner.set.commit();
        inner.set.requestResidency();
    }

    /// Remove and commit; not `blocking` (the emergency), `false` where the lock is held.
    fn retire<'a>(&self, buffers: impl Iterator<Item = &'a ProtocolObject<dyn MTLBuffer>>, blocking: bool) -> bool {
        let inner = if blocking {
            self.lock()
        } else {
            match self.inner.try_lock() {
                Ok(inner) => inner,
                Err(_) => return false,
            }
        };
        for buffer in buffers {
            inner.set.removeAllocation(ProtocolObject::from_ref(buffer));
        }
        inner.set.commit();
        true
    }

    /// Idempotent per queue.
    fn attach(&self, queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>) {
        let mut inner = self.lock();
        if inner.queues.iter().any(|q| Retained::as_ptr(q) == Retained::as_ptr(queue)) {
            return;
        }
        queue.addResidencySet(&inner.set);
        inner.queues.push(queue.clone());
    }
}

/// The present segments' buffers in index order: what a launch declares, a snapshot it holds.
pub type Declared = Arc<Vec<Retained<ProtocolObject<dyn MTLBuffer>>>>;

/// A region's object and mapping, and the queue that writes it: only that queue may declare the object.
#[derive(Clone)]
pub struct Lender {
    pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub base: std::ptr::NonNull<std::ffi::c_void>,
    pub bytes: usize,
    pub queue: usize,
}

// SAFETY: a Metal buffer is thread-safe, and the mapping lives as long as the region.
unsafe impl Send for Lender {}
unsafe impl Sync for Lender {}

struct Live {
    segments: Vec<Option<Segment>>,
    declared: Declared,
    /// The lender, from the first adopt; its object stays in the set.
    lender: Option<Lender>,
    /// `Some` while a segment is borrowed, for a launch on the lender's queue.
    declared_with_lender: Option<Declared>,
}

/// `segments × per_segment` units of one padded size; a segment is absent, taken or borrowed.
pub struct SegmentPool {
    unit_bytes: usize,
    per_segment: usize,
    live: Mutex<Live>,
    /// Each segment's base or null, a lock-free path for the CPU: set before any claim lands, nulled after the last.
    bases: Vec<AtomicPtr<u8>>,
    live_units: AtomicUsize,
    borrowed_units: AtomicUsize,
    set: Arc<Residency>,
    /// A unit's GPU address, null where absent; only under [`Addressing::Table`].
    table: Option<Fixed<u64>>,
}

// SAFETY: every move holds `live`'s lock or the drop's `&mut self`; a racing table read sees null or a valid address.
unsafe impl Send for SegmentPool {}
unsafe impl Sync for SegmentPool {}

impl SegmentPool {
    /// All absent until taken, on its own residency set, in the registry [`release_all`] walks.
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>, unit_bytes: usize, segments: usize, per_segment: usize,
        addressing: Addressing,
    ) -> Result<Arc<Self>> {
        if unit_bytes == 0 || segments == 0 || per_segment == 0 {
            return Err(Error::Layout(format!(
                "SegmentPool: unit_bytes={unit_bytes} segments={segments} per_segment={per_segment}"
            )));
        }
        let set = Residency::new(device)?;
        let unit_bytes = padded_unit(unit_bytes);
        let count = segments * per_segment;
        let table = (addressing == Addressing::Table).then(|| Fixed::new(device, count, 0)).transpose()?;
        let pool = Arc::new(Self {
            unit_bytes,
            per_segment,
            live: Mutex::new(Live {
                segments: (0..segments).map(|_| None).collect(),
                declared: Arc::new(Vec::new()),
                lender: None,
                declared_with_lender: None,
            }),
            bases: (0..segments).map(|_| AtomicPtr::new(std::ptr::null_mut())).collect(),
            live_units: AtomicUsize::new(0),
            borrowed_units: AtomicUsize::new(0),
            set,
            table,
        });
        let mut pools = REGISTRY.lock().expect("the pool registry is poisoned");
        pools.retain(|p| p.strong_count() > 0);
        pools.push(Arc::downgrade(&pool));
        drop(pools);
        Ok(pool)
    }

    /// The plan's line, then the machine's floor.
    fn admit(&self, len: u64) -> Result<()> {
        if let Some((live, limit)) = crate::machine::PLAN.claim(len) {
            return Err(crate::machine::refuse(crate::machine::Refused::Ceiling, len, live, limit).into());
        }
        self.over_the_floor(len)
    }

    /// The floor's refusal of `len` more, its claim given back.
    fn over_the_floor(&self, len: u64) -> Result<()> {
        if let Some((left, floor)) = crate::machine::short_for(len) {
            self.give_back(true, len);
            return Err(crate::machine::refuse(crate::machine::Refused::Floor, len, left, floor).into());
        }
        Ok(())
    }

    /// Give back [`Self::admit`]'s claim, which only a wired take made: a release, or a take that could not map.
    fn give_back(&self, wired: bool, len: u64) {
        if wired {
            crate::machine::PLAN.release(len);
        }
    }

    /// Segment `i`'s table entries, from `base` or null (`base` 0).
    fn table_set(&self, i: usize, units: usize, base: u64) {
        let Some(table) = &self.table else { return };
        for u in 0..units {
            let at = if base == 0 { 0 } else { base + (u * self.unit_bytes) as u64 };
            // SAFETY: the table holds every unit; written under the lock, before any claim and after the last.
            unsafe { table.ptr().add(i * self.per_segment + u).write(at) };
        }
    }

    /// Mapped, wrapped, addressed and, where `wire`, locked and admitted; unwired for the census, as a wired take
    /// would grow the GPU driver's allocation for good.
    pub fn take(&self, device: &ProtocolObject<dyn MTLDevice>, i: usize, units: usize, wire: bool) -> Result<()> {
        if units == 0 || units > self.per_segment {
            return Err(Error::Layout(format!(
                "SegmentPool::take: {units} units of a {}-unit segment",
                self.per_segment
            )));
        }
        let len = units * self.unit_bytes;
        // Checked before the claim, so a present segment's take marks no peak; and again under the lock it maps.
        if self.live.lock().expect("segment pool poisoned").segments[i].is_some() {
            return Err(Error::Layout(format!("SegmentPool::take: segment {i} is present")));
        }
        if wire {
            self.admit(len as u64)?;
        }
        let mut live = self.live.lock().expect("segment pool poisoned");
        if live.segments[i].is_some() {
            self.give_back(wire, len as u64);
            return Err(Error::Layout(format!("SegmentPool::take: segment {i} is present")));
        }
        let raw = map_anon(i, len).inspect_err(|_| self.give_back(wire, len as u64))?;
        self.wire_and_wrap(device, &mut live, i, raw, units, wire)
    }

    /// Every take's tail: lock (where `wire`), wrap, admit, address, publish, or give the mapping and its claim back.
    fn wire_and_wrap(
        &self, device: &ProtocolObject<dyn MTLDevice>, live: &mut Live, i: usize, raw: *mut std::ffi::c_void,
        units: usize, wire: bool,
    ) -> Result<()> {
        let len = units * self.unit_bytes;
        let ptr = NonNull::new(raw).expect("mmap returned non-null");
        // Locked: a residency set is only a request, which the GPU driver revokes under pressure.
        // SAFETY: the mapping was made above, page-aligned and `len` long.
        if wire && unsafe { libc::mlock(raw, len) } != 0 {
            let why = std::io::Error::last_os_error();
            // SAFETY: the mapping was made above and wraps nothing yet.
            unsafe { libc::munmap(raw, len) };
            self.give_back(wire, len as u64);
            let m = crate::machine::MemorySnapshot::read().unwrap_or_default();
            let limit = m.machine_wire_limit.min(m.task_wire_limit);
            let refused = crate::machine::refuse(crate::machine::Refused::Lock, len as u64, m.own_wired, limit);
            tracing::info!(target: "oscar::segments",
                "segment {i} not locked ({why}), pages given back; {:.2} GB held, limits {:.2}/{:.2} GB; {}",
                crate::machine::PLAN.held() as f64 / crate::machine::GB,
                m.machine_wire_limit as f64 / crate::machine::GB,
                m.task_wire_limit as f64 / crate::machine::GB,
                m.ledger(),
            );
            return Err(refused.into());
        }
        // SAFETY: the mapping is page-aligned and `len` long; the buffer goes before the unmap.
        let buffer = unsafe { Self::wrap(device, ptr, len) };
        let Some(buffer) = buffer else {
            // SAFETY: the mapping was made above and wraps nothing yet.
            unsafe { libc::munmap(raw, len) };
            self.give_back(wire, len as u64);
            return Err(Error::Metal(format!("newBufferWithBytesNoCopy for segment {i} returned nil")));
        };
        if wire {
            self.set.admit(std::iter::once(&*buffer));
        }
        if self.table.is_some() {
            self.table_set(i, units, buffer.gpuAddress());
        }
        self.bases[i].store(raw as *mut u8, Ordering::Release);
        live.segments[i] = Some(Segment { ptr, units, wired: wire, buffer: Some(buffer), borrowed: false });
        self.live_units.fetch_add(units, Ordering::Release);
        Self::publish(live);
        Ok(())
    }

    /// Segment `i` at `offset` in the lender's object, nothing mapped, locked or charged; the caller owns the range.
    pub fn adopt(&self, i: usize, units: usize, lender: &Lender, offset: usize) -> Result<()> {
        let len = units * self.unit_bytes;
        if units == 0 || units > self.per_segment || !offset.is_multiple_of(PAGE) || offset + len > lender.bytes {
            return Err(Error::Layout(format!(
                "SegmentPool::adopt: {units} units of a {}-unit segment at {offset} of a {}-byte lender",
                self.per_segment, lender.bytes
            )));
        }
        if self.table.is_none() {
            return Err(Error::Layout(
                "SegmentPool::adopt: a whole-buffer pool hands out buffers, and the lender's is not its own".into(),
            ));
        }
        let mut live = self.live.lock().expect("segment pool poisoned");
        if live.segments[i].is_some() {
            return Err(Error::Layout(format!("SegmentPool::adopt: segment {i} is present")));
        }
        match &live.lender {
            Some(l) if Retained::as_ptr(&l.buffer) != Retained::as_ptr(&lender.buffer) => {
                return Err(Error::Layout("SegmentPool::adopt: a pool borrows from one lender".into()));
            }
            Some(_) => {}
            None => {
                self.set.admit(std::iter::once(&*lender.buffer));
                live.lender = Some(lender.clone());
            }
        }
        // SAFETY: the range is inside the lender's mapping, checked above.
        let raw = unsafe { lender.base.as_ptr().cast::<u8>().add(offset) };
        self.table_set(i, units, lender.buffer.gpuAddress() + offset as u64);
        self.bases[i].store(raw, Ordering::Release);
        live.segments[i] = Some(Segment {
            ptr: NonNull::new(raw.cast()).expect("inside the lender's mapping"),
            units,
            wired: false,
            buffer: Some(lender.buffer.clone()),
            borrowed: true,
        });
        self.live_units.fetch_add(units, Ordering::Release);
        self.borrowed_units.fetch_add(units, Ordering::Release);
        Self::publish(&mut live);
        Ok(())
    }

    /// The caller owns the quiesce: no command buffer may still name the segment.
    pub fn release(&self, i: usize) -> Result<()> {
        let mut live = self.live.lock().expect("segment pool poisoned");
        let Some(seg) = live.segments[i].take() else {
            return Err(Error::Layout(format!("SegmentPool::release: segment {i} is absent")));
        };
        let len = seg.units * self.unit_bytes;
        self.bases[i].store(std::ptr::null_mut(), Ordering::Release);
        self.table_set(i, seg.units, 0);
        if seg.wired {
            self.set.retire(seg.buffer.iter().map(|b| &**b), true);
        }
        self.live_units.fetch_sub(seg.units, Ordering::Release);
        if seg.borrowed {
            self.borrowed_units.fetch_sub(seg.units, Ordering::Release);
        }
        Self::publish(&mut live);
        drop(live);
        // A borrowed segment's mapping and charge are the lender's.
        if seg.borrowed {
            return Ok(());
        }
        // SAFETY: each buffer over the mapping is dropped once, here, before the one unmap.
        drop(seg.buffer);
        unsafe { libc::munmap(seg.ptr.as_ptr(), len) };
        self.give_back(seg.wired, len as u64);
        Ok(())
    }

    /// # Safety
    /// `at` is page-aligned, `len` long, and outlives the buffer.
    unsafe fn wrap(
        device: &ProtocolObject<dyn MTLDevice>, at: NonNull<std::ffi::c_void>, len: usize,
    ) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        unsafe {
            device.newBufferWithBytesNoCopy_length_options_deallocator(
                at,
                len,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        }
    }

    fn publish(live: &mut Live) {
        let own: Vec<_> =
            live.segments.iter().flatten().filter(|s| !s.borrowed).filter_map(|s| s.buffer.clone()).collect();
        live.declared_with_lender = match &live.lender {
            Some(lender) if live.segments.iter().flatten().any(|s| s.borrowed) => {
                Some(Arc::new(own.iter().cloned().chain(std::iter::once(lender.buffer.clone())).collect()))
            }
            _ => None,
        };
        live.declared = Arc::new(own);
    }

    /// Idempotent per queue; the table's own set comes too.
    pub fn attach(&self, queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>) {
        if let Some(table) = self.table.as_ref() {
            table.attach(queue);
        }
        self.set.attach(queue);
    }

    /// The index space, in segments.
    pub fn segments(&self) -> usize {
        self.bases.len()
    }

    pub fn live_units(&self) -> usize {
        self.live_units.load(Ordering::Acquire)
    }

    pub fn units_of(&self, i: usize) -> usize {
        self.live.lock().expect("segment pool poisoned").segments[i].as_ref().map_or(0, |s| s.units)
    }

    pub fn unit_bytes(&self) -> usize {
        self.unit_bytes
    }

    pub fn per_segment(&self) -> usize {
        self.per_segment
    }

    /// Present units' bytes, the lender's borrowed ones excluded.
    pub fn bytes(&self) -> usize {
        (self.live_units() - self.borrowed_units.load(Ordering::Acquire)) * self.unit_bytes
    }

    /// Unit `i`'s bytes for the CPU; the unit must be present.
    pub fn ptr(&self, i: usize) -> *mut u8 {
        let base = self.bases[i / self.per_segment].load(Ordering::Acquire);
        debug_assert!(!base.is_null(), "unit {i} of an absent segment");
        // SAFETY: the unit is inside its segment's mapping.
        unsafe { base.add((i % self.per_segment) * self.unit_bytes) }
    }

    /// A slot and base per segment, and the declared list three times: a launch's, the next take's, the lender's.
    pub fn host_bytes(segments: usize) -> usize {
        use std::mem::size_of;
        let declared = 3 * size_of::<Retained<ProtocolObject<dyn MTLBuffer>>>();
        segments * (size_of::<Option<Segment>>() + size_of::<AtomicPtr<u8>>() + declared)
    }

    /// The pool's own buffers, a snapshot: what a launch declares.
    pub fn declared(&self) -> Declared {
        self.live.lock().expect("segment pool poisoned").declared.clone()
    }

    /// On the lender's queue alone, the lender's object too: another queue declaring it serialises behind its writes.
    pub fn declared_on(&self, queue: usize) -> Declared {
        let live = self.live.lock().expect("segment pool poisoned");
        match (&live.lender, &live.declared_with_lender) {
            (Some(lender), Some(with)) if lender.queue == queue => with.clone(),
            _ => live.declared.clone(),
        }
    }

    pub fn borrowed_bytes(&self) -> usize {
        self.borrowed_units.load(Ordering::Acquire) * self.unit_bytes
    }

    /// `None` while absent.
    pub fn buffer(&self, i: usize) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        self.live.lock().expect("segment pool poisoned").segments[i].as_ref().and_then(|s| s.buffer.clone())
    }

    /// Unit `i`'s GPU address, null when absent; [`Addressing::Table`] only.
    pub fn address(&self, i: usize) -> u64 {
        let table = self.table.as_ref().expect("address() on a whole-buffer pool");
        debug_assert!(i < table.count());
        // SAFETY: the table holds every unit; a racing take or release leaves null or a valid address.
        unsafe { table.ptr().add(i).read_volatile() }
    }

    /// The address table, a u64 per unit, for rows a kernel resolves.
    pub fn address_buffer(&self) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        self.table.as_ref().expect("address_buffer() on a pool with no table").buffer()
    }
}

impl Drop for SegmentPool {
    fn drop(&mut self) {
        // The caller owns the quiesce; this pool's buffers leave the shared set, never the set itself.
        let live = self.live.get_mut().expect("segment pool poisoned");
        // A borrowed segment's bytes are the lender's.
        live.segments.iter_mut().filter(|s| s.as_ref().is_some_and(|s| s.borrowed)).for_each(|s| *s = None);
        self.set.retire(live.segments.iter().flatten().filter_map(|s| s.buffer.as_deref()), true);
        live.declared = Arc::new(Vec::new());
        for seg in live.segments.iter_mut().filter_map(Option::take) {
            let len = seg.units * self.unit_bytes;
            // SAFETY: each buffer is dropped once, before its mapping's one unmap.
            drop(seg.buffer);
            unsafe { libc::munmap(seg.ptr.as_ptr(), len) };
            if seg.wired {
                crate::machine::PLAN.release(len as u64);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{device, test_lock};

    #[test]
    fn segments_are_taken_released_distinct_and_addressed_and_a_whole_buffer_pool_has_no_table() {
        let _g = test_lock();
        let device = device();
        let whole = SegmentPool::new(&device, PAGE, 4, 64, Addressing::Whole).expect("a whole pool");
        assert!(whole.table.is_none(), "no table under Addressing::Whole");
        whole.take(&device, 1, 8, true).unwrap();
        assert_eq!((whole.live_units(), whole.buffer(1).is_some()), (8, true));
        whole.release(1).unwrap();
        assert_eq!(whole.live_units(), 0);
        let pool = SegmentPool::new(&device, 100, 3, 2, Addressing::Table).expect("three segments of two units");
        assert!(pool.table.is_some(), "a table under Addressing::Table");
        assert_eq!(pool.unit_bytes(), PAGE, "a unit pads to whole pages");
        assert_eq!((pool.segments() * pool.per_segment(), pool.live_units(), pool.bytes()), (6, 0, 0));
        assert!(pool.declared().is_empty());
        pool.take(&device, 0, 2, true).unwrap();
        pool.take(&device, 2, 1, true).unwrap();
        assert!(pool.take(&device, 2, 1, true).is_err(), "a present segment is not taken twice");
        assert!(pool.take(&device, 1, 3, true).is_err(), "no more than a segment's units");
        assert_eq!((pool.live_units(), pool.bytes(), pool.units_of(1)), (3, 3 * PAGE, 0));
        for i in [0, 1, 4] {
            // SAFETY: the unit is present and CPU-writable.
            unsafe { pool.ptr(i).write_bytes(0xA0 + i as u8, 100) };
        }
        for i in [0, 1, 4] {
            // SAFETY: as above.
            let got = unsafe { std::slice::from_raw_parts(pool.ptr(i), 100) };
            assert!(got.iter().all(|&b| b == 0xA0 + i as u8), "unit {i} holds its own bytes");
        }
        let declared = pool.declared();
        assert_eq!(declared.len(), 2, "two present segments declared");
        let (base, second) = (declared[0].gpuAddress(), declared[1].gpuAddress());
        assert_eq!((pool.address(0), pool.address(1), pool.address(4)), (base, base + PAGE as u64, second));
        assert_eq!((pool.address(2), pool.address(3), pool.address(5)), (0, 0, 0), "absent units are null");
        pool.release(0).unwrap();
        assert!(pool.release(0).is_err(), "an absent segment is not released");
        assert_eq!((pool.live_units(), pool.address(0), pool.declared().len()), (1, 0, 1));
        assert!(Arc::ptr_eq(&declared, &declared), "the launch's snapshot outlives the release");
        pool.take(&device, 0, 1, true).unwrap();
        assert_eq!(pool.live_units(), 2);
        assert_ne!(pool.address(0), 0);
        // Idempotent, as a queue takes at most 32 sets.
        let queue = device.newCommandQueue().expect("a queue");
        let before = pool.set.lock().queues.len();
        pool.attach(&queue);
        pool.attach(&queue);
        assert_eq!(pool.set.lock().queues.len(), before + 1, "attach is idempotent per queue");
    }

    #[test]
    fn a_borrowed_segment_is_the_lenders_bytes_and_its_release_gives_nothing_back() {
        let _g = test_lock();
        let device = device();
        let region = SegmentPool::new(&device, PAGE, 1, 16, Addressing::Whole).expect("the lender");
        region.take(&device, 0, 16, true).unwrap();
        let lender = Lender {
            buffer: region.buffer(0).expect("the lender's buffer"),
            base: NonNull::new(region.ptr(0).cast()).expect("the lender's mapping"),
            bytes: 16 * PAGE,
            queue: 7,
        };
        let pool = SegmentPool::new(&device, PAGE, 3, 4, Addressing::Table).expect("a table pool");
        pool.take(&device, 0, 4, true).unwrap();
        assert!(pool.adopt(2, 4, &lender, 8 * PAGE + 1).is_err(), "at a page offset");
        assert!(pool.adopt(2, 4, &lender, 13 * PAGE).is_err(), "inside the lender");
        assert!(pool.adopt(0, 4, &lender, 8 * PAGE).is_err(), "never over a present segment");
        pool.adopt(2, 4, &lender, 8 * PAGE).expect("segment 2 inside the lender");
        let own = (pool.live_units(), pool.bytes(), pool.borrowed_bytes());
        assert_eq!(own, (8, 4 * PAGE, 4 * PAGE), "the pool's own bytes leave the lender's out");
        let lent_at = lender.buffer.gpuAddress() + (9 * PAGE) as u64;
        assert_eq!(pool.address(9), lent_at, "a unit resolves into the lender's object");
        assert_eq!(pool.ptr(9) as usize, lender.base.as_ptr() as usize + 9 * PAGE);
        // SAFETY: the unit is present and CPU-writable.
        unsafe { pool.ptr(9).write_bytes(0xB7, PAGE) };
        let declared = (pool.declared().len(), pool.declared_on(7).len(), pool.declared_on(0).len());
        assert_eq!(declared, (1, 2, 1), "only the lender's queue declares its object");
        assert!(pool.take(&device, 2, 4, true).is_err(), "nor taken over");
        pool.release(2).unwrap();
        let after = (pool.live_units(), pool.address(9), pool.borrowed_bytes(), pool.declared_on(7).len());
        assert_eq!(after, (4, 0, 0, 1));
        // SAFETY: the lender's segment is present; the release unmapped nothing.
        let lent = unsafe { std::slice::from_raw_parts(region.ptr(9), PAGE) };
        assert!(lent.iter().all(|&b| b == 0xB7), "the lender's bytes stand through the release");
        pool.adopt(1, 2, &lender, 4 * PAGE).expect("adopted again, anywhere above the bound");
        drop(pool);
        let lent = unsafe { std::slice::from_raw_parts(region.ptr(9), PAGE) };
        assert!(lent.iter().all(|&b| b == 0xB7), "and through the pool's drop");
    }
}
