//! Candle's allocator: the fixed pool and a working region per command queue, as hazard tracking does not cross queues.

use crate::fixed::fixed_pool;
use crate::region::Region;
use crate::{Error, Result};
use candle_metal_kernels::metal::{Buffer, BufferPool, CommandQueue, Device, Fences, Use, STREAM_QUEUE_KEYS};
use candle_metal_kernels::MetalKernelError;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLDevice};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// One lock: regions and queues are read together, and two locks taken in both orders deadlock.
pub struct TensorPools {
    live: Mutex<Live>,
}

#[derive(Default)]
struct Live {
    /// One region per queue key, made at its first allocation.
    regions: HashMap<usize, Arc<Region>>,
    /// The queue each key names, which a region made later joins.
    queues: HashMap<usize, CommandQueue>,
    /// Device queues' regions: the census's, mapped only, until the plan draws them wired; zero bytes before.
    device: Sizing,
    /// Thread streams' regions: at the census's bound, mapped only, until the drafter's record sizes them wired.
    stream: Sizing,
}

#[derive(Default, Clone, Copy)]
struct Sizing {
    bytes: usize,
    wired: bool,
}

impl TensorPools {
    fn new() -> Self {
        Self { live: Mutex::new(Live::default()) }
    }

    fn live(&self) -> std::sync::MutexGuard<'_, Live> {
        self.live.lock().expect("tensor pools poisoned")
    }

    /// `queue`'s region, taken at its planned size on first use.
    fn region(&self, device: &ProtocolObject<dyn MTLDevice>, queue: usize, fences: &Fences) -> Result<Arc<Region>> {
        let mut live = self.live();
        if let Some(r) = live.regions.get(&queue) {
            return Ok(Arc::clone(r));
        }
        let Sizing { bytes, wired } = if queue >= STREAM_QUEUE_KEYS { live.stream } else { live.device };
        if bytes == 0 {
            return Err(Error::Layout(format!(
                "queue {queue} has no working region: the plan has not sized {}",
                if queue >= STREAM_QUEUE_KEYS {
                    "a stream's region - the census has not run"
                } else {
                    "the working memory"
                }
            )));
        }
        let made = Region::take(device, queue, bytes, fences.clone(), wired)?;
        if let Some(q) = live.queues.get(&queue) {
            made.attach(q);
        }
        live.regions.insert(queue, Arc::clone(&made));
        Ok(made)
    }

    /// Every device queue's region at `bytes` from here, re-taken where one exists; the first refusal comes back.
    pub fn plan_region(&self, bytes: usize) -> Result<()> {
        let regions: Vec<Arc<Region>> = {
            let mut live = self.live();
            live.device = Sizing { bytes, wired: true };
            live.regions.iter().filter(|(&q, _)| q < STREAM_QUEUE_KEYS).map(|(_, r)| Arc::clone(r)).collect()
        };
        regions.iter().try_for_each(|r| r.retake(bytes).map_err(|e| Error::Layout(format!("queue {}: {e}", r.queue()))))
    }

    /// The load's first stage: device regions mapped, not wired or charged, until the plan draws them.
    pub fn plan_census_region(&self, bytes: usize) {
        self.live().device = Sizing { bytes, wired: false };
    }

    /// The load's first stage: stream regions at the census's bound, mapped, not wired, until the record sizes them.
    pub fn plan_census_stream_region(&self) {
        let mut live = self.live();
        live.stream = Sizing { bytes: live.device.bytes, wired: false };
    }

    /// Stream regions at the record's size, wired, re-taken where one exists.
    pub fn plan_stream_region(&self, bytes: usize) -> Result<()> {
        self.live().stream = Sizing { bytes, wired: true };
        self.hold_stream_regions(true)
    }

    /// At the record's size or one page: the drafter's memory as the launch books it; each must be idle.
    pub fn hold_stream_regions(&self, held: bool) -> Result<()> {
        let (bytes, regions): (usize, Vec<Arc<Region>>) = {
            let live = self.live();
            (
                if held { live.stream.bytes } else { 0 },
                live.regions.iter().filter(|(&q, _)| q >= STREAM_QUEUE_KEYS).map(|(_, r)| Arc::clone(r)).collect(),
            )
        };
        regions
            .iter()
            .try_for_each(|r| r.retake(bytes).map_err(|e| Error::Layout(format!("stream {}: {e}", r.queue()))))
    }

    pub fn regions(&self) -> Vec<Arc<Region>> {
        let mut v: Vec<Arc<Region>> = self.live().regions.values().map(Arc::clone).collect();
        v.sort_by_key(|r| r.queue());
        v
    }

    /// Every region's placement: the census's starting state.
    pub fn snapshots(&self) -> HashMap<usize, crate::region::Placement> {
        self.live().regions.iter().map(|(&q, r)| (q, r.snapshot())).collect()
    }

    /// Summed over the regions, for the memory ledger.
    pub fn levels(&self) -> crate::region::Levels {
        let mut sum = crate::region::Levels::default();
        for r in self.regions() {
            let l = r.levels();
            sum.size += l.size;
            sum.lent += l.lent;
            sum.live += l.live;
            sum.asked += l.asked;
            sum.fenced += l.fenced;
        }
        sum
    }

    pub fn describe(&self) -> String {
        self.regions().iter().map(|r| r.describe()).collect()
    }
}

impl BufferPool for TensorPools {
    fn allocate(
        &self, device: &Device, bytes: usize, r#use: Use, queue: usize, fences: &Fences, name: &'static str,
    ) -> std::result::Result<Buffer, MetalKernelError> {
        let raw = device.as_ref();
        let made = match r#use {
            Use::Fixed => fixed_pool(raw).and_then(|p| p.take(raw, bytes)).map(|(b, padded)| {
                crate::census::note_take(bytes, padded, r#use, queue, name);
                Buffer::new(b)
            }),
            // A refused cut fails its request: under a plan that bounds every allocation it is a defect.
            Use::Scratch | Use::Staged => self.region(raw, queue, fences).and_then(|r| r.object(bytes, r#use, name)),
        };
        made.map_err(|e| MetalKernelError::FailedToCreateResource(e.to_string()))
    }

    fn declare(&self, command_buffer: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>, queue: &CommandQueue) {
        crate::segments::declare_pools(command_buffer, queue);
    }

    fn attach(&self, queue: &CommandQueue, key: usize) {
        let mut live = self.live();
        if live.queues.contains_key(&key) {
            return;
        }
        // The floor is read from every queue; a region by its own.
        if let Ok(fixed) = fixed_pool(&queue.device()) {
            fixed.attach(queue);
        }
        if let Some(region) = live.regions.get(&key) {
            region.attach(queue);
        }
        live.queues.insert(key, queue.clone());
    }
}

static POOLS: OnceLock<Arc<TensorPools>> = OnceLock::new();

/// The pools, installed as candle's one allocator. Idempotent.
pub fn install() -> Arc<TensorPools> {
    let pools = Arc::clone(POOLS.get_or_init(|| Arc::new(TensorPools::new())));
    candle_metal_kernels::metal::set_buffer_pool(Arc::clone(&pools) as Arc<dyn BufferPool>);
    pools
}

pub fn pools() -> Option<&'static Arc<TensorPools>> {
    POOLS.get()
}

/// A test process has no plan: a region per test stream, sized for a kernel test's largest object.
pub const TEST_REGION_BYTES: usize = 1 << 30;

/// The pools with test regions for the device's queue and every stream: a test process's one entry.
pub fn install_for_tests() -> Arc<TensorPools> {
    let pools = install();
    pools.plan_region(TEST_REGION_BYTES).expect("the test region");
    pools.plan_stream_region(TEST_REGION_BYTES).expect("the test streams' region");
    pools
}

/// A test region, and this thread on its own stream: the pool orders a thread's work only while alone on its queue.
pub fn test_device() -> candle_core::Result<candle_core::Device> {
    install_for_tests();
    let device = candle_core::Device::new_metal(0)?;
    if let candle_core::Device::Metal(md) = &device {
        md.install_thread_stream(None)?;
    }
    Ok(device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::machine::PAGE;
    use crate::test_support::{device, test_lock};
    use objc2::rc::Retained;
    use objc2_metal::MTLSharedEvent;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::MutexGuard;

    type Dev = Retained<ProtocolObject<dyn MTLDevice>>;
    type Event = Retained<ProtocolObject<dyn MTLSharedEvent>>;

    /// A queue's progress moved by hand, nothing behind it.
    fn fences(device: &ProtocolObject<dyn MTLDevice>, inflight: u64) -> (Fences, Arc<AtomicU64>, Event) {
        let committed = Arc::new(AtomicU64::new(0));
        let completed = device.newSharedEvent().expect("a shared event");
        // Nothing encoded: the release fence is the committed count.
        let encoded = Arc::new(vec![Arc::new(AtomicUsize::new(0))]);
        (Fences::new(Arc::clone(&committed), completed.clone(), inflight, encoded), committed, completed)
    }

    /// A region of `pages` under the lock, with [`fences`] of `inflight`.
    fn region(
        pages: usize, inflight: u64, wired: bool,
    ) -> (MutexGuard<'static, ()>, Dev, Arc<Region>, Arc<AtomicU64>, Event) {
        let (guard, device) = (test_lock(), device());
        let (f, committed, completed) = fences(&device, inflight);
        let region = Region::take(&device, 0, pages * PAGE, f, wired).expect("a region");
        (guard, device, region, committed, completed)
    }

    #[test]
    fn a_region_cuts_one_buffer_at_offsets_and_reuses_a_dropped_run_at_once() {
        let (_g, _device, region, committed, completed) = region(8, 2, true);
        let a = region.object(3 * PAGE, Use::Scratch, "a").expect("a");
        let b = region.object(PAGE, Use::Scratch, "b").expect("b");
        assert_eq!((a.base(), b.base(), a.length()), (0, 3 * PAGE, 3 * PAGE), "the second object follows the first");
        assert_eq!(b.contents() as usize, a.contents() as usize + 3 * PAGE);
        assert!(std::ptr::eq(a.as_ref(), b.as_ref()), "one buffer object");
        assert_eq!(region.levels().live, 4 * PAGE);
        drop(a);
        let c = region.object(2 * PAGE, Use::Scratch, "c").expect("c");
        assert_eq!(c.base(), 0, "the dropped run is cut again at once, lowest first");
        // A queue that has committed nothing has passed every stamp; after a commit, only what it completed.
        committed.store(5, Ordering::SeqCst);
        drop(c);
        assert_eq!(region.levels().fenced, 2 * PAGE, "the run just given back; the first was passed while idle");
        completed.setSignaledValue(7);
        let s = region.object(PAGE, Use::Staged, "s").expect("s");
        assert_eq!((s.base(), region.levels().fenced), (7 * PAGE, 0), "a host fill takes the top end, once passed");
        drop((b, s));
        assert_eq!(region.levels().live, 0);
    }

    #[test]
    fn a_region_refuses_what_it_cannot_cut_and_maps_nothing_for_it() {
        let (_g, _device, region, _c, _e) = region(4, 2, true);
        let a = region.object(3 * PAGE, Use::Scratch, "a").expect("a");
        let why = region.object(2 * PAGE, Use::Scratch, "b").expect_err("no run of two pages is free").to_string();
        assert!(why.contains("no free run"), "{why}");
        assert_eq!(region.levels().size, 4 * PAGE, "a refusal maps nothing");
        drop(a);
        region.object(4 * PAGE, Use::Scratch, "c").expect("the whole region, once its one object is gone");
        assert_eq!(region.levels().size, 4 * PAGE, "and a served cut maps nothing either");
    }

    #[test]
    fn a_regions_bound_moves_inside_its_one_mapping_and_only_while_idle() {
        let (_g, _device, region, _c, _e) = region(8, 2, true);
        let lender = region.lender();
        assert_eq!((lender.bytes, lender.queue, region.size(), region.bound()), (8 * PAGE, 0, 8 * PAGE, 8 * PAGE));
        let a = region.object(PAGE, Use::Scratch, "a").expect("a");
        let why = region.move_bound(2 * PAGE).expect_err("an object is in hand").to_string();
        assert!(why.contains("not idle"), "{why}");
        assert_eq!(region.bound(), 8 * PAGE);
        drop(a);
        region.move_bound(2 * PAGE).expect("idle: the bound moves");
        let levels = region.levels();
        assert_eq!((levels.size, levels.lent, region.bound()), (8 * PAGE, 6 * PAGE, 2 * PAGE));
        assert!(region.object(3 * PAGE, Use::Scratch, "wide").is_err(), "what is lent is not the region's to cut");
        let s = region.object(PAGE, Use::Staged, "s").expect("s");
        assert_eq!(s.base(), PAGE, "a host fill lands under the bound");
        assert!(std::ptr::eq(s.as_ref(), &*lender.buffer), "the one buffer object");
        drop(s);
        assert!(region.move_bound(9 * PAGE).is_err(), "never past what is mapped");
        region.move_bound(8 * PAGE).expect("and back");
        assert_eq!(region.levels().lent, 0);
        region.object(8 * PAGE, Use::Scratch, "whole").expect("the whole mapping is the region's again");
    }

    /// It can fail: compare the sizes alone and the census's mapping is never locked.
    #[test]
    fn the_plans_region_is_locked_even_at_the_size_the_census_mapped() {
        let (_g, _device, region, _c, _e) = region(8, 2, false);
        assert!(!region.is_locked());
        region.retake(8 * PAGE).expect("idle: re-taken");
        assert!(region.is_locked(), "the same size, locked now");
        let a = region.object(PAGE, Use::Scratch, "a").expect("a");
        region.retake(8 * PAGE).expect("locked at the plan's size already");
        let why = region.retake(4 * PAGE).expect_err("an object names its mapping").to_string();
        assert!(why.contains("1 object(s)"), "{why}");
        drop(a);
        region.retake(4 * PAGE).expect("idle: re-taken");
        assert_eq!(region.size(), 4 * PAGE);
    }

    /// Attaching reads every region, and a new region every queue.
    #[test]
    fn a_region_belongs_to_one_queue_and_attaching_and_allocating_at_once_makes_progress() {
        let _g = test_lock();
        let device = device();
        let pools = TensorPools::new();
        let (f, _c, _e) = fences(&device, 4);
        assert!(pools.region(&device, 0, &f).is_err(), "no plan, no region");
        pools.plan_region(4 * PAGE).expect("no region to re-take");
        let engine = pools.region(&device, 0, &f).expect("a region");
        let other = pools.region(&device, 1, &f).expect("a region");
        assert!(!Arc::ptr_eq(&engine, &other), "a queue's region is its own");
        assert!(pools.region(&device, STREAM_QUEUE_KEYS, &f).is_err(), "a stream's region waits for the record");
        pools.plan_stream_region(2 * PAGE).expect("no stream region to re-take");
        assert_eq!(pools.region(&device, STREAM_QUEUE_KEYS, &f).expect("a stream's region").size(), 2 * PAGE);
        assert_eq!(pools.regions().len(), 3);
        let pools = Arc::new(TensorPools::new());
        pools.plan_region(64 * PAGE).expect("the region");
        let (done, finished) = std::sync::mpsc::channel();
        for t in 0..8 {
            let (pools, done) = (Arc::clone(&pools), done.clone());
            std::thread::spawn(move || {
                let dev = Device::system_default().expect("a Metal device");
                for i in 0..16 {
                    let queue = dev.new_command_queue().expect("a queue");
                    let c = candle_metal_kernels::metal::Commands::new(queue.clone(), t, None).expect("commands");
                    let f = Fences::new(
                        c.completion_counter(),
                        c.completion_event(),
                        c.inflight_bound(),
                        c.encoded_counts(),
                    );
                    pools.attach(&queue, t);
                    drop(pools.allocate(&dev, (i + 1) * PAGE, Use::Scratch, t, &f, "test").expect("an object"));
                }
                let _ = done.send(t);
            });
        }
        drop(done);
        for _ in 0..8 {
            finished.recv_timeout(std::time::Duration::from_secs(30)).expect("a timeout here is the pools deadlocking");
        }
    }

    /// The phase is process-wide.
    #[test]
    fn a_phase_opens_at_what_the_region_already_holds() {
        let (_g, _device, region, _c, _e) = region(32, 4, true);
        let phase = crate::machine::phase();
        let kept = region.object(4 * PAGE, Use::Scratch, "kept").expect("kept");
        let gone = region.object(4 * PAGE, Use::Scratch, "gone").expect("gone");
        let above = region.object(4 * PAGE, Use::Scratch, "above").expect("above");
        drop(gone);
        let marks = region.marks();
        assert_eq!(marks.peak[phase].high, 12, "the high-water is what the region's two ends have taken");
        assert_eq!(marks.out[phase], 12 * PAGE);
        let _hole = region.object(2 * PAGE, Use::Scratch, "hole").expect("hole");
        assert_eq!(region.marks().peak[phase].high, 12, "a cut into a hole below the end moves nothing");
        let _wide = region.object(8 * PAGE, Use::Scratch, "wide").expect("wide");
        let peak = &region.marks().peak[phase];
        assert_eq!((peak.high, &peak.holders[..]), (20, &["kept", "hole", "above", "wide"][..]));
        let _host = region.object(PAGE, Use::Staged, "host").expect("host");
        assert_eq!(region.marks().peak[phase].high, 21, "and the other end is part of what the region must hold");
        drop((kept, above));
    }

    /// It can fail: mark it unconditionally and the first assertion reads the census's cut.
    #[test]
    fn a_censuss_cut_marks_no_phase() {
        let (_g, _device, region, _c, _e) = region(8, 4, true);
        let phase = crate::machine::phase();
        candle_metal_kernels::metal::pool::set_census(true);
        let armed = region.object(2 * PAGE, Use::Scratch, "census");
        candle_metal_kernels::metal::pool::set_census(false);
        let armed = armed.expect("an object");
        assert_eq!(region.marks().peak[phase].high, 0, "a census's cut marks nothing");
        // A request's cut marks, counting the census's run still in hand.
        let _request = region.object(PAGE, Use::Scratch, "request").expect("an object");
        let marks = region.marks();
        assert_eq!((marks.peak[phase].high, &marks.peak[phase].holders[..]), (3, &["census", "request"][..]));
        drop(armed);
    }
}
