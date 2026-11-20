//! The reads: outstanding fills, smallest `(deadline, class, push order)` first, and the readers draining them.

use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use oscar_offload::io::{ShardId, PENDING};

use super::bank::{Bank, LayerExpertKey};
use super::counters::COUNTERS;
use crate::gpu::ExpertExtent;

/// Which thread waits: the engine's (the forward, the resolver, the readers), which the barrier counts, or the drafter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lane {
    Engine,
    Drafter,
}

/// The highest tick demand was planned for, the reads' "now": an engine that demanded nothing has nothing late.
static DEMAND_CLOCK: AtomicU32 = AtomicU32::new(0);

/// Relaxed: the clock steers timeliness, never correctness.
pub fn note_demand_tick(tick: u32) {
    DEMAND_CLOCK.fetch_max(tick, Ordering::Relaxed);
}

pub fn demand_clock() -> u32 {
    DEMAND_CLOCK.load(Ordering::Relaxed)
}

/// At prefill→decode: a waved prefill leaves the clock far above decode's ticks, past every staged read's deadline.
pub fn reset_demand_clock() {
    DEMAND_CLOCK.store(0, Ordering::Relaxed);
}

/// Who owns a read's slot and what a missed deadline means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchClass {
    /// The engine is, or will be, parked on this key.
    Demand,
    /// Speculative with a reserved slot (prefill run-ahead): a waiter may park on it, so past due it is demand.
    Reserved,
    /// An offer without a slot, admitted at the pop (a fresher victim decision): past its deadline it drops for free.
    Staged,
}

/// The extent's fields flat, as the host line charges a queued read by its size.
pub struct FetchReq {
    pub shard_id: ShardId,
    pub offset: u64,
    pub len: usize,
    /// A `Staged` read takes its slot at the pop.
    pub slot: usize,
    pub key: LayerExpertKey,
    pub class: FetchClass,
    pub bank: Arc<Bank>,
    /// A tick: demand's now, a prefetch's predicted use.
    pub deadline: u32,
}

impl FetchReq {
    pub fn new(
        bank: &Arc<Bank>, key: LayerExpertKey, extent: ExpertExtent, slot: usize, class: FetchClass, deadline: u32,
    ) -> Self {
        let ExpertExtent { shard_id, file_offset: offset, byte_count: len } = extent;
        Self { shard_id, offset, len, slot, key, class, bank: Arc::clone(bank), deadline }
    }

    /// At an equal deadline demand pops first: a stage pushed a layer earlier would front-run that tick's demand.
    fn order(&self, serial: u64) -> (u32, u8, u64) {
        (self.deadline, (self.class != FetchClass::Demand) as u8, serial)
    }

    fn staged_extent(&self) -> Option<(ShardId, u64)> {
        (self.class == FetchClass::Staged).then_some((self.shard_id, self.offset))
    }
}

/// Smallest order first; the push `Instant` beside each read.
type Pending = BTreeMap<(u32, u8, u64), (FetchReq, Instant)>;

#[derive(Default)]
struct Queued {
    pending: Pending,
    /// A window offered again whole carries the same deadlines, so deduping loses no urgency; demand always enqueues.
    staged: HashSet<(ShardId, u64)>,
    next_serial: u64,
    shutdown: bool,
}

/// Deadline order fits no lock-free primitive.
#[derive(Default)]
pub struct FetchQueue {
    queued: Mutex<Queued>,
    cv: Condvar,
}

impl FetchQueue {
    /// At `depth` pending, for the host line; the bounce buffers are [`oscar_offload::io::bounce_bytes`].
    pub fn host_bytes(depth: usize) -> usize {
        depth * std::mem::size_of::<((u32, u8, u64), (FetchReq, Instant))>()
            + oscar_offload::machine::hash_map_bytes::<(ShardId, u64), ()>(depth)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Queued> {
        self.queued.lock().expect("FetchQueue state poisoned")
    }

    /// Under one lock and one `notify_all`, each stamped with its push `Instant` and a FIFO serial.
    pub fn push_many(&self, reqs: impl IntoIterator<Item = FetchReq>) {
        let mut q = self.lock();
        let at = Instant::now();
        let mut any = false;
        for req in reqs {
            if req.staged_extent().is_some_and(|extent| !q.staged.insert(extent)) {
                continue;
            }
            any = true;
            let serial = q.next_serial;
            q.next_serial = serial.wrapping_add(1);
            q.pending.insert(req.order(serial), (req, at));
            PENDING.fetch_add(1, Ordering::Relaxed);
        }
        if any {
            self.cv.notify_all();
        }
    }

    /// The next read with its push `Instant`; `None` once shut down and drained (`wait`: `None` at once when empty).
    fn pop(&self, wait: bool) -> Option<(FetchReq, Instant)> {
        let mut q = self.lock();
        loop {
            if let Some((_, (req, at))) = q.pending.pop_first() {
                // A fresh offer for this extent is meaningful again.
                if let Some(extent) = req.staged_extent() {
                    q.staged.remove(&extent);
                }
                PENDING.fetch_sub(1, Ordering::Relaxed);
                return Some((req, at));
            }
            if q.shutdown || !wait {
                return None;
            }
            q = self.cv.wait(q).expect("FetchQueue cv poisoned");
        }
    }

    pub fn len(&self) -> usize {
        self.lock().pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn shutdown(&self) {
        self.lock().shutdown = true;
        self.cv.notify_all();
    }
}

/// A read is served by the popping reader's drive: a reader pops only when free, so each drive drains at its pace.
pub struct Readers {
    queue: Arc<FetchQueue>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl Readers {
    /// `readers[d]` threads on drive `d`, each drive saturating at its own concurrency.
    pub fn new(queue: Arc<FetchQueue>, readers: &[usize]) -> Self {
        assert!(readers.iter().sum::<usize>() > 0, "IoWorkerPool needs at least one worker");
        let threads = readers
            .iter()
            .enumerate()
            .flat_map(|(d, &n)| (0..n).map(move |i| (d, i)))
            .map(|(d, i)| {
                let q = queue.clone();
                std::thread::Builder::new()
                    .name(format!("oscar-io-{d}-{i}"))
                    .spawn(move || {
                        while let Some((req, at)) = q.pop(true) {
                            fill(req, at, d as u8);
                        }
                    })
                    .expect("spawn IoWorkerPool worker")
            })
            .collect();
        Self { queue, threads }
    }

    pub fn count(&self) -> usize {
        self.threads.len()
    }
}

impl Drop for Readers {
    fn drop(&mut self) {
        self.queue.shutdown();
        for h in self.threads.drain(..) {
            let _ = h.join();
        }
    }
}

/// One read whatever the class: decode IO is bytes-bound, so ordering reads shuffles who waits, never the total.
fn fill(req: FetchReq, enqueued: Instant, device: u8) {
    let (bank, key) = (&req.bank, req.key);
    // Read once, so the read gets one verdict: past once demand is planned to its deadline.
    let now = demand_clock();
    let past = now >= req.deadline;
    let demand = req.class == FetchClass::Demand || (req.class == FetchClass::Reserved && past);
    // Demand only, the one class the barrier waits on; each lap's end starts the next.
    let mut mark = demand.then_some(enqueued);
    let mut lap = |total: &std::sync::atomic::AtomicU64| {
        if let Some(t0) = mark {
            let t = Instant::now();
            total.fetch_add(t.saturating_duration_since(t0).as_nanos() as u64, Ordering::Relaxed);
            mark = Some(t);
        }
    };
    lap(&COUNTERS.demand_queue_wait_ns);
    // A speculative fill never parks on a GPU fence: a slot not yet safe aborts it.
    let slot = if req.class == FetchClass::Staged {
        if past {
            return;
        }
        // Refused: the producer offers again next token, the pool churned by then.
        let Some(slot) = bank.stage(key, now, req.deadline) else { return };
        if !bank.gpu_safe_now(slot) {
            if let Err(e) = bank.abort(slot, key) {
                panic!("staged fill abort refused: {e}");
            }
            return;
        }
        slot
    } else {
        bank.wait_gpu_safe(req.slot);
        req.slot
    };
    lap(&COUNTERS.demand_gpu_safe_wait_ns);
    bank.wait_draft_safe(slot);
    let extent = ExpertExtent { shard_id: req.shard_id, file_offset: req.offset, byte_count: req.len };
    let read = bank.read_into(device, slot, extent);
    lap(&COUNTERS.demand_read_ns);
    match read {
        Ok(()) => {
            // A refused completion would leave the slot in flight for good, so it stops oscar.
            if let Err(e) = bank.publish(slot, key, !demand) {
                panic!("fill completion refused: {e}");
            }
            let done = if demand { &COUNTERS.completed_demand } else { &COUNTERS.completed_prefetch };
            done.fetch_add(1, Ordering::Relaxed);
        }
        Err(e) => {
            tracing::error!(target: "oscar::experts", "read of (L={}, E={}) into slot {slot} failed: {e}", key.0, key.1);
            COUNTERS.read_errors.fetch_add(1, Ordering::Relaxed);
            if let Err(e) = bank.abort(slot, key) {
                panic!("failed read's abort refused: {e}");
            }
        }
    }
}

#[cfg(test)]
mod tests;
