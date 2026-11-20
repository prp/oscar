//! A call's resolve over its layer's three banks: plan, push, wait and pin; the offers; the pins a launch holds until
//! its command buffer completes; the prefill-ahead thread.

use std::sync::Arc;
use std::sync::Mutex;

use candle_core::MetalDevice;
use objc2_metal::MTLSharedEvent;
use oscar_kernels::ExpertRow;
use oscar_offload::segments::Declared;

use super::bank::{pin_all_blocking, Bank, LayerExpertKey, PinOrder, Plan};
use super::reads::{demand_clock, note_demand_tick, FetchClass, Lane};
use super::{Error, ExpertPool, Worker};

/// A layer's three banks at a call: their common capacity and the segments every launch on them declares.
pub struct BankPools {
    pub metal_device: MetalDevice,
    pub min_capacity: usize,
    declared: [Declared; 3],
}

impl BankPools {
    pub fn new(device: &MetalDevice, banks: [&Arc<Bank>; 3]) -> Self {
        let min_capacity = banks.iter().map(|b| b.capacity()).min().expect("three banks");
        let declared = banks.map(|b| b.pool().declared_on(device.queue_key()));
        Self { metal_device: device.clone(), min_capacity, declared }
    }

    /// The three rows as the chain takes them, each declaring its bank's segments.
    pub fn rows<'r>(&'r self, rows: [&'r candle_core::Tensor; 3]) -> [ExpertRow<'r>; 3] {
        [0, 1, 2].map(|i| ExpertRow { row: rows[i], segments: &self.declared[i] })
    }
}

/// `(generation, bank, slots)`, pinned one command buffer, not one token, as token-long pins need `hot_budget >= topk`.
#[derive(Default)]
pub struct PinLedger(Mutex<Vec<(u64, Arc<Bank>, Vec<usize>)>>);

impl PinLedger {
    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<(u64, Arc<Bank>, Vec<usize>)>> {
        self.0.lock().expect("pin ledger poisoned")
    }

    pub fn push(&self, need: u64, bank: Arc<Bank>, slots: Vec<usize>) {
        self.lock().push((need, bank, slots));
    }

    pub fn stamp(&self, need: u64) {
        self.lock().iter_mut().for_each(|e| e.0 = need);
    }

    /// The entries the GPU reached (`all`: every one), each slot fenced so it stays so until its kernel finishes;
    /// out, the slots released and the highest fence written.
    pub fn release(&self, metal_device: &MetalDevice, all: bool) -> candle_core::Result<(usize, u64)> {
        let event = metal_device.completion_event()?;
        // Completed, not committed: a reader blocking on an uncommitted generation can be why it never commits.
        let reached = event.signaledValue();
        let drained: Vec<_> = self.lock().extract_if(.., |&mut (need, _, _)| all || need <= reached).collect();
        let (mut released, mut highest) = (0, 0);
        for (need, bank, slots) in drained {
            bank.set_completion_event(&event);
            bank.fence(&slots, need.max(reached));
            bank.unpin(&slots);
            released += slots.len();
            highest = highest.max(need.max(reached));
        }
        Ok((released, highest))
    }
}

/// Pins left at a run's end or on an uncommitted error path are unpinned unfenced.
impl Drop for PinLedger {
    fn drop(&mut self) {
        for (_, bank, slots) in self.lock().drain(..) {
            bank.unpin(&slots);
        }
    }
}

/// The batch's pins fenced at its flush's generation, so no eviction overwrites a slot before its kernel ran.
pub fn commit_pins(metal_device: &MetalDevice, pins: &PinLedger) -> candle_core::Result<()> {
    // `next_completion_generation` does not under-name a flush that commits several pool entries.
    pins.stamp(metal_device.next_completion_generation()?);
    let flushed = metal_device.flush_commands();
    pins.release(metal_device, true)?;
    flushed
}

/// How a wave's keys stood at its resolve, over all three banks: `in_flight` delivered late, `cold` never issued.
#[derive(Debug, Clone, Copy, Default)]
pub struct WaveResidency {
    pub resident: u32,
    pub in_flight: u32,
    pub cold: u32,
}

/// What failed in a resolve, and where.
pub type ResolveError = (&'static str, Error);

impl ExpertPool {
    /// Each key a staged read on its three banks at its deadline, at most `cold` of them not resident on the gate bank.
    /// No congestion gate: a speculative read that lands is a demand read that never happens.
    pub fn offer(&self, mut cold: usize, keys: impl IntoIterator<Item = (LayerExpertKey, u32)>) {
        let mut reads = Vec::new();
        for (key, deadline) in keys {
            let banks = self.banks_for(key.0 as usize);
            if cold != usize::MAX && !banks[0].is_resident(key) {
                if cold == 0 {
                    continue;
                }
                cold -= 1;
            }
            reads.extend(banks.iter().filter_map(|bank| bank.staged_read(key, deadline)));
        }
        if !reads.is_empty() {
            self.queue.push_many(reads);
        }
    }

    /// Staged reads for each `(layer, deadline, experts)`; offering a window again every boundary is harmless.
    pub fn offer_window<'w>(&self, window: impl IntoIterator<Item = (usize, u32, &'w [u16])>) {
        let keys = window.into_iter().flat_map(|(l, deadline, es)| es.iter().map(move |&e| ((l as u16, e), deadline)));
        self.offer(usize::MAX, keys);
    }

    /// The `Reserved` plans pushed at `class`, due at `target_tick`, per bank: a later bank's failure must not strand
    /// earlier reservations, their keys in flight for good; a demand plan is the resolve's count.
    fn plan_push(
        &self, banks: [&Arc<Bank>; 3], keys: &[LayerExpertKey], target_tick: u32, class: FetchClass,
    ) -> std::result::Result<WaveResidency, ResolveError> {
        let mut out = WaveResidency::default();
        for bank in banks {
            let plans =
                bank.push_demand(&self.queue, keys, target_tick, class).map_err(|e| ("plan_demand_fetch_batch", e))?;
            if class == FetchClass::Demand {
                Bank::note_resolve(&plans);
            }
            for plan in &plans {
                match plan {
                    Plan::Resident(_) => out.resident += 1,
                    Plan::Waiting => out.in_flight += 1,
                    Plan::Reserved { .. } => out.cold += 1,
                }
            }
        }
        Ok(out)
    }

    /// Plan, push and wait, then pin every key on every bank in `order`, all or nothing; `timed` gets the fetch's wait
    /// and the pins' time.
    pub fn resolve_blocking(
        &self, banks: [&Arc<Bank>; 3], keys: &[LayerExpertKey], target_tick: u32, order: PinOrder, lane: Lane,
        timed: Option<&dyn Fn(u64, u64)>,
    ) -> std::result::Result<([Vec<usize>; 3], WaveResidency), ResolveError> {
        note_demand_tick(target_tick);
        let residency = self.plan_push(banks, keys, target_tick, FetchClass::Demand)?;
        let t_wait = std::time::Instant::now();
        for bank in banks {
            bank.wait(keys, lane);
        }
        let fetch_ns = t_wait.elapsed().as_nanos() as u64;
        let t_pin = timed.map(|_| std::time::Instant::now());
        let slots = pin_all_blocking(banks, &self.queue, keys, target_tick, order, lane).map_err(|e| ("pin", e))?;
        if let Some(timed) = timed {
            timed(fetch_ns, t_pin.map_or(0, |t| t.elapsed().as_nanos() as u64));
        }
        Ok((slots, residency))
    }
}

/// Slot 0 for every key, with no fetch, wait or pin: a census submits nothing, so a row need only be well formed.
pub fn resolve_census(keys: &[LayerExpertKey]) -> ([Vec<usize>; 3], WaveResidency) {
    (std::array::from_fn(|_| vec![0usize; keys.len()]), WaveResidency::default())
}

/// Plans and pushes the prefill stream off the engine thread; an error drops the job, as the wave's resolve plans again.
pub(crate) type PrefetchThread = Worker<std::sync::mpsc::SyncSender<PrefetchJob>>;

pub struct PrefetchJob {
    pub keys: Vec<LayerExpertKey>,
    pub due: Due,
    pub layer: usize,
}

/// A certain wave reserves through the demand plan; a predicted one is only offered, for admission to refuse.
pub enum Due {
    /// A tick of this forward's schedule.
    At(u32),
    /// Ticks past the demand clock at the offer, a high-water mark a follow-up turn's schedule may sit behind.
    Ahead(u32),
}

impl ExpertPool {
    pub(crate) fn spawn_prefetch(self: &Arc<Self>) -> std::result::Result<PrefetchThread, Error> {
        let pool = Arc::clone(self);
        let channel = std::sync::mpsc::sync_channel::<PrefetchJob>(512);
        Worker::spawn("oscar-prefill-ahead", channel, move |rx: std::sync::mpsc::Receiver<PrefetchJob>| {
            while let Ok(job) = rx.recv() {
                match job.due {
                    Due::At(tick) => {
                        let _ = pool.plan_push(pool.banks_for(job.layer), &job.keys, tick, FetchClass::Reserved);
                    }
                    Due::Ahead(lead) => {
                        let deadline = demand_clock().saturating_add(lead);
                        pool.offer(usize::MAX, job.keys.iter().map(|&key| (key, deadline)));
                    }
                }
            }
        })
        .map_err(|e| super::layout_error(format!("spawn oscar-prefill-ahead: {e}")))
    }
}
