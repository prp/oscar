//! The probe on candle's `CommandBufferObserver`: thread roles, the wake detector and the abort on an errored buffer.

use candle_metal_kernels::metal::command_buffer::{set_command_buffer_observer, CommandBufferObserver, Completion};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// A committing thread that is neither forward's: IO, fetch blits.
pub const CB_ROLE_OTHER: u8 = 0;
pub const CB_ROLE_ENGINE: u8 = 1;
pub const CB_ROLE_DRAFT: u8 = 2;
thread_local! {
    static CB_ROLE_TL: std::cell::Cell<u8> = const { std::cell::Cell::new(CB_ROLE_OTHER) };
}
pub fn mark_draft_thread() {
    CB_ROLE_TL.with(|c| c.set(CB_ROLE_DRAFT));
}
pub fn mark_engine_thread() {
    CB_ROLE_TL.with(|c| {
        if c.get() == CB_ROLE_OTHER {
            c.set(CB_ROLE_ENGINE);
        }
    });
}
#[inline]
pub fn cb_role() -> u8 {
    CB_ROLE_TL.with(|c| c.get())
}

pub fn arm_wake_detector() {
    WAKE_ARMED.store(1, Ordering::Relaxed);
}

// The wake detector: work-start latency, `kernelStartTime` to `GPUStartTime`, whose silent collapse nothing predicts.
static WAKE_ARMED: AtomicU64 = AtomicU64::new(0);
static WAKE_N: AtomicU64 = AtomicU64::new(0);
static WAKE_GPU_BUSY_US: AtomicU64 = AtomicU64::new(0);
static WAKE_WARNED: AtomicU64 = AtomicU64::new(0);
/// Bucket `i` holds `[2^i, 2^(i+1))` µs.
const WAKE_BUCKETS: usize = 16;
static WAKE_HIST: [AtomicU64; WAKE_BUCKETS] = [const { AtomicU64::new(0) }; WAKE_BUCKETS];
/// On the median, never a high quantile: the latency counts dependency waits, which paged decode has constantly.
const WAKE_ALERT_BUCKET: usize = 13;
/// Buffers skipped before the histogram opens: the first after prefill run cold (pipelines, pools, clock).
const WAKE_WARMUP: u64 = 4096;
static WAKE_SEEN: AtomicU64 = AtomicU64::new(0);
static WAKE_SPAN_START_US: AtomicU64 = AtomicU64::new(u64::MAX);
static WAKE_SPAN_END_US: AtomicU64 = AtomicU64::new(0);
/// Below this busy share of the decode span, a slow start means the governor rather than a queue.
const WAKE_ALERT_MAX_OCCUPANCY: f64 = 0.5;

/// The share of the decode span the GPU spent executing the buffers the detector reads.
fn wake_occupancy() -> Option<f64> {
    let (a, b) = (WAKE_SPAN_START_US.load(Ordering::Relaxed), WAKE_SPAN_END_US.load(Ordering::Relaxed));
    (b > a && a != u64::MAX).then(|| WAKE_GPU_BUSY_US.load(Ordering::Relaxed) as f64 / (b - a) as f64)
}

#[inline]
fn wake_note(kernel_start: f64, gpu_start: f64, gpu_end: f64) {
    if WAKE_ARMED.load(Ordering::Relaxed) == 0 || kernel_start <= 0.0 || gpu_start < kernel_start {
        return;
    }
    if WAKE_SEEN.fetch_add(1, Ordering::Relaxed) < WAKE_WARMUP {
        return;
    }
    let us = ((gpu_start - kernel_start) * 1e6) as u64;
    let n = WAKE_N.fetch_add(1, Ordering::Relaxed) + 1;
    if gpu_end > gpu_start {
        WAKE_GPU_BUSY_US.fetch_add(((gpu_end - gpu_start) * 1e6) as u64, Ordering::Relaxed);
        // The span by min and max, so out-of-order completions are harmless.
        WAKE_SPAN_START_US.fetch_min((gpu_start * 1e6) as u64, Ordering::Relaxed);
        WAKE_SPAN_END_US.fetch_max((gpu_end * 1e6) as u64, Ordering::Relaxed);
    }
    let b = (usize::BITS - 1 - us.max(1).leading_zeros().min(usize::BITS - 1)) as usize;
    WAKE_HIST[b.min(WAKE_BUCKETS - 1)].fetch_add(1, Ordering::Relaxed);
    // Checked on a power-of-two cadence so the scan is amortised to nothing, and never before 4096 samples.
    if n % 4096 == 0 {
        wake_check_alert();
    }
}

fn wake_quantile_bucket(n: u64, q: f64) -> usize {
    let target = ((n as f64) * (1.0 - q)) as u64;
    let mut above = 0u64;
    for b in (0..WAKE_BUCKETS).rev() {
        above += WAKE_HIST[b].load(Ordering::Relaxed);
        if above > target {
            return b;
        }
    }
    0
}

fn wake_check_alert() {
    let n = WAKE_N.load(Ordering::Relaxed);
    if wake_quantile_bucket(n, 0.50) < WAKE_ALERT_BUCKET {
        return;
    }
    // A saturated GPU starts late because it is busy, not asleep: only an idle, slow one is the pathology.
    if wake_occupancy().is_none_or(|o| o >= WAKE_ALERT_MAX_OCCUPANCY) {
        return;
    }
    if WAKE_WARNED.swap(1, Ordering::Relaxed) != 0 {
        return;
    }
    tracing::info!(target: "oscar::heartbeat",
        "decode waiting to start work - median command-buffer start latency over {} ms across {n} buffers, about \
         1.5 ms when GPU schedules normally; throughput lost to scheduling delay, text unaffected",
        1 << (WAKE_ALERT_BUCKET - 10),
    );
}

/// The run log's work-start line; `None` if decode never ran.
pub fn wake_summary() -> Option<String> {
    let n = WAKE_N.load(Ordering::Relaxed);
    if n == 0 {
        return None;
    }
    let ms = |b: usize| (1u64 << b) as f64 / 1000.0;
    Some(format!(
        "cb_start_ms p50>={:.2} p99>={:.2} n={n} gpu_busy_s={:.1} occupancy={:.2}",
        ms(wake_quantile_bucket(n, 0.50)),
        ms(wake_quantile_bucket(n, 0.99)),
        WAKE_GPU_BUSY_US.load(Ordering::Relaxed) as f64 / 1e6,
        wake_occupancy().unwrap_or(0.0),
    ))
}

/// Checked for its error alone: every buffer no instrument watches carries it, so no error goes unseen.
const CB_TOKEN_QUIET: u64 = 3;
/// The eval's observer's `arm_completion` tokens start here; lower ones are the runtime's: a `CB_ROLE_*`, or quiet.
pub const CB_TOKEN_EVAL: u64 = 8;
/// A token's top bit: the wake detector reads this buffer, an engine buffer committed while it was armed.
const WAKE_WATCHED: u64 = 1 << 63;

/// The runtime's callbacks, then the eval's observer's.
struct CbProbe {
    eval: Option<Arc<dyn CommandBufferObserver>>,
}

thread_local! {
    /// The kernels encoded on this thread since its last commit.
    static OPEN_KERNELS: std::cell::RefCell<Vec<Arc<str>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Per role, the last committed buffers' kernels, oldest first, a completion dropping the oldest: at a failure the
/// ring names what the GPU held, the oldest entry the buffer that did not finish; deeper than any encode-ahead.
type Ring = std::collections::VecDeque<(u64, Vec<Arc<str>>)>;
static IN_FLIGHT_KERNELS: std::sync::Mutex<[Ring; 3]> = std::sync::Mutex::new([Ring::new(), Ring::new(), Ring::new()]);
static COMMITS: AtomicU64 = AtomicU64::new(0);
/// The committing thread's role, in the token's bits under the watch bit; the eval's tokens stay below.
const ROLE_SHIFT: u32 = 61;

/// Each role's buffers, each buffer's kernels in order, a run of one kernel as `name x n`.
fn in_flight_kernels() -> String {
    let rings = IN_FLIGHT_KERNELS.lock().unwrap_or_else(|e| e.into_inner());
    let mut out = String::new();
    for (role, ring) in ["other", "engine", "drafter"].iter().zip(rings.iter()) {
        if ring.is_empty() {
            continue;
        }
        out.push_str(&format!(" {role}:"));
        for (seq, names) in ring.iter() {
            out.push_str(&format!(" #{seq}:"));
            let mut i = 0;
            while i < names.len() {
                let n = names[i..].iter().take_while(|x| **x == names[i]).count();
                out.push_str(&if n > 1 { format!(" {} x {n}", names[i]) } else { format!(" {}", names[i]) });
                i += n;
            }
            out.push(';');
        }
    }
    out
}

impl CommandBufferObserver for CbProbe {
    fn note_encoder(&self) {
        if let Some(e) = &self.eval {
            e.note_encoder();
        }
    }

    fn note_pipeline(&self, name: &Arc<str>) {
        OPEN_KERNELS.with(|k| k.borrow_mut().push(Arc::clone(name)));
        if let Some(e) = &self.eval {
            e.note_pipeline(name);
        }
    }

    fn note_resource(&self, ptr: usize) {
        if let Some(e) = &self.eval {
            e.note_resource(ptr);
        }
    }

    fn arm_completion(&self) -> Option<u64> {
        let role = cb_role();
        let names = OPEN_KERNELS.with(|k| std::mem::take(&mut *k.borrow_mut()));
        let mut rings = IN_FLIGHT_KERNELS.lock().unwrap_or_else(|e| e.into_inner());
        let ring = &mut rings[role as usize % 3];
        ring.push_back((COMMITS.fetch_add(1, Ordering::Relaxed), names));
        if ring.len() > 64 {
            ring.pop_front();
        }
        drop(rings);
        let watched = WAKE_ARMED.load(Ordering::Relaxed) != 0 && role == CB_ROLE_ENGINE;
        // The eval side's token first, as it carries what its handler needs; the watch and the role ride on top.
        let token = self.eval.as_ref().and_then(|e| e.arm_completion()).unwrap_or(CB_TOKEN_QUIET);
        let token = token | ((role as u64) << ROLE_SHIFT);
        Some(if watched { token | WAKE_WATCHED } else { token })
    }

    fn note_completion(&self, mut c: Completion<'_>) {
        // The first errored buffer is fatal: every later buffer on its queue fails, and the engine would park on a fence
        // nothing signals.
        if let Some(msg) = c.error {
            tracing::error!(target: "oscar::kernels",
                "command buffer failed: {msg} (GPUStart={:.3} GPUEnd={:.3}); aborting",
                c.gpu_start, c.gpu_end,
            );
            // A buffer that never started is what the GPU shows past its machine-wide in-use limit: the reading here,
            // on the machine where it happens.
            let gb = |b: u64| b as f64 / crate::machine::GB;
            if let Ok(m) = crate::machine::MemorySnapshot::read() {
                tracing::error!(target: "oscar::kernels",
                    "GPU memory in use {:.1} GB of {:.1} GB limit, {:.1} GB allocated ({:.1} GB by this process); \
                     over limit no buffer starts - close GPU-heavy applications or lower --mem; machine: {}",
                    gb(crate::machine::gpu_inuse().0), gb(m.gpu_ceiling), gb(m.gpu_alloc), gb(m.own_gpu), m.ledger(),
                );
            }
            // What the buffer waited for: the resolver's routing event, and the expert reads behind it.
            if let Some((asked, answered)) = crate::engine::event_lane::routing_progress() {
                let c = &crate::pool::counters::COUNTERS;
                let n = |a: &std::sync::atomic::AtomicU64| a.load(Ordering::Relaxed);
                let ([present, free, pinned, reading], _) =
                    crate::engine::offload_dispatcher().map_or(([0; 4], 0), |pd| pd.pool().slot_occupancy());
                tracing::error!(target: "oscar::kernels",
                    "routing event asked {asked}, answered {answered}; expert reads: {} pending, {} demand landed, {} \
                     misses, resolver waited {} ms on reads; expert slots: {present} present, {free} free, {pinned} \
                     pinned, {reading} reading",
                    oscar_offload::io::PENDING.load(Ordering::Relaxed),
                    n(&c.completed_demand),
                    n(&c.misses),
                    oscar_offload::io::raw_to_ns(n(&c.barrier_wait_raw)) / 1_000_000,
                );
            }
            tracing::error!(target: "oscar::kernels",
                "command buffers in flight, oldest first, each its kernels in order:{}",
                in_flight_kernels()
            );
            let given_up = oscar_offload::segments::release_all();
            tracing::error!(target: "oscar::kernels", "memory released before stopping - {given_up}");
            std::process::abort();
        }
        if c.token & WAKE_WATCHED != 0 {
            wake_note(c.kernel_start, c.gpu_start, c.gpu_end);
            c.token &= !WAKE_WATCHED;
        }
        let role = ((c.token >> ROLE_SHIFT) & 3) as usize;
        c.token &= !(3 << ROLE_SHIFT);
        IN_FLIGHT_KERNELS.lock().unwrap_or_else(|e| e.into_inner())[role % 3].pop_front();
        if c.token == CB_TOKEN_QUIET {
            return;
        }
        if let Some(e) = &self.eval {
            e.note_completion(c);
        }
    }
}

/// Once, before the model makes the device; `eval` the eval's observer, if it watches.
pub fn install(eval: Option<Arc<dyn CommandBufferObserver>>) {
    set_command_buffer_observer(Arc::new(CbProbe { eval }));
}
