//! What the pool counts about itself, process-wide, and the decode window the lines difference; nothing here may
//! influence a forward.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

use oscar_offload::io::{raw_to_ns, Reads, MAX_DEVICES};

/// Declares the counters and [`Count`], every one of them at one instant.
macro_rules! counters {
    ($($(#[$doc:meta])* $name:ident),* $(,)?) => {
        pub struct Counters {
            $($(#[$doc])* pub $name: AtomicU64,)*
            /// Decode's split of the misses counts only while armed: from a decode token to the next prefill's waves.
            pub decode_armed: AtomicBool,
        }

        #[derive(Clone, Copy, Debug, Default)]
        pub struct Count {
            $(pub $name: u64,)*
        }

        pub static COUNTERS: Counters = Counters { $($name: AtomicU64::new(0),)* decode_armed: AtomicBool::new(false) };

        impl Counters {
            pub fn now(&self) -> Count {
                Count { $($name: self.$name.load(Relaxed),)* }
            }
        }
    };
}

counters!(
    /// Each key of a demand resolve once: a hit (no read waited on), a miss (reserved), a wait on a read in flight.
    hits,
    misses,
    waits,
    /// Demand accesses a staged fill served (a subset of `hits`).
    prefetch_saves,
    evicted,
    /// Staged fills evicted unused, and of them those evicted by their promised tick: churn, not the predictor.
    wasted_prefetch,
    wasted_churn,
    /// Cold misses: a key's first-ever read, or a refetch after its eviction; and decode's.
    first_touch,
    refetch,
    first_touch_decode,
    refetch_decode,
    /// The waits for reads in flight that blocked (mach ticks): the keys missing at entry, those missing two or more,
    /// the keys asked, the time, and the spread from the first arrival to the last.
    barrier_blocked,
    barrier_missing,
    barrier_multi,
    barrier_keys,
    barrier_wait_raw,
    barrier_spread_raw,
    /// The readers': fills published by class, failed reads, and demand's queue wait, GPU-fence wait and read time.
    completed_demand,
    completed_prefetch,
    read_errors,
    demand_queue_wait_ns,
    demand_gpu_safe_wait_ns,
    demand_read_ns,
);

impl Count {
    pub fn accesses(&self) -> u64 {
        self.hits + self.misses + self.waits
    }

    /// The accesses that blocked: cold, or staged but not landed.
    pub fn blocked(&self) -> u64 {
        self.misses + self.waits
    }
}

/// At prefill→decode, so the readers' counters but their errors read decode only.
pub fn reset_reads() {
    let c = &COUNTERS;
    for counter in [
        &c.completed_demand,
        &c.completed_prefetch,
        &c.demand_queue_wait_ns,
        &c.demand_gpu_safe_wait_ns,
        &c.demand_read_ns,
    ] {
        counter.store(0, Relaxed);
    }
}

/// Every counter at an instant, taken before each decode token's IO: consecutive ones differ by its cost.
#[derive(Clone, Copy, Debug, Default)]
pub struct TokenSample {
    pub window: u32,
    pub token: u32,
    /// Since the window's first sample.
    pub wall_ns: u64,
    pub reads: Reads,
    pub pool: Count,
}

impl TokenSample {
    pub fn now() -> Self {
        Self { reads: oscar_offload::io::reads(), pool: COUNTERS.now(), ..Default::default() }
    }
}

/// The difference of two samples, the rates obeying `tps = duty × GB/s while busy ÷ GB/token`.
#[derive(Clone, Copy, Debug)]
pub struct DecodeWindow {
    first: TokenSample,
    last: TokenSample,
}

impl DecodeWindow {
    /// `None` when it holds no token or no time.
    pub fn between(first: &TokenSample, last: &TokenSample) -> Option<Self> {
        let w = Self { first: *first, last: *last };
        (w.tokens() > 0 && w.secs() > 0.0).then_some(w)
    }
    pub fn delta(&self, pick: impl Fn(&TokenSample) -> u64) -> u64 {
        pick(&self.last).saturating_sub(pick(&self.first))
    }
    pub fn tokens(&self) -> u64 {
        self.last.token.saturating_sub(self.first.token) as u64
    }
    pub fn secs(&self) -> f64 {
        self.delta(|s| s.wall_ns) as f64 / 1e9
    }
    pub fn tps(&self) -> f64 {
        self.tokens() as f64 / self.secs().max(1e-9)
    }
    pub fn bytes(&self) -> u64 {
        self.delta(|s| s.reads.bytes)
    }
    pub fn gb_per_token(&self) -> f64 {
        self.bytes() as f64 / 1e9 / self.tokens().max(1) as f64
    }
    pub fn gb_per_s(&self) -> f64 {
        self.bytes() as f64 / 1e9 / self.secs().max(1e-9)
    }
    pub fn busy_ns(&self) -> u64 {
        raw_to_ns(self.delta(|s| s.reads.busy_raw))
    }
    pub fn duty(&self) -> f64 {
        self.busy_ns() as f64 / (self.secs().max(1e-9) * 1e9)
    }
    /// Bytes per nanosecond is GB/s.
    pub fn gb_per_s_while_busy(&self) -> f64 {
        self.bytes() as f64 / self.busy_ns().max(1) as f64
    }
    /// Per device that moved bytes: (device, bytes, busy ns).
    pub fn devices(&self) -> Vec<(usize, u64, u64)> {
        (0..MAX_DEVICES)
            .map(|d| (d, self.delta(|s| s.reads.bytes_dev[d]), raw_to_ns(self.delta(|s| s.reads.busy_raw_dev[d]))))
            .filter(|(_, b, _)| *b > 0)
            .collect()
    }
    /// Saturates while every reader is busy however idle the device; [`Self::duty`] is the bandwidth figure.
    pub fn occupancy(&self, io_threads: usize) -> f64 {
        self.delta(|s| s.reads.ns) as f64 / (self.secs().max(1e-9) * 1e9 * io_threads.max(1) as f64)
    }
    pub fn per_worker_gb_s(&self) -> f64 {
        self.bytes() as f64 / self.delta(|s| s.reads.ns).max(1) as f64
    }
    pub fn barrier_wait_ns(&self) -> u64 {
        raw_to_ns(self.delta(|s| s.pool.barrier_wait_raw))
    }
    pub fn barrier_spread_ns(&self) -> u64 {
        raw_to_ns(self.delta(|s| s.pool.barrier_spread_raw))
    }
    /// The serve lane's per-request line; `None` when the sampler did not run; the histogram's broken buckets as zeros.
    pub fn depth_line(&self) -> Option<String> {
        let h: [u64; 9] =
            (self.delta(|s| s.reads.duty[0]) > 0).then(|| std::array::from_fn(|i| self.delta(|s| s.reads.depth[i])))?;
        Some(format!(
            "decode-depth | samples {} busy {} | d 0:{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} 8+:{} \
             | miss {}/{} | barriers {} missing {} keys {} multi {} \
             | bwait {:.0}ms spread {:.0}ms",
            self.delta(|s| s.reads.duty[0]),
            self.delta(|s| s.reads.duty[1]),
            h[0],
            h[1],
            h[2],
            h[3],
            h[4],
            h[5],
            h[6],
            h[7],
            h[8],
            self.delta(|s| s.pool.blocked()),
            self.delta(|s| s.pool.accesses()),
            self.delta(|s| s.pool.barrier_blocked),
            self.delta(|s| s.pool.barrier_missing),
            self.delta(|s| s.pool.barrier_keys),
            self.delta(|s| s.pool.barrier_multi),
            self.barrier_wait_ns() as f64 / 1e6,
            self.barrier_spread_ns() as f64 / 1e6,
        ))
    }

    /// `(busy, nothing queued, queued unserved)` as fractions; `None` when the diagnostics-gated sampler did not run.
    pub fn idle_split(&self) -> Option<(f64, f64, f64)> {
        let n = self.delta(|s| s.reads.duty[0]);
        let f = |i: usize| self.delta(|s| s.reads.duty[i]) as f64 / n as f64;
        (n > 0).then(|| (f(1), f(2), f(3)))
    }
}

/// The line a lane prints beside its request.
impl std::fmt::Display for DecodeWindow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "decode       | {} tok in {} @ {:.2} tok/s | drive busy {:.1}% at {:.2} GB/s \
             | {:.3} GB/token",
            self.tokens(),
            mistralrs_core::live::duration_text(self.secs()),
            self.tps(),
            100.0 * self.duty(),
            self.gb_per_s_while_busy(),
            self.gb_per_token(),
        )?;
        match self.idle_split() {
            Some((_, empty, backlog)) => {
                write!(f, " | idle {:.1}% nothing queued, {:.1}% queued", 100.0 * empty, 100.0 * backlog,)
            }
            None => Ok(()),
        }
    }
}

/// One sample per decode token for the eval, ungated so a throughput and its explanation come from one run.
#[derive(Default)]
pub struct TokenSeries(Mutex<Series>);

#[derive(Default)]
struct Series {
    window: u32,
    token: u32,
    started: Option<std::time::Instant>,
    first: Option<TokenSample>,
    /// Each window is reported once, or a one-token reply, with no decode forward, would report the previous request's.
    reported: Option<u32>,
    /// Set by a prefill: the load's one-token forwards look exactly like decode tokens.
    armed: bool,
}

impl TokenSeries {
    fn lock(&self) -> std::sync::MutexGuard<'_, Series> {
        self.0.lock().expect("token series")
    }

    /// At the prefill→decode boundary.
    pub fn begin_window(&self) {
        let mut g = self.lock();
        if g.armed {
            g.window += 1;
        }
        g.armed = true;
        g.token = 0;
        g.started = None;
        g.first = None;
    }

    /// Every counter now, stamped with the window, its token and the time since its first sample.
    fn sample(g: &Series, started: std::time::Instant, now: std::time::Instant) -> TokenSample {
        let wall_ns = now.saturating_duration_since(started).as_nanos() as u64;
        TokenSample { window: g.window, token: g.token, wall_ns, ..TokenSample::now() }
    }

    /// `None` outside a window a prefill opened.
    pub fn record(&self) -> Option<TokenSample> {
        let now = std::time::Instant::now();
        let mut g = self.lock();
        if !g.armed {
            return None;
        }
        let started = *g.started.get_or_insert(now);
        let sample = Self::sample(&g, started, now);
        g.first.get_or_insert(sample);
        g.token += 1;
        Some(sample)
    }

    /// From the open window's first sample to a live one, handed out once; `None` before its first decode token.
    pub fn window_now(&self) -> Option<DecodeWindow> {
        let now = std::time::Instant::now();
        let mut g = self.lock();
        if g.reported == Some(g.window) {
            return None;
        }
        let (first, started) = (g.first?, g.started?);
        let window = DecodeWindow::between(&first, &Self::sample(&g, started, now))?;
        g.reported = Some(g.window);
        Some(window)
    }
}
