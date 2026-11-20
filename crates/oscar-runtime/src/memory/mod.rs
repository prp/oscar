//! The adaptive memory: the launch's first reading and squeeze, the sensor and watchdog, the controller and crossing.

pub(crate) mod controller;
pub mod crossing;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::machine::GB;
use crate::machine::{
    claimant_charge, claimant_step, machine_room_bytes, under_pressure, Bounds, MemorySnapshot, CHANGE_QUANTUM,
};
use oscar_offload::machine;

/// In samples: a median of eight hides a transient of up to three and shows a claimant after four.
const SETTLING_WINDOW: usize = 8;

/// One settled window read before oscar holds memory, the driver's host side booked; an unreadable machine is refused.
pub fn first_stage_bounds(budget: u64) -> anyhow::Result<Bounds> {
    static READING: OnceLock<Result<Bounds, &'static str>> = OnceLock::new();
    let reading = *READING.get_or_init(|| {
        let bounds = settled_first_reading()?;
        let keep = crate::config::watchdog_floor_bytes() + claimant_charge();
        let room = machine_room_bytes(bounds.physical, bounds.foreign, budget, keep);
        if room < CHANGE_QUANTUM as i64 && squeeze(budget, keep) {
            return settled_first_reading();
        }
        Ok(bounds)
    });
    reading
        .map_err(|what| anyhow::anyhow!("refused - {what} unreadable: no plan bounded, no watchdog armed without it"))
}

fn settled_first_reading() -> Result<Bounds, &'static str> {
    let mut ring = [controller::Foreign::default(); SETTLING_WINDOW];
    let mut last = MemorySnapshot::default();
    for (i, slot) in ring.iter_mut().enumerate() {
        if i > 0 {
            std::thread::sleep(CADENCE);
        }
        last = MemorySnapshot::read()?;
        *slot = controller::Foreign::of(&last);
    }
    Ok(with_the_drivers_host_side(controller::settled_bounds(ring.into_iter(), &last)))
}

/// Booked on the machine wire bound, which counts the GPU driver's pages (wired while the GPU works), not the task's.
fn with_the_drivers_host_side(b: Bounds) -> Bounds {
    Bounds { foreign_wired: b.foreign_wired.saturating_add(crate::plan::host_line::GPU_DRIVER_HOST_BYTES), ..b }
}

/// A locked balloon holds free at half the kernel's free target so it compresses others' idle pages before the load.
fn squeeze(budget: u64, keep: u64) -> bool {
    let (Ok(start), Some(target)) = (MemorySnapshot::read(), machine::pages(c"kern.vm_page_free_target")) else {
        return false;
    };
    // Only others' pageable memory compresses; their wired pages and what is compressed already stay as they are.
    let compressible = start.foreign().saturating_sub(start.foreign_wired()).saturating_sub(start.compressor);
    if compressible < CHANGE_QUANTUM {
        return false;
    }
    let gb = |b: u64| b as f64 / GB;
    tracing::info!(target: "oscar::planner",
        "other processes hold {:.1} GB, too much beside {:.1} GB budget: compressing their idle memory...",
        gb(start.foreign()),
        gb(budget),
    );
    let (t0, page) = (Instant::now(), oscar_offload::segments::PAGE as u64);
    let (mut balloon, mut freed, mut others) = (Balloon::default(), t0, start.foreign());
    let stopped = loop {
        if t0.elapsed() >= SQUEEZE_TIME_LIMIT {
            break ": time limit";
        }
        let free = machine::live_free().unwrap_or(0);
        let take = if free > target + 2 * claimant_step() {
            claimant_step()
        } else {
            free.saturating_sub(target / 2) / page * page
        };
        if take > 0 {
            if !balloon.grow(take as usize) {
                break ": system refused to lock more";
            }
            freed = Instant::now();
            continue;
        }
        if freed.elapsed() >= SQUEEZE_IDLE_LIMIT {
            break ": nothing more idle";
        }
        std::thread::sleep(machine::TICK);
        let Ok(s) = MemorySnapshot::read() else { break ": machine unreadable" };
        // Count a reading once its wired pages show the balloon: the lists can predate the balloon's last pages.
        if s.wired + CHANGE_QUANTUM >= start.wired + balloon.held {
            others = s.foreign();
            if machine_room_bytes(s.physical, others, budget, keep) >= CHANGE_QUANTUM as i64 {
                break "";
            }
        }
    };
    drop(balloon);
    // The kernel's wired count carries the balloon's pages for a moment past the unmap, which the settled reading
    // after this would take for others' memory: wait for it to fall back, bounded.
    let released = Instant::now();
    while released.elapsed() < RELEASE_LIMIT {
        match MemorySnapshot::read() {
            Ok(s) if s.wired > start.wired + CHANGE_QUANTUM => std::thread::sleep(machine::TICK),
            Ok(s) => {
                others = s.foreign();
                break;
            }
            Err(_) => break,
        }
    }
    tracing::info!(target: "oscar::planner",
        "other processes hold {:.1} GB after {} compressing{stopped}",
        gb(others),
        mistralrs_core::live::duration_text(t0.elapsed().as_secs_f64()),
    );
    true
}

const SQUEEZE_TIME_LIMIT: Duration = Duration::from_secs(30);
const SQUEEZE_IDLE_LIMIT: Duration = Duration::from_secs(2);
/// The balloon's pages leave the wired count within this, else the reading proceeds on what the machine shows.
const RELEASE_LIMIT: Duration = Duration::from_secs(10);

/// Memory this process maps and locks for the launch's squeeze, given back whole when it drops.
#[derive(Default)]
struct Balloon {
    chunks: Vec<(*mut libc::c_void, usize)>,
    held: u64,
}

impl Balloon {
    /// `bytes` more, mapped and locked; false where the lock is refused.
    fn grow(&mut self, bytes: usize) -> bool {
        // SAFETY: a fresh anonymous mapping, this balloon's alone until its drop unlocks and unmaps it.
        unsafe {
            let at = libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_PRIVATE,
                -1,
                0,
            );
            if at == libc::MAP_FAILED {
                return false;
            }
            if libc::mlock(at, bytes) != 0 {
                libc::munmap(at, bytes);
                return false;
            }
            self.chunks.push((at, bytes));
        }
        self.held += bytes as u64;
        true
    }
}

impl Drop for Balloon {
    fn drop(&mut self) {
        for &(at, bytes) in &self.chunks {
            // SAFETY: each mapping was made and locked by `grow`, once.
            unsafe {
                libc::munlock(at, bytes);
                libc::munmap(at, bytes);
            }
        }
    }
}

/// The sensor's cadence and the watchdog's strike (three to the abort), short as a load wires GB/s.
const CADENCE: Duration = Duration::from_millis(250);
const FAST_CADENCE: Duration = machine::TICK;

static TRANSITIONS: AtomicU64 = AtomicU64::new(0);
/// The sensor thread, for the interrupt handler to wake.
static SENSOR: OnceLock<std::thread::Thread> = OnceLock::new();
static LATEST: Mutex<Option<MemorySnapshot>> = Mutex::new(None);

pub fn latest_snapshot() -> Option<MemorySnapshot> {
    *LATEST.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn pressure_transitions() -> u64 {
    TRANSITIONS.load(Ordering::Relaxed)
}

/// Through [`stop`]: alone a panic ends one thread, and the mistral.rs fork reboots the engine over pools given up.
pub fn end_the_process_on_a_panic() {
    std::panic::set_hook(Box::new(|info| {
        // A thread tripping over pools a give-up is releasing is part of that ending: it waits, silent, for the abort.
        if oscar_offload::segments::giving_up() {
            loop {
                std::thread::park();
            }
        }
        // Through the console's writer: the default hook writes stderr under a lock an unread terminal holds for ever.
        let trace = std::backtrace::Backtrace::capture();
        let trace = match trace.status() {
            std::backtrace::BacktraceStatus::Captured => format!("stack backtrace:\n{trace}\n"),
            _ => String::new(),
        };
        let thread = std::thread::current();
        crate::cli::console::say(
            format!("thread '{}' {info}\n{trace}", thread.name().unwrap_or("<unnamed>")).into_bytes(),
        );
        crate::cli::console::drain();
        stop("sensor", format_args!("panic ends process"));
    }));
}

/// A refusal at the load (the machine cannot host what was asked), said once and the process ended with status 1:
/// no defect, so no panic and no `ERROR`.
pub fn refuse(why: std::fmt::Arguments) -> ! {
    tracing::info!(target: "oscar::planner", "{why}");
    let _ = oscar_offload::segments::release_all();
    crate::cli::console::drain();
    std::process::exit(1)
}

/// The one way oscar stops itself; the pools go before the abort, as dying returns locked memory too late.
fn stop(head: &'static str, why: std::fmt::Arguments) -> ! {
    crate::engine::event_lane::answer_waiting_launches();
    crate::cli::console::urgent(head, true, why);
    let given_up = oscar_offload::segments::release_all();
    crate::cli::console::urgent(head, false, format_args!("memory released before stopping - {given_up}"));
    std::process::abort()
}

#[derive(Default)]
struct WatchdogFuse {
    strikes: u32,
    below_since: Option<Instant>,
    last: Option<u64>,
}

impl WatchdogFuse {
    /// The headroom to stop at and whether it fell too fast to wait; a tick that took at least what is left kills.
    fn tick(&mut self, free: u64, at: Instant, floor: u64, floor_gb: f64) -> Option<(u64, bool)> {
        let fell = self.last.replace(free).map_or(0, |was| was.saturating_sub(free));
        if free >= floor {
            (self.strikes, self.below_since) = (0, None);
            return None;
        }
        if fell >= free {
            return Some((free, true));
        }
        let since = *self.below_since.get_or_insert(at);
        let mut due = 1 + (at.duration_since(since).as_millis() / CADENCE.as_millis()) as u32;
        // Under a quantum a strike falls each tick: the machine survived the stops above a quantum, not those below.
        if free < CHANGE_QUANTUM {
            due = due.max(self.strikes + 1);
        }
        if due <= self.strikes {
            return None;
        }
        self.strikes = due;
        if due >= 3 {
            return Some((free, false));
        }
        tracing::warn!(target: "oscar::watchdog",
            "Memory is running low: headroom {:.2} GB under the {floor_gb:.1} GB watchdog floor (strike {due} of 3){}",
            free as f64 / GB,
            if due == 2 { " - oscar stops at next" } else { "" },
        );
        None
    }
}

/// A pair (asked, answered) unanswered past `limit` fires once, so nothing waits on oscar; idle never fires.
struct RoutingWatch {
    seen: (u64, u64),
    since: Instant,
}

impl RoutingWatch {
    fn observe(&mut self, progress: (u64, u64), at: Instant, limit: Duration) -> bool {
        let (asked, answered) = progress;
        let fired = progress == self.seen && asked > answered && at.duration_since(self.since) > limit;
        // Fired, the launches are answered up to `asked`: the state the next sample compares against.
        if progress != self.seen || fired {
            (self.seen, self.since) = (if fired { (asked, asked) } else { progress }, at);
        }
        fired
    }
}

/// One thread: the floor every `FAST_CADENCE`, as a floor can be crossed within one `CADENCE`, and the snapshot every
/// `CADENCE`, fast under pressure and at once on a pressure transition.
pub fn start_sensors(plan_ceiling_bytes: u64) {
    if SENSOR.get().is_some() {
        return;
    }
    // The plan's ceiling, armed with the floor so they cannot drift apart; the load's tail re-arms it.
    machine::PLAN.arm(plan_ceiling_bytes);
    let (floor, floor_gb) = (crate::config::watchdog_floor_bytes(), crate::config::params().misc.mem_watchdog_gb);
    // The pools refuse a segment crossing this same floor: refusal and kill are one number, the refusal first.
    machine::set_floor(floor);
    let handle = std::thread::Builder::new()
        .name("oscar-memory".into())
        .spawn(move || {
            use crate::engine::event_lane::{routing_progress, ROUTING_STALL};
            let mut routing = RoutingWatch { seen: (0, 0), since: Instant::now() };
            let (mut fuse, mut cadence, mut sampled, mut seen) =
                (WatchdogFuse::default(), CADENCE, Instant::now(), pressure_transitions());
            loop {
                std::thread::park_timeout(FAST_CADENCE);
                let free = machine::reclaimable().filter(|_| floor > 0);
                if let Some((free, falling)) = free.and_then(|free| fuse.tick(free, Instant::now(), floor, floor_gb)) {
                    let how = if falling { "fell too fast to wait at" } else { "stayed below" };
                    if let Ok(s) = MemorySnapshot::read() {
                        tracing::error!(target: "oscar::watchdog", "machine at stop: {}", s.ledger());
                        crate::cli::console::drain();
                    }
                    stop(
                        "watchdog",
                        format_args!(
                            "Oscar is stopping to protect the system: memory headroom {how} the {floor_gb:.1} GB watchdog floor \
                             ({:.2} GB)",
                            free as f64 / GB
                        ),
                    );
                }
                if sampled.elapsed() < cadence && pressure_transitions() == seen {
                    continue;
                }
                (sampled, seen) = (Instant::now(), pressure_transitions());
                let Ok(snap) = MemorySnapshot::read() else { continue };
                cadence = if under_pressure(&snap, floor) { FAST_CADENCE } else { CADENCE };
                *LATEST.lock().unwrap_or_else(|e| e.into_inner()) = Some(snap);
                if let Some(peak) = controller::observe(&snap) {
                    crossing::request(peak);
                }
                crate::engine::eval::hooks().memory_sample(&snap);
                // A queue nothing answers is dead: after a buffer the GPU driver refused, every later one on it fails.
                if routing_progress().is_some_and(|p| routing.observe(p, snap.at, ROUTING_STALL)) {
                    tracing::error!(target: "oscar::sensor", "machine at stall: {}{}", snap.ledger(), crate::cli::console::stamp());
                    crate::cli::console::drain();
                    stop(
                        "sensor",
                        format_args!(
                            "nothing has answered the GPU for {:.0} s - stopping oscar (a queue the driver has \
                             refused never recovers){}",
                            ROUTING_STALL.as_secs_f64(),
                            crate::cli::console::stamp()
                        ),
                    );
                }
            }
        })
        .expect("memory sensor thread spawns");
    let _ = SENSOR.set(handle.thread().clone());
    install_pressure_interrupt();
}

/// A memory-pressure source wakes the sensor thread on level transitions, so every snapshot has the one producer.
fn install_pressure_interrupt() {
    use std::ffi::c_void;
    type DispatchObject = *mut c_void;
    extern "C" {
        static _dispatch_source_type_memorypressure: c_void;
        fn dispatch_source_create(
            ty: *const c_void, handle: usize, mask: usize, queue: DispatchObject,
        ) -> DispatchObject;
        fn dispatch_source_set_event_handler_f(source: DispatchObject, handler: extern "C" fn(*mut c_void));
        fn dispatch_resume(obj: DispatchObject);
        fn dispatch_get_global_queue(identifier: isize, flags: usize) -> DispatchObject;
    }
    const DISPATCH_MEMORYPRESSURE_ALL: usize = 0x1 | 0x2 | 0x4;
    extern "C" fn pressure_event(_ctx: *mut c_void) {
        TRANSITIONS.fetch_add(1, Ordering::Relaxed);
        if let Some(t) = SENSOR.get() {
            t.unpark();
        }
    }
    // SAFETY: plain libdispatch C API use; the source is leaked for the process lifetime.
    unsafe {
        let queue = dispatch_get_global_queue(0, 0);
        if queue.is_null() {
            return;
        }
        let src = dispatch_source_create(&_dispatch_source_type_memorypressure, 0, DISPATCH_MEMORYPRESSURE_ALL, queue);
        if !src.is_null() {
            dispatch_source_set_event_handler_f(src, pressure_event);
            dispatch_resume(src);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::machine::Bound;

    const G: u64 = 1 << 30;

    /// A strike a CADENCE at any tick rate, one a tick under a quantum of headroom; a fall past what is left kills. A
    /// routing pair standing unanswered past the limit fires once; progress and idle never do. A step under half the
    /// window settles to its base; the booked host side is room again once the driver holds it.
    #[test]
    fn the_backstops_fire_on_a_stall_alone_and_the_first_stage_settles_the_window() {
        let gib = |g: f64| (g * G as f64) as u64;
        let (floor, dip) = (4 * G, 2 * G);
        // (floor, the headroom at each fast tick, the strikes, whether the last tick fell too fast to wait): five fast
        // ticks inside one CADENCE are one strike, the second a CADENCE later, back above the floor none; from 4.12 GB
        // a tick at 1.36 has lost more than it holds, a 0.10 GB fall onto 1.30 has ticks left; under a quantum, strike.
        let cases: [(u64, Vec<u64>, u32, Option<bool>); 7] = [
            (floor, vec![dip; 5], 1, None),
            (floor, vec![dip; 10], 2, None),
            (floor, [vec![dip; 10], vec![floor]].concat(), 0, None),
            (gib(1.5), vec![gib(4.12), gib(1.36)], 0, Some(true)),
            (gib(1.5), vec![gib(1.40), gib(1.30)], 1, None),
            (floor, vec![CHANGE_QUANTUM / 2; 2], 2, None),
            (floor, vec![dip; 2], 1, None),
        ];
        for (floor, headroom, strikes, stop) in cases {
            let (t0, mut fuse, mut last) = (Instant::now(), WatchdogFuse::default(), None);
            for (k, &free) in headroom.iter().enumerate() {
                last = fuse.tick(free, t0 + FAST_CADENCE * k as u32, floor, floor as f64 / G as f64);
            }
            assert_eq!((fuse.strikes, last.map(|(_, falling)| falling)), (strikes, stop), "{headroom:?}");
            assert_eq!(fuse.below_since.is_none(), strikes == 0, "{headroom:?}");
        }
        let limit = Duration::from_secs(120);
        let t0 = Instant::now();
        let mut w = RoutingWatch { seen: (0, 0), since: t0 };
        let late = t0 + limit + Duration::from_secs(1);
        // Progress, however far apart the two run; idle: equal and standing for ever is the engine with nothing to do.
        assert!(!w.observe((7, 3), t0, limit));
        assert!(!w.observe((9, 8), late, limit), "a pair that moves is progress");
        assert!(!w.observe((9, 9), late, limit) && !w.observe((9, 9), late + limit * 2, limit));
        // A stall: unanswered and standing past the limit, once.
        assert!(!w.observe((12, 9), late, limit), "the stall's clock starts here");
        assert!(!w.observe((12, 9), late + limit, limit), "not yet past the limit");
        assert!(w.observe((12, 9), late + limit * 2, limit), "nothing answered for the limit");
        assert!(!w.observe((12, 9), late + limit * 4, limit), "answered to 12 already: the reading fires once");
        // A later layer stalls in turn.
        assert!(!w.observe((15, 12), late + limit * 4, limit) && w.observe((15, 12), late + limit * 6, limit));
        // The first stage.
        use controller::{settled_bounds, Foreign};
        let base = Foreign { non_droppable: 6 * G, gpu: G, wired: 3 * G, ..Default::default() };
        let (level, margin) = (crate::plan::host_line::GPU_DRIVER_HOST_BYTES, G);
        let mut ring = [base; SETTLING_WINDOW];
        ring[..SETTLING_WINDOW / 2 - 1].fill(Foreign { wired: 3 * G + G / 2, ..base });
        let limits = MemorySnapshot {
            physical: 64 * G,
            gpu_ceiling: 54 * G,
            machine_wire_limit: 56 * G,
            task_wire_limit: 56 * G,
            ..Default::default()
        };
        let quiet = settled_bounds(ring.into_iter(), &limits);
        assert_eq!((quiet.foreign, quiet.foreign_wired), (6 * G, 3 * G));
        let first = with_the_drivers_host_side(quiet);
        let locked = 56 * G - (3 * G + level) - margin;
        assert_eq!(first.task_wire_limit, quiet.task_wire_limit);
        assert_eq!(first.room(40 * G, 40 * G, locked, margin, 0), (0, Bound::WireMachine));
        let held = Bounds { foreign_wired: 3 * G + level / 2, ..quiet };
        assert_eq!(held.room(40 * G, 40 * G, locked, margin, 0).0, (level / 2) as i64);
    }
}
