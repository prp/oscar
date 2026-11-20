//! The plan at a new budget, applied as a diff at a parked boundary, its frees and takes on a background thread.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use mistralrs_core::paged_attention::{self as engine_hooks, KvBlockMemory};

use super::controller::PlanTotals;
use crate::cli::console::stamp;
use crate::config::KvCharge;
use crate::gpu::kv_pool::KvPool;
use crate::machine::GB;
use crate::plan::{self, MemPlan, ModelLayoutStats};

/// The next crossing's target: the plan's two inputs, each as last asked.
#[derive(Default)]
struct Target {
    ask: Option<u64>,
    kv_ask: usize,
    requested_at: Option<Instant>,
    deferred: bool,
}

impl Target {
    fn pending(&self) -> bool {
        self.ask.is_some() || self.kv_ask > 0
    }
}

struct Decision {
    plan: MemPlan,
    at_floor: bool,
    carried: u64,
    solves: u32,
}

struct Crossing {
    stats: &'static ModelLayoutStats,
    /// The drafter's memory with it, the launch's for good.
    launch_charge: KvCharge,
    ceiling_budget: u64,
    launch_blocks: u64,
    /// Held across a park; its `kv_blocks` is the block pool's live count.
    live: Mutex<MemPlan>,
    target: Mutex<Target>,
}

static CROSSING: OnceLock<Crossing> = OnceLock::new();
/// From the park to the mover's last freed segment: the ledger sees a release a second late, which reads as a claimant.
static MOVE_IN_FLIGHT: AtomicBool = AtomicBool::new(false);
static MOVER: OnceLock<std::thread::Thread> = OnceLock::new();

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Reached only past the install: the controller starts after the load, a move after a crossing.
fn crossing() -> &'static Crossing {
    CROSSING.get().expect("the crossing is reached only past its install at the load's tail")
}

/// Solved at the load's tail from the census's record and re-solved by a crossing; `None` before, or with no plan.
pub fn live() -> Option<MemPlan> {
    CROSSING.get().map(|a| lock(&a.live).clone())
}

/// At the load's tail, before any request; the drafter's region is released at once where the plan books nothing.
pub fn install(stats: &'static ModelLayoutStats, plan: &MemPlan, ceiling: u64) {
    if CROSSING.set(Crossing::new(stats, plan, ceiling)).is_err() {
        return;
    }
    let booked = plan.kv_charge.drafter_booked;
    crate::drafter::set_held(booked);
    if let Some(Err(e)) = oscar_offload::pools::pools().filter(|_| !booked).map(|p| p.hold_stream_regions(false)) {
        tracing::error!(target: "oscar::drafter", "drafter's stream region release failed - {e}");
    }
    let crossing = crossing();
    engine_hooks::set_memory_crossing_hook(Box::new(move |dev| crossing.cross(dev)));
}

fn set_target(set: impl FnOnce(&mut Target)) {
    let mut t = lock(&crossing().target);
    set(&mut t);
    t.requested_at = Some(Instant::now());
    drop(t);
    engine_hooks::request_memory_crossing();
}

/// The controller's request, solved at the next token or chunk boundary; the latest ask wins.
pub(super) fn request(peak: u64) {
    set_target(|t| t.ask = Some(peak));
    tracing::info!(target: "oscar::crossing", "requested for {:.2} GB peak{}", peak as f64 / GB, stamp());
}

/// `blocks` more where a plan at the ceiling hosts them, the request waiting for its crossing; the machine is not read.
pub fn request_kv_blocks(blocks: usize) {
    let (Some(crossing), Some(kv)) = (CROSSING.get(), KvPool::installed()) else { return };
    let live = lock(&crossing.live).kv_blocks;
    let mut t = lock(&crossing.target);
    if let Err(why) = admit_ask(live, kv.blocks() as u64, t.kv_ask as u64, blocks as u64) {
        tracing::warn!(target: "oscar::crossing",
            "A request is waiting for memory: {blocks} KV blocks refused at the {:.2} GB ceiling until running requests end, {why}{}",
            crossing.ceiling_budget as f64 / GB,
            stamp()
        );
        return;
    }
    t.kv_ask += blocks;
    t.requested_at = Some(Instant::now());
    tracing::info!(target: "oscar::crossing",
        "requested {} KV blocks, over pool's hold ({blocks} this request); granted as total in force allows{}",
        t.kv_ask,
        stamp()
    );
    drop(t);
    engine_hooks::request_memory_crossing();
}

/// Asked blocks still coming, carried or owed: the scheduler waits rather than counting toward its starvation exit.
pub fn kv_blocks_pending() -> bool {
    CROSSING.get().is_some_and(|a| lock(&a.target).kv_ask > 0) || KvPool::installed().is_some_and(|kv| kv.pending())
}

/// A move stopped by `what`: the plan re-lands at what the pools hold; never a panic or a retry.
pub fn re_land(shape: Option<(usize, usize)>, kv_blocks: Option<u64>, what: &str, why: &str) {
    let crossing = crossing();
    let mut live = lock(&crossing.live);
    let from = live.clone();
    let (carriers, hot_budget) = shape.unwrap_or((from.carrier_layers, from.hot_budget));
    let inputs = plan::PlanInputs::live(from.mem_budget_bytes as f64 / GB, crossing.launch_charge)
        .blocks(kv_blocks.unwrap_or(from.kv_blocks));
    match inputs.shape(carriers, hot_budget).solve(crossing.stats) {
        Ok(plan) => {
            let gb = |b: u64| b as f64 / GB;
            tracing::error!(target: "oscar::crossing",
                "{what} mid-move ({why}); plan re-lands at pools' hold - expert pool {} ({:.2} GB), KV pool {} \
                 blocks ({:.2} GB), {:.2} GB in all, budget {:.2} -> {:.2} GB{}",
                plan.pool_shape(),
                gb(plan.expert_pool_bytes),
                plan.kv_blocks,
                gb(plan.kv_cache_bytes),
                gb(plan.total_allocated_bytes()),
                gb(from.mem_budget_bytes),
                gb(plan.mem_budget_bytes),
                stamp(),
            );
            crossing.land(&mut live, plan, false);
        }
        Err(e) => tracing::error!(target: "oscar::crossing",
            "{what} mid-move ({why}), no plan describes pools' hold ({e}); live plan stands{}",
            stamp()
        ),
    }
}

fn shape(p: &MemPlan) -> (usize, usize) {
    (p.hot_budget, p.carrier_layers)
}

/// The free blocks (uncached, cached) a shed may withdraw: those past the room live requests were `admitted` to.
fn sheddable(allocatable: u64, free: u64, free_uncached: u64, admitted: u64) -> (u64, u64) {
    let spare = allocatable.saturating_sub(admitted).min(free);
    let uncached = free_uncached.min(spare);
    (uncached, spare - uncached)
}

/// Room for `blocks` beside `live` and `asked` in the `index` ids; the ceiling hosts every count the space holds.
fn admit_ask(live: u64, index: u64, asked: u64, blocks: u64) -> Result<(), String> {
    let room = index.saturating_sub(live + asked);
    if blocks == 0 || blocks > room {
        return Err(format!(
            "the block pool's index space of {index} holds {live} block(s) with {asked} asked, room for {room}"
        ));
    }
    Ok(())
}

impl Crossing {
    fn new(stats: &'static ModelLayoutStats, plan: &MemPlan, ceiling: u64) -> Self {
        Self {
            stats,
            launch_charge: plan.kv_charge,
            ceiling_budget: ceiling,
            launch_blocks: plan.kv_blocks,
            live: Mutex::new(plan.clone()),
            target: Mutex::default(),
        }
    }

    /// At the launch charge; the controller's peak is the only bound.
    fn plan_at(&self, budget: u64, kv_blocks: u64) -> anyhow::Result<MemPlan> {
        plan::PlanInputs::live(budget as f64 / GB, self.launch_charge).blocks(kv_blocks).solve(self.stats)
    }

    /// The plan goes live and the controller follows; the chunk, the working region and the states stay.
    fn land(&self, live: &mut MemPlan, plan: MemPlan, at_floor: bool) {
        super::controller::crossed(PlanTotals::of(&plan, self.ceiling_budget), at_floor);
        *live = plan;
    }

    /// The largest budget under the ceiling whose plan `room` hosts at `kv_blocks`, else the smallest (`at_floor`).
    fn hostable(
        &self, room: impl Fn(&MemPlan) -> i64, fits: u64, kv_blocks: u64, solves: &Cell<u32>,
    ) -> (MemPlan, bool) {
        let solve = |gb: f64| {
            solves.set(solves.get() + 1);
            self.plan_at((gb * GB) as u64, kv_blocks).ok()
        };
        match plan::hostable_budget(self.ceiling_budget as f64 / GB, room, solve) {
            Ok(Ok((_, plan))) => (plan, false),
            Ok(Err((plan, _))) => (plan, true),
            Err(_) => {
                solves.set(solves.get() + 1);
                (self.plan_at(fits, kv_blocks).expect("the live budget fits its own plan"), true)
            }
        }
    }

    /// One solve: the KV blocks counted first, then the largest budget fitting the peak; an unhosted ask is carried.
    fn decide(
        &self, from: &MemPlan, ask: Option<u64>, kv_ask: u64, free_uncached: u64, free_cached: u64, index: u64,
    ) -> Decision {
        let (live_blocks, live_budget) = (from.kv_blocks, from.mem_budget_bytes);
        let total = from.total_allocated_bytes();
        let kv_object = plan::kv_block_object_bytes(self.stats).max(1);
        let solves = Cell::new(0u32);
        let asked_peak = ask.unwrap_or(total);
        let solve_at = |peak: u64, blocks: u64| {
            let room = |p: &MemPlan| peak as i64 - p.total_allocated_bytes() as i64;
            self.hostable(room, live_budget, blocks, &solves)
        };
        let mut count = live_blocks;
        if asked_peak < total {
            count -= free_uncached.min((total - asked_peak).div_ceil(kv_object)).min(count);
        }
        let granted = kv_ask.min(index.saturating_sub(count));
        count += granted;
        let (mut plan, mut at_floor);
        if asked_peak < total {
            (plan, at_floor) = solve_at(asked_peak, count);
            if at_floor && plan.total_allocated_bytes() > asked_peak {
                let n = free_cached
                    .min((plan.total_allocated_bytes() - asked_peak).div_ceil(kv_object))
                    .min(count - granted);
                if n > 0 {
                    count -= n;
                    (plan, at_floor) = solve_at(asked_peak, count);
                }
            }
        } else {
            // A regrow: the charge's blocks (the guarantee) refilled first, the expert pool at the peak less that
            // refill, the rest of the blocks to the launch count once every layer seals (an offloaded launch holds
            // the charge); never below the live plan.
            let charged = plan::charge_blocks(self.stats, self.launch_charge, from.working.recorded_width)
                .min(self.launch_blocks);
            let refill_bytes = charged.saturating_sub(count) * kv_object;
            let expert_peak = asked_peak.saturating_sub(refill_bytes).max(total);
            (plan, at_floor) = solve_at(expert_peak, count);
            let room = asked_peak.saturating_sub(plan.total_allocated_bytes()) / kv_object;
            let to = if plan.offloading { charged } else { self.launch_blocks };
            let mut refill = if ask.is_some() && !at_floor { to.saturating_sub(count).min(room) } else { 0 };
            while refill > 0 {
                let (p, floor) = solve_at(asked_peak, count + refill);
                if !floor && p.expert_pool_bytes >= from.expert_pool_bytes {
                    (count, plan) = (count + refill, p);
                    break;
                }
                let holding = p.total_allocated_bytes() + from.expert_pool_bytes.saturating_sub(p.expert_pool_bytes);
                refill = refill.saturating_sub(holding.saturating_sub(asked_peak).div_ceil(kv_object).max(1));
            }
        }
        let mut carried = kv_ask - granted;
        if granted > 0 && at_floor && plan.total_allocated_bytes() > asked_peak {
            let base = count - granted;
            let hosted = plan::largest(0, granted, |n| {
                let (p, floor) = solve_at(asked_peak, base + n);
                !(floor && p.total_allocated_bytes() > asked_peak)
            });
            carried += granted - hosted;
            count = base + hosted;
            (plan, at_floor) = solve_at(asked_peak, count);
        }
        debug_assert_eq!(plan.kv_blocks, count);
        Decision { plan, at_floor, carried, solves: solves.get() }
    }

    /// At a parked boundary on the engine thread; a park meeting a move in flight leaves the target to the mover.
    fn cross(&self, dev: &candle_core::Device) {
        // A give-up is unlocking the pools: nothing moves them now.
        if oscar_offload::segments::giving_up() {
            return;
        }
        let (ask, kv_ask, requested_at) = {
            let mut t = lock(&self.target);
            if !t.pending() {
                return;
            }
            t.deferred = move_in_flight();
            if t.deferred {
                return;
            }
            (t.ask.take(), std::mem::take(&mut t.kv_ask) as u64, t.requested_at.take().unwrap_or_else(Instant::now))
        };
        let parked = Instant::now();
        let mut live = lock(&self.live);
        let from = live.clone();
        let pd = crate::engine::offload_dispatcher().expect("a crossing is installed past the dispatcher");
        let kv = KvPool::installed().expect("the block pool is built before any crossing");
        // A park that cannot lock the block state leaves the KV pool be; admitted requests' room is not free.
        let mut manager = engine_hooks::kv_manager().and_then(|m| m.try_lock().ok());
        let admitted = engine_hooks::admitted_blocks() as u64;
        let (free_uncached, free_cached) = manager.as_deref().map_or((0, 0), |m| {
            sheddable(
                m.allocatable_blocks() as u64,
                m.num_free_blocks() as u64,
                m.free_uncached_blocks() as u64,
                admitted,
            )
        });
        let index = kv.blocks() as u64;
        let Decision { plan, at_floor, carried, solves } =
            self.decide(&from, ask, kv_ask, free_uncached, free_cached, index);
        if carried > 0 {
            lock(&self.target).kv_ask += carried as usize;
        }
        // The line bounds the larger plan while the pools move, the new one once they settle (`owe_to_mover`).
        let line = &oscar_offload::machine::PLAN;
        line.arm(plan.mem_budget_bytes.max(line.limit()));
        // The pools follow by the difference; the background thread settles every move.
        let withdrawn = match manager.as_deref_mut() {
            Some(m) if plan.kv_blocks < from.kv_blocks => kv.shed(m, (from.kv_blocks - plan.kv_blocks) as usize),
            _ => 0,
        };
        let taken =
            if plan.kv_blocks > from.kv_blocks { kv.regrow((plan.kv_blocks - from.kv_blocks) as usize) } else { 0 };
        drop(manager);
        let moved = (shape(&plan) != shape(&from)).then(|| {
            // The park's half of a move (unseals, retires) cannot be refused: an error here is a broken invariant.
            let t = Instant::now();
            MOVE_IN_FLIGHT.store(true, Ordering::Release);
            let m = pd
                .move_pool(plan.hot_budget, plan.carrier_layers, dev)
                .expect("the park's half of an expert pool move has no take to refuse");
            (m, t.elapsed())
        });
        if moved.is_some() || withdrawn > 0 || taken > 0 {
            owe_to_mover();
        }
        let to = |a: u64, b: u64| format!("{:.2} -> {:.2} GB", a as f64 / GB, b as f64 / GB);
        let at_floor_said = if at_floor { " (the smallest plan at this context bound)" } else { "" };
        let kept = if admitted > 0 { format!("; {admitted} kept for requests in flight") } else { String::new() };
        let (coming, asked) = (kv_ask.saturating_sub(carried), kv_ask > 0);
        let asked =
            asked.then(|| format!(" - {kv_ask} asked for a prompt, {coming} of them coming, {carried} carried"));
        let pool_moved = moved.map_or(String::new(), |(m, t)| {
            let (ms, (u, s, r)) = (t.as_millis(), (m.unsealed, m.sealing, m.retired));
            format!(" (the pool moved in {ms} ms: {u} layer(s) unsealed, {s} sealing, {r} segment(s) retired)")
        });
        tracing::info!(target: "oscar::crossing",
            "budget {} for {:.2} GB peak{at_floor_said}; expert pool {} -> {} ({}), KV pool {} -> {} blocks \
             ({}{kept}){}; {solves} solves, parked {} ms after request{pool_moved}, {} ms in all{}",
            to(from.mem_budget_bytes, plan.mem_budget_bytes),
            ask.unwrap_or(from.total_allocated_bytes()) as f64 / GB,
            from.pool_shape(),
            plan.pool_shape(),
            to(from.expert_pool_bytes, plan.expert_pool_bytes),
            from.kv_blocks,
            plan.kv_blocks,
            to(from.kv_cache_bytes, plan.kv_cache_bytes),
            asked.unwrap_or_default(),
            parked.duration_since(requested_at).as_millis(),
            parked.elapsed().as_millis(),
            stamp(),
        );
        self.land(&mut live, plan, at_floor && withdrawn == 0);
    }
}

pub(super) fn move_in_flight() -> bool {
    MOVE_IN_FLIGHT.load(Ordering::Acquire)
}

/// Wakes the mover (spawned at the first) for the pools' owed moves; once idle it asks again for a deferred target.
fn owe_to_mover() {
    MOVE_IN_FLIGHT.store(true, Ordering::Release);
    let mover = MOVER.get_or_init(|| {
        std::thread::Builder::new()
            .name("oscar-pool-mover".into())
            .spawn(|| {
                let pd = crate::engine::offload_dispatcher().expect("the background thread moves the installed pools");
                let mut held_since = None;
                loop {
                    let re_land = |shape, what: &str, why: &str| re_land(Some(shape), None, what, why);
                    if pd.settle_moves(&mut held_since, crate::drafter::in_session, re_land, settle_owed) {
                        if let Some(plan) = live() {
                            oscar_offload::machine::PLAN.arm(plan.mem_budget_bytes);
                        }
                        MOVE_IN_FLIGHT.store(false, Ordering::Release);
                        if std::mem::take(&mut lock(&crossing().target).deferred) {
                            engine_hooks::request_memory_crossing();
                        }
                        std::thread::park();
                    } else {
                        std::thread::park_timeout(Duration::from_millis(20));
                    }
                }
            })
            .expect("the pool mover thread spawns")
            .thread()
            .clone()
    });
    mover.unpark();
}

/// After the expert pool's frees: the KV blocks back to the engine.
fn settle_owed() {
    let (Some(kv), Some(manager)) = (KvPool::installed(), engine_hooks::kv_manager()) else { return };
    if !kv.pending() {
        return;
    }
    let t = Instant::now();
    let (released, taken, refused) = kv.settle();
    if !taken.is_empty() {
        manager.blocking_lock().restore(&taken);
    }
    kv.handed_back();
    tracing::info!(target: "oscar::kv_pool",
        "{released} blocks released, {} taken and handed back in {}; {} present, {} absent{}{}",
        taken.len(),
        mistralrs_core::live::duration_text(t.elapsed().as_secs_f64()),
        kv.live_blocks(),
        kv.absent(),
        refused.as_ref().map_or(String::new(), |e| format!("; a take was refused ({e}) and the plan re-lands at the blocks present")),
        stamp()
    );
    if let Some(e) = refused {
        re_land(None, Some(kv.live_blocks() as u64), "a mapping was refused", &e.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::KV_CHARGE_BENCH;
    use crate::plan::tests::routed;

    fn crossing_at(stats: ModelLayoutStats, ceiling_gb: f64) -> Crossing {
        let stats: &'static ModelLayoutStats = Box::leak(Box::new(stats));
        let plan = plan::PlanInputs::live(ceiling_gb, KV_CHARGE_BENCH).solve(stats).expect("plans");
        Crossing::new(stats, &plan, (ceiling_gb * GB) as u64)
    }

    /// A shed takes uncached free blocks first, a regrow refills to the launch count, an ask is carried, never refused.
    /// Holding 703 of the 1,687 blocks it may grow to in a pool of 1,877, a request leaves 190; none admitted, all go.
    #[test]
    fn the_kv_term_is_decided_as_counts_and_an_ask_waits() {
        let a = crossing_at(routed(60, 128, 1 << 30), 40.0);
        let from = a.live.lock().unwrap().clone();
        let (object, live) = (plan::kv_block_object_bytes(a.stats), from.kv_blocks);
        let index = live + 10_000;
        let decide = |ask, kv_ask, free| a.decide(&from, ask, kv_ask, free, 0, index);
        // Two slots per layer short, a fifth of the pool free: free blocks first, the rest from the expert pool.
        let peak = from.total_allocated_bytes() - 2 * a.stats.bytes_per_hb_unit;
        let d = decide(Some(peak), 0, live / 5);
        assert_eq!(d.plan.kv_blocks, live - live / 5);
        assert!(d.plan.total_allocated_bytes() <= peak && d.plan.hot_budget < from.hot_budget && !d.at_floor);
        // Forty blocks short with more free: blocks alone, the expert pool untouched.
        let d = decide(Some(from.total_allocated_bytes() - 40 * object), 0, live / 2);
        assert_eq!((d.plan.kv_blocks, d.plan.hot_budget), (live - 40, from.hot_budget));
        // An ask of 400 blocks at the peak in force: the expert pool pays.
        let d = decide(None, 400, 0);
        assert_eq!((d.plan.kv_blocks, d.carried), (live + 400, 0));
        assert!(d.plan.hot_budget < from.hot_budget);
        // An ask no plan hosts at the floor: what the floor plan hosts comes, the rest is carried.
        let d = decide(Some(from.non_expert_bytes + from.working.bytes() + from.kv_cache_bytes), 5_000, 0);
        assert!(d.carried > 0 && d.carried <= 5_000, "carried {}", d.carried);
        assert_eq!(d.plan.kv_blocks, live + 5_000 - d.carried);
        // Refused past the index space, accepted inside it; the ceiling hosts every count the space holds.
        assert!(admit_ask(live, index, 0, 10_001).is_err() && admit_ask(live, index, 9_000, 1_001).is_err());
        assert!(admit_ask(live, index, 0, 400).is_ok());
        let space = plan::PoolCeilings::solve(a.stats, &from, 40.0).kv_index_blocks;
        assert!(a.plan_at(a.ceiling_budget, space).is_ok(), "the ceiling hosts the index space");
        assert_eq!(sheddable(1877, 1174, 1100, 1687), (190, 0));
        assert_eq!(sheddable(1877, 1174, 100, 1687), (100, 90));
        assert_eq!(sheddable(1877, 1174, 1100, 0), (1100, 74));
        assert_eq!(sheddable(1200, 400, 400, 1687), (0, 0));
    }

    /// As a claimant leaves in thirds of a slot, no regrow passes its peak, loses ground or takes over three searches.
    /// A resident launch holds the remainder (the 35B: 47k blocks at the gate's charge of a few hundred); a regrow
    /// refills the charge, then the experts, then the rest of the blocks.
    #[test]
    fn a_chain_of_regrows_climbs_back_to_the_launch_the_charge_first_then_the_experts_then_the_rest() {
        let mut a = crossing_at(routed(60, 128, 1 << 30), 40.0);
        a.launch_charge = KvCharge::for_interactive(65_536, 4_096, 0);
        let launch = plan::PlanInputs::live(40.0, a.launch_charge).solve(a.stats).expect("plans");
        a.launch_blocks = launch.kv_blocks;
        let charged = plan::charge_blocks(a.stats, a.launch_charge, launch.working.recorded_width);
        assert_eq!((launch.offloading, charged), (true, launch.kv_blocks), "an offloaded launch holds the charge");
        let (index, object, unit) =
            (launch.kv_blocks + 10_000, plan::kv_block_object_bytes(a.stats), a.stats.bytes_per_hb_unit);
        let free = launch.kv_blocks - launch.kv_blocks / 20;
        let search = 1 + 2 * (a.ceiling_budget as f64 / (1u64 << 20) as f64).log2().ceil() as u32;
        let decide = |from: &MemPlan, peak, free| a.decide(from, Some(peak), 0, free, 0, index);
        let shed = decide(&launch, launch.total_allocated_bytes() - free * object - 3 * unit, free);
        let mut d = decide(&shed.plan, shed.plan.total_allocated_bytes(), 0);
        let tables_peak = d.plan.total_allocated_bytes() + (launch.kv_blocks - d.plan.kv_blocks + 1) * object;
        for step in 1u64.. {
            let peak = shed.plan.total_allocated_bytes() + step * unit / 3;
            let next = decide(&d.plan, peak, 0);
            let (was, now) =
                ((d.plan.expert_pool_bytes, d.plan.kv_blocks), (next.plan.expert_pool_bytes, next.plan.kv_blocks));
            let within = next.plan.total_allocated_bytes() <= peak && now.1 <= launch.kv_blocks;
            assert!(within, "step {step}: past its peak or the launch count");
            let (climbs, searches) = (now.0 >= was.0 && now.1 >= was.1, next.solves <= 3 * search + 1);
            assert!(climbs && searches, "step {step}: {now:?} after {was:?}, {} solves", next.solves);
            if (now.1, shape(&next.plan)) == (launch.kv_blocks, shape(&launch)) {
                break;
            }
            assert!(step < 1_000, "the chain never climbed back to the launch");
            d = next;
        }
        let tight = decide(&shed.plan, tables_peak, 0);
        let blocks = tight.plan.kv_blocks;
        let fits = |p: &MemPlan| tables_peak as i64 - p.total_allocated_bytes() as i64;
        let (more, over) = a.hostable(fits, tight.plan.mem_budget_bytes, blocks + 1, &Cell::new(0));
        let one_more = !over && more.expert_pool_bytes >= shed.plan.expert_pool_bytes;
        assert!(blocks == launch.kv_blocks || !one_more, "{blocks} blocks where one more fits");
        let back = decide(&decide(&launch, 0, free).plan, launch.total_allocated_bytes() + 8 * unit, 0);
        assert_eq!((back.plan.kv_blocks, shape(&back.plan)), (launch.kv_blocks, shape(&launch)), "refilled");
        // The resident launch.
        let mut a = crossing_at(routed(40, 128, 128 << 20), 40.0);
        a.launch_charge = KvCharge::for_interactive(4_096, 1_024, 0);
        let launch = plan::PlanInputs::live(40.0, a.launch_charge).solve(a.stats).expect("plans");
        a.launch_blocks = launch.kv_blocks;
        let charged = plan::charge_blocks(a.stats, a.launch_charge, launch.working.recorded_width);
        assert!(!launch.offloading && charged * 10 < launch.kv_blocks, "{charged} of {} blocks", launch.kv_blocks);
        let (index, object) = (launch.kv_blocks + 10_000, plan::kv_block_object_bytes(a.stats));
        let decide = |from: &MemPlan, peak, free| a.decide(from, Some(peak), 0, free, 0, index);
        // The apex: all but a few blocks shed, then four GB of experts.
        let free = launch.kv_blocks - 8;
        let apex = decide(&launch, launch.total_allocated_bytes() - free * object - 4 * (1 << 30), free).plan;
        assert!(apex.kv_blocks < charged && apex.expert_pool_bytes < launch.expert_pool_bytes);
        let mut d = apex.clone();
        for step in 1u64..=40 {
            let next = decide(&d, apex.total_allocated_bytes() + step * (2 << 30), 0).plan;
            assert!(next.kv_blocks >= charged, "step {step}: the charge first");
            if next.offloading {
                assert!(next.expert_pool_bytes > d.expert_pool_bytes, "step {step}: the experts next");
                assert_eq!(next.kv_blocks, charged, "step {step}: blocks past the charge while a layer pages");
            } else {
                assert!(next.kv_blocks > d.kv_blocks || next.kv_blocks == launch.kv_blocks, "step {step}: the rest");
            }
            d = next;
            if (d.kv_blocks, shape(&d)) == (launch.kv_blocks, shape(&launch)) {
                break;
            }
            assert!(step < 40, "the chain never climbed back to the launch");
        }
        assert_eq!((d.kv_blocks, shape(&d)), (launch.kv_blocks, shape(&launch)), "the remainder last");
    }
}
