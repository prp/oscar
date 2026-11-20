//! The adaptation policy's sensing half: the live plan's room under the bounds, a quantum of hysteresis either way.

use std::collections::VecDeque;
use std::sync::{Mutex, MutexGuard};

use super::SETTLING_WINDOW;
use crate::cli::console::stamp;
use crate::machine::{claimant_charge, signed, under_pressure, Bound, Bounds, MemorySnapshot, CHANGE_QUANTUM, GB};
use crate::plan::MemPlan;

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct PlanTotals {
    peak: u64,
    metal: u64,
    locked: u64,
    budget: u64,
    ceiling: u64,
    elastic: u64,
}

impl PlanTotals {
    pub(super) fn of(plan: &MemPlan, ceiling: u64) -> Self {
        Self {
            peak: plan.total_allocated_bytes(),
            metal: plan.metal_bytes(),
            locked: plan.locked_bytes(),
            budget: plan.mem_budget_bytes,
            ceiling,
            elastic: plan.expert_pool_bytes + plan.kv_cache_bytes,
        }
    }
}

fn rest_line(floor: u64) -> u64 {
    floor + claimant_charge() + CHANGE_QUANTUM
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Steady,
    Shed(u64),
    Regrow(u64),
}

/// The machine's terms one sample reads, settled through the window.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Foreign {
    pub(super) non_droppable: u64,
    pub(super) gpu: u64,
    pub(super) wired: u64,
    pub(super) reclaimable: u64,
}

impl Foreign {
    pub(super) fn of(s: &MemorySnapshot) -> Self {
        Self { non_droppable: s.foreign(), gpu: s.foreign_gpu(), wired: s.foreign_wired(), reclaimable: s.reclaimable }
    }
}

fn median(terms: impl Iterator<Item = u64>) -> u64 {
    let mut v: Vec<u64> = terms.collect();
    v.sort_unstable();
    (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2
}

/// Each foreign term its median over the window, the limits as the snapshot reads them.
pub(super) fn settled_bounds(window: impl Iterator<Item = Foreign> + Clone, s: &MemorySnapshot) -> Bounds {
    let settled = |term: fn(&Foreign) -> u64| median(window.clone().map(|f| term(&f)));
    Bounds {
        foreign: settled(|f| f.non_droppable),
        foreign_gpu: settled(|f| f.gpu),
        foreign_wired: settled(|f| f.wired),
        ..s.bounds()
    }
}

#[derive(Default)]
struct Controller {
    plan: PlanTotals,
    floor: u64,
    /// Since the last crossing or turn, so a verdict acts only on a window that holds it throughout.
    window: VecDeque<Foreign>,
    verdict: Option<Verdict>,
    /// Held, so a crossing that lands nothing is asked once until the room moves a quantum.
    asked: Option<u64>,
    /// Samples the verdict has held since its turn, past the first window.
    held: usize,
    at_floor: bool,
    /// A shed the KV pool alone paid leaves the budget at the ceiling over a smaller plan.
    largest_peak: u64,
}

impl Controller {
    fn new(plan: PlanTotals, floor: u64) -> Self {
        Self { plan, floor, largest_peak: plan.peak, ..Default::default() }
    }

    /// One sample in, the peak to ask for out; asked outside this lock, as the engine reports here holding the plan's.
    fn observe(&mut self, s: &MemorySnapshot, moving: bool) -> Option<u64> {
        // A move's freed segments read as foreign for a second: none of it reaches the window.
        if moving {
            return None;
        }
        if self.window.len() == SETTLING_WINDOW {
            self.window.pop_front();
        }
        self.window.push_back(Foreign::of(s));
        if under_pressure(s, self.floor) {
            let target = self.plan.peak.saturating_sub(rest_line(self.floor).saturating_sub(s.reclaimable));
            if self.at_floor || self.asked.is_some_and(|a| a <= target) {
                return None;
            }
            let at = s.reclaimable as f64 / GB;
            return self.ask(target, format_args!("pressure: {at:.2} GB reclaimable, under plan's rest"));
        }
        if self.window.len() < SETTLING_WINDOW {
            return None;
        }
        let bounds = settled_bounds(self.window.iter().copied(), s);
        let (room, bound) = bounds.room(self.plan.peak, self.plan.metal, self.plan.locked, self.floor, 0);
        // Rest in the watchdog's own quantity: reclaimable memory sits below the bounds' room by the file cache.
        let rest = signed(median(self.window.iter().map(|f| f.reclaimable))) - signed(rest_line(self.floor));
        let (room, bound) = if self.floor > 0 && rest < room { (rest, Bound::Reclaimable) } else { (room, bound) };
        let quantum = signed(CHANGE_QUANTUM);
        let verdict = if room <= -quantum {
            Verdict::Shed(-room as u64)
        } else if room >= quantum && (self.plan.budget < self.plan.ceiling || self.plan.peak < self.largest_peak) {
            // The whole room, as a launch sizes to the bound; the crossing's solve caps the target at the ceiling.
            Verdict::Regrow(room as u64)
        } else {
            Verdict::Steady
        };
        // A turn, the first after a crossing too, acts only on a window that holds it throughout: sized settled, not at
        // the median's midpoint, and clear of the second for which the kernel's counters still show a released segment.
        let turned = self.verdict.replace(verdict).map(|v| std::mem::discriminant(&v));
        if turned != Some(std::mem::discriminant(&verdict)) {
            self.say(room, bound, &bounds, s);
            self.window.clear();
            self.held = 0;
            return None;
        }
        self.held += 1;
        let (target, why) = match verdict {
            Verdict::Shed(short) if !self.at_floor => (self.plan.peak.saturating_sub(short), "settled shortfall"),
            // A regrow waits a second window: room that outlasts the machine's own swing, or the next shortfall is its own.
            Verdict::Regrow(room) if self.held > SETTLING_WINDOW => {
                (self.plan.peak.saturating_add(room), "room two windows on")
            }
            _ => return None,
        };
        if self.asked.is_some_and(|a| a.abs_diff(target) < CHANGE_QUANTUM) {
            return None;
        }
        self.ask(target, format_args!("{why}"))
    }

    /// The verdict as it turns: the room, others' memory and the budget, under -v the bound, the plan and the ledger.
    fn say(&self, room: i64, bound: Bound, bounds: &Bounds, s: &MemorySnapshot) {
        let (gb, p) = (|b: u64| b as f64 / GB, self.plan);
        let what = match self.verdict {
            Some(Verdict::Shed(_)) if self.at_floor => {
                tracing::info!(target: "oscar::controller",
                    "at floor - holding {:.2} GB, smallest plan at this context bound, machine {:.2} GB short; \
                     nothing more sheds, watchdog is backstop{}",
                    gb(p.budget),
                    -room as f64 / GB,
                    stamp()
                );
                return;
            }
            Some(Verdict::Shed(b)) => format!("shed {:.2} GB", gb(b)),
            Some(Verdict::Regrow(b)) => format!("regrow {:.2} GB", gb(b)),
            _ => "steady".to_string(),
        };
        let detail = if crate::cli::console::verbose() {
            format!(
                " | the {} bound: foreign GPU {:.2} wired {:.2}; plan peak {:.2} GB, Metal {:.2}, locked {:.2}; {} | {}{}",
                bound.label(),
                gb(bounds.foreign_gpu),
                gb(bounds.foreign_wired),
                gb(p.peak),
                gb(p.metal),
                gb(p.locked),
                self.word(),
                s.ledger(),
                stamp(),
            )
        } else {
            String::new()
        };
        crate::detail!(target: "oscar::controller",
            "{what} - room {:+.2} GB, other programs {:.2} GB, budget {:.2} of {:.2} GB{detail}",
            room as f64 / GB,
            gb(bounds.foreign),
            gb(p.budget),
            gb(p.ceiling),
        );
    }

    /// The request, pending from here.
    fn ask(&mut self, target: u64, why: std::fmt::Arguments) -> Option<u64> {
        let (verb, by) = match target.checked_sub(self.plan.peak) {
            Some(up) => ("regrowing", up),
            None => ("shedding", self.plan.peak - target),
        };
        tracing::info!(target: "oscar::controller",
            "{verb} {:.2} GB to {:.2} GB peak - {why}{}",
            by as f64 / GB,
            target as f64 / GB,
            stamp()
        );
        self.asked = Some(target);
        Some(target)
    }

    /// The bar's word: at floor only while short, as the room is what lifts it.
    fn word(&self) -> &'static str {
        match self.verdict {
            Some(Verdict::Shed(_)) | None if self.at_floor => "at floor",
            Some(Verdict::Shed(_)) => "shedding",
            Some(Verdict::Regrow(_)) => "regrowing",
            _ => "steady",
        }
    }

    /// The plan as landed: the window restarts, so the next verdict reads no sample from before it.
    fn crossed(&mut self, plan: PlanTotals, at_floor: bool) {
        (self.plan, self.at_floor, self.verdict) = (plan, at_floor, None);
        self.largest_peak = self.largest_peak.max(plan.peak);
        self.window.clear();
    }
}

static CONTROLLER: Mutex<Option<Controller>> = Mutex::new(None);

fn controller() -> MutexGuard<'static, Option<Controller>> {
    CONTROLLER.lock().unwrap_or_else(|e| e.into_inner())
}

/// Hand over the plan once the engine is built, so the load's transients are never judged; adaptive launches only.
pub fn control(plan: &MemPlan, ceiling: u64) {
    *controller() = Some(Controller::new(PlanTotals::of(plan, ceiling), crate::config::watchdog_floor_bytes()));
}

/// The sensor's sample for the controller, a pool move in flight withheld: the request, pending from here.
pub(super) fn observe(s: &MemorySnapshot) -> Option<u64> {
    controller().as_mut()?.observe(s, super::crossing::move_in_flight())
}

pub(super) fn crossed(plan: PlanTotals, at_floor: bool) {
    if let Some(c) = controller().as_mut() {
        c.crossed(plan, at_floor);
    }
}

/// The state word, the live plan's elastic bytes and the rest of its total; `None` on a static plan.
pub fn bar() -> Option<(&'static str, u64, u64)> {
    controller().as_ref().map(|c| (c.word(), c.plan.elastic, c.plan.peak.saturating_sub(c.plan.elastic)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: u64 = 1 << 30;

    /// The controller as the sensor loop drives it, every request's peak kept; `moving` as the background thread's flag.
    struct Rig {
        c: Controller,
        asked: Vec<u64>,
        moving: bool,
    }

    impl Rig {
        fn new() -> Self {
            Self { c: Controller::new(totals(54 * G), G), asked: Vec::new(), moving: false }
        }

        fn ticks(&mut self, n: usize, s: &MemorySnapshot) {
            for _ in 0..n {
                if let Some(peak) = self.c.observe(s, self.moving) {
                    self.asked.push(peak);
                }
            }
        }
    }

    /// A budget-bound plan at `budget`: allocated to the budget, a 4 GB prefill peak under it.
    fn totals(budget: u64) -> PlanTotals {
        PlanTotals {
            peak: budget - 4 * G,
            metal: budget,
            locked: budget - 12 * G,
            budget,
            ceiling: 54 * G,
            elastic: 30 * G,
        }
    }

    /// A healthy machine with `foreign_gb` wired: 20 GB reclaimable, well above the pressure line.
    fn at(foreign_gb: u64) -> MemorySnapshot {
        MemorySnapshot {
            physical: 64 * G,
            wired: foreign_gb * G,
            reclaimable: 20 * G,
            gpu_ceiling: 80 * G,
            ..Default::default()
        }
    }

    /// A transient under half the window is no verdict; a lasting claimant turns it, and a window holding the turn asks
    /// once, the shortfall off the peak; a move in flight is not sampled; a landed plan is asked the room two windows on;
    /// pressure sheds from the sample and rests a quantum clear of the line; at the floor the controller holds.
    #[test]
    fn the_controller_asks_once_a_window_the_shortfall_or_the_room_and_holds_at_the_floor() {
        let mut r = Rig::new();
        // 64 − 8 foreign − 1 floor − 50 peak = 5 GB of room, the largest plan at the ceiling: steady.
        r.ticks(2 * SETTLING_WINDOW, &at(8));
        assert_eq!(r.c.verdict, Some(Verdict::Steady));
        r.ticks(3, &at(20));
        assert_eq!(r.c.verdict, Some(Verdict::Steady), "a three-sample spike");
        // Lasting, a 20 GB claimant turns the median on its fourth sample, at its midpoint (room −1): a new window.
        r.ticks(1, &at(20));
        assert_eq!((r.c.verdict, r.c.window.len(), r.c.word()), (Some(Verdict::Shed(G)), 0, "shedding"));
        r.ticks(SETTLING_WINDOW - 1, &at(20));
        assert!(r.asked.is_empty(), "asked inside the window");
        r.ticks(1, &at(20));
        assert_eq!(r.asked, vec![43 * G], "the shortfall off the peak");
        r.ticks(SETTLING_WINDOW, &at(20));
        assert_eq!(r.asked.len(), 1, "a peak within a quantum of the one asked");
        // Landed at 43: 64 − 20 − 1 − 43 = 0 GB of room, steady, its turn a window on.
        r.c.crossed(totals(47 * G), false);
        r.ticks(SETTLING_WINDOW, &at(20));
        assert_eq!(r.c.word(), "steady");
        r.ticks(SETTLING_WINDOW, &at(20));
        assert_eq!((r.c.verdict, r.asked.len()), (Some(Verdict::Steady), 1));
        // The move in flight.
        let mut r = Rig::new();
        r.ticks(2 * SETTLING_WINDOW, &at(8));
        r.moving = true;
        r.ticks(2 * SETTLING_WINDOW, &at(20));
        assert_eq!(r.c.verdict, Some(Verdict::Steady), "the move's transient was judged");
        r.moving = false;
        r.ticks(SETTLING_WINDOW, &at(20));
        assert_eq!((r.c.verdict, r.asked.len()), (Some(Verdict::Shed(G)), 0), "turned, asked a window on");
        // The memory must still come back: a shed the KV pool alone paid leaves the budget at the ceiling, and a
        // crossing landing nothing (the next segment wider than the room) is asked once till the room moves a quantum.
        // A quiet machine: 64 − 8 − 1 − 47 = 8 GB of room, asked whole; under the ceiling 64 − 8 − 1 − 46 = 9, 55 asked
        let landed = |t| {
            let mut r = Rig::new();
            r.c.crossed(t, false);
            r.ticks(3 * SETTLING_WINDOW - 1, &at(8));
            assert!(r.asked.is_empty(), "a regrow inside its second window");
            r.ticks(1, &at(8));
            assert_eq!(r.asked, vec![55 * G]);
            r
        };
        landed(PlanTotals { peak: 47 * G, metal: 51 * G, ..totals(54 * G) });
        let mut r = landed(totals(50 * G));
        r.c.crossed(totals(50 * G), false);
        r.ticks(3 * SETTLING_WINDOW, &at(8));
        assert_eq!(r.asked.len(), 1);
        r.ticks(SETTLING_WINDOW, &at(7));
        assert_eq!(r.asked, vec![55 * G, 56 * G], "a quantum's move asks again");
        // One request at once; a shallower sample asks nothing, a deeper one asks again; above the line, the window.
        let mut r = Rig::new();
        r.ticks(SETTLING_WINDOW, &at(8));
        let under = |reclaimable| MemorySnapshot { reclaimable, ..at(8) };
        let shed = |peak: u64, reclaimable| peak - (rest_line(G) - reclaimable);
        r.ticks(1, &under(G + G / 2));
        assert_eq!(r.asked, vec![shed(50 * G, G + G / 2)]);
        r.ticks(1, &under(G + G / 2));
        r.ticks(1, &under(G + G / 2 + G / 64));
        assert_eq!(r.asked.len(), 1);
        r.c.crossed(totals(48 * G), false);
        r.ticks(1, &under(G / 4));
        assert_eq!(r.asked.last(), Some(&shed(44 * G, G / 4)));
        r.c.crossed(totals(40 * G), false);
        r.ticks(3 * SETTLING_WINDOW, &at(8));
        assert_eq!(r.asked.len(), 3, "the settled regrow, two windows on");
        // Room under every bound sheds to the rest line; a regrow takes only what reclaimable memory leaves above it:
        // 5 GB of room under the bounds (64 − 8 − 1 − 50), but only a claimant's charge reclaimable above the floor.
        let mut r = Rig::new();
        r.ticks(2 * SETTLING_WINDOW, &under(G + claimant_charge()));
        assert_eq!(r.c.verdict, Some(Verdict::Shed(CHANGE_QUANTUM)), "a quantum short of where it rests");
        assert_eq!(r.asked, vec![50 * G - CHANGE_QUANTUM]);
        r.c.crossed(totals(54 * G - CHANGE_QUANTUM), false);
        r.ticks(2 * SETTLING_WINDOW, &under(rest_line(G)));
        assert_eq!(r.c.verdict, Some(Verdict::Steady), "at rest with 5 GB of room under the bounds");
        r.ticks(SETTLING_WINDOW, &under(rest_line(G) + G));
        assert_eq!(r.c.verdict, Some(Verdict::Regrow(G)));
        // A crossing no plan can host lands on the smallest; the controller asks no more until the claimant leaves:
        // 64 − 60 − 1 − 50 = −47, asked down by that.
        let mut r = Rig::new();
        r.ticks(2 * SETTLING_WINDOW, &at(60));
        assert_eq!(r.asked, vec![3 * G]);
        r.c.crossed(totals(20 * G), true);
        assert_eq!(r.c.word(), "at floor");
        r.ticks(2 * SETTLING_WINDOW, &at(60));
        assert_eq!((r.c.verdict, r.c.word(), r.asked.len()), (Some(Verdict::Shed(13 * G)), "at floor", 1));
        r.ticks(SETTLING_WINDOW, &at(8));
        assert_eq!(r.c.word(), "regrowing", "the floor lifts with the room");
    }
}
