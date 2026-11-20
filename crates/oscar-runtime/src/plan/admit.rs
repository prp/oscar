//! The told budget against the machine: the largest budget its bounds host, and the complaint where none does.

use super::*;
use crate::machine::{claimant_charge, Bound, Bounds};

/// A launch's plan at `charge` with the segment size that launch would fix, from its own ceiling solve.
pub fn plan_at_charge(stats: &ModelLayoutStats, mem_gb: f64, charge: KvCharge) -> Result<MemPlan> {
    let k = derive_segment_slots(stats, &PlanInputs::at(mem_gb, charge, None, None).solve(stats)?);
    PlanInputs::at(mem_gb, charge, None, Some(k)).solve(stats)
}

impl MemPlan {
    /// [`Bounds::room`] at the plan's launch, a claimant's charge kept standing (a running plan's rest line holds it).
    pub fn room_for(&self, bounds: &Bounds, margin: u64) -> (i64, Bound) {
        bounds.room(self.total_allocated_bytes(), self.metal_bytes(), self.locked_bytes(), margin, claimant_charge())
    }
}

/// `Ok` where every bound hosts the plan with `floor` kept, else the shortfall, the binding bound and its remedy.
pub fn hosted_by(plan: &MemPlan, machine: &crate::machine::MemorySnapshot, floor: u64) -> Result<()> {
    let bounds = machine.bounds();
    let (room, bound) = plan.room_for(&bounds, floor);
    if room < 0 {
        anyhow::bail!("{}", shortfall_text(plan, &bounds, floor, room.unsigned_abs(), bound, false));
    }
    Ok(())
}

/// The largest budget to `ceiling_gb` whose plan `room` hosts, a bisection to a MB as a plan's peak is monotone in it;
/// `Err` with the smallest plan the planner solves and its shortfall where not even that is hosted.
pub fn hostable_budget(
    ceiling_gb: f64, room: impl Fn(&MemPlan) -> i64, solve: impl Fn(f64) -> Option<MemPlan>,
) -> Result<std::result::Result<(f64, MemPlan), (MemPlan, u64)>> {
    let top =
        solve(ceiling_gb).ok_or_else(|| anyhow::anyhow!("no memory plan solves at {ceiling_gb:.1} GB ceiling"))?;
    if room(&top) >= 0 {
        return Ok(Ok((ceiling_gb, top)));
    }
    let ceiling = (ceiling_gb * GB) as u64;
    let solve = |bytes: u64| solve(bytes as f64 / GB);
    let (min, min_plan) = boundary((ceiling, top), 0, 1 << 20, &solve);
    let r = room(&min_plan);
    if r < 0 {
        return Ok(Err((min_plan, r.unsigned_abs())));
    }
    // Hosted at the minimum and not at the ceiling: the largest hosted budget is one more cut between them.
    let (best, plan) = boundary((min, min_plan), ceiling, 1 << 20, |bytes| solve(bytes).filter(|p| room(p) >= 0));
    Ok(Ok((best as f64 / GB, plan)))
}

/// The largest count in `lo..=hi` that the monotone `fits` takes; `lo` where none above it does.
pub(crate) fn largest(lo: u64, hi: u64, fits: impl Fn(u64) -> bool) -> u64 {
    if fits(hi) {
        return hi;
    }
    boundary((lo, ()), hi, 1, |n| fits(n).then_some(())).0
}

/// The boundary, to `tol`, between `yes` (a value the monotone `probe` takes, with its answer) and `no`.
fn boundary<T>(mut yes: (u64, T), mut no: u64, tol: u64, probe: impl Fn(u64) -> Option<T>) -> (u64, T) {
    while yes.0.abs_diff(no) > tol {
        let mid = yes.0.min(no) + yes.0.abs_diff(no) / 2;
        match probe(mid) {
            Some(t) => yes = (mid, t),
            None => no = mid,
        }
    }
    yes
}

/// The one hostability complaint: what the plan needs, what the binding bound leaves, the shortfall and the way out;
/// `smallest` where no smaller plan exists, so a lower budget is no way out.
pub fn shortfall_text(
    plan: &MemPlan, bounds: &crate::machine::Bounds, floor: u64, shortfall: u64, bound: crate::machine::Bound,
    smallest: bool,
) -> String {
    use crate::machine::Bound;
    let gb = |b: u64| format!("{:.1} GB", b as f64 / GB);
    let what = if smallest { "smallest plan" } else { "plan" };
    let (peak, locked) = (plan.total_allocated_bytes(), plan.locked_bytes());
    let locks = || format!("{} locked ({} + {} floor)", gb(locked + floor), gb(locked), gb(floor));
    // What the plan needs on the bound, and what the bound leaves: its limit less what other processes hold of it.
    let left = |name: &str, limit: u64, others: u64| {
        format!(
            "{name} leaves {} ({} limit, {} other processes)",
            gb(limit.saturating_sub(others)),
            gb(limit),
            gb(others)
        )
    };
    let (needs, leaves) = match bound {
        Bound::Machine => {
            let (step, b) = (claimant_charge(), bounds);
            (
                format!("{} ({} peak + {} floor + {} step)", gb(peak + floor + step), gb(peak), gb(floor), gb(step)),
                format!(
                    "machine has {} ({} paged, {} other processes, {} of it GPU)",
                    gb(b.physical.saturating_sub(b.foreign)),
                    gb(b.physical),
                    gb(b.foreign),
                    gb(b.foreign_gpu)
                ),
            )
        }
        Bound::Gpu => {
            (format!("{} on GPU", gb(plan.metal_bytes())), left("GPU limit", bounds.gpu_ceiling, bounds.foreign_gpu))
        }
        Bound::WireMachine => (locks(), left("machine wire limit", bounds.machine_wire_limit, bounds.foreign_wired)),
        Bound::WireTask => (locks(), left("task wire limit", bounds.task_wire_limit, 0)),
        Bound::Reclaimable => (gb(peak + floor), "less is reclaimable".to_string()),
    };
    let way = match bound.limits_line(bounds) {
        Some(line) => line,
        None if smallest => "free memory".to_string(),
        None => "free memory or lower --mem".to_string(),
    };
    format!("{what} needs {needs}, {leaves}: {} short - {way}", gb(shortfall))
}

/// The told budget as a ceiling, sized to the largest plan `bounds` host, with the line saying a limit sized it.
pub fn solve_under_limits(
    stats: &ModelLayoutStats, at: PlanInputs, bounds: Option<crate::machine::Bounds>,
) -> Result<(MemPlan, Option<String>)> {
    let plan = at.solve(stats)?;
    let Some(bounds) = bounds else {
        return Ok((plan, None)); // no reading: nothing to solve under
    };
    let ceiling_gb = plan.mem_budget_bytes as f64 / GB;
    let floor = watchdog_floor_bytes();
    let solve = |gb: f64| PlanInputs { mem_gb: gb, ..at }.solve(stats).ok();
    let (sized_gb, sized) =
        hostable_budget(ceiling_gb, |p| p.room_for(&bounds, floor).0, solve)?.or_else(|(plan, short)| {
            let (_, bound) = plan.room_for(&bounds, floor);
            let text = shortfall_text(&plan, &bounds, floor, short, bound, true);
            Err(anyhow::anyhow!("refused under {ceiling_gb:.1} GB budget: {text}"))
        })?;
    if sized_gb + 0.05 >= ceiling_gb {
        return Ok((sized, None));
    }
    let (room, bound) = plan.room_for(&bounds, floor);
    let (others, gpu) = (bounds.foreign as f64 / GB, bounds.foreign_gpu as f64 / GB);
    let mut said = if crate::config::params().source == crate::config::BudgetSource::Default {
        format!(
            "budget {sized_gb:.1} GB, what machine hosts ({:.1} GB paged; other processes {others:.1} GB, {gpu:.1} \
             GB GPU)",
            bounds.physical as f64 / GB
        )
    } else {
        format!("budget {ceiling_gb:.1} GB told, {sized_gb:.1} GB hosted - other processes hold {others:.1} GB ({gpu:.1} GB GPU)")
    };
    if !crate::cli::console::quiet() {
        said.push('\n');
        said.push_str(&shortfall_text(&plan, &bounds, floor, room.unsigned_abs(), bound, false));
    }
    Ok((sized, Some(said)))
}

/// The largest region beside the dense weights, host line and tables under every bound and `budget`, `margin` kept.
pub fn stage_one_region_bytes(stats: &ModelLayoutStats, bounds: &Bounds, budget: u64, margin: u64) -> Result<u64> {
    let dense = stats.non_expert_bytes;
    let host = crate::plan::host_line::host_line(stats, budget)
        .bytes()
        .saturating_add(crate::plan::host_line::pool_tables_bytes(stats, budget));
    let room = bounds.room(host, 0, 0, margin, 0).0.max(0) as u64;
    // Refused here only where nothing is left beside the weights: a width the region cannot hold ends the recording.
    let region = budget.saturating_sub(host).min(room).saturating_sub(dense);
    if region == 0 {
        anyhow::bail!(
            "launch REFUSED: machine leaves nothing beside {:.2} GB non-routed weights and {:.2} GB host line and \
             pools' tables",
            dense as f64 / GB,
            host as f64 / GB,
        );
    }
    Ok(region)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::KV_CHARGE_BENCH;
    use crate::machine::MemorySnapshot;
    use crate::plan::tests::routed;

    /// A plan is hosted or the binding bound is named; the hostable budget fits or names the smallest plan.
    #[test]
    fn a_plan_is_hosted_or_the_binding_bound_is_named_and_the_hostable_budget_fits_or_names_the_smallest() {
        let stats = routed(60, 128, 1 << 30);
        let plan = PlanInputs::live(54.0, KV_CHARGE_BENCH).solve(&stats).expect("plans");
        let gb = |g: f64| (g * GB) as u64;
        let ample = MemorySnapshot {
            physical: gb(64.0),
            wired: gb(4.0),
            gpu_ceiling: gb(58.0),
            machine_wire_limit: gb(60.0),
            task_wire_limit: gb(60.0),
            ..Default::default()
        };
        let floor = gb(1.5);
        assert!(hosted_by(&plan, &ample, floor).is_ok());
        let low_ceiling = MemorySnapshot { gpu_ceiling: gb(38.0), ..ample };
        let e = hosted_by(&plan, &low_ceiling, floor).expect_err("the ceiling refuses").to_string();
        assert!(e.contains("on GPU") && e.contains("GPU limit leaves") && e.contains("GB short"), "{e}");
        let tight_wire = MemorySnapshot { machine_wire_limit: gb(40.0), ..ample };
        let e = hosted_by(&plan, &tight_wire, floor).expect_err("the wire limit refuses").to_string();
        assert!(e.contains("machine wire limit leaves") && e.contains("GB short"), "{e}");
        let mut streamed = plan.clone();
        streamed.working.stream = 1 << 30;
        assert_eq!(streamed.locked_bytes(), plan.locked_bytes() + (1 << 30), "a stream's region is wired");
        // The hostable budget.
        let charge = KvCharge::for_interactive(6144, 2048, 0);
        let room = |m: &MemorySnapshot, p: &MemPlan| p.room_for(&m.bounds(), floor).0;
        let solve = |budget_gb: f64| PlanInputs::live(budget_gb, charge).solve(&stats).ok();
        let foreign = |held: f64| MemorySnapshot { physical: gb(64.0), wired: gb(held), ..Default::default() };
        // (the machine, sized below the ceiling, a bound on the plan's Metal figure): near-idle, 64 less 5 foreign, the
        // 1.5 floor and the 2.53 claimant's charge leave 54.97, the ceiling hosts; foreign demand holding 30 GB; a GPU
        // ceiling below the plan under an ample machine bound, the host share outside it; a foreign GPU client's 20 GB.
        for (machine, sized, metal) in [
            (foreign(5.0), false, None),
            (foreign(30.0), true, None),
            (MemorySnapshot { gpu_ceiling: gb(38.0), ..foreign(5.0) }, true, Some(38.0)),
            (MemorySnapshot { gpu_ceiling: gb(58.0), gpu_alloc: gb(20.0), ..foreign(6.0) }, true, Some(38.0)),
        ] {
            let (budget_gb, plan) =
                hostable_budget(54.0, |p| room(&machine, p), solve).expect("solves").expect("hosted");
            assert!(if sized { budget_gb < 54.0 } else { budget_gb == 54.0 }, "hosted at {budget_gb} GB");
            assert!(room(&machine, &plan) >= 0);
            assert!(metal.is_none_or(|m| plan.metal_bytes() as f64 <= m * GB), "the GPU bound holds the Metal figure");
        }
        // The sealed slack: a 35.5 GB peak from 54 down to 36, tracking the budget below.
        let sealed_slack = |budget_gb: f64| {
            (budget_gb >= 20.0).then(|| {
                let mut p = PlanInputs::live(54.0, charge).solve(&stats).expect("a plan to shape");
                let peak = if budget_gb >= 36.0 { 35.5 } else { budget_gb - 0.5 };
                p.mem_budget_bytes = gb(budget_gb);
                // The pool sized so the plan's total is exactly the peak.
                p.expert_pool_bytes = 0;
                p.expert_pool_bytes = gb(peak).saturating_sub(p.total_allocated_bytes());
                p
            })
        };
        // The machine hosts 0.95 GB less than the footprint beside the floor and the claimant's charge.
        let squeeze = MemorySnapshot { wired: gb(64.0) - gb(35.5 - 0.95) - floor - claimant_charge(), ..foreign(0.0) };
        let hosted = hostable_budget(54.0, |p| room(&squeeze, p), sealed_slack).expect("solves");
        let (budget_gb, plan) = hosted.expect("hosted, 20 GB of slack below the footprint");
        assert!(budget_gb < 36.0 && budget_gb > 34.0, "the carrier plan under the footprint, got {budget_gb}");
        assert!(room(&squeeze, &plan) >= 0);
        // A machine with 1 GB hostable: the smallest plan, named, and short.
        let hosted = hostable_budget(54.0, |p| room(&foreign(63.0), p), solve).expect("solves");
        let (plan, shortfall) = hosted.expect_err("a 1 GB machine hosted a plan");
        let budget_gb = plan.mem_budget_bytes as f64 / GB;
        assert!(budget_gb < 54.0 && shortfall > 0 && plan.hot_budget >= 1);
        assert!(solve(budget_gb - 0.01).is_none(), "the smallest plan: {budget_gb}");
    }
}
