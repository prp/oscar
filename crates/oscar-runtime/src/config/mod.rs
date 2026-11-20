//! The launch's configuration ([`params`]) and the config parameter set it resolves over ([`launch_params`]).

pub mod allocator;
pub mod drives;
pub mod params;

pub use params::{configure, params, EvictionChannels, Params};

use oscar_models::rows::ModelPreset;

use crate::machine::GB;

/// Inside the 50-400 µs range where the gain is flat; 1000 gives up a third of it.
const HEARTBEAT_US: u32 = 200;

/// Beneath every model's own row ([`ModelPreset::param_defaults`]).
const COMMON_DEFAULTS: &[(&str, &str)] = &[("OSCAR_ORACLE", "transition")];

/// Stated, not derived: the automatic map's sizing pass refuses a model whose weights deliberately do not fit.
pub fn device_mapping() -> mistralrs::DeviceMapSetting {
    mistralrs::DeviceMapSetting::dummy()
}

#[derive(Debug, Clone)]
pub struct ParamSetting {
    pub key: String,
    pub value: String,
    /// Set over a user's export (the plan's budget); otherwise the user's env wins.
    pub forced: bool,
}

/// A clean env's launch, first wins: the forced budget, the model's row, [`COMMON_DEFAULTS`], the heartbeat.
pub fn launch_params(preset: ModelPreset, budget_bytes: u64) -> Vec<ParamSetting> {
    let mut out: Vec<ParamSetting> = Vec::new();
    let mut push = |key: &str, value: String, forced: bool| {
        if !out.iter().any(|k| k.key == key) {
            out.push(ParamSetting { key: key.to_string(), value, forced });
        }
    };
    // The told budget is a ceiling: the load's tail solves the plan under it from the record (`load::stage_two`).
    push("OSCAR_MEM_TOTAL_GB", (budget_bytes as f64 / crate::machine::GB).to_string(), true);
    // A row's drafter and lookahead follow the live pool, idle on a sealed plan: prefetch hints are their only product.
    for (k, v) in preset.def().params.iter().chain(COMMON_DEFAULTS) {
        push(k, v.to_string(), false);
    }
    // On every row: a paged decode idles the GPU between command buffers, and the heartbeat holds its clock.
    push("OSCAR_GPU_HEARTBEAT_US", HEARTBEAT_US.to_string(), false);
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetSource {
    ExplicitEnv,
    CliMem,
    Default,
}

impl BudgetSource {
    pub fn label(self) -> &'static str {
        match self {
            BudgetSource::ExplicitEnv => "explicit env (OSCAR_MEM_TOTAL_GB)",
            BudgetSource::CliMem => "--mem",
            BudgetSource::Default => "default (what machine hosts)",
        }
    }
}

/// A ceiling, for bench, chat and serve alike: what the machine hosts of it is the plan's question.
pub fn told_budget_gb(mem_gb: Option<f64>) -> anyhow::Result<(f64, BudgetSource)> {
    if let Some(gb) = mem_gb {
        return Ok((gb, BudgetSource::CliMem));
    }
    let key = "OSCAR_MEM_TOTAL_GB";
    if let Some(raw) = std::env::var(key).ok().filter(|s| !s.is_empty()) {
        let gb = raw.parse().map_err(|e| anyhow::anyhow!("{key}={raw:?} failed to parse as GB: {e}"))?;
        return Ok((gb, BudgetSource::ExplicitEnv));
    }
    let (pageable, _) = crate::machine::memory_totals()
        .ok_or_else(|| anyhow::anyhow!("machine memory capacity undetectable and no budget given"))?;
    Ok((default_budget_gb(pageable as f64 / GB), BudgetSource::Default))
}

/// The watchdog's floor where no config parameter sets it.
pub const DEFAULT_WATCHDOG_FLOOR_GB: f64 = 1.5;

/// Without `--mem` the budget is what the machine can host: everything it pages less the floor and the claimant's
/// charge the planner keeps beside any plan; the plan is then sized to what remains, as for a told budget.
pub fn default_budget_gb(pageable_gb: f64) -> f64 {
    (pageable_gb - DEFAULT_WATCHDOG_FLOOR_GB - crate::machine::claimant_charge() as f64 / GB).max(0.0)
}

/// The margin every bound keeps for the OS.
pub fn watchdog_floor_bytes() -> u64 {
    (params().misc.mem_watchdog_gb * GB) as u64
}

/// One guaranteed sequence (X in + Y out) and headroom; retained turns are cache in the slack, as eviction offloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCharge {
    /// X as asked: the enforced guarantee, which the census records every width at ([`crate::plan::MemPlan`]).
    pub max_input_tokens: u64,
    /// Y, at a full-X prompt: only the total is enforced (`max_seq_len`), so a short prompt may generate longer.
    pub max_output_tokens: u64,
    /// Kept from the scheduler's allocator (`plan::arm_draft_reserve`), so drafts never displace a sequence's blocks.
    pub draft_reserve_tokens: u64,
    /// Its stream region and hand-off state, the launch's for good: no crossing books or releases them.
    pub drafter_booked: bool,
    /// Provenance, not arithmetic: an explicit charge at the bench point's figures is not the bench charge.
    pub bench: bool,
    /// Given on the command line: refused where it does not fit, never lowered as a default is.
    pub asked: bool,
}

/// A block of partial-block re-base, the pending window and a top-up's drafts, with margin: four blocks of 32.
pub const DRAFT_KV_RESERVE_TOKENS: u64 = 128;

fn draft_reserve(armed: bool) -> u64 {
    u64::from(armed) * DRAFT_KV_RESERVE_TOKENS
}

/// The bench point the banked rows are priced at.
pub const KV_CHARGE_BENCH: KvCharge = KvCharge {
    max_input_tokens: 4096,
    max_output_tokens: 0,
    draft_reserve_tokens: 0,
    drafter_booked: false,
    bench: true,
    asked: false,
};

impl KvCharge {
    pub fn total_tokens(&self) -> u64 {
        self.max_input_tokens.saturating_add(self.max_output_tokens)
    }

    /// The window alone: a lowering changes it, an unbooking keeps it.
    pub fn window(&self) -> (u64, u64) {
        (self.max_input_tokens, self.max_output_tokens)
    }

    /// The drafter's reserve and hand-off released: a small pool's charge.
    pub fn unbooked(self) -> Self {
        Self { draft_reserve_tokens: 0, drafter_booked: false, ..self }
    }

    /// A chunk at the plan's `width` (the whole prompt before a record), so cache slack is never reclaimed mid-chunk.
    pub fn headroom(&self, width: Option<usize>) -> u64 {
        if self.bench {
            return 0;
        }
        width.map_or(self.max_input_tokens, |w| (w as u64).min(self.max_input_tokens))
    }

    /// A context as the frontier splits it, three quarters in and one out: feasibility, not a tuned default.
    pub fn of_total(tokens: u64) -> Self {
        Self::for_interactive(tokens / 4 * 3, tokens / 4, draft_reserve(drafter_on()))
    }

    /// The bench point where `xy` is `None`, an armed drafter's reserve beside either.
    pub fn of_launch(xy: Option<(u64, u64, bool)>, draft: bool) -> Self {
        match xy {
            Some((x, y, asked)) => KvCharge { asked, ..Self::for_interactive(x, y, draft_reserve(draft)) },
            None => KvCharge { draft_reserve_tokens: draft_reserve(draft), drafter_booked: draft, ..KV_CHARGE_BENCH },
        }
    }

    pub fn for_interactive(max_input_tokens: u64, max_output_tokens: u64, draft_reserve_tokens: u64) -> Self {
        Self {
            max_input_tokens,
            max_output_tokens,
            draft_reserve_tokens,
            drafter_booked: draft_reserve_tokens > 0,
            bench: false,
            asked: false,
        }
    }

    /// The next smaller default window where this one fits neither the budget nor the machine: both halved, the output
    /// never under 4096 tokens, the input never under 16384 (an opencode session's first request is about 10k); an
    /// asked window or the bench's has none.
    pub fn lowered(self) -> Option<Self> {
        if self.asked || self.bench || self.max_input_tokens / 2 < 16384 {
            return None;
        }
        Some(Self {
            max_input_tokens: self.max_input_tokens / 2,
            max_output_tokens: (self.max_output_tokens / 2).max(4096),
            ..self
        })
    }
}

/// What the first stage settles: the charge (a default window lowered to fit, the drafter unbooked under a small
/// pool) and whether the pool is small, under two routed sets a layer: the drafter off, the lookahead's staging kept.
static LAUNCH: std::sync::OnceLock<(KvCharge, bool)> = std::sync::OnceLock::new();

pub fn set_launch(charge: KvCharge, small_pool: bool) {
    let _ = LAUNCH.set((charge, small_pool));
}

/// What every reader of the window takes.
pub fn charge() -> KvCharge {
    LAUNCH.get().map_or_else(|| params().charge, |l| l.0)
}

pub fn small_pool() -> bool {
    LAUNCH.get().is_some_and(|l| l.1)
}

pub fn drafter_on() -> bool {
    params().draft && !small_pool()
}

/// A bench's sequence from `OSCAR_PLAN_KV`: unset is the bench point, a malformed value refused.
pub fn plan_kv_from(raw: Option<String>) -> Option<(u64, u64)> {
    let raw = raw?;
    let xy = raw.split_once('+').and_then(|(x, y)| Some((x.parse().ok()?, y.parse().ok()?)));
    Some(xy.unwrap_or_else(|| panic!("OSCAR_PLAN_KV={raw:?}: expected <max_input_tokens>+<max_output_tokens>")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A default window lowers by halves to the floor, an asked one never; a small pool's charge keeps its window and
    /// releases the drafter; the default budget is the machine less the planner's margins.
    #[test]
    fn a_default_window_lowers_by_halves_an_asked_one_never_and_a_small_pools_charge_releases_the_drafter() {
        let steps: Vec<(u64, u64)> =
            std::iter::successors(Some(KvCharge::for_interactive(131072, 16384, 0)), |c| c.lowered())
                .map(|c| (c.max_input_tokens, c.max_output_tokens))
                .collect();
        assert_eq!(steps, [(131072, 16384), (65536, 8192), (32768, 4096), (16384, 4096)]);
        assert!(KvCharge { asked: true, ..KvCharge::for_interactive(131072, 16384, 0) }.lowered().is_none());
        assert!(KV_CHARGE_BENCH.lowered().is_none());
        assert_eq!(KvCharge::of_launch(Some((8192, 1024, true)), false).asked, true);
        let c = KvCharge::for_interactive(65536, 16384, DRAFT_KV_RESERVE_TOKENS);
        assert!(c.drafter_booked && c.draft_reserve_tokens > 0);
        let u = c.unbooked();
        assert_eq!((u.window(), u.drafter_booked, u.draft_reserve_tokens), (c.window(), false, 0));
        assert_ne!(u, c, "the unbooked charge is installed as a lowered one");
        assert_eq!(u.lowered().map(|l| l.window()), Some((32768, 8192)));
        let margins = DEFAULT_WATCHDOG_FLOOR_GB + crate::machine::claimant_charge() as f64 / GB;
        assert!((default_budget_gb(62.27) - (62.27 - margins)).abs() < 1e-9);
        assert_eq!(default_budget_gb(1.0), 0.0, "never negative");
    }
}
