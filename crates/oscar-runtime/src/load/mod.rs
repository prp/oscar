//! Stage one admits the told budget, sizing nothing from a plan; after the census, [`stage_two`] solves the plan once.

pub mod census;
pub mod report;

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use mistralrs_core::set_global_sample_trace;
use oscar_models::moe::set_global_moe_dispatcher;

use crate::config::KvCharge;
use crate::engine::dispatcher::OffloadDispatcher;
use crate::machine::{Bounds, GB};
use crate::plan::{self, MemPlan, ModelLayoutStats};

/// Stage one for every MoE launch, `None` for any other; an evaluation's hooks, installed before, may drive its plane.
pub fn install_offload_dispatcher_from_env(gguf_paths: &[PathBuf]) -> Option<Arc<OffloadDispatcher>> {
    // Oscar's own GGUF architectures, registered before a model loads.
    oscar_models::register();
    crate::drafter::sampler::register();
    // The memory phase is the engine's step kind, announced before a step's inputs are made.
    mistralrs_core::paged_attention::set_step_kind_hook(Box::new(|step| {
        let moved =
            crate::engine::offload_dispatcher().map_or(Ok(()), |pd| pd.phase_change(step, crate::drafter::quiesce));
        oscar_offload::machine::begin_step(step.prompt);
        moved
    }));
    // Every Metal buffer's pools and the commit counter, live on every launch before the model makes the device.
    oscar_offload::pools::install();
    crate::gpu::state_pool::install();
    crate::gpu::probe::install(crate::engine::eval::hooks().gpu_observer());
    crate::memory::end_the_process_on_a_panic();
    // The launch's configuration; a bench's is parsed from its environment here.
    let _ = crate::config::params();
    // The model's file, read once; one declaring experts whose routed layout cannot be read refuses the load.
    let model = match plan::read_model_layout_stats(gguf_paths) {
        Ok(model) => Some(model),
        Err(e) if plan::read_declaration(gguf_paths).is_ok_and(|d| d.experts.count > 0) => {
            panic!("routed layout unreadable: {e:#}")
        }
        Err(_) => None,
    };
    // The row-gather store, where the artifact carries one: one it cannot read refuses the load.
    if let Some(t) = model.as_ref().and_then(|(s, _)| s.ple_table.as_ref()) {
        if let Err(e) = crate::plan::install_ple_store(gguf_paths, t) {
            panic!("PLE row-gather store install failed ({e})");
        }
    }
    crate::config::params::reject_unknown_oscar_env();
    // One dispatcher for every MoE launch: residency is the plan's property (a resident fit seals), never a mode.
    let Some((stats, layout)) = model.filter(|(s, _)| s.block_count > 0 && s.num_experts > 0) else {
        crate::gpu::heartbeat::start();
        return None;
    };
    // Stage one: the told budget against the machine as read once a process.
    let (budget, machine, k) = first_stage(gguf_paths, &stats, crate::config::params().charge)
        .unwrap_or_else(|e| crate::memory::refuse(format_args!("{e}")));
    crate::memory::start_sensors(budget);
    // Beside the dense weights, mapped, not wired, for the census to climb under; the tail re-takes it at the record.
    let largest =
        objc2_metal::MTLCreateSystemDefaultDevice().map_or(usize::MAX, |d| oscar_offload::region::largest_bytes(&d));
    let bytes = plan::stage_one_region_bytes(&stats, &machine, budget, crate::config::watchdog_floor_bytes())
        .unwrap_or_else(|e| crate::memory::refuse(format_args!("{e}")));
    let bytes = (bytes as usize).min(largest);
    crate::detail!(target: "oscar::loader",
        "stage one - working region up to {:.2} GB beside {:.2} GB non-routed weights, mapped as census touches it",
        bytes as f64 / GB,
        stats.non_expert_bytes as f64 / GB,
    );
    if let Some(pools) = oscar_offload::pools::pools() {
        pools.plan_census_region(bytes);
    }
    let _ = STAGE_ONE.set(StageOne { stats, machine: Some(machine), budget });
    // Fixed before the pool is built, from the plan at the told budget: the widest the load knows before the record.
    let stage = STAGE_ONE.get().expect("just set");
    crate::detail!(target: "oscar::experts",
        "segments of {k} slots/layer/bank, from {:.2} GB plan (every bank within {} segments)",
        budget as f64 / GB,
        plan::DECLARED_SEGMENTS_PER_BANK,
    );
    let pd = Arc::new(OffloadDispatcher::new(gguf_paths, layout, k).expect("OffloadDispatcher::new"));
    let _ = crate::engine::PAGING_DISPATCHER.set(pd.clone());
    set_global_moe_dispatcher(pd.clone());
    // The bar's progress through a prompt chunk is routed layers run over this.
    mistralrs_core::live::set_prompt_layers(stage.stats.block_count as u64);
    // The census runs inside the load: the one window with the dense weights placed and no elastic pool sized.
    mistralrs_core::paged_attention::set_census_hook(Box::new(crate::load::census::run_at_load));
    set_global_sample_trace(crate::engine::eval::hooks().clone());
    Some(pd)
}

/// For the tools that make no load (`print-config`, `check-model`): a launch's first-stage admission, the plan at the
/// told budget sized to what the machine hosts and the line saying a limit sized it.
pub fn plan_before_load(
    gguf_paths: &[PathBuf], stats: &ModelLayoutStats, charge: KvCharge,
) -> anyhow::Result<(MemPlan, Option<String>)> {
    let (_, bounds, _) = first_stage(gguf_paths, stats, charge)?;
    let at = plan::PlanInputs::live(crate::config::params().budget_gb, crate::config::charge());
    plan::solve_under_limits(stats, at, Some(bounds))
}

/// The model said, the machine read once a process, the window and the segment fixed: the budget, its bounds, the
/// segment's slots. A default window that fits neither is lowered by halves to the floor, an asked one refused as
/// asked; under two routed sets a layer the drafter is off, its memory free for slots; each change said as a warning.
fn first_stage(
    gguf_paths: &[PathBuf], stats: &ModelLayoutStats, charge: KvCharge,
) -> anyhow::Result<(u64, Bounds, usize)> {
    report::say_the_model(gguf_paths, stats);
    let budget_gb = crate::config::params().budget_gb;
    let budget = (budget_gb * GB) as u64;
    let two_sets = 2 * stats.experts_per_token;
    let (mut bounds, mut tried, mut small) = (None::<Bounds>, charge, None::<usize>);
    loop {
        let at = plan::PlanInputs::at(budget_gb, tried, None, None);
        let hosted = at.solve(stats).and_then(|ceiling| {
            let b = match bounds {
                Some(b) => b,
                None => *bounds.insert(crate::memory::first_stage_bounds(budget)?),
            };
            plan::solve_under_limits(stats, at, Some(b)).map(|(plan, _)| (b, ceiling, plan))
        });
        match hosted {
            Ok((b, ceiling, plan)) => {
                if plan.offloading && plan.hot_budget < two_sets && small.is_none() {
                    (small, tried) = (Some(plan.hot_budget), tried.unbooked());
                    continue;
                }
                if tried.window() != charge.window() {
                    tracing::warn!(target: "oscar::planner",
                        "The context window was reduced: {}+{} does not fit; lowered to {}+{}",
                        charge.max_input_tokens,
                        charge.max_output_tokens,
                        tried.max_input_tokens,
                        tried.max_output_tokens,
                    );
                }
                if let Some(slots) = small {
                    tracing::warn!(target: "oscar::planner",
                        "drafter off: the expert pool's {slots} slots/layer are under two routed sets ({two_sets}); {} \
                         slots/layer without it",
                        plan.hot_budget,
                    );
                }
                crate::config::set_launch(tried, small.is_some());
                return Ok((budget, b, plan::install_segment_slots(stats, &ceiling)));
            }
            Err(e) => match tried.lowered() {
                Some(next) => tried = next,
                None if tried != charge => {
                    return Err(anyhow::anyhow!(
                        "{e} (context window lowered to {}+{}, its floor)",
                        tried.max_input_tokens,
                        tried.max_output_tokens
                    ))
                }
                None => return Err(e),
            },
        }
    }
}

struct StageOne {
    stats: ModelLayoutStats,
    machine: Option<crate::machine::Bounds>,
    budget: u64,
}

static STAGE_ONE: OnceLock<StageOne> = OnceLock::new();

pub fn stage_stats() -> Option<&'static ModelLayoutStats> {
    STAGE_ONE.get().map(|s| &s.stats)
}

/// The plan solved once from the record; the KV block pool reads it as the engine builds it, the rest sized here.
pub fn stage_two() {
    let Some(stage) = STAGE_ONE.get() else {
        return;
    };
    // The chunk and the region first, from the record alone, so the region is re-taken before the plan is solved.
    let at =
        |r| plan::PlanInputs::at(stage.budget as f64 / GB, crate::config::charge(), Some(r), plan::segment_slots());
    let solve = |r| plan::solve_under_limits(&stage.stats, at(r), stage.machine).ok().map(|(p, _)| p);
    let _ = plan::WIDEST
        .set(crate::load::census::prefill_regions().and_then(|r| plan::widest_width(&stage.stats, &r, solve)));
    crate::plan::chunks::install();
    // The plan at the told budget under the machine as the load found it, with the line naming the limit that held it.
    let at = plan::PlanInputs::live(crate::config::params().budget_gb, crate::config::charge());
    let (plan, budget_line) = plan::solve_under_limits(&stage.stats, at, stage.machine)
        .unwrap_or_else(|e| crate::memory::refuse(format_args!("{e}")));
    oscar_offload::machine::PLAN.arm(plan.mem_budget_bytes);
    mistralrs_core::paged_attention::set_paged_snapshot_cap(plan.paged_snapshot_entries as usize);
    // The guarantee the mistral.rs fork enforces: prompts past X refused, `max_seq_len` capped at X+Y; the bench none.
    if !plan.kv_charge.bench {
        let (x, y) = plan.guarantee();
        mistralrs_core::paged_attention::set_plan_ctx_limit(x as usize, y as usize);
    }
    crate::memory::crossing::install(&stage.stats, &plan, stage.budget);
    let ceilings = plan::PoolCeilings::solve(&stage.stats, &plan, stage.budget as f64 / GB);
    plan::install_pool_ceilings(ceilings);
    if let Some(pool) = crate::gpu::state_pool::installed() {
        if let Err(e) = pool.launch(plan.sequence_state_slots as usize) {
            panic!("sequence-state pool launch at plan's count failed: {e}");
        }
    }
    if plan.store_staging_bytes > 0 {
        crate::kv_store::set_staging_bytes(plan.store_staging_bytes);
    }
    crate::detail!(target: "oscar::planner",
        "ceilings at told {:.2} GB - expert pool {} slots/layer, KV index {} blocks",
        stage.budget as f64 / GB,
        ceilings.hot_budget,
        ceilings.kv_index_blocks,
    );
    // The launch's report, said once, where the budget is decided.
    crate::load::report::launch_report(stage.budget, &plan, budget_line.as_deref());
    // The rings are sized once, at the most slots per layer any plan may give.
    if let Some(pd) = crate::engine::offload_dispatcher() {
        if !pd.launched() {
            // The live plan, as the crossing's install may have re-landed it.
            let live = crate::memory::crossing::live().expect("the crossing is installed above");
            if let Err(e) = crate::engine::eval::hooks().launch_pool(
                &pd,
                plan.hot_budget,
                plan.carrier_layers,
                ceilings.hot_budget,
                live.offloading,
            ) {
                panic!("expert pool launch at plan's shape failed: {e}");
            }
            let decode = live.decode.map(|d| (d.hot_budget, d.lent_bytes));
            pd.lend_at_load((live.hot_budget, live.carrier_layers), live.decode_bound, decode);
        }
        match candle_core::Device::new_metal(0) {
            Ok(candle_core::Device::Metal(device)) => {
                if let Err(e) =
                    pd.build_paged_lanes(&device, stage.stats.experts_per_token, stage.stats.hidden_dim as usize)
                {
                    panic!("paged lanes build at load failed: {e}");
                }
            }
            other => panic!("paged lanes need Metal device at load: {other:?}"),
        }
    }
    crate::gpu::heartbeat::start();
    // The warm-up, one forward through the pools as launched, is the last thing before the first prompt.
    tracing::info!(target: "oscar::loader", "warm-up forward ...");
}
