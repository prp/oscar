//! `print-config`: a model's launch config parameters as `KEY=VALUE` lines, for a caller to run at `chat`'s point.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use crate::config::launch_params;
use crate::load::plan_before_load;

use super::args::{parse_mem_gb, Target};

#[derive(Debug, Parser)]
pub struct PrintConfigArgs {
    #[command(flatten)]
    pub target: Target,

    /// Machine memory capacity the planner assumes, as for `chat`.
    #[arg(long, value_name = "GB", value_parser = parse_mem_gb)]
    pub physical_mem: Option<f64>,

    /// A second drive holding a replica of the model, as for `chat`.
    #[arg(long, value_name = "[DRIVE:]DIR")]
    pub mirror: Option<String>,

    /// Also emit the model's weights, tokenizer and chat template as
    /// `OSCAR_BENCH_*` assignments, so a caller runs the template `chat` uses.
    #[arg(long)]
    pub paths: bool,

    /// Emit `export KEY=VALUE` instead of bare `KEY=VALUE`.
    #[arg(long)]
    pub export: bool,

    /// Print what each X+Y costs in pool slots instead; requires --mem.
    #[arg(long)]
    pub frontier: bool,
}

/// The config parameters to stdout and their provenance to stderr, so `eval "$(... print-config ...)"` stays clean.
pub fn print_config_main(args: PrintConfigArgs) -> Result<()> {
    let Target { model: preset, quant, gguf_dir: dir, mem } = args.target;
    let (gguf_dir, gguf_files, gguf_paths) = crate::model_store::locate(preset, quant.as_deref(), dir.as_deref(), &[])?;

    crate::machine::assume_physical(args.physical_mem).map_err(anyhow::Error::msg)?;
    // The launch's point: an `OSCAR_MEM_TOTAL_GB` left in the caller's environment must not move it.
    let budget = crate::config::told_budget_gb(mem)?;
    crate::config::configure(crate::config::Params::for_launch(preset, budget, None, None));
    if args.frontier {
        return print_frontier(&gguf_paths, mem);
    }
    crate::config::drives::install(args.mirror.as_deref())?;
    // The bench charge, which the banked rows are priced at; an interactive entry's guarantee may run a smaller pool.
    let (stats, _) = crate::plan::read_model_layout_stats(&gguf_paths)
        .map_err(|e| anyhow::anyhow!("no memory plan for {gguf_dir:?}: configuration unresolved ({e})"))?;
    let (plan, budget_line) = plan_before_load(&gguf_paths, &stats, crate::config::KV_CHARGE_BENCH)?;
    // No load follows, so the budget is decided and said here.
    for line in budget_line.iter().flat_map(|l| l.lines()) {
        tracing::info!(target: "oscar::planner", "{line}");
    }
    let source = crate::config::params().source;
    let params = launch_params(preset, (crate::config::params().budget_gb * crate::machine::GB) as u64);
    let states = plan.sequence_state_bytes as f64 / crate::machine::GB;
    tracing::info!(
        target: "oscar::planner",
        "{:?} offloading={} row={} sequence_states={states:.3}GB \
         hot_budget={} segment_slots={} budget_source={} params={}",
        preset,
        plan.offloading,
        if plan.offloading { "with-offloading" } else { "without-offloading" },
        plan.hot_budget,
        plan.segment_slots.unwrap_or(0),
        source.label(),
        params.len(),
    );
    let emit = |key: &str, value: &str| {
        if args.export {
            println!("export {key}={value}");
        } else {
            println!("{key}={value}");
        }
    };
    for k in &params {
        emit(&k.key, &k.value);
    }
    if args.paths {
        emit("OSCAR_BENCH_GGUF_DIR", &gguf_dir.display().to_string());
        emit("OSCAR_BENCH_GGUF_FILE", &gguf_files.join(","));
        emit("OSCAR_BENCH_TOK_REPO", preset.def().tok_repo);
        emit("OSCAR_BENCH_JINJA", &preset.materialise_chat_template()?);
    }
    Ok(())
}

/// A row per candidate X+Y at the told budget: the KV it charges and the expert pool left.
fn print_frontier(gguf_paths: &[PathBuf], mem_gb: Option<f64>) -> Result<()> {
    use crate::config::KvCharge;
    use crate::machine::GB;
    use crate::plan;

    let Some(mem_gb) = mem_gb else {
        anyhow::bail!("--frontier needs --mem: pool unconstrained without budget");
    };
    let (stats, _) = plan::read_model_layout_stats(gguf_paths)
        .map_err(|e| anyhow::anyhow!("frontier needs GGUF header stats: {e}"))?;
    // The segment size at the told budget, as a launch fixes it for the host line; each row's pool takes its own.
    let charge = crate::config::params().charge;
    plan::install_segment_slots(&stats, &plan::PlanInputs::at(mem_gb, charge, None, None).solve(&stats)?);
    let machine = crate::machine::MemorySnapshot::read().ok();
    let floor = crate::config::watchdog_floor_bytes();
    println!(
        "# guaranteed sequence frontier at --mem {mem_gb:.0} (asked as X = 3/4 of total, Y = 1/4; \
         the guarantee the asked pair, a resident model's pool the remainder)"
    );
    println!("{:>8} {:>8} {:>8} {:>10} {:>12}  note", "X+Y", "X", "Y", "kv GB", "pool slots");
    for total in [4096u64, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304] {
        let charge = KvCharge::of_total(total);
        let (x, y) = (charge.max_input_tokens, charge.max_output_tokens);
        let row = match plan::plan_at_charge(&stats, mem_gb, charge) {
            Ok(plan) => {
                // The machine's refusal, said per row rather than bailing.
                let note = match machine.as_ref().map(|m| plan::hosted_by(&plan, m, floor)) {
                    Some(Err(e)) => format!("REFUSED on this machine: {}", e.to_string().lines().next().unwrap_or("")),
                    _ if !plan.offloading => "the remainder".to_string(),
                    _ => String::new(),
                };
                let (x, y) = plan.guarantee();
                format!(
                    "{:>8} {:>8} {:>8} {:>10.2} {:>12}  {}",
                    x + y,
                    x,
                    y,
                    plan.kv_cache_bytes as f64 / GB,
                    plan.hot_budget,
                    note
                )
            }
            Err(e) => format!(
                "{:>8} {:>8} {:>8} {:>10} {:>12}  REFUSED: {}",
                total,
                x,
                y,
                "-",
                "-",
                e.to_string().lines().next().unwrap_or("")
            ),
        };
        println!("{row}");
    }
    Ok(())
}
