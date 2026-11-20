//! `check-model`: whether a model's files load and plan, from their GGUF headers alone, with no tensor read.

use anyhow::{Context, Result};
use clap::Parser;

use crate::plan;

use super::args::{ModelPreset, Target};

#[derive(Debug, Parser)]
pub struct CheckModelArgs {
    #[command(flatten)]
    pub target: Target,

    /// Plan at this total sequence (3/4 input, 1/4 output) rather than the bench charge.
    #[arg(long, value_name = "TOKENS")]
    pub ctx: Option<u64>,
}

/// A check's line: what held, or what failed.
type Verdict = Result<String, String>;

fn line(name: &str, v: &Verdict) -> String {
    let (mark, body) = match v {
        Ok(s) => ("ok  ", s),
        Err(s) => ("FAIL", s),
    };
    format!("  [{mark}] {name:<22} {body}")
}

pub fn check_model_main(args: CheckModelArgs) -> Result<()> {
    let preset = args.target.model;
    let mut verdicts: Vec<(&str, Verdict)> = Vec::new();

    // The one check that must hold before a download, so it runs with no shards.
    verdicts.push((
        "download source",
        match crate::model_store::conversion::validate(&preset.def().conversion) {
            Ok(()) => Ok(format!("{} -> {}", preset.def().conversion.source_repo, preset.def().production_variant,)),
            Err(e) => Err(format!("{e:#}")),
        },
    ));

    // Every shard the model names must exist; a missing variant or file fails this row only.
    let located =
        crate::model_store::locate(preset, args.target.quant.as_deref(), args.target.gguf_dir.as_deref(), &[]);
    let (gguf_dir, paths) = match located {
        Ok((dir, _, paths)) => (dir, paths),
        Err(e) => {
            verdicts.push(("shards", Err(format!("{e:#}"))));
            return report(preset, verdicts, None);
        }
    };
    verdicts.push(("shards", Ok(format!("{} file(s) under {}", paths.len(), gguf_dir.display()))));

    // Under the launch's configuration; a missing tensor, one in two shards or metadata that is not MoE fails.
    let budget = crate::config::told_budget_gb(args.target.mem)?;
    crate::config::configure(crate::config::Params::for_launch(preset, budget, None, None));
    let (stats, l) = match plan::read_model_layout_stats(&paths) {
        Ok(read) => read,
        Err(e) => {
            verdicts.push(("routed layout", Err(format!("{e:#}"))));
            return report(preset, verdicts, None);
        }
    };
    verdicts.push((
        "routed layout",
        Ok(format!(
            "{} MoE layers x {} experts, top-{}; MTP {}",
            l.block_count,
            l.num_experts,
            l.experts_per_token.map_or_else(|| "?".into(), |k| k.to_string()),
            match l.nextn_blocks {
                0 => "absent".to_string(),
                n => format!("{n} block(s), skipped"),
            },
        )),
    ));

    // A quantisation with no kernel fails at the first token; fused gate and up need a gate+up instantiation.
    let mut unsupported: std::collections::BTreeSet<String> = Default::default();
    let mut seen: std::collections::BTreeSet<String> = Default::default();
    for ((_, kind), proj) in &l.projections {
        seen.insert(format!("{:?}", proj.dtype));
        let gate_or_up = !matches!(kind, crate::gpu::ProjectionKind::Down);
        if !plan::bank_runs(proj.dtype, gate_or_up, l.activation) {
            unsupported.insert(format!("{:?} ({kind:?})", proj.dtype));
        }
    }
    let dtypes: Vec<String> = seen.into_iter().collect();
    verdicts.push((
        "expert dtypes",
        if unsupported.is_empty() {
            Ok(format!("{} at {:?}", dtypes.join(", "), l.activation))
        } else {
            Err(format!("no fused gate+up kernel for {}", unsupported.into_iter().collect::<Vec<_>>().join(", ")))
        },
    ));

    // The reader refuses a value it cannot interpret rather than charging zero.
    verdicts.push((
        "plan inputs",
        Ok(format!(
            "{:.1} GB non-routed, {:.1} MB/expert-unit, {} B/token KV",
            stats.non_expert_bytes as f64 / crate::machine::GB,
            stats.bytes_per_hb_unit as f64 / (1024.0 * 1024.0),
            stats.kv_bytes_per_token,
        )),
    ));

    // A plan that cannot be made is a model that will not start.
    let mut block: Option<String> = None;
    let charge = args.ctx.map_or(crate::config::KV_CHARGE_BENCH, crate::config::KvCharge::of_total);
    match crate::load::plan_before_load(&paths, &stats, charge).context("resolving a memory plan") {
        Ok((plan, budget_line)) => {
            let source = crate::config::params().source;
            // A limit that sized the plan down is this machine's fact, not a verdict.
            if let Some(line) = budget_line {
                verdicts.push(("memory budget", Ok(line.replace('\n', "; "))));
            }
            verdicts.push((
                "memory plan",
                Ok(format!(
                    "{:.1} GB total, {} carrier(s) of {} at {} slots + {} sealed, offloading={} ({})",
                    plan.total_allocated_bytes() as f64 / crate::machine::GB,
                    plan.carrier_layers,
                    plan.block_count,
                    plan.hot_budget,
                    plan.block_count - plan.carrier_layers,
                    plan.offloading,
                    source.label(),
                )),
            ));
            // The plan line by line: a total cannot say where the memory went.
            block = Some(plan::render_mem_plan(&plan));
        }
        Err(e) => verdicts.push(("memory plan", Err(format!("{e:#}")))),
    }

    report(preset, verdicts, block)
}

fn report(preset: ModelPreset, verdicts: Vec<(&str, Verdict)>, block: Option<String>) -> Result<()> {
    println!("check-model {preset:?}");
    for (name, v) in &verdicts {
        println!("{}", line(name, v));
    }
    if let Some(b) = block {
        println!("  {b}");
    }
    let failed = verdicts.iter().filter(|(_, v)| matches!(v, Err(_))).count();
    if failed > 0 {
        anyhow::bail!("{failed} check(s) failed");
    }
    println!("  all checks passed: the files load and plan; the outputs are the gate suite's question.");
    Ok(())
}
