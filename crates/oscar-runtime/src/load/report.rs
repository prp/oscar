//! The launch's lines: the model's name, its first line, and the report at the load's tail from the plan that runs.

use std::path::PathBuf;

use crate::plan::{self, MemPlan};

/// `<model>-GGUF/<variant>/` gives `<model>-<quant>`, the variant less its lowercase prefix; else the shard's stem.
pub fn model_name(gguf_paths: &[PathBuf]) -> String {
    fn named(d: Option<&std::path::Path>) -> Option<&str> {
        d.and_then(|d| d.file_name()).and_then(|n| n.to_str())
    }
    let variant_dir = gguf_paths.first().and_then(|p| p.parent());
    let model = named(variant_dir.and_then(|d| d.parent())).and_then(|n| n.strip_suffix("-GGUF"));
    if let (Some(model), Some(variant)) = (model, named(variant_dir)) {
        let quant = variant.trim_start_matches(|c: char| c.is_ascii_lowercase());
        return format!("{model}-{}", if quant.is_empty() { variant } else { quant });
    }
    let files: Vec<String> =
        gguf_paths.iter().filter_map(|p| p.file_name()).map(|f| f.to_string_lossy().into_owned()).collect();
    let stem = files
        .iter()
        .find(|f| f.contains("-00001-of-"))
        .or(files.first())
        .map(|f| f.trim_end_matches(".gguf"))
        .unwrap_or("");
    stem.find("-00001-of-").map_or(stem, |i| &stem[..i]).to_string()
}

/// A launch's first line, said at its first stage, or by a tool from its plan; each once.
pub fn say_the_model(gguf_paths: &[PathBuf], stats: &plan::ModelLayoutStats) {
    let shards = gguf_paths.len();
    let on_disk: u64 = gguf_paths.iter().filter_map(|p| std::fs::metadata(p).ok()).map(|m| m.len()).sum();
    tracing::info!(target: "oscar::loader",
        "{} - {:.0}B parameters, {} layers, {} experts/layer, top-{}{}{}, context {} tokens; {shards} shard{}, \
         {:.1} GB on disk",
        model_name(gguf_paths),
        stats.total_params as f64 / 1e9,
        stats.block_count,
        stats.num_experts,
        stats.experts_per_token,
        if stats.recurrent_layers > 0 { format!(", {} recurrent layers", stats.recurrent_layers) } else { String::new() },
        if stats.mla_absorbed { ", absorbed MLA attention" } else { "" },
        stats.context_length,
        if shards == 1 { "" } else { "s" },
        on_disk as f64 / crate::machine::GB,
    );
}

/// Once at the load's tail, from the plan that runs.
pub fn launch_report(told: u64, plan: &MemPlan, budget_line: Option<&str>) {
    let gb = |b: u64| b as f64 / crate::machine::GB;
    let kn = crate::config::params();
    let drives = match crate::config::drives::installed() {
        Some(m) => format!("internal-nvme x{} + {} x{}", kn.misc.io_threads, m.drive.name, m.drive.readers),
        None => format!("internal-nvme x{}", kn.misc.io_threads),
    };
    let physical = crate::machine::memory_totals().map_or(String::new(), |(pageable, installed)| {
        let word = if crate::machine::physical_is_assumed() { "assumed" } else { "pageable" };
        format!(" of {:.1} GB {word} ({:.0} GB installed)", gb(pageable), gb(installed))
    });
    // Without --mem the budget is what the machine hosts, so the plan's own figure is the one to say.
    let said = if kn.source == crate::config::BudgetSource::Default { plan.mem_budget_bytes } else { told };
    tracing::info!(target: "oscar::planner",
        "budget {:.1} GB{physical}; {}; experts read by {drives}",
        gb(said),
        if plan.offloading { "experts offloaded to NVMe" } else { "every expert resident" },
    );
    for line in budget_line.into_iter().flat_map(str::lines) {
        tracing::info!(target: "oscar::planner", "{line}");
    }
    tracing::info!(target: "oscar::planner",
        "ceiling {:.2} GB: plan is upper bound no allocation exceeds; claim past it fails its request",
        gb(plan.mem_budget_bytes)
    );
    if kn.misc.mem_watchdog_gb > 0.0 {
        tracing::info!(target: "oscar::watchdog",
            "backstop: watchdog stops oscar if free memory stays under {:.1} GB",
            kn.misc.mem_watchdog_gb
        );
    }
    tracing::info!(target: "oscar::planner", "memory plan -\n{}", plan::render_mem_plan(plan).trim_end());
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::ValueEnum;
    use oscar_models::rows::ModelPreset;

    #[test]
    fn every_model_is_named_by_its_quant_after_a_dash() {
        let named = |paths: &[&str]| model_name(&paths.iter().map(PathBuf::from).collect::<Vec<_>>());
        let want = [
            (ModelPreset::Qwen35B, "Qwen3.6-35B-A3B-Q4"),
            (ModelPreset::Qwen235B, "Qwen3-235B-A22B-Q4im"),
            (ModelPreset::Qwen397B, "Qwen3.5-397B-A17B-Q3"),
            (ModelPreset::Glm52, "GLM-5.2-Q3"),
            (ModelPreset::KimiK3, "Kimi-K3-UD-IQ1_M"),
            (ModelPreset::KimiK31, "Kimi-K3.1-Pro-KLX-IQ1_S"),
            (ModelPreset::Qwen3CoderNext, "Qwen3-Coder-Next-Q4im"),
            (ModelPreset::Glm53Flash, "GLM-5.3-Flash-Q3im"),
            (ModelPreset::Qwen38FlashNext, "Qwen3.8-Flash-Next-Q3im"),
        ];
        assert_eq!(want.len(), ModelPreset::value_variants().len());
        for (preset, name) in want {
            let shard = format!("/models/{}/{}/shard.gguf", preset.def().family_dir, preset.def().production_variant);
            assert_eq!(named(&[shard.as_str()]), name);
        }
        let served = ModelPreset::Qwen38FlashNext.served_model_ids("Qwen3.8-Flash-Next-Q3im");
        let off = Some(false);
        let want = [
            ("Qwen/Qwen3.8-Flash-Next-Q3im", None),
            ("Qwen/Qwen3.8-Flash-Next-Q3im+nothink", off),
            ("default+nothink", off),
        ];
        assert_eq!(served, want.map(|(id, think)| (id.to_string(), think)));
        assert_eq!(named(&["/m/Qwen3.5-397B-A17B-GGUF/Q4_K_S/a.gguf"]), "Qwen3.5-397B-A17B-Q4_K_S");
        assert_eq!(named(&["/data/Foo-draft.gguf", "/data/Foo-00001-of-00002.gguf"]), "Foo");
    }
}
