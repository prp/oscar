//! The model build `chat` and `serve` share, so every front end runs the identical model.

use std::path::PathBuf;

use anyhow::{Context, Result};
use mistralrs::GgufModelBuilder;

use crate::engine::dispatcher::OffloadDispatcher;
use crate::load::install_offload_dispatcher_from_env;

use crate::cli::args::{ModelArgs, ModelPreset};
use crate::cli::settings::{self, Settings};
use crate::cli::ui::{BarDriver, StatusBar, StatusBarGuard};

pub struct BuiltModel {
    pub model: mistralrs::Model,
    pub preset: ModelPreset,
    pub dispatcher: std::sync::Arc<OffloadDispatcher>,
    pub model_name: String,
    pub bar: StatusBarGuard,
}

/// At serve's planned sequence or chat's; the status bar, entered before the load, counts it where there is a
/// terminal, and says it on stdout under `status_lines`.
pub async fn build_model(args: &ModelArgs, serve: bool, status_lines: bool) -> Result<BuiltModel> {
    let bar = StatusBar::default();
    bar.watch_resizes();
    let guard = StatusBarGuard(bar.clone());
    // The load's own lines are held back below `-v`.
    crate::cli::console::set_quiet(args.verbose == 0);
    // The mirror is a machine property, like the memory ceiling.
    let settings = Settings::load()?;
    crate::config::drives::install(args.mirror.as_deref().or(settings.mirror.as_deref()))?;
    // Explicit flags win over the model's; `--quant` re-points it at another variant, which `--gguf-*` still override.
    let (preset, quant) = settings.preset(args.model, args.quant.as_deref())?;
    let (gguf_dir, gguf_files, gguf_paths) =
        crate::model_store::locate(preset, quant.as_deref(), args.gguf_dir.as_deref(), &args.gguf_file)?;
    let tok_repo = args.tok_repo.clone().unwrap_or_else(|| preset.def().tok_repo.to_string());

    // Per-model, per-entry defaults; the flags override either side.
    let (default_x, default_y) = preset.planned_sequence_defaults(serve);
    let max_input = args.max_input_tokens.unwrap_or(default_x);
    let max_output = args.max_output_tokens.unwrap_or(default_y);
    // The default directory is keyed by model, quant and template, so no KV restores into a model it does not describe.
    let kv_cache_dir = match args.kv_cache_dir.as_deref() {
        Some("") => None,
        Some(d) => Some(PathBuf::from(d)),
        None => {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            gguf_dir.hash(&mut h); // carries the quant-variant subdirectory
            gguf_files.hash(&mut h);
            preset.def().chat_template.hash(&mut h);
            Some(settings::dir().join("kv_cache").join(format!("{preset:?}-{:016x}", h.finish())))
        }
    };
    crate::machine::assume_physical(args.physical_mem).map_err(anyhow::Error::msg)?;
    // The flags over a user's exports over the model's row; diagnostics, which corrupt the transcript, off below `-vv`.
    let budget = crate::config::told_budget_gb(args.mem)?;
    crate::config::configure(crate::config::Params::for_launch(
        preset,
        budget,
        Some((max_input, max_output, args.max_input_tokens.is_some() || args.max_output_tokens.is_some())),
        Some(args.verbose),
    ));
    crate::kv_store::configure(kv_cache_dir, crate::config::params().misc.kv_cache_disk_gb);
    crate::detail!(target: "oscar::loader",
        "settings: {}",
        if preset.def().params.is_empty() { "common defaults, none measured for this model" } else { "measured for this model" },
    );
    let model_name = crate::load::report::model_name(&gguf_paths);
    // Stage one: a launch the machine cannot host stops before the weights; the sensor and its watchdog start.
    let dispatcher = install_offload_dispatcher_from_env(&gguf_paths).context("the model routes no experts")?;

    // The KV store, before anything builds a prefix cache.
    mistralrs_core::prefix_cacher::set_diagnostics(crate::config::params().misc.diag);
    mistralrs_core::prefix_cacher::set_paged_kv_store_factory(crate::kv_store::factory);

    // The vendor's sampling rows `generation_config.json` cannot hold; serve picks by the request's `enable_thinking`.
    mistralrs_core::set_mode_generation_defaults(preset.vendor_sampling());
    // The identity line a client sending no system message gets, as the vendor's own service states it.
    if let Some(system) = preset.def().default_system {
        mistralrs_core::set_default_system_prompt(system.to_string());
    }

    tracing::info!(target: "oscar::loader",
        "loading {} shard{} from {} (tokenizer {}) ...",
        gguf_paths.len(),
        if gguf_paths.len() == 1 { "" } else { "s" },
        gguf_dir.display(),
        tok_repo,
    );
    // Prefix caching, as agents re-send near-identical context; later turns depart the cold prefill's FP trajectory.
    crate::drafter::register_round();
    let mut builder = GgufModelBuilder::new(gguf_dir.to_string_lossy().as_ref(), gguf_files.clone())
        .with_device_mapping(crate::config::device_mapping())
        .with_tok_model_id(&tok_repo)
        .with_prefix_cache(true);
    // A segment per block at the live plan's count, read after the load's tail; the index space is the tail's ceiling.
    {
        let block_size = crate::plan::paged_block_size() as usize;
        crate::gpu::kv_pool::install(move |layout| {
            let Some(plan) = crate::memory::crossing::live() else {
                candle_core::bail!("memory plan not live: KV block pool built after load's tail");
            };
            let index = crate::plan::pool_ceilings().map_or(0, |c| c.kv_index_blocks as usize);
            let per_token = crate::load::stage_stats().map_or(0, |s| s.kv_bytes_per_token) as usize;
            if layout.block_bytes() != per_token * layout.block_size {
                candle_core::bail!(
                    "engine's KV block {} bytes, plan charges {} per block",
                    layout.block_bytes(),
                    per_token * layout.block_size
                );
            }
            let blocks = plan.kv_blocks as usize;
            if blocks == 0 {
                candle_core::bail!("memory plan charges no KV blocks");
            }
            Ok((blocks, index.max(blocks)))
        });
        crate::plan::arm_draft_reserve(crate::config::charge().draft_reserve_tokens, block_size);
        builder = builder
            .with_paged_attn(mistralrs::PagedAttentionMetaBuilder::default().with_block_size(block_size).build()?);
    }
    builder = builder.with_max_num_seqs(crate::plan::LIVE_SEQUENCES as usize);
    // `--jinja`, else the model's own: published templates fail under minijinja, shift tokens or drop prior reasoning.
    let jinja = match args.jinja.as_deref().filter(|s| !s.is_empty()) {
        Some(explicit) => explicit.to_string(),
        None => preset.materialise_chat_template().map_err(|e| anyhow::anyhow!("chat template write failed: {e}"))?,
    };
    builder = builder.with_jinja_explicit(jinja);
    if args.verbose >= 1 {
        builder = builder.with_logging();
    }
    // `build()` maps the weights, runs the census, solves the plan and warms up; the bar runs from here for life.
    let stats = crate::load::stage_stats().context("the load's first stage has not run")?;
    let figures = (stats.block_count as u64 * stats.num_experts as u64, stats.total_params);
    let loaded = BarDriver::drive(bar, model_name.clone(), dispatcher.clone(), figures, status_lines);
    let model = builder.build().await.context("GgufModelBuilder::build");
    loaded.store(true, std::sync::atomic::Ordering::Release);
    let model = model?;
    // The loaded model is the remembered one: the app and a bare `serve` start with it.
    if let Err(e) = (Settings { model: Some(preset.label()), quant: quant.clone(), ..settings }).save() {
        tracing::warn!(target: "oscar::loader", "The model choice was not saved: {e:#}");
    }
    // The live plan the load's tail solved: a load without one fails in the KV pool's build.
    let live = crate::memory::crossing::live().context("the load ended with no live plan")?;
    // An allocation's pool is its lifetime: checked here, the one moment nothing is in flight.
    crate::engine::eval::hooks().load_done();
    // Held back until the load's bar phase ends, or it lands mid-frame.
    if let Some(note) = crate::kv_store::startup_note() {
        tracing::info!(target: "oscar::store", "{note}");
    }

    // An ask for blocks the pool cannot hold goes to the crossing and the controller follows the machine; a static
    // plan keeps its load-time sizes.
    if !args.static_mem {
        mistralrs_core::paged_attention::set_kv_ask_hook(Box::new(crate::memory::crossing::request_kv_blocks));
        mistralrs_core::paged_attention::set_kv_pending_hook(Box::new(crate::memory::crossing::kv_blocks_pending));
        crate::memory::controller::control(&live, (crate::config::params().budget_gb * crate::machine::GB) as u64);
    }

    Ok(BuiltModel { model, preset, dispatcher, model_name, bar: guard })
}
