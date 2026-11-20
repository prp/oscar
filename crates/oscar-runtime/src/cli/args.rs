use clap::{Args, Parser};
use mistralrs::SamplingParams;
use mistralrs_core::ModelGenerationDefaults;

pub use oscar_models::rows::ModelPreset;

/// The model and runtime flags every model-building subcommand (`chat`,
/// `serve`) flattens in, so their CLI surfaces match.
#[derive(Debug, Args)]
pub struct ModelArgs {
    /// Predefined model to load (sets sensible defaults for --gguf-dir,
    /// --gguf-file, and --tok-repo; the individual flags still override).
    /// Omitted, the one `~/.oscar/settings.toml` remembers: the Oscar menu's pick.
    #[arg(long, value_enum)]
    pub model: Option<ModelPreset>,

    /// Quantisation variant of the model (e.g. `Q4_K_M`, `UD-Q4_K_S`):
    /// loads `<models>/<family>-GGUF/<QUANT>/*.gguf` instead of the default
    /// variant; --gguf-dir / --gguf-file still override it.
    #[arg(long)]
    pub quant: Option<String>,

    /// Directory containing the GGUF file(s). Overrides the model's
    /// default subdirectory under the models root.
    #[arg(long)]
    pub gguf_dir: Option<String>,

    /// GGUF filename(s) inside `--gguf-dir`. Repeatable for sharded
    /// GGUFs. When omitted, the model's own filenames are used.
    #[arg(long, num_args = 1..)]
    pub gguf_file: Vec<String>,

    /// Tokenizer HF repo id (or local path). Overrides the model's own.
    #[arg(long)]
    pub tok_repo: Option<String>,

    /// A .jinja chat-template path, overriding the model's own template.
    #[arg(long)]
    pub jinja: Option<String>,

    /// Directory of the durable KV store, where finished turns' KV is kept for
    /// a later prompt to resume from. Default
    /// `~/.oscar/kv_cache/<model>-<fingerprint>`; `--kv-cache-dir=` turns it
    /// off.
    #[arg(long, value_name = "DIR")]
    pub kv_cache_dir: Option<String>,

    /// Verbosity: `-v` prints the load's own lines and the engine's INFO
    /// tracing; `-vv` also streams periodic diagnostics during generation (an
    /// exported `OSCAR_DBG_DIAG` wins).
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// The most memory oscar may hold, in GB (a `G`/`GB` suffix accepted): the
    /// plan at this budget where the machine hosts it, else the largest it
    /// hosts, saying which bound held it. Unset: what the machine hosts, read
    /// at launch. Experts that do not fit are paged.
    #[arg(long, value_name = "GB", value_parser = parse_mem_gb)]
    pub mem: Option<f64>,

    /// Machine memory the planner assumes, to emulate a smaller machine: the
    /// `--mem` default and every bound follow it. Unset, the memory the kernel
    /// pages.
    #[arg(long, value_name = "GB", value_parser = parse_mem_gb)]
    pub physical_mem: Option<f64>,

    /// A second drive with a replica of the model, as `[<drive>:]<dir>` in the
    /// store's own layout (`<family>/<variant>/<shards>.gguf`): expert reads
    /// then span both drives. Unset, one drive.
    #[arg(long, value_name = "[DRIVE:]DIR")]
    pub mirror: Option<String>,

    /// The plan keeps its load-time sizes: no shedding as other applications
    /// take memory, no regrowing up to `--mem` as they give it back.
    #[arg(long)]
    pub static_mem: bool,

    /// Largest prompt (tokens) admission accepts; more shrinks the expert pool.
    /// Default: the model's pinned per-entry value.
    #[arg(long, value_name = "TOKENS")]
    pub max_input_tokens: Option<u64>,

    /// Generation guaranteed at a full-length prompt; a shorter prompt may
    /// generate past it, only the total being a hard stop. Defaulted like
    /// --max-input-tokens.
    #[arg(long, value_name = "TOKENS")]
    pub max_output_tokens: Option<u64>,
}

#[derive(Debug, Parser)]
pub struct ChatArgs {
    #[command(flatten)]
    pub common: ModelArgs,

    /// Max tokens per assistant turn, generous as a thinking model may reason
    /// for thousands before answering.
    #[arg(long, default_value_t = 16384)]
    pub max_len: usize,

    /// Sampling temperature, over the model's published row; 0 is greedy.
    #[arg(long)]
    pub temperature: Option<f64>,

    /// Answer without a reasoning pass (the template's `enable_thinking=false`);
    /// a thinking model's default effort can run minutes of tokens before the answer.
    #[arg(long)]
    pub no_think: bool,

    /// Stream the model's reasoning, dimmed, before its answer; without it a
    /// line counts the seconds it thinks.
    #[arg(long)]
    pub show_thinking: bool,

    /// System prompt prepended once at session start; unset, the model's
    /// identity line where it has one.
    #[arg(long)]
    pub system: Option<String>,

    /// Reasoning effort: the GLM models take `low` or `high` (Max otherwise);
    /// the Qwen models have none.
    #[arg(long, value_parser = ["low", "medium", "high"])]
    pub reasoning_effort: Option<String>,
}

/// The model `check-model` and `print-config` report on, and the budget they
/// plan at.
#[derive(Debug, Args)]
pub struct Target {
    /// Predefined model to report on.
    #[arg(long, value_enum)]
    pub model: ModelPreset,

    /// Quantisation variant, as for `chat`; variants share the model's settings.
    #[arg(long)]
    pub quant: Option<String>,

    /// Directory containing the GGUF file(s), overriding the model's.
    #[arg(long)]
    pub gguf_dir: Option<String>,

    /// The most memory oscar may hold, in GB, as for `chat`; omitted, chat's default.
    #[arg(long, value_name = "GB", value_parser = parse_mem_gb)]
    pub mem: Option<f64>,
}

#[derive(Debug, Parser)]
pub struct ServeArgs {
    #[command(flatten)]
    pub common: ModelArgs,

    /// Interface to bind the HTTP server to.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port for the OpenAI-compatible endpoint; omitted, the remembered one, else 11434.
    #[arg(long, short = 'p')]
    pub port: Option<u16>,

    /// The bar's figures as a JSON line a tick on stdout, for the menu bar.
    #[arg(long)]
    pub status_lines: bool,
}

/// The flags, the rest from the mode's row as serve applies it, else from `shipped`, the model's own file.
pub fn sampling_params(args: &ChatArgs, shipped: Option<&ModelGenerationDefaults>) -> SamplingParams {
    let mut sp = SamplingParams::neutral();
    sp.max_len = Some(args.max_len);
    sp.temperature = args.temperature;
    sp.apply_mode_defaults(!args.no_think, shipped);
    sp
}

/// A positive number of GB, bare or with a `G` or `GB` suffix in any case.
pub fn parse_mem_gb(raw: &str) -> std::result::Result<f64, String> {
    let s = raw.trim();
    if s.is_empty() {
        return Err("--mem value is empty".to_string());
    }
    let lower = s.to_ascii_lowercase();
    let num_str = lower.strip_suffix("gb").or_else(|| lower.strip_suffix('g')).unwrap_or(lower.as_str()).trim();
    let value: f64 = num_str.parse().map_err(|e| format!("invalid --mem value '{raw}': {e}"))?;
    if !(value.is_finite() && value > 0.0) {
        return Err(format!("--mem must be a positive number, got '{raw}'"));
    }
    Ok(value)
}
