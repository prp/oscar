//! The store's subcommands, presentation only: arguments, console lines, the bar, prompts; the work is the store's.

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::model_store::download::{download_variant, size_text, Event};
use crate::model_store::{self, hub};
use clap::{Args, ValueEnum};
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use oscar_models::rows::ModelPreset;

use super::args::Target;
use super::check_model::{check_model_main, CheckModelArgs};

#[derive(Debug, Args)]
pub struct DownloadArgs {
    /// The model to download.
    #[arg(long, value_enum)]
    pub model: ModelPreset,
    /// Every step's line, and the model's plan on this machine once stored.
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count)]
    pub verbose: u8,
}

#[derive(Debug, Args)]
pub struct DeleteArgs {
    /// Which model to delete from the managed store.
    #[arg(long, value_enum)]
    pub model: ModelPreset,

    /// Variant to delete; defaults to the one `download` stores.
    #[arg(long)]
    pub quant: Option<String>,

    /// Delete without asking.
    #[arg(long)]
    pub yes: bool,
}

/// A line beside the bar goes through it, so the two never interleave; without one, through the console.
fn say(bar: &Option<ProgressBar>, ev: Event) {
    let (line, detail) = match ev {
        Event::Line(l) => (l, false),
        Event::Detail(l) => (l, true),
        _ => return,
    };
    if detail && crate::cli::console::quiet() {
        return;
    }
    match bar {
        Some(b) => b.println(format!("{}{line}", crate::cli::console::Head("store"))),
        None => tracing::info!(target: "oscar::store", "{line}"),
    }
}

/// The bar's line, as wide as the summary line: sixty-six columns of figures at fixed widths, the bar the rest; the
/// last 29 the timer's: the link's rate and wait, what the run does while the link is idle, or how long no byte came.
/// The sizes in oscar's units, powers of two labelled GB and MB, which the library's own formats are not.
fn style(summary: usize) -> ProgressStyle {
    let width = summary.saturating_sub(66).max(10);
    let gb = |b: u64| b as f64 / crate::machine::GB;
    ProgressStyle::with_template(&format!(
        "{{bar:{width}}} {{percent:>3}}% {{done:>8}}/{{len:>8}} {{elapsed_precise}} {{msg:<29}}"
    ))
    .expect("template")
    .with_key("done", move |s: &ProgressState, w: &mut dyn std::fmt::Write| {
        let _ = write!(w, "{:.1} GB", gb(s.pos()));
    })
    .with_key("len", move |s: &ProgressState, w: &mut dyn std::fmt::Write| {
        let _ = write!(w, "{:.1} GB", gb(s.len().unwrap_or(0)));
    })
}

/// Over the last few seconds' fetched bytes, so the rate follows the link; kept bytes move the bar, never the rate.
const RATE_WINDOW: usize = 5;

/// The bar's right-hand figures for the second: what the run does, else a stall's length, else the rate and the wait.
fn figures(doing: Option<&str>, window: &std::collections::VecDeque<u64>, idle_s: u64, left: u64) -> String {
    if let Some(what) = doing {
        return what.to_string();
    }
    if idle_s >= 5 {
        return format!("no data for {idle_s} s");
    }
    // The samples are a second apart: the bytes between the first and the last over the seconds between them.
    let span = window.len().saturating_sub(1).max(1) as u64;
    let per_s = window.back().unwrap_or(&0).saturating_sub(*window.front().unwrap_or(&0)) / span;
    let eta = left
        .checked_div(per_s)
        .map_or("--:--:--".to_string(), |s| format!("{:02}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60));
    format!("{:>9.1} MB/s eta {eta}", per_s as f64 / (1u64 << 20) as f64)
}

pub async fn download_main(args: DownloadArgs) -> anyhow::Result<()> {
    crate::cli::console::set_quiet(args.verbose == 0);
    let preset = args.model;
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    // The model's plan on this machine, as information only.
    let report = || match args.verbose {
        0 => Ok(()),
        _ => check_model_main(CheckModelArgs {
            target: Target { model: preset, quant: None, gguf_dir: None, mem: None },
            ctx: None,
        }),
    };
    let client = model_store::download::client()?;

    // The bar is drawn once a second from a timer, never from a chunk's arrival: a steady line whatever the link does.
    let (fetched, kept) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    // What the run does while the link is idle (a hash, a rewrite waited for), cleared by the next fetched bytes.
    let doing = Arc::new(std::sync::Mutex::new(None::<String>));
    // The summary line's width, for the bar's line to match: the head and the line.
    let head = crate::cli::console::Head("store").to_string().chars().count();
    let mut summary = 0usize;
    let mut bar: Option<ProgressBar> = None;
    let mut timer: Option<tokio::task::JoinHandle<()>> = None;
    let dir = download_variant(&client, hub::HF_BASE, preset, threads, &mut |ev| match ev {
        Event::FetchTotal(total) => {
            let b = ProgressBar::new(total).with_style(style(summary));
            let (drawn, fetched, kept, doing) = (b.clone(), fetched.clone(), kept.clone(), doing.clone());
            timer = Some(tokio::spawn(async move {
                let mut tick = tokio::time::interval(Duration::from_secs(1));
                let (mut window, mut idle_s) = (std::collections::VecDeque::new(), 0u64);
                loop {
                    tick.tick().await;
                    let f = fetched.load(Ordering::Relaxed);
                    idle_s = if window.back() == Some(&f) { idle_s + 1 } else { 0 };
                    window.push_back(f);
                    if window.len() > RATE_WINDOW + 1 {
                        window.pop_front();
                    }
                    let at = f + kept.load(Ordering::Relaxed);
                    drawn.set_position(at);
                    let doing = doing.lock().unwrap_or_else(|e| e.into_inner()).clone();
                    drawn.set_message(figures(doing.as_deref(), &window, idle_s, total.saturating_sub(at)));
                }
            }));
            bar = Some(b);
        }
        Event::Fetched(delta) => {
            fetched.fetch_add(delta, Ordering::Relaxed);
            *doing.lock().unwrap_or_else(|e| e.into_inner()) = None;
        }
        Event::Doing(what) => {
            *doing.lock().unwrap_or_else(|e| e.into_inner()) = Some(what.to_string());
        }
        Event::Kept(bytes) => {
            kept.fetch_add(bytes, Ordering::Relaxed);
        }
        line => {
            if let Event::Line(l) = &line {
                summary = head + l.chars().count();
            }
            say(&bar, line)
        }
    })
    .await;
    if let Some(t) = timer {
        t.abort();
    }
    if let Some(b) = &bar {
        b.set_position(fetched.load(Ordering::Relaxed) + kept.load(Ordering::Relaxed));
        b.finish();
        // The bar's last draw leaves the cursor on its line; the lines after it start on their own.
        crate::cli::console::say(b"\n".to_vec());
    }
    let dir = dir?;
    tracing::info!(
        target: "oscar::store",
        "{}: {} at {}, {}",
        preset.label(),
        preset.def().production_variant,
        dir.display(),
        size_text(model_store::dir_bytes(&dir))
    );

    prefetch_tokenizer(preset.def().tok_repo).await;
    report()
}

/// So a downloaded model serves offline; best effort, a miss surfacing at load.
pub(crate) async fn prefetch_tokenizer(repo: &str) {
    let repo = repo.to_string();
    let done = tokio::task::spawn_blocking(move || {
        // Without the client's own progress bars, which would draw over the console's lines.
        let api = hf_hub::api::sync::ApiBuilder::new().with_progress(false).build().ok()?;
        let model = api.model(repo);
        for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"] {
            let _ = model.get(f);
        }
        Some(())
    })
    .await;
    if !matches!(done, Ok(Some(()))) {
        tracing::warn!(target: "oscar::store", "The tokenizer was not fetched: prefetch skipped; fetched at first load");
    }
}

pub fn list_main() -> anyhow::Result<()> {
    let roots = model_store::roots();
    let mut rows: Vec<[String; 4]> = vec![["model", "variant", "location", "size"].map(String::from)];
    let mut absent: Vec<String> = Vec::new();
    for preset in ModelPreset::value_variants() {
        let mut found = false;
        for root in &roots {
            for (variant, bytes) in model_store::variants_in_root(root, preset.def().family_dir) {
                let marker =
                    if variant == preset.def().production_variant { format!("{variant} (default)") } else { variant };
                rows.push([preset.label(), marker, root.display().to_string(), size_text(bytes)]);
                found = true;
            }
        }
        if !found {
            absent.push(preset.label());
        }
    }

    let mut w = [0usize; 4];
    for r in &rows {
        for (w, cell) in w.iter_mut().zip(r) {
            *w = (*w).max(cell.len());
        }
    }
    let [w0, w1, w2, w3] = w;
    for r in &rows {
        println!("{:<w0$}  {:<w1$}  {:<w2$}  {:>w3$}", r[0], r[1], r[2], r[3]);
    }
    if !absent.is_empty() {
        println!("\nnot on disk: {}", absent.join(", "));
    }

    let managed = model_store::managed_root();
    let managed_bytes: u64 = ModelPreset::value_variants()
        .iter()
        .flat_map(|p| model_store::variants_in_root(&managed, p.def().family_dir))
        .map(|(_, b)| b)
        .sum();
    let free = oscar_offload::machine::volume_free(&managed).map_or_else(|| "unknown".to_string(), size_text);
    println!("\nstore {}: {} , volume free {}", managed.display(), size_text(managed_bytes), free);
    Ok(())
}

/// One variant of the managed store, never model bytes elsewhere (`OSCAR_MODELS_DIR`).
pub fn delete_main(args: DeleteArgs) -> anyhow::Result<()> {
    let variant = args.quant.as_deref().unwrap_or(args.model.def().production_variant);
    let (dir, bytes, files) = model_store::managed_variant(args.model, variant)?;
    println!("delete {} - {} in {} file(s)", dir.display(), size_text(bytes), files);
    if !args.yes {
        print!("proceed? [y/N] ");
        std::io::stdout().flush()?;
        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer)?;
        if !matches!(answer.trim(), "y" | "Y" | "yes") {
            println!("not deleted");
            return Ok(());
        }
    }
    println!("deleted, {} freed", size_text(model_store::remove_variant(args.model, variant)?));
    Ok(())
}

#[cfg(test)]
#[test]
fn the_bar_says_the_rate_of_the_last_seconds_a_stall_and_what_the_run_does() {
    let mb = 1u64 << 20;
    // The last five seconds at 10 MB/s, after a slow first second that has left the window.
    let window: std::collections::VecDeque<u64> = [1, 11, 21, 31, 41, 51].iter().map(|m| m * mb).collect();
    assert_eq!(figures(None, &window, 0, 3600 * 10 * mb), "     10.0 MB/s eta 01:00:00");
    assert_eq!(figures(None, &window, 12, 0), "no data for 12 s");
    assert_eq!(figures(Some("rewriting shard 3/6 40%"), &window, 30, 0), "rewriting shard 3/6 40%");
    assert!(figures(None, &[5 * mb].into(), 0, mb).ends_with("eta --:--:--"), "no rate yet");
}
