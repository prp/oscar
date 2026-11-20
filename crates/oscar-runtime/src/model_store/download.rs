//! A conversion's published source into the files oscar runs, each output verified before its rename.
//! Files fetch one at a time, each saturating the link; a shard's rewrite runs beside the next shard's fetch, so a
//! shard-local conversion keeps two source shards on disk, one when one is left.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use candle_core::quantized::{gguf_file, GgmlDType};
use oscar_models::rows::conversion::Transform;
use oscar_models::rows::ModelPreset;

use super::conversion::ImatrixMap;
use super::gguf_rw::{self, Metadata};
use super::hub::{self, RemoteFile};
use super::imatrix;
use crate::model_store;

pub enum Event<'a> {
    /// A line always said: the run's summary before any bytes move, a quality note at its end.
    Line(&'a str),
    /// A line for `-v`: the listing, a shard's sizes, the disk charge's terms.
    Detail(&'a str),
    /// The bar's length: every source and the imatrix, those a produced output consumed counted as kept at once.
    FetchTotal(u64),
    /// Fetched over the link.
    Fetched(u64),
    /// Already on disk (a resumed chunk, a verified or adopted file): the bar's position, never its rate.
    Kept(u64),
    /// What runs while the link is idle (a shard's rewrite, a source verified on disk, a overlay, a derived tensor).
    Doing(&'a str),
}

/// GB to a decimal, under one MB: powers of two, as every size oscar says.
pub fn size_text(bytes: u64) -> String {
    if bytes < 1 << 30 {
        format!("{} MB", bytes >> 20)
    } else {
        format!("{:.1} GB", bytes as f64 / crate::machine::GB)
    }
}

/// A stream that stops sending raises nothing by itself: the read timeout turns it into an error its chunk retries.
pub fn client() -> anyhow::Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .user_agent("oscar-runtime")
        .connect_timeout(std::time::Duration::from_secs(30))
        .read_timeout(std::time::Duration::from_secs(60))
        .build()?)
}

fn basename(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// The model's own file, or a shard of its split: `<model>-<select>.gguf` or `<model>-<select>-NNNNN-of-NNNNN.gguf`,
/// `<model>` the repo's name less `-GGUF`; a side file sharing the selector (an MTP head) is not the model.
fn is_source(repo: &str, select: &str, path: &str) -> bool {
    let model = repo.rsplit('/').next().unwrap_or(repo).trim_end_matches("-GGUF");
    let Some(rest) = basename(path).strip_prefix(model).and_then(|r| r.strip_prefix('-')) else { return false };
    let Some(rest) = rest.strip_prefix(select).and_then(|r| r.strip_suffix(".gguf")) else { return false };
    let digits = |s: &str| s.len() == 5 && s.bytes().all(|b| b.is_ascii_digit());
    rest.is_empty()
        || rest.strip_prefix('-').and_then(|r| r.split_once("-of-")).is_some_and(|(i, n)| digits(i) && digits(n))
}

#[cfg(test)]
#[test]
fn the_source_is_the_models_own_split_never_a_side_file() {
    let repo = "unsloth/Qwen3.8-Flash-Next-GGUF";
    let kept: Vec<&str> = [
        "MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf",
        "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
        "Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf",
        "Q8_0/Qwen3.8-Flash-Next-Q8_0-00006-of-00006.gguf",
        "mtp-Qwen3.8-Flash-Next-Q8_0.gguf",
        "UD-Q8_K_XL/Qwen3.8-Flash-Next-UD-Q8_K_XL-00001-of-00006.gguf",
        "Qwen3.8-Flash-Next-Q8_0.gguf.part",
    ]
    .into_iter()
    .filter(|p| is_source(repo, "Q8_0", p))
    .collect();
    assert_eq!(
        kept,
        ["Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf", "Q8_0/Qwen3.8-Flash-Next-Q8_0-00006-of-00006.gguf"]
    );
    assert!(is_source("unsloth/Qwen3.6-35B-A3B-GGUF", "Q8_0", "Qwen3.6-35B-A3B-Q8_0.gguf"));
    assert!(is_source("unsloth/Kimi-K3-GGUF", "UD-IQ1_M", "UD-IQ1_M/Kimi-K3-UD-IQ1_M-00015-of-00015.gguf"));
}

enum Make {
    /// Source shard `i` rewritten alone, deleted once the output is renamed.
    Shard(usize),
    /// Source shard `i` cloned into the artifact unchanged.
    Clone(usize),
    /// Drawn from every source shard.
    Overlay,
}

/// Fetch, transform and promote, resumably: verified outputs are kept, their sources not fetched again.
pub async fn download_variant(
    client: &reqwest::Client, base: &str, preset: ModelPreset, threads: usize, on: &mut dyn FnMut(Event),
) -> anyhow::Result<PathBuf> {
    let conversion = &preset.def().conversion;
    let (family, variant, repo) = (preset.def().family_dir, preset.def().production_variant, conversion.source_repo);

    if let Ok((dir, _)) = model_store::resolve_variant(family, variant) {
        anyhow::bail!(
            "{} already at {} - `delete --model {}` removes managed copy",
            preset.label(),
            dir.display(),
            preset.label()
        );
    }

    on(Event::Detail(&format!("listing {repo}")));
    let listing = hub::manifest(client, base, repo).await?;
    let mut sources: Vec<RemoteFile> =
        listing.iter().filter(|f| is_source(repo, conversion.source_select, &f.path)).cloned().collect();
    // Shard order is processing order; the listing's is not contractual.
    sources.sort_by(|a, b| a.path.cmp(&b.path));
    anyhow::ensure!(!sources.is_empty(), "{repo} holds no .gguf matching \"{}\"", conversion.source_select);
    let imatrix_file = match conversion.imatrix {
        None => None,
        Some((irepo, path)) => {
            let listing = if irepo == repo { listing.clone() } else { hub::manifest(client, base, irepo).await? };
            let file = listing
                .into_iter()
                .find(|f| f.path == path)
                .ok_or_else(|| anyhow::anyhow!("{irepo} does not publish {path}"))?;
            Some((irepo, file))
        }
    };

    let staging = model_store::staging_dir(family, variant);
    let src_dir = staging.join("src");
    std::fs::create_dir_all(&src_dir)?;
    let adopt_dirs: Vec<PathBuf> = model_store::roots()
        .iter()
        .flat_map(|root| {
            model_store::variants_in_root(root, family).into_iter().map(|(v, _)| root.join(family).join(v))
        })
        .collect();

    let outputs: Vec<(String, Make)> = match &conversion.transform {
        Transform::Overlay { overlay, .. } => (0..sources.len())
            .map(|i| (basename(&sources[i].path).to_string(), Make::Clone(i)))
            .chain([(overlay.to_string(), Make::Overlay)])
            .collect(),
        Transform::PassThrough | Transform::Rewrite { .. } => (0..sources.len())
            .map(|i| (basename(&sources[i].path).replace(conversion.source_select, variant), Make::Shard(i)))
            .collect(),
    };
    let done: Vec<bool> = outputs.iter().map(|(name, _)| staging.join(name).exists()).collect();
    let mut needed = vec![false; sources.len()];
    for ((_, make), _) in outputs.iter().zip(&done).filter(|(_, d)| !**d) {
        match *make {
            Make::Shard(i) | Make::Clone(i) => needed[i] = true,
            Make::Overlay => needed.fill(true),
        }
    }
    let local: Vec<Option<PathBuf>> =
        sources.iter().zip(&needed).map(|(f, &n)| if n { hub::adoptable(&adopt_dirs, f) } else { None }).collect();
    let needed_sizes = || sources.iter().zip(&needed).filter(|(_, n)| **n).map(|(f, _)| f.size);

    // The disk admission, before any payload moves, a GB over for the headers, which are priced over every one for the
    // model's layer count (a first split's `*.block_count`), not a shard's.
    let imatrix_bytes = imatrix_file.as_ref().map_or(0, |(_, f)| f.size);
    let hdrs = fetch_headers(client, base, repo, &sources, &src_dir).await?;
    let staged = model_store::dir_bytes(&src_dir);
    let price =
        gguf_rw::plan(&hdrs, |name, _, dims, n| super::conversion::decide(&conversion.transform, name, dims, n))?;
    let layers = price.n_layers;
    let exact = price.entries.iter().filter(|e| needed[e.src_index]).map(|e| e.out_bytes).sum::<u64>();
    let out = exact + (1 << 30);
    // A overlay run keeps every source, cloned into the store at no cost; a rewrite keeps two shards beside its output,
    // one when a single source is left to fetch.
    let (need, terms) = if let Transform::Overlay { .. } = conversion.transform {
        let adopted: u64 = sources.iter().zip(&local).filter(|(_, l)| l.is_some()).map(|(f, _)| f.size).sum();
        let to_fetch = (needed_sizes().sum::<u64>() + imatrix_bytes).saturating_sub(adopted).saturating_sub(staged);
        let terms = format!(
            "{} source still to fetch ({} adoptable by local clone, {} staged) + {} overlay",
            size_text(to_fetch),
            size_text(adopted),
            size_text(staged),
            size_text(out)
        );
        (to_fetch + out, terms)
    } else {
        let (largest, held) = (needed_sizes().max().unwrap_or(0), needed_sizes().count().min(2) as u64);
        let terms = format!(
            "{} exact output + {} of {} + {} imatrix, less {} staged",
            size_text(out),
            if held == 2 { "two source shards" } else { "one source shard" },
            size_text(largest),
            size_text(imatrix_bytes),
            size_text(staged)
        );
        ((out + held * largest + imatrix_bytes).saturating_sub(staged), terms)
    };
    let todo = done.iter().filter(|d| !**d).count();
    let fetch = needed_sizes().sum::<u64>() + imatrix_bytes;
    let summary =
        format!("{} {} to fetch, {} {variant} to write", size_text(fetch), conversion.source_select, size_text(exact));
    admit(on, "download", &preset.label(), (todo, outputs.len()), need, &terms, &summary)?;
    // The bar over the whole model: the sources a produced output consumed count as kept from the start.
    let consumed: u64 = sources.iter().zip(&needed).filter(|(_, n)| !**n).map(|(f, _)| f.size).sum();
    on(Event::FetchTotal(fetch + consumed));
    if consumed > 0 {
        on(Event::Kept(consumed));
    }

    let imap: ImatrixMap = match &imatrix_file {
        Some((irepo, f)) => imatrix::load(
            &stage(client, base, irepo, f, hub::adoptable(&adopt_dirs, f), &src_dir, on, "the imatrix").await?,
        )?,
        None => ImatrixMap::new(),
    };
    let imap = Arc::new(imap);
    let decide = |name: &str, _: GgmlDType, dims: &[usize], _: usize| {
        super::conversion::decide(&conversion.transform, name, dims, layers)
    };
    // Source shard `i` staged: fetched or adopted, verified once; the clones keep theirs for the overlay.
    macro_rules! shard {
        ($i:expr) => {{
            let what = format!("shard {}/{}", $i + 1, sources.len());
            stage(client, base, repo, &sources[$i], local[$i].clone(), &src_dir, on, &what).await?
        }};
    }
    let mut staged_src: Vec<Option<PathBuf>> = vec![None; sources.len()];
    let mut unweighted = (0usize, None::<String>);
    let mut rewriting: Option<Rewrite> = None;
    for ((name, make), done) in outputs.iter().zip(done) {
        if done {
            on(Event::Detail(&format!("{name}: present, kept")));
            continue;
        }
        match *make {
            Make::Shard(i) => {
                let src = shard!(i);
                let plan = gguf_rw::plan(std::slice::from_ref(&src), &decide)?;
                tally(&plan, &imap, &mut unweighted);
                // The previous shard's rewrite ran beside this fetch; one rewrite at a time.
                if let Some(job) = rewriting.take() {
                    job.finish(&sources, on).await?;
                }
                rewriting = Some(Rewrite::start(plan, &staging, name, &imap, threads, i, src)?);
            }
            Make::Clone(i) => {
                let src = shard!(i);
                hub::clone_file(&src, &staging.join(name))?;
                staged_src[i] = Some(src);
                on(Event::Detail(&format!("cloned {name}")));
            }
            Make::Overlay => {
                let mut all = Vec::with_capacity(sources.len());
                for i in 0..sources.len() {
                    all.push(match staged_src[i].take() {
                        Some(src) => src,
                        None => shard!(i),
                    });
                }
                let plan = gguf_rw::plan(&all, &decide)?;
                anyhow::ensure!(!plan.entries.is_empty(), "overlay rules select no tensors");
                tally(&plan, &imap, &mut unweighted);
                // Only the architecture key: metadata is first-wins, so more keys could shadow the shards' values.
                let meta = Metadata::Replace(vec![("general.architecture".to_string(), architecture(&all[0])?)]);
                on(Event::Doing(&format!("writing {name}")));
                write(&plan, &staging, name, meta, &imap, threads, |_| {})?;
                on(Event::Detail(&format!("wrote {name} ({} tensors)", plan.entries.len())));
            }
        }
    }
    if let Some(job) = rewriting.take() {
        job.finish(&sources, on).await?;
    }
    if unweighted.0 > 0 {
        on(Event::Line(&format!(
            "{} tensors without imatrix entry, quantised unweighted (first: {})",
            unweighted.0,
            unweighted.1.as_deref().unwrap_or("?"),
        )));
    }

    // Every output is a rewrite or a clone, so the sources go whole.
    std::fs::remove_dir_all(&src_dir)?;
    let dir = model_store::promote(family, variant)?;
    derive_into(preset, &dir, threads, on)?;
    Ok(dir)
}

/// Each derived tensor requantised into a `000-` file of its own beside the artifact at `dir`, an existing one kept.
fn derive_into(preset: ModelPreset, dir: &Path, threads: usize, on: &mut dyn FnMut(Event)) -> anyhow::Result<()> {
    let (conversion, variant) = (&preset.def().conversion, preset.def().production_variant);
    let names = model_store::gguf_names(dir);
    let shards: Vec<PathBuf> = names.iter().filter(|n| !n.contains("-000-")).map(|n| dir.join(n)).collect();
    let first = shards.first().ok_or_else(|| anyhow::anyhow!("{} holds no shards", dir.display()))?;
    // `<stem>-<variant>-NNNNN-of-NNNNN.gguf` → `<stem>-<variant>`.
    let base = basename(first.to_str().unwrap_or_default());
    let prefix = base
        .rfind("-of-")
        .and_then(|i| base[..i].rfind('-').map(|j| &base[..j]))
        .unwrap_or_else(|| base.trim_end_matches(".gguf"));
    let mut todo = Vec::new();
    for d in conversion.derived {
        let name = format!("{prefix}-{}.gguf", d.file);
        if dir.join(&name).exists() {
            on(Event::Detail(&format!("{name}: present, kept")));
            continue;
        }
        let mut plan = gguf_rw::plan(&shards, |n, _, _, _| {
            if n == d.from {
                gguf_rw::Disposition::Requant(d.dtype)
            } else {
                gguf_rw::Disposition::Skip
            }
        })?;
        anyhow::ensure!(
            plan.entries.len() == 1,
            "{name}: {variant} holds {} tensor(s) named {}",
            plan.entries.len(),
            d.from
        );
        plan.entries[0].written_as = Some(d.name.to_string());
        todo.push((name, d, plan));
    }
    let need = todo.iter().map(|(_, _, plan)| plan.payload_bytes()).sum();
    let summary = format!("{} to write", size_text(need));
    admit(on, "derive", &preset.label(), (todo.len(), conversion.derived.len()), need, "the exact output", &summary)?;
    let arch = architecture(first)?;
    for (name, d, plan) in todo {
        let meta = Metadata::Replace(vec![("general.architecture".to_string(), arch.clone())]);
        on(Event::Doing(&format!("deriving {}", d.file)));
        write(&plan, dir, &name, meta, &ImatrixMap::new(), threads, |_| {})?;
        on(Event::Detail(&format!("{name}: {} from {} as {:?}, written", d.name, d.from, d.dtype)));
    }
    Ok(())
}

/// The admission: one line with `summary` (what moves), the charge's `terms` for `-v`, and the refusal; `files` is
/// (still to write, of all).
fn admit(
    on: &mut dyn FnMut(Event), verb: &str, label: &str, files: (usize, usize), need: u64, terms: &str, summary: &str,
) -> anyhow::Result<()> {
    let free = oscar_offload::machine::volume_free(&model_store::managed_root()).unwrap_or(0);
    let line = match files.0 {
        0 => format!("{verb} {label}: all files written"),
        n => format!("{verb} {label}: {n}/{} files, {summary}, {} free", files.1, size_text(free)),
    };
    on(Event::Line(&line));
    on(Event::Detail(&format!("needs {}: {terms}", size_text(need))));
    anyhow::ensure!(
        free >= need,
        "volume cannot host {verb}: {} needed ({terms}), {} free - `list` shows what can be deleted",
        size_text(need),
        size_text(free)
    );
    Ok(())
}

/// `file` into `src/`, cloned from a verified `local` copy, else fetched; `what` names it on the bar while verified.
async fn stage(
    client: &reqwest::Client, base: &str, repo: &str, file: &RemoteFile, local: Option<PathBuf>, src_dir: &Path,
    on: &mut dyn FnMut(Event), what: &str,
) -> anyhow::Result<PathBuf> {
    let dest = src_dir.join(basename(&file.path));
    if local.is_some_and(|l| hub::clone_file(&l, &dest).is_ok()) {
        on(Event::Detail(&format!("{}: identical local copy, cloned", basename(&file.path))));
        on(Event::Kept(file.size));
    } else {
        let verifying = format!("verifying {what}");
        hub::fetch(client, base, repo, file, &dest, |p| {
            on(match p {
                hub::Progress::Fetched(b) => Event::Fetched(b),
                hub::Progress::Kept(b) => Event::Kept(b),
                hub::Progress::Verifying => Event::Doing(&verifying),
            })
        })
        .await?;
    }
    Ok(dest)
}

/// A shard's rewrite on its own thread, so the next shard's fetch runs beside it; waited for with the bar saying how
/// far it is, its source deleted after. Dropped unfinished (an error elsewhere) it is still waited for, so its output
/// lands whole and its thread is not left behind.
struct Rewrite {
    task: Option<std::thread::JoinHandle<anyhow::Result<u64>>>,
    written: Arc<AtomicU64>,
    exact: u64,
    i: usize,
    src: PathBuf,
    name: String,
}

impl Rewrite {
    fn start(
        plan: gguf_rw::Plan, staging: &Path, name: &str, imap: &Arc<ImatrixMap>, threads: usize, i: usize, src: PathBuf,
    ) -> anyhow::Result<Self> {
        let written = Arc::new(AtomicU64::new(0));
        let (exact, dir, out, imap, count) =
            (plan.payload_bytes(), staging.to_path_buf(), name.to_string(), imap.clone(), written.clone());
        let task = std::thread::Builder::new().name("oscar-rewrite".into()).spawn(move || {
            write(&plan, &dir, &out, Metadata::Copy, &imap, threads, |b| {
                count.fetch_add(b, Ordering::Relaxed);
            })
        })?;
        Ok(Self { task: Some(task), written, exact, i, src, name: name.to_string() })
    }

    async fn finish(mut self, sources: &[RemoteFile], on: &mut dyn FnMut(Event)) -> anyhow::Result<()> {
        let task = self.task.take().expect("a rewrite finishes once");
        while !task.is_finished() {
            let pct = 100 * self.written.load(Ordering::Relaxed) / self.exact.max(1);
            on(Event::Doing(&format!("requantising shard {}/{} {pct:>3}%", self.i + 1, sources.len())));
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }
        let exact = task.join().map_err(|_| anyhow::anyhow!("rewrite of {} panicked", self.name))??;
        std::fs::remove_file(&self.src)?;
        let f = &sources[self.i];
        on(Event::Detail(&format!(
            "shard {}/{}: {} ({}) -> {} ({}), source deleted",
            self.i + 1,
            sources.len(),
            basename(&f.path),
            size_text(f.size),
            self.name,
            size_text(exact),
        )));
        Ok(())
    }
}

impl Drop for Rewrite {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            if task.join().is_ok_and(|r| r.is_ok()) {
                let _ = std::fs::remove_file(&self.src);
            }
        }
    }
}

/// Via `name.partial`, renamed once its header parses and lists every planned tensor; `written` is told each tensor.
fn write(
    plan: &gguf_rw::Plan, dir: &Path, name: &str, meta: Metadata, imap: &ImatrixMap, threads: usize,
    mut written: impl FnMut(u64),
) -> anyhow::Result<u64> {
    let exact = plan.payload_bytes();
    let free = oscar_offload::machine::volume_free(&model_store::managed_root()).unwrap_or(0);
    anyhow::ensure!(free >= exact, "volume cannot host {name}: {} needed, {} free", size_text(exact), size_text(free));
    let partial = dir.join(format!("{name}.partial"));
    gguf_rw::execute(plan, &partial, meta, imap, threads, |done, _| written(plan.entries[done - 1].out_bytes))?;
    let wrote = crate::plan::header(&partial)
        .map_err(|e| anyhow::anyhow!("{}: written output does not parse: {e}", partial.display()))?
        .tensor_infos
        .len();
    anyhow::ensure!(
        wrote == plan.entries.len(),
        "{}: wrote {wrote} tensors, the plan says {}",
        partial.display(),
        plan.entries.len()
    );
    std::fs::rename(&partial, dir.join(name))?;
    Ok(exact)
}

/// Counts requantised tensors without an imatrix entry, naming the first.
fn tally(plan: &gguf_rw::Plan, imap: &ImatrixMap, unweighted: &mut (usize, Option<String>)) {
    if imap.is_empty() {
        return;
    }
    for e in plan.entries.iter().filter(|e| e.is_requant() && !imap.contains_key(&e.name)) {
        unweighted.0 += 1;
        unweighted.1.get_or_insert_with(|| e.name.clone());
    }
}

/// Each source's header into `src/<name>.hdr`, the prefix grown until it parses; a resume reuses it.
async fn fetch_headers(
    client: &reqwest::Client, base: &str, repo: &str, sources: &[RemoteFile], src_dir: &Path,
) -> anyhow::Result<Vec<PathBuf>> {
    let mut out = Vec::with_capacity(sources.len());
    for f in sources {
        let hdr = src_dir.join(format!("{}.hdr", basename(&f.path)));
        let mut len: u64 = 4 << 20;
        while crate::plan::header(&hdr).is_err() {
            anyhow::ensure!(len <= (512 << 20), "{}: no GGUF header within 512 MB", f.path);
            hub::fetch_prefix(client, base, repo, f, len, &hdr).await?;
            len *= 4;
        }
        out.push(hdr);
    }
    Ok(out)
}

fn architecture(shard: &Path) -> anyhow::Result<gguf_file::Value> {
    crate::plan::header(shard)?
        .metadata
        .remove("general.architecture")
        .ok_or_else(|| anyhow::anyhow!("{} carries no architecture key", shard.display()))
}
