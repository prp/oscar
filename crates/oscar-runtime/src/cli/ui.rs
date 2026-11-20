//! Terminal rendering: the status bar chat and serve paint.

use serde::{Deserialize, Serialize};

use crate::engine::dispatcher::OffloadDispatcher;
use crate::machine::GB;
use crate::pool::TokenSample;

/// Hit and prefetch shares of every expert access in %, a sealed layer's all hits; `None` where carriers saw no access.
pub fn expert_shares(accesses: u64, misses: u64, prefetch_saves: u64, paged_share: f64) -> Option<(f64, f64)> {
    if accesses == 0 {
        return (paged_share == 0.0).then_some((100.0, 0.0));
    }
    let of = |x: u64| 100.0 * (x as f64 / accesses as f64).min(1.0) * paged_share;
    Some((100.0 - of(misses), of(prefetch_saves)))
}

const BAR_TICK: std::time::Duration = std::time::Duration::from_millis(250);

/// NVMe GB/s per tick; the shares and GB a token per 1 s window, held between windows against flicker.
pub struct BarDriver {
    prefix: String,
    pd: std::sync::Arc<OffloadDispatcher>,
    /// The load's start and the header's (experts, parameters), until `loaded`.
    load: Option<(std::time::Instant, (u64, u64))>,
    loaded: std::sync::Arc<std::sync::atomic::AtomicBool>,
    sample: (std::time::Instant, u64),
    window: (std::time::Instant, TokenSample, u64),
    nvme_gbs: f64,
    hit_pct: f64,
    pref_pct: f64,
    gb_per_tok: f64,
    /// The engine's completion count where the decode on show began.
    decode_began: Option<u64>,
    last: Option<(u64, f64)>,
}

impl BarDriver {
    const WINDOW: std::time::Duration = std::time::Duration::from_millis(1000);

    /// On a thread of its own: the load's cells until the returned flag is set, the engine's phase after; with
    /// `lines`, the same figures as a JSON line a tick on stdout (a reader gone ends serve by its stdin's EOF).
    pub fn drive(
        bar: StatusBar, prefix: String, pd: std::sync::Arc<OffloadDispatcher>, figures: (u64, u64), lines: bool,
    ) -> std::sync::Arc<std::sync::atomic::AtomicBool> {
        let now = std::time::Instant::now();
        let loaded = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut driver = Self {
            prefix,
            pd,
            load: Some((now, figures)),
            loaded: loaded.clone(),
            sample: (now, oscar_offload::io::reads().bytes),
            window: (now, TokenSample::default(), 0),
            nvme_gbs: 0.0,
            hit_pct: 100.0,
            pref_pct: 0.0,
            gb_per_tok: 0.0,
            decode_began: None,
            last: None,
        };
        std::thread::Builder::new()
            .name("oscar-bar".into())
            .spawn(move || {
                let mut parked = false;
                loop {
                    let prompt = crate::cli::console::prompt_up();
                    if !(prompt && parked) {
                        let (phase, elapsed_s, fields) = driver.tick();
                        bar.show(&status_label(&driver.prefix, phase, elapsed_s, &fields));
                        if let Some(line) =
                            lines.then(|| serde_json::to_string(&Status::new(phase, elapsed_s, &fields)).ok()).flatten()
                        {
                            let _ = writeln!(std::io::stdout(), "{line}");
                        }
                    }
                    parked = prompt;
                    std::thread::sleep(BAR_TICK);
                }
            })
            .expect("the status bar's thread spawns");
        loaded
    }

    fn tick(&mut self) -> (&'static str, Option<f64>, StatusFields) {
        use mistralrs_core::live::{self, Phase};
        let now = std::time::Instant::now();
        let bytes = oscar_offload::io::reads().bytes;
        let dt = now.duration_since(self.sample.0).as_secs_f64();
        if dt > 0.0 && bytes >= self.sample.1 {
            self.nvme_gbs = (bytes - self.sample.1) as f64 / crate::machine::GB / dt;
        }
        self.sample = (now, bytes);
        if let Some((start, figures)) = self.load {
            if !self.loaded.load(std::sync::atomic::Ordering::Acquire) {
                // The header's own figures, so the columns hold when the pool comes in.
                let fields = StatusFields {
                    mem: MemCells::latest(),
                    experts: Some((None, figures.0)),
                    params: Some(figures.1),
                    ..Default::default()
                };
                return ("load", Some(start.elapsed().as_secs_f64()), fields);
            }
            self.load = None;
        }
        let l = live::now();
        let rate = |n: u64| if l.secs > 0.0 { n as f64 / l.secs } else { 0.0 };
        match l.phase {
            Phase::Prefill => {
                self.decode_began = None;
                ("prefill", Some(l.secs), self.fields(Some((l.prompt, rate(l.prompt))), false, false))
            }
            Phase::Decode => {
                if self.decode_began.is_none() {
                    self.decode_began = Some(l.completion_total - l.completion);
                    self.window = (now, TokenSample::now(), 0);
                }
                self.sample_window(now, l.completion);
                self.last = Some((l.completion, rate(l.completion)));
                ("decode", Some(l.secs), self.fields(self.last, true, true))
            }
            Phase::Idle => {
                // Idle starts where the decode ended: its exact count at the last tick's rate.
                if let Some(began) = self.decode_began.take() {
                    self.last = self.last.map(|(_, rate)| (l.completion_total.saturating_sub(began), rate));
                }
                ("idle", None, self.fields(self.last, true, false))
            }
        }
    }

    fn sample_window(&mut self, now: std::time::Instant, tokens: u64) {
        let (since, base, base_tokens) = self.window;
        if now.duration_since(since) < Self::WINDOW {
            return;
        }
        let snap = TokenSample::now();
        let accesses = snap.pool.accesses().saturating_sub(base.pool.accesses());
        let misses = snap.pool.blocked().saturating_sub(base.pool.blocked());
        let prefetch_saves = snap.pool.prefetch_saves.saturating_sub(base.pool.prefetch_saves);
        if let Some((hit, pref)) = expert_shares(accesses, misses, prefetch_saves, self.pd.pool().paged_share()) {
            (self.hit_pct, self.pref_pct) = (hit, pref);
        }
        let wtok = tokens.saturating_sub(base_tokens);
        if wtok > 0 {
            self.gb_per_tok = snap.reads.bytes.saturating_sub(base.reads.bytes) as f64 / 1e9 / wtok as f64;
        }
        self.window = (now, snap, tokens);
    }

    /// `count` as (tokens, rate); per-token cells only where `per_token`, the drafter drafting only where `drafting`.
    fn fields(&self, count: Option<(u64, f64)>, per_token: bool, drafting: bool) -> StatusFields {
        let plan = crate::memory::crossing::live();
        // The pool's present blocks, as a controller sheds and regrows them; the plan's figure before the pool.
        let kv_gb = crate::gpu::kv_pool::KvPool::installed()
            .map_or_else(|| plan.as_ref().map_or(0, |p| p.kv_cache_bytes), |p| p.present_bytes() as u64)
            as f64
            / GB;
        let disk_gb = crate::kv_store::disk_bytes() as f64 / GB;
        let experts = plan.as_ref().map(|plan| {
            // A resident expert holds a slot in each of its three banks: occupied slots over three.
            let (sum, _) = self.pd.pool().slot_occupancy();
            (Some(sum[0].saturating_sub(sum[1]) as u64 / 3), (plan.block_count * plan.num_experts) as u64)
        });
        let decode = count.filter(|_| per_token);
        StatusFields {
            tokens: count.map(|(n, _)| n),
            tok_per_s: count.map(|(_, r)| r),
            hit_pct: decode.map(|_| self.hit_pct),
            pref_pct: decode.map(|_| self.pref_pct),
            nvme_gbs: Some(self.nvme_gbs),
            gb_per_tok: decode.map(|_| self.gb_per_tok),
            pread_bytes: Some(oscar_offload::io::reads().bytes),
            mem: MemCells::latest(),
            drafter: drafter_cell().map(|armed| armed && drafting),
            experts,
            params: plan.as_ref().map(|plan| plan.total_params),
            expert_gb: Some(self.pd.pool().expert_bytes() as f64 / GB),
            kv_gb: Some((kv_gb, disk_gb)),
        }
    }
}

fn stdin_is_terminal_or_pipe() -> bool {
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    unsafe { libc::fstat(0, &mut st) == 0 && (libc::isatty(0) == 1 || st.st_mode & libc::S_IFMT == libc::S_IFIFO) }
}

/// Ctrl-C, SIGTERM or, with `watch_eof`, a terminal's or pipe's EOF: the rows back, the store flushed (a second
/// signal skips it).
/// The default action skips Drop and leaves the bar's rows reserved.
pub fn exit_on_signal(bar: StatusBar, watch_eof: bool) {
    tokio::spawn(async move {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = signal(SignalKind::terminate()).expect("SIGTERM handler");
        // A terminal or a pipe (the app's, ending when it dies) is read; /dev/null would end at once, and a
        // never-ending blocking task would hold the runtime's drop.
        let eof = async move {
            if watch_eof && stdin_is_terminal_or_pipe() {
                let _ = tokio::task::spawn_blocking(|| {
                    let mut line = String::new();
                    while matches!(std::io::stdin().read_line(&mut line), Ok(n) if n > 0) {
                        line.clear();
                    }
                })
                .await;
            } else {
                std::future::pending::<()>().await;
            }
        };
        let code = tokio::select! {
            _ = tokio::signal::ctrl_c() => 130,
            _ = term.recv() => 143,
            _ = eof => 0,
        };
        bar.restore();
        if !watch_eof {
            crate::cli::console::say(b"\n".to_vec());
        }
        let flush = tokio::task::spawn_blocking(|| crate::kv_store::flush_global(std::time::Duration::from_secs(20)));
        tokio::select! {
            _ = flush => {}
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
        crate::cli::console::drain();
        std::process::exit(code);
    });
}

// The status bar: every sequence saves and restores the cursor, so a paint never disturbs the line being written.

use std::io::Write;

/// The bar's rows: the two it paints and the gutter above them.
const BAR_ROWS: usize = 3;

/// Painted by the console's writer, the one thread that writes the terminal, so a terminal not reading stalls no lane.
#[derive(Clone, Default)]
pub struct StatusBar(std::sync::Arc<BarState>);

#[derive(Default)]
struct BarState {
    /// The columns and rows the region was set for, packed; 0 = none.
    size: std::sync::atomic::AtomicU64,
    last: std::sync::Mutex<BarCells>,
}

impl StatusBar {
    /// The terminal's size, where stdout is one with room for the bar.
    pub fn size() -> Option<(usize, usize)> {
        terminal_size::terminal_size()
            .map(|(w, h)| (w.0 as usize, h.0 as usize))
            .filter(|_| std::io::IsTerminal::is_terminal(&std::io::stdout()))
            .filter(|&(w, h)| h > BAR_ROWS + 1 && w >= 24)
    }

    /// Drawn when the writer gets to them.
    pub fn show(&self, cells: &BarCells) {
        *self.0.last.lock().unwrap() = cells.clone();
        self.request();
    }

    fn request(&self) {
        let bar = self.clone();
        crate::cli::console::paint(Box::new(move |out| bar.draw(out)));
    }

    /// The latest cells at the terminal's size now: the writer's side.
    fn draw(&self, out: &mut dyn Write) {
        let Some((w, h)) = Self::size() else { return };
        let cells = self.0.last.lock().unwrap().clone();
        let key = ((w as u64) << 32) | h as u64;
        let was = self.0.size.swap(key, std::sync::atomic::Ordering::AcqRel);
        if was != key {
            // The first entry scrolls a screen into the scrollback: a window grown later pulls it back and the old
            // rows land under the new bar (Terminal anchors the bottom only where there is scrollback).
            if was == 0 {
                let _ = write!(out, "\x1b[{h};1H{}\x1b[1;1H", "\x1bD".repeat(h));
            }
            // A narrowed window reflows the old rows into more lines above the bar: clear the lines they grew by.
            let was_w = (was >> 32) as usize;
            let grown = if was_w > w { 2 * (was_w.div_ceil(w) - 1) } else { 0 };
            enter_rows(out, h, grown.min(h.saturating_sub(BAR_ROWS + 1)));
        }
        let (top, bottom) = layout(&cells, w);
        // `\x1b[0m` first, so the stream's colour does not mix with the bar's; the cursor save keeps the stream's own.
        let _ = write!(
            out,
            "\x1b7\x1b[{};1H\x1b[2K\x1b[0m{}\x1b[{};1H\x1b[2K\x1b[0m{}\x1b8",
            h - 1,
            paint_row(&top, BAR_MODEL),
            h,
            paint_row(&bottom, BAR_PHASE),
        );
    }

    /// At a prompt nothing else shows the bar, and a resize resets the terminal's margins; needs a runtime.
    pub fn watch_resizes(&self) {
        let bar = self.clone();
        tokio::spawn(async move {
            let mut resized = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::window_change())
                .expect("SIGWINCH handler");
            while resized.recv().await.is_some() {
                bar.request();
            }
        });
    }

    /// Full-screen scrolling back, the rows cleared: the bar's last paint, waited for as an ending waits.
    pub fn restore(&self) {
        let bar = self.clone();
        crate::cli::console::close(Box::new(move |out| {
            let key = bar.0.size.swap(0, std::sync::atomic::Ordering::AcqRel);
            if key > 0 {
                leave_rows(out, (key & 0xffff_ffff) as usize);
            }
        }));
    }
}

/// Restores the rows on every exit path, `?` and panics included; a signal handler holds a clone of the bar instead.
pub struct StatusBarGuard(pub StatusBar);

impl Drop for StatusBarGuard {
    fn drop(&mut self) {
        self.0.restore();
    }
}

/// Confine scrolling to the rows above the bar, the cursor inside: index moves keep the column a pty's newline resets.
/// Setting the margins homes the cursor, so it sits in a save.
fn enter_rows(out: &mut dyn Write, height: usize, grown: usize) {
    let _ = write!(out, "\x1b7\x1b[r\x1b8\x1bD\x1bD\x1bD\x1b[{BAR_ROWS}A\x1b7\x1b[1;{}r", height - BAR_ROWS);
    for row in (height - 2 - grown)..=(height - 2) {
        let _ = write!(out, "\x1b[{row};1H\x1b[2K");
    }
    let _ = write!(out, "\x1b8");
}

/// Full-screen scrolling again, the rows cleared; the reset sits in a cursor save, as it homes the cursor too.
fn leave_rows(out: &mut dyn Write, height: usize) {
    let _ = write!(
        out,
        "\x1b7\x1b[r\x1b[{};1H\x1b[2K\x1b[{};1H\x1b[2K\x1b[{};1H\x1b[2K\x1b8",
        height - 2,
        height - 1,
        height,
    );
}

/// Columns of two rows, a topic each; each cell's items (a label and its figure) spread across the column.
pub type BarCells = Vec<[Vec<String>; 2]>;

/// Every column as wide as its wider cell, the items a space apart; a narrow terminal cuts the rows on the right.
pub fn layout(cells: &BarCells, width: usize) -> (String, String) {
    let text = |items: &[String]| items.join(" ");
    let widths: Vec<usize> =
        cells.iter().map(|c| c.iter().map(|items| text(items).chars().count()).max().unwrap_or(0)).collect();
    let row = |i: usize| {
        let texts: Vec<String> = cells.iter().zip(&widths).map(|(c, w)| fit(&text(&c[i]), *w)).collect();
        fit(&texts.join(" │ "), width)
    };
    (row(0), row(1))
}

/// `label` padded or cut to `width`.
fn fit(label: &str, width: usize) -> String {
    format!("{:<width$}", label.chars().take(width).collect::<String>())
}

// 256-colour, so the bar reads the same on light and dark themes: an explicit background, not reverse video.
const BAR_BG: &str = "\x1b[48;5;236m"; // charcoal bar
const BAR_FG: &str = "\x1b[38;5;252m"; // primary text
const BAR_MODEL: &str = "\x1b[1;38;5;81m"; // model name: bold cyan
const BAR_DIM: &str = "\x1b[38;5;243m"; // separators
const BAR_PHASE: &str = "\x1b[1;38;5;180m"; // the phase and its timer
const BAR_OK: &str = "\x1b[38;5;114m"; // green: the drafter drafting
const BAR_CRIT: &str = "\x1b[1;38;5;203m"; // red: memory critical

/// Zero-width escapes only, so the plain alignment holds: the first cell in `first`, else the separators and states.
/// Warn keeps the bar's colour: it is the kernel's ordinary reading under a full plan, and a tint would flicker.
fn paint_row(plain: &str, first: &str) -> String {
    let cells: Vec<String> = plain
        .split('│')
        .enumerate()
        .map(|(i, cell)| match i {
            0 => format!("{first}{}", cell.replace("drafter", &format!("{BAR_OK}drafter{first}"))),
            _ if cell.trim_end().ends_with(" crit") => {
                format!(
                    "{BAR_FG}{}{BAR_CRIT}crit{BAR_FG}{}",
                    cell.trim_end().trim_end_matches("crit"),
                    " ".repeat(cell.len() - cell.trim_end().len())
                )
            }
            _ => format!("{BAR_FG}{}", cell.replace("drafter", &format!("{BAR_OK}drafter{BAR_FG}"))),
        })
        .collect();
    format!("{BAR_BG}{}\x1b[0m", cells.join(&format!("{BAR_DIM}│")))
}

/// `None` renders a right-aligned `-` of the same width, so every phase shares the column positions.
#[derive(Default, Clone, Copy)]
pub struct StatusFields {
    pub tokens: Option<u64>,
    pub tok_per_s: Option<f64>,
    pub hit_pct: Option<f64>,
    pub pref_pct: Option<f64>,
    pub nvme_gbs: Option<f64>,
    pub gb_per_tok: Option<f64>,
    pub pread_bytes: Option<u64>,
    pub mem: MemCells,
    /// `None` where the model has no drafter, else whether it drafts.
    pub drafter: Option<bool>,
    /// (resident, total), the resident count `None` during the load.
    pub experts: Option<(Option<u64>, u64)>,
    pub params: Option<u64>,
    pub expert_gb: Option<f64>,
    /// (in memory as allocated, the context store on disk).
    pub kv_gb: Option<(f64, f64)>,
}

pub fn drafter_cell() -> Option<bool> {
    crate::config::drafter_on().then(crate::drafter::drafting)
}

/// The menu bar's reading of the bar: the phase, its timer and count, the memory over what the kernel pages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Status {
    /// load, prefill, decode or idle.
    pub phase: String,
    pub elapsed_s: Option<f64>,
    pub tokens: Option<u64>,
    pub tok_per_s: Option<f64>,
    /// Read from the drive, GB/s.
    pub nvme_gb_s: Option<f64>,
    pub own_gb: Option<f64>,
    pub other_gb: Option<f64>,
    pub total_gb: Option<f64>,
    pub level: Option<String>,
}

impl Status {
    pub fn new(phase: &str, elapsed_s: Option<f64>, f: &StatusFields) -> Self {
        Self {
            phase: phase.to_string(),
            elapsed_s,
            tokens: f.tokens,
            tok_per_s: f.tok_per_s,
            nvme_gb_s: f.nvme_gbs,
            own_gb: f.mem.own_gb,
            other_gb: f.mem.other_gb,
            total_gb: f.mem.total_gb,
            level: f.mem.level.map(str::to_string),
        }
    }
}

/// `elastic_gb` (what a shed can release), `fixed_gb` and `state` only under the adaptive controller.
#[derive(Default, Clone, Copy)]
pub struct MemCells {
    /// The kernel's memory-pressure level: normal, warn or critical.
    pub level: Option<&'static str>,
    /// What the kernel pages: the menu's bar is drawn over it.
    pub total_gb: Option<f64>,
    pub own_gb: Option<f64>,
    pub elastic_gb: Option<f64>,
    pub fixed_gb: Option<f64>,
    pub other_gb: Option<f64>,
    pub free_gb: Option<f64>,
    pub state: Option<&'static str>,
}

impl MemCells {
    /// From the sensor's latest snapshot and the controller's live plan, with no syscall.
    pub fn latest() -> Self {
        let Some(s) = crate::memory::latest_snapshot() else {
            return Self::default();
        };
        let gb = |b: u64| Some(b as f64 / GB);
        let bar = crate::memory::controller::bar();
        // A warn reading holds three seconds, as the level moves sample to sample around a plan that fills the machine.
        use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
        static WARN_UNTIL_MS: AtomicU64 = AtomicU64::new(0);
        let now_ms =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64);
        let word = crate::machine::level_word(s.level);
        if word == "warn" {
            WARN_UNTIL_MS.store(now_ms + 3000, Relaxed);
        }
        let level = if word == "normal" && now_ms < WARN_UNTIL_MS.load(Relaxed) { "warn" } else { word };
        Self {
            level: Some(level),
            total_gb: gb(s.physical),
            own_gb: gb(s.own_footprint),
            elastic_gb: bar.map(|(_, elastic, _)| elastic as f64 / GB),
            fixed_gb: bar.map(|(_, _, fixed)| fixed as f64 / GB),
            other_gb: gb(s.foreign()),
            free_gb: gb(s.reclaimable),
            state: bar.map(|(state, ..)| state),
        }
    }
}

fn cell_u(v: Option<u64>, w: usize) -> String {
    match v {
        Some(x) => format!("{x:>w$}"),
        None => format!("{:>w$}", "-"),
    }
}

fn cell_f(v: Option<f64>, w: usize, p: usize) -> String {
    match v {
        Some(x) => format!("{x:>w$.p$}"),
        None => format!("{:>w$}", "-"),
    }
}

/// Six characters: whole GB to 1024, then TB to one decimal up to 99.9, whole above.
fn cell_io(v: Option<u64>) -> String {
    match v {
        Some(b) => {
            let g = b as f64 / crate::machine::GB;
            if g >= 1025.0 {
                let t = g / 1024.0;
                if t >= 99.95 {
                    format!("{t:>4.0}TB")
                } else {
                    format!("{t:>4.1}TB")
                }
            } else {
                format!("{g:>4.0}GB")
            }
        }
        None => format!("{:>4}GB", "-"),
    }
}

/// One template for every phase: three columns of two rows, every figure fixed-width so no column moves, absent `-`.
pub fn status_label(model: &str, phase: &str, elapsed_s: Option<f64>, f: &StatusFields) -> BarCells {
    let experts = match f.experts {
        Some((r, t)) => format!(
            "experts {:>w$} (mem) / {t} (total)",
            r.map_or("-".to_string(), |r| r.to_string()),
            w = t.to_string().len()
        ),
        None => "experts - (mem) / - (total)".to_string(),
    };
    // The loader's parameter count, rounded as its first line says it.
    let model = match f.params {
        Some(p) => format!("{model} ({:.0}B)", p as f64 / 1e9),
        None => model.to_string(),
    };
    // The drafter's liveness last, blank while idle and absent without one; every item at a fixed width.
    let mut phase = vec![
        format!("{phase:<7}"),
        format!("{} tok/s", cell_rate(f.tok_per_s)),
        format!("{} tokens", cell_u(f.tokens, 6)),
        elapsed_s.map_or(" ".repeat(6), timer),
    ];
    if let Some(drafting) = f.drafter {
        phase.push(format!("{:<7}", if drafting { "drafter" } else { "" }));
    }
    // A fixed tag, blank when normal, so the bar never shifts when the kernel's level moves.
    let level = match f.mem.level {
        Some("warn") => "warn",
        Some("critical") => "crit",
        _ => "",
    };
    // Widths hold the largest realistic figures: 999.9 GB, 999,999 tok/s and tokens, 99.99 GB a token, 99.9 GB/s.
    let mem = match (f.mem.elastic_gb, f.mem.state) {
        (Some(elastic), Some(state)) => vec![
            format!("mem own: {}GB", cell_f(f.mem.own_gb, 5, 1)),
            format!("elastic: {}GB", cell_f(Some(elastic), 5, 1)),
            format!("fixed: {}GB", cell_f(f.mem.fixed_gb, 5, 1)),
            format!("other: {}GB", cell_f(f.mem.other_gb, 5, 1)),
            format!("free: {}GB", cell_f(f.mem.free_gb, 5, 1)),
            format!("{state:<9}"),
            format!("{level:<4}"),
        ],
        _ => vec![
            format!("mem own: {}GB", cell_f(f.mem.own_gb, 5, 1)),
            format!("expert: {}GB", cell_f(f.expert_gb, 5, 1)),
            format!("other: {}GB", cell_f(f.mem.other_gb, 5, 1)),
            format!("free: {}GB", cell_f(f.mem.free_gb, 5, 1)),
            format!("{level:<4}"),
        ],
    };
    // One item, so the two KV figures stay together however wide the column.
    let kv = vec![format!(
        "KV mem: {}GB      disk: {}GB",
        cell_f(f.kv_gb.map(|k| k.0), 5, 1),
        cell_f(f.kv_gb.map(|k| k.1), 5, 1)
    )];
    let offloading =
        vec![experts, format!("hit: {}%", cell_pct(f.hit_pct)), format!("prefetch: {}%", cell_pct(f.pref_pct))];
    let io = vec![
        format!("NVMe {}GB/s", cell_f(f.nvme_gbs, 4, 1)),
        format!("{}GB/tok", cell_f(f.gb_per_tok, 5, 2)),
        format!("read {}", cell_io(f.pread_bytes)),
    ];
    vec![[vec![model], phase], [offloading, mem], [io, kv]]
}

/// Six characters: one decimal to 9,999.9, whole numbers past it.
fn cell_rate(v: Option<f64>) -> String {
    match v {
        Some(x) if x >= 9_999.95 => format!("{:>6.0}", x.min(999_999.0)),
        other => cell_f(other, 6, 1),
    }
}

/// Four characters: one decimal to 99.9, then "100".
fn cell_pct(v: Option<f64>) -> String {
    match v {
        Some(x) if x >= 99.95 => format!("{:>4.0}", x),
        Some(x) => format!("{x:>4.1}"),
        None => format!("{:>4}", "-"),
    }
}

/// Six characters: seconds under a minute, then m:ss up to three-digit minutes.
fn timer(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    if s < 60 {
        format!("{s:>5}s")
    } else {
        format!("{:>3}:{:02}", s / 60, s % 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layout aligns the columns and cuts a narrow terminal on the right; the painted row keeps the plain text
    /// intact and tints only state; the rows are entered and left inside a cursor save; the labels' columns are fixed
    /// across phases, the cells formatted (the IO cell past 1024 GB in TB), a hit an access that waited on no read.
    #[test]
    fn the_bar_is_laid_out_painted_and_its_labels_fixed_across_phases() {
        let item = |s: &str| s.to_string();
        let cells: BarCells = vec![
            [vec![item("glm-5.2")], vec![item("decode    1:15")]],
            [vec![item("experts 1/2")], vec![item("tokens    161"), item("5.6 tok/s")]],
            [vec![item("own: 1.0"), item("free: 2.0GB")], vec![item("hit 99.0%")]],
        ];
        let (top, bottom) = layout(&cells, 70);
        assert_eq!(top, format!("{:<70}", "glm-5.2        │ experts 1/2             │ own: 1.0 free: 2.0GB"));
        assert_eq!(bottom, format!("{:<70}", "decode    1:15 │ tokens    161 5.6 tok/s │ hit 99.0%"));
        assert_eq!(layout(&cells, 10).0, "glm-5.2   ");
        let plain = fit("glm-5.2 │ own: 12.0 free: 2.0GB crit │ drafter │ read 2GB", 80);
        let painted = paint_row(&plain, BAR_MODEL);
        let stripped = painted
            .split('\x1b')
            .enumerate()
            .map(|(i, s)| if i == 0 { s } else { &s[s.find('m').unwrap() + 1..] })
            .collect::<String>();
        assert_eq!(stripped, plain);
        assert!(painted.starts_with(&format!("{BAR_BG}{BAR_MODEL}glm-5.2")));
        assert!(painted.contains(&format!("{BAR_CRIT}crit")) && painted.contains(&format!("{BAR_OK}drafter")));
        assert!(!paint_row(" own: 1.0 free: 2.0GB warn ", BAR_PHASE).contains(BAR_CRIT));
        // The drafter in the first cell tints too and gives the cell its own colour back.
        let phase = paint_row("decode 1:15 drafter │ tokens 1", BAR_PHASE);
        assert!(phase.contains(&format!("{BAR_OK}drafter{BAR_PHASE}")), "{phase:?}");
        let written = |write: &dyn Fn(&mut dyn Write)| {
            let mut buf: Vec<u8> = Vec::new();
            write(&mut buf);
            String::from_utf8(buf).unwrap()
        };
        // Up three after the index moves, as three rows are reserved: fewer leaves the prompt on the gutter.
        let s = written(&|out| enter_rows(out, 40, 0));
        assert_eq!(s, "\x1b7\x1b[r\x1b8\x1bD\x1bD\x1bD\x1b[3A\x1b7\x1b[1;37r\x1b[38;1H\x1b[2K\x1b8");
        // A window narrowed: the lines the old rows grew by go too.
        let s = written(&|out| enter_rows(out, 40, 2));
        assert!(s.contains("\x1b[36;1H\x1b[2K\x1b[37;1H\x1b[2K\x1b[38;1H\x1b[2K\x1b8"), "{s:?}");
        let s = written(&|out| leave_rows(out, 40));
        assert!(s.starts_with("\x1b7\x1b[r") && s.ends_with("\x1b[40;1H\x1b[2K\x1b8"), "{s:?}");
        // The labels.
        let full = StatusFields {
            tokens: Some(161),
            tok_per_s: Some(5.6),
            hit_pct: Some(95.9),
            pref_pct: Some(0.8),
            nvme_gbs: Some(7.8),
            gb_per_tok: Some(1.42),
            pread_bytes: Some(847 << 30),
            mem: MemCells {
                level: Some("warn"),
                total_gb: Some(64.0),
                own_gb: Some(54.0),
                elastic_gb: Some(38.4),
                fixed_gb: Some(15.6),
                other_gb: Some(4.1),
                free_gb: Some(5.9),
                state: Some("steady"),
            },
            experts: Some((Some(5580), 30720)),
            params: Some(396_958_000_000),
            expert_gb: Some(33.5),
            kv_gb: Some((0.9, 12.3)),
            drafter: Some(true),
        };
        let decode = status_label("qwen", "decode", Some(75.0), &full);
        let prefill = status_label("qwen", "prefill", Some(12.0), &StatusFields { drafter: Some(false), ..full });
        let idle = status_label("qwen", "idle", None, &StatusFields { drafter: Some(false), ..full });
        // Before the pool holds any: the total at its width; before the plan, the name alone. A static plan has no
        // elastic part or state word, and a model with no drafter no drafter item.
        let loading = StatusFields { experts: Some((None, 30720)), ..StatusFields::default() };
        let loading = status_label("qwen", "load", Some(3.0), &loading);
        let bare = status_label("qwen", "idle", None, &StatusFields::default());
        assert_eq!((&decode[0][0], &loading[0][0]), (&vec!["qwen (397B)".to_string()], &vec!["qwen".to_string()]));
        let mem = "mem own:  54.0GB elastic:  38.4GB fixed:  15.6GB other:   4.1GB free:   5.9GB steady    warn";
        for (items, text) in [
            (&decode[0][1], "decode     5.6 tok/s    161 tokens   1:15 drafter"),
            (&loading[1][0], "experts     - (mem) / 30720 (total) hit:    -% prefetch:    -%"),
            (&prefill[0][1], "prefill    5.6 tok/s    161 tokens    12s        "),
            (&idle[0][1], "idle       5.6 tok/s    161 tokens               "),
            (&decode[1][0], "experts  5580 (mem) / 30720 (total) hit: 95.9% prefetch:  0.8%"),
            (&decode[1][1], mem),
            (&decode[2][0], "NVMe  7.8GB/s  1.42GB/tok read  847GB"),
            (&decode[2][1], "KV mem:   0.9GB      disk:  12.3GB"),
            (&bare[0][1], "idle         - tok/s      - tokens       "),
            (&bare[1][0], "experts - (mem) / - (total) hit:    -% prefetch:    -%"),
            (&bare[1][1], "mem own:     -GB expert:     -GB other:     -GB free:     -GB     "),
            (&bare[2][0], "NVMe    -GB/s     -GB/tok read    -GB"),
            (&bare[2][1], "KV mem:     -GB      disk:     -GB"),
        ] {
            assert_eq!(items.join(" "), text);
        }
        // The separators line up, and the adaptive controller's full bar, its widest at 184 columns, shows every cell.
        let full_bar = status_label("Qwen3.5-397B-A17B-Q3", "decode", Some(75.0), &full);
        let (top, bottom) = layout(&full_bar, 184);
        let bars =
            |s: &str| -> Vec<usize> { s.chars().enumerate().filter(|(_, c)| *c == '│').map(|(i, _)| i).collect() };
        assert_eq!(bars(&top), bars(&bottom), "{top}\n{bottom}");
        assert_eq!(bars(&top).len(), 2, "{top}");
        assert!(top.starts_with("Qwen3.5-397B-A17B-Q3 (397B) "), "{top}");
        assert!(top.contains("│ experts  5580 (mem) / 30720 (total) "), "{top}");
        assert!(top.trim_end().ends_with("│ NVMe  7.8GB/s  1.42GB/tok read  847GB"), "{top}");
        assert!(bottom.starts_with("decode     5.6 tok/s    161 tokens   1:15 drafter │ mem own:"), "{bottom}");
        assert!(bottom.trim_end().ends_with("│ KV mem:   0.9GB      disk:  12.3GB"), "{bottom}");
        // Narrower, the rows are cut on the right.
        let (_, narrow) = layout(&full_bar, 155);
        assert_eq!(narrow.chars().count(), 155, "{narrow}");
        assert!([(100.0, " 100"), (99.94, "99.9"), (0.0, " 0.0")].iter().all(|&(x, s)| cell_pct(Some(x)) == s));
        let rates = [(5.6, "   5.6"), (567.5, " 567.5"), (4040.0, "4040.0"), (16077.1, " 16077"), (2e6, "999999")];
        assert!(rates.iter().all(|&(x, s)| cell_rate(Some(x)) == s), "{rates:?}");
        let ios = [(847, " 847GB"), (1024, "1024GB"), (1412, " 1.4TB"), (23700, "23.1TB"), (123000, " 120TB")];
        assert!(ios.iter().all(|&(gb, cell)| cell_io(Some(gb << 30)) == cell), "{ios:?}");
        assert_eq!((cell_rate(None).as_str(), cell_io(None).as_str()), ("     -", "   -GB"));
        // Every layer sealed, or no offloading at all: nothing was fetched; carriers that saw no access say nothing
        // about the window; every layer a carrier: the pool's counts are the whole story.
        assert_eq!(expert_shares(0, 0, 0, 0.0), Some((100.0, 0.0)));
        assert_eq!(expert_shares(0, 0, 0, 0.5), None);
        assert_eq!(expert_shares(1000, 100, 50, 1.0), Some((90.0, 5.0)));
        // Flash-next's decode: 20 of 48 layers carriers, 4.1% of their accesses missed, 6.3% prefetched in time.
        let (hit, pref) = expert_shares(1000, 41, 63, 20.0 / 48.0).unwrap();
        assert!((hit - 98.29).abs() < 0.01 && (pref - 2.625).abs() < 0.001, "{hit} {pref}");
        // The status line carries the phase, its figures and the memory over the machine's total, absent ones null.
        let status = Status::new("decode", Some(75.0), &full);
        let text = serde_json::to_string(&status).unwrap();
        assert_eq!(serde_json::from_str::<Status>(&text).unwrap(), status);
        assert!(text.starts_with(r#"{"phase":"decode","elapsed_s":75.0,"tokens":161,"tok_per_s":5.6,"#), "{text}");
        assert!(text.ends_with(r#""own_gb":54.0,"other_gb":4.1,"total_gb":64.0,"level":"warn"}"#), "{text}");
        let idle = Status::new("idle", None, &StatusFields::default());
        assert_eq!(
            serde_json::to_string(&idle).unwrap(),
            r#"{"phase":"idle","elapsed_s":null,"tokens":null,"tok_per_s":null,"nvme_gb_s":null,"own_gb":null,"other_gb":null,"total_gb":null,"level":null}"#
        );
    }
}
