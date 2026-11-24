//! One formatter, `HH:MM:SS [component]  message`; `WARN:` marks trouble that is no defect, `ERROR:` a code defect.
//! The terminal has one writer, this module's thread, so a terminal that is not reading stops the writer alone; the
//! log of a model's run, `~/.oscar/oscar.log`, has another.

use std::collections::VecDeque;
use std::fmt;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::{FmtContext, FormatEvent, FormatFields};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

struct Console;

/// The component column, brackets included, as wide as `[controller]`.
const COLUMN: usize = 13;

fn component(target: &str) -> String {
    match target {
        t if t.starts_with("oscar::") => t["oscar::".len()..].replace('_', " "),
        t if t.starts_with("oscar_runtime::serve") => "serve".into(),
        t if t.starts_with("oscar_runtime::chat") => "chat".into(),
        t if t.starts_with("oscar_runtime::kv_store") => "store".into(),
        t if t.starts_with("oscar_") => "runtime".into(),
        _ => "engine".into(),
    }
}

/// `-v`: full detail; without it, a line's main figures.
static VERBOSE: AtomicBool = AtomicBool::new(false);

pub fn verbose() -> bool {
    VERBOSE.load(Ordering::Relaxed)
}

static QUIET: AtomicBool = AtomicBool::new(false);

pub fn set_quiet(on: bool) {
    QUIET.store(on, Ordering::Relaxed);
}

pub fn quiet() -> bool {
    QUIET.load(Ordering::Relaxed)
}

/// The load's own detail, printed unless the entry asked for quiet (the interactive entries below `-v`).
#[macro_export]
macro_rules! detail {
    ($($arg:tt)*) => {
        if !$crate::cli::console::quiet() {
            tracing::info!($($arg)*);
        }
    };
}

/// From the tag to the line's end: dark red for an error, dark yellow for a warning, on a terminal only.
fn tint(level: &Level) -> Option<&'static str> {
    let tint = match *level {
        Level::ERROR => "\x1b[31m",
        Level::WARN => "\x1b[33m",
        _ => return None,
    };
    stderr_is_terminal().then_some(tint)
}

/// Where colour and the prompt's redraw mean something.
fn stderr_is_terminal() -> bool {
    static TERMINAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TERMINAL.get_or_init(|| std::io::IsTerminal::is_terminal(&std::io::stderr()))
}

/// The local wall clock to the second, as a server's record spans hours; allocates nothing.
pub fn clock() -> Clock {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let t = secs as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    // SAFETY: valid pointers for the call; localtime_r writes only `tm`.
    unsafe { libc::localtime_r(&t, &mut tm) };
    Clock(tm.tm_hour, tm.tm_min, tm.tm_sec)
}

/// `HH:MM:SS`, written where it is used.
pub struct Clock(i32, i32, i32);

impl fmt::Display for Clock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:02}:{:02}:{:02}", self.0, self.1, self.2)
    }
}

/// The `HH:MM:SS [component]` head of every line; allocates nothing.
pub struct Head<'a>(pub &'a str);

impl fmt::Display for Head<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [{}]{:pad$}", clock(), self.0, "", pad = COLUMN.saturating_sub(self.0.len() + 2))
    }
}

impl<S, N> FormatEvent<S, N> for Console
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(&self, ctx: &FmtContext<'_, S, N>, mut w: Writer<'_>, event: &Event<'_>) -> fmt::Result {
        let meta = event.metadata();
        write!(w, "{}", Head(&component(meta.target())))?;
        let tint = tint(meta.level()).filter(|_| w.has_ansi_escapes());
        if *meta.level() <= Level::WARN {
            write!(w, "{}{}: ", tint.unwrap_or(""), meta.level().as_str())?;
        }
        ctx.field_format().format_fields(w.by_ref(), event)?;
        writeln!(w, "{}", tint.map_or("", |_| "\x1b[0m"))
    }
}

/// Below `-v` the runtime's lines and the engine's warnings, with `records` serve's records, without them no line
/// of a memory change (a conversation shows it on the bar); `-v` all; `-vv` the memory-side trace lines too;
/// `RUST_LOG` wins. With `logged` the lines are kept for [`open_log`], serve's records where the terminal shows less.
pub fn install(verbose: u8, records: bool, logged: bool) {
    const RECORDS: &str = "warn,oscar=info,oscar_runtime=warn,oscar_runtime::serve=info,oscar_runtime::chat=info,\
                           oscar_runtime::kv_store=info";
    const CONVERSATION: &str = "warn,oscar=info,oscar_runtime=warn,oscar_runtime::kv_store=warn,\
                                oscar::controller=warn,oscar::crossing=warn,oscar::experts=warn,oscar::kv_pool=warn";
    VERBOSE.store(verbose > 0, Ordering::Relaxed);
    if verbose > 1 {
        MEMORY_TRACE.store(true, Ordering::Relaxed);
    }
    let shown = if verbose > 0 {
        "info"
    } else if std::env::var("MISTRALRS_DEBUG").unwrap_or_default().contains('1') {
        "debug"
    } else if records {
        RECORDS
    } else {
        CONVERSATION
    };
    let kept = if shown == CONVERSATION { RECORDS } else { shown };
    let filter = |default| EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default));
    let logged = LOGGED.fetch_and(logged, Ordering::Relaxed) && logged;
    static WRITER: std::sync::Once = std::sync::Once::new();
    WRITER.call_once(|| {
        let spawned = std::thread::Builder::new()
            .name("oscar-console".into())
            .spawn(|| TERMINAL.run(&mut std::io::stderr(), &mut std::io::stdout(), stderr_is_terminal()));
        RUNNING.store(spawned.is_ok(), Ordering::Release);
    });
    let log = logged.then(|| layer(keep, false, filter(kept)));
    let _ = tracing_subscriber::registry().with(layer(show, true, filter(shown))).with(log).try_init();
}

/// The lines `filter` lets through, formatted and handed `to` a writer.
fn layer<S>(to: fn(Vec<u8>), tinted: bool, filter: EnvFilter) -> impl Layer<S>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    let lines = tracing_subscriber::fmt::layer().event_format(Console).with_ansi(tinted);
    lines.with_writer(move || Line(Vec::new(), to)).with_filter(filter)
}

/// The most one log file holds, and how many are kept: `oscar.log` the run's own, `oscar.log.1` to `.9` those before.
const LOG_BYTES: u64 = 16 << 20;
const LOGS: u32 = 10;

/// Off for the eval, whose runs keep their own records, and wherever [`install`] was told so.
static LOGGED: AtomicBool = AtomicBool::new(true);

pub fn set_logged(on: bool) {
    LOGGED.store(on, Ordering::Relaxed);
}

static LOG: Terminal = Terminal::new("the disk was not writing");
/// The log's descriptor once its writer runs, the same for the process's life.
static LOG_FD: AtomicI32 = AtomicI32::new(-1);

/// A log file, appended to; a full one starts the next.
pub struct Log {
    path: PathBuf,
    file: std::fs::File,
    written: u64,
}

impl Log {
    /// A run's own, empty: the ones before it move up a number and the oldest goes.
    pub fn start(path: &Path) -> std::io::Result<Self> {
        let numbered = |n| if n == 0 { path.to_path_buf() } else { path.with_extension(format!("log.{n}")) };
        (0..LOGS - 1).rev().for_each(|n| drop(std::fs::rename(numbered(n), numbered(n + 1))));
        Self::resume(path)
    }

    /// The current one, as it stands.
    pub fn resume(path: &Path) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new().append(true).create(true).open(path)?;
        Ok(Self { path: path.to_path_buf(), written: file.metadata()?.len(), file })
    }
}

impl Write for Log {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self.written + bytes.len() as u64 > LOG_BYTES {
            let next = Self::start(&self.path)?;
            // SAFETY: both descriptors are open files of this process; the next file takes this one's number, which
            // [`urgent`] may hold.
            if unsafe { libc::dup2(next.file.as_raw_fd(), self.file.as_raw_fd()) } < 0 {
                return Err(std::io::Error::last_os_error());
            }
            self.written = 0;
        }
        self.file.write_all(bytes)?;
        self.written += bytes.len() as u64;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// A model's run keeps its lines in `~/.oscar/oscar.log` from here on, those said before included.
pub fn open_log() {
    if !LOGGED.load(Ordering::Relaxed) {
        return;
    }
    let path = super::settings::dir().join("oscar.log");
    match std::fs::create_dir_all(super::settings::dir()).and_then(|_| Log::start(&path)) {
        Ok(mut log) => {
            let fd = log.file.as_raw_fd();
            let spawned = std::thread::Builder::new()
                .name("oscar-log".into())
                .spawn(move || LOG.run(&mut log, &mut std::io::sink(), false));
            LOG_FD.store(if spawned.is_ok() { fd } else { -1 }, Ordering::Release);
        }
        Err(e) => tracing::warn!(target: "oscar::loader", "No log is kept: {}: {e}", path.display()),
    }
}

/// A paint of the status bar, run by the writer against stdout.
pub type Paint = Box<dyn FnOnce(&mut dyn Write) + Send>;

/// The most an unread terminal holds back (a thousand-odd requests' records), the oldest let go first and counted.
const HELD: usize = 1 << 20;

/// The wait for the writer at the end: a reading terminal takes a full queue in milliseconds, an unread one never will.
const ENDING: Duration = Duration::from_secs(1);

static TERMINAL: Terminal = Terminal::new("the terminal was not reading");
static RUNNING: AtomicBool = AtomicBool::new(false);

/// What a writer has been handed, and why it may fall behind.
struct Terminal {
    queue: Mutex<Queue>,
    work: Condvar,
    idle: Condvar,
    held_up: &'static str,
}

struct Queue {
    lines: VecDeque<Vec<u8>>,
    bytes: usize,
    /// Lines let go since the writer last said so.
    dropped: u64,
    /// A later paint replaces it, as a paint draws the bar as it is.
    paint: Option<Paint>,
    /// The bar's rows are given back: no paint runs after.
    closed: bool,
    prompt: Option<&'static str>,
    prompt_due: bool,
    /// The writer holds work it took off the queue and has not finished.
    busy: bool,
}

impl Terminal {
    const fn new(held_up: &'static str) -> Self {
        let queue = Queue {
            lines: VecDeque::new(),
            bytes: 0,
            dropped: 0,
            paint: None,
            closed: false,
            prompt: None,
            prompt_due: false,
            busy: false,
        };
        Self { queue: Mutex::new(queue), work: Condvar::new(), idle: Condvar::new(), held_up }
    }

    fn lock(&self) -> MutexGuard<'_, Queue> {
        self.queue.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn push(&self, line: Vec<u8>) {
        let mut q = self.lock();
        q.bytes += line.len();
        q.lines.push_back(line);
        while q.bytes > HELD {
            let Some(old) = q.lines.pop_front() else { break };
            q.bytes -= old.len();
            q.dropped += 1;
        }
        self.work.notify_one();
    }

    fn paint(&self, paint: Paint, last: bool) {
        let mut q = self.lock();
        if !q.closed {
            q.paint = Some(paint);
            q.closed = last;
            self.work.notify_one();
        }
    }

    fn prompt(&self, text: Option<&'static str>) {
        let mut q = self.lock();
        q.prompt = text;
        q.prompt_due = text.is_some();
        self.work.notify_one();
    }

    /// Until everything handed over is written, or `bound` has passed.
    fn drain(&self, bound: Duration) {
        let deadline = Instant::now() + bound;
        let mut q = self.lock();
        while q.busy || !q.lines.is_empty() || q.paint.is_some() {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return;
            }
            q = self.idle.wait_timeout(q, left).unwrap_or_else(PoisonError::into_inner).0;
        }
    }

    /// Lines in order, then the bar's latest paint, then a due prompt, which a line under it clears and redraws.
    fn run(&self, err: &mut dyn Write, out: &mut dyn Write, clear: bool) {
        let mut q = self.lock();
        loop {
            q.busy = false;
            while q.lines.is_empty() && q.paint.is_none() && !q.prompt_due {
                self.idle.notify_all();
                q = self.work.wait(q).unwrap_or_else(PoisonError::into_inner);
            }
            q.busy = true;
            let dropped = std::mem::take(&mut q.dropped);
            let line = q.lines.pop_front();
            q.bytes -= line.as_ref().map_or(0, Vec::len);
            let paint = if line.is_none() { q.paint.take() } else { None };
            let at_prompt = q.prompt.filter(|_| line.is_some() || dropped > 0);
            if at_prompt.is_some() {
                q.prompt_due = true;
            }
            let prompt = if line.is_none() && paint.is_none() && q.prompt_due {
                q.prompt_due = false;
                q.prompt
            } else {
                None
            };
            drop(q);
            if at_prompt.is_some() && clear {
                let _ = err.write_all(b"\r\x1b[2K");
            }
            if dropped > 0 {
                let (on, off) = tint(&Level::WARN).filter(|_| clear).map_or(("", ""), |t| (t, "\x1b[0m"));
                let (head, why) = (Head(&component("oscar_runtime")), self.held_up);
                let _ = writeln!(err, "{head}{on}WARN: Log lines were dropped: {dropped} not shown, {why}{off}");
            }
            if let Some(line) = line {
                let _ = err.write_all(&line);
            }
            if let Some(paint) = paint {
                paint(out);
                let _ = out.flush();
            }
            if let Some(prompt) = prompt {
                let _ = out.write_all(prompt.as_bytes());
                let _ = out.flush();
            }
            q = self.lock();
        }
    }
}

/// One record on its way to a writer, written whole and handed over on drop.
struct Line(Vec<u8>, fn(Vec<u8>));

impl Write for Line {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Drop for Line {
    fn drop(&mut self) {
        (self.1)(std::mem::take(&mut self.0));
    }
}

/// The writer, while it runs; else each hand-off writes here itself.
fn writer() -> Option<&'static Terminal> {
    RUNNING.load(Ordering::Acquire).then_some(&TERMINAL)
}

/// Hand the writer bytes for stderr; with no writer running, write them here.
fn show(bytes: Vec<u8>) {
    match writer() {
        _ if bytes.is_empty() => {}
        Some(w) => w.push(bytes),
        None => drop(std::io::stderr().write_all(&bytes)),
    }
}

/// Hand the log's writer bytes, where a log is kept.
pub fn keep(bytes: Vec<u8>) {
    if LOGGED.load(Ordering::Relaxed) && !bytes.is_empty() {
        LOG.push(bytes);
    }
}

/// Bytes for stderr and the log.
pub fn say(bytes: Vec<u8>) {
    keep(bytes.clone());
    show(bytes);
}

/// Hand the writer the bar's next paint; with no writer running it runs here.
pub fn paint(paint: Paint) {
    match writer() {
        Some(w) => w.paint(paint, false),
        None => {
            let mut out = std::io::stdout();
            paint(&mut out);
            let _ = out.flush();
        }
    }
}

/// Raise (`Some`) or leave (`None`) the prompt, drawn after the lines before it and under it; with no writer, here.
pub fn prompt(text: Option<&'static str>) {
    match (writer(), text) {
        (Some(w), text) => w.prompt(text),
        (None, Some(text)) => {
            let mut out = std::io::stdout();
            let _ = out.write_all(text.as_bytes());
            let _ = out.flush();
        }
        (None, None) => {}
    }
}

pub fn prompt_up() -> bool {
    TERMINAL.lock().prompt.is_some()
}

/// The bar's last paint (its rows given back), after which none runs; then [`drain`].
pub fn close(last: Paint) {
    match writer() {
        Some(w) => {
            w.paint(last, true);
            drain();
        }
        None => paint(last),
    }
}

/// Wait at most [`ENDING`] for each writer, so an ending process's last lines reach the terminal and the log.
pub fn drain() {
    if let Some(w) = writer() {
        w.drain(ENDING);
    }
    if LOG_FD.load(Ordering::Acquire) >= 0 {
        LOG.drain(ENDING);
    }
}

/// A line that cannot block, for a process about to end: built on the stack, written once with stderr non-blocking,
/// then once to the log. `eprintln!` allocates and locks stderr, either of which a starved machine can block on for
/// ever.
pub fn urgent(head: &str, error: bool, what: fmt::Arguments) {
    let tinted = tint(&Level::ERROR).filter(|_| error).map_or(("", ""), |on| (on, "\x1b[0m"));
    for (fd, (on, off)) in [(2, tinted), (LOG_FD.load(Ordering::Acquire), ("", ""))] {
        let mut buf = [0u8; 256];
        let mut line = std::io::Cursor::new(&mut buf[..]);
        let _ = writeln!(line, "{}{on}{}{what}{off}", Head(head), if error { "ERROR: " } else { "" });
        let len = line.position() as usize;
        // SAFETY: fcntl on this process's own descriptor and one write of a buffer on this stack; nothing is read
        // back, and a descriptor of -1 (no log) fails both.
        unsafe {
            let flags = libc::fcntl(fd, libc::F_GETFL);
            if flags >= 0 {
                libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
            }
            libc::write(fd, buf.as_ptr().cast(), len);
            if flags >= 0 {
                libc::fcntl(fd, libc::F_SETFL, flags);
            }
        }
    }
}

static MEMORY_TRACE: AtomicBool = AtomicBool::new(false);

/// The eval's, at its install: memory lines then carry a wall-clock stamp.
pub fn set_memory_trace(on: bool) {
    MEMORY_TRACE.store(on, Ordering::Relaxed);
}

/// The gate of every memory-side eval line.
pub fn traced() -> bool {
    MEMORY_TRACE.load(Ordering::Relaxed)
}

/// ` t=<epoch ms>` under the memory trace, else empty: what memory lines join an instrument's log on.
pub fn stamp() -> String {
    let ms = || SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis());
    if traced() {
        format!(" t={}", ms())
    } else {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    /// A terminal that stopped reading: its first write waits for the test's release unless `.2`; all is kept.
    struct Stalled(Arc<Barrier>, Arc<Mutex<Vec<u8>>>, bool);

    impl Write for Stalled {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if !self.2 {
                self.2 = true;
                self.0.wait();
            }
            self.1.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// A writer thread on its own terminal, its stderr stalled until the gate unless `reading`; what both received.
    fn writer(reading: bool, clear: bool) -> (&'static Terminal, Arc<Barrier>, impl Fn() -> String) {
        let terminal: &'static Terminal = Box::leak(Box::new(Terminal::new("the terminal was not reading")));
        let (gate, seen) = (Arc::new(Barrier::new(2)), Arc::new(Mutex::new(Vec::new())));
        let mut err = Stalled(gate.clone(), seen.clone(), reading);
        let mut out = Stalled(gate.clone(), seen.clone(), true);
        std::thread::spawn(move || terminal.run(&mut err, &mut out, clear));
        (terminal, gate, move || String::from_utf8(seen.lock().unwrap().clone()).unwrap())
    }

    /// A terminal that is not reading stops the writer alone; a line at the prompt clears it and the prompt returns;
    /// a duration that was not nothing never reads zero.
    #[test]
    fn a_terminal_that_is_not_reading_stops_the_writer_alone() {
        let (terminal, gate, written) = writer(false, false);
        terminal.push(b"first\n".to_vec());
        // The writer must hold the first line before the flood, or it would be let go as the oldest.
        while !terminal.lock().busy {
            std::thread::yield_now();
        }
        let began = Instant::now();
        (0..4096).for_each(|_| terminal.push([[b'x'; 1023].as_slice(), b"\n"].concat()));
        (0..3u8).for_each(|n| terminal.paint(Box::new(move |out| drop(out.write_all(&[b'0' + n, b'\n']))), false));
        terminal.paint(Box::new(|out| drop(out.write_all(b"rows back\n"))), true);
        terminal.paint(Box::new(|out| drop(out.write_all(b"after close\n"))), false);
        assert!(began.elapsed() < Duration::from_millis(500), "a hand-over waited on the terminal");
        let held = terminal.lock().bytes;
        assert!(held <= HELD && held > HELD - 2048, "the newest megabyte is kept: {held}");
        gate.wait();
        terminal.drain(Duration::from_secs(10));
        let seen = written();
        assert!(seen.starts_with("first\n"), "the line the writer held comes first");
        let warn = seen.lines().nth(1).unwrap();
        assert!(warn.contains("WARN: Log lines were dropped: 3072 not shown, the terminal was not reading"), "{warn}");
        assert_eq!(seen.lines().filter(|l| l.starts_with('x')).count(), 1024);
        assert!(seen.ends_with("rows back\n") && !seen.contains("after close"), "the last paint only");
        assert!(!seen.contains("0\n1\n"), "a paint replaced by a later one is not drawn");
        // The prompt.
        let (terminal, _, written) = writer(true, true);
        let settle = |want: &str| {
            let began = Instant::now();
            while !written().ends_with(want) && began.elapsed() < Duration::from_secs(5) {
                std::thread::yield_now();
            }
            assert!(written().ends_with(want), "written: {:?}", written());
        };
        terminal.push(b"before\n".to_vec());
        terminal.prompt(Some("you> "));
        settle("before\nyou> ");
        terminal.push(b"meanwhile\n".to_vec());
        settle("before\nyou> \r\x1b[2Kmeanwhile\nyou> ");
        terminal.prompt(None);
        terminal.push(b"after\n".to_vec());
        settle("meanwhile\nyou> after\n");
        // A duration that was not nothing never reads zero.
        let secs = [0.0, 0.00002, 0.0003, 0.00094, 0.00096, 0.0012, 0.4, 0.9996, 3.24, 224.0];
        let said = ["0 ms", "<0.1 ms", "0.3 ms", "0.9 ms", "1 ms", "1 ms", "400 ms", "1.0 s", "3.2 s", "224.0 s"];
        secs.into_iter()
            .zip(said)
            .for_each(|(s, said)| assert_eq!(mistralrs_core::live::duration_text(s), said, "{s}"));
    }

    /// Each run's log moves the ones before it up a number, the tenth going; a full file starts the next on the same
    /// descriptor, and a resumed one is appended to.
    #[test]
    fn ten_logs_are_kept_and_none_grows_past_its_bound() {
        let dir = crate::model_store::fixtures::TempDir::new("log");
        let path = dir.0.join("oscar.log");
        let read = |n: u32| std::fs::read_to_string(dir.0.join(format!("oscar.log.{n}"))).ok();
        for run in 0..12 {
            Log::start(&path).unwrap().write_all(format!("run {run}\n").as_bytes()).unwrap();
        }
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "run 11\n");
        assert_eq!((read(1).as_deref(), read(9).as_deref(), read(10)), (Some("run 10\n"), Some("run 2\n"), None));
        let mut log = Log::resume(&path).unwrap();
        let (fd, line) = (log.file.as_raw_fd(), [b'x'; 1 << 20]);
        (0..17).for_each(|_| log.write_all(&line).unwrap());
        assert_eq!((log.file.as_raw_fd(), log.written), (fd, 2 << 20));
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 2 << 20);
        assert_eq!(read(1).map(|s| s.len()), Some("run 11\n".len() + (15 << 20)));
        assert_eq!((read(2).as_deref(), read(9).as_deref()), (Some("run 10\n"), Some("run 3\n")));
    }
}
