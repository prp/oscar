//! The serving child: `serve` from this binary, its status lines read from its stdout, its stderr appended to the
//! log with the last trouble line kept for the menu.

use std::io::{BufRead, Write};
use std::path::Path;
use std::process::{ExitStatus, Stdio};
use std::sync::{Arc, Mutex};

use anyhow::Context;

use crate::cli::ui::Status;

pub struct Child {
    /// Its stdin a pipe the app holds: serve exits at the EOF should the app die first.
    process: std::process::Child,
    /// Its last status line; none once its stdout has ended.
    status: Arc<Mutex<Option<Status>>>,
    /// The last `WARN:` or `ERROR:` line with its tag, or the error the child ended with, its causes on lines after
    /// it; without the time and component.
    trouble: Arc<Mutex<Option<String>>>,
    /// The port serve took, from its ready line: its wish, or the next free one.
    port: Arc<Mutex<Option<u16>>>,
}

fn ready_port(line: &str) -> Option<u16> {
    line.split_once("http://127.0.0.1:").and_then(|(_, r)| r.split_once("/v1 - ")).and_then(|(p, _)| p.parse().ok())
}

#[cfg(test)]
#[test]
fn the_ready_line_names_the_port() {
    assert_eq!(
        ready_port("12:00:00 [serve]      http://127.0.0.1:11435/v1 - OpenAI-compatible; model ids"),
        Some(11435)
    );
    assert_eq!(ready_port("12:00:00 [serve]      request done | 1 s"), None);
}

/// `each` with every line of `from` on a thread of its own, then with `None` at its end.
fn lines(
    name: &str, from: impl std::io::Read + Send + 'static, mut each: impl FnMut(Option<String>) + Send + 'static,
) -> anyhow::Result<()> {
    std::thread::Builder::new()
        .name(name.into())
        .spawn(move || {
            for line in std::io::BufReader::new(from).lines().map_while(Result::ok) {
                each(Some(line));
            }
            each(None);
        })
        .with_context(|| format!("the {name} thread"))?;
    Ok(())
}

/// The trouble a log line leaves: a `WARN:` or `ERROR:` line whole, an `Error:` line with each cause under it (its
/// number off) on a line of its own.
fn keep(kept: &mut Option<String>, line: &str) {
    if line.contains("WARN: ") || line.contains("ERROR: ") || line.starts_with("Error: ") {
        *kept = Some(line.to_string());
    } else if let (Some(cause), Some(error)) =
        (line.strip_prefix("    "), kept.as_mut().filter(|k| k.starts_with("Error: ")))
    {
        let cause = cause.split_once(": ").filter(|(n, _)| n.parse::<u8>().is_ok()).map_or(cause, |(_, c)| c);
        error.push('\n');
        error.push_str(cause);
    }
}

#[cfg(test)]
#[test]
fn the_trouble_is_the_last_tagged_line_or_the_error_with_its_causes() {
    let mut kept = None;
    let log =
        "12:00:00 [a] fine\n12:00:01 [planner]    WARN: refused: x\nError: GgufModelBuilder::build\n\nCaused by:\n    \
               0: Metal error y: z\n    1: z";
    log.lines().for_each(|line| keep(&mut kept, line));
    assert_eq!(kept.as_deref(), Some("Error: GgufModelBuilder::build\nMetal error y: z\nz"));
    assert_eq!(headline(kept.as_deref().unwrap()), (false, "z"));
    assert_eq!(headline("Error: the model routes no experts"), (false, "the model routes no experts"));
    keep(&mut kept, "12:00:02 [b] ERROR: bad: the detail");
    keep(&mut kept, "    KV index 0.006 GB");
    assert_eq!(kept.as_deref(), Some("12:00:02 [b] ERROR: bad: the detail"));
    assert_eq!(headline(kept.as_deref().unwrap()), (false, "bad"));
    assert_eq!(
        headline("12:00:03 [c] WARN: Memory is running low: 0.8 GB under the floor"),
        (true, "Memory is running low")
    );
}

/// The words the menu shows: a warning's or error's message up to its first colon, an error block's innermost cause,
/// a plain `Error:` line's message; `true` for a warning.
pub fn headline(entry: &str) -> (bool, &str) {
    let tagged = entry.find("WARN: ").or_else(|| entry.find("ERROR: ")).map(|at| &entry[at..]);
    let message = tagged.and_then(|t| t.split_once(": ")).map_or(entry.lines().last().unwrap_or_default(), |(_, m)| m);
    let message = message.strip_prefix("Error: ").unwrap_or(message);
    (tagged.is_some_and(|t| t.starts_with("WARN")), message.split_once(": ").map_or(message, |(h, _)| h))
}

impl Child {
    /// `serve --model <row> --status-lines`; its lines to `log`.
    pub fn spawn(model: &str, quant: Option<&str>, log: &Path) -> anyhow::Result<Self> {
        let mut cmd = std::process::Command::new(std::env::current_exe()?);
        cmd.args(["serve", "--model", model]);
        if let Some(quant) = quant {
            cmd.args(["--quant", quant]);
        }
        cmd.arg("--status-lines");
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut process = cmd.spawn().context("starting oscar serve")?;
        let (stdout, stderr) = (process.stdout.take().expect("piped"), process.stderr.take().expect("piped"));
        let status = Arc::new(Mutex::new(None));
        let latest = status.clone();
        lines("oscar-status", stdout, move |line| {
            *latest.lock().unwrap_or_else(|e| e.into_inner()) = line.and_then(|l| serde_json::from_str(&l).ok());
        })?;
        let mut file = std::fs::OpenOptions::new().append(true).create(true).open(log)?;
        let trouble = Arc::new(Mutex::new(None));
        let port = Arc::new(Mutex::new(None));
        let (kept, bound) = (trouble.clone(), port.clone());
        lines("oscar-log", stderr, move |line| {
            let Some(line) = line else { return };
            keep(&mut kept.lock().unwrap_or_else(|e| e.into_inner()), &line);
            if let Some(p) = ready_port(&line) {
                *bound.lock().unwrap_or_else(|e| e.into_inner()) = Some(p);
            }
            let _ = writeln!(file, "{line}");
        })?;
        Ok(Self { process, status, trouble, port })
    }

    pub fn status(&self) -> Option<Status> {
        self.status.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn trouble(&self) -> Option<String> {
        self.trouble.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn port(&self) -> Option<u16> {
        *self.port.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn exited(&mut self) -> Option<ExitStatus> {
        self.process.try_wait().ok().flatten()
    }

    /// SIGTERM: the store flushes, up to 20 s, then it ends; `exited` says when.
    pub fn terminate(&self) {
        unsafe { libc::kill(self.process.id() as i32, libc::SIGTERM) };
    }

    /// Ended, waited for.
    pub fn stop(mut self) {
        self.terminate();
        let _ = self.process.wait();
    }
}
