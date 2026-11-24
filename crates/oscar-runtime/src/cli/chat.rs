//! Streaming chat loop for the `chat` subcommand, and its transcript's markup. Through the server when one answers on
//! the port and no flag names a model (the app's, say), so the model is not loaded twice; else over its own model.

use std::io::{self, BufRead, IsTerminal, Write};
use std::time::{Duration, Instant};

use anyhow::Result;
use mistralrs::{RequestBuilder, Response, TextMessageRole, TextMessages};
use tokio::sync::mpsc;

use super::args::{sampling_params, ChatArgs};
use super::ui::StatusBar;
use crate::cli::build::build_model;

/// One streamed step from either source; an error carries whether it is a defect (else a refused request).
struct Delta {
    reasoning: Option<String>,
    content: Option<String>,
    finish: Option<String>,
    usage: Option<Use>,
}

/// A reply's usage, or a session's summed.
struct Use {
    prompt: f64,
    cached: f64,
    completion: f64,
    prompt_s: f64,
    completion_s: f64,
    total_s: f64,
}

/// The prompt's and the completion's figures, as serve's record states each request's.
impl std::fmt::Display for Use {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Use { prompt, cached, completion, .. } = self;
        write!(
            f,
            "prompt {prompt} = {cached} cached + {} new @ {:.0} tok/s | completion {completion} @ {:.1} tok/s",
            prompt - cached,
            (prompt - cached) / self.prompt_s.max(1e-9),
            completion / self.completion_s.max(1e-9),
        )
    }
}

type Failed = (String, bool);

/// The engine's own stream, which borrows the model, or the server's channel.
enum Source<'a> {
    Local(mistralrs::Stream<'a>),
    Remote(mpsc::Receiver<Result<Delta, Failed>>),
}

impl Source<'_> {
    async fn next(&mut self) -> Option<Result<Delta, Failed>> {
        match self {
            Source::Remote(rx) => rx.recv().await,
            Source::Local(stream) => loop {
                break Some(match stream.next().await? {
                    Response::Chunk(c) => {
                        let Some(ch) = c.choices.first() else { continue };
                        Ok(Delta {
                            reasoning: ch.delta.reasoning_content.clone(),
                            content: ch.delta.content.clone(),
                            finish: ch.finish_reason.clone(),
                            usage: c.usage.as_ref().map(|u| Use {
                                prompt: u.prompt_tokens as f64,
                                cached: u.prompt_tokens_details.cached_tokens as f64,
                                completion: u.completion_tokens as f64,
                                prompt_s: u.total_prompt_time_sec as f64,
                                completion_s: u.total_completion_time_sec as f64,
                                total_s: u.total_time_sec as f64,
                            }),
                        })
                    }
                    Response::ModelError(err, _) => Err((format!("model error - {err}"), true)),
                    Response::InternalError(err) => Err((err.to_string(), true)),
                    Response::ValidationError(err) => Err((err.to_string(), false)),
                    _ => continue,
                });
            },
        }
    }
}

/// The server's event stream as deltas: `data:` lines of chunks, `[DONE]` the end.
fn remote(base: String, body: serde_json::Value) -> mpsc::Receiver<Result<Delta, Failed>> {
    let (tx, rx) = mpsc::channel(64);
    tokio::spawn(async move {
        let sent = async {
            let resp = reqwest::Client::new().post(format!("{base}/v1/chat/completions")).json(&body).send().await?;
            if !resp.status().is_success() {
                let status = resp.status();
                return Err(anyhow::anyhow!("{}: {}", status, resp.text().await.unwrap_or_default()));
            }
            Ok(resp)
        };
        let mut resp = match sent.await {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(Err((e.to_string(), false))).await;
                return;
            }
        };
        let mut buf = String::new();
        while let Ok(Some(bytes)) = resp.chunk().await {
            buf.push_str(&String::from_utf8_lossy(&bytes));
            while let Some(at) = buf.find('\n') {
                let line = buf[..at].trim().to_string();
                buf.drain(..=at);
                let Some(data) = line.strip_prefix("data:") else { continue };
                let data = data.trim();
                if data == "[DONE]" {
                    return;
                }
                let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else { continue };
                let text = |k: &str| v["choices"][0]["delta"][k].as_str().map(str::to_string);
                let num = |k: &[&str]| k.iter().fold(&v["usage"], |u, k| &u[k]).as_f64().unwrap_or(0.0);
                let delta = Delta {
                    reasoning: text("reasoning_content"),
                    content: text("content"),
                    finish: v["choices"][0]["finish_reason"].as_str().map(str::to_string),
                    usage: v["usage"].is_object().then(|| Use {
                        prompt: num(&["prompt_tokens"]),
                        cached: num(&["prompt_tokens_details", "cached_tokens"]),
                        completion: num(&["completion_tokens"]),
                        prompt_s: num(&["total_prompt_time_sec"]),
                        completion_s: num(&["total_completion_time_sec"]),
                        total_s: num(&["total_time_sec"]),
                    }),
                };
                if tx.send(Ok(delta)).await.is_err() {
                    return;
                }
            }
        }
    });
    rx
}

/// The server on the port or the next nine (serve takes the first free one): one answering as oscar, or one still
/// loading (the port bound, no answer yet) waited for; none when no port is oscar's.
async fn served(port: u16) -> Option<String> {
    let client = reqwest::Client::new();
    let mut waiting = false;
    'ports: for p in port..port + 10 {
        let base = format!("http://127.0.0.1:{p}");
        loop {
            match client.get(format!("{base}/v1/models")).timeout(Duration::from_millis(300)).send().await {
                // Ollama answers /v1/models too; oscar's list has the `+nothink` sibling.
                Ok(r) => match r.text().await.ok().filter(|t| t.contains("+nothink")) {
                    Some(_) => return Some(base),
                    None => continue 'ports,
                },
                Err(e) if e.is_connect() => continue 'ports,
                Err(_) => {}
            }
            if !waiting {
                tracing::info!(target: "oscar::chat", "waiting for the server at {base}/v1 to finish loading ...");
                waiting = true;
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }
    None
}

pub async fn chat_main(args: ChatArgs) -> Result<()> {
    let mut stdout = io::stdout();
    let settings = super::settings::Settings::load()?;
    let base = match args.common.model {
        None => served(settings.port(None)).await,
        Some(_) => None,
    };
    // Through the server: its model, no bar and no line of ours (the app's menu shows the figures); over a model
    // of our own, the bar is ours.
    let (built, preset) = match &base {
        Some(_) => (None, settings.preset(None, None)?.0),
        None => {
            let built = build_model(&args.common, false, false).await?;
            tracing::info!(target: "oscar::chat", "ready - Enter sends, Ctrl-D exits\n");
            let preset = built.preset;
            (Some(built), preset)
        }
    };
    // The model's shipped row, for a mode it publishes none for.
    let shipped = built.as_ref().and_then(|b| b.model.config().ok()).and_then(|c| c.generation_defaults);
    let bar = built.as_ref().map_or_else(StatusBar::default, |b| b.bar.0.clone());

    let system = args.system.as_deref().or_else(|| preset.def().default_system).map(str::to_string);
    // (role, content, reasoning) in order; rebuilt into either source's form each turn.
    let mut history: Vec<(TextMessageRole, String, String)> = Vec::new();
    // Each reply's usage, summed into the session's line at Ctrl-D.
    let mut usages: Vec<Use> = Vec::new();

    let stdin = io::stdin();
    let tty = io::stdout().is_terminal();
    // Ctrl-C or SIGTERM; chat reads its own EOF.
    super::ui::exit_on_signal(bar.clone(), false);

    loop {
        // The console's lines go out before the user types.
        crate::cli::console::drain();
        crate::cli::console::prompt(Some("you> "));
        let mut buf = String::new();
        let n = stdin.lock().read_line(&mut buf)?;
        crate::cli::console::prompt(None);
        if n == 0 {
            println!(); // EOF: the shell's prompt on its own line
            break;
        }
        let user_text = buf.trim_end_matches(['\n', '\r']).to_string();
        if user_text.is_empty() {
            continue;
        }
        // Back to an independent conversation, generating nothing.
        if user_text == "/clear" || user_text == "/reset" {
            history.clear();
            continue;
        }
        history.push((TextMessageRole::User, user_text, String::new()));

        let mut source = match (&base, &built) {
            (Some(base), _) => {
                let role = |r: &TextMessageRole| match r {
                    TextMessageRole::User => "user",
                    TextMessageRole::Assistant => "assistant",
                    _ => "system",
                };
                let messages: Vec<serde_json::Value> = system
                    .iter()
                    .map(|s| serde_json::json!({ "role": "system", "content": s }))
                    .chain(history.iter().map(|(r, c, _)| serde_json::json!({ "role": role(r), "content": c })))
                    .collect();
                let mut body = serde_json::json!({
                    "model": if args.no_think { "default+nothink" } else { "default" },
                    "messages": messages,
                    "stream": true,
                    "stream_options": { "include_usage": true },
                    "max_tokens": args.max_len,
                });
                if let Some(t) = args.temperature {
                    body["temperature"] = t.into();
                }
                if let Some(effort) = args.reasoning_effort.as_deref() {
                    body["reasoning_effort"] = effort.into();
                }
                Source::Remote(remote(base.clone(), body))
            }
            (None, Some(built)) => {
                let mut messages = TextMessages::new();
                if let Some(s) = &system {
                    messages = messages.add_message(TextMessageRole::System, s);
                }
                for (r, c, reasoning) in &history {
                    messages = messages.add_message_with_reasoning(r.clone(), c, reasoning);
                }
                // `enable_thinking` goes on the builder, as `From<TextMessages>` resets the messages' value to `None`.
                let mut req = RequestBuilder::from(messages).set_sampling(sampling_params(&args, shipped.as_ref()));
                if args.no_think {
                    req = req.enable_thinking(false);
                }
                if let Some(effort) = args.reasoning_effort.as_deref() {
                    req = req.reasoning_effort(match effort {
                        "low" => mistralrs_core::ReasoningEffort::Low,
                        "high" => mistralrs_core::ReasoningEffort::High,
                        _ => mistralrs_core::ReasoningEffort::Medium,
                    });
                }
                let stream = built
                    .model
                    .stream_chat_request(req)
                    .await
                    .map_err(|e| anyhow::anyhow!("stream_chat_request: {e}"))?;
                Source::Local(stream)
            }
            (None, None) => unreachable!("a chat has a server or a model"),
        };

        const PROMPT: &str = "oscar> ";
        let mut assistant_text = String::new();
        // Kept for the history, so the next prompt is the one serve would build.
        let mut reasoning_text = String::new();
        let mut md = InlineMd::default();
        let mut content_started = false;
        let mut reasoning_started = false;
        let mut finish_reason: Option<String> = None;

        let mut thinking_since = Instant::now();
        let mut last_think = Instant::now();
        let (mut decoded, mut used) = (false, false);

        const CLOCK: Duration = Duration::from_millis(200);
        // A failed reply ends its partial line, then says why: an error for a defect, a warning for a refused request.
        let failed = |text: String, defect: bool| {
            println!();
            if defect {
                tracing::error!(target: "oscar::chat", "{text}")
            } else {
                tracing::warn!(target: "oscar::chat", "{text}")
            }
        };

        loop {
            // A tight timeout, so the clock ticks between chunks.
            let next = tokio::time::timeout(CLOCK, source.next()).await;
            let delta = match next {
                Ok(Some(Ok(d))) => Some(d),
                Ok(Some(Err((text, defect)))) => break failed(text, defect),
                Ok(None) => break, // the source closed
                Err(_) => None,    // a timeout: the clock only
            };

            if let Some(d) = delta {
                decoded |= d.content.is_some() || d.reasoning.is_some();
                // Shown raw, as markdown would hold half-open emphasis into the answer.
                if let Some(text) = d.reasoning.as_ref() {
                    reasoning_text.push_str(text);
                    if !content_started {
                        if !reasoning_started {
                            reasoning_started = true;
                            thinking_since = Instant::now();
                            if args.show_thinking {
                                print!("\x1b[2m");
                            }
                        }
                        if args.show_thinking {
                            print!("{text}");
                        }
                    }
                }
                if let Some(text) = d.content.as_ref() {
                    assistant_text.push_str(text);
                    // The answer opens at its first visible character, the template's blank lines left out.
                    let shown = if content_started { text.as_str() } else { text.trim_start() };
                    if !shown.is_empty() {
                        if !content_started {
                            if reasoning_started && args.show_thinking {
                                // Close the dim run before the answer.
                                print!("\x1b[0m\n\n");
                            } else if reasoning_started && tty {
                                print!(
                                    "\r\x1b[2K\x1b[2mthought for {:.0} s\x1b[0m\n",
                                    thinking_since.elapsed().as_secs_f64()
                                );
                            }
                            print!("{PROMPT}");
                        }
                        content_started = true;
                        let mut rendered = String::new();
                        md.push_str(shown, &mut rendered);
                        print!("{rendered}");
                    }
                }
                if let Some(u) = d.usage {
                    usages.push(u);
                    used = true;
                }
                if d.finish.is_some() {
                    finish_reason = d.finish;
                    if base.is_none() {
                        break;
                    }
                }
            }

            // The reasoning's clock while it runs unseen, so a long think never looks hung.
            if tty && !args.show_thinking && reasoning_started && !content_started && last_think.elapsed() >= CLOCK {
                print!("\r\x1b[2K\x1b[2mthinking... {:.0} s\x1b[0m", thinking_since.elapsed().as_secs_f64());
                last_think = Instant::now();
            }

            // One flush per chunk, so a fast decode never backpressures the response channel.
            stdout.flush().ok();
        }

        let mut tail = String::new();
        md.finish(&mut tail);
        print!("{tail}");
        // A reasoning-only reply says why, or its line would look empty.
        if let Some(reason) = finish_reason.as_deref().filter(|_| !content_started) {
            let note = match reason {
                "length" => {
                    "[no answer - token budget exhausted while thinking; raise --max-len, or --no-think]".to_string()
                }
                r => format!("[no answer - finish_reason: {r}]"),
            };
            if reasoning_started && args.show_thinking {
                print!("\x1b[0m\n");
            } else if reasoning_started && tty {
                print!("\r\x1b[2K");
            }
            print!("{PROMPT}\x1b[2m{note}\x1b[0m");
        }
        println!();
        stdout.flush().ok();
        // The reply's record over a model of our own, as serve's per request: in the log, on the terminal at -v.
        if let (Some(b), Some(u), Some(finish), true) = (&built, usages.last(), &finish_reason, used) {
            let duration = mistralrs_core::live::duration_text;
            tracing::info!(target: "oscar_runtime::chat",
                "reply        | {} | TTFT {} | {u} | {finish}", duration(u.total_s), duration(u.prompt_s));
            if let Some(w) = b.dispatcher.pool().decode_window().filter(|_| decoded) {
                tracing::info!(target: "oscar_runtime::chat", "{w}");
            }
        }
        history.push((TextMessageRole::Assistant, assistant_text, reasoning_text));
    }

    // The terminal's scrolling and the bar's rows back before the exit.
    drop(built);
    // The session's rates, its replies' figures summed as serve's record states each request's.
    if !usages.is_empty() {
        let sum = |f: fn(&Use) -> f64| usages.iter().map(f).sum::<f64>();
        let all = Use {
            prompt: sum(|u| u.prompt),
            cached: sum(|u| u.cached),
            completion: sum(|u| u.completion),
            prompt_s: sum(|u| u.prompt_s),
            completion_s: sum(|u| u.completion_s),
            total_s: sum(|u| u.total_s),
        };
        let total = mistralrs_core::live::duration_text(all.total_s);
        tracing::info!(target: "oscar::chat", "session - {} replies in {total} | {all}", usages.len());
    }

    // Ctrl-D: persist the store's dirty tail so the next run can resume.
    crate::kv_store::flush_global(std::time::Duration::from_secs(20));
    Ok(())
}

/// Streaming `**bold**` and `` `code` `` to ANSI across token boundaries; the terminal wraps, so a resize reflows.
#[derive(Default)]
struct InlineMd {
    saw_star: bool,
    in_bold: bool,
    in_code: bool,
}

impl InlineMd {
    fn push_str(&mut self, s: &str, out: &mut String) {
        for ch in s.chars() {
            if self.saw_star {
                self.saw_star = false;
                if ch == '*' {
                    self.in_bold = !self.in_bold;
                    out.push_str(if self.in_bold { "\x1b[1m" } else { "\x1b[22m" });
                    continue;
                }
                out.push('*');
            }
            match ch {
                '*' => self.saw_star = true,
                '`' => {
                    self.in_code = !self.in_code;
                    out.push_str(if self.in_code { "\x1b[36m" } else { "\x1b[39m" });
                }
                _ => out.push(ch),
            }
        }
    }

    fn finish(&mut self, out: &mut String) {
        if self.saw_star {
            out.push('*');
            self.saw_star = false;
        }
        self.in_bold = false;
        self.in_code = false;
        out.push_str("\x1b[0m");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toggle may straddle two chunks; a lone star is text; the terminal does the wrapping.
    #[test]
    fn styles_bold_and_code_across_chunks() {
        for (text, want) in [
            ("a big **elephant**", "a big \x1b[1melephant\x1b[22m"),
            ("a 2*3 grid", "a 2*3 grid"),
            ("`code` here", "\x1b[36mcode\x1b[39m here"),
            ("one\n  two three", "one\n  two three"),
        ] {
            let (mut md, mut out) = (InlineMd::default(), String::new());
            let (head, tail) = text.split_at(text.len() / 2);
            md.push_str(head, &mut out);
            md.push_str(tail, &mut out);
            md.finish(&mut out);
            assert_eq!(out.replace("\x1b[0m", ""), want, "{text:?}");
        }
    }
}
