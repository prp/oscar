//! `serve`: the OpenAI-compatible endpoint over the model `chat` builds, one sequence at a time, the rest queued.

use anyhow::{Context, Result};
use mistralrs_server_core::mistralrs_server_router_builder::MistralRsServerRouterBuilder;

use super::args::ServeArgs;
use crate::cli::build::build_model;

/// `wanted`, or the next free port within ten of it: 11434 is Ollama's too; chat and the menu look the same way.
async fn bind(host: &str, wanted: u16) -> Result<(tokio::net::TcpListener, u16)> {
    for port in wanted..wanted + 10 {
        match tokio::net::TcpListener::bind((host, port)).await {
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {}
            bound => return Ok((bound.with_context(|| format!("[serve] binding {host}:{port}"))?, port)),
        }
    }
    anyhow::bail!("[serve] ports {wanted} to {} are all taken", wanted + 9)
}

#[cfg(test)]
#[test]
fn a_taken_port_moves_serve_to_the_next_free_one() {
    let taken = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let wanted = taken.local_addr().unwrap().port();
    let (_, port) = tokio::runtime::Runtime::new().unwrap().block_on(bind("127.0.0.1", wanted)).unwrap();
    assert!(port > wanted && port < wanted + 10, "{port} for {wanted}");
}

pub async fn serve_main(args: ServeArgs) -> Result<()> {
    // Bound before the build, so a taken port is known at once; an early client waits in the backlog.
    let wanted = super::settings::Settings::load()?.port(args.port);
    let (listener, port) = bind(&args.host, wanted).await?;
    let addr = format!("{}:{port}", args.host);
    if port != wanted {
        tracing::info!(target: "oscar::serve", "port {wanted} is taken, serving on {port}");
    }
    let built = build_model(&args.common, true, args.status_lines).await?;
    // The record's second line: what the request cost below the API, which only the dispatcher's samples know.
    let pd = built.dispatcher.clone();
    mistralrs_server_core::util::set_request_epilogue(Box::new(move || {
        text::flush();
        let w = pd.pool().decode_window()?;
        crate::engine::eval::hooks().request_end(&w);
        if let Some(d) = w.depth_line() {
            tracing::info!(target: "oscar::serve", "{d}");
        }
        Some(w.to_string())
    }));
    mistralrs_server_core::util::set_text_tap(Box::new(text::tap));
    text::toggle_on_signal();
    // The model's name and its non-thinking sibling, each an alias of the loaded model so it routes on every endpoint.
    let advertised = built.preset.served_model_ids(&built.model_name);
    mistralrs_server_core::util::set_advertised_models(advertised.clone());
    let engine = built.model.inner_arc();
    let loaded = engine
        .get_default_model_id()
        .map_err(|e| anyhow::anyhow!("loaded model id: {e}"))?
        .context("engine reports no default model to advertise")?;
    for (id, _) in &advertised {
        engine.register_model_alias(id.clone(), &loaded).map_err(|e| anyhow::anyhow!("advertising {id}: {e}"))?;
    }

    let router = MistralRsServerRouterBuilder::new()
        .with_mistralrs(engine)
        .build()
        .await
        .map_err(|e| anyhow::anyhow!("server router build: {e}"))?;

    tracing::info!(target: "oscar::serve",
        "http://{addr}/v1 - OpenAI-compatible; model ids \"default\", {}; Ctrl-D exits; SIGUSR1 logs the generated text",
        advertised
            .iter()
            .map(|(id, _)| format!("\"{id}\""))
            .collect::<Vec<_>>()
            .join(", "),
    );
    // Ctrl-D too, as serve reads no stdin of its own.
    super::ui::exit_on_signal(built.bar.0.clone(), true);
    // `built` owns the model, the dispatcher and the bar's rows: alive for the server's life.
    axum::serve(listener, router).await.context("axum::serve")?;
    Ok(())
}

/// The generated text in the log while it is on: a `[think]` or `[text]` line a second with the deltas since the last,
/// line breaks shown as `⏎`; `kill -USR1` turns it on and off.
mod text {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    static ON: AtomicBool = AtomicBool::new(false);
    static HELD: Mutex<Option<(Instant, bool, String)>> = Mutex::new(None);

    extern "C" fn toggle(_: libc::c_int) {
        ON.fetch_xor(true, Ordering::Relaxed);
    }

    pub fn toggle_on_signal() {
        unsafe { libc::signal(libc::SIGUSR1, toggle as extern "C" fn(libc::c_int) as libc::sighandler_t) };
    }

    pub fn tap(reasoning: &str, content: &str) {
        if !ON.load(Ordering::Relaxed) {
            return;
        }
        let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
        for (think, delta) in [(true, reasoning), (false, content)] {
            if delta.is_empty() {
                continue;
            }
            // A change of kind, or a second gone, puts the held text out.
            if let Some((since, kind, text)) = held.take() {
                if kind == think && since.elapsed() < Duration::from_secs(1) {
                    *held = Some((since, kind, text + delta));
                    continue;
                }
                say(kind, &text);
            }
            *held = Some((Instant::now(), think, delta.to_string()));
        }
    }

    /// At a request's end: what the last second held.
    pub fn flush() {
        if let Some((_, think, text)) = HELD.lock().unwrap_or_else(|e| e.into_inner()).take() {
            say(think, &text);
        }
    }

    fn say(think: bool, text: &str) {
        let shown = text.replace('\n', "⏎");
        if think {
            tracing::info!(target: "oscar::think", "{shown}");
        } else {
            tracing::info!(target: "oscar::text", "{shown}");
        }
    }
}
