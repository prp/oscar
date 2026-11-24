//! The `oscar-runtime` command line; an evaluation's own binary runs [`main`] after installing its hooks.

pub mod args;
pub mod build;
pub mod chat;
pub mod check_model;
pub mod console;
pub mod download;
pub mod print_config;
pub mod serve;
pub mod settings;
pub mod ui;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "oscar", about = "Oscar runtime CLI", version = env!("OSCAR_VERSION"))]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Interactive chat against a local GGUF.
    Chat(args::ChatArgs),
    /// OpenAI-compatible HTTP endpoint over the model `chat` builds; requests
    /// queue.
    Serve(args::ServeArgs),
    /// Print a model's config parameter set as `KEY=VALUE` lines, as `chat` applies it.
    PrintConfig(print_config::PrintConfigArgs),
    /// Check a model's files from their GGUF headers alone: shards, routed
    /// layout, decode kernels, the plan and its fit.
    CheckModel(check_model::CheckModelArgs),
    /// Fetch a model from Hugging Face, requantise it and store it, ready to run.
    Download(download::DownloadArgs),
    /// Every supported model, what is on disk where, and the store's size and
    /// free space.
    List,
    /// Remove one variant of a model from the managed store.
    Delete(download::DeleteArgs),
}

/// `args` with the program name first, as `std::env::args` gives them.
pub fn main(args: Vec<String>) -> Result<()> {
    // No subcommand from a double-click, launchd or the login item (no terminal, no ssh) is the menu bar app; from
    // a terminal, the help. The allocator restart comes after: a re-executed app gets no place in the menu bar.
    let Some(cmd) = Cli::parse_from(args).cmd else {
        if crate::app::wanted() {
            return crate::app::main();
        }
        use clap::CommandFactory;
        return Cli::command().print_help().map_err(Into::into);
    };
    crate::config::allocator::restart_with_large_cache_off();
    // Before any lane's first line.
    crate::cli::console::install(
        match &cmd {
            Cmd::Chat(a) => a.common.verbose,
            Cmd::Serve(a) => a.common.verbose,
            Cmd::Download(a) => a.verbose,
            _ => 0,
        },
        matches!(cmd, Cmd::Serve(_)),
        // The app keeps its child's lines itself.
        matches!(&cmd, Cmd::Chat(_)) || matches!(&cmd, Cmd::Serve(a) if !a.status_lines),
    );
    let runtime = tokio::runtime::Builder::new_multi_thread().worker_threads(4).enable_all().build()?;
    let ended = runtime.block_on(async {
        match cmd {
            Cmd::Chat(args) => chat::chat_main(args).await,
            Cmd::Serve(args) => serve::serve_main(args).await,
            Cmd::PrintConfig(args) => print_config::print_config_main(args),
            Cmd::CheckModel(args) => check_model::check_model_main(args),
            Cmd::Download(args) => download::download_main(args).await,
            Cmd::List => download::list_main(),
            Cmd::Delete(args) => download::delete_main(args),
        }
    });
    // The console's last lines are said before an error's own, which the log keeps too.
    if let Err(e) = &ended {
        crate::cli::console::keep(format!("Error: {e:?}\n").into_bytes());
    }
    crate::cli::console::drain();
    ended
}
