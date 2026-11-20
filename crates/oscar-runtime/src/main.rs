//! `oscar-runtime`: the command line in `oscar_runtime::cli`.

fn main() -> anyhow::Result<()> {
    oscar_runtime::cli::main(std::env::args().collect())
}
