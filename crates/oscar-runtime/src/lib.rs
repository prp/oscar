#![deny(clippy::correctness)]
#![warn(clippy::suspicious, clippy::style, clippy::complexity, clippy::perf)]

//! Oscar's runtime: [`engine::dispatcher::OffloadDispatcher`] resolves each MoE call's experts against the banks.

pub mod app;
pub mod cli;
pub mod config;
pub mod drafter;
pub mod engine;
pub mod gpu;
pub mod kv_store;
pub mod load;
pub use oscar_offload::machine;
pub mod memory;
pub mod model_store;
pub mod plan;
pub mod pool;
