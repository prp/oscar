#![deny(clippy::correctness)]
#![warn(clippy::suspicious, clippy::style, clippy::complexity, clippy::perf)]

//! The wired memory every pool takes from (candle's allocator, the segments, the census's clock, the machine's lines)
//! and the drives the expert pool reads from.

pub mod census;
pub mod error;
pub mod fixed;
pub mod io;
pub mod machine;
pub mod pools;
pub mod region;
pub mod segments;
#[cfg(test)]
mod test_support;

pub use error::{Error, Result};
