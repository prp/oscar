//! Oscar's Metal kernels and their launchers, over the candle fork alone.

macro_rules! refuse_if {
    ($cond:expr, $($why:tt)+) => {
        if $cond {
            candle_core::bail!($($why)+)
        }
    };
}

mod loader;

pub mod gdn;

pub mod mhc;
pub mod mla;
pub mod paged;

pub mod metal_kernel;

mod drafter;
mod embed;
mod matvec;
mod moe_routed;
mod routing;
mod sampler;

pub use drafter::*;
pub use embed::*;
pub use loader::{tensor_over, Metal};
pub use matvec::*;
pub use mhc::*;
pub use moe_routed::*;
pub use routing::*;
pub use sampler::*;

#[cfg(test)]
mod test_support;
