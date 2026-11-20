//! GPU plumbing: the KV and sequence-state pools, the wave ring, the address rows and the heartbeat.

pub(crate) mod expert_rows;
pub mod heartbeat;
pub mod kv_pool;
pub mod probe;
pub mod state_pool;
pub mod wave_pool;

pub use oscar_models::layout::{layout_host_bytes, ExpertExtent, GgufRoutedLayout, ProjectionKind, ProjectionLayout};

/// `count` words of `T` in a shared buffer, its view made once.
pub(crate) type Words<T> = oscar_offload::fixed::Fixed<T>;

/// `count` one-unit segments of `unit_bytes`, attached to the device's queue: the KV and state pools' shape.
pub(crate) fn unit_pool(
    md: &candle_core::MetalDevice, what: &str, unit_bytes: usize, count: usize,
    addressing: oscar_offload::segments::Addressing,
) -> candle_core::Result<std::sync::Arc<oscar_offload::segments::SegmentPool>> {
    let segments = oscar_offload::segments::SegmentPool::new(md.device().as_ref(), unit_bytes, count, 1, addressing)
        .map_err(|e| candle_core::Error::Msg(format!("{what}: {e}")))?;
    segments.attach(&md.command_queue()?);
    Ok(segments)
}

/// Unit `id` mapped, locked and wired.
pub(crate) fn take_unit(
    segments: &oscar_offload::segments::SegmentPool, md: &candle_core::MetalDevice, what: &str, id: usize,
) -> candle_core::Result<()> {
    segments.take(md.device().as_ref(), id, 1, true).map_err(|e| candle_core::Error::Msg(format!("{what} {id}: {e}")))
}

/// The label names what could not be taken.
pub(crate) fn words<T>(
    device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>, count: usize, fill: u8, what: &str,
) -> Result<Words<T>, String> {
    Words::new(device, count, fill).map_err(|e| format!("{what}: {e}"))
}

/// `len` elements of `dtype` from element `at`, no copy.
pub(crate) fn words_tensor<T>(
    words: &Words<T>, md: &candle_core::MetalDevice, dtype: candle_core::DType, at: usize, len: usize,
) -> candle_core::Result<candle_core::Tensor> {
    let total = words.count() * std::mem::size_of::<T>() / dtype.size_in_bytes();
    oscar_kernels::tensor_over(words.view().clone(), md, total, dtype).narrow(0, at, len)
}

/// Drained at scope end: a thread dispatching GPU work in a loop pushes one per iteration, or keeps every encoder.
pub struct AutoreleaseScope(*mut std::ffi::c_void);

impl AutoreleaseScope {
    pub fn push() -> Self {
        // SAFETY: a LIFO push/pop pair on this thread; the scope is never sent.
        Self(unsafe { objc2::ffi::objc_autoreleasePoolPush() })
    }
}

impl Drop for AutoreleaseScope {
    fn drop(&mut self) {
        // SAFETY: the token came from `push` on this thread.
        unsafe { objc2::ffi::objc_autoreleasePoolPop(self.0) }
    }
}
