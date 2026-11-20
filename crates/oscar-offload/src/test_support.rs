//! The crate's test fixtures.

use crate::census::{peek, take_record, Asked};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;
use std::sync::{Mutex, MutexGuard};

/// Held by a test that takes memory, begins a step or arms the census: all three are process-wide.
pub(crate) fn test_lock() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

pub(crate) fn device() -> Retained<ProtocolObject<dyn MTLDevice>> {
    objc2_metal::MTLCreateSystemDefaultDevice().expect("a Metal device")
}

/// The census disarmed and its record emptied, under the lock every test that arms it holds.
pub(crate) fn fresh() -> MutexGuard<'static, ()> {
    let guard = test_lock();
    candle_metal_kernels::metal::pool::set_census(false);
    let _ = take_record();
    guard
}

pub(crate) fn record() -> Vec<Asked> {
    peek(<[Asked]>::to_vec)
}
