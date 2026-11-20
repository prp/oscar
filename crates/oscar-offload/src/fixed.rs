//! The plan's floor: the fixed pool, and the words kept in it for the process's life.

use crate::machine::PAGE;
use crate::segments::{padded_unit, Addressing, SegmentPool};
use crate::{Error, Result};
use candle_metal_kernels::metal::{Buffer, CommandQueue};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLDevice};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

/// A segment per fixed object: well above a model's thousands of dense weights.
const FIXED_SEGMENTS: usize = 16384;
/// The largest object, in pages: a bound, not a policy; past it is a loud failure.
const MAX_OBJECT_PAGES: usize = (8 << 30) / PAGE;

/// The plan's floor, a segment per object, so it maps exactly the sum of its page-rounded objects, the plan's figure.
pub struct FixedPool {
    segments: Arc<SegmentPool>,
    next: AtomicUsize,
}

impl FixedPool {
    fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        Ok(Self {
            segments: SegmentPool::new(device, PAGE, FIXED_SEGMENTS, MAX_OBJECT_PAGES, Addressing::Whole)?,
            next: AtomicUsize::new(0),
        })
    }

    /// `bytes` of floor in whole pages: the buffer and its length.
    pub fn take(
        &self, device: &ProtocolObject<dyn MTLDevice>, bytes: usize,
    ) -> Result<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)> {
        let want = padded_unit(bytes.max(1)) / PAGE;
        if want > MAX_OBJECT_PAGES {
            return Err(Error::Layout(format!(
                "fixed pool: {bytes} bytes is past the {} the largest object may be",
                MAX_OBJECT_PAGES * PAGE
            )));
        }
        let seg = self.next.fetch_add(1, Ordering::Relaxed);
        if seg >= FIXED_SEGMENTS {
            return Err(Error::Layout(format!("fixed pool: its index space of {FIXED_SEGMENTS} objects is full")));
        }
        self.segments.take(device, seg, want, true)?;
        let buffer = self.segments.buffer(seg).expect("a whole-buffer pool wraps the segment it just took");
        Ok((buffer, want * PAGE))
    }

    pub fn bytes(&self) -> usize {
        self.segments.bytes()
    }

    pub fn attach(&self, queue: &CommandQueue) {
        self.segments.attach(queue);
    }

    /// Its index space on the host, whatever it holds.
    pub fn host_bytes() -> usize {
        SegmentPool::host_bytes(FIXED_SEGMENTS)
    }
}

/// `count` words of `T` in the fixed pool, never given back: build each once per process; clones share them.
#[derive(Clone)]
pub struct Fixed<T> {
    view: Buffer,
    ptr: *mut T,
    count: usize,
}

// SAFETY: the pages live for the process; callers own what they write.
unsafe impl<T> Send for Fixed<T> {}
unsafe impl<T> Sync for Fixed<T> {}

impl<T> Fixed<T> {
    /// At least one word, every byte `fill` (`0xFF` where a sentinel must not pre-match).
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, count: usize, fill: u8) -> Result<Self> {
        let count = count.max(1);
        let (buffer, bytes) = fixed_pool(device)?.take(device, count * std::mem::size_of::<T>())?;
        let ptr = buffer.contents().as_ptr().cast::<T>();
        // SAFETY: the run is `bytes` long and nothing else names it yet.
        unsafe { std::ptr::write_bytes(ptr.cast::<u8>(), fill, bytes) };
        Ok(Self { view: Buffer::new(buffer), ptr, count })
    }

    pub fn ptr(&self) -> *mut T {
        self.ptr
    }

    pub fn count(&self) -> usize {
        self.count
    }

    /// Made once: a view per bind would add an objc retain per layer.
    pub fn view(&self) -> &Buffer {
        &self.view
    }

    pub fn buffer(&self) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        objc2::Message::retain(self.view.as_ref())
    }

    /// Put the fixed pool's set on `queue`.
    pub fn attach(&self, queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>) {
        if let Some(pool) = fixed_pool_if_built() {
            pool.attach(queue);
        }
    }
}

static FIXED: OnceLock<FixedPool> = OnceLock::new();

/// The fixed pool, made on first use.
pub fn fixed_pool(device: &ProtocolObject<dyn MTLDevice>) -> Result<&'static FixedPool> {
    if let Some(p) = FIXED.get() {
        return Ok(p);
    }
    let made = FixedPool::new(device)?;
    let _ = FIXED.set(made);
    Ok(FIXED.get().expect("the fixed pool was just set"))
}

pub fn fixed_pool_if_built() -> Option<&'static FixedPool> {
    FIXED.get()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{device, test_lock};

    #[test]
    fn the_fixed_pool_maps_each_object_in_whole_pages_and_nothing_beside_and_a_fixed_buffer_is_filled() {
        let _g = test_lock();
        let device = device();
        let pool = FixedPool::new(&device).expect("a fixed pool");
        let (a, a_bytes) = pool.take(&device, 3 * PAGE + 1).expect("a run");
        let (b, b_bytes) = pool.take(&device, PAGE).expect("a second run");
        assert_eq!((a_bytes, b_bytes), (4 * PAGE, PAGE), "each object is whole pages");
        assert_eq!(pool.bytes(), 5 * PAGE, "the pool maps what it hands out");
        let (pa, pb) = (a.contents().as_ptr() as usize, b.contents().as_ptr() as usize);
        assert_ne!(pa, pb, "two mappings");
        // A write through one object is not seen through the other.
        // SAFETY: both runs are mapped, `bytes` long, and this thread owns them.
        unsafe {
            std::ptr::write_bytes(pa as *mut u8, 0xA1, a_bytes);
            std::ptr::write_bytes(pb as *mut u8, 0xB2, b_bytes);
            assert!((0..a_bytes).all(|i| *(pa as *const u8).add(i) == 0xA1));
            assert!((0..b_bytes).all(|i| *(pb as *const u8).add(i) == 0xB2));
        }
        let big = (64 << 20) + PAGE;
        let (c, c_bytes) = pool.take(&device, big).expect("a large object");
        assert_eq!(c_bytes, big, "a large object gets exactly its pages");
        assert_ne!(c.contents().as_ptr() as usize, pa, "a distinct mapping");
        assert_eq!(pool.bytes(), 5 * PAGE + big, "and the pool still maps what it hands out");
        // A fixed buffer is filled and padded: every byte, a sentinel that stops short pre-matches.
        let fixed = Fixed::<u8>::new(&device, 3 * PAGE + 1, 0xFF).expect("a fixed buffer");
        let bytes = fixed.buffer().length();
        assert_eq!(bytes, 4 * PAGE, "a fixed buffer pads to whole pages");
        // SAFETY: the mapping is `bytes` long and this thread owns it.
        assert!((0..bytes).all(|i| unsafe { *fixed.ptr().add(i) } == 0xFF));
    }
}
