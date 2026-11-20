//! The staging arena, one Metal buffer for the process's life: no store path allocates host memory per record.

use std::collections::VecDeque;
use std::sync::Arc;

use candle_core::backend::BackendStorage;
use candle_core::{DType, MetalDevice, MetalStorage, Result, Shape, Storage, Tensor};
use candle_metal_kernels::metal::Buffer;

use super::record::RecordKind;

/// Record starts are aligned so every element type addresses its range.
const ALIGN: u64 = 16;

/// A byte range of the arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ArenaRange {
    pub off: u64,
    pub len: u64,
}

/// A staged record awaiting the writer.
pub(crate) struct Record {
    pub kind: RecordKind,
    pub hash: u64,
    pub parent: u64,
    pub pos: u32,
    pub range: ArenaRange,
}

/// A queued record, or padding where `record` is `None`.
pub(crate) struct RingUnit {
    span: u64,
    pub record: Option<Record>,
}

/// The head a transient scope gives back to.
pub(crate) struct Mark {
    head: u64,
}

/// A FIFO ring: the engine thread allocates at the head, the writer releases at the tail; no record straddles the end.
#[derive(Default)]
pub(crate) struct Ring {
    cap: u64,
    head: u64,
    used: u64,
    pub queue: VecDeque<RingUnit>,
    /// Padding excluded, as the deposit line reports it.
    pub queued_bytes: u64,
    /// What an open transient scope took, padding included; none of it is queued, so no release reaches it.
    scope: Option<u64>,
    /// A wrap's padding, queued before the first range past the wrap so the writer releases it after the ranges before it.
    pad: Option<u64>,
}

fn span(len: u64) -> u64 {
    len.div_ceil(ALIGN) * ALIGN
}

impl Ring {
    fn new(cap: u64) -> Self {
        Self { cap, ..Default::default() }
    }

    pub fn capacity(&self) -> u64 {
        self.cap
    }

    /// At the head; `None`: wait for the writer or store less.
    pub fn alloc(&mut self, len: u64) -> Option<ArenaRange> {
        let need = span(len);
        if need == 0 || self.used + need > self.cap {
            return None;
        }
        let mut pad = 0;
        if self.head + need > self.cap {
            pad = self.cap - self.head;
            if self.used + pad + need > self.cap {
                return None;
            }
            if self.scope.is_none() && pad > 0 {
                debug_assert!(self.pad.is_none(), "a batch wraps at most once");
                self.pad = Some(pad);
            }
            self.used += pad;
            self.head = 0;
        }
        let range = ArenaRange { off: self.head, len };
        self.head += need;
        self.used += need;
        if let Some(taken) = self.scope.as_mut() {
            *taken += pad + need;
        }
        Some(range)
    }

    /// In allocation order, as the writer releases in queue order; without a record, padding the writer skips.
    pub fn enqueue(&mut self, range: ArenaRange, record: Option<Record>) {
        debug_assert!(self.scope.is_none(), "a range was enqueued inside a transient scope");
        if let Some(pad) = self.pad.take_if(|_| range.off == 0) {
            self.queue.push_back(RingUnit { span: pad, record: None });
        }
        self.queued_bytes += record.as_ref().map_or(0, |r| r.range.len);
        self.queue.push_back(RingUnit { span: span(range.len), record });
    }

    /// A superseded record becomes padding the writer releases unwritten.
    pub fn cancel(&mut self, range: ArenaRange) {
        let Some(p) = self.queue.iter_mut().find(|p| p.record.as_ref().is_some_and(|r| r.range == range)) else {
            return;
        };
        p.record = None;
        self.queued_bytes -= range.len;
    }

    /// Once the writer sealed or skipped the front unit.
    pub fn release_front(&mut self) -> Option<RingUnit> {
        let p = self.queue.pop_front()?;
        self.used -= p.span;
        if let Some(r) = &p.record {
            self.queued_bytes -= r.range.len;
        }
        Some(p)
    }

    /// A transient scope, given back whole at [`Self::reset_to`]; on an empty ring from byte 0, for the whole arena.
    pub fn mark(&mut self) -> Mark {
        debug_assert!(self.scope.is_none(), "transient scopes do not nest");
        if self.used == 0 {
            debug_assert!(self.queue.is_empty());
            self.head = 0;
        }
        self.scope = Some(0);
        Mark { head: self.head }
    }

    /// Wrap padding included: the writer never saw those bytes, so `used` falls by exactly what the scope took.
    pub fn reset_to(&mut self, m: Mark) {
        let taken = self.scope.take().expect("a transient scope is open");
        self.used -= taken;
        self.head = m.head;
    }
}

/// Cloned out of the lock for copies, reads and writes; its pages are the arena's own pool, one locked segment.
#[derive(Clone)]
pub(crate) struct ArenaBuf {
    buf: Arc<Buffer>,
    /// The buffer's segment, held for the arena's life.
    #[allow(dead_code)]
    pool: Arc<oscar_offload::segments::SegmentPool>,
    dev: MetalDevice,
    cap: u64,
}

impl ArenaBuf {
    fn storage(&self, dtype: DType) -> MetalStorage {
        let count = (self.cap / dtype.size_in_bytes() as u64) as usize;
        MetalStorage::new(Arc::clone(&self.buf), self.dev.clone(), count, dtype)
    }

    fn elems(&self, byte_off: u64, dtype: DType) -> Result<usize> {
        let es = dtype.size_in_bytes() as u64;
        if byte_off % es != 0 {
            candle_core::bail!("[context store] arena byte {byte_off} is not aligned to {dtype:?}");
        }
        Ok((byte_off / es) as usize)
    }

    pub fn view(&self, range: ArenaRange, shape: &[usize], dtype: DType) -> Result<Tensor> {
        let start = self.elems(range.off, dtype)?;
        let n: usize = shape.iter().product();
        if (n as u64) * dtype.size_in_bytes() as u64 != range.len {
            candle_core::bail!("[context store] arena view {shape:?} {dtype:?} does not fill its range");
        }
        let count = (self.cap / dtype.size_in_bytes() as u64) as usize;
        let flat = Tensor::from((Storage::Metal(self.storage(dtype)), Shape::from_dims(&[count])));
        flat.narrow(0, start, n)?.reshape(shape)
    }

    /// `d1` rows of `d2` elements, `src_s` apart in `src`, `dst_s` apart from byte `dst_off`; contiguous is one blit.
    pub fn copy_in(&self, src: &Tensor, d1: usize, d2: usize, src_s: usize, dst_off: u64, dst_s: usize) -> Result<()> {
        let dtype = src.dtype();
        let dst_o = self.elems(dst_off, dtype)?;
        let (guard, layout) = src.storage_and_layout();
        let Storage::Metal(ms) = &*guard else {
            candle_core::bail!("[context store] source tensor not on Metal device");
        };
        let mut dst = self.storage(dtype);
        ms.copy2d(&mut dst, d1, d2, src_s, dst_s, layout.start_offset(), dst_o)
    }

    pub fn device(&self) -> candle_core::Device {
        candle_core::Device::Metal(self.dev.clone())
    }

    /// The copies encoded so far, before a host read or a reuse.
    pub fn wait(&self) -> Result<()> {
        self.dev.wait_until_completed()
    }

    /// # Safety
    /// `range` is allocated with no GPU copy into it in flight (`wait` first), the slice dropped before its release.
    pub unsafe fn bytes(&self, range: ArenaRange) -> &[u8] {
        debug_assert!(range.off + range.len <= self.cap);
        std::slice::from_raw_parts(self.buf.contents().add(range.off as usize), range.len as usize)
    }

    /// A pointer, not a slice: only the caller knows no other reference exists.
    ///
    /// # Safety
    /// As [`Self::bytes`], and no GPU copy reads the range until the write is done.
    pub unsafe fn bytes_mut(&self, range: ArenaRange) -> *mut u8 {
        debug_assert!(range.off + range.len <= self.cap);
        self.buf.contents().add(range.off as usize)
    }
}

pub(crate) struct Arena {
    pub ring: Ring,
    pub buf: ArenaBuf,
}

impl Arena {
    pub fn new(dev: &MetalDevice, cap: u64) -> Result<Self> {
        use oscar_offload::segments::{Addressing, SegmentPool};
        let raw = dev.device().as_ref();
        let err = |e: oscar_offload::Error| candle_core::Error::Msg(format!("kv store arena: {e}"));
        let pool = SegmentPool::new(raw, cap as usize, 1, 1, Addressing::Whole).map_err(err)?;
        pool.take(raw, 0, 1, true).map_err(err)?;
        pool.attach(&dev.command_queue()?);
        let buf = Arc::new(Buffer::new(pool.buffer(0).expect("the arena's segment was just taken")));
        Ok(Self { ring: Ring::new(cap), buf: ArenaBuf { buf, pool, dev: dev.clone(), cap } })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(range: ArenaRange) -> Record {
        Record { kind: RecordKind::Block, hash: range.off, parent: 0, pos: 0, range }
    }

    /// A ring of `bytes` with a `first`-byte record queued at its start.
    fn ring(bytes: u64, first: u64) -> (Ring, ArenaRange) {
        let mut r = Ring::new(bytes);
        let a = r.alloc(first).unwrap();
        r.enqueue(a, Some(rec(a)));
        (r, a)
    }

    /// Padding released ahead of the ranges before the wrap would count their space free while they are still copied:
    /// a wrap's padding is queued in ring order. A transient scope gives back what it took: one wrapping while the
    /// writer releases, one on an empty ring (an arena-sized run then fits), one popped after a release.
    #[test]
    fn the_ring_wraps_with_padding_refuses_what_it_cannot_hold_and_a_scope_gives_back_what_it_took() {
        let (mut r, a) = ring(100, 40);
        let b = r.alloc(40).unwrap();
        r.enqueue(b, Some(rec(b)));
        assert_eq!((a.off, b.off), (0, 48));
        // 96 used of 100: nothing more fits until the writer releases.
        assert!(r.alloc(10).is_none());
        // Release a: 48 free but 4 at the end, so a 40-byte record wraps.
        assert!(r.release_front().is_some());
        let c = r.alloc(40).unwrap();
        assert_eq!(c.off, 0);
        r.enqueue(c, Some(rec(c)));
        assert_eq!((r.queue.len(), r.queue[1].record.is_none(), r.queued_bytes), (3, true, 80), "b, padding, c");
        // Releasing b then the padding leaves c alone.
        r.release_front();
        r.release_front();
        assert_eq!(r.used, 48);
        // A batch of two: x before the end, y past it at 0, 20 bytes of padding between them.
        let (mut r, _) = ring(100, 40);
        r.release_front();
        let (x, y) = (r.alloc(30).unwrap(), r.alloc(30).unwrap());
        assert_eq!((x.off, y.off), (48, 0));
        assert!(r.release_front().is_none(), "nothing is queued before the batch is");
        assert!(r.alloc(30).is_none(), "the padding counts until it is released in its turn");
        r.enqueue(x, Some(rec(x)));
        r.enqueue(y, Some(rec(y)));
        assert!(r.queue[1].record.is_none(), "the padding sits between x and y");
        r.release_front();
        r.release_front();
        assert_eq!(r.alloc(30).unwrap().off, 32, "x's and the padding's bytes are free once both left in order");
        // The scopes.
        let (mut r, _) = ring(100, 40);
        let m = r.mark();
        assert_eq!(r.alloc(40).unwrap().off, 48);
        r.release_front();
        // 96 + 32 passes the end: 4 bytes of padding, and the range at 0.
        assert_eq!(r.alloc(30).unwrap().off, 0);
        r.reset_to(m);
        assert_eq!((r.used, r.queue.len()), (0, 0));
        assert!(r.release_front().is_none());
        assert_eq!(r.alloc(48).unwrap().off, 48);
        // The empty ring.
        let (mut r, _) = ring(112, 40);
        r.release_front();
        let m = r.mark();
        assert_eq!(r.alloc(112).unwrap().off, 0);
        r.reset_to(m);
        assert_eq!((r.head, r.used), (0, 0));
        // The release meanwhile.
        let (mut r, _) = ring(100, 30);
        let m = r.mark();
        let (t1, t2) = (r.alloc(20).unwrap(), r.alloc(20).unwrap());
        assert_eq!((t1.off, t2.off), (32, 64));
        r.release_front();
        r.reset_to(m);
        assert_eq!((r.head, r.used), (32, 0));
        // And the space is reusable from the mark.
        assert_eq!(r.alloc(60).unwrap().off, 32);
    }

    /// Both paths, contiguous and strided.
    #[test]
    fn copies_land_where_the_range_says() {
        let candle_core::Device::Metal(dev) = oscar_offload::pools::test_device().unwrap() else {
            panic!("the arena's tests need the Metal device")
        };
        let device = candle_core::Device::Metal(dev.clone());
        let t = Tensor::arange(0f32, 24f32, &device).unwrap();
        let arena = Arena::new(&dev, 4096).unwrap();
        arena.buf.copy_in(&t, 1, 24, 24, 256, 24).unwrap();
        // Rows of 6 elements, 8 apart: the strided path.
        arena.buf.copy_in(&t, 4, 6, 6, 1024, 8).unwrap();
        arena.buf.wait().unwrap();
        let f32s = |r: ArenaRange| -> Vec<f32> {
            unsafe { arena.buf.bytes(r) }.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
        };
        assert_eq!(f32s(ArenaRange { off: 256, len: 96 }), (0..24).map(|i| i as f32).collect::<Vec<_>>());
        let strided = f32s(ArenaRange { off: 1024, len: 128 });
        assert_eq!(&strided[..8], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0]);
        assert_eq!(&strided[8..14], &[6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    }
}
