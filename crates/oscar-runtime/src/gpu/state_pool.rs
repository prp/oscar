//! Each recurrent state is one object, its tensors at page offsets: a copy is one blit, ordered against every kernel.

use std::sync::{Arc, Mutex, OnceLock};

use candle_core::{DType, Device, MetalDevice, Result, Tensor};
use candle_metal_kernels::metal::Buffer as KernelBuffer;
use mistralrs_core::kv_cache::{RecurrentState, RecurrentStateLayout, RecurrentStateMemory};
use oscar_kernels::tensor_over;
use oscar_models::state_slots::{runs_of, Run};
use oscar_offload::segments::{Addressing, SegmentPool};

pub struct SequenceStatePool {
    md: MetalDevice,
    segments: Arc<SegmentPool>,
    layout: RecurrentStateLayout,
    /// Per recurrent layer, `[conv, recurrent]`.
    runs: Vec<[Run; 2]>,
    object_bytes: usize,
    live: Mutex<Live>,
}

/// Segments are mapped in id order, so `mapped` is the count and the next id.
#[derive(Default)]
struct Live {
    free: Vec<usize>,
    mapped: usize,
    max_mapped: usize,
}

/// A state's hold on its segment: the last handle's drop frees it.
struct Hold {
    pool: Arc<SequenceStatePool>,
    id: usize,
}

impl Drop for Hold {
    fn drop(&mut self) {
        self.pool.live.lock().expect("state pool poisoned").free.push(self.id);
    }
}

static POOL: OnceLock<Arc<SequenceStatePool>> = OnceLock::new();

/// Where the mistral.rs fork's cache takes states: the pool, built at the first take with the index ceiling.
struct Memory;

impl RecurrentStateMemory for Memory {
    fn take(&self, layout: &RecurrentStateLayout, device: &Device) -> Result<RecurrentState> {
        let pool = match POOL.get() {
            Some(pool) => pool,
            None => {
                let built = SequenceStatePool::build(layout, device, crate::plan::STATE_SLOTS_CEILING)?;
                let _ = POOL.set(built);
                POOL.get().expect("just set")
            }
        };
        if pool.layout != *layout {
            candle_core::bail!("sequence-state pool holds one layout; take asks another");
        }
        pool.take()
    }
}

/// Before the model is built; the pool follows at the first take.
pub fn install() {
    mistralrs_core::kv_cache::set_recurrent_state_memory(Arc::new(Memory));
}

/// The pool, once the first state has been taken.
pub fn installed() -> Option<&'static Arc<SequenceStatePool>> {
    POOL.get()
}

/// Until the plan is solved: the census's live state and its drafter's copy.
const BEFORE_THE_PLAN: usize = 2;

impl SequenceStatePool {
    fn build(layout: &RecurrentStateLayout, device: &Device, slots: usize) -> Result<Arc<Self>> {
        let Device::Metal(md) = device else {
            candle_core::bail!("sequence-state pool needs Metal device");
        };
        let (runs, offset) = runs_of(layout);
        if offset == 0 {
            candle_core::bail!("sequence state of no bytes");
        }
        let segments = super::unit_pool(md, "sequence-state pool", offset, slots, Addressing::Whole)?;
        Ok(Arc::new(Self {
            md: md.clone(),
            segments,
            layout: layout.clone(),
            runs,
            object_bytes: offset,
            live: Mutex::new(Live { max_mapped: BEFORE_THE_PLAN, ..Live::default() }),
        }))
    }

    /// A state over a free segment, mapped if the count allows; not zeroed.
    fn take(self: &Arc<Self>) -> Result<RecurrentState> {
        let id = {
            let mut live = self.live.lock().expect("state pool poisoned");
            match live.free.pop() {
                Some(id) => id,
                None if live.mapped < live.max_mapped => {
                    let id = live.mapped;
                    super::take_unit(&self.segments, &self.md, "sequence state", id)?;
                    live.mapped += 1;
                    id
                }
                None => candle_core::bail!(
                    "[state pool] all {} of {:.1} MB held (plan's count); take not counted",
                    live.max_mapped,
                    self.object_bytes as f64 / (1u64 << 20) as f64,
                ),
            }
        };
        let buffer =
            KernelBuffer::new(self.segments.buffer(id).expect("a whole-buffer pool wraps the segment it holds"));
        let bytes = tensor_over(buffer.clone(), &self.md, self.object_bytes, DType::U8);
        self.state_over(&buffer, bytes, Box::new(Hold { pool: Arc::clone(self), id }))
    }

    /// The object as `bytes`, and each layer's runs as storages over `buffer`.
    fn state_over(
        &self, buffer: &KernelBuffer, bytes: Tensor, hold: Box<dyn std::any::Any + Send + Sync>,
    ) -> Result<RecurrentState> {
        let view = |run: &Run| {
            let es = run.dtype.size_in_bytes();
            tensor_over(buffer.clone(), &self.md, self.object_bytes / es, run.dtype)
                .narrow(0, run.offset / es, run.shape.elem_count())?
                .reshape(run.shape.clone())
        };
        let layers = self.runs.iter().map(|[c, r]| Ok((view(c)?, view(r)?))).collect::<Result<Vec<_>>>()?;
        Ok(RecurrentState::new(Some(bytes), layers, hold))
    }

    /// One blit into a run of the caller's stream: GPU-written, the run is cut again at once when given back.
    pub fn stream_copy(&self, state: &RecurrentState) -> Result<RecurrentState> {
        let run = self.md.new_buffer(self.object_bytes, DType::U8, "sequence-state")?;
        let buffer = (*run).clone();
        let bytes = tensor_over(run, &self.md, self.object_bytes, DType::U8);
        let mine = self.state_over(&buffer, bytes, Box::new(()))?;
        mine.copy_from(state)?;
        Ok(mine)
    }

    /// At the load's tail: the plan's count, every segment mapped now.
    pub fn launch(&self, count: usize) -> Result<()> {
        let mut live = self.live.lock().expect("state pool poisoned");
        live.max_mapped = count;
        while live.mapped < count {
            let id = live.mapped;
            super::take_unit(&self.segments, &self.md, "sequence state", id)?;
            live.mapped += 1;
            live.free.push(id);
        }
        Ok(())
    }

    /// One state's bytes, its runs page-rounded.
    pub fn object_bytes(&self) -> usize {
        self.object_bytes
    }

    /// For the host line before the pool exists: segment state and a list entry per slot.
    pub fn host_bytes(slots: usize) -> u64 {
        (SegmentPool::host_bytes(slots) + slots * std::mem::size_of::<usize>()) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{IndexOp, Storage};

    fn same_object(a: &Tensor, b: &Tensor) -> bool {
        let (sa, _) = a.storage_and_layout();
        let (sb, _) = b.storage_and_layout();
        match (&*sa, &*sb) {
            (Storage::Metal(x), Storage::Metal(y)) => std::ptr::eq(x.buffer().as_ref(), y.buffer().as_ref()),
            _ => false,
        }
    }

    #[test]
    fn a_state_is_one_object_and_a_copy_is_its_bytes() -> Result<()> {
        let dev = oscar_offload::pools::test_device()?;
        let layout = RecurrentStateLayout {
            layers: vec![
                ((1, 4, 3), DType::F32, vec![1, 2, 4, 4], DType::F32),
                ((1, 8, 3), DType::BF16, vec![1, 2, 8, 8], DType::F32),
            ],
        };
        let pool = SequenceStatePool::build(&layout, &dev, crate::plan::STATE_SLOTS_CEILING)?;
        pool.launch(3)?;
        assert_eq!(pool.object_bytes(), 4 * oscar_offload::segments::PAGE, "four tensors, a page each");
        let live = pool.take()?;
        live.zero()?;
        assert_eq!((live.layers().len(), live.bytes()), (2, pool.object_bytes()));
        let (c0, r0) = &live.layers()[0];
        let (c1, r1) = &live.layers()[1];
        assert!(same_object(c0, r0) && same_object(c0, c1) && same_object(c0, r1), "one buffer object");
        assert_eq!((c0.dims(), r1.dims()), (&[1, 4, 3][..], &[1, 2, 8, 8][..]));
        assert!(c0.flatten_all()?.to_vec1::<f32>()?.iter().all(|&v| v == 0.0), "zeroed");
        c0.slice_set(&Tensor::full(1f32, (1, 4, 3), &dev)?, 0, 0)?;
        r1.slice_set(&Tensor::full(3f32, (1, 2, 8, 8), &dev)?, 0, 0)?;
        let snap = pool.take()?;
        snap.copy_from(&live)?;
        c0.slice_set(&Tensor::full(2f32, (1, 4, 3), &dev)?, 0, 0)?;
        let (sc0, _) = &snap.layers()[0];
        let (_, sr1) = &snap.layers()[1];
        assert!(sc0.flatten_all()?.to_vec1::<f32>()?.iter().all(|&v| v == 1.0), "the snapshot kept the bytes");
        assert!(sr1.i((0, 1, 7))?.flatten_all()?.to_vec1::<f32>()?.iter().all(|&v| v == 3.0), "every layer's");
        assert!(c0.flatten_all()?.to_vec1::<f32>()?.iter().all(|&v| v == 2.0), "the live state moved on");
        live.copy_from(&snap)?;
        assert!(c0.flatten_all()?.to_vec1::<f32>()?.iter().all(|&v| v == 1.0), "restored");
        let third = pool.take()?;
        assert!(pool.take().is_err(), "the plan's count is three");
        drop(third);
        assert!(pool.take().is_ok(), "a state given back is taken again");
        Ok(())
    }
}
