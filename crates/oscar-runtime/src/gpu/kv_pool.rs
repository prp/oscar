//! The KV block pool: a segment per block over a fixed index space, addressed through per-layer K and V tables.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use candle_core::{DType, Device, MetalDevice, Result, Tensor};
use candle_metal_kernels::metal::Buffer as KernelBuffer;
use mistralrs_core::paged_attention::{KVCacheManager, KvBlockLayout, KvBlockMemory};
use oscar_kernels::tensor_over;
use oscar_offload::segments::{Addressing, SegmentPool};

use super::{words, Words};

/// Written under the pool's lock and read racily by the kernels: an absent block's entry is null (zero).
struct AddressTable {
    words: Words<u64>,
    tensor: Tensor,
}

impl AddressTable {
    fn new(md: &MetalDevice, entries: usize) -> Result<Self> {
        let words = words(md.device().as_ref(), entries, 0, "KV address table").map_err(candle_core::Error::Msg)?;
        let tensor = tensor_over(words.view().clone(), md, entries, DType::I64);
        Ok(Self { words, tensor })
    }

    fn write(&self, i: usize, address: u64) {
        // SAFETY: `i` is inside the index space the table was sized for.
        unsafe { self.words.ptr().add(i).write_volatile(address) };
    }
}

pub struct KvPool {
    md: MetalDevice,
    segments: Arc<SegmentPool>,
    layout: KvBlockLayout,
    /// (model layer, K table, V table) per KV layer.
    tables: Vec<(usize, AddressTable, AddressTable)>,
    /// Every layer's (K, V) pair for the kernels; a layer without KV shares a null entry.
    layer_tables: Vec<(Tensor, Tensor)>,
    move_lock: Mutex<()>,
    absent_stack: Mutex<Vec<usize>>,
    owed: Mutex<OwedMoves>,
    settle_in_flight: AtomicBool,
}

#[derive(Default)]
struct OwedMoves {
    release: Vec<usize>,
    take: Vec<usize>,
}

static POOL: OnceLock<Arc<KvPool>> = OnceLock::new();

/// Built when the engine asks: `counts` gives its layout's launch count and index space; an error refuses the load.
pub fn install(counts: impl Fn(&KvBlockLayout) -> Result<(usize, usize)> + Send + Sync + 'static) {
    mistralrs_core::paged_attention::set_kv_memory_factory(Box::new(move |device, layout| {
        let (live, index) = counts(layout)?;
        let pool = KvPool::build(device, layout, live, index)?;
        crate::detail!(target: "oscar::kv_pool",
            "{live} blocks x {} KB = {:.2} GB, index space {index}, one segment per block",
            pool.block_bytes() / 1024,
            pool.present_bytes() as f64 / crate::machine::GB,
        );
        Ok(pool)
    }));
}

impl KvPool {
    /// For the host line before the pool exists: segment state, absent list and the engine's block table.
    pub fn host_bytes(blocks: usize) -> u64 {
        (SegmentPool::host_bytes(blocks)
            + blocks * std::mem::size_of::<usize>()
            + mistralrs_core::paged_attention::block_pool::BlockPool::host_bytes(blocks)) as u64
    }

    /// Charged on the plan's tables line: a K and a V table per KV layer and the pool's own unit table, in whole pages.
    pub fn table_bytes(blocks: u64, kv_layers: u64) -> u64 {
        let page = oscar_offload::segments::PAGE as u64;
        (kv_layers * 2 + 1) * (blocks * std::mem::size_of::<u64>() as u64).next_multiple_of(page)
    }

    /// `blocks` one-block segments for `layout`, the first `live` taken, the residency set on the device's queue.
    pub fn build(device: &Device, layout: &KvBlockLayout, live: usize, blocks: usize) -> Result<Arc<Self>> {
        let Device::Metal(md) = device else {
            candle_core::bail!("KV block pool needs Metal device");
        };
        if live == 0 || live > blocks {
            candle_core::bail!("KV block pool: {live} live blocks over index space {blocks}");
        }
        let segments = super::unit_pool(md, "KV block pool", layout.block_bytes(), blocks, Addressing::Table)?;
        let mut tables = Vec::new();
        for (layer, _) in layout.kv_layers() {
            tables.push((layer, AddressTable::new(md, blocks)?, AddressTable::new(md, blocks)?));
        }
        let null = AddressTable::new(md, 1)?;
        let layer_tables = layout
            .rows
            .iter()
            .enumerate()
            .map(|(layer, rows)| match rows {
                Some(_) => {
                    let (_, k, v) = tables.iter().find(|(l, _, _)| *l == layer).expect("a KV layer has tables");
                    (k.tensor.clone(), v.tensor.clone())
                }
                None => (null.tensor.clone(), null.tensor.clone()),
            })
            .collect();
        let pool = Arc::new(Self {
            md: md.clone(),
            segments,
            layout: layout.clone(),
            tables,
            layer_tables,
            move_lock: Mutex::new(()),
            absent_stack: Mutex::new((live..blocks).rev().collect()),
            owed: Mutex::new(OwedMoves::default()),
            settle_in_flight: AtomicBool::new(false),
        });
        pool.take(&(0..live).collect::<Vec<_>>())?;
        let _ = POOL.set(Arc::clone(&pool));
        Ok(pool)
    }

    /// Up to `blocks` free blocks withdrawn, those without a cached prefix first, their release owed.
    pub fn shed(&self, manager: &mut KVCacheManager, blocks: usize) -> usize {
        let mut ids = manager.withdraw(blocks, false);
        if ids.len() < blocks {
            ids.extend(manager.withdraw(blocks - ids.len(), true));
        }
        let n = ids.len();
        self.owed.lock().expect("KV pool poisoned").release.extend(ids);
        n
    }

    /// Up to `blocks` absent blocks, owed to the background thread.
    pub fn regrow(&self, blocks: usize) -> usize {
        let mut absent = self.absent_stack.lock().expect("KV pool poisoned");
        let n = blocks.min(absent.len());
        let at = absent.len() - n;
        let ids: Vec<usize> = absent.drain(at..).collect();
        self.owed.lock().expect("KV pool poisoned").take.extend(ids);
        n
    }

    /// The mover's half, off the block-state lock: releases, then takes; a refused take ends it, the rest absent again.
    pub fn settle(&self) -> (usize, Vec<usize>, Option<candle_core::Error>) {
        self.settle_in_flight.store(true, Ordering::Release);
        let OwedMoves { release, take } = std::mem::take(&mut *self.owed.lock().expect("KV pool poisoned"));
        if let Err(e) = KvBlockMemory::release(self, &release) {
            return (0, Vec::new(), Some(e));
        }
        let released = release.len();
        self.absent_stack.lock().expect("KV pool poisoned").extend(release);
        let mut taken = Vec::with_capacity(take.len());
        for (i, &id) in take.iter().enumerate() {
            if let Err(e) = KvBlockMemory::take(self, &[id]) {
                self.absent_stack.lock().expect("KV pool poisoned").extend(take[i..].iter().rev());
                return (released, taken, Some(e));
            }
            taken.push(id);
        }
        (released, taken, None)
    }

    /// The takes are the allocator's: a waiting request can be seated.
    pub fn handed_back(&self) {
        self.settle_in_flight.store(false, Ordering::Release);
    }

    /// Blocks still coming, owed or settled but not handed back: the scheduler waits rather than starving a request.
    pub fn pending(&self) -> bool {
        let p = self.owed.lock().expect("KV pool poisoned");
        !p.release.is_empty() || !p.take.is_empty() || self.settle_in_flight.load(Ordering::Acquire)
    }

    pub fn absent(&self) -> usize {
        self.absent_stack.lock().expect("KV pool poisoned").len()
    }

    pub fn installed() -> Option<&'static Arc<KvPool>> {
        POOL.get()
    }

    /// A block's bytes as the machine holds them: the object, whole pages.
    pub fn block_bytes(&self) -> usize {
        self.segments.unit_bytes()
    }

    pub fn present_bytes(&self) -> usize {
        self.segments.bytes()
    }
}

impl KvBlockMemory for KvPool {
    fn blocks(&self) -> usize {
        self.segments.segments()
    }

    fn live_blocks(&self) -> usize {
        self.segments.live_units()
    }

    fn layer_tables(&self) -> Vec<(Tensor, Tensor)> {
        self.layer_tables.clone()
    }

    fn block(&self, id: usize) -> Result<Tensor> {
        let Some(buffer) = self.segments.buffer(id) else {
            candle_core::bail!("KV block {id} is absent");
        };
        let es = self.layout.dtype.size_in_bytes();
        Ok(tensor_over(KernelBuffer::new(buffer), &self.md, self.layout.block_bytes() / es, self.layout.dtype))
    }

    /// Map, lock and wire each block, then write its rows' addresses.
    fn take(&self, ids: &[usize]) -> Result<()> {
        let _moves = self.move_lock.lock().expect("KV pool poisoned");
        for &id in ids {
            super::take_unit(&self.segments, &self.md, "KV block", id)?;
            let base = self.segments.address(id);
            for (layer, k, v) in &self.tables {
                let rows = self.layout.rows[*layer].as_ref().expect("a KV layer has rows");
                k.write(id, base + rows.k_offset as u64);
                v.write(id, base + rows.v_offset as u64);
            }
        }
        Ok(())
    }

    /// Null the entries and give the pages back; the caller ensures no command buffer names them.
    fn release(&self, ids: &[usize]) -> Result<()> {
        let _moves = self.move_lock.lock().expect("KV pool poisoned");
        for &id in ids {
            for (_, k, v) in &self.tables {
                k.write(id, 0);
                v.write(id, 0);
            }
            self.segments.release(id).map_err(|e| candle_core::Error::Msg(format!("KV block {id}: {e}")))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mistralrs_core::paged_attention::LayerRows;
    use oscar_offload::segments::padded_unit;

    /// Blocks are taken, addressed and released; sheds and regrows move through the engine's block state, and a held
    /// block is never withdrawn.
    #[test]
    fn blocks_are_taken_addressed_released_shed_and_regrown() -> Result<()> {
        let device = oscar_offload::pools::test_device()?;
        // K 2×4×8×8 and V 2×64×8 bf16: 3,072 bytes a layer, two a block.
        let rows =
            |k_offset: usize| LayerRows { k: vec![2, 4, 8, 8], v: vec![2, 64, 8], k_offset, v_offset: k_offset + 1024 };
        let layout =
            KvBlockLayout { block_size: 8, dtype: DType::BF16, rows: vec![None, Some(rows(0)), Some(rows(3072))] };
        assert_eq!(layout.block_bytes(), 6144);
        let pool = KvPool::build(&device, &layout, 3, 5)?;
        assert_eq!((pool.blocks(), pool.live_blocks(), pool.block_bytes()), (5, 3, padded_unit(layout.block_bytes())));
        let tables = pool.layer_tables();
        assert_eq!((tables.len(), tables[0].0.dims1()?), (3, 1), "a layer without KV gets one null entry");
        let (k1, v2): (Vec<i64>, Vec<i64>) = (tables[1].0.to_vec1()?, tables[2].1.to_vec1()?);
        assert!(k1[..3].iter().all(|&a| a != 0) && k1[3..].iter().all(|&a| a == 0), "{k1:?}");
        assert_eq!(v2[0] - k1[0], layout.rows[2].as_ref().unwrap().v_offset as i64);
        assert_eq!(pool.block(1)?.dims1()?, layout.block_bytes() / 2);
        assert!(pool.block(4).is_err(), "an absent block has no bytes");
        pool.release(&[1])?;
        assert_eq!((pool.live_blocks(), tables[1].0.to_vec1::<i64>()?[1]), (2, 0));
        pool.take(&[1, 4])?;
        assert_eq!(pool.live_blocks(), 4);
        assert_ne!(tables[1].1.to_vec1::<i64>()?[4], 0);
        // The sheds and regrows.
        let rows = LayerRows { k: vec![1, 2, 8, 8], v: vec![1, 16, 8], k_offset: 0, v_offset: 256 };
        let layout = KvBlockLayout { block_size: 8, dtype: DType::BF16, rows: vec![Some(rows)] };
        let (live, index) = (6, 9);
        let pool = KvPool::build(&device, &layout, live, index)?;
        let mut manager = KVCacheManager::new(index, 8, true, vec![0]);
        manager.withdraw_ids(&(live..index).collect::<Vec<_>>());
        assert_eq!((manager.allocatable_blocks(), pool.absent()), (5, 3));
        // A request holds two blocks; three free remain beside the null one.
        assert_eq!(manager.allocate_slots(7, 16, &[]).expect("two blocks").len(), 2);
        assert_eq!(pool.shed(&mut manager, 10), 3, "only the free blocks go");
        assert!(pool.pending());
        let (released, taken, refused) = pool.settle();
        assert!(taken.is_empty() && refused.is_none());
        assert_eq!((released, pool.live_blocks(), pool.absent(), manager.allocatable_blocks()), (3, 3, 6, 2));
        assert_eq!(pool.regrow(4), 4);
        let (released, taken, refused) = pool.settle();
        assert!(refused.is_none());
        assert!(pool.pending(), "the takes are out of the owed list but not yet the allocator's: still coming");
        manager.restore(&taken);
        pool.handed_back();
        assert_eq!((released, taken.len(), pool.live_blocks(), pool.absent()), (0, 4, 7, 2));
        assert_eq!((manager.allocatable_blocks(), manager.num_free_blocks()), (6, 4));
        assert!(!pool.pending());
        manager.free(7);
        Ok(())
    }
}
