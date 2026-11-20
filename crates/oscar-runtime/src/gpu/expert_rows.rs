//! Per-layer expert address rows, each expert's bytes in its bank or null: one shared buffer per projection.

use candle_core::{DType, MetalDevice, Result, Tensor};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use super::{words, words_tensor, Words};

pub(crate) struct ExpertRows {
    pub(crate) rows: [RowBank; 3],
    pub(crate) num_experts: usize,
}

impl ExpertRows {
    /// `I64 [num_experts]` windows, valid at encode: the resolver's writes land before the event releasing the kernel.
    pub fn layer_rows(&self, device: &MetalDevice, layer_idx: usize) -> Result<[Tensor; 3]> {
        let n = self.num_experts;
        let [g, u, d] = self.rows.each_ref().map(|b| words_tensor(b, device, DType::I64, layer_idx * n, n));
        Ok([g?, u?, d?])
    }

    pub(crate) fn new(
        device: &ProtocolObject<dyn MTLDevice>, num_experts: usize, block_count: usize,
    ) -> std::result::Result<Self, String> {
        // Zero is the null address an unwritten row must carry.
        let mk = || words(device, num_experts * block_count, 0, "ExpertRows");
        Ok(Self { rows: [mk()?, mk()?, mk()?], num_experts })
    }
}

/// One projection's address rows, one per layer.
pub(crate) type RowBank = Words<u64>;

/// U32 tallies a kernel bumps: a lane's unresolved routed slots per layer, or the drafter's torn reads.
pub(crate) struct MissCounter(Words<u32>);

impl MissCounter {
    pub(crate) fn new(device: &ProtocolObject<dyn MTLDevice>, slots: usize) -> std::result::Result<Self, String> {
        words(device, slots, 0, "MissCounter").map(Self)
    }

    /// The sum over every slot, valid once the command buffer bumping them completes.
    pub(crate) fn load(&self) -> u64 {
        // SAFETY: the buffer outlives self; volatile reads past the CB barrier.
        (0..self.0.count()).map(|i| unsafe { self.0.ptr().add(i).read_volatile() } as u64).sum()
    }

    /// A U32 tensor, for the caller to narrow to a layer's slot.
    pub(crate) fn as_tensor(&self, metal_device: &MetalDevice) -> Result<Tensor> {
        words_tensor(&self.0, metal_device, DType::U32, 0, self.0.count())
    }
}

/// Each `(expert, address)` entry into the layer's row, the rest null.
pub(crate) fn write_row(
    layer_idx: usize, num_experts: usize, bank: &RowBank, entries: impl Iterator<Item = (u16, u64)>,
) {
    let off = layer_idx * num_experts;
    debug_assert!(off + num_experts <= bank.count());
    // SAFETY: a layer-disjoint window, written by the one forward thread before the reading command buffer commits.
    unsafe {
        let dst = bank.ptr().add(off);
        std::ptr::write_bytes(dst, 0, num_experts);
        for (e, address) in entries {
            debug_assert!((e as usize) < num_experts);
            *dst.add(e as usize) = address;
        }
    }
}
