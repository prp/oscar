//! A ring of shared buffers for the prefill waves' address rows and expert lists, so no wave allocates.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use candle_core::{DType, MetalDevice, Result, Tensor};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLSharedEvent};

use oscar_kernels::tensor_over;
use oscar_offload::segments::SegmentPool;

use super::{words, Words};

/// Ring claims that waited on the slot's reuse fence; none unless run-ahead outgrows the ring.
pub static WAVE_RING_WRAPS: AtomicU64 = AtomicU64::new(0);

pub(crate) struct WavePool {
    /// Each: three rows of `num_experts` u64s, then the expert list of `num_experts` u32s.
    slots: Vec<Words<u32>>,
    num_experts: usize,
    cursor: AtomicUsize,
    /// The generation each slot's last user committed at; 0 = never used.
    slot_gen: Vec<AtomicU64>,
}

impl WavePool {
    /// Twice the worst wave fan-out, about 32.
    const RING_SIZE: usize = 64;

    pub(crate) fn new(device: &ProtocolObject<dyn MTLDevice>, num_experts: usize) -> std::result::Result<Self, String> {
        let slots = (0..Self::RING_SIZE)
            .map(|_| words(device, 7 * num_experts.max(1), 0, "WavePool"))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Self {
            slots,
            num_experts,
            cursor: AtomicUsize::new(0),
            slot_gen: (0..Self::RING_SIZE).map(|_| AtomicU64::new(0)).collect(),
        })
    }

    fn next_slot(&self) -> usize {
        self.cursor.fetch_add(1, Ordering::Relaxed) % Self::RING_SIZE
    }

    /// The next slot, past its last user's generation: by size alone a raised queue cap wraps onto an unstarted buffer.
    pub(crate) fn claim_slot(&self, metal_device: &MetalDevice, timeout: std::time::Duration) -> Result<usize> {
        let idx = self.next_slot();
        let need = self.slot_gen[idx].load(Ordering::Acquire);
        if need == 0 {
            return Ok(idx);
        }
        let event = metal_device.completion_event()?;
        if event.signaledValue() >= need {
            return Ok(idx);
        }
        if WAVE_RING_WRAPS.fetch_add(1, Ordering::Relaxed) == 0 {
            tracing::error!(target: "oscar::experts",
                "wave pool: ring wrapped onto live slot {idx} - previous wave committed at generation {need}, GPU \
                 reached {}; without reuse fence routing table overwritten mid-read",
                event.signaledValue()
            );
        }
        let deadline = std::time::Instant::now() + timeout;
        while event.signaledValue() < need {
            assert!(
                std::time::Instant::now() < deadline,
                "wave pool: ring reuse TIMEOUT on slot {idx} (needs generation {need}, signalled {}) - \
                 a wave written over it would read stale routing",
                event.signaledValue(),
            );
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
        Ok(idx)
    }

    /// After the wave's flush: the generation slot `idx`'s next user waits for.
    pub(crate) fn publish_slot_gen(&self, idx: usize, gen: u64) {
        self.slot_gen[idx].store(gen, Ordering::Release);
    }

    /// Each bank's row `experts[i]` → `slots[i]`'s address, null elsewhere, as `I64` rows; the list as `U32`.
    pub(crate) fn write_rows(
        &self, idx: usize, metal_device: &MetalDevice, experts: &[u16], banks: [&SegmentPool; 3], slots: [&[usize]; 3],
    ) -> Result<([Tensor; 3], Tensor)> {
        let slot = &self.slots[idx];
        let e = self.num_experts;
        debug_assert!(experts.len() <= e);
        // SAFETY: the slot is this wave's until its command buffer completes (`claim_slot`); one thread writes.
        unsafe {
            let rows = slot.ptr() as *mut u64;
            std::ptr::write_bytes(rows, 0, 3 * e);
            for (p, (bank, bank_slots)) in banks.iter().zip(slots).enumerate() {
                debug_assert_eq!(bank_slots.len(), experts.len());
                let row = rows.add(p * e);
                for (&expert, &s) in experts.iter().zip(bank_slots) {
                    *row.add(expert as usize) = bank.address(s);
                }
            }
            let list = slot.ptr().add(6 * e);
            for (i, &expert) in experts.iter().enumerate() {
                *list.add(i) = expert as u32;
            }
        }
        let rows = tensor_over(slot.view().clone(), metal_device, 3 * e, DType::I64);
        let list = tensor_over(slot.view().clone(), metal_device, 7 * e, DType::U32).narrow(0, 6 * e, experts.len())?;
        Ok(([rows.narrow(0, 0, e)?, rows.narrow(0, e, e)?, rows.narrow(0, 2 * e, e)?], list))
    }
}

/// Slabs `[0,K),[K,2K),…` of expert ids, sorted, the cut the prefetcher makes too, so their ticks align.
pub(crate) fn waves(ids: &mut [u16], width: usize) -> impl Iterator<Item = &[u16]> {
    ids.sort_unstable();
    ids.chunk_by(move |a, b| *a as usize / width == *b as usize / width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_waves_are_the_slabs_of_the_wave_width_without_the_empty_ones() {
        // (ids, wave width, the waves).
        let rows: [(&[u16], usize, &[&[u16]]); 7] = [
            (&[0, 5, 9, 60, 130], 4, &[&[0], &[5], &[9], &[60], &[130]]),
            (&[], 4, &[]),
            (&[7, 5], 4, &[&[5, 7]]),
            (&[0, 1, 7, 8, 9, 14, 15], 8, &[&[0, 1, 7], &[8, 9, 14, 15]]),
            (&[0, 1, 2, 3], 4, &[&[0, 1, 2, 3]]),
            (&[0, 5, 9], 32, &[&[0, 5, 9]]),
            (&[0, 3, 5, 8, 9], 4, &[&[0, 3], &[5], &[8, 9]]),
        ];
        for (ids, width, want) in rows {
            let got: Vec<&[u16]> = waves(&mut ids.to_vec(), width).map(|w| w.to_vec().leak() as &[u16]).collect();
            assert_eq!(got, want, "ids {ids:?}, width {width}");
        }
        // Unsorted, some in every slab: each lands once, in its slab's wave; a dropped expert would corrupt the output.
        let mut ids: Vec<u16> = (0..128).filter(|e| e % 3 == 0).rev().collect();
        let all: Vec<u16> = waves(&mut ids.clone(), 16).flatten().copied().collect();
        ids.sort_unstable();
        assert_eq!(all, ids, "each id once");
        for (w, wave) in waves(&mut ids, 16).enumerate() {
            assert!(wave.iter().all(|&e| e as usize / 16 == w), "wave {w}: {wave:?}");
        }
    }
}
