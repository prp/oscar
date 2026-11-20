//! A recurrent layer's state borrowed from its pool slot: one sequence's is updated in place, a batch's is copied.

use candle_core::{DType, Result, Shape, Storage, Tensor};
use mistralrs_core::kv_cache::RecurrentStateLayout;
use oscar_offload::segments::padded_unit;

/// Conv `(batch, channels, taps)` (KDA packs q|k|v), recurrent F32 `(batch, heads, head_k_dim, head_v_dim)`; `emits`
/// the chunk positions the layer writes its state at, the engine's for one sequence.
#[derive(Debug, Clone)]
pub struct RecurrentState {
    pub conv_state: Tensor,
    pub recurrent_state: Tensor,
    pub seqlen_offset: usize,
    pub emits: Vec<Emit>,
}

/// The state after the chunk's `at`-th token goes to `conv` and `recurrent`, laid out as the slot's.
#[derive(Debug, Clone)]
pub struct Emit {
    pub at: usize,
    pub conv: Tensor,
    pub recurrent: Tensor,
}

/// Each emit's conv taps: the `taps` columns of `x` (`(b, c, carried + s)`, its first `carried` the chunk's carried
/// window) ending at its position, zero-padded where the prompt starts inside them.
pub(crate) fn emit_conv_taps(x: &Tensor, carried: usize, taps: usize, emits: &[Emit]) -> Result<()> {
    for e in emits {
        let end = carried + e.at;
        let window = match end.checked_sub(taps) {
            Some(from) => x.narrow(2, from, taps)?.contiguous()?,
            None => {
                let pad = Tensor::zeros((x.dim(0)?, x.dim(1)?, taps - end), x.dtype(), x.device())?;
                Tensor::cat(&[pad, x.narrow(2, 0, end)?], 2)?.contiguous()?
            }
        };
        e.conv.slice_set(&window, 2, 0)?;
    }
    Ok(())
}

/// One tensor's run in a state object, `offset` in bytes.
pub struct Run {
    pub offset: usize,
    pub shape: Shape,
    pub dtype: DType,
}

/// Each layer's conv then recurrent run, page-rounded, and the object's bytes: what the pool maps, the plan prices.
pub fn runs_of(layout: &RecurrentStateLayout) -> (Vec<[Run; 2]>, usize) {
    let mut runs = Vec::with_capacity(layout.layers.len());
    let mut offset = 0usize;
    for (conv, conv_dtype, recurrent, recurrent_dtype) in &layout.layers {
        let conv_shape = Shape::from_dims(&[conv.0, conv.1, conv.2]);
        let recurrent_shape = Shape::from_dims(recurrent);
        let mut run = |shape: Shape, dtype: DType| {
            let at = offset;
            offset += padded_unit(shape.elem_count() * dtype.size_in_bytes());
            Run { offset: at, shape, dtype }
        };
        let c = run(conv_shape, *conv_dtype);
        let r = run(recurrent_shape, *recurrent_dtype);
        runs.push([c, r]);
    }
    (runs, offset)
}

/// Same object, run and offset: working memory is one object cut into runs, so a fresh state shares the slot's object.
pub(crate) fn same_buffer(a: &Tensor, b: &Tensor) -> bool {
    let (sa, la) = a.storage_and_layout();
    let (sb, lb) = b.storage_and_layout();
    match (&*sa, &*sb) {
        (Storage::Metal(x), Storage::Metal(y)) => {
            std::ptr::eq(x.buffer().as_ref(), y.buffer().as_ref())
                && x.buffer().base() == y.buffer().base()
                && la.start_offset() == lb.start_offset()
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Result};
    use mistralrs_core::kv_cache::RecurrentStatePool;

    /// Fails if the run is left out: two fresh tensors share the region's object and the zero offset. The drafter
    /// and the prefix cache keep gathers across in-place rewrites; a gather by `Tensor::copy` fails here.
    #[test]
    fn two_fresh_tensors_are_two_buffers_and_a_gathered_state_does_not_move_with_its_slot() -> Result<()> {
        let dev = oscar_offload::pools::test_device()?;
        let a = Tensor::zeros((4, 8), DType::F32, &dev)?;
        let b = Tensor::zeros((4, 8), DType::F32, &dev)?;
        assert!(!same_buffer(&a, &b), "two objects of one region are two objects");
        assert!(same_buffer(&a, &a.reshape((8, 4))?), "a reshape is the same bytes");
        let mut pool = RecurrentStatePool::new(4, 3, vec![2, 4, 4], DType::F32, None, &dev)?;
        let slot = 0u32;
        pool.hold(
            slot as usize,
            Tensor::zeros((1, 4, 3), DType::F32, &dev)?,
            Tensor::zeros((1, 2, 4, 4), DType::F32, &dev)?,
        );
        let first = Tensor::full(1f32, (1, 2, 4, 4), &dev)?;
        pool.scatter_recurrent_state_slots(&[slot], &first)?;
        let kept = pool.gather_recurrent_state(&[slot])?;
        pool.scatter_recurrent_state_slots(&[slot], &Tensor::full(2f32, (1, 2, 4, 4), &dev)?)?;
        assert!(!same_buffer(&kept, &pool.state(slot as usize)?.recurrent), "the gather shares the slot's buffer");
        let (kept, live) = (
            kept.flatten_all()?.to_vec1::<f32>()?,
            pool.state(slot as usize)?.recurrent.flatten_all()?.to_vec1::<f32>()?,
        );
        assert!(kept.iter().all(|&v| v == 1.0), "the gathered state moved with its slot");
        assert!(live.iter().all(|&v| v == 2.0), "the slot took its new state");
        Ok(())
    }
}
