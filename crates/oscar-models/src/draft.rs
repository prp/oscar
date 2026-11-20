//! The drafter's seam: the model it drafts with, its private cache and paged view, and where a layer's state lives.

use std::sync::{Arc, OnceLock};

use candle_core::{Device, Result, Tensor};
use mistralrs_core::kv_cache::{HybridCache, HybridLayerCache, RecurrentStatePool};
use mistralrs_core::text_models_inputs_processor::{PagedAttentionInputMetadata, PagedRows};

use crate::state_slots::{same_buffer, Emit, RecurrentState};

static SELF_DRAFT: OnceLock<bool> = OnceLock::new();

/// The launch's `OSCAR_DRAFT`, set once before a model is built ([`set_self_draft`]); off where never set.
pub fn self_draft() -> bool {
    SELF_DRAFT.get().copied().unwrap_or(false)
}

pub fn set_self_draft(on: bool) {
    let _ = SELF_DRAFT.set(on);
}

/// Only what the pools cannot hold, empty for a dense model and `None` until grounded by a deep copy; mutated in
/// place, so never engine storage.
pub type DraftCache = Vec<Option<RecurrentState>>;

/// The engine's pools (`None` at recurrent layers); a draft writes only the table's drafter-reserved tail blocks.
pub struct DraftPagedView {
    pub pools: Vec<Option<(Tensor, Tensor)>>,
    pub table: Vec<u32>,
    pub block_size: usize,
}

impl DraftPagedView {
    /// Context `[0, rows]` through the table, the new K/V row at `rows`.
    pub(crate) fn decode_metadata(&self, rows: usize, device: &Device) -> Result<PagedAttentionInputMetadata> {
        let nblk = (rows + 1).div_ceil(self.block_size);
        if self.table.len() < nblk {
            candle_core::bail!("draft paged view: table covers {} block(s), row {rows} needs {nblk}", self.table.len());
        }
        let mut step = PagedRows::default();
        let table = self.table[..nblk].iter().map(|&block| block as usize).collect();
        step.push(table, rows..rows + 1, self.block_size);
        step.decode_step(None, self.block_size, std::slice::from_ref(device)).map_err(candle_core::Error::msg)
    }

    pub(crate) fn layer_pools(&self, i: usize) -> Result<(Tensor, Tensor)> {
        match self.pools.get(i).and_then(|p| p.as_ref()) {
            Some(kv) => Ok(kv.clone()),
            None => candle_core::bail!("draft forward: layer {i} expects block-pool handles"),
        }
    }
}

/// Its output is prefetch hints only, so the engine's output is byte-exact whatever a draft does.
pub trait DraftableModel: Send + Sync {
    fn device(&self) -> &Device;
    fn make_private_draft_cache(&self) -> Result<DraftCache>;
    /// One-token decode of `token` at position `rows`: `view`'s context in, the new row into its table's slot.
    fn draft_forward(&self, token: u32, rows: usize, view: &DraftPagedView, cache: &mut DraftCache) -> Result<Tensor>;
}

/// The drafter over `forward_with_cache`: a hybrid's recurrent layers ungrounded (`None`) until it adopts the engine's.
#[macro_export]
macro_rules! draftable {
    ($ty:ty $(, $a:ident)?) => {
        impl$(<$a: $crate::plain::PlainAttention>)? $crate::draft::DraftableModel for $ty {
            fn device(&self) -> &candle_core::Device {
                &self.device
            }
            fn make_private_draft_cache(&self) -> candle_core::Result<$crate::draft::DraftCache> {
                let hybrid_layers = if self.cache.is_hybrid() { self.layers.len() } else { 0 };
                Ok(std::iter::repeat_with(|| None).take(hybrid_layers).collect())
            }
            fn draft_forward(
                &self, token: u32, rows: usize, view: &$crate::draft::DraftPagedView,
                cache: &mut $crate::draft::DraftCache,
            ) -> candle_core::Result<candle_core::Tensor> {
                let input = candle_core::Tensor::new(&[token], &self.device)?.unsqueeze(0)?;
                self.forward_with_cache(&input, &[rows], vec![(0, 1)], view, cache)
            }
        }
    };
}

/// Each sequence's recurrent slot, on the host so a layer reads its slot with no GPU sync.
pub(crate) fn state_indices(cache: &HybridCache) -> Result<Arc<[u32]>> {
    cache.state_indices_host().map(Arc::from).ok_or_else(|| {
        candle_core::Error::Msg(
            "linear-attention layers need recurrent state indices, hybrid cache has none set (clone_in_cache not run?)"
                .into(),
        )
    })
}

/// Layer `i`'s emits for the one slot `indices` names; a batch has none.
pub(crate) fn layer_emits(cache: &HybridCache, indices: &[u32], i: usize) -> Vec<Emit> {
    let part = |(at, conv, recurrent)| Emit { at, conv, recurrent };
    match indices {
        [slot] => cache.layer_emits(*slot as usize, i).into_iter().map(part).collect(),
        _ => Vec::new(),
    }
}

/// Where a forward's layer state lives: the engine's, or the drafter's view of the engine's pools and private states.
pub(crate) enum Source<'a> {
    Engine {
        cache: &'a mut HybridCache,
        indices: Arc<[u32]>,
        metadata: (Vec<(Tensor, Tensor)>, &'a PagedAttentionInputMetadata),
    },
    Draft {
        view: &'a DraftPagedView,
        meta: &'a PagedAttentionInputMetadata,
        private: &'a mut [Option<RecurrentState>],
    },
}

impl<'a> Source<'a> {
    pub(crate) fn engine(
        cache: &'a mut HybridCache, metadata: Option<(Vec<(Tensor, Tensor)>, &'a PagedAttentionInputMetadata)>,
    ) -> Result<Self> {
        let metadata = metadata.ok_or_else(|| candle_core::Error::Msg("paged KV only".into()))?;
        Ok(Source::Engine { indices: state_indices(cache)?, cache, metadata })
    }

    pub(crate) fn is_engine(&self) -> bool {
        matches!(self, Source::Engine { .. })
    }

    /// Layer `i`'s emits, the engine's.
    pub(crate) fn emits(&self, i: usize) -> Vec<Emit> {
        match self {
            Source::Engine { cache, indices, .. } => layer_emits(cache, indices, i),
            Source::Draft { .. } => Vec::new(),
        }
    }

    /// A full-attention layer's block pools and paged metadata.
    pub(crate) fn attention(&self, i: usize) -> Result<((Tensor, Tensor), &PagedAttentionInputMetadata)> {
        match self {
            Source::Engine { metadata: (kv, meta), .. } => Ok((kv[i].clone(), *meta)),
            Source::Draft { view, meta, .. } => Ok((view.layer_pools(i)?, *meta)),
        }
    }

    /// `taps` is the width the layer runs over where the conv row is wider (qwen4exp's PLE columns).
    pub(crate) fn recurrent(&mut self, i: usize, taps: Option<usize>) -> Result<LayerState<'_>> {
        match self {
            Source::Engine { cache, indices, .. } => LayerState::engine(cache, indices, i, taps),
            Source::Draft { private, .. } => {
                let Some(Some(cache)) = private.get_mut(i) else {
                    candle_core::bail!("draft forward: layer {i} has no grounded private state");
                };
                let row = match taps {
                    Some(w) => {
                        let row = cache.conv_state.clone();
                        cache.conv_state = row.narrow(2, 0, w)?.contiguous()?;
                        Some(row)
                    }
                    None => None,
                };
                Ok(LayerState::Draft { cache, _row: row })
            }
        }
    }
}

/// The engine's slot, a view the kernels write in place, or the drafter's private state, updated in place.
pub(crate) enum LayerState<'s> {
    Engine {
        cache: RecurrentState,
        /// The slot's own `(conv, recurrent)` for one slot, gathered copies for several.
        slot: (Tensor, Tensor),
        pool: &'s mut RecurrentStatePool,
        indices: &'s [u32],
        first_offset: usize,
    },
    /// `_row` holds the whole conv row while the layer runs over its taps.
    Draft { cache: &'s mut RecurrentState, _row: Option<Tensor> },
}

impl<'s> LayerState<'s> {
    /// The slot `indices` names for layer `i`, one offset across the batch.
    pub(crate) fn engine(
        cache: &'s mut HybridCache, indices: &'s [u32], i: usize, taps: Option<usize>,
    ) -> Result<Self> {
        let emits = layer_emits(cache, indices, i);
        let Some(HybridLayerCache::Recurrent(pool)) = cache.get_mut(i) else {
            candle_core::bail!("forward: layer {i} linear-attention, its cache slot not");
        };
        if indices.is_empty() {
            candle_core::bail!("forward: recurrent state_indices empty");
        }
        let first_offset = pool.get_seqlen_offset(indices[0] as usize);
        if indices.iter().any(|&idx| pool.get_seqlen_offset(idx as usize) != first_offset) {
            candle_core::bail!("forward: recurrent seqlen offsets diverged within batch at layer {i}");
        }
        let slot = match indices {
            [slot] => {
                let state = pool.state(*slot as usize)?;
                (state.conv.clone(), state.recurrent.clone())
            }
            _ => (pool.gather_conv_state(indices)?, pool.gather_recurrent_state(indices)?),
        };
        let conv_state = match taps {
            Some(w) => slot.0.narrow(2, 0, w)?.contiguous()?,
            None => slot.0.clone(),
        };
        let cache = RecurrentState { conv_state, recurrent_state: slot.1.clone(), seqlen_offset: first_offset, emits };
        Ok(LayerState::Engine { cache, slot, pool, indices, first_offset })
    }

    pub(crate) fn cache(&mut self) -> &mut RecurrentState {
        match self {
            LayerState::Engine { cache, .. } => cache,
            LayerState::Draft { cache, .. } => cache,
        }
    }

    /// After the layer: `extra` (the conv row past the taps) rejoins, the engine's slot written back and advanced.
    pub(crate) fn commit(&mut self, extra: &[Tensor]) -> Result<()> {
        let row = match extra {
            [] => None,
            _ => Some(Tensor::cat(&[std::slice::from_ref(&self.cache().conv_state), extra].concat(), 2)?),
        };
        match self {
            LayerState::Engine { cache, slot, pool, indices, first_offset } => {
                let delta = cache.seqlen_offset.saturating_sub(*first_offset);
                // Nothing for a state still in its view; a scatter for a copy or a fresh tensor.
                let conv = row.as_ref().unwrap_or(&cache.conv_state);
                if !same_buffer(conv, &slot.0) {
                    pool.scatter_conv_state_slots(indices, conv)?;
                }
                if !same_buffer(&cache.recurrent_state, &slot.1) {
                    pool.scatter_recurrent_state_slots(indices, &cache.recurrent_state)?;
                }
                for &idx in indices.iter() {
                    let updated = pool.get_seqlen_offset(idx as usize) + delta;
                    pool.set_seqlen_offset(idx as usize, updated);
                }
            }
            LayerState::Draft { cache, .. } => {
                if let Some(row) = row {
                    cache.conv_state = row;
                }
            }
        }
        Ok(())
    }
}
