//! The drafter's kept routing, hints window and a decode that pins nothing, all on its thread: the output stays exact.

use super::*;

use std::collections::VecDeque;
use std::sync::Mutex;

use crate::gpu::{words, words_tensor, Words};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;
use oscar_kernels::tensor_over;

thread_local! {
    // (layer, ids, weights, wide selection), read back after the drafter's logits sync at no extra sync.
    static DRAFT_ROUTING_STASH: std::cell::RefCell<
        Vec<(usize, candle_core::Tensor, candle_core::Tensor, Option<candle_core::Tensor>)>,
    > = const { std::cell::RefCell::new(Vec::new()) };
}

/// The routed top-k and then the runner-ups, read only while the eval scores hints.
const DRAFT_HINT_WIDTH: usize = 16;

const DRAFT_WINDOW_TOKENS: usize = 16;

pub(super) struct DraftScratch {
    pub(super) scratch: LaneScratch,
    pub(super) snapshot: DraftSnapshot,
    /// Cumulative: a delta from `dirty_last`, its value at the previous check, is a read torn inside this token.
    pub(super) dirty: MissCounter,
    pub(super) dirty_last: AtomicU64,
}

impl OffloadDispatcher {
    /// Once: the window outlives crossings, and each write pushes the oracle its claims and rollout seed.
    pub(super) fn arm_draft_window(&self) {
        let dw = Arc::new(DraftedWindow::new(self.layout.block_count, DRAFT_WINDOW_TOKENS));
        if self.draft_window.set(dw).is_ok() {
            tracing::info!(target: "oscar::drafter",
                "self-draft K={} - next tokens' experts prefetched ahead of decode",
                crate::config::params().draft_k
            );
        }
    }

    /// Whether a key is resident in the first bank now, the eval's hint-precision question.
    fn residency(&self) -> impl Fn(LayerExpertKey) -> bool + '_ {
        let bank = self.pool.first_bank();
        move |key| bank.is_resident(key)
    }

    /// The draft token's per-layer hints, the kept ids cleared: read back here, settled by the caller's logits sync.
    pub fn take_draft_routings(&self) -> (Vec<Vec<u16>>, RunnerUps) {
        let block_count = self.layout.block_count;
        let stashed: Vec<(usize, Tensor, Tensor, Option<Tensor>)> =
            DRAFT_ROUTING_STASH.with(|s| std::mem::take(&mut *s.borrow_mut()));
        if stashed.is_empty() {
            return Default::default();
        }
        let hot_budget = self.hot_budget();
        // The θ below which a hint is not worth staging, with `OSCAR_DRAFT_APPROX_ROUTING` and θ armed; else `None`.
        let params = crate::config::params();
        let theta_filter = params.approx_theta.filter(|_| params.draft_approx_routing);
        let mut sig = RunnerUps::default();
        sig.extras = vec![Vec::new(); block_count];
        // The weights are read back only for the θ filter, which no user's run arms.
        let want_ws = theta_filter.is_some();
        let mut out = vec![Vec::new(); block_count];
        for (layer_idx, ids, ws, wide) in stashed {
            let host: Vec<u32> = match ids.flatten_all().and_then(|t| t.to_vec1()) {
                Ok(v) => v,
                Err(_) => continue, // hint-quality only; never fail a draft
            };
            // The runner-ups: the wide selection past the routed width, best first.
            if let Some(w) = wide.and_then(|t| t.flatten_all().and_then(|t| t.to_vec1::<u32>()).ok()) {
                if let Some(x) = sig.extras.get_mut(layer_idx) {
                    *x = w
                        .iter()
                        .skip(host.len())
                        .filter(|e| !host.contains(e) && **e <= u32::from(u16::MAX))
                        .map(|&e| e as u16)
                        .collect();
                }
            }
            // Weights pair with ids by position; on a length mismatch both consumers stand down.
            let weighted: Option<Vec<(u32, f32)>> = want_ws
                .then(|| ws.flatten_all().and_then(|t| t.to_vec1::<f32>()).ok())
                .flatten()
                .filter(|wsv| wsv.len() == host.len())
                .map(|wsv| {
                    let mut w: Vec<(u32, f32)> = host.iter().copied().zip(wsv).collect();
                    // A resident hint passes: it costs no staging and keeps its eviction protection as the target.
                    if let Some(theta) = theta_filter {
                        let (banks, l16) = (self.pool.banks_for(layer_idx), layer_idx as u16);
                        w.retain(|&(id, w)| w >= theta || banks.iter().all(|b| b.is_resident((l16, id as u16))));
                    }
                    w
                });
            let flat: Vec<u32> = match &weighted {
                Some(w) => w.iter().map(|&(id, _)| id).collect(),
                None => host,
            };
            let uniq = dedup_u32_to_u16(&flat, hot_budget);
            if let Some(slot) = out.get_mut(layer_idx) {
                *slot = uniq;
            }
        }
        (out, sig)
    }

    /// After the take's sync, a dirty-count delta since the last check is a slot torn during this token's matvecs.
    pub fn draft_token_was_torn(&self) -> bool {
        let total = self.draft.dirty.load();
        self.draft.dirty_last.swap(total, Ordering::Relaxed) != total
    }

    /// The draft-read fence's retire side, once the drafter's stream is synced: every bank's window closes.
    pub fn end_draft_forward(&self) {
        self.pool.end_draft_forward();
    }

    /// Front is the engine's next token; a diverged boundary replaces the window whole, its entries wrong-lineage.
    pub fn push_draft_window(&self, window: Vec<Vec<Vec<u16>>>, diverged: bool) {
        if let Some(dw) = self.draft_window.get() {
            let eff = if diverged { dw.replace(window) } else { dw.boundary_refresh(window) };
            self.eval.note_draft_window(&eff, &self.residency());
            self.feed_draft_window(&eff);
        }
    }

    /// To the oracle at every update, so eviction never lags the drafter, then to staging; skipped experts count too.
    fn feed_draft_window(&self, window: &[Vec<Vec<u16>>]) {
        let pool = &self.pool;
        // Claims take the whole window: a wrong claim costs one eviction decision, not a read, so depth pays here.
        if let Some(oracle) = pool.oracle() {
            oracle.note_draft_window(pool.tick.load(Ordering::Acquire), window);
        }
        self.stage_draft_window(window);
    }

    /// `window[d]`, `d + 1` tokens ahead, is offered at its deadline; the queue refuses resident and in-flight keys.
    fn stage_draft_window(&self, window: &[Vec<Vec<u16>>]) {
        let pool = &self.pool;
        let staged = crate::config::params().ep.staging_planes & crate::config::params::PLANE_DRAFT != 0;
        let Some(SchedulePacer::Procedural(pacer)) = pool.pacer().filter(|_| staged) else {
            return;
        };
        // Aimed at tokens the engine has not started.
        let cur_cc = pool.tick.load(Ordering::Acquire) / pacer.ticks_per_token().max(1);
        let ahead =
            window.iter().take(crate::config::params().ep.draft_stage_lead_tokens).enumerate().flat_map(|(d, tok)| {
                let cc = cur_cc + 2 + d as u32;
                tok.iter().enumerate().map(move |(l, e)| (l, pacer.target_tick(l, cc), e.as_slice()))
            });
        pool.offer_window(ahead);
    }

    /// The router's ids stay on the GPU, kept for the hint readback; the layer runs against a residency snapshot.
    pub(super) fn forward_decode_draft(&self, call: Call<'_>) -> Result<Finalized> {
        let logits = call.layer.gate_forward(&call.xs)?;
        let (ids, weights) = call.layer.route_topk(&logits)?;
        // The original top-k, before any θ substitution: what the engine routes, so what hints predict.
        let wide = (self.eval.scores_hints() && ids.elem_count() < DRAFT_HINT_WIDTH)
            .then(|| oscar_kernels::metal_topk_softmax(&logits, DRAFT_HINT_WIDTH, false))
            .and_then(|r| r.ok())
            .map(|(_, ids)| ids);
        DRAFT_ROUTING_STASH.with(|s| s.borrow_mut().push((call.idx, ids.clone(), weights.clone(), wide)));
        let pools = BankPools::new(&call.device, call.banks);
        self.dispatch_draft_layer(call, &pools, &ids, &weights, &logits)
    }
}

impl OffloadDispatcher {
    pub(super) fn lane_scratch(&self) -> &LaneScratch {
        if on_draft_thread() {
            &self.draft.scratch
        } else {
            &self.ep.scratch
        }
    }
    /// No pins or waits: the GPU nulls a slot whose epoch moved, and a slot torn during the matvecs poisons the token.

    pub(super) fn dispatch_draft_layer(
        &self, call: Call<'_>, pools: &BankPools, topk_ids_u32: &Tensor, topk_weights_f32: &Tensor,
        router_logits: &Tensor,
    ) -> Result<Finalized> {
        let (layer, layer_idx, shape, num_tokens) = (call.layer, call.idx, call.shape, call.tokens);
        let topk = topk_ids_u32.elem_count() / num_tokens.max(1);
        let (metal_device, num_experts) = (&call.device, shape.num_experts);
        let (snapshot, dirty) = (&self.draft.snapshot, &self.draft.dirty);

        // Experts the pool does not hold are skip-zeroed: the drafter's output is hints, where approximate is fine.
        let rows = validate_layer_rows(snapshot, &call, &self.draft.scratch.rows)?;
        let [scr_g, scr_u, scr_d] = &rows.rows;

        // The fence after the snapshot write: a claim before it is seen, one after commits behind this buffer.
        let hi = layer_idx as u32 + 1;
        call.banks.iter().for_each(|b| b.note_draft_encoded(hi));

        // Each cold selection becomes the best usable expert (θ = 2) in fresh tensors: the kept ids are the original.
        let mut topk_ids_2d = topk_ids_u32.reshape((num_tokens, topk))?;
        let mut topk_weights_2d = topk_weights_f32.reshape((num_tokens, topk))?;
        if crate::config::params().draft_approx_routing
            && num_tokens == 1
            && topk <= 16
            && layer.gate_selection_bias().is_none()
            && layer.norm_topk_prob()
        {
            let sub = || -> Result<(Tensor, Tensor)> {
                let logits_c = router_logits.to_dtype(DType::F32)?.contiguous()?;
                oscar_kernels::oscar_draft_theta_substitute(
                    scr_g,
                    &logits_c,
                    &topk_ids_2d.contiguous()?,
                    &topk_weights_2d.contiguous()?,
                    topk,
                    num_experts,
                    2.0,
                )
            };
            (topk_ids_2d, topk_weights_2d) = sub()?;
        }

        // Stale or cold entries count on the drafter's miss counter.
        let miss_tensor = self.miss_slot(&call)?;
        let routed_2d = route_rows(
            layer,
            &call.xs,
            pools,
            [scr_g, scr_u, scr_d],
            None,
            &topk_ids_2d,
            &topk_weights_2d,
            &miss_tensor,
            num_tokens,
            None,
        )?;

        // The recheck and the fence's retire run over the executed ids: the slots the matvecs read.
        let dirty_t = dirty.as_tensor(metal_device)?;
        rows.epoch_recheck(&topk_ids_2d, &dirty_t, num_experts, hi as i64)?;

        // No per-layer flush or wait: nothing is pinned, and the token's sampling readback waits the tail.
        layer.finalize_routed(Some(&routed_2d), &call.xs, call.stream, shape)
    }
}

struct ValidatedRows {
    rows: [Tensor; 3],
    snapshot: [Tensor; 3],
    /// The snapshot's epochs, which the recheck holds against the live ones.
    expected: [Tensor; 3],
    /// The live slot epochs, viewed without a copy.
    epochs: [Tensor; 3],
    slot_counts: [usize; 3],
    watermarks: [candle_metal_kernels::metal::Buffer; 3],
}

impl ValidatedRows {
    /// A slot torn after the validation bumps `counter`; `wm_val` (`layer + 1`) retires the layer's draft-read fence.
    fn epoch_recheck(&self, ids_exec_2d: &Tensor, counter: &Tensor, num_experts: usize, wm_val: i64) -> Result<()> {
        oscar_kernels::oscar_draft_epoch_recheck(
            ids_exec_2d,
            [&self.snapshot[0], &self.snapshot[1], &self.snapshot[2]],
            [&self.rows[0], &self.rows[1], &self.rows[2]],
            [&self.expected[0], &self.expected[1], &self.expected[2]],
            [&self.epochs[0], &self.epochs[1], &self.epochs[2]],
            counter,
            num_experts,
            self.slot_counts,
            (&self.watermarks, wm_val),
        )
    }
}

/// The GPU checks the snapshot against the live epochs before any matvec: eviction bumps the epoch before refilling.
fn validate_layer_rows(snapshot: &DraftSnapshot, call: &Call<'_>, lane: &ExpertRows) -> Result<ValidatedRows> {
    let (layer_idx, num_experts, device) = (call.idx, call.shape.num_experts, &call.device);
    let banks = call.banks;

    for (p, bank) in banks.iter().enumerate() {
        snapshot.write_row(p, layer_idx, bank);
    }

    // Views, no copy: the live slot epochs (host bumps race the reads by design) and the slot addresses.
    use candle_metal_kernels::metal::Buffer as KernelBuffer;
    let slot_counts = banks.map(|b| b.slot_count());
    let epochs = banks.map(|b| tensor_over(KernelBuffer::new(b.epoch_buffer()), device, b.slot_count(), DType::U32));
    let tables =
        banks.map(|b| tensor_over(KernelBuffer::new(b.pool().address_buffer()), device, b.slot_count(), DType::I64));
    let (snap_rows, expected) = snapshot.layer_rows(device, layer_idx)?;
    let rows = lane.layer_rows(device, layer_idx)?;
    oscar_kernels::oscar_draft_row_validate(
        [&snap_rows[0], &snap_rows[1], &snap_rows[2]],
        [&expected[0], &expected[1], &expected[2]],
        [&epochs[0], &epochs[1], &epochs[2]],
        [&tables[0], &tables[1], &tables[2]],
        [&rows[0], &rows[1], &rows[2]],
        num_experts,
        slot_counts,
    )?;
    let watermarks = banks.map(|b| KernelBuffer::new(b.draft_watermark_buffer()));
    Ok(ValidatedRows { rows, snapshot: snap_rows, expected, epochs, slot_counts, watermarks })
}

/// Per projection, each expert's slot (the slot count if absent) and its epoch as seen.
pub(super) struct DraftSnapshot {
    slots: [Words<u32>; 3],
    epochs: [Words<u32>; 3],
    num_experts: usize,
}

impl DraftSnapshot {
    pub(super) fn new(
        device: &ProtocolObject<dyn MTLDevice>, num_experts: usize, block_count: usize,
    ) -> std::result::Result<Self, String> {
        let mk = || words(device, num_experts * block_count, 0, "DraftSnapshot");
        Ok(Self { slots: [mk()?, mk()?, mk()?], epochs: [mk()?, mk()?, mk()?], num_experts })
    }

    /// A layer's slot rows, then epoch rows, as `U32 [num_experts]` views.
    fn layer_rows(&self, device: &candle_core::MetalDevice, layer_idx: usize) -> Result<([Tensor; 3], [Tensor; 3])> {
        let n = self.num_experts;
        let view = |b: &Words<u32>| words_tensor(b, device, DType::U32, layer_idx * n, n);
        Ok((
            [view(&self.slots[0])?, view(&self.slots[1])?, view(&self.slots[2])?],
            [view(&self.epochs[0])?, view(&self.epochs[1])?, view(&self.epochs[2])?],
        ))
    }

    /// Projection `p`'s dense snapshot of a layer, unpinned and racy: the GPU validation is the authority at execution.
    fn write_row(&self, p: usize, layer_idx: usize, bank: &Bank) {
        let (num_experts, slots, epochs) = (self.num_experts, &self.slots[p], &self.epochs[p]);
        let off = layer_idx * num_experts;
        debug_assert!(off + num_experts <= slots.count());
        debug_assert!(off + num_experts <= epochs.count());
        let absent = bank.slot_count() as u32;
        // SAFETY: layer-disjoint windows, one drafter-thread writer, and the reading command buffer committed after.
        unsafe {
            let sdst = slots.ptr().add(off);
            let edst = epochs.ptr().add(off);
            // The map → epoch → map recheck (`observe_resident`) closes the claim-between-reads window; no allocation,
            // as the dense draft snapshot runs it per expert, layer and token.
            for e in 0..num_experts {
                let found = bank.observe_resident((layer_idx as u16, e as u16));
                let (slot, epoch) = found.map_or((absent, 0), |r| (r.slot as u32, r.epoch));
                *sdst.add(e) = slot;
                *edst.add(e) = epoch;
            }
        }
    }
}

/// Per layer, the routed expert ids deduplicated; empty is no hint.
pub type TokenHints = Vec<Vec<u16>>;

/// The next tokens' per-layer routings, touching the offload only; a re-sync replaces it whole, so ±1 lasts a token.
#[derive(Debug)]
pub struct DraftedWindow {
    block_count: usize,
    capacity: usize,
    inner: Mutex<VecDeque<TokenHints>>,
}

impl DraftedWindow {
    pub fn new(block_count: usize, capacity: usize) -> Self {
        Self { block_count, capacity: capacity.max(1), inner: Mutex::new(VecDeque::new()) }
    }

    /// `tokens[d]` holds the hints `d + 1` tokens past the engine's newest; returns the window as truncated.
    pub fn replace(&self, mut tokens: Vec<TokenHints>) -> Vec<TokenHints> {
        tokens.truncate(self.capacity);
        for t in tokens.iter_mut() {
            t.truncate(self.block_count);
        }
        let mut w = self.inner.lock().expect("DraftedWindow poisoned");
        w.clear();
        w.extend(tokens.iter().cloned());
        tokens
    }

    /// A window deeper than `mirror` is kept, slid a token, else replaced; a diverged boundary must [`Self::replace`].
    pub fn boundary_refresh(&self, mirror: Vec<TokenHints>) -> Vec<TokenHints> {
        {
            let mut w = self.inner.lock().expect("DraftedWindow poisoned");
            if w.len() > mirror.len() + 1 {
                w.pop_front();
                return w.iter().cloned().collect();
            }
        }
        self.replace(mirror)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every token's hints as the window holds them, distance 0 first.
    fn held(w: &DraftedWindow) -> Vec<TokenHints> {
        w.inner.lock().unwrap().iter().cloned().collect()
    }

    /// The window holds what a replace returns: the tokens cut to its capacity, each to the model's layers. A boundary
    /// refresh slides a window deeper than its mirror, else replaces it.
    #[test]
    fn a_replace_holds_the_window_cut_to_its_capacity_and_a_boundary_refresh_slides_or_replaces_it() {
        let w = DraftedWindow::new(3, 4);
        let pushed = vec![vec![vec![1, 2], vec![], vec![3]], vec![vec![4], vec![5], vec![]]];
        assert_eq!(w.replace(pushed.clone()), pushed);
        assert_eq!(held(&w), pushed);
        assert!(w.replace(Vec::new()).is_empty() && held(&w).is_empty(), "an empty push clears it");
        let w = DraftedWindow::new(2, 2);
        let eff = w.replace(vec![vec![vec![1], vec![2], vec![9]], vec![vec![3], vec![]], vec![vec![7], vec![]]]);
        assert_eq!(eff, vec![vec![vec![1], vec![2]], vec![vec![3], vec![]]], "the third layer and token dropped");
        assert_eq!(held(&w), eff);
        let w = DraftedWindow::new(1, 4);
        w.replace(vec![vec![vec![1]], vec![vec![2]], vec![vec![3]]]);
        let eff = w.boundary_refresh(vec![vec![vec![9]]]);
        assert_eq!(eff, vec![vec![vec![2]], vec![vec![3]]], "a mirror of depth 1 against 3 pushed: kept, slid a token");
        assert_eq!(held(&w), eff);
        let eff = w.boundary_refresh(vec![vec![vec![7]], vec![vec![8]]]);
        assert_eq!(eff, vec![vec![vec![7]], vec![vec![8]]], "a mirror as deep as the window replaces it");
        assert_eq!(held(&w), eff);
    }
}
