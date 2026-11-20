//! Every predictor of the routing ahead of its demand; they only stage, so a layer's true path recovers any miss.

use super::*;

use std::time::Duration;

use crate::engine::event_lane::{gen_byte, poll, refill, Block, Resolver, WorkItem};

/// `Some(stage_io)` if staging or eviction reads plane `src`; not a gate on the GPU prediction, which paces prefill.
pub(crate) fn consumers(oracle: Option<&dyn crate::pool::Oracle>, plane: u8) -> Option<bool> {
    let io = crate::config::params().ep.staging_planes & plane != 0;
    (io || oracle.is_some_and(|o| o.feeds_near_claims())).then_some(io)
}

impl OffloadDispatcher {
    /// Once per layer, from every readback, so the book is complete from the second pass on.
    pub(super) fn snapshot_gate(&self, layer_idx: usize, layer: &RoutedLayer) {
        if let Some(cell) = self.gate_book.get(layer_idx) {
            cell.get_or_init(|| (layer.gate_matmul(), layer.gate_selection_bias()));
        }
    }

    /// `layer_idx`'s gate and selection bias, if some pass has seen that layer.
    pub(super) fn pred_gate(&self, layer_idx: usize) -> Option<(candle_core::quantized::QMatMul, Option<Tensor>)> {
        self.gate_book.get(layer_idx)?.get().cloned()
    }

    /// The experts of a layer treated as the conversation's head, capped at a quarter of the pool to stay a minority.
    pub(super) fn auto_head(pool: &ExpertPool, routed_width: usize) -> usize {
        (routed_width / 2).min(pool.hot_budget() / 4).max(1)
    }

    /// The handover's prediction: each next layer's router on this layer's hidden, up to a gate not in the book.
    pub(super) fn predict_layers_ahead(
        &self, first: usize, count: usize, xs_for_moe: &Tensor, topk: usize, keep: usize,
    ) -> Vec<Vec<u16>> {
        // One readback for every predicted layer: a GPU sync per layer costs what streaming saves.
        let ranked = |all: Vec<u32>, n: usize| -> Vec<Vec<u16>> {
            let per = all.len() / n.max(1);
            (0..n)
                .map(|i| {
                    crate::engine::router::rank_by_token_demand(
                        &all[i * per..(i + 1) * per],
                        self.layout.num_experts,
                        keep,
                    )
                })
                .collect()
        };
        // The routed top-k: the experts it misses are not in the tail of a wider ranking.
        let cand_k = topk.min(self.layout.num_experts);
        // The gated become `num_experts`, a sentinel the vote ignores, so the shape and the one readback stay.
        let margin = crate::config::params().ep.lookahead_stage_margin;
        let predict = || -> Result<Vec<Vec<u16>>> {
            let mut rows = Vec::with_capacity(count);
            for step in 0..count {
                let Some((gate, bias)) = self.pred_gate(first + step) else {
                    break;
                };
                let scores = crate::engine::router::selection_scores(&gate, bias.as_ref(), xs_for_moe)?;
                let (values, indices) = scores.sort_last_dim(false)?;
                let ids = indices.narrow(candle_core::D::Minus1, 0, cand_k)?;
                rows.push(if margin > 0.0 {
                    let cut = values.narrow(candle_core::D::Minus1, topk - 1, 1)?;
                    let keep =
                        values.narrow(candle_core::D::Minus1, 0, cand_k)?.broadcast_ge(&(cut + margin as f64)?)?;
                    let gated = Tensor::full(self.layout.num_experts as u32, ids.shape(), ids.device())?;
                    keep.where_cond(&ids, &gated)?
                } else {
                    ids
                });
            }
            if rows.is_empty() {
                return Ok(Vec::new());
            }
            let n = rows.len();
            let all = Tensor::cat(&rows, 0)?.to_dtype(candle_core::DType::U32)?.flatten_all()?.to_vec1::<u32>()?;
            Ok(ranked(all, n))
        };
        predict().unwrap_or_default()
    }
}

impl MoeLookahead for OffloadDispatcher {
    /// Large, so wanted only by a host-plane consumer and until held; a sealed pool has no resolver to read them.
    fn wants_router_tables(&self) -> bool {
        let oracle = self.pool.oracle().map(|o| o.as_ref());
        offloading_live()
            && consumers(oracle, crate::config::params::PLANE_HOST).is_some()
            && self.router_tables.get().is_none()
    }

    /// Per layer `(ffn_norm, gate, selection bias)`, kept as F32; the first call wins, and inconsistent tables panic.
    fn provide_router_tables(&self, tables: Vec<(Tensor, Tensor, Option<Tensor>)>) {
        // `wants_router_tables` is the caller's test; arriving here is a request.
        if self.router_tables.get().is_some() {
            return;
        }
        let build = || -> Result<RouterTables> {
            let (mut hidden, mut num_experts) = (0usize, 0usize);
            let mut layers = Vec::with_capacity(tables.len());
            for (norm_t, gate_t, bias_t) in &tables {
                let g = gate_t.to_dtype(DType::F32)?.contiguous()?;
                let (ne, h) = g.dims2()?;
                let n = norm_t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                if n.len() != h {
                    candle_core::bail!("ffn_norm len {} != gate hidden {h}", n.len());
                }
                if (hidden != 0 && hidden != h) || (num_experts != 0 && num_experts != ne) {
                    candle_core::bail!("inconsistent per-layer gate shapes");
                }
                (hidden, num_experts) = (h, ne);
                let bias = match bias_t {
                    Some(b) => {
                        let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                        if b.len() != ne {
                            candle_core::bail!("selection bias len {} != experts {ne}", b.len());
                        }
                        b
                    }
                    None => Vec::new(),
                };
                let gate = g.flatten_all()?.to_vec1::<f32>()?;
                layers.push(LayerTable { norm: n, gate, bias });
            }
            Ok(RouterTables { hidden, num_experts, layers })
        };
        match build() {
            Ok(t) => {
                let _ = self.router_tables.set(t);
            }
            Err(e) => panic!("lookahead router tables inconsistent ({e})"),
        }
    }

    /// Tagged with the layer's upcoming wait value and committed before its attention, so staging leads demand.
    fn offer_layer_residual(&self, moe_ordinal: usize, trunk: &Tensor, residual: &dyn Fn(&Tensor) -> Result<Tensor>) {
        // Engine lane and decode only: the drafter would blit stale-gen garbage, and a prefill's residual is not read.
        if current_lane() != Lane::Engine || trunk.dims()[..2] != [1, 1] {
            return;
        }
        let Some(ep) = self.ep.state.get() else {
            return;
        };
        if ep.layout.late_tap_words == 0 {
            return;
        }
        let span = ep.layout.span(moe_ordinal, Block::Tap(Tap::Late));
        let encode = || -> Result<()> {
            let residual = residual(trunk)?;
            let (b, s, hid) = residual.dims3()?;
            if b != 1 || s != 1 || hid != span.words {
                return Ok(()); // shape drift
            }
            let src = residual.to_dtype(DType::F32)?.contiguous()?;
            let m = Metal::of(&src, "late-tap residual")?;
            let labels = ["", "oscar_event_lane_late", "oscar_event_lane_late_gen"];
            gpu_dispatch::write_block(m.dev, ep, span, ep.peek_wait_value(), labels, |enc| {
                enc.copy_from_buffer(&m.buf, m.off, ep.mailbox_buffer(), span.offset(), span.bytes());
            })?;
            // Commit now: the resolver polls on reaching the layer; `src` outlives the flush, else its buffer recycles.
            m.dev.flush_commands()?;
            drop(src);
            Ok(())
        };
        if let Err(e) = encode() {
            tracing::error!(target: "oscar::oracle",
                "event lane: late-tap blit failed at ordinal {moe_ordinal}: {e} - skipped",
            );
        }
    }
}

/// The next-layer prediction's width on the GPU, a floor raised to the routing top-k.
pub(crate) const GPU_PRED_WIDTH: usize = 8;
/// The chain's horizon: it targets steps 2..=`HOST_MAX_STEPS`.
pub(crate) const HOST_MAX_STEPS: usize = 3;
/// Candidates the chain ranks per step, a floor raised to the routing top-k.
pub(crate) const HOST_CANDIDATES: usize = 8;
/// Cold experts the chain may stage per (step, prediction) at top-8 routing.
pub(crate) const HOST_COLD_BUDGET: usize = 6;

/// Early: the router-input hidden of the layer being resolved; late: the post-FFN residual entering it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Tap {
    Early,
    Late,
}

/// Host copies of every layer's router, indexed by dispatcher ordinal; a dense layer's are empty and skipped.
pub(crate) struct RouterTables {
    pub(crate) hidden: usize,
    pub(crate) num_experts: usize,
    pub(crate) layers: Vec<LayerTable>,
}

/// One layer's router, each part empty where absent.
pub(crate) struct LayerTable {
    /// The `ffn_norm` weight, `hidden` long.
    pub(crate) norm: Vec<f32>,
    /// `num_experts × hidden`, row-major per expert.
    pub(crate) gate: Vec<f32>,
    /// Empty for the softmax family; else ranked by `sigmoid(logit) + bias`, the model's own selection.
    pub(crate) bias: Vec<f32>,
}

impl RouterTables {
    /// For the host line before the tables exist: a norm, a gate and a bias per layer.
    pub(crate) fn host_bytes(layers: usize, num_experts: usize, hidden: usize) -> usize {
        layers
            * ((hidden + num_experts * hidden + num_experts) * std::mem::size_of::<f32>()
                + 3 * std::mem::size_of::<Vec<f32>>())
    }
}

/// Reused by [`Resolver::predict_and_stage_ahead`], the resolver being one thread.
#[derive(Default)]
pub(crate) struct PredictScratch {
    pub(crate) h: Vec<f32>,
    pub(crate) z: Vec<f32>,
    pub(crate) logits: Vec<f32>,
    pub(crate) order: Vec<u16>,
    pub(crate) admitted: Vec<u16>,
}

struct AheadRequest {
    layer_idx: usize,
    expect_gen: u8,
    current_tick: u32,
    tap: Tap,
    /// Layer offsets ahead of `layer_idx` this call targets.
    steps: std::ops::RangeInclusive<usize>,
    timeout: Duration,
    stage_io: bool,
}

// Accelerate runs the router product, the resolver thread's largest CPU cost, on the matrix coprocessor.
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemv(
        order: i32, trans: i32, m: i32, n: i32, alpha: f32, a: *const f32, lda: i32, x: *const f32, incx: i32,
        beta: f32, y: *mut f32, incy: i32,
    );
}

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;

/// The rms scale every chain eval applies to its source hidden.
fn rms_inv(h: &[f32]) -> f32 {
    let mut sumsq = 0f32;
    for &v in h {
        sumsq += v * v;
    }
    1.0 / (sumsq / h.len() as f32 + 1e-6).sqrt()
}

/// `logits[e] = gate[e] · z`; Accelerate may sum out of order and flip a near tie, harmless as predictions only stage.
fn router_scores(gate: &[f32], z: &[f32], hid: usize, ne: usize, logits: &mut [f32]) {
    // SAFETY: `gate` is `ne * hid` f32s, `z` is `hid` and `logits` is `ne`, as the callers check against the tables.
    unsafe {
        let (m, n) = (ne as i32, hid as i32);
        cblas_sgemv(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            m,
            n,
            1.0,
            gate.as_ptr(),
            n,
            z.as_ptr(),
            1,
            0.0,
            logits.as_mut_ptr(),
            1,
        );
    }
}

/// Leaves `order[..k]` the top-`k` ids, sorted; a selection bias first turns `logits` into `sigmoid(logit) + bias`.
fn rank_top_k(logits: &mut [f32], bias: Option<&[f32]>, order: &mut Vec<u16>, k: usize) {
    let ne = logits.len();
    if let Some(bias) = bias {
        for e in 0..ne {
            logits[e] = 1.0 / (1.0 + (-logits[e]).exp()) + bias[e];
        }
    }
    order.clear();
    order.extend(0..ne as u16);
    let desc = |a: &u16, b: &u16| logits[*b as usize].total_cmp(&logits[*a as usize]);
    if k < ne {
        order.select_nth_unstable_by(k - 1, desc);
    }
    order[..k].sort_unstable_by(desc);
}

/// In best-first order: the margin drops the low-confidence tail, whose staged bytes are waste; the cap bounds width.
fn filter_admitted(admitted: &mut Vec<u16>, logits: &[f32], threshold: f32, margin: f32, cap: usize) {
    if margin > 0.0 {
        admitted.retain(|&e| logits[e as usize] >= threshold + margin);
    }
    if cap > 0 {
        admitted.truncate(cap);
    }
}

impl Resolver {
    /// Scaled with the routing width, not bank capacity: starving the chain costs more refetches than capacity saves.
    fn host_cold_budget(&self) -> usize {
        (HOST_COLD_BUDGET * self.layout.topk.max(8)).div_ceil(8)
    }

    /// The cold experts the GPU predicted at the previous dispatch; the poll is short, as the prediction trails by µs.
    pub(crate) fn stage_gpu_pred(&self, target_layer: usize, expect_gen: u8, current_tick: u32) {
        // The drafter owns the tokens not yet started, this plane the near one.
        let Some(stage_io) = consumers(self.oracle.as_deref(), crate::config::params::PLANE_GPU) else { return };
        let span = self.layout.span(target_layer, Block::Pred);
        let Some(base) = poll(&self.mailbox, span, expect_gen, Duration::from_micros(200), None, |_| true) else {
            return;
        };
        let unique = self.read_pred_candidates(base);
        refill(&self.mailbox, span, true);
        if unique.is_empty() {
            return;
        }
        // Due the next layer-tick, and claimed whatever staging decides.
        let deadline = current_tick.saturating_add(1);
        self.note_near_claims(target_layer, &unique, deadline);
        if stage_io {
            let keys = unique.iter().map(|&e| ((target_layer as u16, e), deadline));
            self.pool.offer(usize::MAX, keys);
        }
    }

    /// The ids at or above the predicted rank-`topk` logit, in descending-logit order: a prefix is the confident head.
    fn read_pred_candidates(&self, base: *mut u32) -> Vec<u16> {
        let b = self.layout.gpu_pred_width;
        let cut = self.layout.topk.min(b).saturating_sub(1);
        // SAFETY (every read here): the pred block `poll` handed over.
        let threshold = f32::from_bits(unsafe { base.add(b + cut).read_volatile() });
        let mut admitted: Vec<u16> = Vec::with_capacity(b);
        for i in 0..b {
            let id = unsafe { base.add(i).read_volatile() };
            let logit = f32::from_bits(unsafe { base.add(b + i).read_volatile() });
            // NaN-safe: a non-finite logit fails the comparison and is gated.
            if (id as usize) < self.layout.num_experts && logit >= threshold && !admitted.contains(&(id as u16)) {
                admitted.push(id as u16);
            }
        }
        admitted
    }

    /// At prediction time, whatever staging decided: a prediction's IO and its eviction evidence are gated apart.
    fn note_near_claims(&self, target_layer: usize, experts: &[u16], deadline: u32) {
        if let Some(oracle) = &self.oracle {
            for &e in experts {
                oracle.note_near_claim((target_layer as u16, e), deadline);
            }
        }
    }

    /// The early tap stages from step 2, as the GPU predicts step 1; the late tap from step 0, this layer.
    pub(crate) fn stage_tap(&self, w: &WorkItem, tap: Tap, steps: std::ops::RangeInclusive<usize>, timeout: Duration) {
        if self.layout.words(Block::Tap(tap)) == 0
            || steps.is_empty()
            || w.layer_idx + steps.start() >= self.block_count
        {
            return;
        }
        let Some(stage_io) = consumers(self.oracle.as_deref(), crate::config::params::PLANE_HOST) else {
            return;
        };
        self.predict_and_stage_ahead(&AheadRequest {
            layer_idx: w.layer_idx,
            expect_gen: gen_byte(w.wait_value),
            current_tick: w.target_tick,
            tap,
            steps,
            timeout,
            stage_io,
        })
    }

    /// Both taps': each step's layer ranked against the source state, staging what clears the gate; nothing allocates.
    fn predict_and_stage_ahead(&self, req: &AheadRequest) {
        let (layer_idx, tap) = (req.layer_idx, req.tap);
        let Some(tables) = self.router_tables.get() else {
            return; // the port has not provisioned them (yet)
        };
        let hid = self.layout.words(Block::Tap(tap));
        if tables.hidden != hid || tables.num_experts != self.layout.num_experts {
            return; // tables do not describe this model
        }
        // The early state is post-norm, divided back through its layer's `ffn_norm`; the late residual is un-normed.
        let source_norm = match tap {
            Tap::Late => None,
            Tap::Early => match tables.layers.get(layer_idx).map(|t| &t.norm) {
                Some(w) if w.len() == hid => Some(w.as_slice()),
                _ => return,
            },
        };
        let mut scratch = self.predict_scratch.borrow_mut();
        let Some(rms_scale) = self.read_tap_state(req, hid, source_norm, &mut scratch) else {
            return;
        };

        let candidates = HOST_CANDIDATES.max(self.layout.topk).min(self.layout.num_experts);
        let cut_rank = self.layout.topk.min(candidates);
        for step in req.steps.clone() {
            let target = layer_idx + step;
            if target >= self.block_count {
                break;
            }
            self.stage_one_step(req, (tables, hid, rms_scale, candidates, cut_rank), target, step, &mut scratch);
        }
    }

    fn stage_one_step(
        &self, req: &AheadRequest, pass: (&RouterTables, usize, f32, usize, usize), target: usize, step: usize,
        scratch: &mut PredictScratch,
    ) {
        // The late tap's step 0 stands in for the GPU prediction of the layer about to be demanded: staged uncapped.
        let imminent = req.tap == Tap::Late && step == 0;
        if !self.admit_for_layer(pass, target, scratch) || scratch.admitted.is_empty() {
            return;
        }
        crate::engine::eval::hooks().note_lookahead(target, step, &scratch.order[..pass.4], &scratch.admitted);
        let deadline = req.current_tick.saturating_add(step as u32);
        self.note_near_claims(target, &scratch.admitted, deadline);
        if req.stage_io {
            let budget = if imminent { usize::MAX } else { self.host_cold_budget() };
            let keys = scratch.admitted.iter().map(|&e| ((target as u16, e), deadline));
            self.pool.offer(budget, keys);
        }
    }

    /// The tap's words, `source_norm` divided out, and their rms scale; `None` if the GPU did not publish in time.
    fn read_tap_state(
        &self, req: &AheadRequest, hid: usize, source_norm: Option<&[f32]>, scratch: &mut PredictScratch,
    ) -> Option<f32> {
        let span = self.layout.span(req.layer_idx, Block::Tap(req.tap));
        let base = poll(&self.mailbox, span, req.expect_gen, req.timeout, None, |_| true)?;
        let retire = req.tap == Tap::Early || *req.steps.start() > 0;
        let h = &mut scratch.h;
        h.resize(hid, 0.0);
        // SAFETY: the tap block `poll` handed over, stable until this layer's next blit.
        unsafe { std::ptr::copy_nonoverlapping(base as *const f32, h.as_mut_ptr(), hid) };
        if retire {
            refill(&self.mailbox, span, false);
        }
        if let Some(norm) = source_norm {
            for (v, w) in h.iter_mut().zip(norm) {
                *v /= w;
            }
        }
        Some(rms_inv(h))
    }

    /// Into `scratch.admitted`, `target`'s experts at or above the rank-`topk` score; false with no usable table.
    fn admit_for_layer(
        &self, (tables, hid, rms_scale, candidates, cut_rank): (&RouterTables, usize, f32, usize, usize),
        target: usize, scratch: &mut PredictScratch,
    ) -> bool {
        let params = &crate::config::params().ep;
        let (margin, cap) = (params.lookahead_stage_margin, params.lookahead_stage_max_experts);
        let Some(LayerTable { norm, gate, bias }) = tables.layers.get(target) else {
            return false;
        };
        if gate.len() != self.layout.num_experts * hid || norm.len() != hid {
            return false;
        }
        let PredictScratch { h, z, logits, order, admitted } = scratch;
        z.resize(hid, 0.0);
        logits.resize(self.layout.num_experts, 0.0);
        for i in 0..hid {
            z[i] = h[i] * rms_scale * norm[i];
        }
        router_scores(gate, z, hid, self.layout.num_experts, &mut logits[..self.layout.num_experts]);
        let bias = Some(bias.as_slice()).filter(|b| b.len() == self.layout.num_experts);
        rank_top_k(&mut logits[..self.layout.num_experts], bias, order, candidates);
        let threshold = logits[order[cut_rank - 1] as usize];
        admitted.clear();
        admitted.extend(order.iter().take(candidates).filter(|&&e| logits[e as usize] >= threshold));
        filter_admitted(admitted, logits, threshold, margin, cap);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The routing top-`k` the obvious way (serial dot, full sort), sharing no code with the implementation.
    fn reference_top_k(t: &RouterTables, h: &[f32], l: usize, k: usize) -> (Vec<u16>, Vec<f32>) {
        let (hid, ne) = (t.hidden, t.num_experts);
        let (gate, w) = (&t.layers[l].gate, &t.layers[l].norm);
        let rms_scale = 1.0 / (h.iter().map(|v| v * v).sum::<f32>() / hid as f32 + 1e-6).sqrt();
        let mut lg: Vec<f32> =
            (0..ne).map(|e| (0..hid).map(|i| gate[e * hid + i] * (h[i] * rms_scale * w[i])).sum()).collect();
        if t.layers[l].bias.len() == ne {
            lg.iter_mut().zip(&t.layers[l].bias).for_each(|(s, b)| *s = 1.0 / (1.0 + (-*s).exp()) + b);
        }
        let mut ids: Vec<u16> = (0..ne as u16).collect();
        ids.sort_by(|&a, &b| lg[b as usize].total_cmp(&lg[a as usize]));
        ids.truncate(k);
        (ids, lg)
    }

    /// A biased MoE layer and an unbiased one, every score well clear of its neighbours: Accelerate sums out of source
    /// order, so ids compare only where no two scores sit within rounding of each other. The stage gate filters by
    /// confidence, then width.
    #[test]
    fn scores_and_ranks_match_reference_and_the_stage_gate_filters_by_confidence_then_width() {
        let (hid, ne, k) = (16, 12, 5);
        let gate = |seed: f32| -> Vec<f32> {
            let at = |e: f32, i: f32| ((e * 1.7 + i * 0.31 + seed).sin() + e * 0.11) * 0.5;
            (0..ne * hid).map(|j| at((j / hid) as f32, (j % hid) as f32)).collect()
        };
        let norm = |seed: f32| -> Vec<f32> { (0..hid).map(|i| 1.0 + 0.1 * ((i as f32) * 0.7 + seed).cos()).collect() };
        let biased = (0..ne).map(|e| e as f32 * 0.03).collect();
        let layers = vec![
            LayerTable { norm: norm(0.0), gate: gate(0.0), bias: biased },
            LayerTable { norm: norm(2.0), gate: gate(3.0), bias: Vec::new() },
        ];
        let t = RouterTables { hidden: hid, num_experts: ne, layers };
        let mut order = Vec::with_capacity(ne);
        for s in 0..4 {
            let state = |i: usize| (s as f32 * 0.9 + i as f32 * 0.13).sin() * (1.0 + s as f32 * 0.25);
            let h: Vec<f32> = (0..hid).map(state).collect();
            let rms_scale = rms_inv(&h);
            for l in 0..2 {
                let (gate, norm) = (&t.layers[l].gate, &t.layers[l].norm);
                let scaled: Vec<f32> = (0..hid).map(|i| h[i] * rms_scale * norm[i]).collect();
                let mut logits = vec![0.0f32; ne];
                router_scores(gate, &scaled, hid, ne, &mut logits);
                let bias = Some(t.layers[l].bias.as_slice()).filter(|b| b.len() == ne);
                rank_top_k(&mut logits, bias, &mut order, k);
                let (want, mut lg) = reference_top_k(&t, &h, l, k);
                lg.sort_by(f32::total_cmp);
                assert!(lg.windows(2).all(|w| w[1] - w[0] > 1e-4), "fixture scores too close to rank: {lg:?}");
                assert_eq!(order[..k], want[..], "state {s}, layer {l}");
            }
        }
        // The full width: no selection first, every id sorted.
        rank_top_k(&mut [0.5, 0.1, 0.9, 0.3], None, &mut order, 4);
        assert_eq!(order, vec![2u16, 0, 3, 1]);
        // Over the boundary 0.50 by the margin 0.1: expert 0 alone, expert 2's 0.01 not.
        let rows: [(f32, usize, &[u16], &str); 3] = [
            (0.0, 0, &[0, 1, 2], "neither control armed is a no-op"),
            (0.1, 0, &[0], "margin must drop the low-confidence tail"),
            (0.0, 2, &[0, 1], "cap must keep the best-ranked prefix"),
        ];
        for (margin, cap, want, why) in rows {
            let mut admitted = vec![0u16, 1, 2];
            filter_admitted(&mut admitted, &[0.90, 0.55, 0.51, 0.30], 0.50, margin, cap);
            assert_eq!(admitted, want, "{why}");
        }
    }
}
