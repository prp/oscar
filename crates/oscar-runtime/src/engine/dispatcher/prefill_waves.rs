//! A prefill call cut into token batches, each into waves of experts that fit resident together.

use super::*;

use std::sync::mpsc::TrySendError;
use std::time::{Duration, Instant};

use crate::engine::eval::{EvalHooks, PrefillEvent, WaveReport};
use crate::pool::policy::{prefill_batch, PREFILL_BATCH_TOKENS, PREFILL_CB_WAVES};
use crate::pool::{commit_pins, PinLedger};

/// A bound on prediction cost, a gate evaluation and a readback a layer; fixed, not a config parameter.
const MAX_AHEAD_LAYERS: usize = 12;

/// One slice of tokens, and the span of waves covering its routed experts.
struct TokenBatch {
    token_lo: usize,
    token_hi: usize,
    wave_lo: usize,
    wave_hi: usize,
}

/// A pure function of the routed ids and the rule's widths, so the demand clock is independent of execution order.
struct WaveSchedule {
    /// Every wave's expert ids, concatenated; each wave's `[lo, hi)` and tick.
    flat_ids: Vec<u16>,
    wave_bounds: Vec<(usize, usize)>,
    wave_ticks: Vec<u32>,
    batches: Vec<TokenBatch>,
}

impl WaveSchedule {
    /// Waves of `wave_width` consecutive expert ids per slice, one tick each; a slice with no wave emits nothing.
    fn plan(ids: &[u32], shape: ScheduleShape, target_tick: u32) -> Self {
        let ScheduleShape { num_tokens, topk, tokens_per_batch, wave_width, num_experts, min_capacity } = shape;
        let mut s = Self { flat_ids: Vec::new(), wave_bounds: Vec::new(), wave_ticks: Vec::new(), batches: Vec::new() };
        for token_lo in (0..num_tokens).step_by(tokens_per_batch) {
            let token_hi = (token_lo + tokens_per_batch).min(num_tokens);
            let mut unique = dedup_u32_to_u16(&ids[token_lo * topk..token_hi * topk], num_experts);
            debug_assert!(
                wave_width < num_experts || unique.len() <= min_capacity,
                "one wave of {} uniques against {min_capacity} slots cannot be resident",
                unique.len(),
            );
            let (wave_lo, base) = (s.wave_bounds.len(), target_tick.saturating_add(s.wave_ticks.len() as u32));
            for (i, wave) in waves(&mut unique, wave_width).enumerate() {
                s.wave_bounds.push((s.flat_ids.len(), s.flat_ids.len() + wave.len()));
                s.flat_ids.extend_from_slice(wave);
                s.wave_ticks.push(base.saturating_add(i as u32));
            }
            if s.wave_bounds.len() > wave_lo {
                s.batches.push(TokenBatch { token_lo, token_hi, wave_lo, wave_hi: s.wave_bounds.len() });
            }
        }
        s
    }
}

#[derive(Clone, Copy)]
struct ScheduleShape {
    num_tokens: usize,
    topk: usize,
    tokens_per_batch: usize,
    wave_width: usize,
    num_experts: usize,
    min_capacity: usize,
}

/// Where the share of waves finding every expert resident saturates; [`run_ahead_depth`] clamps it to the pool.
const PREFILL_AHEAD_WAVES: usize = 16;

/// Reserved run-ahead slots and the batch's pins must leave one slot per bank to eviction, or the pass wedges.
fn run_ahead_depth(want: usize, allocatable: usize, widest_wave: usize, cb_waves: usize) -> usize {
    let room = allocatable.saturating_sub(1).saturating_sub(cb_waves.saturating_mul(widest_wave));
    want.min(room / widest_wave.max(1))
}

/// `(reserving, offering)` reach while `wave_idx` runs; offers take no slot, so they reach the stream's end.
fn stream_limits(wave_idx: usize, depth: usize, reserving_len: usize, stream_len: usize) -> (usize, usize) {
    let reach = wave_idx.saturating_add(depth).saturating_add(1);
    (reserving_len.min(reach), stream_len)
}

/// This layer's waves, then the next layers' slabs; the reserving jobs come first, the rest only offer.
#[derive(Default)]
struct AheadStream {
    jobs: Vec<Option<PrefetchJob>>,
    this_layer_waves: usize,
    reserving_len: usize,
    /// The reserving and the offering cursors; neither waits on the other.
    next: [usize; 2],
}

impl AheadStream {
    /// A full channel stops both cursors, the wave's own resolve doing the work; out, when the last job went.
    fn push(
        &mut self, prefetch: &PrefetchThread, wave_idx: usize, depth: usize, eval: &dyn EvalHooks,
    ) -> Option<Instant> {
        self.next[0] = self.next[0].max(wave_idx + 1);
        let (reserve, offer) = stream_limits(wave_idx, depth, self.reserving_len, self.jobs.len());
        let mut issued = None;
        for (c, limit) in [(0, reserve), (1, offer)] {
            while self.next[c] < limit {
                let i = self.next[c];
                let job = self.jobs[i].take().expect("each job is pushed once");
                let (keys, speculative) = (job.keys.len() as u64, matches!(job.due, Due::Ahead(_)));
                if let Err(TrySendError::Full(job) | TrySendError::Disconnected(job)) = prefetch.tx().try_send(job) {
                    self.jobs[i] = Some(job);
                    eval.prefill(PrefillEvent::StreamBlocked);
                    return issued;
                }
                eval.prefill(PrefillEvent::Pushed { keys, speculative, crossing: i >= self.this_layer_waves });
                issued = Some(Instant::now());
                self.next[c] += 1;
            }
        }
        issued
    }
}

impl OffloadDispatcher {
    /// A sweeping forward streams the next layers' id ranges, a narrow one their gates' capped prediction as a lead.
    fn ahead_stream(&self, call: &Call<'_>, r: &Routing, s: &WaveSchedule, wave_width: usize) -> AheadStream {
        let (pool, ne) = (call.pool, call.shape.num_experts);
        let job = |layer: usize, ids: &[u16], due| PrefetchJob {
            keys: ids.iter().map(|&e| (layer as u16, e)).collect(),
            due,
            layer,
        };
        let mut jobs: Vec<Option<PrefetchJob>> = (0..s.wave_ticks.len())
            .map(|w| Some(job(call.idx, &s.flat_ids[s.wave_bounds[w].0..s.wave_bounds[w].1], Due::At(s.wave_ticks[w]))))
            .collect();
        let this_layer_waves = jobs.len();
        let sweeping = r.unique.len().saturating_mul(2) >= ne;
        // The stream's only speculative part rides the lookahead selector and its volume cap.
        let ep = &crate::config::params().ep;
        let keep = match (ep.lookahead, ep.lookahead_stage_max_experts) {
            (crate::config::params::Lookahead::None, _) => 0,
            (_, 0) => Self::auto_head(pool, r.unique.len()),
            (_, cap) => Self::auto_head(pool, r.unique.len()).min(cap),
        };
        // The span is the run-ahead this layer cannot spend, for an idle drive; more would name resident experts.
        let slabs = ne.div_ceil(wave_width);
        let span = if sweeping { 1 } else { PREFILL_AHEAD_WAVES.div_ceil(slabs.max(1)).clamp(1, MAX_AHEAD_LAYERS) };
        let predicted = match (sweeping, keep) {
            (true, _) | (false, 0) => Vec::new(),
            (false, keep) => {
                let p =
                    self.predict_layers_ahead(call.idx + 1, span, &call.xs, r.host.len() / call.tokens.max(1), keep);
                if p.is_empty() {
                    self.eval.prefill(PrefillEvent::Unpredicted);
                }
                p
            }
        };
        // Due from this layer's last tick: a staged read is dropped unread once its deadline passes.
        let base = s.wave_ticks.iter().copied().max().unwrap_or(call.tick);
        for step in 0..if sweeping || keep > 0 { span } else { 0 } {
            let next = call.idx + 1 + step;
            // The gate prediction alone; the previous phase's demand would assume one request predicts the next.
            let mut ids: Vec<u16> =
                if sweeping { (0..ne as u16).collect() } else { predicted.get(step).cloned().unwrap_or_default() };
            if next >= pool.layer_count() || ids.is_empty() {
                break;
            }
            for wave in waves(&mut ids, wave_width) {
                // Ticks to the slab plus the span as slack: slack costs only order, a passed deadline the read.
                let w = wave[0] as usize / wave_width;
                let lead = ((step + 1 + span) * this_layer_waves + w).max(1) as u32;
                let due = if sweeping { Due::At(base.saturating_add(1 + w as u32)) } else { Due::Ahead(lead) };
                jobs.push(Some(job(next, wave, due)));
            }
        }
        // A sweeping forward's whole stream reserves, so the depth bounds it: unbounded, unlanded reads fill the banks.
        let reserving_len = if sweeping { jobs.len() } else { this_layer_waves };
        AheadStream { jobs, this_layer_waves, reserving_len, next: [0, reserving_len] }
    }

    /// A call cut by [`WaveSchedule`], its waves streamed ahead and walked in order; the caller sets the active window.
    pub(super) fn forward_expert_waves(&self, call: Call<'_>, r: Routing) -> Result<Finalized> {
        let (pool, layer_idx, banks, ne) = (call.pool, call.idx, call.banks, call.shape.num_experts);
        // From the routed set, not `num_experts_per_tok`.
        let topk = r.host.len() / call.tokens.max(1);
        // A census runs this loop for its buffers alone, every side effect suppressed.
        let live = !oscar_offload::census::armed();
        COUNTERS.decode_armed.store(false, Ordering::Release);
        let pools = BankPools::new(&call.device, banks);
        // A live pool partitions against the smallest bank. A census pins nothing, so one wave could take the layer; it
        // is cut into the fewest waves filling the pipeline, so the record holds what a live batch keeps alive at once.
        let capacity = if live { pool.min_bank_allocatable(false) } else { 0 };
        let tokens_per_batch = PREFILL_BATCH_TOKENS as usize;
        let wave_width = match prefill_batch(ne as u32, capacity) as usize {
            0 if live => ne.max(1),
            0 => ne.div_ceil(PREFILL_CB_WAVES + 2).max(1),
            k => k,
        };
        let miss = self.miss_slot(&call)?;
        let t_schedule = Instant::now();
        let min_capacity = pools.min_capacity;
        let shape = ScheduleShape {
            num_tokens: call.tokens,
            topk,
            tokens_per_batch,
            wave_width,
            num_experts: ne,
            min_capacity,
        };
        let schedule = WaveSchedule::plan(&r.host, shape, call.tick);
        if live {
            let ns = t_schedule.elapsed().as_nanos() as u64;
            self.eval.prefill(PrefillEvent::Schedule { ns, tokens: call.tokens as u64 });
        }
        // A bank's claim is `(cb_waves + depth) × widest` slots: a buffer's waves stay pinned until it commits.
        let cb_waves = PREFILL_CB_WAVES;
        let widest = schedule.wave_bounds.iter().map(|&(lo, hi)| hi - lo).max().unwrap_or(1).max(1);
        let allocatable = pool.min_bank_allocatable(true) as usize;
        let depth = run_ahead_depth(PREFILL_AHEAD_WAVES, allocatable, widest, cb_waves);
        if live && !self.logged_run_ahead.swap(true, Ordering::Relaxed) && crate::config::params().misc.diag {
            tracing::info!(target: "oscar::experts",
                "prefill run-ahead pool={allocatable} slots/bank widest_wave={widest} \
                 cb_waves={cb_waves} depth={depth} (live claim {} of {allocatable} slots)",
                (cb_waves + depth) * widest,
            );
        }
        let mut stream =
            if live { self.ahead_stream(&call, &r, &schedule, wave_width) } else { AheadStream::default() };
        let pins = PinLedger::default();
        /// A wave takes tens of ms, so a ring slot unreleased after this is a stalled GPU.
        const WAVE_RING_REUSE_TIMEOUT: Duration = Duration::from_secs(5);
        // Ring slots claimed since the last commit, reusable at its generation.
        let mut pending_ring: Vec<usize> = Vec::new();
        let mut outputs: Vec<Tensor> = Vec::with_capacity(schedule.batches.len());
        for batch in &schedule.batches {
            let len = batch.token_hi - batch.token_lo;
            let (xs, weights) = (call.xs.narrow(0, batch.token_lo, len)?, r.weights.narrow(0, batch.token_lo, len)?);
            let ids = r.ids.narrow(0, batch.token_lo, len)?;
            let mut down: Option<Tensor> = None;
            for wave_idx in batch.wave_lo..batch.wave_hi {
                let t_wave = Instant::now();
                let (lo, hi) = schedule.wave_bounds[wave_idx];
                let (wave_ids, wave_tick) = (&schedule.flat_ids[lo..hi], schedule.wave_ticks[wave_idx]);
                let t_stream = Instant::now();
                let issued = stream.push(&self.prefetch, wave_idx, depth, self.eval.as_ref());
                let stream_ns = t_stream.elapsed().as_nanos() as u64;
                if live {
                    pool.tick.swap(wave_tick, Ordering::AcqRel);
                }
                let ring_slot = self.wave_pool.claim_slot(&pools.metal_device, WAVE_RING_REUSE_TIMEOUT)?;
                let keys: Vec<LayerExpertKey> = wave_ids.iter().map(|&e| (layer_idx as u16, e)).collect();
                let t_wait = Instant::now();
                let ([g, u, d], residency) = if live {
                    let order = crate::pool::PinOrder::BankOuter;
                    let resolved = (pool.resolve_blocking(banks, &keys, wave_tick, order, Lane::Engine, None))
                        .map_err(|(op, e)| candle_core::Error::Msg(format!("HotPoolBank::{op}: {e}")))?;
                    for (bank, s) in banks.into_iter().zip(&resolved.0) {
                        pins.push(0, Arc::clone(bank), s.clone());
                    }
                    resolved
                } else {
                    crate::pool::resolve_census(&keys)
                };
                let (wait_ns, t_enc) = (t_wait.elapsed().as_nanos() as u64, Instant::now());
                let fetch_window_ns = issued.map(|at| at.elapsed().as_nanos() as u64);
                // Down outputs gather unsummed, summed once after the batch's last wave: no bit depends on the waves.
                let (pinned, owners) = ([&g[..], &u[..], &d[..]], banks.map(|b| b.pool()));
                let (rows, experts) =
                    self.wave_pool.write_rows(ring_slot, &pools.metal_device, wave_ids, owners, pinned)?;
                let rows = [&rows[0], &rows[1], &rows[2]];
                route_rows(
                    call.layer,
                    &xs,
                    &pools,
                    rows,
                    Some(&experts),
                    &ids,
                    &weights,
                    &miss,
                    call.tokens,
                    Some(&mut down),
                )?;
                pending_ring.push(ring_slot);
                if pending_ring.len() >= cb_waves || wave_idx + 1 == batch.wave_hi {
                    if live {
                        commit_pins(&pools.metal_device, &pins)?;
                    }
                    let gen = pools.metal_device.completion_counter()?.load(Ordering::SeqCst);
                    pending_ring.drain(..).for_each(|i| self.wave_pool.publish_slot_gen(i, gen));
                }
                if live {
                    let (resident, in_flight, cold) =
                        (residency.resident as u64, residency.in_flight as u64, residency.cold as u64);
                    let (wave_ns, encode_ns) = (t_wave.elapsed().as_nanos() as u64, t_enc.elapsed().as_nanos() as u64);
                    let report = WaveReport {
                        wave_ns,
                        wait_ns,
                        encode_ns,
                        stream_ns,
                        resident,
                        in_flight,
                        cold,
                        fetch_window_ns,
                    };
                    self.eval.prefill(PrefillEvent::Wave(report));
                }
            }
            // Every routed slot was resolved by one of the batch's waves, so the sum's row resolves every expert.
            let down = down.expect("a token batch runs at least one wave");
            let resolved = Tensor::from_vec(vec![1i64; ne], ne, down.device())?;
            let resolved = pools.rows([&resolved, &resolved, &resolved])[2];
            outputs.push(oscar_kernels::moe_topk_weighted_sum(&down, &weights, resolved, &ids)?);
        }
        let routed = match <[Tensor; 1]>::try_from(outputs) {
            Ok([only]) => only,
            Err(many) => Tensor::cat(&many, 0)?,
        };
        pools.metal_device.flush_commands()?;
        gpu_dispatch::wait_encode_ahead(&pools.metal_device);
        call.layer.finalize_routed(Some(&routed), &call.xs, call.stream, call.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::{run_ahead_depth, stream_limits};

    /// The batch's pins and the run-ahead's reservations leave every pool a slot to evict, or nothing runs ahead. The
    /// reserving cursor stops `depth` waves on; offers take no slot, so they reach the stream's end.
    #[test]
    fn the_run_ahead_never_claims_the_last_slot_and_only_the_reserving_stream_is_bounded_by_the_depth() {
        for want in [1usize, 2, 4, 8, 64] {
            for pool in [1usize, 2, 7, 8, 9, 24, 66, 132, 3420, 7980, 8550] {
                for wave in [1usize, 22, 32, 43, 49, 56] {
                    for cb in [1usize, 2, 4] {
                        let depth = run_ahead_depth(want, pool, wave, cb);
                        let claim = (cb + depth) * wave;
                        assert!(claim < pool || depth == 0, "want={want} pool={pool} wave={wave} cb={cb}: {claim}");
                    }
                }
            }
        }
        // (want, pool, wave, depth), one wave a buffer: 86 = 2 × 43 leaves no slot to evict; (176 − 1 − 32) / 32 = 4.
        let rows = [(1, 86, 43, 0), (1, 87, 43, 1), (8, 7980, 43, 8), (8, 44, 43, 0), (8, 1, 43, 0), (16, 176, 32, 4)];
        for (want, pool, wave, depth) in rows {
            assert_eq!(run_ahead_depth(want, pool, wave, 1), depth, "want={want} pool={pool} wave={wave}");
        }
        for (reserving, stream) in [(56, 56), (6, 24)] {
            for depth in [0usize, 2, 4, 16] {
                for wave_idx in 0..56 {
                    let (reserve, offer) = stream_limits(wave_idx, depth, reserving, stream);
                    assert!(reserve <= wave_idx + depth + 1 && offer == stream, "depth {depth}, wave {wave_idx}");
                }
            }
        }
        // A short forward's own six waves clear at once under a depth of 16.
        assert_eq!(stream_limits(0, 16, 6, 24).0, 6);
    }
}
