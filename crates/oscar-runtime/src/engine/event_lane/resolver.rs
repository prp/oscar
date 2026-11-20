//! The resolver thread: on any failure the rows stay null and the event is signalled anyway, so the GPU never wedges.

use super::*;

use crate::engine::eval::{ResolverProbe, ResolverStage};

pub(crate) struct Resolver {
    pub(crate) layout: MailboxLayout,
    pub(crate) mailbox: KernelBuffer,
    pub(crate) block_count: usize,
    pub(super) routing_event: SharedEventHandle,
    pub(super) completion_event: SharedEventHandle,
    pub(crate) pool: Arc<ExpertPool>,
    /// The kernel's address rows, `[gate, up, down]`.
    pub(super) rows: [RowBank; 3],
    pub(crate) oracle: Option<Arc<dyn crate::pool::Oracle>>,
    /// Provided at the model's first decode, possibly after this thread starts.
    pub(crate) router_tables: Arc<OnceLock<RouterTables>>,
    pub(super) pins: Arc<super::PinLedger>,
    pub(super) failed_layer: Arc<AtomicUsize>,
    pub(super) probe: Option<Arc<dyn ResolverProbe>>,
    pub(super) theta: Option<f32>,
    pub(crate) predict_scratch: std::cell::RefCell<PredictScratch>,
}

// SAFETY: KernelBuffer, RowBank and SharedEventHandle are retained Metal objects, thread-safe as each type documents.
unsafe impl Send for Resolver {}

pub(super) fn resolver_main(ctx: Resolver, rx: Receiver<WorkItem>) {
    crate::engine::set_interactive_qos();
    while let Ok(w) = rx.recv() {
        ctx.open_dispatch();

        // This layer's own prediction, issued before the routing it beats.
        ctx.stage_tap(&w, Tap::Late, 0..=0, Duration::from_micros(2_000));
        ctx.close(ResolverStage::Late);

        match ctx.obtain_routing(&w) {
            Some(ids) => {
                ctx.resolve_dispatch(&w, &ids);
                // Refill before signalling: the GPU cannot re-blit until the event releases.
                ctx.refill_sentinel(w.layer_idx);
                ctx.signal(&w);
                ctx.close(ResolverStage::Signal);
                // Deeper staging after the signal: before the resolve it would delay this layer's demand fetches.
                let deeper = Duration::from_micros(200);
                ctx.stage_tap(&w, Tap::Early, 2..=HOST_MAX_STEPS, deeper);
                ctx.stage_tap(&w, Tap::Late, 1..=HOST_MAX_STEPS.saturating_sub(1).max(2), deeper);
            }
            None => {
                // No refill: it would race blits in flight.
                ctx.skip_dispatch(&w);
                ctx.signal(&w);
                ctx.close(ResolverStage::Signal);
            }
        }
        ctx.close(ResolverStage::Post);
    }
}

impl Resolver {
    fn open_dispatch(&self) {
        if let Some(p) = &self.probe {
            p.open();
        }
    }

    fn close(&self, stage: ResolverStage) {
        if let Some(p) = &self.probe {
            p.close(stage);
        }
    }

    /// The one place the two id sources differ: mailbox ids are waited for and may not arrive.
    fn obtain_routing(&self, w: &WorkItem) -> Option<Vec<u32>> {
        let probe = self.probe.as_deref();
        if let Some(p) = probe {
            p.poll_begin();
        }
        let (deficit, polled) = match w.ids {
            RoutingIds::Mailbox => (
                probe.map(|_| (w.chain_gen - 1).saturating_sub(self.completion_event.0.signaledValue())),
                self.poll_mailbox(w.layer_idx, w.wait_value, w.poll_timeout()),
            ),
            RoutingIds::Verified(ref ids) => (probe.map(|_| 0), Some(ids.clone())),
        };
        if let (Some(p), Some(d)) = (probe, deficit) {
            p.poll_end(d);
        }
        self.close(ResolverStage::Visible);
        polled
    }

    fn resolve_dispatch(&self, w: &WorkItem, ids: &[u32]) {
        let Some(unique) = self.dedup_routed(w, ids) else {
            return; // an invalid id skip-rowed the layer
        };
        // Bookkeeping first: a token's first layer rolls the eviction horizon, so victims are picked on fresh scores.
        self.observe_routing(w, &unique);
        // Stage L+1 before resolving L, so its IO overlaps L's demand IO and FFN and L+1's attention.
        if self.layout.gpu_pred_width > 0 && w.layer_idx + 1 < self.block_count {
            self.stage_gpu_pred(w.layer_idx + 1, gen_byte(w.wait_value), w.target_tick);
        }
        self.close(ResolverStage::Book);
        self.process_layer(w, &unique);
        self.close(ResolverStage::Resolve);
    }

    fn skip_dispatch(&self, w: &WorkItem) {
        tracing::error!(target: "oscar::oracle",
            "event lane: mailbox poll timed out at layer {} (wait value {}) - its rows skipped",
            w.layer_idx, w.wait_value,
        );
        self.write_skip_rows(w.layer_idx);
    }

    fn signal(&self, w: &WorkItem) {
        self.routing_event.0.setSignaledValue(w.wait_value);
    }

    fn poll_mailbox(&self, layer_idx: usize, wait_value: u64, dispatch_timeout: Duration) -> Option<Vec<u32>> {
        // The first four tokens since the load, a wait value per layer dispatch.
        let timeout = if wait_value <= (4 * self.block_count) as u64 {
            dispatch_timeout.max(POLL_TIMEOUT_AFTER_LOAD)
        } else {
            dispatch_timeout
        };
        poll_ids(&self.mailbox, &self.layout, layer_idx, wait_value, timeout)
    }

    /// So a stale routing is never taken for the next token's; prediction blocks are refilled by their consumers.
    fn refill_sentinel(&self, layer_idx: usize) {
        refill(&self.mailbox, self.layout.span(layer_idx, Block::Ids), true);
    }

    /// The consumed routing to the eval's checks and the eviction oracle.
    fn observe_routing(&self, w: &WorkItem, unique: &[u16]) {
        let hooks = crate::engine::eval::hooks();
        hooks.check_routing(w.layer_idx, w.call_count, 1, unique);
        hooks.note_actual_routing(self.block_count, w.layer_idx, unique);
        let Some(oracle) = &self.oracle else {
            return;
        };
        oracle.observe_routing(w.layer_idx, unique);
    }

    fn process_layer(&self, w: &WorkItem, unique: &[u16]) {
        let owned = self.pool.banks_for(w.layer_idx).map(Arc::clone);
        let banks = owned.each_ref();

        // θ before any pin or fetch: `exec[i]` runs for `unique[i]`, the rows keyed by the original ids.
        let exec = self.theta_substitute_layer(w, unique, banks).unwrap_or_else(|| unique.to_vec());
        let keys: Vec<LayerExpertKey> = exec.iter().map(|&e| (w.layer_idx as u16, e)).collect();

        match self.pin_or_fetch(w, banks, &keys) {
            Ok(out) => self.publish_layer(w, unique, out, owned),
            Err(e) => {
                tracing::error!(target: "oscar::oracle",
                    "event lane: resolve failed at layer {}: {e} - skip rows",
                    w.layer_idx,
                );
                self.write_skip_rows(w.layer_idx);
            }
        }
    }

    /// The routed ids deduplicated in order; `None` (the layer skip-rowed) when one is out of range.
    fn dedup_routed(&self, w: &WorkItem, ids: &[u32]) -> Option<Vec<u16>> {
        let mut unique: Vec<u16> = Vec::with_capacity(self.layout.topk);
        for &id in ids {
            if (id as usize) >= self.layout.num_experts {
                tracing::error!(target: "oscar::oracle",
                    "event lane: invalid expert id {id} at layer {} - skip rows",
                    w.layer_idx,
                );
                self.write_skip_rows(w.layer_idx);
                return None;
            }
            let e = id as u16;
            if !unique.contains(&e) {
                unique.push(e);
            }
        }
        Some(unique)
    }

    /// Pin the routed set, fetching what is cold; lock-free when all is resident.
    fn pin_or_fetch(
        &self, w: &WorkItem, banks: [&Arc<Bank>; 3], keys: &[LayerExpertKey],
    ) -> std::result::Result<ResolveOut, String> {
        let Some(slots) = crate::pool::pin_all(banks, keys, w.target_tick) else {
            for b in banks {
                b.set_routed(keys.iter().copied());
            }
            let timed = self.probe.as_deref().map(|p| move |fetch_ns, pin_ns| p.resolve_block(fetch_ns, pin_ns));
            // This thread waits on the engine's behalf, not its own.
            let (order, lane) = (crate::pool::PinOrder::KeyOuter, crate::pool::Lane::Engine);
            let bound =
                (self.pool.resolve_blocking(banks, keys, w.target_tick, order, lane, timed.as_ref().map(|t| t as _)))
                    .map(|(slots, _)| slots)
                    .map_err(|(op, e)| format!("{op}: {e}"));
            for b in banks {
                b.clear_routed();
            }
            return bound;
        };
        Ok(slots)
    }

    /// Each routed expert's slot address into the kernel's rows, the pins held until its command buffer completes.
    fn publish_layer(&self, w: &WorkItem, unique: &[u16], out: ResolveOut, banks: [Arc<Bank>; 3]) {
        for ((rows, bank), slots) in self.rows.iter().zip(&banks).zip(&out) {
            write_row(
                w.layer_idx,
                self.layout.num_experts,
                rows,
                unique.iter().zip(slots).map(|(&e, &s)| (e, bank.pool().address(s))),
            );
        }
        // The chain's own buffer: the committed count here is past it, its successor none at a step's last layer.
        for (bank, pins) in banks.into_iter().zip(out) {
            self.pins.push(w.chain_gen, bank, pins);
        }
    }

    /// The new weights written back to the mailbox; `None` for an exact layer, whose write-back round-trips bitwise.
    fn theta_substitute_layer(&self, w: &WorkItem, unique: &[u16], banks: [&Arc<Bank>; 3]) -> Option<Vec<u16>> {
        let theta = self.theta?;
        // Verified ids came read back, substituted there, with no θ block.
        if unique.len() != self.layout.topk || matches!(w.ids, RoutingIds::Verified(_)) {
            return None;
        }
        // θ is blitted right after the routing; any lag skips θ for this layer, and the resolve is exact.
        let span = self.layout.span(w.layer_idx, Block::Theta);
        let Some(base) = poll(&self.mailbox, span, gen_byte(w.wait_value), Duration::from_micros(500), None, |_| true)
        else {
            crate::engine::approx::GEN_SKIPS.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        // SAFETY: the θ block `poll` handed over; the GPU cannot re-blit it until this dispatch's event is signalled.
        let (scores_ptr, ne) = (base as *mut f32, self.layout.num_experts);
        let weights_ptr = unsafe { scores_ptr.add(ne) };
        let scores: Vec<f32> = (0..ne).map(|i| unsafe { scores_ptr.add(i).read_volatile() }).collect();
        let weights: Vec<f32> = (0..self.layout.topk).map(|i| unsafe { weights_ptr.add(i).read_volatile() }).collect();
        let layer = w.layer_idx as u16;
        let is_res = |e: u16| banks.iter().all(|b| b.is_resident((layer, e)));
        let sub = crate::engine::approx::theta_substitute(theta, unique, &weights, &scores, &is_res)?;
        for (i, &v) in sub.weights.iter().enumerate() {
            // SAFETY: as above; aligned 4-byte stores, no GPU writer until the event.
            unsafe { weights_ptr.add(i).write_volatile(v) };
        }
        std::sync::atomic::fence(Ordering::Release);
        Some(sub.exec)
    }

    /// Every row null and the failure recorded, so the sequence aborts; always before the layer's event is signalled.
    fn write_skip_rows(&self, layer_idx: usize) {
        super::SKIP_ROW_LAYERS.fetch_add(1, Ordering::Relaxed);
        self.failed_layer.fetch_min(layer_idx, Ordering::AcqRel);
        write_skip_rows_into(self.rows.each_ref(), layer_idx, self.layout.num_experts);
    }
}

/// The pinned slots per bank, each both a row entry (as its address) and a pin until the kernel completes.
type ResolveOut = [Vec<usize>; 3];
