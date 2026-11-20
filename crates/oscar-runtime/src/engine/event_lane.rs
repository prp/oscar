//! Exact paged decode, no routing sync: a kernel waits on an event the resolver signals once its experts are pinned.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use candle_core::{MetalDevice, Result};
use candle_metal_kernels::metal::Buffer as KernelBuffer;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLSharedEvent};

use crate::gpu::expert_rows::{write_row, ExpertRows, RowBank};
use crate::pool::{Bank, ExpertPool, LayerExpertKey, PinLedger};

pub(crate) mod mailbox;

// The protocol is lookahead-blind but for the mailbox layout and the resolver's hooks.
pub(crate) use crate::engine::dispatcher::lookahead::{
    PredictScratch, RouterTables, Tap, GPU_PRED_WIDTH, HOST_MAX_STEPS,
};
pub(crate) use mailbox::{alloc_mailbox, alloc_shared_event, poll, poll_ids, refill};
pub(crate) use mailbox::{gen_byte, gen_word, Block, MailboxLayout, SharedEventHandle, Span, POLL_TIMEOUT};
use mailbox::{POLL_TIMEOUT_AFTER_LOAD, WARMUP_TIMEOUT, WARMUP_TOKENS};

fn write_skip_rows_into(rows: [&RowBank; 3], layer_idx: usize, num_experts: usize) {
    for bank in rows {
        write_row(layer_idx, num_experts, bank, std::iter::empty());
    }
}

/// One resolve lane, two id sources: only the mailbox polls, so a host-routed dispatch cannot time out.
pub(crate) enum RoutingIds {
    /// Published by the GPU once it completes the buffer before the chain's, if not before.
    Mailbox,
    /// Read back by the forward thread under a recording: no poll.
    Verified(Vec<u32>),
}

/// One per-layer offload job, as the forward thread hands it to the resolver.
pub(crate) struct WorkItem {
    pub(crate) ids: RoutingIds,
    pub(crate) layer_idx: usize,
    pub(crate) target_tick: u32,
    pub(crate) wait_value: u64,
    pub(crate) call_count: u32,
    /// Decode tokens since the last prefill, before this one: the first few poll longer, the load's backlog ahead.
    pub(crate) decode_token: u64,
    /// The command buffer the layer's chain rides, the one open at the send: its pins' fence.
    pub(crate) chain_gen: u64,
}

impl WorkItem {
    fn poll_timeout(&self) -> Duration {
        if self.decode_token < WARMUP_TOKENS {
            WARMUP_TIMEOUT
        } else {
            POLL_TIMEOUT
        }
    }
}

/// Layers given null rows: every resolver degradation goes through `write_skip_rows`.
pub static SKIP_ROW_LAYERS: AtomicU64 = AtomicU64::new(0);

/// Read lock-free: the memory sensor's backstop must act on it whatever else has stopped.
static ROUTING: OnceLock<(SharedEventHandle, Arc<AtomicU64>)> = OnceLock::new();

/// (highest wait value handed out, highest signalled); the first leads only while a layer is being resolved.
pub(crate) fn routing_progress() -> Option<(u64, u64)> {
    let (event, counter) = ROUTING.get()?;
    Some((counter.load(Ordering::SeqCst), event.0.signaledValue()))
}

/// Signals the highest value handed out, never more, without a lock or IO; unresolved rows stay null, as on a timeout.
pub(crate) fn answer_waiting_launches() -> Option<u64> {
    let (event, counter) = ROUTING.get()?;
    let armed = counter.load(Ordering::SeqCst);
    event.0.setSignaledValue(armed);
    Some(armed)
}

/// The resolver's own longest deadline: nothing answering for longer means nothing is going to.
pub(crate) const ROUTING_STALL: Duration = POLL_TIMEOUT_AFTER_LOAD;

/// Field order is drop order: the resolver joins first, so it pushes no pin to a dropped ledger or reads a freed page.
pub(crate) struct EventLaneState {
    helper: crate::pool::Worker<Sender<WorkItem>>,
    pub(crate) layout: MailboxLayout,
    mailbox: crate::gpu::Words<u8>,
    routing_event: SharedEventHandle,
    wait_counter: Arc<AtomicU64>,
    pins: Arc<PinLedger>,
    /// `usize::MAX` until the resolver gives up on a layer; the next dispatch then aborts the sequence.
    failed_layer: Arc<AtomicUsize>,
    commit_counter: Arc<AtomicU64>,
    probe: Option<Arc<dyn crate::engine::eval::ResolverProbe>>,
}

impl EventLaneState {
    pub(crate) fn new(
        metal_device: &MetalDevice, rows: &ExpertRows,
        (topk, num_experts, block_count, hidden_dim): (usize, usize, usize, usize), pool: Arc<ExpertPool>,
        router_tables: Arc<OnceLock<RouterTables>>,
    ) -> std::result::Result<Self, String> {
        let device = metal_device.device().as_ref();
        let layout = MailboxLayout::plan(topk, num_experts, hidden_dim);
        let mailbox = alloc_mailbox(device, block_count * layout.region_words() * std::mem::size_of::<u32>())?;
        let routing_event = alloc_shared_event(device)?;
        let pins = Arc::new(PinLedger::default());
        let counter = metal_device.completion_counter().map_err(|e| format!("completion_counter: {e}"))?;
        let completion = metal_device.completion_event().map_err(|e| format!("completion_event: {e}"))?;
        let failed_layer = Arc::new(AtomicUsize::new(usize::MAX));
        let hooks = crate::engine::eval::hooks();
        let probe = hooks.resolver_probe();
        let ctx = Resolver {
            layout,
            mailbox: mailbox.view().clone(),
            block_count,
            routing_event: routing_event.clone(),
            completion_event: SharedEventHandle(completion),
            oracle: pool.oracle().cloned(),
            pool,
            rows: rows.rows.clone(),
            router_tables,
            pins: pins.clone(),
            failed_layer: failed_layer.clone(),
            probe: probe.clone(),
            theta: crate::config::params().approx_theta,
            predict_scratch: Default::default(),
        };
        let helper = crate::pool::Worker::spawn("oscar-event-lane", std::sync::mpsc::channel(), move |rx| {
            resolver_main(ctx, rx)
        })
        .map_err(|e| format!("event lane: helper spawn failed: {e}"))?;
        let wait_counter = Arc::new(AtomicU64::new(0));
        let _ = ROUTING.set((routing_event.clone(), wait_counter.clone()));
        Ok(Self {
            helper,
            layout,
            mailbox,
            routing_event,
            wait_counter,
            pins,
            failed_layer,
            commit_counter: counter,
            probe,
        })
    }

    pub(crate) fn mailbox_buffer(&self) -> &KernelBuffer {
        self.mailbox.view()
    }

    pub(crate) fn theta_on(&self) -> bool {
        self.layout.theta_words > 0
    }

    pub(crate) fn routing_event(&self) -> &Retained<ProtocolObject<dyn MTLSharedEvent>> {
        &self.routing_event.0
    }

    /// The next dispatch's wait value, not taken: the late-tap hook, run before layer T's dispatch, reads T's.
    pub(crate) fn peek_wait_value(&self) -> u64 {
        self.wait_counter.load(Ordering::SeqCst) + 1
    }

    pub(crate) fn next_wait_value(&self) -> u64 {
        self.wait_counter.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Errors only if the resolver thread died.
    pub(crate) fn send_work(&self, item: WorkItem) -> Result<()> {
        self.helper.tx().send(item).map_err(|_| candle_core::Error::Msg("event lane: helper thread exited".into()))
    }

    pub(crate) fn reset_probe(&self) {
        if let Some(p) = &self.probe {
            p.reset();
        }
    }

    pub(crate) fn check_resolver(&self) -> candle_core::Result<()> {
        match self.failed_layer.load(Ordering::Acquire) {
            usize::MAX => Ok(()),
            layer => candle_core::bail!(
                "event lane: resolver could not stage layer {layer}'s routed experts (cause in its message); \
                 sequence aborted rather than continued without them"
            ),
        }
    }

    /// Every pin, at each token boundary, park and lane switch, so none is stranded; a fence past the committed count
    /// names a chain no step flushed (a failed step's), which the take-back's rule clears.
    pub(crate) fn drain_unpins(&self, metal_device: &MetalDevice) -> Result<usize> {
        let (released, highest) = self.pins.release(metal_device, true)?;
        let committed = self.commit_counter.load(Ordering::Relaxed);
        if highest > committed {
            tracing::warn!(target: "oscar::experts",
                "A bookkeeping check failed: pins drained past the committed count, {released} at fence {highest}, committed {committed}"
            );
        }
        Ok(released)
    }

    pub(crate) fn commit_generation(&self) -> u64 {
        self.commit_counter.load(Ordering::Relaxed)
    }

    pub(crate) fn release_completed_pins(&self, metal_device: &MetalDevice) -> Result<usize> {
        if let Some(p) = &self.probe {
            let committed = self.commit_counter.load(Ordering::Relaxed);
            let reached = metal_device.completion_event()?.signaledValue();
            p.decode_layer(committed, reached);
        }
        self.pins.release(metal_device, false).map(|(released, _)| released)
    }
}

mod resolver;
use resolver::resolver_main;
pub(crate) use resolver::Resolver;
