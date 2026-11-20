//! Every hook an evaluation has into oscar; the runtime's own are no-ops.

use std::sync::{Arc, OnceLock};

use crate::engine::dispatcher::OffloadDispatcher;
use crate::pool::{Count, DecodeWindow, LayerExpertKey, TokenSample};
use candle_core::Tensor;
use candle_metal_kernels::metal::command_buffer::CommandBufferObserver;
use oscar_models::moe::RoutedLayer;

/// Whether a key is resident in a pool's bank now.
pub type Residency<'a> = &'a dyn Fn(LayerExpertKey) -> bool;

/// Spans since the previous edge: late-Δ1 staging, visible routing, bookkeeping, resolve, signal, post-signal staging.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolverStage {
    Late,
    Visible,
    Book,
    Resolve,
    Signal,
    Post,
}

/// The resolver thread's clock partition and routing-wait anatomy; without a probe the resolver reads no clock.
pub trait ResolverProbe: Send + Sync {
    /// A dispatch opens: the span since the last edge was idle.
    fn open(&self);
    fn close(&self, stage: ResolverStage);
    fn poll_begin(&self);
    /// `deficit` is the GPU work that stood ahead of this layer's routing.
    fn poll_end(&self, deficit: u64);
    /// A blocking resolve's fetch wait and pin wait.
    fn resolve_block(&self, fetch_ns: u64, pin_ns: u64);
    /// One decode layer: the commit counter and the generation the GPU reached.
    fn decode_layer(&self, committed: u64, reached: u64);
    /// The first decode token after a prefill: the partition restarts.
    fn reset(&self);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WaveReport {
    pub wave_ns: u64,
    pub wait_ns: u64,
    pub encode_ns: u64,
    pub stream_ns: u64,
    /// The wave's keys at its own resolve: in the pool, in flight, never issued.
    pub resident: u64,
    pub in_flight: u64,
    pub cold: u64,
    /// From the stream's issue of the wave's fetches to their landing, where the stream issued them.
    pub fetch_window_ns: Option<u64>,
}

/// What a prefill forward reports as it runs, from the engine lane only.
#[derive(Debug)]
pub enum PrefillEvent {
    Dispatch {
        t0: std::time::Instant,
    },
    Router {
        router_ns: u64,
        readback_ns: u64,
    },
    /// The wave schedule's build time and the forward's token count.
    Schedule {
        ns: u64,
        tokens: u64,
    },
    /// The previous phase's wanted experts, and how many the pool still holds.
    Survival {
        wanted: u64,
        held: u64,
    },
    Unpredicted,
    /// Keys pushed to the prefill stream: `speculative` where predicted, `crossing` where of the next chunk.
    Pushed {
        keys: u64,
        speculative: bool,
        crossing: bool,
    },
    StreamBlocked,
    Wave(WaveReport),
    End(Count),
}

/// The hooks, the sampling seam's trace among them; `pd` is the dispatcher the call is about.
pub trait EvalHooks: mistralrs_core::SampleTrace {
    /// The pool's one launch, at the tail's plan: the runtime's pacer and oracle unless the hooks drive their own.
    fn launch_pool(
        &self, pd: &OffloadDispatcher, hot_budget: usize, carrier_layers: usize, ceiling_hot_budget: usize,
        offloads: bool,
    ) -> crate::pool::Result<()> {
        pd.launch_pool(hot_budget, carrier_layers, ceiling_hot_budget, offloads)
    }
    /// A recording keeps every layer's routing on the host: the pool launches all-carrier and nothing seals.
    fn records_routing(&self) -> bool {
        false
    }
    /// Every engine-thread call, once its call count is known.
    fn on_call(&self, _pd: &OffloadDispatcher, _layer_idx: usize, _call_count: u32, _num_tokens: usize) {}
    /// The prefill is dispatched: the forwards from here are what a trace describes.
    fn note_prefill(&self) {}
    /// The live routed set of a call, deduplicated: the readback's and the event lane resolver's resolved routing.
    fn check_routing(&self, _layer_idx: usize, _call_count: u32, _num_tokens: usize, _unique: &[u16]) {}
    /// `Some(Some(ids, weights))` replaces a decode call's routing, `Some(None)` routes it exactly (no record).
    #[allow(clippy::option_option)]
    fn routing_override(&self, _layer_idx: usize, _call_count: u32) -> Option<Option<(Vec<u32>, Vec<f32>)>> {
        None
    }
    /// A routed call under a recording, with a decode call's raw router logits.
    #[allow(clippy::too_many_arguments)]
    fn record_call(
        &self, _pd: &OffloadDispatcher, _layer: &RoutedLayer, _layer_idx: usize, _call_count: u32, _num_tokens: u32,
        _ids: &[u32], _weights: &[f32], _router_logits: &Tensor,
    ) -> candle_core::Result<()> {
        Ok(())
    }
    /// The engine's phase at a forward's layer 0: decode, or a prefill.
    fn phase(&self, _decode: bool) {}
    /// One decode token's sample of the cumulative counters.
    fn token_sample(&self, _sample: &TokenSample) {}
    /// A decode step's wall time, and whether the drafter's round ran it.
    fn note_step(&self, _drafted: bool, _took: std::time::Duration) {}
    /// Taken once, when the resolver thread is built.
    fn resolver_probe(&self) -> Option<Arc<dyn ResolverProbe>> {
        None
    }
    /// Every command buffer's callbacks after the runtime's own, taken once at the probe's install.
    fn gpu_observer(&self) -> Option<Arc<dyn CommandBufferObserver>> {
        None
    }
    /// A blit binds no pipeline, so its label names the row.
    fn note_blit(&self, _label: &str) {}
    fn prefill(&self, _e: PrefillEvent) {}
    /// Scoring computes the runner-ups past the top-k and keeps the routings that ran.
    fn scores_hints(&self) -> bool {
        false
    }
    /// A decode token's routing at `layer_idx` as it ran, deduplicated.
    fn note_actual_routing(&self, _block_count: usize, _layer_idx: usize, _unique: &[u16]) {}
    /// The host chain's ranking for `target`, `step` layers past its tap, and the experts it admitted to stage.
    fn note_lookahead(&self, _target: usize, _step: usize, _ranked: &[u16], _admitted: &[u16]) {}
    /// A consumed drafted-window entry at distance `dist`, accepted or not.
    fn score_hints(&self, _dist: u32, _accepted: bool, _hints: &[Vec<u16>], _extras: &[Vec<u16>]) {}
    fn note_draft_window(&self, _window: &[Vec<Vec<u16>>], _residency: Residency<'_>) {}
    /// The drafter on a fixed `(on, off)` schedule of boundaries, repeating, in place of the controller.
    fn drafter_pinned(&self) -> Option<(u64, u64)> {
        None
    }
    fn request_end(&self, _w: &DecodeWindow) {}
    fn memory_sample(&self, _snap: &crate::machine::MemorySnapshot) {}
    /// The load is done, warm-up included: nothing is in flight.
    fn load_done(&self) {}
    /// The census's record, read before it is dropped.
    fn census_kept(&self, _census: &crate::load::census::Census, _record: &crate::load::census::Record) {}
}

struct Production;

impl EvalHooks for Production {}
impl mistralrs_core::SampleTrace for Production {}

static HOOKS: OnceLock<Arc<dyn EvalHooks>> = OnceLock::new();

/// Once, before the model is built.
pub fn install(hooks: Arc<dyn EvalHooks>) {
    if HOOKS.set(hooks).is_err() {
        panic!("eval hooks installed once, before model build");
    }
}

pub fn hooks() -> &'static Arc<dyn EvalHooks> {
    HOOKS.get_or_init(|| Arc::new(Production))
}
