//! The MoE forward over the offload runtime, one sequence in flight: routed experts stream from disk into GPU slots.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use candle_core::{DType, Result, Tensor};
use oscar_kernels::{
    fused_down_alignment, fused_gateup_alignment, moe_down, moe_gateup, moe_gemm, moe_matvec, moe_topk_weighted_sum,
    Metal,
};
use oscar_models::ffn::apply_expert_activation;
use oscar_models::moe::{MoeKernelDispatcher, MoeLookahead, RoutedLayer, RunnerUps};

use crate::engine::eval::PrefillEvent;
use crate::engine::event_lane::RoutingIds;
use crate::engine::router::dedup_u32_to_u16;
use crate::gpu::expert_rows::{ExpertRows, MissCounter};
use crate::gpu::wave_pool::{waves, WavePool};
use crate::gpu::GgufRoutedLayout;
use crate::pool::{
    Bank, BankPools, Due, ExpertPool, Lane, Launch, LayerExpertKey, PrefetchJob, PrefetchThread, RecordedSchedule,
    SchedulePacer, COUNTERS,
};
use oscar_models::ffn::{Finalized, ForwardShape, Stream};

mod draft;
mod gpu_dispatch;
pub(crate) mod lookahead;
mod prefill_waves;
mod readback;

/// Decode's command buffers committed ahead of the GPU; the planner's floor holds their routed sets.
pub(crate) const ENCODE_AHEAD_CBS: u64 = 8;

/// By shape, a drafting model's verify step included; the memory phase is the engine's step kind, never this.
fn is_multi_token_forward(num_tokens: usize) -> bool {
    num_tokens > 1
}

thread_local! {
    // Thread-scoped, as the drafter enters through the global trait with no argument channel.
    static MOE_LANE: std::cell::Cell<Lane> = const { std::cell::Cell::new(Lane::Engine) };
}

fn current_lane() -> Lane {
    MOE_LANE.with(|c| c.get())
}

fn on_draft_thread() -> bool {
    current_lane() == Lane::Drafter
}

/// The event lane state, set at the load's tail by [`OffloadDispatcher::build_paged_lanes`].
pub(super) struct EventLane {
    pub(super) state: OnceLock<crate::engine::event_lane::EventLaneState>,
    pub(super) scratch: LaneScratch,
}

/// One per lane, so the drafter never co-writes the engine's address rows.
pub(super) struct LaneScratch {
    pub(super) rows: ExpertRows,
    pub(super) miss_counter: MissCounter,
}

/// Every MoE launch installs one: residency is the plan's (a resident-fit model seals), never a mode. Field order is
/// drop order: the resolver and the prefill-ahead thread go before the rows and the pool they read.
pub struct OffloadDispatcher {
    ep: EventLane,
    draft: draft::DraftScratch,
    wave_pool: WavePool,
    prefetch: PrefetchThread,
    pool: Arc<ExpertPool>,
    layout: GgufRoutedLayout,
    eval: Arc<dyn crate::engine::eval::EvalHooks>,
    per_layer_call_counts: Vec<AtomicU64>,
    gate_book: Vec<OnceLock<(candle_core::quantized::QMatMul, Option<Tensor>)>>,
    /// An `Arc`, as the event lane state's build and the tables' hand-over come in either order.
    router_tables: Arc<OnceLock<crate::engine::event_lane::RouterTables>>,
    /// Kept across crossings, as it holds layer ordinals, not slots.
    draft_window: OnceLock<Arc<draft::DraftedWindow>>,
    /// From 1, so the load's dummy decode opens nothing; a prefill zeroes it.
    decode_tokens: AtomicU64,
    logged_run_ahead: AtomicBool,
}

thread_local! {
    /// The address of the pool whose residency sets this thread's queue holds.
    static POOL_ATTACHED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) struct Call<'a> {
    pub(super) pool: &'a ExpertPool,
    pub(super) layer: &'a RoutedLayer,
    pub(super) idx: usize,
    pub(super) call_count: u32,
    pub(super) tick: u32,
    /// The layer's gate, up and down banks.
    pub(super) banks: [&'a Arc<Bank>; 3],
    /// `[tokens, hidden]` in F32.
    pub(super) xs: Tensor,
    pub(super) tokens: usize,
    pub(super) shape: ForwardShape,
    pub(super) device: candle_core::MetalDevice,
    pub(super) stream: Option<Stream<'a>>,
}

/// The routing read back to the host after any intervention: what the pins and rows are built from.
pub(super) struct Routing {
    ids: Tensor,
    weights: Tensor,
    host: Vec<u32>,
    unique: Vec<u16>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Route {
    Sealed,
    Draft,
    /// Its routing published by GPU blit, never read back.
    Paged,
    /// Then the paged decode off the host's ids where the layer can take it, else the wave loop.
    ReadBack,
}

impl OffloadDispatcher {
    /// `gguf_paths` in shard-id order; [`Self::launch_pool`] or [`Self::install_recorded_plane`] shapes the pool.
    pub fn new(gguf_paths: &[PathBuf], layout: GgufRoutedLayout, segment_slots: usize) -> crate::pool::Result<Self> {
        let pool = Arc::new(ExpertPool::build(&layout, gguf_paths, segment_slots)?);
        let prefetch = pool.spawn_prefetch()?;
        let refused = |what: &'static str| move |e: String| oscar_offload::Error::Layout(format!("{what}: {e}"));
        let (n, layers, dev) = (layout.num_experts, layout.block_count, pool.device());
        let wave_pool = WavePool::new(dev, n).map_err(refused("the wave ring"))?;
        crate::detail!(target: "oscar::experts", "offload runtime: {} IO workers", pool.reader_count());
        let lane = || {
            Ok::<_, String>(LaneScratch {
                rows: ExpertRows::new(dev, n, layers)?,
                miss_counter: MissCounter::new(dev, layers)?,
            })
        };
        let lanes = || -> std::result::Result<_, String> {
            let (snapshot, dirty) = (draft::DraftSnapshot::new(dev, n, layers)?, MissCounter::new(dev, 1)?);
            Ok((lane()?, draft::DraftScratch { scratch: lane()?, snapshot, dirty, dirty_last: AtomicU64::new(0) }))
        };
        let (scratch, draft) = lanes().map_err(refused("the paged lanes"))?;
        let pd = Self {
            ep: EventLane { state: OnceLock::new(), scratch },
            draft,
            wave_pool,
            prefetch,
            pool,
            per_layer_call_counts: (0..layout.block_count).map(|_| AtomicU64::new(0)).collect(),
            gate_book: (0..layout.block_count).map(|_| OnceLock::new()).collect(),
            layout,
            eval: crate::engine::eval::hooks().clone(),
            router_tables: Arc::new(OnceLock::new()),
            draft_window: OnceLock::new(),
            decode_tokens: AtomicU64::new(1),
            logged_run_ahead: AtomicBool::new(false),
        };
        pd.note_pool_regime();
        Ok(pd)
    }

    pub fn pool(&self) -> &Arc<ExpertPool> {
        &self.pool
    }

    pub fn layout(&self) -> &GgufRoutedLayout {
        &self.layout
    }

    /// Whether the pool has the plan's shape.
    pub fn launched(&self) -> bool {
        self.pool.launched().is_some()
    }

    /// Records whether the pool's shape has carriers, for [`offloading_live`].
    pub(super) fn note_pool_regime(&self) {
        let carriers = self.pool.carrier_count() > 0;
        PAGING_LIVE.store(carriers, Ordering::Relaxed);
        crate::drafter::note_pool_moved();
        if carriers && crate::config::drafter_on() && self.launched() {
            self.arm_draft_window();
        }
    }

    /// The pool's slots per layer; 0 before the launch.
    pub fn hot_budget(&self) -> usize {
        self.pool.hot_budget()
    }
}

impl OffloadDispatcher {
    /// The layer's position in the token, the token boundary at layer 0, the demand clock's tick, the input in F32.
    fn begin_call<'a>(
        &self, pool: &'a ExpertPool, layer: &'a RoutedLayer, xs: &Tensor, stream: Option<Stream<'a>>,
    ) -> Result<Call<'a>> {
        let idx = layer.dispatch_ordinal();
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let tokens = b_size * seq_len;
        // The drafter and a census observe without advancing: the stager is lead-paced and eviction scores by tick.
        let observing = on_draft_thread() || oscar_offload::census::armed();
        let call_count = self.advance_call_count(idx, tokens, observing);
        if !observing {
            self.eval.on_call(self, idx, call_count, tokens);
            crate::gpu::probe::mark_engine_thread();
            if idx == 0 {
                self.token_boundary(pool, tokens, xs)?;
            }
        }
        // Unlaunched, the pool has no schedule: only the census forwards over it, stamping nothing that outlives it.
        let tick = pool.pacer().map_or_else(|| pool.tick.load(Ordering::Relaxed), |p| p.target_tick(idx, call_count));
        if !observing {
            pool.tick.swap(tick, Ordering::AcqRel);
        }
        let shape =
            ForwardShape { b_size, seq_len, hidden_dim, out_dtype: xs.dtype(), num_experts: self.layout.num_experts };
        Ok(Call {
            pool,
            layer,
            idx,
            call_count,
            tick,
            banks: pool.banks_for(idx),
            xs: xs.reshape(((), hidden_dim))?.to_dtype(DType::F32)?,
            tokens,
            shape,
            device: xs.device().as_metal_device()?.clone(),
            stream,
        })
    }

    /// A prefill closes the previous window at its start, as a one-token turn has no decode forward to close it.
    fn token_boundary(&self, pool: &ExpertPool, num_tokens: usize, xs: &Tensor) -> Result<()> {
        let prefill = is_multi_token_forward(num_tokens);
        let opens_decode = if prefill {
            self.decode_tokens.store(0, Ordering::Relaxed);
            false
        } else {
            self.decode_tokens.fetch_add(1, Ordering::Relaxed) == 0
        };
        self.eval.phase(!prefill);
        if prefill {
            self.eval.prefill(PrefillEvent::End(COUNTERS.now()));
        }
        if opens_decode {
            let t0 = std::time::Instant::now();
            self.eval.prefill(PrefillEvent::End(COUNTERS.now()));
            crate::pool::reset_demand_clock();
            crate::pool::counters::reset_reads();
            if crate::config::params().misc.diag {
                tracing::info!(target: "oscar::experts", "prefill-boundary: release={}ms", t0.elapsed().as_millis());
            }
            pool.series.begin_window();
        }
        if !prefill {
            COUNTERS.decode_armed.store(true, Ordering::Release);
            if let Some(sample) = pool.series.record() {
                self.eval.token_sample(&sample);
            }
            crate::gpu::probe::arm_wake_detector();
        }
        if let Some(ep) = self.ep.state.get() {
            // Reset at decode's first token, not at the prefill, whose idle wait would swamp the probe's bucket.
            if opens_decode {
                ep.reset_probe();
            }
            ep.drain_unpins(xs.device().as_metal_device()?)?;
        }
        Ok(())
    }

    /// This call's position in the token, the shared counter left as it is while `observing`.
    fn advance_call_count(&self, layer_idx: usize, num_tokens: usize, observing: bool) -> u32 {
        let counter = &self.per_layer_call_counts[layer_idx];
        if observing {
            return counter.load(Ordering::Relaxed) as u32;
        }
        if is_multi_token_forward(num_tokens) {
            // Position 0 of a new generation.
            counter.store(0, Ordering::Relaxed);
            self.eval.note_prefill();
            return counter.fetch_add(1, Ordering::Relaxed) as u32;
        }
        counter.fetch_add(1, Ordering::Relaxed) as u32
    }
}

impl MoeKernelDispatcher for OffloadDispatcher {
    fn forward(&self, layer: &RoutedLayer, xs: &Tensor, stream: Option<Stream<'_>>) -> Result<Finalized> {
        self.forward_any(layer, xs, stream)
    }
}

impl OffloadDispatcher {
    /// Any token count: every expert of a sealed layer is resident, so no mailbox, resolver, pins or commit.
    fn forward_sealed(&self, call: Call<'_>) -> Result<Finalized> {
        let rows = call.pool.sealed_rows().layer_rows(&call.device, call.idx)?;
        let pools = BankPools::new(&call.device, call.banks);
        let logits = call.layer.gate_forward(&call.xs)?;
        let (ids, weights) = call.layer.route_topk(&logits)?;
        self.encode_layer(call, &pools, &rows, &ids, &weights, false)
    }
}

impl OffloadDispatcher {
    /// Once at the thread's start; the drafter's lane never takes the paged route, whose mailbox is the engine's.
    pub fn set_thread_lane(&self, lane: Lane) {
        MOE_LANE.with(|c| c.set(lane));
    }
}

/// Each projection's kernel per call, GEMM or matvec, keyed on the whole call's token count, never a slice's: the
/// two differ in FP. Gate and up cross over at 256 tokens on Q4_K; down at every multi-token call, as per-slot outputs
/// let a carrier sum its waves once, in a sealed layer's order.
#[derive(Clone, Copy)]
struct MoeFormulation {
    mm_gate_up: bool,
    mm_down: bool,
}

/// F32 `(tokens, down's n)`, no shared expert; `experts` are what the rows resolve, the GEMM's columns (`None`: all).
/// `down_into` keeps down's per-slot outputs unsummed, in one tensor the first call makes and later calls fill.
#[allow(clippy::too_many_arguments)]
fn route_rows(
    layer: &RoutedLayer, xs: &Tensor, pools: &BankPools, rows: [&Tensor; 3], experts: Option<&Tensor>,
    topk_ids: &Tensor, topk_weights: &Tensor, miss_count: &Tensor, call_tokens: usize,
    down_into: Option<&mut Option<Tensor>>,
) -> Result<Tensor> {
    let (gate, up, down) = layer.expert_tensors();
    let experts = experts.unwrap_or(&gate.all_experts);
    let xs_for_moe = &layer.expert_input(xs)?;
    if gate.n != up.n || gate.k != up.k {
        candle_core::bail!(
            "gate/up shape mismatch: gate=({gn},{gk}) up=({un},{uk})",
            gn = gate.n,
            gk = gate.k,
            un = up.n,
            uk = up.k,
        );
    }
    if down.k != gate.n {
        candle_core::bail!("down K ({}) != gate N ({}) - MoE shape mismatch", down.k, gate.n);
    }
    let num_experts = gate.num_experts;
    let [gate_row, up_row, down_row] = pools.rows(rows);
    let (activation, f) =
        (layer.activation, MoeFormulation { mm_gate_up: call_tokens >= 256, mm_down: call_tokens >= 2 });

    // The fused gate+up takes one format for both stacks.
    let fused_gateup = gate.dtype == up.dtype
        && fused_gateup_alignment(gate.dtype, activation).is_some_and(|a| gate.n.is_multiple_of(a));
    let activated = if f.mm_gate_up {
        let g = moe_gemm(gate_row, gate.dtype, experts, num_experts, gate.n, gate.k, xs_for_moe, topk_ids, None)?;
        let u = moe_gemm(up_row, up.dtype, experts, num_experts, up.n, up.k, xs_for_moe, topk_ids, None)?;
        apply_expert_activation(&g, &u, activation)?
    } else if fused_gateup {
        moe_gateup(
            gate_row,
            up_row,
            gate.dtype,
            activation,
            num_experts,
            gate.n,
            gate.k,
            topk_ids,
            xs_for_moe,
            miss_count,
        )?
    } else {
        let g = moe_matvec(gate_row, gate.dtype, num_experts, gate.n, gate.k, xs_for_moe, topk_ids, None)?;
        let u = moe_matvec(up_row, up.dtype, num_experts, up.n, up.k, xs_for_moe, topk_ids, None)?;
        apply_expert_activation(&g, &u, activation)?
    };

    let topk_w = topk_weights.to_dtype(DType::F32)?;
    let fused_down = fused_down_alignment(down.dtype).is_some_and(|a| down.n.is_multiple_of(a));
    if !f.mm_down && fused_down && down_into.is_none() {
        return moe_down(down_row, down.dtype, num_experts, down.n, down.k, topk_ids, &topk_w, &activated);
    }
    let into = down_into.as_ref().and_then(|slot| slot.as_ref());
    let down_out = if f.mm_down {
        moe_gemm(down_row, down.dtype, experts, num_experts, down.n, down.k, &activated, topk_ids, into)?
    } else {
        moe_matvec(down_row, down.dtype, num_experts, down.n, down.k, &activated, topk_ids, into)?
    };
    match down_into {
        Some(slot) => Ok(slot.get_or_insert(down_out).clone()),
        None => moe_topk_weighted_sum(&down_out, &topk_w, down_row, topk_ids),
    }
}

impl OffloadDispatcher {
    fn forward_any(&self, layer: &RoutedLayer, xs: &Tensor, stream: Option<Stream<'_>>) -> Result<Finalized> {
        let t_dispatch = std::time::Instant::now();
        let tokens = xs.dim(0).unwrap_or(0) * xs.dim(1).unwrap_or(0);
        // A crossing asked inside a chunk parks here, before this call holds the pool; decode, at its token boundary.
        if is_multi_token_forward(tokens) {
            mistralrs_core::paged_attention::park_for_memory_crossing(xs.device());
        }
        // Refused at load, where the census visits every routed layer.
        if oscar_offload::census::armed() && !layer.has_fused_gateup() {
            panic!("routed layer {}: no fused gate+up kernel for its format and activation", layer.dispatch_ordinal());
        }
        if let candle_core::Device::Metal(md) = xs.device() {
            self.ensure_queue(md)?;
        }
        let result = self.forward_routed(self.begin_call(&self.pool, layer, xs, stream)?);
        if is_multi_token_forward(tokens) {
            self.eval.prefill(PrefillEvent::Dispatch { t0: t_dispatch });
            // The bar's prompt count moves by the layer.
            mistralrs_core::live::prompt_layer(tokens as u64);
        }
        result
    }

    /// A census takes the arm asked of it, at decode width the sealed one unless drafting, whose routing it records.
    fn route(&self, call: &Call<'_>) -> Route {
        let draft = on_draft_thread();
        let one_token = call.tokens <= 1;
        assert!(one_token || !draft, "the drafter forwards one token at a time");
        if oscar_offload::census::armed() {
            if draft {
                return Route::Draft;
            }
            if one_token || oscar_offload::census::arm_of() == oscar_offload::census::Arm::Sealed {
                return Route::Sealed;
            }
            return Route::ReadBack;
        }
        let recording = self.eval.records_routing();
        if call.pool.layer_sealed(call.idx) && !recording {
            return Route::Sealed;
        }
        match (one_token, draft) {
            (_, true) => Route::Draft,
            (true, false) if !recording => Route::Paged,
            _ => Route::ReadBack,
        }
    }

    fn forward_routed(&self, call: Call<'_>) -> Result<Finalized> {
        match self.route(&call) {
            Route::Sealed => self.forward_sealed(call),
            Route::Draft => self.forward_decode_draft(call),
            Route::Paged => self.forward_paged(call, None),
            Route::ReadBack => {
                self.leave_gpu_routed_lane(&call)?;
                let (banks, routing) = (call.banks, self.read_back(&call)?);
                let result = if self.takes_host_routing(&call, &routing) {
                    self.forward_paged(call, Some(routing))
                } else {
                    self.forward_expert_waves(call, routing)
                };
                banks.iter().for_each(|b| b.clear_routed());
                result
            }
        }
    }

    /// The rows are raw windows, filled before the event signals; `commit` flushes so the GPU follows the resolver.
    fn encode_layer(
        &self, call: Call<'_>, pools: &BankPools, rows: &[Tensor; 3], ids: &Tensor, weights: &Tensor, commit: bool,
    ) -> Result<Finalized> {
        debug_assert_eq!(ids.dims()[0], call.tokens, "the routing is [tokens, k]");
        let miss = self.miss_slot(&call)?;
        let rows = [&rows[0], &rows[1], &rows[2]];
        let routed = route_rows(call.layer, &call.xs, pools, rows, None, ids, weights, &miss, call.tokens, None)?;
        if commit {
            call.device.flush_commands()?;
        }
        call.layer.finalize_routed(Some(&routed), &call.xs, call.stream, call.shape)
    }

    /// This lane's miss counter at the call's layer: the kernel honours the view's `start_offset`.
    fn miss_slot(&self, call: &Call<'_>) -> Result<Tensor> {
        self.lane_scratch().miss_counter.as_tensor(&call.device)?.narrow(0, call.idx, 1)
    }

    /// Once per thread, before its first launch reads a slot: the banks' residency sets and a performance QoS.
    fn ensure_queue(&self, md: &candle_core::MetalDevice) -> Result<()> {
        let me = Arc::as_ptr(&self.pool) as usize;
        if POOL_ATTACHED.with(|c| c.get()) == me {
            return Ok(());
        }
        crate::engine::set_interactive_qos();
        self.pool.attach_queue(&md.command_queue()?);
        POOL_ATTACHED.with(|c| c.set(me));
        Ok(())
    }
}

impl OffloadDispatcher {
    /// Once, the pacer and oracle sized at `ceiling_hot_budget`; sealed layers are read and verified here.
    pub fn launch_pool(
        &self, hot_budget: usize, carrier_layers: usize, ceiling_hot_budget: usize, offloads: bool,
    ) -> crate::pool::Result<()> {
        // A recording needs every layer's routing on the host: all-carrier, nothing seals.
        let carrier_layers = if self.eval.records_routing() { self.layout.block_count } else { carrier_layers };
        let pacer = SchedulePacer::Procedural(self.pool.decode_pacer(ceiling_hot_budget));
        let oracle = self.pool.select_oracle(ceiling_hot_budget, None, offloads);
        self.pool.launch(hot_budget, carrier_layers, Launch { hot_budget: ceiling_hot_budget, pacer, oracle })?;
        self.note_pool_regime();
        Ok(())
    }

    /// An evaluation's launch at the plan's shape, all-carrier, the schedule's ticks and routings pacing the pool.
    pub fn install_recorded_plane(
        &self, schedule: Arc<dyn RecordedSchedule>, hot_budget: usize, ceiling_hot_budget: usize,
    ) -> crate::pool::Result<()> {
        let oracle = self.pool.select_oracle(ceiling_hot_budget, Some(schedule.clone()), false);
        let launch = Launch { hot_budget: ceiling_hot_budget, pacer: SchedulePacer::Recorded(schedule), oracle };
        self.pool.launch(hot_budget, self.layout.block_count, launch)?;
        self.note_pool_regime();
        Ok(())
    }

    /// At a park or a take-back: the completion event set on every bank, then the resolver's unpins; out, their count.
    fn sweep_pins(&self, md: &candle_core::MetalDevice) -> Result<usize> {
        self.pool.set_completion_event(&md.completion_event()?);
        self.ep.state.get().map_or(Ok(0), |ep| ep.drain_unpins(md))
    }

    /// At a parked boundary: the unseals and retires now, the admits, seals and frees owed to [`Self::settle_moves`].
    pub fn move_pool(
        &self, hot_budget: usize, carrier_layers: usize, dev: &candle_core::Device,
    ) -> Result<crate::pool::Moves> {
        // One move at a time: the crossing parks only once the mover's last free has landed, every index given back.
        let carrier_of = crate::pool::carrier_of(carrier_layers, self.layout.block_count);
        // A park sweeps the resolver's pins as a token boundary would: a stale pin holds its retired segment for good.
        // The queue drains first so the fence is met as set; a resolver behind it would evict the slots and deadlock.
        let gen = match dev {
            candle_core::Device::Metal(md) => {
                md.flush_commands()?;
                md.wait_until_completed()?;
                let swept = self.sweep_pins(md)?;
                let gen = md.fence_generation()?;
                if crate::cli::console::traced() {
                    tracing::info!(target: "oscar::experts",
                        "parked - {swept} pins swept, fence {gen} (committed {}){}",
                        md.completion_counter()?.load(Ordering::SeqCst),
                        crate::cli::console::stamp()
                    );
                }
                gen
            }
            _ => 0,
        };
        let moves = self.pool.move_to(hot_budget, &carrier_of, gen);
        self.note_pool_regime();
        Ok(moves)
    }

    /// The mover's half: the admits and seals, the frees as they clear, then `owed`; true once the move is over.
    pub fn settle_moves(
        &self, held_since: &mut Option<std::time::Instant>, drafter_busy: impl Fn() -> bool,
        re_land: impl FnOnce((usize, usize), &str, &str), owed: impl FnOnce(),
    ) -> bool {
        self.pool.settle_moves(held_since, drafter_busy, re_land, owed, || self.note_pool_regime())
    }

    /// After the pool's launch and the region's take, at the live plan's shape; decode's then, until a prompt.
    pub fn lend_at_load(&self, shape: (usize, usize), decode_bound: u64, decode: Option<(usize, u64)>) {
        self.pool.lend_at_load(shape, decode_bound, decode, self.eval.records_routing());
    }

    /// A completion step lends down to decode's bound; a prompt left short fails here, not later in an allocation.
    pub fn phase_change(
        &self, step: &mistralrs_core::paged_attention::StepKind<'_>, quiesce: impl FnOnce(),
    ) -> std::result::Result<(), String> {
        self.pool.phase_change(step, quiesce, |md| self.sweep_pins(md).map_err(|e| e.to_string()))
    }

    /// At the load's tail, every plane alike: an all-sealed launch first pages under a squeeze.
    pub fn build_paged_lanes(
        &self, device: &candle_core::MetalDevice, topk: usize, hidden_dim: usize,
    ) -> std::result::Result<(), String> {
        let dims = (topk, self.layout.num_experts, self.layout.block_count, hidden_dim);
        let (pool, tables) = (Arc::clone(&self.pool), Arc::clone(&self.router_tables));
        let ep = crate::engine::event_lane::EventLaneState::new(device, &self.ep.scratch.rows, dims, pool, tables)?;
        let _ = self.ep.state.set(ep);
        Ok(())
    }
}

/// Whether the live pool has carrier layers, the regime every prefetch hint follows; written at each shape it takes.
static PAGING_LIVE: AtomicBool = AtomicBool::new(false);

pub fn offloading_live() -> bool {
    PAGING_LIVE.load(Ordering::Relaxed)
}
