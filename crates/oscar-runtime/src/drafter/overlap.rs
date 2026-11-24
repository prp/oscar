//! The engine's side of the drafter: window alignment, the session hand-off and the shedding controller.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use candle_core::{Result, Tensor};

use super::thread;
use mistralrs_core::kv_cache::EitherCache;
use mistralrs_core::sequence::Sequence;
use mistralrs_core::text_models_inputs_processor::PagedAttentionMeta;
use mistralrs_core::{CacheBackendMetadata, Pipeline};

/// One session in flight: a second queued behind it lands every draft late.
const SPAWN_DEPTH: usize = 1;

/// Lock-free. Offloading only, a hint being its only product; else the pipelined step and the device sampler serve.
pub fn drafting() -> bool {
    DRAFTER.get().is_some()
        && crate::engine::dispatcher::offloading_live()
        && !shed_now()
        && HELD.load(Ordering::Relaxed)
}

static HELD: AtomicBool = AtomicBool::new(false);

/// The launch's: no session is handed off where the plan books no drafter memory.
pub fn set_held(held: bool) {
    HELD.store(held, Ordering::Relaxed);
}

static SHED_NOW: AtomicBool = AtomicBool::new(false);
static POOL_MOVED: AtomicBool = AtomicBool::new(false);

/// The pool's shape moved (its launch, a crossing's move): the controller's verdict was the old shape's.
pub fn note_pool_moved() {
    POOL_MOVED.store(true, Ordering::Relaxed);
}

/// The eval's fixed `(on, off)` boundary schedule in place of the verdicts; `None` outside the eval.
fn pinned() -> Option<(u64, u64)> {
    static P: OnceLock<Option<(u64, u64)>> = OnceLock::new();
    *P.get_or_init(|| crate::engine::eval::hooks().drafter_pinned())
}

pub(super) fn shed_now() -> bool {
    SHED_NOW.load(Ordering::Relaxed)
}

/// Outlives requests: the pace regime is the machine's and the plan's, not a request's.
#[derive(Default)]
struct Control {
    shed: Shed,
    last_boundary: Option<std::time::Instant>,
    pinned_boundaries: u64,
    /// The drafted window's depths summed, at the last boundary.
    depth_seen: u64,
    /// Decode boundaries since the last prompt.
    decoded: u32,
}

static CONTROL: std::sync::Mutex<Option<Control>> = std::sync::Mutex::new(None);

/// On every step's path: the step clock and the controller; a prompt or batch (`!decoding`) restarts the clock.
pub(super) fn boundary(decoding: bool, context: usize) {
    let mut guard = CONTROL.lock().unwrap();
    let c = guard.get_or_insert_with(Control::default);
    let now = std::time::Instant::now();
    // A stall an order above a step (a crossing's move, a pause) is not pace.
    let ms = c
        .last_boundary
        .filter(|_| decoding)
        .map(|prev| now.duration_since(prev).as_secs_f64() * 1e3)
        .filter(|&ms| ms < 2_000.0);
    c.last_boundary = decoding.then_some(now);
    c.decoded = if decoding { c.decoded + 1 } else { 0 };
    // A reply's first tokens run while the pool follows the prompt, as a block's do after a switch: not pace.
    let pace = ms.filter(|_| c.decoded > Shed::SKIP);
    // Whether this token began with a draft ahead of it.
    let depth = super::DRAFT.depth_sum.load(Ordering::Relaxed);
    let ahead = depth != std::mem::replace(&mut c.depth_seen, depth);
    let Some(drafter) = DRAFTER.get().filter(|_| decoding) else { return };
    let was = !SHED_NOW.load(Ordering::Relaxed);
    // Nothing to draft while sealed.
    let offloading = crate::engine::dispatcher::offloading_live();
    let drafting = if !offloading {
        false
    } else if let Some((on, off)) = pinned() {
        c.pinned_boundaries += 1;
        (c.pinned_boundaries - 1) % (on + off) < on
    } else {
        if POOL_MOVED.swap(false, Ordering::Relaxed) {
            c.shed.rearm();
        }
        if let Some((switched, said)) = c.shed.observe(pace, ahead, context) {
            if switched {
                tracing::info!(target: "oscar::drafter", "{said}");
            } else {
                crate::detail!(target: "oscar::drafter", "{said}");
            }
        }
        c.shed.drafting()
    };
    SHED_NOW.store(!drafting, Ordering::Relaxed);
    if ms.is_some() {
        super::DRAFT.boundaries.fetch_add(1, Ordering::Relaxed);
        super::DRAFT.off.fetch_add(u64::from(!drafting), Ordering::Relaxed);
    }
    // Off: the session and its staged window go; no round will slide it out.
    if was && !drafting {
        if let Some(st) = DRAFT_STATE.lock().unwrap().as_mut() {
            st.cancel(drafter);
        }
        if let Some(pd) = crate::engine::offload_dispatcher() {
            pd.push_draft_window(Vec::new(), true);
        }
    }
}

/// Drafting only while it makes decode faster. A drafter whose drafts are not ahead of decode is shed at once. At a
/// hold's end a probe runs one block in the other state, judged against the hold's last tokens before it and a block
/// in the current state after, so the text's own drift cancels.
#[derive(Default)]
struct Shed {
    shed: bool,
    phase: ShedPhase,
    /// Tokens into the hold, or into the probe's block.
    tokens: u32,
    /// The probe's block: odd the other state's, even the current state's.
    block: u32,
    /// This block's measured tokens: their count, their ms summed and squared.
    measured: u32,
    sum: f64,
    sumsq: f64,
    /// The hold's last measured tokens, four blocks' worth: their ms (the last block the probe's first reference,
    /// the blocks' spread its noise), and whether each of the drafting hold's last tokens began with a draft ahead.
    recent: std::collections::VecDeque<f64>,
    ahead: std::collections::VecDeque<bool>,
    /// The hold's spread block to block: a slow stretch is the hold's, not a state's time.
    block_sd: f64,
    current: Option<Block>,
    other: Option<Block>,
    /// Each pair: the other state's block less its neighbours' mean, ms a token, with its standard error.
    diffs: Vec<(f64, f64)>,
    /// The next hold's tokens, doubled by each standing verdict.
    hold: u32,
    /// The verdict holds while the context stays within a hold's tokens of where its probe began.
    verdict_context: usize,
    probe_context: usize,
}

#[derive(Default, Clone, Copy, PartialEq, Debug)]
enum ShedPhase {
    #[default]
    Hold,
    Probe,
}

/// A block's measured tokens: their count, mean and variance, ms a token; one token has no spread to speak of.
#[derive(Clone, Copy, Debug)]
struct Block {
    n: u32,
    mean: f64,
    var: f64,
}

impl Block {
    fn of(n: u32, sum: f64, sumsq: f64) -> Option<Self> {
        let (count, mean) = (f64::from(n), sum / f64::from(n.max(1)));
        let var = if n > 1 { (sumsq / count - mean * mean).max(0.0) * count / (count - 1.0) } else { f64::INFINITY };
        (n > 0).then_some(Self { n, mean, var })
    }

    /// The squared standard error of the mean.
    fn se2(&self) -> f64 {
        self.var / f64::from(self.n)
    }
}

impl Shed {
    /// Tokens before a probe: first, after a switch, at most; the controller costs only a probe's other-state tokens.
    const HOLD_FIRST: u32 = 128;
    const HOLD: u32 = 2048;
    const HOLD_MAX: u32 = 16 * Self::HOLD;
    /// A probe's block, and the tokens after a switch it leaves out while the lead lands and the pool follows.
    const BLOCK: u32 = 48;
    const SKIP: u32 = 16;
    /// The tokens a block measures, and the reference the hold keeps: four blocks' worth.
    const MEASURED: usize = (Self::BLOCK - Self::SKIP) as usize;
    const RECENT: usize = 4 * Self::MEASURED;
    const PAIRS_MAX: usize = 8;

    fn drafting(&self) -> bool {
        let other = self.phase == ShedPhase::Probe && self.block % 2 == 1;
        self.shed == other
    }

    /// One boundary's time (`None` for a request's first), whether its token began with a draft ahead and the
    /// context; a verdict, whether it switched and its console line.
    fn observe(&mut self, ms: Option<f64>, ahead: bool, context: usize) -> Option<(bool, String)> {
        if self.verdict_context > 0 && context.abs_diff(self.verdict_context) > self.hold as usize {
            self.rearm();
        }
        self.tokens += 1;
        if self.phase == ShedPhase::Hold {
            self.probe_context = context;
            if !self.shed {
                // Behind for most of its last tokens, the drafter prefetches nothing in time: no probe is needed.
                push(&mut self.ahead, ahead);
                let behind = self.ahead.iter().filter(|a| !**a).count();
                if behind > Self::RECENT / 2 {
                    let why = format!("{behind} of the last {} tokens began with no draft ahead", self.ahead.len());
                    return Some(self.settle(true, why));
                }
            }
            if let Some(ms) = ms {
                push(&mut self.recent, ms);
            }
            if self.recent.len() < Self::RECENT || self.tokens < self.hold.max(Self::HOLD_FIRST) {
                return None;
            }
            let recent: Vec<f64> = self.recent.drain(..).collect();
            let means: Vec<f64> =
                recent.chunks(Self::MEASURED).map(|c| c.iter().sum::<f64>() / c.len() as f64).collect();
            let tail = &recent[Self::RECENT - Self::MEASURED..];
            let (sum, sumsq) = tail.iter().fold((0.0, 0.0), |(s, q), &x| (s + x, q + x * x));
            (self.current, self.block_sd) = (Block::of(tail.len() as u32, sum, sumsq), sd(&means));
            (self.phase, self.tokens, self.block, self.other) = (ShedPhase::Probe, 0, 1, None);
            (self.measured, self.sum, self.sumsq) = (0, 0.0, 0.0);
            self.diffs.clear();
            return None;
        }
        if let Some(ms) = ms.filter(|_| self.tokens > Self::SKIP) {
            (self.measured, self.sum, self.sumsq) = (self.measured + 1, self.sum + ms, self.sumsq + ms * ms);
        }
        let rates = |off: f64, on: f64| format!("decode {:.1} tok/s without it, {:.1} with it", 1e3 / off, 1e3 / on);
        // Half a block without the drafter already slower than the drafted tokens before it, by more than two standard
        // errors: the drafter pays, and the rest of the probe would spend its gain.
        if self.block % 2 == 1 && !self.shed && self.tokens == Self::SKIP + Self::MEASURED as u32 / 2 {
            if let (Some(on), Some(off)) = (self.current, Block::of(self.measured, self.sum, self.sumsq)) {
                let se = (off.se2() + on.se2()).sqrt().max(self.block_sd * 3f64.sqrt());
                if off.mean - on.mean > 2.0 * se {
                    return Some(self.settle(false, rates(off.mean, on.mean)));
                }
            }
        }
        if self.tokens < Self::BLOCK {
            return None;
        }
        let block = Block::of(self.measured, self.sum, self.sumsq);
        (self.tokens, self.measured, self.sum, self.sumsq) = (0, 0, 0.0, 0.0);
        self.block += 1;
        if self.block % 2 == 0 {
            // The other state's block ended: judged once the current state's block after it has.
            self.other = block;
            return None;
        }
        if let (Some(a0), Some(b), Some(a1)) = (self.current, self.other.take(), block) {
            // The three blocks' token spread, or the hold's spread block to block, whichever is the larger.
            let tokens = (b.se2() + (a0.se2() + a1.se2()) / 4.0).sqrt();
            self.diffs.push((b.mean - (a0.mean + a1.mean) / 2.0, tokens.max(self.block_sd * 1.5f64.sqrt())));
        }
        self.current = block;
        let n = self.diffs.len();
        let (m, se) = verdict_noise(&self.diffs);
        if n < Self::PAIRS_MAX && !(n >= 1 && m.abs() > 3.0 * se) {
            return None;
        }
        // The drafter must pay: it comes on, and stays on, only where decode is faster with it by more than two
        // standard errors, the measurement's own noise; `m` is the other state's excess over the current one's.
        let switch = if self.shed { m + 2.0 * se < 0.0 } else { m - 2.0 * se <= 0.0 };
        let cur = self.current.map_or(0.0, |b| b.mean);
        let (off, on) = if self.shed { (cur, cur + m) } else { (cur + m, cur) };
        Some(self.settle(switch, rates(off, on)))
    }

    /// A verdict at the hold's end: the next hold, doubled by a standing one; its line names the state and the why.
    fn settle(&mut self, switch: bool, why: String) -> (bool, String) {
        self.hold = if switch || self.hold < Self::HOLD { Self::HOLD } else { (2 * self.hold).min(Self::HOLD_MAX) };
        self.verdict_context = self.probe_context;
        self.ahead.clear();
        (self.phase, self.tokens) = (ShedPhase::Hold, 0);
        super::DRAFT.verdicts.fetch_add(1, Ordering::Relaxed);
        super::DRAFT.switches.fetch_add(u64::from(switch), Ordering::Relaxed);
        self.shed ^= switch;
        let (state, stands) = (if self.shed { "off" } else { "on" }, if switch { "" } else { " stands" });
        (switch, format!("{state}{stands} - {why}"))
    }

    /// The regime may have moved: the next probe comes soon.
    fn rearm(&mut self) {
        self.hold = self.hold.min(Self::HOLD_FIRST);
    }
}

/// The hold's reference holds the last [`Shed::RECENT`] measured tokens.
fn push<T>(ring: &mut std::collections::VecDeque<T>, x: T) {
    if ring.len() == Shed::RECENT {
        ring.pop_front();
    }
    ring.push_back(x);
}

/// The sample standard deviation; 0 below two values.
fn sd(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64).sqrt()
}

/// The pairs' mean difference and its error: the larger of the pairs' own errors combined and their spread.
fn verdict_noise(diffs: &[(f64, f64)]) -> (f64, f64) {
    let n = diffs.len().max(1) as f64;
    let ds: Vec<f64> = diffs.iter().map(|d| d.0).collect();
    let within = diffs.iter().map(|d| d.1 * d.1).sum::<f64>().sqrt() / n;
    (ds.iter().sum::<f64>() / n, within.max(sd(&ds) / n.sqrt()))
}

/// A drafted position; `sig` is eval-only, `dist` the chain-precision depth.
struct Drafted {
    token: u32,
    hints: Vec<Vec<u16>>,
    sig: oscar_models::moe::RunnerUps,
    dist: u32,
}

/// The engine thread owns all window alignment, the mirror of drafted positions past its own tokens.
#[derive(Default)]
struct OverlapState {
    seq_id: usize,
    last_len: usize,
    mirror: std::collections::VecDeque<Drafted>,
    sessions_inflight: usize,
    needs_grounding: bool,
}

impl OverlapState {
    /// So no stale draft runs and no stale hint is pushed.
    fn cancel(&mut self, drafter: &thread::OverlapDrafter) {
        drafter.epoch.fetch_add(1, Ordering::AcqRel);
        self.mirror.clear();
        self.sessions_inflight = 0;
    }
}

/// True when it proves the lineage diverged: a late result whose token mismatches the engine's.
fn absorb_draft_result(
    st: &mut OverlapState, r: thread::DraftResult, cur_epoch: u64, toks: &[u32], toks_len: usize,
) -> bool {
    // A cancelled session's result retires nothing: the cancel let its count go, so it would end the one in flight.
    if r.epoch != cur_epoch {
        return false;
    }
    if r.last {
        st.sessions_inflight = st.sessions_inflight.saturating_sub(1);
    }
    if r.token == u32::MAX {
        // An idle: a torn session's recurrent state is invalid, so a snapshot is owed, not left to an empty mirror.
        st.needs_grounding = true;
        return false;
    }
    if r.pos < toks_len {
        super::DRAFT.late.fetch_add(1, Ordering::Relaxed);
        let mismatch = toks.get(r.pos) != Some(&r.token);
        // A late result for the engine's last position scores the attainable precision; depth-1 hints land only here.
        if r.pos + 1 == toks_len && !r.hints.is_empty() {
            crate::engine::eval::hooks().score_hints(r.dist, !mismatch, &r.hints, &r.sig.extras);
        }
        return mismatch;
    }
    // A re-draft replaces, the next position extends, a gap is dropped.
    let i = r.pos - toks_len;
    let d = Drafted { token: r.token, hints: r.hints, sig: r.sig, dist: r.dist };
    if i < st.mirror.len() {
        st.mirror[i] = d;
    } else if i == st.mirror.len() {
        st.mirror.push_back(d);
    }
    false
}

static DRAFT_STATE: std::sync::Mutex<Option<OverlapState>> = std::sync::Mutex::new(None);
static DRAFTER: OnceLock<thread::OverlapDrafter> = OnceLock::new();

/// In the load before the plan is solved, so its recorded width precedes every forward and its region is planned.
pub(crate) fn spawn_once() -> &'static thread::OverlapDrafter {
    DRAFTER.get_or_init(|| {
        let model = thread::draft_model()
            .unwrap_or_else(|| panic!("OSCAR_DRAFT set, model has no draft path - set OSCAR_DRAFT=none"));
        match thread::OverlapDrafter::spawn(model) {
            Ok(d) => d,
            Err(e) => panic!("drafter cannot run on this model: {e} - set OSCAR_DRAFT=none"),
        }
    })
}

pub fn in_session() -> bool {
    DRAFTER.get().is_some_and(|d| d.busy.load(Ordering::SeqCst))
}

/// The longest a cancelled session may take to end: its forward in hand.
const QUIESCE_CAP: std::time::Duration = std::time::Duration::from_secs(5);

/// A phase change's clean-up: the session cancelled and waited out, so no draft reads a pool slot while memory moves.
pub fn quiesce() {
    let Some(drafter) = DRAFTER.get() else { return };
    match DRAFT_STATE.lock().unwrap().as_mut() {
        Some(st) => st.cancel(drafter),
        None => {
            drafter.epoch.fetch_add(1, Ordering::AcqRel);
        }
    }
    let began = std::time::Instant::now();
    while drafter.busy.load(Ordering::SeqCst) {
        assert!(began.elapsed() < QUIESCE_CAP, "a cancelled drafter session did not end in {QUIESCE_CAP:?}");
        std::thread::sleep(std::time::Duration::from_micros(100));
    }
}

/// The drafting round before the step, which is the pipeline's own, blind to drafter state.
pub(super) fn draft(
    this: &dyn Pipeline, input_seqs: &mut [&mut Sequence], backend_metadata: &CacheBackendMetadata,
) -> Result<()> {
    use thread::SessionMsg;

    // Every build runs the paged scheduler: the drafter reads its blocks.
    let CacheBackendMetadata::PagedAttention { metadata: pm } = backend_metadata else {
        candle_core::bail!("drafting round takes paged batches only");
    };

    let k = crate::config::params().draft_k.max(1);
    let seq_id = *input_seqs[0].id();
    let toks_len = input_seqs[0].get_toks().len();
    let dispatcher = crate::engine::offload_dispatcher();

    let drafter = spawn_once();

    // The engine's token count, which the drafter's stale-session abort reads.
    drafter.committed.store(toks_len, Ordering::Release);
    let mut guard = DRAFT_STATE.lock().unwrap();
    let st = guard.get_or_insert_with(OverlapState::default);

    if st.seq_id != seq_id {
        drafter.epoch.fetch_add(1, Ordering::AcqRel);
        *st = OverlapState { seq_id, last_len: toks_len, ..Default::default() };
    }

    // An empty mirror is reconciled by `pos` in the drain.
    let mut diverged = false;
    for &t in input_seqs[0].get_toks().iter().take(toks_len).skip(st.last_len) {
        let Some(d) = st.mirror.pop_front() else { break };
        crate::engine::eval::hooks().score_hints(d.dist, d.token == t, &d.hints, &d.sig.extras);
        if d.token != t {
            diverged = true;
            break;
        }
    }
    st.last_len = toks_len;

    let cur_epoch = drafter.epoch.load(Ordering::Acquire);
    {
        let rx = drafter.rx.lock().unwrap();
        let toks = input_seqs[0].get_toks();
        while let Ok(r) = rx.try_recv() {
            diverged |= absorb_draft_result(st, r, cur_epoch, toks, toks_len);
        }
    }

    if diverged {
        st.cancel(drafter);
        super::DRAFT.diverges.fetch_add(1, Ordering::Relaxed);
    }

    // Recurrent state cannot be row-synced, so a session carries a copy; one hand-off at a time, as the plan books.
    let want_gdn = st.mirror.is_empty() || st.needs_grounding;
    let deferred =
        st.sessions_inflight < SPAWN_DEPTH && st.mirror.len() < k && want_gdn && thread::handoffs_alive() > 0;
    if deferred {
        super::DRAFT.ground_deferred.fetch_add(1, Ordering::Relaxed);
    }
    // Row i is position i only without a prefix-restore offset: a step with one is declined, a hint not worth a stop.
    let offset = input_seqs[0].token_offset() != 0;
    if offset {
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| {
            tracing::info!(target: "oscar::drafter",
                "draft sessions declined: the decode step carries a prefix-restore offset"
            )
        });
    }
    if st.sessions_inflight < SPAWN_DEPTH && st.mirror.len() < k && !deferred && !offset {
        let handoff_t0 = std::time::Instant::now();
        let n = toks_len - 1;
        let kv = kv_view(this, pm.block_size);
        let handed = handoff(this, pm, seq_id, n, want_gdn)?;
        let counter = if handed.is_some() { &super::DRAFT.handoff_pooled } else { &super::DRAFT.skip_handoff };
        counter.fetch_add(1, Ordering::Relaxed);
        super::DRAFT.handoff_us.fetch_add(handoff_t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        if let Some((table, engine_gdn)) = handed {
            let had_gdn = engine_gdn.is_some();
            let msg = SessionMsg {
                sent_at: std::time::Instant::now(),
                epoch: drafter.epoch.load(Ordering::Acquire),
                seq: seq_id,
                ctx: input_seqs[0].get_toks().to_vec(),
                pending: st.mirror.iter().map(|d| d.token).collect(),
                kv,
                table,
                engine_gdn,
                target_k: k,
                sampler: input_seqs[0].sampler(),
            };
            if drafter.tx.lock().unwrap().send(msg).is_ok() {
                st.sessions_inflight += 1;
                super::DRAFT.sessions.fetch_add(1, Ordering::Relaxed);
                // Cleared only when a snapshot went out: a session without state forgives no hybrid's debt.
                if had_gdn {
                    st.needs_grounding = false;
                    super::DRAFT.groundings.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    // A diverged boundary's flag drops the dispatcher's retained window.
    if let Some(disp) = dispatcher {
        disp.push_draft_window(st.mirror.iter().map(|d| d.hints.clone()).collect(), diverged);
    }
    super::DRAFT.depth_sum.fetch_add(st.mirror.len() as u64, Ordering::Relaxed);
    Ok(())
}

/// Taken once and kept, its tables fixed over the pool's index space; a pool a draft cannot write stops oscar here.
fn kv_view(this: &dyn Pipeline, block_size: usize) -> Arc<thread::KvView> {
    use mistralrs_core::kv_cache::HybridLayerCache;
    static VIEW: OnceLock<Arc<thread::KvView>> = OnceLock::new();
    let view = || -> std::result::Result<thread::KvView, &'static str> {
        let md = this.get_metadata();
        let engine = md.cache_engine.as_ref().ok_or("the pipeline has no paged cache engine")?;
        if mistralrs_core::paged_attention::cache_dtype() != md.activation_dtype {
            return Err("the KV cache is not in the dtype the draft forward writes");
        }
        let pair = |(k, v): &(Tensor, Tensor)| {
            (k.dtype() == candle_core::DType::I64 && !k.dims1().is_ok_and(|n| n <= 1)).then(|| (k.clone(), v.clone()))
        };
        let tables = engine.get_kv_cache();
        let pools = match this.cache() {
            EitherCache::Normal(_) => tables.iter().map(|t| pair(t).map(Some)).collect::<Option<Vec<_>>>(),
            EitherCache::Hybrid(h) => h
                .lock()
                .unwrap()
                .caches
                .iter()
                .zip(tables.iter())
                .map(|(c, t)| match c {
                    HybridLayerCache::Attention(_) => pair(t).map(Some),
                    HybridLayerCache::Recurrent(_) => Some(None),
                })
                .collect(),
            _ => return Err("the drafter reads a Normal or a Hybrid cache"),
        };
        let pools = pools.ok_or("a KV layer's table is not a block table")?;
        // The reserved tail: the top ids at launch, charged by the plan and withheld from every sequence's table.
        let cfg = md
            .cache_config
            .as_ref()
            .filter(|c| c.drafter_reserved_blocks > 0)
            .ok_or("no tail is reserved for the drafter")?;
        let reserved = (cfg.num_gpu_blocks - cfg.drafter_reserved_blocks) as u32..cfg.num_gpu_blocks as u32;
        Ok(thread::KvView { pools, memory: Arc::clone(engine.memory()), block_size, reserved })
    };
    Arc::clone(
        VIEW.get_or_init(|| Arc::new(view().unwrap_or_else(|why| panic!("[drafter] {why}: set OSCAR_DRAFT=none")))),
    )
}

/// The engine's blocks and, grounding a hybrid, a state copy, which alone pays the cross-queue sync; `None` declines.
#[allow(clippy::type_complexity)]
fn handoff(
    this: &dyn Pipeline, pm: &PagedAttentionMeta, seq_id: usize, n: usize, want_gdn: bool,
) -> Result<Option<(Vec<u32>, Option<thread::GdnHandoff>)>> {
    let nblk = n.div_ceil(pm.block_size);
    let Some(table) = pm.kv_cache_manager.try_lock().ok().and_then(|m| {
        m.get_block_ids(seq_id)
            .filter(|ids| n > 0 && ids.len() >= nblk)
            .map(|ids| ids[..nblk].iter().map(|&b| b as u32).collect())
    }) else {
        return Ok(None);
    };
    let (EitherCache::Hybrid(h), true) = (this.cache(), want_gdn) else { return Ok(Some((table, None))) };
    let state = {
        let h = h.lock().unwrap();
        // One copy of the state, a pool object the drafter reads once.
        let Some(&[slot]) = h.state_indices_host() else { return Ok(None) };
        thread::GdnHandoff::new(h.snapshot_recurrent_state(slot as usize)?, h.recurrent_layers())
    };
    // The copy rides the engine queue; the drafter's queue adopts it.
    let sync_t0 = std::time::Instant::now();
    this.device().synchronize()?;
    super::DRAFT.handoff_sync_us.fetch_add(sync_t0.elapsed().as_micros() as u64, Ordering::Relaxed);
    Ok(Some((table, Some(state))))
}

#[cfg(test)]
mod tests {
    use super::thread::DraftResult;
    use super::*;

    /// A probe's tokens: `pairs` pairs of an other-state block and a current-state one.
    fn probe(pairs: usize) -> u32 {
        2 * pairs as u32 * Shed::BLOCK
    }

    /// `n` boundaries on a row with every expert resident, each the state's time plus `drift(i)` and seeded `±noise`;
    /// the switches.
    fn run_with(
        shed: &mut Shed, n: u32, drafting_ms: f64, shed_ms: f64, noise: f64, drift: impl Fn(u32) -> f64,
    ) -> Vec<bool> {
        let mut turns = Vec::new();
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        for i in 0..n {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let wobble = noise * ((x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0);
            let ms = if shed.drafting() { drafting_ms } else { shed_ms } + wobble + drift(i);
            if shed.observe(Some(ms), true, 1000).is_some_and(|(switched, _)| switched) {
                turns.push(shed.shed);
            }
        }
        turns
    }

    fn run(shed: &mut Shed, n: u32, drafting_ms: f64, shed_ms: f64, noise: f64) -> Vec<bool> {
        run_with(shed, n, drafting_ms, shed_ms, noise, |_| 0.0)
    }

    /// Its memory goes with it for the hold and is asked again through each probe, which begins at the hold's end. A
    /// standing verdict doubles the hold to its cap; a regime that moved resets it.
    #[test]
    fn a_drafter_is_shed_while_decode_is_faster_without_it() {
        let mut shed = Shed::default();
        assert!(shed.drafting(), "the drafter starts on");
        // A clear difference ends the probe at its first pair: one block without the drafter, one back with it.
        assert!(run(&mut shed, Shed::HOLD_FIRST + probe(1) - 1, 45.0, 38.0, 1.0).is_empty());
        assert_eq!(run(&mut shed, 1, 45.0, 38.0, 1.0), [true]);
        assert!(!shed.drafting(), "shed for the hold");
        // A hold later it probes the drafter again, the memory asked ahead of the other state's block, and stays shed.
        assert!(run(&mut shed, Shed::HOLD - 1, 45.0, 38.0, 1.0).is_empty());
        assert_eq!(shed.phase, ShedPhase::Hold, "no probe inside the hold");
        assert!(run(&mut shed, 1, 45.0, 38.0, 1.0).is_empty());
        assert_eq!((shed.phase, shed.drafting()), (ShedPhase::Probe, true), "the probe begins in the other state");
        assert!(run(&mut shed, probe(1), 45.0, 38.0, 1.0).is_empty());
        assert!(!shed.drafting(), "the verdict stands: given up again");
        // The regime moves; the next probe, a doubled hold on, finds it.
        assert_eq!(run(&mut shed, 2 * Shed::HOLD + probe(1), 30.0, 38.0, 1.0), [false]);
        assert!(shed.drafting(), "the drafter pays again: kept");
        let mut shed = Shed::default();
        let holds: Vec<u32> = (0..7)
            .map(|_| {
                let n = shed.hold.max(Shed::HOLD_FIRST) + probe(1);
                assert!(run(&mut shed, n, 30.0, 38.0, 1.0).is_empty(), "the drafter pays: it stays");
                shed.hold
            })
            .collect();
        assert_eq!(holds, [2048, 4096, 8192, 16384, 32768, 32768, 32768]);
        shed.rearm();
        assert_eq!(shed.hold, Shed::HOLD_FIRST);
        // Half a block without the drafter clearly slower ends the probe there: the drafter's gain is not spent on it.
        let mut shed = Shed::default();
        assert!(run(&mut shed, Shed::HOLD_FIRST + Shed::SKIP + Shed::MEASURED as u32 / 2, 30.0, 38.0, 1.0).is_empty());
        assert_eq!((shed.phase, shed.hold, shed.drafting()), (ShedPhase::Hold, Shed::HOLD, true));
    }

    /// Within the hold, a long reply and the next request probe nothing; a context moved by more than the hold
    /// measures again.
    #[test]
    fn a_context_moved_past_the_hold_measures_again() {
        for (hold, near, far) in [(Shed::HOLD, 2_500, 3_100), (8 * Shed::HOLD, 12_000, 17_500)] {
            let mut shed = Shed { hold, verdict_context: 1000, ..Default::default() };
            shed.observe(Some(40.0), true, near);
            assert_eq!(shed.hold, hold, "a context within the hold of the verdict's keeps the hold");
            shed.observe(Some(40.0), true, far);
            assert_eq!(shed.hold, Shed::HOLD_FIRST);
        }
        let mut shed = Shed { hold: 8 * Shed::HOLD, verdict_context: 20_000, ..Default::default() };
        shed.observe(Some(40.0), true, 1_000);
        assert_eq!(shed.hold, Shed::HOLD_FIRST, "a request far below the verdict's context measures again");
        // A reply to an 817-token prompt: the probe begins at 945, and the next request's 818 is within the hold.
        let mut shed = Shed::default();
        let mut context = 817;
        for _ in 0..Shed::HOLD_FIRST + probe(Shed::PAIRS_MAX) {
            context += 1;
            shed.observe(Some(40.0), true, context);
        }
        assert_eq!((shed.phase, shed.verdict_context), (ShedPhase::Hold, 817 + Shed::HOLD_FIRST as usize));
        shed.observe(Some(40.0), true, 818);
        assert_eq!(shed.hold, Shed::HOLD, "the next request keeps the hold");
    }

    /// A drafter most of whose last tokens began with no draft ahead is shed with no probe, and probed a hold on as
    /// any shed drafter is; one ahead for half its tokens stays for the probe.
    #[test]
    fn a_drafter_whose_drafts_are_not_ahead_is_shed_unmeasured() {
        let feed = |shed: &mut Shed, n: u32, ahead: bool| -> Vec<(bool, String)> {
            (0..n).filter_map(|_| shed.observe(Some(40.0), ahead, 1000)).collect()
        };
        let mut shed = Shed::default();
        assert!(feed(&mut shed, Shed::RECENT as u32 / 2, false).is_empty());
        let line = (true, "off - 65 of the last 65 tokens began with no draft ahead".to_string());
        assert_eq!(feed(&mut shed, 1, false), [line]);
        assert_eq!((shed.phase, shed.hold, shed.drafting()), (ShedPhase::Hold, Shed::HOLD, false));
        assert!(feed(&mut shed, Shed::HOLD, false).is_empty());
        assert_eq!((shed.phase, shed.drafting()), (ShedPhase::Probe, true), "shed, it is judged by the probe alone");
        let mut shed = Shed::default();
        assert!((0..Shed::HOLD_FIRST - 1).all(|i| shed.observe(Some(40.0), i % 2 == 0, 1000).is_none()));
        assert_eq!((shed.phase, shed.drafting()), (ShedPhase::Hold, true));
    }

    /// A slow stretch in the hold's tail is the hold's own spread, not the other state's gain, and a difference inside
    /// the tokens' own spread is not decided at one pair; a drafter not faster beyond the noise is shed, and a request's
    /// first boundary measures nothing; the text's drift, a ramp ten times the states' difference, cancels in a pair.
    #[test]
    fn a_pair_decides_nothing_under_the_holds_own_spread_noise_or_drift() {
        let mut shed = Shed { shed: true, ..Default::default() };
        let slow_tail = |i: u32| if (Shed::HOLD_FIRST - 32..Shed::HOLD_FIRST).contains(&i) { 13.0 } else { 0.0 };
        assert!(run_with(&mut shed, Shed::HOLD_FIRST + probe(1), 35.0, 36.0, 1.0, slow_tail).is_empty());
        assert_eq!((shed.phase, shed.diffs.len()), (ShedPhase::Probe, 1), "the slow tail is the hold's spread");
        assert!(shed.block_sd > 5.0, "{}", shed.block_sd);
        let mut shed = Shed::default();
        assert!(run(&mut shed, Shed::HOLD_FIRST + probe(1), 40.0, 39.5, 3.0).is_empty());
        assert_eq!((shed.phase, shed.diffs.len()), (ShedPhase::Probe, 1), "one pair in, undecided");
        run(&mut shed, probe(Shed::PAIRS_MAX), 40.0, 39.5, 3.0);
        assert_eq!(shed.phase, ShedPhase::Hold, "decided by the cap");
        // A drafter that does not pay beyond the noise is shed, and does not come back on the same evidence.
        let mut shed = Shed::default();
        assert_eq!(run(&mut shed, 4 * (Shed::HOLD + probe(Shed::PAIRS_MAX)), 40.0, 39.9, 6.0), [true]);
        assert!(!shed.drafting());
        let mut shed = Shed::default();
        let quiet = (0..Shed::HOLD_FIRST + probe(1)).all(|_| shed.observe(None, true, 1000).is_none());
        assert_eq!((quiet, shed.phase, shed.diffs.len()), (true, ShedPhase::Hold, 0), "no boundary measured: no probe");
        let ramp = |i: u32| f64::from(i % 1024) * 0.03;
        let first = Shed::HOLD_FIRST + probe(Shed::PAIRS_MAX);
        let found = run_with(&mut Shed::default(), first, 40.0, 38.0, 1.0, ramp);
        assert_eq!(found, [true], "a 5% cost is found under the drift");
        assert_eq!(
            run_with(&mut Shed::default(), first, 40.0, 40.0, 1.0, ramp),
            [true],
            "no gain: the drafter is shed"
        );
        let mut off = Shed { shed: true, ..Default::default() };
        assert!(run_with(&mut off, first, 40.0, 40.0, 1.0, ramp).is_empty(), "no difference is invented");
        // An idle of the live epoch retires its session and owes a snapshot; a cancelled session's retire nothing.
        for (r, inflight, grounding) in [
            (DraftResult::idle(7), 0, true),
            (DraftResult { token: 5, pos: 3, ..DraftResult::idle(6) }, 1, false),
            (DraftResult::idle(6), 1, false),
        ] {
            let mut st = OverlapState { sessions_inflight: 1, ..Default::default() };
            assert!(!absorb_draft_result(&mut st, r, 7, &[1, 2, 3], 3));
            assert_eq!((st.sessions_inflight, st.needs_grounding, st.mirror.len()), (inflight, grounding, 0));
        }
    }
}
