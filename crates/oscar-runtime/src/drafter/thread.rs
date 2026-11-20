// The draft loop beside the engine's exact steps, which never see its state: drafts write only reserved tail blocks.
// Recurrent state is a copy of the engine's, never a handle, as the kernels mutate it in place.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// One session's send → first-result latency, into the turnaround counters.
pub(super) fn turn_first_result(sent_at: std::time::Instant, fwd1_us: u64) {
    let total = sent_at.elapsed().as_micros() as u64;
    super::DRAFT.turn_first.fetch_add(1, Ordering::Relaxed);
    super::DRAFT.turn_fwd1_us.fetch_add(fwd1_us, Ordering::Relaxed);
    super::DRAFT.turn_total_us.fetch_add(total, Ordering::Relaxed);
    super::DRAFT.turn_total_max_us.fetch_max(total, Ordering::Relaxed);
    let prev = super::DRAFT.turn_ewma_us.load(Ordering::Relaxed);
    let ewma = if prev == 0 { total } else { (prev * 7 + total) / 8 };
    super::DRAFT.turn_ewma_us.store(ewma, Ordering::Relaxed);
}

use anyhow::Context as _;
use candle_core::{DType, Device, Result, Tensor};
use rand_isaac::Isaac64Rng;

use mistralrs_core::sampler::Sampler;
use oscar_models::draft::DraftCache;
use oscar_models::state_slots::RecurrentState;
use oscar_offload::census::{self, Arm};

/// All its state `&self`; the private cache holds only what the block pools cannot, a hybrid's recurrent frontier.
pub type DraftModel = Arc<dyn oscar_models::draft::DraftableModel>;

/// The loaded model, when its architecture drafts at all.
pub(crate) fn draft_model() -> Option<DraftModel> {
    oscar_models::loaded_model()?.draft.clone()
}

/// Engine → drafter: a session extending the surviving `pending`, as re-drafting them would cost window depth.
pub(crate) struct SessionMsg {
    pub sent_at: std::time::Instant,
    pub epoch: u64,
    /// The engine's tokens and the tip, which alone has no engine KV row yet.
    pub ctx: Vec<u32>,
    /// Surviving drafted tokens, their rows already in reserved tail blocks.
    pub pending: Vec<u32>,
    /// A new sequence id clears the tail-block assignments and the frontier.
    pub seq: usize,
    pub kv: Arc<KvView>,
    /// The session's engine blocks.
    pub table: Vec<u32>,
    /// The engine's recurrent state at its rows, held through the session; `Some` only when grounding.
    pub engine_gdn: Option<GdnHandoff>,
    /// Draft until `pending.len()` reaches this window depth.
    pub target_k: usize,
    /// The sequence's sampler; its penalty context derives from `ctx`.
    pub sampler: Arc<Sampler>,
}

/// The engine's KV pool as a draft reads it, in place on the drafter's queue: `(K, V)` tables, `None` where recurrent.
pub(crate) struct KvView {
    pub pools: Vec<Option<(Tensor, Tensor)>>,
    pub memory: std::sync::Arc<dyn mistralrs_core::paged_attention::KvBlockMemory>,
    pub block_size: usize,
    /// The only blocks a draft may write.
    pub reserved: std::ops::Range<u32>,
}

pub(crate) struct DraftResult {
    pub epoch: u64,
    pub token: u32,
    /// Absolute, so the engine places a late result too.
    pub pos: usize,
    /// Per layer, the routed expert ids: the hints for the engine's step at this position.
    pub hints: Vec<Vec<u16>>,
    /// Eval-only; default when unarmed.
    pub sig: oscar_models::moe::RunnerUps,
    /// The session's last result: the drafter is idle.
    pub last: bool,
    /// `pos` less the engine's rows: 1 the first past them, 0 on an idle.
    pub dist: u32,
}

impl DraftResult {
    /// A stale or torn session's end, its `token` the `u32::MAX` sentinel.
    pub(crate) fn idle(epoch: u64) -> Self {
        Self {
            epoch,
            token: u32::MAX,
            pos: usize::MAX,
            hints: Vec::new(),
            sig: Default::default(),
            last: true,
            dist: 0,
        }
    }
}

pub(crate) struct OverlapDrafter {
    pub tx: Mutex<Sender<SessionMsg>>,
    pub rx: Mutex<Receiver<DraftResult>>,
    /// Bumped by the engine to cancel the session in flight.
    pub epoch: Arc<AtomicU64>,
    /// The engine's token count: a session behind it aborts, as its drafts could only arrive late.
    pub committed: Arc<AtomicUsize>,
    /// Raised while a session runs, for a crossing to wait out.
    pub busy: Arc<AtomicBool>,
}

/// The thread's own.
struct Private {
    cache: DraftCache,
    /// A hybrid's next input position, the forwards its state has absorbed; `None` awaits a snapshot.
    frontier: Option<usize>,
}

impl OverlapDrafter {
    /// Returns once the thread's recorded width has run: it arms the process-wide census, which no forward may overlap.
    pub fn spawn(model: DraftModel) -> Result<Self> {
        let (tx, session_rx) = std::sync::mpsc::channel::<SessionMsg>();
        let (result_tx, rx) = std::sync::mpsc::channel::<DraftResult>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<std::result::Result<(), String>>();
        let epoch = Arc::new(AtomicU64::new(0));
        let epoch_t = epoch.clone();
        let committed = Arc::new(AtomicUsize::new(0));
        let committed_t = committed.clone();
        let busy = Arc::new(AtomicBool::new(false));
        let busy_t = busy.clone();
        std::thread::Builder::new()
            .name("oscar-overlap-drafter".into())
            .spawn(move || drafter_main(model, session_rx, result_tx, epoch_t, committed_t, busy_t, ready_tx))
            .map_err(candle_core::Error::msg)?;
        // The thread is ready, or says why it cannot draft.
        ready_rx.recv().map_err(candle_core::Error::msg)?.map_err(candle_core::Error::msg)?;
        Ok(Self { tx: Mutex::new(tx), rx: Mutex::new(rx), epoch, committed, busy })
    }
}

/// Encoders a draft command buffer: the GPU starts a waiting engine buffer between draft buffers this short, not
/// between buffers of ten or more, where the engine's wait is a whole draft forward in half the runs.
const DRAFT_COMPUTE_PER_BUFFER: usize = 3;

fn drafter_main(
    model: DraftModel, sessions: Receiver<SessionMsg>, results: Sender<DraftResult>, epoch: Arc<AtomicU64>,
    committed: Arc<AtomicUsize>, busy: Arc<AtomicBool>, ready: Sender<std::result::Result<(), String>>,
) {
    // The private stream first: everything the thread encodes or syncs rides it.
    if let Device::Metal(md) = model.device() {
        if let Err(e) = md.install_thread_stream(Some(DRAFT_COMPUTE_PER_BUFFER)) {
            let _ = ready.send(Err(format!("its stream could not be installed ({e})")));
            return;
        }
        crate::gpu::probe::mark_draft_thread();
    }
    if let Some(pd) = crate::engine::offload_dispatcher() {
        pd.set_thread_lane(crate::pool::Lane::Drafter);
    }
    // The recorded width before the cache exists: the region is re-taken at what the step and the cache hold together.
    let cache = record_width(&model)
        .map_err(|e| format!("{e:#}"))
        .and_then(|()| model.make_private_draft_cache().map_err(|e| e.to_string()));
    let mut private = match cache {
        Ok(cache) => Private { cache, frontier: None },
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    let _ = ready.send(Ok(()));
    let rng = Arc::new(std::sync::Mutex::new(<Isaac64Rng as rand::SeedableRng>::seed_from_u64(0)));
    // Survives sessions so pending rows stay addressable; reset per sequence.
    let mut tail = DraftTailBlocks::default();
    let mut cur_seq: Option<usize> = None;

    while let Ok(mut msg) = sessions.recv() {
        // An autorelease pool per session, drained on every exit path.
        let _pool = crate::gpu::AutoreleaseScope::push();
        // Coalesce: only the newest queued session matters.
        loop {
            match sessions.try_recv() {
                Ok(newer) => msg = newer,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }
        if cur_seq != Some(msg.seq) {
            cur_seq = Some(msg.seq);
            tail.clear();
            private.frontier = None;
        }
        // Busy before the epoch check: a clean-up bumps the epoch, then waits on busy, so it misses no session.
        busy.store(true, Ordering::SeqCst);
        if msg.epoch != epoch.load(Ordering::SeqCst) {
            busy.store(false, Ordering::SeqCst);
            continue;
        }
        super::DRAFT.turn_runs.fetch_add(1, Ordering::Relaxed);
        super::DRAFT.turn_wake_us.fetch_add(msg.sent_at.elapsed().as_micros() as u64, Ordering::Relaxed);
        let Private { cache, frontier } = &mut private;
        let session_result = run_session(&model, cache, &msg, &results, &epoch, &committed, &rng, &mut tail, frontier);
        // A failed session is a defect, never a state to idle through.
        if let Err(e) = session_result {
            panic!("drafter session failed: {e}");
        }
        busy.store(false, Ordering::SeqCst);
    }
}

#[allow(clippy::too_many_arguments)]
fn run_session(
    model: &DraftModel, cache: &mut DraftCache, msg: &SessionMsg, results: &Sender<DraftResult>, epoch: &AtomicU64,
    committed: &AtomicUsize, rng: &Arc<std::sync::Mutex<Isaac64Rng>>, tail: &mut DraftTailBlocks,
    state_count: &mut Option<usize>,
) -> Result<()> {
    let n = msg.ctx.len() - 1;

    // On a hybrid, survivors are usable only with the state at their frontier.
    let frontier = n + msg.pending.len();
    let survivors: &[u32] = if cache.is_empty() {
        &msg.pending
    } else if msg.engine_gdn.is_some() {
        // Grounding: survivors are re-drafted from the tip.
        &[]
    } else if *state_count == Some(frontier) {
        &msg.pending
    } else {
        // Past the private state with no snapshot: a hand-off defect.
        candle_core::bail!("gdn frontier desync (state at {:?}, session wants {frontier})", *state_count)
    };

    let ground_start = Instant::now();
    if let Some(snap) = msg.engine_gdn.as_ref() {
        adopt_gdn_states(cache, snap, n)?;
        *state_count = Some(n);
    }
    let view = assemble_view(&msg.kv, &msg.table, n, n + msg.target_k, tail)?;
    super::DRAFT.ground_us.fetch_add(ground_start.elapsed().as_micros() as u64, Ordering::Relaxed);

    let mut toks: Vec<u32> = Vec::with_capacity(msg.ctx.len() + msg.target_k);
    toks.extend_from_slice(&msg.ctx);
    toks.extend_from_slice(survivors);
    let mut pending_len = survivors.len();
    let mut first_sent = false;

    let dispatcher = crate::engine::offload_dispatcher();

    while pending_len < msg.target_k {
        if epoch.load(Ordering::Acquire) != msg.epoch {
            return Ok(()); // Cancelled: no `last`.
        }
        // Input = newest token (no KV row yet), at absolute position rows.
        let rows = n + pending_len;
        let pos = toks.len();
        // The engine past it, this draft would arrive late: idle, and the empty mirror grounds the next session.
        if pos < committed.load(Ordering::Acquire) {
            super::DRAFT.abort_stale.fetch_add(1, Ordering::Relaxed);
            let _ = results.send(DraftResult::idle(msg.epoch));
            return Ok(());
        }
        let fwd_start = Instant::now();
        let (logits, hints, sig) = {
            let tip = *toks.last().unwrap();
            let logits = model.draft_forward(tip, rows, &view, cache)?;
            let enc_us = fwd_start.elapsed().as_micros() as u64;
            super::DRAFT.enc_us.fetch_add(enc_us, Ordering::Relaxed);
            let (hints, sig) = dispatcher.map(|d| d.take_draft_routings()).unwrap_or_default();
            (logits, hints, sig)
        };
        // The take synced the stream: the forward's fence windows close.
        if let Some(d) = dispatcher {
            d.end_draft_forward();
        }
        // A torn token read a slot overwritten mid-matvec: dropped before sampling, its rows rewritten before any read.
        let torn = dispatcher.map(|d| d.draft_token_was_torn()).unwrap_or(false);
        if torn {
            super::DRAFT.torn.fetch_add(1, Ordering::Relaxed);
            if !cache.is_empty() {
                *state_count = None;
            }
            let _ = results.send(DraftResult::idle(msg.epoch));
            return Ok(());
        }
        let sampled_token = {
            let sel_logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            msg.sampler.sample(sel_logits, &toks, false, rng.clone(), false, false)?.token
        };
        super::DRAFT.fwds.fetch_add(1, Ordering::Relaxed);
        super::DRAFT.fwd_us.fetch_add(fwd_start.elapsed().as_micros() as u64, Ordering::Relaxed);
        if !cache.is_empty() {
            *state_count = Some(rows + 1);
        }
        toks.push(sampled_token);
        pending_len += 1;
        let last = pending_len >= msg.target_k;
        let fwd_us = fwd_start.elapsed().as_micros() as u64;
        if results
            .send(DraftResult { epoch: msg.epoch, token: sampled_token, pos, hints, sig, last, dist: (pos - n) as u32 })
            .is_err()
        {
            return Ok(());
        }
        if !first_sent {
            first_sent = true;
            turn_first_result(msg.sent_at, fwd_us);
        }
    }
    Ok(())
}

/// Reserved-tail assignments, `(block index, drafter block)`.
#[derive(Default)]
struct DraftTailBlocks {
    assigned: Vec<(usize, u32)>,
}

impl DraftTailBlocks {
    fn clear(&mut self) {
        self.assigned.clear();
    }

    /// Blocks for indices `[full, last]`, kept or free reserved ones; those below `full` retire first to recycle.
    fn assign(&mut self, full: usize, last: usize, reserved: &std::ops::Range<u32>) -> Result<Vec<u32>> {
        self.assigned.retain(|(idx, _)| *idx >= full);
        let mut out = Vec::with_capacity(last + 1 - full);
        for idx in full..=last {
            if let Some((_, id)) = self.assigned.iter().find(|(i, _)| *i == idx) {
                out.push(*id);
                continue;
            }
            let Some(id) = reserved.clone().find(|id| !self.assigned.iter().any(|(_, a)| a == id)) else {
                candle_core::bail!(
                    "drafter tail blocks exhausted ({} reserved, {} assigned)",
                    reserved.len(),
                    self.assigned.len()
                );
            };
            self.assigned.push((idx, id));
            out.push(id);
        }
        Ok(out)
    }
}

/// Full engine blocks, then reserved ones, a partial block re-based: a draft row there would race the exact write.
fn assemble_view(
    kv: &KvView, committed: &[u32], n: usize, branch_max: usize, tail: &mut DraftTailBlocks,
) -> Result<oscar_models::draft::DraftPagedView> {
    let bs = kv.block_size;
    let full = n / bs;
    let last = branch_max.saturating_sub(1).max(n) / bs;
    if committed.len() < full {
        candle_core::bail!("paged handoff table covers {} block(s), committed prefix needs {full}", committed.len());
    }
    let mut table = Vec::with_capacity(last + 1);
    table.extend_from_slice(&committed[..full]);
    table.extend(tail.assign(full, last, &kv.reserved)?);
    let part = n - full * bs;
    if part > 0 {
        let (src, dst) = (committed[full] as usize, table[full] as usize);
        rebase_partial_block(&*kv.memory, src, dst, part)?;
    }
    Ok(oscar_models::draft::DraftPagedView { pools: kv.pools.clone(), table, block_size: bs })
}

/// Engine block `src`'s first `part` rows over drafter block `dst`'s, every layer's K and V, on the drafter's queue.
fn rebase_partial_block(
    memory: &dyn mistralrs_core::paged_attention::KvBlockMemory, src: usize, dst: usize, part: usize,
) -> Result<()> {
    let layout = mistralrs_core::paged_attention::kv_layout();
    let es = layout.dtype.size_in_bytes();
    let (src_block, dst_block) = (memory.block(src)?, memory.block(dst)?);
    for (_, rows) in layout.kv_layers() {
        for (dims, offset) in [(&rows.k, rows.k_offset), (&rows.v, rows.v_offset)] {
            let elems: usize = dims.iter().product();
            if elems == 0 {
                continue;
            }
            let token_dim = if dims.len() == 2 { 0 } else { 2 };
            let row = |block: &Tensor| block.narrow(0, offset / es, elems)?.reshape(dims.as_slice());
            let head = row(&src_block)?.narrow(token_dim, 0, part)?.contiguous()?;
            row(&dst_block)?.slice_set(&head, token_dim, 0)?;
        }
    }
    Ok(())
}

static HANDOFFS_ALIVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Hand-offs held now, against the one the plan books.
pub fn handoffs_alive() -> usize {
    HANDOFFS_ALIVE.load(std::sync::atomic::Ordering::Acquire)
}

/// A grounding session's state and its layers' model indices, counted alive until it drops.
pub struct GdnHandoff {
    pub state: mistralrs_core::kv_cache::RecurrentState,
    pub layers: Vec<usize>,
    _alive: HandoffAlive,
}

struct HandoffAlive;

impl Drop for HandoffAlive {
    fn drop(&mut self) {
        HANDOFFS_ALIVE.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

impl GdnHandoff {
    pub fn new(state: mistralrs_core::kv_cache::RecurrentState, layers: Vec<usize>) -> Self {
        HANDOFFS_ALIVE.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Self { state, layers, _alive: HandoffAlive }
    }
}

/// The engine's state at `n` as a run of the drafter's own stream, one blit: in place it would serialise the queues.
pub(crate) fn adopt_gdn_states(layers: &mut DraftCache, snap: &GdnHandoff, n: usize) -> Result<()> {
    if layers.is_empty() {
        return Ok(());
    }
    let pool = crate::gpu::state_pool::installed().ok_or_else(|| {
        candle_core::Error::Msg("the sequence-state pool exists before the drafter's first session".into())
    })?;
    // The previous state goes back first: the stream holds one at a time.
    for state in layers.iter_mut() {
        *state = None;
    }
    let mine = pool.stream_copy(&snap.state)?;
    for (&i, (conv, recurrent)) in snap.layers.iter().zip(mine.layers()) {
        if let Some(state) = layers.get_mut(i) {
            *state = Some(RecurrentState {
                conv_state: conv.clone(),
                recurrent_state: recurrent.clone(),
                seqlen_offset: n,
                emits: Vec::new(),
            });
        }
    }
    Ok(())
}

/// Set by [`record_width`] before the plan is solved.
pub(crate) static STREAM_REGION: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

/// Recorded on the drafter's stream before its first session, so its region is re-taken at what step and cache hold.
fn record_width(model: &DraftModel) -> anyhow::Result<()> {
    // Released with the recorded width, as the trunk's census is.
    let _pool = crate::gpu::AutoreleaseScope::push();
    let device = model.device().clone();
    let Device::Metal(md) = &device else {
        return Ok(());
    };
    let pools = oscar_offload::pools::pools().context("the pools are installed before the drafter")?;
    let queue = md.queue_key();
    let began = Instant::now();
    let (ran, rec) = crate::load::census::recorded(&device, |_| {
        census::note_width(1, Arm::Sealed);
        (record_round(model, &device), ())
    })?;
    ran?;
    let peak = rec.peak(queue, &[(1, Arm::Sealed), (1, Arm::Sealed)]);
    let bytes = peak.high * oscar_offload::segments::PAGE;
    crate::detail!(target: "oscar::census",
        "drafter's width - {} buffers on its stream in {}; stream region {}",
        rec.entries.len(),
        mistralrs_core::live::duration_text(began.elapsed().as_secs_f64()),
        oscar_offload::census::peak_row(&peak),
    );
    let _ = STREAM_REGION.set(bytes as u64);
    pools.plan_stream_region(bytes)?;
    Ok(())
}

/// A draft round of one-token steps ending at the context's end, its session's cache made inside the record.
fn record_round(model: &DraftModel, device: &Device) -> anyhow::Result<()> {
    let rows = 1usize;
    let mut cache = model.make_private_draft_cache()?;
    // Grounded as a session is: a hybrid's draft forward refuses a recurrent state with no snapshot adopted.
    let arch = oscar_models::loaded_model().context("the model is built before the drafter")?;
    let arch = arch.arch.clone();
    if arch.cache().is_hybrid() {
        let mut hybrid = arch.cache().hybrid();
        if let Some(idx) = hybrid.allocate_seq()? {
            let snap = hybrid.snapshot_recurrent_state(idx)?;
            let layers = hybrid.recurrent_layers();
            hybrid.free_seq(idx);
            drop(hybrid);
            adopt_gdn_states(&mut cache, &GdnHandoff::new(snap, layers), rows)?;
        }
    }
    // The view reaches the deepest context admitted: paged attention's scratch grows per split, none at one block.
    let k = crate::config::params().draft_k;
    let block_size = crate::plan::paged_block_size() as usize;
    let arch = oscar_models::loaded_model().context("the model is built before the drafter")?;
    let context = arch.arch.max_seq_len().max(rows + k);
    let blocks = context.div_ceil(block_size).max(1);
    let layers = arch.arch.num_layers();
    let null = Tensor::zeros(blocks, DType::I64, device)?;
    let view = oscar_models::draft::DraftPagedView {
        pools: (0..layers).map(|_| Some((null.clone(), null.clone()))).collect(),
        table: (0..blocks as u32).collect(),
        block_size,
    };
    // The round's last token sits at the context's end, so every step of it attends over nearly a whole context.
    let first = context - k;
    let dispatcher = crate::engine::offload_dispatcher();
    for i in 0..k {
        let logits = model.draft_forward(1, first + i, &view, &mut cache)?;
        // The step's own tail, as the record is the step: each layer's kept routing is handed back here.
        if let Some(d) = dispatcher {
            let _ = d.take_draft_routings();
            d.end_draft_forward();
        }
        let row = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let _room = row.affine(1.0, 0.0)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `nblk` host blocks of K `[1, 2, 4, 2]` then V `[1, 4, 4]`, all values distinct, the drafter's tail `reserved`.
    fn host_blocks(nblk: usize, reserved: std::ops::Range<u32>) -> KvView {
        use mistralrs_core::paged_attention::{KvBlockLayout, LayerRows};
        let layout = KvBlockLayout {
            block_size: 4,
            dtype: DType::F32,
            rows: vec![Some(LayerRows { k: vec![1, 2, 4, 2], v: vec![1, 4, 4], k_offset: 0, v_offset: 64 }), None],
        };
        mistralrs_core::paged_attention::set_kv_layout(layout.clone());
        let dev = Device::Cpu;
        let blocks = (0..nblk)
            .map(|b| {
                let k = Tensor::arange((b * 16) as f32, ((b + 1) * 16) as f32, &dev).unwrap();
                let v = Tensor::arange((1000 + b * 16) as f32, (1000 + (b + 1) * 16) as f32, &dev).unwrap();
                Tensor::cat(&[k, v], 0).unwrap()
            })
            .collect();
        let null = Tensor::zeros(1, DType::I64, &dev).unwrap();
        KvView {
            pools: vec![Some((null.clone(), null)), None],
            memory: std::sync::Arc::new(crate::kv_store::TensorBlocks { blocks, layout }),
            block_size: 4,
            reserved,
        }
    }

    /// A hand-off counts itself alive until it drops. Tail assignments persist, retire below the floor, and exhaust
    /// with an error, never reusing a live block; the view re-bases a partial block and keeps the drafted rows.
    #[test]
    fn a_hand_off_is_counted_and_the_view_rebases_the_partial_block_over_reused_tail_blocks() -> Result<()> {
        let state = || -> Result<mistralrs_core::kv_cache::RecurrentState> {
            let t = Tensor::zeros((2,), candle_core::DType::F32, &Device::Cpu)?;
            Ok(mistralrs_core::kv_cache::RecurrentState::from_layers(vec![(t.clone(), t)], 0))
        };
        let before = handoffs_alive();
        let a = GdnHandoff::new(state()?, vec![0]);
        let b = GdnHandoff::new(state()?, vec![1]);
        assert_eq!(handoffs_alive(), before + 2);
        drop(a);
        assert_eq!(handoffs_alive(), before + 1);
        drop(b);
        assert_eq!(handoffs_alive(), before);
        // Same span next session: identical assignment; the floor advancing retires index 2, its block recycled for
        // index 4; three live indices against two reserved blocks: refused.
        let mut tail = DraftTailBlocks::default();
        let reserved = 8u32..10;
        assert_eq!(tail.assign(2, 3, &reserved).unwrap(), vec![8, 9]);
        assert_eq!(tail.assign(2, 3, &reserved).unwrap(), vec![8, 9]);
        assert_eq!(tail.assign(3, 4, &reserved).unwrap(), vec![9, 8]);
        assert!(tail.assign(3, 5, &reserved).is_err());
        // n = 9: full = 2 engine blocks, 1 engine row in block index 2.
        let (paged, table, mut tail) = (host_blocks(6, 4..6), [0, 1, 2], DraftTailBlocks::default());
        let view = assemble_view(&paged, &table, 9, 14, &mut tail).unwrap();
        assert_eq!(view.table, vec![0, 1, 4, 5]);
        // K row `[1, 2, 4, 2]` and V row `[1, 4, 4]` of a block, token `t`.
        let k_row = |b: usize| paged.memory.block(b).unwrap().narrow(0, 0, 16).unwrap().reshape((1, 2, 4, 2)).unwrap();
        let v_row = |b: usize| paged.memory.block(b).unwrap().narrow(0, 16, 16).unwrap().reshape((1, 4, 4)).unwrap();
        let token = |row: &Tensor, t: usize| -> Vec<f32> {
            row.narrow(2, t, 1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
        };
        assert_eq!(token(&k_row(2), 0), token(&k_row(4), 0), "committed K row re-based");
        assert_eq!(token(&v_row(2), 0), token(&v_row(4), 0), "committed V row re-based");
        // A second session at n = 10 keeps the drafter's row 2.
        let marker = Tensor::full(-7f32, (1, 2, 1, 2), &Device::Cpu).unwrap();
        k_row(4).slice_set(&marker, 2, 2).unwrap();
        let view2 = assemble_view(&paged, &table, 10, 14, &mut tail).unwrap();
        assert_eq!(view2.table, vec![0, 1, 4, 5]);
        assert_eq!(token(&k_row(4), 2), vec![-7f32; 4], "drafted row survives the re-base");
        assert_eq!(token(&k_row(2), 1), token(&k_row(4), 1), "the second committed row re-based");
        Ok(())
    }
}
