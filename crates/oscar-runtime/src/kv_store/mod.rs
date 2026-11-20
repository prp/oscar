//! The durable KV block store behind the paged pool: a block is named by its chain hash, so none is stored twice.

use std::path::{Path, PathBuf};

use std::sync::mpsc::{Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use candle_core::{Device, Result};
use oscar_offload::machine::GB;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use mistralrs_core::kv_cache::{recurrent_state_memory, RecurrentState};
use mistralrs_core::paged_attention::{KvBlockLayout, KvBlockMemory};
use mistralrs_core::prefix_cacher::{PagedKvStore, SealedBlockMeta, TailDeposit};

mod arena;
mod index;
mod record;
#[cfg(test)]
mod tests;

use arena::{Arena, ArenaRange, Record};
use index::{BlockIndex, DiskLoc, SHARED};
use record::{RecordKind, RecordMeta, SnapMeta, TensorGeom, FORMAT_VERSION};

/// Shorter entries are not worth a disk round trip.
const MIN_PERSIST_TOKENS: usize = 64;

const MB: f64 = (1u64 << 20) as f64;
pub const STAGING_CAP_BYTES: u64 = 768 * 1024 * 1024;

static STAGING_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(STAGING_CAP_BYTES);

/// The plan's charge for the arena; call before the store is built.
pub fn set_staging_bytes(bytes: u64) {
    STAGING_BYTES.store(bytes.max(1), std::sync::atomic::Ordering::Relaxed);
}
/// A deposit's wait for writeback before storing less: generous for a healthy drive, a bound on a wedged one.
const STAGING_FLUSH_BOUND: Duration = Duration::from_secs(30);

/// Kept in `key.json`: a mismatch at bind sweeps the store rather than misread it.
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug)]
struct StoreKey {
    version: u32,
    block_size: usize,
    /// (layer, K row, V row) per KV-bearing layer.
    layers: Vec<(usize, TensorGeom, TensorGeom)>,
    /// Chain hash of a fixed token pattern: fingerprints the hash function.
    hash_probe: u64,
}

struct Geometry {
    block_size: usize,
    layers: Vec<(usize, TensorGeom, TensorGeom)>,
}

/// A block record's length: every layer's rows, K then V; `None` where a row cannot be sized.
fn record_len(layers: &[(usize, TensorGeom, TensorGeom)]) -> Option<u64> {
    layers.iter().map(|(_, k, v)| Some(k.nbytes().ok()? + v.nbytes().ok()?)).sum()
}

struct Inner {
    index: BlockIndex,
    geom: Option<Geometry>,
    memory: Option<Arc<dyn KvBlockMemory>>,
    /// Allocated at bind, once the pool's device is known.
    arena: Option<Arena>,
    /// The engine binds at every prompt step; the first decides for the run.
    bound: bool,
}

enum Job {
    Drain,
    Flush(SyncSender<()>),
}

pub struct PagedBlockStore {
    dir: PathBuf,
    budget_bytes: u64,
    staging_bytes: u64,
    inner: Arc<Mutex<Inner>>,
    jobs: Sender<Job>,
}

static GLOBAL: OnceLock<Arc<PagedBlockStore>> = OnceLock::new();
static STARTUP_NOTE: OnceLock<String> = OnceLock::new();

/// What the store found on disk and whether the bind turned it off; `None` without a store.
pub fn startup_note() -> Option<String> {
    let off = GLOBAL.get().map(|s| s.inner.lock().unwrap()).is_some_and(|g| g.bound && g.geom.is_none());
    STARTUP_NOTE.get().map(|n| if off { format!("{n}; the store is off this run") } else { n.clone() })
}

/// 0 without a store.
pub fn disk_bytes() -> u64 {
    GLOBAL.get().map_or(0, |s| s.inner.lock().unwrap().index.disk_bytes)
}

/// Left free on the volume: a write that would cross it peels the coldest records, or waits with none left.
const VOLUME_RESERVE_BYTES: u64 = 4 << 30;

/// The rate while demand IO is busy, a few percent of a drive's reads: an agent loop leaves no idle gap to wait for.
const LOAD_WRITE_BYTES_PER_S: u64 = 256 << 20;

/// Nothing on an idle drive; under load, the wait until the last write is paid for at the rate.
fn write_wait(bytes: u64) -> Duration {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    static PAID_AT_MS: AtomicU64 = AtomicU64::new(0);
    if oscar_offload::io::PENDING.load(Relaxed) == 0 {
        return Duration::ZERO;
    }
    static ORIGIN: OnceLock<std::time::Instant> = OnceLock::new();
    let now_ms = ORIGIN.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64;
    let paid_at = PAID_AT_MS.load(Relaxed).max(now_ms);
    PAID_AT_MS.store(paid_at + bytes.saturating_mul(1000) / LOAD_WRITE_BYTES_PER_S, Relaxed);
    Duration::from_millis(paid_at - now_ms)
}

/// Bounded flush for exit paths; a no-op without a store.
pub fn flush_global(bound: Duration) {
    if let Some(store) = GLOBAL.get() {
        store.flush(bound);
    }
}

/// The factory installed into `mistralrs-core`; `None` without a directory.
pub fn factory(_device: &Device) -> Option<Arc<dyn PagedKvStore>> {
    let (dir, budget_gb) = CONFIGURED.get()?.clone()?;
    let staging_bytes = STAGING_BYTES.load(std::sync::atomic::Ordering::Relaxed);
    let store = Arc::new(PagedBlockStore::open(dir, budget_gb, staging_bytes)?);
    let _ = GLOBAL.set(Arc::clone(&store));
    Some(store)
}

static CONFIGURED: OnceLock<Option<(PathBuf, u64)>> = OnceLock::new();

/// The directory and its disk budget in GB, before the factory runs; no directory, no store.
pub fn configure(dir: Option<PathBuf>, budget_gb: u64) {
    let _ = CONFIGURED.set(dir.map(|d| (d, budget_gb)));
}

pub fn configured() -> bool {
    CONFIGURED.get().is_some_and(|d| d.is_some())
}

impl PagedBlockStore {
    /// Sweeps `dir`, rebuilds the index from the record files and starts the writer.
    pub fn open(dir: PathBuf, budget_gb: u64, staging_bytes: u64) -> Option<Self> {
        if let Err(e) = std::fs::create_dir_all(&dir) {
            warn!("The reply store is off: cannot create {}: {e}", dir.display());
            return None;
        }
        let (records, other_version) = record::sweep(&dir);
        let index = BlockIndex::rebuild(records);
        let n_blocks = index.map.len();
        let disk_gb = index.disk_bytes as f64 / GB;
        // Kept, not logged: a write during the load lands mid-spinner.
        let _ = STARTUP_NOTE.set(format!(
            "{} - {n_blocks} block{} stored ({disk_gb:.2} GB) of a {budget_gb} GB budget, staged through a {:.2} GB arena{}",
            dir.display(),
            if n_blocks == 1 { "" } else { "s" },
            staging_bytes as f64 / GB,
            if other_version > 0 { format!("; {other_version} files of an older store format removed") } else { String::new() },
        ));
        let (jobs, rx) = std::sync::mpsc::channel();
        let inner = Arc::new(Mutex::new(Inner { index, geom: None, memory: None, arena: None, bound: false }));
        let store = Self {
            dir: dir.clone(),
            budget_bytes: budget_gb.saturating_mul(1024 * 1024 * 1024),
            staging_bytes,
            inner: Arc::clone(&inner),
            jobs,
        };
        store.spawn_writeback(rx);
        Some(store)
    }

    fn key_path(&self) -> PathBuf {
        self.dir.join(record::KEY_FILE)
    }

    fn check_key(&self, inner: &mut Inner, key: StoreKey) {
        let on_disk: Option<StoreKey> =
            record::read_file(&self.key_path()).ok().and_then(|b| serde_json::from_slice(&b).ok());
        if on_disk.as_ref() == Some(&key) {
            return;
        }
        let stale = !inner.index.map.is_empty() || on_disk.is_some();
        if stale {
            // A format change was reported at open, which swept its files.
            if on_disk.as_ref().is_none_or(|k| k.version == FORMAT_VERSION) {
                warn!(
                    "The reply store was cleared: layout changed ({} stored blocks under an old key), clearing {}",
                    inner.index.map.len(),
                    self.dir.display()
                );
            }
            record::clear(&self.dir);
            inner.index = BlockIndex::default();
        }
        if let Ok(bytes) = serde_json::to_vec_pretty(&key) {
            let _ = record::write_file(&self.key_path(), &bytes);
        }
    }

    fn spawn_writeback(&self, rx: Receiver<Job>) {
        let inner = Arc::clone(&self.inner);
        let dir = self.dir.clone();
        let budget = self.budget_bytes;
        std::thread::Builder::new()
            .name("oscar-kvwb".into())
            .spawn(move || loop {
                let job = rx.recv_timeout(Duration::from_millis(500));
                match job {
                    Ok(Job::Flush(ack)) => {
                        Self::drain(&inner, &dir, budget, true);
                        let _ = ack.send(());
                    }
                    Ok(Job::Drain) | Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        Self::drain(&inner, &dir, budget, false);
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                }
            })
            .expect("kv store writeback thread");
    }

    /// Writes queued records in arena order, a file each; `force` (exit, a full arena) writes held turns, at any rate.
    fn drain(inner: &Arc<Mutex<Inner>>, dir: &Path, budget: u64, force: bool) {
        loop {
            // The front record under the lock; the write outside it.
            let (buf, meta, range) = {
                let mut g = inner.lock().unwrap();
                let Inner { index, geom, arena, .. } = &mut *g;
                let (Some(arena), Some(_)) = (arena.as_mut(), geom.as_ref()) else { return };
                // Front padding (superseded records too) is not written.
                while arena.ring.queue.front().is_some_and(|p| p.record.is_none()) {
                    arena.ring.release_front();
                }
                let Some(r) = arena.ring.queue.front().and_then(|p| p.record.as_ref()) else { return };
                let mut kind = r.kind.clone();
                if let RecordKind::Tail { group, tokens, .. } = &mut kind {
                    // Its turn when written: a retry adopts what it repeats. A turn's records (a branch's not) wait to
                    // be superseded or forced: a live conversation writes KV only.
                    *group = index.map.get(&r.hash).and_then(|b| b.tail(r.pos, tokens)).map_or(*group, |t| t.group);
                    if *group != SHARED && !force {
                        return;
                    }
                }
                let last_access = index.map.get(&r.hash).map_or(0, |b| b.last_access);
                (
                    arena.buf.clone(),
                    RecordMeta { kind, hash: r.hash, parent: r.parent, pos: r.pos, last_access },
                    r.range,
                )
            };
            if !force {
                std::thread::sleep(write_wait(range.len));
            }

            // The volume bounds the budget: peel to the reserve, else wait.
            static WAITING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            let need = range.len + VOLUME_RESERVE_BYTES;
            let free = || oscar_offload::machine::volume_free(dir).unwrap_or(u64::MAX);
            if free() < need {
                {
                    let mut g = inner.lock().unwrap();
                    let Inner { index, arena, .. } = &mut *g;
                    let peeled = index.peel_to_budget(index.disk_bytes.saturating_sub(need - free()));
                    if let Some(arena) = arena.as_mut() {
                        for r in peeled {
                            arena.ring.cancel(r);
                        }
                    }
                }
                if free() < need {
                    if !WAITING.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        warn!(
                            "The disk is nearly full: {} MB free on the volume; writes wait for room ({} GB kept free)",
                            free() >> 20,
                            VOLUME_RESERVE_BYTES >> 30
                        );
                    }
                    return;
                }
            }
            WAITING.store(false, std::sync::atomic::Ordering::Relaxed);
            // SAFETY: allocated until this writer releases it; its GPU copies completed before it was queued.
            let path = match record::write(dir, &meta, unsafe { buf.bytes(range) }) {
                Ok(path) => path,
                Err(e) => {
                    warn!("A saved reply could not be written: record write failed: {e}");
                    return;
                }
            };
            let mut g = inner.lock().unwrap();
            let Inner { index, arena, .. } = &mut *g;
            // A turn superseded mid-write leaves a file nothing in the index names.
            let loc = DiskLoc { path, len: range.len };
            if !index.mark_sealed(&meta, &loc) {
                let _ = std::fs::remove_file(&loc.path);
            }
            let peeled = index.peel_to_budget(budget);
            // Only this thread releases, so the front is the record's.
            if let Some(arena) = arena.as_mut() {
                for r in peeled {
                    arena.ring.cancel(r);
                }
                arena.ring.release_front();
            }
        }
    }
}

/// A record's bytes during a restore: staged, or its file and length on disk.
enum Src {
    Staged(ArenaRange),
    Disk(PathBuf, u64),
}

/// The boundary record a restore resumes at.
struct TailPick {
    /// Full blocks below the partial rows.
    depth: usize,
    /// Matched tokens past `depth`: a hybrid's whole stored tail, its state sitting at the end, else the longest match.
    n: usize,
    /// Rows stored; those past `n` are installed but re-encoded before a read.
    n_stored: usize,
    meta: SnapMeta,
}

impl PagedBlockStore {
    /// After one bounded writeback if the ring is full; `None`: store less.
    fn alloc(&self, len: u64) -> Option<ArenaRange> {
        if let Some(r) = self.inner.lock().unwrap().arena.as_mut()?.ring.alloc(len) {
            return Some(r);
        }
        self.flush(STAGING_FLUSH_BOUND);
        self.inner.lock().unwrap().arena.as_mut()?.ring.alloc(len)
    }

    /// Staged records where they sit, stored ones read direct, while the ring takes them: fewer ranges than sources
    /// is a full ring, an error a failed read.
    fn read_records(&self, buf: &arena::ArenaBuf, srcs: &[Src]) -> (Vec<ArenaRange>, Option<candle_core::Error>) {
        let mut out = Vec::with_capacity(srcs.len());
        for src in srcs {
            match src {
                Src::Staged(r) => out.push(*r),
                Src::Disk(path, len) => match self.read_file(buf, path, *len) {
                    Ok(Some(r)) => out.push(r),
                    Ok(None) => break,
                    Err(e) => return (out, Some(e)),
                },
            }
        }
        (out, None)
    }

    /// A record's payload into a page-aligned range of the ring; `None`: the ring cannot take its pages.
    fn read_file(&self, buf: &arena::ArenaBuf, path: &Path, len: u64) -> Result<Option<ArenaRange>> {
        let pages = len.next_multiple_of(record::PAGE);
        let Some(r) = self.alloc(pages + record::PAGE) else {
            return Ok(None);
        };
        let off = r.off.next_multiple_of(record::PAGE);
        let f = record::open(path)?;
        // SAFETY: a fresh transient range no GPU copy addresses, page-aligned (the buffer starts on one), `pages` long.
        unsafe { record::read_payload(&f, pages, len, buf.bytes_mut(ArenaRange { off, len: pages })) }?;
        Ok(Some(ArenaRange { off, len }))
    }

    /// A transient scope on the ring for `f`'s reads and copies, given back once the copies completed.
    fn staged<R>(&self, buf: &arena::ArenaBuf, f: impl FnOnce() -> R) -> Option<R> {
        let mark = self.inner.lock().unwrap().arena.as_mut()?.ring.mark();
        let out = f();
        if let Err(e) = buf.wait() {
            warn!("A saved reply could not be read: restore copies did not complete: {e}");
        }
        self.inner.lock().unwrap().arena.as_mut()?.ring.reset_to(mark);
        Some(out)
    }

    /// Records into the pool and the states in runs the arena holds: the blocks, then the tail record's rows and state.
    #[allow(clippy::too_many_arguments)]
    fn restore_from_arena(
        &self, buf: &arena::ArenaBuf, srcs: Vec<Src>, tail_pick: Option<TailPick>, ram_blocks: usize,
        dst_block_ids: &[usize], hybrid: bool, block_size: usize, layers: &[(usize, TensorGeom, TensorGeom)],
        memory: &Arc<dyn KvBlockMemory>, t0: std::time::Instant,
    ) -> Option<(usize, Option<RecurrentState>)> {
        let rec_len = record_len(layers)?;
        let dtype = record::tag_dtype(&layers[0].1.dtype).ok()?;
        let rec_elems = rec_len as usize / dtype.size_in_bytes();
        let fill = |payload: ArenaRange, dst: usize| -> Result<()> {
            let block = memory.block(dst)?;
            block.narrow(0, 0, rec_elems)?.slice_set(&buf.view(payload, &[rec_elems], dtype)?, 0, 0)
        };
        // The tail's record is the last source, read only once every block below it is placed.
        let blocks = srcs.len() - usize::from(tail_pick.is_some());
        let (mut placed, mut runs) = (0, 0);
        while placed < blocks {
            let (n, more) = self.staged(buf, || {
                let (payloads, failed) = self.read_records(buf, &srcs[placed..blocks]);
                match &failed {
                    Some(e) => warn!("A saved reply could not be read: block read failed mid-restore: {e}"),
                    None if payloads.is_empty() => {
                        warn!("A saved reply could not be read: staging arena takes no record of restore")
                    }
                    None => {}
                }
                let mut n = 0;
                for p in &payloads {
                    if let Err(e) = fill(*p, dst_block_ids[placed + n]) {
                        warn!("A saved reply could not be read: block fill failed: {e}");
                        return (n, false);
                    }
                    n += 1;
                }
                (n, failed.is_none() && n > 0)
            })?;
            placed += n;
            runs += 1;
            if !more {
                break;
            }
        }

        // A state from its pool, where the plan counts states; a pool at its count refuses, and the request re-encodes.
        let rebuild_state = |payload: ArenaRange, skip: usize, meta: &SnapMeta| -> Option<RecurrentState> {
            let fill = || -> Result<RecurrentState> {
                let state = recurrent_state_memory()?.take(&meta.layout()?, &buf.device())?;
                let mut off = payload.off + skip as u64;
                for ((conv, recurrent), geoms) in state.layers().iter().zip(meta.tensors.chunks_exact(2)) {
                    for (dst, geom) in [(conv, &geoms[0]), (recurrent, &geoms[1])] {
                        let n = geom.nbytes()?;
                        let dtype = record::tag_dtype(&geom.dtype)?;
                        dst.slice_set(&buf.view(ArenaRange { off, len: n }, &geom.shape, dtype)?, 0, 0)?;
                        off += n;
                    }
                }
                // One offset per layer on disk; the layers advance together.
                Ok(state.at(meta.seqlen_offsets.first().copied().unwrap_or(0)))
            };
            match fill() {
                Ok(state) => Some(state),
                Err(e) => {
                    warn!("A saved reply could not be read: state rebuild failed: {e}");
                    None
                }
            }
        };
        // The tail's block fills as one more, its rows past the boundary re-encoded before a read; below RAM only its
        // state is used.
        let tail = match &tail_pick {
            Some(tp) if placed == blocks => self.staged(buf, || -> Option<(usize, Option<RecurrentState>)> {
                let (mut payloads, failed) = self.read_records(buf, &srcs[blocks..]);
                let Some(p) = payloads.pop() else {
                    warn!(
                        "A saved reply could not be read: tail read failed: {}",
                        failed.map_or("staging arena full".to_string(), |e| e.to_string())
                    );
                    return None;
                };
                let rows_len = rec_len * u64::from(tp.n_stored > 0);
                if tp.n_stored > 0 && tp.depth >= ram_blocks {
                    if let Err(e) = fill(ArenaRange { off: p.off, len: rec_len }, dst_block_ids[tp.depth - ram_blocks])
                    {
                        warn!("A saved reply could not be read: tail fill failed: {e}");
                        return None;
                    }
                }
                let snaps = (!tp.meta.seqlen_offsets.is_empty())
                    .then(|| rebuild_state(p, rows_len as usize, &tp.meta))
                    .flatten();
                Some((tp.depth * block_size + tp.n, snaps))
            })?,
            _ => None,
        };
        tracing::debug!(
            "kv store restore: {placed} blocks in {runs} runs, a tail record: {}, {:.0} ms",
            tail.is_some(),
            t0.elapsed().as_secs_f64() * 1e3
        );
        match tail {
            Some((at, snaps)) if !(hybrid && snaps.is_none()) => Some((at, snaps)),
            Some(_) => None,
            // A non-hybrid's blocks alone: the RAM match and the blocks read.
            None if !hybrid && placed > 0 => Some(((ram_blocks + placed) * block_size, None)),
            None => None,
        }
    }

    /// A turn's pair, or with `shared` a branch's state, staged root-first in runs the ring takes, the writer flushed
    /// between them; a run that takes nothing defers the rest to a later turn.
    fn deposit(&self, blocks: &[SealedBlockMeta], tails: Vec<TailDeposit>, toks: &[u32], shared: bool) {
        enum Unit<'a> {
            Block(&'a SealedBlockMeta),
            Tail(TailDeposit),
        }
        let t0 = std::time::Instant::now();
        let (units, group, layers, block_size, memory) = {
            let mut g = self.inner.lock().unwrap();
            let (Some(geom), Some(memory), Some(_)) = (g.geom.as_ref(), g.memory.clone(), g.arena.as_ref()) else {
                return;
            };
            if blocks.len() * geom.block_size < MIN_PERSIST_TOKENS {
                return;
            }
            let block_size = geom.block_size;
            let layers = geom.layers.clone();
            // The turns this one continues are superseded; a record re-deposited unchanged stays, as this turn's.
            let group = if shared { SHARED } else { g.index.tick() };
            let upto = tails.iter().map(|t| t.pos as usize * block_size + t.tokens.len()).min().unwrap_or(0);
            let chain: Vec<u64> = blocks.iter().map(|b| b.hash).collect();
            let continued = if shared { Vec::new() } else { g.index.groups_on_chain(&chain, toks, block_size, upto) };
            let mut fresh: Vec<TailDeposit> = Vec::with_capacity(tails.len());
            for t in tails {
                let ix = &mut g.index;
                let stays =
                    !shared && ix.adopt(t.parent, t.pos, &t.tokens, group) || ix.holds(t.parent, t.pos, &t.tokens);
                // A record rides its parent block, held or in this deposit, which no supersede below removes.
                let rides = blocks.iter().any(|b| b.hash == t.parent) || ix.map.contains_key(&t.parent);
                if rides && !stays {
                    fresh.push(t);
                }
            }
            let cancelled: Vec<ArenaRange> = continued.into_iter().flat_map(|c| g.index.supersede(c)).collect();
            if let Some(arena) = g.arena.as_mut() {
                for r in cancelled {
                    arena.ring.cancel(r);
                }
            }
            // Fresh blocks, a record each; a held block is a touch, not a copy.
            let mut units: Vec<Unit> = Vec::new();
            for b in blocks {
                if g.index.map.contains_key(&b.hash) {
                    g.index.touch(b.hash);
                } else {
                    units.push(Unit::Block(b));
                }
            }
            units.extend(fresh.into_iter().map(Unit::Tail));
            (units, group, layers, block_size, memory)
        };
        let Some(rec_len) = record_len(&layers) else { return };
        let tensor_bytes = |snaps: Option<&RecurrentState>| -> u64 {
            snaps
                .map_or(&[][..], RecurrentState::layers)
                .iter()
                .flat_map(|(c, r)| [c, r])
                .map(|t| (t.elem_count() * t.dtype().size_in_bytes()) as u64)
                .sum()
        };
        let unit_len = |u: &Unit| -> u64 {
            match u {
                Unit::Block(_) => rec_len,
                Unit::Tail(t) => rec_len * u64::from(!t.tokens.is_empty()) + tensor_bytes(t.snapshots.as_ref()),
            }
        };
        let mut done = 0;
        while done < units.len() {
            // A run: the ranges the ring takes now; the next run's first alloc flushes the writer, this run included.
            let mut ranges: Vec<ArenaRange> = Vec::with_capacity(units.len() - done);
            for u in &units[done..] {
                match self.alloc(unit_len(u)) {
                    Some(r) => ranges.push(r),
                    None => break,
                }
            }
            if ranges.is_empty() {
                let (cap, queued) = {
                    let g = self.inner.lock().unwrap();
                    g.arena.as_ref().map_or((0, 0), |a| (a.ring.capacity(), a.ring.queued_bytes))
                };
                info!(
                    "staging arena full - {} of {} records deferred to later turn: {:.0} MB asked of {:.0} MB \
                     arena, {:.0} MB still queued for disk",
                    units.len() - done,
                    units.len(),
                    units.iter().map(|u| unit_len(u)).sum::<u64>() as f64 / MB,
                    cap as f64 / MB,
                    queued as f64 / MB,
                );
                break;
            }
            let run = &units[done..done + ranges.len()];
            done += ranges.len();

            // Copied before returning, as the pool may then recycle the blocks: a blit each, a record's states after its block.
            let (backlog, held) = {
                let mut g = self.inner.lock().unwrap();
                let Some(buf) = g.arena.as_ref().map(|a| a.buf.clone()) else { return };
                let es = record::tag_dtype(&layers[0].1.dtype).map_or(0, |d| d.size_in_bytes());
                let rec_elems = rec_len as usize / es.max(1);
                // `at`, the boundary the state is valid at: the reply end's is the live state, which carries no position.
                let copy_states =
                    |snaps: Option<&RecurrentState>, at: usize, mut off: u64| -> Result<(SnapMeta, u64)> {
                        let mut meta = SnapMeta { seqlen_offsets: Vec::new(), tensors: Vec::new() };
                        for (c, r) in snaps.map_or(&[][..], RecurrentState::layers) {
                            for t in [c, r] {
                                let t = t.contiguous()?;
                                let n = t.elem_count();
                                buf.copy_in(&t, 1, n, n, off, n)?;
                                off += (n * t.dtype().size_in_bytes()) as u64;
                                meta.tensors.push(TensorGeom::new(t.dims().to_vec(), t.dtype())?);
                            }
                            meta.seqlen_offsets.push(at);
                        }
                        Ok((meta, off))
                    };
                // Each range with its record, `None` where the copy failed.
                let mut staged: Vec<(ArenaRange, Option<Record>)> = Vec::with_capacity(ranges.len());
                let block_in = |id: usize, off: u64| -> Result<()> {
                    let block = memory.block(id)?;
                    buf.copy_in(&block.narrow(0, 0, rec_elems)?, 1, rec_elems, rec_elems, off, rec_elems)
                };
                for (u, range) in run.iter().zip(&ranges) {
                    let copied: Result<Record> = match u {
                        Unit::Block(b) => block_in(b.block_id, range.off).map(|()| {
                            let (hash, parent, pos, range) = (b.hash, b.parent, b.pos, *range);
                            Record { kind: RecordKind::Block, hash, parent, pos, range }
                        }),
                        Unit::Tail(t) => {
                            let n = t.tokens.len();
                            (|| {
                                let mut off = range.off;
                                if n > 0 {
                                    block_in(t.block_id, off)?;
                                    off += rec_len;
                                }
                                copy_states(t.snapshots.as_ref(), t.pos as usize * block_size + n, off)
                            })()
                            .map(|(meta, _)| {
                                let kind = RecordKind::Tail { group, tokens: t.tokens.clone(), meta };
                                Record { kind, hash: t.parent, parent: 0, pos: t.pos, range: *range }
                            })
                        }
                    };
                    staged.push((
                        *range,
                        copied
                            .inspect_err(|e| warn!("A saved reply could not be written: deposit copy failed: {e}"))
                            .ok(),
                    ));
                }
                // The records exist only now, in allocation order; a range holding none is padding the writer gives back.
                let waited = buf
                    .wait()
                    .inspect_err(|e| warn!("A saved reply could not be written: deposit copies did not complete: {e}"))
                    .is_ok();
                let Inner { index, arena, .. } = &mut *g;
                let Some(arena) = arena.as_mut() else { return };
                for (range, copied) in staged {
                    let record = copied.filter(|_| waited).and_then(|mut record| {
                        match &record.kind {
                            RecordKind::Block => index.insert_staged(record.hash, record.parent, record.range),
                            RecordKind::Tail { tokens, meta, .. } => {
                                // Its parent block may have been peeled while the ranges were taken: padding then.
                                record.parent = index.map.get(&record.hash)?.parent;
                                let (tokens, meta) = (tokens.clone(), meta.clone());
                                index.attach_staged_tail(record.hash, group, record.pos, tokens, meta, record.range);
                            }
                        }
                        Some(record)
                    });
                    arena.ring.enqueue(range, record);
                }
                (arena.ring.queued_bytes, index.map.len())
            };
            // Per run, so a footprint climb can be pinned on the backlog; the wall is the engine thread's at the turn's end.
            info!(
                "{} MB queued for disk | {} blocks held | {} of {} records staged in {:.0} ms",
                backlog / (1024 * 1024),
                held,
                done,
                units.len(),
                t0.elapsed().as_secs_f64() * 1e3
            );
            if backlog > 0 {
                let _ = self.jobs.send(Job::Drain);
            }
        }
    }
}

impl PagedKvStore for PagedBlockStore {
    fn bind(&self, memory: Arc<dyn KvBlockMemory>, layout: &KvBlockLayout, hash_probe: u64) {
        let mut g = self.inner.lock().unwrap();
        if std::mem::replace(&mut g.bound, true) {
            return;
        }
        let refuse = |why: String| warn!("The reply store is off: {why}");
        let block_size = layout.block_size;
        // A record is a block's bytes as the pool lays them out.
        let mut layers = Vec::new();
        let mut offset = 0usize;
        for (i, rows) in layout.kv_layers() {
            let (Ok(kg), Ok(vg)) =
                (TensorGeom::new(rows.k.clone(), layout.dtype), TensorGeom::new(rows.v.clone(), layout.dtype))
            else {
                return refuse(format!("unsupported KV dtype at layer {i}"));
            };
            let (kn, vn) = (kg.nbytes().expect("a dtype `new` took"), vg.nbytes().expect("a dtype `new` took"));
            if rows.k_offset != offset || rows.v_offset != offset + kn as usize {
                return refuse(format!("layer {i}'s rows are not where a record puts them"));
            }
            offset += (kn + vn) as usize;
            layers.push((i, kg, vg));
        }
        if layers.is_empty() {
            return refuse("no KV-bearing layers found at bind".into());
        }
        self.check_key(&mut g, StoreKey { version: FORMAT_VERSION, block_size, layers: layers.clone(), hash_probe });
        let device = memory.block(0).map(|b| b.device().clone());
        let Ok(Device::Metal(md)) = device else {
            return refuse("the staging arena needs the Metal device".into());
        };
        match Arena::new(&md, self.staging_bytes) {
            Ok(arena) => g.arena = Some(arena),
            // The refusal itself was announced by the bound that made it.
            Err(_) => return refuse(format!("no room for the {:.2} GB staging arena", self.staging_bytes as f64 / GB)),
        }
        g.geom = Some(Geometry { block_size, layers });
        g.memory = Some(memory);
    }

    fn note_finished_sequence(&self, blocks: &[SealedBlockMeta], tails: Vec<TailDeposit>, toks: &[u32]) {
        self.deposit(blocks, tails, toks, false);
    }

    fn note_shared(&self, blocks: &[SealedBlockMeta], record: TailDeposit) {
        self.deposit(blocks, vec![record], &[], true);
    }

    fn fork_depth(&self, chain: &[u64], toks: &[u32]) -> Option<usize> {
        let g = self.inner.lock().unwrap();
        let block_size = g.geom.as_ref()?.block_size;
        g.index.fork_depth(chain, toks, block_size)
    }

    fn restore_extension(
        &self, chain: &[u64], ram_blocks: usize, dst_block_ids: &[usize], hybrid: bool, ram_state_tokens: usize,
        toks: &[u32],
    ) -> Option<(usize, Option<RecurrentState>)> {
        let t0 = std::time::Instant::now();
        let (picked, ram_resume, block_size, layers, memory) = {
            let mut g = self.inner.lock().unwrap();
            let (Some(geom), Some(memory)) = (g.geom.as_ref(), g.memory.clone()) else {
                return None;
            };
            let block_size = geom.block_size;
            let layers = geom.layers.clone();
            let src_of = |tier: &index::Tier| -> Option<Src> {
                match tier {
                    index::Tier::Staged(p) => Some(Src::Staged(*p)),
                    index::Tier::Disk(loc) => Some(Src::Disk(loc.path.clone(), loc.len)),
                }
            };
            let max_depth = chain.len().min(ram_blocks + dst_block_ids.len());
            let mut d_ext = ram_blocks;
            while d_ext < max_depth && g.index.map.contains_key(&chain[d_ext]) {
                d_ext += 1;
            }
            // The deepest matching boundary record; a hybrid searches below the RAM match too, as its KV needs a state.
            let tail_floor = if hybrid { 1 } else { ram_blocks.max(1) };
            let mut tail_pick: Option<(TailPick, Src)> = None;
            for d in (tail_floor..=d_ext).rev() {
                let Some(rec) = g.index.map.get(&chain[d - 1]) else {
                    continue;
                };
                let start = d * block_size;
                let best = rec
                    .tails
                    .iter()
                    .filter(|t| t.pos as usize == d)
                    .filter(|t| !hybrid || !t.meta.seqlen_offsets.is_empty())
                    .filter_map(|t| {
                        if t.tokens.is_empty() {
                            // State-only record at a block-aligned boundary.
                            return (start < toks.len()).then_some((t, 0usize));
                        }
                        if d >= ram_blocks && dst_block_ids.len() <= d - ram_blocks {
                            return None;
                        }
                        // Capped so one token remains to re-prefill for logits.
                        let cap = t.tokens.len().min(toks.len().saturating_sub(start + 1));
                        let n = t.tokens.iter().zip(&toks[start..start + cap]).take_while(|(a, b)| a == b).count();
                        if hybrid {
                            (n == t.tokens.len()).then_some((t, n))
                        } else {
                            (n > 0).then_some((t, n))
                        }
                    })
                    .max_by_key(|(_, n)| *n);
                if let Some((t, n)) = best {
                    let src = src_of(&t.tier)?;
                    tail_pick = Some((TailPick { depth: d, n, n_stored: t.tokens.len(), meta: t.meta.clone() }, src));
                    break;
                }
            }
            let ram_resume = if hybrid { ram_state_tokens } else { ram_blocks * block_size };
            // A record must pass what resumes without one: a non-hybrid's stored blocks, the caller's RAM anchor.
            let blocks_resume = !hybrid && d_ext > ram_blocks;
            let floor = if blocks_resume { (d_ext * block_size).max(ram_state_tokens) } else { ram_state_tokens };
            let tail_pick = tail_pick.filter(|(t, _)| t.depth * block_size + t.n > floor);
            let picked = if tail_pick.is_none() && !blocks_resume {
                None
            } else {
                let (tail_pick, tail_src) = tail_pick.unzip();
                let depth = tail_pick.as_ref().map_or(d_ext, |t| t.depth);
                // A below-RAM tail brings back state only.
                let mut srcs = Vec::with_capacity(depth.saturating_sub(ram_blocks) + 1);
                for &h in &chain[ram_blocks..depth.max(ram_blocks)] {
                    let rec = g.index.map.get(&h)?;
                    srcs.push(src_of(&rec.tier)?);
                }
                srcs.extend(tail_src);
                for &h in &chain[..depth] {
                    g.index.touch(h);
                }
                Some((srcs, tail_pick))
            };
            (picked, ram_resume, block_size, layers, memory)
        };

        // Outside the lock, on the engine thread at admission: the fetch queue is empty, so the reads need no ordering.
        let restored = picked.and_then(|(srcs, tail_pick)| {
            let buf = self.inner.lock().unwrap().arena.as_ref()?.buf.clone();
            self.restore_from_arena(
                &buf,
                srcs,
                tail_pick,
                ram_blocks,
                dst_block_ids,
                hybrid,
                block_size,
                &layers,
                &memory,
                t0,
            )
        });
        // One line per attempt, declines included: a silent decline measures nothing.
        let resume = restored.as_ref().map_or(ram_resume, |(n, _)| *n);
        let took = mistralrs_core::live::duration_text(t0.elapsed().as_secs_f64());
        let took = if resume > ram_resume { format!(" in {took}") } else { String::new() };
        info!(
            "reuse | prompt {} tok | without the store {ram_resume} | from the store {}{took} | re-encode {}",
            toks.len(),
            resume.saturating_sub(ram_resume),
            toks.len().saturating_sub(resume),
        );
        restored
    }

    fn flush(&self, bound: Duration) {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        if self.jobs.send(Job::Flush(tx)).is_err() {
            return;
        }
        if rx.recv_timeout(bound).is_err() {
            warn!("A saved reply was not fully written: writeback unfinished within {bound:?}; unwritten tail not on disk");
        }
    }
}

/// For tests: a tensor per block, every block live, no address tables.
#[cfg(test)]
pub(crate) struct TensorBlocks {
    pub blocks: Vec<candle_core::Tensor>,
    pub layout: KvBlockLayout,
}

#[cfg(test)]
impl KvBlockMemory for TensorBlocks {
    fn blocks(&self) -> usize {
        self.blocks.len()
    }
    fn live_blocks(&self) -> usize {
        self.blocks.len()
    }
    fn layer_tables(&self) -> Vec<(candle_core::Tensor, candle_core::Tensor)> {
        Vec::new()
    }
    fn block(&self, id: usize) -> Result<candle_core::Tensor> {
        Ok(self.blocks[id].clone())
    }
    fn take(&self, _ids: &[usize]) -> Result<()> {
        Ok(())
    }
    fn release(&self, _ids: &[usize]) -> Result<()> {
        Ok(())
    }
}
