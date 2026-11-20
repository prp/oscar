//! The store's invariants, on synthetic pools.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use super::record::{self, RecordKind, RecordMeta};
use super::{PagedBlockStore, TensorBlocks};
use candle_core::{DType, Device, Tensor};
use mistralrs_core::kv_cache::RecurrentState;
use mistralrs_core::paged_attention::{KvBlockLayout, KvBlockMemory, LayerRows};
use mistralrs_core::prefix_cacher::{PagedKvStore, SealedBlockMeta, TailDeposit};

const BLOCK: usize = 32;
const NUM_BLOCKS: usize = 16;
const PROBE: u64 = 0xC0FFEE;
/// Small: these pools are kilobytes, and a small ring exercises its wrap.
const ARENA: u64 = 4 << 20;
const FLUSH: Duration = Duration::from_secs(10);

/// One device for every test: each `test_device()` is its own queue, and a wait on one never sees another's copies.
fn metal() -> Device {
    static DEV: std::sync::OnceLock<Device> = std::sync::OnceLock::new();
    let dev = DEV
        .get_or_init(|| {
            mistralrs_core::kv_cache::set_recurrent_state_memory(Arc::new(PlainStates));
            oscar_offload::pools::test_device().expect("the store's tests need the Metal device")
        })
        .clone();
    if let Device::Metal(md) = &dev {
        // A stream per test thread: threads sharing a queue commit out of each other's order.
        md.install_thread_stream(None).expect("a stream for this test's thread");
    }
    dev
}

/// The states a restore fills, as tensors of their own.
struct PlainStates;

impl mistralrs_core::kv_cache::RecurrentStateMemory for PlainStates {
    fn take(
        &self, layout: &mistralrs_core::kv_cache::RecurrentStateLayout, device: &Device,
    ) -> candle_core::Result<RecurrentState> {
        let zeros = |(c, cd, r, rd): &(_, _, Vec<usize>, _)| {
            Ok((Tensor::zeros(*c, *cd, device)?, Tensor::zeros(r.as_slice(), *rd, device)?))
        };
        Ok(RecurrentState::from_layers(layout.layers.iter().map(zeros).collect::<candle_core::Result<_>>()?, 0))
    }
}

/// A half's rows, `[heads, head_dim, block]`: dim 2 is the block axis.
const ROW: usize = 2 * 4 * BLOCK;

/// `NUM_BLOCKS` blocks over `halves`, a layer's `(K dims, V dims)` or `None` without KV, packed in order; element `i`
/// of a layer's half from block 0 is `fill(layer, i)` in K, plus 0.5 in V.
fn pool_of(halves: &[Option<(Vec<usize>, Vec<usize>)>], fill: impl Fn(usize, usize) -> f32) -> Arc<TensorBlocks> {
    let dev = metal();
    let count = |dims: &[usize]| dims.iter().product::<usize>();
    let (mut rows, mut off) = (Vec::new(), 0);
    for half in halves {
        rows.push(half.as_ref().map(|(k, v)| {
            let row = LayerRows { k: k.clone(), v: v.clone(), k_offset: off, v_offset: off + count(k) * 4 };
            off += (count(k) + count(v)) * 4;
            row
        }));
    }
    let blocks = (0..NUM_BLOCKS)
        .map(|b| {
            let mut bytes = Vec::with_capacity(off / 4);
            for (layer, (k, v)) in halves.iter().enumerate().filter_map(|(l, h)| Some((l, h.as_ref()?))) {
                let (kn, vn) = (count(k), count(v));
                bytes.extend((0..kn).map(|i| fill(layer, b * kn + i)));
                bytes.extend((0..vn).map(|i| fill(layer, b * vn + i) + 0.5));
            }
            Tensor::from_vec(bytes, off / 4, &dev).unwrap()
        })
        .collect();
    Arc::new(TensorBlocks { blocks, layout: KvBlockLayout { block_size: BLOCK, dtype: DType::F32, rows } })
}

/// KV-bearing layers 0 and 2 (`[2, 4, block]` halves) around one without.
fn synth_pool(fill: impl Fn(usize, usize) -> f32) -> Arc<TensorBlocks> {
    let half = || Some((vec![2, 4, BLOCK], vec![2, 4, BLOCK]));
    pool_of(&[half(), None, half()], fill)
}

/// A standard layer 0 and a latent layer 1 (K `[block, 3]`, V `[block, 2]`): the block axis first, as the MLA rows.
fn latent_pool(fill: impl Fn(usize, usize) -> f32) -> Arc<TensorBlocks> {
    pool_of(&[Some((vec![2, 4, BLOCK], vec![2, 4, BLOCK])), Some((vec![BLOCK, 3], vec![BLOCK, 2]))], fill)
}

fn open_at(d: &Path, pool: &Arc<TensorBlocks>, budget_gb: u64, arena: u64, probe: u64) -> PagedBlockStore {
    let store = PagedBlockStore::open(d.to_path_buf(), budget_gb, arena).unwrap();
    store.bind(Arc::clone(pool) as Arc<dyn KvBlockMemory>, &pool.layout, probe);
    store
}

fn open(d: &Path, pool: &Arc<TensorBlocks>) -> PagedBlockStore {
    open_at(d, pool, 4, ARENA, PROBE)
}

/// A cold process: a fresh store over `d`, bound to a zeroed pool.
fn restart(d: &Path) -> (PagedBlockStore, Arc<TensorBlocks>) {
    let pool = synth_pool(|_, _| 0.0);
    (open(d, &pool), pool)
}

fn chain_meta(hashes: &[u64], block_ids: &[usize]) -> Vec<SealedBlockMeta> {
    let parent = |i: usize| i.checked_sub(1).map_or(0, |p| hashes[p]);
    (0..hashes.len())
        .map(|i| SealedBlockMeta { hash: hashes[i], parent: parent(i), pos: i as u32, block_id: block_ids[i] })
        .collect()
}

/// A chain with no boundary records, through the writer.
fn store_chain(store: &PagedBlockStore, hashes: &[u64], block_ids: &[usize]) {
    store.note_finished_sequence(&chain_meta(hashes, block_ids), vec![], &[]);
    store.flush(FLUSH);
}

/// `hashes` through `ids` with `tails`, flushed, the process gone.
fn stored(d: &Path, pool: &Arc<TensorBlocks>, hashes: &[u64], ids: &[usize], tails: Vec<TailDeposit>) {
    let store = open(d, pool);
    store.note_finished_sequence(&chain_meta(hashes, ids), tails, &[]);
    store.flush(FLUSH);
}

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join("oscar_block_store_tests").join(name);
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn block_bytes(pool: &Arc<TensorBlocks>, block_id: usize) -> Vec<f32> {
    pool.blocks[block_id].to_vec1::<f32>().unwrap()
}

/// Each `(s, t)`: block `s` of `pool` as block `t` of `pool2` holds it.
fn assert_blocks(pool: &Arc<TensorBlocks>, pool2: &Arc<TensorBlocks>, ids: impl IntoIterator<Item = (usize, usize)>) {
    ids.into_iter().for_each(|(s, t)| assert_eq!(block_bytes(pool, s), block_bytes(pool2, t), "block {s} changed"));
}

/// Rows `0..n` of each layer half of a block, on the block axis.
fn tail_rows(pool: &Arc<TensorBlocks>, block_id: usize, n: usize) -> Vec<f32> {
    block_bytes(pool, block_id).chunks(BLOCK).flat_map(|row| row[..n].to_vec()).collect()
}

/// One layer's state, its conv `0..8` and its recurrence `9..25`, at `off`.
fn state_at(off: usize) -> RecurrentState {
    let dev = metal();
    let conv = Tensor::arange(0f32, 8f32, &dev).unwrap().reshape((1, 2, 4)).unwrap();
    let rec = Tensor::arange(9f32, 25f32, &dev).unwrap().reshape((4, 4)).unwrap();
    RecurrentState::from_layers(vec![(conv, rec)], off)
}

/// The state `state_at` made, read back exactly, resuming at `depth`.
fn assert_state(state: Option<RecurrentState>, depth: usize) {
    let state = state.expect("a state restored");
    assert_eq!(state.seqlen_offset(), depth);
    let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let (conv, rec) = &state.layers()[0];
    assert_eq!(flat(conv), (0..8).map(|i| i as f32).collect::<Vec<_>>(), "the conv state read back exactly");
    assert_eq!(flat(rec), (9..25).map(|i| i as f32).collect::<Vec<_>>(), "the recurrence read back exactly");
}

/// A boundary record at `end`, riding the full block below it.
fn boundary(hashes: &[u64], toks: &[u32], end: usize, block_id: usize) -> TailDeposit {
    let pos = end / BLOCK;
    TailDeposit {
        parent: hashes[pos - 1],
        pos: pos as u32,
        block_id,
        tokens: toks[pos * BLOCK..end].to_vec(),
        snapshots: Some(state_at(end)),
    }
}

/// A turn's chain through `block_ids` and its pair: the reply end, then the prompt end, each `(end, block)`.
fn turn(
    store: &PagedBlockStore, h: &[u64], block_ids: &[usize], toks: &[u32], reply: (usize, usize),
    prompt: (usize, usize),
) {
    let pair = vec![boundary(h, toks, reply.0, reply.1), boundary(h, toks, prompt.0, prompt.1)];
    store.note_finished_sequence(&chain_meta(&h[..block_ids.len()], block_ids), pair, &toks[..reply.0]);
}

/// Turn 1: prompt to 5 tokens into block 1, reply to 9 into 2; turn 2 continues it verbatim, to 4 into 3 and 7 into 4.
fn two_turns() -> (Vec<u64>, Vec<u32>) {
    let hashes: Vec<u64> = (1..=4).map(|i| 0x5100 + i).collect();
    let toks: Vec<u32> = (0..(4 * BLOCK + 7) as u32).map(|i| 1000 + i).collect();
    (hashes, toks)
}

/// Turn 1 of `two_turns` deposited over `d`, the pool filled with `salt`.
fn after_turn_one(name: &str, salt: usize) -> (PathBuf, Arc<TensorBlocks>, PagedBlockStore, Vec<u64>, Vec<u32>) {
    let d = dir(name);
    let pool = synth_pool(move |l, i| (l + i % salt) as f32);
    let (h, toks) = two_turns();
    let store = open(&d, &pool);
    turn(&store, &h, &[0, 1], &toks, (2 * BLOCK + 9, 2), (BLOCK + 5, 1));
    (d, pool, store, h, toks)
}

/// Every record file in `d`.
fn on_disk(d: &Path) -> Vec<RecordMeta> {
    record::sweep(d).0.into_iter().map(|(_, meta, _)| meta).collect()
}

fn blocks_on_disk(d: &Path) -> usize {
    on_disk(d).iter().filter(|r| matches!(r.kind, RecordKind::Block)).count()
}

/// Each boundary record's `(pos, tokens)`, sorted.
fn tails_on_disk(d: &Path) -> Vec<(u64, usize)> {
    let tail = |r: &RecordMeta| match &r.kind {
        RecordKind::Tail { tokens, .. } => Some((r.pos as u64, tokens.len())),
        RecordKind::Block => None,
    };
    let mut t: Vec<(u64, usize)> = on_disk(d).iter().filter_map(tail).collect();
    t.sort();
    t
}

/// The writer takes a turn's blocks and holds its pair: `n` blocks on disk within the flush interval.
fn blocks_written(d: &Path, n: usize) {
    let t0 = std::time::Instant::now();
    while blocks_on_disk(d) < n {
        assert!(t0.elapsed() < FLUSH, "the writer never wrote the turn's blocks");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// By mincore(2) over a read-only mapping of each of `dir`'s files, which faults nothing in.
fn cached_pages(dir: &Path) -> usize {
    use std::os::fd::AsRawFd;
    // SAFETY: sysconf has no preconditions.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let mut resident = 0;
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let f = std::fs::File::open(e.path()).unwrap();
        let len = f.metadata().unwrap().len() as usize;
        if len == 0 {
            continue;
        }
        let mut vec = vec![0 as libc::c_char; len.div_ceil(page)];
        // SAFETY: a read-only shared mapping of a live descriptor, unmapped here; `vec` holds a byte a page.
        unsafe {
            let addr = libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, f.as_raw_fd(), 0);
            assert_ne!(addr, libc::MAP_FAILED);
            assert_eq!(libc::mincore(addr, len, vec.as_mut_ptr()), 0);
            libc::munmap(addr, len);
        }
        resident += vec.iter().filter(|&&b| b & 1 != 0).count();
    }
    resident
}

/// A first bind finding rows where no record puts them leaves the store off; another hash-function fingerprint
/// loses everything; a restart sweeps unpublished and old-format files. A conversation survives a restart byte for
/// byte, uncached; each block's KV is on disk once; a query restores to its first unknown hash.
#[test]
fn the_store_refuses_what_it_cannot_read_and_keeps_a_conversation_byte_for_byte_uncached() {
    let d = dir("refused_bind");
    let pool = synth_pool(|l, i| (l * 7 + i % 97) as f32);
    let hashes: Vec<u64> = (1..=2).map(|i| 0xB000 + i).collect();
    let store = PagedBlockStore::open(d.clone(), 4, ARENA).unwrap();
    let mut moved = pool.layout.clone();
    moved.rows[2].as_mut().expect("layer 2 bears KV").k_offset += 4;
    [&moved, &pool.layout].into_iter().for_each(|l| store.bind(Arc::clone(&pool) as Arc<dyn KvBlockMemory>, l, PROBE));
    store_chain(&store, &hashes, &[3, 5]);
    drop(store);
    let (store, _) = restart(&d);
    let restored = store.restore_extension(&hashes, 0, &[1, 2], false, 0, &[]);
    assert!(restored.is_none(), "the refused store deposited nothing");
    let d = dir("keysweep");
    let hashes: Vec<u64> = vec![0xF001, 0xF002];
    store_chain(&open(&d, &pool), &hashes, &[0, 1]);
    let store = open_at(&d, &pool, 4, ARENA, PROBE ^ 1);
    assert!(store.restore_extension(&hashes, 0, &[2, 3], false, 0, &[]).is_none());
    let names = |d: &Path| -> Vec<String> {
        std::fs::read_dir(d).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect()
    };
    assert!(!names(&d).iter().any(|n| n.ends_with(".okv")), "old-key records deleted");
    let d = dir("sweep");
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(d.join("0000000000000007.tmp"), b"half-written").unwrap();
    std::fs::write(d.join("ext00000003.oxt"), b"old format").unwrap();
    std::fs::write(d.join("e123_1.json"), b"{}").unwrap();
    // A record of the previous format version: gone as the rest.
    let trailer = [b"{}".as_slice(), &2u64.to_le_bytes(), &3u32.to_le_bytes(), b"OSKR"].concat();
    std::fs::write(d.join("0000000000000003.okv"), [&[0u8; 64], trailer.as_slice()].concat()).unwrap();
    let _store = PagedBlockStore::open(d.clone(), 4, ARENA).unwrap();
    let left = names(&d);
    assert!(left.is_empty(), "unpublished and old-format files swept, found {left:?}");
    // Neither the writes nor the restore leave a page of the store's in the page cache.
    let d = dir("restart");
    let pool = synth_pool(|l, i| (l * 7 + i % 97) as f32);
    let hashes: Vec<u64> = (1..=4).map(|i| 0xA000 + i).collect();
    let (src, dst) = ([3usize, 5, 0, 6], [1usize, 2, 4, 7]);
    store_chain(&open(&d, &pool), &hashes, &src);
    assert_eq!(cached_pages(&d), 0, "the writes left the store's pages in the page cache");
    let (store, pool2) = restart(&d);
    let (depth, snaps) = store.restore_extension(&hashes, 0, &dst, false, 0, &[]).expect("stored chain restores");
    assert_eq!((depth, snaps.is_none()), (4 * BLOCK, true));
    assert_blocks(&pool, &pool2, src.into_iter().zip(dst));
    assert_eq!(cached_pages(&d), 0, "the restore left the store's pages in the page cache");
    let d = dir("prefix");
    let store = open(&d, &pool);
    let stored: Vec<u64> = vec![0xB001, 0xB002, 0xB003];
    store_chain(&store, &stored[..2], &[0, 1]);
    store_chain(&store, &stored, &[0, 1, 2]);
    let mut hashes: Vec<u64> = on_disk(&d).iter().map(|r| r.hash).collect();
    hashes.sort();
    assert_eq!(hashes, stored, "each block's KV on disk once");
    // Query diverges after two blocks: only those restore.
    let query: Vec<u64> = vec![0xB001, 0xB002, 0xDEAD, 0xBEEF];
    let (depth, _) = store.restore_extension(&query, 0, &[4, 5, 6, 7], false, 0, &[]).unwrap();
    assert_eq!(depth, 2 * BLOCK, "the walk stops at the first unknown hash");
    // A RAM hit below the stored depth extends to the stored depth only; fully covered by RAM, nothing to restore.
    let (depth, _) = store.restore_extension(&stored, 1, &[6, 7], false, 0, &[]).unwrap();
    assert_eq!(depth, 3 * BLOCK);
    assert!(store.restore_extension(&stored, 3, &[], false, 0, &[]).is_none());
}

/// Across a restart, whatever blocks the store holds. RAM holding the blocks and no state: the stored state at or
/// below the match resumes, state only, unless a RAM anchor is at least as deep; a zero disk budget peels everything.
#[test]
fn a_hybrid_resumes_only_where_a_boundary_record_holds_its_state() {
    let d = dir("hybrid");
    let pool = synth_pool(|l, i| (l + i % 29) as f32);
    let hashes: Vec<u64> = vec![0xD001, 0xD002, 0xD003];
    let toks = vec![1u32; 3 * BLOCK + 4];
    // A state-only record at depth 2; block 3 has KV but no state.
    stored(&d, &pool, &hashes, &[0, 1, 2], vec![boundary(&hashes, &toks, 2 * BLOCK, 0)]);
    let (store, _) = restart(&d);
    let (depth, restored) = store.restore_extension(&hashes, 0, &[4, 5, 6], true, 0, &toks).expect("hybrid restore");
    assert_eq!(depth, 2 * BLOCK, "a hybrid resumes where its state is, not at the deepest block");
    assert_state(restored, 2 * BLOCK);
    // Without a stored state a hybrid restore declines outright.
    let bare: Vec<u64> = vec![0xE001, 0xE002];
    store_chain(&store, &bare, &[0, 1]);
    assert!(store.restore_extension(&bare, 0, &[4, 5], true, 0, &toks).is_none());
    // (blocks in RAM, the query's tokens, the records' (end, block), the depth resumed): a block-aligned prompt end,
    // a block-aligned reply end over a prompt boundary, a prompt boundary inside the last RAM block.
    let cases: [(usize, usize, &[(usize, usize)], usize); 3] = [
        (2, 2 * BLOCK + 5, &[(2 * BLOCK, 0)], 2 * BLOCK),
        (3, 3 * BLOCK + 9, &[(3 * BLOCK, 0), (2 * BLOCK + 5, 2)], 3 * BLOCK),
        (3, 3 * BLOCK + 2, &[(2 * BLOCK + 5, 2)], 2 * BLOCK + 5),
    ];
    for (i, (ram, len, records, depth)) in cases.into_iter().enumerate() {
        let d = dir(&format!("state_only_{i}"));
        let pool = synth_pool(|l, i| (l + i % 19) as f32);
        let hashes: Vec<u64> = (1..=ram as u64).map(|b| 0x9100 + b).collect();
        let toks = vec![4u32; len];
        let store = open(&d, &pool);
        let tails = records.iter().map(|&(end, block)| boundary(&hashes, &toks, end, block)).collect();
        store.note_finished_sequence(&chain_meta(&hashes, &[0, 1, 2][..ram]), tails, &[]);
        let (restored, snaps) =
            store.restore_extension(&hashes, ram, &[], true, 0, &toks).expect("a state-only restore");
        assert_eq!(restored, depth, "case {i} resumes at the stored state's depth");
        assert_state(snaps, depth);
        assert!(store.restore_extension(&hashes, ram, &[], true, depth, &toks).is_none(), "case {i}: the anchor wins");
    }
    let pool = synth_pool(|l, i| (l + i % 7) as f32);
    let store = open_at(&dir("peel"), &pool, 0, ARENA, PROBE);
    let hashes: Vec<u64> = vec![0xAB01, 0xAB02, 0xAB03];
    store_chain(&store, &hashes, &[0, 1, 2]);
    assert!(store.restore_extension(&hashes, 0, &[4, 5, 6], false, 0, &[]).is_none());
}

/// A 56 KB arena takes three to six of twelve 4 KB block records at a time (a run's pages and its alignment page):
/// the restore reads them in runs, the boundary record in a run of its own; a 20 KB arena stages a deposit's eight
/// in runs, none deferred; a held pair (kept till superseded or forced) is wrapped around, every block whole.
#[test]
fn a_match_or_a_deposit_larger_than_the_arena_moves_whole_in_runs_and_a_held_pair_is_wrapped_around() {
    let d = dir("small_arena");
    let pool = synth_pool(|l, i| (l * 3 + i % 61) as f32);
    let hashes: Vec<u64> = (1..=12).map(|i| 0xD100 + i).collect();
    let ids: Vec<usize> = (0..13).collect();
    let query: Vec<u32> = [vec![0; 12 * BLOCK], (900..907).collect(), vec![1, 2, 3]].concat();
    stored(&d, &pool, &hashes, &ids[..12], vec![boundary(&hashes, &query, 12 * BLOCK + 7, 12)]);
    let pool2 = synth_pool(|_, _| 0.0);
    let store = open_at(&d, &pool2, 4, 56 << 10, PROBE);
    let (restored, snaps) = store.restore_extension(&hashes, 0, &ids, true, 0, &query).expect("a restore in runs");
    assert_eq!(restored, 12 * BLOCK + 7, "every block and the boundary record, through an arena that holds a few");
    assert_state(snaps, 12 * BLOCK + 7);
    assert_blocks(&pool, &pool2, (0..12).map(|b| (b, b)));
    assert_eq!(tail_rows(&pool, 12, 7), tail_rows(&pool2, 12, 7), "the tail's rows in the run of their own");
    let d = dir("small_arena_deposit");
    let pool = synth_pool(|l, i| (l * 5 + i % 59) as f32);
    let hashes: Vec<u64> = (1..=8).map(|i| 0xD200 + i).collect();
    let ids: Vec<usize> = (0..8).collect();
    store_chain(&open_at(&d, &pool, 4, 20 << 10, PROBE), &hashes, &ids);
    assert_eq!(blocks_on_disk(&d), 8, "every block written, in two runs");
    // Read back through an arena that holds the chain: the deposit's runs are what this test is about.
    let (store2, pool2) = restart(&d);
    let (restored, _) = store2.restore_extension(&hashes, 0, &ids, false, 0, &[]).expect("a whole restore");
    assert_eq!(restored, 8 * BLOCK);
    assert_blocks(&pool, &pool2, (0..8).map(|b| (b, b)));
    let d = dir("wrap_held_pair");
    let pool = synth_pool(|l, i| (l * 11 + i % 67) as f32);
    let store = open_at(&d, &pool, 4, 40 << 10, PROBE);
    // Turn A: two blocks and its pair; the writer takes the blocks and holds the pair.
    let a: Vec<u64> = vec![0xE101, 0xE102];
    let a_toks: Vec<u32> = (0..(2 * BLOCK + 9) as u32).collect();
    turn(&store, &a, &[0, 1], &a_toks, (2 * BLOCK + 9, 2), (2 * BLOCK + 5, 2));
    blocks_written(&d, 2);
    // Chain B: thirteen 4 KB records through the 40 KB ring, around the held pair.
    let b: Vec<u64> = (1..=13).map(|i| 0xE200 + i).collect();
    let ids: Vec<usize> = (3..16).collect();
    store_chain(&store, &b, &ids);
    drop(store);
    let (store2, pool2) = restart(&d);
    let (restored, _) = store2.restore_extension(&b, 0, &ids, false, 0, &[]).expect("chain B whole");
    assert_eq!(restored, 13 * BLOCK);
    assert_blocks(&pool, &pool2, ids.iter().map(|&id| (id, id)));
    let follow = [a_toks, vec![1, 2, 3]].concat();
    let (at, snaps) = store2.restore_extension(&a, 0, &[0, 1, 2], true, 0, &follow).expect("A's pair");
    assert_eq!(at, 2 * BLOCK + 9, "turn A's reply end survives the wrap");
    assert_state(snaps, 2 * BLOCK + 9);
}

/// A latent row (the block axis first, as the MLA rows have it) binds and its boundary record restores as a standard
/// row's does: the record is the whole block, so every row of it survives the restart. The live state carries no
/// position, yet the record is valid at its boundary. A non-hybrid tail matches its longest token prefix.
#[test]
fn a_boundary_record_restores_the_exact_prompt_boundary_across_a_restart() {
    let d = dir("tail");
    let pool = synth_pool(|l, i| (l * 5 + i % 61) as f32);
    let hashes: Vec<u64> = vec![0x7001, 0x7002];
    // The prompt ran 2 blocks + 7 tokens into block 2; the follow-up shares them and brings 3 new ones.
    let query: Vec<u32> = [vec![0; 2 * BLOCK], (900..907).collect(), vec![1, 2, 3]].concat();
    stored(&d, &pool, &hashes, &[0, 1], vec![boundary(&hashes, &query, 2 * BLOCK + 7, 2)]);
    let (store, pool2) = restart(&d);
    let (restored, snaps) = store.restore_extension(&hashes, 0, &[4, 5, 6], true, 0, &query).expect("tail restore");
    assert_eq!(restored, 2 * BLOCK + 7, "resumes at the exact prompt boundary");
    assert_state(snaps, 2 * BLOCK + 7);
    assert_eq!(tail_rows(&pool, 2, 7), tail_rows(&pool2, 6, 7), "partial-block rows survive the restart exactly");
    // A follow-up whose tail token differs must not install the record.
    let mut wrong = query.clone();
    wrong[2 * BLOCK + 3] ^= 1;
    let restore = |q: &[u32]| store.restore_extension(&hashes, 0, &[4, 5, 6], true, 0, q);
    assert!(restore(&wrong).is_none(), "an element-wise mismatch falls back (hybrid: no state, no restore)");
    assert!(restore(&query[..2 * BLOCK + 7]).is_none(), "at least one token must remain to re-prefill for logits");
    let d = dir("latent_tail");
    let pool = latent_pool(|l, i| (l * 7 + i % 53) as f32);
    let hashes: Vec<u64> = vec![0x7101, 0x7102];
    let query: Vec<u32> = [vec![0; 2 * BLOCK], (900..907).collect(), vec![1, 2, 3]].concat();
    stored(&d, &pool, &hashes, &[0, 1], vec![boundary(&hashes, &query, 2 * BLOCK + 7, 2)]);
    let pool2 = latent_pool(|_, _| 0.0);
    let store = open(&d, &pool2);
    let (restored, snaps) = store.restore_extension(&hashes, 0, &[4, 5, 6], true, 0, &query).expect("tail restore");
    assert_eq!(restored, 2 * BLOCK + 7, "resumes at the exact prompt boundary");
    assert_state(snaps, 2 * BLOCK + 7);
    assert_eq!(block_bytes(&pool, 2), block_bytes(&pool2, 6), "the boundary's block survives the restart whole");
    assert!(block_bytes(&pool, 2)[2 * ROW..].iter().any(|&v| v != 0.0), "the latent rows carry data");
    let d = dir("tail-live");
    let pool = synth_pool(|l, i| (l * 3 + i % 53) as f32);
    let hashes: Vec<u64> = vec![0x7101, 0x7102];
    let query: Vec<u32> = [vec![0; 2 * BLOCK], (700..711).collect(), vec![1]].concat();
    let live = TailDeposit { snapshots: Some(state_at(0)), ..boundary(&hashes, &query, 2 * BLOCK + 11, 2) };
    let store = open(&d, &pool);
    store.note_finished_sequence(&chain_meta(&hashes, &[0, 1]), vec![live], &[]);
    let (restored, snaps) = store.restore_extension(&hashes, 0, &[4, 5, 6], true, 0, &query).expect("tail restore");
    assert_eq!(restored, 2 * BLOCK + 11, "resumes at the reply end");
    assert_state(snaps, 2 * BLOCK + 11);
    let d = dir("tail_prefix");
    let pool = synth_pool(|l, i| (l * 2 + i % 41) as f32);
    let hashes: Vec<u64> = vec![0x7301, 0x7302];
    let store = open(&d, &pool);
    let tail = TailDeposit { parent: hashes[1], pos: 2, block_id: 3, tokens: (700..707).collect(), snapshots: None };
    store.note_finished_sequence(&chain_meta(&hashes, &[0, 1]), vec![tail], &[]);
    // A re-render diverging after 4 of the 7 stored tokens: a non-hybrid restore takes the matched rows.
    let query: Vec<u32> = [vec![0; 2 * BLOCK], vec![700, 701, 702, 703, 999, 1, 2, 3]].concat();
    let (restored, snaps) =
        store.restore_extension(&hashes, 0, &[4, 5, 6], false, 0, &query).expect("prefix tail restore");
    assert_eq!((restored, snaps.is_none()), (2 * BLOCK + 4, true), "resumes at the divergence point");
    // The tip record resumes a verbatim follow-up at the reply end.
    let d = dir("tip_record");
    let pool = synth_pool(|l, i| (l + i % 31) as f32);
    let hashes: Vec<u64> = vec![0xA201, 0xA202, 0xA203];
    // Prompt to 2 blocks + 5, reply to 3 blocks + 9: both boundary records.
    let toks = vec![6u32; 3 * BLOCK + 9];
    let store = open(&d, &pool);
    turn(&store, &hashes, &[0, 1, 2], &toks, (3 * BLOCK + 9, 3), (2 * BLOCK + 5, 2));
    // A verbatim follow-up resumes at the tip, its partial rows installed into the given block.
    let query = [toks, vec![1, 2, 3]].concat();
    let (restored, snaps) = store.restore_extension(&hashes, 3, &[6], true, 0, &query).expect("tip restore");
    assert_eq!(restored, 3 * BLOCK + 9, "resumes at the reply end");
    assert_state(snaps, 3 * BLOCK + 9);
    assert_eq!(tail_rows(&pool, 3, 9), tail_rows(&pool, 6, 9), "the reply's partial rows install for the follow-up");
    // A follow-up that rewrites the reply falls back to the prompt boundary.
    let mut rewritten = query;
    rewritten[3 * BLOCK + 4] ^= 1;
    let (restored, snaps) =
        store.restore_extension(&hashes, 3, &[7], true, 0, &rewritten).expect("prompt-boundary fallback");
    assert_eq!(restored, 2 * BLOCK + 5);
    assert_state(snaps, 2 * BLOCK + 5);
}

/// What resumed at the earlier turn's pair resumes at the later turn's records; a superseded pair already on disk
/// leaves with its file. A retry keeps the prompt boundary it repeats: the first reply's end goes, and a follow-up of
/// either reply resumes at the prompt boundary or deeper. A conversation sharing only a block supersedes nothing.
#[test]
fn a_turn_supersedes_the_turn_it_continues_and_no_other() {
    let (d, _, store, h, toks) = after_turn_one("supersede", 23);
    blocks_written(&d, 2);
    assert!(tails_on_disk(&d).is_empty(), "the pair waits staged");
    turn(&store, &h, &[0, 1, 2, 3], &toks, (4 * BLOCK + 7, 4), (3 * BLOCK + 4, 3));
    store.flush(FLUSH);
    assert_eq!(tails_on_disk(&d), vec![(3, 4), (4, 7)], "only the later turn's pair was written");
    // A follow-up of turn 2 resumes at its reply end; one parting from turn 2 inside turn 1's reply finds nothing.
    let follow = [toks.clone(), vec![1, 2, 3]].concat();
    let (restored, _) = store.restore_extension(&h, 0, &[4, 5, 6, 7, 0], true, 0, &follow).expect("turn 2's reply end");
    assert_eq!(restored, 4 * BLOCK + 7);
    let parted = [toks[..2 * BLOCK + 9].to_vec(), vec![7; 40]].concat();
    assert!(store.restore_extension(&h[..2], 0, &[4, 5, 6], true, 0, &parted).is_none());
    // A restart finds the same.
    drop(store);
    let (store, _) = restart(&d);
    let (restored, snaps) =
        store.restore_extension(&h, 0, &[4, 5, 6, 7, 0], true, 0, &follow).expect("after a restart");
    assert_eq!(restored, 4 * BLOCK + 7);
    assert_state(snaps, 4 * BLOCK + 7);
    // The superseded pair on disk: gone with its file at the next turn, the new pair held until the flush.
    let (d, pool, store, h, toks) = after_turn_one("supersede_disk", 31);
    store.flush(FLUSH);
    drop(store);
    assert_eq!(tails_on_disk(&d), vec![(1, 5), (2, 9)]);
    let store = open(&d, &pool);
    turn(&store, &h, &[0, 1, 2, 3], &toks, (4 * BLOCK + 7, 4), (3 * BLOCK + 4, 3));
    assert!(tails_on_disk(&d).is_empty(), "the superseded pair's file is gone, the new pair held");
    store.flush(FLUSH);
    assert_eq!(tails_on_disk(&d), vec![(3, 4), (4, 7)]);
    // The retry.
    let (d, _, store, h, toks) = after_turn_one("retry", 37);
    // The retry's reply differs from the first token on: a different block 1.
    let retry = [toks[..BLOCK + 5].to_vec(), (9000..9000 + BLOCK as u32 + 3).collect()].concat();
    let rh = [h[0], 0x5200];
    turn(&store, &rh, &[0, 5], &retry, (2 * BLOCK + 8, 6), (BLOCK + 5, 5));
    store.flush(FLUSH);
    assert_eq!(tails_on_disk(&d), vec![(1, 5), (2, 8)], "the prompt boundary once, the retry's reply end");
    let first = [toks[..2 * BLOCK + 9].to_vec(), vec![1, 2, 3]].concat();
    let (restored, _) = store.restore_extension(&h[..2], 0, &[4, 5, 6], true, 0, &first).expect("the prompt boundary");
    assert_eq!(restored, BLOCK + 5, "the first reply's end went with its turn");
    let second = [retry, vec![1, 2, 3]].concat();
    let (restored, _) = store.restore_extension(&rh, 0, &[4, 6, 7], true, 0, &second).expect("the retry's reply end");
    assert_eq!(restored, 2 * BLOCK + 8);
    // Another conversation.
    let (d, _, store, h, toks) = after_turn_one("two_conversations", 41);
    let other = [toks[..BLOCK].to_vec(), (7000..7000 + BLOCK as u32 + 6).collect()].concat();
    let oh = [h[0], 0x5300];
    turn(&store, &oh, &[0, 5], &other, (2 * BLOCK + 6, 6), (BLOCK + 2, 5));
    store.flush(FLUSH);
    assert_eq!(tails_on_disk(&d), vec![(1, 2), (1, 5), (2, 6), (2, 9)], "both pairs");
    let (restored, _) =
        store.restore_extension(&h[..2], 0, &[4, 6, 7], true, 0, &first).expect("the first conversation");
    assert_eq!(restored, 2 * BLOCK + 9);
    // A fork's record no turn owns, which a later turn of the stored conversation never supersedes.
    let d = dir("fork");
    let pool = synth_pool(|l, i| (l + i % 43) as f32);
    // Conversation A: three blocks, its pair past the third.
    let a: Vec<u64> = vec![0x6101, 0x6102, 0x6103];
    let a_toks: Vec<u32> = (0..(3 * BLOCK + 9) as u32).collect();
    let store = open(&d, &pool);
    turn(&store, &a, &[0, 1, 2], &a_toks, (3 * BLOCK + 9, 3), (3 * BLOCK + 2, 3));
    // Prompt B shares the first two blocks of A and parts in the third.
    let b: Vec<u64> = vec![0x6101, 0x6102, 0x6203];
    let b_toks = [a_toks[..2 * BLOCK].to_vec(), (5000..5000 + BLOCK as u32 + 4).collect()].concat();
    assert_eq!(store.fork_depth(&b, &b_toks), Some(2), "B parts from A after two blocks");
    // A prompt continuing the chain of A parts from nothing; nor does a follow-up re-rendering the reply of A.
    let a_more = [a_toks.clone(), (1..40).collect()].concat();
    assert_eq!(store.fork_depth(&[a[0], a[1], a[2], 0x6404], &a_more), None);
    let rerendered = [a_toks[..3 * BLOCK + 2].to_vec(), (4000..4007).collect(), (1..60).collect()].concat();
    assert_eq!(store.fork_depth(&[a[0], a[1], a[2], 0x6504], &rerendered), None, "a re-render is no branch");
    // B keeps the branch's state; a third prompt parting there resumes at it.
    store.note_shared(&chain_meta(&b[..2], &[0, 1]), boundary(&b, &b_toks, 2 * BLOCK, 0));
    let c: Vec<u64> = vec![0x6101, 0x6102, 0x6303];
    let c_toks = [a_toks[..2 * BLOCK].to_vec(), (6000..6000 + BLOCK as u32 + 4).collect()].concat();
    let (restored, snaps) = store.restore_extension(&c, 0, &[4, 5, 6], true, 0, &c_toks).expect("the fork's state");
    assert_eq!(restored, 2 * BLOCK);
    assert_state(snaps, 2 * BLOCK);
    // A later turn of A supersedes the pair of A and not the branch's record.
    let a2: Vec<u64> = vec![0x6101, 0x6102, 0x6103, 0x6104];
    let a2_toks: Vec<u32> = (0..(4 * BLOCK + 6) as u32).collect();
    turn(&store, &a2, &[0, 1, 2, 3], &a2_toks, (4 * BLOCK + 6, 4), (4 * BLOCK + 1, 4));
    store.flush(FLUSH);
    assert_eq!(tails_on_disk(&d), vec![(2, 0), (4, 1), (4, 6)], "the fork's record and A's latest pair");
    let (restored, _) = store.restore_extension(&c, 0, &[4, 5, 6], true, 0, &c_toks).expect("still the fork's state");
    assert_eq!(restored, 2 * BLOCK);
}
