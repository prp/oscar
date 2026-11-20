//! The in-memory block index over the staged and sealed tiers, rebuilt from the record files alone.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::path::PathBuf;

use super::arena::ArenaRange;
use super::record::{RecordKind, RecordMeta, SnapMeta};

/// A sealed record's file and its payload's length.
#[derive(Clone, Debug)]
pub(crate) struct DiskLoc {
    pub path: PathBuf,
    pub len: u64,
}

/// Staged in the arena until the writer seals the record into its file.
#[derive(Clone, Debug)]
pub(crate) enum Tier {
    Staged(ArenaRange),
    Disk(DiskLoc),
}

impl Tier {
    pub fn disk(&self) -> Option<&DiskLoc> {
        match self {
            Tier::Disk(loc) => Some(loc),
            Tier::Staged(_) => None,
        }
    }
}

/// A branch's group: every later conversation sharing the prefix resumes at its state, and no turn supersedes it.
pub(crate) const SHARED: u64 = 0;

/// Partial rows matched element-wise by `tokens`, and the state; dies with its parent block, a turn's with its group.
pub(crate) struct TailState {
    /// The depositing turn, or [`SHARED`].
    pub group: u64,
    /// Block position of the partial rows (= parent position + 1).
    pub pos: u32,
    pub tokens: Vec<u32>,
    pub meta: SnapMeta,
    pub tier: Tier,
}

pub(crate) struct BlockRec {
    pub parent: u64,
    pub tier: Tier,
    pub kids_on_disk: u32,
    pub last_access: u64,
    pub tails: Vec<TailState>,
}

impl BlockRec {
    pub fn on_disk(&self) -> bool {
        self.tier.disk().is_some()
    }

    /// Position and partial rows name a record under its parent, as several tips can share one.
    pub fn tail(&self, pos: u32, tokens: &[u32]) -> Option<&TailState> {
        self.tails.iter().find(|t| t.pos == pos && t.tokens == tokens)
    }
}

#[derive(Default)]
pub(crate) struct BlockIndex {
    pub map: HashMap<u64, BlockRec>,
    /// Coldest first, lazily invalidated: a block is evictable exactly when no on-disk child chains from it.
    evictable: BinaryHeap<Reverse<(u64, u64)>>,
    /// Persisted per record, as its `last_access`.
    clock: u64,
    pub disk_bytes: u64,
}

impl BlockIndex {
    pub fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    pub fn touch(&mut self, hash: u64) {
        let now = self.tick();
        if let Some(rec) = self.map.get_mut(&hash) {
            rec.last_access = now;
            if rec.on_disk() && rec.kids_on_disk == 0 {
                self.evictable.push(Reverse((now, hash)));
            }
        }
    }

    /// A block the deposit found absent.
    pub fn insert_staged(&mut self, hash: u64, parent: u64, payload: ArenaRange) {
        let last_access = self.tick();
        let rec = BlockRec { parent, tier: Tier::Staged(payload), kids_on_disk: 0, last_access, tails: Vec::new() };
        self.map.insert(hash, rec);
    }

    pub fn attach_staged_tail(
        &mut self, parent: u64, group: u64, pos: u32, tokens: Vec<u32>, meta: SnapMeta, payload: ArenaRange,
    ) {
        let now = self.tick();
        let Some(rec) = self.map.get_mut(&parent) else {
            return;
        };
        rec.last_access = now;
        rec.tails.push(TailState { group, pos, tokens, meta, tier: Tier::Staged(payload) });
    }

    /// The turns a deposit continues: groups with a record on its chain at or below `upto` tokens.
    pub fn groups_on_chain(&self, chain: &[u64], toks: &[u32], block_size: usize, upto: usize) -> Vec<u64> {
        let mut out = Vec::new();
        for pos in 1..=chain.len().min(upto / block_size) {
            let Some(rec) = self.map.get(&chain[pos - 1]) else { break };
            let start = pos * block_size;
            for t in rec.tails.iter().filter(|t| t.pos as usize == pos && t.group != SHARED) {
                let end = start + t.tokens.len();
                if end <= upto && toks.get(start..end) == Some(t.tokens.as_slice()) && !out.contains(&t.group) {
                    out.push(t.group);
                }
            }
        }
        out
    }

    /// A record re-deposited unchanged, as a retry's prompt boundary, joins `group` uncopied; false when none is held.
    pub fn adopt(&mut self, parent: u64, pos: u32, tokens: &[u32], group: u64) -> bool {
        let Some(t) = self
            .map
            .get_mut(&parent)
            .and_then(|r| r.tails.iter_mut().find(|t| t.pos == pos && t.tokens == tokens && t.group != SHARED))
        else {
            return false;
        };
        t.group = group;
        true
    }

    pub fn holds(&self, parent: u64, pos: u32, tokens: &[u32]) -> bool {
        self.map.get(&parent).is_some_and(|r| r.tail(pos, tokens).is_some())
    }

    /// The depth where a held sequence parts from the prompt; a prompt passing a turn's record continues it, no branch.
    pub fn fork_depth(&self, chain: &[u64], toks: &[u32], block_size: usize) -> Option<usize> {
        let depth = chain.iter().take_while(|h| self.map.contains_key(h)).count();
        let parent = *chain.get(depth.checked_sub(1)?)?;
        let start = depth * block_size;
        if !self.groups_on_chain(chain, toks, block_size, start + block_size).is_empty() {
            return None;
        }
        let parts = self.map.values().any(|r| r.parent == parent)
            || self.map[&parent].tails.iter().any(|t| {
                t.pos as usize == depth && toks.get(start..start + t.tokens.len()) != Some(t.tokens.as_slice())
            });
        parts.then_some(depth)
    }

    /// Drops `group`'s records; returns their staged ranges to cancel.
    pub fn supersede(&mut self, group: u64) -> Vec<ArenaRange> {
        let gone: Vec<_> =
            self.map.values_mut().flat_map(|r| r.tails.extract_if(.., move |t| t.group == group)).collect();
        self.drop_tails(gone)
    }

    /// A staged record's range comes back to be cancelled; a sealed one's disk space is freed.
    fn drop_tails(&mut self, tails: impl IntoIterator<Item = TailState>) -> Vec<ArenaRange> {
        let mut staged = Vec::new();
        for t in tails {
            match t.tier {
                Tier::Staged(r) => staged.push(r),
                Tier::Disk(loc) => self.free_disk(loc),
            }
        }
        staged
    }

    /// A sealed record's file goes with it.
    fn free_disk(&mut self, loc: DiskLoc) {
        self.disk_bytes = self.disk_bytes.saturating_sub(loc.len);
        let _ = std::fs::remove_file(&loc.path);
    }

    /// False when nothing live is the record's, as a superseded turn's.
    pub fn mark_sealed(&mut self, rec_meta: &RecordMeta, loc: &DiskLoc) -> bool {
        let hash = rec_meta.hash;
        let parent = match &rec_meta.kind {
            RecordKind::Block => {
                let Some(rec) = self.map.get_mut(&hash) else { return false };
                rec.tier = Tier::Disk(loc.clone());
                Some(rec.parent)
            }
            RecordKind::Tail { tokens, .. } => {
                let pos = rec_meta.pos;
                let Some(tail) = self.map.get_mut(&hash).and_then(|rec| {
                    rec.tails
                        .iter_mut()
                        .find(|t| t.pos == pos && t.tokens == *tokens && matches!(t.tier, Tier::Staged(_)))
                }) else {
                    return false;
                };
                tail.tier = Tier::Disk(loc.clone());
                None
            }
        };
        self.disk_bytes += loc.len;
        // A block on disk counts under its parent and is evictable as a leaf.
        if let Some(parent) = parent {
            if let Some(p) = self.map.get_mut(&parent) {
                p.kids_on_disk += 1;
            }
            let rec = &self.map[&hash];
            if rec.kids_on_disk == 0 {
                self.evictable.push(Reverse((rec.last_access, hash)));
            }
        }
        true
    }

    /// The coldest suffixes first; returns the staged ranges of their records to cancel, as [`Self::supersede`] does.
    pub fn peel_to_budget(&mut self, budget: u64) -> Vec<ArenaRange> {
        let mut staged = Vec::new();
        while self.disk_bytes > budget {
            let Some(Reverse((stamp, hash))) = self.evictable.pop() else {
                break;
            };
            let Some(rec) = self.map.get(&hash) else { continue };
            // A stale entry: a live key was re-pushed when touched.
            if !rec.on_disk() || rec.kids_on_disk != 0 || rec.last_access != stamp {
                continue;
            }
            staged.extend(self.drop_block_from_disk(hash));
        }
        staged
    }

    /// A leaf block and its records, its parent then perhaps evictable; returns its records' staged ranges.
    fn drop_block_from_disk(&mut self, hash: u64) -> Vec<ArenaRange> {
        let Some(rec) = self.map.remove(&hash) else { return Vec::new() };
        if let Tier::Disk(loc) = rec.tier {
            self.free_disk(loc);
        }
        // Boundary records chain from this block, so they die with it.
        let staged = self.drop_tails(rec.tails);
        if let Some(p) = self.map.get_mut(&rec.parent) {
            p.kids_on_disk = p.kids_on_disk.saturating_sub(1);
            if p.kids_on_disk == 0 && p.on_disk() {
                self.evictable.push(Reverse((p.last_access, rec.parent)));
            }
        }
        staged
    }

    /// Blocks, then their boundary records, child counts and the eviction order, each at its persisted coldness; the
    /// clock passes every stamp and group. A boundary record without its parent block loses its file.
    pub fn rebuild(mut records: Vec<(PathBuf, RecordMeta, u64)>) -> Self {
        let mut ix = BlockIndex::default();
        records.sort_by_key(|(_, r, _)| matches!(r.kind, RecordKind::Tail { .. }));
        for (path, r, len) in records {
            let (loc, last_access) = (DiskLoc { path, len }, r.last_access);
            match r.kind {
                RecordKind::Block => {
                    let rec = BlockRec {
                        parent: r.parent,
                        tier: Tier::Disk(loc),
                        kids_on_disk: 0,
                        last_access,
                        tails: Vec::new(),
                    };
                    ix.map.insert(r.hash, rec);
                }
                RecordKind::Tail { group, tokens, meta } => {
                    let Some(rec) = ix.map.get_mut(&r.hash) else {
                        let _ = std::fs::remove_file(&loc.path);
                        continue;
                    };
                    rec.tails.push(TailState { group, pos: r.pos, tokens, meta, tier: Tier::Disk(loc) });
                    ix.clock = ix.clock.max(group);
                }
            }
            ix.disk_bytes += len;
            ix.clock = ix.clock.max(last_access);
        }
        let parents: Vec<u64> = ix.map.values().map(|r| r.parent).collect();
        for p in parents {
            if let Some(rec) = ix.map.get_mut(&p) {
                rec.kids_on_disk += 1;
            }
        }
        let leaves = ix.map.iter().filter(|(_, r)| r.kids_on_disk == 0);
        ix.evictable = leaves.map(|(h, r)| Reverse((r.last_access, *h))).collect();
        ix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(n: u64) -> ArenaRange {
        ArenaRange { off: 0, len: n }
    }

    fn geom() -> SnapMeta {
        SnapMeta { seqlen_offsets: vec![0], tensors: Vec::new() }
    }

    /// Block size 4, the chain 1 → 2 → 3; turn 7's prompt boundary 2 tokens past block 1, its reply end 3 past block 2.
    fn turn_seven() -> BlockIndex {
        let mut ix = BlockIndex::default();
        ix.insert_staged(1, 0, range(100));
        ix.insert_staged(2, 1, range(100));
        ix.attach_staged_tail(1, 7, 1, vec![5, 6], geom(), range(500));
        ix.attach_staged_tail(2, 7, 2, vec![8, 9, 10], geom(), range(500));
        ix
    }

    /// A follow-up finds the turn it continues whether or not it keeps the reply; a superseded turn leaves, and a retry
    /// keeps the record it repeats; a block deposited again counts once, and a peeled block gives back its staged
    /// records.
    #[test]
    fn a_follow_up_finds_the_turn_it_continues_and_a_superseded_turn_leaves() {
        let mut ix = turn_seven();
        let verbatim = [0, 0, 0, 0, 5, 6, 7, 7, 8, 9, 10, 1, 2];
        assert_eq!(ix.groups_on_chain(&[1, 2, 3], &verbatim, 4, 12), vec![7]);
        let rewritten = [0, 0, 0, 0, 5, 6, 7, 7, 8, 9, 99, 1, 2];
        assert_eq!(ix.groups_on_chain(&[1, 2, 3], &rewritten, 4, 12), vec![7], "the prompt boundary lies on it");
        // A branch inside block 1 changes every chain hash from block 1 on.
        let branched = [0, 0, 0, 0, 5, 99, 7, 7, 8, 9, 10, 1, 2];
        assert!(ix.groups_on_chain(&[1, 20, 30], &branched, 4, 12).is_empty(), "a branch behind the turn leaves it");
        let below = ix.groups_on_chain(&[1, 2, 3], &verbatim, 4, 5);
        assert!(below.is_empty(), "nothing at or below the deposit's boundary");
        assert!(ix.adopt(1, 1, &[5, 6], 8), "the retry repeats the prompt boundary");
        assert!(!ix.adopt(1, 1, &[5, 7], 8));
        let gone = ix.supersede(7);
        assert_eq!(gone.len(), 1, "the reply end's staged range comes back to be cancelled");
        let groups = |ix: &BlockIndex, block: u64| ix.map[&block].tails.iter().map(|t| t.group).collect::<Vec<_>>();
        assert_eq!((groups(&ix, 1), groups(&ix, 2)), (vec![8], vec![]));
        assert_eq!((ix.supersede(8).len(), ix.map[&1].tails.len()), (1, 0));
        let block = |hash, len| {
            let meta = RecordMeta { kind: RecordKind::Block, hash, parent: 0, pos: 0, last_access: 1 };
            (PathBuf::from(format!("/nonexistent/{hash}")), meta, len)
        };
        let mut ix = BlockIndex::rebuild(vec![block(9, 10), block(8, 5)]);
        assert_eq!((ix.disk_bytes, ix.map[&9].on_disk()), (15, true));
        ix.attach_staged_tail(8, 7, 1, vec![5], geom(), range(500));
        ix.touch(8);
        let staged = ix.peel_to_budget(0);
        assert_eq!(staged.len(), 1, "block 8's queued boundary record comes back to be cancelled");
        assert!(ix.map.is_empty());
    }
}
