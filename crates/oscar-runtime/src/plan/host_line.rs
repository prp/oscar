//! What the process holds on the host, summed from the declarations of the structures that allocate it.

use crate::pool::policy::NextUseOracle;
use oscar_offload::machine::hash_map_bytes;
use oscar_offload::segments::PAGE;

use super::{kv_block_object_bytes, paged_block_size, ModelLayoutStats};

/// One booked level for every model: the largest of the models' readings by category, 0.1-0.4 GB.
pub const GPU_DRIVER_HOST_BYTES: u64 = 384 << 20;

/// The host line by term: the run's sum to [`HostLine::bytes`], the load's peak to [`HostLine::load_heap_bytes`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HostLine {
    pub pools: u64,
    pub oracle: u64,
    pub kv_index: u64,
    pub tokenizer: u64,
    pub sequences: u64,
    /// Dropped before the plan is solved, so the load's alone.
    pub record: u64,
    pub lookahead: u64,
    pub ngram_rows: u64,
    pub page_tables: u64,
    pub gpu_driver_host: u64,
    pub loader: u64,
    /// The load's alone, both: the drafter's head requantised where the file ships none, the seal's readers' buffers.
    pub draft_requant: u64,
    pub seal_readers: u64,
}

impl HostLine {
    pub fn bytes(&self) -> u64 {
        self.run_heap_bytes().saturating_add(self.page_tables).saturating_add(self.gpu_driver_host)
    }

    pub fn run_heap_bytes(&self) -> u64 {
        self.pools + self.oracle + self.kv_index + self.tokenizer + self.sequences + self.lookahead + self.ngram_rows
    }

    /// At its peak: the staging beside what is built by then.
    pub fn load_heap_bytes(&self) -> u64 {
        self.pools
            + self.tokenizer
            + self.sequences
            + self.record
            + self.loader
            + self.draft_requant
            + self.seal_readers
    }
}

/// The most blocks `budget` could hold at all, so the line bounds every KV index space without a plan.
fn kv_index_bound(stats: &ModelLayoutStats, budget: u64) -> u64 {
    budget / kv_block_object_bytes(stats).max(1)
}

/// From the header's vocabulary: the byte-pair model held twice (both vocabulary maps, a malloc block per token string,
/// the merges, the 10,000-entry word cache), and the trie at its build's doubling beside the slicer's three copies.
pub fn tokenizer_bytes(tokens: u64, token_bytes: u64, merges: u64) -> u64 {
    use std::mem::size_of;
    let maps = hash_map_bytes::<String, u32>(tokens as usize)
        + hash_map_bytes::<u32, String>(tokens as usize)
        + hash_map_bytes::<(u32, u32), (u32, u32)>(merges as usize);
    let strings = 2 * (token_bytes + tokens * 32);
    let cache = 10_000 * (2 * size_of::<String>() + 16 * 24);
    let model = maps as u64 + strings + cache as u64;
    let nodes = token_bytes + tokens;
    let trie = 5 * nodes * 8 + nodes + tokens * 12;
    2 * model + trie
}

pub fn host_line(stats: &ModelLayoutStats, budget: u64) -> HostLine {
    use std::mem::size_of;
    let (layers, experts) = (stats.block_count, stats.num_experts);
    // The segment size as fixed, else one slot a layer: the most segments, an upper bound.
    let segment_slots = crate::plan::segment_slots().unwrap_or(1);
    let banks: u64 = stats
        .expert_classes()
        .iter()
        .map(|c| crate::pool::Bank::host_bytes(c.layers.len(), experts, segment_slots) as u64)
        .sum();
    // A bounce buffer for each IO worker on every drive, and one more, each held for its thread's life.
    let mirror_readers = crate::config::drives::installed().map_or(0, |m| m.drive.readers);
    let readers = (crate::config::params::io_threads() + mirror_readers) as u64 + 1;
    let states = if stats.sequence_state_bound > 0 {
        crate::gpu::state_pool::SequenceStatePool::host_bytes(crate::plan::STATE_SLOTS_CEILING)
    } else {
        0
    };
    let pools = banks
        + states
        + oscar_offload::fixed::FixedPool::host_bytes() as u64
        + crate::gpu::layout_host_bytes(layers, experts) as u64
        + crate::pool::FetchQueue::host_bytes(layers * experts) as u64
        + readers * oscar_offload::io::bounce_bytes(stats.largest_expert_projection_bytes as usize) as u64;
    let oracle = NextUseOracle::host_bytes(layers, experts, stats.experts_per_token) as u64;
    let blocks = kv_index_bound(stats, budget);
    let context = stats.context_length.max(1);
    // The turns the index space could retain, each a token list and a hash per block: bytes bound retention, no count.
    let turns = blocks * paged_block_size() / context;
    let kv_index = crate::gpu::kv_pool::KvPool::host_bytes(blocks as usize)
        + turns * context * size_of::<u32>() as u64
        + (1 + turns) * context.div_ceil(paged_block_size()) * size_of::<u64>() as u64;
    let tokenizer = tokenizer_bytes(stats.vocab_tokens, stats.vocab_bytes, stats.merges);
    // Each sequence's tokens and prompt text, and the sampler's row of the vocabulary with its ranked pairs.
    let sequences = (1 + turns) * context * 2 * size_of::<u32>() as u64 + stats.vocab_tokens * 12;
    // The model's CPU copies of every gate stand beside the dispatcher's at the hand-over: the first forward holds two.
    let lookahead = if crate::config::params::lookahead_is_host() {
        2 * crate::engine::event_lane::RouterTables::host_bytes(layers, experts, stats.hidden_dim as usize) as u64
    } else {
        0
    };
    HostLine {
        pools,
        oracle,
        kv_index,
        tokenizer,
        sequences,
        record: crate::load::census::census().map_or(0, |c| c.record_bytes),
        lookahead,
        ngram_rows: stats.ple_row_cache_bytes,
        page_tables: budget.div_ceil(PAGE as u64) * size_of::<u64>() as u64,
        gpu_driver_host: GPU_DRIVER_HOST_BYTES,
        loader: stats.largest_dense_tensor_bytes.next_multiple_of(PAGE as u64) + PAGE as u64,
        draft_requant: if crate::config::small_pool() { 0 } else { stats.draft_requant_bytes },
        seal_readers: (crate::pool::SEAL_READERS
            * oscar_offload::io::bounce_bytes(stats.largest_expert_projection_bytes as usize))
            as u64,
    }
}

/// Metal, in whole pages at the budget's index spaces: the KV address tables, each class's address table and epochs,
/// the sealed and draft rows and epochs, the wave ring, the miss counter and the draft watermark.
pub fn pool_tables_bytes(stats: &ModelLayoutStats, budget: u64) -> u64 {
    use std::mem::size_of;
    let page = PAGE as u64;
    let pages = |bytes: u64| bytes.next_multiple_of(page).max(page);
    let (layers, experts) = (stats.block_count as u64, stats.num_experts as u64);
    let kv = crate::gpu::kv_pool::KvPool::table_bytes(kv_index_bound(stats, budget), stats.kv_bearing_layers);
    let segment_slots = crate::plan::segment_slots().unwrap_or(1);
    let banks: u64 = stats
        .expert_classes()
        .iter()
        .map(|c| {
            let (segments, per_segment) =
                crate::pool::Bank::index_space(c.layers.len(), experts as usize, segment_slots);
            let slots = (segments * per_segment) as u64;
            pages(slots * size_of::<u64>() as u64) + pages(slots * size_of::<u32>() as u64) + page
        })
        .sum();
    let rows =
        3 * pages(layers * experts * size_of::<u64>() as u64) + 6 * pages(layers * experts * size_of::<u32>() as u64);
    let wave = 64 * pages(experts * (3 * size_of::<u64>() + size_of::<u32>()) as u64);
    kv + banks + rows + wave + 2 * page
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_host_line_is_the_declarations_summed() {
        let mut stats = crate::plan::tests::routed(60, 128, 1 << 20);
        (stats.vocab_tokens, stats.vocab_bytes, stats.merges) = (100_000, 700_000, 90_000);
        stats.largest_dense_tensor_bytes = 300 << 20;
        stats.largest_expert_projection_bytes = 4_243_456;
        stats.context_length = 32_768;
        let line = host_line(&stats, 54 << 30);
        // Each reading thread's bounce buffer: a 259-page extent read through 260 pages.
        assert_eq!(oscar_offload::io::bounce_bytes(4_243_456), 260 * 16384);
        assert!(line.pools >= 2 * 260 * 16384);
        assert_eq!(line.oracle, NextUseOracle::host_bytes(60, 128, 8) as u64);
        assert!(line.oracle > 60 * 128 * 128 * 4);
        assert_eq!(line.page_tables, (54u64 << 30) / 16384 * 8);
        assert_eq!(line.tokenizer, tokenizer_bytes(100_000, 700_000, 90_000));
        assert_eq!(line.loader, (300 << 20) + 16384);
        assert_eq!(line.bytes(), line.run_heap_bytes() + line.page_tables + line.gpu_driver_host);
        assert_eq!(line.seal_readers, 8 * 260 * 16384, "a seal's eight readers' bounce buffers");
        let load = line.pools + line.tokenizer + line.sequences + line.record + line.loader + line.seal_readers;
        assert_eq!(line.load_heap_bytes(), load);
        stats.draft_requant_bytes = 300 << 20;
        let requant = host_line(&stats, 54 << 30).load_heap_bytes() - line.load_heap_bytes();
        assert_eq!(requant, 300 << 20, "the load's alone");
        stats.draft_requant_bytes = 0;
        assert!(line.bytes() < 1 << 30, "a 60-layer, 128-expert model's line is well under a GB: {}", line.bytes());
        // The n-gram rows are the run's, not the load's.
        stats.ple_row_cache_bytes = 70 << 20;
        let ngram = host_line(&stats, 54 << 30);
        assert_eq!(ngram.run_heap_bytes() - line.run_heap_bytes(), 70 << 20);
        assert_eq!(ngram.load_heap_bytes(), line.load_heap_bytes());
        let wider = host_line(&stats, 108 << 30);
        assert!(wider.kv_index > line.kv_index);
        assert_eq!((wider.oracle, wider.tokenizer, wider.pools), (line.oracle, line.tokenizer, line.pools));
        assert!(pool_tables_bytes(&stats, 108 << 30) > pool_tables_bytes(&stats, 54 << 30));
    }
}
