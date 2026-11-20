//! The paged KV kernels over the block pool's address tables: a token's K/V written into its block, attention over
//! the block tables for decode and prompt, and a sequence's rows gathered for the contiguous SDPA.

use candle_core::{DType, Result, Tensor};

use crate::loader::{launch, out, pipeline_with, set_params, Family, Metal};

const NUM_SIMD_LANES: usize = 32;
const PROMPT_THREADS: usize = 128;
/// Query rows a threadgroup serves per head on a prompt, the simdgroup-matrix kernel's tile; decode serves one.
const PROMPT_TILE: usize = 16;
/// Keys per window of the partitioned kernels.
const PARTITION_SIZE: usize = 512;

/// The kernel instantiations' name for a dtype.
fn metal_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("float"),
        DType::BF16 => Ok("bfloat16_t"),
        DType::F16 => Ok("half"),
        other => candle_core::bail!("paged attention: no kernel for {other:?}"),
    }
}

/// A layer's address table as the kernels bind it: rank-1 I64, one GPU address per block.
fn table<'a>(t: &'a Tensor, what: &str) -> Result<Metal<'a>> {
    refuse_if!(
        t.dtype() != DType::I64 || t.rank() != 1,
        "{what} must be a rank-1 I64 address table, got {:?} {:?}",
        t.dtype(),
        t.shape()
    );
    Metal::of(t, what)
}

/// `key` and `value`, `(tokens, kv_heads, head_size)`, written into the blocks `slot_mapping` (I64, negative a pad)
/// names: K as `(kv_heads, head_size / x, block_size, x)` rows with `x` 16 bytes of elements, V as `(kv_heads,
/// head_size, block_size)`.
pub fn reshape_and_cache(
    key: &Tensor, value: &Tensor, k_addrs: &Tensor, v_addrs: &Tensor, block_size: usize, slot_mapping: &Tensor,
) -> Result<()> {
    let (tokens, heads, head_size) = key.dims3()?;
    refuse_if!(
        value.dims3()? != (tokens, heads, head_size),
        "reshape_and_cache: key {:?} and value {:?} differ",
        key.shape(),
        value.shape()
    );
    refuse_if!(
        slot_mapping.dims1()? != tokens,
        "reshape_and_cache: {:?} slots for {tokens} tokens",
        slot_mapping.shape()
    );
    let (k, v) = (Metal::of(key, "key")?, Metal::of(value, "value")?);
    let (ka, va) = (table(k_addrs, "k_addrs")?, table(v_addrs, "v_addrs")?);
    let slots = Metal::of(slot_mapping, "slot_mapping")?;
    let t = metal_name(key.dtype())?;
    let p = pipeline_with(k.dev, Family::Paged, &format!("reshape_and_cache_kv_{t}_cache_{t}"), &[(50, false)])?;
    let (ks, vs, x) = (key.stride()[0] as i32, value.stride()[0] as i32, (16 / key.dtype().size_in_bytes()) as i32);
    launch(k.dev, &p, "reshape-and-cache", [tokens, 1, 1], [(heads * head_size).min(512), 1, 1], |e| {
        set_params!(
            e,
            (&k, &v, &ka, &va, &slots, (), (), ks, vs, heads as i32, head_size as i32, block_size as i32, x)
        );
        Ok(())
    })
}

/// A one-token step's splits over `max_context_len` keys, with the blocks each covers: MLX's table for this GPU
/// class (sdpa_vector_2pass on a Max: 64 splits to 1,024 keys, doubling to 1,024 beyond 65,536), rounded to whole
/// blocks. One split is the unpartitioned kernel, which takes no scratch.
pub fn decode_splits(max_context_len: usize, block_size: usize) -> (usize, usize) {
    let total_blocks = max_context_len.div_ceil(block_size).max(1);
    let target = match max_context_len {
        n if n <= 1024 => 64,
        n if n <= 8192 => 128,
        n if n <= 32768 => 256,
        n if n <= 65536 => 512,
        _ => 1024,
    };
    let blocks_per_split = total_blocks.div_ceil(target.min(total_blocks));
    (total_blocks.div_ceil(blocks_per_split), blocks_per_split)
}

/// The context up to `admitted` keys whose one-token step splits most, the largest such: the rounding to whole
/// blocks makes the splits fall and rise with the context (208 at 26,624 keys, 256 at 24,576), so the widest
/// scratch a decode step can ask for is not always the deepest's.
pub fn widest_decode_context(admitted: usize, block_size: usize) -> usize {
    (1..=admitted.div_ceil(block_size).max(1))
        .map(|b| (b * block_size).min(admitted))
        .max_by_key(|&c| (decode_splits(c, block_size).0, c))
        .unwrap_or(admitted)
}

/// The group kernel's query heads split over the fewest threadgroups a KV head whose threads fit `limit`: each head's
/// simdgroup computes alone, so a split moves no arithmetic. A prompt tile is returned as it is.
fn split_group(tile: usize, (threads, width, shared): (usize, usize, usize), limit: usize) -> (usize, usize, usize) {
    let heads = threads / NUM_SIMD_LANES;
    match (2..=heads).find(|s| heads % s == 0 && heads / s * NUM_SIMD_LANES <= limit) {
        Some(splits) if tile == 1 && threads > limit => (threads / splits, width * splits, shared / splits),
        _ => (threads, width, shared),
    }
}

/// Attention over the block pool, decode and prompt alike: `q` one row per query token, `(seqs * max_q_len, heads,
/// head_size)` padded per sequence, `context_lens` (U32) each row's visible keys with 0 a pad, `block_tables` (U32)
/// `(seqs, blocks)`; the pool holds the query's dtype. Out `(rows, heads, head_size)`.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention(
    q: &Tensor, k_addrs: &Tensor, v_addrs: &Tensor, num_kv_heads: usize, block_size: usize, block_tables: &Tensor,
    context_lens: &Tensor, max_context_len: usize, softmax_scale: f32, softcapping: f32,
) -> Result<Tensor> {
    let (rows, heads, head_size) = q.dims3()?;
    refuse_if!(![64, 128, 256].contains(&head_size), "paged attention: head size {head_size} is not 64, 128 or 256");
    let (seqs, blocks_per_seq) = block_tables.dims2()?;
    refuse_if!(seqs == 0 || rows % seqs != 0, "paged attention: {rows} query rows over {seqs} block tables");
    refuse_if!(
        context_lens.dims1()? != rows,
        "paged attention: {:?} context lengths for {rows} rows",
        context_lens.shape()
    );
    let max_q_len = rows / seqs;
    let t = metal_name(q.dtype())?;
    let qm = Metal::of(q, "q")?;
    let (ka, va) = (table(k_addrs, "k_addrs")?, table(v_addrs, "v_addrs")?);
    let (bt, cl) = (Metal::of(block_tables, "block_tables")?, Metal::of(context_lens, "context_lens")?);
    let dev = qm.dev;
    // The splits, by shape alone: decode's are `decode_splits`; a prompt tile brings its own parallelism, so splits
    // are added only until ~1024 threadgroups are in flight, which also bounds the scratch.
    let tile = if max_q_len == 1 { 1 } else { PROMPT_TILE };
    let tiles = seqs * max_q_len.div_ceil(tile);
    let windows = max_context_len.div_ceil(PARTITION_SIZE);
    let (splits, per_split) = if tile == 1 {
        decode_splits(max_context_len, block_size)
    } else {
        let splits = if tiles * heads >= 1024 { 1 } else { windows.min(1024usize.div_ceil(tiles * heads)) };
        (splits, windows.div_ceil(splits))
    };
    let partitioned = splits > 1;
    let (windows_per_split, ps) = if partitioned { (per_split, PARTITION_SIZE) } else { (0, 0) };
    // Decode is the group kernel, the GQA group as the threadgroup and the grid over the kv heads; a prompt tile the
    // simdgroup-matrix kernel at PROMPT_THREADS: (name, threads, grid width, dynamic threadgroup bytes).
    let (name, threads, width, shared) = if tile == 1 {
        let group = heads / num_kv_heads;
        let name =
            format!("paged_attention_group_{t}_cache_{t}_hs{head_size}_bs{block_size}_nsl{NUM_SIMD_LANES}_ps{ps}");
        (name, group * NUM_SIMD_LANES, num_kv_heads, group * head_size * q.dtype().size_in_bytes())
    } else {
        let name = format!(
            "paged_attention_mma_{t}_cache_{t}_hs{head_size}_bs{block_size}_nt{PROMPT_THREADS}_nsl{NUM_SIMD_LANES}_ps{ps}_t{tile}"
        );
        (name, PROMPT_THREADS, heads, 0)
    };
    // The function constants: partitioning, then alibi, F8 scales and sinks, which no model here carries.
    let p = pipeline_with(dev, Family::Paged, &name, &[(10, partitioned), (20, false), (30, false), (40, false)])?;
    let (threads, width, shared) = split_group(tile, (threads, width, shared), p.max_total_threads_per_threadgroup());
    let (output, om) = out(dev, (rows, heads, head_size), q.dtype(), "paged-attention-out")?;
    // The partitioned kernel's scratch per row, head and split: the partial output, its exp sum and its max logit.
    let scratch = match partitioned {
        true => Some((
            out(dev, (rows, heads, splits, head_size), q.dtype(), "paged-attention-tmpout")?,
            out(dev, (rows, heads, splits), DType::F32, "paged-attention-expsums")?,
            out(dev, (rows, heads, splits), DType::F32, "paged-attention-maxlogits")?,
        )),
        false => None,
    };
    let (q_stride, kv_head_stride) = (q.stride()[0] as i32, (head_size * block_size) as i32);
    launch(dev, &p, "paged-attention", [width, tiles, splits], [threads, 1, 1], |e| {
        if shared > 0 {
            e.set_threadgroup_memory_length(0, shared);
        }
        let buf = |slot: usize, m: &Metal| e.set_buffer(slot, Some(&m.buf), m.off);
        match &scratch {
            Some((tmp, sums, maxes)) => [(0, &sums.1), (1, &maxes.1), (2, &tmp.1)].iter().for_each(|&(s, m)| buf(s, m)),
            None => buf(2, &om),
        }
        [(3, &qm), (4, &ka), (5, &va), (11, &bt), (12, &cl)].iter().for_each(|&(s, m)| buf(s, m));
        e.set_bytes(8, &(num_kv_heads as i32));
        e.set_bytes(9, &softmax_scale);
        e.set_bytes(10, &softcapping);
        e.set_bytes(13, &(blocks_per_seq as i32));
        e.set_bytes(15, &q_stride);
        e.set_bytes(17, &kv_head_stride);
        e.set_bytes(19, &(max_q_len as i32));
        e.set_bytes(20, &(windows_per_split as i32));
        Ok(())
    })?;
    // The splits folded per row and head, 32 simdgroups striding them.
    if let Some((tmp, sums, maxes)) = &scratch {
        const REDUCE_THREADS: usize = 1024;
        let name = format!(
            "paged_attention_v2_reduce_{t}_hs{head_size}_nt{REDUCE_THREADS}_nsl{NUM_SIMD_LANES}_ps{PARTITION_SIZE}"
        );
        let p = pipeline_with(dev, Family::Paged, &name, &[(40, false)])?;
        launch(dev, &p, "paged-attention", [heads, rows, 1], [REDUCE_THREADS, 1, 1], |e| {
            e.set_threadgroup_memory_length(0, 2 * splits * std::mem::size_of::<f32>());
            set_params!(e, (&om, &sums.1, &maxes.1, &tmp.1, splits as i32));
            Ok(())
        })?;
    }
    Ok(output)
}

/// One sequence's K and V rows out of the block pool, `(kv_heads, blocks * block_size, head_size)` each in the pool's
/// `dtype`, the prompt path's contiguous SDPA input; `block_table` (U32) the sequence's first `blocks` ids.
#[allow(clippy::too_many_arguments)]
pub fn gather_kv_cache(
    k_addrs: &Tensor, v_addrs: &Tensor, block_table: &Tensor, blocks: usize, block_size: usize, num_kv_heads: usize,
    head_size: usize, dtype: DType,
) -> Result<(Tensor, Tensor)> {
    refuse_if!(
        block_table.dtype() != DType::U32 || block_table.dims1()? < blocks,
        "gather_kv_cache: block table {:?} {:?} does not hold {blocks} blocks",
        block_table.dtype(),
        block_table.shape()
    );
    let block_table = block_table.contiguous()?;
    let (ka, va, bt) =
        (table(k_addrs, "k_addrs")?, table(v_addrs, "v_addrs")?, Metal::of(&block_table, "block table")?);
    let tokens = blocks * block_size;
    let (k, km) = out(ka.dev, (num_kv_heads, tokens, head_size), dtype, "gather-kv-k")?;
    let (v, vm) = out(ka.dev, (num_kv_heads, tokens, head_size), dtype, "gather-kv-v")?;
    let p = pipeline_with(ka.dev, Family::Paged, &format!("gather_kv_cache_{}", metal_name(dtype)?), &[])?;
    let x = (16 / dtype.size_in_bytes()) as i32;
    launch(ka.dev, &p, "gather-kv-cache", [tokens, 1, 1], [(num_kv_heads * head_size).min(512), 1, 1], |e| {
        set_params!(
            e,
            (&ka, &va, &km, &vm, &bt, tokens as i32, block_size as i32, num_kv_heads as i32, head_size as i32, x)
        );
        Ok(())
    })?;
    Ok((k, v))
}

#[cfg(test)]
mod tests;
