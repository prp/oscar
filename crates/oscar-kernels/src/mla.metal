// Absorbed MLA: the paged latent rows' writer and gather, and the prefill's online softmax.
#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

typedef bfloat bfloat16_t;

// One scores block's online softmax in place, a simdgroup a row: the visible keys' max and exponentials against the
// row's running (m, l), none on the first block, and `alpha` the rescale of its output. Row r is query position
// r % q_len, seeing keys below kv_offset + position + 1; the block holds keys [b0, b0 + n).
[[kernel]] void mla_online_softmax(
    device float *S [[buffer(0)]],     // [rows, n]: the scores in, their exponentials out
    device float *m [[buffer(1)]],     // [rows]
    device float *l [[buffer(2)]],     // [rows]
    device float *alpha [[buffer(3)]], // [rows]
    const constant int &rows, const constant int &n, const constant int &q_len,
    const constant int &kv_offset, const constant int &b0, const constant int &first,
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const int row = int(tgid * 8 + simd_gid);
  if (row >= rows)
    return;
  const int bound = min(n, kv_offset + row % q_len + 1 - b0);
  device float *s = S + int64_t(row) * n;
  float mx = -INFINITY;
  for (int j = int(lane); j < bound; j += 32)
    mx = max(mx, s[j]);
  mx = simd_max(mx);
  const float m_old = first ? -INFINITY : m[row];
  const float m_new = max(m_old, mx);
  float sum = 0.0f;
  for (int j = int(lane); j < n; j += 32) {
    const float v = (j < bound) ? exp(s[j] - m_new) : 0.0f;
    s[j] = v;
    sum += v;
  }
  sum = simd_sum(sum);
  if (lane == 0) {
    const float a = exp(m_old - m_new);
    m[row] = m_new;
    l[row] = (first ? 0.0f : l[row]) * a + sum;
    alpha[row] = a;
  }
}

// The block's products into the output: o·alpha + pv, pv alone on the first block, over l on the last.
[[kernel]] void mla_accumulate(
    device float *o [[buffer(0)]],           // [rows, dv]
    const device float *pv [[buffer(1)]],    // [rows, dv]
    const device float *alpha [[buffer(2)]], // [rows]
    const device float *l [[buffer(3)]],     // [rows]
    const constant int &rows, const constant int &dv, const constant int &first, const constant int &last,
    uint2 tid [[thread_position_in_grid]]) {
  const int c = int(tid.x);
  const int r = int(tid.y);
  if (c >= dv || r >= rows)
    return;
  const int64_t i = int64_t(r) * dv + c;
  float v = first ? pv[i] : o[i] * alpha[r] + pv[i];
  if (last)
    v /= l[r];
  o[i] = v;
}

// Each token's latent and rope-key rows into its slot's block via the two address tables; a negative slot is padding.
// The pool is bf16 for every model.
[[kernel]] void concat_and_cache_mla(
    const device bfloat16_t *ckv [[buffer(0)]],       // [num_tokens, rank]
    const device bfloat16_t *kpe [[buffer(1)]],       // [num_tokens, rope]
    const device uint64_t *ckv_addrs [[buffer(2)]],   // [blocks]: rows [block_size, rank]
    const device uint64_t *kpe_addrs [[buffer(3)]],   // [blocks]: rows [block_size, rope]
    const device int64_t *slot_mapping [[buffer(4)]], // [num_tokens]
    const constant int &rank, const constant int &rope,
    const constant int &block_size,
    uint2 tid [[thread_position_in_grid]]) {
  const int64_t slot = slot_mapping[tid.y];
  if (slot < 0)
    return;
  const int64_t block = slot / block_size;
  const int64_t row = slot % block_size;
  const int i = int(tid.x);
  if (i < rank) {
    ((device bfloat16_t *)ckv_addrs[block])[row * rank + i] = ckv[int64_t(tid.y) * rank + i];
  } else if (i < rank + rope) {
    const int j = i - rank;
    ((device bfloat16_t *)kpe_addrs[block])[row * rope + j] = kpe[int64_t(tid.y) * rope + j];
  }
}

// The rows at `slots`, in token order, gathered out of the blocks.
[[kernel]] void gather_latent_rows(
    const device uint64_t *addrs [[buffer(0)]],  // [blocks]: rows [block_size, width]
    const device uint32_t *slots [[buffer(1)]],  // [n]
    device bfloat16_t *out [[buffer(2)]],        // [n, width]
    const constant int &width, const constant int &block_size,
    uint2 tid [[thread_position_in_grid]]) {
  const int i = int(tid.x);
  if (i >= width)
    return;
  const int64_t slot = slots[tid.y];
  const device bfloat16_t *row = (const device bfloat16_t *)addrs[slot / block_size] + (slot % block_size) * width;
  out[int64_t(tid.y) * width + i] = row[i];
}
