// Oscar's paged KV kernels over the block pool's address tables, after mistral.rs's paged attention,
// itself after vLLM's and MLX's (Apache 2.0): attention for decode and prompt, a token's K/V written
// into its block, a sequence's rows gathered for the contiguous SDPA. The helpers are the fork's.

#include "utils.metal"
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

// ========================================== Generic vector types

// A vector type to store Q, K, V elements.
template <typename T, int VEC_SIZE> struct Vec {};

// A vector type to store FP32 accumulators.
template <typename T> struct FloatVec {};

// Template vector operations.
template <typename Acc, typename A, typename B> inline Acc mul(A a, B b);

template <typename T> inline float sum(T v);

template <typename T> inline float dot(T a, T b) {
  return sum(mul<T, T, T>(a, b));
}

template <typename A, typename T> inline float dot(T a, T b) {
  return sum(mul<A, T, T>(a, b));
}

// FP32 vector data types.
struct Float8_ {
  float4 x;
  float4 y;
};

template <> struct Vec<float, 1> {
  using Type = float;
};
template <> struct Vec<float, 2> {
  using Type = float2;
};
template <> struct Vec<float, 4> {
  using Type = float4;
};
template <> struct Vec<float, 8> {
  using Type = Float8_;
};

template <> struct FloatVec<float> {
  using Type = float;
};
template <> struct FloatVec<float2> {
  using Type = float2;
};
template <> struct FloatVec<float4> {
  using Type = float4;
};
template <> struct FloatVec<Float8_> {
  using Type = Float8_;
};

template <> inline float mul(float a, float b) { return a * b; }

template <> inline float2 mul(float2 a, float2 b) { return a * b; }

template <> inline float4 mul(float4 a, float4 b) { return a * b; }

template <> inline Float8_ mul(Float8_ a, Float8_ b) {
  Float8_ c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

template <> inline float sum(float a) { return a; }

template <> inline float sum(float2 a) { return a.x + a.y; }

template <> inline float sum(float4 a) { return a.x + a.y + a.z + a.w; }

template <> inline float sum(Float8_ a) { return sum(a.x) + sum(a.y); }

inline Float8_ fma(Float8_ a, Float8_ b, Float8_ c) {
  Float8_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline void from_float(thread float &dst, float src) { dst = src; }
inline void from_float(thread float2 &dst, float2 src) { dst = src; }
inline void from_float(thread float4 &dst, float4 src) { dst = src; }
inline void from_float(thread Float8_ &dst, Float8_ src) { dst = src; }

// BF16 vector data types.
struct Bfloat2_ {
  bfloat16_t x;
  bfloat16_t y;
};

struct Bfloat4_ {
  Bfloat2_ x;
  Bfloat2_ y;
};

struct Bfloat8_ {
  Bfloat4_ x;
  Bfloat4_ y;
};

template <> struct Vec<bfloat16_t, 1> {
  using Type = bfloat16_t;
};
template <> struct Vec<bfloat16_t, 2> {
  using Type = Bfloat2_;
};
template <> struct Vec<bfloat16_t, 4> {
  using Type = Bfloat4_;
};
template <> struct Vec<bfloat16_t, 8> {
  using Type = Bfloat8_;
};

template <> struct FloatVec<bfloat16_t> {
  using Type = float;
};
template <> struct FloatVec<Bfloat2_> {
  using Type = float2;
};
template <> struct FloatVec<Bfloat4_> {
  using Type = float4;
};
template <> struct FloatVec<Bfloat8_> {
  using Type = Float8_;
};

template <> inline float mul(bfloat16_t a, bfloat16_t b) {
  return (float)a * (float)b;
}
template <> inline bfloat16_t mul(bfloat16_t a, bfloat16_t b) { return a * b; }

template <> inline float2 mul(Bfloat2_ a, Bfloat2_ b) {
  float2 a_f((float)a.x, (float)a.y);
  float2 b_f((float)b.x, (float)b.y);
  return a_f * b_f;
}
template <> inline Bfloat2_ mul(Bfloat2_ a, Bfloat2_ b) {
  Bfloat2_ c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

template <> inline float4 mul(Bfloat4_ a, Bfloat4_ b) {
  float2 x = mul<float2, Bfloat2_, Bfloat2_>(a.x, b.x);
  float2 y = mul<float2, Bfloat2_, Bfloat2_>(a.y, b.y);
  float4 c;
  c.x = x.x;
  c.y = x.y;
  c.z = y.x;
  c.w = y.y;
  return c;
}
template <> inline Bfloat4_ mul(Bfloat4_ a, Bfloat4_ b) {
  Bfloat4_ c;
  c.x = mul<Bfloat2_, Bfloat2_, Bfloat2_>(a.x, b.x);
  c.y = mul<Bfloat2_, Bfloat2_, Bfloat2_>(a.y, b.y);
  return c;
}

template <> inline Float8_ mul(Bfloat8_ a, Bfloat8_ b) {
  Float8_ c;
  c.x = mul<float4, Bfloat4_, Bfloat4_>(a.x, b.x);
  c.y = mul<float4, Bfloat4_, Bfloat4_>(a.y, b.y);
  return c;
}
template <> inline Bfloat8_ mul(Bfloat8_ a, Bfloat8_ b) {
  Bfloat8_ c;
  c.x = mul<Bfloat4_, Bfloat4_, Bfloat4_>(a.x, b.x);
  c.y = mul<Bfloat4_, Bfloat4_, Bfloat4_>(a.y, b.y);
  return c;
}

template <> inline float sum(bfloat16_t a) { return (float)a; }

template <> inline float sum(Bfloat2_ a) { return (float)a.x + (float)a.y; }

template <> inline float sum(Bfloat4_ a) { return sum(a.x) + sum(a.y); }

template <> inline float sum(Bfloat8_ a) { return sum(a.x) + sum(a.y); }

inline float fma(bfloat16_t a, bfloat16_t b, float c) {
  return (float)a * (float)b + c;
}
inline bfloat16_t fma(bfloat16_t a, bfloat16_t b, bfloat16_t c) {
  return a * b + c;
}

inline float2 fma(Bfloat2_ a, Bfloat2_ b, float2 c) {
  float2 a_f((float)a.x, (float)a.y);
  float2 b_f((float)b.x, (float)b.y);
  return a_f * b_f + c;
}
inline Bfloat2_ fma(Bfloat2_ a, Bfloat2_ b, Bfloat2_ c) {
  Bfloat2_ res;
  res.x = a.x * b.x + c.x;
  res.y = a.y * b.y + c.y;
  return res;
}

inline float4 fma(Bfloat4_ a, Bfloat4_ b, float4 c) {
  float4 res;
  res.x = fma(a.x.x, b.x.x, c.x);
  res.y = fma(a.x.y, b.x.y, c.y);
  res.z = fma(a.y.x, b.y.x, c.z);
  res.w = fma(a.y.y, b.y.y, c.w);
  return res;
}
inline Bfloat4_ fma(Bfloat4_ a, Bfloat4_ b, Bfloat4_ c) {
  Bfloat4_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline Float8_ fma(Bfloat8_ a, Bfloat8_ b, Float8_ c) {
  float4 x = fma(a.x, b.x, c.x);
  float4 y = fma(a.y, b.y, c.y);
  Float8_ res;
  res.x = x;
  res.y = y;
  return res;
}
inline Bfloat8_ fma(Bfloat8_ a, Bfloat8_ b, Bfloat8_ c) {
  Bfloat8_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline void from_float(thread bfloat16_t &dst, float src) {
  dst = static_cast<bfloat16_t>(src);
}
inline void from_float(thread Bfloat2_ &dst, float2 src) {
  dst.x = static_cast<bfloat16_t>(src.x);
  dst.y = static_cast<bfloat16_t>(src.y);
}
inline void from_float(thread Bfloat4_ &dst, float4 src) {
  dst.x.x = static_cast<bfloat16_t>(src.x);
  dst.x.y = static_cast<bfloat16_t>(src.y);
  dst.y.x = static_cast<bfloat16_t>(src.z);
  dst.y.y = static_cast<bfloat16_t>(src.w);
}
inline void from_float(thread Bfloat8_ &dst, Float8_ src) {
  Bfloat4_ x;
  Bfloat4_ y;
  from_float(x, src.x);
  from_float(y, src.y);
  dst.x = x;
  dst.y = y;
}

// ========================================== FP8 (uchar) vector data types.

// 8‑lane uchar vector – Metal only provides up to uchar4, so build our own.
struct Uchar8_ {
  uchar4 x;
  uchar4 y;
};

// Vec specialisations so Vec<uchar, N>::Type resolves correctly.
template <> struct Vec<uchar, 1> {
  using Type = uchar;
};
template <> struct Vec<uchar, 2> {
  using Type = uchar2;
};
template <> struct Vec<uchar, 4> {
  using Type = uchar4;
};
template <> struct Vec<uchar, 8> {
  using Type = Uchar8_;
};

// FP16 vector data types.
struct Half8_ {
  half4 x;
  half4 y;
};

template <> struct Vec<half, 1> {
  using Type = half;
};
template <> struct Vec<half, 2> {
  using Type = half2;
};
template <> struct Vec<half, 4> {
  using Type = half4;
};
template <> struct Vec<half, 8> {
  using Type = Half8_;
};

template <> struct FloatVec<half> {
  using Type = float;
};
template <> struct FloatVec<half2> {
  using Type = float2;
};
template <> struct FloatVec<half4> {
  using Type = float4;
};
template <> struct FloatVec<Half8_> {
  using Type = Float8_;
};

template <> inline float mul(half a, half b) { return (float)a * (float)b; }
template <> inline half mul(half a, half b) { return a * b; }

template <> inline float2 mul(half2 a, half2 b) {
  return (float2)a * (float2)b;
}
template <> inline half2 mul(half2 a, half2 b) { return a * b; }

template <> inline float4 mul(half4 a, half4 b) {
  return (float4)a * (float4)b;
}
template <> inline half4 mul(half4 a, half4 b) { return a * b; }

template <> inline Float8_ mul(Half8_ a, Half8_ b) {
  float4 x = mul<float4, half4, half4>(a.x, b.x);
  float4 y = mul<float4, half4, half4>(a.y, b.y);
  Float8_ c;
  c.x = x;
  c.y = y;
  return c;
}
template <> inline Half8_ mul(Half8_ a, Half8_ b) {
  Half8_ c;
  c.x = mul<half4, half4, half4>(a.x, b.x);
  c.y = mul<half4, half4, half4>(a.y, b.y);
  return c;
}

template <> inline float sum(half a) { return (float)a; }

template <> inline float sum(half2 a) { return (float)a.x + (float)a.y; }

template <> inline float sum(half4 a) { return a.x + a.y + a.z + a.w; }

template <> inline float sum(Half8_ a) { return sum(a.x) + sum(a.y); }

inline float fma(half a, half b, float c) { return (float)a * (float)b + c; }

inline float2 fma(half2 a, half2 b, float2 c) {
  return (float2)a * (float2)b + c;
}

inline float4 fma(half4 a, half4 b, float4 c) {
  return (float4)a * (float4)b + c;
}

inline Float8_ fma(Half8_ a, Half8_ b, Float8_ c) {
  float4 x = fma(a.x, b.x, c.x);
  float4 y = fma(a.y, b.y, c.y);
  Float8_ res;
  res.x = x;
  res.y = y;
  return res;
}
inline Half8_ fma(Half8_ a, Half8_ b, Half8_ c) {
  Half8_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline void from_float(thread half &dst, float src) {
  dst = static_cast<half>(src);
}
inline void from_float(thread half2 &dst, float2 src) {
  dst.x = static_cast<half>(src.x);
  dst.y = static_cast<half>(src.y);
}
inline void from_float(thread half4 &dst, float4 src) {
  dst.x = static_cast<half>(src.x);
  dst.y = static_cast<half>(src.y);
  dst.z = static_cast<half>(src.z);
  dst.w = static_cast<half>(src.w);
}
inline void from_float(thread Half8_ &dst, Float8_ src) {
  half4 x;
  half4 y;
  from_float(x, src.x);
  from_float(y, src.y);
  dst.x = x;
  dst.y = y;
}

// General case: not uchar
template <typename T> inline constexpr bool is_uchar() { return false; }

// Specialization: T is uchar
template <> inline constexpr bool is_uchar<uchar>() { return true; }

// Generic fallback – will fail to compile if a required specialisation is missing.
template <typename Vec, typename Quant_vec>
inline Vec fp8_convert(const thread Quant_vec &, float scale) {
  static_assert(sizeof(Vec) == 0, "Missing fp8_convert specialisation");
}

// ========================================== Block sum utility

// Utility function for attention softmax.
template <int NUM_WARPS, int NUM_SIMD_LANES>
inline float block_sum(threadgroup float *red_smem, float sum, uint simd_tid,
                       uint simd_lid) {
  // Compute the sum per simdgroup.
#pragma unroll
  for (int mask = NUM_SIMD_LANES / 2; mask >= 1; mask /= 2) {
    sum += simd_shuffle_xor(sum, mask);
  }

  // Simd leaders store the data to shared memory.
  if (simd_lid == 0) {
    red_smem[simd_tid] = sum;
  }

  // Make sure the data is in shared memory.
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // The warps compute the final sums.
  if (simd_lid < NUM_WARPS) {
    sum = red_smem[simd_lid];
  }

  // Parallel reduction inside the simd group.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += simd_shuffle_xor(sum, mask);
  }

  // Broadcast to other threads.
  return simd_shuffle(sum, 0);
}

// ========================================== Paged Attention kernel

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

constant bool use_partitioning [[function_constant(10)]];
constant bool use_alibi [[function_constant(20)]];
constant bool use_fp8_scales [[function_constant(30)]];
constant bool use_sinks [[function_constant(40)]];

// Group decode kernel: one query row per sequence, the GQA group as the threadgroup, simdgroup g
// query head kv_head * group + g, all walking the same blocks (MLX's sdpa_vector_2pass_1 shape on
// the block table). The grid is (kv_heads, seqs, splits); a split owns `windows_per_split` blocks.
template <typename T, typename CACHE_T, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_SIMD_LANES, int PARTITION_SIZE = 0>
[[kernel]] void paged_attention_group(
    device float *exp_sums
    [[buffer(0), function_constant(use_partitioning)]], // [num_seqs, num_heads,
                                                        // max_num_partitions]
    device float *max_logits
    [[buffer(1), function_constant(use_partitioning)]], // [num_seqs, num_heads,
                                                        // max_num_partitions]
    device T *out
    [[buffer(2)]], // [num_seqs, num_heads, max_num_partitions, head_size]
    device const T *q [[buffer(3)]], // [num_seqs, num_heads, head_size]
    device const uint64_t *k_addrs
    [[buffer(4)]], // [num_blocks]: each block's K row
                   // [num_kv_heads, head_size/x, block_size, x]
    device const uint64_t *v_addrs
    [[buffer(5)]], // [num_blocks]: each block's V row
                   // [num_kv_heads, head_size, block_size]
    const device float *__restrict__ k_scale
    [[buffer(6), function_constant(use_fp8_scales)]], // [1]
    const device float *__restrict__ v_scale
    [[buffer(7), function_constant(use_fp8_scales)]], // [1]
    const constant int &num_kv_heads [[buffer(8)]],
    const constant float &scale [[buffer(9)]],
    const constant float &softcapping [[buffer(10)]],
    device const uint32_t *block_tables
    [[buffer(11)]], // [num_seqs, max_num_blocks_per_seq]
    device const uint32_t *context_lens [[buffer(12)]], // [num_seqs]
    const constant int &max_num_blocks_per_seq [[buffer(13)]],
    device const float *alibi_slopes
    [[buffer(14), function_constant(use_alibi)]], // [num_heads]
    const constant int &q_stride [[buffer(15)]],
    const constant int &kv_head_stride [[buffer(17)]],
    const device float *sinks
    [[buffer(18), function_constant(use_sinks)]], // [num_heads]
    const constant int &windows_per_split [[buffer(20)]],
    threadgroup char *shared_mem [[threadgroup(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  static_assert(BLOCK_SIZE <= NUM_SIMD_LANES,
                "a simdgroup scores a whole block per iteration");
  // Score geometry: THREAD_GROUP_SIZE lanes share one token, each holding
  // NUM_VECS_PER_THREAD 16-byte chunks of its key.
  constexpr int THREAD_GROUP_SIZE = NUM_SIMD_LANES / BLOCK_SIZE;
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(T)), 1);
  using K_vec = typename Vec<T, VEC_SIZE>::Type;
  using Q_vec = typename Vec<T, VEC_SIZE>::Type;
  using Quant_vec = typename Vec<CACHE_T, VEC_SIZE>::Type;
  constexpr int NUM_VECS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE / VEC_SIZE;
  constexpr int x = 16 / sizeof(CACHE_T);
  // Value geometry: a lane owns V_VEC_SIZE consecutive tokens of one
  // head-dim row per iteration, NUM_ROWS_PER_THREAD rows over the block.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(T), BLOCK_SIZE);
  using V_vec = typename Vec<T, V_VEC_SIZE>::Type;
  using V_quant_vec = typename Vec<CACHE_T, V_VEC_SIZE>::Type;
  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = NUM_SIMD_LANES / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD =
      DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  // A threadgroup holds one KV head's query heads, or a run of them where the group is split over threadgroups.
  const int seq_idx = threadgroup_position_in_grid.y;
  const int split_idx = threadgroup_position_in_grid.z;
  const int max_num_partitions = threadgroups_per_grid.z;
  const int heads_per_tg = threads_per_threadgroup.x / NUM_SIMD_LANES;
  const int num_heads = threadgroups_per_grid.x * heads_per_tg;
  const int head_idx = threadgroup_position_in_grid.x * heads_per_tg + simd_tid;
  const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
  const int lane = simd_lid;

  const int ctx_len = int(context_lens[seq_idx]);
  const int seq_blocks = DIVIDE_ROUND_UP(ctx_len, BLOCK_SIZE);
  // [b0, b1) is the block range this threadgroup reduces over.
  const int b0 = USE_PARTITIONING ? split_idx * windows_per_split : 0;
  const int b1 = USE_PARTITIONING ? MIN(b0 + windows_per_split, seq_blocks)
                                  : seq_blocks;

  const int64_t head_row = int64_t(seq_idx) * num_heads + head_idx;
  const int64_t part_idx = head_row * max_num_partitions + split_idx;
  device T *out_ptr = USE_PARTITIONING ? out + part_idx * HEAD_SIZE
                                       : out + head_row * HEAD_SIZE;

  // A split past this sequence's context (a shorter sequence of a ragged batch) or a padding row still publishes: the
  // reduce reads every (row, head, split) entry, and padding rows must not leave garbage behind.
  if (b0 >= b1) {
    if (USE_PARTITIONING) {
      if (lane == 0) {
        exp_sums[part_idx] = 0.f;
        max_logits[part_idx] = -FLT_MAX;
      }
    } else {
      for (int i = lane; i < HEAD_SIZE; i += NUM_SIMD_LANES) {
        out_ptr[i] = T(0);
      }
    }
    return;
  }

  // Stage this simdgroup's query row in the per-lane interleave the dot
  // product reads: lane offset o holds chunks o, o + THREAD_GROUP_SIZE, ...
  const int thread_group_offset = lane % THREAD_GROUP_SIZE;
  const int token_in_block = lane / THREAD_GROUP_SIZE;
  threadgroup Q_vec *q_slot =
      reinterpret_cast<threadgroup Q_vec *>(shared_mem) +
      simd_tid * THREAD_GROUP_SIZE * NUM_VECS_PER_THREAD;
  {
    const device T *q_ptr =
        q + int64_t(seq_idx) * q_stride + int64_t(head_idx) * HEAD_SIZE;
    for (int v = lane; v < THREAD_GROUP_SIZE * NUM_VECS_PER_THREAD;
         v += NUM_SIMD_LANES) {
      q_slot[(v % THREAD_GROUP_SIZE) * NUM_VECS_PER_THREAD +
             v / THREAD_GROUP_SIZE] =
          *reinterpret_cast<const device Q_vec *>(q_ptr + v * VEC_SIZE);
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const threadgroup Q_vec(&q_vecs)[NUM_VECS_PER_THREAD] =
      *reinterpret_cast<const threadgroup Q_vec(*)[NUM_VECS_PER_THREAD]>(
          q_slot + thread_group_offset * NUM_VECS_PER_THREAD);

  const float alibi_slope = !use_alibi ? 0.f : alibi_slopes[head_idx];
  const device uint32_t *block_table =
      block_tables + int64_t(seq_idx) * max_num_blocks_per_seq;
  const int last_ctx_block = seq_blocks - 1;
  // This lane's value slice: row v_row0 (+ i * NUM_ROWS_PER_ITER), tokens v_tok0 .. v_tok0 + V_VEC_SIZE of the block.
  const int v_tok0 = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
  const int v_row0 = lane / NUM_V_VECS_PER_ROW;

  float m_run = -FLT_MAX;
  float l_run = 0.f;
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  for (int block_idx = b0; block_idx < b1; block_idx++) {
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);
    const int token_idx = block_idx * BLOCK_SIZE + token_in_block;

    // Score phase: this lane's token against this simdgroup's head, the key streamed chunk by chunk into the dot — no
    // key array lives in registers, which is what lets enough simdgroups share a core to hide the loads.
    const device CACHE_T *k_ptr =
        (const device CACHE_T *)k_addrs[physical_block_number] +
        kv_head_idx * kv_head_stride + token_in_block * x;
    using A_vec = typename FloatVec<K_vec>::Type;
    A_vec qk_vec = {};
#pragma unroll
    for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
      const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
      const device CACHE_T *chunk = k_ptr +
                                    ((vec_idx * VEC_SIZE) / x) * BLOCK_SIZE * x +
                                    (vec_idx * VEC_SIZE) % x;
      K_vec k_vec;
      if constexpr (is_uchar<CACHE_T>()) {
        Quant_vec k_vec_quant = *reinterpret_cast<const device Quant_vec *>(chunk);
        k_vec = fp8_convert<K_vec, Quant_vec>(k_vec_quant, *k_scale);
      } else {
        k_vec = *reinterpret_cast<const device K_vec *>(chunk);
      }
      qk_vec = fma(q_vecs[j], k_vec, qk_vec);
    }
    float qk = sum(qk_vec);
#pragma unroll
    for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
      qk += simd_shuffle_xor(qk, mask);
    }
    qk *= scale;
    if (softcapping != 1.0) {
      qk = tanh(qk / softcapping) * softcapping;
    }
    qk += (alibi_slope != 0) ? alibi_slope * (token_idx - ctx_len + 1) : 0;
    const bool masked = token_idx >= ctx_len;

    // The block's online-softmax step, in registers: the block max, the
    // rescale of the running state, this block's probabilities.
    const float block_max = simd_max(masked ? -FLT_MAX : qk);
    const float m_new = max(m_run, block_max);
    const float f = m_run == -FLT_MAX ? 0.f : exp(m_run - m_new);
    const float p = masked ? 0.f : exp(qk - m_new);
    // Every lane of a token's group holds its probability; count it once.
    l_run = l_run * f + simd_sum(thread_group_offset == 0 ? p : 0.f);
    m_run = m_new;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      accs[i] *= f;
    }
    // The probabilities of this lane's value tokens, from the lanes that
    // scored them (every lane of a token's group holds its score).
    float p_vec[V_VEC_SIZE];
#pragma unroll
    for (int j = 0; j < V_VEC_SIZE; j++) {
      p_vec[j] = simd_shuffle(p, ushort((v_tok0 + j) * THREAD_GROUP_SIZE));
    }

    // Value phase. Slots past the context in the last block may hold anything (NaN included) against a zero
    // probability, so they are dropped explicitly.
    const device CACHE_T *v_ptr =
        (const device CACHE_T *)v_addrs[physical_block_number] +
        kv_head_idx * kv_head_stride + v_tok0;
    const int visible = ctx_len - (block_idx * BLOCK_SIZE + v_tok0);
    const bool clip = block_idx == last_ctx_block && visible < V_VEC_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = v_row0 + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        V_vec v_vec;
        if constexpr (is_uchar<CACHE_T>()) {
          V_quant_vec v_quant_vec = *reinterpret_cast<const device V_quant_vec *>(
              v_ptr + row_idx * BLOCK_SIZE);
          v_vec = fp8_convert<V_vec, V_quant_vec>(v_quant_vec, *v_scale);
        } else {
          v_vec = *reinterpret_cast<const device V_vec *>(v_ptr +
                                                          row_idx * BLOCK_SIZE);
        }
        const thread T *v_elems = reinterpret_cast<const thread T *>(&v_vec);
        float acc = accs[i];
#pragma unroll
        for (int j = 0; j < V_VEC_SIZE; j++) {
          const float vj = (clip && j >= visible) ? 0.f : float(v_elems[j]);
          acc += p_vec[j] * vj;
        }
        accs[i] = acc;
      }
    }
  }

  // The NUM_V_VECS_PER_ROW lanes of a row hold disjoint token slices.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += simd_shuffle_xor(acc, mask);
    }
    accs[i] = acc;
  }

  // Write the row: normalized by the split's own exp-sum in both modes, the per-split contract the reduce pass and the
  // prompt kernel share. An unpartitioned row folds the sink in here; a partitioned one leaves it to the reduce.
  float m = m_run;
  float l = l_run;
  float out_scale = 1.f;
  if (!USE_PARTITIONING && use_sinks) {
    const float s = sinks[head_idx];
    const float m2 = max(m, s);
    out_scale = exp(m - m2);
    l = l * out_scale + exp(s - m2);
    m = m2;
  }
  const float inv = l > 0.f ? 1.f / (l + 1e-6f) : 0.f;
  if (USE_PARTITIONING && lane == 0) {
    exp_sums[part_idx] = l;
    max_logits[part_idx] = m;
  }
  if (lane % NUM_V_VECS_PER_ROW == 0) {
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = v_row0 + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        out_ptr[row_idx] = T(accs[i] * out_scale * inv);
      }
    }
  }
}

// Simdgroup-matrix prompt kernel: a threadgroup owns TILE query rows of one sequence and head
// and reduces over the context online in WINDOW-token spans (fp32 throughout, splits merged by
// paged_attention_v2_reduce). Causality is `context_lens`, one visible-key count per query row.
template <typename T, typename CACHE_T, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, int NUM_SIMD_LANES, int PARTITION_SIZE = 0,
          int TILE = 16>
[[kernel]] void paged_attention_mma(
    device float *exp_sums
    [[buffer(0), function_constant(use_partitioning)]],
    device float *max_logits
    [[buffer(1), function_constant(use_partitioning)]],
    device T *out [[buffer(2)]],
    device const T *q [[buffer(3)]],
    device const uint64_t *k_addrs [[buffer(4)]],
    device const uint64_t *v_addrs [[buffer(5)]],
    const device float *__restrict__ k_scale
    [[buffer(6), function_constant(use_fp8_scales)]],
    const device float *__restrict__ v_scale
    [[buffer(7), function_constant(use_fp8_scales)]],
    const constant int &num_kv_heads [[buffer(8)]],
    const constant float &scale [[buffer(9)]],
    const constant float &softcapping [[buffer(10)]],
    device const uint32_t *block_tables [[buffer(11)]],
    device const uint32_t *context_lens [[buffer(12)]],
    const constant int &max_num_blocks_per_seq [[buffer(13)]],
    device const float *alibi_slopes
    [[buffer(14), function_constant(use_alibi)]],
    const constant int &q_stride [[buffer(15)]],
    const constant int &kv_head_stride [[buffer(17)]],
    const device float *sinks
    [[buffer(18), function_constant(use_sinks)]],
    const constant int &max_q_len [[buffer(19)]],
    const constant int &windows_per_split [[buffer(20)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  constexpr int WINDOW = USE_PARTITIONING ? PARTITION_SIZE : 512;
  constexpr int NUM_WARPS = NUM_THREADS / NUM_SIMD_LANES;
  constexpr int STRIPS = TILE / 8;
  // K-tile depth: sized so the fp32 K staging tile stays at 8 KB.
  constexpr int BK = HEAD_SIZE <= 128 ? 16 : 8;
  constexpr int KHALVES = NUM_WARPS / STRIPS;
  constexpr int KSTEPS = HEAD_SIZE / 8 / KHALVES;
  constexpr int O_TILES_PER_WARP = HEAD_SIZE / 8 / (NUM_WARPS / STRIPS);
  static_assert(HEAD_SIZE % (8 * KHALVES) == 0, "head dim must tile");
  static_assert(TILE % 8 == 0 && BK % 8 == 0, "MMA tiles are 8x8");

  const int head_idx = threadgroup_position_in_grid.x;
  const int num_heads = threadgroups_per_grid.x;
  const int tile_idx = threadgroup_position_in_grid.y;
  const int split_idx = threadgroup_position_in_grid.z;
  const int max_num_partitions = threadgroups_per_grid.z;
  const int thread_idx = int(thread_position_in_threadgroup.x);
  const int warp_idx = int(simd_tid);
  const int lane = int(simd_lid);

  const int tiles_per_seq = DIVIDE_ROUND_UP(max_q_len, TILE);
  const int seq_idx = tile_idx / tiles_per_seq;
  const int tile_local0 = (tile_idx % tiles_per_seq) * TILE;

  // K as [d][t] for the QKt loads, then V as [t][d], then the final O rows;
  // also the Q strip staging lane while queries park to registers.
  threadgroup float kv_smem[BK * HEAD_SIZE];
  // S partials per k-half; sp_smem[0] holds P after the mask pass.
  threadgroup float sp_smem[KHALVES][TILE * BK];
  threadgroup float m_smem[TILE], l_smem[TILE], f_smem[TILE];
  threadgroup int ctx_smem[TILE];
  // Per warp, not per strip: the two warps sharing a strip would otherwise race on identical bytes.
  threadgroup float diag_smem[NUM_WARPS][64];

  int tile_max_ctx = 0;
  if (thread_idx < TILE) {
    const int q_local = tile_local0 + thread_idx;
    ctx_smem[thread_idx] =
        q_local < max_q_len
            ? int(context_lens[seq_idx * max_q_len + q_local])
            : 0;
    m_smem[thread_idx] = -FLT_MAX;
    l_smem[thread_idx] = 0.f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
  for (int r = 0; r < TILE; r++) {
    tile_max_ctx = max(tile_max_ctx, ctx_smem[r]);
  }

  const int total_windows = DIVIDE_ROUND_UP(tile_max_ctx, WINDOW);
  const int w0 = USE_PARTITIONING ? split_idx * windows_per_split : 0;
  const int w1 = USE_PARTITIONING
                     ? MIN(w0 + windows_per_split, total_windows)
                     : total_windows;

  // A split past the tile's context (or an all-padding tile) still publishes (an empty partial, or a zero row), so the
  // reduce and the consumer never read an unwritten slot.
  if (w0 >= w1) {
    if (USE_PARTITIONING) {
      if (thread_idx == 0) {
        for (int r = 0; r < TILE; r++) {
          const int q_local = tile_local0 + r;
          if (q_local >= max_q_len) {
            continue;
          }
          const int64_t q_row = int64_t(seq_idx) * max_q_len + q_local;
          const int64_t idx = q_row * num_heads * max_num_partitions +
                              int64_t(head_idx) * max_num_partitions +
                              split_idx;
          exp_sums[idx] = 0.f;
          max_logits[idx] = -FLT_MAX;
        }
      }
    } else {
      for (int r = 0; r < TILE; r++) {
        const int q_local = tile_local0 + r;
        if (q_local >= max_q_len) {
          continue;
        }
        device T *out_ptr = out +
                            (int64_t(seq_idx) * max_q_len + q_local) *
                                num_heads * HEAD_SIZE +
                            int64_t(head_idx) * HEAD_SIZE;
        for (int i = thread_idx; i < HEAD_SIZE; i += NUM_THREADS) {
          out_ptr[i] = T(0);
        }
      }
    }
    return;
  }

  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope = !use_alibi ? 0.f : alibi_slopes[head_idx];
  constexpr int x = 16 / sizeof(CACHE_T);

  // Park the query tile in registers, one strip at a time through kv_smem.
  simdgroup_float8x8 q_regs[KSTEPS];
  for (int s = 0; s < STRIPS; s++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = thread_idx; i < 8 * HEAD_SIZE; i += NUM_THREADS) {
      const int r = s * 8 + i / HEAD_SIZE;
      const int d = i % HEAD_SIZE;
      const int q_local = tile_local0 + r;
      float qv = 0.f;
      if (q_local < max_q_len && ctx_smem[r] > 0) {
        qv = float(q[(int64_t(seq_idx) * max_q_len + q_local) * q_stride +
                     int64_t(head_idx) * HEAD_SIZE + d]);
      }
      kv_smem[i] = qv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp_idx / KHALVES == s) {
#pragma unroll
      for (int kk = 0; kk < KSTEPS; kk++) {
        const int d0 = ((warp_idx % KHALVES) * KSTEPS + kk) * 8;
        simdgroup_load(q_regs[kk], kv_smem + d0, HEAD_SIZE);
      }
    }
  }

  simdgroup_float8x8 o_acc[O_TILES_PER_WARP];
#pragma unroll
  for (int i = 0; i < O_TILES_PER_WARP; i++) {
    o_acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
  }

  const device uint32_t *block_table =
      block_tables + int64_t(seq_idx) * max_num_blocks_per_seq;
  const int last_ctx_block = DIVIDE_ROUND_UP(tile_max_ctx, BLOCK_SIZE) - 1;
  const int ctx_blocks_end = (last_ctx_block + 1) * BLOCK_SIZE;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int walk0 = w0 * WINDOW;
  const int walk_end = MIN(w1 * WINDOW, ctx_blocks_end);
  {
    const int span_end = walk_end;
    for (int t0 = walk0; t0 < span_end; t0 += BK) {
      // Stage K transposed: kv_smem[d * BK + t]; tokens past the last
      // context block have no physical block and stage zeros.
      constexpr int KSTAGE = BK * HEAD_SIZE / NUM_THREADS;
      static_assert(HEAD_SIZE % KSTAGE == 0, "K runs must not span tokens");
      if constexpr (!is_uchar<CACHE_T>() && KSTAGE % x == 0 && x % 4 == 0) {
        // Vector path: the thread's run is whole x-blocks of one token —
        // 4-wide loads from the cache, 4-wide stores to the [t][d] slab.
        const int i0 = thread_idx * KSTAGE;
        const int t = i0 / HEAD_SIZE;
        const int d0 = i0 % HEAD_SIZE;
        const int tok = t0 + t;
        if (tok < span_end) {
          const int64_t pb = int64_t(block_table[tok / BLOCK_SIZE]);
          const device CACHE_T *k_ptr = (const device CACHE_T *)k_addrs[pb] +
                                        kv_head_idx * kv_head_stride +
                                        (tok % BLOCK_SIZE) * x;
#pragma unroll
          for (int u = 0; u < KSTAGE / 4; u++) {
            const int d = d0 + u * 4;
            const auto v4 = *reinterpret_cast<const device vec<CACHE_T, 4> *>(
                k_ptr + (d / x) * BLOCK_SIZE * x + d % x);
            *reinterpret_cast<threadgroup float4 *>(kv_smem + t * HEAD_SIZE +
                                                    d) = float4(v4);
          }
        } else {
#pragma unroll
          for (int u = 0; u < KSTAGE / 4; u++) {
            *reinterpret_cast<threadgroup float4 *>(
                kv_smem + t * HEAD_SIZE + d0 + u * 4) = float4(0.f);
          }
        }
      } else {
        for (int i = thread_idx; i < BK * HEAD_SIZE; i += NUM_THREADS) {
          const int t = i / HEAD_SIZE;
          const int d = i % HEAD_SIZE;
          const int tok = t0 + t;
          float kv = 0.f;
          if (tok < span_end) {
            const int64_t pb = int64_t(block_table[tok / BLOCK_SIZE]);
            const device CACHE_T *k_ptr =
                (const device CACHE_T *)k_addrs[pb] + kv_head_idx * kv_head_stride;
            const int idx =
                (d / x) * BLOCK_SIZE * x + (tok % BLOCK_SIZE) * x + d % x;
            if constexpr (is_uchar<CACHE_T>()) {
              kv = __dequant_single(k_ptr[idx], *k_scale);
            } else {
              kv = float(k_ptr[idx]);
            }
          }
          kv_smem[t * HEAD_SIZE + d] = kv;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // S = Q.Kt, one strip x k-half per warp, partials summed later.
      {
        const int s = warp_idx / KHALVES;
        const int h = warp_idx % KHALVES;
#pragma unroll
        for (int c = 0; c < BK / 8; c++) {
          simdgroup_float8x8 acc =
              make_filled_simdgroup_matrix<float, 8, 8>(0.f);
#pragma unroll
          for (int kk = 0; kk < KSTEPS; kk++) {
            const int d0 = (h * KSTEPS + kk) * 8;
            simdgroup_float8x8 b;
            simdgroup_load(b, kv_smem + (c * 8) * HEAD_SIZE + d0, HEAD_SIZE,
                           0, true);
            simdgroup_multiply_accumulate(acc, q_regs[kk], b, acc);
          }
          simdgroup_store(acc, &sp_smem[h][(s * 8) * BK + c * 8], BK);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Stage V as [t][d] into the region K just vacated; values past the
      // tile's context are zeroed (the cache may hold NaNs there).
      if constexpr (!is_uchar<CACHE_T>() && BK % 4 == 0 &&
                    KSTAGE % BK == 0 && BLOCK_SIZE % BK == 0) {
        // Vector path: the thread's run is whole token spans of KSTAGE/BK dims. BK must divide BLOCK_SIZE so the span's
        // single block-table lookup is right; smaller blocks take the scalar path.
        const int i0 = thread_idx * KSTAGE;
        const int d0 = i0 / BK;
        const int64_t pb = int64_t(block_table[t0 / BLOCK_SIZE]);
        const device CACHE_T *v_ptr = (const device CACHE_T *)v_addrs[pb] +
                                      kv_head_idx * kv_head_stride +
                                      (t0 % BLOCK_SIZE);
        const int lim = tile_max_ctx - t0;
#pragma unroll
        for (int dd = 0; dd < KSTAGE / BK; dd++) {
          const int d = d0 + dd;
          float vv[BK];
#pragma unroll
          for (int u = 0; u < BK / 4; u++) {
            const auto v4 = *reinterpret_cast<const device vec<CACHE_T, 4> *>(
                v_ptr + d * BLOCK_SIZE + u * 4);
            const float4 f4 = float4(v4);
#pragma unroll
            for (int w = 0; w < 4; w++) {
              const int t = u * 4 + w;
              vv[t] = t < lim ? f4[w] : 0.f;
            }
          }
#pragma unroll
          for (int u = 0; u < BK / 4; u++) {
            *reinterpret_cast<threadgroup float4 *>(kv_smem + d * BK + u * 4) =
                float4(vv[u * 4], vv[u * 4 + 1], vv[u * 4 + 2], vv[u * 4 + 3]);
          }
        }
      } else {
        for (int i = thread_idx; i < BK * HEAD_SIZE; i += NUM_THREADS) {
          const int t = i / HEAD_SIZE;
          const int d = i % HEAD_SIZE;
          const int tok = t0 + t;
          float vv = 0.f;
          if (tok < tile_max_ctx) {
            const int64_t pb = int64_t(block_table[tok / BLOCK_SIZE]);
            const device CACHE_T *v_ptr =
                (const device CACHE_T *)v_addrs[pb] + kv_head_idx * kv_head_stride;
            const int idx = d * BLOCK_SIZE + tok % BLOCK_SIZE;
            if constexpr (is_uchar<CACHE_T>()) {
              vv = __dequant_single(v_ptr[idx], *v_scale);
            } else {
              vv = float(v_ptr[idx]);
            }
          }
          kv_smem[d * BK + t] = vv;
        }
      }

      // Mask, scale, softcap, alibi, and the per-row online update; one
      // thread per query row. sp_smem[0] leaves this pass holding P.
      if (thread_idx < TILE) {
        const int r = thread_idx;
        const int ctx = ctx_smem[r];
        float row_max = -FLT_MAX;
#pragma unroll
        for (int c = 0; c < BK; c++) {
          const int tok = t0 + c;
          float sv = sp_smem[0][r * BK + c];
#pragma unroll
          for (int h = 1; h < KHALVES; h++) {
            sv += sp_smem[h][r * BK + c];
          }
          sv *= scale;
          if (softcapping != 1.0) {
            sv = tanh(sv / softcapping) * softcapping;
          }
          sv += (alibi_slope != 0) ? alibi_slope * (tok - ctx + 1) : 0;
          const bool mask = tok >= ctx;
          sp_smem[0][r * BK + c] = mask ? -FLT_MAX : sv;
          row_max = mask ? row_max : max(row_max, sv);
        }
        if (row_max == -FLT_MAX) {
          // Nothing visible for this row in this K-tile: P is zero and the running state is untouched.
          f_smem[r] = 1.f;
#pragma unroll
          for (int c = 0; c < BK; c++) {
            sp_smem[0][r * BK + c] = 0.f;
          }
        } else {
          const float m_new = max(m_smem[r], row_max);
          // With an empty accumulator (m == -FLT_MAX) l and O are exactly zero, so f = 1 skips a pointless rescale.
          const float f =
              m_smem[r] == -FLT_MAX ? 1.f : exp(m_smem[r] - m_new);
          float sum = 0.f;
#pragma unroll
          for (int c = 0; c < BK; c++) {
            const float lv = sp_smem[0][r * BK + c];
            const float e = lv == -FLT_MAX ? 0.f : exp(lv - m_new);
            sp_smem[0][r * BK + c] = e;
            sum += e;
          }
          m_smem[r] = m_new;
          l_smem[r] = l_smem[r] * f + sum;
          f_smem[r] = f;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Rescale O where the running max moved, then O += P.V. Each warp owns (strip, column half) for the whole kernel,
      // so o_acc never crosses warps.
      {
        const int s = warp_idx / (NUM_WARPS / STRIPS);
        const int ch = warp_idx % (NUM_WARPS / STRIPS);
        const float fv = lane < 8 ? f_smem[s * 8 + lane] : 1.f;
        if (simd_any(fv != 1.f)) {
          for (int j = lane; j < 64; j += NUM_SIMD_LANES) {
            diag_smem[warp_idx][j] =
                (j / 8 == j % 8) ? f_smem[s * 8 + j / 8] : 0.f;
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          simdgroup_float8x8 dm;
          simdgroup_load(dm, diag_smem[warp_idx], 8);
#pragma unroll
          for (int ot = 0; ot < O_TILES_PER_WARP; ot++) {
            simdgroup_float8x8 tmp;
            simdgroup_multiply(tmp, dm, o_acc[ot]);
            o_acc[ot] = tmp;
          }
        }
#pragma unroll
        for (int ot = 0; ot < O_TILES_PER_WARP; ot++) {
          const int d0 = (ch * O_TILES_PER_WARP + ot) * 8;
#pragma unroll
          for (int kk = 0; kk < BK / 8; kk++) {
            simdgroup_float8x8 p, v;
            simdgroup_load(p, &sp_smem[0][(s * 8) * BK + kk * 8], BK);
            simdgroup_load(v, kv_smem + d0 * BK + kk * 8, BK, 0, true);
            simdgroup_multiply_accumulate(o_acc[ot], p, v, o_acc[ot]);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // Final store, one strip at a time through kv_smem: the strip's warps
  // park their O tiles, then every thread normalizes and writes rows.
  for (int so = 0; so < STRIPS; so++) {
    if (warp_idx / (NUM_WARPS / STRIPS) == so) {
      const int ch = warp_idx % (NUM_WARPS / STRIPS);
#pragma unroll
      for (int ot = 0; ot < O_TILES_PER_WARP; ot++) {
        const int d0 = (ch * O_TILES_PER_WARP + ot) * 8;
        simdgroup_store(o_acc[ot], kv_smem + d0, HEAD_SIZE);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = thread_idx; i < 8 * HEAD_SIZE; i += NUM_THREADS) {
      const int r8 = i / HEAD_SIZE;
      const int d = i % HEAD_SIZE;
      const int r = so * 8 + r8;
      const int q_local = tile_local0 + r;
      if (q_local >= max_q_len) {
        continue;
      }
      const int64_t q_row = int64_t(seq_idx) * max_q_len + q_local;
      float m = m_smem[r];
      float l = l_smem[r];
      float out_scale = 1.f;
      // Non-partitioned mode folds the sink once at the end; partitioned mode leaves it to the reduce kernel.
      if (!USE_PARTITIONING && use_sinks && ctx_smem[r] > 0) {
        const float s = sinks[head_idx];
        const float m2 = max(m, s);
        out_scale = m == -FLT_MAX ? 0.f : exp(m - m2);
        l = l * out_scale + exp(s - m2);
        m = m2;
      }
      const float inv = l > 0.f ? 1.f / (l + 1e-6f) : 0.f;
      device T *out_ptr;
      if (USE_PARTITIONING) {
        if (d == 0) {
          const int64_t idx = q_row * num_heads * max_num_partitions +
                              int64_t(head_idx) * max_num_partitions +
                              split_idx;
          exp_sums[idx] = l;
          max_logits[idx] = m;
        }
        out_ptr = out + q_row * num_heads * max_num_partitions * HEAD_SIZE +
                  int64_t(head_idx) * max_num_partitions * HEAD_SIZE +
                  int64_t(split_idx) * HEAD_SIZE;
      } else {
        out_ptr = out + q_row * num_heads * HEAD_SIZE +
                  int64_t(head_idx) * HEAD_SIZE;
      }
      out_ptr[d] = T(kv_smem[r8 * HEAD_SIZE + d] * out_scale * inv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// The split merge: one threadgroup per (row, head) folds the splits' normalized partials by
// their rescaled exp-sums, the simdgroups striding the splits (MLX's sdpa_vector_2pass_2 shape)
// and a 32 x 32 transpose through threadgroup memory summing them, one round per 32 dims.
template <typename T, int HEAD_SIZE, int NUM_THREADS, int NUM_SIMD_LANES,
          int PARTITION_SIZE = 0>
[[kernel]] void paged_attention_v2_reduce(
    device T *out [[buffer(0)]], const device float *exp_sums [[buffer(1)]],
    const device float *max_logits [[buffer(2)]],
    const device T *tmp_out [[buffer(3)]],
    const constant int &max_num_partitions [[buffer(4)]],
    const device float *sinks
    [[buffer(5), function_constant(use_sinks)]], // [num_heads]
    threadgroup char *shared_mem [[threadgroup(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int NUM_WARPS = NUM_THREADS / NUM_SIMD_LANES;
  static_assert(NUM_WARPS == NUM_SIMD_LANES,
                "the cross-simdgroup transpose is square");
  constexpr int DIMS_PER_LANE = DIVIDE_ROUND_UP(HEAD_SIZE, NUM_SIMD_LANES);
  const int num_heads = threadgroups_per_grid.x;
  const int head_idx = threadgroup_position_in_grid.x;
  const int seq_idx = threadgroup_position_in_grid.y;
  const int thread_idx = thread_position_in_threadgroup.x;
  const int warp_idx = simd_tid;
  const int lane = simd_lid;
  // Splits a row never reached carry exp_sum == 0 / max_logit == -FLT_MAX and an unwritten tmp_out slice, so every path
  // below must treat a zero exp_sum as absent rather than multiply by it.
  const int num_partitions = max_num_partitions;
  const int64_t head_row = int64_t(seq_idx) * num_heads + head_idx;
  device T *out_ptr = out + head_row * HEAD_SIZE;
  const device T *tmp_out_ptr = tmp_out + head_row * max_num_partitions * HEAD_SIZE;
  if (num_partitions == 1 && !use_sinks) {
    // No need to reduce. Only copy tmp_out to out — unless the single split
    // was empty (a padding row), which writes zeros.
    const bool empty = exp_sums[head_row] == 0.f;
    for (int i = thread_idx; i < HEAD_SIZE; i += NUM_THREADS) {
      out_ptr[i] = empty ? T(0) : tmp_out_ptr[i];
    }
    return;
  }

  threadgroup float red_smem[2 * NUM_WARPS];
  threadgroup float transpose[NUM_SIMD_LANES * NUM_WARPS];

  // The global max over the splits (and the sink), through shared memory.
  threadgroup float *shared_max_logits =
      reinterpret_cast<threadgroup float *>(shared_mem);
  const device float *max_logits_ptr =
      max_logits + head_row * max_num_partitions;
  float max_logit = -FLT_MAX;
  for (int i = thread_idx; i < num_partitions; i += NUM_THREADS) {
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l;
    max_logit = max(max_logit, l);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
  for (int mask = NUM_SIMD_LANES / 2; mask >= 1; mask /= 2) {
    max_logit = max(max_logit, simd_shuffle_xor(max_logit, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = max_logit;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    max_logit = max(max_logit, simd_shuffle_xor(max_logit, mask));
  }
  max_logit = simd_shuffle(max_logit, 0);
  if (use_sinks) {
    max_logit = max(max_logit, sinks[head_idx]);
  }

  // Each split's exp-sum rescaled to the global max, and their total.
  threadgroup float *shared_exp_sums = reinterpret_cast<threadgroup float *>(
      shared_mem + sizeof(float) * num_partitions);
  const device float *exp_sums_ptr = exp_sums + head_row * max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = thread_idx; i < num_partitions; i += NUM_THREADS) {
    const float e = exp_sums_ptr[i];
    const float rescaled = e == 0.f ? 0.f : e * exp(shared_max_logits[i] - max_logit);
    global_exp_sum += rescaled;
    shared_exp_sums[i] = rescaled;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  global_exp_sum = block_sum<NUM_WARPS, NUM_SIMD_LANES>(
      &red_smem[NUM_WARPS], global_exp_sum, simd_tid, simd_lid);
  if (use_sinks) {
    global_exp_sum += exp(sinks[head_idx] - max_logit);
  }
  const float inv_global_exp_sum = 1.0f / (global_exp_sum + 1e-6f);

  // Simdgroup w folds splits w, w + NUM_WARPS, ... for this lane's dims, skipping splits that never contributed.
  float acc[DIMS_PER_LANE];
#pragma unroll
  for (int k = 0; k < DIMS_PER_LANE; k++) {
    acc[k] = 0.f;
  }
  for (int j = warp_idx; j < num_partitions; j += NUM_WARPS) {
    const float w = shared_exp_sums[j] * inv_global_exp_sum;
    if (w == 0.f) {
      continue;
    }
    const device T *part = tmp_out_ptr + int64_t(j) * HEAD_SIZE;
#pragma unroll
    for (int k = 0; k < DIMS_PER_LANE; k++) {
      const int d = lane + k * NUM_SIMD_LANES;
      if (d < HEAD_SIZE) {
        acc[k] += w * float(part[d]);
      }
    }
  }

  // Sum the simdgroups: round k transposes the partials of dims
  // k * 32 + lane so simdgroup w's simd_sum lands on dim k * 32 + w.
#pragma unroll
  for (int k = 0; k < DIMS_PER_LANE; k++) {
    transpose[lane * NUM_WARPS + warp_idx] = acc[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = simd_sum(transpose[warp_idx * NUM_SIMD_LANES + lane]);
    const int d = warp_idx + k * NUM_SIMD_LANES;
    if (lane == 0 && d < HEAD_SIZE) {
      out_ptr[d] = T(total);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#define instantiate_paged_attention_v2_reduce_inner(                           \
    type, head_size, num_threads, num_simd_lanes, partition_size)              \
  template [[host_name("paged_attention_v2_reduce_" #type "_hs" #head_size     \
                       "_nt" #num_threads "_nsl" #num_simd_lanes               \
                       "_ps" #partition_size)]] [[kernel]] void                \
  paged_attention_v2_reduce<type, head_size, num_threads, num_simd_lanes,      \
                            partition_size>(                                   \
      device type * out [[buffer(0)]],                                         \
      const device float *exp_sums [[buffer(1)]],                              \
      const device float *max_logits [[buffer(2)]],                            \
      const device type *tmp_out [[buffer(3)]],                                \
      const constant int &max_num_partitions [[buffer(4)]],                    \
      const device float *sinks [[buffer(5), function_constant(use_sinks)]],   \
      threadgroup char *shared_mem [[threadgroup(0)]],                         \
      uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],     \
      uint3 threadgroups_per_grid [[threadgroups_per_grid]],                   \
      uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], \
      uint3 threads_per_threadgroup [[threads_per_threadgroup]],               \
      uint simd_tid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_paged_attention_group_inner(type, cache_type, head_size,    \
                                                block_size, num_simd_lanes,    \
                                                partition_size)                \
  template [[host_name("paged_attention_group_" #type "_cache_" #cache_type    \
                       "_hs" #head_size "_bs" #block_size                      \
                       "_nsl" #num_simd_lanes "_ps" #partition_size)]]         \
  [[kernel]] void paged_attention_group<type, cache_type, head_size,           \
                                        block_size, num_simd_lanes,            \
                                        partition_size>(                       \
      device float *exp_sums                                                   \
      [[buffer(0), function_constant(use_partitioning)]],                      \
      device float *max_logits                                                 \
      [[buffer(1), function_constant(use_partitioning)]],                      \
      device type *out [[buffer(2)]], device const type *q [[buffer(3)]],      \
      device const uint64_t *k_addrs [[buffer(4)]],                             \
      device const uint64_t *v_addrs [[buffer(5)]],                             \
      device const float *k_scale                                              \
      [[buffer(6), function_constant(use_fp8_scales)]],                        \
      device const float *v_scale                                              \
      [[buffer(7), function_constant(use_fp8_scales)]],                        \
      const constant int &num_kv_heads [[buffer(8)]],                          \
      const constant float &scale [[buffer(9)]],                               \
      const constant float &softcapping [[buffer(10)]],                        \
      device const uint32_t *block_tables [[buffer(11)]],                      \
      device const uint32_t *context_lens [[buffer(12)]],                      \
      const constant int &max_num_blocks_per_seq [[buffer(13)]],               \
      device const float *alibi_slopes                                         \
      [[buffer(14), function_constant(use_alibi)]],                            \
      const constant int &q_stride [[buffer(15)]],                             \
      const constant int &kv_head_stride [[buffer(17)]],                       \
      const device float *sinks [[buffer(18), function_constant(use_sinks)]],  \
      const constant int &windows_per_split [[buffer(20)]],                    \
      threadgroup char *shared_mem [[threadgroup(0)]],                         \
      uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],     \
      uint3 threadgroups_per_grid [[threadgroups_per_grid]],                   \
      uint3 threads_per_threadgroup [[threads_per_threadgroup]],               \
      uint simd_tid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]);

// Decode shape: the group kernel, at the head sizes the models carry.
#define instantiate_paged_attention_group_heads(                               \
    type, cache_type, block_size, num_simd_lanes, partition_size)              \
  instantiate_paged_attention_group_inner(type, cache_type, 64, block_size,    \
                                          num_simd_lanes, partition_size);    \
  instantiate_paged_attention_group_inner(type, cache_type, 128, block_size,   \
                                          num_simd_lanes, partition_size);    \
  instantiate_paged_attention_group_inner(type, cache_type, 256, block_size,   \
                                          num_simd_lanes, partition_size);

#define instantiate_paged_attention_mma_inner(type, cache_type, head_size,      \
                                              block_size, num_threads,          \
                                              num_simd_lanes, partition_size,   \
                                              tile)                             \
  template [[host_name("paged_attention_mma_" #type "_cache_" #cache_type      \
                       "_hs" #head_size "_bs" #block_size "_nt" #num_threads   \
                       "_nsl" #num_simd_lanes                                   \
                       "_ps" #partition_size "_t" #tile)]] [[kernel]] void     \
  paged_attention_mma<type, cache_type, head_size, block_size, num_threads,    \
                      num_simd_lanes, partition_size, tile>(                    \
      device float *exp_sums                                                   \
      [[buffer(0), function_constant(use_partitioning)]],                       \
      device float *max_logits                                                 \
      [[buffer(1), function_constant(use_partitioning)]],                       \
      device type *out [[buffer(2)]], device const type *q [[buffer(3)]],      \
      device const uint64_t *k_addrs [[buffer(4)]],                             \
      device const uint64_t *v_addrs [[buffer(5)]],                             \
      device const float *k_scale                                              \
      [[buffer(6), function_constant(use_fp8_scales)]],                         \
      device const float *v_scale                                              \
      [[buffer(7), function_constant(use_fp8_scales)]],                         \
      const constant int &num_kv_heads [[buffer(8)]],                          \
      const constant float &scale [[buffer(9)]],                               \
      const constant float &softcapping [[buffer(10)]],                        \
      device const uint32_t *block_tables [[buffer(11)]],                      \
      device const uint32_t *context_lens [[buffer(12)]],                      \
      const constant int &max_num_blocks_per_seq [[buffer(13)]],               \
      device const float *alibi_slopes                                         \
      [[buffer(14), function_constant(use_alibi)]],                             \
      const constant int &q_stride [[buffer(15)]],                             \
      const constant int &kv_head_stride [[buffer(17)]],                       \
      const device float *sinks [[buffer(18), function_constant(use_sinks)]],  \
      const constant int &max_q_len [[buffer(19)]],                            \
      const constant int &windows_per_split [[buffer(20)]],                    \
      uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],     \
      uint3 threadgroups_per_grid [[threadgroups_per_grid]],                   \
      uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], \
      uint simd_tid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]);

// Prompt shape: the simdgroup-matrix kernel, a 16-row tile at 128 threads.
#define instantiate_paged_attention_tile_heads(                                \
    type, cache_type, block_size, num_simd_lanes, partition_size)              \
  instantiate_paged_attention_mma_inner(type, cache_type, 64, block_size,      \
                                        128, num_simd_lanes,                   \
                                        partition_size, 16);                   \
  instantiate_paged_attention_mma_inner(type, cache_type, 128, block_size,     \
                                        128, num_simd_lanes,                   \
                                        partition_size, 16);                   \
  instantiate_paged_attention_mma_inner(type, cache_type, 256, block_size,     \
                                        128, num_simd_lanes,                   \
                                        partition_size, 16);

#define instantiate_paged_attention_v2_reduce_heads(                           \
    type, num_threads, num_simd_lanes, partition_size)                         \
  instantiate_paged_attention_v2_reduce_inner(type, 64, num_threads,           \
                                              num_simd_lanes, partition_size); \
  instantiate_paged_attention_v2_reduce_inner(type, 128, num_threads,          \
                                              num_simd_lanes, partition_size); \
  instantiate_paged_attention_v2_reduce_inner(type, 256, num_threads,          \
                                              num_simd_lanes, partition_size);

#define instantiate_paged_attention_block_size(type, cache_type,              \
                                               num_simd_lanes, partition_size) \
  instantiate_paged_attention_group_heads(type, cache_type, 8,                 \
                                          num_simd_lanes, partition_size);     \
  instantiate_paged_attention_group_heads(type, cache_type, 16,                \
                                          num_simd_lanes, partition_size);     \
  instantiate_paged_attention_group_heads(type, cache_type, 32,                \
                                          num_simd_lanes, partition_size);     \
  instantiate_paged_attention_tile_heads(type, cache_type, 8,                  \
                                         num_simd_lanes, partition_size);      \
  instantiate_paged_attention_tile_heads(type, cache_type, 16,                 \
                                         num_simd_lanes, partition_size);      \
  instantiate_paged_attention_tile_heads(type, cache_type, 32,                 \
                                         num_simd_lanes, partition_size);

// NOTE: partition_size = 0
#define instantiate_paged_attention_v1(type, cache_type, num_simd_lanes)       \
  instantiate_paged_attention_block_size(type, cache_type, num_simd_lanes, 0);

// NOTE: partition_size = 512
#define instantiate_paged_attention_v2(type, cache_type, num_simd_lanes)       \
  instantiate_paged_attention_block_size(type, cache_type, num_simd_lanes, 512);

// The reduce runs 32 simdgroups: its cross-simdgroup transpose is square.
#define instantiate_paged_attention_v2_reduce(type, num_simd_lanes)            \
  instantiate_paged_attention_v2_reduce_heads(type, 1024, num_simd_lanes, 512);

instantiate_paged_attention_v1(float, float, 32);
instantiate_paged_attention_v1(bfloat16_t, bfloat16_t, 32);

instantiate_paged_attention_v2_reduce(float, 32);
instantiate_paged_attention_v2_reduce(bfloat16_t, 32);

instantiate_paged_attention_v2(float, float, 32);
instantiate_paged_attention_v2(bfloat16_t, bfloat16_t, 32);

// ========================================== Writing a token's K/V into its block

template <typename KV_T, typename CACHE_T>
inline CACHE_T to_cache(KV_T v) = delete;

template <> inline float to_cache<float, float>(float v) { return v; }

template <> inline bfloat16_t to_cache<bfloat16_t, bfloat16_t>(bfloat16_t v) {
  return v;
}

template <> inline half to_cache<half, half>(half v) { return v; }

constant bool scale_fp8 [[function_constant(50)]];

template <typename KV_T, typename CACHE_T>
[[kernel]] void reshape_and_cache(
    const device KV_T *__restrict__ key
    [[buffer(0)]], // [num_tokens, num_heads, head_size]
    const device KV_T *__restrict__ value
    [[buffer(1)]], // [num_tokens, num_heads, head_size]
    const device uint64_t *__restrict__ k_addrs
    [[buffer(2)]], // [num_blocks]: each block's K row
                   // [num_heads, head_size/x, block_size, x]
    const device uint64_t *__restrict__ v_addrs
    [[buffer(3)]], // [num_blocks]: each block's V row
                   // [num_heads, head_size, block_size]
    const device int64_t *__restrict__ slot_mapping
    [[buffer(4)]], // [num_tokens]
    const device float *__restrict__ k_scale
    [[buffer(5), function_constant(scale_fp8)]], // [1]
    const device float *__restrict__ v_scale
    [[buffer(6), function_constant(scale_fp8)]], // [1]
    device const int &key_stride [[buffer(7)]],
    device const int &value_stride [[buffer(8)]],
    device const int &num_heads [[buffer(9)]],
    device const int &head_size [[buffer(10)]],
    device const int &block_size [[buffer(11)]],
    device const int &x [[buffer(12)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]) {
  const int64_t token_idx = gid;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  device CACHE_T *key_cache = (device CACHE_T *)k_addrs[block_idx];
  device CACHE_T *value_cache = (device CACHE_T *)v_addrs[block_idx];

  const int n = num_heads * head_size;
  for (int i = tid; i < n; i += threads_per_threadgroup) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int64_t tgt_key_idx = head_idx * (head_size / x) * block_size * x +
                                x_idx * block_size * x + block_offset * x +
                                x_offset;
    const int64_t tgt_value_idx =
        head_idx * head_size * block_size + head_offset * block_size +
        block_offset;

    if (scale_fp8) {
      key_cache[tgt_key_idx] =
          to_cache<KV_T, CACHE_T>(KV_T((float)key[src_key_idx] / *k_scale));
      value_cache[tgt_value_idx] =
          to_cache<KV_T, CACHE_T>(KV_T((float)value[src_value_idx] / *v_scale));
    } else {
      key_cache[tgt_key_idx] = to_cache<KV_T, CACHE_T>(key[src_key_idx]);
      value_cache[tgt_value_idx] =
          to_cache<KV_T, CACHE_T>(value[src_value_idx]);
    }
  }
}

#define instantiate_reshape_and_cache(kv_type, cache_type)                     \
  template [[host_name("reshape_and_cache_kv_" #kv_type                        \
                       "_cache_" #cache_type)]] [[kernel]] void                \
  reshape_and_cache<kv_type, cache_type>(                                      \
      const device kv_type *__restrict__ key [[buffer(0)]],                    \
      const device kv_type *__restrict__ value [[buffer(1)]],                  \
      const device uint64_t *__restrict__ k_addrs [[buffer(2)]],               \
      const device uint64_t *__restrict__ v_addrs [[buffer(3)]],               \
      const device int64_t *__restrict__ slot_mapping [[buffer(4)]],           \
      const device float *__restrict__ k_scale                                 \
      [[buffer(5), function_constant(scale_fp8)]],                        \
      const device float *__restrict__ v_scale                                 \
      [[buffer(6), function_constant(scale_fp8)]],                        \
      device const int &key_stride [[buffer(7)]],                              \
      device const int &value_stride [[buffer(8)]],                            \
      device const int &num_heads [[buffer(9)]],                               \
      device const int &head_size [[buffer(10)]],                              \
      device const int &block_size [[buffer(11)]],                             \
      device const int &x [[buffer(12)]],                                      \
      uint gid [[threadgroup_position_in_grid]],                               \
      uint tid [[thread_position_in_threadgroup]],                             \
      uint threads_per_threadgroup [[threads_per_threadgroup]]);

instantiate_reshape_and_cache(float, float);
instantiate_reshape_and_cache(bfloat16_t, bfloat16_t);
instantiate_reshape_and_cache(half, half);

// ========================================== Gathering a sequence's rows

// Gather one sequence's K and V rows out of the block pool into contiguous
// [num_kv_heads, tokens, head_size] tensors, the prompt path's SDPA input: one threadgroup per
// token, the block's rows reached through the layer's address tables. A pure copy in the cache dtype.
template <typename T>
[[kernel]] void gather_kv_cache(
    const device uint64_t *__restrict__ k_addrs
    [[buffer(0)]], // [num_blocks]: each block's K row
                   // [num_kv_heads, head_size/x, block_size, x]
    const device uint64_t *__restrict__ v_addrs
    [[buffer(1)]], // [num_blocks]: each block's V row
                   // [num_kv_heads, head_size, block_size]
    device T *__restrict__ k_out [[buffer(2)]], // [num_kv_heads, tokens, head_size]
    device T *__restrict__ v_out [[buffer(3)]], // [num_kv_heads, tokens, head_size]
    const device uint32_t *__restrict__ block_table
    [[buffer(4)]], // [tokens / block_size]
    const constant int &tokens [[buffer(5)]],
    const constant int &block_size [[buffer(6)]],
    const constant int &num_kv_heads [[buffer(7)]],
    const constant int &head_size [[buffer(8)]],
    const constant int &x [[buffer(9)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]) {
  const int t = gid;
  const int64_t pb = block_table[t / block_size];
  const int slot = t % block_size;
  const device T *k_row = (const device T *)k_addrs[pb];
  const device T *v_row = (const device T *)v_addrs[pb];
  const int n = num_kv_heads * head_size;
  for (int i = tid; i < n; i += threads_per_threadgroup) {
    const int h = i / head_size;
    const int d = i % head_size;
    const int64_t out = (int64_t(h) * tokens + t) * head_size + d;
    k_out[out] = k_row[(h * (head_size / x) + d / x) * block_size * x +
                       slot * x + d % x];
    v_out[out] = v_row[(h * head_size + d) * block_size + slot];
  }
}

#define instantiate_gather_kv_cache(type)                                      \
  template [[host_name("gather_kv_cache_" #type)]] [[kernel]] void             \
  gather_kv_cache<type>(                                                       \
      const device uint64_t *__restrict__ k_addrs [[buffer(0)]],               \
      const device uint64_t *__restrict__ v_addrs [[buffer(1)]],               \
      device type *__restrict__ k_out [[buffer(2)]],                           \
      device type *__restrict__ v_out [[buffer(3)]],                           \
      const device uint32_t *__restrict__ block_table [[buffer(4)]],           \
      const constant int &tokens [[buffer(5)]],                                \
      const constant int &block_size [[buffer(6)]],                            \
      const constant int &num_kv_heads [[buffer(7)]],                          \
      const constant int &head_size [[buffer(8)]],                             \
      const constant int &x [[buffer(9)]],                                     \
      uint gid [[threadgroup_position_in_grid]],                               \
      uint tid [[thread_position_in_threadgroup]],                             \
      uint threads_per_threadgroup [[threads_per_threadgroup]]);

instantiate_gather_kv_cache(float);
instantiate_gather_kv_cache(bfloat16_t);
instantiate_gather_kv_cache(half);
