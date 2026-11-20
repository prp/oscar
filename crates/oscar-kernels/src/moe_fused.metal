// The routed-MoE kernels, the quantised matvecs, the router and its mailbox publish, and the drafter's row checks.

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// Always inlined: the compiler's choice counts a helper's callers, so a new call site would move other kernels' bytes.
#define OSCAR_INLINE inline __attribute__((always_inline))

// Q4_K as the GGUF lays it out, llama.cpp's `block_q4_K`.
#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
  union {
    struct {
      half d;     // super-block scale for quantized scales
      half dmin;  // super-block scale for quantized mins
    };
    half2 dm;
  };
  uint8_t scales[K_SCALE_SIZE];  // scales and mins, quantized with 6 bits
  uint8_t qs[QK_K / 2];           // 4-bit quants
} block_q4_K;

// One row's partial dot over one Q4_K super-block, as candle's `kernel_mul_mv_q4_K_f32_impl` computes it.
static OSCAR_INLINE float q4k_block_dot(
    thread const float yl[16],
    thread const float yh[16],
    float4 sumy,
    device const uint16_t *q1,
    device const uint16_t *sc,
    device const half *dh) {
  const uint16_t kmask1 = 0x3f3f;
  const uint16_t kmask2 = 0x0f0f;
  const uint16_t kmask3 = 0xc0c0;

  uint16_t sc16[4];
  thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

  sc16[0] = sc[0] & kmask1;
  sc16[1] = sc[2] & kmask1;
  sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
  sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

  device const uint16_t *q2 = q1 + 32;

  float4 acc1 = {0.f, 0.f, 0.f, 0.f};
  float4 acc2 = {0.f, 0.f, 0.f, 0.f};
  for (int i = 0; i < 8; i += 2) {
    acc1[0] += yl[i + 0] * (q1[i / 2] & 0x000F);
    acc1[1] += yl[i + 1] * (q1[i / 2] & 0x0F00);
    acc1[2] += yl[i + 8] * (q1[i / 2] & 0x00F0);
    acc1[3] += yl[i + 9] * (q1[i / 2] & 0xF000);
    acc2[0] += yh[i + 0] * (q2[i / 2] & 0x000F);
    acc2[1] += yh[i + 1] * (q2[i / 2] & 0x0F00);
    acc2[2] += yh[i + 8] * (q2[i / 2] & 0x00F0);
    acc2[3] += yh[i + 9] * (q2[i / 2] & 0xF000);
  }

  float dall = dh[0];
  float dmin = dh[1];
  return dall * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                 (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                 (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                 (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
         dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] +
                 sumy[3] * sc8[7]);
}

// Q6_K as the GGUF lays it out, llama.cpp's `block_q6_K`.
typedef struct {
  uint8_t ql[QK_K / 2];      // lower 4 bits of quants
  uint8_t qh[QK_K / 4];      // upper 2 bits of quants
  int8_t scales[QK_K / 16];  // 8-bit per-16 scales
  half d;                    // super-block scale
} block_q6_K;

// Q3_K as the GGUF lays it out, llama.cpp's `block_q3_K`.
typedef struct {
  uint8_t hmask[QK_K / 8];  // quants, high bit
  uint8_t qs[QK_K / 4];     // quants, low 2 bits
  uint8_t scales[12];       // scales, quantized with 6 bits
  half d;                   // super-block scale
} block_q3_K;

// The Q3_K row body is candle's `kernel_mul_mv_q3_K_f32_impl` exactly, its masks and its accumulation order.

typedef struct {
  int ix;
  int il;
  int l0;
  int q_offset;
  int y_offset;
  int shift;
  float v1, v2;
  ushort4 hm;
  int4 qm;
  uint16_t s_shift1, s_shift2;
} q3k_thread_ctx;

static OSCAR_INLINE q3k_thread_ctx q3k_make_ctx(uint tiisg) {
  const int tid = (int)tiisg / 4;
  const int ix = (int)tiisg % 4;
  const int ip = tid / 4;
  const int il = 2 * ((tid % 4) / 2);
  const int ir = tid % 2;
  const int l0 = 8 * ir;
  const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},
                         {0x0004, 0x0400, 0x0008, 0x0800},
                         {0x0010, 0x1000, 0x0020, 0x2000},
                         {0x0040, 0x4000, 0x0080, 0x8000}};
  const int4 qmt[2] = {{0x0003, 0x0300, 0x000c, 0x0c00},
                       {0x0030, 0x3000, 0x00c0, 0xc000}};
  q3k_thread_ctx c;
  c.ix = ix;
  c.il = il;
  c.l0 = l0;
  c.q_offset = 32 * ip + l0;
  c.y_offset = 128 * ip + 32 * il + l0;
  c.shift = 2 * il;
  c.v1 = il == 0 ? 4.f : 64.f;
  c.v2 = 4.f * c.v1;
  c.hm = mm[2 * ip + il / 2];
  c.qm = qmt[il / 2];
  c.s_shift1 = (uint16_t)(4 * ip);
  c.s_shift2 = (uint16_t)(c.s_shift1 + il);
  return c;
}

// One super-block of one row: the impl's two half-passes into (sumf1, sumf2), in its exact order.
static OSCAR_INLINE void q3k_row_block_acc(
    thread const float yl[32],
    device const uint16_t *q,
    device const uint16_t *h,
    device const uint16_t *a,
    device const half *dh,
    thread const q3k_thread_ctx &c,
    thread float &sumf1,
    thread float &sumf2) {
  const float d_all = (float)dh[0];

  uint32_t scales32, aux32;
  thread uint16_t *scales16 = (thread uint16_t *)&scales32;
  thread const int8_t *scales = (thread const int8_t *)&scales32;
  scales16[0] = a[4];
  scales16[1] = a[5];
  aux32 = ((scales32 >> c.s_shift2) << 4) & 0x30303030;
  scales16[0] = a[c.il + 0];
  scales16[1] = a[c.il + 1];
  scales32 = ((scales32 >> c.s_shift1) & 0x0f0f0f0f) | aux32;

  float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
  for (int l = 0; l < 8; l += 2) {
    const int32_t qs = q[l / 2];
    s1 += yl[l + 0] * (qs & c.qm[0]);
    s2 += yl[l + 1] * (qs & c.qm[1]);
    s3 += ((h[l / 2] & c.hm[0]) ? 0.f : yl[l + 0]) +
          ((h[l / 2] & c.hm[1]) ? 0.f : yl[l + 1]);
    s4 += yl[l + 16] * (qs & c.qm[2]);
    s5 += yl[l + 17] * (qs & c.qm[3]);
    s6 += ((h[l / 2] & c.hm[2]) ? 0.f : yl[l + 16]) +
          ((h[l / 2] & c.hm[3]) ? 0.f : yl[l + 17]);
  }
  float d1 = d_all * (s1 + 1.f / 256.f * s2 - s3 * c.v1);
  float d2 = d_all * (s4 + 1.f / 256.f * s5 - s6 * c.v2);
  sumf1 += d1 * (scales[0] - 32);
  sumf2 += d2 * (scales[2] - 32);

  s1 = s2 = s3 = s4 = s5 = s6 = 0;
  for (int l = 0; l < 8; l += 2) {
    const int32_t qs = q[l / 2 + 8];
    s1 += yl[l + 8] * (qs & c.qm[0]);
    s2 += yl[l + 9] * (qs & c.qm[1]);
    s3 += ((h[l / 2 + 8] & c.hm[0]) ? 0.f : yl[l + 8]) +
          ((h[l / 2 + 8] & c.hm[1]) ? 0.f : yl[l + 9]);
    s4 += yl[l + 24] * (qs & c.qm[2]);
    s5 += yl[l + 25] * (qs & c.qm[3]);
    s6 += ((h[l / 2 + 8] & c.hm[2]) ? 0.f : yl[l + 24]) +
          ((h[l / 2 + 8] & c.hm[3]) ? 0.f : yl[l + 25]);
  }
  d1 = d_all * (s1 + 1.f / 256.f * s2 - s3 * c.v1);
  d2 = d_all * (s4 + 1.f / 256.f * s5 - s6 * c.v2);
  sumf1 += d1 * (scales[1] - 32);
  sumf2 += d2 * (scales[3] - 32);
}

// The gated activation of a (gate, up) pair; its two parameters come from the GGUF, so they arrive as arguments.

struct ActSilu {
  static OSCAR_INLINE float apply(float g, float u, float, float) {
    return (g / (1.f + exp(-g))) * u;
  }
};

// SiTU-GLU (Kimi K3) in the oracle's factor order (`oscar_models::situ`); capping up too bounds the output by b1·b2.
struct ActSituGlu {
  static OSCAR_INLINE float apply(float g, float u, float beta_gate, float beta_up) {
    const float gated = (beta_gate * tanh(g / beta_gate)) / (1.f + exp(-g));
    return gated * (beta_up * tanh(u / beta_up));
  }
};

// Clamped SwiGLU (`glm5next`), its limit from the GGUF (`swiglu_clamp_*`).
struct ActSiluClamp {
  static OSCAR_INLINE float apply(float g, float u, float limit, float) {
    const float gc = min(g, limit);
    const float uc = clamp(u, -limit, limit);
    return (gc / (1.f + exp(-gc))) * uc;
  }
};

// One per expert id: its quantised bytes, or null where not to be read (not held, or outside this prefill wave).
struct ExpertRow {
  device const char *base;
};

// A (token, slot)'s gate and up bases; a null in either row counts one miss per slot, a malformed id none.
struct RoutedRowPair {
  device const ExpertRow *gate;
  device const ExpertRow *up;
  device const uint32_t *topk_ids;
  device atomic_uint *miss_count;
  int64_t num_experts;
  int topk;
  OSCAR_INLINE bool resolve(int token, int slot, uint tgx, uint tiisg,
                            thread device const char *&g,
                            thread device const char *&u) const thread {
    const uint32_t e = topk_ids[token * topk + slot];
    if ((int64_t)e >= num_experts) {
      return false;
    }
    g = gate[e].base;
    u = up[e].base;
    if (g == nullptr || u == nullptr) {
      if (tgx == 0 && tiisg == 0) {
        atomic_fetch_add_explicit(&miss_count[0], 1u, memory_order_relaxed);
      }
      return false;
    }
    return true;
  }
};

// One token's routed ids through a row: `slot()` is false where the slot contributes nothing.
struct RoutedRow {
  device const ExpertRow *row;
  device const uint32_t *ids_row;
  int64_t num_experts;
  OSCAR_INLINE bool slot(int k, thread device const char *&base) const thread {
    const uint32_t e = ids_row[k];
    if ((int64_t)e >= num_experts) {
      return false;
    }
    base = row[e].base;
    return base != nullptr;
  }
};

// ggml's `block_q5_0`, for down rows not a multiple of 256; the value is (q - 16)·d, qh bit j element j's fifth bit.
// Elements 0..15 are qs's low nibbles, 16..31 its high ones.
typedef struct {
  half d;
  uint8_t qh[4];
  uint8_t qs[16];
} oscar_block_q5_0;

// The routing table by expert (llama.cpp's `kernel_mul_mm_id_map0`): ne21 rows of ne20 i32 ids, nb21 bytes apart, into
// htpe[ntg] u32 counts and hids[ntg·ne21] i32 token·ne20 + slot, the first htpe[e] valid; ntg·ne20·2 threadgroup bytes.
template <short ne20>
kernel void mistralrs_moe_mat_mm_id_map0(
    device const char *src2 [[buffer(0)]],
    device char *htpe [[buffer(1)]],
    device char *hids [[buffer(2)]],
    device const uint32_t *experts [[buffer(3)]],
    constant int64_t &ne21 [[buffer(4)]],
    constant uint64_t &nb21 [[buffer(5)]],
    threadgroup char *shmem [[threadgroup(0)]],
    ushort tpitg [[thread_position_in_threadgroup]],
    ushort ntg [[threads_per_threadgroup]]) {
  const short ide = tpitg;
  const uint16_t mine = (uint16_t)experts[ide];

  uint32_t n_all = 0;
  device int32_t *ids_i32 = (device int32_t *)hids + ide * ne21;

  for (int i21 = 0; i21 < ne21; i21 += ntg) {
    if (i21 + tpitg < ne21) {
      device const int32_t *src2_i32 =
          (device const int32_t *)(src2 + (i21 + tpitg) * nb21);

      threadgroup uint16_t *sids =
          (threadgroup uint16_t *)shmem + tpitg * ne20;

#pragma unroll(ne20)
      for (short i20 = 0; i20 < ne20; i20++) {
        sids[i20] = (uint16_t)src2_i32[i20];
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short t = 0; t < ntg; t++) {
      if (i21 + t >= ne21) {
        break;
      }

      threadgroup const uint16_t *sids =
          (threadgroup const uint16_t *)shmem + t * ne20;

      short sel = 0;
#pragma unroll(ne20)
      for (short i20 = 0; i20 < ne20; i20++) {
        sel += (sids[i20] == mine) * (i20 + 1);
      }

      ids_i32[n_all] = (i21 + t) * ne20 + sel - 1;
      n_all += sel > 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  device uint32_t *tpe_u32 = (device uint32_t *)(htpe);
  tpe_u32[ide] = n_all;
}

typedef decltype(mistralrs_moe_mat_mm_id_map0<8>) mistralrs_moe_mat_mm_id_map0_t;

template [[host_name("mistralrs_moe_mat_mm_id_map0_ne20_8")]] kernel mistralrs_moe_mat_mm_id_map0_t mistralrs_moe_mat_mm_id_map0<8>;
template [[host_name("mistralrs_moe_mat_mm_id_map0_ne20_10")]] kernel mistralrs_moe_mat_mm_id_map0_t mistralrs_moe_mat_mm_id_map0<10>;
template [[host_name("mistralrs_moe_mat_mm_id_map0_ne20_16")]] kernel mistralrs_moe_mat_mm_id_map0_t mistralrs_moe_mat_mm_id_map0<16>;

// Grid (ceil(hidden / 32), num_tokens); a null row entry is skipped, not weighted by zero: its slice may hold NaN.
[[host_name("mistralrs_moe_topk_weighted_sum_f32")]]
kernel void moe_topk_weighted_sum_f32(
    device const float    *y           [[buffer(0)]],
    device const float    *w           [[buffer(1)]],
    device const ExpertRow *row        [[buffer(2)]],
    device const uint32_t *topk_ids    [[buffer(3)]],
    device float          *out         [[buffer(4)]],
    constant int          &topk        [[buffer(5)]],
    constant int          &hidden      [[buffer(6)]],
    constant int          &num_experts [[buffer(7)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]]) {
  const int t = (int)tgpig.y;
  const int h = (int)tgpig.x * 32 + (int)tiisg;
  if (h >= hidden) return;

  device const float    *y_t  = y + t * topk * hidden + h;
  device const float    *w_t  = w + t * topk;
  device const uint32_t *id_t = topk_ids + t * topk;

  float acc = 0.0f;
  for (int k = 0; k < topk; k++) {
    const uint32_t e = id_t[k];
    if ((int)e >= num_experts || row[e].base == nullptr) continue;
    const float wk = w_t[k];
    if (wk != 0.0f) {
      acc = fma(y_t[k * hidden], wk, acc);
    }
  }
  out[t * hidden + h] = acc;
}

// llama.cpp's `block_q8_0`: one F16 scale per 32 quants.
#define QK8_0 32

typedef struct {
  half d;
  int8_t qs[QK8_0];
} block_q8_0;

// The group's (max, lowest index on ties) on thread 0, index -1 where all were -inf or NaN; `wv`, `wi` per simdgroup.
static OSCAR_INLINE void tg_argmax(thread float &v, thread int &i, threadgroup float *wv,
                                   threadgroup int *wi, int num_warps, uint lane_id,
                                   uint warp_id) {
  for (uint stride = 16; stride > 0; stride >>= 1) {
    const float ov = simd_shuffle_xor(v, stride);
    const int oi = simd_shuffle_xor(i, stride);
    if ((ov > v) || (ov == v && oi >= 0 && (i < 0 || oi < i))) {
      v = ov;
      i = oi;
    }
  }
  if (num_warps == 1) {
    return;
  }
  if (lane_id == 0) {
    wv[warp_id] = v;
    wi[warp_id] = i;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  v = (warp_id == 0 && lane_id < (uint)num_warps) ? wv[lane_id] : -INFINITY;
  i = (warp_id == 0 && lane_id < (uint)num_warps) ? wi[lane_id] : -1;
  if (warp_id == 0) {
    for (uint stride = 16; stride > 0; stride >>= 1) {
      const float ov = simd_shuffle_xor(v, stride);
      const int oi = simd_shuffle_xor(i, stride);
      if ((ov > v) || (ov == v && oi >= 0 && (i < 0 || oi < i))) {
        v = ov;
        i = oi;
      }
    }
  }
}

// The router's static threadgroup arrays, sized for every model's experts (K3: 896).
#define MAX_NCOLS 1024
#define MAX_K     16
#define MAX_WARPS 32

// One group per row. Mode 0, softmax: the winners by logit (top-k then softmax equals softmax, top-k, renormalise).
// Mode 1, biased sigmoid (DeepSeek's): the winners by sigmoid(logit) + bias[i], the bias steering selection only.
[[host_name("mistralrs_topk_softmax_f32")]]
kernel void mistralrs_topk_softmax_f32(
    device const float    *input        [[buffer(0)]],
    device       float    *weights_out  [[buffer(1)]],
    device       uint32_t *indices_out  [[buffer(2)]],
    constant int          &ncols        [[buffer(3)]],
    constant int          &k            [[buffer(4)]],
    constant int          &norm         [[buffer(5)]],
    device const float    *bias         [[buffer(6)]],
    constant int          &mode         [[buffer(7)]],
    constant float        &scale        [[buffer(8)]],
    constant float        &sum_eps      [[buffer(9)]],
    uint  row            [[threadgroup_position_in_grid]],
    uint  tid            [[thread_position_in_threadgroup]],
    uint  tg_size        [[threads_per_threadgroup]],
    uint  lane_id        [[thread_index_in_simdgroup]],
    uint  warp_id        [[simdgroup_index_in_threadgroup]]) {
  threadgroup float  s_data[MAX_NCOLS];
  threadgroup float  s_score[MAX_NCOLS];
  threadgroup uchar  s_used[MAX_NCOLS];
  threadgroup float  s_topk_vals[MAX_K];
  threadgroup int    s_topk_idx[MAX_K];
  threadgroup float  warp_maxes[MAX_WARPS];
  threadgroup int    warp_indices[MAX_WARPS];

  const int row_i = (int)row;
  device const float    *row_in      = input       + row_i * ncols;
  device       float    *row_weights = weights_out + row_i * k;
  device       uint32_t *row_indices = indices_out + row_i * k;

  // float4 loads where ncols allows: Apple silicon issues one transaction per load whatever its width.
  const int n4 = ncols >> 2;
  const bool ncols_aligned = (ncols & 3) == 0;
  if (ncols_aligned) {
    device const float4 *row_in4 = (device const float4 *)row_in;
    threadgroup float4  *s_data4 = (threadgroup float4 *)s_data;
    for (int j = (int)tid; j < n4; j += (int)tg_size) {
      s_data4[j] = row_in4[j];
    }
  } else {
    for (int i = (int)tid; i < ncols; i += (int)tg_size) {
      s_data[i] = row_in[i];
    }
  }
  for (int i = (int)tid; i < ncols; i += (int)tg_size) {
    s_used[i] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (mode == 1) {
    // The score beside the selection key: the key ranks, the score weighs.
    for (int i = (int)tid; i < ncols; i += (int)tg_size) {
      const float score = 1.0f / (1.0f + exp(-s_data[i]));
      s_score[i] = score;
      s_data[i] = score + bias[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const int num_warps = (int)((tg_size + 31u) / 32u);

  for (int ki = 0; ki < k; ki++) {
    // No isnan(): router logits come from healthy weights.
    float local_max = -INFINITY;
    int   local_idx = -1;
    for (int i = (int)tid; i < ncols; i += (int)tg_size) {
      float v = s_data[i];
      bool valid = !s_used[i];
      if (valid && v > local_max) {
        local_max = v;
        local_idx = i;
      }
    }

    tg_argmax(local_max, local_idx, warp_maxes, warp_indices, num_warps, lane_id, warp_id);

    if (tid == 0) {
      // The Rust binding enforces K ≤ ncols, so a winner always exists.
      s_topk_vals[ki] = (mode == 1) ? s_score[local_idx] : local_max;
      s_topk_idx[ki]  = local_idx;
      s_used[local_idx] = 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (warp_id == 0 && lane_id < (uint)MAX_K) {
    const uint i = lane_id;
    const bool active = (int)i < k;
    float w;
    if (mode == 1) {
      const float val = active ? s_topk_vals[i] : 0.0f;
      const float row_sum = simd_sum(val);
      w = (norm != 0) ? (val / (row_sum + sum_eps)) : val;
      w *= scale;
    } else {
      float val = active ? s_topk_vals[i] : -INFINITY;
      float row_max = simd_max(val);
      float e = active ? exp(val - row_max) : 0.0f;
      float row_sum = simd_sum(e);
      w = (norm != 0) ? (e * (1.0f / row_sum)) : e;
    }
    if (active) {
      row_weights[i] = w;
      row_indices[i] = (uint32_t)s_topk_idx[i];
    }
  }
}

// One dispatch a layer writes its ids, tap and pred blocks. CPU visibility inside a command buffer is best-effort, so
// no reader relies on write order: ids need both gen bytes and no sentinel, tap and pred (prefetch hints) one gen byte.
#define EP_PRED_MAX_B 16

[[host_name("oscar_ep_publish")]]
kernel void oscar_ep_publish(
    device const uint32_t *topk_ids     [[buffer(0)]],
    device const float    *logits       [[buffer(1)]],
    device const float    *xs           [[buffer(2)]],
    device       uint32_t *mailbox      [[buffer(3)]],
    constant     int64_t  &k            [[buffer(4)]],
    constant     int64_t  &ids_woff     [[buffer(5)]],
    constant     uint32_t &ids_gen      [[buffer(6)]],
    constant     int64_t  &num_experts  [[buffer(7)]],
    constant     int64_t  &look_b       [[buffer(8)]],
    constant     int64_t  &pred_woff    [[buffer(9)]],
    constant     uint32_t &pred_gen     [[buffer(10)]],
    constant     int64_t  &tap_words    [[buffer(11)]],
    constant     int64_t  &tap_woff     [[buffer(12)]],
    constant     uint32_t &tap_gen      [[buffer(13)]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint warp_id [[simdgroup_index_in_threadgroup]]) {
  // The ids block: the ids, then the host's gen word (gen in bytes 0 and 1, 0xFF above, as a refill leaves it).
  if ((int64_t)tid < k) {
    mailbox[ids_woff + tid] = topk_ids[tid];
  }
  if (tid == 0) {
    mailbox[ids_woff + k] = ids_gen;
  }

  // The tap block: the router-input hidden state, then its own gen word.
  for (int64_t i = tid; i < tap_words; i += tg_size) {
    mailbox[tap_woff + i] = as_type<uint>(xs[i]);
  }
  if (tid == 0 && tap_words > 0) {
    mailbox[tap_woff + tap_words] = tap_gen;
  }
  if (look_b <= 0) {
    return;
  }

  // The pred block: the next layer's top `look_b` ids, score bits and gen word; read from device memory, any width.
  threadgroup int   s_picked[EP_PRED_MAX_B];
  threadgroup float warp_maxes[32];
  threadgroup int   warp_indices[32];
  const int ncols = (int)num_experts;

  const int num_warps = (int)((tg_size + 31u) / 32u);
  for (int bi = 0; bi < (int)look_b; bi++) {
    float local_max = -INFINITY;
    int   local_idx = -1;
    for (int i = (int)tid; i < ncols; i += (int)tg_size) {
      bool picked = false;
      for (int p = 0; p < bi; p++) {
        picked = picked || (s_picked[p] == i);
      }
      if (!picked && logits[i] > local_max) {
        local_max = logits[i];
        local_idx = i;
      }
    }
    tg_argmax(local_max, local_idx, warp_maxes, warp_indices, num_warps, lane_id, warp_id);
    if (tid == 0) {
      // -1 (all remaining NaN) matches no candidate: a degenerate pass excludes nothing and writes no word.
      s_picked[bi] = local_idx;
      if (local_idx >= 0) {
        mailbox[pred_woff + bi] = (uint32_t)local_idx;
        mailbox[pred_woff + look_b + bi] = as_type<uint>(local_max);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    mailbox[pred_woff + 2 * look_b] = pred_gen;
  }
}

// The drafter's rows, checked by epoch: the bank bumps a slot's epoch on every claim, before its first overwrite byte.

// Before the matvecs: null where a slot's epoch moved, so a stale slot reads as a counted miss, never wrong weights.
[[host_name("oscar_draft_row_validate")]]
kernel void oscar_draft_row_validate(
    device const uint32_t *snap_gate   [[buffer(0)]],
    device const uint32_t *snap_up     [[buffer(1)]],
    device const uint32_t *snap_down   [[buffer(2)]],
    device const uint32_t *elut_gate   [[buffer(3)]],
    device const uint32_t *elut_up     [[buffer(4)]],
    device const uint32_t *elut_down   [[buffer(5)]],
    device const uint32_t *epochs_gate [[buffer(6)]],
    device const uint32_t *epochs_up   [[buffer(7)]],
    device const uint32_t *epochs_down [[buffer(8)]],
    device const ExpertRow *slots_gate [[buffer(9)]],
    device const ExpertRow *slots_up   [[buffer(10)]],
    device const ExpertRow *slots_down [[buffer(11)]],
    device       ExpertRow *out_gate   [[buffer(12)]],
    device       ExpertRow *out_up     [[buffer(13)]],
    device       ExpertRow *out_down   [[buffer(14)]],
    constant     int64_t  &num_experts [[buffer(15)]],
    constant     int64_t  &cap_gate    [[buffer(16)]],
    constant     int64_t  &cap_up      [[buffer(17)]],
    constant     int64_t  &cap_down    [[buffer(18)]],
    uint tid [[thread_position_in_grid]]) {
  if ((int64_t)tid >= num_experts) return;
  // All three projections or none, as every CPU row writer: only gate and up count misses, so a null down is silent.
  const uint32_t g = snap_gate[tid];
  const uint32_t u = snap_up[tid];
  const uint32_t d = snap_down[tid];
  const bool ok = ((int64_t)g < cap_gate && epochs_gate[g] == elut_gate[tid])
               && ((int64_t)u < cap_up   && epochs_up[u]   == elut_up[tid])
               && ((int64_t)d < cap_down && epochs_down[d] == elut_down[tid]);
  out_gate[tid].base = ok ? slots_gate[g].base : nullptr;
  out_up[tid].base   = ok ? slots_up[u].base : nullptr;
  out_down[tid].base = ok ? slots_down[d].base : nullptr;
}

// Between the row check and the matvecs, one thread (k ≤ 16): each cold selection under θ (2: all) becomes the best
// usable expert by logit, the weights a softmax over the final set as in `dispatch::approx`; the inputs stay intact.
[[host_name("oscar_draft_theta_substitute")]]
kernel void oscar_draft_theta_substitute(
    device const ExpertRow *scr_gate   [[buffer(0)]],
    device const float    *logits      [[buffer(1)]],
    device const uint32_t *in_ids      [[buffer(2)]],
    device const float    *in_w        [[buffer(3)]],
    device       uint32_t *out_ids     [[buffer(4)]],
    device       float    *out_w       [[buffer(5)]],
    constant     int64_t  &k           [[buffer(6)]],
    constant     int64_t  &num_experts [[buffer(7)]],
    constant     float    &theta       [[buffer(8)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  const int K = int(k);
  float tot = 0.f;
  for (int i = 0; i < K; i++) {
    out_ids[i] = in_ids[i];
    out_w[i] = in_w[i];
    tot += in_w[i];
  }
  if (K > 16 || !(tot > 0.f)) return;
  // Heaviest first: the heaviest replaced selection takes the best candidate.
  int order[16];
  for (int i = 0; i < K; i++) order[i] = i;
  for (int a = 0; a < K; a++) {
    int b = a;
    for (int c = a + 1; c < K; c++)
      if (in_w[order[c]] > in_w[order[b]]) b = c;
    int t = order[a]; order[a] = order[b]; order[b] = t;
  }
  uint32_t chosen[16];
  int nch = 0, nrepl = 0;
  for (int oi = 0; oi < K; oi++) {
    const int pos = order[oi];
    const uint32_t e = in_ids[pos];
    if ((int64_t)e >= num_experts) continue;
    if (in_w[pos] / tot >= theta) continue;
    if (scr_gate[e].base != nullptr) continue;  // non-null: all three resident and valid
    int best = -1;
    float bv = -INFINITY;
    for (int c = 0; c < int(num_experts); c++) {
      if (scr_gate[c].base == nullptr) continue;
      bool taken = false;
      for (int j = 0; j < K && !taken; j++)
        if (in_ids[j] == (uint32_t)c) taken = true;
      for (int j = 0; j < nch && !taken; j++)
        if (chosen[j] == (uint32_t)c) taken = true;
      if (taken) continue;
      if (logits[c] > bv) { bv = logits[c]; best = c; }
    }
    if (best < 0) continue;  // no usable candidate: kept, for the cold skip
    out_ids[pos] = (uint32_t)best;
    chosen[nch++] = (uint32_t)best;
    nrepl++;
  }
  if (nrepl == 0) return;
  float m = -INFINITY;
  for (int i = 0; i < K; i++) {
    const float v = logits[out_ids[i]];
    if (v > m) m = v;
  }
  float s = 0.f;
  for (int i = 0; i < K; i++) s += exp(logits[out_ids[i]] - m);
  for (int i = 0; i < K; i++) out_w[i] = exp(logits[out_ids[i]] - m) / s;
}

// After the down matvec: `dirty` counts entries whose epoch moved mid-read; the host discards the token at its sync.
[[host_name("oscar_draft_epoch_recheck")]]
kernel void oscar_draft_epoch_recheck(
    device const uint32_t *topk_ids    [[buffer(0)]],
    device const uint32_t *snap_gate   [[buffer(1)]],
    device const uint32_t *snap_up     [[buffer(2)]],
    device const uint32_t *snap_down   [[buffer(3)]],
    device const ExpertRow *row_gate   [[buffer(4)]],
    device const ExpertRow *row_up     [[buffer(5)]],
    device const ExpertRow *row_down   [[buffer(6)]],
    device const uint32_t *elut_gate   [[buffer(7)]],
    device const uint32_t *elut_up     [[buffer(8)]],
    device const uint32_t *elut_down   [[buffer(9)]],
    device const uint32_t *epochs_gate [[buffer(10)]],
    device const uint32_t *epochs_up   [[buffer(11)]],
    device const uint32_t *epochs_down [[buffer(12)]],
    device       atomic_uint *dirty    [[buffer(13)]],
    constant     int64_t  &count       [[buffer(14)]],
    constant     int64_t  &num_experts [[buffer(15)]],
    constant     int64_t  &cap_gate    [[buffer(16)]],
    constant     int64_t  &cap_up      [[buffer(17)]],
    constant     int64_t  &cap_down    [[buffer(18)]],
    device       uint32_t *wm_gate     [[buffer(19)]],
    device       uint32_t *wm_up       [[buffer(20)]],
    device       uint32_t *wm_down     [[buffer(21)]],
    constant     int64_t  &wm_val      [[buffer(22)]],
    uint tid [[thread_position_in_grid]]) {
  if ((int64_t)tid >= count) return;
  // Draft lane only (the engine passes -1): `wm_val` = layer + 1 retires the layer's draft-read fence.
  if (tid == 0 && wm_val >= 0) {
    wm_gate[0] = (uint32_t)wm_val;
    wm_up[0]   = (uint32_t)wm_val;
    wm_down[0] = (uint32_t)wm_val;
  }
  const uint32_t e = topk_ids[tid];
  if ((int64_t)e >= num_experts) return;
  // A nulled entry was never read: only a resolved entry can be torn.
  bool torn = false;
  const uint32_t g = snap_gate[e];
  if (row_gate[e].base != nullptr && (int64_t)g < cap_gate && epochs_gate[g] != elut_gate[e]) torn = true;
  const uint32_t u = snap_up[e];
  if (row_up[e].base != nullptr && (int64_t)u < cap_up && epochs_up[u] != elut_up[e]) torn = true;
  const uint32_t d = snap_down[e];
  if (row_down[e].base != nullptr && (int64_t)d < cap_down && epochs_down[d] != elut_down[e]) torn = true;
  if (torn) {
    atomic_fetch_add_explicit(&dirty[0], 1u, memory_order_relaxed);
  }
}

// One simdgroup per router row, where a GEMM tile fills too few groups; its sum order is not the matmul's.
kernel void oscar_router_gemv_f32(
    device const float* w [[buffer(0)]],   // (n, k) row-major
    device const float* x [[buffer(1)]],   // (k)
    device float* y       [[buffer(2)]],   // (n)
    constant int64_t& n   [[buffer(3)]],
    constant int64_t& k   [[buffer(4)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]) {
  const int64_t row = (int64_t)tgpig * 4 + sgitg;  // 4 simdgroups / TG
  if (row >= n) {
    return;
  }
  device const float* wr = w + row * k;
  float acc = 0.f;
  // Lanes take consecutive float4s, so each pass's loads coalesce.
  int64_t i = (int64_t)tiisg * 4;
  for (; i + 4 <= k; i += 128) {
    float4 wv = *(device const float4*)(wr + i);
    float4 xv = *(device const float4*)(x + i);
    acc += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
  }
  float total = simd_sum(acc);
  if (tiisg == 0) {
    y[row] = total;
  }
}

// The i-quants (the Kimi K3 experts), their inner sums candle's `kernel_mul_mv_iq*_f32_impl`; `finish` applies candle's
// post-reduction scale before the reduction, a power of two, so the value is the same (a subnormal apart).

#define QK_IQ_SUB 32  // weights per i-quant sub-block (QK_K / 8)

typedef struct {
  half d;
  uint8_t qs[QK_K / 8];
  uint16_t qh[QK_K / 32];
} block_iq1_s;

typedef struct {
  half d;
  uint16_t qs[QK_K / 8];
} block_iq2_xxs;

typedef struct {
  half d;
  uint8_t qs[3 * QK_K / 8];
} block_iq3_xxs;

static_assert(sizeof(block_iq1_s) == 50, "block_iq1_s must be 50 bytes");
static_assert(sizeof(block_iq2_xxs) == 66, "block_iq2_xxs must be 66 bytes");
static_assert(sizeof(block_iq3_xxs) == 98, "block_iq3_xxs must be 98 bytes");

// The lane map and, where the format stages its codebook, the table it reads.
struct IqCtx {
  int ix, ib;
  threadgroup const uchar *tg;
};

// The staged activations and their sum, which the IQ1_S dot needs.
struct IqY {
  float yl[QK_IQ_SUB];
  float sumy;
};

template <bool SUMY>
static OSCAR_INLINE void iq_stage(thread IqY &t, device const float *y4) {
  t.sumy = 0.f;
  for (int i = 0; i < QK_IQ_SUB; ++i) {
    t.yl[i] = y4[i];
  }
  if (SUMY) {
    for (int i = 0; i < QK_IQ_SUB; ++i) t.sumy += t.yl[i];
  }
}

// `tables` holds a barrier, so it runs after the entry's last return, which one simdgroup a group keeps uniform.
#define IQ_POLICY(BLOCK, BYTES, SUMY, SCALE)                                         \
  typedef BLOCK block_t;                                                            \
  enum : int { BLOCKS_PER_PASS = 4, QK = QK_K, ACCS = 1, TG_BYTES = BYTES };       \
  typedef IqCtx Ctx;                                                                \
  typedef IqY Y;                                                                    \
  static OSCAR_INLINE Ctx ctx(uint tiisg) {                                         \
    Ctx c;                                                                          \
    c.ix = (int)tiisg / 8;                                                          \
    c.ib = (int)tiisg % 8;                                                          \
    return c;                                                                       \
  }                                                                                 \
  static OSCAR_INLINE int y_offset(thread const Ctx &c) {                           \
    return c.ix * QK_K + QK_IQ_SUB * c.ib;                                          \
  }                                                                                 \
  static OSCAR_INLINE void stage(thread Y &t, device const float *y4,               \
                                 thread const Ctx &) {                              \
    iq_stage<SUMY>(t, y4);                                                          \
  }                                                                                 \
  static OSCAR_INLINE float finish(thread const float *s, thread const Ctx &) {     \
    return s[0] * SCALE;                                                            \
  }

struct WIq1S {
  IQ_POLICY(block_iq1_s, 0, true, 1.f)

  // IQ1_S reads its codebook straight from constant memory: no table.
  static OSCAR_INLINE void tables(threadgroup uchar *, uint, thread Ctx &) {}

  static OSCAR_INLINE void acc(thread float *s, thread const Y &t,
                               device const block_t *w, int ib,
                               thread const Ctx &c) {
    device const block_t *b = w + ib;
    device const uint8_t *qs = b->qs + 4 * c.ib;
    device const uint16_t *qh = b->qh + c.ib;
    constant uint8_t *grid1 =
        (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
    constant uint8_t *grid2 =
        (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 5) & 0x700)));
    constant uint8_t *grid3 =
        (constant uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[0] << 2) & 0x700)));
    constant uint8_t *grid4 =
        (constant uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[0] >> 1) & 0x700)));
    float sum = 0.f;
    for (int j = 0; j < 4; ++j) {
      sum += t.yl[j + 0] * (grid1[j] & 0xf) + t.yl[j + 4] * (grid1[j] >> 4) +
             t.yl[j + 8] * (grid2[j] & 0xf) + t.yl[j + 12] * (grid2[j] >> 4) +
             t.yl[j + 16] * (grid3[j] & 0xf) + t.yl[j + 20] * (grid3[j] >> 4) +
             t.yl[j + 24] * (grid4[j] & 0xf) + t.yl[j + 28] * (grid4[j] >> 4);
    }
    const float delta = (qh[0] & 0x8000) ? -1.f - IQ1S_DELTA : -1.f + IQ1S_DELTA;
    s[0] += (float)b->d * (sum + t.sumy * delta) * (2 * ((qh[0] >> 12) & 7) + 1);
  }
};

struct WIq2XXS {
  // 256 grid entries (u64) + 128 sign bytes.
  IQ_POLICY(block_iq2_xxs, 256 * 8 + 128, false, 0.25f)

  static OSCAR_INLINE void tables(threadgroup uchar *tg, uint tiisg, thread Ctx &c) {
    threadgroup uint64_t *values = (threadgroup uint64_t *)tg;
    threadgroup uint8_t *signs = (threadgroup uint8_t *)(values + 256);
    for (uint i = tiisg; i < 256; i += 32) values[i] = iq2xxs_grid[i];
    for (uint i = tiisg; i < 128; i += 32) signs[i] = ksigns_iq2xs[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    c.tg = tg;
  }

  static OSCAR_INLINE void acc(thread float *s, thread const Y &t,
                               device const block_t *w, int ib,
                               thread const Ctx &c) {
    threadgroup const uint64_t *values = (threadgroup const uint64_t *)c.tg;
    threadgroup const uint8_t *signs_tbl =
        (threadgroup const uint8_t *)(values + 256);
    device const uint16_t *q2 = w[ib].qs + 4 * c.ib;
    device const uint8_t *aux8 = (device const uint8_t *)q2;
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float)w[ib].d * (0.5f + (aux32 >> 28));
    float sum = 0.f;
    for (int l = 0; l < 4; ++l) {
      threadgroup const uint8_t *grid =
          (threadgroup const uint8_t *)(values + aux8[l]);
      const uint8_t signs = signs_tbl[(aux32 >> 7 * l) & 127];
      for (int j = 0; j < 8; ++j) {
        sum += t.yl[8 * l + j] * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
      }
    }
    s[0] += d * sum;
  }
};

struct WIq3XXS {
  // 256 grid entries (u32) + 128 sign bytes.
  IQ_POLICY(block_iq3_xxs, 256 * 4 + 128, false, 0.5f)

  static OSCAR_INLINE void tables(threadgroup uchar *tg, uint tiisg, thread Ctx &c) {
    threadgroup uint32_t *values = (threadgroup uint32_t *)tg;
    threadgroup uint8_t *signs = (threadgroup uint8_t *)(values + 256);
    for (uint i = tiisg; i < 256; i += 32) values[i] = iq3xxs_grid[i];
    for (uint i = tiisg; i < 128; i += 32) signs[i] = ksigns_iq2xs[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    c.tg = tg;
  }

  static OSCAR_INLINE void acc(thread float *s, thread const Y &t,
                               device const block_t *w, int ib,
                               thread const Ctx &c) {
    threadgroup const uint32_t *values = (threadgroup const uint32_t *)c.tg;
    threadgroup const uint8_t *signs_tbl =
        (threadgroup const uint8_t *)(values + 256);
    device const uint8_t *q3 = w[ib].qs + 8 * c.ib;
    device const uint16_t *gas =
        (device const uint16_t *)(w[ib].qs + QK_K / 4) + 2 * c.ib;
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float)w[ib].d * (0.5f + (aux32 >> 28));
    float2 sum = {0.f, 0.f};
    for (int l = 0; l < 4; ++l) {
      threadgroup const uint8_t *grid1 =
          (threadgroup const uint8_t *)(values + q3[2 * l + 0]);
      threadgroup const uint8_t *grid2 =
          (threadgroup const uint8_t *)(values + q3[2 * l + 1]);
      const uint8_t signs = signs_tbl[(aux32 >> 7 * l) & 127];
      for (int j = 0; j < 4; ++j) {
        sum[0] += t.yl[8 * l + j + 0] * grid1[j] *
                  ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
        sum[1] += t.yl[8 * l + j + 4] * grid2[j] *
                  ((signs & kmask_iq2xs[j + 4]) ? -1.f : 1.f);
      }
    }
    s[0] += d * (sum[0] + sum[1]);
  }
};

// The rows a batch reads, dequantised as candle's CPU `to_float` does, so bit-identical; one group per id, clamped.

// Element `j` of a quantised block, as candle's CPU `to_float` computes it.
inline float embed_elem(device const block_q4_K *b, int j) {
  const int g = j >> 6;
  const int hi = (j >> 5) & 1;  // low nibbles first, then high
  const int is = 2 * g + hi;    // scale/min pair for this half-group
  uchar sc, mn;
  if (is < 4) {
    sc = b->scales[is] & 63;
    mn = b->scales[is + 4] & 63;
  } else {
    sc = (b->scales[is + 4] & 0xF) | ((b->scales[is - 4] >> 6) << 4);
    mn = (b->scales[is + 4] >> 4) | ((b->scales[is] >> 6) << 4);
  }
  const uchar q = b->qs[32 * g + (j & 31)];
  const float scale = (float)b->d * (float)sc;
  const float offset = (float)b->dmin * (float)mn;
  const float w = (float)(hi ? (q >> 4) : (q & 0xF));
  return scale * w - offset;
}

inline float embed_elem(device const block_q8_0 *b, int j) {
  return (float)b->qs[j] * (float)b->d;
}

template <typename B, int QK>
kernel void embed_gather(
    device const B *table [[buffer(0)]],
    device const uint *ids [[buffer(1)]],
    device float *dst [[buffer(2)]],
    constant int64_t &hidden [[buffer(3)]],
    constant int64_t &vocab [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tiitg [[thread_position_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]]) {
  const int hidden_i = (int)hidden;
  const uint row = min(ids[tgpig.x], (uint)(vocab - 1));
  device const B *xb = table + (uint64_t)row * (uint64_t)(hidden_i / QK);
  device float *out = dst + (uint64_t)tgpig.x * (uint64_t)hidden_i;

  for (int e = (int)tiitg.x; e < hidden_i; e += (int)tptg.x) {
    out[e] = embed_elem(xb + e / QK, e % QK);
  }
}

template [[host_name("oscar_embed_gather_q4_K_f32")]] kernel decltype(embed_gather<block_q4_K, QK_K>)
    embed_gather<block_q4_K, QK_K>;
template [[host_name("oscar_embed_gather_q8_0_f32")]] kernel decltype(embed_gather<block_q8_0, QK8_0>)
    embed_gather<block_q8_0, QK8_0>;

// mHC mix shaping (glm5next), as the vendor's DeepseekV4HyperConnection: a threadgroup of h² lanes a row, one `comb`
// entry each, the row and column sums walking the lanes in the vendor's serial order; h² ≤ 32 keeps a row inside
// one simdgroup (h ≤ 5).
[[host_name("mistralrs_mhc_mix_f32")]]
kernel void mistralrs_mhc_mix_f32(
    device const float *mix    [[buffer(0)]],
    device       float *dst    [[buffer(1)]],
    device const float *base   [[buffer(2)]],
    device const float *scales [[buffer(3)]],
    constant     int64_t &hc_mult [[buffer(4)]],
    constant     int64_t &iters [[buffer(5)]],
    constant     float   &eps  [[buffer(6)]],
    uint tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]) {
  const int h = (int)hc_mult;
  const int width = (2 + h) * h;
  device const float *row = mix + (int64_t)tgpig * width;
  device float *out = dst + (int64_t)tgpig * width;
  const int i = tiisg / h, j = tiisg % h;

  // The 2h gates, one a lane (two at h == 1).
  for (int g = tiisg; g < 2 * h; g += h * h) {
    if (g < h) {
      const float p = row[g] * scales[0] + base[g];
      out[g] = 1.0f / (1.0f + exp(-p)) + eps;
    } else {
      const float q = row[g] * scales[1] + base[g];
      out[g] = 2.0f / (1.0f + exp(-q));
    }
  }

  float c = row[2 * h + tiisg] * scales[2] + base[2 * h + tiisg];
  float mx = -INFINITY;
  for (int jj = 0; jj < h; ++jj) mx = max(mx, simd_shuffle(c, (ushort)(i * h + jj)));
  c = exp(c - mx);
  float denom = 0.0f;
  for (int jj = 0; jj < h; ++jj) denom += simd_shuffle(c, (ushort)(i * h + jj));
  c = c / denom + eps;
  // A column pass, then (iters - 1) row/col pairs: the vendor's schedule.
  for (int64_t it = 0; it < iters; ++it) {
    if (it > 0) {
      float r = 0.0f;
      for (int jj = 0; jj < h; ++jj) r += simd_shuffle(c, (ushort)(i * h + jj));
      r += eps;
      c /= r;
    }
    float s = 0.0f;
    for (int ii = 0; ii < h; ++ii) s += simd_shuffle(c, (ushort)(ii * h + j));
    s += eps;
    c /= s;
  }
  out[2 * h + tiisg] = c;
}

// One matvec loop: the format, rows per simdgroup, simdgroups and matrices per pass are policies or parameters.
// No matvec holds the RMSNorm: inside, it would re-read the row per threadgroup and the weight per simdgroup.

// A format policy: `ctx` maps the lane (`ix` its first block, `BLOCKS_PER_PASS` the stride), `stage` loads activations,
// `acc` adds a row's block into `ACCS` accumulators, and `finish` folds them before the simdgroup reduction.
#define NO_TABLES                                                            \
  enum : int { TG_BYTES = 0 };                                               \
  template <class C>                                                         \
  static OSCAR_INLINE void tables(threadgroup uchar *, uint, thread C &) {}

struct WQ4K {
  NO_TABLES
  typedef block_q4_K block_t;
  enum : int { BLOCKS_PER_PASS = 4, QK = QK_K, ACCS = 1 };

  struct Ctx {
    int ix, iq, ir;
  };
  static OSCAR_INLINE Ctx ctx(uint tiisg) {
    const int it = (int)tiisg % 8;
    Ctx c;
    c.ix = (int)tiisg / 8;
    c.iq = it / 4;
    c.ir = it % 4;
    return c;
  }
  static OSCAR_INLINE int y_offset(thread const Ctx &c) {
    return c.ix * QK_K + 64 * c.iq + 8 * c.ir;
  }

  struct Y {
    float yl[16];
    float yh[16];
    float4 sumy;
  };

  static OSCAR_INLINE void stage(thread Y &t, device const float *y4,
                                 thread const Ctx &) {
    t.sumy = {0.f, 0.f, 0.f, 0.f};
    for (int i = 0; i < 8; ++i) {
      t.yl[i + 0] = y4[i + 0];
      t.sumy[0] += t.yl[i + 0];
      t.yl[i + 8] = y4[i + 32];
      t.sumy[1] += t.yl[i + 8];
      t.yh[i + 0] = y4[i + 128];
      t.sumy[2] += t.yh[i + 0];
      t.yh[i + 8] = y4[i + 160];
      t.sumy[3] += t.yh[i + 8];
    }
  }

  static OSCAR_INLINE void acc(thread float *s, thread const Y &t,
                               device const block_t *w, int ib,
                               thread const Ctx &c) {
    device const uint16_t *sc = (device const uint16_t *)w[ib].scales + c.iq;
    device const uint16_t *q1 =
        (device const uint16_t *)w[ib].qs + 16 * c.iq + 4 * c.ir;
    s[0] += q4k_block_dot(t.yl, t.yh, t.sumy, q1, sc, &w[ib].d);
  }

  static OSCAR_INLINE float finish(thread const float *s, thread const Ctx &) { return s[0]; }
};

struct WQ6K {
  NO_TABLES
  typedef block_q6_K block_t;
  enum : int { BLOCKS_PER_PASS = 2, QK = QK_K, ACCS = 1 };

  struct Ctx {
    int ix, y_off, q_offset_l, q_offset_h, is;
  };
  static OSCAR_INLINE Ctx ctx(uint tiisg) {
    const int tid = (int)tiisg / 2;
    const int ip = tid / 8;
    const int il = tid % 8;
    const int l0 = 4 * il;
    Ctx c;
    c.ix = (int)tiisg % 2;
    c.y_off = c.ix * QK_K + 128 * ip + l0;
    c.q_offset_l = 64 * ip + l0;
    c.q_offset_h = 32 * ip + l0;
    c.is = 8 * ip + l0 / 16;
    return c;
  }
  static OSCAR_INLINE int y_offset(thread const Ctx &c) { return c.y_off; }

  // Q6_K reads its activations inside the dot (+0, +32, +64, +96), so its staging is the pass's pointer.
  struct Y {
    device const float *y;
  };

  static OSCAR_INLINE void stage(thread Y &t, device const float *y4,
                                 thread const Ctx &) {
    t.y = y4;
  }

  static OSCAR_INLINE void acc(thread float *s, thread const Y &t,
                               device const block_t *w, int ib,
                               thread const Ctx &c) {
    const uint8_t kmask1 = 0x03;
    const uint8_t kmask2 = 0x0C;
    const uint8_t kmask3 = 0x30;
    const uint8_t kmask4 = 0xC0;

    device const uint8_t *q1 = w[ib].ql + c.q_offset_l;
    device const uint8_t *q2 = q1 + 32;
    device const uint8_t *qh = w[ib].qh + c.q_offset_h;
    device const int8_t *sc = w[ib].scales + c.is;
    const float dall = w[ib].d;

    float4 sums = {0.f, 0.f, 0.f, 0.f};
    for (int l = 0; l < 4; ++l) {
      sums[0] += t.y[l + 0] *
                 ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
      sums[1] += t.y[l + 32] *
                 ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
      sums[2] += t.y[l + 64] *
                 ((int8_t)((q1[l] >> 4) | ((qh[l] & kmask3) << 0)) - 32);
      sums[3] += t.y[l + 96] *
                 ((int8_t)((q2[l] >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
    }
    s[0] += dall * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] +
                    sums[3] * sc[6]);
  }

  static OSCAR_INLINE float finish(thread const float *s, thread const Ctx &) { return s[0]; }
};

struct WQ3K {
  NO_TABLES
  typedef block_q3_K block_t;
  // Two accumulators a row, the format's two scale halves, folded with the lane's shift by `finish`.
  enum : int { BLOCKS_PER_PASS = 4, QK = QK_K, ACCS = 2 };

  typedef q3k_thread_ctx Ctx;
  static OSCAR_INLINE Ctx ctx(uint tiisg) { return q3k_make_ctx(tiisg); }
  static OSCAR_INLINE int y_offset(thread const Ctx &c) {
    return c.ix * QK_K + c.y_offset;
  }

  struct Y {
    float yl[32];
  };

  static OSCAR_INLINE void stage(thread Y &t, device const float *y1,
                                 thread const Ctx &) {
    for (int l = 0; l < 8; ++l) {
      t.yl[l + 0] = y1[l + 0];
      t.yl[l + 8] = y1[l + 16];
      t.yl[l + 16] = y1[l + 32];
      t.yl[l + 24] = y1[l + 48];
    }
  }

  static OSCAR_INLINE void acc(thread float *s, thread const Y &t,
                               device const block_t *w, int ib,
                               thread const Ctx &c) {
    q3k_row_block_acc(t.yl, (device const uint16_t *)(w[ib].qs + c.q_offset),
                      (device const uint16_t *)(w[ib].hmask + c.l0),
                      (device const uint16_t *)(w[ib].scales), &w[ib].d, c,
                      s[0], s[1]);
  }

  static OSCAR_INLINE float finish(thread const float *s, thread const Ctx &c) {
    return (s[0] + 0.25f * s[1]) / (1 << c.shift);
  }
};

struct WQ8_0 {
  NO_TABLES
  typedef block_q8_0 block_t;
  enum : int { BLOCKS_PER_PASS = 8, QK = QK8_0, ACCS = 1 };

  struct Ctx {
    int ix, il;
  };
  static OSCAR_INLINE Ctx ctx(uint tiisg) {
    Ctx c;
    c.ix = (int)tiisg / 4;
    c.il = (int)tiisg % 4;
    return c;
  }
  static OSCAR_INLINE int y_offset(thread const Ctx &c) {
    return c.ix * QK8_0 + 8 * c.il;
  }

  struct Y {
    float yl[8];
  };

  static OSCAR_INLINE void stage(thread Y &t, device const float *y4,
                                 thread const Ctx &) {
    for (int i = 0; i < 8; ++i) {
      t.yl[i] = y4[i];
    }
  }

  static OSCAR_INLINE void acc(thread float *s, thread const Y &t,
                               device const block_t *w, int ib,
                               thread const Ctx &c) {
    device const int8_t *qs = w[ib].qs + 8 * c.il;
    float sumq = 0.f;
    for (int iq = 0; iq < 8; ++iq) {
      sumq += qs[iq] * t.yl[iq];
    }
    s[0] += sumq * (float)w[ib].d;
  }

  static OSCAR_INLINE float finish(thread const float *s, thread const Ctx &) { return s[0]; }
};

struct WQ5_0 {
  NO_TABLES
  typedef oscar_block_q5_0 block_t;
  // A lane owns one nibble half (16 values) of every sixteenth block, so a pass covers 16 blocks.
  enum : int { BLOCKS_PER_PASS = 16, QK = 32, ACCS = 1 };

  struct Ctx {
    int ix, h;
  };
  static OSCAR_INLINE Ctx ctx(uint tiisg) {
    Ctx c;
    c.ix = (int)(tiisg >> 1);
    c.h = (int)(tiisg & 1);
    return c;
  }
  static OSCAR_INLINE int y_offset(thread const Ctx &c) {
    return c.ix * 32 + c.h * 16;
  }

  // Read in place inside the dot.
  struct Y {
    device const float *y;
  };

  static OSCAR_INLINE void stage(thread Y &t, device const float *y4,
                                 thread const Ctx &) {
    t.y = y4;
  }

  static OSCAR_INLINE void acc(thread float *s, thread const Y &t,
                               device const block_t *w, int ib,
                               thread const Ctx &c) {
    device const block_t *blk = w + ib;
    const float d = (float)blk->d;
    const uint32_t qh = ((uint32_t)blk->qh[0]) | ((uint32_t)blk->qh[1] << 8) |
                        ((uint32_t)blk->qh[2] << 16) |
                        ((uint32_t)blk->qh[3] << 24);
    float sum = 0.f;
    for (int i = 0; i < 16; ++i) {
      const int q = c.h ? (blk->qs[i] >> 4) : (blk->qs[i] & 0xF);
      const int e = c.h * 16 + i;
      const int x = q | (int)(((qh >> e) & 1u) << 4);
      sum += (float)(x - 16) * t.y[i];
    }
    s[0] += d * sum;
  }

  static OSCAR_INLINE float finish(thread const float *s, thread const Ctx &) { return s[0]; }
};

// One k-loop over `NMAT` matrices sharing a staging pass (`w1` read only when `NMAT > 1`), rows `nb` blocks apart.
template <class W, int ROWS, int NMAT>
static OSCAR_INLINE void qmv_pass(device const typename W::block_t *w0,
                                  device const typename W::block_t *w1,
                                  device const float *y, int nb,
                                  thread const typename W::Ctx &c,
                                  thread float *s0, thread float *s1) {
  device const float *y4 = y + W::y_offset(c);
  typename W::Y t;
  for (int ib = c.ix; ib < nb; ib += W::BLOCKS_PER_PASS) {
    W::stage(t, y4, c);
    for (int row = 0; row < ROWS; ++row) {
      W::acc(s0 + row * W::ACCS, t, w0 + row * nb, ib, c);
    }
    if (NMAT > 1) {
      for (int row = 0; row < ROWS; ++row) {
        W::acc(s1 + row * W::ACCS, t, w1 + row * nb, ib, c);
      }
    }
    y4 += W::BLOCKS_PER_PASS * W::QK;
  }
}

// The host runs `SG`×32 threads a group and ceil(N / (`SG`×`ROWS`)) groups along x.
template <int ROWS, int SG>
static OSCAR_INLINE int dense_first_row(uint tgx, uint sgitg) {
  return ((int)tgx * SG + (int)sgitg) * ROWS;
}

// Decode only: dst[t, n] = w[n, :]·src1[t, :], src0 [N, K] in format blocks, src1 and dst F32, a token per grid z;
// at `NMAT` 2 the shared expert's gate (src0), up (src0b) and SiLU from one staging pass, else src0b is unread.
template <class W, int ROWS, int SG, int NMAT>
kernel void qmatvec_f32(
    device const char *src0 [[buffer(0)]],
    device const char *src0b [[buffer(1)]],
    device const float *src1 [[buffer(2)]],
    device float *dst [[buffer(3)]],
    constant int64_t &K [[buffer(4)]],
    constant int64_t &N [[buffer(5)]],
    constant int64_t &num_tokens [[buffer(6)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]) {
  const int K_i = (int)K;
  const int N_i = (int)N;
  const int nb = K_i / W::QK;
  const int token = (int)tgpig.z;
  if (token >= (int)num_tokens) {
    return;
  }
  const int first_row = dense_first_row<ROWS, SG>(tgpig.x, sgitg);
  if (first_row >= N_i) {
    return;
  }

  device const float *y = src1 + token * K_i;
  device float *dst_cur = dst + token * N_i;
  device const typename W::block_t *gw =
      (device const typename W::block_t *)src0 + first_row * nb;
  device const typename W::block_t *uw =
      (device const typename W::block_t *)src0b + first_row * nb;
  const typename W::Ctx c = W::ctx(tiisg);

  float acc_gate[ROWS * W::ACCS] = {0.f};
  float acc_up[ROWS * W::ACCS] = {0.f};
  qmv_pass<W, ROWS, NMAT>(gw, uw, y, nb, c, acc_gate, acc_up);

  for (int row = 0; row < ROWS; ++row) {
    const float gate_sum = simd_sum(W::finish(acc_gate + row * W::ACCS, c));
    const float up_sum = NMAT > 1 ? simd_sum(W::finish(acc_up + row * W::ACCS, c)) : 0.f;
    if (tiisg == 0 && first_row + row < N_i) {
      if (NMAT == 1) {
        dst_cur[first_row + row] = gate_sum;
      } else {
        const float silu = gate_sum / (1.0f + exp(-gate_sum));
        dst_cur[first_row + row] = silu * up_sum;
      }
    }
  }
}

// Each format at its fastest geometry; the suffix is rows per simdgroup, then simdgroups.
typedef decltype(qmatvec_f32<WQ4K, 2, 2, 1>) qmatvec_t;
template [[host_name("mistralrs_qmatvec_q4k_r2s2_f32")]] kernel qmatvec_t qmatvec_f32<WQ4K, 2, 2, 1>;
template [[host_name("mistralrs_qmatvec_q6k_r1s2_f32")]] kernel qmatvec_t qmatvec_f32<WQ6K, 1, 2, 1>;
template [[host_name("mistralrs_qmatvec_gateup_silu_q4k_r2s2_f32")]] kernel qmatvec_t qmatvec_f32<WQ4K, 2, 2, 2>;
template [[host_name("mistralrs_qmatvec_gateup_silu_q8_0_r4s2_f32")]] kernel qmatvec_t qmatvec_f32<WQ8_0, 4, 2, 2>;

// `PAR`: a simdgroup per slot, simdgroup 0 summing the partials in slot order, bitwise the serial walk on Q4_K and Q5_0
// (the Q3_K and Q6_K dots contract differently). Threadgroup memory: topk × `ROWS` × 32 floats (0), topk flags (1).
template <class W, int ROWS, bool PAR>
static OSCAR_INLINE void down_walk(thread const RoutedRow &src, device const char *src1,
                                   device const float *topk_w_row, int token,
                                   int64_t topk, int nb, int64_t ne00, int first_row,
                                   threadgroup float *part, threadgroup int *live,
                                   threadgroup uchar *tg, uint tiisg, uint sgitg,
                                   thread float *sum) {
  typename W::Ctx c = W::ctx(tiisg);
  W::tables(tg, tiisg, c);
  if (PAR) {
    const int k_slot = (int)sgitg;
    device const char *ebase;
    const bool on = src.slot(k_slot, ebase);
    if (tiisg == 0) {
      live[k_slot] = on ? 1 : 0;
    }
    if (on) {
      device const typename W::block_t *w_base =
          (device const typename W::block_t *)ebase + first_row * nb;
      device const float *y =
          (device const float *)(src1 + (token * topk + k_slot) * (ne00 * 4));
      float acc[ROWS * W::ACCS] = {0.f};
      qmv_pass<W, ROWS, 1>(w_base, w_base, y, nb, c, acc, acc);
      for (int row = 0; row < ROWS; ++row) {
        part[(k_slot * ROWS + row) * 32 + tiisg] = W::finish(acc + row * W::ACCS, c);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg != 0) {
      return;
    }
    for (int k = 0; k < (int)topk; ++k) {
      if (!live[k]) {
        continue;
      }
      const float w = topk_w_row[k];
      for (int row = 0; row < ROWS; ++row) {
        sum[row] += w * part[(k * ROWS + row) * 32 + tiisg];
      }
    }
    return;
  }
  for (int k_slot = 0; k_slot < (int)topk; ++k_slot) {
    device const char *ebase;
    if (!src.slot(k_slot, ebase)) {
      continue;
    }
    const float w = topk_w_row[k_slot];
    device const typename W::block_t *w_base =
        (device const typename W::block_t *)ebase + first_row * nb;
    device const float *y =
        (device const float *)(src1 + (token * topk + k_slot) * (ne00 * 4));

    float acc[ROWS * W::ACCS] = {0.f};
    qmv_pass<W, ROWS, 1>(w_base, w_base, y, nb, c, acc, acc);
    for (int row = 0; row < ROWS; ++row) {
      sum[row] += w * W::finish(acc + row * W::ACCS, c);
    }
  }
}

// dst[t, :] = Σ_k topk_w[t, k]·(down[ids[t, k]] @ src1[t, k]): src1 [num_tokens, topk, K], dst [num_tokens, N], F32.
// Grid (N / rows per group, 1, tokens) of topk×32 threads in parallel, `SG`×32 serial.
template <class W, int ROWS, int SG, bool PAR>
kernel void moe_down_f32(
    device const ExpertRow *row [[buffer(0)]],
    device const char *src1 [[buffer(1)]],
    device const float *topk_w [[buffer(2)]],
    device float *dst [[buffer(3)]],
    device const uint32_t *topk_ids [[buffer(4)]],
    constant int64_t &topk [[buffer(5)]],
    constant int64_t &K [[buffer(6)]],
    constant int64_t &N [[buffer(7)]],
    constant int64_t &num_experts [[buffer(8)]],
    threadgroup float *part [[threadgroup(0)]],
    threadgroup int *live [[threadgroup(1)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]) {
  threadgroup uchar tg[W::TG_BYTES > 0 ? W::TG_BYTES : 1];
  const int token = (int)tgpig.z;
  // Parallel, the threadgroup shares one row tile (its simdgroups are slots); serial, a tile per simdgroup.
  const int first_row = PAR ? (int)tgpig.x * ROWS : dense_first_row<ROWS, SG>(tgpig.x, sgitg);
  if (first_row >= (int)N) {
    return;
  }
  const int nb = (int)K / W::QK;

  const RoutedRow src = {row, topk_ids + token * topk, num_experts};
  float sum[ROWS] = {0.f};
  down_walk<W, ROWS, PAR>(src, src1, topk_w + token * topk, token, topk, nb,
                                     K, first_row, part, live, tg, tiisg, sgitg, sum);
  if (PAR && sgitg != 0) {
    return;
  }
  for (int row = 0; row < ROWS; ++row) {
    const float total = simd_sum(sum[row]);
    if (tiisg == 0 && first_row + row < (int)N) {
      dst[token * N + first_row + row] = total;
    }
  }
}

// Each format at its fastest geometry at the models' shapes.
typedef decltype(moe_down_f32<WQ4K, 2, 1, true>) moe_down_t;
template [[host_name("mistralrs_moe_fused_down_weighted_sum_q4k_f32")]] kernel moe_down_t moe_down_f32<WQ4K, 2, 1, true>;
template [[host_name("mistralrs_moe_fused_down_weighted_sum_q6k_f32")]] kernel moe_down_t moe_down_f32<WQ6K, 1, 2, false>;
template [[host_name("mistralrs_moe_fused_down_weighted_sum_q3k_f32")]] kernel moe_down_t moe_down_f32<WQ3K, 4, 1, false>;
template [[host_name("mistralrs_moe_fused_down_weighted_sum_q5_0_f32")]] kernel moe_down_t moe_down_f32<WQ5_0, 4, 1, true>;
template [[host_name("mistralrs_moe_fused_down_weighted_sum_iq1s_f32")]] kernel moe_down_t moe_down_f32<WIq1S, 4, 1, false>;
template [[host_name("mistralrs_moe_fused_down_weighted_sum_iq3xxs_f32")]] kernel moe_down_t moe_down_f32<WIq3XXS, 4, 1, false>;

// dst F32 [num_tokens, topk, N], grid (ceil(N / rows per group), 1, tokens × topk); `act_p0`/`act_p1` per `Act`.
template <class W, int ROWS, int SG, class Act>
kernel void moe_gateup_f32(
    device const ExpertRow *gate_row [[buffer(0)]],
    device const ExpertRow *up_row [[buffer(1)]],
    device const float *src1 [[buffer(2)]],
    device float *dst [[buffer(3)]],
    device const uint32_t *topk_ids [[buffer(4)]],
    constant int64_t &K [[buffer(5)]],
    constant int64_t &N [[buffer(6)]],
    constant int64_t &topk [[buffer(7)]],
    constant int64_t &num_tokens [[buffer(8)]],
    constant int64_t &num_experts [[buffer(9)]],
    device atomic_uint *miss_count [[buffer(10)]],
    constant float &act_p0 [[buffer(11)]],
    constant float &act_p1 [[buffer(12)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]) {
  threadgroup uchar tg[W::TG_BYTES > 0 ? W::TG_BYTES : 1];
  const int K_i = (int)K;
  const int N_i = (int)N;
  const int topk_i = (int)topk;
  const int token = (int)tgpig.z / topk_i;
  const int slot = (int)tgpig.z % topk_i;
  if (token >= (int)num_tokens) {
    return;
  }
  const int first_row = dense_first_row<ROWS, SG>(tgpig.x, sgitg);
  if (first_row >= N_i) {
    return;
  }
  device float *dst_cur = dst + (token * topk_i + slot) * N_i;
  const RoutedRowPair src = {gate_row, up_row, topk_ids, miss_count, num_experts, topk_i};
  device const char *gbase;
  device const char *ubase;
  if (!src.resolve(token, slot, tgpig.x, tiisg, gbase, ubase)) {
    // Unresolved: its rows written as zero, so the host need not zero-fill.
    if (tiisg == 0) {
      for (int row = 0; row < ROWS; ++row) {
        if (first_row + row < N_i) {
          dst_cur[first_row + row] = 0.f;
        }
      }
    }
    return;
  }
  const int nb = K_i / W::QK;
  device const typename W::block_t *gw =
      (device const typename W::block_t *)gbase + first_row * nb;
  device const typename W::block_t *uw =
      (device const typename W::block_t *)ubase + first_row * nb;
  device const float *y = src1 + token * K_i;
  typename W::Ctx c = W::ctx(tiisg);
  W::tables(tg, tiisg, c);

  float acc_gate[ROWS * W::ACCS] = {0.f};
  float acc_up[ROWS * W::ACCS] = {0.f};
  qmv_pass<W, ROWS, 2>(gw, uw, y, nb, c, acc_gate, acc_up);

  for (int row = 0; row < ROWS; ++row) {
    const float gate_sum = simd_sum(W::finish(acc_gate + row * W::ACCS, c));
    const float up_sum = simd_sum(W::finish(acc_up + row * W::ACCS, c));
    if (tiisg == 0 && first_row + row < N_i) {
      dst_cur[first_row + row] = Act::apply(gate_sum, up_sum, act_p0, act_p1);
    }
  }
}

typedef decltype(moe_gateup_f32<WQ4K, 4, 1, ActSilu>) moe_gateup_t;
template [[host_name("mistralrs_moe_fused_gateup_silu_q4k_f32")]] kernel moe_gateup_t moe_gateup_f32<WQ4K, 4, 1, ActSilu>;
template [[host_name("mistralrs_moe_fused_gateup_silu_q3k_f32")]] kernel moe_gateup_t moe_gateup_f32<WQ3K, 2, 1, ActSilu>;
template [[host_name("mistralrs_moe_fused_gateup_siluclamp_q3k_f32")]] kernel moe_gateup_t moe_gateup_f32<WQ3K, 2, 1, ActSiluClamp>;
template [[host_name("mistralrs_moe_fused_gateup_situ_iq1s_f32")]] kernel moe_gateup_t moe_gateup_f32<WIq1S, 4, 1, ActSituGlu>;
template [[host_name("mistralrs_moe_fused_gateup_situ_iq2xxs_f32")]] kernel moe_gateup_t moe_gateup_f32<WIq2XXS, 4, 1, ActSituGlu>;

