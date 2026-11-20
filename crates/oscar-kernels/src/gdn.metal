// The Gated DeltaNet kernels (the KDA recurrence too) and the glue of the Qwen family's gated attention.

#include <metal_stdlib>
using namespace metal;

// Metal's bf16 is `bfloat` (`bfloat16` is a reserved incomplete struct); the kernels say `bfloat16_t`.
typedef bfloat bfloat16_t;

#define GDN_L2NORM_TG_THREADS 32
#define GDN_L2NORM_MAX_PER_THREAD 16  // covers Dk/Dv up to 512

// A token loop's emits: after its `emit_at[e]`-th token (ascending; -1 unasked) the state registers go to `emit<e>`,
// laid out as `state`, the row a launch cut there writes.
#define GDN_MAX_EMITS 4
#define GDN_EMIT_PARAMS                                                                                    \
  device float *emit0 [[buffer(9)]], device float *emit1 [[buffer(10)]], device float *emit2 [[buffer(11)]], \
  device float *emit3 [[buffer(12)]], constant int *emit_at [[buffer(13)]]
static inline device float *emit_dst(int e, device float *e0, device float *e1, device float *e2, device float *e3) {
  return e == 0 ? e0 : e == 1 ? e1 : e == 2 ? e2 : e3;
}

// One token of the delta rule on a lane's V column: the state row `s` decayed (`PER_CHANNEL`, KDA: one decay a key
// channel, `g_t` a row of `BK`), corrected by `beta_t·(v_t − s·k_t)`, then read against `q_t`; `BV` lanes stage the
// row's k and q (and decays) through the group's buffers.
template <int BK, int BV, bool PER_CHANNEL>
static inline float delta_step(thread float *s, const device float *q_t, const device float *k_t,
                               const device float *g_t, float beta_t, float v_t,
                               threadgroup float *k_buf, threadgroup float *q_buf,
                               threadgroup float *a_buf, uint tid) {
  for (int j = (int)tid; j < BK; j += BV) {
    k_buf[j] = k_t[j];
    if (PER_CHANNEL)
      a_buf[j] = exp(g_t[j]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float decay = PER_CHANNEL ? 1.0f : exp(g_t[0]);
  float kv_mem = 0.0f;
  for (int j = 0; j < BK; j++) {
    s[j] *= PER_CHANNEL ? a_buf[j] : decay;
    kv_mem = fma(s[j], k_buf[j], kv_mem);
  }

  float delta = (v_t - kv_mem) * beta_t;

  for (int j = (int)tid; j < BK; j += BV) {
    q_buf[j] = q_t[j];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // The output reads the state after this token's write.
  float y_t = 0.0f;
  for (int j = 0; j < BK; j++) {
    s[j] = fma(k_buf[j], delta, s[j]);
    y_t = fma(s[j], q_buf[j], y_t);
  }
  return y_t;
}

// One row's RMS norm over `V` (a multiple of 32, ≤ 512), weighted, times its gate z or sigmoid(z); a lane holds V/32
// consecutive elements.
static inline void norm_gate_row(const device float *y, const device float *z, const device float *w, device float *out,
                                 int V, float eps, int sigmoid_gate, uint lane) {
  const int per_thread = V / GDN_L2NORM_TG_THREADS;
  float y_vals[GDN_L2NORM_MAX_PER_THREAD];
  float z_vals[GDN_L2NORM_MAX_PER_THREAD];
  float w_vals[GDN_L2NORM_MAX_PER_THREAD];
  float sum_sq = 0.0f;

  for (int i = 0; i < per_thread; i++) {
    const int j = (int)lane * per_thread + i;
    if (j < V) {
      y_vals[i] = y[j];
      z_vals[i] = z[j];
      w_vals[i] = w[j];
      sum_sq += y_vals[i] * y_vals[i];
    } else {
      y_vals[i] = 0.0f;
      z_vals[i] = 0.0f;
      w_vals[i] = 0.0f;
    }
  }

  sum_sq = simd_sum(sum_sq);

  const float V_f = (float)V;
  const float factor = rsqrt(sum_sq / V_f + eps);

  for (int i = 0; i < per_thread; i++) {
    const int j = (int)lane * per_thread + i;
    if (j < V) {
      const float normed = y_vals[i] * w_vals[i] * factor;
      const float sig_z = 1.0f / (1.0f + exp(-z_vals[i]));
      const float gate_z = sigmoid_gate != 0 ? sig_z : z_vals[i] * sig_z;
      out[j] = normed * gate_z;
    }
  }
}

// Grid (ceil(V / `BV`), B·H) of `BV` threads; `BK` is a template constant so a thread's state stays in registers.
// q, k `[BH,S,K]`; v, output `[BH,S,V]`; state `[BH,K,V]`; beta, g `[BH,S]` (`PER_CHANNEL`, KDA: g `[BH,S,K]`).
template <int BK, int BV, bool PER_CHANNEL>
[[kernel]] void delta_rule_kernel(
    const device float *q [[buffer(0)]], const device float *k [[buffer(1)]],
    const device float *v [[buffer(2)]], const device float *g [[buffer(3)]],
    const device float *beta [[buffer(4)]], device float *state [[buffer(5)]],
    device float *output [[buffer(6)]], constant int &seq_len [[buffer(7)]],
    constant int &v_dim [[buffer(8)]], GDN_EMIT_PARAMS,
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const int v_tile = tgpig.x;
  const int bh = tgpig.y;
  const int v_idx = v_tile * BV + (int)tid;

  if (v_idx >= v_dim)
    return;

  constexpr int G = PER_CHANNEL ? BK : 1;
  const device float *q_bh = q + bh * seq_len * BK;
  const device float *k_bh = k + bh * seq_len * BK;
  const device float *v_bh = v + bh * seq_len * v_dim;
  const device float *g_bh = g + bh * seq_len * G;
  const device float *beta_bh = beta + bh * seq_len;
  device float *state_bh = state + bh * BK * v_dim;
  device float *out_bh = output + bh * seq_len * v_dim;

  threadgroup float k_buf[BK];
  threadgroup float q_buf[BK];
  threadgroup float a_buf[G];

  float s[BK];
  for (int j = 0; j < BK; j++) {
    s[j] = state_bh[j * v_dim + v_idx];
  }

  int ne = 0;
  for (int t = 0; t < seq_len; t++) {
    const float beta_t = beta_bh[t];
    const float v_t = v_bh[t * v_dim + v_idx];
    out_bh[t * v_dim + v_idx] = delta_step<BK, BV, PER_CHANNEL>(
        s, q_bh + t * BK, k_bh + t * BK, g_bh + t * G, beta_t, v_t, k_buf, q_buf, a_buf, tid);
    if (ne < GDN_MAX_EMITS && t + 1 == emit_at[ne]) {
      device float *dst = emit_dst(ne++, emit0, emit1, emit2, emit3) + bh * BK * v_dim;
      for (int j = 0; j < BK; j++) {
        dst[j * v_dim + v_idx] = s[j];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (int j = 0; j < BK; j++) {
    state_bh[j * v_dim + v_idx] = s[j];
  }
}

// Decode only, a 128-thread group per head, simdgroup 0 then applying gdn_norm_gate_kernel in that kernel's order.
template <int BK, int BV>
[[kernel]] void gated_delta_rule_norm_gate_kernel(
    const device float *q [[buffer(0)]], const device float *k [[buffer(1)]],
    const device float *v [[buffer(2)]], const device float *g [[buffer(3)]],
    const device float *beta [[buffer(4)]], device float *state [[buffer(5)]],
    device float *output [[buffer(6)]], constant int &seq_len [[buffer(7)]],
    constant int &v_dim [[buffer(8)]],
    const device float *z [[buffer(9)]], const device float *w [[buffer(10)]],
    constant float &eps [[buffer(11)]], constant int &sigmoid_gate [[buffer(12)]],
    constant int &Hv [[buffer(13)]], constant int &z_stride_bt [[buffer(14)]],
    constant int &z_stride_g [[buffer(15)]], constant int &z_vpg [[buffer(16)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const int bh = tgpig.y;
  const int v_idx = (int)tid;
  threadgroup float k_buf[BK];
  threadgroup float q_buf[BK];
  threadgroup float y_buf[BV];

  const device float *q_bh = q + bh * seq_len * BK;
  const device float *k_bh = k + bh * seq_len * BK;
  const device float *v_bh = v + bh * seq_len * v_dim;
  device float *state_bh = state + bh * BK * v_dim;
  device float *out_bh = output + bh * seq_len * v_dim;

  float s[BK];
  for (int j = 0; j < BK; j++) {
    s[j] = state_bh[j * v_dim + v_idx];
  }
  // The decay buffer is unread without a decay per channel.
  y_buf[v_idx] = delta_step<BK, BV, false>(s, q_bh, k_bh, g + bh, beta[bh], v_bh[v_idx], k_buf, q_buf, k_buf, tid);
  for (int j = 0; j < BK; j++) {
    state_bh[j * v_dim + v_idx] = s[j];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < 32) {
    const int b = bh / Hv;
    const int h = bh % Hv;
    const int z_row = b * z_stride_bt + (h / z_vpg) * z_stride_g + (h % z_vpg) * BV;
    // Written out, not `norm_gate_row`: the driver's compiler lowers that call differently here and the bits move.
    const int per_thread = BV / 32;
    float y_vals[GDN_L2NORM_MAX_PER_THREAD];
    float z_vals[GDN_L2NORM_MAX_PER_THREAD];
    float w_vals[GDN_L2NORM_MAX_PER_THREAD];
    float sum_sq = 0.0f;
    for (int i = 0; i < per_thread; i++) {
      const int j = (int)tid * per_thread + i;
      y_vals[i] = y_buf[j];
      z_vals[i] = z[z_row + j];
      w_vals[i] = w[j];
      sum_sq += y_vals[i] * y_vals[i];
    }
    sum_sq = simd_sum(sum_sq);
    const float V_f = (float)BV;
    const float factor = rsqrt(sum_sq / V_f + eps);
    for (int i = 0; i < per_thread; i++) {
      const int j = (int)tid * per_thread + i;
      const float normed = y_vals[i] * w_vals[i] * factor;
      const float sig_z = 1.0f / (1.0f + exp(-z_vals[i]));
      const float gate_z = sigmoid_gate != 0 ? sig_z : z_vals[i] * sig_z;
      out_bh[j] = normed * gate_z;
    }
  }
}

typedef decltype(gated_delta_rule_norm_gate_kernel<128, 128>) norm_gate_t;
template [[host_name("gated_delta_rule_norm_gate_128_128")]] [[kernel]] norm_gate_t gated_delta_rule_norm_gate_kernel<128, 128>;
typedef decltype(delta_rule_kernel<128, 64, false>) delta_rule_t;
template [[host_name("gated_delta_rule_128_64")]] [[kernel]] delta_rule_t delta_rule_kernel<128, 64, false>;
template [[host_name("kda_delta_rule_128_64")]] [[kernel]] delta_rule_t delta_rule_kernel<128, 64, true>;

// mlx-lm's gated_delta_step_packed_btree geometry: four lanes a V column, 16 columns a 64-thread group.
// A lane's 32 state rows run as eight four-element chains, summed by a pinned butterfly (in-lane, then xor 1 and 2).
template <int BK>
[[kernel]] void gated_delta_rule_seq_v4_kernel(
    const device float *q [[buffer(0)]], const device float *k [[buffer(1)]],
    const device float *v [[buffer(2)]], const device float *g [[buffer(3)]],
    const device float *beta [[buffer(4)]], device float *state [[buffer(5)]],
    device float *output [[buffer(6)]], constant int &seq_len [[buffer(7)]],
    constant int &v_dim [[buffer(8)]], GDN_EMIT_PARAMS,
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]) {
#pragma clang fp reassociate(off)
  constexpr int LANES_PER_ROW = 4;
  constexpr int ROWS_PER_SG = 32 / LANES_PER_ROW;
  constexpr int PER_LANE = BK / LANES_PER_ROW;
  constexpr int CHAINS = PER_LANE / 4;
  const int lane_in_row = (int)tpitg.x % LANES_PER_ROW;
  const int bh = (int)tgpig.y;
  const int v_idx = ((int)tgpig.x * 2 + (int)tpitg.y) * ROWS_PER_SG + (int)tpitg.x / LANES_PER_ROW;

  if (v_idx >= v_dim) return;

  const device float *q_bh = q + bh * seq_len * BK;
  const device float *k_bh = k + bh * seq_len * BK;
  const device float *v_bh = v + bh * seq_len * v_dim;
  const device float *g_bh = g + bh * seq_len;
  const device float *beta_bh = beta + bh * seq_len;
  device float *state_bh = state + bh * BK * v_dim;
  device float *out_bh = output + bh * seq_len * v_dim;

  const int k0 = lane_in_row * PER_LANE;
  float ls[PER_LANE];
  for (short e = 0; e < PER_LANE; e++) {
    ls[e] = state_bh[(k0 + e) * v_dim + v_idx];
  }

  int ne = 0;
  for (int t = 0; t < seq_len; t++) {
    const float decay = fast::exp(g_bh[t]);
    const float beta_t = beta_bh[t];
    const float v_t = v_bh[t * v_dim + v_idx];
    const device float *k_t = k_bh + t * BK + k0;
    const device float *q_t = q_bh + t * BK + k0;

    float part[CHAINS];
    for (short c = 0; c < CHAINS; c++) {
      float acc = 0.0f;
      for (short i = 0; i < 4; i++) {
        const short e = c * 4 + i;
        ls[e] *= decay;
        acc = fma(ls[e], k_t[e], acc);
      }
      part[c] = acc;
    }
    float s_k = ((part[0] + part[1]) + (part[2] + part[3])) +
                ((part[4] + part[5]) + (part[6] + part[7]));
    s_k += simd_shuffle_xor(s_k, 1);
    s_k += simd_shuffle_xor(s_k, 2);

    const float d = (v_t - s_k) * beta_t;

    for (short c = 0; c < CHAINS; c++) {
      float acc = 0.0f;
      for (short i = 0; i < 4; i++) {
        const short e = c * 4 + i;
        ls[e] = fma(k_t[e], d, ls[e]);
        acc = fma(ls[e], q_t[e], acc);
      }
      part[c] = acc;
    }
    float y = ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
    y += simd_shuffle_xor(y, 1);
    y += simd_shuffle_xor(y, 2);

    if (lane_in_row == 0) {
      out_bh[t * v_dim + v_idx] = y;
    }
    if (ne < GDN_MAX_EMITS && t + 1 == emit_at[ne]) {
      device float *dst = emit_dst(ne++, emit0, emit1, emit2, emit3) + bh * BK * v_dim;
      for (short e = 0; e < PER_LANE; e++) {
        dst[(k0 + e) * v_dim + v_idx] = ls[e];
      }
    }
  }

  for (short e = 0; e < PER_LANE; e++) {
    state_bh[(k0 + e) * v_dim + v_idx] = ls[e];
  }
}

typedef decltype(gated_delta_rule_seq_v4_kernel<128>) seq_v4_t;
template [[host_name("gated_delta_rule_seq_v4_128")]] [[kernel]] seq_v4_t gated_delta_rule_seq_v4_kernel<128>;

[[kernel]] void save_conv_state_kernel(const device float *x [[buffer(0)]],
                                       device float *conv_state_out [[buffer(1)]],
                                       constant int &batch_size [[buffer(2)]],
                                       constant int &conv_dim [[buffer(3)]],
                                       constant int &seq_len [[buffer(4)]],
                                       constant int &kernel_size [[buffer(5)]],
                                       uint2 gid [[thread_position_in_grid]]) {
  const int ch = gid.x;
  const int b = gid.y;

  if (ch >= conv_dim || b >= batch_size)
    return;

  const device float *x_bch = x + (b * conv_dim + ch) * seq_len;
  device float *cs = conv_state_out + (b * conv_dim + ch) * kernel_size;

  int pad = kernel_size - seq_len;
  for (int i = 0; i < kernel_size; i++) {
    if (i < pad) {
      cs[i] = 0.0f;
    } else {
      cs[i] = x_bch[seq_len - kernel_size + i];
    }
  }
}

// A thread's `POS_PER_THREAD` outputs of one (b, ch) share its weights, loaded once; kernel_size ≤ 4 (`wv`).
[[kernel]] void causal_conv1d_full_v2_kernel(
    const device float *x [[buffer(0)]], const device float *weight [[buffer(1)]],
    device float *output [[buffer(2)]], constant int &batch_size [[buffer(3)]],
    constant int &conv_dim [[buffer(4)]], constant int &seq_len [[buffer(5)]],
    constant int &kernel_size [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]) {
  constexpr int POS_PER_THREAD = 8;
  const int ch = gid.x;
  const int pos_base = (int)gid.y * POS_PER_THREAD;
  const int b = gid.z;

  if (ch >= conv_dim || pos_base >= seq_len || b >= batch_size) return;

  const device float *x_bch = x + (b * conv_dim + ch) * seq_len;
  const device float *w = weight + ch * kernel_size;

  float wv[4];
  for (int i = 0; i < kernel_size; i++) wv[i] = w[i];

  device float *out_bch = output + (b * conv_dim + ch) * seq_len;

  for (int p = 0; p < POS_PER_THREAD; p++) {
    const int pos = pos_base + p;
    if (pos >= seq_len) break;

    float acc = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
      const int src_pos = pos - (kernel_size - 1) + i;
      const float x_val = (src_pos >= 0) ? x_bch[src_pos] : 0.0f;
      acc = fma(x_val, wv[i], acc);
    }
    const float sig = 1.0f / (1.0f + fast::exp(-acc));
    out_bch[pos] = acc * sig;
  }
}

template <bool CONV>
static inline float gdn_prep_channel(const device float *src, const device float *conv_w,
                                     device float *conv_state, int b, int t, int S,
                                     int conv_dim, int ch, int group_off, int ksize) {
  if (CONV) {
    device float *cs = conv_state + (b * conv_dim + ch) * ksize;
    const device float *w = conv_w + ch * ksize;
    for (int i = 0; i < ksize - 1; i++) {
      cs[i] = cs[i + 1];
    }
    cs[ksize - 1] = src[group_off];
    float acc = 0.0f;
    for (int i = 0; i < ksize; i++) {
      acc += cs[i] * w[i];
    }
    const float sig = 1.0f / (1.0f + exp(-acc));
    return acc * sig;
  }
  return src[(b * S + t) * conv_dim + ch];
}

// The recurrence's inputs as (B·Hv, S, D), q and k repeated per v-head; `CONV` steps decode's conv state in place.
// One 32-lane group per (t, h_k, b), alone on its channels' conv state; a lane sums D/32 consecutive elements in order.
template <bool CONV>
[[kernel]] void gdn_prep_kernel(
    const device float *src [[buffer(0)]],       // `CONV`: mixed_qkvz (B, Hk·(2Dk + 2·vpg·Dv)), else (B, S, conv_dim)
    const device float *conv_w [[buffer(1)]],    // (conv_dim, K); `CONV` only
    device float *conv_state [[buffer(2)]],      // (B, conv_dim, K), updated in place; `CONV` only
    const device float *mixed_ba [[buffer(3)]],  // (B, S, Hk, 2·vpg)
    const device float *a_log [[buffer(4)]],     // (Hv)
    const device float *dt_bias [[buffer(5)]],   // (Hv)
    const device float *l2_w [[buffer(6)]],      // (Dk)
    device float *q_bh [[buffer(7)]],
    device float *k_bh [[buffer(8)]],
    device float *v_bh [[buffer(9)]],
    device float *beta_out [[buffer(10)]],
    device float *g_out [[buffer(11)]],
    constant int &B [[buffer(12)]],
    constant int &S [[buffer(13)]],
    constant int &Hk [[buffer(14)]],
    constant int &Dk [[buffer(15)]],
    constant int &Dv [[buffer(16)]],
    constant int &vpg [[buffer(17)]],
    constant int &ksize [[buffer(18)]],
    constant float &scale [[buffer(19)]],
    constant float &eps [[buffer(20)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]]) {
  const int t = (int)tgpig.x;
  const int h_k = (int)tgpig.y;
  const int b = (int)tgpig.z;
  if (t >= S || h_k >= Hk || b >= B)
    return;
  const int Hv = Hk * vpg;
  const int key_dim = Hk * Dk;
  const int v_chunk = vpg * Dv;
  const int conv_dim = 2 * key_dim + Hk * v_chunk;
  const int group_base = (b * Hk + h_k) * (2 * Dk + 2 * v_chunk);

  const int qk_per = Dk / GDN_L2NORM_TG_THREADS;
  const int v_per = Dv / GDN_L2NORM_TG_THREADS;
  float q_vals[GDN_L2NORM_MAX_PER_THREAD];
  float k_vals[GDN_L2NORM_MAX_PER_THREAD];
  float w_vals[GDN_L2NORM_MAX_PER_THREAD];
  float q_sumsq = 0.0f;
  float k_sumsq = 0.0f;
  for (int i = 0; i < qk_per; i++) {
    const int j = (int)tiisg * qk_per + i;
    q_vals[i] = gdn_prep_channel<CONV>(src, conv_w, conv_state, b, t, S, conv_dim,
                                       h_k * Dk + j, group_base + j, ksize);
    k_vals[i] = gdn_prep_channel<CONV>(src, conv_w, conv_state, b, t, S, conv_dim,
                                       key_dim + h_k * Dk + j, group_base + Dk + j, ksize);
    w_vals[i] = l2_w[j];
    q_sumsq += q_vals[i] * q_vals[i];
    k_sumsq += k_vals[i] * k_vals[i];
  }
  q_sumsq = simd_sum(q_sumsq);
  k_sumsq = simd_sum(k_sumsq);
  const float D_f = (float)Dk;
  const float q_factor = scale * sqrt(D_f / (q_sumsq + eps));
  const float k_factor = sqrt(D_f / (k_sumsq + eps));

  for (int g = 0; g < vpg; g++) {
    const int h_v = h_k * vpg + g;
    const int qk_out = ((b * Hv + h_v) * S + t) * Dk;
    for (int i = 0; i < qk_per; i++) {
      const int j = (int)tiisg * qk_per + i;
      q_bh[qk_out + j] = q_vals[i] * w_vals[i] * q_factor;
      k_bh[qk_out + j] = k_vals[i] * w_vals[i] * k_factor;
    }
    const int v_out = ((b * Hv + h_v) * S + t) * Dv;
    for (int i = 0; i < v_per; i++) {
      const int j = (int)tiisg * v_per + i;
      v_bh[v_out + j] = gdn_prep_channel<CONV>(src, conv_w, conv_state, b, t, S, conv_dim,
                                               2 * key_dim + h_k * v_chunk + g * Dv + j,
                                               group_base + 2 * Dk + g * Dv + j, ksize);
    }
  }

  if ((int)tiisg < vpg) {
    const int g = (int)tiisg;
    const int h_v = h_k * vpg + g;
    const int in_base = ((b * S + t) * Hk + h_k) * 2 * vpg;
    const float b_val = mixed_ba[in_base + g];
    const float a_val = mixed_ba[in_base + vpg + g];
    const float softplus_val = log(1.0f + exp(a_val + dt_bias[h_v]));
    beta_out[(b * Hv + h_v) * S + t] = 1.0f / (1.0f + exp(-b_val));
    g_out[(b * Hv + h_v) * S + t] = -exp(a_log[h_v]) * softplus_val;
  }
}

typedef decltype(gdn_prep_kernel<true>) gdn_prep_t;
template [[host_name("gdn_prep_conv")]] [[kernel]] gdn_prep_t gdn_prep_kernel<true>;
template [[host_name("gdn_prep")]] [[kernel]] gdn_prep_t gdn_prep_kernel<false>;

// One simdgroup per (b, t, h) row, V/32 elements a lane (V a multiple of 32, ≤ 512); y and out (B, S, Hv, V).

[[kernel]] void gdn_norm_gate_kernel(
    const device float *y [[buffer(0)]],
    const device float *z [[buffer(1)]],
    const device float *w [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant int &B [[buffer(4)]],
    constant int &S [[buffer(5)]],
    constant int &Hv [[buffer(6)]],
    constant int &V [[buffer(7)]],
    constant float &eps [[buffer(8)]],
    constant int &sigmoid_gate [[buffer(9)]],
    constant int &z_stride_bt [[buffer(10)]],
    constant int &z_stride_g [[buffer(11)]],
    constant int &z_vpg [[buffer(12)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]]) {
  const int t = (int)tgpig.x;
  const int h = (int)tgpig.y;
  const int b = (int)tgpig.z;
  if (t >= S || h >= Hv || b >= B)
    return;

  const int row_offset = ((b * S + t) * Hv + h) * V;
  // z is (B, S, G, z_vpg·V), head h at (h / z_vpg, (h % z_vpg)·V); z_vpg = 1 is a contiguous (B, S, Hv, V).
  const int z_row = (b * S + t) * z_stride_bt + (h / z_vpg) * z_stride_g + (h % z_vpg) * V;
  norm_gate_row(y + row_offset, z + z_row, w, out + row_offset, V, eps, sigmoid_gate, tiisg);
}

// y (B,H,S,D), the attention's bf16 widened exactly, times sigmoid(gate), as F32 (B,S,H·D) for o_proj.
// The gate is read in place, each head's [q | gate] second half; one simdgroup per (b, t, h), D a multiple of 32.

[[kernel]] void attn_output_gate_kernel(
    const device bfloat16_t *y [[buffer(0)]],
    const device float *gate [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant int &B [[buffer(3)]],
    constant int &H [[buffer(4)]],
    constant int &S [[buffer(5)]],
    constant int &D [[buffer(6)]],
    constant int &gate_row [[buffer(7)]],
    constant int &gate_head [[buffer(8)]],
    constant int &gate_off [[buffer(9)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]]) {
  const int t = (int)tgpig.x;
  const int h = (int)tgpig.y;
  const int b = (int)tgpig.z;
  if (t >= S || h >= H || b >= B)
    return;

  const int per_thread = D / GDN_L2NORM_TG_THREADS;
  const int y_row_off = ((b * H + h) * S + t) * D;
  const int gate_row_off = (b * S + t) * gate_row + h * gate_head + gate_off;
  const int out_row_off = (b * S + t) * (H * D) + h * D;
  for (int i = 0; i < per_thread; i++) {
    const int j = (int)tiisg * per_thread + i;
    if (j < D) {
      const float y_val = (float)y[y_row_off + j];
      const float g_val = gate[gate_row_off + j];
      const float sig_g = 1.0f / (1.0f + exp(-g_val));
      out[out_row_off + j] = y_val * sig_g;
    }
  }
}

// mixed_qkvz (B,S,Hk, per k-head q | k | v | z) into the conv's mixed_qkv (B,S,[q | k | v]); the norm gate reads z in place.
// One simdgroup per (b, s, h_k), Dk and vpg·Dv multiples of 32; v ordered g·Dv + j, h_v = h_k·vpg + g.

[[kernel]] void gdn_qkvz_split_kernel(
    const device float *mixed_qkvz [[buffer(0)]],
    device float *mixed_qkv [[buffer(1)]],
    constant int &B [[buffer(2)]],
    constant int &S [[buffer(3)]],
    constant int &Hk [[buffer(4)]],
    constant int &Dk [[buffer(5)]],
    constant int &Dv [[buffer(6)]],
    constant int &v_per_group [[buffer(7)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]]) {
  const int t = (int)tgpig.x;
  const int h_k = (int)tgpig.y;
  const int b = (int)tgpig.z;
  if (t >= S || h_k >= Hk || b >= B)
    return;

  const int Hv = Hk * v_per_group;
  const int v_chunk = v_per_group * Dv;
  const int group_size = 2 * Dk + 2 * v_chunk;
  const int key_dim = Hk * Dk;
  const int value_dim = Hv * Dv;
  const int conv_dim = 2 * key_dim + value_dim;

  // q, k, then v, which keeps its inner (g, j) order, so the chunk copies straight.
  const int qkvz_row_base = ((b * S + t) * Hk + h_k) * group_size;
  const int qkv_row_base = (b * S + t) * conv_dim;
  const int src[3] = {qkvz_row_base, qkvz_row_base + Dk, qkvz_row_base + 2 * Dk};
  const int dst[3] = {qkv_row_base + h_k * Dk, qkv_row_base + key_dim + h_k * Dk,
                      qkv_row_base + 2 * key_dim + h_k * v_chunk};
  const int width[3] = {Dk, Dk, v_chunk};
  for (int part = 0; part < 3; part++) {
    const int per_thread = width[part] / GDN_L2NORM_TG_THREADS;
    for (int i = 0; i < per_thread; i++) {
      const int j = (int)tiisg * per_thread + i;
      if (j < width[part]) {
        mixed_qkv[dst[part] + j] = mixed_qkvz[src[part] + j];
      }
    }
  }
}

// One threadgroup per output row (b, h, s), from input row (b, s, h): q's rows, then k's, then v's, written bf16.
// q (each head's [q | gate] first half) and k take candle's RMS norm and rope at its block width, byte for byte.

[[kernel]] void attn_pack_kernel(
    const device float *qg [[buffer(0)]],
    const device float *k [[buffer(1)]],
    const device float *v [[buffer(2)]],
    const device float *qw [[buffer(3)]],
    const device float *kw [[buffer(4)]],
    const device float *cos_t [[buffer(5)]],
    const device float *sin_t [[buffer(6)]],
    device bfloat16_t *q_out [[buffer(7)]],
    device bfloat16_t *k_out [[buffer(8)]],
    device bfloat16_t *v_out [[buffer(9)]],
    constant int &S [[buffer(10)]],
    constant int &H [[buffer(11)]],
    constant int &Hkv [[buffer(12)]],
    constant int &D [[buffer(13)]],
    constant int &rot [[buffer(14)]],
    constant float &eps [[buffer(15)]],
    constant int &q_rows [[buffer(16)]],
    constant int &kv_rows [[buffer(17)]],
    threadgroup float *shm [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint block_dim [[threads_per_threadgroup]]) {
  const device float *src;
  const device float *w = nullptr;
  device bfloat16_t *dst;
  int r, heads, head_stride;
  if ((int)row < q_rows) {
    r = (int)row; heads = H; head_stride = 2 * D; src = qg; w = qw; dst = q_out;
  } else if ((int)row < q_rows + kv_rows) {
    r = (int)row - q_rows; heads = Hkv; head_stride = D; src = k; w = kw; dst = k_out;
  } else if ((int)row < q_rows + 2 * kv_rows) {
    r = (int)row - q_rows - kv_rows; heads = Hkv; head_stride = D; src = v; dst = v_out;
  } else {
    return;
  }
  const int s = r % S;
  const int h = (r / S) % heads;
  const int b = r / (S * heads);
  const device float *x = src + ((size_t)(b * S + s) * heads + h) * head_stride;
  device bfloat16_t *y = dst + (size_t)r * D;
  if (w == nullptr) {
    for (int d = (int)tid; d < D; d += (int)block_dim)
      y[d] = (bfloat16_t)x[d];
    return;
  }
  // candle's rmsnorm_f32, its reduction order step for step.
  float tmp = 0.0f;
  for (int d = (int)tid; d < D; d += (int)block_dim)
    tmp += (x[d] * x[d]);
  shm[tid] = tmp;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint st = block_dim / 2; st >= 64; st >>= 1) {
    if (tid < st)
      shm[tid] = shm[tid] + shm[tid + st];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float sum = 0.0f;
  if (tid < 32) {
    sum = shm[tid] + shm[tid + 32];
    simdgroup_barrier(mem_flags::mem_none);
    sum = sum + simd_shuffle_down(sum, 16);
    sum = sum + simd_shuffle_down(sum, 8);
    sum = sum + simd_shuffle_down(sum, 4);
    sum = sum + simd_shuffle_down(sum, 2);
    sum = sum + simd_shuffle_down(sum, 1);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0)
    shm[0] = rsqrt(fast::divide(sum, float(D)) + eps);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total = shm[0];
  const int hr = rot / 2;
  for (int d = (int)tid; d < D; d += (int)block_dim) {
    float out;
    if (d >= rot) {
      float val = x[d] * total;
      val *= w[d];
      out = val;
    } else {
      const int i = d < hr ? d : d - hr;
      float x1 = x[i] * total;
      x1 *= w[i];
      float x2 = x[i + hr] * total;
      x2 *= w[i + hr];
      const float c = cos_t[s * hr + i];
      const float sn = sin_t[s * hr + i];
      // candle's rope as its kernel compiles: x1·c fused with the rounded x2·sn, as fmas the compiler cannot regroup.
      out = d < hr ? fma(x1, c, -(x2 * sn)) : fma(x2, c, x1 * sn);
    }
    y[d] = (bfloat16_t)out;
  }
}
