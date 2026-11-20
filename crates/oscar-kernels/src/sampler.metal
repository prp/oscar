#include <metal_stdlib>
using namespace metal;

// The token sampler over one row of logits: (token, log-probability bits) for the host and the next step's embedding.

constant uint KMAX = 128;

struct SampleParams {
    uint n;
    uint top_k;     // 1..KMAX
    float inv_temp;
    float top_p;    // off outside (0, 1)
    float min_p;    // off outside (0, 1)
    uint seed_lo;
    uint seed_hi;
    uint pos_lo;
    uint pos_hi;
    float penalty;
};

inline uint fkey(float f) {
    uint u = as_type<uint>(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

template <typename T>
inline T tg_sum(T v, threadgroup T* shm, uint tid, uint bd) {
    shm[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = bd / 2; s > 0; s >>= 1) {
        if (tid < s) shm[tid] += shm[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    T r = shm[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

struct MaxAt {
    float v;
    uint i;
};

// The threadgroup's maximum, the last index among equal maxima (the CPU sampler's `max_by`); every thread gets it.
inline MaxAt tg_max_last(MaxAt t, threadgroup float* shf, threadgroup uint* shu, uint tid, uint bd) {
    shf[tid] = t.v;
    shu[tid] = t.i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = bd / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = shf[tid], b = shf[tid + s];
            if (b > a || (b == a && shu[tid + s] > shu[tid])) {
                shf[tid] = b;
                shu[tid] = shu[tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    MaxAt r = MaxAt{shf[0], shu[0]};
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

inline ulong splitmix64(ulong z) {
    z += 0x9e3779b97f4a7c15ul;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ul;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebul;
    return z ^ (z >> 31);
}

// `penalty` comes off a logit `seen` marks (the prompt's by the host, the rest by the kernel), before the temperature.
template <typename T>
inline float logit(const device T* x, uint i, const device uint* seen, float penalty) {
    float v = float(x[i]);
    return (penalty != 0.0f && ((seen[i >> 5] >> (i & 31)) & 1u)) ? v - penalty : v;
}

inline void mark_seen(device uint* seen, uint tok) {
    seen[tok >> 5] |= 1u << (tok & 31);
}

// Greedy stops at the merge; a sampled row gathers the logits within 16 of the maximum (where any model's top-k lives)
// in tiers by distance, and select sorts the tightest tiers holding k, or bisects the whole row where they overflow.
constant uint TIERS = 4;
constant uint TIER_CAP = 512;
constant uint CAND_MAX = TIERS * TIER_CAP;
constant float TIER_DELTA[TIERS] = {2.0f, 4.0f, 8.0f, 16.0f};

struct SlicePartial {
    float max;
    uint idx;
    float sum;
};

struct SampleScratch {
    float m;
    uint argmax;
    float z;
    uint count[TIERS];
};

template <typename T>
kernel void oscar_sample_slices(
    const device T* x [[buffer(0)]],
    device SlicePartial* partials [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant float& inv_temp [[buffer(3)]],
    const device uint* seen [[buffer(4)]],
    constant float& penalty [[buffer(5)]],
    uint g [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint bd [[threads_per_threadgroup]]
) {
    threadgroup float shf[256];
    threadgroup uint shu[256];
    const uint per = (n + groups - 1) / groups;
    const uint lo = g * per;
    const uint hi = min(lo + per, n);
    float best = -INFINITY;
    uint bi = 0;
    for (uint i = lo + tid; i < hi; i += bd) {
        float v = logit(x, i, seen, penalty);
        if (v >= best) { best = v; bi = i; }
    }
    const MaxAt top = tg_max_last(MaxAt{best, bi}, shf, shu, tid, bd);
    const float m = top.v;
    const uint argmax = top.i;
    float part = 0.0f;
    if (m > -INFINITY) {
        for (uint i = lo + tid; i < hi; i += bd) part += exp((logit(x, i, seen, penalty) - m) * inv_temp);
    }
    const float z = tg_sum(part, shf, tid, bd);
    if (tid == 0) {
        partials[g].max = m;
        partials[g].idx = argmax;
        partials[g].sum = z;
    }
}

template [[host_name("oscar_sample_slices_f32")]] kernel decltype(oscar_sample_slices<float>) oscar_sample_slices<float>;
template [[host_name("oscar_sample_slices_bf16")]] kernel decltype(oscar_sample_slices<bfloat>) oscar_sample_slices<bfloat>;

kernel void oscar_sample_merge(
    const device SlicePartial* partials [[buffer(0)]],
    device uint* out [[buffer(1)]],
    device SampleScratch* scratch [[buffer(2)]],
    constant uint& groups [[buffer(3)]],
    constant float& inv_temp [[buffer(4)]],
    device uint* seen [[buffer(5)]],
    constant uint& mark [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint bd [[threads_per_threadgroup]]
) {
    threadgroup float shf[256];
    threadgroup uint shu[256];
    float best = -INFINITY;
    uint bg = 0;
    for (uint g = tid; g < groups; g += bd) {
        float v = partials[g].max;
        if (v >= best) { best = v; bg = g; }
    }
    const MaxAt top = tg_max_last(MaxAt{best, bg}, shf, shu, tid, bd);
    const float m = top.v;
    const uint gbest = top.i;
    float part = 0.0f;
    for (uint g = tid; g < groups; g += bd) {
        if (partials[g].max > -INFINITY) part += partials[g].sum * exp((partials[g].max - m) * inv_temp);
    }
    const float z = tg_sum(part, shf, tid, bd);
    if (tid == 0) {
        out[0] = partials[gbest].idx;
        out[1] = as_type<uint>(-log(z));
        scratch->m = m;
        scratch->argmax = partials[gbest].idx;
        scratch->z = z;
        for (uint t = 0; t < TIERS; t++) scratch->count[t] = 0;
        // A greedy row is decided here; a sampled row's select marks its own.
        if (mark) mark_seen(seen, partials[gbest].idx);
    }
}

template <typename T>
kernel void oscar_sample_compact(
    const device T* x [[buffer(0)]],
    device SampleScratch* scratch [[buffer(1)]],
    device float* cand_v [[buffer(2)]],
    device uint* cand_i [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    const device uint* seen [[buffer(5)]],
    constant float& penalty [[buffer(6)]],
    uint g [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint bd [[threads_per_threadgroup]]
) {
    const uint per = (n + groups - 1) / groups;
    const uint lo = g * per;
    const uint hi = min(lo + per, n);
    const float m = scratch->m;
    device atomic_uint* count = (device atomic_uint*)scratch->count;
    for (uint i = lo + tid; i < hi; i += bd) {
        float v = logit(x, i, seen, penalty);
        float d = m - v;
        if (d < TIER_DELTA[TIERS - 1]) {
            uint t = 0;
            while (d >= TIER_DELTA[t]) t++;
            uint slot = atomic_fetch_add_explicit(count + t, 1u, memory_order_relaxed);
            if (slot < TIER_CAP) { cand_v[t * TIER_CAP + slot] = v; cand_i[t * TIER_CAP + slot] = i; }
        }
    }
}

template [[host_name("oscar_sample_compact_f32")]] kernel decltype(oscar_sample_compact<float>) oscar_sample_compact<float>;
template [[host_name("oscar_sample_compact_bf16")]] kernel decltype(oscar_sample_compact<bfloat>) oscar_sample_compact<bfloat>;

// Over survivors sorted descending: top-p keeps the element crossing the bound, min-p is against the largest; one draw.
inline void draw(threadgroup const float* sv, threadgroup const uint* si, uint nc,
                 float m, float z, constant SampleParams& p, device uint* out) {
    const float it = p.inv_temp;
    float prob[KMAX];
    float cum = 0.0f;
    float total = 0.0f;
    const float max_p = exp((sv[0] - m) * it) / z;
    const bool use_top_p = p.top_p > 0.0f && p.top_p < 1.0f;
    const bool use_min_p = p.min_p > 0.0f && p.min_p < 1.0f;
    for (uint j = 0; j < nc; j++) {
        float pj = exp((sv[j] - m) * it) / z;
        bool keep = true;
        if (use_top_p) { if (cum >= p.top_p) keep = false; else cum += pj; }
        if (use_min_p && max_p * p.min_p >= pj) keep = false;
        prob[j] = keep ? pj : 0.0f;
        total += prob[j];
    }
    ulong seed = (ulong(p.seed_hi) << 32) | ulong(p.seed_lo);
    ulong pos = (ulong(p.pos_hi) << 32) | ulong(p.pos_lo);
    ulong h = splitmix64(seed ^ splitmix64(pos));
    float u = float(h >> 40) * (1.0f / 16777216.0f);
    float target = u * total;
    uint pick = 0;
    float acc = 0.0f;
    for (uint j = 0; j < nc; j++) {
        if (prob[j] <= 0.0f) continue;
        pick = j;
        acc += prob[j];
        if (acc > target) break;
    }
    out[0] = si[pick];
    out[1] = as_type<uint>(log(prob[pick] > 0.0f ? prob[pick] : max_p));
}

// Bitonic sort of `slots` entries (a power of two), descending by value, ascending index on ties.
inline void sort_desc(threadgroup float* v, threadgroup uint* ix, uint slots, uint tid, uint bd) {
    for (uint size = 2; size <= slots; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            for (uint t = tid; t < slots; t += bd) {
                uint j = t ^ stride;
                if (j > t) {
                    bool up = ((t & size) == 0);
                    float a = v[t], b = v[j];
                    uint ia = ix[t], ib = ix[j];
                    bool a_before_b = (a > b) || (a == b && ia < ib);
                    if (up != a_before_b) {
                        v[t] = b; v[j] = a; ix[t] = ib; ix[j] = ia;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

template <typename T>
kernel void oscar_sample_select(
    const device T* x [[buffer(0)]],
    const device SampleScratch* scratch [[buffer(1)]],
    const device float* cand_v [[buffer(2)]],
    const device uint* cand_i [[buffer(3)]],
    device uint* out [[buffer(4)]],
    constant SampleParams& p [[buffer(5)]],
    device uint* seen [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint bd [[threads_per_threadgroup]]
) {
    threadgroup float sv[CAND_MAX];
    threadgroup uint si[CAND_MAX];
    threadgroup uint shu[1024];
    threadgroup atomic_uint ncand;
    const uint n = p.n;
    const uint k = min(p.top_k, KMAX);
    const float m = scratch->m;
    const float z = scratch->z;
    // The tightest tiers that hold k candidates, none of them overflowed.
    uint use = 0, found = 0;
    for (uint t = 0; t < TIERS; t++) {
        uint c = scratch->count[t];
        if (c > TIER_CAP) break;
        found += c;
        use = t + 1;
        if (found >= k) break;
    }
    uint nc;
    if (found >= k) {
        uint slots = KMAX;
        while (slots < found) slots <<= 1;
        uint w = 0;
        for (uint t = 0; t < use; t++) {
            uint c = scratch->count[t];
            for (uint j = tid; j < c; j += bd) {
                sv[w + j] = cand_v[t * TIER_CAP + j];
                si[w + j] = cand_i[t * TIER_CAP + j];
            }
            w += c;
        }
        for (uint j = found + tid; j < slots; j += bd) { sv[j] = -INFINITY; si[j] = 0xffffffffu; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sort_desc(sv, si, slots, tid, bd);
        nc = min(found, k);
    } else {
        // The window missed: the k-th largest key by bisection over the row.
        ulong lo = 0, hi = 0x100000000ul;
        while (hi - lo > 1) {
            ulong mid = lo + (hi - lo) / 2;
            uint c = 0;
            for (uint i = tid; i < n; i += bd) c += (fkey(logit(x, i, seen, p.penalty)) >= uint(mid)) ? 1u : 0u;
            c = tg_sum(c, shu, tid, bd);
            if (c >= k) lo = mid; else hi = mid;
        }
        const uint tau = uint(lo);
        if (tid == 0) atomic_store_explicit(&ncand, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < n; i += bd) {
            float v = logit(x, i, seen, p.penalty);
            if (fkey(v) >= tau) {
                uint slot = atomic_fetch_add_explicit(&ncand, 1u, memory_order_relaxed);
                if (slot < KMAX) { sv[slot] = v; si[slot] = i; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        nc = min(atomic_load_explicit(&ncand, memory_order_relaxed), KMAX);
        for (uint t = tid; t < KMAX; t += bd) {
            if (t >= nc) { sv[t] = -INFINITY; si[t] = 0xffffffffu; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sort_desc(sv, si, KMAX, tid, bd);
    }
    if (tid == 0) {
        draw(sv, si, nc, m, z, p, out);
        if (p.penalty != 0.0f) mark_seen(seen, out[0]);
    }
}

template [[host_name("oscar_sample_select_f32")]] kernel decltype(oscar_sample_select<float>) oscar_sample_select<float>;
template [[host_name("oscar_sample_select_bf16")]] kernel decltype(oscar_sample_select<bfloat>) oscar_sample_select<bfloat>;
