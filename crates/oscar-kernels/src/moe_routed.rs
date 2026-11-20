//! The routed-expert kernels: address rows let a resident layer, the resolver and a wave dispatch the same entry.

use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, MetalDevice, Result, Tensor};
use candle_metal_kernels::metal::{ComputeCommandEncoder, MetalResource, ProtocolObject, Retained};
use objc2_metal::{MTLBuffer, MTLResourceUsage};

use crate::loader::{candle_pipeline, launch, out, pipeline, set_params, size, Family, Metal};

/// The routed experts' quantisation, which picks each launch's kernel and geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QkkMoeDType {
    Q3K,
    Q4K,
    Q6K,
    /// For expert rows not a multiple of the 256-weight superblock (640-wide down projections).
    Q5_0,
    /// The Kimi K3 i-quants (`UD-IQ1_M`): gate and up IQ1_S or IQ2_XXS, down IQ1_S or IQ3_XXS.
    Iq1S,
    Iq2XXS,
    Iq3XXS,
}

/// A template parameter of the fused gate+up body: [`has_fused_gateup_kernel`] says which pairings exist.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ExpertActivation {
    /// `silu(gate) * up`.
    #[default]
    Silu,
    /// SiTU-GLU (Kimi K3), `(β₁·tanh(g/β₁) · σ(g)) * (β₂·tanh(u/β₂))`, its betas GGUF parameters.
    SituGlu { beta_gate: f32, beta_up: f32 },
    /// Clamped SwiGLU, `silu(min(g, L)) * clamp(u, -L, L)`, the limit a GGUF parameter (`swiglu_clamp_*`).
    SiluClamp { limit: f32 },
}

/// An expert format's facts, the table the launches read.
struct Format {
    /// (bytes, elements) per quantised block.
    block: (usize, usize),
    /// The unfused routed matvec: (kernel, (threads x, threads y, rows per threadgroup), codebook threadgroup bytes).
    mv: (&'static str, (usize, usize, usize), usize),
    /// The routed matmul's kernels: f32 tiles, then tensor-op tiles.
    mm: [&'static str; 2],
}

/// A row of the table from the format's suffix in the candle fork's kernel names.
macro_rules! format_row {
    ($sfx:literal, $block:expr, $geometry:expr, $tg_bytes:expr) => {
        &Format {
            block: $block,
            mv: (concat!("kernel_mul_mv_id_addressed_", $sfx, "_f32"), $geometry, $tg_bytes),
            mm: [
                concat!("kernel_mul_mm_id_with_hids_", $sfx, "_f32"),
                concat!("kernel_mul_mm_id_with_hids_nax_", $sfx, "_f32"),
            ],
        }
    };
}

impl QkkMoeDType {
    pub fn from_ggml(dtype: GgmlDType) -> Option<Self> {
        match dtype {
            GgmlDType::Q3K => Some(Self::Q3K),
            GgmlDType::Q4K => Some(Self::Q4K),
            GgmlDType::Q6K => Some(Self::Q6K),
            GgmlDType::Q5_0 => Some(Self::Q5_0),
            GgmlDType::Iq1S => Some(Self::Iq1S),
            GgmlDType::Iq2XXS => Some(Self::Iq2XXS),
            GgmlDType::Iq3XXS => Some(Self::Iq3XXS),
            _ => None,
        }
    }

    fn format(self) -> &'static Format {
        match self {
            // Its kernel's own geometry, not candle's table: 4 rows a threadgroup, 2 a simdgroup.
            Self::Q3K => format_row!("q3_K", (110, 256), (2, 32, 4), 0),
            Self::Q4K => format_row!("q4_K", (144, 256), (4, 16, 4), 0),
            Self::Q6K => format_row!("q6_K", (210, 256), (2, 32, 2), 0),
            Self::Q5_0 => format_row!("q5_0", (22, 32), (8, 8, 8), 0),
            // 64 threads exactly: the codebook staging loop, `32*sgitg + tiisg`, covers its table with two simdgroups.
            Self::Iq1S => format_row!("iq1_s", (50, 256), (2, 32, 8), 0),
            Self::Iq2XXS => format_row!("iq2_xxs", (66, 256), (2, 32, 8), 8192),
            Self::Iq3XXS => format_row!("iq3_xxs", (98, 256), (2, 32, 8), 8192),
        }
    }

    pub(crate) fn block_bytes(self) -> usize {
        self.format().block.0
    }

    fn block_elems(self) -> usize {
        self.format().block.1
    }

    /// A row of `k`, which its callers have refused unless it divides into blocks.
    fn row_bytes(self, k: usize) -> usize {
        k / self.block_elems() * self.block_bytes()
    }
}

/// What the routed kernels take: a top-k the GEMM's id map is built for, a layer's experts in its routing table.
pub fn routed_fits(num_experts: usize, topk: usize) -> bool {
    map0_kernel_for_topk(topk).is_some() && num_experts <= 1024 && crate::topk_fits(num_experts, topk)
}

/// map0 mis-sorts duplicate ids, which no token's top-k holds.
fn map0_kernel_for_topk(topk: usize) -> Option<&'static str> {
    match topk {
        8 => Some("mistralrs_moe_mat_mm_id_map0_ne20_8"),
        10 => Some("mistralrs_moe_mat_mm_id_map0_ne20_10"),
        16 => Some("mistralrs_moe_mat_mm_id_map0_ne20_16"),
        _ => None,
    }
}

/// One projection's experts: `row` holds I64 GPU addresses `[num_experts]`, null to skip, into `segments`, which a
/// launch declares whole, as declaring only those read pays as declaring none.
#[derive(Clone, Copy)]
pub struct ExpertRow<'a> {
    pub row: &'a Tensor,
    pub segments: &'a [Retained<ProtocolObject<dyn MTLBuffer>>],
}

impl ExpertRow<'_> {
    fn declare(self, e: &ComputeCommandEncoder) {
        let refs: Vec<&MetalResource> = self.segments.iter().map(|b| ProtocolObject::from_ref(&**b)).collect();
        e.use_resources(&refs, MTLResourceUsage::Read);
    }
}

pub fn stack_addresses(weight_bytes: &Tensor, num_experts: usize, bytes_per_expert: usize) -> Result<Tensor> {
    let m = Metal::of(weight_bytes, "expert stack")?;
    let base = m.buf.gpu_address() + m.off as u64;
    let addresses: Vec<i64> = (0..num_experts).map(|e| (base + (e * bytes_per_expert) as u64) as i64).collect();
    Tensor::from_vec(addresses, (num_experts,), weight_bytes.device())
}

pub fn all_experts(num_experts: usize, device: &Device) -> Result<Tensor> {
    Tensor::arange(0u32, num_experts as u32, device)
}

/// The kernel and the row alignment its geometry needs; a pairing no model needs has none.
fn fused_gateup_kernel(dtype: QkkMoeDType, activation: ExpertActivation) -> Option<(&'static str, usize)> {
    use ExpertActivation as A;
    use QkkMoeDType as D;
    match (dtype, activation) {
        (D::Q4K, A::Silu) => Some(("mistralrs_moe_fused_gateup_silu_q4k_f32", 4)),
        (D::Q3K, A::Silu) => Some(("mistralrs_moe_fused_gateup_silu_q3k_f32", 2)),
        (D::Q3K, A::SiluClamp { .. }) => Some(("mistralrs_moe_fused_gateup_siluclamp_q3k_f32", 2)),
        (D::Iq1S, A::SituGlu { .. }) => Some(("mistralrs_moe_fused_gateup_situ_iq1s_f32", 4)),
        (D::Iq2XXS, A::SituGlu { .. }) => Some(("mistralrs_moe_fused_gateup_situ_iq2xxs_f32", 4)),
        _ => None,
    }
}

/// Callers ask this, never a dtype list of their own, which would drift from the kernels.
pub fn has_fused_gateup_kernel(dtype: QkkMoeDType, activation: ExpertActivation) -> bool {
    fused_gateup_kernel(dtype, activation).is_some()
}

/// The chain takes the fused entry only where N divides into this.
pub fn fused_gateup_alignment(dtype: QkkMoeDType, activation: ExpertActivation) -> Option<usize> {
    fused_gateup_kernel(dtype, activation).map(|(_, align)| align)
}

pub fn fused_down_alignment(dtype: QkkMoeDType) -> Option<usize> {
    fused_down_kernel(dtype).map(|(_, rows, _)| rows)
}

/// The kernel, its rows a threadgroup, and its threads: `Some(n)` a fixed count walking the slots serially, `None` a
/// simdgroup a slot, the partials in threadgroup memory. Iq2XXS occurs on no supported model's down projection.
fn fused_down_kernel(dtype: QkkMoeDType) -> Option<(&'static str, usize, Option<usize>)> {
    use QkkMoeDType as D;
    match dtype {
        D::Q4K => Some(("mistralrs_moe_fused_down_weighted_sum_q4k_f32", 2, None)),
        D::Q6K => Some(("mistralrs_moe_fused_down_weighted_sum_q6k_f32", 2, Some(64))),
        D::Q3K => Some(("mistralrs_moe_fused_down_weighted_sum_q3k_f32", 4, Some(32))),
        D::Q5_0 => Some(("mistralrs_moe_fused_down_weighted_sum_q5_0_f32", 4, None)),
        D::Iq1S => Some(("mistralrs_moe_fused_down_weighted_sum_iq1s_f32", 4, Some(32))),
        D::Iq3XXS => Some(("mistralrs_moe_fused_down_weighted_sum_iq3xxs_f32", 4, Some(32))),
        D::Iq2XXS => None,
    }
}

/// The caller's tensor where given, so a layer's waves, each writing its resolved slices, fill one buffer.
fn routed_output(
    given: Option<&Tensor>, dev: &MetalDevice, shape: (usize, usize, usize), label: &'static str,
) -> Result<Tensor> {
    let (t, s, n) = shape;
    match given {
        Some(o) if o.dims() == [t, s, n] && o.dtype() == DType::F32 && o.is_contiguous() => Ok(o.clone()),
        Some(o) => candle_core::bail!(
            "{label}: output must be contiguous F32 ({t}, {s}, {n}), got {:?} {:?}",
            o.dtype(),
            o.dims()
        ),
        None => Ok(out(dev, shape, DType::F32, label)?.0),
    }
}

fn check_row(r: ExpertRow<'_>, num_experts: usize, what: &str) -> Result<()> {
    refuse_if!(
        r.row.dtype() != DType::I64 || r.row.elem_count() != num_experts,
        "{what} row must be I64 [{num_experts}], got {:?} [{}]",
        r.row.dtype(),
        r.row.elem_count()
    );
    Ok(())
}

fn topk_shape(topk_ids: &Tensor, what: &str) -> Result<(usize, usize)> {
    refuse_if!(topk_ids.dtype() != DType::U32, "{what}: topk_ids must be U32 (got {:?})", topk_ids.dtype());
    match topk_ids.dims() {
        &[t, k] => Ok((t, k)),
        d => candle_core::bail!("{what}: topk_ids must be rank-2 [num_tokens, topk], got {d:?}"),
    }
}

/// `[t,s,:] = act(gate[e]·x[t], up[e]·x[t])`, `e = topk_ids[t,s]`; an unresolved slot gets zeros and counts a miss.
#[allow(clippy::too_many_arguments)]
pub fn moe_gateup(
    gate: ExpertRow<'_>,
    up: ExpertRow<'_>,
    dtype: QkkMoeDType,
    activation: ExpertActivation,
    num_experts: usize,
    n: usize,
    k: usize,
    topk_ids: &Tensor,
    x: &Tensor,
    // A shared atomic_uint[1], one count per unresolved (token, slot); the draft lane reads it.
    miss_count: &Tensor,
) -> Result<Tensor> {
    let Some((kernel_name, align)) = fused_gateup_kernel(dtype, activation) else {
        candle_core::bail!("gateup: no fused kernel for {dtype:?} with {activation:?}")
    };
    refuse_if!(!k.is_multiple_of(dtype.block_elems()), "gateup: K ({k}) does not divide into {dtype:?} blocks");
    refuse_if!(!n.is_multiple_of(align), "gateup: N ({n}) must be divisible by {align}");
    refuse_if!(
        miss_count.dtype() != DType::U32 || miss_count.elem_count() != 1,
        "gateup: miss_count must be a 1-element U32 tensor"
    );
    check_row(gate, num_experts, "gate")?;
    check_row(up, num_experts, "up")?;
    let (num_tokens, topk) = topk_shape(topk_ids, "gateup")?;
    refuse_if!(x.dims() != [num_tokens, k], "gateup expects the input as (num_tokens, k), got {:?}", x.dims());
    let x = x.contiguous()?;
    refuse_if!(x.dtype() != DType::F32, "gateup requires F32 input, got {:?}", x.dtype());
    let ids = topk_ids.contiguous()?;

    let xm = Metal::of(&x, "input")?;
    let gate_row = Metal::of(gate.row, "gate row")?;
    let up_row = Metal::of(up.row, "up row")?;
    let idm = Metal::of(&ids, "topk_ids")?;
    let miss = Metal::of(miss_count, "miss_count")?;
    let p = pipeline(&xm.dev, Family::Moe, kernel_name)?;
    let (y, o) = out(&xm.dev, (num_tokens, topk, n), DType::F32, "moe-gateup-output")?;
    let (p0, p1) = match activation {
        ExpertActivation::Silu => (0.0f32, 0.0f32),
        ExpertActivation::SiluClamp { limit } => (limit, 0.0),
        ExpertActivation::SituGlu { beta_gate, beta_up } => (beta_gate, beta_up),
    };
    launch(&xm.dev, &p, kernel_name, [n.div_ceil(align), 1, num_tokens * topk], [32, 1, 1], |e| {
        set_params!(
            e,
            (
                &gate_row,
                &up_row,
                &xm,
                &o,
                &idm,
                k as i64,
                n as i64,
                topk as i64,
                num_tokens as i64,
                num_experts as i64,
                &miss,
                p0,
                p1
            )
        );
        gate.declare(e);
        up.declare(e);
        Ok(())
    })?;
    Ok(y)
}

/// `[t,:] = Σ_s w[t,s]·(down[e]·activated[t,s,:])`, `e = topk_ids[t,s]`, an unresolved slot contributing nothing.
#[allow(clippy::too_many_arguments)]
pub fn moe_down(
    down: ExpertRow<'_>, dtype: QkkMoeDType, num_experts: usize, n: usize, k: usize, topk_ids: &Tensor,
    topk_weights: &Tensor, activated: &Tensor,
) -> Result<Tensor> {
    let Some((kernel_name, rows, serial)) = fused_down_kernel(dtype) else {
        candle_core::bail!("down: no fused kernel for {dtype:?} - on no model's down projection")
    };
    refuse_if!(!k.is_multiple_of(dtype.block_elems()), "down: K ({k}) does not divide into {dtype:?} blocks");
    refuse_if!(!n.is_multiple_of(rows), "down: N ({n}) must be divisible by {rows}");
    check_row(down, num_experts, "down")?;
    let (num_tokens, topk) = topk_shape(topk_ids, "down")?;
    refuse_if!(
        activated.dims() != [num_tokens, topk, k],
        "down expects activations (num_tokens, topk, k) = ({num_tokens}, {topk}, {k}), got {:?}",
        activated.dims()
    );
    refuse_if!(
        topk_weights.dims() != [num_tokens, topk],
        "down: topk_weights shape {:?} != ({num_tokens}, {topk})",
        topk_weights.dims()
    );
    let activated = activated.contiguous()?;
    let weights = topk_weights.contiguous()?;
    refuse_if!(
        activated.dtype() != DType::F32 || weights.dtype() != DType::F32,
        "down requires F32 activations and weights"
    );
    let ids = topk_ids.contiguous()?;

    let am = Metal::of(&activated, "activations")?;
    let row = Metal::of(down.row, "down row")?;
    let wm = Metal::of(&weights, "topk_weights")?;
    let idm = Metal::of(&ids, "topk_ids")?;
    let p = pipeline(&am.dev, Family::Moe, kernel_name)?;
    let (y, o) = out(&am.dev, (num_tokens, n), DType::F32, "moe-down-output")?;
    let grid = [n.div_ceil(rows), 1, num_tokens];
    launch(&am.dev, &p, kernel_name, grid, [serial.unwrap_or(topk * 32), 1, 1], |e| {
        set_params!(e, (&row, &am, &wm, &o, &idm, topk as i64, k as i64, n as i64, num_experts as i64));
        down.declare(e);
        if serial.is_none() {
            e.set_threadgroup_memory_length(0, topk * rows * 32 * std::mem::size_of::<f32>());
            e.set_threadgroup_memory_length(1, topk * std::mem::size_of::<i32>());
        }
        Ok(())
    })?;
    Ok(y)
}

/// The routed matmul's tiles: 64 x 64 tensor-op over half inputs where candle allows, else 64 x 32 in f32.
struct MmGeometry {
    nax: bool,
    token_tile: usize,
    tg_bytes: usize,
}

impl MmGeometry {
    fn new(dev: &MetalDevice, k: usize, num_tokens: usize) -> Self {
        let nax = candle_metal_kernels::quantized_matmul_mm_nax(dev.device(), num_tokens, k);
        if nax {
            Self { nax, token_tile: 64, tg_bytes: candle_metal_kernels::NAX_MM_TG_BYTES }
        } else {
            // Tile staging, then the tile's 32 `ushort2` row ids.
            Self { nax, token_tile: 32, tg_bytes: 8192 + 32 * std::mem::size_of::<u16>() * 2 }
        }
    }

    fn input(&self, x: Tensor) -> Result<Tensor> {
        if self.nax {
            x.to_dtype(DType::F16)
        } else {
            Ok(x)
        }
    }
}

/// The input's slot count (1: a row per token, broadcast across the slots) and the contiguous input.
fn routed_input(x: &Tensor, num_tokens: usize, topk: usize, k: usize) -> Result<(usize, Tensor)> {
    match x.dims() {
        &[t, kk] if t == num_tokens && kk == k => Ok((1, x.contiguous()?)),
        &[t, mid, kk] if t == num_tokens && mid == topk && kk == k => Ok((topk, x.contiguous()?)),
        d => candle_core::bail!("routed input must be ({num_tokens}, {k}) or ({num_tokens}, {topk}, {k}), got {d:?}"),
    }
}

/// `[t,s,:] = w[e]·x[t(,s),:]` for the listed `experts`, a layer's or a wave's; other and unresolved slices unwritten.
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm(
    w: ExpertRow<'_>, dtype: QkkMoeDType, experts: &Tensor, num_experts: usize, n: usize, k: usize, x: &Tensor,
    topk_ids: &Tensor, out: Option<&Tensor>,
) -> Result<Tensor> {
    refuse_if!(!k.is_multiple_of(dtype.block_elems()), "gemm: K ({k}) does not divide into {dtype:?} blocks");
    check_row(w, num_experts, "gemm")?;
    let (num_tokens, topk) = topk_shape(topk_ids, "gemm")?;
    let map0_kernel_name = map0_kernel_for_topk(topk).ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "gemm requires topk in {{8, 10, 16}} (got {topk}); instantiate \
             mistralrs_moe_mat_mm_id_map0_ne20_<N> in moe_fused.metal"
        ))
    })?;
    refuse_if!(
        experts.dtype() != DType::U32 || experts.dims().len() != 1,
        "gemm: the expert list must be a U32 vector"
    );
    let columns = experts.elem_count();
    refuse_if!(columns == 0 || columns > 1024, "gemm: {columns} experts in the list; the routing table takes 1..=1024");
    let (input_topk_dim, x) = routed_input(x, num_tokens, topk, k)?;
    refuse_if!(x.dtype() != DType::F32, "gemm requires F32 input, got {:?}", x.dtype());
    let geom = MmGeometry::new(Metal::of(&x, "input")?.dev, k, num_tokens);
    let x = geom.input(x)?;
    let ids = topk_ids.contiguous()?;
    let experts = experts.contiguous()?;

    let xm = Metal::of(&x, "input")?;
    let dev = &xm.dev;
    let row = Metal::of(w.row, "gemm row")?;
    let idm = Metal::of(&ids, "topk_ids")?;
    let list = Metal::of(&experts, "expert list")?;

    // Pass 1: the routing table, one thread per listed expert.
    let map0 = pipeline(dev, Family::Moe, map0_kernel_name)?;
    let htpe = dev.new_buffer(columns, DType::U32, "moe-gemm-htpe")?;
    let hids = dev.new_buffer(columns * num_tokens, DType::U32, "moe-gemm-hids")?;
    let e = dev.command_encoder()?;
    let e: &ComputeCommandEncoder = &e;
    e.set_label("moe-gemm-map0");
    e.set_compute_pipeline_state(&map0);
    set_params!(e, (&idm, &*htpe, &*hids, &list, num_tokens as i64, (topk * std::mem::size_of::<u32>()) as u64));
    e.set_threadgroup_memory_length(0, columns * topk * std::mem::size_of::<u16>());
    e.dispatch_thread_groups(size([1, 1, 1]), size([columns, 1, 1]));

    // Pass 2: the tiled matmul over the table.
    let mm = candle_pipeline(dev, dtype.format().mm[usize::from(geom.nax)])?;
    let output = routed_output(out, dev, (num_tokens, topk, n), "moe-gemm-output")?;
    let om = Metal::of(&output, "gemm output")?;
    e.set_label("moe-gemm");
    e.set_compute_pipeline_state(&mm);
    let es = x.dtype().size_in_bytes();
    let bytes_per_row = dtype.row_bytes(k);
    set_params!(
        e,
        (
            &row,
            &xm,
            &om,
            &*htpe,
            &*hids,
            &list,
            topk as i64,                      // nei0
            num_tokens as i64,                // nei1
            k as i64,                         // ne00
            columns as i64,                   // ne02
            bytes_per_row as u64,             // nb01
            (n * bytes_per_row) as u64,       // nb02
            input_topk_dim as i64,            // ne11
            num_tokens as i64,                // ne12
            1i64,                             // ne13
            es as u64,                        // nb10
            (k * es) as u64,                  // nb11
            (input_topk_dim * k * es) as u64, // nb12
            n as i64,                         // ne0
            topk as i64,                      // ne1
            (n * 4) as u64                    // nb1
        )
    );
    e.set_threadgroup_memory_length(0, geom.tg_bytes);
    w.declare(e);
    // x token tiles (one past its expert's match count returns at once), y n / 64, z the listed experts.
    e.dispatch_thread_groups(size([num_tokens.div_ceil(geom.token_tile), n.div_ceil(64), columns]), size([128, 1, 1]));
    Ok(output)
}

/// The candle fork's kernel for formats with no fused entry: `[t,s,:] = w[e]·x[t(,s),:]`, unresolved slices unwritten.
#[allow(clippy::too_many_arguments)]
pub fn moe_matvec(
    w: ExpertRow<'_>, dtype: QkkMoeDType, num_experts: usize, n: usize, k: usize, x: &Tensor, topk_ids: &Tensor,
    out: Option<&Tensor>,
) -> Result<Tensor> {
    refuse_if!(!k.is_multiple_of(dtype.block_elems()), "matvec: K ({k}) does not divide into {dtype:?} blocks");
    let (kernel_name, (nth0, nth1, align), tg_mem) = dtype.format().mv;
    refuse_if!(!n.is_multiple_of(align), "matvec: N ({n}) must be divisible by {align}");
    check_row(w, num_experts, "matvec")?;
    let (num_tokens, topk) = topk_shape(topk_ids, "matvec")?;
    let (input_topk_dim, x) = routed_input(x, num_tokens, topk, k)?;
    refuse_if!(x.dtype() != DType::F32, "matvec requires F32 input, got {:?}", x.dtype());
    let ids = topk_ids.contiguous()?;

    let xm = Metal::of(&x, "input")?;
    let row = Metal::of(w.row, "matvec row")?;
    let idm = Metal::of(&ids, "topk_ids")?;
    let p = candle_pipeline(&xm.dev, kernel_name)?;
    let output = routed_output(out, &xm.dev, (num_tokens, topk, n), "moe-matvec-output")?;
    let om = Metal::of(&output, "matvec output")?;
    let bytes_per_row = dtype.row_bytes(k);
    launch(&xm.dev, &p, kernel_name, [n.div_ceil(align), 1, num_tokens * topk], [nth0, nth1, 1], |e| {
        // The candle fork's generic parameter block (`kernel_mul_mv_id`'s).
        set_params!(
            e,
            (
                &row,
                &xm,
                &om,
                &idm,
                topk as i64,                                // nei0
                num_tokens as i64,                          // nei1
                (topk * std::mem::size_of::<u32>()) as u64, // nbi1
                k as i64,                                   // ne00
                n as i64,                                   // ne01
                num_experts as i64,                         // ne02
                0u64,                                       // nb00
                bytes_per_row as u64,                       // nb01
                (n * bytes_per_row) as u64,                 // nb02
                k as i64,                                   // ne10
                input_topk_dim as i64,                      // ne11
                num_tokens as i64,                          // ne12
                1i64,                                       // ne13
                4u64,                                       // nb10
                (k * 4) as u64,                             // nb11
                (input_topk_dim * k * 4) as u64,            // nb12
                n as i64,                                   // ne0
                topk as i64,                                // ne1
                (n * 4) as u64                              // nb1
            )
        );
        w.declare(e);
        // Only the i-quant matvecs stage a codebook in threadgroup memory.
        if tg_mem > 0 {
            e.set_threadgroup_memory_length(0, tg_mem);
        }
        Ok(())
    })?;
    Ok(output)
}

/// `[t,h] = Σ_s down_out[t,s,h]·topk_w[t,s]`, skipping an unresolved slot, whose slice the matmul never wrote.
pub fn moe_topk_weighted_sum(
    down_out: &Tensor, topk_w: &Tensor, down: ExpertRow<'_>, topk_ids: &Tensor,
) -> Result<Tensor> {
    let (num_tokens, topk, hidden) = match down_out.dims() {
        &[t, k, h] => (t, k, h),
        d => candle_core::bail!("weighted sum: down_out must be (num_tokens, topk, hidden), got {d:?}"),
    };
    refuse_if!(
        topk_w.dims() != [num_tokens, topk],
        "weighted sum: topk_w shape {:?} != ({num_tokens}, {topk})",
        topk_w.dims()
    );
    refuse_if!(
        topk_ids.dims() != [num_tokens, topk] || topk_ids.dtype() != DType::U32,
        "weighted sum: topk_ids must be U32 ({num_tokens}, {topk})"
    );
    refuse_if!(down_out.dtype() != DType::F32 || topk_w.dtype() != DType::F32, "weighted sum: F32 only");
    // The row's own length is the expert count, so only its dtype is checked.
    let num_experts = down.row.elem_count();
    check_row(down, num_experts, "weighted sum")?;
    let y = down_out.contiguous()?;
    let w = topk_w.contiguous()?;
    let ids = topk_ids.contiguous()?;
    let ym = Metal::of(&y, "down_out")?;
    let wm = Metal::of(&w, "topk_w")?;
    let row = Metal::of(down.row, "down row")?;
    let idm = Metal::of(&ids, "topk_ids")?;
    let p = pipeline(&ym.dev, Family::Moe, "mistralrs_moe_topk_weighted_sum_f32")?;
    let (sum, o) = out(&ym.dev, (num_tokens, hidden), DType::F32, "moe-topk-weighted-sum-output")?;
    launch(&ym.dev, &p, "moe-topk-weighted-sum", [hidden.div_ceil(32), num_tokens, 1], [32, 1, 1], |e| {
        set_params!(e, (&ym, &wm, &row, &idm, &o, topk as i32, hidden as i32, num_experts as i32));
        Ok(())
    })?;
    Ok(sum)
}

#[cfg(test)]
mod tests;
