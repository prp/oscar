//! Gated-DeltaNet kernels; Qwen3.5's full-attention glue (`attn_pack`, `attn_output_gate`) shares their shader file.

use candle_core::{DType, Result, Tensor};
use candle_metal_kernels::metal::ComputeCommandEncoder;

use crate::loader::{launch, out, pipeline, set_params, Family, Metal};

/// The delta-rule kernels' key width, a state row in registers; the models' load refuses another.
pub const KEY_WIDTH: usize = 128;

/// The emits a token loop takes: `(at, into)`, the state after the chunk's `at`-th token (1 ≤ at ≤ S, ascending)
/// written to `into`, laid out as the state; the row a launch cut there writes, bit for bit.
pub type Emits<'a> = &'a [(usize, Tensor)];
const MAX_EMITS: usize = 4;

/// Contiguous copies of `named`, each refused unless F32.
fn f32_contiguous<const N: usize>(what: &str, named: [(&str, &Tensor); N]) -> Result<[Tensor; N]> {
    let each = |(name, t): &(&str, &Tensor)| {
        refuse_if!(t.dtype() != DType::F32, "{what}: {name} must be F32, got {:?}", t.dtype());
        t.contiguous()
    };
    let all: Vec<Tensor> = named.iter().map(each).collect::<Result<_>>()?;
    Ok(all.try_into().ok().expect("one tensor per name"))
}

/// The lane-strided kernels take a width that is a multiple of 32 up to 512.
fn width_fits(what: &str, name: &str, w: usize) -> Result<()> {
    refuse_if!(w % 32 != 0 || w / 32 > 16, "{what}: {name}={w} must be a multiple of 32 up to 512");
    Ok(())
}

/// Contiguous F32: q, k `[BH,S,K]`; v `[BH,S,V]`; g `[BH,S]`, or `[BH,S,K]` for KDA; beta `[BH,S]`.
struct Inputs {
    t: [Tensor; 5],
    bh: usize,
    s: usize,
    vd: usize,
}

impl Inputs {
    fn of(what: &str, [q, k, v, g, beta]: [&Tensor; 5]) -> Result<Self> {
        let t = f32_contiguous(what, [("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)])?;
        let ((bh, s, kd), vd) = (t[0].dims3()?, t[2].dim(2)?);
        refuse_if!(kd != KEY_WIDTH, "{what}: the delta rule's kernels take keys of {KEY_WIDTH}, got {kd}");
        // A partial last tile's idle threads return before loading their share of the key rows the others read.
        refuse_if!(vd % 64 != 0, "{what}: a value width of {vd} is not a multiple of the delta rule's 64-column tile");
        Ok(Self { t, bh, s, vd })
    }

    /// `emits` bind slots 9 to 13 where the kernel takes them, the state filling an unasked slot; `tail` binds the
    /// kernel's arguments from slot 8; the output is unfilled, as the grid writes every cell.
    fn launch(
        &self, name: &str, (out_label, label): (&'static str, &str), state: &Tensor,
        (grid, tg): ([usize; 3], [usize; 3]), emits: Option<Emits<'_>>,
        tail: impl FnOnce(&ComputeCommandEncoder) -> Result<()>,
    ) -> Result<Tensor> {
        let [q, k, v, g, beta] = &self.t;
        let (qm, km, vm) = (Metal::of(q, "q")?, Metal::of(k, "k")?, Metal::of(v, "v")?);
        let (gm, bm, sm) = (Metal::of(g, "g")?, Metal::of(beta, "beta")?, Metal::of(state, "state")?);
        let p = pipeline(qm.dev, Family::Gdn, name)?;
        let (y, o) = out(qm.dev, (self.bh, self.s, self.vd), DType::F32, out_label)?;
        launch(qm.dev, &p, label, grid, tg, |e| {
            set_params!(e, (&qm, &km, &vm, &gm, &bm, &sm, &o, self.s as i32));
            if let Some(emits) = emits {
                refuse_if!(emits.len() > MAX_EMITS, "{label}: {} emits, the kernels take {MAX_EMITS}", emits.len());
                let mut at = [-1i32; MAX_EMITS];
                for (i, (pos, into)) in emits.iter().enumerate() {
                    refuse_if!(
                        *pos == 0 || *pos > self.s || i > 0 && *pos <= emits[i - 1].0,
                        "{label}: an emit after token {pos} of {} is not inside the chunk in order",
                        self.s
                    );
                    refuse_if!(
                        into.dtype() != DType::F32 || !into.is_contiguous() || into.dims() != state.dims(),
                        "{label}: an emit's state must be contiguous F32 {:?}, got {:?} {:?}",
                        state.dims(),
                        into.dtype(),
                        into.dims()
                    );
                    let m = Metal::of(into, "emit")?;
                    e.set_buffer(9 + i, Some(&m.buf), m.off);
                    at[i] = *pos as i32;
                }
                for i in emits.len()..MAX_EMITS {
                    e.set_buffer(9 + i, Some(&sm.buf), sm.off);
                }
                e.set_bytes(13, &at);
            }
            tail(e)
        })?;
        Ok(y)
    }
}

/// State `[BH,K,V]` in place, output `[BH,S,V]`; `norm_gate` holds [`norm_gate`]'s z, weight, eps and sigmoid flag.
/// It runs fused at one token a row, V = 128 and no emit, as the bool returned says; else the caller applies it.
pub fn gated_delta_rule_recurrence(
    q: &Tensor, k: &Tensor, v: &Tensor, g: &Tensor, beta: &Tensor, state: &mut Tensor,
    norm_gate: Option<(&Tensor, &Tensor, f32, bool)>, emits: Emits<'_>,
) -> Result<(Tensor, bool)> {
    let x = Inputs::of("gated_delta_rule_recurrence", [q, k, v, g, beta])?;
    let vd = x.vd;
    let fused = match norm_gate.filter(|_| x.s == 1 && vd == 128 && emits.is_empty()) {
        Some((z, w, eps, sigmoid)) => Some((z, w.contiguous()?, eps, sigmoid, z_layout(z, x.bh, vd)?)),
        None => None,
    };
    let (name, bv) =
        if fused.is_some() { ("gated_delta_rule_norm_gate_128_128", 128) } else { ("gated_delta_rule_128_64", 64) };
    let geometry = ([vd.div_ceil(bv), x.bh, 1], [bv, 1, 1]);
    let emits = fused.is_none().then_some(emits);
    let y = x.launch(name, ("gdn-recurrence-output", "gdn-recurrence"), state, geometry, emits, |e| {
        e.set_bytes(8, &(vd as i32));
        if let Some((z, w, eps, sigmoid, layout)) = &fused {
            let (zm, wm) = (Metal::of(z, "z")?, Metal::of(w, "weight")?);
            e.set_buffer(9, Some(&zm.buf), zm.off);
            e.set_buffer(10, Some(&wm.buf), wm.off);
            e.set_bytes(11, eps);
            e.set_bytes(12, &i32::from(*sigmoid));
            for (i, v) in layout.iter().enumerate() {
                e.set_bytes(13 + i, &(*v as i32));
            }
        }
        Ok(())
    })?;
    Ok((y, fused.is_some()))
}

/// [`gated_delta_rule_recurrence`] with a decay per K channel: g `[BH,S,K]`, in log space.
pub fn kda_delta_rule_recurrence(
    q: &Tensor, k: &Tensor, v: &Tensor, g: &Tensor, beta: &Tensor, state: &mut Tensor, emits: Emits<'_>,
) -> Result<Tensor> {
    let x = Inputs::of("kda_delta_rule_recurrence", [q, k, v, g, beta])?;
    let (bh, s, vd) = (x.bh, x.s, x.vd);
    refuse_if!(
        x.t[3].dims3()? != (bh, s, KEY_WIDTH),
        "kda_delta_rule_recurrence: g must be [BH, S, K] = ({bh}, {s}, {KEY_WIDTH}), got {:?}",
        x.t[3].dims()
    );
    let geometry = ([vd.div_ceil(64), bh, 1], [64, 1, 1]);
    x.launch("kda_delta_rule_128_64", ("kda-recurrence-output", "kda-recurrence"), state, geometry, Some(emits), |e| {
        e.set_bytes(8, &(vd as i32));
        Ok(())
    })
}

/// The prefill form of [`gated_delta_rule_recurrence`]: one token-sequential launch.
pub fn chunked_gated_delta_rule_recurrence(
    q: &Tensor, k: &Tensor, v: &Tensor, g: &Tensor, beta: &Tensor, state: &mut Tensor, emits: Emits<'_>,
) -> Result<Tensor> {
    let x = Inputs::of("chunked_gated_delta_rule_recurrence", [q, k, v, g, beta])?;
    let vd = x.vd;
    // V / 16 columns per threadgroup.
    let geometry = ([vd.div_ceil(16), x.bh, 1], [32, 2, 1]);
    let labels = ("gdn-chunked-recurrence-output", "gdn-chunked-recurrence-seq");
    x.launch("gated_delta_rule_seq_v4_128", labels, state, geometry, Some(emits), |e| {
        e.set_bytes(8, &(vd as i32));
        Ok(())
    })
}

/// `x` `[B, conv_dim, S]`: `(output, new_conv_state)`, the state `x`'s last `kernel_size` positions, zero-padded.
pub fn causal_conv1d(x: &Tensor, weight: &Tensor, kernel_size: usize) -> Result<(Tensor, Tensor)> {
    let [x, weight] = f32_contiguous("causal_conv1d", [("x", x), ("weight", weight)])?;
    let (xm, wm) = (Metal::of(&x, "x")?, Metal::of(&weight, "weight")?);
    let dev = xm.dev;
    let (batch_size, conv_dim, seq_len) = x.dims3()?;

    // Outputs of one (b, ch) per thread, the kernel's `POS_PER_THREAD`.
    const POS_PER_THREAD: usize = 8;
    let p = pipeline(dev, Family::Gdn, "causal_conv1d_full_v2_kernel")?;
    let (output, o) = out(dev, (batch_size, conv_dim, seq_len), DType::F32, "gdn-conv1d-full-output")?;
    let grid = [conv_dim.div_ceil(256), seq_len.div_ceil(POS_PER_THREAD), batch_size];
    let ints = [batch_size, conv_dim, seq_len, kernel_size].map(|v| v as i32);
    launch(dev, &p, "gdn-conv1d-full", grid, [256, 1, 1], |e| {
        set_params!(e, (&xm, &wm, &o, ints[0], ints[1], ints[2], ints[3]));
        Ok(())
    })?;

    let p = pipeline(dev, Family::Gdn, "save_conv_state_kernel")?;
    let (new_conv_state, cs) = out(dev, (batch_size, conv_dim, kernel_size), DType::F32, "gdn-conv1d-new-conv-state")?;
    launch(dev, &p, "gdn-conv1d-save-state", [conv_dim.div_ceil(256), batch_size, 1], [256, 1, 1], |e| {
        set_params!(e, (&xm, &cs, ints[0], ints[1], ints[2], ints[3]));
        Ok(())
    })?;

    Ok((output, new_conv_state))
}

/// The recurrence's F32 inputs in one launch: the conv (decode's over `mixed_qkvz`, state in place, where `conv` is
/// given; else `src` is prefill's conv output), q and k's L2 norm, the v pack and the gating.
#[allow(clippy::too_many_arguments)]
pub fn prep(
    src: &Tensor, conv: Option<(&Tensor, &Tensor)>, mixed_ba: &Tensor, a_log: &Tensor, dt_bias: &Tensor, l2_w: &Tensor,
    num_k_heads: usize, head_k_dim: usize, head_v_dim: usize, v_per_group: usize, kernel_size: usize, scale: f32,
    eps: f32,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let named = [("src", src), ("mixed_ba", mixed_ba), ("a_log", a_log), ("dt_bias", dt_bias), ("l2_w", l2_w)];
    let [src, mixed_ba, a_log, dt_bias, l2_w] = f32_contiguous("gdn prep", named)?;
    let (hk, dk, dv, vpg) = (num_k_heads, head_k_dim, head_v_dim, v_per_group);
    let hv = hk * vpg;
    let conv_dim = 2 * hk * dk + hk * vpg * dv;
    let ba_dims = mixed_ba.dims();
    let (b_size, s_size) = match ba_dims {
        [b, s, w] if *w == hk * 2 * vpg => (*b, *s),
        _ => candle_core::bail!(
            "gdn prep: expected mixed_ba (B, S, Hk * 2 * vpg) with Hk={hk} vpg={vpg}, got {ba_dims:?}"
        ),
    };
    width_fits("gdn prep", "Dk", dk)?;
    width_fits("gdn prep", "Dv", dv)?;
    refuse_if!(vpg > 32, "gdn prep: vpg={vpg} must be at most 32");
    refuse_if!(
        a_log.elem_count() != hv || dt_bias.elem_count() != hv || l2_w.elem_count() != dk,
        "gdn prep: a_log/dt_bias need Hv={hv} elements, l2_w Dk={dk}"
    );
    let src_elems = match conv {
        Some(_) => {
            refuse_if!(s_size != 1, "gdn prep: the conv update takes one token per row, got S={s_size}");
            b_size * hk * (2 * dk + 2 * vpg * dv)
        }
        None => b_size * s_size * conv_dim,
    };
    refuse_if!(src.elem_count() != src_elems, "gdn prep: src has {} elements, expected {src_elems}", src.elem_count());
    if let Some((w, state)) = conv {
        refuse_if!(
            w.dtype() != DType::F32 || state.dtype() != DType::F32,
            "gdn prep: conv weight and state must be F32"
        );
        refuse_if!(
            w.elem_count() != conv_dim * kernel_size || state.elem_count() != b_size * conv_dim * kernel_size,
            "gdn prep: conv weight (conv_dim, K) and state (B, conv_dim, K) sizes disagree with K={kernel_size}"
        );
        refuse_if!(!w.is_contiguous() || !state.is_contiguous(), "gdn prep: conv weight and state must be contiguous");
    }

    let srcm = Metal::of(&src, "src")?;
    let dev = srcm.dev;
    let p = pipeline(dev, Family::Gdn, if conv.is_some() { "gdn_prep_conv" } else { "gdn_prep" })?;

    let (q_bh, qm) = out(dev, (b_size * hv, s_size, dk), DType::F32, "gdn-prep-q")?;
    let (k_bh, km) = out(dev, (b_size * hv, s_size, dk), DType::F32, "gdn-prep-k")?;
    let (v_bh, vm) = out(dev, (b_size * hv, s_size, dv), DType::F32, "gdn-prep-v")?;
    let (beta, betam) = out(dev, (b_size * hv, s_size), DType::F32, "gdn-prep-beta")?;
    let (g, gm) = out(dev, (b_size * hv, s_size), DType::F32, "gdn-prep-g")?;

    // Without a conv the weight and state slots are unread; `src` fills them.
    let (wm, csm) = match conv {
        Some((w, state)) => (Metal::of(w, "conv weight")?, Metal::of(state, "conv state")?),
        None => (Metal::of(&src, "src")?, Metal::of(&src, "src")?),
    };
    let (bam, alogm, dtbm, lwm) = (
        Metal::of(&mixed_ba, "mixed_ba")?,
        Metal::of(&a_log, "a_log")?,
        Metal::of(&dt_bias, "dt_bias")?,
        Metal::of(&l2_w, "l2_w")?,
    );
    let ints = [b_size, s_size, hk, dk, dv, vpg, kernel_size].map(|v| v as i32);
    launch(dev, &p, "gdn-prep", [s_size, hk, b_size], [32, 1, 1], |e| {
        set_params!(
            e,
            (
                &srcm, &wm, &csm, &bam, &alogm, &dtbm, &lwm, &qm, &km, &vm, &betam, &gm, ints[0], ints[1], ints[2],
                ints[3], ints[4], ints[5], ints[6], scale, eps
            )
        );
        Ok(())
    })?;
    Ok((q_bh, k_bh, v_bh, beta, g))
}

/// `[Hv, z_stride_bt, z_stride_g, z_vpg]` of a `(B, S, G, k·V)` z with unit last stride holding `rows` rows of V.
fn z_layout(z: &Tensor, rows: usize, v_dim: usize) -> Result<[usize; 4]> {
    let (zd, zs) = (z.dims(), z.stride());
    refuse_if!(zd.len() != 4 || zd[3] % v_dim != 0 || zs[3] != 1 || zs[0] != zd[1] * zs[1]
        || zd.iter().product::<usize>() != rows * v_dim,
        "norm gate: z must be (B, S, G, k*V) with unit last stride for {rows} rows of {v_dim}, got {zd:?} strides {zs:?}");
    Ok([zd[2] * zd[3] / v_dim, zs[1], zs[2], zd[3] / v_dim])
}

/// y's RMS norm over V, weighted, times its gate z or sigmoid(z): y `(B, S, Hv, V)`, z a `(B, S, G, k·V)` view read
/// in place, F32, into y's layout.
pub fn norm_gate(y: &Tensor, z: &Tensor, weight: &Tensor, eps: f32, sigmoid_gate: bool) -> Result<Tensor> {
    let [y, weight] = f32_contiguous("norm_gate", [("y", y), ("weight", weight)])?;
    refuse_if!(z.dtype() != DType::F32, "norm_gate: z must be F32, got {:?}", z.dtype());
    let dims = y.dims();
    refuse_if!(dims.len() != 4, "norm_gate: expected rank-4 y (B, S, Hv, V), got {dims:?}");
    let (b_size, s_size, hv, v_dim) = (dims[0], dims[1], dims[2], dims[3]);
    // z is a view of the qkvz projection (G = Hk, W = vpg*V) or a contiguous (B, S, Hv, V).
    let [z_hv, z_stride_bt, z_stride_g, z_vpg] = z_layout(z, b_size * s_size * hv, v_dim)?;
    refuse_if!(
        z_hv != hv || z.dims()[..2] != [b_size, s_size],
        "norm_gate: z must be (B, S, G, k*V) for y {dims:?}, got {:?}",
        z.dims()
    );
    width_fits("norm_gate", "V", v_dim)?;
    refuse_if!(
        weight.elem_count() != v_dim,
        "norm_gate: weight must have V={v_dim} elements, got {}",
        weight.elem_count()
    );

    let (ym, zm, wm) = (Metal::of(&y, "y")?, Metal::of(z, "z")?, Metal::of(&weight, "weight")?);
    let dev = ym.dev;
    let p = pipeline(dev, Family::Gdn, "gdn_norm_gate_kernel")?;
    let (y_out, o) = out(dev, (b_size, s_size, hv, v_dim), DType::F32, "gdn-norm-gate-out")?;
    let ints = [b_size, s_size, hv, v_dim].map(|v| v as i32);
    let zl = [z_stride_bt, z_stride_g, z_vpg].map(|v| v as i32);
    launch(dev, &p, "gdn-norm-gate", [s_size, hv, b_size], [32, 1, 1], |e| {
        set_params!(
            e,
            (&ym, &zm, &wm, &o, ints[0], ints[1], ints[2], ints[3], eps, i32::from(sigmoid_gate), zl[0], zl[1], zl[2])
        );
        Ok(())
    })?;
    Ok(y_out)
}

/// `y * sigmoid(gate)` into F32 `(B, S, H * D)`, y `(B, H, S, D)` BF16, the gate at `off` of each head's `head_stride`.
pub fn attn_output_gate(y: &Tensor, q_gate: &Tensor, head_stride: usize, off: usize) -> Result<Tensor> {
    let y = y.contiguous()?;
    refuse_if!(y.dtype() != DType::BF16, "attn_output_gate: y must be BF16, got {:?}", y.dtype());
    let [q_gate] = f32_contiguous("attn_output_gate", [("q_gate", q_gate)])?;
    let (b_size, h, s_size, d) = y.dims4()?;
    refuse_if!(
        q_gate.elem_count() != b_size * s_size * h * head_stride || off + d > head_stride,
        "attn_output_gate: gate {:?} is not (B={b_size}, S={s_size}, H={h} * {head_stride}) with D={d} at {off}",
        q_gate.dims()
    );
    width_fits("attn_output_gate", "D", d)?;
    let (ym, gm) = (Metal::of(&y, "y")?, Metal::of(&q_gate, "q_gate")?);
    let dev = ym.dev;
    let p = pipeline(dev, Family::Gdn, "attn_output_gate_kernel")?;
    let (y_out, o) = out(dev, (b_size, s_size, h * d), DType::F32, "gdn-attn-output-gate-out")?;
    let params = [b_size, h, s_size, d, h * head_stride, head_stride, off].map(|x| x as i32);
    launch(dev, &p, "gdn-attn-output-gate", [s_size, h, b_size], [32, 1, 1], |e| {
        set_params!(e, (&ym, &gm, &o, params[0], params[1], params[2], params[3], params[4], params[5], params[6]));
        Ok(())
    })?;
    Ok(y_out)
}

/// `mixed_qkvz` `(B,S,Hk*(2*Dk+2*vpg*Dv))` into the conv input `(B,S,2*Hk*Dk+Hv*Dv)`; the norm gate reads z in place.
pub fn qkvz_split(
    mixed_qkvz: &Tensor, num_k_heads: usize, head_k_dim: usize, head_v_dim: usize, v_per_group: usize,
) -> Result<Tensor> {
    let [mixed_qkvz] = f32_contiguous("qkvz_split", [("mixed_qkvz", mixed_qkvz)])?;
    let dims = mixed_qkvz.dims();
    refuse_if!(dims.len() != 3, "qkvz_split: expected rank-3 mixed_qkvz (B, S, Hk*group), got {dims:?}");
    let (b_size, s_size) = (dims[0], dims[1]);
    let group_size_qkvz = 2 * head_k_dim + 2 * v_per_group * head_v_dim;
    refuse_if!(
        dims[2] != num_k_heads * group_size_qkvz,
        "qkvz_split: last dim {} != Hk*group ({} * {})",
        dims[2],
        num_k_heads,
        group_size_qkvz
    );

    width_fits("qkvz_split", "head_k_dim", head_k_dim)?;
    let v_chunk = v_per_group * head_v_dim;
    width_fits("qkvz_split", "v_per_group*head_v_dim", v_chunk)?;

    let qkvzm = Metal::of(&mixed_qkvz, "mixed_qkvz")?;
    let dev = qkvzm.dev;
    let p = pipeline(dev, Family::Gdn, "gdn_qkvz_split_kernel")?;

    let num_v_heads = num_k_heads * v_per_group;
    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let conv_dim = 2 * key_dim + value_dim;

    let (mixed_qkv, qkvm) = out(dev, (b_size, s_size, conv_dim), DType::F32, "gdn-qkvz-split-mixed-qkv")?;
    let ints = [b_size, s_size, num_k_heads, head_k_dim, head_v_dim, v_per_group].map(|v| v as i32);
    launch(dev, &p, "gdn-qkvz-split", [s_size, num_k_heads, b_size], [32, 1, 1], |e| {
        set_params!(e, (&qkvzm, &qkvm, ints[0], ints[1], ints[2], ints[3], ints[4], ints[5]));
        Ok(())
    })?;
    Ok(mixed_qkv)
}

/// q's and k's RMS norms and rope over the first `rot` channels, from `q_gate` `(B,S,H*2D)`, each head `[q | gate]`,
/// and k, v `(B,S,Hkv*D)`, into BF16 q `(B,H,S,D)`, k and v `(B,Hkv,S,D)`, byte-equal to candle's chain.
#[allow(clippy::too_many_arguments)]
pub fn attn_pack(
    q_gate: &Tensor, k: &Tensor, v: &Tensor, qw: &Tensor, kw: &Tensor, eps: f32, cos: &Tensor, sin: &Tensor,
    rot: usize, heads: usize, kv_heads: usize, d: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let named = [("q_gate", q_gate), ("k", k), ("v", v), ("qw", qw), ("kw", kw), ("cos", cos), ("sin", sin)];
    let [q_gate, k, v, qw, kw, cos, sin] = f32_contiguous("attn_pack", named)?;
    let (b, s, _) = q_gate.dims3()?;
    refuse_if!(
        q_gate.elem_count() != b * s * heads * 2 * d
            || k.elem_count() != b * s * kv_heads * d
            || v.elem_count() != b * s * kv_heads * d
            || qw.elem_count() != d
            || kw.elem_count() != d,
        "attn_pack: shapes q_gate {:?} k {:?} v {:?} qw {:?} kw {:?} disagree with H={heads} Hkv={kv_heads} D={d}",
        q_gate.dims(),
        k.dims(),
        v.dims(),
        qw.dims(),
        kw.dims()
    );
    refuse_if!(
        rot == 0 || rot % 2 != 0 || rot > d || cos.elem_count() != s * rot / 2 || sin.elem_count() != s * rot / 2,
        "attn_pack: rot {rot} over D {d} needs cos/sin of S*rot/2 = {} elements, got {:?} {:?}",
        s * rot / 2,
        cos.dims(),
        sin.dims()
    );
    let (qg, km, vm) = (Metal::of(&q_gate, "q_gate")?, Metal::of(&k, "k")?, Metal::of(&v, "v")?);
    let dev = qg.dev;
    let p = pipeline(dev, Family::Gdn, "attn_pack_kernel")?;
    // candle's rmsnorm block width, so the norm's tree is candle's: a power of two under any GPU's pipeline limit.
    let block_dim = 1usize << p.max_total_threads_per_threadgroup().min((d / 2).next_power_of_two()).ilog2();
    let (q_out, qo) = out(dev, (b, heads, s, d), DType::BF16, "attn-pack-q")?;
    let (k_out, ko) = out(dev, (b, kv_heads, s, d), DType::BF16, "attn-pack-k")?;
    let (v_out, vo) = out(dev, (b, kv_heads, s, d), DType::BF16, "attn-pack-v")?;
    let (qwm, kwm, cm, sm) =
        (Metal::of(&qw, "qw")?, Metal::of(&kw, "kw")?, Metal::of(&cos, "cos")?, Metal::of(&sin, "sin")?);
    let (q_rows, kv_rows) = (b * heads * s, b * kv_heads * s);
    let ints = [s, heads, kv_heads, d, rot].map(|x| x as i32);
    launch(dev, &p, "attn-pack", [q_rows + 2 * kv_rows, 1, 1], [block_dim, 1, 1], |e| {
        set_params!(
            e,
            (
                &qg,
                &km,
                &vm,
                &qwm,
                &kwm,
                &cm,
                &sm,
                &qo,
                &ko,
                &vo,
                ints[0],
                ints[1],
                ints[2],
                ints[3],
                ints[4],
                eps,
                q_rows as i32,
                kv_rows as i32
            )
        );
        e.set_threadgroup_memory_length(0, block_dim * std::mem::size_of::<f32>());
        Ok(())
    })?;
    Ok((q_out, k_out, v_out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_bitwise, fill, host};
    use candle_core::D;

    /// Each token loop: a state emitted after token `at` is the state a launch cut there writes, bit for bit, and
    /// the emitting run's output and final state are the plain run's; 2,048 columns, so a moved order would show.
    #[test]
    fn an_emitted_state_is_the_cut_launchs_state_to_the_bit() -> Result<()> {
        let dev = oscar_offload::pools::test_device()?;
        let (bh, s, k, vd) = (8usize, 203usize, KEY_WIDTH, 256usize);
        // Unit keys and queries, as the models' prep writes them: the delta rule diverges on long ones.
        let unit_rows = |seed: usize| -> Vec<f32> {
            let mut rows = fill(bh * s * k, seed);
            for row in rows.chunks_mut(k) {
                let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                row.iter_mut().for_each(|x| *x /= norm);
            }
            rows
        };
        let (q, k_t) =
            (Tensor::from_vec(unit_rows(1), (bh, s, k), &dev)?, Tensor::from_vec(unit_rows(2), (bh, s, k), &dev)?);
        let v = Tensor::from_vec(fill(bh * s * vd, 3), (bh, s, vd), &dev)?;
        let decays = |n: usize| -> Vec<f32> { fill(n, 4).iter().map(|x| -x.abs() * 0.3).collect() };
        let g_head = Tensor::from_vec(decays(bh * s), (bh, s), &dev)?;
        let g_chan = Tensor::from_vec(decays(bh * s * k), (bh, s, k), &dev)?;
        let beta = Tensor::from_vec(fill(bh * s, 5).iter().map(|x| x.abs() / 3.0).collect(), (bh, s), &dev)?;
        let state0 = Tensor::from_vec(fill(bh * k * vd, 7).iter().map(|x| x * 0.05).collect(), (bh, k, vd), &dev)?;
        type Run<'a> =
            &'a dyn Fn(&Tensor, &Tensor, &Tensor, &Tensor, &Tensor, &mut Tensor, Emits<'_>) -> Result<Tensor>;
        let loops: [(&str, &Tensor, Run); 3] = [
            ("seq_v4", &g_head, &|q, k, v, g, b, st, e| chunked_gated_delta_rule_recurrence(q, k, v, g, b, st, e)),
            ("delta_rule", &g_head, &|q, k, v, g, b, st, e| {
                Ok(gated_delta_rule_recurrence(q, k, v, g, b, st, None, e)?.0)
            }),
            ("kda", &g_chan, &|q, k, v, g, b, st, e| kda_delta_rule_recurrence(q, k, v, g, b, st, e)),
        ];
        for (name, g, run) in loops {
            let cut = |len: usize, emits: Emits<'_>| -> Result<(Vec<f32>, Vec<f32>)> {
                let narrow = |t: &Tensor| t.narrow(1, 0, len)?.contiguous();
                // A `copy` shares a Metal buffer; the kernel updates the state in place.
                let mut st = state0.force_contiguous()?;
                let y = run(&narrow(&q)?, &narrow(&k_t)?, &narrow(&v)?, &narrow(g)?, &narrow(&beta)?, &mut st, emits)?;
                Ok((host(&y)?, host(&st)?))
            };
            let ats = [1usize, 37, 64, s];
            let emits: Vec<(usize, Tensor)> =
                ats.iter().map(|&at| Ok((at, Tensor::zeros((bh, k, vd), DType::F32, &dev)?))).collect::<Result<_>>()?;
            let (want_y, want_state) = cut(s, &[])?;
            assert!(want_y.iter().chain(&want_state).all(|x| x.is_finite()), "{name}: the run diverged");
            let (y, state) = cut(s, &emits)?;
            assert_bitwise(&y, &want_y, &format!("{name}: the emitting run's output"));
            assert_bitwise(&state, &want_state, &format!("{name}: the emitting run's final state"));
            for (at, into) in &emits {
                let (_, at_cut) = cut(*at, &[])?;
                assert_bitwise(&host(into)?, &at_cut, &format!("{name}: the state emitted after token {at}"));
            }
            assert!(cut(s, &[(0, emits[0].1.clone())]).is_err() && cut(s, &[(s + 1, emits[0].1.clone())]).is_err());
        }
        Ok(())
    }

    /// Both activations against candle's chain as a host loop; the sigmoid, qwen4exp's, has no other coverage.
    #[test]
    fn norm_gate_matches_a_host_loop_for_both_activations() -> Result<()> {
        let dev = oscar_offload::pools::test_device()?;
        let (b, s, hv, v) = (1usize, 3usize, 4usize, 128usize);
        let n = b * s * hv * v;
        let y: Vec<f32> = (0..n).map(|i| ((i * 37 % 97) as f32 - 48.0) / 23.0).collect();
        let z: Vec<f32> = (0..n).map(|i| ((i * 53 % 89) as f32 - 44.0) / 17.0).collect();
        let w: Vec<f32> = (0..v).map(|i| 0.5 + (i as f32) / (v as f32)).collect();
        let t = |x: &[f32]| Tensor::from_vec(x.to_vec(), (b, s, hv, v), &dev);
        for sigmoid in [false, true] {
            let got = host(&norm_gate(&t(&y)?, &t(&z)?, &Tensor::new(w.as_slice(), &dev)?, 1e-6, sigmoid)?)?;
            for (i, (yr, zr)) in y.chunks(v).zip(z.chunks(v)).enumerate() {
                let rms = (yr.iter().map(|x| x * x).sum::<f32>() / v as f32 + 1e-6).sqrt();
                for j in 0..v {
                    let e = (-zr[j]).exp();
                    let want = yr[j] / rms * w[j] * if sigmoid { 1.0 / (1.0 + e) } else { zr[j] / (1.0 + e) };
                    let (out, tol) = (got[i * v + j], 2e-5 * (1.0 + want.abs()));
                    assert!((out - want).abs() < tol, "sigmoid={sigmoid} row {i} elem {j}: kernel {out} vs {want}");
                }
            }
        }
        Ok(())
    }

    /// The output gate over thousands of rows at a non-power-of-two width.
    #[test]
    fn attn_pack_matches_the_chain_to_the_bit_and_the_output_gate_reads_the_attentions_bf16() -> Result<()> {
        let dev = oscar_offload::pools::test_device()?;
        // The 35B's head shape, and enough rows to tell one summation tree from another.
        let (b, s, h, hkv, d, rot) = (1usize, 1usize, 2048usize, 512usize, 256usize, 64usize);
        let q_gate = Tensor::from_vec(fill(b * s * h * 2 * d, 1), (b, s, h * 2 * d), &dev)?;
        let k = Tensor::from_vec(fill(b * s * hkv * d, 2), (b, s, hkv * d), &dev)?;
        let v = Tensor::from_vec(fill(b * s * hkv * d, 3), (b, s, hkv * d), &dev)?;
        let qw = Tensor::from_vec(fill(d, 4), d, &dev)?;
        let kw = Tensor::from_vec(fill(d, 5), d, &dev)?;
        let cos = Tensor::from_vec(fill(s * rot / 2, 6), (s, rot / 2), &dev)?;
        let sin = Tensor::from_vec(fill(s * rot / 2, 7), (s, rot / 2), &dev)?;
        let eps = 1e-6f32;
        let chain = |x: &Tensor, heads: usize, w: &Tensor| -> Result<Tensor> {
            let x = x.reshape((b, heads, s, d))?;
            let n = candle_nn::ops::rms_norm(&x.flatten(0, 2)?, w, eps)?.reshape((b, heads, s, d))?;
            let rotated = candle_nn::rotary_emb::rope(&n.narrow(D::Minus1, 0, rot)?.contiguous()?, &cos, &sin)?;
            Tensor::cat(&[rotated, n.narrow(D::Minus1, rot, d - rot)?], D::Minus1)?.to_dtype(DType::BF16)
        };
        let q_in = q_gate.reshape((b, s, h, 2 * d))?.narrow(D::Minus1, 0, d)?.contiguous()?;
        let want = [chain(&q_in, h, &qw)?, chain(&k, hkv, &kw)?, v.reshape((b, hkv, s, d))?.to_dtype(DType::BF16)?];
        let got = attn_pack(&q_gate, &k, &v, &qw, &kw, eps, &cos, &sin, rot, h, hkv, d)?;
        for (name, (w, g)) in ["q", "k", "v"].iter().zip(want.iter().zip([&got.0, &got.1, &got.2])) {
            assert_bitwise(&host(&g.to_dtype(DType::F32)?)?, &host(&w.to_dtype(DType::F32)?)?, name);
        }
        let (b, s, h, d) = (2usize, 37usize, 48usize, 224usize);
        let y = Tensor::from_vec(fill(b * h * s * d, 11), (b, h, s, d), &dev)?.to_dtype(DType::BF16)?;
        let gate = fill(b * s * h * 2 * d, 12);
        let got = host(&attn_output_gate(&y, &Tensor::from_vec(gate.clone(), (b, s, h * 2 * d), &dev)?, 2 * d, d)?)?;
        let y = host(&y.to_dtype(DType::F32)?)?;
        for (i, &out) in got.iter().enumerate() {
            // `out` is (b, s, h, d), `y` (b, h, s, d), the gate's half of each head the second.
            let (bi, si, hi, j) = (i / (s * h * d), i / (h * d) % s, i / d % h, i % d);
            let yv = y[((bi * h + hi) * s + si) * d + j];
            let want = yv * (1.0 / (1.0 + (-gate[((bi * s + si) * h + hi) * 2 * d + d + j]).exp()));
            assert!((out - want).abs() <= 1e-5 * (1.0 + want.abs()), "b{bi} s{si} h{hi} j{j}: kernel {out} vs {want}");
        }
        Ok(())
    }
}
