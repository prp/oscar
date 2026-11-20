#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Gated Delta Net for qwen35moe, qwen3next and qwen4exp GGUFs on Metal; the mistral.rs fork's `gdn` serves safetensors.

use crate::load::qmatvec_or_matmul;
use crate::qwen_hybrid::GdnProps;
use crate::state_slots::{emit_conv_taps, RecurrentState};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::QuantMethod;
use std::sync::Arc;

use oscar_kernels::gdn;

/// `(1/sqrt(D))·ones(D)` in F32, the weight turning `rms_norm` into `l2_norm`.
pub fn build_l2_norm_weight(dim: usize, device: &Device) -> Result<Tensor> {
    let scale = 1.0 / (dim as f64).sqrt();
    Tensor::ones((dim,), DType::F32, &Device::Cpu)?.affine(scale, 0.0)?.to_device_kept(device)
}

pub struct GatedDeltaNet {
    pub in_proj_qkvz: Arc<dyn QuantMethod>,
    pub in_proj_ba: Arc<dyn QuantMethod>,
    /// `(conv_dim, kernel)`, as the Metal conv reads it.
    pub conv1d_weight_f32: Tensor,
    pub dt_bias_f32: Tensor,
    pub a_log_f32: Tensor,
    /// The output's `rms_norm(y) * weight * act(gate)`, the activation silu but for qwen4exp's sigmoid.
    pub norm_weight_f32: Tensor,
    pub norm_eps: f32,
    pub sigmoid_gate: bool,
    pub out_proj: Arc<dyn QuantMethod>,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_kernel_size: usize,
    pub value_dim: usize,
    pub l2_norm_weight: Tensor,
    pub l2_norm_eps: f32,
}

impl GatedDeltaNet {
    pub fn forward(&self, x: &Tensor, cache: &mut RecurrentState) -> Result<Tensor> {
        let (batch_size, seq_len, _hidden) = x.dims3()?;
        let dtype = x.dtype();
        let v_per_group = self.num_v_heads / self.num_k_heads;

        // The quantized projections read and produce F32.
        let x_proj = x.to_dtype(DType::F32)?;
        let mixed_qkvz = qmatvec_or_matmul(&x_proj, &*self.in_proj_qkvz)?;
        let mixed_ba = qmatvec_or_matmul(&x_proj, &*self.in_proj_ba)?;

        // One launch prepares the recurrence's inputs over the projections' own layouts; the norm gate reads z as a view.
        let v_chunk = v_per_group * self.head_v_dim;
        let group = 2 * self.head_k_dim + 2 * v_chunk;
        let z = mixed_qkvz.reshape((batch_size, seq_len, self.num_k_heads, group))?.narrow(
            3,
            2 * self.head_k_dim + v_chunk,
            v_chunk,
        )?;
        let scale = (1.0 / (self.head_k_dim as f64).sqrt()) as f32;
        // A decode's conv update rides the prep, a prefill's full conv runs first; both die before the recurrence.
        let (q, k, v, beta, g) = {
            let (src, conv_state) = if cache.seqlen_offset > 0 && seq_len == 1 {
                (mixed_qkvz.clone(), Some(cache.conv_state.contiguous()?))
            } else {
                let mixed_qkv =
                    gdn::qkvz_split(&mixed_qkvz, self.num_k_heads, self.head_k_dim, self.head_v_dim, v_per_group)?;
                (causal_conv1d_full(&mixed_qkv, &self.conv1d_weight_f32, cache)?, None)
            };
            let prepped = gdn::prep(
                &src,
                conv_state.as_ref().map(|state| (&self.conv1d_weight_f32, state)),
                &mixed_ba,
                &self.a_log_f32,
                &self.dt_bias_f32,
                &self.l2_norm_weight,
                self.num_k_heads,
                self.head_k_dim,
                self.head_v_dim,
                v_per_group,
                self.conv_kernel_size,
                scale,
                self.l2_norm_eps,
            )?;
            if let Some(state) = conv_state {
                for e in &cache.emits {
                    e.conv.slice_set(&state, 2, 0)?;
                }
                cache.conv_state = state;
            }
            prepped
        };
        let norm_gate = (&z, &self.norm_weight_f32, self.norm_eps, self.sigmoid_gate);

        // Over `gdn::prep`'s F32 q, k, v `(B·Hv, S, D)` and g, β `(B·Hv, S)`; `gated` where the kernel applied `norm_gate`.
        let (num_heads, k_head, v_head) = (self.num_v_heads, self.head_k_dim, self.head_v_dim);
        let mut state_flat = cache
            .recurrent_state
            .to_dtype(DType::F32)?
            .reshape((batch_size * num_heads, k_head, v_head))?
            .contiguous()?;

        // The kernel writes an emit's state as it writes the slot's: the object's tensor, flat.
        let emit =
            |e: &crate::state_slots::Emit| Ok((e.at, e.recurrent.reshape((batch_size * num_heads, k_head, v_head))?));
        let emits: Vec<(usize, Tensor)> = cache.emits.iter().map(emit).collect::<Result<_>>()?;
        const CHUNK_THRESHOLD: usize = 64;
        let (out_bh, gated) = if seq_len >= CHUNK_THRESHOLD {
            (gdn::chunked_gated_delta_rule_recurrence(&q, &k, &v, &g, &beta, &mut state_flat, &emits)?, false)
        } else {
            gdn::gated_delta_rule_recurrence(&q, &k, &v, &g, &beta, &mut state_flat, Some(norm_gate), &emits)?
        };

        // The state stays F32: a BF16 round trip can feed back zeros that NaN the F32 residual.
        cache.recurrent_state = state_flat.reshape((batch_size, num_heads, k_head, v_head))?;

        // (B*Hv, S, V) is (B, S, Hv, V) in memory when S == 1.
        let out = if seq_len == 1 {
            out_bh.reshape((batch_size, 1, num_heads, v_head))?
        } else {
            out_bh.reshape((batch_size, num_heads, seq_len, v_head))?.transpose(1, 2)?.contiguous()?
        };
        let y = out.to_dtype(dtype)?;
        let out_4d = if gated { y } else { gdn::norm_gate(&y, &z, norm_gate.1, norm_gate.2, norm_gate.3)? };
        let y = out_4d.reshape((batch_size, seq_len, self.value_dim))?;
        cache.seqlen_offset += seq_len;

        qmatvec_or_matmul(&y.to_dtype(DType::F32)?, &*self.out_proj)?.to_dtype(dtype)
    }
}

/// A continuing chunk's first `kernel - 1` outputs read the carried state (newest last); a first chunk reads zeros.
fn causal_conv1d_full(x: &Tensor, weight: &Tensor, cache: &mut RecurrentState) -> Result<Tensor> {
    let seq_len = x.dim(1)?;
    let kernel = weight.dim(1)?;
    let carried = if cache.seqlen_offset > 0 { kernel - 1 } else { 0 };
    let x_t = {
        let x_t = x.transpose(1, 2)?.contiguous()?;
        if carried > 0 {
            let before = cache.conv_state.narrow(2, kernel - carried, carried)?.to_dtype(x_t.dtype())?.contiguous()?;
            Tensor::cat(&[&before, &x_t], 2)?
        } else {
            x_t
        }
    };

    emit_conv_taps(&x_t, carried, kernel, &cache.emits)?;
    let (output, new_conv_state) = gdn::causal_conv1d(&x_t, weight, kernel)?;
    cache.conv_state = new_conv_state;
    output.narrow(2, carried, seq_len)?.transpose(1, 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_cuts_match_whole, fresh_state};

    /// Bitwise, at cuts leaving continuations shorter and far longer than the kernel.
    #[test]
    fn a_prompt_in_chunks_convolves_as_one_forward() -> Result<()> {
        let (s, c, k) = (3001usize, 1003usize, 4usize);
        let x_v: Vec<f32> = (0..s * c).map(|i| ((i * 7919 % 10007) as f32 - 5003.0) / 2501.0).collect();
        let w_v: Vec<f32> = (0..c * k).map(|i| ((i * 31 % 101) as f32 - 50.0) / 97.0).collect();
        let dev = oscar_offload::pools::test_device()?;
        let (x, w) = (Tensor::from_vec(x_v, (1, s, c), &dev)?, Tensor::from_vec(w_v, (c, k), &dev)?);
        let fresh = || fresh_state(&dev, (1, c, k), Tensor::zeros((1, 1, 1, 1), DType::F32, &dev)?);
        let cuts = [&[1500][..], &[1000, 2998], &[2, 3, 1700]];
        assert_cuts_match_whole(&x, &cuts, fresh, |x, st| causal_conv1d_full(x, &w, st))
    }
}

/// From the SSM keys: head dims `ssm.state_size`, k heads `ssm.group_count`, v heads `ssm.time_step_rank`.
#[derive(Clone, Copy, Debug)]
pub struct GdnDims {
    pub head_dim: usize,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub conv_kernel: usize,
}

impl GdnDims {
    pub fn key_dim(&self) -> usize {
        self.num_k_heads * self.head_dim
    }
    pub fn value_dim(&self) -> usize {
        self.num_v_heads * self.head_dim
    }
    pub fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }
    pub fn state_dims(&self) -> Vec<usize> {
        vec![self.num_v_heads, self.head_dim, self.head_dim]
    }
    pub fn recurrent(&self, conv_width: usize) -> mistralrs_core::kv_cache::RecurrentLayerConfig {
        mistralrs_core::kv_cache::RecurrentLayerConfig {
            conv_dim: self.conv_dim(),
            conv_width,
            state_dims: self.state_dims(),
        }
    }
}

/// GGUF's `(v_per_group, num_k_heads, rest…)` → the forward's `(num_k_heads, v_per_group, rest…)`, in `t`'s shape.
fn permute_v_per_group(t: &Tensor, v_per_group: usize, num_k_heads: usize) -> Result<Tensor> {
    let rest = t.elem_count() / (v_per_group * num_k_heads);
    t.reshape((v_per_group, num_k_heads, rest))?.transpose(0, 1)?.contiguous()?.reshape(t.dims())
}

/// `v_reordered` takes the qwen35moe-style writer's v-per-group order, `sigmoid_gate` qwen4exp's norm gate.
/// The regroup runs on the host: on the device each consumed tensor would strand a dead copy in the floor.
pub fn load_gdn_layer<R: std::io::Seek + std::io::Read>(
    ct: &mut mistralrs_core::gguf::Content<'_, R>, prefix: &str, device: &Device, p: &GdnProps, dtype: DType,
    v_reordered: bool, sigmoid_gate: bool,
) -> Result<GatedDeltaNet> {
    let dims = p.dims();
    let (hidden_size, rms_norm_eps) = (p.embedding_length, p.rms_norm_eps);
    let GdnDims { head_dim, num_k_heads, num_v_heads, conv_kernel } = dims;
    let (head_k_dim, head_v_dim) = (head_dim, head_dim);
    if num_v_heads % num_k_heads != 0 {
        candle_core::bail!("ssm.time_step_rank ({num_v_heads}) must be a multiple of ssm.group_count ({num_k_heads}).");
    }
    let v_per_group = num_v_heads / num_k_heads;
    let v_chunk = v_per_group * head_v_dim;
    let fits = |d: usize| d % 32 == 0 && d / 32 <= 16;
    if !(fits(head_k_dim) && head_k_dim == gdn::KEY_WIDTH && fits(head_v_dim) && fits(v_chunk) && 2 * v_per_group <= 32)
    {
        candle_core::bail!(
            "no GDN Metal kernel takes heads of k {head_k_dim} / v {head_v_dim} with {v_per_group} v heads per k head"
        );
    }
    let (key_dim, value_dim) = (dims.key_dim(), dims.value_dim());
    // The regrouped F32 weight dies before its quantised form becomes a projection.
    let requant = |w: Tensor, dtype: GgmlDType| -> Result<Arc<dyn QuantMethod>> {
        let q = QTensor::quantize_onto(&w, dtype, device)?;
        drop(w);
        crate::load::quant(q)
    };

    let host = Device::Cpu;
    let qkv_q = ct.tensor(&format!("{prefix}.attn_qkv.weight"), &host)?;
    let qkvz_dtype = qkv_q.dtype();
    let qkv = qkv_q.dequantize(&host)?;
    let expected_qkv_rows = key_dim * 2 + value_dim;
    if qkv.dim(0)? != expected_qkv_rows || qkv.dim(1)? != hidden_size {
        candle_core::bail!(
            "{prefix}.attn_qkv.weight shape {:?} disagrees with derived ({expected_qkv_rows}, {hidden_size}). \
             Check ssm.state_size / ssm.group_count / ssm.time_step_rank vs the GGUF.",
            qkv.shape()
        );
    }
    let z_raw = ct.tensor(&format!("{prefix}.attn_gate.weight"), &host)?.dequantize(&host)?;
    let q_w = qkv.narrow(0, 0, key_dim)?;
    let k_w = qkv.narrow(0, key_dim, key_dim)?;
    let v_w_raw = qkv.narrow(0, key_dim * 2, value_dim)?;
    let (v_w, z) = if v_reordered {
        (
            permute_v_per_group(&v_w_raw, v_per_group, num_k_heads)?,
            permute_v_per_group(&z_raw, v_per_group, num_k_heads)?,
        )
    } else {
        (v_w_raw, z_raw)
    };
    let q_grouped = q_w.reshape((num_k_heads, head_k_dim, hidden_size))?;
    let k_grouped = k_w.reshape((num_k_heads, head_k_dim, hidden_size))?;
    let v_grouped = v_w.reshape((num_k_heads, v_per_group * head_v_dim, hidden_size))?;
    let z_grouped = z.reshape((num_k_heads, v_per_group * head_v_dim, hidden_size))?;
    let qkvz_out = key_dim * 2 + value_dim * 2;
    let qkvz_w_f32 = Tensor::cat(&[q_grouped, k_grouped, v_grouped, z_grouped], 1)?
        .reshape((qkvz_out, hidden_size))?
        .contiguous()?;
    let in_proj_qkvz = requant(qkvz_w_f32, qkvz_dtype)?;

    let in_proj_ba = if v_reordered {
        // b and a come apart in v-per-group order: grouped to `[b | a]` in `attn_qkv`'s dtype, one for the dispatch.
        let beta_w_raw = ct.tensor(&format!("{prefix}.ssm_beta.weight"), &host)?.dequantize(&host)?;
        let alpha_w_raw = ct.tensor(&format!("{prefix}.ssm_alpha.weight"), &host)?.dequantize(&host)?;
        let beta_w = permute_v_per_group(&beta_w_raw, v_per_group, num_k_heads)?;
        let alpha_w = permute_v_per_group(&alpha_w_raw, v_per_group, num_k_heads)?;
        let b_grouped = beta_w.reshape((num_k_heads, v_per_group, hidden_size))?;
        let a_grouped = alpha_w.reshape((num_k_heads, v_per_group, hidden_size))?;
        let ba_w_f32 =
            Tensor::cat(&[b_grouped, a_grouped], 1)?.reshape((num_v_heads * 2, hidden_size))?.contiguous()?;
        requant(ba_w_f32, qkvz_dtype)?
    } else {
        // Fused (b | a), already per-group: verbatim.
        let ba_q = ct.tensor(&format!("{prefix}.ssm_ba.weight"), device)?;
        let ba_dims = ba_q.shape().dims().to_vec();
        if ba_dims != [num_v_heads * 2, hidden_size] {
            candle_core::bail!(
                "{prefix}.ssm_ba.weight shape {ba_dims:?} disagrees with derived ({}, {hidden_size})",
                num_v_heads * 2,
            );
        }
        crate::load::quant(ba_q)?
    };

    // `ssm_a` is `-exp(A_log)` on disk; the kernel wants A_log.
    let mut read = |name: &str| -> Result<Tensor> {
        ct.tensor(&format!("{prefix}.{name}"), &host)?.dequantize(&host)?.to_dtype(dtype)
    };
    let (mut a, mut dt_bias, mut conv) = (read("ssm_a")?, read("ssm_dt.bias")?, read("ssm_conv1d.weight")?);
    let norm_weight_f32 = read("ssm_norm.weight")?.to_dtype(DType::F32)?;
    if v_reordered {
        a = permute_v_per_group(&a, v_per_group, num_k_heads)?;
        dt_bias = permute_v_per_group(&dt_bias, v_per_group, num_k_heads)?;
        // Only the conv's v section (rows 2·key_dim onward) is reordered.
        let v = permute_v_per_group(&conv.narrow(0, key_dim * 2, value_dim)?, v_per_group, num_k_heads)?;
        conv = Tensor::cat(&[conv.narrow(0, 0, key_dim * 2)?, v], 0)?;
    }
    let a_log = a.neg()?.log()?;
    let f32_form = |t: &Tensor| t.to_dtype(DType::F32)?.to_device_kept(device);
    let (conv1d_weight_f32, dt_bias_f32, a_log_f32) = (f32_form(&conv)?, f32_form(&dt_bias)?, f32_form(&a_log)?);
    let norm_weight_f32 = norm_weight_f32.to_device_kept(device)?;
    let l2_norm_weight = build_l2_norm_weight(head_dim, device)?;

    let out_proj = if v_reordered {
        // `ssm_out`'s value_dim columns take the v-per-group permute too, requantised to its dtype.
        let out_qtensor = ct.tensor(&format!("{prefix}.ssm_out.weight"), &host)?;
        let out_dtype = out_qtensor.dtype();
        let out_dequant = out_qtensor.dequantize(&host)?;
        let out_rows = out_dequant.dim(0)?;
        let out_permuted = out_dequant
            .reshape((out_rows, v_per_group, num_k_heads, head_v_dim))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((out_rows, value_dim))?;
        requant(out_permuted, out_dtype)?
    } else {
        crate::load::quant(ct.tensor(&format!("{prefix}.ssm_out.weight"), device)?)?
    };

    Ok(GatedDeltaNet {
        in_proj_qkvz,
        in_proj_ba,
        conv1d_weight_f32,
        dt_bias_f32,
        a_log_f32,
        norm_weight_f32,
        norm_eps: rms_norm_eps,
        sigmoid_gate,
        out_proj,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_kernel_size: conv_kernel,
        value_dim,
        l2_norm_weight,
        l2_norm_eps: 1e-6,
    })
}
