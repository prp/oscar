//! Shared attention: the absorbed MLA of GLM-5.2, GLM-5.3-Flash and K3, and the Qwen family's gated GQA.

use std::sync::Arc;

use crate::load::qmatvec_or_matmul;
use crate::qwen_hybrid::GdnProps;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::D;
use candle_core::{DType, Device, Result, Tensor};
use mistralrs_core::attention::SdpaParams;
use mistralrs_core::gguf::Content;
use mistralrs_core::layers::{MatMul, QRmsNorm, Qwen3VLRotaryEmbedding, RotaryEmbedding};
use mistralrs_core::paged_attention::{block_size, cache_dtype};
use mistralrs_core::text_models_inputs_processor::PagedAttentionInputMetadata;
use mistralrs_quant::QuantMethod;

/// `attn_k_b` or `attn_v_b`: the GGUF's Q8_0 `(H, N, K)`, the batched matvec's layout, kept quantized on Metal.
pub struct AbsorbedProj {
    qt: Arc<QTensor>,
    h: usize,
    n: usize,
    k: usize,
}

impl AbsorbedProj {
    /// Another format or shape is refused: no kernel takes it, and every conversion keeps both Q8_0.
    pub(crate) fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>, prefix: &str, name: &str, device: &Device,
    ) -> Result<Self> {
        let qt = ct.tensor(&format!("{prefix}.{name}"), device)?;
        match *qt.shape().dims() {
            [h, n, k] if qt.dtype() == GgmlDType::Q8_0 && k % 32 == 0 && n % 8 == 0 => {
                Ok(Self { qt: Arc::new(qt), h, n, k })
            }
            ref dims => candle_core::bail!(
                "{prefix}.{name}: absorbed matvec takes Q8_0 (H, N, K), not {:?} {dims:?}",
                qt.dtype()
            ),
        }
    }

    /// `x` F32 `(B, H, S, K)` contiguous → `(B, H, S, N)` in `out_dtype`.
    pub(crate) fn forward(&self, x: &Tensor, out_dtype: DType) -> Result<Tensor> {
        let (b, h_in, s, k_in) = x.dims4()?;
        if h_in != self.h || k_in != self.k {
            candle_core::bail!("AbsorbedProj: input (H={h_in}, K={k_in}) != weight (H={}, K={})", self.h, self.k);
        }
        let Some(storage) = self.qt.metal_storage() else {
            candle_core::bail!("AbsorbedProj weight not on Metal");
        };
        let x3 = x.reshape((b * h_in, s, k_in))?;
        oscar_kernels::metal_batched_qmatvec(storage, self.h, self.n, self.k, &x3)?
            .reshape((b, h_in, s, self.n))?
            .to_dtype(out_dtype)
    }
}

/// What a model's MLA layers share: MQA over one latent KV head, K rows the normed latent and an `extra`-wide key
/// part (rotated where `rope` is set).
pub struct MlaDims {
    pub n_head: usize,
    pub nope: usize,
    pub extra: usize,
    pub kv_lora_rank: usize,
    pub v_head_dim: usize,
    pub softmax_scale: f64,
    pub dtype: DType,
    pub rope: Option<Arc<RotaryEmbedding>>,
}

/// One layer's absorbed-MLA projections, loaded by the same names in every MLA architecture.
pub struct MlaAttention {
    attn_q_a: Arc<dyn QuantMethod>,
    attn_q_a_norm: QRmsNorm,
    attn_q_b: Arc<dyn QuantMethod>,
    attn_kv_a_mqa: Arc<dyn QuantMethod>,
    attn_kv_a_norm: QRmsNorm,
    /// `(n_head, kv_lora, qk_nope)`: q_nope into the latent space.
    attn_k_b: AbsorbedProj,
    /// `(n_head, v_head_dim, kv_lora)`: decompresses the output.
    attn_v_b: AbsorbedProj,
    attn_output: Arc<dyn QuantMethod>,
    /// K3's `attn_gate`, a sigmoid over the head output.
    gate: Option<Arc<dyn QuantMethod>>,
}

/// Bounds the absorbed-q chain and the prefill's score blocks, the ∝S transients; a slab's rows are independent.
const MLA_PREFILL_SLAB: usize = 256;

impl MlaAttention {
    pub fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>, prefix: &str, device: &Device, rms_norm_eps: f32, gated: bool,
    ) -> Result<Self> {
        let qmm = |ct: &mut Content<'_, R>, name: &str| crate::load::qmm(ct, prefix, name, device);
        Ok(Self {
            attn_q_a: qmm(ct, "attn_q_a.weight")?,
            attn_q_a_norm: crate::load::norm(ct, prefix, "attn_q_a_norm.weight", device, rms_norm_eps)?,
            attn_q_b: qmm(ct, "attn_q_b.weight")?,
            attn_kv_a_mqa: qmm(ct, "attn_kv_a_mqa.weight")?,
            attn_kv_a_norm: crate::load::norm(ct, prefix, "attn_kv_a_norm.weight", device, rms_norm_eps)?,
            attn_k_b: AbsorbedProj::load(ct, prefix, "attn_k_b.weight", device)?,
            attn_v_b: AbsorbedProj::load(ct, prefix, "attn_v_b.weight", device)?,
            attn_output: qmm(ct, "attn_output.weight")?,
            // Not `ssm_g`: the same HF name, another tensor, by layer type.
            gate: gated.then(|| qmm(ct, "attn_gate.weight")).transpose()?,
        })
    }

    /// `x` is the post-`attn_norm` hidden `(b, s, hidden)`; layer `layer`'s latent rows go to `run`'s blocks.
    pub fn forward_attn(
        &self, d: &MlaDims, x: &Tensor, start_offsets: &[usize], run: &PagedLatentRun, layer: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let q = MatMul.qmethod_matmul(x, &*self.attn_q_a)?;
        let q = self.attn_q_a_norm.forward(&q)?;
        let q = MatMul.qmethod_matmul(&q, &*self.attn_q_b)?;
        let q = q.reshape((b_sz, seq_len, d.n_head, d.nope + d.extra))?.transpose(1, 2)?;
        let q_nope = q.narrow(D::Minus1, 0, d.nope)?.contiguous()?;
        let q_extra = match d.extra {
            0 => None,
            n => Some(q.narrow(D::Minus1, d.nope, n)?.contiguous()?),
        };

        let kv = MatMul.qmethod_matmul(x, &*self.attn_kv_a_mqa)?;
        let ckv = self.attn_kv_a_norm.forward(&kv.narrow(D::Minus1, 0, d.kv_lora_rank)?.contiguous()?)?;
        // Decoupled parts: q's `(b, h, s, extra)`, k's `(b, 1, s, extra)`.
        let extra = match q_extra {
            None => None,
            Some(q_extra) => {
                let k_extra = kv
                    .narrow(D::Minus1, d.kv_lora_rank, d.extra)?
                    .contiguous()?
                    .reshape((b_sz, seq_len, 1, d.extra))?
                    .transpose(1, 2)?
                    .contiguous()?;
                Some(match &d.rope {
                    Some(rope) => rope.forward(&q_extra, &k_extra, start_offsets)?,
                    None => (q_extra, k_extra),
                })
            }
        };
        let ckv = ckv.to_dtype(d.dtype)?.reshape((b_sz, 1, seq_len, d.kv_lora_rank))?;
        let k_extra = extra.as_ref().map(|(_, k)| k.to_dtype(d.dtype)).transpose()?;
        let (k, v) = run.append(layer, &ckv, k_extra.as_ref())?;

        // The rows `[s0, s0 + len)` absorbed into the latent at `dtype`: a prefill's slab in F32, a decode's in the cache's.
        let absorb_q = |s0: usize, len: usize, dtype: DType| -> Result<Tensor> {
            let qn = self.attn_k_b.forward(&q_nope.narrow(2, s0, len)?.contiguous()?, dtype)?;
            match &extra {
                Some((q_extra, _)) => Tensor::cat(&[&qn, &q_extra.narrow(2, s0, len)?.to_dtype(dtype)?], D::Minus1),
                None => Ok(qn),
            }
        };
        let (cache_dim, out_dtype, kv_len) = (d.kv_lora_rank + d.extra, x.dtype(), k.dim(2)?);
        // A prefill runs the blocked causal pass over the latent in F32, a decode the folded chain.
        let out = if seq_len > 1 {
            let past = kv_len - seq_len;
            let ring = run.prefill_ring(k.device(), d.n_head * MLA_PREFILL_SLAB, d.kv_lora_rank)?;
            let k = k.reshape((kv_len, cache_dim))?.to_dtype(DType::F32)?;
            // The NoPE latent is its own value row.
            let v = if cache_dim == d.kv_lora_rank {
                k.clone()
            } else {
                v.reshape((kv_len, d.kv_lora_rank))?.to_dtype(DType::F32)?
            };
            let mut slabs: Vec<Tensor> = Vec::new();
            let mut s0 = 0usize;
            while s0 < seq_len {
                let len = MLA_PREFILL_SLAB.min(seq_len - s0);
                let q_slab = absorb_q(s0, len, DType::F32)?.reshape((d.n_head, len, cache_dim))?;
                let kv_end = past + s0 + len;
                let o = oscar_kernels::mla::attend_prefill(
                    &q_slab,
                    &k.narrow(0, 0, kv_end)?,
                    &v.narrow(0, 0, kv_end)?,
                    d.softmax_scale,
                    &ring,
                )?;
                let o = self.attn_v_b.forward(&o.reshape((b_sz, d.n_head, len, d.kv_lora_rank))?, out_dtype)?;
                slabs.push(o.transpose(1, 2)?.reshape((b_sz, len, d.n_head * d.v_head_dim))?);
                s0 += len;
            }
            if slabs.len() == 1 {
                slabs.into_iter().next().expect("len == 1")
            } else {
                Tensor::cat(&slabs, 1)?
            }
        } else {
            // The heads folded into the query rows against the one KV head; the softmax in F32, as llama.cpp's.
            let q = absorb_q(0, seq_len, d.dtype)?;
            let q_folded = q.contiguous()?.reshape((b_sz, 1, d.n_head * seq_len, cache_dim))?;
            let att = q_folded.matmul(&k.transpose(2, 3)?.contiguous()?)?;
            let att = (att * d.softmax_scale)?;
            let att = att.reshape((b_sz, d.n_head, seq_len, kv_len))?;
            let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?.to_dtype(d.dtype)?;
            let out = att.reshape((b_sz, 1, d.n_head * seq_len, kv_len))?.matmul(&v.contiguous()?)?.reshape((
                b_sz,
                d.n_head,
                seq_len,
                d.kv_lora_rank,
            ))?;
            let out = self.attn_v_b.forward(&out.to_dtype(DType::F32)?, out_dtype)?;
            out.transpose(1, 2)?.reshape((b_sz, seq_len, d.n_head * d.v_head_dim))?
        };
        let out = out.contiguous()?;
        let out = match &self.gate {
            Some(gate) => {
                let g = candle_nn::ops::sigmoid(&MatMul.qmethod_matmul(x, &**gate)?.to_dtype(DType::F32)?)?;
                (out.to_dtype(DType::F32)? * g)?.to_dtype(x.dtype())?
            }
            None => out,
        };
        MatMul.qmethod_matmul(&out.contiguous()?, &*self.attn_output)
    }
}

/// A forward's paged latent state: each MLA layer's address tables and two slot tensors, built once from the metadata.
pub struct PagedLatentRun {
    caches: Vec<(Tensor, Tensor)>,
    block_size: usize,
    /// The new tokens' slots, `[seq_len]` I64, negative for padding.
    write_slots: Tensor,
    /// The visible context's slots in token order, `[kv_len]` U32.
    gather_slots: Tensor,
    kv_len: usize,
    /// The prefill's working set, made at the first prefill layer; a decode forward never makes one.
    prefill_ring: std::cell::RefCell<Option<oscar_kernels::mla::PrefillRing>>,
}

impl PagedLatentRun {
    /// Paged KV is the only path, at batch 1: the gather is per sequence.
    pub fn new(x: &Tensor, metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>) -> Result<Self> {
        let (b_sz, seq_len) = x.dims2()?;
        let Some((caches, meta)) = metadata.filter(|_| b_sz == 1) else {
            candle_core::bail!("paged KV only, at batch 1 (batch {b_sz})");
        };
        let loc = x.device().location();
        let missing = |what: &str| candle_core::Error::Msg(format!("no {what} on device"));
        let bt = meta.block_tables.as_ref().and_then(|t| t.get(&loc)).ok_or_else(|| missing("block table"))?;
        let write_slots = meta.slot_mappings.get(&loc).ok_or_else(|| missing("slot mapping"))?;
        let kv_len = meta.max_context_len.ok_or_else(|| missing("max context len"))?;
        let block_size = mistralrs_core::paged_attention::block_size();
        Ok(Self {
            caches,
            block_size,
            write_slots: write_slots.flatten_all()?.narrow(0, 0, seq_len)?,
            gather_slots: slot_grid(bt, block_size, kv_len, x.device())?,
            kv_len,
            prefill_ring: Default::default(),
        })
    }

    /// The forward's prefill ring, sized at its first use: `rows_max` query rows a slab, `dv` the value width.
    pub fn prefill_ring(
        &self, device: &Device, rows_max: usize, dv: usize,
    ) -> Result<std::cell::Ref<'_, oscar_kernels::mla::PrefillRing>> {
        if self.prefill_ring.borrow().is_none() {
            let Device::Metal(dev) = device else { candle_core::bail!("MLA prefill runs on Metal") };
            *self.prefill_ring.borrow_mut() = Some(oscar_kernels::mla::PrefillRing::new(dev, rows_max, dv)?);
        }
        Ok(std::cell::Ref::map(self.prefill_ring.borrow(), |r| r.as_ref().expect("made above")))
    }

    /// Writes `layer`'s rows; returns the context's `k` `[ckv | kpe]` and `v` in token order, `kpe` the extra key part.
    pub fn append(&self, layer: usize, ckv: &Tensor, kpe: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        use oscar_kernels::mla::{gather_latent_rows, write_latent_rows};
        let (_b, _one, s, rank) = ckv.dims4()?;
        let (ckv_addrs, kpe_addrs) = &self.caches[layer];
        let gather = |addrs: &Tensor, width: usize| {
            gather_latent_rows(addrs, &self.gather_slots, width, self.block_size)?.reshape((1, 1, self.kv_len, width))
        };
        let Some(kpe) = kpe else {
            write_latent_rows(&ckv.reshape((s, rank))?, None, ckv_addrs, self.block_size, &self.write_slots)?;
            let ckv_all = gather(ckv_addrs, rank)?;
            return Ok((ckv_all.clone(), ckv_all));
        };
        let rope = kpe.dim(D::Minus1)?;
        let (ckv2, kpe2) = (ckv.reshape((s, rank))?, kpe.reshape((s, rope))?);
        write_latent_rows(&ckv2, Some((&kpe2, kpe_addrs)), ckv_addrs, self.block_size, &self.write_slots)?;
        let ckv_all = gather(ckv_addrs, rank)?;
        Ok((Tensor::cat(&[&ckv_all, &gather(kpe_addrs, rope)?], D::Minus1)?, ckv_all))
    }
}

/// The context's slot ids `table[t/bs]·bs + t%bs` in token order, built on the device.
fn slot_grid(bt: &Tensor, block_size: usize, kv_len: usize, device: &Device) -> Result<Tensor> {
    let (_num_seqs, max_blocks) = bt.dims2()?;
    let base = bt.narrow(0, 0, 1)?.reshape((max_blocks, 1))?;
    let base = base.broadcast_mul(&Tensor::full(block_size as u32, (1, 1), device)?)?;
    let offsets = Tensor::arange(0u32, block_size as u32, device)?.reshape((1, block_size))?;
    base.broadcast_add(&offsets)?.reshape((max_blocks * block_size,))?.narrow(0, 0, kv_len)
}

/// GLM-5.3-Flash's and K3's per-layer mixer: absorbed MLA over the paged run, or KDA over the recurrent slot.
pub enum LatentMixer {
    Kda(Box<crate::kda::KdaWeights>),
    Mla(Box<MlaAttention>),
}

impl LatentMixer {
    pub(crate) fn forward(
        &self, d: &MlaDims, h: &Tensor, i: usize, cache: &mut mistralrs_core::kv_cache::HybridCache, indices: &[u32],
        run: &PagedLatentRun,
    ) -> Result<Tensor> {
        match self {
            Self::Mla(m) => m.forward_attn(d, h, &[], run, i),
            Self::Kda(k) => {
                let mut state = crate::draft::LayerState::engine(cache, indices, i, None)?;
                let out = k.forward(h, state.cache())?;
                state.commit(&[])?;
                Ok(out)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Across out-of-order blocks and a partial tail.
    #[test]
    fn gather_slots_enumerate_tokens_in_order() -> Result<()> {
        let (table, block_size, kv_len) = ([7u32, 2, 5], 4usize, 10usize);
        let bt = Tensor::from_vec(table.to_vec(), (1, table.len()), &Device::Cpu)?;
        let got: Vec<u32> = slot_grid(&bt, block_size, kv_len, &Device::Cpu)?.to_vec1()?;
        let want: Vec<u32> =
            (0..kv_len).map(|t| table[t / block_size] * block_size as u32 + (t % block_size) as u32).collect();
        assert_eq!(got, want);
        Ok(())
    }
}

/// `q_proj` writes each head as `[q | gate]`, the gate's sigmoid scaling the attention output.
pub struct FullAttentionLayer {
    pub q_proj: Arc<dyn QuantMethod>,
    pub k_proj: Arc<dyn QuantMethod>,
    pub v_proj: Arc<dyn QuantMethod>,
    pub o_proj: Arc<dyn QuantMethod>,
    pub q_norm: QRmsNorm,
    pub k_norm: QRmsNorm,
    pub rotary_emb: Arc<Qwen3VLRotaryEmbedding>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rot_dim: usize,
    pub sdpa_params: SdpaParams,
}

impl FullAttentionLayer {
    /// `x` already normalised: the layer's `attn_norm`, or the previous finalize's fold of it.
    pub fn forward(
        &self, x: &Tensor, cos_sin: &(Tensor, Tensor), metadata: ((Tensor, Tensor), &PagedAttentionInputMetadata),
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        // The quantized projections take and give F32: no cast.
        debug_assert!(x.dtype() == DType::F32);
        let q_gate = qmatvec_or_matmul(x, &*self.q_proj)?;
        let k = qmatvec_or_matmul(x, &*self.k_proj)?;
        let v = qmatvec_or_matmul(x, &*self.v_proj)?;

        // The decode kernels and the gate read `[q | gate]` in place; only the tensor chain narrows it.
        let q_gate = q_gate.reshape((b_sz, seq_len, self.num_heads * self.head_dim * 2))?;
        let k = k.reshape((b_sz, seq_len, self.num_kv_heads * self.head_dim))?;
        let v = v.reshape((b_sz, seq_len, self.num_kv_heads * self.head_dim))?;

        // A decode's one launch applies the norms and partial rope and writes q, k, v as bf16.
        let (q, k, v) = if seq_len == 1 {
            oscar_kernels::gdn::attn_pack(
                &q_gate,
                &k,
                &v,
                &self.q_norm.weight_for_dtype(DType::F32)?,
                &self.k_norm.weight_for_dtype(DType::F32)?,
                self.q_norm.eps_f32(),
                &cos_sin.0,
                &cos_sin.1,
                self.rot_dim,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )?
        } else {
            let q = q_gate.reshape((b_sz, seq_len, self.num_heads, self.head_dim * 2))?.narrow(
                D::Minus1,
                0,
                self.head_dim,
            )?;
            let dims = (b_sz, seq_len, self.num_heads, self.num_kv_heads, self.head_dim);
            let (mut q, mut k, v) = normed_heads(q, &k, &v, dims, &self.q_norm, &self.k_norm)?;
            if self.rot_dim < self.head_dim {
                let mut q_rot = q.narrow(D::Minus1, 0, self.rot_dim)?;
                let q_pass = q.narrow(D::Minus1, self.rot_dim, self.head_dim - self.rot_dim)?;
                let mut k_rot = k.narrow(D::Minus1, 0, self.rot_dim)?;
                let k_pass = k.narrow(D::Minus1, self.rot_dim, self.head_dim - self.rot_dim)?;

                self.rotary_emb.forward(cos_sin, &mut q_rot, &mut k_rot)?;
                q = Tensor::cat(&[q_rot, q_pass], D::Minus1)?;
                k = Tensor::cat(&[k_rot, k_pass], D::Minus1)?;
            } else {
                self.rotary_emb.forward(cos_sin, &mut q, &mut k)?;
            }
            (q, k, v)
        };

        // BF16 (F32 at head_dim 256 needs 48 KB of threadgroup memory, over Metal's 32 KB); the scope frees the copies.
        let ((key_cache, value_cache), input_metadata) = metadata;
        let y = {
            let (q, k, v) = (q.to_dtype(DType::BF16)?, k.to_dtype(DType::BF16)?, v.to_dtype(DType::BF16)?);
            paged_attention(&q, &k, &v, (&key_cache, &value_cache), input_metadata, &self.sdpa_params)?
        };
        // At decode one kernel gates the attention's bf16.
        let y = if seq_len == 1 {
            // Paged decode returns y as (B, H, D), the bytes of (B, H, 1, D).
            let y4 = if y.rank() == 3 { y.reshape((b_sz, self.num_heads, seq_len, self.head_dim))? } else { y };
            oscar_kernels::gdn::attn_output_gate(&y4, &q_gate, 2 * self.head_dim, self.head_dim)?
        } else {
            let y = y.to_dtype(DType::F32)?.reshape((b_sz, seq_len, ()))?;
            let gate = q_gate
                .reshape((b_sz, seq_len, self.num_heads, self.head_dim * 2))?
                .narrow(D::Minus1, self.head_dim, self.head_dim)?
                .reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;
            let gate_sig = candle_nn::ops::sigmoid(&gate.to_dtype(y.dtype())?)?;
            y.broadcast_mul(&gate_sig)?
        };
        qmatvec_or_matmul(&y, &*self.o_proj)
    }
}

/// `q` `(b, s, heads, d)`, `k` and `v` `(b, s, kv_heads·d)` as `(b, heads, s, d)` heads, q and k through their norms.
pub(crate) fn normed_heads(
    q: Tensor, k: &Tensor, v: &Tensor, (b_sz, seq_len, heads, kv_heads, head_dim): (usize, usize, usize, usize, usize),
    q_norm: &QRmsNorm, k_norm: &QRmsNorm,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (q, k, v) = if seq_len != 1 {
        let kv = |t: &Tensor| t.reshape((b_sz, seq_len, kv_heads, head_dim))?.transpose(1, 2);
        (q.transpose(1, 2)?, kv(k)?, kv(v)?)
    } else {
        let kv = |t: &Tensor| t.reshape((b_sz, kv_heads, seq_len, head_dim));
        (q.reshape((b_sz, heads, seq_len, head_dim))?, kv(k)?, kv(v)?)
    };
    let (q_flat, k_flat) = (q.flatten(0, 2)?, k.flatten(0, 2)?);
    let (q_flat, k_flat) = (q_norm.forward(&q_flat)?, k_norm.forward(&k_flat)?);
    let q = q_flat.reshape((b_sz, heads, seq_len, head_dim))?;
    Ok((q, k_flat.reshape((b_sz, kv_heads, seq_len, head_dim))?, v))
}

pub fn load_gated_attention_layer<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>, prefix: &str, device: &Device, p: &GdnProps, rotary_emb: Arc<Qwen3VLRotaryEmbedding>,
    dtype: DType,
) -> Result<FullAttentionLayer> {
    if p.head_dim % 32 != 0 || p.head_dim > 512 {
        candle_core::bail!("{prefix}: no attention decode kernel takes heads {} wide", p.head_dim);
    }
    let q_proj: Arc<dyn QuantMethod> = crate::load::quant(ct.tensor(&format!("{prefix}.attn_q.weight"), device)?)?;
    let k_proj: Arc<dyn QuantMethod> = crate::load::quant(ct.tensor(&format!("{prefix}.attn_k.weight"), device)?)?;
    let v_proj: Arc<dyn QuantMethod> = crate::load::quant(ct.tensor(&format!("{prefix}.attn_v.weight"), device)?)?;
    let o_proj: Arc<dyn QuantMethod> = crate::load::quant(ct.tensor(&format!("{prefix}.attn_output.weight"), device)?)?;

    let mut norm = |name: &str| {
        let scale = ct.tensor(&format!("{prefix}.{name}.weight"), &Device::Cpu)?;
        QRmsNorm::new_with_dtype(scale, p.rms_norm_eps, dtype, device)
    };
    let (q_norm, k_norm) = (norm("attn_q_norm")?, norm("attn_k_norm")?);

    Ok(FullAttentionLayer {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
        rotary_emb,
        num_heads: p.head_count,
        num_kv_heads: p.head_count_kv,
        head_dim: p.head_dim,
        rot_dim: p.rope_dim_count,
        sdpa_params: SdpaParams {
            n_kv_groups: p.head_count / p.head_count_kv,
            softcap: None,
            softmax_scale: 1.0 / (p.head_dim as f32).sqrt(),
            sliding_window: None,
            sinks: None,
        },
    })
}

/// Attention over the block pool for prompt and decode: this call's K/V written into their blocks, then the paged
/// kernel over the block tables with each query row's visible-key count; a single sequence of 256+ rows gathers its
/// context and runs the contiguous SDPA instead. `q`, `k`, `v` `(b, heads, seq, hd)`; out `(b * seq, heads, hd)`.
pub fn paged_attention(
    q: &Tensor, k: &Tensor, v: &Tensor, (k_addrs, v_addrs): (&Tensor, &Tensor), meta: &PagedAttentionInputMetadata,
    p: &SdpaParams,
) -> Result<Tensor> {
    let (b, heads, seq_len, head_size) = q.dims4()?;
    let kv_heads = k.dim(1)?;
    let dev = q.device().location();
    let tables = meta.block_tables.as_ref().and_then(|t| t.get(&dev));
    let lens = meta.context_lens.as_ref().and_then(|t| t.get(&dev));
    let (Some(tables), Some(lens), Some(visible), Some(slots)) =
        (tables, lens, meta.max_context_len, meta.slot_mappings.get(&dev))
    else {
        candle_core::bail!("paged attention: a step without block tables");
    };
    if cache_dtype() != q.dtype() {
        candle_core::bail!("paged attention: the KV pool holds {:?}, the query {:?}", cache_dtype(), q.dtype());
    }
    let block_size = block_size();
    // The kernels take one row per token.
    let rows = |t: &Tensor, h: usize| {
        if seq_len > 1 {
            t.transpose(1, 2)?.reshape(((), h, head_size))
        } else {
            t.reshape(((), h, head_size))
        }
    };
    let slots = if slots.rank() > 1 { slots.flatten_all()? } else { slots.clone() };
    oscar_kernels::paged::reshape_and_cache(
        &rows(k, kv_heads)?,
        &rows(v, kv_heads)?,
        k_addrs,
        v_addrs,
        block_size,
        &slots,
    )?;
    // The paged prompt kernel re-reads K/V quadratically above the contiguous kernel's coefficient, whose causal
    // predicate is the same per-row rule.
    const PROMPT_GATHER_MIN_ROWS: usize = 256;
    let softcap = p.softcap.unwrap_or(1.0);
    if seq_len >= PROMPT_GATHER_MIN_ROWS
        && b == 1
        && softcap == 1.0
        && [32, 64, 72, 80, 96, 128, 256].contains(&head_size)
    {
        let blocks = visible.div_ceil(block_size);
        let table = tables.flatten_all()?.narrow(0, 0, blocks)?;
        let (k_g, v_g) = oscar_kernels::paged::gather_kv_cache(
            k_addrs,
            v_addrs,
            &table,
            blocks,
            block_size,
            kv_heads,
            head_size,
            cache_dtype(),
        )?;
        let (k_g, v_g) = (k_g.narrow(1, 0, visible)?.unsqueeze(0)?, v_g.narrow(1, 0, visible)?.unsqueeze(0)?);
        let out = candle_nn::ops::sdpa(q, &k_g, &v_g, None, true, p.softmax_scale, softcap)?;
        return out.transpose(1, 2)?.contiguous()?.reshape((b * seq_len, heads, head_size));
    }
    oscar_kernels::paged::paged_attention(
        &rows(q, heads)?,
        k_addrs,
        v_addrs,
        kv_heads,
        block_size,
        tables,
        lens,
        visible,
        p.softmax_scale,
        softcap,
    )
}
