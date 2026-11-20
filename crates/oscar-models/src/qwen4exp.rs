#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! `qwen4exp` (Qwen3.8-Flash-Next): a GDN/attention hybrid under hyper-connected residual streams, with a PLE layer.
//! The QSA indexer is never loaded: the dense paged path is the reference's answer without one.

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor, D};
use mistralrs_quant::{MatMul, QuantMethod};

use crate::draft::Source;
use crate::embedding::QuantizedEmbedding;
use crate::hc::{stream_scatter, LowRankHyperConnection};
use crate::moe::{load_moe_with_shared, RoutedLayer};
use crate::ple::PleParams;
use crate::qwen_hybrid::{mrope_cos_sin, GdnProps, Mixer};
use crate::state_slots::{Emit, RecurrentState};
use mistralrs_core::device_map::DeviceMapper;
use mistralrs_core::extract_logits;
use mistralrs_core::gguf::ArchDeclaration;
use mistralrs_core::gguf::Content;
use mistralrs_core::kv_cache::{EitherCache, HybridLayerCache, RecurrentLayerConfig};
use mistralrs_core::layers::Qwen3VLRotaryEmbedding;
use mistralrs_core::paged_attention::AttentionImplementation;
use mistralrs_core::text_models_inputs_processor::PagedAttentionInputMetadata;
use mistralrs_core::utils::model_config as ModelConfig;

use crate::declare::Meta;

#[derive(Debug, Clone)]
pub(crate) struct PropsGGUF {
    pub gdn: GdnProps,
    /// `hyper_connection.count`, the residual stream count.
    pub hc_mult: usize,
    /// The one injection layer; `None` without a PLE key group.
    pub ple_layer: Option<usize>,
    pub ple_params: Option<PleParams>,
    pub ple_conv_kernel: usize,
}

impl std::ops::Deref for PropsGGUF {
    type Target = GdnProps;
    fn deref(&self) -> &GdnProps {
        &self.gdn
    }
}

impl PropsGGUF {
    /// The PLE layer's conv row also carries the PLE history and the id tail.
    fn ple_override(&self) -> Result<Vec<(usize, RecurrentLayerConfig)>> {
        let (Some(li), Some(p)) = (self.ple_layer, &self.ple_params) else { return Ok(vec![]) };
        let dims = self.dims();
        crate::architecture::ensure(
            dims.conv_dim() == self.hc_mult * self.embedding_length,
            "qwen4exp: the PLE history rides the GDN conv row, which needs \
             conv_dim == hc_mult * hidden",
        )?;
        let extra = PleBlock::hist_width(self.ple_conv_kernel, p.ngram_size) + 1;
        Ok(vec![(li, dims.recurrent(self.ssm_conv_kernel + extra))])
    }
}

const ARCHS: &[&str] = &["qwen4exp"];

/// The family's declaration, plus the PLE row's wide conv.
pub(crate) fn declare<R: std::io::Seek + std::io::Read>(ct: &Content<'_, R>) -> Result<ArchDeclaration> {
    let m = Meta::of(ct.get_metadata(), ARCHS)?;
    let props = PropsGGUF::read(&m).or_else(|e| candle_core::bail!("{e}"))?;
    props.declare(&m, props.ple_override()?)
}

impl PropsGGUF {
    fn read(m: &Meta<'_>) -> anyhow::Result<Self> {
        let gdn = GdnProps::parse(m)?;
        // The PLE key group is all or nothing: a partial one is refused.
        let (ple_layer, ple_params, ple_conv_kernel) = if m.get("ple.layers").is_ok() {
            let layers = m.u64_array("ple.layers")?;
            anyhow::ensure!(layers.len() == 1, "qwen4exp supports exactly one PLE layer");
            let ngram_size = m.usize("ple.ngram_size")?;
            let heads_per_ngram = m.usize("ple.heads_per_ngram")?;
            let n_heads = (ngram_size - 1) * heads_per_ngram;
            let multipliers = m.u64_array("ple.layer_multipliers")?;
            anyhow::ensure!(
                multipliers.len() >= ngram_size,
                "ple.layer_multipliers has {} entries for ngram size {ngram_size}",
                multipliers.len()
            );
            let offsets = m.u64_array("ple.head_offsets")?;
            let vocabs = m.u64_array("ple.head_vocab_sizes")?;
            anyhow::ensure!(
                offsets.len() >= n_heads && vocabs.len() >= n_heads,
                "PLE head ranges cover {} / {} of {n_heads} heads",
                offsets.len(),
                vocabs.len()
            );
            let head_dim = m.usize("embedding_length_per_layer_input")?;
            let hidden = m.usize("embedding_length")?;
            anyhow::ensure!(
                n_heads * head_dim == hidden,
                "PLE heads x row width ({n_heads} x {head_dim}) must equal the hidden \
                 width ({hidden}) - the gathered embedding IS the layer input"
            );
            let narrow = |v: Vec<u64>| -> anyhow::Result<Vec<u32>> {
                v.into_iter()
                    .take(n_heads)
                    .map(|x| u32::try_from(x).map_err(|_| anyhow::anyhow!("PLE head range over u32 row index")))
                    .collect()
            };
            (
                Some(layers[0] as usize),
                Some(PleParams {
                    ngram_size,
                    heads_per_ngram,
                    eos_token_id: m.usize("ple.eos_token_id")? as u32,
                    multipliers,
                    head_offsets: narrow(offsets)?,
                    head_vocab_sizes: narrow(vocabs)?,
                }),
                m.usize("ple.conv_kernel")?,
            )
        } else {
            (None, None, 0)
        };
        Ok(PropsGGUF { gdn, hc_mult: m.usize("hyper_connection.count")?, ple_layer, ple_params, ple_conv_kernel })
    }
}

struct PleBlock {
    key: Arc<dyn QuantMethod>,
    value: Arc<dyn QuantMethod>,
    norm_key: Tensor,
    norm_query: Tensor,
    norm_conv: Tensor,
    /// `(hc_dim, kernel)` F32, tap axis last as candle reverses the GGUF `ne`: tap k of channel c is `conv_w[c][k]`.
    conv_w: Tensor,
    params: PleParams,
    conv_kernel: usize,
    /// The n-gram size, per the reference.
    dilation: usize,
    /// The conv row: the GDN taps, then the PLE history and the id tail; the session lifecycle covers all three as one.
    gdn_width: usize,
    hist_width: usize,
    store: &'static std::sync::Arc<crate::ple::RowGatherStore>,
}

impl PleBlock {
    fn hist_width(conv_kernel: usize, ngram_size: usize) -> usize {
        (conv_kernel - 1) * ngram_size
    }
}

struct Layer {
    hc_attn: LowRankHyperConnection,
    hc_ffn: LowRankHyperConnection,
    attn: Mixer,
    moe: RoutedLayer,
    ple: Option<PleBlock>,
}

pub struct ModelWeights {
    tok_embeddings: QuantizedEmbedding,
    layers: Vec<Layer>,
    hc_head: LowRankHyperConnection,
    output: Arc<dyn QuantMethod>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    rope: Arc<Qwen3VLRotaryEmbedding>,
    hidden: usize,
    hc_mult: usize,
    rms_norm_eps: f64,
    unit_gain: Tensor,
}

fn load_ple_block<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>, prefix: &str, device: &Device, props: &PropsGGUF,
) -> Result<PleBlock> {
    let params = props.ple_params.clone().expect("caller checked ple_layer is set");
    let store = crate::ple::store().ok_or_else(|| {
        candle_core::Error::Msg(
            "qwen4exp: the PLE row-gather store is not installed - the runtime must \
             install it from the artifact before the build"
                .to_string(),
        )
    })?;
    crate::architecture::ensure(
        params.n_heads() * store.head_dim == props.embedding_length,
        "qwen4exp: PLE head_dim x heads must equal the hidden width",
    )?;
    let f32_of =
        |name: &str, ct: &mut Content<'_, R>| crate::load::f32_dequant(ct, &format!("{prefix}.{name}"), device);
    Ok(PleBlock {
        key: crate::load::qmm(ct, prefix, "ple_key.weight", device)?,
        value: crate::load::qmm(ct, prefix, "ple_value.weight", device)?,
        norm_key: f32_of("ple_norm_key.weight", ct)?,
        norm_query: f32_of("ple_norm_query.weight", ct)?,
        norm_conv: f32_of("ple_norm_conv.weight", ct)?,
        conv_w: f32_of("ple_conv1d.weight", ct)?,
        dilation: params.ngram_size,
        conv_kernel: props.ple_conv_kernel,
        gdn_width: props.ssm_conv_kernel,
        hist_width: PleBlock::hist_width(props.ple_conv_kernel, params.ngram_size),
        params,
        store,
    })
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>, device: &Device, mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation, dtype: DType,
    ) -> Result<Self> {
        crate::load::admit(&attention_mechanism, &*mapper, device)?;
        let props = PropsGGUF::read(&Meta::of(ct.get_metadata(), ARCHS)?).or_else(|e| candle_core::bail!("{e}"))?;

        let tok_embeddings = QuantizedEmbedding::load(&mut ct, device, props.embedding_length, dtype)?;
        let output_qtensor = crate::load::output_head(&mut ct, device)?;
        // The final mixer is the output norm: no separate one ships.
        let hc_head = LowRankHyperConnection {
            norm_w: crate::load::f32_dequant(&mut ct, "output_hc_norm.weight", device)?,
            down: crate::load::quant(ct.tensor("output_hc_down.weight", device)?)?,
            up: crate::load::quant(ct.tensor("output_hc_up.weight", device)?)?,
            inject: None,
        };

        let rope = props.mrope(device)?;

        let mut layers = Vec::with_capacity(props.n_layer);
        for i in crate::load::layers(props.n_layer) {
            let prefix = format!("blk.{i}");

            let hc_attn = LowRankHyperConnection::load(&mut ct, &prefix, "attn", device, true)?;
            let hc_ffn = LowRankHyperConnection::load(&mut ct, &prefix, "ffn", device, true)?;
            let moe =
                load_moe_with_shared(&mut ct, &prefix, i, device, props.embedding_length, props.expert_used_count)?;
            let ple =
                if props.ple_layer == Some(i) { Some(load_ple_block(&mut ct, &prefix, device, &props)?) } else { None };

            // The site mixer already normalised the attention's input; the vendor's one GDN difference is a sigmoid gate.
            let attn = props.mixer(&mut ct, &prefix, i, device, &rope, dtype, true, true)?;
            layers.push(Layer { hc_attn, hc_ffn, attn, moe, ple });
        }

        let cache = crate::declare::hybrid_cache(&declare(&ct)?, props.max_seq_len, dtype, device)?;

        Ok(Self {
            tok_embeddings,
            layers,
            hc_head,
            output: crate::load::quant(output_qtensor)?,
            device: device.clone(),
            cache,
            max_seq_len: props.max_seq_len,
            rope,
            hidden: props.embedding_length,
            hc_mult: props.hc_mult,
            rms_norm_eps: props.rms_norm_eps as f64,
            unit_gain: Tensor::ones(props.embedding_length, DType::F32, &Device::Cpu)?.to_device_kept(device)?,
        })
    }
}

/// Per-stream RMS with the unit gain, then a `(h·d,)` gamma.
fn grouped_norm(x: &Tensor, gamma: &Tensor, unit: &Tensor, h: usize, d: usize, eps: f64) -> Result<Tensor> {
    let (b, s, hd) = x.dims3()?;
    debug_assert_eq!(hd, h * d);
    let normed = candle_nn::ops::rms_norm(&x.reshape((b, s, h, d))?.contiguous()?, unit, eps as f32)?;
    normed.reshape((b, s, h * d))?.broadcast_mul(gamma)
}

/// The PLE layer's conv row: the engine's slot's, or the drafter's copy.
fn ple_row(source: &mut Source<'_>, i: usize) -> Result<Tensor> {
    match source {
        Source::Engine { cache, indices, .. } => {
            let Some(HybridLayerCache::Recurrent(pool)) = cache.get_mut(i) else {
                candle_core::bail!("qwen4exp: PLE layer {i} wants state pool");
            };
            pool.gather_conv_state(indices)
        }
        Source::Draft { private, .. } => match private.get(i) {
            Some(Some(state)) => Ok(state.conv_state.clone()),
            _ => candle_core::bail!("qwen4exp draft forward: PLE layer {i} GDN state ungrounded"),
        },
    }
}

impl ModelWeights {
    pub fn forward(
        &self, x: &Tensor, start_offsets: &[usize], context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (streams, tokens) = self.streams(x)?;
        let mut cache = self.cache.hybrid();
        let mut source = Source::engine(&mut cache, metadata)?;
        let cos_sin = mrope_cos_sin(&self.rope, start_offsets, x)?;
        self.provide_router_tables(x);
        let streams = self.run_layers(streams, &mut source, &cos_sin, &tokens)?;
        self.head(&streams, context_lens)
    }

    /// The F32 stream stack, each stream starting as the embedding, and the token ids the PLE layer hashes.
    fn streams(&self, x: &Tensor) -> Result<(Tensor, Vec<u32>)> {
        let (b_sz, seq_len) = x.dims2()?;
        let tokens: Vec<u32> = x.flatten_all()?.to_vec1::<u32>()?;
        let emb = self.tok_embeddings.forward(x)?.to_dtype(DType::F32)?;
        let streams = emb
            .reshape((b_sz, seq_len, 1, self.hidden))?
            .broadcast_as((b_sz, seq_len, self.hc_mult, self.hidden))?
            .contiguous()?;
        Ok((streams, tokens))
    }

    fn head(&self, streams: &Tensor, context_lens: Vec<(usize, usize)>) -> Result<Tensor> {
        let (_, collapsed) = self.hc_head.forward(streams, &self.unit_gain, self.rms_norm_eps)?;
        let xs = extract_logits(&collapsed, context_lens)?;
        MatMul.qmethod_matmul(&xs.contiguous()?, &*self.output)
    }

    /// The mixer's gamma averaged over streams stands in for the hc_ffn mix, which no norm gives: predictions only.
    fn provide_router_tables(&self, x: &Tensor) {
        crate::moe::provide_router_tables(x, || {
            self.layers
                .iter()
                .map(|layer| {
                    let gamma = layer.hc_ffn.norm_w.to_device(&Device::Cpu)?;
                    Ok((gamma.reshape((self.hc_mult, self.hidden))?.mean(0)?, layer.moe.gate_matmul(), None))
                })
                .collect()
        });
    }

    /// Returns the streams and the conv row's new `(hist, id_tail)` columns, written at each emit's position too.
    #[allow(clippy::too_many_arguments)]
    fn forward_ple(
        &self, ple: &PleBlock, streams: &Tensor, hist: &Tensor, id_tail: &Tensor, tokens: &[u32], n_seqs: usize,
        emits: &[Emit],
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, s, h, d) = streams.dims4()?;
        debug_assert_eq!(b, n_seqs);
        let hc_dim = h * d;
        let store = ple.store;

        // The id tail holds each sequence's last predecessors as id+1, 0 for missing (hashed as EOS).
        let n_prev = ple.params.ngram_size - 1;
        let tail: Vec<f32> = id_tail.flatten_all()?.to_vec1::<f32>()?;
        let n_heads = ple.params.n_heads();
        let mut ids = Vec::with_capacity(b * s * n_heads);
        for seq in 0..b {
            let toks = &tokens[seq * s..(seq + 1) * s];
            for j in 0..s {
                let mut ctx: Vec<Option<u32>> = Vec::with_capacity(ple.params.ngram_size);
                ctx.push(Some(toks[j]));
                for back in 1..=n_prev {
                    ctx.push(if j >= back {
                        Some(toks[j - back])
                    } else {
                        // Tail channel back-1-j: the id `back` before token j.
                        let v = tail[seq * hc_dim + (back - 1 - j)];
                        (v >= 1.0).then(|| v as u32 - 1)
                    });
                }
                ple.params.hash_token(&ctx, &mut ids);
            }
        }
        let emb = store.gather_tensor(&ids, (b, s, n_heads * store.head_dim), &self.device)?;

        let key = MatMul.qmethod_matmul(&emb, &*ple.key)?;
        let value = MatMul.qmethod_matmul(&emb, &*ple.value)?;
        let key = grouped_norm(&key, &ple.norm_key, &self.unit_gain, h, d, self.rms_norm_eps)?;
        let query =
            grouped_norm(&streams.reshape((b, s, hc_dim))?, &ple.norm_query, &self.unit_gain, h, d, self.rms_norm_eps)?;

        let sg = key.reshape((b, s, h, d))?.broadcast_mul(&query.reshape((b, s, h, d))?)?.sum(D::Minus1)?;
        let sg = (sg / (self.hidden as f64).sqrt())?;
        let mag = sg.abs()?.clamp(1e-6f32, f32::INFINITY)?.sqrt()?;
        let sgn = sg.sign()?;
        let gate = candle_nn::ops::sigmoid(&sgn.mul(&mag)?)?;

        let gated = value.reshape((b, s, 1, d))?.broadcast_mul(&gate.unsqueeze(D::Minus1)?)?.reshape((b, s, hc_dim))?;

        let normalized = grouped_norm(&gated, &ple.norm_conv, &self.unit_gain, h, d, self.rms_norm_eps)?;

        // Dilated depthwise causal conv over [history | chunk]: tap k reads (kernel-1-k)·dilation back.
        let hist_w = ple.hist_width;
        let padded = Tensor::cat(&[hist.transpose(1, 2)?.contiguous()?, normalized.clone()], 1)?;
        let mut conv_out: Option<Tensor> = None;
        for k in 0..ple.conv_kernel {
            let start = hist_w - (ple.conv_kernel - 1 - k) * ple.dilation;
            let shifted = padded.narrow(1, start, s)?;
            let wk = ple.conv_w.narrow(1, k, 1)?.contiguous()?.reshape(hc_dim)?;
            let term = shifted.broadcast_mul(&wk)?;
            conv_out = Some(match conv_out {
                Some(acc) => (acc + term)?,
                None => term,
            });
        }
        let conv_out = candle_nn::ops::silu(&conv_out.expect("kernel > 0"))?;

        let add = (gated + conv_out)?.reshape((b, s, h, d))?;
        let new_streams = (streams + add)?;

        // After `n` tokens: the history is the last hist_w positions, channel-major as the pool; the id tail the
        // trailing predecessors as id+1, the previous tail's shifted deeper where the chunk is shorter.
        let hist_at = |n: usize| padded.narrow(1, n, hist_w)?.transpose(1, 2)?.contiguous();
        let tail_at = |n: usize| -> Result<Tensor> {
            let mut tail_new = vec![0f32; b * hc_dim];
            for seq in 0..b {
                let toks = &tokens[seq * s..(seq + 1) * s];
                for back in 1..=n_prev {
                    tail_new[seq * hc_dim + (back - 1)] =
                        if n >= back { toks[n - back] as f32 + 1.0 } else { tail[seq * hc_dim + (back - 1 - n)] };
                }
            }
            Tensor::from_vec(tail_new, (b, hc_dim, 1), &self.device)
        };
        for e in emits {
            e.conv.slice_set(&hist_at(e.at)?, 2, ple.gdn_width)?;
            e.conv.slice_set(&tail_at(e.at)?, 2, ple.gdn_width + hist_w)?;
        }
        Ok((new_streams, hist_at(s)?, tail_at(s)?))
    }

    /// The qwen35moe loop on the wide residual, with the PLE injection and the site mixers.
    fn run_layers(
        &self, mut streams: Tensor, source: &mut Source<'_>, cos_sin: &(Tensor, Tensor), tokens: &[u32],
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _, _) = streams.dims4()?;
        for (i, layer) in self.layers.iter().enumerate() {
            // The late tap, the engine's only: the stream mean stands in at the hidden width, made only where read.
            if source.is_engine() {
                crate::moe::offer_late_tap(i, &streams, &|s| s.mean(2));
            }

            // PLE first, as the reference updates the wide residual before the site mixers.
            let mut ple_carry: Vec<Tensor> = Vec::new();
            if let Some(ple) = &layer.ple {
                let row = ple_row(source, i)?;
                let (gw, hw) = (ple.gdn_width, ple.hist_width);
                let hist = row.narrow(2, gw, hw)?.contiguous()?;
                let id_tail = row.narrow(2, gw + hw, 1)?.contiguous()?;
                let (new_streams, new_hist, new_tail) =
                    self.forward_ple(ple, &streams, &hist, &id_tail, tokens, b_sz, &source.emits(i))?;
                streams = new_streams;
                ple_carry = vec![new_hist, new_tail];
            }

            // Wait out the GPU work before an attention layer: a CPU running ahead would hold more layers' activations.
            if matches!(layer.attn, Mixer::Full(_)) && seq_len >= 256 {
                if let Device::Metal(md) = streams.device() {
                    let _ = md.wait_until_completed();
                }
            }
            let residual = streams;
            let (inject, collapsed) = layer.hc_attn.forward(&residual, &self.unit_gain, self.rms_norm_eps)?;
            let inject = inject.expect("layer mixers carry the inject head");
            let attn_out = match &layer.attn {
                Mixer::Full(attn) => {
                    let meta = source.attention(i)?;
                    attn.forward(&collapsed, cos_sin, meta)?.to_dtype(DType::F32)?
                }
                Mixer::Linear(attn) => {
                    let taps = layer.ple.as_ref().map(|ple| ple.gdn_width);
                    let mut state = source.recurrent(i, taps)?;
                    let out = attn.forward(&collapsed, state.cache())?;
                    // The PLE columns rejoin the row.
                    state.commit(&ple_carry)?;
                    out.to_dtype(DType::F32)?
                }
            };
            streams = stream_scatter(&residual, &inject, &attn_out)?;

            let residual = streams;
            let (inject, collapsed) = layer.hc_ffn.forward(&residual, &self.unit_gain, self.rms_norm_eps)?;
            let inject = inject.expect("layer mixers carry the inject head");
            let moe_out = layer.moe.forward(&collapsed, None)?.stream.to_dtype(DType::F32)?;
            streams = stream_scatter(&residual, &inject, &moe_out)?;
        }
        Ok(streams)
    }

    /// The drafter's one-token decode; the PLE slot's private state is the wider row, deep-copied at the hand-off.
    pub fn forward_with_cache(
        &self, x: &Tensor, start_offsets: &[usize], context_lens: Vec<(usize, usize)>,
        view: &crate::draft::DraftPagedView, cache: &mut [Option<RecurrentState>],
    ) -> Result<Tensor> {
        let (streams, tokens) = self.streams(x)?;
        let cos_sin = mrope_cos_sin(&self.rope, start_offsets, x)?;
        let meta = view.decode_metadata(start_offsets[0], &self.device)?;
        let mut source = Source::Draft { view, meta: &meta, private: cache };
        let streams = self.run_layers(streams, &mut source, &cos_sin, &tokens)?;
        self.head(&streams, context_lens)
    }
}

impl crate::architecture::Architecture for ModelWeights {
    const ROUTER_TABLES: bool = true;
    const LATE_TAP: bool = true;
    fn draftable(model: Arc<Self>) -> Option<Arc<dyn crate::draft::DraftableModel>> {
        Some(model)
    }
}

crate::draftable!(ModelWeights);
