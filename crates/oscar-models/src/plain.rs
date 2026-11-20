//! The plain-residual trunk of GLM-5.2 and Qwen3-MoE: pre-norm attention and FFN, each added to one hidden stream.

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_core::extract_logits;
use mistralrs_core::gguf::GgufArch;
use mistralrs_core::kv_cache::EitherCache;
use mistralrs_core::layers::{MatMul, QRmsNorm};
use mistralrs_core::text_models_inputs_processor::PagedAttentionInputMetadata;
use mistralrs_quant::QuantMethod;

use crate::draft::DraftPagedView;
use crate::embedding::QuantizedEmbedding;
use crate::moe::Ffn;

/// One architecture's attention over the trunk: what every layer shares, a forward's paged state, a layer's pass.
pub trait PlainAttention: Send + Sync + 'static {
    /// The widths, rope and KV dtype every layer reads.
    type Shared: Send + Sync;
    /// A forward's paged state, from the engine's metadata or the drafter's view.
    type Run<'a>;
    /// It offers the late tap (GLM-5.2), and drafts (Qwen3).
    const LATE_TAP: bool;
    const DRAFTS: bool;
    fn engine_run<'a>(
        x: &Tensor, metadata: Option<(Vec<(Tensor, Tensor)>, &'a PagedAttentionInputMetadata)>,
    ) -> Result<Self::Run<'a>>;
    fn draft_run<'a>(view: &'a DraftPagedView, meta: &'a PagedAttentionInputMetadata) -> Result<Self::Run<'a>>;
    /// `x` is the post-`attn_norm` hidden `(b, s, hidden)` of layer `i`.
    fn forward(
        &self, shared: &Self::Shared, x: &Tensor, start_offsets: &[usize], run: &Self::Run<'_>, i: usize,
    ) -> Result<Tensor>;
}

pub struct PlainLayer<A> {
    pub attn_norm: QRmsNorm,
    pub attn: A,
    pub ffn_norm: QRmsNorm,
    pub mlp: Ffn,
}

pub struct PlainTrunk<A: PlainAttention> {
    pub tok_embeddings: QuantizedEmbedding,
    pub layers: Vec<PlainLayer<A>>,
    pub shared: A::Shared,
    pub norm: QRmsNorm,
    pub output: Arc<dyn QuantMethod>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    pub leading_dense: usize,
}

impl<A: PlainAttention> PlainTrunk<A> {
    pub fn forward(
        &self, x: &Tensor, start_offsets: &[usize], context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let run = A::engine_run(x, metadata)?;
        self.provide_router_tables(x);
        let layer_in = self.tok_embeddings.forward(x)?.to_dtype(DType::F32)?;
        let final_layer_in = self.run_layers(layer_in, start_offsets, &run)?;
        self.decode_logits(&final_layer_in, context_lens)
    }

    /// The drafter's one-token decode beside the engine: state is `&self`, the K/V row to the view's reserved tail.
    pub fn forward_with_cache(
        &self, x: &Tensor, start_offsets: &[usize], context_lens: Vec<(usize, usize)>, view: &DraftPagedView,
        _private: &mut [Option<crate::state_slots::RecurrentState>],
    ) -> Result<Tensor> {
        let meta = view.decode_metadata(start_offsets[0], &self.device)?;
        let run = A::draft_run(view, &meta)?;
        let layer_in = self.tok_embeddings.forward(x)?.to_dtype(DType::F32)?;
        let final_layer_in = self.run_layers(layer_in, start_offsets, &run)?;
        self.decode_logits(&final_layer_in, context_lens)
    }

    fn run_layers(&self, mut layer_in: Tensor, start_offsets: &[usize], run: &A::Run<'_>) -> Result<Tensor> {
        for (i, layer) in self.layers.iter().enumerate() {
            // The late tap, for prediction only: the residual entering layer i, before its attention.
            if A::LATE_TAP && i >= self.leading_dense {
                crate::moe::offer_late_tap(i - self.leading_dense, &layer_in, &|x| Ok(x.clone()));
            }
            let x = layer_in;
            let residual = &x;
            let x = layer.attn_norm.forward(&x)?;
            let attn = layer.attn.forward(&self.shared, &x, start_offsets, run, i)?;
            let x = (attn + residual)?;

            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            layer_in = layer.mlp.forward(&x, Some(residual))?;
        }
        Ok(layer_in)
    }

    /// The router tables, every MoE layer in ordinal order: `ffn_norm`, the gate and its `exp_probs_b` bias.
    fn provide_router_tables(&self, x: &Tensor) {
        let rows = self.layers.iter().skip(self.leading_dense).map(|l| (&l.ffn_norm, l.mlp.moe()));
        crate::moe::provide_router_rows(x, rows);
    }

    fn decode_logits(&self, layer_in: &Tensor, context_lens: Vec<(usize, usize)>) -> Result<Tensor> {
        let x = self.norm.forward(layer_in)?;
        let x = extract_logits(&x, context_lens)?;
        MatMul.qmethod_matmul(&x.contiguous()?, &*self.output)
    }
}

impl<A: PlainAttention> GgufArch for PlainTrunk<A> {
    fn forward(
        &self, input_ids: &Tensor, seqlen_offsets: &[usize], context_lens: Vec<(usize, usize)>,
        paged_attn_meta: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        Self::forward(self, input_ids, seqlen_offsets, context_lens, paged_attn_meta)
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn num_layers(&self) -> usize {
        self.cache.normal().0.len()
    }
}

impl<A: PlainAttention> crate::architecture::Architecture for PlainTrunk<A> {
    const ROUTER_TABLES: bool = true;
    const LATE_TAP: bool = A::LATE_TAP;
    fn draftable(model: Arc<Self>) -> Option<Arc<dyn crate::draft::DraftableModel>> {
        A::DRAFTS.then(|| model as Arc<dyn crate::draft::DraftableModel>)
    }
}

crate::draftable!(PlainTrunk<A>, A);
