//! The routed layer and its dispatcher, installed once by [`set_global_moe_dispatcher`] before any routed stack loads.

use std::io::{Read, Seek};
use std::sync::{Arc, OnceLock};

use candle_core::quantized::{GgmlDType, QMatMul};
use candle_core::{DType, Device, Result, Tensor};
use mistralrs_core::gguf::Content;
use mistralrs_core::layers::QRmsNorm;
use mistralrs_quant::{MatMul, QuantMethod};
use oscar_kernels::{has_fused_gateup_kernel, metal_topk_biased_sigmoid, metal_topk_softmax, ExpertActivation};

use crate::ffn::{kept_router_gate, moe_finalize, Finalized, ForwardShape, GatedMlp, RoutedExperts, Stream};
use crate::load::gate_matvec_or_forward;

/// Per layer, a drafted token's router candidates past its top-k, best first, for the eval's hint score.
#[derive(Clone, Debug, Default)]
pub struct RunnerUps {
    pub extras: Vec<Vec<u16>>,
}

/// Prediction input only: none of it may perturb what the exact forward reads.
pub trait MoeLookahead: Send + Sync {
    /// Per MoE layer `(norm before the router, gate, selection bias)`, F32 on the CPU.
    fn provide_router_tables(&self, _tables: Vec<(Tensor, Tensor, Option<Tensor>)>) {}

    /// Asked first: building those tables costs as much as holding them.
    fn wants_router_tables(&self) -> bool {
        false
    }

    /// The late tap: `residual(trunk)` enters this layer before its attention, made only where the dispatcher reads it.
    fn offer_layer_residual(
        &self, _moe_ordinal: usize, _trunk: &Tensor, _residual: &dyn Fn(&Tensor) -> Result<Tensor>,
    ) {
    }
}

/// One forward for every architecture: the layer a [`RoutedLayer`], `xs` always at the model width.
pub trait MoeKernelDispatcher: MoeLookahead {
    /// Given a `Stream`, the block's output is added to the residual and normalised for the next layer.
    fn forward(&self, layer: &RoutedLayer, xs: &Tensor, stream: Option<Stream<'_>>) -> Result<Finalized>;
}

static GLOBAL_DISPATCHER: OnceLock<Arc<dyn MoeKernelDispatcher>> = OnceLock::new();

/// First caller wins.
pub fn set_global_moe_dispatcher(dispatcher: Arc<dyn MoeKernelDispatcher>) {
    let _ = GLOBAL_DISPATCHER.set(dispatcher);
}

pub fn global_moe_dispatcher() -> Option<&'static Arc<dyn MoeKernelDispatcher>> {
    GLOBAL_DISPATCHER.get()
}

/// The late tap to the dispatcher, where one is installed.
pub(crate) fn offer_late_tap(moe_ordinal: usize, trunk: &Tensor, residual: &dyn Fn(&Tensor) -> Result<Tensor>) {
    if let Some(d) = global_moe_dispatcher() {
        d.offer_layer_residual(moe_ordinal, trunk, residual);
    }
}

/// Top-k over the gate's logits: a softmax (the Qwen families) or the DeepSeek family's biased sigmoid (`noaux_tc`),
/// whose bias moves the selection only; `sum_eps` is the reference's guard in the renormalising denominator.
pub(crate) enum Router {
    Softmax { gate: QMatMul, topk: usize },
    Biased { gate: QMatMul, bias: Tensor, topk: usize, norm: bool, scale: f64, sum_eps: f64 },
}

impl Router {
    pub(crate) fn softmax<R: Seek + Read>(
        ct: &mut Content<'_, R>, prefix: &str, device: &Device, topk: usize, requant_to: Option<GgmlDType>,
    ) -> Result<Self> {
        let gate = kept_router_gate(ct, prefix, device, requant_to)?;
        Ok(Self::Softmax { gate, topk })
    }

    pub(crate) fn biased<R: Seek + Read>(
        ct: &mut Content<'_, R>, prefix: &str, device: &Device, topk: usize, norm: bool, scale: f64, sum_eps: f64,
    ) -> Result<Self> {
        let gate = kept_router_gate(ct, prefix, device, None)?;
        let bias = crate::load::f32_dequant(ct, &format!("{prefix}.exp_probs_b.bias"), device)?;
        Ok(Self::Biased { gate, bias, topk, norm, scale, sum_eps })
    }

    fn gate(&self) -> &QMatMul {
        match self {
            Self::Softmax { gate, .. } | Self::Biased { gate, .. } => gate,
        }
    }
}

/// All the dispatch seam knows of a model: one routed layer, its stacks the dispatcher's pool's.
pub struct RoutedLayer {
    pub(crate) router: Router,
    /// `(gate, up, down)`.
    pub(crate) stacks: [RoutedExperts; 3],
    /// The routed experts': a wrong one is silent, as the shapes still match.
    pub activation: ExpertActivation,
    /// At SiLU it rides the fused gate-up matvec, the one activation that has it; any other runs unfused.
    pub(crate) shared: Option<GatedMlp>,
    /// The Qwen hybrids' `(1, hidden)` F32, dotted with each token's input under a sigmoid.
    pub(crate) shared_gate: Option<Tensor>,
    /// K3's `(down, norm, up)`: the routed experts run and sum in a latent; the router reads the full width.
    pub(crate) latent: Option<(Arc<dyn QuantMethod>, QRmsNorm, Arc<dyn QuantMethod>)>,
}

impl RoutedLayer {
    /// The stacks (`ffn_{gate,up,down}_exps`), then the shared expert (`ffn_*_shexp`) at `shared`'s activation.
    pub(crate) fn load<R: Seek + Read>(
        ct: &mut Content<'_, R>, prefix: &str, ordinal: usize, device: &Device, router: Router,
        activation: ExpertActivation, shared: Option<ExpertActivation>,
    ) -> Result<Self> {
        if global_moe_dispatcher().is_none() {
            let key = format!("{prefix}.ffn_gate_exps.weight");
            candle_core::bail!("{key}: routed experts served by offload dispatcher, none installed");
        }
        let mut stack =
            |p: &str| RoutedExperts::from_qtensor(ct, prefix, &format!("ffn_{p}_exps.weight"), ordinal, device);
        let stacks = [stack("gate")?, stack("up")?, stack("down")?];
        let shared = shared.map(|act| GatedMlp::load(ct, prefix, "_shexp", device, act)).transpose()?;
        Ok(Self { router, stacks, activation, shared, shared_gate: None, latent: None })
    }

    /// The router's logits.
    pub fn gate_forward(&self, xs: &Tensor) -> Result<Tensor> {
        gate_matvec_or_forward(self.router.gate(), xs)
    }

    pub fn num_experts_per_tok(&self) -> usize {
        match self.router {
            Router::Softmax { topk, .. } | Router::Biased { topk, .. } => topk,
        }
    }

    pub fn norm_topk_prob(&self) -> bool {
        match self.router {
            Router::Softmax { .. } => true,
            Router::Biased { norm, .. } => norm,
        }
    }

    /// A handle clone of the router gate, which the cross-layer predictor applies to an earlier layer's hidden.
    pub fn gate_matmul(&self) -> QMatMul {
        self.router.gate().clone()
    }

    /// `exp_probs_b`: top-k ranks by `sigmoid(logit) + bias`, which a prediction must apply too; `None`: logits rank.
    pub fn gate_selection_bias(&self) -> Option<Tensor> {
        match &self.router {
            Router::Biased { bias, .. } => Some(bias.clone()),
            Router::Softmax { .. } => None,
        }
    }

    /// `(ids U32, weights F32)`: the dispatcher runs the gate itself and must select with exactly this math.
    pub fn route_topk(&self, logits: &Tensor) -> Result<(Tensor, Tensor)> {
        match self.router {
            Router::Biased { ref bias, topk, norm, scale, sum_eps, .. } => {
                let (weights, ids) = metal_topk_biased_sigmoid(logits, bias, topk, norm, scale as f32, sum_eps as f32)?;
                Ok((ids, weights))
            }
            Router::Softmax { topk, .. } => {
                let (weights, ids) = metal_topk_softmax(logits, topk, true)?;
                Ok((ids, weights))
            }
        }
    }

    /// `(gate, up, down)`: the dispatcher's kernels see a layer's shape only through these.
    pub fn expert_tensors(&self) -> (&RoutedExperts, &RoutedExperts, &RoutedExperts) {
        let [gate, up, down] = &self.stacks;
        (gate, up, down)
    }

    /// The layer's place among the routed layers, the dispatcher's table index.
    pub fn dispatch_ordinal(&self) -> usize {
        self.stacks[0].dispatch_ordinal
    }

    /// The kernel wrapper's table answers, never a list here; one row body runs over both stacks, so one dtype.
    pub fn has_fused_gateup(&self) -> bool {
        let [gate, up, _] = &self.stacks;
        gate.dtype == up.dtype && has_fused_gateup_kernel(gate.dtype, self.activation)
    }

    /// The router's input, projected here where the experts run in a latent; no kernel checks the activation width.
    pub fn expert_input(&self, xs: &Tensor) -> Result<Tensor> {
        match &self.latent {
            Some((down, _, _)) => MatMul.qmethod_matmul(xs, &**down)?.to_dtype(DType::F32),
            None => Ok(xs.clone()),
        }
    }

    /// The block's output as the layer's stream, any `Stream` folded in; `routed` is `None` when nothing routed.
    pub fn finalize_routed(
        &self, routed: Option<&Tensor>, xs: &Tensor, stream: Option<Stream<'_>>, shape: ForwardShape,
    ) -> Result<Finalized> {
        // Summed in the latent: one up-projection a token, not one an expert.
        let projected = match (&self.latent, routed) {
            (Some((_, norm, up)), Some(r)) => Some(up.forward(&norm.forward(&r.to_dtype(DType::F32)?)?)?),
            _ => None,
        };
        let routed = projected.as_ref().or(routed);
        let Some(shared) = &self.shared else {
            // Qwen3's: the output plus the residual, no `next_norm` folded, as that architecture applies each norm itself.
            let out = match (routed, stream) {
                (Some(r), Some(s)) => (r.reshape(shape.dims())?.to_dtype(shape.out_dtype)? + s.residual)?,
                (Some(r), None) => r.reshape(shape.dims())?.to_dtype(shape.out_dtype)?,
                (None, Some(s)) => s.residual.to_dtype(shape.out_dtype)?,
                (None, None) => Tensor::zeros(shape.dims(), shape.out_dtype, xs.device())?,
            };
            return Ok(Finalized::stream(out));
        };
        let shared_out = if shared.act == ExpertActivation::Silu { shared.swiglu(xs)? } else { shared.forward(xs)? };
        let gate = self.shared_gate.as_ref().map(|g| (xs, g));
        moe_finalize(routed, &shared_out, gate, stream, shape.dims(), shape.out_dtype)
    }

    /// Through the dispatcher, which every MoE launch installs before its stacks load.
    pub fn forward(&self, xs: &Tensor, stream: Option<Stream<'_>>) -> Result<Finalized> {
        global_moe_dispatcher().expect("routed stacks load only under the dispatcher").forward(self, xs, stream)
    }
}

/// The routed layer by the Qwen hybrids' tensor names: a softmax router and a gated shared expert.
pub(crate) fn load_moe_with_shared<R: Seek + Read>(
    ct: &mut Content<'_, R>, prefix: &str, dispatch_ordinal: usize, device: &Device, hidden_size: usize,
    num_experts_per_tok: usize,
) -> Result<RoutedLayer> {
    // Q4_K so the decode gate rides the quantised matvec; softmax top-k is robust to its small perturbations.
    let router = Router::softmax(ct, prefix, device, num_experts_per_tok, Some(GgmlDType::Q4K))?;
    let silu = ExpertActivation::Silu;
    let mut layer = RoutedLayer::load(ct, prefix, dispatch_ordinal, device, router, silu, Some(silu))?;
    let gate = ct.tensor(&format!("{prefix}.ffn_gate_inp_shexp.weight"), device)?.dequantize(device)?;
    layer.shared_gate = Some(gate.reshape((1, hidden_size))?);
    Ok(layer)
}

pub(crate) enum Ffn {
    Dense(GatedMlp),
    Moe(RoutedLayer),
}

impl Ffn {
    /// The output plus `residual` where given; an MoE folds it into its finalize.
    pub(crate) fn forward(&self, xs: &Tensor, residual: Option<&Tensor>) -> Result<Tensor> {
        match (self, residual) {
            (Self::Dense(m), Some(r)) => m.forward(xs)? + r,
            (Self::Dense(m), None) => m.forward(xs),
            (Self::Moe(m), r) => Ok(m.forward(xs, r.map(Stream::residual))?.stream),
        }
    }

    pub(crate) fn moe(&self) -> Option<&RoutedLayer> {
        match self {
            Self::Moe(m) => Some(m),
            Self::Dense(_) => None,
        }
    }
}

/// `(norm, gate, selection bias)`, the norm and bias on the host: a copy leaves the live norm's dtype cache alone.
pub(crate) fn router_row(norm: &QRmsNorm, layer: &RoutedLayer) -> Result<(Tensor, QMatMul, Option<Tensor>)> {
    let norm = norm.to_device(&Device::Cpu, DType::F32)?.weight_for_dtype(DType::F32)?;
    Ok((norm, layer.gate_matmul(), layer.gate_selection_bias().map(|b| b.to_device(&Device::Cpu)).transpose()?))
}

/// [`provide_router_tables`] over `(the norm before the router, its layer)` rows; a dense layer among them is refused.
pub(crate) fn provide_router_rows<'a>(x: &Tensor, rows: impl Iterator<Item = (&'a QRmsNorm, Option<&'a RoutedLayer>)>) {
    provide_router_tables(x, || {
        rows.map(|(norm, moe)| match moe {
            Some(m) => router_row(norm, m),
            None => candle_core::bail!("non-MoE layer past dense lead"),
        })
        .collect()
    });
}

/// At a decode-shaped forward whose dispatcher wants them, until it holds them; never under a census (one token wide).
pub(crate) fn provide_router_tables(
    x: &Tensor, layers: impl FnOnce() -> Result<Vec<(Tensor, QMatMul, Option<Tensor>)>>,
) {
    let decode = x.dims2().is_ok_and(|(b, s)| b == 1 && s == 1);
    let Some(dispatcher) =
        global_moe_dispatcher().filter(|d| decode && !oscar_offload::census::armed() && d.wants_router_tables())
    else {
        return;
    };
    let build = || -> Result<Vec<(Tensor, Tensor, Option<Tensor>)>> {
        let host = |gate: &QMatMul| match gate {
            QMatMul::QTensor(q) => q.dequantize(&Device::Cpu),
            QMatMul::Tensor(t) | QMatMul::TensorF16(t) => t.to_device(&Device::Cpu),
        };
        layers()?.into_iter().map(|(norm, gate, bias)| Ok((norm, host(&gate)?, bias))).collect()
    };
    match build() {
        Ok(t) if !t.is_empty() => dispatcher.provide_router_tables(t),
        Ok(_) => panic!("lookahead router tables empty: model names no MoE layer"),
        Err(e) => panic!("lookahead router tables build failed ({e})"),
    }
}
