//! The model's file, read once: the architecture's declaration and the tensor table, for the plan and the dispatcher.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use candle_core::quantized::gguf_file::{Content, TensorInfo};
use candle_core::quantized::GgmlDType;
use mistralrs_core::gguf::{declare_arch, ArchDeclaration, GgufReader, LayerFfn};
use oscar_offload::io::ShardId;
use oscar_offload::segments::PAGE;

/// BF16 for every model; the KV pool's install refuses a block the engine lays out at another width.
const KV_BYTES_PER_ELEMENT: u64 = 2;

/// One `(layer, expert, projection)` block's extent on disk (`shard_id` 0 when unsharded).
#[derive(Debug, Clone, Copy)]
pub struct ExpertExtent {
    pub shard_id: ShardId,
    pub file_offset: u64,
    pub byte_count: usize,
}

/// `Ord` fixes the order of bank containers keyed by projection, which must be reproducible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ProjectionKind {
    Gate,
    Up,
    Down,
}

impl ProjectionKind {
    pub fn gguf_suffix(self) -> &'static str {
        match self {
            Self::Gate => "ffn_gate_exps.weight",
            Self::Up => "ffn_up_exps.weight",
            Self::Down => "ffn_down_exps.weight",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
        }
    }

    /// Position in [`Self::ALL`].
    pub const fn index(self) -> usize {
        match self {
            Self::Gate => 0,
            Self::Up => 1,
            Self::Down => 2,
        }
    }

    pub const ALL: [ProjectionKind; 3] = [ProjectionKind::Gate, ProjectionKind::Up, ProjectionKind::Down];
}

#[derive(Debug, Clone)]
pub struct ProjectionLayout {
    pub dtype: GgmlDType,
    /// One expert's weight is `n` rows × `k` columns.
    pub n: usize,
    pub k: usize,
    pub bytes_per_expert: usize,
    /// By expert id, each offset absolute in its file.
    pub experts: Vec<ExpertExtent>,
}

pub fn layout_host_bytes(layers: usize, experts: usize) -> usize {
    layers
        * ProjectionKind::ALL.len()
        * (experts * std::mem::size_of::<ExpertExtent>() + std::mem::size_of::<ProjectionLayout>())
        + oscar_offload::machine::hash_map_bytes::<(usize, ProjectionKind), ProjectionLayout>(
            layers * ProjectionKind::ALL.len(),
        )
}

#[derive(Debug, Clone, Default)]
pub struct GgufRoutedLayout {
    /// The header's `block_count` less the dense lead and trailing MTP blocks; `projections` is keyed by MoE ordinal.
    pub block_count: usize,
    /// Trailing MTP prediction blocks (`*.nextn_predict_layers`), never loaded.
    pub nextn_blocks: usize,
    pub num_experts: usize,
    /// `None` where the header omits `expert_used_count`: the decode pacer refuses to run without it.
    pub experts_per_token: Option<usize>,
    pub projections: HashMap<(usize, ProjectionKind), ProjectionLayout>,
    /// The experts' activation as the header declares it, the variant the kernel table keys on.
    pub activation: oscar_kernels::ExpertActivation,
}

/// One (projection, dtype) class of routed experts, one bank: its layers in dispatch order, one expert's padded bytes.
#[derive(Debug, Clone)]
pub struct ExpertClass {
    /// 0 gate, 1 up, 2 down.
    pub projection: usize,
    pub dtype: GgmlDType,
    pub layers: Vec<usize>,
    pub unit: u64,
}

impl ExpertClass {
    /// How many of the class's layers lie among the model's first `sealed`, and the rest.
    pub fn split(&self, sealed: usize) -> (usize, usize) {
        let s = self.layers.partition_point(|&l| l < sealed);
        (s, self.layers.len() - s)
    }

    /// The model's first `sealed` layers at `experts` slots each, the rest carriers at `residual`.
    pub fn slots(&self, sealed: usize, experts: usize, residual: usize) -> usize {
        let (s, c) = self.split(sealed);
        s * experts + c * residual
    }

    /// The whole segments holding `slots` at `segment_slots` per layer, and one segment's slots.
    pub fn segments(&self, slots: usize, segment_slots: usize) -> (usize, usize) {
        let per_segment = segment_slots.max(1) * self.layers.len().max(1);
        (slots.div_ceil(per_segment), per_segment)
    }

    /// [`Self::slots`] in whole segments once the segment size is fixed.
    pub fn held(&self, sealed: usize, experts: usize, residual: usize, segment_slots: Option<usize>) -> usize {
        let slots = self.slots(sealed, experts, residual);
        segment_slots.map_or(slots, |k| {
            let (segments, per_segment) = self.segments(slots, k);
            segments * per_segment
        })
    }

    pub fn bytes(&self, sealed: usize, experts: usize, residual: usize, segment_slots: Option<usize>) -> u64 {
        self.held(sealed, experts, residual, segment_slots) as u64 * self.unit
    }

    pub fn label(&self) -> String {
        format!("{}-{}", ["gate", "up", "down"][self.projection], format!("{:?}", self.dtype).to_lowercase())
    }
}

/// The experts' activation as the header declares it, the variant the kernel table keys on: SiTU-GLU where a
/// `situ_beta` is declared, clamped SwiGLU where a `swiglu_clamp_exp` is, else SiLU; the parameters are the model's.
fn expert_activation<'a>(keys: impl IntoIterator<Item = &'a str>) -> oscar_kernels::ExpertActivation {
    use oscar_kernels::ExpertActivation as A;
    match keys.into_iter().find(|k| k.ends_with(".activation.situ_beta") || k.ends_with(".swiglu_clamp_exp")) {
        Some(k) if k.ends_with("situ_beta") => A::SituGlu { beta_gate: 0.0, beta_up: 0.0 },
        Some(_) => A::SiluClamp { limit: 0.0 },
        None => A::Silu,
    }
}

/// Every tensor from one walk of the tensor tables: the routed stacks as per-expert extents, the rest the fixed side
/// as the pool maps it, each loading tensor in its own whole pages.
#[derive(Debug, Clone, Default)]
pub struct TensorTable {
    pub routed: GgufRoutedLayout,
    pub resident_bytes: u64,
    pub largest_bytes: u64,
    /// Never resident (its rows page in through the PLE store), so apart from `resident_bytes`.
    pub ple_table_bytes: u64,
    pub ple_table: Option<PleTable>,
    pub total_params: u64,
    /// `output.weight`'s elements where no draft head ships (what a drafter would requantise), else 0.
    pub output_head_elems: u64,
}

/// Where the n-gram row-gather table lies, `offset` absolute in its shard's file.
#[derive(Debug, Clone)]
pub struct PleTable {
    pub shard: ShardId,
    pub offset: u64,
    pub dims: Vec<usize>,
    pub dtype: GgmlDType,
}

fn tensor_bytes(info: &TensorInfo) -> u64 {
    let elements: u64 = info.shape.dims().iter().product::<usize>() as u64;
    let block_size = info.ggml_dtype.block_size() as u64;
    // An F16/BF16 weight is kept dense at F32 (`load::qmm`).
    let type_size = match info.ggml_dtype {
        GgmlDType::F16 | GgmlDType::BF16 => 4,
        d => d.type_size() as u64,
    };
    if block_size <= 1 || !elements.is_multiple_of(block_size) {
        elements.saturating_mul(type_size)
    } else {
        (elements / block_size).saturating_mul(type_size)
    }
}

/// The `N` of a `blk.N.…` name.
pub fn block_index(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split_once('.')?.0.parse().ok()
}

/// Its metadata and tensor table, no tensor data.
pub fn header(path: &Path) -> Result<Content> {
    Ok(Content::read(&mut BufReader::new(File::open(path)?))?)
}

/// An expert kernel for `dtype`, and the fused SiLU gate+up kernel where `fused_gate_or_up`.
pub fn bank_runs(dtype: GgmlDType, gate_or_up: bool, activation: oscar_kernels::ExpertActivation) -> bool {
    use oscar_kernels::{has_fused_gateup_kernel, QkkMoeDType};
    QkkMoeDType::from_ggml(dtype).is_some_and(|d| !gate_or_up || has_fused_gateup_kernel(d, activation))
}

/// A tensor in several files loads once, the first by name in file order, so a overlay's shadowed copies cost nothing;
/// a routed block's stack in two files is refused.
fn table_of(shards: &[(ShardId, &Content)], declaration: &ArchDeclaration) -> Result<TensorTable> {
    let num_experts = declaration.experts.count as usize;
    ensure!(num_experts > 0, "the architecture declares no experts");
    let leading_dense =
        declaration.layers.iter().position(|l| l.ffn == LayerFfn::Routed).unwrap_or(declaration.layers.len());
    let (nextn, block_count) = (declaration.nextn_layers(), declaration.moe_layers());
    ensure!(
        block_count > 0,
        "no MoE layers: blocks={}, leading_dense={leading_dense}, nextn={nextn}",
        declaration.layers.len()
    );
    let has_draft_head = shards.iter().any(|(_, c)| c.tensor_infos.contains_key("output_draft.weight"));
    let (mut t, mut seen) = (TensorTable::default(), HashSet::new());
    let mut projections = HashMap::with_capacity(block_count * 3);
    for &(shard, content) in shards {
        for (name, info) in &content.tensor_infos {
            let block = block_index(name);
            let routed = ProjectionKind::ALL.into_iter().find(|k| name.ends_with(k.gguf_suffix()));
            // By MoE ordinal: the dense lead's and the trailing MTP blocks' stacks are no routed layer's.
            let layer = block.and_then(|b| b.checked_sub(leading_dense)).filter(|&l| l < block_count);
            if let (Some(projection), Some(layer)) = (routed, layer) {
                let offset = content.tensor_data_offset;
                let layout = build_projection_layout(offset, layer, projection, num_experts, info, shard)?;
                if let Some(prev) = projections.insert((layer, projection), layout) {
                    bail!("tensor {name} present in multiple shards ({} and {shard})", prev.experts[0].shard_id);
                }
            }
            if !seen.insert(name.as_str()) {
                continue; // shadowed by an earlier file
            }
            let elems = info.shape.dims().iter().product::<usize>() as u64;
            t.total_params = t.total_params.saturating_add(elems);
            // Never loaded: a trailing draft block, the sparse-attention indexer (the dense fallback), a routed stack.
            if block.is_some_and(|b| b >= declaration.trunk_layers()) || name.contains(".indexer.") || routed.is_some()
            {
                continue;
            }
            let bytes = tensor_bytes(info);
            if name == "per_layer_token_embd.weight" {
                let (offset, dims) = (content.tensor_data_offset + info.offset, info.shape.dims().to_vec());
                (t.ple_table_bytes, t.ple_table) =
                    (bytes, Some(PleTable { shard, offset, dims, dtype: info.ggml_dtype }));
                continue;
            }
            let paged = bytes.next_multiple_of(PAGE as u64);
            (t.resident_bytes, t.largest_bytes) = (t.resident_bytes.saturating_add(paged), t.largest_bytes.max(paged));
            if name == "output.weight" && !has_draft_head {
                t.output_head_elems = elems;
            }
        }
    }
    let wanted = (0..block_count).flat_map(|l| ProjectionKind::ALL.map(|p| (l, p)));
    if let Some((layer, p)) = wanted.into_iter().find(|key| !projections.contains_key(key)) {
        bail!("missing tensor blk.{}.{} across all shards", layer + leading_dense, p.gguf_suffix());
    }
    let experts_per_token = (declaration.experts.per_token > 0).then_some(declaration.experts.per_token as usize);
    let activation = expert_activation(shards.iter().flat_map(|(_, c)| c.metadata.keys().map(String::as_str)));
    t.routed =
        GgufRoutedLayout { block_count, nextn_blocks: nextn, num_experts, experts_per_token, projections, activation };
    Ok(t)
}

fn build_projection_layout(
    tensor_data_offset: u64, layer: usize, projection: ProjectionKind, num_experts: usize, info: &TensorInfo,
    shard_id: ShardId,
) -> Result<ProjectionLayout> {
    let dims = info.shape.dims();
    let suffix = projection.gguf_suffix();
    ensure!(dims.len() == 3, "blk.{layer}.{suffix} has rank {} (expected 3 = (num_experts, n, k))", dims.len());
    let (e, n, k) = (dims[0], dims[1], dims[2]);
    ensure!(e == num_experts, "blk.{layer}.{suffix}: tensor reports {e} experts, metadata says {num_experts}");
    // A row must be whole blocks, or an expert's extent is not a byte range.
    let block_size = info.ggml_dtype.block_size();
    ensure!(k % block_size == 0, "blk.{layer}.{suffix}: k={k} not divisible by block_size={block_size}");
    // The stack's bytes split e ways: the same expression the plan prices.
    let bytes_per_expert = tensor_bytes(info) as usize / e;
    let stack_offset = tensor_data_offset + info.offset;
    let experts = (0..num_experts)
        .map(|expert_idx| ExpertExtent {
            shard_id,
            file_offset: stack_offset + (expert_idx as u64) * (bytes_per_expert as u64),
            byte_count: bytes_per_expert,
        })
        .collect();
    Ok(ProjectionLayout { dtype: info.ggml_dtype, n, k, bytes_per_expert, experts })
}

/// A dtype the store cannot read row-wise is a loud refusal, never a silent zero embedding.
pub fn install_ple_store<P: AsRef<Path>>(paths: &[P], t: &PleTable) -> Result<()> {
    const TABLE: &str = "per_layer_token_embd.weight";
    ensure!(t.dims.len() == 2, "{TABLE}: rank {} (expected 2 = (rows, head_dim))", t.dims.len());
    ensure!(t.dtype == GgmlDType::Q8_0, "{TABLE}: dtype {:?} - the row-gather store reads Q8_0 rows only", t.dtype);
    let path = paths[t.shard as usize].as_ref();
    let store = crate::ple::RowGatherStore::open_q8_0(path, t.offset, t.dims[1], t.dims[0] as u64)
        .map_err(|e| anyhow::anyhow!("{TABLE}: {e}"))?;
    crate::ple::install_store(std::sync::Arc::new(store));
    Ok(())
}

#[derive(Default)]
pub struct ModelLayoutStats {
    pub block_count: usize,
    pub num_experts: usize,
    /// A hot-budget unit: one slot per dispatch layer, an expert's padded gate, up and down bytes in each.
    pub bytes_per_hb_unit: u64,
    /// The unit's terms by dispatch layer, gate, up and down: layers may differ in dtype.
    pub expert_bytes_per_layer: Vec<[u64; 3]>,
    pub non_expert_bytes: u64,
    /// As the paged cache stores them, over the trunk's KV-bearing layers (an absorbed MLA latent row once).
    pub kv_bytes_per_token: u64,
    pub total_params: u64,
    /// `*.embedding_length`, 0 where the header omits it.
    pub hidden_dim: u64,
    /// The most a plan's guarantee may promise; 0 where the header omits it.
    pub context_length: u64,
    /// The slots one token needs resident at once, below which the pool runs degraded; 0 where the header omits it.
    pub experts_per_token: usize,
    pub recurrent_layers: usize,
    pub mla_absorbed: bool,
    /// Gate, up and down per dispatch layer; each (projection, dtype) class is a bank of its own.
    pub expert_dtypes_per_layer: Vec<[GgmlDType; 3]>,
    /// The row-gather store's cache at capacity; the table itself is never resident or charged.
    pub ple_row_cache_bytes: u64,
    pub ple_table: Option<PleTable>,
    pub kv_bearing_layers: u64,
    pub vocab_tokens: u64,
    pub vocab_bytes: u64,
    pub merges: u64,
    pub largest_dense_tensor_bytes: u64,
    /// Unpadded, a read's extent: what a reading thread's bounce buffer is sized by.
    pub largest_expert_projection_bytes: u64,
    /// One sequence's state at the declared layout until the census records it; 0 with no recurrent layer.
    pub sequence_state_bound: u64,
    /// An armed drafter's head requantised at the load where the file ships none: the Q4_K head beside a 4,096-row
    /// chunk at F32 and in both quantisations (six bytes an element bounds the three); 0 where the file ships it.
    pub draft_requant_bytes: u64,
}

impl ModelLayoutStats {
    /// The banks the pool is built of, in its own order: gate, up, down, then by dtype.
    pub fn expert_classes(&self) -> Vec<ExpertClass> {
        let mut classes: Vec<ExpertClass> = Vec::new();
        for (layer, dtypes) in self.expert_dtypes_per_layer.iter().enumerate() {
            let bytes = self.expert_bytes_per_layer.get(layer).copied().unwrap_or([0; 3]);
            for (projection, &dtype) in dtypes.iter().enumerate() {
                match classes.iter_mut().find(|c| c.projection == projection && c.dtype == dtype) {
                    Some(c) => {
                        debug_assert_eq!(c.unit, bytes[projection], "a class's unit is uniform");
                        c.layers.push(layer);
                    }
                    None => {
                        classes.push(ExpertClass { projection, dtype, layers: vec![layer], unit: bytes[projection] })
                    }
                }
            }
        }
        classes.sort_by_key(|c| (c.projection, c.dtype as u32));
        classes
    }
}

/// The routed half as the pool holds it, each expert's projection in whole pages, in dispatch order.
fn expert_charge(routed: &GgufRoutedLayout) -> ModelLayoutStats {
    let mut c = ModelLayoutStats {
        expert_bytes_per_layer: vec![[0; 3]; routed.block_count],
        expert_dtypes_per_layer: vec![[GgmlDType::Q4K; 3]; routed.block_count],
        ..Default::default()
    };
    for ((layer, projection), p) in &routed.projections {
        let padded = (p.bytes_per_expert as u64).next_multiple_of(PAGE as u64);
        c.bytes_per_hb_unit = c.bytes_per_hb_unit.saturating_add(padded);
        c.largest_expert_projection_bytes = c.largest_expert_projection_bytes.max(p.bytes_per_expert as u64);
        c.expert_bytes_per_layer[*layer][projection.index()] = padded;
        c.expert_dtypes_per_layer[*layer][projection.index()] = p.dtype;
    }
    c
}

/// An armed drafter's head as Q4_K where the file ships none (the download writes one), a fixed-side matrix.
fn draft_head_bytes(head_elems: u64) -> u64 {
    if head_elems == 0 || !crate::draft::self_draft() {
        return 0;
    }
    let q4k = GgmlDType::Q4K;
    head_elems / q4k.block_size() as u64 * q4k.type_size() as u64
}

/// From the header alone, the reading the loader builds from, so the plan charges what will be built.
pub fn read_declaration(gguf_paths: &[PathBuf]) -> Result<ArchDeclaration> {
    // The declarations live beside the factories, and a planner may run before any load has registered them.
    crate::register();
    let mut files = Vec::with_capacity(gguf_paths.len());
    for path in gguf_paths {
        let file = std::fs::File::open(path).with_context(|| format!("opening GGUF {}", path.display()))?;
        files.push(Box::new(std::io::BufReader::new(file)) as GgufReader);
    }
    let mut readers: Vec<&mut GgufReader> = files.iter_mut().collect();
    let content = mistralrs_core::gguf::Content::from_readers(&mut readers)
        .map_err(|e| anyhow::anyhow!("GGUF header read: {e}"))?;
    declare_arch(&content).map_err(|e| anyhow::anyhow!("{e}"))
}

/// The geometry the plan prices and the routed layout the dispatcher keeps, from one walk of the tensor tables; each
/// extent's `shard_id` is its file's index in `gguf_paths`.
pub fn read_model_layout_stats(gguf_paths: &[PathBuf]) -> Result<(ModelLayoutStats, GgufRoutedLayout)> {
    let declaration = read_declaration(gguf_paths)?;
    if declaration.experts.count == 0 {
        bail!("architecture declares no experts - not a MoE model");
    }
    ensure!(!gguf_paths.is_empty(), "no shards supplied");
    let most = ShardId::MAX as usize + 1;
    ensure!(gguf_paths.len() <= most, "too many shards: {} (max {most})", gguf_paths.len());
    let contents: Vec<Content> = gguf_paths.iter().map(|p| header(p)).collect::<Result<_>>()?;
    let shards: Vec<(ShardId, &Content)> = contents.iter().enumerate().map(|(i, c)| (i as ShardId, c)).collect();
    let table = table_of(&shards, &declaration)?;
    // By the pool's own arithmetic, the number the census's first take records.
    let sequence_state_bound = declaration
        .recurrent_state_layout(candle_core::DType::F32)
        .map_or(0, |layout| crate::state_slots::runs_of(&layout).1 as u64);
    // The drafter's head is the plan's charge, not the file's: an object of its own, in whole pages like every other.
    let draft_head = draft_head_bytes(table.output_head_elems).next_multiple_of(PAGE as u64);
    let stats = ModelLayoutStats {
        block_count: declaration.moe_layers(),
        num_experts: declaration.experts.count as usize,
        non_expert_bytes: table.resident_bytes.saturating_add(draft_head),
        kv_bytes_per_token: declaration.kv_bytes_per_token(KV_BYTES_PER_ELEMENT),
        total_params: table.total_params,
        hidden_dim: declaration.hidden,
        context_length: declaration.context_length,
        recurrent_layers: declaration.recurrent_layers(),
        experts_per_token: declaration.experts.per_token as usize,
        mla_absorbed: declaration.attention.kv_lora_rank.is_some(),
        ple_row_cache_bytes: if table.ple_table_bytes > 0 {
            // The cache's rows dequantised to F32, plus about 96 B of map and key per row.
            (crate::ple::RowGatherStore::CACHE_ROWS as u64)
                .saturating_mul(declaration.ple_row_width.saturating_mul(4).saturating_add(96))
        } else {
            0
        },
        ple_table: table.ple_table,
        kv_bearing_layers: declaration.kv_bearing_layers() as u64,
        vocab_tokens: declaration.vocab.tokens,
        vocab_bytes: declaration.vocab.bytes,
        merges: declaration.vocab.merges,
        largest_dense_tensor_bytes: table.largest_bytes.max(draft_head),
        sequence_state_bound,
        draft_requant_bytes: match draft_head_bytes(table.output_head_elems) {
            0 => 0,
            q4k => q4k + 4096 * (table.output_head_elems / declaration.vocab.tokens.max(1)) * 6,
        },
        ..expert_charge(&table.routed)
    };
    Ok((stats, table.routed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mistralrs_core::gguf::{AttentionDecl, ExpertDecl, LayerAttention, LayerDecl, LayerRole};

    /// The variant alone names the kernel row, and no key declared keeps SiLU; the layout on a synthetic file, real
    /// ones being check-model's.
    #[test]
    fn the_experts_activation_and_the_routed_layout_are_read_from_the_headers() {
        use candle_core::quantized::{gguf_file, QTensor};
        use oscar_kernels::ExpertActivation as A;
        let plain =
            ["qwen35moe.attention.head_count", "some-future-arch.activation.something_else", "general.architecture"];
        assert!(matches!(expert_activation(plain), A::Silu));
        assert!(matches!(expert_activation(["kimi-k3.activation.situ_beta"]), A::SituGlu { .. }));
        assert!(matches!(expert_activation(["glm5next.swiglu_clamp_exp"]), A::SiluClamp { .. }));
        let declaration = ArchDeclaration {
            layers: [LayerFfn::Dense, LayerFfn::Routed, LayerFfn::Routed]
                .map(|ffn| LayerDecl { attention: LayerAttention::Kv { heads: 1 }, ffn, role: LayerRole::Trunk })
                .to_vec(),
            experts: ExpertDecl { count: 4, per_token: 2 },
            ..ArchDeclaration::default()
        };
        // One block's three stacks: `experts` × (2, 64) Q8_0, four 34-byte blocks an expert, or the rows flat.
        let block = |l: usize, experts: usize, rank3: bool| -> Vec<(String, QTensor)> {
            let shape = if rank3 { vec![experts, 2, 64] } else { vec![experts * 2, 64] };
            let zeros = candle_core::Tensor::zeros(shape, candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
            ProjectionKind::ALL
                .into_iter()
                .map(|p| (format!("blk.{l}.{}", p.gguf_suffix()), QTensor::quantize(&zeros, GgmlDType::Q8_0).unwrap()))
                .collect()
        };
        let both = |a: Vec<(String, QTensor)>, b: Vec<(String, QTensor)>| a.into_iter().chain(b).collect::<Vec<_>>();
        let shard = |tensors: Vec<(String, QTensor)>| {
            let mut buf = std::io::Cursor::new(Vec::new());
            let tensors: Vec<(&str, &QTensor)> = tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
            gguf_file::write(&mut buf, &[] as &[(&str, &gguf_file::Value)], &tensors).unwrap();
            buf.set_position(0);
            Content::read(&mut buf).unwrap()
        };
        let (first, second) = (shard(block(1, 4, true)), shard(block(2, 4, true)));
        let layout = table_of(&[(0, &first), (1, &second)], &declaration).unwrap().routed;
        assert_eq!((layout.block_count, layout.num_experts, layout.experts_per_token), (2, 4, Some(2)));
        for (layer, shard_id) in [(0usize, 0 as ShardId), (1, 1)] {
            for projection in ProjectionKind::ALL {
                let p = &layout.projections[&(layer, projection)];
                assert_eq!((p.dtype, p.n, p.k, p.bytes_per_expert), (GgmlDType::Q8_0, 2, 64, 4 * 34));
                assert!(p.experts.iter().all(|e| e.shard_id == shard_id && e.byte_count == p.bytes_per_expert));
                assert!(p.experts.windows(2).all(|w| w[1].file_offset == w[0].file_offset + w[0].byte_count as u64));
            }
        }
        let refused = |shards: &[(ShardId, &Content)]| table_of(shards, &declaration).unwrap_err().to_string();
        let twice = shard(both(block(1, 4, true), block(2, 4, true)));
        assert!(refused(&[(0, &first), (1, &twice)]).contains("present in multiple shards"));
        let flat = shard(both(block(1, 4, false), block(2, 4, true)));
        assert!(refused(&[(0, &flat)]).contains("has rank 2"));
        let five = shard(both(block(1, 5, true), block(2, 4, true)));
        assert!(refused(&[(0, &five)]).contains("reports 5 experts"));
    }

    /// Absorbed MLA charges its latent row once; the expert charge is the layout's extents in whole pages.
    #[test]
    fn the_kv_charge_matches_audited_storage_rates_and_the_expert_charge_the_layouts_whole_pages() {
        use mistralrs_core::paged_attention::ModelConfigLike;
        let kv = |heads| LayerAttention::Kv { heads };
        // The blocks in file order, the last `nextn` of them draft blocks.
        let declaration = |key, value, lora, attn: Vec<LayerAttention>, nextn: usize| {
            let trunk = attn.len() - nextn;
            ArchDeclaration {
                layers: attn
                    .into_iter()
                    .enumerate()
                    .map(|(i, attention)| LayerDecl {
                        attention,
                        ffn: LayerFfn::Routed,
                        role: if i < trunk { LayerRole::Trunk } else { LayerRole::Nextn },
                    })
                    .collect(),
                attention: AttentionDecl { head_count: 64, key_len: key, value_len: value, kv_lora_rank: lora },
                ..ArchDeclaration::default()
            }
        };
        let hybrid = |blocks: usize, interval: usize, heads: u64| -> Vec<LayerAttention> {
            (0..blocks).map(|i| if (i + 1) % interval == 0 { kv(heads) } else { LayerAttention::Recurrent }).collect()
        };
        let mut k3 = hybrid(93, 4, 1);
        k3[92] = kv(1);
        // (model, declaration, what the plan charges a token in KB)
        let cases: Vec<(&str, ArchDeclaration, f64)> = vec![
            // GLM-5.2: absorbed MLA (K 576 = 512 latent + 64 RoPE, V 512), BF16; the last of 79 blocks an uncached MTP.
            ("glm-5.2", declaration(576, 512, Some(512), vec![kv(1); 79], 1), 87.75),
            // 235B: plain GQA, BF16, every one of 94 layers caches K and V.
            ("qwen3-235b", declaration(128, 128, None, vec![kv(4); 94], 0), 188.0),
            // 397B: GQA + GDN, BF16 KV (the F32 residual cast at append), interval 4: 15 of 60 layers hold KV.
            ("qwen3.5-397b", declaration(256, 256, None, hybrid(60, 4, 2), 0), 30.0),
            // K3: absorbed MLA on 24 of 93 layers; `value_length` is the writer's default 74, stored V the 512 latent.
            ("kimi-k3", declaration(576, 74, Some(512), k3.clone(), 0), 27.0),
            // K3.1 writes `value_length` 512: under the latent rank the charge is the same.
            ("kimi-k31", declaration(576, 512, Some(512), k3, 0), 27.0),
        ];
        for (model, d, want_paged) in cases {
            let charged = d.kv_bytes_per_token(KV_BYTES_PER_ELEMENT);
            assert_eq!(charged as f64 / 1024.0, want_paged, "{model}: the charged rate diverged from the audit");
            // The mistral.rs fork sizes its blocks from this declaration: the pool allocates what the plan charges.
            let elements = d.kv_cache_elements_per_token() as u64 * d.num_kv_bearing_layers() as u64;
            assert_eq!(elements * KV_BYTES_PER_ELEMENT, charged, "{model}: the block pool's bytes a token");
        }
        // The expert charge is the layout's extents in whole pages.
        let (page, q4, q3) = (oscar_offload::segments::PAGE, GgmlDType::Q4K, GgmlDType::Q3K);
        let projection =
            |bytes: usize, dtype| ProjectionLayout { dtype, n: 1, k: bytes, bytes_per_expert: bytes, experts: vec![] };
        let mut routed = GgufRoutedLayout { block_count: 2, ..GgufRoutedLayout::default() };
        // Layer 0 takes whole pages exactly; layer 1, a byte past one page, is charged two, as the pool maps it.
        for (layer, bytes, dtype) in [(0, page, q4), (1, page + 1, q3)] {
            for proj in ProjectionKind::ALL {
                routed.projections.insert((layer, proj), projection(bytes, dtype));
            }
        }
        let (c, page) = (expert_charge(&routed), page as u64);
        assert_eq!(c.expert_bytes_per_layer, [[page; 3], [2 * page; 3]], "each projection in whole pages");
        assert_eq!(c.expert_dtypes_per_layer, [[q4; 3], [q3; 3]], "a layer's bank classes are the layout's dtypes");
        assert_eq!(c.bytes_per_hb_unit, 9 * page, "one slot per layer, every projection");
        assert_eq!(c.largest_expert_projection_bytes, page + 1, "the read extent is the tensor's bytes, unpadded");
    }
}
