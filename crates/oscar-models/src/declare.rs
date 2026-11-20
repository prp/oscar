//! The shared parts of every architecture's [`ArchDeclaration`], the one source for the loader, the caches and the plan.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::quantized::gguf_file::Value;
use candle_core::{DType, Device, Result};
use mistralrs_core::gguf::{
    ArchDeclaration, AttentionDecl, ExpertDecl, LayerAttention, LayerDecl, LayerFfn, LayerRole, RecurrentDecl,
};
use mistralrs_core::kv_cache::{
    EitherCache, HybridCache, HybridCacheConfig, HybridLayerType, RecurrentLayerConfig, RecurrentStateDtypes,
};

/// An integral GGUF value of any width or sign: per-layer arrays are I32, which candle's `to_u64` rejects.
pub fn any_int(v: &Value) -> Option<i64> {
    if let Ok(u) = v.to_u64() {
        return i64::try_from(u).ok();
    }
    match v {
        Value::I8(n) => Some(i64::from(*n)),
        Value::I16(n) => Some(i64::from(*n)),
        Value::I32(n) => Some(i64::from(*n)),
        Value::I64(n) => Some(*n),
        _ => None,
    }
}

/// One architecture's header keys, each required unless it states a default: a silent default loads garbage.
pub(crate) struct Meta<'a> {
    pub md: &'a HashMap<String, Value>,
    pub arch: &'static str,
}

impl<'a> Meta<'a> {
    /// The header of a file whose `general.architecture` is one of `accepted`.
    pub fn of(md: &'a HashMap<String, Value>, accepted: &[&'static str]) -> Result<Self> {
        let actual = md.get("general.architecture").and_then(|v| v.to_string().ok()).map_or("", |s| s.as_str());
        let Some(&arch) = accepted.iter().find(|a| **a == actual) else {
            candle_core::bail!("expected `general.architecture` in {accepted:?}, got `{actual}`");
        };
        Ok(Self { md, arch })
    }
    pub fn get(&self, k: &str) -> Result<&Value> {
        self.md
            .get(&format!("{}.{k}", self.arch))
            .ok_or_else(|| candle_core::Error::Msg(format!("{}: missing metadata key `{}.{k}`", self.arch, self.arch)))
    }
    pub fn usize(&self, k: &str) -> Result<usize> {
        any_int(self.get(k)?).and_then(|v| usize::try_from(v).ok()).ok_or_else(|| {
            candle_core::Error::Msg(format!("{}: `{}.{k}` not a non-negative integer", self.arch, self.arch))
        })
    }
    pub fn f64(&self, k: &str) -> Result<f64> {
        self.get(k)?
            .to_f32()
            .map(f64::from)
            .map_err(|e| candle_core::Error::Msg(format!("{}: `{}.{k}`: {e}", self.arch, self.arch)))
    }
    pub fn f32_array(&self, k: &str) -> Result<Vec<f32>> {
        match self.get(k)? {
            Value::Array(vs) => vs.iter().map(|v| v.to_f32().map_err(candle_core::Error::wrap)).collect(),
            other => candle_core::bail!("{}: `{}.{k}` must be array, got {other:?}", self.arch, self.arch),
        }
    }
    /// An optional integer, `default` where the key is absent or not integral.
    pub fn int_or(&self, k: &str, default: i64) -> i64 {
        self.get(k).ok().and_then(any_int).unwrap_or(default)
    }
    pub fn f64_or(&self, k: &str, default: f64) -> f64 {
        self.f64(k).unwrap_or(default)
    }
    pub fn bool_or(&self, k: &str, default: bool) -> bool {
        self.get(k).ok().and_then(|v| v.to_bool().ok()).unwrap_or(default)
    }
    /// An integer array, a signed entry cast as it is.
    pub fn u64_array(&self, k: &str) -> Result<Vec<u64>> {
        let int = |v: &Value| v.to_u64().ok().or_else(|| any_int(v).map(|n| n as u64));
        match self.get(k)? {
            Value::Array(vs) => {
                vs.iter().map(|v| int(v).ok_or_else(|| self.err(k, "has a non-integer entry"))).collect()
            }
            other => Err(self.err(k, &format!("must be an array, got {other:?}"))),
        }
    }
    /// A scalar or one entry a block, 0 a recurrent block as is one past the end; `per_layer` requires the full array.
    pub fn kv_heads(&self, blocks: usize, per_layer: bool) -> Result<Vec<u64>> {
        let (k, int) = ("attention.head_count_kv", |v: &Value| any_int(v).and_then(|n| u64::try_from(n).ok()));
        match self.get(k)? {
            Value::Array(a) if per_layer && (a.len() != blocks || a.iter().any(|v| int(v).is_none())) => {
                Err(self.err(k, &format!("has {} entries for {blocks} blocks, or one not an integer", a.len())))
            }
            Value::Array(a) => Ok((0..blocks).map(|i| a.get(i).and_then(int).unwrap_or(0)).collect()),
            v if !per_layer => Ok(vec![int(v).ok_or_else(|| self.err(k, "is not an integer or an array"))?; blocks]),
            v => Err(self.err(k, &format!("must be a per-layer array, got {v:?}"))),
        }
    }
    fn err(&self, k: &str, what: &str) -> candle_core::Error {
        candle_core::Error::Msg(format!("{}: `{}.{k}` {what}", self.arch, self.arch))
    }
}

/// `rope.dimension_sections`, the MRoPE 4-section array; absent, one section of plain partial RoPE.
pub(crate) fn rope_sections(m: &Meta<'_>, rope_dim_count: usize) -> Vec<u32> {
    match m.md.get(&format!("{}.rope.dimension_sections", m.arch)) {
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| match v {
                Value::U32(x) => Some(*x),
                Value::I32(x) => Some(*x as u32),
                _ => None,
            })
            .collect(),
        _ => vec![(rope_dim_count / 2) as u32, 0, 0, 0],
    }
}

/// The blocks in file order, the last `nextn` the draft blocks: outside the causal forward, never pool KV layers.
pub(crate) fn layers(
    block_count: usize, nextn: usize, leading_dense: usize, attention_of: impl Fn(usize) -> LayerAttention,
) -> Vec<LayerDecl> {
    let trunk = block_count.saturating_sub(nextn);
    (0..block_count)
        .map(|i| LayerDecl {
            attention: attention_of(i),
            ffn: if i < leading_dense { LayerFfn::Dense } else { LayerFfn::Routed },
            role: if i < trunk { LayerRole::Trunk } else { LayerRole::Nextn },
        })
        .collect()
}

/// What an architecture's declaration differs in; [`build`] makes the rest.
pub(crate) struct Decl<F: Fn(usize) -> LayerAttention> {
    pub blocks: usize,
    pub nextn: usize,
    pub leading_dense: usize,
    pub attention_of: F,
    pub head_count: usize,
    /// `None` reads `attention.key_length` / `value_length` (0 where absent).
    pub key_len: Option<usize>,
    pub value_len: Option<usize>,
    pub kv_lora_rank: Option<usize>,
    pub per_token: usize,
    /// The recurrent state's layout and an architecture's per-layer overrides.
    pub recurrent: Option<(RecurrentLayerConfig, Vec<(usize, RecurrentLayerConfig)>)>,
    pub hidden: usize,
    /// The context the architecture admits, the header's under its cap: the most a plan's guarantee may promise.
    pub max_seq_len: usize,
}

/// The declaration: `d`, and the counts every architecture reads alike from header keys its props do not keep.
/// A routed shape no kernel takes is refused here, before the weights, where the census would refuse it after.
pub(crate) fn build(m: &Meta<'_>, d: Decl<impl Fn(usize) -> LayerAttention>) -> Result<ArchDeclaration> {
    let header = |k: &str| m.int_or(k, 0) as u64;
    let experts = header("expert_count") as usize;
    if experts > 0 && !oscar_kernels::routed_fits(experts, d.per_token) {
        candle_core::bail!("{}: no routed kernel takes top-{} of {experts} experts", m.arch, d.per_token);
    }
    Ok(ArchDeclaration {
        layers: layers(d.blocks, d.nextn, d.leading_dense, d.attention_of),
        attention: AttentionDecl {
            head_count: d.head_count as u64,
            key_len: d.key_len.map_or_else(|| header("attention.key_length"), |n| n as u64),
            value_len: d.value_len.map_or_else(|| header("attention.value_length"), |n| n as u64),
            kv_lora_rank: d.kv_lora_rank.map(|n| n as u64),
        },
        experts: ExpertDecl { count: header("expert_count"), per_token: d.per_token as u64 },
        recurrent: d.recurrent.map(|(config, overrides)| RecurrentDecl {
            config,
            // The recurrent math's and the quantised matvec's type: no BF16↔F32 round trips.
            dtypes: RecurrentStateDtypes { state: Some(DType::F32), conv: Some(DType::F32) },
            overrides,
        }),
        hidden: d.hidden as u64,
        context_length: d.max_seq_len as u64,
        ple_row_width: header("embedding_length_per_layer_input"),
        vocab: Default::default(),
    })
}

/// The trailing draft blocks the file declares.
pub(crate) fn nextn(m: &Meta<'_>) -> usize {
    m.int_or("nextn_predict_layers", 0) as usize
}

/// The causal forward's blocks: the file's, less its trailing draft blocks.
pub(crate) fn trunk(m: &Meta<'_>) -> Result<usize> {
    let blocks = m.usize("block_count")?;
    blocks.checked_sub(nextn(m)).ok_or_else(|| m.err("nextn_predict_layers", &format!("exceeds the {blocks} blocks")))
}

pub(crate) fn leading_dense(m: &Meta<'_>) -> usize {
    m.int_or("leading_dense_block_count", 0) as usize
}

/// The recurrent state pool the declaration lays out: its trunk's layer kinds, its state's layout and dtypes.
pub(crate) fn hybrid_cache(
    decl: &ArchDeclaration, max_seq_len: usize, dtype: DType, device: &Device,
) -> Result<EitherCache> {
    let Some(RecurrentDecl { config, dtypes, overrides }) = decl.recurrent.clone() else {
        candle_core::bail!("hybrid cache needs declared recurrent state");
    };
    let kind = |l: &LayerDecl| match l.attention {
        LayerAttention::Recurrent => HybridLayerType::Recurrent,
        LayerAttention::Kv { .. } => HybridLayerType::Attention,
    };
    let layer_types = decl.layers.iter().filter(|l| l.role == LayerRole::Trunk).map(kind).collect();
    let config = HybridCacheConfig { layer_types, max_seq_len, recurrent: config, recurrent_overrides: overrides };
    let cache = HybridCache::new(config, dtype, dtypes, device)
        .map_err(|e| candle_core::Error::Msg(format!("hybrid cache init failed: {e}")))?;
    Ok(EitherCache::Hybrid(Arc::new(std::sync::Mutex::new(cache))))
}

#[cfg(test)]
mod tests {
    use super::Meta;
    use candle_core::quantized::gguf_file::Value;
    use std::collections::HashMap;

    /// K3's irregular I32 array (1 on its 24 MLA layers) read as written, a block past its end holding no KV; the
    /// scalar form for every block; an unreadable or absent count is an error, as a 0 would price a cache at nothing.
    #[test]
    fn head_count_kv_is_read_as_written_or_refused() {
        let key = "kimi-k3.attention.head_count_kv";
        let kv_heads = |v: Option<Value>, blocks: usize, per_layer: bool| {
            let md: HashMap<String, Value> = v.into_iter().map(|v| (key.to_string(), v)).collect();
            let meta = Meta { md: &md, arch: "kimi-k3" };
            meta.kv_heads(blocks, per_layer)
        };
        let mut per_layer = vec![0i32; 93];
        for l in (3..93).step_by(4).chain([92]) {
            per_layer[l] = 1;
        }
        let array = || Some(Value::Array(per_layer.iter().map(|&h| Value::I32(h)).collect()));
        let heads = kv_heads(array(), 93, true).expect("the array form");
        assert_eq!(heads.iter().filter(|h| **h > 0).count(), 24);
        assert_eq!((heads[91], heads[92]), (1, 1), "no stride reaches the trailing layer");
        assert_eq!(kv_heads(array(), 94, false).expect("the array form")[93], 0);
        assert_eq!(kv_heads(Some(Value::U32(4)), 3, false).expect("the scalar form"), vec![4, 4, 4]);
        let err = kv_heads(Some(Value::String("four".into())), 3, false).expect_err("a string is not a head count");
        assert!(err.to_string().contains("head_count_kv"), "names the key: {err}");
        assert!(kv_heads(None, 3, false).is_err(), "an absent key is not zero heads");
    }
}
