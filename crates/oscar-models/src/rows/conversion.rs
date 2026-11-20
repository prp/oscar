//! What `download` fetches per model and how it becomes the files oscar runs, dtype for dtype.

use candle_core::quantized::GgmlDType;

pub struct Conversion {
    pub source_repo: &'static str,
    /// The quant directory or file stem the source `.gguf` paths contain.
    pub source_select: &'static str,
    /// `(repo, file)` of a published imatrix, `.dat` or GGUF.
    pub imatrix: Option<(&'static str, &'static str)>,
    pub transform: Transform,
    /// Written at the download's end.
    pub derived: &'static [Derived],
}

/// `from` requantised to `dtype` as `name`, in a `<shards>-<file>.gguf`; its own name shadows nothing.
pub struct Derived {
    pub file: &'static str,
    pub from: &'static str,
    pub name: &'static str,
    pub dtype: GgmlDType,
}

/// The drafter's own LM head, lighter per draft forward than the Q6_K head at no cost to the hints' recall.
pub const DRAFT_HEAD: Derived =
    Derived { file: "000-draft-head", from: "output.weight", name: "output_draft.weight", dtype: GgmlDType::Q4K };

pub enum Transform {
    /// The published shards, repacked to the store alignment.
    PassThrough,
    /// Every shard rewritten: rule tensors requantised, the rest copied.
    Rewrite { rules: &'static [&'static [Rule]] },
    /// The re-quantised overlay shard: the shards untouched, rule tensors into `overlay`, whose `000-` name sorts
    /// first so it shadows them at load.
    Overlay { overlay: &'static str, rules: &'static [&'static [Rule]] },
}

/// A tensor class (name with `blk.N.` stripped) and its target; a conversion's groups read as one list.
pub type Rule = (&'static str, Target);

#[derive(Clone, Copy)]
pub enum Target {
    Fixed(GgmlDType),
    /// `more` on the layers `llama-quantize`'s `use_more_bits` picks, else `base`.
    MoreBits {
        base: GgmlDType,
        more: GgmlDType,
    },
}
