//! A model's conversion checked before any fetch, and its rules applied tensor by tensor.

use std::collections::HashMap;

use candle_core::quantized::GgmlDType;
use oscar_models::rows::conversion::{Conversion, Target, Transform};

use super::gguf_rw::Disposition;

/// Strip `blk.N.` so a rule is written once per tensor class, not per layer.
pub fn class(name: &str) -> &str {
    name.strip_prefix("blk.").and_then(|rest| rest.split_once('.')).map_or(name, |(_, rest)| rest)
}

fn use_more_bits(i_layer: usize, n_layer: usize) -> bool {
    i_layer < n_layer / 8 || i_layer >= 7 * n_layer / 8 || (i_layer - n_layer / 8) % 3 == 2
}

fn resolve(target: Target, name: &str, n_layers: usize) -> GgmlDType {
    match target {
        Target::Fixed(dtype) => dtype,
        Target::MoreBits { base, more } => match crate::plan::block_index(name) {
            Some(i) if use_more_bits(i, n_layers) => more,
            _ => base,
        },
    }
}

/// By full tensor name: `n_per_row` weights each, or `n_experts * n_per_row` where calibration was per expert.
pub type ImatrixMap = HashMap<String, Vec<f32>>;

/// The conversion alone, before any fetch: each fault it refuses would otherwise fail an hour in.
pub fn validate(conversion: &Conversion) -> anyhow::Result<()> {
    let repo_ok = |r: &str| matches!(r.split_once('/'), Some((org, name)) if !org.is_empty() && !name.is_empty() && !name.contains('/'));
    anyhow::ensure!(repo_ok(conversion.source_repo), "source_repo {:?} is not `org/name`", conversion.source_repo);
    anyhow::ensure!(!conversion.source_select.is_empty(), "source_select is empty");

    if let Some((repo, file)) = conversion.imatrix {
        anyhow::ensure!(repo_ok(repo), "imatrix repo {repo:?} is not `org/name`");
        // The loader reads `.dat` or GGUF, which some repos name `.gguf_file`: the check is on the form.
        anyhow::ensure!(
            std::path::Path::new(file)
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e == "dat" || e.starts_with("gguf")),
            "imatrix file {file:?} neither `.dat` nor `.gguf` form"
        );
    }

    let rules = match &conversion.transform {
        Transform::PassThrough => return Ok(()),
        Transform::Rewrite { rules } => rules,
        Transform::Overlay { overlay, rules } => {
            anyhow::ensure!(
                std::path::Path::new(overlay).extension().is_some_and(|e| e == "gguf") && overlay.len() > ".gguf".len(),
                "overlay name {overlay:?} not a `.gguf` filename"
            );
            rules
        }
    };
    anyhow::ensure!(rules.iter().any(|g| !g.is_empty()), "rewriting transform with no rules: pass-through");

    let mut seen: Vec<&str> = Vec::new();
    for (class, target) in rules.iter().copied().flatten() {
        anyhow::ensure!(!class.is_empty(), "rule names empty tensor class");
        anyhow::ensure!(!seen.contains(class), "tensor class {class:?} has two rules");
        seen.push(class);
        let dtypes = match *target {
            Target::Fixed(d) => vec![d],
            Target::MoreBits { base, more } => vec![base, more],
        };
        for d in dtypes {
            anyhow::ensure!(super::kquant::is_writable(d), "rule {class:?}: no ggml quantiser linked for {d:?}");
            // An expert kernel for the dtype; the fused gate+up row is the model's activation's, check-model's ask.
            if matches!(*class, "ffn_gate_exps.weight" | "ffn_up_exps.weight" | "ffn_down_exps.weight") {
                anyhow::ensure!(
                    oscar_kernels::QkkMoeDType::from_ggml(d).is_some(),
                    "rule {class:?}: {d:?} has no expert kernel in bank"
                );
            }
        }
    }
    Ok(())
}

/// A overlay conversion's fate is the overlay's; its shards clone.
pub fn decide(transform: &Transform, name: &str, dims: &[usize], n_layers: usize) -> Disposition {
    let (rules, unmatched) = match transform {
        Transform::PassThrough => return Disposition::Copy,
        Transform::Rewrite { rules } => (rules, Disposition::Copy),
        Transform::Overlay { rules, .. } => (rules, Disposition::Skip),
    };
    match rules.iter().copied().flatten().find(|(c, _)| *c == class(name)) {
        None => unmatched,
        Some((_, target)) => {
            let dtype = resolve(*target, name, n_layers);
            // A row off the dtype's block keeps its shipped form, out of a overlay.
            if dims.last().is_none_or(|r| r % dtype.block_size() != 0) {
                unmatched
            } else {
                Disposition::Requant(dtype)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType::{Q3K, Q4K, Q5K, Q8_0};
    use oscar_models::rows::conversion::Rule;
    use oscar_models::rows::conversion::Target::{Fixed, MoreBits};

    fn base(transform: Transform) -> Conversion {
        Conversion { source_repo: "org/name", source_select: "Q8_0", imatrix: None, transform, derived: &[] }
    }

    fn rewrite(rules: &'static [&'static [Rule]]) -> Conversion {
        base(Transform::Rewrite { rules })
    }

    /// Q5_K has a quantiser but no expert kernel (the fused gate+up row is check-model's ask, at the model's
    /// activation); Q8_0 has no linked quantiser. Every model's rules validate.
    #[test]
    fn a_faulty_conversion_is_refused_naming_its_fault_and_every_row_validates() {
        validate(&rewrite(&[&[("attn_q.weight", Fixed(Q5K))]])).unwrap();
        for (conversion, fault) in [
            (rewrite(&[&[("ffn_gate_exps.weight", Fixed(Q5K))]]), "Q5K has no expert kernel"),
            (rewrite(&[&[("ffn_down_exps.weight", Fixed(Q5K))]]), "Q5K has no expert kernel"),
            (rewrite(&[&[("attn_q.weight", MoreBits { base: Q4K, more: Q8_0 })]]), "no ggml quantiser linked for Q8_0"),
            (Conversion { source_repo: "no-slash", ..base(Transform::PassThrough) }, "is not `org/name`"),
            (Conversion { imatrix: Some(("org/name", "imatrix.bin")), ..base(Transform::PassThrough) }, ".dat"),
            (rewrite(&[]), "no rules"),
            (rewrite(&[&[("attn_q.weight", Fixed(Q4K)), ("attn_q.weight", Fixed(Q3K))]]), "two rules"),
            (base(Transform::Overlay { overlay: ".gguf", rules: &[&[("attn_q.weight", Fixed(Q3K))]] }), "overlay name"),
        ] {
            let err = validate(&conversion).unwrap_err().to_string();
            assert!(err.contains(fault), "{err}");
        }
        use clap::ValueEnum;
        for preset in oscar_models::rows::ModelPreset::value_variants() {
            if let Err(e) = validate(&preset.def().conversion) {
                panic!("{} conversion row invalid: {e:#}", preset.def().family_dir);
            }
        }
    }
}
