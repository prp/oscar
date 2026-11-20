//! One module per model holds its [`ModelDef`] row; a family's differences are its architecture's, never a predicate.

use candle_core::quantized::GgmlDType;
use mistralrs_core::ModelGenerationDefaults;

pub mod conversion;
mod models;
use conversion::{Conversion, Rule, Target};
use models::MODELS;

const Q3: Target = Target::Fixed(GgmlDType::Q3K);
const Q4: Target = Target::Fixed(GgmlDType::Q4K);
const Q6: Target = Target::Fixed(GgmlDType::Q6K);
/// Q4_K, Q6_K on `use_more_bits` layers, as in the published Qwen files.
const Q4_MORE6: Target = Target::MoreBits { base: GgmlDType::Q4K, more: GgmlDType::Q6K };
/// For rows no k-quant can hold (640-wide down projections).
const Q5_0: Target = Target::Fixed(GgmlDType::Q5_0);

/// The Qwen hybrid rows' dense side as their published files ship it; a row adds its experts and own tensors.
const QWEN_DENSE: &[Rule] = &[
    ("token_embd.weight", Q4),
    ("attn_q.weight", Q4),
    ("attn_k.weight", Q4),
    ("attn_v.weight", Q4_MORE6),
    ("attn_output.weight", Q4),
    ("attn_qkv.weight", Q4_MORE6),
    ("attn_gate.weight", Q4),
    ("ssm_out.weight", Q4),
    ("ffn_gate_shexp.weight", Q4),
    ("ffn_up_shexp.weight", Q4),
    ("output.weight", Q6),
];

/// Qwen's published non-thinking row, which three cards state alike.
const QWEN_NON_THINKING: ModelGenerationDefaults = vendor(0.7, Some(20), 0.8, Some(1.5));

/// What chat plans, or the serve row's where smaller.
const CHAT_SEQUENCE: (u64, u64) = (65536, 16384);

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ModelPreset {
    /// Qwen3.6-35B-A3B expertsQ4 (~20 GB, 40 MoE layers, 256 experts top-8). Default.
    #[value(name = "qwen3.6-35b-a3b")]
    Qwen35B,
    /// Qwen3-235B-A22B dQ4im (~124 GB, 94 MoE layers, 128 experts): imatrix
    /// Q4_K, 22.8% fewer bytes per token than Q4_K_M. `--quant` loads others.
    #[value(name = "qwen3-235b-a22b")]
    Qwen235B,
    /// Qwen3.5-397B-A17B allQ3 (~163 GB, 60 hybrid layers, 512 experts top-10
    /// plus a shared), experts at Q3_K (23.6% under Q4_K_S).
    #[value(name = "qwen3.5-397b-a17b")]
    Qwen397B,
    /// GLM-5.2 expertsQ3 (~313 GB, 75 MoE layers, 256 experts top-8, absorbed MLA).
    #[value(name = "glm-5.2")]
    Glm52,
    /// Kimi K3 UD-IQ1_M (~629 GB, 2.8T total / 104B active, 93 KDA and MLA
    /// layers, 896 i-quant experts top-16): a demonstration, not for daily use.
    #[value(name = "kimi-k3")]
    KimiK3,
    /// Kimi K3.1 Pro KLX-IQ1_S (~577 GB; K3 with a sparse-attention indexer
    /// on its 24 MLA layers, run dense): a demonstration.
    #[value(name = "kimi-k3.1")]
    KimiK31,
    /// Qwen3-Coder-Next 80B expertsQ4im (~42 GB, 48 hybrid layers, 512 experts top-10 plus a shared).
    #[value(name = "qwen3-coder-next")]
    Qwen3CoderNext,
    /// GLM-5.3-Flash 320B expertsQ3im (~131 GB, 34 KDA + 12 MLA layers, 288 experts top-8 plus a shared).
    #[value(name = "glm-5.3-flash")]
    Glm53Flash,
    /// Qwen3.8-Flash-Next 180B expertsQ3im (~113 GB, 48 hybrid layers, 512
    /// experts top-10 plus a shared, a paged ~51 GB n-gram table).
    #[value(name = "qwen3.8-flash-next")]
    Qwen38FlashNext,
}

/// Qwen's own system identity, which its four models' vendor service sends.
const QWEN_IDENTITY: &str =
    "You are Qwen, a large language model developed by Alibaba Cloud. You are a helpful assistant.";

/// What differs between models, one row each in [`MODELS`].
pub struct ModelDef {
    pub preset: ModelPreset,
    /// The model's directory in every store root.
    pub family_dir: &'static str,
    /// Total parameters in billions, as the loader counts them, for a row of a model not downloaded.
    pub parameters: u64,
    /// The least budget a plan takes, at the context window's floor: the planner's refusal to `--mem 1`, as measured.
    /// What a machine must host for the model to load at all.
    pub least_gb: f64,
    /// Its directory under `family_dir`, whose sorted `*.gguf` glob gives the shards.
    pub production_variant: &'static str,
    /// How `download` produces the files oscar runs.
    pub conversion: Conversion,
    pub tok_repo: &'static str,
    /// The pinned template served in place of the shipped one; the row says where and why it differs.
    pub chat_template: &'static str,
    /// Measured paged; applied if absent, ahead of the common defaults, so the user's env wins.
    pub params: &'static [(&'static str, &'static str)],
    pub serve_sequence: (u64, u64),
    /// One mode's complete vendor sampling row, so serving a mode is one lookup, not a merge; `None`: none published.
    pub sampling_thinking: Option<ModelGenerationDefaults>,
    pub sampling_non_thinking: Option<ModelGenerationDefaults>,
    /// Sent when the client gives none: the vendor service's identity, or none where the template carries one.
    pub default_system: Option<&'static str>,
}

/// A vendor sampling row's four published fields.
const fn vendor(
    temperature: f64, top_k: Option<usize>, top_p: f64, presence_penalty: Option<f32>,
) -> ModelGenerationDefaults {
    ModelGenerationDefaults {
        do_sample: None,
        temperature: Some(temperature),
        top_k,
        top_p: Some(top_p),
        min_p: None,
        presence_penalty,
        repetition_penalty: None,
        max_new_tokens: None,
        max_length: None,
    }
}

/// Rows in variant order, so `self as usize` is the lookup; a missing row fails `every_preset_has_a_row`.
const _: () = {
    let mut i = 0;
    while i < MODELS.len() {
        assert!(MODELS[i].preset as usize == i, "MODELS is out of variant order");
        i += 1;
    }
};

impl ModelPreset {
    pub fn def(self) -> &'static ModelDef {
        &MODELS[self as usize]
    }

    /// Written afresh each call, so never stale: mistralrs takes a path, not text.
    pub fn materialise_chat_template(self) -> std::io::Result<String> {
        let path = std::env::temp_dir().join(format!("oscar-{self:?}-chat-template.jinja"));
        std::fs::write(&path, self.def().chat_template).map(|()| path.to_string_lossy().into_owned())
    }

    /// The `--model` spelling.
    pub fn label(self) -> String {
        use clap::ValueEnum;
        self.to_possible_value().expect("every preset has a CLI name").get_name().to_string()
    }

    /// The size the help says, "~131 GB": what a row not on disk shows.
    pub fn help_size_gb(self) -> u64 {
        use clap::ValueEnum;
        let help = self.to_possible_value().and_then(|v| v.get_help().map(|h| h.to_string())).unwrap_or_default();
        help.split_once("(~").and_then(|(_, r)| r.split_once(" GB")).and_then(|(n, _)| n.parse().ok()).unwrap_or(0)
    }

    /// In `/v1/models` order: the name under the tokenizer repo's org, its `+nothink` sibling and `default+nothink`,
    /// with each one's mode.
    pub fn served_model_ids(self, name: &str) -> Vec<(String, Option<bool>)> {
        let org = self.def().tok_repo.split('/').next().unwrap_or_default();
        let id = format!("{org}/{name}");
        let off = Some(false);
        vec![(id.clone(), None), (format!("{id}+nothink"), off), ("default+nothink".to_string(), off)]
    }

    /// Serve's source for omitted fields where it says more than `generation_config.json`; `None` keeps the file.
    pub fn vendor_sampling(self) -> mistralrs_core::ModeGenerationDefaults {
        let def = self.def();
        mistralrs_core::ModeGenerationDefaults {
            thinking: def.sampling_thinking.clone(),
            non_thinking: def.sampling_non_thinking.clone(),
        }
    }

    /// (input, output) tokens the plan guarantees: serve's the row's, chat's the smaller of it and [`CHAT_SEQUENCE`].
    pub fn planned_sequence_defaults(self, serve: bool) -> (u64, u64) {
        let (x, y) = self.def().serve_sequence;
        if serve {
            (x, y)
        } else {
            (x.min(CHAT_SEQUENCE.0), y.min(CHAT_SEQUENCE.1))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::ValueEnum;

    #[test]
    fn every_preset_has_a_row() {
        assert_eq!(MODELS.len(), ModelPreset::value_variants().len());
        for (i, preset) in ModelPreset::value_variants().iter().enumerate() {
            assert_eq!(MODELS[i].preset as usize, *preset as usize);
            assert!(!preset.def().family_dir.is_empty());
            assert!(!preset.def().production_variant.is_empty());
            assert!(!preset.def().tok_repo.is_empty());
            assert!(preset.help_size_gb() >= 20, "{preset:?}'s help names its size");
        }
    }
}
