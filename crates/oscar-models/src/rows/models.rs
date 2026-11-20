//! Each model's row, in [`ModelPreset`]'s variant order.

use super::conversion::{Conversion, Rule, Transform, DRAFT_HEAD};
use super::{vendor, ModelDef, ModelPreset, Q3, Q4, Q4_MORE6, Q5_0, Q6, QWEN_DENSE, QWEN_IDENTITY, QWEN_NON_THINKING};

/// The requantised dense side frees memory for the expert pool at a negligible KL; K3.1 shares the rules.
const K3_OVERLAY: &[&[Rule]] = &[&[
    // No rule for `attn_k_b`/`attn_v_b`: their Metal path is Q8_0, so a requant costs memory.
    ("attn_q.weight", Q3),
    ("attn_k.weight", Q3),
    ("attn_v.weight", Q3),
    ("attn_output.weight", Q3),
    ("attn_gate.weight", Q3),
    ("attn_q_a.weight", Q3),
    ("attn_q_b.weight", Q3),
    ("attn_kv_a_mqa.weight", Q3),
    ("ssm_f_a.weight", Q3),
    ("ssm_f_b.weight", Q3),
    ("ssm_g.weight", Q3),
    ("ffn_gate_shexp.weight", Q4),
    ("ffn_up_shexp.weight", Q4),
    ("ffn_down_shexp.weight", Q4),
    ("ffn_routed_down.weight", Q4),
    ("ffn_routed_up.weight", Q4),
    ("ffn_gate.weight", Q4),
    ("ffn_up.weight", Q4),
    ("ffn_down.weight", Q4),
    ("output.weight", Q6),
]];

// Kimi K3: the demonstration row (i-quant experts, dense-G overlay).
const KIMI_K3: ModelDef = ModelDef {
    preset: ModelPreset::KimiK3,
    family_dir: "Kimi-K3-GGUF",
    parameters: 2779,
    least_gb: 42.31,
    production_variant: "UD-IQ1_M",
    // The overlay sorts first and lookup is first-file-wins, so its requantised weights shadow the shipped Q8_0 ones.
    conversion: Conversion {
        source_repo: "unsloth/Kimi-K3-GGUF",
        source_select: "UD-IQ1_M",
        imatrix: None,
        derived: &[],
        transform: Transform::Overlay { overlay: "Kimi-K3-000-dense-G.gguf", rules: K3_OVERLAY },
    },
    tok_repo: "moonshotai/Kimi-K3",
    // minijinja cannot run the published template; this copy renders the same tokens and raises on media.
    chat_template: include_str!("../../assets/k3_chat_template.jinja"),
    // No drafter, as no KDA architecture drafts; the IO pool at the machine's knee, not the model's.
    params: &[
        ("OSCAR_DRAFT", "none"),
        ("OSCAR_IO_THREADS", "4"),
        // The GPU plane: the architecture provisions no host router tables, so the chain has nothing to read.
        ("OSCAR_LOOKAHEAD", "gpu"),
    ],
    // The paged rows' common window, as GLM-5.2's: the slot a layer its last 16k costs buys no decode here.
    serve_sequence: (131072, 16384),
    sampling_thinking: Some(vendor(1.0, None, 0.95, Some(0.0))),
    sampling_non_thinking: None,
    default_system: None,
};

pub(super) const MODELS: &[ModelDef] = &[
    // Qwen3.6-35B-A3B: the resident baseline row.
    ModelDef {
        preset: ModelPreset::Qwen35B,
        family_dir: "Qwen3.6-35B-A3B-GGUF",
        parameters: 35,
        least_gb: 4.41,
        production_variant: "expertsQ4",
        // No imatrix exists, hence no `im` suffix.
        conversion: Conversion {
            source_repo: "unsloth/Qwen3.6-35B-A3B-GGUF",
            source_select: "Q8_0",
            imatrix: None,
            derived: &[DRAFT_HEAD],
            transform: Transform::Rewrite {
                // No rule for ssm_alpha/ssm_beta: F32 at widths no k-quant holds, as the vendor's tiers leave them.
                rules: &[
                    QWEN_DENSE,
                    &[
                        ("ffn_gate_exps.weight", Q4),
                        ("ffn_up_exps.weight", Q4),
                        ("ffn_down_exps.weight", Q4_MORE6),
                        ("ffn_down_shexp.weight", Q4_MORE6),
                    ],
                ],
            },
        },
        tok_repo: "Qwen/Qwen3.6-35B-A3B",
        // The published template drops past thoughts at a user turn; this copy keeps real ones and adds no empty one.
        chat_template: include_str!("../../assets/qwen35b_chat_template.jinja"),
        // Paged, the host lookahead is precise and the drafter helps where decode waits on reads: its controller decides.
        params: &[
            ("OSCAR_IO_THREADS", "8"),
            ("OSCAR_DRAFT", "self"),
            ("OSCAR_DRAFT_K", "4"),
            ("OSCAR_LOOKAHEAD", "host"),
        ],
        // The guarantee its pool affords beside real traffic's p99 of 130k; the census records every width at it.
        serve_sequence: (131072, 16384),
        // Its file omits the card's `presence_penalty` for both modes, so the thinking row is carried whole.
        sampling_thinking: Some(vendor(1.0, Some(20), 0.95, Some(1.5))),
        sampling_non_thinking: Some(QWEN_NON_THINKING),
        default_system: Some(QWEN_IDENTITY),
    },
    // Qwen3-235B-A22B: the mid-residency paged row.
    ModelDef {
        preset: ModelPreset::Qwen235B,
        family_dir: "Qwen3-235B-A22B-GGUF",
        parameters: 235,
        least_gb: 13.75,
        production_variant: "dQ4im",
        conversion: Conversion {
            source_repo: "unsloth/Qwen3-235B-A22B-GGUF",
            source_select: "Q8_0",
            imatrix: Some(("unsloth/Qwen3-235B-A22B-GGUF", "imatrix_unsloth.dat")),
            derived: &[],
            transform: Transform::Rewrite {
                rules: &[&[
                    ("token_embd.weight", Q4),
                    ("attn_q.weight", Q4),
                    ("attn_k.weight", Q4),
                    ("attn_v.weight", Q4_MORE6),
                    ("attn_output.weight", Q4),
                    ("ffn_gate_exps.weight", Q4),
                    ("ffn_up_exps.weight", Q4),
                    ("ffn_down_exps.weight", Q4),
                    ("output.weight", Q6),
                ]],
            },
        },
        tok_repo: "Qwen/Qwen3-235B-A22B",
        // The published template drops past thoughts at a user turn; this copy keeps real ones and adds no empty one.
        chat_template: include_str!("../../assets/qwen235b_chat_template.jinja"),
        // No drafter: the predictor already serves nearly every access, so a draft round buys no lead and costs GPU time.
        params: &[
            ("OSCAR_IO_THREADS", "8"),
            ("OSCAR_DRAFT", "none"),
            // The two planes are near-perfect substitutes here, the host chain a little ahead.
            ("OSCAR_LOOKAHEAD", "host"),
        ],
        // `context_length` is 40960 with no rope scaling: the pair fills the trained window, not a KV choice.
        serve_sequence: (32768, 8192),
        sampling_thinking: None,
        sampling_non_thinking: Some(vendor(0.7, Some(20), 0.8, None)),
        default_system: Some(QWEN_IDENTITY),
    },
    // Qwen3.5-397B-A17B: the deep-offloading hybrid row with the self-drafter.
    ModelDef {
        preset: ModelPreset::Qwen397B,
        family_dir: "Qwen3.5-397B-A17B-GGUF",
        parameters: 397,
        least_gb: 10.73,
        production_variant: "allQ3",
        conversion: Conversion {
            source_repo: "unsloth/Qwen3.5-397B-A17B-GGUF",
            source_select: "Q8_0",
            imatrix: Some(("unsloth/Qwen3.5-397B-A17B-GGUF", "imatrix_unsloth.gguf_file")),
            derived: &[DRAFT_HEAD],
            transform: Transform::Rewrite {
                rules: &[
                    QWEN_DENSE,
                    &[
                        ("ffn_gate_exps.weight", Q3),
                        ("ffn_up_exps.weight", Q3),
                        ("ffn_down_exps.weight", Q3),
                        ("ffn_down_shexp.weight", Q4_MORE6),
                        ("ssm_alpha.weight", Q4),
                        ("ssm_beta.weight", Q4),
                    ],
                ],
            },
        },
        tok_repo: "Qwen/Qwen3.5-397B-A17B",
        // The published template drops past thoughts at a user turn and renders an empty scaffold; this copy does neither.
        chat_template: include_str!("../../assets/qwen397b_chat_template.jinja"),
        // IO pool sized to the cores (deeper costs drafter GPU contention).
        params: &[
            ("OSCAR_IO_THREADS", "4"),
            // The drafter fills the near horizon: a lookahead earns neither its host matvec nor its gate tables.
            ("OSCAR_LOOKAHEAD", "none"),
            ("OSCAR_DRAFT", "self"),
            ("OSCAR_DRAFT_K", "8"),
        ],
        // The paged rows' common window: nine slots a layer over the shipped 262144, the sessions past 131k refused.
        serve_sequence: (131072, 16384),
        // The card's rows: a 1.5 presence penalty for non-thinking only.
        sampling_thinking: Some(vendor(0.6, Some(20), 0.95, Some(0.0))),
        sampling_non_thinking: Some(QWEN_NON_THINKING),
        default_system: Some(QWEN_IDENTITY),
    },
    // GLM-5.2: a deep expert-offloading row (absorbed-MLA attention).
    ModelDef {
        preset: ModelPreset::Glm52,
        family_dir: "GLM-5.2-GGUF",
        parameters: 754,
        least_gb: 24.77,
        production_variant: "expertsQ3",
        // The dense side stays Q8_0; the experts' step down is the saving.
        conversion: Conversion {
            source_repo: "unsloth/GLM-5.2-GGUF",
            source_select: "Q8_0",
            imatrix: None,
            derived: &[],
            transform: Transform::Rewrite {
                rules: &[&[
                    ("ffn_gate_exps.weight", Q3),
                    ("ffn_up_exps.weight", Q3),
                    ("ffn_down_exps.weight", Q3),
                    ("output.weight", Q6),
                ]],
            },
        },
        tok_repo: "zai-org/GLM-5.2",
        // This copy writes `m.content[0]` for `m.content.0.type`, a float to minijinja, and keeps past thoughts.
        chat_template: include_str!("../../assets/glm52_chat_template.jinja"),
        // The drafter is zero-sum at this residency, every fill evicting a live entry; recency beats transition.
        params: &[
            ("OSCAR_DRAFT", "none"),
            ("OSCAR_IO_THREADS", "4"),
            ("OSCAR_ORACLE", "recency"),
            // The chain on the late tap (the post-FFN residual), a little ahead of the alternatives.
            ("OSCAR_LOOKAHEAD", "host"),
            ("OSCAR_LOOKAHEAD_STAGE_MAX_EXPERTS", "8"),
            ("OSCAR_LOOKAHEAD_STAGE_MARGIN", "0.05"),
        ],
        // The paged rows' common window: the slot a layer its last 16k costs buys no decode here.
        serve_sequence: (131072, 16384),
        sampling_thinking: None,
        sampling_non_thinking: None,
        default_system: None,
    },
    KIMI_K3,
    // Kimi K3.1 Pro: K3 with a sparse-attention indexer on its MLA layers (never loaded); the row K3's but its files,
    // KarLexAI's until unsloth publishes (experts IQ1_S, 19 down projections IQ2_XXS, the dense tensors Q6_K).
    ModelDef {
        preset: ModelPreset::KimiK31,
        family_dir: "Kimi-K3.1-Pro-GGUF",
        least_gb: 38.03,
        production_variant: "KLX-IQ1_S",
        conversion: Conversion {
            source_repo: "KarLexAI/Kimi-K3.1-Pro-GGUF",
            source_select: "KLX-IQ1_S",
            // The embedding table ships Q6_K, which the gather kernel lacks (Q4_K or Q8_0; K3's ships Q8_0).
            transform: Transform::Overlay {
                overlay: "Kimi-K3.1-Pro-000-dense-H.gguf",
                rules: &[K3_OVERLAY[0], &[("token_embd.weight", Q4)]],
            },
            ..KIMI_K3.conversion
        },
        tok_repo: "moonshotai/Kimi-K3.1-Pro",
        ..KIMI_K3
    },
    // Qwen3-Coder-Next 80B: the fine-grained-expert row, loaded by the `qwen35_moe` module (`qwen3next` arch).
    ModelDef {
        preset: ModelPreset::Qwen3CoderNext,
        family_dir: "Qwen3-Coder-Next-GGUF",
        parameters: 80,
        least_gb: 4.64,
        production_variant: "expertsQ4im",
        conversion: Conversion {
            source_repo: "unsloth/Qwen3-Coder-Next-GGUF",
            source_select: "Q8_0",
            imatrix: Some(("unsloth/Qwen3-Coder-Next-GGUF", "imatrix_unsloth.gguf_file")),
            derived: &[DRAFT_HEAD],
            transform: Transform::Rewrite {
                // The gate rides in attn_q on full-attention layers; GDN layers carry attn_qkv, attn_gate and ssm_ba.
                rules: &[
                    QWEN_DENSE,
                    &[
                        ("ssm_ba.weight", Q4),
                        ("ffn_gate_exps.weight", Q4),
                        ("ffn_up_exps.weight", Q4),
                        ("ffn_down_exps.weight", Q4),
                        ("ffn_down_shexp.weight", Q4_MORE6),
                    ],
                ],
            },
        },
        tok_repo: "Qwen/Qwen3-Coder-Next",
        // The published template verbatim, pinned against upstream edits; tool calls parse as `parse_qwen_xml_function`.
        chat_template: include_str!("../../assets/qwen3_coder_next_chat_template.jinja"),
        // Paged, the host lookahead is precise and the drafter helps where decode waits on reads: its controller decides.
        params: &[
            ("OSCAR_IO_THREADS", "8"),
            ("OSCAR_DRAFT", "self"),
            ("OSCAR_DRAFT_K", "4"),
            ("OSCAR_LOOKAHEAD", "host"),
        ],
        // The guarantee its pool affords beside real traffic's p99 of 130k; the census records every width at it.
        serve_sequence: (131072, 16384),
        // Nothing published beyond `generation_config.json`, used as shipped.
        sampling_thinking: None,
        sampling_non_thinking: None,
        default_system: None,
    },
    // GLM-5.3-Flash 320B: the hybrid KDA/MLA row.
    ModelDef {
        preset: ModelPreset::Glm53Flash,
        family_dir: "GLM-5.3-Flash-GGUF",
        parameters: 321,
        least_gb: 8.71,
        production_variant: "expertsQ3im",
        conversion: Conversion {
            source_repo: "unsloth/GLM-5.3-Flash-GGUF",
            source_select: "Q8_0",
            imatrix: Some(("unsloth/GLM-5.3-Flash-GGUF", "imatrix_unsloth.gguf_file")),
            derived: &[],
            transform: Transform::Rewrite {
                // Unruled: `attn_k_b`/`attn_v_b` (a requant costs memory), the 128-wide `ssm_f_b`/`ssm_g_b` (no k-quant).
                rules: &[&[
                    ("ffn_gate_exps.weight", Q3),
                    ("ffn_up_exps.weight", Q3),
                    ("ffn_down_exps.weight", Q3),
                    ("attn_q.weight", Q4),
                    ("attn_k.weight", Q4),
                    ("attn_v.weight", Q4),
                    ("attn_output.weight", Q4),
                    ("attn_q_a.weight", Q4),
                    ("attn_q_b.weight", Q4),
                    ("attn_kv_a_mqa.weight", Q4),
                    ("ssm_beta.weight", Q4),
                    ("ssm_f_a.weight", Q4),
                    ("ssm_g_a.weight", Q4),
                    ("ffn_gate.weight", Q4),
                    ("ffn_up.weight", Q4),
                    ("ffn_down.weight", Q4),
                    ("ffn_gate_shexp.weight", Q4),
                    ("ffn_up_shexp.weight", Q4),
                    ("ffn_down_shexp.weight", Q4),
                    ("output.weight", Q6),
                ]],
            },
        },
        tok_repo: "zai-org/GLM-5.3-Flash",
        // The published template, rewritten for minijinja and `enable_thinking`; tool calls parse as `parse_glm_tool_call`.
        chat_template: include_str!("../../assets/glm53_flash_chat_template.jinja"),
        // The host chain on the late tap pays at this residency; the architecture has no draft path.
        params: &[
            ("OSCAR_DRAFT", "none"),
            ("OSCAR_IO_THREADS", "4"),
            ("OSCAR_ORACLE", "recency"),
            ("OSCAR_LOOKAHEAD", "host"),
            ("OSCAR_LOOKAHEAD_STAGE_MAX_EXPERTS", "8"),
            ("OSCAR_LOOKAHEAD_STAGE_MARGIN", "0.05"),
        ],
        // KV only on the 12 MLA layers plus fixed KDA state: the plan affords real traffic's p99 of 130k.
        serve_sequence: (131072, 16384),
        // Nothing published beyond `generation_config.json`, used as shipped.
        sampling_thinking: None,
        sampling_non_thinking: None,
        default_system: None,
    },
    // Qwen3.8-Flash-Next 180B: hybrid GDN/GQA layers under 4-stream hyper-connections, its n-gram table never resident.
    ModelDef {
        preset: ModelPreset::Qwen38FlashNext,
        family_dir: "Qwen3.8-Flash-Next-GGUF",
        parameters: 178,
        least_gb: 6.75,
        production_variant: "expertsQ3im",
        conversion: Conversion {
            source_repo: "unsloth/Qwen3.8-Flash-Next-GGUF",
            source_select: "Q8_0",
            imatrix: Some(("unsloth/Qwen3.8-Flash-Next-GGUF", "imatrix_unsloth.gguf_file")),
            derived: &[DRAFT_HEAD],
            transform: Transform::Rewrite {
                // Unruled, as shipped: `ffn_down_shexp`, which the dense Q8_0 path reads, and the 320-wide `hc_*` mixers.
                rules: &[
                    QWEN_DENSE,
                    &[
                        ("ffn_gate_exps.weight", Q3),
                        ("ffn_up_exps.weight", Q3),
                        // No k-quant holds 640-wide rows; Q5_0 sits above the vendor's own down tier for Q3 gate/up.
                        ("ffn_down_exps.weight", Q5_0),
                        ("ssm_alpha.weight", Q4),
                        ("ssm_beta.weight", Q4),
                    ],
                ],
            },
        },
        tok_repo: "Qwen/Qwen3.8-Flash-Next",
        // The published template, rewritten for minijinja token for token; tool calls parse as `parse_qwen_xml_function`.
        chat_template: include_str!("../../assets/qwen38_flash_next_chat_template.jinja"),
        // The drafter pays where decode waits on reads: its controller decides; lookahead tables miss the four-stream mix.
        params: &[
            ("OSCAR_DRAFT", "self"),
            ("OSCAR_DRAFT_K", "2"),
            ("OSCAR_IO_THREADS", "4"),
            ("OSCAR_ORACLE", "recency"),
            ("OSCAR_LOOKAHEAD", "none"),
        ],
        // KV only on the 12 full-attention layers plus fixed GDN/PLE state: the plan affords real traffic's p99 of 130k.
        serve_sequence: (131072, 16384),
        // Both rows are the card's (its min_p 0 and repetition 1.0 are unset).
        sampling_thinking: Some(vendor(1.0, Some(20), 0.95, Some(0.0))),
        sampling_non_thinking: Some(QWEN_NON_THINKING),
        default_system: Some(QWEN_IDENTITY),
    },
];
