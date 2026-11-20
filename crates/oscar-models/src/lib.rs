//! Oscar's GGUF architectures, each a [`GgufArch`] factory under its name; [`register`] runs before a model loads.

mod architecture;
mod attention;
mod declare;
pub mod draft;
mod embedding;
pub mod ffn;
mod gdn;
mod glm5next;
mod glm_dsa;
mod hc;
mod kda;
mod kimi_k3;
mod latent;
pub mod layout;
mod load;
pub mod moe;
mod plain;
pub mod ple;
mod qwen35_moe;
mod qwen3_moe;
mod qwen4exp;
mod qwen_hybrid;
pub mod rows;
mod situ;
pub mod state_slots;
#[cfg(test)]
mod test_support;

pub use declare::any_int;
pub use load::gate_matvec_or_forward;

use std::sync::{Arc, OnceLock};

use candle_core::{Device, Result, Tensor};
use mistralrs_core::gguf::GGUFArchitecture;
use mistralrs_core::gguf::{register_gguf_arch, GgufArch};
use mistralrs_core::kv_cache::EitherCache;
use mistralrs_core::text_models_inputs_processor::PagedAttentionInputMetadata;
use mistralrs_core::utils::model_config::FromGGUF;

use crate::architecture::Architecture;
use crate::draft::DraftableModel;

/// The loaded model, for the drafter and the lookahead planes: what its architecture states.
#[derive(Clone)]
pub struct LoadedModel {
    /// The pipeline's view: the one way to drive a forward outside the engine.
    pub arch: Arc<dyn GgufArch>,
    pub draft: Option<Arc<dyn DraftableModel>>,
    pub router_tables: bool,
    pub late_tap: bool,
}

static LOADED: OnceLock<LoadedModel> = OnceLock::new();

pub fn loaded_model() -> Option<&'static LoadedModel> {
    LOADED.get()
}

macro_rules! arch {
    ($arch:ident, $model:ty, $declare:path) => {
        register_gguf_arch(
            GGUFArchitecture::$arch,
            |ct, dev, mapper, attn, dtype| build(<$model>::from_gguf(ct, dev, mapper, attn, dtype)?),
            |ct| $declare(ct),
        );
    };
}

/// Every architecture this crate implements; idempotent.
pub fn register() {
    arch!(Qwen3MoE, qwen3_moe::ModelWeights, qwen3_moe::declare);
    arch!(Qwen35MoE, qwen35_moe::ModelWeights, qwen35_moe::declare);
    // qwen3next is the qwen35_moe family's: its deltas are declared there.
    arch!(Qwen3Next, qwen35_moe::ModelWeights, qwen35_moe::declare);
    arch!(GlmDsa, glm_dsa::ModelWeights, glm_dsa::declare);
    arch!(KimiK3, kimi_k3::ModelWeights, kimi_k3::declare);
    arch!(KimiK31, kimi_k3::ModelWeights, kimi_k3::declare);
    arch!(Glm5Next, glm5next::ModelWeights, glm5next::declare);
    arch!(Qwen4Exp, qwen4exp::ModelWeights, qwen4exp::declare);
}

/// One model for the pipeline and [`loaded_model`], with its architecture's statements.
fn build<M: Architecture>(model: M) -> Result<Arc<dyn GgufArch>> {
    let model = Arc::new(model);
    let _ = LOADED.set(LoadedModel {
        arch: model.clone(),
        draft: M::draftable(model.clone()),
        router_tables: M::ROUTER_TABLES,
        late_tap: M::LATE_TAP,
    });
    Ok(model)
}

/// The pipeline's view of each model: every engine forward begins here.
macro_rules! impl_gguf_arch {
    ($($ty:ty),+) => {$(
        impl GgufArch for $ty {
            fn forward(
                &self,
                input_ids: &Tensor,
                seqlen_offsets: &[usize],
                context_lens: Vec<(usize, usize)>,
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
                if self.cache.is_hybrid() { self.cache.hybrid().num_layers() } else { self.cache.normal().0.len() }
            }
        }
    )+};
}

impl_gguf_arch!(qwen35_moe::ModelWeights, kimi_k3::ModelWeights, glm5next::ModelWeights, qwen4exp::ModelWeights);
