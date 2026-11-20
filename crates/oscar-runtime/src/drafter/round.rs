//! Every engine step, prefill chunks too, passes through the mistral.rs fork's [`StepRound`]: heartbeat and drafter.

use std::any::Any;
use std::sync::Arc;
use std::time::Duration;

use candle_core::Result;
use mistralrs_core::prefix_cacher::PrefixCacheManagerV2;
use mistralrs_core::sequence::Sequence;
use mistralrs_core::{CacheBackendMetadata, Pipeline, StepRound};
use rand_isaac::Isaac64Rng;

pub(super) struct DraftRound;

#[async_trait::async_trait]
impl StepRound for DraftRound {
    #[allow(clippy::too_many_arguments)]
    async fn step(
        &self, pipeline: &mut dyn Pipeline, input_seqs: &mut [&mut Sequence], is_prompt: bool, return_raw_logits: bool,
        prefix_cacher: &mut PrefixCacheManagerV2, disable_eos_stop: bool, rng: Arc<std::sync::Mutex<Isaac64Rng>>,
        backend_metadata: CacheBackendMetadata,
    ) -> Result<Duration> {
        let _step = crate::gpu::heartbeat::step();
        // Armed here, so even a fully resident model's decode is watched.
        if !is_prompt {
            crate::gpu::probe::mark_engine_thread();
            crate::gpu::probe::arm_wake_detector();
        }
        // The controller may switch the drafter before the round is chosen.
        let decoding = !is_prompt && input_seqs.len() == 1;
        super::overlap::boundary(decoding, input_seqs.first().map_or(0, |s| s.get_toks().len()));
        let drafted = !is_prompt && super::eligible(pipeline, input_seqs, return_raw_logits);
        let t0 = std::time::Instant::now();
        // The drafting round, then the pipeline's own step either way: byte-exact by construction.
        let out = async {
            if drafted {
                super::overlap::draft(pipeline, input_seqs, &backend_metadata)?;
            }
            pipeline
                .step(input_seqs, is_prompt, return_raw_logits, prefix_cacher, disable_eos_stop, rng, backend_metadata)
                .await
        }
        .await;
        if !is_prompt {
            crate::engine::eval::hooks().note_step(drafted, t0.elapsed());
        }
        out
    }

    fn chunk(&self) -> Box<dyn Any> {
        Box::new(crate::gpu::heartbeat::step())
    }
}
