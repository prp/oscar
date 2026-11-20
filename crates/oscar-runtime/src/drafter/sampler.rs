//! The mistral.rs fork's device sampler of its pipelined step, off while drafting: a queued step eats the draft lead.
use std::sync::Arc;

use candle_core::{Device, Tensor};
use mistralrs_core::{DeviceSampleSpec, DeviceSampler};

struct KernelSampler;

impl KernelSampler {
    fn spec(spec: &DeviceSampleSpec) -> oscar_kernels::DeviceSampleSpec {
        oscar_kernels::DeviceSampleSpec {
            temperature: spec.temperature,
            top_k: spec.top_k,
            top_p: spec.top_p,
            min_p: spec.min_p,
            presence_penalty: spec.presence_penalty,
        }
    }
}

impl DeviceSampler for KernelSampler {
    fn active(&self) -> bool {
        !super::drafting()
    }
    fn supports(&self, spec: &DeviceSampleSpec) -> bool {
        Self::spec(spec).on_device()
    }
    fn output(&self, device: &Device) -> candle_core::Result<Tensor> {
        oscar_kernels::sample_output(device)
    }
    fn context(&self, device: &Device, vocab: usize, tokens: &[u32]) -> candle_core::Result<Tensor> {
        oscar_kernels::sample_context(device, vocab, tokens)
    }
    fn sample(
        &self, logits: &Tensor, spec: &DeviceSampleSpec, seed: u64, position: u64, out: &Tensor, seen: Option<&Tensor>,
    ) -> candle_core::Result<()> {
        oscar_kernels::sample_on_device(logits, &Self::spec(spec), seed, position, out, seen)
    }
    fn read(&self, out: &Tensor) -> candle_core::Result<(u32, f32)> {
        oscar_kernels::read_sample(out)
    }
}

/// Idempotent.
pub fn register() {
    mistralrs_core::register_device_sampler(Arc::new(KernelSampler));
}
