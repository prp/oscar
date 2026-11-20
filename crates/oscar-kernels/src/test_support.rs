//! The crate's test fixtures.

use candle_core::{Device, Result, Shape, Tensor};

pub(crate) fn sine(n: usize, step: f32, phase: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * step + phase).sin() * 0.5).collect()
}

/// Signed values up to 2.8, `seed` decorrelating one tensor from the next.
pub(crate) fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 7919 + seed * 104729) % 10007) as f32 - 5003.0) / 1777.0).collect()
}

/// Built on the host, then copied to `device`.
pub(crate) fn staged(v: Vec<f32>, shape: impl Into<Shape>, device: &Device) -> Result<Tensor> {
    Tensor::from_vec(v, shape, &Device::Cpu)?.to_device(device)
}

pub(crate) fn host(t: &Tensor) -> Result<Vec<f32>> {
    t.flatten_all()?.to_vec1()
}

pub(crate) fn assert_bitwise(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths differ");
    let bad = got.iter().zip(want).filter(|(g, w)| g.to_bits() != w.to_bits()).count();
    assert_eq!(bad, 0, "{what}: {bad} of {} outputs differ", got.len());
}
