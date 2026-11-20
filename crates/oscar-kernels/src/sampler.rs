//! The device token sampler: a greedy row's argmax in two launches, a sampled row's draw in four.
use candle_core::{DType, Result, Storage, Tensor};

use crate::loader::{launch, out, pipeline, set_params, Family, Metal};

/// The most survivors a sampled row keeps on the device.
pub const SAMPLE_KMAX: u32 = 128;

/// `softmax(x / T)` under top-k, top-p, then min-p, the host's order, after the penalty on the context's tokens.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeviceSampleSpec {
    pub temperature: Option<f32>,
    pub top_k: u32,
    pub top_p: f32,
    pub min_p: f32,
    pub presence_penalty: f32,
}

impl DeviceSampleSpec {
    pub const GREEDY: Self = Self { temperature: None, top_k: 0, top_p: 0.0, min_p: 0.0, presence_penalty: 0.0 };

    pub fn on_device(&self) -> bool {
        self.temperature.is_none() || (1..=SAMPLE_KMAX).contains(&self.top_k)
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SampleParams {
    n: u32,
    top_k: u32,
    inv_temp: f32,
    top_p: f32,
    min_p: f32,
    seed_lo: u32,
    seed_hi: u32,
    pos_lo: u32,
    pos_hi: u32,
    penalty: f32,
}

/// Slices of the row the reduce and the compaction run in parallel.
const SAMPLE_SLICES: usize = 64;
/// The most candidates the compaction keeps; the select kernel's window.
const SAMPLE_CAND_MAX: usize = 2048;

/// Kept across steps, as a buffer cut per token costs its first GPU touch.
pub fn sample_output(device: &candle_core::Device) -> Result<Tensor> {
    let candle_core::Device::Metal(device) = device else {
        candle_core::bail!("sample_output: Metal only");
    };
    Ok(out(device, 2, DType::U32, "sampler-argmax")?.0)
}

/// A bit per vocabulary entry, set for `tokens`; built once per sequence, as the sampler marks each token it decides.
pub fn sample_context(device: &candle_core::Device, vocab: usize, tokens: &[u32]) -> Result<Tensor> {
    let mut words = vec![0u32; vocab.div_ceil(32)];
    for &t in tokens.iter().filter(|&&t| (t as usize) < vocab) {
        words[(t >> 5) as usize] |= 1 << (t & 31);
    }
    Tensor::new(words.as_slice(), device)
}

/// `out`: the token and its log-probability's bits. The draw's uniform hashes `(seed, position)`.
pub fn sample_on_device(
    logits: &Tensor, spec: &DeviceSampleSpec, seed: u64, position: u64, out: &Tensor, seen: Option<&Tensor>,
) -> Result<()> {
    refuse_if!(!spec.on_device(), "sample_on_device: {spec:?} is a host sampler's row");
    let om = Metal::of(out, "sample_on_device: out")?;
    refuse_if!(out.dtype() != DType::U32 || out.elem_count() != 2, "sample_on_device: out must be a (2,) U32 tensor");
    let x = logits.flatten_all()?;
    let n = x.elem_count();
    let xm = Metal::of(&x, "sample_on_device: logits")?;
    let (slices, compact, select) = match x.dtype() {
        DType::F32 => ("oscar_sample_slices_f32", "oscar_sample_compact_f32", "oscar_sample_select_f32"),
        DType::BF16 => ("oscar_sample_slices_bf16", "oscar_sample_compact_bf16", "oscar_sample_select_bf16"),
        other => candle_core::bail!("sample_on_device: logits must be F32 or BF16, got {other:?}"),
    };
    let dev = &xm.dev;
    // Without a penalty the bitmap is unread; the output stands in.
    let penalty = spec.presence_penalty;
    let seenm = match seen {
        Some(seen) if penalty != 0.0 => {
            refuse_if!(
                seen.dtype() != DType::U32 || seen.elem_count() < n.div_ceil(32),
                "sample_on_device: the context must be a U32 bitmap over {n} entries"
            );
            Metal::of(seen, "sample_on_device: the context")?
        }
        Some(_) => Metal::of(out, "sample_on_device: out")?,
        None if penalty != 0.0 => candle_core::bail!("sample_on_device: presence penalty needs its context"),
        None => Metal::of(out, "sample_on_device: out")?,
    };
    // A top-1 row is the argmax whatever its temperature.
    let greedy = spec.temperature.is_none() || spec.top_k == 1;
    let inv_temp: f32 = if greedy { 1.0 } else { 1.0 / spec.temperature.unwrap_or(1.0) };
    let partials = dev.new_buffer(SAMPLE_SLICES * 3, DType::F32, "sample-partials")?;
    let scratch = dev.new_buffer(3 + 4, DType::U32, "sample-scratch")?;
    let (partials, scratch) = (&*partials, &*scratch);
    let p = pipeline(dev, Family::Sampler, slices)?;
    launch(dev, &p, "sample-slices", [SAMPLE_SLICES, 1, 1], [256, 1, 1], |e| {
        set_params!(e, (&xm, partials, n as u32, inv_temp, &seenm, penalty));
        Ok(())
    })?;
    let p = pipeline(dev, Family::Sampler, "oscar_sample_merge")?;
    launch(dev, &p, "sample-merge", [1, 1, 1], [256, 1, 1], |e| {
        let mark = u32::from(greedy && penalty != 0.0);
        set_params!(e, (partials, &om, scratch, SAMPLE_SLICES as u32, inv_temp, &seenm, mark));
        Ok(())
    })?;
    if greedy {
        return Ok(());
    }
    let cand_v = dev.new_buffer(SAMPLE_CAND_MAX, DType::F32, "sample-cand-v")?;
    let cand_i = dev.new_buffer(SAMPLE_CAND_MAX, DType::U32, "sample-cand-i")?;
    let (cand_v, cand_i) = (&*cand_v, &*cand_i);
    let p = pipeline(dev, Family::Sampler, compact)?;
    launch(dev, &p, "sample-compact", [SAMPLE_SLICES, 1, 1], [256, 1, 1], |e| {
        set_params!(e, (&xm, scratch, cand_v, cand_i, n as u32, &seenm, penalty));
        Ok(())
    })?;
    let p = pipeline(dev, Family::Sampler, select)?;
    let params = SampleParams {
        n: n as u32,
        top_k: spec.top_k,
        inv_temp,
        top_p: spec.top_p,
        min_p: spec.min_p,
        seed_lo: seed as u32,
        seed_hi: (seed >> 32) as u32,
        pos_lo: position as u32,
        pos_hi: (position >> 32) as u32,
        penalty,
    };
    // The sort and the row fallback want a power of two of at least `SAMPLE_KMAX` threads.
    let max_tg = p.max_total_threads_per_threadgroup() as usize;
    let width = 1024usize.min(1usize << max_tg.ilog2()).max(SAMPLE_KMAX as usize);
    launch(dev, &p, "sample-select", [1, 1, 1], [width, 1, 1], |e| {
        set_params!(e, (&xm, scratch, cand_v, cand_i, &om, &[params][..], &seenm));
        Ok(())
    })
}

/// A completed sample's token and log-probability.
pub fn read_sample(out: &Tensor) -> Result<(u32, f32)> {
    let (storage, layout) = out.storage_and_layout();
    let Storage::Metal(storage) = &*storage else {
        candle_core::bail!("read_sample: not Metal tensor");
    };
    let ptr = storage.buffer().contents() as *const u32;
    refuse_if!(ptr.is_null(), "read_sample: the sample buffer is not host-visible");
    // SAFETY: a Shared buffer holding two u32 words past the offset, its writing command buffer waited for.
    let words = unsafe { std::slice::from_raw_parts(ptr.add(layout.start_offset()), 2) };
    Ok((words[0], f32::from_bits(words[1])))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn splitmix64(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// The host sampler's penalty: `logit - penalty` where the context holds the token, before anything else.
    fn penalised(row: &[f32], context: &[u32], penalty: f32) -> Vec<f32> {
        let mut seen = vec![false; row.len()];
        context.iter().filter(|&&t| (t as usize) < row.len()).for_each(|&t| seen[t as usize] = true);
        row.iter().zip(&seen).map(|(&v, &seen)| if seen { v - penalty } else { v }).collect()
    }

    /// The host sampler's rule: the last index among equal maxima.
    fn host_argmax(row: &[f32]) -> u32 {
        row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).expect("finite logits")).map_or(0, |(i, _)| i as u32)
    }

    /// The host's chain with the kernel's uniform: the token, and whether the draw sat within rounding of a cut.
    fn host_sample(row: &[f32], spec: &DeviceSampleSpec, seed: u64, pos: u64) -> (u32, bool) {
        let t = spec.temperature.unwrap();
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = row.iter().map(|&v| ((v - m) / t).exp()).sum();
        let mut idx: Vec<u32> = (0..row.len() as u32).collect();
        idx.sort_by(|&a, &b| row[b as usize].partial_cmp(&row[a as usize]).unwrap().then(a.cmp(&b)));
        idx.truncate(spec.top_k as usize);
        let probs: Vec<f32> = idx.iter().map(|&i| ((row[i as usize] - m) / t).exp() / z).collect();
        let max_p = probs[0];
        let (mut kept, mut cum, mut near) = (Vec::new(), 0.0f32, false);
        for &pj in &probs {
            let mut keep = true;
            if spec.top_p > 0.0 && spec.top_p < 1.0 {
                near |= (cum - spec.top_p).abs() < 1e-4;
                keep = cum < spec.top_p;
                cum += if keep { pj } else { 0.0 };
            }
            if spec.min_p > 0.0 && spec.min_p < 1.0 {
                near |= (max_p * spec.min_p - pj).abs() < 1e-6;
                keep &= max_p * spec.min_p < pj;
            }
            kept.push(if keep { pj } else { 0.0 });
        }
        let total: f32 = kept.iter().sum();
        let target = (splitmix64(seed ^ splitmix64(pos)) >> 40) as f32 / 16_777_216.0 * total;
        let (mut acc, mut pick) = (0.0f32, 0usize);
        for (j, &pj) in kept.iter().enumerate().filter(|(_, pj)| **pj > 0.0) {
            (pick, acc) = (j, acc + pj);
            if acc > target {
                break;
            }
        }
        near |= (acc - target).abs() < 1e-3 * total || (acc - kept[pick] - target).abs() < 1e-3 * total;
        (idx[pick], near)
    }

    /// The host's token under the context's penalty, and whether it sat within rounding of a cut (a greedy one never).
    fn host_pick(row: &[f32], context: &[u32], spec: &DeviceSampleSpec, seed: u64, pos: u64) -> (u32, bool) {
        let row = penalised(row, context, spec.presence_penalty);
        spec.temperature.map_or((host_argmax(&row), false), |_| host_sample(&row, spec, seed, pos))
    }

    /// The device's token and its log-probability, read once the sample completes.
    fn sample(
        x: &Tensor, spec: &DeviceSampleSpec, seed: u64, pos: u64, out: &Tensor, seen: Option<&Tensor>,
    ) -> Result<(u32, f32)> {
        sample_on_device(x, spec, seed, pos, out, seen)?;
        x.device().synchronize()?;
        read_sample(out)
    }

    fn rows(device: &Device, n: usize, v: usize, seed: u64) -> Result<Vec<Vec<f32>>> {
        let mut rows = Tensor::randn(0f32, 4f32, (n, v), device)?.to_vec2::<f32>()?;
        let mut s = seed;
        for r in rows.iter_mut() {
            // Ties at the maximum across the row test the tie rule.
            let m = r.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            for _ in 0..3 {
                s = splitmix64(s);
                r[(s % v as u64) as usize] = m;
            }
        }
        Ok(rows)
    }

    /// A context holding the row's maximum and a spread of other tokens.
    fn context(row: &[f32], seed: u64) -> Vec<u32> {
        let mut ctx: Vec<u32> = (0..600).map(|j| (splitmix64(seed + j) % row.len() as u64) as u32).collect();
        ctx.push(host_argmax(row));
        ctx
    }

    /// Greedy without a context and at bf16 (the host's rule exactly), under a penalty that moves the maximum off
    /// the context's tokens, then sampled: the 35B's two served rows (presence 1.5) and rows without one. Then the
    /// pipelined step's contract: the context is built once and the sampler marks each token it decides.
    #[test]
    fn every_row_follows_the_host_chain_and_the_sampler_marks_the_tokens_it_decides() -> Result<()> {
        let device = oscar_offload::pools::test_device()?;
        let v = 248_320;
        let mut rows = rows(&device, 24, v, 7)?;
        // A flat row and a peaked one, the window over the whole vocabulary or under k, take the select's row path.
        rows.push(rows[0].iter().map(|v| v * 0.01).collect());
        let mut peaked = rows[1].clone();
        peaked[5] += 200.0;
        rows.push(peaked);
        let specs = [
            DeviceSampleSpec::GREEDY,
            DeviceSampleSpec { presence_penalty: 1.5, ..DeviceSampleSpec::GREEDY },
            DeviceSampleSpec { temperature: Some(1.0), top_k: 20, top_p: 0.95, min_p: 0.0, presence_penalty: 1.5 },
            DeviceSampleSpec { temperature: Some(0.7), top_k: 20, top_p: 0.8, min_p: 0.0, presence_penalty: 1.5 },
            DeviceSampleSpec { temperature: Some(1.0), top_k: 5, top_p: 0.0, min_p: 0.1, presence_penalty: 0.0 },
            DeviceSampleSpec { temperature: Some(0.7), top_k: 128, top_p: 0.95, min_p: 0.05, presence_penalty: 0.0 },
        ];
        let out = sample_output(&device)?;
        let mut boundary = 0;
        for (r, row) in rows.iter().enumerate() {
            let x = Tensor::new(row.as_slice(), &device)?.reshape((1, 1, v))?;
            let xb = x.to_dtype(DType::BF16)?;
            let (tok, _) = sample(&xb, &DeviceSampleSpec::GREEDY, 0, r as u64, &out, None)?;
            assert_eq!(tok, host_argmax(&xb.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?), "row {r} bf16");
            let ctx = context(row, 200 + r as u64);
            for (s, spec) in specs.iter().enumerate() {
                // Fresh per spec, the sampler marking the token it decides; the first spec without one.
                let seen = sample_context(&device, v, &ctx)?;
                let seed = 0x5eed_0000 + s as u64;
                let (tok, logp) = sample(&x, spec, seed, r as u64, &out, (s > 0).then_some(&seen))?;
                let (want, near) = host_pick(row, &ctx, spec, seed, r as u64);
                assert!(tok == want || near, "row {r} spec {s}: device {tok} host {want}, not at a cut");
                boundary += usize::from(tok != want);
                assert!(logp <= 0.0 && logp.is_finite(), "row {r} spec {s}: logp {logp}");
            }
        }
        assert!(boundary <= 2, "{boundary} rows landed on a cut");
        let (row, x) = (&rows[0], Tensor::new(rows[0].as_slice(), &device)?);
        // A penalty wider than the row's spread: a marked token cannot recur.
        let specs = [
            DeviceSampleSpec { presence_penalty: 30.0, ..DeviceSampleSpec::GREEDY },
            DeviceSampleSpec { temperature: Some(0.7), top_k: 20, top_p: 0.8, min_p: 0.0, presence_penalty: 30.0 },
        ];
        for (s, spec) in specs.iter().enumerate() {
            let mut ctx = context(row, 7);
            let seen = sample_context(&device, v, &ctx)?;
            for step in 0..8u64 {
                let (tok, _) = sample(&x, spec, 0x5eed, step, &out, Some(&seen))?;
                // Device and host part only within rounding of a cut; the device's token stands from there.
                let (want, near) = host_pick(row, &ctx, spec, 0x5eed, step);
                assert!(tok == want || near, "spec {s} step {step}: device {tok} host {want}, not at a cut");
                assert!(!ctx.contains(&tok), "spec {s} step {step}: {tok} drawn twice under the penalty");
                ctx.push(tok);
            }
        }
        Ok(())
    }
}
