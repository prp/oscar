//! The model's own forward runs at load with no GPU work, the pools recording its buffers: the plan replays the record.
//! Nothing is extrapolated: the planner chooses among recorded widths, each run at the deepest admitted context.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_metal_kernels::metal::STREAM_QUEUE_KEYS;
use mistralrs_core::gguf::GgufArch;
use mistralrs_core::text_models_inputs_processor::PagedRows;

use oscar_offload::census::{self as record, region_peak, Arm, Asked};
use oscar_offload::region::Placement;
use oscar_offload::segments::PAGE;

use std::sync::OnceLock;

/// Every buffer the forwards asked for and each queue's placement as the record armed; dropped before the solve.
pub struct Record {
    pub entries: Vec<Asked>,
    pub snapshots: HashMap<usize, Placement>,
}

impl Record {
    /// One queue's region peak from the census's start through `passes`, each a forward behind the next.
    pub fn peak(&self, queue: usize, passes: &[(usize, Arm)]) -> oscar_offload::region::Peak {
        let empty = Placement::default();
        oscar_offload::census::region_peak(&self.entries, self.snapshots.get(&queue).unwrap_or(&empty), queue, passes)
    }

    pub(crate) fn host_bytes(&self) -> u64 {
        (self.entries.len() * std::mem::size_of::<Asked>()
            + self.snapshots.values().map(Placement::host_bytes).sum::<usize>()) as u64
    }
}

/// What the census keeps: its figures.
pub struct Census {
    pub widths: Vec<(usize, Replayed)>,
    /// The GPU driver's level, read across the first forward alone: a longer window books whatever else allocates.
    pub gpu_driver: u64,
    /// The drafter's stream region; zero where none drafts.
    pub stream: u64,
    pub state_unit: u64,
    pub record_bytes: u64,
}

/// Per arm of [`arms_of`]: the larger span a width's forward needs twice over (the second a forward behind the
/// first) at its recorded contexts, and the spread between those spans; the region their sum.
#[derive(Clone, Copy, Default)]
pub struct Replayed {
    pub by_arm: [u64; 2],
    pub spread: [u64; 2],
}

impl Replayed {
    /// The larger arm's span plus its spread: the interval the census measured is the margin a run's layout gets.
    pub(crate) fn most(&self) -> u64 {
        self.by_arm.iter().zip(&self.spread).map(|(span, spread)| span + spread).max().unwrap_or(0)
    }
}

pub(crate) static CENSUS: OnceLock<Census> = OnceLock::new();

impl Census {
    /// A width's larger arm, or decode's if larger: the phases never overlap, so one region covers the largest.
    fn region_of(&self, at: Replayed) -> u64 {
        let decode = self.widths.iter().find(|&&(w, _)| w == 1).map_or(0, |&(_, d)| d.most());
        at.most().max(decode)
    }
}

/// The census's figures; `None` where it has not run.
pub fn census() -> Option<&'static Census> {
    CENSUS.get()
}

/// Sealing is the plan's choice, so prefill records both arms; decode's other routes need the resolver a census lacks.
pub fn arms_of(width: usize) -> &'static [Arm] {
    if width <= 1 {
        &[Arm::Sealed]
    } else {
        &[Arm::Sealed, Arm::Carrier]
    }
}

/// The narrowest recorded width's region at or above `width`; `None` before a record or past the widest.
pub fn region_at(width: usize) -> Option<u64> {
    let census = census()?;
    census.widths.iter().find(|&&(w, _)| w >= width).map(|&(_, at)| census.region_of(at))
}

/// Decode's bound where nothing is lent, the narrowest prefill region: the one-token figure has no room across steps.
pub fn idle_decode_region() -> Option<u64> {
    prefill_regions()?.first().map(|&(_, region)| region)
}

/// The plan's chunk choices; `None` before a record.
pub fn prefill_regions() -> Option<Vec<(usize, u64)>> {
    let census = census()?;
    Some(census.widths.iter().filter(|&&(w, _)| w > 1).map(|&(w, at)| (w, census.region_of(at))).collect())
}

/// What `f` asks the pools for, the record armed around it; what it holds is freed after and its work discarded.
pub(crate) fn recorded<T, H>(
    device: &Device, f: impl FnOnce(&HashMap<usize, Placement>) -> (T, H),
) -> Result<(T, Record)> {
    let snapshots = oscar_offload::pools::pools().map(|p| p.snapshots()).unwrap_or_default();
    candle_metal_kernels::metal::pool::set_census(true);
    let (out, held) = f(&snapshots);
    drop(held);
    candle_metal_kernels::metal::pool::set_census(false);
    // Nothing was submitted: until discarded, a pool's fence keeps the census's set mapped under every later peak.
    if let Device::Metal(md) = device {
        md.discard_commands()?;
    }
    Ok((out, Record { entries: oscar_offload::census::take_record(), snapshots }))
}

/// On the code's switch points (256 leaves the matvec path, 512 and 2048 change kernels, 4096 the recurrent slicing).
pub const RECORDED_WIDTHS: &[usize] = &[1, 256, 512, 1024, 2048, 4096, 8192, 16384];

/// What a census cost: its forwards, their encoding with the commands' discard, and the replays.
pub struct Spent {
    forwards: usize,
    encoded: Duration,
    replayed: Duration,
}

/// One forward's region on the largest device queue from the starting placements, run twice over.
fn span_of(forward: &[Asked], snapshots: &HashMap<usize, Placement>, width: usize, arm: Arm) -> u64 {
    let mut queues: Vec<usize> = forward.iter().map(|e| e.queue).filter(|&q| q < STREAM_QUEUE_KEYS).collect();
    queues.sort_unstable();
    queues.dedup();
    let empty = Placement::default();
    let passes = [(width, arm), (width, arm)];
    queues
        .iter()
        .map(|&q| region_peak(forward, snapshots.get(&q).unwrap_or(&empty), q, &passes).high * PAGE)
        .max()
        .unwrap_or(0) as u64
}

/// The census keeps its figures and stage two solves the plan from them; no record refuses the load, as none stands in.
pub fn run_at_load() {
    // The window runs to the report's end, split by the line: the report's replays can cost what the forwards do.
    let began = Instant::now();
    let ran = run(RECORDED_WIDTHS);
    // The drafter's recorded width runs on its thread before the plan, so the stream's region is in the plan's sum.
    if ran.is_ok() {
        if let Some(pools) = oscar_offload::pools::pools() {
            pools.plan_census_stream_region();
        }
    }
    crate::drafter::spawn_for_record();
    match ran {
        Ok((mut census, rec, spent)) => {
            census.stream = crate::drafter::thread::STREAM_REGION.get().copied().unwrap_or(0);
            census.state_unit = crate::gpu::state_pool::installed().map_or(0, |p| p.object_bytes() as u64);
            let reporting = Instant::now();
            report(&census, &rec);
            crate::engine::eval::hooks().census_kept(&census, &rec);
            // The window's rest, past the forwards, replays and report, is the load's own work.
            let took = |t: Duration| mistralrs_core::live::duration_text(t.as_secs_f64());
            tracing::info!(target: "oscar::census",
                "{} forwards at {} prompt widths in {} (encoding {}, replays {}, report {}); plan bounds every \
                 allocation from record",
                spent.forwards,
                census.widths.len(),
                took(began.elapsed()),
                took(spent.encoded),
                took(spent.replayed),
                took(reporting.elapsed()),
            );
            census.record_bytes = rec.host_bytes();
            drop(rec);
            let _ = crate::load::census::CENSUS.set(census);
        }
        Err(e) => {
            panic!("memory plan REFUSED: no record of model's forward ({e:#}); plan books working region from it")
        }
    }
    crate::load::stage_two();
}

/// The census line, per width and arm; the asked and held figures and a region's holders are the eval's.
fn report(census: &Census, rec: &Record) {
    // MB, as a GB to two places rounds the smaller figures that decide a chunk width to nothing.
    let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
    // Each recorded width's region on its larger arm: the only points the planner evaluates.
    let per_width: String = census.widths.iter().map(|&(w, at)| format!(" {w}:{:.0}", mib(at.most()))).collect();
    // Both arms of each prefill width, each its larger span plus the spread of its contexts: the plan takes the
    // larger arm, and which one says what the layer kind costs.
    let arms: String = census
        .widths
        .iter()
        .filter(|&&(w, _)| w > 1)
        .map(|&(w, at)| {
            format!(
                " {w}:{:.0}+{:.0}/{:.0}+{:.0}",
                mib(at.by_arm[0]),
                mib(at.spread[0]),
                mib(at.by_arm[1]),
                mib(at.spread[1])
            )
        })
        .collect();
    // The widths run of every model's list, then the measured terms the plan books beside the region.
    crate::detail!(target: "oscar::census",
        "{} buffers at {} of {} widths; widths region MB:{per_width}; arms sealed/carrier MB:{arms}; \
         GPU driver {} MB; drafter's stream {} MB; sequence state {} MB",
        rec.entries.len(),
        census.widths.len(),
        RECORDED_WIDTHS.len(),
        census.gpu_driver >> 20,
        census.stream >> 20,
        census.state_unit >> 20,
    );
}

/// Records every width the model, charge and stage-one region allow, and decides nothing: the plan chooses.
pub fn run(widths: &[usize]) -> Result<(Census, Record, Spent)> {
    // Released when the census ends: uncommitted buffers free only when deallocated, and this thread drains no pool.
    let _pool = crate::gpu::AutoreleaseScope::push();
    let loaded = oscar_models::loaded_model().context("the census needs the model, which is built before it runs")?;
    let model = loaded.arch.clone();
    let device = model.device().clone();
    // The forward drives the routed chain over the expert pool's own rows.
    anyhow::ensure!(
        crate::engine::offload_dispatcher().is_some(),
        "the census needs the expert pool, which the installer builds before it"
    );

    // No width past admission's input bound; one the stage-one region cannot hold ends the record, those below kept.
    let widest = crate::config::charge().max_input_tokens as usize;
    let mut spent = Spent { forwards: 0, encoded: Duration::ZERO, replayed: Duration::ZERO };
    let mut figures: Vec<(usize, Replayed)> = Vec::new();
    let mut gpu_driver: Option<u64> = None;
    // The one flag: the record takes entries, each layer the noted arm whatever its seal; one-shot provisioning waits.
    let (ran, rec) = recorded(&device, |snapshots| {
        let mut kept: Vec<Kept> = Vec::new();
        let ran = (|| -> Result<()> {
            for &width in widths {
                // A second chunk: twice the width of context.
                if width == 0 || width > widest.max(1) || width.saturating_mul(2) > model.max_seq_len() {
                    continue;
                }
                // Entries from here are this width's and arm's, so none held from the width before reads as this one's.
                let mut at = Replayed::default();
                for (i, &arm) in arms_of(width).iter().enumerate() {
                    record::note_width(width, arm);
                    // A prompt chunk lays out differently at the prompt's start and at the admitted context, and a run's
                    // layout differs again: each context's span is replayed on its own, the region the larger plus
                    // their spread, the interval the census measured. A decode token once.
                    let mut spans: Vec<u64> = Vec::new();
                    for offset in contexts_of(model.as_ref(), width) {
                        // Released with its width, so the GPU driver holds no more than a run at that width.
                        let scope = crate::gpu::AutoreleaseScope::push();
                        let before = gpu_driver.is_none().then(crate::machine::foreign_gpu_now);
                        let from = record::peek(<[Asked]>::len);
                        let began = Instant::now();
                        let ran = trunk_step(model.as_ref(), &device, width, offset)
                            .with_context(|| format!("the census forward at {width} tokens ({arm:?}) from {offset}"));
                        if let Some(before) = before {
                            gpu_driver = Some(crate::machine::foreign_gpu_now().saturating_sub(before));
                        }
                        if let candle_core::Device::Metal(md) = &device {
                            md.discard_commands()?;
                        }
                        drop(scope);
                        (spent.forwards, spent.encoded) = (spent.forwards + 1, spent.encoded + began.elapsed());
                        kept.push(ran?);
                        let began = Instant::now();
                        spans.push(record::peek(|rec| span_of(&rec[from..], snapshots, width, arm)));
                        spent.replayed += began.elapsed();
                    }
                    let (least, most) = (spans.iter().min().copied(), spans.iter().max().copied());
                    at.by_arm[i] = most.unwrap_or(0);
                    at.spread[i] = most.unwrap_or(0) - least.unwrap_or(0);
                }
                figures.push((width, at));
            }
            Ok(())
        })();
        (ran, kept)
    })?;
    // A failed width keeps the widths below it, and is said, so the record is never read as the whole set.
    if let Err(e) = ran {
        tracing::info!(target: "oscar::census",
            "stopped after {} widths - {e:#}; plan evaluates widths below only",
            figures.len()
        );
    }
    Ok((
        Census { widths: figures, gpu_driver: gpu_driver.unwrap_or(0), stream: 0, state_unit: 0, record_bytes: 0 },
        rec,
        spent,
    ))
}

/// The sampler's output and presence context, kept so the second pass holds the first's, as a queued step does.
struct Kept {
    _out: Tensor,
    _seen: Tensor,
}

/// The device sampler's widest row: sampled, under every filter, with a presence penalty.
const SAMPLED: oscar_kernels::DeviceSampleSpec = oscar_kernels::DeviceSampleSpec {
    temperature: Some(1.0),
    top_k: oscar_kernels::SAMPLE_KMAX,
    top_p: 0.9,
    min_p: 0.1,
    presence_penalty: 1.0,
};

/// The contexts `width` tokens are recorded at: a prompt chunk at the prompt's start and ending at the admitted
/// context (what grows the gathers), a decode token once, where its attention splits most.
fn contexts_of(model: &dyn GgufArch, width: usize) -> Vec<usize> {
    let charge = crate::config::charge();
    let admitted = ((charge.max_input_tokens + charge.max_output_tokens) as usize).min(model.max_seq_len());
    let block_size = crate::plan::paged_block_size() as usize;
    if width == 1 {
        // Decode's attention scratch falls and rises with context by its split count: recorded where it splits most.
        return vec![oscar_kernels::paged::widest_decode_context(admitted, block_size).saturating_sub(1).max(1)];
    }
    let end = admitted.saturating_sub(width).max(width);
    vec![0, end]
}

/// `width` tokens from `offset`, or one decode token at it.
fn trunk_step(model: &dyn GgufArch, device: &Device, width: usize, offset: usize) -> Result<Kept> {
    let charge = crate::config::charge();
    let admitted = ((charge.max_input_tokens + charge.max_output_tokens) as usize).min(model.max_seq_len());
    let block_size = crate::plan::paged_block_size() as usize;
    // Spread ids, never zero: a zero hashes as end-of-text in some architectures' n-gram gather, a forward no request runs.
    let ids: Vec<u32> = (0..width).map(|i| (i % 1024) as u32 + 1).collect();
    let input = Tensor::from_vec(ids, (1, width), device)?;
    let seqlen_offsets = [offset];
    let context_lens = vec![(width.saturating_sub(1), 1)];

    let slot = recurrent_slot(model, device, offset)?;
    // The engine's own builder, over block ids of the census's choosing, so what a step uploads is recorded as made.
    let blocks = (offset + width).div_ceil(block_size);
    let mut rows = PagedRows::default();
    rows.push((0..blocks).collect(), offset..offset + width, block_size);
    let on = std::slice::from_ref(device);
    let meta = if width == 1 { rows.decode_step(None, block_size, on)? } else { rows.prompt_chunk(false, on)? };
    // Stubs: the host reads only their length and no kernel reading them runs, so the census precedes the KV pool.
    let tables = {
        let null = Tensor::zeros(blocks, DType::I64, device)?;
        (0..model.num_layers()).map(|_| (null.clone(), null.clone())).collect()
    };
    // The sampler's output stands before the step, as in a decoding sequence.
    let sampled = oscar_kernels::sample_output(device)?;
    // The step's own tail, as the record is the step: the last row sampled on the device, a second row beside it.
    let out = model.forward(&input, &seqlen_offsets, context_lens, Some((tables, &meta))).and_then(|logits| {
        let vocab = logits.dim(candle_core::D::Minus1)?;
        let row = logits.flatten_all()?.narrow(0, logits.elem_count() - vocab, vocab)?.to_dtype(DType::F32)?;
        let seen = oscar_kernels::sample_context(device, vocab, &[])?;
        oscar_kernels::sample_on_device(&row, &SAMPLED, 1, admitted as u64, &sampled, Some(&seen))?;
        row.affine(1.0, 0.0)?;
        Ok(seen)
    });
    if let Some(idx) = slot {
        let mut hybrid = model.cache().hybrid();
        hybrid.free_seq(idx);
        // The indices go with the slot: left set, their tensor stays live in the region, which cannot follow the plan.
        hybrid.set_state_indices(None);
    }
    Ok(Kept { _out: sampled, _seen: out? })
}

/// A hybrid's forward needs state indices: a slot as chunked prefill takes, `offset` tokens in so each layer continues.
fn recurrent_slot(model: &dyn GgufArch, device: &Device, offset: usize) -> Result<Option<usize>> {
    if !model.cache().is_hybrid() {
        return Ok(None);
    }
    let mut hybrid = model.cache().hybrid();
    let Some(idx) = hybrid.allocate_seq()? else {
        return Ok(None);
    };
    #[allow(clippy::cast_possible_truncation)]
    hybrid.set_state_indices_from_host(vec![idx as u32], device)?;
    for layer in &mut hybrid.caches {
        if let Some(pool) = layer.as_recurrent_pool_mut() {
            pool.set_seqlen_offset(idx, offset);
        }
    }
    Ok(Some(idx))
}

#[cfg(test)]
mod tests {
    /// The split count falls and rises with the context, so the census records decode at the widest.
    #[test]
    fn decode_is_recorded_where_it_splits_most() {
        use oscar_kernels::paged::{decode_splits, widest_decode_context};
        assert_eq!(decode_splits(26_624, 32).0, 208);
        assert_eq!(decode_splits(15_362, 32).0, 241);
        assert_eq!(decode_splits(24_576, 32).0, 256);
        let widest = widest_decode_context(26_624, 32);
        assert_eq!(widest, 24_576);
        assert!((1..=26_624).all(|c| decode_splits(c, 32).0 <= decode_splits(widest, 32).0));
        // Where the admitted context is itself the widest, it stays.
        assert_eq!(widest_decode_context(32_768, 32), 32_768);
    }
}
