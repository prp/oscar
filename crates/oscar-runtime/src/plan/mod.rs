//! The plan, a closed form: geometry, record, budget and charge in; every pool's size and the working memory out.

mod admit;
mod charge;
pub mod chunks;
pub mod host_line;

use crate::config::{watchdog_floor_bytes, KvCharge};
use crate::machine::GB;
pub use admit::*;
pub use charge::*;
pub use oscar_models::layout::*;

use anyhow::Result;

use host_line::HostLine;

/// The working memory the plan books; before the load's record, the host line alone.
#[derive(Debug, Clone, Copy)]
pub struct Working {
    pub host: HostLine,
    /// The engine's region, the record replayed at the recorded width.
    pub region: u64,
    /// What the GPU driver holds for this process, read across the census's first forward; the GPU ceiling counts it.
    pub gpu_driver: u64,
    /// The drafter's stream region at its recorded width; zero where nothing drafts.
    pub stream: u64,
    pub recorded_width: Option<usize>,
}

impl Working {
    pub fn bytes(&self) -> u64 {
        self.host.bytes().saturating_add(self.region).saturating_add(self.gpu_driver).saturating_add(self.stream)
    }
}

/// The load solves at the widest ([`widest_width`]) and a crossing at that one, so every chunk term is one width's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordedWidth {
    pub width: usize,
    pub region: u64,
}

/// Chosen once at the load's tail from the whole record ([`widest_width`]).
pub(crate) static WIDEST: std::sync::OnceLock<Option<RecordedWidth>> = std::sync::OnceLock::new();

/// The widest prefill width; `None` before a record or where no recorded width solves.
pub fn chosen_width() -> Option<RecordedWidth> {
    WIDEST.get().copied().flatten()
}

/// The widest recorded width whose launch plan (`solve`) reads each paged layer once: past one token batch, only while
/// the smallest offloading bank's own slots (whole segments less the sealed stacks) exceed a layer's experts and the
/// executing waves' pins, the partition the launch runs.
pub fn widest_width(
    stats: &ModelLayoutStats, regions: &[(usize, u64)], solve: impl Fn(RecordedWidth) -> Option<MemPlan>,
) -> Option<RecordedWidth> {
    let e = stats.num_experts;
    let once = |r: &RecordedWidth, plan: MemPlan| {
        let sealed = plan.block_count - plan.carrier_layers;
        let classes = stats.expert_classes();
        let paged = classes.iter().filter(|c| c.split(sealed).1 > 0);
        let own =
            paged.map(|c| c.held(sealed, e, plan.carrier_residual_slots, plan.segment_slots) - c.split(sealed).0 * e);
        let own = own.min();
        let width = crate::pool::policy::prefill_batch(e as u32, own.unwrap_or(0).min(u32::MAX as usize) as u32);
        let pinned = crate::pool::policy::PREFILL_CB_WAVES * if width == 0 { e } else { width as usize };
        r.width <= crate::pool::policy::PREFILL_BATCH_TOKENS as usize || own.is_none_or(|own| own > e + pinned)
    };
    let widths = regions.iter().map(|&(width, region)| RecordedWidth { width, region });
    let chosen = widths.filter(|r| solve(*r).is_some_and(|plan| once(r, plan))).max_by_key(|r| r.width);
    let said = chosen.map_or("none solved".to_string(), |r| format!("{} tokens", r.width));
    crate::detail!(target: "oscar::planner", "widest prefill width {said}");
    chosen
}

fn working(stats: &ModelLayoutStats, budget: u64, recorded: Option<RecordedWidth>, drafter: bool) -> Working {
    let host = crate::plan::host_line::host_line(stats, budget);
    match recorded {
        None => Working { host, region: 0, gpu_driver: 0, stream: 0, recorded_width: None },
        Some(RecordedWidth { width, region }) => {
            let census = crate::load::census::census();
            Working {
                host,
                region,
                gpu_driver: census.map_or(0, |c| c.gpu_driver),
                stream: if drafter { census.map_or(0, |c| c.stream) } else { 0 },
                recorded_width: Some(width),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemPlan {
    pub hot_budget: usize,
    pub block_count: usize,
    /// A segment's slots per layer of a bank; `None` before the size is fixed, where slots are priced exactly.
    pub segment_slots: Option<usize>,
    /// Empty before the segment size is fixed.
    pub bank_segments: Vec<BankSegments>,
    /// The last `carrier_layers` dispatch layers page; the rest are sealed at their full stacks.
    pub carrier_layers: usize,
    /// Uniform across the carrier layers ([`derive_carriers`]); `0` when nothing offloads.
    pub carrier_residual_slots: usize,
    pub num_experts: usize,
    pub expert_pool_bytes: u64,
    pub non_expert_bytes: u64,
    /// The block objects' bytes, each in whole pages.
    pub kv_cache_bytes: u64,
    pub kv_blocks: u64,
    /// As asked: every plan's guarantee, and an offloaded plan's KV term.
    pub kv_charge: KvCharge,
    /// The header's `*.context_length`, the most a guarantee may promise; 0 where the header omits it.
    pub context_bound: u64,
    pub sequence_state_bytes: u64,
    /// [`LIVE_SEQUENCES`] × 2 + `paged_snapshot_entries`, one more while the drafter's memory is booked.
    pub sequence_state_slots: u64,
    /// The record's, or the header's bound before a record ([`sequence_state_unit`]).
    pub sequence_state_unit: u64,
    /// The mistral.rs fork's snapshot cap: each live sequence's ladder ([`states`]).
    pub paged_snapshot_entries: u64,
    /// Zero when the context store is off.
    pub store_staging_bytes: u64,
    /// At the budget's index spaces.
    pub pool_tables_bytes: u64,
    pub working: Working,
    /// The budget solved at, or a stated shape's own total ([`PlanInputs::shape`]).
    pub mem_budget_bytes: u64,
    /// False is the sealed, fully resident regime: the same dispatcher, nothing offloaded.
    pub offloading: bool,
    pub total_params: u64,
    /// `None` where the working region lends nothing, as before a record and where nothing offloads.
    pub decode: Option<DecodePlan>,
    /// The one-token region where the plan lends above it, else the narrowest prefill width's; zero before a record.
    pub decode_bound: u64,
    /// Reported only: layers decode's solve would seal that this width keeps paged, as no stack seals in lent memory.
    pub kept_paged: usize,
}

/// The phases do not overlap: through decode, the region above its bound is lent to paged layers in whole segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodePlan {
    pub hot_budget: usize,
    pub lent_bytes: u64,
}

impl MemPlan {
    pub fn total_allocated_bytes(&self) -> u64 {
        self.non_expert_bytes
            + self.expert_pool_bytes
            + self.kv_cache_bytes
            + self.sequence_state_bytes
            + self.store_staging_bytes
            + self.pool_tables_bytes
            + self.working.bytes()
    }

    /// What the GPU ceiling counts: decode's bound twice, the region's first bytes being an object of their own.
    pub fn metal_bytes(&self) -> u64 {
        self.total_allocated_bytes().saturating_sub(self.working.host.bytes()).saturating_add(self.decode_bound)
    }

    /// A paged layer's slots through decode: the pool's own and what the working region lends.
    pub fn decode_hot_budget(&self) -> usize {
        self.decode.map_or(self.hot_budget, |d| d.hot_budget)
    }

    /// All but the host line and the GPU driver's level, which the GPU ceiling counts and the wired set does not.
    pub fn locked_bytes(&self) -> u64 {
        self.total_allocated_bytes() - self.working.host.bytes() - self.working.gpu_driver
    }

    /// (X in, Y out): the charge within the header's bound, whatever a resident plan's pool holds past it.
    pub fn guarantee(&self) -> (u64, u64) {
        guarantee_of(self.kv_charge, self.context_bound)
    }

    pub fn pool_shape(&self) -> String {
        let (n, per_layer) = (self.block_count, self.num_experts);
        match self.carrier_layers {
            0 => format!("all {n} layers sealed, {per_layer} experts each"),
            k if k == n => format!("{} slots/layer in all {k} layers", self.hot_budget),
            k => format!(
                "{} slots/layer in {}-{}; 0-{} sealed, {per_layer} experts each",
                self.carrier_residual_slots,
                n - k,
                n - 1,
                n - k - 1
            ),
        }
    }
}

/// The most any crossing lands on, at the told budget: what the rings and the block ids are sized at.
#[derive(Debug, Clone, Copy)]
pub struct PoolCeilings {
    pub hot_budget: usize,
    pub kv_index_blocks: u64,
}

impl PoolCeilings {
    /// Never below `plan`'s shape, at its charge: the drafter's memory is the launch's for good.
    pub fn solve(stats: &ModelLayoutStats, plan: &MemPlan, ceiling_gb: f64) -> Self {
        let at = PlanInputs::live(ceiling_gb, plan.kv_charge);
        let at_ceiling = at.solve(stats).ok();
        let hot_budget = at_ceiling.as_ref().map_or(0, MemPlan::decode_hot_budget).max(plan.decode_hot_budget());
        let kv_index_blocks = match &at_ceiling {
            Some(p) if !p.offloading => p.kv_blocks.max(plan.kv_blocks),
            _ => {
                let most = ((ceiling_gb * GB) as u64 / kv_block_object_bytes(stats)).max(plan.kv_blocks);
                largest(plan.kv_blocks, most, |n| at.blocks(n).solve(stats).is_ok())
            }
        };
        Self { hot_budget, kv_index_blocks }
    }
}

static POOL_CEILINGS: std::sync::OnceLock<PoolCeilings> = std::sync::OnceLock::new();

pub fn install_pool_ceilings(ceilings: PoolCeilings) {
    assert!(POOL_CEILINGS.set(ceilings).is_ok(), "the pools' ceilings are solved once, at the load's tail");
}

/// `None` before the load's tail.
pub fn pool_ceilings() -> Option<PoolCeilings> {
    POOL_CEILINGS.get().copied()
}

/// One bank as the plan books it: `segments` whole segments of `slots` slots each.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BankSegments {
    pub label: String,
    pub segments: usize,
    pub slots: usize,
    pub bytes: u64,
}

/// A raw-address launch declares each present segment: 120 over a layer's three banks cost nothing measurable.
pub const DECLARED_SEGMENTS_PER_BANK: usize = 40;

/// The expert pool's segment in slots per layer of a bank, fixed before the pool is built.
static SEGMENT_SLOTS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// `None` before the segment size is fixed, where every plan prices its slots exactly.
pub fn segment_slots() -> Option<usize> {
    SEGMENT_SLOTS.get().copied()
}

/// Smallest within [`DECLARED_SEGMENTS_PER_BANK`]: a divisor where all seal, a row under half a layer on a mixed bank.
pub fn derive_segment_slots(stats: &ModelLayoutStats, ceiling: &MemPlan) -> usize {
    let classes = stats.expert_classes();
    let e = stats.num_experts.max(1);
    let (sealed, residual) = (ceiling.block_count - ceiling.carrier_layers, ceiling.carrier_residual_slots);
    let by_count = (1..=e)
        .find(|&k| {
            classes.iter().all(|c| c.segments(c.slots(sealed, e, residual / k * k), k).0 <= DECLARED_SEGMENTS_PER_BANK)
        })
        .unwrap_or(e);
    if ceiling.carrier_layers == 0 {
        return (by_count..=e).find(|&k| e % k == 0).unwrap_or(e);
    }
    classes
        .iter()
        .filter(|c| matches!(c.split(sealed), (s, r) if s > 0 && r > 0))
        .map(|c| ((e - 1) / (2 * c.layers.len())).max(1))
        .fold(by_count, usize::min)
}

/// Fix the segment size from `ceiling`, the told budget's plan before the record, once; no crossing moves it.
pub fn install_segment_slots(stats: &ModelLayoutStats, ceiling: &MemPlan) -> usize {
    *SEGMENT_SLOTS.get_or_init(|| derive_segment_slots(stats, ceiling))
}

/// Below it a prefill expert wave takes the per-token matvec path, re-reading each expert per routed token.
pub const PREFILL_CHUNK_MIN_TOKENS: u64 = 256;

/// A carrier's floor in routed widths: 17 keeps every optimum and refuses each model where sealing loses at any depth.
const CARRIER_FLOOR_PER_ROUTED: usize = 17;

/// What a carrier must hold through decode.
fn carrier_floor(routed_width: usize, num_experts: usize) -> usize {
    routed_width.max(1).saturating_mul(CARRIER_FLOOR_PER_ROUTED).min(num_experts)
}

/// The first `sealed` layers sealed, the rest carriers at `residual`; whole segments once the size is fixed.
fn pool_bytes(
    classes: &[ExpertClass], experts: usize, sealed: usize, residual: usize, segment_slots: Option<usize>,
) -> u64 {
    classes.iter().map(|c| c.bytes(sealed, experts, residual, segment_slots)).sum()
}

/// The largest residual in whole segments' slots whose pool fits `available`; `None` where the sealed alone do not.
fn residual_at(
    classes: &[ExpertClass], experts: usize, sealed: usize, available: u64, segment_slots: Option<usize>,
) -> Option<usize> {
    if pool_bytes(classes, experts, sealed, 0, segment_slots) > available {
        return None;
    }
    let step = segment_slots.unwrap_or(1).max(1);
    let fits = |n: u64| pool_bytes(classes, experts, sealed, n as usize * step, segment_slots) <= available;
    Some(largest(0, (experts / step) as u64, fits) as usize * step)
}

/// A paged layer's slots with `room` bytes lent in whole segments, and their bytes; `None` where none fits.
fn lent(
    classes: &[ExpertClass], experts: usize, sealed: usize, residual: usize, k: usize, room: u64,
) -> Option<(usize, u64)> {
    let bytes = |h: usize| -> u64 {
        classes
            .iter()
            .map(|c| {
                let (own, per_segment) = c.segments(c.slots(sealed, experts, residual), k);
                let (with, _) = c.segments(c.slots(sealed, experts, h), k);
                (with - own) as u64 * per_segment as u64 * c.unit
            })
            .sum()
    };
    let paged = classes.iter().any(|c| c.split(sealed).1 > 0);
    let h = (residual / k + 1..=experts / k).map(|n| n * k).take_while(|&h| bytes(h) <= room).last();
    h.filter(|_| paged).map(|h| (h, bytes(h)))
}

/// The routed sets a decode pins at once in a bank: the command buffers encoded ahead and the one resolving.
const PINNED_SETS: usize = crate::engine::dispatcher::ENCODE_AHEAD_CBS as usize + 1;

/// A carrier's slots, one at least, for each bank to hold [`PINNED_SETS`] routed sets over its layers among carriers.
fn pinned_floor(classes: &[ExpertClass], sealed: usize, routed_width: usize) -> usize {
    let layers = classes.iter().map(|c| c.split(sealed).1).filter(|&n| n > 0).min().unwrap_or(1);
    (PINNED_SETS * routed_width).div_ceil(layers).max(1)
}

/// The fewest last layers (late layers route concentrated sets) whose uniform residual holds every carrier's floor,
/// else every layer a carrier; `None` where its banks cannot hold the pinned sets.
fn derive_carriers(
    classes: &[ExpertClass], block_count: usize, num_experts: usize, routed_width: usize, available: u64,
    segment_slots: Option<usize>, room: u64,
) -> Option<(usize, usize)> {
    let (l, e) = (block_count, num_experts);
    if pool_bytes(classes, e, l, 0, segment_slots) <= available {
        return Some((0, 0));
    }
    let floor = carrier_floor(routed_width, num_experts);
    // The floor is held on decode's slots, own and lent; its own must hold the pinned sets.
    let fit = |sealed: usize| {
        let r = residual_at(classes, e, sealed, available, segment_slots)?;
        (r >= pinned_floor(classes, sealed, routed_width)).then_some(r)
    };
    let decode = |sealed: usize, r: usize| {
        segment_slots.and_then(|k| lent(classes, e, sealed, r, k, room)).map_or(r, |(h, _)| h)
    };
    let carriers = (1..=l).find(|&c| fit(l - c).is_some_and(|r| decode(l - c, r) >= floor)).unwrap_or(l);
    fit(l - carriers).map(|r| (carriers, r))
}

/// What the closed form is solved at beside the geometry; `shape` is (carriers, slots), `decode_bounds` (step, idle).
#[derive(Debug, Clone, Copy)]
pub struct PlanInputs {
    pub mem_gb: f64,
    pub charge: KvCharge,
    pub recorded: Option<RecordedWidth>,
    pub segment_slots: Option<usize>,
    pub kv_blocks: Option<u64>,
    pub shape: Option<(usize, usize)>,
    pub decode_bounds: (Option<u64>, Option<u64>),
}

impl PlanInputs {
    pub fn at(mem_gb: f64, charge: KvCharge, recorded: Option<RecordedWidth>, segment_slots: Option<usize>) -> Self {
        // Decode's bound is the one-token region, drafting models too: every engine step is an ordinary one.
        let decode_bounds = (crate::load::census::region_at(1), crate::load::census::idle_decode_region());
        Self { mem_gb, charge, recorded, segment_slots, kv_blocks: None, shape: None, decode_bounds }
    }

    /// At the load's widest prefill width and the pool's segment size.
    pub fn live(mem_gb: f64, charge: KvCharge) -> Self {
        Self::at(mem_gb, charge, chosen_width(), segment_slots())
    }

    pub fn blocks(self, kv_blocks: u64) -> Self {
        Self { kv_blocks: Some(kv_blocks), ..self }
    }

    /// A shape the pools hold, as a re-land states it: its budget its own total, and no floor refuses it.
    pub fn shape(self, carrier_layers: usize, hot_budget: usize) -> Self {
        Self { shape: Some((carrier_layers, hot_budget)), ..self }
    }

    /// At the given block count, else the charge's; a resident plan's KV line is the remainder, its charge as asked.
    pub fn solve(self, stats: &ModelLayoutStats) -> Result<MemPlan> {
        let width = self.recorded.map(|r| r.width);
        let kv_blocks = self.kv_blocks.unwrap_or_else(|| charge_blocks(stats, self.charge, width));
        let plan = closed_form(stats, &self, kv_blocks)?;
        // A launch sealing every layer never books the drafter, which would idle; a stated count is a crossing's,
        // which keeps the launch's booking.
        if !plan.offloading && self.charge.drafter_booked && self.kv_blocks.is_none() {
            return Self { charge: self.charge.unbooked(), ..self }.solve(stats);
        }
        if plan.offloading || self.kv_blocks.is_some() {
            return Ok(plan);
        }
        let others = plan.total_allocated_bytes() - plan.kv_cache_bytes;
        let remainder = plan.mem_budget_bytes.saturating_sub(others) / kv_block_object_bytes(stats).max(1);
        if remainder <= kv_blocks {
            return Ok(plan);
        }
        closed_form(stats, &self, remainder)
    }
}

/// Per phase: decode's plan beside the pool's own given decode's bound; before a record or nothing paged, one plan.
fn closed_form(stats: &ModelLayoutStats, at: &PlanInputs, kv_blocks: u64) -> Result<MemPlan> {
    let PlanInputs { mem_gb, charge, recorded, segment_slots, shape, decode_bounds: (bound, idle_bound), .. } = *at;
    // What decode lends: the mapped region above the bound, given a record, a segment size and a bound under it.
    let mapped = recorded.map_or(0, |r| oscar_offload::segments::padded_unit(r.region as usize) as u64);
    let bound = bound.map(|b| oscar_offload::segments::padded_unit(b as usize) as u64).filter(|&b| b < mapped);
    let room = bound.filter(|_| segment_slots.is_some()).map_or(0, |b| mapped - b);
    let mem_budget_bytes = (mem_gb * GB) as u64;
    let working = working(stats, mem_budget_bytes, recorded, charge.drafter_booked);
    let kv_cache_bytes = kv_blocks.saturating_mul(kv_block_object_bytes(stats));
    let staging_bytes = store_staging_bytes(stats, charge, working.recorded_width.map_or(0, |w| w as u64));
    let pool_tables_bytes = crate::plan::host_line::pool_tables_bytes(stats, mem_budget_bytes);
    let state_unit = sequence_state_unit(stats);
    let (state_slots, snapshot_entries) = if state_unit > 0 { states(charge) } else { (0, 0) };
    let reserved = stats
        .non_expert_bytes
        .saturating_add(kv_cache_bytes)
        .saturating_add(staging_bytes)
        .saturating_add(pool_tables_bytes)
        .saturating_add(working.bytes())
        .saturating_add(state_slots.saturating_mul(state_unit));
    let available_for_experts = mem_budget_bytes.saturating_sub(reserved);
    let classes = stats.expert_classes();
    let (e, k) = (stats.num_experts, stats.experts_per_token);
    let derive = |available, room| derive_carriers(&classes, stats.block_count, e, k, available, segment_slots, room);
    let shape = match shape {
        Some((carriers, residual)) => Some((carriers.min(stats.block_count), if carriers == 0 { 0 } else { residual })),
        None => derive(available_for_experts, room),
    };
    let Some((carrier_layers, carrier_residual_slots)) = shape else {
        let floor = pinned_floor(&classes, 0, k) as u64;
        anyhow::bail!(
            "Not enough memory for this model: --mem {mem_gb:.2} GB too small, needs {:.2} GB, of it non-routed weights {:.2} GB, KV for one {}+{}-token \
             sequence {:.2} GB, working memory {:.2} GB, expert pool {floor} slots/layer {:.2} GB (every bank holds \
             {PINNED_SETS} routed sets of top-{k} in flight) - raise --mem or lower context window",
            (reserved + floor * stats.bytes_per_hb_unit) as f64 / GB,
            stats.non_expert_bytes as f64 / GB,
            charge.max_input_tokens,
            charge.max_output_tokens,
            kv_cache_bytes as f64 / GB,
            working.bytes() as f64 / GB,
            (floor * stats.bytes_per_hb_unit) as f64 / GB,
        );
    };
    // What the pacer, oracle and run-ahead are sized to: the carrier residual, or the full stack where all seal.
    let hot_budget = if carrier_layers == 0 { e } else { carrier_residual_slots };
    let sealed_layers = stats.block_count - carrier_layers;
    let expert_pool_bytes = pool_bytes(&classes, e, sealed_layers, carrier_residual_slots, segment_slots);
    let decode = segment_slots.filter(|_| room > 0).and_then(|k| {
        let (hot_budget, lent_bytes) = lent(&classes, e, sealed_layers, carrier_residual_slots, k, room)?;
        Some(DecodePlan { hot_budget, lent_bytes })
    });
    let kept_paged = match room {
        0 => 0,
        room => derive(available_for_experts + room, 0).map_or(0, |(c, _)| carrier_layers.saturating_sub(c)),
    };
    let bank_segments = segment_slots.map_or_else(Vec::new, |k| {
        classes
            .iter()
            .map(|c| {
                let (segments, slots) = c.segments(c.slots(sealed_layers, e, carrier_residual_slots), k);
                BankSegments { label: c.label(), segments, slots, bytes: (segments * slots) as u64 * c.unit }
            })
            .collect()
    });
    let mut plan = MemPlan {
        hot_budget,
        block_count: stats.block_count,
        segment_slots,
        bank_segments,
        carrier_layers,
        carrier_residual_slots,
        num_experts: e,
        expert_pool_bytes,
        non_expert_bytes: stats.non_expert_bytes,
        kv_cache_bytes,
        kv_blocks,
        kv_charge: charge,
        context_bound: stats.context_length,
        sequence_state_bytes: state_slots.saturating_mul(state_unit),
        sequence_state_slots: state_slots,
        sequence_state_unit: state_unit,
        paged_snapshot_entries: snapshot_entries,
        store_staging_bytes: staging_bytes,
        pool_tables_bytes,
        working,
        mem_budget_bytes,
        offloading: carrier_layers > 0 && carrier_residual_slots < e,
        total_params: stats.total_params,
        decode,
        decode_bound: if decode.is_some() { bound } else { idle_bound.filter(|&b| b < mapped) }.unwrap_or(0),
        kept_paged,
    };
    if at.shape.is_some() {
        plan.mem_budget_bytes = plan.total_allocated_bytes();
    }
    Ok(plan)
}

/// Each part's GB, its parts indented, the total against the budget; beneath, what has no GB of its own.
pub fn render_mem_plan(plan: &MemPlan) -> String {
    let ((x, y), w, host) = (plan.guarantee(), &plan.working, plan.working.host);
    let charged = format!("charged at {} tokens", plan.kv_charge.total_tokens());
    let bound = if x + y == plan.context_bound { ", model's context bound" } else { "" };
    let provenance = match (plan.offloading, plan.kv_charge.bench) {
        (true, true) => charged,
        (true, false) => format!("guarantee {x} in + {y} out"),
        (false, true) => format!("remainder; {charged}"),
        (false, false) => format!("remainder; guarantee {x} in + {y} out{bound}"),
    };
    let mut rows: Vec<(usize, &str, u64, String)> = vec![
        (0, "non-routed weights", plan.non_expert_bytes, String::new()),
        (0, "expert pool", plan.expert_pool_bytes, plan.pool_shape()),
    ];
    for b in &plan.bank_segments {
        rows.push((1, &b.label, b.bytes, format!("{} segments of {} slots", b.segments, b.slots)));
    }
    let blocks = format!("{} blocks of {} tokens, {provenance}", plan.kv_blocks, paged_block_size());
    rows.push((0, "KV cache", plan.kv_cache_bytes, blocks));
    if plan.sequence_state_bytes > 0 {
        let fixed = plan.sequence_state_slots - plan.paged_snapshot_entries;
        let recorded = plan.sequence_state_unit == crate::load::census::census().map_or(0, |c| c.state_unit);
        let states = format!(
            "{} x {:.1} MB: {LIVE_SEQUENCES} live x 2{}, {} retained{}",
            plan.sequence_state_slots,
            plan.sequence_state_unit as f64 / (1u64 << 20) as f64,
            if fixed > LIVE_SEQUENCES * 2 { ", drafter hand-off" } else { "" },
            plan.paged_snapshot_entries,
            if recorded { "" } else { "; sized from header" },
        );
        rows.push((0, "sequence states", plan.sequence_state_bytes, states));
    }
    if plan.store_staging_bytes > 0 {
        rows.push((0, "context store", plan.store_staging_bytes, "staging arena".to_string()));
    }
    rows.push((0, "pools' tables", plan.pool_tables_bytes, "address tables, epochs, rows".to_string()));
    let before = if w.recorded_width.is_none() { "before record" } else { "" };
    rows.push((0, "working memory", w.bytes(), before.to_string()));
    if let Some(width) = w.recorded_width {
        rows.push((1, "region", w.region, format!("at {width}-token width")));
        rows.push((1, "GPU driver", w.gpu_driver, String::new()));
        if w.stream > 0 {
            rows.push((1, "drafter's stream", w.stream, String::new()));
        }
    }
    rows.push((1, "host", host.bytes(), String::new()));
    let mut terms = [
        ("pools", host.pools),
        ("oracle", host.oracle),
        ("KV index", host.kv_index),
        ("tokenizer", host.tokenizer),
        ("sequences", host.sequences),
        ("lookahead", host.lookahead),
        ("n-gram rows", host.ngram_rows),
        ("page tables", host.page_tables),
        ("GPU driver", host.gpu_driver_host),
        ("draft head requantised", host.draft_requant),
        ("seal readers", host.seal_readers),
    ];
    terms.sort_by(|a, b| b.1.cmp(&a.1));
    rows.extend(
        terms.into_iter().filter(|&(_, bytes)| bytes > 0).map(|(label, bytes)| (2, label, bytes, String::new())),
    );
    let budget = format!("of {:.2} GB budget", plan.mem_budget_bytes as f64 / GB);
    rows.push((0, "total", plan.total_allocated_bytes(), budget));
    let mut lines: Vec<String> = rows
        .into_iter()
        .map(|(depth, label, bytes, detail)| {
            let label = format!("{}{label}", "  ".repeat(depth + 1));
            format!("{label:<22}{:>8.3} GB  {detail}", bytes as f64 / GB).trim_end().to_string()
        })
        .collect();
    let kept_paged = match plan.kept_paged {
        0 => String::new(),
        n => format!("; {n} layers kept paged for prefill width"),
    };
    match plan.decode {
        Some(d) => lines.push(format!(
            "  decode: {} slots/paged layer, {} own + {} lent by working region ({:.3} GB; it keeps {:.3} GB)\
             {kept_paged}",
            d.hot_budget,
            plan.hot_budget,
            d.hot_budget.saturating_sub(plan.hot_budget),
            d.lent_bytes as f64 / GB,
            plan.decode_bound as f64 / GB,
        )),
        None if plan.offloading && w.recorded_width.is_none() => {
            lines.push("  decode: solved at load from record".into())
        }
        None if plan.kept_paged > 0 => lines.push(format!("  decode: pool's own slots, nothing lent{kept_paged}")),
        None => {}
    }
    let segments = plan.segment_slots.map_or(String::new(), |k| format!("; segment = {k} slots/layer/bank"));
    let mode = if crate::config::small_pool() { "; drafter off" } else { "" };
    let total = plan.block_count * plan.num_experts;
    let resident = total - plan.carrier_layers * plan.num_experts.saturating_sub(plan.carrier_residual_slots);
    lines.push(format!(
        "  experts: {resident} of {total} resident, {} read from NVMe{segments}{mode}",
        total.saturating_sub(resident),
    ));
    if host.loader > 0 {
        lines.push(format!("  beside plan: {:.3} GB loader staging, load's peak not run's", host.loader as f64 / GB));
    }
    lines.join("\n") + "\n"
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::config::KV_CHARGE_BENCH;
    use candle_core::quantized::GgmlDType;

    /// `experts` experts in each of `layers` layers, a slot in every layer `unit` bytes, a projection a third rounded down.
    pub(crate) fn routed(layers: usize, experts: usize, unit: u64) -> ModelLayoutStats {
        let projection = unit / (3 * layers as u64);
        ModelLayoutStats {
            block_count: layers,
            num_experts: experts,
            bytes_per_hb_unit: projection * 3 * layers as u64,
            expert_bytes_per_layer: vec![[projection; 3]; layers],
            kv_bytes_per_token: 30720,
            hidden_dim: 4096,
            experts_per_token: 8,
            kv_bearing_layers: layers as u64,
            expert_dtypes_per_layer: vec![[GgmlDType::Q4K; 3]; layers],
            ..Default::default()
        }
    }

    /// Layers of one cost form one class, on the gate alone.
    fn classes_of(per_layer: &[u64]) -> Vec<ExpertClass> {
        let mut classes: Vec<ExpertClass> = Vec::new();
        for (layer, &unit) in per_layer.iter().enumerate() {
            match classes.iter_mut().find(|c| c.unit == unit) {
                Some(c) => c.layers.push(layer),
                None => classes.push(ExpertClass { projection: 0, dtype: GgmlDType::Q4K, layers: vec![layer], unit }),
            }
        }
        classes
    }

    fn carriers(per_layer: &[u64], experts: usize, width: usize, available: u64, k: Option<usize>) -> (usize, usize) {
        derive_carriers(&classes_of(per_layer), per_layer.len(), experts, width, available, k, 0).expect("a pool")
    }

    /// At a recorded width and its region, a segment of `k` slots, the bench charge's blocks and decode's `bound`.
    fn recorded(
        stats: &ModelLayoutStats, k: usize, (width, region): (usize, u64), bound: Option<u64>,
    ) -> Option<MemPlan> {
        let at = PlanInputs::at(54.0, KV_CHARGE_BENCH, Some(RecordedWidth { width, region }), Some(k));
        let blocks = charge_blocks(stats, KV_CHARGE_BENCH, Some(width));
        PlanInputs { decode_bounds: (bound, None), ..at.blocks(blocks) }.solve(stats).ok()
    }

    /// The K3 shape: top-16, dtype classes of 64/28 (gate, up) and 81/11 (down) layers: the 11-layer bank holds nine
    /// routed sets in flight (144 experts) at 14 slots a layer, not at 13. The carrier solver finds the minimal K at
    /// the working-set floor; the carrier side a multiple of the size: the 397B's 142 slots at 4 a layer land at 140.
    #[test]
    fn the_pinned_floor_holds_the_routed_sets_in_flight_and_the_carrier_solver_finds_the_minimal_k() {
        let mut stats = routed(92, 896, 1 << 30);
        (stats.hidden_dim, stats.mla_absorbed, stats.experts_per_token) = (7168, true, 16);
        for (layer, dtypes) in stats.expert_dtypes_per_layer.iter_mut().enumerate() {
            let gate_up = if layer < 64 { GgmlDType::Iq1S } else { GgmlDType::Iq2XXS };
            *dtypes = [gate_up, gate_up, if layer < 81 { GgmlDType::Iq2XXS } else { GgmlDType::Iq3XXS }];
        }
        // Over every layer, and over the last 11 alone as carriers, every class there 11 wide.
        assert_eq!(
            (pinned_floor(&stats.expert_classes(), 0, 16), pinned_floor(&stats.expert_classes(), 81, 16)),
            (14, 14)
        );
        let kv = charge_blocks(&stats, KV_CHARGE_BENCH, None) * kv_block_object_bytes(&stats);
        let base = (stats.non_expert_bytes + kv + working(&stats, 0, None, false).bytes()) as f64;
        let unit = stats.bytes_per_hb_unit as f64;
        let at = |slots: f64| PlanInputs::live((base + slots * unit) / GB, KV_CHARGE_BENCH).solve(&stats);
        let err = at(13.5).expect_err("13 slots per layer over an 11-layer class must be refused");
        assert!(
            err.to_string().contains("14 slots/layer") && err.to_string().contains("9 routed sets of top-16"),
            "{err}"
        );
        assert_eq!(at(14.5).expect("14 slots per layer clear the floor").hot_budget, 14);
        // Everything fits → the whole pool seals.
        assert_eq!(carriers(&[10; 4], 8, 2, 320, None), (0, 0));
        // E=512, top-10: floor 170; the last 2 layers carry at 171, the first K whose residual clears it.
        assert_eq!(carriers(&[1; 48], 512, 10, 46 * 512 + 2 * 171, None), (2, 171));
        // A budget that would starve 5 carriers to 30 slots seals fewer layers, so no carrier sits under its floor.
        assert_eq!(carriers(&[1; 48], 512, 10, 43 * 512 + 5 * 30, None), (8, 210));
        // Flash-next's shape (403 slots a layer all-carrier): seals 32 of 48 and carries 16 at 185.
        assert_eq!(carriers(&[1; 48], 512, 10, 403 * 48, None), (16, 185));
        // The 397B (142 slots a layer over 60 layers) clears no K: sealing loses there at every depth.
        assert_eq!(carriers(&[1; 60], 512, 10, 142 * 60, None), (60, 142));
        // The 235B (E=128, top-8, 94 layers, 47 slots' worth): its floor clamps at the layer width, past any split.
        assert_eq!(carriers(&[1; 94], 128, 8, 47 * 94, None), (94, 47));
        // Heterogeneous layers price the tail carriers exactly: one expensive last layer and nine cheap sealed ones.
        assert_eq!(carriers(&[1, 1, 1, 1, 1, 1, 1, 1, 1, 100], 256, 8, 9 * 256 + 100 * 140, None), (1, 140));
        assert_eq!(carriers(&[1; 60], 512, 10, 142 * 60, Some(4)), (60, 140));
        assert_eq!(carriers(&[1; 60], 512, 10, 142 * 60, Some(1)), (60, 142));
        // Flash-next exactly priced: 27 sealed, 21 at 183; at 10 a layer the last whole segment costs a sealed layer.
        assert_eq!(carriers(&[1; 48], 512, 10, 27 * 512 + 21 * 183, None), (21, 183));
        assert_eq!(carriers(&[1; 48], 512, 10, 27 * 512 + 21 * 183, Some(10)), (22, 180));
        let classes = classes_of(&[1; 48]);
        let c = &classes[0];
        assert_eq!(c.segments(c.slots(26, 512, 180), 10).0, 36);
        assert_eq!(c.segments(c.slots(27, 512, 180), 10).0, 37);
        assert_eq!(c.segments(c.slots(27, 512, 180), 9).0, 41);
        assert_eq!(c.bytes(27, 512, 180, Some(10)), 37 * 480);
        assert_eq!(c.bytes(27, 512, 180, None), 27 * 512 + 21 * 180);
        // The plan at the segment size books the banks whole, the carrier side a multiple of it, and names them.
        let stats = routed(60, 128, 1 << 30);
        let at = |k| PlanInputs::at(40.0, KV_CHARGE_BENCH, None, k).solve(&stats).expect("plans");
        let (plan, exact) = (at(Some(4)), at(None));
        assert_eq!((plan.segment_slots, plan.hot_budget % 4, plan.bank_segments.len()), (Some(4), 0, 3));
        assert_eq!(plan.bank_segments[0].slots, 4 * 60);
        assert_eq!(plan.expert_pool_bytes, plan.bank_segments.iter().map(|b| b.bytes).sum::<u64>());
        let unit = stats.bytes_per_hb_unit / 180;
        assert!(plan.bank_segments.iter().all(|b| b.bytes == (b.segments * b.slots) as u64 * unit));
        assert!(plan.hot_budget <= exact.hot_budget && plan.hot_budget + 4 > exact.hot_budget);
    }

    /// Smallest within the declared segments: a divisor where all seal, a row under half a sealed layer when mixed. The
    /// working set before a record is the host line and no region; the pools' ceilings are the plan at the told budget.
    #[test]
    fn the_segment_size_is_bounded_by_the_row_and_the_ceilings_are_the_plan_at_the_told_budget() {
        let mut b35 = routed(40, 256, 1 << 20);
        for l in 20..40 {
            b35.expert_dtypes_per_layer[l][2] = GgmlDType::Q6K;
            b35.expert_bytes_per_layer[l][2] = b35.expert_bytes_per_layer[l][2] * 3 / 2;
        }
        let banks: Vec<_> = b35.expert_classes().iter().map(|c| (c.projection, c.layers.len())).collect();
        assert_eq!(banks, [(0, 40), (1, 40), (2, 20), (2, 20)]);
        assert_eq!(routed(48, 512, 1 << 30).expert_classes().len(), 3);
        let bound = Some(DECLARED_SEGMENTS_PER_BANK);
        // (shape, the ceiling's carriers and residual, the size the segment count alone picks, the size): the 397B's
        // 142 slots in all 60 layers, one a layer 142 segments; 128 experts over 48 sealed layers, 32 segments at 4 a
        // layer, 43 at 3; Flash-next's 21 carriers at 183, 10 a layer past half a sealed layer's 512, 5 under it; the
        // 35B's mixed down bank, 7 dividing no stack; the coder's 512 over 48 layers.
        for (stats, (carriers, residual), by_count, k) in [
            (&routed(60, 128, 1 << 30), (60, 142), 4, 4),
            (&routed(48, 128, 1 << 20), (0, 0), 4, 4),
            (&routed(48, 512, 1 << 30), (21, 183), 10, 5),
            (&b35, (0, 0), 7, 8),
            (&routed(48, 512, 1 << 20), (0, 0), 13, 16),
        ] {
            let mut ceiling = PlanInputs::at(40.0, KV_CHARGE_BENCH, None, None).solve(stats).expect("plans");
            let (classes, e, sealed) = (stats.expert_classes(), stats.num_experts, stats.block_count - carriers);
            let most = |k: usize| classes.iter().map(|c| c.segments(c.slots(sealed, e, residual / k * k), k).0).max();
            assert!(most(by_count) <= bound && most(by_count - 1) > bound, "the count alone picks {by_count}");
            if carriers == 0 {
                assert_eq!((ceiling.carrier_layers, ceiling.offloading), (0, false), "resident at the ceiling");
                for c in &classes {
                    let slots = c.slots(sealed, e, 0);
                    assert_eq!((c.segments(slots, k).0, slots % c.segments(0, k).1), (32, 0), "whole segments");
                }
            }
            (ceiling.carrier_layers, ceiling.carrier_residual_slots) = (carriers, residual);
            assert_eq!(derive_segment_slots(stats, &ceiling), k);
        }
        // At 8 no spare slot is booked; at 7 the spare row comes out of the KV line.
        let at = |k| PlanInputs::at(40.0, KV_CHARGE_BENCH, None, k).solve(&b35).expect("plans");
        let (exact, at8, at7) = (at(None), at(Some(8)), at(Some(7)));
        assert_eq!((at8.expert_pool_bytes, at8.kv_cache_bytes), (exact.expert_pool_bytes, exact.kv_cache_bytes));
        assert!(at7.expert_pool_bytes > exact.expert_pool_bytes && at7.kv_cache_bytes < exact.kv_cache_bytes);
        // The working set before a record.
        let live = |stats: &ModelLayoutStats, gb| PlanInputs::live(gb, KV_CHARGE_BENCH).solve(stats).expect("plans");
        let resident = routed(60, 128, 1 << 20);
        let budget = (54.0 * GB) as u64;
        let w = working(&resident, budget, None, false);
        assert_eq!((w.region, w.host), (0, crate::plan::host_line::host_line(&resident, budget)));
        assert_eq!((w.gpu_driver, w.stream, w.recorded_width, w.bytes()), (0, 0, None, w.host.bytes()));
        // The total carries the host share and the Metal figure leaves it out.
        let plan = live(&resident, 54.0);
        assert_eq!(plan.total_allocated_bytes() - plan.metal_bytes(), plan.working.host.bytes());
        assert!(plan.pool_tables_bytes > 0 && plan.working.recorded_width.is_none(), "the tables a Metal line");
        let gqa = routed(60, 128, 1 << 30);
        let plan = live(&gqa, 40.0);
        let own = PoolCeilings::solve(&gqa, &plan, 40.0);
        assert!(own.hot_budget == plan.hot_budget && own.kv_index_blocks >= plan.kv_blocks);
        let wider = PoolCeilings::solve(&gqa, &plan, 54.0);
        assert!(wider.hot_budget > plan.hot_budget, "a wider budget hosts more slots");
        assert!(wider.kv_index_blocks > own.kv_index_blocks, "and more blocks at the expert floor");
        // A resident model's index space is the remainder's count at each budget.
        let plan = live(&resident, 40.0);
        let own = PoolCeilings::solve(&resident, &plan, 40.0);
        assert_eq!((own.hot_budget, own.kv_index_blocks), (resident.num_experts, plan.kv_blocks));
        let wider = PoolCeilings::solve(&resident, &plan, 54.0);
        assert_eq!(wider.kv_index_blocks, live(&resident, 54.0).kv_blocks);
        assert!(wider.kv_index_blocks > own.kv_index_blocks);
    }

    /// A mixed model seals as decode's solve does while memory holds, the paged layers at the floor through decode.
    /// The widest prefill width reads each paged layer once: too few own slots pass a width over, none holding a
    /// layer gives one batch, and nothing paged takes the widest.
    #[test]
    fn decode_is_planned_on_its_own_and_the_widest_prefill_width_reads_each_paged_layer_once() {
        let bound = 128u64 << 20;
        let solve =
            |stats: &ModelLayoutStats, k, region: u64, bound| recorded(stats, k, (4096, region), bound).expect("plans");
        let mixed = routed(48, 512, 108 << 20);
        // region MB → the one plan, the pool's own, decode's slots and `kept_paged`
        for (region, one, own, decode) in [
            (600u64, (3, 240), (3, 240), None),
            (900, (3, 240), (2, 100), Some((220, 0))),
            (1843, (4, 185), (3, 80), Some((320, 1))),
        ] {
            let (p1, p2) = (solve(&mixed, 5, region << 20, None), solve(&mixed, 5, region << 20, Some(bound)));
            assert_eq!(((p1.carrier_layers, p1.hot_budget), p1.decode), (one, None), "{region} MB: the one plan");
            assert_eq!((p2.carrier_layers, p2.hot_budget), own, "{region} MB: the pool's own");
            assert_eq!(p2.decode.map(|d| (d.hot_budget, p2.kept_paged)), decode, "{region} MB: decode's");
            assert!(p2.total_allocated_bytes() <= 54 << 30, "{region} MB: the own plan beside the whole region");
            if let Some(d) = p2.decode {
                assert!(d.hot_budget >= carrier_floor(8, 512), "{region} MB: at the floor");
                assert_eq!(d.hot_budget % 5, 0, "{region} MB: whole segments");
                assert!(d.lent_bytes <= (region << 20) - bound, "{region} MB: lent in the region, above the bound");
            }
        }
        let paged = routed(60, 128, 1 << 30);
        let (one, two) = (solve(&paged, 1, 1843 << 20, None), solve(&paged, 1, 1843 << 20, Some(bound)));
        let shape = |p: &MemPlan| (p.hot_budget, p.carrier_layers, p.expert_pool_bytes);
        assert_eq!(shape(&two), shape(&one), "an all-paged model's own plan is the one plan");
        assert_eq!(two.decode.map(|d| d.hot_budget), Some(one.hot_budget + 1), "1.7 GB of GB slots: one more a layer");
        let resident = routed(60, 128, 1 << 20);
        assert!(solve(&resident, 1, 1843 << 20, Some(bound)).decode.is_none(), "a resident model lends nothing");
        let widest = |stats: &ModelLayoutStats, k: usize, regions: &[(usize, u64)]| {
            widest_width(stats, regions, |r| recorded(stats, k, (r.width, r.region), Some(128 << 20))).map(|r| r.width)
        };
        // 1.8 GB of region leaves three carriers 80 slots each: 240 own, under a layer's 512 and a wave's pins.
        let regions = [(1024usize, 600u64 << 20), (2048, 900 << 20), (4096, 1843 << 20)];
        assert_eq!(widest(&routed(48, 512, 108 << 20), 5, &regions), Some(2048), "no width holds a layer: one batch");
        // An all-paged pool of 60 layers holds a layer at every width; a region past the budget is no width.
        assert_eq!(widest(&paged, 1, &regions), Some(4096));
        assert_eq!(widest(&paged, 1, &[(4096, 1843 << 20), (8192, 60 << 30)]), Some(4096));
        assert_eq!(widest(&routed(60, 128, 1 << 20), 1, &regions), Some(4096), "nothing pages, nothing to hold");
    }
}
