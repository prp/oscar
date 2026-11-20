//! The expert pool: a bank per (projection, format) class, its reads, the sealed rows, the moves a park makes and the
//! mover settles, the loans, the policy and the counters. Built empty for the census, launched once at the plan's shape.

mod bank;
pub mod counters;
mod lending;
pub mod policy;
mod reads;
mod resolve;

pub(crate) use bank::SEAL_READERS;
pub use bank::{pin_all, pin_all_blocking, Bank, BankDtype, LayerExpertKey, Oracle, PinOrder, Plan, Resident, Seg};
pub use counters::{Count, DecodeWindow, TokenSample, COUNTERS};
pub use lending::phase_changes;
pub use policy::{RecordedSchedule, SchedulePacer};
pub use reads::{demand_clock, note_demand_tick, reset_demand_clock, FetchClass, FetchQueue, FetchReq, Lane, Readers};
pub(crate) use resolve::PrefetchThread;
pub use resolve::{commit_pins, resolve_census, BankPools, Due, PinLedger, PrefetchJob, WaveResidency};

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use candle_core::quantized::GgmlDType;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLDevice};
use oscar_offload::io::TieredFile;
use oscar_offload::segments::padded_unit;

use crate::gpu::expert_rows::{write_row, ExpertRows};
use crate::gpu::{ExpertExtent, GgufRoutedLayout, ProjectionKind, ProjectionLayout};
use policy::{decode_layers, DecodePacer, NextUseOracle};

/// What the pool refuses; the memory's and the drives' errors pass through as their own.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Offload(#[from] oscar_offload::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("pool exhausted: no evictable slot among {hot} - every one is pinned by running work or holding an unlanded read")]
    Exhausted { hot: usize },
    #[error("unknown key: {0}")]
    UnknownKey(String),
    #[error("slot lifecycle: {0}")]
    Lifecycle(String),
    #[error("unsupported expert dtype {dtype} for blk.{layer}.{projection:?}")]
    UnsupportedDtype { layer: usize, projection: ProjectionKind, dtype: String },
    #[error("blk.{layer}: gate is {gate:?} and up {up:?}; the fused kernel takes one format")]
    GateUpDtypes { layer: usize, gate: GgmlDType, up: GgmlDType },
    #[error("blk.{layer}: gate / up shape disagree: gate=({gn}, {gk}) up=({un}, {uk})")]
    GateUpShapeMismatch { layer: usize, gn: usize, gk: usize, un: usize, uk: usize },
    /// The check is `down.n == gate.k`, never the hidden width: a latent MoE runs its experts in a narrower latent.
    #[error("blk.{layer}: down is ({dn}, {dk}), expected ({gk}, {gn}) to close the expert block")]
    DownShapeMismatch { layer: usize, dn: usize, dk: usize, gn: usize, gk: usize },
}

pub type Result<T> = std::result::Result<T, Error>;

fn layout_error(why: String) -> Error {
    oscar_offload::Error::Layout(why).into()
}

/// A thread fed through channel sender `S`, joined on drop so it cannot outlive the pointers it holds.
pub(crate) struct Worker<S> {
    tx: Option<S>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl<S> Worker<S> {
    pub(crate) fn spawn<R: Send + 'static>(
        name: &str, (tx, rx): (S, R), main: impl FnOnce(R) + Send + 'static,
    ) -> std::io::Result<Self> {
        let join = std::thread::Builder::new().name(name.into()).spawn(move || main(rx))?;
        Ok(Self { tx: Some(tx), join: Some(join) })
    }

    pub(crate) fn tx(&self) -> &S {
        self.tx.as_ref().expect("the channel lives until the drop")
    }
}

impl<S> Drop for Worker<S> {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

/// A class's format as its bank holds it.
fn bank_dtype(d: GgmlDType) -> Option<BankDtype> {
    Some(match d {
        GgmlDType::Q3K => BankDtype::Q3K,
        GgmlDType::Q4K => BankDtype::Q4K,
        GgmlDType::Q6K => BankDtype::Q6K,
        // The K3 stacks' formats: a bank needs only the format's tag and `bytes_per_expert`.
        GgmlDType::Iq1S => BankDtype::Iq1S,
        GgmlDType::Iq2XXS => BankDtype::Iq2XXS,
        GgmlDType::Iq3XXS => BankDtype::Iq3XXS,
        GgmlDType::Q5_0 => BankDtype::Q5_0,
        _ => return None,
    })
}

/// At load: gate and up of one format and shape, down closing the block; the fused kernel is the census's question.
fn check_expert_block(layer: usize, [gate, up, down]: [&ProjectionLayout; 3]) -> Result<()> {
    if gate.dtype != up.dtype {
        return Err(Error::GateUpDtypes { layer, gate: gate.dtype, up: up.dtype });
    }
    if gate.n != up.n || gate.k != up.k {
        return Err(Error::GateUpShapeMismatch { layer, gn: gate.n, gk: gate.k, un: up.n, uk: up.k });
    }
    if down.n != gate.k || down.k != gate.n {
        return Err(Error::DownShapeMismatch { layer, dn: down.n, dk: down.k, gn: gate.n, gk: gate.k });
    }
    Ok(())
}

/// The last `count` of `layers` carry the offload machinery, as late layers route concentrated sets.
pub fn carrier_of(count: usize, layers: usize) -> Vec<bool> {
    (0..layers).map(|l| l >= layers - count).collect()
}

pub type Class = (ProjectionKind, BankDtype);

/// A segment a working region lends: its bank's class, index and byte range in the lender.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Loan {
    pub class: Class,
    pub seg: usize,
    pub offset: usize,
    pub len: usize,
}

/// What a move did, for the crossing's line.
#[derive(Debug, Default, Clone, Copy)]
pub struct Moves {
    pub unsealed: usize,
    pub sealing: usize,
    pub retired: usize,
    pub admitted: usize,
    pub copied: usize,
    pub read: usize,
}

/// What a move leaves the mover, admitted off the engine thread, where page faults stall nothing.
struct Settle {
    seal: Vec<usize>,
    targets: BTreeMap<Class, usize>,
}

/// The launch's choices, made once.
pub struct Launch {
    /// The ceiling plan's slots per layer, which sizes the pacer, oracle and rings; the live shape is the banks'.
    pub hot_budget: usize,
    pub pacer: SchedulePacer,
    pub oracle: Arc<dyn Oracle>,
}

/// The model's experts in their banks, and every thread that moves them but the engine's.
pub struct ExpertPool {
    /// Ordered as the classes sort: gate, up, then down by format.
    banks: BTreeMap<Class, Arc<Bank>>,
    /// Each layer's formats, indexed by `ProjectionKind::index`.
    dtype_for_layer: Vec<[BankDtype; 3]>,
    /// Per layer, set only once the seal's bytes are in: the routed chain reads it as sealed.
    sealed: Vec<AtomicBool>,
    sealed_rows: ExpertRows,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    num_experts: usize,
    /// `None` where the header omits `expert_used_count`: the decode pacer refuses to run without it.
    topk: Option<usize>,
    pub queue: Arc<FetchQueue>,
    readers: Readers,
    /// Advanced by the engine thread alone, at each call and wave.
    pub tick: AtomicU32,
    launch: OnceLock<Launch>,
    /// As [`Self::plan_loans`] laid them, the first at the lender's top; one stands while its segment is borrowed.
    loans: Mutex<Vec<Loan>>,
    settling: Mutex<Option<Settle>>,
    /// What the pool borrows from through decode, fixed at the first step.
    lending: OnceLock<Option<lending::Lending>>,
    pub series: counters::TokenSeries,
}

impl ExpertPool {
    /// Every segment absent and every row null: a forward encodes over it, so the census runs before the plan sizes it.
    pub fn build(layout: &GgufRoutedLayout, gguf_paths: &[PathBuf], segment_slots: usize) -> Result<Self> {
        let device = objc2_metal::MTLCreateSystemDefaultDevice()
            .ok_or_else(|| oscar_offload::Error::Metal("MTLCreateSystemDefaultDevice returned None".into()))?;
        // An unresolvable `--mirror` is refused, never dropped: one drive would silently answer a different question.
        let mut file = TieredFile::open_many(gguf_paths)?;
        let mut readers = vec![crate::config::params().misc.io_threads];
        if let Some(mirror) = crate::config::drives::installed() {
            file.add_replica(&mirror.shard_paths(gguf_paths).map_err(|e| layout_error(format!("--mirror: {e:#}")))?)?;
            readers.push(mirror.drive.readers);
        }
        let (file, block_count, num_experts) = (Arc::new(file), layout.block_count, layout.num_experts);
        // Per class: its keys' extents, its layers, its experts' size.
        let mut classes: BTreeMap<Class, (HashMap<LayerExpertKey, ExpertExtent>, usize, usize)> = BTreeMap::new();
        let mut dtype_for_layer: Vec<[BankDtype; 3]> = Vec::with_capacity(block_count);
        for layer in 0..block_count {
            // The extractor holds every (layer, projection) or refuses the file.
            let projections = ProjectionKind::ALL.map(|kind| &layout.projections[&(layer, kind)]);
            check_expert_block(layer, projections)?;
            let mut dtypes = [BankDtype::Q4K; 3];
            for (kind, pl) in ProjectionKind::ALL.into_iter().zip(projections) {
                let dt = bank_dtype(pl.dtype).ok_or_else(|| Error::UnsupportedDtype {
                    layer,
                    projection: kind,
                    dtype: format!("{:?}", pl.dtype),
                })?;
                dtypes[kind.index()] = dt;
                let class = classes.entry((kind, dt)).or_insert_with(|| (HashMap::new(), 0, pl.bytes_per_expert));
                if class.2 != pl.bytes_per_expert {
                    return Err(layout_error(format!(
                        "layer {layer} {}-{dt:?}: bytes_per_expert {} != previously seen {}",
                        kind.label(),
                        pl.bytes_per_expert,
                        class.2
                    )));
                }
                class.1 += 1;
                // Dense `0..num_experts`: the extractor validated it.
                class.0.extend((0..num_experts).map(|e| ((layer as u16, e as u16), pl.experts[e])));
            }
            dtype_for_layer.push(dtypes);
        }
        // A bank of a class's worth of segments and as much again, all absent.
        let mut banks = BTreeMap::new();
        for ((kind, dt), (keys, layers, bytes_per_expert)) in classes {
            let blocks = keys.keys().map(|(l, _)| *l).max().map_or(0, |m| m + 1);
            let (segments, per_segment) = Bank::index_space(layers, num_experts, segment_slots);
            let bank = Bank::new(
                &device,
                file.clone(),
                segments,
                per_segment,
                bytes_per_expert,
                keys,
                dt,
                blocks,
                num_experts as u16,
            )?;
            banks.insert((kind, dt), Arc::new(bank));
        }
        let sealed_rows = ExpertRows::new(&device, num_experts, block_count).map_err(oscar_offload::Error::Metal)?;
        let queue = Arc::new(FetchQueue::default());
        let readers = Readers::new(queue.clone(), &readers);
        // The duty sampler: the only honest reading of whether the drive is kept fed.
        if crate::config::params().misc.diag {
            oscar_offload::io::start_duty_sampler();
        }
        Ok(Self {
            banks,
            dtype_for_layer,
            sealed: (0..block_count).map(|_| AtomicBool::new(false)).collect(),
            sealed_rows,
            device,
            num_experts,
            topk: layout.experts_per_token,
            queue,
            readers,
            tick: AtomicU32::new(0),
            launch: OnceLock::new(),
            loans: Mutex::new(Vec::new()),
            settling: Mutex::new(None),
            lending: OnceLock::new(),
            series: Default::default(),
        })
    }

    pub fn reader_count(&self) -> usize {
        self.readers.count()
    }

    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    pub fn launched(&self) -> Option<&Launch> {
        self.launch.get()
    }

    /// 0 before the launch, when only the census forwards.
    pub fn hot_budget(&self) -> usize {
        self.launch.get().map_or(0, |l| l.hot_budget)
    }

    pub fn pacer(&self) -> Option<&SchedulePacer> {
        self.launch.get().map(|l| &l.pacer)
    }

    pub fn oracle(&self) -> Option<&Arc<dyn Oracle>> {
        self.launch.get().map(|l| &l.oracle)
    }

    /// The runtime's tick schedule at `hot_budget`.
    pub fn decode_pacer(&self, hot_budget: usize) -> DecodePacer {
        let topk = self
            .topk
            .unwrap_or_else(|| panic!("GGUF metadata has no expert_used_count - decode pacer needs router top-k"));
        DecodePacer::new(self.layer_count(), decode_layers(hot_budget as u32, topk as u32))
    }

    /// As `OSCAR_ORACLE` names it: unset, `transition`, or under a recorded plane its clairvoyant `belady`.
    pub fn select_oracle(&self, hot_budget: usize, belady: Option<Arc<dyn Oracle>>, offloads: bool) -> Arc<dyn Oracle> {
        let kind = crate::config::params().oracle.as_str();
        if kind.is_empty() && belady.is_some() {
            return belady.expect("just checked");
        }
        if !kind.is_empty() && !["transition", "recency", "lru"].iter().any(|k| kind.eq_ignore_ascii_case(k)) {
            return belady.unwrap_or_else(|| {
                panic!(
                    "OSCAR_ORACLE={kind}: unknown oracle; production transition|recency|lru (unset = transition), \
                     belady under replay only"
                )
            });
        }
        let channels = crate::config::params().eviction;
        let transition =
            NextUseOracle::new(self.decode_pacer(hot_budget), self.layer_count(), self.num_experts, channels);
        if offloads {
            tracing::info!(target: "oscar::oracle",
                "transition oracle - {} experts/layer predicted per step",
                policy::HORIZON_K,
            );
        }
        if kind.eq_ignore_ascii_case("lru") {
            crate::detail!(target: "oscar::oracle", "eviction by recency (LRU)");
            Arc::new(transition.recency_alone())
        } else if kind.eq_ignore_ascii_case("recency") {
            crate::detail!(target: "oscar::oracle", "recency prior armed");
            Arc::new(transition.with_recency_prior())
        } else {
            Arc::new(transition)
        }
    }

    /// Once: the plan's shape settled on the loading thread, so sealed forwards serve the first token.
    pub fn launch(&self, hot_budget: usize, carrier_layers: usize, launch: Launch) -> Result<()> {
        let carrier_of = carrier_of(carrier_layers, self.layer_count());
        if hot_budget == 0 && carrier_layers > 0 {
            return Err(layout_error("ProjectionGroup: carrier layers need hot_budget > 0".into()));
        }
        for bank in self.banks.values() {
            bank.attach_oracle(launch.oracle.clone());
        }
        self.move_to(hot_budget, &carrier_of, 0);
        let settle = self.settling.lock().expect("settling poisoned").take().expect("a move owes its settle");
        self.settle(&settle, || true)?;
        let label = match &launch.pacer {
            SchedulePacer::Procedural(p) => format!("procedural pacer ({} ticks/token)", p.ticks_per_token()),
            SchedulePacer::Recorded(o) => format!("recorded schedule ({} ticks)", o.tick_count()),
        };
        crate::detail!(target: "oscar::experts",
            "pool launched - {:.2} GB, {label}",
            self.pool_bytes() as f64 / crate::machine::GB
        );
        assert!(self.launch.set(launch).is_ok(), "the expert pool is launched once");
        let min_cap = self.min_bank_allocatable(false);
        let wave_k = policy::prefill_batch(self.num_experts as u32, min_cap);
        crate::detail!(target: "oscar::experts", "prefill partition tokens={} wave_k={wave_k} min_bank={min_cap}", policy::PREFILL_BATCH_TOKENS);
        Ok(())
    }

    fn layer_keys(&self, layer: usize) -> Vec<LayerExpertKey> {
        (0..self.num_experts).map(|e| (layer as u16, e as u16)).collect()
    }

    /// Per class, its `(sealed, carrier)` layer counts.
    fn layer_counts(&self, carrier_of: &[bool]) -> BTreeMap<Class, (usize, usize)> {
        let mut counts: BTreeMap<Class, (usize, usize)> = self.banks.keys().map(|&class| (class, (0, 0))).collect();
        for (layer, &carrier) in carrier_of.iter().enumerate() {
            for kind in ProjectionKind::ALL {
                let (sealed, carriers) = counts.get_mut(&self.class_of(layer, kind)).expect("a bank for every class");
                *if carrier { carriers } else { sealed } += 1;
            }
        }
        counts
    }

    /// Each bank's sealed stacks and `hot_budget` slots per carrier, in whole segments, as the plan counts them.
    fn targets(&self, hot_budget: usize, carrier_of: &[bool]) -> BTreeMap<Class, usize> {
        let slots = |(sealed, carriers): (usize, usize)| sealed * self.num_experts + carriers * hot_budget;
        let counts = self.layer_counts(carrier_of).into_iter();
        counts.map(|(class, n)| (class, slots(n).div_ceil(self.banks[&class].units_per_segment()))).collect()
    }

    fn class_of(&self, layer: usize, kind: ProjectionKind) -> Class {
        (kind, self.dtype_for_layer[layer][kind.index()])
    }

    /// At a parked boundary: new carriers unsealed (fenced at `gen`), surplus segments retired; [`Self::settle_moves`]
    /// admits and seals.
    pub fn move_to(&self, hot_budget: usize, carrier_of: &[bool], gen: u64) -> Moves {
        let tick = self.tick.load(Ordering::Relaxed);
        let mut moves = Moves::default();
        // The unsealed layers' segments, which the retire takes first.
        let mut unsealed: BTreeMap<Class, BTreeSet<usize>> = BTreeMap::new();
        for layer in (0..carrier_of.len()).filter(|&l| carrier_of[l] && self.layer_sealed(l)) {
            let keys = self.layer_keys(layer);
            for (kind, bank) in ProjectionKind::ALL.into_iter().zip(self.banks_for(layer)) {
                bank.unseal(&keys, gen);
                write_row(layer, self.num_experts, &self.sealed_rows.rows[kind.index()], std::iter::empty());
                let per = bank.units_per_segment();
                let segments = keys.iter().filter_map(|&key| bank.resident_slot(key)).map(|slot| slot / per);
                unsealed.entry(self.class_of(layer, kind)).or_default().extend(segments);
            }
            self.sealed[layer].store(false, Ordering::Release);
            moves.unsealed += 1;
        }
        let seal: Vec<usize> = (0..carrier_of.len()).filter(|&l| !carrier_of[l] && !self.layer_sealed(l)).collect();
        moves.sealing = seal.len();
        let targets = self.targets(hot_budget, carrier_of);
        self.retire_to(&targets, &unsealed, tick, &mut moves);
        *self.settling.lock().expect("settling poisoned") = Some(Settle { seal, targets });
        moves
    }

    /// The unsealed layers' segments first (a whole layer's experts, more than a carrier keeps), then the highest.
    fn retire_to(
        &self, targets: &BTreeMap<Class, usize>, unsealed: &BTreeMap<Class, BTreeSet<usize>>, tick: u32,
        moves: &mut Moves,
    ) {
        for (class, bank) in &self.banks {
            let segs = bank.segments();
            let own = |i: &usize| segs[*i].standing() && !segs[*i].borrowed;
            let mut free: Vec<usize> = (0..segs.len()).filter(|i| own(i) && segs[*i].sealed == 0).collect();
            let first = unsealed.get(class);
            free.sort_by_key(|&seg| (!first.is_some_and(|f| f.contains(&seg)), std::cmp::Reverse(seg)));
            let surplus = (0..segs.len()).filter(own).count().saturating_sub(targets[class]);
            for &seg in free.iter().take(surplus) {
                bank.retire(seg, tick);
                moves.retired += 1;
            }
        }
    }

    /// Standing segments a bank holds, borrowed ones excluded: the pool's own shape.
    fn own_standing(segs: &[Seg]) -> usize {
        segs.iter().filter(|s| s.standing() && !s.borrowed).count()
    }

    /// After a refused admit: the carriers and the most slots a carrier every bank holds, those above retired to it.
    fn reached(&self, tick: u32) -> (Vec<bool>, usize) {
        let carrier_of: Vec<bool> = (0..self.layer_count()).map(|l| !self.layer_sealed(l)).collect();
        let counts = self.layer_counts(&carrier_of);
        let mut hot_budget = usize::MAX;
        for (class, bank) in self.banks.iter().filter(|(class, _)| counts[*class].1 > 0) {
            let (sealed, carriers) = counts[class];
            let own = Self::own_standing(&bank.segments()) * bank.units_per_segment();
            hot_budget = hot_budget.min(own.saturating_sub(sealed * self.num_experts) / carriers);
        }
        let hot_budget = if hot_budget == usize::MAX { self.num_experts } else { hot_budget };
        self.retire_to(&self.targets(hot_budget, &carrier_of), &BTreeMap::new(), tick, &mut Moves::default());
        (carrier_of, hot_budget)
    }

    /// The mover's half: banks grown, then a layer at a time sealed; a refused admit returns.
    fn settle(&self, settle: &Settle, drafter_idle: impl Fn() -> bool) -> Result<Moves> {
        let mut moves = Moves::default();
        for (class, bank) in &self.banks {
            loop {
                let segs = bank.segments();
                if Self::own_standing(&segs) >= settle.targets[class] {
                    break;
                }
                // A loan's index is the loan's, lent or handed back.
                let loans: Vec<usize> = self.loans().iter().filter(|l| l.class == *class).map(|l| l.seg).collect();
                let absent = |i: &usize| !segs[*i].present && !segs[*i].borrowed && !loans.contains(i);
                let seg = (0..segs.len()).find(absent).ok_or_else(|| {
                    layout_error(format!(
                        "no absent segment in the {:?}-{:?} bank: a previous move's retired segments are still freeing",
                        class.0, class.1
                    ))
                })?;
                bank.admit(&self.device, seg)?;
                moves.admitted += 1;
            }
        }
        for &layer in &settle.seal {
            let tick = self.tick.load(Ordering::Relaxed);
            let (copied, read) = self.seal_layer(layer, tick)?;
            (moves.copied, moves.read) = (moves.copied + copied, moves.read + read);
            self.housekeep(tick, drafter_idle());
        }
        Ok(moves)
    }

    /// `(copied, read)`: `E` low slots a bank, each expert copied or read; a refusal leaves a carrier.
    fn seal_layer(&self, layer: usize, tick: u32) -> Result<(usize, usize)> {
        let (e, keys) = (self.num_experts, self.layer_keys(layer));
        let (mut copied, mut read) = (0, 0);
        let banks = self.banks_for(layer);
        for (i, bank) in banks.iter().enumerate() {
            match bank.seal(&keys, tick) {
                Ok((c, r)) => (copied, read) = (copied + c, read + r),
                Err(err) => {
                    for (done, rows) in banks.iter().zip(&self.sealed_rows.rows).take(i) {
                        done.unseal(&keys, 0);
                        write_row(layer, e, rows, std::iter::empty());
                    }
                    return Err(err);
                }
            }
            let address =
                |key: &LayerExpertKey| bank.pool().address(bank.resident_slot(*key).expect("a sealed key is resident"));
            write_row(layer, e, &self.sealed_rows.rows[i], keys.iter().map(|key| (key.1, address(key))));
        }
        self.sealed[layer].store(true, Ordering::Release);
        Ok((copied, read))
    }

    /// Free retired segments nothing reads; freed, left, and what holds one.
    fn housekeep(&self, tick: u32, drafter_idle: bool) -> (usize, usize, Option<String>) {
        let (mut freed, mut left, mut held) = (0, 0, None);
        for (class, bank) in &self.banks {
            let segs = bank.segments();
            for seg in (0..segs.len()).filter(|&i| segs[i].present && segs[i].retiring && !segs[i].borrowed) {
                match bank.free(seg, tick, drafter_idle) {
                    Ok(()) => freed += 1,
                    Err(why) => {
                        left += 1;
                        held.get_or_insert_with(|| {
                            format!("segment {seg} of the {:?}-{:?} bank: {why}", class.0, class.1)
                        });
                    }
                }
            }
        }
        (freed, left, held)
    }

    /// The mover's half: the admits and seals, the frees as they clear, then `owed`; true once the move is over.
    /// `reshaped` follows every change of the pool's shape.
    pub fn settle_moves(
        &self, held_since: &mut Option<std::time::Instant>, drafter_busy: impl Fn() -> bool,
        re_land: impl FnOnce((usize, usize), &str, &str), owed: impl FnOnce(), reshaped: impl Fn(),
    ) -> bool {
        let drafter_idle = !drafter_busy();
        let settle = self.settling.lock().expect("settling poisoned").take();
        if let Some(settle) = settle {
            let t = std::time::Instant::now();
            match self.settle(&settle, || !drafter_busy()) {
                Ok(m) => {
                    reshaped();
                    tracing::info!(target: "oscar::experts",
                        "settled in {} - {} layers sealed ({} experts copied, {} read), {} segments admitted, {} retired{}",
                        mistralrs_core::live::duration_text(t.elapsed().as_secs_f64()),
                        settle.seal.len(),
                        m.copied,
                        m.read,
                        m.admitted,
                        m.retired,
                        crate::cli::console::stamp()
                    );
                }
                Err(e) => {
                    let what = match e {
                        Error::Offload(oscar_offload::Error::Io(_) | oscar_offload::Error::ShortRead { .. }) => {
                            "an expert read failed"
                        }
                        _ => "a mapping was refused",
                    };
                    let (carrier_of, hot_budget) = self.reached(self.tick.load(Ordering::Relaxed));
                    reshaped();
                    re_land((carrier_of.iter().filter(|&&c| c).count(), hot_budget), what, &e.to_string());
                }
            }
        }
        let (freed, left, held) = self.housekeep(self.tick.load(Ordering::Relaxed), drafter_idle);
        if freed > 0 {
            tracing::info!(target: "oscar::experts", "{freed} retired segments released, {left} to go{}", crate::cli::console::stamp());
        }
        // Unlanded last frees hold every later crossing, so what holds them is said every ten seconds.
        if left == 0 || freed > 0 {
            *held_since = None;
        } else if held_since.get_or_insert_with(std::time::Instant::now).elapsed() >= std::time::Duration::from_secs(10)
        {
            *held_since = Some(std::time::Instant::now());
            tracing::warn!(target: "oscar::experts",
                "Memory was slow to move: retired segments held 10 s past their move, {left} - {}; drafter {}{}",
                held.unwrap_or_default(),
                if drafter_idle { "idle" } else { "in a session" },
                crate::cli::console::stamp()
            );
        }
        owed();
        left == 0 && self.settling.lock().expect("settling poisoned").is_none()
    }

    /// Laid, not lent: the segments taking each bank from `own` slots a carrier to `total`, from `mapped` to `floor`.
    /// A level across the banks at a time, so a hand-back takes from every bank alike.
    pub fn plan_loans(&self, own: usize, total: usize, carrier_of: &[bool], mapped: usize, floor: usize) -> usize {
        let mut loans = self.loans.lock().expect("loans poisoned");
        let borrowed = |l: &Loan| self.banks[&l.class].segments()[l.seg].borrowed;
        assert!(!loans.iter().any(borrowed), "the loans are laid while nothing is lent");
        loans.clear();
        let mut offset = mapped;
        let mut taken: BTreeMap<Class, usize> = BTreeMap::new();
        let mut below = self.targets(own, carrier_of);
        for level in own + 1..=total.min(self.num_experts) {
            let at = self.targets(level, carrier_of);
            let mut laid = Vec::new();
            for (class, bank) in &self.banks {
                let len = bank.units_per_segment() * padded_unit(bank.bytes_per_expert());
                for _ in below[class]..at[class] {
                    let n = taken.entry(*class).or_default();
                    let seg = (bank.slot_count() / bank.units_per_segment()).checked_sub(1 + *n);
                    let (Some(seg), Some(under)) = (seg, offset.checked_sub(len).filter(|&o| o >= floor)) else {
                        return loans.len();
                    };
                    (*n, offset) = (*n + 1, under);
                    laid.push(Loan { class: *class, seg, offset, len });
                }
            }
            loans.append(&mut laid);
            below = at;
        }
        loans.len()
    }

    pub fn loans(&self) -> Vec<Loan> {
        self.loans.lock().expect("loans poisoned").clone()
    }

    /// The loans adopted now and the experts resident in them.
    pub fn loans_standing(&self) -> (usize, usize) {
        let loans = self.loans.lock().expect("loans poisoned");
        let standing = loans.iter().filter(|l| self.banks[&l.class].segments()[l.seg].borrowed);
        standing.fold((0, 0), |(n, r), l| (n + 1, r + self.banks[&l.class].residents_in(l.seg)))
    }

    /// The segments the banks hold now, borrowed ones included, none retiring.
    pub fn held_segments(&self) -> usize {
        self.banks.values().map(|b| b.segments().iter().filter(|s| s.standing()).count()).sum()
    }

    /// The bound down to `bound` and the loans above it adopted, no OS call; a busy region moves nothing.
    pub fn lend(&self, region: &oscar_offload::region::Region, bound: usize) -> Result<usize> {
        region.move_bound(bound)?;
        let (lender, bound) = (region.lender(), region.bound());
        let mut adopted = 0;
        // Lowest first: a segment's slots enter the free list on top.
        for loan in self.loans.lock().expect("loans poisoned").iter().rev().filter(|l| l.offset >= bound) {
            let bank = &self.banks[&loan.class];
            if !bank.segments()[loan.seg].borrowed {
                bank.adopt(loan.seg, &lender, loan.offset)?;
                adopted += 1;
            }
        }
        Ok(adopted)
    }

    /// The drafter quiesced, the queue drained to `committed` and the region idle: the loans under `bound` cleared
    /// before it moves up, those above kept.
    pub fn take_back(
        &self, region: &oscar_offload::region::Region, bound: usize, tick: u32, committed: u64,
    ) -> Result<usize> {
        let bound = padded_unit(bound);
        if !region.is_idle() {
            return Err(layout_error(format!(
                "the working region is not idle, so no loan is taken back -{}",
                region.describe()
            )));
        }
        // A stamp of the next generation outlives the step that never committed it; nothing waits for it here.
        let orphans: usize = self.banks.values().map(|b| b.clear_orphan_fences(committed)).sum();
        if orphans > 0 && crate::cli::console::traced() {
            tracing::info!(target: "oscar::experts",
                "{orphans} slot fence(s) above committed count {committed} cleared at take-back{}",
                crate::cli::console::stamp()
            );
        }
        let loans = self.loans.lock().expect("loans poisoned");
        let mut back: Vec<&Loan> =
            loans.iter().filter(|l| l.offset < bound && self.banks[&l.class].segments()[l.seg].borrowed).collect();
        for loan in &back {
            let bank = &self.banks[&loan.class];
            if !bank.segments()[loan.seg].retiring {
                bank.retire(loan.seg, tick);
            }
        }
        let handed_back = back.len();
        let cap = self.banks.values().map(|b| b.stall_cap()).max().unwrap_or_default();
        let deadline = std::time::Instant::now() + cap;
        loop {
            let mut held = Vec::new();
            back.retain(|loan| {
                let why = self.banks[&loan.class].free(loan.seg, tick, true).err();
                held.extend(why.as_ref().map(|why| format!("{:?} segment {}: {why}", loan.class, loan.seg)));
                why.is_some()
            });
            if back.is_empty() {
                break;
            }
            if std::time::Instant::now() >= deadline {
                return Err(layout_error(format!(
                    "{} loan(s) under the bound did not clear in {cap:?} - {}",
                    held.len(),
                    held.join("; ")
                )));
            }
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
        drop(loans);
        region.move_bound(bound)?;
        Ok(handed_back)
    }

    /// A carrier layer's entries are null.
    pub(crate) fn sealed_rows(&self) -> &ExpertRows {
        &self.sealed_rows
    }

    pub fn layer_sealed(&self, layer: usize) -> bool {
        self.sealed[layer].load(Ordering::Acquire)
    }

    pub fn carrier_count(&self) -> usize {
        self.sealed.iter().filter(|s| !s.load(Ordering::Acquire)).count()
    }

    pub fn layer_count(&self) -> usize {
        self.dtype_for_layer.len()
    }

    /// The carriers' share of the expert layers, which scales a figure the banks count to every expert access.
    pub fn paged_share(&self) -> f64 {
        self.carrier_count() as f64 / self.layer_count().max(1) as f64
    }

    /// A layer's gate, up and down banks.
    pub fn banks_for(&self, layer: usize) -> [&Arc<Bank>; 3] {
        ProjectionKind::ALL.map(|kind| &self.banks[&self.class_of(layer, kind)])
    }

    pub fn banks(&self) -> impl Iterator<Item = (&Class, &Arc<Bank>)> {
        self.banks.iter()
    }

    /// For eval instruments that sample one bank.
    pub fn first_bank(&self) -> &Arc<Bank> {
        self.banks.values().next().expect("a built pool has at least one bank")
    }

    /// Every bank's residency set onto a queue reading the pool; idempotent.
    pub fn attach_queue(&self, queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>) {
        self.banks.values().for_each(|b| b.pool().attach(queue));
    }

    /// The smallest bank's claimable slots: a wave's ceiling without loans, which no phase's may move, the run-ahead's
    /// with them. A bank with none is left out, else "nothing fits" would disable a mandatory rotation.
    pub fn min_bank_allocatable(&self, with_loans: bool) -> u32 {
        let min = self.banks.values().map(|b| b.claimable(with_loans)).filter(|&c| c > 0).min().unwrap_or(0);
        min.min(u32::MAX as usize) as u32
    }

    /// Present slots' bytes, as the machine holds them.
    pub fn pool_bytes(&self) -> u64 {
        self.banks.values().map(|b| b.pool().bytes() as u64).sum()
    }

    /// The pool's expert memory now, slots lent from the working region included.
    pub fn expert_bytes(&self) -> u64 {
        self.banks.values().map(|b| (b.pool().bytes() + b.pool().borrowed_bytes()) as u64).sum()
    }

    /// Every bank's [`Bank::occupancy`] summed, and the fewest present in any one bank.
    pub fn slot_occupancy(&self) -> ([usize; 4], usize) {
        let (mut sum, mut fewest) = ([0; 4], usize::MAX);
        for occupancy in self.banks.values().map(|b| b.occupancy()) {
            fewest = fewest.min(occupancy[0]);
            sum.iter_mut().zip(occupancy).for_each(|(t, n)| *t += n);
        }
        (sum, fewest)
    }

    /// The completion event every bank's fences wait on.
    pub fn set_completion_event(&self, event: &Retained<ProtocolObject<dyn objc2_metal::MTLSharedEvent>>) {
        self.banks.values().for_each(|b| b.set_completion_event(event));
    }

    /// The draft-read fence's retire side, once the drafter's stream is synced: every bank's window closes.
    pub fn end_draft_forward(&self) {
        self.banks.values().for_each(|b| b.end_draft_forward());
    }

    /// The decode window a finished request opened, handed out once; `None` before its first decode token.
    pub fn decode_window(&self) -> Option<DecodeWindow> {
        self.series.window_now()
    }
}

#[cfg(test)]
mod tests;
