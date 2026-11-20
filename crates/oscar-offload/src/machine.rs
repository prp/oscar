//! The machine: the one reader of the kernel's counters, the snapshot and the four bounds, and the plan's line on what
//! this process maps and the floor under what is left, both refusing before a take on different quantities.

use std::ffi::CStr;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use objc2_metal::MTLDevice as _;

/// The kernel's page: mappings and units are whole pages.
pub const PAGE: usize = 16384;

/// As hashbrown lays a map out: buckets the power of two above 8/7 of `entries`, four at least, a control byte each.
pub fn hash_map_bytes<K, V>(entries: usize) -> usize {
    let buckets = if entries < 4 { 4 } else { (entries * 8).div_ceil(7).next_power_of_two() };
    buckets * (std::mem::size_of::<(K, V)>() + 1) + 16
}

/// The unit every figure is said in.
pub const GB: f64 = (1u64 << 30) as f64;

/// The headroom this process will not take the machine below, in bytes; 0 disarms it.
static FLOOR: AtomicU64 = AtomicU64::new(0);

/// The run's high-water is the largest of three unrelated demands, so each peak is read against its phase's charge.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    /// Everything before the engine's first step.
    Load = 0,
    Prefill = 1,
    Decode = 2,
}

impl Phase {
    /// A prompt step is a prefill, a completion step a decode whatever its width.
    pub(crate) fn of_step(prompt: bool) -> Self {
        if prompt {
            Self::Prefill
        } else {
            Self::Decode
        }
    }
}

pub const PHASES: usize = 3;

/// One writer, [`begin_step`], before the step's inputs are made.
static PHASE: AtomicU64 = AtomicU64::new(Phase::Load as u64);

/// Every claim from here is the step's phase's; the census's forwards run outside any step.
pub fn begin_step(prompt: bool) {
    let p = Phase::of_step(prompt);
    if PHASE.swap(p as u64, Ordering::Relaxed) != p as u64 {
        PLAN.open(p as usize);
    }
}

pub fn phase() -> usize {
    (PHASE.load(Ordering::Relaxed) as usize).min(PHASES - 1)
}

/// The line the plan drew, what is held against it and each phase's peak: the plan's accuracy read straight off.
pub struct Line {
    held: AtomicU64,
    peak: [AtomicU64; PHASES],
    limit: AtomicU64,
}

impl Line {
    const fn new() -> Self {
        Self {
            held: AtomicU64::new(0),
            peak: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
            limit: AtomicU64::new(0),
        }
    }

    /// 0 disarms the line.
    pub fn arm(&self, bytes: u64) {
        self.limit.store(bytes, Ordering::Relaxed);
    }

    pub fn limit(&self) -> u64 {
        self.limit.load(Ordering::Relaxed)
    }

    pub fn held(&self) -> u64 {
        self.held.load(Ordering::Relaxed)
    }

    /// The run's high-water: the phases cannot overlap.
    pub fn peak(&self) -> u64 {
        self.phase_peaks().into_iter().max().unwrap_or(0)
    }

    pub(crate) fn phase_peaks(&self) -> [u64; PHASES] {
        std::array::from_fn(|i| self.peak[i].load(Ordering::Relaxed))
    }

    /// A phase's peak opens at what is held: a phase that claims nothing still holds what it inherited.
    pub fn open(&self, phase: usize) {
        self.peak[phase.min(PHASES - 1)].fetch_max(self.held(), Ordering::Relaxed);
    }

    /// `None` is room, taken; `Some((live, limit))` is over, nothing taken: test and take are one atomic step.
    pub fn claim(&self, bytes: u64) -> Option<(u64, u64)> {
        let limit = self.limit();
        // A census's claims mark no peak.
        let mark = |held: u64| {
            if !crate::census::armed() {
                self.peak[phase()].fetch_max(held, Ordering::Relaxed);
            }
        };
        if limit == 0 {
            mark(self.held.fetch_add(bytes, Ordering::Relaxed) + bytes);
            return None;
        }
        match self.held.try_update(Ordering::AcqRel, Ordering::Acquire, |held| {
            over(held, bytes, limit).is_none().then(|| held.saturating_add(bytes))
        }) {
            Ok(was) => {
                mark(was.saturating_add(bytes));
                None
            }
            Err(held) => Some((held, limit)),
        }
    }

    /// Give a claim back, or roll back a take that could not map.
    pub fn release(&self, bytes: u64) {
        self.held.fetch_sub(bytes.min(self.held()), Ordering::Relaxed);
    }
}

/// Every byte this process holds mapped, armed at the solved budget: the loosest number still an upper bound.
pub static PLAN: Line = Line::new();

/// Armed at the watchdog's own value, so the refusal and the kill cannot drift apart.
pub fn set_floor(bytes: u64) {
    FLOOR.store(bytes, Ordering::Relaxed);
}

pub fn floor() -> u64 {
    FLOOR.load(Ordering::Relaxed)
}

/// Each bound's refusals, counted, never inferred; the largest shortfall is how far the plan was wrong.
static REFUSED: [AtomicU64; 4] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static SHORTFALL: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refused {
    /// Past the plan's ceiling: whatever sized the claim is wrong.
    Ceiling = 0,
    /// A forward asked its region for a run none holds: the plan's working memory is wrong.
    Region = 1,
    /// Memory arrived from outside this process.
    Floor = 2,
    /// The kernel's wire limits moved under a solved plan.
    Lock = 3,
}

/// A defect under a plan that bounds every allocation: counted, said with its arithmetic, and the request fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Refusal {
    pub bound: Refused,
    pub bytes: u64,
    /// What stood against the claim, by bound: held, in hand or fenced, the machine's headroom, or wired.
    pub held: u64,
    pub limit: u64,
    pub short_by: u64,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gb = |b: u64| b as f64 / (1u64 << 30) as f64;
        let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
        match self.bound {
            Refused::Ceiling => write!(
                f,
                "{:.2} GB segment with {:.2} GB held exceeds plan ceiling {:.2} GB",
                gb(self.bytes),
                gb(self.held),
                gb(self.limit)
            ),
            Refused::Region => write!(
                f,
                "{:.1} MB object, {:.1} MB in hand or fenced, no free run in {:.1} MB working region",
                mib(self.bytes),
                mib(self.held),
                mib(self.limit)
            ),
            Refused::Floor => write!(
                f,
                "{:.2} GB segment leaves free memory {:.2} GB, under {:.2} GB floor",
                gb(self.bytes),
                gb(self.held.saturating_sub(self.bytes)),
                gb(self.limit)
            ),
            Refused::Lock => write!(
                f,
                "the system refused to lock {:.2} GB more of the pool against the wire limit: {:.2} GB wired, the limit {:.2} GB",
                gb(self.bytes),
                gb(self.held),
                gb(self.limit)
            ),
        }
    }
}

/// The one place a bound turns a mapping down: counted and logged, so a caller swallowing the error hides nothing.
pub fn refuse(bound: Refused, bytes: u64, held: u64, limit: u64) -> Refusal {
    let short_by = match bound {
        Refused::Ceiling | Refused::Region => held.saturating_add(bytes).saturating_sub(limit),
        Refused::Floor => limit.saturating_sub(held.saturating_sub(bytes)),
        Refused::Lock => bytes,
    };
    // A census's refusals are its own answer.
    if !crate::census::armed() {
        REFUSED[bound as usize].fetch_add(1, Ordering::Relaxed);
        SHORTFALL.fetch_max(short_by, Ordering::Relaxed);
    }
    let refusal = Refusal { bound, bytes, held, limit, short_by };
    // A census probing a width the region cannot hold ends its record there: the designed answer, not a defect.
    match bound {
        Refused::Region if crate::census::armed() => {
            tracing::info!(target: "oscar::region", "No room in working memory: {refusal}")
        }
        Refused::Region => tracing::error!(target: "oscar::region", "No room in working memory: {refusal}"),
        Refused::Lock => tracing::warn!(target: "oscar::segments", "The system refused to lock memory: {refusal}"),
        Refused::Ceiling | Refused::Floor => {
            tracing::warn!(target: "oscar::planner", "Not enough memory for this request: {refusal}")
        }
    }
    refusal
}

/// Refusals by bound (ceiling, region, floor, lock) and the largest shortfall.
pub fn refusals() -> ([u64; 4], u64) {
    (std::array::from_fn(|i| REFUSED[i].load(Ordering::Relaxed)), SHORTFALL.load(Ordering::Relaxed))
}

/// This process's wired bytes and footprint, by the kernel's accounting: `mlock` is charged against the wired.
pub fn own_rusage() -> (u64, u64) {
    // SAFETY: zero is a valid `rusage_info_v2`, and the v2 flavor writes exactly one.
    let mut info: libc::rusage_info_v2 = unsafe { std::mem::zeroed() };
    let (pid, at) = (std::process::id() as libc::c_int, std::ptr::addr_of_mut!(info).cast());
    let rc = unsafe { libc::proc_pid_rusage(pid, libc::RUSAGE_INFO_V2, at) };
    if rc == 0 {
        (info.ri_wired_size, info.ri_phys_footprint)
    } else {
        (0, 0)
    }
}

/// An integer the kernel may answer narrower (zero-extended), or `xsw_usage`; `None` where it has no such name.
pub fn sysctl<T: Copy>(name: &CStr) -> Option<T> {
    let mut len = std::mem::size_of::<T>();
    // SAFETY: zero is a value of `T` (an integer or `xsw_usage`), and the kernel writes at most `len` bytes.
    unsafe {
        let mut val: T = std::mem::zeroed();
        let rc =
            libc::sysctlbyname(name.as_ptr(), std::ptr::addr_of_mut!(val).cast(), &mut len, std::ptr::null_mut(), 0);
        (rc == 0 && len > 0).then_some(val)
    }
}

/// A count of the kernel's pages, in bytes: its page is the pools' [`PAGE`].
pub fn pages(name: &CStr) -> Option<u64> {
    Some(sysctl::<u64>(name)? * PAGE as u64)
}

/// This instant's free pages: the sysctl answers every call, where `host_statistics64` answers up to a second old.
pub fn live_free() -> Option<u64> {
    pages(c"vm.page_free_count")
}

/// Free bytes on `path`'s volume, read at its nearest existing ancestor.
pub fn volume_free(path: &std::path::Path) -> Option<u64> {
    let probe = path.ancestors().find(|p| p.exists())?;
    let c = std::ffi::CString::new(probe.as_os_str().as_encoded_bytes()).ok()?;
    // SAFETY: zero is a valid `statvfs`, which the call fills from a valid C string.
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    (unsafe { libc::statvfs(c.as_ptr(), &mut st) } == 0).then(|| u64::from(st.f_bavail) * u64::from(st.f_frsize))
}

/// The rule on numbers: `held` plus `bytes` over `ceiling` (0 disarms it).
fn over(held: u64, bytes: u64, ceiling: u64) -> Option<(u64, u64)> {
    (ceiling > 0 && held.saturating_add(bytes) > ceiling).then_some((held, ceiling))
}

/// The kernel's page counts, in bytes.
#[derive(Debug, Clone, Copy, Default)]
pub struct VmPages {
    /// The live `vm.page_free_count`: the views count speculative pages free, and on macOS 27 pages no claimant gets.
    pub free: u64,
    pub inactive: u64,
    pub speculative: u64,
    pub wired: u64,
    pub purgeable: u64,
    pub compressor: u64,
    pub internal: u64,
    /// `total_uncompressed_pages_in_compressor`, a u64 at words 36-37.
    pub compressor_stored: u64,
}

impl VmPages {
    /// Free, inactive and speculative: memory handed out without reclaim, the floor's quantity.
    pub fn reclaimable(&self) -> u64 {
        self.free + self.inactive + self.speculative
    }
}

/// The process asks the kernel at most once a tick.
pub const TICK: std::time::Duration = std::time::Duration::from_millis(50);

/// `Some((headroom, floor))` where mapping `bytes` would cross the floor; a disarmed floor or silent machine is room.
pub(crate) fn short_for(bytes: u64) -> Option<(u64, u64)> {
    short(reclaimable()?, bytes, floor())
}

/// The rule on numbers: `left` less `bytes` under `floor` (0 disarms it).
fn short(left: u64, bytes: u64, floor: u64) -> Option<(u64, u64)> {
    (floor > 0 && left.saturating_sub(bytes) < floor).then_some((left, floor))
}

/// The memorystatus sysctl's encoding: 1 normal, 2 warn, 4 critical; 0 when unreadable.
pub type PressureLevel = u32;

pub fn level_word(level: PressureLevel) -> &'static str {
    match level {
        1 => "normal",
        2 => "warn",
        4 => "critical",
        _ => "-",
    }
}

/// One reading of every memory signal, in bytes.
#[derive(Debug, Clone, Copy)]
pub struct MemorySnapshot {
    pub at: Instant,
    /// What the kernel pages (the installed RAM less its boot reserve), or `--physical-mem`.
    pub physical: u64,
    pub installed: u64,
    pub reclaimable: u64,
    pub wired: u64,
    pub compressor: u64,
    pub compressor_stored: u64,
    /// Excludes wired pages, so the non-droppable sum counts no page twice.
    pub anonymous: u64,
    pub purgeable: u64,
    pub swap_used: u64,
    pub level: PressureLevel,
    pub own_footprint: u64,
    pub own_wired: u64,
    pub machine_wire_limit: u64,
    pub task_wire_limit: u64,
    pub own_gpu: u64,
    /// Every client's: the in-use figure counts only GPU-wired buffers, below oscar's own at idle.
    pub gpu_alloc: u64,
    pub gpu_ceiling: u64,
}

impl Default for MemorySnapshot {
    fn default() -> Self {
        Self {
            at: Instant::now(),
            physical: 0,
            installed: 0,
            reclaimable: 0,
            wired: 0,
            compressor: 0,
            compressor_stored: 0,
            anonymous: 0,
            purgeable: 0,
            swap_used: 0,
            level: 0,
            own_footprint: 0,
            own_wired: 0,
            machine_wire_limit: u64::MAX,
            task_wire_limit: u64::MAX,
            own_gpu: 0,
            gpu_alloc: 0,
            gpu_ceiling: u64::MAX,
        }
    }
}

impl MemorySnapshot {
    /// File pages are out, as a load may evict them.
    pub fn non_droppable(&self) -> u64 {
        (self.wired + self.compressor + self.anonymous).saturating_sub(self.purgeable)
    }

    /// The machine's non-droppable memory less this process's footprint.
    pub fn foreign(&self) -> u64 {
        self.non_droppable().saturating_sub(self.own_footprint)
    }

    pub fn foreign_gpu(&self) -> u64 {
        self.gpu_alloc.saturating_sub(self.own_gpu)
    }

    /// Every wired page but this process's locks, oscar's GPU buffers among them.
    pub fn foreign_wired(&self) -> u64 {
        self.wired.saturating_sub(self.own_wired)
    }

    pub fn bounds(&self) -> Bounds {
        Bounds {
            physical: self.physical,
            installed: self.installed,
            foreign: self.foreign(),
            foreign_gpu: self.foreign_gpu(),
            gpu_ceiling: self.gpu_ceiling,
            foreign_wired: self.foreign_wired(),
            machine_wire_limit: self.machine_wire_limit,
            task_wire_limit: self.task_wire_limit,
        }
    }

    /// Every term as key=value in GB, the tail of every memory line, so a decision replays from its record.
    pub fn ledger(&self) -> String {
        // The floor as mapped; the regions as mapped (fixed at load), in hand, asked for, and freed awaiting the GPU.
        let floor = crate::fixed::fixed_pool_if_built().map_or(0, |p| p.bytes() as u64);
        let r = crate::pools::pools().map_or_else(Default::default, |p| p.levels());
        let terms = [
            ("own", self.own_footprint),
            ("other", self.foreign()),
            ("free", self.reclaimable),
            ("wired", self.wired),
            ("compressed", self.compressor),
            ("stored", self.compressor_stored),
            ("anon", self.anonymous),
            ("purgeable", self.purgeable),
            ("swap", self.swap_used),
            ("own_wired", self.own_wired),
            ("own_gpu", self.own_gpu),
            ("gpu_alloc", self.gpu_alloc),
            ("gpu_ceiling", self.gpu_ceiling),
            ("physical", self.physical),
            ("floor", floor),
            ("region", r.size as u64),
            ("region_live", r.live as u64),
            ("region_asked", r.asked as u64),
            ("region_fenced", r.fenced as u64),
            ("region_lent", r.lent as u64),
        ];
        let mut line = String::new();
        for (key, bytes) in terms {
            let _ = write!(line, "{key}={:.2} ", bytes as f64 / GB);
        }
        line + "level=" + level_word(self.level)
    }

    /// The error names the reading that did not answer.
    pub fn read() -> Result<Self, &'static str> {
        // Both set at boot, so read once.
        static WIRE_LIMITS: OnceLock<Option<(u64, u64)>> = OnceLock::new();
        let vm = vm_pages().ok_or("kernel page counts")?;
        let (own_wired, own_footprint) = own_rusage();
        let device = objc2_metal::MTLCreateSystemDefaultDevice().ok_or("GPU device")?;
        let (own_gpu, gpu_ceiling) = (device.currentAllocatedSize() as u64, device.recommendedMaxWorkingSetSize());
        let (physical, installed) = memory_totals().ok_or("memory totals")?;
        let limits = || Some((sysctl(c"vm.global_user_wire_limit")?, sysctl(c"vm.user_wire_limit")?));
        let (machine_wire_limit, task_wire_limit) = (*WIRE_LIMITS.get_or_init(limits)).ok_or("wire limits")?;
        Ok(Self {
            at: Instant::now(),
            physical,
            installed,
            reclaimable: vm.reclaimable(),
            wired: vm.wired,
            compressor: vm.compressor,
            compressor_stored: vm.compressor_stored,
            anonymous: vm.internal,
            purgeable: vm.purgeable,
            swap_used: sysctl::<libc::xsw_usage>(c"vm.swapusage").map_or(0, |s| s.xsu_used),
            level: sysctl(c"kern.memorystatus_vm_pressure_level").unwrap_or(0),
            own_footprint,
            own_wired,
            machine_wire_limit,
            task_wire_limit,
            own_gpu,
            gpu_alloc: gpu_stat(c"Alloc system memory").ok_or("the GPU driver's allocation")?,
            gpu_ceiling,
        })
    }
}

/// The GPU driver's resident set, which its ceiling bounds, and its own resident memory; two IOKit walks.
pub fn gpu_inuse() -> (u64, u64) {
    let stat = |name: &CStr| gpu_stat(name).unwrap_or(0);
    (stat(c"In use system memory"), stat(c"In use system memory (driver)"))
}

/// A GPU-driver statistic from IOKit's PerformanceStatistics, such as `In use system memory`; `None` unreadable.
fn gpu_stat(name: &CStr) -> Option<u64> {
    use std::ffi::c_void;
    use std::os::raw::{c_char, c_int, c_uint};
    #[link(name = "IOKit", kind = "framework")]
    extern "C" {
        fn IOServiceMatching(name: *const c_char) -> *mut c_void;
        fn IOServiceGetMatchingService(main_port: c_uint, matching: *mut c_void) -> c_uint;
        fn IORegistryEntryCreateCFProperty(
            entry: c_uint, key: *const c_void, allocator: *const c_void, options: u32,
        ) -> *const c_void;
        fn IOObjectRelease(obj: c_uint) -> c_int;
    }
    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFStringCreateWithCString(alloc: *const c_void, s: *const c_char, encoding: u32) -> *const c_void;
        fn CFDictionaryGetValue(dict: *const c_void, key: *const c_void) -> *const c_void;
        fn CFNumberGetValue(num: *const c_void, ty: isize, out: *mut c_void) -> bool;
        fn CFRelease(cf: *const c_void);
    }
    const CF_STRING_UTF8: u32 = 0x0800_0100;
    const CF_NUMBER_SINT64: isize = 4;
    // SAFETY: the service lookup consumes the matching dictionary, and all else is released before returning.
    unsafe {
        let matching = IOServiceMatching(c"IOAccelerator".as_ptr());
        let service = if matching.is_null() { 0 } else { IOServiceGetMatchingService(0, matching) };
        if service == 0 {
            return None;
        }
        let key = |s: &CStr| CFStringCreateWithCString(std::ptr::null(), s.as_ptr(), CF_STRING_UTF8);
        let (stats_key, stat_key) = (key(c"PerformanceStatistics"), key(name));
        let stats = IORegistryEntryCreateCFProperty(service, stats_key, std::ptr::null(), 0);
        let mut out: i64 = 0;
        let num = if stats.is_null() { std::ptr::null() } else { CFDictionaryGetValue(stats, stat_key) };
        let ok = !num.is_null() && CFNumberGetValue(num, CF_NUMBER_SINT64, std::ptr::addr_of_mut!(out).cast());
        if !stats.is_null() {
            CFRelease(stats);
        }
        CFRelease(stats_key);
        CFRelease(stat_key);
        IOObjectRelease(service);
        ok.then(|| out.max(0) as u64)
    }
}

/// The capacity `--physical-mem` states, in bytes; 0 = the machine's own.
static ASSUMED_PHYSICAL: AtomicU64 = AtomicU64::new(0);

/// Emulate a smaller machine, set before the budget: its default and every bound follow; a larger one is refused.
pub fn assume_physical(gb: Option<f64>) -> std::result::Result<(), String> {
    let bytes = gb.filter(|gb| *gb > 0.0).map_or(0, |gb| (gb * GB) as u64);
    let pageable = machine_totals().map_or(u64::MAX, |(pageable, _)| pageable);
    if bytes > pageable {
        return Err(format!(
            "--physical-mem {:.1} GB over {:.1} GB this machine pages: emulates smaller machine only",
            bytes as f64 / GB,
            pageable as f64 / GB
        ));
    }
    ASSUMED_PHYSICAL.store(bytes, Ordering::Relaxed);
    Ok(())
}

pub fn physical_is_assumed() -> bool {
    ASSUMED_PHYSICAL.load(Ordering::Relaxed) > 0
}

/// What the kernel pages and the installed RAM, boot constants read once.
fn machine_totals() -> Option<(u64, u64)> {
    static TOTALS: OnceLock<Option<(u64, u64)>> = OnceLock::new();
    *TOTALS.get_or_init(|| Some((pages(c"vm.pages")?, sysctl(c"hw.memsize")?)))
}

/// The capacity the plan assumes (what the kernel pages, or `--physical-mem`) and the installed RAM.
pub fn memory_totals() -> Option<(u64, u64)> {
    let (pageable, installed) = machine_totals()?;
    let assumed = ASSUMED_PHYSICAL.load(Ordering::Relaxed);
    Some((if assumed > 0 { assumed } else { pageable }, installed))
}

/// The machine bound, for launch, admission and controller alike; negative is short.
pub fn machine_room_bytes(physical: u64, foreign: u64, peak: u64, margin: u64) -> i64 {
    signed(physical.saturating_sub(foreign)) - signed(peak.saturating_add(margin))
}

/// The wire caps are two sysctls: raising the machine-wide one leaves the per-task one, and the nearer refuses a lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bound {
    Machine,
    Gpu,
    WireMachine,
    WireTask,
    /// A running plan's alone: reclaimable memory less the controller's rest line.
    Reclaimable,
}

impl Bound {
    pub fn label(self) -> &'static str {
        match self {
            Bound::Machine => "machine",
            Bound::Gpu => "GPU",
            Bound::WireMachine => "machine wire",
            Bound::WireTask => "task wire",
            Bound::Reclaimable => "reclaimable",
        }
    }

    /// The machine's one setting ([`machine_limits_line`]) where a limit under it binds; nothing to set otherwise.
    pub fn limits_line(self, b: &Bounds) -> Option<String> {
        let limit = match self {
            Bound::Gpu => b.gpu_ceiling,
            Bound::WireMachine => b.machine_wire_limit,
            Bound::WireTask => b.task_wire_limit,
            Bound::Machine | Bound::Reclaimable => u64::MAX,
        };
        (limit < b.installed.saturating_sub(UNWIRED_RESERVE_BYTES)).then(|| machine_limits_line(b.installed))
    }
}

/// Kept unwired for the watchdog's floor and others' wiring: the three limits are the installed RAM less this.
pub const UNWIRED_RESERVE_BYTES: u64 = 5 << 30;

/// The machine's one setting, run once a boot: the unwired reserve, and the task wire and GPU limits at one figure.
pub fn machine_limits_line(installed: u64) -> String {
    let wirable = installed.saturating_sub(UNWIRED_RESERVE_BYTES);
    format!(
        "sudo sysctl vm.global_no_user_wire_amount={UNWIRED_RESERVE_BYTES} vm.user_wire_limit={wirable} \
         iogpu.wired_limit_mb={}",
        wirable >> 20
    )
}

/// The four bounds' terms, as a snapshot reads them or the controller settles them.
#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    pub physical: u64,
    pub installed: u64,
    pub foreign: u64,
    /// Inside `foreign` on the machine bound; on the GPU bound, others' share of the ceiling.
    pub foreign_gpu: u64,
    pub gpu_ceiling: u64,
    pub foreign_wired: u64,
    pub machine_wire_limit: u64,
    pub task_wire_limit: u64,
}

impl Bounds {
    /// Room (negative: short) and the binding bound; the GPU ceiling is the machine's, so others' allocation counts.
    pub fn room(&self, peak: u64, metal: u64, locked: u64, margin: u64, step: u64) -> (i64, Bound) {
        let locked = signed(locked.saturating_add(margin));
        [
            (machine_room_bytes(self.physical, self.foreign, peak, margin.saturating_add(step)), Bound::Machine),
            (signed(self.gpu_ceiling.saturating_sub(self.foreign_gpu)) - signed(metal), Bound::Gpu),
            (signed(self.machine_wire_limit.saturating_sub(self.foreign_wired)) - locked, Bound::WireMachine),
            (signed(self.task_wire_limit) - locked, Bound::WireTask),
        ]
        .into_iter()
        .min_by_key(|&(room, _)| room)
        .expect("four bounds")
    }
}

pub fn signed(bytes: u64) -> i64 {
    bytes.min(i64::MAX as u64) as i64
}

/// The windowed foreign term's widest swing, quiet and under a prefill peak: hysteresis, not a unit anything moves by.
pub const CHANGE_QUANTUM: u64 = (0.53 * GB) as u64;

/// The pressure line: below it the controller sheds from the sample in hand and the sensor samples fast.
pub fn under_pressure(s: &MemorySnapshot, floor: u64) -> bool {
    floor > 0 && s.reclaimable < floor + CHANGE_QUANTUM
}

/// The most a claimant takes at once, about a heavy application's launch, that a plan at rest absorbs with no strike:
/// one thirty-second of the installed memory, 2 GB on a 64 GB machine and 0.75 GB on 24 GB.
pub fn claimant_step() -> u64 {
    machine_totals().map_or(2 << 30, |(_, installed)| installed / 32)
}

/// The step as the watchdog charges it: at the bottom a claimant also re-activates pages and grows the compressor.
pub fn claimant_charge() -> u64 {
    claimant_step() + 2 * CHANGE_QUANTUM
}

pub fn foreign_gpu_now() -> u64 {
    MemorySnapshot::read().map_or(0, |m| m.foreign_gpu())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::test_lock;

    const G: u64 = 1 << 30;

    #[test]
    fn snapshot_reads_and_is_sane() {
        let s = MemorySnapshot::read().expect("the syscalls work on macOS");
        // More than a page, less than the machine: catches index and page-size bugs.
        assert!(s.reclaimable > 16384, "reclaimable {} implausibly small", s.reclaimable);
        assert!(s.non_droppable() < s.physical, "non-droppable {} past physical {}", s.non_droppable(), s.physical);
        assert!(s.own_footprint > 0 && s.own_footprint < s.non_droppable() && s.gpu_ceiling > 0);
        assert!(matches!(s.level, 1 | 2 | 4), "the memorystatus sysctl is present on every supported macOS");
        let (pageable, installed) = memory_totals().expect("both sysctls answer");
        assert_eq!((s.physical, s.installed), (pageable, installed));
        assert!(pageable <= installed && pageable > installed / 2, "pageable {pageable} of installed {installed}");
    }

    #[test]
    fn the_ledger_says_every_key_in_its_order() {
        let s = MemorySnapshot { own_footprint: G, wired: 3 * G, reclaimable: 2 * G, level: 1, ..Default::default() };
        let line = s.ledger();
        let keys: Vec<&str> = line.split(' ').map(|kv| kv.split('=').next().unwrap_or_default()).collect();
        let want = "own other free wired compressed stored anon purgeable swap \
                    own_wired own_gpu gpu_alloc gpu_ceiling physical floor region region_live region_asked \
                    region_fenced region_lent level";
        assert_eq!(keys.join(" "), want);
        assert!(line.starts_with("own=1.00 other=2.00 free=2.00 wired=3.00 compressed=0.00 "), "{line}");
        assert!(line.ends_with(" level=normal"), "{line}");
    }

    #[test]
    fn room_is_the_smaller_bound_and_signed() {
        // Half-GB units keep every value exact. 64 − 6 = 58 ≥ 53.5 + 1.5.
        const H: u64 = 1 << 29;
        assert_eq!(machine_room_bytes(128 * H, 12 * H, 107 * H, 3 * H), 6 * H as i64);
        assert_eq!(machine_room_bytes(128 * H, 22 * H, 107 * H, 3 * H), -4 * (H as i64));
        assert_eq!(machine_room_bytes(128 * H, 18 * H, 107 * H, 3 * H), 0);
        // The GPU bound binds when foreign GPU allocation eats the ceiling; own shares never read as foreign.
        let s = MemorySnapshot { physical: 128 * H, installed: 128 * H, wired: 12 * H, ..Default::default() };
        let s = MemorySnapshot { gpu_ceiling: 110 * H, gpu_alloc: 10 * H, own_gpu: 2 * H, ..s };
        assert_eq!((s.foreign(), s.foreign_gpu()), (12 * H, 8 * H));
        // The ceiling is the machine's: the 8 H other clients hold comes out of it, as out of physical memory.
        assert_eq!(s.bounds().room(107 * H, 100 * H, 90 * H, 3 * H, 0), (2 * H as i64, Bound::Gpu));
        let ample = MemorySnapshot { gpu_ceiling: 120 * H, ..s };
        assert_eq!(ample.bounds().room(107 * H, 100 * H, 90 * H, 3 * H, 0), (6 * H as i64, Bound::Machine));
        // A launch's step comes off the machine bound and no other.
        assert_eq!(ample.bounds().room(107 * H, 100 * H, 90 * H, 3 * H, 2 * H), (4 * H as i64, Bound::Machine));
        assert_eq!(s.bounds().room(107 * H, 100 * H, 90 * H, 3 * H, 2 * H), (2 * H as i64, Bound::Gpu));
        let own = MemorySnapshot { own_footprint: 20 * H, own_gpu: 20 * H, ..s };
        assert_eq!((own.foreign(), own.foreign_gpu()), (0, 0));
        // The machine-wide wire bound binds when others' wired pages, the plan's locks and the margin reach the limit.
        let wire = MemorySnapshot { machine_wire_limit: 112 * H, own_wired: 2 * H, ..s };
        assert_eq!(wire.foreign_wired(), 10 * H);
        assert_eq!(wire.bounds().room(107 * H, 100 * H, 98 * H, 3 * H, 0), (H as i64, Bound::WireMachine));
        assert_eq!(wire.bounds().room(107 * H, 100 * H, 100 * H, 3 * H, 0), (-(H as i64), Bound::WireMachine));
        // The per-task cap binds alone once the machine-wide limit is raised.
        let task = MemorySnapshot { machine_wire_limit: 130 * H, task_wire_limit: 102 * H, ..wire };
        assert_eq!(task.bounds().room(107 * H, 100 * H, 100 * H, 3 * H, 0), (-(H as i64), Bound::WireTask));
        // A limit under the machine's one setting names that line; one at the setting has nothing to set.
        let line = "sudo sysctl vm.global_no_user_wire_amount=5368709120 vm.user_wire_limit=63350767616 iogpu.wired_limit_mb=60416";
        assert_eq!(machine_limits_line(128 * H), line);
        assert_eq!(Bound::WireTask.limits_line(&task.bounds()).as_deref(), Some(line));
        let set = MemorySnapshot { task_wire_limit: 118 * H, ..task };
        assert!(Bound::WireTask.limits_line(&set.bounds()).is_none());
        assert!(Bound::Machine.limits_line(&task.bounds()).is_none());
    }

    /// Else a drafting model's verify would change phase every round.
    #[test]
    fn a_completion_step_of_several_tokens_is_a_decode() {
        let _g = test_lock();
        begin_step(false);
        assert_eq!(phase(), Phase::Decode as usize);
        begin_step(true);
        assert_eq!(phase(), Phase::Prefill as usize, "a prompt step is a prefill");
    }

    #[test]
    fn the_floor_and_the_ceiling_refuse_a_page_past_the_line_and_nothing_disarmed() {
        let gib = |g: f64| (g * (1u64 << 30) as f64) as u64;
        let page = PAGE as u64;
        // (refused, the floor's or the ceiling's answer, the case)
        for (refused, answer, what) in [
            (true, short(gib(1.36), gib(0.875), gib(1.5)), "the claim that took the machine"),
            (false, short(gib(49.18), gib(0.875), gib(1.5)), "a healthy row is refused nothing"),
            (false, short(gib(2.5), gib(1.0), gib(1.5)), "exactly the floor is room enough"),
            (true, short(gib(2.5) - page, gib(1.0), gib(1.5)), "a page under the floor is not"),
            (false, short(gib(0.01), gib(8.0), 0), "a disarmed floor refuses nothing, however short the machine is"),
            (false, over(gib(42.15), gib(9.77), gib(54.0)), "a byte-exact row is refused nothing"),
            (true, over(gib(42.15), gib(12.71), gib(54.0)), "the pool grown at a 70,000-character prompt"),
            (false, over(gib(53.0), gib(1.0), gib(54.0)), "exactly the ceiling is room"),
            (true, over(gib(53.0), gib(1.0) + page, gib(54.0)), "a page past the ceiling is not"),
            (false, over(gib(64.0), gib(64.0), 0), "a disarmed ceiling refuses nothing, however far past the machine"),
        ] {
            assert_eq!(answer.is_some(), refused, "{what}");
        }
    }
}

pub use macos::{reclaimable, vm_pages};

mod macos {
    use super::VmPages;
    use libc::{host_statistics, host_statistics64, HOST_VM_INFO, HOST_VM_INFO64};
    use std::os::raw::{c_int, c_uint};
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    extern "C" {
        fn mach_host_self() -> c_uint;
    }

    /// `vm_statistics64_data_t` is exactly 38 32-bit words.
    const WORDS: usize = 38;
    const INACTIVE: usize = 2;
    const WIRED: usize = 3;
    const PURGEABLE: usize = 22;
    const COMPRESSOR: usize = 32;
    const INTERNAL: usize = 35;
    const COMPRESSED_UNCOMPRESSED: usize = 36;

    /// The kernel serves each view fresh only a few times a second, so the reader rotates all four.
    const VIEWS: [(bool, c_int, c_uint); 4] =
        [(true, HOST_VM_INFO64, 38), (true, HOST_VM_INFO64, 24), (false, HOST_VM_INFO, 15), (false, HOST_VM_INFO, 14)];
    const FULL: usize = 0;

    fn ask(view: usize) -> Option<[u32; WORDS]> {
        static HOST: OnceLock<c_uint> = OnceLock::new();
        // SAFETY: mach_host_self has no preconditions; one port ref, cached.
        let host = *HOST.get_or_init(|| unsafe { mach_host_self() });
        let (wide, flavor, mut count) = VIEWS[view];
        let mut out = [0u32; WORDS];
        // SAFETY: `out` holds the `count` words the flavor writes.
        let rc = unsafe {
            if wide {
                host_statistics64(host, flavor, out.as_mut_ptr().cast(), &mut count)
            } else {
                host_statistics(host, flavor, out.as_mut_ptr().cast(), &mut count)
            }
        };
        (rc == 0).then_some(out)
    }

    /// A call within a [`super::TICK`] of the last reads what it read, so a burst of takes asks the kernel nothing.
    struct Reading {
        asked: Option<Instant>,
        asked_full: Option<Instant>,
        seen_full: bool,
        next: usize,
        last: [[u32; WORDS]; VIEWS.len()],
        pages: VmPages,
    }

    impl Reading {
        /// False for the kernel's repeat, an answer bit-identical to the view's last: a fresh one's fault counts move.
        fn take(&mut self, view: usize, words: &[u32; WORDS], live: (u64, u64)) -> bool {
            (self.pages.free, self.pages.speculative) = live;
            if *words == self.last[view] {
                return false;
            }
            self.last[view] = *words;
            let page = super::PAGE as u64;
            let bytes = |i: usize| u64::from(words[i]) * page;
            if view == FULL {
                self.seen_full = true;
                self.pages = VmPages {
                    free: live.0,
                    inactive: bytes(INACTIVE),
                    speculative: live.1,
                    wired: bytes(WIRED),
                    purgeable: bytes(PURGEABLE),
                    compressor: bytes(COMPRESSOR),
                    internal: bytes(INTERNAL),
                    compressor_stored: (u64::from(words[COMPRESSED_UNCOMPRESSED])
                        | (u64::from(words[COMPRESSED_UNCOMPRESSED + 1]) << 32))
                        .saturating_mul(page),
                };
            } else {
                // The floor's inactive word sits where the full view has it.
                self.pages.inactive = bytes(INACTIVE);
            }
            true
        }
    }

    const BLANK: Reading = Reading {
        asked: None,
        asked_full: None,
        seen_full: false,
        next: FULL,
        last: [[0; WORDS]; VIEWS.len()],
        pages: VmPages {
            free: 0,
            inactive: 0,
            speculative: 0,
            wired: 0,
            purgeable: 0,
            compressor: 0,
            internal: 0,
            compressor_stored: 0,
        },
    };

    static READING: Mutex<Reading> = Mutex::new(BLANK);

    /// Past a tick, asks the full view if `whole` is wanted and was last asked four ticks ago or more, else the next.
    fn read(whole: bool) -> Option<VmPages> {
        let mut r = READING.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let now = Instant::now();
        if r.asked.is_some_and(|at| now.duration_since(at) < super::TICK) {
            return r.seen_full.then_some(r.pages);
        }
        let due = r.asked_full.map_or(true, |at| now.duration_since(at) >= super::TICK * VIEWS.len() as u32);
        let view = if whole && due { FULL } else { r.next };
        r.next = (view + 1) % VIEWS.len();
        r.asked = Some(now);
        if view == FULL {
            r.asked_full = Some(now);
        }
        let words = ask(view)?;
        let live = (super::live_free()?, super::pages(c"vm.page_speculative_count")?);
        r.take(view, &words, live);
        r.seen_full.then_some(r.pages)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// `x`'s distance from the band two readings either side of it make.
        fn miss(before: u64, x: u64, after: u64) -> u64 {
            let (lo, hi) = (before.min(after), before.max(after));
            lo.saturating_sub(x).max(x.saturating_sub(hi))
        }

        /// Forty rounds a half tick apart.
        fn rounds(mut round: impl FnMut()) {
            for _ in 0..40 {
                round();
                std::thread::sleep(super::super::TICK / 2);
            }
        }

        #[test]
        fn a_repeat_is_no_reading_and_a_narrow_view_moves_the_floors_words() {
            let page = super::super::PAGE as u64;
            let mut r = BLANK;
            let mut full = [0u32; WORDS];
            (full[INACTIVE], full[WIRED], full[INTERNAL]) = (300, 5000, 700);
            assert!(r.take(FULL, &full, (900 * page, 100 * page)));
            let read = (r.pages.free, r.pages.speculative, r.pages.wired, r.pages.internal, r.pages.reclaimable());
            assert_eq!(read, (900 * page, 100 * page, 5000 * page, 700 * page, 1300 * page));
            assert!(!r.take(FULL, &full, (800 * page, 100 * page)), "the same words again are the kernel's repeat");
            assert_eq!(r.pages.reclaimable(), 1200 * page, "the live counts move all the same");
            let mut narrow = [0u32; WORDS];
            (narrow[INACTIVE], narrow[WIRED]) = (250, 9999);
            assert!(r.take(3, &narrow, (300 * page, 100 * page)));
            assert_eq!(r.pages.reclaimable(), 650 * page, "the live free and speculative, the narrow view's inactive");
            let rest = (r.pages.wired, r.pages.internal);
            assert_eq!(rest, (5000 * page, 700 * page), "the rest as the full view read it");
            assert!(!r.take(3, &narrow, (300 * page, 100 * page)));
        }

        /// Each read placed against the live count read either side, the nearest of the rounds counting: a miss past
        /// half the speculative bytes says the reading's free term is not the kernel's live count.
        #[test]
        fn free_is_the_kernels_live_count() {
            let mut nearest = (u64::MAX, 0, 0, 0);
            rounds(|| {
                let before = super::super::live_free().expect("the kernel answers");
                let vm = vm_pages().expect("the machine answers");
                let after = super::super::live_free().expect("the kernel answers");
                let miss = miss(before, vm.free, after);
                if miss < nearest.0 {
                    nearest = (miss, vm.free, before.min(after), vm.speculative);
                }
            });
            let (miss, free, lo, speculative) = nearest;
            assert!(miss <= speculative / 2, "free {free}: band from {lo} missed by {miss}, {speculative} speculative");
        }

        /// Each view against the full view read either side, the nearest of a second's rounds.
        #[test]
        fn the_four_views_agree_on_the_inactive_pages() {
            let page = super::super::PAGE as u64;
            let of = |v: usize| ask(v).map(|w| u64::from(w[INACTIVE]) * page).expect("the view answers");
            let mut nearest = [u64::MAX; VIEWS.len()];
            rounds(|| {
                for (v, near) in nearest.iter_mut().enumerate().skip(1) {
                    let (before, x, after) = (of(FULL), of(v), of(FULL));
                    assert!(before > 0, "the full view answers a quantity");
                    *near = (*near).min(miss(before, x, after));
                }
            });
            assert!(nearest[1..].iter().all(|&n| n < 1 << 28), "nearest misses of the full view: {nearest:?}");
        }
    }

    /// The floor's quantity, from whichever view answers next.
    pub fn reclaimable() -> Option<u64> {
        Some(read(false)?.reclaimable())
    }

    pub fn vm_pages() -> Option<VmPages> {
        read(true)
    }
}
