//! The drives: the shards read synchronously on `F_NOCACHE` descriptors, the reads' counters and the duty sampler.

use crate::machine::PAGE;
use crate::{Error, Result};
use std::fs::File;
use std::os::fd::{AsRawFd, RawFd};
use std::path::{Path, PathBuf};
use std::sync::atomic::{
    AtomicU64, AtomicUsize,
    Ordering::{self, Relaxed},
};

/// The shard file an extent's `file_offset` is in; `0` when single-file.
pub type ShardId = u8;

/// Per drive (the store and its `--mirror` replicas); drives past the last share its row.
pub const MAX_DEVICES: usize = 4;

/// Reads queued for a drive: an idle drive with an empty queue lacks producers, with a full one something else stops it.
pub static PENDING: AtomicUsize = AtomicUsize::new(0);

/// The drives' counters at an instant; busy times in mach ticks, racy and so biased low: a duty of 1.0 is no artifact.
#[derive(Clone, Copy, Debug, Default)]
pub struct Reads {
    /// Successful reads, their bytes (a short read is uncredited) and their time.
    pub calls: u64,
    pub bytes: u64,
    pub ns: u64,
    pub busy_raw: u64,
    pub bytes_dev: [u64; MAX_DEVICES],
    pub busy_raw_dev: [u64; MAX_DEVICES],
    /// The duty sampler's: samples, busy, idle with nothing queued, idle with reads queued; by reads in flight, 8 the last.
    pub duty: [u64; 4],
    pub depth: [u64; 9],
}

struct Counters {
    calls: AtomicU64,
    bytes: AtomicU64,
    ns: AtomicU64,
    /// Reads in flight and the tick their count rose from zero: the whole, then per drive.
    in_flight: [AtomicUsize; MAX_DEVICES + 1],
    rise: [AtomicU64; MAX_DEVICES + 1],
    busy: [AtomicU64; MAX_DEVICES + 1],
    bytes_dev: [AtomicU64; MAX_DEVICES],
    duty: [AtomicU64; 4],
    depth: [AtomicU64; 9],
}

const ZERO: AtomicU64 = AtomicU64::new(0);
static COUNTERS: Counters = Counters {
    calls: ZERO,
    bytes: ZERO,
    ns: ZERO,
    in_flight: [const { AtomicUsize::new(0) }; MAX_DEVICES + 1],
    rise: [ZERO; MAX_DEVICES + 1],
    busy: [ZERO; MAX_DEVICES + 1],
    bytes_dev: [ZERO; MAX_DEVICES],
    duty: [ZERO; 4],
    depth: [ZERO; 9],
};

/// Every drive counter now.
pub fn reads() -> Reads {
    let c = &COUNTERS;
    let load = |a: &[AtomicU64]| -> [u64; MAX_DEVICES] { std::array::from_fn(|d| a[d].load(Relaxed)) };
    Reads {
        calls: c.calls.load(Relaxed),
        bytes: c.bytes.load(Relaxed),
        ns: c.ns.load(Relaxed),
        busy_raw: c.busy[MAX_DEVICES].load(Relaxed),
        bytes_dev: load(&c.bytes_dev),
        busy_raw_dev: load(&c.busy),
        duty: std::array::from_fn(|i| c.duty[i].load(Relaxed)),
        depth: std::array::from_fn(|i| c.depth[i].load(Relaxed)),
    }
}

/// Busy runs from the 0→1 rise of reads in flight to the 1→0 fall.
fn rise(i: usize) {
    if COUNTERS.in_flight[i].fetch_add(1, Ordering::AcqRel) == 0 {
        COUNTERS.rise[i].store(now_raw(), Ordering::Release);
    }
}

fn fall(i: usize) {
    if COUNTERS.in_flight[i].fetch_sub(1, Ordering::AcqRel) == 1 {
        let rise = COUNTERS.rise[i].swap(0, Ordering::AcqRel);
        if rise != 0 {
            COUNTERS.busy[i].fetch_add(now_raw().saturating_sub(rise), Relaxed);
        }
    }
}

/// The host's mach clock, which the busy and barrier counters add raw: two calls a read are noise against a token.
#[allow(deprecated)]
pub fn now_raw() -> u64 {
    // SAFETY: no argument.
    unsafe { libc::mach_absolute_time() }
}

/// Converted once at read time.
#[allow(deprecated)]
pub fn raw_to_ns(ticks: u64) -> u64 {
    static TB: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    let (numer, denom) = *TB.get_or_init(|| {
        let mut tb = libc::mach_timebase_info { numer: 1, denom: 1 };
        // SAFETY: a valid out-pointer.
        unsafe { libc::mach_timebase_info(&mut tb) };
        (tb.numer as u64, tb.denom as u64)
    });
    ticks.saturating_mul(numer) / denom.max(1)
}

/// Under diag, every 200 µs (shorter than a read, ~600 µs): busy is added only at the last read's fall, so it is sampled.
pub fn start_duty_sampler() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let c = &COUNTERS;
        std::thread::Builder::new()
            .name("oscar-duty".into())
            .spawn(move || loop {
                let depth = c.in_flight[MAX_DEVICES].load(Relaxed);
                let at = if depth > 0 {
                    1
                } else if PENDING.load(Relaxed) > 0 {
                    3
                } else {
                    2
                };
                for i in [0, at] {
                    c.duty[i].fetch_add(1, Relaxed);
                }
                c.depth[depth.min(8)].fetch_add(1, Relaxed);
                std::thread::sleep(std::time::Duration::from_micros(200));
            })
            .expect("spawn oscar-duty");
    });
}

/// `files[device][shard]`: every device holds the same bytes at the same offsets; device 0 is the store's own.
pub struct TieredFile {
    paths: Vec<PathBuf>,
    files: Vec<Vec<File>>,
}

impl TieredFile {
    /// Every shard of `paths`, the index its shard id.
    pub fn open_many<P: AsRef<Path>>(paths: impl IntoIterator<Item = P>) -> Result<Self> {
        let (paths, files) = Self::open_device(paths)?;
        Ok(Self { paths, files: vec![files] })
    }

    /// The same shards on another device, in shard-id order; a different count would name other tensors.
    pub fn add_replica<P: AsRef<Path>>(&mut self, replicas: impl IntoIterator<Item = P>) -> Result<()> {
        let (paths, files) = Self::open_device(replicas)?;
        if files.len() != self.files[0].len() {
            return Err(Error::Layout(format!(
                "TieredFile::add_replica: {} shard(s) against the store's {}",
                files.len(),
                self.files[0].len()
            )));
        }
        self.paths.extend(paths);
        self.files.push(files);
        Ok(())
    }

    fn open_device<P: AsRef<Path>>(paths: impl IntoIterator<Item = P>) -> Result<(Vec<PathBuf>, Vec<File>)> {
        let mut opened = (Vec::new(), Vec::new());
        for path in paths {
            let path = path.as_ref().to_path_buf();
            let file = open_nocache(&path).inspect_err(
                |e| tracing::warn!(target: "oscar::experts", "A model file could not be opened: shard {}: {e}", path.display()),
            )?;
            opened.0.push(path);
            opened.1.push(file);
        }
        if opened.1.is_empty() {
            return Err(Error::Layout("TieredFile::open_many: no shards supplied".into()));
        }
        Ok(opened)
    }

    pub fn path(&self) -> &Path {
        &self.paths[0]
    }

    /// Every fill comes through here; `device` picks a replica, a matter of bandwidth, never correctness.
    ///
    /// # Safety
    /// `dst` must be writable for `len` bytes, untouched by any other thread or GPU launch until the call returns.
    pub unsafe fn pread(&self, device: u8, shard_id: ShardId, dst: *mut u8, len: usize, offset: u64) -> Result<()> {
        let out_of_range = |what: &str, n: u8, have: usize, of: &str| {
            Error::Layout(format!("TieredFile::pread: {what}={n} out of range (have {have} {of}(s))"))
        };
        let shards = self
            .files
            .get(device as usize)
            .ok_or_else(|| out_of_range("device", device, self.files.len(), "device"))?;
        let fd =
            shards.get(shard_id as usize).ok_or_else(|| out_of_range("shard_id", shard_id, shards.len(), "shard"))?;
        let dev = (device as usize).min(MAX_DEVICES - 1);
        rise(MAX_DEVICES);
        rise(dev);
        let t0 = std::time::Instant::now();
        // SAFETY: `dst` is the caller's `len`-byte target; `fd` is this file's live descriptor.
        let res = unsafe { aligned_pread(fd.as_raw_fd(), dst, len, offset) };
        let ns = t0.elapsed().as_nanos() as u64;
        fall(MAX_DEVICES);
        fall(dev);
        res?;
        let c = &COUNTERS;
        c.calls.fetch_add(1, Relaxed);
        c.bytes.fetch_add(len as u64, Relaxed);
        c.bytes_dev[dev].fetch_add(len as u64, Relaxed);
        c.ns.fetch_add(ns, Relaxed);
        Ok(())
    }
}

/// `F_NOCACHE` and no read-ahead, read through [`aligned_pread`] so the kernel keeps no transit pages.
pub fn open_nocache(path: &Path) -> Result<File> {
    let file = File::open(path)?;
    let fd = file.as_raw_fd();
    // SAFETY: fcntl on a descriptor `file` owns.
    unsafe {
        if libc::fcntl(fd, libc::F_NOCACHE, 1) < 0 {
            tracing::info!(target: "oscar::experts",
                "F_NOCACHE failed on {}: {}",
                path.display(),
                std::io::Error::last_os_error()
            );
        }
        libc::fcntl(fd, libc::F_RDAHEAD, 0);
    }
    Ok(file)
}

const BOUNCE_MIN: usize = 4 * 1024 * 1024;

/// A reading thread's bounce buffer after an extent of `largest` bytes, held for the thread's life.
pub fn bounce_bytes(largest: usize) -> usize {
    (largest.next_multiple_of(PAGE) + PAGE).max(BOUNCE_MIN)
}

/// Through a per-thread page-aligned bounce buffer: only an aligned offset and buffer stay out of the cache.
///
/// # Safety
/// `dst` must be valid for `len` writes; `fd` a live descriptor.
pub unsafe fn aligned_pread(fd: RawFd, dst: *mut u8, len: usize, offset: u64) -> Result<()> {
    use std::cell::RefCell;
    struct AlignedBuf {
        ptr: *mut u8,
        cap: usize,
    }
    impl Drop for AlignedBuf {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe { std::alloc::dealloc(self.ptr, std::alloc::Layout::from_size_align(self.cap, PAGE).unwrap()) }
            }
        }
    }
    thread_local! {
        static BUF: RefCell<AlignedBuf> = const { RefCell::new(AlignedBuf { ptr: std::ptr::null_mut(), cap: 0 }) };
    }
    let aligned_off = offset & !(PAGE as u64 - 1);
    let head = (offset - aligned_off) as usize;
    let aligned_len = (head + len).next_multiple_of(PAGE);
    BUF.with(|b| {
        let mut b = b.borrow_mut();
        if b.cap < aligned_len {
            let new_cap = aligned_len.next_multiple_of(PAGE).max(BOUNCE_MIN);
            let layout =
                std::alloc::Layout::from_size_align(new_cap, PAGE).map_err(|e| Error::Layout(e.to_string()))?;
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                return Err(Error::Layout("aligned_pread: bounce alloc failed".into()));
            }
            *b = AlignedBuf { ptr, cap: new_cap };
        }
        let n = unsafe { libc::pread(fd, b.ptr as *mut libc::c_void, aligned_len, aligned_off as i64) };
        if n < 0 {
            return Err(Error::Io(std::io::Error::last_os_error()));
        }
        if (n as usize) < head + len {
            return Err(Error::ShortRead { expected: len, got: n, offset });
        }
        unsafe { std::ptr::copy_nonoverlapping(b.ptr.add(head), dst, len) };
        Ok(())
    })
}
