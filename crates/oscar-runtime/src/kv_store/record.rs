//! The record file: the payload from byte 0, page-aligned for its direct read, then its description as
//! `[json][json_len: u64 LE][version: u32 LE][b"OSKR"]`. Kept out of the page cache, where a page read again turns
//! active, in neither of the plan's two quantities.

use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

use candle_core::{DType, Result};
use serde::{Deserialize, Serialize};

/// A read stays out of the page cache only when its file offset and its buffer are aligned to this.
pub(crate) const PAGE: u64 = oscar_offload::segments::PAGE as u64;

const MAGIC: &[u8; 4] = b"OSKR";
const TRAILER: u64 = 16;
/// Bumped with the payload layout; other versions are swept at startup.
pub(crate) const FORMAT_VERSION: u32 = 4;
/// The store's key beside its records.
pub(crate) const KEY_FILE: &str = "key.json";

/// The dtypes a payload holds, by their tags.
const DTYPE_TAGS: [(DType, &str); 6] = [
    (DType::F32, "F32"),
    (DType::F16, "F16"),
    (DType::BF16, "BF16"),
    (DType::U8, "U8"),
    (DType::U32, "U32"),
    (DType::I64, "I64"),
];

fn dtype_tag(d: DType) -> Result<&'static str> {
    let tag = DTYPE_TAGS.iter().find(|(t, _)| *t == d).map(|(_, s)| *s);
    tag.ok_or_else(|| candle_core::Error::Msg(format!("[context store] unsupported dtype {d:?}")))
}

pub(crate) fn tag_dtype(s: &str) -> Result<DType> {
    let dtype = DTYPE_TAGS.iter().find(|(_, t)| *t == s).map(|(d, _)| *d);
    dtype.ok_or_else(|| candle_core::Error::Msg(format!("[context store] unsupported dtype tag {s}")))
}

/// A tensor's geometry in a payload, offsets implied by order.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
pub(crate) struct TensorGeom {
    pub shape: Vec<usize>,
    pub dtype: String,
}

impl TensorGeom {
    pub fn new(shape: Vec<usize>, dtype: DType) -> Result<Self> {
        Ok(Self { shape, dtype: dtype_tag(dtype)?.to_string() })
    }
    pub fn nbytes(&self) -> Result<u64> {
        let n: usize = self.shape.iter().product();
        Ok((n * tag_dtype(&self.dtype)?.size_in_bytes()) as u64)
    }
}

/// A recurrent state's geometry, to rebuild its tensors.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct SnapMeta {
    pub seqlen_offsets: Vec<usize>,
    pub tensors: Vec<TensorGeom>,
}

impl SnapMeta {
    /// The pool's layout of the state: each layer's conv then recurrent tensor.
    pub(crate) fn layout(&self) -> Result<mistralrs_core::kv_cache::RecurrentStateLayout> {
        let mut layers = Vec::with_capacity(self.tensors.len() / 2);
        for pair in self.tensors.chunks_exact(2) {
            let (conv, recurrent) = (&pair[0], &pair[1]);
            let &[a, b, c] = conv.shape.as_slice() else {
                candle_core::bail!("[context store] conv state of {} dims", conv.shape.len());
            };
            layers.push(((a, b, c), tag_dtype(&conv.dtype)?, recurrent.shape.clone(), tag_dtype(&recurrent.dtype)?));
        }
        Ok(mistralrs_core::kv_cache::RecurrentStateLayout { layers })
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) enum RecordKind {
    /// One KV block: per KV-bearing layer, its K row then its V row.
    Block,
    /// A boundary record: the block holding the boundary's rows (none at a block-aligned boundary), then `meta`'s states;
    /// named by its parent's hash, position and `tokens`, as no chain hash covers a partial block.
    Tail {
        group: u64,
        tokens: Vec<u32>,
        /// Inline: the format keeps the state's two fields beside `tokens`.
        #[serde(flatten)]
        meta: SnapMeta,
    },
}

/// A record's description: what the index needs without the payload.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct RecordMeta {
    pub kind: RecordKind,
    /// A block's chain hash, or a boundary record's parent's.
    pub hash: u64,
    /// `0` for a chain root: a real hash of 0 is astronomically unlikely, and would only miss a dedup.
    pub parent: u64,
    /// Depth − 1 for a block; the full blocks below it for a boundary record.
    pub pos: u32,
    /// The coldness stamp at write time, so LRU order survives a restart.
    pub last_access: u64,
}

impl RecordMeta {
    /// A block's file by its chain hash; a boundary record's by its parent's, its position and a fold of its tokens.
    fn file_name(&self) -> String {
        match &self.kind {
            RecordKind::Block => format!("{:016x}.okv", self.hash),
            RecordKind::Tail { tokens, .. } => {
                let fold = |h: u64, t: &u32| (h ^ u64::from(*t)).wrapping_mul(0x100000001b3);
                format!("{:016x}-{}-{:016x}.okv", self.hash, self.pos, tokens.iter().fold(0xcbf29ce484222325, fold))
            }
        }
    }
}

/// Under a `.tmp` name until renamed: no reader sees a partial record, and crash recovery is a startup sweep.
pub(crate) fn write(dir: &Path, meta: &RecordMeta, payload: &[u8]) -> Result<PathBuf> {
    let path = dir.join(meta.file_name());
    let tmp = path.with_extension("tmp");
    let written = (|| -> std::io::Result<()> {
        let file = std::fs::File::create(&tmp)?;
        uncached(&file);
        let mut w = std::io::BufWriter::with_capacity(1 << 20, file);
        w.write_all(payload)?;
        let json = serde_json::to_vec(meta)?;
        w.write_all(&json)?;
        w.write_all(&(json.len() as u64).to_le_bytes())?;
        w.write_all(&FORMAT_VERSION.to_le_bytes())?;
        w.write_all(MAGIC)?;
        // fsync, not a drive flush: a flush a record would hold the writer to a third of the drive.
        // SAFETY: a live descriptor.
        unsafe { libc::fsync(w.into_inner()?.as_raw_fd()) };
        std::fs::rename(&tmp, &path)
    })();
    written.map(|()| path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        candle_core::Error::wrap(e)
    })
}

/// A record's description and its payload's length, from small reads of the trailer; `None` for another version's.
pub(crate) fn read_meta(path: &Path) -> Result<Option<(RecordMeta, u64)>> {
    let f = open(path)?;
    let total = f.metadata().map_err(candle_core::Error::wrap)?.len();
    let fail = |why: &str| candle_core::Error::Msg(format!("[context store] {}: {why}", path.display()));
    if total < TRAILER {
        return Err(fail("too short for a record"));
    }
    let trailer = read_bytes(&f, total - TRAILER, TRAILER as usize)?;
    if &trailer[12..] != MAGIC {
        return Err(fail("no record magic"));
    }
    if trailer[8..12] != FORMAT_VERSION.to_le_bytes() {
        return Ok(None);
    }
    let json_len = u64::from_le_bytes(trailer[..8].try_into().unwrap());
    if json_len + TRAILER > total {
        return Err(fail("description longer than the file"));
    }
    let json = read_bytes(&f, total - TRAILER - json_len, json_len as usize)?;
    let meta = serde_json::from_slice(&json).map_err(candle_core::Error::wrap)?;
    Ok(Some((meta, total - TRAILER - json_len)))
}

/// Deletes every file but the key and the readable current records; returns those with their payload lengths, and
/// another version's count.
pub(crate) fn sweep(dir: &Path) -> (Vec<(PathBuf, RecordMeta, u64)>, usize) {
    let (mut out, mut other_version) = (Vec::new(), 0);
    let Ok(rd) = std::fs::read_dir(dir) else {
        return (out, 0);
    };
    for p in rd.flatten().map(|e| e.path()) {
        if p.file_name().is_some_and(|n| n == KEY_FILE) {
            continue;
        }
        match p.extension().is_some_and(|x| x == "okv").then(|| read_meta(&p)) {
            Some(Ok(Some((meta, len)))) => {
                out.push((p, meta, len));
                continue;
            }
            Some(Ok(None)) => other_version += 1,
            Some(Err(e)) => tracing::info!("unreadable record swept: {} ({e})", p.display()),
            // A temporary, or an older format's file.
            None => {}
        }
        let _ = std::fs::remove_file(&p);
    }
    (out, other_version)
}

/// Every record gone, the key kept.
pub(crate) fn clear(dir: &Path) {
    for p in std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()) {
        if p.extension().is_some_and(|x| x == "okv") {
            let _ = std::fs::remove_file(&p);
        }
    }
}

/// F_NOCACHE: a write stays out of the page cache at any alignment.
fn uncached(f: &std::fs::File) {
    // SAFETY: a live descriptor, and F_NOCACHE takes an int.
    unsafe { libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1) };
}

/// A store file opened for its reads: the page cache off, no read-ahead.
pub(crate) fn open(path: &Path) -> Result<std::fs::File> {
    oscar_offload::io::open_nocache(path)
        .map_err(|e| candle_core::Error::Msg(format!("[context store] {}: {e}", path.display())))
}

/// `[0, run)` of `f` into `dst`, direct; it may pass the payload's end but must return `need` bytes.
///
/// # Safety
/// `dst` is page-aligned and writable for `run` bytes nothing else touches until the call returns.
pub(crate) unsafe fn read_payload(f: &std::fs::File, run: u64, need: u64, dst: *mut u8) -> Result<()> {
    debug_assert!(run % PAGE == 0 && dst as u64 % PAGE == 0);
    let mut got = 0u64;
    while got < run {
        // At most INT_MAX bytes a call.
        let n = (run - got).min(1 << 30) as usize;
        let r = unsafe { libc::pread(f.as_raw_fd(), dst.add(got as usize).cast(), n, got as libc::off_t) };
        if r > 0 {
            got += r as u64;
        } else if r == 0 {
            break;
        } else if std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted {
            return Err(candle_core::Error::wrap(std::io::Error::last_os_error()));
        }
    }
    if got < need {
        candle_core::bail!("[context store] record ended {got} bytes into {need}-byte read");
    }
    Ok(())
}

/// Direct through io's page-aligned bounce buffer, for the small reads with no arena range to land in.
pub(crate) fn read_bytes(f: &std::fs::File, offset: u64, len: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; len];
    if len > 0 {
        // SAFETY: `out` is `len` bytes this call owns, and `f` is open.
        unsafe { oscar_offload::io::aligned_pread(f.as_raw_fd(), out.as_mut_ptr(), len, offset) }
            .map_err(|e| candle_core::Error::Msg(format!("[context store] {e}")))?;
    }
    Ok(out)
}

/// A small store file (the key) read whole.
pub(crate) fn read_file(path: &Path) -> Result<Vec<u8>> {
    let f = open(path)?;
    let len = f.metadata().map_err(candle_core::Error::wrap)?.len();
    read_bytes(&f, 0, len as usize)
}

/// A small store file (the key) written whole.
pub(crate) fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut f = std::fs::File::create(path).map_err(candle_core::Error::wrap)?;
    uncached(&f);
    f.write_all(bytes).map_err(candle_core::Error::wrap)
}
