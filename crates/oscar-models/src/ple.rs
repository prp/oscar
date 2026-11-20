//! Per-layer n-gram hash embeddings: a row-gather table too large to reside, read from the artifact on demand.

use std::collections::HashMap;
use std::os::unix::fs::FileExt;
use std::sync::{Mutex, OnceLock};

use candle_core::Result;

/// The hash parameters, transcribed from the GGUF `*.ple.*` keys.
#[derive(Debug, Clone)]
pub struct PleParams {
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub eos_token_id: u32,
    /// One per window position, the token's own first.
    pub multipliers: Vec<u64>,
    pub head_offsets: Vec<u32>,
    pub head_vocab_sizes: Vec<u32>,
}

impl PleParams {
    pub fn n_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// Appends one token's `n_heads()` row ids; `ctx[s]` is the token `s` back, `None` before the start.
    pub fn hash_token(&self, ctx: &[Option<u32>], out: &mut Vec<u32>) {
        let n = self.ngram_size;
        debug_assert_eq!(ctx.len(), n);
        // A missing or EOS predecessor turns it and all older to EOS; the token's own EOS cuts nothing.
        let mut resolved = vec![0u64; n];
        resolved[0] = u64::from(ctx[0].unwrap_or(self.eos_token_id));
        let mut cut = false;
        for s in 1..n {
            let t = if cut { None } else { ctx[s] };
            cut = cut || t.is_none() || t == Some(self.eos_token_id);
            resolved[s] = u64::from(if cut { self.eos_token_id } else { t.unwrap() });
        }
        for order in 2..=n {
            let mut mixed = resolved[0].wrapping_mul(self.multipliers[0]);
            for j in 1..order {
                mixed ^= resolved[j].wrapping_mul(self.multipliers[j]);
            }
            let base = (order - 2) * self.heads_per_ngram;
            for g in 0..self.heads_per_ngram {
                let h = base + g;
                let row = mixed % u64::from(self.head_vocab_sizes[h]) + u64::from(self.head_offsets[h]);
                out.push(row as u32);
            }
        }
    }
}

/// Where the table's rows live on disk and how one dequantises.
pub struct RowGatherStore {
    file: std::fs::File,
    base_offset: u64,
    pub head_dim: usize,
    row_bytes: usize,
    n_rows: u64,
    /// FIFO at [`Self::CACHE_ROWS`].
    cache: Mutex<RowCache>,
}

struct RowCache {
    rows: HashMap<u32, Vec<f32>>,
    order: std::collections::VecDeque<u32>,
}

impl RowGatherStore {
    /// The plan charges it as a fixed line.
    pub const CACHE_ROWS: usize = 100_000;
    /// One reader per row of a decode gather at the widest table, 16 heads.
    pub const READERS: usize = 16;
    /// 256 rows a reader, bounding the raw bytes a prefill gather holds at a time.
    pub const READ_BATCH: usize = 256 * Self::READERS;

    /// From the shard's tensor header; Q8_0, the one form the conversions keep.
    pub fn open_q8_0(path: &std::path::Path, offset: u64, head_dim: usize, n_rows: u64) -> Result<Self> {
        crate::architecture::ensure(head_dim % 32 == 0, "PLE head_dim must divide into Q8_0 blocks")?;
        let file =
            std::fs::File::open(path).map_err(|e| candle_core::Error::Msg(format!("PLE table open {path:?}: {e}")))?;
        Ok(Self {
            file,
            base_offset: offset,
            head_dim,
            row_bytes: head_dim / 32 * 34,
            n_rows,
            cache: Mutex::new(RowCache { rows: HashMap::new(), order: std::collections::VecDeque::new() }),
        })
    }

    /// The cache first, then the misses, each written out before it is cached: one gather may outgrow the cache.
    pub fn gather_into(&self, ids: &[u32], out: &mut [f32]) -> Result<()> {
        let hd = self.head_dim;
        crate::architecture::ensure(out.len() == ids.len() * hd, "PLE gather: the output is not one row per id")?;
        let mut miss_at: Vec<usize> = Vec::new();
        {
            let cache = self.cache.lock().unwrap();
            for (i, &id) in ids.iter().enumerate() {
                match cache.rows.get(&id) {
                    Some(row) => out[i * hd..(i + 1) * hd].copy_from_slice(row),
                    None => miss_at.push(i),
                }
            }
        }
        // The misses by row, each with the positions that want it.
        let mut wanted: Vec<(u32, usize)> = miss_at.iter().map(|&i| (ids[i], i)).collect();
        wanted.sort_unstable();
        let mut miss_ids: Vec<u32> = wanted.iter().map(|&(id, _)| id).collect();
        miss_ids.dedup();
        for &id in &miss_ids {
            crate::architecture::ensure(u64::from(id) < self.n_rows, "PLE row id out of range")?;
        }
        let rb = self.row_bytes;
        let mut raw = vec![0u8; miss_ids.len().min(Self::READ_BATCH) * rb];
        let mut row = vec![0f32; hd];
        let mut positions = wanted.iter().peekable();
        for batch in miss_ids.chunks(Self::READ_BATCH) {
            let raw = &mut raw[..batch.len() * rb];
            let per = batch.len().div_ceil(Self::READERS);
            let results: Vec<std::io::Result<()>> = std::thread::scope(|s| {
                let handles: Vec<_> = batch
                    .chunks(per)
                    .zip(raw.chunks_mut(per * rb))
                    .map(|(chunk, buf)| {
                        s.spawn(move || -> std::io::Result<()> {
                            for (j, &id) in chunk.iter().enumerate() {
                                let off = self.base_offset + u64::from(id) * rb as u64;
                                self.file.read_exact_at(&mut buf[j * rb..(j + 1) * rb], off)?;
                            }
                            Ok(())
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().expect("PLE reader")).collect()
            });
            for r in results {
                r.map_err(|e| candle_core::Error::Msg(format!("PLE row pread: {e}")))?;
            }
            let mut cache = self.cache.lock().unwrap();
            for (j, &id) in batch.iter().enumerate() {
                dequant_q8_0_row(&raw[j * rb..(j + 1) * rb], &mut row);
                while let Some(&(_, i)) = positions.next_if(|&&(want, _)| want == id) {
                    out[i * hd..(i + 1) * hd].copy_from_slice(&row);
                }
                if cache.rows.len() >= Self::CACHE_ROWS {
                    if let Some(old) = cache.order.pop_front() {
                        cache.rows.remove(&old);
                    }
                }
                cache.rows.insert(id, row.clone());
                cache.order.push_back(id);
            }
        }
        Ok(())
    }

    /// On Metal written where it is read, a CPU-written object of the working memory: no host copy of the gather.
    pub fn gather_tensor(
        &self, ids: &[u32], shape: (usize, usize, usize), device: &candle_core::Device,
    ) -> Result<candle_core::Tensor> {
        let candle_core::Device::Metal(md) = device else {
            candle_core::bail!("PLE gather: rows written into Metal memory, not onto {device:?}");
        };
        let n = ids.len() * self.head_dim;
        let buffer = md.new_staged_buffer(n, candle_core::DType::F32, "ple-gather")?;
        // SAFETY: `n` F32s of shared storage, this handle its only one, nothing encoded against it yet.
        let out = unsafe { std::slice::from_raw_parts_mut(buffer.contents().cast::<f32>(), n) };
        self.gather_into(ids, out)?;
        Ok(oscar_kernels::tensor_over(buffer, md, shape, candle_core::DType::F32))
    }
}

fn dequant_q8_0_row(raw: &[u8], out: &mut [f32]) {
    for (b, chunk) in raw.chunks_exact(34).enumerate() {
        let d = half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
        for (j, &q) in chunk[2..34].iter().enumerate() {
            out[b * 32 + j] = d * f32::from(q as i8);
        }
    }
}

/// Installed by the runtime once it resolves the shard paths, which the constructor cannot see.
static STORE: OnceLock<std::sync::Arc<RowGatherStore>> = OnceLock::new();

pub fn install_store(store: std::sync::Arc<RowGatherStore>) {
    let _ = STORE.set(store);
}

pub fn store() -> Option<&'static std::sync::Arc<RowGatherStore>> {
    STORE.get()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Goldens from the u64 wrapping mul and xor.
    #[test]
    fn hash_matches_reference_semantics() {
        let p = PleParams {
            ngram_size: 3,
            heads_per_ngram: 2,
            eos_token_id: 9,
            multipliers: vec![23703573157769, 20109073645365, 8052911324071],
            head_offsets: vec![0, 101, 211, 331],
            head_vocab_sizes: vec![101, 110, 120, 130],
        };
        let m = |t: u64, i: usize| t.wrapping_mul(p.multipliers[i]);
        // (the context, what it resolves to): a missing prev2 reads as EOS, an EOS at prev cuts prev and prev2 to
        // EOS, the token's own EOS does not cut its own context.
        for (ctx, (a, b, c)) in [
            ([Some(7), Some(5), Some(3)], (7, 5, 3)),
            ([Some(7), Some(5), None], (7, 5, 9)),
            ([Some(7), Some(9), Some(3)], (7, 9, 9)),
            ([Some(9), Some(5), Some(3)], (9, 5, 3)),
        ] {
            let (bi, mut got) = (m(a, 0) ^ m(b, 1), Vec::new());
            let tri = bi ^ m(c, 2);
            p.hash_token(&ctx, &mut got);
            assert_eq!(got, [bi % 101, bi % 110 + 101, tri % 120 + 211, tri % 130 + 331].map(|h| h as u32), "{ctx:?}");
        }
    }

    /// A Q8_0 table file, block `b` of row `r` scaled by `d(r)` over the bytes `q(r, b, j)`, and the store over it.
    fn table(
        name: &str, rows: usize, hd: usize, d: impl Fn(usize) -> f32, q: impl Fn(usize, usize, usize) -> u8,
    ) -> (RowGatherStore, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!("oscar-ple-{name}-{}.bin", std::process::id()));
        let (d, q) = (&d, &q);
        let bytes: Vec<u8> = (0..rows)
            .flat_map(|r| {
                let scale = half::f16::from_f32(d(r)).to_le_bytes();
                (0..hd / 32).flat_map(move |b| scale.into_iter().chain((0..32).map(move |j| q(r, b, j))))
            })
            .collect();
        std::fs::write(&path, &bytes).expect("write the table");
        (RowGatherStore::open_q8_0(&path, 0, hd, rows as u64).expect("open the table"), path)
    }

    #[test]
    fn a_gather_wider_than_the_cache_returns_every_row_and_a_gathered_tensor_is_the_host_gather() {
        let rows = RowGatherStore::CACHE_ROWS + 1000;
        let (store, path) = table("wide", rows, 32, |_| 1.0, |r, _, _| (r % 127) as u8);
        let ids: Vec<u32> = (0..rows as u32).chain([7, 7, rows as u32 - 1]).collect();
        let mut out = vec![0f32; ids.len() * 32];
        store.gather_into(&ids, &mut out).expect("the gather");
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(out[i * 32], (id % 127) as f32, "position {i}, row {id}");
        }
        let _ = std::fs::remove_file(&path);
        oscar_offload::pools::install_for_tests();
        let (rows, hd) = (3000usize, 64usize);
        let q = |r: usize, b: usize, j: usize| ((r * 3 + b * 5 + j) % 251) as u8;
        let (store, path) = table("tensor", rows, hd, |r| 0.25 + (r % 7) as f32, q);
        let ids: Vec<u32> = (0..2 * rows as u32).map(|i| (i * 7919) % rows as u32).collect();
        let mut host = vec![0f32; ids.len() * hd];
        store.gather_into(&ids, &mut host).expect("the host gather");
        let device = candle_core::Device::new_metal(0).expect("a Metal device");
        let tensor = store.gather_tensor(&ids, (1, ids.len(), hd), &device).expect("the gather in place");
        let read: Vec<f32> = tensor.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(read, host);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn q8_0_row_roundtrip() {
        let d = half::f16::from_f32(0.5);
        let mut raw = d.to_le_bytes().to_vec();
        for q in 0..32i8 {
            raw.push((q - 16) as u8);
        }
        let mut out = vec![0f32; 32];
        dequant_q8_0_row(&raw, &mut out);
        for (j, v) in out.iter().enumerate() {
            let want = d.to_f32() * (j as f32 - 16.0);
            assert!((v - want).abs() < 1e-6, "elem {j}: {v} vs {want}");
        }
    }
}
