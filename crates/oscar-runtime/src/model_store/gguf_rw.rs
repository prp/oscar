//! Streaming GGUF rewrite: [`plan`] decides and prices each tensor exactly, [`execute`] streams one at a time.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use candle_core::quantized::{gguf_file, GgmlDType};
use candle_core::Device;

use super::conversion::ImatrixMap;
use super::kquant;

/// The page size: a mapped tensor needs it to reach Metal without a bounce copy.
pub const STORE_ALIGNMENT: u64 = 16384;

pub enum Disposition {
    Copy,
    /// A requant to the tensor's own dtype copies.
    Requant(GgmlDType),
    Skip,
}

pub enum Metadata {
    /// The first source's keys, `general.alignment` set to the store's.
    Copy,
    /// Exactly these keys, plus the alignment.
    Replace(Vec<(String, gguf_file::Value)>),
}

pub struct PlanEntry {
    pub name: String,
    /// A derived tensor's own name; reads use `name`.
    pub written_as: Option<String>,
    /// candle order (ne0 last); written to GGUF reversed.
    dims: Vec<usize>,
    pub src_index: usize,
    src_dtype: GgmlDType,
    src_abs_offset: u64,
    pub out_dtype: GgmlDType,
    pub out_bytes: u64,
}

impl PlanEntry {
    pub fn is_requant(&self) -> bool {
        self.out_dtype != self.src_dtype
    }
}

pub struct Plan {
    sources: Vec<PathBuf>,
    pub entries: Vec<PlanEntry>,
    /// A source's declared `*.block_count`, else the `blk.N` count over every source.
    pub n_layers: usize,
}

impl Plan {
    /// Exact, padding included; the header is the caller's slack.
    pub fn payload_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.out_bytes + pad_to(e.out_bytes)).sum()
    }
}

fn pad_to(n: u64) -> u64 {
    n.div_ceil(STORE_ALIGNMENT) * STORE_ALIGNMENT - n
}

fn read_content(path: &Path) -> anyhow::Result<gguf_file::Content> {
    crate::plan::header(path).map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))
}

/// `decide` sees each tensor in file order: name, dtype, candle dims and [`Plan::n_layers`].
pub fn plan(
    sources: &[PathBuf], mut decide: impl FnMut(&str, GgmlDType, &[usize], usize) -> Disposition,
) -> anyhow::Result<Plan> {
    let mut contents = Vec::with_capacity(sources.len());
    let mut n_layers = 0usize;
    for path in sources {
        let content = read_content(path)?;
        for i in content.tensor_infos.keys().filter_map(|name| crate::plan::block_index(name)) {
            n_layers = n_layers.max(i + 1);
        }
        contents.push(content);
    }
    let mut declared = contents.iter().flat_map(|c| &c.metadata).filter(|(k, _)| k.ends_with(".block_count"));
    let n_layers = declared.find_map(|(_, v)| usize::try_from(oscar_models::any_int(v)?).ok()).unwrap_or(n_layers);

    let mut entries: Vec<PlanEntry> = Vec::new();
    for (src_index, content) in contents.iter().enumerate() {
        let mut infos: Vec<(&String, &gguf_file::TensorInfo)> = content.tensor_infos.iter().collect();
        infos.sort_by_key(|(_, info)| info.offset);
        for (name, info) in infos {
            let dims = info.shape.dims().to_vec();
            let elems: usize = dims.iter().product();
            let out_dtype = match decide(name, info.ggml_dtype, &dims, n_layers) {
                Disposition::Skip => continue,
                Disposition::Copy => info.ggml_dtype,
                Disposition::Requant(dtype) if dtype == info.ggml_dtype => info.ggml_dtype,
                Disposition::Requant(dtype) => {
                    anyhow::ensure!(kquant::is_writable(dtype), "{name}: no ggml quantiser linked for {dtype:?}");
                    anyhow::ensure!(
                        elems % dtype.block_size() == 0,
                        "{name}: {elems} elements do not fill {dtype:?} blocks"
                    );
                    dtype
                }
            };
            anyhow::ensure!(!entries.iter().any(|e| e.name == *name), "{name} appears in more than one source shard");
            entries.push(PlanEntry {
                name: name.clone(),
                written_as: None,
                src_index,
                src_dtype: info.ggml_dtype,
                src_abs_offset: content.tensor_data_offset + info.offset,
                out_dtype,
                out_bytes: super::kquant::quant_bytes(out_dtype, elems) as u64,
                dims,
            });
        }
    }
    // Empty is legitimate, as a split's tensor-less first file; a caller for whom it is not says so.
    Ok(Plan { sources: sources.to_vec(), entries, n_layers })
}

/// A requantised tensor is weighted by its imatrix entry: per column, or per expert slice as `llama-quantize` does.
pub fn execute(
    plan: &Plan, out: &Path, metadata: Metadata, imatrix: &ImatrixMap, threads: usize,
    mut progress: impl FnMut(usize, usize),
) -> anyhow::Result<u64> {
    let mut meta: Vec<(String, gguf_file::Value)> = match metadata {
        Metadata::Copy => {
            let content = read_content(&plan.sources[0])?;
            let mut pairs: Vec<_> = content
                .metadata
                .iter()
                .filter(|(k, _)| k.as_str() != "general.alignment")
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            pairs.sort_by(|a, b| a.0.cmp(&b.0));
            pairs
        }
        Metadata::Replace(pairs) => pairs,
    };
    meta.push(("general.alignment".to_string(), gguf_file::Value::U32(STORE_ALIGNMENT as u32)));

    let mut header = Vec::with_capacity(1 << 20);
    header.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // "GGUF"
    header.extend_from_slice(&3u32.to_le_bytes()); // version 3
    header.extend_from_slice(&(plan.entries.len() as u64).to_le_bytes());
    header.extend_from_slice(&(meta.len() as u64).to_le_bytes());
    for (k, v) in &meta {
        gguf_file::write_string(&mut header, k)?;
        header.extend_from_slice(&v.value_type().to_u32().to_le_bytes());
        v.write(&mut header)?;
    }
    let mut offset = 0u64;
    for e in &plan.entries {
        gguf_file::write_string(&mut header, e.written_as.as_deref().unwrap_or(&e.name))?;
        header.extend_from_slice(&(e.dims.len() as u32).to_le_bytes());
        for &d in e.dims.iter().rev() {
            header.extend_from_slice(&(d as u64).to_le_bytes());
        }
        header.extend_from_slice(&e.out_dtype.to_u32().to_le_bytes());
        header.extend_from_slice(&offset.to_le_bytes());
        offset += e.out_bytes + pad_to(e.out_bytes);
    }

    let mut w = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(out)?);
    w.write_all(&header)?;
    w.write_all(&vec![0u8; pad_to(header.len() as u64) as usize])?;

    // One file per source, opened on demand; a requant reads its tensor at the plan's absolute offset.
    let mut files: Vec<Option<std::fs::File>> = plan.sources.iter().map(|_| None).collect();
    let mut buf = vec![0u8; 8 << 20];
    for (done, e) in plan.entries.iter().enumerate() {
        let file = match &mut files[e.src_index] {
            Some(f) => f,
            slot => slot.insert(std::fs::File::open(&plan.sources[e.src_index])?),
        };
        if e.is_requant() {
            let info = gguf_file::TensorInfo {
                ggml_dtype: e.src_dtype,
                shape: e.dims.clone().into(),
                offset: e.src_abs_offset,
            };
            let qt = info.read(file, 0, &Device::Cpu)?;
            let f32s: Vec<f32> = qt.dequantize(&Device::Cpu)?.flatten_all()?.to_vec1()?;
            let row_len = *e.dims.last().expect("planned tensors have rank >= 1");
            let bytes = match imatrix.get(&e.name).map(Vec::as_slice) {
                // Per-expert calibration: each leading-dim slice quantised with its own row.
                Some(im) if im.len() != row_len => {
                    let slices = im.len() / row_len;
                    anyhow::ensure!(
                        im.len() % row_len == 0 && e.dims.first() == Some(&slices),
                        "{}: imatrix holds {} weights for rows of {row_len} across {:?}",
                        e.name,
                        im.len(),
                        e.dims
                    );
                    let per = f32s.len() / slices;
                    let mut all = Vec::with_capacity(e.out_bytes as usize);
                    for i in 0..slices {
                        all.extend(kquant::quantize(
                            e.out_dtype,
                            &f32s[i * per..(i + 1) * per],
                            row_len,
                            Some(&im[i * row_len..(i + 1) * row_len]),
                            threads,
                        )?);
                    }
                    all
                }
                other => kquant::quantize(e.out_dtype, &f32s, row_len, other, threads)?,
            };
            anyhow::ensure!(bytes.len() as u64 == e.out_bytes, "{}: byte count drifted", e.name);
            w.write_all(&bytes)?;
        } else {
            file.seek(SeekFrom::Start(e.src_abs_offset))?;
            let mut left = e.out_bytes as usize;
            while left > 0 {
                let n = left.min(buf.len());
                file.read_exact(&mut buf[..n])?;
                w.write_all(&buf[..n])?;
                left -= n;
            }
        }
        w.write_all(&vec![0u8; pad_to(e.out_bytes) as usize])?;
        progress(done + 1, plan.entries.len());
    }
    w.flush()?;

    let written = w.get_mut().stream_position()?;
    let expect = header.len() as u64 + pad_to(header.len() as u64) + plan.payload_bytes();
    anyhow::ensure!(written == expect, "wrote {written} B, the header promised {expect}");
    Ok(written)
}
