//! Published imatrix files, `.dat` or GGUF, as per-column weights by full tensor name.
//! `llama-quantize`'s vectors less its per-tensor scale, which cannot change which encoding minimises the error.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use super::conversion::ImatrixMap;

pub fn load(path: &Path) -> anyhow::Result<ImatrixMap> {
    // GGUF starts with its magic; anything else is `.dat`.
    let mut head = [0u8; 4];
    std::fs::File::open(path)?.read_exact(&mut head)?;
    if head == *b"GGUF" {
        load_gguf(path)
    } else {
        load_dat(path)
    }
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_f32s(r: &mut impl Read, n: usize) -> anyhow::Result<Vec<f32>> {
    let mut bytes = vec![0u8; n * 4];
    r.read_exact(&mut bytes)?;
    Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
}

/// An entry count, then per entry: name length, name, call count, per-column summed squared activations.
fn load_dat(path: &Path) -> anyhow::Result<ImatrixMap> {
    let mut r = std::io::BufReader::new(std::fs::File::open(path)?);
    let n_entries = read_u32(&mut r)? as usize;
    anyhow::ensure!(n_entries > 0 && n_entries < 1 << 20, "not an imatrix file: {n_entries} entries");
    let mut map = HashMap::with_capacity(n_entries);
    for _ in 0..n_entries {
        let len = read_u32(&mut r)? as usize;
        let mut name = vec![0u8; len];
        r.read_exact(&mut name)?;
        let _ncall = read_u32(&mut r)?;
        let nval = read_u32(&mut r)? as usize;
        map.insert(String::from_utf8(name)?, read_f32s(&mut r, nval)?);
    }
    Ok(map)
}

/// Per tensor, `<name>.in_sum2` over `<name>.counts` row by row; an uncalibrated row weighs 1, as in `llama-quantize`.
fn load_gguf(path: &Path) -> anyhow::Result<ImatrixMap> {
    use candle_core::quantized::gguf_file;
    let mut r = std::io::BufReader::new(std::fs::File::open(path)?);
    let content = gguf_file::Content::read(&mut r)?;
    let dev = candle_core::Device::Cpu;
    let mut map = HashMap::new();
    for (name, _) in &content.tensor_infos {
        let Some(base) = name.strip_suffix(".in_sum2") else {
            continue;
        };
        let counts_name = format!("{base}.counts");
        anyhow::ensure!(
            content.tensor_infos.contains_key(&counts_name),
            "imatrix {}: {name} has no matching {counts_name}",
            path.display()
        );
        let sums = content.tensor(&mut r, name, &dev)?.dequantize(&dev)?;
        let counts = content.tensor(&mut r, &counts_name, &dev)?.dequantize(&dev)?;
        let ne0 = *sums.dims().last().unwrap_or(&1);
        let sums: Vec<f32> = sums.flatten_all()?.to_vec1()?;
        let counts: Vec<f32> = counts.flatten_all()?.to_vec1()?;
        anyhow::ensure!(
            !counts.is_empty() && sums.len() == counts.len() * ne0,
            "imatrix {}: {name} is {} values against {} counts x {ne0}",
            path.display(),
            sums.len(),
            counts.len()
        );
        let weights = sums
            .chunks_exact(ne0)
            .zip(&counts)
            .flat_map(|(row, &c)| row.iter().map(move |&v| if c > 0.0 { v / c } else { 1.0 }))
            .collect();
        map.insert(base.to_string(), weights);
    }
    anyhow::ensure!(!map.is_empty(), "no imatrix entries in {}", path.display());
    Ok(map)
}
