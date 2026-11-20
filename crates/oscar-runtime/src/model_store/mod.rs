//! The model store, `<root>/<family_dir>/<variant>/*.gguf`: a variant's sorted glob is its tensor-lookup order.

pub mod conversion;
pub mod download;
#[cfg(test)]
pub(crate) mod fixtures;
pub mod gguf_rw;
pub mod hub;
pub mod imatrix;
pub mod kquant;

use std::path::{Path, PathBuf};

use oscar_models::rows::ModelPreset;

/// Where `download` writes and what `delete` manages.
pub fn managed_root() -> PathBuf {
    crate::cli::settings::dir().join("models")
}

/// The managed store, or `OSCAR_MODELS_DIR` in its place.
pub fn roots() -> Vec<PathBuf> {
    match std::env::var("OSCAR_MODELS_DIR") {
        Ok(d) if !d.is_empty() => vec![PathBuf::from(d)],
        _ => vec![managed_root()],
    }
}

/// Sorted: the tensor-lookup order.
pub fn gguf_names(dir: &Path) -> Vec<String> {
    let entries = std::fs::read_dir(dir).into_iter().flatten();
    let mut names: Vec<String> =
        entries.filter_map(|e| e.ok()?.file_name().into_string().ok().filter(|n| n.ends_with(".gguf"))).collect();
    names.sort();
    names
}

/// Allocated, not logical, bytes of the files directly in `dir`: a fetch preallocates a sparse file.
pub fn dir_bytes(dir: &Path) -> u64 {
    use std::os::unix::fs::MetadataExt;
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries.filter_map(|ent| ent.ok()?.metadata().ok()).filter(|m| m.is_file()).map(|m| m.blocks() * 512).sum()
}

/// `<root>/<family_dir>`'s variants holding a shard, with sizes, by name.
pub fn variants_in_root(root: &Path, family_dir: &str) -> Vec<(String, u64)> {
    let family = root.join(family_dir);
    let Ok(entries) = std::fs::read_dir(&family) else {
        return Vec::new();
    };
    let mut variants: Vec<(String, u64)> = entries
        .filter_map(|ent| {
            let ent = ent.ok()?;
            let name = ent.file_name().into_string().ok()?;
            let dir = ent.path();
            (!gguf_names(&dir).is_empty()).then(|| (name, dir_bytes(&dir)))
        })
        .collect();
    variants.sort();
    variants
}

/// A variant of the managed store: its directory, bytes and shard count; absent, a line naming `list`.
pub fn managed_variant(preset: ModelPreset, variant: &str) -> anyhow::Result<(PathBuf, u64, usize)> {
    let dir = managed_root().join(preset.def().family_dir).join(variant);
    let files = gguf_names(&dir).len();
    anyhow::ensure!(
        files > 0,
        "{} holds no {variant} variant of {}; `list` shows what is on disk",
        managed_root().display(),
        preset.label()
    );
    let bytes = dir_bytes(&dir);
    Ok((dir, bytes, files))
}

/// The bytes freed; a family directory left empty goes too, as clutter, not state.
pub fn remove_variant(preset: ModelPreset, variant: &str) -> anyhow::Result<u64> {
    let (dir, bytes, _) = managed_variant(preset, variant)?;
    std::fs::remove_dir_all(&dir)?;
    let _ = std::fs::remove_dir(dir.parent().expect("a variant has a family"));
    Ok(bytes)
}

/// Hidden under a dotted name until promoted by a rename, so a variant is verified or invisible.
pub fn staging_dir(family_dir: &str, variant: &str) -> PathBuf {
    managed_root().join(".partial").join(family_dir).join(variant)
}

pub fn promote(family_dir: &str, variant: &str) -> anyhow::Result<PathBuf> {
    let from = staging_dir(family_dir, variant);
    let to = managed_root().join(family_dir).join(variant);
    anyhow::ensure!(!to.exists(), "{} already exists", to.display());
    std::fs::create_dir_all(to.parent().expect("variant dirs have parents"))?;
    std::fs::rename(&from, &to)?;
    Ok(to)
}

/// The first root whose `<family_dir>/<variant>/` holds at least one shard.
pub fn resolve_variant(family_dir: &str, variant: &str) -> anyhow::Result<(PathBuf, Vec<String>)> {
    let roots = roots();
    for root in &roots {
        let dir = root.join(family_dir).join(variant);
        let names = gguf_names(&dir);
        if !names.is_empty() {
            return Ok((dir, names));
        }
    }
    anyhow::bail!(
        "no GGUF shards for {family_dir}/{variant} under {} - `oscar download` fetches models into store",
        roots.iter().map(|r| r.display().to_string()).collect::<Vec<_>>().join(" or ")
    )
}

/// `<stem>-00003-of-00008.gguf`'s eight parts; any other name itself.
fn split_parts(name: &str) -> Vec<String> {
    let parts = name.strip_suffix(".gguf").and_then(|s| s.rsplit_once("-of-")).and_then(|(head, n)| {
        let (stem, _) = head.rsplit_once('-').filter(|(_, k)| k.parse::<usize>().is_ok())?;
        let n: usize = n.parse().ok()?;
        Some((1..=n).map(|k| format!("{stem}-{k:05}-of-{n:05}.gguf")).collect())
    });
    parts.unwrap_or_else(|| vec![name.to_string()])
}

/// [`resolve_gguf`]'s answer plus the paths, every one present with every part of its split.
pub fn locate(
    preset: ModelPreset, quant: Option<&str>, dir: Option<&str>, files: &[String],
) -> anyhow::Result<(PathBuf, Vec<String>, Vec<PathBuf>)> {
    let (dir, names) = resolve_gguf(preset, quant, dir, files)?;
    if let Some(n) = names.iter().flat_map(|n| split_parts(n)).find(|n| !dir.join(n).exists()) {
        anyhow::bail!("{n} missing from {} (set --gguf-dir / --gguf-file / --model)", dir.display());
    }
    let paths = names.iter().map(|f| dir.join(f)).collect();
    Ok((dir, names, paths))
}

/// `--quant` picks a sibling variant, `--gguf-dir` wins over both, `--gguf-file` names over the sorted glob.
pub fn resolve_gguf(
    preset: ModelPreset, quant: Option<&str>, dir: Option<&str>, files: &[String],
) -> anyhow::Result<(PathBuf, Vec<String>)> {
    let (dir, names) = match dir {
        Some(d) => (PathBuf::from(d), gguf_names(Path::new(d))),
        None => resolve_variant(preset.def().family_dir, quant.unwrap_or(preset.def().production_variant))?,
    };
    let files = if files.is_empty() { names } else { files.to_vec() };
    anyhow::ensure!(!files.is_empty(), "no .gguf files in {}", dir.display());
    Ok((dir, files))
}

#[cfg(test)]
mod tests {
    use super::fixtures::{det_bytes, TempDir};
    use super::*;

    #[test]
    fn staged_credit_counts_allocated_not_logical_bytes() {
        use std::os::unix::fs::FileExt;
        let tmp = TempDir::new("sparse-credit");
        // A preallocated fetch in flight is sparse: 1 GB logical, one chunk written.
        let f = std::fs::File::create(tmp.0.join("pre.gguf")).unwrap();
        f.set_len(1 << 30).unwrap();
        f.write_at(&det_bytes(1 << 20, 7), 0).unwrap();
        drop(f);
        let credited = dir_bytes(&tmp.0);
        assert!(credited < (64 << 20), "a sparse preallocation credited its logical length: {credited}");
        assert!(credited >= (1 << 20), "the written bytes must be credited: {credited}");
    }

    #[test]
    fn a_split_with_a_part_missing_is_refused_by_its_name() {
        let tmp = TempDir::new("split-parts");
        let dir = tmp.0.to_str().unwrap();
        for name in ["Fam-v1-000-draft-head.gguf", "Fam-v1-00001-of-00002.gguf"] {
            std::fs::write(tmp.0.join(name), b"x").unwrap();
        }
        let err = locate(ModelPreset::Qwen35B, None, Some(dir), &[]).unwrap_err().to_string();
        assert!(err.contains("Fam-v1-00002-of-00002.gguf"), "{err}");
        std::fs::write(tmp.0.join("Fam-v1-00002-of-00002.gguf"), b"x").unwrap();
        assert_eq!(locate(ModelPreset::Qwen35B, None, Some(dir), &[]).unwrap().1.len(), 3);
    }
}
