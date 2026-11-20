//! The drives the expert read path spans; a drive's reader count is its own property, not the model's.

use std::path::{Path, PathBuf};

pub struct DrivePreset {
    pub name: &'static str,
    /// Concurrent readers that saturate this drive; past it a reader adds no bandwidth.
    pub readers: usize,
}

pub const DRIVES: &[DrivePreset] =
    &[DrivePreset { name: "internal-nvme", readers: 4 }, DrivePreset { name: "thunderbolt-nvme", readers: 2 }];

pub fn drive(name: &str) -> anyhow::Result<&'static DrivePreset> {
    DRIVES.iter().find(|d| d.name == name).ok_or_else(|| {
        anyhow::anyhow!("unknown drive {name}: have {}", DRIVES.iter().map(|d| d.name).collect::<Vec<_>>().join(", "))
    })
}

/// `--mirror [<drive>:]<dir>`: a replica root holding the store's own layout.
pub struct Mirror {
    pub root: PathBuf,
    pub drive: &'static DrivePreset,
}

impl Mirror {
    /// Without a drive prefix, a `thunderbolt-nvme`.
    pub fn parse(spec: &str) -> anyhow::Result<Self> {
        let (name, dir) = match spec.split_once(':') {
            // A bare absolute path keeps its leading component.
            Some((n, d)) if !n.is_empty() && !n.starts_with('/') => (n, d),
            _ => ("thunderbolt-nvme", spec),
        };
        Ok(Mirror { root: PathBuf::from(dir), drive: drive(name)? })
    }

    /// Each shard's `<family>/<variant>/<file>` tail under the root, all present: a short set corrupts tensor lookup.
    pub fn shard_paths(&self, gguf_paths: &[PathBuf]) -> anyhow::Result<Vec<PathBuf>> {
        let mut out = Vec::with_capacity(gguf_paths.len());
        for p in gguf_paths {
            let tail = tail3(p)
                .ok_or_else(|| anyhow::anyhow!("shard path {} has no <family>/<variant>/<file> tail", p.display()))?;
            let mirrored = self.root.join(tail);
            anyhow::ensure!(mirrored.exists(), "mirror {} holds no {}", self.root.display(), mirrored.display());
            out.push(mirrored);
        }
        Ok(out)
    }
}

/// The `<family>/<variant>/<file>` tail of a shard path.
fn tail3(p: &Path) -> Option<PathBuf> {
    let parts: Vec<_> = p.components().collect();
    (parts.len() >= 3).then(|| parts[parts.len() - 3..].iter().collect())
}

static MIRROR: std::sync::OnceLock<Option<Mirror>> = std::sync::OnceLock::new();

/// Parsed here so a bad drive name fails at the flag, not at the pool build.
pub fn install(spec: Option<&str>) -> anyhow::Result<()> {
    let m = spec.map(Mirror::parse).transpose()?;
    let _ = MIRROR.set(m);
    Ok(())
}

pub fn installed() -> Option<&'static Mirror> {
    MIRROR.get().and_then(|m| m.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bare path is the default drive's, a prefix names one, an unknown one is refused; a mirror holds the
    /// family/variant/file tail, and a missing replica is refused rather than silently short.
    #[test]
    fn a_mirror_is_parsed_into_its_drive_and_root_and_mirrors_the_family_variant_file_tail() {
        for (spec, root, drive, readers) in [
            ("/Volumes/TB5-1/oscar/models", "/Volumes/TB5-1/oscar/models", "thunderbolt-nvme", 2),
            ("internal-nvme:/mnt/second", "/mnt/second", "internal-nvme", 4),
        ] {
            let m = Mirror::parse(spec).expect("parse");
            assert_eq!((m.root, m.drive.name, m.drive.readers), (PathBuf::from(root), drive, readers));
        }
        assert!(Mirror::parse("no-such-drive:/mnt/x").is_err());
        let m = Mirror::parse("/replica").expect("parse");
        let shard = PathBuf::from("/Users/x/.oscar/models/Fam-GGUF/varQ3/m-00001-of-00002.gguf");
        assert_eq!(tail3(&shard).expect("tail"), PathBuf::from("Fam-GGUF/varQ3/m-00001-of-00002.gguf"));
        assert!(m.shard_paths(&[shard]).is_err());
    }
}
