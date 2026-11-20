//! `~/.oscar/settings.toml`, what both entries remember: the model row and its variant, the port, the mirror drive.
//! The menu bar's pick writes it; a flag wins over it.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::ValueEnum;
use oscar_models::rows::ModelPreset;
use serde::{Deserialize, Serialize};

pub const DEFAULT_PORT: u16 = 11434;

/// Every key optional; an unknown key stops the launch, as a misspelt one would silently lose its setting.
#[derive(Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    /// The `--model` spelling.
    pub model: Option<String>,
    pub quant: Option<String>,
    pub port: Option<u16>,
    pub mirror: Option<String>,
    /// The app's first-launch offer of the command made, whatever the answer.
    pub setup_offered: Option<bool>,
    /// The folder Launch OpenCode last opened.
    pub opencode_dir: Option<String>,
}

/// `~/.oscar`: the settings, the models, the KV store, the log and the opencode file.
pub fn dir() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(".")).join(".oscar")
}

pub fn path() -> PathBuf {
    dir().join("settings.toml")
}

impl Settings {
    /// The file, or nothing remembered where there is none; a file that does not read stops the launch.
    pub fn load() -> Result<Self> {
        Self::read(&path())
    }

    fn read(path: &Path) -> Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => toml::from_str(&text).with_context(|| format!("reading {}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
        }
    }

    /// The flags over the file; `--model` alone drops the file's variant, which was the remembered model's. With
    /// neither, the rows to choose from.
    pub fn preset(&self, flag: Option<ModelPreset>, quant: Option<&str>) -> Result<(ModelPreset, Option<String>)> {
        if let Some(preset) = flag {
            return Ok((preset, quant.map(str::to_string)));
        }
        let rows = ModelPreset::value_variants().iter().map(|p| p.label()).collect::<Vec<_>>().join(", ");
        let Some(name) = self.model.as_deref() else {
            anyhow::bail!(
                "no model: --model <one of {rows}>, or the Oscar menu's pick, which {} remembers",
                path().display()
            )
        };
        let preset = ModelPreset::from_str(name, false)
            .map_err(|_| anyhow::anyhow!("{} remembers an unknown model {name:?}; one of {rows}", path().display()))?;
        Ok((preset, quant.map(str::to_string).or_else(|| self.quant.clone())))
    }

    pub fn port(&self, flag: Option<u16>) -> u16 {
        flag.or(self.port).unwrap_or(DEFAULT_PORT)
    }

    /// Written whole, the menu's pick.
    pub fn save(&self) -> Result<()> {
        let p = path();
        std::fs::create_dir_all(dir())?;
        std::fs::write(&p, toml::to_string(self)?).with_context(|| format!("writing {}", p.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No file remembers nothing; the flags win over the file, `--model` alone dropping the file's variant; without a
    /// model the rows are named; a file that does not read, or holds a key oscar does not know, names itself.
    #[test]
    fn the_flags_win_over_the_file_and_no_model_names_the_rows() {
        let dir = crate::model_store::fixtures::TempDir::new("settings");
        let p = dir.0.join("settings.toml");
        let none = Settings::read(&p).unwrap();
        assert_eq!(none, Settings::default());
        let err = none.preset(None, Some("Q4")).unwrap_err().to_string();
        assert!(err.contains("--model <one of qwen3.6-35b-a3b, ") && err.contains("glm-5.3-flash"), "{err}");
        assert_eq!(none.port(None), DEFAULT_PORT);
        std::fs::write(&p, "model = \"glm-5.3-flash\"\nquant = \"Q3im\"\nport = 11435\nmirror = \"ext:/Volumes/x\"\n")
            .unwrap();
        let s = Settings::read(&p).unwrap();
        assert_eq!(toml::to_string(&s).unwrap(), std::fs::read_to_string(&p).unwrap());
        let (preset, quant) = s.preset(None, None).unwrap();
        assert_eq!((preset.label(), quant.as_deref()), ("glm-5.3-flash".to_string(), Some("Q3im")));
        let (preset, quant) = s.preset(Some(ModelPreset::KimiK3), None).unwrap();
        assert_eq!((preset.label(), quant), ("kimi-k3".to_string(), None));
        assert_eq!(s.preset(None, Some("Q4")).unwrap().1.as_deref(), Some("Q4"));
        assert_eq!((s.port(None), s.port(Some(8080)), s.mirror.as_deref()), (11435, 8080, Some("ext:/Volumes/x")));
        std::fs::write(&p, "model = \"glm-6\"\n").unwrap();
        let err = Settings::read(&p).unwrap().preset(None, None).unwrap_err().to_string();
        assert!(err.contains("unknown model \"glm-6\"") && err.contains("settings.toml"), "{err}");
        std::fs::write(&p, "modle = \"glm-5.2\"\n").unwrap();
        let err = format!("{:#}", Settings::read(&p).unwrap_err());
        assert!(err.contains("settings.toml") && err.contains("modle"), "{err}");
    }
}
