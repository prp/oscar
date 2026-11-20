//! `cargo app` (an alias in .cargo/config.toml): builds the release runtime and writes Oscar.app beside it,
//! target/release/Oscar.app, or <dir>/Oscar.app with a directory argument. No Dock tile (LSUIElement); the icon from
//! the O path by the command line tools' swift, sips and iconutil, signed ad hoc (the notification centre takes
//! no unsigned bundle). The app's binary is the CLI too.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::{env, fs};

use anyhow::Context;

const PLIST: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Oscar</string>
  <key>CFBundleDisplayName</key><string>Oscar</string>
  <key>CFBundleIdentifier</key><string>uk.ac.imperial.oscar</string>
  <key>CFBundleVersion</key><string>VERSION</string>
  <key>CFBundleShortVersionString</key><string>VERSION</string>
  <key>CFBundleExecutable</key><string>oscar</string>
  <key>CFBundleIconFile</key><string>oscar</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSUIElement</key><true/>
  <key>LSMinimumSystemVersion</key><string>27.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
"#;

fn run(cmd: &mut Command) -> anyhow::Result<()> {
    let tool = cmd.get_program().to_string_lossy().into_owned();
    let status = cmd.status().with_context(|| tool.clone())?;
    anyhow::ensure!(status.success(), "{tool} failed ({status})");
    Ok(())
}

fn bundle(root: &Path, app: &Path) -> anyhow::Result<()> {
    // cargo run's package variables are dropped: build scripts (ring's) fingerprint them, and with them the runtime
    // built here and by a plain `cargo build` would each recompile the other's.
    let mut cargo = Command::new(env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    for (key, _) in env::vars_os() {
        let name = key.to_string_lossy();
        if name.starts_with("CARGO_PKG_") || name.starts_with("CARGO_MANIFEST_") {
            cargo.env_remove(&key);
        }
    }
    run(cargo.args(["build", "--release", "-p", "oscar-runtime"]).current_dir(root))?;
    let work = env::temp_dir().join(format!("oscar-app-{}", std::process::id()));
    let set = work.join("oscar.iconset");
    fs::create_dir_all(&set)?;
    let png = work.join("icon.png");
    run(Command::new("swift")
        .arg(root.join("scripts/render_icon.swift"))
        .arg(root.join("crates/oscar-runtime/assets/oscar_o.svg"))
        .arg(&png)
        .arg("1024"))?;
    for side in [16, 32, 128, 256, 512] {
        for (scale, suffix) in [(1, ""), (2, "@2x")] {
            let px = (side * scale).to_string();
            let out = set.join(format!("icon_{side}x{side}{suffix}.png"));
            run(Command::new("sips").args(["-z", &px, &px]).arg(&png).arg("--out").arg(out).stdout(Stdio::null()))?;
        }
    }
    let (macos, resources) = (app.join("Contents/MacOS"), app.join("Contents/Resources"));
    fs::create_dir_all(&macos)?;
    fs::create_dir_all(&resources)?;
    run(Command::new("iconutil").args(["-c", "icns"]).arg(&set).arg("-o").arg(resources.join("oscar.icns")))?;
    fs::remove_dir_all(&work)?;
    // Never over a binary in place: a running app keeps its own file, and a copy over it is killed at its next exec.
    let exe = macos.join("oscar");
    let _ = fs::remove_file(&exe);
    fs::copy(root.join("target/release/oscar-runtime"), &exe).context("copying the runtime binary")?;
    let _ = fs::remove_file(app.with_file_name("oscar"));
    std::os::unix::fs::symlink(&exe, app.with_file_name("oscar")).context("linking the command line")?;
    fs::write(app.join("Contents/Info.plist"), PLIST.replace("VERSION", env!("CARGO_PKG_VERSION")))?;
    run(Command::new("codesign").args(["--force", "--deep", "--sign", "-"]).arg(app))
}

fn main() -> anyhow::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize()?;
    let dir = env::args().nth(1).map_or_else(|| root.join("target/release"), PathBuf::from);
    let app = dir.join("Oscar.app");
    bundle(&root, &app)?;
    println!("app: {} written; open it, or copy it to /Applications", app.display());
    Ok(())
}
