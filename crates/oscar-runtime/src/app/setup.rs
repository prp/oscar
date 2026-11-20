//! The first launch from a bundle offers, once, the `oscar` command: a link to the bundle's binary under
//! /usr/local/bin, which belongs to the system, so Oscar's own alert says what is asked before macOS's password
//! dialog (whose words cannot be changed) asks for it. A quarantined download runs translocated, from a random
//! read-only copy where that link and Start at Login would point: offered a move to the Applications folder first.

use std::path::Path;
use std::process::Command;

use anyhow::Context;
use objc2::MainThreadMarker;
use objc2_app_kit::{NSAlert, NSAlertFirstButtonReturn};
use objc2_foundation::NSString;

use crate::cli::settings::Settings;

const LINK: &str = "/usr/local/bin/oscar";

pub fn translocated(exe: &Path) -> bool {
    exe.to_string_lossy().contains("/AppTranslocation/")
}

#[cfg(test)]
#[test]
fn a_translocated_path_is_known() {
    assert!(translocated(Path::new("/private/var/folders/x/T/AppTranslocation/1-2/d/Oscar.app/Contents/MacOS/oscar")));
    assert!(!translocated(Path::new("/Applications/Oscar.app/Contents/MacOS/oscar")));
}

/// The move offered to a translocated launch: Move copies the bundle to /Applications (the user's own Applications
/// folder when that is not writable), lifts the quarantine from the copy and relaunches it; `true` then, for this
/// launch to end. Not Now runs on as is, the other offers held back.
pub fn relocate(mtm: MainThreadMarker) -> anyhow::Result<bool> {
    let exe = std::env::current_exe()?;
    if !translocated(&exe) {
        return Ok(false);
    }
    let bundle = exe.ancestors().nth(3).context("the bundle around the binary")?;
    let alert = NSAlert::new(mtm);
    alert.setMessageText(&NSString::from_str("Move Oscar to the Applications folder?"));
    alert.setInformativeText(&NSString::from_str(
        "Oscar was opened from where it was downloaded, which macOS runs as a temporary copy. From the Applications \
         folder the oscar command and Start at Login work. Nothing else changes.",
    ));
    alert.addButtonWithTitle(&NSString::from_str("Move"));
    alert.addButtonWithTitle(&NSString::from_str("Not Now"));
    super::activate(mtm);
    if alert.runModal() != NSAlertFirstButtonReturn {
        return Ok(false);
    }
    let home = std::env::var_os("HOME").map(std::path::PathBuf::from).unwrap_or_default();
    let folder = [Path::new("/Applications").to_path_buf(), home.join("Applications")]
        .into_iter()
        .find(|d| {
            std::fs::create_dir_all(d).and_then(|()| std::fs::metadata(d)).is_ok_and(|m| !m.permissions().readonly())
        })
        .context("no Applications folder to write")?;
    let dest = folder.join("Oscar.app");
    let _ = std::fs::remove_dir_all(&dest);
    for (tool, args) in [
        ("ditto", vec![bundle.as_os_str(), dest.as_os_str()]),
        ("xattr", vec!["-dr".as_ref(), "com.apple.quarantine".as_ref(), dest.as_os_str()]),
        ("open", vec!["-n".as_ref(), dest.as_os_str()]),
    ] {
        let out = Command::new(tool).args(&args).output().with_context(|| tool.to_string())?;
        anyhow::ensure!(out.status.success(), "{tool}: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(true)
}

/// Asked once, from a bundle not translocated; the answer is remembered as asked, whatever it was.
pub fn offer(mtm: MainThreadMarker) -> anyhow::Result<()> {
    let exe = std::env::current_exe()?;
    let mut settings = Settings::load()?;
    if settings.setup_offered == Some(true)
        || !exe.to_string_lossy().contains(".app/Contents/MacOS/")
        || translocated(&exe)
    {
        return Ok(());
    }
    settings.setup_offered = Some(true);
    settings.save()?;
    let alert = NSAlert::new(mtm);
    alert.setMessageText(&NSString::from_str("Install the oscar command?"));
    alert.setInformativeText(&NSString::from_str(&format!(
        "Oscar can also be used from a terminal, as `oscar`. For that, it links its program at {LINK}. That \
         directory belongs to the system, so macOS will ask for your administrator password next. The link is the \
         only change made."
    )));
    alert.addButtonWithTitle(&NSString::from_str("Install"));
    alert.addButtonWithTitle(&NSString::from_str("Not Now"));
    super::activate(mtm);
    if alert.runModal() != NSAlertFirstButtonReturn {
        return Ok(());
    }
    let script = format!(
        "do shell script \"mkdir -p /usr/local/bin && ln -sf '{}' {LINK}\" with administrator privileges",
        exe.display()
    );
    let out = std::process::Command::new("osascript").args(["-e", &script]).output().context("osascript")?;
    let err = String::from_utf8_lossy(&out.stderr);
    // Cancel is an answer, not trouble.
    anyhow::ensure!(out.status.success() || err.contains("(-128)"), "the link was not made: {}", err.trim());
    Ok(())
}
