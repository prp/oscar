//! Launch OpenCode: oscar's own `~/.oscar/opencode.json` (the `oscar` provider, the served ids as its models) and a
//! terminal running OpenCode with `OPENCODE_CONFIG` set to it; the user's own config untouched.

use std::path::PathBuf;

use anyhow::Context;
use objc2_app_kit::{NSApplicationActivationOptions, NSRunningApplication};
use objc2_foundation::NSString;
use oscar_models::rows::ModelPreset;

/// On the PATH, else where Homebrew, a manual install and OpenCode's own installer put it.
pub fn find() -> Option<PathBuf> {
    let path = std::env::var_os("PATH").unwrap_or_default();
    let own = std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".opencode/bin"));
    std::env::split_paths(&path)
        .chain([PathBuf::from("/opt/homebrew/bin"), PathBuf::from("/usr/local/bin")])
        .chain(own)
        .map(|dir| dir.join("opencode"))
        .find(|p| p.is_file())
}

/// The served ids of the row, as serve advertises them, less the generic `default+nothink`.
fn served(preset: ModelPreset, quant: Option<&str>) -> anyhow::Result<Vec<String>> {
    let (_, _, paths) = crate::model_store::locate(preset, quant, None, &[])?;
    let name = crate::load::report::model_name(&paths);
    Ok(preset.served_model_ids(&name).into_iter().map(|(id, _)| id).filter(|id| id.contains('/')).collect())
}

/// Written whole once each child's port is known; the file's path. The served model is the default and its `+nothink`
/// id the small one, over the user's own config, which OpenCode merges before this file.
pub fn write(preset: ModelPreset, quant: Option<&str>, port: u16) -> anyhow::Result<PathBuf> {
    let ids = served(preset, quant)?;
    let name = |id: &str| id.rsplit('/').next().unwrap_or(id).to_string();
    let models: serde_json::Map<String, serde_json::Value> =
        ids.iter().map(|id| (name(id), serde_json::json!({ "id": id, "name": name(id) }))).collect();
    let pick = |nothink: bool| {
        ids.iter().find(|id| id.ends_with("+nothink") == nothink).map(|id| format!("oscar/{}", name(id)))
    };
    let mut config = serde_json::json!({
        "$schema": "https://opencode.ai/config.json",
        "provider": { "oscar": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Oscar",
            "options": { "baseURL": format!("http://127.0.0.1:{port}/v1") },
            "models": models,
        } },
    });
    if let Some(model) = pick(false) {
        config["small_model"] = pick(true).unwrap_or_else(|| model.clone()).into();
        config["model"] = model.into();
    }
    let path = crate::cli::settings::dir().join("opencode.json");
    std::fs::write(&path, serde_json::to_string_pretty(&config)?)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

/// A Terminal window in `dir` running OpenCode.
pub fn launch(opencode: &std::path::Path, dir: &std::path::Path) -> anyhow::Result<()> {
    let config = crate::cli::settings::dir().join("opencode.json");
    terminal(
        &format!("cd '{}' && OPENCODE_CONFIG='{}' '{}'", dir.display(), config.display(), opencode.display()),
        None,
    )
    .map(drop)
}

const REFUSED: &str = "Terminal refused Oscar: allow it under System Settings > Privacy & Security > Automation";

/// A Terminal window running `line`, under the `profile` named when one is, its tty returned; Terminal started if
/// it is not, its startup window then the one used; brought to the front once the window exists (activated with
/// none, Terminal opens one of its own) and again from here, as a panel's close hands activation back to Oscar.
pub fn terminal(line: &str, profile: Option<&str>) -> anyhow::Result<String> {
    let line = line.replace('"', "\\\"");
    let profile =
        profile.map_or(String::new(), |p| format!("try\nset current settings of t to settings set \"{p}\"\nend try\n"));
    let script = format!(
        "set fresh to not (application \"Terminal\" is running)\ntell application \"Terminal\"\nif fresh then\n\
         repeat 50 times\nif (count of windows) > 0 then exit repeat\ndelay 0.1\nend repeat\nend if\n\
         if fresh and (count of windows) > 0 then\nset t to do script \"{line}\" in front window\nelse\n\
         set t to do script \"{line}\"\nend if\n{profile}activate\nreturn tty of t\nend tell"
    );
    let out = std::process::Command::new("osascript").args(["-e", &script]).output().context("running osascript")?;
    let err = String::from_utf8_lossy(&out.stderr);
    // -1743: the user refused Oscar's control of Terminal when macOS asked.
    let why =
        if err.contains("-1743") { REFUSED.to_string() } else { format!("Terminal did not open: {}", err.trim()) };
    anyhow::ensure!(out.status.success(), "{why}");
    for terminal in
        NSRunningApplication::runningApplicationsWithBundleIdentifier(&NSString::from_str("com.apple.Terminal")).iter()
    {
        terminal.activateWithOptions(NSApplicationActivationOptions::ActivateAllWindows);
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The window on `tty` closed once no process is left in it (its shell exited; a profile that closes on exit has
/// closed it already): from outside, as a close asked from inside is confirmed with the asker still running.
/// A thread waits on the script, so it leaves no zombie.
pub fn close_when_done(tty: &str) {
    let script = format!(
        "tell application \"Terminal\"\nrepeat\ndelay 1\ntry\nset w to first window whose tty of selected tab is \"{tty}\"\n\
         on error\nexit repeat\nend try\nif (count of processes of selected tab of w) = 0 then\nclose w\nexit repeat\n\
         end if\nend repeat\nend tell"
    );
    std::thread::spawn(move || {
        let _ = std::process::Command::new("osascript").args(["-e", &script]).output();
    });
}

#[cfg(test)]
#[test]
fn the_config_names_the_served_ids_under_the_oscar_provider() {
    let text = serde_json::to_string(
        &serde_json::json!({ "options": { "baseURL": format!("http://127.0.0.1:{}/v1", 11434) } }),
    )
    .unwrap();
    assert_eq!(text, r#"{"options":{"baseURL":"http://127.0.0.1:11434/v1"}}"#);
    assert!(find().is_none_or(|p| p.ends_with("opencode")));
}
