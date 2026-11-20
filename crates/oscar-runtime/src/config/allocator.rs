//! libmalloc keeps freed large blocks' pages for the process's life unless told at its start, so oscar restarts itself.

use std::os::unix::process::CommandExt;

const LARGE_CACHE: &str = "MallocLargeCache";

/// Returns only where the switch is already set, or where the restart failed (said aloud).
pub fn restart_with_large_cache_off() {
    if std::env::var_os(LARGE_CACHE).is_some() {
        return;
    }
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let err = std::process::Command::new(exe).args(std::env::args_os().skip(1)).env(LARGE_CACHE, "0").exec();
    tracing::warn!(target: "oscar::loader",
        "Memory could not be returned to the system: allocator restart failed, freed large blocks are kept, {LARGE_CACHE}=0 ({err})"
    );
}
