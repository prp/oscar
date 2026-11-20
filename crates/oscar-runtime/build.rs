use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../../third_party/ggml-quants");
    // The version shown is the crate's number and the commit built, "1.0.0 (fe9d3f76)"; the number alone outside git.
    println!("cargo:rerun-if-changed=../../.git/logs/HEAD");
    let commit = Command::new("git").args(["rev-parse", "--short", "HEAD"]).output().ok();
    let commit = commit.filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    let suffix = commit.map_or_else(String::new, |c| format!(" ({c})"));
    println!("cargo:rustc-env=OSCAR_VERSION={}{suffix}", env!("CARGO_PKG_VERSION"));
    cc::Build::new()
        .file("../../third_party/ggml-quants/ggml-quants.c")
        .file("../../third_party/ggml-quants/shim.c")
        .include("../../third_party/ggml-quants")
        .opt_level(2)
        // The headers are vendored verbatim, helpers this build never calls included.
        .flag_if_supported("-Wno-unused-function")
        .compile("ggml_kquants");
}
