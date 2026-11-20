//! Compiles `src/moe_fused.metal`, behind candle's own i-quant codebooks, and `src/paged.metal`, over the mistral.rs
//! fork's Metal helpers, into the metallibs the crate embeds.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The codebooks candle's i-quant kernels index, and ours: taken from it at each build, so no copy exists to drift.
const CANDLE_QUANTIZED: &str = "../../third_party/oscar-candle/candle-metal-kernels/src/metal_src/quantized.metal";
/// The paged kernels' vector types and bfloat helpers, included from the fork at each build.
const FORK_KERNELS: &str = "../../third_party/oscar-mistral.rs/mistralrs-paged-attn/src/metal/kernels";

fn main() -> Result<(), String> {
    for path in ["src/moe_fused.metal", "src/paged.metal", CANDLE_QUANTIZED, "build.rs"] {
        println!("cargo::rerun-if-changed={path}");
    }
    for helper in ["utils.metal", "float8.metal"] {
        println!("cargo::rerun-if-changed={FORK_KERNELS}/{helper}");
    }
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").map_err(|_| "OUT_DIR not set")?);
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return Err("oscar runs only on macOS (Apple Silicon)".into());
    }
    let tables = codebooks()?;
    let body = std::fs::read_to_string("src/moe_fused.metal").map_err(|e| e.to_string())?;
    let (src, air) = (out_dir.join("moe_fused_full.metal"), out_dir.join("moe_fused.air"));
    std::fs::write(&src, format!("{tables}{body}")).map_err(|e| e.to_string())?;
    xcrun(&["metal", "-std=metal3.1", "-O3", "-c"], &src, &air)?;
    xcrun(&["metallib"], &air, &out_dir.join("oscar_kernels.metallib"))?;
    let air = out_dir.join("paged.air");
    xcrun(&["metal", "-std=metal3.1", "-O3", "-c", "-I", FORK_KERNELS], Path::new("src/paged.metal"), &air)?;
    xcrun(&["metallib"], &air, &out_dir.join("oscar_paged.metallib"))
}

/// The four defines and the five tables the fused i-quant kernels read, as candle's `quantized.metal` spells them.
fn codebooks() -> Result<String, String> {
    let text = std::fs::read_to_string(CANDLE_QUANTIZED).map_err(|e| format!("{CANDLE_QUANTIZED}: {e}"))?;
    let missing = |what: &str| format!("{CANDLE_QUANTIZED}: no {what}");
    let mut out = String::new();
    for define in ["#define GGML_TABLE_BEGIN", "#define GGML_TABLE_END", "#define NGRID_IQ1S", "#define IQ1S_DELTA"] {
        out += text.lines().find(|l| l.starts_with(define)).ok_or_else(|| missing(define))?;
        out += "\n";
    }
    const END: &str = "GGML_TABLE_END()";
    for name in ["iq1s_grid_gpu", "iq2xxs_grid", "iq3xxs_grid", "ksigns_iq2xs", "kmask_iq2xs"] {
        let start = text.find(&format!(", {name}, ")).and_then(|i| text[..i].rfind("GGML_TABLE_BEGIN"));
        let table = start.and_then(|i| Some(&text[i..i + text[i..].find(END)? + END.len()]));
        out += table.ok_or_else(|| missing(name))?;
        out += "\n";
    }
    Ok(out)
}

fn xcrun(args: &[&str], input: &Path, output: &Path) -> Result<(), String> {
    let run = Command::new("xcrun").args(["--sdk", "macosx"]).args(args).arg(input).arg("-o").arg(output).output();
    let run = run.map_err(|e| format!("xcrun {}: {e}", args[0]))?;
    if !run.status.success() {
        return Err(format!("xcrun {} failed:\n{}", args[0], String::from_utf8_lossy(&run.stderr)));
    }
    Ok(())
}
