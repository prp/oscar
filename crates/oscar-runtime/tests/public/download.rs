//! The download engine end to end on the local stub: its phases point `HOME` and `OSCAR_MODELS_DIR` at sandboxes,
//! which every store call reads, so a test touching either holds `ENV` and runs in order.

use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Mutex;

use candle_core::quantized::{gguf_file, GgmlDType};
use oscar_models::rows::ModelPreset;
use oscar_runtime::model_store::conversion::decide;
use oscar_runtime::model_store::download::{download_variant, Event};
use oscar_runtime::model_store::{self, gguf_rw};
use oscar_runtime::plan::header;

use crate::helpers::{gguf_bytes, manifest_json, quantised, serve, sha_hex, TempDir};

/// Held by every test that sets `HOME` or `OSCAR_MODELS_DIR`: the process has one environment.
static ENV: Mutex<()> = Mutex::new(());

/// A copy-through tensor, plus on the last shard `output.weight`, which the GLM-5.2 conversion requantises to Q6_K.
fn glm_shard(i: usize, last: bool) -> Vec<u8> {
    let plain = (format!("blk.{i}.plain.weight"), quantised(8, 512, 100 + i as u32, GgmlDType::Q8_0));
    let out = ("output.weight".to_string(), quantised(4, 512, 900, GgmlDType::Q8_0));
    let tensors: Vec<(&str, &_)> =
        std::iter::once(&plain).chain(last.then_some(&out)).map(|(n, t)| (n.as_str(), t)).collect();
    gguf_bytes(&[], &tensors)
}

/// One tensor the overlay rules requantise, one they leave to the pass-through.
fn k3_shard(i: usize) -> Vec<u8> {
    let attn = quantised(8, 512, 200 + i as u32, GgmlDType::Q8_0);
    let other = quantised(4, 512, 300 + i as u32, GgmlDType::Q8_0);
    gguf_bytes(&[], &[(&format!("blk.{i}.attn_q.weight"), &attn), (&format!("blk.{i}.other.weight"), &other)])
}

/// A split's published paths, `<stem>-0000i-of-0000n.gguf`.
fn split(stem: &str, n: usize) -> Vec<String> {
    (1..=n).map(|i| format!("{stem}-{i:05}-of-{n:05}.gguf")).collect()
}

/// The stub's files: each path with its body.
fn files<'a>(paths: &'a [String], bodies: &'a [Vec<u8>]) -> Vec<(&'a str, &'a [u8])> {
    paths.iter().map(String::as_str).zip(bodies.iter().map(Vec::as_slice)).collect()
}

/// Points the store and resolution into a fresh sandbox; dropped, it puts the variables back as it found them.
struct Sandbox {
    _held: [TempDir; 2],
    before: [Option<OsString>; 2],
}

const VARS: [&str; 2] = ["HOME", "OSCAR_MODELS_DIR"];

fn sandbox(tag: &str) -> Sandbox {
    let (home, root) = (TempDir::new(&format!("pipe-home-{tag}")), TempDir::new(&format!("pipe-root-{tag}")));
    let before = VARS.map(std::env::var_os);
    std::env::set_var(VARS[0], &home.0);
    std::env::set_var(VARS[1], &root.0);
    Sandbox { _held: [home, root], before }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        for (var, value) in VARS.iter().zip(&self.before) {
            match value {
                Some(v) => std::env::set_var(var, v),
                None => std::env::remove_var(var),
            }
        }
    }
}

/// `preset`'s download from the stub at `base`, and what it said: the lines, then the details.
async fn download(base: &str, preset: ModelPreset) -> (anyhow::Result<PathBuf>, Vec<String>, Vec<String>) {
    let (mut lines, mut details) = (Vec::new(), Vec::new());
    let result = download_variant(&reqwest::Client::new(), base, preset, 2, &mut |ev| match ev {
        Event::Line(l) => lines.push(l.to_string()),
        Event::Detail(l) => details.push(l.to_string()),
        Event::FetchTotal(_) | Event::Fetched(_) | Event::Kept(_) | Event::Doing(_) => {}
    })
    .await;
    (result, lines, details)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pipelined_download_end_to_end() {
    let _env = ENV.lock().unwrap_or_else(|e| e.into_inner());
    // A staged variant is invisible until the promote, into the managed store alone.
    let _guard = sandbox("promote");
    std::env::remove_var("OSCAR_MODELS_DIR");
    let staging = model_store::staging_dir("Fam-GGUF", "v1");
    std::fs::create_dir_all(staging.join("src")).unwrap();
    std::fs::write(staging.join("out.gguf"), b"artifact").unwrap();
    std::fs::write(staging.join("src").join("source.gguf"), b"source").unwrap();
    let resolved = model_store::resolve_variant("Fam-GGUF", "v1");
    assert!(resolved.is_err(), "a staged variant must be invisible to resolution");
    std::fs::remove_dir_all(staging.join("src")).unwrap();
    let dir = model_store::promote("Fam-GGUF", "v1").unwrap();
    assert_eq!(model_store::resolve_variant("Fam-GGUF", "v1").unwrap(), (dir, vec!["out.gguf".to_string()]));
    assert!(model_store::promote("Fam-GGUF", "v1").is_err(), "a second promote must refuse");

    // A failed run, then the resume to completion.
    let _guard = sandbox("glm");
    let shards: Vec<Vec<u8>> = (0..3).map(|i| glm_shard(i, i == 2)).collect();
    let paths = split("Q8_0/GLM-5.2-Q8_0", 3);
    let (state, base) = serve("unsloth/GLM-5.2-GGUF", &files(&paths, &shards));
    // Cut near shard 2's end: past its header, so pricing succeeds and its body fails, after shard 1.
    let shard2 = format!("/unsloth/GLM-5.2-GGUF/resolve/main/{}", paths[1]);
    state.lock().unwrap().truncate_at.insert(shard2, shards[1].len() - 16);
    let (result, lines, details) = download(&base, ModelPreset::Glm52).await;
    assert!(result.is_err(), "the cut shard-2 stream must fail the run");
    let staging = model_store::staging_dir("GLM-5.2-GGUF", "expertsQ3");
    assert!(staging.join("GLM-5.2-expertsQ3-00001-of-00003.gguf").exists(), "shard 1's verified output must be kept");
    let source = staging.join("src").join("GLM-5.2-Q8_0-00001-of-00003.gguf");
    assert!(!source.exists(), "shard 1's source is deleted once its rewrite is done");
    let ruled = |details: &[String]| details.iter().any(|l| l.contains("two source shards"));
    assert!(ruled(&details), "the admission's detail must state the ruled charge: {details:?}");
    assert!(lines.iter().any(|l| l.contains("/3 files,")), "{lines:?}");
    // `src/` keeps shard 1's header prefix, so the resume fetches nothing of it.
    state.lock().unwrap().truncate_at.clear();
    let before = state.lock().unwrap().requests.len();
    let (result, _, details) = download(&base, ModelPreset::Glm52).await;
    let dir = result.expect("resume completes");
    assert!(details.iter().any(|l| l.contains("present, kept")), "{details:?}");
    let requests = state.lock().unwrap().requests[before..].to_vec();
    let shard1 = requests.iter().filter(|(p, _)| p.ends_with("GLM-5.2-Q8_0-00001-of-00003.gguf")).count();
    assert_eq!(shard1, 0, "the resumed run must not fetch shard 1 again");
    let produced = model_store::gguf_names(&dir);
    assert_eq!(produced.len(), 3, "{produced:?}");
    let scratch = TempDir::new("pipe-direct");
    let (src, direct) = (scratch.0.join("shard3.gguf"), scratch.0.join("direct.gguf"));
    std::fs::write(&src, &shards[2]).unwrap();
    let conversion = &ModelPreset::Glm52.def().conversion;
    let plan = gguf_rw::plan(&[src], |n, _, dims, l| decide(&conversion.transform, n, dims, l)).unwrap();
    gguf_rw::execute(&plan, &direct, gguf_rw::Metadata::Copy, &Default::default(), 2, |_, _| {}).unwrap();
    let pipelined = std::fs::read(dir.join("GLM-5.2-expertsQ3-00003-of-00003.gguf")).unwrap();
    assert_eq!(pipelined, std::fs::read(&direct).unwrap(), "pipelined output diverges from the engine run directly");

    // Shards of a third of the free space each: a summed charge would refuse, the exact one admits a KB output and the
    // two shards a rewrite beside the next fetch keeps.
    let _guard = sandbox("admit");
    let third = oscar_offload::machine::volume_free(&model_store::managed_root()).unwrap() / 3;
    let lying = |paths: &[String]| {
        manifest_json(&paths.iter().map(|p| (p.as_str(), third, sha_hex(b"whatever"))).collect::<Vec<_>>())
    };
    let (state, base) = serve("unsloth/GLM-5.2-GGUF", &files(&paths, &shards));
    state.lock().unwrap().manifest = lying(&paths);
    let (result, _, details) = download(&base, ModelPreset::Glm52).await;
    let err = result.unwrap_err().to_string();
    assert!(!err.contains("cannot host"), "output-plus-two-shards must admit these sizes, to fail at the stub: {err}");
    assert!(ruled(&details), "{details:?}");
    // A overlay run keeps every source: three such shards fill the volume before the overlay's own byte.
    let _guard = sandbox("admit-k3");
    let (paths, empty) = (split("UD-IQ1_M/Kimi-K3-UD-IQ1_M", 3), vec![gguf_bytes(&[], &[]); 3]);
    let (state, base) = serve("unsloth/Kimi-K3-GGUF", &files(&paths, &empty));
    state.lock().unwrap().manifest = lying(&paths);
    let err = download(&base, ModelPreset::KimiK3).await.0.unwrap_err().to_string();
    assert!(err.contains("cannot host download"), "fetch-all must refuse: {err}");

    // `block_count = 8` in shard 1, blk.4 alone in shard 2: Q4_K by the model's 8 layers, Q6_K by the shard's 5.
    let _guard = sandbox("morebits");
    // Shard 1 is tensor-less, like a real split's first file.
    let shards = [
        gguf_bytes(&[("synthetic.block_count", gguf_file::Value::U32(8))], &[]),
        gguf_bytes(&[], &[("blk.4.attn_v.weight", &quantised(4, 512, 504, GgmlDType::Q8_0))]),
    ];
    // A `.dat` imatrix of one all-ones entry: entry count, name length, name, call count, weight count, weights.
    let name = b"blk.4.attn_v.weight";
    let le = |w: u32| w.to_le_bytes().to_vec();
    let dat = [le(1), le(name.len() as u32), name.to_vec(), le(1), le(512), 1f32.to_le_bytes().repeat(512)].concat();
    let paths = split("Q8_0/Qwen3-235B-A22B-Q8_0", 2);
    let mut published = files(&paths, &shards);
    published.push(("imatrix_unsloth.dat", dat.as_slice()));
    let (_state, base) = serve("unsloth/Qwen3-235B-A22B-GGUF", &published);
    let dir = download(&base, ModelPreset::Qwen235B).await.0.expect("morebits split download completes");
    let read_dtype = |file: &str, tensor: &str| header(&dir.join(file)).unwrap().tensor_infos[tensor].ggml_dtype;
    let stepped = read_dtype("Qwen3-235B-A22B-dQ4im-00002-of-00002.gguf", "blk.4.attn_v.weight");
    assert_eq!(stepped, GgmlDType::Q4K, "blk.4 of 8 must step by the model layer count, not the shard's");
    let c = header(&dir.join("Qwen3-235B-A22B-dQ4im-00001-of-00002.gguf")).unwrap();
    assert!(c.tensor_infos.is_empty() && matches!(c.metadata["synthetic.block_count"], gguf_file::Value::U32(8)));

    // The overlay conversion keeps the fetch-all shape.
    let _guard = sandbox("k3");
    let shards: Vec<Vec<u8>> = (0..2).map(k3_shard).collect();
    let paths = split("UD-IQ1_M/Kimi-K3-UD-IQ1_M", 2);
    let (_state, base) = serve("unsloth/Kimi-K3-GGUF", &files(&paths, &shards));
    let (result, _, details) = download(&base, ModelPreset::KimiK3).await;
    let produced = model_store::gguf_names(&result.expect("overlay download completes"));
    let want = [vec!["Kimi-K3-000-dense-G.gguf".to_string()], split("Kimi-K3-UD-IQ1_M", 2)].concat();
    assert_eq!(produced, want, "the overlay sorts first, the shards pass through under their own names");
    assert!(details.iter().any(|l| l.contains("adoptable by local clone")), "{details:?}");

    // A drafting row's download ends with its draft head derived.
    let _guard = sandbox("derive");
    std::env::remove_var("OSCAR_MODELS_DIR");
    let shard = gguf_bytes(&[], &[("output.weight", &quantised(4, 512, 700, GgmlDType::Q8_0))]);
    let paths = split("Q8_0/Qwen3.6-35B-A3B-Q8_0", 1);
    let (_state, base) = serve("unsloth/Qwen3.6-35B-A3B-GGUF", &files(&paths, std::slice::from_ref(&shard)));
    let (result, _, details) = download(&base, ModelPreset::Qwen35B).await;
    let dir = result.expect("a drafting row's download completes");
    assert!(details.iter().any(|l| l.contains("one source shard")), "a single source is charged once: {details:?}");
    let head = "Qwen3.6-35B-A3B-expertsQ4-000-draft-head.gguf";
    assert_eq!(model_store::gguf_names(&dir), [head, "Qwen3.6-35B-A3B-expertsQ4-00001-of-00001.gguf"]);
    assert!(details.iter().any(|l| l.starts_with(head) && l.ends_with("written")), "{details:?}");
    let c = header(&dir.join(head)).unwrap();
    assert_eq!((c.tensor_infos.len(), c.tensor_infos["output_draft.weight"].ggml_dtype), (1, GgmlDType::Q4K));
}
