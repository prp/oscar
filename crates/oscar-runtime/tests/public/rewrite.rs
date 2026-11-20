//! The store's rewrite: ggml's k-quants, the streaming GGUF rewrite on synthetic GGUFs, a conversion against its artifact.

use std::path::{Path, PathBuf};

use candle_core::quantized::{gguf_file, GgmlDType};
use candle_core::Device;
use oscar_models::rows::conversion::Transform;
use oscar_models::rows::ModelPreset;
use oscar_runtime::model_store::conversion::{decide, ImatrixMap};
use oscar_runtime::model_store::gguf_rw::{self, Disposition, Metadata, STORE_ALIGNMENT};
use oscar_runtime::model_store::kquant;

use crate::helpers::{det_floats, fixture, gguf_bytes, quantised, TempDir};

fn f32s(bytes: Vec<u8>) -> Vec<f32> {
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

/// The expected bytes are `fixtures/kquant/input.f32` through ggml's `ggml_quantize_chunk`, plain and imatrix-weighted;
/// row chunking across threads is byte-identical.
#[test]
fn kquants_match_ggml_byte_for_byte() {
    let input = f32s(fixture("fixtures/kquant/input.f32"));
    let imatrix = f32s(fixture("fixtures/kquant/imatrix.f32"));
    for (dtype, name, weights) in [
        (GgmlDType::Q3K, "q3_k", None),
        (GgmlDType::Q4K, "q4_k", None),
        (GgmlDType::Q5K, "q5_k", None),
        (GgmlDType::Q6K, "q6_k", None),
        (GgmlDType::Q3K, "q3_k_imat", Some(imatrix.as_slice())),
        (GgmlDType::Q4K, "q4_k_imat", Some(imatrix.as_slice())),
    ] {
        let ours = kquant::quantize(dtype, &input, 512, weights, 1).unwrap();
        assert_eq!(ours, fixture(&format!("fixtures/kquant/{name}.bin")), "{name} diverges from the ggml reference");
        if let Some(plain) = name.strip_suffix("_imat") {
            assert_ne!(ours, fixture(&format!("fixtures/kquant/{plain}.bin")), "{name}: the imatrix had no effect");
        }
    }
    let whole = kquant::quantize(GgmlDType::Q4K, &input, 512, None, 1).unwrap();
    for threads in [2, 3, 8] {
        let chunked = kquant::quantize(GgmlDType::Q4K, &input, 512, None, threads).unwrap();
        assert_eq!(whole, chunked, "{threads}-way row chunking is not byte-identical");
    }
}

/// An F32, a Q8_0 to requantise and a Q8_0 to keep, beside an output path; the first Q8_0 comes back dequantised.
fn source(tag: &str) -> (TempDir, PathBuf, PathBuf, Vec<f32>) {
    let tmp = TempDir::new(tag);
    let (src, out) = (tmp.0.join("src.gguf"), tmp.0.join("out.gguf"));
    let tb = quantised(8, 512, 11, GgmlDType::Q8_0);
    let names = vec![gguf_file::Value::String("x".into()), gguf_file::Value::String("y".into())];
    let meta = [
        ("synthetic.count", gguf_file::Value::U32(3)),
        ("synthetic.scale", gguf_file::Value::F32(0.25)),
        ("synthetic.names", gguf_file::Value::Array(names)),
    ];
    let tensors = [
        ("a.weight", &quantised(4, 64, 7, GgmlDType::F32)),
        ("blk.0.b.weight", &tb),
        ("blk.0.c.weight", &quantised(2, 256, 13, GgmlDType::Q8_0)),
    ];
    std::fs::write(&src, gguf_bytes(&meta, &tensors)).unwrap();
    let deq_b = tb.dequantize(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    (tmp, src, out, deq_b)
}

fn read_back(path: &Path) -> gguf_file::Content {
    gguf_file::Content::read(&mut std::io::BufReader::new(std::fs::File::open(path).unwrap())).unwrap()
}

fn raw_bytes(path: &Path, name: &str) -> Vec<u8> {
    let mut r = std::io::BufReader::new(std::fs::File::open(path).unwrap());
    let c = gguf_file::Content::read(&mut r).unwrap();
    c.tensor(&mut r, name, &Device::Cpu).unwrap().data().unwrap().into_owned()
}

fn requant_b(name: &str, _: GgmlDType, _: &[usize], _: usize) -> Disposition {
    if name == "blk.0.b.weight" {
        Disposition::Requant(GgmlDType::Q4K)
    } else {
        Disposition::Copy
    }
}

/// The rewrite is aligned, faithful and loadable; a overlay holds only the selected tensors; the imatrix reaches the
/// quantiser, per expert slice when it has one.
#[test]
fn a_requant_rewrite_is_aligned_faithful_and_loadable() {
    let (_tmp, src, out, deq_b) = source("requant");
    let plan = gguf_rw::plan(&[src.clone()], requant_b).unwrap();
    assert_eq!((plan.entries.len(), plan.entries.iter().filter(|e| e.is_requant()).count(), plan.n_layers), (3, 1, 1));
    let written = gguf_rw::execute(&plan, &out, Metadata::Copy, &ImatrixMap::new(), 2, |_, _| {}).unwrap();
    assert_eq!(written, std::fs::metadata(&out).unwrap().len());
    let c = read_back(&out);
    assert_eq!(c.metadata.get("general.alignment").unwrap().to_u32().unwrap(), STORE_ALIGNMENT as u32);
    for (name, info) in &c.tensor_infos {
        assert_eq!((c.tensor_data_offset + info.offset) % STORE_ALIGNMENT, 0, "{name} starts off-alignment");
    }
    // The requantised tensor equals the quantiser run on the dequantised source.
    assert_eq!(raw_bytes(&out, "a.weight"), raw_bytes(&src, "a.weight"));
    assert_eq!(raw_bytes(&out, "blk.0.c.weight"), raw_bytes(&src, "blk.0.c.weight"));
    let expect = kquant::quantize(GgmlDType::Q4K, &deq_b, 512, None, 1).unwrap();
    assert_eq!(raw_bytes(&out, "blk.0.b.weight"), expect);
    assert_eq!(c.metadata.get("synthetic.count").unwrap().to_u32().unwrap(), 3);
    assert_eq!(c.metadata.get("synthetic.names").unwrap().to_vec().unwrap().len(), 2);
    // The overlay.
    let overlay = src.with_file_name("000-overlay.gguf");
    let plan_overlay = gguf_rw::plan(&[src.clone()], |name, dt, dims, n| match requant_b(name, dt, dims, n) {
        d @ Disposition::Requant(_) => d,
        _ => Disposition::Skip,
    })
    .unwrap();
    let arch = vec![("general.architecture".to_string(), gguf_file::Value::String("synthetic".into()))];
    gguf_rw::execute(&plan_overlay, &overlay, Metadata::Replace(arch), &ImatrixMap::new(), 1, |_, _| {}).unwrap();
    let c = read_back(&overlay);
    assert!(c.tensor_infos.len() == 1 && c.tensor_infos.contains_key("blk.0.b.weight"));
    // Per-column weights.
    let per_col: Vec<f32> = det_floats(512, 17).iter().map(|v| v.abs() + 0.05).collect();
    let mut imatrix = ImatrixMap::from([("blk.0.b.weight".to_string(), per_col.clone())]);
    gguf_rw::execute(&plan, &out, Metadata::Copy, &imatrix, 1, |_, _| {}).unwrap();
    let got = raw_bytes(&out, "blk.0.b.weight");
    assert_eq!(got, kquant::quantize(GgmlDType::Q4K, &deq_b, 512, Some(&per_col), 1).unwrap());
    assert_ne!(got, expect);
    // Per-expert weights over the leading dim of 8, one row each.
    let per_expert: Vec<f32> = det_floats(8 * 512, 19).iter().map(|v| v.abs() + 0.05).collect();
    imatrix.insert("blk.0.b.weight".to_string(), per_expert.clone());
    gguf_rw::execute(&plan, &out, Metadata::Copy, &imatrix, 1, |_, _| {}).unwrap();
    let sliced: Vec<u8> = deq_b
        .chunks(512)
        .zip(per_expert.chunks(512))
        .flat_map(|(rows, weights)| kquant::quantize(GgmlDType::Q4K, rows, 512, Some(weights), 1).unwrap())
        .collect();
    assert_eq!(raw_bytes(&out, "blk.0.b.weight"), sliced);
}

#[test]
#[ignore = "reads a real model; pass its directory in OSCAR_BENCH_GGUF_DIR"]
fn k3_rules_predict_the_production_overlay() {
    let dir = PathBuf::from(std::env::var("OSCAR_BENCH_GGUF_DIR").expect("set OSCAR_BENCH_GGUF_DIR"));
    let conversion = &ModelPreset::KimiK3.def().conversion;
    let Transform::Overlay { overlay, .. } = &conversion.transform else {
        panic!("the K3 conversion is a overlay conversion");
    };
    let mut sources: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| Some(e.ok()?.path()))
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or_default();
            name.ends_with(".gguf") && name.contains(conversion.source_select)
        })
        .collect();
    sources.sort();
    assert!(!sources.is_empty(), "no {} shards in {}", conversion.source_select, dir.display());
    let plan = gguf_rw::plan(&sources, |name, _, dims, n| decide(&conversion.transform, name, dims, n)).unwrap();
    let produced = oscar_runtime::plan::header(&dir.join(overlay)).unwrap();
    assert_eq!(plan.entries.len(), produced.tensor_infos.len(), "tensor count differs");
    let mut mismatches = Vec::new();
    for e in &plan.entries {
        match produced.tensor_infos.get(&e.name) {
            None => mismatches.push(format!("{} planned but absent", e.name)),
            Some(info) => {
                let bytes = kquant::quant_bytes(info.ggml_dtype, info.shape.elem_count()) as u64;
                if info.ggml_dtype != e.out_dtype || bytes != e.out_bytes {
                    mismatches.push(format!(
                        "{}: planned {:?}/{} B, artifact holds {:?}/{bytes} B",
                        e.name, e.out_dtype, e.out_bytes, info.ggml_dtype
                    ));
                }
            }
        }
    }
    assert!(mismatches.is_empty(), "{} mismatches:\n{}", mismatches.len(), mismatches.join("\n"));
    println!("{}/{} tensors match the production overlay", plan.entries.len(), plan.entries.len());
}
