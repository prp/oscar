//! Fixtures on files: temporary directories, the data under `tests/`, a local Hugging Face stub, synthetic GGUFs;
//! the unit tests and the test binary include it, each using a part.
#![allow(dead_code)]

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use sha2::{Digest, Sha256};

pub struct TempDir(pub std::path::PathBuf);

impl TempDir {
    pub fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("oscar-fetch-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A data file under `tests/`.
pub fn fixture(path: &str) -> Vec<u8> {
    std::fs::read(format!("{}/tests/{path}", env!("CARGO_MANIFEST_DIR"))).unwrap_or_else(|e| panic!("{path}: {e}"))
}

#[derive(Default)]
pub struct StubState {
    pub files: HashMap<String, Vec<u8>>,
    /// Per path, the body cut at this length while its full length is advertised: a drop mid-transfer.
    pub truncate_at: HashMap<String, usize>,
    /// Per path, how many of the next requests answer 500.
    pub fail_first: HashMap<String, usize>,
    /// Per path, the range starts that always answer 500.
    pub fail_starts: HashMap<String, Vec<usize>>,
    /// (path, range header) per request.
    pub requests: Vec<(String, Option<String>)>,
    pub manifest: String,
    pub ignore_range: bool,
}

/// A stub serving `files` under `repo` and a manifest listing them; the state is the test's to change before a request.
pub fn serve(repo: &str, files: &[(&str, &[u8])]) -> (Arc<Mutex<StubState>>, String) {
    let listed: Vec<_> = files.iter().map(|(p, b)| (*p, b.len() as u64, sha_hex(b))).collect();
    let state = Arc::new(Mutex::new(StubState {
        manifest: manifest_json(&listed),
        files: files.iter().map(|(p, b)| (format!("/{repo}/resolve/main/{p}"), b.to_vec())).collect(),
        ..Default::default()
    }));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let served = Arc::clone(&state);
    std::thread::spawn(move || {
        for mut stream in listener.incoming().map_while(Result::ok) {
            let state = Arc::clone(&served);
            std::thread::spawn(move || {
                let (mut buf, mut req) = ([0u8; 8192], Vec::new());
                while !req.windows(4).any(|w| w == b"\r\n\r\n") {
                    match stream.read(&mut buf) {
                        Ok(n) if n > 0 => req.extend_from_slice(&buf[..n]),
                        _ => return,
                    }
                }
                let text = String::from_utf8_lossy(&req);
                let path = text.split_whitespace().nth(1).unwrap_or("/");
                let header = |name: &str| text.lines().find(|l| l.to_ascii_lowercase().starts_with(name));
                let range = header("range:").map(|l| l.split_once(':').unwrap().1.trim());
                let (status, extra, advertised, body) = respond(&mut state.lock().unwrap(), path, range);
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status}\r\n{extra}Content-Length: {advertised}\r\nConnection: close\r\n\r\n"
                );
                let _ = stream.write_all(&body);
            });
        }
    });
    (state, base)
}

/// Status, extra headers, the advertised length and the body sent, which a cut leaves short of it.
fn respond(s: &mut StubState, path: &str, range: Option<&str>) -> (&'static str, String, usize, Vec<u8>) {
    s.requests.push((path.to_string(), range.map(str::to_string)));
    let start = range.and_then(|r| parse_range(r, usize::MAX)).map(|(a, _)| a);
    let failing = match s.fail_first.get_mut(path) {
        Some(n) if *n > 0 => {
            *n -= 1;
            true
        }
        _ => s.fail_starts.get(path).zip(start).is_some_and(|(starts, a)| starts.contains(&a)),
    };
    let whole = |status: &'static str, body: Vec<u8>| (status, String::new(), body.len(), body);
    if failing {
        return whole("500 Internal Server Error", b"boom".to_vec());
    }
    if path.starts_with("/api/") {
        return whole("200 OK", s.manifest.clone().into_bytes());
    }
    let Some(full) = s.files.get(path) else {
        return whole("404 Not Found", b"missing".to_vec());
    };
    let (status, extra, mut body) = match range.filter(|_| !s.ignore_range).and_then(|r| parse_range(r, full.len())) {
        Some((from, to)) => {
            let extra = format!("Content-Range: bytes {from}-{}/{}\r\n", to - 1, full.len());
            ("206 Partial Content", extra, full[from..to].to_vec())
        }
        None => ("200 OK", String::new(), full.clone()),
    };
    let advertised = body.len();
    body.truncate(s.truncate_at.get(path).copied().unwrap_or(advertised));
    (status, extra, advertised, body)
}

/// "bytes=a-b" or "bytes=a-" -> (a, exclusive end).
pub fn parse_range(r: &str, len: usize) -> Option<(usize, usize)> {
    let r = r.strip_prefix("bytes=")?;
    let (a, b) = r.split_once('-')?;
    let start: usize = a.parse().ok()?;
    let end = if b.is_empty() { len } else { b.parse::<usize>().ok()? + 1 };
    (start <= len && end <= len && start < end).then_some((start, end))
}

/// The tree listing of `(path, size, sha256)` entries, sizes as declared.
pub fn manifest_json(entries: &[(&str, u64, String)]) -> String {
    let items: Vec<String> = entries
        .iter()
        .map(|(p, size, sha)| {
            format!(r#"{{"type":"file","path":"{p}","size":{size},"lfs":{{"oid":"{sha}","size":{size}}}}}"#)
        })
        .collect();
    format!("[{}]", items.join(","))
}

pub fn sha_hex(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

fn lcg(mut seed: u32) -> impl Iterator<Item = u32> {
    std::iter::repeat_with(move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        seed
    })
}

pub fn det_bytes(n: usize, seed: u32) -> Vec<u8> {
    lcg(seed).take(n).map(|s| (s >> 16) as u8).collect()
}

pub fn det_floats(n: usize, seed: u32) -> Vec<f32> {
    lcg(seed).take(n).map(|s| ((s >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0).collect()
}

pub fn quantised(rows: usize, cols: usize, seed: u32, dtype: GgmlDType) -> QTensor {
    QTensor::quantize(&Tensor::from_vec(det_floats(rows * cols, seed), (rows, cols), &Device::Cpu).unwrap(), dtype)
        .unwrap()
}

/// `general.architecture = synthetic`, then `meta` and `tensors`.
pub fn gguf_bytes(meta: &[(&str, gguf_file::Value)], tensors: &[(&str, &QTensor)]) -> Vec<u8> {
    let arch = gguf_file::Value::String("synthetic".into());
    let meta: Vec<(&str, &gguf_file::Value)> =
        std::iter::once(("general.architecture", &arch)).chain(meta.iter().map(|(k, v)| (*k, v))).collect();
    let mut buf = std::io::Cursor::new(Vec::new());
    gguf_file::write(&mut buf, &meta, tensors).unwrap();
    buf.into_inner()
}
