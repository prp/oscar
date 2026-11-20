//! The Hugging Face side of `download`: a repo's listing, and fetches with chunk resume and sha256 verification.

use std::io::Read;
use std::path::Path;

use anyhow::Context;
use sha2::{Digest, Sha256};

pub const HF_BASE: &str = "https://huggingface.co";

#[derive(Debug, Clone)]
pub struct RemoteFile {
    pub path: String,
    pub size: u64,
    /// Lowercase hex; a small non-LFS file carries none.
    pub sha256: Option<String>,
}

/// A fetch's progress in bytes: over the link, or already on disk (a resumed chunk, a file that verifies).
#[derive(Debug, Clone, Copy)]
pub enum Progress {
    Fetched(u64),
    Kept(u64),
    /// The file on disk being hashed against the manifest: the link idle.
    Verifying,
}

impl Progress {
    pub fn bytes(self) -> u64 {
        match self {
            Progress::Fetched(b) | Progress::Kept(b) => b,
            Progress::Verifying => 0,
        }
    }
}

fn token() -> Option<String> {
    if let Ok(t) = std::env::var("HF_TOKEN") {
        if !t.is_empty() {
            return Some(t);
        }
    }
    let home = std::env::var_os("HOME")?;
    let t = std::fs::read_to_string(Path::new(&home).join(".cache/huggingface/token")).ok()?;
    let t = t.trim();
    (!t.is_empty()).then(|| t.to_string())
}

fn request(client: &reqwest::Client, url: &str) -> reqwest::RequestBuilder {
    let req = client.get(url);
    match token() {
        Some(t) => req.bearer_auth(t),
        None => req,
    }
}

/// Every file in the repo: the tree endpoint, following its pagination.
pub async fn manifest(client: &reqwest::Client, base: &str, repo: &str) -> anyhow::Result<Vec<RemoteFile>> {
    let mut url = format!("{base}/api/models/{repo}/tree/main?recursive=true");
    let mut files = Vec::new();
    loop {
        let resp = request(client, &url).send().await?;
        anyhow::ensure!(resp.status().is_success(), "{repo}: listing failed with {} for {url}", resp.status());
        let next = resp.headers().get(reqwest::header::LINK).and_then(|l| l.to_str().ok()).and_then(next_link);
        let entries: serde_json::Value = resp.json().await?;
        for e in entries.as_array().into_iter().flatten() {
            if e["type"].as_str() != Some("file") {
                continue;
            }
            let Some(path) = e["path"].as_str() else {
                continue;
            };
            files.push(RemoteFile {
                path: path.to_string(),
                size: e["lfs"]["size"].as_u64().or(e["size"].as_u64()).unwrap_or(0),
                sha256: e["lfs"]["oid"].as_str().map(str::to_string),
            });
        }
        match next {
            Some(n) => url = n,
            None => return Ok(files),
        }
    }
}

fn next_link(header: &str) -> Option<String> {
    header.split(',').find_map(|part| {
        let (url, rel) = part.split_once(';')?;
        rel.contains("rel=\"next\"").then(|| url.trim().trim_start_matches('<').trim_end_matches('>').to_string())
    })
}

fn sha256_of(path: &Path) -> anyhow::Result<String> {
    let mut hasher = Sha256::new();
    let mut f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn verified(path: &Path, file: &RemoteFile) -> anyhow::Result<bool> {
    let Ok(m) = std::fs::metadata(path) else {
        return Ok(false);
    };
    if m.len() != file.size {
        return Ok(false);
    }
    match &file.sha256 {
        Some(want) => Ok(sha256_of(path)? == *want),
        None => Ok(true),
    }
}

/// Unverified, as the fetch proper verifies; truncated in case the server ignores the range.
pub async fn fetch_prefix(
    client: &reqwest::Client, base: &str, repo: &str, file: &RemoteFile, len: u64, dest: &Path,
) -> anyhow::Result<()> {
    let want = len.min(file.size);
    let url = format!("{base}/{repo}/resolve/main/{}", file.path);
    let mut resp = request(client, &url)
        .header(reqwest::header::RANGE, format!("bytes=0-{}", want.saturating_sub(1)))
        .send()
        .await?
        .error_for_status()
        .with_context(|| format!("ranged header fetch of {}", file.path))?;
    let mut out: Vec<u8> = Vec::with_capacity(want as usize);
    // A cut stream keeps what arrived: the caller asks for more until the header parses, failing loudly there.
    while let Ok(Some(chunk)) = resp.chunk().await {
        out.extend_from_slice(&chunk);
        if out.len() as u64 >= want {
            out.truncate(want as usize);
            break;
        }
    }
    std::fs::write(dest, &out)?;
    Ok(())
}

/// A copy of `file` in `candidates` that verifies, cloned in place of a fetch.
pub fn adoptable(candidates: &[std::path::PathBuf], file: &RemoteFile) -> Option<std::path::PathBuf> {
    let base = Path::new(&file.path).file_name()?;
    candidates.iter().map(|dir| dir.join(base)).find(|cand| verified(cand, file).unwrap_or(false))
}

/// A clone where the volume can, else a copy.
pub fn clone_file(src: &Path, dest: &Path) -> std::io::Result<()> {
    let _ = std::fs::remove_file(dest);
    let s = std::ffi::CString::new(src.as_os_str().as_encoded_bytes())?;
    let d = std::ffi::CString::new(dest.as_os_str().as_encoded_bytes())?;
    // SAFETY: two valid C paths; clonefile touches nothing else.
    if unsafe { libc::clonefile(s.as_ptr(), d.as_ptr(), 0) } == 0 {
        return Ok(());
    }
    std::fs::copy(src, dest).map(|_| ())
}

/// The unit of work and of resume: a lost stream re-fetches little, and no file's tail waits on one slow stream.
const CHUNK_BYTES: u64 = 10 << 20;

/// Hugging Face caps a stream at 2–5 MB/s: 32 streams cover a 600 Mbps link near the cap's floor.
const FETCH_WORKERS: usize = 32;

/// Per chunk; a transient failure retries on a doubling backoff from [`CHUNK_BACKOFF`].
const CHUNK_ATTEMPTS: usize = 4;
const CHUNK_BACKOFF: std::time::Duration = std::time::Duration::from_millis(500);

/// Resumes an earlier attempt's chunks and keeps no bytes failing the sha256; a server ignoring ranges gets one chunk.
pub async fn fetch(
    client: &reqwest::Client, base: &str, repo: &str, file: &RemoteFile, dest: &Path,
    mut progress: impl FnMut(Progress),
) -> anyhow::Result<()> {
    match fetch_ranged(client, base, repo, file, dest, FETCH_WORKERS, CHUNK_BYTES, &mut progress).await {
        Err(e) if e.is::<RangeUnsupported>() => {
            fetch_ranged(client, base, repo, file, dest, 1, file.size, &mut progress).await
        }
        done => done,
    }
}

#[derive(Debug, thiserror::Error)]
#[error("server ignores Range requests")]
struct RangeUnsupported;

#[derive(Debug, thiserror::Error)]
#[error("chunk fetch failed with {0}")]
struct HttpStatus(reqwest::StatusCode);

/// Drops, short bodies, 5xx and 429 are transient; an ignored range and the other 4xx are facts, not weather.
fn transient(e: &anyhow::Error) -> bool {
    if e.is::<RangeUnsupported>() {
        return false;
    }
    match e.downcast_ref::<HttpStatus>() {
        Some(HttpStatus(s)) => s.is_server_error() || *s == reqwest::StatusCode::TOO_MANY_REQUESTS,
        None => true,
    }
}

/// One chunk, retried on a doubling backoff while its failure is transient; `whole` when it is the whole file.
async fn fetch_chunk(
    client: &reqwest::Client, url: &str, out: &std::fs::File, (start, end): (u64, u64), whole: bool,
) -> anyhow::Result<()> {
    let mut backoff = CHUNK_BACKOFF;
    for attempt in 1.. {
        let got = async {
            let range = format!("bytes={start}-{}", end - 1);
            let mut resp = request(client, url).header(reqwest::header::RANGE, range).send().await?;
            match resp.status() {
                reqwest::StatusCode::PARTIAL_CONTENT => {}
                // A 200 is the whole file: what a whole-file chunk asked for.
                s if s.is_success() && whole => {}
                s if s.is_success() => return Err(RangeUnsupported.into()),
                s => return Err(HttpStatus(s).into()),
            }
            let mut at = start;
            while let Some(chunk) = resp.chunk().await? {
                let take = chunk.len().min((end - at) as usize);
                use std::os::unix::fs::FileExt;
                out.write_at(&chunk[..take], at)?;
                at += take as u64;
                if at >= end {
                    break;
                }
            }
            anyhow::ensure!(at == end, "chunk ended at {at}, wanted {end}");
            Ok(())
        }
        .await;
        match got {
            Err(e) if attempt < CHUNK_ATTEMPTS && transient(&e) => {
                tokio::time::sleep(backoff).await;
                backoff *= 2;
            }
            done => {
                return done.with_context(|| format!("chunk at {start} (attempt {attempt}); a re-run resumes here"))
            }
        }
    }
    unreachable!("the attempts end in a return")
}

/// `dest` with `suffix` appended: `.ok` holds the manifest's sha once the file hashed to it, so a resume keeps the file
/// without hashing again; `.fetch.json` the chunk map.
fn beside(dest: &Path, suffix: &str) -> std::path::PathBuf {
    let mut p = dest.as_os_str().to_owned();
    p.push(suffix);
    std::path::PathBuf::from(p)
}

/// The done flags are unsynced: a crash may overstate them, which the final sha256 catches.
fn load_state(dest: &Path, file: &RemoteFile, chunks: usize) -> Option<Vec<bool>> {
    let text = std::fs::read_to_string(beside(dest, ".fetch.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let done: Vec<bool> = v["done"].as_str()?.bytes().map(|b| b == b'1').collect();
    (v["size"].as_u64() == Some(file.size) && v["sha256"].as_str() == file.sha256.as_deref() && done.len() == chunks)
        .then_some(done)
}

fn save_state(dest: &Path, file: &RemoteFile, done: &[std::sync::atomic::AtomicBool]) {
    let flags: String =
        done.iter().map(|d| if d.load(std::sync::atomic::Ordering::Relaxed) { '1' } else { '0' }).collect();
    let v = serde_json::json!({ "size": file.size, "sha256": file.sha256, "done": flags });
    let tmp = beside(dest, ".fetch.json").with_extension("json.tmp");
    if std::fs::write(&tmp, v.to_string()).is_ok() {
        let _ = std::fs::rename(&tmp, beside(dest, ".fetch.json"));
    }
}

/// Complete when the whole file hashes to the manifest's sha256.
async fn fetch_ranged(
    client: &reqwest::Client, base: &str, repo: &str, file: &RemoteFile, dest: &Path, workers: usize, chunk_bytes: u64,
    mut progress: impl FnMut(Progress),
) -> anyhow::Result<()> {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    // `dest` is preallocated, so its length proves nothing: the shortcut needs the sha and no resume state. A file
    // this fetch verified before carries its sha beside it, so a resume does not hash it again.
    if !beside(dest, ".fetch.json").exists()
        && file.sha256.is_some()
        && std::fs::metadata(dest).is_ok_and(|m| m.len() == file.size)
    {
        if std::fs::read_to_string(beside(dest, ".ok")).ok().as_deref() == file.sha256.as_deref() {
            progress(Progress::Kept(file.size));
            return Ok(());
        }
        progress(Progress::Verifying);
        if verified(dest, file)? {
            let _ = std::fs::write(beside(dest, ".ok"), file.sha256.as_deref().unwrap_or_default());
            progress(Progress::Kept(file.size));
            return Ok(());
        }
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let chunk_bytes = chunk_bytes.max(1);
    let chunks = file.size.div_ceil(chunk_bytes) as usize;
    let chunk_len = |i: usize| file.size.min((i as u64 + 1) * chunk_bytes) - i as u64 * chunk_bytes;

    // Resume only what the resume state vouches for.
    let done = load_state(dest, file, chunks).filter(|_| std::fs::metadata(dest).is_ok_and(|m| m.len() == file.size));
    if done.is_none() {
        let _ = std::fs::remove_file(dest);
    }
    let done: std::sync::Arc<Vec<AtomicBool>> =
        std::sync::Arc::new((0..chunks).map(|i| AtomicBool::new(done.as_ref().is_some_and(|d| d[i]))).collect());
    let out = std::sync::Arc::new(std::fs::OpenOptions::new().create(true).write(true).read(true).open(dest)?);
    out.set_len(file.size)?;
    let resumed: u64 = (0..chunks).filter(|&i| done[i].load(Ordering::Relaxed)).map(chunk_len).sum();
    if resumed > 0 {
        progress(Progress::Kept(resumed));
    }

    let url = format!("{base}/{repo}/resolve/main/{}", file.path);
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<u64>();
    let cursor = std::sync::Arc::new(AtomicUsize::new(0));
    let mut tasks = tokio::task::JoinSet::new();
    for _ in 0..workers.max(1).min(chunks) {
        let (client, url, out, tx, done, cursor) =
            (client.clone(), url.clone(), out.clone(), tx.clone(), done.clone(), cursor.clone());
        let size = file.size;
        tasks.spawn(async move {
            loop {
                let i = cursor.fetch_add(1, Ordering::Relaxed);
                if i >= done.len() {
                    return anyhow::Ok(());
                }
                if done[i].load(Ordering::Relaxed) {
                    continue;
                }
                let start = i as u64 * chunk_bytes;
                let end = size.min(start + chunk_bytes);
                fetch_chunk(&client, &url, &out, (start, end), start == 0 && end == size).await?;
                done[i].store(true, Ordering::Relaxed);
                // Per completed chunk, so a retry never counts twice.
                let _ = tx.send(end - start);
            }
        });
    }
    drop(tx);

    // Progress drained and state saved each second; a failed chunk fails the fetch once the others stop.
    let mut failure: Option<anyhow::Error> = None;
    let mut tick = tokio::time::interval(std::time::Duration::from_secs(1));
    loop {
        tokio::select! {
            delta = rx.recv() => match delta {
                Some(d) => progress(Progress::Fetched(d)),
                None => break,
            },
            joined = tasks.join_next() => match joined {
                Some(res) => {
                    if let Err(e) = res.map_err(anyhow::Error::from).and_then(|r| r) {
                        failure.get_or_insert(e);
                        tasks.abort_all();
                    }
                }
                None => { if rx.is_empty() { break; } }
            },
            _ = tick.tick() => save_state(dest, file, &done),
        }
    }
    while let Some(res) = tasks.join_next().await {
        if let Err(e) = res.map_err(anyhow::Error::from).and_then(|r| r) {
            failure.get_or_insert(e);
        }
    }
    save_state(dest, file, &done);
    if let Some(e) = failure {
        if e.is::<RangeUnsupported>() {
            let _ = std::fs::remove_file(dest);
            let _ = std::fs::remove_file(beside(dest, ".fetch.json"));
        }
        return Err(e).with_context(|| format!("{}: ranged fetch", file.path));
    }

    out.sync_all()?;
    if let Some(want) = &file.sha256 {
        let got = sha256_of(dest)?;
        if got != *want {
            std::fs::remove_file(dest)?;
            let _ = std::fs::remove_file(beside(dest, ".fetch.json"));
            anyhow::bail!("{}: sha256 mismatch (got {got}, manifest says {want})", file.path);
        }
    }
    let _ = std::fs::remove_file(beside(dest, ".fetch.json"));
    let _ = std::fs::write(beside(dest, ".ok"), file.sha256.as_deref().unwrap_or_default());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_store::fixtures::{det_bytes, parse_range, serve, sha_hex, StubState, TempDir};
    use std::collections::BTreeSet;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    /// Where the stub serves t/r's one file.
    const AT: &str = "/t/r/resolve/main/Q8_0/model.gguf";

    /// `body` served as t/r's one file, as its listing gives it, and a destination in a fresh directory.
    async fn one(
        tag: &str, body: &[u8],
    ) -> (TempDir, Arc<Mutex<StubState>>, String, reqwest::Client, RemoteFile, PathBuf) {
        let tmp = TempDir::new(tag);
        let (state, base) = serve("t/r", &[("Q8_0/model.gguf", body)]);
        let client = reqwest::Client::new();
        let files = manifest(&client, &base, "t/r").await.unwrap();
        assert_eq!(files.len(), 1, "{files:?}");
        let dest = tmp.0.join("model.gguf");
        (tmp, state, base, client, files[0].clone(), dest)
    }

    /// The range starts the file's requests asked for, from the `from`th request on.
    fn range_starts(state: &Mutex<StubState>, from: usize, len: usize) -> Vec<usize> {
        let requests = state.lock().unwrap().requests[from..].to_vec();
        let ranges = requests.iter().filter(|(p, _)| p.ends_with("model.gguf")).filter_map(|(_, r)| r.as_deref());
        ranges.filter_map(|r| parse_range(r, len)).map(|(a, _)| a).collect()
    }

    /// A cut transfer fails and a whole one verifies; a transient failure backs off and recovers; corrupt bytes are
    /// refused and nothing kept; a server ignoring ranges on a file of several chunks (so the first attempt asks for
    /// part of it) falls back to the single stream. Verified local bytes are adopted without a fetch.
    #[tokio::test]
    async fn a_fetch_asks_each_aligned_chunk_once_and_a_faulty_server_is_retried_refused_or_streamed_whole() {
        let shard = det_bytes(200_000, 3);
        let (_tmp, state, base, client, file, dest) = one("resume", &shard).await;
        assert_eq!((file.size, file.sha256.as_deref()), (shard.len() as u64, Some(sha_hex(&shard).as_str())));
        // Every connection drops mid-body.
        state.lock().unwrap().truncate_at.insert(AT.into(), 80_000);
        let err = fetch(&client, &base, "t/r", &file, &dest, |_| {}).await;
        assert!(err.is_err(), "a dropped transfer must not verify");
        state.lock().unwrap().truncate_at.clear();
        fetch(&client, &base, "t/r", &file, &dest, |_| {}).await.unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), shard);
        let body = det_bytes(500_000, 47);
        let (tmp, state, base, client, file, dest) = one("ranged-backoff", &body).await;
        state.lock().unwrap().fail_first.insert(AT.into(), 2);
        fetch_ranged(&client, &base, "t/r", &file, &dest, 2, 100_000, |_| {}).await.unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert!(!tmp.0.join("model.gguf.fetch.json").exists(), "state must clear on success");
        let requests = state.lock().unwrap().requests.iter().filter(|(p, _)| p.ends_with("model.gguf")).count();
        assert_eq!(requests, 7, "expected 5 chunks plus 2 retries");
        // The manifest describes `body`; the stub serves it with a byte flipped.
        let body = det_bytes(400_000, 41);
        let (tmp, state, base, client, file, dest) = one("ranged-corrupt", &body).await;
        let mut wrong = body.clone();
        wrong[123_456] ^= 0xff;
        state.lock().unwrap().files.insert(AT.into(), wrong);
        let err = fetch_ranged(&client, &base, "t/r", &file, &dest, 4, 100_000, |_| {}).await;
        assert!(err.unwrap_err().to_string().contains("sha256 mismatch"));
        assert!(!dest.exists() && !tmp.0.join("model.gguf.fetch.json").exists(), "corrupt bytes or state kept");
        let body = det_bytes(16 << 20, 55);
        let (tmp, state, base, client, file, dest) = one("ranged-fallback", &body).await;
        state.lock().unwrap().ignore_range = true;
        fetch(&client, &base, "t/r", &file, &dest, |_| {}).await.unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert!(!tmp.0.join("model.gguf.fetch.json").exists());
        // The adoption.
        let tmp = TempDir::new("adopt");
        let shard = det_bytes(50_000, 9);
        let local_dir = tmp.0.join("existing");
        std::fs::create_dir_all(&local_dir).unwrap();
        std::fs::write(local_dir.join("model.gguf"), &shard).unwrap();
        let file =
            RemoteFile { path: "UD/model.gguf".to_string(), size: shard.len() as u64, sha256: Some(sha_hex(&shard)) };
        let local = adoptable(&[local_dir.clone()], &file).expect("verified local bytes are adoptable");
        let dest = tmp.0.join("staged.gguf");
        clone_file(&local, &dest).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), shard);
        for other in
            [RemoteFile { size: 10, ..file.clone() }, RemoteFile { sha256: Some(sha_hex(b"different")), ..file }]
        {
            assert!(adoptable(&[local_dir.clone()], &other).is_none(), "{other:?}");
        }
        // Each aligned chunk asked once, a verified file kept whole on a resume; a failing chunk's resume refetches it.
        let body = det_bytes(1_000_000, 21);
        let (tmp, state, base, client, file, dest) = one("ranged", &body).await;
        let mut got = 0u64;
        fetch_ranged(&client, &base, "t/r", &file, &dest, 4, 100_000, |p| got += p.bytes()).await.unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert_eq!(got, body.len() as u64);
        assert!(!tmp.0.join("model.gguf.fetch.json").exists(), "state must clear on success");
        let starts = range_starts(&state, 0, body.len());
        assert_eq!(starts.len(), 10, "expected 10 chunk requests: {starts:?}");
        assert_eq!(starts.iter().collect::<BTreeSet<_>>().len(), 10, "a chunk was fetched twice: {starts:?}");
        assert!(starts.iter().all(|a| a % 100_000 == 0), "unaligned chunk start: {starts:?}");
        // Verified once, the file is kept whole on a resume: no request, no hashing.
        let before = state.lock().unwrap().requests.len();
        let mut kept = 0u64;
        fetch_ranged(&client, &base, "t/r", &file, &dest, 4, 100_000, |p| match p {
            Progress::Kept(b) => kept += b,
            Progress::Fetched(_) | Progress::Verifying => panic!("a verified file is kept as it is: {p:?}"),
        })
        .await
        .unwrap();
        assert_eq!(kept, body.len() as u64);
        assert_eq!(state.lock().unwrap().requests.len(), before, "a kept file asks the server nothing");
        // One chunk's start answers 500 persistently; the other chunks complete.
        let body = det_bytes(800_000, 33);
        let (tmp, state, base, client, file, dest) = one("ranged-resume", &body).await;
        state.lock().unwrap().fail_starts.insert(AT.into(), vec![300_000]);
        let err = fetch_ranged(&client, &base, "t/r", &file, &dest, 2, 100_000, |_| {}).await;
        assert!(err.is_err(), "a persistently failing chunk must fail the fetch");
        assert!(tmp.0.join("model.gguf.fetch.json").exists(), "state must persist on failure");
        assert_eq!(std::fs::metadata(&dest).unwrap().len(), body.len() as u64, "preallocated");
        let failed_tries = range_starts(&state, 0, body.len()).iter().filter(|&&a| a == 300_000).count();
        assert_eq!(failed_tries, 4, "the bounded backoff schedule is 4 attempts");
        state.lock().unwrap().fail_starts.clear();
        let before = state.lock().unwrap().requests.len();
        // The resumed chunks are reported as kept, so a rate read from the fetched bytes alone stays the link's.
        let (mut kept, mut fetched) = (0u64, 0u64);
        let progress = |p| match p {
            Progress::Kept(b) => kept += b,
            Progress::Fetched(b) => fetched += b,
            Progress::Verifying => {}
        };
        fetch_ranged(&client, &base, "t/r", &file, &dest, 2, 100_000, progress).await.unwrap();
        assert_eq!((kept, fetched), (700_000, 100_000), "kept the seven chunks on disk, fetched the one missing");
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        let resumed = range_starts(&state, before, body.len());
        assert_eq!(resumed, [300_000], "resume must refetch exactly the missing chunk");
    }
}
