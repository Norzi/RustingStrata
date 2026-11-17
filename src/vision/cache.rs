//! Host cache of vision-encoder results.
//!
//! A chat resends its whole history, images included, on every request, so the same image reaches `load_images`
//! again and again (and a conversation with two images re-encodes the first whenever the second arrives). The ViT
//! output depends only on the image bytes and the projector, so the last few results are kept here, keyed by the
//! projector file and the image source. `STRATA_VISION_CACHE=N` sets how many images are kept (0 disables, the
//! default is 4); each entry costs `nx * ny * n_embd * 4` bytes in host memory (~10 MB for a 30x40 grid).

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{LazyLock, Mutex};

struct Entry {
    key: u64,
    grid: (usize, usize),
    embs: Vec<Vec<f32>>,
}

static CACHE: LazyLock<Mutex<VecDeque<Entry>>> = LazyLock::new(|| Mutex::new(VecDeque::new()));

/// How many encoder results to keep (`STRATA_VISION_CACHE`, default 4, 0 disables the cache).
fn room() -> usize {
    static ROOM: LazyLock<usize> = LazyLock::new(|| std::env::var("STRATA_VISION_CACHE").ok().and_then(|v| v.parse().ok()).unwrap_or(4));
    *ROOM
}

/// FNV-1a over the bytes, with the length mixed in so a prefix never matches.
fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes { h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3); }
    (h ^ bytes.len() as u64).wrapping_mul(0x0000_0100_0000_01b3)
}

/// The cache key of an image source (`data:` URL, path or URL) under the projector at `mmproj`: the projector's path
/// and file size, then the source itself (a data URL carries the whole image).
pub fn key(mmproj: &Path, src: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    h = fnv(h, mmproj.to_string_lossy().as_bytes());
    h = fnv(h, &std::fs::metadata(mmproj).map_or(0, |m| m.len()).to_le_bytes());
    fnv(h, src.as_bytes())
}

/// The encoder result for `key`, moved to the most-recently-used end.
pub fn get(key: u64) -> Option<((usize, usize), Vec<Vec<f32>>)> {
    if room() == 0 { return None; }
    let mut c = CACHE.lock().unwrap();
    let i = c.iter().position(|e| e.key == key)?;
    let e = c.remove(i)?;
    let out = (e.grid, e.embs.clone());
    c.push_back(e);
    Some(out)
}

/// Keep `embs` (the rows of a `grid` image) under `key`, dropping the oldest results past the limit.
pub fn put(key: u64, grid: (usize, usize), embs: &[Vec<f32>]) {
    let n = room();
    if n == 0 { return; }
    let mut c = CACHE.lock().unwrap();
    if let Some(i) = c.iter().position(|e| e.key == key) { c.remove(i); }
    c.push_back(Entry { key, grid, embs: embs.to_vec() });
    while c.len() > n { c.pop_front(); }
}

/// Drop every cached result (a model change, or a test).
pub fn clear() { CACHE.lock().unwrap().clear() }
