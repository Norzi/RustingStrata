//! GPU (CUDA) offload of the vision encoder: the Qwen3-VL matrix multiplies (patch convs, the ViT's QKV / attention
//! output / FFN and the `mm.0`/`mm.2` merger) and the ViT's bidirectional attention run on the device. The rest of the
//! encoder stays on the CPU: the patch gather, LayerNorm, RoPE, the GELU and the bias adds.
//!
//! One weight matrix is resident at a time, so the projector's ~0.9 GB of F16 weights never has to fit in VRAM
//! beside a large text model: each multiply uploads its matrix as F32 (rows zero-padded to the GEMM's 64-deep K
//! step, which is all `Backend::gemm_dense` asks for), runs the GEMM and drops it again. Under `--gpu` the encoder
//! uses this path automatically; `STRATA_VISION_CPU=1` keeps it on the CPU. The device and host buffers are reused
//! across the 27 layers (`Scratch`): allocating and zeroing 20-85 MB three times per multiply cost more than some
//! of the multiplies.

use crate::backend::{self, Backend, Buf, GpuTensor};
use crate::quant::GgmlType;
use crate::vision::image::Rgb;
use crate::vision::mmproj::ProjectorKind;
use crate::vision::projector::{MatMul, VisionProjector};
use crate::weights::Tensor;
use anyhow::Context;
use half::f16;
use rayon::prelude::*;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

/// Free VRAM below which the encoder stays on the CPU. One image's buffers take roughly 100-250 MB.
const MIN_FREE_VRAM: usize = 512 << 20;

/// Device memory and host staging reused across the encoder's calls. Every slot grows on demand and stays for the
/// rest of the image.
#[derive(Default)]
struct Scratch {
    /// The one weight matrix resident at a time (F32, rows padded to the GEMM's 64-deep K step).
    w: Option<Buf>,
    wcap: usize,
    /// The GEMM's input rows (padded the same way) and its result.
    x: Option<Buf>,
    xcap: usize,
    y: Option<Buf>,
    ycap: usize,
    /// The attention path's q, k, v and output ([n][nh][hd] floats each).
    aq: Option<Buf>,
    aqcap: usize,
    ak: Option<Buf>,
    akcap: usize,
    av: Option<Buf>,
    avcap: usize,
    ay: Option<Buf>,
    aycap: usize,
    /// Host staging: padded weights, padded x, the read-back result.
    wh: Vec<f32>,
    xh: Vec<f32>,
    yh: Vec<f32>,
}

pub struct VisionGpu {
    b: Box<dyn Backend>,
    scratch: Mutex<Scratch>,
    /// GEMMs run, and nanoseconds in each phase of the encoder, printed by `encode_image` under
    /// `STRATA_VISION_DEBUG`: weight conversion, upload, GEMM, read-back, attention.
    gemms: AtomicU64,
    conv_ns: AtomicU64,
    up_ns: AtomicU64,
    gemm_ns: AtomicU64,
    down_ns: AtomicU64,
    attn_ns: AtomicU64,
}

impl VisionGpu {
    /// The CUDA backend for `device`, or None for the CPU path: no device, `STRATA_VISION_CPU`, a non-CUDA
    /// backend, or less than `MIN_FREE_VRAM` free.
    pub fn open(device: Option<usize>) -> Option<VisionGpu> {
        let dev = device?;
        if std::env::var_os("STRATA_VISION_CPU").is_some() { return None; }
        let b = match backend::open(None, dev) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("image: no GPU for the vision encoder ({e:#}); running it on the CPU");
                return None;
            }
        };
        if b.name() != "cuda" {
            eprintln!("image: the vision encoder needs CUDA ({} is {}); running it on the CPU", b.name(), b.device());
            return None;
        }
        match b.mem_info() {
            Ok((free, _)) if free < MIN_FREE_VRAM => {
                eprintln!("image: {} MB of VRAM free, too little for the vision encoder; running it on the CPU", free >> 20);
                None
            }
            Ok(_) => Some(VisionGpu {
                b,
                scratch: Mutex::new(Scratch::default()),
                gemms: AtomicU64::new(0),
                conv_ns: AtomicU64::new(0),
                up_ns: AtomicU64::new(0),
                gemm_ns: AtomicU64::new(0),
                down_ns: AtomicU64::new(0),
                attn_ns: AtomicU64::new(0),
            }),
            Err(e) => {
                eprintln!("image: {e:#}; running the vision encoder on the CPU");
                None
            }
        }
    }

    fn lock(&self) -> MutexGuard<'_, Scratch> { self.scratch.lock().unwrap_or_else(|e| e.into_inner()) }

    /// y[t * w.rows + r] = w[r] · x[t * w.cols..] on the device.
    fn gemm(&self, w: &Tensor, x: &[f32], n: usize, y: &mut [f32], add: bool) -> anyhow::Result<()> {
        anyhow::ensure!(w.cols > 0 && x.len() == n * w.cols, "vision gemm: x is {} values, not {n}x{}", x.len(), w.cols);
        anyhow::ensure!(y.len() == n * w.rows, "vision gemm: y is {} values, not {n}x{}", y.len(), w.rows);
        // The GEMM's K step is 64 columns: pad the weight rows and the x rows with zeros to reach it.
        let cols = w.cols;
        let pw = cols.div_ceil(64) * 64;
        let (wbytes, xbytes, ybytes) = (w.rows * pw * 4, n * pw * 4, y.len() * 4);

        let mut s = self.lock();
        let s = &mut *s;

        let t = Instant::now();
        s.wh.resize(w.rows * pw, 0.0);
        s.wh.par_chunks_exact_mut(pw).enumerate().for_each(|(r, dst)| {
            weight_row_into(w, r, &mut dst[..cols]);
            dst[cols..].fill(0.0);
        });
        s.xh.resize(n * pw, 0.0);
        if pw == cols {
            s.xh[..n * cols].copy_from_slice(x);
        } else {
            s.xh.par_chunks_exact_mut(pw).enumerate().for_each(|(t, dst)| {
                dst[..cols].copy_from_slice(&x[t * cols..(t + 1) * cols]);
                dst[cols..].fill(0.0);
            });
        }
        bump(&self.conv_ns, t);

        let wb = reserve(&*self.b, &mut s.w, &mut s.wcap, wbytes)?;
        let xb = reserve(&*self.b, &mut s.x, &mut s.xcap, xbytes)?;
        let yb = reserve(&*self.b, &mut s.y, &mut s.ycap, ybytes)?;

        let t = Instant::now();
        self.b.put(&s.wh, wb, 0)?;
        self.b.put(&s.xh, xb, 0)?;
        bump(&self.up_ns, t);

        let t = Instant::now();
        let wt = GpuTensor::new(GgmlType::F32, w.rows, pw, alias(wb));
        self.b.gemm_dense(&wt, n, xb.addr(), pw, yb.addr(), w.rows, 0)?;
        bump(&self.gemm_ns, t);

        let t = Instant::now();
        s.yh.resize(y.len(), 0.0);
        self.b.get_into(yb, 0, &mut s.yh)?;
        bump(&self.down_ns, t);

        if add {
            for (a, b) in y.iter_mut().zip(&s.yh) { *a += *b; }
        } else {
            y.copy_from_slice(&s.yh);
        }
        Ok(())
    }

    /// The ViT's bidirectional attention on the device, into `out`; false when the backend has no kernel for this
    /// `hd` (the caller then runs it on the CPU).
    fn attention_into(&self, q: &[f32], k: &[f32], v: &[f32], n: usize, nh: usize, hd: usize, scale: f32,
                      out: &mut [f32]) -> anyhow::Result<bool> {
        if hd == 0 || hd > 96 { return Ok(false); }
        let len = n * nh * hd;
        anyhow::ensure!(out.len() == len && q.len() == len && k.len() == len && v.len() == len,
            "vision attention: {n}x{nh}x{hd} needs {len} q, k, v and out values, got {} / {} / {} / {}",
            q.len(), k.len(), v.len(), out.len());
        let bytes = len * 4;
        let t = Instant::now();

        let mut s = self.lock();
        let s = &mut *s;
        let qb = reserve(&*self.b, &mut s.aq, &mut s.aqcap, bytes)?;
        let kb = reserve(&*self.b, &mut s.ak, &mut s.akcap, bytes)?;
        let vb = reserve(&*self.b, &mut s.av, &mut s.avcap, bytes)?;
        let yb = reserve(&*self.b, &mut s.ay, &mut s.aycap, bytes)?;
        self.b.put(q, qb, 0)?;
        self.b.put(k, kb, 0)?;
        self.b.put(v, vb, 0)?;
        if !self.b.vision_attn(qb.addr(), kb.addr(), vb.addr(), yb.addr(), n, nh, hd, scale)? { return Ok(false); }
        self.b.get_into(yb, 0, out)?;
        bump(&self.attn_ns, t);
        Ok(true)
    }

    /// GEMMs run and the seconds in (convert, upload, gemm, read-back, attention) so far (`STRATA_VISION_DEBUG`).
    pub fn timing(&self) -> (u64, [f64; 5]) {
        let secs = |a: &AtomicU64| a.load(Relaxed) as f64 * 1e-9;
        (self.gemms.load(Relaxed),
         [secs(&self.conv_ns), secs(&self.up_ns), secs(&self.gemm_ns), secs(&self.down_ns), secs(&self.attn_ns)])
    }
}

/// `slot` grown to at least `need` bytes (and zeroed then; it is reused untouched after that).
fn reserve<'a>(b: &dyn Backend, slot: &'a mut Option<Buf>, cap: &mut usize, need: usize) -> anyhow::Result<&'a Buf> {
    if *cap < need {
        *slot = Some(b.zeros::<u8>(need.max(1))?);
        *cap = need;
    }
    Ok(slot.as_ref().expect("scratch buffer reserved"))
}

/// A `GpuTensor` owns a `Buf`; this one only carries the address of memory `Scratch` owns.
fn alias(b: &Buf) -> Buf { Buf::new(b.addr(), b.len(), ()) }

fn bump(a: &AtomicU64, t: Instant) { a.fetch_add(t.elapsed().as_nanos() as u64, Relaxed); }

impl MatMul for VisionGpu {
    fn mm(&self, w: &Tensor, x: &[f32], n: usize, y: &mut [f32], add: bool) -> anyhow::Result<()> {
        self.gemms.fetch_add(1, Relaxed);
        self.gemm(w, x, n, y, add)
    }

    fn attn(&self, q: &[f32], k: &[f32], v: &[f32], n: usize, n_head: usize, d_head: usize, scale: f32,
            out: &mut [f32]) -> anyhow::Result<bool> {
        self.attention_into(q, k, v, n, n_head, d_head, scale, out)
    }
}

/// Row `r` of `w` as f32. The projector's F16 rows are converted directly: `Tensor::row` dequantizes one element per
/// block, which costs about twice as much on the 443M weights of a 27-block projector.
fn weight_row_into(w: &Tensor, r: usize, out: &mut [f32]) {
    if w.ty == GgmlType::F16 && out.len() == w.cols {
        let row = &w.data[r * 2 * w.cols..][..2 * w.cols];
        for (o, b) in out.iter_mut().zip(row.chunks_exact(2)) { *o = f16::from_le_bytes([b[0], b[1]]).to_f32(); }
    } else {
        w.row(r, out);
    }
}

/// Run the Qwen3-VL encoder for `img` on the GPU when `gpu` is given (and on any GPU error, with a warning), else
/// on the CPU. Returns the merged image tokens and their grid.
pub fn encode_image(projector: &VisionProjector, img: &Rgb, gpu: Option<&VisionGpu>) -> anyhow::Result<(Vec<Vec<f32>>, usize, usize)> {
    let pre = projector.preprocess(img).context("preprocess image")?;
    let (nx, ny) = projector.count_tokens(pre.width, pre.height)?;
    let gpu = gpu.filter(|_| matches!(projector.cfg().kind, ProjectorKind::Qwen3Vl));
    let feats = match gpu {
        None => projector.encode(&pre)?,
        Some(g) => {
            eprintln!("image: vision encoder on the GPU ({})", g.b.device());
            let r = projector.encode_with(&pre, Some(g as &dyn MatMul));
            if std::env::var_os("STRATA_VISION_DEBUG").is_some() {
                let (gemms, [conv, up, gemm, down, attn]) = g.timing();
                eprintln!("image: vision GPU time: {gemms} matmuls {:.2} s (convert {conv:.2} s, upload {up:.2} s, \
                           gemm {gemm:.2} s, readback {down:.2} s), attention {attn:.2} s", conv + up + gemm + down);
            }
            match r {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("image: the GPU vision encoder failed ({e:#}); running it on the CPU");
                    projector.encode(&pre)?
                }
            }
        }
    };
    anyhow::ensure!(feats.len() == nx * ny, "encoder returned {} tokens, expected {nx}x{ny}", feats.len());
    Ok((feats, nx, ny))
}
