//! GPU (CUDA) offload of the vision encoder: the Qwen3-VL matrix multiplies (patch convs, the ViT's QKV / attention
//! output / FFN and the `mm.0`/`mm.2` merger) and the ViT's bidirectional attention run on the device. The rest of the
//! encoder stays on the CPU: the patch gather, LayerNorm, RoPE, the GELU and the bias adds.
//!
//! One weight matrix is resident at a time, so the projector's ~0.9 GB of F16 weights never has to fit in VRAM
//! beside a large text model: each multiply uploads its matrix as F32 (rows zero-padded to the GEMM's 64-deep K
//! step, which is all `Backend::gemm_dense` asks for), runs the GEMM and drops it again. Under `--gpu` the encoder
//! uses this path automatically; `STRATA_VISION_CPU=1` keeps it on the CPU.

use crate::backend::{self, Backend, GpuTensor};
use crate::quant::GgmlType;
use crate::vision::image::Rgb;
use crate::vision::mmproj::ProjectorKind;
use crate::vision::projector::{MatMul, VisionProjector};
use crate::weights::Tensor;
use anyhow::Context;
use half::f16;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Free VRAM below which the encoder stays on the CPU. One image's buffers take roughly 100-250 MB.
const MIN_FREE_VRAM: usize = 512 << 20;

pub struct VisionGpu {
    b: Box<dyn Backend>,
    /// Nanoseconds in the GEMMs and in the attention kernel, printed by `encode_image` under `STRATA_VISION_DEBUG`.
    mm_ns: AtomicU64,
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
            Ok(_) => Some(VisionGpu { b, mm_ns: AtomicU64::new(0), attn_ns: AtomicU64::new(0) }),
            Err(e) => {
                eprintln!("image: {e:#}; running the vision encoder on the CPU");
                None
            }
        }
    }

    /// y[t * w.rows + r] = w[r] · x[t * w.cols..] on the device.
    fn gemm(&self, w: &Tensor, x: &[f32], n: usize, y: &mut [f32], add: bool) -> anyhow::Result<()> {
        anyhow::ensure!(w.cols > 0 && x.len() == n * w.cols, "vision gemm: x is {} values, not {n}x{}", x.len(), w.cols);
        anyhow::ensure!(y.len() == n * w.rows, "vision gemm: y is {} values, not {n}x{}", y.len(), w.rows);
        // The GEMM's K step is 64 columns: pad the weight rows and the x rows with zeros to reach it. The output
        // needs no padding (it is masked by `rows`).
        let cols = w.cols;
        let pw = cols.div_ceil(64) * 64;

        let mut wp = vec![0f32; w.rows * pw];
        wp.par_chunks_exact_mut(pw).enumerate().for_each_init(|| vec![0f32; cols], |row, (r, dst)| {
            weight_row(w, r, row);
            dst[..cols].copy_from_slice(&row[..]);
        });
        let wb = self.b.zeros::<f32>(wp.len())?;
        self.b.put(&wp, &wb, 0)?;
        let wt = GpuTensor::new(GgmlType::F32, w.rows, pw, wb);
        drop(wp);

        let xb = self.b.zeros::<f32>(n * pw)?;
        if pw == cols {
            self.b.put(x, &xb, 0)?;
        } else {
            let mut xp = vec![0f32; n * pw];
            xp.par_chunks_exact_mut(pw).enumerate().for_each(|(t, dst)| dst[..cols].copy_from_slice(&x[t * cols..(t + 1) * cols]));
            self.b.put(&xp, &xb, 0)?;
        }

        let yb = self.b.zeros::<f32>(y.len())?;
        self.b.gemm_dense(&wt, n, xb.addr(), pw, yb.addr(), w.rows, 0)?;
        let out = self.b.get::<f32>(&yb, 0, y.len())?;
        if add {
            for (a, b) in y.iter_mut().zip(&out) { *a += *b; }
        } else {
            y.copy_from_slice(&out);
        }
        Ok(())
    }

    /// The ViT's bidirectional attention on the device; None when the backend has no kernel for this `hd`.
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], n: usize, nh: usize, hd: usize, scale: f32) -> anyhow::Result<Option<Vec<f32>>> {
        if hd == 0 || hd > 96 { return Ok(None); }
        let len = n * nh * hd;
        anyhow::ensure!(q.len() == len && k.len() == len && v.len() == len,
            "vision attention: {n}x{nh}x{hd} needs {len} q, k and v values, got {} / {} / {}", q.len(), k.len(), v.len());
        let (qb, kb, vb, yb) = (self.b.zeros::<f32>(len)?, self.b.zeros::<f32>(len)?, self.b.zeros::<f32>(len)?, self.b.zeros::<f32>(len)?);
        self.b.put(q, &qb, 0)?;
        self.b.put(k, &kb, 0)?;
        self.b.put(v, &vb, 0)?;
        if !self.b.vision_attn(qb.addr(), kb.addr(), vb.addr(), yb.addr(), n, nh, hd, scale)? { return Ok(None); }
        Ok(Some(self.b.get::<f32>(&yb, 0, len)?))
    }

    /// Milliseconds in the GEMMs and in the attention kernel so far (`STRATA_VISION_DEBUG`).
    pub fn timing(&self) -> (f64, f64) {
        (self.mm_ns.load(Relaxed) as f64 * 1e-6, self.attn_ns.load(Relaxed) as f64 * 1e-6)
    }
}

/// Row `r` of `w` as f32. The projector's F16 rows are converted directly: `Tensor::row` dequantizes one element per
/// block, which costs about twice as much on the 443M weights of a 27-block projector.
fn weight_row(w: &Tensor, r: usize, out: &mut [f32]) {
    if w.ty == GgmlType::F16 && out.len() == w.cols {
        let row = &w.data[r * 2 * w.cols..(r + 1) * 2 * w.cols];
        for (o, b) in out.iter_mut().zip(row.chunks_exact(2)) { *o = f16::from_le_bytes([b[0], b[1]]).to_f32(); }
    } else {
        w.row(r, out);
    }
}

impl MatMul for VisionGpu {
    fn mm(&self, w: &Tensor, x: &[f32], n: usize, y: &mut [f32], add: bool) -> anyhow::Result<()> {
        let t = std::time::Instant::now();
        let r = self.gemm(w, x, n, y, add);
        self.mm_ns.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        r
    }

    fn attn(&self, q: &[f32], k: &[f32], v: &[f32], n: usize, nh: usize, hd: usize, scale: f32) -> anyhow::Result<Option<Vec<f32>>> {
        let t = std::time::Instant::now();
        let r = self.attention(q, k, v, n, nh, hd, scale);
        self.attn_ns.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        r
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
                let (mm, attn) = g.timing();
                eprintln!("image: vision GPU time: matmuls {:.2} s, attention {:.2} s", mm / 1e3, attn / 1e3);
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
    anyhow::ensure!(feats.len() == nx * ny, "encoder returned {} tokens, expected {nx}x{ny}", feats.len(), nx, ny);
    Ok((feats, nx, ny))
}
