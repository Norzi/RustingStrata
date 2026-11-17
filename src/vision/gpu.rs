//! GPU (CUDA) offload of the vision encoder: the Qwen3-VL matrix multiplies (patch convs, the ViT's QKV / attention
//! output / FFN and the `mm.0`/`mm.2` merger) run on the device. The rest of the encoder stays on the CPU: the patch
//! gather, LayerNorm, RoPE, the bidirectional softmax attention, the GELU and the bias adds.
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

/// Free VRAM below which the encoder stays on the CPU. One image's buffers take roughly 100-250 MB.
const MIN_FREE_VRAM: usize = 512 << 20;

pub struct VisionGpu {
    b: Box<dyn Backend>,
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
            Ok(_) => Some(VisionGpu { b }),
            Err(e) => {
                eprintln!("image: {e:#}; running the vision encoder on the CPU");
                None
            }
        }
    }
}

impl MatMul for VisionGpu {
    fn mm(&self, w: &Tensor, x: &[f32], n: usize, y: &mut [f32], add: bool) -> anyhow::Result<()> {
        anyhow::ensure!(w.cols > 0 && x.len() == n * w.cols, "vision gemm: x is {} values, not {n}x{}", x.len(), w.cols);
        anyhow::ensure!(y.len() == n * w.rows, "vision gemm: y is {} values, not {n}x{}", y.len(), w.rows);
        // The GEMM's K step is 64 columns: pad the weight rows and the x rows with zeros to reach it. The output
        // needs no padding (it is masked by `rows`).
        let cols = w.cols;
        let pw = cols.div_ceil(64) * 64;

        let mut wp = vec![0f32; w.rows * pw];
        let mut row = vec![0f32; cols];
        for (r, dst) in wp.chunks_exact_mut(pw).enumerate() {
            w.row(r, &mut row);
            dst[..cols].copy_from_slice(&row);
        }
        let wb = self.b.zeros::<f32>(wp.len())?;
        self.b.put(&wp, &wb, 0)?;
        let wt = GpuTensor::new(GgmlType::F32, w.rows, pw, wb);
        drop(wp);

        let xb = self.b.zeros::<f32>(n * pw)?;
        if pw == cols {
            self.b.put(x, &xb, 0)?;
        } else {
            let mut xp = vec![0f32; n * pw];
            for (t, dst) in xp.chunks_exact_mut(pw).enumerate() {
                dst[..cols].copy_from_slice(&x[t * cols..(t + 1) * cols]);
            }
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
            match projector.encode_with(&pre, Some(g as &dyn MatMul)) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("image: the GPU vision encoder failed ({e:#}); running it on the CPU");
                    projector.encode(&pre)?
                }
            }
        }
    };
    anyhow::ensure!(feats.len() == nx * ny, "encoder returned {} tokens, expected {}x{}", feats.len(), nx, ny);
    Ok((feats, nx, ny))
}
