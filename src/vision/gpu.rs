//! GPU (CUDA) offload of the Qwen3-VL vision encoder.
//!
//! The merger (`mm.0`/`mm.2`), the patch convs and, when the device has room, the whole ViT stack run on the device.
//! The stack is the win: `MatMul::blocks` keeps the 27 layers resident — QKV, attention, output projection, FFN, the
//! LayerNorm, bias adds, activation and M-RoPE are all device kernels there — so an encode costs one upload and one
//! read-back instead of ~15 GB of PCIe round trips. `STRATA_VISION_STREAM=1` forces the streaming path, which stays
//! as the fallback (and is what a device without ~240 MB of free VRAM gets): one weight matrix resident at a time,
//! each multiply uploaded as F32 with rows zero-padded to the GEMM's 64-deep K step, run, and dropped again. That
//! keeps the projector's ~0.9 GB of F16 weights from having to fit beside a large text model.
//!
//! `STRATA_VISION_CPU=1` keeps everything on the CPU. The device and host buffers of the streaming path are reused
//! across the layers (`Scratch`): allocating and zeroing 20-85 MB three times per multiply cost more than some of
//! the multiplies.

use crate::backend::{self, Backend, Buf, GpuTensor};
use crate::quant::GgmlType;
use crate::vision::image::Rgb;
use crate::vision::mmproj::{FfnKind, ProjectorKind};
use crate::vision::projector::{BlockGeom, MatMul, VisionProjector};
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
    /// `STRATA_VISION_DEBUG`: weight conversion, upload, GEMM, read-back, attention, and the device-resident blocks.
    gemms: AtomicU64,
    conv_ns: AtomicU64,
    up_ns: AtomicU64,
    gemm_ns: AtomicU64,
    down_ns: AtomicU64,
    attn_ns: AtomicU64,
    blk_ns: AtomicU64,
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
                blk_ns: AtomicU64::new(0),
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
        if !self.b.vision_attn(qb.addr(), kb.addr(), vb.addr(), yb.addr(), n, nh, hd, scale, nh * hd)? { return Ok(false); }
        self.b.get_into(yb, 0, out)?;
        bump(&self.attn_ns, t);
        Ok(true)
    }

    /// GEMMs run and the seconds in (convert, upload, gemm, read-back, attention, blocks) so far
    /// (`STRATA_VISION_DEBUG`).
    pub fn timing(&self) -> (u64, [f64; 6]) {
        let secs = |a: &AtomicU64| a.load(Relaxed) as f64 * 1e-9;
        (self.gemms.load(Relaxed),
         [secs(&self.conv_ns), secs(&self.up_ns), secs(&self.gemm_ns), secs(&self.down_ns), secs(&self.attn_ns),
          secs(&self.blk_ns)])
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

/// `gelu_rows` mode for the projector's activation (`projector::activate`).
fn act_mode(f: FfnKind) -> i32 {
    match f {
        FfnKind::Gelu => 0,
        FfnKind::GeluErf => 1,
        FfnKind::GeluQuick => 2,
        FfnKind::Silu => 3,
    }
}

impl VisionGpu {
    /// Upload one block weight into `dst` as F32 rows of `cols`, dequantizing the stored type and zero-padding the
    /// tail to the GEMM's 64-column K step. The GEMM has F32 and BF16 kernels (`gemm_f32` / `gemm_bf16`); the
    /// projector's F16 weights are converted here, as `gemm()` does for the streaming path.
    fn put_weight(&self, w: &Tensor, cols: usize, dst: &Buf, stage: &mut Vec<f32>) -> anyhow::Result<usize> {
        anyhow::ensure!(cols >= w.cols, "vision weight: {cols} columns cannot hold its {}", w.cols);
        anyhow::ensure!(w.rows * cols * 4 <= dst.len(), "vision weight {}x{cols} needs {} bytes, the buffer has {}",
            w.rows, w.rows * cols * 4, dst.len());
        stage.resize(w.rows * cols, 0.0);
        stage.par_chunks_exact_mut(cols).enumerate().for_each(|(r, d)| {
            weight_row_into(w, r, &mut d[..w.cols]);
            d[w.cols..].fill(0.0);
        });
        self.b.put(stage, dst, 0)?;
        Ok(w.rows)
    }

    /// The whole ViT block stack on the device: one upload, ~11 kernels a layer, one read-back, so the residual
    /// stream never leaves the device and no layer pays for a PCIe round trip.
    fn blocks_dev(&self, p: &VisionProjector, x: &mut [f32], g: &BlockGeom) -> anyhow::Result<bool> {
        let (n, e, nh, hd) = (g.n, g.n_embd, g.n_head, g.d_head);
        if n == 0 || e == 0 || nh == 0 || hd < 2 || !hd.is_multiple_of(2) || hd > 96 || g.n_layer == 0 { return Ok(false); }
        if x.len() != n * e { return Ok(false); }
        if std::env::var_os("STRATA_VISION_STREAM").is_some() { return Ok(false); }

        // The FFN width, and its rows padded to the GEMM's 64-column K step.
        let ff = p.model.tensor("v.blk.0.ffn_up.weight")?.rows;
        let ffp = ff.div_ceil(64) * 64;
        let half = hd / 2;

        // The largest weight matrix (F32 after the conversion) and bias, for the one reusable buffer of each.
        let (mut max_w, mut max_b) = (0usize, 3 * e);
        for il in 0..g.n_layer {
            for (name, cols) in [("attn_qkv.weight", e), ("attn_out.weight", e), ("ffn_up.weight", e),
                                 ("ffn_down.weight", ffp)] {
                let w = p.model.tensor(&format!("v.blk.{il}.{name}"))?;
                max_w = max_w.max(w.rows * cols * 4);
            }
            for name in ["attn_qkv.bias", "ffn_up.bias"] {
                max_b = max_b.max(p.model.tensor(&format!("v.blk.{il}.{name}"))?.cols);
            }
        }

        let (n3, q3) = (n * e, 3 * e);
        // QKV (q3 values a row) is dead once attention has run, so the padded FFN buffer (ffp values a row, the
        // larger of the two) reuses its memory; `memset` clears the row padding before each up-projection.
        let wide = q3.max(ffp);
        let bytes = (n3 * 4 * 2 + n * wide * 4 + n * hd * 4 + max_w + e * 4 + max_b * 4) as u64;
        // The streaming path's buffers (~200 MB) are idle now: drop them so the graph fits beside a resident text
        // model. A device without room keeps the streaming path rather than failing half-way.
        *self.lock() = Scratch::default();
        let free = self.b.mem_info()?.0 as u64;
        if free < bytes + (64 << 20) {
            eprintln!("image: {} MB of VRAM free, {} MB needed for the vision blocks on the device; streaming them",
                free >> 20, bytes >> 20);
            return Ok(false);
        }

        let b = &*self.b;
        let t0 = Instant::now();
        // x and h (residual stream and the normalized copy), the packed QKV / padded FFN buffer, the RoPE tables
        // and one layer's weight matrix. The attention result lands in `hn`: it is dead from the QKV GEMM until the
        // next LayerNorm writes it, and the output projection reads it before that.
        let xd = b.zeros::<f32>(n3)?;
        let hn = b.zeros::<f32>(n3)?;
        let qd = b.zeros::<f32>(n * wide)?;
        let cst = b.zeros::<f32>(2 * n * half)?;
        let wd = b.zeros::<u8>(max_w)?;
        let lw = b.zeros::<f32>(e)?;
        let bb = b.zeros::<f32>(max_b)?;

        // RoPE tables: the cosines then the sines of this token grid, with `ops::rope_vision`'s own arithmetic (the
        // first half of the pairs turn by y, the second by x, each section starting the theta exponent over).
        let section = half / 2;
        let theta_scale = g.base.powf(-2.0 / half as f32);
        let mut cs = vec![0f32; 2 * n * half];
        for t in 0..n {
            let dx = t % g.m;
            let dy = (t / g.m) % g.m;
            let blk = t / (g.m * g.m);
            let (w2, h2) = (blk % g.mx, blk / g.mx);
            let (px, py) = ((w2 * g.m + dx) as i32, (h2 * g.m + dy) as i32);
            for i in 0..half {
                let (pos, local) = if i < section { (py, i) } else { (px, i - section) };
                let theta = pos as f32 * theta_scale.powi(local as i32);
                cs[t * half + i] = theta.cos();
                cs[n * half + t * half + i] = theta.sin();
            }
        }
        b.put(x, &xd, 0)?;
        b.put(&cs, &cst, 0)?;

        let scale = 1.0 / (hd as f32).sqrt();
        let mut stage: Vec<f32> = Vec::new();
        let mut bias = vec![0f32; max_b];
        // One bias of `cols` values in device buffer `bb` (uploaded fresh for each kernel that reads it).
        let row = |bias: &mut Vec<f32>, name: &str, cols: usize| -> anyhow::Result<u64> {
            p.model.tensor(name)?.row(0, &mut bias[..cols]);
            b.put(&bias[..cols], &bb, 0)?;
            Ok(bb.addr())
        };
        // The next block weight, in `wd`, as a tensor the GEMM reads.
        let weight = |name: &str, cols: usize, stage: &mut Vec<f32>| -> anyhow::Result<GpuTensor> {
            let rows = self.put_weight(&p.model.tensor(name)?, cols, &wd, stage)?;
            Ok(GpuTensor::new(GgmlType::F32, rows, cols, alias(&wd)))
        };
        for il in 0..g.n_layer {
            let pre = format!("v.blk.{il}.");
            // attention: layer-norm, QKV, bias, RoPE, attention, output projection added to the stream.
            b.put(&p.model.to_f32(&format!("{pre}ln1.weight"))?, &lw, 0)?;
            let p1 = row(&mut bias, &format!("{pre}ln1.bias"), e)?;
            b.ln_rows(xd.addr(), lw.addr(), p1, hn.addr(), e, n, e, g.eps)?;
            let wt = weight(&format!("{pre}attn_qkv.weight"), e, &mut stage)?;
            b.gemm_dense(&wt, n, hn.addr(), e, qd.addr(), q3, 0)?;
            let pq = row(&mut bias, &format!("{pre}attn_qkv.bias"), q3)?;
            b.bias_rows(qd.addr(), pq, q3, n, q3)?;
            b.rope_vis_rows(qd.addr(), cst.addr(), 0, nh, hd, n, q3)?;
            b.rope_vis_rows(qd.addr(), cst.addr(), e, nh, hd, n, q3)?;
            let (qp, kp, vp) = (qd.addr(), qd.addr() + (e * 4) as u64, qd.addr() + (2 * e * 4) as u64);
            if !b.vision_attn(qp, kp, vp, hn.addr(), n, nh, hd, scale, q3)? { return Ok(false); }
            let wt = weight(&format!("{pre}attn_out.weight"), e, &mut stage)?;
            b.gemm_dense(&wt, n, hn.addr(), e, xd.addr(), e, 2)?;
            let po = row(&mut bias, &format!("{pre}attn_out.bias"), e)?;
            b.bias_rows(xd.addr(), po, e, n, e)?;

            // FFN: layer-norm, up (written padded), bias + activation, down added to the stream. The QKV result is
            // dead now, so the up-projection reuses its memory; clearing it keeps the row padding zero for the
            // down-projection (which reads `ffp` columns).
            b.put(&p.model.to_f32(&format!("{pre}ln2.weight"))?, &lw, 0)?;
            let p2 = row(&mut bias, &format!("{pre}ln2.bias"), e)?;
            b.ln_rows(xd.addr(), lw.addr(), p2, hn.addr(), e, n, e, g.eps)?;
            let wt = weight(&format!("{pre}ffn_up.weight"), e, &mut stage)?;
            b.memset(&qd)?;
            b.gemm_dense(&wt, n, hn.addr(), e, qd.addr(), ffp, 0)?;
            let pu = row(&mut bias, &format!("{pre}ffn_up.bias"), ff)?;
            b.gelu_rows(qd.addr(), pu, ff, n, ffp, act_mode(g.ffn))?;
            let wt = weight(&format!("{pre}ffn_down.weight"), ffp, &mut stage)?;
            b.gemm_dense(&wt, n, qd.addr(), ffp, xd.addr(), e, 2)?;
            let pd = row(&mut bias, &format!("{pre}ffn_down.bias"), e)?;
            b.bias_rows(xd.addr(), pd, e, n, e)?;
        }
        b.get_into(&xd, 0, x)?;
        self.gemms.fetch_add((4 * g.n_layer) as u64, Relaxed);
        if std::env::var_os("STRATA_VISION_DEBUG").is_some() {
            eprintln!("image: vision blocks on the GPU: {n} tokens x {e}, {} layers", g.n_layer);
        }
        bump(&self.blk_ns, t0);
        Ok(true)
    }
}

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

    fn blocks(&self, p: &VisionProjector, x: &mut [f32], g: &BlockGeom) -> anyhow::Result<bool> {
        self.blocks_dev(p, x, g)
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

/// Encode `img` with this encoder's own device (None on the CPU).
impl VisionGpu {
    pub fn encode(&self, p: &VisionProjector, img: &Rgb) -> anyhow::Result<(Vec<Vec<f32>>, usize, usize)> {
        encode_image(p, img, Some(self))
    }
}

/// A vision encoder with a device of its own: the CUDA context is opened on the first image and dropped with it, so a
/// run that sends no images keeps that VRAM. Used by the `--image` paths and by serve requests, whose runner lends it
/// expert slots to make room when it has them (`GpuModel::lend_out`) and takes them back when the encode is done.
pub struct Owned {
    device: Option<usize>,
    gpu: Option<VisionGpu>,
    tried: bool,
}

impl Owned {
    pub fn new(device: Option<usize>) -> Owned { Owned { device, gpu: None, tried: false } }

    /// Encode `img` on the GPU when the device opened and the projector has a GPU path, on the CPU otherwise (decided
    /// once, on the first image; the context is dropped with the `Owned`).
    pub fn encode(&mut self, p: &VisionProjector, img: &Rgb) -> anyhow::Result<(Vec<Vec<f32>>, usize, usize)> {
        if !self.tried {
            // Only the Qwen3-VL encoder has a GPU path; the Gemma projectors stay on the CPU.
            self.gpu = match p.cfg().kind {
                ProjectorKind::Qwen3Vl => VisionGpu::open(self.device),
                _ => None,
            };
            self.tried = true;
        }
        match &self.gpu {
            Some(g) => g.encode(p, img),
            None => encode_image(p, img, None),
        }
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
                let (gemms, [conv, up, gemm, down, attn, blk]) = g.timing();
                eprintln!("image: vision GPU time: {gemms} matmuls {:.2} s (convert {conv:.2} s, upload {up:.2} s, \
                           gemm {gemm:.2} s, readback {down:.2} s), attention {attn:.2} s, blocks {blk:.2} s",
                          conv + up + gemm + down);
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
