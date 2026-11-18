# Vision / multimodal plan

Status: Qwen3-VL image input works on the CPU path and on the GPU path. The projector, image
preprocessing, prompt expansion and M-RoPE text integration run against `qwen3.5-2b` +
`qwen3.5-2b-mmproj.gguf`; `generate`/`chat --image` and `serve` image requests (OpenAI `image_url` and
Anthropic image blocks) work with or without `--gpu`.

Gemma 4 (`gemma4v`) image input works through `generate --image`, `chat --image` and `serve` (OpenAI
`image_url`/Anthropic `image` blocks); `chat`/`serve` use the model's `tokenizer.chat_template` (system
turn, `<|think|>` when `--think`, empty `<|channel>thought<channel|>` when not). Its encoder (`2*x-1` patches, x/y
position tables, RMS ViT with split NEOX RoPE, 3x3 average pooling, `v.std_bias`/`v.std_scale` and
`mm.input_projection`) matches `llama-mtmd-debug -p encode` on the 26B-A4B mmproj. Gemma 4 text uses
normal RoPE, so image tokens take sequential positions and the prompt inserts `<|image>` (BOI), the
soft tokens and `<image|>` (EOI). The `gemma4uv` unified embedder (big 48x48 conv patch, three
LayerNorms, x/y position tables, unweighted RMS norm, `mm.input_projection`) is verified against
`llama-mtmd-debug -p encode` with the 0.8B unified mmproj. GLM-4V and DeepSeek-OCR have no matching text
forward in this repo and stay rejected.

GPU image prompts prefill on the GPU's tensor-core chunked prefill (`GpuModel::prefill_image`), with the
projector's rows for image tokens and a per-token M-RoPE position table (`rope_mrope_rows` in kernels.cu).
The KV row stays contiguous while the RoPE position follows the image grid, and text after the image
resumes at the projector's `next_position` through `GpuModel::rope_offset`/`rope_buf`. A chat keeps the
image across turns: `prefill_image` continues the state saved by the previous turn when its ids are a prefix
of the new prompt and an image-content hash (`saved_image`) matches, so the image is prefilled once.
Backends without the batched M-RoPE kernel (Vulkan, CPU) fall back to one token a forward through
`step_embed`. Gemma 4 uses normal RoPE and is
wired through `generate`, `chat` and `serve`; `generate` is raw completion, so a chat-tuned Gemma 4
needs the template in `--prompt` (or use `chat`), while `chat`/`serve` apply it. GLM-4V/DeepSeek-OCR need a matching text forward before
their projectors can be used.

The goal is image input for the Qwen family first (`qwen35`, `qwen35moe`, `qwen4exp`, `qwen3next`,
`qwen3`, `qwen3moe`) with a shared `clip` loader and projector registry. The registry also carries Gemma 4
(`gemma4v`, `gemma4uv`). DeepSeek V4.1-Flash and GLM-5.3 are text-only upstream; their multimodal
variants use different text architectures, so there is no matching projector to wire. Audio projectors are
out of scope.

## Layout

```
src/vision/mod.rs        VisionSpec, mmproj discovery/validation, registry
src/vision/mmproj.rs     CLIP GGUF metadata + tensor access
src/vision/image.rs      decode, data URL/HTTP fetch, smart resize, normalize
src/vision/projector.rs  VisionProjector struct + Qwen3VL encoder
src/vision/gpu.rs       CUDA offload of the Qwen3VL encoder's matrix multiplies and attention
src/vision/prompt.rs     image spans, placeholder expansion, token embeddings/positions
```

Dependencies: `image` (png/jpeg/webp) and `base64`. Reuse `ureq` for remote URLs. Both are already in
the local cargo cache.

## Projector interface

```rust
pub struct PreprocessedImage { pub width: usize, pub height: usize, pub rgb: Vec<f32> } // interleaved RGB

pub struct VisionProjector {
    pub cfg: ClipConfig,
    pub model: ClipModel,
}

impl VisionProjector {
    pub fn preprocess(&self, image: &image::RgbImage) -> anyhow::Result<PreprocessedImage>;
    pub fn count_tokens(&self, width: usize, height: usize) -> anyhow::Result<(usize, usize)>;
    pub fn encode(&self, img: &PreprocessedImage) -> anyhow::Result<Vec<Vec<f32>>>; // merged_tokens x projection_dim
    pub fn image_token_count(&self, img: &PreprocessedImage) -> anyhow::Result<usize>;
    pub fn positions(&self, start: usize, width: usize, height: usize) -> Vec<[i32; 4]>;
}
```

`count_tokens` returns the number of merged image tokens (`nx * ny / merge^2`) and the token grid
(`nx/merge`, `ny/merge`).

## Qwen3-VL projector

The mmproj metadata is `clip.projector_type=qwen3vl_merger`, `clip.vision.image_size=768`,
`patch_size=16`, `spatial_merge_size=2`, `image_mean/std=[0.5; 3]`, `use_gelu=true`. The tensor set is
`v.patch_embd.weight`, `v.patch_embd.weight.1`, `v.patch_embd.bias`, `v.position_embd.weight`,
`v.blk.N.{ln1,ln2,attn_qkv,attn_out,ffn_up,ffn_down}.{weight,bias}`, `v.post_ln.{weight,bias}`,
`mm.0.{weight,bias}`, `mm.2.{weight,bias}`.

Pipeline:

1. Smart resize: preserve aspect ratio, align to `patch_size * merge = 32`, keep pixel count in
   `[8*32*32, 4096*32*32]`; bilinear resize with align-corners like `mtmd_image.cpp`. For Qwen3-VL and
   GLM-4V the merged-token grid is then capped at 1200 tokens and 40 a side (`ClipConfig::resize`,
   `STRATA_VISION_MAX_TOKENS` / `STRATA_VISION_MAX_SIDE`), i.e. about 30x40 tokens for a photo; the Gemma
   projectors keep their own 40..280 range.
2. Normalize RGB to `(v/255 - mean) / std`, keep interleaved RGB `[w][h][c]`.
3. Patch conv 16x16 stride 16, two convs summed; add patch bias. Tensor `v.patch_embd.weight` is
   `[16,16,3,1024]` and `.1` the same, row order `(kx, ky, c)`.
4. Resize `v.position_embd.weight` from 48x48 to `(nx, ny)` with bilinear + antialias
   (`ggml_interpolate` defaults), add to patches.
5. LayerNorm (not RMS) pre-LN if present; 24/27 blocks: LayerNorm -> fused QKV -> M-RoPE (vision,
   2-D: first 16 pairs y, next 16 x) -> softmax attention -> output projection + bias -> residual;
   LayerNorm -> FFN up/GELU/down -> residual. NORMAL LayerNorm and exact tanh GELU.
6. Post LayerNorm. Rearrange patches into 2x2 merged blocks: within a block the 4 patch features are
   `(h1=0,w1=0), (h1=0,w1=1), (h1=1,w1=0), (h1=1,w1=1)`, i.e. row-major. Blocks are ordered
   `(h2, w2)` row-major.
7. `mm.0` (4*n_embd -> 4*n_embd) + GELU + `mm.2` (4*n_embd -> projection_dim). Deepstack features are
   concatenated when metadata enables them.

Positions for the text model come from `mtmd_image_tokens_get_decoder_pos`: image tokens start at the
placeholder position `p0`; `t = p0`, `x = p0 + i % nx`, `y = p0 + i / nx`, `z = 0`; text resumes at
`p0 + max(nx, ny)`.

## Text-model integration

- `ModelConfig.mrope_sections` parses `{arch}.rope.dimension_sections`.
- `State.rope_pos` is the M-RoPE position of the next token; `State.pos` remains the KV row. Image tokens
  advance the KV row by one but the RoPE position by `max(nx, ny)`.
- `Model::step_embed` feeds a precomputed embedding at explicit positions; text-only `step` is unchanged.
- `generate --image` and `chat --image` load the projector, encode the images, expand `<image>`, and feed the
  prompt through `feed_prompt`.
- `serve` parses OpenAI `image_url` blocks and Anthropic `image` blocks (base64 or URL); `Msg` carries the
  image URLs and the template renders one `<image>` placeholder per image. The chat page has an attach
  button, drag/paste, previews and sends `image_url` blocks with the message.
- GPU image prompts: `GpuModel::prefill_image` runs the prompt on the tensor-core chunked prefill, a chunk
  at a time. A chunk with image tokens is fed the projector's rows; other chunks embed normally. Every
  token carries its `[t, y, x, z]` M-RoPE position (`Prompt::positions`), uploaded as one table per chunk
  and applied by `rope_mrope_rows`, so no per-token round trip and no CPU prefill. The chunk's last hidden
  state goes through `prefill_logits` (output norm + projection). MTP is disabled on image prompts (the
  draft head has not seen the image rows). The state after the prompt-minus-last-token is saved with the
  image's content hash; the next chat turn restores it and prefills only the new suffix. A backend
  without `rope_mrope_rows` (Vulkan, CPU) still prefills one token a forward via `step_embed`.
- GPU vision encoder (`src/vision/gpu.rs`): under `--gpu` the Qwen3-VL encoder runs its matrix multiplies
  (patch convs, `attn_qkv`, `attn_out`, `ffn_up`, `ffn_down`, `mm.0`, `mm.2`) and the ViT's bidirectional
  attention on CUDA, with `STRATA_VISION_CPU=1` (or a non-CUDA backend, or under 512 MB of free VRAM) as the
  CPU fallback. The 27 ViT blocks normally run **entirely on the device** (`MatMul::blocks`): one upload of the
  patch embeddings, then per layer `ln_rows` (LayerNorm with affine), the QKV GEMM, `bias_rows`,
  `rope_vis_rows` (M-RoPE from a host-made cos/sin table), `attn_vision`, the output-projection GEMM accumulated
  into the residual stream by the GEMM's own `act=2`, `ln_rows`, the up-projection GEMM (written into a buffer
  whose rows are padded to the GEMM's 64-wide K step; the padding stays zero for the down-projection),
  `gelu_rows` (bias + activation), and the down-projection GEMM into the residual stream. One read-back at the
  end, so ~11 kernels a layer and no PCIe round trips. Weights go up as F32 (the GEMM types are `f32` and `bf16`),
  each converted and zero-padded to the 64-wide K step as it is uploaded; that costs ~170 MB of VRAM, and a device
  without room (or `STRATA_VISION_STREAM=1`) falls back to the streaming path: one weight matrix resident at a
  time, each multiply uploaded as F32 with rows zero-padded to the K step, run, and dropped again, so the ~0.9 GB
  of F16 weights never has to fit in VRAM
  beside the text model.
  Attention runs on `attn_vision` (kernels.cu): one warp serves 4 patches of one head, so each key's K and V rows
  are read once for 4 queries, with an online softmax and no mask. It reads q/k/v with a token stride, so the
  packed QKV rows are used in place, and supports `head_dim <= 96`, which covers the Qwen3-VL ViT (16 heads x
  72); anything wider stays on the CPU. `STRATA_VISION_DEBUG=1` prints the time split: matmul count and seconds,
  broken into weight conversion, upload, GEMM and read-back, plus attention and the device-resident blocks.
  Weights and x are rounded to f16 inside the GEMM (f32 accumulation), the same as the model's own prefill.
- `load_images` caches encoder results in host memory (`vision::cache`): the key is the projector's path and size
  plus the image source (a data URL carries the whole image), and `STRATA_VISION_CACHE` sets how many images are kept
  (default 4, 0 disables). A chat resends its history, images included, on every request, so the same image is
  encoded once; attaching a new image encodes only the new one. `serve` accepts bodies up to 64 MiB, and the web page
  downscales attachments to a 1280 px side (JPEG) before sending them, since the model sees at most a 30x40 grid.
- The VRAM the encoder needs does not sit idle between images: when a projector is configured (`--mmproj` or a
  sibling mmproj, `STRATA_VISION_MMPROJ`), all but 256 MB of `--vram-margin` becomes a separate expert-slot pool
  holding the hottest layers of the largest slot-size class (`ExpertCache::new`), and a serve request that misses the
  encoder cache frees that pool (`GpuModel::lend_out`), encodes with its own CUDA context, and puts the experts back
  from host RAM right after (`lend_back`). Freeing the pool drops the recorded CUDA graphs (`Backend::forget_graphs`),
  since the slots may come back at another address. The load log prints
  `vision loan: N MB of expert slots are freed while an image is encoded`, and each encode prints how much came back
  and how long the re-upload took.
- `tests/gpu_model.rs::gpu_image_prefill_matches_cpu` checks the CPU prefill logits against the GPU on
  CUDA and Vulkan, the KV-row/RoPE-offset split, and that decoding names the image color. It runs on the
  default model (whose sibling `mmproj-F16.gguf` is found by `find_sibling`) or `STRATA_MODEL`.
- Gemma 4 `generate --image` builds `<|image>` + soft tokens + `<image|>`, feeds the projector rows
  through `Gemma4Model::step_embed_opt` (no sqrt(n_embd) scaling, matching embedding batches) and
  decodes greedily.
- `tests/vision.rs::gemma4v_*` compare the encoder with `llama-mtmd-debug -p encode -n 96` for white and
  red images, and check the 40-token preprocessing floor (64x64 -> 336x336 -> 7x7 tokens).
- `tests/vision.rs::gemma4uv_*` do the same for the unified projector (96x96 -> 2x2 tokens; sum and
  edge values match llama.cpp, and 64x64 still grows to 336x336/49 tokens).
- Dense Gemma 4 text layers (no expert router) are supported, so the 0.8B unified GGUF can be loaded
  and driven with the uv projector; the 0.8B community checkpoint is a toy model, so the numeric
  projector check is the meaningful parity test.
- Gemma 4 `chat`/`serve` encode each image once, expand the message's `<image>` markers, reuse the KV
  prefix while the rendered conversation extends it, and decode greedily. `serve` is API-only (no web
  chat page) and rejects tools (the text template supports them, but the server does not parse calls yet);
  it needs `--model` because the `--models-dir` picker goes through `ModelConfig`.
- Done: GPU-native image prefill (chunked, with per-row M-RoPE) and cross-turn image state reuse. The vision
  encoder's remaining CPU work is the patch gather, LayerNorm, RoPE, GELU and bias adds (the attention and
  every matmul are on the GPU now).

## Validation

- Golden embeddings from `llama-mtmd-debug -m <text> --mmproj <mmproj> -p encode --image <file>`.
- First target: `qwen3.5-2b` + `qwen3.5-2b-mmproj.gguf` (24 blocks, 1024 hidden, F32 patch/position,
  BF16 block weights).
- Second: `Qwen3.6-35B-A3B` mmproj-F16 (27 blocks, 1152 hidden).
- Gemma 4: `gemma-4-26B-A4B` mmproj-F16 (`gemma4v`) and `gemma-4-unified-0.8B-tiny` mmproj-f16
  (`gemma4uv`, 48x48 conv patch, 3840-wide embedder, 2048 projection) with its Q2_K text GGUF.
- Unit tests: smart resize, bilinear/antialias resize, normalization, patch merge order, M-RoPE vision
  pairs, position mapping.
