# Sampler, VRAM and Performance Guide

These recommendations describe the current 0.6.50 line.

## Production Default

Start with:

```text
sampler_mode   = auto
memory_mode    = auto
attention_mode = auto
```

Keep ComfyUI Dynamic VRAM enabled.

Do not use `--disable-dynamic-vram` for normal production runs; LongMedia's oversized-model execution relies on coordinated dynamic residency.

## Memory Modes

### `auto`
Recommended. Selects a profile from model size, quantization/backend, GPU VRAM and packed sequence geometry.

### `normal`
Use when model + activation workspace fit comfortably. In 0.6.50 this is the **user-authoritative high-VRAM profile**: Sampler chunk/reserve/guard values are preserved instead of being silently replaced by low-VRAM floors. `mlp_chunk_tokens=0` truly disables LongMedia MLP chunking. With `attention_mode=existing`, `vram_activation_reserve_mb=0` and block/step guards disabled, a resident model can use the stock ComfyUI H3 DiT block path.

### `low_vram`
Uses tighter activation/residency limits and more aggressive chunking.

### `ultra_low_vram`
Last-resort execution for very constrained GPUs or extreme geometry; expect more transfer overhead.

## Attention Modes

### `auto`
Recommended. Selects a compatible exact/bounded path from the active backend and geometry.

### `existing`
Keeps the selected ComfyUI attention family. Current Comfy Kitchen INT8 can use an **exact query-streaming** path when full fused QKV itself cannot fit on a 16 GB-class GPU.

### `sol`
Uses the embedded H3 Sol bounded long-sequence path.

### `scheduled_sol`
Uses Sol with explicit sigma/tau scheduling controls.

## 16 GB-Class Native INT8

Native INT8 H3 on <=18.5 GB GPUs is treated as a constrained-residency case:

- speculative `prefetch_dynamic_vbars` is gated;
- blocks fault on demand;
- the next block does not reserve a competing VBAR/cast destination while the current activation set is live;
- full-model RAM pinning is rejected when physical-memory pressure would be unsafe.

## Repeat Queue Runs

Changing only the seed can cause ComfyUI to reuse cached Setup/guider state.

LongMedia therefore creates an execution-memory boundary on every Sampler invocation before `prepare_sampling()`:

- synchronize active CUDA work;
- cleanup stale prefetch queues;
- reset cast buffers;
- reset AIMDO/VBAR watermark state;
- release prior registered model residency;
- release dead allocator cache;
- clear LongMedia transient CUDA references that could survive through cached runtime/guider state;
- clear persistent FastH3/VSA CUDA geometry caches between executions.

The Latent Hi-Res model is returned to CPU through `try/finally`, including OOM/exception/cancel paths.

Repeated identical Queue runs should therefore start from the same residency assumptions instead of monotonically increasing the LongMedia-owned baseline.

## Manual VRAM Controls

The Sampler exposes detailed tuning controls:

```text
mlp_chunk_tokens
sol_qkv_chunk_tokens
sol_out_proj_chunk_tokens
vram_activation_reserve_mb
inter_block_vram_guard_mb
inter_block_guard_cooldown_blocks
inter_block_guard_emergency_mb
late_block_guard_start
late_block_guard_target_mb
step_boundary_cleanup_mb
```

In `normal`, `0` is an actual off switch for MLP chunking and the documented zero-capable VRAM guards/reserves. Positive MLP values up to 131072 reach the runtime unchanged, so 8192/16384/32768/65536/131072 are real A/B compute settings. `low_vram` and `ultra_low_vram` retain bounded safety caps.

Do not shrink every chunk pre-emptively. Start with Auto and change one pressure point at a time.

## Latent Hi-Res

Spatial latent upscaling increases token count quickly. For 16 GB-class GPUs:

- start with `latent_hires_scale=1.2–1.5`;
- keep `memory_mode=auto`;
- keep reference payload reasonable;
- move toward 2× only after the smaller high-resolution pass is stable.

See [Integrated Refine and Latent Hi-Res](TWO_PASS_LATENT_HIRES_REFINER_GUIDE_EN.md).

## Reference Budget

Large Picture, Video and Audio references add packed conditioning rows.

For long or memory-constrained runs start with:

```text
reference_budget = low
```

Increase only when the extra identity/style/reference fidelity is needed.

## Segmentation vs One Giant Pass

For one continuous scene, shorter fixed segments often provide better throughput/stability than an enormous single packed sequence.

Starting points:

- 16 GB: 5–10 s depending on references/edit complexity;
- 12 GB: 5–8 s with low reference budget;
- 8 GB: 4–6 s and expect transfer-bound execution.

These are starting points, not model limits.

## Windows Triton / RTX 50-Series

LongMedia includes a process-local Triton TinyCC compatibility bootstrap for embedded Windows Python environments where bundled TCC cannot find WinAPI headers. It does not permanently patch `site-packages`.

## Preview Compatibility

A broken external latent-preview hook should not terminate successful H3 inference. LongMedia isolates VideoHelperSuite animated-preview exceptions only when the traceback actually originates in VHS preview code; unrelated callback errors remain fatal.

## Debugging OOM

Capture:

```text
GPU total/free
PyTorch allocated/reserved
AIMDO/VBAR residency
pinned RAM
active attention backend
packed sequence geometry
failure before block 0 / QKV / MLP / final output
```

A failure inside `prefetch_queue_pop` is different from a fused-QKV allocation failure and should be treated as a transport/residency issue rather than "fixed" by changing attention math.
