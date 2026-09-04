# Sampler, VRAM and Performance Guide

These recommendations apply to the 0.5.40 release.

## Production Default

Start with:

```text
sampler_mode   = auto
memory_mode    = auto
attention_mode = auto
```

Keep ComfyUI Dynamic VRAM enabled.

Do not launch with:

```text
--disable-dynamic-vram
```

LongMedia's large-model execution relies on coordinated dynamic residency.

## Memory Modes

### `auto`
Recommended. Chooses an effective profile from model size, quantization/backend, GPU VRAM, and packed sequence geometry.

### `normal`
Use only when the model and activation workspace fit comfortably.

### `low_vram`
Uses tighter activation/residency limits and more aggressive chunking.

### `ultra_low_vram`
Last-resort execution for very constrained GPUs or extreme sequence geometry. Expect additional transfer overhead.

## Attention Modes

### `auto`
Recommended. LongMedia selects a compatible exact/bounded path from the active backend and sequence geometry.

### `existing`
Keeps the installed/selected ComfyUI attention family.

With current Comfy Kitchen INT8, LongMedia can switch the projection lifetime to an **exact query-streaming** path when the fused full QKV tensor itself cannot fit on a 16 GB-class GPU. This preserves the existing Comfy Kitchen attention contract rather than silently replacing it with Sol.

### `sol`
Uses the embedded H3 Sol path with bounded long-sequence execution.

### `scheduled_sol`
Uses the Sol path with the explicit sigma/tau scheduling controls.

## 16 GB-Class Native INT8

Current LongMedia treats native INT8 H3 on <=18.5 GB GPUs as a constrained-residency case.

In this situation:

- speculative `prefetch_dynamic_vbars` is disabled;
- blocks are faulted on demand;
- the next block does not reserve a competing VBAR/cast destination while the current activation set is still live;
- full-model RAM pinning is rejected when it would consume too much physical memory or leave insufficient headroom.

This protects both VRAM and system RAM on Windows/portable ComfyUI builds.

## Repeat Runs

Changing only the seed can cause ComfyUI to reuse cached Setup output.

LongMedia therefore applies a Sampler execution memory boundary on every invocation before `prepare_sampling()`:

- synchronize active CUDA work;
- cleanup stale prefetch queues;
- reset cast buffers;
- reset AIMDO/VBAR watermark state;
- release prior registered model residency;
- release dead allocator cache.

A second or third seed run should therefore start from the same residency assumptions as the first run.

## Manual VRAM Controls

The Sampler still exposes detailed controls for controlled tuning:

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

Do not shrink every chunk pre-emptively. Start with Auto and change one pressure point at a time.

## Latent Hi-Res

Spatial latent upscaling increases token count quickly.

For 16 GB-class GPUs:

- begin with `latent_hires_scale=1.2–1.5`;
- keep `memory_mode=auto`;
- keep reference payload reasonable;
- increase toward 2× only after the smaller high-resolution pass is stable.

See [Two-Stage Sampling, Latent Hi-Res and Refiner](TWO_PASS_LATENT_HIRES_REFINER_GUIDE.md).

## Reference Budget

Large Picture, Video, and Audio references add packed conditioning rows.

Start with:

```text
reference_budget = low
```

for long or memory-constrained runs. Increase only when the additional identity/style/reference fidelity is needed.

## Segmentation vs One Giant Pass

When the content is one continuous scene, shorter fixed segments often provide better throughput and stability than forcing an enormous single packed sequence through extreme streaming.

Starting points:

- 16 GB: 5–10 s segments depending on references and edit complexity;
- 12 GB: 5–8 s with low reference budget;
- 8 GB: 4–6 s and expect transfer-bound execution.

These are starting points, not model limits.

## Windows Triton / RTX 50-Series

LongMedia includes a process-local Triton TinyCC compatibility bootstrap for embedded Windows Python environments where Triton's bundled TCC fails to find its own WinAPI headers.

It does not patch `site-packages` permanently.

## Preview Compatibility

A broken external latent-preview hook should not terminate a successful H3 denoise pass. LongMedia isolates VideoHelperSuite animated-preview exceptions only when the traceback actually originates in VHS preview code; unrelated callback errors remain fatal.

## Debugging OOM

Useful facts to capture:

```text
GPU total/free
PyTorch allocated/reserved
AIMDO/VBAR residency
pinned RAM
active attention backend
packed sequence geometry
whether failure happens before block 0, in QKV, MLP, or final output
```

A failure inside `prefetch_queue_pop` is different from a fused-QKV allocation failure and should be treated as a transport/residency issue, not "fixed" by changing attention math.
