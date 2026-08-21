# Sampler, VRAM and Performance Rules

This document describes the release policy for the Long Media Sampler in 0.4.2.

## First rule: use `sampler_mode=auto`

For normal production work, leave:

```text
sampler_mode = auto
memory_mode  = auto
attention_mode = auto
```

The runtime inspects the active H3 model, quantization, GPU VRAM and packed sequence geometry. It can lower the memory envelope and reject an unsafe full-sequence Sage/existing attention path before QKV allocation.

Manual mode is for controlled A/B tuning, not the default production path.

## Dynamic VRAM

Keep ComfyUI Dynamic VRAM enabled. Do not launch with:

```text
--disable-dynamic-vram
```

LongMedia's out-of-core policy relies on dynamic model residency for checkpoints larger than physical VRAM.

## Memory modes

### `auto`
Recommended. Selects an effective profile from model-size/VRAM ratio and runtime geometry.

### `normal`
Use only when the checkpoint and sequence comfortably fit. Maximizes residency and throughput.

### `low_vram`
Use when the model is larger than VRAM or long packed sequences cause pressure. Reduces MLP chunks, adds stronger barriers and enables bounded streamed attention paths when necessary.

### `ultra_low_vram`
Last-resort profile for very constrained GPUs / extremely oversized checkpoints. It minimizes simultaneous residency and increases transfer overhead substantially.

## Attention policy

### `auto`
Recommended. LongMedia keeps an exact attention family where possible and uses geometry-aware preflight for dangerous sequences.

### `existing`
Uses the installed/stock patched attention implementation. Usually fastest when the complete attention workspace fits.

### `sol`
Uses the embedded H3 Sol path. On long sequences LongMedia can stream QKV and compress K/V to bound peak activation memory.

### `scheduled_sol`
Same Sol path with sigma-aware tau scheduling. Use only when you intentionally want that approximation policy.

## 16 GB GPU production starting point

Use Auto first. If manual intervention is required:

```text
memory_mode                    = low_vram
attention_mode                 = sol
mlp_chunk_tokens               = 2048-4096
sol_qkv_chunk_tokens           = 8192
sol_out_proj_chunk_tokens      = 8192-24576
vram_activation_reserve_mb     = 2048-4096
inter_block_vram_guard_mb      = 2048
inter_block_guard_emergency_mb = 512
```

Prefer reducing `segment_duration` before aggressively shrinking every kernel chunk. Shorter clips reduce sequence length globally and are usually faster than making one enormous clip execute through extreme streaming.

## 12 GB class GPU

Recommended priority order:

1. Use fixed segmentation around 5–8 s.
2. Set `reference_budget=low`.
3. Use `memory_mode=low_vram`; escalate to `ultra_low_vram` only if required.
4. Use `attention_mode=sol` for long sequences that cannot use full attention safely.
5. Reduce MLP chunk size in 512-token steps.
6. Reduce QKV/out-projection chunks only after segment duration and reference payload are already controlled.

## 8 GB class GPU

H3 model residency is highly out-of-core. Favor:

- 4–6 s segments;
- low reference budget;
- low output resolution during exploration;
- `ultra_low_vram` when Auto cannot maintain sufficient headroom;
- minimal simultaneous reference media.

Expect transfer-bound execution. There is no chunk setting that makes a checkpoint dramatically larger than VRAM behave like a resident model.

## Speed optimization order

When a run is stable and you want more speed, change one variable at a time:

1. Increase `segment_duration` moderately to reduce handoffs.
2. Keep `memory_mode=auto` and let the governor exploit measured free VRAM.
3. Prefer `existing` attention for sequences that clearly fit; otherwise keep Auto.
4. Increase MLP chunk size.
5. Increase output-projection chunk size.
6. Increase QKV chunk size only when there is enough attention workspace headroom.
7. Reduce unnecessary Picture/Video/Audio references.

Do **not** disable safety guards simply to gain a small throughput improvement.

## OOM prevention behavior

For very long sequences, Governor V4 considers token geometry in addition to instantaneous `cudaMemGetInfo`. A large sequence cannot be classified as fast merely because VRAM happens to be free before transformer weights/workspaces are resident.

On constrained GPUs, unsafe full-sequence Sage attention is rejected **before** its large FP8/QKV workspace allocation and routed to bounded streamed Sol attention.

This is preventive routing, not an OOM retry.

## Reference budget

Reference media consumes packed sequence length. If memory rises unexpectedly:

1. reduce number of references;
2. use `reference_budget=low`;
3. shorten video references;
4. shorten `segment_duration`;
5. only then reduce low-level chunks.

A smaller chunk does not remove the persistent cost of a large reference payload.

## MultiClip vs fixed segmentation for performance

- MultiClip: duration is dictated by story/editing needs. Optimize individual long cards if one becomes a VRAM hotspot.
- Segmentation: use equal 5–10 s visible chunks as the primary memory-control lever.

Both use the same executor in 0.4.2, so performance differences come mainly from clip geometry and prompt/reference payload rather than separate sampler implementations.

## Quality safeguards

- Keep the same attention family across the passes of one long movie unless deliberately testing otherwise.
- Do not combine unrelated cache/attention replacement systems indiscriminately; multiple wrappers may alter the same transformer blocks and defeat LongMedia's memory assumptions.
- Change one optimization at a time and keep seed/prompt/media constant for A/B tests.
- If a custom resident MLP parity check fails, LongMedia falls back to stock math rather than accepting numerical drift.

## Practical presets

### Balanced 16 GB

```text
sampler_mode = auto
memory_mode = auto
attention_mode = auto
segment_duration = 7-10 s for long movies
reference_budget = low or medium
```

### Maximum safety 16 GB

```text
sampler_mode = manual
memory_mode = low_vram
attention_mode = sol
segment_duration = 5-8 s
reference_budget = low
mlp_chunk_tokens = 2048
sol_qkv_chunk_tokens = 8192
sol_out_proj_chunk_tokens = 8192
```

### Maximum speed when sequence fits

```text
sampler_mode = auto
memory_mode = normal
attention_mode = existing
```

Only use the final preset after confirming the actual reference payload and sequence fit with headroom.
