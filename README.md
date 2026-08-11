# ComfyUI-MiniMax-H3-LongMedia

Custom ComfyUI nodes and low-VRAM execution patches for **MiniMax H3**, focused on long single-pass video/audio generation and predictable memory behavior on consumer GPUs.

## Current release

**v0.2.44 — SAFE / Long-Generation baseline**

This release is the current stability baseline after extensive VRAM profiling and long-sequence testing. It keeps the normal ComfyUI workflow model while adding H3-specific streaming/chunking paths where the stock execution path can exceed VRAM.

## Main nodes

The public workflow surface is intentionally small:

- **Long Media Setup**
- **Long Media Sampler**
- **Long Media Decode**

Internal helper nodes remain hidden from the normal Add Node/search UI.

## Compatibility note

The project was renamed from **ComfyUI-MiniMax-H3-LatentLab** to **ComfyUI-MiniMax-H3-LongMedia**.

Existing internal ComfyUI node class identifiers still use the legacy `MiniMaxH3LatentLab...` names on purpose. This preserves compatibility with workflows created before the rename. The public node names remain **Long Media Setup**, **Long Media Sampler**, and **Long Media Decode**.

## Highlights

- Long-media planning with segment/context handling.
- Video + audio latent support for MiniMax H3.
- Temporal continuity support for multi-segment workflows.
- Reference image/video/audio conditioning support.
- Setup-stage VRAM isolation around Qwen/CLIP/reference encoding.
- Post-sampling CUDA cache cleanup.
- Token-axis MLP chunking.
- Fused chunked `norm2 -> modulation -> MLP -> gate -> residual` path.
- Embedded/adapted Sol-Attn path for H3.
- Compressed streamed K/V storage and streamed Q processing for long sequences.
- Chunked Sol output projection.
- Adaptive inter-block VRAM guard with emergency cooldown.
- Late-block hard guard for very long sequences.
- Streamed final H3 output layer to avoid full-sequence FP32 hidden allocations.
- Sol OOM adaptive retry ladder instead of falling into catastrophic full-attention/NVFP4 dequantization fallback.

## Installation

Clone or copy this repository into:

```text
ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-LongMedia
```

Then restart ComfyUI.

No upstream `ComfyUI-sol-attn` node installation is required for the embedded Sol path used here.

## Recommended runtime behavior

For long / memory-constrained runs, **ComfyUI Dynamic VRAM should remain enabled**. In other words, do **not** launch ComfyUI with:

```text
--disable-dynamic-vram
```

The SAFE long-run configuration used during development typically uses:

```text
attention_mode                       = sol
mlp_chunk_tokens                     = 24576
sol_qkv_chunk_tokens                 = 8192
sol_out_proj_chunk_tokens            = 24576
vram_activation_reserve_mb           = 4096
inter_block_vram_guard_mb            = 2048
inter_block_guard_cooldown_blocks    = 4
inter_block_guard_emergency_mb       = 512
inter_block_guard_emergency_cooldown_blocks = 3
late_block_guard_start               = 40
late_block_guard_target_mb           = 6144
step_boundary_cleanup_mb             = 2048
```

These values are a conservative baseline, not universal optimums. Shorter clips can usually run faster with larger chunks and/or the existing attention path.

## Tested milestone

The current SAFE path has completed a **15-second, 1920×1088, 8-step MiniMax H3 single-pass generation on a 16 GB GPU** in the development setup with Dynamic VRAM enabled.

This is not a guarantee for every model build, LoRA, reference payload, ComfyUI version, or GPU. Reference count and conditioning length directly affect packed sequence length and VRAM use.

## Attention modes

### `existing`
Uses the attention implementation already installed/patched in ComfyUI. This is generally the fastest choice when the sequence comfortably fits in VRAM.

### `sol`
Uses the embedded H3-specific Sol path. For long sequences it can switch to compressed streamed K/V + streamed Q processing to reduce peak activation memory.

### `scheduled_sol`
Uses the same embedded Sol implementation with sigma-aware tau scheduling.

## Low-VRAM strategy

The long-sequence path avoids several large full-sequence intermediate tensors:

1. QKV projection is streamed in token chunks.
2. K/V are stored as INT8 + per-token scales; Sol summaries remain BF16.
3. Q is recomputed/consumed chunk-by-chunk.
4. Attention output is projected immediately back to hidden width.
5. The second transformer half is fused and chunked:
   `norm2 -> modulation -> MLP -> gate -> residual`.
6. The final H3 output layer is streamed so the full video hidden tensor is never converted to FP32 at once.
7. If Sol itself hits OOM, v0.2.44 retries with smaller QKV chunks instead of falling through to a large NVFP4 dequantization fallback.

## Notes on references

`reference_budget` limits reference conditioning, but packed sequence length still depends on how many reference items are attached and how they are tokenized. For very long clips, reducing reference count can recover substantial VRAM headroom.

## Example workflow

A clean public version of the validated SAFE workflow is included at:

```text
workflows/MiniMax-H3-LongMedia-SAFE-1080p-15s.json
```

It preserves the tested LongMedia memory settings, uses placeholder input media, and contains the recommended acceleration path used during validation. See `workflows/README.md` for dependencies and bypass instructions.

### Recommended acceleration nodes

The example workflow includes **ComfyUI-MiniMax-H3-Turbo** and the MiniMax H3 Sage Attention patches from **ComfyUI-KJNodes**. They are recommended for the validated setup, but they are not part of LongMedia itself.

If you have these nodes installed but do not want to use them, simply **Bypass** the corresponding nodes in the model chain. LongMedia's embedded Sol path remains available independently.

## Third-party code

This repository includes an adapted subset of **Saganaki22/ComfyUI-sol-attn** under the Apache License 2.0.

See:

- `THIRD_PARTY_NOTICE.md`
- `THIRD_PARTY_APACHE_2_0.txt`
- `sol_kernel/`

The embedded code is adapted for this MiniMax H3 integration and does not require the upstream custom node to be installed.

## Status

`0.2.44` is intentionally treated as the **SAFE baseline**. Future performance-oriented changes should preserve this memory-safe path rather than replacing it.
