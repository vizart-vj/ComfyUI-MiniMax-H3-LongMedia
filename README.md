# ComfyUI-MiniMax-H3-LongMedia

Custom ComfyUI nodes and low-VRAM execution patches for **MiniMax H3**, focused on long single-pass video/audio generation and predictable memory behavior on consumer GPUs.

### Audio output policy

`audio_mode=preserve_reference` uses connected audio as native H3 audio conditioning (including rhythm/timing and `lip_sync` driving), but discards the model-generated audio after sampling and restores the original input track directly at decode/output. This avoids Turbo-LoRA audio reconstruction artifacts while retaining audiovisual interaction. `preserve` restores the original track without intentionally using it as a reference; `reference_only` keeps reference conditioning but uses generated H3 audio.

## 0.3.0 interface modes
- UI fix: workflow_mode now updates dynamic socket labels (first_frame/last_frame/source_video/etc.) in the Setup node.
- UI fix: Sampler `manual` mode now forces expert widgets to reappear reliably after mode changes and workflow reloads.

The release UI keeps the complete backend schema stable for workflow compatibility while hiding low-level controls in normal use. `workflow_mode` selects `hybrid_auto`, `ref2va_full`, `loop`, or `manual`. `sampler_mode=auto` uses the validated production policy; `manual` restores every tuning control. Decode obtains both VAEs from `LONG_MEDIA_PLAN`, so it has no public VAE sockets.


## Current release

**v0.3.1 — Audio passthrough & reference-routing hotfix**

v0.3.1 is a hotfix release on top of the v0.3.0 workflow redesign. It fixes audio passthrough/reference routing so attached source audio in `auto`, `preserve`, and `preserve_reference` is carried through the media plan and bypasses AudioVAE reconstruction at decode. The Hybrid / Ref2VA / Loop / video-reference / lip-sync workflow surface introduced in v0.3.0 remains unchanged.


## Workflow modes (0.3.0)

- **hybrid_auto** — recommended. `image_1` is the first frame. If `image_2` is connected it becomes the last frame; remaining images are `<Picture N>` references.
- **ref2va_full** — every connected `image_1..9` is a normal `<Picture N>` reference; no first/last-frame anchors are added.
- **loop** — `image_1` is internally sent to both the first and last frame anchors, matching the proven `hybrid_auto` setup with the same image wired to i1+i2. `image_2` is reserved/ignored and Picture refs begin at `image_3`.
- **manual** — exposes the legacy conditioning, segmentation, attention and VRAM controls for development and A/B tests.

The sampler has matching **auto/manual** presentation. Auto uses the validated production policy; Manual exposes the full tuning surface.

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

## Video reference audio

`video_1` / `video_2` / `video_3` are **IMAGE frame batches only**; ComfyUI does not carry an audio stream inside an `IMAGE` connection. If a reference video has audio that you want LongMedia to use, extract/load that audio separately and connect it to the matching `audio_1` / `audio_2` / `audio_3` input. For the primary video-to-video path, `video_1` and `audio_1` are treated as the matching source pair.


### 0.3.0 workflow modes

- `hybrid_auto` — `image_1` becomes the first frame anchor; `image_2` becomes the last frame when connected; remaining images are normal Picture refs.
- `video_ref_edit` — `video_1` is the main motion/camera/composition source, `image_1..9` are Picture refs for identity/style replacement, and `audio_1` can be the paired source soundtrack.
- `ref2va_full` — all connected images are plain Picture refs; no first/last keyframe semantics.
- `loop` — `image_1` is reused as both first and last frame for loop-friendly shots.
- `manual` — legacy expert controls, segmentation, and explicit conditioning widgets.