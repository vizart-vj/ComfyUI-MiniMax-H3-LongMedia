# ComfyUI-MiniMax-H3-LongMedia 0.6.54

This public release is based on the GitHub `v0.6.50` baseline and consolidates the validated runtime work developed afterward.

## Selective regeneration is actually selective

- `Regenerate Clip` can sample only the requested GENERATED clip while approved prefix/suffix TAKEs remain cache hits.
- `Regenerate From Here` samples only the selected clip and dependent GENERATED suffix.
- Adding a new clip reuses the previous TAKE continuation and does not resample approved history.
- Cached clips skip redundant text/reference conditioning during selective Setup preparation.

## Native continuation for generated and imported video

Director BASE blocks are now formally split into `GENERATED` and immutable `MEDIA`.

```text
GENERATED -> GENERATED
    saved native TAKE continuation -> one fresh child sample

MEDIA -> GENERATED
    external tail only -> VideoVAE tail encode -> native frame-0 H3 guide -> one fresh child sample
```

Imported MEDIA is never an H3 render unit, never copied into target x0, and never sent through a full-video VAE/diffusion roundtrip just to continue the timeline. Trimmed/reused media continues from the end of the visible BASE range.

## Mixed final assembly and audio

- H3 VideoVAE decodes GENERATED runs only.
- Original MEDIA RGB/audio is inserted directly into the final timeline.
- Hidden overlap is removed from the visible generated child.
- External tail continuation is cached on CPU.
- Per-block audio continuation supports `AUTO`, `CONTINUE`, `FRESH`.
- Mono/stereo pieces are assembled with an explicit `[1,C,L]` contract; mono is duplicated exactly when stereo is required.

## Quantized runtime and Windows memory

- AUTO policy routes INT8/W4A8/NVFP4 from physical packed storage + activation headroom rather than BF16-equivalent logical size.
- Explicit manual MLP/VRAM controls remain authoritative in NORMAL/AUTO-normal.
- Large Windows safetensors stay file-backed through ComfyUI `ModelMMAP` / `TensorFileSlice` when available; no eager whole-model RAM materialization.
- Setup conditioning is scoped to required clips and releases TE/transient references before diffusion sampling without aggressive Windows working-set trimming.
- Pinned transfer admission is based on real projected host-memory headroom.

## Preserved behavior

This release does not change H3 quality math, does not introduce SPEED/progressive-resolution, and preserves the existing Refiner AV contract: Stage 2 is video-only and exact Stage-1 audio is restored. Existing Director NLE/TAKE features remain intact.
