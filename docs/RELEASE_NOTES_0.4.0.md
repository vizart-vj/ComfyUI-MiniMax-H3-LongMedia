# ComfyUI-MiniMax-H3-LongMedia 0.4.0

0.4.0 is the first stable release built from the validated LongMedia 0.3.x development line.

## Major changes

- Unified **MultiClip** and **fixed segmentation** under one continuation executor.
- Added the standalone LongMedia Planner for per-clip prompt, duration and seed control.
- Fixed Planner ownership: a connected Planner is active only in `workflow_mode=multiclip`.
- Fixed Planner vertical resizing so an expanded node can shrink again.
- Removed obsolete manual reference-strength controls from the UI and backend schema.
- Consolidated source-audio/lip-sync timing around `audio_1` without an exposed strength weight.
- Added geometry-aware OOM Governor V4 and preventive attention preflight.
- Added bounded streamed Sol attention routing for very long sequences on constrained VRAM.
- Preserved low-VRAM DynamicVRAM/AIMDO support and stock-math fallbacks.
- Removed release-unnecessary dev hot reload, standalone VRAM tracker helper and accumulated per-build notes.

## New documentation

- `docs/PROMPTING_MULTICLIP.md`
- `docs/PROMPTING_SEGMENTATION.md`
- `docs/SAMPLER_OPTIMIZATION.md`
- `docs/ARCHITECTURE.md`
- `docs/RELEASE_AUDIT.md`

## Recommended production defaults

```text
sampler_mode   = auto
memory_mode    = auto
attention_mode = auto
```

Keep ComfyUI Dynamic VRAM enabled.

For long continuous work on a 16 GB GPU, fixed segments around 7–10 seconds are a practical starting point. Use shorter 5–8 second segments for difficult identity transfer, video-reference editing or lip-sync.

## Upgrade notes

Existing saved workflows continue to use the legacy internal `MiniMaxH3LatentLab...` class IDs intentionally. Public display names remain MiniMax H3 LongMedia.

Saved workflows that contained the removed strength widgets should load through the current node schema, but the values no longer affect 0.4.0 execution because the controls and backend plumbing were removed.
