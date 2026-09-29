# ComfyUI-MiniMax-H3-LongMedia 0.6.62

This release adds per-clip regeneration controls and reversible adjacent-clip merging to Director.

## Changes

- Enable or disable GENERATED MAIN clips for ordinary RUN from the timeline, context menu, or clip settings. Disabled clips remain on the timeline and replay their active TAKE.
- RUN with disabled clips requires MultiClip and a compatible active TAKE for every GENERATED clip. If a TAKE is missing or clip geometry changed, execution stops with a clear error instead of silently widening the render queue.
- Select adjacent GENERATED clips and merge them into one timeline block. Original TAKEs remain intact; Undo or **Restore original clip boundaries** restores the separate clips without diffusion.
- The merged block inspector keeps an individual RUN switch for each original clip.
- Settings persist in `director_json`; older documents default to all clips enabled.

## Validation

- Clip-activation contract regression checks: 4/4.
- Python, JavaScript, and packaged JSON syntax checks passed.
- No live ComfyUI execution or GPU render was available during packaging.
