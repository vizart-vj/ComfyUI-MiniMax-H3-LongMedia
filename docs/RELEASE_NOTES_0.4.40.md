# MiniMax H3 LongMedia 0.4.40

## Highlights

- Fixed segmented-continuation decode isolation. The native per-clip continuous VideoVAE path now requires `plan.mode == "multiclip"` in both the sampler metadata producer and decoder consumer, preventing stale metadata from reproducing the latent-geometry mismatch reported in issue #8.
- Restored exact final-latent continuation semantics and expanded MultiClip hidden H3 context to 22 frames while preserving visible-duration accounting.
- Preserved split main/refine sampling and repaired legacy positional widget state from workflows saved during the sectioned-UI transition.
- Added the integrated MiniMax H3 latent hi-res path with AV separation, learned video-latent upscale and an independent high-resolution second sampling pass.
- Consolidated memory-safe SLA routing, bounded router allocations, exact resident INT8 MLP execution and adaptive VRAM safeguards.
- Production console is quiet by default. Full internal diagnostics remain available through `release_guard = false`.

## Compatibility

- Existing `MiniMaxH3LatentLab...` class identifiers are retained for workflow compatibility.
- `segmented_continuation` continues to use its stitched latent decode path.
- `multiclip` alone owns native per-clip continuous VideoVAE assembly metadata.
- In the original 0.4.40 package, the README still referenced the bundled `ex.png` screenshot.

- Fixed `segmented_continuation` native-decode routing by using `workflow_mode` as the authoritative workflow discriminator in both Sampler and Decode.

- Hires-disabled workflows no longer require a latent-upscaler checkpoint; the hidden model input uses a valid `(disabled)` sentinel and the last selected model is remembered for re-enable.
- Setup production defaults are now `manual_duration=10s` and `segment_duration=5s`.
- Setup and Sampler widget state is persisted by widget name, not only positional `widgets_values`, so values survive workflow reloads, page refreshes and workflow-mode changes without section-header shifts.
- `refine_enabled` production default is synchronized to ON with 2 refine steps.
