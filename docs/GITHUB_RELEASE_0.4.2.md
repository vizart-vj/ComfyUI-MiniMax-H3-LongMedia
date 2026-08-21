# 0.4.2 — Native continuous MultiClip decode

This release consolidates the validated 0.4.11 runtime with the fixes developed afterward.

### Highlights

- MultiClip now assembles sequential video latents on one valid native H3 temporal grid and performs a single VideoVAE decode for the full timeline.
- Each continuation removes only the repeated native 5-frame / 2-video-token prefix before assembly, eliminating the per-clip VideoVAE reset that caused visible boundary artifacts.
- Added compatibility for scalar and multi-element MiniMax modulation-row layouts in the low-VRAM chunked MLP path and final video/audio output heads.
- Removed the `release_guard` workflow/runtime flag and cleaned release logging: routine internal diagnostics are silent, while actionable failures remain visible.
- Updated README, architecture, prompting, optimization and release-audit documentation.

The 0.4.11 unified H3 lifecycle, two-stage refiner, segmentation isolation, AIMDO Text-Encoder gate and low-VRAM/streamed-Sol policies are preserved.
