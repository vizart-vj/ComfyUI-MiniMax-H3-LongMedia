# ComfyUI-MiniMax-H3-LongMedia 0.4.2

0.4.2 consolidates the validated 0.4.11 runtime baseline with the fixes developed afterward.

## MultiClip continuity

- Reworked MultiClip output onto one native H3 video-latent timeline.
- The first clip keeps the native startup prefix; each following continuation drops only the repeated 5-frame / 2-video-token prefix.
- The assembled video latent is validated on the native `T=5*k+2` grid and decoded once through the H3 VideoVAE.
- Removed the active per-clip VideoVAE decode/RGB seam-repair path, including hidden preroll, temporal seam selection, photometric matching and brightness outlier correction.
- Generated audio stays on its own timeline and is trimmed by absolute output timing instead of applying video-latent arithmetic to audio.

## Runtime compatibility

- Hardened low-VRAM MiniMax modulation handling for both scalar and multi-element row layouts.
- Fixed multi-row handling in chunked norm/MLP/gate processing and in the final video/audio output heads.

## Release cleanup

- Removed the `release_guard` runtime flag from `LongMediaPlan`, Setup output and reports.
- Release logging is now fixed and quiet: routine internal diagnostics are suppressed while actionable failures remain visible.
- Routine compatibility-patch INFO messages were demoted to DEBUG.
- Updated README, architecture, prompting, optimization and release-audit documentation for 0.4.2.

## Preserved from 0.4.11

- unified H3 model lifecycle for sequential LongMedia sampling;
- strict segmentation ownership;
- true two-stage refiner inside the same lifecycle;
- AIMDO/Text-Encoder pinned-memory gate;
- ComfyUI-native safe `model_options` cloning;
- existing low-VRAM governor, streamed Sol and quantized execution paths.
