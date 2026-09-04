# MiniMax H3 LongMedia 0.5.40

## Changes since 0.4.40

0.5.40 consolidates the development line after the public 0.4.40 release into a release-ready package. It keeps legacy workflow compatibility while moving the public UI and documentation to the current semantic contracts.

## Semantic Setup

- Added the independent `control_mode`, `h3_mode`, `timeline_mode`, and `duration_source` controls.
- Legacy `workflow_mode` remains serialized for compatibility but is no longer the primary public mental model.
- `duration_source` is always visible and controls timeline length independently from audio output policy and prompt conditioning.
- Added native `t2va`, `fl2va`, `ref2va`, `hybrid`, and `video_ref_edit` semantic conditioning families.

## Planner and Cameras

- Planner cards now have stable `clip_id`, editable names, prompts, durations, and optional seeds.
- Cards can be reordered without losing their identity/timeline ownership.
- Added local clip presets plus JSON import/export.
- Added **Long Media Cameras** with per-clip shot size, rig/support, camera body, lens, stabilization, movement path, speed, spatial relationship, entity continuity, and transition policy.
- Cameras can auto-sync to Planner order by stable `clip_id`.
- Camera instructions are compiled separately from diegetic Planner prompts to avoid competing cinematography instructions.

## Native Source-Video Editing and Audio

- Reworked `video_ref_edit` around native MiniMax H3 Ref2VA source-video conditioning.
- Preserve-style source editing can pair Video1+Audio1 as one native source-performance block while also locking Audio1 to the target AV clock.
- `lip_sync` supports arbitrary redubbing: Video1 remains the visual/motion reference while Audio1 can be completely new speech or singing.
- Audio2/Audio3 remain prompt-addressable semantic/music references.
- In `video_ref_edit`, Audio1 alone owns the preserved/passthrough soundtrack; Audio2/Audio3 are conditioning references and are not mixed into final source audio.
- Prompts can use audio references for audio-reactive visuals such as percussion-driven lighting or bass-driven environmental motion.
- `duration_source=video|audio|manual|longest_input` can deliberately trim or extend a `video_ref_edit` target.
- Video inputs remain IMAGE batches only; source soundtrack must be connected separately.

## Two-Stage H3 / Latent Hi-Res

- Hardened the integrated learned MiniMax H3 latent-upscale path.
- High-resolution reconstruction starts from the denoised Stage-1 x0, not an in-progress noisy solver state.
- Only video latent is spatially upscaled; audio latent is preserved.
- With Refiner enabled, Stage 2 is an independent same-seed fresh-noise high-resolution H3 pass using a subset of the connected scheduler curve.
- Target-grid VIDEO keyframes are rebuilt/removed when their old spatial geometry no longer matches after latent upscale, while Ref2VA references and audio guides are preserved.
- FastVideo VSA keeps a strict four-call base Sampler #1 while workflow-owned refiner steps remain separately controlled.

## Video Reconstruction and Looping

- Expanded the Video Reconstructor through native Ref2VA source editing, continuity policies, and detail-recovery passes.
- Added reconstruction detail recovery that preserves stable low-frequency geometry while restoring controlled higher-frequency detail.
- Reworked Loop Closure into a latent/H3 macro-state return rather than an RGB crossfade or forced exact frame copy.

## Fast H3 Compatibility

- Added isolated support for H3ddle/PulpCut FastH3 VSA packages.
- Added Kijai FastVideo VSA support.
- Added structural contract validation, zero-copy input-major INT8 handling, runtime rollback/isolation, and portable learned-VSA fallback where current Comfy Kitchen APIs are unavailable.
- Switching back to stock H3 clears FastH3-specific runtime state.

## Current ComfyUI / Windows Compatibility

- Added a process-local Windows Triton TinyCC include bootstrap for embedded Python/Triton builds.
- Fixed giant fused-QKV OOMs for `attention_mode=existing + Comfy Kitchen` with an exact query-streaming projection/attention path instead of silently substituting approximate attention.
- Isolated VideoHelperSuite animated latent-preview failures from successful H3 inference.
- Added a sampler-entry memory boundary so seed-only reruns start from a clean DynamicVRAM/AIMDO state even when Setup is cached.
- Guarded native INT8 VBAR prefetch on <=18.5 GB GPUs to prevent speculative next-block transfer spikes.
- Added RAM-pressure-aware full-model pinned-memory gating.

## Documentation and Examples

- Re-audited the documentation against the current semantic UI and runtime.
- Added `MODES_GUIDE.md`.
- Added `TWO_PASS_LATENT_HIRES_REFINER_GUIDE.md`.
- Rewrote Architecture, Sampler Optimization, MultiClip prompting, and Segmentation guides to remove stale 0.4.x production instructions.
- Added a sanitized Latent Upscale / Detailer example workflow with external media and local preview paths replaced by neutral placeholders.

## Compatibility

- Legacy `MiniMaxH3LatentLab...` node class IDs remain registered.
- Legacy `workflow_mode` values are migrated.
- Historical 0.4.30/0.4.40 release notes remain in `docs/` as historical records.
- Comfy Registry publisher: `noise`.
- GitHub repository owner: `vizart-vj`.
