# Changelog

## 0.6.50

- Fixed Refiner AV noise-stream device ownership on the high-VRAM/native path: frozen Stage-1 audio noise now follows ComfyUI's stock `prepare_noise()` device/dtype contract in both exact and windowed Stage-2 sampling, preventing mixed CPU/CUDA nested noise before `guider.sample()`.
- Expanded Director TAKE management with checkbox multi-select, Select All/Clear, batch delete, folder-tree navigation and `MOVE TO` for moving one or many TAKEs into existing folders.
- Reworked Director layers as first-class editor objects with selection, rename, duplicate/delete where valid, enable/lock/mute controls, per-layer properties, and consistent context-menu targeting.
- Added persistent per-layer vertical resizing by dragging each track separator; the global track-height control can still reset all tracks to one shared height.
- Fixed vertical layout stretch/gap artifacts after timeline, prompt and inspector resizing; removed the redundant outer Inspector resize shell while preserving prompt-field resizing.
- Reorganized the Director toolbar into icon-only groups separated by dividers, removed duplicate import actions, and moved Selected TAKE directly above the timeline.
- Added NLE-style timeline context menus and keyboard editing actions for Duplicate, Copy, Cut, Paste, Delete, clip knife and all-layer knife operations.
- Fixed `Knife clip here` so it splits only the context-clicked clip/block; `Knife all layers here` remains the explicit global cut action.
- Added movable timeline blocks with less aggressive Magnet snapping and Alt-drag snap bypass for free positioning.
- BASE clips can now be trimmed or shifted to author real gaps without automatic collapse; unresolved BASE gaps are rejected before execution with an explicit Fill Gap message instead of being silently rewritten.
- Added type-aware `Fill Gap` actions for BASE, CAMERA, AUDIO and custom layers, creating a correctly typed block across the clicked empty interval.
- Added a dedicated blank timeline creation zone whose context menu can create Prompt, Embedding, Character, Reference, Video and Audio layers plus applicable core timeline blocks.
- Global scissors now split all unlocked blocks crossing the playhead, while the dedicated MultiClip `+ CLIP` action remains MAIN-only.
- Improved TAKE-panel scrolling/opacity and folder presentation so preview cards no longer bleed above the panel header.

## 0.6.42

- Restored a real high-VRAM/native H3 compute path: `memory_mode=normal` now preserves Sampler MLP/VRAM controls instead of silently forcing low-VRAM floors and caps.
- `mlp_chunk_tokens=0` now truly disables LongMedia MLP chunking; compatible `normal + existing + guards=0` runs can execute stock ComfyUI H3 DiT blocks without the LongMedia block wrapper.
- Manual MLP chunk values up to the public 131072-token limit now reach runtime in `normal`, while `low_vram` and `ultra_low_vram` retain bounded safety envelopes and still honor explicit zero/off controls.
- Corrected Sampler diagnostics so requested, policy-effective and runtime compute settings are distinguishable; inactive Sol controls no longer report Sol streaming/implementation as active.

## 0.6.41

- Fixed the monolithic Stage-2 Refiner AV device handoff: refined video is restored to the Stage-1 storage device/dtype before native AV repacking, while exact Stage-1 audio remains untouched.
- Added fail-fast AV device-contract validation around native latent packing to catch mixed CPU/CUDA stream ownership at the producing stage instead of promoting the whole AV latent to VRAM.

## 0.6.40

- Consolidated the Director 0.6 production line around semantic `t2va`, `fl2va`, `ref2va`, `hybrid` and `video_ref_edit` routing.
- Completed TAKE-centric Director persistence: full-state CREATE/Restore/Duplicate/Rename, active-draft render finalization, project folders and persistent resizable TAKE gallery.
- Added persistent Director UI layout/zoom state, MAIN drag reorder, Ripple behavior, segmented controls, MultiClip boundary markers and Director-owned resolution/aspect/megapixel policy.
- Fixed same-source FIRST+LAST endpoint geometry and strengthened Locked-Off/Static composition preservation.
- Fixed segmented short-cycle repetition by using exact frozen continuation tails plus deterministic per-segment seed offsets.
- Fixed chained Sampler motion-context layout for mixed visual/audio conditioning.
- Integrated Refine as internal Sampler Stage 2 using dedicated Refine Sigmas. Large/high-resolution passes can use bounded bidirectional overlap-save windows. The `windowed_refine` switch can force a single full-latent Stage 2 pass for continuity-sensitive shots.
- Refine is video-only: exact Stage-1 audio is preserved through Latent Hi-Res/Refine and is never re-denoised by Stage 2.
- Added native H3 VAE tile batching with OOM fallback.
- Hardened repeat-Queue VRAM lifetime: LongMedia transient CUDA references and FastH3/VSA geometry caches are execution-scoped, and Latent Hi-Res model residency is returned to CPU on all exit paths.
- Release package cleanup: stable version metadata, current bilingual EN/RU documentation only, exactly three maintained example workflows (Director, LatentUpscale-Detailer, SAFE-1080p-15s), no development audit scripts or per-build release-note clutter.

## 0.5.40

- Consolidated the validated 0.5 runtime line into a release package.
- Introduced the semantic Setup contract (`control_mode`, `h3_mode`, `timeline_mode`, `duration_source`, `audio_mode`) while preserving legacy workflow compatibility.
- Added the two-stage Latent Hi-Res / Refiner production workflow and sanitized example graphs.
- Corrected `video_ref_edit` audio ownership so the paired source soundtrack remains distinct from additional audio references.

Older development history is available in the Git repository history and tagged releases.
