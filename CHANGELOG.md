# Changelog

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
