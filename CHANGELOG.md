# Changelog

## 0.6.54

Public release delta from the GitHub `v0.6.50` baseline.

### Director selective regeneration and incremental append

- `Regenerate Clip` now keeps validated prefix/suffix TAKEs and samples only the selected GENERATED clip when the two-sided continuation contract is valid.
- `Regenerate From Here` keeps the validated prefix and samples the selected GENERATED clip plus its dependent GENERATED suffix.
- Appending a new GENERATED clip reuses the previous approved TAKE continuation; already-rendered clips are replayed from cache rather than sampled again.
- Setup conditioning is scoped to the bootstrap/required clips during selective runs so cached clips do not repeat text/reference encoding.

### Native GENERATED / MEDIA continuation

- Director BASE blocks are explicitly typed as `GENERATED` or immutable `MEDIA`.
- `GENERATED -> GENERATED` continuation uses the saved native TAKE latent/audio continuation state.
- `MEDIA -> GENERATED` creates a fresh H3 child target and VAE-encodes only the phase-safe external tail, attaching it as a native frame-0 `minimax_keyframes` guide. The full imported video never becomes target x0 and is never diffusion-sampled.
- External tail continuation is cached by media fingerprint, editorial trim range, geometry and overlap so repeated append/regeneration avoids re-encoding unchanged tails.
- Trimmed/reused MEDIA blocks resolve continuation from the end of the visible editorial range rather than the physical end of the source file.
- Mixed final assembly decodes only GENERATED runs; immutable MEDIA RGB/audio is spliced directly into the final timeline.
- Per-clip audio continuation supports `AUTO`, `CONTINUE` and `FRESH`.
- Mixed mono/stereo Director audio is normalized to canonical `[1,C,L]`; the only channel conversion is exact mono-to-stereo duplication when required.

### Quantized runtime, Windows and memory

- AUTO memory routing for INT8/W4A8/NVFP4 uses packed physical storage plus activation headroom instead of BF16-equivalent logical model size, avoiding false `ultra_low_vram` demotion on 16 GB-class GPUs.
- Explicit user sampler settings remain authoritative; NORMAL/AUTO-normal no longer silently clamps requested `mlp_chunk_tokens` to legacy low-VRAM values.
- Windows large-safetensors loading preserves ComfyUI `ModelMMAP` / `TensorFileSlice`, quantization metadata and AIMDO/DynamicVRAM behavior instead of eagerly materializing 20–30 GB checkpoints in RAM.
- Pinned H2D admission is based on projected available host RAM rather than a fixed fraction heuristic.
- Setup releases text-encoder/transient conditioning references before diffusion sampling while intentionally avoiding aggressive Windows `EmptyWorkingSet` trimming and file-cache churn.

### Preserved contracts

- Refiner remains video-only; exact Stage-1 audio is preserved and Stage-2 audio noise/mask stays frozen.
- Existing Director NLE/TAKE behavior, camera/embedding timing, DynamicVRAM/AIMDO integration and user-authored sampler controls are preserved.
- SPEED / progressive-resolution is intentionally not part of this release.

## 0.6.53

### LongForge-style external MEDIA continuation (Thanos root-cause fix)

- Replaced the 0.6.51/0.6.52 pseudo-`previous_av` external-video handoff with a fresh child H3 target plus a native `minimax_keyframes` guide at frame 0, matching the stock `MiniMaxH3AddGuide`/LongForge continuation contract.
- Imported `MEDIA` is never copied into target x0, never noise-masked as an old diffusion prefix, and never becomes an H3 render unit. Only the phase-safe tail window is VideoVAE-encoded.
- External continuation cache semantics bumped to v2 so stale 0.6.51/0.6.52 tail entries cannot be reused under the new guide contract.
- Fixed reused/trimmed MEDIA sources: continuation tail and final assembly now resolve the same editorial `source_in`/`duration` boundary instead of accidentally using the physical end of the source file.
- Preserved the 0.6.50 GENERATED -> GENERATED native TAKE path and selective regeneration semantics unchanged.
- Added render-unit and static continuation regressions for GENERATED chains, MEDIA -> GENERATED chains, MEDIA -> GENERATED -> MEDIA -> GENERATED, native frame-0 handoff, and tail-only VAE guards.

## 0.6.52

- Fixed the Director empty-TAKE import path for Thanos continuation: dropping a Video onto a newly-created or content-empty MAIN BASE block now promotes that block to immutable `MEDIA` immediately.
- Prevents the imported prefix from remaining mislabeled as `GENERATED`, which previously made selective regeneration treat the missing TAKE as invalid and restart sampling from clip 1.
- Non-empty GENERATED clips retain the existing video-reference behavior; explicit BASE TYPE selection remains authoritative.

## 0.6.51

- Director continuation contract: immutable `MEDIA` and H3 `GENERATED` BASE blocks.
- GENERATED → GENERATED append reuses cached TAKE/native latent continuation and samples only the new clip.
- MEDIA → GENERATED append encodes only the external tail window; the source media is never diffusion-sampled or full-video VAE round-tripped.
- Mixed final assembly decodes GENERATED runs only and splices immutable MEDIA RGB/audio directly.
- Added `AUTO` / `CONTINUE` / `FRESH` audio-continuation metadata and external-tail continuation cache.
- Preserved 0.6.50 selective-regeneration, AV/refiner, DynamicVRAM/AIMDO, and user-authoritative sampler contracts.

## 0.6.50

- Restored quantized NORMAL throughput after selective-regeneration profiling: legacy AUTO reserve/late-guard/step-cleanup defaults are geometry-calibrated on <=18.5 GiB GPUs, explicit MLP chunk requests remain authoritative, new Samplers default to 24576-token MLP chunks, and first-forward diagnostics report actual loaded/offloaded residency.
- Replaced the RAM-eager Windows large-safetensors workaround with ComfyUI file-backed `ModelMMAP`/`TensorFileSlice` loading whenever available; this avoids both the native `safe_open` access violation and pre-Setup full-RAM materialization. Automatic `EmptyWorkingSet` trimming is disabled so Windows can manage clean file-backed cache normally.
- Pinned H2D admission now uses projected available host RAM instead of rejecting the common ~20 GiB INT8 H3 solely because it exceeds 25% of a 64 GiB workstation, restoring threaded/pinned streaming when real RAM headroom is healthy.
- Fixed AUTO memory routing for quantized H3: TensorWise INT8/W4A8/NVFP4 now route from packed storage footprint plus activation margin instead of BF16-equivalent logical `model_size`, preventing false `ultra_low_vram` demotion and 2048-token MLP chunking on 16 GB GPUs.
- Reduced Setup host-RAM pressure for Director selective regeneration: cached clips skip redundant TE/control encoding, the text encoder stays resident across the bootstrap+required continuation encodes instead of being unloaded/reloaded mid-Setup, then TE residency/pins are explicitly released before diffusion sampling; host-RAM diagnostics remain while automatic Windows working-set trimming is disabled.
- Reworked the Windows large-safetensors crash workaround so large checkpoints stay file-backed through ComfyUI ModelMMAP/TensorFileSlice whenever available; RAM-eager pread/raw reads are last-resort only and quantization metadata remains preserved.
- Added selective-regeneration geometry diagnostics so a regenerated Director clip reports its actual local AV latent span instead of leaving full-timeline work ambiguous.
- Made Director rendering incremental at clip level: normal Queue reuses approved cached prefixes, appending a new MAIN clip samples only the new dependency suffix, historical one-shot TAKEs can seed the first MultiClip append when their exact geometry/semantics still match, and reused parent seam metadata is promoted for later clip-only regeneration.
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
