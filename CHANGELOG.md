# Changelog

## 0.6.60

### Director, RefMod, MultiClip and documentation

- Added Sampler presets to the top of the node. Users can create, overwrite and delete workflow-persisted presets; the status line reports changed settings. Seed stays independent and is neither applied nor compared.
- Motion Repair now survives workflow reloads: the Setup UI can reorder widgets for presentation, but workflow serialization writes their values in the original Python `INPUT_TYPES` order. Build label: `director-motion-repair-toggle-persist`.
- MultiClip Latent Hi-Res + Refine now seeds each generated clip's hidden overlap from the preceding clip's refined high-resolution tail and freezes that prefix during Stage 2. This gives the Refiner consistent geometry/color context across clip boundaries while keeping Stage 1 continuation, clip prompts, final single VAE decode, and exact audio passthrough unchanged. Build label: `director-upscale-refine-multiclip-seam-fix`.
- Character RefMods now keep identity authority across MultiClip boundaries: on target-video queries only, their visual attention share is balanced against longer competing visual-reference streams, while audio and other RefMod concepts keep their existing routing. Character guidance distinguishes identity from motion and scene inherited from video context. Build label: `director-refmod-character-multiclip-balance`.
- RefMod Inspector now uses the upstream extraction controls and meanings: downscale-only short-edge resolution before VAE encode, visual token cap, aspect-fitted grid long edge, mode-aware clip-frame limit, audio-only voice seconds, and Compressed-only refinement steps. ENCODE applies these before/after the corresponding VAE steps; legacy `spatial_grid` and `temporal_frames` documents migrate into the clearer controls. Build label: `refmod-upstream-controls`.
- Director shows BASE VIDEO SCALE only when the BASE clip owns a Video reference; an old explicitly saved non-default scale remains visible as EXTERNAL VIDEO 1 SCALE. VIDEO LAYER SCALE appears only on a layer clip with an assigned Video. Empty BASE clips at the default 100% no longer force a shared video slot back to full resolution when another clip asks for a smaller scale. Build label: `director-scale-ownership`.
- Director now releases detached timeline video-strip decoders on every full UI refresh, unloads old TAKE gallery thumbnails when switching views, limits decoded preview-image retention to eight LRU entries, and reuses the Program Monitor video element instead of replacing it. TAKE gallery thumbnails load lazily. The global layer-height slider now sits beside timeline zoom and Fit. Build label: `director-preview-memory`.
- Director timeline gap controls now render only inside real gaps, show the gap duration, fill the gap with a new clip, or close it while ripple-shifting later clips. Timeline zoom controls are grouped beside playback, and Fit resets horizontal scroll before fitting the entire timeline. The context menu now has distinct left/right cut-to-cursor actions; Knife remains a split. Trimmed media edges show directional markers with source-time details on hover.
- Fixed Director timelines made entirely of immutable MEDIA clips. They now bypass H3 sampler/model preparation and go directly to the existing media assembly path; previously the sampler skipped every clip but left `stitched` unset, then raised `Unified LongMedia runtime produced no segment output`.
- Fixed Director file drops by removing root capture-phase interception that could preempt native WHO & WHAT and subject-card handlers. Media drops on Character, Reference, Video and Audio layers now route to that layer directly and create a clip when the target time has no clip.
- Director `Concept Type` now adds type-specific positive-prompt guidance to each active RefMod interval (Description remains metadata). Global Prompt adds scene-wide selectors for style, atmosphere, palette, lighting, texture and era, with 15 built-in choices per category and custom presets saved inside the workflow. Build label: `concept-guidance-presets`.
- Added a dedicated CREATE FROM SOURCES RefMod editor for up to 128 image/video/audio sources, per-source selection/crop, clipboard image paste and token estimates before ENCODE & CREATE.
- TAKE gallery now offers explicit sort choices and preserves its scroll position while switching the selected take.

### Attention backend routing

- RefMod and temporal routes now check the final merged attention mask for identity before running the dense masked path. When every hidden key is restored at unit weight, the mask changes no logits, so the sampler keeps the selected attention backend, including zero-copy SLA. This uses the actual H3 route instead of comparing authored duration with H3's frame-cell count.

## 0.6.59

- Fixed the main LongMedia sampler dropping H3 SLA's `optimized_attention_override` when cloning `guider.model_options`. It now inherits the ModelPatcher-owned override unless the guider already has its own; this lets the existing SLA fast path receive the actual hook and avoids silently selecting dense/SOL attention because the sampler clone lost it.

## 0.6.58

- Full-span RefMods at exact native strength now keep the selected native H3 attention backend, including zero-copy SLA. Their visibility mask is an identity operation when every target frame sees every RefMod key; only partial windows, reduced strengths or combined temporal text controls use the masked exact path. This avoids the dense masked-attention slowdown for whole-clip references.
- RefMod Inspector now explains that Concept Type adds broad positive-prompt guidance while Description remains library metadata. It recommends entering specific effects in the Global Prompt or shot prompt, and ENCODE after changing artifact settings.

## 0.6.57

- Director **SYNC** now reconciles the RefMod panel with the shared library instead of unioning into it: records whose `.safetensors` no longer exists are dropped from the document, so emptying `output/longmedia_refmods` and pressing SYNC actually empties the panel. Unsaved drafts (no `path`) are kept, survivors are refreshed, and the button reports `+added ~updated −removed` plus any timeline clips left pointing at a removed RefMod. `Ctrl+Z` restores the previous list.
- RefMod library listing failures are no longer silently converted into an empty library (`refmodList` swallowed every error and returned `[]`, which made SYNC a no-op on installs where the backend request failed). SYNC now aborts with "Nothing was changed" and the panel never loses state on a transport/server error.
- A Director RefMod layer clip pointing at a RefMod that is no longer in the document is now a hard Setup error naming the id and time window, instead of being silently skipped. The silent skip is what made a render look exactly as if the RefMod were never applied (SYNC already reported `N timeline clip(s) reference a removed RefMod`, and Setup said nothing). Setup also prints what it packed: `[DIRECTOR REFMOD] packed N block(s): name#member kind HxW a.bs..b.bs str 1.00`, so "is the RefMod actually conditioning this render?" has a console answer instead of a guess.
- Removed the dead `Resolution` and `Max tokens` fields from the RefMod record and inspector. Neither ever reached the encoder (`_lm_refmod_encode_record` consumes only `spatial_grid`, `temporal_frames` and `refinement_steps`), the router, or any VAE/descriptor path — `resolution` does not appear in `nodes.py` / `refmod_backend.py` / `refmod_routing.py` / `__init__.py` at all, and `max_tokens` there only in unrelated sampler memory budgeting. They were normalised in three places (`defaultRefmod`, `normalizeRefmod`, `director_plan._normalize_refmod`) and consumed nowhere, which is exactly what made them look meaningful in the inspector. The inspector now states which four inputs actually shape the artifact. Old documents carrying the keys are unaffected: normalisation simply drops them. A regression guard keeps them out.
- Fixed Director RefMod render aborting with `shape '[1, 24, 1, 1, 30, 2, 20, 2]' is invalid for input of size 59040`. Host `comfy/ldm/minimax/model.py:42 patchify_video` rewrites the latent as `(b, c, t, pt, h, ph, w, pw)` with `h = H // 2` / `w = W // 2` and then reshapes to exactly `h*ph` by `w*pw` columns, so an odd latent H or W loses a row/column to floor-division and the reshape dies inside the H3 forward (`_cond_video_rows`). The failing latent was exactly `(1, 24, 1, 60, 41)`. RefMod sources are arbitrary user media and `encode_full` stored their VAE latents as-is (`RefMod.__post_init__` only checks `[1, 24, T, H, W]`, and `pool_latent` enforces evenness only for `COMPRESSED` pool targets — FULL mode enforced nothing), so an odd latent width sailed through to the host and blew up far from anything RefMod-shaped. `convert_to_native_block` now trims the visual latent onto the 2x2 DiT patch grid and reports `latent_h`/`latent_w` from the tensor actually handed to the host. Existing artifacts work without re-encoding.
- Fixed Director RefMod render aborting with `Director RefMod descriptor mapping failed: RefMod descriptor identity not found in native minimax_refs`. `convert_to_native_block` emits only the native descriptor schema (`kind`/`latent_h`/`latent_w`/`latent`/`latent_t`/`ref_audio_t`/`audio_latent`) and host `PackedLayout(refs=...)` contracts to that schema alone, so the `longmedia_refmod_*` identity annotations were outside every host guarantee and got dropped on the `minimax_refs` → CONDITIONING → `payload['refs']` path. Both recovery layers above the authored specs read that same runtime container, so the one identity-complete source (`spec['native_block']`) was checked last and never reached. Descriptors are now re-stamped from the authored specs before `resolve_refmod_spec_index` matches on them, using the same `native_ref_count + appended_block_index` layout contract `_lm_localize_refmod_specs` already uses for `ref_index` (host native refs precede RefMod refs in `minimax_refs`).
- RefMod cards with no library artifact are labelled `DRAFT`.
- Director: a library RefMod can now be **placed on the REFMOD layer** (`PLACE ON LAYER` puts it at the playhead) and the clip inspector gained `↔` to stretch it across the whole timeline. Previously the only way to make an encoded RefMod affect the render was to drop its `.safetensors` onto the timeline, which re-imported it and duplicated the library entry.
- Director **ENCODE** now reports itself on the ComfyUI console: `start` (with the parameters it will use), `stage · VAEs loaded`, `stage · latents encoded`, then `done · <artifact> · N member(s) · X.XXs`, or `FAILED · <reason> · X.XXs`. ENCODE produces no sampler progress bar, so without these lines a finished run was indistinguishable from a stalled one.
- Fixed the RefMod inspector resetting its settings after **ENCODE** / **SAVE**: the response was a disk projection (`_refmod_record` rebuilds a record from the artifact's saved metadata and carries no `resolution`/`spatial_grid`/`temporal_frames`/`refinement_steps`/`max_tokens`/`description`), and the client fed it through `normalizeRefmod`, which re-defaulted every missing key to `1920x1080` / `32x32` / `24` / `20` / `512` / `1.00`. `spatial_grid`, `temporal_frames` and `refinement_steps` are real encoder inputs (`_lm_refmod_encode_record`), so this silently changed the encoded artifact too. Save/ENCODE now answer with the authored record and overlay only the artifact-owned fields (`path`, `folder`, `state`, `kind`, `format_version`, `members`), matching what the pending-encode branch already did; the client merges through `applyRefmodServerState`, so no merge site can re-default an authored setting.

- Director **ENCODE** is now a separate VAE-only RefMod preparation operation (button in the REFMODS inspector, `POST /longmedia/refmods/encode`): it loads only the MiniMax H3 VideoVAE/AudioVAE and writes the latent artifact. Text encoder, DiT, sampler and Long Media Setup are never touched.
- Long Media Setup no longer lazy-encodes RefMods on the render path. An unprepared RefMod aborts with a clear "is not encoded yet … press ENCODE" message instead of pulling the VAE into the execution log.
- Fixed `Director RefMod routing has authored specs but no native minimax_refs descriptors.`: native `minimax_refs` descriptors are now recovered through a layered fallback (side channel → `PackedLayout.ref_blocks` → descriptors embedded in the authored specs), and when the host CONDITIONING hop drops them entirely the exact ordered descriptors are re-injected into `minimax_payload` with matching `cond_video_latents`/`cond_audio_latents` before `PackedLayout` packs the forward.
- RefMod-only `inactive` WHO & WHAT pictures can be used as a RefMod source. They are loaded as Director RefMod media (`loaded_as='refmod_source'`) without becoming normal H3 image inputs and without exposing unrelated inactive pictures.

- RefMod-only Picture sources are now loaded separately from ordinary H3 image references, so an inactive WHO & WHAT image can be encoded into a RefMod without becoming an unintended prompt reference.
- RefMod routed attention now dispatches through ComfyUI's selected mask-aware attention backend rather than forcing the PyTorch SDPA path.

- Fixed a duplicate block-scoped `files` declaration in the Director drop handler that prevented the Director frontend module from loading.
- RefMod storage is now a shared catalog at `output/longmedia_refmods`, with folder grouping independent of Director project IDs; existing UUID-based project folders migrate on library sync.
- Reworked the RefMod Strength slider so its value readout tracks the thumb and model updates commit only after the drag ends; slider constraints are now applied before the initial value to prevent integer-step snapping.

## 0.6.56

### RefMod library workflow

- Added folder creation and folder-filtered RefMod library browsing.
- RefMod `.safetensors` files can be imported by dropping onto the timeline (creates a RefMod layer) or an existing RefMod layer; imports use a dedicated upload route and are validated before entering the library.
- RefMod inspector is single-selection/collapsible, with a tidied two-column layout and full-width horizontal Strength slider.
- Track labels and timeline rows now share vertical scroll position.

## 0.6.55

### Director-integrated RefMods

- Added a collapsible `REFMODS` library inside Director with Character, Reference, Image, Video, Audio, selected timeline clip and TAKE sources.
- Added MiniMaxH3Mod-compatible v4 standalone and v5 bundle persistence, including `refmod_meta`, `ref_0...`, legacy `latent`, FULL and COMPRESSED modes.
- RefMods use the already-loaded H3 VideoVAE/AudioVAE and attach as native unlabelled `minimax_refs`; no extra model or manual VAE connection is required.
- TAKE-derived RefMods consume cached native visual/audio latents without decode/re-encode when available.
- Authored RefMod intervals gate native reference keys on the target H3 frame lattice. Native reference RoPE cursors are never interpreted as Director time.
- Temporal text embeddings and RefMods share one query-axis attention partition while remaining independent controls.
- Selective regeneration skips RefMod loading/VAE work for cached clips; GENERATED/MEDIA continuation and TAKE state remain separate and unchanged.
- Windowed Stage-2 refine and motion-repair passes now rebase every RefMod descriptor, preserving inactive zero-width gates so native refs cannot fall back to accidental full-span visibility.
- Added stable identity-based native block routing and an authored/native target-clock assertion to fail loudly on packed-layout drift.

### Preserved contracts

- GENERATED → GENERATED cached/native TAKE continuation, MEDIA → GENERATED tail-only VAE/native keyframe continuation, trim-aware tails, mixed mono/stereo assembly, exact Stage-1 audio through Refine, and PDD multi-head output remain intact.
- Director document schema is version 10; TAKE snapshots include RefMod catalog, bindings, intervals and strengths.

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
