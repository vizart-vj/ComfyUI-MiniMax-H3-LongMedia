# v0.3.0 — 2026-08-13

First public 0.3.x release centered on the validated single-pass MiniMax H3 workflow surface.

Highlights:
- `hybrid_auto`, `ref2va_full`, `loop`, `video_ref_edit`, and `manual` workflow modes.
- Loop parity mode duplicates `image_1` internally into both first/last anchors while keeping prompt-driven motion.
- `generation_mode=lip_sync` is visible in the normal UI and supports audio-driven lip synchronization.
- `audio_mode=preserve_reference` lets H3 react to rhythm/timing/lip-sync while restoring the untouched source soundtrack at output.
- Conservative AUTO SOL tau policy (`1.30 -> 1.85`, capped near `2.0`) for improved motion/temporal fidelity.
- Live Sampler `auto/manual` switching without page reload.
- Dynamic mode-aware image/video/audio socket labels.
- Release schema audit tooling and compatibility migration for older serialized workflows.

# 0.3.0 generation-mode UI hotfix
- `generation_mode` is now always visible in Setup instead of being hidden outside Manual mode.
- Selecting `lip_sync` immediately reveals its first-frame controls without a page reload.
- Public-mode sanitization no longer forces `generation_mode` back to `auto`.
- Lip-sync socket labels now identify `image_1` as the identity anchor and `audio_1` as the driving track.
- Existing backend compatibility remains unchanged: lip_sync uses the native Ref2VA/audio-reference path and cannot be combined with hybrid first/last conditioning.

# 0.3.0 preserve-reference audio hotfix
- Added `audio_mode=preserve_reference` (appended to the combo for serialized workflow compatibility).
- The original audio is used as H3 reference/driver for rhythm, timing and lip-sync, while generated audio is discarded and the untouched source track is restored at output.
- `generation_mode=lip_sync` remains supported; `lip_sync + preserve_reference` uses `audio_1` as the driving track and restores that exact source audio after video generation.
- `preserve` is separated from reference intent where the conditioning route permits it.

# 0.3.0 AUTO SOL tau quality-safety hotfix
- Lowered AUTO SOL base schedule from `1.70 -> 2.10` to `1.30 -> 1.85`.
- Reduced long-sequence geometry boost from max `+0.30` to max `+0.15`.
- Geometry boost slope is reduced from `0.22 / 60k tokens` to `0.12 / 60k tokens`.
- AUTO SOL now tops out around `1.45 -> 2.00` instead of `2.00 -> 2.40`, prioritizing motion and temporal fidelity over the most aggressive sparsity.
- Manual SOL controls remain unchanged (`1.30 -> 0.80` defaults).

# 0.3.0 sampler live-manual hotfix
- Fixed Sampler `auto -> manual` UI switching on ComfyUI frontend builds that replace combo callbacks after node creation.
- Added a lightweight value watcher for `sampler_mode`, so expert controls appear immediately without reloading the page.
- Watcher is cleaned up when the node is removed and does not modify backend widget serialization/order.

# 0.3.0 consolidated UI + loop parity hotfix
- Loop mode now mirrors the proven hybrid_auto same-image i1+i2 behavior.
- Loop socket labels: image_1=first+last_frame, image_2=reserved/ignored, image_3+=Picture refs.
- Consolidated on top of the UI hotfix so workflow-mode socket labels and Sampler manual expansion are both present.

# 0.3.0 release audit hotfix
- Removed the duplicate `prompt_input` socket from Long Media Setup; the single `prompt` field remains widget-connectable and legacy workflows are migrated automatically.

- Audited the public Setup / Sampler / Decode schemas against their Python execution signatures.
- Fixed stale or shifted combo values (including numeric `conditioning_mode=3`) being submitted to ComfyUI validation.
- Auto mode now forces every hidden legacy sampler widget to a validator-safe production default before queueing.
- Manual sampler mode reliably restores the complete low-level tuning UI without changing serialized widget order.
- Setup public modes sanitize hidden legacy conditioning/segmentation controls; Manual preserves valid user values.
- Decode remains a two-input public node (`final_av`, `long_media_plan`) for model data; video/audio VAEs are read from the plan.
- Added `tools/release_schema_audit.py` so schema/function drift and invalid combo defaults are caught before packaging.

# 0.3.0

## 0.3.0
- UI fix: workflow_mode now updates dynamic socket labels (first_frame/last_frame/source_video/etc.) in the Setup node.
- UI fix: Sampler `manual` mode now forces expert widgets to reappear reliably after mode changes and workflow reloads.

- Added public `workflow_mode=video_ref_edit` for clearer ref-driven video recasting: `video_1` is the main motion/camera source while `image_1..9` stay ordinary Picture refs for character/style replacement.
- Updated Setup socket tooltips and frontend value repair to understand `video_ref_edit`.

- Hotfix: remove legacy Decode `video_vae` / `audio_vae` sockets on load; Decode uses VAEs stored in `LONG_MEDIA_PLAN`. hotfix — serialized widget compatibility

- Restored the legacy Setup/Sampler widget ordering and append the new release-mode widgets instead of inserting them mid-schema.
- Added automatic repair for workflows saved with the first 0.3.0 build where primitive widget values were shifted by one slot.
- Auto/manual UI hiding is frontend-only; backend input schemas remain stable for old workflows.


- Release-candidate UI centered on the validated single-pass H3 engine.
- New `workflow_mode`: `hybrid_auto`, `ref2va_full`, `loop`, `manual`.
- `hybrid_auto`: image_1 first frame; optional image_2 last frame; remaining images are Picture refs.
- `ref2va_full`: all connected images remain native Ref2VA Picture references.
- `loop`: image_1 is encoded as both first and last frame; image_2..9 remain Picture refs.
- Normal modes disable experimental segmentation/continuation; legacy controls remain available in Manual.
- Sampler gets `auto` vs `manual`; Auto freezes the validated production attention/VRAM policy while Manual exposes all tuning.
- Native video/audio reference sockets are retained for the 0.3.x validation cycle.

## V63 STORYBOARD BRIDGE
- Added `conditioning_mode=storyboard_bridge` two-pass prototype.
- `image_1` = panel A, `image_2` = exact shared bridge panel B, `image_3..9` = refs.
- Panel B is fitted and VAE-encoded once; the exact same latent anchors pass 0 end and pass 1 start.
- Storyboard mode disables LongMedia latent overlap, V60 motion context, and temporal-offset continuation logic.
- Decode removes the duplicated boundary frame and the matching audio slice.

## V62 - same-seed + hidden-preroll timeline correction

- Reuse the same base seed for every pass of one long-media shot.
- Treat segment_starts as context-window origins; prompt-visible time starts after overlap.
- Explicit `Continue directly from the preceding video...` sections are mapped one-per-pass with local timestamps preserved.
- Stitch overlap remains hidden (existing latent stitch already removes it).

# V61 — Continuation Identity Sheet

- Continuation passes collapse 2+ hybrid image references into one combined identity sheet.
- Pass 0 keeps the original individual H3 image references unchanged.
- Continuation Picture tags are remapped to `<Picture 1>` so H3 sees one visual reference block instead of competing per-subject refs.
- V60 true motion-context head guides and frozen latent overlap are preserved.
- CLIP/TE remains Setup-only; no text encoder object is retained in LongMediaPlan.
- Mixed video/audio reference workflows fall back to V60 semantics for this prototype.

## 0.2.49 - Segment timeline semantics fix

- `segment_seconds` now means new output timeline per pass.
- `overlap_frames` is additional continuation context instead of reducing useful segment duration.
- 15 s / 5 s now plans exactly 3 passes rather than 3 full passes plus a short tail pass.
- H3 temporal alignment is preserved and final stitched output is trimmed to the requested duration.

# V48 Cumulative Hybrid Continuity + PR#3

- Consolidated the current production test line into one build.
- Includes standalone Hybrid conditioning inside Long Media Setup.
- Includes hybrid segment continuity fix: first-frame anchor only on pass 0; intermediate passes continue from inherited overlap/context; last-frame anchor is reserved for the final pass in hybrid_first_last mode.
- Includes PR #3 routing fix for image refs + audio + no source video on auto_refs: keep NativeReferenceToVideo instead of falling into audio_to_video.
- Includes PR #3 bounded audio latent copy and source-device preservation.
- Keeps the established V40 SOL / INT8 / W4A8 execution path unchanged.

# Changelog

## v0.2.46

### Fixed
- Fixed video-reference setup crash: `Boolean value of Tensor with more than one value is ambiguous`.
- Optional video/audio Tensor inputs are now tested explicitly with `is not None` instead of Python truth-value coercion.
- Covers both `source_video` and `source_audio` assignments in the video-to-video plan path.

## v0.2.45

- Fixed Long Media Setup on current ComfyUI builds where `CLIP.encode()` no longer accepts the legacy `control` keyword.
- Prompt encoding now uses ComfyUI's canonical `encode_from_tokens_scheduled()` API with legacy fallbacks.
- Clarified video/audio reference inputs: `video_N` carries frames only; extracted source audio should be connected separately to the corresponding `audio_N`.

## Project rename

- Public repository/package name: `ComfyUI-MiniMax-H3-LongMedia`.
- Legacy internal node class IDs are intentionally preserved for workflow compatibility.

## 0.2.44

Current SAFE / long-generation baseline.

- Added adaptive Embedded Sol OOM retry.
- OOM retries reduce streamed QKV chunk size before giving up.
- Prevented Sol OOM from falling directly into generic attention/NVFP4 dequantization fallback.
- Successful smaller retry chunks persist for later blocks/steps.

## 0.2.43

- Added streamed final-output layer.
- Avoids creating the full video-target FP32 hidden tensor after transformer block 49.

## 0.2.42

- Added late-block hard VRAM guard.
- Added step-boundary cleanup.

## 0.2.41

- Fused and chunked the full second transformer half:
  `norm2 -> modulation -> MLP -> gate -> residual`.

## 0.2.40

- Fused chunked MLP + gate + residual path.

## 0.2.39

- Added a separate cooldown for emergency inter-block VRAM trims.

## 0.2.38

- Added denoise step-boundary profiling.

## 0.2.37

- Added adaptive inter-block VRAM guard.

## 0.2.34

- Added setup-stage CLIP/Qwen/reference VRAM isolation.

## 0.2.32

- Added compressed streamed K/V storage for long Sol sequences.

## 0.2.30

- Added chunked Sol output projection and early QKV release.

## 0.2.27

- Added embedded/adapted Sol-Attn modes.

## 0.2.24

- Added token-axis MLP chunking.

## 0.2.21

- Reworked frontend dynamic-input handling without mutating ordinary widgets.

## V24 INT8/W4A8 hot-reload final-layer A/B
- Combined the V22 0/24/49 transformer stage probes with V23 development hot reload.
- Added bounded post-transformer FinalLayer A/B probes for video/audio norm+AdaLN and output heads.
- Reference diagnostics sample at most 16 rows and never rebuild the full FP32 final hidden tensor.

### 0.3.0 hotfix — Manual sampler UI
- Fixed `sampler_mode=manual` not restoring the hidden legacy sampler controls on some ComfyUI frontend builds.
- Auto/Manual visibility now watches the actual mode value during node drawing, so it updates immediately even when the frontend does not invoke the combo widget callback.