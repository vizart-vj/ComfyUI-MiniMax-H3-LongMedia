# Changelog

## 0.5.40 — release consolidation since 0.4.40

- Promoted the validated 0.5.38 runtime line to a release package; H3 sampling math is unchanged by the 0.5.40 cleanup itself.
- Re-audited current documentation against the semantic Setup contract (`control_mode`, `h3_mode`, `timeline_mode`, always-visible `duration_source`) and removed stale 0.4.x production guidance from current guides.
- Added operating-mode and two-stage Latent Hi-Res / Refiner guides.
- Added 0.5.40 release notes covering the development line since public v0.4.40.
- Added a sanitized Latent Upscale / Detailer example workflow with user media, saved previews, and local output paths removed.
- Fixed `video_ref_edit` multi-audio ownership: Audio1 alone owns preserved/passthrough source audio, while Audio2/Audio3 remain prompt-conditioning references instead of being mixed into the final soundtrack.
- Preserved historical release notes and legacy `workflow_mode` migration for old workflows.
- Package identity remains GitHub `vizart-vj/ComfyUI-MiniMax-H3-LongMedia` and Comfy Registry `PublisherId = "noise"`.

## 0.5.38-DEV — bounded host pinning + guarded INT8 prefetch on 16 GB

- Fixes a policy inversion where recent AIMDO native INT8 re-enabled `prefetch_dynamic_vbars` on <=18.5 GB GPUs even though the same runtime had already classified them as low-VRAM guarded streaming.
- Native tensorwise INT8 on constrained GPUs now uses one-block demand residency, preventing the speculative second VBAR/cast destination that appeared as large VRAM `other` usage and could OOM before block 0 on repeat runs.
- Full-model pinned host-memory fastpath is now RAM-pressure aware. A 19.5 GB H3 model is not page-locked wholesale when it would consume >=25% of physical RAM or leave less than a healthy post-pin reserve. Existing model pins are released before sampling in that case.
- 128 GB+ systems with sufficient available RAM can still retain the native pinned-transfer fastpath.
- H3 weights, attention math, conditioning, audio/video logic, seeds and output quality are unchanged.

## 0.5.37-DEV — repeat-run sampler memory boundary

- Fixes seed-only / sampler-only reruns that could OOM before block 0 after a successful previous generation.
- Adds a hard Sampler execution boundary before every `prepare_sampling()` call: synchronize CUDA, cleanup prefetch queues, reset cast buffers, reset AIMDO/VBAR watermark limits, unload registered models, and release dead allocator cache.
- The boundary runs even when ComfyUI reuses cached Setup outputs, so a rerun starts with the same residency conditions as the first generation.
- No H3 math, conditioning, attention, audio/video semantics, seed derivation, or output quality path is changed.
- Sampler report now includes `sampler_execution_boundary` with before/after allocator counters and performed lifecycle events.

## 0.5.36-DEV — independent duration ownership + multi-audio video_ref_edit conditioning

- `duration_source` is now a first-class Setup control in both Auto and Manual control modes; the widget stays visible instead of being hidden behind `control_mode=manual`.
- In `video_ref_edit`, `duration_source=auto` still resolves to `video_1`, while explicit `video`, `audio`, `manual`, and `longest_input` selections are preserved exactly.
- Explicit shorter target horizons crop the active Video1 reference window; longer target horizons keep the full Video1 reference and create a fresh longer H3 target so the scene can continue naturally beyond the source clip.
- Timeline ownership, final-audio policy, and prompt/audio conditioning are now independent contracts. Changing `duration_source` never removes connected audio references from H3 conditioning.
- `video_ref_edit + lip_sync` now treats Audio1 as an independent authoritative redub/timing source instead of falsely pairing arbitrary replacement speech with Video1's original facial performance.
- `video_ref_edit + auto/preserve/preserve_reference` retains the paired native Video1+Audio1 source-performance block and the locked target-audio clock.
- Audio2/Audio3 remain standalone native H3 `<Audio 2>` / `<Audio 3>` references in `video_ref_edit`, including preserve-style modes. Prompts can address multiple connected tracks independently from timeline/output policy.
- Lip-sync keeps Audio1 as the sole timing/output authority while Audio2/Audio3 remain prompt-addressable semantic/music references.
- Added timeline ownership and conditioning-audio counts to the Setup report.
- Updated user documentation with duration-source behavior, arbitrary redub, continuation, multi-audio prompting, and audio-reactive examples.

## 0.5.35-DEV — video_ref_edit paired source AV / tail-sync fix

- Fixed `video_ref_edit` source-performance sync fading before the source soundtrack ended.
- Root cause: the single-pass edit route fell through to generic V2V target initialization; `image_1..9` were not native Picture refs there, and `audio_1` in `preserve` was not presented as the soundtrack paired with `<Video 1>`.
- `video_ref_edit` now uses the upstream MiniMax H3 native paired reference contract: `ref_video_audio_0 + ref_video_0` form one `video_audio` reference block, preserving their shared temporal span.
- `audio_mode=auto/preserve/preserve_reference` keeps the paired reference and also freezes Audio1 into the target audio stream, so source AV transfer and absolute target timing are both present.
- `reference_only` and `lip_sync` also pair Audio1 with Video1 when available; `generate` keeps input audio out of the source pair.
- `duration_source=auto` in single `video_ref_edit` now resolves to the `video_1` timeline, avoiding final-performance truncation from tiny audio/container duration differences.
- Strengthened the positive sync instruction to carry articulation through the final audible phoneme before settling.
- Added report field `video_ref_edit_paired_source_av`.

## 0.5.34-DEV

- Fixed `video_ref_edit` source-performance sync for `audio_mode=auto`, `preserve`, and `preserve_reference` when `audio_1` is connected. Preserving the waveform at final mux is no longer the only behavior: the exact source-audio window is now encoded into the target AV stream and frozen as the authoritative timing clock while replacement video is regenerated.
- `video_ref_edit + preserve/preserve_reference` now fails early when `audio_1` is missing, with an explicit reminder that `video_1` carries IMAGE frames only.
- Added a positive source-performance continuity prompt for replacement facial articulation, speech/singing timing, breathing rhythm, expression timing, and body timing.
- Explicit `lip_sync` remains unchanged; the new implicit source-AV lock is scoped only to `video_ref_edit` paired-source preservation modes.
- Updated Setup input labeling, tooltips, README, and `docs/AUDIO_MODES_GUIDE.md` to document the synchronization contract.

## 0.5.33-DEV

- Added English quick-start documentation for Long Media Cameras and MultiClip.
- Added a dedicated MultiClip prompting guide with camera-ownership and continuity examples.
- Added `docs/AUDIO_MODES_GUIDE.md` documenting source-audio connection requirements, especially `video_ref_edit`: `auto` accepts a disconnected `audio_1`, while preserve-style and lip-sync contracts require connected source audio.
- Clarified that `video_1` is an IMAGE batch and never carries the source movie soundtrack.
- Added `docs/README.md` documentation index and linked the new guides from the root README.
- Runtime, sampling, H3 math, memory policy and node contracts are unchanged from 0.5.32-DEV.

## 0.5.32-DEV

- Fixed sampler aborts caused by ComfyUI-VideoHelperSuite animated latent previews receiving MiniMax H3/TAEHV 5D latents that its 2D bilinear resize path cannot handle.
- VHS preview exceptions are now isolated from H3 inference only when the traceback originates inside `videohelpersuite/latent_preview.py`; unrelated callback exceptions still propagate normally.
- The original ComfyUI callback is retried on every denoise step instead of being permanently disabled, preserving normal callback-side `x0` bookkeeping even when animated preview rendering remains incompatible.
- Added explicit preview-guard diagnostics to the internal sampler profile state. No H3 math, sampler, attention, VAE, latent, or VRAM policy is changed.

## 0.5.31-DEV

- Fixed Latent Hi-Res reconstruction OOM with explicit `attention_mode=existing` + Comfy Kitchen INT8 on 16 GB-class GPUs when the fused H3 QKV output itself no longer fits.
- Root cause from the failing 1.2x geometry: one monolithic `qkv_proj(x)` requested ~7.61 GiB while ~8.84 GiB was already active, so no cache/offload reserve can make the dense fused-QKV lifetime valid on a 15.89 GiB device.
- Added an exact Comfy-Kitchen EXISTING query-streaming path that activates only for structurally giant dense-QKV workloads on <=18.5 GiB GPUs; normal EXISTING remains unchanged.
- Full floating K/V is built in token chunks without ever materializing full QKV, then Comfy Kitchen prequantizes the complete K/V once so its global K-anchor detector and V scaling still see the entire sequence.
- Floating K/V is released before Q replay. Q is re-projected in 128-aligned chunks and attended against the single shared prequantized K/V store using the same CK split-prequant + prequantized-attention APIs as current ComfyUI.
- Query replay uses a 1025-row dummy K/V only for Q prequantization so CK selects the same long-sequence Hadamard rotation/CTA geometry as the real full K/V; dummy packed K/V is discarded.
- Native quantized qkv/out-proj weights are prepared once per pass and row-sliced when their Comfy Kitchen layout supports it; otherwise the stock projection is still chunked, preventing the giant fused allocation from returning.
- DynamicVRAM headroom is yielded only at KV-build, KV-prequant, and query-replay phase boundaries; there is no per-query synchronization/offload loop.
- SOL is not substituted, attention scale/mask semantics are unchanged, and the user's `ModelAttentionBackend = comfy kitchen attention` contract is preserved.

## 0.5.30-DEV

- Fixed chained Latent Hi-Res -> second LongMedia Sampler failure with `H3 LAYOUT SELF-HEAL` visual conditioning row mismatch.
- Root cause: native H3 VIDEO keyframes are packed on the current target spatial grid, while a sampler chained after learned latent upscaling receives a new H/W but can still inherit pre-hires keyframe latents from the original guider.
- The diffusion-model preflight now removes only geometry-incompatible VIDEO keyframe rows from the runtime payload and matching `cond_video_latents` entries before `PackedLayout` rebuild.
- Audio data attached to those keyframes is preserved; Ref2VA image/video/audio references are preserved because upstream `PackedLayout` gives refs their own spatial geometry.
- The repair is payload-local and does not mutate cached/original conditioning, plan objects, external guider state, target latent math, SOL/EXISTING attention, or Latent Hi-Res output.
- Remaining visual row mismatches now report that keyframe filtering already ran, narrowing any future failure to ref/payload ordering or metadata geometry.

## 0.5.29-DEV

- Fixed late-step OOM with explicit `attention_mode=existing` + Comfy Kitchen INT8 on 16 GB-class GPUs.
- Root cause: the prior INT8 residency policy disabled routine guards after block 0 even though dense EXISTING must materialize a multi-gigabyte fused QKV output; on later denoise steps active dynamic-model residency could grow until QKV no longer fit.
- Added a pre-QKV activation-workspace governor that preserves the selected EXISTING attention backend/math and partially offloads only active DynamicVRAM model residency when required.
- Comfy Kitchen budgeting accounts for its INT8 prequantized Q/K/V coexisting transiently with BF16/FP16 QKV.
- cudaMallocAsync headroom accounting is bounded by physical `total - allocated`; virtual allocator reservation is no longer treated as guaranteed dense-activation capacity.
- INT8 routine inter-block/late-block/step-boundary guards remain enabled for dense EXISTING, while bounded SOL keeps the resident-cache throughput policy.
- The single OOM retry now requests an additional 1536 MB residency headroom instead of repeating the same allocation after cache-only cleanup.
- No attention-family substitution, SOL routing change, tensor math change, or Comfy Kitchen backend replacement.

## 0.5.28-DEV

- Fixed Windows Triton bootstrap failure before embedded SM120 SOL kernels launch.
- LongMedia now augments bundled TinyCC JIT include paths with `triton/runtime/tcc/include/winapi` and `triton/runtime/tcc/include` without modifying `site-packages`.
- Embedded-Python `libs` is added defensively when the matching `pythonXY.lib` exists.
- SOL math, compressed-KV layout, routing thresholds, CUDA device ownership and fallback policy are unchanged.

## 0.5.27-DEV — Setup canvas-header alignment

- Re-anchor Setup canvas-only section headers after every dynamic widget reorder.
- Fix CONTROL / H3 CONDITIONING / PROMPT and TIMELINE / LOOP header drift introduced by the 0.5.26 frontend performance refactor.
- Keep the 0.5.26 DOM/pan/zoom performance optimizations intact.
- No backend, sampling, conditioning, planner, or camera behavior changes.

# 0.5.26-DEV — Frontend Interaction Performance

- Replaced 13 decorative Setup/Sampler DOM section headers with lightweight canvas-only custom widgets.
- Removed per-field `onDrawForeground` mode watchers; pan/zoom no longer performs LongMedia mode scans every frame.
- Collapsed 9 Setup + 3 Sampler 100ms polling timers into one low-frequency fallback timer per node.
- Planner/Cameras DOM editors now use ComfyUI `hideOnZoom` and CSS containment to reduce browser layout/paint work during graph navigation.
- Coalesced mode refresh callbacks and avoided redundant Setup/Sampler resize + whole-graph dirty operations.
- Backend generation, sampler, decode, Planner/Cameras data contracts remain unchanged.

# 0.5.25-DEV — Draggable Planner/Cameras + portable clip presets

- Planner clip cards are now draggable by their header/grab handle; moving a card reorders the actual clip object, not only its visual position.
- Added stable per-clip `clip_id` values so prompt, duration, seed, custom name, and matching Camera settings stay attached to the same clip after reordering.
- Added editable clip names and derived `start -> end` timeline readouts that recalculate immediately after duration changes or drag/drop reorder.
- Cameras now use the same draggable-card UX. With Planner + Auto Sync connected, dragging in Cameras reorders the linked Planner sequence and both nodes remain aligned by `clip_id`; standalone Cameras can still be reordered independently.
- Camera serialization now preserves `transition_type`, `space_relation`, and `entity_continuity` during card edits/reorders in addition to the existing camera fields.
- Added user clip presets in Planner: save/overwrite a card as a preset, apply the selected preset to an existing clip without changing its identity, or add a new clip from a preset.
- User clip presets are kept in browser storage and can be exported/imported as `MiniMax-H3-LongMedia-clip-presets.json` for backup or migration across ComfyUI reinstalls/node-folder deletion. Import merges by preset name, replacing same-name entries.
- Old Planner/Camera JSON without `clip_id`/names migrates transparently by current position; backend Auto Sync also understands stable clip identity.
- Setup, Sampler, Decode, attention/runtime, and LongMedia generation semantics are unchanged from 0.5.24.

# 0.5.24-DEV — Setup modes without legacy sentinels

- Removed user-visible `legacy` from `control_mode`, `h3_mode`, and `timeline_mode`.
- New Setup defaults are `control_mode=auto`, `h3_mode=hybrid`, `timeline_mode=single`.
- Preserved invisible migration for 0.5.23/pre-semantic workflows: serialized `legacy` values are translated from the old `workflow_mode` before validation.
- Setup-only change; Sampler/Decode/runtime execution logic is unchanged from 0.5.23.

# 0.5.23-DEV — Setup semantic modes

- Setup-only architecture split: `control_mode`, `h3_mode`, `timeline_mode`, independent Loop Closure.
- H3 modes: T2VA, pure native FL2VA, Ref2VA, Hybrid, Video Ref Edit.
- Timeline modes: Single, Segmented, MultiClip.
- `manual` is now a control level that reveals advanced Setup controls instead of being a workflow.
- Added public `transition_frames` for both Segmented and MultiClip; existing sampler consumes the same `plan.overlap_frames` contract.
- Restored Hybrid image-injection controls (`first_frame_mode`, denoise, blend span). Pure FL2VA forces `native_keyframe` and never inherits Hybrid injection.
- Clip Plan is authoritative only for `timeline_mode=multiclip`; otherwise it is ignored.
- Old `workflow_mode` / `overlap_frames` remain serialized migration state but are hidden from the public Setup UI.
- Sampling/runtime node code is unchanged.

## 0.5.22-DEV - FastVideo VSA two-sampler ownership

- Corrected the Kijai/FastVideo VSA contract for LongMedia workflows that intentionally use two sampler stages.
- The four-call FastVideo student schedule now applies only to sampler #1/base generation.
- Sampler #2/refiner is no longer auto-disabled and no longer receives the remainder of an inferred 8 -> 4+4 split.
- `refine_steps` remains user-owned: sampler #2 takes exactly the requested effective tail from the workflow-connected SIGMAS schedule.
- Latent-hires sampler #2 likewise derives its second-pass schedule from the connected workflow SIGMAS rather than the forced four-step base schedule.
- Added `FastVideo VSA REFINER OWNERSHIP` telemetry showing sampler #1=4 and the independently requested sampler #2 step count.
- Stock H3/PDD behavior is unchanged; H3ddle FastH3 retains its existing fail-closed extra-pass policy.

## 0.5.21-DEV - FastVideo VSA automatic 4-call sanitation

- Kijai/FastVideo VSA no longer aborts when a saved LongMedia workflow has `refine` or latent-hires enabled.
- For FastVideo VSA only, incompatible extra denoise passes are disabled locally for that run while the saved workflow/widget values remain unchanged.
- Stock H3, PDD, and other model families keep their existing refine/latent-hires behavior.
- H3ddle FastH3 remains fail-closed and is not re-enabled by this change.
- Added explicit `FastVideo VSA AUTO-SANITIZE` telemetry before the forced four-call schedule.

## 0.5.20-DEV - Kijai FastVideo VSA native compatibility

- Added a separate compatibility path for `Kijai/MiniMax-H3-experimental` `minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors`.
- Strictly detects the 50 `blocks.*.attn.to_gate_compress` INT8 ConvRot projections and validates their full quant payload and logical shape before loading.
- Creates missing `to_gate_compress` modules using the same Comfy mixed-precision Linear ABI as H3 `qkv_proj`, preserving AIMDO/dynamic-VRAM ownership; native upstream modules are reused when present.
- Keeps this backend isolated from H3ddle/PulpCut FastH3: no input-major remap, no seven-row AdaLN substitution, and no `h3.fasth3.*` contract reuse.
- Runs the published FastVideo VSA recipe (tile 64, 90% sparsity / top-k 0.10, learned coarse gate) through Comfy Kitchen `sol_attn` when available or the existing LongMedia memory-bounded blockwise VSA fallback on older Kitchen builds.
- Forces the published four-transformer-forward MiniMax-H3 shift-12 sigma ladder and blocks refine/latent-hires partial denoise schedules for this student.
- Non-plain T2VA LongMedia packing falls back to exact dense H3 attention rather than applying invalid VSA geometry; the four-step student schedule is retained.
- Added transactional runtime cleanup when switching from Kijai VSA back to ordinary H3 checkpoints so dynamically inserted gate modules cannot leak across model reuse.

## 0.5.19-DEV - FastH3 runtime isolation / stock-H3 reset

- Fixed stale FastH3 state leaking into ordinary H3/FL2VA/Ref2VA checkpoints when Comfy reuses a MiniMaxH3 model object after checkpoint switching.
- Stock H3 loads now restore all 51 AdaLN forwards, remove LongMedia-owned VSA gates, clear FastH3 timestep/cache/contracts, and proceed through the untouched stock runtime.
- FastH3 detection is now strict and prefix-local: the complete format-2 marker + input-major marker + 50 VSA gates + 50 block AdaLN tables + final AdaLN/times/tile/sparsity contract must all be present.
- Partial FastH3 packages fail closed; ordinary input-major or pruned H3 checkpoints are never promoted to FastH3.
- FastH3 load failures are transactional: any partially installed gates/patches are rolled back before the exception escapes.

## 0.5.18-DEV - FastH3 zero-copy input-major loading

- Fixed a severe Windows host-memory/pagefile regression in the FastH3 compatibility loader.
- Removed eager `.t().contiguous()` materialisation of all 200 FastH3 core projections and 50 learned VSA gates (~19.74 GiB of duplicate INT8 host storage).
- Input-major FastH3 weights are now remapped with zero-copy transpose views; Comfy Dynamic VRAM/AIMDO remains the owner of residency and staging.
- QuantizedTensor payloads, when already wrapped, are rebuilt around the same underlying qdata storage with corrected logical shape and without setting the slow `transposed` fallback flag.
- Added storage-alias assertions so a future PyTorch/Comfy ABI change fails before silently allocating another full checkpoint copy.
- FastH3 VSA portable fallback and the exact 7-row AdaLN contract are otherwise unchanged from 0.5.17.

## 0.5.17-DEV - FastH3 VSA portable compatibility executor

- Removed the hard startup dependency on the newest `comfy_kitchen.sol_attn` public API.
- Added a native LongMedia blockwise learned-VSA executor for older portable ComfyUI / Comfy Kitchen builds.
- The fallback preserves FastH3 Preview tile-64 routing, per-head top-k fine attention, exact sink/neighbor blocks, learned coarse gate, and block-length padding without materializing a full T x T score matrix.
- Current Comfy Kitchen `sol_attn` remains preferred automatically when its learned-VSA arguments are available.
- Startup/runtime telemetry now reports the active learned-VSA executor.
- Dynamic VRAM, quantized residency, four-step FastH3 schedule, and exact seven-row AdaLN lookup remain unchanged.

## 0.5.16-DEV - FastH3 input-major model-detection fix

- Fixed the 0.5.15 startup mismatch where stock Comfy interpreted H3ddle input-major `qkv_proj` / `fc1` dimensions as output-major and instantiated a 14-head / FFN-2688 MiniMax H3 instead of the real 56-head / FFN-14336 architecture.
- Added a pre-construction FastH3 detector shim: marked FastH3 packages are validated before `MiniMaxH3Model` exists, then only `num_attention_heads` and `ffn_hidden_size` are corrected in Comfy's detected config.
- Added exact header-only validation for all 200 input-major core projections and all 50 VSA gate matrices using the published H3ddle MiniMax H3 shapes. A single malformed block now fails before model allocation.
- Kept the 0.5.15 dynamic/weightless Linear ABI, exact seven-row AdaLN lookup, VSA gate loading, and LongMedia VRAM/offload ownership unchanged.

## 0.5.15-DEV - FastH3 dynamic-loader + exact AdaLN contract

- Fixed current ComfyUI `MixedPrecisionOps.Linear` compatibility: FastH3 preflight no longer assumes a materialized `.weight` exists before dynamic model loading. Logical `[out,in]` geometry is validated from `out_features` / `in_features` / `_orig_shape`.
- Preserves Comfy's weightless dynamic-Linears and avoids allocating dummy BF16 weights or disabling LongMedia VRAM/offload ownership.
- Corrected the FastH3 functional contract after auditing H3ddle's package converter: the 50 block + final `h3.fasth3.*.adaln` BF16 lookup tables are trained-function data, not disposable metadata.
- Validates the exact seven dual-clock AdaLN rows for the published 4-call video-shift-12 / audio-shift-3 schedule before loading.
- Streams only selected AdaLN lookup rows from CPU to the active block, keeping the ~68 MB lookup bank outside model GPU residency.
- Adds post-load validation for all 50 learned VSA gates, pruned-curve coordinate width, block/final AdaLN ABI, and lookup reshape/chunk semantics.
- Unsupported continuation/reference timesteps fail closed with a `FastH3 T2VA CONTRACT` error instead of silently approximating an AdaLN row.
- Startup sampling preflight now requires the exact FastH3 AdaLN runtime to be installed before the first denoise call.

## 0.5.14-DEV - FastH3 VSA native compatibility

- Added fail-fast support for `PulpCut/FastH3-VSA-INT8-ConvRot`.
- Detects H3ddle input-major FastH3 metadata before MiniMax H3 weight loading.
- Validates all 200 DiT core projection shapes and all 50 learned VSA gates before sampling.
- Converts input-major INT8 ConvRot projection storage back to ComfyUI output-major layout without dequantizing the checkpoint.
- Installs 50 learned VSA gate Linear modules under native Comfy mixed-precision/dynamic-VRAM ownership.
- Consumes FastH3/H3ddle metadata instead of leaving it as `unet unexpected`.
- Uses Comfy Kitchen `sol_attn` learned-VSA API (tile 64, 90% sparsity) for plain T2VA packed layouts.
- Falls back to exact dense H3 attention for continuation/reference layouts unsupported by FastH3 Preview v1 instead of crashing.
- Forces the published 4-call shift-12 sigma ladder and rejects extra refine/latent-hires denoise passes.
- Existing LongMedia memory/offload/residency logic remains the owner of model lifecycle.

## 0.5.13-DEV - loop closure v3 macro-return + grouped UI

- Reworked Loop Closure so H3 no longer has to chase a strong terminal anchor.
- H3 receives only a light opening-structure hint; after sampling, only low-frequency latent macro geometry is progressively returned toward the opening state.
- Fine detail and local motion remain authored by H3; no RGB crossfade and no exact full-latent endpoint copy.
- Nonlinear Loop Strength response makes 0.65 a strong practical default while 1.0 reaches an exact macro-state return.
- Loop Closure controls are now presented together under a dedicated LOOP CLOSURE group: enabled, frames, strength.
- Backend INPUT_TYPES ordering remains unchanged for saved-workflow compatibility.

## 0.5.12-DEV - context-preserving loop closure v2

- Loop Closure UI is now the intentionally small three-control surface: `loop_closure_enabled`, `loop_closure_frames` (default 57), and `loop_closure_strength` (default 0.65).
- Replaced the exact opening-frame terminal latent with a macro-structure attractor: the opening frame contributes low-frequency geometry/composition while the generated tail retains its own high-frequency latent detail.
- Loop Strength controls structural attraction only; motion/detail are deliberately left freer to avoid the end-of-video acceleration caused by chasing an exact frame-0 latent.
- Closure sampling now uses a strength-aware low/mid-sigma suffix instead of starting near sigma-max, reducing unnecessary re-authoring of the ending.
- Closure Frames now snaps to the nearest valid H3 `17*k+5` length (ties downward), so 57 frames becomes 56 rather than expanding to 73.
- Audio remains bit-for-bit preserved and the first latent step of the closure window remains exact.

## 0.5.11-DEV - native loop-closure regeneration

- replaced the old post-decode RGB loop-closure blend with a native MiniMax H3 tail-regeneration pass
- when `loop_closure_enabled` is ON, LongMedia now takes the final tail region, preserves its opening frame exactly, anchors the very end to the movie's real first frame, and lets H3 resample the transition geometry in latent space
- the pass keeps audio bit-exact, works across ordinary long-video workflows including MultiClip by updating the final segment latent itself, and records detailed closure telemetry in the runtime/decode reports
- loop-closure frame requests are now snapped internally to the nearest valid H3 frame count instead of applying an arbitrary RGB-only tail mix

## 0.5.10-DEV - current ComfyUI / PDD FinalLayer compatibility

- updated LongMedia final-output streaming for the current MiniMax H3 `FinalLayer.forward(x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts)` contract
- preserved backward compatibility with pre-PDD ComfyUI builds where FinalLayer is called with the legacy four payload arguments
- added exact PDD output-head-bank handling: the active sigma interval selects the same dt-weighted effective video/audio head as stock ComfyUI, then LongMedia applies that effective head in token chunks to retain the low-VRAM final-output path
- standard non-PDD H3 checkpoints remain on the existing single-head streaming path
- no changes to MultiClip, Reconstruction, Detail Recovery V3, loop closure, planner, or continuation semantics

## 0.5.9-DEV - global loop closure option

- added an all-workflow loop closure option to LongMedia Setup with two new controls: `loop_closure_enabled` and `loop_closure_frames`
- loop closure is applied post-decode and gradually reshapes only the trailing frames toward the opening sequence so seamless loops can better reconcile end-state geometry, motion phase, and photometric drift
- the loop-closure policy is stored in `LongMediaPlan`, propagated through runtime reports, and can be disabled per workflow without affecting ordinary generation semantics

## 0.5.8-DEV - MultiClip latent motion continuity

- ordinary MultiClip clip 2+ now inherits the exact generated video/audio latent overlap from the previous clip instead of starting from a fresh full-denoise continuation head
- the inherited 22-frame MultiClip overlap is frozen at denoise=0, preserving boundary motion velocity/phase and preventing the characteristic mid-clip motion acceleration seen on continuation clips
- native MiniMax motion-context keyframes remain active as auxiliary conditioning rather than the sole carrier of continuity
- reconstruction and Reconstruction Detail Recovery V3 paths are unchanged

## 0.5.7-DEV - maximum-quality reconstruction detail recovery v3

- keeps the working segmented native Ref2VA reconstruction path unchanged
- replaces the single detail trajectory with a dual-candidate detail ensemble: broader structure-detail synthesis plus independent low-sigma microtexture synthesis
- adds temporal stabilization to transferred detail residuals to suppress flicker while retaining local motion detail
- adds bounded amplification of medium/high-frequency information already present in the stable base latent, addressing soft decode when the latent itself is detail-dense
- re-locks all low-frequency latent content to the completed two-pass reconstruction so camera, staging, body placement and broad luminance remain authoritative
- quality floor now uses at least 4 structure-detail evaluations and 3 micro-detail evaluations whenever detail recovery is enabled

## 0.5.6-DEV - reconstruction detail recovery v2

- upgraded the post-global reconstruction detail layer from narrow high-frequency residual transfer to bounded multi-band detail recovery
- detail pass scheduler now starts slightly earlier in the sigma tail so the model can paint more facial/clothing/edge structure while the stable two-pass reconstruction still owns low-frequency geometry
- detail merge now separates medium-band and high-band residuals, uses edge-aware confidence weighting, and re-locks low frequencies to the completed two-pass reconstruction
- updated reconstruction reporting to surface the multiband detail layer semantics

## 0.5.5-DEV - reconstruction detail recovery layer

- leaves the working segmented native Ref2VA reconstruction path unchanged
- adds optional Detail Recovery controls to Video Reconstructor: enable, strength and steps
- detail layer executes only after sampler #2 has produced the global continuous reconstruction
- runs a short independent video-noise detail trajectory inside the same H3 model lifecycle
- transfers only the spatial high-frequency latent residual from that candidate into the stable two-pass result
- preserves low-frequency video geometry/motion and restores the pre-detail audio latent exactly
- uses a per-channel soft limiter to suppress ringing/noise amplification
- does not resize/re-encode the target latent and does not touch reconstruction Ref2VA source-window logic

## 0.5.4-DEV - Reconstruction V5 native Ref2VA edit

- abandons the FL2VA reconstruction experiment; FL2VA remains only for first/last-frame style anchoring
- reconstruction now uses native Ref2VA video-edit semantics with the current source window as `<Video 1>`
- target video latent is fresh; only generated overlap is inherited between LongMedia segments
- reconstruction prompt explicitly declares the target as a restored/edited version of `<Video 1>` and preserves source action, timing, camera, composition, identities and shot order
- each continuation pass refreshes the actual `minimax_refs[*].latent` video block from the matching source window
- source reference refresh preserves one authoritative source-fit transform and one stored target canvas across all segments
- reconstruction strength now controls native Ref2VA visual condition augmentation instead of target-latent denoise
- source audio remains target-locked/preserved when connected; it is not regenerated

## 0.5.3-DEV - V4 tokenized FL2VA guide compatibility

- after two failed multi-frame PackedLayout self-heal attempts, reconstruction V4 now avoids multi-frame keyframe metadata entirely
- the full source guide latent is split into one-latent-token FL2VA anchors positioned at the exact native H3 pixel-frame origins from `FRAME_PER_TOKEN=(1,4,4,4,4)`
- this is temporally equivalent to a native multi-frame clip guide but remains compatible with process-global PackedLayout wrappers that collapse a multi-frame keyframe to T=1
- added strict token-to-frame coverage validation: mapped source guide frames must equal the target H3 frame count exactly
- target latent geometry and reconstruction source encoding are unchanged

## 0.5.2-DEV - V4 authoritative conditioning metadata rebind

- after two V4 layout failures, reworked the root contract instead of adding another geometry heuristic
- runtime `cond_video_latents` / `cond_audio_latents` are now the single source of truth for H3 conditioning geometry
- keyframe/ref metadata is rebound in the exact upstream ordering used by `MiniMaxH3.extra_conds`: visual keyframes, then visual refs; same for audio
- PackedLayout is rebuilt only after metadata points at the exact tensors that `MiniMaxH3Model._cond_video_rows()` / `_cond_audio_rows()` will patchify
- added explicit block-count assertions so wrapper mutations cannot silently add/remove condition streams
- reconstruction target latent / V4 frame-aligned FL2VA guide architecture is unchanged

## 0.5.1-DEV - V4 PackedLayout condition-row self-heal

- fixed Reconstruction V4 multi-frame FL2VA guide crashing with `cond_video_rows` vs `img_update` broadcast mismatch
- runtime PackedLayout validation now compares cached condition-row counts against the exact `cond_video_latents`/`cond_audio_latents` tensors, not only the target signature
- stale layouts are rebuilt whenever conditioning geometry changes even if target H/W/T and text length are unchanged
- visual keyframe metadata is reconciled against H3's authoritative `cond_video_latents` order before rebuilding layout
- preserves the V4 frame-aligned fresh-target architecture; no fallback to direct source-latent injection

## 0.5.0-DEV - Reconstruction V4 frame-aligned FL2VA

- replaced generic Ref2VA reconstruction with native full-clip FL2VA guide conditioning
- source video is encoded per segment on the exact target latent H/W/T grid and attached at local frame 0 as a multi-frame guide
- target video latent remains fresh/full-denoise; source pixels are never copied into the target latent
- reconstruction strength now maps to native `minimax_visual_cond_noise_aug` in a conservative H3-safe range near 0.999
- continuation refreshes the frame-aligned source guide for every segment while preserving generated latent overlap
- reconstruction guide metadata explicitly clears stale refs/keyframes to prevent PackedLayout row desynchronization
- source audio is fitted into the target audio stream and frozen when connected; final audio passthrough remains untouched
- added hard guide-vs-target latent geometry assertion before sampling

## 0.4.99-DEV - reconstruction V3 latent contract fix

- fixed a Reconstruction V3 setup crash caused by passing raw `NestedTensor` samples into `unpack_av_samples()` instead of the enclosing LATENT dictionary
- added a targeted contract audit for all `unpack_av_samples()` call sites so the helper is only called with LATENT dictionaries
- no reconstruction algorithm changes from 0.4.98; this build fixes only the setup contract violation

## 0.4.98-DEV - reconstruction V3 source-reference path

- reconstruction V3 no longer injects degraded source video into the target latent
- first reconstruction pass now builds a fresh target latent and uses the source clip as a native video reference
- continuation passes keep overlap continuity while refreshing the source reference window per segment
- reconstruction video refs now re-encode to explicit per-ref latent geometry for stable PackedLayout row counts
- direct source-latent filling is disabled for reconstruct continuation to avoid blur-preserving behavior and latent geometry pitfalls
- added explicit `reconstruction_guidance` contract field to keep setup/guider/continuation behavior aligned

## 0.4.97-DEV - source-faithful reconstruction authority

- neural_remaster no longer attenuates the encoded source latent toward zero
- reconstruction_strength now maps into a bounded profile-specific denoise window instead of approaching free regeneration
- neural_remaster maximum reconstruction denoise is capped to preserve source action, composition, identity, wardrobe and camera
- source preprocessing is now mild artifact suppression rather than aggressive low-pass destruction
- reconstruction continuation no longer injects generic previous-AV / native video keyframe context; continuity is owned by the copied latent overlap plus the authoritative source window
- reconstruction source video is explicitly excluded from per-pass mutable Ref2VA video-reference handling
- fixes the 4991 -> 9982 PackedLayout conditional-row mismatch seen on reconstruction continuation

## 0.4.96-DEV - Setup UI rollback to proven 0.4.82 contract

- rolled Setup presentation logic back to the proven 0.4.82 visibility model instead of the fragile blanket per-mode widget matrix
- restored all shared Setup widgets that had become permanently hidden across workflow modes
- `workflow_mode` is always visible and remains owned by Setup
- Reconstruction hides only global duration ownership (`manual_duration`, `duration_source`); its chunk controls remain owned by the Reconstructor node
- MultiClip no longer creates/opens clip cards inside Setup; LongMedia Planner is the sole clip editor
- Manual again exposes the advanced segmentation/conditioning controls; Segmented Continuation exposes only its segment controls
- retained current reconstruction backend and V2 reconstruction fixes; this change is frontend/presentation only

## 0.4.95-DEV - reconstruction target-geometry fix

- fixed Reconstruction V2 first-segment target stub using a 1x1 latent spatial grid
- target AV geometry stub now uses the real H3 latent grid (`height/16`, `width/16`)
- prevents MiniMax H3 VideoVAE reflect-padding failure on encoded reconstruction frames

## 0.4.94-DEV - Setup per-workflow UI ownership

- rebuilt LongMedia Setup visibility around an explicit workflow ownership matrix
- `manual` is the diagnostic superset and exposes all active Setup controls
- `segmented_continuation` owns fixed segment duration and overlap controls
- `multiclip` no longer exposes Setup prompt/timeline/clip-card controls; Planner owns clip prompts and durations
- `reconstruct` exposes only target canvas/resolution plus universal workflow/debug controls; source FPS, timing, audio policy and chunking stay owned by Video Reconstructor
- `hybrid_auto`, `loop`, `ref2va_full`, and `video_ref_edit` hide segmentation-only controls
- first-frame tuning is shown only for modes that actually own first/last anchors
- empty section headers are hidden dynamically

## 0.4.93-DEV - Setup UI ownership correction

- `workflow_mode` is always visible in LongMedia Setup, including Reconstruction
- Reconstruction hides only source-owned timeline duration controls; normal Media/References and Workflow/Debug controls remain visible
- MultiClip clip cards/editor are no longer shown inside Setup; clip editing belongs to the dedicated LongMedia Planner
- hidden `multiclip_json` remains only as backend/legacy compatibility storage

## 0.4.92-DEV - reconstruction Setup UI cleanup

- when the Reconstruction contract is connected, LongMedia Setup now hides controls owned by the Reconstructor: duration, duration source, segment duration, overlap, resolution/reference policy, source FPS, video/audio mode and debug guard
- workflow selector is hidden while the Reconstruction socket is connected because backend ownership is forced to `reconstruct`; disconnecting the contract restores the selector
- empty Timeline / Media / Workflow section headers are hidden in reconstruct mode
- prompt and target width/height remain visible because they still affect reconstruction output

## 0.4.91-DEV - reconstruction V2 functional import fix

- fixed reconstruction preprocessing crash caused by using the local alias `F` outside its defining scope
- reconstruction low-pass preprocessing now calls `torch.nn.functional.interpolate` explicitly
- added static/runtime smoke checks for the reconstruction helper before packaging

## 0.4.90-DEV - reconstruction V2 source-decoupling

- reconstruction no longer treats the degraded source video latent as a nearly direct pixel authority
- source frames are low-pass preprocessed before VAE encoding in `balanced` and `neural_remaster` profiles
- added profile-aware source-latent attenuation so `reconstruction_strength` gives H3 more room to re-synthesize detail
- added a profile-aware denoise floor for reconstruction video masks
- first reconstruction segment now uses the same source-fit geometry contract as later segments
- reconstruction report now surfaces the new low-pass source-authority execution model

# 0.4.87-DEV — Reconstruction continuation geometry fix

- Fixed reconstruction/segmented continuation crash on pass 2 when Latent Hi-Res is enabled.
- The next segment target was correctly built from the low-res `previous_segment_continuation`, but native motion-context conditioning was incorrectly built from the hi-res `previous_segment`.
- H3 `PackedLayout` therefore reserved condition rows on the target grid while `cond_video_latents` were patchified from a different H/W grid, causing `all_video_rows[~img_update] = cond_video_rows` shape mismatch.
- Continuation guider now uses the exact same `previous_segment_continuation` geometry authority as `NextSegment.prepare`, matching the existing MultiClip contract.
- No change to reconstruction strength, source resize, audio lock, sampling schedule, or final hi-res output path.

# 0.4.86-DEV — Reconstruction plan contract fix

- Added the missing `reconstruction_resize_mode` field to the immutable `LongMediaPlan` dataclass.
- Fixes `LongMediaPlan.__init__() got an unexpected keyword argument 'reconstruction_resize_mode'` introduced by the 0.4.85 source-fit plumbing.
- The selected reconstruction `source_fit` policy is now carried authoritatively from Setup into every segment encode.
- No generation, MultiClip, Cameras, sampler, refiner, or decode behavior was changed.

# 0.4.85-DEV — Reconstruction source-fit fix

- Added `source_fit` to **MiniMax H3 LongMedia Video Reconstructor**: `center_crop` (default), `stretch`, or `strict`.
- Reconstruction source frames now use the selected resize policy when the source geometry differs from the LongMedia target canvas.
- Fixes the crash `Input is WxH, target is WxH. Choose stretch or center_crop.` for reconstruction workflows such as 480x360 → 960x544.
- Existing non-reconstruction encode paths remain unchanged.

# 0.4.84-DEV — Reconstruction workflow UI fix

- Fixed frontend workflow sanitizers that still used the pre-reconstructor workflow list. Selecting `reconstruct` was therefore treated as an invalid value and immediately reset to `hybrid_auto`.
- Added `reconstruct` to both `web/node_facade.js` and `web/long_media_dynamic_inputs.js` workflow registries so the UI selection now remains stable and matches the Python `INPUT_TYPES` contract.
- No sampler/reconstruction backend behavior changed from 0.4.83-DEV.

# 0.4.83-DEV — LongMedia Video Reconstructor

- Added **MiniMax H3 LongMedia Video Reconstructor** setup node. It carries full source video/audio into the existing LongMedia engine without duplicating sampler/refiner/decoder code.
- Added `reconstruct` workflow ownership to Long Media Setup. Reconstruction automatically uses source-video duration, per-window VAE encoding, fixed temporal chunks and continuation overlap.
- Added `conservative`, `balanced`, and `neural_remaster` reconstruction profiles plus an explicit `reconstruction_strength` fidelity control.
- Source video is sliced and encoded **per segment**; full-duration source latents are never materialized, so GPU memory scales with the local reconstruction window rather than total video duration.
- Original source audio can be preserved untouched at decode while its local H3 audio latent is frozen as the temporal authority during reconstruction.
- Reconstruction participates in the existing segment storage / chained global-refine contract, so sampler #2 can refine a continuous reconstructed timeline using the same LongMedia runtime.
- Existing generation, MultiClip, Cameras, video_ref_edit, lip-sync, and manual workflows remain unchanged.

## 0.4.82-DEV — PackedLayout frame_count compatibility

- Built on v0.4.81.
- Fixes `TypeError: PackedLayout.__init__() got an unexpected keyword argument 'frame_count'` in ComfyUI revisions whose MiniMax H3 PackedLayout constructor predates/omits the optional `frame_count` parameter.
- Runtime layout self-heal now inspects the active PackedLayout constructor and passes `frame_count` only when the current implementation explicitly accepts it (or accepts `**kwargs`).
- On older constructors the rebuild proceeds with `keyframes` + `refs` only, matching that ComfyUI revision's native contract.
- All v0.4.81 wrapper-chain interoperability and stale-layout row self-heal behavior are preserved.
- No diffusion, attention, camera, audio, hires, noise or decode changes.

## 0.4.81-DEV — Payload/layout interoperability + stale-layout self-heal

- Built on v0.4.80; Existing-attention memory restoration, Cameras/Planner continuity, V8 giant-QKV, V7 residency, pinned H2D and global continuous refine are preserved.
- `MiniMaxH3.extra_conds` no longer fails solely because another custom node already wraps it. LongMedia now composes outside the current wrapper chain, calls that owner first, then performs its payload repair.
- Existing compatible LongMedia/Motion-Context markers are detected anywhere in the `__wrapped__` chain to avoid duplicate patching.
- PackedLayout patching likewise composes outside an existing owner instead of treating a foreign module name as an automatic collision.
- Adds a DIFFUSION_MODEL-entry H3 layout self-heal. The cached payload layout is compared against the actual target video/audio tensors that are about to be patchified.
- Layout is rebuilt from runtime `(text_len, latent_t, latent_h, latent_w, audio_t)` when the signature, target-video row count, or reference metadata is stale.
- Ref metadata (`latent_t/h/w`, `ref_audio_t`, video/video_audio kind) is normalized from the actual conditioning tensors before rebuild.
- Detects layouts whose signature appears current but whose `img_update` target-row count is stale/corrupt.
- After rebuild, validates target rows and conditioning rows before H3 reaches the opaque broadcast assignment.
- Replaces `shape [10590,96] cannot broadcast to [9120,96]` with either an automatic `[H3 LAYOUT REBUILD]` recovery or a precise guide/reference geometry error.
- No diffusion math, attention algorithm, sigmas, noise, hires math, audio generation, camera trajectories or decode changes.

## 0.4.80-DEV — Existing attention 16GB H3 memory restoration

- Built on v0.4.79; all Camera/Planner continuity work, V8 QKV, V7 residency, pinned H2D and global continuous refine are preserved.
- Fixes the current ComfyUI MiniMax H3 `existing`-attention regression on 16GB-class GPUs caused by the upstream long-sequence `v = v.clone()` allocation.
- Explicit `attention_mode=existing` is now honored on long H3 sequences instead of being silently replaced by SOL.
- Long existing-attention jobs on <=18.5GB GPUs use a local H3 Attention.forward equivalent that preserves the user-selected ComfyUI `optimized_attention` backend (comfy-kitchen / Sage / PyTorch/etc.) but omits the redundant full-size V clone.
- The point-wise output projection is chunked for long sequences with identical Linear math to reduce the post-attention peak.
- If the restored existing path still OOMs, it trims allocator cache and retries the same existing backend once. It does not silently substitute SOL.
- AUTO attention keeps the existing safety policy and may still select SOL when appropriate.
- No changes to attention algorithm selection, diffusion math, sigmas, noise, latent hires, audio, camera continuity or decode.

## 0.4.79-DEV — Shot continuity contract / scene-population lock

- Built on v0.4.78 and preserves production trajectory presets, boundary state lock, V8 giant-QKV, V7 residency, pinned H2D and global continuous refine.
- Adds per-boundary Transition Type: Continuous / Same Shot, Threshold Entry, Occluded Hidden Cut, Hard Cut.
- Adds Space Relation: Same Space, Adjacent Space, Different Space.
- Adds Entity Continuity: Lock Population / Layout, Preserve Main Subjects, Allow Background Evolution.
- Continuous transitions now explicitly lock population, architecture, props and major spatial landmarks; no new foreground people may spawn during viewpoint travel.
- Same Space forbids exterior/interior or room/location jumps at clip boundaries.
- Threshold Entry forces a visible connected threshold: current clip ends at/entering the threshold and next clip opens from that exact state, revealing the destination progressively.
- Manual opposite motion vectors are detected and compiled with a no-abrupt-reversal safeguard.
- Adds production preset `Approach → Threshold → Interior`.
- Camera transition checkbox is now a full shot-continuity contract, not camera-only metadata.
- No diffusion math, noise, sigma, latent-hires, audio or decode changes.

## 0.4.78-DEV — Production camera trajectories + boundary state lock

- Adds seven production camera-motion presets generated across the current 2–16 clip Planner count.
- Presets keep neighboring clips compatible in motion vector, lens family, stabilization and progressive shot-size change.
- Continuation clips inherit the exact previous terminal viewpoint/motion state instead of conceptually restarting at their own target settings.
- Current clip settings are progressive target cinematography; transitions explicitly require matching last-frame/first-frame direction and velocity.
- Manual camera edits switch the preset selector back to Custom.
- Built on v0.4.77; diffusion math, V8 QKV, V7 residency, noise, hires, audio and decode are unchanged.

## 0.4.76-DEV — V8 explicit giant-QKV contract

- Built directly on v0.4.75.
- Fixes both silent 8192 caps that prevented an explicit `sol_qkv_chunk_tokens=16384` sampler setting from reaching native TensorWise INT8 SOL.
- Startup INT8 runtime policy now preserves explicit QKV requests up to 16384.
- Block0 AUTO VRAM now performs the only giant-QKV safety decision: for >=90k tokens, 16K is kept when real recoverable headroom is >=1.8 GB and >=10%; otherwise it falls back to the proven 8192 baseline.
- Adds `[V8 QKV OWNERSHIP]` and `[V8 GIANT QKV CONTRACT]` logs with requested/effective/chunk-count/fallback.
- The actual streamed-Q implementation already consumes `state['sol_qkv_chunk_tokens']`; no SOL math was changed.
- V7 residency, pinned H2D, Governor V5, giant MLP 2048, global continuous refine and non-diegetic Cameras are preserved.
- MLP, output projection, sigmas, noise, latent geometry, audio and decode are unchanged.

## 0.4.75-DEV — V7 giant global-refine persistent residency

- Built on v0.4.74 and preserves the non-diegetic Cameras compiler, Governor V5, native pinned H2D fastpath, and the measured giant INT8 MLP 2048 floor.
- Targets only the second-sampler giant global-refine SOL storage/cache policy.
- Fixes the measured forward-2/3 cache destruction where `INT8 SOL STORAGE GUARD` emptied 2–4 GB of PyTorch allocator cache solely because driver-visible free memory was below the old floor.
- For native TensorWise INT8 giant sequences (`>=90k` tokens), the guard now evaluates effective reusable headroom as `driver_free + PyTorch cached VRAM`.
- If effective headroom is at least 3584 MB, opportunistic `soft_empty_cache()` is skipped and current residency is preserved for the next diffusion forward.
- Forced cleanup and genuinely low effective-headroom cleanup remain unchanged; emergency/OOM safety is not disabled.
- QKV chunk size, SOL workspace math, output projection chunking, MLP math, sigmas, noise, latent geometry, audio, cameras and decode are unchanged.
- Adds `[V7 GIANT RESIDENCY PRESERVE]` diagnostics whenever an old opportunistic trim is suppressed.

## 0.4.74-DEV — strict non-diegetic Cameras compiler

- Built directly on v0.4.73; all sampler/performance changes are preserved unchanged.
- Camera Rig Builder UI remains intact, including real rig/body/lens names for user-facing creative control.
- Model-facing camera text no longer emits physical rig descriptions such as tripod, crane, drone, gimbal, Steadicam, operator, dolly, robot arm, cable-cam or camera body.
- Each rig is compiled into the resulting viewpoint geometry/motion instead: aerial viewpoint, elevated rise, fixed viewpoint, stabilized tracking, articulated motion-control path, etc.
- Camera bodies are compiled into image-character descriptions rather than physical devices. Smartphone/DV/VHS/film selections now describe imaging characteristics only.
- Stabilization and movement descriptions are converted to viewpoint behavior instead of visible equipment behavior.
- Transitions no longer repeat raw rig/body UI keys into the prompt. They transition between safe semantic cinematography descriptions.
- Every camera instruction carries a strict non-diegetic visibility guard forbidding filming devices, supports, aerial platforms, operators, crew, production gear and reflections of that equipment.
- Planner/Cameras data structure, cards, Camera Rig Builder UI, clip count, transitions and Setup integration remain compatible.

## 0.4.73-DEV — native pinned transport + measured giant-refine MLP floor

- Performance fix #2, built directly on the validated v0.4.72 Governor V5 branch.
- Keeps the v0.4.71 global continuous refiner architecture unchanged: one continuous AV latent, one sampler #2 diffusion timeline, no seam passes or blends.
- Diffusion sampler pinned-memory policy is now transport-aware. On recent AIMDO + comfy-kitchen native TensorWise INT8, LongMedia preserves the user's pinned-memory-enabled state instead of forcing `disable_pinned_memory=True` and unpinning the model.
- This change is sampler-only. NativeReferenceToVideo / TE reference encoding keeps the older conservative pinned-memory gate to avoid reintroducing Windows HostBuffer exhaustion failures.
- After the first completed H3 block of a 90k–150k-token resident INT8 forward, real free+reclaimable VRAM is measured. If at least 1.8 GB / 10% remains and exact MLP parity passed, the giant-sequence MLP floor is promoted from 1024 to 2048.
- The 2048 floor halves MLP chunk-loop count for ~127k-token global refine while preserving stock quantized math.
- True HARD_SAFE/CAUTION pressure may still demote the chunk size, so OOM safety remains active.
- Governor V5 anti-thrash behavior for 28k–90k continuation forwards remains unchanged.
- No changes to H3 model math, SOL math, sampler sigmas, noise, latent hires, continuity, camera, audio, or decode.

## 0.4.72-DEV — Governor V5 forward anti-thrash lock

- Performance fix #1 only. The global continuous refiner and giant-sequence policy are unchanged from v0.4.71.
- Fixes the measured continuation-forward governor oscillation where ~30k-token passes alternated MLP `2048 <-> 4096` across transformer blocks as free VRAM hovered around the V4 threshold.
- For TensorWise INT8/W4A8 forwards in the 28k–90k token range, blocks 0 and 1 act as a real-memory probe.
- After block 1, Governor V5 locks the safest MLP chunk and barrier state observed during those two blocks for the remainder of that diffusion forward.
- The lock resets on every new H3 diffusion forward using the existing INT8 SOL forward-generation counter.
- The policy never promotes above the safest measured probe result, so this removes policy thrash without increasing OOM risk.
- Short ~20k-token first clips are intentionally left on the existing adaptive policy to avoid regressing their already-fast path.
- Giant >=90k-token global-refine behavior is intentionally untouched for the next isolated optimization stage.
- No sampler math, SOL math, quantized math, latent hires, continuity, camera, audio, or decode behavior is changed.

## 0.4.71-DEV — global continuous chained refiner

- Drops the unsuccessful per-boundary seam-solver experiments entirely.
- Keeps the proven v0.4.67 fixes: ordinary seeded refine noise, no duplicate `previous_av`, stale VIDEO-keyframe removal, v0.4.52 Camera Rig Builder, and dual MultiClip / segmented_continuation support.
- Sampler #1 still generates and latent-upscales segments independently for bounded VRAM.
- Before sampler #2, all stored high-res AV segments are assembled with the phase-safe LongMedia stitch contract into one synchronized native H3 timeline.
- Sampler #2 now runs exactly ONE custom-sigma diffusion solve over that continuous movie latent instead of one solve per clip plus seam solves.
- The output no longer carries per-clip decode metadata; VideoVAE receives the already-continuous refined latent directly.
- No seam-specific diffusion passes, no RGB blend, no latent feather, no overlap restore.
- Supported for both `multiclip` and `segmented_continuation`.

## 0.4.67-DEV — dual-workflow chained refiner + v0.4.52 Camera Rig Builder

- Base restored to the user's v0.4.50 camera-transitions build.
- Imported the complete v0.4.52 Cameras/Rig Builder implementation and matching UI: rig/support, camera body, lens, stabilization, movement path, intensity and transitions.
- Chained LongMedia sampler refinement now uses the same segment-container contract in both `multiclip` and `segmented_continuation`.
- First sampler stores native per-pass AV latents for later chained refinement in both workflows.
- Second sampler refines those stored high-resolution segments directly instead of regenerating clip 2+.
- External refine keeps ordinary seeded sampler noise for non-zero custom sigma schedules; the old forced-zero-noise path is removed.
- Continuation `noise_mask` metadata is cleared before sampler #2.
- Clips 2+ do not synthesize a second `previous_av` motion context.
- Geometry-bound VIDEO keyframes are removed during external hi-res refinement while refs and audio-only keyframes are preserved.
- Existing exact inherited-overlap protection is retained; no feather/crossfade/post-blend is introduced.

## 0.4.49-DEV — LongMedia Cameras

- Added `MiniMax H3 LongMedia Cameras`, a per-clip camera-direction layer between Planner and Setup or usable standalone.
- Four per-clip dropdowns: Shot Size / framing, Camera / Capture Profile, Camera Behavior, Motion Speed.
- Added cinematic cameras, DSLR, DV, VHS/VHS-C, Hi8, film, smartphone, action/bodycam, broadcast and surveillance capture profiles.
- Added shot sizes from Extreme Wide through Extreme Close-Up plus Macro/Detail, OTS, Two-Shot and POV.
- Planner-connected mode inherits clip count and durations; standalone mode keeps Add/Remove controls.
- Camera direction is appended per clip while preserving the original Planner base prompt.

## 0.4.46-DEV — tolerant MultiClip prompt import

- Expanded Planner Import Prompt parsing in both browser and backend paths.
- Accepts inline `clip_1: text`, standalone headers, Markdown labels, `Clip N (5s)`,
  YAML-style prompt wrappers, JSON, XML clip blocks, and strict numbered lists.
- Imported structured prompt metadata never overrides existing Planner duration/seed.
- No Planner or Sampler UI changes.

# 0.4.41-DEV lip-sync target-audio layout hotfix

- Fixed current ComfyUI stock H3 incompatibility with LongMedia audio-only `minimax_keyframes`.
- `lip_sync_target_audio_locked` now uses the frozen target audio stream as the sole timing authority and does not emit audio-only keyframes.
- Legacy/pre-encoded LongMedia audio-only keyframes are stripped defensively before stock `MiniMaxH3.extra_conds` / `PackedLayout`.
- Preserved supported visual keyframes, native Ref2VA references, target audio locking, segmented continuation handoff, and final source-audio restore.

# 0.4.41 DEV — lip_sync segmented audio-layout guard

- Fixed a MiniMax H3 packed-audio shape mismatch in `lip_sync + segmented_continuation` when `ref_audio_t` metadata no longer matched the actual per-pass `audio_latent` length.
- Audio reference geometry is now normalized from the encoded latent tensor before H3 builds `PackedLayout`, including refs added or rewritten during continuation handoff.
- The fix is metadata-only: it does not resample, crop, pad, or otherwise alter the encoded reference audio values.
- Included the corrected segmentation prompting documentation: fixed segmentation is a continuous-prompt VRAM/continuation mechanism, not a per-segment timestamp scheduler.

# 0.4.40

- Production release consolidating current continuity, segmented decode isolation, latent hi-res, SLA/VRAM, UI serialization and console-cleanup fixes.
- Issue #8 hardened with a decoder-side `plan.mode == "multiclip"` guard.
- Release metadata synchronized to 0.4.40.

# 0.4.31

- Fixed `nodes.py` import failure on Python 3.10/3.11: `f-string: unmatched '('` in the motion-context diagnostic at line 8333.
- No sampling, conditioning, memory, Planner, MultiClip, segmentation, audio, or UI behavior changed.
- PyTorch `KernelPreference` / `ScaleCalculationMode` deprecation messages are unrelated warnings and are not caused by LongMedia.

# 0.4.30

- MultiClip prompt authoring belongs to **MiniMax H3 Long Media Planner**: separate Global Prompt and Multiple Clips Prompt, Auto Import, and manual Import Prompt.
- Imported `clip_N` / `shot_N` sections are copied into editable Planner clip cards; duration and seed remain visual per-card controls.
- Long Media Setup only consumes the Planner clip plan; it no longer owns the 0.4.30 prompt-import UI.
- Preserves the stock `CFGGuider.sample()` extension contract so KJ Model Preview Override remains compatible.
- Import Prompt never toggles Auto Import; frontend-readable connected STRING sources import immediately, while backend-only dynamic sources use an independent one-shot request.
- Setup preserves the user-selected node width when switching to MultiClip; MultiClip card layout is responsive.

## v0.4.11

- Consolidated all post-0.4.1 runtime fixes into one clean release baseline.
- Preserved the unified LongMedia H3 lifecycle: one prepare/pre_run/cleanup per LongMedia execution while clips/segments are sampled sequentially.
- Segmentation is strictly isolated to `manual` and `segmented_continuation`; MultiClip and every other workflow cannot inherit segmentation controls or hidden passes.
- Restored real two-stage refinement inside the unified lifecycle: base sigma head followed by zero-noise low-sigma tail with the same seed, sampler and guider.
- Restored the AIMDO/Text-Encoder pinned-memory gate before `NativeReferenceToVideo`, preventing the known HostBuffer weight-fault path on constrained Windows systems.
- Hardened all remaining LongMedia `model_options` wrapper clones against Python `deepcopy()` of CUDA/AIMDO-backed storage.
- Removed abandoned refiner regression tests and unreleased experimental release-note files; synchronized package version metadata.

## v0.4.8

- Restored the proven AIMDO pinned-memory gate for MiniMaxH3TEModel / NativeReferenceToVideo setup.
- The gate now runs before reference/text-encoder weight faults, not only during ultra-low-VRAM sampling.
- Existing CLIP/TE pins are released, pinned memory is temporarily disabled for native reference conditioning, and the original ComfyUI setting is restored immediately afterward.
- No changes to refiner, segmentation isolation, or unified runtime behavior.

## v0.4.7

- Restored the real two-stage refiner execution.
- Base and refine tail now run inside the same unified H3 model lifecycle.
- Refiner uses zero added noise, the same effective seed, sampler and guider wrappers.
- Segmentation isolation and all v0.4.6 workflow gating remain unchanged.

## v0.4.6

- Segmentation is now strictly owned by `manual` and `segmented_continuation` workflows only.
- `segment_duration` and `overlap_frames` are ignored outside those two modes.
- Ordinary workflows are hard-collapsed to a single H3 pass and can no longer accidentally enter continuation segmentation.
- MultiClip remains multi-clip, but is explicitly not segmentation; segmentation overlap is forced off there.
- Runtime reports no longer infer segmentation from `passes > 1`; they use the explicit plan flag.

## v0.4.5

- Fixed unified runtime crash caused by Python `deepcopy()` of CUDA/AIMDO-backed `model_options`.
- Unified runtime and per-segment guider cloning now use ComfyUI `create_model_options_clone()` exclusively.
- Preserves the single model lifecycle introduced in v0.4.4.

## v0.4.4

- Reworked LongMedia multi-segment sampling into one unified model lifecycle.
- `prepare_sampling`, model `pre_run`, and cleanup now happen once per LongMedia generation instead of once per segment.
- Segment continuation still runs sequentially, but through `guider.inner_sample()` while H3 stays prepared/resident.
- Removed graph-level sampler nodes from the LongMedia segment loop, preventing repeated H3 model initialization between clips.
- Refiner remains a logical low-noise tail inside the same single sampler schedule and does not create a second render cycle.

## v0.4.3

- Refiner now always runs inside a single SamplerCustomAdvanced execution.
- Removed the second graph-level sampler pass that could restart the full render/execution lifecycle.
- `refine_steps` now marks the logical low-noise tail within the same connected schedule for reporting and UI semantics.

# 0.4.2

- Fixed refiner VRAM/OOM regression by removing the separate `comfy.sample.sample()` / reconstructed CFGGuider execution path.
- Stage 2 now runs through the same LongMedia `GUIDER`, `SAMPLER`, Sol/MLP wrappers and VRAM governor path as stage 1 using a second native `SamplerCustomAdvanced`.
- Refiner SIGMAS are now the exact low-sigma tail `sigmas[switch_step:]`, sharing the boundary sigma with stage 1 like stock ComfyUI `SplitSigmas` chaining.
- Added internal seeded zero-noise carrier: stage 2 injects no new noise while preserving the exact effective seed forwarded into `guider.sample()`.
- Removed the obsolete stock-refiner runtime node that could bypass LongMedia execution wrappers and fall back into external MiniMax attention patches.
- Sampling remains one continuous trajectory: `main_steps + refine_steps = total connected SIGMAS steps`.

# 0.4.1

- Fixed the LongMedia refiner to use a true two-stage KSampler Advanced sampling trajectory instead of replaying low-sigma steps on an already completed latent.
- With refinement enabled, the connected SIGMAS schedule is split at `total_steps - refine_steps`: stage 1 stops with leftover noise, and stage 2 continues the same trajectory with `add_noise=disable` to the final denoise.
- Refiner uses the same effective seed as the corresponding main sampler pass; no new starting noise is generated for stage 2.
- Removed abandoned experimental refiner paths (identity carrier, direct-Euler replay, x0 substitution and manual tail-start wrappers) from the release runtime.
- `release_guard` remains permanently enabled internally and is no longer exposed as a Long Media Setup switch.
- No changes to the Unified Clip Engine, MultiClip/segmentation timeline ownership, continuity policy, lip-sync pipeline, OOM Governor V4 or streamed Sol execution.

# 0.4.0

- Promoted the validated 0.3.115 baseline to the first stable 0.4 release.
- Unified fixed segmentation and planned MultiClip under one clip executor; only timeline-boundary math differs.
- Fixed Planner workflow ownership and bidirectional vertical resize behavior.
- Removed obsolete public `video_strength` / `audio_strength` controls and their backend plumbing.
- Hardened lip-sync/source-audio handling around the authoritative `audio_1` timeline.
- Added geometry-aware OOM Governor V4 and preventive attention preflight for very long sequences on constrained GPUs.
- Added bounded streamed Sol QKV routing for sequences that cannot safely allocate full Sage attention workspaces.
- Removed development hot-reload/runtime helpers and accumulated per-dev-build notes from the release package.
- Added release documentation for MultiClip prompting, fixed segmentation prompting, architecture, sampler/VRAM optimization, and release audit.
- Set package, project and VERSION metadata to `0.4.0`.

## 0.3.115-dev

- Fixed LongMedia Planner vertical resize becoming non-shrinkable after expansion by decoupling runtime viewport height from LiteGraph minimum-size computation.

# v0.3.108-dev

- Fix lip-sync regression: preserve authoritative local-0 source-audio guides across continuation Motion Context; restore pass-0 guide; suppress competing sampled audio tail in lip-sync mode.

## v0.3.105-dev
- lip_sync: remove extra per-clip Audio Guide; restore full Extender-style AV Motion Context audio tail from the same sampled AV latent while preserving full native Audio1 Ref2VA conditioning.

## v0.3.103-dev
- Diagnostic lip-sync build based directly on v0.3.96.
- `audio_mode=lip_sync` now outputs H3-generated target audio instead of restoring `audio_1`.
- Conditioning / MultiClip / Motion Context are unchanged from the v0.3.96 baseline.

## v0.3.90-dev
- Fix LongMediaPlan schema mismatch for `release_guard`; Setup no longer crashes in `dataclasses.replace()`.
- No generation or memory-policy changes.

## v0.3.89-dev
- Fixed Planner connection UI lifecycle: connecting `clip_plan` now immediately hides Setup's duplicate MultiClip editor and redundant workflow selector.
- Dynamic input labels now follow external Planner MultiClip state.

## 0.3.88-dev

- Added standalone **MiniMax H3 LongMedia Planner** node for per-clip prompt/duration/seed planning.
- Long Media Setup now accepts an optional `clip_plan` and automatically switches to MultiClip execution when connected.
- The embedded MultiClip editor remains as a fallback when no Planner is connected.

## 0.3.87-dev
- MultiClip card editor UI with per-clip prompt/seed/duration; hides irrelevant global duration controls in MultiClip mode.

## 0.3.84-dev
- Two-pass identity-only latent re-anchor without Picture tokenizer re-entry.

## 0.3.83-dev
- Native H3 motion-context backend for the first handoff of exactly two segmented passes.
- Fresh continuation head + exact temporal keyframes/audio timeline; no copied frozen head in this path.
- 3+ pass behavior unchanged.

## 0.3.82-dev
- AIMDO Setup/TE lifecycle hardening for native DynamicVRAM fastpath.
- Setup boundaries now synchronize CUDA, cleanup Comfy prefetch queues, reset cast buffers and reset AIMDO VBAR watermark limits.
- One lifecycle-only retry for transient `Fault failed` / `device not ready` TE faults.
- H3 native AIMDO fastpath, two-pass continuity, SIGMAS/refine/audio math unchanged.

## 0.3.80-dev

- First-handoff bridge for `segmented_continuation`: only segment 0 -> 1 receives up to 56 frames of generated raw paired AV history as conditioning-only context.
- Frozen overlap/stitch remains exactly user-selected (22f in the standard case).
- Original still-image refs remain decoupled after pass 0; later continuation->continuation boundaries retain the proven 0.3.77 behavior.
- Native AIMDO fastpath and H3 sampling math are unchanged.

## 0.3.77-dev
- Native AIMDO fastpath A/B for oversized INT8 on comfy-aimdo >=0.4.6; legacy hard-gate bypassed, H3 math unchanged.

## 0.3.75-dev
- conservative 0.3.65-based RAM/file-cache prewarm for oversized AIMDO-backed H3 checkpoints; no H3 math, VBAR, or runtime-governor changes

## 0.3.65-dev
- Fix DEV HOT RELOAD baseline capture: fingerprint is recorded at package import instead of first queue, so replacing package files after ComfyUI startup reliably reloads the new runtime on the very first execution; preserves 0.3.64 bound VBAR governor and 0.3.61 quality-safe math.

## 0.3.64-dev
- bind VBAR residency governor directly to the active H3 ModelPatcher/state via the DIFFUSION_MODEL wrapper; removes reliance on transformer_options object-reference propagation and adds first-forward wiring diagnostics

## 0.3.63-dev
- moves AIMDO/VBAR residency promotion from sampler callback to the authoritative H3 diffusion-forward boundary; first forward remains safe, subsequent forwards reopen the model watermark only above mode-specific real driver-free VRAM floors

## 0.3.62-dev
- adaptive AIMDO/VBAR residency governor: safely reopens the VBAR offload watermark at completed denoise-step boundaries when driver-free VRAM is above a per-memory-mode promotion floor; preserves v0.3.61 H3 math exactly

## 0.3.61-dev
- replace dangerous tiny-probe cached INT8 MLP shortcut with real-chunk parity gating, explicit SwiGLU, QuantizedTensor/F.linear dispatch, and correct fc2 preparation on post-SwiGLU input shape; keeps Governor V3 memory policy

## 0.3.60-dev
- restores block-resident native INT8 fc1/fc2 reuse to eliminate v0.3.59 per-chunk weight re-streaming; block-0 stock-vs-cached numerical parity gate disables the fast path automatically on divergence
- Adaptive Residency Governor V3 lowers artificial planning reserves and uses more available VRAM in normal/low/ultra while preserving hard runtime floors and out-of-core pinned/prefetch protections

## 0.3.59-dev
- Stock-math H3 execution: disable custom prepared/cached INT8 MLP fast-path in every memory mode; token chunking is activation-only and calls stock Comfy block.mlp()
- Adaptive Residency Governor v2 is active for normal, low_vram, ultra_low_vram and AUTO-selected profiles; modes now define different safety envelopes instead of fixed memory behavior
- reduce ultra fixed activation reserve from the old 8GB-class policy to a 3-4GB planning envelope so safe VRAM can remain resident instead of idling
- out-of-core models disable pinned-memory AIMDO and whole-block dynamic VBAR prefetch regardless of manually selected memory mode
- governor uses model/VRAM ratio, observed packed token count and real CUDA driver-free memory; RAM availability is sampled during planning

## 0.3.58-dev
- H3 single-lifecycle refine: execute the full connected SIGMAS schedule in one SamplerCustomAdvanced call, eliminating the intermediate noisy-latent handoff and second sampler initialization; preserve exact frozen AV overlap after continuation passes

## 0.3.57-dev
- MiniMax H3 native LoRA compatibility layer: adds Diffusers/PEFT-style transformer_blocks.* aliases for ff, attn out_proj, and fused qkv slice mapping; includes transformer/base_model/lycoris-style prefixes without changing sampler or memory logic

## 0.3.56-dev
- adaptive driver-free VRAM governor for AUTO out-of-core H3; fixed ultra_low_vram remains safe fallback

## 0.3.55-dev
- ultra_low_vram sampler-local pinned-memory gate for AIMDO HostBuffer OOMs; no CLI flags required

## 0.3.54-dev
- ultra_low_vram sequential MLP weight streaming: disable simultaneous cached fc1+fc2 residency; 128-token FFN chunks.

## 0.3.53-dev
- ultra-low-VRAM attention-to-FFN stage residency barrier for 32+ GB staged H3 models.

## 0.3.52-dev
- hard-gate Comfy dynamic VBAR prefetch after BaseModel override for low/ultra memory modes; works with existing/Sage attention too.

## 0.3.51-dev
- sampler-local auto/normal/low_vram/ultra_low_vram out-of-core residency profiles; no CLI flags required.

## 0.3.50-dev
- replaces the old additive refine pass with a true two-stage split of one connected SIGMAS schedule
- `steps=12, refine_steps=3` now executes 9 main intervals + 3 refine intervals = 12 total model steps, not 15
- refine always uses DisableNoise and continues from the partially denoised main latent
- audio now follows the same complete diffusion trajectory; only exact frozen continuation AV overlap is restored after refine
- legacy refine_add_noise/refine_seed inputs remain in the schema only for saved-workflow compatibility and are ignored
- keeps v0.3.49 resolution safety and all v0.3.45 segmented-continuation/reference-decoupling behavior intact

## 0.3.49-dev
- decouple still-image Ref2VA conditioning resolution from output resolution; cap refs to stable ~0.60 MP and enforce 32px patch-safe geometry
- normalize connected target width/height values to the H3 32px grid
- based on v0.3.45; segmented continuation semantics unchanged

## 0.3.45-dev
- Advanced refine now derives its interval automatically from connected SIGMAS; only refine_steps is user-controlled

## 0.3.44-dev
- optional Advanced-style per-segment refine with protected audio and continuation overlap

## 0.3.43-dev
- segmented_continuation decouples original still-image refs after pass 0 to prevent literal source-image reinsertion

## 0.3.41-dev
- stronger 2-frame visible startup-anchor suppression after decode

## 0.3.40-dev
- segmented_continuation now supports dedicated opening_frame while preserving the 0.3.39 regression-safe continuation path

## 0.3.39-dev
- regression-safe segmented mode restores v0.3.32 hybrid_first_frame sampling and suppresses only decoded frame-0 anchor flash

## v0.3.31-dev — Stateful multi-pass continuity

- Carry completed timeline events forward as established scene state rather than replay commands.
- Add strict motion/composition continuation contract for every continuation pass.
- Bound state history and validate 4-pass operation.
- Sort merged motion keyframes chronologically.
- Disable hidden-overlap latent crossfade; overlap is frozen context preroll and is trimmed at stitch.
- Preserve v0.3.28 workflow-mode UI lifecycle and v0.3.30 single-step latent injection.

## v0.3.30 — Single-Step Hybrid Latent Injection

- fixes the v0.3.29 Hybrid startup crash `MiniMax H3 video latent time must be 5*k+2, got 1`;
- recognizes a VAE-encoded keyframe `[B, 24, 1, H, W]` as a valid **single-step injection payload**, without treating it as a complete standalone H3 video;
- preserves strict `5*k+2` validation for every complete video/AV stream and retains exact batch/channel/spatial compatibility checks before copying the keyframe into the target latent;
- keeps the v0.3.29 segment-local prompt, stable native-reference topology, Picture-tag repair and Manual UI fixes unchanged;
- adds a torch-free regression that reproduces the exact `T=1` contract and rejects malformed channel layouts.

## v0.3.29 — Stable Native References + Segment Timeline

- scopes timestamped prompt events for **every** pass, including pass 0; a future `07 sec` event can no longer leak into a 5-second opening pass and then replay after the join;
- preserves the exact native H3 reference count, order, tokenizer presentation and latent geometry across segmented Hybrid passes; distinct character refs are never collapsed into one continuation-only identity sheet;
- repairs unambiguous input-socket-style `<Picture N>` tags in Hybrid prompts, where first/last-frame anchors do not consume native Picture ordinals, and reports the applied mapping;
- makes `first_frame_mode=latent_inject` effective in Hybrid mode by pinning the already-encoded opening keyframe into the first target latent step before sampling;
- applies `pixel_override` and `blend` to Hybrid output as well as lip-sync, and relocates final-frame anchors to the final pass's local last frame;
- relabels Manual image sockets from `conditioning_mode`, refreshes those labels immediately when the mode changes, and keeps connection-triggered labels consistent;
- adds a pure timeline/reference policy module plus regressions built from the embedded workflow in `MiniMax_H3_00002-audio.mp4`.

## v0.3.28 — LongMediaSetup Manual UI Lifecycle

- fixes Manual controls remaining collapsed after repeated public-mode refreshes: widget presentation state is now captured exactly once, including the valid case where `computeSize`/`draw` are inherited or undefined;
- preserves valid expert values across `manual -> public -> manual` instead of destructively replacing first-frame and conditioning settings with public defaults;
- makes workflow loading atomic for the frontend: callbacks are installed during graph restoration, while visibility, labels and node size are reconciled once after configuration completes;
- assigns workflow-mode visibility and resizing to one frontend owner, removing the competing asynchronous callback previously installed by the dynamic-socket extension;
- keeps lip-sync socket labels stable after connections add or remove dynamic inputs, and adds a standalone Node regression suite for visibility, sizing, value persistence, callback ownership and socket lifecycle;
- keeps the v0.3.27 segment-continuity pipeline and serialized backend schema unchanged.

## v0.3.27 — Segment Continuity Integrity

- removes the v0.3.26 pass-0 startup echoes at frames 5/22/39; segmented Hybrid now uses only the native frame-0 anchor, preventing the hard anchor-release identity/pose flip observed at frames 53–55 (2.208–2.292 s at 24 fps);
- disables the v0.3.23 cross-time visible-seam latent interpolation; the aligned hidden overlap still smoothstep-blends, but the first new continuation latent step is no longer replaced by a 50/50 mix with the preceding time step;
- restores the v0.3.25 continuation identity rule: `image_1` remains an opening timeline anchor and is never promoted into the pass-2 Picture sheet; zero/one explicit Picture refs keep native semantics, while only 2+ non-terminal Picture refs may be combined;
- locks segmented `attention_mode=auto` to `existing` for every pass, preventing an H3-alignment/token-threshold change from silently switching one movie from existing/Sage to approximate Sol mid-run; explicitly forced `sol` and `scheduled_sol` remain unchanged;
- keeps the 22-frame frozen overlap, runtime motion guides, same-seed policy, source/reference timeline alignment, low-VRAM INT8 demand residency, audio, and tiled VAE decode unchanged.

## v0.3.26 — Startup Stabilizer + Continuation Identity Gate

- adds a conservative **pass-0 startup stabilizer** for segmented hybrid runs: the opening keyframe (`image_1`) is echoed as short-range internal anchors at early frames (5 / 22 / 39 when available) to reduce the head-flip / identity wobble seen at the start of some clips.
- those startup anchors are **explicitly stripped from continuation passes**, so the stabilizer only affects the opening beat and does not freeze later motion.
- changes continuation hybrid-image handling so pass 2+ use **one continuation-only identity sheet** built from `image_1` plus non-terminal image refs. In `hybrid_first_last`, `image_2` remains the terminal destination anchor and is **not promoted as a continuation reference**.
- keeps the existing segmentation pipeline, overlap stitching, seed reuse, runtime motion handoff, and the v0.3.25 low-VRAM INT8 residency behavior unchanged.

## v0.3.25 — Low-VRAM INT8 Demand-Loaded Residency

- fixes a native INT8/ConvRot startup OOM where Comfy/AIMDO `prefetch_queue_pop()` attempted a speculative 64 MB device copy before the first denoise step;
- on GPUs with <=18.5 GB VRAM, native INT8 now disables dynamic-VBAR prefetch and uses demand-loaded quantized-weight residency;
- W4A8 keeps its existing prefetch-off policy; >18.5 GB native INT8 keeps the previous Comfy prefetch behavior;
- preserves the v0.3.24 INT8 AUTO-SOL quality-safe tau profile and does not change segmentation, runtime motion handoff, seed/sigmas, audio, or decode.

## v0.3.23

- keeps the v0.3.22 segmented runtime intact but softens the *visible* inter-segment handoff in addition to the hidden overlap blend;
- stitches now smooth one tiny latent step on **both sides** of the visible seam, reducing the 4–6 second “double jump” / cadence break many segmented 10s renders showed around the 5s boundary;
- length, FPS, and H3 `5*k+2` latent-grid invariants are preserved exactly; the existing boundary auditor still validates stitched frame/audio synchronization;
- does not change seeds, sigma policy, SOL tau, source prompt timing, or the established runtime motion-context handoff.

## v0.3.22

- keeps all sampler tuning widgets visible in `sampler_mode=auto`;
- AUTO starts from the same production defaults but now honors explicit widget overrides, enabling clean AUTO+existing vs AUTO+SOL A/B tests;
- stops frontend refresh/mode changes from resetting AUTO tuning widgets;
- preserves v0.3.21 INT8 residency hysteresis and the frozen segmentation/continuation implementation.

## v0.3.21 — AUTO INT8 Residency Hysteresis Fix

- made native INT8 AUTO residency policy safe for oversubscribed models (for example a ~20 GB H3 checkpoint on a 16 GB GPU);
- emergency cleanup now uses effective reclaimable headroom (`driver_free + allocator cache`) instead of treating low driver-free VRAM alone as an OOM condition;
- added an 8-block cooldown after a genuine emergency trim to prevent block-by-block `soft_empty_cache()` thrashing;
- reduced the cache-reclaim floor and made cleanup a strict dual-threshold emergency path;
- does not change SOL tau, chunk sizing, seed/sigma policy, segmentation, runtime motion handoff, audio, or VAE decode behavior.

## v0.3.20

- moved continuation motion-context guider construction from GraphBuilder expansion to a runtime node that depends on the completed previous sampler output;
- V60 motion guides can now inspect the actual previous AV latent tail instead of receiving a graph-output proxy;
- added `[V320 RUNTIME MOTION HANDOFF]` diagnostics with previous-frame, overlap and guide-span information;
- kept the v0.3.19 PackedLayout compatibility fix, v0.3.18 timeline/boundary audit, and existing frozen-overlap/stitch behavior unchanged so this build tests one continuation hypothesis only.

## v0.3.19

- fixed `PackedLayout.__init__()` compatibility with current ComfyUI builds that do not expose the optional `frame_count` keyword; the motion-context layout wrapper now forwards it only when the wrapped constructor actually accepts it;
- stopped V60 auxiliary motion-guide setup from treating GraphBuilder output proxies as runtime LATENT dictionaries; the real frozen latent overlap remains active via `LongMediaNextSegment`;
- preserved the v0.3.18 unified timeline/boundary auditor and all v0.3.17/v0.3.16 continuation/decode fixes.

## v0.3.18 — Unified Segment Timeline + Boundary Auditor

- formalized one canonical timeline contract per pass: `context_start`, `visible_start`, `local_visible_offset`, visible length/end;
- fixed continuation audio-reference alignment: full H3 reference windows now begin at `context_start`, keeping hidden overlap at local t=0..overlap and the music's visible boundary aligned with the target at local t=overlap;
- source video/audio continuation windows remain context-aligned while only their post-overlap latent region populates new visible media;
- added `[V318 TIMELINE]` diagnostics for context/visible/source/reference origins on segment 2+;
- added a strict `[V318 BOUNDARY AUDIT]` after every stitch to verify `previous + next - overlap == stitched` and exact H3 audio/video latent synchronization;
- preserves v0.3.17 seam blending and all v0.3.16 decode-memory, v0.3.15 layout, v0.3.14 UI, and v0.3.13 segment-duration fixes.

## v0.3.17

- improved segmented stitching by enabling smooth latent overlap blending for automatic multi-pass continuation, reducing visible seams at segment joins;
- corrected continuation audio/music reference timing so segment audio refs start at the user-visible boundary after overlap, improving sync on segment 2+;
- added `[V317 AUDIO TIMELINE SYNC]` logging when audio reference slicing is shifted from context origin to visible origin;
- preserved all v0.3.16 decode-memory, v0.3.15 tag/layout guard, v0.3.14 manual UI, and v0.3.13 segment-duration baseline fixes.

## 0.3.16 — 2026-08-14

- Fixed fatal long-video VAE decode OOM/abort seen after successful segmented sampling.
- `enable_tiling=True` now bypasses regular full decode and goes directly through ComfyUI `VAE.decode_tiled()`.
- Added hard pre-decode model unload/cache barrier and VRAM diagnostics.
- Added direct spatial + temporal tile parameter conversion matching ComfyUI VAEDecodeTiled semantics.
- Preserved v0.3.15 segmented tag/layout guard and all v0.3.11+ audio/AUTO/low-VRAM behavior.

## v0.3.15 — Segmented H3 tag/layout guard

- Fixed an intermittent segmented Hybrid/Loop crash in stock MiniMax H3: `IndexError: list index out of range` while walking `text_token_tags`.
- At the DIFFUSION_MODEL boundary, LongMedia now aligns presentation-tag length to the actual encoded context length without mutating cached conditioning. Missing tail rows are ordinary text modality (`tag=1`); unreachable surplus rows are truncated.
- Added a cheap PackedLayout integrity check (contiguous segments, `seq_len`, `position_ids`, and text-span/context agreement) so a real layout corruption reports explicit dimensions instead of an opaque H3 index error.
- Keeps v0.3.14 live Manual UI, v0.3.13 segment-duration semantics, v0.3.12 anchor timeline correction, and the v0.3.11 audio/low-VRAM baseline unchanged.

## v0.3.14 — Live Manual UI fix

- Fixed Sampler `sampler_mode=manual` controls not appearing until a page reload on current ComfyUI frontends.
- Dynamic widget visibility now uses `options.hidden` and keeps the original widget type stable.
- Added legacy LiteGraph size/draw fallback without converting widgets to `converted-widget`.
- Preserves v0.3.13 segment-duration semantics and v0.3.12 hybrid anchor timeline fix.

## v0.3.13 — Segment Duration + Anchor Timeline baseline


- Restored public multi-pass segmentation for `hybrid_auto`, `ref2va_full`, `loop`, and `video_ref_edit`.
- `segment_seconds` remains the serialized backend field for workflow compatibility, but is displayed as `segment_duration` in the Setup UI.
- `segment_duration` means **new visible output timeline per pass**. `overlap_frames` is extra hidden continuation context and no longer reduces the requested segment duration.
- Removed the old public-mode override that forced `segment_seconds=600` and `overlap_frames=0`, which effectively disabled segmentation outside Manual.
- Public modes now honor the planned overlap; the overlap control itself remains visible only in Manual to keep the normal UI compact.
- Includes the v0.3.12 ref-aware first/last anchor timeline placement fix.
- Keeps the v0.3.11 audio, lip-sync, AUTO Sol, schema, and low-VRAM Manual baseline unchanged.

# v0.3.12 — Hybrid Anchor Timeline Fix

- Fixed first/last H3 keyframe anchors being displaced into the past when Hybrid or Loop conditioning is combined with packed image/video/audio references.
- Hybrid keyframes now carry the shared `motion_context_index` target-frame marker.
- The marker-gated `PackedLayout` adjustment is activated only when keyframes and refs coexist, translating anchor rows onto the true target-video origin after the packed reference span.
- No-ref workflows retain stock H3 keyframe placement.
- Built directly on the v0.3.11 baseline: audio passthrough/preserve fixes, lip-sync, quality-safe AUTO Sol, live Manual UI, schema compatibility, and low-VRAM fine-step controls are unchanged.

# v0.3.11 — Low-VRAM Manual Tuning & Audio Stability Baseline

- New development baseline for subsequent LongMedia builds.
- Manual Sampler token/chunk controls now use 512-token increments for practical tuning on 8–12 GB GPUs.
- VRAM reserve/late-target controls use 256 MB increments; smaller guard/cleanup thresholds use 128 MB increments.
- Existing defaults and serialized widget order are unchanged, preserving workflow compatibility.
- Includes the current audio passthrough/reference fixes: `auto`, `preserve`, and `preserve_reference` consistently preserve source audio where intended and bypass invalid AudioVAE reconstruction paths.
- Includes Hybrid Auto, Ref2VA Full, Loop, Video Reference Edit, Lip Sync, dynamic UI, live Manual sampler controls, and the quality-safe AUTO Sol policy.
- Future releases should be developed on top of this baseline.

# v0.3.1 — Audio Passthrough & Reference Routing Hotfix

- Manual Sampler low-VRAM tuning now uses fine-grained steps: 512 tokens for MLP/QKV/out-proj chunks, 256 MB for activation reserve/late target, and 128 MB for smaller VRAM guards/cleanup thresholds. Defaults and serialized widget order are unchanged.
- Fixed `audio_mode=auto` passthrough in native Ref2VA/T2V/reference branches: attached source audio is now preserved consistently instead of falling back to model-audio decode.
- `preserve` and `preserve_reference` now hard-bypass AudioVAE reconstruction at final decode and return the source waveform.
- Prevents AudioVAE normalizer crashes such as `19296 vs 128` when Turbo LoRA/model audio latent geometry does not match the decoder expectation.
- Keeps source audio available as H3 reference/driver for rhythm, timing and lip-sync while allowing the untouched source soundtrack to be restored at output.
- Retains the v0.3.0 Hybrid / Ref2VA / Loop / video reference edit / lip-sync UI and quality-safe AUTO Sol policy.

# 0.3.0 audio passthrough branch fix
- Fixed `audio_mode=auto` inconsistency: attached audio is now preserved in every workflow branch, including native Ref2VA/T2V.
- Setup guarantees `final_audio_override` for attached audio in `auto`, `preserve`, and `preserve_reference`.
- Decode hard-bypasses AudioVAE whenever a passthrough soundtrack is available.
- Added explicit H3 audio-latent geometry validation before generated-audio decode to replace opaque AudioVAE broadcast failures with a useful error.

# 0.3.0 preserve-audio decode hotfix
- `preserve` and `preserve_reference` now carry an explicit `audio_output_mode` in `LONG_MEDIA_PLAN`.
- Decode treats preserve modes as a hard audio-VAE bypass; sampled/model audio is never reconstructed.
- If preserve is requested without a connected source audio track, Decode raises a clear configuration error instead of attempting to decode an invalid audio latent.
- Fixes `19296 vs 128` Audio VAE normalizer shape crashes seen with Turbo LoRAs.

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

## 0.4.0 refiner correction
- Corrected additive refiner math: the base sampler now always executes its complete SIGMAS schedule.
- `refine_steps` are extra model steps, not steps removed from the base sampler.
- Refiner SIGMAS are copied from the exact final `refine_steps` intervals of the base schedule.
- Example: `steps=12`, `refine_steps=3` executes 12 base + 3 refine = 15 model evaluations; the refiner uses `sigmas[-4:]`.
- Refiner uses `DisableNoise`; continuation overlap is restored after the additive refine pass.


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

## 0.4.36-DEV
- Fixed loss of ModelPatcher-owned H3 SLA attention override in LongMedia.
- Added native zero-copy SLA execution inside the H3 block path.
- Eliminated full-size SLA `o_s` and contiguous Q/K/V copies.
- Added strided LightX2V-compatible routing/kernel and streamed out projection.
