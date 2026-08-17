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
