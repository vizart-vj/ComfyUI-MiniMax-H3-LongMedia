# LongMedia Architecture

This document describes the current LongMedia 0.6.42 runtime architecture. Public control names and internal identifiers are kept exactly as they appear in the UI/code.

## Semantic Setup Contract

The public Setup is no longer defined by one monolithic `workflow_mode`.

New workflows use independent controls:

```text
control_mode
h3_mode
timeline_mode
duration_source
audio_mode
```

`workflow_mode` remains internal only for compatibility with older saved workflows.

See [LongMedia Operating Modes](MODES_GUIDE_EN.md).

## Conditioning Families

`h3_mode` selects the H3 conditioning family:

- `t2va` — pure text-to-video/audio;
- `fl2va` — native first/last-frame anchors;
- `ref2va` — native Picture/Video/Audio references;
- `hybrid` — frame anchors plus LongMedia reference behavior;
- `video_ref_edit` — source-video editing / replacement.

The conditioning family is independent from timeline construction.

## Timeline Engine

`timeline_mode` selects timeline ownership:

```text
single
segmented
multiclip
```

### Single

One target AV latent and one logical movie timeline.

### Segmented

LongMedia creates fixed-duration internal units and carries native H3 continuation context between them. It is intended for one continuous semantic scene, not storyboard cuts.

### MultiClip

Planner owns clip prompts, durations, names and optional per-clip seeds. Cameras can add non-diegetic cinematography while preserving stable clip identity through `clip_id`.

Planner is authoritative only in `timeline_mode=multiclip`.

## Shared AV Latent

MiniMax H3 operates on a nested AV latent:

```text
video: [B, 24, T, H, W]
audio: [B, 32, 2, T40]
```

LongMedia preserves native MiniMax temporal-lattice rules and validates AV duration alignment at assembly boundaries.

## References and Source Editing

Picture, Video and Audio inputs are separate modalities.

A ComfyUI video IMAGE batch never carries soundtrack data. Source audio must be connected separately.

For `video_ref_edit`, LongMedia can build native paired Video1+Audio1 source-performance conditioning while also using Audio1 as the authoritative timing/output source in preserve-style modes. `lip_sync` instead treats Audio1 as an independent dub so replacement speech does not need to match Video1's original soundtrack.

Audio2/Audio3 remain independent prompt-addressable references.

## Cameras Layer

Long Media Cameras is a separate non-diegetic direction layer.

Recommended MultiClip ownership:

```text
Planner scene/action prompts
        │
        ▼
Cameras cinematography compiler
        │
        ▼
Setup conditioning
```

When camera guidance exists, camera directives are stripped from Planner text before compiled camera instructions are appended. This prevents competing framing/motion ownership.

## Director Temporal Controls

Director keeps the common scene/subject/reference presentation shared across a complete MAIN shot. Partial CAMERA and EMBEDDING blocks are appended as independent temporal controls rather than repeated copies of the full scene prompt.

Camera and embedding boundaries are independent. Authored timing uses the shared 24 fps clock; local sampler/refine windows project that global clock rather than creating new editorial timelines.

## Integrated Refine / Latent Hi-Res

The current Sampler contains an internal Stage 2 controlled by `refine_enabled`.

Contract:

1. MAIN Sampler completes Stage-1 x0;
2. when Latent Hi-Res is active, the learned upscaler changes only the video latent;
3. Stage-1 audio becomes authoritative exact passthrough;
4. Refine uses the separate `refine_sigmas` schedule;
5. long/high-resolution video latents use bounded temporal windows;
6. final audio is restored exactly from Stage 1.

Refine is not an audio refiner.

See [Integrated Refine and Latent Hi-Res](TWO_PASS_LATENT_HIRES_REFINER_GUIDE_EN.md).

## Dynamic VRAM and Exact Attention

LongMedia coordinates activation lifetime with ComfyUI Dynamic VRAM/AIMDO.

Current mechanisms include:

- streamed/chunked transformer MLP and output projections;
- embedded Sol paths for bounded long-sequence execution;
- exact Comfy Kitchen EXISTING query streaming when full fused QKV cannot fit;
- inter-block and step-boundary VRAM guards;
- guarded native INT8 residency on constrained GPUs;
- sampler-entry memory isolation for cache-driven reruns;
- release of transient CUDA references that could survive through cached guider/runtime state;
- RAM-pressure-aware pinned-host-memory policy.

On constrained native INT8 systems, speculative dynamic-VBAR prefetch is gated so the next block does not reserve a competing transfer destination while the current activation set is live.

## Model Compatibility Layers

LongMedia contains isolated compatibility paths for:

- stock MiniMax H3;
- supported native INT8/W4A8 ComfyUI weights;
- H3ddle/PulpCut FastH3 VSA packages;
- Kijai FastVideo VSA packages.

FastH3/FastVideo adaptations are detected structurally and fail closed when their trained contract is not satisfied. Runtime state is reset when returning to ordinary H3 checkpoints.

## Reconstruction

The Video Reconstructor builds source-video edit plans on top of the same H3/Ref2VA foundation. Later reconstruction revisions add detail-recovery passes while preserving low-frequency source geometry and AV timing contracts.

## Loop Closure

Loop Closure is independent from timeline/conditioning selection. It regenerates or attracts the tail toward the opening macro-state in latent/H3 space rather than applying an RGB crossfade.

## Compatibility

Legacy internal class identifiers such as `MiniMaxH3LatentLab...` remain registered so older ComfyUI workflows can resolve the nodes.

Legacy `workflow_mode` values are migrated into semantic Setup controls at load/runtime boundaries.
