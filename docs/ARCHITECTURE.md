# LongMedia Architecture

This document describes the current LongMedia runtime used by the 0.5.40 release.

## Semantic Setup Contract

The public Setup UI is no longer defined by one monolithic `workflow_mode`.

New workflows use four independent controls:

```text
control_mode
h3_mode
timeline_mode
duration_source
```

`workflow_mode` is retained internally only for compatibility with older saved workflows.

See [LongMedia Operating Modes](MODES_GUIDE.md).

## Conditioning Families

`h3_mode` selects the H3 conditioning family:

- `t2va` — pure text-to-video/audio;
- `fl2va` — native first/last frame;
- `ref2va` — native Picture/Video/Audio references;
- `hybrid` — opening keyframe plus LongMedia reference behavior;
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

LongMedia creates fixed-duration internal units and carries native H3 continuation context between them. It is intended for one continuous semantic prompt.

### MultiClip

The Planner owns clip prompts, durations, names, and optional per-clip seeds. Cameras can add non-diegetic cinematography while preserving clip identity through stable `clip_id`.

The Planner is authoritative only in `timeline_mode=multiclip`.

## Shared AV Latent

MiniMax H3 operates on a nested AV latent:

```text
video: [B, 24, T, H, W]
audio: [B, 32, 2, T40]
```

LongMedia preserves native MiniMax temporal lattice rules and validates AV duration alignment at assembly boundaries.

## References and Source Editing

Picture, Video, and Audio inputs are separate modalities.

A ComfyUI video IMAGE batch never carries soundtrack data. Soundtrack audio must be connected separately.

For `video_ref_edit`, current LongMedia can build native paired Video1+Audio1 source-performance conditioning while also keeping Audio1 as the frozen target timing source for preserve-style modes. `lip_sync` instead treats Audio1 as an independent authoritative dub so replacement speech does not pretend to be the original Video1 soundtrack.

Audio2/Audio3 remain independent prompt-addressable references.

## Cameras Layer

Long Media Cameras is a separate, non-diegetic direction layer.

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

When camera guidance exists, camera directives are removed from Planner text before compiled camera instructions are appended. This prevents two subsystems from fighting over framing and motion.

## Two-Stage Sampling

The sampler can split one scheduler into a base pass and refine pass.

Without Latent Hi-Res, Stage 2 is the low-sigma tail and continues with zero new noise.

With Latent Hi-Res, LongMedia:

1. obtains the Stage-1 denoised x0;
2. learned-upscales only the video latent;
3. preserves the audio latent;
4. rebuilds target-grid conditioning for the new H/W;
5. runs an independent same-seed fresh-noise high-resolution H3 pass.

See [Two-Stage Sampling, Latent Hi-Res and Refiner](TWO_PASS_LATENT_HIRES_REFINER_GUIDE.md).

## Dynamic VRAM and Exact Attention

LongMedia coordinates activation lifetimes with current ComfyUI Dynamic VRAM/AIMDO.

Important current behaviors include:

- streamed/chunked transformer MLP and output projections;
- embedded Sol paths for bounded long-sequence execution;
- exact Comfy Kitchen EXISTING query streaming when full fused QKV cannot fit;
- inter-block and step-boundary VRAM guards;
- guarded native INT8 residency on constrained GPUs;
- sampler-entry memory isolation for cache-driven reruns;
- RAM-pressure-aware pinned-host-memory policy.

On native INT8 systems at or below the constrained-GPU threshold, speculative dynamic-VBAR prefetch is hard-gated so the next block does not allocate a second transfer destination while the current block is still active.

## Model Compatibility Layers

LongMedia contains isolated compatibility paths for:

- stock MiniMax H3;
- supported native INT8/W4A8 ComfyUI weights;
- H3ddle/PulpCut FastH3 VSA packages;
- Kijai FastVideo VSA packages.

FastH3/FastVideo adaptations are detected structurally and fail closed when their trained contract is not satisfied. Runtime state is reset when switching back to ordinary H3 checkpoints.

## Reconstruction

The Video Reconstructor builds source-video edit plans on top of the same H3/Ref2VA foundation. Later reconstruction revisions added detail-recovery passes while preserving low-frequency source geometry and AV timing contracts.

## Loop Closure

Loop Closure is independent from timeline/conditioning selection. It regenerates or attracts the tail toward the opening macro-state in latent/H3 space rather than applying an RGB crossfade.

## Compatibility

Legacy internal class identifiers such as `MiniMaxH3LatentLab...` remain intentionally registered so older ComfyUI workflows can resolve the nodes.

Legacy `workflow_mode` values are migrated into the semantic Setup controls at load/runtime boundaries.
