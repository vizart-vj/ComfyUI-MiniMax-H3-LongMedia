# ComfyUI-MiniMax-H3-LongMedia

Production-oriented ComfyUI nodes for **MiniMax H3** long-form video/audio generation, native reference editing, Director timeline authoring, MultiClip planning, camera direction, segmentation, lip-sync/redubbing, latent hi-res refinement, reconstruction and adaptive low-VRAM execution.

![screenshot](ex.png)

**Current release: 0.6.40.**

## Documentation

- [English documentation index](docs/README_EN.md)
- [Russian documentation index](docs/README_RU.md)
- [LongMedia Director — Complete Guide](docs/DIRECTOR_GUIDE_EN.md)
- [Operating Modes](docs/MODES_GUIDE_EN.md)
- [Audio Modes and `video_ref_edit`](docs/AUDIO_MODES_GUIDE_EN.md)
- [Integrated Refine and Latent Hi-Res](docs/TWO_PASS_LATENT_HIRES_REFINER_GUIDE_EN.md)
- [Sampler, VRAM and Performance](docs/SAMPLER_OPTIMIZATION_EN.md)

## Main nodes

- **MiniMax H3 • Long Media Setup**
- **MiniMax H3 • Long Media Planner**
- **MiniMax H3 • LongMedia Director**
- **MiniMax H3 • Long Media Cameras**
- **MiniMax H3 • Long Media Sampler**
- **MiniMax H3 • Long Media Decode**
- **MiniMax H3 • Long Media Video Reconstructor**

Legacy internal `MiniMaxH3LatentLab...` class identifiers remain registered for workflow compatibility.

## Installation

Install into:

```text
ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-LongMedia
```

or install from the Comfy Registry.

Package identity:

```text
GitHub:             vizart-vj/ComfyUI-MiniMax-H3-LongMedia
Comfy PublisherId:  noise
```

Restart ComfyUI after installation/update.

## Current semantic Setup model

New workflows use independent controls:

```text
control_mode
h3_mode
timeline_mode
duration_source
audio_mode
motion_repair
```

H3 conditioning families:

```text
t2va
fl2va
ref2va
hybrid
video_ref_edit
```

Timeline modes:

```text
single
segmented
multiclip
```

Duration ownership:

```text
auto
video
audio
manual
longest_input
```

See [Operating Modes](docs/MODES_GUIDE_EN.md).

## LongMedia Director 0.6.40

Director is the unified authoring surface for MAIN timing, prompts, WHO & WHAT media, REF/FIRST/LAST roles, cameras, temporal embeddings, audio policy, resolution policy, Program Monitor review and TAKE management.

```text
LongMedia Director
        ↓ director
Long Media Setup  (control_mode=director)
```

Current TAKE storage is TAKE-centric. CREATE makes an active empty workspace; the next successful render fills that same TAKE. Restore applies the complete saved Director state atomically.

See [Director Complete Guide](docs/DIRECTOR_GUIDE_EN.md).

## Integrated Refine / Latent Hi-Res

One Long Media Sampler contains the MAIN stage and an optional internal Stage 2 controlled by `refine_enabled`. When Refine is enabled, `windowed_refine` can be turned off to force a single full-latent Stage 2 pass for maximum temporal/lighting continuity.

```text
Stage 1 MAIN x0
    ↓
(optional) learned video-latent upscale
    ↓
Stage 2 Refine using Refine Sigmas
    ↓
final video + exact Stage-1 audio
```

Long/high-resolution Stage 2 can use bounded temporal windows when `windowed_refine` is enabled. Refine is video-authoritative: Stage-1 audio is preserved exactly and is not re-denoised as an audio refiner.

See [Integrated Refine and Latent Hi-Res](docs/TWO_PASS_LATENT_HIRES_REFINER_GUIDE_EN.md).

## Sampler / memory

Recommended starting point:

```text
sampler_mode   = auto
memory_mode    = auto
attention_mode = auto
```

Keep ComfyUI Dynamic VRAM enabled. Current runtime includes repeat-Queue transient CUDA reference cleanup, guarded native INT8 residency, exact Existing/Kitchen fallback paths and safe Latent Hi-Res model offload.

See [Sampler, VRAM and Performance](docs/SAMPLER_OPTIMIZATION_EN.md).
