# ComfyUI-MiniMax-H3-LongMedia

Production-oriented ComfyUI nodes for **MiniMax H3** long-form video/audio generation, native reference editing, MultiClip planning, camera direction, fixed segmentation, lip-sync/redubbing, latent hi-res refinement, and adaptive low-VRAM execution.

![screenshot](ex.png)

**Current release: 0.5.40**

## Main Nodes

- **MiniMax H3 • Long Media Setup**
- **MiniMax H3 • Long Media Planner**
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

or install the package from the Comfy Registry.

Package identity:

```text
GitHub:             vizart-vj/ComfyUI-MiniMax-H3-LongMedia
Comfy PublisherId:  noise
```

Restart ComfyUI after installation/update.

## Current Setup Model

New workflows use independent semantic controls:

```text
control_mode
h3_mode
timeline_mode
duration_source
audio_mode
```

This is the main change in how the project should be understood compared with the public 0.4.40 documentation.

See [Operating Modes](docs/MODES_GUIDE.md).

### H3 Conditioning

```text
t2va
fl2va
ref2va
hybrid
video_ref_edit
```

### Timeline

```text
single
segmented
multiclip
```

### Duration Ownership

```text
auto
video
audio
manual
longest_input
```

`duration_source` controls timeline length only. It does not remove audio references or change final-audio policy.

## `video_ref_edit`

Typical source-character replacement:

```text
video_1 = source video frames
image_1 = replacement identity
audio_1 = source soundtrack or new dub
```

A Video input is an IMAGE batch and never contains soundtrack data.

For preserve-style modes, Video1+Audio1 can be presented as a native paired source-performance reference while Audio1 also owns the target timing/output waveform.

For `audio_mode=lip_sync`, Audio1 is intentionally independent from Video1's original facial performance so completely new dialogue or singing can drive the replacement character.

Audio2/Audio3 remain prompt-addressable references for music, percussion, bass, ambience, or other semantic timing. In `video_ref_edit`, they are conditioning references only; Audio1 remains the sole preserved/passthrough source soundtrack.

See [Audio Modes and video_ref_edit](docs/AUDIO_MODES_GUIDE.md).

## MultiClip + Cameras

Recommended connection:

```text
Long Media Planner
        ↓ clip_plan
Long Media Cameras
        ↓ clip_plan
Long Media Setup
```

Use:

```text
timeline_mode = multiclip
```

Planner owns diegetic scene/action prompts, durations, names, and optional seeds. Cameras owns framing, rig/lens, movement, speed, spatial relation, entity continuity, and transitions.

See:

- [MultiClip Guide](docs/MULTICLIP_GUIDE.md)
- [Cameras Guide](docs/CAMERAS_GUIDE.md)
- [MultiClip Prompting](docs/MULTICLIP_PROMPTING_GUIDE.md)

## Segmented Long Form

Use:

```text
timeline_mode = segmented
```

for one continuous semantic movie split into fixed-duration internal units for VRAM/stability. It is not a storyboard scheduler.

See [Fixed Segmentation Prompting](docs/PROMPTING_SEGMENTATION.md).

## Two-Stage H3 / Latent Hi-Res

The Long Media Sampler can:

1. generate a low-resolution Stage-1 denoised x0;
2. learned-upscale the **video latent only**;
3. preserve the audio latent;
4. rebuild target-grid conditioning;
5. optionally run an independent same-seed fresh-noise high-resolution H3 pass.

Without Latent Hi-Res, the Refiner remains a continuous zero-noise low-sigma tail.

See [Two-Stage Sampling, Latent Hi-Res and Refiner](docs/TWO_PASS_LATENT_HIRES_REFINER_GUIDE.md).

## Sampler / Memory

Production starting point:

```text
sampler_mode   = auto
memory_mode    = auto
attention_mode = auto
```

Keep ComfyUI Dynamic VRAM enabled.

Current memory-safety work includes exact Comfy Kitchen query streaming for structurally impossible fused-QKV workloads, repeat-run memory isolation, guarded native INT8 VBAR prefetch on constrained GPUs, and RAM-pressure-aware pinned host memory.

See [Sampler, VRAM and Performance Guide](docs/SAMPLER_OPTIMIZATION.md).

## FastH3 / FastVideo VSA

LongMedia includes isolated compatibility paths for supported H3ddle/PulpCut FastH3 VSA and Kijai FastVideo VSA packages.

These paths use strict structural detection and reset their runtime state when switching back to ordinary H3 checkpoints.

## Loop Closure

Loop Closure returns the generated tail toward the opening macro-state in latent/H3 space. It is independent from conditioning/timeline mode and does not use an RGB crossfade as its primary mechanism.

## Documentation

Start at [docs/README.md](docs/README.md).

Important guides:

- [Operating Modes](docs/MODES_GUIDE.md)
- [Audio Modes](docs/AUDIO_MODES_GUIDE.md)
- [Two-Stage Sampling / Latent Hi-Res / Refiner](docs/TWO_PASS_LATENT_HIRES_REFINER_GUIDE.md)
- [Sampler / VRAM](docs/SAMPLER_OPTIMIZATION.md)
- [Architecture](docs/ARCHITECTURE.md)
- [0.5.40 changes since 0.4.40](docs/RELEASE_NOTES_0.5.40.md)

## Example Workflows

- `workflows/MiniMax-H3-LongMedia-SAFE-1080p-15s.json`
- `workflows/MiniMax-H3-LongMedia-LatentUpscale-Detailer.json`

Release workflows contain neutral media placeholders rather than user media or local preview paths.

See [workflows/README.md](workflows/README.md).

## Third-Party Code

This repository contains adapted third-party components under their respective licenses, including:

- Saganaki22/ComfyUI-sol-attn-derived code under Apache-2.0;
- MiniMax H3 latent-upscaler-derived code under Apache-2.0.

See `THIRD_PARTY_NOTICE.md` and `THIRD_PARTY_APACHE_2_0.txt`.

## Release History

0.5.40 is the release consolidation after public v0.4.40.

Historical release notes remain in `docs/` for compatibility/reference. They may use the old `workflow_mode` terminology.

## License

See [LICENSE](LICENSE).
