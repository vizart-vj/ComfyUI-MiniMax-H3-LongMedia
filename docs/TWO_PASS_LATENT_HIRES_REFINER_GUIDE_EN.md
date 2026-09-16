# Integrated Refine and Latent Hi-Res

`Refine` is an independent post-x0 stage inside the same **MiniMax H3 • Long Media Sampler**. The MAIN sampler is not shortened or replaced.

## Sampler inputs

```text
sigmas          -> MAIN generation schedule
refine_sigmas   -> REFINE schedule (separate SIGMAS socket)

latent_hires_enabled
latent_hires_model
latent_hires_scale
latent_hires_precision
latent_hires_align

refine_enabled
windowed_refine
```

Legacy `refine_steps`, `refine_add_noise`, and `refine_seed` fields remain serialized for old workflows but are hidden/ignored. Refine step count is `len(refine_sigmas) - 1`.

## Execution contract

```text
MAIN sigmas
    ↓
complete MAIN x0
    ↓
(optional) learned H3 latent upscale — video only
    ↓
post-x0 Stage 2 using refine_sigmas
    ↓
exact Stage-1 audio restore
```

Enabling Refine never changes the MAIN sigma schedule. MultiClip/segmented continuation remains owned by the normal MAIN planner.

## Internal Sampler #2

Conceptually, one Long Media Sampler node contains two sequential stages:

```text
Stage 1 = MAIN sampler
Stage 2 = Refine sampler
```

`refine_enabled` behaves like bypass/unbypass of Stage 2. Stage 2 receives the Stage-1 result and uses only `refine_sigmas`.

Short/sufficiently small latents use a monolithic Stage 2. With **windowed refine = ON**, long/high-resolution latents may use bounded temporal windows to stay within VRAM. With **windowed refine = OFF**, Stage 2 is exact-only and always refines the complete latent in one pass.

`windowed_refine` is shown only while Refine is enabled. It defaults to ON for the memory-safe release behavior. Turning it OFF is the continuity-first option for shots where neighboring refine windows cause lighting, framing or image-position jumps. Exact-only mode intentionally does not fall back to windows after OOM.

## Temporal-windowed Refine

When **windowed refine = ON**, long/high-resolution x0 is not forced through one giant H3 pass. Video refine uses legal windows aligned to H3's native `(1,4,4,4,4)` temporal pattern.

On a 16 GB-class GPU, the initial safe window is approximately:

```text
chunk ≈ 73 decoded frames = 22 H3 temporal tokens
```

The planner preserves native temporal phase. If a window still OOMs, LongMedia retries with progressively smaller legal H3 geometry rather than changing MAIN generation or attention math.

## Window continuity

Windowing is a Refine implementation detail. It is **not** LongMedia segmentation and does not create Director clips.

Each window receives neighboring context and writes only its owned refined core. The current approach avoids crossfading two independent absolute latent geometries while also preserving the full useful H3 refine result instead of reducing it to a detail-only residual.

Each window retains the source segment's global temporal position.

## Director / embeddings

Camera and embedding ranges are projected from the global segment clock into each local refine window. Text-row ownership is unchanged, so a refine-window boundary does not restart a Director camera or embedding range.

## Audio / lip-sync

Refine is **video-authoritative**:

```text
video: refine with refine_sigmas
audio: exact Stage-1 passthrough
```

Stage 2 is not an audio refiner:

- audio noise = 0;
- audio denoise mask = 0;
- Stage-2 audio output is not accepted as a replacement soundtrack;
- final audio latent is restored exactly from Stage 1.

With Latent Hi-Res, callback/x0 is used for the video path only and does not replace authoritative Stage-1 audio.

For lip-sync, the forced target audio remains the same authoritative clock. `audio_strength=1.0` is the expected MiniMax-H3 lip-sync setting.

## Latent Hi-Res

With `latent_hires_enabled=true`, only the complete denoised 24-channel video latent passes through the learned H3 latent upscaler. Stage 2 then refines the new H/W.

```text
low-res clean video x0
    ↓
learned latent upscaler
    ↓
high-res video x0
    ↓
H3 Refine (windowed or full-latent, controlled by windowed refine)
```

Audio never passes through the latent upscaler.

The cached upscaler is returned to CPU through `try/finally`, including error/OOM/cancel paths, to avoid repeat-Queue VRAM retention.

## Workflow topology

Only one Long Media Sampler is required:

```text
Long Media Setup
       ↓
Long Media Sampler
   MAIN Sigmas ────────┐
   Refine Sigmas ──────┤
                       ↓
                 final AV latent
                       ↓
                 Long Media Decode
```

The old chained second-Long-Media-Sampler handoff remains only for backward compatibility and specialized reconstruction workflows. The recommended Latent Hi-Res/Refine topology is the internal Stage 2 of one Sampler node.
