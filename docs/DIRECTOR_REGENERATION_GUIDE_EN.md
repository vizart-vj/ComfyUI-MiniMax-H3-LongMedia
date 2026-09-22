# LongMedia Director — Selective Regeneration and TAKE Workflow

This guide describes the current 0.6.54 Director regeneration model.

## TAKE is the saved revision

Director storage is TAKE-centric. A TAKE stores the complete authored Director state plus compatible clip-level cache/continuation data. `CREATE TAKE` creates the active empty revision; the next successful render fills that same TAKE. Restore replaces the Director document atomically, including MAIN, prompts, WHO & WHAT media, FIRST/LAST/REF roles, CAMERA/AUDIO/EMBEDDING tracks, resolution policy and Director modes.

## BASE types participate differently

```text
GENERATED = H3 render unit + TAKE/cache + native continuation state
MEDIA     = immutable external video; never an H3 render unit
```

MEDIA can be a continuation parent for the immediately following GENERATED clip, but it is never sampled itself.

## Selective MultiClip rules

### Regenerate Clip

For a normal generated chain:

```text
Clip1 -> Clip2 -> Clip3
Regenerate Clip3

Clip1 = TAKE HIT
Clip2 = TAKE HIT
Clip3 = SAMPLE
```

When both incoming and outgoing seam contracts are valid, clip-only regeneration samples exactly one target. If a safe two-sided lock is impossible, the backend expands the dependency range rather than pretending an incompatible cached suffix is valid.

For a MEDIA chain:

```text
Video MEDIA -> Clip1 -> Clip2
Regenerate Clip2

Video = untouched
Clip1 = TAKE HIT
Clip2 = SAMPLE
```

Regenerating Clip1 leaves Video untouched and samples Clip1 only. Clip2 becomes stale but is not automatically sampled.

### Regenerate From Here

Keeps the validated prefix and samples the selected GENERATED clip plus all dependent GENERATED clips after it. MEDIA blocks stay immutable and are never added to `sampled_indices`.

## Appending a new clip

Appending after a cached GENERATED parent uses the saved native TAKE tail/motion/audio continuation and samples only the new clip. Appending after MEDIA builds a fresh child target, encodes only the external overlap tail, and attaches that tail as a native frame-0 H3 keyframe.

The runtime invariant is:

```text
one new GENERATED clip = one H3 sampling target
```

## External MEDIA cache

External continuation cache keys include source identity, editorial trim range, geometry and overlap. Cached tail latents are retained on CPU. Reusing the same unchanged MEDIA boundary therefore skips VideoVAE/audio-tail re-encoding.

The visible MEDIA range is authoritative: a trimmed or reused source continues from the end of the BASE block, not from the physical end of the source file.

## Final assembly

- GENERATED runs are decoded through H3 VideoVAE.
- MEDIA RGB/audio is inserted directly into the final timeline.
- The hidden overlap used for MEDIA continuation is removed from the visible child output.
- Original MEDIA audio is not regenerated.
- Mixed mono/stereo pieces are normalized to `[1,C,L]`; mono is duplicated exactly when stereo output is required.

## Refine / Latent Hi-Res

Selective regeneration does not change the Refine contract. Stage 2 is video-only; Stage-1 audio remains exact passthrough.

See also:

- [Director Complete Guide](DIRECTOR_GUIDE_EN.md)
- [Architecture](ARCHITECTURE_EN.md)
- [Sampler, VRAM and Performance](SAMPLER_OPTIMIZATION_EN.md)
