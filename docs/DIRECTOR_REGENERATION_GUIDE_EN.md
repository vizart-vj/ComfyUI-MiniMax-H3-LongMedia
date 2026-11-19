# LongMedia Director — Selective Regeneration and TAKE Workflow

This guide describes the Director regeneration model in LongMedia 0.6.69.

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

### Regenerate a CUT range

Range regeneration edits an interval inside one GENERATED MAIN clip. It requires the Director timeline in MultiClip mode, an active compatible TAKE for that clip, and Loop Closure disabled. The saved TAKE must still match the clip duration and geometry.

1. Add at least one CUT marker inside the clip. CUT markers mark an interval; they do not split the clip as Knife does.
2. Open the clip's context menu in the interval between two markers, or between a marker and the clip edge. Choose **Regenerate … · keep source audio** or **Regenerate … · regenerate audio**.
3. Check the confirmation: the requested range is expanded to the native H3 video grid and the actual range is displayed. Confirm to queue the run.

The active TAKE supplies the source latent; the edit preserves the locked latent outside the selected range. Keeping audio preserves the source audio latent for the clip. If the TAKE, grid or required boundaries are incompatible, the operation stops instead of silently turning into a full render.

### Two-sided Bridge

Bridge creates a generated transition with cached GENERATED clips immediately on both sides. It requires MultiClip, Loop Closure disabled, compatible active TAKEs for both neighbors and the other GENERATED clips that must be replayed, and touching MAIN boundaries. MEDIA clips cannot be used as the two native bridge parents in this path.

1. Select a boundary between two touching clips and choose **Insert bridge at selected / nearest boundary** from the MAIN context menu. A new clip is inserted and later MAIN clips move right; when Ripple is enabled, later camera, audio and layer blocks also move. Split or move a block that crosses the insertion boundary first.
2. Open the bridge clip's context menu and choose **Render two-sided bridge from neighboring Takes**. If its duration is off the H3 grid, the menu offers available shorter/longer aligned durations; resizing also moves the following MAIN clips.
3. Review the resulting pending bridge TAKE in the Program Monitor. **ACCEPT** rechecks the frozen source revisions, timeline, global prompt and both AV boundaries before activation. **DISCARD** removes the candidate without activating it.

Range and Bridge have regression coverage for their contracts; a complete end-to-end GPU reproduction of every range/bridge scenario has not been established for this release.

## Partial RUN: frozen TAKE + new suffix (0.6.64)

Disable completed GENERATED clips for RUN and leave new targets enabled. Previous TAKEs are needed only for disabled replay and right locks before frozen GENERATED successors. New adjacent targets and appended suffixes do not need historical TAKEs. A disabled single-shot prefix remains frozen across the switch to MultiClip even with a randomized seed; its duration/geometry must match the TAKE.

With all clips enabled, ordinary RUN remains semantic incremental: changed seed/prompt/refs can invalidate the prefix. Keep the seed fixed for automatic append. Cache read/initialization errors stop the run instead of silently initiating full diffusion.

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
