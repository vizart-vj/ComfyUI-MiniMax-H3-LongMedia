# Fixed Segmentation Prompting Rules

These rules target `workflow_mode=segmented_continuation` in ComfyUI-MiniMax-H3-LongMedia 0.4.40.

## Mental model

Fixed segmentation and MultiClip use the **same LongMedia clip executor**. The difference is timeline math:

- MultiClip: each clip has an individual duration supplied by the Planner.
- Segmentation: LongMedia creates fixed-duration clips automatically from `segment_duration`.

Motion Context, reference handling, audio slicing, lip-sync, handoff and stitching use the shared engine.

## Prompt ownership

Fixed segmentation has one primary Setup prompt. There are no Planner card prompts.

Write the prompt as a **continuous long-video instruction**, not as a description of independent generations.

Good:

```text
A single continuous shot. The woman walks slowly forward through the battlefield.
The camera tracks backward at her pace. She keeps the same direction, expression,
wardrobe and walking rhythm throughout the sequence.
```

Avoid:

```text
First clip: ...
Second clip: ...
Each segment begins...
```

The model should not be told that internal segmentation exists.

## Timestamped events

`segmented_continuation` is not currently a timeline scheduler.

The same continuous prompt is used as semantic guidance for every internal continuation segment. Timestamp ranges such as:

```text
00:00-00:05  She starts walking.
00:05-00:10  She raises both arms.
00:10-00:15  She crouches near the water.
```

are **not automatically split, remapped, or localized** to the corresponding internal segment.

Because every continuation segment still receives the same overall prompt, strongly timestamped action sequences may be interpreted more than once.

For `segmented_continuation`, describe the intended video as one continuous action or scene:

```text
A single continuous shot. The woman walks along the shoreline throughout the
sequence. She occasionally looks toward the camera, smiles, plays with her hair,
and later slows near the water before continuing forward. The camera remains
beside her and maintains the same direction and distance throughout.
```

Use continuity-oriented language such as:

- `continues walking`;
- `keeps moving in the same direction`;
- `maintains the same camera relationship`;
- `without stopping`;
- `throughout the sequence`.

Avoid detailed timestamp schedules when specific actions must occur at exact moments of the final video.

If you need explicit chronological control, different actions at specific times, or separate prompts for different parts of the movie, use `multiclip` instead.
## Choosing `segment_duration`

`segment_duration` means **new visible output timeline per segment**. Overlap is additional hidden continuation context and does not subtract from it.

Recommended starting points:

- 5–6 s: maximum control / constrained VRAM / difficult identity transfer;
- 7–10 s: general long-form default;
- 10–15 s: fewer seams when VRAM and model stability allow it;
- 20–30 s: use only when a single long pass has a clear quality reason and the memory governor can sustain it.

For 16 GB GPUs, shorter fixed segments are normally preferable to forcing a giant single pass, especially with multiple references or video reference editing.

## Overlap

Keep the validated default unless there is a concrete reason to change it. Overlap is hidden temporal context used for handoff and stitch continuity.

Do not write prompt instructions that explicitly refer to the overlap region.

## Continuous actions

Describe sustained actions with continuity verbs:

- `continues walking`;
- `keeps singing`;
- `camera steadily tracks`;
- `without stopping`;
- `maintains the same direction`.

Avoid repeated startup actions such as `begins walking` or `starts singing` unless the action truly starts at that global time.

## Camera continuity

For continuous shots, define a camera trajectory instead of isolated framings:

```text
The camera begins in a wide frontal tracking shot, slowly arcs to her left over
the middle of the sequence, pushes into a close-up, then falls behind her near the end.
```

For intentional edits or explicitly scheduled shot changes, use `multiclip` so each planned section can own its prompt and duration.

## Reference modes

### `hybrid_auto`
Use `image_1` as the opening anchor and subsequent images as configured references. Best when the first frame must be strongly controlled.

### `ref2va_full`
All connected images are normal Picture references. Use when identity/style references should guide the video without a literal opening-frame anchor.

### `video_ref_edit`
`video_1` owns motion/camera/composition; images provide identity/style replacements. Keep prompts focused on **what changes**, while explicitly preserving the source video's timing and choreography.

Example:

```text
Replace only the woman in white from Video 1 with Subject 1 from Picture 1.
Preserve Video 1 camera path, body motion, scene timing, battle choreography and all
other characters. Maintain Subject 1 identity consistently for the full sequence.
```

## Audio and lip-sync

When `audio_mode=lip_sync`:

- connect source performance audio to `audio_1`;
- describe speaking/singing as a continuous performance;
- do not split lyrics or phonemes manually at internal segment boundaries;
- LongMedia uses the original source timeline for per-segment audio context and final output.

## When to use segmentation instead of MultiClip

Choose `segmented_continuation` when:

- the movie is fundamentally one continuous scene or action;
- the same semantic prompt can remain valid throughout the whole video;
- equal internal segment sizes are desirable;
- segmentation is primarily being used to control VRAM or generation stability;
- you do not require exact action timing per segment.

Choose `multiclip` when:

- different parts of the video require different prompts;
- actions must happen in a specific chronological order;
- shot durations must be controlled individually;
- you need explicit per-clip seeds or timing;
- the final video is better described as a sequence of planned shots or actions.

A useful rule of thumb:

```text
One continuous action, split internally for VRAM
→ segmented_continuation

Explicit timeline / different actions / planned shots
→ multiclip
```
## Checklist

- Prompt describes one continuous final scene, not the internal segments.
- The same prompt should remain semantically valid throughout the whole video.
- Avoid detailed timestamp ranges for scheduled actions.
- Use continuity language such as `continues`, `keeps`, `maintains`, `throughout`.
- Do not describe repeated startup actions such as `begins walking` or `starts singing` unless repetition is actually intended.
- `segment_duration` is chosen for quality / VRAM / stability, not storytelling.
- Reference identity wording remains consistent.
- Lip-sync source is connected to `audio_1` when required.
- Use `multiclip` when exact chronological action control is required.
