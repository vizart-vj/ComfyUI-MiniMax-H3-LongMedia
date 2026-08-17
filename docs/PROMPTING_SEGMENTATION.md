# Fixed Segmentation Prompting Rules

These rules target `workflow_mode=segmented_continuation` in ComfyUI-MiniMax-H3-LongMedia 0.4.0.

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

For planned events in a long prompt, use global timestamps or shot markers that refer to the final movie timeline:

```text
00:00-00:08  Wide frontal tracking shot; she walks toward camera.
00:08-00:16  Camera gradually moves to her left side without interrupting her walk.
00:16-00:24  Push into a close-up while she continues singing.
00:24-00:30  Camera falls behind her as she walks away.
```

LongMedia localizes the prompt against each segment window. Keep timestamps global and monotonic.

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

For intentional edits, make the global cut time explicit.

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

Choose segmentation when:

- the movie is fundamentally one continuous prompt/action;
- equal clip sizes are desirable;
- you are using segmentation primarily to cap per-pass VRAM;
- you do not need separate prompts or durations for each clip.

Choose MultiClip when shot durations and prompts must differ clip by clip.

## Checklist

- Prompt describes one final movie, not internal segments.
- Global timestamps are used for scheduled events.
- `segment_duration` is chosen for quality/VRAM, not storytelling.
- No repeated `begins/starts/establishing` language at invisible boundaries.
- Reference identity wording remains consistent.
- Lip-sync source is connected to `audio_1` when required.
