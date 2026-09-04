# LongMedia Operating Modes

LongMedia separates **how H3 is conditioned**, **how the timeline is built**, **who owns the duration**, and **what happens to audio**. These controls are independent on purpose.

## 1. Control Mode

### `auto`
Recommended for normal use. LongMedia derives the legacy internal workflow contract from the semantic controls below.

### `manual`
Exposes advanced/legacy controls for diagnostics, A/B testing, and deliberately custom conditioning.

For production workflows, start with `control_mode=auto`.

---

## 2. H3 Conditioning Mode

### `t2va`
Pure text-to-video/audio generation.

Use when the shot should be generated from the prompt without Picture/Video reference conditioning.

### `fl2va`
Native MiniMax H3 first/last-frame conditioning.

- `image_1` is required as the first frame.
- `image_2` can be used as the last frame.
- Extra Picture/Video/reference-audio inputs are not part of this conditioning family.
- LongMedia keeps this on the native H3 keyframe path.

### `ref2va`
Native MiniMax H3 reference-to-video/audio conditioning.

Connected Picture, Video, and Audio inputs are presented as native references and can be addressed in the prompt as `<Picture N>`, `<Video N>`, and `<Audio N>`.

### `hybrid`
LongMedia opening-keyframe workflow.

- `image_1` is the opening anchor.
- `image_2` can be the final anchor where the active policy permits it.
- Additional references can still guide identity, style, environment, or motion.

This mode exposes `first_frame_mode` for advanced opening-frame behavior.

### `video_ref_edit`
Uses `video_1` as the main source-performance / motion / camera / composition reference while Picture references define replacements or style changes.

Typical character replacement:

```text
video_1  = source performance
image_1  = replacement identity
audio_1  = source soundtrack or new dub
```

`video_1` is an IMAGE frame batch only. It never contains the soundtrack; load/extract audio separately.

---

## 3. Timeline Mode

### `single`
One H3 target timeline.

Best for ordinary T2VA, FL2VA, Ref2VA, Hybrid, and single-source `video_ref_edit`.

### `segmented`
LongMedia divides one continuous movie into fixed-duration internal segments.

Use it when:

- one semantic prompt continues through the whole movie;
- equal segment sizes are useful;
- segmentation is primarily used to bound VRAM or improve long-run stability.

`segment_duration` is the amount of new visible timeline generated per segment. `transition_frames` is hidden continuation context.

### `multiclip`
The **Long Media Planner** owns clip prompts, durations, names, and optional seeds.

Recommended connection when Cameras is used:

```text
Long Media Planner
        ↓ clip_plan
Long Media Cameras
        ↓ clip_plan
Long Media Setup
```

`clip_plan` is authoritative only when `timeline_mode=multiclip`. It is ignored by `single` and `segmented`.

---

## 4. Duration Source

`duration_source` is always visible. It controls **timeline length only**; it does not change the semantic role of audio or remove references from H3 conditioning.

### `auto`
Mode-aware default.

For `video_ref_edit`, `auto` resolves to the `video_1` duration.

### `video`
Use the `video_1` duration explicitly.

### `audio`
Use the `audio_1` duration.

If Audio1 is longer than Video1 in `video_ref_edit`, the generated target can continue after the source video ends. If Audio1 is shorter, the target is shortened to the audio timeline.

### `manual`
Use `manual_duration`.

This is useful when the output should deliberately be shorter or longer than every connected source.

### `longest_input`
Use the longest connected Video/Audio source.

Example:

```text
video_1 = 6 s
audio_1 = 11 s
duration_source = longest_input
→ target = 11 s
```

MultiClip durations remain Planner-owned. Reconstruction workflows remain source-plan-owned.

---

## 5. Audio Mode

### `auto`
Flexible default.

In `video_ref_edit`:

- with `audio_1`: Audio1 is treated as the source soundtrack/performance clock and is preserved at output;
- without `audio_1`: H3 can generate audio.

### `preserve`
Preserve the connected source soundtrack at output.

For `video_ref_edit`, connect `audio_1`; Video1 carries frames only.

### `generate`
Use H3-generated final audio.

### `reference_only`
Use connected audio as H3 semantic/reference conditioning while keeping H3-generated final audio.

### `preserve_reference`
Use connected audio as H3 reference/timing context and preserve the source waveform at output.

### `lip_sync`
Audio1 becomes the authoritative performance clock and final soundtrack.

For `video_ref_edit`, this is also the **redub** mode: Audio1 can be completely different speech or singing from the source Video1. Video1 remains the visual/motion reference while Audio1 drives the new articulation.

Current LongMedia lip-sync requires `image_1 + audio_1`.

See [Audio Modes and video_ref_edit](AUDIO_MODES_GUIDE.md) for the full connection contract.

---

## 6. Multiple Audio References

`audio_1`, `audio_2`, and `audio_3` can have different jobs. In `video_ref_edit`, Audio1 owns the source/final passthrough soundtrack; Audio2/Audio3 remain conditioning references and are not mixed into that preserved track.

Example:

```text
Audio 1 = dialogue / lip-sync
Audio 2 = percussion reference
Audio 3 = bass reference
```

A prompt can address them independently:

```text
<Audio 2> defines the percussion timing.
Street lights pulse with the percussion accents.
Architectural glitch patterns react to the strongest drum hits.
The road illumination moves in smooth waves following the bass rhythm from <Audio 3>.
```

LongMedia does not perform automatic stem separation. If several musical elements are supplied inside one mixed track, H3 interprets them multimodally from that track.

---

## 7. Loop Closure

Loop Closure is orthogonal to the H3 and timeline modes.

Enable it when the tail of the generated movie should return toward the opening macro-state. LongMedia performs the closure in latent/H3 space rather than by RGB crossfading.

Main controls:

- `loop_closure_enabled`
- `loop_closure_frames`
- `loop_closure_strength`

---

## Common Recipes

### Text-to-video/audio

```text
control_mode    = auto
h3_mode         = t2va
timeline_mode   = single
duration_source = manual
audio_mode      = generate
```

### Native first/last frame

```text
control_mode  = auto
h3_mode       = fl2va
timeline_mode = single
image_1       = first frame
image_2       = optional last frame
```

### Reference-driven generation

```text
control_mode  = auto
h3_mode       = ref2va
timeline_mode = single
```

### Character replacement with original soundtrack

```text
control_mode    = auto
h3_mode         = video_ref_edit
timeline_mode   = single
duration_source = auto
audio_mode      = preserve

video_1 = source video frames
image_1 = replacement character
audio_1 = source soundtrack
```

### Character replacement with a new dub and continuation

```text
control_mode    = auto
h3_mode         = video_ref_edit
timeline_mode   = single
duration_source = audio
audio_mode      = lip_sync

video_1 = source performance
image_1 = replacement character
audio_1 = new longer dialogue
```

If Audio1 is longer than Video1, H3 can continue the scene beyond the source clip while following the new audio clock.

### Long continuous movie

```text
control_mode  = auto
h3_mode       = hybrid or ref2va
timeline_mode = segmented
```

### Directed storyboard

```text
control_mode  = auto
h3_mode       = ref2va
timeline_mode = multiclip
Planner → Cameras → Setup
```

## Compatibility

Old workflows may still contain `workflow_mode` values such as `hybrid_auto`, `ref2va_full`, `segmented_continuation`, or `multiclip`. LongMedia keeps those serialized values for backward compatibility and migrates them to the semantic controls above. New workflows should use `control_mode`, `h3_mode`, and `timeline_mode`.
