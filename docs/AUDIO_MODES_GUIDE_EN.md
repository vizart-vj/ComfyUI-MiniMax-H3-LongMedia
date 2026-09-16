# Audio Modes and `video_ref_edit`

This guide explains when an audio input is required and what happens to the final soundtrack.

## Important: `video_1` never contains audio

The `video_1` socket receives an **IMAGE batch containing video frames only**.

Even when those frames were loaded from a movie file that has a soundtrack, that soundtrack is not carried through the `video_1` connection.

If the source video's original soundtrack is needed, extract/load it separately and connect it to `audio_1` (or another audio input where appropriate).

Recommended source-video workflow:

```text
Source movie
├── video frames ──> video_1
└── extracted audio ──> audio_1
```

## `video_ref_edit` audio requirements

`video_ref_edit` always requires `video_1`. Audio depends on `audio_mode`.

## Source-performance synchronization in `video_ref_edit`

When a replacement character is generated from Picture references, restoring the original soundtrack only at final mux is not sufficient to preserve speaking or singing synchronization. LongMedia therefore treats a connected paired source soundtrack as part of the source-performance contract.

For `video_ref_edit`:

```text
audio_mode = auto + audio_1 connected
audio_mode = preserve
audio_mode = preserve_reference
```

`audio_1` is encoded onto the target AV timeline and frozen as the authoritative audio clock while the video stream is regenerated. The replacement subject is generated against the exact source performance timing, and the untouched source waveform is still restored at final output.

This preserves synchronization for:

- mouth articulation;
- speech and singing timing;
- breathing rhythm;
- expression timing;
- head/body performance timing coupled to the source soundtrack.

`reference_only` keeps connected audio as standalone H3 audio references while H3 owns the final soundtrack. `generate` also keeps H3 in control of the final soundtrack. In `video_ref_edit + lip_sync`, `audio_1` is intentionally **not** asserted to be Video1's original soundtrack: it is an independent authoritative dub/timing source, so completely different speech or singing can drive the replacement character.

| `audio_mode` | Is `audio_1` required? | Final soundtrack behavior |
| --- | --- | --- |
| `auto` | **No** | Connected `audio_1` is preserved/restored. If Audio1 is disconnected, LongMedia uses model-generated H3 audio; Audio2/Audio3 can still remain conditioning references. |
| `preserve` | **Yes** for source-audio preservation | Restores the untouched connected source audio. In `video_ref_edit`, Audio1 is also paired natively with Video1 as its soundtrack and locked to the target AV clock. |
| `generate` | No | Uses model-generated H3 audio for the final output. |
| `reference_only` | Only when an audio reference is intended | Connected audio can participate as an H3 reference while the final soundtrack is model-generated. |
| `preserve_reference` | **Yes** | Uses connected audio as an H3 reference and restores the untouched source track at output. |
| `lip_sync` | **Yes: `audio_1`** | `audio_1` is the authoritative timing/content source for native H3 lip-sync and the untouched track is restored at output. Current lip-sync setup also requires `image_1`. |

## `auto`

`auto` is the flexible default.

### With `audio_1` connected

```text
video_1  = source frames
audio_1  = source soundtrack
audio_mode = auto
```

LongMedia preserves the attached audio and restores it as the final soundtrack. In `video_ref_edit`, the connected `audio_1` is also locked into the target AV stream as the authoritative source-performance clock, so replacement facial and mouth motion are generated against the original soundtrack timing.

### Without `audio_1`

```text
video_1  = source frames
audio_1  = disconnected
audio_mode = auto
```

This is valid. LongMedia allows H3 to produce the output audio and decodes the generated audio stream.

Therefore **`audio_1` is optional for `video_ref_edit + auto`**.

Audio2/Audio3 do not implicitly become the passthrough soundtrack when Audio1 is disconnected; they remain prompt-conditioning references.

## `preserve`

Use `preserve` when the original soundtrack must survive unchanged.

```text
video_1  = source frames
audio_1  = extracted original soundtrack
audio_mode = preserve
```

The source audio is restored at output rather than being replaced by the sampled H3 audio stream. In `video_ref_edit`, `preserve` also makes that source track the authoritative target-audio timing stream; it is therefore used to preserve mouth/facial performance synchronization while the replacement identity is generated.

A preserve-style mode needs an actual connected source soundtrack. If it is missing, LongMedia cannot restore audio that never entered the workflow.

Therefore **connect `audio_1` for `video_ref_edit + preserve`**.

## `preserve_reference`

Use this when the source audio should influence H3 conditioning and the exact source waveform should also be restored at output. In `video_ref_edit`, it also activates the same authoritative source-performance timing lock used by `auto + audio_1` and `preserve`.

```text
video_1  = source frames
audio_1  = source soundtrack / reference
audio_mode = preserve_reference
```

This mode requires connected source audio for its intended contract.

## `generate`

Use this when a new soundtrack should be produced by H3.

```text
video_1  = source frames
audio_mode = generate
```

An input soundtrack is not required for the final-audio contract.

## `reference_only`

Use this when audio is supplied as a conditioning reference while H3 still owns the final generated soundtrack.

```text
audio_1 = reference audio
audio_mode = reference_only
```

The audio connection is meaningful when you actually want an audio reference. The final output remains generated rather than source-audio passthrough.

## `lip_sync`

For native LongMedia H3 lip-sync:

```text
image_1 = visual subject / opening image
audio_1 = authoritative speech or singing performance
audio_mode = lip_sync
```

`audio_1` remains native H3 audio conditioning, drives the per-clip H3 Audio Guide timing, and is restored untouched at final output.

For current LongMedia lip-sync, both `image_1` and `audio_1` are required. `audio_2` and `audio_3` may also be connected as additional prompt-addressable H3 references; they do not replace Audio1 as the lip-sync clock or final passthrough track.

## Quick decision table

If the source movie has important original audio:

```text
Want automatic behavior?           -> auto + connect audio_1
Want guaranteed untouched audio?   -> preserve + connect audio_1
Want audio as ref + untouched out? -> preserve_reference + connect audio_1
Want lip-sync to that track?       -> lip_sync + connect image_1 + audio_1
Want a new H3 soundtrack?          -> generate
Want audio only as H3 reference?   -> reference_only + connect audio_1
```

If the source movie has no useful soundtrack:

```text
auto     -> audio_1 may stay disconnected; H3 generates audio
generate -> audio_1 may stay disconnected; H3 generates audio
```

## `video_ref_edit`: Paired Source AV Performance

For character replacement/editing, `video_1` and its original soundtrack should be treated as one source performance.

With `audio_mode = auto`, `preserve`, or `preserve_reference` and `audio_1` connected, LongMedia now uses two complementary timing mechanisms:

1. `video_1 + audio_1` are sent to MiniMax H3 as one native paired `video_audio` reference block. This preserves the relationship between the source facial/body performance and its soundtrack.
2. `audio_1` is also written into the target audio stream and frozen as the generation clock. The untouched source waveform is restored at final output.

For `video_ref_edit`, `duration_source = auto` follows the `video_1` timeline. This prevents a slightly shorter encoded audio/container duration from cutting off the final visual performance.

The intended setup is:

```text
h3_mode:    video_ref_edit
video_1:    source performance frames
image_1:    replacement character / identity reference
audio_1:    soundtrack extracted from video_1
audio_mode: preserve   (or auto / preserve_reference)
```

`video_1` remains an IMAGE batch and never carries audio by itself. Connect the extracted soundtrack separately to `audio_1`.


## `duration_source`: timeline ownership is independent

`duration_source` controls **only the target timeline length**. It does not decide whether connected audio is available to H3, and it does not replace `audio_mode`.

In `video_ref_edit`:

| `duration_source` | Target duration | Typical use |
| --- | --- | --- |
| `auto` | `video_1` duration | Safe source-edit default. |
| `video` | `video_1` duration | Explicitly keep the source-video horizon. |
| `audio` | `audio_1` duration | Redub to a shorter/longer track; longer audio can continue the scene past Video1. |
| `manual` | `manual_duration` | Force any target duration. |
| `longest_input` | Longest connected video/audio input | Automatically follow the longest supplied source/reference. |

Examples:

```text
video_1 = 6 s
audio_1 = 11 s
h3_mode = video_ref_edit
audio_mode = lip_sync
duration_source = audio
```

The target is ~11 seconds. Video1 establishes the first ~6 seconds of scene/camera/performance reference; H3 then continues the scene while Audio1 remains the authoritative redub clock.

```text
video_1 = 10 s
audio_1 = 6 s
duration_source = audio
```

The target uses the opening ~6 seconds of Video1 and completes on the audio-owned horizon.

```text
video_1 = 6 s
manual_duration = 8 s
duration_source = manual
```

The target is ~8 seconds regardless of input-media lengths.

Changing `duration_source` never removes `<Audio N>` references from prompt conditioning.

Passthrough audio is fitted to the selected timeline at final output: a shorter target crops the waveform at the target boundary, while a longer target preserves the complete source waveform and appends silence after it. Samples inside the retained source range are not resampled or re-timed.

## Arbitrary redub in `video_ref_edit`

For replacement speech or singing that is **different from the source video's original performance**:

```text
h3_mode = video_ref_edit
video_1 = source scene / movement / camera
image_1 = replacement character
audio_1 = NEW speech or singing
audio_mode = lip_sync
duration_source = video | audio | manual | longest_input
```

Audio1 is treated as a standalone authoritative target-performance clock. It is not falsely declared to be the original soundtrack paired with Video1.

Use `duration_source=audio` when the new performance should own the output length. Use `video` when the edit must stay within the source-video duration.

## Multiple audio references

`video_ref_edit` can use more than one connected audio input. Native prompt tags remain available as `<Audio 1>`, `<Audio 2>`, and `<Audio 3>` according to connected reference order.

The roles can be different:

```text
Audio 1 = original soundtrack or authoritative dub
Audio 2 = percussion / rhythm reference
Audio 3 = bass / music / ambience reference
```

`audio_mode` controls final-audio/timing semantics; it does not erase Audio2/Audio3 from prompt conditioning. In `video_ref_edit`, **Audio1 is the only source/final passthrough soundtrack authority** for `auto`, `preserve`, `preserve_reference`, and `lip_sync`. Audio2/Audio3 are additional semantic/music references and are not mixed into the preserved output track.

### Audio-reactive prompting

You can address musical structure directly in the prompt. For a single mixed soundtrack:

```text
<Video 1> defines the original street, motion and camera path.
<Picture 1> defines the replacement character.
<Audio 1> defines the musical rhythm and temporal structure.

Street lights illuminate rhythmically with the percussion transients from <Audio 1>.
Architectural glitch patterns pulse on the strongest drum hits.
The road surface, reflections and flowing light respond to the low-frequency bass rhythm.
Each bass accent sends a smooth wave of illumination forward along the street.
```

With separate references/stems:

```text
<Audio 2> defines the percussion timing.
<Audio 3> defines the bass rhythm.
Street lights pulse with <Audio 2>.
Glitch patterns travel across the buildings on the strongest <Audio 2> hits.
The road and reflections move in smooth low-frequency waves following <Audio 3>.
```

These are H3 semantic AV-conditioning instructions. LongMedia does not perform DSP stem separation itself; when drums and bass are inside one mixed soundtrack, describe the desired components from that `<Audio N>` explicitly in the prompt.

## Contract summary

Think of the controls as three independent dimensions:

```text
duration_source -> who owns target length
audio_mode      -> what audio is authoritative/preserved/generated
prompt          -> how each <Audio N> affects the visuals and performance
```

This separation allows source character replacement, exact source-performance preservation, arbitrary redubbing, source-video continuation, trimming, and audio-reactive visual edits without introducing separate workflow modes.
