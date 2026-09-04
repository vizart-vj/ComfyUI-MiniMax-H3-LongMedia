# Long Media MultiClip — Quick Guide

**MultiClip** creates one long sequence from multiple individually controlled clips.

Each Planner clip has its own:

- Name
- Prompt
- Duration
- optional Seed

LongMedia manages temporal continuation and clip handoff between neighboring clips.

## Basic connection

```text
Long Media Planner
        ↓ clip_plan
Long Media Setup
```

With dedicated camera control:

```text
Long Media Planner
        ↓ clip_plan
Long Media Cameras
        ↓ clip_plan
Long Media Setup
```

In Setup select:

```text
Timeline Mode: multiclip
```

A connected Planner becomes authoritative only while `timeline=multiclip`.

## Global Prompt

Use **Global Prompt** for properties that should remain consistent across the whole sequence:

- characters and identity;
- clothing and persistent props;
- persistent environment;
- visual style;
- lighting logic;
- atmosphere;
- other global continuity constraints.

Clip cards then describe the local timeline changes rather than repeating the complete world description.

## Clip cards

Each clip prompt should describe what happens during that part of the timeline.

```text
Clip 1
Duration: 6s
The woman walks slowly through the ruined city.

Clip 2
Duration: 5s
She continues walking and gradually raises her right hand.

Clip 3
Duration: 7s
Her raised hand reaches her face as she looks toward the burning buildings.
```

Durations may be different for every clip.

A practical starting range:

```text
4–8 seconds   — high control
7–12 seconds  — good general-purpose range
12+ seconds   — fewer boundaries, with more opportunity for visual drift
```

## Seed

An empty clip Seed means:

```text
auto
```

LongMedia derives a per-clip seed from the sampler base seed and clip identity/order.

Use a fixed clip seed when repeatable A/B testing is useful.

## Reordering clips

Planner cards can be reordered by dragging them.

The card keeps its:

- prompt;
- duration;
- seed;
- name;
- `clip_id`.

When Cameras is connected with **Auto Sync Planner**, the corresponding camera card follows the same `clip_id`.

## Importing multiple prompts

Planner supports structured prompt import.

```text
clip_1:
The woman enters the hall.

clip_2:
She approaches the central altar.

clip_3:
She slowly raises both hands.
```

The `shot_N:` alias is also supported:

```text
shot_1:
...

shot_2:
...
```

Use **Import Prompt** to convert the structured text into normal editable cards.

**Auto Import Prompt** is useful when the structured text comes from another node or an external LLM.

## Presets

Clip cards support reusable presets, including JSON import/export.

Presets are useful for recurring action structures, transition patterns, or production templates.

## Audio

Audio behavior is controlled by `audio_mode` in Setup. For source-video editing, read [Audio Modes and video_ref_edit](AUDIO_MODES_GUIDE.md).
