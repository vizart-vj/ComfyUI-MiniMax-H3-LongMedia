# MultiClip Prompting Guide

The core rule is:

**Global Prompt describes what remains constant.**

**Clip Prompt describes what changes over time.**

## Global Prompt

Put persistent scene information in the Global Prompt.

```text
A pale woman in a dark ceremonial robe stands inside an enormous ancient
sci-fi temple. Cold metallic architecture, monumental scale, dim amber ritual
light, realistic materials, cinematic dark atmosphere.
```

This is the shared world state for the sequence.

## Clip prompts

Each next clip should continue from the state created by the previous clip.

```text
clip_1:
The woman slowly walks toward the central altar. Her robe moves naturally with
each step. The surrounding crowd remains still and attentive.

clip_2:
She continues the same walk and gradually raises her right hand. The ritual
lights begin pulsing softly across the walls.

clip_3:
Her raised hand reaches the altar surface. The symbols surrounding it gradually
activate and fill the chamber with warm light.
```

## Useful continuity language

Useful temporal phrases include:

- `continues`
- `keeps moving`
- `gradually`
- `slowly`
- `the movement develops`
- `the same action continues`
- `reaches`
- `begins`
- `moves closer`
- `transitions into`

They give the next clip a clear motion/state to inherit.

## Split actions across clip boundaries

For smooth handoff, let an action cross the boundary naturally.

```text
clip_2:
She gradually raises her right hand toward the glowing surface.

clip_3:
Her right hand continues the same movement and gently touches the glowing surface.
```

The second prompt starts from an already established direction of motion.

## Prompting with Long Media Cameras

When **Long Media Cameras** is connected, keep camera direction in Cameras.

Use clip prompts for:

- character actions;
- body movement;
- environment;
- lighting changes;
- atmosphere;
- object interaction;
- scene progression.

Example:

```text
clip_1:
The man walks slowly through the crowded street.

clip_2:
He continues forward and turns his head toward the neon signs.

clip_3:
He reaches the entrance, stops beside it and looks inside.
```

Shot size, lens, movement path, stabilization and transition behavior are then supplied by Cameras.

## Prompting without Cameras

When Cameras is not connected, operator direction can be included directly in each clip prompt.

```text
clip_1:
The woman walks through the hall. A wide frontal tracking shot moves slowly backward.

clip_2:
She continues walking as the camera smoothly arcs toward her left side.

clip_3:
The continuous camera movement gradually approaches a medium close framing.
```

## MiniMax-H3 wording style

Prefer positive descriptions of the visual state you want to preserve.

```text
The architecture remains stable and preserves its original proportions.
The same characters remain in their established positions.
The movement continues smoothly at the same slow pace.
```

Positive state descriptions are especially useful for continuity, identity, architecture and slow camera motion.

## Audio and lip-sync wording

When `audio_mode=lip_sync`, connect the authoritative source performance to `audio_1` and describe the visible performance semantically.

```text
She continues singing with clear natural mouth articulation synchronized to
Audio 1. Her body movement remains slow and cinematic.
```

LongMedia owns the timing from the connected audio source; prompts should describe the intended visible performance rather than manually spelling phonetic timing.

## Structured MultiClip prompt

For Planner import:

```text
clip_1:
...

clip_2:
...

clip_3:
...
```

The number of `clip_N` sections should match the intended video structure. Duration and Seed remain editable inside the Planner cards.
