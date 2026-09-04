# MultiClip Prompting Rules

For the current quick-start guide, see [MultiClip Prompting Guide](MULTICLIP_PROMPTING_GUIDE.md).

## Ownership

Use MultiClip when:

```text
timeline_mode = multiclip
```

The Planner owns the clip timeline. A connected `clip_plan` is ignored by `single` and `segmented` timelines.

Recommended camera workflow:

```text
Planner → Cameras → Setup
```

When Cameras is connected, keep Planner prompts focused on diegetic content: subjects, actions, environment, lighting, atmosphere, continuity, and scene changes. Cameras owns framing, lens, rig, movement path, speed, and transition behavior.

## Global Prompt vs Clip Prompt

**Global Prompt** describes what should remain coherent across the sequence.

**Clip Prompt** describes what changes during one clip.

Global Prompt example:

```text
A monumental nocturnal city with wet black stone streets, deep blue atmosphere,
consistent architecture, realistic materials, and stable identity for the main subject.
```

Clip prompts:

```text
clip_1:
The subject walks toward the central plaza while distant signs slowly illuminate.

clip_2:
The same walk continues. The plaza lighting grows brighter and reflections spread across the wet road.

clip_3:
The subject reaches the center while the surrounding architecture enters its fully illuminated state.
```

## Continue Actions Across Boundaries

When an action crosses a clip boundary, continue it explicitly:

```text
clip_1:
She gradually raises her right hand toward the luminous surface.

clip_2:
Her right hand continues the same movement and gently touches the luminous surface.
```

This gives H3 a temporal direction to carry through the continuation context.

## Positive State Language

Describe the desired state directly:

```text
The architecture preserves its proportions.
The same subjects remain in their established positions.
The movement continues smoothly at the same pace.
```

Positive state descriptions are preferable to long negative constraint lists.

## Structured Import

Planner accepts contiguous sections beginning at 1:

```text
clip_1:
...

clip_2:
...

clip_3:
...
```

or:

```text
shot_1:
...

shot_2:
...
```

Import owns prompt text. Existing per-card duration/seed settings remain card-owned.

## Audio-Reactive MultiClip

Connected audio references can be addressed semantically:

```text
<Audio 2> defines the percussion timing.
The street lights brighten on the percussion accents.
The building surfaces develop short glitch pulses on the strongest hits.
```

Timeline duration, final-audio policy, and audio conditioning remain separate controls.

## Camera Continuity

For a continuous shot across clips, use Cameras with `Continuous / Same Shot` and preserve the same-space/entity-continuity contract.

For intentional transitions, choose the transition in Cameras rather than describing competing camera edits inside each Planner prompt.
