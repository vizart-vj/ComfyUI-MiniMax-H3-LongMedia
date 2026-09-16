# Long Media Cameras — Quick Guide

**Long Media Cameras** controls camera direction independently from scene content.

## Recommended connection

```text
Long Media Planner
        ↓ clip_plan
Long Media Cameras
        ↓ clip_plan
Long Media Setup
```

In **Long Media Setup**, select:

```text
Timeline Mode: multiclip
```

The Cameras node can also be used standalone, but Planner → Cameras → Setup is the recommended MultiClip workflow.

## Auto Sync Planner

Keep **Auto Sync Planner = ON** when a Planner is connected.

Cameras then:

- creates one camera card for each Planner clip;
- keeps camera cards associated through the stable `clip_id`;
- follows Planner clip reordering;
- preserves camera settings when clips move;
- disables `Transition to Next` automatically on the final clip.

## Camera-card controls

Each camera card controls one clip:

- **Shot Size** — framing and camera distance.
- **Rig / Support** — physical camera support or capture behavior.
- **Camera Body** — camera / sensor character.
- **Lens** — lens family and focal length.
- **Stabilization** — stabilization character.
- **Movement Path** — camera trajectory.
- **Movement Intensity** — movement speed and strength.
- **Transition Type** — relation to the following clip.
- **Space Relation** — whether the next clip remains in the same, adjacent, or different space.
- **Entity Continuity** — how strongly people and scene layout remain spatially consistent.
- **Transition to Next** — enables the camera-transition contract for the next card.

## Transition Type

### Continuous / Same Shot

Use this when the next clip should feel like the same uninterrupted camera shot.

This is the best default for long continuous sequences.

### Threshold Entry

Use when the camera moves through a physical boundary such as a doorway, arch, gate, corridor entrance, tunnel, or similar transition between adjacent spaces.

### Occluded Hidden Cut

Uses visual occlusion as the transition boundary. This is useful when the sequence should feel continuous while allowing a hidden editorial transition.

### Hard Cut

Creates an intentional visible editorial cut. It is normally paired with **Different Space** when the next clip changes location.

## Camera presets

The node includes sequence presets:

- Continuous Push-In
- Reveal Pull-Back
- Ritual Orbit
- Lateral Reveal
- Descent Into Scene
- Slow Cinematic Drift
- Static Tension → Push
- Approach → Threshold → Interior

A preset fills the camera cards for the complete sequence. Every card remains editable afterward.

## Prompt ownership

When **Long Media Cameras** is connected, let Cameras own operator language.

Use Planner prompts mainly for:

- characters;
- actions and body movement;
- environment;
- lighting;
- atmosphere;
- object interaction;
- scene development.

Cameras compiles its own camera instructions into the Planner output and strips conflicting camera directives from Planner text. Keeping scene content and camera direction separate gives the cleanest control.

## Example

Planner:

```text
clip_1:
The woman walks slowly through the empty ceremonial hall.

clip_2:
She continues forward and gradually raises her right hand toward the altar.

clip_3:
Her hand reaches the glowing surface and the surrounding symbols activate.
```

Cameras:

```text
Clip 1: Wide Shot · Track Forward · Slow
Clip 2: Medium Shot · Track Forward · Slow
Clip 3: Medium Close-Up · Push-In · Ultra Slow
Transition: Continuous / Same Shot
Space: Same Space
Entity Continuity: Lock Population / Layout
```
