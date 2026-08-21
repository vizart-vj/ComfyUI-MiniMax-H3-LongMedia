# MultiClip Prompting Rules

These rules target the `workflow_mode=multiclip` path in ComfyUI-MiniMax-H3-LongMedia 0.4.2.

## Mental model

MultiClip is one long sequence built from multiple planned clips. Every clip has its own local prompt, duration and optional seed, while the LongMedia engine supplies temporal handoff, Motion Context, reference conditioning and audio slicing between clips.

The Planner controls **where clips begin and end**. It does not replace the Setup workflow mode: a connected Planner is authoritative only when `workflow_mode=multiclip`.

## Prompt hierarchy

1. Put constraints that must survive the entire movie in the **main Setup prompt**.
2. Put actions, shot changes and clip-specific events in the **Planner card prompt** for that clip.
3. A blank clip prompt inherits the main prompt.
4. Do not repeat the full global prompt in every card unless a property is genuinely at risk of drifting.

## Global prompt: what belongs there

Keep these stable across all clips:

- subject identity and immutable physical traits;
- wardrobe and persistent props;
- environment that should remain continuous;
- overall visual language, lens family, texture and lighting logic;
- persistent behavioral constraints;
- audio role definitions such as who is speaking or singing.

Example:

```text
<Subject 1>: pale blonde woman, cream robe, amber eyes.
A single continuous cinematic sequence. Preserve Subject 1 identity, wardrobe,
scale and facial structure across all clips. Gritty desaturated battlefield,
handheld but controlled camera, realistic motion, natural temporal continuity.
```

## Per-clip prompt: what belongs there

Each Planner card should describe only the **new visible action** for that interval:

```text
Clip 1: She walks toward camera through the battlefield. Medium-wide tracking shot.
Clip 2: Camera moves to her left side. She keeps walking and begins singing.
Clip 3: Tight close-up while she sings; warriors remain blurred in the background.
Clip 4: Camera pulls back behind her as she walks away.
```

Use direct temporal language: `continues`, `keeps`, `without stopping`, `same direction`, `camera remains on the same side` when continuity matters.

## Avoid continuity resets

Do not start every card with language such as:

```text
A new scene...
The video begins...
Establishing shot...
A woman appears...
```

unless you actually want a reset/cut. Those phrases encourage the model to reinterpret the beginning of each clip.

Prefer:

```text
Continue the same shot...
She keeps walking...
Without changing identity or wardrobe...
The camera continues its rightward track...
```

## Shot changes

A deliberate cut is allowed. State it explicitly at the beginning of the target card:

```text
Hard cut to a low-angle close shot of her feet, preserving the same subject,
wardrobe, battlefield and time of day.
```

For a continuous camera move, do not use `cut`, `new shot`, or `establishing`.

## Identity stability

- Use `image_1..image_9` as stable Picture references rather than restating face details differently in every card.
- Keep the subject label and wording identical across cards.
- Do not introduce conflicting age, hair, wardrobe, body or face descriptions later.
- If a clip changes pose or camera angle, describe the **pose/camera**, not a new appearance.

## Motion continuity

For motion that crosses a boundary, explicitly carry the action through both neighboring cards:

```text
Clip N: ...she raises her right hand toward her face.
Clip N+1: She continues the same right-hand movement and touches her cheek...
```

Do not describe the completed state in clip N and restart the action in clip N+1.

## Audio and lip-sync

When `audio_mode=lip_sync`:

- connect the authoritative source to `audio_1`;
- the engine slices the original source audio against the global clip timeline;
- describe the performance semantically (`she sings`, `he speaks`) but do not invent phonetic timing in the prompt;
- avoid asking a different character to speak in a card unless the audio actually changes speaker;
- for singing, state that visible mouth articulation follows the vocal performance and music.

Example:

```text
She continues singing with clear natural mouth articulation synchronized to the
female vocal in Audio 1. Her body motion remains restrained and cinematic.
```

## Durations

Use clip durations based on narrative needs, not memory management. MultiClip may use unequal durations by design.

Practical guidance:

- 4–8 s: strong control, frequent editorial changes;
- 7–12 s: good default for continuous dramatic actions;
- longer clips: fewer boundaries but more VRAM and greater drift risk.

The model's H3 temporal geometry is aligned internally by LongMedia; use human-readable durations in the Planner.

## Seeds

- `null`: LongMedia derives a per-clip seed from the sampler base seed and clip index.
- explicit seed: use only when you intentionally want repeatable per-card exploration.
- Do not change seeds as a substitute for fixing a bad continuity prompt.

## Recommended structure

```text
MAIN SETUP PROMPT
  Subject definitions
  Persistent world/style
  Persistent identity/wardrobe
  Global camera/realism constraints

PLANNER CLIP 1
  Opening action and framing

PLANNER CLIP 2
  Continuation action / deliberate cut

PLANNER CLIP 3
  Continuation action / deliberate cut
```

## Checklist

- Global properties live in Setup.
- Local actions live in Planner cards.
- No accidental `new scene` language at a continuation boundary.
- Same subject labels and reference numbering in every clip.
- Cross-boundary motion is described as continuation.
- Audio performance wording matches the actual `audio_1` content.
- Planner is used only with `workflow_mode=multiclip`.

## 0.4.30 structured prompt import

MultiClip Setup can import clip-local prompts from a separate **Multiple Clips Prompt** field while keeping **Global Prompt** independent.

Supported section aliases:

```text
clip_1:
...
clip_2:
...
```

or:

```text
shot_1:
...
shot_2:
...
```

Sections must be contiguous from 1 and at least two sections must be present. `clip_N` and `shot_N` are aliases; durations and seeds are intentionally not encoded in the prompt syntax.

**Import Prompt** copies section text into the visible clip cards. The copied text is fully editable afterward. **Auto Import Prompt** is a one-shot convenience for connected/dynamic prompt sources and switches itself off after a successful import.
