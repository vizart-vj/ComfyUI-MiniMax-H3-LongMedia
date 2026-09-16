# LongMedia Director 0.6.40 — Complete Guide

LongMedia Director is the timeline-oriented authoring surface for **MiniMax H3 • LongMedia**. It combines shot timing, semantic references, FIRST/LAST frame anchors, cameras, temporal embeddings, audio policy, preview/review and TAKE management in one Director document.

## Recommended wiring

```text
MiniMax H3 • LongMedia Director.director
                    ↓
MiniMax H3 • Long Media Setup.director
```

Set `control_mode = director` in Setup. Director then owns authoring/timeline state while Setup and Sampler keep runtime, memory, model and decode controls.

## What Director owns

Director owns:

- MAIN shot/clip timeline and authored duration;
- global and per-shot prompts;
- semantic Picture / Video / Audio references;
- FIRST and LAST frame roles;
- camera tracks and temporal H3 embedding tracks;
- Director audio policy;
- Director resolution/aspect/megapixel policy;
- TAKE snapshots, review state and selective regeneration metadata.

Setup still owns runtime-facing options that are not explicitly overridden by Director, including model/runtime compatibility, reference budget, loop closure, motion repair and downstream sampling/memory behavior.

Explicit Setup media sockets remain valid overrides. For example, `Setup.image_1` replaces Director `<Picture 1>` while the rest of the Director document remains active.

## Timeline modes

**One MAIN shot stays one H3 sampler pass.** Temporal CAMERA/EMBEDDING controls do not secretly split it into additional generation passes.

The top `TIMELINE` selector controls the Director execution model.

### `single`

One MAIN shot is one H3 generation pass. This is the normal choice for one continuous shot or a single source-video edit.

### `multiclip`

MAIN boundaries become clip boundaries. Each clip has its own prompt/seed/timing while LongMedia carries native continuation context across adjacent clips.

Director shows draggable clip markers. `+ CLIP` splits at the playhead and `− CLIP` removes/merges the selected or nearest boundary.

### `segmented`

One semantic scene is split into internal fixed-duration execution segments for VRAM/stability. `SEG` / `SEG LEN` control the segmentation policy. Segmentation is not a storyboard cut and should preserve one continuous scene.

## MAIN shots, Ripple and snapping

MAIN clips can be selected, moved, reordered, split and resized. The **Magnet** control enables snapping to relevant edges/playhead positions.

`Ripple` controls whether timeline-bound CAMERA/AUDIO/other authored layers follow MAIN edits:

- Ripple ON — dependent layers move with the MAIN edit;
- Ripple OFF — those layers keep their absolute authored time.

## H3 semantic routing

Director separates **semantic role** from the physical media slot.

### REF

A REF item is a native semantic reference. It remains a reference and is not silently converted into FIRST or LAST.

Prompt-addressable reference tags use the native H3 form:

```text
<Picture 1>
<Video 1>
<Audio 1>
```

### FIRST

FIRST is the native opening-frame anchor.

### LAST

LAST is the native closing-frame anchor.

FIRST and LAST are globally exclusive roles. The same source may be assigned to both FIRST and LAST to create a same-source loop anchor. LongMedia reuses one consistent geometry transform for that same-source case so the two endpoints do not acquire different crop geometry.

## Automatic conditioning family

Director derives the safe H3 family from semantic roles:

```text
text only                    -> t2va
REF media                    -> ref2va
FIRST and/or LAST only       -> fl2va
FIRST/LAST + REF media       -> hybrid
```

A real REF added to an FL2VA-authored shot promotes the runtime family to Hybrid rather than forcing incompatible media into the pure FL2VA family.

### Source-video editing

Ordinary Video media placed as REF remains native Ref2VA reference conditioning. Source-performance / replacement editing uses the dedicated native character/source-edit path, which builds the `video_ref_edit` contract around the source video and optional source/dub audio. See [Audio Modes and `video_ref_edit`](AUDIO_MODES_GUIDE_EN.md).

## Resolution controls

Director can own output geometry independently of Setup `width` / `height`.

Top controls:

- `RES SRC` — Base layer, 16:9, 9:16, 1:1, 4:3, 3:4, 21:9, 2.39:1 or Custom;
- `RES` — explicit W×H aspect source when Custom is selected;
- `MP` — target megapixel budget.

`Base layer` derives aspect from the first MAIN visual item and falls back to 16:9 when no visual source exists. Final dimensions are snapped to H3-safe multiples.

When `control_mode=director`, Setup width/height widgets remain visible/connectable for workflow compatibility but Director geometry is authoritative.

## Prompt editors

Director has a global prompt plus per-MAIN prompt editors. Prompt editor heights are user-resizable and persist with the node UI state.

For temporal camera/embedding controls, the common scene/subject/reference prompt is encoded once for the complete MAIN shot. Partial CAMERA and EMBEDDING layers are factorized controls rather than repeated copies of the entire scene prompt.

## Cameras

CAMERA blocks are non-diegetic cinematography controls. They should describe framing/movement rather than duplicate scene content.

Important timing behavior:

- blocks use authored half-open intervals;
- gaps mean no camera instruction;
- when camera blocks overlap, the latest start wins; ties use the later document entry;
- adjacent camera blocks use a short centered handoff instead of an implicit hard shot cut;
- one full-shot camera can remain on the native/global conditioning path.

H3 camera control is learned language conditioning, not explicit calibrated 3D camera extrinsics.

See [Long Media Cameras](CAMERAS_GUIDE_EN.md).

## H3 embedding tracks

Use `✦` to add an EMBEDDING track and select an installed H3 embedding. Move/trim the block to select its active time range; `↔` expands it across the full timeline.

Overlapping embedding layers combine. Disabled/muted layers are inactive. Camera and embedding boundaries are independent: an embedding transition does not restart camera presentation and a camera boundary does not clone the complete scene prompt.

## Audio selector

The top `AUDIO` selector exposes the normal LongMedia audio contract:

- `auto` — automatic LongMedia policy;
- `lip_sync` — Audio 1 is the authoritative speech/singing timing and final restored waveform;
- `generate` — H3 owns final generated audio;
- `reference_only` — input audio is H3 conditioning, final audio is generated;
- `preserve_reference` — input audio is conditioning/timing reference and Audio 1 is restored at output;
- `preserve` — preserve Audio 1 at output without treating it as an ordinary H3 reference.

For native lip-sync, use `audio_strength = 1.0`. Audio 1 is the authoritative target clock; additional audio references can still be used for music, rhythm or ambience where the selected mode permits it.

See [Audio Modes](AUDIO_MODES_GUIDE_EN.md).

## Program Monitor

Decoded TAKE previews appear in the Director Program Monitor. The monitor supports play/pause, timeline seeking, clip-boundary navigation, looping and a draggable playhead.

Preview quality controls `25 / 50 / 75 / FULL` change browser-side preview rendering only; they do not create multiple encoded media files.

## TAKE system

A TAKE is a complete Director editor-state snapshot, not just a thumbnail.

A TAKE snapshot includes the state needed to return to the authored version:

- MAIN timeline;
- global/per-shot prompts;
- WHO & WHAT media;
- FIRST/LAST/REF assignments;
- CAMERA / AUDIO / EMBEDDING and extra tracks;
- H3/timeline/audio/resolution controls;
- seed and render metadata;
- cached latent/preview data when available.

### CREATE TAKE

`CREATE TAKE` creates a new empty TAKE and immediately makes it the active editor workspace. The next successful render fills **that same TAKE** instead of silently creating a second one.

### Restore

Clicking/restoring a TAKE replaces the Director document atomically with that TAKE snapshot. Restoring an empty TAKE therefore intentionally returns the editor to its empty state.

### Duplicate / Delete / Rename

TAKEs can be duplicated, deleted and renamed. Rename updates both the UI label and the corresponding TAKE folder when possible.

### Project folders

The TAKE library starts in `Unsorted`. User-created folders can organize TAKEs and TAKE cards can be moved between them.

Current storage is TAKE-centric:

```text
longmedia_director/
├─ Unsorted/
│  └─ Take_Name__<id>/
│     ├─ take.json
│     ├─ state/
│     │  ├─ director.json
│     │  └─ global_prompt.txt
│     ├─ clips/
│     │  └─ <clip_id>/latent.safetensors
│     └─ previews/
│        ├─ poster.png
│        ├─ sprite.jpg
│        └─ audio.wav
├─ <user project folders>/
└─ _runtime/
```

The exact set of generated files depends on whether the TAKE is empty, rendered, cached and decoded.

## TAKE gallery layout

The TAKE panel is horizontally resizable. Its scrollbar is placed on the left so the right edge remains a clean resize handle. Resizing the TAKE panel also expands/contracts the Director node itself. Cards use an adaptive multi-column grid.

Timeline zoom, TAKE width, inspector height, prompt editor sizes and TAKE visibility are serialized in Director UI state and survive normal rerenders/workflow saves.

## Selective regeneration

Director can preserve approved work and rerender only the required dependency range.

Key actions include:

- **Regenerate Clip** — rerender the selected clip when both seam constraints can be preserved safely;
- **Reroll Seed** — change the selected clip seed and use the selective path;
- **Regenerate From Here** — keep the validated prefix and regenerate the selected clip plus dependent suffix;
- **Recast / source replacement tools** — rebuild the required native-reference branch while preserving unaffected timeline state where the dependency contract allows it.

If exact seam locking is unsafe, LongMedia expands the invalidated region instead of pretending that an incompatible cached suffix is safe.

See [Director Selective Regeneration](DIRECTOR_REGENERATION_GUIDE_EN.md).

## Refine / Latent Hi-Res interaction

Director timing remains global even when the Sampler performs windowed post-x0 Refine. Camera and embedding ranges are projected into each refine window using the original segment clock; refine windows do not become new Director clips.

The current Refine path is video-authoritative. Stage-1 audio is preserved exactly while the video latent may be upscaled/refined. See [Integrated Refine and Latent Hi-Res](TWO_PASS_LATENT_HIRES_REFINER_GUIDE_EN.md).

## Recommended workflow

1. Connect Director to Setup and select `control_mode=director`.
2. Choose `TIMELINE` and desired resolution policy.
3. Add MAIN media and prompts.
4. Assign semantic roles deliberately: REF, FIRST, LAST.
5. Add CAMERA / EMBEDDING / AUDIO layers only where needed.
6. Choose the Director `AUDIO` policy.
7. Create a TAKE before an alternate experiment when you want a clean branch.
8. Render through the normal LongMedia Sampler/Decode path.
9. Review in Program Monitor and use selective regeneration rather than rerendering approved regions unnecessarily.

## Troubleshooting

### A reference behaves like a frame anchor
Check its semantic role. REF, FIRST and LAST are separate contracts.

### FIRST/LAST loop slowly reframes
Use the same media source for both roles. Current same-source handling reuses consistent endpoint geometry.

### Restore returns an empty editor
Confirm whether the selected TAKE is an intentionally empty TAKE or a rendered TAKE. A rendered TAKE should contain its finalized snapshot/state files.

### TAKE panel will not widen
Drag the right edge of the TAKE pane. The scrollbar is intentionally on the left; the node width follows the resize operation.

### Director resolution disagrees with Setup width/height
In `control_mode=director`, Director resolution is authoritative by design.

### Camera changes look like cuts
Avoid using scene-change language inside CAMERA blocks. Scene/action continuity belongs in MAIN prompts; CAMERA controls framing and motion.

## Related documentation

- [Operating Modes](MODES_GUIDE_EN.md)
- [Audio Modes and video_ref_edit](AUDIO_MODES_GUIDE_EN.md)
- [Director Selective Regeneration](DIRECTOR_REGENERATION_GUIDE_EN.md)
- [Cameras Guide](CAMERAS_GUIDE_EN.md)
- [Integrated Refine and Latent Hi-Res](TWO_PASS_LATENT_HIRES_REFINER_GUIDE_EN.md)
- [Sampler / VRAM / Performance](SAMPLER_OPTIMIZATION_EN.md)
