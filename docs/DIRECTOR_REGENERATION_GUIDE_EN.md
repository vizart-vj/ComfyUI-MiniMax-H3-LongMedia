# LongMedia Director — Selective Regeneration and TAKE Workflow

This guide describes the current 0.6.50 Director regeneration model.

## TAKE is the top-level saved revision

Director TAKE storage is TAKE-centric. User-visible revisions live under the LongMedia Director library rather than being owned by individual clip-cache folders.

Conceptually:

```text
longmedia_director/
├── Unsorted/
│   └── <take>/
│       ├── take.json
│       ├── state/
│       │   ├── director.json
│       │   └── global_prompt.txt
│       ├── clips/
│       │   └── <clip_id>/latent.safetensors
│       └── previews/
└── _runtime/
```

User-created project folders sit beside `Unsorted`. `_runtime` contains service/runtime pointers and is not a TAKE collection.

## CREATE TAKE

CREATE makes a complete empty TAKE workspace and immediately makes it the active editor revision. The next successful render fills that same TAKE; it does not create a second sibling revision just because the render completed.

## Restore

Selecting/Restoring a TAKE replaces the Director document from the stored snapshot rather than restoring only a preview branch. The restored state includes timeline media, WHO & WHAT media, prompts, GLOBAL PROMPT, FIRST/LAST/REF roles, camera/audio/extra tracks, resolution policy and Director modes.

Browser media caches are invalidated so the UI reflects the restored revision instead of stale preview state.

## Duplicate, rename and folders

- Duplicate creates a new TAKE from the complete source snapshot.
- Rename changes both the visible TAKE name and its directory name.
- TAKE cards can be moved between user-created project folders.
- `Unsorted` is the default collection when no project folder is selected.

## Selective MultiClip regeneration

For MultiClip, LongMedia can reuse approved cached clip states and regenerate only the invalidated range. Dependency checks remain conservative: if an incoming/outgoing continuation is incompatible, the invalidated range expands rather than pretending an unsafe cached suffix is reusable.

Continuation and display states can differ when Latent Hi-Res is active, so the runtime keeps the state needed for both visible output and downstream clip handoff.

## Single and segmented timelines

Single/segmented renders finalize the active TAKE from the final stitched AV. This is separate from MultiClip selective-cache ownership: the user-visible TAKE is still the top-level revision, while runtime clip pointers remain implementation detail under `_runtime`.

## Refine / Latent Hi-Res

Selective regeneration does not change the Refine contract. Stage 2 is a video refiner; Stage-1 audio is preserved exactly. Large high-resolution Stage 2 work can use bounded temporal windows.

See also:

- [Director Complete Guide](DIRECTOR_GUIDE_EN.md)
- [Integrated Refine and Latent Hi-Res](TWO_PASS_LATENT_HIRES_REFINER_GUIDE_EN.md)
- [Operating Modes](MODES_GUIDE_EN.md)
