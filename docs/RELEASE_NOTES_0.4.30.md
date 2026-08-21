# 0.4.30

Release-ready MultiClip prompt workflow and extension-compatibility update.

## MultiClip Planner prompt workflow

`MiniMax H3 • Long Media Planner` owns the new prompt authoring/import controls:

- **Global Prompt** — shared context, style and instructions applied to every clip.
- **Multiple Clips Prompt** — structured import source using `clip_N:` or `shot_N:` sections.
- **Import Prompt** — explicit manual import. It never enables or changes Auto Import.
- **Auto Import Prompt** — imports a valid structured source when that source changes. Repeated refreshes of the same source do not overwrite edits.

Example:

```text
clip_1:
The man stands still and looks into camera.

clip_2:
He starts walking toward camera.

clip_3:
He turns sharply to the side.
```

Import copies local text into ordinary editable clip-card prompt fields. There is no live binding after import. Existing card duration and seed values are preserved; newly created cards inherit the last existing duration and use automatic seed.

If **Multiple Clips Prompt** is connected to an ordinary frontend-readable STRING/Text source, manual Import updates the cards immediately. If the connected source only resolves at backend execution (for example an LLM/API node), manual Import queues an independent one-shot request for the next workflow execution without modifying Auto Import.

At sampling time the effective per-clip text is:

```text
Global Prompt + local Clip Prompt
```

The two remain separate and independently editable in the Planner UI. Long Media Setup only consumes the resulting `clip_plan`; it does not own these prompt controls.

## Sampling extension compatibility

0.4.30 preserves the stock ComfyUI `CFGGuider.sample()` extension contract while retaining LongMedia's unified H3 lifecycle. Nested AV packing, `latent_shapes`, callback adaptation, model-option/hook handling and `OUTER_SAMPLE` wrappers stay on the stock ComfyUI path. This restores compatibility with KJ Model Preview Override without repeating H3 prepare/pre-run/cleanup for every clip.

## MultiClip UI and runtime

- Long Media Setup keeps the user-selected node width when switching to MultiClip.
- MultiClip cards remain responsive and editable.
- Native continuous H3 video-latent assembly and one continuous VideoVAE decode are retained.
- Existing lip-sync, refiner, modulation-row and low-VRAM runtime behavior is retained.

## Packaging

The plugin folder name remains exactly `ComfyUI-MiniMax-H3-LongMedia`.
