# ComfyUI-MiniMax-H3-LongMedia 0.6.60

**The package version remains 0.6.60.** This release packages the latest Director line and runtime fixes together and aligns the documentation with the parameters in the actual build.

## Highlights

- **Complete English and Russian parameter references.** Setup, Director, RefMods, Planner/Cameras, Video Reconstructor, Sampler and Decode inputs and controls are documented, including visibility conditions, hidden service fields and legacy fields that no longer affect execution.
- **RefMod Sources builder.** Add up to 128 images, videos and audio sources, paste an image from the clipboard, and inspect estimated visual tokens per source and for the whole set. Short-edge resolution, visual token cap, grid, clip frames, voice seconds and Compressed refinement reach their corresponding VAE stages.
- **RefMod is not a substitute for style transfer.** Concept Type adds broad positive prompt guidance; Description remains library metadata. Strength controls the native reference over its timeline interval. These fields do not change latent meaning and cannot guarantee style transfer by themselves.
- **RefMod attention path.** A full-span native RefMod at Strength=1 keeps the selected H3 attention backend, including zero-copy SLA, when the merged mask is an identity operation. The selected ComfyUI SLA override is no longer lost when guider options are cloned.
- **Character RefMod in MultiClip.** Identity authority is retained across clip boundaries on target-video queries, without changing audio or other RefMod concept routing.
- **Latent Hi-Res + Refine at seams.** Each following clip receives the preceding clip's refined overlap and freezes that context during Stage 2; Stage-1 continuation and exact audio passthrough remain intact.
- **Director timeline.** Gap controls appear only in actual gaps and show their duration; Fit includes the whole timeline after any horizontal scroll. Cut markers belong to their clip; Cut-to-cursor is distinct from Knife split; trim handles expose source time. TAKE sorting options and gallery scroll preservation are included.
- **Preview-memory lifecycle.** Director releases detached filmstrip decoders on refresh, unloads old TAKE thumbnails, loads gallery previews lazily and bounds its decoded-image cache with an LRU. This covers Director preview/media caching; Sampler controls generation VRAM.
- **motion_repair persistence.** The selected value survives workflow save and reload.
- **LongMedia Sampler presets.** Create, overwrite and delete presets from the top of the node. Presets travel with the workflow, and the status line marks settings that differ from the selected preset.
- **MEDIA-only rendering.** BASE timelines made entirely of immutable MEDIA clips bypass H3 model preparation and go directly through media assembly.

## Parameter notes

- For normal runs, keep sampler_mode=auto, memory_mode=auto and attention_mode=auto.
- Press ENCODE after configuring a RefMod. VAE artifact preparation does not run the DiT or Sampler.
- RefMod Refinement steps apply only to COMPRESSED mode. Sampler Refine steps come from the separate Refine Sigmas input; the old refine_steps field is hidden and retained only for workflow compatibility.
- A Director RefMod is active only within its timeline clip interval. Stretch it across the full timeline for a scene-wide reference.
- motion_repair is an optional post-pass: off keeps the previous path; auto is a practical starting point for problematic fast motion.

## Documentation

- [Complete Parameter Reference — EN](PARAMETER_REFERENCE_EN.md)
- [Полный справочник параметров — RU](PARAMETER_REFERENCE_RU.md)
- [Director Complete Guide](DIRECTOR_GUIDE_EN.md)
- [Sampler, VRAM and Performance](SAMPLER_OPTIMIZATION_EN.md)

The archive contains package VERSION 0.6.60; these notes document the current build without changing its version number.
