# LongMedia 0.6.69 — cumulative changes since 0.6.63

This release brings the local 0.6.64–0.6.69 development line together for users of the public 0.6.63 release. Existing node identifiers and workflow formats remain supported. Some operations require compatible saved native TAKEs; an MP4 by itself is not a continuation cache.

## Director and TAKEs

- **Append after approved clips:** an enabled new GENERATED suffix no longer needs a historical TAKE. Disabled GENERATED clips still require compatible active TAKEs for replay, and the boundaries needed for a frozen successor are checked. Cache read or initialization failures stop the run rather than silently requesting a full render.
- **CUT range regeneration:** choose the interval between CUT marks, preserve or regenerate its audio, review the actual H3-grid-aligned range, and queue an edit using the active source TAKE.
- **Two-sided Bridge:** insert a GENERATED bridge between compatible GENERATED neighbors and render it with both boundaries locked to their active TAKEs. The result remains pending for preview and explicit **ACCEPT** or **DISCARD**. Acceptance rechecks the source revisions, timeline, global prompt and both AV boundaries. If the bridge duration is off the grid, the menu offers available nearest durations and shifts the following MAIN clips when resizing.
- **TAKE responsiveness:** filesystem work for TAKE HTTP handlers is moved off the event loop, status is read-only, and large tensor writes are staged before publication under the project lock.
- A silent video reference without a soundtrack no longer fails solely because the video loader exposes a lazy audio output.

Range and Bridge require MultiClip and compatible source TAKEs; Loop Closure must be off. The native two-sided Bridge path currently requires GENERATED parents, not MEDIA parents. See [Selective Regeneration and TAKE Workflow](DIRECTOR_REGENERATION_GUIDE_EN.md) for steps and constraints.

## Runtime and compatibility

- Frames-aware ComfyUI H3 VideoVAE keeps its native temporal slicing, tile batching and blending. LongMedia's tile-batching adapter remains available for compatible older H3 VAEs. This addresses the decode signature mismatch reported in [#25](https://github.com/vizart-vj/ComfyUI-MiniMax-H3-LongMedia/issues/25) and [#26](https://github.com/vizart-vj/ComfyUI-MiniMax-H3-LongMedia/issues/26).
- The streamed H3 final output head is scoped to the LongMedia forward and restored afterward, including on an exception. Current per-stream PDD grids are supported. This addresses the shared-model state leak described in [#16](https://github.com/vizart-vj/ComfyUI-MiniMax-H3-LongMedia/issues/16).
- Director `lip_sync` keeps the full Audio 1 waveform, including a silent opening, and selects each clip's guide audio on the global timeline ([#28](https://github.com/vizart-vj/ComfyUI-MiniMax-H3-LongMedia/issues/28)).
- The unified Sampler restores its guider and hooks after errors and cleans up the model/control branches acquired during preparation. A cleanup failure no longer replaces the original render error. Log messages identify preparation, cleanup, TAKE save and Decode stages.
- MultiClip Decode joins video latent views in one allocation where concatenation is required, retaining the native continuous VAE route.

For frames-aware H3 VAEs, Decode's `batch_size` does not control tile batching. See [Decode parameters](PARAMETER_REFERENCE_EN.md#decode) for the legacy and generic VAE behavior.

## Sampler and canvas

- Sampler `View` remains above the inputs and presets when widgets are recreated. Basic and Advanced hide empty sections; workflow saves retain the backend widget order.
- Preset actions are separate **New preset**, **Overwrite preset**, and **Delete preset** buttons. Overwrite and Delete become available after a preset is selected. Native canvas controls avoid the floating overlay reported in [#30](https://github.com/vizart-vj/ComfyUI-MiniMax-H3-LongMedia/issues/30).
- Director panning is batched by animation frame; overview zoom uses a lightweight placeholder, and redundant Director/Setup updates are coalesced.

## Verification and limits

Automated regression tests cover the frames-aware and legacy VAE contracts, temporary output-head restoration, PDD grid math on test tensors, silent-opening Audio 1 routing, Sampler cleanup, and Director Range/Bridge contracts. Frontend tests cover the Sampler preset controls and Director interaction. These tests do not replace the original reporters' workflows or hardware.

Two ordinary Director MultiClip runs completed sequentially in one local ComfyUI 0.46.0 session on an RTX 5080. Completion alone does not verify visual continuity, lip sync, or every workflow. A full before/after stock-H3 GPU comparison for #16, an end-to-end silent-opening lip-sync render for #28, interrupted execution, and comprehensive Range/Bridge, low-VRAM and multi-GPU generation remain unverified for this release.

The cause of a Windows PID remaining listed after the ComfyUI terminal closes remains unconfirmed. Cleanup and diagnostic logging changed, but this release does not claim that the original process-lifetime problem is fully fixed.

## Update

Back up your existing node folder and user data, stop ComfyUI, replace the old `ComfyUI-MiniMax-H3-LongMedia` folder under `custom_nodes` with the release folder, and restart ComfyUI. Do not keep two active copies of the pack. Preserve your workflows, models, and TAKE library. See the [documentation index](README_EN.md) and [changelog](../CHANGELOG.md).
