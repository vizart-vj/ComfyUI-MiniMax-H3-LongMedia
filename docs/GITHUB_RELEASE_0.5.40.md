# GitHub Release — v0.5.40

## Title

**v0.5.40 — Semantic Modes, Native AV Editing & Memory-Safe Two-Pass H3**

## Description

LongMedia 0.5.40 is the release consolidation of the development line since v0.4.40.

The public Setup model is now split into independent semantic controls for H3 conditioning, timeline construction, duration ownership, and audio behavior. This makes `video_ref_edit`, redubbing, MultiClip, segmentation, reference conditioning, and duration extension composable instead of being hidden behind one legacy workflow selector.

### Highlights

- **Semantic Setup:** `control_mode`, `h3_mode`, `timeline_mode`, and always-visible `duration_source`, with legacy workflow migration.
- **Planner + Cameras:** stable clip identity, drag/reorder, presets, per-clip cinematography, and continuity-aware transitions.
- **Native `video_ref_edit`:** paired source AV preservation, arbitrary new-dub lip-sync, independent duration ownership, multiple Audio references, and audio-reactive prompting.
- **Two-stage H3:** denoised-x0 learned latent upscale plus optional independent high-resolution same-seed H3 refinement.
- **Reconstruction + Loop Closure:** native Ref2VA source editing, detail recovery, and latent macro-state loop return.
- **FastH3 / FastVideo VSA compatibility:** strict structural validation, runtime isolation, and portable learned-VSA execution.
- **16 GB / Windows reliability:** Triton TinyCC bootstrap, exact Comfy Kitchen query streaming for impossible full QKV, preview isolation, repeat-run memory boundaries, bounded pinned RAM, and guarded native INT8 prefetch.
- **Documentation refresh:** current modes, audio contracts, two-pass sampling, sampler/VRAM guidance, and sanitized example workflows.

### Upgrade Notes

Existing workflows remain supported through legacy node class IDs and `workflow_mode` migration. New workflows should use the semantic Setup controls documented in `docs/MODES_GUIDE.md`.

Keep ComfyUI Dynamic VRAM enabled for oversized H3 checkpoints.

### Package Identity

- **GitHub:** `vizart-vj/ComfyUI-MiniMax-H3-LongMedia`
- **Comfy Registry PublisherId:** `noise`

See `docs/RELEASE_NOTES_0.5.40.md` for the full changes since v0.4.40.
