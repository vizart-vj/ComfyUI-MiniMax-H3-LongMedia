# ComfyUI-MiniMax-H3-LongMedia 0.6.40

0.6.40 is the consolidated production release of the 0.6 Director line.

## Director

- Director is operational for `t2va`, `fl2va`, `ref2va`, `hybrid`, and native `video_ref_edit` routing.
- MAIN timing, CAMERA/AUDIO/embedding tracks, Ripple editing, MultiClip markers, segmented controls, resolution/aspect policy and Program Monitor are integrated into one authoring surface.
- FIRST, LAST and REF remain separate semantic roles. Same-source FIRST+LAST is supported as a loop/endpoint anchor.
- TAKE storage is TAKE-centric. CREATE makes an active empty TAKE; the next successful render fills that same TAKE. Restore applies the complete saved Director state atomically.
- TAKE gallery width, timeline zoom, inspector height and editable text areas persist with the workflow.

## Refine and Latent Hi-Res

- One Long Media Sampler contains MAIN Stage 1 and an optional internal Stage 2 enabled by `refine_enabled`.
- Stage 2 uses the dedicated Refine Sigmas input.
- Long/high-resolution refine can use bounded temporal windows with overlap-save core ownership to avoid duplicate seam writes.
- A new **windowed refine** switch appears when Refine is enabled. ON keeps the memory-bounded adaptive path; OFF forces one full-latent Stage 2 pass and never silently falls back to windows after OOM.
- Refine is video-authoritative: Stage-1 audio is preserved exactly and is never re-denoised by Stage 2.

## Segmented and chained execution

- Segmented continuation uses the exact previous latent tail as the frozen target prefix and deterministic per-segment seed offsets, eliminating the repeated short-motion cycle caused by duplicated motion context/noise phase.
- Chained Sampler motion-context layout supports mixed visual and audio conditioning segments.

## Memory and runtime

- Repeat-Queue cleanup severs LongMedia-owned transient CUDA references that could survive through cached guider state.
- FastH3/VSA execution geometry caches are execution-scoped.
- Latent Hi-Res model residency is protected by `try/finally`, so the cached upscaler returns to CPU after success, error, OOM or cancellation.
- Existing ComfyUI Dynamic VRAM residency policy remains intact; the release does not add an unconditional post-sampling model unload.

## VAE and compatibility

- MiniMax H3 VAE decode can batch native spatial tiles with safe OOM fallback.
- Legacy internal `MiniMaxH3LatentLab...` class identifiers remain registered for saved workflow compatibility.

## Documentation

The release archive contains only current bilingual user documentation (`*_EN.md` and `*_RU.md`). Historical per-build release notes and development audit scripts are intentionally excluded from the user package; development history remains in `CHANGELOG.md`.
