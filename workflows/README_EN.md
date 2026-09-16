# Example Workflows

The release contains exactly three maintained workflows.

## MiniMax-H3-LongMedia-Director-Unified.json

Primary **LongMedia Director** workflow. Director owns the MAIN / CAMERA / AUDIO timeline, WHO & WHAT references, FIRST/LAST/REF roles, resolution policy, Program Monitor and TAKE workspace.

```text
MiniMax H3 • LongMedia Director
        ↓ director
MiniMax H3 • Long Media Setup
                 ↓
Long Media Sampler
                 ↓
Long Media Decode
```

See `docs/DIRECTOR_GUIDE_EN.md` and `docs/DIRECTOR_REGENERATION_GUIDE_EN.md`.

## MiniMax-H3-LongMedia-LatentUpscale-Detailer.json

Production graph for learned Latent Hi-Res and Refine. The integrated Refine path is the internal Stage 2 of the same Long Media Sampler.

When Refine is enabled, **windowed refine** controls its memory/continuity policy:

- ON — LongMedia may use bounded temporal windows for long/high-resolution Stage 2.
- OFF — Stage 2 refines the complete latent in one monolithic pass. This maximizes temporal/lighting continuity but can require substantially more VRAM.

See `docs/TWO_PASS_LATENT_HIRES_REFINER_GUIDE_EN.md`.

## MiniMax-H3-LongMedia-SAFE-1080p-15s.json

Conservative 1080p / 15-second production example.

## Recommended runtime defaults

```text
sampler_mode   = auto
memory_mode    = auto
attention_mode = auto
```

Keep ComfyUI Dynamic VRAM enabled.

## Media placeholders

Release workflows do not ship user media. Loader selections use neutral `REPLACE_ME_*` placeholders and saved local preview/output metadata is removed. Select your own image/video/audio files after loading a workflow.

## Dependencies

Required:

- **ComfyUI-MiniMax-H3-LongMedia**

A saved example can also contain optional utility nodes from packages such as ComfyUI-KJNodes, ComfyUI-VideoHelperSuite or WAS Node Suite. Those are workflow dependencies, not LongMedia runtime dependencies.
