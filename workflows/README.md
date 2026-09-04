# Example Workflows

## MiniMax-H3-LongMedia-SAFE-1080p-15s.json

Conservative long-sequence example.

## MiniMax-H3-LongMedia-LatentUpscale-Detailer.json

Sanitized production graph showing the current LongMedia Setup/Sampler/Decode topology, learned latent-hires controls, refiner controls, and an optional explicit second Long Media Sampler.

The explicit second Sampler is **bypassed by default**. It is an advanced chained stage and is not required for the integrated `latent_hires_enabled + refine_enabled` two-stage path.

See:

- `docs/MODES_GUIDE.md`
- `docs/TWO_PASS_LATENT_HIRES_REFINER_GUIDE.md`
- `docs/SAMPLER_OPTIMIZATION.md`

## Recommended Runtime Defaults

For new production workflows:

```text
sampler_mode   = auto
memory_mode    = auto
attention_mode = auto
```

Keep ComfyUI Dynamic VRAM enabled.

## Media Placeholders

Release workflows do not ship user media.

Loader selections are replaced with neutral `REPLACE_ME_*` placeholders and saved video-preview/local-output metadata is removed. Select your own image/video/audio files after loading the workflow.

## Dependencies

Required:

- **ComfyUI-MiniMax-H3-LongMedia**

The example graphs can also contain optional nodes from packages such as:

- ComfyUI-KJNodes
- ComfyUI-VideoHelperSuite
- WAS Node Suite
- other utility packs used by the saved graph

These packages are workflow dependencies, not LongMedia runtime dependencies.
