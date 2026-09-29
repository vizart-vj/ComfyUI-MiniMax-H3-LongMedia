# ComfyUI-MiniMax-H3-LongMedia 0.6.61

This release fixes two RefMod failures found in long Director renders and in the RefMod ENCODE flow. The bilingual parameter references and Director documentation describe the 0.6.61 package.

## Fixes

- **Long-sequence RefMod attention:** interval-routed RefMods now use Comfy Kitchen's streamed-query INT8 path when the dense fused Q/K/V allocation is unavailable. K/V remain globally quantized, Q is replayed in 128-row-aligned tiles, and per-query visibility, strength and character masks remain active. The MAIN shot stays a single sampler pass.
- **RefMod ENCODE sources:** ENCODE now uses original media persisted by CREATE FROM SOURCES. Imported `.safetensors` artifacts without saved raw sources remain ready to use and no longer offer an invalid VAE re-encode against the artifact path.

## Notes

- Start with `sampler_mode=auto`, `memory_mode=auto` and `attention_mode=auto` for normal runs.
- No live ComfyUI/GPU render was available during release validation. Python and JavaScript syntax, package contents and the installed Comfy Kitchen masked INT8 API contract were checked.

## Documentation

- [Complete Parameter Reference — EN](PARAMETER_REFERENCE_EN.md)
- [Полный справочник параметров — RU](PARAMETER_REFERENCE_RU.md)
- [Director Guide](DIRECTOR_GUIDE_EN.md)
