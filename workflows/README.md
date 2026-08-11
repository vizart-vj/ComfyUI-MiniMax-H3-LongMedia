# Example workflow

## MiniMax-H3-LongMedia-SAFE-1080p-15s.json

Clean public version of the workflow used to validate the SAFE long-sequence path.

### Validated LongMedia settings

- Duration: 15 seconds
- Output target: 1920×1088 via the 2.0 MP / 16:9 sizing path
- Sampler: Euler
- Steps: 8
- `mlp_chunk_tokens`: 24576
- `attention_mode`: `sol`
- `sol_tau_start`: 1.3
- `sol_tau_end`: 0.8
- `sol_qkv_chunk_tokens`: 8192
- `sol_out_proj_chunk_tokens`: 24576
- `vram_activation_reserve_mb`: 4096
- `inter_block_vram_guard_mb`: 2048
- `inter_block_guard_cooldown_blocks`: 4
- `inter_block_guard_emergency_mb`: 512
- `inter_block_guard_emergency_cooldown_blocks`: 3
- `late_block_guard_start`: 40
- `late_block_guard_target_mb`: 6144
- `late_block_guard_min_cached_mb`: 512
- `step_boundary_cleanup_mb`: 2048

### Replace before running

The workflow intentionally contains placeholder media names:

- `REPLACE_ME_reference_*.png`
- `REPLACE_ME_audio.wav`

Select your own reference images/audio after loading the workflow. Model widgets keep the tested filenames; if your models live in subfolders, select them again in the corresponding loaders.

## Workflow dependencies

### Required for this example

- **ComfyUI-MiniMax-H3-LongMedia** — Long Media Setup / Sampler / Decode
- **ComfyUI-KJNodes** — Set/Get helpers plus the recommended MiniMax H3 Sage Attention patches used in the model chain
- **WAS Node Suite** — `Text Multiline` prompt node
- **ComfyUI-VideoHelperSuite** — ProRes video output

### Recommended acceleration

- **ComfyUI-MiniMax-H3-Turbo** — the `MiniMaxH3TurboLoRA` node and Turbo LoRA path used in the validated workflow
- **ComfyUI-KJNodes MiniMax H3 Sage patches** — the `PathchSageAttentionKJ` and `MiniMaxH3MemoryEfficientSageAttentionPatch` nodes

The Turbo and KJ Sage nodes are **recommended, not mandatory for LongMedia itself**. If you have them installed but do not want to use them, select the node(s) and set them to **Bypass**. The model chain will pass through to the remaining LongMedia/Sol path.

> Note: if a custom-node package is not installed at all, ComfyUI may show a missing-node warning when opening this exact workflow. Install the package first, then bypass the node if you do not want its effect.

### Dynamic VRAM

The validated SAFE run used ComfyUI Dynamic VRAM. Do not start ComfyUI with `--disable-dynamic-vram` for this preset.
