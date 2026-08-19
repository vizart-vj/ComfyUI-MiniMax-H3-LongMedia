# Example workflow

## MiniMax-H3-LongMedia-SAFE-1080p-15s.json

This workflow is a public example of the LongMedia safe long-sequence path.

## 0.4.0 recommendation

For new production workflows, start with the Sampler in:

```text
sampler_mode   = auto
memory_mode    = auto
attention_mode = auto
```

The included workflow may preserve explicit conservative settings from the validation configuration. They are useful as a reproducible manual baseline, but are not required for normal 0.4.0 use.

## Conservative manual reference

```text
mlp_chunk_tokens                     = 24576
attention_mode                       = sol
sol_tau_start                        = 1.3
sol_tau_end                          = 0.8
sol_qkv_chunk_tokens                 = 8192
sol_out_proj_chunk_tokens            = 24576
vram_activation_reserve_mb           = 4096
inter_block_vram_guard_mb            = 2048
inter_block_guard_cooldown_blocks    = 4
inter_block_guard_emergency_mb       = 512
inter_block_guard_emergency_cooldown_blocks = 3
late_block_guard_start               = 40
late_block_guard_target_mb           = 6144
late_block_guard_min_cached_mb       = 512
step_boundary_cleanup_mb             = 2048
```

These are conservative reference values, not universal optimums. See `docs/SAMPLER_OPTIMIZATION.md` for the release tuning rules.

## Replace before running

The workflow contains placeholder or environment-specific model/media selections. Choose your own H3 model, references and audio after loading it.

## Workflow dependencies

### Required

- **ComfyUI-MiniMax-H3-LongMedia**

### Used by the example / optional acceleration

Depending on the exact saved graph, the example can contain nodes from:

- **ComfyUI-KJNodes**
- **ComfyUI-MiniMax-H3-Turbo**
- **WAS Node Suite**
- **ComfyUI-VideoHelperSuite**

These packages are not all required by LongMedia itself. Optional acceleration nodes can be bypassed when installed but not desired.

## Dynamic VRAM

Keep ComfyUI Dynamic VRAM enabled. Do not launch ComfyUI with:

```text
--disable-dynamic-vram
```
