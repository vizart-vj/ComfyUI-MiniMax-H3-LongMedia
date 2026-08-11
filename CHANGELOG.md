# Changelog

## Project rename

- Public repository/package name: `ComfyUI-MiniMax-H3-LongMedia`.
- Legacy internal node class IDs are intentionally preserved for workflow compatibility.

## 0.2.44

Current SAFE / long-generation baseline.

- Added adaptive Embedded Sol OOM retry.
- OOM retries reduce streamed QKV chunk size before giving up.
- Prevented Sol OOM from falling directly into generic attention/NVFP4 dequantization fallback.
- Successful smaller retry chunks persist for later blocks/steps.

## 0.2.43

- Added streamed final-output layer.
- Avoids creating the full video-target FP32 hidden tensor after transformer block 49.

## 0.2.42

- Added late-block hard VRAM guard.
- Added step-boundary cleanup.

## 0.2.41

- Fused and chunked the full second transformer half:
  `norm2 -> modulation -> MLP -> gate -> residual`.

## 0.2.40

- Fused chunked MLP + gate + residual path.

## 0.2.39

- Added a separate cooldown for emergency inter-block VRAM trims.

## 0.2.38

- Added denoise step-boundary profiling.

## 0.2.37

- Added adaptive inter-block VRAM guard.

## 0.2.34

- Added setup-stage CLIP/Qwen/reference VRAM isolation.

## 0.2.32

- Added compressed streamed K/V storage for long Sol sequences.

## 0.2.30

- Added chunked Sol output projection and early QKV release.

## 0.2.27

- Added embedded/adapted Sol-Attn modes.

## 0.2.24

- Added token-axis MLP chunking.

## 0.2.21

- Reworked frontend dynamic-input handling without mutating ordinary widgets.
