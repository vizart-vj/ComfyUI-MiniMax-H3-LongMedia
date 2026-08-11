# Changelog

## v0.2.46

### Fixed
- Fixed video-reference setup crash: `Boolean value of Tensor with more than one value is ambiguous`.
- Optional video/audio Tensor inputs are now tested explicitly with `is not None` instead of Python truth-value coercion.
- Covers both `source_video` and `source_audio` assignments in the video-to-video plan path.

## v0.2.45

- Fixed Long Media Setup on current ComfyUI builds where `CLIP.encode()` no longer accepts the legacy `control` keyword.
- Prompt encoding now uses ComfyUI's canonical `encode_from_tokens_scheduled()` API with legacy fallbacks.
- Clarified video/audio reference inputs: `video_N` carries frames only; extracted source audio should be connected separately to the corresponding `audio_N`.

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
