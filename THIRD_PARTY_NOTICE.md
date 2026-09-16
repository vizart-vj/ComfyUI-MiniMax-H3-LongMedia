# Third-party notices

This package includes an adapted subset of the Sol-Attn implementation from:

- Saganaki22/ComfyUI-sol-attn
- https://github.com/Saganaki22/ComfyUI-sol-attn
- License: Apache License 2.0

Files under `sol_kernel/` are modified/adapted for MiniMax-H3-LongMedia. The current experimental embedded path focuses on the BF16 SM120 pointer implementation used by RTX 50-series GPUs. It is not a verbatim copy of the upstream package and does not require the upstream ComfyUI custom node to be installed.

The upstream repository in turn documents its kernel as based on NVIDIA Sol-Attn reference work under Apache-2.0. See the bundled Apache-2.0 license text.

## Minimax H3 Latent Upscaler
Portions of `latent_hires.py` derive from LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler.
License: Apache-2.0. Upstream: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler

## MiniMaxDirector design attribution

LongMedia Director is inspired by the timeline-director and WHO & WHAT authoring design of `imbutus/ComfyUI-MiniMaxDirector` (MIT, copyright Petr Zenin). The current LongMedia implementation is LongMedia-native rather than a byte-for-byte vendoring of that editor. The upstream MIT text and inspected reference are preserved in `THIRD_PARTY_MINIMAXDIRECTOR_MIT.txt`.

## NVIDIA Sana / Sol-H3

The 0.6.21 Sol-H3 runtime profile and AdaLN trajectory-cache design are adapted from concepts and public implementation details in NVIDIA Research `NVlabs/Sana`, branch `sol-engine`, `models/minimax_h3/Sol-H3/h3_runtime` (Apache License 2.0). LongMedia adapts those ideas to ComfyUI's native MiniMax-H3 model/offload lifecycle and does not vendor NVIDIA checkpoint weights. The existing Comfy Kitchen QK/RoPE and SwiGLU implementations remain authoritative where current Comfy already provides equivalent fused operators.
