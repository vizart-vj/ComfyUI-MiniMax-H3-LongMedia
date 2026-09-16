"""Lossless SM120 elementwise fusion used by LongMedia's Sol-H3 profile.

Current Comfy H3 already fuses QK RMSNorm+partial RoPE and quantization-aware
SwiGLU.  The missing useful fusion for the LongMedia low-VRAM block schedule is
RMSNorm + AdaLN scale/shift.  We intentionally do not fuse the preceding gated
residual with norm2: doing so would keep the full attention output alive through
the streamed MLP stage, increasing peak VRAM on 16-GB cards.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..triton_windows_compat import install_windows_triton_build_compat

_TRITON_WINDOWS_BUILD_COMPAT = install_windows_triton_build_compat()


@triton.jit
def _rmsnorm_modulate_kernel(
    out_ptr, x_ptr, weight_ptr, shift_ptr, scale_ptr, row_ids_ptr,
    n_cols, x_stride, table_stride, row_offset,
    eps: tl.constexpr, PER_ROW: tl.constexpr, SCALAR_ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / n_cols
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normed = x * tl.math.rsqrt(variance + eps) * weight
    if PER_ROW:
        mod_row = tl.load(row_ids_ptr + row + row_offset).to(tl.int64)
    else:
        mod_row = SCALAR_ROW
    table = mod_row * table_stride + cols
    shift = tl.load(shift_ptr + table, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + table, mask=mask, other=0.0).to(tl.float32)
    out = normed * (1.0 + scale) + shift
    tl.store(out_ptr + row * x_stride + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


def _warps(block: int) -> int:
    return 16 if block >= 8192 else (8 if block >= 2048 else 4)


def _next_pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def _validate(x: torch.Tensor, weight: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> None:
    if x.device.type != 'cuda':
        raise ValueError('Sol-H3 fused RMSNorm requires CUDA')
    if torch.cuda.get_device_capability(x.device) != (12, 0):
        raise RuntimeError('Sol-H3 fused RMSNorm path currently targets SM120')
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f'expected fp16/bf16 activation, got {x.dtype}')
    if x.ndim != 2 or x.stride(-1) != 1:
        raise ValueError('activation must be [rows, hidden] with contiguous hidden dimension')
    if weight.ndim != 1 or int(weight.shape[0]) != int(x.shape[1]):
        raise ValueError('RMSNorm weight shape mismatch')
    if shift.ndim != 2 or scale.shape != shift.shape or int(shift.shape[1]) != int(x.shape[1]):
        raise ValueError('AdaLN shift/scale shape mismatch')
    if shift.stride(-1) != 1 or scale.stride(-1) != 1 or shift.stride(0) != scale.stride(0):
        raise ValueError('AdaLN shift/scale must share row-addressable layout')


def fused_rmsnorm_modulate_segment(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    *,
    eps: float,
    scalar_row: int | None = None,
    row_ids: torch.Tensor | None = None,
    row_offset: int = 0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """RMSNorm + AdaLN scale/shift for one contiguous sequence segment."""
    _validate(x, weight, shift, scale)
    if (scalar_row is None) == (row_ids is None):
        raise ValueError('provide exactly one of scalar_row or row_ids')
    rows, cols = int(x.shape[0]), int(x.shape[1])
    if out is None:
        out = torch.empty_like(x)
    elif out.shape != x.shape or out.dtype != x.dtype or out.device != x.device or out.stride(-1) != 1:
        raise ValueError('provided output view does not match activation contract')
    block = _next_pow2(cols)
    dummy = shift if row_ids is None else row_ids
    if row_ids is not None:
        if row_ids.device != x.device or row_ids.dtype != torch.long:
            row_ids = row_ids.to(device=x.device, dtype=torch.long)
        if row_ids.ndim != 1:
            row_ids = row_ids.reshape(-1)
        if int(row_offset) < 0 or int(row_offset) + rows > int(row_ids.numel()):
            raise ValueError('row_ids slice does not cover segment')
        dummy = row_ids
    _rmsnorm_modulate_kernel[(rows,)](
        out, x, weight, shift, scale, dummy,
        cols, x.stride(0), shift.stride(0), int(row_offset),
        eps=float(eps), PER_ROW=row_ids is not None,
        SCALAR_ROW=int(0 if scalar_row is None else scalar_row),
        BLOCK=block, num_warps=_warps(block),
    )
    return out
