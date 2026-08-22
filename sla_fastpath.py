"""Zero-copy H3 SLA fast path for LongMedia/VECTRA.

The routing algorithm and forward kernel are derived from LightX2V SLA via the
Apache-2.0 ComfyUI-H3-SLA-Attention port, but adapted for H3's fused QKV layout:

* Q/K/V remain strided views of the single [S, 3*hidden] projection. No 3x
  contiguous copies are materialized.
* sparse attention writes in-place into the Q slice of the fused QKV buffer.
* output projection is streamed in token chunks into the residual-width norm1
  buffer, so there is no full-width attention-output allocation.

This is inference-only and preserves the exact SLA block-selection semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class SLAConfig:
    sparsity_ratio: float = 0.90
    block_q: int = 64
    block_k: int = 64
    protect_audio: bool = True
    plugin_state: dict[str, Any] | None = None


def _mean_pool_strided(x: torch.Tensor, block: int, *, block_chunk: int = 16) -> torch.Tensor:
    """Mean-pool BLHD blocks with bounded temporary memory.

    v0.4.46 fixes the hidden full-sequence FP32 materialization in the previous
    implementation.  ``full.float().mean(dim=2)`` converted the complete Q/K
    view to FP32 before reduction, which costs ~3.2 GiB for H3 S~=112k.  We now
    allocate only the compact FP32 pooled output and reduce a small number of
    blocks at a time.  The reduction math is unchanged.
    """
    if x.ndim != 4:
        raise ValueError(f"SLA expects BLHD tensor, got {tuple(x.shape)}")
    b, length, heads, dim = map(int, x.shape)
    block = int(block)
    if block <= 0:
        raise ValueError(f"invalid SLA block size {block}")
    if dim > 256:
        raise ValueError(f"unsupported SLA head dim {dim}")

    nblocks = (length + block - 1) // block
    pooled = torch.empty((b, heads, nblocks, dim), device=x.device, dtype=torch.float32)
    full_blocks = length // block
    chunk_blocks = max(1, int(block_chunk))

    for bs in range(0, full_blocks, chunk_blocks):
        be = min(full_blocks, bs + chunk_blocks)
        token_start = bs * block
        token_end = be * block
        # The reshape is a view over the fused-QKV strided slice.  Only this
        # bounded token window is converted to FP32 for the reduction.
        chunk = x[:, token_start:token_end].reshape(b, be - bs, block, heads, dim)
        pooled[:, :, bs:be, :].copy_(
            chunk.float().mean(dim=2).permute(0, 2, 1, 3)
        )

    if full_blocks < nblocks:
        tail_start = full_blocks * block
        pooled[:, :, full_blocks, :].copy_(
            x[:, tail_start:length].float().mean(dim=1)
        )
    return pooled


def build_lut(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    sparsity_ratio: float,
    block_q: int,
    block_k: int,
    protect_upto: int = 0,
    head_chunk: int = 2,
) -> tuple[torch.Tensor, int, int]:
    """Build the SLA top-k LUT with bounded routing workspace.

    v0.4.46 also removes the monolithic ``scores=[B,H,NQ,NK]`` tensor.  At
    S~=112k, H=56 and 128x64 routing that tensor is ~330 MiB and ``topk`` can
    require much larger transient workspace.  Routing is independent per head,
    so we compute a small head group, write its final INT32 indices, then free
    the score workspace before moving to the next group.
    """
    topk_ratio = 1.0 - float(sparsity_ratio)
    pooled_q = _mean_pool_strided(q, int(block_q))
    pooled_k = _mean_pool_strided(k, int(block_k))

    length = int(k.shape[1])
    nk = int(pooled_k.shape[-2])
    if nk == 1 or length % int(block_k) == 0:
        mu = pooled_k.mean(dim=-2)
    else:
        full = nk - 1
        tail = length - full * int(block_k)
        weighted = pooled_k[..., :full, :].sum(dim=-2) * float(block_k)
        weighted = weighted + pooled_k[..., full, :] * float(tail)
        mu = weighted / float(length)
    pooled_k.sub_(mu.unsqueeze(-2))

    if pooled_q.shape[1] != pooled_k.shape[1]:
        qh, kh = int(pooled_q.shape[1]), int(pooled_k.shape[1])
        if qh % kh:
            raise ValueError(f"SLA GQA head mismatch q={qh} kv={kh}")
        pooled_k = pooled_k.repeat_interleave(qh // kh, dim=1)

    b, heads, nq, _ = map(int, pooled_q.shape)
    nk = int(pooled_k.shape[-2])
    topk = max(1, min(nk, int(topk_ratio * nk)))
    pinned = 0
    if int(protect_upto) > 0:
        pinned = min((int(protect_upto) + int(block_k) - 1) // int(block_k), nk)
        if pinned:
            topk = min(nk, topk + pinned)

    lut = torch.empty((b, heads, nq, topk), device=q.device, dtype=torch.int32)
    hc = max(1, int(head_chunk))
    for hs in range(0, heads, hc):
        he = min(heads, hs + hc)
        scores = pooled_q[:, hs:he] @ pooled_k[:, hs:he].transpose(-1, -2)
        if pinned:
            scores[..., :pinned] = float("inf")
        idx = torch.topk(scores, topk, dim=-1, sorted=False).indices
        lut[:, hs:he].copy_(idx.to(torch.int32))
        del scores, idx

    del pooled_q, pooled_k, mu
    return lut.contiguous(), topk, pinned


@triton.jit
def _attn_fwd_strided(
    Q, K, V, O,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    LUT,
    H: tl.constexpr,
    LQ: tl.constexpr,
    LK: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    QS_B, QS_L, QS_H, QS_D,
    KS_B, KS_L, KS_H, KS_D,
    VS_B, VS_L, VS_H, VS_D,
    OS_B, OS_L, OS_H, OS_D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)
    idx_b = idx_bh // H
    idx_h = idx_bh % H
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    q_ptrs = Q + idx_b * QS_B + offs_m[:, None] * QS_L + idx_h * QS_H + offs_d[None, :] * QS_D
    o_ptrs = O + idx_b * OS_B + offs_m[:, None] * OS_L + idx_h * OS_H + offs_d[None, :] * OS_D
    lut_ptr = LUT + lut_offset

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < LQ, other=0.0)

    for block_idx in tl.range(topk):
        idx_n = tl.load(lut_ptr + block_idx).to(tl.int64)
        k_start = idx_n * BLOCK_N
        k_pos = k_start + offs_n
        k_mask = k_pos < LK
        k_ptrs = K + idx_b * KS_B + k_pos[None, :] * KS_L + idx_h * KS_H + offs_d[:, None] * KS_D
        v_ptrs = V + idx_b * VS_B + k_pos[:, None] * VS_L + idx_h * VS_H + offs_d[None, :] * VS_D
        kk = tl.load(k_ptrs, mask=k_mask[None, :], other=0.0)
        qk = tl.dot(q, kk) * (qk_scale * 1.4426950408889634)
        qk = tl.where(k_mask[None, :], qk, float("-inf"))
        vv = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        local_m = tl.max(qk, 1)
        new_m = tl.maximum(m_i, local_m)
        qk = qk - new_m[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - new_m)
        o_s = o_s * alpha[:, None]
        o_s += tl.dot(p.to(vv.dtype), vv)
        l_i = l_i * alpha + l_ij
        m_i = new_m

    o_s = o_s / l_i[:, None]
    tl.store(o_ptrs, o_s.to(O.type.element_ty), mask=offs_m[:, None] < LQ)


_LADDER = {
    (128, 64): ((8, 3), (4, 3), (8, 2), (4, 1)),
    (128, 128): ((8, 2), (4, 2), (8, 1), (4, 1)),
    (64, 128): ((4, 2), (8, 2), (4, 1)),
    (64, 64): ((4, 1), (4, 3), (8, 3), (8, 1)),
}
_CHOSEN: dict[tuple[int, int, int], tuple[int, int]] = {}


def sparse_attention_into(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lut: torch.Tensor,
    topk: int,
    block_q: int,
    block_k: int,
) -> torch.Tensor:
    """Run block-sparse attention into caller-owned BLHD output storage."""
    if not lut.is_contiguous():
        lut = lut.contiguous()
    b, lq, heads, dim = map(int, q.shape)
    lk = int(k.shape[1])
    if tuple(out.shape) != tuple(q.shape):
        raise ValueError(f"SLA output shape mismatch {tuple(out.shape)} != {tuple(q.shape)}")
    mblocks = triton.cdiv(lq, int(block_q))
    key = (int(block_q), int(block_k), dim)
    ladder = (_CHOSEN[key],) if key in _CHOSEN else _LADDER[(int(block_q), int(block_k))]
    last: Exception | None = None
    for num_warps, num_stages in ladder:
        try:
            _attn_fwd_strided[(mblocks, b * heads)](
                q, k, v, out,
                float(dim ** -0.5), int(topk), lut,
                heads, lq, lk, mblocks, dim,
                *map(int, q.stride()),
                *map(int, k.stride()),
                *map(int, v.stride()),
                *map(int, out.stride()),
                int(block_q), int(block_k),
                num_warps=int(num_warps), num_stages=int(num_stages),
            )
        except triton.runtime.errors.OutOfResources as exc:
            last = exc
            continue
        _CHOSEN[key] = (int(num_warps), int(num_stages))
        return out
    raise last if last is not None else RuntimeError("no viable SLA launch config")
