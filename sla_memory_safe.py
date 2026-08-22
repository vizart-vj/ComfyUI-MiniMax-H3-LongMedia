"""Memory-safe compatibility path for BetaDoggo H3 SLA Attention.

The external project is Apache-2.0 and derives its SLA kernels from LightX2V.
This module preserves its block selection / sparsity semantics while avoiding
its full-size ``torch.empty_like(q)`` output allocation.  Output is written
in-place into Q storage after block-map construction, which is safe because Q
is dead after self-attention.  Block-score construction is streamed by query
blocks to reduce peak VRAM without changing selected blocks.
"""
from __future__ import annotations

import torch


def _external_ops():
    # The installed custom node exposes these as top-level ``sla`` modules.
    from sla.block_map import mean_pool
    import triton
    import triton.language as tl
    return mean_pool, triton, tl


# Triton must see the JIT function at import time. Import lazily-safe here: this
# module itself is only imported when an external SLA override is detected.
try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - only reached without SLA/Triton installed
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _attn_fwd_inplace(
        Q, K, V, qk_scale: tl.constexpr, topk: tl.constexpr, LUT,
        H: tl.constexpr, LQ: tl.constexpr, LK: tl.constexpr,
        M_BLOCKS: tl.constexpr, D: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        idx_m = tl.program_id(0).to(tl.int64)
        idx_bh = tl.program_id(1).to(tl.int64)
        idx_b = idx_bh // H
        idx_h = idx_bh % H
        HD: tl.constexpr = H * D
        q_offset = idx_b * LQ * HD + idx_h * D
        kv_offset = idx_b * LK * HD + idx_h * D
        lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk
        offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        Q_ptrs = Q + q_offset + offs_m[:, None] * HD + offs_d[None, :]
        # Reuse Q storage as O storage. Every program reads its complete Q block
        # before writing that same disjoint block, so there is no cross-program
        # dependency or alias with K/V.
        O_ptrs = Q_ptrs
        LUT_ptr = LUT + lut_offset
        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)
        q = tl.load(Q_ptrs, mask=offs_m[:, None] < LQ, other=0.0)
        for block_idx in tl.range(topk):
            idx_n = tl.load(LUT_ptr + block_idx).to(tl.int64)
            k_start = idx_n * BLOCK_N
            k_mask = (k_start + offs_n) < LK
            K_ptrs = K + kv_offset + (k_start + offs_n)[None, :] * HD + offs_d[:, None]
            V_ptrs = V + kv_offset + (k_start + offs_n)[:, None] * HD + offs_d[None, :]
            k = tl.load(K_ptrs, mask=k_mask[None, :], other=0.0)
            qk = tl.dot(q, k) * (qk_scale * 1.4426950408889634)
            qk = tl.where(k_mask[None, :], qk, float("-inf"))
            v = tl.load(V_ptrs, mask=k_mask[:, None], other=0.0)
            local_m = tl.max(qk, 1)
            new_m = tl.maximum(m_i, local_m)
            qk = qk - new_m[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - new_m)
            o_s = o_s * alpha[:, None]
            o_s += tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + l_ij
            m_i = new_m
        o_s = o_s / l_i[:, None]
        tl.store(O_ptrs, o_s.to(Q.type.element_ty), mask=offs_m[:, None] < LQ)


_LADDER = {
    (128, 64): ((8, 3), (4, 3), (8, 2), (4, 1)),
    (128, 128): ((8, 2), (4, 2), (8, 1), (4, 1)),
    (64, 128): ((4, 2), (8, 2), (4, 1)),
    (64, 64): ((4, 1), (4, 3), (8, 3), (8, 1)),
}
_CHOSEN = {}


def _streamed_block_map(q, k, topk_ratio, blkq, blkk, protect_upto=0, qblock_chunk=64):
    from sla.block_map import mean_pool
    pooled_q = mean_pool(q, blkq)
    mu = k.mean(dim=1, dtype=torch.float32)
    pooled_k = mean_pool(k, blkk) - mu[:, :, None, :]
    nq_heads, nk_heads = pooled_q.shape[1], pooled_k.shape[1]
    if nq_heads != nk_heads:
        if nq_heads % nk_heads:
            raise RuntimeError('SLA GQA head mismatch')
        pooled_k = pooled_k.repeat_interleave(nq_heads // nk_heads, dim=1)
    nk = int(pooled_k.shape[-2])
    base_topk = max(1, min(nk, int(float(topk_ratio) * nk)))
    n_pinned = 0
    topk = base_topk
    if protect_upto > 0:
        n_pinned = min((int(protect_upto) + blkk - 1) // blkk, nk)
        topk = min(nk, base_topk + n_pinned)
    nq = int(pooled_q.shape[-2])
    lut = torch.empty(
        (int(pooled_q.shape[0]), int(pooled_q.shape[1]), nq, int(topk)),
        device=pooled_q.device, dtype=torch.int32,
    )
    pooled_k_t = pooled_k.transpose(-1, -2)
    for start in range(0, nq, int(qblock_chunk)):
        stop = min(nq, start + int(qblock_chunk))
        score = pooled_q[:, :, start:stop, :] @ pooled_k_t
        if n_pinned:
            score[..., :n_pinned] = float('inf')
        indices = torch.topk(score, topk, dim=-1, sorted=False).indices
        lut[:, :, start:stop, :].copy_(indices)
        del indices, score
    return lut.contiguous(), topk


def _block_sparse_attention_inplace(q, k, v, lut, topk, block_m, block_n, qk_scale=None):
    if triton is None:
        raise RuntimeError('Triton unavailable for memory-safe SLA')
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and lut.is_contiguous()):
        raise RuntimeError('memory-safe SLA requires contiguous BLHD Q/K/V/LUT')
    b, lq, h, d = q.shape
    lk = int(k.shape[1])
    if qk_scale is None:
        qk_scale = d ** -0.5
    m_blocks = triton.cdiv(lq, block_m)
    grid = (m_blocks, b * h)
    key = (block_m, block_n, d)
    ladder = (_CHOSEN[key],) if key in _CHOSEN else _LADDER[(block_m, block_n)]
    last = None
    for cfg in ladder:
        try:
            _attn_fwd_inplace[grid](
                q, k, v, qk_scale, topk, lut,
                h, lq, lk, m_blocks, d, block_m, block_n,
                num_warps=cfg[0], num_stages=cfg[1],
            )
        except triton.runtime.errors.OutOfResources as exc:
            last = exc
            continue
        _CHOSEN[key] = cfg
        return q
    raise last if last is not None else RuntimeError('no viable SLA launch config')


def make_memory_safe_sla_override(*, ext_state, sparsity_ratio, blkq, blkk, min_seq_len, protect_audio=True):
    topk_ratio = 1.0 - float(sparsity_ratio)
    if ext_state is None:
        ext_state = {}

    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        def dense():
            ext_state['dense'] = int(ext_state.get('dense', 0)) + 1
            return func(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                        skip_reshape=skip_reshape,
                        skip_output_reshape=skip_output_reshape, **kwargs)

        if ext_state.get('backend') is None:
            ext_state['backend'] = getattr(func, '__name__', repr(func))
        to = kwargs.get('transformer_options') or {}
        if (
            not skip_reshape or mask is not None or q.ndim != 4 or q.shape[-1] != 128
            or q.dtype not in (torch.bfloat16, torch.float16)
            or int(q.shape[2]) < int(min_seq_len) or to.get('_h3sla_dense', False)
        ):
            return dense()
        try:
            b, h, s, d = q.shape
            qb, kb, vb = (t.transpose(1, 2) for t in (q, k, v))
            # H3's transpose-back is normally already contiguous. Do not create
            # multi-GiB emergency copies if a foreign layout reaches this path.
            if not (qb.is_contiguous() and kb.is_contiguous() and vb.is_contiguous()):
                return dense()
            prefix = int(to.get('_h3sla_prefix', 0) or 0) if protect_audio else 0
            if prefix >= s:
                prefix = 0
            lut, topk = _streamed_block_map(qb, kb, topk_ratio, blkq, blkk, protect_upto=prefix)
            out = _block_sparse_attention_inplace(qb, kb, vb, lut, topk, blkq, blkk)
            ext_state['calls'] = int(ext_state.get('calls', 0)) + 1
            ext_state['seq'] = int(s)
            ext_state['kept'] = int(topk)
            ext_state['blocks'] = int((s + blkk - 1) // blkk)
            ext_state['pinned'] = int((prefix + blkk - 1) // blkk)
            ext_state['longmedia_memory_safe'] = True
            if skip_output_reshape:
                return out.transpose(1, 2)
            return out.reshape(b, s, h * d)
        except torch.cuda.OutOfMemoryError:
            # Never follow a sparse OOM with the external dense fallback: dense
            # requires more workspace and was the second OOM in the reported log.
            ext_state['failed'] = 'CUDA OOM in memory-safe SLA (dense fallback suppressed)'
            raise
        except Exception as exc:
            if ext_state.get('failed') is None:
                ext_state['failed'] = f'{type(exc).__name__}: {exc}'
            return dense()
    return override
