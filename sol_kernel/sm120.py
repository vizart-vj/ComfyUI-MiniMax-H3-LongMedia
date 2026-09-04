# Adapted from Saganaki22/ComfyUI-sol-attn (Apache-2.0), 2026.
# Modified for MiniMax-H3-LongMedia: narrow SM120 BF16 pointer path,
# no dependency on the external custom-node package.
#
# IMPORTANT: Triton 3.x does not allow arbitrary Python globals to be captured
# from @triton.jit functions. All compile-time constants used by kernels are
# therefore explicit tl.constexpr parameters.

import torch
import triton
import triton.language as tl

from ..triton_windows_compat import install_windows_triton_build_compat

# Must run before the first Triton kernel launch: Windows initializes cuda_utils
# lazily from driver.active on that first launch.  The shim only augments JIT
# include/library search paths and is a no-op on non-Windows platforms.
_TRITON_WINDOWS_BUILD_COMPAT = install_windows_triton_build_compat()

BLOCK_SIZE = 64
GROUP_SIZE = 32
HEAD_DIM = 128


@triton.jit
def _reduce_k_kernel(
    k_ptr, kc_ptr, T, s_b, s_t, s_h,
    H: tl.constexpr, NB: tl.constexpr, D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    rows = block * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    valid = rows < T
    vals = tl.load(
        k_ptr + batch * s_b + rows[:, None].to(tl.int64) * s_t + head * s_h + d[None, :],
        mask=valid[:, None], other=0.0,
    )
    block_len = tl.minimum(BLOCK, T - block * BLOCK).to(tl.float32)
    mean = tl.sum(vals.to(tl.float32), axis=0) / block_len
    tl.store(kc_ptr + ((batch * NB + block) * H + head) * D + d, mean)


@triton.jit
def _reduce_v_kernel(
    v_ptr, vc_ptr, T, s_b, s_t, s_h,
    H: tl.constexpr, NB: tl.constexpr, D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    rows = block * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    valid = rows < T
    vals = tl.load(
        v_ptr + batch * s_b + rows[:, None].to(tl.int64) * s_t + head * s_h + d[None, :],
        mask=valid[:, None], other=0.0,
    )
    summed = tl.sum(vals, axis=0)
    tl.store(vc_ptr + ((batch * NB + block) * H + head) * D + d, summed)


@triton.jit
def _kc_stats_kernel(
    kc_ptr, mean_ptr, var_ptr,
    H: tl.constexpr, NB: tl.constexpr, D: tl.constexpr,
):
    batch_head = tl.program_id(0)
    batch, head = batch_head // H, batch_head % H
    d = tl.arange(0, D)
    total = tl.zeros((D,), dtype=tl.float32)
    total_sq = tl.zeros((D,), dtype=tl.float32)
    for i in range(0, NB):
        value = tl.load(kc_ptr + ((batch * NB + i) * H + head) * D + d).to(tl.float32)
        total += value
        total_sq += value * value
    mean = total / NB
    var = tl.maximum(total_sq / NB - mean * mean, 0.0)
    tl.store(mean_ptr + batch_head * D + d, mean)
    tl.store(var_ptr + batch_head * D + d, var)


@triton.jit
def _threshold_kernel(
    q_ptr, mean_ptr, var_ptr, thr_ptr,
    softmax_scale, tau, T, s_b, s_t, s_h,
    H: tl.constexpr, NB: tl.constexpr, D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    q_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    rows = q_block * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    valid = rows < T
    q = tl.load(
        q_ptr + batch * s_b + rows[:, None].to(tl.int64) * s_t + head * s_h + d[None, :],
        mask=valid[:, None], other=0.0,
    ).to(tl.float32)
    q_len = tl.minimum(BLOCK, T - q_block * BLOCK).to(tl.float32)
    centroid = tl.sum(q, axis=0) / q_len
    mean_k = tl.load(mean_ptr + batch_head * D + d)
    var_k = tl.load(var_ptr + batch_head * D + d)
    scale_log2 = softmax_scale * 1.4426950408889634
    mu = tl.sum(centroid * mean_k, axis=0) * scale_log2
    vv = tl.sum(centroid * centroid * var_k, axis=0) * (scale_log2 * scale_log2)
    threshold = mu + tau * tl.sqrt(tl.maximum(vv, 0.0) + 1.0e-6)
    tl.store(thr_ptr + (batch * NB + q_block) * H + head, threshold)


@triton.jit
def _forward_ptr(
    q_ptr, k_ptr, v_ptr, kc_ptr, vc_ptr, threshold, out_ptr,
    scale, sink_start, sink_end, sink_q_start, sink_q_end,
    T, sq_b, sq_t, sq_h, sk_b, sk_t, sk_h, sv_b, sv_t, sv_h,
    H: tl.constexpr, D: tl.constexpr, NB: tl.constexpr,
    BLOCK: tl.constexpr, GROUP: tl.constexpr,
):
    q_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    q_rows = q_block * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    valid_q = q_rows < T
    q = tl.load(
        q_ptr + batch * sq_b + q_rows[:, None].to(tl.int64) * sq_t + head * sq_h + d[None, :],
        mask=valid_q[:, None], other=0.0,
    )
    q_len = tl.minimum(BLOCK, T - q_block * BLOCK).to(tl.float32)
    scale_log2 = scale * 1.4426950408889634
    route_threshold = tl.load(threshold + (batch * NB + q_block) * H + head)
    q_in_sink = (q_block >= sink_q_start) & (q_block < sink_q_end)

    output = tl.zeros((BLOCK, D), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK,), dtype=tl.float32)
    row_max = tl.full((BLOCK,), -float('inf'), tl.float32)
    tail_len = T - (NB - 1) * BLOCK
    group_offsets = tl.max_contiguous(tl.arange(0, GROUP), GROUP)
    token_offsets = tl.max_contiguous(tl.arange(0, BLOCK), BLOCK)

    for group_start in range(0, NB, GROUP):
        block_indices = group_start + group_offsets
        valid_blocks = block_indices < NB
        kc = tl.load(
            kc_ptr + ((batch * NB + block_indices[:, None]) * H + head) * D + d[None, :],
            mask=valid_blocks[:, None], other=0.0,
        )
        vc = tl.load(
            vc_ptr + ((batch * NB + block_indices[:, None]) * H + head) * D + d[None, :],
            mask=valid_blocks[:, None], other=0.0,
        )
        scores = tl.dot(q, kc.T).to(tl.float32) * scale_log2
        sink_kv = (block_indices >= sink_start) & (block_indices < sink_end)
        routed = (
            (tl.sum(scores, axis=0) / q_len > route_threshold)
            | (tl.abs(q_block - block_indices) <= 1)
            | sink_kv
        ) & valid_blocks
        exact = tl.where(q_in_sink, valid_blocks, routed)

        approximate = valid_blocks & ~exact
        approx_scores = tl.where(approximate[None, :], scores, -float('inf'))
        new_max = tl.maximum(row_max, tl.max(approx_scores, axis=1))
        alpha = tl.math.exp2(tl.where(row_max == new_max, 0.0, row_max - new_max))
        probs = tl.where(
            approximate[None, :], tl.math.exp2(approx_scores - new_max[:, None]), 0.0
        )
        output = output * alpha[:, None] + tl.dot(probs.to(vc.dtype), vc)
        lengths = tl.where(block_indices == NB - 1, tail_len, BLOCK).to(tl.float32)
        row_sum = row_sum * alpha + tl.sum(probs * lengths[None, :], axis=1)
        row_max = new_max

        exact_offsets = tl.where(exact, group_offsets, GROUP)
        for _ in range(tl.sum(exact.to(tl.int32))):
            offset = tl.min(exact_offsets)
            block_index = group_start + offset
            exact_offsets = tl.where(group_offsets == offset, GROUP, exact_offsets)
            k_rows = block_index * BLOCK + token_offsets
            valid_k = k_rows < T
            k = tl.load(
                k_ptr + batch * sk_b + k_rows[:, None].to(tl.int64) * sk_t + head * sk_h + d[None, :],
                mask=valid_k[:, None], other=0.0,
            )
            exact_scores = tl.dot(q, k.T).to(tl.float32) * scale_log2
            exact_scores += tl.where(valid_k[None, :], 0.0, -float('inf'))
            new_max = tl.maximum(row_max, tl.max(exact_scores, axis=1))
            alpha = tl.math.exp2(row_max - new_max)
            probability = tl.math.exp2(exact_scores - new_max[:, None])
            row_sum = row_sum * alpha + tl.sum(probability, axis=1)
            v = tl.load(
                v_ptr + batch * sv_b + k_rows[:, None].to(tl.int64) * sv_t + head * sv_h + d[None, :],
                mask=valid_k[:, None], other=0.0,
            )
            output = output * alpha[:, None] + tl.dot(probability.to(v.dtype), v)
            row_max = new_max

    tl.store(
        out_ptr + ((batch * T + q_rows[:, None]) * H + head) * D + d[None, :],
        (output / row_sum[:, None]).to(tl.bfloat16),
        mask=valid_q[:, None],
    )


def _validate(q, k, v):
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError('q/k/v must be [B,T,H,128] and share shape')
    if q.shape[-1] != HEAD_DIM:
        raise ValueError('Sol-Attn requires head_dim=128')
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise TypeError('Sol-Attn requires bfloat16 q/k/v')
    if q.device.type != 'cuda' or k.device != q.device or v.device != q.device:
        raise ValueError('q/k/v must share one CUDA device')
    arch = torch.cuda.get_device_capability(q.device)
    if arch != (12, 0):
        raise RuntimeError(f'LongMedia Sol path currently targets SM120; got SM{arch[0]}{arch[1]}')


def sol_attn_sm120(q, k, v, *, tau=1.3, scale=None, sink_blocks=(0, 0), sink_q=(0, 0)):
    """SM120 BF16 pointer Sol-Attn adapted from ComfyUI-sol-attn.

    q/k/v may be non-contiguous strided views into H3's single fused QKV
    projection buffer. No contiguous copies are made by this function.
    """
    _validate(q, k, v)
    B, T, H, D = q.shape
    NB = triton.cdiv(T, BLOCK_SIZE)
    scale = float(D ** -0.5 if scale is None else scale)

    # Only compact block summaries + thresholds are materialized here.
    kc = torch.empty((B, NB, H, D), device=q.device, dtype=torch.bfloat16)
    vc = torch.empty_like(kc)
    _reduce_k_kernel[(NB, B * H)](
        k, kc, T, k.stride(0), k.stride(1), k.stride(2),
        H, NB, D, BLOCK_SIZE, num_warps=4,
    )
    _reduce_v_kernel[(NB, B * H)](
        v, vc, T, v.stride(0), v.stride(1), v.stride(2),
        H, NB, D, BLOCK_SIZE, num_warps=4,
    )
    kc_mean = torch.empty((B, H, D), device=q.device, dtype=torch.float32)
    kc_var = torch.empty_like(kc_mean)
    _kc_stats_kernel[(B * H,)](kc, kc_mean, kc_var, H, NB, D, num_warps=4)
    threshold = torch.empty((B, NB, H), device=q.device, dtype=torch.float32)
    _threshold_kernel[(NB, B * H)](
        q, kc_mean, kc_var, threshold, scale, float(tau), T,
        q.stride(0), q.stride(1), q.stride(2), H, NB, D, BLOCK_SIZE,
        num_warps=4,
    )
    # Kernels below store out_ptr with contiguous [B,T,H,D] indexing.
    # Q may be a strided view into fused QKV, so make the output contract
    # explicit instead of relying on empty_like(preserve_format) heuristics.
    out = torch.empty((B, T, H, D), device=q.device, dtype=q.dtype)
    sinks = tuple(int(value) for value in (*sink_blocks, *sink_q))
    _forward_ptr[(NB, B * H)](
        q, k, v, kc, vc, threshold, out, scale, *sinks, T,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        H, D, NB, BLOCK_SIZE, GROUP_SIZE,
        num_warps=8, num_stages=2,
    )
    return out


# --- LongMedia streamed-QKV rectangular query path -------------------------
# Added in LongMedia 0.2.31. K/V remain full sequence, while Q is projected
# and consumed in chunks. This avoids materializing the full 3-way QKV tensor.

@triton.jit
def _forward_ptr_rect(
    q_ptr, k_ptr, v_ptr, kc_ptr, vc_ptr, threshold, out_ptr,
    scale, sink_start, sink_end, sink_q_start, sink_q_end,
    TQ, TK, q_block_offset,
    sq_b, sq_t, sq_h, sk_b, sk_t, sk_h, sv_b, sv_t, sv_h,
    H: tl.constexpr, D: tl.constexpr, NBQ: tl.constexpr, NBK: tl.constexpr,
    BLOCK: tl.constexpr, GROUP: tl.constexpr,
):
    q_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    q_rows = q_block * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    valid_q = q_rows < TQ
    q = tl.load(
        q_ptr + batch * sq_b + q_rows[:, None].to(tl.int64) * sq_t + head * sq_h + d[None, :],
        mask=valid_q[:, None], other=0.0,
    )
    q_len = tl.minimum(BLOCK, TQ - q_block * BLOCK).to(tl.float32)
    scale_log2 = scale * 1.4426950408889634
    route_threshold = tl.load(threshold + (batch * NBQ + q_block) * H + head)
    global_q_block = q_block_offset + q_block
    q_in_sink = (global_q_block >= sink_q_start) & (global_q_block < sink_q_end)

    output = tl.zeros((BLOCK, D), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK,), dtype=tl.float32)
    row_max = tl.full((BLOCK,), -float('inf'), tl.float32)
    tail_len = TK - (NBK - 1) * BLOCK
    group_offsets = tl.max_contiguous(tl.arange(0, GROUP), GROUP)
    token_offsets = tl.max_contiguous(tl.arange(0, BLOCK), BLOCK)
    exact_total = tl.zeros((), dtype=tl.int32)
    approx_total = tl.zeros((), dtype=tl.int32)
    score_route_total = tl.zeros((), dtype=tl.int32)
    local_force_total = tl.zeros((), dtype=tl.int32)
    sink_force_total = tl.zeros((), dtype=tl.int32)

    for group_start in range(0, NBK, GROUP):
        block_indices = group_start + group_offsets
        valid_blocks = block_indices < NBK
        kc = tl.load(
            kc_ptr + ((batch * NBK + block_indices[:, None]) * H + head) * D + d[None, :],
            mask=valid_blocks[:, None], other=0.0,
        )
        vc = tl.load(
            vc_ptr + ((batch * NBK + block_indices[:, None]) * H + head) * D + d[None, :],
            mask=valid_blocks[:, None], other=0.0,
        )
        scores = tl.dot(q, kc.T).to(tl.float32) * scale_log2
        sink_kv = (block_indices >= sink_start) & (block_indices < sink_end)
        routed = (
            (tl.sum(scores, axis=0) / q_len > route_threshold)
            | (tl.abs(global_q_block - block_indices) <= 1)
            | sink_kv
        ) & valid_blocks
        exact = tl.where(q_in_sink, valid_blocks, routed)

        approximate = valid_blocks & ~exact
        approx_scores = tl.where(approximate[None, :], scores, -float('inf'))
        new_max = tl.maximum(row_max, tl.max(approx_scores, axis=1))
        alpha = tl.math.exp2(tl.where(row_max == new_max, 0.0, row_max - new_max))
        probs = tl.where(
            approximate[None, :], tl.math.exp2(approx_scores - new_max[:, None]), 0.0
        )
        output = output * alpha[:, None] + tl.dot(probs.to(vc.dtype), vc)
        lengths = tl.where(block_indices == NBK - 1, tail_len, BLOCK).to(tl.float32)
        row_sum = row_sum * alpha + tl.sum(probs * lengths[None, :], axis=1)
        row_max = new_max

        exact_offsets = tl.where(exact, group_offsets, GROUP)
        for _ in range(tl.sum(exact.to(tl.int32))):
            offset = tl.min(exact_offsets)
            block_index = group_start + offset
            exact_offsets = tl.where(group_offsets == offset, GROUP, exact_offsets)
            k_rows = block_index * BLOCK + token_offsets
            valid_k = k_rows < TK
            k = tl.load(
                k_ptr + batch * sk_b + k_rows[:, None].to(tl.int64) * sk_t + head * sk_h + d[None, :],
                mask=valid_k[:, None], other=0.0,
            )
            exact_scores = tl.dot(q, k.T).to(tl.float32) * scale_log2
            exact_scores += tl.where(valid_k[None, :], 0.0, -float('inf'))
            new_max = tl.maximum(row_max, tl.max(exact_scores, axis=1))
            alpha = tl.math.exp2(row_max - new_max)
            probability = tl.math.exp2(exact_scores - new_max[:, None])
            row_sum = row_sum * alpha + tl.sum(probability, axis=1)
            vv = tl.load(
                v_ptr + batch * sv_b + k_rows[:, None].to(tl.int64) * sv_t + head * sv_h + d[None, :],
                mask=valid_k[:, None], other=0.0,
            )
            output = output * alpha[:, None] + tl.dot(probability.to(vv.dtype), vv)
            row_max = new_max

    tl.store(
        out_ptr + ((batch * TQ + q_rows[:, None]) * H + head) * D + d[None, :],
        (output / row_sum[:, None]).to(tl.bfloat16),
        mask=valid_q[:, None],
    )


def prepare_kv_sm120(k, v):
    """Prepare full-sequence K/V and compact Sol summaries once."""
    if k.ndim != 4 or v.shape != k.shape or k.shape[-1] != HEAD_DIM:
        raise ValueError('streamed Sol K/V must be matching [B,T,H,128]')
    if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError('streamed Sol K/V requires bfloat16')
    B, TK, H, D = k.shape
    NBK = triton.cdiv(TK, BLOCK_SIZE)
    kc = torch.empty((B, NBK, H, D), device=k.device, dtype=torch.bfloat16)
    vc = torch.empty_like(kc)
    _reduce_k_kernel[(NBK, B * H)](
        k, kc, TK, k.stride(0), k.stride(1), k.stride(2),
        H, NBK, D, BLOCK_SIZE, num_warps=4,
    )
    _reduce_v_kernel[(NBK, B * H)](
        v, vc, TK, v.stride(0), v.stride(1), v.stride(2),
        H, NBK, D, BLOCK_SIZE, num_warps=4,
    )
    kc_mean = torch.empty((B, H, D), device=k.device, dtype=torch.float32)
    kc_var = torch.empty_like(kc_mean)
    _kc_stats_kernel[(B * H,)](kc, kc_mean, kc_var, H, NBK, D, num_warps=4)
    return kc, vc, kc_mean, kc_var


def sol_attn_query_sm120(q, k, v, prepared, *, q_offset=0, tau=1.3, scale=None,
                          sink_blocks=(0, 0), sink_q=(0, 0)):
    """Run Sol for a query slice against full K/V.

    q_offset must be 64-token aligned so local query block IDs map exactly to
    H3's global packed sequence block IDs.
    """
    if q.ndim != 4 or q.shape[-1] != HEAD_DIM:
        raise ValueError('streamed Sol Q must be [B,Tq,H,128]')
    if q.dtype != torch.bfloat16 or q.device.type != 'cuda':
        raise TypeError('streamed Sol Q requires CUDA bfloat16')
    if int(q_offset) % BLOCK_SIZE:
        raise ValueError('q_offset must be 64-token aligned')
    B, TQ, H, D = q.shape
    BK, TK, HK, DK = k.shape
    if (B,H,D) != (BK,HK,DK) or v.shape != k.shape:
        raise ValueError('streamed Sol Q/K/V batch/head dimensions mismatch')
    NBQ = triton.cdiv(TQ, BLOCK_SIZE)
    NBK = triton.cdiv(TK, BLOCK_SIZE)
    kc, vc, kc_mean, kc_var = prepared
    scale = float(D ** -0.5 if scale is None else scale)
    threshold = torch.empty((B, NBQ, H), device=q.device, dtype=torch.float32)
    _threshold_kernel[(NBQ, B * H)](
        q, kc_mean, kc_var, threshold, scale, float(tau), TQ,
        q.stride(0), q.stride(1), q.stride(2), H, NBQ, D, BLOCK_SIZE,
        num_warps=4,
    )
    out = torch.empty((B, TQ, H, D), device=q.device, dtype=q.dtype)
    sinks = tuple(int(value) for value in (*sink_blocks, *sink_q))
    _forward_ptr_rect[(NBQ, B * H)](
        q, k, v, kc, vc, threshold, out, scale, *sinks,
        TQ, TK, int(q_offset) // BLOCK_SIZE,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        H, D, NBQ, NBK, BLOCK_SIZE, GROUP_SIZE,
        num_warps=8, num_stages=2,
    )
    return out

# --- LongMedia 0.2.32 compressed streamed-KV path -------------------------
# Long H3 sequences cannot retain full BF16 K+V on 16 GB GPUs.  Project K/V
# in chunks, keep BF16 64-token Sol summaries, and retain token-level K/V as
# INT8 + per-token scales for exact routed blocks.  This halves token-level KV
# storage while preserving the routing summaries in BF16.

@triton.jit
def _compress_kv_chunk_kernel(
    k_ptr, v_ptr,
    kc_ptr, vc_ptr,
    k8_ptr, ks_ptr, v8_ptr, vs_ptr,
    TCHUNK, TOTAL_T, GLOBAL_START,
    sk_b, sk_t, sk_h, sv_b, sv_t, sv_h,
    H: tl.constexpr, NB_TOTAL: tl.constexpr, D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    local_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    local_rows = local_block * BLOCK + tl.arange(0, BLOCK)
    global_rows = GLOBAL_START + local_rows
    d = tl.arange(0, D)
    valid = local_rows < TCHUNK
    k = tl.load(
        k_ptr + batch * sk_b + local_rows[:, None].to(tl.int64) * sk_t + head * sk_h + d[None, :],
        mask=valid[:, None], other=0.0,
    )
    v = tl.load(
        v_ptr + batch * sv_b + local_rows[:, None].to(tl.int64) * sv_t + head * sv_h + d[None, :],
        mask=valid[:, None], other=0.0,
    )
    block_len = tl.minimum(BLOCK, TCHUNK - local_block * BLOCK).to(tl.float32)
    kmean_fp32 = tl.sum(k.to(tl.float32), axis=0) / block_len
    # Quantization is relative to the BF16-stored block mean, so the exact path
    # reconstructs the same mean value used by the routing summary.
    kmean_bf16 = kmean_fp32.to(tl.bfloat16)
    global_block = GLOBAL_START // BLOCK + local_block
    tl.store(kc_ptr + ((batch * NB_TOTAL + global_block) * H + head) * D + d, kmean_bf16)
    vsum = tl.sum(v, axis=0)
    tl.store(vc_ptr + ((batch * NB_TOTAL + global_block) * H + head) * D + d, vsum)

    kres = tl.where(valid[:, None], k.to(tl.float32) - kmean_bf16[None, :].to(tl.float32), 0.0)
    kamax = tl.max(tl.abs(kres), axis=1)
    kscale = tl.maximum(kamax / 127.0, 1.0e-8)
    kq = tl.where(kres >= 0, (kres / kscale[:, None] + 0.5).to(tl.int32),
                  (kres / kscale[:, None] - 0.5).to(tl.int32)).to(tl.int8)

    vf = v.to(tl.float32)
    vamax = tl.max(tl.abs(vf), axis=1)
    vscale = tl.maximum(vamax / 127.0, 1.0e-8)
    vq = tl.where(vf >= 0, (vf / vscale[:, None] + 0.5).to(tl.int32),
                  (vf / vscale[:, None] - 0.5).to(tl.int32)).to(tl.int8)

    dst = ((batch * TOTAL_T + global_rows[:, None]) * H + head) * D + d[None, :]
    tl.store(k8_ptr + dst, kq, mask=valid[:, None])
    tl.store(v8_ptr + dst, vq, mask=valid[:, None])
    scale_dst = (batch * TOTAL_T + global_rows) * H + head
    tl.store(ks_ptr + scale_dst, kscale, mask=valid)
    tl.store(vs_ptr + scale_dst, vscale, mask=valid)



@triton.jit
def _forward_ptr_rect_compressed(
    q_ptr, kc_ptr, vc_ptr,
    k8_ptr, ks_ptr, v8_ptr, vs_ptr,
    threshold, out_ptr, telemetry_ptr,
    scale, sink_start, sink_end, sink_q_start, sink_q_end,
    TQ, TK, q_block_offset,
    sq_b, sq_t, sq_h,
    H: tl.constexpr, D: tl.constexpr, NBQ: tl.constexpr, NBK: tl.constexpr,
    BLOCK: tl.constexpr, GROUP: tl.constexpr, COLLECT_TELEMETRY: tl.constexpr,
):
    q_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    q_rows = q_block * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    valid_q = q_rows < TQ
    q = tl.load(
        q_ptr + batch * sq_b + q_rows[:, None].to(tl.int64) * sq_t + head * sq_h + d[None, :],
        mask=valid_q[:, None], other=0.0,
    )
    q_len = tl.minimum(BLOCK, TQ - q_block * BLOCK).to(tl.float32)
    scale_log2 = scale * 1.4426950408889634
    route_threshold = tl.load(threshold + (batch * NBQ + q_block) * H + head)
    global_q_block = q_block_offset + q_block
    q_in_sink = (global_q_block >= sink_q_start) & (global_q_block < sink_q_end)

    output = tl.zeros((BLOCK, D), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK,), dtype=tl.float32)
    row_max = tl.full((BLOCK,), -float('inf'), tl.float32)
    tail_len = TK - (NBK - 1) * BLOCK
    group_offsets = tl.max_contiguous(tl.arange(0, GROUP), GROUP)
    token_offsets = tl.max_contiguous(tl.arange(0, BLOCK), BLOCK)
    exact_total = tl.zeros((), dtype=tl.int32)
    approx_total = tl.zeros((), dtype=tl.int32)
    score_route_total = tl.zeros((), dtype=tl.int32)
    local_force_total = tl.zeros((), dtype=tl.int32)
    sink_force_total = tl.zeros((), dtype=tl.int32)

    for group_start in range(0, NBK, GROUP):
        block_indices = group_start + group_offsets
        valid_blocks = block_indices < NBK
        kc = tl.load(
            kc_ptr + ((batch * NBK + block_indices[:, None]) * H + head) * D + d[None, :],
            mask=valid_blocks[:, None], other=0.0,
        )
        vc = tl.load(
            vc_ptr + ((batch * NBK + block_indices[:, None]) * H + head) * D + d[None, :],
            mask=valid_blocks[:, None], other=0.0,
        )
        scores = tl.dot(q, kc.T).to(tl.float32) * scale_log2
        sink_kv = (block_indices >= sink_start) & (block_indices < sink_end)
        score_routed = (tl.sum(scores, axis=0) / q_len > route_threshold) & valid_blocks
        local_forced = (tl.abs(global_q_block - block_indices) <= 1) & valid_blocks
        sink_forced = sink_kv & valid_blocks
        routed = (score_routed | local_forced | sink_forced) & valid_blocks
        exact = tl.where(q_in_sink, valid_blocks, routed)

        approximate = valid_blocks & ~exact
        if COLLECT_TELEMETRY:
            exact_total += tl.sum(exact.to(tl.int32))
            approx_total += tl.sum(approximate.to(tl.int32))
            score_route_total += tl.sum(score_routed.to(tl.int32))
            local_force_total += tl.sum(local_forced.to(tl.int32))
            sink_force_total += tl.sum(sink_forced.to(tl.int32))
        approx_scores = tl.where(approximate[None, :], scores, -float('inf'))
        new_max = tl.maximum(row_max, tl.max(approx_scores, axis=1))
        alpha = tl.math.exp2(tl.where(row_max == new_max, 0.0, row_max - new_max))
        probs = tl.where(approximate[None, :], tl.math.exp2(approx_scores - new_max[:, None]), 0.0)
        output = output * alpha[:, None] + tl.dot(probs.to(vc.dtype), vc)
        lengths = tl.where(block_indices == NBK - 1, tail_len, BLOCK).to(tl.float32)
        row_sum = row_sum * alpha + tl.sum(probs * lengths[None, :], axis=1)
        row_max = new_max

        exact_offsets = tl.where(exact, group_offsets, GROUP)
        for _ in range(tl.sum(exact.to(tl.int32))):
            offset = tl.min(exact_offsets)
            block_index = group_start + offset
            exact_offsets = tl.where(group_offsets == offset, GROUP, exact_offsets)
            k_rows = block_index * BLOCK + token_offsets
            valid_k = k_rows < TK
            src = ((batch * TK + k_rows[:, None]) * H + head) * D + d[None, :]
            k8 = tl.load(k8_ptr + src, mask=valid_k[:, None], other=0).to(tl.float32)
            ks = tl.load(ks_ptr + (batch * TK + k_rows) * H + head, mask=valid_k, other=1.0)
            kmean = tl.load(kc_ptr + ((batch * NBK + block_index) * H + head) * D + d).to(tl.float32)
            k = (k8 * ks[:, None] + kmean[None, :]).to(tl.bfloat16)
            exact_scores = tl.dot(q, k.T).to(tl.float32) * scale_log2
            exact_scores += tl.where(valid_k[None, :], 0.0, -float('inf'))
            new_max = tl.maximum(row_max, tl.max(exact_scores, axis=1))
            alpha = tl.math.exp2(row_max - new_max)
            probability = tl.math.exp2(exact_scores - new_max[:, None])
            row_sum = row_sum * alpha + tl.sum(probability, axis=1)
            v8 = tl.load(v8_ptr + src, mask=valid_k[:, None], other=0).to(tl.float32)
            vs = tl.load(vs_ptr + (batch * TK + k_rows) * H + head, mask=valid_k, other=1.0)
            vv = (v8 * vs[:, None]).to(tl.bfloat16)
            output = output * alpha[:, None] + tl.dot(probability.to(tl.bfloat16), vv)
            row_max = new_max

    if COLLECT_TELEMETRY:
        tbase = ((batch * NBQ + q_block) * H + head) * 6
        tl.store(telemetry_ptr + tbase + 0, exact_total)
        tl.store(telemetry_ptr + tbase + 1, approx_total)
        tl.store(telemetry_ptr + tbase + 2, score_route_total)
        tl.store(telemetry_ptr + tbase + 3, local_force_total)
        tl.store(telemetry_ptr + tbase + 4, sink_force_total)
        tl.store(telemetry_ptr + tbase + 5, q_in_sink.to(tl.int32))
    tl.store(
        out_ptr + ((batch * TQ + q_rows[:, None]) * H + head) * D + d[None, :],
        (output / row_sum[:, None]).to(tl.bfloat16),
        mask=valid_q[:, None],
    )


def allocate_compressed_kv_sm120(batch, tokens, heads, head_dim, device):
    if int(head_dim) != HEAD_DIM:
        raise ValueError('compressed Sol KV requires head_dim=128')
    blocks = triton.cdiv(int(tokens), BLOCK_SIZE)
    shape = (int(batch), int(tokens), int(heads), int(head_dim))
    kc = torch.empty((int(batch), blocks, int(heads), int(head_dim)), device=device, dtype=torch.bfloat16)
    vc = torch.empty_like(kc)
    k8 = torch.empty(shape, device=device, dtype=torch.int8)
    v8 = torch.empty(shape, device=device, dtype=torch.int8)
    ks = torch.empty((int(batch), int(tokens), int(heads)), device=device, dtype=torch.float32)
    vs = torch.empty_like(ks)
    return {'kc': kc, 'vc': vc, 'k8': k8, 'v8': v8, 'ks': ks, 'vs': vs,
            'tokens': int(tokens), 'heads': int(heads), 'head_dim': int(head_dim), 'blocks': blocks}


def append_compressed_kv_sm120(storage, k, v, start):
    if k.ndim != 4 or v.shape != k.shape or k.shape[-1] != HEAD_DIM:
        raise ValueError('compressed Sol K/V chunk must be matching [B,T,H,128]')
    if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError('compressed Sol K/V chunk requires bfloat16')
    start = int(start)
    if start % BLOCK_SIZE:
        raise ValueError('compressed Sol KV chunk start must be 64-token aligned')
    B, TC, H, D = k.shape
    if B != 1 or H != storage['heads'] or D != storage['head_dim']:
        raise ValueError('compressed Sol K/V chunk geometry mismatch')
    local_blocks = triton.cdiv(TC, BLOCK_SIZE)
    _compress_kv_chunk_kernel[(local_blocks, B * H)](
        k, v, storage['kc'], storage['vc'], storage['k8'], storage['ks'], storage['v8'], storage['vs'],
        TC, storage['tokens'], start,
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        H, storage['blocks'], D, BLOCK_SIZE,
        num_warps=4,
    )


def finalize_compressed_kv_sm120(storage):
    kc = storage['kc']
    B, NB, H, D = kc.shape
    kc_mean = storage.get('kc_mean')
    kc_var = storage.get('kc_var')
    expected = (B, H, D)
    if (
        kc_mean is None
        or tuple(kc_mean.shape) != expected
        or kc_mean.device != kc.device
        or kc_mean.dtype != torch.float32
    ):
        kc_mean = torch.empty(expected, device=kc.device, dtype=torch.float32)
        storage['kc_mean'] = kc_mean
    if (
        kc_var is None
        or tuple(kc_var.shape) != expected
        or kc_var.device != kc.device
        or kc_var.dtype != torch.float32
    ):
        kc_var = torch.empty(expected, device=kc.device, dtype=torch.float32)
        storage['kc_var'] = kc_var
    _kc_stats_kernel[(B * H,)](kc, kc_mean, kc_var, H, NB, D, num_warps=4)
    return storage


def sol_attn_query_compressed_sm120(q, storage, *, q_offset=0, tau=1.3, scale=None,
                                      sink_blocks=(0, 0), sink_q=(0, 0), telemetry=False):
    if q.ndim != 4 or q.shape[-1] != HEAD_DIM or q.dtype != torch.bfloat16 or q.device.type != 'cuda':
        raise ValueError('compressed streamed Sol Q requires CUDA BF16 [B,Tq,H,128]')
    if int(q_offset) % BLOCK_SIZE:
        raise ValueError('q_offset must be 64-token aligned')
    B, TQ, H, D = q.shape
    TK = int(storage['tokens'])
    NBQ = triton.cdiv(TQ, BLOCK_SIZE)
    NBK = int(storage['blocks'])
    scale = float(D ** -0.5 if scale is None else scale)
    threshold = torch.empty((B, NBQ, H), device=q.device, dtype=torch.float32)
    telemetry_buf = torch.empty((B, NBQ, H, 6), device=q.device, dtype=torch.int32) if telemetry else torch.empty((1,), device=q.device, dtype=torch.int32)
    profile = bool(telemetry)
    if profile:
        import time as _time
        torch.cuda.synchronize()
        _t0 = _time.perf_counter()
    _threshold_kernel[(NBQ, B * H)](
        q, storage['kc_mean'], storage['kc_var'], threshold, scale, float(tau), TQ,
        q.stride(0), q.stride(1), q.stride(2), H, NBQ, D, BLOCK_SIZE,
        num_warps=4,
    )
    if profile:
        torch.cuda.synchronize()
        threshold_s = _time.perf_counter() - _t0
        _t0 = _time.perf_counter()
    else:
        threshold_s = 0.0
    out = torch.empty((B, TQ, H, D), device=q.device, dtype=q.dtype)
    sinks = tuple(int(value) for value in (*sink_blocks, *sink_q))
    _forward_ptr_rect_compressed[(NBQ, B * H)](
        q, storage['kc'], storage['vc'], storage['k8'], storage['ks'], storage['v8'], storage['vs'],
        threshold, out, telemetry_buf, scale, *sinks, TQ, TK, int(q_offset) // BLOCK_SIZE,
        q.stride(0), q.stride(1), q.stride(2),
        H, D, NBQ, NBK, BLOCK_SIZE, GROUP_SIZE, COLLECT_TELEMETRY=bool(telemetry),
        num_warps=8, num_stages=2,
    )
    if not profile:
        return out
    torch.cuda.synchronize()
    forward_s = _time.perf_counter() - _t0
    # Tiny telemetry copy: [query blocks, heads, six counters].
    t = telemetry_buf.detach().cpu()
    thr = threshold.detach()
    stats = {
        'threshold_s': float(threshold_s),
        'forward_s': float(forward_s),
        'threshold_min': float(thr.min().item()),
        'threshold_mean': float(thr.mean().item()),
        'threshold_max': float(thr.max().item()),
        'exact': int(t[..., 0].sum().item()),
        'approx': int(t[..., 1].sum().item()),
        'score_routed': int(t[..., 2].sum().item()),
        'local_forced': int(t[..., 3].sum().item()),
        'sink_forced': int(t[..., 4].sum().item()),
        'q_sink_programs': int(t[..., 5].sum().item()),
        'programs': int(B * NBQ * H),
        'nbq': int(NBQ),
        'nbk': int(NBK),
    }
    denom = max(1, stats['exact'] + stats['approx'])
    stats['exact_ratio'] = float(stats['exact']) / float(denom)
    # Distribution of per-(query-block,head) exact fractions.
    ratios = t[..., 0].float() / max(1, NBK)
    stats['exact_ratio_min'] = float(ratios.min().item())
    stats['exact_ratio_mean'] = float(ratios.mean().item())
    stats['exact_ratio_max'] = float(ratios.max().item())
    return out, stats
