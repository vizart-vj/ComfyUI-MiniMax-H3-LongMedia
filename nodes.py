"""ComfyUI nodes for direct MiniMax H3 audio/video latent control."""

from __future__ import annotations
import copy
import builtins
import gc
import json
import math
import os
import re
import sys
import time
import uuid as _uuid_mod
import importlib.metadata as _importlib_metadata


def _pkg_version_tuple(name: str):
    try:
        raw = _importlib_metadata.version(name)
    except Exception:
        return None, None
    nums = []
    for part in raw.split('.'):
        digits = ''.join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        nums.append(int(digits))
    while len(nums) < 3:
        nums.append(0)
    return raw, tuple(nums[:3])

import torch
import torchaudio
import comfy.nested_tensor
import comfy.model_management

try:
    import comfy.model_prefetch as _comfy_model_prefetch
except Exception:
    _comfy_model_prefetch = None

try:
    import comfy_aimdo as _comfy_aimdo
except Exception:
    _comfy_aimdo = None

try:
    from .latent_ops import (
        AUDIO_LATENT_FPS, FPS, _fit_stream, _validate_audio, _validate_video,
        align_frame_count, apply_video_inpaint_mask, audio_latent_t,
        describe_av, frame_count_from_video_t, inject_leading_video_frame,
        merge_av_latents, pack_av_latents, prepare_continuation, replace_stream,
        set_stream_denoise, split_av_latent, stitch_continuation,
        unpack_av_samples, video_latent_t,
    )
except ImportError:
    from latent_ops import (
        AUDIO_LATENT_FPS, FPS, _fit_stream, _validate_audio, _validate_video,
        align_frame_count, apply_video_inpaint_mask, audio_latent_t,
        describe_av, frame_count_from_video_t, inject_leading_video_frame,
        merge_av_latents, pack_av_latents, prepare_continuation, replace_stream,
        set_stream_denoise, split_av_latent, stitch_continuation,
        unpack_av_samples, video_latent_t,
    )

from dataclasses import replace as _dc_replace

try:
    from .media_plan import (
        build_media_plan, LongMediaPlan, slice_video_segment,
        slice_audio_segment, collect_numbered_inputs,
    )
except ImportError:
    from media_plan import (
        build_media_plan, LongMediaPlan, slice_video_segment,
        slice_audio_segment, collect_numbered_inputs,
    )

try:
    from .temporal_positioning import (
        TEMPORAL_OFFSET_OPTION, temporal_offset_for_frame, h3_temporal_offset_wrapper,
    )
except ImportError:
    from temporal_positioning import (
        TEMPORAL_OFFSET_OPTION, temporal_offset_for_frame, h3_temporal_offset_wrapper,
    )

try:
    from .continuity_policy import (
        build_segment_prompt as _policy_build_segment_prompt,
        normalize_hybrid_picture_tags,
    )
except ImportError:
    from continuity_policy import (
        build_segment_prompt as _policy_build_segment_prompt,
        normalize_hybrid_picture_tags,
    )

try:
    from .refine_policy import split_refine_sigmas
except ImportError:
    from refine_policy import split_refine_sigmas

try:
    from comfy.patcher_extension import WrappersMP
except Exception:  # pragma: no cover - only available inside ComfyUI
    WrappersMP = None

CATEGORY = 'MiniMax H3/LongMedia'
CATEGORY_STREAMS = f'{CATEGORY}/Streams'
CATEGORY_CONTINUATION = f'{CATEGORY}/Continuation'
CATEGORY_LONGMEDIA = f'{CATEGORY}/LongMedia'
CATEGORY_UTIL = f'{CATEGORY}/Utility'
MAX_RESOLUTION = 16384
CANVAS_MULTIPLE = 32
NestedTensor = comfy.nested_tensor.NestedTensor

NativeReferenceToVideo = None


def _clone_model_options_safe(model_options):
    """Clone ComfyUI model_options without deep-copying tensor/AIMDO storage."""
    source = model_options or {}
    try:
        import comfy.model_patcher
        return comfy.model_patcher.create_model_options_clone(source)
    except Exception:
        # Last-resort structural copy. Never deepcopy tensor-backed storage.
        cloned = dict(source)
        if isinstance(source.get("transformer_options"), dict):
            cloned["transformer_options"] = dict(source["transformer_options"])
        return cloned


def _reconstruction_profile_settings(profile: str) -> dict[str, float]:
    """V4 frame-aligned reconstruction guide policy."""
    profile = str(profile or 'balanced')
    presets = {
        'conservative': {'lowpass_scale': 1.0, 'guide_aug_min': 0.998, 'guide_aug_max': 0.995},
        'balanced': {'lowpass_scale': 0.94, 'guide_aug_min': 0.997, 'guide_aug_max': 0.985},
        'neural_remaster': {'lowpass_scale': 0.90, 'guide_aug_min': 0.994, 'guide_aug_max': 0.970},
    }
    return dict(presets.get(profile, presets['balanced']))


def _reconstruction_preprocess_frames(frames: torch.Tensor, profile: str) -> torch.Tensor:
    """Mildly suppress source ringing while preserving geometry and motion."""
    settings = _reconstruction_profile_settings(profile)
    scale = float(settings.get('lowpass_scale', 1.0))
    if frames is None or scale >= 0.999:
        return frames
    if not hasattr(frames, 'shape') or len(frames.shape) != 4:
        return frames
    frame_bchw = frames.movedim(-1, 1)
    height = int(frame_bchw.shape[-2]); width = int(frame_bchw.shape[-1])
    down_h = max(2, min(height, int(round(height * scale))))
    down_w = max(2, min(width, int(round(width * scale))))
    if down_h == height and down_w == width:
        return frames
    low = torch.nn.functional.interpolate(frame_bchw, size=(down_h, down_w), mode='bilinear', align_corners=False, antialias=True)
    restored = torch.nn.functional.interpolate(low, size=(height, width), mode='bicubic', align_corners=False, antialias=True)
    return restored.movedim(1, -1).contiguous()


def _reconstruction_visual_aug(strength: float, profile: str) -> float:
    """Map UI strength to native H3 visual condition noise augmentation."""
    settings = _reconstruction_profile_settings(profile)
    strength = max(0.0, min(1.0, float(strength)))
    a0 = float(settings.get('guide_aug_min', 0.997)); a1 = float(settings.get('guide_aug_max', 0.985))
    return max(0.0, min(1.0, a0 + (a1 - a0) * strength))


def _reconstruction_video_mask_value(strength: float, profile: str) -> float:
    del strength, profile
    return 1.0


def _reconstruction_apply_source_authority(video_latent: torch.Tensor, strength: float, profile: str) -> torch.Tensor:
    del strength, profile
    return video_latent


def _reconstruction_fit_source_frames(frames: torch.Tensor, width: int, height: int, resize_mode: str) -> torch.Tensor:
    """Apply the authoritative reconstruction source-fit transform to target aspect.

    Ref2VA later downsizes the fitted clip to its own reference canvas, but every
    segment first sees the exact same crop/stretch policy so composition does not
    jump at LongMedia boundaries.
    """
    mode = str(resize_mode or 'center_crop')
    if mode == 'none':
        if int(frames.shape[2]) != int(width) or int(frames.shape[1]) != int(height):
            raise ValueError(
                f'Reconstruction source_fit=strict requires {int(width)}x{int(height)}, '
                f'got {int(frames.shape[2])}x{int(frames.shape[1])}.'
            )
        return frames
    return _resize_frames(frames, int(width), int(height), mode)


def _reconstruction_set_ref_strength(positive, strength: float, profile: str):
    """Set native H3 reference-conditioning noise augmentation for reconstruction."""
    try:
        import node_helpers
        return node_helpers.conditioning_set_values(
            positive,
            {'minimax_visual_cond_noise_aug': _reconstruction_visual_aug(strength, profile)},
        )
    except Exception as exc:
        raise RuntimeError(f'Could not apply reconstruction Ref2VA strength: {exc}') from exc


def _reconstruction_detail_sigmas(full_sigmas, steps: int, strength: float):
    """Build a short detail-focused mid-sigma schedule.

    V2 intentionally starts slightly earlier than the original detail layer.  The
    two-pass reconstruction already provides stable geometry, so the detail pass
    can safely revisit a somewhat noisier suffix and use its limited freedom to
    paint facial features, clothing edges and local contrast.  The subsequent
    merge transfers only bounded spatial detail bands back into the stable movie.
    """
    if not torch.is_tensor(full_sigmas):
        full_sigmas = torch.as_tensor(full_sigmas, dtype=torch.float32)
    sig = full_sigmas.detach().float().cpu().flatten()
    intervals = max(0, int(sig.numel()) - 1)
    if intervals < 1:
        return None, []
    steps = max(1, min(int(steps), intervals))
    strength = max(0.0, min(1.0, float(strength)))
    base_span = min(intervals, steps + 1)
    extra_span = int(round((0.35 + 0.65 * strength) * max(0, intervals - base_span)))
    span = max(steps, min(intervals, base_span + extra_span))
    start = max(0, intervals - span)
    raw = torch.linspace(float(start), float(intervals), steps + 1)
    idx = [int(round(x)) for x in raw.tolist()]
    idx[0] = start
    idx[-1] = intervals
    uniq = []
    for i in idx:
        i = max(start, min(intervals, int(i)))
        if not uniq or i > uniq[-1]:
            uniq.append(i)
    if len(uniq) < 2:
        uniq = [max(0, intervals - 1), intervals]
    out = sig[uniq].clone()
    return out, uniq


def _reconstruction_micro_detail_sigmas(full_sigmas, steps: int, strength: float):
    """Very-low-sigma suffix used only for microtexture synthesis.

    This pass intentionally stays close to x0 so it cannot meaningfully move
    geometry.  It exists to give H3 a second, independent chance to paint tiny
    texture/edge information after the broader structure-detail pass.
    """
    if not torch.is_tensor(full_sigmas):
        full_sigmas = torch.as_tensor(full_sigmas, dtype=torch.float32)
    sig = full_sigmas.detach().float().cpu().flatten()
    intervals = max(0, int(sig.numel()) - 1)
    if intervals < 1:
        return None, []
    steps = max(2, min(int(steps), intervals))
    strength = max(0.0, min(1.0, float(strength)))
    # Keep this pass in the final 2-4 intervals.  Strength only nudges how far
    # upward it reaches; the structure pass already handles medium frequencies.
    span = min(intervals, max(2, int(round(2 + 2 * strength))))
    start = max(0, intervals - span)
    raw = torch.linspace(float(start), float(intervals), steps + 1)
    idx = [int(round(x)) for x in raw.tolist()]
    idx[0] = start
    idx[-1] = intervals
    uniq = []
    for i in idx:
        i = max(start, min(intervals, int(i)))
        if not uniq or i > uniq[-1]:
            uniq.append(i)
    if len(uniq) < 2:
        uniq = [max(0, intervals - 1), intervals]
    return sig[uniq].clone(), uniq


def _nearest_valid_h3_frame_count(length: int) -> int:
    """Snap to the nearest MiniMax H3 ``17*k+5`` frame count.

    Loop closure is an authored tail window, so silently expanding a request by
    almost a full stride is undesirable. Ties prefer the lower valid value to
    preserve more of the original ending.
    """
    requested = max(5, int(length))
    k = max(0, (requested - 5) // 17)
    lower = 5 + 17 * k
    upper = lower + 17
    if requested - lower <= upper - requested:
        return lower
    return upper


def _loop_closure_sigmas(full_sigmas, steps: int = 4, strength: float = 0.65):
    """Compact low/mid-sigma schedule for context-preserving loop closure.

    Loop Strength controls how much structural freedom the closure pass receives.
    It intentionally does *not* start near sigma_max: a loop should steer the
    existing motion back toward the opening context, not regenerate the whole
    ending or create an acceleration ramp while chasing an exact terminal latent.
    """
    if not torch.is_tensor(full_sigmas):
        full_sigmas = torch.as_tensor(full_sigmas, dtype=torch.float32)
    sig = full_sigmas.detach().float().cpu().flatten()
    intervals = max(0, int(sig.numel()) - 1)
    if intervals < 1:
        return None, []
    strength = max(0.0, min(1.0, float(strength)))
    steps = max(2, min(int(steps), intervals))
    # At the default 0.65 this starts around the lower-middle part of a normal
    # schedule. Higher strength gives H3 more geometry freedom; lower strength
    # keeps the pass increasingly close to the already generated tail.
    start_fraction = 0.18 + 0.42 * (1.0 - strength)
    start = int(round(intervals * start_fraction))
    start = max(1 if intervals >= 3 else 0, min(intervals - 1, start))
    raw = torch.linspace(float(start), float(intervals), steps + 1)
    idx = [int(round(x)) for x in raw.tolist()]
    idx[0] = start
    idx[-1] = intervals
    uniq = []
    for i in idx:
        i = max(start, min(intervals, int(i)))
        if not uniq or i > uniq[-1]:
            uniq.append(i)
    if len(uniq) < 2:
        uniq = [max(0, intervals - 1), intervals]
    return sig[uniq].clone(), uniq


def _loop_structural_anchor(
    opening_anchor: torch.Tensor,
    current_tail_end: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Build a macro-context terminal anchor without copying opening microdetail.

    The opening frame contributes only low-frequency spatial structure. Fine
    latent detail remains that of the generated tail, so H3 is free to preserve
    motion diversity, brush strokes, water texture, hair, cloth, etc.
    """
    if tuple(opening_anchor.shape) != tuple(current_tail_end.shape):
        raise ValueError(
            f'Loop structural anchor shape mismatch: {tuple(opening_anchor.shape)} vs {tuple(current_tail_end.shape)}'
        )
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0:
        return current_tail_end.clone()
    import torch.nn.functional as F

    def lowpass(x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        # 5x5 is intentionally broad enough to represent composition/geometry
        # rather than pixel/texture identity. Fall back gracefully on tiny maps.
        kernel = 5 if min(h, w) >= 5 else (3 if min(h, w) >= 3 else 1)
        if kernel == 1:
            return x
        y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        y = F.avg_pool2d(y, kernel_size=kernel, stride=1, padding=kernel // 2)
        return y.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)

    opening_macro = lowpass(opening_anchor.float())
    tail_macro = lowpass(current_tail_end.float())
    macro_delta = (opening_macro - tail_macro) * strength
    return (current_tail_end.float() + macro_delta).to(
        device=current_tail_end.device, dtype=current_tail_end.dtype
    )


def _apply_loop_macro_return(
    generated_video: torch.Tensor,
    opening_anchor: torch.Tensor,
    strength: float,
) -> tuple[torch.Tensor, float]:
    """Return the generated tail toward the opening *macro* state only.

    This is deliberately not an RGB crossfade and not an exact latent endpoint
    lock.  H3 remains the motion/detail author.  After sampling we progressively
    correct only low spatial frequencies, leaving the high-frequency residual
    (brush strokes, water texture, hair, cloth, local deformation) untouched.

    The correction starts late and reaches its maximum only at the last latent
    step, so the first closure latent remains exact and the model never has to
    accelerate its own trajectory in order to "catch" frame zero.
    """
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0:
        return generated_video, 0.0
    if generated_video.ndim != 5 or opening_anchor.ndim != 5:
        raise ValueError('Loop macro return expects B,C,T,H,W video latents.')
    if generated_video.shape[0] != opening_anchor.shape[0] or generated_video.shape[1] != opening_anchor.shape[1]:
        raise ValueError('Loop macro return batch/channel mismatch.')
    if generated_video.shape[-2:] != opening_anchor.shape[-2:]:
        raise ValueError('Loop macro return spatial geometry mismatch.')

    import torch.nn.functional as F

    def lowpass(x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        kernel = 7 if min(h, w) >= 7 else (5 if min(h, w) >= 5 else (3 if min(h, w) >= 3 else 1))
        if kernel == 1:
            return x
        y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        y = F.avg_pool2d(y, kernel_size=kernel, stride=1, padding=kernel // 2)
        return y.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)

    work = generated_video.float()
    opening = opening_anchor[:, :, :1].float()
    opening_macro = lowpass(opening)
    current_macro = lowpass(work)

    # Perceptual response: 0.65 should already be a strong loop, while 1.0
    # becomes an exact macro-state return.  Zero remains a true no-op.
    effective = 1.0 - (1.0 - strength) ** 1.8

    t = int(work.shape[2])
    if t <= 1:
        return generated_video, float(effective)
    phase = torch.linspace(0.0, 1.0, t, device=work.device, dtype=work.dtype)
    # Leave the first quarter of the closure untouched, then ease the structural
    # correction in smoothly.  This avoids a new velocity impulse at the entry.
    u = ((phase - 0.25) / 0.75).clamp(0.0, 1.0)
    ramp = u * u * (3.0 - 2.0 * u)
    ramp = (ramp * effective).view(1, 1, t, 1, 1)

    macro_delta = opening_macro.expand(-1, -1, t, -1, -1) - current_macro
    corrected = work + macro_delta * ramp
    # Guarantee exact continuity at the closure entry.
    corrected[:, :, :1] = work[:, :, :1]
    return corrected.to(device=generated_video.device, dtype=generated_video.dtype), float(effective)


def _reconstruction_merge_detail_residual(
    base_video: torch.Tensor,
    structure_video: torch.Tensor,
    strength: float,
    texture_video: torch.Tensor | None = None,
) -> torch.Tensor:
    """Detail Recovery V3: dual-candidate, multiband, temporally stable merge.

    ``structure_video`` comes from a slightly broader mid-sigma trajectory and
    contributes facial/clothing/edge structure. ``texture_video`` comes from a
    very-low-sigma independent trajectory and contributes only microtexture.
    The stable two-pass reconstruction remains the low-frequency authority.

    V3 also applies a small bounded amplification to detail already present in
    the base latent.  This addresses the observed case where the latent is dense
    enough but the decoded result remains optically soft.
    """
    if tuple(base_video.shape) != tuple(structure_video.shape):
        raise ValueError(
            f'Reconstruction detail layer changed latent geometry: {tuple(base_video.shape)} -> {tuple(structure_video.shape)}'
        )
    if texture_video is not None and tuple(base_video.shape) != tuple(texture_video.shape):
        raise ValueError(
            f'Reconstruction micro-detail layer changed latent geometry: {tuple(base_video.shape)} -> {tuple(texture_video.shape)}'
        )
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0:
        return base_video
    b, c, t, h, w = base_video.shape
    if h < 3 or w < 3:
        return base_video
    if texture_video is None:
        texture_video = structure_video

    def _blur(x: torch.Tensor, kernel: int) -> torch.Tensor:
        pad = max(0, int(kernel) // 2)
        y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
        y = torch.nn.functional.avg_pool2d(
            y, kernel_size=int(kernel), stride=1, padding=pad, count_include_pad=False
        )
        return y.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).to(dtype=x.dtype)

    def _temporal_stabilize(delta: torch.Tensor, amount: float) -> torch.Tensor:
        if int(delta.shape[2]) < 3 or amount <= 0.0:
            return delta
        prev = torch.cat((delta[:, :, :1], delta[:, :, :-1]), dim=2)
        nxt = torch.cat((delta[:, :, 1:], delta[:, :, -1:]), dim=2)
        smooth = 0.25 * prev + 0.50 * delta + 0.25 * nxt
        return delta * (1.0 - amount) + smooth * amount

    base_b3 = _blur(base_video, 3)
    base_b7 = _blur(base_video, 7)
    struct_b3 = _blur(structure_video, 3)
    struct_b7 = _blur(structure_video, 7)
    tex_b3 = _blur(texture_video, 3)

    base_hi = base_video - base_b3
    base_mid = base_b3 - base_b7
    struct_hi = structure_video - struct_b3
    struct_mid = struct_b3 - struct_b7
    tex_hi = texture_video - tex_b3

    delta_mid = _temporal_stabilize(struct_mid - base_mid, 0.32)
    # Use the low-sigma candidate for microtexture. A small structure-candidate
    # contribution prevents texture-only speckle from dominating fine edges.
    delta_hi = 0.72 * (tex_hi - base_hi) + 0.28 * (struct_hi - base_hi)
    delta_hi = _temporal_stabilize(delta_hi, 0.18)

    eps = 1e-6
    base_edge = base_mid.abs() + 0.50 * base_hi.abs()
    new_edge = struct_mid.abs() + 0.35 * struct_hi.abs() + 0.25 * tex_hi.abs()
    edge_ratio = new_edge / (base_edge + new_edge + eps)
    confidence = (0.34 + 0.66 * edge_ratio).clamp_(0.22, 1.00)

    scale = base_video.float().std(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-4).to(base_video.dtype)
    mid_limit = scale * (0.22 + 0.36 * strength)
    hi_limit = scale * (0.14 + 0.26 * strength)
    delta_mid = mid_limit * torch.tanh(delta_mid / mid_limit.clamp_min(1e-6))
    delta_hi = hi_limit * torch.tanh(delta_hi / hi_limit.clamp_min(1e-6))

    # Expose information already present in the base latent. Gains are deliberately
    # modest and bounded; they sharpen decode without changing scene semantics.
    native_mid_boost = base_mid * (0.08 + 0.16 * strength)
    native_hi_boost = base_hi * (0.025 + 0.075 * strength)

    gain_mid = 0.68 + 0.92 * strength
    gain_hi = 0.40 + 0.70 * strength
    merged = base_video + confidence * (
        delta_mid * gain_mid + delta_hi * gain_hi + native_mid_boost + native_hi_boost
    )

    # Absolute low-frequency lock. Only detail bands survive from the two detail
    # trajectories; camera, staging, body placement and broad luminance remain
    # exactly owned by the stable two-pass reconstruction.
    merged_low = _blur(merged, 7)
    merged = merged - merged_low + base_b7
    return merged.to(dtype=base_video.dtype)


def _reconstruction_encode_frame_aligned_guide(vae, frames: torch.Tensor, target_video: torch.Tensor, resize_mode: str, profile: str) -> torch.Tensor:
    """Encode source to exactly the target H3 video latent H/W/T geometry."""
    frames = _reconstruction_preprocess_frames(frames, profile)
    target_frames = frame_count_from_video_t(int(target_video.shape[2]))
    target_audio_stub = torch.zeros((1, 32, 2, audio_latent_t(target_frames)), dtype=torch.float32)
    target_av = {'samples': NestedTensor((target_video, target_audio_stub))}
    encoded = MiniMaxH3LatentLabVideoEncode().encode(vae, frames, 'strict', resize_mode, target_av)[0]['samples']
    if tuple(encoded.shape) != tuple(target_video.shape):
        raise RuntimeError(f'Reconstruction V4 guide geometry mismatch: guide={tuple(encoded.shape)} target={tuple(target_video.shape)}')
    return encoded


def _reconstruction_attach_frame_aligned_guide(positive, guide_latent: torch.Tensor, frame_count: int, visual_aug: float):
    """Attach a frame-aligned V4 guide as one-latent-token FL2VA anchors.

    Stock H3 supports a multi-frame clip in one keyframe, but process-global
    PackedLayout wrappers from other extensions can legally wrap/replace the
    initializer and some of them collapse that clip metadata to one temporal
    token. Splitting the guide into one-token keyframes is mathematically
    equivalent to H3's native clip grid and composition-safe: every metadata
    block really has T=1, while the full temporal sequence is preserved by the
    resolved pixel-frame positions.
    """
    if guide_latent is None or not hasattr(guide_latent, 'shape') or len(guide_latent.shape) != 5:
        raise ValueError('Reconstruction V4 guide must be [B,C,T,H,W].')
    latent_t = int(guide_latent.shape[2])
    if latent_t <= 0:
        raise ValueError('Reconstruction V4 guide contains no latent timesteps.')

    try:
        from comfy.ldm.minimax.model import FRAME_PER_TOKEN
        frame_pattern = tuple(int(v) for v in FRAME_PER_TOKEN)
    except Exception:
        frame_pattern = (1, 4, 4, 4, 4)

    keyframes = []
    pixel_frame = 0
    for token_index in range(latent_t):
        if pixel_frame >= int(frame_count):
            raise RuntimeError(
                f'Reconstruction V4 token/frame contract overflow: token={token_index} '
                f'frame={pixel_frame} frame_count={int(frame_count)}.'
            )
        token_latent = guide_latent[:, :, token_index:token_index + 1].contiguous()
        keyframes.append({
            'resolved_frame_index': int(pixel_frame),
            'latent': token_latent,
            'longmedia_reconstruction_v4_token': int(token_index),
        })
        pixel_frame += frame_pattern[token_index % len(frame_pattern)]

    # The native temporal mapping must cover exactly the target pixel timeline.
    # For valid H3 17*k+5 clips, sum(FRAME_PER_TOKEN over latent T) == frame_count.
    if int(pixel_frame) != int(frame_count):
        raise RuntimeError(
            f'Reconstruction V4 token/frame contract mismatch: latent_t={latent_t} '
            f'mapped_frames={pixel_frame} target_frames={int(frame_count)}.'
        )

    def patch_meta(meta):
        meta = dict(meta)
        meta.pop('minimax_refs', None)
        meta['minimax_keyframes'] = keyframes
        meta['minimax_frame_count'] = int(frame_count)
        meta['minimax_visual_cond_noise_aug'] = float(visual_aug)
        meta['longmedia_reconstruction_v4'] = True
        meta['longmedia_reconstruction_v4_tokenized_guide'] = True
        return meta

    out = []
    attached = False
    for entry in (positive or []):
        if isinstance(entry, dict):
            out.append(patch_meta(entry)); attached = True
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], dict):
            ne = list(entry); ne[1] = patch_meta(entry[1]); out.append(ne); attached = True
        else:
            out.append(entry)
    if not attached:
        raise RuntimeError('Reconstruction V4 could not attach FL2VA guide to conditioning metadata.')
    return out


# Release console policy.  release_guard=True keeps production output quiet;
# release_guard=False restores the full development console so profiling and
# execution-path diagnostics are visible without maintaining a separate build.
_LONGMEDIA_ALWAYS_CONSOLE = ("cuda oom", "[error]", "exception", "terminal failure", "failed to")
_LONGMEDIA_RELEASE_GUARD = True

def _set_longmedia_release_guard(enabled: bool) -> bool:
    global _LONGMEDIA_RELEASE_GUARD
    _LONGMEDIA_RELEASE_GUARD = bool(enabled)
    return _LONGMEDIA_RELEASE_GUARD

def _lm_print(*args, **kwargs):
    if not _LONGMEDIA_RELEASE_GUARD:
        return builtins.print(*args, **kwargs)
    text = " ".join(str(a) for a in args).lower()
    if any(marker in text for marker in _LONGMEDIA_ALWAYS_CONSOLE):
        return builtins.print(*args, **kwargs)
    return None



_RAM_FILECACHE_PREWARM_SEEN = set()

def _iter_quant_tensor_leaves(value, _seen=None):
    """Yield real tensor leaves from normal or Comfy QuantizedTensor values."""
    if _seen is None:
        _seen = set()
    oid = id(value)
    if oid in _seen:
        return
    _seen.add(oid)
    if torch.is_tensor(value):
        # QuantizedTensor is a Tensor subclass; expose its backing tensors too.
        try:
            names, _ctx = value.__tensor_flatten__()
        except Exception:
            names = None
        if names:
            for name in names:
                try:
                    child = getattr(value, name)
                except Exception:
                    continue
                yield from _iter_quant_tensor_leaves(child, _seen)
        else:
            yield value
        return
    if isinstance(value, dict):
        for child in value.values():
            yield from _iter_quant_tensor_leaves(child, _seen)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _iter_quant_tensor_leaves(child, _seen)


def _collect_h3_mmap_payloads(model_patcher):
    """Return unique AIMDO mmap payload memoryviews backing this model."""
    model = getattr(model_patcher, 'model', None)
    if model is None:
        return []
    try:
        sd = model.state_dict()
    except Exception:
        return []
    payloads = []
    seen_maps = set()
    for value in sd.values():
        for tensor in _iter_quant_tensor_leaves(value):
            try:
                storage = tensor.untyped_storage()
                refs = getattr(storage, '_comfy_tensor_mmap_refs', None)
            except Exception:
                refs = None
            if not refs or len(refs) < 2:
                continue
            mmap_obj, payload = refs[0], refs[1]
            key = id(mmap_obj)
            if key in seen_maps:
                continue
            seen_maps.add(key)
            try:
                nbytes = int(payload.nbytes)
            except Exception:
                try:
                    nbytes = len(payload)
                except Exception:
                    nbytes = 0
            if nbytes > 0:
                payloads.append((mmap_obj, payload, nbytes))
    return payloads


def _prewarm_h3_file_cache(model_patcher, model_size_bytes=0, min_ram_headroom_gb=10.0):
    """Warm AIMDO-backed checkpoint pages into the OS file cache without tensor copies.

    This intentionally changes neither model tensors nor VRAM residency.  On Windows
    it asks the memory manager to prefetch mapped checkpoint pages, then lightly
    touches one byte per 64 KiB region so the request is observable before sampling.
    The warm budget is capped by currently available RAM minus a hard headroom.
    """
    result = {
        'status': 'skipped', 'payloads': 0, 'payload_bytes': 0,
        'budget_bytes': 0, 'touched_bytes': 0, 'seconds': 0.0,
        'reason': None,
    }
    try:
        import psutil
        vm = psutil.virtual_memory()
        available = int(vm.available)
    except Exception as exc:
        result['reason'] = f'RAM query failed: {type(exc).__name__}'
        return result

    hard_headroom = int(float(min_ram_headroom_gb) * (1024 ** 3))
    budget = max(0, available - hard_headroom)
    if budget < 2 * (1024 ** 3):
        result['reason'] = f'available RAM headroom too small ({available/(1024**3):.1f}GB available)'
        return result

    payloads = _collect_h3_mmap_payloads(model_patcher)
    result['payloads'] = len(payloads)
    total_payload = sum(x[2] for x in payloads)
    result['payload_bytes'] = int(total_payload)
    if not payloads:
        result['reason'] = 'no AIMDO mmap payloads found on model tensors'
        return result

    # Never consume the user's entire free RAM.  The cap is deliberately smaller
    # than the mapped model on a 64 GB machine so Windows retains normal working
    # set/pagefile headroom while still caching a large fraction of a 32 GB H3.
    budget = min(budget, total_payload)
    if model_size_bytes:
        budget = min(budget, int(model_size_bytes))
    result['budget_bytes'] = int(budget)

    signature = tuple(sorted((id(mm), nbytes) for mm, _mv, nbytes in payloads))
    if signature in _RAM_FILECACHE_PREWARM_SEEN:
        result['status'] = 'already_warm'
        result['reason'] = 'same AIMDO mappings already prewarmed in this process'
        return result

    t0 = time.perf_counter()
    remaining = int(budget)
    touched = 0
    checksum = 0
    try:
        import numpy as np
        kernel32 = None
        process_handle = None
        range_type = None
        if os.name == 'nt':
            try:
                import ctypes
                from ctypes import wintypes
                class _WIN32_MEMORY_RANGE_ENTRY(ctypes.Structure):
                    _fields_ = [('VirtualAddress', ctypes.c_void_p), ('NumberOfBytes', ctypes.c_size_t)]
                kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
                prefetch = kernel32.PrefetchVirtualMemory
                prefetch.argtypes = [wintypes.HANDLE, ctypes.c_ulong, ctypes.POINTER(_WIN32_MEMORY_RANGE_ENTRY), ctypes.c_ulong]
                prefetch.restype = wintypes.BOOL
                process_handle = kernel32.GetCurrentProcess()
                range_type = _WIN32_MEMORY_RANGE_ENTRY
            except Exception:
                kernel32 = process_handle = range_type = None

        for _mmap_obj, payload, nbytes in payloads:
            if remaining <= 0:
                break
            take = min(int(nbytes), remaining)
            if take <= 0:
                continue
            arr = np.frombuffer(payload, dtype=np.uint8, count=take)
            if kernel32 is not None and process_handle is not None and range_type is not None:
                try:
                    entry = range_type(ctypes.c_void_p(int(arr.ctypes.data)), ctypes.c_size_t(int(take)))
                    kernel32.PrefetchVirtualMemory(process_handle, 1, ctypes.byref(entry), 0)
                except Exception:
                    pass
            # One touch per Windows allocation-granularity-sized region.  AIMDO's
            # mmap remains the owner; this creates no duplicate tensor allocation.
            stride = 64 * 1024
            for start in range(0, take, 512 * 1024 * 1024):
                end = min(take, start + 512 * 1024 * 1024)
                probe = arr[start:end:stride]
                if probe.size:
                    checksum ^= int(probe.sum(dtype=np.uint64)) & 0xFFFFFFFF
            touched += take
            remaining -= take
        _RAM_FILECACHE_PREWARM_SEEN.add(signature)
        result['status'] = 'warmed'
        result['touched_bytes'] = int(touched)
        result['checksum'] = int(checksum)
    except Exception as exc:
        result['status'] = 'failed'
        result['reason'] = f'{type(exc).__name__}: {exc}'
    result['seconds'] = float(time.perf_counter() - t0)
    return result

def _resolve_native_reference_to_video():
    """Lazy-load MiniMaxH3ReferenceToVideo from stock nodes."""
    try:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        return MiniMaxH3ReferenceToVideo
    except ImportError:
        return None


def _is_valid_frame_count(count: int) -> bool:
    return count >= 5 and (count - 5) % 17 == 0


def _fit_frames(frames: torch.Tensor, target_count: int, mode: str) -> torch.Tensor:
    count = frames.shape[0]
    if count == 0:
        raise ValueError('Video input contains no frames.')
    if mode == 'strict':
        if count != target_count:
            raise ValueError(
                f'Strict frame fit requires {target_count} frames, got {count}.'
            )
        return frames
    elif mode == 'crop_or_pad_last':
        if count >= target_count:
            return frames[:target_count]
        else:
            pad = frames[-1:].expand(target_count - count, *frames.shape[1:])
            return torch.cat((frames, pad), dim=0)
    elif mode == 'loop':
        indices = torch.arange(target_count, device=frames.device) % count
        return frames[indices]
    else:
        raise ValueError(f'Unknown frame fit mode: {mode}')


def _resize_frames(frames: torch.Tensor, width: int, height: int, resize_mode: str) -> torch.Tensor:
    frames = frames[..., :3]
    if frames.shape[2] == width and frames.shape[1] == height:
        return frames
    elif resize_mode == 'none':
        raise ValueError(
            f'Input is {frames.shape[2]}x{frames.shape[1]}, '
            f'target is {width}x{height}. Choose stretch or center_crop.'
        )
    else:
        crop = 'disabled' if resize_mode == 'stretch' else 'center'
        import comfy.utils
        samples = frames.movedim(-1, 1)
        samples = comfy.utils.common_upscale(samples, width, height, 'lanczos', crop)
        return samples.movedim(1, -1)


# H3 spatial patchification consumes VAE latents in 2x2 spatial patches.
# With a 16x VAE spatial reduction, pixel geometry therefore needs to stay on
# a 32px grid. More importantly, native Ref2VA `match` grows reference-token
# area with the target canvas; empirically that changes H3 behaviour sharply
# between the long-tested ~0.6 MP regime and ~1 MP. Keep generation resolution
# independent from reference-conditioning resolution: target may be any H3-safe
# 32px canvas, while still-image refs are capped to a stable token budget.
_H3_SAFE_REF_PIXELS = int(round(0.60 * 1024 * 1024))

def _h3_safe_dim(value: int, multiple: int = CANVAS_MULTIPLE) -> int:
    value = max(int(multiple), int(value))
    return max(int(multiple), int(round(value / float(multiple))) * int(multiple))

def _h3_safe_target_canvas(width: int, height: int):
    requested_w, requested_h = int(width), int(height)
    safe_w = _h3_safe_dim(requested_w)
    safe_h = _h3_safe_dim(requested_h)
    return safe_w, safe_h, {
        'requested_width': requested_w, 'requested_height': requested_h,
        'safe_width': safe_w, 'safe_height': safe_h,
        'changed': bool((safe_w, safe_h) != (requested_w, requested_h)),
        'pixel_multiple': int(CANVAS_MULTIPLE),
    }

def _h3_safe_reference_image(image: torch.Tensor, max_pixels: int = _H3_SAFE_REF_PIXELS):
    if image is None:
        return None, None
    h, w = int(image.shape[1]), int(image.shape[2])
    area = max(1, w * h)
    scale = min(1.0, math.sqrt(float(max_pixels) / float(area)))
    # Floor after scaling so rounding can never push the reference back over
    # the conditioning budget. A 32px grid guarantees even H/16 and W/16.
    tw = max(CANVAS_MULTIPLE, int(math.floor((w * scale) / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)
    th = max(CANVAS_MULTIPLE, int(math.floor((h * scale) / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)
    if tw == w and th == h:
        resized = image[..., :3]
    else:
        resized = _resize_frames(image, tw, th, 'stretch')
    return resized, {
        'source_width': w, 'source_height': h,
        'safe_width': tw, 'safe_height': th,
        'source_pixels': area, 'safe_pixels': tw * th,
        'pixel_budget': int(max_pixels),
        'latent_width': tw // 16, 'latent_height': th // 16,
        'patch_width': (tw // 16) // 2, 'patch_height': (th // 16) // 2,
    }

def _h3_safe_reference_images(images, max_pixels: int = _H3_SAFE_REF_PIXELS):
    prepared, reports = [], []
    for image in images or []:
        if image is None:
            continue
        safe, report = _h3_safe_reference_image(image, max_pixels=max_pixels)
        prepared.append(safe)
        reports.append(report)
    return prepared, reports


def _fit_waveform(waveform: torch.Tensor, samples: int, mode: str) -> torch.Tensor:
    current = waveform.shape[-1]
    if current == 0:
        raise ValueError('Audio input contains no samples.')
    if mode == 'strict':
        return waveform
    elif mode == 'crop_or_pad_silence':
        if current >= samples:
            return waveform[..., :samples]
        else:
            return torch.nn.functional.pad(waveform, (0, samples - current))
    elif mode == 'loop':
        repeats = math.ceil(samples / current)
        return (waveform.repeat(1, 1, repeats))[..., :samples]
    else:
        raise ValueError(f'Unknown audio fit mode: {mode}')


def _target_video_geometry(target_av):
    """Extract (video_tensor, frame_count, width, height) from a target AV latent."""
    video, _ = unpack_av_samples(target_av)
    frames = frame_count_from_video_t(video.shape[2])
    width = video.shape[4] * 16
    height = video.shape[3] * 16
    return video, frames, width, height


def _free_cuda_memory():
    """Force a Python GC pass and let the CUDA allocator release cached blocks.

    This doesn't unload anything by itself; it just gives back memory that
    Python/PyTorch would otherwise keep cached for reuse, which matters right
    after a large tensor (e.g. a stitched multi-pass accumulator) is offloaded
    or freed, so the freed VRAM is actually available to the next pass.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _release_model_memory_for_decode():
    """Create a hard memory boundary before final VAE decode.

    Sampling can leave the diffusion model and several transient CUDA pools
    resident.  A long video VAE decode should not compete with those weights.
    This helper is deliberately best-effort/fail-open: decode still proceeds if
    a particular ComfyUI build does not expose one of the cleanup helpers.
    """
    before = _cuda_memory_snapshot()
    errors = []
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as exc:
        errors.append(f'sync_before: {type(exc).__name__}: {exc}')
    try:
        comfy.model_management.unload_all_models()
    except Exception as exc:
        errors.append(f'unload_all_models: {type(exc).__name__}: {exc}')
    try:
        gc.collect()
        try:
            comfy.model_management.soft_empty_cache(force=True)
        except TypeError:
            comfy.model_management.soft_empty_cache()
    except Exception as exc:
        errors.append(f'soft_empty_cache: {type(exc).__name__}: {exc}')
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception as exc:
        errors.append(f'sync_after: {type(exc).__name__}: {exc}')
    after = _cuda_memory_snapshot()
    if before is not None or after is not None:
        bfree = _mb(before['driver_free']) if before else None
        afree = _mb(after['driver_free']) if after else None
        bres = _mb(before['reserved']) if before else None
        ares = _mb(after['reserved']) if after else None
        _lm_print(
            '[MiniMaxH3 LongMedia][V316 DECODE MEMORY BARRIER] '
            f'driver_free_mb {bfree}->{afree}, reserved_mb {bres}->{ares}'
            + (f', cleanup_errors={errors}' if errors else ''),
            flush=True,
        )
    return {'before': before, 'after': after, 'errors': errors}


def _decode_video_vae_safe(video_vae, video, enable_tiling, tile_size, temporal_size):
    """Decode H3 video latent without triggering ComfyUI's full-decode OOM path.

    When tiling is enabled, call the public VAE.decode_tiled API directly using
    the same unit conversion as ComfyUI's VAEDecodeTiled node.  This avoids the
    regular VAE.decode() attempt that may exhaust VRAM before fallback can run.
    """
    if not bool(enable_tiling):
        _lm_print('[MiniMaxH3 LongMedia][V316 DECODE] regular decode requested', flush=True)
        return video_vae.decode(video), {
            'strategy': 'regular',
            'tile_size_px': None,
            'temporal_size_frames': None,
        }

    tile_size_px = max(64, int(tile_size))
    spatial_overlap_px = min(64, max(16, tile_size_px // 4))
    if tile_size_px < spatial_overlap_px * 4:
        spatial_overlap_px = max(1, tile_size_px // 4)

    temporal_frames = max(8, int(temporal_size))
    temporal_overlap_frames = max(4, min(8, temporal_frames // 4))
    if temporal_frames < temporal_overlap_frames * 2:
        temporal_overlap_frames = max(1, temporal_frames // 2)

    compression = int(video_vae.spacial_compression_decode())
    tile_x = max(1, tile_size_px // compression)
    tile_y = max(1, tile_size_px // compression)
    overlap = max(0, spatial_overlap_px // compression)

    temporal_compression = video_vae.temporal_compression_decode()
    if temporal_compression is not None:
        temporal_compression = int(temporal_compression)
        tile_t = max(2, temporal_frames // temporal_compression)
        overlap_t = max(1, min(tile_t // 2, temporal_overlap_frames // temporal_compression))
    else:
        tile_t = None
        overlap_t = None

    _lm_print(
        '[MiniMaxH3 LongMedia][V316 DECODE] direct tiled decode: '
        f'latent_shape={tuple(video.shape)}, tile_px={tile_size_px}, overlap_px={spatial_overlap_px}, '
        f'tile_latent=({tile_x},{tile_y}), temporal_frames={temporal_frames}, '
        f'tile_t={tile_t}, overlap_t={overlap_t}',
        flush=True,
    )
    images = video_vae.decode_tiled(
        video,
        tile_x=tile_x,
        tile_y=tile_y,
        overlap=overlap,
        tile_t=tile_t,
        overlap_t=overlap_t,
    )
    return images, {
        'strategy': 'direct_tiled',
        'tile_size_px': tile_size_px,
        'spatial_overlap_px': spatial_overlap_px,
        'tile_x_latent': tile_x,
        'tile_y_latent': tile_y,
        'temporal_size_frames': temporal_frames,
        'temporal_overlap_frames': temporal_overlap_frames,
        'tile_t_latent': tile_t,
        'overlap_t_latent': overlap_t,
    }


def _cuda_memory_snapshot(device=None):
    """Best-effort CUDA allocator/driver snapshot; never raises after async OOM."""
    if not torch.cuda.is_available():
        return None
    try:
        if device is None:
            device = torch.cuda.current_device()
        free_driver, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        cached = max(0, reserved - allocated)
        return {
            'device': int(device) if isinstance(device, int) else device,
            'driver_free': int(free_driver),
            'total': int(total),
            'allocated': int(allocated),
            'reserved': int(reserved),
            'cached': int(cached),
        }
    except Exception:
        return None



def _mb(value):
    return round(float(value) / (1024.0 ** 2), 1)


def _soft_empty_cuda_cache():
    """Release unused CUDA allocator blocks without unloading active models."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        comfy.model_management.soft_empty_cache(force=True)
    except TypeError:
        # Compatibility with older ComfyUI builds whose helper had no force kwarg.
        comfy.model_management.soft_empty_cache()


def _aimdo_setup_boundary_reset(label: str):
    """Synchronize and clear transient AIMDO/Comfy loader state before Setup TE work.

    Mirrors the cleanup ComfyUI itself performs at execution boundaries on current
    DynamicVRAM builds: synchronize pending CUDA work, cleanup prefetch queues,
    reset temporary cast buffers, and reset VBAR watermark limits. This is
    scheduling/lifecycle only; it does not change model weights or math.
    """
    events = []
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            events.append('cuda_sync_pre')
        except Exception as exc:
            events.append(f'cuda_sync_pre:{type(exc).__name__}')

    if _comfy_model_prefetch is not None:
        fn = getattr(_comfy_model_prefetch, 'cleanup_prefetch_queues', None)
        if callable(fn):
            try:
                fn()
                events.append('prefetch_cleanup')
            except Exception as exc:
                events.append(f'prefetch_cleanup:{type(exc).__name__}')

    fn = getattr(comfy.model_management, 'reset_cast_buffers', None)
    if callable(fn):
        try:
            fn()
            events.append('cast_buffers_reset')
        except Exception as exc:
            events.append(f'cast_buffers_reset:{type(exc).__name__}')

    if _comfy_aimdo is not None:
        mv = getattr(_comfy_aimdo, 'model_vbar', None)
        fn = getattr(mv, 'vbars_reset_watermark_limits', None) if mv is not None else None
        if callable(fn):
            try:
                fn()
                events.append('vbar_watermarks_reset')
            except Exception as exc:
                events.append(f'vbar_watermarks_reset:{type(exc).__name__}')

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            events.append('cuda_sync_post')
        except Exception as exc:
            events.append(f'cuda_sync_post:{type(exc).__name__}')

    _lm_print(
        '[MiniMaxH3 LongMedia][AIMDO SETUP BOUNDARY] '
        f'{label}: ' + ','.join(events),
        flush=True,
    )
    return events


def _is_aimdo_transient_fault(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        'fault failed' in text
        or 'device not ready' in text
        or 'vram allocation failed (non oom)' in text
        or 'cumemsetaccess' in text
    )


def _setup_clip_encode_retry(fn, *, label: str):
    """Run one TE encode with one lifecycle-only retry for transient AIMDO VBAR races."""
    try:
        return fn()
    except RuntimeError as exc:
        if not _is_aimdo_transient_fault(exc):
            raise
        _lm_print(
            '[MiniMaxH3 LongMedia][AIMDO TE RETRY] '
            f'{label}: transient AIMDO fault ({type(exc).__name__}: {exc}); resetting loader state and retrying once',
            flush=True,
        )
        try:
            comfy.model_management.unload_all_models()
        except Exception:
            pass
        _aimdo_setup_boundary_reset(label + ':retry')
        try:
            _soft_empty_cuda_cache()
        except Exception:
            pass
        return fn()


def _setup_memory_isolation(label, unload_models=True):
    """Create a clean VRAM boundary between Setup's heavy model stages.

    LongMediaSetup may run after a previous H3 execution, so diffusion weights
    can still occupy most of VRAM when Qwen/CLIP starts encoding.  Explicitly
    unload registered models and flush only dead allocator cache before/after
    conditioning stages.  Returned counters are JSON-safe and used in the
    Setup report for diagnostics.
    """
    before = _cuda_memory_snapshot()
    unload_error = None
    # v0.3.82: native AIMDO uses asynchronous VBAR/prefetch state that can
    # outlive the previous heavy stage. Clear it at Setup boundaries before
    # asking Qwen/CLIP to fault a different model into the same GPU.
    _aimdo_setup_boundary_reset(str(label) + ':pre')
    if unload_models:
        try:
            comfy.model_management.unload_all_models()
        except Exception as exc:
            unload_error = f'{type(exc).__name__}: {exc}'
    _aimdo_setup_boundary_reset(str(label) + ':post_unload')
    try:
        _soft_empty_cuda_cache()
    except Exception as exc:
        if unload_error is None:
            unload_error = f'cache: {type(exc).__name__}: {exc}'
    after = _cuda_memory_snapshot()

    def compact(snap):
        if snap is None:
            return None
        return {
            'allocated_mb': _mb(snap['allocated']),
            'reserved_mb': _mb(snap['reserved']),
            'cached_mb': _mb(snap['cached']),
            'driver_free_mb': _mb(snap['driver_free']),
        }

    b = compact(before)
    a = compact(after)
    if a is not None:
        before_alloc = b['allocated_mb'] if b else 0.0
        before_free = b['driver_free_mb'] if b else 0.0
        _lm_print(
            '[MiniMaxH3 LongMedia] Setup memory isolation: '
            f'{label}, allocated {before_alloc:.1f} -> {a["allocated_mb"]:.1f} MB, '
            f'driver free {before_free:.1f} -> {a["driver_free_mb"]:.1f} MB',
            flush=True,
        )
    if unload_error:
        _lm_print(
            '[MiniMaxH3 LongMedia] Setup memory isolation warning: '
            f'{label}: {unload_error}',
            flush=True,
        )
    return {
        'stage': str(label),
        'before': b,
        'after': a,
        'unload_models': bool(unload_models),
        'warning': unload_error,
    }


def _sampler_memory_isolation(label: str = 'sampler_entry', unload_models: bool = True):
    """Create a hard execution boundary before every LongMedia sampler run.

    ComfyUI may cache Setup and rerun only the sampler when a seed or sampler
    parameter changes. In that path the previous downstream VideoVAE/AudioVAE
    and AIMDO/VBAR transport state can still own VRAM even though the new run
    has not started yet. The first run therefore succeeds while an otherwise
    identical seed-only rerun can OOM in model_prefetch before block 0.

    Reset transient loader state, unload registered models, then release dead
    allocator pages before prepare_sampling() performs fresh memory planning.
    This changes lifecycle/residency only; model weights, conditioning and H3
    math are untouched.
    """
    before = _cuda_memory_snapshot()
    warning = None
    events = []

    try:
        events.extend(_aimdo_setup_boundary_reset(str(label) + ':pre'))
    except Exception as exc:
        warning = f'pre_reset: {type(exc).__name__}: {exc}'

    if unload_models:
        try:
            comfy.model_management.unload_all_models()
            events.append('unload_all_models')
        except Exception as exc:
            msg = f'unload_all_models: {type(exc).__name__}: {exc}'
            warning = msg if warning is None else warning + '; ' + msg

    try:
        events.extend(_aimdo_setup_boundary_reset(str(label) + ':post_unload'))
    except Exception as exc:
        msg = f'post_reset: {type(exc).__name__}: {exc}'
        warning = msg if warning is None else warning + '; ' + msg

    try:
        _soft_empty_cuda_cache()
        events.append('soft_empty_cache')
    except Exception as exc:
        msg = f'cache: {type(exc).__name__}: {exc}'
        warning = msg if warning is None else warning + '; ' + msg

    after = _cuda_memory_snapshot()

    def compact(snap):
        if snap is None:
            return None
        return {
            'allocated_mb': _mb(snap['allocated']),
            'reserved_mb': _mb(snap['reserved']),
            'cached_mb': _mb(snap['cached']),
            'driver_free_mb': _mb(snap['driver_free']),
        }

    b = compact(before)
    a = compact(after)
    if a is not None:
        _lm_print(
            '[MiniMaxH3 LongMedia][SAMPLER EXECUTION BOUNDARY] '
            f'{label}; allocated={(b or {}).get("allocated_mb", 0.0):.1f}->{a["allocated_mb"]:.1f}MB; '
            f'reserved={(b or {}).get("reserved_mb", 0.0):.1f}->{a["reserved_mb"]:.1f}MB; '
            f'driver_free={(b or {}).get("driver_free_mb", 0.0):.1f}->{a["driver_free_mb"]:.1f}MB; '
            f'events={events}'
            + (f'; warning={warning}' if warning else ''),
            flush=True,
        )
    elif warning:
        _lm_print(
            '[MiniMaxH3 LongMedia][SAMPLER EXECUTION BOUNDARY] '
            f'{label}; warning={warning}',
            flush=True,
        )

    return {
        'stage': str(label),
        'before': b,
        'after': a,
        'unload_models': bool(unload_models),
        'events': list(events),
        'warning': warning,
    }


def _memory_profile_output_path(kind: str) -> str:
    """Return a Comfy temp path for a CUDA allocator snapshot."""
    try:
        import folder_paths
        base = folder_paths.get_temp_directory()
    except Exception:
        base = os.path.abspath(os.path.join(os.getcwd(), 'temp'))
    directory = os.path.join(base, 'minimax_h3_latentlab_memory')
    os.makedirs(directory, exist_ok=True)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    token = _uuid_mod.uuid4().hex[:8]
    return os.path.join(directory, f'h3_{kind}_{stamp}_{token}.pickle')


def _cuda_allocator_backend():
    """Best-effort name of the active PyTorch CUDA allocator backend."""
    if not torch.cuda.is_available():
        return None
    getter = getattr(getattr(torch.cuda, 'memory', None), 'get_allocator_backend', None)
    if getter is None:
        return None
    try:
        return str(getter())
    except Exception:
        return None


def _start_cuda_memory_history(max_entries=20000):
    """Enable PyTorch CUDA allocator history with compatibility fallbacks.

    cudaMallocAsync currently cannot record allocator history. Detect it up front
    so diagnostic builds do not emit a RuntimeError on every run.
    """
    if not torch.cuda.is_available():
        return False, None
    backend = _cuda_allocator_backend()
    if backend and 'cudamallocasync' in backend.lower():
        return False, f'allocator backend {backend} does not support record_memory_history'
    recorder = getattr(getattr(torch.cuda, 'memory', None), '_record_memory_history', None)
    if recorder is None:
        return False, 'torch.cuda.memory._record_memory_history unavailable'
    try:
        recorder(enabled='all', context='all', stacks='python', max_entries=int(max_entries), clear_history=True)
        return True, None
    except TypeError:
        try:
            recorder(max_entries=int(max_entries))
            return True, None
        except Exception as exc:
            return False, f'{type(exc).__name__}: {exc}'
    except Exception as exc:
        return False, f'{type(exc).__name__}: {exc}'


def _stop_cuda_memory_history():
    recorder = getattr(getattr(torch.cuda, 'memory', None), '_record_memory_history', None)
    if recorder is None:
        return
    try:
        recorder(enabled=None)
    except TypeError:
        try:
            recorder(None)
        except Exception:
            pass
    except Exception:
        pass


def _dump_cuda_memory_snapshot(kind: str):
    """Dump a PyTorch allocator snapshot and return (path, error)."""
    dumper = getattr(getattr(torch.cuda, 'memory', None), '_dump_snapshot', None)
    if dumper is None:
        return None, 'torch.cuda.memory._dump_snapshot unavailable'
    path = _memory_profile_output_path(kind)
    try:
        dumper(path)
        return path, None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'



class _H3QueryChunkAttentionOverride:
    """Low-VRAM full-attention override that chunks only the query sequence.

    K/V remain complete for every chunk, so this is still global/full attention;
    only the query rows are evaluated in smaller batches to cap temporary
    attention workspace. H3 currently calls optimized_attention with
    skip_reshape=True and mask=None, which is the path optimized here.
    """

    def __init__(self, chunk_tokens=8192, state=None):
        self.chunk_tokens = max(256, int(chunk_tokens))
        self.state = state if state is not None else {}

    def __call__(self, func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        state = self.state
        # Keep the override deliberately narrow. Anything that is not H3's
        # unmasked pre-shaped attention path is delegated untouched.
        if (
            mask is not None
            or not skip_reshape
            or getattr(q, 'ndim', 0) != 4
            or int(q.shape[-2]) <= self.chunk_tokens
        ):
            return func(
                q, k, v, heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )

        seq = int(q.shape[-2])
        chunks = (seq + self.chunk_tokens - 1) // self.chunk_tokens
        state['calls'] = int(state.get('calls', 0)) + 1
        state['chunked_calls'] = int(state.get('chunked_calls', 0)) + 1
        state['max_sequence_tokens'] = max(int(state.get('max_sequence_tokens', 0)), seq)
        state['max_chunks_per_call'] = max(int(state.get('max_chunks_per_call', 0)), chunks)

        if not state.get('announced'):
            _lm_print(
                '[MiniMaxH3 LongMedia] Low-VRAM attention enabled: '
                f'query sequence {seq} tokens -> {chunks} chunks of <= {self.chunk_tokens}; K/V stay full',
                flush=True,
            )
            state['announced'] = True

        outputs = []
        for start in range(0, seq, self.chunk_tokens):
            end = min(seq, start + self.chunk_tokens)
            q_chunk = q[..., start:end, :]
            out = func(
                q_chunk, k, v, heads,
                mask=None,
                attn_precision=attn_precision,
                skip_reshape=True,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )
            outputs.append(out)

        # Core Comfy attention functions return [B, Nq, H*D] by default;
        # skip_output_reshape=True keeps [B, H, Nq, D].
        cat_dim = 2 if skip_output_reshape else 1
        return torch.cat(outputs, dim=cat_dim)


class MiniMaxH3LatentLabAttentionChunking:
    """Internal GUIDER wrapper enabling query-chunked full attention."""

    DESCRIPTION = 'Internal H3 low-VRAM query-chunk attention wrapper.'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'guider': ('GUIDER',),
                'memory_mode': (['normal', 'low_vram', 'ultra_low_vram'], {'default': 'normal'}),
                'requested_memory_mode': (['auto', 'normal', 'low_vram', 'ultra_low_vram'], {'default': 'normal'}),
                'chunk_tokens': ('INT', {'default': 8192, 'min': 256, 'max': 65536, 'step': 256}),
            }
        }

    RETURN_TYPES = ('GUIDER', 'H3_ATTENTION_CHUNK_STATE')
    RETURN_NAMES = ('guider', 'attention_chunk_state')
    FUNCTION = 'wrap'
    CATEGORY = CATEGORY_LONGMEDIA

    def wrap(self, guider, chunk_tokens=8192):
        wrapped = copy.copy(guider)
        wrapped.model_options = _clone_model_options_safe(getattr(guider, 'model_options', {}) or {})

        # v0.4.36: ModelPatcher-owned attention overrides (notably H3 SLA) can
        # live only in guider.model_patcher.model_options.  Copying guider's
        # lightweight model_options alone silently drops the hook, which is why
        # H3Utils reported `patch installed but never invoked`.  Inherit the
        # authoritative patcher override before LongMedia builds its block path.
        _patcher_model_options = getattr(getattr(guider, 'model_patcher', None), 'model_options', {}) or {}
        _patcher_to = (_patcher_model_options.get('transformer_options', {}) or {}) if isinstance(_patcher_model_options, dict) else {}
        _patcher_attn_override = _patcher_to.get('optimized_attention_override') if isinstance(_patcher_to, dict) else None
        transformer_options = wrapped.model_options.setdefault('transformer_options', {})
        if _patcher_attn_override is not None and transformer_options.get('optimized_attention_override') is None:
            transformer_options['optimized_attention_override'] = _patcher_attn_override
            _lm_print(
                '[MiniMaxH3 LongMedia][ATTENTION HOOK FIX] inherited '
                'ModelPatcher optimized_attention_override into LongMedia execution options',
                flush=True,
            )
        state = {
            'chunk_tokens': int(chunk_tokens),
            'calls': 0,
            'chunked_calls': 0,
            'max_sequence_tokens': 0,
            'max_chunks_per_call': 0,
            'announced': False,
            'step_boundary_forward_count': 0,
            'step_boundary_transitions': [],
            'step_boundary_forward_times': [],
        }
        existing = transformer_options.get('optimized_attention_override')
        if existing is not None:
            # Do not silently trample a user's/custom node's existing attention
            # override. The diagnostic state makes this visible in the report/log.
            state['enabled'] = False
            state['reason'] = 'existing optimized_attention_override present'
            _lm_print(
                '[MiniMaxH3 LongMedia] Low-VRAM attention NOT installed: '
                'another optimized_attention_override is already present.',
                flush=True,
            )
        else:
            transformer_options['optimized_attention_override'] = _H3QueryChunkAttentionOverride(
                chunk_tokens=int(chunk_tokens), state=state
            )
            state['enabled'] = True
            state['reason'] = 'query chunking active'
        return (wrapped, state)


class _H3BlockMemoryTracePatch:
    """Deep first-block tracer for the first H3 DiT forward only."""

    def __init__(self, index, state):
        self.index = int(index)
        self.state = state

    @staticmethod
    def _extract_block(original_block):
        # Stock H3 creates block_wrap as a closure over the actual DiTBlock.
        closure = getattr(original_block, '__closure__', None) or ()
        for cell in closure:
            try:
                obj = cell.cell_contents
            except Exception:
                continue
            if all(hasattr(obj, name) for name in ('adaln_proj', 'norm1', 'attn', 'norm2', 'mlp')):
                return obj
        return None

    @staticmethod
    def _mod_scale_shift(h, shift, scale, segments):
        for a, b, row in segments:
            h[a:b].mul_(1.0 + scale[row].to(h.dtype)).add_(shift[row].to(h.dtype))
        return h

    @staticmethod
    def _mod_gate(x, gate, other, segments):
        for a, b, row in segments:
            x[a:b].addcmul_(other[a:b], gate[row].to(x.dtype))
        return x

    def _inter_block_pressure_guard(self):
        """TEST: effective-headroom inter-block guard with hysteresis.

        Uses driver-free VRAM + allocator-reclaimable cache instead of driver
        free alone. This avoids destroying useful cache when DynamicVRAM/AIMDO
        already has several GB of reclaimable memory available.
        """
        state = self.state

        # Guarantee AUTO calibration after the first completed block.
        if self.index == 0 and not state.get('auto_vram_controller_done'):
            try:
                token_count = int(state.get('current_token_count', 0) or state.get('last_token_count', 0) or 0)
                self._auto_vram_controller_after_probe(token_count)
            except Exception as exc:
                state['auto_vram_controller_done'] = True
                state['auto_vram_controller_mode'] = 'SAFE'
                _lm_print(
                    f"[MiniMaxH3 LongMedia][AUTO VRAM] calibration failed; SAFE baseline retained: {exc!r}",
                    flush=True,
                )

        if not torch.cuda.is_available():
            return

        guard_mb = float(int(state.get('inter_block_vram_guard_mb', 0) or 0))
        emergency_mb = float(int(state.get('inter_block_guard_emergency_mb', 0) or 0))
        cooldown_blocks = int(state.get('inter_block_guard_cooldown_blocks', 0) or 0)
        emergency_cooldown_blocks = int(state.get('inter_block_guard_emergency_cooldown_blocks', 0) or 0)

        if guard_mb <= 0 and emergency_mb <= 0:
            return

        snap = _cuda_memory_snapshot()
        if not snap:
            return

        mb = 1024.0 ** 2
        free_mb = float(snap['driver_free']) / mb
        cached_mb = float(snap['cached']) / mb
        effective_mb = free_mb + cached_mb

        normal_hyst = float(state.get('inter_block_guard_hysteresis_mb', 1024.0) or 1024.0)
        emergency_hyst = float(state.get('inter_block_emergency_hysteresis_mb', 512.0) or 512.0)

        # Existing cooldown bookkeeping, but now only decremented when evaluated.
        cd = int(state.get('inter_block_guard_cooldown', 0) or 0)
        ecd = int(state.get('inter_block_guard_emergency_cooldown', 0) or 0)

        # Emergency should reflect *effective* pressure, not merely low driver free.
        emergency_trigger = emergency_mb > 0 and effective_mb < emergency_mb
        normal_trigger = guard_mb > 0 and effective_mb < guard_mb

        if not emergency_trigger and not normal_trigger:
            if cd > 0:
                state['inter_block_guard_cooldown'] = cd - 1
            if ecd > 0:
                state['inter_block_guard_emergency_cooldown'] = ecd - 1
            skips = int(state.get('inter_block_effective_skip_count', 0) or 0) + 1
            state['inter_block_effective_skip_count'] = skips
            if skips == 1 or skips % 25 == 0:
                _lm_print(
                    '[MiniMaxH3 LongMedia][VRAM GUARD] skip: '
                    f'block {self.index}, free={free_mb:.0f} + cached={cached_mb:.0f} '
                    f'= effective={effective_mb:.0f} MB, guard={guard_mb:.0f}, emergency={emergency_mb:.0f}',
                    flush=True,
                )
            return

        # Hysteresis band prevents trim/reload ping-pong close to threshold.
        if normal_trigger and effective_mb >= max(0.0, guard_mb - normal_hyst):
            state['inter_block_hysteresis_skip_count'] = int(
                state.get('inter_block_hysteresis_skip_count', 0) or 0
            ) + 1
            if cd > 0:
                state['inter_block_guard_cooldown'] = cd - 1
            return

        if emergency_trigger and effective_mb >= max(0.0, emergency_mb - emergency_hyst):
            state['inter_block_emergency_hyst_skip_count'] = int(
                state.get('inter_block_emergency_hyst_skip_count', 0) or 0
            ) + 1
            if ecd > 0:
                state['inter_block_guard_emergency_cooldown'] = ecd - 1
            return

        # Cooldowns suppress repeated trims unless pressure is materially worse.
        if emergency_trigger:
            hard_emergency = effective_mb < max(0.0, emergency_mb - 1024.0)
            if ecd > 0 and not hard_emergency:
                state['inter_block_emergency_cooldown_skip_count'] = int(
                    state.get('inter_block_emergency_cooldown_skip_count', 0) or 0
                ) + 1
                state['inter_block_guard_emergency_cooldown'] = ecd - 1
                return
        elif normal_trigger:
            hard_normal = effective_mb < max(0.0, guard_mb - 1536.0)
            if cd > 0 and not hard_normal:
                state['inter_block_cooldown_skip_count'] = int(
                    state.get('inter_block_cooldown_skip_count', 0) or 0
                ) + 1
                state['inter_block_guard_cooldown'] = cd - 1
                return

        # If cache is tiny, cleanup is unlikely to help; preserve cache and let
        # Sol adaptive retry / CUDA OOM handling be the final safety net.
        min_reclaim_mb = float(state.get('inter_block_min_reclaim_mb', 256.0) or 256.0)
        if cached_mb < min_reclaim_mb:
            state['inter_block_low_cache_skip_count'] = int(
                state.get('inter_block_low_cache_skip_count', 0) or 0
            ) + 1
            return

        try:
            gc.collect()
            comfy.model_management.soft_empty_cache()
        except Exception as exc:
            _lm_print(
                f"[MiniMaxH3 LongMedia][VRAM GUARD] cleanup failed at block {self.index}: {exc!r}",
                flush=True,
            )
            return

        after = _cuda_memory_snapshot()
        if emergency_trigger:
            state['inter_block_guard_emergency_cooldown'] = emergency_cooldown_blocks
            state['inter_block_emergency_trim_count'] = int(
                state.get('inter_block_emergency_trim_count', 0) or 0
            ) + 1
            label = 'EMERGENCY TRIM'
        else:
            state['inter_block_guard_cooldown'] = cooldown_blocks
            state['inter_block_normal_trim_count'] = int(
                state.get('inter_block_normal_trim_count', 0) or 0
            ) + 1
            label = 'NORMAL TRIM'

        if after:
            free_after = float(after['driver_free']) / mb
            cached_after = float(after['cached']) / mb
            _lm_print(
                f'[MiniMaxH3 LongMedia][VRAM GUARD] {label}: block {self.index}, '
                f'effective={effective_mb:.0f} MB, free {free_mb:.0f}->{free_after:.0f} MB, '
                f'cached {cached_mb:.0f}->{cached_after:.0f} MB',
                flush=True,
            )

    def _measure(self, name, fn, state, device):
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        before = _cuda_memory_snapshot()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        out = fn()
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        after = _cuda_memory_snapshot()
        peak_alloc = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        entry = {
            'stage': name,
            'allocated_before_mb': _mb(before['allocated']),
            'allocated_after_mb': _mb(after['allocated']),
            'reserved_before_mb': _mb(before['reserved']),
            'reserved_after_mb': _mb(after['reserved']),
            'driver_free_after_mb': _mb(after['driver_free']),
            'peak_allocated_mb': _mb(peak_alloc),
            'peak_reserved_mb': _mb(peak_reserved),
            'elapsed_ms': round(elapsed_ms, 1),
        }
        state['stages'].append(entry)
        state['highest_block_peak_allocated_mb'] = max(
            float(state.get('highest_block_peak_allocated_mb') or 0.0), float(entry['peak_allocated_mb']))
        state['highest_block_peak_reserved_mb'] = max(
            float(state.get('highest_block_peak_reserved_mb') or 0.0), float(entry['peak_reserved_mb']))
        if float(entry['peak_allocated_mb']) >= float(state.get('worst_stage_peak_allocated_mb') or -1.0):
            state['worst_stage'] = name
            state['worst_stage_peak_allocated_mb'] = float(entry['peak_allocated_mb'])
        _lm_print(
            '[MiniMaxH3 LongMedia] H3 block0 stage: '
            f"{name}, alloc {entry['allocated_before_mb']:.1f} -> {entry['allocated_after_mb']:.1f} MB, "
            f"peak {entry['peak_allocated_mb']:.1f} MB, reserved peak {entry['peak_reserved_mb']:.1f} MB, "
            f"free {entry['driver_free_after_mb']:.1f} MB, {entry['elapsed_ms']:.1f} ms",
            flush=True,
        )
        return out

    def __call__(self, args, extra_options):
        original_block = extra_options['original_block']
        state = self.state

        if self.index != 0 or state.get('first_forward_complete'):
            return original_block(args)
        if state.get('forward_count', 0) > 0:
            state['first_forward_complete'] = True
            return original_block(args)

        state['forward_count'] = 1
        state['first_forward_started'] = True
        state['first_forward_started_at'] = time.time()
        _lm_print('[MiniMaxH3 LongMedia] H3 block0 deep memory trace: first forward started', flush=True)

        if not torch.cuda.is_available():
            return original_block(args)

        block = self._extract_block(original_block)
        if block is None:
            state['fallback_reason'] = 'could not extract DiTBlock from original_block closure'
            _lm_print('[MiniMaxH3 LongMedia] H3 block0 deep trace fallback: DiTBlock closure not found', flush=True)
            return original_block(args)

        device = torch.cuda.current_device()
        x = args['img']
        t_emb = args['t_emb']
        mod_segments = args['mod_segments']
        rope_freqs = args['rope_freqs']
        transformer_options = args['transformer_options']

        try:
            vals = self._measure('adaln_proj', lambda: block.adaln_proj(t_emb), state, device)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = vals

            h = self._measure(
                'norm1_mod',
                lambda: self._mod_scale_shift(block.norm1(x), shift_msa, scale_msa, mod_segments),
                state, device,
            )
            attn_out = self._measure(
                'attention_full',
                lambda: block.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options),
                state, device,
            )
            x = self._measure(
                'attention_gate_residual',
                lambda: self._mod_gate(x, gate_msa, attn_out, mod_segments),
                state, device,
            )
            del attn_out

            h = self._measure(
                'norm2_mod',
                lambda: self._mod_scale_shift(block.norm2(x), shift_mlp, scale_mlp, mod_segments),
                state, device,
            )
            mlp_out = self._measure('mlp_full', lambda: block.mlp(h), state, device)
            x = self._measure(
                'mlp_gate_residual',
                lambda: self._mod_gate(x, gate_mlp, mlp_out, mod_segments),
                state, device,
            )
            del mlp_out, h

            # Preserve patches_replace contract.
            state['blocks'].append({
                'block': 0,
                'peak_allocated_mb': state.get('highest_block_peak_allocated_mb', 0.0),
                'peak_reserved_mb': state.get('highest_block_peak_reserved_mb', 0.0),
                'deep_trace': True,
            })
            _lm_print(
                '[MiniMaxH3 LongMedia] H3 block0 deep trace summary: '
                f"worst stage {state.get('worst_stage')}, "
                f"peak allocated {state.get('highest_block_peak_allocated_mb', 0.0):.1f} MB",
                flush=True,
            )
            return {'img': x}
        except Exception as exc:
            message = str(exc).lower()
            is_oom = isinstance(exc, getattr(torch, 'OutOfMemoryError', RuntimeError)) or 'out of memory' in message
            if is_oom:
                state['oom'] = True
                state['oom_block'] = 0
                state['oom_stage'] = state.get('stages', [])[-1]['stage'] if state.get('stages') else 'unknown'
                state['oom_message'] = str(exc)[:2000]
                _lm_print(
                    f"[MiniMaxH3 LongMedia] H3 BLOCK0 CUDA OOM near stage {state.get('oom_stage')}: {state['oom_message']}",
                    flush=True,
                )
            raise


class MiniMaxH3LatentLabBlockMemoryTracer:
    """Internal GUIDER wrapper that instruments H3 DiT blocks via patches_replace."""

    DESCRIPTION = 'Internal first-forward H3 transformer block VRAM tracer.'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'guider': ('GUIDER',),
                'max_blocks': ('INT', {'default': 128, 'min': 1, 'max': 256, 'step': 1}),
            }
        }

    RETURN_TYPES = ('GUIDER', 'H3_BLOCK_MEMORY_TRACE_STATE')
    RETURN_NAMES = ('guider', 'block_trace_state')
    FUNCTION = 'wrap'
    CATEGORY = CATEGORY_LONGMEDIA

    def wrap(self, guider, max_blocks=128):
        traced = copy.copy(guider)
        traced.model_options = _clone_model_options_safe(getattr(guider, 'model_options', {}) or {})

        transformer_options = traced.model_options.setdefault('transformer_options', {})
        patches_replace = transformer_options.setdefault('patches_replace', {})
        dit = patches_replace.setdefault('dit', {})
        state = {
            'allocator_backend': _cuda_allocator_backend(),
            'max_blocks': int(max_blocks),
            'forward_count': 0,
            'first_forward_started': False,
            'first_forward_complete': False,
            'first_forward_started_at': None,
            'pre_block0': None,
            'blocks': [],
            'stages': [],
            'worst_stage': None,
            'worst_stage_peak_allocated_mb': 0.0,
            'fallback_reason': None,
            'skipped_existing_patch_indices': [],
            'highest_block_peak_allocated_mb': 0.0,
            'highest_block_peak_reserved_mb': 0.0,
            'worst_block': None,
            'worst_block_peak_allocated_mb': 0.0,
            'oom': False,
            'oom_block': None,
            'oom_message': None,
            'oom_stats': None,
        }
        for i in range(int(max_blocks)):
            key = ('double_block', i)
            if key in dit:
                state['skipped_existing_patch_indices'].append(i)
                continue
            dit[key] = _H3BlockMemoryTracePatch(i, state)
        return (traced, state)



def _install_memory_safe_external_sla(transformer_options, state):
    """Replace BetaDoggo H3 SLA's full-size output allocation with an in-place kernel.

    The external SLA LoRA/selection policy is preserved.  Only the attention
    output buffer strategy changes: after the block map is built, Q storage is
    reused for O instead of allocating ``torch.empty_like(q)`` (~1.3 GiB at the
    failing 1 MP/13 s geometry).  The block-map score matrix is also streamed in
    query-block chunks so peak VRAM stays bounded.
    """
    existing = (transformer_options or {}).get('optimized_attention_override')
    if existing is None:
        return False, 'no external optimized_attention_override'
    module_name = str(getattr(existing, '__module__', '') or '')
    if not (module_name == 'sla.patch' or module_name.endswith('.sla.patch')):
        return False, f'non-SLA override: {module_name or type(existing).__name__}'
    try:
        import inspect
        closure = inspect.getclosurevars(existing).nonlocals
        ext_state = closure.get('state')
        topk_ratio = float(closure.get('topk_ratio', 0.10))
        sparsity_ratio = float(1.0 - topk_ratio)
        blkq = int(closure.get('blkq', 64))
        blkk = int(closure.get('blkk', blkq))
        min_seq_len = int(closure.get('min_seq_len', 8192))
        protect_audio = bool(closure.get('protect_audio', True))
        from .sla_memory_safe import make_memory_safe_sla_override
        replacement = make_memory_safe_sla_override(
            ext_state=ext_state,
            sparsity_ratio=sparsity_ratio,
            blkq=blkq,
            blkk=blkk,
            min_seq_len=min_seq_len,
            protect_audio=protect_audio,
        )
        transformer_options['optimized_attention_override'] = replacement
        state['external_sla_memory_safe'] = True
        state['external_sla_original_module'] = module_name
        state['external_sla_sparsity_ratio'] = sparsity_ratio
        state['external_sla_block_q'] = blkq
        state['external_sla_block_k'] = blkk
        return True, 'memory-safe in-place SLA installed'
    except Exception as exc:
        state['external_sla_memory_safe'] = False
        state['external_sla_memory_safe_error'] = f'{type(exc).__name__}: {exc}'
        return False, state['external_sla_memory_safe_error']


def _h3_sol_span_wrapper(executor, *args, **kwargs):
    """Publish H3's packed video span without altering APPLY_MODEL call semantics.

    ComfyUI APPLY_MODEL wrappers receive the full BaseModel.apply_model argument
    list: (x, t, c_concat, c_crossattn, control, transformer_options, **kwargs).
    MiniMax-specific payload data lives in kwargs.  Keep the positional layout
    untouched so this wrapper composes with other APPLY_MODEL wrappers.
    """
    call_args = list(args)
    options = call_args[5] if len(call_args) > 5 and isinstance(call_args[5], dict) else kwargs.get('transformer_options')
    options = options or {}
    payload = kwargs.get('minimax_payload')
    layout = payload.get('layout') if isinstance(payload, dict) else None
    if layout is not None and hasattr(layout, 'segments'):
        try:
            span = next(((int(a), int(b)) for a, b, kind in layout.segments if kind == 'video'), None)
        except Exception:
            span = None
        if span is not None:
            options = dict(options)
            options['latentlab_sol_h3_video_span'] = span
            if len(call_args) > 5:
                call_args[5] = options
            else:
                kwargs['transformer_options'] = options
    return executor(*call_args, **kwargs)







def _lm_h3_tensor_video_geometry(z):
    """Return (t,h,w,rows) for an H3 video latent tensor."""
    if z is None or not hasattr(z, "shape") or len(z.shape) < 5:
        return None
    t = int(z.shape[-3])
    h = int(z.shape[-2])
    w = int(z.shape[-1])
    # H3 patch size is 1x2x2. Runtime pads the target before patchification.
    hp = ((h + 1) // 2) * 2
    wp = ((w + 1) // 2) * 2
    rows = int(t * (hp // 2) * (wp // 2))
    return t, hp, wp, rows


def _lm_h3_runtime_av_geometry(x):
    try:
        video = x[0]
        audio = x[1]
        vg = _lm_h3_tensor_video_geometry(video)
        if vg is None:
            return None
        audio_t = int(audio.shape[-1])
        return vg[0], vg[1], vg[2], audio_t, vg[3]
    except Exception:
        return None


def _lm_h3_normalize_payload_geometry(payload):
    """Clone H3 payload metadata and bind it to authoritative runtime cond tensors.

    ComfyUI builds ``cond_video_latents`` from visual keyframes first and then
    visual refs. LongMedia/runtime wrappers may refresh those tensors after the
    original metadata/layout was created. PackedLayout, however, derives row
    counts from ``keyframes``/``refs`` metadata while MiniMaxH3Model patchifies
    ``cond_video_latents`` directly. Any drift between the two produces a hard
    broadcast mismatch.

    Runtime condition tensors are therefore the single source of truth here.
    Metadata is rebound to them in *the exact upstream ordering* before any
    PackedLayout reuse/rebuild decision is made.
    """
    if not isinstance(payload, dict):
        return payload, False

    changed = False
    out = dict(payload)
    cond_video_latents = list(payload.get("cond_video_latents") or [])
    cond_audio_latents = list(payload.get("cond_audio_latents") or [])
    video_cursor = 0
    audio_cursor = 0

    keyframes = []
    for raw_kf in list(payload.get("keyframes") or []):
        if not isinstance(raw_kf, dict):
            keyframes.append(raw_kf)
            continue
        kf = dict(raw_kf)
        if kf.get("latent") is not None:
            if video_cursor >= len(cond_video_latents):
                raise RuntimeError(
                    '[H3 LAYOUT SELF-HEAL] visual keyframe metadata declares more '
                    'latents than cond_video_latents provides.'
                )
            authoritative = cond_video_latents[video_cursor]
            video_cursor += 1
            if kf.get("latent") is not authoritative:
                kf["latent"] = authoritative
                changed = True
        if kf.get("audio_latent") is not None:
            if audio_cursor >= len(cond_audio_latents):
                raise RuntimeError(
                    '[H3 LAYOUT SELF-HEAL] audio keyframe metadata declares more '
                    'latents than cond_audio_latents provides.'
                )
            authoritative_audio = cond_audio_latents[audio_cursor]
            audio_cursor += 1
            if kf.get("audio_latent") is not authoritative_audio:
                kf["audio_latent"] = authoritative_audio
                changed = True
        keyframes.append(kf)

    refs = []
    for raw_ref in list(payload.get("refs") or []):
        if not isinstance(raw_ref, dict):
            refs.append(raw_ref)
            continue
        item = dict(raw_ref)

        # Upstream MiniMaxH3.extra_conds appends visual ref latents after all
        # visual keyframe latents. Rebind in exactly that order.
        if "latent" in item:
            if video_cursor >= len(cond_video_latents):
                raise RuntimeError(
                    '[H3 LAYOUT SELF-HEAL] visual ref metadata declares more '
                    'latents than cond_video_latents provides.'
                )
            authoritative = cond_video_latents[video_cursor]
            video_cursor += 1
            if item.get("latent") is not authoritative:
                item["latent"] = authoritative
                changed = True

        latent = item.get("latent")
        geom = _lm_h3_tensor_video_geometry(latent)
        if geom is not None:
            t, h, w, _ = geom
            for key, value in (("latent_t", t), ("latent_h", h), ("latent_w", w)):
                if int(item.get(key, 0) or 0) != int(value):
                    item[key] = int(value)
                    changed = True

        if item.get("audio_latent") is not None:
            if audio_cursor >= len(cond_audio_latents):
                raise RuntimeError(
                    '[H3 LAYOUT SELF-HEAL] audio ref metadata declares more '
                    'latents than cond_audio_latents provides.'
                )
            authoritative_audio = cond_audio_latents[audio_cursor]
            audio_cursor += 1
            if item.get("audio_latent") is not authoritative_audio:
                item["audio_latent"] = authoritative_audio
                changed = True

        audio_latent = item.get("audio_latent")
        actual_audio_t = (
            int(audio_latent.shape[-1])
            if audio_latent is not None and hasattr(audio_latent, "shape")
            else 0
        )
        if int(item.get("ref_audio_t", 0) or 0) != actual_audio_t:
            item["ref_audio_t"] = int(actual_audio_t)
            changed = True
        if item.get("kind") == "video_audio" and actual_audio_t <= 0:
            item["kind"] = "video"
            changed = True
        elif item.get("kind") == "video" and actual_audio_t > 0:
            item["kind"] = "video_audio"
            changed = True
        refs.append(item)

    # A runtime wrapper is allowed to replace conditioning tensors, but it must
    # not silently add/remove visual/audio blocks without matching metadata.
    # Fail here with a precise contract error instead of letting model.py crash
    # later in all_video_rows/all_audio_rows assignment.
    if video_cursor != len(cond_video_latents):
        raise RuntimeError(
            '[H3 LAYOUT SELF-HEAL] cond_video_latents/metadata block-count mismatch: '
            f'metadata_visual_blocks={video_cursor}, runtime_visual_blocks={len(cond_video_latents)}.'
        )
    if audio_cursor != len(cond_audio_latents):
        raise RuntimeError(
            '[H3 LAYOUT SELF-HEAL] cond_audio_latents/metadata block-count mismatch: '
            f'metadata_audio_blocks={audio_cursor}, runtime_audio_blocks={len(cond_audio_latents)}.'
        )

    if keyframes or payload.get("keyframes") is not None:
        out["keyframes"] = keyframes
    if refs or payload.get("refs") is not None:
        out["refs"] = refs

    return out, changed



def _lm_h3_strip_incompatible_video_keyframes(payload, target_h, target_w):
    """Remove only VIDEO keyframe rows that cannot share the runtime target grid.

    Native MiniMax H3 keyframes are different from Ref2VA references:
    ``PackedLayout`` places keyframe video rows on the *target* spatial frame
    grid, while reference blocks carry their own ``latent_h``/``latent_w``.
    Therefore a keyframe latent produced for a pre-hires grid cannot be reused
    after a chained spatial latent upscale. Keeping it makes the layout reserve
    target-grid rows but the model patchifies fewer condition rows and fails
    before attention.

    Preserve audio-only data from a mixed AV keyframe by clearing only its
    visual latent. Ref2VA references and their condition tensors are left
    untouched because they are explicitly geometry-independent in PackedLayout.
    """
    if not isinstance(payload, dict):
        return payload, False, 0

    keyframes_raw = list(payload.get("keyframes") or [])
    cond_video_latents = list(payload.get("cond_video_latents") or [])
    if not keyframes_raw or not cond_video_latents:
        return payload, False, 0

    target_hw = (int(target_h), int(target_w))
    video_cursor = 0
    kept_keyframe_cond = []
    keyframes = []
    dropped = 0
    changed = False
    mismatch_details = []

    for raw_kf in keyframes_raw:
        if not isinstance(raw_kf, dict):
            keyframes.append(raw_kf)
            continue

        kf = dict(raw_kf)
        if kf.get("latent") is not None:
            if video_cursor >= len(cond_video_latents):
                # Let the existing normalization/count guards report a malformed
                # payload; do not silently invent condition ordering here.
                keyframes.append(kf)
                continue

            authoritative = cond_video_latents[video_cursor]
            video_cursor += 1
            geom = _lm_h3_tensor_video_geometry(authoritative)
            incompatible = (
                geom is not None
                and (int(geom[1]), int(geom[2])) != target_hw
            )
            if incompatible:
                dropped += 1
                changed = True
                mismatch_details.append(
                    {
                        "frame": int(kf.get("resolved_frame_index", -1)),
                        "from_hw": (int(geom[1]), int(geom[2])),
                        "to_hw": target_hw,
                        "rows": int(geom[3]),
                    }
                )
                # Keep any audio_latent / resolved-frame metadata. Only the
                # geometry-bound video condition is invalid on the new grid.
                kf["latent"] = None
            else:
                kept_keyframe_cond.append(authoritative)

        keyframes.append(kf)

    if not changed:
        return payload, False, 0

    # cond_video_latents are ordered as visual keyframes first, then refs.
    # Preserve the complete reference tail verbatim.
    kept_keyframe_cond.extend(cond_video_latents[video_cursor:])

    out = dict(payload)
    out["keyframes"] = keyframes
    out["cond_video_latents"] = kept_keyframe_cond
    out["longmedia_dropped_incompatible_video_keyframes"] = {
        "count": int(dropped),
        "target_hw": target_hw,
        "details": mismatch_details,
    }

    _lm_print(
        "[MiniMaxH3 LongMedia][H3 KEYFRAME GRID GUARD] "
        f"dropped_video_keyframes={int(dropped)}; target_hw={target_hw}; "
        f"details={mismatch_details}; refs_preserved=True; audio_keyframes_preserved=True",
        flush=True,
    )
    return out, True, int(dropped)


def _lm_h3_layout_counts(layout):
    if layout is None:
        return None
    try:
        img_update = layout.img_update
        audio_update = layout.audio_update
        target_video = int(img_update.sum().item())
        cond_video = int((~img_update).sum().item())
        target_audio = int(audio_update.sum().item())
        cond_audio = int((~audio_update).sum().item())
        return target_video, cond_video, target_audio, cond_audio
    except Exception:
        return None


def _lm_h3_cond_tensor_counts(payload):
    video_rows = 0
    audio_rows = 0
    for z in list((payload or {}).get("cond_video_latents") or []):
        g = _lm_h3_tensor_video_geometry(z)
        if g is not None:
            video_rows += int(g[3])
    for z in list((payload or {}).get("cond_audio_latents") or []):
        if z is not None and hasattr(z, "shape"):
            audio_rows += int(z.shape[-2] if len(z.shape) >= 2 and int(z.shape[-2]) == 2 else 2) * int(z.shape[-1])
    return int(video_rows), int(audio_rows)


def _lm_h3_rebuild_runtime_layout(payload, x, text_len, reason):
    """Rebuild PackedLayout from the AV tensors that will actually be patchified.

    This is deliberately done at DIFFUSION_MODEL entry, immediately before H3
    consumes the payload. It makes stale cached/foreign layouts harmless across
    resize, latent-hires, continuation and wrapper-chain merges.
    """
    if not isinstance(payload, dict):
        return payload, False

    geom = _lm_h3_runtime_av_geometry(x)
    if geom is None:
        return payload, False
    latent_t, lat_h, lat_w, audio_t, target_video_rows = geom

    try:
        import comfy.ldm.minimax.model as mm
    except Exception as exc:
        _lm_print(
            "[MiniMaxH3 LongMedia][H3 LAYOUT REBUILD] WARNING: "
            f"cannot import PackedLayout: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return payload, False

    normalized, metadata_changed = _lm_h3_normalize_payload_geometry(payload)
    # v0.5.30: a chained Latent Hi-Res sampler can legitimately change target
    # H/W while the original guider still carries low-resolution VIDEO
    # keyframes. Upstream PackedLayout binds keyframes to the target frame grid,
    # so strip only those incompatible visual rows before rebuilding. Ref2VA
    # references keep their own geometry and remain valid.
    normalized, keyframe_grid_changed, _dropped_grid_keyframes = (
        _lm_h3_strip_incompatible_video_keyframes(
            normalized, int(lat_h), int(lat_w)
        )
    )
    metadata_changed = bool(metadata_changed or keyframe_grid_changed)

    expected_signature = (
        int(text_len), int(latent_t), int(lat_h), int(lat_w), int(audio_t)
    )

    layout = normalized.get("layout")
    old_signature = getattr(layout, "signature", None) if layout is not None else None
    old_counts = _lm_h3_layout_counts(layout)

    rebuild = (
        layout is None
        or tuple(old_signature or ()) != expected_signature
        or metadata_changed
    )

    # Even a layout claiming the right signature can be structurally stale when
    # another wrapper mutated its masks/conditioning after construction.
    cond_video_rows, cond_audio_rows = _lm_h3_cond_tensor_counts(normalized)
    if old_counts is not None:
        old_target_video = int(old_counts[0])
        old_cond_video = int(old_counts[1])
        old_cond_audio = int(old_counts[3])
        if old_target_video != int(target_video_rows):
            rebuild = True
            reason = (
                f"{reason}; target_video_rows {old_target_video}->{target_video_rows}"
            )
        # v0.5.1: signature equality says nothing about condition geometry. A
        # refreshed multi-frame guide can keep target H/W/T unchanged while
        # changing condition rows by tens of thousands. Force rebuild before H3
        # reaches all_video_rows[~img_update] assignment.
        if int(cond_video_rows) != old_cond_video:
            rebuild = True
            reason = (
                f"{reason}; cond_video_rows {old_cond_video}->{int(cond_video_rows)}"
            )
        if int(cond_audio_rows) != old_cond_audio:
            rebuild = True
            reason = (
                f"{reason}; cond_audio_rows {old_cond_audio}->{int(cond_audio_rows)}"
            )

    if not rebuild:
        return normalized, False

    import inspect

    _layout_kwargs = {
        "keyframes": normalized.get("keyframes"),
        "refs": normalized.get("refs"),
    }
    _frame_count = normalized.get("frame_count")
    if _frame_count is not None:
        try:
            _sig = inspect.signature(mm.PackedLayout.__init__)
            _params = _sig.parameters
            _accepts_frame_count = (
                "frame_count" in _params
                or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in _params.values()
                )
            )
        except Exception:
            _accepts_frame_count = False

        if _accepts_frame_count:
            _layout_kwargs["frame_count"] = _frame_count
        else:
            if not normalized.get("longmedia_frame_count_compat_announced"):
                _lm_print(
                    "[MiniMaxH3 LongMedia][H3 LAYOUT REBUILD] "
                    "active PackedLayout.__init__ has no frame_count parameter; "
                    "rebuilding with keyframes/refs only",
                    flush=True,
                )

    fresh = mm.PackedLayout(
        int(text_len), int(latent_t), int(lat_h), int(lat_w), int(audio_t),
        **_layout_kwargs,
    )

    new_counts = _lm_h3_layout_counts(fresh)
    cond_video_rows, cond_audio_rows = _lm_h3_cond_tensor_counts(normalized)
    if new_counts is not None:
        target_v, cond_v, target_a, cond_a = new_counts
        if int(target_v) != int(target_video_rows):
            raise RuntimeError(
                "[H3 LAYOUT SELF-HEAL] rebuilt target video rows still disagree: "
                f"layout={target_v}, actual={target_video_rows}, "
                f"signature={expected_signature}"
            )
        if cond_video_rows and int(cond_v) != int(cond_video_rows):
            raise RuntimeError(
                "[H3 LAYOUT SELF-HEAL] visual conditioning geometry mismatch after rebuild: "
                f"layout_cond_rows={cond_v}, actual_cond_rows={cond_video_rows}. "
                "Incompatible VIDEO keyframes were already filtered; the remaining "
                "mismatch is in reference/payload metadata ordering or geometry."
            )
        if cond_audio_rows and int(cond_a) != int(cond_audio_rows):
            raise RuntimeError(
                "[H3 LAYOUT SELF-HEAL] audio conditioning geometry mismatch after rebuild: "
                f"layout_cond_rows={cond_a}, actual_cond_rows={cond_audio_rows}."
            )

    out = dict(normalized)
    out["layout"] = fresh
    if _frame_count is not None and not _accepts_frame_count:
        out["longmedia_frame_count_compat_announced"] = True
    out["longmedia_layout_selfheal"] = {
        "from_signature": tuple(old_signature) if old_signature is not None else None,
        "to_signature": expected_signature,
        "reason": str(reason),
    }

    _lm_print(
        "[MiniMaxH3 LongMedia][H3 LAYOUT REBUILD] "
        f"reason={reason}; signature={old_signature}->{expected_signature}; "
        f"target_video_rows={old_counts[0] if old_counts else 'n/a'}->{target_video_rows}; "
        f"metadata_normalized={metadata_changed}",
        flush=True,
    )
    return out, True


def _h3_segment_layout_guard_wrapper(executor, *args, **kwargs):
    """Repair MiniMax H3 text-tag length drift and validate PackedLayout cheaply.

    Some ComfyUI/H3 revisions carry presentation modality tags separately from
    the encoded context. LongMedia pre-encodes a different prompt per segment;
    after hybrid/reference metadata is reattached, a stale tag tensor can be a
    few rows shorter/longer than ``context.shape[1]``. Stock H3 then indexes the
    list up to the text segment length and raises an opaque ``IndexError``.

    The safe semantic for missing tail rows is the ordinary text modality (tag
    1); surplus rows are unreachable and are truncated. The payload is copied
    before modification so cached/shared segment conditioning remains immutable.
    """
    call_args = list(args)

    # DIFFUSION_MODEL currently receives (..., context, transformer_options,
    # minimax_payload=...), but avoid depending on a single ComfyUI revision.
    context = None
    if len(call_args) >= 3 and hasattr(call_args[2], 'shape'):
        context = call_args[2]
    if context is None:
        context = kwargs.get('context')

    text_len = None
    try:
        if context is not None and len(context.shape) >= 2:
            text_len = int(context.shape[1])
    except Exception:
        text_len = None

    payload = kwargs.get('minimax_payload')
    payload_location = ('kw', 'minimax_payload') if isinstance(payload, dict) else None
    if not isinstance(payload, dict):
        for idx, value in enumerate(call_args):
            if isinstance(value, dict) and any(
                key in value for key in ('text_token_tags', 'keyframes', 'refs', 'layout')
            ):
                payload = value
                payload_location = ('arg', idx)
                break

    if isinstance(payload, dict) and text_len is not None and text_len >= 0:
        # v0.4.81: validate/rebuild PackedLayout against the AV tensors entering
        # this exact forward. This catches stale layouts even when their cached
        # signature was forged/copied by another wrapper.
        runtime_x = call_args[0] if call_args else kwargs.get('x')
        payload, _layout_rebuilt = _lm_h3_rebuild_runtime_layout(
            payload, runtime_x, text_len, 'diffusion_model_preflight'
        )
        if _layout_rebuilt:
            if payload_location and payload_location[0] == 'arg':
                call_args[int(payload_location[1])] = payload
            else:
                kwargs['minimax_payload'] = payload

        tags = payload.get('text_token_tags')
        if tags is not None:
            try:
                flat = tags.reshape(-1) if hasattr(tags, 'reshape') else tags
                tag_len = int(flat.shape[0]) if hasattr(flat, 'shape') else len(flat)
                if tag_len != text_len:
                    new_payload = dict(payload)
                    if hasattr(flat, 'new_full'):
                        if tag_len < text_len:
                            pad = flat.new_full((text_len - tag_len,), 1)
                            import torch
                            fixed = torch.cat((flat, pad), dim=0)
                        else:
                            fixed = flat[:text_len].clone()
                    else:
                        fixed = list(flat[:text_len])
                        if len(fixed) < text_len:
                            fixed.extend([1] * (text_len - len(fixed)))
                    new_payload['text_token_tags'] = fixed
                    new_payload['longmedia_text_tag_alignment'] = {
                        'from': int(tag_len), 'to': int(text_len),
                    }
                    payload = new_payload
                    if payload_location and payload_location[0] == 'arg':
                        call_args[int(payload_location[1])] = new_payload
                    else:
                        kwargs['minimax_payload'] = new_payload
                    _lm_print(
                        '[MiniMaxH3 LongMedia][V315 SEGMENT TAG GUARD] '
                        f'text_token_tags {tag_len}->{text_len}; '
                        + ('padded missing tail as text(tag=1)' if tag_len < text_len else 'truncated unreachable tail'),
                        flush=True,
                    )
            except Exception as exc:
                _lm_print(
                    '[MiniMaxH3 LongMedia][V315 SEGMENT TAG GUARD] WARNING: '
                    f'could not inspect text_token_tags: {type(exc).__name__}: {exc}',
                    flush=True,
                )

        # Fail early with useful dimensions for a genuinely corrupt layout.
        layout = payload.get('layout')
        if layout is not None and hasattr(layout, 'segments'):
            try:
                segments = list(layout.segments)
                expected = 0
                for seg_idx, (a, b, kind) in enumerate(segments):
                    a, b = int(a), int(b)
                    if a != expected or b < a:
                        raise RuntimeError(
                            f'non-contiguous PackedLayout at segment {seg_idx} '
                            f'({kind}: {a}:{b}, expected start {expected})'
                        )
                    expected = b
                seq_len = int(getattr(layout, 'seq_len', expected))
                pos_len = int(layout.position_ids.shape[0]) if hasattr(layout, 'position_ids') else seq_len
                if expected != seq_len or pos_len != seq_len:
                    raise RuntimeError(
                        'PackedLayout size mismatch: '
                        f'segments_end={expected}, seq_len={seq_len}, position_ids={pos_len}, '
                        f'text_len={text_len}'
                    )
                if segments and str(segments[0][2]) == 'text':
                    packed_text_len = int(segments[0][1]) - int(segments[0][0])
                    if packed_text_len != text_len:
                        raise RuntimeError(
                            'PackedLayout/context text mismatch: '
                            f'layout_text={packed_text_len}, context_text={text_len}, '
                            f'tags={getattr(payload.get("text_token_tags"), "shape", None)}'
                        )
            except RuntimeError:
                raise
            except Exception as exc:
                _lm_print(
                    '[MiniMaxH3 LongMedia][V315 LAYOUT GUARD] WARNING: '
                    f'layout inspection skipped: {type(exc).__name__}: {exc}',
                    flush=True,
                )

    # Expose the exact self-healed PackedLayout to LongMedia attention.
    # FastH3 VSA needs tile geometry but must not own or replace the payload.
    if isinstance(payload, dict):
        _layout = payload.get('layout')
        _opts = kwargs.get('transformer_options')
        if not isinstance(_opts, dict) and len(call_args) >= 4 and isinstance(call_args[3], dict):
            _opts = call_args[3]
        if isinstance(_opts, dict):
            if _layout is not None:
                _opts['latentlab_h3_packed_layout'] = _layout
            else:
                _opts.pop('latentlab_h3_packed_layout', None)

    return executor(*call_args, **kwargs)




class _H3LookaheadPrefetchQueue:
    """Retained true-lookahead queue for AIMDO/VBAR H3 transformer blocks."""

    def __init__(self, blocks, device, depth=2):
        self.blocks = list(blocks)
        self.device = device
        self.depth = max(2, int(depth))
        self.index = 0
        self.entries = {}
        self.closed = False


def _h3_collect_vbar_modules(root):
    return [m for m in root.modules() if hasattr(m, '_v')]


def _h3_prefetch_one(queue, index):
    """Enqueue one future H3 block transfer without joining compute."""
    if index < 0 or index >= len(queue.blocks) or index in queue.entries:
        return
    import comfy.model_management as _mm
    import comfy.memory_management as _mem
    import comfy.ops as _ops

    root = queue.blocks[index]
    modules = _h3_collect_vbar_modules(root)
    if not modules:
        queue.entries[index] = (None, root, [])
        return

    registerable_size = 0
    for m in modules:
        registerable_size += _mem.vram_aligned_size([m.weight, m.bias])
        for param_key in ('weight', 'bias'):
            lowvram_fn = getattr(m, param_key + '_lowvram_function', None)
            if lowvram_fn is not None:
                registerable_size += lowvram_fn.memory_required()

    stream, _fully_faulted = _ops.cast_modules_with_vbar(
        modules, None, queue.device, None, True, return_faulted=True
    )
    if not _mm.args.fast_disk:
        _mm.ensure_pin_registerable(registerable_size)
    # No sync_stream here: transfer may overlap current block compute.
    queue.entries[index] = (stream, root, modules)


def _h3_cleanup_prefetch_entry(queue, index, guard_compute=False):
    entry = queue.entries.pop(index, None)
    if entry is None:
        return
    stream, root, modules = entry
    if guard_compute and stream is not None:
        # Match Comfy's stock lifetime rule: before unpinning the block that just
        # executed, make its transfer stream wait behind all compute already
        # queued on the current stream.  This prevents AIMDO from recycling a
        # VBAR page while the preceding block may still be using it asynchronously.
        import comfy.model_management as _mm
        current = _mm.current_stream(queue.device)
        if current is not None:
            stream.wait_stream(current)
    if modules:
        import comfy.model_prefetch as _mp
        _mp.cleanup_prefetched_modules(root, modules)


def _h3_close_lookahead_queue(queue):
    if queue.closed:
        return
    try:
        for stream, _root, _modules in list(queue.entries.values()):
            if stream is not None:
                stream.synchronize()
        for idx in list(queue.entries):
            _h3_cleanup_prefetch_entry(queue, idx)
    finally:
        queue.closed = True


def _h3_install_lookahead_prefetch(depth, registry):
    """Patch Comfy prefetch only during this H3 forward; always restore later."""
    import comfy.model_prefetch as _mp
    original_make = _mp.make_prefetch_queue
    original_pop = _mp.prefetch_queue_pop

    def make_prefetch_queue(blocks, device, transformer_options):
        if not bool(transformer_options.get('latentlab_h3_true_lookahead', False)):
            return original_make(blocks, device, transformer_options)
        try:
            import comfy.model_management as _mm
            if (_mm.NUM_STREAMS <= 0 or _mm.is_device_cpu(device)
                    or not _mm.device_supports_non_blocking(device)):
                return original_make(blocks, device, transformer_options)
        except Exception:
            return original_make(blocks, device, transformer_options)
        q = _H3LookaheadPrefetchQueue(blocks, device, depth=depth)
        registry.append(q)
        return q

    def prefetch_queue_pop(queue, device, module, dtype=None, core=None, enable_graph=False, generator=None):
        if not isinstance(queue, _H3LookaheadPrefetchQueue):
            return original_pop(queue, device, module, dtype=dtype, core=core,
                                enable_graph=enable_graph, generator=generator)
        if module is None:
            _h3_close_lookahead_queue(queue)
            return

        i = int(queue.index)
        if i > 0:
            _h3_cleanup_prefetch_entry(queue, i - 1, guard_compute=True)

        _h3_prefetch_one(queue, i)
        stream, _root, _modules = queue.entries[i]
        if stream is not None:
            import comfy.model_management as _mm
            _mm.sync_stream(device, stream)

        # True one-ahead for depth=2: enqueue i+1 before block i computes.
        end = min(len(queue.blocks), i + queue.depth)
        for j in range(i + 1, end):
            _h3_prefetch_one(queue, j)

        queue.index = i + 1
        if core is not None:
            core()

    _mp.make_prefetch_queue = make_prefetch_queue
    _mp.prefetch_queue_pop = prefetch_queue_pop
    return _mp, original_make, original_pop

def _h3_runtime_prefetch_wrapper(executor, *args, _bound_residency_state=None, _bound_residency_patcher=None, **kwargs):
    """Own H3/AIMDO runtime residency policy at the DIFFUSION_MODEL boundary.

    v0.4.40 adds a persistent resident-window policy for recent AIMDO + W4A8
    on 15-18.5 GB NVIDIA cards.  The important invariant is *monotonic* VBAR
    state across denoise forwards: we set a conservative watermark_limit once
    and never call ``prioritize()`` again on every denoise step.  AIMDO itself
    documents that ``prioritize()`` resets the offload watermark to the end of
    the VBAR; doing that repeatedly causes exactly the fault-in/evict sawtooth
    seen on oversized H3 checkpoints.

    The watermark limit protects only the low-address prefix (AIMDO's intended
    high-priority region).  Remaining weights still stream normally, so model
    sizes above physical VRAM remain supported.  No tensor math is changed.

    The legacy hard-gate path is retained for older/unsafe runtimes.
    """

    call_args = list(args)

    candidates = []
    for idx, value in enumerate(call_args):
        if isinstance(value, dict):
            candidates.append(('arg', idx, value))

    for key, value in kwargs.items():
        if isinstance(value, dict):
            candidates.append(('kw', key, value))

    transformer_options = None
    location = None

    # Strong marker first: survives BaseModel's transformer_options.copy().
    for kind, key, value in candidates:
        if (
            'latentlab_h3_runtime_backend' in value
            or 'latentlab_disable_dynamic_vbar_prefetch' in value
        ):
            transformer_options = value
            location = f'{kind}[{key}]'
            break

    # Fallback to recognizable transformer-options structure.
    if transformer_options is None:
        for kind, key, value in candidates:
            if any(
                marker in value
                for marker in (
                    'patches_replace',
                    'wrappers',
                    'prefetch_dynamic_vbars',
                    'sigmas',
                )
            ):
                transformer_options = value
                location = f'{kind}[{key}]'
                break

    if transformer_options is None:
        _lm_print(
            '[MiniMaxH3 LongMedia][INT8 PREFETCH V7] WARNING: '
            'DIFFUSION_MODEL wrapper could not locate transformer_options; '
            f'dict_candidates={[(k, x) for k, x, _ in candidates]}',
            flush=True,
        )
        return executor(*call_args, **kwargs)

    backend = str(
        transformer_options.get(
            'latentlab_h3_runtime_backend',
            transformer_options.get('model_runtime_backend', 'unknown'),
        )
    ).lower()
    disable_requested = bool(
        transformer_options.get(
            'latentlab_disable_dynamic_vbar_prefetch', False
        )
    )

    before = transformer_options.get('prefetch_dynamic_vbars', '<missing>')

    if disable_requested:
        transformer_options['prefetch_dynamic_vbars'] = False

        if not transformer_options.get(
            'latentlab_prefetch_disable_announced_v8', False
        ):
            _lm_print(
                '[MiniMaxH3 LongMedia][PREFETCH HARD-GATE] '
                f'located transformer_options at {location}; '
                f'prefetch_dynamic_vbars {before!r}->False AFTER BaseModel override; '
                f'backend={backend}; synchronous one-block demand loading active',
                flush=True,
            )
            transformer_options['latentlab_prefetch_disable_announced_v8'] = True

    # v0.4.40: persistent resident window.  Repeated vbar.prioritize() is
    # explicitly forbidden here: AIMDO's prioritize resets the watermark to the
    # end of the VBAR, making each denoise forward re-attempt the whole model and
    # recreating the regular PCIe/VBAR churn pattern.  Instead, protect a bounded
    # low-address prefix with watermark_limit and let the tail stream.
    residency_state = transformer_options.get('latentlab_h3_residency_state') or _bound_residency_state
    residency_patcher = transformer_options.get('latentlab_h3_residency_patcher') or _bound_residency_patcher
    if isinstance(residency_state, dict) and residency_patcher is not None:
        fwd = int(residency_state.get('vbar_forward_count', 0)) + 1
        residency_state['vbar_forward_count'] = fwd
        persistent_window = bool(residency_state.get('vbar_persistent_window_enabled', False))
        try:
            vbar_get = getattr(residency_patcher, '_vbar_get', None)
            vbar = vbar_get(create=False) if callable(vbar_get) else None
            if vbar is not None:
                loaded_before = int(vbar.loaded_size()) if hasattr(vbar, 'loaded_size') else 0
                residency_state['vbar_forward_loaded_before'] = loaded_before
                if persistent_window and not residency_state.get('vbar_window_armed', False):
                    target_b = int(residency_state.get('vbar_window_target_bytes', 0) or 0)
                    if target_b > 0 and hasattr(vbar, 'set_watermark_limit'):
                        # Arm once.  set_watermark_limit does not fault data in; it
                        # only prevents the protected prefix from being evicted
                        # when those pages are subsequently touched.
                        vbar.set_watermark_limit(target_b)
                        residency_state['vbar_window_armed'] = True
                        residency_state['vbar_window_armed_forward'] = fwd
                        residency_state['vbar_window_loaded_at_arm'] = loaded_before
                        _lm_print(
                            '[MiniMaxH3 LongMedia][AIMDO PERSISTENT WINDOW] '
                            f'ARMED forward={fwd} target={target_b/(1024.0**2):.0f}MB '
                            f'loaded={loaded_before/(1024.0**2):.0f}MB; '
                            'watermark_limit fixed; per-forward prioritize DISABLED',
                            flush=True,
                        )
                elif persistent_window and fwd <= 3:
                    _lm_print(
                        '[MiniMaxH3 LongMedia][AIMDO PERSISTENT WINDOW] '
                        f'HOLD forward={fwd} loaded={loaded_before/(1024.0**2):.0f}MB; '
                        'no watermark reset',
                        flush=True,
                    )
                elif disable_requested and fwd <= 3:
                    _lm_print(
                        '[MiniMaxH3 LongMedia][AIMDO GUARDED STREAM] '
                        f'forward={fwd} loaded={loaded_before/(1024.0**2):.0f}MB; '
                        'prefetch hard-gated; no forced prioritize',
                        flush=True,
                    )
        except Exception as exc:
            residency_state['vbar_forward_error'] = f'{type(exc).__name__}: {exc}'
            if not residency_state.get('vbar_forward_error_announced'):
                residency_state['vbar_forward_error_announced'] = True
                _lm_print(
                    '[MiniMaxH3 LongMedia][AIMDO WINDOW] disabled: ' + residency_state['vbar_forward_error'],
                    flush=True,
                )

    # Diagnostic invariant: when out-of-core streaming requested this MUST be
    # False immediately before MiniMaxH3Model._forward calls make_prefetch_queue().
    if disable_requested:
        effective = transformer_options.get('prefetch_dynamic_vbars', '<missing>')
        if not transformer_options.get('latentlab_prefetch_effective_announced_v8', False):
            _lm_print(
                '[MiniMaxH3 LongMedia][PREFETCH HARD-GATE CHECK] '
                f'effective prefetch_dynamic_vbars={effective!r} '
                f'at DIFFUSION_MODEL boundary ({location})',
                flush=True,
            )
            transformer_options['latentlab_prefetch_effective_announced_v8'] = True

    # v0.4.42: Comfy's stock H3 queue faults the same block immediately
    # before it executes and then joins the transfer stream.  For resident-window
    # W4A8, install a retained depth-2 ring so block i+1 transfers while block i
    # computes.  Restore global Comfy functions in finally.
    lookahead = bool(transformer_options.get('latentlab_h3_true_lookahead', False))
    prefetch_registry = []
    patch_tuple = None
    if lookahead:
        depth = max(2, int(transformer_options.get('latentlab_h3_prefetch_depth', 2) or 2))
        patch_tuple = _h3_install_lookahead_prefetch(depth, prefetch_registry)
        if isinstance(residency_state, dict) and not residency_state.get('lookahead_announced', False):
            residency_state['lookahead_announced'] = True
            _lm_print(
                '[MiniMaxH3 LongMedia][TRUE LOOKAHEAD] '
                f'ACTIVE depth={depth} (current + {depth-1} future); '
                'future VBAR transfer overlaps current compute; retained-pin ring enabled',
                flush=True,
            )
    try:
        _h3_result = executor(*call_args, **kwargs)
    finally:
        for q in list(prefetch_registry):
            try:
                _h3_close_lookahead_queue(q)
            except Exception:
                pass
        if patch_tuple is not None:
            _mp, _orig_make, _orig_pop = patch_tuple
            _mp.make_prefetch_queue = _orig_make
            _mp.prefetch_queue_pop = _orig_pop

    if disable_requested and isinstance(residency_state, dict):
        residency_state['vbar_first_forward_complete'] = True
    return _h3_result



def _detect_h3_model_runtime(model_patcher):
    """Best-effort, side-effect-free diffusion quantization/runtime detector.

    IMPORTANT: logical module.weight dtype is not sufficient for Comfy quantized
    models. NVFP4/INT8 may expose BF16 logical weights while the actual execution
    format is carried by QuantizedTensor/layout metadata.
    """
    profile = {
        'backend': 'unknown',
        'model_class': None,
        'diffusion_class': None,
        'weight_dtypes': {},
        'weight_classes': {},
        'layout_types': {},
        'quant_evidence': [],
        'sampled_modules': 0,
        'quantized_weight_count': 0,
        'error': None,
    }

    try:
        try:
            from comfy.quant_ops import QuantizedTensor
        except Exception:
            QuantizedTensor = ()

        base_model = getattr(model_patcher, 'model', None)
        diffusion = getattr(base_model, 'diffusion_model', None)
        if diffusion is None:
            diffusion = getattr(model_patcher, 'diffusion_model', None)

        if base_model is not None:
            profile['model_class'] = (
                f'{type(base_model).__module__}.{type(base_model).__name__}'
            )
        if diffusion is not None:
            profile['diffusion_class'] = (
                f'{type(diffusion).__module__}.{type(diffusion).__name__}'
            )

        root = diffusion if diffusion is not None else base_model
        if root is None:
            return profile

        dtype_counts = {}
        weight_class_counts = {}
        layout_counts = {}
        evidence = []
        evidence_seen = set()

        def _add_evidence(item):
            item = str(item)
            if item not in evidence_seen and len(evidence) < 40:
                evidence.append(item)
                evidence_seen.add(item)

        def _layout_name(value):
            if value is None:
                return None
            if isinstance(value, type):
                return f'{value.__module__}.{value.__name__}'
            cls = type(value)
            # For instances, include both concrete type and readable repr/name.
            name = getattr(value, '__name__', None)
            if name:
                return f'{cls.__module__}.{cls.__name__}:{name}'
            return f'{cls.__module__}.{cls.__name__}:{str(value)[:120]}'

        for idx, (name, module) in enumerate(root.named_modules()):
            if idx >= 3000:
                break
            profile['sampled_modules'] += 1

            mod_cls = f'{type(module).__module__}.{type(module).__name__}'
            mod_low = mod_cls.lower()

            # Module-level quant layout metadata used by mixed-precision ops.
            layout_value = None
            for attr in ('layout_type', 'weight_layout', 'quant_layout'):
                try:
                    candidate = getattr(module, attr, None)
                except Exception:
                    candidate = None
                if candidate is not None:
                    layout_value = candidate
                    lname = _layout_name(candidate)
                    layout_counts[lname] = int(layout_counts.get(lname, 0)) + 1
                    _add_evidence(f'{name or "<root>"}.{attr}={lname}')

            weight = None
            try:
                weight = getattr(module, 'weight', None)
            except Exception:
                weight = None

            if weight is not None:
                wcls = f'{type(weight).__module__}.{type(weight).__name__}'
                weight_class_counts[wcls] = int(weight_class_counts.get(wcls, 0)) + 1

                dtype = getattr(weight, 'dtype', None)
                if dtype is not None:
                    key = str(dtype)
                    dtype_counts[key] = int(dtype_counts.get(key, 0)) + 1

                # Direct QuantizedTensor detection.
                is_quantized_tensor = False
                try:
                    if QuantizedTensor:
                        is_quantized_tensor = isinstance(weight, QuantizedTensor)
                except Exception:
                    is_quantized_tensor = False

                weight_low = wcls.lower()
                if is_quantized_tensor or 'quantizedtensor' in weight_low:
                    profile['quantized_weight_count'] += 1
                    _add_evidence(
                        f'{name or "<root>"}.weight_class={wcls},dtype={dtype}'
                    )

                    # QuantizedTensor implementations may keep layout as an
                    # instance or a class under different attribute names.
                    for attr in (
                        'layout', 'layout_type', '_layout',
                        'tensor_layout', 'quant_layout',
                    ):
                        try:
                            qlayout = getattr(weight, attr, None)
                        except Exception:
                            qlayout = None
                        if qlayout is not None:
                            lname = _layout_name(qlayout)
                            layout_counts[lname] = int(layout_counts.get(lname, 0)) + 1
                            _add_evidence(
                                f'{name or "<root>"}.weight.{attr}={lname}'
                            )

                # Some quant tensors expose an internal data tensor with the real
                # storage dtype (uint8 for NVFP4, int8 for tensorwise INT8).
                for attr in ('qdata', 'data', '_data'):
                    try:
                        qdata = getattr(weight, attr, None)
                    except Exception:
                        qdata = None
                    qdtype = getattr(qdata, 'dtype', None)
                    if qdtype is not None and str(qdtype) in (
                        'torch.uint8', 'torch.int8',
                        'torch.float8_e4m3fn', 'torch.float8_e5m2',
                    ):
                        _add_evidence(
                            f'{name or "<root>"}.weight.{attr}.dtype={qdtype}'
                        )

            # Quantization scales on module are strong evidence even if weight
            # has already been exposed as logical BF16.
            present_scale_attrs = []
            for attr in (
                'weight_scale', 'weight_scale_2', 'input_scale',
                'scale', 'scales',
            ):
                try:
                    value = getattr(module, attr, None)
                except Exception:
                    value = None
                if value is not None:
                    present_scale_attrs.append(attr)
            if present_scale_attrs:
                _add_evidence(
                    f'{name or "<root>"}.scale_attrs={",".join(present_scale_attrs)}'
                )

            # Class-name evidence remains useful for custom loaders.
            if any(term in mod_low for term in (
                'nvfp4', 'int8', 'fp8', 'quant', 'gguf',
                'torchao', 'quanto', 'bitsandbytes',
            )):
                _add_evidence(f'{name or "<root>"}:module={mod_cls}')

        profile['weight_dtypes'] = dtype_counts
        profile['weight_classes'] = weight_class_counts
        profile['layout_types'] = layout_counts
        profile['quant_evidence'] = evidence

        searchable = ' '.join(
            [
                str(profile.get('model_class', '')),
                str(profile.get('diffusion_class', '')),
                *layout_counts.keys(),
                *weight_class_counts.keys(),
                *evidence,
            ]
        ).lower()

        # V28: classify the concrete quant variant independently from the broad
        # backend.  W4A8 checkpoints can also expose TensorWiseINT8Layout, so
        # backend='int8' alone is not sufficient to select a safe activation
        # policy.
        if any(marker in searchable for marker in (
            'asymw4a8int8layout', 'asymw4a8', 'asym_w4a8', 'w4a8',
            'w4a8int8', 'w4a8_mixed',
        )):
            profile['quant_variant'] = 'w4a8'
        elif any(marker in searchable for marker in (
            'tensorcoreconvrotw4a4layout', 'convrot_w4a4', 'w4a4',
        )):
            profile['quant_variant'] = 'convrot-w4a4'
        elif 'tensorwiseint8layout' in searchable or 'int8layout' in searchable:
            profile['quant_variant'] = 'tensorwise-int8'
        else:
            profile['quant_variant'] = None

        # Strongest evidence: named quant layout/class.
        if (
            'tensorcorenvfp4layout' in searchable
            or 'nvfp4layout' in searchable
            or 'nvfp4' in searchable
        ):
            backend = 'nvfp4'
        elif (
            'tensorwiseint8layout' in searchable
            or 'int8layout' in searchable
            or 'int8_tensorwise' in searchable
        ):
            backend = 'int8'
        elif (
            'tensorcoreconvrotw4a4layout' in searchable
            or 'convrot_w4a4' in searchable
        ):
            backend = 'int8-convrot-w4a4'
        elif (
            'tensorcorefp8' in searchable
            or 'float8' in searchable
            or 'fp8layout' in searchable
        ):
            backend = 'fp8'
        elif int(dtype_counts.get('torch.int8', 0)) > 0:
            backend = 'int8'
        elif any(x in searchable for x in (
            'gguf', 'quantizedtensor', 'quant', 'torchao',
            'quanto', 'bitsandbytes',
        )):
            backend = 'quantized-other'
        elif int(dtype_counts.get('torch.bfloat16', 0)) > 0:
            backend = 'bf16'
        elif int(dtype_counts.get('torch.float16', 0)) > 0:
            backend = 'fp16'
        elif int(dtype_counts.get('torch.float32', 0)) > 0:
            backend = 'fp32'
        else:
            backend = 'unknown'

        profile['backend'] = backend
        return profile

    except Exception as exc:
        profile['error'] = f'{type(exc).__name__}: {exc}'
        return profile



def _announce_h3_model_runtime(profile):
    evidence = list(profile.get('quant_evidence') or [])
    _lm_print(
        '[MiniMaxH3 LongMedia][MODEL RUNTIME V2] '
        f'backend={profile.get("backend", "unknown")}, '
        f'diffusion={profile.get("diffusion_class")}, '
        f'weight_dtypes={profile.get("weight_dtypes") or {}}, '
        f'quantized_weights={profile.get("quantized_weight_count", 0)}, '
        f'quant_variant={profile.get("quant_variant")}, '
        f'sampled_modules={profile.get("sampled_modules", 0)}',
        flush=True,
    )
    if profile.get('layout_types'):
        _lm_print(
            '[MiniMaxH3 LongMedia][MODEL RUNTIME LAYOUTS] '
            + str(profile['layout_types']),
            flush=True,
        )
    if profile.get('weight_classes'):
        # Only the most common classes are useful in console.
        classes = sorted(
            profile['weight_classes'].items(),
            key=lambda kv: (-int(kv[1]), kv[0]),
        )[:12]
        _lm_print(
            '[MiniMaxH3 LongMedia][MODEL RUNTIME WEIGHT CLASSES] '
            + str(dict(classes)),
            flush=True,
        )
    if evidence:
        _lm_print(
            '[MiniMaxH3 LongMedia][MODEL RUNTIME EVIDENCE V2] '
            + ' | '.join(evidence[:20]),
            flush=True,
        )
    if profile.get('error'):
        _lm_print(
            '[MiniMaxH3 LongMedia][MODEL RUNTIME WARNING] '
            + str(profile['error']),
            flush=True,
        )




def _h3_runtime_auto_policy(
    backend,
    *,
    quant_variant=None,
    chunk_tokens,
    sol_qkv_chunk_tokens,
    sol_out_proj_chunk_tokens,
    vram_activation_reserve_mb,
):
    """Return conservative backend-aware startup settings.

    NVFP4 is the proven reference path and is intentionally left untouched.
    INT8/BF16-class backends start with more activation headroom and smaller
    activation chunks; block0 AUTO VRAM still adapts guards after the first
    successful block.
    """
    backend = str(backend or 'unknown').lower()

    policy = {
        'backend': backend,
        'name': 'user-defaults',
        'chunk_tokens': int(chunk_tokens),
        'sol_qkv_chunk_tokens': int(sol_qkv_chunk_tokens),
        'sol_out_proj_chunk_tokens': int(sol_out_proj_chunk_tokens),
        'vram_activation_reserve_mb': int(vram_activation_reserve_mb),
    }

    if backend == 'nvfp4':
        policy['name'] = 'nvfp4-proven'
        return policy

    if backend in ('int8', 'int8-convrot-w4a4'):
        quant_variant = str(quant_variant or '').lower()
        policy['quant_variant'] = quant_variant or None
        if quant_variant == 'w4a8':
            # V34: W4A8 throughput pass. The CUDA W4A8 kernel dequantizes grouped
            # int4 weights as part of each Linear invocation, so long-media throughput
            # depends strongly on reducing projection-call count. Keep native kitchen
            # math, but permit larger streamed projection chunks when headroom allows.
            policy['name'] = 'w4a8-cuda-throughput'
            policy['vram_activation_reserve_mb'] = max(
                1536, min(int(vram_activation_reserve_mb), 2304)
            )
            policy['chunk_tokens'] = min(int(chunk_tokens), 8192)
        else:
            policy['name'] = 'int8-native-resident'
            policy['vram_activation_reserve_mb'] = int(vram_activation_reserve_mb)
            policy['chunk_tokens'] = min(int(chunk_tokens), 16384)
        if int(sol_qkv_chunk_tokens) > 0:
            # v0.4.76 V8: explicit sampler QKV ownership.
            # Native TensorWise INT8 is allowed to request up to 16K just like
            # W4A8. Real block0 headroom decides whether giant sequences keep
            # 16K or safely fall back to 8K.
            policy['sol_qkv_chunk_tokens'] = min(
                int(sol_qkv_chunk_tokens), 16384
            )
        if int(sol_out_proj_chunk_tokens) > 0:
            policy['sol_out_proj_chunk_tokens'] = min(
                int(sol_out_proj_chunk_tokens), 16384
            )
        return policy

    if backend in ('bf16', 'fp16', 'fp32'):
        policy['name'] = f'{backend}-conservative'
        policy['vram_activation_reserve_mb'] = max(
            int(vram_activation_reserve_mb), 6144
        )
        policy['chunk_tokens'] = min(int(chunk_tokens), 12288)
        if int(sol_qkv_chunk_tokens) > 0:
            policy['sol_qkv_chunk_tokens'] = min(
                int(sol_qkv_chunk_tokens), 8192
            )
        if int(sol_out_proj_chunk_tokens) > 0:
            policy['sol_out_proj_chunk_tokens'] = min(
                int(sol_out_proj_chunk_tokens), 12288
            )
        return policy

    if backend in ('fp8', 'quantized-other'):
        policy['name'] = f'{backend}-conservative'
        policy['vram_activation_reserve_mb'] = max(
            int(vram_activation_reserve_mb), 5120
        )
        policy['chunk_tokens'] = min(int(chunk_tokens), 16384)
        if int(sol_out_proj_chunk_tokens) > 0:
            policy['sol_out_proj_chunk_tokens'] = min(
                int(sol_out_proj_chunk_tokens), 16384
            )
        return policy

    return policy




def _closure_bindings(fn):
    """Return closure freevars without depending on the external node package name."""
    try:
        names = tuple(getattr(fn.__code__, 'co_freevars', ()) or ())
        cells = tuple(getattr(fn, '__closure__', ()) or ())
        return {name: cell.cell_contents for name, cell in zip(names, cells)}
    except Exception:
        return {}


def _extract_external_sla_config(override):
    """Read Comfy H3 SLA settings from its override closure when available."""
    cfg = {
        'sparsity_ratio': 0.90,
        'block_q': 64,
        'block_k': 64,
        'protect_audio': True,
        'plugin_state': None,
    }
    if override is None:
        return cfg
    b = _closure_bindings(override)
    try:
        cfg['sparsity_ratio'] = float(b.get('sparsity_ratio', cfg['sparsity_ratio']))
        cfg['block_q'] = int(b.get('blkq', cfg['block_q']))
        cfg['block_k'] = int(b.get('blkk', cfg['block_k']))
        cfg['protect_audio'] = bool(b.get('protect_audio', cfg['protect_audio']))
        st = b.get('state')
        if isinstance(st, dict):
            cfg['plugin_state'] = st
    except Exception:
        pass
    return cfg


def _execute_h3_sla_zero_copy(attn, h, rope_freqs, transformer_options, state, measure=None):
    """Run SLA directly inside the LongMedia H3 block without Comfy hook loss.

    The fused QKV projection is kept as one strided allocation.  The sparse
    kernel reads Q/K/V through their native strides and writes into ``h`` after
    QKV projection has consumed it.  The output projection is streamed and
    copied back into the same buffer, removing the fatal full-size ``o_s``
    allocation from the external SLA implementation.
    """
    from .sla_fastpath import SLAConfig, build_lut, sparse_attention_into
    import comfy.quant_ops

    meas = measure or (lambda _name, fn: fn())
    s = int(h.shape[0])
    heads = int(attn.heads)
    head_dim = int(attn.head_dim)
    inner = heads * head_dim
    cfg_raw = dict(state.get('external_sla_config') or {})
    _cfg_block_q = int(cfg_raw.get('block_q', 64))
    _cfg_block_k = int(cfg_raw.get('block_k', 64))
    # v0.4.45: external SLA geometry is authoritative.  The 0.4.44 hidden
    # 64x64 -> 128x64 SM120 override made A/B tests impossible and, more
    # importantly, the release-guard trace proved that the 128x64 run fell out
    # of SLA after block 0 because a transient CUDA OOM switched the whole
    # forward to embedded Sol.  Never silently rewrite the user's SLA node.
    _sm = None
    try:
        _sm = tuple(torch.cuda.get_device_capability(h.device)) if h.is_cuda else None
    except Exception:
        _sm = None
    _sla_geom_reason = 'external_authoritative'

    cfg = SLAConfig(
        sparsity_ratio=float(cfg_raw.get('sparsity_ratio', 0.90)),
        block_q=_cfg_block_q,
        block_k=_cfg_block_k,
        protect_audio=bool(cfg_raw.get('protect_audio', True)),
        plugin_state=cfg_raw.get('plugin_state'),
    )
    if cfg.block_q not in (64, 128) or cfg.block_k not in (64, 128):
        raise RuntimeError(f'unsupported external SLA block geometry {cfg.block_q}x{cfg.block_k}')
    state['sla_effective_geometry'] = f'{cfg.block_q}x{cfg.block_k}'
    state['sla_effective_geometry_reason'] = _sla_geom_reason
    if not state.get('sla_geometry_announced'):
        state['sla_geometry_announced'] = True
        _lm_print(
            '[MiniMaxH3 LongMedia][SLA GEOMETRY] '
            f'external={int(cfg_raw.get("block_q", 64))}x{int(cfg_raw.get("block_k", 64))} '
            f'-> effective={cfg.block_q}x{cfg.block_k}; sm={_sm}; S={s}; '
            'user-selected geometry preserved exactly',
            flush=True,
        )

    state['sla_zero_copy_h_overwritten'] = False
    state['sla_zero_copy_phase'] = 'qkv_proj'
    qkv = meas('sla_qkv_proj', lambda: attn.qkv_proj(h))
    q, k, v = qkv.split(inner, dim=-1)
    q = q.view(1, s, heads, head_dim)
    k = k.view(1, s, heads, head_dim)
    v = v.view(1, s, heads, head_dim)

    def _norm_rope():
        nonlocal q, k
        if rope_freqs is not None:
            qw = comfy.model_management.cast_to(attn.q_norm.weight, device=h.device)
            kw = comfy.model_management.cast_to(attn.k_norm.weight, device=h.device)
            rot = int(rope_freqs.shape[-3] * 2)
            if comfy.model_management.in_training:
                q, k = comfy.quant_ops.ck.rms_rope_split_half(
                    q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
                )
            else:
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
                )
            return q, k
        # Non-RoPE path is rare for H3; avoid assuming in-place RMS support.
        return attn.q_norm(q), attn.k_norm(k)

    state['sla_zero_copy_phase'] = 'rms_rope'
    q, k = meas('sla_rms_rope', _norm_rope)

    prefix = 0
    if cfg.protect_audio:
        try:
            prefix = int((transformer_options or {}).get('_h3sla_prefix', 0) or 0)
        except Exception:
            prefix = 0
        if prefix <= 0:
            span = (transformer_options or {}).get('latentlab_sol_h3_video_span')
            if span is not None:
                try:
                    prefix = max(0, int(span[0]))
                except Exception:
                    prefix = 0
        if prefix >= s:
            prefix = 0

    state['sla_zero_copy_phase'] = 'block_map'
    lut, topk, pinned = meas(
        'sla_block_map',
        lambda: build_lut(
            q, k,
            sparsity_ratio=cfg.sparsity_ratio,
            block_q=cfg.block_q,
            block_k=cfg.block_k,
            protect_upto=prefix,
        ),
    )

    # H3's residual width is NOT the attention inner width (e.g. 5376 vs
    # 56*128=7168 on the current model), so ``h`` cannot be reinterpreted as
    # BLHD attention output storage.  Reuse the Q slice of the fused QKV buffer
    # instead: each Triton program loads its own Q block completely before it
    # stores the corresponding output block, and no other program reads those Q
    # rows. K/V live in disjoint slices, so this is safe and still allocates no
    # full-size ``o_s`` tensor.
    state['sla_zero_copy_h_overwritten'] = False
    raw_out = q
    state['sla_zero_copy_phase'] = 'sparse_kernel'
    meas(
        'sla_kernel_zero_copy',
        lambda: sparse_attention_into(
            q, k, v, raw_out, lut, topk, cfg.block_q, cfg.block_k
        ),
    )
    del k, v, lut

    # Attention output now occupies the Q slice of qkv as [1,S,H,D]. Stream the
    # 7168->hidden out-projection directly into the original residual-width h.
    # Keep q/qkv alive until projection is complete because q aliases qkv.
    attn_flat = raw_out.view(s, inner)
    state['sla_zero_copy_phase'] = 'out_proj'
    proj_chunk = int(state.get('sol_out_proj_chunk_tokens', 0) or 0)
    if proj_chunk <= 0:
        proj_chunk = 8192
    proj_chunk = max(256, int(proj_chunk))
    for start in range(0, s, proj_chunk):
        end = min(s, start + proj_chunk)
        projected = attn.out_proj(attn_flat[start:end])
        h[start:end].copy_(projected)
        del projected
    del attn_flat, raw_out, q, qkv

    pst = cfg.plugin_state
    if isinstance(pst, dict):
        # Keep the external node's end-of-run diagnostic truthful even though
        # LongMedia executes its SLA math directly rather than through the hook.
        pst['calls'] = int(pst.get('calls', 0) or 0) + 1
        pst['seq'] = s
        pst['kept'] = int(topk)
        pst['blocks'] = (s + cfg.block_k - 1) // cfg.block_k
        pst['pinned'] = int(pinned)
        pst['backend'] = 'LongMedia zero-copy SLA'

    state['sla_zero_copy_phase'] = 'done'
    state['sla_zero_copy_calls'] = int(state.get('sla_zero_copy_calls', 0) or 0) + 1
    if not state.get('sla_zero_copy_announced'):
        _lm_print(
            '[MiniMaxH3 LongMedia][ZERO-COPY SLA] ACTIVE: '
            f'S={s} sparsity={cfg.sparsity_ratio:.2f} BLK={cfg.block_q}x{cfg.block_k} geom={_sla_geom_reason} '
            f'topk={topk} pinned={pinned}; bounded_fp32_pool+head_chunked_lut, fused-QKV strided, attention->norm1 buffer, '
            f'out-proj streamed <= {proj_chunk}; no full-size SLA o_s allocation',
            flush=True,
        )
        state['sla_zero_copy_announced'] = True
    return h


def _detect_external_h3_sla(transformer_options=None):
    """Detect ComfyUI-H3-SLA-Attention even when it patches attention globally.

    The SLA plugin does not necessarily expose its override through the current
    ``transformer_options`` object.  In that case v0.4.34 incorrectly classified
    the backend as stock/existing and underestimated peak activation memory.
    """
    import sys

    opts = transformer_options or {}
    candidates = [
        opts.get('optimized_attention_override'),
        opts.get('attention_override'),
    ]
    for fn in candidates:
        if fn is None:
            continue
        mod = str(getattr(fn, '__module__', '') or '').lower()
        qual = str(getattr(fn, '__qualname__', '') or '').lower()
        if mod == 'sla.patch' or mod.endswith('.sla.patch') or ('sla' in mod and 'patch' in mod):
            return True, mod or qual

    # The plugin installs a process-global optimized-attention wrapper.  Detect
    # the loaded module as a second source of truth instead of assuming that the
    # wrapper is serialized into transformer_options.
    for name, module in tuple(sys.modules.items()):
        lname = str(name).lower()
        if lname == 'sla.patch' or lname.endswith('.sla.patch'):
            path = str(getattr(module, '__file__', '') or '')
            return True, path or name
    return False, None


def _h3_sequence_tokens(x):
    """Return packed H3 sequence length for either [S,D] or [B,S,D] layouts."""
    try:
        if int(x.ndim) >= 3:
            return int(x.shape[-2])
        return int(x.shape[0])
    except Exception:
        return 1

def _estimate_existing_attention_peak_bytes(token_count, hidden_dim, element_size, *, external_sla=False):
    """Conservative pre-QKV activation estimate for full-sequence attention.

    H3 self-attention materializes Q/K/V at hidden width.  The external SLA
    implementation additionally materializes a full-size output tensor with
    ``torch.empty_like(q)`` before projection.  Account for norm/rotary/routing
    slack as another full-width tensor rather than relying on a token threshold.
    This is intentionally allocator-facing: weights/reserved CUDA memory are
    handled separately via ``mem_get_info``.
    """
    s = max(1, int(token_count))
    d = max(1, int(hidden_dim))
    e = max(1, int(element_size))
    one = s * d * e
    # Q + K + V + normalized input. External SLA needs another O ~= Q.
    full_tensors = 5 if external_sla else 4
    # LUT/top-k, rope temporaries, allocator granularity and projection overlap.
    workspace = max(256 * 1024**2, int(one * (0.35 if external_sla else 0.20)))
    return int(one * full_tensors + workspace), int(one)


def _auto_select_h3_attention_mode(token_count, state, *, hidden_dim=None, element_size=None):
    """Choose existing vs bounded Sol before QKV allocation using real VRAM budget.

    v0.4.35 keeps the old 120k/180k token guess.  AUTO compares the estimated
    full-sequence attention peak against CUDA driver-free memory, with an explicit
    safety reserve.  The result is latched for segmented LongMedia consistency.
    """
    latched = state.get('auto_attention_selected_mode')
    if latched in ('existing', 'sol'):
        return str(latched), f'latched from first pass -> {latched}'

    s = max(1, int(token_count))
    d = int(hidden_dim or state.get('current_hidden_dim', 7168) or 7168)
    e = int(element_size or state.get('current_element_size', 2) or 2)
    external_sla = bool(state.get('external_sla_detected', False))
    try:
        free_b, total_b = torch.cuda.mem_get_info(torch.cuda.current_device())
    except Exception:
        total_b = int(16 * 1024**3)
        free_b = int(total_b * 0.25)

    estimated_b, one_b = _estimate_existing_attention_peak_bytes(
        s, d, e, external_sla=external_sla
    )
    reserve_mb = max(768, int(state.get('vram_activation_reserve_mb', 2048) or 2048))
    # The configured activation reserve can be very large for quantized models;
    # only the portion relevant to the imminent attention allocation is held back.
    reserve_b = min(int(reserve_mb * 1024**2), int(total_b * 0.18))
    usable_b = max(0, int(free_b) - reserve_b)

    # External SLA's o_s allocation is known from the crash trace to be fatal
    # when it consumes most remaining headroom. Require the whole estimated
    # attention working set to fit before allowing that backend.
    safe = estimated_b <= usable_b

    # v0.4.35 deterministic 16-GB guard.  Driver-free memory sampled before a
    # block is optimistic because later Q/K/V, output and quantized-weight
    # staging overlap.  The observed SLA crash requests a 1.36-GiB full-width
    # output tensor at block 3.  Do not permit a backend that materializes such
    # a tensor on <=18.5-GB GPUs even if mem_get_info() looks temporarily roomy.
    geometry_unsafe = bool(
        int(total_b) <= int(18.5 * 1024**3)
        and int(one_b) >= int(768 * 1024**2)
    )
    if external_sla and geometry_unsafe and not bool(state.get('external_sla_direct_fastpath', False)):
        safe = False
    if external_sla and bool(state.get('external_sla_direct_fastpath', False)):
        # v0.4.36: mem_get_info() is sampled after the block input is already
        # resident, so only *additional* zero-copy SLA allocations belong in
        # this decision. The fast path needs fused QKV (3x width) plus a small
        # routing workspace; it does not allocate contiguous Q/K/V copies or O.
        one = int(one_b)
        direct_estimated_b = int(3 * one + max(256 * 1024**2, int(one * 0.15)))
        direct_usable_b = max(0, int(free_b) - 512 * 1024**2)
        safe = direct_estimated_b <= direct_usable_b
        # The previous external SLA run proved the fused H3 QKV projection itself
        # fits at ~1.36 GiB/tensor; its OOM came from later Q/K/V copies + o_s.
        # Prefer the zero-copy path for this 16-GB geometry and let only a
        # pre-kernel allocation OOM fall back to bounded Sol.
        if int(total_b) <= int(18.5 * 1024**3) and one <= int(1.60 * 1024**3):
            safe = True
        estimated_b = direct_estimated_b
        usable_b = direct_usable_b
        geometry_unsafe = False
    mode = 'existing' if safe else 'sol'
    state['auto_attention_estimated_peak_mb'] = estimated_b / 1024**2
    state['auto_attention_one_tensor_mb'] = one_b / 1024**2
    state['auto_attention_driver_free_mb'] = int(free_b) / 1024**2
    state['auto_attention_usable_mb'] = usable_b / 1024**2
    return mode, (
        f'{s} tokens x hidden={d} x {e}B; '
        f'existing_peak~{estimated_b/1024**2:.0f}MB '
        f'(single tensor={one_b/1024**2:.0f}MB), '
        f'driver_free={int(free_b)/1024**2:.0f}MB, reserve={reserve_b/1024**2:.0f}MB, '
        f'external_SLA={external_sla}, geometry_unsafe={geometry_unsafe} -> {mode}'
    )



def _sol_schedule_tau(transformer_options, state):
    """Geometry-aware Sol tau scheduler used by AUTO high-res mode.

    AUTO keeps the proven 1.70 -> 2.10 base schedule and adds a modest,
    bounded boost from packed token count.  This is a routing-policy change
    only: no H3 tokens are merged, pooled, or discarded.
    """
    mode = str(state.get('sol_mode', 'existing'))
    requested = str(state.get('requested_attention_mode', mode))

    sigma = None
    opts = transformer_options or {}
    for key in ('sigmas', 'sigma', 'timestep'):
        value = opts.get(key)
        if value is None:
            continue
        try:
            if torch.is_tensor(value):
                sigma = float(value.flatten()[0].detach().float().cpu().item())
            else:
                sigma = float(value)
            break
        except Exception:
            pass

    tau_start = float(state.get('sol_tau_start', 1.3))
    tau_end = float(state.get('sol_tau_end', 0.8))
    sigma_hi = float(state.get('sol_sigma_hi', 1.0))
    sigma_lo = float(state.get('sol_sigma_lo', 0.0))
    curve = str(state.get('sol_curve', 'linear'))

    token_count = int(state.get('current_token_count', 0) or 0)
    auto_speed = (
        requested == 'auto'
        and mode == 'sol'
        and token_count >= 120000
    )

    # V39: quantized backends use the SAME geometry-aware SOL routing policy as
    # the proven NVFP4 path.  V38 showed that the old numerical-error calibration
    # was the wrong control signal for SOL throughput: a 1% rel-RMS budget selected
    # tau=-1.50 and still routed ~94% exact.  Backend-specific tau overrides are
    # therefore removed; AUTO existing/SOL selection remains unchanged.
    if auto_speed and _v12_is_int8_family(state) and not state.get('v39_policy_parity_announced', False):
        _lm_print(
            '[MiniMaxH3 LongMedia][V39 SOL POLICY PARITY] '
            'INT8/W4A8 uses the same AUTO geometry-aware tau schedule as NVFP4; '
            'legacy V16/V37 quality-budget tau override disabled',
            flush=True,
        )
        state['v39_policy_parity_announced'] = True

    if auto_speed:
        # v0.3.24: keep NVFP4 on the proven quality-safe AUTO SOL schedule, but
        # bias INT8/ConvRot slightly denser (lower tau) to reduce early geometry
        # instability under the same seed/conditioning.  This is deliberately a
        # modest routing change, not a revival of the old aggressive quality-budget
        # calibration path from V16/V37.
        is_int8_family = _v12_is_int8_family(state)
        if is_int8_family:
            base_start = 1.00
            base_end = 1.45
            boost_cap = 0.08
            boost_scale = 80000.0
        else:
            base_start = 1.30
            base_end = 1.85
            boost_cap = 0.15
            boost_scale = 60000.0

        token_boost = 0.0
        if token_count > 150000:
            token_boost = min(
                boost_cap,
                max(
                    0.0,
                    (float(token_count) - 150000.0) / boost_scale * 0.12,
                ),
            )

        tau_start = base_start + token_boost
        tau_end = base_end + token_boost
        curve = 'linear'
        state['sol_geometry_tau_boost'] = float(token_boost)
        state['sol_geometry_tau_profile'] = (
            'int8_quality_safe' if is_int8_family else 'nvfp4_quality_safe'
        )

        announce_key = 'sol_speed_tau_announced_v324'
        if not state.get(announce_key):
            label = 'AUTO GEO TAU INT8 SAFE' if is_int8_family else 'AUTO GEO TAU'
            _lm_print(
                f'[MiniMaxH3 LongMedia][{label}] '
                f'base {base_start:.2f}->{base_end:.2f}, '
                f'tokens={token_count}, boost={token_boost:.3f} => '
                f'{tau_start:.2f}->{tau_end:.2f}',
                flush=True,
            )
            state[announce_key] = True
            state['sol_speed_tau_announced'] = True

    if mode == 'sol' and not auto_speed:
        return tau_start
    if mode not in ('scheduled_sol', 'sol'):
        return None
    if sigma is None:
        return tau_start

    denom = max(1.0e-8, sigma_hi - sigma_lo)
    progress = max(0.0, min(1.0, (sigma_hi - sigma) / denom))
    if curve == 'ease_in':
        progress *= progress
    elif curve == 'ease_out':
        progress = 1.0 - (1.0 - progress) * (1.0 - progress)
    elif curve == 'smoothstep':
        progress = progress * progress * (3.0 - 2.0 * progress)

    tau = tau_start + (tau_end - tau_start) * progress
    state['last_sol_tau'] = float(tau)
    return float(tau)




def _int8_sync_cast_stream(state, *, block_index=None):
    """Synchronize Comfy's cast/offload stream before Sol storage allocation.

    INT8 dynamic/on-demand casting may use a secondary CUDA stream.  A failure
    on that stream can otherwise surface later at an unrelated torch.empty(),
    making the Sol storage allocator look like the source of the OOM.
    """
    backend = str(state.get('model_runtime_backend', 'unknown')).lower()
    if backend not in ('int8', 'int8-convrot-w4a4'):
        return True

    try:
        stream = getattr(comfy.model_management, 'offload_stream', None)
        if stream is not None:
            stream.synchronize()
        return True
    except Exception as exc:
        _lm_print(
            '[MiniMaxH3 LongMedia][INT8 CAST SYNC] failed before Sol storage: '
            f'block={block_index}, {type(exc).__name__}: {exc}',
            flush=True,
        )
        raise



def _int8_prepare_block_linear(linear, probe_input):
    """Cast/stream one Comfy linear once, keep it valid until explicit release."""
    import comfy.ops
    weight, bias, offload_state = comfy.ops.cast_bias_weight(
        linear,
        probe_input,
        offloadable=True,
        compute_dtype=probe_input.dtype,
        want_requant=True,
    )
    return {
        'linear': linear,
        'weight': weight,
        'bias': bias,
        'offload_state': offload_state,
    }


def _int8_release_block_linear(handle):
    if not handle:
        return
    import comfy.ops
    comfy.ops.uncast_bias_weight(
        handle['linear'],
        handle['weight'],
        handle['bias'],
        handle['offload_state'],
    )


def _int8_cached_linear(handle, x, *, input_act=None):
    """Use a block-resident cast weight with stock Comfy quantized semantics.

    Important:
    - ordinary Linear must go through F.linear(x, QuantizedTensor, bias) so
      comfy-kitchen/QuantizedTensor dispatch can honor layout metadata.
    - only fused input activation mirrors comfy.ops.linear_input_act and calls
      ck.int8_linear directly when the weight is a non-transposed
      TensorWiseINT8Layout.
    """
    import comfy.quant_ops

    weight = handle['weight']
    bias = handle['bias']

    # Stock comfy Linear semantics: QuantizedTensor __torch_dispatch__ decides
    # the correct kernel/layout. Do NOT manually unpack qdata for qkv/fc1/out.
    if input_act is None:
        return torch.nn.functional.linear(x, weight, bias)

    # Mirror comfy.ops.linear_input_act exactly for the cached cast result.
    def _apply_input_act(value, act):
        if act == 'swiglu':
            gate, up = value.chunk(2, dim=-1)
            return torch.nn.functional.silu(gate).mul_(up)
        if act == 'gelu_tanh':
            return torch.nn.functional.gelu(value, approximate='tanh')
        raise ValueError(f'unsupported cached input_act={act!r}')

    QuantizedTensor = getattr(comfy.quant_ops, 'QuantizedTensor', ())
    is_quant = isinstance(weight, QuantizedTensor) if QuantizedTensor else False

    if (
        not is_quant
        or getattr(weight, '_layout_cls', None) != 'TensorWiseINT8Layout'
        or getattr(getattr(weight, '_params', None), 'transposed', False)
    ):
        return torch.nn.functional.linear(
            _apply_input_act(x, input_act),
            weight,
            bias,
        )

    qdata, scale = comfy.quant_ops.TensorWiseINT8Layout.get_plain_tensors(weight)
    return comfy.quant_ops.ck.int8_linear(
        x,
        qdata,
        scale,
        bias,
        x.dtype,
        convrot=getattr(weight._params, 'convrot', False),
        convrot_groupsize=getattr(weight._params, 'convrot_groupsize', 256),
        input_act=input_act,
    )


_V34_KERNEL_BACKEND_ANNOUNCED = set()


def _v33_kitchen_impl(func_name, kwargs):
    """Prefer comfy-kitchen's native CUDA implementation on NVIDIA.

    The registry normally prefers CUDA already, but this makes the long-media
    quant path explicit and logs the implementation actually selected.  If the
    CUDA implementation rejects a shape/version, fall back to normal registry
    dispatch without changing math.
    """
    from comfy_kitchen.registry import registry
    backend = None
    impl = None
    if torch.version.cuda is not None and torch.cuda.is_available():
        try:
            impl = registry.get_implementation(func_name, backend='cuda', kwargs=kwargs)
            backend = 'cuda'
        except Exception:
            impl = None
    if impl is None:
        try:
            backend = registry.get_capable_backend(func_name, kwargs=kwargs)
            impl = registry.get_implementation(func_name, backend=backend, kwargs=kwargs)
        except Exception:
            impl = registry.get_implementation(func_name, kwargs=kwargs)
            backend = getattr(impl, '__module__', 'auto')
    key = (func_name, str(backend))
    if key not in _V34_KERNEL_BACKEND_ANNOUNCED:
        _V34_KERNEL_BACKEND_ANNOUNCED.add(key)
        _lm_print(
            '[MiniMaxH3 LongMedia][V40 KITCHEN BACKEND] '
            f'{func_name} -> {backend} ({getattr(impl, "__module__", "?")}.{getattr(impl, "__name__", "?")})',
            flush=True,
        )
    return impl


def _v32_quant_linear_rows(handle, x, row_start, row_end):
    """Run a row slice of a prepared native Comfy quantized Linear.

    Q/K/V occupy independent contiguous output rows in H3 qkv_proj.  Calling
    the official comfy-kitchen kernel on only the rows required by the current
    streaming phase avoids computing throw-away Q during KV build and
    throw-away K/V during query replay.  No quantization math is reimplemented.
    """
    weight = handle['weight']
    bias = handle['bias']
    layout = getattr(weight, '_layout_cls', None)
    params = getattr(weight, '_params', None)
    if params is None or bool(getattr(params, 'transposed', False)):
        return None
    rs, re = int(row_start), int(row_end)
    if rs < 0 or re <= rs:
        return None
    b = None if bias is None else bias[rs:re]

    # Stock INT8 TensorWise / ConvRot path.
    if layout == 'TensorWiseINT8Layout':
        try:
            qdata, scale = comfy.quant_ops.TensorWiseINT8Layout.get_plain_tensors(weight)
            kwargs = {
                'x': x.contiguous(),
                'weight': qdata[rs:re].contiguous(),
                'weight_scale': scale[rs:re],
                'bias': b,
                'out_dtype': x.dtype,
                'convrot': getattr(params, 'convrot', False),
                'convrot_groupsize': getattr(params, 'convrot_groupsize', 256),
            }
            impl = _v33_kitchen_impl('int8_linear', kwargs)
            return impl(**kwargs)
        except Exception:
            return None

    # Stock grouped W4A8 path. The layout itself routes to this exact kitchen op;
    # slicing output rows is mathematically identical to slicing Linear outputs.
    if layout == 'AsymW4A8Int8Layout':
        try:
            from comfy_kitchen.tensor.w4a8_int8 import (
                AsymW4A8Int8Layout, w4a8_int8_linear,
            )
            qdata, s_rel, s_channel, correction, codebook = (
                AsymW4A8Int8Layout.get_plain_tensors(weight)
            )
            corr = None if correction is None else correction[:, rs:re]
            kwargs = {
                'x': x,
                'qdata': qdata[rs:re],
                's_rel': s_rel[rs:re],
                's_channel': s_channel[rs:re],
                'codebook': codebook,
                'correction': corr,
                'bias': b,
                'group_size': getattr(params, 'group_size', 16),
                'convrot_groupsize': getattr(params, 'convrot_groupsize', 256),
                'out_dtype': x.dtype,
            }
            impl = _v33_kitchen_impl('w4a8_int8_linear', kwargs)
            return impl(**kwargs)
        except Exception:
            return None
    return None


def _v12b_linear_ab_enabled(state, label):
    """Run each numeric A/B once, only on INT8/W4A8 block 0 / forward 1."""
    if not _v12_is_int8_family(state):
        return False
    if int(state.get('active_block_index', -1)) != 0:
        return False
    if int(state.get('v12_int8_sol_forward_generation', 0) or 0) != 1:
        return False
    done = state.setdefault('v12b_linear_ab_done', {})
    return not bool(done.get(str(label), False))


def _v12b_linear_ab_report(state, label, stock, cached):
    """Compare real stock/cached outputs without retaining GPU-sized tensors."""
    label = str(label)
    done = state.setdefault('v12b_linear_ab_done', {})
    try:
        stock32 = stock.detach().to(device='cpu', dtype=torch.float32)
        cached32 = cached.detach().to(device='cpu', dtype=torch.float32)
        if tuple(stock32.shape) != tuple(cached32.shape):
            _lm_print(
                '[MiniMaxH3 LongMedia][V12-B LINEAR A/B] '
                f'{label}: SHAPE MISMATCH stock={tuple(stock32.shape)} '
                f'cached={tuple(cached32.shape)}',
                flush=True,
            )
            done[label] = True
            return

        stock_flat = stock32.flatten()
        cached_flat = cached32.flatten()
        stock_finite = bool(torch.isfinite(stock_flat).all().item())
        cached_finite = bool(torch.isfinite(cached_flat).all().item())
        diff = cached_flat - stock_flat
        diff_finite = bool(torch.isfinite(diff).all().item())

        if stock_flat.numel() == 0:
            rel_rms = max_abs = mean_abs = 0.0
            cosine = 1.0
        else:
            eps = 1.0e-12
            rms_ref = torch.sqrt(torch.mean(stock_flat.square())).item()
            rms_diff = torch.sqrt(torch.mean(diff.square())).item()
            rel_rms = float(rms_diff / max(eps, rms_ref))
            max_abs = float(diff.abs().max().item())
            mean_abs = float(diff.abs().mean().item())
            denom = float(
                torch.linalg.vector_norm(stock_flat).item()
                * torch.linalg.vector_norm(cached_flat).item()
            )
            cosine = float(torch.dot(stock_flat, cached_flat).item() / max(eps, denom))

        verdict = (
            'MATCH'
            if stock_finite and cached_finite and diff_finite
            and rel_rms <= 1.0e-5 and cosine >= 0.99999
            else 'DIVERGED'
        )
        _lm_print(
            '[MiniMaxH3 LongMedia][V12-B LINEAR A/B] '
            f'{label}: {verdict}, shape={tuple(stock32.shape)}, '
            f'rel_rms={rel_rms:.8e}, mean_abs={mean_abs:.8e}, '
            f'max_abs={max_abs:.8e}, cosine={cosine:.10f}, '
            f'finite(stock/cached/diff)='
            f'{stock_finite}/{cached_finite}/{diff_finite}',
            flush=True,
        )
    except Exception as exc:
        _lm_print(
            '[MiniMaxH3 LongMedia][V12-B LINEAR A/B] '
            f'{label}: diagnostic failed: {type(exc).__name__}: {exc}',
            flush=True,
        )
    finally:
        done[label] = True


def _v19_selected_block(state, block_index=None):
    # V25 cleanup build: forensic V22 stage probes disabled.
    return False
    """V22: robust first/middle/last H3 selection for the first diagnostic forward.

    Targets are frozen when the patch set is installed, rather than recomputed
    from mutable runtime state while blocks are executing.  A separate forward
    latch is armed on block 0 and stays true until the last target exits.
    """
    if not _v12_is_int8_family(state):
        return False
    block_index = int(
        state.get('active_block_index', -1)
        if block_index is None else block_index
    )
    targets = state.get('v21_stage_ab_targets')
    if not targets:
        last = int(state.get('last_patched_block_index', -1) or -1)
        targets = (0,) if last < 0 else (0, last // 2, last)
    if block_index not in set(int(v) for v in targets):
        return False
    # Arm exactly once when the first INT8 generation reaches block 0.  Do not
    # depend on the generation counter afterwards; other helpers may mutate or
    # release forward-scoped state before the later target blocks run.
    if block_index == int(targets[0]):
        if not state.get('v21_stage_ab_completed', False) and not state.get('v21_stage_ab_armed', False):
            state['v21_stage_ab_armed'] = True
            state['v21_stage_ab_generation'] = int(state.get('v12_int8_sol_forward_generation', 0) or 0)
            _lm_print(
                '[MiniMaxH3 LongMedia][V22 STAGE A/B TARGET] '
                f'armed generation={state["v21_stage_ab_generation"]}, targets={list(targets)}',
                flush=True,
            )
    return bool(state.get('v21_stage_ab_armed', False))


def _v19_probe_offsets(token_count, chunk_hint=8192):
    """Four aligned probe regions matching the successful V17/V18 samples."""
    token_count = int(token_count)
    chunk = max(64, (int(chunk_hint or 8192) // 64) * 64)
    return sorted({
        0,
        min(chunk, max(0, token_count - 1)),
        (token_count // 2 // chunk) * chunk,
        ((token_count - 1) // chunk) * chunk,
    })


def _v19_report(state, stage, reference, candidate, *, offsets=None):
    """Report a compact combined A/B across all V19 token probes."""
    block_index = int(state.get('active_block_index', -1))
    key = f'{block_index}:{stage}'
    done = state.setdefault('v19_stage_ab_done', set())
    if key in done:
        return
    try:
        ref = reference.detach().to(device='cpu', dtype=torch.float32)
        got = candidate.detach().to(device='cpu', dtype=torch.float32)
        if tuple(ref.shape) != tuple(got.shape):
            _lm_print(
                '[MiniMaxH3 LongMedia][V22 MULTI-BLOCK STAGE A/B] '
                f'block={block_index}, stage={stage}, SHAPE-MISMATCH '
                f'reference={tuple(ref.shape)}, candidate={tuple(got.shape)}',
                flush=True,
            )
            return
        ref_flat = ref.flatten()
        got_flat = got.flatten()
        diff = got_flat - ref_flat
        finite = bool(
            torch.isfinite(ref_flat).all().item()
            and torch.isfinite(got_flat).all().item()
            and torch.isfinite(diff).all().item()
        )

        # V20: never let a numerically noisy FP32 cosine turn a bit-exact
        # comparison into DIVERGED.  V19 exposed this on the real workload:
        # rel_rms/mean_abs/max_abs were all exactly zero while FP32 dot/norm
        # produced cosine=0.99998 (and even >1.0 on another stage).
        exact = bool(torch.equal(ref_flat, got_flat))
        mismatch_count = int(torch.count_nonzero(diff).item())
        if exact:
            rel_rms = 0.0
            cosine = 1.0
            mean_abs = 0.0
            max_abs = 0.0
        else:
            eps = 1.0e-30
            ref64 = ref_flat.to(dtype=torch.float64)
            got64 = got_flat.to(dtype=torch.float64)
            diff64 = got64 - ref64
            rms_ref = float(torch.sqrt(torch.mean(ref64.square())).item())
            rms_diff = float(torch.sqrt(torch.mean(diff64.square())).item())
            rel_rms = rms_diff / max(eps, rms_ref)
            norm_ref = float(torch.linalg.vector_norm(ref64).item())
            norm_got = float(torch.linalg.vector_norm(got64).item())
            denom = norm_ref * norm_got
            if denom <= eps:
                cosine = 1.0 if rms_diff <= eps else 0.0
            else:
                cosine = float(torch.dot(ref64, got64).item() / denom)
                cosine = max(-1.0, min(1.0, cosine))
            mean_abs = float(diff64.abs().mean().item())
            max_abs = float(diff64.abs().max().item())
            del ref64, got64, diff64

        verdict = (
            'MATCH'
            if finite and (exact or (rel_rms <= 1.0e-5 and cosine >= 0.99999))
            else 'DIVERGED'
        )
        _lm_print(
            '[MiniMaxH3 LongMedia][V22 MULTI-BLOCK STAGE A/B] '
            f'block={block_index}, stage={stage}, verdict={verdict}, '
            f'offsets={list(offsets or [])}, rows={int(ref.shape[0])}, '
            f'exact={exact}, mismatches={mismatch_count}, '
            f'rel_rms={rel_rms:.8e}, cosine={cosine:.10f}, '
            f'mean_abs={mean_abs:.8e}, max_abs={max_abs:.8e}, finite={finite}',
            flush=True,
        )
    except Exception as exc:
        _lm_print(
            '[MiniMaxH3 LongMedia][V22 MULTI-BLOCK STAGE A/B] '
            f'block={block_index}, stage={stage}, diagnostic failed: '
            f'{type(exc).__name__}: {exc}',
            flush=True,
        )
    finally:
        done.add(key)


def _v13_exact_attention_from_compressed(q, storage, key_chunk=1024):
    """Exact softmax attention over the token-level compressed Sol K/V.

    Only a few query rows are used by the caller. Keys and values are
    reconstructed in bounded chunks, so this never materializes the full
    Q-by-K attention matrix.
    """
    q32 = q.detach().to(dtype=torch.float32)
    batch, queries, heads, head_dim = q32.shape
    tokens = int(storage['tokens'])
    if batch != int(storage['k8'].shape[0]):
        raise ValueError('V13 exact A/B batch geometry mismatch')

    running_max = torch.full(
        (batch, heads, queries),
        -float('inf'),
        device=q.device,
        dtype=torch.float32,
    )
    running_sum = torch.zeros_like(running_max)
    running_out = torch.zeros(
        (batch, heads, queries, head_dim),
        device=q.device,
        dtype=torch.float32,
    )
    scale = float(head_dim ** -0.5)

    for key_start in range(0, tokens, int(key_chunk)):
        key_end = min(tokens, key_start + int(key_chunk))
        block_indices = (
            torch.arange(key_start, key_end, device=q.device, dtype=torch.long)
            // 64
        )
        k = (
            storage['k8'][:, key_start:key_end].to(dtype=torch.float32)
            * storage['ks'][:, key_start:key_end].unsqueeze(-1)
            + storage['kc'][:, block_indices].to(dtype=torch.float32)
        ).to(dtype=torch.bfloat16).to(dtype=torch.float32)
        v = (
            storage['v8'][:, key_start:key_end].to(dtype=torch.float32)
            * storage['vs'][:, key_start:key_end].unsqueeze(-1)
        ).to(dtype=torch.bfloat16).to(dtype=torch.float32)

        scores = torch.einsum('bqhd,bkhd->bhqk', q32, k) * scale
        chunk_max = scores.amax(dim=-1)
        next_max = torch.maximum(running_max, chunk_max)
        previous_scale = torch.exp(running_max - next_max)
        probability = torch.exp(scores - next_max.unsqueeze(-1))
        running_out = (
            running_out * previous_scale.unsqueeze(-1)
            + torch.einsum('bhqk,bkhd->bhqd', probability, v)
        )
        running_sum = (
            running_sum * previous_scale
            + probability.sum(dim=-1)
        )
        running_max = next_max
        del block_indices, k, v, scores, chunk_max, next_max
        del previous_scale, probability

    return (
        running_out / running_sum.unsqueeze(-1)
    ).permute(0, 2, 1, 3).contiguous()


def _v15_attention_ab_report(label, exact, candidate, storage, tau):
    """Report one candidate against an already-computed exact reference."""
    exact_flat = exact.detach().to(device='cpu', dtype=torch.float32).flatten()
    candidate_flat = (
        candidate[:, :exact.shape[1]]
        .detach()
        .to(device='cpu', dtype=torch.float32)
        .flatten()
    )
    diff = candidate_flat - exact_flat
    exact_finite = bool(torch.isfinite(exact_flat).all().item())
    candidate_finite = bool(torch.isfinite(candidate_flat).all().item())
    diff_finite = bool(torch.isfinite(diff).all().item())
    eps = 1.0e-12
    rms_ref = float(torch.sqrt(torch.mean(exact_flat.square())).item())
    rms_diff = float(torch.sqrt(torch.mean(diff.square())).item())
    rel_rms = rms_diff / max(eps, rms_ref)
    mean_abs = float(diff.abs().mean().item())
    max_abs = float(diff.abs().max().item())
    denom = float(
        torch.linalg.vector_norm(exact_flat).item()
        * torch.linalg.vector_norm(candidate_flat).item()
    )
    cosine = float(torch.dot(exact_flat, candidate_flat).item() / max(eps, denom))
    _lm_print(
        '[MiniMaxH3 LongMedia][V15 TAU CALIBRATION] '
        f'{label}: block=0, forward=1, queries={int(exact.shape[1])}, '
        f'keys={int(storage["tokens"])}, tau={float(tau):.4f}, '
        f'rel_rms={rel_rms:.8e}, mean_abs={mean_abs:.8e}, '
        f'max_abs={max_abs:.8e}, cosine={cosine:.10f}, '
        f'finite(exact/candidate/diff)='
        f'{exact_finite}/{candidate_finite}/{diff_finite}',
        flush=True,
    )



def _v37_auto_calibrate_sol_tau(state, q, storage, sink_blocks, sink_q, q_offset):
    """V38: one-shot *full query-chunk* SOL speed/quality calibration.

    V37 measured quality on 64 rows, but its timing was also for those 64 rows and
    therefore did not predict the real 8192-row query replay cost.  V38 keeps the
    exact same SOL math and AUTO existing/SOL policy, but benchmarks each tau on
    the actual first query chunk after one warmup.  Quality is compared on a small
    prefix against compressed-exact to keep the reference cheap; runtime/exact
    routing are measured on the full chunk.  We print feasible points for several
    numerical budgets so one run exposes the whole speed/quality curve.
    """
    # V39: retained for forensic history only.  Do not alter runtime tau; the
    # scheduler above now owns quantized SOL policy exactly as it does for NVFP4.
    return None

    if not _v12_is_int8_family(state):
        return None
    if str(state.get('requested_attention_mode', '')).lower() != 'auto':
        return None
    if str(state.get('sol_mode', '')).lower() != 'sol':
        return None
    if state.get('v37_auto_quality_tau') is not None:
        return float(state['v37_auto_quality_tau'])
    if bool(state.get('v37_tau_autocal_running', False)):
        return None
    if int(state.get('active_block_index', -1)) != 0 or int(q_offset) != 0:
        return None

    from .sol_kernel import sol_attn_query_compressed_sm120
    state['v37_tau_autocal_running'] = True
    quality_budget = float(state.get('v37_tau_quality_budget', 0.0100))
    # Wide enough to expose the useful speed/quality knee, but not so many points
    # that calibration itself becomes a minutes-long benchmark.
    candidates = (2.00, 1.50, 1.00, 0.50, 0.00, -0.50, -1.00, -1.50, -2.00)
    try:
        full_rows = int(q.shape[1])
        compare_rows = min(8, full_rows)
        q_full = q
        exact = _v13_exact_attention_from_compressed(
            q_full[:, :compare_rows], storage, key_chunk=1024
        ).float()
        exact_rms = torch.sqrt(torch.mean(exact.square())).clamp_min(1.0e-12)

        # Warm up the real 8192-row kernel once so compilation/first-use cost does
        # not poison the first candidate's timing.
        _warm, _warm_telem = sol_attn_query_compressed_sm120(
            q_full, storage, q_offset=int(q_offset), tau=-2.0,
            sink_blocks=sink_blocks, sink_q=sink_q, telemetry=True,
        )
        torch.cuda.synchronize(q.device)
        del _warm, _warm_telem
        _lm_print(
            '[MiniMaxH3 LongMedia][V38 FULL-CHUNK SWEEP] '
            f'warmup complete; rows={full_rows}, quality_compare_rows={compare_rows}, '
            f'candidates={len(candidates)}',
            flush=True,
        )

        rows = []
        selected = None
        for candidate_tau in candidates:
            torch.cuda.synchronize(q.device)
            torch.cuda.reset_peak_memory_stats(q.device)
            t0 = time.perf_counter()
            candidate, telem = sol_attn_query_compressed_sm120(
                q_full, storage, q_offset=int(q_offset), tau=float(candidate_tau),
                sink_blocks=sink_blocks, sink_q=sink_q, telemetry=True,
            )
            torch.cuda.synchronize(q.device)
            elapsed = time.perf_counter() - t0
            cand_cmp = candidate[:, :compare_rows].float()
            diff = cand_cmp - exact
            rel_rms = float((torch.sqrt(torch.mean(diff.square())) / exact_rms).item())
            denom = (torch.linalg.vector_norm(exact) * torch.linalg.vector_norm(cand_cmp)).clamp_min(1.0e-12)
            cosine = float((torch.sum(exact * cand_cmp) / denom).item())
            exact_ratio = float(telem.get('exact_ratio', 1.0))
            peak_mb = float(torch.cuda.max_memory_allocated(q.device)) / (1024.0 * 1024.0)
            rows.append((float(candidate_tau), rel_rms, cosine, exact_ratio, elapsed, peak_mb))
            _lm_print(
                '[MiniMaxH3 LongMedia][V38 TAU FULL] '
                f'tau={candidate_tau:+.2f} rel_rms={rel_rms:.6f} cosine={cosine:.7f} '
                f'exact={exact_ratio*100.0:.2f}% kernel={elapsed:.4f}s '
                f'peak_alloc={peak_mb:.0f}MB',
                flush=True,
            )
            if selected is None and rel_rms <= quality_budget and torch.isfinite(diff).all():
                selected = (float(candidate_tau), rel_rms, cosine, exact_ratio, elapsed, peak_mb)
            del candidate, cand_cmp, diff

        # Report the fastest point satisfying several useful quality budgets.
        for budget in (0.01, 0.05, 0.10, 0.15, 0.20):
            feasible = [r for r in rows if r[1] <= budget]
            if feasible:
                best = min(feasible, key=lambda r: r[4])
                _lm_print(
                    '[MiniMaxH3 LongMedia][V38 BUDGET KNEE] '
                    f'budget={budget*100:.0f}% tau={best[0]:+.2f} '
                    f'rel_rms={best[1]:.6f} exact={best[3]*100.0:.2f}% '
                    f'kernel={best[4]:.4f}s',
                    flush=True,
                )
            else:
                _lm_print(
                    '[MiniMaxH3 LongMedia][V38 BUDGET KNEE] '
                    f'budget={budget*100:.0f}% no-candidate',
                    flush=True,
                )

        # Keep V37's conservative automatic behavior for generation.  V38 is a
        # forensic throughput sweep, not a silent quality-policy change.
        if selected is None:
            selected = min(rows, key=lambda r: r[1])
            reason = 'fallback-lowest-error'
        else:
            reason = 'quality-budget'
        tau_sel, err_sel, cos_sel, exact_sel, time_sel, peak_sel = selected
        state['v37_auto_quality_tau'] = float(tau_sel)
        state['v37_auto_quality_rel_rms'] = float(err_sel)
        state['v37_auto_quality_exact_ratio'] = float(exact_sel)
        state['last_sol_tau'] = float(tau_sel)
        _lm_print(
            '[MiniMaxH3 LongMedia][V38 TAU SELECT] '
            f'tau={tau_sel:+.2f} reason={reason} rel_rms={err_sel:.6f} '
            f'cosine={cos_sel:.7f} exact={exact_sel*100.0:.2f}% '
            f'full_chunk_kernel={time_sel:.4f}s budget={quality_budget:.4f}',
            flush=True,
        )
        del exact
        return float(tau_sel)
    except Exception as exc:
        _lm_print(
            '[MiniMaxH3 LongMedia][V38 TAU AUTOCAL] '
            f'failed: {type(exc).__name__}: {exc}; keeping provisional tau=-2.0',
            flush=True,
        )
        return None
    finally:
        state['v37_tau_autocal_running'] = False


def _v15_tau_calibration(
    state, q, storage, sol_output, tau, sink_blocks, sink_q
):
    """Calibrate routed Sol tau against the same exact reference."""
    if not _v12_is_int8_family(state):
        return
    if int(state.get('active_block_index', -1)) != 0:
        return
    if int(state.get('v12_int8_sol_forward_generation', 0) or 0) != 1:
        return
    if bool(state.get('v15_tau_calibration_done', False)):
        return

    # Mark first so a diagnostic failure cannot repeat for every query chunk.
    state['v15_tau_calibration_done'] = True
    query_count = min(4, int(q.shape[1]))
    try:
        exact = _v13_exact_attention_from_compressed(
            q[:, :query_count], storage, key_chunk=1024
        )
        _v15_attention_ab_report(
            'CURRENT-ROUTED', exact, sol_output, storage, tau
        )

        # A full 64-row query block is required because Sol makes routing
        # decisions per block. Sweep below the aggressive AUTO value to find
        # the quality/performance knee for this real prompt geometry.
        from .sol_kernel import sol_attn_query_compressed_sm120
        probe_query_rows = min(64, int(q.shape[1]))
        q_probe = q[:, :probe_query_rows]
        for candidate_tau in (1.30, 0.80, 0.00, -1.00, -2.00):
            torch.cuda.synchronize(q.device)
            started = time.perf_counter()
            candidate = sol_attn_query_compressed_sm120(
                q_probe,
                storage,
                q_offset=0,
                tau=float(candidate_tau),
                sink_blocks=sink_blocks,
                sink_q=(0, 0),
            )
            torch.cuda.synchronize(q.device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            _v15_attention_ab_report(
                f'TAU-SWEEP elapsed_ms={elapsed_ms:.1f}',
                exact, candidate, storage, candidate_tau,
            )
            del candidate

        # Forced-exact is the accuracy floor of the compressed Triton path.
        torch.cuda.synchronize(q.device)
        started = time.perf_counter()
        forced_exact = sol_attn_query_compressed_sm120(
            q_probe,
            storage,
            q_offset=0,
            tau=float(tau),
            sink_blocks=sink_blocks,
            sink_q=(0, 1),
        )
        torch.cuda.synchronize(q.device)
        forced_ms = (time.perf_counter() - started) * 1000.0
        _v15_attention_ab_report(
            f'FORCED-EXACT elapsed_ms={forced_ms:.1f}',
            exact, forced_exact, storage, tau,
        )
        _lm_print(
            '[MiniMaxH3 LongMedia][V15 STRIDES] '
            f'q={tuple(q.stride())}, routed={tuple(sol_output.stride())}, '
            f'forced={tuple(forced_exact.stride())}, '
            f'contiguous(routed/forced)='
            f'{sol_output.is_contiguous()}/{forced_exact.is_contiguous()}',
            flush=True,
        )
        del exact, forced_exact
    except Exception as exc:
        _lm_print(
            '[MiniMaxH3 LongMedia][V15 TAU CALIBRATION] '
            f'diagnostic failed: {type(exc).__name__}: {exc}',
            flush=True,
        )


def _v17_multi_offset_calibration(
    state, q, storage, sol_output, tau, sink_blocks,
    q_offset, sequence_tokens, query_chunk,
):
    """Measure routed Sol accuracy across conditioning/video/tail regions."""
    if not _v12_is_int8_family(state):
        return
    if int(state.get('active_block_index', -1)) != 0:
        return
    if int(state.get('v12_int8_sol_forward_generation', 0) or 0) != 1:
        return

    query_offset = int(q_offset)
    sequence_tokens = int(sequence_tokens)
    query_chunk = int(query_chunk)
    targets = {
        0,
        min(query_chunk, max(0, sequence_tokens - 1)),
        (sequence_tokens // 2 // query_chunk) * query_chunk,
        ((sequence_tokens - 1) // query_chunk) * query_chunk,
    }
    if query_offset not in targets:
        return
    done = state.setdefault('v17_calibrated_offsets', set())
    if query_offset in done:
        return
    done.add(query_offset)

    query_count = min(4, int(q.shape[1]))
    probe_rows = min(64, int(q.shape[1]))
    q_probe = q[:, :probe_rows]
    try:
        exact = _v13_exact_attention_from_compressed(
            q[:, :query_count], storage, key_chunk=1024
        )

        def _report(label, candidate, candidate_tau, elapsed_ms=None):
            exact_flat = (
                exact.detach().to(device='cpu', dtype=torch.float32).flatten()
            )
            candidate_flat = (
                candidate[:, :query_count]
                .detach()
                .to(device='cpu', dtype=torch.float32)
                .flatten()
            )
            diff = candidate_flat - exact_flat
            eps = 1.0e-12
            rms_ref = float(torch.sqrt(torch.mean(exact_flat.square())).item())
            rms_diff = float(torch.sqrt(torch.mean(diff.square())).item())
            rel_rms = rms_diff / max(eps, rms_ref)
            denom = float(
                torch.linalg.vector_norm(exact_flat).item()
                * torch.linalg.vector_norm(candidate_flat).item()
            )
            cosine = float(
                torch.dot(exact_flat, candidate_flat).item() / max(eps, denom)
            )
            timing = (
                ''
                if elapsed_ms is None
                else f', elapsed_ms={float(elapsed_ms):.1f}'
            )
            _lm_print(
                '[MiniMaxH3 LongMedia][V17 MULTI-OFFSET TAU] '
                f'offset={query_offset}, label={label}, '
                f'tau={float(candidate_tau):.2f}{timing}, '
                f'rel_rms={rel_rms:.8e}, cosine={cosine:.10f}, '
                f'max_abs={float(diff.abs().max().item()):.8e}, '
                f'finite={bool(torch.isfinite(diff).all().item())}',
                flush=True,
            )

        _report('CURRENT', sol_output, tau)

        from .sol_kernel import sol_attn_query_compressed_sm120
        for candidate_tau in (-3.0, -4.0):
            torch.cuda.synchronize(q.device)
            started = time.perf_counter()
            candidate = sol_attn_query_compressed_sm120(
                q_probe,
                storage,
                q_offset=query_offset,
                tau=candidate_tau,
                sink_blocks=sink_blocks,
                sink_q=(0, 0),
            )
            torch.cuda.synchronize(q.device)
            _report(
                'ROUTED', candidate, candidate_tau,
                (time.perf_counter() - started) * 1000.0,
            )
            del candidate

        global_query_block = query_offset // 64
        torch.cuda.synchronize(q.device)
        started = time.perf_counter()
        forced_exact = sol_attn_query_compressed_sm120(
            q_probe,
            storage,
            q_offset=query_offset,
            tau=float(tau),
            sink_blocks=sink_blocks,
            sink_q=(global_query_block, global_query_block + 1),
        )
        torch.cuda.synchronize(q.device)
        _report(
            'FORCED-EXACT', forced_exact, tau,
            (time.perf_counter() - started) * 1000.0,
        )
        del exact, forced_exact
    except Exception as exc:
        _lm_print(
            '[MiniMaxH3 LongMedia][V17 MULTI-OFFSET TAU] '
            f'offset={query_offset}, diagnostic failed: '
            f'{type(exc).__name__}: {exc}',
            flush=True,
        )



def _v12_is_int8_family(state):
    """True for TensorWise INT8 and Asym W4A8 INT8; never NVFP4."""
    backend = str(state.get('model_runtime_backend', 'unknown')).lower()
    if backend != 'int8':
        return False

    profile = state.get('model_runtime_profile') or {}
    evidence = ' '.join([
        *(str(value) for value in (profile.get('layout_types') or {}).keys()),
        *(str(value) for value in (profile.get('weight_classes') or {}).keys()),
        *(str(value) for value in (profile.get('quant_evidence') or [])),
    ]).lower()
    if 'tensorcorenvfp4' in evidence or 'nvfp4' in evidence:
        return False

    tensorwise_int8 = 'tensorwiseint8layout' in evidence
    asym_w4a8_int8 = any(marker in evidence for marker in (
        'asymw4a8int8layout', 'asymw4a8', 'asym_w4a8', 'w4a8',
    ))
    return tensorwise_int8 or asym_w4a8_int8


def _v12_begin_int8_sol_forward(state):
    """Start a fresh Sol-workspace generation for one INT8/W4A8 forward."""
    if not _v12_is_int8_family(state):
        return False
    state['int8_reusable_sol_storage'] = None
    state['int8_reusable_sol_storage_key'] = None
    generation = int(state.get('v12_int8_sol_forward_generation', 0) or 0) + 1
    state['v12_int8_sol_forward_generation'] = generation
    state['v12_int8_sol_forward_active'] = True
    _lm_print(
        '[MiniMaxH3 LongMedia][V12-A INT8 SOL FORWARD SCOPE] '
        f'begin forward={generation}; fresh workspace required; '
        'NVFP4 path unchanged',
        flush=True,
    )
    return True


def _v12_release_int8_sol_forward(state, *, block_index):
    """Drop all INT8/W4A8 Sol references after the final attention call."""
    if (
        not _v12_is_int8_family(state)
        or not state.get('v12_int8_sol_forward_active')
    ):
        return False
    had_storage = state.get('int8_reusable_sol_storage') is not None
    state['int8_reusable_sol_storage'] = None
    state['int8_reusable_sol_storage_key'] = None
    state['v12_int8_sol_forward_active'] = False
    state['v12_int8_sol_forward_release_count'] = int(
        state.get('v12_int8_sol_forward_release_count', 0) or 0
    ) + 1
    _lm_print(
        '[MiniMaxH3 LongMedia][V12-A INT8 SOL FORWARD SCOPE] '
        f'release forward={state.get("v12_int8_sol_forward_generation", 0)} '
        f'after block={int(block_index)}; storage_present={had_storage}; '
        'next denoise step cannot reuse K/V or kc statistics',
        flush=True,
    )
    return had_storage


def _int8_reusable_sol_storage(state, *, tokens, heads, head_dim, device, allocator):
    """V11 reuse, forward-scoped for positively identified INT8/W4A8."""
    backend = str(state.get('model_runtime_backend', 'unknown')).lower()
    if backend not in ('int8', 'int8-convrot-w4a4'):
        return allocator(1, tokens, heads, head_dim, device), False

    base_key = (
        int(tokens), int(heads), int(head_dim),
        str(device), str(getattr(device, 'index', None)),
    )
    int8_family = _v12_is_int8_family(state)
    if int8_family:
        key = (
            int(state.get('v12_int8_sol_forward_generation', 0) or 0),
            *base_key,
        )
    else:
        # Exact V11 key/behavior for non-INT8-family paths.
        key = base_key
    storage = state.get('int8_reusable_sol_storage')
    old_key = state.get('int8_reusable_sol_storage_key')

    if storage is None or old_key != key:
        storage = allocator(1, tokens, heads, head_dim, device)
        state['int8_reusable_sol_storage'] = storage
        state['int8_reusable_sol_storage_key'] = key
        approx = 0
        for value in storage.values():
            if torch.is_tensor(value):
                approx += value.numel() * value.element_size()
        if int8_family:
            _lm_print(
                '[MiniMaxH3 LongMedia][V12-A INT8 SOL FORWARD SCOPE] allocated: '
                f'forward={state.get("v12_int8_sol_forward_generation", 0)}, '
                f'tokens={tokens}, heads={heads}, head_dim={head_dim}, '
                f'workspace={approx / (1024**2):.1f} MB',
                flush=True,
            )
        else:
            _lm_print(
                '[MiniMaxH3 LongMedia][INT8 REUSABLE SOL] allocated once: '
                f'tokens={tokens}, heads={heads}, head_dim={head_dim}, '
                f'workspace={approx / (1024**2):.1f} MB',
                flush=True,
            )
        return storage, True

    return storage, True


def _int8_pre_sol_storage_guard(state, *, block_index=None, force=False):
    """INT8-only guard immediately before Sol compressed storage allocation.

    The cast/offload stream is synchronized first so async INT8 casting errors
    cannot masquerade as Sol torch.empty() failures.
    """
    backend = str(state.get('model_runtime_backend', 'unknown')).lower()
    if backend not in ('int8', 'int8-convrot-w4a4'):
        return False
    if state.get('int8_reusable_sol_storage') is not None:
        return False
    if not torch.cuda.is_available():
        return False

    _int8_sync_cast_stream(state, block_index=block_index)

    snap = _cuda_memory_snapshot()
    if not snap:
        return False

    mb = 1024.0 ** 2
    free_mb = float(snap['driver_free']) / mb
    cached_mb = float(snap['cached']) / mb

    floor_mb = float(state.get('int8_sol_storage_free_floor_mb', 3072) or 3072)
    emergency_mb = float(
        state.get('int8_sol_storage_emergency_free_mb', 2048) or 2048
    )
    min_cached_mb = float(state.get('int8_sol_storage_min_cached_mb', 1024) or 1024)
    cooldown = int(state.get('int8_sol_storage_guard_cooldown_blocks', 4) or 4)
    cooldown_left = int(state.get('int8_sol_storage_guard_cooldown_left', 0) or 0)

    # v0.4.75 / V7: giant-refine persistent residency.
    #
    # CUDA mem_get_info() reports only driver-visible free memory; PyTorch's
    # allocator cache is still reusable by the next allocation. On the measured
    # ~127k-token global refiner, forward 2/3 entered here with low driver-free
    # memory but 2-4 GB of reusable allocator cache. The legacy guard then called
    # soft_empty_cache(), destroying useful residency and forcing the next
    # diffusion forward to fault/rebuild it again.
    #
    # For giant native INT8 sequences, suppress ONLY opportunistic trims when
    # effective reusable headroom is healthy. Forced cleanup and genuinely low
    # effective headroom still use the original trim path unchanged.
    token_count = int(
        state.get('current_token_count', 0)
        or state.get('last_token_count', 0)
        or 0
    )
    effective_mb = free_mb + cached_mb
    giant_native_int8 = bool(
        backend == 'int8'
        and token_count >= 90000
    )
    giant_preserve_floor_mb = float(
        state.get('v7_giant_residency_preserve_floor_mb', 3584) or 3584
    )
    if (
        giant_native_int8
        and not force
        and effective_mb >= giant_preserve_floor_mb
    ):
        state['v7_giant_residency_preserve_count'] = int(
            state.get('v7_giant_residency_preserve_count', 0) or 0
        ) + 1
        if cooldown_left > 0:
            state['int8_sol_storage_guard_cooldown_left'] = max(
                0, cooldown_left - 1
            )
        _lm_print(
            '[MiniMaxH3 LongMedia][V7 GIANT RESIDENCY PRESERVE] '
            f'block={block_index}; tokens={token_count}; '
            f'driver_free={free_mb:.0f}MB; cached={cached_mb:.0f}MB; '
            f'effective={effective_mb:.0f}MB; '
            f'preserve_floor={giant_preserve_floor_mb:.0f}MB; '
            'action=skip_opportunistic_trim; '
            'forced_and_true_low_headroom_cleanup_unchanged=True',
            flush=True,
        )
        return False

    # Healthy driver-visible memory: keep the allocator cache for speed.
    if not force and free_mb >= floor_mb:
        if cooldown_left > 0:
            state['int8_sol_storage_guard_cooldown_left'] = max(
                0, cooldown_left - 1
            )
        return False

    # Do not churn cache every block unless driver-free is actually dangerous.
    if (
        not force
        and cooldown_left > 0
        and free_mb >= emergency_mb
    ):
        state['int8_sol_storage_guard_cooldown_left'] = max(
            0, cooldown_left - 1
        )
        return False

    if cached_mb < min_cached_mb:
        return False

    before_free = free_mb
    before_cached = cached_mb

    try:
        gc.collect()
        comfy.model_management.soft_empty_cache()
    except Exception as exc:
        _lm_print(
            '[MiniMaxH3 LongMedia][INT8 SOL STORAGE GUARD] cleanup failed: '
            f'block={block_index}, {type(exc).__name__}: {exc}',
            flush=True,
        )
        return False

    state['int8_sol_storage_guard_cooldown_left'] = cooldown
    state['int8_sol_storage_trim_count'] = int(
        state.get('int8_sol_storage_trim_count', 0) or 0
    ) + 1

    after = _cuda_memory_snapshot()
    if after:
        free_after = float(after['driver_free']) / mb
        cached_after = float(after['cached']) / mb
        _lm_print(
            '[MiniMaxH3 LongMedia][INT8 SOL STORAGE GUARD] TRIM: '
            f'block={block_index}, driver_free {before_free:.0f}->{free_after:.0f} MB, '
            f'cached {before_cached:.0f}->{cached_after:.0f} MB, '
            f'floor={floor_mb:.0f}, emergency={emergency_mb:.0f} MB, '
            f'cooldown={cooldown}, force={bool(force)}',
            flush=True,
        )
    else:
        _lm_print(
            '[MiniMaxH3 LongMedia][INT8 SOL STORAGE GUARD] TRIM: '
            f'block={block_index}, pre_free={before_free:.0f} MB, '
            f'pre_cached={before_cached:.0f} MB, force={bool(force)}',
            flush=True,
        )
    return True




class _INT8SolStorageOOM(RuntimeError):
    """Terminal INT8 Sol-storage failure; never route to external attention."""
    pass


def _execute_h3_sol_attention(attn, x, rope_freqs, transformer_options, state, tau, measure=None):
    """Execute embedded H3 Sol.

    When sol_qkv_chunk_tokens > 0, use a two-pass streamed-Q path:
      1) project small token chunks and retain only full-sequence K/V;
      2) project Q chunks again, run rectangular Sol against full K/V, and
         overwrite the dead norm1 activation with attention output.

    This trades extra projection compute for much lower peak activation memory
    and is intended for very long single-pass sequences where the full fused
    BF16 QKV tensor itself no longer fits in VRAM.
    """
    from .sol_kernel import sol_attn_sm120, prepare_kv_sm120, sol_attn_query_sm120
    import comfy.quant_ops

    s = int(x.shape[0])
    state['last_sol_tau'] = float(tau)
    inner = int(attn.heads * attn.head_dim)
    meas = measure or (lambda _name, fn: fn())
    qkv_chunk_tokens = int(state.get('sol_qkv_chunk_tokens', 0) or 0)

    span = (transformer_options or {}).get('latentlab_sol_h3_video_span')
    sink_blocks = (0, 0)
    sink_q = (0, 0)
    if state.get('sol_sink_conditioning', 'exact_kv') != 'off' and span is not None:
        video_start, _video_stop = span
        if int(video_start) > 0:
            sink_blocks = (0, (int(video_start) + 63) // 64)
            if state.get('sol_sink_conditioning') == 'exact_kv_and_rows':
                sink_q = sink_blocks

    def _weights():
        qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
        return qw, kw

    # Existing zero-copy full-QKV path for normal sequence lengths / A-B tests.
    if qkv_chunk_tokens <= 0 or s <= qkv_chunk_tokens:
        qkv = meas('sol_qkv_proj', lambda: attn.qkv_proj(x))
        q, k, v = qkv.split(inner, dim=-1)
        q = q.view(1, s, attn.heads, attn.head_dim)
        k = k.view(1, s, attn.heads, attn.head_dim)
        v = v.view(1, s, attn.heads, attn.head_dim)

        def _rope():
            if rope_freqs is not None:
                qw, kw = _weights()
                rot = int(rope_freqs.shape[-3] * 2)
                if comfy.model_management.in_training:
                    return comfy.quant_ops.ck.rms_rope_split_half(
                        q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
                    )
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
                )
                return q, k
            return attn.q_norm(q), attn.k_norm(k)

        q2, k2 = meas('sol_rms_rope', _rope)
        out = meas(
            'sol_kernel',
            lambda: sol_attn_sm120(q2, k2, v, tau=float(tau), sink_blocks=sink_blocks, sink_q=sink_q),
        )
        del q2, k2, q, k, v, qkv, _rope
    else:
        # Long-sequence path: never retain full BF16 K/V.  Token-level K/V is
        # compressed to INT8 + per-token scales as each projection chunk is
        # produced, while Sol's 64-token routing summaries remain BF16.
        from .sol_kernel import (
            allocate_compressed_kv_sm120,
            append_compressed_kv_sm120,
            finalize_compressed_kv_sm120,
            sol_attn_query_compressed_sm120,
        )
        chunk = max(64, (qkv_chunk_tokens // 64) * 64)
        chunks = (s + chunk - 1) // chunk
        state['sol_qkv_streamed_calls'] = int(state.get('sol_qkv_streamed_calls', 0)) + 1
        state['sol_qkv_max_chunks'] = max(int(state.get('sol_qkv_max_chunks', 0)), chunks)
        if not state.get('sol_qkv_announced'):
            _lm_print(
                '[MiniMaxH3 LongMedia] Low-VRAM compressed streamed QKV enabled: '
                f'{s} tokens -> {chunks} query chunks of <= {chunk}; '
                'K/V=INT8+scale, Sol summaries=BF16, Q streamed',
                flush=True,
            )
            state['sol_qkv_announced'] = True

        H, D = int(attn.heads), int(attn.head_dim)
        _active_block = state.get('active_block_index', None)
        _int8_pre_sol_storage_guard(
            state, block_index=_active_block, force=False
        )

        _backend = str(state.get('model_runtime_backend', 'unknown')).lower()
        if _backend in ('int8', 'int8-convrot-w4a4'):
            storage, _reused = _int8_reusable_sol_storage(
                state,
                tokens=s,
                heads=H,
                head_dim=D,
                device=x.device,
                allocator=allocate_compressed_kv_sm120,
            )
        else:
            storage = meas(
                'sol_stream_kv_storage_alloc',
                lambda: allocate_compressed_kv_sm120(1, s, H, D, x.device),
            )

        # V31: keep stock Comfy quant math, but prepare each quantized projection
        # ONCE per H3 block and reuse it across all streamed token chunks.
        # This follows comfy.ops.cast_bias_weight(offloadable=True) contract:
        # cast once, use many times, uncast after the final use.
        # 0.3.112: streamed-Sol must resolve the memory mode locally.
        # _ultra_streaming used to be defined only inside the MLP path, which
        # made the long-sequence attention preflight route fail with NameError
        # before the first QKV chunk. Keep the semantic identical to the MLP
        # governor: ultra_low_vram disables cached INT8 projection residency.
        _ultra_streaming = str(state.get('memory_mode', 'normal')) == 'ultra_low_vram'
        _int8_backend = (
            _v12_is_int8_family(state)
            and not comfy.model_management.in_training
            and not _ultra_streaming
        )
        if _ultra_streaming and _v12_is_int8_family(state) and not state.get('v354_ultra_mlp_stream_announced'):
            _lm_print(
                '[MiniMaxH3 LongMedia][ULTRA MLP STREAM] '
                'cached fc1+fc2 residency disabled; stock Comfy MLP streams linear weights sequentially per token chunk',
                flush=True,
            )
            state['v354_ultra_mlp_stream_announced'] = True
        _qkv_handle = None
        _out_handle = None
        _v19_active = _v19_selected_block(state, _active_block)
        _v19_offsets = _v19_probe_offsets(s, chunk) if _v19_active else []
        _v19_out_inputs = []
        _v19_out_cached = []
        if _int8_backend:
            _probe = x[:4]
            _stock_qkv_probe = None
            if _v12b_linear_ab_enabled(state, 'qkv_proj'):
                _stock_qkv_probe = attn.qkv_proj(_probe).detach()
            _v19_qkv_input = None
            _v19_qkv_stock = None
            if _v19_active:
                _v19_qkv_input = torch.cat([
                    x[int(offset):min(s, int(offset) + 4)]
                    for offset in _v19_offsets
                ], dim=0).detach().clone()
                _v19_qkv_stock = attn.qkv_proj(_v19_qkv_input).detach()
            _qkv_handle = _int8_prepare_block_linear(attn.qkv_proj, _probe)
            if _v19_qkv_stock is not None:
                _v19_qkv_cached = _int8_cached_linear(
                    _qkv_handle, _v19_qkv_input
                )
                _v19_report(
                    state, 'QKV-PROJ', _v19_qkv_stock, _v19_qkv_cached,
                    offsets=_v19_offsets,
                )
                del _v19_qkv_input, _v19_qkv_stock, _v19_qkv_cached
            if _stock_qkv_probe is not None:
                _cached_qkv_probe = _int8_cached_linear(
                    _qkv_handle, _probe
                )
                _v12b_linear_ab_report(
                    state, 'qkv_proj', _stock_qkv_probe, _cached_qkv_probe
                )
                del _stock_qkv_probe, _cached_qkv_probe
            if not state.get('int8_semantic_dispatch_announced'):
                _lm_print(
                    '[MiniMaxH3 LongMedia][INT8 SEMANTIC DISPATCH] '
                    'ordinary qkv/fc1/out use F.linear + QuantizedTensor dispatch; '
                    'fc2 SwiGLU mirrors comfy.ops.linear_input_act',
                    flush=True,
                )
                state['int8_semantic_dispatch_announced'] = True

        qw = kw = None
        if rope_freqs is not None:
            qw, kw = _weights()
            rot = int(rope_freqs.shape[-3] * 2)

        # V18: V13-V17 compared Sol against an exact reference reconstructed
        # from the already-compressed INT8 K/V store.  That proved the routed
        # kernel, but could not measure information lost by K/V compression.
        # For block 0 / forward 1 only, retain four tiny Q probes and stream an
        # online exact-softmax reference over the original BF16 K/V chunks
        # before each chunk is compressed.  Full BF16 K/V is never retained.
        _v18 = None
        if (
            _v12_is_int8_family(state)
            and int(state.get('active_block_index', -1)) == 0
            and int(state.get('v12_int8_sol_forward_generation', 0) or 0) == 1
            and not bool(state.get('v18_bf16_kv_reference_done', False))
        ):
            _v18_targets = sorted({
                0,
                min(chunk, max(0, s - 1)),
                (s // 2 // chunk) * chunk,
                ((s - 1) // chunk) * chunk,
            })
            _v18 = {'targets': {}, 'started': time.perf_counter()}
            for _target in _v18_targets:
                _probe_end = min(s, int(_target) + 4)
                if _qkv_handle is not None:
                    _probe_qkv = _int8_cached_linear(
                        _qkv_handle, x[int(_target):_probe_end]
                    )
                else:
                    _probe_qkv = attn.qkv_proj(x[int(_target):_probe_end])
                _probe_q, _probe_k_dead, _probe_v_dead = _probe_qkv.split(
                    inner, dim=-1
                )
                _probe_n = _probe_end - int(_target)
                _probe_q = _probe_q.view(1, _probe_n, H, D)
                _probe_k_dead = _probe_k_dead.view(1, _probe_n, H, D)
                if rope_freqs is not None:
                    _probe_rf = rope_freqs[:, int(_target):_probe_end]
                    if comfy.model_management.in_training:
                        _probe_q, _probe_k_dead = (
                            comfy.quant_ops.ck.rms_rope_split_half(
                                _probe_q, _probe_k_dead, _probe_rf, qw, kw,
                                epsilon=attn.q_norm.eps, rot_dim=rot,
                            )
                        )
                    else:
                        comfy.quant_ops.ck.rms_rope_split_half_(
                            _probe_q, _probe_k_dead, _probe_rf, qw, kw,
                            epsilon=attn.q_norm.eps, rot_dim=rot,
                        )
                else:
                    _probe_q = attn.q_norm(_probe_q)
                _probe_q32 = _probe_q.detach().to(dtype=torch.float32).clone()
                _probe_queries = int(_probe_q32.shape[1])
                _running_max = torch.full(
                    (1, H, _probe_queries), -float('inf'),
                    device=x.device, dtype=torch.float32,
                )
                _v18['targets'][int(_target)] = {
                    'q': _probe_q32,
                    'running_max': _running_max,
                    'running_sum': torch.zeros_like(_running_max),
                    'running_out': torch.zeros(
                        (1, H, _probe_queries, D),
                        device=x.device, dtype=torch.float32,
                    ),
                }
                del _probe_qkv, _probe_q, _probe_k_dead, _probe_v_dead
                del _probe_q32, _running_max
            _lm_print(
                '[MiniMaxH3 LongMedia][V18 BF16-KV REFERENCE] '
                f'prepared offsets={_v18_targets}, queries_per_offset=4; '
                'streaming exact reference before K/V compression',
                flush=True,
            )

        def _v18_accumulate_original_bf16(k_chunk, v_chunk):
            if _v18 is None:
                return
            k32 = k_chunk.detach().to(dtype=torch.float32)
            v32 = v_chunk.detach().to(dtype=torch.float32)
            attention_scale = float(D ** -0.5)
            for _item in _v18['targets'].values():
                scores = torch.einsum(
                    'bqhd,bkhd->bhqk', _item['q'], k32
                ).mul_(attention_scale)
                chunk_max = scores.amax(dim=-1)
                next_max = torch.maximum(_item['running_max'], chunk_max)
                previous_scale = torch.exp(_item['running_max'] - next_max)
                probability = torch.exp(scores - next_max.unsqueeze(-1))
                _item['running_out'] = (
                    _item['running_out'] * previous_scale.unsqueeze(-1)
                    + torch.einsum('bhqk,bkhd->bhqd', probability, v32)
                )
                _item['running_sum'] = (
                    _item['running_sum'] * previous_scale
                    + probability.sum(dim=-1)
                )
                _item['running_max'] = next_max
                del scores, chunk_max, next_max, previous_scale, probability
            del k32, v32

        def _v18_finalize_original_bf16():
            if _v18 is None:
                return
            for _item in _v18['targets'].values():
                _item['reference'] = (
                    _item['running_out']
                    / _item['running_sum'].unsqueeze(-1)
                ).permute(0, 2, 1, 3).contiguous()
                del _item['running_max'], _item['running_sum']
                del _item['running_out']
            elapsed_ms = (time.perf_counter() - _v18['started']) * 1000.0
            _lm_print(
                '[MiniMaxH3 LongMedia][V18 BF16-KV REFERENCE] '
                f'original BF16 exact references ready, elapsed_ms={elapsed_ms:.1f}',
                flush=True,
            )

        def _build_compressed_kv():
            for start in range(0, s, chunk):
                end = min(s, start + chunk)
                _split_kv = None
                if _qkv_handle is not None:
                    _split_kv = _v32_quant_linear_rows(
                        _qkv_handle, x[start:end], inner, inner * 3
                    )
                if _split_kv is not None:
                    _k, _v = _split_kv.split(inner, dim=-1)
                    qkv_part = None
                else:
                    if _qkv_handle is not None:
                        qkv_part = _int8_cached_linear(_qkv_handle, x[start:end])
                    else:
                        qkv_part = attn.qkv_proj(x[start:end])
                    _q_dead, _k, _v = qkv_part.split(inner, dim=-1)
                n = end - start
                _k = _k.view(1, n, H, D)
                _v = _v.view(1, n, H, D)
                if rope_freqs is not None:
                    rf = rope_freqs[:, start:end]
                    # The public paired RMS+RoPE op is the only H3 path exposing
                    # rot_dim. Reuse K as the discarded Q operand so K remains
                    # bit-faithful while avoiding the expensive Q projection.
                    _q_dummy = _k.clone()
                    if comfy.model_management.in_training:
                        _q_dummy, _k = comfy.quant_ops.ck.rms_rope_split_half(
                            _q_dummy, _k, rf, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
                        )
                    else:
                        comfy.quant_ops.ck.rms_rope_split_half_(
                            _q_dummy, _k, rf, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
                        )
                    del _q_dummy
                else:
                    _k = attn.k_norm(_k)
                _v18_accumulate_original_bf16(_k, _v)
                append_compressed_kv_sm120(storage, _k, _v, start)
                if qkv_part is not None:
                    del qkv_part
                if '_q_dead' in locals():
                    del _q_dead
                del _split_kv, _k, _v
            return storage

        # V40 production baseline: diagnostic CUDA-sync profiling is disabled.
        # Execution math, chunking, SOL routing and quantized kernels are unchanged from V39.
        _v33_profile = False
        _v35_forensic = False
        _v35_chunks = []
        if _v35_forensic and not state.get('v35_forensic_announced'):
            _lm_print(
                '[MiniMaxH3 LongMedia][V35 FORENSIC] enabled for blocks 0..2; '
                'collecting AUTO decision, tau, per-query-chunk exact/approx routing, '
                'thresholds, threshold-kernel time, SOL-forward time, Q/out/MLP and VRAM',
                flush=True,
            )
            state['v35_forensic_announced'] = True
        if _v35_forensic:
            _lm_print(
                '[MiniMaxH3 LongMedia][V35 POLICY] '
                f'block={int(_active_block)} requested={state.get("requested_attention_mode")} '
                f'effective={state.get("sol_mode")} tokens={s} tau={float(tau):.4f} '
                f'tau_start={float(state.get("sol_tau_start", 0.0)):.4f} '
                f'tau_end={float(state.get("sol_tau_end", 0.0)):.4f} '
                f'curve={state.get("sol_curve")} sink_blocks={sink_blocks} sink_q={sink_q} '
                f'qkv_chunk={chunk} query_chunks={chunks} kv_blocks={int(storage.get("blocks", 0))}',
                flush=True,
            )
        if _v33_profile:
            torch.cuda.synchronize()
            _v33_kv_t0 = time.perf_counter()
        meas('sol_stream_kv_projection_compress', _build_compressed_kv)
        meas('sol_stream_kv_summaries', lambda: finalize_compressed_kv_sm120(storage))
        if _v33_profile:
            torch.cuda.synchronize()
            state['v33_last_kvpass_s'] = time.perf_counter() - _v33_kv_t0
        _v18_finalize_original_bf16()

        def _v18_report(offset, q, current):
            if _v18 is None or int(offset) not in _v18['targets']:
                return
            item = _v18['targets'][int(offset)]
            query_count = int(item['reference'].shape[1])
            q_replay = q[:, :query_count].detach().to(dtype=torch.float32)
            q_reference = item['q'][:, :query_count]
            q_delta = q_replay - q_reference
            q_ref_rms = float(
                torch.sqrt(torch.mean(q_reference.square())).item()
            )
            q_rel_rms = float(
                torch.sqrt(torch.mean(q_delta.square())).item()
            ) / max(1.0e-12, q_ref_rms)

            compressed_exact = _v13_exact_attention_from_compressed(
                q[:, :query_count], storage, key_chunk=1024
            )
            from .sol_kernel import sol_attn_query_compressed_sm120
            probe_rows = min(64, int(q.shape[1]))
            torch.cuda.synchronize(q.device)
            started = time.perf_counter()
            tau3 = sol_attn_query_compressed_sm120(
                q[:, :probe_rows], storage, q_offset=int(offset), tau=-3.0,
                sink_blocks=sink_blocks, sink_q=sink_q,
            )
            torch.cuda.synchronize(q.device)
            tau3_ms = (time.perf_counter() - started) * 1000.0

            reference = item['reference'].detach().to(
                device='cpu', dtype=torch.float32
            ).flatten()

            def _report(label, candidate, elapsed_ms=None):
                candidate_flat = (
                    candidate[:, :query_count]
                    .detach().to(device='cpu', dtype=torch.float32).flatten()
                )
                diff = candidate_flat - reference
                eps = 1.0e-12
                rms_ref = float(torch.sqrt(torch.mean(reference.square())).item())
                rms_diff = float(torch.sqrt(torch.mean(diff.square())).item())
                denom = float(
                    torch.linalg.vector_norm(reference).item()
                    * torch.linalg.vector_norm(candidate_flat).item()
                )
                cosine = float(
                    torch.dot(reference, candidate_flat).item()
                    / max(eps, denom)
                )
                timing = (
                    '' if elapsed_ms is None
                    else f', elapsed_ms={float(elapsed_ms):.1f}'
                )
                _lm_print(
                    '[MiniMaxH3 LongMedia][V18 BF16-KV REFERENCE] '
                    f'offset={int(offset)}, label={label}{timing}, '
                    f'rel_rms={rms_diff / max(eps, rms_ref):.8e}, '
                    f'cosine={cosine:.10f}, '
                    f'max_abs={float(diff.abs().max().item()):.8e}, '
                    f'q_replay_rel_rms={q_rel_rms:.8e}, '
                    f'finite={bool(torch.isfinite(diff).all().item())}',
                    flush=True,
                )

            _report('COMPRESSED-EXACT', compressed_exact)
            _report('CURRENT-TAU-2', current)
            _report('SOL-TAU-3', tau3, tau3_ms)
            del compressed_exact, tau3, reference, q_replay, q_reference, q_delta
            del item['q'], item['reference']

        # Important geometry fix: attention inner width is 7168 on H3 while
        # hidden width is 5376.  Never write raw attention output into x.
        # Instead each streamed Q chunk immediately runs out_proj and only the
        # projected [tokens, hidden] result overwrites the dead norm1 slice.
        def _stream_queries_and_project():
            nonlocal _out_handle, tau
            _v34_qproj_s = 0.0
            _v34_rope_s = 0.0
            _v34_sol_s = 0.0
            _v34_outproj_s = 0.0
            _v34_copy_s = 0.0
            for start in range(0, s, chunk):
                end = min(s, start + chunk)
                _split_q = None
                if _v33_profile:
                    torch.cuda.synchronize()
                    _v34_t0 = time.perf_counter()
                if _qkv_handle is not None:
                    _split_q = _v32_quant_linear_rows(
                        _qkv_handle, x[start:end], 0, inner
                    )
                if _split_q is not None:
                    _q = _split_q
                    qkv_part = None
                else:
                    if _qkv_handle is not None:
                        qkv_part = _int8_cached_linear(_qkv_handle, x[start:end])
                    else:
                        qkv_part = attn.qkv_proj(x[start:end])
                    _q, _k_dead, _v_dead = qkv_part.split(inner, dim=-1)
                if _v33_profile:
                    torch.cuda.synchronize()
                    _v34_qproj_s += time.perf_counter() - _v34_t0
                    _v34_t0 = time.perf_counter()
                n = end - start
                _q = _q.view(1, n, H, D)
                if rope_freqs is not None:
                    rf = rope_freqs[:, start:end]
                    # Keep exact H3 q RMS+RoPE semantics; paired public op carries
                    # rot_dim, while the K result is intentionally discarded.
                    _k_dummy = _q.clone()
                    if comfy.model_management.in_training:
                        _q, _k_dummy = comfy.quant_ops.ck.rms_rope_split_half(
                            _q, _k_dummy, rf, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
                        )
                    else:
                        comfy.quant_ops.ck.rms_rope_split_half_(
                            _q, _k_dummy, rf, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
                        )
                    del _k_dummy
                else:
                    _q = attn.q_norm(_q)
                if _v33_profile:
                    torch.cuda.synchronize()
                    _v34_rope_s += time.perf_counter() - _v34_t0
                    _v34_t0 = time.perf_counter()
                if start == 0:
                    _v37_tau = _v37_auto_calibrate_sol_tau(
                        state, _q, storage, sink_blocks, sink_q, start
                    )
                    if _v37_tau is not None:
                        tau = float(_v37_tau)
                        state['last_sol_tau'] = float(tau)
                _v35_ret = sol_attn_query_compressed_sm120(
                    _q, storage, q_offset=start, tau=float(tau),
                    sink_blocks=sink_blocks, sink_q=sink_q,
                    telemetry=_v35_forensic,
                )
                if _v35_forensic:
                    q_out, _v35_stat = _v35_ret
                    _v35_stat['start'] = int(start)
                    _v35_stat['end'] = int(end)
                    _v35_chunks.append(_v35_stat)
                    _lm_print(
                        '[MiniMaxH3 LongMedia][V35 SOL CHUNK] '
                        f'block={int(_active_block)} q={int(start)}:{int(end)} '
                        f'tau={float(tau):.3f} exact={_v35_stat["exact_ratio"]*100.0:.2f}% '
                        f'exact_range={_v35_stat["exact_ratio_min"]*100.0:.1f}-'
                        f'{_v35_stat["exact_ratio_max"]*100.0:.1f}% '
                        f'score_routes={_v35_stat["score_routed"]} '
                        f'local_forced={_v35_stat["local_forced"]} '
                        f'sink_forced={_v35_stat["sink_forced"]} '
                        f'q_sink_programs={_v35_stat["q_sink_programs"]} '
                        f'thr={_v35_stat["threshold_min"]:.3f}/'
                        f'{_v35_stat["threshold_mean"]:.3f}/'
                        f'{_v35_stat["threshold_max"]:.3f} '
                        f't_threshold={_v35_stat["threshold_s"]:.4f}s '
                        f't_forward={_v35_stat["forward_s"]:.4f}s',
                        flush=True,
                    )
                else:
                    q_out = _v35_ret
                if _v33_profile:
                    torch.cuda.synchronize()
                    _v34_sol_s += time.perf_counter() - _v34_t0
                    _v34_t0 = time.perf_counter()
                _v18_report(start, _q, q_out)
                if _int8_backend and _out_handle is None:
                    _out_probe = q_out.view(n, inner)[:4]
                    _stock_out_probe = None
                    if _v12b_linear_ab_enabled(state, 'out_proj'):
                        _stock_out_probe = attn.out_proj(_out_probe).detach()
                    _out_handle = _int8_prepare_block_linear(
                        attn.out_proj, _out_probe
                    )
                    if _stock_out_probe is not None:
                        _cached_out_probe = _int8_cached_linear(
                            _out_handle, _out_probe
                        )
                        _v12b_linear_ab_report(
                            state, 'out_proj',
                            _stock_out_probe, _cached_out_probe,
                        )
                        del _stock_out_probe, _cached_out_probe
                if _out_handle is not None:
                    projected = _int8_cached_linear(
                        _out_handle, q_out.view(n, inner)
                    )
                else:
                    projected = attn.out_proj(q_out.view(n, inner))
                if _v33_profile:
                    torch.cuda.synchronize()
                    _v34_outproj_s += time.perf_counter() - _v34_t0
                    _v34_t0 = time.perf_counter()
                if _v19_active and int(start) in _v19_offsets:
                    _v19_rows = min(4, n)
                    _v19_out_inputs.append(
                        q_out.view(n, inner)[:_v19_rows].detach().clone()
                    )
                    _v19_out_cached.append(
                        projected[:_v19_rows].detach().clone()
                    )
                x[start:end].copy_(projected)
                if _v33_profile:
                    torch.cuda.synchronize()
                    _v34_copy_s += time.perf_counter() - _v34_t0
                if qkv_part is not None:
                    del qkv_part
                if '_k_dead' in locals():
                    del _k_dead
                if '_v_dead' in locals():
                    del _v_dead
                del _split_q, _q, q_out, projected
            if _v35_forensic and _v35_chunks:
                _ex = [z['exact_ratio'] for z in _v35_chunks]
                _tf = [z['forward_s'] for z in _v35_chunks]
                _tt = [z['threshold_s'] for z in _v35_chunks]
                _exact = sum(z['exact'] for z in _v35_chunks)
                _approx = sum(z['approx'] for z in _v35_chunks)
                _den = max(1, _exact + _approx)
                _alloc = torch.cuda.memory_allocated(x.device) / (1024.0 ** 2)
                _reserved = torch.cuda.memory_reserved(x.device) / (1024.0 ** 2)
                _free, _total = torch.cuda.mem_get_info(x.device)
                _lm_print(
                    '[MiniMaxH3 LongMedia][V35 SOL SUMMARY] '
                    f'block={int(_active_block)} tau={float(tau):.3f} chunks={len(_v35_chunks)} '
                    f'exact_global={100.0*_exact/_den:.2f}% '
                    f'exact_chunk_min/avg/max={100.0*min(_ex):.2f}/'
                    f'{100.0*sum(_ex)/len(_ex):.2f}/{100.0*max(_ex):.2f}% '
                    f'forward_chunk_min/avg/max={min(_tf):.4f}/'
                    f'{sum(_tf)/len(_tf):.4f}/{max(_tf):.4f}s '
                    f'threshold_total={sum(_tt):.4f}s forward_total={sum(_tf):.4f}s '
                    f'alloc={_alloc:.0f}MB reserved={_reserved:.0f}MB driver_free={_free/(1024.0**2):.0f}MB',
                    flush=True,
                )
                state['v35_last_sol_summary'] = {
                    'block': int(_active_block), 'tau': float(tau),
                    'exact_global': float(_exact) / float(_den),
                    'threshold_total_s': float(sum(_tt)),
                    'forward_total_s': float(sum(_tf)),
                }
            if _v33_profile:
                state['v34_last_qproj_s'] = _v34_qproj_s
                state['v34_last_rope_s'] = _v34_rope_s
                state['v34_last_sol_s'] = _v34_sol_s
                state['v34_last_outproj_s'] = _v34_outproj_s
                state['v34_last_copy_s'] = _v34_copy_s
            return x

        try:
            if _v33_profile:
                torch.cuda.synchronize()
                _v33_query_t0 = time.perf_counter()
            result = meas('sol_stream_query_kernel_outproj', _stream_queries_and_project)
            if _v33_profile:
                torch.cuda.synchronize()
                state['v33_last_querypass_s'] = time.perf_counter() - _v33_query_t0
        finally:
            if _out_handle is not None:
                _int8_release_block_linear(_out_handle)
            if _qkv_handle is not None:
                _int8_release_block_linear(_qkv_handle)
            if _v18 is not None:
                _v18['targets'].clear()
                state['v18_bf16_kv_reference_done'] = True
        if _v19_active and _v19_out_inputs:
            _v19_out_input = torch.cat(_v19_out_inputs, dim=0)
            _v19_out_stock = attn.out_proj(_v19_out_input).detach()
            _v19_out_got = torch.cat(_v19_out_cached, dim=0)
            _v19_report(
                state, 'OUT-PROJ', _v19_out_stock, _v19_out_got,
                offsets=_v19_offsets,
            )
            del _v19_out_input, _v19_out_stock, _v19_out_got
            _v19_out_inputs.clear()
            _v19_out_cached.clear()
        if _backend not in ('int8', 'int8-convrot-w4a4'):
            del storage
        del qw, kw
        return result, sink_blocks

    def _out_proj():
        flat = out.view(s, inner)
        chunk_tokens = int(state.get('sol_out_proj_chunk_tokens', 24576))
        if chunk_tokens <= 0 or s <= chunk_tokens:
            return attn.out_proj(flat)

        chunks = (s + chunk_tokens - 1) // chunk_tokens
        state['sol_out_proj_chunked_calls'] = int(state.get('sol_out_proj_chunked_calls', 0)) + 1
        state['sol_out_proj_max_chunks'] = max(int(state.get('sol_out_proj_max_chunks', 0)), chunks)
        if not state.get('sol_out_proj_announced'):
            _lm_print(
                '[MiniMaxH3 LongMedia] Low-VRAM Sol out_proj enabled: '
                f'{s} tokens -> {chunks} chunks of <= {chunk_tokens}',
                flush=True,
            )
            state['sol_out_proj_announced'] = True

        # In streamed-QKV mode `out` reuses the dead norm1 activation `x`.
        # Token-wise out_proj can therefore overwrite each already-consumed
        # input slice in place, avoiding another full [S, hidden] allocation.
        reuse_input = out.data_ptr() == x.data_ptr()
        projected = flat if reuse_input else torch.empty_like(x)
        for start in range(0, s, chunk_tokens):
            end = min(s, start + chunk_tokens)
            part = attn.out_proj(flat[start:end])
            projected[start:end].copy_(part)
            del part
        return projected

    result = meas('sol_out_proj', _out_proj)
    return result, sink_blocks

def _sol_exception_is_oom(exc):
    msg = f"{type(exc).__name__}: {exc}".lower()
    return (
        'outofmemoryerror' in msg
        or 'out of memory' in msg
        or 'allocation on device' in msg
        or 'cuda oom' in msg
        or 'would exceed allowed memory' in msg
    )


def _sol_retry_chunk_schedule(current_chunk):
    cur = int(current_chunk or 0)
    if cur <= 0:
        return []
    ladder = []
    for candidate in (cur // 2, 4096, 2048, 1024):
        candidate = max(64, (int(candidate) // 64) * 64)
        if candidate > 0 and candidate < cur and candidate not in ladder:
            ladder.append(candidate)
    return ladder




class _FastH3VSANotPlainT2VA(RuntimeError):
    pass



def _longmedia_fasth3_vsa_blockwise(
    q, k, v, tau=1.0, scale=None, sink_blocks=None, sink_q=None, key_bias=None,
    topk_ratio=0.0, tail=True, block_len=None, coarse_gate=None,
):
    """Memory-bounded learned-VSA executor for older Comfy Kitchen builds.

    Implements the FastH3 Preview serving contract directly: tile-64 routing,
    per-head top-k fine attention, exact sink/neighbor blocks, and the learned
    coarse VSA gate. It never materializes a full T x T score matrix.
    """
    import torch

    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            '[FastH3 VSA fallback] q/k/v must share shape (B,T,H,D), got '
            f'{tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}'
        )
    if key_bias is not None:
        raise ValueError('[FastH3 VSA fallback] key_bias is not part of the FastH3 Preview contract')
    if tail:
        raise ValueError('[FastH3 VSA fallback] FastH3 Preview requires tail=False')
    if float(topk_ratio) <= 0.0:
        raise ValueError('[FastH3 VSA fallback] FastH3 Preview requires topk_ratio > 0')

    bsz, total, heads, dim = map(int, q.shape)
    block = 64
    if total % block:
        raise ValueError(
            '[FastH3 VSA fallback] packed sequence must be padded to tile-64, '
            f'got T={total}'
        )
    nblocks = total // block
    if scale is None:
        scale = dim ** -0.5
    scale = float(scale)

    if block_len is None:
        lengths = torch.full((nblocks,), block, device=q.device, dtype=torch.int32)
    else:
        if int(block_len.numel()) != nblocks:
            raise ValueError(
                '[FastH3 VSA fallback] block_len size mismatch: '
                f'{int(block_len.numel())} != {nblocks}'
            )
        lengths = block_len.to(device=q.device, dtype=torch.int32).clamp(1, block)

    pos = torch.arange(block, device=q.device, dtype=torch.int32)
    live = pos.view(1, block) < lengths.view(nblocks, 1)
    live_f = live.to(dtype=torch.float32).view(1, nblocks, block, 1, 1)
    denom = lengths.to(dtype=torch.float32).view(1, nblocks, 1, 1).clamp_min(1.0)

    qb = q.view(bsz, nblocks, block, heads, dim)
    kb = k.view(bsz, nblocks, block, heads, dim)
    vb = v.view(bsz, nblocks, block, heads, dim)

    qmean = (qb.float() * live_f).sum(dim=2) / denom
    kmean = (kb.float() * live_f).sum(dim=2) / denom
    vmean = (vb.float() * live_f).sum(dim=2) / denom

    # Centering K, as the reference implementation does, only subtracts a
    # constant from every key score for a query and cannot alter top-k routing.
    route = torch.einsum('bqhd,bkhd->bhqk', qmean, kmean) * scale

    sink0, sink1 = (list(sink_blocks or [0, 0]) + [0, 0])[:2]
    sink0 = max(0, min(nblocks, int(sink0)))
    sink1 = max(sink0, min(nblocks, int(sink1)))
    sinkq0, sinkq1 = (list(sink_q or [0, 0]) + [0, 0])[:2]
    sinkq0 = max(0, min(nblocks, int(sinkq0)))
    sinkq1 = max(sinkq0, min(nblocks, int(sinkq1)))

    eligible = nblocks - (sink1 - sink0)
    if eligible <= 1:
        keep = 0
    else:
        keep = max(1, round(float(topk_ratio) * eligible))
        keep = min(eligible - 1, keep)

    exact = torch.zeros((bsz, heads, nblocks, nblocks), device=q.device, dtype=torch.bool)
    if keep:
        ranked = route.clone()
        if sink1 > sink0:
            ranked[..., sink0:sink1] = float('-inf')
        top = ranked.topk(keep, dim=-1).indices
        exact.scatter_(-1, top, True)

    idx = torch.arange(nblocks, device=q.device)
    near = (idx.view(-1, 1) - idx.view(1, -1)).abs() <= 1
    exact |= near.view(1, 1, nblocks, nblocks)
    if sink1 > sink0:
        exact[..., sink0:sink1] = True
    if sinkq1 > sinkq0:
        exact[:, :, sinkq0:sinkq1, :] = True

    out = torch.empty_like(q)
    k_by_head = kb.permute(0, 3, 1, 2, 4).contiguous()  # [B,H,N,64,D]
    v_by_head = vb.permute(0, 3, 1, 2, 4).contiguous()
    gate_blocks = None
    if coarse_gate is not None:
        if coarse_gate.shape != q.shape:
            raise ValueError(
                '[FastH3 VSA fallback] coarse_gate must match q shape, got '
                f'{tuple(coarse_gate.shape)} vs {tuple(q.shape)}'
            )
        gate_blocks = coarse_gate.view(bsz, nblocks, block, heads, dim)

    token_pos = torch.arange(block, device=q.device).view(1, 1, block)
    for bi in range(bsz):
        for qi in range(nblocks):
            mask = exact[bi, :, qi, :]  # [H,N]
            # Keep the gather width deterministic on the host. Avoid `.item()`
            # here: a device scalar read would synchronize CUDA once per query
            # block and erase a meaningful part of VSA's speedup.
            if sinkq0 <= qi < sinkq1:
                max_sel = nblocks
            else:
                max_sel = min(nblocks, max(1, keep + (sink1 - sink0) + 3))
            sel = mask.to(torch.float32).topk(max_sel, dim=-1).indices
            sel_valid = torch.gather(mask, 1, sel)

            gather_idx = sel[:, :, None, None].expand(heads, max_sel, block, dim)
            ks = torch.gather(k_by_head[bi], 1, gather_idx).reshape(heads, max_sel * block, dim)
            vs = torch.gather(v_by_head[bi], 1, gather_idx).reshape(heads, max_sel * block, dim)

            sel_len = lengths[sel]
            key_live = (token_pos < sel_len[:, :, None]) & sel_valid[:, :, None]
            key_live = key_live.reshape(heads, max_sel * block)

            qs = qb[bi, qi].permute(1, 0, 2).contiguous()  # [H,64,D]
            scores = torch.matmul(qs, ks.transpose(-1, -2)).float().mul_(scale)
            scores.masked_fill_(~key_live[:, None, :], float('-inf'))
            probs = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
            fine = torch.matmul(probs, vs)

            if gate_blocks is not None:
                coarse_scores = torch.einsum(
                    'hd,khd->hk', qmean[bi, qi], kmean[bi]
                ).mul_(scale)
                coarse_probs = torch.softmax(coarse_scores, dim=-1)
                coarse = torch.einsum('hk,khd->hd', coarse_probs, vmean[bi]).to(dtype=v.dtype)
                gate = gate_blocks[bi, qi].permute(1, 0, 2).contiguous()
                fine = fine + gate * coarse[:, None, :]

            out[bi, qi * block:(qi + 1) * block] = fine.permute(1, 0, 2).contiguous()

    return out.contiguous()


_longmedia_fasth3_vsa_blockwise._longmedia_executor_label = 'LongMedia blockwise learned-VSA fallback'


def _fast_h3_vsa_executor():
    """Resolve learned-VSA without requiring a specific Comfy Kitchen release."""
    try:
        import inspect
        import comfy_kitchen
        fn = getattr(comfy_kitchen, 'sol_attn', None)
        if callable(fn):
            params = inspect.signature(fn).parameters
            required = {'topk_ratio', 'tail', 'block_len', 'coarse_gate'}
            if required.issubset(params):
                try:
                    fn._longmedia_executor_label = 'Comfy Kitchen sol_attn'
                except Exception:
                    pass
                return fn, None
            reason = 'comfy_kitchen.sol_attn lacks learned-VSA arguments'
        else:
            reason = 'comfy_kitchen.sol_attn is not exported'
    except Exception as exc:
        reason = f'{type(exc).__name__}: {exc}'
    return _longmedia_fasth3_vsa_blockwise, reason

def _fast_h3_vsa_geometry(layout, sequence, device, transformer_options):
    """Build/cache tile-64 destination map for FastH3 Preview plain T2VA packing."""
    segments = list(getattr(layout, 'segments', ()))
    kinds = [str(item[2]) for item in segments]
    if kinds != ['text', 'audio', 'video']:
        raise _FastH3VSANotPlainT2VA(
            'FastH3 Preview v1 learned VSA supports plain T2VA packing only'
        )
    signature = tuple(getattr(layout, 'signature', ()))
    if len(signature) != 5:
        raise _FastH3VSANotPlainT2VA('PackedLayout has no standard H3 signature')
    _text, latent_t, latent_h, latent_w, _audio_t = map(int, signature)
    video_shape = (latent_t, latent_h // 2, latent_w // 2)
    expected_video = int(math.prod(video_shape))
    actual_video = int(segments[-1][1]) - int(segments[-1][0])
    if expected_video != actual_video or int(segments[-1][1]) != int(sequence):
        raise _FastH3VSANotPlainT2VA(
            f'video tile geometry mismatch expected={expected_video} actual={actual_video} seq={sequence}'
        )

    key = (signature, tuple((int(a), int(b), str(k)) for a,b,k in segments), str(device))
    cache = transformer_options.setdefault('_longmedia_fasth3_vsa_geometry_cache', {})
    if key in cache:
        return cache[key]

    tile_t = tile_h = tile_w = 4
    tile_rows = 64
    source_in_tile_order = []
    block_lengths = []
    source_offset = 0
    for a, b, _kind in segments[:-1]:
        length = int(b) - int(a)
        for start in range(0, length, tile_rows):
            count = min(tile_rows, length - start)
            source_in_tile_order.extend(range(source_offset + start, source_offset + start + count))
            block_lengths.append(count)
        source_offset += length
    prefix_blocks = len(block_lengths)

    frames, height, width = video_shape
    for t0 in range(0, frames, tile_t):
        for h0 in range(0, height, tile_h):
            for w0 in range(0, width, tile_w):
                tile_sources = []
                for t in range(t0, min(t0 + tile_t, frames)):
                    for h in range(h0, min(h0 + tile_h, height)):
                        for w in range(w0, min(w0 + tile_w, width)):
                            tile_sources.append(source_offset + (t * height + h) * width + w)
                source_in_tile_order.extend(tile_sources)
                block_lengths.append(len(tile_sources))

    if len(source_in_tile_order) != int(sequence):
        raise RuntimeError(
            '[FastH3 VSA PRECHECK] tile geometry did not cover packed sequence: '
            f'{len(source_in_tile_order)} != {sequence}'
        )
    destination = torch.empty(int(sequence), dtype=torch.long)
    cursor = 0
    for block_index, count in enumerate(block_lengths):
        src = source_in_tile_order[cursor:cursor + count]
        destination[torch.tensor(src, dtype=torch.long)] = (
            block_index * tile_rows + torch.arange(count, dtype=torch.long)
        )
        cursor += count
    padded_rows = len(block_lengths) * tile_rows
    result = (
        destination.to(device=device),
        torch.tensor(block_lengths, dtype=torch.int32, device=device),
        int(prefix_blocks),
        int(padded_rows),
    )
    cache[key] = result
    return result


def _run_h3_fasth3_vsa_attention(attn, x, rope_freqs, transformer_options, state):
    """Learned VSA path shared by H3ddle FastH3 and Kijai FastVideo VSA.

    Both families use the same tile-64 sparse-attention recipe, but their gate
    module names and model contracts differ.  Parameters stay inside Comfy's
    quantized/dynamic-VRAM modules; only attention execution is substituted.
    """
    fastvideo = bool(state.get('fastvideo_vsa_active', False))
    family_label = 'FastVideo VSA' if fastvideo else 'FastH3 VSA'
    gate_attr = 'to_gate_compress' if fastvideo else 'vsa_gate'
    layout = transformer_options.get('latentlab_h3_packed_layout')
    if layout is None:
        raise _FastH3VSANotPlainT2VA('no PackedLayout was exposed to attention')
    sol_attn, err = _fast_h3_vsa_executor()
    if sol_attn is None:
        raise RuntimeError(
            f'[MiniMaxH3 LongMedia][{family_label}] no learned-VSA executor is available: {err}'
        )
    if not hasattr(attn, gate_attr):
        raise RuntimeError(
            f'[MiniMaxH3 LongMedia][{family_label}] learned gate module {gate_attr} is missing'
        )

    seq = int(x.shape[0])
    destination, block_len, prefix_blocks, padded_rows = _fast_h3_vsa_geometry(
        layout, seq, x.device, transformer_options
    )
    x_padded = x.new_zeros((padded_rows, *x.shape[1:]))
    x_padded[destination] = x

    rope_padded = None
    if rope_freqs is not None:
        rope_padded = rope_freqs.new_zeros((rope_freqs.shape[0], padded_rows, *rope_freqs.shape[2:]))
        rope_padded[:, destination] = rope_freqs

    heads = int(attn.heads)
    head_dim = int(attn.head_dim)
    inner = heads * head_dim
    q, k, v = attn.qkv_proj(x_padded).split(inner, dim=-1)
    v = v.view(padded_rows, heads, head_dim)

    if rope_padded is not None:
        import comfy.model_management
        import comfy.quant_ops
        q = q.view(1, padded_rows, heads, head_dim)
        k = k.view(1, padded_rows, heads, head_dim)
        qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
        rot = int(rope_padded.shape[-3]) * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_padded, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
            )
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_padded, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
            )
        q, k = q[0], k[0]
    else:
        q = attn.q_norm(q.view(padded_rows, heads, head_dim))
        k = attn.k_norm(k.view(padded_rows, heads, head_dim))

    gate_module = getattr(attn, gate_attr)
    coarse_gate = gate_module(x_padded).view(1, padded_rows, heads, head_dim)
    topk = float(state.get('fastvideo_vsa_topk_ratio' if fastvideo else 'fasth3_vsa_topk_ratio', 0.10))
    output = sol_attn(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        topk_ratio=topk,
        tail=False,
        block_len=block_len,
        coarse_gate=coarse_gate,
        sink_blocks=[0, prefix_blocks],
        sink_q=[0, prefix_blocks],
    )
    result = attn.out_proj(output[0, destination].flatten(-2))
    announce_key = 'fastvideo_vsa_announced' if fastvideo else 'fasth3_vsa_announced'
    if not state.get(announce_key):
        state[announce_key] = True
        _lm_print(
            f'[MiniMaxH3 LongMedia][{family_label}] ACTIVE: learned tile-64 sparse attention; '
            f'executor={getattr(sol_attn, "_longmedia_executor_label", getattr(sol_attn, "__name__", "unknown"))}; '
            f'topk={topk:.3f}; sequence={seq}; padded={padded_rows}; '
            'LongMedia dynamic-VRAM/quantized residency preserved',
            flush=True,
        )
    return result


def _fast_h3_native_sigmas(device, dtype=torch.float32):
    # Trained 4-call ladder under MiniMax H3 shift=12.  The sampler sees shifted
    # sigma values, corresponding to the published unshifted [1,.75,.5,.25,0].
    base = torch.linspace(1.0, 0.0, 5, device=device, dtype=dtype)
    shift = 12.0
    return (shift * base) / (1.0 + (shift - 1.0) * base)



def _h3_existing_attention_backend_kind(transformer_options: dict) -> str:
    """Identify the user-selected optimized-attention family without calling it.

    ModelPatcher.set_model_optimized_attention wraps the selected function, but
    preserves its container_function.  Comfy Kitchen's container function has a
    stable descriptive name, which lets LongMedia budget its transient INT8
    prequantization workspace while leaving the backend itself untouched.
    """
    try:
        override = (transformer_options or {}).get('optimized_attention_override')
    except Exception:
        override = None
    candidates = (override, getattr(override, 'container_function', None))
    labels: list[str] = []
    for fn in candidates:
        if fn is None:
            continue
        labels.extend((
            str(getattr(fn, '__module__', '') or '').lower(),
            str(getattr(fn, '__name__', '') or '').lower(),
            str(getattr(fn, '__qualname__', '') or '').lower(),
        ))
    joined = ' '.join(labels)
    if 'comfy_kitchen' in joined or 'kitchen_int8' in joined:
        return 'comfy_kitchen_int8'
    if 'sage' in joined:
        return 'sage'
    if 'flash' in joined:
        return 'flash'
    if 'xformers' in joined:
        return 'xformers'
    if 'pytorch' in joined or 'sdpa' in joined:
        return 'pytorch'
    return 'selected_existing'


def _h3_existing_workspace_required_bytes(
    attn,
    x: torch.Tensor,
    transformer_options: dict,
    state: dict,
    *,
    extra_reserve_mb: int = 0,
) -> tuple[int, int, int, str]:
    """Estimate *concurrent* activation bytes required before fused QKV.

    H3 qkv_proj emits [S, 3 * H * D] in the activation dtype.  Comfy Kitchen's
    container path then prequantizes Q/K/V while the BF16/FP16 QKV views are
    still live, so the transient peak additionally contains approximately one
    INT8 copy of Q/K/V (half the BF16/FP16 byte count).  Other existing backends
    need at least one full-width attention output beside QKV.

    The estimate is intentionally activation-only.  Dynamic model residency is
    what this guard shrinks to make room for these tensors.
    """
    seq = max(1, int(x.shape[0]))
    heads = max(1, int(attn.heads))
    head_dim = max(1, int(attn.head_dim))
    inner = heads * head_dim
    elem = max(1, int(x.element_size()))
    one = seq * inner * elem
    qkv = 3 * one
    backend = _h3_existing_attention_backend_kind(transformer_options)
    if backend == 'comfy_kitchen_int8':
        backend_extra = (qkv + 1) // 2  # INT8 Q/K/V while BF16/FP16 QKV is live.
    else:
        backend_extra = one  # attention output / backend workspace floor.

    configured_mb = max(0, int(state.get('vram_activation_reserve_mb', 0) or 0))
    reserve_mb = max(384, min(768, configured_mb // 8 if configured_mb else 384))
    reserve_mb += max(0, int(extra_reserve_mb))
    required = qkv + backend_extra + reserve_mb * 1024**2
    return int(required), int(qkv), int(backend_extra), backend


def _h3_existing_allocatable_headroom_bytes(snapshot: dict) -> int:
    """Conservative bytes that can become active allocations right now.

    ``reserved - allocated`` alone is not trustworthy with cudaMallocAsync: the
    pool can report a virtual reservation larger than physical VRAM.  Bound the
    reclaimable-pool view by ``total - allocated`` and by physical-free + cache.
    This directly models the failure in which a 5+ GiB QKV output cannot coexist
    with 13+ GiB of already-active tensors on a 16 GiB card.
    """
    total = max(0, int(snapshot.get('total', 0) or 0))
    allocated = max(0, int(snapshot.get('allocated', 0) or 0))
    driver_free = max(0, int(snapshot.get('driver_free', 0) or 0))
    cached = max(0, int(snapshot.get('cached', 0) or 0))
    active_budget = max(0, total - allocated)
    allocator_budget = max(0, driver_free + cached)
    if allocator_budget <= 0:
        return active_budget
    return min(active_budget, allocator_budget)


def _ensure_h3_existing_workspace(
    attn,
    x: torch.Tensor,
    transformer_options: dict,
    state: dict,
    *,
    extra_reserve_mb: int = 0,
) -> bool:
    """Create activation headroom for exact EXISTING attention before QKV.

    This never changes the attention algorithm/backend.  Under DynamicVRAM it
    asks the active ModelPatcher to partially offload resident weights until the
    known dense-attention transient workspace can fit.  Crucially, it calls
    ``partially_unload`` directly: the patcher remains attached to the current
    forward even if the requested amount cannot be fully released.
    """
    if not torch.cuda.is_available() or x.device.type != 'cuda':
        return True
    if int(x.shape[0]) < 32768:
        return True

    snap = _cuda_memory_snapshot(x.device)
    if not snap:
        return True
    required, qkv_bytes, backend_extra, backend = _h3_existing_workspace_required_bytes(
        attn, x, transformer_options, state, extra_reserve_mb=extra_reserve_mb,
    )
    headroom = _h3_existing_allocatable_headroom_bytes(snap)
    state['existing_workspace_guard_calls'] = int(state.get('existing_workspace_guard_calls', 0) or 0) + 1
    state['existing_workspace_backend'] = backend
    state['existing_workspace_last_required_mb'] = round(required / (1024.0**2), 1)
    state['existing_workspace_last_qkv_mb'] = round(qkv_bytes / (1024.0**2), 1)
    state['existing_workspace_last_backend_extra_mb'] = round(backend_extra / (1024.0**2), 1)
    state['existing_workspace_last_headroom_mb'] = round(headroom / (1024.0**2), 1)
    if headroom >= required:
        return True

    patcher = (transformer_options or {}).get('latentlab_h3_residency_patcher')
    if patcher is None:
        if not state.get('existing_workspace_missing_patcher_announced'):
            state['existing_workspace_missing_patcher_announced'] = True
            _lm_print(
                '[MiniMaxH3 LongMedia][EXISTING WORKSPACE] active ModelPatcher unavailable; '
                f'backend={backend}, required={required/1024**2:.0f}MB, '
                f'headroom={headroom/1024**2:.0f}MB',
                flush=True,
            )
        return False

    try:
        is_dynamic = bool(patcher.is_dynamic()) if hasattr(patcher, 'is_dynamic') else False
    except Exception:
        is_dynamic = False
    partially_unload = getattr(patcher, 'partially_unload', None)
    offload_device = getattr(patcher, 'offload_device', None)
    if not is_dynamic or not callable(partially_unload) or offload_device is None:
        if not state.get('existing_workspace_nondynamic_announced'):
            state['existing_workspace_nondynamic_announced'] = True
            _lm_print(
                '[MiniMaxH3 LongMedia][EXISTING WORKSPACE] model is not dynamically offloadable; '
                f'backend={backend}, required={required/1024**2:.0f}MB, '
                f'headroom={headroom/1024**2:.0f}MB',
                flush=True,
            )
        return False

    # Add a small hysteresis so the just-reloaded qkv weight does not immediately
    # consume the entire workspace floor.  Cap only the *request*, never the
    # required activation estimate.
    hysteresis = 256 * 1024**2
    request = max(0, int(required - headroom + hysteresis))
    if request <= 0:
        return True

    loaded_before = 0
    try:
        loaded_before = max(0, int(patcher.loaded_size()))
    except Exception:
        pass

    try:
        # Model/AIMDO transfers and previous block kernels must be complete before
        # resident weights are returned to the offload device.
        torch.cuda.synchronize(x.device)
        freed = max(0, int(partially_unload(offload_device, request)))
    except Exception as exc:
        if not state.get('existing_workspace_unload_error_announced'):
            state['existing_workspace_unload_error_announced'] = True
            _lm_print(
                '[MiniMaxH3 LongMedia][EXISTING WORKSPACE] partial offload failed: '
                f'{type(exc).__name__}: {exc}',
                flush=True,
            )
        return False

    state['existing_workspace_release_count'] = int(state.get('existing_workspace_release_count', 0) or 0) + (1 if freed > 0 else 0)
    state['existing_workspace_released_mb'] = round(
        float(state.get('existing_workspace_released_mb', 0.0) or 0.0) + freed / (1024.0**2), 1
    )

    after = _cuda_memory_snapshot(x.device)
    if not after:
        return freed >= request
    after_headroom = _h3_existing_allocatable_headroom_bytes(after)
    state['existing_workspace_last_headroom_mb'] = round(after_headroom / (1024.0**2), 1)
    loaded_after = 0
    try:
        loaded_after = max(0, int(patcher.loaded_size()))
    except Exception:
        pass

    _lm_print(
        '[MiniMaxH3 LongMedia][EXISTING WORKSPACE] '
        f'block={int(state.get("active_block_index", -1))}; backend={backend}; '
        f'QKV={qkv_bytes/1024**2:.0f}MB + backend={backend_extra/1024**2:.0f}MB; '
        f'headroom={headroom/1024**2:.0f}->{after_headroom/1024**2:.0f}MB; '
        f'dynamic_residency_released={freed/1024**2:.0f}MB; '
        f'loaded={loaded_before/1024**2:.0f}->{loaded_after/1024**2:.0f}MB; '
        f'target={required/1024**2:.0f}MB',
        flush=True,
    )
    return after_headroom >= required


def _h3_existing_ck_stream_chunk_tokens(state: dict) -> int:
    """Return an INT8-attention-safe query chunk aligned to CK's Q tile."""
    requested = int(state.get('sol_qkv_chunk_tokens', 8192) or 8192)
    # Comfy Kitchen quantizes Q in independent 128-row blocks.  Keeping every
    # replay boundary on that tile makes per-block Q scales identical to a
    # monolithic prequantization.
    return max(128, (max(128, requested) // 128) * 128)


def _h3_existing_ck_stream_needed(
    attn,
    x: torch.Tensor,
    transformer_options: dict,
    state: dict,
) -> bool:
    """Select exact CK query streaming only when dense QKV is structurally huge.

    Low-resolution EXISTING keeps the normal selected-backend path.  The
    streamed path activates when a single fused QKV/output-prequant peak would
    consume most of a <=18.5 GiB card, which is the Latent-HiRes failure mode.
    """
    if _h3_existing_attention_backend_kind(transformer_options) != 'comfy_kitchen_int8':
        return False
    if not torch.cuda.is_available() or x.device.type != 'cuda':
        return False
    seq = int(x.shape[0])
    chunk = _h3_existing_ck_stream_chunk_tokens(state)
    if seq <= chunk:
        return False
    snap = _cuda_memory_snapshot(x.device)
    if not snap:
        return False
    total = max(1, int(snap.get('total', 0) or 0))
    if total > int(18.5 * 1024**3):
        return False
    required, qkv_bytes, _backend_extra, _backend = _h3_existing_workspace_required_bytes(
        attn, x, transformer_options, state,
    )
    qkv_fraction = float(qkv_bytes) / float(total)
    dense_fraction = float(required) / float(total)
    activate = qkv_fraction >= 0.40 or dense_fraction >= 0.68
    state['existing_ck_stream_qkv_fraction'] = round(qkv_fraction, 4)
    state['existing_ck_stream_dense_fraction'] = round(dense_fraction, 4)
    return bool(activate)


def _h3_existing_make_additional_headroom(
    x: torch.Tensor,
    transformer_options: dict,
    state: dict,
    required_bytes: int,
    *,
    phase: str,
) -> bool:
    """Yield DynamicVRAM residency for a known *additional* stream allocation.

    Unlike the dense workspace governor, callers already own their persistent
    tensors (for example full BF16 K/V).  ``required_bytes`` therefore means
    bytes that still need to become active from the current snapshot.
    """
    required = max(0, int(required_bytes))
    if required <= 0 or not torch.cuda.is_available() or x.device.type != 'cuda':
        return True
    snap = _cuda_memory_snapshot(x.device)
    if not snap:
        return True
    before = _h3_existing_allocatable_headroom_bytes(snap)
    state['existing_ck_stream_last_phase'] = str(phase)
    state['existing_ck_stream_last_required_mb'] = round(required / 1024**2, 1)
    state['existing_ck_stream_last_headroom_mb'] = round(before / 1024**2, 1)
    if before >= required:
        return True

    patcher = (transformer_options or {}).get('latentlab_h3_residency_patcher')
    try:
        dynamic = bool(patcher is not None and patcher.is_dynamic())
    except Exception:
        dynamic = False
    partially_unload = getattr(patcher, 'partially_unload', None) if patcher is not None else None
    offload_device = getattr(patcher, 'offload_device', None) if patcher is not None else None
    if not dynamic or not callable(partially_unload) or offload_device is None:
        _lm_print(
            '[MiniMaxH3 LongMedia][EXISTING CK STREAM] '
            f'phase={phase}; DynamicVRAM residency unavailable; '
            f'need={required/1024**2:.0f}MB headroom={before/1024**2:.0f}MB',
            flush=True,
        )
        return False

    hysteresis = 256 * 1024**2
    request = max(0, required - before + hysteresis)
    if request <= 0:
        return True
    try:
        loaded_before = max(0, int(patcher.loaded_size()))
    except Exception:
        loaded_before = 0
    try:
        # Do this only at phase boundaries, never per query chunk.
        torch.cuda.synchronize(x.device)
        freed = max(0, int(partially_unload(offload_device, int(request))))
    except Exception as exc:
        _lm_print(
            '[MiniMaxH3 LongMedia][EXISTING CK STREAM] '
            f'phase={phase}; partial offload failed: {type(exc).__name__}: {exc}',
            flush=True,
        )
        return False

    after_snap = _cuda_memory_snapshot(x.device)
    after = (
        _h3_existing_allocatable_headroom_bytes(after_snap)
        if after_snap else before + freed
    )
    try:
        loaded_after = max(0, int(patcher.loaded_size()))
    except Exception:
        loaded_after = 0
    state['existing_ck_stream_release_count'] = int(
        state.get('existing_ck_stream_release_count', 0) or 0
    ) + (1 if freed > 0 else 0)
    state['existing_ck_stream_released_mb'] = round(
        float(state.get('existing_ck_stream_released_mb', 0.0) or 0.0)
        + freed / 1024**2,
        1,
    )
    state['existing_ck_stream_last_headroom_mb'] = round(after / 1024**2, 1)
    _lm_print(
        '[MiniMaxH3 LongMedia][EXISTING CK STREAM] '
        f'phase={phase}; headroom={before/1024**2:.0f}->{after/1024**2:.0f}MB; '
        f'released={freed/1024**2:.0f}MB; '
        f'loaded={loaded_before/1024**2:.0f}->{loaded_after/1024**2:.0f}MB; '
        f'target={required/1024**2:.0f}MB',
        flush=True,
    )
    return after >= required


def _run_h3_existing_ck_streamed_attention(
    attn,
    x: torch.Tensor,
    rope_freqs,
    transformer_options: dict,
    state: dict,
):
    """Exact low-VRAM Comfy-Kitchen EXISTING attention for giant H3 sequences.

    Contract:
      1. Build only full BF16/FP16 K/V in token chunks (never full fused QKV).
      2. Let Comfy Kitchen prequantize full K/V once, preserving its global K
         anchor and V scale exactly.
      3. Release floating K/V.
      4. Re-project Q in 128-aligned chunks, prequantize Q with the same public
         CK split API, and attend each Q chunk against the one shared full K/V.

    CK documents unequal non-causal Q/K lengths, and its Q quantizer is local to
    independent 128-row blocks.  Therefore query chunking changes lifetime and
    launch geometry only; it does not change attention math, K anchoring, V
    scaling, or the user-selected Comfy Kitchen backend.
    """
    import comfy.model_management
    import comfy.quant_ops
    from comfy_kitchen.sage_attention import (
        int8_attention_from_prequantized,
        prequantize_int8_attention,
    )

    seq = int(x.shape[0])
    heads = int(attn.heads)
    head_dim = int(attn.head_dim)
    inner = heads * head_dim
    chunk = _h3_existing_ck_stream_chunk_tokens(state)
    chunks = (seq + chunk - 1) // chunk
    elem = max(1, int(x.element_size()))
    one_full = seq * inner * elem
    qkv_full = 3 * one_full
    kv_fp_bytes = 2 * one_full
    kv_packed_bytes = one_full  # K+V INT8 ~= one BF16/FP16 full tensor.
    chunk_one = min(seq, chunk) * inner * elem
    reserve = 768 * 1024**2
    retry_extra = max(0, int(state.pop('existing_workspace_retry_extra_mb', 0) or 0)) * 1024**2

    state['existing_ck_stream_calls'] = int(state.get('existing_ck_stream_calls', 0) or 0) + 1
    state['existing_ck_stream_chunks'] = int(chunks)
    state['existing_ck_stream_chunk_tokens'] = int(chunk)
    state['existing_ck_stream_qkv_mb_avoided'] = round(qkv_full / 1024**2, 1)
    if not state.get('existing_ck_stream_announced'):
        _lm_print(
            '[MiniMaxH3 LongMedia][EXISTING CK STREAM] ACTIVE: '
            f'{seq} tokens -> {chunks} Q chunks <= {chunk}; '
            f'dense fused-QKV {qkv_full/1024**2:.0f}MB avoided; '
            f'full K/V {kv_fp_bytes/1024**2:.0f}MB -> CK INT8 {kv_packed_bytes/1024**2:.0f}MB; '
            'global K anchor/V scale preserved; backend=Comfy Kitchen INT8',
            flush=True,
        )
        state['existing_ck_stream_announced'] = True

    # Before allocating persistent floating K/V, yield enough model residency
    # for both buffers plus the largest chunk projection and safety margin.
    _h3_existing_make_additional_headroom(
        x, transformer_options, state,
        kv_fp_bytes + 3 * chunk_one + reserve + retry_extra,
        phase='kv_build',
    )

    k_seq = x.new_empty((1, seq, heads, head_dim))
    v_seq = x.new_empty((1, seq, heads, head_dim))

    qw = kw = None
    rot = None
    if rope_freqs is not None:
        qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
        rot = int(rope_freqs.shape[-3]) * 2

    use_cached_rows = (
        _v12_is_int8_family(state)
        and not comfy.model_management.in_training
        and str(state.get('memory_mode', 'normal')) != 'ultra_low_vram'
    )

    def _prepare_qkv_handle():
        if not use_cached_rows:
            return None
        try:
            return _int8_prepare_block_linear(attn.qkv_proj, x[:min(4, seq)])
        except Exception as exc:
            if not state.get('existing_ck_stream_row_slice_prepare_failed'):
                _lm_print(
                    '[MiniMaxH3 LongMedia][EXISTING CK STREAM] '
                    f'row-sliced QKV preparation unavailable ({type(exc).__name__}: {exc}); '
                    'using stock chunked qkv_proj',
                    flush=True,
                )
                state['existing_ck_stream_row_slice_prepare_failed'] = True
            return None

    def _normalize_k(k_part: torch.Tensor, start: int, end: int) -> torch.Tensor:
        n = end - start
        k4 = k_part.view(1, n, heads, head_dim)
        if rope_freqs is None:
            return attn.k_norm(k4)
        # The fused H3 RMS+RoPE op exposes rot_dim only in paired form.  The
        # second output is independent from the first, so a chunk-local dummy Q
        # preserves stock K math without retaining real Q.
        q_dummy = k4.clone()
        rf = rope_freqs[:, start:end]
        if comfy.model_management.in_training:
            q_dummy, k4 = comfy.quant_ops.ck.rms_rope_split_half(
                q_dummy, k4, rf, qw, kw,
                epsilon=attn.q_norm.eps, rot_dim=rot,
            )
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q_dummy, k4, rf, qw, kw,
                epsilon=attn.q_norm.eps, rot_dim=rot,
            )
        del q_dummy
        return k4

    def _normalize_q(q_part: torch.Tensor, start: int, end: int) -> torch.Tensor:
        n = end - start
        q4 = q_part.view(1, n, heads, head_dim)
        if rope_freqs is None:
            return attn.q_norm(q4)
        # Symmetric with the K helper: the first output is independent from the
        # dummy K operand and therefore matches stock paired RMS+RoPE.
        k_dummy = q4.clone()
        rf = rope_freqs[:, start:end]
        if comfy.model_management.in_training:
            q4, k_dummy = comfy.quant_ops.ck.rms_rope_split_half(
                q4, k_dummy, rf, qw, kw,
                epsilon=attn.q_norm.eps, rot_dim=rot,
            )
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q4, k_dummy, rf, qw, kw,
                epsilon=attn.q_norm.eps, rot_dim=rot,
            )
        del k_dummy
        return q4

    # Pass 1: project only K/V rows when the native quantized layout supports
    # row slicing.  Fallback still chunks the stock fused projection, so the
    # giant full-sequence QKV allocation can never reappear.
    qkv_handle = _prepare_qkv_handle()
    row_sliced_kv = bool(qkv_handle is not None)
    try:
        for start in range(0, seq, chunk):
            end = min(seq, start + chunk)
            kv_rows = None
            qkv_part = None
            if qkv_handle is not None:
                kv_rows = _v32_quant_linear_rows(qkv_handle, x[start:end], inner, inner * 3)
            if kv_rows is not None:
                k_part, v_part = kv_rows.split(inner, dim=-1)
            else:
                row_sliced_kv = False
                qkv_part = (
                    _int8_cached_linear(qkv_handle, x[start:end])
                    if qkv_handle is not None else attn.qkv_proj(x[start:end])
                )
                _q_dead, k_part, v_part = qkv_part.split(inner, dim=-1)
            k4 = _normalize_k(k_part, start, end)
            n = end - start
            v4 = v_part.view(1, n, heads, head_dim)
            k_seq[:, start:end].copy_(k4)
            v_seq[:, start:end].copy_(v4)
            del k4, v4, k_part, v_part, kv_rows
            if qkv_part is not None:
                del qkv_part, _q_dead
    finally:
        _int8_release_block_linear(qkv_handle)
        qkv_handle = None

    state['existing_ck_stream_row_sliced_kv'] = bool(row_sliced_kv)

    # Pack the *entire* K/V exactly once.  This is the important numerical
    # contract: CK's K anchor detector and V scale both see the whole sequence.
    # A one-row Q view is sufficient because its packed result is discarded.
    _h3_existing_make_additional_headroom(
        x, transformer_options, state,
        kv_packed_bytes + reserve + retry_extra,
        phase='kv_prequant',
    )
    k_hnd = k_seq.transpose(1, 2)
    v_hnd = v_seq.transpose(1, 2)
    dummy_q = k_hnd[:, :, :1, :]
    shared = prequantize_int8_attention(
        dummy_q, k_hnd, v_hnd,
        scale=float(head_dim ** -0.5),
        attn_mask=None,
    )
    del dummy_q, k_hnd, v_hnd, k_seq, v_seq

    # After BF K/V dies, reserve room for the full block result plus one query
    # chunk.  The allocator may reuse the just-freed K/V segments directly.
    result_bytes = int(x.numel()) * elem
    _h3_existing_make_additional_headroom(
        x, transformer_options, state,
        result_bytes + 4 * chunk_one + reserve,
        phase='query_replay',
    )
    result = torch.empty_like(x)

    # CK chooses the Q Hadamard rotation from K length (D128: H4 for <=256,
    # H128 for long K) and chooses CTA128 for long D128/D256 K.  Query replay
    # must therefore use a small *long-shape* dummy K/V, not a one-row dummy,
    # otherwise Q quantization would silently change.  1025 is the minimum
    # length that reproduces the long-sequence rotation + CTA selection while
    # keeping the dummy footprint tiny (~14 MiB at H3 D128/H56 BF16).
    ck_dummy_kv = x.new_zeros((1, heads, 1025, head_dim))

    qkv_handle = _prepare_qkv_handle()
    out_handle = None
    row_sliced_q = bool(qkv_handle is not None)
    try:
        for start in range(0, seq, chunk):
            end = min(seq, start + chunk)
            q_rows = None
            qkv_part = None
            if qkv_handle is not None:
                q_rows = _v32_quant_linear_rows(qkv_handle, x[start:end], 0, inner)
            if q_rows is not None:
                q_part = q_rows
            else:
                row_sliced_q = False
                qkv_part = (
                    _int8_cached_linear(qkv_handle, x[start:end])
                    if qkv_handle is not None else attn.qkv_proj(x[start:end])
                )
                q_part, _k_dead, _v_dead = qkv_part.split(inner, dim=-1)

            q4 = _normalize_q(q_part, start, end)
            q_hnd = q4.transpose(1, 2)
            # Public CK split-prequant API guarantees the returned object keeps
            # no FP Q/K/V references.  The reusable 1025-row dummy K/V keeps
            # CK's Q rotation/CTA dispatch identical to the real long K/V; its
            # packed K/V outputs are discarded immediately.
            q_pack = prequantize_int8_attention(
                q_hnd, ck_dummy_kv, ck_dummy_kv,
                scale=shared.attention_scale,
                attn_mask=None,
            )
            packed = _dc_replace(shared, q=q_pack.q, q_scale=q_pack.q_scale)
            del q_hnd, q4, q_part, q_rows
            if qkv_part is not None:
                del qkv_part, _k_dead, _v_dead

            out_hnd = int8_attention_from_prequantized(packed)
            del packed, q_pack
            n = end - start
            flat = out_hnd.transpose(1, 2).contiguous().view(n, inner)
            del out_hnd

            if out_handle is None and use_cached_rows:
                try:
                    out_handle = _int8_prepare_block_linear(
                        attn.out_proj, flat[:min(4, n)]
                    )
                except Exception:
                    out_handle = False
            part = (
                _int8_cached_linear(out_handle, flat)
                if out_handle not in (None, False) else attn.out_proj(flat)
            )
            result[start:end].copy_(part)
            del part, flat
    finally:
        _int8_release_block_linear(qkv_handle)
        if out_handle not in (None, False):
            _int8_release_block_linear(out_handle)

    del ck_dummy_kv
    state['existing_ck_stream_row_sliced_q'] = bool(row_sliced_q)
    if not state.get('existing_ck_stream_projection_announced'):
        _lm_print(
            '[MiniMaxH3 LongMedia][EXISTING CK STREAM] '
            f'projection rows: KV={"native-sliced" if row_sliced_kv else "stock-chunked"}, '
            f'Q={"native-sliced" if row_sliced_q else "stock-chunked"}; '
            'query replay completed against one globally prequantized K/V store',
            flush=True,
        )
        state['existing_ck_stream_projection_announced'] = True
    return result


def _run_h3_existing_attention_lowmem(attn, x, rope_freqs, transformer_options, state):
    """Memory-restored H3 EXISTING attention path.

    Normal long sequences keep the selected optimized_attention backend.  On a
    16-GB-class GPU, when Comfy Kitchen's *single fused QKV allocation itself*
    becomes structurally too large (notably Latent Hi-Res), switch only its
    lifetime schedule to exact query streaming.  The backend/math remains CK
    INT8; SOL is never substituted.
    """
    import comfy.model_management
    import comfy.quant_ops
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

    seq = int(x.shape[0])
    heads = int(attn.heads)
    head_dim = int(attn.head_dim)
    inner = heads * head_dim

    if _h3_existing_ck_stream_needed(attn, x, transformer_options, state):
        return _run_h3_existing_ck_streamed_attention(
            attn, x, rope_freqs, transformer_options, state
        )

    # Dense EXISTING attention has a large fused-QKV transient.  Preserve the
    # selected backend/math, but make DynamicVRAM residency yield before the
    # allocation instead of discovering the conflict through CUDA OOM.
    _retry_extra_mb = max(0, int(state.pop('existing_workspace_retry_extra_mb', 0) or 0))
    _ensure_h3_existing_workspace(
        attn, x, transformer_options, state, extra_reserve_mb=_retry_extra_mb
    )

    # Keep stock projection semantics. q/k/v remain views of the one qkv buffer.
    q, k, v = attn.qkv_proj(x).split(inner, dim=-1)
    v = v.view(seq, heads, head_dim)

    if rope_freqs is not None:
        q = q.view(1, seq, heads, head_dim)
        k = k.view(1, seq, heads, head_dim)
        qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
        rot = int(rope_freqs.shape[-3]) * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw,
                epsilon=attn.q_norm.eps, rot_dim=rot,
            )
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw,
                epsilon=attn.q_norm.eps, rot_dim=rot,
            )
        q = q[0]
        k = k[0]
    else:
        q = attn.q_norm(q.view(seq, heads, head_dim))
        k = attn.k_norm(k.view(seq, heads, head_dim))

    # IMPORTANT: deliberately no `v = v.clone()` here.
    q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
    k = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
    v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))

    out = optimized_attention(
        q, k, v, heads,
        mask=None,
        skip_reshape=True,
        transformer_options=transformer_options,
    )

    # Chunk only the point-wise output projection for long sequences.
    flat = out.squeeze(0)
    chunk_tokens = int(state.get('existing_out_proj_chunk_tokens', 16384) or 16384)
    if seq <= chunk_tokens:
        result = attn.out_proj(flat)
    else:
        result = torch.empty_like(x)
        for start in range(0, seq, chunk_tokens):
            end = min(seq, start + chunk_tokens)
            part = attn.out_proj(flat[start:end])
            result[start:end].copy_(part)
            del part
        if not state.get('existing_lowmem_outproj_announced'):
            _lm_print(
                '[MiniMaxH3 LongMedia][EXISTING LOWMEM] '
                f'out_proj {seq} tokens -> {(seq + chunk_tokens - 1)//chunk_tokens} '
                f'chunks <= {chunk_tokens}; attention backend unchanged',
                flush=True,
            )
            state['existing_lowmem_outproj_announced'] = True

    return result


def _should_use_h3_existing_lowmem(state, token_count):
    """Use the restored pre-clone path only where the clone is materially costly."""
    try:
        tokens = int(token_count or 0)
        if tokens < 32768:
            return False
        free_b, total_b = torch.cuda.mem_get_info(torch.cuda.current_device())
        # Primarily target 16 GB-class cards; keep stock behavior on roomy GPUs.
        return int(total_b) <= int(18.5 * 1024**3)
    except Exception:
        return int(token_count or 0) >= 32768


def _run_h3_sol_attention(attn, x, rope_freqs, transformer_options, state, measure=None):
    """Embedded H3 Sol path with adaptive low-VRAM retries.

    OOM inside Sol should never fall through to the generic attention fallback,
    because that path can trigger catastrophic NVFP4 dequantization on large H3
    sequences. Instead we trim cache and retry the same embedded Sol path with a
    smaller streamed QKV chunk size.
    """
    token_count = int(x.shape[0])
    min_tokens = int(state.get('sol_min_tokens', 4096))
    if token_count < min_tokens:
        return attn(x, rope_freqs=rope_freqs, transformer_options=transformer_options)
    tau = _sol_schedule_tau(transformer_options, state)
    if tau is None:
        return attn(x, rope_freqs=rope_freqs, transformer_options=transformer_options)

    try:
        result, sink_blocks = _execute_h3_sol_attention(
            attn, x, rope_freqs, transformer_options, state, tau, measure=measure
        )
    except Exception as exc:
        if isinstance(exc, _INT8SolStorageOOM):
            _lm_print(
                '[MiniMaxH3 LongMedia][INT8 SOL STORAGE] terminal failure; '
                'external attention fallback is disabled for this large INT8 sequence',
                flush=True,
            )
            raise

        if _sol_exception_is_oom(exc):
            original_chunk = int(state.get('sol_qkv_chunk_tokens', 0) or 0)
            original_out_proj = int(state.get('sol_out_proj_chunk_tokens', 0) or 0)
            seen = state.setdefault('sol_oom_reasons', [])
            reason = f'{type(exc).__name__}: {exc}'
            if reason not in seen:
                seen.append(reason)
                _lm_print('[MiniMaxH3 LongMedia] Embedded Sol-Attn OOM: ' + reason, flush=True)
            retry_chunks = _sol_retry_chunk_schedule(original_chunk)
            last_exc = exc
            for retry_idx, retry_chunk in enumerate(retry_chunks, start=1):
                try:
                    state['sol_qkv_chunk_tokens'] = int(retry_chunk)
                    if original_out_proj > 0:
                        state['sol_out_proj_chunk_tokens'] = min(original_out_proj, max(retry_chunk * 3, 1024))
                    _lm_print(
                        '[MiniMaxH3 LongMedia] Embedded Sol-Attn retry '
                        f'#{retry_idx}: qkv chunk {original_chunk} -> {retry_chunk}, '
                        f'out_proj <= {int(state.get("sol_out_proj_chunk_tokens", 0) or 0)}',
                        flush=True,
                    )
                    gc.collect()
                    try:
                        import comfy.model_management as _mm
                        _mm.soft_empty_cache()
                    except Exception:
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    result, sink_blocks = _execute_h3_sol_attention(
                        attn, x, rope_freqs, transformer_options, state, tau, measure=measure
                    )
                    if retry_chunk != original_chunk:
                        _lm_print(
                            '[MiniMaxH3 LongMedia] Embedded Sol-Attn retry succeeded: '
                            f'using persistent qkv chunk {retry_chunk}',
                            flush=True,
                        )
                    break
                except Exception as retry_exc:
                    last_exc = retry_exc
                    if not _sol_exception_is_oom(retry_exc):
                        raise
                    _lm_print(
                        '[MiniMaxH3 LongMedia] Embedded Sol-Attn retry failed: '
                        f'qkv chunk {retry_chunk}: {type(retry_exc).__name__}: {retry_exc}',
                        flush=True,
                    )
                    gc.collect()
                    try:
                        import comfy.model_management as _mm
                        _mm.soft_empty_cache()
                    except Exception:
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
            else:
                state['sol_qkv_chunk_tokens'] = int(original_chunk)
                state['sol_out_proj_chunk_tokens'] = int(original_out_proj)
                raise last_exc
        else:
            _backend = str(state.get('model_runtime_backend', 'unknown')).lower()
            if _backend in ('int8', 'int8-convrot-w4a4') and token_count >= min_tokens:
                _lm_print(
                    '[MiniMaxH3 LongMedia][INT8 SOL] non-OOM Sol failure; '
                    'external Sage fallback disabled to avoid full-sequence INT8 QKV allocation: '
                    f'{type(exc).__name__}: {exc}',
                    flush=True,
                )
                raise

            reason = f'{type(exc).__name__}: {exc}'
            seen = state.setdefault('sol_fallbacks', [])
            if reason not in seen:
                seen.append(reason)
                _lm_print('[MiniMaxH3 LongMedia] Embedded Sol-Attn fallback: ' + reason, flush=True)

            try:
                del exc
            except Exception:
                pass
            gc.collect()
            try:
                import comfy.model_management as _mm
                _mm.soft_empty_cache()
            except Exception:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            return attn(x, rope_freqs=rope_freqs, transformer_options=transformer_options)

    state['sol_calls'] = int(state.get('sol_calls', 0)) + 1
    if not state.get('sol_announced'):
        _lm_print(
            '[MiniMaxH3 LongMedia] Embedded Sol-Attn active: '
            f'{token_count} tokens, tau={float(tau):.3f}, sink={sink_blocks}',
            flush=True,
        )
        state['sol_announced'] = True
    return result


class _H3MLPChunkPatch:
    """Exact low-VRAM H3 block replacement that chunks only the token-wise MLP.

    Attention is left untouched. The FFN/MLP is point-wise over the packed token
    axis, so splitting dim 0 and writing each result into a preallocated output
    tensor preserves the full-block result while avoiding a full-sequence
    expanded-hidden activation. Block 0 of the first forward also emits deep
    stage memory telemetry for A/B testing.
    """

    def __init__(self, index, state, chunk_tokens=8192):
        self.index = int(index)
        self.state = state
        self.chunk_tokens = max(256, int(chunk_tokens))

    @staticmethod
    def _extract_block(original_block):
        return _H3BlockMemoryTracePatch._extract_block(original_block)

    @staticmethod
    def _mod_scale_shift(h, shift, scale, segments):
        return _H3BlockMemoryTracePatch._mod_scale_shift(h, shift, scale, segments)

    @staticmethod
    def _mod_gate(x, gate, other, segments):
        return _H3BlockMemoryTracePatch._mod_gate(x, gate, other, segments)

    def _inter_block_pressure_guard(self):
        """TEST FIXED: effective-headroom inter-block guard.

        ACTIVE implementation for _H3MLPChunkPatch.
        Uses driver-free VRAM + reclaimable PyTorch cache instead of free VRAM
        alone, avoiding unnecessary DynamicVRAM/AIMDO cache destruction.
        """
        state = self.state

        # Guaranteed one-shot AUTO calibration after completed block 0.
        if self.index == 0 and not state.get('auto_vram_controller_done'):
            try:
                token_count = int(
                    state.get('current_token_count', 0)
                    or state.get('last_token_count', 0)
                    or 0
                )
                self._auto_vram_controller_after_probe(token_count)
            except Exception as exc:
                state['auto_vram_controller_done'] = True
                state['auto_vram_controller_mode'] = 'SAFE'
                _lm_print(
                    f"[MiniMaxH3 LongMedia][AUTO VRAM] calibration failed; "
                    f"SAFE baseline retained: {exc!r}",
                    flush=True,
                )

        if not torch.cuda.is_available():
            return

        guard_mb = float(int(state.get('inter_block_vram_guard_mb', 0) or 0))
        emergency_mb = float(int(state.get('inter_block_guard_emergency_mb', 0) or 0))
        cooldown_blocks = max(
            0, int(state.get('inter_block_guard_cooldown_blocks', 0) or 0)
        )
        emergency_cooldown_blocks = max(
            0, int(state.get('inter_block_guard_emergency_cooldown_blocks', 0) or 0)
        )

        if guard_mb <= 0 and emergency_mb <= 0:
            return

        guard_call = int(state.get('inter_block_guard_calls', 0)) + 1
        state['inter_block_guard_calls'] = guard_call

        snap = _cuda_memory_snapshot()
        if not snap:
            return

        mb = 1024.0 ** 2
        free_mb = float(snap['driver_free']) / mb
        cached_mb = float(snap['cached']) / mb
        effective_mb = free_mb + cached_mb

        normal_hyst = float(
            state.get('inter_block_guard_hysteresis_mb', 1024.0) or 1024.0
        )
        emergency_hyst = float(
            state.get('inter_block_emergency_hysteresis_mb', 512.0) or 512.0
        )
        min_reclaim_mb = float(
            state.get('inter_block_min_reclaim_mb', 256.0) or 256.0
        )

        normal_trigger = guard_mb > 0 and effective_mb < guard_mb
        emergency_trigger = emergency_mb > 0 and effective_mb < emergency_mb

        # Healthy effective headroom: preserve allocator cache.
        if not normal_trigger and not emergency_trigger:
            state['inter_block_effective_skip_count'] = int(
                state.get('inter_block_effective_skip_count', 0) or 0
            ) + 1
            skips = int(state['inter_block_effective_skip_count'])
            if skips <= 3 or skips % 25 == 0:
                _lm_print(
                    '[MiniMaxH3 LongMedia][VRAM GUARD] skip: '
                    f'block {self.index}, free={free_mb:.0f} + cached={cached_mb:.0f} '
                    f'= effective={effective_mb:.0f} MB, '
                    f'guard={guard_mb:.0f}, emergency={emergency_mb:.0f}',
                    flush=True,
                )
            return

        # Threshold hysteresis.
        if normal_trigger and effective_mb >= max(0.0, guard_mb - normal_hyst):
            state['inter_block_hysteresis_skip_count'] = int(
                state.get('inter_block_hysteresis_skip_count', 0) or 0
            ) + 1
            return

        if emergency_trigger and effective_mb >= max(
            0.0, emergency_mb - emergency_hyst
        ):
            state['inter_block_emergency_hyst_skip_count'] = int(
                state.get('inter_block_emergency_hyst_skip_count', 0) or 0
            ) + 1
            return

        # Re-use the existing monotonically increasing call counter for cooldown.
        last_trim_call = int(
            state.get('inter_block_last_trim_call', -1000000000)
        )
        blocks_since_trim = guard_call - last_trim_call - 1

        if emergency_trigger:
            last_emergency = int(
                state.get('inter_block_last_emergency_trim_call', -1000000000)
            )
            since_emergency = guard_call - last_emergency - 1
            hard_emergency = effective_mb < max(0.0, emergency_mb - 1024.0)
            if (
                emergency_cooldown_blocks > 0
                and since_emergency < emergency_cooldown_blocks
                and not hard_emergency
            ):
                state['inter_block_emergency_cooldown_skip_count'] = int(
                    state.get('inter_block_emergency_cooldown_skip_count', 0) or 0
                ) + 1
                return
        else:
            hard_normal = effective_mb < max(0.0, guard_mb - 1536.0)
            if (
                cooldown_blocks > 0
                and blocks_since_trim < cooldown_blocks
                and not hard_normal
            ):
                state['inter_block_cooldown_skip_count'] = int(
                    state.get('inter_block_cooldown_skip_count', 0) or 0
                ) + 1
                return

        # No useful cache to reclaim: don't force a pointless sync/trim.
        if cached_mb < min_reclaim_mb:
            state['inter_block_low_cache_skip_count'] = int(
                state.get('inter_block_low_cache_skip_count', 0) or 0
            ) + 1
            return

        before = snap
        try:
            _soft_empty_cuda_cache()
        except Exception as exc:
            _lm_print(
                f"[MiniMaxH3 LongMedia][VRAM GUARD] cleanup failed at "
                f"block {self.index}: {exc!r}",
                flush=True,
            )
            return

        after = _cuda_memory_snapshot()
        state['inter_block_last_trim_call'] = guard_call
        state['inter_block_trim_count'] = int(
            state.get('inter_block_trim_count', 0) or 0
        ) + 1

        if emergency_trigger:
            state['inter_block_last_emergency_trim_call'] = guard_call
            state['inter_block_emergency_trim_count'] = int(
                state.get('inter_block_emergency_trim_count', 0) or 0
            ) + 1
            label = 'EMERGENCY TRIM'
        else:
            state['inter_block_normal_trim_count'] = int(
                state.get('inter_block_normal_trim_count', 0) or 0
            ) + 1
            label = 'NORMAL TRIM'

        if after:
            free_after = float(after['driver_free']) / mb
            cached_after = float(after['cached']) / mb
            _lm_print(
                f'[MiniMaxH3 LongMedia][VRAM GUARD] {label}: '
                f'block {self.index}, effective={effective_mb:.0f} MB, '
                f'free {free_mb:.0f}->{free_after:.0f} MB, '
                f'cached {cached_mb:.0f}->{cached_after:.0f} MB',
                flush=True,
            )

    def _adaptive_memory_governor(self, phase='block_start'):
        """v0.3.60 adaptive residency governor shared by every memory mode.

        Memory modes no longer mean fixed chunk presets.  They select a safety
        envelope; runtime policy is then derived from real driver-free VRAM,
        observed packed-token count, model/VRAM oversubscription and host-RAM
        pressure.  The governor may change activation chunk size and barrier
        aggressiveness, but never swaps in custom transformer math.
        """
        state = self.state
        if not bool(state.get('adaptive_memory_governor_enabled', True)):
            return
        if not torch.cuda.is_available():
            return
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            free_mb = float(free_b) / (1024.0 ** 2)
            total_mb = float(total_b) / (1024.0 ** 2)
        except Exception as exc:
            if not state.get('adaptive_memory_probe_error_announced'):
                _lm_print('[MiniMaxH3 LongMedia][GOVERNOR V3] memory probe failed: '
                          f'{type(exc).__name__}: {exc}', flush=True)
                state['adaptive_memory_probe_error_announced'] = True
            return

        mode = str(state.get('memory_policy_mode', state.get('memory_mode', 'normal')))
        _backend = str(state.get('model_runtime_backend', 'unknown')).lower()
        tokens = int(state.get('current_token_count', state.get('last_token_count', 0)) or 0)
        model_b = int(state.get('model_size_bytes', 0) or 0)
        gpu_b = int(state.get('gpu_size_bytes', 0) or 0)
        ratio = (float(model_b) / float(gpu_b)) if model_b and gpu_b else 0.0

        # v0.3.110 Governor V4: sequence geometry is part of the safety envelope.
        # V3 could call a 137k-token forward FAST_PLUS because mem_get_info() was
        # sampled before demand-loaded attention weights/QKV workspaces existed.
        # Large packed sequences therefore promote the *safety* mode even when
        # the model-size probe under-reports an AIMDO/dynamic checkpoint.
        geometry_mode = mode
        if total_mb <= 18.5 * 1024.0:
            if tokens >= 150000:
                geometry_mode = 'ultra_low_vram'
            elif tokens >= 90000 and mode == 'normal':
                geometry_mode = 'low_vram'
        elif total_mb <= 26.0 * 1024.0 and tokens >= 180000 and mode == 'normal':
            geometry_mode = 'low_vram'
        if geometry_mode != mode:
            if not state.get('v110_geometry_mode_announced'):
                _lm_print(
                    '[MiniMaxH3 LongMedia][GOVERNOR V4] '
                    f'geometry safety envelope {mode}->{geometry_mode}; '
                    f'tokens={tokens} VRAM={total_mb/1024.0:.1f}GB',
                    flush=True,
                )
                state['v110_geometry_mode_announced'] = True
            mode = geometry_mode

        # Token-dependent activation margin. V3 capped this too aggressively and
        # only approximated generic activations. V4 keeps a larger dynamic floor;
        # attention itself has a separate geometry preflight before QKV allocation.
        token_margin_mb = min(4096.0, max(512.0, float(tokens) * 0.0120)) if tokens else 1024.0
        base_floor = {
            'normal': 640.0,
            'low_vram': 1152.0,
            'ultra_low_vram': 1792.0,
        }.get(mode, 1536.0)
        if ratio >= 1.5:
            base_floor += 384.0
        if ratio >= 2.0:
            base_floor += 256.0
        hard_floor = min(total_mb * 0.34, base_floor + token_margin_mb)
        soft_floor = min(total_mb * 0.42, hard_floor + (768.0 if mode == 'normal' else 1024.0))

        # Chunking is activation-only and executes stock block.mlp() for every
        # chunk.  Larger chunks reduce repeated weight faults when real headroom
        # exists; demotion is immediate, promotion is one rung per block.
        ladders = {
            'normal': (2048, 4096, 8192, 16384, 32768),
            'low_vram': (1024, 2048, 4096, 8192, 16384),
            'ultra_low_vram': (256, 512, 1024, 2048, 4096),
        }
        ladder = ladders.get(mode, ladders['low_vram'])
        current = int(state.get('chunk_tokens', ladder[0]) or ladder[0])
        if free_mb <= hard_floor:
            target_idx = 0
            zone = 'HARD_SAFE'
            barrier = True
        elif free_mb <= soft_floor:
            target_idx = min(1, len(ladder)-1)
            zone = 'CAUTION'
            barrier = True
        elif free_mb <= soft_floor + 1536.0:
            target_idx = min(2, len(ladder)-1)
            zone = 'BALANCED'
            barrier = (mode == 'ultra_low_vram')
        elif free_mb <= soft_floor + 3072.0:
            target_idx = min(3, len(ladder)-1)
            zone = 'FAST'
            barrier = False
        else:
            target_idx = len(ladder)-1
            zone = 'FAST_PLUS'
            barrier = False

        # V4: driver-free VRAM before demand loading is not enough evidence for a
        # FAST zone on a huge sequence. Cap the performance rung by geometry.
        if total_mb <= 18.5 * 1024.0 and tokens >= 120000:
            target_idx = min(target_idx, 1)
            zone = 'CAUTION_GEOMETRY'
            barrier = True
        elif total_mb <= 18.5 * 1024.0 and tokens >= 90000:
            target_idx = min(target_idx, 2)
            zone = 'BALANCED_GEOMETRY'
            barrier = True

        ci = 0
        for i, v in enumerate(ladder):
            if v <= current:
                ci = i
        if target_idx > ci + 1:
            target_idx = ci + 1
        selected = int(ladder[target_idx])

        # v0.4.49: the post-block AUTO controller is authoritative once the
        # real-shape resident INT8 MLP has passed exact parity.  V4's generic
        # geometry cap was written for the stock/re-faulting MLP path and was
        # immediately undoing the measured-safe 4096->8192 uplift on the very
        # next governor call.  Keep the verified floor while we remain outside
        # true memory-pressure zones; HARD_SAFE/CAUTION may still demote instantly.
        _verified_floor = int(state.get('resident_int8_exact_mlp_floor', 0) or 0)
        _verified_exact = (
            str(state.get('int8_cached_mlp_parity', 'unknown')).lower() == 'verified'
            and _verified_floor > 0
            and _backend in ('int8', 'int8-convrot-w4a4')
        )
        if _verified_exact and zone not in ('HARD_SAFE', 'CAUTION'):
            selected = max(selected, _verified_floor)
            if selected > int(ladder[-1]):
                selected = int(ladder[-1])
            if selected >= 8192 and zone.endswith('_GEOMETRY'):
                zone = 'VERIFIED_RESIDENT_INT8'

        # v0.4.72 / Governor V5: forward-level anti-thrash lock.
        #
        # On ~30k-token continuation forwards the allocator naturally oscillates
        # around the hard/soft floor by ~200 MB. V4 interpreted that temporary
        # block-local swing as a policy change and alternated MLP 2048<->4096
        # across almost every transformer block, forcing barriers repeatedly.
        #
        # For long-but-not-giant forwards, probe blocks 0 and 1 and then lock the
        # safest chunk/barrier observed for the rest of THIS denoise forward.
        # The lock resets automatically when v12_int8_sol_forward_generation
        # increments on the next diffusion step.
        _fwd_gen = int(state.get('v12_int8_sol_forward_generation', 0) or 0)
        _lock_enabled = (
            _v12_is_int8_family(state)
            and 28000 <= int(tokens) < 90000
            and _fwd_gen > 0
        )
        if _lock_enabled:
            if int(state.get('v5_governor_forward_generation', -1)) != _fwd_gen:
                state['v5_governor_forward_generation'] = _fwd_gen
                state['v5_governor_probe_count'] = 0
                state['v5_governor_probe_selected_min'] = None
                state['v5_governor_probe_barrier_any'] = False
                state['v5_governor_locked_chunk'] = None
                state['v5_governor_locked_barrier'] = None
                state['v5_governor_lock_announced'] = False

            _locked_chunk = state.get('v5_governor_locked_chunk')
            if _locked_chunk is None:
                _probe_count = int(state.get('v5_governor_probe_count', 0) or 0) + 1
                state['v5_governor_probe_count'] = _probe_count
                _prev_min = state.get('v5_governor_probe_selected_min')
                state['v5_governor_probe_selected_min'] = (
                    int(selected) if _prev_min is None
                    else min(int(_prev_min), int(selected))
                )
                state['v5_governor_probe_barrier_any'] = bool(
                    state.get('v5_governor_probe_barrier_any', False) or bool(barrier)
                )

                # Block 0 + block 1 are enough to observe the transient residency
                # swing. Lock to the safer observed policy, never a faster one.
                if _probe_count >= 2:
                    state['v5_governor_locked_chunk'] = int(
                        state['v5_governor_probe_selected_min']
                    )
                    state['v5_governor_locked_barrier'] = bool(
                        state['v5_governor_probe_barrier_any']
                    )
                    _locked_chunk = int(state['v5_governor_locked_chunk'])
                    selected = _locked_chunk
                    barrier = bool(state['v5_governor_locked_barrier'])
                    zone = 'FORWARD_LOCKED_SAFE'
                    if not state.get('v5_governor_lock_announced'):
                        _lm_print(
                            '[MiniMaxH3 LongMedia][GOVERNOR V5 FORWARD LOCK] '
                            f'forward={_fwd_gen}; tokens={tokens}; '
                            f'probe_blocks={_probe_count}; '
                            f'mlp_chunk={selected}; barrier={barrier}; '
                            'policy=safest_of_first_two_blocks; '
                            'scope=current_diffusion_forward',
                            flush=True,
                        )
                        state['v5_governor_lock_announced'] = True
            else:
                selected = int(_locked_chunk)
                barrier = bool(state.get('v5_governor_locked_barrier', True))
                zone = 'FORWARD_LOCKED_SAFE'

        old_zone = str(state.get('adaptive_memory_zone', 'CALIBRATION_SAFE'))
        old_chunk = current
        old_barrier = bool(state.get('ultra_stage_barrier_required', True))
        state['chunk_tokens'] = selected
        state['ultra_stage_barrier_required'] = bool(barrier)
        state['adaptive_memory_zone'] = zone
        state['adaptive_memory_last_free_mb'] = round(free_mb, 1)
        state['adaptive_memory_total_mb'] = round(total_mb, 1)
        state['adaptive_memory_hard_floor_mb'] = round(hard_floor, 1)
        state['adaptive_memory_soft_floor_mb'] = round(soft_floor, 1)
        state['adaptive_memory_adjustments'] = int(state.get('adaptive_memory_adjustments', 0)) + 1
        if (old_zone, old_chunk, old_barrier) != (zone, selected, barrier):
            _lm_print(
                '[MiniMaxH3 LongMedia][GOVERNOR V3] '
                f'mode={mode} block={self.index:02d} free={free_mb:.0f}MB '
                f'hard={hard_floor:.0f}MB soft={soft_floor:.0f}MB tokens={tokens} '
                f'zone={zone}; MLP {old_chunk}->{selected}; barrier={barrier}',
                flush=True,
            )

    def _auto_vram_controller_after_probe(self, token_count):
        """TEST controller: tune runtime memory knobs once after block 0.

        User values are treated as the SAFE baseline.  The controller only
        becomes more aggressive when block 0 completed successfully and the
        post-block memory snapshot shows meaningful recoverable headroom.
        """
        state = self.state
        if self.index != 0 or state.get('auto_vram_controller_done'):
            return
        state['auto_vram_controller_done'] = True

        if not torch.cuda.is_available():
            state['auto_vram_controller_mode'] = 'SAFE'
            state['auto_vram_controller_reason'] = 'CUDA unavailable'
            return

        snap = _cuda_memory_snapshot()
        if not snap:
            state['auto_vram_controller_mode'] = 'SAFE'
            state['auto_vram_controller_reason'] = 'memory snapshot unavailable'
            return

        mb = 1024.0 ** 2
        total_mb = float(snap['total']) / mb
        free_mb = float(snap['driver_free']) / mb
        cached_mb = float(snap['cached']) / mb
        recoverable_mb = free_mb + cached_mb
        headroom_ratio = recoverable_mb / max(1.0, total_mb)

        before = {
            'mlp': int(state.get('chunk_tokens', self.chunk_tokens) or self.chunk_tokens),
            'qkv': int(state.get('sol_qkv_chunk_tokens', 0) or 0),
            'out': int(state.get('sol_out_proj_chunk_tokens', 0) or 0),
            'guard': int(state.get('inter_block_vram_guard_mb', 0) or 0),
            'cooldown': int(state.get('inter_block_guard_cooldown_blocks', 0) or 0),
            'late_start': int(state.get('late_block_guard_start', 40) or 40),
            'late_target': int(state.get('late_block_guard_target_mb', 0) or 0),
            'step_cleanup': int(state.get('step_boundary_cleanup_mb', 0) or 0),
        }

        _freeze_large_attention_chunks = int(token_count) >= 180000
        _backend = str(state.get('model_runtime_backend', 'unknown')).lower()

        # Conservative v1 thresholds.  Block 0 has already proven that the
        # sequence fits with the user's SAFE baseline.  We only relax memory
        # management when there is enough free + allocator-reclaimable memory.
        if recoverable_mb >= 6144.0 or headroom_ratio >= 0.38:
            mode = 'FAST'
            state['chunk_tokens'] = before['mlp']
            state['sol_qkv_chunk_tokens'] = before['qkv']
            state['sol_out_proj_chunk_tokens'] = before['out']
            state['inter_block_vram_guard_mb'] = min(before['guard'], 768) if before['guard'] > 0 else 0
            state['inter_block_guard_cooldown_blocks'] = max(before['cooldown'], 8)
            state['late_block_guard_start'] = max(before['late_start'], 46)
            state['late_block_guard_target_mb'] = min(before['late_target'], 4096) if before['late_target'] > 0 else 0
            state['step_boundary_cleanup_mb'] = min(before['step_cleanup'], 1024) if before['step_cleanup'] > 0 else 0
        elif recoverable_mb >= 3584.0 or headroom_ratio >= 0.24:
            mode = 'BALANCED'
            state['chunk_tokens'] = before['mlp']
            state['sol_qkv_chunk_tokens'] = before['qkv']
            state['sol_out_proj_chunk_tokens'] = before['out']
            state['inter_block_vram_guard_mb'] = min(before['guard'], 1024) if before['guard'] > 0 else 0
            state['inter_block_guard_cooldown_blocks'] = max(before['cooldown'], 6)
            state['late_block_guard_start'] = max(before['late_start'], 44)
            state['late_block_guard_target_mb'] = min(before['late_target'], 5120) if before['late_target'] > 0 else 0
            state['step_boundary_cleanup_mb'] = min(before['step_cleanup'], 1536) if before['step_cleanup'] > 0 else 0
        else:
            mode = 'SAFE'
            state['chunk_tokens'] = before['mlp']
            state['sol_qkv_chunk_tokens'] = before['qkv']
            state['sol_out_proj_chunk_tokens'] = before['out']
            state['inter_block_vram_guard_mb'] = before['guard']
            state['inter_block_guard_cooldown_blocks'] = before['cooldown']
            state['late_block_guard_start'] = before['late_start']
            state['late_block_guard_target_mb'] = before['late_target']
            state['step_boundary_cleanup_mb'] = before['step_cleanup']

        # v0.4.48: after resident INT8 MLP parity is verified, chunking becomes a
        # compute-throughput problem rather than a safety problem.  Larger chunks
        # reduce kernel-launch, loop, and per-chunk norm/gate overhead while the
        # exact stock INT8 math is preserved by the verified resident path.
        _resident_int8_exact = (
            _backend in ('int8', 'int8-convrot-w4a4')
            and str(state.get('int8_cached_mlp_parity', 'unknown')).lower() == 'verified'
        )
        if _resident_int8_exact:
            _base_mlp = int(before['mlp'])
            _target_mlp = _base_mlp
            _giant_floor_applied = False

            # v0.4.73: measured giant-sequence floor.
            #
            # On the 127k-token global refiner, V4 enters at 1024 because the
            # pre-demand mem_get_info sample is intentionally conservative. But
            # after the first complete H3 block, the real snapshot shows ~2 GB
            # free+reclaimable headroom and exact resident-INT8 MLP parity has
            # already passed. Promote only to 2048, based on that real probe.
            #
            # This halves MLP chunk-loop count (125 -> ~63 at 127k tokens) while
            # preserving the same stock quantized math. HARD_SAFE/CAUTION can
            # still demote later if actual pressure appears.
            if (
                90000 <= int(token_count) < 150000
                and recoverable_mb >= 1800.0
                and headroom_ratio >= 0.10
            ):
                _target_mlp = max(_target_mlp, 2048)
                _giant_floor_applied = True
            elif recoverable_mb >= 8192.0 or headroom_ratio >= 0.50:
                _target_mlp = max(_target_mlp, 8192)
            elif recoverable_mb >= 5632.0 or headroom_ratio >= 0.34:
                _target_mlp = max(_target_mlp, 6144)

            state['chunk_tokens'] = _target_mlp
            state['resident_int8_exact_mlp_floor'] = int(_target_mlp)
            state['v6_giant_mlp_floor_applied'] = bool(_giant_floor_applied)
            if _giant_floor_applied:
                _lm_print(
                    '[MiniMaxH3 LongMedia][GOVERNOR V6 GIANT INT8 FLOOR] '
                    f'tokens={int(token_count)}; '
                    f'recoverable={recoverable_mb:.0f}MB; '
                    f'headroom={headroom_ratio*100.0:.1f}%; '
                    f'mlp_chunk={_base_mlp}->{int(_target_mlp)}; '
                    'source=post_block0_real_memory_probe; '
                    'hard_pressure_demotion_still_enabled=True',
                    flush=True,
                )
            state['resident_int8_exact_chunk_uplift'] = {
                'enabled': True,
                'recoverable_mb': round(recoverable_mb, 1),
                'headroom_ratio': round(headroom_ratio, 4),
                'before': _base_mlp,
                'after': int(_target_mlp),
            }
        else:
            state['resident_int8_exact_mlp_floor'] = 0
            state['resident_int8_exact_chunk_uplift'] = {
                'enabled': False,
                'recoverable_mb': round(recoverable_mb, 1),
                'headroom_ratio': round(headroom_ratio, 4),
                'before': int(before['mlp']),
                'after': int(state.get('chunk_tokens', before['mlp'])),
            }

        # Backend safety caps.  NVFP4 is intentionally uncapped here because
        # its current settings are the measured reference baseline.
        if _backend in ('int8', 'int8-convrot-w4a4'):
            _dense_existing_contract = bool(
                str(state.get('sol_mode', '')).lower() == 'existing'
                and int(token_count) >= 32768
                and not bool(state.get('fasth3_vsa_active', False))
                and not bool(state.get('fastvideo_vsa_active', False))
                and not bool(state.get('external_sla_direct_fastpath', False))
            )
            if _dense_existing_contract:
                # v0.5.29: EXISTING has a fundamentally different workspace
                # contract from bounded SOL.  Keep the user's SAFE reclamation
                # policy alive; the pre-QKV workspace guard additionally sheds
                # *active* DynamicVRAM residency when a dense allocation needs it.
                state['inter_block_vram_guard_mb'] = before['guard']
                state['inter_block_guard_cooldown_blocks'] = before['cooldown']
                state['late_block_guard_start'] = before['late_start']
                state['late_block_guard_target_mb'] = before['late_target']
                state['step_boundary_cleanup_mb'] = before['step_cleanup']
                state['existing_dense_workspace_policy'] = True
            else:
                # Native QuantizedTensor residency is valuable on bounded SOL,
                # where QKV workspace is streamed/chunked and cannot demand a
                # multi-gigabyte contiguous activation burst.
                state['inter_block_vram_guard_mb'] = 0
                state['late_block_guard_target_mb'] = 0
                state['step_boundary_cleanup_mb'] = 0
                state['existing_dense_workspace_policy'] = False
            state['chunk_tokens'] = min(int(state.get('chunk_tokens', before['mlp'])), 16384)
            if int(state.get('sol_qkv_chunk_tokens', 0) or 0) > 0:
                _qv = str(state.get('runtime_quant_variant') or state.get('quant_variant') or '').lower()
                _requested_qkv = min(int(state['sol_qkv_chunk_tokens']), 16384)
                _is_giant_qkv = int(token_count) >= 90000

                # V8 contract:
                # - ordinary INT8 keeps the explicit request up to 16K;
                # - giant >=90K keeps 16K only after the real block0 probe proves
                #   >=1.8 GB reusable headroom;
                # - otherwise fall back to the proven 8K baseline.
                if _is_giant_qkv and _requested_qkv > 8192:
                    if recoverable_mb >= 1800.0 and headroom_ratio >= 0.10:
                        _effective_qkv = _requested_qkv
                        _qkv_fallback = False
                        _qkv_reason = 'post_block0_headroom_pass'
                    else:
                        _effective_qkv = 8192
                        _qkv_fallback = True
                        _qkv_reason = 'post_block0_headroom_fail'
                else:
                    _effective_qkv = _requested_qkv
                    _qkv_fallback = False
                    _qkv_reason = 'explicit_request'

                state['sol_qkv_chunk_tokens'] = int(_effective_qkv)
                state['v8_giant_qkv_requested'] = int(_requested_qkv)
                state['v8_giant_qkv_effective'] = int(_effective_qkv)
                state['v8_giant_qkv_fallback'] = bool(_qkv_fallback)
                state['v8_giant_qkv_reason'] = str(_qkv_reason)

                if _is_giant_qkv:
                    _q_chunks = max(
                        1,
                        (int(token_count) + int(_effective_qkv) - 1)
                        // int(_effective_qkv),
                    )
                    _lm_print(
                        '[MiniMaxH3 LongMedia][V8 GIANT QKV CONTRACT] '
                        f'tokens={int(token_count)}; '
                        f'requested={int(_requested_qkv)}; '
                        f'effective={int(_effective_qkv)}; '
                        f'chunks={int(_q_chunks)}; '
                        f'fallback={bool(_qkv_fallback)}; '
                        f'recoverable={recoverable_mb:.0f}MB; '
                        f'headroom={headroom_ratio*100.0:.1f}%; '
                        f'reason={_qkv_reason}',
                        flush=True,
                    )
            if int(state.get('sol_out_proj_chunk_tokens', 0) or 0) > 0:
                state['sol_out_proj_chunk_tokens'] = min(
                    int(state['sol_out_proj_chunk_tokens']), 16384
                )
        elif _backend in ('bf16', 'fp16', 'fp32'):
            state['chunk_tokens'] = min(int(state.get('chunk_tokens', before['mlp'])), 12288)
            if int(state.get('sol_qkv_chunk_tokens', 0) or 0) > 0:
                state['sol_qkv_chunk_tokens'] = min(
                    int(state['sol_qkv_chunk_tokens']), 8192
                )
            if int(state.get('sol_out_proj_chunk_tokens', 0) or 0) > 0:
                state['sol_out_proj_chunk_tokens'] = min(
                    int(state['sol_out_proj_chunk_tokens']), 12288
                )
        elif _backend in ('fp8', 'quantized-other'):
            state['chunk_tokens'] = min(int(state.get('chunk_tokens', before['mlp'])), 16384)
            if int(state.get('sol_out_proj_chunk_tokens', 0) or 0) > 0:
                state['sol_out_proj_chunk_tokens'] = min(
                    int(state['sol_out_proj_chunk_tokens']), 16384
                )

        after = {
            'mlp': int(state.get('chunk_tokens', before['mlp'])),
            'qkv': int(state.get('sol_qkv_chunk_tokens', before['qkv'])),
            'out': int(state.get('sol_out_proj_chunk_tokens', before['out'])),
            'guard': int(state.get('inter_block_vram_guard_mb', before['guard'])),
            'cooldown': int(state.get('inter_block_guard_cooldown_blocks', before['cooldown'])),
            'late_start': int(state.get('late_block_guard_start', before['late_start'])),
            'late_target': int(state.get('late_block_guard_target_mb', before['late_target'])),
            'step_cleanup': int(state.get('step_boundary_cleanup_mb', before['step_cleanup'])),
        }

        state['auto_vram_controller_mode'] = mode
        state['auto_vram_controller_probe'] = {
            'tokens': int(token_count),
            'total_mb': round(total_mb, 1),
            'driver_free_mb': round(free_mb, 1),
            'cached_mb': round(cached_mb, 1),
            'recoverable_mb': round(recoverable_mb, 1),
            'headroom_ratio': round(headroom_ratio, 4),
        }
        state['auto_vram_controller_before'] = before
        state['auto_vram_controller_after'] = after

        _lm_print(
            '[MiniMaxH3 LongMedia][AUTO VRAM] block0 probe: '
            f'backend={_backend}, {int(token_count)} tokens, total={total_mb:.0f} MB, '
            f'free={free_mb:.0f} MB, cached={cached_mb:.0f} MB, '
            f'recoverable={recoverable_mb:.0f} MB ({headroom_ratio*100.0:.1f}%) -> {mode}',
            flush=True,
        )
        _lm_print(
            '[MiniMaxH3 LongMedia][AUTO VRAM] runtime tuning: '
            f"MLP {before['mlp']}->{after['mlp']}, "
            f"QKV {before['qkv']}->{after['qkv']}, "
            f"OUT {before['out']}->{after['out']}, "
            f"guard {before['guard']}->{after['guard']} MB, "
            f"cooldown {before['cooldown']}->{after['cooldown']}, "
            f"late {before['late_start']}->{after['late_start']} "
            f"target {before['late_target']}->{after['late_target']} MB, "
            f"step-cleanup {before['step_cleanup']}->{after['step_cleanup']} MB",
            flush=True,
        )

        _uplift = state.get('resident_int8_exact_chunk_uplift') or {}
        if _uplift.get('enabled'):
            _lm_print(
                '[MiniMaxH3 LongMedia][RESIDENT INT8 CHUNK UPLIFT] '
                f"verified_exact_mlp=True recoverable={_uplift.get('recoverable_mb', 0):.1f}MB "
                f"headroom={float(_uplift.get('headroom_ratio', 0.0))*100.0:.1f}% "
                f"MLP {int(_uplift.get('before', before['mlp']))}->{int(_uplift.get('after', after['mlp']))}",
                flush=True,
            )

    def _int8_prefetch_guard(self):
        """V321 oversubscription-aware emergency safety net for native Comfy INT8.

        A model larger than physical VRAM normally runs through Comfy's dynamic
        weight streaming.  In that regime a low CUDA *driver_free* value is not,
        by itself, an emergency: PyTorch's allocator cache is reclaimable and is
        valuable for avoiding block-by-block allocation/transfer thrash.

        Only trim when BOTH physical free VRAM and effective reclaimable headroom
        are critically low.  A block cooldown prevents repeated cache destruction
        when the workload is hovering around the emergency boundary.
        """
        state = self.state
        backend = str(state.get('model_runtime_backend', 'unknown')).lower()
        if backend not in ('int8', 'int8-convrot-w4a4') or not torch.cuda.is_available():
            return

        cooldown_left = int(state.get('int8_residency_guard_cooldown_left', 0) or 0)
        if cooldown_left > 0:
            state['int8_residency_guard_cooldown_left'] = cooldown_left - 1
            return

        snap = _cuda_memory_snapshot()
        if not snap:
            return
        mb = 1024.0 ** 2
        free_mb = float(snap['driver_free']) / mb
        cached_mb = float(snap['cached']) / mb
        effective_mb = free_mb + cached_mb

        emergency_free_mb = float(state.get('int8_residency_emergency_free_mb', 384) or 384)
        emergency_effective_mb = float(
            state.get('int8_residency_emergency_effective_mb', 768) or 768
        )
        min_cached_mb = float(state.get('int8_residency_min_cached_mb', 256) or 256)

        # Fast path for healthy streaming.  Example from a 20 GB model on a 16 GB
        # GPU: free~120 MB + cached~2780 MB => ~2.9 GB effective headroom.  The
        # allocator cache must be preserved in that state.
        if free_mb >= emergency_free_mb or effective_mb >= emergency_effective_mb:
            state['int8_residency_last_effective_mb'] = float(effective_mb)
            return
        if cached_mb < min_cached_mb:
            # There is little useful cache to reclaim, so empty_cache would not
            # materially improve the situation and would only add synchronization.
            return

        before_free, before_cached, before_effective = free_mb, cached_mb, effective_mb
        try:
            gc.collect()
            comfy.model_management.soft_empty_cache()
        except Exception as exc:
            _lm_print('[MiniMaxH3 LongMedia][V321 INT8 RESIDENCY] emergency cleanup failed: '
                  f'block={self.index}, {type(exc).__name__}: {exc}', flush=True)
            return

        state['int8_residency_emergency_trim_count'] = int(
            state.get('int8_residency_emergency_trim_count', 0) or 0
        ) + 1
        state['int8_residency_guard_cooldown_left'] = int(
            state.get('int8_residency_guard_cooldown_blocks', 8) or 8
        )
        after = _cuda_memory_snapshot()
        if after:
            after_free = float(after['driver_free']) / mb
            after_cached = float(after['cached']) / mb
            after_effective = after_free + after_cached
            _lm_print('[MiniMaxH3 LongMedia][V321 INT8 RESIDENCY] EMERGENCY TRIM: '
                  f'block={self.index}, free {before_free:.0f}->{after_free:.0f} MB, '
                  f'cached {before_cached:.0f}->{after_cached:.0f} MB, '
                  f'effective {before_effective:.0f}->{after_effective:.0f} MB, '
                  f'cooldown={state["int8_residency_guard_cooldown_left"]} blocks', flush=True)

    def _late_block_hard_guard(self, phase):
        """TEST: pressure-aware late guard with hysteresis and cooldown.

        Avoids repeated soft_empty_cache() calls when effective headroom is
        oscillating just below the configured late target.
        """
        state = self.state
        start_block = int(state.get('late_block_guard_start', 40) or 40)
        if self.index < start_block or not torch.cuda.is_available():
            return

        target_mb = float(int(state.get('late_block_guard_target_mb', 0) or 0))
        min_cached_mb = float(int(state.get('late_block_guard_min_cached_mb', 0) or 0))
        if target_mb <= 0:
            return

        snap = _cuda_memory_snapshot()
        if not snap:
            return

        mb = 1024.0 ** 2
        free_before = float(snap['driver_free']) / mb
        cached_before = float(snap['cached']) / mb
        effective_before = free_before + cached_before

        auto_mode = str(state.get('auto_vram_controller_mode') or 'SAFE').upper()
        effective_target = target_mb
        if auto_mode == 'FAST':
            effective_target = min(effective_target, 3584.0)
        elif auto_mode == 'BALANCED':
            effective_target = min(effective_target, 4608.0)

        # Hysteresis band:
        # - above target: never trim
        # - inside lower band (target - 1024 MB): avoid repeated trims
        # - below hard threshold: cleanup is allowed immediately
        hysteresis_mb = float(state.get('late_guard_hysteresis_mb', 1024.0) or 1024.0)
        hard_threshold = max(0.0, effective_target - hysteresis_mb)

        # Cooldown is counted in late-guard phases (pre_attention/pre_ffn).
        # A trim starts a cooldown; only a hard pressure drop may override it.
        cooldown_phases = int(state.get('late_guard_cooldown_phases', 4) or 4)
        cooldown_left = int(state.get('late_guard_cooldown_left', 0) or 0)

        if effective_before >= effective_target:
            state['late_guard_skipped_count'] = int(state.get('late_guard_skipped_count', 0) or 0) + 1
            if cooldown_left > 0:
                state['late_guard_cooldown_left'] = max(0, cooldown_left - 1)
            skipped = int(state['late_guard_skipped_count'])
            if skipped == 1 or skipped % 20 == 0:
                _lm_print(
                    '[MiniMaxH3 LongMedia][LATE GUARD] late guard skip: '
                    f'block {self.index} {phase}, effective={effective_before:.0f} MB '
                    f'>= target={effective_target:.0f} MB',
                    flush=True,
                )
            return

        # Inside the hysteresis band, keep cache intact rather than repeatedly
        # reclaiming a few hundred MB and immediately forcing reload/allocation.
        if effective_before >= hard_threshold:
            state['late_guard_hysteresis_skip_count'] = int(
                state.get('late_guard_hysteresis_skip_count', 0) or 0
            ) + 1
            if cooldown_left > 0:
                state['late_guard_cooldown_left'] = max(0, cooldown_left - 1)
            hs = int(state['late_guard_hysteresis_skip_count'])
            if hs == 1 or hs % 20 == 0:
                _lm_print(
                    '[MiniMaxH3 LongMedia][LATE GUARD] hysteresis skip: '
                    f'block {self.index} {phase}, effective={effective_before:.0f} MB, '
                    f'band={hard_threshold:.0f}..{effective_target:.0f} MB',
                    flush=True,
                )
            return

        # Cooldown blocks repeated trims unless we are in genuinely hard pressure.
        hard_pressure = effective_before < max(0.0, effective_target - 2048.0)
        if cooldown_left > 0 and not hard_pressure:
            state['late_guard_cooldown_skip_count'] = int(
                state.get('late_guard_cooldown_skip_count', 0) or 0
            ) + 1
            state['late_guard_cooldown_left'] = max(0, cooldown_left - 1)
            cs = int(state['late_guard_cooldown_skip_count'])
            if cs == 1 or cs % 20 == 0:
                _lm_print(
                    '[MiniMaxH3 LongMedia][LATE GUARD] cooldown skip: '
                    f'block {self.index} {phase}, effective={effective_before:.0f} MB, '
                    f'cooldown_left={cooldown_left}',
                    flush=True,
                )
            return

        # Nothing meaningful to reclaim: let emergency guard / Sol OOM retry
        # handle the true low-memory event instead of forcing a useless trim.
        if cached_before < min_cached_mb:
            state['late_guard_low_cache_skip_count'] = int(
                state.get('late_guard_low_cache_skip_count', 0) or 0
            ) + 1
            if cooldown_left > 0:
                state['late_guard_cooldown_left'] = max(0, cooldown_left - 1)
            return

        try:
            gc.collect()
            comfy.model_management.soft_empty_cache()
        except Exception as exc:
            _lm_print(
                f"[MiniMaxH3 LongMedia][LATE GUARD] cleanup failed "
                f"at block {self.index} {phase}: {exc!r}",
                flush=True,
            )
            return

        state['late_guard_cooldown_left'] = cooldown_phases
        after = _cuda_memory_snapshot()
        state['late_guard_trim_count'] = int(state.get('late_guard_trim_count', 0) or 0) + 1

        if after:
            free_after = float(after['driver_free']) / mb
            cached_after = float(after['cached']) / mb
            _lm_print(
                '[MiniMaxH3 LongMedia][LATE GUARD] late guard TRIM: '
                f'block {self.index} {phase}, effective={effective_before:.0f} MB, '
                f'free {free_before:.0f}->{free_after:.0f} MB, '
                f'cached {cached_before:.0f}->{cached_after:.0f} MB, '
                f'cooldown={cooldown_phases}',
                flush=True,
            )

    def _trace_attention(self, attn, x, rope_freqs, transformer_options, measure):
        """Execute stock H3 Attention.forward in measured substages.

        This mirrors comfy's MiniMax H3 Attention implementation exactly:
        qkv projection -> fused RMSNorm/RoPE -> optimized_attention -> out projection.
        It is used only for block 0 of the first forward so normal execution is
        unaffected after the diagnostic sample.
        """
        from comfy.ldm.modules.attention import optimized_attention
        import comfy.quant_ops

        s = int(x.shape[0])
        inner = int(attn.heads * attn.head_dim)

        q, k, v = measure('attn_qkv_proj', lambda: attn.qkv_proj(x).split(inner, dim=-1))
        v = v.view(s, attn.heads, attn.head_dim)

        def _norm_rope():
            nonlocal q, k
            if rope_freqs is not None:
                qv = q.view(1, s, attn.heads, attn.head_dim)
                kv = k.view(1, s, attn.heads, attn.head_dim)
                qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
                kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
                rot = int(rope_freqs.shape[-3] * 2)
                if comfy.model_management.in_training:
                    qv, kv = comfy.quant_ops.ck.rms_rope_split_half(
                        qv, kv, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
                    )
                else:
                    comfy.quant_ops.ck.rms_rope_split_half_(
                        qv, kv, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
                    )
                return qv[0], kv[0]
            return (
                attn.q_norm(q.view(s, attn.heads, attn.head_dim)),
                attn.k_norm(k.view(s, attn.heads, attn.head_dim)),
            )

        q, k = measure('attn_rms_rope', _norm_rope)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        out = measure(
            'attn_kernel',
            lambda: optimized_attention(
                q, k, v, attn.heads, mask=None, skip_reshape=True,
                transformer_options=transformer_options,
            ),
        )
        # Drop Q/K/V before the output projection just as soon as the attention
        # kernel no longer needs them, so this diagnostic does not inflate the
        # projection peak by extending their lifetime.
        del q, k, v
        return measure('attn_out_proj', lambda: attn.out_proj(out.squeeze(0)))

    def _chunk_mlp(self, block, h):
        """Legacy exact token-chunked MLP path retained for compatibility."""
        token_count = int(h.shape[0])
        if token_count <= self.chunk_tokens:
            return block.mlp(h)

        state = self.state
        chunks = (token_count + self.chunk_tokens - 1) // self.chunk_tokens
        state['mlp_chunked_calls'] = int(state.get('mlp_chunked_calls', 0)) + 1
        state['max_sequence_tokens'] = max(int(state.get('max_sequence_tokens', 0)), token_count)
        state['max_chunks_per_mlp'] = max(int(state.get('max_chunks_per_mlp', 0)), chunks)
        if not state.get('announced'):
            _lm_print(
                '[MiniMaxH3 LongMedia] Low-VRAM MLP enabled: '
                f'sequence {token_count} tokens -> {chunks} chunks of <= {self.chunk_tokens}',
                flush=True,
            )
            state['announced'] = True

        state['mlp_inplace_reuse'] = True
        for start in range(0, token_count, self.chunk_tokens):
            end = min(token_count, start + self.chunk_tokens)
            chunk_out = block.mlp(h[start:end])
            h[start:end].copy_(chunk_out)
            del chunk_out
        return h

    def _chunk_norm2_mlp_gate_residual(self, block, x, shift, scale, gate, segments):
        """Stream the entire second half of an H3 block over token chunks.

        Stock H3 materializes ``norm2(x)`` plus AdaLN modulation as a full
        [tokens, hidden] tensor before entering the FFN.  At ~225k packed tokens
        that BF16 buffer is about 2.26 GiB.  This path keeps the whole sequence
        resident only once in ``x`` and performs, per token chunk:

            norm2 -> scale/shift modulation -> MLP -> gate -> residual add

        Every operation in this chain is token-local.  Later chunks therefore
        read untouched rows of ``x`` and are mathematically independent of rows
        already updated by earlier chunks.  No full-size norm2/modulated or MLP
        output tensor is ever created.
        """
        token_count = int(x.shape[0])
        self.state['current_token_count'] = token_count
        self.state['last_token_count'] = token_count
        # AUTO controller may raise/lower the runtime MLP chunk after
        # block-0 calibration.  self.chunk_tokens remains the user/manual fallback.
        runtime_chunk_tokens = int(self.state.get('chunk_tokens', self.chunk_tokens) or self.chunk_tokens)
        state = self.state
        _ultra_streaming = str(state.get('memory_mode', 'normal')) == 'ultra_low_vram'
        _chunk_floor = 64 if _ultra_streaming else 256
        chunk_tokens = min(max(_chunk_floor, runtime_chunk_tokens), max(1, token_count))
        if bool(state.get('auto_mlp_chunk_enabled')) and (not _ultra_streaming) and torch.cuda.is_available():
            # Estimate usable allocator headroom without forcing an empty_cache().
            # fc1 + SwiGLU temporary storage is roughly 80-90 KiB/token for H3;
            # 96 KiB/token deliberately includes allocator/alignment margin.
            try:
                free_b, _total_b = torch.cuda.mem_get_info(x.device)
                allocated_b = torch.cuda.memory_allocated(x.device)
                reserved_b = torch.cuda.memory_reserved(x.device)
                reclaimable_b = max(0, int(reserved_b) - int(allocated_b))
                effective_b = int(free_b) + reclaimable_b
                safety_b = int(state.get('auto_mlp_chunk_safety_mb', 640)) * 1024 * 1024
                per_token_b = max(1, int(state.get('auto_mlp_chunk_bytes_per_token', 96 * 1024)))
                usable_b = max(0, effective_b - safety_b)
                max_tokens_by_mem = max(1024, usable_b // per_token_b)
                ceiling = min(int(runtime_chunk_tokens), int(max_tokens_by_mem))
                ladder = (16384, 8192, 4096, 2048, 1024)
                selected = 1024
                for candidate in ladder:
                    if candidate <= ceiling:
                        selected = candidate
                        break
                chunk_tokens = min(max(1024, selected), max(1, token_count))
                previous = state.get('auto_mlp_chunk_last')
                if previous != int(chunk_tokens):
                    state['auto_mlp_chunk_last'] = int(chunk_tokens)
                    state['auto_mlp_chunk_changes'] = int(state.get('auto_mlp_chunk_changes', 0)) + 1
                    _lm_print(
                        '[MiniMaxH3 LongMedia][V30 AUTO MLP] '
                        f'block={self.index}, effective_headroom={effective_b/(1024*1024):.0f} MB, '
                        f'safety={safety_b/(1024*1024):.0f} MB, selected={chunk_tokens} tokens',
                        flush=True,
                    )
            except Exception as _auto_mlp_exc:
                if not state.get('auto_mlp_chunk_error_announced'):
                    _lm_print(
                        '[MiniMaxH3 LongMedia][V30 AUTO MLP] fallback to policy chunk: '
                        f'{type(_auto_mlp_exc).__name__}: {_auto_mlp_exc}',
                        flush=True,
                    )
                    state['auto_mlp_chunk_error_announced'] = True
        chunks = (token_count + chunk_tokens - 1) // chunk_tokens
        state['mlp_chunked_calls'] = int(state.get('mlp_chunked_calls', 0)) + 1
        state['mlp_fused_gate_residual_calls'] = int(state.get('mlp_fused_gate_residual_calls', 0)) + 1
        state['norm2_mlp_fused_calls'] = int(state.get('norm2_mlp_fused_calls', 0)) + 1
        state['max_sequence_tokens'] = max(int(state.get('max_sequence_tokens', 0)), token_count)
        state['max_chunks_per_mlp'] = max(int(state.get('max_chunks_per_mlp', 0)), chunks)
        state['mlp_fused_gate_residual'] = True
        state['norm2_mlp_fused_streaming'] = True
        if not state.get('announced'):
            _lm_print(
                '[MiniMaxH3 LongMedia] Low-VRAM fused norm2+MLP+gate+residual enabled: '
                f'sequence {token_count} tokens -> {chunks} chunks of <= {chunk_tokens}',
                flush=True,
            )
            state['announced'] = True

        # Tiny scalar per-segment vectors are cast once and reused. Newer/alternate
        # Comfy H3 layouts may provide a row tensor (one modulation row per token)
        # instead of a single Python scalar. Preserve the stock H3 semantics for
        # both forms instead of coercing a multi-element tensor through int().
        rows = {}

        def _scalar_mod_row(row):
            if torch.is_tensor(row):
                if int(row.numel()) != 1:
                    return None
                return int(row.reshape(-1)[0].item())
            return int(row)

        def _mod_row_indices(row, seg_a, seg_b, lo, hi):
            scalar = _scalar_mod_row(row)
            if scalar is not None:
                return scalar
            flat = row.reshape(-1)
            seg_a = int(seg_a)
            seg_b = int(seg_b)
            lo = int(lo)
            hi = int(hi)
            seg_len = max(0, seg_b - seg_a)
            if int(flat.numel()) == seg_len:
                ids = flat[lo - seg_a:hi - seg_a]
            elif int(flat.numel()) == token_count:
                ids = flat[lo:hi]
            else:
                raise RuntimeError(
                    '[0.4.20 MOD-ROW COMPAT] unsupported multi-row modulation layout: '
                    f'row_shape={tuple(row.shape)}, row_numel={int(flat.numel())}, '
                    f'segment={seg_a}:{seg_b} ({seg_len} tokens), total_tokens={token_count}. '
                    'Update ComfyUI/LongMedia together or report this layout.'
                )
            return ids.to(device=shift.device, dtype=torch.long)

        def _segment_mod_params(row, seg_a, seg_b, lo, hi):
            row_sel = _mod_row_indices(row, seg_a, seg_b, lo, hi)
            if isinstance(row_sel, int):
                cached = rows.get(row_sel)
                if cached is None:
                    cached = (
                        shift[row_sel].to(dtype=x.dtype, device=x.device),
                        scale[row_sel].to(dtype=x.dtype, device=x.device),
                        gate[row_sel].to(dtype=x.dtype, device=x.device),
                    )
                    rows[row_sel] = cached
                return cached
            return (
                shift.index_select(0, row_sel).to(dtype=x.dtype, device=x.device),
                scale.index_select(0, row_sel).to(dtype=x.dtype, device=x.device),
                gate.index_select(0, row_sel).to(dtype=x.dtype, device=x.device),
            )

        for _a, _b, row in segments:
            scalar = _scalar_mod_row(row)
            if scalar is not None and scalar not in rows:
                rows[scalar] = (
                    shift[scalar].to(dtype=x.dtype, device=x.device),
                    scale[scalar].to(dtype=x.dtype, device=x.device),
                    gate[scalar].to(dtype=x.dtype, device=x.device),
                )

        # v0.3.60: block-resident native INT8 MLP weights.  v0.3.59 called
        # stock block.mlp() once per token chunk, which repeated cast/VBAR weight
        # preparation dozens of times per transformer block and roughly doubled
        # iteration time on the 32.4GB/16GB out-of-core case.  Prepare fc1/fc2
        # once per block and reuse them, but only after a strict stock-vs-cached
        # numerical parity probe on block 0.  If parity fails, the fast path is
        # disabled globally for the rest of the job.
        _int8_backend = (
            _v12_is_int8_family(state)
            and not comfy.model_management.in_training
            and state.get('int8_cached_mlp_parity', 'unknown') != 'failed'
        )
        state['stock_mlp_math'] = False
        _fc1_handle = _fc2_handle = None

        _v19_active = _v19_selected_block(state, self.index)
        _v19_offsets = (
            _v19_probe_offsets(token_count, 8192) if _v19_active else []
        )
        _v19_h_reference = None
        _v19_mlp_reference = None
        _v19_final_reference = None
        _v19_h_actual_parts = []
        _v19_mlp_actual_parts = []
        _v19_final_actual_parts = []
        if _v19_active:
            _v19_before_parts = []
            _v19_h_parts = []
            _v19_lengths = []
            for _offset in _v19_offsets:
                _probe_end = min(token_count, int(_offset) + 4)
                _before = x[int(_offset):_probe_end].detach().clone()
                _href = block.norm2(_before)
                for a, b, row in segments:
                    lo = max(int(_offset), int(a))
                    hi = min(_probe_end, int(b))
                    if lo >= hi:
                        continue
                    local_lo = lo - int(_offset)
                    local_hi = hi - int(_offset)
                    shift_row, scale_row, _gate_row = _segment_mod_params(row, a, b, lo, hi)
                    _href[local_lo:local_hi].mul_(
                        1.0 + scale_row
                    ).add_(shift_row)
                _v19_before_parts.append(_before)
                _v19_h_parts.append(_href)
                _v19_lengths.append(int(_href.shape[0]))
            _v19_h_reference = torch.cat(_v19_h_parts, dim=0)
            _v19_mlp_reference = block.mlp(
                _v19_h_reference.clone()
            ).detach()
            _v19_final_parts = []
            _cursor = 0
            for _offset, _before, _length in zip(
                _v19_offsets, _v19_before_parts, _v19_lengths
            ):
                _expected = _before.clone()
                _mlp_piece = _v19_mlp_reference[_cursor:_cursor + _length]
                _probe_end = int(_offset) + _length
                for a, b, row in segments:
                    lo = max(int(_offset), int(a))
                    hi = min(_probe_end, int(b))
                    if lo >= hi:
                        continue
                    local_lo = lo - int(_offset)
                    local_hi = hi - int(_offset)
                    _shift_row, _scale_row, gate_row = _segment_mod_params(row, a, b, lo, hi)
                    _expected[local_lo:local_hi].addcmul_(
                        _mlp_piece[local_lo:local_hi], gate_row
                    )
                _v19_final_parts.append(_expected)
                _cursor += _length
            _v19_final_reference = torch.cat(_v19_final_parts, dim=0)
            del _v19_before_parts, _v19_h_parts, _v19_final_parts

        try:
            for start in range(0, token_count, chunk_tokens):
                end = min(token_count, start + chunk_tokens)
                # norm2 produces only a chunk-sized temporary.
                h_chunk = block.norm2(x[start:end])

                # Apply the same packed-segment AdaLN modulation as the stock full
                # path, but only to intersections inside this chunk.
                for a, b, row in segments:
                    lo = max(start, int(a))
                    hi = min(end, int(b))
                    if lo >= hi:
                        continue
                    local_lo = lo - start
                    local_hi = hi - start
                    shift_row, scale_row, _gate_row = _segment_mod_params(row, a, b, lo, hi)
                    h_chunk[local_lo:local_hi].mul_(1.0 + scale_row).add_(shift_row)

                if _v19_active:
                    for _offset in _v19_offsets:
                        if start <= int(_offset) < end:
                            _local = int(_offset) - start
                            _rows = min(4, end - int(_offset))
                            _v19_h_actual_parts.append(
                                h_chunk[_local:_local + _rows].detach().clone()
                            )

                if _int8_backend and _fc1_handle is None:
                    _mlp_probe = h_chunk[:4]
                    _need_parity = False
                    _stock_mlp_probe = None
                    if _need_parity:
                        # Pay one tiny stock call once per job, before enabling
                        # resident weights for every subsequent block/chunk.
                        _stock_mlp_probe = block.mlp(_mlp_probe).detach()

                    _fc1_handle = _int8_prepare_block_linear(
                        block.mlp.fc1, _mlp_probe
                    )
                    # fc2 must be prepared with its real post-SwiGLU input shape,
                    # not the hidden-size fc1 input used by older builds.
                    _prep_fc1 = torch.nn.functional.linear(
                        _mlp_probe, _fc1_handle['weight'], _fc1_handle['bias']
                    )
                    _prep_gate, _prep_up = _prep_fc1.chunk(2, dim=-1)
                    _fc2_probe = torch.nn.functional.silu(_prep_gate).mul_(_prep_up)
                    _fc2_handle = _int8_prepare_block_linear(
                        block.mlp.fc2, _fc2_probe
                    )
                    del _prep_gate, _prep_up, _prep_fc1, _fc2_probe

                    if _need_parity:
                        _cached_fc1_probe = _int8_cached_linear(
                            _fc1_handle, _mlp_probe
                        )
                        _cached_mlp_probe = _int8_cached_linear(
                            _fc2_handle, _cached_fc1_probe, input_act='swiglu'
                        )
                        try:
                            _s = _stock_mlp_probe.detach().to(device='cpu', dtype=torch.float32)
                            _c = _cached_mlp_probe.detach().to(device='cpu', dtype=torch.float32)
                            _d = _c - _s
                            _eps = 1.0e-12
                            _rms_ref = float(torch.sqrt(torch.mean(_s.square())).item()) if _s.numel() else 0.0
                            _rms_diff = float(torch.sqrt(torch.mean(_d.square())).item()) if _d.numel() else 0.0
                            _rel = _rms_diff / max(_eps, _rms_ref)
                            _den = float(torch.linalg.vector_norm(_s).item() * torch.linalg.vector_norm(_c).item()) if _s.numel() else 1.0
                            _cos = float(torch.dot(_s.flatten(), _c.flatten()).item() / max(_eps, _den)) if _s.numel() else 1.0
                            _finite = bool(torch.isfinite(_s).all() and torch.isfinite(_c).all() and torch.isfinite(_d).all())
                            _ok = _finite and _rel <= 5.0e-5 and _cos >= 0.99999
                        except Exception:
                            _ok, _rel, _cos = False, float('inf'), -1.0
                        state['int8_cached_mlp_parity'] = 'verified' if _ok else 'failed'
                        _lm_print(
                            '[MiniMaxH3 LongMedia][MLP PARITY] '
                            f"{'PASS' if _ok else 'FAIL'} rel_rms={_rel:.3e} cosine={_cos:.8f}; "
                            + ('block-resident fc1/fc2 enabled' if _ok else 'falling back to stock MLP'),
                            flush=True,
                        )
                        del _stock_mlp_probe, _cached_fc1_probe, _cached_mlp_probe
                        if not _ok:
                            _int8_release_block_linear(_fc2_handle)
                            _int8_release_block_linear(_fc1_handle)
                            _fc1_handle = _fc2_handle = None
                            _int8_backend = False
                            state['stock_mlp_math'] = True

                    if _fc1_handle is not None and not state.get('int8_block_mlp_weights_announced'):
                        _lm_print(
                            '[MiniMaxH3 LongMedia][BLOCK-RESIDENT EXACT MLP] '
                            'fc1+fc2 prepared once per H3 block; fc2 uses stock fused SwiGLU INT8 semantics',
                            flush=True,
                        )
                        state['int8_block_mlp_weights_announced'] = True

                if _fc1_handle is not None and _fc2_handle is not None:
                    # v0.4.47: resident MLP must mirror stock Comfy H3 *kernel
                    # semantics*, not merely the mathematical expression.  Stock H3
                    # calls comfy.ops.linear_input_act(fc2, fc1(x), "swiglu").  For
                    # TensorWise INT8 this folds SwiGLU into the activation quantizer
                    # inside ck.int8_linear.  The previous resident path materialized
                    # BF16 SwiGLU first and then quantized it for fc2; that changes the
                    # quantization point and produced the measured ~5e-3 rel-RMS drift.
                    # Reuse the already-resident cast weights while dispatching through
                    # the exact same fused INT8 input_act contract as stock Comfy.
                    _ff = _int8_cached_linear(_fc1_handle, h_chunk)
                    chunk_out = _int8_cached_linear(
                        _fc2_handle, _ff, input_act='swiglu'
                    )
                    del _ff
                else:
                    chunk_out = block.mlp(h_chunk)

                # v0.3.61 real-shape parity gate.  Validate the resident path on
                # the complete first runtime chunk, because INT8 kernels/layouts
                # can be shape-sensitive and the old 4-token probe was not
                # representative.  One duplicate stock MLP call is paid once.
                if (
                    self.index == 0
                    and start == 0
                    and _fc1_handle is not None
                    and _fc2_handle is not None
                    and state.get('int8_cached_mlp_parity', 'unknown') == 'unknown'
                ):
                    _s = _c = _d = None
                    _stock_chunk = None
                    try:
                        _stock_chunk = block.mlp(h_chunk.clone()).detach()
                        _s = _stock_chunk.to(device='cpu', dtype=torch.float32)
                        _c = chunk_out.detach().to(device='cpu', dtype=torch.float32)
                        _d = _c - _s
                        _eps = 1.0e-12
                        _rms_ref = float(torch.sqrt(torch.mean(_s.square())).item()) if _s.numel() else 0.0
                        _rms_diff = float(torch.sqrt(torch.mean(_d.square())).item()) if _d.numel() else 0.0
                        _rel = _rms_diff / max(_eps, _rms_ref)
                        _den = float(torch.linalg.vector_norm(_s).item() * torch.linalg.vector_norm(_c).item()) if _s.numel() else 1.0
                        _cos = float(torch.dot(_s.flatten(), _c.flatten()).item() / max(_eps, _den)) if _s.numel() else 1.0
                        _finite = bool(torch.isfinite(_s).all() and torch.isfinite(_c).all() and torch.isfinite(_d).all())
                        _ok = _finite and _rel <= 2.0e-4 and _cos >= 0.9999
                    except Exception:
                        _ok, _rel, _cos = False, float('inf'), -1.0
                        _stock_chunk = None
                    state['int8_cached_mlp_parity'] = 'verified' if _ok else 'failed'
                    _lm_print(
                        '[MiniMaxH3 LongMedia][REAL-CHUNK MLP PARITY] '
                        f"{'PASS' if _ok else 'FAIL'} rows={int(h_chunk.shape[0])} "
                        f"rel_rms={_rel:.3e} cosine={_cos:.8f}; "
                        + ('resident stock-fused INT8 path enabled' if _ok else 'resident path DISABLED -> stock MLP'),
                        flush=True,
                    )
                    if not _ok:
                        if _stock_chunk is not None:
                            chunk_out = _stock_chunk.to(device=x.device, dtype=x.dtype)
                        _int8_release_block_linear(_fc2_handle)
                        _int8_release_block_linear(_fc1_handle)
                        _fc1_handle = _fc2_handle = None
                        _int8_backend = False
                        state['stock_mlp_math'] = True
                    for _tmp in (_s, _c, _d):
                        try:
                            del _tmp
                        except Exception:
                            pass
                    if _stock_chunk is not None:
                        del _stock_chunk

                del h_chunk

                if _v19_active:
                    for _offset in _v19_offsets:
                        if start <= int(_offset) < end:
                            _local = int(_offset) - start
                            _rows = min(4, end - int(_offset))
                            _v19_mlp_actual_parts.append(
                                chunk_out[_local:_local + _rows].detach().clone()
                            )

                # Consume the FFN result immediately into the corresponding residual
                # rows.  A chunk may cross modality/conditioning boundaries.
                for a, b, row in segments:
                    lo = max(start, int(a))
                    hi = min(end, int(b))
                    if lo >= hi:
                        continue
                    local_lo = lo - start
                    local_hi = hi - start
                    _shift_row, _scale_row, gate_row = _segment_mod_params(row, a, b, lo, hi)
                    x[lo:hi].addcmul_(chunk_out[local_lo:local_hi], gate_row)
                if _v19_active:
                    for _offset in _v19_offsets:
                        if start <= int(_offset) < end:
                            _rows = min(4, end - int(_offset))
                            _v19_final_actual_parts.append(
                                x[int(_offset):int(_offset) + _rows]
                                .detach().clone()
                            )
                del chunk_out

        finally:
            if _fc2_handle is not None:
                _int8_release_block_linear(_fc2_handle)
            if _fc1_handle is not None:
                _int8_release_block_linear(_fc1_handle)

        if _v19_active:
            if _v19_h_actual_parts:
                _v19_report(
                    state, 'NORM2-ADALN', _v19_h_reference,
                    torch.cat(_v19_h_actual_parts, dim=0),
                    offsets=_v19_offsets,
                )
            if _v19_mlp_actual_parts:
                _v19_report(
                    state, 'MLP-FC1-FC2', _v19_mlp_reference,
                    torch.cat(_v19_mlp_actual_parts, dim=0),
                    offsets=_v19_offsets,
                )
            if _v19_final_actual_parts:
                _v19_report(
                    state, 'MLP-GATE-RESIDUAL', _v19_final_reference,
                    torch.cat(_v19_final_actual_parts, dim=0),
                    offsets=_v19_offsets,
                )
            _v19_h_actual_parts.clear()
            _v19_mlp_actual_parts.clear()
            _v19_final_actual_parts.clear()
            del _v19_h_reference, _v19_mlp_reference, _v19_final_reference

        rows.clear()
        return x

    def _measure(self, name, fn, state, device):
        # Only block 0 / first forward is synchronized and measured. Every other
        # block follows the exact same chunked execution without profiler stalls.
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        before = _cuda_memory_snapshot()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        out = fn()
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        after = _cuda_memory_snapshot()
        peak_alloc = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        entry = {
            'stage': name,
            'allocated_before_mb': _mb(before['allocated']),
            'allocated_after_mb': _mb(after['allocated']),
            'reserved_before_mb': _mb(before['reserved']),
            'reserved_after_mb': _mb(after['reserved']),
            'driver_free_after_mb': _mb(after['driver_free']),
            'peak_allocated_mb': _mb(peak_alloc),
            'peak_reserved_mb': _mb(peak_reserved),
            'elapsed_ms': round(elapsed_ms, 1),
        }
        state['stages'].append(entry)
        state['highest_block_peak_allocated_mb'] = max(
            float(state.get('highest_block_peak_allocated_mb') or 0.0), float(entry['peak_allocated_mb']))
        state['highest_block_peak_reserved_mb'] = max(
            float(state.get('highest_block_peak_reserved_mb') or 0.0), float(entry['peak_reserved_mb']))
        if float(entry['peak_allocated_mb']) >= float(state.get('worst_stage_peak_allocated_mb') or -1.0):
            state['worst_stage'] = name
            state['worst_stage_peak_allocated_mb'] = float(entry['peak_allocated_mb'])
        _lm_print(
            '[MiniMaxH3 LongMedia] H3 block0 stage: '
            f"{name}, alloc {entry['allocated_before_mb']:.1f} -> {entry['allocated_after_mb']:.1f} MB, "
            f"peak {entry['peak_allocated_mb']:.1f} MB, reserved peak {entry['peak_reserved_mb']:.1f} MB, "
            f"free {entry['driver_free_after_mb']:.1f} MB, {entry['elapsed_ms']:.1f} ms",
            flush=True,
        )
        return out

    def __call__(self, args, extra_options):
        original_block = extra_options['original_block']
        state = self.state
        if self.index == 0:
            # V12-A/B2 is gated inside this helper. INT8 and W4A8 participate;
            # NVFP4 and floating-point backends do not mutate V11 state.
            _v12_begin_int8_sol_forward(state)
            state['sla_failed_blocks_this_forward'] = 0
        state['active_block_index'] = int(self.index)

        # V40 production baseline: keep the proven V39 execution path but remove
        # first-forward CUDA-synchronized per-block profiling overhead entirely.
        if self.index == 0:
            state['v30_block_metric_active'] = False
        _v30_metric = False
        if _v30_metric:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            _v30_block_t0 = time.perf_counter()
            _v30_attn_s = 0.0
            _v30_mlp_s = 0.0

        # AUTO attention selection happens before any QKV allocation:
        # `x` is assigned later in this wrapper, so use args['img'] directly.
        # This is still before any norm/attention/QKV allocation.
        _preblock_img = args['img']
        _preblock_tokens = _h3_sequence_tokens(_preblock_img)
        state['current_token_count'] = _preblock_tokens

        _requested_mode = str(
            state.get('requested_attention_mode', state.get('sol_mode', 'existing'))
        )
        if _requested_mode == 'auto':
            state['current_hidden_dim'] = int(_preblock_img.shape[-1])
            state['current_element_size'] = int(_preblock_img.element_size())
            _effective_mode, _auto_reason = _auto_select_h3_attention_mode(
                _preblock_tokens, state,
                hidden_dim=int(_preblock_img.shape[-1]),
                element_size=int(_preblock_img.element_size()),
            )
            state['sol_mode'] = _effective_mode
            state['auto_attention_selected_mode'] = _effective_mode
            state['auto_attention_reason'] = _auto_reason
            if not state.get('auto_attention_announced'):
                _lm_print(
                    '[MiniMaxH3 LongMedia][AUTO ATTENTION] '
                    f'{_auto_reason}; selected before QKV allocation; '
                    'full token semantics preserved',
                    flush=True,
                )
                state['auto_attention_announced'] = True
        else:
            state['sol_mode'] = _requested_mode

        # v0.4.35: explicit `existing` receives the same allocator-aware hard
        # safety guard.  This prevents the external SLA failure mode observed in
        # production: block_sparse_attention allocates o_s ~= Q, OOMs, then its
        # own exception path retries dense SDPA and OOMs again.  Route before QKV.
        if state.get('sol_mode') == 'existing':
            try:
                _est_b, _one_b = _estimate_existing_attention_peak_bytes(
                    _preblock_tokens, int(_preblock_img.shape[-1]),
                    int(_preblock_img.element_size()),
                    external_sla=bool(state.get('external_sla_detected', False)),
                )
                _free_b, _total_b = torch.cuda.mem_get_info(torch.cuda.current_device())
                _reserve_mb = max(768, int(state.get('vram_activation_reserve_mb', 2048) or 2048))
                _reserve_b = min(int(_reserve_mb * 1024**2), int(_total_b * 0.18))
                _usable_b = max(0, int(_free_b) - _reserve_b)
                _geom_unsafe = bool(
                    bool(state.get('external_sla_detected', False))
                    and not bool(state.get('external_sla_direct_fastpath', False))
                    and int(_total_b) <= int(18.5 * 1024**3)
                    and int(_one_b) >= int(768 * 1024**2)
                )
                if bool(state.get('external_sla_direct_fastpath', False)):
                    _one_b = (
                        int(_preblock_tokens) * int(_preblock_img.shape[-1])
                        * int(_preblock_img.element_size())
                    )
                    _est_b = int(3 * _one_b + max(256 * 1024**2, int(_one_b * 0.15)))
                    _usable_b = max(0, int(_free_b) - 512 * 1024**2)
                    if int(_total_b) <= int(18.5 * 1024**3) and _one_b <= int(1.60 * 1024**3):
                        _est_b = min(_est_b, _usable_b)
                _unsafe = (_est_b > _usable_b) or _geom_unsafe
            except Exception:
                _unsafe = False
                _est_b = _one_b = _free_b = _usable_b = 0
            if _unsafe:
                _explicit_existing = (
                    str(state.get('requested_attention_mode', '')).lower() == 'existing'
                )
                if _explicit_existing and _should_use_h3_existing_lowmem(state, _preblock_tokens):
                    # v0.4.80: honor the user's explicit EXISTING backend.
                    # Use the restored no-V-clone H3 wrapper instead of silently
                    # changing the attention family to SOL.
                    state['existing_lowmem_forced'] = True
                    if not state.get('existing_lowmem_preflight_announced'):
                        _lm_print(
                            '[MiniMaxH3 LongMedia][EXISTING LOWMEM PREFLIGHT] '
                            f'tokens={_preblock_tokens}; peak_est={_est_b/1024**2:.0f}MB; '
                            f'driver_usable={_usable_b/1024**2:.0f}MB; '
                            'explicit existing preserved; upstream H3 V-clone bypass enabled; '
                            'optimized_attention backend remains user-selected',
                            flush=True,
                        )
                        state['existing_lowmem_preflight_announced'] = True
                else:
                    state['sol_mode'] = 'sol'
                    state['v434_attention_safety_fallback'] = True
                    # AUTO keeps the existing safety behavior.
                    if str(state.get('requested_attention_mode', '')).lower() == 'auto':
                        state['auto_attention_selected_mode'] = 'sol'
                    if not state.get('v434_attention_safety_announced'):
                        _lm_print(
                            '[MiniMaxH3 LongMedia][VRAM PREFLIGHT] '
                            f'existing attention rejected BEFORE QKV: tokens={_preblock_tokens}, '
                            f'single_tensor={_one_b/1024**2:.0f}MB, '
                            f'peak_est={_est_b/1024**2:.0f}MB, '
                            f'driver_usable={_usable_b/1024**2:.0f}MB; '
                            'route=embedded SOL bounded-QKV; no post-OOM dense retry',
                            flush=True,
                        )
                        state['v434_attention_safety_announced'] = True

        block = self._extract_block(original_block)
        if block is None:
            if self.index == 0 and not state.get('fallback_reason'):
                state['fallback_reason'] = 'could not extract DiTBlock from original_block closure'
                _lm_print('[MiniMaxH3 LongMedia] Low-VRAM MLP fallback: DiTBlock closure not found', flush=True)
            return original_block(args)

        x = args['img']
        t_emb = args['t_emb']
        mod_segments = args['mod_segments']
        rope_freqs = args['rope_freqs']
        transformer_options = args['transformer_options']

        # TEST build: block-0 step-boundary timing profiler disabled.
        # Removes profiling-only CUDA synchronization; SAFE guards are untouched.

        # TEST build: deep first-forward H3 stage profiler disabled.
        trace_this = False
        if trace_this:
            state['forward_count'] = 1
            state['first_forward_started'] = True
            state['first_forward_started_at'] = time.time()
            _lm_print('[MiniMaxH3 LongMedia] H3 block0 deep ATTENTION trace + MLP chunking: first forward started', flush=True)
            device = torch.cuda.current_device()
            measure = lambda name, fn: self._measure(name, fn, state, device)
        else:
            measure = lambda name, fn: fn()

        try:
            self._adaptive_memory_governor('block_start')
            vals = measure('adaln_proj', lambda: block.adaln_proj(t_emb))
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = vals

            _v19_block_active = _v19_selected_block(state, self.index)
            if _v19_block_active:
                _lm_print(
                    '[MiniMaxH3 LongMedia][V22 STAGE A/B ENTER] '
                    f'block={self.index}, generation={state.get("v21_stage_ab_generation", 0)}',
                    flush=True,
                )
            _v19_offsets = (
                _v19_probe_offsets(_preblock_tokens, 8192)
                if _v19_block_active else []
            )
            _v19_norm1_reference = None
            if _v19_block_active:
                _v19_norm1_parts = []
                for _offset in _v19_offsets:
                    _probe_end = min(_preblock_tokens, int(_offset) + 4)
                    _href = block.norm1(x[int(_offset):_probe_end])
                    for a, b, row in mod_segments:
                        lo = max(int(_offset), int(a))
                        hi = min(_probe_end, int(b))
                        if lo >= hi:
                            continue
                        local_lo = lo - int(_offset)
                        local_hi = hi - int(_offset)
                        _seg = _href[local_lo:local_hi]
                        if torch.is_tensor(row) and int(row.numel()) != 1:
                            _flat = row.reshape(-1)
                            _seg_len = int(b) - int(a)
                            if int(_flat.numel()) == _seg_len:
                                _ids = _flat[lo-int(a):hi-int(a)]
                            elif int(_flat.numel()) == _preblock_tokens:
                                _ids = _flat[lo:hi]
                            else:
                                raise RuntimeError(
                                    '[0.4.20 NORM1 DIAG MOD-ROW COMPAT] unsupported row layout '
                                    f'{tuple(row.shape)} for segment {int(a)}:{int(b)}'
                                )
                            _ids = _ids.to(device=scale_msa.device, dtype=torch.long)
                            _sc = scale_msa.index_select(0, _ids).to(_href.dtype)
                            _sh = shift_msa.index_select(0, _ids).to(_href.dtype)
                        else:
                            _ri = int(row.reshape(-1)[0].item()) if torch.is_tensor(row) else int(row)
                            _sc = scale_msa[_ri].to(_href.dtype)
                            _sh = shift_msa[_ri].to(_href.dtype)
                        _seg.mul_(1.0 + _sc).add_(_sh)
                    _v19_norm1_parts.append(_href.detach())
                _v19_norm1_reference = torch.cat(_v19_norm1_parts, dim=0)
                del _v19_norm1_parts

            h = measure(
                'norm1_mod',
                lambda: self._mod_scale_shift(block.norm1(x), shift_msa, scale_msa, mod_segments),
            )
            if _v19_block_active:
                _v19_report(
                    state, 'NORM1-ADALN', _v19_norm1_reference,
                    torch.cat([
                        h[int(offset):min(_preblock_tokens, int(offset) + 4)]
                        for offset in _v19_offsets
                    ], dim=0),
                    offsets=_v19_offsets,
                )
                del _v19_norm1_reference
            self._late_block_hard_guard('pre_attention')
            if _v30_metric:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                _v30_attn_t0 = time.perf_counter()
            sol_mode = state.get('sol_mode', 'existing')
            if bool(state.get('fasth3_vsa_active', False)) or bool(state.get('fastvideo_vsa_active', False)):
                try:
                    attn_out = _run_h3_fasth3_vsa_attention(
                        block.attn, h, rope_freqs, transformer_options, state
                    )
                except _FastH3VSANotPlainT2VA as _vsa_layout_exc:
                    # Both published Preview-v1 VSA students are T2AV-trained.
                    # LongMedia continuation/reference spans use exact dense H3
                    # attention rather than applying an invalid sparse geometry.
                    _is_fastvideo = bool(state.get('fastvideo_vsa_active', False))
                    _family = 'FastVideo VSA' if _is_fastvideo else 'FastH3 VSA'
                    _flag = 'fastvideo_vsa_dense_fallback_announced' if _is_fastvideo else 'fasth3_vsa_dense_fallback_announced'
                    if not state.get(_flag):
                        state[_flag] = True
                        _lm_print(
                            f'[MiniMaxH3 LongMedia][{_family}] dense fallback for non-plain T2VA layout: '
                            f'{_vsa_layout_exc}; model weights + trained 4-step schedule retained',
                            flush=True,
                        )
                    attn_out = _run_h3_existing_attention_lowmem(
                        block.attn, h, rope_freqs, transformer_options, state
                    )
            elif sol_mode != 'existing':
                attn_out = _run_h3_sol_attention(
                    block.attn, h, rope_freqs, transformer_options, state,
                    measure=measure if trace_this else None,
                )
            elif bool(state.get('external_sla_direct_fastpath', False)):
                attn_out = None
                _sla_oom = None
                for _sla_attempt in range(2):
                    try:
                        attn_out = _execute_h3_sla_zero_copy(
                            block.attn, h, rope_freqs, transformer_options, state,
                            measure=measure if trace_this else None,
                        )
                        if _sla_attempt:
                            _lm_print(
                                '[MiniMaxH3 LongMedia][SLA OOM RECOVERY] '
                                f'block={self.index} recovered after allocator trim; '
                                f'phase={state.get("sla_zero_copy_phase", "unknown")}; '
                                'SLA remains active',
                                flush=True,
                            )
                        break
                    except torch.cuda.OutOfMemoryError as _exc:
                        _sla_oom = _exc
                        _phase = str(state.get('sla_zero_copy_phase', 'unknown'))
                        state['sla_zero_copy_oom_count'] = int(state.get('sla_zero_copy_oom_count', 0) or 0) + 1
                        try:
                            import comfy.model_management as _mm
                            _mm.soft_empty_cache()
                        except Exception:
                            torch.cuda.empty_cache()
                        if _sla_attempt == 0:
                            _lm_print(
                                '[MiniMaxH3 LongMedia][SLA OOM RECOVERY] '
                                f'block={self.index} phase={_phase}; trimmed allocator cache; '
                                'retrying SAME SLA geometry once (no sticky Sol switch)',
                                flush=True,
                            )
                            continue

                if attn_out is None:
                    # A single transient block must not poison the entire denoise
                    # forward.  Fall back only for this block, release the Sol
                    # workspace immediately, and probe SLA again on the next
                    # block.  Two distinct block failures in one forward are the
                    # circuit breaker: after that, stay on bounded Sol for safety.
                    _failed_blocks = int(state.get('sla_failed_blocks_this_forward', 0) or 0) + 1
                    state['sla_failed_blocks_this_forward'] = _failed_blocks
                    _phase = str(state.get('sla_zero_copy_phase', 'unknown'))
                    if _failed_blocks >= 2:
                        state['sol_mode'] = 'sol'
                        state['auto_attention_selected_mode'] = 'sol'
                        _scope = 'sticky_after_second_failed_block'
                    else:
                        _scope = 'current_block_only'
                    _lm_print(
                        '[MiniMaxH3 LongMedia][SLA FALLBACK] '
                        f'block={self.index} phase={_phase} retries_exhausted=2; '
                        f'fallback={_scope}; next_block_sla={_failed_blocks < 2}',
                        flush=True,
                    )
                    # v0.4.46: force the bounded embedded-Sol implementation for
                    # this emergency block.  Calling _run_h3_sol_attention while
                    # state.sol_mode == "existing" returns to stock H3 attention,
                    # whose v.clone() allocates another ~1.5 GiB at S~=112k and
                    # immediately OOMs.  Temporarily select Sol, then restore the
                    # user's/AUTO mode if the circuit breaker is not sticky.
                    _saved_sol_mode = str(state.get('sol_mode', 'existing'))
                    if _failed_blocks < 2:
                        state['sol_mode'] = 'sol'
                    try:
                        attn_out = _run_h3_sol_attention(
                            block.attn, h, rope_freqs, transformer_options, state,
                            measure=measure if trace_this else None,
                        )
                    finally:
                        if _failed_blocks < 2:
                            state['sol_mode'] = _saved_sol_mode
                    if _failed_blocks < 2:
                        _v12_release_int8_sol_forward(state, block_index=self.index)
            elif trace_this:
                attn_out = self._trace_attention(
                    block.attn, h, rope_freqs, transformer_options, measure
                )
            elif _should_use_h3_existing_lowmem(state, _preblock_tokens):
                _existing_exc = None
                attn_out = None
                for _existing_attempt in range(2):
                    try:
                        attn_out = _run_h3_existing_attention_lowmem(
                            block.attn, h, rope_freqs, transformer_options, state
                        )
                        if not state.get('existing_lowmem_announced'):
                            _lm_print(
                                '[MiniMaxH3 LongMedia][EXISTING LOWMEM] '
                                f'active: tokens={_preblock_tokens}; '
                                'upstream v.clone bypassed; '
                                'ComfyUI optimized_attention backend preserved',
                                flush=True,
                            )
                            state['existing_lowmem_announced'] = True
                        break
                    except torch.cuda.OutOfMemoryError as _exc:
                        _existing_exc = _exc
                        if _existing_attempt == 0:
                            # A backend/version-specific transient can still exceed
                            # the first budget.  Retry once with another 1.5 GiB of
                            # DynamicVRAM headroom.  Giant CK workloads remain on the
                            # exact streamed-EXISTING path; SOL is never substituted.
                            state['existing_workspace_retry_extra_mb'] = 1536
                            try:
                                import comfy.model_management as _mm
                                _mm.soft_empty_cache()
                            except Exception:
                                torch.cuda.empty_cache()
                            _lm_print(
                                '[MiniMaxH3 LongMedia][EXISTING LOWMEM OOM RECOVERY] '
                                f'block={self.index}; retry budget +1536MB; '
                                'same EXISTING backend/math retained; no SOL substitution',
                                flush=True,
                            )
                            continue
                if attn_out is None:
                    raise _existing_exc
            else:
                attn_out = block.attn(
                    h, rope_freqs=rope_freqs, transformer_options=transformer_options
                )
            if _v30_metric:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                _v30_attn_s = time.perf_counter() - _v30_attn_t0

            # The final block has consumed compressed K/V and kc statistics.
            # Release references before its FFN/output head so the next denoise
            # forward is forced to allocate a clean workspace.
            if self.index == int(state.get('last_patched_block_index', -1)):
                _v12_release_int8_sol_forward(
                    state, block_index=self.index
                )
            _v19_attn_reference = None
            if _v19_block_active:
                _v19_attn_parts = []
                for _offset in _v19_offsets:
                    _probe_end = min(_preblock_tokens, int(_offset) + 4)
                    _expected = x[int(_offset):_probe_end].detach().clone()
                    _other = attn_out[int(_offset):_probe_end]
                    for a, b, row in mod_segments:
                        lo = max(int(_offset), int(a))
                        hi = min(_probe_end, int(b))
                        if lo >= hi:
                            continue
                        local_lo = lo - int(_offset)
                        local_hi = hi - int(_offset)
                        _seg_expected = _expected[local_lo:local_hi]
                        _seg_other = _other[local_lo:local_hi]
                        if torch.is_tensor(row) and int(row.numel()) != 1:
                            _flat = row.reshape(-1)
                            _seg_len = int(b) - int(a)
                            if int(_flat.numel()) == _seg_len:
                                _ids = _flat[lo-int(a):hi-int(a)]
                            elif int(_flat.numel()) == _preblock_tokens:
                                _ids = _flat[lo:hi]
                            else:
                                raise RuntimeError(
                                    '[0.4.20 GATE DIAG MOD-ROW COMPAT] unsupported row layout '
                                    f'{tuple(row.shape)} for segment {int(a)}:{int(b)}'
                                )
                            _ids = _ids.to(device=gate_msa.device, dtype=torch.long)
                            _gate = gate_msa.index_select(0, _ids).to(_expected.dtype)
                        else:
                            _ri = int(row.reshape(-1)[0].item()) if torch.is_tensor(row) else int(row)
                            _gate = gate_msa[_ri].to(_expected.dtype)
                        _seg_expected.addcmul_(_seg_other, _gate)
                    _v19_attn_parts.append(_expected)
                _v19_attn_reference = torch.cat(_v19_attn_parts, dim=0)
                del _v19_attn_parts
            x = measure(
                'attention_gate_residual',
                lambda: self._mod_gate(x, gate_msa, attn_out, mod_segments),
            )

            # v0.3.53 ultra-low-VRAM stage barrier.  Stock H3 immediately
            # replaces the full attention-normalized activation with the FFN
            # activation.  Our patched block used to keep the full `h` tensor
            # alive while entering chunked norm2/MLP, which raises the residency
            # peak exactly when the next block weights/cast buffers are needed.
            # On out-of-core 32+ GB models this can abort CUDA before the first
            # FFN kernel even launches.  Retire the attention stage explicitly.
            if bool(state.get('ultra_stage_barrier_required', False)):
                try:
                    torch.cuda.synchronize(device)
                except Exception:
                    pass
                try:
                    del h
                except Exception:
                    pass
                # attn_out is consumed by the residual above; free it before
                # Comfy has to materialize FFN weights.
                try:
                    del attn_out
                except Exception:
                    pass
                try:
                    import comfy.model_management as _mm
                    _mm.soft_empty_cache()
                except Exception:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                if not state.get('v353_stage_barrier_announced', False):
                    _lm_print(
                        '[MiniMaxH3 LongMedia][ULTRA STAGE BARRIER] '
                        'attention activation retired before FFN; CUDA synchronized; cache trimmed',
                        flush=True,
                    )
                    state['v353_stage_barrier_announced'] = True

            if _v19_block_active:
                _v19_report(
                    state, 'ATTENTION-GATE-RESIDUAL', _v19_attn_reference,
                    torch.cat([
                        x[int(offset):min(_preblock_tokens, int(offset) + 4)]
                        for offset in _v19_offsets
                    ], dim=0),
                    offsets=_v19_offsets,
                )
                del _v19_attn_reference
            try:
                del attn_out
            except Exception:
                pass
            # `h` is no longer needed after attention on any path. Releasing it
            # early also lowers normal/low-VRAM peaks without changing math.
            try:
                del h
            except Exception:
                pass

            self._late_block_hard_guard('pre_ffn')
            if _v30_metric:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                _v30_mlp_t0 = time.perf_counter()
            x = self._chunk_norm2_mlp_gate_residual(
                block, x, shift_mlp, scale_mlp, gate_mlp, mod_segments
            )
            if _v30_metric:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                _v30_mlp_s = time.perf_counter() - _v30_mlp_t0
            if _v19_block_active:
                _lm_print(
                    '[MiniMaxH3 LongMedia][V22 STAGE A/B EXIT] '
                    f'block={self.index}', flush=True,
                )
                _targets = tuple(int(v) for v in state.get('v21_stage_ab_targets', ()))
                if _targets and self.index == _targets[-1]:
                    state['v21_stage_ab_completed'] = True
                    state['v21_stage_ab_armed'] = False
                    _lm_print(
                        '[MiniMaxH3 LongMedia][V22 STAGE A/B COMPLETE] '
                        f'targets={list(_targets)}', flush=True,
                    )

            if trace_this:
                state['blocks'].append({
                    'block': 0,
                    'peak_allocated_mb': state.get('highest_block_peak_allocated_mb', 0.0),
                    'peak_reserved_mb': state.get('highest_block_peak_reserved_mb', 0.0),
                    'deep_trace': True,
                    'mlp_chunked': True,
                })
                _lm_print(
                    '[MiniMaxH3 LongMedia] H3 block0 attention+MLP trace summary: '
                    f"worst stage {state.get('worst_stage')}, "
                    f"peak allocated {state.get('highest_block_peak_allocated_mb', 0.0):.1f} MB",
                    flush=True,
                )
            # Drop modulation tables before asking the allocator to return dead
            # pages. They are block-local and no longer needed after the second
            # residual update.
            del vals, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
            self._inter_block_pressure_guard()

            # Leave real driver-free headroom for the next Comfy INT8 prefetch.
            self._int8_prefetch_guard()
            self._adaptive_memory_governor('block_end')

            if _v30_metric:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                _v30_total_s = time.perf_counter() - _v30_block_t0
                _v30_cum_s = time.perf_counter() - float(state.get('v30_block_metric_forward_t0', _v30_block_t0))
                try:
                    _free_b, _total_b = torch.cuda.mem_get_info()
                    _alloc_b = torch.cuda.memory_allocated()
                    _reserved_b = torch.cuda.memory_reserved()
                    _free_mb = _free_b / (1024 * 1024)
                    _alloc_mb = _alloc_b / (1024 * 1024)
                    _reserved_mb = _reserved_b / (1024 * 1024)
                except Exception:
                    _free_mb = _alloc_mb = _reserved_mb = float('nan')
                _mlp_chunk = int(state.get('auto_mlp_chunk_last') or state.get('chunk_tokens') or self.chunk_tokens)
                _other_s = max(0.0, _v30_total_s - _v30_attn_s - _v30_mlp_s)
                _lm_print(
                    '[MiniMaxH3 LongMedia][V35 BLOCK METRIC] '
                    f'block={self.index:02d}/49 total={_v30_total_s:.2f}s '
                    f'attn={_v30_attn_s:.2f}s kvpass={float(state.get("v33_last_kvpass_s", 0.0)):.2f}s '
                    f'querypass={float(state.get("v33_last_querypass_s", 0.0)):.2f}s '
                    f'qproj={float(state.get("v34_last_qproj_s", 0.0)):.2f}s rope={float(state.get("v34_last_rope_s", 0.0)):.2f}s '
                    f'sol={float(state.get("v34_last_sol_s", 0.0)):.2f}s outproj={float(state.get("v34_last_outproj_s", 0.0)):.2f}s copy={float(state.get("v34_last_copy_s", 0.0)):.2f}s '
                    f'mlp={_v30_mlp_s:.2f}s other={_other_s:.2f}s '
                    f'cum={_v30_cum_s:.2f}s qkv_chunk={int(state.get("sol_qkv_chunk_tokens", 0) or 0)} mlp_chunk={_mlp_chunk} '
                    f'alloc={_alloc_mb:.0f}MB reserved={_reserved_mb:.0f}MB driver_free={_free_mb:.0f}MB',
                    flush=True,
                )
                if self.index == int(state.get('last_patched_block_index', 49)):
                    state['v30_block_metric_active'] = False
                    _lm_print(
                        '[MiniMaxH3 LongMedia][V35 BLOCK METRIC] first-forward profiling COMPLETE; '
                        f'total_blocks_time={_v30_cum_s:.2f}s',
                        flush=True,
                    )
            return {'img': x}
        except Exception as exc:
            message = str(exc).lower()
            is_oom = isinstance(exc, getattr(torch, 'OutOfMemoryError', RuntimeError)) or 'out of memory' in message
            if is_oom:
                state['oom'] = True
                state['oom_block'] = self.index
                state['oom_stage'] = state.get('stages', [])[-1]['stage'] if state.get('stages') else 'unknown'
                state['oom_message'] = str(exc)[:2000]
                _lm_print(
                    f"[MiniMaxH3 LongMedia] H3 CUDA OOM in block {self.index} near {state.get('oom_stage')}: {state['oom_message']}",
                    flush=True,
                )
            raise



def _v24_final_report(state, stage, reference, candidate, *, stream, offsets):
    """Robust sampled A/B for the H3 final output layer (V24)."""
    key = f"{stream}:{stage}"
    done = state.setdefault('v24_final_ab_done', set())
    if key in done:
        return
    try:
        ref = reference.detach().to(device='cpu', dtype=torch.float32)
        got = candidate.detach().to(device='cpu', dtype=torch.float32)
        if tuple(ref.shape) != tuple(got.shape):
            _lm_print(
                '[MiniMaxH3 LongMedia][V24 FINAL-LAYER A/B] '
                f'stream={stream}, stage={stage}, verdict=SHAPE-MISMATCH, '
                f'reference={tuple(ref.shape)}, candidate={tuple(got.shape)}',
                flush=True,
            )
            return
        rf = ref.flatten()
        gf = got.flatten()
        diff = gf - rf
        finite = bool(torch.isfinite(rf).all().item() and torch.isfinite(gf).all().item() and torch.isfinite(diff).all().item())
        exact = bool(torch.equal(rf, gf))
        mismatches = int(torch.count_nonzero(diff).item())
        if exact:
            rel_rms = 0.0; cosine = 1.0; mean_abs = 0.0; max_abs = 0.0
        else:
            eps = 1.0e-30
            r64 = rf.to(torch.float64); g64 = gf.to(torch.float64); d64 = g64-r64
            rms_ref = float(torch.sqrt(torch.mean(r64.square())).item())
            rms_diff = float(torch.sqrt(torch.mean(d64.square())).item())
            rel_rms = rms_diff / max(eps, rms_ref)
            nr = float(torch.linalg.vector_norm(r64).item()); ng = float(torch.linalg.vector_norm(g64).item())
            denom = nr * ng
            cosine = 1.0 if denom <= eps and rms_diff <= eps else (0.0 if denom <= eps else float(torch.dot(r64,g64).item()/denom))
            cosine = max(-1.0, min(1.0, cosine))
            mean_abs = float(d64.abs().mean().item()); max_abs = float(d64.abs().max().item())
            del r64, g64, d64
        verdict = 'MATCH' if finite and (exact or (rel_rms <= 1.0e-5 and cosine >= 0.99999)) else 'DIVERGED'
        _lm_print(
            '[MiniMaxH3 LongMedia][V24 FINAL-LAYER A/B] '
            f'stream={stream}, stage={stage}, verdict={verdict}, offsets={list(offsets)}, rows={int(ref.shape[0])}, '
            f'exact={exact}, mismatches={mismatches}, rel_rms={rel_rms:.8e}, cosine={cosine:.10f}, '
            f'mean_abs={mean_abs:.8e}, max_abs={max_abs:.8e}, finite={finite}',
            flush=True,
        )
    except Exception as exc:
        _lm_print('[MiniMaxH3 LongMedia][V24 FINAL-LAYER A/B] '
              f'stream={stream}, stage={stage}, diagnostic failed: {type(exc).__name__}: {exc}', flush=True)
    finally:
        done.add(key)


def _v24_probe_local_offsets(n, max_rows=16):
    """Small deterministic row set spanning a stream without materializing full hidden tensors."""
    n = int(n)
    if n <= 0:
        return []
    anchors = [0, min(n-1, 1), n//4, n//2, (3*n)//4, max(0,n-2), n-1]
    # Fill to at most max_rows with evenly spaced rows.
    if n > 1:
        for i in range(max_rows):
            anchors.append((i*(n-1))//max(1,max_rows-1))
    return sorted(set(int(v) for v in anchors if 0 <= int(v) < n))[:max_rows]


def _install_h3_final_output_streaming(model_patcher, state, chunk_tokens=24576):
    """Patch MiniMax H3 FinalLayer so FP32 output-head inputs never exist full-size.

    Stock H3 FinalLayer normalizes/modulates the complete target video stream and
    then casts that [video_tokens, hidden] tensor to FP32 before video_out.  On
    long clips this FP32 island is several GiB (about 4.5 GiB at 225k x 5376).
    Norm/modulation and the output Linear are token-local, so process the target
    streams in token chunks and write only the small projected outputs into their
    final buffers.
    """
    try:
        base_model = getattr(model_patcher, 'model', None)
        diffusion = getattr(base_model, 'diffusion_model', None)
        final_layer = getattr(diffusion, 'final_layer', None)
        if final_layer is None:
            state['final_output_streaming_error'] = 'MiniMaxH3 final_layer not found'
            return False

        original = getattr(final_layer, '_latentlab_final_output_original_forward', None)
        if original is None:
            original = final_layer.forward
            final_layer._latentlab_final_output_original_forward = original

        # The model object is shared by ModelPatcher clones. Keep mutable runtime
        # settings on the module so a new prompt updates the already-installed wrapper.
        final_layer._latentlab_final_output_state = state
        final_layer._latentlab_final_output_chunk_tokens = max(256, int(chunk_tokens or 24576))

        if (
            getattr(final_layer, '_latentlab_final_output_streaming_installed', False)
            and not globals().get('_LONGMEDIA_HOT_RELOAD_BYPASS', False)
        ):
            return True

        import types

        def _streamed_forward(layer, x, t_emb, video_seg, audio_seg, sigma=None, sample_sigmas=None, shifts=None):
            # ComfyUI 2026 PDD compatibility: FinalLayer gained sigma, the sampler
            # sigma schedule, and per-stream flow shifts. Standard H3 checkpoints
            # still expose one output head (n=1); PDD checkpoints patch that weight
            # into an n-head bank.  Keep both contracts in one streaming wrapper.
            st = getattr(layer, '_latentlab_final_output_state', {}) or {}
            chunk = max(256, int(getattr(layer, '_latentlab_final_output_chunk_tokens', 24576)))
            shift, scale = layer.adaln_proj(t_emb)
            va, vb, vrow = video_seg
            aa, ab, arow = audio_seg

            def _row_mod_params(row, start, stop, local=0, end=None):
                """Return AdaLN scale/shift for a scalar row or per-token row tensor.

                Newer Comfy H3 layouts may carry one modulation-row id per token.
                Keep the streamed final head mathematically equivalent to stock H3
                instead of coercing that tensor through int().
                """
                start = int(start)
                stop = int(stop)
                local = int(local)
                if end is None:
                    end = stop - start
                end = int(end)
                if torch.is_tensor(row):
                    flat = row.reshape(-1)
                    if int(flat.numel()) == 1:
                        idx = int(flat[0].item())
                        return scale[idx], shift[idx]
                    seg_n = max(0, stop - start)
                    if int(flat.numel()) == seg_n:
                        ids = flat[local:end]
                    elif int(flat.numel()) == int(x.shape[0]):
                        ids = flat[start + local:start + end]
                    else:
                        raise RuntimeError(
                            '[0.4.20 FINAL-HEAD MOD-ROW COMPAT] unsupported modulation-row layout: '
                            f'row_shape={tuple(row.shape)}, row_numel={int(flat.numel())}, '
                            f'segment={start}:{stop} ({seg_n} tokens), total_tokens={int(x.shape[0])}.'
                        )
                    ids = ids.to(device=scale.device, dtype=torch.long)
                    return scale.index_select(0, ids), shift.index_select(0, ids)
                idx = int(row)
                return scale[idx], shift[idx]

            def _head_bank_count(head):
                out_features = int(getattr(head, 'out_features', 0) or 0)
                weight = getattr(head, 'weight', None)
                if out_features <= 0 or weight is None or int(weight.shape[0]) % out_features != 0:
                    return 1
                return max(1, int(weight.shape[0]) // out_features)

            def _pdd_interval(head_bank_count):
                if int(head_bank_count) <= 1:
                    return None
                if sigma is None or sample_sigmas is None or shifts is None:
                    raise ValueError("MiniMax H3 PDD heads need the sampler's sigma schedule")
                sigmas = sample_sigmas
                if not torch.is_tensor(sigmas):
                    sigmas = torch.as_tensor(sigmas, device=x.device, dtype=torch.float32)
                if int(sigmas.numel()) < 1:
                    raise ValueError("MiniMax H3 PDD heads received an empty sampler sigma schedule")
                sigma_t = sigma if torch.is_tensor(sigma) else torch.as_tensor(sigma, device=sigmas.device, dtype=sigmas.dtype)
                sigma_t = sigma_t.to(device=sigmas.device, dtype=sigmas.dtype).reshape(-1)[0]
                i = int((sigmas - sigma_t).abs().argmin().item())
                sigma_next = sigmas[min(i + 1, int(sigmas.shape[0]) - 1)]
                return sigma_t, sigma_next

            def _pdd_effective_params(head, h32, flow_shift, interval):
                """Return the exact stock-ComfyUI PDD effective weight/bias.

                PDD stores row block 0 as the full output head and later blocks as
                offsets. The active sigma interval consumes the dt-weighted mean of
                the offset heads it spans.  We form that small effective output head
                once per streamed target segment and apply it chunk-by-chunk.
                """
                import comfy.ops
                import torch.nn.functional as F
                from comfy.ldm.minimax.model import time_shift_sigma

                bank_n = _head_bank_count(head)
                if bank_n <= 1:
                    return None
                sigma_t, sigma_next = interval
                start, stop = (
                    round(float(1.0 - time_shift_sigma(s, float(flow_shift), 1.0)) * bank_n)
                    for s in (sigma_t, sigma_next)
                )
                start = min(int(start), bank_n - 1)
                stop = max(int(stop), start + 1)

                grid = torch.linspace(1.0, 0.0, bank_n + 1, dtype=torch.float64)
                shifted = 1.0 - float(flow_shift) * grid / (1.0 + (float(flow_shift) - 1.0) * grid)
                dt = shifted.diff()[start:stop]
                # CastBiasWeightContext is the same path stock ComfyUI uses, so
                # quantized/LoRA-patched output weights retain native casting rules.
                ctx = comfy.ops.CastBiasWeightContext(head, h32, offloadable=True)
                weight, bias = ctx.__enter__()
                try:
                    rows = weight.reshape(bank_n, -1, weight.shape[1])
                    brows = bias.reshape(bank_n, -1) if bias is not None else None
                    w = (dt / dt.sum()).to(device=weight.device, dtype=weight.dtype)
                    first = max(start, 1)
                    effective_w = rows[0]
                    if first < stop:
                        effective_w = effective_w + torch.einsum('n,noi->oi', w[first - start:], rows[first:stop])
                    effective_b = None
                    if brows is not None:
                        effective_b = brows[0]
                        if first < stop:
                            effective_b = effective_b + torch.einsum('n,no->o', w[first - start:], brows[first:stop])
                    # Clone outside the bank views so the context may release/offload
                    # the full patched head before token streaming begins.
                    effective_w = effective_w.contiguous()
                    if effective_b is not None:
                        effective_b = effective_b.contiguous()
                    return effective_w, effective_b, ctx, F
                except Exception:
                    ctx.__exit__(None, None, None)
                    raise

            def _head_segment(start, stop, row, head, label, flow_shift):
                n = int(stop) - int(start)
                head_bank_n = _head_bank_count(head)
                pdd_interval = _pdd_interval(head_bank_n)
                pdd_params = None
                if n > 0 and head_bank_n > 1:
                    # A one-row fp32 probe is enough to establish the exact patched
                    # weight dtype/device for CastBiasWeightContext.
                    probe_h = layer.norm(x[int(start):int(start) + 1])
                    sc0, sh0 = _row_mod_params(row, start, stop, 0, 1)
                    probe_h = (probe_h * (1.0 + sc0.to(probe_h.dtype)) + sh0.to(probe_h.dtype)).to(torch.float32)
                    pdd_params = _pdd_effective_params(head, probe_h, flow_shift, pdd_interval)
                    del probe_h

                def _project(h32):
                    if pdd_params is None:
                        return head(h32)
                    effective_w, effective_b, _ctx, F = pdd_params
                    return F.linear(h32, effective_w, effective_b)

                if n <= 0:
                    # Defensive fallback; target streams are expected non-empty.
                    sc, sh = _row_mod_params(row, start, stop, 0, n)
                    return _project((layer.norm(x[int(start):int(stop)]) * (1.0 + sc) + sh).to(torch.float32))

                # V24: sampled reference rows are evaluated with the stock mathematical
                # expression while the real generation still uses the streamed path.
                # This stays tiny (<=16 rows) and therefore does not recreate the
                # multi-GiB FP32 hidden tensor that final-output streaming was built to avoid.
                probe_local = _v24_probe_local_offsets(n) if not st.get(f'v24_final_{label}_captured', False) else []
                probe_abs = [int(start) + q for q in probe_local]
                candidate_hidden = {}
                candidate_output = {}

                out_features = getattr(head, 'out_features', None)
                if out_features is None:
                    w = getattr(head, 'weight', None)
                    out_features = int(w.shape[0]) if w is not None else None
                if out_features is None:
                    # Unknown custom Linear implementation: preserve correctness.
                    sc, sh = _row_mod_params(row, start, stop, 0, n)
                    return head((layer.norm(x[int(start):int(stop)]) * (1.0 + sc) + sh).to(torch.float32))

                out = torch.empty((n, int(out_features)), device=x.device, dtype=torch.float32)
                chunks = (n + chunk - 1) // chunk
                for local in range(0, n, chunk):
                    end = min(n, local + chunk)
                    src = x[int(start) + local:int(start) + end]
                    h = layer.norm(src)
                    sc, sh = _row_mod_params(row, start, stop, local, end)
                    h.mul_(1.0 + sc.to(h.dtype)).add_(sh.to(h.dtype))
                    if probe_local:
                        for q_local, q_abs in zip(probe_local, probe_abs):
                            if local <= q_local < end:
                                candidate_hidden[q_abs] = h[q_local-local].detach().clone()
                    h32 = h.to(torch.float32)
                    del h
                    projected = _project(h32)
                    if probe_local:
                        for q_local, q_abs in zip(probe_local, probe_abs):
                            if local <= q_local < end:
                                candidate_output[q_abs] = projected[q_local-local].detach().clone()
                    del h32
                    out[local:end].copy_(projected)
                    del projected
                if not st.get('final_output_streaming_announced'):
                    _lm_print(
                        '[MiniMaxH3 LongMedia] Final-output streaming enabled: '
                        f'{label} {n} tokens -> {chunks} chunks of <= {chunk}; FP32 hidden is chunk-local',
                        flush=True,
                    )
                    st['final_output_streaming_announced'] = True
                st['final_output_streaming_calls'] = int(st.get('final_output_streaming_calls', 0)) + 1
                st['final_output_max_tokens'] = max(int(st.get('final_output_max_tokens', 0)), n)
                st['final_output_max_chunks'] = max(int(st.get('final_output_max_chunks', 0)), chunks)

                if probe_local and candidate_hidden and candidate_output:
                    try:
                        # Gather all sampled rows in one tiny tensor.  Final norm and
                        # AdaLN are token-local; the output projection is checked with
                        # robust FP64 metrics because GEMM batching can change last-bit
                        # rounding even when the operation is mathematically equivalent.
                        idx = torch.tensor(probe_abs, device=x.device, dtype=torch.long)
                        ref_h = layer.norm(x.index_select(0, idx))
                        if torch.is_tensor(row) and int(row.numel()) != 1:
                            flat_row = row.reshape(-1)
                            if int(flat_row.numel()) == n:
                                probe_ids = flat_row.index_select(0, torch.tensor(probe_local, device=flat_row.device, dtype=torch.long))
                            elif int(flat_row.numel()) == int(x.shape[0]):
                                probe_ids = flat_row.index_select(0, idx.to(flat_row.device))
                            else:
                                raise RuntimeError(
                                    '[0.4.20 FINAL-HEAD MOD-ROW COMPAT] unsupported probe row layout: '
                                    f'row_shape={tuple(row.shape)}, row_numel={int(flat_row.numel())}, '
                                    f'segment_tokens={n}, total_tokens={int(x.shape[0])}.'
                                )
                            probe_ids = probe_ids.to(device=scale.device, dtype=torch.long)
                            ref_sc = scale.index_select(0, probe_ids)
                            ref_sh = shift.index_select(0, probe_ids)
                        else:
                            scalar_row = int(row.reshape(-1)[0].item()) if torch.is_tensor(row) else int(row)
                            ref_sc = scale[scalar_row]
                            ref_sh = shift[scalar_row]
                        ref_h = ref_h * (1.0 + ref_sc) + ref_sh
                        got_h = torch.stack([candidate_hidden[q] for q in probe_abs], dim=0)
                        _v24_final_report(st, 'NORM-ADALN', ref_h, got_h, stream=label, offsets=probe_local)
                        ref_out = _project(ref_h.to(torch.float32))
                        got_out = torch.stack([candidate_output[q] for q in probe_abs], dim=0)
                        _v24_final_report(st, 'OUTPUT-HEAD', ref_out, got_out, stream=label, offsets=probe_local)
                        st[f'v24_final_{label}_captured'] = True
                        del idx, ref_h, got_h, ref_out, got_out
                    except Exception as diag_exc:
                        _lm_print('[MiniMaxH3 LongMedia][V24 FINAL-LAYER A/B] '
                              f'stream={label}, diagnostic failed: {type(diag_exc).__name__}: {diag_exc}', flush=True)
                if pdd_params is not None:
                    # Release/offload the full PDD bank after the effective streamed
                    # projection has consumed all target-token chunks.
                    pdd_params[2].__exit__(None, None, None)
                return out

            # TEST build: final-output CUDA timing profiler disabled; streaming unchanged.
            trace = False
            if trace:
                try:
                    torch.cuda.synchronize(torch.cuda.current_device())
                except Exception:
                    pass
                before = _cuda_memory_snapshot()
                torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
                started = time.perf_counter()

            effective_shifts = shifts if shifts is not None else (12.0, 3.0)
            v = _head_segment(va, vb, vrow, layer.video_out, 'video', float(effective_shifts[0]))
            a = _head_segment(aa, ab, arow, layer.audio_out, 'audio', float(effective_shifts[1]))

            if trace:
                try:
                    torch.cuda.synchronize(torch.cuda.current_device())
                except Exception:
                    pass
                after = _cuda_memory_snapshot()
                peak = int(torch.cuda.max_memory_allocated(torch.cuda.current_device()))
                elapsed = (time.perf_counter() - started) * 1000.0
                st['final_output_first_profile_complete'] = True
                st['final_output_peak_allocated_mb'] = _mb(peak)
                st['final_output_elapsed_ms'] = round(elapsed, 1)
                _lm_print(
                    '[MiniMaxH3 LongMedia] Final-output streaming profile: '
                    f'alloc {_mb(before["allocated"]):.1f} -> {_mb(after["allocated"]):.1f} MB, '
                    f'peak {_mb(peak):.1f} MB, reserved {_mb(after["reserved"]):.1f} MB, '
                    f'driver free {_mb(after["driver_free"]):.1f} MB, {elapsed:.1f} ms',
                    flush=True,
                )
            return v, a

        final_layer.forward = types.MethodType(_streamed_forward, final_layer)
        final_layer._latentlab_final_output_streaming_installed = True
        state['final_output_streaming_installed'] = True
        return True
    except Exception as exc:
        state['final_output_streaming_error'] = f'{type(exc).__name__}: {exc}'
        _lm_print('[MiniMaxH3 LongMedia] Final-output streaming fallback: ' + state['final_output_streaming_error'], flush=True)
        return False

class MiniMaxH3LatentLabMLPChunking:
    """Internal GUIDER wrapper enabling exact token-chunked H3 MLP execution."""

    DESCRIPTION = 'Internal H3 low-VRAM token-chunked MLP wrapper.'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'guider': ('GUIDER',),
                'chunk_tokens': ('INT', {'default': 8192, 'min': 256, 'max': 131072, 'step': 256}),
                'max_blocks': ('INT', {'default': 128, 'min': 1, 'max': 256, 'step': 1}),
                'sol_mode': (['auto', 'existing', 'sol', 'scheduled_sol'], {'default': 'existing'}),
                'sol_tau_start': ('FLOAT', {'default': 1.3, 'min': 0.0, 'max': 4.0, 'step': 0.05}),
                'sol_tau_end': ('FLOAT', {'default': 0.8, 'min': 0.0, 'max': 4.0, 'step': 0.05}),
                'sol_curve': (['linear', 'cosine', 'sqrt', 'smoothstep', 'exponential', 'step'], {'default': 'linear'}),
                'sol_min_tokens': ('INT', {'default': 4096, 'min': 256, 'max': 131072, 'step': 256}),
                'sol_dense_percent': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 0.9, 'step': 0.05}),
                'sol_sink_conditioning': (['exact_kv', 'exact_kv_and_rows', 'off'], {'default': 'exact_kv'}),
                'sol_qkv_chunk_tokens': ('INT', {'default': 8192, 'min': 0, 'max': 131072, 'step': 8192}),
                'sol_out_proj_chunk_tokens': ('INT', {'default': 24576, 'min': 0, 'max': 131072, 'step': 8192}),
                'vram_activation_reserve_mb': ('INT', {'default': 4096, 'min': 0, 'max': 12288, 'step': 512}),
                'inter_block_vram_guard_mb': ('INT', {'default': 2048, 'min': 0, 'max': 8192, 'step': 256}),
                'inter_block_guard_cooldown_blocks': ('INT', {'default': 4, 'min': 0, 'max': 32, 'step': 1}),
                'inter_block_guard_emergency_mb': ('INT', {'default': 512, 'min': 0, 'max': 4096, 'step': 256}),
                'inter_block_guard_emergency_cooldown_blocks': ('INT', {'default': 3, 'min': 0, 'max': 32, 'step': 1}),
                'late_block_guard_start': ('INT', {'default': 40, 'min': 0, 'max': 127, 'step': 1}),
                'late_block_guard_target_mb': ('INT', {'default': 6144, 'min': 0, 'max': 12288, 'step': 256}),
                'late_block_guard_min_cached_mb': ('INT', {'default': 512, 'min': 0, 'max': 4096, 'step': 256}),
                'step_boundary_cleanup_mb': ('INT', {'default': 2048, 'min': 0, 'max': 8192, 'step': 256}),
                'sol_sigma_hi': ('FLOAT', {'default': 1.0, 'min': -1000.0, 'max': 1000.0, 'step': 0.0001}),
                'sol_sigma_lo': ('FLOAT', {'default': 0.0, 'min': -1000.0, 'max': 1000.0, 'step': 0.0001}),
            }
        }

    RETURN_TYPES = ('GUIDER', 'H3_BLOCK_MEMORY_TRACE_STATE')
    RETURN_NAMES = ('guider', 'mlp_chunk_state')
    FUNCTION = 'wrap'
    CATEGORY = CATEGORY_LONGMEDIA

    def wrap(self, guider, memory_mode='normal', requested_memory_mode='normal', chunk_tokens=8192, max_blocks=128, sol_mode='existing', sol_tau_start=1.3, sol_tau_end=0.8, sol_curve='linear', sol_min_tokens=4096, sol_dense_percent=0.0, sol_sink_conditioning='exact_kv', sol_qkv_chunk_tokens=8192, sol_out_proj_chunk_tokens=24576, vram_activation_reserve_mb=4096, inter_block_vram_guard_mb=2048, inter_block_guard_cooldown_blocks=4, inter_block_guard_emergency_mb=512, inter_block_guard_emergency_cooldown_blocks=3, late_block_guard_start=40, late_block_guard_target_mb=6144, late_block_guard_min_cached_mb=512, step_boundary_cleanup_mb=2048, sol_sigma_hi=1.0, sol_sigma_lo=0.0):
        wrapped = copy.copy(guider)

        # Detect the actual diffusion execution backend before ComfyUI performs
        # memory planning.  This lets one workflow use backend-specific startup
        # headroom without altering the proven NVFP4 path.
        _runtime_patcher = getattr(guider, 'model_patcher', None)
        runtime_profile = _detect_h3_model_runtime(_runtime_patcher)
        _announce_h3_model_runtime(runtime_profile)

        _fasth3_contract = None
        _fastvideo_vsa_contract = None
        try:
            _fasth3_diffusion = _runtime_patcher.get_model_object('diffusion_model')
            _fasth3_contract = getattr(_fasth3_diffusion, '_longmedia_fasth3_contract', None)
            _fastvideo_vsa_contract = getattr(_fasth3_diffusion, '_longmedia_fastvideo_vsa_contract', None)
        except Exception:
            _fasth3_contract = None
            _fastvideo_vsa_contract = None
        if isinstance(_fasth3_contract, dict):
            if not bool(_fasth3_contract.get('adaln_lookup_ready', False)):
                raise RuntimeError(
                    '[MiniMaxH3 LongMedia][FastH3 STARTUP PRECHECK] checkpoint was detected, but the exact '
                    '7-row FastH3 AdaLN runtime was not installed during model load. Sampling is blocked '
                    'before denoise to avoid running the FastH3 core with template AdaLN.'
                )
            _vsa_fn, _vsa_note = _fast_h3_vsa_executor()
            if _vsa_fn is None:
                raise RuntimeError(
                    '[MiniMaxH3 LongMedia][FastH3 STARTUP PRECHECK] no learned-VSA executor is available; '
                    f'detail={_vsa_note}. No denoise pass was started.'
                )
            _vsa_label = getattr(_vsa_fn, '_longmedia_executor_label', getattr(_vsa_fn, '__name__', 'unknown'))
            _lm_print(
                '[MiniMaxH3 LongMedia][FastH3 STARTUP PRECHECK] PASS: '
                f'4-step contract; VSA tile={int(_fasth3_contract.get("tile_size", 64))}; '
                f'sparsity={float(_fasth3_contract.get("sparsity", .9)):.2f}; '
                f'learned-VSA executor={_vsa_label}; exact FastH3 AdaLN lookup ready; '
                'LongMedia memory ownership retained'
                + (f'; compatibility note={_vsa_note}' if _vsa_note else ''),
                flush=True,
            )

        if isinstance(_fastvideo_vsa_contract, dict):
            _vsa_fn, _vsa_note = _fast_h3_vsa_executor()
            if _vsa_fn is None:
                raise RuntimeError(
                    '[MiniMaxH3 LongMedia][FastVideo VSA STARTUP PRECHECK] no learned-VSA executor is available; '
                    f'detail={_vsa_note}. No denoise pass was started.'
                )
            _vsa_label = getattr(_vsa_fn, '_longmedia_executor_label', getattr(_vsa_fn, '__name__', 'unknown'))
            _lm_print(
                '[MiniMaxH3 LongMedia][FastVideo VSA STARTUP PRECHECK] PASS: '
                f'gates=50/50; tile={int(_fastvideo_vsa_contract.get("tile_size", 64))}; '
                f'sparsity={float(_fastvideo_vsa_contract.get("sparsity", .9)):.2f}; '
                f'topk={float(_fastvideo_vsa_contract.get("topk_ratio", .1)):.2f}; '
                f'learned-VSA executor={_vsa_label}; 4-step T2AV contract; '
                'stock H3 AdaLN/core layout retained; LongMedia memory ownership retained'
                + (f'; compatibility note={_vsa_note}' if _vsa_note else ''),
                flush=True,
            )

        runtime_policy = _h3_runtime_auto_policy(
            runtime_profile.get('backend', 'unknown'),
            quant_variant=runtime_profile.get('quant_variant'),
            chunk_tokens=chunk_tokens,
            sol_qkv_chunk_tokens=sol_qkv_chunk_tokens,
            sol_out_proj_chunk_tokens=sol_out_proj_chunk_tokens,
            vram_activation_reserve_mb=vram_activation_reserve_mb,
        )

        # Apply startup policy locally.  Public node inputs and workflow ABI stay
        # unchanged; the chosen values are stored in state for diagnostics.
        chunk_tokens = int(runtime_policy['chunk_tokens'])
        sol_qkv_chunk_tokens = int(runtime_policy['sol_qkv_chunk_tokens'])
        sol_out_proj_chunk_tokens = int(runtime_policy['sol_out_proj_chunk_tokens'])
        vram_activation_reserve_mb = int(runtime_policy['vram_activation_reserve_mb'])

        _lm_print(
            '[MiniMaxH3 LongMedia][MODEL POLICY] '
            f"backend={runtime_policy['backend']}, profile={runtime_policy['name']}, "
            f"quant_variant={runtime_profile.get('quant_variant')}, "
            f"reserve={vram_activation_reserve_mb} MB, "
            f"MLP={chunk_tokens}, QKV={sol_qkv_chunk_tokens}, "
            f"OUT={sol_out_proj_chunk_tokens}"
            + (
                ", native_weight_prefetch=ON, cache_trim=EMERGENCY_ONLY"
                if str(runtime_policy.get('backend', '')).lower()
                in ('int8', 'int8-convrot-w4a4')
                else ""
            ),
            flush=True,
        )
        _lm_print(
            '[MiniMaxH3 LongMedia][V8 QKV OWNERSHIP] '
            f'input_requested={int(sol_qkv_chunk_tokens)}; '
            f'policy_effective={int(runtime_policy.get("sol_qkv_chunk_tokens", sol_qkv_chunk_tokens))}; '
            'owner=explicit_sampler_then_block0_safety',
            flush=True,
        )


        # Ask ComfyUI's normal prepare_sampling/load_models_gpu path to reserve
        # additional activation headroom before it decides how many H3 weights
        # to keep resident on the GPU.  This is deliberately done by augmenting
        # ModelPatcher.memory_required(), rather than unloading weights from an
        # already-running forward.  ComfyUI can then use its native partial-load
        # / partial-unload machinery and keep the remainder on the offload device.
        reserve_bytes = max(0, int(vram_activation_reserve_mb)) * 1024 * 1024
        reserve_stats = {
            'requested_mb': int(vram_activation_reserve_mb),
            'memory_required_calls': 0,
            'last_base_required_mb': None,
            'last_total_required_mb': None,
            'patcher_cloned': False,
            'error': None,
        }
        if reserve_bytes > 0 and hasattr(guider, 'model_patcher'):
            try:
                reserve_patcher = guider.model_patcher.clone()
                base_memory_required = reserve_patcher.memory_required

                def _memory_required_with_activation_reserve(input_shape, _base=base_memory_required, _reserve=reserve_bytes, _stats=reserve_stats):
                    base = int(_base(input_shape))
                    total = base + int(_reserve)
                    _stats['memory_required_calls'] = int(_stats.get('memory_required_calls', 0)) + 1
                    _stats['last_base_required_mb'] = round(base / (1024 * 1024), 1)
                    _stats['last_total_required_mb'] = round(total / (1024 * 1024), 1)
                    if _stats['memory_required_calls'] == 1:
                        _lm_print(
                            '[MiniMaxH3 LongMedia] Activation VRAM reserve requested: '
                            f"base {base / (1024*1024):.1f} MB + reserve {_reserve / (1024*1024):.1f} MB "
                            f"= {total / (1024*1024):.1f} MB for ComfyUI memory planning",
                            flush=True,
                        )
                    return total

                # Instance attribute intentionally shadows the class method.
                # sampler_helpers.prepare_sampling() calls this before model load.
                reserve_patcher.memory_required = _memory_required_with_activation_reserve
                wrapped.model_patcher = reserve_patcher
                reserve_stats['patcher_cloned'] = True
            except Exception as exc:
                reserve_stats['error'] = f'{type(exc).__name__}: {exc}'
                _lm_print('[MiniMaxH3 LongMedia] Activation reserve fallback: ' + reserve_stats['error'], flush=True)
        wrapped.model_options = _clone_model_options_safe(getattr(guider, 'model_options', {}) or {})

        transformer_options = wrapped.model_options.setdefault('transformer_options', {})
        _runtime_backend = str(
            runtime_profile.get('backend', 'unknown')
        ).lower()
        transformer_options['latentlab_h3_runtime_backend'] = _runtime_backend
        transformer_options['model_runtime_backend'] = _runtime_backend
        # V27: native INT8 keeps Comfy's dynamic-VBAR prefetch enabled.
        # LongMedia owns activation memory, but upstream owns quantized-weight residency.
        _runtime_quant_variant = str(runtime_profile.get('quant_variant') or '').lower()

        # v0.3.25: native INT8 checkpoints can be larger than physical VRAM. On
        # 8-18 GB cards, speculative Comfy dynamic-VBAR prefetch may issue the next
        # 64 MB AIMDO device copy while the current block/activations already occupy
        # the remaining headroom, causing HostBuffer.read_file_slice -> CUDA OOM
        # before the first denoise step. Use demand loading on constrained cards.
        _device_vram_gb = None
        if torch.cuda.is_available():
            try:
                _device_vram_gb = float(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory) / (1024.0 ** 3)
            except Exception:
                _device_vram_gb = None
        _int8_low_vram_streaming = (
            _runtime_backend in ('int8', 'int8-convrot-w4a4')
            and _runtime_quant_variant != 'w4a8'
            and _device_vram_gb is not None
            and _device_vram_gb <= 18.5
        )
        _forced_streaming_mode = str(memory_mode) in ('low_vram', 'ultra_low_vram')
        _model_size_b = _h3_model_size_bytes_from_guider(guider) or 0
        _gpu_size_b = int((_device_vram_gb or 0.0) * (1024 ** 3))
        _out_of_core_streaming = bool(_model_size_b and _gpu_size_b and _model_size_b > int(_gpu_size_b * 1.05))

        # v0.3.75: for oversized AIMDO-backed H3 models on RAM-rich hosts, warm
        # checkpoint mmap pages into the OS file cache before the first denoise
        # forward.  This never creates a second tensor copy and never changes H3
        # math or VRAM policy; it only aims to replace repeated NVMe reads with
        # reclaimable filesystem-cache hits.
        if _out_of_core_streaming:
            _prewarm = _prewarm_h3_file_cache(
                getattr(guider, 'model_patcher', None),
                model_size_bytes=int(_model_size_b or 0),
                min_ram_headroom_gb=10.0,
            )
            _lm_print(
                '[MiniMaxH3 LongMedia][RAM FILE-CACHE PREWARM] '
                f"status={_prewarm.get('status')} payloads={_prewarm.get('payloads',0)} "
                f"payload={_prewarm.get('payload_bytes',0)/(1024**3):.1f}GB "
                f"budget={_prewarm.get('budget_bytes',0)/(1024**3):.1f}GB "
                f"touched={_prewarm.get('touched_bytes',0)/(1024**3):.1f}GB "
                f"time={_prewarm.get('seconds',0.0):.2f}s "
                f"reason={_prewarm.get('reason') or '-'}",
                flush=True,
            )
        # v0.3.77: recent Comfy/AIMDO (0.4.6+) contains threaded-loader and
        # DynamicVRAM fixes that did not exist when our 0.3.52 hard gate was
        # introduced.  For oversized *native* INT8, hand residency back to the
        # native DynamicVRAM loader as a clean A/B. W4A8 remains on the proven
        # guarded path. This changes scheduling only, never H3 math.
        _aimdo_raw, _aimdo_ver = _pkg_version_tuple('comfy-aimdo')
        _kitchen_raw, _kitchen_ver = _pkg_version_tuple('comfy-kitchen')
        _recent_aimdo = bool(_aimdo_ver is not None and _aimdo_ver >= (0, 4, 6))
        # v0.5.38: on <=18.5 GB cards native INT8 is explicitly classified
        # as guarded out-of-core streaming.  The previous condition below
        # accidentally let `_native_aimdo_fastpath` win over
        # `_int8_low_vram_streaming`, re-enabling Comfy's one-ahead VBAR
        # prefetch on exactly the constrained cards for which the guard exists.
        # That creates a transient second destination/cast allocation which is
        # reported by the memory UI as VRAM `other` and can OOM before block 0,
        # especially on a seed-only rerun.
        _native_aimdo_fastpath = bool(
            _recent_aimdo
            and _out_of_core_streaming
            and _runtime_backend in ('int8', 'int8-convrot-w4a4')
            and _runtime_quant_variant != 'w4a8'
            and not _int8_low_vram_streaming
        )
        # v0.4.41: decide the W4A8 resident-window candidate BEFORE the
        # prefetch hard-gate.  v0.4.40 armed a persistent VBAR floor but then
        # accidentally left W4A8 classified as the legacy guarded path, which
        # forced prefetch_dynamic_vbars=False at the DIFFUSION_MODEL boundary.
        # The result was exactly the observed ~regular GPU-util sawtooth: every
        # H3 block had to synchronously fault its weights instead of overlapping
        # the next block through Comfy's native threaded prefetch queue.
        #
        # The fixed policy is deliberately two-dimensional:
        #   residency: persistent watermark floor keeps a hot low-address prefix
        #   transport: native AIMDO threaded prefetch streams the unprotected tail
        # This preserves exact model math and still leaves an activation reserve.
        # v0.4.40: static AIMDO resident window for the exact problematic
        # class: recent AIMDO, oversized W4A8 H3, 15-18.5 GB dedicated VRAM.
        # AIMDO evicts from high VBAR addresses downward and honors
        # watermark_limit as a non-evictable low-address floor.  Protect ~66%
        # of physical VRAM, capped so 2.75-3+ GB remains for activations, sparse
        # attention workspace, allocator fragmentation and the streamed tail.
        _persistent_window_candidate = bool(
            _recent_aimdo
            and _out_of_core_streaming
            and _runtime_quant_variant == 'w4a8'
            and _device_vram_gb is not None
            and 15.0 <= _device_vram_gb <= 18.5
        )
        _vbar_window_target_bytes = 0
        if _persistent_window_candidate:
            _total_mb = float(_device_vram_gb) * 1024.0
            _target_mb = min(11264.0, max(8192.0, _total_mb * 0.66))
            _reserve_floor_mb = 2816.0
            _target_mb = min(_target_mb, max(0.0, _total_mb - _reserve_floor_mb))
            _vbar_window_target_bytes = int(_target_mb * 1024.0 * 1024.0)

        _w4a8_resident_fastpath = bool(_persistent_window_candidate)
        _native_transport_fastpath = bool(_native_aimdo_fastpath or _w4a8_resident_fastpath)
        _disable_dynamic_vbar_prefetch = (
            (
                _forced_streaming_mode or _out_of_core_streaming or (
                    _runtime_backend in ('int8', 'int8-convrot-w4a4')
                    and (_runtime_quant_variant == 'w4a8' or _int8_low_vram_streaming)
                )
            )
            and not _native_transport_fastpath
        )

        _lm_print(
            '[MiniMaxH3 LongMedia][NATIVE AIMDO FASTPATH] '
            f'aimdo={_aimdo_raw or "unknown"} kitchen={_kitchen_raw or "unknown"} '
            f'recent_aimdo={_recent_aimdo} native_int8_fastpath={_native_aimdo_fastpath} '
            f'out_of_core={_out_of_core_streaming} requested_mode={memory_mode}; '
            f'prefetch={"NATIVE_THREADED" if _native_transport_fastpath else "GUARDED_SYNC"}; '
            f'persistent_window={_persistent_window_candidate} '
            f'target={_vbar_window_target_bytes/(1024.0**2):.0f}MB; H3 math=UNCHANGED',
            flush=True,
        )
        transformer_options['latentlab_disable_dynamic_vbar_prefetch'] = bool(_disable_dynamic_vbar_prefetch)
        if _runtime_backend in ('int8', 'int8-convrot-w4a4') or _forced_streaming_mode:
            transformer_options['prefetch_dynamic_vbars'] = not bool(_disable_dynamic_vbar_prefetch)
        if _w4a8_resident_fastpath:
            # Hard invariant for the 16 GB W4A8 resident-window class: the
            # watermark floor without asynchronous transport only reduces VRAM
            # churn; it does not hide PCIe faults.  Never let the legacy hard
            # gate silently override the pipelined resident policy again.
            transformer_options['latentlab_disable_dynamic_vbar_prefetch'] = False
            transformer_options['prefetch_dynamic_vbars'] = True
            transformer_options['latentlab_h3_true_lookahead'] = False
            transformer_options['latentlab_h3_prefetch_depth'] = 2
            transformer_options['latentlab_h3_prefetch_pipeline'] = 'native_one_ahead'
            transformer_options['latentlab_h3_residency_strategy'] = 'persistent_prefix_plus_prefetched_tail'
            _lm_print(
                '[MiniMaxH3 LongMedia][COMPUTE-FIRST POLICY] '
                'W4A8 resident_window => prefetch_dynamic_vbars=True, hard_gate=False; '
                'pipeline=persistent_prefix+native_one_ahead; custom lookahead disabled after no-gain A/B',
                flush=True,
            )
        if _forced_streaming_mode:
            _lm_print('[MiniMaxH3 LongMedia][OUT-OF-CORE] '
                f'memory_mode={memory_mode}: speculative prefetch disabled; demand residency + activation reserve active', flush=True)

        if _runtime_backend in ('int8', 'int8-convrot-w4a4'):
            if _runtime_quant_variant == 'w4a8' and _w4a8_resident_fastpath:
                _residency_message = (
                    f'W4A8 on {_device_vram_gb:.1f} GB GPU: recent AIMDO DynamicVRAM/threaded prefetch ENABLED; '
                    'resident-window policy keeps more weights hot and uses bounded activation reserve'
                )
            elif _runtime_quant_variant == 'w4a8':
                _residency_message = 'W4A8 legacy AIMDO: guarded demand streaming retained for safety'
            elif _int8_low_vram_streaming:
                _residency_message = (
                    f'native INT8 on {_device_vram_gb:.1f} GB GPU: dynamic-VBAR prefetch DISABLED; '
                    'single-block demand residency prevents speculative destination2/cast-buffer OOM'
                )
            elif _native_aimdo_fastpath:
                _residency_message = (
                    f'native INT8 on {_device_vram_gb:.1f} GB GPU: recent AIMDO native DynamicVRAM/threaded prefetch ENABLED; '
                    'legacy LongMedia hard-gate bypassed only where VRAM headroom permits'
                )
            else:
                _residency_message = 'native Comfy dynamic-VBAR prefetch ENABLED; quantized-weight residency owned by Comfy'
            _lm_print(
                '[MiniMaxH3 LongMedia][V325 QUANT RESIDENCY] ' + _residency_message,
                flush=True,
            )
            if _runtime_quant_variant == 'w4a8':
                _lm_print(
                    '[MiniMaxH3 LongMedia][W4A8 PIPELINED RESIDENT WINDOW] persistent AIMDO watermark floor + native threaded VBAR prefetch + AUTO MLP ceiling=8192; '
                    'resident prefix stays hot while streamed tail is prefetched one block ahead',
                    flush=True,
                )
        if WrappersMP is not None:
            # Segment/presentation compatibility is independent of SOL and must
            # protect every H3 execution mode.
            wrappers = transformer_options.setdefault('wrappers', {})
            diffusion_model = wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})
            diffusion_model['MiniMaxH3LatentLabSegmentLayoutGuard'] = [
                _h3_segment_layout_guard_wrapper
            ]

            # The prefetch hard-gate is a memory policy, not an attention policy.
            # It must remain installed even when the user runs existing/Sage
            # attention; otherwise BaseModel._apply_model() re-enables dynamic
            # VBAR prefetch from current_patcher.is_dynamic() and ultra_low_vram
            # can still OOM before the first denoise step.
            # v0.3.64: defer runtime-prefetch wrapper attachment until the
            # residency state exists, so the wrapper can capture the real
            # ModelPatcher/state directly instead of relying on copied options.

            if str(sol_mode) in ('auto', 'sol', 'scheduled_sol'):
                apply_model = wrappers.setdefault(WrappersMP.APPLY_MODEL, {})
                apply_model['MiniMaxH3LatentLabSolSpan'] = [_h3_sol_span_wrapper]
        patches_replace = transformer_options.setdefault('patches_replace', {})
        dit = patches_replace.setdefault('dit', {})
        state = {
            'mode': 'token_chunked_mlp',
            'fasth3_vsa_active': isinstance(_fasth3_contract, dict),
            'fasth3_vsa_contract': dict(_fasth3_contract or {}),
            'fasth3_vsa_topk_ratio': float((_fasth3_contract or {}).get('topk_ratio', 0.10)),
            'fastvideo_vsa_active': isinstance(_fastvideo_vsa_contract, dict),
            'fastvideo_vsa_contract': dict(_fastvideo_vsa_contract or {}),
            'fastvideo_vsa_topk_ratio': float((_fastvideo_vsa_contract or {}).get('topk_ratio', 0.10)),
            'model_runtime_profile': runtime_profile,
            'model_runtime_backend': str(runtime_profile.get('backend', 'unknown')),
            'model_runtime_quant_variant': runtime_profile.get('quant_variant'),
            'model_runtime_policy': dict(runtime_policy),
            'memory_mode': str(memory_mode),
            'requested_memory_mode': str(requested_memory_mode),
            'adaptive_memory_governor_enabled': True,
            'adaptive_memory_zone': 'CALIBRATION_SAFE',
            'memory_policy_mode': str(memory_mode),
            'model_size_bytes': int(_model_size_b or 0),
            'gpu_size_bytes': int(_gpu_size_b or 0),
            'stock_transformer_math': True,
            'adaptive_memory_adjustments': 0,
            'ultra_stage_barrier_required': True,
            # V29 backend-aware throughput AUTO MLP controller. NVFP4 remains on the
            # proven fixed path; W4A8 adapts up to 8192 tokens from actual CUDA headroom.
            'auto_mlp_chunk_enabled': str(runtime_profile.get('quant_variant') or '').lower() == 'w4a8',
            'auto_mlp_chunk_last': None,
            'auto_mlp_chunk_changes': 0,
            'auto_mlp_chunk_safety_mb': 640,
            'auto_mlp_chunk_bytes_per_token': 96 * 1024,
            'int8_reusable_sol_storage': None,
            'int8_reusable_sol_storage_key': None,
            'v12_int8_sol_forward_generation': 0,
            'v12_int8_sol_forward_active': False,
            'v12_int8_sol_forward_release_count': 0,
            # V16 is a quality candidate, not a diagnostic build. Keep the
            # proven A/B helpers dormant so generation has no probe overhead.
            'v12b_linear_ab_done': {
                'qkv_proj': True,
                'out_proj': True,
                'fc1': True,
                'mlp_fc1_fc2': True,
            },
            'v13_sol_exact_ab_done': True,
            'v14_sol_exact_ab_done': True,
            'v15_tau_calibration_done': True,
            'v16_int8_quality_tau_announced': False,
            'v17_calibrated_offsets': set(),
            'v18_bf16_kv_reference_done': True,
            'v19_stage_ab_done': set(),
            'int8_block_mlp_weights_announced': False,
            'int8_cached_mlp_parity': 'unknown',
            'int8_cached_mlp_disabled_reason': None,
            'int8_semantic_dispatch_announced': False,
            # V321 native INT8: oversubscription-aware residency hysteresis.
            # Keep allocator cache during normal 20 GB-on-16 GB streaming; trim only
            # when physical free AND reclaimable effective headroom are both critical.
            'int8_residency_emergency_free_mb': 384,
            'int8_residency_emergency_effective_mb': 768,
            'int8_residency_min_cached_mb': 256,
            'int8_residency_guard_cooldown_blocks': 8,
            'int8_residency_guard_cooldown_left': 0,
            'int8_residency_last_effective_mb': 0.0,
            'int8_residency_emergency_trim_count': 0,
            'int8_sol_storage_free_floor_mb': 3072,
            'int8_sol_storage_emergency_free_mb': 2048,
            'int8_sol_storage_min_cached_mb': 1024,
            'int8_sol_storage_guard_cooldown_blocks': 4,
            'int8_sol_storage_guard_cooldown_left': 0,
            'int8_sol_storage_trim_count': 0,
            'requested_attention_mode': str(sol_mode),
            'sol_mode': str(sol_mode),
            'existing_dense_workspace_policy': False,
            'existing_workspace_guard_calls': 0,
            'existing_workspace_release_count': 0,
            'existing_workspace_released_mb': 0.0,
            'existing_workspace_last_required_mb': 0.0,
            'existing_workspace_last_qkv_mb': 0.0,
            'existing_workspace_last_backend_extra_mb': 0.0,
            'existing_workspace_last_headroom_mb': 0.0,
            'existing_workspace_backend': 'unknown',
            'existing_workspace_retry_extra_mb': 0,
            'auto_attention_selected_mode': None,
            'auto_attention_reason': None,
            'auto_attention_announced': False,
            'last_sol_tau': 0.0,
            'active_block_index': -1,
            'sol_tau_start': float(sol_tau_start),
            'sol_tau_end': float(sol_tau_end),
            'sol_curve': str(sol_curve),
            'sol_min_tokens': int(sol_min_tokens),
            'sol_dense_percent': float(sol_dense_percent),
            'sol_sink_conditioning': str(sol_sink_conditioning),
            'sol_qkv_chunk_tokens': int(sol_qkv_chunk_tokens),
            'sol_out_proj_chunk_tokens': int(sol_out_proj_chunk_tokens),
            'vram_activation_reserve_mb': int(vram_activation_reserve_mb),
            'inter_block_vram_guard_mb': int(inter_block_vram_guard_mb),
            'inter_block_guard_cooldown_blocks': int(inter_block_guard_cooldown_blocks),
            'inter_block_guard_emergency_mb': int(inter_block_guard_emergency_mb),
            'inter_block_guard_emergency_cooldown_blocks': int(inter_block_guard_emergency_cooldown_blocks),
            'late_block_guard_start': int(late_block_guard_start),
            'late_block_guard_target_mb': int(late_block_guard_target_mb),
            'late_block_guard_min_cached_mb': int(late_block_guard_min_cached_mb),
            'step_boundary_cleanup_mb': int(step_boundary_cleanup_mb),
            'late_block_guard_trim_count': 0,
            'late_block_guard_reclaimed_mb': 0.0,
            'step_boundary_cleanup_count': 0,
            'step_boundary_cleanup_reclaimed_mb': 0.0,
            'inter_block_guard_calls': 0,
            'inter_block_last_trim_call': -1000000000,
            'inter_block_last_emergency_trim_call': -1000000000,
            'inter_block_cooldown_skip_count': 0,
            'inter_block_emergency_cooldown_skip_count': 0,
            'inter_block_emergency_trim_count': 0,
            'inter_block_trim_count': 0,
            'inter_block_reclaimed_mb': 0.0,
            'mlp_inplace_reuse': False,
            'activation_reserve': reserve_stats,
            'sol_out_proj_chunked_calls': 0,
            'sol_out_proj_max_chunks': 0,
            'sol_out_proj_announced': False,
            'sol_calls': 0,
            'sol_announced': False,
            'sol_fallbacks': [],
            'sol_sigma_hi': float(sol_sigma_hi),
            'sol_sigma_lo': float(sol_sigma_lo),
            'sol_geometry_tau_boost': 0.0,
            'chunk_tokens': int(chunk_tokens),
            # AUTO VRAM controller state.
            'auto_vram_controller_enabled': True,
            'auto_vram_controller_done': False,
            'auto_vram_controller_mode': None,
            'auto_vram_controller_probe': None,
            'auto_vram_controller_before': None,
            'auto_vram_controller_after': None,
            'inter_block_guard_hysteresis_mb': 1024,
            'inter_block_emergency_hysteresis_mb': 512,
            'inter_block_min_reclaim_mb': 256,
            'inter_block_effective_skip_count': 0,
            'inter_block_hysteresis_skip_count': 0,
            'inter_block_emergency_hyst_skip_count': 0,
            'inter_block_cooldown_skip_count': 0,
            'inter_block_emergency_cooldown_skip_count': 0,
            'inter_block_low_cache_skip_count': 0,
            'inter_block_normal_trim_count': 0,
            'inter_block_emergency_trim_count': 0,
            'late_guard_hysteresis_mb': 1024,
            'late_guard_cooldown_phases': 4,
            'late_guard_cooldown_left': 0,
            'late_guard_hysteresis_skip_count': 0,
            'late_guard_cooldown_skip_count': 0,
            'allocator_backend': _cuda_allocator_backend(),
            'max_blocks': int(max_blocks),
            'forward_count': 0,
            'first_forward_started': False,
            'first_forward_complete': False,
            'first_forward_started_at': None,
            'blocks': [],
            'stages': [],
            'worst_stage': None,
            'worst_stage_peak_allocated_mb': 0.0,
            'fallback_reason': None,
            'skipped_existing_patch_indices': [],
            'patched_block_indices': [],
            'last_patched_block_index': -1,
            'highest_block_peak_allocated_mb': 0.0,
            'highest_block_peak_reserved_mb': 0.0,
            'worst_block': 0,
            'worst_block_peak_allocated_mb': 0.0,
            'oom': False,
            'oom_block': None,
            'oom_message': None,
            'oom_stats': None,
            'mlp_chunked_calls': 0,
            'max_sequence_tokens': 0,
            'max_chunks_per_mlp': 0,
            'announced': False,
            'final_output_streaming_installed': False,
            'final_output_streaming_calls': 0,
            'final_output_streaming_announced': False,
            'final_output_first_profile_complete': False,
            'final_output_streaming_error': None,
            'v24_final_ab_done': set(),
            # V25 cleanup build: V24 final-layer forensic probes disabled.
            'v24_final_video_captured': True,
            'v24_final_audio_captured': True,
            'v25_native_quant_announced': False,
            # v0.3.62 AIMDO/VBAR residency governor. Keep a reference only in
            # runtime state; this never enters workflow serialization.
            'residency_model_patcher': getattr(wrapped, 'model_patcher', None),
            'vbar_promote_count': 0,
            'vbar_last_loaded_bytes': 0,
            'vbar_last_promote_free_mb': 0.0,
            'vbar_governor_skip_pressure': 0,
            'vbar_governor_skip_hysteresis': 0,
            # v0.4.40 persistent AIMDO window: runtime-only scheduling state.
            'vbar_persistent_window_enabled': bool(_persistent_window_candidate),
            'vbar_window_target_bytes': int(_vbar_window_target_bytes),
            'vbar_window_armed': False,
            'vbar_window_armed_forward': None,
        }
        # v0.4.35: preserve external SLA only when allocator preflight proves it fits.
        # The v0.4.32 in-place compatibility kernel removed the full-size output
        # allocation, but its per-query-block Triton launch strategy is much slower
        # on real H3 workloads (especially SM120 / RTX 50-series).  Preserve the
        # external kernel as the fast existing-attention path. AUTO can still route
        # large/unsafe geometries to LongMedia's bounded embedded Sol path before
        # QKV allocation, so we keep both speed and the low-VRAM escape hatch.
        _existing_attn = transformer_options.get('optimized_attention_override')
        _existing_module = str(getattr(_existing_attn, '__module__', '') or '') if _existing_attn is not None else ''
        _external_sla, _external_sla_source = _detect_external_h3_sla(transformer_options)
        state['external_sla_detected'] = bool(_external_sla)
        state['external_sla_original_module'] = _external_sla_source if _external_sla else None
        state['external_sla_memory_safe'] = False
        state['external_sla_config'] = _extract_external_sla_config(_existing_attn) if _external_sla else None
        state['external_sla_direct_fastpath'] = bool(_external_sla and _existing_attn is not None)
        state['sla_zero_copy_calls'] = 0
        state['sla_zero_copy_announced'] = False
        state['external_sla_memory_safe_reason'] = (
            'v0.4.35 fast-path: external SLA preserved only inside allocator-safe envelope; unsafe geometry uses embedded Sol'
            if _external_sla else
            ('non-SLA override preserved' if _existing_attn is not None else 'no external optimized_attention_override')
        )
        if _external_sla:
            _lm_print(
                '[MiniMaxH3 LongMedia][SLA FAST PATH] external SLA detected; '
                'native zero-copy SLA execution enabled; ModelPatcher hook loss bypassed; '
                'fused-QKV strides preserved and full-size SLA o_s allocation eliminated',
                flush=True,
            )

        # v0.3.63 authoritative residency wiring. These are runtime-only object
        # references carried through shallow transformer_options copies.
        transformer_options['latentlab_h3_residency_state'] = state
        transformer_options['latentlab_h3_residency_patcher'] = getattr(wrapped, 'model_patcher', None)
        state['vbar_forward_count'] = 0
        state['vbar_first_forward_complete'] = False
        state['vbar_forward_promote_count'] = 0
        state['vbar_forward_skip_pressure'] = 0

        if WrappersMP is not None and (_disable_dynamic_vbar_prefetch or _persistent_window_candidate):
            import functools
            wrappers = transformer_options.setdefault('wrappers', {})
            diffusion_model = wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})
            _bound_runtime_wrapper = functools.partial(
                _h3_runtime_prefetch_wrapper,
                _bound_residency_state=state,
                _bound_residency_patcher=getattr(wrapped, 'model_patcher', None),
            )
            diffusion_model['MiniMaxH3LatentLabRuntimePrefetch'] = [_bound_runtime_wrapper]
            _lm_print(
                '[MiniMaxH3 LongMedia][VBAR BIND] persistent/guarded runtime wrapper bound directly to active ModelPatcher/state',
                flush=True,
            )

        if _runtime_backend in ('int8', 'int8-convrot-w4a4') and not state.get('v28_native_quant_announced'):
            _lm_print(
                '[MiniMaxH3 LongMedia][V31 NATIVE QUANT] INT8/W4A8 math delegated to stock Comfy quantized modules; LongMedia owns memory/chunking/SOL only',
                flush=True,
            )
            state['v28_native_quant_announced'] = True

        for i in range(int(max_blocks)):
            key = ('double_block', i)
            if key in dit:
                state['skipped_existing_patch_indices'].append(i)
                continue
            dit[key] = _H3MLPChunkPatch(i, state, chunk_tokens=int(chunk_tokens))
            state['patched_block_indices'].append(i)
        if state['patched_block_indices']:
            state['last_patched_block_index'] = max(
                state['patched_block_indices']
            )
            # MiniMax H3 DiT has 50 transformer blocks, indexed 0..49.
            # max_blocks is only the patch-install scan ceiling (default 128) and
            # must not be used as the architectural block count for diagnostics.
            _v22_h3_last = 49
            state['v21_stage_ab_targets'] = (0, _v22_h3_last // 2, _v22_h3_last)
            state['v21_stage_ab_armed'] = False
            state['v21_stage_ab_completed'] = False
            state['v21_stage_ab_generation'] = 0
            _lm_print(
                '[MiniMaxH3 LongMedia][V22 STAGE A/B TARGET] '
                f'configured targets={list(state["v21_stage_ab_targets"])}',
                flush=True,
            )
        if state['skipped_existing_patch_indices']:
            _lm_print(
                '[MiniMaxH3 LongMedia] Low-VRAM MLP skipped existing DiT patches at indices: '
                + ','.join(map(str, state['skipped_existing_patch_indices'])),
                flush=True,
            )
        if hasattr(wrapped, 'model_patcher'):
            _install_h3_final_output_streaming(wrapped.model_patcher, state, chunk_tokens=int(chunk_tokens))
        return (wrapped, state)



def _h3_vbar_residency_step_governor(block_state, snapshot, step=None):
    """v0.3.62: safely reopen AIMDO's residency watermark between denoise steps.

    ModelVBAR intentionally lowers a watermark after VRAM pressure; once lowered,
    later weights stop faulting resident even if several GB become free.  At a
    completed step boundary there are no live per-block temporaries, so this is
    the safest point to let the *next* forward attempt more persistent residency.
    We do not pin or preload weights ourselves and never touch H3 math.
    """
    if not isinstance(block_state, dict) or not snapshot:
        return
    patcher = block_state.get('residency_model_patcher')
    if patcher is None:
        return
    mode = str(block_state.get('memory_policy_mode', block_state.get('memory_mode', 'normal')))
    policy = {
        'normal': (1536.0, 2816.0, 512.0),
        'low_vram': (2304.0, 3584.0, 768.0),
        'ultra_low_vram': (3072.0, 4608.0, 1024.0),
    }
    hard_mb, promote_mb, hyst_mb = policy.get(mode, policy['low_vram'])
    free_mb = float(snapshot.get('driver_free', 0)) / (1024.0 ** 2)
    # Never reopen the watermark near the hard floor. Promotion requires a
    # generous one-step envelope above it, with per-mode conservatism.
    if free_mb < promote_mb:
        block_state['vbar_governor_skip_pressure'] = int(block_state.get('vbar_governor_skip_pressure', 0)) + 1
        return
    try:
        vbar_get = getattr(patcher, '_vbar_get', None)
        if not callable(vbar_get):
            return
        vbar = vbar_get(create=False)
        if vbar is None:
            return
        loaded_before = int(vbar.loaded_size()) if hasattr(vbar, 'loaded_size') else 0
        last_loaded = int(block_state.get('vbar_last_loaded_bytes', 0) or 0)
        last_free = float(block_state.get('vbar_last_promote_free_mb', 0.0) or 0.0)
        # Avoid resetting the watermark every step after residency has converged.
        # Reopen only if there is materially more headroom or residency is still
        # growing/near-zero. This gives AIMDO hysteresis instead of oscillation.
        residency_stalled = loaded_before <= max(last_loaded + 64 * 1024 * 1024, 256 * 1024 * 1024)
        more_headroom = free_mb >= last_free + hyst_mb
        if block_state.get('vbar_promote_count', 0) > 0 and not residency_stalled and not more_headroom:
            block_state['vbar_governor_skip_hysteresis'] = int(block_state.get('vbar_governor_skip_hysteresis', 0)) + 1
            block_state['vbar_last_loaded_bytes'] = loaded_before
            return
        vbar.prioritize()
        block_state['vbar_promote_count'] = int(block_state.get('vbar_promote_count', 0)) + 1
        block_state['vbar_last_loaded_bytes'] = loaded_before
        block_state['vbar_last_promote_free_mb'] = free_mb
        block_state['vbar_hard_floor_mb'] = hard_mb
        block_state['vbar_promote_floor_mb'] = promote_mb
        _lm_print(
            '[MiniMaxH3 LongMedia][VBAR RESIDENCY] '
            f'step={step} mode={mode} free={free_mb:.0f}MB '
            f'loaded={loaded_before/(1024.0**2):.0f}MB; watermark reopened for next forward',
            flush=True,
        )
    except Exception as exc:
        block_state['vbar_governor_error'] = f'{type(exc).__name__}: {exc}'
        if not block_state.get('vbar_governor_error_announced'):
            block_state['vbar_governor_error_announced'] = True
            _lm_print('[MiniMaxH3 LongMedia][VBAR RESIDENCY] disabled: ' + block_state['vbar_governor_error'], flush=True)

def _is_vhs_latent_preview_exception(exc: BaseException) -> bool:
    """Return True only when an exception originated inside VHS latent preview code.

    VideoHelperSuite wraps ComfyUI's previewer and runs preview decoding inside the
    sampler callback. Preview generation is a UI side effect, not part of H3 sampling,
    so a VHS-only failure must not invalidate a completed denoise forward.
    """
    traceback_node = exc.__traceback__
    while traceback_node is not None:
        filename = str(traceback_node.tb_frame.f_code.co_filename).replace('\\', '/').casefold()
        if '/videohelpersuite/latent_preview.py' in filename:
            return True
        traceback_node = traceback_node.tb_next
    return False


class _FirstStepMemoryProfilerSampler:
    """Transparent SAMPLER proxy that profiles allocator activity from before step 1."""

    def __init__(self, inner_sampler, state):
        self.inner_sampler = inner_sampler
        self.state = state

    def __getattr__(self, name):
        return getattr(self.inner_sampler, name)

    def sample(self, model_wrap, sigmas, extra_args, callback, noise,
               latent_image=None, denoise_mask=None, disable_pbar=False):
        state = self.state
        if not torch.cuda.is_available():
            return self.inner_sampler.sample(
                model_wrap, sigmas, extra_args, callback, noise,
                latent_image, denoise_mask, disable_pbar,
            )

        device = torch.cuda.current_device()
        # TEST build: skip profiling-only sampler-entry sync and allocator history.
        before = _cuda_memory_snapshot()
        state['before_sampling'] = {k + '_mb': _mb(v) for k, v in before.items()} if before else None
        enabled, history_error = False, 'disabled in TEST remove-deep-profiling build'
        state['history_enabled'] = False
        state['history_error'] = history_error
        first_callback_seen = False
        preview_guard_announced = False

        def profiled_callback(*args, **kwargs):
            nonlocal first_callback_seen, preview_guard_announced
            # TEST build: no profiling-only CUDA synchronization at step boundary.
            callback_arrival = time.perf_counter()
            snapshot = _cuda_memory_snapshot()
            result = None
            if callback is not None:
                try:
                    result = callback(*args, **kwargs)
                except Exception as exc:
                    if not _is_vhs_latent_preview_exception(exc):
                        raise
                    state['preview_callback_errors'] = int(state.get('preview_callback_errors', 0) or 0) + 1
                    state['preview_callback_error'] = f'{type(exc).__name__}: {exc}'[:2000]
                    state['preview_callback_guarded'] = True
                    if not preview_guard_announced:
                        preview_guard_announced = True
                        _lm_print(
                            '[MiniMaxH3 LongMedia][VHS PREVIEW GUARD] '
                            'VideoHelperSuite latent preview failed; sampling continues. '
                            f'Preview error: {type(exc).__name__}: {str(exc)[:500]}',
                            flush=True,
                        )
            block_state = state.get('block_trace_state')
            # Optional hard cleanup at a completed denoise-step boundary.  This
            # runs after the sampler callback has consumed the step result, so it
            # cannot invalidate live denoised tensors; it only returns dead CUDA
            # allocator pages before AIMDO prepares the next forward.
            if isinstance(block_state, dict):
                target_mb = max(0, int(block_state.get('step_boundary_cleanup_mb', 0) or 0))
                if target_mb > 0 and snapshot:
                    free_mb0 = snapshot['driver_free'] / (1024.0 ** 2)
                    cached_mb0 = snapshot['cached'] / (1024.0 ** 2)
                    if free_mb0 < target_mb and cached_mb0 >= 256.0:
                        _soft_empty_cuda_cache()
                        cleaned = _cuda_memory_snapshot()
                        if cleaned:
                            block_state['step_boundary_cleanup_count'] = int(block_state.get('step_boundary_cleanup_count', 0)) + 1
                            reclaimed = max(0, int(cleaned['driver_free']) - int(snapshot['driver_free']))
                            block_state['step_boundary_cleanup_reclaimed_mb'] = round(
                                float(block_state.get('step_boundary_cleanup_reclaimed_mb', 0.0)) + reclaimed / (1024.0 ** 2), 1
                            )
                            _lm_print(
                                '[MiniMaxH3 LongMedia] Step-boundary hard cleanup: '
                                f'free {free_mb0:.1f} -> {cleaned["driver_free"]/(1024.0**2):.1f} MB, '
                                f'cached {cached_mb0:.1f} -> {cleaned["cached"]/(1024.0**2):.1f} MB, target={target_mb}',
                                flush=True,
                            )
                            snapshot = cleaned
            step = None
            total_steps = None
            if len(args) >= 1:
                try:
                    step = int(args[0]) + 1
                except Exception:
                    pass
            if len(args) >= 4:
                try:
                    total_steps = int(args[3])
                except Exception:
                    pass
            # v0.3.63: VBAR promotion moved to the authoritative diffusion-forward
            # boundary. Keep callback profiling only; never mutate residency here.
            entry = {
                'step': step,
                'total_steps': total_steps,
                'allocated_mb': _mb(snapshot['allocated']) if snapshot else None,
                'reserved_mb': _mb(snapshot['reserved']) if snapshot else None,
                'cached_mb': _mb(snapshot['cached']) if snapshot else None,
                'driver_free_mb': _mb(snapshot['driver_free']) if snapshot else None,
                'peak_allocated_mb': _mb(torch.cuda.max_memory_allocated(device)),
                'peak_reserved_mb': _mb(torch.cuda.max_memory_reserved(device)),
            }
            if isinstance(block_state, dict) and block_state.get('blocks'):
                # Per-block tracing resets CUDA peak counters at every block boundary.
                # Replace the sampler-level peak with the maximum captured across blocks.
                entry['peak_allocated_mb'] = max(
                    float(entry['peak_allocated_mb']),
                    float(block_state.get('highest_block_peak_allocated_mb') or 0.0),
                )
                entry['peak_reserved_mb'] = max(
                    float(entry['peak_reserved_mb']),
                    float(block_state.get('highest_block_peak_reserved_mb') or 0.0),
                )
                entry['peak_source'] = 'max_of_h3_block_trace'
            else:
                entry['peak_source'] = 'torch_cuda_global'
            state['steps'].append(entry)
            if isinstance(block_state, dict):
                current = block_state.get('step_boundary_current_forward')
                if isinstance(current, dict):
                    forward_ms = max(0.0, (callback_arrival - float(current.get('block0_started_perf', callback_arrival))) * 1000.0)
                    forward_entry = {
                        'step': step,
                        'forward': current.get('forward'),
                        'block0_to_step_end_ms': round(forward_ms, 1),
                        'end_allocated_mb': entry['allocated_mb'],
                        'end_reserved_mb': entry['reserved_mb'],
                        'end_driver_free_mb': entry['driver_free_mb'],
                    }
                    block_state.setdefault('step_boundary_forward_times', []).append(forward_entry)
                    _lm_print(
                        '[MiniMaxH3 LongMedia] Step compute profile: '
                        f"step {step}, block0 -> callback {forward_entry['block0_to_step_end_ms']:.1f} ms; "
                        f"end alloc/res/free {entry['allocated_mb']:.1f}/{entry['reserved_mb']:.1f}/{entry['driver_free_mb']:.1f} MB",
                        flush=True,
                    )
                block_state['step_boundary_pending_callback'] = {
                    'step': step,
                    'time_perf': callback_arrival,
                    'allocated_mb': entry['allocated_mb'],
                    'reserved_mb': entry['reserved_mb'],
                    'driver_free_mb': entry['driver_free_mb'],
                }
            if not first_callback_seen:
                first_callback_seen = True
                path, error = _dump_cuda_memory_snapshot('after_step1') if enabled else (None, history_error)
                state['first_step_snapshot'] = path
                state['first_step_snapshot_error'] = error
                _lm_print(
                    '[MiniMaxH3 LongMedia] First-step memory profile: '
                    f"allocated {entry['allocated_mb']:.1f} MB, reserved {entry['reserved_mb']:.1f} MB, "
                    f"peak allocated {entry['peak_allocated_mb']:.1f} MB, peak reserved {entry['peak_reserved_mb']:.1f} MB, "
                    f"driver free {entry['driver_free_mb']:.1f} MB",
                    flush=True,
                )
                if isinstance(block_state, dict) and block_state.get('blocks'):
                    block_state['first_forward_complete'] = True
                    _lm_print(
                        '[MiniMaxH3 LongMedia] H3 block trace summary: '
                        f"{len(block_state['blocks'])} blocks, worst block {block_state.get('worst_block')}, "
                        f"peak allocated {block_state.get('highest_block_peak_allocated_mb', 0.0):.1f} MB, "
                        f"peak reserved {block_state.get('highest_block_peak_reserved_mb', 0.0):.1f} MB",
                        flush=True,
                    )
                if path:
                    _lm_print(f'[MiniMaxH3 LongMedia] First-step allocator snapshot: {path}', flush=True)
                elif error:
                    _lm_print(f'[MiniMaxH3 LongMedia] Snapshot unavailable: {error}', flush=True)
            return result

        try:
            output = self.inner_sampler.sample(
                model_wrap, sigmas, extra_args, profiled_callback, noise,
                latent_image, denoise_mask, disable_pbar,
            )
            state['completed'] = True
            return output
        except Exception as exc:
            message = str(exc).lower()
            is_oom = isinstance(exc, getattr(torch, 'OutOfMemoryError', RuntimeError)) or 'out of memory' in message
            if is_oom:
                state['oom'] = True
                state['oom_message'] = str(exc)[:2000]
                snapshot = _cuda_memory_snapshot()
                if snapshot:
                    state['oom_counters'] = {k + '_mb': _mb(v) for k, v in snapshot.items()}
                state['oom_peak_allocated_mb'] = _mb(torch.cuda.max_memory_allocated(device))
                state['oom_peak_reserved_mb'] = _mb(torch.cuda.max_memory_reserved(device))
                path, error = _dump_cuda_memory_snapshot('oom') if enabled else (None, history_error)
                state['oom_snapshot'] = path
                state['oom_snapshot_error'] = error
                _lm_print(
                    '[MiniMaxH3 LongMedia] CUDA OOM captured by first-step profiler. '
                    f"peak allocated {state['oom_peak_allocated_mb']:.1f} MB, "
                    f"peak reserved {state['oom_peak_reserved_mb']:.1f} MB",
                    flush=True,
                )
                if path:
                    _lm_print(f'[MiniMaxH3 LongMedia] OOM allocator snapshot: {path}', flush=True)
                elif error:
                    _lm_print(f'[MiniMaxH3 LongMedia] OOM snapshot unavailable: {error}', flush=True)
            raise
        finally:
            _stop_cuda_memory_history()


class MiniMaxH3LatentLabFirstStepMemoryProfiler:
    """Internal allocator profiler used to diagnose first-step H3 VRAM peaks."""

    DESCRIPTION = 'Internal first-step CUDA allocator profiler for Long Media sampling.'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'sampler': ('SAMPLER',),
                'max_history_entries': ('INT', {'default': 20000, 'min': 1000, 'max': 200000, 'step': 1000}),
            },
            'optional': {
                'block_trace_state': ('H3_BLOCK_MEMORY_TRACE_STATE',),
            },
        }

    RETURN_TYPES = ('SAMPLER', 'H3_MEMORY_PROFILE_STATE')
    RETURN_NAMES = ('sampler', 'profile_state')
    FUNCTION = 'wrap'
    CATEGORY = CATEGORY_LONGMEDIA

    def wrap(self, sampler, max_history_entries=20000, block_trace_state=None):
        state = {
            'max_history_entries': int(max_history_entries),
            'history_enabled': False,
            'history_error': None,
            'before_sampling': None,
            'steps': [],
            'first_step_snapshot': None,
            'first_step_snapshot_error': None,
            'oom': False,
            'oom_message': None,
            'oom_counters': None,
            'oom_snapshot': None,
            'oom_snapshot_error': None,
            'completed': False,
            'preview_callback_guarded': False,
            'preview_callback_errors': 0,
            'preview_callback_error': None,
            'block_trace_state': block_trace_state,
        }
        return (_FirstStepMemoryProfilerSampler(sampler, state), state)


class _VRAMPressureGuardSampler:
    """Transparent SAMPLER proxy that flushes CUDA cache only under real pressure."""

    def __init__(self, inner_sampler, state):
        self.inner_sampler = inner_sampler
        self.state = state

    def __getattr__(self, name):
        return getattr(self.inner_sampler, name)

    def sample(self, model_wrap, sigmas, extra_args, callback, noise,
               latent_image=None, denoise_mask=None, disable_pbar=False):
        state = self.state

        def guarded_callback(*args, **kwargs):
            result = callback(*args, **kwargs) if callback is not None else None
            state['checks'] += 1
            if not torch.cuda.is_available() or state['flushes'] >= state['max_flushes']:
                return result

            snapshot = _cuda_memory_snapshot()
            if snapshot is None:
                return result

            if (
                snapshot['driver_free'] < state['free_threshold_bytes']
                and snapshot['cached'] > state['cache_threshold_bytes']
            ):
                step = None
                total_steps = None
                if len(args) >= 1:
                    try:
                        step = int(args[0]) + 1
                    except Exception:
                        step = None
                if len(args) >= 4:
                    try:
                        total_steps = int(args[3])
                    except Exception:
                        total_steps = None

                before = snapshot
                _soft_empty_cuda_cache()
                after = _cuda_memory_snapshot()
                state['flushes'] += 1
                event = {
                    'step': step,
                    'total_steps': total_steps,
                    'cached_before_mb': _mb(before['cached']),
                    'cached_after_mb': _mb(after['cached']),
                    'reserved_before_mb': _mb(before['reserved']),
                    'reserved_after_mb': _mb(after['reserved']),
                    'driver_free_before_mb': _mb(before['driver_free']),
                    'driver_free_after_mb': _mb(after['driver_free']),
                }
                state['events'].append(event)
                step_text = (
                    f"step {step}/{total_steps}"
                    if step is not None and total_steps is not None
                    else 'sampling step'
                )
                _lm_print(
                    '[MiniMaxH3 LongMedia] VRAM pressure guard: '
                    f"{step_text}, cached {_mb(before['cached']):.1f} -> {_mb(after['cached']):.1f} MB, "
                    f"driver free {_mb(before['driver_free']):.1f} -> {_mb(after['driver_free']):.1f} MB",
                    flush=True,
                )
            return result

        return self.inner_sampler.sample(
            model_wrap, sigmas, extra_args, guarded_callback, noise,
            latent_image, denoise_mask, disable_pbar,
        )


class MiniMaxH3LatentLabVRAMPressureGuard:
    """Internal SAMPLER wrapper for adaptive intra-sampling cache cleanup."""

    DESCRIPTION = 'Internal adaptive VRAM pressure guard for Long Media sampling.'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'sampler': ('SAMPLER',),
                'free_threshold_mb': ('INT', {'default': 768, 'min': 128, 'max': 8192, 'step': 64}),
                'cache_threshold_mb': ('INT', {'default': 4096, 'min': 256, 'max': 32768, 'step': 256}),
                'max_flushes': ('INT', {'default': 2, 'min': 0, 'max': 16, 'step': 1}),
            }
        }

    RETURN_TYPES = ('SAMPLER', 'H3_VRAM_GUARD_STATE')
    RETURN_NAMES = ('sampler', 'guard_state')
    FUNCTION = 'wrap'
    CATEGORY = CATEGORY_LONGMEDIA

    def wrap(self, sampler, free_threshold_mb=768, cache_threshold_mb=4096, max_flushes=2):
        state = {
            'checks': 0,
            'flushes': 0,
            'max_flushes': int(max_flushes),
            'free_threshold_mb': int(free_threshold_mb),
            'cache_threshold_mb': int(cache_threshold_mb),
            'free_threshold_bytes': int(free_threshold_mb) * 1024 * 1024,
            'cache_threshold_bytes': int(cache_threshold_mb) * 1024 * 1024,
            'events': [],
        }
        return (_VRAMPressureGuardSampler(sampler, state), state)


class MiniMaxH3LatentLabVRAMCacheCleanup:
    """Internal passthrough used after sampling to measure and release CUDA cache."""

    DESCRIPTION = 'Internal post-sampling CUDA cache cleanup and diagnostics.'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'latent': ('LATENT',),
                'sampler_report': ('STRING', {'default': '', 'multiline': True}),
            },
            'optional': {
                'vram_guard_state': ('H3_VRAM_GUARD_STATE',),
                'memory_profile_state': ('H3_MEMORY_PROFILE_STATE',),
                'block_trace_state': ('H3_BLOCK_MEMORY_TRACE_STATE',),
            },
        }

    RETURN_TYPES = ('LATENT', 'STRING')
    RETURN_NAMES = ('latent', 'report')
    FUNCTION = 'cleanup'
    CATEGORY = CATEGORY_LONGMEDIA

    def cleanup(self, latent, sampler_report, vram_guard_state=None, memory_profile_state=None, block_trace_state=None):
        if not torch.cuda.is_available():
            return (latent, sampler_report)

        # The dependency on ``latent`` guarantees the sampler has completed.
        # Synchronize once so the before/after counters describe a stable point.
        torch.cuda.synchronize()
        before = _cuda_memory_snapshot()
        _soft_empty_cuda_cache()
        torch.cuda.synchronize()
        after = _cuda_memory_snapshot()

        released_reserved = max(0, before['reserved'] - after['reserved'])
        released_cached = max(0, before['cached'] - after['cached'])
        cleanup_data = {
            'allocated_before_mb': _mb(before['allocated']),
            'allocated_after_mb': _mb(after['allocated']),
            'reserved_before_mb': _mb(before['reserved']),
            'reserved_after_mb': _mb(after['reserved']),
            'cached_before_mb': _mb(before['cached']),
            'cached_after_mb': _mb(after['cached']),
            'released_reserved_mb': _mb(released_reserved),
            'released_cached_mb': _mb(released_cached),
            'driver_free_before_mb': _mb(before['driver_free']),
            'driver_free_after_mb': _mb(after['driver_free']),
        }

        # TEST cleanup fix: removed invalid AUTO summary that referenced
        # an out-of-scope local `state`. Core post-sampling cleanup is unchanged.

        _lm_print(
            '[MiniMaxH3 LongMedia] Post-sampling VRAM cleanup: '
            f"cached {_mb(before['cached']):.1f} -> {_mb(after['cached']):.1f} MB, "
            f"reserved {_mb(before['reserved']):.1f} -> {_mb(after['reserved']):.1f} MB, "
            f"driver free {_mb(before['driver_free']):.1f} -> {_mb(after['driver_free']):.1f} MB",
            flush=True,
        )

        try:
            report_data = json.loads(sampler_report) if sampler_report else {}
            if not isinstance(report_data, dict):
                report_data = {'sampler_report': sampler_report}
        except Exception:
            report_data = {'sampler_report': sampler_report}
        report_data['post_sampling_vram_cleanup'] = cleanup_data
        if isinstance(memory_profile_state, dict):
            report_data['first_step_memory_profile'] = {
                'max_history_entries': memory_profile_state.get('max_history_entries'),
                'history_enabled': memory_profile_state.get('history_enabled'),
                'history_error': memory_profile_state.get('history_error'),
                'before_sampling': memory_profile_state.get('before_sampling'),
                'steps': list(memory_profile_state.get('steps', [])),
                'first_step_snapshot': memory_profile_state.get('first_step_snapshot'),
                'first_step_snapshot_error': memory_profile_state.get('first_step_snapshot_error'),
                'oom': memory_profile_state.get('oom', False),
                'oom_message': memory_profile_state.get('oom_message'),
                'oom_counters': memory_profile_state.get('oom_counters'),
                'oom_peak_allocated_mb': memory_profile_state.get('oom_peak_allocated_mb'),
                'oom_peak_reserved_mb': memory_profile_state.get('oom_peak_reserved_mb'),
                'oom_snapshot': memory_profile_state.get('oom_snapshot'),
                'oom_snapshot_error': memory_profile_state.get('oom_snapshot_error'),
                'completed': memory_profile_state.get('completed', False),
                'preview_callback_guarded': memory_profile_state.get('preview_callback_guarded', False),
                'preview_callback_errors': int(memory_profile_state.get('preview_callback_errors', 0) or 0),
                'preview_callback_error': memory_profile_state.get('preview_callback_error'),
            }
        if isinstance(block_trace_state, dict):
            report_data['h3_block_memory_trace'] = {
                'allocator_backend': block_trace_state.get('allocator_backend'),
                'pre_block0': block_trace_state.get('pre_block0'),
                'block_count_traced': len(block_trace_state.get('blocks', [])),
                'blocks': list(block_trace_state.get('blocks', [])),
                'stages': list(block_trace_state.get('stages', [])),
                'worst_stage': block_trace_state.get('worst_stage'),
                'worst_stage_peak_allocated_mb': block_trace_state.get('worst_stage_peak_allocated_mb'),
                'fallback_reason': block_trace_state.get('fallback_reason'),
                'highest_block_peak_allocated_mb': block_trace_state.get('highest_block_peak_allocated_mb'),
                'highest_block_peak_reserved_mb': block_trace_state.get('highest_block_peak_reserved_mb'),
                'worst_block': block_trace_state.get('worst_block'),
                'worst_block_peak_allocated_mb': block_trace_state.get('worst_block_peak_allocated_mb'),
                'skipped_existing_patch_indices': list(block_trace_state.get('skipped_existing_patch_indices', [])),
                'oom': block_trace_state.get('oom', False),
                'oom_block': block_trace_state.get('oom_block'),
                'oom_message': block_trace_state.get('oom_message'),
                'oom_stats': block_trace_state.get('oom_stats'),
                'activation_reserve': block_trace_state.get('activation_reserve'),
                'vram_activation_reserve_mb': block_trace_state.get('vram_activation_reserve_mb'),
                'step_boundary_transitions': list(block_trace_state.get('step_boundary_transitions', [])),
                'step_boundary_forward_times': list(block_trace_state.get('step_boundary_forward_times', [])),
            }
        if isinstance(vram_guard_state, dict):
            report_data['intra_sampling_vram_guard'] = {
                'free_threshold_mb': vram_guard_state.get('free_threshold_mb'),
                'cache_threshold_mb': vram_guard_state.get('cache_threshold_mb'),
                'max_flushes': vram_guard_state.get('max_flushes'),
                'checks': vram_guard_state.get('checks', 0),
                'flushes': vram_guard_state.get('flushes', 0),
                'events': list(vram_guard_state.get('events', [])),
            }
        return (latent, json.dumps(report_data, indent=2))


def _match_frames_color_to_reference(images: torch.Tensor, reference_index: int, strength: float) -> torch.Tensor:
    """Nudge every frame's per-channel color statistics toward one reference frame.

    Blends each frame between its own colors (strength=0) and colors rescaled to
    match the reference frame's per-channel mean/std (strength=1). Useful when one
    frame is pinned exactly to a source/reference image but the rest of the clip has
    drifted slightly in color, which shows up as a visible jump at a loop seam.
    images is [T, H, W, C] in the 0..1 range. The reference frame itself is left
    untouched.
    """
    strength = float(strength)
    if strength <= 0.0 or images.shape[0] <= 1:
        return images
    strength = min(1.0, strength)
    reference = images[reference_index]
    ref_mean = reference.mean(dim=(0, 1), keepdim=True)
    ref_std = reference.std(dim=(0, 1), keepdim=True).clamp_min(1e-5)
    frame_mean = images.mean(dim=(1, 2), keepdim=True)
    frame_std = images.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
    normalized = (images - frame_mean) / frame_std * ref_std + ref_mean
    matched = torch.lerp(images, normalized.clamp(0.0, 1.0), strength)
    matched[reference_index] = images[reference_index]
    return matched


def _blend_leading_frames_to_reference(
    images: torch.Tensor, reference: torch.Tensor, n_frames: int
) -> torch.Tensor:
    """Cross-fade the first n_frames of a decoded clip toward a reference frame.

    Frame 0 is a full blend toward the reference (still not pixel-identical unless
    n_frames == 1 with an implicit weight of 1.0 handled by the caller), tapering
    linearly to 0 by frame n_frames-1. Cheaper than latent_inject — pure
    post-decode compositing, no extra sampling cost — but the reference frame
    itself is never pixel-perfect in the output. reference is [H, W, C] in 0..1.
    """
    n_frames = max(1, min(int(n_frames), images.shape[0]))
    weights = torch.linspace(1.0, 0.0, n_frames + 1, device=images.device)[:n_frames]
    reference = reference.to(images.dtype).to(images.device)
    for i in range(n_frames):
        images[i] = torch.lerp(images[i], reference, weights[i])
    return images


def _apply_loop_closure_to_tail(
    images: torch.Tensor,
    closure_frames: int,
) -> torch.Tensor:
    """Close the tail of a decoded clip toward the opening frames.

    This is a lightweight all-workflow output policy for seamless loops.  It
    does not re-run H3; instead it gradually transforms only the final decoded
    frames so the end of the clip approaches the opening sequence both in look
    and geometry.  The target sequence is phased toward the loop start, so the
    very last frame lands on frame 0 while earlier tail frames still follow the
    early-motion trend instead of snapping straight to one still image.
    """
    if not torch.is_tensor(images) or images.ndim != 4 or int(images.shape[0]) < 2:
        return images
    total_frames = int(images.shape[0])
    n = max(2, min(int(closure_frames), total_frames - 1))
    if n < 2:
        return images

    out = images.clone()
    head = images[:n].to(out.device, dtype=out.dtype).clone()
    # Shift the target so the tail naturally arrives back at frame 0 on the
    # last output frame, instead of ending on a frame that is still far away in
    # phase from the loop start.
    shifted = torch.cat((head[1:], head[:1]), dim=0)
    x = torch.linspace(0.0, 1.0, n, device=out.device, dtype=out.dtype)
    smooth = x * x * (3.0 - 2.0 * x)
    target = torch.lerp(head, shifted, smooth.view(n, 1, 1, 1))

    # Match the head-derived target sequence to the current tail statistics so
    # closure fixes geometry/phase without causing a color/exposure jump.
    stat_frames = max(1, min(12, n))
    tail_ref = out[-stat_frames:]
    target_ref = target[:stat_frames]
    tail_mean = tail_ref.mean(dim=(0, 1, 2), keepdim=True)
    tail_std = tail_ref.std(dim=(0, 1, 2), keepdim=True).clamp_min(1e-5)
    target_mean = target_ref.mean(dim=(0, 1, 2), keepdim=True)
    target_std = target_ref.std(dim=(0, 1, 2), keepdim=True).clamp_min(1e-5)
    target = ((target - target_mean) / target_std) * tail_std + tail_mean
    target = target.clamp(0.0, 1.0)

    for i in range(n):
        src_idx = total_frames - n + i
        w = smooth[i]
        out[src_idx] = torch.lerp(out[src_idx], target[i], w)
    return out


def _match_leading_frames_photometrically(
    images: torch.Tensor,
    reference_tail: torch.Tensor | None,
    stat_frames: int = 12,
    decay_frames: int = 24,
    strength: float = 1.0,
) -> torch.Tensor:
    """Photometrically align the start of a decoded clip to the previous clip tail.

    This applies only a per-channel mean/std correction to the *new* clip and
    fades that correction out over the first frames. Motion stays untouched; we
    do not crossfade pixels between clips, which would ghost moving content.
    images and reference_tail are [T, H, W, C] in 0..1.
    """
    if reference_tail is None or not torch.is_tensor(reference_tail):
        return images
    if images.ndim != 4 or reference_tail.ndim != 4 or int(images.shape[0]) < 1 or int(reference_tail.shape[0]) < 1:
        return images
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0:
        return images

    stat_frames = max(1, min(int(stat_frames), int(images.shape[0]), int(reference_tail.shape[0])))
    decay_frames = max(1, min(int(decay_frames), int(images.shape[0])))

    ref = reference_tail[-stat_frames:].to(images.device, dtype=images.dtype)
    cur = images[:stat_frames]
    ref_mean = ref.mean(dim=(0, 1, 2), keepdim=True)
    ref_std = ref.std(dim=(0, 1, 2), keepdim=True).clamp_min(1e-5)
    cur_mean = cur.mean(dim=(0, 1, 2), keepdim=True)
    cur_std = cur.std(dim=(0, 1, 2), keepdim=True).clamp_min(1e-5)

    corrected = ((images - cur_mean) / cur_std) * ref_std + ref_mean
    corrected = corrected.clamp(0.0, 1.0)

    matched = images.clone()
    weights = torch.linspace(strength, 0.0, decay_frames, device=images.device, dtype=images.dtype)
    for i in range(decay_frames):
        matched[i] = torch.lerp(images[i], corrected[i], weights[i])
    return matched


def _mix_audio_tracks(audio_list, total_duration=None):
    """Mix multiple audio dicts into one, padding channels and duration."""
    if not audio_list:
        return None
    if len(audio_list) == 1:
        return audio_list[0]
    sample_rate = audio_list[0]['sample_rate']
    max_channels = max(a['waveform'][:1].shape[1] for a in audio_list)
    if total_duration is not None:
        max_samples = round(total_duration * sample_rate)
    else:
        max_samples = max(a['waveform'][:1].shape[-1] for a in audio_list)
    mixed = torch.zeros(1, max_channels, max_samples)
    for audio in audio_list:
        wf = audio['waveform'][:1]
        if wf.shape[1] < max_channels:
            wf = wf.expand(1, max_channels, -1).clone()
        if wf.shape[-1] < max_samples:
            wf = torch.nn.functional.pad(wf, (0, max_samples - wf.shape[-1]))
        mixed = mixed + wf[:, :max_channels, :max_samples].to(mixed)
    return {'waveform': mixed, 'sample_rate': sample_rate}


def _fit_passthrough_audio_to_timeline(audio, total_duration):
    """Crop/pad an untouched AUDIO waveform to the selected target horizon.

    Values inside the retained source range are not resampled or modified. A
    shorter target crops the tail; a longer target appends silence. This keeps
    duration_source independent from audio_mode while guaranteeing mux duration
    matches the generated video timeline.
    """
    if not isinstance(audio, dict) or audio.get('waveform') is None:
        return audio, {'action': 'invalid_passthrough', 'source_samples': None, 'target_samples': None}
    waveform = audio['waveform']
    sample_rate = int(audio.get('sample_rate', 0) or 0)
    if sample_rate <= 0 or not torch.is_tensor(waveform):
        return audio, {'action': 'invalid_passthrough', 'source_samples': None, 'target_samples': None}
    target_samples = max(1, int(round(float(total_duration) * sample_rate)))
    source_samples = int(waveform.shape[-1])
    if source_samples == target_samples:
        return audio, {'action': 'exact', 'source_samples': source_samples, 'target_samples': target_samples}
    if source_samples > target_samples:
        fitted = waveform[..., :target_samples].clone()
        action = 'crop'
    else:
        fitted = torch.nn.functional.pad(waveform, (0, target_samples - source_samples))
        action = 'pad_silence'
    out = dict(audio)
    out['waveform'] = fitted
    out['sample_rate'] = sample_rate
    return out, {'action': action, 'source_samples': source_samples, 'target_samples': target_samples}


def _normalize_decoded_audio(decoded, sample_rate, target_samples=None):
    """Normalize Audio VAE decode output to ComfyUI AUDIO: waveform [B,C,L].

    MiniMax H3 Audio VAE encode consumes [B,L,C], and some decode implementations
    return that same layout. Passing [B,L,C] straight to ComfyUI makes L look like
    the channel count (e.g. 165600 channels), which later explodes in ffmpeg.
    """
    if isinstance(decoded, dict):
        waveform = decoded.get('waveform')
        sample_rate = int(decoded.get('sample_rate', sample_rate))
        if waveform is None:
            raise ValueError('Audio VAE decode returned an AUDIO dict without waveform.')
    else:
        waveform = decoded

    if not torch.is_tensor(waveform):
        raise ValueError(f'Audio VAE decode returned unsupported type: {type(waveform)!r}.')

    if waveform.ndim == 2:
        # [L,C] or [C,L]
        if waveform.shape[-1] <= 8 and waveform.shape[0] > waveform.shape[-1]:
            waveform = waveform.transpose(0, 1)
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 3:
        # Prefer [B,C,L]. H3 is stereo, so a tiny last dimension strongly means [B,L,C].
        if waveform.shape[-1] <= 8 and waveform.shape[1] > waveform.shape[-1]:
            waveform = waveform.movedim(-1, 1)
    else:
        raise ValueError(
            f'Audio VAE decode must return [B,C,L] or [B,L,C], got {tuple(waveform.shape)}.'
        )

    if waveform.shape[0] != 1:
        waveform = waveform[:1]
    # H3 audio is stereo. Do not ever let a time axis leak into the channel axis.
    if waveform.shape[1] > 2:
        if waveform.shape[-1] <= 2:
            waveform = waveform.movedim(-1, 1)
        else:
            raise ValueError(
                f'Decoded H3 audio has {waveform.shape[1]} channels; expected mono/stereo. '
                f'Raw shape: {tuple(waveform.shape)}.'
            )
    if target_samples is not None:
        target_samples = max(1, int(target_samples))
        if waveform.shape[-1] > target_samples:
            waveform = waveform[..., :target_samples]
        elif waveform.shape[-1] < target_samples:
            waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.shape[-1]))

    return {'waveform': waveform.contiguous(), 'sample_rate': int(sample_rate)}



def _suppress_boundary_photometric_outliers(
    images: torch.Tensor,
    seam_frame: int,
    radius: int = 8,
    threshold: float = 0.012,
    strength: float = 0.90,
) -> tuple[torch.Tensor, dict]:
    """Suppress isolated exposure pulses near a MultiClip RGB seam without shifting time.

    Only single-frame luminance outliers whose immediate neighbours agree with each
    other are corrected.  The correction is a per-channel mean offset toward the
    neighbour baseline; pixels are never temporally blended, duplicated, dropped,
    or re-ordered.  This is intentionally local to the seam window so genuine scene
    lighting changes elsewhere remain untouched.
    """
    if not torch.is_tensor(images) or images.ndim != 4 or int(images.shape[0]) < 3:
        return images, {'applied': False, 'corrected_frames': []}
    n = int(images.shape[0])
    seam_frame = max(0, min(int(seam_frame), n - 1))
    radius = max(1, int(radius))
    lo = max(1, seam_frame - radius)
    hi = min(n - 1, seam_frame + radius + 1)
    if hi <= lo:
        return images, {'applied': False, 'corrected_frames': []}

    work = images.clone()
    # Rec.709 luma weights are used only for outlier detection; the applied
    # correction preserves each channel independently.
    weights = torch.tensor((0.2126, 0.7152, 0.0722), device=work.device, dtype=work.dtype)
    lum = (work[..., :3] * weights).sum(dim=-1).mean(dim=(1, 2))
    corrected = []
    details = []
    threshold = max(0.001, float(threshold))
    strength = max(0.0, min(1.0, float(strength)))

    # Two passes allow a pair of adjacent pulse frames to settle without ever
    # moving time.  Candidate acceptance still requires matching neighbours.
    for _pass in range(2):
        changed = False
        lum = (work[..., :3] * weights).sum(dim=-1).mean(dim=(1, 2))
        for i in range(lo, hi):
            prev_l = float(lum[i - 1].item())
            cur_l = float(lum[i].item())
            next_l = float(lum[i + 1].item())
            baseline_l = 0.5 * (prev_l + next_l)
            spike = cur_l - baseline_l
            neighbour_gap = abs(prev_l - next_l)
            # Require an isolated impulse: the frame must be a clear outlier while
            # the frames on both sides remain mutually consistent.
            if abs(spike) < threshold or neighbour_gap > max(0.010, threshold * 0.85):
                continue
            prev_mean = work[i - 1].mean(dim=(0, 1), keepdim=True)
            next_mean = work[i + 1].mean(dim=(0, 1), keepdim=True)
            target_mean = 0.5 * (prev_mean + next_mean)
            cur_mean = work[i].mean(dim=(0, 1), keepdim=True)
            delta = (target_mean - cur_mean) * strength
            # Bound any single correction.  Larger changes are more likely to be
            # semantic lighting rather than a decode pulse and should be left alone.
            delta = delta.clamp(-0.08, 0.08)
            work[i] = (work[i] + delta).clamp(0.0, 1.0)
            if i not in corrected:
                corrected.append(i)
                details.append({
                    'frame': int(i),
                    'luma_before': float(cur_l),
                    'luma_baseline': float(baseline_l),
                    'luma_delta': float(spike),
                })
            changed = True
        if not changed:
            break

    return work, {
        'applied': bool(corrected),
        'window_start': int(lo),
        'window_end': int(max(lo, hi - 1)),
        'threshold': float(threshold),
        'strength': float(strength),
        'corrected_frames': [int(x) for x in corrected],
        'details': details,
        'temporal_blend': False,
        'frame_count_changed': False,
    }


def _multiclip_temporal_seam_index(previous_tail: torch.Tensor, next_head: torch.Tensor) -> tuple[int, dict]:
    """Choose a hard RGB cut only inside the stable part of a real decoded overlap.

    MiniMax H3 can carry a short startup/VAE transient at the beginning of an
    independently decoded continuation.  For the 34-frame MultiClip preroll we
    therefore reserve the first 17 decoded frames as an *unsafe startup zone*.
    They may be used as hidden conditioning/preroll but can never become visible.
    A small tail guard is reserved as well.  Inside the remaining safe window we
    compare normalized structure and motion.  The selected seam is a hard ownership
    switch: no RGB temporal blend, no duplicated frames, and no time hold.
    """
    import torch.nn.functional as F
    n = min(int(previous_tail.shape[0]), int(next_head.shape[0]))
    if n <= 2:
        return max(0, n // 2), {
            'overlap_frames': n,
            'method': 'center_fallback',
            'startup_guard_frames': 0,
            'tail_guard_frames': 0,
        }
    a = previous_tail[:n].float().permute(0, 3, 1, 2)
    b = next_head[:n].float().permute(0, 3, 1, 2)
    target_h = min(64, int(a.shape[-2]))
    target_w = min(64, int(a.shape[-1]))
    if target_h != int(a.shape[-2]) or target_w != int(a.shape[-1]):
        a = F.interpolate(a, size=(target_h, target_w), mode='bilinear', align_corners=False)
        b = F.interpolate(b, size=(target_h, target_w), mode='bilinear', align_corners=False)

    def norm_frames(x):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(1.0e-4)
        return (x - mean) / std

    an = norm_frames(a)
    bn = norm_frames(b)

    # v0.4.17: 17f is one observed H3 output temporal pulse.  With the native
    # 34f preroll, hide that whole startup phase and keep a 4f tail margin.
    startup_guard = 17 if n >= 34 else max(1, n // 2)
    tail_guard = 4 if n >= 12 else 1
    lo = max(1, min(startup_guard, n - 2))
    hi = max(lo + 1, n - tail_guard)
    hi = min(hi, n - 1)
    if hi <= lo:
        lo = max(1, n // 2)
        hi = min(n - 1, lo + 1)

    center = 0.5 * float(lo + max(lo, hi - 1))
    best_j = max(lo, min(hi - 1, int(round(center))))
    best_score = None
    scores = []
    for j in range(lo, hi):
        w0 = max(lo, j - 2)
        w1 = min(hi, j + 3)
        structure = (an[w0:w1] - bn[w0:w1]).abs().mean()
        va = an[j] - an[j - 1]
        vb = bn[min(n - 1, j + 1)] - bn[j]
        motion = (va - vb).abs().mean()
        center_penalty = abs(float(j) - center) / max(1.0, float(hi - lo))
        score = float(structure.item()) + 0.35 * float(motion.item()) + 0.04 * center_penalty
        scores.append((j, score))
        if best_score is None or score < best_score:
            best_j, best_score = int(j), float(score)
    return best_j, {
        'overlap_frames': int(n),
        'method': 'safe_window_normalized_structure_plus_motion',
        'selected_index': int(best_j),
        'score': float(best_score if best_score is not None else 0.0),
        'search_start': int(lo),
        'search_end': int(max(lo, hi - 1)),
        'startup_guard_frames': int(startup_guard),
        'tail_guard_frames': int(tail_guard),
        'startup_frames_visible': False,
    }


def _multiclip_hidden_overlap_frames() -> int:
    # 34 = 2*17. Adding it to any native H3 17*k+5 clip keeps the generated
    # clip on the native frame grid while providing a real overlap for seam search.
    return 34


def _decode_multiclip_audio_segments(segment_latents, audio_vae, segment_lengths, total_duration,
                                     hidden_overlaps=None, seam_indices=None):
    """Decode native per-clip H3 audio and splice at the same temporal seams as video."""
    if audio_vae is None:
        return None, []
    sr = int(getattr(audio_vae, 'audio_sample_rate', 32000))
    hidden_overlaps = list(hidden_overlaps or [])
    seam_indices = list(seam_indices or [])
    pieces = []
    reports = []
    for i, clip_av in enumerate(segment_latents or []):
        clip_video, clip_audio = unpack_av_samples(clip_av)
        if not hasattr(clip_audio, 'shape') or clip_audio.ndim != 4 or int(clip_audio.shape[1]) != 32 or int(clip_audio.shape[2]) != 2:
            raise RuntimeError(
                'LongMedia MultiClip received an invalid generated H3 audio latent for per-clip AudioVAE decode: '
                f'clip={i}, shape={tuple(clip_audio.shape) if hasattr(clip_audio, "shape") else type(clip_audio).__name__}'
            )
        decoded = audio_vae.decode(clip_audio)
        actual_frames = int(frame_count_from_video_t(clip_video.shape[2]))
        target_samples = int(round((actual_frames / FPS) * sr))
        norm = _normalize_decoded_audio(decoded, sr, target_samples)
        pieces.append(norm['waveform'])
        reports.append({'clip_index': int(i), 'samples': int(norm['waveform'].shape[-1]), 'decoded_frames': actual_frames})
    if not pieces:
        return None, reports

    waveform = pieces[0]
    for i in range(1, len(pieces)):
        hidden = int(hidden_overlaps[i]) if i < len(hidden_overlaps) else 0
        seam = int(seam_indices[i]) if i < len(seam_indices) else hidden // 2
        if hidden <= 0:
            waveform = torch.cat((waveform, pieces[i]), dim=-1)
            continue
        overlap_samples = int(round((hidden / FPS) * sr))
        seam_samples = int(round((seam / FPS) * sr))
        overlap_samples = min(overlap_samples, int(waveform.shape[-1]), int(pieces[i].shape[-1]))
        seam_samples = max(0, min(seam_samples, overlap_samples))
        cut_left = int(waveform.shape[-1]) - overlap_samples + seam_samples
        right_start = seam_samples
        left = waveform[..., :cut_left].clone()
        right = pieces[i][..., right_start:].clone()
        # Tiny amplitude ramps suppress a PCM click without blending/re-timing content.
        fade = min(128, int(left.shape[-1]), int(right.shape[-1]))
        if fade > 1:
            down = torch.linspace(1.0, 0.0, fade, device=left.device, dtype=left.dtype)
            up = torch.linspace(0.0, 1.0, fade, device=right.device, dtype=right.dtype)
            left[..., -fade:] *= down
            right[..., :fade] *= up
        waveform = torch.cat((left, right), dim=-1)
        reports[i]['hidden_overlap_frames'] = hidden
        reports[i]['seam_index_frames'] = seam

    total_samples = int(round(float(total_duration) * sr))
    if int(waveform.shape[-1]) < total_samples:
        waveform = torch.nn.functional.pad(waveform, (0, total_samples - int(waveform.shape[-1])))
    waveform = waveform[..., :total_samples]
    return {'waveform': waveform, 'sample_rate': sr}, reports


def _slice_source_audio_for_segment(source_audio, start_frame, length_frames):
    """Slice source audio using H3 audio-latent-aware sample counting."""
    waveform = source_audio['waveform'][:1]
    sample_rate = int(source_audio['sample_rate'])
    audio_t = audio_latent_t(length_frames)
    target_samples = round(audio_t / AUDIO_LATENT_FPS * sample_rate)
    start_sample = math.floor(start_frame / FPS * sample_rate)
    available = waveform[..., start_sample:start_sample + target_samples]
    if available.shape[-1] < target_samples:
        available = torch.nn.functional.pad(available, (0, target_samples - available.shape[-1]))
    return available, target_samples


def _segment_timeline_contract(plan, segment_index):
    """Return the canonical global/local timeline for one LongMedia pass.

    ``segment_starts`` is the global origin of the *context window*. Continuation
    passes contain ``overlap_frames`` of hidden preroll before their user-visible
    output. Any full local conditioning stream (audio/video reference) must start
    at context_start so local t=0 stays aligned with the inherited latent overlap.
    New visible media begins at visible_start/local_visible_offset.
    """
    idx = int(segment_index)
    context_start = int(plan.segment_starts[idx])
    overlap = int(getattr(plan, 'overlap_frames', 0) or 0) if idx > 0 else 0
    length = int(plan.segment_lengths[idx])
    visible_start = context_start + overlap
    visible_frames = max(1, length - overlap)
    return {
        'segment_index': idx,
        'context_start': context_start,
        'visible_start': visible_start,
        'local_visible_offset': overlap,
        'length_frames': length,
        'visible_frames': visible_frames,
        'visible_end': visible_start + visible_frames,
    }


def _visible_segment_start_frame(plan, segment_index):
    """Compatibility helper for code that only needs the visible global origin."""
    return int(_segment_timeline_contract(plan, segment_index)['visible_start'])


def _build_lipsync_prompt(prompt, plan, has_image, has_audio):
    """Build prompt for automatic lip sync mode."""
    parts = [prompt]
    if has_image:
        parts.append('<Picture 1>')
    if has_audio:
        parts.append('<Audio 1>')
    parts.append(
        'Focus on natural mouth movement and lip synchronization with the audio.'
    )
    if plan.passes > 1:
        parts.append(
            'Maintain a single continuous uninterrupted shot. '
            'No cuts. No scene reset. Consistent character and lighting throughout.'
        )
    return '\n'.join(parts)


def _build_video_ref_edit_audio_sync_prompt(prompt):
    """Keep a replacement subject locked to the paired source performance audio."""
    parts = [str(prompt or '').strip()]
    parts.append(
        'Preserve the exact facial performance, mouth articulation, speech timing, singing timing, '
        'breathing rhythm, head motion, body timing and expression timing from the paired <Video 1> and <Audio 1> source performance. '
        'Keep the replacement subject precisely synchronized to <Audio 1> throughout the shot. '
        'Continue mouth articulation through every audible phoneme of the final phrase, then let the mouth settle naturally after the final source phoneme.'
    )
    return '\n'.join(part for part in parts if part)


def _build_video_ref_edit_timeline_prompt(prompt, *, source_video_seconds, target_seconds, duration_source):
    """Describe target/source horizon ownership without changing reference semantics."""
    parts = [str(prompt or '').strip()]
    src = max(0.0, float(source_video_seconds or 0.0))
    target = max(0.0, float(target_seconds or 0.0))
    # One 24 fps frame of tolerance avoids text churn from container rounding.
    eps = 1.0 / float(FPS)
    if src > 0.0 and target > src + eps:
        parts.append(
            f'<Video 1> establishes the source scene, motion, camera path, composition and performance for its available {src:.3f} seconds. '
            f'The target timeline continues naturally to {target:.3f} seconds, preserving the established scene logic, replacement identity, camera continuity and ongoing actions after the source video reaches its end. '
            'Connected <Audio N> references remain available as temporal and semantic conditioning throughout their own audible ranges.'
        )
    elif src > 0.0 and target + eps < src:
        parts.append(
            f'The target uses the opening {target:.3f} seconds of <Video 1> as the active edit horizon. '
            'Preserve the source motion, camera path, composition and performance timing across that target interval.'
        )
    return '\n'.join(part for part in parts if part)


_V57_SEGMENT_EVENT_RE = re.compile(
    r'^\s*(?P<sec>\d+(?:\.\d+)?)\s*(?::|sec\s*:|sec:|s\s*:|s:)\s*(?P<body>.+?)\s*$',
    re.IGNORECASE,
)


def _conditioning_meta(entry):
    """Return metadata dict for both canonical [tensor, meta] and legacy dict entries."""
    if isinstance(entry, dict):
        return entry
    if isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], dict):
        return entry[1]
    return None


def _v041_normalize_minimax_audio_ref_geometry(conditioning):
    """Keep MiniMax ref layout metadata exactly aligned with audio latent tensors.

    Stock H3 builds ``PackedLayout`` from ``minimax_refs[*].ref_audio_t`` but
    builds the actual condition rows from ``minimax_refs[*].audio_latent``.
    Segmented lip-sync rewrites the source-audio window per pass, so a stale
    ``ref_audio_t`` from the original/full reference can otherwise make those
    two independently-built row counts diverge and fail in model._forward at
    ``all_audio_rows[~audio_update] = cond_audio_rows``.

    Normalize from the tensor (the source of truth) immediately before the
    conditioning reaches Comfy's extra_conds/layout builder.  This is metadata
    only: no resampling, padding, cropping, or audio-value modification occurs.
    """
    if not conditioning:
        return conditioning
    for entry in conditioning:
        meta = _conditioning_meta(entry)
        if not meta:
            continue
        refs = meta.get('minimax_refs')
        if not refs:
            continue
        normalized = []
        changed = False
        for ref in refs:
            if not isinstance(ref, dict):
                normalized.append(ref)
                continue
            item = dict(ref)
            kind = str(item.get('kind', '') or '').lower()
            if kind in ('audio', 'video_audio'):
                audio_latent = item.get('audio_latent')
                actual_t = 0
                if audio_latent is not None and hasattr(audio_latent, 'shape') and len(audio_latent.shape) >= 1:
                    actual_t = int(audio_latent.shape[-1])
                declared_t = int(item.get('ref_audio_t', 0) or 0)
                if declared_t != actual_t:
                    item['ref_audio_t'] = actual_t
                    changed = True
                if kind == 'video_audio' and actual_t <= 0:
                    item['kind'] = 'video'
                    changed = True
            normalized.append(item)
        if changed:
            meta['minimax_refs'] = normalized
    return conditioning


def _v57_format_local_time(seconds_value):
    seconds_value = max(0.0, float(seconds_value))
    rounded = int(round(seconds_value))
    if abs(seconds_value - rounded) < 1e-6:
        return f'{rounded:02d} sec'
    return f'{seconds_value:.2f} sec'


def _v62_explicit_prompt_sections(base_prompt):
    """Split an author-written prompt into shot 0 + explicit continuation sections.

    A line beginning with ``Continue directly from the preceding video`` starts a
    new local-time section.  This lets users write ``00 sec`` / ``02 sec`` inside
    the continuation without those values being mistaken for global movie time.
    """
    text = str(base_prompt or '').strip()
    if not text:
        return []
    marker_re = re.compile(r'(?im)^(?=\s*continue\s+directly\s+from\s+the\s+preceding\s+video(?:\s+scene|\s+segment)?\b)')
    starts = [m.start() for m in marker_re.finditer(text)]
    if not starts:
        return [text]
    chunks = []
    first = text[:starts[0]].strip()
    if first:
        chunks.append(first)
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _v57_build_segment_prompt(base_prompt, plan, segment_index):
    """Create one pass-local prompt, including pass 0, on the global timeline."""
    selected, policy = _policy_build_segment_prompt(
        base_prompt,
        segment_index=int(segment_index),
        segment_starts=tuple(plan.segment_starts),
        segment_lengths=tuple(plan.segment_lengths),
        overlap_frames=int(getattr(plan, 'overlap_frames', 0) or 0),
        passes=int(getattr(plan, 'passes', len(plan.segment_lengths))),
        fps=float(FPS),
    )
    # v0.3.81 release polish: for the only supported release continuation
    # shape (exactly two passes), keep the established shot/identity contract
    # alive for the *whole* second pass instead of relying on a long AV ref
    # whose finite span can create a visible release point mid-segment.
    if (
        int(segment_index) == 1
        and int(getattr(plan, 'passes', len(plan.segment_lengths))) == 2
        and getattr(plan, 'mode', None) == 'segmented_continuation'
    ):
        continuity_lock = (
            "\n\nContinuity lock for this continuation: continue the exact same uninterrupted shot from the preceding generated video. "
            "Preserve the established camera direction, framing, subject order, relative scale, motion direction, clothing, face identity, "
            "and all visible accessories or facial coverings exactly as established. Do not replace a mask with glasses or introduce/remove accessories. "
            "Do not re-stage, re-frame, cut, reset, or start a new shot; only continue the ongoing action naturally."
        )
        selected = (str(selected).rstrip() + continuity_lock).strip()
        _lm_print(
            '[MiniMaxH3 LongMedia][TWO-PASS CONTINUITY LOCK] '
            'pass=1 full-segment shot/identity lock active; AV carry restored to overlap-sized baseline',
            flush=True,
        )
    if policy.get('mode') == 'explicit_local_section':
        _lm_print(
            f'[MiniMaxH3 LongMedia][V331 SEGMENT PROMPT] pass={int(segment_index)} '
            f'uses explicit local-time section {int(policy.get("section_index", 0))+1}/'
            f'{int(policy.get("section_count", 1))}',
            flush=True,
        )
    else:
        _lm_print(
            f'[MiniMaxH3 LongMedia][V331 SEGMENT PROMPT] pass={int(segment_index)} '
            f'visible={int(policy.get("visible_start_frame", 0))}..'
            f'{int(policy.get("visible_end_frame", 0))}f '
            f'events={int(policy.get("events_selected", 0))}/'
            f'{int(policy.get("events_total", 0))} '
            f'header_actions_dropped={int(policy.get("header_sentences_dropped", 0))}',
            flush=True,
        )
    return selected


def _v57_attach_minimax_metadata(encoded_positive, source_positive, plan, segment_index, *, drop_image_refs=False):
    """Attach H3 payload by REFERENCE, optionally stripping still-image refs after pass 0."""
    if not encoded_positive or not source_positive:
        return encoded_positive
    source_meta = _conditioning_meta(source_positive[0])
    if not source_meta:
        return encoded_positive

    is_hybrid = getattr(plan, 'mode', None) == 'hybrid'
    is_final = int(segment_index) == int(plan.passes) - 1
    for entry in encoded_positive:
        meta = _conditioning_meta(entry)
        if meta is None:
            continue
        # Keep all native MiniMax payload fields. Values intentionally stay
        # shared/read-only, except that later segmented passes may strip original
        # still-image refs to prevent literal source-frame re-entry.
        for key, value in source_meta.items():
            if not str(key).startswith('minimax_'):
                continue
            if drop_image_refs and key == 'minimax_refs':
                refs = [ref for ref in (value or [])
                        if not (isinstance(ref, dict) and str(ref.get('kind', '')).lower() == 'image')]
                if refs:
                    meta[key] = refs
                else:
                    meta.pop(key, None)
            else:
                meta[key] = value

        # Hybrid frame-0 is a global opening anchor, never a reset anchor for pass > 0.
        if is_hybrid and segment_index > 0:
            keyframes = source_meta.get('minimax_keyframes') or []
            kept = []
            if is_final:
                kept = _v329_terminal_keyframes_for_segment(
                    keyframes, plan, segment_index,
                )
            if kept:
                meta['minimax_keyframes'] = kept
            else:
                meta.pop('minimax_keyframes', None)
                meta.pop('minimax_frame_count', None)
    return encoded_positive


def _v329_terminal_keyframes_for_segment(keyframes, plan, segment_index):
    """Move a global terminal guide to the final pass's local last frame."""

    local_last = max(0, int(plan.segment_lengths[int(segment_index)]) - 1)
    result = []
    for keyframe in keyframes or []:
        if (
            float(keyframe.get('resolved_frame_index', 0)) <= 0.0
            or bool(keyframe.get('longmedia_startup_anchor'))
        ):
            continue
        item = dict(keyframe)
        item['resolved_frame_index'] = local_last
        if 'motion_context_index' in item:
            item['motion_context_index'] = local_last
        result.append(item)
    return result


def _v43_filter_continuation_ref_payload(ref_items, ref_blocks, *, drop_image_refs=False):
    """Filter pass>0 ref payloads so original still-image refs do not persist.

    H3 can occasionally literalize pass-0 image refs inside later passes. In
    segmented_continuation we therefore allow original image refs only on pass 0.
    Video/audio refs remain available because they represent temporal/audio
    context rather than static source images.
    """
    items = list(ref_items or [])
    blocks = list(ref_blocks or [])
    if not drop_image_refs:
        return items, blocks, 0

    filtered_items = []
    filtered_blocks = []
    dropped_images = 0
    for item, block in zip(items, blocks):
        kind = str(block.get('kind', '') or '').lower() if isinstance(block, dict) else ''
        if kind == 'image':
            dropped_images += 1
            continue
        filtered_items.append(item)
        filtered_blocks.append(block)
    return filtered_items, filtered_blocks, dropped_images


def _v43_strip_image_refs_from_conditioning_meta(encoded_positive, source_positive):
    """Copy MiniMax metadata except original still-image refs."""
    if not encoded_positive or not source_positive:
        return encoded_positive, 0
    source_meta = _conditioning_meta(source_positive[0])
    if not source_meta:
        return encoded_positive, 0
    dropped_images = 0
    for entry in encoded_positive:
        meta = _conditioning_meta(entry)
        if meta is None:
            continue
        for key, value in source_meta.items():
            if not str(key).startswith('minimax_'):
                continue
            if key == 'minimax_refs':
                refs = []
                for ref in (value or []):
                    if isinstance(ref, dict) and str(ref.get('kind', '')).lower() == 'image':
                        dropped_images += 1
                        continue
                    refs.append(ref)
                if refs:
                    meta[key] = refs
                else:
                    meta.pop(key, None)
            else:
                meta[key] = value
    return encoded_positive, dropped_images


def _v329_encode_continuation_native_refs(
    clip, prompt, positive, plan, segment_index, ref_items, ref_blocks,
):
    """Re-encode continuation text with the exact pass-0 Ref2VA presentation.

    Picture/video/audio ordering and every reference latent geometry stay
    unchanged across passes. This never exceeds pass-0 reference cost and avoids
    a conditioning-family discontinuity at the visible join.
    """
    import node_helpers

    items = list(ref_items or [])
    blocks = list(ref_blocks or [])
    if items:
        tokens = clip.tokenize(prompt, minimax_ref_items=items)
    else:
        tokens = clip.tokenize(prompt)
    scheduled = getattr(clip, 'encode_from_tokens_scheduled', None)
    encoded = scheduled(tokens) if callable(scheduled) else clip.encode(tokens)

    source_meta = _conditioning_meta(positive[0]) or {}
    values = {}
    if blocks:
        values['minimax_refs'] = blocks
    for key in ('minimax_visual_cond_noise_aug', 'minimax_audio_cond_noise_aug'):
        if key in source_meta:
            values[key] = source_meta[key]

    # Opening anchors are global and never reappear. A real terminal anchor is
    # retained only on the final pass, with the local pass frame count.
    if int(segment_index) == int(plan.passes) - 1:
        terminal = _v329_terminal_keyframes_for_segment(
            source_meta.get('minimax_keyframes') or [], plan, segment_index,
        )
        if terminal:
            values['minimax_keyframes'] = terminal
            values['minimax_frame_count'] = int(plan.segment_lengths[int(segment_index)])
    return node_helpers.conditioning_set_values(encoded, values)


def _v57_preencode_segment_conditionings(clip, base_prompt, positive, plan, v329_native_refs=None, lip_sync_audio=None, audio_vae=None):
    """Encode every pass in Setup and store Comfy's *converted* guider format.

    Raw CONDITIONING is ``[[cross_attn, metadata], ...]``. ``CFGGuider.original_conds``
    is deliberately a different representation: ``list[dict]`` produced by
    ``comfy.sampler_helpers.convert_cond``.  V57 accidentally stored raw CONDITIONING
    and later assigned it directly to ``original_conds['positive']``.  The first pass
    was unaffected because it used the externally-created guider, but pass 2 crashed
    when Comfy called ``kk.get(...)`` on the raw list entry.  Convert here while TE is
    already resident; no CLIP/TE/model object is retained in LongMediaPlan.
    """
    import comfy.sampler_helpers

    raw_result = [positive]
    prompts = [_v57_build_segment_prompt(base_prompt, plan, 0)]
    decouple_image_refs = bool(getattr(plan, 'decouple_original_image_refs_after_pass0', False))
    total_dropped_image_refs = 0
    for segment_index in range(1, int(getattr(plan, 'passes', 1))):
        segment_prompt = _v57_build_segment_prompt(base_prompt, plan, segment_index)
        drop_image_refs = bool(decouple_image_refs and int(segment_index) > 0)
        if v329_native_refs is not None:
            ref_items, ref_blocks = v329_native_refs
            identity_reanchor = bool(
                drop_image_refs
                and int(segment_index) == 1
                and int(getattr(plan, 'passes', 1)) == 2
                and getattr(plan, 'mode', None) == 'segmented_continuation'
            )
            if identity_reanchor:
                # v0.3.84: preserve the original still latents only as an
                # unlabelled visual prior.  Do NOT tokenize Picture items again:
                # Picture labels were the composition/literal-source re-entry path
                # fixed by v0.3.43.  Native Motion Context remains the temporal and
                # compositional authority for pass 1.
                non_image_items = [
                    item for item in (ref_items or [])
                    if not (isinstance(item, dict) and str(item.get('type', '')).lower() == 'image')
                ]
                identity_blocks = [dict(block) for block in (ref_blocks or [])]
                encoded = _encode_prompt(clip, segment_prompt)
                encoded = _v57_attach_minimax_metadata(
                    encoded, positive, plan, segment_index, drop_image_refs=True,
                )
                for entry in encoded:
                    meta = _conditioning_meta(entry)
                    if meta is None:
                        continue
                    existing = [dict(ref) for ref in (meta.get('minimax_refs', []) or [])]
                    # Keep non-image refs already carried by the normal decoupled
                    # path, then append still-image latents without tokenizer labels.
                    seen = set()
                    merged = []
                    for ref in existing + identity_blocks:
                        if not isinstance(ref, dict):
                            continue
                        kind = str(ref.get('kind', '')).lower()
                        if kind == 'image':
                            latent = ref.get('latent')
                            key = ('image', id(latent))
                        else:
                            key = (kind, id(ref.get('audio_latent')), id(ref.get('latent')))
                        if key in seen:
                            continue
                        seen.add(key)
                        merged.append(ref)
                    if merged:
                        meta['minimax_refs'] = merged
                    meta['longmedia_identity_reanchor'] = True
                dropped = sum(
                    1 for item in (ref_items or [])
                    if isinstance(item, dict) and str(item.get('type', '')).lower() == 'image'
                )
                total_dropped_image_refs += int(dropped)
                _lm_print(
                    '[MiniMaxH3 LongMedia][IDENTITY RE-ANCHOR] '
                    f'pass=1 retained {sum(1 for b in identity_blocks if str(b.get("kind", "")).lower() == "image")} '
                    'still-image latent blocks WITHOUT Picture tokenizer items; native motion context owns shot/motion',
                    flush=True,
                )
            else:
                ref_items, ref_blocks, dropped = _v43_filter_continuation_ref_payload(
                    ref_items, ref_blocks, drop_image_refs=drop_image_refs,
                )
                total_dropped_image_refs += int(dropped)
                if ref_items or ref_blocks:
                    encoded = _v329_encode_continuation_native_refs(
                        clip, segment_prompt, positive, plan, segment_index,
                        ref_items, ref_blocks,
                    )
                else:
                    encoded = _encode_prompt(clip, segment_prompt)
                    encoded = _v57_attach_minimax_metadata(encoded, positive, plan, segment_index, drop_image_refs=True)
        else:
            encoded = _encode_prompt(clip, segment_prompt)
            encoded = _v57_attach_minimax_metadata(
                encoded, positive, plan, segment_index, drop_image_refs=drop_image_refs,
            )
        if lip_sync_audio is not None:
            encoded = _v104_attach_native_lipsync_guide(
                encoded, audio_vae, lip_sync_audio, plan, segment_index,
            )
        encoded = _v041_normalize_minimax_audio_ref_geometry(encoded)
        raw_result.append(encoded)
        prompts.append(segment_prompt)

    converted_result = tuple(comfy.sampler_helpers.convert_cond(cond) for cond in raw_result)
    if v329_native_refs is not None and len(converted_result) > 1:
        ref_items, ref_blocks = v329_native_refs
        _lm_print(
            '[MiniMaxH3 LongMedia][V329 STABLE NATIVE REFS] '
            f'continuation preserves {len(ref_items)} tokenizer items and '
            f'{len(ref_blocks)} latent blocks in pass-0 order/geometry; no identity sheet',
            flush=True,
        )
    if decouple_image_refs and total_dropped_image_refs > 0:
        _lm_print(
            '[MiniMaxH3 LongMedia][REF DECOUPLING] '
            f'pass>0 removed {int(total_dropped_image_refs)} original still-image ref blocks; '
            'later passes keep generated AV context and any non-image refs only',
            flush=True,
        )
    _lm_print(
        f'[MiniMaxH3 LongMedia][V58 CONDITIONING FORMAT] pre-encoded {len(converted_result)} pass conditionings '
        'inside Setup and converted to CFGGuider list[dict] format; CLIP/TE is NOT stored in LongMediaPlan',
        flush=True,
    )
    return converted_result, tuple(prompts)


def _v60_context_step_offsets(latent_t):
    frame_per_token = (1, 4, 4, 4, 4)
    out, acc = [], 0
    for k in range(int(latent_t)):
        out.append(acc)
        acc += frame_per_token[k % 5]
    return out, acc


def _v60_attach_previous_head_guides(positive_list, previous_av, plan, segment_index):
    """Pin the previous latent tail onto the HEAD of the current H3 timeline.

    Unlike V59, this is NOT a Ref2VA video reference.  Each latent step is a
    never-denoised MiniMax keyframe guide at its real target-relative time.
    The carried span intentionally matches LongMedia's frozen overlap so the
    existing stitch removes exactly the repeated head after sampling.
    """
    if not positive_list or previous_av is None or int(segment_index) <= 0:
        return positive_list, 0
    # During GraphBuilder expansion ``previous_av`` is normally a graph-output proxy,
    # not the runtime LATENT dictionary.  Do not treat that proxy as an AV latent.
    # The actual continuation overlap is already copied/frozen at runtime by
    # MiniMaxH3LatentLabLongMediaNextSegment, so skipping the auxiliary V60 guides
    # here preserves real motion context instead of emitting a misleading error.
    if not isinstance(previous_av, dict) or 'samples' not in previous_av:
        _lm_print(
            '[MiniMaxH3 LongMedia][V319 MOTION CONTEXT] auxiliary V60 guides skipped '
            '(previous segment is a GraphBuilder proxy); frozen latent overlap remains active',
            flush=True,
        )
        return positive_list, 0
    try:
        prev_video, _prev_audio = unpack_av_samples(previous_av)
        try:
            from . import motion_context_layout_patch
        except Exception:
            import importlib
            motion_context_layout_patch = importlib.import_module(__package__ + '.motion_context_layout_patch')
        if not motion_context_layout_patch.apply_patch():
            raise RuntimeError('PackedLayout motion-context patch could not activate')
        overlap = int(getattr(plan, 'overlap_frames', 0) or 0)
        # Native H3 video-run grid.  Match (never exceed) the frozen overlap.
        run = next((g for g in (56, 39, 22, 5, 1) if g <= overlap), 0)
        if run <= 0:
            return positive_list, 0
        context_t = int(video_latent_t(run))
        context_t = min(context_t, int(prev_video.shape[2]))
        offsets, covered = _v60_context_step_offsets(context_t)
        if covered != run:
            raise RuntimeError(
                f'context latent grid mismatch: {context_t} steps cover {covered} frames, wanted {run}')
        tail = prev_video[:, :, -context_t:]
        keyframes = [
            {
                'resolved_frame_index': 0,  # stock-safe; PackedLayout patch uses marker below
                'motion_context_index': int(offset),
                'latent': tail[:, :, k:k + 1],
                'longmedia_motion_context': True,
            }
            for k, offset in enumerate(offsets)
        ]

        out = []
        attached = False
        frame_count = int(plan.segment_lengths[int(segment_index)])
        for entry in positive_list:
            if isinstance(entry, dict):
                meta = dict(entry)
                prior = list(meta.get('minimax_keyframes', []) or [])
                # Continuation head owns 0..run-1.  Preserve only anchors after it
                # (e.g. an explicit final-frame destination).
                kept = []
                for kf in prior:
                    pos = float(kf.get('motion_context_index', kf.get('resolved_frame_index', 0)))
                    if pos >= run:
                        kept.append(kf)
                merged_keyframes = kept + keyframes
                merged_keyframes.sort(
                    key=lambda kf: float(kf.get('motion_context_index', kf.get('resolved_frame_index', 0)))
                )
                meta['minimax_keyframes'] = merged_keyframes
                meta['minimax_frame_count'] = frame_count
                meta['longmedia_motion_context_frames'] = int(run)
                out.append(meta)
                attached = True
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], dict):
                new_entry = list(entry)
                meta = dict(entry[1])
                prior = list(meta.get('minimax_keyframes', []) or [])
                kept = []
                for kf in prior:
                    pos = float(kf.get('motion_context_index', kf.get('resolved_frame_index', 0)))
                    if pos >= run:
                        kept.append(kf)
                merged_keyframes = kept + keyframes
                merged_keyframes.sort(
                    key=lambda kf: float(kf.get('motion_context_index', kf.get('resolved_frame_index', 0)))
                )
                meta['minimax_keyframes'] = merged_keyframes
                meta['minimax_frame_count'] = frame_count
                meta['longmedia_motion_context_frames'] = int(run)
                new_entry[1] = meta
                out.append(new_entry)
                attached = True
            else:
                out.append(entry)
        if attached:
            _lm_print(
                f'[MiniMaxH3 LongMedia][V60 TRUE MOTION CONTEXT] previous tail pinned as '
                f'{len(keyframes)} timeline keyframe blocks covering {run} frames; '
                f'indices={offsets}; Ref2VA semantics NOT used; repeated head trimmed by overlap',
                flush=True,
            )
            return out, int(run)
    except Exception as exc:
        _lm_print(
            f'[MiniMaxH3 LongMedia][V60 TRUE MOTION CONTEXT] disabled: '
            f'{type(exc).__name__}: {exc}', flush=True,
        )
    return positive_list, 0






def _v83_native_guide_api_supported():
    """Return True when current ComfyUI supports arbitrary native H3 guide positions."""
    try:
        import inspect
        import comfy.ldm.minimax.model as minimax_model
        cls = getattr(minimax_model, 'PackedLayout', None)
        if cls is None:
            return False
        params = inspect.signature(cls.__init__).parameters
        # Current native API removed the legacy frame_count-only restriction and
        # accepts arbitrary resolved_frame_index guide rows directly.
        return 'frame_count' not in params
    except Exception:
        return False


def _v83_attach_native_motion_context(positive_list, previous_av, plan, segment_index):
    """Attach previous sampled AV tail as native H3 temporal keyframes.

    Release scope is deliberately narrow: only the first continuation of an
    exactly two-pass segmented generation.  The target latent remains fresh;
    the first 22 decoded frames are generated under never-denoised temporal
    guide rows and are trimmed by the existing stitch.
    """
    if (
        not positive_list or previous_av is None
        or int(segment_index) <= 0
        or (
            getattr(plan, 'mode', None) != 'multiclip'
            and not (
                int(segment_index) == 1
                and int(getattr(plan, 'passes', 0) or 0) == 2
                and getattr(plan, 'mode', None) == 'segmented_continuation'
            )
        )
        or not _v83_native_guide_api_supported()
    ):
        return positive_list, 0
    if not isinstance(previous_av, dict) or 'samples' not in previous_av:
        return positive_list, 0

    prev_video, prev_audio = unpack_av_samples(previous_av)
    overlap = int(getattr(plan, 'overlap_frames', 0) or 0)
    run = next((g for g in (56, 39, 22, 5) if g <= overlap), 0)
    if run <= 0:
        return positive_list, 0

    context_t = int(video_latent_t(run))
    if context_t > int(prev_video.shape[2]):
        return positive_list, 0
    offsets, covered = _v60_context_step_offsets(context_t)
    if int(covered) != int(run):
        raise RuntimeError(
            f'0.3.83 native motion context grid mismatch: {context_t} steps cover {covered}f, expected {run}f'
        )

    video_tail = prev_video[:, :, -context_t:]
    # MultiClip v0.4.16 uses a decoded hidden preroll that is longer than the
    # native motion guide span. Place the previous tail at the END of that hidden
    # preroll so its time coordinates line up with the actual global boundary.
    # Example: hidden overlap 34f, native guide 22f -> guide rows live at 12..34.
    guide_shift = max(0, int(overlap) - int(run)) if getattr(plan, 'mode', None) == 'multiclip' else 0
    keyframes = [
        {
            'resolved_frame_index': int(guide_shift + offset),
            'latent': video_tail[:, :, k:k + 1].clone(),
            'longmedia_native_motion_context': True,
        }
        for k, offset in enumerate(offsets)
    ]

    # Carry audio from the same sampled AV latent and place it on the same
    # target timeline.  This mirrors H3's native guide coordinate system rather
    # than treating the tail as a generic reference-audio embedding.
    audio_t = min(int(audio_latent_t(run)), int(prev_audio.shape[-1]))
    # v0.3.108: source Audio1 is the authoritative lip-sync clock.  Keep the
    # previous VIDEO latent as Motion Context, but do not add a second sampled
    # audio clock when a native local-0 lip-sync guide is active.  Competing
    # sampled/source audio conditions weaken articulation and can drift.
    if bool(getattr(plan, 'lip_sync_native_audio_guide', False)):
        audio_t = 0
    if audio_t > 0:
        source_frames = frame_count_from_video_t(int(prev_video.shape[2]))
        frame_rescale = float(AUDIO_LATENT_FPS) / float(FPS)
        overhang = float(prev_audio.shape[-1]) - frame_rescale * float(source_frames)
        if not (0.0 <= overhang < 1.0):
            _lm_print(
                '[MiniMaxH3 LongMedia][AUDIO GRID] '
                f'unexpected previous AV audio grid: audio_t={int(prev_audio.shape[-1])} '
                f'video_frames={int(source_frames)} raw_overhang={overhang:.6f}; using 0.0',
                flush=True,
            )
            overhang = 0.0
        end_frame = float(run) + overhang / frame_rescale
        end_coord = round(frame_rescale * end_frame)
        end_frame = float(end_coord) / frame_rescale
        audio_start_frame = end_frame - float(audio_t) / frame_rescale + float(guide_shift)
        keyframes.append({
            'resolved_frame_index': float(audio_start_frame),
            'audio_latent': prev_audio[..., -audio_t:].clone(),
            'longmedia_native_motion_audio': True,
        })

    out = []
    attached = False
    for entry in positive_list:
        if isinstance(entry, dict):
            meta = dict(entry)
            prior = [dict(kf) for kf in (meta.get('minimax_keyframes', []) or [])
                     if not bool(kf.get('longmedia_native_motion_context'))
                     and not bool(kf.get('longmedia_native_motion_audio'))]
            # The continuation guide owns the repeated 0..run-1 head. Keep only
            # explicit destination guides after that region.
            prior = [
                kf for kf in prior
                if bool(kf.get('longmedia_lipsync_audio_guide'))
                or float(kf.get('resolved_frame_index', 0.0)) >= float(guide_shift + run)
            ]
            meta['minimax_keyframes'] = prior + [dict(kf) for kf in keyframes]
            meta.pop('minimax_frame_count', None)  # native arbitrary-guide API
            meta['longmedia_native_motion_context_frames'] = int(run)
            out.append(meta)
            attached = True
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], dict):
            new_entry = list(entry)
            meta = dict(entry[1])
            prior = [dict(kf) for kf in (meta.get('minimax_keyframes', []) or [])
                     if not bool(kf.get('longmedia_native_motion_context'))
                     and not bool(kf.get('longmedia_native_motion_audio'))]
            prior = [
                kf for kf in prior
                if bool(kf.get('longmedia_lipsync_audio_guide'))
                or float(kf.get('resolved_frame_index', 0.0)) >= float(guide_shift + run)
            ]
            meta['minimax_keyframes'] = prior + [dict(kf) for kf in keyframes]
            meta.pop('minimax_frame_count', None)
            meta['longmedia_native_motion_context_frames'] = int(run)
            new_entry[1] = meta
            out.append(new_entry)
            attached = True
        else:
            out.append(entry)

    if attached:
        _lm_print(
            '[MiniMaxH3 LongMedia][VIDEO MOTION CONTEXT + SOURCE AUDIO CLOCK] '
            f'segment0->1 context={run}f hidden_overlap={overlap}f guide_shift={guide_shift}f video_steps={context_t} indices={offsets}; '
            f"audio_steps={audio_t}; source_audio_clock={bool(getattr(plan, 'lip_sync_native_audio_guide', False))}; target head is FRESH and generated under native minimax_keyframes; "
            'existing stitch trims the repeated guide span',
            flush=True,
        )
        return out, int(run)
    return positive_list, 0


def _v0322_audio_grid_offset(frame_count: int, actual_audio_t: int) -> float:
    """Signed H3 audio-grid phase at the end of a video frame span."""
    return float(actual_audio_t) - (float(frame_count) * float(AUDIO_LATENT_FPS) / float(FPS))


def _v80_native_av_context_frames(previous_frames, overlap_frames, segment_index, first_handoff_bridge=False):
    """Choose a native H3 AV-reference span for one continuation boundary.

    The first segmented handoff is special: pass 0 was image/startup-conditioned,
    while pass 1 is the first pass after still-image ref decoupling.  A context
    span limited to the visible frozen overlap (normally 22f) is often too short
    to preserve the established shot/composition through that conditioning-family
    transition.  Give *only* segment 1 a longer conditioning-only raw AV tail.

    Later continuation->continuation boundaries stay on the proven overlap-sized
    path so this does not accumulate context or change their already-good joins.
    """
    previous_frames = max(0, int(previous_frames))
    overlap_frames = max(0, int(overlap_frames))
    segment_index = int(segment_index)
    if bool(first_handoff_bridge) and segment_index == 1:
        cap = min(previous_frames, 56)
        candidates = (56, 39, 22, 5)
    else:
        cap = min(previous_frames, overlap_frames)
        candidates = (39, 22, 5)
    return next((frames for frames in candidates if frames <= cap), 0)


def _v0322_attach_native_av_context_ref(positive_list, previous_av, plan, segment_index):
    """Attach the previous raw AV tail as one native paired H3 context reference.

    This follows Continuum's latent-first handoff principle: the previous video and
    audio tails travel together in ``minimax_refs`` while the visible overlap remains
    frozen in the target latent and is trimmed after sampling.  Identity refs stay
    intact and this context ref is appended without adding tokenizer Picture tokens.
    """
    if not positive_list or previous_av is None or int(segment_index) <= 0:
        return positive_list, 0
    if not isinstance(previous_av, dict) or 'samples' not in previous_av:
        return positive_list, 0
    prev_video, prev_audio = unpack_av_samples(previous_av)
    previous_frames = frame_count_from_video_t(int(prev_video.shape[2]))
    overlap = int(getattr(plan, 'overlap_frames', 0) or 0)
    first_handoff_bridge = False  # v0.3.81 release polish: restore proven overlap-sized AV carry
    # v0.3.81: segment 0 -> 1 is the only boundary where conditioning changes
    # from startup/Picture-driven to still-ref-decoupled continuation.  Preserve
    # the exact 22f frozen head, but feed up to 56f of the already-generated raw
    # AV tail as conditioning-only history.  Later joins retain the 0.3.77 path.
    run = _v80_native_av_context_frames(
        previous_frames, overlap, segment_index,
        first_handoff_bridge=first_handoff_bridge,
    )
    if run <= 0:
        return positive_list, 0
    video_t = int(video_latent_t(run))
    audio_t = int(audio_latent_t(run))
    if int(prev_video.shape[2]) < video_t or int(prev_audio.shape[-1]) < audio_t:
        return positive_list, 0
    video_tail = prev_video[:, :, -video_t:].contiguous()
    audio_tail = prev_audio[..., -audio_t:].contiguous()
    grid_offset = _v0322_audio_grid_offset(previous_frames, int(prev_audio.shape[-1]))
    context_ref = {
        'kind': 'video_audio',
        'latent_t': int(video_tail.shape[2]),
        'latent_h': int(video_tail.shape[-2]),
        'latent_w': int(video_tail.shape[-1]),
        'ref_audio_t': int(audio_tail.shape[-1]),
        'latent': video_tail,
        'audio_latent': audio_tail,
        'longmedia_native_av_context': True,
        'longmedia_context_frames': int(run),
        'longmedia_audio_grid_offset': float(grid_offset),
        'longmedia_source_segment_index': int(segment_index) - 1,
        'longmedia_first_handoff_bridge': bool(first_handoff_bridge),
    }
    out = []
    attached = False
    for entry in positive_list:
        if isinstance(entry, dict):
            meta = dict(entry)
            refs = [dict(ref) for ref in (meta.get('minimax_refs', []) or [])
                    if not bool(ref.get('longmedia_native_av_context'))]
            refs.append(dict(context_ref))
            meta['minimax_refs'] = refs
            meta['longmedia_native_av_context_frames'] = int(run)
            out.append(meta)
            attached = True
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], dict):
            new_entry = list(entry)
            meta = dict(entry[1])
            refs = [dict(ref) for ref in (meta.get('minimax_refs', []) or [])
                    if not bool(ref.get('longmedia_native_av_context'))]
            refs.append(dict(context_ref))
            meta['minimax_refs'] = refs
            meta['longmedia_native_av_context_frames'] = int(run)
            new_entry[1] = meta
            out.append(new_entry)
            attached = True
        else:
            out.append(entry)
    if attached:
        _lm_print(
            '[MiniMaxH3 LongMedia][V0322 NATIVE AV CONTEXT] '
            f'segment={int(segment_index)} context={run}f video_t={video_t} '
            f'audio_t={audio_t} audio_grid_offset={grid_offset:+.6f}; '
            f'first_handoff_bridge={bool(first_handoff_bridge)}; '
            'paired raw AV tail appended to minimax_refs',
            flush=True,
        )
        if first_handoff_bridge:
            _lm_print(
                '[MiniMaxH3 LongMedia][FIRST HANDOFF BRIDGE] '
                f'segment0->1 uses {int(run)}f generated raw AV history while frozen overlap remains {int(overlap)}f; '
                'original still-image refs remain decoupled; pass>=2 unchanged from 0.3.77',
                flush=True,
            )
        return out, int(run)
    return positive_list, 0

def _clone_guider_with_segment_audio(guider, plan, segment_index, previous_av=None):
    """Clone guider cheaply, select pass conditioning, and add previous motion context."""
    shifted = copy.copy(guider)
    # model_options may contain CUDA/AIMDO-backed tensors and storages. Never deepcopy them.
    import comfy.model_patcher
    shifted.model_options = comfy.model_patcher.create_model_options_clone(
        getattr(guider, 'model_options', {}) or {}
    )
    timeline = _segment_timeline_contract(plan, segment_index)
    start_frame = int(timeline['context_start'])
    visible_start_frame = int(timeline['visible_start'])

    # IMPORTANT V57: do not deepcopy the whole conditioning/media payload. Hybrid refs can
    # contain encoded image/video/audio latents. Only make a new dict shell and swap positive.
    shifted.original_conds = dict(getattr(guider, 'original_conds', {}) or {})
    segment_conds = getattr(plan, 'segment_positive_conditionings', None)
    if segment_conds and int(segment_index) < len(segment_conds):
        shifted.original_conds['positive'] = segment_conds[int(segment_index)]

    # paired AV continuation: carry the previous RAW video+audio tails together as one
    # native H3 context reference. Keep the existing ordered motion guides as a
    # complementary C1/pose-phase signal; the paired AV ref owns soundtrack phase.
    reconstruction_active = bool(getattr(plan, 'reconstruction_active', False))
    if (
        not reconstruction_active
        and getattr(plan, 'mode', None) != 'storyboard_bridge'
        and int(segment_index) > 0
        and previous_av is not None
    ):
        native_two_pass = (
            (
                getattr(plan, 'mode', None) == 'multiclip' and int(segment_index) > 0
            )
            or (
                int(segment_index) == 1
                and int(getattr(plan, 'passes', 0) or 0) == 2
                and getattr(plan, 'mode', None) == 'segmented_continuation'
            )
        ) and _v83_native_guide_api_supported()
        if native_two_pass:
            motion_positive, _motion_frames = _v83_attach_native_motion_context(
                shifted.original_conds.get('positive', []), previous_av, plan, segment_index,
            )
            shifted.original_conds['positive'] = motion_positive
        else:
            av_positive, _av_frames = _v0322_attach_native_av_context_ref(
                shifted.original_conds.get('positive', []), previous_av, plan, segment_index,
            )
            shifted.original_conds['positive'] = av_positive
            motion_positive, _motion_frames = _v60_attach_previous_head_guides(
                shifted.original_conds.get('positive', []), previous_av, plan, segment_index,
            )
            shifted.original_conds['positive'] = motion_positive

    transformer_options = shifted.model_options.setdefault('transformer_options', {})
    if getattr(plan, 'mode', None) != 'storyboard_bridge':
        transformer_options[TEMPORAL_OFFSET_OPTION] = temporal_offset_for_frame(start_frame)
    else:
        transformer_options.pop(TEMPORAL_OFFSET_OPTION, None)
    if WrappersMP is not None:
        wrappers = transformer_options.setdefault('wrappers', {})
        apply_model = wrappers.setdefault(WrappersMP.APPLY_MODEL, {})
        apply_model['MiniMaxH3LatentLabTemporalOffset'] = [h3_temporal_offset_wrapper]

    # Only source/reference streams that genuinely change per pass require a tiny mutable
    # metadata shell. Hybrid global refs are read-only and remain shared with zero copies.
    reference_audio = getattr(plan, 'reference_audio', None) or plan.source_audio
    positive_list = shifted.original_conds.get('positive', [])
    needs_mutable_refs = bool(
        reference_audio is not None
        or plan.source_video is not None
    )
    if needs_mutable_refs and positive_list:
        cloned_positive = []
        for entry in positive_list:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], dict):
                new_meta = dict(entry[1])
                if 'minimax_refs' in new_meta:
                    new_meta['minimax_refs'] = [dict(ref) for ref in new_meta.get('minimax_refs', [])]
                new_entry = list(entry)
                new_entry[1] = new_meta
                cloned_positive.append(new_entry)
            elif isinstance(entry, dict):
                new_entry = dict(entry)
                if 'minimax_refs' in new_entry:
                    new_entry['minimax_refs'] = [dict(ref) for ref in new_entry.get('minimax_refs', [])]
                cloned_positive.append(new_entry)
            else:
                cloned_positive.append(entry)
        shifted.original_conds['positive'] = cloned_positive
        positive_list = cloned_positive


    refs = []
    if positive_list:
        meta0 = _conditioning_meta(positive_list[0])
        if meta0:
            refs = meta0.get('minimax_refs', []) or []

    audio_timeline_logged = False
    for ref in refs:
        if ref.get('kind') == 'audio' and reference_audio is not None and plan.audio_vae is not None:
            length_frames = int(timeline['length_frames'])
            # V318 timeline contract: the H3 reference stream covers the whole local
            # pass, including hidden overlap. Starting it at visible_start would put
            # the music visible boundary at local t=0 while the target boundary is
            # local t=overlap, creating an overlap-sized phase error on pass 2+.
            audio_start_frame = int(timeline['context_start'])
            if (not audio_timeline_logged) and int(segment_index) > 0:
                _lm_print(
                    '[MiniMaxH3 LongMedia][V318 TIMELINE] '
                    f'segment={int(segment_index)} context_start={int(timeline["context_start"])}f '
                    f'visible_start={int(timeline["visible_start"])}f '
                    f'local_visible_offset={int(timeline["local_visible_offset"])}f '
                    f'audio_ref_window_start={audio_start_frame}f',
                    flush=True,
                )
                audio_timeline_logged = True
            available, _ = _slice_source_audio_for_segment(reference_audio, audio_start_frame, length_frames)
            waveform_for_encode = available.movedim(1, -1)
            audio_lat = plan.audio_vae.encode(waveform_for_encode)
            ref['ref_audio_t'] = audio_lat.shape[-1]
            ref['audio_latent'] = audio_lat
            ref['longmedia_context_start_frame'] = int(timeline['context_start'])
            ref['longmedia_visible_start_frame'] = int(timeline['visible_start'])
            ref['longmedia_local_visible_offset_frames'] = int(timeline['local_visible_offset'])
        elif (
            ref.get('kind') in ('video', 'video_audio')
            and plan.source_video is not None
            and plan.video_vae is not None
        ):
            length_frames = int(timeline['length_frames'])
            source_frames = slice_video_segment(plan.source_video, start_frame, length_frames, plan.video_fps)
            if reconstruction_active:
                target_w = int(getattr(plan, 'reconstruction_target_width', 0) or 0)
                target_h = int(getattr(plan, 'reconstruction_target_height', 0) or 0)
                if target_w <= 0 or target_h <= 0:
                    raise RuntimeError('Reconstruction V5 target canvas contract is missing.')
                source_frames = _reconstruction_fit_source_frames(
                    source_frames, target_w, target_h,
                    str(getattr(plan, 'reconstruction_resize_mode', 'center_crop')),
                )
                ref_h = int(ref.get('latent_h', 0) or 0)
                ref_w = int(ref.get('latent_w', 0) or 0)
                if ref_h <= 0 or ref_w <= 0:
                    latent0 = ref.get('latent')
                    if latent0 is not None and hasattr(latent0, 'shape') and len(latent0.shape) == 5:
                        ref_h = int(latent0.shape[3]); ref_w = int(latent0.shape[4])
                if ref_h <= 0 or ref_w <= 0:
                    raise RuntimeError('Reconstruction V5 source reference is missing latent H/W geometry.')
                # Source-fit already established target composition. This second
                # resize only maps that composition onto Ref2VA's reference canvas.
                ref_frames = _resize_frames(source_frames, ref_w * 16, ref_h * 16, 'stretch')
                ref_latent = plan.video_vae.encode(ref_frames)
                ref['latent'] = ref_latent
                ref['latent_t'] = int(ref_latent.shape[2])
                ref['latent_h'] = int(ref_latent.shape[3])
                ref['latent_w'] = int(ref_latent.shape[4])
            else:
                ref_latent = plan.video_vae.encode(source_frames)
                ref['latent'] = ref_latent
                ref['latent_t'] = int(ref_latent.shape[2])
                ref['latent_h'] = int(ref_latent.shape[3])
                ref['latent_w'] = int(ref_latent.shape[4])
            ref['longmedia_context_start_frame'] = int(timeline['context_start'])
            ref['longmedia_visible_start_frame'] = int(timeline['visible_start'])
            ref['longmedia_local_visible_offset_frames'] = int(timeline['local_visible_offset'])


    # AV Motion Context and per-pass source slicing can add/replace refs after
    # Setup pre-encoding.  Reconcile their declared audio lengths one final
    # time before Comfy builds the H3 PackedLayout for this sampling pass.
    shifted.original_conds['positive'] = _v041_normalize_minimax_audio_ref_geometry(
        shifted.original_conds.get('positive', [])
    )
    return shifted

def _encode_prompt(clip, prompt):
    """Encode a text prompt into CONDITIONING across supported ComfyUI CLIP APIs."""
    tokens = clip.tokenize(prompt)

    # Current ComfyUI's canonical CONDITIONING path.  This keeps hook/LoRA
    # schedules attached to the conditioning and avoids relying on the old
    # MiniMax-era CLIP.encode(..., control=None) signature, which was removed.
    scheduled = getattr(clip, "encode_from_tokens_scheduled", None)
    if callable(scheduled):
        return scheduled(tokens)

    # Compatibility fallback for older ComfyUI builds that exposed the
    # MiniMax-specific control keyword on CLIP.encode().
    try:
        return clip.encode(tokens, control=None)
    except TypeError as exc:
        if "control" not in str(exc):
            raise

    # Last-resort compatibility for CLIP implementations where encode() takes
    # raw text rather than pre-tokenized input.
    return clip.encode(prompt)


class MiniMaxH3LatentLabRuntimeContinuationGuider:
    """Build continuation conditioning after the previous segment exists at runtime."""

    DESCRIPTION = 'Internal runtime continuation guider with real previous-segment motion context.'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'guider': ('GUIDER',),
                'long_media_plan': ('LONG_MEDIA_PLAN',),
                'previous_av': ('LATENT',),
                'segment_index': ('INT', {'default': 1, 'min': 1, 'max': 100, 'step': 1}),
            }
        }

    RETURN_TYPES = ('GUIDER', 'STRING')
    RETURN_NAMES = ('guider', 'report')
    FUNCTION = 'build'
    CATEGORY = CATEGORY_LONGMEDIA

    def build(self, guider, long_media_plan, previous_av, segment_index):
        seg_idx = int(segment_index)
        if not isinstance(previous_av, dict) or 'samples' not in previous_av:
            raise ValueError(
                "V320 runtime continuation handoff expected completed previous LATENT with 'samples'."
            )
        prev_video, _prev_audio = unpack_av_samples(previous_av)
        previous_frames = frame_count_from_video_t(int(prev_video.shape[2]))
        shifted = _clone_guider_with_segment_audio(
            guider, long_media_plan, seg_idx, previous_av=previous_av,
        )
        overlap = int(getattr(long_media_plan, 'overlap_frames', 0) or 0)
        run = next((g for g in (56, 39, 22, 5, 1) if g <= overlap), 0)
        _lm_print(
            '[MiniMaxH3 LongMedia][V320 RUNTIME MOTION HANDOFF] '
            f'segment={seg_idx} previous_frames={previous_frames}f overlap={overlap}f '
            f'guide_span={run}f runtime_previous_latent=yes',
            flush=True,
        )
        report = json.dumps({
            'segment_index': seg_idx,
            'previous_frames': int(previous_frames),
            'overlap_frames': overlap,
            'motion_guide_span_frames': int(run),
            'runtime_previous_latent': True,
        }, indent=2)
        return (shifted, report)


class MiniMaxH3LatentLabVideoEncode:
    DESCRIPTION = (
        'Encode IMAGE frames as a standalone MiniMax H3 video stream. '
        'Connect target_av to force exact H3 canvas and temporal shape '
        'before packing/replacement.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'vae': ('VAE',),
                'frames': ('IMAGE',),
                'frame_fit': (['strict', 'crop_or_pad_last', 'loop'],),
                'resize_mode': (['none', 'stretch', 'center_crop'],),
            },
            'optional': {
                'target_av': (
                    'LATENT',
                    {'tooltip': 'Optional H3 AV latent whose video shape is the target.'},
                ),
            },
        }

    RETURN_TYPES = ('LATENT', 'INT', 'INT', 'INT')
    RETURN_NAMES = ('video_latent', 'frames', 'width', 'height')
    FUNCTION = 'encode'
    CATEGORY = CATEGORY_STREAMS

    def encode(self, vae, frames, frame_fit, resize_mode, target_av=None):
        if target_av is not None:
            target_video, target_count, width, height = _target_video_geometry(target_av)
        else:
            target_video = None
            source_count = int(frames.shape[0])
            if frame_fit == 'strict':
                if not _is_valid_frame_count(source_count):
                    raise ValueError(
                        f'MiniMax H3 frame count must be 17*k+5, got {source_count}.'
                    )
                target_count = source_count
            else:
                target_count = align_frame_count(source_count)
            height = int(frames.shape[1])
            width = int(frames.shape[2])
            if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
                if resize_mode == 'none':
                    raise ValueError(
                        f'H3 canvas must be divisible by 32, got {width}x{height}.'
                    )
                width = max(CANVAS_MULTIPLE, round(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                height = max(CANVAS_MULTIPLE, round(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        fitted = _fit_frames(frames, target_count, frame_fit)
        fitted = _resize_frames(fitted, width, height, resize_mode)
        latent = vae.encode(fitted)
        _validate_video(latent)
        if target_video is not None and tuple(latent.shape) != tuple(target_video.shape):
            raise ValueError(
                f'Video VAE produced {tuple(latent.shape)}, '
                f'target AV requires {tuple(target_video.shape)}.'
            )
        return ({'samples': latent}, target_count, width, height)


class MiniMaxH3LatentLabAudioEncode:
    DESCRIPTION = (
        'Resample and encode AUDIO as a standalone MiniMax H3 audio stream. '
        'target_av makes the encoded stream exactly match the target duration.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'audio_vae': ('VAE',),
                'audio': ('AUDIO',),
                'fit_mode': (['strict', 'crop_or_pad_silence', 'loop'],),
            },
            'optional': {
                'target_av': (
                    'LATENT',
                    {'tooltip': 'Optional H3 AV latent whose audio shape is the target.'},
                ),
            },
        }

    RETURN_TYPES = ('LATENT', 'FLOAT', 'INT')
    RETURN_NAMES = ('audio_latent', 'duration_seconds', 'sample_rate')
    FUNCTION = 'encode'
    CATEGORY = CATEGORY_STREAMS

    def encode(self, audio_vae, audio, fit_mode, target_av=None):
        waveform = audio['waveform'][:1]
        source_rate = int(audio['sample_rate'])
        vae_rate = int(getattr(audio_vae, 'audio_sample_rate', 32000))
        if source_rate != vae_rate:
            waveform = torchaudio.functional.resample(waveform, source_rate, vae_rate)
        target_audio = None
        if target_av is not None:
            _, target_audio = unpack_av_samples(target_av)
            target_samples = round(target_audio.shape[-1] / AUDIO_LATENT_FPS * vae_rate)
            waveform = _fit_waveform(waveform, target_samples, fit_mode)
        latent = audio_vae.encode(waveform.movedim(1, -1))
        if target_audio is not None:
            if fit_mode == 'strict':
                latent = _fit_stream(latent, target_audio, 'audio', 'strict', 'start')
            else:
                latent = _fit_stream(latent, target_audio, 'audio', 'crop_pad', 'start')
        duration = latent.shape[-1] / AUDIO_LATENT_FPS
        return ({'samples': latent}, float(duration), vae_rate)



class MiniMaxH3LatentLabPackAV:
    DESCRIPTION = (
        'Pack MiniMax H3 video/audio streams back into AV latent form. '
        'Automatically rebuilds LongMedia MultiClip native containers when the '
        'streams originated from Split AV Streams; ordinary streams remain ordinary AV.'
    )

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'video_latent': ('LATENT',),
                'audio_latent': ('LATENT',),
            }
        }

    RETURN_TYPES = ('LATENT',)
    RETURN_NAMES = ('av_latent',)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = 'pack'
    CATEGORY = CATEGORY_STREAMS

    @staticmethod
    def _bridge_source(video, audio):
        for item in (video, audio):
            if isinstance(item, dict) and (
                item.get('_lm_av_stream_bridge') or item.get('_lm_av_list_bridge')
            ):
                return item
        return {}

    @staticmethod
    def _clean_bridge_meta(latent):
        if not isinstance(latent, dict):
            return latent
        out = dict(latent)
        for key in (
            '_lm_av_stream_bridge',
            '_lm_av_stream_bridge_multiclip',
            '_lm_av_stream_bridge_container',
            '_lm_av_stream_bridge_index',
            '_lm_av_stream_bridge_count',
            '_lm_av_list_bridge',
            '_lm_av_list_bridge_multiclip',
            '_lm_av_list_bridge_container',
            '_lm_av_list_bridge_index',
            '_lm_av_list_bridge_count',
        ):
            out.pop(key, None)
        return out

    @classmethod
    def _bridge_index(cls, pair):
        video, audio = pair
        meta = cls._bridge_source(video, audio)
        return int(
            meta.get(
                '_lm_av_stream_bridge_index',
                meta.get('_lm_av_list_bridge_index', 0),
            ) or 0
        )

    def pack(self, video_latent, audio_latent):
        videos = list(video_latent or [])
        audios = list(audio_latent or [])
        if len(videos) != len(audios):
            raise ValueError(
                f'Pack AV Streams requires matching video/audio stream counts, '
                f'got {len(videos)} and {len(audios)}.'
            )
        if not videos:
            raise ValueError('Pack AV Streams received no stream items.')

        pairs = list(zip(videos, audios))
        pairs.sort(key=self._bridge_index)

        packed_segments = [
            pack_av_latents(
                self._clean_bridge_meta(video),
                self._clean_bridge_meta(audio),
                NestedTensor,
            )
            for video, audio in pairs
        ]

        first_meta = self._bridge_source(*pairs[0])
        is_multiclip = bool(
            first_meta.get(
                '_lm_av_stream_bridge_multiclip',
                first_meta.get('_lm_av_list_bridge_multiclip', False),
            )
        )
        container_meta = (
            first_meta.get('_lm_av_stream_bridge_container')
            or first_meta.get('_lm_av_list_bridge_container')
            or {}
        )

        if not is_multiclip:
            return (packed_segments[0],)

        expected_count = int(
            first_meta.get(
                '_lm_av_stream_bridge_count',
                first_meta.get('_lm_av_list_bridge_count', len(packed_segments)),
            ) or len(packed_segments)
        )
        if expected_count != len(packed_segments):
            raise ValueError(
                f'Pack AV Streams expected {expected_count} MultiClip items but '
                f'received {len(packed_segments)}.'
            )

        output = dict(container_meta)
        output['samples'] = packed_segments[0]['samples']
        if 'noise_mask' in packed_segments[0]:
            output['noise_mask'] = packed_segments[0]['noise_mask']
        output['_lm_per_clip_native_video_decode'] = True
        output['_lm_segment_latents'] = packed_segments
        output['_lm_external_av_stream_transform'] = True
        output['_lm_external_av_stream_clip_count'] = len(packed_segments)
        return (output,)


class MiniMaxH3LatentLabSplitAV:
    DESCRIPTION = (
        'Split MiniMax H3 AV latents into editable video and audio streams. '
        'Automatically expands LongMedia MultiClip containers into their native '
        'per-clip latents; ordinary AV latents are handled as a one-item stream.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'av_latent': ('LATENT',)}}

    RETURN_TYPES = ('LATENT', 'LATENT')
    RETURN_NAMES = ('video_latent', 'audio_latent')
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = 'split'
    CATEGORY = CATEGORY_STREAMS

    @staticmethod
    def _bridge_meta(*, multiclip, index, count, container_meta):
        return {
            '_lm_av_stream_bridge': True,
            '_lm_av_stream_bridge_multiclip': bool(multiclip),
            '_lm_av_stream_bridge_container': container_meta,
            '_lm_av_stream_bridge_index': int(index),
            '_lm_av_stream_bridge_count': int(count),
        }

    def split(self, av_latent):
        segments = None
        if (
            isinstance(av_latent, dict)
            and bool(av_latent.get('_lm_per_clip_native_video_decode'))
            and isinstance(av_latent.get('_lm_segment_latents'), (list, tuple))
        ):
            candidate = [
                seg for seg in av_latent.get('_lm_segment_latents')
                if isinstance(seg, dict) and seg.get('samples') is not None
            ]
            if candidate:
                segments = candidate

        if not segments:
            video, audio = split_av_latent(av_latent)
            container_meta = {
                k: v for k, v in av_latent.items()
                if k not in {'samples', 'noise_mask'}
            } if isinstance(av_latent, dict) else {}
            bridge = self._bridge_meta(
                multiclip=False, index=0, count=1, container_meta=container_meta
            )
            video = {**video, **bridge}
            audio = {**audio, **bridge}
            return ([video], [audio])

        container_meta = {
            k: v for k, v in av_latent.items()
            if k not in {'samples', 'noise_mask', '_lm_segment_latents'}
        }
        videos, audios = [], []
        count = len(segments)
        for idx, seg in enumerate(segments):
            video, audio = split_av_latent(seg)
            video.pop('_lm_segment_latents', None)
            audio.pop('_lm_segment_latents', None)
            bridge = self._bridge_meta(
                multiclip=True, index=idx, count=count, container_meta=container_meta
            )
            video.update(bridge)
            audio.update(bridge)
            videos.append(video)
            audios.append(audio)
        return (videos, audios)


class MiniMaxH3LatentLabReplaceStream:
    """Replace one whole stream (video or audio) in an H3 AV latent.

    FIXED_KIND is None on the primary node (stream picked via widget) and
    'video'/'audio' on the two legacy subclasses kept below so old saved
    graphs (which have no 'stream' widget) keep loading and running.
    """

    FIXED_KIND = None
    DESCRIPTION = (
        'Replace the video or audio stream in an H3 AV latent. '
        'strict is lossless; crop_pad center-crops/pads spatially '
        'and uses alignment for time.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            'av_latent': ('LATENT',),
            'replacement': ('LATENT',),
            'fit_mode': (['strict', 'crop_pad'],),
            'alignment': (['start', 'end', 'center'],),
            'denoise': (
                'FLOAT',
                {
                    'default': 0.0,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'tooltip': '0 preserves the replacement exactly; 1 fully denoises it.',
                },
            ),
        }
        if cls.FIXED_KIND is None:
            required['stream'] = (['video', 'audio'],)
        return {'required': required}

    RETURN_TYPES = ('LATENT',)
    RETURN_NAMES = ('av_latent',)
    FUNCTION = 'replace'
    CATEGORY = CATEGORY_STREAMS

    def replace(self, av_latent, replacement, fit_mode, alignment, denoise, stream=None):
        kind = self.FIXED_KIND or stream
        return (
            replace_stream(av_latent, replacement, kind, fit_mode, alignment, denoise, NestedTensor),
        )


class MiniMaxH3LatentLabReplaceVideo(MiniMaxH3LatentLabReplaceStream):
    FIXED_KIND = 'video'
    DESCRIPTION = (
        'Deprecated — kept only so old saved graphs keep loading. '
        'Use "MiniMax H3 \u2022 Replace Stream" for new graphs (stream=video).'
    )


class MiniMaxH3LatentLabReplaceAudio(MiniMaxH3LatentLabReplaceStream):
    FIXED_KIND = 'audio'
    DESCRIPTION = (
        'Deprecated — kept only so old saved graphs keep loading. '
        'Use "MiniMax H3 \u2022 Replace Stream" for new graphs (stream=audio).'
    )


class MiniMaxH3LatentLabStreamDenoise:
    DESCRIPTION = (
        "Independent H3 stream control through ComfyUI's stock noise_mask path. "
        '0 preserves a stream; 1 fully regenerates it.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'av_latent': ('LATENT',),
                'video_denoise': (
                    'FLOAT',
                    {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'audio_denoise': (
                    'FLOAT',
                    {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'merge_mode': (['replace', 'multiply', 'minimum', 'maximum'],),
            }
        }

    RETURN_TYPES = ('LATENT',)
    RETURN_NAMES = ('av_latent',)
    FUNCTION = 'control'
    CATEGORY = CATEGORY_STREAMS

    def control(self, av_latent, video_denoise, audio_denoise, merge_mode):
        return (
            set_stream_denoise(
                av_latent, video_denoise, audio_denoise, merge_mode, NestedTensor
            ),
        )


class MiniMaxH3LatentLabLipSyncSetup:
    DESCRIPTION = (
        'Replace the native H3 audio stream and configure independent '
        'video/audio denoise controls for lip-sync-oriented sampling.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'av_latent': ('LATENT',),
                'audio_latent': ('LATENT',),
                'fit_mode': (['strict', 'crop_pad'],),
                'alignment': (['start', 'end', 'center'],),
                'video_denoise': (
                    'FLOAT',
                    {'default': 0.35, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'audio_denoise': (
                    'FLOAT',
                    {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
            }
        }

    RETURN_TYPES = ('LATENT',)
    RETURN_NAMES = ('av_latent',)
    FUNCTION = 'setup'
    CATEGORY = CATEGORY_UTIL

    def setup(self, av_latent, audio_latent, fit_mode, alignment, video_denoise, audio_denoise):
        replaced = replace_stream(
            av_latent, audio_latent, 'audio', fit_mode, alignment, audio_denoise, NestedTensor
        )
        return (
            set_stream_denoise(replaced, video_denoise, audio_denoise, 'replace', NestedTensor),
        )


class MiniMaxH3LatentLabVideoInpaint:
    DESCRIPTION = (
        'Map a ComfyUI MASK onto the H3 video latent grid. '
        'White uses denoise_inside, black uses denoise_outside; '
        'audio is controlled separately.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'av_latent': ('LATENT',),
                'mask': ('MASK',),
                'denoise_inside': (
                    'FLOAT',
                    {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'denoise_outside': (
                    'FLOAT',
                    {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'audio_denoise': (
                    'FLOAT',
                    {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'merge_mode': (['replace', 'multiply', 'minimum', 'maximum'],),
            }
        }

    RETURN_TYPES = ('LATENT',)
    RETURN_NAMES = ('av_latent',)
    FUNCTION = 'apply'
    CATEGORY = CATEGORY_STREAMS

    def apply(self, av_latent, mask, denoise_inside, denoise_outside, audio_denoise, merge_mode):
        return (
            apply_video_inpaint_mask(
                av_latent, mask, denoise_inside, denoise_outside, audio_denoise, merge_mode,
                NestedTensor,
            ),
        )


class MiniMaxH3LatentLabMergeAV:
    DESCRIPTION = (
        'Blend video and audio independently from a source H3 AV latent '
        'into a target H3 AV latent. The target defines output geometry.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'target_av': ('LATENT',),
                'source_av': ('LATENT',),
                'video_mix': (
                    'FLOAT',
                    {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'audio_mix': (
                    'FLOAT',
                    {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'fit_mode': (['strict', 'crop_pad'],),
                'alignment': (['start', 'end', 'center'],),
                'video_denoise': (
                    'FLOAT',
                    {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'audio_denoise': (
                    'FLOAT',
                    {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
            }
        }

    RETURN_TYPES = ('LATENT',)
    RETURN_NAMES = ('av_latent',)
    FUNCTION = 'merge'
    CATEGORY = CATEGORY_STREAMS

    def merge(self, target_av, source_av, video_mix, audio_mix, fit_mode, alignment, video_denoise, audio_denoise):
        return (
            merge_av_latents(
                target_av, source_av, video_mix, audio_mix, fit_mode, alignment,
                video_denoise, audio_denoise, NestedTensor,
            ),
        )


class MiniMaxH3LatentLabPrepareContinuation:
    DESCRIPTION = (
        'Create a new H3 AV latent whose opening is the synchronized tail '
        'of a previous result. Sample it, then use Stitch Continuation to '
        'remove overlap.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'source_av': ('LATENT',),
                'length': (
                    'INT',
                    {'default': 124, 'min': 5, 'max': 3600, 'step': 17},
                ),
                'overlap_frames': (
                    'INT',
                    {
                        'default': 22,
                        'min': 5,
                        'max': 3600,
                        'step': 17,
                        'tooltip': 'Snapped down to the 17*k+5 H3 grid.',
                    },
                ),
                'video_context_denoise': (
                    'FLOAT',
                    {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'audio_context_denoise': (
                    'FLOAT',
                    {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
            }
        }

    RETURN_TYPES = ('LATENT', 'INT', 'INT', 'FLOAT')
    RETURN_NAMES = ('continuation_av', 'frame_count', 'actual_overlap_frames', 'overlap_seconds')
    FUNCTION = 'prepare'
    CATEGORY = CATEGORY_CONTINUATION

    def prepare(self, source_av, length, overlap_frames, video_context_denoise, audio_context_denoise):
        output, frame_count, actual_overlap = prepare_continuation(
            source_av, length, overlap_frames, video_context_denoise, audio_context_denoise,
            NestedTensor,
        )
        return (output, frame_count, actual_overlap, actual_overlap / FPS)


class MiniMaxH3LatentLabStitchContinuation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'previous_av': ('LATENT',),
                'sampled_continuation_av': ('LATENT',),
                'overlap_frames': (
                    'INT',
                    {'default': 22, 'min': 5, 'max': 3600, 'step': 17},
                ),
            },
            'optional': {
                'blend_video_overlap': (
                    'BOOLEAN',
                    {
                        'default': False,
                        'tooltip': 'Smoothstep blend the video overlap seam.',
                    },
                ),
                'offload_to_cpu': (
                    'BOOLEAN',
                    {
                        'default': False,
                        'tooltip': (
                            'Move the stitched result to CPU RAM instead of leaving it '
                            'on the GPU. Safe to enable for long multi-pass runs: this '
                            'accumulator is never read back by the sampler, only by the '
                            'next stitch and the final decode, so keeping it resident on '
                            'the GPU across many passes just wastes VRAM.'
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ('LATENT', 'INT')
    RETURN_NAMES = ('stitched_av', 'total_frames')
    FUNCTION = 'stitch'
    CATEGORY = CATEGORY_CONTINUATION

    def stitch(self, previous_av, sampled_continuation_av, overlap_frames,
               blend_video_overlap=False, offload_to_cpu=False):
        prev_video, prev_audio = unpack_av_samples(previous_av)
        next_video, next_audio = unpack_av_samples(sampled_continuation_av)
        prev_frames = frame_count_from_video_t(prev_video.shape[2])
        next_frames = frame_count_from_video_t(next_video.shape[2])
        result = stitch_continuation(
            previous_av, sampled_continuation_av, overlap_frames, NestedTensor,
            blend_video_overlap, bool(offload_to_cpu),
        )
        stitched_av, reported_total_frames = result
        stitched_video, stitched_audio = unpack_av_samples(stitched_av)
        stitched_frames = frame_count_from_video_t(stitched_video.shape[2])
        expected_audio_t = audio_latent_t(stitched_frames)
        actual_overlap = prev_frames + next_frames - int(reported_total_frames)
        if stitched_frames != int(reported_total_frames):
            raise RuntimeError(
                '[V318 BOUNDARY AUDIT] stitched video frame mismatch: '
                f'actual={stitched_frames}, expected={int(reported_total_frames)}, '
                f'previous={prev_frames}, next={next_frames}, overlap={actual_overlap}'
            )
        if int(stitched_audio.shape[-1]) != int(expected_audio_t):
            raise RuntimeError(
                '[V318 BOUNDARY AUDIT] stitched AV sync mismatch: '
                f'audio_t={int(stitched_audio.shape[-1])}, expected_audio_t={int(expected_audio_t)}, '
                f'frames={stitched_frames}'
            )
        visible_seam_latent = 0
        _lm_print(
            '[MiniMaxH3 LongMedia][V327 PHASE-SAFE BOUNDARY AUDIT] PASS '
            f'previous={prev_frames}f next={next_frames}f overlap={actual_overlap}f '
            f'stitched={stitched_frames}f audio_t={int(stitched_audio.shape[-1])} '
            f'cross_time_blend_latent_t={visible_seam_latent}',
            flush=True,
        )
        if offload_to_cpu:
            _free_cuda_memory()
        return result


class MiniMaxH3LatentLabInfo:
    DESCRIPTION = 'Validate and inspect a MiniMax H3 NestedTensor AV latent.'
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'av_latent': ('LATENT',)}}

    RETURN_TYPES = ('STRING', 'INT', 'INT', 'INT', 'FLOAT', 'BOOLEAN')
    RETURN_NAMES = ('report', 'width', 'height', 'frames', 'duration_seconds', 'synchronized')
    FUNCTION = 'inspect'
    CATEGORY = CATEGORY_UTIL

    def inspect(self, av_latent):
        info = describe_av(av_latent)
        report = json.dumps(info, indent=2, ensure_ascii=False)
        return (
            report, info['width'], info['height'], info['frames'],
            info['duration_seconds'], info['synchronized'],
        )



# -----------------------------------------------------------------------------
# V43/V64: self-contained hybrid conditioning + ref-aware anchor placement
# -----------------------------------------------------------------------------

# Shared Motion Context / Contex Loop marker: target-timeline pixel frame an
# anchor belongs to. The marker-gated PackedLayout helper already ships with
# LongMedia and only moves conditioning rows carrying this key.
MC_ANCHOR_KEY = 'motion_context_index'

def _activate_longmedia_hybrid_support():
    """Activate LongMedia's self-contained keyframe+Ref2VA payload merge.

    This fixes the stock ``cond_video_latents`` last-writer-wins collision.
    Keyframe positions additionally need the marker-gated PackedLayout helper
    whenever refs are packed before the target timeline.
    If Contex Loop/Motion Context already owns the compatible payload merge,
    the shared marker makes this helper stand down cleanly.
    """
    try:
        from . import hybrid_payload_patch
    except Exception:
        import importlib
        pkg = __package__
        if not pkg:
            raise RuntimeError('LongMedia hybrid payload helper could not be imported')
        hybrid_payload_patch = importlib.import_module(pkg + '.hybrid_payload_patch')
    if not hybrid_payload_patch.apply_patch():
        raise RuntimeError(
            'LongMedia hybrid payload merge could not activate. Check the console '
            'for a MiniMaxH3.extra_conds patch collision.'
        )
    return hybrid_payload_patch


def _activate_longmedia_anchor_layout():
    """Keep hybrid first/last anchors on the ref-shifted target timeline.

    Stock H3 places keyframe conditioning rows relative to text_len, while the
    actual target video grid starts after all packed reference rows. Stock H3
    normally keeps FL2VA keyframes and Ref2VA references in separate workflows,
    so it never has to reconcile those origins. LongMedia hybrid/loop combines
    them, therefore refs can otherwise push both anchors into the past.

    The existing marker-gated ``motion_context_layout_patch`` builds marked
    keyframe rows legally, then translates only those rows to the true target
    origin. It is activated only when keyframes and refs coexist.
    """
    try:
        from . import motion_context_layout_patch
    except Exception:
        import importlib
        pkg = __package__
        if not pkg:
            raise RuntimeError('LongMedia anchor layout helper could not be imported')
        motion_context_layout_patch = importlib.import_module(
            pkg + '.motion_context_layout_patch'
        )
    if not motion_context_layout_patch.apply_patch():
        raise RuntimeError(
            'LongMedia keyframe anchors require the PackedLayout patch whenever '
            'references are connected. Check the console for a PackedLayout '
            'patch collision.'
        )
    return motion_context_layout_patch


def _hybrid_encode_ref_audio(audio_vae, audio):
    waveform = audio['waveform']
    sr = int(audio['sample_rate'])
    vae_sr = int(getattr(audio_vae, 'audio_sample_rate', 32000))
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    return z, int(z.shape[-1])


def _build_longmedia_hybrid_conditioning(
    clip, vae, audio_vae, prompt, width, height, length, resolution_mode,
    first_frame=None, last_frame=None, ref_images=None, ref_videos=None,
    ref_audios=None, ref_video_audios=None, first_latent_override=None, last_latent_override=None,
):
    """Build H3 keyframes + Ref2VA references in one conditioning payload.

    image_1/image_2 role assignment is handled by Setup; this helper receives
    only the already-separated anchors and reference lists.
    """
    payload_patch = _activate_longmedia_hybrid_support()
    try:
        from comfy_extras.nodes_minimax_h3 import (
            _empty_av_latent, _resize, adapt_canvas,
            REF_IMAGE_SHORT_EDGE, FPS as H3_FPS,
        )
        import node_helpers
    except Exception as exc:
        raise RuntimeError('Current ComfyUI MiniMax H3 helpers are unavailable: %s' % exc)

    latent, frame_count = _empty_av_latent(int(width), int(height), int(length))
    mc_key = getattr(payload_patch, 'LM_KEY', 'longmedia_hybrid_keyframe')
    native_guides = True

    keyframes = []
    keyframe_images = []

    def add_keyframe(frame_index, image, crop, latent_override=None):
        if image is None:
            return None
        img = _resize(image[:1], int(width), int(height), crop)
        resolved = int(frame_index) if native_guides else 0
        entry = {
            'resolved_frame_index': resolved,
            mc_key: True,
            # Target-timeline pixel frame for the anchor. Without refs this is
            # identical to stock placement; with refs the layout patch uses it
            # to compensate for the packed reference span.
            MC_ANCHOR_KEY: int(frame_index),
            'latent': latent_override if latent_override is not None else vae.encode(img),
        }
        keyframes.append(entry)
        keyframe_images.append(img)
        return entry

    first_keyframe = add_keyframe(0, first_frame, 'disabled', first_latent_override)
    add_keyframe(frame_count - 1, last_frame, 'center', last_latent_override)

    ref_items = []
    ref_blocks = []

    for img in ref_images or []:
        if img is None:
            continue
        h, w = int(img.shape[1]), int(img.shape[2])
        if resolution_mode == 'match':
            scale = min(1.0, math.sqrt((int(width) * int(height)) / float(w * h)))
        else:
            scale = min(1.0, float(REF_IMAGE_SHORT_EDGE) / float(min(w, h)))
        tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        resized = _resize(img[:1], tw, th, 'disabled')
        ref_items.append({'type': 'image', 'data': resized})
        ref_blocks.append({
            'kind': 'image', 'latent_h': th // 16, 'latent_w': tw // 16,
            'latent': vae.encode(resized),
        })

    paired_video_audios = list(ref_video_audios or [])
    for video_index, video_frames in enumerate(ref_videos or []):
        if video_frames is None:
            continue
        paired_soundtrack = (
            paired_video_audios[video_index]
            if video_index < len(paired_video_audios)
            else None
        )
        vh, vw = int(video_frames.shape[1]), int(video_frames.shape[2])
        cw, ch = adapt_canvas(vw, vh)
        if vw * vh < cw * ch:
            cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        frames = _resize(video_frames, cw, ch, 'disabled')
        if frames.shape[0] > frame_count:
            frames = frames[:frame_count]
        n = int(frames.shape[0])
        if n < 5:
            raise ValueError('MiniMax H3 reference videos need at least 5 frames.')
        while n % 17 != 5:
            n -= 1
        frames = frames[:n]
        sample_step = max(1, int(H3_FPS) // 2)
        sample_idx = list(range(0, int(frames.shape[0]), sample_step))
        paired_audio_latent = None
        paired_audio_t = 0
        if paired_soundtrack is not None:
            paired_audio_latent, paired_audio_t = _hybrid_encode_ref_audio(
                audio_vae, paired_soundtrack
            )
            # Match upstream MiniMaxH3ReferenceToVideo ordering exactly: the
            # soundtrack gets its own <Audio j> presentation item immediately
            # before the same-numbered <Video k>, while both tensors live in one
            # `video_audio` reference block and therefore share one time span.
            ref_items.append({'type': 'audio'})
        ref_items.append({
            'type': 'video', 'data': frames[sample_idx],
            'timestamps': [i / 2.0 for i in range(len(sample_idx))],
        })
        video_latent = vae.encode(frames)
        ref_blocks.append({
            'kind': ('video_audio' if paired_audio_t else 'video'),
            'latent_t': int(video_latent.shape[2]),
            'latent_h': ch // 16, 'latent_w': cw // 16,
            'ref_audio_t': int(paired_audio_t),
            'latent': video_latent, 'audio_latent': paired_audio_latent,
        })

    for audio in ref_audios or []:
        if audio is None:
            continue
        audio_latent, ref_audio_t = _hybrid_encode_ref_audio(audio_vae, audio)
        ref_items.append({'type': 'audio'})
        ref_blocks.append({
            'kind': 'audio', 'ref_audio_t': ref_audio_t,
            'audio_latent': audio_latent,
        })

    # Refs occupy packed timeline rows before the target video. When first/last
    # anchors coexist with refs, stock keyframe coordinates are therefore early
    # by the total reference span. Activate the marker-gated layout adjustment
    # only for that combined case; graphs without refs remain completely stock.
    if keyframes and ref_blocks:
        _activate_longmedia_anchor_layout()

    # Same semantics as upstream minimax-h3-hybrid-cond: refs use the H3
    # minimax_ref_items tokenizer path; keyframes live in payload guides.
    if ref_items:
        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    else:
        tokens = clip.tokenize(prompt, images=keyframe_images)
    scheduled = getattr(clip, 'encode_from_tokens_scheduled', None)
    cond = _setup_clip_encode_retry(
        lambda: (scheduled(tokens) if callable(scheduled) else clip.encode(tokens)),
        label='hybrid_conditioning',
    )

    values = {}
    if keyframes:
        values['minimax_keyframes'] = keyframes
        values['minimax_frame_count'] = frame_count
    if ref_blocks:
        values['minimax_refs'] = ref_blocks
    if values:
        cond = node_helpers.conditioning_set_values(cond, values)
    return cond, latent, {
        'keyframes': len(keyframes),
        'image_refs': len(ref_images or []),
        'video_refs': len(ref_videos or []),
        'audio_refs': len(ref_audios or []),
        'paired_video_audio_refs': sum(1 for a in paired_video_audios if a is not None),
        'native_guides': native_guides,
    }, {
        'ref_items': tuple(ref_items),
        'ref_blocks': tuple(ref_blocks),
        'first_keyframe_latent': (
            first_keyframe.get('latent') if first_keyframe is not None else None
        ),
    }


def _clone_positive_with_loop_keyframes(positive_list, first_latent, last_latent, frame_count: int):
    """Reuse the existing prompt/refs but swap in closure-specific anchors.

    This is the native loop-closure contract: keep the already-encoded text and
    any Ref2VA references, then ask H3 to re-solve only the tail segment between
    the actual tail-start frame and the movie's real opening frame.
    """
    if positive_list is None:
        raise RuntimeError('Loop closure requires a positive conditioning payload.')
    payload_patch = _activate_longmedia_hybrid_support()
    mc_key = getattr(payload_patch, 'LM_KEY', 'longmedia_hybrid_keyframe')
    keyframes = [
        {
            'resolved_frame_index': 0,
            mc_key: True,
            MC_ANCHOR_KEY: 0,
            'latent': first_latent,
        },
        {
            'resolved_frame_index': max(0, int(frame_count) - 1),
            mc_key: True,
            MC_ANCHOR_KEY: max(0, int(frame_count) - 1),
            'latent': last_latent,
        },
    ]
    out = []
    attached = False
    has_refs = False
    for entry in (positive_list or []):
        if isinstance(entry, dict):
            meta = dict(entry)
            has_refs = has_refs or bool(meta.get('minimax_refs'))
            meta['minimax_keyframes'] = [dict(kf) for kf in keyframes]
            meta['minimax_frame_count'] = int(frame_count)
            out.append(meta)
            attached = True
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], dict):
            new_entry = list(entry)
            meta = dict(entry[1])
            has_refs = has_refs or bool(meta.get('minimax_refs'))
            meta['minimax_keyframes'] = [dict(kf) for kf in keyframes]
            meta['minimax_frame_count'] = int(frame_count)
            new_entry[1] = meta
            out.append(new_entry)
            attached = True
        else:
            out.append(entry)
    if not attached:
        raise RuntimeError('Loop closure could not attach H3 keyframes to the positive conditioning payload.')
    if has_refs:
        _activate_longmedia_anchor_layout()
    return out, has_refs



def _v111_build_fixed_clip_specs(total_duration, segment_seconds, overlap_frames, prompt):
    """Build fixed-duration clip specs for the unified clip executor.

    Every generated pass has identical H3-aligned length.  The final pass is
    intentionally generated at full length and trimmed after stitching; this
    keeps tensor geometry, Motion Context, lip-sync slicing and memory behavior
    identical across all passes.
    """
    output_frames = max(1, int(math.floor(float(total_duration) * float(FPS))))
    overlap = int(overlap_frames)
    # segment_seconds keeps its historical meaning: NEW visible timeline.  The
    # fixed clip itself also carries the hidden overlap context, so every pass
    # uses one equal full clip length while its visible stride stays close to
    # segment_seconds (modulo H3 temporal alignment).
    visible_frames = max(1, round(float(segment_seconds) * float(FPS)))
    clip_frames = int(align_frame_count(max(5, int(visible_frames) + overlap)))
    if clip_frames <= overlap:
        raise ValueError(
            f"Fixed segmentation clip length ({clip_frames}f) must be greater than overlap_frames={overlap}."
        )
    step = int(clip_frames - overlap)
    if output_frames <= clip_frames:
        passes = 1
    else:
        passes = 1 + int(math.ceil(float(output_frames - clip_frames) / float(step)))
    if passes > 64:
        raise ValueError(f"Fixed segmentation requires {passes} clips; maximum is 64. Increase segment duration.")
    duration = float(clip_frames) / float(FPS)
    specs = tuple({'prompt': str(prompt or ''), 'duration': duration, 'seed': None} for _ in range(passes))
    lengths = tuple(clip_frames for _ in range(passes))
    starts = tuple(int(i * step) for i in range(passes))
    generated = int(clip_frames + max(0, passes - 1) * step)
    return specs, lengths, starts, generated



_V046_MULTICLIP_HEADER_RE = re.compile(
    r"^\[?(?:clip|shot)[ _-]*(\d{1,2})\]?(?:\s*\([^)]*\))?\s*(?:(?::|=|[-–—])\s*)?(.*)$",
    re.IGNORECASE,
)
_V046_MULTICLIP_XML_RE = re.compile(
    r"<(?:clip|shot)[ _-]*(\d{1,2})\b[^>]*>(.*?)</(?:clip|shot)[ _-]*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)


def _v046_strip_prompt_fence(raw):
    text = str(raw or '').replace('\r\n', '\n').replace('\r', '\n').strip()
    m = re.match(r'^```(?:json|yaml|yml|text|markdown|md)?\s*\n(.*?)\n```\s*$', text, re.I | re.S)
    return m.group(1).strip() if m else text


def _v046_prompt_body(lines):
    body = [str(x) for x in (lines or [])]
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    if not body:
        return ''

    prompt_idx = None
    prompt_inline = ''
    prompt_re = re.compile(r'^\s*(?:[-*+]\s*)?prompt\s*:\s*(?:[|>][-+]?)?\s*(.*)$', re.I)
    for i, line in enumerate(body):
        m = prompt_re.match(line)
        if m:
            prompt_idx = i
            prompt_inline = str(m.group(1) or '').strip()
            break

    meta_re = re.compile(r'^\s*(?:duration|seed)\s*:\s*.*$', re.I)
    if prompt_idx is not None:
        selected = body[prompt_idx + 1:]
        if prompt_inline:
            selected.insert(0, prompt_inline)
        while selected and meta_re.match(selected[-1]):
            selected.pop()
        body = selected
    else:
        while body and meta_re.match(body[0]):
            body.pop(0)
        while body and meta_re.match(body[-1]):
            body.pop()

    nonempty = [line for line in body if line.strip()]
    if nonempty:
        min_indent = min(len(line) - len(line.lstrip()) for line in nonempty)
        if min_indent > 0:
            body = [line[min_indent:] for line in body]
    return '\n'.join(body).strip()


def _v046_validate_sections(sections):
    if len(sections) < 2:
        return tuple()
    max_idx = max(sections)
    expected = list(range(1, max_idx + 1))
    if sorted(sections) != expected:
        missing = [str(i) for i in expected if i not in sections]
        raise ValueError('MultiClip prompt sections must be contiguous from 1; missing: ' + ', '.join(missing))
    return tuple(str(sections[i] or '').strip() for i in expected)


def _v046_try_json_prompt(text):
    try:
        value = json.loads(text)
    except Exception:
        return tuple()

    entries = None
    if isinstance(value, list):
        entries = value
    elif isinstance(value, dict) and isinstance(value.get('clips'), list):
        entries = value.get('clips')

    if entries is not None:
        prompts = []
        for item in entries[:16]:
            if isinstance(item, str):
                prompts.append(item.strip())
            elif isinstance(item, dict):
                prompts.append(str(item.get('prompt') or item.get('text') or item.get('description') or '').strip())
            else:
                prompts.append('')
        return tuple(prompts) if len(prompts) >= 2 else tuple()

    if isinstance(value, dict):
        sections = {}
        for key, item in value.items():
            m = re.match(r'^(?:clip|shot)[ _-]*(\d{1,2})$', str(key), re.I)
            if not m:
                continue
            idx = int(m.group(1))
            if idx < 1 or idx > 16:
                raise ValueError(f'MultiClip prompt section index {idx} is outside the supported 1..16 range.')
            if idx in sections:
                raise ValueError(f'MultiClip prompt contains duplicate clip/shot section {idx}.')
            if isinstance(item, dict):
                item = item.get('prompt') or item.get('text') or item.get('description') or ''
            sections[idx] = str(item or '').strip()
        return _v046_validate_sections(sections) if sections else tuple()
    return tuple()


def _v046_try_xml_prompt(text):
    sections = {}
    for match in _V046_MULTICLIP_XML_RE.finditer(text):
        idx = int(match.group(1))
        if idx < 1 or idx > 16:
            raise ValueError(f'MultiClip prompt section index {idx} is outside the supported 1..16 range.')
        if idx in sections:
            raise ValueError(f'MultiClip prompt contains duplicate clip/shot section {idx}.')
        sections[idx] = _v046_prompt_body(str(match.group(2) or '').split('\n'))
    return _v046_validate_sections(sections) if sections else tuple()


def _v046_header(line):
    s = re.sub(r'^\s{0,3}#{1,6}\s*', '', str(line or '')).strip()
    s = re.sub(r'^\s*(?:[-+]\s+)(?=(?:\*\*)?\[?(?:clip|shot)\b)', '', s, flags=re.I)
    s = s.replace('**', '').replace('`', '').strip()
    m = _V046_MULTICLIP_HEADER_RE.match(s)
    if not m:
        return None
    idx = int(m.group(1))
    if idx < 1 or idx > 16:
        raise ValueError(f'MultiClip prompt section index {idx} is outside the supported 1..16 range.')
    return idx, str(m.group(2) or '').strip()


def _v046_try_header_sections(text):
    sections = {}
    current = None
    body = []

    def flush():
        nonlocal current, body
        if current is None:
            return
        if current in sections:
            raise ValueError(f'MultiClip prompt contains duplicate clip/shot section {current}.')
        sections[current] = _v046_prompt_body(body)
        body = []

    for line in text.split('\n'):
        header = _v046_header(line)
        if header is not None:
            flush()
            current, inline = header
            body = [inline] if inline else []
        elif current is not None:
            body.append(line)
    flush()
    return _v046_validate_sections(sections) if sections else tuple()


def _v046_try_numbered_list(text):
    lines = [line for line in text.split('\n') if line.strip()]
    if len(lines) < 2:
        return tuple()
    sections = {}
    for line in lines:
        m = re.match(r'^\s*(\d{1,2})\s*[.)]\s+(.+?)\s*$', line)
        if not m:
            return tuple()
        idx = int(m.group(1))
        if idx < 1 or idx > 16 or idx in sections:
            return tuple()
        sections[idx] = str(m.group(2) or '').strip()
    return _v046_validate_sections(sections)


def _v043_parse_multiclip_prompt_text(raw):
    """Parse common LLM MultiClip formats into contiguous clip prompts.

    Accepted without changing Planner timing/seed fields:
      - clip_1: text / Clip 1 - text / SHOT-2 = text
      - standalone clip_N headers followed by multiline text
      - Markdown headings/bold labels and [clip_N] labels
      - Clip N (5s): text labels
      - YAML-like clip sections with prompt:/duration:/seed: wrappers
      - JSON arrays, {"clips":[...]}, or {"clip_1": ...} objects
      - <clip_1>...</clip_1> XML-style blocks
      - strict contiguous numbered lists: 1. ..., 2. ...

    The parser intentionally imports prompt text only; existing card durations and
    seeds remain authoritative.
    """
    text = _v046_strip_prompt_fence(raw)
    if not text:
        return tuple()

    for parser in (
        _v046_try_json_prompt,
        _v046_try_xml_prompt,
        _v046_try_header_sections,
        _v046_try_numbered_list,
    ):
        parsed = parser(text)
        if parsed:
            return tuple(parsed[:16])
    return tuple()

def _v043_import_multiclip_prompts(clips, structured_prompt, fallback_duration=7.5):
    """Copy structured prompt sections into clip cards while preserving timing/seed."""
    sections = _v043_parse_multiclip_prompt_text(structured_prompt)
    if not sections:
        return tuple(clips), False
    out = [dict(c) for c in clips]
    default_duration = float(out[-1].get('duration', fallback_duration)) if out else float(fallback_duration)
    while len(out) < len(sections):
        out.append({'clip_id': f"clip-{len(out)+1}", 'name': '', 'prompt': '', 'duration': default_duration, 'seed': None})
    for i, prompt_text in enumerate(sections):
        out[i]['prompt'] = str(prompt_text)
    return tuple(out[:16]), True

def _v043_join_global_local_prompt(global_prompt, local_prompt):
    global_text = str(global_prompt or '').strip()
    local_text = str(local_prompt or '').strip()
    if global_text and local_text:
        return f'{global_text}\n\n{local_text}'
    return global_text or local_text

def _v85_parse_multiclip_json(raw, fallback_prompt, fallback_duration):
    try:
        payload = json.loads(raw or '[]')
    except Exception as exc:
        raise ValueError(f"MultiClip JSON is invalid: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError('MultiClip JSON must be an array of clip objects.')
    if len(payload) < 2:
        raise ValueError('MultiClip requires at least 2 clips.')
    if len(payload) > 16:
        raise ValueError('MultiClip prototype supports at most 16 clips.')
    clips = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f'MultiClip clip {i+1} must be an object.')
        prompt = str(item.get('prompt') or fallback_prompt or '').strip()
        try:
            duration = float(item.get('duration', fallback_duration))
        except Exception:
            raise ValueError(f'MultiClip clip {i+1} has invalid duration.')
        if duration <= 0.0 or duration > 150.0:
            raise ValueError(f'MultiClip clip {i+1} duration must be >0 and <=150 seconds.')
        seed = item.get('seed', None)
        if seed is not None:
            try:
                seed = int(seed) & 0xFFFFFFFFFFFFFFFF
            except Exception:
                raise ValueError(f'MultiClip clip {i+1} has invalid seed.')
        clip_id = str(item.get('clip_id') or item.get('id') or '').strip() or f"clip-{i+1}"
        name = str(item.get('name') or item.get('clip_name') or '').strip()[:120]
        clips.append({'clip_id': clip_id, 'name': name, 'prompt': prompt, 'duration': duration, 'seed': seed})
    return tuple(clips)


def _v85_multiclip_geometry(clips, overlap_frames):
    overlap = int(overlap_frames)
    # v0.4.56 continuity: MultiClip previously used the minimal 5-frame native
    # H3 continuation head. That is enough to keep the temporal lattice valid,
    # but too short to stabilize exposure/illumination across independently
    # conditioned clips. Use a longer native context while preserving the
    # *visible* duration contract of the old 5-frame path.
    #
    # H3 aligned clip lengths are 17*k+5. Replacing the hidden overlap 5 -> 22
    # adds exactly one 17-frame H3 period to the hidden prefix, so extend every
    # continuation clip by the same delta. After stripping the larger prefix at
    # decode, each clip contributes exactly the same number of visible frames
    # as before. No duration drift, no RGB blending, no frame duplication.
    baseline_overlap = 5
    base_lengths = [int(align_frame_count(max(5, round(float(c['duration']) * FPS)))) for c in clips]
    extra_hidden = max(0, int(overlap) - int(baseline_overlap))
    if extra_hidden % 17 != 0:
        raise ValueError(
            f'MultiClip internal overlap must differ from the 5-frame baseline by whole 17-frame H3 periods; '
            f'got overlap_frames={overlap}.'
        )
    lengths = []
    for i, n in enumerate(base_lengths):
        lengths.append(int(n if i == 0 else n + extra_hidden))
    lengths = tuple(lengths)
    if any(n <= overlap for n in lengths[1:]):
        raise ValueError(f'MultiClip every continuation clip must be longer than overlap_frames={overlap}.')
    starts = [0]
    visible_end = int(lengths[0])
    for n in lengths[1:]:
        starts.append(int(visible_end - overlap))
        visible_end += int(n - overlap)
    return lengths, tuple(starts), int(visible_end)



def _v104_slice_lipsync_guide_audio(source_audio, start_frame, length_frames):
    """Return the exact source waveform window for one local H3 pass.

    The continuation pass begins at the global context_start, not at its visible
    start.  This gives the native H3 Audio Guide the same hidden preroll as the
    video Motion Context while keeping the guide on local frame 0.
    """
    waveform = source_audio['waveform'][:1]
    sr = int(source_audio['sample_rate'])
    start_frame = int(start_frame)
    length_frames = int(length_frames)
    start_sample = int(round(float(start_frame) * float(sr) / float(FPS)))
    end_sample = int(round(float(start_frame + length_frames) * float(sr) / float(FPS)))
    expected = max(1, end_sample - start_sample)
    left_pad = max(0, -start_sample)
    src0 = max(0, start_sample)
    src1 = max(src0, end_sample)
    sliced = waveform[..., src0:src1]
    if left_pad:
        sliced = torch.nn.functional.pad(sliced, (left_pad, 0))
    if int(sliced.shape[-1]) < expected:
        sliced = torch.nn.functional.pad(sliced, (0, expected - int(sliced.shape[-1])))
    elif int(sliced.shape[-1]) > expected:
        sliced = sliced[..., :expected]
    return {'waveform': sliced.contiguous(), 'sample_rate': sr}, start_sample, end_sample


def _v113_lock_source_audio_in_target(target_av, audio_vae, source_audio, start_frame, length_frames):
    """Put the exact source-audio window into the target AV latent and freeze it.

    H3 is a joint audio/video transformer.  Ref2VA audio and AddGuide audio are
    conditioning references; they do not make the target audio stream authoritative.
    For actual speech-driven video we instead expose the source speech as the target
    audio tokens (noise mask 0) while leaving the target video fully denoised.
    """
    if target_av is None or audio_vae is None or source_audio is None:
        return target_av
    sliced, start_sample, end_sample = _v104_slice_lipsync_guide_audio(
        source_audio, int(start_frame), int(length_frames),
    )
    waveform = sliced['waveform']
    sr = int(sliced['sample_rate'])
    vae_sr = int(getattr(audio_vae, 'audio_sample_rate', 32000))
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    audio_latent = audio_vae.encode(waveform[:1].movedim(1, -1))
    locked = replace_stream(
        target_av, {'samples': audio_latent}, 'audio', 'crop_pad', 'start', 0.0, NestedTensor,
    )
    video, audio = unpack_av_samples(locked)
    locked = set_stream_denoise(locked, 1.0, 0.0, 'replace', NestedTensor)
    _lm_print(
        '[MiniMaxH3 LongMedia][LOCKED TARGET AUDIO] '
        f'global={int(start_frame)}f length={int(length_frames)}f '
        f'samples={start_sample}..{end_sample} target_audio_t={int(audio.shape[-1])}; '
        'video_denoise=1 audio_denoise=0',
        flush=True,
    )
    return locked


def _v104_attach_native_lipsync_guide(positive, audio_vae, source_audio, plan, segment_index):
    """Attach a native audio guide only when the runtime actually needs one.

    Current LongMedia lip-sync uses the exact source-audio window as the target
    audio latent and freezes that stream (``lip_sync_target_audio_locked``).
    That target stream is already the authoritative local clock.  Current stock
    MiniMax H3 ``PackedLayout`` treats every ``minimax_keyframes`` entry as a
    *visual* condition block, so inserting an audio-only keyframe creates extra
    ``img_update`` rows with no matching video latent and fails before denoising.

    Therefore locked-target lip-sync deliberately does NOT add an audio-only
    keyframe.  Audio1 may still remain a normal Ref2VA audio reference for
    semantic/content conditioning; timing is owned by the frozen target audio.
    """
    if source_audio is None or audio_vae is None:
        return positive
    if bool(getattr(plan, 'lip_sync_target_audio_locked', False)):
        _lm_print(
            '[MiniMaxH3 LongMedia][LIP SYNC GUIDE] skipped audio-only keyframe; '
            'frozen target audio is the authoritative timing stream',
            flush=True,
        )
        return positive
    timeline = _segment_timeline_contract(plan, int(segment_index))
    start_frame = int(timeline['context_start'])
    length_frames = int(timeline['length_frames'])
    sliced, start_sample, end_sample = _v104_slice_lipsync_guide_audio(
        source_audio, start_frame, length_frames,
    )
    audio_latent, _ = _hybrid_encode_ref_audio(audio_vae, sliced)
    max_rt = int(audio_latent_t(length_frames))
    if int(audio_latent.shape[-1]) > max_rt:
        audio_latent = audio_latent[..., :max_rt].clone()

    def patch_meta(meta):
        meta = dict(meta)
        keyframes = [dict(kf) for kf in (meta.get('minimax_keyframes', []) or [])
                     if not bool(kf.get('longmedia_lipsync_audio_guide'))]
        keyframes.append({
            'resolved_frame_index': 0,
            'audio_latent': audio_latent,
            'longmedia_lipsync_audio_guide': True,
            'longmedia_audio_context_start': start_frame,
        })
        meta['minimax_keyframes'] = keyframes
        meta.pop('minimax_frame_count', None)
        return meta

    out=[]
    attached=False
    for entry in (positive or []):
        if isinstance(entry, dict):
            out.append(patch_meta(entry)); attached=True
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], dict):
            ne=list(entry); ne[1]=patch_meta(entry[1]); out.append(ne); attached=True
        else:
            out.append(entry)
    if not attached:
        raise RuntimeError('lip_sync: native H3 Audio Guide could not attach to conditioning metadata')
    _lm_print(
        '[MiniMaxH3 LongMedia][LIP SYNC GUIDE] '
        f'clip={int(segment_index)+1} local=0f global={start_frame}f '
        f'visible={int(timeline["visible_start"])}f length={length_frames}f '
        f'samples={start_sample}..{end_sample} latent_t={int(audio_latent.shape[-1])}; '
        'Audio1 remains native Ref2VA reference',
        flush=True,
    )
    return out




def _v107_attach_visible_lipsync_guide(positive, audio_vae, source_audio, plan, segment_index):
    """Anchor only the NEW visible source-audio span on continuation passes.

    Full Audio1 remains the untouched native Ref2VA reference on every clip.
    The hidden overlap is owned by Extender-style AV Motion Context (video+audio
    tail from the previous sampled AV latent).  This guide starts exactly at the
    first visible local frame, so the two temporal conditions do not overlap.
    """
    idx = int(segment_index)
    if idx <= 0 or source_audio is None or audio_vae is None:
        return positive
    if bool(getattr(plan, 'lip_sync_target_audio_locked', False)):
        return positive
    timeline = _segment_timeline_contract(plan, idx)
    mark_in = int(timeline['visible_start'])
    visible_frames = int(timeline['visible_frames'])
    local_in = int(timeline['local_visible_offset'])
    waveform = source_audio['waveform'][:1]
    sr = int(source_audio['sample_rate'])
    start_sample = int(round(float(mark_in) * float(sr) / float(FPS)))
    end_sample = int(round(float(mark_in + visible_frames) * float(sr) / float(FPS)))
    expected = max(1, end_sample - start_sample)
    sliced = waveform[..., max(0, start_sample):max(0, end_sample)]
    if int(sliced.shape[-1]) < expected:
        sliced = torch.nn.functional.pad(sliced, (0, expected - int(sliced.shape[-1])))
    elif int(sliced.shape[-1]) > expected:
        sliced = sliced[..., :expected]
    audio = {'waveform': sliced.contiguous(), 'sample_rate': sr}
    audio_latent, encoded_t = _hybrid_encode_ref_audio(audio_vae, audio)

    # Match stock MiniMaxH3AddGuide: guide audio may occupy only the remaining
    # target-audio timeline after resolved_frame_index.
    target_audio_t = int(audio_latent_t(int(timeline['length_frames'])))
    max_rt = max(1, int(math.floor(float(target_audio_t) -
                                   (float(AUDIO_LATENT_FPS) / float(FPS)) * float(local_in))))
    if int(audio_latent.shape[-1]) > max_rt:
        audio_latent = audio_latent[..., :max_rt].clone()

    def patch(meta):
        meta = dict(meta)
        keyframes = [dict(kf) for kf in (meta.get('minimax_keyframes', []) or [])
                     if not bool(kf.get('longmedia_v107_visible_lipsync_guide'))]
        keyframes.append({
            'resolved_frame_index': int(local_in),
            'audio_latent': audio_latent,
            'longmedia_v107_visible_lipsync_guide': True,
            'longmedia_global_visible_start': int(mark_in),
        })
        meta['minimax_keyframes'] = keyframes
        meta.pop('minimax_frame_count', None)
        return meta

    out=[]; attached=False
    for entry in (positive or []):
        if isinstance(entry, dict):
            out.append(patch(entry)); attached=True
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], dict):
            ne=list(entry); ne[1]=patch(entry[1]); out.append(ne); attached=True
        else:
            out.append(entry)
    if not attached:
        raise RuntimeError('lip_sync: visible native H3 Audio Guide could not attach')
    _lm_print(
        '[MiniMaxH3 LongMedia][VISIBLE LIP SYNC GUIDE] '
        f'clip={idx+1} local_start={local_in}f global_start={mark_in}f '
        f'visible_frames={visible_frames}f samples={start_sample}..{end_sample} '
        f'encoded_ref_t={int(encoded_t)} guide_t={int(audio_latent.shape[-1])} '
        f'target_audio_t={target_audio_t}; hidden 0..{local_in-1}f remains AV Motion Context only',
        flush=True,
    )
    return out

def _v85_preencode_multiclip_conditionings(clip, positive, plan, prompts, v329_native_refs=None, lip_sync_audio=None, audio_vae=None):
    import comfy.sampler_helpers
    # pass 0 is attached once in Setup because the external guider samples that
    # exact CONDITIONING object. Continuation passes are attached below.
    raw = [positive]
    for idx in range(1, len(prompts)):
        text = str(prompts[idx])
        if v329_native_refs is not None:
            ref_items, ref_blocks = v329_native_refs
            encoded = _v329_encode_continuation_native_refs(
                clip, text, positive, plan, idx, ref_items, ref_blocks,
            )
        else:
            encoded = _encode_prompt(clip, text)
            encoded = _v57_attach_minimax_metadata(
                encoded, positive, plan, idx, drop_image_refs=False,
            )
        if lip_sync_audio is not None:
            encoded = _v104_attach_native_lipsync_guide(
                encoded, audio_vae, lip_sync_audio, plan, idx,
            )
        raw.append(encoded)
    converted = tuple(comfy.sampler_helpers.convert_cond(cond) for cond in raw)
    _lm_print(
        '[MiniMaxH3 LongMedia][MULTICLIP LIP SYNC CONDITIONING] '
        f'pre-encoded {len(converted)} clips; shared native refs preserved; lip_sync_guide={bool(lip_sync_audio is not None)}',
        flush=True,
    )
    return converted, tuple(str(x) for x in prompts)


def _v85_segment_seed(plan, base_seed, segment_index):
    idx = int(segment_index)
    seeds = getattr(plan, 'segment_seeds', None)
    if seeds and idx < len(seeds) and seeds[idx] is not None:
        return int(seeds[idx]) & 0xFFFFFFFFFFFFFFFF
    return (int(base_seed) + idx) & 0xFFFFFFFFFFFFFFFF




_LONGMEDIA_CAMERA_RIGS = {
    "Tripod / Locked Head": "camera mounted on a rigid tripod with a locked or controlled head",
    "Fluid Head Tripod": "camera mounted on a professional fluid-head tripod",
    "Dolly / Track": "camera mounted on a cinema dolly or linear track",
    "Slider": "camera mounted on a compact motorized slider",
    "Jib / Crane": "camera mounted on a jib or crane arm",
    "Technocrane": "camera mounted on a telescopic Technocrane",
    "Steadicam": "camera mounted on a body-worn Steadicam stabilization rig",
    "3-Axis Gimbal": "camera mounted on a motorized three-axis gimbal",
    "Shoulder Rig": "camera mounted on a shoulder rig",
    "Handheld": "camera operated handheld",
    "Vehicle Mount": "camera mounted to a moving vehicle or pursuit platform",
    "Cable Cam": "camera suspended on a cable-cam system",
    "Robot Arm · Bolt": "camera mounted on a high-speed MRMC Bolt-style cinema robot arm",
    "Robot Arm · KUKA": "camera mounted on an industrial KUKA-style motion-control robot arm",
    "Drone · Heavy-Lift Cinema": "camera carried by a heavy-lift professional cinema drone",
    "Drone · DJI Inspire 3": "camera carried by a DJI Inspire 3 professional aerial platform",
    "Drone · DJI Mavic 3 Cine": "camera carried by a DJI Mavic 3 Cine aerial platform",
    "Drone · DJI Air 3S": "camera carried by a DJI Air 3S aerial platform",
    "Drone · DJI Mini 4 Pro": "camera carried by a DJI Mini 4 Pro compact aerial platform",
    "FPV · DJI Avata 2": "camera carried by a DJI Avata 2 FPV platform",
    "FPV · Cinewhoop": "camera carried by a compact cinewhoop FPV platform",
    "FPV · Racing": "camera carried by a high-speed racing FPV platform",
    "Bodycam Mount": "camera fixed to a body-worn mount",
    "Helmet / Head Mount": "camera fixed to a head or helmet mount",
    "Static Security Mount": "camera fixed to a rigid surveillance mount",
}

_LONGMEDIA_CAMERA_BODIES = {
    "Cinematic Neutral": "high-end neutral digital cinema camera response",
    "ARRI Alexa 35": "ARRI Alexa 35 digital cinema camera with natural highlight roll-off and rich dynamic range",
    "ARRI Alexa Mini LF": "ARRI Alexa Mini LF large-format cinema camera with soft highlight roll-off",
    "Sony VENICE 2": "Sony VENICE 2 full-frame digital cinema camera",
    "RED V-RAPTOR XL": "RED V-RAPTOR XL high-resolution digital cinema camera",
    "RED KOMODO-X": "RED KOMODO-X compact global-shutter cinema camera",
    "Blackmagic URSA Cine 12K": "Blackmagic URSA Cine 12K digital cinema camera",
    "Sony FX3": "Sony FX3 compact full-frame cinema camera",
    "Sony FX6": "Sony FX6 documentary-oriented full-frame cinema camera",
    "Canon C400": "Canon C400 digital cinema camera",
    "Canon EOS R5 C": "Canon EOS R5 C hybrid cinema camera",
    "Canon EOS 5D Mark II": "Canon EOS 5D Mark II DSLR video camera",
    "Nikon D850": "Nikon D850 DSLR camera",
    "Sony DCR-VX1000": "Sony DCR-VX1000 MiniDV camcorder",
    "Canon XL1": "Canon XL1 MiniDV camcorder",
    "Panasonic DVX100": "Panasonic DVX100 MiniDV camcorder",
    "VHS Camcorder": "full-size consumer VHS camcorder",
    "VHS-C Camcorder": "compact VHS-C analog camcorder",
    "Sony Hi8 Handycam": "Sony Hi8 analog Handycam",
    "Super 8 Camera": "Super 8 small-gauge film camera",
    "Aaton XTR 16mm": "Aaton XTR 16mm motion-picture camera",
    "Arricam LT 35mm": "Arricam LT 35mm motion-picture camera",
    "IMAX 65mm": "IMAX 65mm large-format motion-picture camera",
    "Smartphone · Snapshot": "modern flagship smartphone in casual snapshot video mode",
    "Smartphone · Cinematic": "modern flagship smartphone in computational cinematic video mode",
    "Action Camera": "compact wide-angle action camera",
    "Broadcast ENG": "professional broadcast ENG camera",
    "CCTV Sensor": "utilitarian surveillance camera sensor",
    "Webcam": "consumer webcam imaging system",
}

_LONGMEDIA_LENSES = {
    "Auto / Native Lens": "natural lens choice appropriate to the selected camera body, rig and shot size",
    "Ultra-Wide 10mm": "10mm rectilinear ultra-wide lens with extreme spatial expansion",
    "Ultra-Wide 12mm": "12mm ultra-wide lens with strong environmental perspective",
    "Ultra-Wide 14mm": "14mm ultra-wide cinema lens",
    "Wide 18mm": "18mm wide-angle cinema lens",
    "Wide 21mm": "21mm wide-angle cinema lens",
    "Wide 24mm": "24mm wide-angle cinema lens",
    "Wide 28mm": "28mm moderate wide-angle lens",
    "Natural 35mm": "35mm natural wide-normal cinema lens",
    "Natural 40mm": "40mm natural perspective cinema lens",
    "Standard 50mm": "50mm standard lens with natural perspective",
    "Portrait 65mm": "65mm short-tele portrait cinema lens",
    "Portrait 85mm": "85mm portrait lens with compressed perspective and shallow depth",
    "Telephoto 100mm": "100mm telephoto lens",
    "Telephoto 135mm": "135mm telephoto lens with strong spatial compression",
    "Long Telephoto 200mm": "200mm long telephoto lens",
    "Long Telephoto 300mm": "300mm long telephoto lens with very strong compression",
    "Macro 60mm": "60mm macro lens for close detail",
    "Macro 100mm": "100mm macro lens for extreme close detail",
    "Anamorphic 28mm": "28mm anamorphic cinema lens",
    "Anamorphic 35mm": "35mm anamorphic cinema lens with horizontal field character and oval bokeh",
    "Anamorphic 50mm": "50mm anamorphic cinema lens with cinematic compression and oval bokeh",
    "Anamorphic 75mm": "75mm anamorphic cinema lens with portrait compression",
    "Vintage Spherical · Wide": "vintage wide spherical cinema lens with softer contrast and organic aberrations",
    "Vintage Spherical · Normal": "vintage normal spherical cinema lens with softer contrast and organic aberrations",
    "Vintage Spherical · Portrait": "vintage portrait spherical cinema lens with soft roll-off and organic aberrations",
    "Probe Lens": "long probe macro lens for extreme close-range moving shots",
    "Tilt-Shift": "tilt-shift lens with selective plane-of-focus control",
    "Fisheye": "fisheye lens with extreme curved ultra-wide perspective",
    "Smartphone Ultra-Wide": "smartphone computational ultra-wide lens",
    "Smartphone Wide": "smartphone computational wide lens",
    "Smartphone Tele": "smartphone computational telephoto lens",
}

_LONGMEDIA_RIG_STABILIZATION = {
    "Rig Native": "use the natural stabilization behavior of the selected rig",
    "Hard Locked": "mechanically locked orientation with no operator drift",
    "Fluid Controlled": "fluid controlled stabilized motion with gentle acceleration",
    "Gyro Stabilized": "strong gyroscopic stabilization with horizon control",
    "Gimbal Smooth": "motorized gimbal stabilization with polished floating motion",
    "Steadicam Organic": "Steadicam stabilization with subtle organic operator drift",
    "Handheld Controlled": "restrained handheld micro-motion",
    "Handheld Raw": "raw handheld movement with stronger natural micro-jitter",
    "FPV Stabilized": "stabilized FPV motion retaining agile flight characteristics",
    "FPV Raw": "direct FPV flight feel with stronger banking and rotation",
}


_LONGMEDIA_SHOT_SIZES = {
    "Extreme Wide Shot": "extreme wide shot, subject very small within a vast environment",
    "Wide Shot": "wide shot showing the full subject and substantial environment",
    "Full Shot": "full-body shot from head to toe",
    "Cowboy Shot": "cowboy shot framed from approximately mid-thigh upward",
    "Medium Full Shot": "medium-full shot framed roughly from the knees upward",
    "Medium Shot": "medium shot framed approximately from the waist upward",
    "Medium Close-Up": "medium close-up framed from the chest or shoulders upward",
    "Close-Up": "close-up emphasizing the face and upper shoulders",
    "Extreme Close-Up": "extreme close-up emphasizing facial details or a single expressive feature",
    "Macro / Detail": "macro detail shot with very tight framing on a small subject detail",
    "Over-the-Shoulder": "over-the-shoulder framing with a foreground shoulder or silhouette",
    "Two-Shot": "two-shot composed to hold two subjects clearly in the frame",
    "POV Framing": "subjective point-of-view framing from the observer's perspective",
}

_LONGMEDIA_CAMERA_MOVEMENTS = {
    "Locked-Off / Static": "camera remains completely locked-off and static",
    "Push-In": "camera moves directly forward toward the subject",
    "Pull-Out": "camera moves directly backward away from the subject",
    "Track Forward": "camera translates forward through the scene",
    "Track Backward": "camera translates backward through the scene",
    "Track Left": "camera translates laterally to the left",
    "Track Right": "camera translates laterally to the right",
    "Pan Left": "camera rotates horizontally to the left from its position",
    "Pan Right": "camera rotates horizontally to the right from its position",
    "Tilt Up": "camera rotates vertically upward from its position",
    "Tilt Down": "camera rotates vertically downward from its position",
    "Crane Up": "camera rises vertically upward",
    "Crane Down": "camera descends vertically downward",
    "Pedestal Up": "camera body moves straight upward while preserving viewing direction",
    "Pedestal Down": "camera body moves straight downward while preserving viewing direction",
    "Arc Left": "camera moves on a partial circular arc around the subject toward the left",
    "Arc Right": "camera moves on a partial circular arc around the subject toward the right",
    "Orbit Clockwise": "camera performs a circular orbit around the subject in a clockwise direction",
    "Orbit Counterclockwise": "camera performs a circular orbit around the subject in a counterclockwise direction",
    "Full 360 Orbit Clockwise": "camera completes a full 360-degree circular fly-around around the subject clockwise",
    "Full 360 Orbit Counterclockwise": "camera completes a full 360-degree circular fly-around around the subject counterclockwise",
    "Half Orbit Clockwise": "camera performs an approximately 180-degree circular fly-around around the subject clockwise",
    "Half Orbit Counterclockwise": "camera performs an approximately 180-degree circular fly-around around the subject counterclockwise",
    "Spiral In Clockwise": "camera circles clockwise while gradually moving closer to the subject",
    "Spiral In Counterclockwise": "camera circles counterclockwise while gradually moving closer to the subject",
    "Spiral Out Clockwise": "camera circles clockwise while gradually moving farther from the subject",
    "Spiral Out Counterclockwise": "camera circles counterclockwise while gradually moving farther from the subject",
    "Diagonal Forward Left": "camera moves diagonally forward and to the left",
    "Diagonal Forward Right": "camera moves diagonally forward and to the right",
    "Diagonal Backward Left": "camera moves diagonally backward and to the left",
    "Diagonal Backward Right": "camera moves diagonally backward and to the right",
    "Rise + Push-In": "camera rises while simultaneously moving forward toward the subject",
    "Descend + Push-In": "camera descends while simultaneously moving forward toward the subject",
    "Rise + Pull-Out": "camera rises while simultaneously moving backward away from the subject",
    "Descend + Pull-Out": "camera descends while simultaneously moving backward away from the subject",
}



_LONGMEDIA_CAMERA_SPEEDS = {
    "Static": "no visible viewpoint motion",
    "Ultra Slow": "extremely slow, almost imperceptible viewpoint motion",
    "Slow": "slow and deliberate viewpoint motion",
    "Controlled": "measured, polished, controlled viewpoint motion",
    "Medium": "moderate natural viewpoint motion",
    "Fast": "fast purposeful viewpoint motion",
    "Aggressive": "aggressive high-energy viewpoint motion",
    "Variable / Ramping": "viewpoint speed changes smoothly with cinematic acceleration and deceleration",
}


def _lm_camera_default_card():
    return {
        "clip_id": "",
        "clip_name": "",
        "shot_size": "Medium Shot",
        "rig": "Tripod / Locked Head",
        "camera_body": "Cinematic Neutral",
        "lens": "Auto / Native Lens",
        "stabilization": "Rig Native",
        "movement": "Locked-Off / Static",
        "speed": "Static",
        "transition_type": "Continuous / Same Shot",
        "space_relation": "Same Space",
        "entity_continuity": "Lock Population / Layout",
        "transition_to_next": False,
    }


def _lm_parse_camera_cards(raw, count=None):
    try:
        data = json.loads(str(raw or "[]"))
    except Exception:
        data = []
    if not isinstance(data, list):
        data = []
    cards = []
    for idx, item in enumerate(data[:16]):
        item = item if isinstance(item, dict) else {}
        d = _lm_camera_default_card()
        d["clip_id"] = str(item.get("clip_id") or "").strip() or f"clip-{idx+1}"
        d["clip_name"] = str(item.get("clip_name") or item.get("name") or "").strip()[:120]
        legacy_profile = item.get("camera_profile")
        for key in ("shot_size", "rig", "camera_body", "lens", "stabilization", "movement", "speed", "transition_type", "space_relation", "entity_continuity"):
            value = item.get(key)
            if key == "camera_body" and not value and legacy_profile:
                value = legacy_profile
            d[key] = str(value or d[key])

        legacy_body = d["camera_body"]
        legacy_drone_map = {
            "Drone · DJI Inspire 3": "Drone · DJI Inspire 3",
            "Drone · DJI Mavic 3 Cine": "Drone · DJI Mavic 3 Cine",
            "Drone · DJI Air 3S": "Drone · DJI Air 3S",
            "Drone · DJI Mini 4 Pro": "Drone · DJI Mini 4 Pro",
            "FPV Drone · DJI Avata 2": "FPV · DJI Avata 2",
            "FPV Drone · Racing": "FPV · Racing",
            "FPV Drone · Cinewhoop": "FPV · Cinewhoop",
            "Heavy-Lift Cinema Drone": "Drone · Heavy-Lift Cinema",
        }
        if legacy_body in legacy_drone_map and not item.get("rig"):
            d["rig"] = legacy_drone_map[legacy_body]
            d["camera_body"] = "Cinematic Neutral"

        d["transition_to_next"] = bool(item.get("transition_to_next", False))
        cards.append(d)
    target = int(count) if count is not None else max(2, len(cards))
    target = max(2, min(16, target))
    while len(cards) < target:
        cards.append(dict(cards[-1] if cards else _lm_camera_default_card()))
    return cards[:target]


# v0.4.74: non-diegetic camera compiler.
# UI keeps real-world rig/body names for creative control, but model-facing text
# describes only the resulting viewpoint, motion and image characteristics.
# Physical filming equipment nouns are deliberately removed from positive prompt
# text so H3 does not materialize cranes, tripods, drones, operators, gimbals, etc.

_LONGMEDIA_RIG_PROMPT_SAFE = {
    "Tripod / Locked Head": "a stable ground-level viewpoint with disciplined composition",
    "Fluid Head Tripod": "a stable ground-level viewpoint with smooth controlled orientation changes",
    "Dolly / Track": "a stable ground-level cinematic viewpoint suited to smooth linear travel",
    "Slider": "a precise close-control viewpoint suited to short lateral or forward travel",
    "Jib / Crane": "an elevated cinematic viewpoint suited to graceful vertical and arcing motion",
    "Technocrane": "a long-reach elevated cinematic viewpoint suited to telescoping and vertical motion",
    "Steadicam": "a stabilized body-height viewpoint with organic human flow",
    "3-Axis Gimbal": "a floating highly stabilized body-height viewpoint",
    "Shoulder Rig": "a body-height viewpoint with restrained human energy",
    "Handheld": "a human-height viewpoint with natural micro-motion and responsive framing",
    "Vehicle Mount": "a fast traveling pursuit viewpoint at low-to-mid height",
    "Cable Cam": "a suspended elevated traveling viewpoint with smooth linear motion",
    "Robot Arm · Bolt": "a precisely repeatable motion-control viewpoint with crisp positional precision",
    "Robot Arm · KUKA": "a precisely repeatable articulated motion-control viewpoint",
    "Drone · Heavy-Lift Cinema": "a high aerial viewpoint with broad cinematic spatial freedom",
    "Drone · DJI Inspire 3": "a high aerial viewpoint with smooth controlled cinematic flight",
    "Drone · DJI Mavic 3 Cine": "a compact aerial viewpoint with smooth controlled flight",
    "Drone · DJI Air 3S": "an agile aerial viewpoint with smooth stabilized flight",
    "Drone · DJI Mini 4 Pro": "a lightweight agile aerial viewpoint with smooth stabilized flight",
    "FPV · DJI Avata 2": "an agile low-altitude first-person aerial viewpoint with dynamic banking",
    "FPV · Cinewhoop": "an agile close-proximity aerial viewpoint suited to tight interior or exterior paths",
    "FPV · Racing": "a very fast first-person aerial viewpoint with aggressive banking and directional changes",
    "Bodycam Mount": "a body-attached first-person viewpoint moving directly with the subject",
    "Helmet / Head Mount": "a head-height first-person viewpoint aligned with the wearer's gaze",
    "Static Security Mount": "a fixed elevated observational viewpoint with disciplined framing",
}
_LONGMEDIA_BODY_PROMPT_SAFE = {
    "Cinematic Neutral": "neutral high-end digital-cinema tonality with natural highlight roll-off and wide dynamic range",
    "ARRI Alexa 35": "refined digital-cinema tonality with natural color separation, rich dynamic range and soft highlight roll-off",
    "ARRI Alexa Mini LF": "large-format cinematic tonality with gentle highlight roll-off, natural skin response and dimensional depth",
    "Sony VENICE 2": "clean full-frame cinematic tonality with wide dynamic range and controlled highlight response",
    "RED V-RAPTOR XL": "crisp high-resolution cinematic rendering with strong micro-detail and broad dynamic range",
    "RED KOMODO-X": "crisp compact-cinema rendering with global-shutter motion character and clean detail",
    "Blackmagic URSA Cine 12K": "high-resolution cinematic rendering with rich tonal detail and broad dynamic range",
    "Sony FX3": "compact full-frame cinematic rendering with clean low-light response and natural depth",
    "Sony FX6": "documentary-oriented full-frame cinematic rendering with natural motion and controlled highlights",
    "Canon C400": "clean cinematic rendering with natural color, soft highlight response and full-frame depth",
    "Canon EOS R5 C": "hybrid cinematic rendering with crisp detail and photographic depth-of-field character",
    "Canon EOS 5D Mark II": "early full-frame DSLR-video character with shallow photographic depth, softer motion cadence and mild highlight clipping",
    "Nikon D850": "high-resolution DSLR-video character with photographic color and shallow depth-of-field rendering",
    "Sony DCR-VX1000": "late-1990s MiniDV video character with interlaced texture, electronic sharpness and clipped highlights",
    "Canon XL1": "late-1990s MiniDV video character with soft standard-definition detail and video-like highlight response",
    "Panasonic DVX100": "early-2000s MiniDV cinematic-video character with soft detail and film-inspired motion cadence",
    "VHS Camcorder": "consumer VHS analog-video character with low resolution, chroma bleed, tape noise and unstable tracking texture",
    "VHS-C Camcorder": "compact VHS-C analog-video character with soft detail, chroma bleed, tape noise and consumer exposure behavior",
    "Sony Hi8 Handycam": "Hi8 analog-video character with soft detail, chroma noise and late-1990s consumer-video tonality",
    "Super 8 Camera": "Super 8 film character with coarse grain, soft detail, halation and small-gauge motion texture",
    "Aaton XTR 16mm": "16mm film character with organic grain, soft highlight roll-off and textured motion cadence",
    "Arricam LT 35mm": "35mm motion-picture film character with fine grain, rich latitude and natural highlight roll-off",
    "IMAX 65mm": "large-format 65mm film character with extremely fine detail, expansive depth and smooth tonal roll-off",
    "Smartphone · Snapshot": "casual modern phone-video imaging with computational sharpening, automatic exposure and compact-lens depth",
    "Smartphone · Cinematic": "modern computational phone-video imaging with stabilized cinematic depth simulation and controlled exposure",
    "Action Camera": "compact ultra-wide action-video imaging with deep focus, strong stabilization and high local contrast",
    "Broadcast ENG": "broadcast-news video character with deep focus, crisp electronic detail and fast automatic exposure response",
    "CCTV Sensor": "utilitarian surveillance-video character with deep focus, fixed exposure feel and limited dynamic range",
    "Webcam": "consumer webcam-video character with fixed perspective, deep focus and compressed digital tonality",
}

_LONGMEDIA_STABILIZATION_PROMPT_SAFE = {
    "Rig Native": "natural motion stabilization appropriate to the selected viewpoint behavior",
    "Hard Locked": "perfectly locked orientation with no drift",
    "Fluid Controlled": "smooth controlled motion with gentle acceleration and deceleration",
    "Gyro Stabilized": "strong horizon-stable motion with minimal rotational drift",
    "Gimbal Smooth": "polished floating stabilization with very smooth orientation changes",
    "Steadicam Organic": "smooth stabilized tracking with subtle organic body-motion drift",
    "Handheld Controlled": "restrained natural micro-motion with controlled framing",
    "Handheld Raw": "stronger natural handheld micro-jitter and reactive framing",
    "FPV Stabilized": "smooth agile flight motion with controlled banking and horizon behavior",
    "FPV Raw": "direct agile first-person flight motion with stronger banking, rotation and acceleration",
}

_LONGMEDIA_LENS_PROMPT_SAFE_OVERRIDES = {
    "Auto / Native Lens": "natural optical perspective appropriate to the framing and viewpoint",
    "Smartphone Ultra-Wide": "computational ultra-wide perspective with deep focus and compact-lens spatial expansion",
    "Smartphone Wide": "computational wide perspective with deep focus and compact-lens rendering",
    "Smartphone Tele": "computational telephoto perspective with compressed depth and compact-lens rendering",
    "Probe Lens": "extreme close-range elongated macro perspective with very close subject access",
}


_LONGMEDIA_CAMERA_VISIBILITY_GUARD = (
    "All cinematography is non-diegetic. No capture equipment, production hardware, crew, or behind-the-scenes elements may appear or be reflected anywhere in the scene."
)

_LONGMEDIA_STATIC_MOVEMENT_KEYS = {"Locked-Off / Static"}

_LONGMEDIA_CAMERA_SANITIZER_PATTERNS = (
    re.compile(r"^\s*\[(?:NON-DIEGETIC|CAMERA|VIEWPOINT)[^\]]*\]", re.IGNORECASE),
    re.compile(r"\b(?:camera|viewpoint|framing|lens|focal length|optical perspective|zoom|pan|tilt|roll|parallax|push-?in|pull-?out|track(?:ing)?|orbit|spiral|crane|pedestal|dolly|slider|gimbal|steadicam|tripod|handheld|drone|fpv|stabilization)\b", re.IGNORECASE),
    re.compile(r"\b(?:extreme close[- ]up|close[- ]up|medium close[- ]up|medium shot|wide shot|very wide shot|full shot|long shot|establishing shot|over[- ]the[- ]shoulder|two[- ]shot|macro shot)\b", re.IGNORECASE),
)


def _lm_camera_safe_speed_text(speed_key, movement_key):
    movement_key = str(movement_key or "Locked-Off / Static")
    speed_key = str(speed_key or "Static")
    if movement_key in _LONGMEDIA_STATIC_MOVEMENT_KEYS:
        speed_key = "Static"
    elif speed_key == "Static":
        speed_key = "Ultra Slow"
    raw = str(_LONGMEDIA_CAMERA_SPEEDS.get(speed_key, speed_key))
    raw = raw.replace("camera", "viewpoint")
    return raw


def _lm_is_camera_directive_text(text):
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact:
        return False
    return any(p.search(compact) for p in _LONGMEDIA_CAMERA_SANITIZER_PATTERNS)


def _lm_strip_camera_directives(prompt_text):
    raw = str(prompt_text or "").replace("\r\n", "\n")
    if not raw.strip():
        return "", []
    kept_lines = []
    removed = []
    for line in raw.split("\n"):
        stripped = line.strip()
        if not stripped:
            if kept_lines and kept_lines[-1] != "":
                kept_lines.append("")
            continue
        sentences = re.split(r"(?<=[.!?])\s+", stripped)
        kept_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if _lm_is_camera_directive_text(sentence):
                removed.append(sentence)
            else:
                kept_sentences.append(sentence)
        if kept_sentences:
            kept_lines.append(" ".join(kept_sentences))
    cleaned = "\n".join(kept_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned, removed


def _lm_camera_safe_lens_text(lens_key):
    if lens_key in _LONGMEDIA_LENS_PROMPT_SAFE_OVERRIDES:
        return _LONGMEDIA_LENS_PROMPT_SAFE_OVERRIDES[lens_key]
    raw = str(_LONGMEDIA_LENSES.get(lens_key, lens_key))
    raw = raw.replace("cinema lens", "cinematic optical perspective")
    raw = raw.replace("lens", "optical perspective")
    return raw


def _lm_camera_safe_movement_text(movement_key):
    raw = str(_LONGMEDIA_CAMERA_MOVEMENTS.get(movement_key, movement_key))
    raw = raw.replace("camera body", "viewpoint")
    raw = raw.replace("camera", "viewpoint")
    return raw


def _lm_camera_state_text(card):
    shot_key = str(card.get("shot_size") or "Medium Shot")
    rig_key = str(card.get("rig") or "Tripod / Locked Head")
    body_key = str(card.get("camera_body") or card.get("camera_profile") or "Cinematic Neutral")
    lens_key = str(card.get("lens") or "Auto / Native Lens")
    stabilization_key = str(card.get("stabilization") or "Rig Native")
    movement_key = str(card.get("movement") or "Locked-Off / Static")
    speed_key = str(card.get("speed") or "Static")

    return {
        "shot_key": shot_key,
        "rig_key": rig_key,
        "body_key": body_key,
        "lens_key": lens_key,
        "stabilization_key": stabilization_key,
        "movement_key": movement_key,
        "speed_key": speed_key,
        "shot": _LONGMEDIA_SHOT_SIZES.get(shot_key, shot_key),
        "rig": _LONGMEDIA_RIG_PROMPT_SAFE.get(rig_key, "a natural viewpoint appropriate to the requested framing"),
        "body": _LONGMEDIA_BODY_PROMPT_SAFE.get(body_key, "neutral cinematic image tonality"),
        "lens": _lm_camera_safe_lens_text(lens_key),
        "stabilization": _LONGMEDIA_STABILIZATION_PROMPT_SAFE.get(
            stabilization_key,
            "natural controlled motion stabilization",
        ),
        "movement": _lm_camera_safe_movement_text(movement_key),
        "speed": _lm_camera_safe_speed_text(speed_key, movement_key),
    }


def _lm_camera_transition_clauses(state, nxt):
    clauses = []
    if state["shot_key"] != nxt["shot_key"]:
        clauses.append(f"framing gradually shifts from {state['shot']} to {nxt['shot']}")
    if state["movement_key"] != nxt["movement_key"]:
        clauses.append(f"viewpoint motion evolves from {state['movement']} to {nxt['movement']}")
    if state["speed_key"] != nxt["speed_key"]:
        clauses.append(f"motion pace evolves from {state['speed']} to {nxt['speed']}")
    if state["rig_key"] != nxt["rig_key"]:
        clauses.append(f"viewpoint support feel shifts from {state['rig']} to {nxt['rig']}")
    if state["stabilization_key"] != nxt["stabilization_key"]:
        clauses.append(f"motion feel evolves from {state['stabilization']} to {nxt['stabilization']}")
    if state["lens_key"] != nxt["lens_key"]:
        clauses.append(f"optical perspective evolves from {state['lens']} to {nxt['lens']}")
    if state["body_key"] != nxt["body_key"]:
        clauses.append(f"image character evolves from {state['body']} to {nxt['body']}")
    return clauses



_LONGMEDIA_TRANSITION_TYPES = {
    "Continuous / Same Shot": "continuous_same_shot",
    "Threshold Entry": "threshold_entry",
    "Occluded Hidden Cut": "hidden_cut",
    "Hard Cut": "hard_cut",
}

_LONGMEDIA_SPACE_RELATIONS = {
    "Same Space": "same_space",
    "Adjacent Space": "adjacent_space",
    "Different Space": "different_space",
}

_LONGMEDIA_ENTITY_CONTINUITY = {
    "Lock Population / Layout": "lock_population_layout",
    "Preserve Main Subjects": "preserve_main_subjects",
    "Allow Background Evolution": "allow_background_evolution",
}


def _lm_scene_continuity_contract(card, prev_card=None, next_card=None):
    transition_type = str(card.get("transition_type") or "Continuous / Same Shot")
    space_relation = str(card.get("space_relation") or "Same Space")
    entity_mode = str(card.get("entity_continuity") or "Lock Population / Layout")

    transition_key = _LONGMEDIA_TRANSITION_TYPES.get(transition_type, "continuous_same_shot")
    space_key = _LONGMEDIA_SPACE_RELATIONS.get(space_relation, "same_space")
    entity_key = _LONGMEDIA_ENTITY_CONTINUITY.get(entity_mode, "lock_population_layout")

    clauses = ["[SCENE CONTINUITY CONTRACT]"]

    if entity_key == "lock_population_layout":
        clauses.extend([
            "The established population, subject count, foreground/background occupancy, architecture, props, and major spatial landmarks are locked across this clip boundary.",
            "Do not introduce, spawn, reveal, duplicate, remove, or reposition people or major objects merely because the viewpoint moves.",
            "People already present may continue their existing actions naturally, but no new foreground figures may appear during camera travel.",
        ])
    elif entity_key == "preserve_main_subjects":
        clauses.extend([
            "Preserve all principal subjects, their identity, clothing, relative position, and action continuity across the boundary.",
            "Background population may evolve only gradually and must not pop into existence near the viewpoint path.",
        ])
    else:
        clauses.append(
            "Background details may evolve gradually, but principal subjects and major spatial landmarks must remain continuous."
        )

    if transition_key == "continuous_same_shot":
        clauses.extend([
            "This is one uninterrupted shot, not a new scene.",
            "Do not perform a semantic scene reset at the clip boundary.",
        ])
        if space_key == "same_space":
            clauses.extend([
                "Remain inside the exact same physical space throughout the transition.",
                "Do not jump from exterior to interior, from one room to another, or to a different location.",
                "If later prompt wording suggests a new location, defer that location change until a dedicated spatial transition is explicitly allowed.",
            ])
        elif space_key == "adjacent_space":
            clauses.extend([
                "The destination is an adjacent physically connected space.",
                "Reach it only by visibly traversing the connecting geometry; never teleport across the boundary.",
            ])
        else:
            clauses.append(
                "A different space is requested, but because this transition is continuous, the spatial change must be visibly traversed rather than cut or teleported."
            )

    elif transition_key == "threshold_entry":
        clauses.extend([
            "This transition crosses a visible physical threshold into an adjacent space.",
            "The threshold itself must remain continuously visible and spatially coherent while it is crossed.",
            "End the current clip at or entering the threshold; the next clip must begin from that exact threshold state and continue through it.",
            "Do not jump directly from outside to deep inside the destination.",
            "Reveal the destination progressively only after crossing the threshold.",
        ])

    elif transition_key == "hidden_cut":
        clauses.extend([
            "A concealed edit is allowed only when the entire frame is naturally occluded by a story-world element such as darkness, smoke, a passing foreground surface, or a full-frame light event.",
            "The outgoing and incoming motion direction must still match so the edit reads as intentional continuity.",
            "Do not expose the edit while the scene is unobstructed.",
        ])

    elif transition_key == "hard_cut":
        clauses.extend([
            "An intentional visible editorial cut is allowed at this boundary.",
            "Do not attempt to morph the old space into the new one.",
        ])

    return " ".join(clauses)


def _lm_transition_compatibility(card, next_card):
    if not isinstance(next_card, dict):
        return "final", []
    cur_move = str(card.get("movement") or "Locked-Off / Static")
    next_move = str(next_card.get("movement") or "Locked-Off / Static")
    transition_type = str(card.get("transition_type") or "Continuous / Same Shot")
    warnings = []

    opposites = {
        "Track Forward": {"Track Backward", "Pull-Out", "Rise + Pull-Out", "Descend + Pull-Out"},
        "Track Backward": {"Track Forward", "Push-In", "Rise + Push-In", "Descend + Push-In"},
        "Track Left": {"Track Right"},
        "Track Right": {"Track Left"},
        "Pan Left": {"Pan Right"},
        "Pan Right": {"Pan Left"},
        "Crane Up": {"Crane Down", "Descend + Push-In", "Descend + Pull-Out"},
        "Crane Down": {"Crane Up", "Rise + Push-In", "Rise + Pull-Out"},
        "Orbit Clockwise": {"Orbit Counterclockwise", "Full 360 Orbit Counterclockwise", "Half Orbit Counterclockwise"},
        "Orbit Counterclockwise": {"Orbit Clockwise", "Full 360 Orbit Clockwise", "Half Orbit Clockwise"},
    }
    if transition_type in ("Continuous / Same Shot", "Threshold Entry") and next_move in opposites.get(cur_move, set()):
        warnings.append(f"motion reversal: {cur_move} -> {next_move}")
    if transition_type == "Continuous / Same Shot":
        if str(card.get("space_relation") or "Same Space") == "Different Space":
            warnings.append("continuous transition requests Different Space")
    return ("warning" if warnings else "safe"), warnings


def _lm_camera_instruction(card, prev_card=None, next_card=None):
    state = _lm_camera_state_text(card)
    parts = ["[NON-DIEGETIC VIEWPOINT DIRECTION]"]

    if isinstance(prev_card, dict):
        prev = _lm_camera_state_text(prev_card)
        parts.extend([
            "At the opening frame, inherit the exact terminal viewpoint state and motion vector from the previous clip.",
            "Do not reset, teleport, snap, reframe, or restart the observed viewpoint at the clip boundary.",
            f"Opening continuity state: {prev['shot']} framing; {prev['movement']}; {prev['speed']}.",
            "Treat the settings below as the progressive target / dominant cinematography for this clip, reached smoothly from that inherited opening state.",
        ])
    else:
        parts.append("Establish the following cinematography as the opening viewpoint state.")

    parts.extend([
        f"Target framing: {state['shot']}.",
        f"Target viewpoint feel: {state['rig']}.",
        f"Image character: {state['body']}.",
        f"Optical perspective: {state['lens']}.",
        f"Motion feel: {state['stabilization']}.",
        f"Target viewpoint motion: {state['movement']}.",
        f"Target motion pace: {state['speed']}.",
        "Treat these as cinematography-only guidance for how the scene is observed.",
        _LONGMEDIA_CAMERA_VISIBILITY_GUARD,
        _lm_scene_continuity_contract(card, prev_card, next_card),
    ])
    text = " ".join(p for p in parts if p)

    if bool(card.get("transition_to_next")) and isinstance(next_card, dict):
        nxt = _lm_camera_state_text(next_card)
        clauses = _lm_camera_transition_clauses(state, nxt)
        transition_type = str(card.get("transition_type") or "Continuous / Same Shot")
        space_relation = str(card.get("space_relation") or "Same Space")
        compat, compat_warnings = _lm_transition_compatibility(card, next_card)

        transition_parts = [
            "[NON-DIEGETIC VIEWPOINT TRANSITION]",
            f"Transition contract: {transition_type}; spatial relation: {space_relation}.",
        ]

        if transition_type == "Hard Cut":
            transition_parts.extend([
                "Perform an intentional editorial cut at the boundary.",
                "Preserve identity and narrative continuity, but do not morph geometry between the two spaces.",
            ])
        elif transition_type == "Occluded Hidden Cut":
            transition_parts.extend([
                "Preserve continuous apparent velocity into a naturally full-frame occlusion and resume from matching apparent velocity after it.",
                "The edit itself must remain completely concealed.",
            ])
        else:
            transition_parts.extend([
                "During the final portion of this clip, preserve continuous velocity and smoothly hand the observed viewpoint into the next clip without a visible cut.",
                "The last frame of this clip and the first frame of the next clip must represent the same viewpoint state, orientation, spatial relation, scene population, and motion vector.",
                "Never stop and restart the motion at the boundary.",
            ])

        transition_parts.append(
            "Preserve subject identity, established population, scene geometry, lighting continuity, and action continuity while the viewpoint evolves."
        )

        if clauses:
            transition_parts.append("Camera transition goals: " + "; ".join(clauses) + ".")

        if compat == "warning":
            transition_parts.append(
                "Compatibility safeguard: do not execute any abrupt direction reversal at the boundary; inherit the outgoing vector first and evolve toward the new vector only after the next clip has begun."
            )

        if transition_type == "Threshold Entry":
            transition_parts.append(
                "The current clip must finish at the visible threshold, not deep inside the destination; the next clip begins from that same threshold and continues through it."
            )
        elif transition_type == "Continuous / Same Shot":
            transition_parts.append(
                "Do not use the clip boundary as permission to change location, population, architecture, or scene topology."
            )

        transition_parts.append(_LONGMEDIA_CAMERA_VISIBILITY_GUARD)
        text += " " + " ".join(transition_parts)

    return text


class MiniMaxH3LongMediaCameras:
    DESCRIPTION = (
        "Per-clip LongMedia camera direction with stable Planner clip identity: shot size, camera/capture profile, "
        "camera behavior and motion speed. Insert between LongMedia Planner and "
        "Long Media Setup, or use standalone."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "auto_sync_planner": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When Planner is connected, keep camera cards synchronized to its clip count.",
                }),
                "cameras_json": ("STRING", {
                    "default": json.dumps([_lm_camera_default_card(), _lm_camera_default_card()]),
                    "multiline": True,
                    "tooltip": "Internal serialized camera-card state.",
                }),
                "sync_request": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Internal one-shot Planner sync request.",
                }),
            },
            "optional": {
                "clip_plan": ("H3_LONGMEDIA_CLIP_PLAN", {
                    "tooltip": "Optional LongMedia Planner output. Connect Planner -> Cameras -> Setup.",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("H3_LONGMEDIA_CLIP_PLAN", "H3_LONGMEDIA_CAMERA_PLAN", "INT", "STRING")
    RETURN_NAMES = ("clip_plan", "camera_plan", "clip_count", "report")
    FUNCTION = "build"
    CATEGORY = CATEGORY_LONGMEDIA

    def build(self, auto_sync_planner, cameras_json, sync_request=False, clip_plan=None, unique_id=None):
        incoming = clip_plan if isinstance(clip_plan, dict) else None
        incoming_clips = incoming.get("clips") if incoming else None
        incoming_valid = (
            isinstance(incoming_clips, list)
            and len(incoming_clips) >= 1
            and str(incoming.get("kind") or "") == "h3_longmedia_clip_plan"
        )

        if incoming_valid:
            target_count = max(2, min(16, len(incoming_clips)))
        else:
            try:
                raw_cards = json.loads(str(cameras_json or "[]"))
                target_count = len(raw_cards) if isinstance(raw_cards, list) else 2
            except Exception:
                target_count = 2
            target_count = max(2, min(16, target_count))

        cards = _lm_parse_camera_cards(cameras_json, target_count)

        # v0.5.25: Planner clips carry stable clip_id values.  Auto Sync therefore
        # follows clip identity/order rather than merely matching the card count.
        # Old camera JSON without ids is migrated positionally once, then remains stable.
        if bool(auto_sync_planner) and incoming_valid:
            by_id = {str(c.get("clip_id") or ""): dict(c) for c in cards if str(c.get("clip_id") or "").strip()}
            ordered = []
            for idx, src in enumerate(incoming_clips[:target_count]):
                src = src if isinstance(src, dict) else {}
                cid = str(src.get("clip_id") or "").strip() or f"clip-{idx+1}"
                card = by_id.get(cid)
                if card is None:
                    # Migration path for pre-id workflows: preserve current camera
                    # settings by position instead of resetting a user's storyboard.
                    fallback = cards[idx] if idx < len(cards) else _lm_camera_default_card()
                    card = dict(fallback)
                card["clip_id"] = cid
                card["clip_name"] = str(src.get("name") or src.get("clip_name") or "").strip()[:120]
                ordered.append(card)
            cards = ordered

        if bool(sync_request) or (bool(auto_sync_planner) and incoming_valid):
            if unique_id is not None:
                try:
                    from server import PromptServer
                    PromptServer.instance.send_sync(
                        "minimax_h3_cameras_sync",
                        {
                            "node_id": str(unique_id),
                            "clip_count": int(target_count),
                            "clear_request": bool(sync_request),
                        },
                    )
                except Exception:
                    pass

        incoming_global_prompt = str(incoming.get("global_prompt") or "").strip() if incoming_valid else ""
        sanitized_global_prompt, removed_global_camera_sentences = _lm_strip_camera_directives(incoming_global_prompt)

        outgoing_clips = []
        camera_clips = []
        removed_local_camera_sentences = 0
        for idx in range(target_count):
            camera = dict(cards[idx])
            if idx >= target_count - 1:
                camera["transition_to_next"] = False
            prev_camera = dict(cards[idx - 1]) if idx > 0 else None
            next_camera = dict(cards[idx + 1]) if idx + 1 < target_count else None
            instruction = _lm_camera_instruction(camera, prev_camera, next_camera)
            if incoming_valid and idx < len(incoming_clips):
                src = incoming_clips[idx] if isinstance(incoming_clips[idx], dict) else {}
                base_prompt_raw = str(src.get("base_prompt") if "base_prompt" in src else src.get("prompt") or "").strip()
                clip = dict(src)
            else:
                base_prompt_raw = ""
                clip = {
                    "clip_id": str(camera.get("clip_id") or "").strip() or f"clip-{idx+1}",
                    "name": str(camera.get("clip_name") or "").strip()[:120],
                    "prompt": "", "duration": 7.5, "seed": None,
                }

            base_prompt, removed_camera_sentences = _lm_strip_camera_directives(base_prompt_raw)
            removed_local_camera_sentences += len(removed_camera_sentences)
            clip["base_prompt"] = base_prompt
            clip["camera"] = camera
            clip["camera_instruction"] = instruction
            clip["prompt"] = (base_prompt + "\n\n" + instruction).strip()
            outgoing_clips.append(clip)
            compatibility, compatibility_warnings = _lm_transition_compatibility(camera, next_camera)
            camera_clips.append({
                "index": idx + 1,
                **camera,
                "instruction": instruction,
                "transition_compatibility": compatibility,
                "transition_warnings": compatibility_warnings,
            })

        if incoming_valid:
            output_plan = dict(incoming)
            output_plan["source"] = "MiniMax H3 LongMedia Planner + LongMedia Cameras"
            output_plan["global_prompt"] = sanitized_global_prompt
            output_plan["clips"] = outgoing_clips
            output_plan["camera_plan"] = camera_clips
        else:
            output_plan = {
                "version": 1,
                "kind": "h3_longmedia_clip_plan",
                "source": "MiniMax H3 LongMedia Cameras (standalone)",
                "global_prompt": "",
                "clips": outgoing_clips,
                "camera_plan": camera_clips,
            }

        camera_plan = {
            "version": 1,
            "kind": "h3_longmedia_camera_plan",
            "source": "MiniMax H3 LongMedia Cameras",
            "clip_count": int(target_count),
            "clips": camera_clips,
        }
        report = json.dumps({
            "planner_connected": bool(incoming_valid),
            "clip_count": int(target_count),
            "camera_prompt_compiler": "scene_continuity_v5",
            "physical_rig_terms_emitted": False,
            "strict_hide_rig_guard": True,
            "planner_prompt_camera_sanitizer": True,
            "camera_boundary_state_lock": True,
            "camera_motion_vector_handoff": True,
            "scene_population_lock": True,
            "spatial_transition_contract": True,
            "threshold_entry_support": True,
            "removed_global_camera_sentences": len(removed_global_camera_sentences),
            "removed_local_camera_sentences": int(removed_local_camera_sentences),
            "camera_clips": camera_clips,
        }, indent=2)
        return (output_plan, camera_plan, int(target_count), report)


class MiniMaxH3LongMediaPlanner:
    """User-facing MultiClip script/timeline planner.

    The frontend renders clip cards; Python receives one stable JSON widget and
    emits an opaque H3_LONGMEDIA_CLIP_PLAN consumed by Long Media Setup.
    """

    DESCRIPTION = (
        'Build a MultiClip storyboard with a stable identity, name, prompt, duration, and optional seed per clip. '
        'Connect clip_plan to Long Media Setup; the Setup automatically uses MultiClip mode.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'global_prompt': ('STRING', {
                    'default': '',
                    'multiline': True,
                    'tooltip': 'Global Prompt shared by every MultiClip clip. Kept separate from editable per-clip prompts and joined only at runtime.',
                }),
                'multiclip_prompt': ('STRING', {
                    'default': '',
                    'multiline': True,
                    'tooltip': 'Multiple Clips Prompt import source. Use clip_1:/clip_2: or shot_1:/shot_2:. Import copies text into editable clip cards.',
                }),
                'multiclip_auto_import': ('BOOLEAN', {
                    'default': False,
                    'tooltip': 'Automatically import a valid structured Multiple Clips Prompt when that source changes.',
                }),
                'clips_json': ('STRING', {
                    'default': '[{"prompt":"","duration":7.5,"seed":null},{"prompt":"","duration":7.5,"seed":null}]',
                    'multiline': True,
                    'tooltip': 'Internal serialized clip-card state. The LongMedia Planner frontend hides this field and renders clip cards instead.',
                }),
                'multiclip_import_request': ('BOOLEAN', {
                    'default': False,
                    'tooltip': 'Internal one-shot manual import request used when Multiple Clips Prompt is connected to a dynamic source.',
                }),
                'multiclip_last_import_source': ('STRING', {
                    'default': '',
                    'multiline': True,
                    'tooltip': 'Internal source snapshot so Auto Import never overwrites edited cards unless the structured source actually changes.',
                }),
            },
            'hidden': {
                'unique_id': 'UNIQUE_ID',
            },
        }

    RETURN_TYPES = ('H3_LONGMEDIA_CLIP_PLAN', 'INT', 'FLOAT', 'STRING')
    RETURN_NAMES = ('clip_plan', 'clip_count', 'requested_seconds', 'report')
    FUNCTION = 'build'
    CATEGORY = CATEGORY_LONGMEDIA

    def build(self, global_prompt, multiclip_prompt, multiclip_auto_import, clips_json, multiclip_import_request=False, multiclip_last_import_source='', unique_id=None):
        clips = _v85_parse_multiclip_json(clips_json, '', 7.5)
        manual_import = bool(multiclip_import_request)
        automatic_import = bool(multiclip_auto_import)
        import_source = str(multiclip_prompt or '')
        source_changed = import_source != str(multiclip_last_import_source or '')
        if manual_import or (automatic_import and source_changed):
            imported = False
            try:
                clips, imported = _v043_import_multiclip_prompts(clips, multiclip_prompt, fallback_duration=7.5)
            except ValueError:
                # Import is a UI convenience; malformed import text must not destroy
                # an otherwise valid Planner execution or existing editable cards.
                imported = False
            if unique_id is not None:
                try:
                    from server import PromptServer
                    payload = {
                        'node_id': str(unique_id),
                        'clear_request': bool(manual_import),
                    }
                    if imported:
                        payload['prompts'] = [str(c.get('prompt') or '') for c in clips]
                        payload['source_text'] = import_source
                    PromptServer.instance.send_sync('minimax_h3_planner_prompt_import', payload)
                except Exception:
                    pass
        normalized = []
        for idx, clip in enumerate(clips):
            normalized.append({
                'clip_id': str(clip.get('clip_id') or '').strip() or f"clip-{idx+1}",
                'name': str(clip.get('name') or clip.get('clip_name') or '').strip()[:120],
                'prompt': str(clip.get('prompt') or ''),
                'duration': float(clip.get('duration') or 7.5),
                'seed': None if clip.get('seed') is None else int(clip.get('seed')),
            })
        if len(normalized) < 2:
            raise ValueError('MiniMax H3 LongMedia Planner requires at least 2 clips.')
        requested = sum(float(c['duration']) for c in normalized)
        plan = {
            'version': 1,
            'kind': 'h3_longmedia_clip_plan',
            'source': 'MiniMax H3 LongMedia Planner',
            'global_prompt': str(global_prompt or '').strip(),
            'clips': normalized,
        }
        report = json.dumps({
            'clip_count': len(normalized),
            'requested_seconds': requested,
            'clips': normalized,
        }, indent=2)
        return (plan, int(len(normalized)), float(requested), report)



def _longmedia_native_reference_execute_safe(native_cls, *, clip, **kwargs):
    """Run MiniMax H3 native reference conditioning with the old AIMDO pinned-memory gate.

    NativeReferenceToVideo invokes the large MiniMaxH3TEModel before the sampler
    memory policy exists. On Windows/AIMDO this can fail in HostBuffer.read_file_slice
    (GetOverlappedResult error 1450) when pinned host mappings are exhausted. Mirror
    the proven 0.3.59 sampler-local workaround around the TE/reference encode only,
    then restore the user's global ComfyUI setting.
    """
    previous = False
    changed = False
    try:
        from comfy.cli_args import args as _args
        previous = bool(getattr(_args, 'disable_pinned_memory', False))
        if not previous:
            _args.disable_pinned_memory = True
            changed = True

        # Release any already-established pins on CLIP/TE patchers before the
        # first AIMDO weight fault. Different ComfyUI versions expose the patcher
        # at slightly different locations, so probe conservatively.
        candidates = []
        for obj in (
            clip,
            getattr(clip, 'patcher', None),
            getattr(clip, 'model_patcher', None),
            getattr(clip, 'cond_stage_model', None),
            getattr(getattr(clip, 'cond_stage_model', None), 'patcher', None),
            getattr(getattr(clip, 'cond_stage_model', None), 'model_patcher', None),
        ):
            if obj is not None and obj not in candidates:
                candidates.append(obj)
        unpinned = 0
        for obj in candidates:
            fn = getattr(obj, 'unpin_all_weights', None)
            if callable(fn):
                try:
                    fn()
                    unpinned += 1
                except Exception as exc:
                    _lm_print(
                        '[MiniMaxH3 LongMedia][TE PINNED-MEMORY GATE] '
                        f'unpin warning {type(exc).__name__}: {exc}',
                        flush=True,
                    )
        try:
            import comfy.model_management as _mm
            if hasattr(_mm, 'soft_empty_cache'):
                _mm.soft_empty_cache()
        except Exception:
            pass
        _lm_print(
            '[MiniMaxH3 LongMedia][TE PINNED-MEMORY GATE] '
            f'disable_pinned_memory {previous}->True for NativeReferenceToVideo; '
            f'unpinned_patchers={unpinned}; restore_after_encode=True',
            flush=True,
        )
        return native_cls.execute(clip=clip, **kwargs)
    finally:
        if changed:
            try:
                from comfy.cli_args import args as _args
                _args.disable_pinned_memory = previous
                _lm_print(
                    '[MiniMaxH3 LongMedia][TE PINNED-MEMORY RESTORE] '
                    f'disable_pinned_memory restored to {previous}',
                    flush=True,
                )
            except Exception as exc:
                _lm_print(
                    '[MiniMaxH3 LongMedia][TE PINNED-MEMORY RESTORE] '
                    f'warning {type(exc).__name__}: {exc}',
                    flush=True,
                )


class MiniMaxH3LongMediaVideoReconstructor:
    """Build a lightweight source-video reconstruction contract for LongMedia Setup.

    This node intentionally does no VAE/model work. It only owns source media and
    chunk/fidelity policy, so the existing LongMedia sampler, continuation engine,
    VRAM governor, refiner and decoder remain the single execution backend.
    """

    DESCRIPTION = (
        'Prepare arbitrarily long low-quality source video for LongMedia neural reconstruction. '
        'The source is processed in local temporal chunks, so total video duration does not scale VRAM.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'source_video': ('IMAGE', {'tooltip': 'Full source video as an IMAGE batch. Frames are sliced lazily per LongMedia chunk.'}),
                'source_fps': ('FLOAT', {'default': 24.0, 'min': 1.0, 'max': 120.0, 'step': 0.001}),
                'profile': (['conservative', 'balanced', 'neural_remaster'], {'default': 'balanced'}),
                'source_fit': (
                    ['center_crop', 'stretch', 'strict'],
                    {
                        'default': 'center_crop',
                        'tooltip': 'How source frames are fit to the LongMedia target canvas. center_crop preserves geometry, stretch preserves the full frame but can distort aspect ratio, strict requires an exact size match.',
                    },
                ),
                'reconstruction_strength': (
                    'FLOAT',
                    {
                        'default': 0.55, 'min': 0.05, 'max': 1.0, 'step': 0.01,
                        'tooltip': 'Controls native Ref2VA source-reference authority for reconstruction. Balanced start: 0.55. Detail recovery is controlled separately below.',
                    },
                ),
                'detail_recovery': (
                    'BOOLEAN',
                    {
                        'default': True,
                        'tooltip': 'Run the dual-candidate Detail Recovery V3 after the second/global pass. It synthesizes structure detail and microtexture separately while low-frequency geometry, motion and audio remain locked.',
                    },
                ),
                'detail_strength': (
                    'FLOAT',
                    {
                        'default': 0.35, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                        'tooltip': 'Strength of bounded multi-band detail transfer from the dedicated detail pass. 0 disables it; 0.35-0.55 is the recommended restoration range.',
                    },
                ),
                'detail_steps': (
                    'INT',
                    {
                        'default': 3, 'min': 1, 'max': 8, 'step': 1,
                        'tooltip': 'Model evaluations used by the separate detail pass. Three is the recommended start; higher values cost more and can invent texture.',
                    },
                ),
                'segment_seconds': (
                    'FLOAT',
                    {
                        'default': 5.0, 'min': 1.0, 'max': 30.0, 'step': 0.5,
                        'tooltip': 'Local reconstruction window. Total source duration may be arbitrarily longer; VRAM follows this window, not total duration.',
                    },
                ),
                'overlap_frames': (
                    'INT',
                    {
                        'default': 22, 'min': 5, 'max': 360, 'step': 17,
                        'tooltip': 'Hidden temporal continuation context between reconstruction chunks.',
                    },
                ),
            },
            'optional': {
                'source_audio': ('AUDIO', {'tooltip': 'Optional original soundtrack. It is preserved untouched at final output and is not regenerated.'}),
            },
        }

    RETURN_TYPES = ('H3_LONGMEDIA_RECONSTRUCTION', 'STRING')
    RETURN_NAMES = ('reconstruction', 'report')
    FUNCTION = 'build'
    CATEGORY = CATEGORY_LONGMEDIA

    def build(self, source_video, source_fps, profile, source_fit, reconstruction_strength,
              detail_recovery, detail_strength, detail_steps,
              segment_seconds, overlap_frames, source_audio=None):
        if source_video is None or not hasattr(source_video, 'shape') or len(source_video.shape) != 4:
            raise ValueError('source_video must be a ComfyUI IMAGE batch [frames, height, width, channels].')
        frame_count = int(source_video.shape[0])
        if frame_count <= 0:
            raise ValueError('source_video contains no frames.')
        fps = float(source_fps)
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError(f'source_fps must be positive, got {source_fps!r}.')
        strength = max(0.05, min(1.0, float(reconstruction_strength)))
        detail_enabled = bool(detail_recovery)
        detail_strength = max(0.0, min(1.0, float(detail_strength)))
        detail_steps = max(1, min(8, int(detail_steps)))
        source_fit = str(source_fit or 'center_crop')
        if source_fit not in ('center_crop', 'stretch', 'strict'):
            raise ValueError(f'Unknown source_fit={source_fit!r}.')
        duration = float(frame_count) / fps
        payload = {
            'kind': 'h3_longmedia_reconstruction',
            'version': 1,
            'source_video': source_video,
            'source_audio': source_audio,
            'source_fps': fps,
            'profile': str(profile),
            'source_fit': source_fit,
            'reconstruction_strength': strength,
            'detail_recovery': detail_enabled,
            'detail_strength': detail_strength,
            'detail_steps': detail_steps,
            'segment_seconds': float(segment_seconds),
            'overlap_frames': int(overlap_frames),
            'frame_count': frame_count,
            'duration_seconds': duration,
        }
        report = json.dumps({
            'kind': payload['kind'],
            'version': payload['version'],
            'profile': payload['profile'],
            'source_fit': source_fit,
            'reconstruction_strength': strength,
            'detail_recovery': detail_enabled,
            'detail_strength': detail_strength,
            'detail_steps': detail_steps,
            'source_frames': frame_count,
            'source_fps': fps,
            'source_duration_seconds': duration,
            'segment_seconds': float(segment_seconds),
            'overlap_frames': int(overlap_frames),
            'source_audio_connected': bool(source_audio is not None),
            'execution_contract': 'streamed_local_source_windows_constant_vram',
            'source_authority_model': 'segmented_native_ref2va_edit_plus_post_global_dual_candidate_detail_v3',
        }, indent=2)
        return (payload, report)


class MiniMaxH3LatentLabLongMediaSetup:
    DESCRIPTION = (
        'Orchestrate long-media generation: build a multi-segment plan, '
        'encode source media, and set up references for NativeReferenceToVideo.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'clip': ('CLIP',),
                'vae': ('VAE',),
                'audio_vae': ('VAE',),
                'prompt': (
                    'STRING',
                    {
                        'default': '',
                        'multiline': True,
                        'tooltip': 'Prompt text. Can also be connected directly through the prompt input socket.',
                    },
                ),
                'width': (
                    'INT',
                    {'default': 512, 'min': 32, 'max': 8192, 'step': 32},
                ),
                'height': (
                    'INT',
                    {'default': 512, 'min': 32, 'max': 8192, 'step': 32},
                ),
                'manual_duration': (
                    'FLOAT',
                    {'default': 10.0, 'min': 0.1, 'max': 600.0, 'step': 0.1},
                ),
                'duration_source': (
                    ['auto', 'manual', 'audio', 'video', 'longest_input'],
                    {
                        'tooltip': (
                            'Controls target timeline length independently from audio_mode and prompt conditioning. '
                            'In video_ref_edit, auto follows video_1. video follows video_1 explicitly; audio follows audio_1; '
                            'manual uses manual_duration; longest_input follows the longest connected audio/video input. '
                            'Choosing a shorter source trims the target timeline; choosing a longer source allows the edited scene to continue beyond video_1. MultiClip Planner and Video Reconstructor own their own timeline lengths.'
                        ),
                    },
                ),
                'segment_seconds': (
                    'FLOAT',
                    {
                        'default': 5.0, 'min': 1.0, 'max': 60.0, 'step': 0.5,
                        'tooltip': 'New output timeline per segment. transition_frames is added as continuation context and does not reduce this duration.',
                    },
                ),
                'overlap_frames': (
                    'INT',
                    {'default': 22, 'min': 5, 'max': 3600, 'step': 17},
                ),
                'loop_closure_enabled': (
                    'BOOLEAN',
                    {
                        'default': False,
                        'tooltip': 'Context-preserving seamless loop closure. Runs one extra H3 tail pass guided toward the opening-frame macro context while keeping motion and fine detail free. Available in every workflow.'
                    },
                ),
                'loop_closure_frames': (
                    'INT',
                    {
                        'default': 57, 'min': 2, 'max': 720, 'step': 1,
                        'tooltip': 'Approximate tail region used for loop closure. It snaps to the nearest valid H3 frame count without unnecessarily expanding the ending.'
                    },
                ),
                'resolution_mode': (['match', 'max'],),
                'reference_budget': (['low', 'medium', 'high', 'max'],),
                'video_fps': (
                    'FLOAT',
                    {'default': 24.0, 'min': 1.0, 'max': 120.0, 'step': 1.0},
                ),
                'video_mode': (['auto', 'preserve', 'transform'],),
                'audio_mode': (['auto', 'preserve', 'generate', 'reference_only', 'preserve_reference', 'lip_sync'], {'tooltip': 'auto: legacy behavior; in video_ref_edit, connected audio_1 is also the authoritative source-performance timing clock and is restored untouched. preserve: restore original audio at output; in video_ref_edit it also locks replacement facial/mouth timing to audio_1. generate: generate final H3 audio. reference_only: use input audio as H3 reference but output generated audio. preserve_reference: use input audio as H3 reference/timing source and restore the untouched original track; in video_ref_edit it also locks source-performance sync. lip_sync: audio_1 stays native <Audio 1> Ref2VA content conditioning and is also time-anchored per clip; the untouched audio_1 is restored at output.'}),
                'conditioning_mode': (
                    ['auto_refs', 'hybrid_first_frame', 'hybrid_first_last', 'multiclip_ref2va'],
                    {
                        'default': 'auto_refs',
                        'tooltip': (
                            'auto_refs keeps the original LongMedia behavior. '
                            'hybrid_first_frame: image_1 is the opening keyframe, image_2..image_9 become '
                            '<Picture 1>..<Picture 8> identity/style refs. '
                            'hybrid_first_last: image_1 is the first keyframe, image_2 is the last keyframe, '
                            'and image_3..image_9 become <Picture 1>..<Picture 7> refs. '
                            'Manual mode only. auto_refs sends all connected images as Picture refs. hybrid_first_frame uses image_1 as the opening keyframe. hybrid_first_last uses image_1/image_2 as first/last keyframes. video_1..3 and audio_1..3 remain native H3 refs.'
                        ),
                    },
                ),
            },
            'optional': {
                'release_guard': (
                    'BOOLEAN',
                    {
                        'default': True,
                        'tooltip': (
                            'Production console guard. ON suppresses routine LongMedia diagnostics and keeps only actionable failures. '
                            'OFF prints the full internal execution/memory/attention diagnostics for profiling and A/B tests.'
                        ),
                    },
                ),
                'clip_plan': ('H3_LONGMEDIA_CLIP_PLAN', {
                    'tooltip': 'Connect MiniMax H3 LongMedia Planner. It is authoritative only when timeline_mode=multiclip; single/segmented timelines ignore the connected Planner.',
                }),
                'workflow_mode': (
                    ['hybrid_auto', 'segmented_continuation', 'multiclip', 'reconstruct', 'ref2va_full', 'loop', 'manual', 'video_ref_edit'],
                    {
                        'default': 'hybrid_auto',
                        'tooltip': (
                            'hybrid_auto: image_1 is first frame; if image_2 is connected it is last frame; '
                            'segmented_continuation: fixed-duration timeline policy using the exact same Ref2VA/Motion-Context clip executor as MultiClip; every generated clip has the same H3-aligned length and the final excess is trimmed. segment_duration controls fixed clip size; Planner is ignored. '
                            'remaining images are Picture refs. video_ref_edit: video_1 is the main motion/camera/composition '
                            'reference, image_1..9 are Picture refs for identity/style replacement, and audio_1 can carry the '
                            'paired source soundtrack. ref2va_full: all connected images are Picture refs with no first/last '
                            'anchors. loop: image_1 is reused as BOTH first and last frame for a seam-friendly viral loop; '
                            'image_2..9 are Picture refs. manual: exposes legacy conditioning and segmentation controls for '
                            'advanced diagnostics and A/B tests.'
                        ),
                    },
                ),
                'reconstruction': ('H3_LONGMEDIA_RECONSTRUCTION', {
                    'tooltip': 'Optional MiniMax H3 LongMedia Video Reconstructor contract. When connected it owns source video/audio, FPS, chunking and reconstruction strength.',
                }),
                'multiclip_json': ('STRING', {
                    'default': '[{"prompt":"","duration":7.5,"seed":null},{"prompt":"","duration":7.5,"seed":null}]',
                    'multiline': True,
                    'tooltip': 'MultiClip backend storage. JSON array: [{prompt, duration, seed}, ...]. Prompt is local to the clip; null seed uses sampler seed + clip index.',
                }),
                'image_1': ('IMAGE', {'tooltip': 'Native MiniMax H3 <Picture 1> reference.'}),
                'image_2': ('IMAGE', {'lazy': True, 'tooltip': 'Native MiniMax H3 <Picture 2> reference.'}),
                'image_3': ('IMAGE', {'lazy': True, 'tooltip': 'Native MiniMax H3 <Picture 3> reference.'}),
                'image_4': ('IMAGE', {'lazy': True, 'tooltip': 'Native MiniMax H3 <Picture 4> reference.'}),
                'image_5': ('IMAGE', {'lazy': True, 'tooltip': 'Native MiniMax H3 <Picture 5> reference.'}),
                'image_6': ('IMAGE', {'lazy': True, 'tooltip': 'Native MiniMax H3 <Picture 6> reference.'}),
                'image_7': ('IMAGE', {'lazy': True, 'tooltip': 'Native MiniMax H3 <Picture 7> reference.'}),
                'image_8': ('IMAGE', {'lazy': True, 'tooltip': 'Native MiniMax H3 <Picture 8> reference.'}),
                'image_9': ('IMAGE', {'lazy': True, 'tooltip': 'Native MiniMax H3 <Picture 9> reference.'}),
                'video_1': ('IMAGE', {'lazy': True, 'tooltip': 'Video frames only (IMAGE batch). video_ref_edit: primary motion/camera/composition source as <Video 1>. Other modes: regular <Video 1> reference. If the source video has audio, connect that extracted audio to audio_1.'}),
                'video_2': ('IMAGE', {'lazy': True, 'tooltip': 'Video frames only (IMAGE batch). Passed as <Video 2> reference. Pair with audio_2 when they come from the same source.'}),
                'video_3': ('IMAGE', {'lazy': True, 'tooltip': 'Video frames only (IMAGE batch). Passed as <Video 3> reference. Pair with audio_3 when they come from the same source.'}),
                'audio_1': ('AUDIO', {'tooltip': 'Native MiniMax H3 <Audio 1> reference. In video_ref_edit with auto/preserve/preserve_reference, audio_1 is the paired source-performance clock and can be addressed from the prompt. With audio_mode=lip_sync, Audio 1 is an independent authoritative dub/timing source rather than the soundtrack paired to Video 1. duration_source controls timeline length separately.'}),
                'audio_2': ('AUDIO', {'lazy': True, 'tooltip': 'Optional second native H3 audio reference. In video_ref_edit it remains standalone <Audio 2> prompt conditioning, independent from Audio 1 timing/output ownership.'}),
                'audio_3': ('AUDIO', {'lazy': True, 'tooltip': 'Optional third native H3 audio reference. In video_ref_edit it remains standalone <Audio 3> prompt conditioning, independent from Audio 1 timing/output ownership.'}),
                'loop_closure_strength': (
                    'FLOAT',
                    {
                        'default': 0.65, 'min': 0.0, 'max': 1.0, 'step': 0.05,
                        'tooltip': 'Structural attraction of the regenerated tail toward the opening-frame context. 0 keeps the existing ending; 1 applies the strongest macro-geometry guidance. Fine detail remains free.'
                    },
                ),
                'control_mode': (
                    ['auto', 'manual'],
                    {
                        'default': 'auto',
                        'tooltip': 'Setup control surface. auto derives safe internal policies from H3 Mode + Timeline. manual exposes the advanced Setup-only conditioning/transition controls.',
                    },
                ),
                'h3_mode': (
                    ['t2va', 'fl2va', 'ref2va', 'hybrid', 'video_ref_edit'],
                    {
                        'default': 'hybrid',
                        'tooltip': 'Native H3 conditioning family. FL2VA uses image_1 as first frame and optional image_2 as last frame without LongMedia latent injection. Ref2VA keeps images as Picture refs. Hybrid enables LongMedia first-frame injection controls. video_ref_edit uses video_1 as the source motion/camera/composition stream.',
                    },
                ),
                'timeline_mode': (
                    ['single', 'segmented', 'multiclip'],
                    {
                        'default': 'single',
                        'tooltip': 'LongMedia timeline only. single is one H3 pass; segmented uses fixed-duration continuation clips; multiclip consumes Clip Plan / per-clip prompts. It does not choose the H3 conditioning family.',
                    },
                ),
                'transition_frames': (
                    'INT',
                    {
                        'default': 22, 'min': 5, 'max': 3600, 'step': 17,
                        'tooltip': 'Transition/context length between adjacent Segmented or MultiClip units. Native H3 values are 5, 22, 39, 56... frames. Replaces the old hidden overlap_frames control for public timeline modes.',
                    },
                ),
                'first_frame_mode': (
                    ['native_keyframe', 'latent_inject', 'pixel_override', 'blend'],
                    {
                        'default': 'latent_inject',
                        'tooltip': 'Hybrid image injection policy. native_keyframe is pure native H3 frame conditioning; latent_inject seeds the leading target latent; pixel_override/blend are decode-time opening-frame policies. FL2VA always forces native_keyframe.',
                    },
                ),
                'first_frame_denoise': (
                    'FLOAT',
                    {
                        'default': 0.25, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                        'tooltip': 'Hybrid latent-injection denoise amount. Used only when first_frame_mode=latent_inject.',
                    },
                ),
                'first_frame_blend_frames': (
                    'INT',
                    {
                        'default': 3, 'min': 1, 'max': 17, 'step': 1,
                        'tooltip': 'Hybrid decode blend span. Used only when first_frame_mode=blend.',
                    },
                ),
            },
            'hidden': {
                'unique_id': 'UNIQUE_ID',
            },
        }

    RETURN_TYPES = ('CONDITIONING', 'LATENT', 'LONG_MEDIA_PLAN', 'FLOAT', 'INT', 'STRING')
    RETURN_NAMES = ('positive', 'long_media_av', 'long_media_plan', 'duration_seconds', 'passes', 'report')
    FUNCTION = 'setup'
    CATEGORY = CATEGORY_LONGMEDIA

    @classmethod
    def check_lazy_status(cls, reference_budget='low', video_mode='auto', audio_mode='auto',
                          duration_source='auto', generation_mode='auto', **kwargs):
        # Only request inputs that are actually connected in the graph (present in kwargs).
        # Never request an unconnected input — ComfyUI crashes with NodeInputError.
        candidates = [
            'image_1', 'image_2', 'image_3', 'image_4', 'image_5', 'image_6', 'image_7', 'image_8', 'image_9',
            'video_1', 'video_2', 'video_3',
            'audio_1', 'audio_2', 'audio_3',
        ]
        return [name for name in candidates if name in kwargs]

    def setup(self, clip, vae, audio_vae, prompt, width, height, manual_duration,
              duration_source, segment_seconds, overlap_frames, loop_closure_enabled, loop_closure_frames, resolution_mode,
              reference_budget, video_fps, video_mode, audio_mode,
              release_guard=True, clip_plan=None, reconstruction=None, workflow_mode='hybrid_auto', generation_mode='auto', conditioning_mode='auto_refs',
              control_mode=None, h3_mode=None, timeline_mode=None, transition_frames=22,
              first_frame_mode='latent_inject',
              first_frame_denoise=0.25, first_frame_blend_frames=3,
              opening_frame=None, multiclip_json=None,
              image_1=None, image_2=None, image_3=None, image_4=None, image_5=None,
              image_6=None, image_7=None, image_8=None, image_9=None,
              video_1=None, video_2=None, video_3=None,
              audio_1=None, audio_2=None, audio_3=None, loop_closure_strength=0.65, unique_id=None):
        global NativeReferenceToVideo

        _set_longmedia_release_guard(bool(release_guard))
        if not bool(release_guard):
            builtins.print(
                '[MiniMaxH3 LongMedia] full diagnostics enabled',
                flush=True,
            )

        setup_memory_events = []
        # Start Setup from a clean model residency state.  This is especially
        # important when re-running a workflow after H3 occupied most of VRAM.
        setup_memory_events.append(_setup_memory_isolation('setup_entry', unload_models=True))

        effective_prompt = prompt
        segment0_prompt = None
        hybrid_artifacts = None
        audio_mode = str(audio_mode or 'auto')
        # v0.3.95: lip-sync is an audio policy, not a separate generation workflow.
        # Migrate legacy workflows that still carry generation_mode=lip_sync.
        if str(generation_mode or 'auto') == 'lip_sync' and audio_mode != 'lip_sync':
            audio_mode = 'lip_sync'
        lip_sync_enabled = audio_mode == 'lip_sync'
        # video_ref_edit has an additional paired-source contract resolved after
        # semantic H3 mode migration.  Keep it separate from explicit lip_sync so
        # preserve/auto retain their normal output semantics in every other mode.
        video_ref_edit_audio_sync = False
        video_ref_edit_conditioning_audio_count = 0
        preserve_audio_output = audio_mode in ('preserve', 'preserve_reference')
        use_audio_as_reference = audio_mode != 'preserve'

        reconstruction_cfg = reconstruction if isinstance(reconstruction, dict) else None
        reconstruction_strength = 1.0
        reconstruction_profile = 'balanced'
        reconstruction_guidance = 'segmented_ref2va_edit'
        reconstruction_resize_mode = 'center_crop'
        reconstruction_detail_enabled = True
        reconstruction_detail_strength = 0.35
        reconstruction_detail_steps = 3
        if reconstruction_cfg is not None:
            if str(reconstruction_cfg.get('kind') or '') != 'h3_longmedia_reconstruction' or int(reconstruction_cfg.get('version', 0) or 0) != 1:
                raise ValueError('Invalid H3_LONGMEDIA_RECONSTRUCTION payload. Rebuild it with MiniMax H3 LongMedia Video Reconstructor.')
            source_video_cfg = reconstruction_cfg.get('source_video')
            if source_video_cfg is None or not hasattr(source_video_cfg, 'shape') or int(source_video_cfg.shape[0]) <= 0:
                raise ValueError('Video Reconstructor requires a non-empty source_video IMAGE batch.')
            video_1 = source_video_cfg
            source_audio_cfg = reconstruction_cfg.get('source_audio')
            if source_audio_cfg is not None:
                audio_1 = source_audio_cfg
            video_fps = float(reconstruction_cfg.get('source_fps', video_fps))
            segment_seconds = float(reconstruction_cfg.get('segment_seconds', segment_seconds))
            overlap_frames = int(reconstruction_cfg.get('overlap_frames', overlap_frames))
            reconstruction_strength = max(0.0, min(1.0, float(reconstruction_cfg.get('reconstruction_strength', 0.55))))
            reconstruction_profile = str(reconstruction_cfg.get('profile') or 'balanced')
            reconstruction_guidance = 'segmented_ref2va_edit'
            reconstruction_detail_enabled = bool(reconstruction_cfg.get('detail_recovery', True))
            reconstruction_detail_strength = max(0.0, min(1.0, float(reconstruction_cfg.get('detail_strength', 0.35))))
            reconstruction_detail_steps = max(1, min(8, int(reconstruction_cfg.get('detail_steps', 3))))
            _source_fit = str(reconstruction_cfg.get('source_fit') or 'center_crop')
            reconstruction_resize_mode = 'none' if _source_fit == 'strict' else _source_fit
            if reconstruction_resize_mode not in ('none', 'stretch', 'center_crop'):
                raise ValueError(f'Invalid reconstruction source_fit={_source_fit!r}.')
            workflow_mode = 'reconstruct'
            duration_source = 'video'
            video_mode = 'transform'
            # Reconstruction should preserve the source soundtrack bit-for-bit unless
            # the user intentionally leaves source_audio disconnected.
            if source_audio_cfg is not None:
                audio_mode = 'preserve'
                preserve_audio_output = True
                use_audio_as_reference = False
            profile_prompts = {
                'conservative': (
                    'The target video is a faithfully restored version of <Video 1>. '
                    '<Video 1> is fully preserved for subjects, identity, action, timing, camera motion, composition, framing, wardrobe, background and shot order. '
                    'Only remove degradation and recover genuinely missing natural detail; do not redesign or invent events.'
                ),
                'balanced': (
                    'The target video is an enhanced restored version of <Video 1>. '
                    '<Video 1> is fully preserved for subjects, identity, exact action, choreography, timing, camera motion, composition, framing, wardrobe, background and shot order. '
                    'Remove blur, compression artifacts and noise, reconstruct natural faces, edges and textures, and add plausible fine detail without changing events.'
                ),
                'neural_remaster': (
                    'The target video is a high-quality neural remaster of <Video 1>. '
                    '<Video 1> is fully preserved for subjects, identity, exact action, choreography, timing, camera motion, composition, framing, wardrobe, background and shot order. '
                    'Aggressively rebuild lost visual detail and texture while keeping the source video content and motion unchanged; do not invent new events.'
                ),
            }
            scaffold = profile_prompts.get(reconstruction_profile, profile_prompts['balanced'])
            if str(prompt or '').strip():
                prompt = scaffold + ' ' + str(prompt).strip()
            else:
                prompt = scaffold
            effective_prompt = prompt

        images = [v for v in [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9] if v is not None]
        segmented_opening_frame = opening_frame
        videos = [v for v in [video_1, video_2, video_3] if v is not None]
        audios = [a for a in [audio_1, audio_2, audio_3] if a is not None]
        if lip_sync_enabled:
            if image_1 is None or audio_1 is None:
                missing = []
                if image_1 is None:
                    missing.append('image_1')
                if audio_1 is None:
                    missing.append('audio_1')
                raise ValueError(
                    "audio_mode='lip_sync' requires connected image_1 and audio_1; missing: "
                    + ', '.join(missing)
                )
            # Audio 1 is the authoritative speech/singing clock. Audio 2/3 remain
            # ordinary native H3 audio references so prompts can address additional
            # music, percussion, bass, ambience, etc. They never replace Audio 1's
            # lip-sync timing authority or the clean final Audio 1 output track.

        # v0.3.49: connected INT nodes can bypass the UI's step=32 constraint.
        # Normalize every target canvas here so all downstream H3 paths see the
        # same patch-safe geometry. This is a no-op for existing 32px canvases.
        width, height, h3_target_geometry = _h3_safe_target_canvas(width, height)
        safe_images, h3_ref_geometry = _h3_safe_reference_images(images)
        safe_image_by_id = {id(src): safe for src, safe in zip(images, safe_images)}

        # v0.5.23 Setup architecture: conditioning, timeline and loop are
        # independent user concepts.  The sampler/runtime contract remains unchanged;
        # Setup translates the public controls back into the proven legacy plan values.
        legacy_workflow_mode = str(workflow_mode or 'hybrid_auto')
        control_mode = 'legacy' if control_mode is None else str(control_mode)
        h3_mode = 'legacy' if h3_mode is None else str(h3_mode)
        timeline_mode = 'legacy' if timeline_mode is None else str(timeline_mode)

        # Backend migration for workflows saved before the semantic Setup controls
        # existed.  Frontend performs the same migration visually, but this keeps API/
        # headless execution deterministic too.
        legacy_loop_anchor = False
        if control_mode == 'legacy' or h3_mode == 'legacy' or timeline_mode == 'legacy':
            legacy_map = {
                'hybrid_auto': ('auto', 'hybrid', 'single'),
                'segmented_continuation': ('auto', 'ref2va', 'segmented'),
                'multiclip': ('auto', 'ref2va', 'multiclip'),
                'ref2va_full': ('auto', 'ref2va', 'single'),
                'video_ref_edit': ('auto', 'video_ref_edit', 'single'),
                'manual': ('manual', 'hybrid', 'segmented'),
                'loop': ('auto', 'hybrid', 'single'),
                'reconstruct': ('auto', 'video_ref_edit', 'segmented'),
            }
            migrated = legacy_map.get(legacy_workflow_mode, ('auto', 'hybrid', 'single'))
            if control_mode == 'legacy':
                control_mode = migrated[0]
            if h3_mode == 'legacy':
                h3_mode = migrated[1]
            if timeline_mode == 'legacy':
                timeline_mode = migrated[2]
            legacy_loop_anchor = legacy_workflow_mode == 'loop'

        if control_mode not in ('auto', 'manual'):
            raise ValueError(f'Unknown control_mode={control_mode!r}')
        if h3_mode not in ('t2va', 'fl2va', 'ref2va', 'hybrid', 'video_ref_edit'):
            raise ValueError(f'Unknown h3_mode={h3_mode!r}')
        if timeline_mode not in ('single', 'segmented', 'multiclip'):
            raise ValueError(f'Unknown timeline_mode={timeline_mode!r}')

        # Old workflow_mode=loop may already have been visually migrated by JS,
        # leaving no 'legacy' sentinel for the backend migration branch above.
        # Preserve its exact image_1 first+last semantics only while the migrated
        # semantic controls still match that legacy profile; changing H3/timeline
        # automatically exits the compatibility behavior.
        legacy_loop_anchor = bool(
            legacy_loop_anchor or (
                legacy_workflow_mode == 'loop'
                and control_mode == 'auto'
                and h3_mode == 'hybrid'
                and timeline_mode == 'single'
            )
        )

        # Native H3 continuation geometry: public transition values are snapped to
        # the nearest 5+17*k frame count.  Normal UI values are already exact.
        requested_transition_frames = max(5, int(transition_frames))
        transition_k = max(0, int(round((requested_transition_frames - 5) / 17.0)))
        effective_transition_frames = 5 + 17 * transition_k

        # Reconstruction is an attached Setup contract and remains authoritative.
        if reconstruction_cfg is not None:
            workflow_mode = 'reconstruct'
            timeline_mode = 'segmented'
            h3_mode = 'video_ref_edit'
            segmentation_active = True
        else:
            segmentation_active = timeline_mode == 'segmented'
            if timeline_mode == 'multiclip':
                workflow_mode = 'multiclip'
            elif timeline_mode == 'segmented':
                workflow_mode = 'segmented_continuation'
            elif h3_mode == 'video_ref_edit':
                workflow_mode = 'video_ref_edit'
            elif h3_mode in ('ref2va', 't2va'):
                workflow_mode = 'ref2va_full'
            else:
                # Pure FL2VA and LongMedia Hybrid both reuse the proven hybrid
                # conditioning implementation; h3_mode below decides whether image
                # injection is native-only or LongMedia-enhanced.
                workflow_mode = 'hybrid_auto'

        external_clip_plan = clip_plan if isinstance(clip_plan, dict) else None
        planner_global_prompt = ''
        if timeline_mode == 'multiclip' and external_clip_plan is not None:
            kind = str(external_clip_plan.get('kind') or '')
            version = int(external_clip_plan.get('version', 0) or 0)
            clips_payload = external_clip_plan.get('clips')
            if kind != 'h3_longmedia_clip_plan' or version != 1 or not isinstance(clips_payload, list):
                raise ValueError('Invalid H3_LONGMEDIA_CLIP_PLAN payload. Rebuild it with MiniMax H3 LongMedia Planner.')
            multiclip_json = json.dumps(clips_payload, ensure_ascii=False)
            planner_global_prompt = str(external_clip_plan.get('global_prompt') or '').strip()
            planner_has_camera_guidance = bool(external_clip_plan.get('camera_plan')) or any(
                isinstance(c, dict) and c.get('camera_instruction') for c in clips_payload
            )
            if planner_has_camera_guidance:
                planner_global_prompt, _removed_planner_camera_sentences = _lm_strip_camera_directives(planner_global_prompt)
            _lm_print(
                f'[MiniMaxH3 LongMedia][PLANNER] timeline=multiclip; external clip_plan selected; clips={len(clips_payload)}',
                flush=True,
            )
        elif external_clip_plan is not None:
            _lm_print(
                f'[MiniMaxH3 LongMedia][PLANNER] clip_plan connected but timeline={timeline_mode}; planner is ignored',
                flush=True,
            )
        _lm_print(
            f'[MiniMaxH3 LongMedia][SETUP OWNERSHIP] control={control_mode}; h3={h3_mode}; timeline={timeline_mode}; '
            f'legacy_workflow={legacy_workflow_mode}; internal_workflow={workflow_mode}; '
            f'planner_connected={external_clip_plan is not None}; '
            f'planner_active={bool(timeline_mode == "multiclip" and external_clip_plan is not None)}; '
            f'transition={effective_transition_frames}f',
            flush=True,
        )

        loop_last_override = None
        multiclip_clips = None

        if control_mode == 'manual':
            # Manual exposes the legacy low-level conditioning selector without
            # turning Manual itself into a timeline/workflow mode.
            conditioning_mode = str(conditioning_mode or 'auto_refs')
            if conditioning_mode not in ('auto_refs', 'hybrid_first_frame', 'hybrid_first_last', 'multiclip_ref2va'):
                raise ValueError(f'Unknown manual conditioning_mode={conditioning_mode!r}')
        elif h3_mode == 't2va':
            active_ref_audio = bool(use_audio_as_reference and audios)
            if images or videos or active_ref_audio:
                raise ValueError(
                    "h3_mode='t2va' is a pure text-to-video/audio conditioning family. "
                    "Disconnect image/video references and reference audio, or choose ref2va/hybrid."
                )
            conditioning_mode = 'multiclip_ref2va' if timeline_mode in ('segmented', 'multiclip') else 'auto_refs'
        elif h3_mode == 'fl2va':
            if image_1 is None:
                raise ValueError("h3_mode='fl2va' requires image_1 as the native first frame.")
            extra_stills = [v for v in [image_3, image_4, image_5, image_6, image_7, image_8, image_9] if v is not None]
            if extra_stills or videos or (use_audio_as_reference and audios):
                raise ValueError(
                    "h3_mode='fl2va' is the pure native first/last-frame family. "
                    "Use only image_1 and optional image_2; choose hybrid/ref2va when additional refs are needed."
                )
            conditioning_mode = 'hybrid_first_last' if image_2 is not None else 'hybrid_first_frame'
            # Critical separation: FL2VA never inherits LongMedia's image injection.
            first_frame_mode = 'native_keyframe'
        elif h3_mode == 'ref2va':
            conditioning_mode = 'multiclip_ref2va' if timeline_mode in ('segmented', 'multiclip') else 'auto_refs'
        elif h3_mode == 'hybrid':
            if image_1 is None:
                raise ValueError("h3_mode='hybrid' requires image_1 as the opening keyframe.")
            conditioning_mode = 'hybrid_first_last' if image_2 is not None else 'hybrid_first_frame'
        elif h3_mode == 'video_ref_edit':
            if not videos:
                raise ValueError("h3_mode='video_ref_edit' requires video_1 as the source clip.")
            if timeline_mode != 'single' and reconstruction_cfg is None:
                raise ValueError(
                    "video_ref_edit currently owns a single source-video timeline. "
                    "Segmented source-window editing remains the Video Reconstructor contract; "
                    "choose timeline=single or connect the Reconstructor."
                )
            if reconstruction_cfg is None and audio_mode in ('preserve', 'preserve_reference') and audio_1 is None:
                raise ValueError(
                    f"h3_mode='video_ref_edit' with audio_mode={audio_mode!r} requires audio_1. "
                    "video_1 contains IMAGE frames only; connect the source soundtrack separately to audio_1."
                )
            video_ref_edit_audio_sync = bool(
                reconstruction_cfg is None
                and audio_1 is not None
                and audio_mode in ('auto', 'preserve', 'preserve_reference')
            )
            if video_ref_edit_audio_sync:
                effective_prompt = _build_video_ref_edit_audio_sync_prompt(effective_prompt)
                _lm_print(
                    '[MiniMaxH3 LongMedia][VIDEO EDIT SOURCE AV SYNC] armed; '
                    f'audio_mode={audio_mode}; Audio1 will be locked into the target AV timeline while the untouched waveform is preserved at output',
                    flush=True,
                )
            conditioning_mode = 'auto_refs'

        # Compatibility for old workflow_mode=loop only.  New Loop Closure is an
        # independent boundary policy and never rewrites H3 conditioning.
        if legacy_loop_anchor and control_mode != 'manual':
            if image_1 is None:
                raise ValueError("Legacy workflow_mode='loop' requires image_1.")
            conditioning_mode = 'hybrid_first_last'
            loop_last_override = image_1

        # Timeline owns clip planning independently of H3 conditioning.  Planner
        # payload is authoritative when connected; multiclip_json remains legacy
        # storage/fallback for saved workflows.
        if timeline_mode == 'multiclip':
            local_clips = _v85_parse_multiclip_json(multiclip_json, '', manual_duration)
            global_prompt_effective = planner_global_prompt if external_clip_plan is not None else str(prompt or '').strip()
            multiclip_clips = tuple({
                **dict(c),
                'prompt': _v043_join_global_local_prompt(global_prompt_effective, c.get('prompt', '')),
            } for c in local_clips)
            effective_prompt = multiclip_clips[0]['prompt']

        effective_segment_seconds = float(segment_seconds)
        effective_overlap_frames = int(effective_transition_frames) if segmentation_active else 0
        effective_manual_duration = float(manual_duration)
        effective_duration_source = duration_source
        if h3_mode == 'video_ref_edit' and reconstruction_cfg is None and str(effective_duration_source) == 'auto':
            # Source-video editing owns the visual timeline.  The soundtrack may be
            # a few encoder/mux samples shorter or longer than the IMAGE batch; using
            # audio as the duration authority can silently truncate the final source
            # performance.  Keep video_1 authoritative and fit/restore audio against it.
            effective_duration_source = 'video'
            _lm_print(
                '[MiniMaxH3 LongMedia][VIDEO EDIT TIMELINE] duration_source auto->video; '
                'video_1 owns the complete source-performance horizon',
                flush=True,
            )
        if timeline_mode == 'multiclip':
            effective_manual_duration = float(multiclip_clips[0]['duration'])
            effective_segment_seconds = float(multiclip_clips[0]['duration'])
            effective_duration_source = 'manual'

        plan = build_media_plan(
            audios=audios,
            videos=videos,
            manual_duration=effective_manual_duration,
            duration_source=effective_duration_source,
            segment_seconds=effective_segment_seconds,
            overlap_frames=effective_overlap_frames,
            video_fps=float(video_fps),
            resolution_mode=resolution_mode,
        )
        if control_mode != 'manual' and h3_mode in ('t2va', 'ref2va'):
            # T2VA and Ref2VA are conditioning families, not source-stream editing.
            # Ref2VA video/audio sockets stay native references instead of turning
            # the target into the legacy video_to_video/audio_to_video route.
            plan = _dc_replace(plan, mode='t2v', source_video=None, source_audio=None)

        plan = _dc_replace(
            plan,
            workflow_mode=workflow_mode,
            segmentation_active=bool(segmentation_active),
            reconstruction_active=bool(workflow_mode == 'reconstruct'),
            reconstruction_strength=float(reconstruction_strength if workflow_mode == 'reconstruct' else 1.0),
            reconstruction_profile=str(reconstruction_profile),
            reconstruction_guidance=str(reconstruction_guidance if workflow_mode == 'reconstruct' else 'segmented_ref2va_edit'),
            reconstruction_resize_mode=str(reconstruction_resize_mode),
            reconstruction_target_width=int(width if workflow_mode == 'reconstruct' else 0),
            reconstruction_target_height=int(height if workflow_mode == 'reconstruct' else 0),
            loop_closure_enabled=bool(loop_closure_enabled),
            loop_closure_frames=max(2, int(loop_closure_frames)),
            loop_closure_strength=max(0.0, min(1.0, float(loop_closure_strength))),
            reconstruction_detail_enabled=bool(reconstruction_detail_enabled if workflow_mode == 'reconstruct' else False),
            reconstruction_detail_strength=float(reconstruction_detail_strength if workflow_mode == 'reconstruct' else 0.0),
            reconstruction_detail_steps=int(reconstruction_detail_steps if workflow_mode == 'reconstruct' else 1),
            reconstruction_audio_locked=bool(workflow_mode == 'reconstruct' and audio_1 is not None),
            release_guard=bool(release_guard),
        )

        # Strict isolation: all ordinary non-segmentation workflows are exactly one
        # H3 pass regardless of segment_duration. MultiClip owns its own clip count
        # below and is intentionally excluded from segmentation.
        if (not segmentation_active) and timeline_mode != 'multiclip':
            full_frames = int(align_frame_count(max(5, int(plan.output_frames))))
            plan = _dc_replace(
                plan,
                segment_frames=full_frames,
                segment_lengths=(full_frames,),
                segment_starts=(0,),
                overlap_frames=0,
                step_frames=full_frames,
                passes=1,
                generated_frames=full_frames,
                trim_frames=max(0, full_frames - int(plan.output_frames)),
                timeline_policy='single',
            )
            _lm_print(
                '[MiniMaxH3 LongMedia][SEGMENTATION ISOLATION] '
                f'workflow={workflow_mode}; segmentation_active=False; passes=1; '
                'segment_duration_ignored=True; overlap_ignored=True',
                flush=True,
            )

        if timeline_mode == 'segmented':
            multiclip_clips, fixed_lengths, fixed_starts, fixed_generated = _v111_build_fixed_clip_specs(
                float(plan.total_duration), float(effective_segment_seconds), int(plan.overlap_frames), effective_prompt,
            )
            plan = _dc_replace(
                plan,
                mode='multiclip', duration_basis=f'{plan.duration_basis}:fixed_segments',
                workflow_mode='segmented_continuation', segmentation_active=True,
                segment_frames=int(fixed_lengths[0]), segment_lengths=tuple(fixed_lengths),
                segment_starts=tuple(fixed_starts), overlap_frames=int(plan.overlap_frames),
                step_frames=max(1, int(fixed_lengths[0]) - int(plan.overlap_frames)),
                passes=len(fixed_lengths), generated_frames=int(fixed_generated),
                trim_frames=max(0, int(fixed_generated) - int(plan.output_frames)),
                segment_seeds=tuple(None for _ in fixed_lengths), timeline_policy='fixed',
            )
            _lm_print(
                '[MiniMaxH3 LongMedia][UNIFIED CLIP ENGINE] '
                f'workflow=segmented_continuation timeline=fixed clips={len(fixed_lengths)} '
                f'length={int(fixed_lengths[0])}f starts={list(fixed_starts)} overlap={int(plan.overlap_frames)}f '
                f'final={int(plan.output_frames)}f generated={int(fixed_generated)}f trim={int(plan.trim_frames)}f; '
                'executor=multiclip',
                flush=True,
            )

        if timeline_mode == 'multiclip':
            # v0.4.21: MultiClip is assembled on the native H3 temporal lattice and
            # decoded ONCE by VideoVAE. H3's smallest complete continuation prefix is
            # 5 decoded frames == 2 video-latent tokens. Every clip after the first
            # carries that real continuation head; the head is removed in latent space
            # before the single continuous decode. This keeps the concatenated video
            # latent on T=5*k+2 and avoids resetting VideoVAE temporal state per clip.
            # v0.4.56: 22f = one full native H3 continuation period beyond the
            # minimal 5f head. The geometry helper compensates continuation clip
            # lengths so final visible duration stays identical to the old 5f path.
            multiclip_native_overlap = int(effective_transition_frames)
            mc_lengths, mc_starts, mc_output_frames = _v85_multiclip_geometry(
                multiclip_clips, multiclip_native_overlap
            )
            plan = _dc_replace(
                plan,
                mode='multiclip', duration_basis='multiclip:native_continuous_vae',
                workflow_mode='multiclip', segmentation_active=False,
                total_duration=float(mc_output_frames) / float(FPS), output_frames=int(mc_output_frames),
                segment_frames=int(mc_lengths[0]), segment_lengths=tuple(mc_lengths), segment_starts=tuple(mc_starts),
                overlap_frames=int(multiclip_native_overlap), step_frames=max(1, int(mc_lengths[0]) - int(multiclip_native_overlap)),
                passes=len(mc_lengths), generated_frames=int(mc_output_frames), trim_frames=0,
                segment_seeds=tuple(c['seed'] for c in multiclip_clips), timeline_policy='native_continuous_vae',
            )
            _lm_print(
                '[MiniMaxH3 LongMedia][MULTICLIP NATIVE CONTINUOUS VAE] '
                f'clips={len(mc_lengths)} lengths={list(mc_lengths)} starts={list(mc_starts)} '
                f'native_overlap={int(multiclip_native_overlap)}f latent_overlap={int(video_latent_t(multiclip_native_overlap))}t '
                f'final={int(mc_output_frames)}f; video_decode=single_continuous',
                flush=True,
            )
        if h3_mode == 'video_ref_edit' and reconstruction_cfg is None and video_1 is not None:
            _source_video_seconds = float(video_1.shape[0]) / max(float(video_fps), 1e-6)
            _target_seconds = float(plan.total_duration)
            effective_prompt = _build_video_ref_edit_timeline_prompt(
                effective_prompt,
                source_video_seconds=_source_video_seconds,
                target_seconds=_target_seconds,
                duration_source=str(effective_duration_source),
            )
            _delta = _target_seconds - _source_video_seconds
            _relation = 'continue' if _delta > (1.0 / float(FPS)) else ('trim' if _delta < -(1.0 / float(FPS)) else 'match')
            _lm_print(
                '[MiniMaxH3 LongMedia][VIDEO EDIT TIMELINE OWNER] '
                f'requested={duration_source}; effective={effective_duration_source}; resolved_basis={plan.duration_basis}; '
                f'video1={_source_video_seconds:.3f}s target={_target_seconds:.3f}s relation={_relation}',
                flush=True,
            )

        mode = plan.mode
        if control_mode != 'manual':
            _lm_print(
                f'[MiniMaxH3 LongMedia][MODE] workflow={workflow_mode} conditioning={conditioning_mode} '
                f'passes={int(plan.passes)} segment_duration={float(effective_segment_seconds):.3f}s '
                f'overlap={int(plan.overlap_frames)}f',
                flush=True,
            )
            if timeline_mode == 'segmented':
                _lm_print(
                    '[MiniMaxH3 LongMedia][FIXED TIMELINE] '
                    'fixed segmentation uses the shared MultiClip executor; only clip-boundary math differs',
                    flush=True,
                )
            if h3_mode == 'video_ref_edit':
                _lm_print(
                    '[MiniMaxH3 LongMedia][VIDEO EDIT] video_1 drives motion/camera/staging; '
                    'image_1..9 stay as <Picture N> replacement refs; '
                    f'paired_source_audio_sync={bool(video_ref_edit_audio_sync)}',
                    flush=True,
                )
            if workflow_mode == 'reconstruct':
                _lm_print(
                    '[MiniMaxH3 LongMedia][VIDEO RECONSTRUCTION] '
                    f'profile={reconstruction_profile} strength={float(reconstruction_strength):.3f}; '
                    f'source_fps={float(video_fps):.3f}; chunks={int(plan.passes)}; '
                    'source video is sliced per chunk and fed as a native reference stream; target latent stays fresh and VRAM is independent of total duration',
                    flush=True,
                )

        # PR#3 compatibility fix: when image refs are connected together with
        # audio but there is no source video, keep the NativeReferenceToVideo
        # route instead of audio_to_video. The audio_to_video path ignores
        # image refs and creates a 1x1 spatial video latent.
        if (
            mode == 'audio_to_video'
            and conditioning_mode == 'auto_refs'
            and len(images) > 0
            and len(videos) == 0
        ):
            mode = 't2v'

        target_av = None

        if NativeReferenceToVideo is None:
            NativeReferenceToVideo = _resolve_native_reference_to_video()

        hybrid_info = None
        if lip_sync_enabled:
            # Orthogonal policy: keep the selected workflow/conditioning family,
            # but make audio_1 an explicit speech/lip driver for every local pass.
            effective_prompt = _build_lipsync_prompt(
                effective_prompt, plan, image_1 is not None, audio_1 is not None
            )
            generation_mode = 'auto'
            _lm_print(
                '[MiniMaxH3 LongMedia][LIP SYNC] audio_mode=lip_sync; '
                'workflow preserved; Audio 1 remains native Ref2VA content reference + native per-clip H3 Audio Guide timing',
                flush=True,
            )

        # Every generation route must encode pass 0 from the same global-to-local
        # timeline policy used by continuation passes.  Before v0.3.29 pass 0 saw
        # future events (for example a 07 sec kiss inside a 5 sec segment), while
        # pass 1 saw shifted local events and therefore replayed the action.
        segment0_prompt = _v57_build_segment_prompt(effective_prompt, plan, 0)

        if conditioning_mode == 'multiclip_ref2va':
            # Extender parity: all image/video/audio references are shared native
            # Ref2VA payload on every clip. No startup keyframe consumes image_1.
            multiclip_ref_images = [safe_image_by_id.get(id(v), v) for v in
                                    [image_1, image_2, image_3, image_4, image_5,
                                     image_6, image_7, image_8, image_9] if v is not None]
            multiclip_ref_videos = [v for v in [video_1, video_2, video_3] if v is not None]
            multiclip_ref_audios = ([v for v in [audio_1, audio_2, audio_3] if v is not None]
                                    if use_audio_as_reference else [])
            setup_memory_events.append(
                _setup_memory_isolation('before_multiclip_ref2va_conditioning', unload_models=True)
            )
            positive, target_av, hybrid_info, hybrid_artifacts = _build_longmedia_hybrid_conditioning(
                clip=clip, vae=vae, audio_vae=audio_vae,
                prompt=segment0_prompt, width=width, height=height,
                length=plan.segment_lengths[0], resolution_mode=resolution_mode,
                first_frame=None, last_frame=None,
                ref_images=multiclip_ref_images,
                ref_videos=multiclip_ref_videos,
                ref_audios=multiclip_ref_audios,
                first_latent_override=None, last_latent_override=None,
            )
            hybrid_info.update({
                'multiclip_ref2va': True,
                'shared_picture_refs': len(multiclip_ref_images),
                'continuation_reference_policy': 'extender_shared_ref2va_every_clip',
            })
            plan = _dc_replace(
                plan, mode='multiclip', source_video=None, source_audio=None,
                reference_audio=(audio_1 if (use_audio_as_reference and audio_1 is not None) else None),
                final_audio_override=(_mix_audio_tracks(audios) if (preserve_audio_output and audios) else None),
                final_audio_track_count=(len(audios) if preserve_audio_output else 0),
                first_frame_override=None, first_frame_latent_injected=False,
                audio_vae=audio_vae, video_vae=vae,
            )
            _lm_print(
                '[MiniMaxH3 LongMedia][EXTENDER REF2VA PARITY] '
                f'all {len(multiclip_ref_images)} image refs remain native Picture refs on every clip; '
                'no first-frame anchor; fresh AV target per clip + native Motion Context for clip 2+',
                flush=True,
            )
            setup_memory_events.append(
                _setup_memory_isolation('after_multiclip_ref2va_conditioning_release', unload_models=True)
            )

        elif conditioning_mode == 'storyboard_bridge':
            if image_1 is None or image_2 is None or image_3 is None:
                raise ValueError(
                    'storyboard_bridge V64 requires image_1=panel A, image_2=shared panel B, image_3=panel C.'
                )
            if int(plan.passes) != 2:
                raise ValueError(
                    f'storyboard_bridge V64 requires exactly 2 passes; current plan has {int(plan.passes)}. '
                    'Use manual_duration=10 and segment_seconds=5 for the first test.'
                )
            nominal = max(5, int(math.floor(float(segment_seconds) * FPS)))
            first_len = int(align_frame_count(nominal))
            remaining = max(5, int(plan.output_frames) - first_len + 1)
            second_len = int(align_frame_count(remaining))
            plan = _dc_replace(
                plan, mode='storyboard_bridge', overlap_frames=0,
                segment_lengths=(first_len, second_len), segment_starts=(0, nominal),
                segment_frames=first_len, generated_frames=first_len + second_len - 1,
                trim_frames=max(0, first_len + second_len - 1 - int(plan.output_frames)), passes=2,
                source_video=None, source_audio=None, reference_audio=None,
                final_audio_override=None, final_audio_track_count=0,
            )
            try:
                from comfy_extras.nodes_minimax_h3 import _resize as _h3_resize
            except Exception as exc:
                raise RuntimeError(f'Current ComfyUI MiniMax H3 resize helper unavailable: {exc}')

            # V64 true storyboard: every panel is fitted ONCE and encoded ONCE.
            # Panel B is therefore bit-identical on both sides of the join.
            panel_a = _h3_resize(image_1[:1], int(width), int(height), 'center')
            panel_b = _h3_resize(image_2[:1], int(width), int(height), 'center')
            panel_c = _h3_resize(image_3[:1], int(width), int(height), 'center')
            panel_a_lat = vae.encode(panel_a)
            panel_b_lat = vae.encode(panel_b)
            panel_c_lat = vae.encode(panel_c)

            # image_1..3 are storyboard panels, never Ref2VA images in this mode.
            storyboard_refs = [v for v in [image_4, image_5, image_6, image_7, image_8, image_9] if v is not None]
            storyboard_videos = [v for v in [video_1, video_2, video_3] if v is not None]
            storyboard_audios = [v for v in [audio_1, audio_2, audio_3] if v is not None]
            setup_memory_events.append(_setup_memory_isolation('before_storyboard_conditioning', unload_models=True))

            pass0_prompt = segment0_prompt
            positive, target_av, sb0, _sb0_artifacts = _build_longmedia_hybrid_conditioning(
                clip=clip, vae=vae, audio_vae=audio_vae, prompt=pass0_prompt,
                width=width, height=height, length=first_len, resolution_mode=resolution_mode,
                first_frame=panel_a, last_frame=panel_b, ref_images=storyboard_refs,
                ref_videos=storyboard_videos, ref_audios=storyboard_audios,
                first_latent_override=panel_a_lat, last_latent_override=panel_b_lat,
            )
            pass1_prompt = _v57_build_segment_prompt(effective_prompt, plan, 1)
            positive_1, target_av_1, sb1, _sb1_artifacts = _build_longmedia_hybrid_conditioning(
                clip=clip, vae=vae, audio_vae=audio_vae, prompt=pass1_prompt,
                width=width, height=height, length=second_len, resolution_mode=resolution_mode,
                first_frame=panel_b, last_frame=panel_c, ref_images=storyboard_refs,
                ref_videos=storyboard_videos, ref_audios=storyboard_audios,
                first_latent_override=panel_b_lat, last_latent_override=panel_c_lat,
            )
            import comfy.sampler_helpers
            segment_positive_conditionings = (
                comfy.sampler_helpers.convert_cond(positive),
                comfy.sampler_helpers.convert_cond(positive_1),
            )
            plan = _dc_replace(
                plan, video_vae=vae, audio_vae=audio_vae,
                segment_positive_conditionings=segment_positive_conditionings,
                segment_prompt_summaries=(pass0_prompt, pass1_prompt),
                storyboard_segment_avs=(target_av, target_av_1),
                storyboard_bridge_frame=first_len,
            )
            hybrid_info = {
                'storyboard_bridge': True, 'panels': 3,
                'pass0': sb0, 'pass1': sb1, 'refs': len(storyboard_refs),
            }
            _lm_print(
                f'[MiniMaxH3 LongMedia][V64 TRUE 3-PANEL STORYBOARD] A->B then B->C; '
                f'panel B latent reused exactly on both sides; lengths={first_len},{second_len}; '
                f'overlap=0; image refs start at image_4; refs={len(storyboard_refs)}',
                flush=True,
            )
            setup_memory_events.append(_setup_memory_isolation('after_storyboard_conditioning_release', unload_models=True))

        elif conditioning_mode in ('hybrid_first_frame', 'hybrid_first_last'):
            if workflow_mode == 'segmented_continuation' and segmented_opening_frame is not None:
                hybrid_first = segmented_opening_frame
                if conditioning_mode == 'hybrid_first_last':
                    hybrid_last = (loop_last_override if loop_last_override is not None else image_2)
                else:
                    hybrid_last = None
                loop_latent_override = None
                if conditioning_mode == 'hybrid_first_last' and image_2 is None and loop_last_override is None:
                    raise ValueError(
                        'hybrid_first_last requires image_2 as the final-frame keyframe.'
                    )
                # With explicit opening_frame, every image_N remains a native Picture ref.
                hybrid_ref_images = [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9]
            else:
                if image_1 is None:
                    raise ValueError(
                        'Hybrid conditioning requires image_1 as the first-frame keyframe.'
                    )
                hybrid_first = image_1
                hybrid_last = (loop_last_override if loop_last_override is not None else image_2) if conditioning_mode == 'hybrid_first_last' else None
                loop_latent_override = None
                if conditioning_mode == 'hybrid_first_last' and image_2 is None and loop_last_override is None:
                    raise ValueError(
                        'hybrid_first_last requires image_2 as the final-frame keyframe.'
                    )
                if workflow_mode == 'loop':
                    # Hybrid parity: exactly emulate manually wiring the same IMAGE object
                    # into image_1 (first frame) and image_2 (last frame) in hybrid_auto.
                    # image_2 itself is reserved/ignored in loop mode; refs begin at image_3.
                    hybrid_ref_images = [image_3, image_4, image_5, image_6, image_7, image_8, image_9]
                else:
                    hybrid_ref_images = (
                        [image_3, image_4, image_5, image_6, image_7, image_8, image_9]
                        if conditioning_mode == 'hybrid_first_last'
                        else [image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9]
                    )
            hybrid_ref_images = [safe_image_by_id.get(id(v), v) for v in hybrid_ref_images if v is not None]
            hybrid_ref_videos = [v for v in [video_1, video_2, video_3] if v is not None]
            hybrid_ref_audios = ([v for v in [audio_1, audio_2, audio_3] if v is not None] if use_audio_as_reference else [])

            # Keyframe anchors are target-timeline guides and are not numbered as
            # native <Picture N> references.  Detect the unambiguous legacy/input-
            # socket convention (present in the supplied workflow) and translate it
            # before any pass is tokenized.  Native-numbered prompts stay untouched.
            anchor_roles = ['opening frame']
            if conditioning_mode == 'hybrid_first_last':
                anchor_roles.append('ending frame')
            effective_prompt, picture_tag_policy = normalize_hybrid_picture_tags(
                effective_prompt,
                anchor_roles=tuple(anchor_roles),
                reference_count=len(hybrid_ref_images),
            )
            segment0_prompt = _v57_build_segment_prompt(effective_prompt, plan, 0)
            setup_memory_events.append(
                _setup_memory_isolation('before_hybrid_conditioning', unload_models=True)
            )
            positive, target_av, hybrid_info, hybrid_artifacts = _build_longmedia_hybrid_conditioning(
                clip=clip, vae=vae, audio_vae=audio_vae,
                prompt=segment0_prompt, width=width, height=height,
                length=plan.segment_lengths[0], resolution_mode=resolution_mode,
                first_frame=hybrid_first, last_frame=hybrid_last,
                ref_images=hybrid_ref_images, ref_videos=hybrid_ref_videos,
                ref_audios=hybrid_ref_audios,
                first_latent_override=loop_latent_override,
                last_latent_override=loop_latent_override,
            )
            first_keyframe_latent = hybrid_artifacts.get('first_keyframe_latent')
            first_frame_latent_injected = False
            if (not lip_sync_enabled) and first_frame_mode == 'latent_inject' and first_keyframe_latent is not None:
                # A native H3 keyframe is a conditioning guide, not a frozen target
                # latent.  Reuse the already-encoded anchor latent for one leading
                # target step so the opening composition cannot unpack as a collage
                # of the connected references before converging several frames later.
                target_av = inject_leading_video_frame(
                    target_av,
                    {'samples': first_keyframe_latent},
                    float(first_frame_denoise),
                    NestedTensor,
                )
                first_frame_latent_injected = True
            hybrid_info.update({
                'picture_tag_policy': picture_tag_policy,
                'continuation_reference_policy': 'native_order_geometry_stable',
                'first_frame_policy': (
                    'target_latent_inject' if first_frame_latent_injected
                    else str(first_frame_mode)
                ),
            })
            _lm_print(
                '[MiniMaxH3 LongMedia][V327 STARTUP CONTINUITY] '
                'pass0 uses only the native frame-0 anchor; repeated 5/22/39 anchors disabled',
                flush=True,
            )
            setup_memory_events.append(
                _setup_memory_isolation('after_hybrid_conditioning_release', unload_models=True)
            )
            if workflow_mode == 'loop':
                _lm_print('[MiniMaxH3 LongMedia][LOOP] hybrid parity: image_1 is internally used as both first+last frame; image_2 ignored; refs start at image_3', flush=True)
            # In hybrid mode connected video/audio sockets are conditioning references,
            # not source streams to inject into later long-media segments.
            plan = _dc_replace(
                plan, mode=(plan.mode if timeline_mode in ('segmented', 'multiclip') else 'hybrid'), source_video=None, source_audio=None,
                reference_audio=(audio_1 if (use_audio_as_reference and audio_1 is not None) else None),
                final_audio_override=(_mix_audio_tracks(audios) if (preserve_audio_output and audios) else None),
                final_audio_track_count=(len(audios) if preserve_audio_output else 0),
                first_frame_override=(
                    hybrid_first if ((not lip_sync_enabled) and first_frame_mode in ('pixel_override', 'blend')) else None
                ),
                first_frame_mode=('disabled' if lip_sync_enabled else first_frame_mode),
                first_frame_denoise=(0.0 if lip_sync_enabled else float(first_frame_denoise)),
                first_frame_blend_frames=(0 if lip_sync_enabled else int(first_frame_blend_frames)),
                first_frame_latent_injected=(False if lip_sync_enabled else first_frame_latent_injected),
                audio_vae=audio_vae, video_vae=vae,
            )
        elif h3_mode == 'video_ref_edit' and reconstruction_cfg is None:
            # v0.5.35: true source-AV edit ownership.  The previous single-pass
            # implementation fell through to the generic video_to_video target
            # path.  That path could preserve coarse source motion, but it did not
            # present image_1..9 as Picture refs and, critically for preserve mode,
            # it did not present audio_1 as the soundtrack paired with <Video 1>.
            #
            # Upstream MiniMax H3 has an explicit paired reference contract:
            # ref_video_audio_N + ref_video_N become one `video_audio` block.  The
            # audio/video time grids inside that block share one reference span.
            # Keep that exact native contract here, then (for preserve/auto) also
            # freeze Audio1 into the target AV stream below.  The two mechanisms
            # have different jobs: paired Ref2VA transfers source performance;
            # locked target audio provides the absolute generation clock.
            edit_ref_images = [img for img in safe_images if img is not None]
            edit_ref_videos = [vid for vid in videos if vid is not None]

            # Only source-preservation modes assert that Audio 1 is the original
            # soundtrack belonging to Video 1. lip_sync intentionally keeps Audio 1
            # standalone: it may be an arbitrary replacement dub whose phonemes do
            # not match the source face. reference_only is likewise a free prompt
            # reference rather than an asserted source soundtrack relationship.
            pair_source_audio = bool(
                audio_1 is not None
                and audio_mode in ('auto', 'preserve', 'preserve_reference')
            )
            edit_ref_video_audios = [
                (audio_1 if pair_source_audio and i == 0 else None)
                for i in range(len(edit_ref_videos))
            ]
            # Prompt conditioning is independent from the final audio policy.
            # Audio 2/3 always remain standalone native <Audio N> references in
            # video_ref_edit. If Audio 1 is not paired to Video 1, it is also a
            # standalone reference. This enables prompts to address several tracks
            # (for example voice, percussion and bass) while duration_source and
            # audio_mode independently own timeline/output semantics.
            edit_ref_audios = [aud for aud in (audio_2, audio_3) if aud is not None]
            if (not pair_source_audio) and audio_1 is not None:
                edit_ref_audios.insert(0, audio_1)
            video_ref_edit_conditioning_audio_count = int(len(edit_ref_audios) + (1 if pair_source_audio else 0))

            setup_memory_events.append(
                _setup_memory_isolation('before_video_ref_edit_paired_conditioning', unload_models=True)
            )
            positive, target_av, hybrid_info, _edit_artifacts = _build_longmedia_hybrid_conditioning(
                clip=clip, vae=vae, audio_vae=audio_vae,
                prompt=segment0_prompt, width=width, height=height,
                length=plan.segment_lengths[0], resolution_mode=resolution_mode,
                first_frame=None, last_frame=None,
                ref_images=edit_ref_images, ref_videos=edit_ref_videos,
                ref_video_audios=edit_ref_video_audios, ref_audios=edit_ref_audios,
                first_latent_override=None, last_latent_override=None,
            )
            setup_memory_events.append(
                _setup_memory_isolation('after_video_ref_edit_paired_conditioning_release', unload_models=True)
            )
            plan = _dc_replace(
                plan,
                mode='video_ref_edit',
                source_video=None,
                source_audio=None,
                reference_audio=None,
                # In video_ref_edit Audio1 owns source/output audio semantics.
                # Audio2/Audio3 are independent prompt-addressable H3 references
                # and must never be mixed into the preserved source soundtrack.
                final_audio_override=(
                    audio_1
                    if audio_mode in ('auto', 'preserve', 'preserve_reference') and audio_1 is not None
                    else None
                ),
                final_audio_track_count=(
                    1 if audio_mode in ('auto', 'preserve', 'preserve_reference') and audio_1 is not None else 0
                ),
                audio_vae=audio_vae,
                video_vae=vae,
            )
            hybrid_info.update({
                'video_ref_edit_native_ref2va': True,
                'source_video_audio_paired': bool(pair_source_audio),
                'image_refs': len(edit_ref_images),
                'video_refs': len(edit_ref_videos),
                'standalone_audio_refs': len(edit_ref_audios),
                'conditioning_audio_refs_total': int(video_ref_edit_conditioning_audio_count),
                'audio1_role': (
                    'paired_source_soundtrack' if pair_source_audio
                    else ('authoritative_redub' if lip_sync_enabled else ('standalone_reference' if audio_1 is not None else 'disconnected'))
                ),
            })
            _lm_print(
                '[MiniMaxH3 LongMedia][VIDEO EDIT NATIVE AV REFS] '
                f'Pictures={len(edit_ref_images)}; Videos={len(edit_ref_videos)}; '
                f'Video1+Audio1_paired={bool(pair_source_audio)}; '
                f'standalone_audio_refs={len(edit_ref_audios)}; '
                f'Audio1_role={hybrid_info.get("audio1_role")}; target=fresh_ref2va',
                flush=True,
            )

        elif mode == 'automatic_lip_sync':
            ref_audio_waveform = audio_1['waveform'][:1]
            segment_samples = int(round(segment_seconds * audio_1['sample_rate']))
            if ref_audio_waveform.shape[-1] > segment_samples:
                ref_audio_waveform = ref_audio_waveform[..., :segment_samples]
            ref_audio = {'waveform': ref_audio_waveform, 'sample_rate': audio_1['sample_rate']}
            ref_images = {}
            ref_videos = {}
            ref_audios = {}
            for i, img in enumerate(safe_images):
                ref_images[f'ref_image_{i}'] = img
            for i, vid in enumerate(videos):
                ref_videos[f'ref_video_{i}'] = vid
            if ref_audio is not None:
                ref_audios['ref_audio_0'] = ref_audio
            setup_memory_events.append(_setup_memory_isolation('before_native_reference', unload_models=True))
            positive, target_av = _longmedia_native_reference_execute_safe(
                NativeReferenceToVideo, clip=clip, vae=vae, audio_vae=audio_vae,
                prompt=segment0_prompt, width=width, height=height,
                length=plan.segment_lengths[0], ref_image_size=('max' if images else resolution_mode),
                ref_images=ref_images, ref_videos=ref_videos, ref_audios=ref_audios,
            )
            setup_memory_events.append(_setup_memory_isolation('after_native_reference_release', unload_models=True))
            if first_frame_mode == 'latent_inject':
                # Write image_1 into the video latent's leading frame *before*
                # sampling, with a partial-denoise mask, so the sampler can
                # soften the transition instead of a hard post-decode pixel
                # splice. Falls back to a strict pixel splice at denoise=0.
                # Only the minimal 5-frame H3 unit is encoded here (not the
                # full segment) — inject_leading_video_frame only needs its
                # first latent frame, and encoding the whole segment length
                # just to discard the rest would waste VRAM for nothing.
                min_frames = align_frame_count(5)
                held = _fit_frames(image_1, min_frames, 'crop_or_pad_last')
                held = _resize_frames(held, width, height, 'center_crop')
                frame_latent = {'samples': vae.encode(held)}
                setup_memory_events.append(_setup_memory_isolation('after_first_frame_vae_release', unload_models=True))
                target_av = inject_leading_video_frame(
                    target_av, frame_latent, float(first_frame_denoise), NestedTensor,
                )
            plan = _dc_replace(
                plan, mode='automatic_lip_sync',
                source_audio=None,
                reference_audio=audio_1,
                final_audio_override=(audio_1 if audio_mode in ('auto', 'preserve', 'preserve_reference') else None),
                final_audio_track_count=max(1, len(audios)),
                first_frame_override=image_1, audio_vae=audio_vae, video_vae=vae,
                first_frame_mode=first_frame_mode,
                first_frame_denoise=float(first_frame_denoise),
                first_frame_blend_frames=int(first_frame_blend_frames),
                first_frame_latent_injected=(first_frame_mode == 'latent_inject'),
            )
            if len(audios) > 1 and audio_mode in ('auto', 'preserve', 'preserve_reference'):
                mixed = _mix_audio_tracks(audios)
                plan = _dc_replace(
                    plan, final_audio_override=mixed, final_audio_track_count=len(audios),
                )
        elif mode == 't2v':
            ref_images = {}
            ref_videos = {}
            ref_audios = {}
            for i, img in enumerate(safe_images):
                ref_images[f'ref_image_{i}'] = img
            for i, vid in enumerate(videos):
                ref_videos[f'ref_video_{i}'] = vid
            if use_audio_as_reference:
                for i, aud in enumerate(audios):
                    ref_audios[f'ref_audio_{i}'] = aud
            setup_memory_events.append(_setup_memory_isolation('before_native_reference', unload_models=True))
            positive, target_av = _longmedia_native_reference_execute_safe(
                NativeReferenceToVideo, clip=clip, vae=vae, audio_vae=audio_vae,
                width=width, height=height, length=plan.segment_lengths[0],
                prompt=segment0_prompt, ref_image_size=('max' if images else resolution_mode),
                ref_images=ref_images, ref_videos=ref_videos, ref_audios=ref_audios,
            )
            setup_memory_events.append(_setup_memory_isolation('after_native_reference_release', unload_models=True))
            if preserve_audio_output and audios:
                plan = _dc_replace(plan, final_audio_override=_mix_audio_tracks(audios), final_audio_track_count=len(audios))
        elif mode == 'audio_to_video':
            length_frames = plan.segment_lengths[0]
            start_frame = plan.segment_starts[0]
            available, _ = _slice_source_audio_for_segment(audio_1, start_frame, length_frames)
            waveform_for_encode = available.movedim(1, -1)
            frozen_audio_latent = audio_vae.encode(waveform_for_encode)
            video_t = video_latent_t(length_frames)
            audio_t = audio_latent_t(length_frames)
            audio_lat = frozen_audio_latent
            if audio_lat.shape[-1] != audio_t:
                audio_lat = torch.zeros(
                    (1, 32, 2, audio_t),
                    dtype=frozen_audio_latent.dtype,
                    device=frozen_audio_latent.device,
                )
                copy_len = min(audio_t, frozen_audio_latent.shape[-1])
                audio_lat[..., :copy_len] = frozen_audio_latent[..., :copy_len]
            video_lat = torch.zeros((1, 24, video_t, 1, 1), dtype=frozen_audio_latent.dtype)
            av_samples = NestedTensor((video_lat, audio_lat))
            video_mask = torch.ones((1, 1, video_t, 1, 1), dtype=torch.float32)
            audio_denoise = (
                1.0 if audio_mode in ('generate', 'reference_only') else 0.0
            )
            audio_mask = torch.full(
                (1, 1, 1, audio_lat.shape[-1]), audio_denoise, dtype=torch.float32
            )
            mask_samples = NestedTensor((video_mask, audio_mask))
            target_av = {'samples': av_samples, 'noise_mask': mask_samples}
            setup_memory_events.append(_setup_memory_isolation('before_clip_encode', unload_models=True))
            positive = _encode_prompt(clip, segment0_prompt)
            setup_memory_events.append(_setup_memory_isolation('after_clip_release', unload_models=True))
            mixed_audio = _mix_audio_tracks(audios) if audios else None
            plan = _dc_replace(
                plan, mode='audio_to_video',
                source_audio=audio_1 if audio_mode in ('generate', 'reference_only', 'preserve_reference') else None,
                final_audio_override=(mixed_audio if audio_mode in ('auto', 'preserve', 'preserve_reference') else None),
                final_audio_track_count=len(audios),
                audio_vae=audio_vae, video_vae=vae,
            )
        elif mode in ('video_to_video', 'video_audio_to_video'):
            length_frames = plan.segment_lengths[0]
            start_frame = plan.segment_starts[0]
            source_frames = slice_video_segment(
                video_1, start_frame, length_frames, video_fps,
            )
            if workflow_mode == 'reconstruct':
                # V5: native Ref2VA video-edit semantics. The source is a true
                # <Video 1> reference block; the target AV latent is fresh.
                # This is the H3 task family designed for source-video editing,
                # unlike FL2VA which only owns first/last-frame anchors.
                fitted_source_frames = _reconstruction_fit_source_frames(
                    source_frames, int(width), int(height), reconstruction_resize_mode,
                )
                reconstruct_ref_images = [safe_image_by_id.get(id(v), v) for v in
                                          [image_1, image_2, image_3, image_4, image_5,
                                           image_6, image_7, image_8, image_9] if v is not None]
                setup_memory_events.append(_setup_memory_isolation('before_reconstruction_ref2va_conditioning', unload_models=True))
                positive, target_av, hybrid_info, hybrid_artifacts = _build_longmedia_hybrid_conditioning(
                    clip=clip, vae=vae, audio_vae=audio_vae,
                    prompt=segment0_prompt, width=width, height=height,
                    length=plan.segment_lengths[0], resolution_mode=resolution_mode,
                    first_frame=None, last_frame=None,
                    ref_images=reconstruct_ref_images, ref_videos=[fitted_source_frames],
                    ref_audios=[], first_latent_override=None, last_latent_override=None,
                )
                positive = _reconstruction_set_ref_strength(
                    positive, reconstruction_strength, reconstruction_profile,
                )
                setup_memory_events.append(_setup_memory_isolation('after_reconstruction_ref2va_conditioning_release', unload_models=True))
                video_lat = None
            else:
                import comfy.utils
                samples = source_frames.movedim(-1, 1)
                samples = comfy.utils.common_upscale(samples, width, height, 'lanczos', 'disabled')
                source_frames = samples.movedim(1, -1)
                video_lat = vae.encode(source_frames)
            audio_lat = None
            if audio_1 is not None:
                available, _ = _slice_source_audio_for_segment(audio_1, start_frame, length_frames)
                waveform_for_encode = available.movedim(1, -1)
                audio_lat = audio_vae.encode(waveform_for_encode)
            if workflow_mode != 'reconstruct':
                if audio_lat is None:
                    a_t = audio_latent_t(length_frames)
                    audio_lat = torch.zeros((1, 32, 2, a_t), dtype=video_lat.dtype)
                av_samples = NestedTensor((video_lat, audio_lat))
                video_mask = torch.full(
                    (1, 1, video_lat.shape[2], 1, 1),
                    float(
                        _reconstruction_video_mask_value(reconstruction_strength, reconstruction_profile)
                        if workflow_mode == 'reconstruct' else 1.0
                    ),
                    dtype=torch.float32,
                )
                audio_denoise = (
                    0.0 if workflow_mode == 'reconstruct' and audio_1 is not None
                    else (1.0 if audio_mode in ('generate', 'reference_only') else 0.0)
                )
                audio_mask = torch.full(
                    (1, 1, 1, audio_lat.shape[-1]), audio_denoise, dtype=torch.float32
                )
                mask_samples = NestedTensor((video_mask, audio_mask))
                target_av = {'samples': av_samples, 'noise_mask': mask_samples}
                setup_memory_events.append(_setup_memory_isolation('before_clip_encode', unload_models=True))
                positive = _encode_prompt(clip, segment0_prompt)
                setup_memory_events.append(_setup_memory_isolation('after_clip_release', unload_models=True))
            elif target_av is not None:
                target_video, target_audio = unpack_av_samples(target_av)
                if audio_lat is not None:
                    target_audio = _fit_stream(audio_lat, target_audio, 'audio', 'crop_pad', 'start')
                    target_av['samples'] = NestedTensor((target_video, target_audio))
                video_mask = torch.ones((1, 1, target_video.shape[2], 1, 1), dtype=torch.float32)
                audio_mask = torch.full(
                    (1, 1, 1, target_audio.shape[-1]), 0.0 if audio_lat is not None else 1.0, dtype=torch.float32
                )
                target_av['noise_mask'] = NestedTensor((video_mask, audio_mask))
            else:
                raise RuntimeError('reconstruct setup expected target_av from hybrid conditioning but received None.')
            mixed_audio = _mix_audio_tracks(audios) if audios else None
            plan = _dc_replace(
                plan,
                source_audio=(
                    audio_1 if (workflow_mode == 'reconstruct' and audio_1 is not None)
                    else (audio_1 if (audio_1 is not None and use_audio_as_reference) else None)
                ),
                source_video=video_1 if video_1 is not None else None,
                mode=('video_to_video' if workflow_mode == 'reconstruct' else plan.mode),
                final_audio_override=(mixed_audio if audio_mode in ('auto', 'preserve', 'preserve_reference') else None),
                final_audio_track_count=len(audios),
                audio_vae=audio_vae,
                video_vae=vae,
            )
        else:
            ref_images = {}
            ref_videos = {}
            ref_audios = {}
            for i, img in enumerate(safe_images):
                ref_images[f'ref_image_{i}'] = img
            for i, vid in enumerate(videos):
                ref_videos[f'ref_video_{i}'] = vid
            if use_audio_as_reference:
                for i, aud in enumerate(audios):
                    ref_audios[f'ref_audio_{i}'] = aud
            setup_memory_events.append(_setup_memory_isolation('before_native_reference', unload_models=True))
            positive, target_av = _longmedia_native_reference_execute_safe(
                NativeReferenceToVideo, clip=clip, vae=vae, audio_vae=audio_vae,
                width=width, height=height, length=plan.segment_lengths[0],
                prompt=segment0_prompt, ref_image_size=('max' if images else resolution_mode),
                ref_images=ref_images, ref_videos=ref_videos, ref_audios=ref_audios,
            )
            setup_memory_events.append(_setup_memory_isolation('after_native_reference_release', unload_models=True))
            if preserve_audio_output and audios:
                plan = _dc_replace(plan, final_audio_override=_mix_audio_tracks(audios), final_audio_track_count=len(audios))

        # Normalize passthrough semantics across *all* workflow branches. Historically
        # auto preserved an attached source soundtrack in lip-sync / A2V / V2V paths,
        # but native Ref2VA/T2V branches forgot to populate final_audio_override. That
        # allowed Turbo-distilled model audio with incompatible latent geometry to reach
        # AudioVAE.decode(). If an input soundtrack exists, auto/preserve/preserve_reference
        # always retain the untouched waveform for final output. generate/reference_only
        # are the only modes that intentionally decode model-generated audio.
        passthrough_audio_mode = audio_mode in ('auto', 'preserve', 'preserve_reference')
        # video_ref_edit has one output/source soundtrack authority: Audio1.
        # Additional Audio2/Audio3 inputs stay conditioning references only.
        # Other H3 modes preserve the existing multi-track passthrough/mix contract.
        passthrough_audio = (
            audio_1 if h3_mode == 'video_ref_edit'
            else (_mix_audio_tracks(audios) if audios else None)
        )
        passthrough_track_count = (
            1 if h3_mode == 'video_ref_edit' and audio_1 is not None
            else (len(audios) if passthrough_audio is not None else 0)
        )
        if passthrough_audio_mode and passthrough_audio is not None and getattr(plan, 'final_audio_override', None) is None:
            plan = _dc_replace(
                plan,
                final_audio_override=passthrough_audio,
                final_audio_track_count=passthrough_track_count,
            )
        if video_ref_edit_audio_sync:
            # video_ref_edit + paired source soundtrack: preserving the waveform
            # at mux time is not enough. Identity replacement can otherwise rebuild
            # the mouth independently of the source performance. Make Audio1 the
            # authoritative target-audio clock exactly like the proven locked-target
            # lip-sync path, while keeping audio_mode's final-output semantics.
            plan = _dc_replace(
                plan,
                source_audio=audio_1,
                lip_sync_target_audio_locked=True,
            )
            target_av = _v113_lock_source_audio_in_target(
                target_av, audio_vae, audio_1,
                int(plan.segment_starts[0]), int(plan.segment_lengths[0]),
            )
            _lm_print(
                '[MiniMaxH3 LongMedia][VIDEO EDIT SOURCE AV SYNC] active; '
                'Audio1=frozen target audio clock; replacement subject is generated against the exact source performance timeline; '
                'final source waveform remains untouched',
                flush=True,
            )

        if lip_sync_enabled:
            # v0.3.104: preserve current UI/output semantics.  Audio 1 stays a
            # native Ref2VA content reference and is additionally anchored to the
            # local H3 timeline via minimax_keyframes.  Final mux restores the
            # pristine input track; sampled H3 audio is not the timing authority.
            plan = _dc_replace(
                plan,
                source_audio=audio_1,
                final_audio_override=audio_1,
                final_audio_track_count=1,
                lip_sync_native_audio_guide=True,
                lip_sync_target_audio_locked=True,
            )
            _lm_print(
                '[MiniMaxH3 LongMedia][AUTHORITATIVE LOCAL-0 LIP SYNC] '
                'Audio1=full native Ref2VA reference + native local-0 timing guide on every clip; continuation keeps VIDEO Motion Context while preserving the source-audio guide through hidden overlap; original Audio1 restored at output',
                flush=True,
            )
            # v0.3.113: Ref2VA refs/guide remain useful semantic conditioning, but
            # exact timing authority now comes from the frozen target audio stream.
            target_av = _v113_lock_source_audio_in_target(
                target_av, audio_vae, audio_1,
                int(plan.segment_starts[0]), int(plan.segment_lengths[0]),
            )

        # Persist the requested audio output policy in the plan. Decode must not infer
        # preserve semantics from the shape/content of the sampled audio stream: Turbo
        # LoRAs may leave a stream that is invalid for the stock Audio VAE decoder.
        plan = _dc_replace(
            plan,
            audio_output_mode=audio_mode,
            suppress_visible_opening_anchor=False,
            regression_safe_segmented_conditioning=bool(segmentation_active),
            decouple_original_image_refs_after_pass0=False,
        )

        # V57: build every per-pass TEXT conditioning now, while TE is intentionally available.
        # The plan receives only ready CONDITIONING tensors; never CLIP/TE/model-patcher objects.
        plan = _dc_replace(plan, video_vae=vae, audio_vae=audio_vae)
        # v0.3.108: pass 0 is sampled from the Setup output CONDITIONING itself,
        # so attach the authoritative native source-audio clock directly here.
        if lip_sync_enabled:
            positive = _v104_attach_native_lipsync_guide(
                positive, audio_vae, audio_1, plan, 0,
            )
        # v0.4.41: the PackedLayout row count must be derived from the same
        # audio latent tensors Comfy will pack.  This is especially important
        # for lip_sync + segmented_continuation where the audio window changes
        # for each pass while the native Ref2VA metadata originated globally.
        positive = _v041_normalize_minimax_audio_ref_geometry(positive)
        v329_native_refs = None
        if (
            (conditioning_mode in ('hybrid_first_frame', 'hybrid_first_last', 'multiclip_ref2va')
             or workflow_mode == 'reconstruct')
            and hybrid_artifacts is not None
            and (hybrid_artifacts.get('ref_items') is not None)
        ):
            # Keep tokenizer presentation AND latent block geometry identical for
            # every pass.  Combining distinct people into one sheet changed the
            # reference count and target packed-layout origin at the visible join.
            v329_native_refs = (
                hybrid_artifacts['ref_items'],
                hybrid_artifacts['ref_blocks'],
            )

        if conditioning_mode != 'storyboard_bridge':
            if getattr(plan, 'mode', None) == 'multiclip':
                mc_prompts = tuple(c['prompt'] for c in multiclip_clips)
                segment_positive_conditionings, segment_prompt_summaries = _v85_preencode_multiclip_conditionings(
                    clip, positive, plan, mc_prompts, v329_native_refs=v329_native_refs,
                    lip_sync_audio=(audio_1 if lip_sync_enabled else None), audio_vae=audio_vae,
                )
            else:
                segment_positive_conditionings, segment_prompt_summaries = _v57_preencode_segment_conditionings(
                    clip, effective_prompt, positive, plan, v329_native_refs=v329_native_refs,
                    lip_sync_audio=(audio_1 if lip_sync_enabled else None), audio_vae=audio_vae,
                )
            plan = _dc_replace(
                plan,
                segment_positive_conditionings=segment_positive_conditionings,
                segment_prompt_summaries=segment_prompt_summaries,
            )
        setup_memory_events.append(_setup_memory_isolation('setup_exit_release', unload_models=True))

        report = json.dumps({
            'mode': plan.mode,
            'passes': plan.passes,
            'segmentation_active': bool(getattr(plan, 'segmentation_active', False)),
            'manual_duration_seconds': float(plan.total_duration),
            'duration_source_requested': str(duration_source),
            'duration_source_effective': str(effective_duration_source),
            'duration_basis_resolved': str(getattr(plan, 'duration_basis', 'unknown')),
            'segment_seconds_requested': float(segment_seconds),
            'control_mode': control_mode,
            'h3_mode': h3_mode,
            'timeline_mode': timeline_mode,
            'transition_frames_requested': int(requested_transition_frames),
            'transition_frames': int(effective_transition_frames),
            'multiclip_enabled': bool(timeline_mode == 'multiclip'),
            'clip_engine_enabled': bool(getattr(plan, 'mode', None) == 'multiclip'),
            'timeline_policy': getattr(plan, 'timeline_policy', 'legacy'),
            'selected_workflow_mode': workflow_mode,
            'legacy_workflow_mode_input': legacy_workflow_mode,
            'multiclip_plan_source': ('external_planner' if (timeline_mode == 'multiclip' and external_clip_plan is not None) else ('setup_editor' if timeline_mode == 'multiclip' else ('fixed_segment_math' if timeline_mode == 'segmented' else None))),
            'multiclip_clip_durations': ([float(c['duration']) for c in multiclip_clips] if multiclip_clips else None),
            'multiclip_segment_seeds': ([c['seed'] for c in multiclip_clips] if multiclip_clips else None),
            'segment_seconds_semantics': ('new_output_timeline_plus_extra_overlap_context' if segmentation_active else 'ignored_outside_manual_and_segmented_continuation'),
            'segment_lengths_frames': list(plan.segment_lengths),
            'segment_starts_frames': list(plan.segment_starts),
            'overlap_frames': plan.overlap_frames,
            'loop_closure_enabled': bool(getattr(plan, 'loop_closure_enabled', False)),
            'loop_closure_frames': int(getattr(plan, 'loop_closure_frames', 0) or 0),
            'loop_closure_strength': float(getattr(plan, 'loop_closure_strength', 0.65)),
            'segment_conditioning_policy': 'preencoded_in_setup_no_clip_in_plan',
            'segment_conditionings_preencoded': int(len(getattr(plan, 'segment_positive_conditionings', ()) or ())),
            'decouple_original_image_refs_after_pass0': bool(getattr(plan, 'decouple_original_image_refs_after_pass0', False)),
            'h3_target_geometry': h3_target_geometry,
            'h3_reference_geometry': h3_ref_geometry,
            'h3_reference_pixel_budget': int(_H3_SAFE_REF_PIXELS),
            'h3_reference_resolution_policy': 'independent_safe_0.60MP_max_no_upscale',
            'output_frames': int(plan.output_frames),
            'trim_frames': int(plan.trim_frames),
            'final_audio_tracks': plan.final_audio_track_count,
            'audio_mode': audio_mode,
            'audio_reference_enabled': bool(
                video_ref_edit_conditioning_audio_count > 0
                if h3_mode == 'video_ref_edit' and reconstruction_cfg is None
                else use_audio_as_reference
            ),
            'conditioning_audio_reference_count': int(
                video_ref_edit_conditioning_audio_count
                if h3_mode == 'video_ref_edit' and reconstruction_cfg is None
                else len(audios) if use_audio_as_reference else 0
            ),
            'lip_sync_enabled': bool(lip_sync_enabled),
            'video_ref_edit_audio_sync': bool(video_ref_edit_audio_sync),
            'video_ref_edit_paired_source_av': bool(
                isinstance(hybrid_info, dict) and hybrid_info.get('source_video_audio_paired', False)
            ),
            'source_audio_target_locked': bool(getattr(plan, 'lip_sync_target_audio_locked', False)),
            'audio_output_bypass': bool(getattr(plan, 'final_audio_override', None) is not None),
            'workflow_mode': workflow_mode,
            'reconstruction_active': bool(workflow_mode == 'reconstruct'),
            'reconstruction_profile': (reconstruction_profile if workflow_mode == 'reconstruct' else None),
            'reconstruction_strength': (float(reconstruction_strength) if workflow_mode == 'reconstruct' else None),
            'conditioning_mode': conditioning_mode,
            'startup_anchor_frames': [],
            'first_frame_mode': first_frame_mode,
            'first_frame_latent_injected': bool(
                getattr(plan, 'first_frame_latent_injected', False)
            ),
            'continuation_reference_policy': (
                'native_order_geometry_stable' if v329_native_refs is not None
                else 'native_metadata_passthrough'
            ),
            'hybrid': hybrid_info,
            'setup_memory_isolation': setup_memory_events,
        }, indent=2)
        return (positive, target_av, plan, plan.total_duration, plan.passes, report)


class MiniMaxH3LatentLabLongMediaNextSegment:
    DESCRIPTION = (
        'Prepare the next segment AV latent by injecting the frozen overlap '
        'from the previous result and source media after the overlap.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'long_media_plan': ('LONG_MEDIA_PLAN',),
                'previous_av': ('LATENT',),
                'segment_index': (
                    'INT',
                    {'default': 1, 'min': 1, 'max': 100, 'step': 1},
                ),
                'video_context_denoise': (
                    'FLOAT',
                    {
                        'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                        'tooltip': '0 preserves the inherited overlap exactly; 1 fully denoises it.',
                    },
                ),
                'audio_context_denoise': (
                    'FLOAT',
                    {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
            }
        }

    RETURN_TYPES = ('LATENT', 'STRING')
    RETURN_NAMES = ('continuation_av', 'report')
    FUNCTION = 'prepare'
    CATEGORY = CATEGORY_LONGMEDIA

    def prepare(self, long_media_plan, previous_av, segment_index,
                video_context_denoise=0.0, audio_context_denoise=0.0):
        plan = long_media_plan
        seg_idx = int(segment_index)
        timeline = _segment_timeline_contract(plan, seg_idx)
        start_frame = int(timeline['context_start'])
        visible_start_frame = int(timeline['visible_start'])
        length_frames = int(timeline['length_frames'])
        overlap = int(timeline['local_visible_offset'])

        prev_video, prev_audio = unpack_av_samples(previous_av)
        target_video_t = video_latent_t(length_frames)
        target_audio_t = audio_latent_t(length_frames)
        overlap_video_t = video_latent_t(overlap) if overlap else 0
        overlap_audio_t = round(overlap / FPS * AUDIO_LATENT_FPS) if overlap else 0
        # MultiClip continuation must inherit the *actual generated latent* at the
        # boundary.  Native motion keyframes remain useful as an extra guide, but
        # they must not replace latent continuity: doing so created a fresh/full-
        # denoise head on clip 2+ and let motion energy ramp up after the guide span.
        multiclip_latent_overlap = bool(
            getattr(plan, 'mode', None) == 'multiclip' and seg_idx > 0
        )
        native_motion_head = bool(
            seg_idx == 1
            and int(getattr(plan, 'passes', 0) or 0) == 2
            and getattr(plan, 'mode', None) == 'segmented_continuation'
        ) and _v83_native_guide_api_supported()

        video = torch.zeros(
            (prev_video.shape[0], prev_video.shape[1], target_video_t,
             prev_video.shape[3], prev_video.shape[4]),
            dtype=prev_video.dtype, device=prev_video.device,
        )
        audio = torch.zeros(
            (prev_audio.shape[0], prev_audio.shape[1], prev_audio.shape[2], target_audio_t),
            dtype=prev_audio.dtype, device=prev_audio.device,
        )
        if overlap_video_t and not native_motion_head:
            video[:, :, :overlap_video_t] = prev_video[:, :, -overlap_video_t:]
        if overlap_audio_t and not native_motion_head:
            audio[..., :overlap_audio_t] = prev_audio[..., -overlap_audio_t:]

        if multiclip_latent_overlap:
            video_overlap_policy = 'multiclip_exact_generated_latent_overlap'
        else:
            video_overlap_policy = ('native_motion_context_fresh_head' if native_motion_head else 'zero_fill')
        latent_value_transform = 'none'

        if plan.source_video is not None and plan.video_vae is not None and not bool(getattr(plan, 'reconstruction_active', False)):
            source_frames = slice_video_segment(
                plan.source_video, start_frame, length_frames, plan.video_fps,
            )
            reconstruction_active = bool(getattr(plan, 'reconstruction_active', False))
            reconstruction_profile = str(getattr(plan, 'reconstruction_profile', 'balanced'))
            reconstruction_strength = float(getattr(plan, 'reconstruction_strength', 1.0))
            if reconstruction_active:
                source_frames = _reconstruction_preprocess_frames(source_frames, reconstruction_profile)
            target_av_for_encode = {'samples': NestedTensor((video, audio))}
            reconstruction_resize_mode = (
                str(getattr(plan, 'reconstruction_resize_mode', 'center_crop'))
                if reconstruction_active else 'none'
            )
            encoded_result = MiniMaxH3LatentLabVideoEncode().encode(
                plan.video_vae, source_frames, 'strict', reconstruction_resize_mode, target_av_for_encode,
            )
            source_video_latent = encoded_result[0]['samples']
            if reconstruction_active:
                source_video_latent = _reconstruction_apply_source_authority(
                    source_video_latent, reconstruction_strength, reconstruction_profile,
                )
            if overlap_video_t:
                video[:, :, overlap_video_t:] = source_video_latent[:, :, overlap_video_t:].to(video)
            else:
                video = source_video_latent.to(video)
            video_overlap_policy = ('lowpass_source_authority' if reconstruction_active else 'exact_frozen')

        if plan.source_audio is not None and plan.audio_vae is not None:
            available, _ = _slice_source_audio_for_segment(
                plan.source_audio, start_frame, length_frames
            )
            waveform_for_encode = available.movedim(1, -1)
            source_audio_latent = plan.audio_vae.encode(waveform_for_encode)
            if bool(getattr(plan, 'lip_sync_target_audio_locked', False)):
                # Source speech is the authoritative local clock.  Use the exact
                # global slice across the whole clip, including hidden overlap; do
                # not inherit sampled/generated audio from the previous clip.
                audio = _fit_stream(source_audio_latent, audio, 'audio', 'crop_pad', 'start')
            elif overlap_audio_t:
                audio[..., overlap_audio_t:] = source_audio_latent[
                    ..., :audio.shape[-1] - overlap_audio_t
                ].to(audio)
            else:
                audio = source_audio_latent.to(audio)

        _reconstruction_strength = (
            max(0.0, min(1.0, float(getattr(plan, 'reconstruction_strength', 1.0))))
            if bool(getattr(plan, 'reconstruction_active', False)) else 1.0
        )
        reconstruction_profile = str(getattr(plan, 'reconstruction_profile', 'balanced'))
        video_mask = torch.full(
            (1, 1, target_video_t, 1, 1),
            _reconstruction_video_mask_value(_reconstruction_strength, reconstruction_profile),
            dtype=torch.float32,
        )
        audio_mask = torch.ones((1, 1, 1, target_audio_t), dtype=torch.float32)
        if bool(getattr(plan, 'lip_sync_target_audio_locked', False)) or bool(getattr(plan, 'reconstruction_audio_locked', False)):
            audio_mask.zero_()
        if overlap_video_t and not native_motion_head:
            # MultiClip uses an exact frozen generated overlap. Other continuation
            # workflows retain the user-selected partial-denoise policy.
            video_mask[:, :, :overlap_video_t] = (
                0.0 if multiclip_latent_overlap else float(video_context_denoise)
            )
        if overlap_audio_t and not native_motion_head and not bool(getattr(plan, 'lip_sync_target_audio_locked', False)) and not bool(getattr(plan, 'reconstruction_audio_locked', False)):
            audio_mask[..., :overlap_audio_t] = (
                0.0 if multiclip_latent_overlap else float(audio_context_denoise)
            )
        if multiclip_latent_overlap:
            _lm_print(
                '[MiniMaxH3 LongMedia][MULTICLIP LATENT MOTION INERTIA] '
                f'segment={seg_idx} overlap={overlap}f target head inherits exact generated latent; '
                'video/audio overlap denoise=0; native motion context remains auxiliary',
                flush=True,
            )
        elif native_motion_head:
            _lm_print(
                '[MiniMaxH3 LongMedia][FRESH CONTINUATION HEAD] '
                f'segment=1 overlap={overlap}f target head is zero-init/full-denoise; '
                'continuity is owned by native minimax_keyframes, not latent copying',
                flush=True,
            )

        av_samples = NestedTensor((video, audio))
        mask_samples = NestedTensor((video_mask, audio_mask))
        output = {k: v for k, v in previous_av.items() if k not in ('noise_mask', 'samples')}
        output['samples'] = av_samples
        output['noise_mask'] = mask_samples

        _lm_print(
            '[MiniMaxH3 LongMedia][V318 TIMELINE] '
            f'segment={seg_idx} context_start={start_frame}f visible_start={visible_start_frame}f '
            f'local_visible_offset={overlap}f visible_frames={int(timeline["visible_frames"])}f '
            f'source_video_window_start={start_frame if plan.source_video is not None else "none"} '
            f'source_audio_window_start={start_frame if plan.source_audio is not None else "none"} '
            f'locked_target_audio={bool(getattr(plan, "lip_sync_target_audio_locked", False))}',
            flush=True,
        )
        report = json.dumps({
            'segment_index': seg_idx,
            'context_start_frame': start_frame,
            'visible_start_frame': visible_start_frame,
            'local_visible_offset_frames': overlap,
            'visible_frames': int(timeline['visible_frames']),
            'visible_end_frame': int(timeline['visible_end']),
            'source_video_window_start_frame': (start_frame if plan.source_video is not None else None),
            'source_audio_window_start_frame': (start_frame if plan.source_audio is not None else None),
            'length_frames': length_frames,
            'overlap_frames': overlap,
            'video_overlap_policy': video_overlap_policy,
            'overlap_mask_policy': ('multiclip_exact_frozen' if multiclip_latent_overlap else ('native_motion_context_full_denoise' if native_motion_head else 'constant_overlap_denoise')),
            'native_motion_context_head': bool(native_motion_head),
            'multiclip_latent_overlap': bool(multiclip_latent_overlap),
            'latent_value_transform': latent_value_transform,
            'video_context_denoise': float(video_context_denoise),
            'audio_context_denoise': float(audio_context_denoise),
        }, indent=2)
        return (output, report)


class MiniMaxH3LatentLabRefineSigmas:
    """Describe the true base/refiner split executed inside one unified model lifecycle."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'sigmas': ('SIGMAS',),
                'refine_steps': ('INT', {'default': 2, 'min': 1, 'max': 1000, 'step': 1}),
            }
        }

    RETURN_TYPES = ('SIGMAS', 'SIGMAS', 'INT', 'INT', 'INT', 'STRING')
    RETURN_NAMES = ('main_sigmas', 'refine_sigmas', 'total_steps', 'switch_step', 'refine_steps_effective', 'report')
    FUNCTION = 'build'
    CATEGORY = CATEGORY_LONGMEDIA

    def build(self, sigmas, refine_steps=2):
        main, refine, total_steps, switch_step, effective, requested = split_refine_sigmas(
            sigmas, refine_steps
        )
        report = json.dumps({
            'total_steps': total_steps,
            'main_steps': int(max(0, switch_step)),
            'refine_steps_requested': requested,
            'refine_steps_effective': effective,
            'switch_step': switch_step,
            'main_sigma_points': int(main.numel()),
            'refine_sigma_points': int(refine.numel()),
            'main_sigma_start': float(main[0].detach().float().cpu()),
            'main_sigma_end': float(main[-1].detach().float().cpu()),
            'refine_sigma_start': float(refine[0].detach().float().cpu()),
            'refine_sigma_end': float(refine[-1].detach().float().cpu()),
            'scheduler_source': 'connected_sigmas_true_advanced_split_unified_runtime',
            'main_return_with_leftover_noise': True,
            'refine_add_noise': False,
            'intervals_total': total_steps,
            'intervals_main': int(max(0, switch_step)),
            'intervals_refine': effective,
            'intervals_total_model_evaluations': total_steps,
        })
        return (main, refine, total_steps, switch_step, effective, report)





class _MiniMaxH3SeededEmptyNoise:
    """Zero-noise carrier that preserves the stage-1 effective seed.

    Stock ComfyUI DisableNoise uses seed=0. SamplerCustomAdvanced forwards the
    NOISE object's seed into guider.sample(), so a two-stage LongMedia trajectory
    must carry the original effective seed even though stage 2 adds no noise.
    """
    def __init__(self, seed):
        self.seed = int(seed) & 0xFFFFFFFFFFFFFFFF

    def generate_noise(self, input_latent):
        import comfy.sample
        return comfy.sample.prepare_empty_noise(input_latent["samples"])


class MiniMaxH3LatentLabSeededDisableNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ('NOISE',)
    RETURN_NAMES = ('noise',)
    FUNCTION = 'build'
    CATEGORY = 'MiniMax H3/LongMedia/Internal'

    def build(self, seed=0):
        _lm_print(
            '[MiniMaxH3 LongMedia][SEEDED DISABLE NOISE] '
            f'add_noise=False; forwarded_seed={int(seed) & 0xFFFFFFFFFFFFFFFF}',
            flush=True,
        )
        return (_MiniMaxH3SeededEmptyNoise(seed),)


class MiniMaxH3LatentLabProtectRefineAV:
    """Keep refine video-only and restore the frozen continuation head."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'base_av': ('LATENT',),
                'refined_av': ('LATENT',),
                'overlap_frames': ('INT', {'default': 0, 'min': 0, 'max': 10000, 'step': 1}),
            }
        }

    RETURN_TYPES = ('LATENT', 'STRING')
    RETURN_NAMES = ('av', 'report')
    FUNCTION = 'protect'
    CATEGORY = CATEGORY_LONGMEDIA

    def protect(self, base_av, refined_av, overlap_frames=0):
        base_video, base_audio = unpack_av_samples(base_av)
        refined_video, refined_audio = unpack_av_samples(refined_av)
        if tuple(base_video.shape) != tuple(refined_video.shape):
            raise ValueError(
                f'Refine changed video latent geometry: {tuple(base_video.shape)} -> {tuple(refined_video.shape)}'
            )
        if tuple(base_audio.shape) != tuple(refined_audio.shape):
            raise ValueError(
                f'Refine changed audio latent geometry: {tuple(base_audio.shape)} -> {tuple(refined_audio.shape)}'
            )
        out_video = refined_video.clone()
        out_audio = refined_audio.clone()
        protected_video_t = 0
        protected_audio_t = 0
        overlap_frames = max(0, int(overlap_frames))
        if overlap_frames > 0:
            try:
                protected_video_t = min(int(video_latent_t(overlap_frames)), int(out_video.shape[2]))
            except Exception:
                protected_video_t = 0
            try:
                protected_audio_t = min(int(audio_latent_t(overlap_frames)), int(out_audio.shape[-1]))
            except Exception:
                protected_audio_t = 0
            if protected_video_t > 0:
                out_video[:, :, :protected_video_t] = base_video[:, :, :protected_video_t].to(
                    device=out_video.device, dtype=out_video.dtype
                )
            if protected_audio_t > 0:
                out_audio[..., :protected_audio_t] = base_audio[..., :protected_audio_t].to(
                    device=out_audio.device, dtype=out_audio.dtype
                )

        out = dict(refined_av)
        out['samples'] = NestedTensor((out_video, out_audio))
        # Preserve the original context mask metadata. The low-noise stage is a
        # continuation of the same schedule, so audio is refined too; only the
        # exact frozen overlap is restored from the pre-sampling segment input.
        if 'noise_mask' in base_av:
            out['noise_mask'] = base_av['noise_mask']
        else:
            out.pop('noise_mask', None)
        report = json.dumps({
            'audio_restored_from_main_pass': False,
            'audio_refined_as_same_trajectory': True,
            'protected_overlap_frames': overlap_frames,
            'protected_video_latent_steps': protected_video_t,
            'protected_audio_latent_steps': protected_audio_t,
        })
        return (out, report)


def _h3_model_size_bytes_from_guider(guider):
    """Best-effort model storage size for sampler-local residency policy."""
    patcher = getattr(guider, 'model_patcher', None)
    if patcher is None:
        return None
    for name in ('model_size', 'loaded_size'):
        fn = getattr(patcher, name, None)
        if callable(fn):
            try:
                value = int(fn())
                if value > 0:
                    return value
            except Exception:
                pass
    return None



class MiniMaxH3LatentLabUltraPinnedMemoryGate:
    """Temporarily mirror ComfyUI --disable-pinned-memory for ultra H3 sampling."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'guider': ('GUIDER',), 'enable': ('BOOLEAN', {'default': True})}}
    RETURN_TYPES = ('GUIDER', 'BOOLEAN')
    RETURN_NAMES = ('guider', 'previous_disable_pinned_memory')
    FUNCTION = 'apply'
    CATEGORY = CATEGORY_LONGMEDIA

    def apply(self, guider, enable=True):
        previous = False
        if bool(enable):
            try:
                from comfy.cli_args import args as _args
                previous = bool(getattr(_args, 'disable_pinned_memory', False))
                patcher = getattr(guider, 'model_patcher', None)

                # v0.4.73: transport-aware sampler gate.
                #
                # The old workaround globally disabled pinned memory for every
                # oversized H3 sampler. That was introduced for older AIMDO Windows
                # HostBuffer failures. With recent AIMDO native threaded DynamicVRAM,
                # doing so also disables the best host->GPU transport path exactly
                # while weights are being asynchronously streamed.
                #
                # Preserve the user's pinned-memory ON state only for the measured
                # native TensorWise INT8 fastpath. TE/reference encoding keeps its
                # separate conservative gate and is intentionally unchanged.
                _profile = _detect_h3_model_runtime(patcher) if patcher is not None else {}
                _backend = str(_profile.get('backend') or 'unknown').lower()
                _qvariant = str(_profile.get('quant_variant') or '').lower()
                _aimdo_raw, _aimdo_ver = _pkg_version_tuple('comfy-aimdo')
                _kitchen_raw, _kitchen_ver = _pkg_version_tuple('comfy-kitchen')
                _recent_aimdo = bool(_aimdo_ver is not None and _aimdo_ver >= (0, 4, 6))
                _recent_kitchen = bool(_kitchen_ver is not None and _kitchen_ver >= (0, 2, 0))
                _native_tensorwise_int8 = bool(
                    _backend == 'int8'
                    and ('tensorwise' in _qvariant or _qvariant in ('', 'int8', 'tensorwise-int8'))
                )
                # v0.5.38: full-model host pinning is fast only when RAM has
                # enough real headroom.  Comfy/AIMDO may otherwise page-lock a
                # host buffer roughly the size of the whole quantized H3 model
                # (19.5 GB for the common comfy-int8 checkpoint).  On a 64 GB
                # workstation that can leave ~10 GB reclaimable RAM after refs,
                # CLIP/VAEs and file cache are present.  Pinned pages cannot be
                # reclaimed by Windows, so this is a transport optimization that
                # can turn into system pressure and makes repeat-run behavior less
                # deterministic.  Keep the fastpath only when the projected
                # post-pin RAM reserve is healthy.
                _model_size_b = int(_h3_model_size_bytes_from_guider(guider) or 0)
                _ram_total_b = 0
                _ram_available_b = 0
                try:
                    import psutil as _psutil
                    _vm = _psutil.virtual_memory()
                    _ram_total_b = int(_vm.total)
                    _ram_available_b = int(_vm.available)
                except Exception:
                    pass
                try:
                    import comfy.model_management as _mm
                    _total_pinned_b = int(getattr(_mm, 'TOTAL_PINNED_MEMORY', 0) or 0)
                except Exception:
                    _total_pinned_b = 0

                _ram_ratio = (float(_model_size_b) / float(_ram_total_b)) if (_model_size_b and _ram_total_b) else 0.0
                _pin_already_materialized = bool(_model_size_b and _total_pinned_b >= int(_model_size_b * 0.50))
                _projected_available_b = int(_ram_available_b)
                if _model_size_b and not _pin_already_materialized:
                    _projected_available_b = max(0, _projected_available_b - _model_size_b)
                _reserve_floor_b = max(12 * 1024**3, int(_ram_total_b * 0.20)) if _ram_total_b else 12 * 1024**3
                _ram_pressure = bool(
                    (_ram_total_b and _model_size_b and _ram_ratio >= 0.25)
                    or (_ram_available_b and _projected_available_b < _reserve_floor_b)
                )

                _keep_pinned = bool(
                    not previous
                    and _recent_aimdo
                    and _recent_kitchen
                    and _native_tensorwise_int8
                    and not _ram_pressure
                )

                if _keep_pinned:
                    # No global flip, no unpin_all_weights, no forced empty_cache.
                    # Native AIMDO owns pinned page lifecycle and threaded prefetch.
                    _lm_print(
                        '[MiniMaxH3 LongMedia][PINNED-MEMORY FASTPATH] '
                        f'disable_pinned_memory {previous}->{previous}; '
                        f'backend={_backend}; quant={_qvariant or "tensorwise-int8"}; '
                        f'aimdo={_aimdo_raw or "unknown"}; kitchen={_kitchen_raw or "unknown"}; '
                        'pinned_h2d=True; prefetch=NATIVE_THREADED; '
                        'legacy_unpin=False; scope=diffusion_sampler_only',
                        flush=True,
                    )
                else:
                    _args.disable_pinned_memory = True
                    if patcher is not None and hasattr(patcher, 'unpin_all_weights'):
                        try:
                            patcher.unpin_all_weights()
                        except Exception as exc:
                            _lm_print('[MiniMaxH3 LongMedia][PINNED-MEMORY GATE] unpin warning: '
                                      f'{type(exc).__name__}: {exc}', flush=True)
                    try:
                        import comfy.model_management as _mm
                        if hasattr(_mm, 'soft_empty_cache'):
                            _mm.soft_empty_cache()
                    except Exception:
                        pass
                    _pin_reason = (
                        'ram_pressure_full_model_pin_rejected'
                        if _ram_pressure and _native_tensorwise_int8
                        else 'legacy_or_user_disabled_transport_path'
                    )
                    _lm_print(
                        '[MiniMaxH3 LongMedia][PINNED-MEMORY GATE] '
                        f'disable_pinned_memory {previous}->True; '
                        f'backend={_backend}; quant={_qvariant or "unknown"}; '
                        f'reason={_pin_reason}; '
                        f'model={_model_size_b/(1024.0**3):.1f}GB; '
                        f'ram_total={_ram_total_b/(1024.0**3):.1f}GB; '
                        f'ram_available={_ram_available_b/(1024.0**3):.1f}GB; '
                        f'pinned_total={_total_pinned_b/(1024.0**3):.1f}GB; '
                        f'projected_after_pin={_projected_available_b/(1024.0**3):.1f}GB; '
                        'existing model pins released before first weight fault',
                        flush=True,
                    )
            except Exception as exc:
                _lm_print('[MiniMaxH3 LongMedia][PINNED-MEMORY GATE] unavailable: '
                          f'{type(exc).__name__}: {exc}', flush=True)
        return (guider, previous)


class MiniMaxH3LatentLabUltraPinnedMemoryRestore:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'final_av': ('LATENT',),
            'previous_disable_pinned_memory': ('BOOLEAN', {'default': False}),
            'restore': ('BOOLEAN', {'default': True}),
        }}
    RETURN_TYPES = ('LATENT',)
    RETURN_NAMES = ('final_av',)
    FUNCTION = 'apply'
    CATEGORY = CATEGORY_LONGMEDIA

    def apply(self, final_av, previous_disable_pinned_memory=False, restore=True):
        if bool(restore):
            try:
                from comfy.cli_args import args as _args
                _args.disable_pinned_memory = bool(previous_disable_pinned_memory)
                _lm_print(
                    '[MiniMaxH3 LongMedia][PINNED-MEMORY RESTORE] '
                    f'disable_pinned_memory restored to {bool(previous_disable_pinned_memory)}',
                    flush=True,
                )
            except Exception as exc:
                _lm_print('[MiniMaxH3 LongMedia][PINNED-MEMORY RESTORE] warning: '
                          f'{type(exc).__name__}: {exc}', flush=True)
        return (final_av,)


def _resolve_h3_memory_mode(guider, requested):
    requested = str(requested or 'auto').lower()
    if requested not in ('auto', 'normal', 'low_vram', 'ultra_low_vram'):
        requested = 'auto'
    try:
        gpu_bytes = int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory) if torch.cuda.is_available() else 0
    except Exception:
        gpu_bytes = 0
    model_bytes = _h3_model_size_bytes_from_guider(guider)
    if requested != 'auto':
        effective, reason = requested, 'forced by sampler memory_mode'
    else:
        ratio = (float(model_bytes) / float(gpu_bytes)) if (model_bytes and gpu_bytes) else None
        if ratio is not None and ratio >= 1.75:
            effective, reason = 'ultra_low_vram', f'model/VRAM ratio={ratio:.2f} >= 1.75'
        elif ratio is not None and ratio >= 1.10:
            effective, reason = 'low_vram', f'model/VRAM ratio={ratio:.2f} >= 1.10'
        elif gpu_bytes and gpu_bytes <= int(18.5 * 1024**3) and model_bytes and model_bytes >= int(24 * 1024**3):
            effective, reason = 'ultra_low_vram', 'large model on <=18.5GB GPU'
        else:
            effective, reason = 'normal', 'model fits normal residency policy'
    return {'requested': requested, 'effective': effective, 'reason': reason, 'model_bytes': model_bytes, 'gpu_bytes': gpu_bytes}



class MiniMaxH3LatentLabUnifiedRuntimeSampler:
    """Execute every LongMedia segment inside one ComfyUI sampling lifecycle.

    The stock CFGGuider.sample() owns prepare_sampling/pre_run/cleanup, which means
    calling it once per segment reloads/reinitializes H3 each time.  This runtime
    node opens that lifecycle once, runs guider.inner_sample() for every segment,
    updates continuation conditioning/latents between segments, then cleans up once.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'initial_av': ('LATENT',),
                'long_media_plan': ('LONG_MEDIA_PLAN',),
                'guider': ('GUIDER',),
                'sampler': ('SAMPLER',),
                'sigmas': ('SIGMAS',),
                'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
                'video_context_denoise': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                'audio_context_denoise': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                'offload_completed_segments': ('BOOLEAN', {'default': True}),
                'latent_hires_enabled': ('BOOLEAN', {'default': False}),
                'latent_hires_model': ('STRING', {'default': ''}),
                'latent_hires_scale': ('FLOAT', {'default': 2.0, 'min': 1.0, 'max': 4.0, 'step': 0.1}),
                'latent_hires_precision': (['fp16', 'bf16', 'fp32'], {'default': 'fp16'}),
                'latent_hires_align': ('INT', {'default': 32, 'min': 16, 'max': 256, 'step': 16}),
                'refine_enabled': ('BOOLEAN', {'default': False}),
                'refine_steps': ('INT', {'default': 2, 'min': 1, 'max': 1000, 'step': 1}),
            }
        }

    RETURN_TYPES = ('LATENT', 'STRING')
    RETURN_NAMES = ('final_av', 'runtime_report')
    FUNCTION = 'run'
    CATEGORY = 'MiniMax H3/LongMedia/Internal'

    @staticmethod
    def _copy_conds(original_conds):
        import comfy.samplers
        conds = {}
        for key, values in (original_conds or {}).items():
            conds[key] = [item.copy() if hasattr(item, 'copy') else item for item in values]
        comfy.samplers.preprocess_conds_hooks(conds)
        return conds

    @staticmethod
    def _pack_segment_inputs(latent, model_patcher, seed, device):
        import comfy.sample
        import comfy.utils
        import comfy.sampler_helpers

        local = latent.copy()
        samples = comfy.sample.fix_empty_latent_channels(
            model_patcher,
            local['samples'],
            local.get('downscale_ratio_spacial', None),
            local.get('downscale_ratio_temporal', None),
        )
        local['samples'] = samples
        batch_inds = local.get('batch_index', None)
        noise = comfy.sample.prepare_noise(samples, int(seed) & 0xFFFFFFFFFFFFFFFF, batch_inds)

        if getattr(samples, 'is_nested', False):
            stream_shapes = [tuple(x.shape) for x in samples.unbind()]
            packed_samples, latent_shapes = comfy.utils.pack_latents(samples.unbind())
            packed_noise, _ = comfy.utils.pack_latents(noise.unbind())
        else:
            stream_shapes = [tuple(samples.shape)]
            latent_shapes = [samples.shape]
            packed_samples = samples
            packed_noise = noise

        denoise_mask = local.get('noise_mask', None)
        if denoise_mask is not None:
            if getattr(denoise_mask, 'is_nested', False):
                masks = list(denoise_mask.unbind())[:len(latent_shapes)]
            else:
                masks = [denoise_mask]
            for i in range(len(masks), len(latent_shapes)):
                masks.append(torch.ones(latent_shapes[i]))
            for i in range(len(masks)):
                masks[i] = comfy.sampler_helpers.prepare_mask(masks[i], latent_shapes[i], device)
            if len(masks) > 1:
                denoise_mask, _ = comfy.utils.pack_latents(masks)
            else:
                denoise_mask = masks[0]
            denoise_mask = denoise_mask.float()

        return local, packed_samples, packed_noise, denoise_mask, latent_shapes, stream_shapes

    @staticmethod
    def _unpack_segment_output(template_latent, packed_output, latent_shapes):
        import comfy.utils
        import comfy.nested_tensor
        out = template_latent.copy()
        out.pop('downscale_ratio_spacial', None)
        out.pop('downscale_ratio_temporal', None)
        if len(latent_shapes) > 1:
            out['samples'] = comfy.nested_tensor.NestedTensor(
                comfy.utils.unpack_latents(packed_output, latent_shapes)
            )
        else:
            out['samples'] = packed_output
        return out

    @staticmethod
    def _run_stock_sample_with_reused_lifecycle(
        guider, device, noise, latent_image, sampler, sigmas, denoise_mask,
        callback, disable_pbar, seed,
    ):
        """Run the official CFGGuider.sample() contract without reopening H3 residency.

        LongMedia owns prepare_sampling/pre_run/cleanup once for the whole sequence.
        Third-party integrations, however, attach to CFGGuider.sample() itself (notably
        OUTER_SAMPLE wrappers such as KJ Model Preview Override).  Keep the stock sample
        path intact and replace only this guider instance's outer_sample terminal with a
        lifecycle-reuse implementation that delegates directly to inner_sample().

        This preserves ComfyUI's native NestedTensor packing, latent_shapes propagation,
        callback adaptation, model_options cloning, hook filtering and wrapper dispatch.
        """
        import types
        import comfy.samplers

        had_instance_outer = 'outer_sample' in getattr(guider, '__dict__', {})
        previous_instance_outer = getattr(guider, '__dict__', {}).get('outer_sample', None)

        def _outer_sample_reuse(
            self, wrapped_noise, wrapped_latent, wrapped_sampler, wrapped_sigmas,
            wrapped_mask=None, wrapped_callback=None, wrapped_disable_pbar=False,
            wrapped_seed=None, latent_shapes=None,
        ):
            with comfy.model_management.cuda_device_context(device):
                wrapped_noise = wrapped_noise.to(device=device, dtype=torch.float32)
                wrapped_latent = wrapped_latent.to(device=device, dtype=torch.float32)
                wrapped_sigmas = wrapped_sigmas.to(device)
                if wrapped_mask is not None:
                    wrapped_mask = wrapped_mask.to(device)
                # sample() may have cloned/cast wrapper state for this pass.  Apply those
                # casts, but deliberately do not call prepare_sampling/pre_run/cleanup.
                comfy.samplers.cast_to_load_options(
                    self.model_options, device=device, dtype=self.model_patcher.model_dtype()
                )
                return self.inner_sample(
                    wrapped_noise, wrapped_latent, device, wrapped_sampler, wrapped_sigmas,
                    wrapped_mask, wrapped_callback, wrapped_disable_pbar, wrapped_seed,
                    latent_shapes=latent_shapes,
                )

        guider.outer_sample = types.MethodType(_outer_sample_reuse, guider)
        try:
            return guider.sample(
                noise, latent_image, sampler, sigmas, denoise_mask, callback,
                disable_pbar, seed,
            )
        finally:
            if had_instance_outer:
                guider.outer_sample = previous_instance_outer
            else:
                try:
                    del guider.outer_sample
                except Exception:
                    pass

    @staticmethod
    def _cpu_latent_copy(latent):
        import comfy.nested_tensor
        copied = latent.copy()
        samples = copied.get('samples')
        if getattr(samples, 'is_nested', False):
            copied['samples'] = comfy.nested_tensor.NestedTensor([x.detach().cpu() for x in samples.unbind()])
        elif torch.is_tensor(samples):
            copied['samples'] = samples.detach().cpu()
        noise_mask = copied.get('noise_mask', None)
        if getattr(noise_mask, 'is_nested', False):
            copied['noise_mask'] = comfy.nested_tensor.NestedTensor([x.detach().cpu() for x in noise_mask.unbind()])
        elif torch.is_tensor(noise_mask):
            copied['noise_mask'] = noise_mask.detach().cpu()
        return copied

    @staticmethod
    def _hires_second_pass_sigmas(full_sigmas, requested_steps: int):
        """Build an independent hi-res schedule from the original scheduler.

        The reference workflow does not continue the low-sigma tail.  It starts a
        fresh pass from an upscaled x0 using every-other scheduler point beginning
        after sigma_max (8-step simple -> indices 1,3,5 for a 3-step pass), then 0.
        This preserves the user's scheduler curve without inventing linear sigmas.
        """
        if not torch.is_tensor(full_sigmas):
            full_sigmas = torch.as_tensor(full_sigmas, dtype=torch.float32)
        sig = full_sigmas.detach().flatten()
        if sig.numel() < 2:
            raise RuntimeError('Latent Hi-Res needs a sigma schedule with at least one denoise step.')
        steps = max(1, int(requested_steps))
        nonzero_last = int(sig.numel()) - 2 if float(sig[-1]) == 0.0 else int(sig.numel()) - 1
        candidates = list(range(1, nonzero_last + 1, 2))
        if not candidates:
            candidates = [0]
        if len(candidates) < steps:
            for idx in range(1, nonzero_last + 1):
                if idx not in candidates:
                    candidates.append(idx)
                if len(candidates) >= steps:
                    break
        chosen = sorted(candidates[:steps])
        selected = sig[torch.tensor(chosen, device=sig.device, dtype=torch.long)]
        zero = torch.zeros((1,), device=sig.device, dtype=sig.dtype)
        return torch.cat((selected, zero)), chosen

    @staticmethod
    def _assemble_external_refine_global_av(segment_latents, overlap_frames):
        """Assemble sampler #1's native high-res segments into one AV timeline.

        This uses the same phase-safe LongMedia stitch contract as final timeline
        assembly: repeated continuation overlap is removed exactly, no blending is
        performed, and AV synchronization is validated after every append.

        Crucially, sampler #2 then denoises this ONE continuous latent, so there
        are no independent diffusion solutions on opposite sides of a clip seam.
        """
        if not isinstance(segment_latents, (list, tuple)) or not segment_latents:
            raise RuntimeError('Global external refine requires stored segment latents.')

        clean = []
        for i, av in enumerate(segment_latents):
            if not isinstance(av, dict) or av.get('samples') is None:
                raise RuntimeError(
                    f'Global external refine segment {i} is not a valid AV latent.'
                )
            item = dict(av)
            item.pop('noise_mask', None)
            clean.append(item)

        assembled = clean[0]
        total_frames = int(frame_count_from_video_t(unpack_av_samples(assembled)[0].shape[2]))
        for item in clean[1:]:
            assembled, total_frames = stitch_continuation(
                assembled,
                item,
                int(overlap_frames),
                NestedTensor,
                False,   # blend_video_overlap
                True,    # offload_to_cpu: keep the growing movie out of VRAM
                0,       # visible seam blend disabled
            )

        video, audio = unpack_av_samples(assembled)
        actual_frames = int(frame_count_from_video_t(video.shape[2]))
        expected_audio_t = int(audio_latent_t(actual_frames))
        if actual_frames != int(total_frames):
            raise RuntimeError(
                '[GLOBAL REFINE ASSEMBLY] frame accounting mismatch: '
                f'actual={actual_frames}, reported={int(total_frames)}.'
            )
        if int(audio.shape[-1]) != expected_audio_t:
            raise RuntimeError(
                '[GLOBAL REFINE ASSEMBLY] AV sync mismatch: '
                f'frames={actual_frames}, audio_t={int(audio.shape[-1])}, '
                f'expected_audio_t={expected_audio_t}.'
            )
        # H3 video latent timelines must stay on their native 5*k+2 lattice.
        if (int(video.shape[2]) - 2) % 5 != 0:
            raise RuntimeError(
                '[GLOBAL REFINE ASSEMBLY] invalid H3 temporal lattice: '
                f'video_t={int(video.shape[2])}, expected 5*k+2.'
            )

        assembled = dict(assembled)
        assembled.pop('noise_mask', None)
        assembled.pop('_lm_per_clip_native_video_decode', None)
        assembled.pop('_lm_segment_latents', None)
        assembled.pop('_lm_segment_lengths', None)
        assembled.pop('_lm_segment_hidden_overlaps', None)
        assembled.pop('_lm_segment_workflow', None)
        assembled.pop('_lm_external_refine_ready', None)
        assembled['_lm_global_continuous_refine_input'] = True
        assembled['_lm_global_continuous_refine_frames'] = actual_frames
        return assembled

    @staticmethod
    def _prepare_external_refine_input(segment_av):
        """Return a clean stored segment for chained sampler refinement.

        First-pass continuation masks describe how the segment was generated.
        They are NOT part of the contract of a later custom-sigma refiner.
        """
        out = dict(segment_av or {})
        had_noise_mask = out.pop('noise_mask', None) is not None
        return out, had_noise_mask

    @staticmethod
    def _hires_conditioning_without_video_keyframes(original_conds):
        """Clone conditioning shells and drop geometry-bound VIDEO keyframes.

        Native H3 keyframe rows must use the target spatial grid.  After spatial
        latent upscale, low-res keyframe latents cannot be inserted into the new
        layout (the exact 2550->5610 row mismatch seen in 0.4.52).  Ref2VA refs are
        allowed to keep their own geometry, and audio-only keyframes are geometry
        independent, so both remain intact.  The upscaled x0 itself carries the
        image/continuation structure into the independent second pass.
        """
        out = {}
        dropped = 0
        for cond_name, entries in (original_conds or {}).items():
            cloned = []
            for entry in entries or []:
                if isinstance(entry, dict):
                    meta = dict(entry)
                    kfs = list(meta.get('minimax_keyframes', []) or [])
                    if kfs:
                        keep = [dict(kf) for kf in kfs if kf.get('latent') is None]
                        dropped += len(kfs) - len(keep)
                        if keep:
                            meta['minimax_keyframes'] = keep
                        else:
                            meta.pop('minimax_keyframes', None)
                    cloned.append(meta)
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], dict):
                    new_entry = list(entry)
                    meta = dict(entry[1])
                    kfs = list(meta.get('minimax_keyframes', []) or [])
                    if kfs:
                        keep = [dict(kf) for kf in kfs if kf.get('latent') is None]
                        dropped += len(kfs) - len(keep)
                        if keep:
                            meta['minimax_keyframes'] = keep
                        else:
                            meta.pop('minimax_keyframes', None)
                    new_entry[1] = meta
                    cloned.append(new_entry)
                else:
                    cloned.append(entry)
            out[cond_name] = cloned
        return out, dropped

    def run(self, initial_av, long_media_plan, guider, sampler, sigmas, seed,
            video_context_denoise=0.0, audio_context_denoise=0.0,
            offload_completed_segments=True,
            latent_hires_enabled=False, latent_hires_model='', latent_hires_scale=2.0,
            latent_hires_precision='fp16', latent_hires_align=32,
            refine_enabled=False, refine_steps=2):
        import comfy.samplers
        import comfy.sampler_helpers
        import comfy.model_management
        import comfy.model_patcher
        import comfy.hooks
        import comfy.utils
        import comfy.multigpu
        import latent_preview

        plan = long_media_plan
        passes = max(1, int(getattr(plan, 'passes', 1) or 1))
        segmentation_active = bool(getattr(plan, 'segmentation_active', False))
        workflow_mode = str(getattr(plan, 'workflow_mode', '') or '')

        # FastH3 Preview is a 4-call DMD2 student. Detect the loader-attached
        # contract before any sigma splitting so a workflow cannot accidentally
        # feed 8/20-step schedules into it. This changes sampling cadence only;
        # LongMedia's lifecycle, segmentation and VRAM policy stay untouched.
        _sampler_model_patcher = guider.model_patcher
        _sampler_fasth3_contract = None
        _sampler_fastvideo_vsa_contract = None
        try:
            _sampler_diffusion = _sampler_model_patcher.get_model_object('diffusion_model')
            _sampler_fasth3_contract = getattr(_sampler_diffusion, '_longmedia_fasth3_contract', None)
            _sampler_fastvideo_vsa_contract = getattr(_sampler_diffusion, '_longmedia_fastvideo_vsa_contract', None)
        except Exception:
            _sampler_fasth3_contract = None
            _sampler_fastvideo_vsa_contract = None
        _four_call_contract = (
            _sampler_fastvideo_vsa_contract
            if isinstance(_sampler_fastvideo_vsa_contract, dict)
            else _sampler_fasth3_contract
        )
        _is_fastvideo_vsa = False
        if isinstance(_four_call_contract, dict):
            _is_fastvideo_vsa = isinstance(_sampler_fastvideo_vsa_contract, dict)
            _family = 'FastVideo VSA' if _is_fastvideo_vsa else 'FastH3'
            # The Kijai/FastVideo VSA checkpoint constrains only sampler #1: its
            # distilled base trajectory is exactly four transformer forwards.
            # LongMedia's sampler #2/refiner is user-owned and must keep the
            # refine_steps/schedule selected by the workflow.  Do NOT split the
            # four-step student schedule into base+refine and do NOT auto-disable
            # sampler #2.  H3ddle FastH3 keeps its older fail-closed policy.
            if (not _is_fastvideo_vsa) and (bool(refine_enabled) or bool(latent_hires_enabled)):
                raise RuntimeError(
                    f'[MiniMaxH3 LongMedia][{_family} STARTUP PRECHECK] Preview v1 is exactly four calls; '
                    'LongMedia refine/latent-hires would add off-distribution denoise calls. Disable both.'
                )
            _sig_device = sigmas.device if torch.is_tensor(sigmas) else _sampler_model_patcher.load_device
            _sig_dtype = sigmas.dtype if torch.is_tensor(sigmas) and sigmas.dtype.is_floating_point else torch.float32
            full_sigmas = _fast_h3_native_sigmas(_sig_device, _sig_dtype)
            _lm_print(
                f'[MiniMaxH3 LongMedia][{_family} 4-STEP] sampler_1 forced trained sigma ladder; '
                f'sigmas={[round(float(v), 8) for v in full_sigmas.detach().cpu().tolist()]}; '
                'video_shift=12 audio_shift=3',
                flush=True,
            )
            if _is_fastvideo_vsa and bool(refine_enabled):
                _lm_print(
                    '[MiniMaxH3 LongMedia][FastVideo VSA REFINER OWNERSHIP] '
                    f'sampler_1_steps=4; sampler_2_refine_steps_requested={int(refine_steps)}; '
                    'sampler #2 keeps the workflow-selected refine schedule; no 8->4+4 split is inferred from sampler #1.',
                    flush=True,
                )
        else:
            full_sigmas = sigmas

        main_steps = max(0, int(full_sigmas.numel()) - 1) if torch.is_tensor(full_sigmas) else max(0, len(full_sigmas) - 1)
        total_steps = main_steps
        if bool(refine_enabled):
            if _is_fastvideo_vsa:
                # Sampler #1 is the fixed four-step student.  Sampler #2 is an
                # independent LongMedia refiner whose step count comes only from
                # refine_steps.  Its sigma tail is selected from the workflow's
                # connected SIGMAS schedule, never from the forced four-step base.
                _unused_main, runtime_refine_sigmas, _connected_total, _connected_switch, refine_steps_effective, _ = split_refine_sigmas(
                    sigmas, int(refine_steps)
                )
                runtime_main_sigmas = full_sigmas
                refine_switch_step = main_steps
                total_steps = main_steps + int(refine_steps_effective)
            else:
                runtime_main_sigmas, runtime_refine_sigmas, total_steps, refine_switch_step, refine_steps_effective, _ = split_refine_sigmas(
                    full_sigmas, int(refine_steps)
                )
        else:
            runtime_main_sigmas = full_sigmas
            runtime_refine_sigmas = None
            refine_switch_step = main_steps
            refine_steps_effective = 0
        hires_second_sigmas = None
        hires_sigma_indices = []
        if bool(latent_hires_enabled) and bool(refine_enabled) and int(refine_steps_effective) > 0:
            _secondary_sigma_source = sigmas if _is_fastvideo_vsa else full_sigmas
            hires_second_sigmas, hires_sigma_indices = self._hires_second_pass_sigmas(_secondary_sigma_source, int(refine_steps_effective))
        _lm_print(
            '[MiniMaxH3 LongMedia][TWO-PASS HIRES] '
            f'hires={bool(latent_hires_enabled)} refine={bool(refine_enabled)}; total_steps={int(total_steps)}; '
            f'lowres_steps={int(refine_switch_step)}; second_pass_steps={int(refine_steps_effective)}; '
            f'second_sigma_indices={hires_sigma_indices if hires_second_sigmas is not None else None}; '
            'model_lifecycle=single; second_model_load=False',
            flush=True,
        )
        if (not segmentation_active) and workflow_mode not in ('multiclip',) and passes != 1:
            raise RuntimeError(
                f'LongMedia invariant violated: segmentation is disabled for workflow={workflow_mode!r} but passes={passes}. '
                'Only manual/segmented_continuation may create segmentation passes.'
            )
        model_patcher = guider.model_patcher
        device = model_patcher.load_device

        # v0.5.37: seed-only / sampler-only reruns can bypass Setup entirely due
        # ComfyUI graph caching. The previous downstream decoder may therefore
        # still own several GB of VRAM and AIMDO/VBAR cast/prefetch state. Make
        # every sampler invocation start from the same clean memory boundary as
        # a cold first run, before any latent is moved to CUDA or prepare_sampling
        # performs its residency calculation.
        sampler_execution_boundary = _sampler_memory_isolation(
            'sampler_entry', unload_models=True
        )

        # Save externally-owned guider state. The runtime temporarily mutates it
        # exactly as CFGGuider.sample() does, but restores everything at the end.
        original_model_options = guider.model_options
        original_original_conds = getattr(guider, 'original_conds', None)
        runtime_template_guider = copy.copy(guider)
        runtime_template_guider.original_conds = dict(original_original_conds or {})
        # Structured Comfy clone only: model_options can contain invalidated / device-backed
        # storages that Python deepcopy() cannot safely clone.
        runtime_template_guider.model_options = comfy.model_patcher.create_model_options_clone(
            original_model_options or {}
        )
        original_hook_mode = model_patcher.hook_mode
        original_inner_model = getattr(guider, 'inner_model', None)
        original_conds_runtime = getattr(guider, 'conds', None)
        original_loaded_models = getattr(guider, 'loaded_models', None)

        first_seed = _v85_segment_seed(plan, seed, 0) if getattr(plan, 'mode', None) == 'multiclip' else int(seed) & 0xFFFFFFFFFFFFFFFF

        # v0.4.71: sampler #2 refines one continuous high-res AV timeline.
        #
        # Sampler #1 may still generate clip-by-clip for VRAM efficiency and native
        # H3 Motion Context. But once high-res x0 segments exist, sampler #2 must not
        # solve the same final movie as independent clip diffusion problems.
        workflow_name = str(getattr(plan, 'workflow_mode', '') or '')
        external_refine_segments = None
        external_refine_global_mode = False

        if workflow_name in ('multiclip', 'segmented_continuation', 'reconstruct') and isinstance(initial_av, dict):
            _candidate_segments = initial_av.get('_lm_segment_latents')
            _candidate_workflow = str(initial_av.get('_lm_segment_workflow') or workflow_name)
            if (
                _candidate_workflow == workflow_name
                and isinstance(_candidate_segments, (list, tuple))
                and len(_candidate_segments) >= int(passes)
                and all(isinstance(x, dict) and x.get('samples') is not None
                        for x in _candidate_segments[:int(passes)])
            ):
                external_refine_segments = list(_candidate_segments[:int(passes)])
                external_refine_global_mode = True

        external_refine_mode = bool(external_refine_global_mode)

        if external_refine_global_mode:
            original_segment_passes = int(passes)
            first_source_av = self._assemble_external_refine_global_av(
                external_refine_segments,
                int(getattr(plan, 'overlap_frames', 0) or 0),
            )
            # Sampler #2 is intentionally one diffusion problem / one progress pass.
            passes = 1
            first_refine_mask_cleared = True
        else:
            original_segment_passes = int(passes)
            first_source_av = initial_av
            first_refine_mask_cleared = False

        first_local, first_latent, first_noise, first_mask, first_shapes, _ = self._pack_segment_inputs(
            first_source_av, model_patcher, first_seed, device
        )

        if external_refine_global_mode:
            _global_v, _global_a = unpack_av_samples(first_source_av)
            _lm_print(
                '[MiniMaxH3 LongMedia][GLOBAL CONTINUOUS REFINE] '
                f'workflow={workflow_name}; source_segments={original_segment_passes}; '
                'refine_passes=1; one_diffusion_timeline=True; '
                f'video_t={int(_global_v.shape[2])}; '
                f'frames={int(frame_count_from_video_t(_global_v.shape[2]))}; '
                f'audio_t={int(_global_a.shape[-1])}; '
                'fresh_noise=True; noise_contract=ordinary_prepare_noise; '
                'segment_seam_solvers=0; duplicate_previous_av=False; '
                'rgb_blend=False; latent_blend=False',
                flush=True,
            )

        # Prepare hooks/model exactly once for the whole movie.
        guider.conds = self._copy_conds(getattr(guider, 'original_conds', {}) or {})
        guider.model_options = comfy.model_patcher.create_model_options_clone(original_model_options)
        if comfy.samplers.get_total_hook_groups_in_conds(guider.conds) <= 1:
            model_patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
        comfy.sampler_helpers.prepare_model_patcher(model_patcher, guider.conds, guider.model_options)
        comfy.samplers.filter_registered_hooks_on_conds(guider.conds, guider.model_options)

        inner_model = loaded_models = multigpu_patchers = None
        runtime_thread_pool = None
        stitched = previous_segment = previous_segment_continuation = None
        completed = 0
        store_per_clip_native_decode = (
            str(getattr(plan, 'workflow_mode', '') or '') == 'multiclip'
            and not bool(external_refine_global_mode)
        )
        store_refine_segments = (
            str(getattr(plan, 'workflow_mode', '') or '') in ('multiclip', 'segmented_continuation', 'reconstruct')
            and not bool(external_refine_global_mode)
        )
        per_clip_segment_latents = []
        per_clip_segment_lengths = []
        per_clip_hidden_overlaps = []
        loop_closure_report = {
            'enabled': bool(getattr(plan, 'loop_closure_enabled', False)),
            'applied': False,
            'mode': 'disabled',
        }
        try:
            inner_model, prepared_conds, loaded_models = comfy.sampler_helpers.prepare_sampling(
                model_patcher, first_noise.shape, guider.conds, guider.model_options
            )
            guider.inner_model = inner_model
            guider.conds = prepared_conds
            guider.loaded_models = loaded_models

            multigpu_patchers = comfy.sampler_helpers.prepare_model_patcher_multigpu_clones(
                model_patcher, loaded_models, guider.model_options
            )
            if multigpu_patchers:
                all_devices = [device] + [p.load_device for p in multigpu_patchers]
                runtime_thread_pool = comfy.multigpu.MultiGPUThreadPool(all_devices)
                guider.model_options['multigpu_thread_pool'] = runtime_thread_pool

            with comfy.model_management.cuda_device_context(device):
                comfy.samplers.cast_to_load_options(
                    guider.model_options, device=device, dtype=model_patcher.model_dtype()
                )
                model_patcher.pre_run()
                for patcher in multigpu_patchers:
                    patcher.pre_run()

                for segment_index in range(passes):
                    if external_refine_global_mode:
                        segment_av = first_local
                        external_refine_mask_cleared = True
                        effective_seed = int(seed) & 0xFFFFFFFFFFFFFFFF
                        # One global guider shell. Geometry-bound VIDEO keyframes are
                        # removed below because they were encoded on sampler #1's
                        # pre-hires grids. Text/refs/audio-only conditioning remain.
                        segment_guider = runtime_template_guider
                    elif external_refine_mode:
                        # Kept only as a defensive fallback; production chained refine
                        # for supported long-video workflows should enter global mode.
                        segment_av, external_refine_mask_cleared = self._prepare_external_refine_input(
                            first_local if segment_index == 0 else external_refine_segments[segment_index]
                        )
                        effective_seed = (int(seed) + int(segment_index)) & 0xFFFFFFFFFFFFFFFF
                        segment_guider = runtime_template_guider
                    elif segment_index == 0:
                        segment_av = first_local
                        effective_seed = first_seed
                        segment_guider = runtime_template_guider
                    else:
                        if getattr(plan, 'mode', None) == 'storyboard_bridge':
                            storyboard_avs = getattr(plan, 'storyboard_segment_avs', None)
                            if not storyboard_avs or segment_index >= len(storyboard_avs):
                                raise RuntimeError('Unified runtime: storyboard pass latent is missing from LongMediaPlan')
                            segment_av = storyboard_avs[segment_index]
                            segment_guider = _clone_guider_with_segment_audio(
                                runtime_template_guider, plan, segment_index, previous_av=None
                            )
                        elif workflow_mode == 'multiclip':
                            # v0.4.21: generate the clip directly on the planned native
                            # continuation overlap. Clip 2+ inherits the exact generated latent
                            # head with denoise=0; native motion context is auxiliary. No RGB
                            # seam search or per-clip VideoVAE reset. The repeated head is
                            # removed later in LATENT space before one continuous decode.
                            segment_av = MiniMaxH3LatentLabLongMediaNextSegment().prepare(
                                plan, previous_segment_continuation, segment_index, 0.0, 0.0
                            )[0]
                            segment_guider = _clone_guider_with_segment_audio(
                                runtime_template_guider, plan, segment_index, previous_av=previous_segment_continuation
                            )
                        elif segmentation_active:
                            segment_av = MiniMaxH3LatentLabLongMediaNextSegment().prepare(
                                plan, previous_segment_continuation, segment_index,
                                float(video_context_denoise), float(audio_context_denoise)
                            )[0]
                            # Geometry authority must be identical for the next target latent
                            # and its native motion-context keyframes.  With Latent Hi-Res,
                            # ``previous_segment`` is the upscaled display/output branch while
                            # ``previous_segment_continuation`` intentionally remains on the
                            # low-resolution H3 grid.  Feeding the hi-res branch into the guider
                            # makes PackedLayout reserve target-grid rows for keyframes whose
                            # actual cond latents live on another H/W grid, producing:
                            #   all_video_rows[~img_update] = cond_video_rows shape mismatch.
                            # Always derive continuation conditioning from the same latent that
                            # owns ``segment_av`` geometry.  This is also the pre-hires continuity
                            # contract used by MultiClip.
                            segment_guider = _clone_guider_with_segment_audio(
                                runtime_template_guider, plan, segment_index,
                                previous_av=previous_segment_continuation
                            )
                        else:
                            raise RuntimeError(
                                f'Unexpected extra LongMedia pass outside segmentation/MultiClip: workflow={workflow_mode!r}, index={segment_index}'
                            )
                        effective_seed = (
                            _v85_segment_seed(plan, seed, segment_index)
                            if getattr(plan, 'mode', None) == 'multiclip'
                            else int(seed) & 0xFFFFFFFFFFFFFFFF
                        )

                    local, packed_latent, packed_noise, denoise_mask, latent_shapes, _ = self._pack_segment_inputs(
                        segment_av, model_patcher, effective_seed, device
                    )
                    packed_latent = packed_latent.to(device=device, dtype=torch.float32)
                    packed_noise = packed_noise.to(device=device, dtype=torch.float32)
                    # External sampler #2 intentionally keeps _pack_segment_inputs()
                    # seeded noise. A non-zero custom starting sigma must receive the
                    # same noise contract as ordinary KSampler refinement.
                    local_sigmas = runtime_main_sigmas.to(device)
                    if denoise_mask is not None:
                        denoise_mask = denoise_mask.to(device)

                    # Switch only the per-segment conditioning/model options;
                    # model residency and pre_run state remain untouched.
                    guider.original_conds = dict(getattr(segment_guider, 'original_conds', {}) or {})
                    if external_refine_mode:
                        _clean_refine_conds, _dropped_refine_video_kfs = (
                            self._hires_conditioning_without_video_keyframes(
                                guider.original_conds
                            )
                        )
                        guider.original_conds = _clean_refine_conds
                        _lm_print(
                            '[MiniMaxH3 LongMedia][GLOBAL REFINE GEOMETRY GUARD] '
                            f'workflow={workflow_name}; unit={segment_index + 1}/{passes}; '
                            f'global_mode={bool(external_refine_global_mode)}; '
                            f'dropped_geometry_bound_video_keyframes={int(_dropped_refine_video_kfs)}; '
                            f'cleared_continuation_noise_mask={bool(external_refine_mask_cleared)}; '
                            'new_previous_av_context=False; refs_preserved=True; '
                            'audio_keyframes_preserved=True',
                            flush=True,
                        )
                    guider.conds = self._copy_conds(guider.original_conds)
                    guider.model_options = comfy.model_patcher.create_model_options_clone(
                        getattr(segment_guider, 'model_options', original_model_options)
                    )
                    if runtime_thread_pool is not None:
                        guider.model_options['multigpu_thread_pool'] = runtime_thread_pool

                    x0_output = {}
                    # Pass the native callback into CFGGuider.sample(). Stock ComfyUI owns
                    # NestedTensor callback adaptation, which is also the contract expected
                    # by OUTER_SAMPLE integrations such as KJ Model Preview Override.
                    callback = latent_preview.prepare_callback(
                        model_patcher, max(0, int(local_sigmas.shape[-1]) - 1), x0_output
                    )

                    if len(latent_shapes) > 1:
                        native_latent = comfy.nested_tensor.NestedTensor(
                            comfy.utils.unpack_latents(packed_latent, latent_shapes)
                        )
                        native_noise = comfy.nested_tensor.NestedTensor(
                            comfy.utils.unpack_latents(packed_noise, latent_shapes)
                        )
                        native_mask = None if denoise_mask is None else comfy.nested_tensor.NestedTensor(
                            comfy.utils.unpack_latents(denoise_mask, latent_shapes)
                        )
                    else:
                        native_latent = packed_latent
                        native_noise = packed_noise
                        native_mask = denoise_mask

                    _lm_print(
                        '[MiniMaxH3 LongMedia][STOCK SAMPLE CONTRACT] '
                        f'unit={segment_index + 1}/{passes}; workflow={getattr(plan, "workflow_mode", "unknown")}; seed={int(effective_seed)}; '
                        'cfg_guider_sample=True; outer_sample_lifecycle=reused',
                        flush=True,
                    )
                    # Stage 1: low-resolution pass.  When Latent Hi-Res is enabled
                    # the callback's denoised x0 is the authoritative bridge, matching
                    # SamplerCustomAdvanced.denoised_output in the author's workflow.
                    sampled_native = self._run_stock_sample_with_reused_lifecycle(
                        guider, device, native_noise, native_latent, sampler, local_sigmas,
                        native_mask, callback, False, int(effective_seed),
                    )
                    if getattr(sampled_native, 'is_nested', False):
                        sampled_packed, _ = comfy.utils.pack_latents(sampled_native.unbind())
                    else:
                        sampled_packed = sampled_native

                    lowres_x0 = x0_output.get('x0')
                    if bool(latent_hires_enabled):
                        if lowres_x0 is None:
                            raise RuntimeError(
                                'Latent Hi-Res requires the low-res denoised x0 callback output; '
                                'refusing to upscale the noisy solver state.'
                            )
                        # SamplerCustomAdvanced.denoised_output applies process_latent_out
                        # before exposing x0. Mirror that contract exactly: callback x0 is
                        # still in model-internal latent space and must not be fed directly
                        # to the learned upscaler.
                        lowres_x0 = model_patcher.model.process_latent_out(lowres_x0.cpu())
                        if not getattr(lowres_x0, 'is_nested', False):
                            raise RuntimeError('Latent Hi-Res expected native H3 NestedTensor denoised x0.')
                        x0_streams = list(lowres_x0.unbind())
                        x0_shapes = [x.shape for x in x0_streams]
                        x0_packed, _ = comfy.utils.pack_latents(x0_streams)
                        # MultiClip/segmented continuation must stay low-resolution and clean.
                        # Carry x0, never the partial noisy solver state, into the next unit.
                        base_continuation_output = self._unpack_segment_output(local, x0_packed, x0_shapes)
                    else:
                        base_continuation_output = self._unpack_segment_output(local, sampled_packed, latent_shapes)

                    if bool(latent_hires_enabled):
                        streams = list(lowres_x0.unbind())
                        if len(streams) != 2:
                            raise RuntimeError(f'Latent Hi-Res expected 2 AV x0 streams, got {len(streams)}')
                        hires_video, hires_audio = streams
                        if not str(latent_hires_model or '').strip() or str(latent_hires_model).startswith('('):
                            raise RuntimeError('Latent Hi-Res is enabled but no model is selected in models/latent_upscale_models.')
                        from .latent_hires import upscale_video
                        _lm_print(
                            '[MiniMaxH3 LongMedia][LATENT HIRES X0] '
                            f'unit={segment_index + 1}/{passes}; source=denoised_output(process_latent_out(x0)); model={latent_hires_model}; '
                            f'scale={float(latent_hires_scale):.2f} precision={latent_hires_precision}; '
                            f'video_before={tuple(hires_video.shape)}; audio_x0_preserved=True; audio_preserved=True',
                            flush=True,
                        )
                        try:
                            comfy.model_management.soft_empty_cache()
                        except Exception:
                            pass
                        hires_video = upscale_video(
                            hires_video, str(latent_hires_model), float(latent_hires_scale),
                            str(latent_hires_precision), device, int(latent_hires_align),
                        )
                        sampled_native = comfy.nested_tensor.NestedTensor((hires_video, hires_audio))
                        latent_shapes = [x.shape for x in sampled_native.unbind()]
                        sampled_packed, _ = comfy.utils.pack_latents(sampled_native.unbind())
                        _lm_print(
                            '[MiniMaxH3 LongMedia][LATENT HIRES X0] '
                            f'video_after={tuple(hires_video.shape)}; audio_shape={tuple(hires_audio.shape)}; '
                            f'independent_second_pass={bool(refine_enabled and int(refine_steps_effective) > 0)}',
                            flush=True,
                        )
                        try:
                            comfy.model_management.soft_empty_cache()
                        except Exception:
                            pass

                    # Stage 2. With Latent Hi-Res this is an INDEPENDENT pass: fresh
                    # same-seed noise + a new high-sigma schedule over the upscaled x0.
                    # Without Latent Hi-Res, retain the legacy continuous low-sigma tail.
                    if bool(refine_enabled) and int(refine_steps_effective) > 0:
                        refine_x0_output = {}
                        if bool(latent_hires_enabled):
                            if hires_second_sigmas is None:
                                raise RuntimeError('Latent Hi-Res second-pass sigmas were not constructed.')
                            refine_sigmas_device = hires_second_sigmas.to(device)
                            import comfy.sample
                            refine_latent = sampled_native
                            refine_noise = comfy.sample.prepare_noise(refine_latent, int(effective_seed)).to(device=device, dtype=torch.float32)
                            refine_mask = None

                            # Reprocess conditioning for the new target H/W. Native VIDEO
                            # keyframes are target-grid bound and therefore cannot be reused
                            # at low-res. Ref2VA refs and audio-only guides remain valid.
                            hires_conds, dropped_keyframes = self._hires_conditioning_without_video_keyframes(
                                getattr(segment_guider, 'original_conds', {}) or {}
                            )
                            guider.original_conds = hires_conds
                            _lm_print(
                                '[MiniMaxH3 LongMedia][HIRES CONDITIONING] '
                                f'unit={segment_index + 1}/{passes}; dropped_lowres_video_keyframes={int(dropped_keyframes)}; '
                                'refs_preserved=True; audio_keyframes_preserved=True; source=x0',
                                flush=True,
                            )
                        else:
                            refine_sigmas_device = runtime_refine_sigmas.to(device)
                            refine_latent = sampled_native
                            refine_mask = native_mask
                            if getattr(refine_latent, 'is_nested', False):
                                refine_noise = comfy.nested_tensor.NestedTensor([torch.zeros_like(x) for x in refine_latent.unbind()])
                            else:
                                refine_noise = torch.zeros_like(refine_latent)

                        refine_callback = latent_preview.prepare_callback(
                            model_patcher, max(0, int(refine_sigmas_device.shape[-1]) - 1), refine_x0_output
                        )
                        sampled_native = self._run_stock_sample_with_reused_lifecycle(
                            guider, device, refine_noise, refine_latent, sampler, refine_sigmas_device,
                            refine_mask, refine_callback, False, int(effective_seed),
                        )
                        if getattr(sampled_native, 'is_nested', False):
                            sampled_packed, _ = comfy.utils.pack_latents(sampled_native.unbind())
                            latent_shapes = [x.shape for x in sampled_native.unbind()]
                        else:
                            sampled_packed = sampled_native
                        _lm_print(
                            '[MiniMaxH3 LongMedia][HIRES SECOND PASS] '
                            f'unit={segment_index + 1}/{passes}; seed={int(effective_seed)}; '
                            f'steps={int(refine_steps_effective)}; fresh_noise={bool(latent_hires_enabled)}; '
                            f'sigma_indices={hires_sigma_indices if bool(latent_hires_enabled) else None}; '
                            f'mode={"independent_x0_hires" if bool(latent_hires_enabled) else "continuous_tail"}',
                            flush=True,
                        )

                    # Reconstruction Detail Recovery V3. The existing two-pass Ref2VA
                    # reconstruction remains authoritative. After sampler #2 we run two
                    # short independent trajectories: a broader structure-detail pass and
                    # a low-sigma microtexture pass. Their bounded detail bands are merged
                    # into the stable result while low-frequency video geometry and the
                    # complete audio latent are preserved exactly.
                    _detail_enabled = bool(
                        workflow_name == 'reconstruct'
                        and external_refine_global_mode
                        and bool(getattr(plan, 'reconstruction_detail_enabled', False))
                        and float(getattr(plan, 'reconstruction_detail_strength', 0.0) or 0.0) > 0.0
                    )
                    if _detail_enabled:
                        detail_strength = max(0.0, min(1.0, float(getattr(plan, 'reconstruction_detail_strength', 0.35))))
                        detail_steps = max(1, min(8, int(getattr(plan, 'reconstruction_detail_steps', 3) or 3)))
                        structure_steps = max(4, detail_steps)
                        detail_sigmas_cpu, detail_sigma_indices = _reconstruction_detail_sigmas(
                            full_sigmas, structure_steps, detail_strength
                        )
                        if detail_sigmas_cpu is not None and int(detail_sigmas_cpu.numel()) >= 2:
                            if not getattr(sampled_native, 'is_nested', False):
                                raise RuntimeError('Reconstruction Detail Recovery expected native H3 NestedTensor output.')
                            detail_base_streams = list(sampled_native.unbind())
                            if len(detail_base_streams) != 2:
                                raise RuntimeError(
                                    f'Reconstruction Detail Recovery expected 2 AV streams, got {len(detail_base_streams)}.'
                                )
                            detail_base_video = detail_base_streams[0]
                            detail_base_audio = detail_base_streams[1]

                            import comfy.sample
                            detail_noise_all = comfy.sample.prepare_noise(
                                sampled_native, int(effective_seed) ^ 0xD37A11
                            )
                            if getattr(detail_noise_all, 'is_nested', False):
                                detail_noise_streams = list(detail_noise_all.unbind())
                                detail_noise = comfy.nested_tensor.NestedTensor((
                                    detail_noise_streams[0],
                                    torch.zeros_like(detail_noise_streams[1]),
                                ))
                            else:
                                raise RuntimeError('Reconstruction Detail Recovery could not create native H3 nested noise.')

                            detail_sigmas_device = detail_sigmas_cpu.to(device)
                            detail_x0_output = {}
                            detail_callback = latent_preview.prepare_callback(
                                model_patcher, max(0, int(detail_sigmas_device.shape[-1]) - 1), detail_x0_output
                            )
                            _lm_print(
                                '[MiniMaxH3 LongMedia][RECON DETAIL LAYER V3] '
                                f'strength={detail_strength:.3f}; requested_steps={detail_steps}; structure_steps={structure_steps}; '
                                f'structure_sigma_indices={detail_sigma_indices}; fresh_video_noise=True; '
                                'audio_noise=False; low_frequency_lock=True; source=completed_global_refine; mode=dual_candidate',
                                flush=True,
                            )
                            structure_candidate = self._run_stock_sample_with_reused_lifecycle(
                                guider, device, detail_noise, sampled_native, sampler, detail_sigmas_device,
                                None, detail_callback, False, int(effective_seed) ^ 0xD37A11,
                            )
                            if not getattr(structure_candidate, 'is_nested', False):
                                raise RuntimeError('Reconstruction Detail Recovery structure pass returned non-native H3 output.')
                            structure_streams = list(structure_candidate.unbind())
                            if len(structure_streams) != 2:
                                raise RuntimeError(
                                    f'Reconstruction Detail Recovery structure pass returned {len(structure_streams)} streams, expected 2.'
                                )

                            micro_steps = max(3, min(4, detail_steps + 1))
                            micro_sigmas_cpu, micro_sigma_indices = _reconstruction_micro_detail_sigmas(
                                full_sigmas, micro_steps, detail_strength
                            )
                            texture_video = structure_streams[0]
                            if micro_sigmas_cpu is not None and int(micro_sigmas_cpu.numel()) >= 2:
                                micro_noise_all = comfy.sample.prepare_noise(
                                    sampled_native, int(effective_seed) ^ 0x9E3779B1
                                )
                                if not getattr(micro_noise_all, 'is_nested', False):
                                    raise RuntimeError('Reconstruction Detail Recovery micro pass could not create native H3 nested noise.')
                                micro_noise_streams = list(micro_noise_all.unbind())
                                micro_noise = comfy.nested_tensor.NestedTensor((
                                    micro_noise_streams[0],
                                    torch.zeros_like(micro_noise_streams[1]),
                                ))
                                micro_sigmas_device = micro_sigmas_cpu.to(device)
                                micro_x0_output = {}
                                micro_callback = latent_preview.prepare_callback(
                                    model_patcher, max(0, int(micro_sigmas_device.shape[-1]) - 1), micro_x0_output
                                )
                                _lm_print(
                                    '[MiniMaxH3 LongMedia][RECON DETAIL LAYER V3] '
                                    f'micro_steps={micro_steps}; micro_sigma_indices={micro_sigma_indices}; '
                                    'seed_mode=independent; target=microtexture',
                                    flush=True,
                                )
                                micro_candidate = self._run_stock_sample_with_reused_lifecycle(
                                    guider, device, micro_noise, sampled_native, sampler, micro_sigmas_device,
                                    None, micro_callback, False, int(effective_seed) ^ 0x9E3779B1,
                                )
                                if not getattr(micro_candidate, 'is_nested', False):
                                    raise RuntimeError('Reconstruction Detail Recovery micro pass returned non-native H3 output.')
                                micro_streams = list(micro_candidate.unbind())
                                if len(micro_streams) != 2:
                                    raise RuntimeError(
                                        f'Reconstruction Detail Recovery micro pass returned {len(micro_streams)} streams, expected 2.'
                                    )
                                texture_video = micro_streams[0]

                            merged_video = _reconstruction_merge_detail_residual(
                                detail_base_video, structure_streams[0], detail_strength, texture_video
                            )
                            # Audio is restored bit-for-bit from the completed two-pass
                            # reconstruction. Both detail candidates are strictly video-only.
                            sampled_native = comfy.nested_tensor.NestedTensor((
                                merged_video, detail_base_audio
                            ))
                            sampled_packed, _ = comfy.utils.pack_latents(sampled_native.unbind())
                            latent_shapes = [x.shape for x in sampled_native.unbind()]
                            _lm_print(
                                '[MiniMaxH3 LongMedia][RECON DETAIL LAYER] '
                                f'completed=True; video_shape={tuple(merged_video.shape)}; '
                                'audio_restored_exact=True; low_frequency_source=two_pass_reconstruction; residual_mode=dual_candidate_multiband_v3',
                                flush=True,
                            )

                    sampled_output = self._unpack_segment_output(local, sampled_packed, latent_shapes)

                    # A chained external refiner must never alter the already
                    # generated continuation head. Restore the native overlap
                    # bit-for-bit from sampler #1 after custom sigma refinement.
                    if external_refine_global_mode:
                        sampled_output.pop('noise_mask', None)
                        sampled_output['_lm_global_continuous_refine_output'] = True
                        _lm_print(
                            '[MiniMaxH3 LongMedia][GLOBAL REFINE OUTPUT] '
                            f'workflow={workflow_name}; one_continuous_latent=True; '
                            'segment_boundary_processing=False; post_blend=False',
                            flush=True,
                        )

                    # v0.4.55 continuity regression fix:
                    # With Latent Hi-Res disabled, preserve the exact pre-0.4.53
                    # MultiClip/segmented-continuation contract: the next clip must
                    # inherit the SAME final (including low-sigma refine) latent that
                    # is emitted/stored for the current clip.  Using the pre-refine
                    # base here makes the visible clip and its continuation anchor
                    # disagree in pose/exposure and produces boundary seams.
                    #
                    # With Latent Hi-Res enabled, the displayed result is a different
                    # spatial geometry, so continuation intentionally stays on the
                    # low-res branch.
                    previous_segment_continuation = (
                        base_continuation_output if bool(latent_hires_enabled) else sampled_output
                    )
                    previous_segment = sampled_output
                    if store_refine_segments:
                        stored_segment = self._cpu_latent_copy(sampled_output) if bool(offload_completed_segments) else sampled_output
                        stored_segment = dict(stored_segment)
                        stored_segment.pop('noise_mask', None)
                        per_clip_segment_latents.append(stored_segment)
                        visible_len = int(plan.segment_lengths[segment_index])
                        per_clip_segment_lengths.append(visible_len)
                        seg_video, _seg_audio = unpack_av_samples(sampled_output)
                        actual_len = int(frame_count_from_video_t(seg_video.shape[2]))
                        if int(segment_index) == 0:
                            per_clip_hidden_overlaps.append(0)
                        else:
                            per_clip_hidden_overlaps.append(int(getattr(plan, 'overlap_frames', 0) or 0))
                    if stitched is None:
                        stitched = sampled_output
                    elif store_per_clip_native_decode:
                        # MultiClip clips are independently sampled native H3 temporal grids.
                        # Never concatenate those video latents into one fake H3 timeline:
                        # e.g. T=37 + T=37 -> T=74 is invalid (H3 requires 5*k+2) and even
                        # padding such a concat causes 17-frame VAE phase pulses. Keep a
                        # valid placeholder latent here; final Decode consumes the stored
                        # per-clip native latents and assembles decoded RGB/audio instead.
                        pass
                    else:
                        stitch_overlap = int(plan.overlap_frames) if segmentation_active else 0
                        stitched = MiniMaxH3LatentLabStitchContinuation().stitch(
                            stitched, sampled_output,
                            stitch_overlap,
                            False, bool(offload_completed_segments)
                        )[0]
                    completed += 1

                if bool(getattr(plan, 'loop_closure_enabled', False)):
                    requested_loop_frames = max(2, int(getattr(plan, 'loop_closure_frames', 0) or 0))
                    loop_strength = max(0.0, min(1.0, float(getattr(plan, 'loop_closure_strength', 0.65))))
                    closure_seed = (int(seed) ^ 0x1C005E) & 0xFFFFFFFFFFFFFFFF
                    if loop_strength <= 0.0:
                        loop_closure_report = {
                            'enabled': True,
                            'applied': False,
                            'mode': 'disabled_by_strength',
                            'requested_frames': int(requested_loop_frames),
                            'loop_strength': 0.0,
                        }
                    base_positive = None
                    segment_conds = getattr(plan, 'segment_positive_conditionings', None) if loop_strength > 0.0 else None
                    if isinstance(segment_conds, (list, tuple)) and segment_conds:
                        base_positive = segment_conds[min(max(0, completed - 1), len(segment_conds) - 1)]
                    if base_positive is None and loop_strength > 0.0:
                        base_positive = (getattr(runtime_template_guider, 'original_conds', {}) or {}).get('positive')
                    if loop_strength <= 0.0:
                        pass
                    elif base_positive is None:
                        loop_closure_report = {
                            'enabled': True,
                            'applied': False,
                            'mode': 'unavailable',
                            'reason': 'missing_positive_conditioning',
                        }
                    else:
                        target_av = None
                        target_label = 'global'
                        if bool(store_per_clip_native_decode) and per_clip_segment_latents:
                            target_av = per_clip_segment_latents[-1]
                            target_label = 'last_segment'
                        elif stitched is not None:
                            target_av = stitched
                        head_av = per_clip_segment_latents[0] if per_clip_segment_latents else stitched
                        if target_av is None or head_av is None:
                            loop_closure_report = {
                                'enabled': True,
                                'applied': False,
                                'mode': 'unavailable',
                                'reason': 'missing_latent_source',
                            }
                        else:
                            head_video, _head_audio = unpack_av_samples(head_av)
                            target_video, target_audio = unpack_av_samples(target_av)
                            target_frames_total = int(frame_count_from_video_t(target_video.shape[2]))
                            # Largest valid H3 frame count that does not exceed the available target length.
                            max_closure_frames = max(5, 5 + 17 * max(0, (target_frames_total - 5) // 17))
                            actual_closure_frames = int(_nearest_valid_h3_frame_count(max(5, requested_loop_frames)))
                            if actual_closure_frames > max_closure_frames:
                                actual_closure_frames = max_closure_frames
                            if actual_closure_frames >= 5 and target_frames_total > 5:
                                closure_video_t = int(video_latent_t(actual_closure_frames))
                                closure_audio_t = int(audio_latent_t(actual_closure_frames))
                                closure_video = target_video[:, :, -closure_video_t:].clone().to(device=device)
                                closure_audio = target_audio[..., -closure_audio_t:].clone().to(device=device)
                                head_anchor = head_video[:, :, :1].clone().to(device=device)
                                tail_anchor = closure_video[:, :, :1].clone()
                                tail_end = closure_video[:, :, -1:].clone()
                                # H3 receives only a light structural hint.  The actual loop
                                # closure is enforced after sampling as a low-frequency macro
                                # return, so the model is not forced to accelerate motion just
                                # to hit an exact terminal state.
                                model_loop_strength = min(0.35, loop_strength * 0.35)
                                structural_anchor = _loop_structural_anchor(
                                    head_anchor, tail_end, model_loop_strength
                                )

                                closure_video_mask = torch.ones(
                                    (1, 1, int(closure_video.shape[2]), 1, 1),
                                    dtype=torch.float32, device=device,
                                )
                                closure_audio_mask = torch.zeros(
                                    (1, 1, 1, int(closure_audio.shape[-1])),
                                    dtype=torch.float32, device=device,
                                )
                                closure_av = {
                                    'samples': NestedTensor((closure_video, closure_audio)),
                                    'noise_mask': NestedTensor((closure_video_mask, closure_audio_mask)),
                                }
                                closure_av = inject_leading_video_frame(
                                    closure_av,
                                    {'samples': tail_anchor},
                                    0.0,
                                    NestedTensor,
                                )
                                loop_positive, loop_has_refs = _clone_positive_with_loop_keyframes(
                                    base_positive, tail_anchor, structural_anchor, actual_closure_frames
                                )

                                guider.original_conds = dict(getattr(runtime_template_guider, 'original_conds', {}) or {})
                                guider.original_conds['positive'] = loop_positive
                                guider.conds = self._copy_conds(guider.original_conds)
                                guider.model_options = comfy.model_patcher.create_model_options_clone(
                                    getattr(runtime_template_guider, 'model_options', original_model_options)
                                )
                                guider.model_options.setdefault('transformer_options', {}).pop(TEMPORAL_OFFSET_OPTION, None)
                                if runtime_thread_pool is not None:
                                    guider.model_options['multigpu_thread_pool'] = runtime_thread_pool

                                closure_sigmas_cpu, closure_sigma_indices = _loop_closure_sigmas(full_sigmas, 4, loop_strength)
                                if closure_sigmas_cpu is None or int(closure_sigmas_cpu.numel()) < 2:
                                    loop_closure_report = {
                                        'enabled': True,
                                        'applied': False,
                                        'mode': 'unavailable',
                                        'reason': 'invalid_sigma_schedule',
                                    }
                                else:
                                    import comfy.sample
                                    closure_noise_all = comfy.sample.prepare_noise(
                                        closure_av['samples'], int(closure_seed)
                                    )
                                    if not getattr(closure_noise_all, 'is_nested', False):
                                        raise RuntimeError('Native loop closure could not create a MiniMax H3 nested noise tensor.')
                                    closure_noise_streams = list(closure_noise_all.unbind())
                                    if len(closure_noise_streams) != 2:
                                        raise RuntimeError(
                                            f'Native loop closure expected 2 AV noise streams, got {len(closure_noise_streams)}.'
                                        )
                                    closure_noise = comfy.nested_tensor.NestedTensor((
                                        closure_noise_streams[0].to(device=device, dtype=torch.float32),
                                        torch.zeros_like(closure_noise_streams[1], device=device, dtype=torch.float32),
                                    ))
                                    closure_sigmas = closure_sigmas_cpu.to(device)
                                    closure_x0_output = {}
                                    closure_callback = latent_preview.prepare_callback(
                                        model_patcher, max(0, int(closure_sigmas.shape[-1]) - 1), closure_x0_output
                                    )
                                    _lm_print(
                                        '[MiniMaxH3 LongMedia][NATIVE LOOP CLOSURE] '
                                        f'target={target_label}; requested={requested_loop_frames}f actual={actual_closure_frames}f; '
                                        f'sigma_indices={closure_sigma_indices}; keep_audio_exact=True; keep_tail_start_exact=True; '
                                        f'loop_strength={loop_strength:.2f}; return_anchor=opening_macro_context; refs_preserved={bool(loop_has_refs)}',
                                        flush=True,
                                    )
                                    closure_out = self._run_stock_sample_with_reused_lifecycle(
                                        guider, device, closure_noise, closure_av['samples'], sampler, closure_sigmas,
                                        closure_av.get('noise_mask'), closure_callback, False, int(closure_seed),
                                    )
                                    if not getattr(closure_out, 'is_nested', False):
                                        raise RuntimeError('Native loop closure expected a MiniMax H3 NestedTensor output.')
                                    out_video, _out_audio = list(closure_out.unbind())
                                    out_video, macro_return_effective = _apply_loop_macro_return(
                                        out_video, head_anchor, loop_strength
                                    )
                                    merged_video = target_video.clone()
                                    merged_audio = target_audio.clone()
                                    merged_video[:, :, -closure_video_t:] = out_video.to(
                                        device=merged_video.device, dtype=merged_video.dtype
                                    )
                                    # Audio continuity is already correct; keep it bit-for-bit.
                                    merged_audio[..., -closure_audio_t:] = target_audio[..., -closure_audio_t:]
                                    merged_target = dict(target_av)
                                    merged_target['samples'] = NestedTensor((merged_video, merged_audio))
                                    merged_target.pop('noise_mask', None)
                                    if bool(store_per_clip_native_decode) and per_clip_segment_latents:
                                        per_clip_segment_latents[-1] = merged_target
                                    else:
                                        stitched = merged_target
                                    loop_closure_report = {
                                        'enabled': True,
                                        'applied': True,
                                        'mode': 'native_tail_regeneration_macro_return',
                                        'target': target_label,
                                        'requested_frames': int(requested_loop_frames),
                                        'actual_frames': int(actual_closure_frames),
                                        'sigma_indices': list(closure_sigma_indices),
                                        'seed': int(closure_seed),
                                        'preserved_audio_exact': True,
                                        'preserved_tail_start_exact': True,
                                        'return_anchor': 'opening_macro_context',
                                        'loop_strength': float(loop_strength),
                                        'model_guidance_strength': float(model_loop_strength),
                                        'macro_return_effective_strength': float(macro_return_effective),
                                        'detail_policy': 'tail_microdetail_free_macro_only_return',
                                        'refs_preserved': bool(loop_has_refs),
                                    }
                            else:
                                loop_closure_report = {
                                    'enabled': True,
                                    'applied': False,
                                    'mode': 'skipped',
                                    'reason': 'target_too_short',
                                    'requested_frames': int(requested_loop_frames),
                                    'available_frames': int(target_frames_total),
                                }

                _lm_print(
                    '[MiniMaxH3 LongMedia][UNIFIED RUNTIME] '
                    f'completed_segments={completed}; one_prepare_sampling=True; one_pre_run=True; one_cleanup=True',
                    flush=True,
                )
        finally:
            try:
                if runtime_thread_pool is not None:
                    runtime_thread_pool.shutdown()
                    runtime_thread_pool = None
            except Exception:
                pass
            try:
                model_patcher.cleanup()
                for patcher in (multigpu_patchers or []):
                    patcher.cleanup()
            finally:
                if loaded_models is not None:
                    try:
                        comfy.sampler_helpers.cleanup_models(guider.conds, loaded_models)
                    except Exception:
                        pass
                model_patcher.restore_hook_patches()
                model_patcher.hook_mode = original_hook_mode
                guider.model_options = original_model_options
                guider.original_conds = original_original_conds
                guider.inner_model = original_inner_model
                guider.conds = original_conds_runtime
                guider.loaded_models = original_loaded_models

        if stitched is None:
            raise RuntimeError('Unified LongMedia runtime produced no segment output.')

        if external_refine_global_mode:
            stitched = dict(stitched)
            stitched.pop('noise_mask', None)
            stitched.pop('_lm_per_clip_native_video_decode', None)
            stitched.pop('_lm_segment_latents', None)
            stitched.pop('_lm_segment_lengths', None)
            stitched.pop('_lm_segment_hidden_overlaps', None)
            stitched.pop('_lm_segment_workflow', None)
            stitched.pop('_lm_external_refine_ready', None)
            stitched['_lm_global_continuous_refine_output'] = True
        if store_refine_segments and per_clip_segment_latents:
            stitched = stitched.copy()
            if store_per_clip_native_decode:
                stitched['_lm_per_clip_native_video_decode'] = True
            else:
                stitched.pop('_lm_per_clip_native_video_decode', None)
            stitched['_lm_segment_latents'] = per_clip_segment_latents
            stitched['_lm_segment_lengths'] = list(per_clip_segment_lengths)
            stitched['_lm_segment_hidden_overlaps'] = list(per_clip_hidden_overlaps)
            stitched['_lm_segment_workflow'] = str(getattr(plan, 'workflow_mode', '') or '')
            stitched['_lm_external_refine_ready'] = True
        stitched['_lm_loop_closure_report'] = dict(loop_closure_report or {})
        return (stitched, json.dumps({
            'runtime': 'unified_single_model_lifecycle',
            'passes': passes,
            'completed_segments': completed,
            'prepare_sampling_calls': 1,
            'pre_run_calls': 1,
            'cleanup_calls': 1,
            'sampler_execution_boundary': sampler_execution_boundary,
            'guider_sample_calls': 0,
            'model_reload_between_segments': False,
            'refine_enabled': bool(refine_enabled),
            'refine_steps_effective': int(refine_steps_effective),
            'refine_switch_step': int(refine_switch_step),
            'refine_model_reload': False,
            'external_refine_handoff': bool(external_refine_mode),
            'external_refine_global_continuous': bool(external_refine_global_mode),
            'external_refine_source_segments': (int(original_segment_passes) if external_refine_global_mode else None),
            'loop_closure': loop_closure_report,
            'external_refine_runtime_passes': int(passes),
            'external_refine_fresh_noise': True if external_refine_mode else None,
            'external_refine_noise_contract': ('ordinary_prepare_noise' if external_refine_mode else None),
            'external_refine_workflow': (workflow_name if external_refine_mode else None),
            'external_refine_duplicate_previous_av': False if external_refine_mode else None,
            'external_refine_geometry_guard': bool(external_refine_mode),
            'external_refine_per_clip_native': False if external_refine_global_mode else bool(external_refine_mode),
            'reconstruction_detail_enabled': bool(
                workflow_name == 'reconstruct'
                and bool(getattr(plan, 'reconstruction_detail_enabled', False))
            ),
            'reconstruction_detail_strength': (
                float(getattr(plan, 'reconstruction_detail_strength', 0.0) or 0.0)
                if workflow_name == 'reconstruct' else None
            ),
            'reconstruction_detail_steps': (
                int(getattr(plan, 'reconstruction_detail_steps', 0) or 0)
                if workflow_name == 'reconstruct' else None
            ),
            'reconstruction_detail_execution': (
                'post_global_refine_dual_candidate_multiband_detail_v3'
                if workflow_name == 'reconstruct' and external_refine_global_mode
                and bool(getattr(plan, 'reconstruction_detail_enabled', False))
                else None
            ),
            'external_refine_seam_restore': False,
        }))


class MiniMaxH3LatentLabLongMediaSampler:
    DESCRIPTION = 'Expand a long-media plan into a sequential multi-pass sampler graph. When manual_duration exceeds segment_seconds, segments are sampled one after another, context is carried across segment boundaries, and the result is stitched into one final AV latent.'
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        try:
            from .latent_hires import scan_models as _scan_hires_models
        except Exception:
            _scan_hires_models = lambda: ['(place model in models/latent_upscale_models)']
        _scanned_hires_models = [
            str(x) for x in _scan_hires_models()
            if str(x).strip() and not str(x).startswith('(')
        ]
        _hires_models = ['(disabled)'] + _scanned_hires_models
        return {
            'required': {
                'initial_av': ('LATENT',),
                'long_media_plan': ('LONG_MEDIA_PLAN',),
                'guider': ('GUIDER',),
                'sampler': ('SAMPLER',),
                'sigmas': ('SIGMAS',),
                'seed': (
                    'INT',
                    {'default': 0, 'min': 0, 'max': 18446744073709551615},
                ),
                'video_context_denoise': (
                    'FLOAT',
                    {
                        'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                        'tooltip': '0 preserves each inherited overlap exactly; 1 fully denoises it.',
                    },
                ),
                'audio_context_denoise': (
                    'FLOAT',
                    {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'offload_completed_segments': (
                    'BOOLEAN',
                    {
                        'default': True,
                        'tooltip': (
                            'Move each pass\'s stitched result to CPU RAM once it has '
                            'been folded in, instead of leaving the whole growing clip '
                            'resident on the GPU for the rest of the run. Only the '
                            'accumulator moves — the small per-pass sampling context '
                            'stays on the GPU as before — so this has no effect on '
                            'output, only on peak VRAM during long multi-pass runs. '
                            'Turn off only to restore the previous (all-GPU) behavior.'
                        ),
                    },
                ),
                'mlp_chunk_tokens': (
                    'INT',
                    {
                        'default': 8192, 'min': 0, 'max': 131072, 'step': 512,
                        'tooltip': (
                            'Token chunk size for the low-VRAM H3 MLP path. Manual mode uses 512-token increments so low-VRAM users can select 4096/3072/2048/1536/1024/512. '
                            '8192 is the current safe default. Larger values are faster '
                            'but use more VRAM. Set 0 to effectively disable MLP '
                            'chunking for A/B testing.'
                        ),
                    },
                ),
                'attention_mode': (
                    ['auto', 'existing', 'sol', 'scheduled_sol'],
                    {'default': 'auto', 'tooltip': 'auto selects existing/Sage for smaller sequences and embedded Sol for large sequences without changing H3 tokens. existing forces current Sage/Comfy attention. sol/scheduled_sol force the embedded Apache-2.0 SM120 Sol path.'},
                ),
                'sol_tau_start': ('FLOAT', {'default': 1.30, 'min': 0.0, 'max': 4.0, 'step': 0.05}),
                'sol_tau_end': ('FLOAT', {'default': 0.80, 'min': 0.0, 'max': 4.0, 'step': 0.05}),
                'sol_curve': (['linear', 'cosine', 'sqrt', 'smoothstep', 'exponential', 'step'], {'default': 'linear'}),
                'sol_min_tokens': ('INT', {'default': 4096, 'min': 256, 'max': 131072, 'step': 256}),
                'sol_dense_percent': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 0.9, 'step': 0.05}),
                'sol_sink_conditioning': (['exact_kv', 'exact_kv_and_rows', 'off'], {'default': 'exact_kv'}),
                'sol_qkv_chunk_tokens': (
                    'INT',
                    {
                        'default': 8192, 'min': 0, 'max': 131072, 'step': 512,
                        'tooltip': (
                            'Stream H3 QKV projection in token chunks. In streamed mode token-level '
                            'K/V are retained as INT8+scale while Sol block summaries stay BF16; Q is '
                            'reprojected and consumed chunk-by-chunk. This targets very long single-pass '
                            'clips on limited VRAM. Manual mode uses 512-token increments so 4096/3072/2048/1536/1024/512 are selectable. 0 restores the full fused-QKV path.'
                        ),
                    },
                ),
                'sol_out_proj_chunk_tokens': (
                    'INT',
                    {
                        'default': 24576, 'min': 0, 'max': 131072, 'step': 512,
                        'tooltip': (
                            'Token chunk size for the embedded Sol output projection. '
                            'Smaller values reduce peak VRAM; larger values are faster. Manual mode uses 512-token increments for fine low-VRAM tuning. '
                            '0 disables out_proj chunking.'
                        ),
                    },
                ),
                'vram_activation_reserve_mb': (
                    'INT',
                    {
                        'default': 4096, 'min': 0, 'max': 12288, 'step': 256,
                        'tooltip': (
                            'Extra VRAM headroom requested from ComfyUI before model loading. '
                            'ComfyUI will keep fewer H3 weights resident and offload more to RAM, '
                            'leaving this space for long-sequence activations. 0 disables the extra reserve.'
                        ),
                    },
                ),
                'inter_block_vram_guard_mb': (
                    'INT',
                    {
                        'default': 2048, 'min': 0, 'max': 8192, 'step': 128,
                        'tooltip': (
                            'Minimum driver-free VRAM target between H3 transformer blocks. '
                            'When free VRAM falls below this value and PyTorch is holding >=256 MB '
                            'of dead reserved cache, LongMedia returns that cache to the driver. '
                            '0 disables inter-block trimming.'
                        ),
                    },
                ),
                'inter_block_guard_cooldown_blocks': (
                    'INT',
                    {
                        'default': 4, 'min': 0, 'max': 32, 'step': 1,
                        'tooltip': (
                            'Completed H3 blocks to wait between normal cache trims. '
                            'Emergency pressure bypasses this cooldown. 0 restores the 0.2.36 behavior.'
                        ),
                    },
                ),
                'inter_block_guard_emergency_mb': (
                    'INT',
                    {
                        'default': 512, 'min': 0, 'max': 4096, 'step': 128,
                        'tooltip': (
                            'Emergency driver-free VRAM threshold. Below this value the emergency guard '
                            'may trim even while the normal guard is cooling down. 0 disables emergency mode.'
                        ),
                    },
                ),
                'inter_block_guard_emergency_cooldown_blocks': (
                    'INT',
                    {
                        'default': 3, 'min': 0, 'max': 32, 'step': 1,
                        'tooltip': (
                            'Minimum completed H3 blocks between EMERGENCY cache trims. '
                            'This prevents Dynamic VRAM/AIMDO free==0 states from causing a trim storm. '
                            '0 restores the 0.2.37 immediate-emergency behavior.'
                        ),
                    },
                ),
                'late_block_guard_start': (
                    'INT',
                    {
                        'default': 40, 'min': 0, 'max': 127, 'step': 1,
                        'tooltip': 'First H3 transformer block where the late hard guard is allowed to run. 40 targets only the tail of the 50-block H3 stack.',
                    },
                ),
                'late_block_guard_target_mb': (
                    'INT',
                    {
                        'default': 6144, 'min': 0, 'max': 12288, 'step': 256,
                        'tooltip': 'Driver-free VRAM target before attention/FFN in late H3 blocks. 0 disables the late-block hard guard.',
                    },
                ),
                'late_block_guard_min_cached_mb': (
                    'INT',
                    {
                        'default': 512, 'min': 0, 'max': 4096, 'step': 128,
                        'tooltip': 'Minimum reclaimable PyTorch CUDA cache required before a late-block hard trim is attempted.',
                    },
                ),
                'step_boundary_cleanup_mb': (
                    'INT',
                    {
                        'default': 2048, 'min': 0, 'max': 8192, 'step': 128,
                        'tooltip': 'Minimum driver-free VRAM target after each completed denoise step. Dead allocator cache is returned before the next H3 forward. 0 disables.',
                    },
                ),
                'latent_hires_enabled': (
                    'BOOLEAN',
                    {'default': False, 'tooltip': 'Learned H3 latent hi-res stage between base sampling and refine. Video latent only; audio is preserved exactly.'},
                ),
                'latent_hires_model': (
                    _hires_models,
                    {'default': _hires_models[0], 'tooltip': 'Checkpoint from ComfyUI/models/latent_upscale_models.'},
                ),
                'latent_hires_scale': (
                    'FLOAT',
                    {'default': 2.0, 'min': 1.0, 'max': 4.0, 'step': 0.1, 'tooltip': 'Spatial latent upscale multiplier. Model supports continuous 1.0x-4.0x.'},
                ),
                'latent_hires_precision': (
                    ['fp16', 'bf16', 'fp32'],
                    {'default': 'fp16', 'tooltip': 'Upscaler inference precision. fp16 is the practical default.'},
                ),
                'latent_hires_align': (
                    'INT',
                    {'default': 32, 'min': 16, 'max': 256, 'step': 16, 'tooltip': 'Output pixel alignment. 32 is recommended by the upstream model to avoid edge/light-band artifacts.'},
                ),
                'refine_enabled': (
                    'BOOLEAN',
                    {'default': True, 'tooltip': 'Split the connected SIGMAS schedule into the main pass plus the final low-noise refine tail. Recommended production default: ON with 2 refine steps.'},
                ),
                'refine_add_noise': (
                    'BOOLEAN',
                    {'default': False, 'tooltip': 'Legacy compatibility input. Ignored: a true refine stage always continues with no fresh noise.'},
                ),
                'refine_seed': (
                    'INT',
                    {'default': 0, 'min': 0, 'max': 18446744073709551615, 'tooltip': 'Legacy compatibility input. Ignored: refine continues the same trajectory and does not generate fresh noise.'},
                ),
                'refine_steps': (
                    'INT',
                    {
                        'default': 2, 'min': 1, 'max': 1000, 'step': 1,
                        'tooltip': (
                            'How many extra low-noise steps to run after the complete base sampler. '
                            'Without Latent Hi-Res: stage2 runs the final low-sigma tail. With Latent Hi-Res: stage1 stops early, the denoised x0 is learned-upscaled, then refine_steps runs as an independent same-seed fresh-noise hi-res pass.'
                        ),
                    },
                ),
                'memory_mode': (
                    ['auto', 'normal', 'low_vram', 'ultra_low_vram'],
                    {'default': 'auto', 'tooltip': 'Sampler-local residency policy. auto selects from model-size/VRAM ratio; low_vram and ultra_low_vram work without ComfyUI launch flags.'},
                ),
                'sampler_mode': (
                    ['auto', 'manual'],
                    {
                        'default': 'auto',
                        'tooltip': 'auto uses the validated production attention/VRAM policy. manual exposes all low-level tuning widgets.',
                    },
                ),
            }
        }

    RETURN_TYPES = ('LATENT', 'INT', 'INT', 'INT', 'STRING')
    RETURN_NAMES = ('final_av', 'total_frames', 'trim_frames', 'passes', 'report')
    FUNCTION = 'sample'
    CATEGORY = CATEGORY_LONGMEDIA

    def sample(self, initial_av, long_media_plan, guider, sampler, sigmas, seed,
               video_context_denoise=0.0, audio_context_denoise=0.0,
               offload_completed_segments=True, mlp_chunk_tokens=8192,
               attention_mode='auto', sol_tau_start=1.3, sol_tau_end=0.8,
               sol_curve='linear', sol_min_tokens=4096, sol_dense_percent=0.0,
               sol_sink_conditioning='exact_kv', sol_qkv_chunk_tokens=8192, sol_out_proj_chunk_tokens=24576,
               vram_activation_reserve_mb=4096, inter_block_vram_guard_mb=2048,
               inter_block_guard_cooldown_blocks=4, inter_block_guard_emergency_mb=512, inter_block_guard_emergency_cooldown_blocks=3,
               late_block_guard_start=40, late_block_guard_target_mb=6144, late_block_guard_min_cached_mb=512,
               step_boundary_cleanup_mb=2048,
               latent_hires_enabled=False, latent_hires_model='', latent_hires_scale=2.0,
               latent_hires_precision='fp16', latent_hires_align=32,
               refine_enabled=False, refine_add_noise=False, refine_seed=0,
               refine_steps=2,
               memory_mode='auto',
               sampler_mode='auto'):
        from comfy_execution.graph_utils import GraphBuilder

        plan = long_media_plan
        _set_longmedia_release_guard(bool(getattr(plan, 'release_guard', True)))
        if not bool(getattr(plan, 'release_guard', True)):
            builtins.print(
                '[MiniMaxH3 LongMedia] sampler verbose diagnostics active',
                flush=True,
            )
        graph = GraphBuilder()

        # v0.4.59: runtime-side repair for a known positional serialization
        # corruption introduced while decorative UI section widgets were being
        # serialized into LiteGraph widgets_values.  The fingerprint is deliberately
        # strict so legitimate manual tuning is never rewritten.
        _v459_corrupt = (
            abs(float(sol_tau_start) - 4.0) < 1e-6
            and str(sol_curve) == 'exponential'
            and int(sol_min_tokens) == 256
            and int(sol_qkv_chunk_tokens) == 0
            and int(vram_activation_reserve_mb) == 512
            and int(inter_block_vram_guard_mb) == 8192
            and int(inter_block_guard_emergency_mb) == 4096
            and int(inter_block_guard_emergency_cooldown_blocks) == 32
            and int(late_block_guard_start) == 4
            and int(late_block_guard_target_mb) == 4096
            and int(late_block_guard_min_cached_mb) == 32
            and int(step_boundary_cleanup_mb) == 5
        )
        if _v459_corrupt:
            sol_tau_start = 1.3
            sol_tau_end = 0.8
            sol_curve = 'linear'
            sol_min_tokens = 4096
            sol_dense_percent = 0.0
            sol_sink_conditioning = 'exact_kv'
            sol_qkv_chunk_tokens = 8192
            sol_out_proj_chunk_tokens = 24576

            vram_activation_reserve_mb = 4096
            inter_block_vram_guard_mb = 2048
            inter_block_guard_cooldown_blocks = 4
            inter_block_guard_emergency_mb = 512
            inter_block_guard_emergency_cooldown_blocks = 3
            late_block_guard_start = 40
            late_block_guard_target_mb = 6144
            late_block_guard_min_cached_mb = 512
            step_boundary_cleanup_mb = 2048
            _lm_print(
                '[MiniMaxH3 LongMedia] repaired legacy sampler widget serialization state',
                flush=True,
            )

        sampler_mode = str(sampler_mode or 'auto')
        requested_memory_mode = str(memory_mode or 'auto')
        memory_profile = _resolve_h3_memory_mode(guider, requested_memory_mode)
        memory_mode = str(memory_profile['effective'])
        # v0.3.60: every mode uses the same adaptive governor.  Modes differ
        # only by safety envelope; no mode is allowed to blindly run into OOM.
        # Fixed 8GB ultra reserves left several GB of a 16GB card idle, so reserve
        # is now a smaller planning margin and runtime driver-free floors own safety.
        _ms, _gs = memory_profile.get('model_bytes'), memory_profile.get('gpu_bytes')
        _ratio = (float(_ms) / float(_gs)) if (_ms and _gs) else 0.0
        try:
            import psutil as _psutil
            _vm = _psutil.virtual_memory()
            _ram_avail_gb = float(_vm.available) / (1024.0 ** 3)
        except Exception:
            _ram_avail_gb = 0.0
        offload_completed_segments = True if memory_mode in ('low_vram','ultra_low_vram') or _ratio > 1.0 else bool(offload_completed_segments)
        if memory_mode == 'normal':
            vram_activation_reserve_mb = max(256, min(int(vram_activation_reserve_mb), 768))
            inter_block_vram_guard_mb = max(768, min(int(inter_block_vram_guard_mb), 1280))
            late_block_guard_target_mb = max(1792, min(int(late_block_guard_target_mb), 3072))
            step_boundary_cleanup_mb = max(1024, min(int(step_boundary_cleanup_mb), 1792))
            mlp_chunk_tokens = min(max(int(mlp_chunk_tokens), 512), 8192)
        elif memory_mode == 'low_vram':
            vram_activation_reserve_mb = max(768, min(int(vram_activation_reserve_mb), 1536))
            inter_block_vram_guard_mb = max(1280, min(int(inter_block_vram_guard_mb), 2048))
            late_block_guard_target_mb = max(2560, min(int(late_block_guard_target_mb), 4096))
            step_boundary_cleanup_mb = max(1536, min(int(step_boundary_cleanup_mb), 2560))
            mlp_chunk_tokens = min(max(int(mlp_chunk_tokens), 256), 4096)
        else:  # ultra_low_vram
            vram_activation_reserve_mb = max(1792, min(int(vram_activation_reserve_mb), 2816))
            inter_block_vram_guard_mb = max(2048, min(int(inter_block_vram_guard_mb), 3072))
            late_block_guard_target_mb = max(3328, min(int(late_block_guard_target_mb), 4608))
            step_boundary_cleanup_mb = max(2048, min(int(step_boundary_cleanup_mb), 3072))
            inter_block_guard_cooldown_blocks = min(max(int(inter_block_guard_cooldown_blocks), 1), 3)
            mlp_chunk_tokens = min(max(int(mlp_chunk_tokens), 128), 2048)

        _lm_print('[MiniMaxH3 LongMedia][MEMORY POLICY V3] '
            f"requested={memory_profile['requested']} effective={memory_mode}; model={(float(_ms)/(1024**3)) if _ms else 0.0:.1f}GB GPU={(float(_gs)/(1024**3)) if _gs else 0.0:.1f}GB; "
            f"reason={memory_profile['reason']}; MLP={int(mlp_chunk_tokens)} QKV={int(sol_qkv_chunk_tokens)} OUT={int(sol_out_proj_chunk_tokens)} reserve={int(vram_activation_reserve_mb)}MB", flush=True)
        requested_attention_mode = str(attention_mode or 'auto')
        # v0.4.33 keeps AUTO alive for segmented jobs.  v0.4.32 forced every
        # segmented AUTO run to `existing`, which in turn forced the external SLA
        # compatibility path and could make each denoise step several minutes.
        # AUTO is now resolved inside the first H3 block and latched in runtime
        # state, so all later passes keep the same attention family without losing
        # the large-geometry embedded-Sol safety/performance route.
        effective_attention_mode = requested_attention_mode
        attention_mode = effective_attention_mode
        if int(getattr(plan, 'passes', 1)) > 1 and requested_attention_mode == 'auto':
            _lm_print(
                '[MiniMaxH3 LongMedia][ATTENTION CONTINUITY] '
                'segmented AUTO remains enabled; first resolved attention family is latched across passes',
                flush=True,
            )
        if sampler_mode == 'auto':
            # v0.3.22 A/B override mode: INPUT_TYPES defaults remain the validated
            # production AUTO policy, but explicit widget edits are honored. This
            # lets AUTO routing be compared against forced existing/SOL without
            # switching to Manual and changing any other sampler state.
            _lm_print(
                '[MiniMaxH3 LongMedia][V322 AUTO OVERRIDES] production defaults active; '
                f'attention_mode={requested_attention_mode}->{effective_attention_mode}, '
                f'tau={float(sol_tau_start):.3f}->{float(sol_tau_end):.3f}, '
                f'mlp_chunk={int(mlp_chunk_tokens)}',
                flush=True,
            )
        requested_mlp_chunk_tokens = int(mlp_chunk_tokens)
        effective_mlp_chunk_tokens = requested_mlp_chunk_tokens if requested_mlp_chunk_tokens > 0 else (1 << 30)
        mlp_chunking_enabled = requested_mlp_chunk_tokens > 0
        try:
            _sig = sigmas.detach().float().cpu() if torch.is_tensor(sigmas) else torch.as_tensor(sigmas, dtype=torch.float32)
            sol_sigma_hi = float(_sig[0]) if _sig.numel() else 1.0
            _nonzero = _sig[_sig > 0]
            sol_sigma_lo = float(_nonzero[-1]) if _nonzero.numel() else 0.0
        except Exception:
            sol_sigma_hi, sol_sigma_lo = 1.0, 0.0
        mlp_chunker = graph.node(
            "MiniMaxH3LatentLabMLPChunking",
            guider=guider,
            chunk_tokens=effective_mlp_chunk_tokens,
            max_blocks=128,
            sol_mode=str(attention_mode),
            sol_tau_start=float(sol_tau_start),
            sol_tau_end=float(sol_tau_end),
            sol_curve=str(sol_curve),
            sol_min_tokens=int(sol_min_tokens),
            sol_dense_percent=float(sol_dense_percent),
            sol_sink_conditioning=str(sol_sink_conditioning),
            sol_qkv_chunk_tokens=int(sol_qkv_chunk_tokens),
            sol_out_proj_chunk_tokens=int(sol_out_proj_chunk_tokens),
            vram_activation_reserve_mb=int(vram_activation_reserve_mb),
            inter_block_vram_guard_mb=int(inter_block_vram_guard_mb),
            inter_block_guard_cooldown_blocks=int(inter_block_guard_cooldown_blocks),
            inter_block_guard_emergency_mb=int(inter_block_guard_emergency_mb),
            inter_block_guard_emergency_cooldown_blocks=int(inter_block_guard_emergency_cooldown_blocks),
            late_block_guard_start=int(late_block_guard_start),
            late_block_guard_target_mb=int(late_block_guard_target_mb),
            late_block_guard_min_cached_mb=int(late_block_guard_min_cached_mb),
            step_boundary_cleanup_mb=int(step_boundary_cleanup_mb),
            sol_sigma_hi=float(sol_sigma_hi),
            sol_sigma_lo=float(sol_sigma_lo),
            memory_mode=str(memory_mode),
            requested_memory_mode=str(requested_memory_mode),
        )
        traced_guider = mlp_chunker.out(0)
        ultra_pin_previous = None
        _out_of_core = bool(_ms and _gs and float(_ms) > float(_gs) * 1.05)
        if _out_of_core:
            ultra_pin_gate = graph.node(
                "MiniMaxH3LatentLabUltraPinnedMemoryGate",
                guider=traced_guider,
                enable=True,
            )
            traced_guider = ultra_pin_gate.out(0)
            ultra_pin_previous = ultra_pin_gate.out(1)
        block_trace_state = mlp_chunker.out(1)
        memory_profiler = graph.node(
            "MiniMaxH3LatentLabFirstStepMemoryProfiler",
            sampler=sampler,
            max_history_entries=20000,
            block_trace_state=block_trace_state,
        )
        profiled_sampler = memory_profiler.out(0)
        memory_profile_state = memory_profiler.out(1)
        # 0.2.16 diagnostic build deliberately disables intra-step cache flushing:
        # we want an undistorted first-step allocator trace, including OOM events.
        guard_state = None
        # 0.4.7 refiner: true two-stage Advanced split restored inside the unified runtime.
        # The model lifecycle stays open across both stages; only inner_sample is
        # invoked for the base and low-noise tail phases.
        main_sigmas = sigmas
        refine_sigmas = None
        refine_steps_effective = 0
        refine_tail_start = 0
        base_steps = max(0, int(sigmas.numel()) - 1) if torch.is_tensor(sigmas) else max(0, len(sigmas) - 1)
        if bool(refine_enabled):
            main_sigmas, refine_sigmas, base_steps, refine_tail_start, refine_steps_effective, _ = split_refine_sigmas(
                sigmas, int(refine_steps)
            )
            _lm_print(
                '[MiniMaxH3 LongMedia][REFINER RESTORED] '
                f'total_steps={int(base_steps)}; main_steps={int(refine_tail_start)}; '
                f'refine_steps={int(refine_steps_effective)}; switch_step={int(refine_tail_start)}; '
                'true_refiner=True; same_model_lifecycle=True; second_model_load=False',
                flush=True,
            )

        unified_runtime = graph.node(
            "MiniMaxH3LatentLabUnifiedRuntimeSampler",
            initial_av=initial_av,
            long_media_plan=plan,
            guider=traced_guider,
            sampler=profiled_sampler,
            sigmas=sigmas,
            seed=int(seed),
            video_context_denoise=float(video_context_denoise),
            audio_context_denoise=float(audio_context_denoise),
            offload_completed_segments=bool(offload_completed_segments),
            latent_hires_enabled=bool(latent_hires_enabled),
            latent_hires_model=str(latent_hires_model),
            latent_hires_scale=float(latent_hires_scale),
            latent_hires_precision=str(latent_hires_precision),
            latent_hires_align=int(latent_hires_align),
            refine_enabled=bool(refine_enabled),
            refine_steps=int(refine_steps),
        )
        stitched = unified_runtime.out(0)
        _lm_print(
            '[MiniMaxH3 LongMedia][UNIFIED EXECUTOR] '
            f'passes={int(plan.passes)}; graph_level_sampler_nodes=0; '
            'runtime_sampler_lifecycle=1; model_reload_between_segments=False',
            flush=True,
        )

        try:
            _profile_video, _profile_audio = unpack_av_samples(initial_av)
            input_geometry = {
                "video_shape": list(_profile_video.shape),
                "audio_shape": list(_profile_audio.shape),
                "video_dtype": str(_profile_video.dtype),
                "audio_dtype": str(_profile_audio.dtype),
                "latent_payload_mb": _mb(
                    _profile_video.numel() * _profile_video.element_size()
                    + _profile_audio.numel() * _profile_audio.element_size()
                ),
            }
        except Exception as _profile_exc:
            input_geometry = {"error": f"{type(_profile_exc).__name__}: {_profile_exc}"}

        report = json.dumps({
            "passes": plan.passes,
            "segmentation_active": bool(getattr(plan, "segmentation_active", False)),
            "segment_lengths_frames": list(plan.segment_lengths),
            "segment_starts_frames": list(plan.segment_starts),
            "overlap_frames": int(plan.overlap_frames),
            "sequential_context_carry": bool(int(getattr(plan, "passes", 1) or 1) > 1),
            "latent_hires_enabled": bool(latent_hires_enabled),
            "latent_hires_model": str(latent_hires_model) if bool(latent_hires_enabled) else None,
            "latent_hires_scale": float(latent_hires_scale) if bool(latent_hires_enabled) else 1.0,
            "latent_hires_precision": str(latent_hires_precision) if bool(latent_hires_enabled) else None,
            "advanced_refine_enabled": bool(refine_enabled),
            "advanced_refine_add_noise": bool(latent_hires_enabled),
            "advanced_refine_legacy_add_noise_input_ignored": bool(refine_add_noise),
            "advanced_refine_seed": None,
            "advanced_refine_legacy_seed_input_ignored": int(refine_seed) & 0xFFFFFFFFFFFFFFFF,
            "advanced_refine_steps_requested": int(refine_steps),
            "advanced_refine_steps_effective": int(refine_steps_effective),
            "advanced_refine_base_steps": int(base_steps),
            "advanced_refine_total_model_steps": int(base_steps),
            "advanced_refine_tail_start": int(refine_tail_start),
            "advanced_refine_interval_policy": ("independent_hires_second_pass" if bool(latent_hires_enabled) else "true_two_stage_split_inside_single_model_lifecycle"),
            "advanced_refine_latent_source": ("stage1_denoised_x0_then_learned_upscale" if bool(latent_hires_enabled) else "stage1_in_progress_solver_state"),
            "advanced_refine_latent_clone": False,
            "advanced_refine_latent_rebuild": False,
            "advanced_refine_renoise": bool(latent_hires_enabled),
            "advanced_refine_scheduler_source": ("independent_every_other_full_schedule" if bool(latent_hires_enabled) else "unified_runtime_true_advanced_split"),
            "advanced_refine_audio_policy": "joint_AV_continues_through_refiner",
            "advanced_refine_overlap_policy": "unchanged_single_execution",
            "external_sampler_refine_auto_detect": True,
            "external_sampler_refine_policy": "per_clip_native_zero_noise_preserve_overlap",
            "hybrid_keyframe_scope": (
                "first_only_pass0_last_only_final"
                if getattr(plan, 'mode', None) == 'hybrid' and plan.passes > 1
                else "unchanged"
            ),
            "continuation_driver": ("V64_storyboard_A_to_B_to_C_exact_shared_bridge" if getattr(plan, 'mode', None) == 'storyboard_bridge' else "V331_stateful_frozen_overlap_motion_context"),
            "motion_context": {
                "enabled": bool(plan.passes > 1),
                "prototype_frames": 56,
                "source": "previous_generated_h3_video_latent_tail",
                "vae_roundtrip": False,
                "te_roundtrip": False,
            },
            "segment_prompting": "V331_iterative_completed_event_state_plus_local_timeline_native_refs",
            "conditioning_payload_copy": "shared_read_only_media_metadata_no_deepcopy",
            "stitched_single_output": bool(getattr(plan, 'mode', None) != 'multiclip'),
            "multiclip_native_clip_storage": bool(getattr(plan, 'mode', None) == 'multiclip'),
            "stitch_policy": {
                "hidden_overlap": "exact_context_trim_no_blend",
                "cross_time_visible_latent_blend": False,
                "first_visible_continuation_step_preserved": True,
            },
            "memory_mode": {
                "requested": str(memory_profile.get('requested')),
                "effective": str(memory_profile.get('effective')),
                "reason": str(memory_profile.get('reason')),
                "model_gb": (round(float(memory_profile.get('model_bytes')) / (1024**3), 3) if memory_profile.get('model_bytes') else None),
                "gpu_gb": (round(float(memory_profile.get('gpu_bytes')) / (1024**3), 3) if memory_profile.get('gpu_bytes') else None),
            },
            "transport_policy": {
                "sampler_pinned_memory_fastpath": "recent_aimdo_native_tensorwise_int8_preserve_user_pins",
                "te_reference_pinned_memory_gate": "unchanged_conservative",
                "giant_int8_mlp_floor": "post_block0_probe_2048_if_90k_150k_and_headroom",
            },
            "low_vram_mlp": {
                "mode": "token_chunk_exact",
                "enabled": bool(mlp_chunking_enabled),
                "chunk_tokens_requested": int(requested_mlp_chunk_tokens),
                "chunk_tokens_effective": int(effective_mlp_chunk_tokens),
                "attention_unchanged": str(attention_mode) in ("auto", "existing"),
            },
            "attention": {
                "mode": str(effective_attention_mode),
                "requested_mode": str(requested_attention_mode),
                "continuity_locked": False,
                "continuity_policy": (
                    "first_auto_resolution_latched_across_passes"
                    if int(getattr(plan, 'passes', 1)) > 1 and requested_attention_mode == 'auto'
                    else "explicit_or_single_pass"
                ),
                "embedded_sol": str(attention_mode) in ("sol", "scheduled_sol"),
                "sol_tau_start": float(sol_tau_start),
                "sol_tau_end": float(sol_tau_end),
                "sol_curve": str(sol_curve),
                "sol_min_tokens": int(sol_min_tokens),
                "sol_dense_percent": float(sol_dense_percent),
                "sol_sink_conditioning": str(sol_sink_conditioning),
                "sol_qkv_chunk_tokens": int(sol_qkv_chunk_tokens),
                "sol_qkv_streaming_enabled": int(sol_qkv_chunk_tokens) > 0,
                "sol_out_proj_chunk_tokens": int(sol_out_proj_chunk_tokens),
                "sol_out_proj_chunking_enabled": int(sol_out_proj_chunk_tokens) > 0,
                "vram_activation_reserve_mb": int(vram_activation_reserve_mb),
                "inter_block_vram_guard_mb": int(inter_block_vram_guard_mb),
                "inter_block_guard_cooldown_blocks": int(inter_block_guard_cooldown_blocks),
                "inter_block_guard_emergency_mb": int(inter_block_guard_emergency_mb),
                "inter_block_guard_emergency_cooldown_blocks": int(inter_block_guard_emergency_cooldown_blocks),
                "late_block_guard_start": int(late_block_guard_start),
                "late_block_guard_target_mb": int(late_block_guard_target_mb),
                "late_block_guard_min_cached_mb": int(late_block_guard_min_cached_mb),
                "step_boundary_cleanup_mb": int(step_boundary_cleanup_mb),
                "mlp_inplace_reuse": True,
                "implementation": "LongMedia embedded SM120 BF16 Sol-Attn (Apache-2.0 adapted)",
            },
            "input_geometry": input_geometry,
            "total_frames": plan.output_frames,
            "trim_frames": plan.trim_frames,
            "audio_reference_timeline": (
                "cropped_per_pass" if plan.mode == "automatic_lip_sync" else "full"
            ),
            "video_context_denoise": float(video_context_denoise),
            "audio_context_denoise": float(audio_context_denoise),
            "offload_completed_segments": bool(offload_completed_segments),
        }, indent=2)

        # Always run once after the final sampling pass, including the common
        # one-segment case where manual_duration == segment duration. This only
        # releases unused allocator cache; it deliberately does not unload H3.
        cleanup = graph.node(
            "MiniMaxH3LatentLabVRAMCacheCleanup",
            latent=stitched,
            sampler_report=report,
            memory_profile_state=memory_profile_state,
            block_trace_state=block_trace_state,
        )
        final_latent = cleanup.out(0)
        if ultra_pin_previous is not None:
            ultra_pin_restore = graph.node(
                "MiniMaxH3LatentLabUltraPinnedMemoryRestore",
                final_av=final_latent,
                previous_disable_pinned_memory=ultra_pin_previous,
                restore=True,
            )
            final_latent = ultra_pin_restore.out(0)
        return {
            "result": (final_latent, plan.output_frames, plan.trim_frames, plan.passes, cleanup.out(1)),
            "expand": graph.finalize(),
        }


class MiniMaxH3LatentLabLongMediaDecode:
    DESCRIPTION = (
        'Decode the final stitched H3 AV latent back to pixel frames and audio. '
        'This is the single combined result after any multi-pass long-media segmentation.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'final_av': ('LATENT',),
                'long_media_plan': ('LONG_MEDIA_PLAN',),
                'enable_tiling': ('BOOLEAN', {'default': True}),
                'tile_size': (
                    'INT',
                    {'default': 256, 'min': 32, 'max': 2048, 'step': 32},
                ),
                'width': (
                    'INT',
                    {'default': 512, 'min': 32, 'max': 8192, 'step': 32},
                ),
                'temporal_size': (
                    'INT',
                    {'default': 32, 'min': 1, 'max': 256, 'step': 1},
                ),
                'batch_size': (
                    'INT',
                    {'default': 1, 'min': 1, 'max': 16, 'step': 1},
                ),
                'color_match_strength': (
                    'FLOAT',
                    {
                        'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                        'tooltip': (
                            '0 disables. >0 nudges every frame\'s color statistics '
                            'toward frame 0, useful when frame 0 is pinned to a '
                            'reference and the rest of the clip drifts in color '
                            '(visible as a jump at a loop seam).'
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ('IMAGE', 'AUDIO', 'FLOAT', 'STRING')
    RETURN_NAMES = ('images', 'audio', 'duration_seconds', 'report')
    FUNCTION = 'decode'
    CATEGORY = CATEGORY_LONGMEDIA

    def decode(self, final_av, long_media_plan, enable_tiling, tile_size, width, temporal_size,
               batch_size, color_match_strength=0.0):
        plan = long_media_plan
        decode_barrier = _release_model_memory_for_decode()
        video, audio = unpack_av_samples(final_av)
        video_vae = plan.video_vae
        audio_vae = plan.audio_vae
        use_per_clip_native_video_decode = (
            str(getattr(plan, 'workflow_mode', '') or '') == 'multiclip'
            and bool(final_av.get('_lm_per_clip_native_video_decode'))
            and bool(final_av.get('_lm_segment_latents'))
        )
        segment_latents = list(final_av.get('_lm_segment_latents') or []) if use_per_clip_native_video_decode else []
        segment_lengths = list(final_av.get('_lm_segment_lengths') or []) if use_per_clip_native_video_decode else []
        segment_hidden_overlaps = list(final_av.get('_lm_segment_hidden_overlaps') or []) if use_per_clip_native_video_decode else []
        multiclip_seam_indices = [0 for _ in segment_latents]
        output_frames = plan.output_frames
        if use_per_clip_native_video_decode:
            # v0.4.21 native continuous decode: preserve sequential H3 generation,
            # but do NOT decode clips independently. Strip the repeated native
            # continuation prefix from every clip after the first, concatenate the
            # remaining VIDEO latents, validate T=5*k+2, then invoke VideoVAE once.
            video_parts = []
            latent_reports = []
            native_overlap_frames = int(getattr(plan, 'overlap_frames', 0) or 0)
            overlap_video_t = int(video_latent_t(native_overlap_frames)) if native_overlap_frames > 0 else 0
            for clip_index, clip_av in enumerate(segment_latents):
                clip_video, _clip_audio = unpack_av_samples(clip_av)
                if clip_index == 0:
                    contribution = clip_video
                    stripped_t = 0
                else:
                    if overlap_video_t <= 0:
                        contribution = clip_video
                        stripped_t = 0
                    else:
                        if int(clip_video.shape[2]) <= overlap_video_t:
                            raise RuntimeError(
                                'MultiClip native continuous decode cannot strip continuation prefix: '
                                f'clip={clip_index}, clip_t={int(clip_video.shape[2])}, overlap_t={overlap_video_t}.'
                            )
                        contribution = clip_video[:, :, overlap_video_t:].contiguous()
                        stripped_t = overlap_video_t
                if video_parts:
                    ref = video_parts[0]
                    if (int(contribution.shape[0]) != int(ref.shape[0]) or
                        int(contribution.shape[1]) != int(ref.shape[1]) or
                        tuple(contribution.shape[-2:]) != tuple(ref.shape[-2:])):
                        raise RuntimeError(
                            'MultiClip native continuous VideoVAE requires identical spatial latent geometry; '
                            f'clip={clip_index}, first={tuple(ref.shape)}, current={tuple(contribution.shape)}.'
                        )
                video_parts.append(contribution)
                latent_reports.append({
                    'clip_index': int(clip_index),
                    'native_t': int(clip_video.shape[2]),
                    'stripped_overlap_t': int(stripped_t),
                    'contribution_t': int(contribution.shape[2]),
                })
            if not video_parts:
                raise RuntimeError('MultiClip native continuous decode found no stored segment video latents.')
            continuous_video = torch.cat(video_parts, dim=2)
            continuous_frames = int(frame_count_from_video_t(int(continuous_video.shape[2])))
            expected_frames = int(plan.output_frames)
            if continuous_frames != expected_frames:
                raise RuntimeError(
                    'MultiClip native continuous latent geometry mismatch: '
                    f'assembled_t={int(continuous_video.shape[2])} -> {continuous_frames} frames, '
                    f'plan expects {expected_frames} frames; overlap={native_overlap_frames}f/{overlap_video_t}t.'
                )
            images, video_decode_info = _decode_video_vae_safe(
                video_vae, continuous_video, enable_tiling, tile_size, temporal_size,
            )
            if images.dim() == 5:
                images = images[0]
            multiclip_seam_indices = [0 for _ in segment_latents]
            video_decode_info = dict(video_decode_info or {})
            video_decode_info.update({
                'mode': 'single_native_continuous_video_vae_decode',
                'clip_count': int(len(segment_latents)),
                'native_overlap_frames': int(native_overlap_frames),
                'native_overlap_video_t': int(overlap_video_t),
                'continuous_video_t': int(continuous_video.shape[2]),
                'continuous_frames': int(continuous_frames),
                'rgb_seam_processing': False,
                'per_clip_video_vae_decode': False,
                'clips': latent_reports,
            })
            _lm_print(
                '[MiniMaxH3 LongMedia][NATIVE CONTINUOUS VIDEO VAE] '
                f'clips={len(segment_latents)} overlap={native_overlap_frames}f/{overlap_video_t}t '
                f'assembled_t={int(continuous_video.shape[2])} frames={continuous_frames}; '
                'single_decode=True rgb_seam_processing=False',
                flush=True,
            )
        else:
            images, video_decode_info = _decode_video_vae_safe(
                video_vae, video, enable_tiling, tile_size, temporal_size,
            )
            if images.dim() == 5:
                images = images[0]
        storyboard_duplicate_removed = False
        if getattr(plan, 'mode', None) == 'storyboard_bridge':
            bridge = int(getattr(plan, 'storyboard_bridge_frame', -1))
            if 0 < bridge < int(images.shape[0]):
                images = torch.cat((images[:bridge], images[bridge + 1:]), dim=0)
                storyboard_duplicate_removed = True
        trimmed = 0
        if images.shape[0] > output_frames:
            trimmed = images.shape[0] - output_frames
            images = images[:output_frames]
        opening_anchor_suppressed = False
        opening_anchor_suppressed_frames = 0
        if bool(getattr(plan, 'suppress_visible_opening_anchor', False)) and int(images.shape[0]) > 1:
            # Keep the proven sampling-time H3 opening anchor exactly as in the
            # known-good 0.3.32/0.3.39 path, but hide the first anchor-biased visible
            # decode frames at output time only. In practice frame 1 can still carry a
            # weak residual of the startup anchor, so for segmented_continuation we
            # promote the first clearly-generated frame (frame 2 when present) into the
            # first visible slots. This leaves all latents, continuation guides,
            # segment conditioning, and later frames untouched.
            images = images.clone()
            if int(images.shape[0]) > 2:
                replacement = images[2]
                images[0] = replacement
                images[1] = replacement
                opening_anchor_suppressed_frames = 2
            else:
                images[0] = images[1]
                opening_anchor_suppressed_frames = 1
            opening_anchor_suppressed = True
        def _compact_decode_snap(snap):
            if snap is None:
                return None
            return {
                'driver_free_mb': _mb(snap['driver_free']),
                'allocated_mb': _mb(snap['allocated']),
                'reserved_mb': _mb(snap['reserved']),
                'cached_mb': _mb(snap['cached']),
            }

        report_data = {
            'model_memory_released_before_decode': True,
            'decode_memory_barrier_before': _compact_decode_snap(decode_barrier.get('before')),
            'decode_memory_barrier_after': _compact_decode_snap(decode_barrier.get('after')),
            'decode_memory_barrier_errors': decode_barrier.get('errors', []),
            'decode_uses_plan_vaes': True,
            'video_decode': video_decode_info,
            'per_clip_native_video_decode': bool(use_per_clip_native_video_decode),
            'trimmed_video_frames': trimmed,
            'storyboard_duplicate_boundary_frame_removed': storyboard_duplicate_removed,
            'segmented_visible_opening_anchor_suppressed': opening_anchor_suppressed,
            'segmented_visible_opening_anchor_suppressed_frames': opening_anchor_suppressed_frames,
        }
        audio_output_mode = str(getattr(plan, 'audio_output_mode', 'auto') or 'auto')
        passthrough_audio_mode = audio_output_mode in ('auto', 'preserve', 'preserve_reference', 'lip_sync')
        preserve_audio_bypass = audio_output_mode in ('preserve', 'preserve_reference', 'lip_sync')

        # Preserve means literal bypass: never send the sampled/model audio stream back
        # through the H3 Audio VAE. This is intentionally checked before every generated
        # audio decode path, because distilled/Turbo LoRAs can alter the sampled audio
        # stream geometry and make it invalid for the VAE normalizer.
        if preserve_audio_bypass and plan.final_audio_override is None:
            raise RuntimeError(
                f"audio_mode={audio_output_mode!r} requires a connected source audio track, "
                "but LongMediaPlan has no final_audio_override. Connect audio_1 (or another "
                "audio input) or switch audio_mode to generate/reference_only."
            )

        # Pixel override/blend is an output policy, not a lip-sync-only feature.
        # Apply it for Manual hybrid first-frame workflows as well. latent_inject
        # is already baked into the sampled target and needs no post-decode edit.
        first_frame_mode = getattr(plan, 'first_frame_mode', 'latent_inject')
        first_frame_override = getattr(plan, 'first_frame_override', None)
        first_frame_latent_injected = bool(
            getattr(plan, 'first_frame_latent_injected', False)
        )
        if first_frame_override is not None and first_frame_mode in ('pixel_override', 'blend'):
            first_frame = first_frame_override
            if first_frame.dim() == 4:
                first_frame = first_frame[0]
            target_h, target_w = images.shape[1], images.shape[2]
            if first_frame.shape[0] != target_h or first_frame.shape[1] != target_w:
                import comfy.utils
                ff = first_frame.unsqueeze(0).movedim(-1, 0)
                ff = comfy.utils.common_upscale(ff, target_w, target_h, 'lanczos', 'disabled')
                first_frame = ff.squeeze(0).movedim(0, -1)
            first_frame = first_frame.to(images.dtype)
            if first_frame_mode == 'pixel_override':
                images[0] = first_frame
            else:  # blend
                images = _blend_leading_frames_to_reference(
                    images, first_frame, plan.first_frame_blend_frames,
                )
        if first_frame_override is not None or first_frame_latent_injected:
            report_data['first_frame_mode'] = first_frame_mode
            report_data['first_frame_restored'] = True
            report_data['first_frame_latent_injected'] = first_frame_latent_injected

        if plan.mode == 'automatic_lip_sync':
            if (passthrough_audio_mode and plan.final_audio_override is not None) or preserve_audio_bypass:
                audio, passthrough_fit = _fit_passthrough_audio_to_timeline(
                    plan.final_audio_override, plan.total_duration
                )
                report_data['audio_passthrough_timeline_fit'] = passthrough_fit
                report_data['original_audio_restored'] = True
                report_data['generated_audio_decoded'] = False
                report_data['audio_output_mode'] = audio_output_mode
                report_data['audio_vae_bypassed'] = True
            elif audio_vae is not None:
                if not hasattr(audio, 'shape') or audio.ndim != 4 or int(audio.shape[1]) != 32 or int(audio.shape[2]) != 2:
                    shape = tuple(audio.shape) if hasattr(audio, 'shape') else type(audio).__name__
                    raise RuntimeError(
                        'LongMedia received an invalid generated H3 audio latent for AudioVAE decode: '
                        f'shape={shape}, audio_mode={audio_output_mode!r}. '
                        'Use preserve_reference/preserve (or auto with attached audio) to bypass AudioVAE.'
                    )
                sr = int(getattr(audio_vae, 'audio_sample_rate', 32000))
                decoded_audio = audio_vae.decode(audio)
                audio = _normalize_decoded_audio(
                    decoded_audio, sr, round(plan.total_duration * sr)
                )
                report_data['original_audio_restored'] = False
                report_data['generated_audio_decoded'] = True
        else:
            # Restore original audio only when requested; otherwise decode model audio.
            if (passthrough_audio_mode and plan.final_audio_override is not None) or preserve_audio_bypass:
                audio, passthrough_fit = _fit_passthrough_audio_to_timeline(
                    plan.final_audio_override, plan.total_duration
                )
                report_data['audio_passthrough_timeline_fit'] = passthrough_fit
                report_data['generated_audio_decoded'] = False
                report_data['audio_output_mode'] = audio_output_mode
                report_data['audio_vae_bypassed'] = True
            elif use_per_clip_native_video_decode and audio_vae is not None:
                native_audio_overlap = int(getattr(plan, 'overlap_frames', 0) or 0)
                audio, multiclip_audio_reports = _decode_multiclip_audio_segments(
                    segment_latents, audio_vae, segment_lengths, plan.total_duration,
                    hidden_overlaps=[0] + [native_audio_overlap] * max(0, len(segment_latents) - 1),
                    seam_indices=[0] + [native_audio_overlap] * max(0, len(segment_latents) - 1),
                )
                report_data['multiclip_per_clip_audio_decode'] = True
                report_data['multiclip_audio_clips'] = multiclip_audio_reports
            elif audio_vae is not None and hasattr(audio, 'shape') and audio.ndim == 4:
                # MiniMax H3 AudioVAE expects latent layout [B, 32, 2, T]. A Turbo LoRA
                # or wrong routing can leave a packed/non-audio tensor here. Never hand
                # such a tensor to the VAE normalizer, which otherwise fails with an
                # opaque 19296-vs-128 broadcast error.
                if int(audio.shape[1]) != 32 or int(audio.shape[2]) != 2:
                    raise RuntimeError(
                        'LongMedia received an invalid generated H3 audio latent for AudioVAE decode: '
                        f'shape={tuple(audio.shape)}, audio_mode={audio_output_mode!r}. '
                        'For an attached source soundtrack use audio_mode=preserve_reference/preserve '
                        '(or auto, which now preserves attached audio). Use generate/reference_only only '
                        'when model-generated audio is intended.'
                    )
                sr = int(getattr(audio_vae, 'audio_sample_rate', 32000))
                decoded_audio = audio_vae.decode(audio)
                audio = _normalize_decoded_audio(
                    decoded_audio, sr, round(plan.total_duration * sr)
                )
                if getattr(plan, 'mode', None) == 'storyboard_bridge' and storyboard_duplicate_removed:
                    wave = audio['waveform']
                    cut_at = int(round((int(getattr(plan, 'storyboard_bridge_frame', 0)) / FPS) * sr))
                    one = int(round(sr / FPS))
                    if 0 <= cut_at and cut_at + one <= int(wave.shape[-1]):
                        wave = torch.cat((wave[..., :cut_at], wave[..., cut_at + one:]), dim=-1)
                        target = int(round(plan.total_duration * sr))
                        wave = torch.nn.functional.pad(wave, (0, max(0, target - int(wave.shape[-1]))))[..., :target]
                        audio = {'waveform': wave, 'sample_rate': sr}
                        report_data['storyboard_duplicate_audio_frame_removed'] = True
                report_data['generated_audio_decoded'] = True
            report_data['original_audio_restored'] = bool((passthrough_audio_mode and plan.final_audio_override is not None) or preserve_audio_bypass)
            report_data['first_frame_restored'] = False
        loop_closure_enabled = bool(getattr(plan, 'loop_closure_enabled', False))
        loop_closure_frames = max(2, int(getattr(plan, 'loop_closure_frames', 0) or 0))
        loop_closure_strength = max(0.0, min(1.0, float(getattr(plan, 'loop_closure_strength', 0.65))))
        loop_closure_runtime = final_av.get('_lm_loop_closure_report') if isinstance(final_av, dict) else None
        loop_closure_applied = bool(isinstance(loop_closure_runtime, dict) and loop_closure_runtime.get('applied'))
        if color_match_strength > 0.0:
            images = _match_frames_color_to_reference(images, 0, color_match_strength)
        report_data['color_match_strength'] = float(color_match_strength)
        report_data['loop_closure_enabled'] = loop_closure_enabled
        report_data['loop_closure_frames_requested'] = int(loop_closure_frames if loop_closure_enabled else 0)
        report_data['loop_closure_strength'] = float(loop_closure_strength if loop_closure_enabled else 0.0)
        report_data['loop_closure_frames'] = int(
            loop_closure_runtime.get('actual_frames', loop_closure_frames)
            if isinstance(loop_closure_runtime, dict) and loop_closure_enabled
            else (loop_closure_frames if loop_closure_enabled else 0)
        )
        report_data['loop_closure_applied'] = bool(loop_closure_applied)
        report_data['loop_closure_mode'] = (
            str(loop_closure_runtime.get('mode'))
            if isinstance(loop_closure_runtime, dict) and loop_closure_runtime.get('mode') is not None
            else ('disabled' if not loop_closure_enabled else 'missing_runtime_report')
        )
        report_data['loop_closure_runtime'] = loop_closure_runtime if isinstance(loop_closure_runtime, dict) else None
        report = json.dumps(report_data, indent=2)
        return (images, audio, plan.total_duration, report)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3LatentLabUltraPinnedMemoryGate": MiniMaxH3LatentLabUltraPinnedMemoryGate,
    "MiniMaxH3LatentLabUltraPinnedMemoryRestore": MiniMaxH3LatentLabUltraPinnedMemoryRestore,
    'MiniMaxH3LatentLabVideoEncode': MiniMaxH3LatentLabVideoEncode,
    'MiniMaxH3LatentLabAudioEncode': MiniMaxH3LatentLabAudioEncode,
    'MiniMaxH3LatentLabPackAV': MiniMaxH3LatentLabPackAV,
    'MiniMaxH3LatentLabSplitAV': MiniMaxH3LatentLabSplitAV,
    'MiniMaxH3LatentLabReplaceStream': MiniMaxH3LatentLabReplaceStream,
    'MiniMaxH3LatentLabReplaceVideo': MiniMaxH3LatentLabReplaceVideo,  # deprecated alias
    'MiniMaxH3LatentLabReplaceAudio': MiniMaxH3LatentLabReplaceAudio,  # deprecated alias
    'MiniMaxH3LatentLabStreamDenoise': MiniMaxH3LatentLabStreamDenoise,
    'MiniMaxH3LatentLabLipSyncSetup': MiniMaxH3LatentLabLipSyncSetup,
    'MiniMaxH3LatentLabVideoInpaint': MiniMaxH3LatentLabVideoInpaint,
    'MiniMaxH3LatentLabMergeAV': MiniMaxH3LatentLabMergeAV,
    'MiniMaxH3LatentLabPrepareContinuation': MiniMaxH3LatentLabPrepareContinuation,
    'MiniMaxH3LatentLabStitchContinuation': MiniMaxH3LatentLabStitchContinuation,
    'MiniMaxH3LatentLabInfo': MiniMaxH3LatentLabInfo,
    'MiniMaxH3LongMediaPlanner': MiniMaxH3LongMediaPlanner,
    'MiniMaxH3LongMediaCameras': MiniMaxH3LongMediaCameras,
    'MiniMaxH3LongMediaVideoReconstructor': MiniMaxH3LongMediaVideoReconstructor,
    'MiniMaxH3LatentLabLongMediaSetup': MiniMaxH3LatentLabLongMediaSetup,
    'MiniMaxH3LatentLabLongMediaNextSegment': MiniMaxH3LatentLabLongMediaNextSegment,
    'MiniMaxH3LatentLabRuntimeContinuationGuider': MiniMaxH3LatentLabRuntimeContinuationGuider,
    'MiniMaxH3LatentLabRefineSigmas': MiniMaxH3LatentLabRefineSigmas,
    'MiniMaxH3LatentLabSeededDisableNoise': MiniMaxH3LatentLabSeededDisableNoise,
    'MiniMaxH3LatentLabProtectRefineAV': MiniMaxH3LatentLabProtectRefineAV,
    'MiniMaxH3LatentLabUnifiedRuntimeSampler': MiniMaxH3LatentLabUnifiedRuntimeSampler,
    'MiniMaxH3LatentLabLongMediaSampler': MiniMaxH3LatentLabLongMediaSampler,
    'MiniMaxH3LatentLabLongMediaDecode': MiniMaxH3LatentLabLongMediaDecode,
    'MiniMaxH3LatentLabAttentionChunking': MiniMaxH3LatentLabAttentionChunking,
    'MiniMaxH3LatentLabBlockMemoryTracer': MiniMaxH3LatentLabBlockMemoryTracer,
    'MiniMaxH3LatentLabMLPChunking': MiniMaxH3LatentLabMLPChunking,
    'MiniMaxH3LatentLabFirstStepMemoryProfiler': MiniMaxH3LatentLabFirstStepMemoryProfiler,
    'MiniMaxH3LatentLabVRAMPressureGuard': MiniMaxH3LatentLabVRAMPressureGuard,
    'MiniMaxH3LatentLabVRAMCacheCleanup': MiniMaxH3LatentLabVRAMCacheCleanup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'MiniMaxH3LatentLabVideoEncode': 'MiniMax H3 \u2022 Encode Video Stream',
    'MiniMaxH3LatentLabAudioEncode': 'MiniMax H3 \u2022 Encode Audio Stream',
    'MiniMaxH3LatentLabPackAV': 'MiniMax H3 \u2022 Pack AV Streams',
    'MiniMaxH3LatentLabSplitAV': 'MiniMax H3 \u2022 Split AV Streams',
    'MiniMaxH3LatentLabReplaceStream': 'MiniMax H3 \u2022 Replace Stream',
    'MiniMaxH3LatentLabReplaceVideo': 'MiniMax H3 \u2022 Replace Video Stream (deprecated)',
    'MiniMaxH3LatentLabReplaceAudio': 'MiniMax H3 \u2022 Replace Audio Stream (deprecated)',
    'MiniMaxH3LatentLabStreamDenoise': 'MiniMax H3 \u2022 Stream Denoise Controls',
    'MiniMaxH3LatentLabLipSyncSetup': 'MiniMax H3 \u2022 LipSync Latent Setup',
    'MiniMaxH3LatentLabVideoInpaint': 'MiniMax H3 \u2022 Video Inpaint',
    'MiniMaxH3LatentLabMergeAV': 'MiniMax H3 \u2022 Merge AV Latents',
    'MiniMaxH3LatentLabPrepareContinuation': 'MiniMax H3 \u2022 Prepare Continuation',
    'MiniMaxH3LatentLabStitchContinuation': 'MiniMax H3 \u2022 Stitch Continuation',
    'MiniMaxH3LatentLabInfo': 'MiniMax H3 \u2022 AV Latent Info',
    'MiniMaxH3LongMediaPlanner': 'MiniMax H3 LongMedia Planner',
    'MiniMaxH3LongMediaCameras': 'MiniMax H3 LongMedia Cameras',
    'MiniMaxH3LongMediaVideoReconstructor': 'MiniMax H3 LongMedia Video Reconstructor',
    'MiniMaxH3LatentLabLongMediaSetup': 'MiniMax H3 \u2022 Long Media Setup',
    'MiniMaxH3LatentLabLongMediaNextSegment': 'MiniMax H3 \u2022 Long Media Next Segment',
    'MiniMaxH3LatentLabRuntimeContinuationGuider': 'MiniMax H3 \u2022 Runtime Continuation Guider',
    'MiniMaxH3LatentLabLongMediaSampler': 'MiniMax H3 \u2022 Long Media Sampler',
    'MiniMaxH3LatentLabLongMediaDecode': 'MiniMax H3 \u2022 Long Media Decode',
    'MiniMaxH3LatentLabAttentionChunking': 'MiniMax H3 \u2022 Low-VRAM Attention Chunking (internal)',
    'MiniMaxH3LatentLabMLPChunking': 'MiniMax H3 \u2022 Low-VRAM MLP Chunking (internal)',
    'MiniMaxH3LatentLabFirstStepMemoryProfiler': 'MiniMax H3 \u2022 First-Step Memory Profiler (internal)',
    'MiniMaxH3LatentLabVRAMPressureGuard': 'MiniMax H3 \u2022 VRAM Pressure Guard (internal)',
    'MiniMaxH3LatentLabVRAMCacheCleanup': 'MiniMax H3 \u2022 VRAM Cache Cleanup (internal)',
}

replace = _dc_replace
