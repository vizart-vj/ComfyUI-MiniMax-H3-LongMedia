"""Temporal-position continuity helpers for MiniMax H3 multipass sampling."""

from __future__ import annotations

import copy


H3_OUTPUT_FPS = 24.0
H3_TEMPORAL_ROPE_HZ = 40.0
H3_ROPE_UNITS_PER_FRAME = H3_TEMPORAL_ROPE_HZ / H3_OUTPUT_FPS
TEMPORAL_OFFSET_OPTION = "latentlab_h3_temporal_offset"
_SHIFTED_SEGMENT_KINDS = frozenset({"cond", "ref_audio", "audio", "video"})


def temporal_offset_for_frame(start_frame: int) -> float:
    """Convert a global 24-fps frame origin to H3's 40-Hz RoPE timeline."""
    start_frame = int(start_frame)
    if start_frame < 0:
        raise ValueError("H3 temporal start frame must be non-negative.")
    return start_frame * H3_ROPE_UNITS_PER_FRAME


def apply_h3_temporal_offset(layout, offset: float):
    """Copy an H3 PackedLayout and shift temporal media rows only.

    Text and static image-reference coordinates stay fixed. Target audio/video,
    temporal keyframe conditions, and reference-audio rows move together so
    their relative AV timing remains unchanged. Latent values are not accepted
    or modified by this function.
    """
    offset = float(offset)
    shifted = copy.copy(layout)
    # Do not copy the source cache into the shifted layout itself; that would
    # create an unnecessary shifted->cache->shifted reference cycle.
    if hasattr(shifted, "_latentlab_temporal_offsets"):
        delattr(shifted, "_latentlab_temporal_offsets")
    position_ids = layout.position_ids.clone()
    if offset:
        for start, stop, kind in layout.segments:
            if kind in _SHIFTED_SEGMENT_KINDS:
                position_ids[start:stop, 0] += offset
    shifted.position_ids = position_ids
    return shifted


def _cached_shifted_layout(layout, offset: float):
    """Cache one immutable shifted copy on the per-sampling PackedLayout."""
    cache = getattr(layout, "_latentlab_temporal_offsets", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(layout, "_latentlab_temporal_offsets", cache)
    key = float(offset)
    shifted = cache.get(key)
    if shifted is None:
        shifted = apply_h3_temporal_offset(layout, key)
        cache[key] = shifted
    return shifted


def h3_temporal_offset_wrapper(executor, *args, **kwargs):
    """APPLY_MODEL wrapper applying a pass-global H3 RoPE/layout offset.

    APPLY_MODEL wrappers receive BaseModel.apply_model positional arguments,
    not MiniMaxH3Model._forward arguments.  Preserve that call layout and only
    replace transformer_options / minimax_payload when an offset is active.
    """
    call_args = list(args)
    options = call_args[5] if len(call_args) > 5 and isinstance(call_args[5], dict) else kwargs.get("transformer_options")
    options = options or {}
    offset = float(options.get(TEMPORAL_OFFSET_OPTION, 0.0) or 0.0)
    payload = kwargs.get("minimax_payload")
    layout = payload.get("layout") if isinstance(payload, dict) else None
    if offset and layout is not None:
        payload = dict(payload)
        payload["layout"] = _cached_shifted_layout(layout, offset)
        kwargs["minimax_payload"] = payload
    return executor(*call_args, **kwargs)

