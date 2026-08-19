"""LongMedia self-contained MiniMax H3 keyframe+Ref2VA payload compatibility.

This is intentionally narrow: it fixes the stock MiniMaxH3.extra_conds
last-writer-wins collision only for conditioning records marked by LongMedia
(or the shared Motion Context marker ABI).  It does not patch PackedLayout and
therefore does not own arbitrary-position continuation guides.
"""
import logging

import comfy.model_base as model_base

_LOG = logging.getLogger("minimax_h3_longmedia")
MC_KEY = "motion_context_index"
LM_KEY = "longmedia_hybrid_keyframe"
PATCH_MARKER = "_h3_motion_context_payload_patch"

_orig_extra_conds = None
_applied = False


def _needs_merge(keyframes, refs):
    if not keyframes or not refs:
        return False
    return any((MC_KEY in kf) or bool(kf.get(LM_KEY)) for kf in keyframes)


def _patched_extra_conds(self, **kwargs):
    out = _orig_extra_conds(self, **kwargs)
    keyframes = kwargs.get("minimax_keyframes")
    refs = kwargs.get("minimax_refs")
    if not _needs_merge(keyframes, refs):
        return out

    holder = out.get("minimax_payload")
    payload = getattr(holder, "cond", None) if holder is not None else None
    if not isinstance(payload, dict):
        _LOG.warning("LongMedia hybrid: could not reach MiniMax H3 payload")
        return out

    payload["cond_video_latents"] = (
        [kf["latent"] for kf in keyframes if "latent" in kf]
        + [ref["latent"] for ref in refs if "latent" in ref]
    )
    payload["cond_audio_latents"] = [
        ref["audio_latent"] for ref in refs
        if ref.get("audio_latent") is not None
    ]
    frame_count = kwargs.get("minimax_frame_count")
    if frame_count is not None:
        payload["frame_count"] = frame_count
    return out


setattr(_patched_extra_conds, PATCH_MARKER, True)


def apply_patch():
    global _orig_extra_conds, _applied
    if _applied:
        return True
    cls = getattr(model_base, "MiniMaxH3", None)
    current = getattr(cls, "extra_conds", None) if cls is not None else None
    if current is None:
        _LOG.warning("LongMedia hybrid: MiniMaxH3.extra_conds unavailable")
        return False

    # A compatible Contex Loop / Motion Context copy already owns the merge.
    if getattr(current, PATCH_MARKER, False):
        _applied = True
        _LOG.debug("LongMedia hybrid: compatible H3 payload merge already active")
        return True

    # Fail closed on unknown wrappers instead of stacking process-global patches.
    home = getattr(cls, "__module__", None)
    where = getattr(current, "__module__", None)
    if hasattr(current, "__wrapped__") or (home and where and where != home):
        _LOG.warning(
            "LongMedia hybrid: MiniMaxH3.extra_conds is already wrapped by %r; "
            "refusing to stack another payload patch", where)
        return False

    _orig_extra_conds = current
    cls.extra_conds = _patched_extra_conds
    _applied = True
    _LOG.debug("LongMedia hybrid: self-contained keyframe+Ref2VA payload merge enabled")
    return True


def is_applied():
    return _applied
