"""LongMedia MiniMax H3 keyframe + Ref2VA payload compatibility.

Current stock MiniMax H3 treats every ``minimax_keyframes`` entry as a visual
condition block.  Audio-only keyframes are therefore not representable in the
stock ``PackedLayout``: they either raise ``KeyError("latent")`` in
``MiniMaxH3.extra_conds`` or, if forced through, create visual mask rows without
matching video condition rows.

LongMedia lip-sync no longer depends on legacy lip-sync audio-only keyframes. Native H3 continuation audio keyframes are supported by current core and must remain represented in cond_audio_latents when refs are merged.  Its authoritative
clock is the exact source-audio window placed in the target AV latent with the
audio stream frozen.  This patch defensively strips legacy LongMedia audio-only
keyframes before calling stock while preserving the supported visual-keyframe +
Ref2VA merge behavior.
"""
import logging
import functools

import comfy.model_base as model_base

_LOG = logging.getLogger("minimax_h3_longmedia")
MC_KEY = "motion_context_index"
LM_KEY = "longmedia_hybrid_keyframe"
PATCH_MARKER = "_h3_motion_context_payload_patch"

_orig_extra_conds = None
_applied = False


def _is_longmedia_audio_only_keyframe(kf):
    if not isinstance(kf, dict):
        return False
    if kf.get("audio_latent") is None or kf.get("latent") is not None:
        return False
    return bool(
        kf.get("longmedia_lipsync_audio_guide")
        or kf.get("longmedia_v107_visible_lipsync_guide")
    )


def _visual_keyframes(keyframes):
    return [
        kf for kf in (keyframes or [])
        if isinstance(kf, dict) and kf.get("latent") is not None
    ]


def _normalize_ref_audio_geometry(refs):
    """Clone H3 refs and make ``ref_audio_t`` authoritative from audio_latent.

    PackedLayout allocates ref-audio rows from the metadata field ``ref_audio_t``,
    while MiniMaxH3._cond_audio_rows() emits rows from the actual ``audio_latent``.
    If a video/audio reference was trimmed/resampled/cached and those two drift,
    the layout can reserve (for example) 272 stereo steps while only 235 exist,
    producing the characteristic 544-vs-470 row broadcast failure.

    Never pad/crop audio to satisfy stale metadata.  The tensor is the source of
    truth; metadata follows it.
    """
    if refs is None:
        return None
    out = []
    for ref in refs:
        if not isinstance(ref, dict):
            out.append(ref)
            continue
        item = dict(ref)
        audio = item.get("audio_latent")
        if audio is not None and hasattr(audio, "shape") and len(audio.shape) >= 1:
            actual_t = int(audio.shape[-1])
            item["ref_audio_t"] = actual_t
            if item.get("kind") == "video" and actual_t > 0:
                item["kind"] = "video_audio"
        else:
            item["ref_audio_t"] = 0
            if item.get("kind") == "video_audio":
                item["kind"] = "video"
        out.append(item)
    return out


def _needs_merge(keyframes, refs):
    if not keyframes or not refs:
        return False
    return any((MC_KEY in kf) or bool(kf.get(LM_KEY)) for kf in keyframes)


def _payload_dict(out):
    holder = out.get("minimax_payload") if isinstance(out, dict) else None
    payload = getattr(holder, "cond", None) if holder is not None else None
    return payload if isinstance(payload, dict) else None


def _patched_extra_conds(self, **kwargs):
    original_keyframes = kwargs.get("minimax_keyframes") or []
    refs = _normalize_ref_audio_geometry(kwargs.get("minimax_refs"))
    # Keep every native H3 keyframe except LongMedia's legacy lip-sync-only
    # guides.  Current H3 PackedLayout legitimately supports audio-only
    # keyframes (used by MultiClip motion/audio continuation), and dropping them
    # from payload["keyframes"] after layout construction creates the exact
    # 37-audio-step / 74-row mismatch seen on clip 2 with Video Ref.
    supported_keyframes = [
        dict(kf) if isinstance(kf, dict) else kf
        for kf in original_keyframes
        if not _is_longmedia_audio_only_keyframe(kf)
    ]
    visual_keyframes = _visual_keyframes(supported_keyframes)
    stripped_audio_only = len(supported_keyframes) != len(original_keyframes)

    # Never let an audio-only LongMedia guide reach stock MiniMaxH3.extra_conds
    # or PackedLayout.  Stock keyframes are visual-only by construction.
    # Normalize reference audio geometry BEFORE stock extra_conds builds
    # PackedLayout.  Fixing the payload afterwards is too late because the
    # layout's audio_update mask has already been sized from ref_audio_t.
    call_kwargs = dict(kwargs)
    if refs is not None:
        call_kwargs["minimax_refs"] = refs
    if stripped_audio_only:
        if supported_keyframes:
            call_kwargs["minimax_keyframes"] = supported_keyframes
        else:
            call_kwargs.pop("minimax_keyframes", None)
        # frame_count only belongs to legacy first/last visual-keyframe math.
        if not visual_keyframes:
            call_kwargs.pop("minimax_frame_count", None)

    out = _orig_extra_conds(self, **call_kwargs)

    merge_refs = _needs_merge(supported_keyframes, refs)
    if not merge_refs:
        return out

    payload = _payload_dict(out)
    if payload is None:
        _LOG.warning("LongMedia hybrid: could not reach MiniMax H3 payload")
        return out

    # Stock currently overwrites cond_video_latents when refs are present.
    # Restore the supported visual keyframe + reference order exactly as
    # PackedLayout emits it.  Audio reference geometry has already been normalized from the actual
    # audio_latent tensors before stock PackedLayout construction.
    # CRITICAL: preserve native audio-only continuation keyframes in the
    # payload. PackedLayout was built from them, so _cond_audio_rows must see
    # the same keyframe list later in the forward pass.
    payload["keyframes"] = supported_keyframes
    if refs is not None:
        payload["refs"] = refs
    payload["cond_video_latents"] = (
        [kf["latent"] for kf in visual_keyframes]
        + [ref["latent"] for ref in (refs or [])
           if isinstance(ref, dict) and ref.get("latent") is not None]
    )
    # PackedLayout emits audio condition rows in the same order as the payload:
    # audio-bearing keyframes first, then audio-bearing refs.  Preserve BOTH.
    #
    # This matters on MultiClip continuation with native motion context:
    # segment > 0 can carry a visual continuation keyframe plus an audio-only
    # motion keyframe while Video/Audio refs are also present.  The old merge
    # rebuilt cond_audio_latents from refs only, so PackedLayout allocated rows
    # for the motion-audio keyframe but _cond_audio_rows() had no tensor for
    # them -> all_audio_rows[~audio_update] shape mismatch.
    payload["cond_audio_latents"] = (
        [kf["audio_latent"] for kf in (supported_keyframes or [])
         if isinstance(kf, dict) and kf.get("audio_latent") is not None]
        + [ref["audio_latent"] for ref in (refs or [])
           if isinstance(ref, dict) and ref.get("audio_latent") is not None]
    )
    return out


setattr(_patched_extra_conds, PATCH_MARKER, True)



def _wrapper_chain(fn, limit=32):
    chain = []
    seen = set()
    cur = fn
    for _ in range(int(limit)):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        chain.append(cur)
        cur = getattr(cur, "__wrapped__", None)
    return chain


def _chain_description(fn):
    out = []
    for item in _wrapper_chain(fn):
        out.append(
            "%s.%s" % (
                getattr(item, "__module__", "?"),
                getattr(item, "__qualname__", getattr(item, "__name__", "?")),
            )
        )
    return " -> ".join(out) if out else "<empty>"


def apply_patch():
    global _orig_extra_conds, _applied
    if _applied:
        return True

    cls = getattr(model_base, "MiniMaxH3", None)
    current = getattr(cls, "extra_conds", None) if cls is not None else None
    if current is None:
        _LOG.warning("LongMedia hybrid: MiniMaxH3.extra_conds unavailable")
        return False

    chain = _wrapper_chain(current)
    if any(getattr(item, PATCH_MARKER, False) for item in chain):
        _applied = True
        _LOG.info(
            "LongMedia hybrid: compatible payload merge already present in wrapper chain: %s",
            _chain_description(current),
        )
        return True

    # v0.4.81: compose outside an existing wrapper instead of refusing the stack.
    #
    # Our wrapper is post-processing and calls the current owner first. This lets
    # metadata/conditioning extensions preserve their behavior while LongMedia
    # only repairs the MiniMax payload after they return. Unknown wrappers are
    # therefore no longer treated as a collision merely because they expose
    # __wrapped__ or live in another module.
    _orig_extra_conds = current
    functools.update_wrapper(_patched_extra_conds, current)
    setattr(_patched_extra_conds, PATCH_MARKER, True)
    setattr(_patched_extra_conds, "_longmedia_wrapped_owner", current)

    cls.extra_conds = _patched_extra_conds
    _applied = True
    _LOG.info(
        "LongMedia hybrid: payload merge stacked safely outside existing owner; chain=%s",
        _chain_description(current),
    )
    return True


def is_applied():
    return _applied
