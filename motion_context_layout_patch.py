"""Marker-gated MiniMax H3 continuation-guide layout support for LongMedia.

Only conditioning keyframes carrying ``motion_context_index`` are changed.
Ordinary H3 graphs remain stock.  The marker ABI intentionally matches the
H3 Motion Context / Contex Loop family so a compatible installed owner can be
reused instead of stacking a second process-global patch.
"""
import logging
import inspect
import comfy.ldm.minimax.model as mm

MC_KEY = "motion_context_index"
PATCH_MARKER = "_h3_motion_context_layout_patch"
SOLATTN_LAYOUT_MODULE_SUFFIX = "._morton_h3"
_LOG = logging.getLogger("minimax_h3_longmedia")
_orig_init = None
_applied = False


def _target_origin(layout):
    a, b, kind = layout.segments[-1]
    if kind != "video" or b <= a:
        raise RuntimeError(
            "LongMedia motion context: expected target video as final PackedLayout segment")
    return float(layout.position_ids[a, 0])


def _orig_accepts_frame_count():
    try:
        sig = inspect.signature(_orig_init)
    except Exception:
        return False
    params = sig.parameters
    return (
        "frame_count" in params
        or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    )


def _call_orig(self, text_len, latent_t, latent_h, latent_w, audio_t,
               *, keyframes=None, refs=None, frame_count=None):
    kwargs = {"keyframes": keyframes, "refs": refs}
    if frame_count is not None and _orig_accepts_frame_count():
        kwargs["frame_count"] = frame_count
    return _orig_init(
        self, text_len, latent_t, latent_h, latent_w, audio_t, **kwargs
    )


def _patched_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                  keyframes=None, refs=None, frame_count=None):
    marked = bool(keyframes) and any(kf.get(MC_KEY) is not None for kf in keyframes)
    if not marked:
        return _call_orig(
            self, text_len, latent_t, latent_h, latent_w, audio_t,
            keyframes=keyframes, refs=refs, frame_count=frame_count,
        )

    # Stock H3 only accepts first/last guide positions.  Build our marked guide
    # rows legally at frame 0, then translate only those rows after construction.
    safe = []
    for kf in keyframes:
        item = dict(kf)
        if item.get(MC_KEY) is not None:
            item["resolved_frame_index"] = 0
        safe.append(item)

    _call_orig(
        self, text_len, latent_t, latent_h, latent_w, audio_t,
        keyframes=safe, refs=refs, frame_count=frame_count,
    )

    cond_spans = [(a, b) for a, b, kind in self.segments if kind == "cond"]
    if len(cond_spans) != len(keyframes):
        raise RuntimeError(
            "LongMedia motion context: conditioning/layout segment count mismatch")
    origin = _target_origin(self)
    for (a, b), kf in zip(cond_spans, keyframes):
        p = kf.get(MC_KEY)
        if p is None:
            continue
        wanted = origin + mm.FRAME_RESCALE * float(p)
        current = float(self.position_ids[a, 0])
        self.position_ids[a:b, 0] += (wanted - current)


setattr(_patched_init, PATCH_MARKER, True)


def apply_patch():
    global _orig_init, _applied
    if _applied:
        return True
    current = getattr(mm.PackedLayout, "__init__", None)
    if current is None:
        return False
    # Compatible Motion Context / Contex Loop owner already active.
    if getattr(current, PATCH_MARKER, False):
        _applied = True
        _LOG.info("LongMedia motion context: compatible PackedLayout owner already active")
        return True

    where = getattr(current, "__module__", "") or ""
    home = getattr(mm.PackedLayout, "__module__", "") or ""
    # SolAttn's Morton layout wrapper is known-compatible; wrapping outside it
    # is safe because it returns the same layout object before our fixup runs.
    if where != home and not where.endswith(SOLATTN_LAYOUT_MODULE_SUFFIX):
        _LOG.warning(
            "LongMedia motion context: PackedLayout already wrapped by %r; refusing unknown stack",
            where)
        return False

    _orig_init = current
    mm.PackedLayout.__init__ = _patched_init
    _applied = True
    _LOG.info("LongMedia motion context: marker-gated head-guide PackedLayout patch enabled")
    return True


def is_applied():
    return _applied
