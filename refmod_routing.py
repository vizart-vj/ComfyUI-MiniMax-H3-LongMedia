"""
PackedLayout-aware temporal RefMod routing (authored-frame model).

Architecture contract
---------------------
*  Native PackedLayout reference RoPE cursors are ADJACENT to the target
   timeline; they are position-encoding bookkeeping only and are NEVER used
   as activation windows here.
*  Visibility is derived from AUTHORED ``start_frame``/``end_frame`` on the
   ordered reference block descriptors, evaluated against the target
   ``latent_t`` FRAME_PER_TOKEN cells: cell *k* owns target-timeline frames
   ``[F_k, F_k + FRAME_PER_TOKEN[k % len])`` with ``F`` the cumulative frame
   offset.  The target audio query spans the full target frame window
   ``[0, total_target_frames)``.
*  Members are never inferred from adjacent ref segments.  Callers pass the
   ORDERED NATIVE REF BLOCK DESCRIPTORS (the ``minimax_refs`` dicts, plus
   optional authored ``start_frame``/``end_frame`` and ``strength``); the
   segment table is then signature-walked in packed order::

       text -> keyframes (cond/cond_audio)* ->
       per descriptor:
         image          => one ref_img segment
         audio          => ref_audio when ref_audio_t > 0, else NO segment
         video/video_audio => optional ref_audio (ref_audio_t > 0)
                              + ref_img as ONE member
       -> audio target -> video target

   Every step is validated against the layout signature; any mismatch is a
   deterministic ``ValueError``.
*  Ref / text / keyframe rows AS QUERIES keep stock attention: their query
   interval carries an empty key tuple.  Only the target audio and target
   video query intervals gate RefMod keys, listing the exact active key
   ranges with their per-member strengths — a shape practical for nodes.py
   combined masking.

Public API
----------
    RefModRange            one routed reference block (stable member id)
    RefModRoute            route: query intervals, key masks, cells, members
    FakePackedLayout       deterministic native-order test scaffold
    group_refmod_segments  signature-walk descriptors -> member row ranges
    compute_temporal_gates -> RefModRoute for target-only gating
    refmod_row_mask        flat boolean key-side visibility mask

This module has zero ComfyUI / GPU / torch dependencies.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

__all__ = [
    "RefModRange",
    "RefModRoute",
    "TemporalGateSpec",
    "FakePackedLayout",
    "group_refmod_segments",
    "compute_temporal_gates",
    "merge_attention_routes",
    "resolve_refmod_spec_index",
    "refmod_row_mask",
]

# Native FRAME_PER_TOKEN: frames owned by latent video cell k (k % 5).
_FRAME_PATTERN_DEFAULT = (1, 4, 4, 4, 4)
# Half-open window overlap tolerance for float authored frame values.
_FRAME_EPS = 1e-9

_SEGMENT_KIND_TEXT = "text"
_SEGMENT_KIND_COND = ("cond", "cond_audio")
_SEGMENT_KIND_AUDIO = "audio"
_SEGMENT_KIND_VIDEO = "video"
_REF_SEGMENT_KINDS = frozenset({"ref_img", "ref_audio"})

# Host segment spellings -> canonical PackedLayout segment kinds.
_SEGMENT_KIND_ALIASES = {
    "ref_image": "ref_img",
    "ref_video": "ref_img",
    "ref_img": "ref_img",
    "ref_audio": "ref_audio",
}

# Descriptor kind aliases -> canonical native block kinds.
_BLOCK_KIND_ALIASES = {
    "ref_img": "image",
    "ref_image": "image",
    "picture": "image",
    "ref_audio": "audio",
    "ref_video": "video",
    "video_audio": "video_audio",
}
_BLOCK_KINDS = ("image", "audio", "video", "video_audio")

_SIG_MSG = ("layout.signature must be "
            "(text_len, latent_t, latent_h, latent_w, audio_t)")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _normalize_segment_kind(kind: Any) -> str:
    """Canonicalise a segment kind (tolerates host variant spellings)."""
    k = str(kind).strip().lower()
    return _SEGMENT_KIND_ALIASES.get(k, k)


def _validated_segments(layout: Any) -> list[tuple[int, int, str]]:
    """Validate layout.segments: iterable of (int start, int stop, kind)."""
    segments = getattr(layout, "segments", None)
    if segments is None or isinstance(segments, (str, bytes)) \
            or not hasattr(segments, "__iter__"):
        raise ValueError("layout.segments must be iterable of (start, stop, kind)")
    try:
        iterator = iter(segments)
    except TypeError:
        raise ValueError(
            "layout.segments must be iterable of (start, stop, kind)") from None
    out: list[tuple[int, int, str]] = []
    for i, seg in enumerate(iterator):
        if isinstance(seg, (str, bytes)):
            raise ValueError(
                f"layout.segments[{i}] must be (start, stop, kind), got {seg!r}")
        try:
            a, b, kind = seg
        except (TypeError, ValueError):
            raise ValueError(
                f"layout.segments[{i}] must be (start, stop, kind), got {seg!r}"
            ) from None
        try:
            ai, bi = int(a), int(b)
        except (TypeError, ValueError):
            raise ValueError(
                f"layout.segments[{i}] has non-integer bounds: {a!r}, {b!r}"
            ) from None
        if ai < 0:
            raise ValueError(
                f"Malformed segment range at index {i}: negative start {ai}")
        if bi < ai:
            raise ValueError(
                f"Malformed segment range at index {i}: stop {bi} before start {ai}")
        out.append((ai, bi, str(kind)))
    return out


def _require_signature(layout: Any) -> tuple[int, int, int, int, int]:
    """Validate layout.signature and return it as five non-negative ints."""
    sig = getattr(layout, "signature", None)
    if sig is None:
        raise ValueError(_SIG_MSG)
    try:
        parts = tuple(sig)
    except TypeError:
        raise ValueError(_SIG_MSG) from None
    if len(parts) < 5:
        raise ValueError(_SIG_MSG)
    try:
        values = tuple(int(x) for x in parts[:5])
    except (TypeError, ValueError):
        raise ValueError(_SIG_MSG) from None
    if any(v < 0 for v in values):
        raise ValueError(_SIG_MSG)
    return values  # type: ignore[return-value]


def _frame_rows(latent_h: int, latent_w: int) -> int:
    """Rows of one latent frame = (h // 2) * (w // 2), matching native
    ``_frame_grid`` (axis lengths are ``dim // patch``)."""
    return (latent_h // 2) * (latent_w // 2)


def _validated_pattern(frame_pattern: Sequence[int]) -> tuple[int, ...]:
    try:
        pattern = tuple(int(x) for x in frame_pattern)
    except (TypeError, ValueError):
        raise ValueError(
            "frame_pattern must be a non-empty sequence of positive ints"
        ) from None
    if not pattern or any(x <= 0 for x in pattern):
        raise ValueError(
            "frame_pattern must be a non-empty sequence of positive ints")
    return pattern


def _resolve_blocks(layout: Any, ref_blocks: Any) -> list[Any]:
    blocks = ref_blocks if ref_blocks is not None \
        else getattr(layout, "ref_blocks", None)
    if blocks is None:
        raise ValueError(
            "ordered native ref block descriptors required: pass ref_blocks= "
            "or set layout.ref_blocks (members are never inferred from "
            "adjacent ref segments)")
    if isinstance(blocks, (str, bytes)) or not hasattr(blocks, "__iter__"):
        raise ValueError(
            "ref_blocks must be an iterable of native block descriptors")
    try:
        return list(blocks)
    except TypeError:
        raise ValueError(
            "ref_blocks must be an iterable of native block descriptors") from None


def _block_kind(blk: Any, i: int) -> str:
    if blk is None or not hasattr(blk, "get"):
        raise ValueError(f"ref_blocks[{i}] must be a native block descriptor mapping")
    raw = blk.get("kind", "image")
    k = "" if raw is None else str(raw).strip().lower()
    k = _BLOCK_KIND_ALIASES.get(k, k)
    if k not in _BLOCK_KINDS:
        raise ValueError(
            f"ref_blocks[{i}] has unknown kind {raw!r}; expected one of "
            f"{'/'.join(_BLOCK_KINDS)}")
    return k


def _int_field(blk: Any, key: str, default: int, i: int, minimum: int) -> int:
    raw = blk.get(key, default)
    if raw is None:
        raw = default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"ref_blocks[{i}].{key} must be an integer >= {minimum}") from None
    if isinstance(raw, float) and not float(raw).is_integer():
        raise ValueError(
            f"ref_blocks[{i}].{key} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(
            f"ref_blocks[{i}].{key} must be an integer >= {minimum} (got {value})")
    return value


def _required_int_field(blk: Any, key: str, i: int, minimum: int) -> int:
    if blk.get(key) is None:
        raise ValueError(f"ref_blocks[{i}] requires {key} >= {minimum}")
    return _int_field(blk, key, default=minimum, i=i, minimum=minimum)


def _authored_window(blk: Any, i: int) -> tuple[float, float]:
    """Authored activation window in target-timeline frames.

    Absent fields default to ``[0, +inf)`` (full-span conditioning).
    """
    s_raw = blk.get("start_frame", None)
    e_raw = blk.get("end_frame", None)
    if s_raw is None and e_raw is None:
        return (0.0, math.inf)
    start = 0.0
    end = math.inf
    if s_raw is not None:
        try:
            start = float(s_raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"ref_blocks[{i}].start_frame must be numeric") from None
        if not math.isfinite(start):
            raise ValueError(f"ref_blocks[{i}].start_frame must be finite")
        if start < 0.0:
            raise ValueError(f"ref_blocks[{i}].start_frame must be >= 0")
    if e_raw is not None:
        try:
            end = float(e_raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"ref_blocks[{i}].end_frame must be numeric") from None
        if not math.isfinite(end):
            raise ValueError(f"ref_blocks[{i}].end_frame must be finite")
        if end < start:
            raise ValueError(
                f"ref_blocks[{i}] has inverted frame window: "
                f"start_frame {start} > end_frame {end}")
    return (start, end)


def _strength(blk: Any, i: int) -> float:
    raw = blk.get("strength", 1.0)
    if raw is None:
        raw = 1.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"ref_blocks[{i}].strength must be a number in [0, 1]") from None
    if not (0.0 <= value <= 1.0):
        raise ValueError(
            f"ref_blocks[{i}].strength must be in [0, 1] (got {raw!r})")
    return value


def _frames_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    """Half-open authored-window overlap with float tolerance."""
    return (min(a1, b1) - max(a0, b0)) > _FRAME_EPS


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class RefModRange:
    """One routed reference block: exact packed rows + authored window.

    ``member_index`` is the stable block id and equals the descriptor's
    position in the ordered ``ref_blocks`` list (segmentless descriptors,
    e.g. audio with ref_audio_t == 0, keep their id with an empty row range).
    """
    __slots__ = (
        "member_index", "kind", "row_start", "row_stop",
        "start_frame", "end_frame", "segment_groups", "strength",
    )

    def __init__(
        self,
        member_index: int,
        kind: str,
        row_start: int,
        row_stop: int,
        start_frame: float,
        end_frame: float,
        segment_groups: tuple[tuple[int, int, str], ...] = (),
        strength: float = 1.0,
    ):
        self.member_index = member_index
        self.kind = kind  # canonical block kind: image/audio/video/video_audio
        self.row_start = row_start
        self.row_stop = row_stop
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.segment_groups = segment_groups
        self.strength = strength

    @property
    def rows(self) -> int:
        return self.row_stop - self.row_start

    def __repr__(self) -> str:
        return (
            f"RefModRange(idx={self.member_index}, kind={self.kind!r}, "
            f"rows={self.row_start}:{self.row_stop}, "
            f"frames={self.start_frame}:{self.end_frame}, "
            f"strength={self.strength})"
        )


class RefModRoute:
    """Combined-masking-ready route for target-only RefMod attention.

    ``query_intervals``
        ``((row_start, row_stop, keys), ...)`` sorted by row, tiling the
        whole packed sequence.  ``keys`` is a tuple of
        ``(key_start, key_stop, member_index, strength)`` with the EXACT
        active RefMod key ranges for that query interval.  An empty tuple
        means stock attention (text, keyframes, and RefMod rows as queries).
    ``key_masks``
        ``{(cell_start, cell_stop): {ref_row_start: visible}}`` for each
        target video cell and the target audio span.
    ``members`` / ``cells``
        Routed descriptors and target cells; each cell is
        ``(row_start, row_stop, frame_start, frame_stop, modality)``.
    """
    __slots__ = (
        "query_intervals", "key_masks", "members", "cells",
        "text_span", "video_span", "audio_span",
    )

    def __init__(
        self,
        query_intervals: tuple[tuple[int, int, tuple[tuple[int, int, int, float], ...]], ...],
        key_masks: dict[tuple[int, int], dict[int, bool]],
        members: tuple[RefModRange, ...],
        cells: tuple[tuple[int, int, float, float, str], ...],
        text_span: tuple[int, int],
        video_span: tuple[int, int],
        audio_span: tuple[int, int],
    ):
        self.query_intervals = query_intervals
        self.key_masks = key_masks
        self.members = members
        self.cells = cells
        self.text_span = text_span
        self.video_span = video_span
        self.audio_span = audio_span

    def key_mask_for_cell(self, cell_start: int, cell_stop: int) -> dict[int, bool]:
        """Return {ref_row_start: visible} for a target cell's ref rows."""
        return self.key_masks.get((cell_start, cell_stop), {})


# Backwards-compatible alias: earlier revisions returned this name.
TemporalGateSpec = RefModRoute


# ---------------------------------------------------------------------------
# FakePackedLayout: deterministic native-order scaffold
# ---------------------------------------------------------------------------

class FakePackedLayout:
    """Deterministic PackedLayout scaffold reproducing native segment order:
    text -> keyframes -> refs -> audio -> video.  Validates itself through
    the signature-walk at construction time.
    """

    def __init__(
        self,
        text_len: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
        refs: Sequence[dict] | None = None,
        keyframes: Sequence[dict] | None = None,
        frame_pattern: tuple[int, ...] = _FRAME_PATTERN_DEFAULT,
    ):
        text_len = int(text_len)
        latent_t = int(latent_t)
        latent_h = int(latent_h)
        latent_w = int(latent_w)
        audio_t = int(audio_t)
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        self.frame_pattern = tuple(frame_pattern)
        frame_rows = _frame_rows(latent_h, latent_w)

        segments: list[tuple[int, int, str]] = []
        off = 0

        # Text.
        segments.append((off, off + text_len, _SEGMENT_KIND_TEXT))
        off += text_len

        # Keyframes pack between text and refs (cond then cond_audio each).
        for kf in (keyframes or ()):
            vt = int(kf.get("latent_t", 0) or 0)
            if vt > 0:
                segments.append((off, off + vt * frame_rows, "cond"))
                off += vt * frame_rows
            at = int(kf.get("audio_t", 0) or 0)
            if at > 0:
                segments.append((off, off + at * 2, "cond_audio"))
                off += at * 2

        # Reference blocks, in descriptor order.
        for blk in (refs or ()):
            kind = str(blk.get("kind", "image") or "image").strip().lower()
            kind = _BLOCK_KIND_ALIASES.get(kind, kind)
            rt = int(blk.get("ref_audio_t", 0) or 0)
            raw_h = blk.get("latent_h")
            raw_w = blk.get("latent_w")
            rh = (latent_h if raw_h is None else int(raw_h)) // 2
            rw = (latent_w if raw_w is None else int(raw_w)) // 2
            if kind == "image":
                n = rh * rw
                segments.append((off, off + n, "ref_img"))
                off += n
            elif kind == "audio":
                if rt > 0:
                    segments.append((off, off + rt * 2, "ref_audio"))
                    off += rt * 2
                # rt == 0: no segment at all.
            elif kind in ("video", "video_audio"):
                if rt > 0:
                    segments.append((off, off + rt * 2, "ref_audio"))
                    off += rt * 2
                vt = int(blk.get("latent_t", 0) or 0)
                n = vt * rh * rw
                segments.append((off, off + n, "ref_img"))
                off += n
            # Unknown kinds emit nothing; the signature-walk rejects them.

        # Target audio then target video, always last.
        audio_start = off
        segments.append((off, off + audio_t * 2, _SEGMENT_KIND_AUDIO))
        off += audio_t * 2
        audio_stop = off
        video_start = off
        segments.append((off, off + latent_t * frame_rows, _SEGMENT_KIND_VIDEO))
        off += latent_t * frame_rows
        video_stop = off

        self.segments = segments
        self.seq_len = off
        self.text_span = (segments[0][0], segments[0][1])
        self.audio_span = (audio_start, audio_stop)
        self.video_span = (video_start, video_stop)
        self.ref_blocks = tuple(refs or ())
        # Self-validate through the real signature-walk.
        self.ref_ranges: list[RefModRange] = group_refmod_segments(self)


# ---------------------------------------------------------------------------
# Signature-walk: ordered descriptors -> exact member row ranges
# ---------------------------------------------------------------------------

def _walk_segments(
    segments: list[tuple[int, int, str]],
    blocks: list[Any],
    sig: tuple[int, int, int, int, int],
) -> tuple[list[tuple], tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Walk the packed table in native order; every mismatch raises.

    Returns (records, text_span, audio_span, video_span) where each record
    is (index, kind, row_start, row_stop, start_frame, end_frame, groups,
    strength).
    """
    text_len, latent_t, latent_h, latent_w, audio_t = sig

    # -- phase 1: leading text segment -----------------------------------
    a0, b0, k0 = segments[0]
    if _normalize_segment_kind(k0) != _SEGMENT_KIND_TEXT:
        raise ValueError(
            f"layout must start with the text segment, found {k0!r}")
    if b0 - a0 != text_len:
        raise ValueError(
            f"text segment rows {b0 - a0} != signature text_len {text_len}")
    text_span = (a0, b0)
    cursor = 1

    # -- phase 2: keyframes (cond / cond_audio) pack before refs ----------
    while (cursor < len(segments)
           and _normalize_segment_kind(segments[cursor][2]) in _SEGMENT_KIND_COND):
        cursor += 1

    # -- phase 3: one run of segments per ordered descriptor --------------
    records: list[tuple] = []
    for i, blk in enumerate(blocks):
        kind = _block_kind(blk, i)
        rt = _int_field(blk, "ref_audio_t", 0, i, minimum=0)

        # Expected segment runs: (kind, check, value)
        #   check: "exact" rows == value | "multiple" rows % value == 0
        #          | None -> rows > 0 only
        expected: list[tuple[str, str | None, int | None]] = []
        if kind == "image":
            h = blk.get("latent_h")
            w = blk.get("latent_w")
            if h is None and w is None:
                expected.append(("ref_img", None, None))
            else:
                if h is None or w is None:
                    raise ValueError(
                        f"ref_blocks[{i}] needs both latent_h and latent_w")
                try:
                    hi, wi = int(h), int(w)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"ref_blocks[{i}] latent_h/latent_w must be integers"
                    ) from None
                if hi < 0 or wi < 0:
                    raise ValueError(
                        f"ref_blocks[{i}] latent_h/latent_w must be >= 0")
                expected.append(("ref_img", "exact", _frame_rows(hi, wi)))
        elif kind == "audio":
            if rt > 0:
                expected.append(("ref_audio", "exact", rt * 2))
            # rt == 0 => consumes no segment.
        else:  # video / video_audio: optional ref_audio + ref_img, one member
            vt = _required_int_field(blk, "latent_t", i, minimum=1)
            if rt > 0:
                expected.append(("ref_audio", "exact", rt * 2))
            h = blk.get("latent_h")
            w = blk.get("latent_w")
            if h is None and w is None:
                expected.append(("ref_img", "multiple", vt))
            else:
                if h is None or w is None:
                    raise ValueError(
                        f"ref_blocks[{i}] needs both latent_h and latent_w")
                try:
                    hi, wi = int(h), int(w)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"ref_blocks[{i}] latent_h/latent_w must be integers"
                    ) from None
                if hi < 0 or wi < 0:
                    raise ValueError(
                        f"ref_blocks[{i}] latent_h/latent_w must be >= 0")
                expected.append(("ref_img", "exact", vt * _frame_rows(hi, wi)))

        groups: list[tuple[int, int, str]] = []
        for exp_kind, check, value in expected:
            if cursor >= len(segments):
                raise ValueError(
                    f"ref_blocks[{i}] ({kind}) expects a {exp_kind} segment, "
                    f"but layout.segments ended")
            sa, sb, sk = segments[cursor]
            nk = _normalize_segment_kind(sk)
            if nk != exp_kind:
                raise ValueError(
                    f"ref_blocks[{i}] ({kind}) expects {exp_kind} segment, "
                    f"found {nk!r} at rows {sa}:{sb}")
            rows = sb - sa
            if rows <= 0:
                raise ValueError(
                    f"ref segment rows must be > 0 (rows {sa}:{sb} of "
                    f"ref_blocks[{i}])")
            if check == "exact" and rows != value:
                raise ValueError(
                    f"ref_blocks[{i}] {exp_kind} rows {rows} != expected {value}")
            if check == "multiple" and rows % int(value) != 0:
                raise ValueError(
                    f"ref_blocks[{i}] {exp_kind} rows {rows} not divisible by "
                    f"latent_t {value}")
            groups.append((sa, sb, nk))
            cursor += 1

        if groups:
            row_start = min(g[0] for g in groups)
            row_stop = max(g[1] for g in groups)
        else:
            # Segmentless member (e.g. audio with rt == 0): anchored where
            # it would have been packed, id preserved.
            anchor = segments[cursor - 1][1] if cursor > 0 else 0
            row_start = row_stop = anchor
        start_f, end_f = _authored_window(blk, i)
        strength = _strength(blk, i)
        records.append((i, kind, row_start, row_stop, start_f, end_f,
                        tuple(groups), strength))

    # -- phase 4: exactly the audio then the video target segment ---------
    if cursor >= len(segments):
        raise ValueError("layout.segments missing 'audio' target segment")
    a, b, k = segments[cursor]
    nk = _normalize_segment_kind(k)
    if nk in _REF_SEGMENT_KINDS:
        raise ValueError(
            f"unexpected {nk} segment at rows {a}:{b}: more ref segments "
            f"than ref_blocks descriptors")
    if nk != _SEGMENT_KIND_AUDIO:
        raise ValueError(
            f"layout.segments missing 'audio' target segment (found {k!r})")
    if b - a != audio_t * 2:
        raise ValueError(
            f"audio segment rows {b - a} != signature audio_t * 2 ({audio_t * 2})")
    audio_span = (a, b)
    cursor += 1

    if cursor >= len(segments):
        raise ValueError("layout.segments missing 'video' target segment")
    a, b, k = segments[cursor]
    nk = _normalize_segment_kind(k)
    if nk in _REF_SEGMENT_KINDS:
        raise ValueError(
            f"unexpected {nk} segment at rows {a}:{b}: more ref segments "
            f"than ref_blocks descriptors")
    if nk != _SEGMENT_KIND_VIDEO:
        raise ValueError(
            f"layout.segments missing 'video' target segment (found {k!r})")
    expected_video = latent_t * _frame_rows(latent_h, latent_w)
    if b - a != expected_video:
        raise ValueError(
            f"video segment rows {b - a} != signature latent_t * frame_rows "
            f"({expected_video})")
    video_span = (a, b)
    cursor += 1
    if cursor < len(segments):
        raise ValueError(
            f"unexpected segment {segments[cursor]!r} after the video "
            f"target segment")

    return records, text_span, audio_span, video_span


def _analyze(
    layout: Any, ref_blocks: Any = None,
) -> tuple[tuple[RefModRange, ...],
           tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
           list[tuple[int, int, str]],
           tuple[int, int, int, int, int]]:
    """Validate layout + descriptors once; shared by all public entry points."""
    segments = _validated_segments(layout)
    if not segments:
        raise ValueError("layout.segments is empty")
    sig = _require_signature(layout)
    blocks = _resolve_blocks(layout, ref_blocks)
    records, text_span, audio_span, video_span = _walk_segments(
        segments, blocks, sig)
    members = tuple(
        RefModRange(
            member_index=i,
            kind=kind,
            row_start=row_start,
            row_stop=row_stop,
            start_frame=start_f,
            end_frame=end_f,
            segment_groups=groups,
            strength=strength,
        )
        for (i, kind, row_start, row_stop, start_f, end_f, groups, strength)
        in records
    )
    return members, (text_span, audio_span, video_span), segments, sig


def group_refmod_segments(layout: Any, ref_blocks: Any = None) -> list[RefModRange]:
    """Signature-walk layout.segments against the ordered native ref block
    descriptors and return one RefModRange per descriptor.

    *  member ids are stable: ``member_index`` == descriptor position.
    *  ``segment_groups`` are the exact packed (start, stop, kind) ranges.
    *  audio descriptors with ``ref_audio_t == 0`` consume no segment but
       still produce a (rowless) member so ids never shift.
    *  keyframe (cond/cond_audio) segments before refs are skipped.
    *  every malformed structure raises a deterministic ValueError.
    """
    members, _spans, _segments, _sig = _analyze(layout, ref_blocks)
    return list(members)


# ---------------------------------------------------------------------------
# Target cells: authored-frame windows over FRAME_PER_TOKEN
# ---------------------------------------------------------------------------

def _build_cells(
    video_span: tuple[int, int],
    audio_span: tuple[int, int],
    sig: tuple[int, int, int, int, int],
    pattern: tuple[int, ...],
) -> tuple[list[tuple[int, int, float, float, str]], int]:
    """Target cells as (row_start, row_stop, frame_start, frame_stop, kind).

    Video cell k owns frames [F_k, F_k + pattern[k % len]) with F the
    cumulative frame offset (native FRAME_PER_TOKEN).  The target audio span
    is one cell over the full target frame window [0, total_frames).
    """
    _text_len, latent_t, _h, _w, _audio_t = sig
    video_a, video_b = video_span
    audio_a, audio_b = audio_span
    cells: list[tuple[int, int, float, float, str]] = []
    total_frames = 0
    if latent_t > 0 and video_b > video_a:
        frame_rows = (video_b - video_a) // latent_t  # exact per walk
        frame = 0
        for t in range(latent_t):
            fr = pattern[t % len(pattern)]
            row_a = video_a + t * frame_rows
            cells.append((row_a, row_a + frame_rows,
                          float(frame), float(frame + fr), "video"))
            frame += fr
        total_frames = frame
    if audio_b > audio_a:
        cells.append((audio_a, audio_b, 0.0, float(total_frames), "audio"))
    return cells, total_frames


def _keys_for_window(
    members: tuple[RefModRange, ...],
    frame_start: float,
    frame_stop: float,
) -> tuple[tuple[int, int, int, float], ...]:
    """Exact active key ranges (start, stop, member_index, strength)."""
    keys: list[tuple[int, int, int, float]] = []
    for member in members:
        if _frames_overlap(member.start_frame, member.end_frame,
                           frame_start, frame_stop):
            for gs, ge, _kind in member.segment_groups:
                keys.append((gs, ge, member.member_index, member.strength))
    keys.sort(key=lambda k: (k[0], k[1], k[2]))
    return tuple(keys)


# ---------------------------------------------------------------------------
# Public: compute_temporal_gates -> RefModRoute
# ---------------------------------------------------------------------------

def compute_temporal_gates(
    layout: Any,
    ref_blocks: Any = None,
    frame_pattern: Sequence[int] = _FRAME_PATTERN_DEFAULT,
    expected_frame_count: int | None = None,
) -> RefModRoute:
    """Build the target-only RefMod attention route.

    Target video cell queries and the target audio query receive the exact
    RefMod key ranges whose AUTHORED ``start_frame``/``end_frame`` window
    overlaps their target-timeline frame window; every other query (text,
    keyframes, RefMod rows themselves) keeps stock attention — an empty key
    tuple.  Each key range carries ``(start, stop, member_index, strength)``
    so nodes.py can fold it directly into a combined attention mask.
    """
    members, spans, segments, sig = _analyze(layout, ref_blocks)
    pattern = _validated_pattern(frame_pattern)
    text_span, audio_span, video_span = spans
    cells, total_frames = _build_cells(video_span, audio_span, sig, pattern)
    if expected_frame_count is not None:
        expected = int(expected_frame_count)
        if expected < 0 or total_frames != expected:
            raise ValueError(
                "RefMod authored target clock does not match the native H3 lattice: "
                f"authored={expected}, native={total_frames}, latent_t={sig[1]}."
            )

    key_masks: dict[tuple[int, int], dict[int, bool]] = {}
    cell_keys: dict[tuple[int, int],
                    tuple[tuple[int, int, int, float], ...]] = {}
    for row_a, row_b, frame_a, frame_b, _modality in cells:
        mask: dict[int, bool] = {}
        for member in members:
            active = _frames_overlap(member.start_frame, member.end_frame,
                                     frame_a, frame_b)
            for gs, ge, _kind in member.segment_groups:
                mask[gs] = active
                mask[ge - 1] = active  # sentinel for the group's last row
        key_masks[(row_a, row_b)] = mask
        cell_keys[(row_a, row_b)] = _keys_for_window(members, frame_a, frame_b)

    # Everything except the trailing audio/video targets keeps stock
    # attention; the walk guarantees those are the last two segments.
    intervals: list[tuple[int, int,
                           tuple[tuple[int, int, int, float], ...]]] = [
        (a, b, ()) for (a, b, _k) in segments[:-2]
    ]
    audio_a, audio_b = audio_span
    if audio_b > audio_a:
        intervals.append((audio_a, audio_b, cell_keys[(audio_a, audio_b)]))
    for row_a, row_b, _fa, _fb, modality in cells:
        if modality == "video":
            intervals.append((row_a, row_b, cell_keys[(row_a, row_b)]))
    intervals.sort(key=lambda iv: (iv[0], iv[1]))

    return RefModRoute(
        query_intervals=tuple(intervals),
        key_masks=key_masks,
        members=members,
        cells=tuple(cells),
        text_span=text_span,
        video_span=video_span,
        audio_span=audio_span,
    )


def merge_attention_routes(
    seq_len: int,
    *,
    always_hidden: Sequence[tuple[int, int]] = (),
    temporal_intervals: Sequence[
        tuple[int, int, Sequence[tuple[int, int, float]]]
    ] = (),
    refmod_route: RefModRoute | None = None,
) -> tuple[
    tuple[
        int,
        int,
        tuple[tuple[int, int], ...],
        tuple[tuple[int, int, float], ...],
    ],
    ...,
]:
    """Merge temporal text-key and RefMod-key routing into one query partition.

    Each returned row is ``(q_start, q_stop, hidden_keys, active_keys)``.
    ``hidden_keys`` are set to negative infinity first; ``active_keys`` then
    restore the same key rows with either zero bias (strength 1) or
    ``log(strength)``.  RefMod rows are hidden only for target audio/video
    queries.  Text, condition and reference rows therefore retain stock native
    attention exactly as required by :class:`RefModRoute`.

    The boundary union guarantees that every packed query row is evaluated
    exactly once even when embedding windows and target latent cells do not
    share boundaries.
    """
    try:
        seq = int(seq_len)
    except (TypeError, ValueError):
        raise ValueError("seq_len must be a non-negative integer") from None
    if seq < 0:
        raise ValueError("seq_len must be a non-negative integer")

    def _key_range(raw: Any, *, weighted: bool) -> tuple:
        try:
            a, b = int(raw[0]), int(raw[1])
        except (TypeError, ValueError, IndexError):
            raise ValueError("attention key ranges must contain integer start/stop") from None
        if a < 0 or b <= a or b > seq:
            raise ValueError(f"attention key range {a}:{b} is outside 0:{seq}")
        if not weighted:
            return (a, b)
        try:
            weight = float(raw[2])
        except (TypeError, ValueError, IndexError):
            raise ValueError("active attention key ranges require a numeric weight") from None
        if not math.isfinite(weight) or not (0.0 <= weight <= 1.0):
            raise ValueError(f"attention key weight must be in [0, 1] (got {weight!r})")
        return (a, b, weight)

    hidden_base = tuple(sorted({_key_range(v, weighted=False) for v in always_hidden}))
    temporal = []
    boundaries = {0, seq}
    for raw in temporal_intervals or ():
        try:
            qa, qb, active = raw
            qa, qb = int(qa), int(qb)
        except (TypeError, ValueError):
            raise ValueError("temporal intervals must be (q_start, q_stop, active_keys)") from None
        if qa < 0 or qb <= qa or qb > seq:
            raise ValueError(f"temporal query interval {qa}:{qb} is outside 0:{seq}")
        keys = tuple(sorted({_key_range(v, weighted=True) for v in (active or ())}))
        temporal.append((qa, qb, keys))
        boundaries.update((qa, qb))

    ref_cells: list[tuple[int, int, tuple[tuple[int, int, float], ...]]] = []
    all_ref_ranges: tuple[tuple[int, int], ...] = ()
    if refmod_route is not None:
        ranges = {
            (int(gs), int(ge))
            for member in refmod_route.members
            for gs, ge, _kind in member.segment_groups
            if int(ge) > int(gs)
        }
        all_ref_ranges = tuple(sorted(ranges))
        interval_keys = {
            (int(a), int(b)): tuple((int(k0), int(k1), float(weight))
                                    for k0, k1, _member, weight in keys)
            for a, b, keys in refmod_route.query_intervals
        }
        for cell in refmod_route.cells:
            qa, qb = int(cell[0]), int(cell[1])
            if qa < 0 or qb <= qa or qb > seq:
                raise ValueError(f"RefMod query cell {qa}:{qb} is outside 0:{seq}")
            ref_cells.append((qa, qb, interval_keys.get((qa, qb), ())))
            boundaries.update((qa, qb))

    ordered = sorted(boundaries)
    merged = []
    for qa, qb in zip(ordered, ordered[1:]):
        if qb <= qa:
            continue
        hidden = set(hidden_base)
        active: set[tuple[int, int, float]] = set()
        for ta, tb, keys in temporal:
            if ta <= qa and qb <= tb:
                active.update(keys)
        for ca, cb, keys in ref_cells:
            if ca <= qa and qb <= cb:
                hidden.update(all_ref_ranges)
                active.update(keys)
        merged.append((qa, qb, tuple(sorted(hidden)), tuple(sorted(active))))

    if not merged and seq == 0:
        return ()
    cursor = 0
    for qa, qb, _hidden, _active in merged:
        if qa != cursor:
            raise RuntimeError(
                f"combined attention route is not contiguous: expected {cursor}, got {qa}:{qb}"
            )
        cursor = qb
    if cursor != seq:
        raise RuntimeError(
            f"combined attention route does not cover sequence: covered={cursor}, seq={seq}"
        )
    return tuple(merged)


def resolve_refmod_spec_index(ref_blocks: Sequence[dict], spec: dict) -> int:
    """Resolve one authored spec to the exact native minimax_refs descriptor."""
    blocks = list(ref_blocks or ())
    item = dict(spec or {})
    refmod_id = str(item.get("refmod_id") or "").strip()
    if refmod_id:
        member_index = int(item.get("member_index", 0) or 0)
        appended_index = int(item.get("appended_block_index", 0) or 0)
        matches = [
            index for index, block in enumerate(blocks)
            if str(block.get("longmedia_refmod_id") or "").strip() == refmod_id
            and int(block.get("longmedia_refmod_member_index", 0) or 0) == member_index
            and int(block.get("longmedia_refmod_appended_block_index", 0) or 0) == appended_index
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(
                "RefMod descriptor identity not found in native minimax_refs: "
                f"id={refmod_id!r}, member={member_index}, appended={appended_index}."
            )
        raise ValueError(
            "RefMod descriptor identity is ambiguous in native minimax_refs: "
            f"id={refmod_id!r}, member={member_index}, appended={appended_index}, "
            f"matches={matches}."
        )

    try:
        index = int(item.get("ref_index", -1))
    except (TypeError, ValueError):
        index = -1
    if index < 0 or index >= len(blocks):
        raise ValueError(
            "Legacy RefMod descriptor index is outside native minimax_refs: "
            f"index={index}, refs={len(blocks)}."
        )
    return index


# ---------------------------------------------------------------------------
# Public: refmod_row_mask — flat boolean key-side visibility
# ---------------------------------------------------------------------------

def refmod_row_mask(
    layout: Any,
    ref_blocks: Any = None,
    target_cell: tuple[int, int] | None = None,
    frame_pattern: Sequence[int] = _FRAME_PATTERN_DEFAULT,
) -> list[bool]:
    """Boolean mask of length ``seq_len``: True for stock-visible rows,
    and for RefMod rows that are ACTIVE for the requested target query.

    *  ``target_cell=None``: a ref member is visible if its authored window
       overlaps ANY target cell (union view).
    *  ``target_cell=(row_start, row_stop)``: the frame window is the union
       of the covered target cells (exact cell, sub-range, or full-span
       query all work); rows outside every target cell hide all refs.
    """
    members, spans, segments, sig = _analyze(layout, ref_blocks)
    pattern = _validated_pattern(frame_pattern)
    _text_span, audio_span, video_span = spans

    max_stop = max((b for _a, b, _k in segments), default=0)
    seq = getattr(layout, "seq_len", None)
    if seq is None:
        seq = max_stop
    else:
        try:
            seq = int(seq)
        except (TypeError, ValueError):
            raise ValueError("layout.seq_len must be an integer") from None
    if seq < 0:
        raise ValueError(f"layout.seq_len must be >= 0 (got {seq})")
    if seq < max_stop:
        raise ValueError(
            f"layout.seq_len {seq} is shorter than packed segments ({max_stop})")
    mask = [True] * seq

    if not any(member.segment_groups for member in members):
        return mask  # nothing routable: stock visibility everywhere

    cells, _total = _build_cells(video_span, audio_span, sig, pattern)

    if target_cell is None:
        def _visible(member: RefModRange) -> bool:
            return any(
                _frames_overlap(member.start_frame, member.end_frame,
                                cell[2], cell[3])
                for cell in cells)
    else:
        try:
            tc0 = int(target_cell[0])
            tc1 = int(target_cell[1])
        except (TypeError, ValueError, IndexError):
            raise ValueError(
                "target_cell must be (row_start, row_stop)") from None
        if tc1 < tc0:
            raise ValueError(
                f"Malformed target_cell: stop {tc1} before start {tc0}")
        covering = [cell for cell in cells if cell[0] < tc1 and tc0 < cell[1]]
        if covering:
            frame_start = min(cell[2] for cell in covering)
            frame_stop = max(cell[3] for cell in covering)

            def _visible(member: RefModRange) -> bool:
                return _frames_overlap(member.start_frame, member.end_frame,
                                       frame_start, frame_stop)
        else:
            def _visible(member: RefModRange) -> bool:
                return False  # query touches no target cell: refs hidden

    for member in members:
        visible = _visible(member)
        for gs, ge, _kind in member.segment_groups:
            for row in range(gs, ge):
                mask[row] = visible
    return mask
