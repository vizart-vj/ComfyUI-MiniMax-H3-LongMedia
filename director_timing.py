"""Director clocks and temporal-control routing contracts (no Comfy/GPU dependency).

The common scene presentation is encoded once. Partial CAMERA and EMBEDDING
controls own explicit appended text rows and are exposed only to the intended
target-video temporal groups. Camera controls can use a short weighted handoff;
embedding timing follows native H3 temporal-cell overlap. The model's learned
response and VAE temporal receptive field are deliberately not claimed to be
frame-exact.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence


def conditioning_regions(
    duration: float,
    cameras: Sequence[Mapping[str, Any]],
    embeddings: Sequence[Mapping[str, Any]],
    compiler: Callable[..., str] | None,
) -> list[dict[str, Any]]:
    """Half-open intervals; latest camera start wins, then document order."""
    points = {0.0, float(duration)}
    for item in (*cameras, *embeddings):
        for key in ("start_seconds", "end_seconds"):
            value = float(item[key])
            if not math.isfinite(value):
                raise ValueError("Director interval endpoints must be finite.")
            points.add(max(0.0, min(duration, value)))
    ordered = sorted(points)
    result: list[dict[str, Any]] = []
    for a, b in zip(ordered, ordered[1:]):
        if b <= a:
            continue
        active_cameras = [(i, c) for i, c in enumerate(cameras)
                          if float(c["start_seconds"]) <= a < float(c["end_seconds"])]
        winner = max(active_cameras, key=lambda pair: (float(pair[1]["start_seconds"]), pair[0]))[1] if active_cameras else None
        camera = dict(winner["camera"]) if winner else None
        note = str(winner.get("note") or "").strip() if winner else ""
        if camera is not None:
            camera["transition_to_next"] = False
            # Compiler-only trajectory metadata.  Conditioning regions may be
            # split by EMBEDDING boundaries, so camera progress must be measured
            # against the original authored CAMERA block rather than restarted
            # for every derived region.
            cam_start = float(winner["start_seconds"])
            cam_end = float(winner["end_seconds"])
            cam_duration = max(1e-9, cam_end - cam_start)
            camera["director_interval_seconds"] = float(b - a)
            camera["director_camera_block_duration_seconds"] = float(cam_duration)
            camera["director_trajectory_phase_start"] = max(0.0, min(1.0, (float(a) - cam_start) / cam_duration))
            camera["director_trajectory_phase_end"] = max(0.0, min(1.0, (float(b) - cam_start) / cam_duration))
        instruction = compiler(camera, None, None) if compiler and camera is not None else ""
        instruction = " ".join(x for x in (instruction, note) if x)
        names = list(dict.fromkeys(str(e["name"]) for e in embeddings
                                   if float(e["start_seconds"]) <= a < float(e["end_seconds"])))
        text = "\n\n".join(x for x in (instruction, "\n".join("embedding:" + n for n in names)) if x)
        if result and result[-1]["prompt"] == text:
            result[-1]["end_seconds"] = b
            continue
        result.append({"start_seconds": a, "end_seconds": b, "prompt": text,
                       "camera": camera, "camera_instruction": instruction, "embeddings": names})
    return result


def conditioning_controls(
    duration: float,
    cameras: Sequence[Mapping[str, Any]],
    embeddings: Sequence[Mapping[str, Any]],
    compiler: Callable[..., str] | None,
    *,
    camera_handoff_seconds: float = 8.0 / 24.0,
) -> list[dict[str, Any]]:
    """Compile factorized Director controls without duplicating the scene prompt.

    Camera blocks and embeddings stay independent even when their authored
    intervals overlap.  This is intentionally different from
    :func:`conditioning_regions`: an EMBEDDING boundary must not manufacture a
    new camera presentation, and a camera boundary must not clone the complete
    base prompt.  Adjacent camera blocks get a short centred handoff window; the
    authored cut time is the 50/50 point rather than a semantic scene reset.
    """
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("Director duration must be finite and positive.")
    handoff = max(0.0, float(camera_handoff_seconds))

    camera_items: list[dict[str, Any]] = []
    camera_points = {0.0, duration}
    normalized_cameras: list[tuple[int, Mapping[str, Any], float, float]] = []
    for index, raw in enumerate(cameras):
        start = max(0.0, min(duration, float(raw["start_seconds"])))
        end = max(start, min(duration, float(raw["end_seconds"])))
        if end <= start + 1e-9:
            continue
        normalized_cameras.append((index, raw, start, end))
        camera_points.add(start)
        camera_points.add(end)

    # Preserve Director's established overlap rule: latest authored camera start
    # wins, ties use document order.  Unlike 0.6.18, each resolved interval only
    # carries camera-control semantics; it never carries a cloned scene prompt.
    resolved_index = 0
    ordered_points = sorted(camera_points)
    for a, b in zip(ordered_points, ordered_points[1:]):
        if b <= a:
            continue
        active = [item for item in normalized_cameras if item[2] <= a < item[3]]
        if not active:
            continue
        source_index, raw, source_start, source_end = max(active, key=lambda item: (item[2], item[0]))
        source_duration = max(1e-9, source_end - source_start)
        card = dict(raw.get("camera") or {})
        card["transition_to_next"] = False
        card["director_interval_seconds"] = float(b - a)
        card["director_camera_block_duration_seconds"] = float(source_duration)
        card["director_trajectory_phase_start"] = max(0.0, min(1.0, (a - source_start) / source_duration))
        card["director_trajectory_phase_end"] = max(0.0, min(1.0, (b - source_start) / source_duration))
        instruction = compiler(card, None, None) if compiler else ""
        note = str(raw.get("note") or "").strip()
        prompt = " ".join(x for x in (instruction, note) if x)
        camera_items.append({
            "kind": "camera_control",
            "name": f"camera:{raw.get('block_id') or source_index}:{resolved_index}",
            "order": resolved_index,
            "source_order": source_index,
            "start_seconds": float(a),
            "end_seconds": float(b),
            "prompt": prompt,
            "camera": card,
            "camera_instruction": instruction,
            "handoff_in_seconds": 0.0,
            "handoff_out_seconds": 0.0,
        })
        resolved_index += 1

    # Only truly adjacent authored camera blocks cross-fade.  Gaps remain gaps;
    # otherwise a camera instruction would leak into an interval the user left
    # intentionally unowned.
    for index, item in enumerate(camera_items):
        span = max(0.0, float(item["end_seconds"]) - float(item["start_seconds"]))
        if index > 0:
            prev = camera_items[index - 1]
            if abs(float(prev["end_seconds"]) - float(item["start_seconds"])) <= 1e-6:
                width = min(handoff, span * 0.25,
                            max(0.0, float(prev["end_seconds"]) - float(prev["start_seconds"])) * 0.25)
                item["handoff_in_seconds"] = width
                prev["handoff_out_seconds"] = width

    out: list[dict[str, Any]] = list(camera_items)
    for index, raw in enumerate(embeddings):
        start = max(0.0, min(duration, float(raw["start_seconds"])))
        end = max(start, min(duration, float(raw["end_seconds"])))
        if end <= start + 1e-9:
            continue
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        out.append({
            "kind": "embedding_control",
            "name": f"embedding:{index}:{name}",
            "embedding_name": name,
            "order": index,
            "start_seconds": start,
            "end_seconds": end,
            "prompt": "embedding:" + name,
            "handoff_in_seconds": 0.0,
            "handoff_out_seconds": 0.0,
        })

    out.sort(key=lambda item: (
        float(item["start_seconds"]),
        0 if item["kind"] == "camera_control" else 1,
        int(item.get("order", 0)),
        str(item["name"]),
    ))
    return out


def control_frame_spans(
    controls: Sequence[Mapping[str, Any]], *, context_start: int,
    segment_length: int, shot_start_seconds: float, shot_duration: float,
    fps: int = 24,
) -> list[dict[str, Any]]:
    """Map overlapping factorized controls onto one local H3 segment clock."""
    result: list[dict[str, Any]] = []
    epsilon = 0.5 / max(1.0, float(fps))
    for raw in controls:
        start_s = max(0.0, min(float(shot_duration), float(raw["start_seconds"])))
        end_s = max(start_s, min(float(shot_duration), float(raw["end_seconds"])))
        a = round((float(shot_start_seconds) + start_s) * fps) - int(context_start)
        b = round((float(shot_start_seconds) + end_s) * fps) - int(context_start)
        if start_s <= epsilon:
            a = 0
        if end_s >= float(shot_duration) - epsilon:
            b = int(segment_length)
        item = dict(raw)
        item["start_frame"] = max(0, min(int(segment_length), int(a)))
        item["end_frame"] = max(item["start_frame"], min(int(segment_length), int(b)))
        item["handoff_in_frames"] = max(0, int(round(float(raw.get("handoff_in_seconds") or 0.0) * fps)))
        item["handoff_out_frames"] = max(0, int(round(float(raw.get("handoff_out_seconds") or 0.0) * fps)))
        if item["end_frame"] > item["start_frame"]:
            result.append(item)
    return result


def temporal_control_attention_route(
    layout: Any,
    specs: Sequence[Mapping[str, Any]],
    seq_len: int,
    frame_pattern: Sequence[int] = (1, 4, 4, 4, 4),
) -> dict[str, Any]:
    """Build a factorized, optionally feathered temporal-control route.

    Common text/reference queries never read temporal-control rows.  Each
    control's own text queries read common + self.  Target video queries read
    common plus the active control set; target audio stays common-only.  Camera controls use a short
    centred handoff; embeddings remain exact apart from unavoidable native H3
    temporal-cell overlap.
    """
    segments = [(int(a), int(b), str(k)) for a, b, k in layout.segments]
    text = next((a, b) for a, b, k in segments if k == "text")
    video = next((a, b) for a, b, k in segments if k == "video")
    sig = layout.signature
    latent_t = int(sig[1])
    if latent_t <= 0 or (video[1] - video[0]) % latent_t:
        raise ValueError("Director temporal controls: incompatible target video geometry.")
    if max(b for _, b, _ in segments) != seq_len:
        raise ValueError("Director temporal controls: packed sequence size mismatch.")
    frame_rows = (video[1] - video[0]) // latent_t

    row_ranges: dict[str, tuple[int, int]] = {}
    labels = [""] * seq_len
    normalized: list[dict[str, Any]] = []
    for raw in specs:
        name = str(raw["name"])
        a = text[0] + int(raw["text_start"])
        b = text[0] + int(raw["text_stop"])
        if not text[0] <= a < b <= text[1]:
            raise ValueError(f"Director conditioning rows outside text context: {name} {a}:{b}.")
        if any(labels[a:b]):
            raise ValueError("Director temporal-control text regions overlap.")
        labels[a:b] = [name] * (b - a)
        row_ranges[name] = (a, b)
        item = dict(raw)
        item["start_frame"] = int(item.get("start_frame", 0) or 0)
        item["end_frame"] = int(item.get("end_frame", 0) or 0)
        item["handoff_in_frames"] = int(item.get("handoff_in_frames", 0) or 0)
        item["handoff_out_frames"] = int(item.get("handoff_out_frames", 0) or 0)
        normalized.append(item)

    def weight(item: Mapping[str, Any], a: float, b: float) -> float:
        start = float(item["start_frame"])
        end = float(item["end_frame"])
        if b <= a:
            return 0.0
        mid = 0.5 * (a + b)
        kind = str(item.get("kind") or "")
        if kind == "camera_control":
            fade_in = float(item.get("handoff_in_frames") or 0.0)
            fade_out = float(item.get("handoff_out_frames") or 0.0)
            w = 1.0 if start <= mid < end else 0.0
            if fade_in > 0.0:
                lo, hi = start - fade_in * 0.5, start + fade_in * 0.5
                if lo < mid < hi:
                    w = (mid - lo) / max(1e-9, hi - lo)
                elif mid <= lo:
                    w = 0.0
            if fade_out > 0.0:
                lo, hi = end - fade_out * 0.5, end + fade_out * 0.5
                if lo < mid < hi:
                    w = min(w if mid < end else 1.0, (hi - mid) / max(1e-9, hi - lo))
                elif mid >= hi:
                    w = 0.0
            return max(0.0, min(1.0, w))

        # Exact embedding semantics.  A native temporal token can straddle an
        # authored boundary, so use the geometric overlap fraction of that one
        # token instead of arbitrarily assigning the whole cell to either side.
        overlap = max(0.0, min(b, end) - max(a, start))
        return max(0.0, min(1.0, overlap / max(1e-9, b - a)))

    cursor = 0
    video_controls: list[tuple[tuple[str, float], ...]] = []
    for tidx in range(latent_t):
        end = cursor + int(frame_pattern[tidx % len(frame_pattern)])
        active = tuple(
            (str(item["name"]), float(w))
            for item in normalized
            if (w := weight(item, cursor, end)) > 1e-6
        )
        video_controls.append(active)
        cursor = end

    # Query intervals are compressed by equal active sets/weights; every query
    # row is still evaluated exactly once. Build them in PackedLayout segment
    # order so target-audio rows are never accidentally covered twice by a
    # generic pre-video interval.
    query_intervals: list[tuple[int, int, tuple[tuple[str, float], ...]]] = []

    # Text: common rows read common only; each control presentation reads
    # common + itself.
    control_text = sorted((a, b, name) for name, (a, b) in row_ranges.items())
    qcursor = text[0]
    for a, b, name in control_text:
        if qcursor < a:
            query_intervals.append((qcursor, a, tuple()))
        query_intervals.append((a, b, ((name, 1.0),)))
        qcursor = b
    if qcursor < text[1]:
        query_intervals.append((qcursor, text[1], tuple()))

    audio_t = int(sig[4])
    for seg_a, seg_b, kind in segments:
        if kind == "text":
            continue
        if kind == "video":
            group_start = 0
            group_active = video_controls[0] if video_controls else tuple()
            for tidx in range(1, latent_t + 1):
                current = video_controls[tidx] if tidx < latent_t else None
                if current != group_active:
                    query_intervals.append((
                        seg_a + group_start * frame_rows,
                        seg_a + tidx * frame_rows,
                        group_active,
                    ))
                    group_start = tidx
                    group_active = current
            continue
        if kind == "audio":
            if audio_t <= 0 or (seg_b - seg_a) % audio_t:
                raise ValueError("Director temporal controls: incompatible channel-major audio geometry.")
            # Camera/visual-embedding controls are video authority.  Keeping the
            # target-audio query stream common-only avoids manufacturing audio
            # regime boundaries and preserves the forced lip-sync clock.  Video
            # can still interact with audio through the ordinary joint stream.
            query_intervals.append((seg_a, seg_b, tuple()))
            continue
        # Reference/keyframe/cond rows remain common-only.
        query_intervals.append((seg_a, seg_b, tuple()))

    query_intervals.sort(key=lambda x: x[0])
    return {
        "temporal_controls": True,
        "regional": True,
        "row_ranges": row_ranges,
        "query_intervals": tuple(query_intervals),
        "key_ranges": tuple((a, b, name) for name, (a, b) in row_ranges.items()),
        "decoded_frames": cursor,
        "latent_t": latent_t,
        "text_span": text,
        "video_span": video,
    }


def native_timeline_geometry(clips: Sequence[Mapping[str, Any]], overlap: int, fps: int = 24) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Round cumulative boundaries, never each duration independently.

    The joined H3 stream is 5+17*k frames; continuation visible lengths are
    multiples of 17. Only the terminal padding is trimmed by the decoder.
    """
    if overlap < 5 or (overlap - 5) % 17:
        raise ValueError("Director overlap must be 5+17*k frames.")
    ends: list[int] = []
    seconds = 0.0
    for i, clip in enumerate(clips):
        duration = float(clip["duration"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Director duration must be finite and positive.")
        seconds += duration
        target = seconds * fps
        k = math.ceil((round(target) - 5) / 17) if i == len(clips) - 1 else math.floor((target - 5) / 17 + 0.5)
        end = 5 + 17 * max(0, k)
        minimum = ends[-1] + 17 if ends else 5
        if end < minimum:
            raise ValueError("Director MAIN shots are too short for distinct native H3 boundaries; extend or merge these MAIN shots. CAMERA/EMBEDDING layers may be shorter.")
        ends.append(end)
    lengths = tuple(end if i == 0 else end - ends[i - 1] + overlap for i, end in enumerate(ends))
    starts = tuple(0 if i == 0 else ends[i - 1] - overlap for i in range(len(ends)))
    return lengths, starts, ends[-1]


def region_frame_spans(regions: Sequence[Mapping[str, Any]], *, context_start: int,
                       segment_length: int, shot_start_seconds: float, fps: int = 24) -> list[dict[str, Any]]:
    """One seconds -> frame mapping. No per-shot stretch to aligned duration."""
    out = []
    for i, raw in enumerate(regions):
        a = round((shot_start_seconds + float(raw["start_seconds"])) * fps) - context_start
        b = round((shot_start_seconds + float(raw["end_seconds"])) * fps) - context_start
        # The incoming state also owns hidden continuation context. Terminal
        # alignment padding belongs to the final state and is trimmed at output.
        if i == 0:
            a = 0
        if i == len(regions) - 1:
            b = segment_length
        item = dict(raw)
        item.update(start_frame=max(0, min(segment_length, a)), end_frame=max(0, min(segment_length, b)))
        out.append(item)
    return out


def regional_attention_route(layout: Any, specs: Sequence[Mapping[str, Any]], seq_len: int,
                             frame_pattern: Sequence[int] = (1, 4, 4, 4, 4)) -> dict[str, Any]:
    """Partition joint Q/K rows with explicit context offsets, never tail guesses.

    A latent token belongs to the region with greatest decoded-frame overlap;
    ties go to the later region. Each query is evaluated once with a 1xK mask.
    Audio packing is channel-major, so both channels get the same clock.
    """
    segments = [(int(a), int(b), str(k)) for a, b, k in layout.segments]
    text = next((a, b) for a, b, k in segments if k == "text")
    video = next((a, b) for a, b, k in segments if k == "video")
    sig = layout.signature
    latent_t = int(sig[1])
    if latent_t <= 0 or (video[1] - video[0]) % latent_t:
        raise ValueError("Director temporal routing: incompatible target video geometry.")
    if max(b for _, b, _ in segments) != seq_len:
        raise ValueError("Director temporal routing: packed sequence size mismatch.")
    frame_rows = (video[1] - video[0]) // latent_t
    labels = [""] * seq_len  # host integers/strings only, no CUDA synchronization
    row_ranges: dict[str, tuple[int, int]] = {}
    for item in specs:
        name = str(item["name"])
        a, b = text[0] + int(item["text_start"]), text[0] + int(item["text_stop"])
        if not text[0] <= a < b <= text[1]:
            raise ValueError(f"Director conditioning rows outside text context: {name} {a}:{b}.")
        if any(labels[a:b]):
            raise ValueError("Director conditioning text regions overlap.")
        labels[a:b] = [name] * (b - a)
        row_ranges[name] = (a, b)

    def owner(a: float, b: float) -> str:
        choices = [(max(0.0, min(b, float(s["end_frame"])) - max(a, float(s["start_frame"]))), i, str(s["name"])) for i, s in enumerate(specs)]
        amount, _, name = max(choices, default=(0.0, 0, ""))
        return name if amount > 0 else ""

    cursor = 0
    for t in range(latent_t):
        end = cursor + int(frame_pattern[t % len(frame_pattern)])
        a = video[0] + t * frame_rows
        labels[a:a + frame_rows] = [owner(cursor, end)] * frame_rows
        cursor = end
    # PackedLayout.signature = (text_len, latent_t, lat_h, lat_w, audio_t).
    audio_t = int(sig[4])
    for a, b, kind in segments:
        if kind != "audio":
            continue
        if audio_t <= 0 or (b - a) % audio_t:
            raise ValueError("Director temporal routing: incompatible channel-major audio geometry.")
        # H3 target audio spans the same local clip duration as target video.
        audio_labels = [owner(t * cursor / audio_t, (t + 1) * cursor / audio_t) for t in range(audio_t)]
        labels[a:b] = audio_labels * ((b - a) // audio_t)
    intervals = []
    a = 0
    for b in range(1, seq_len + 1):
        if b == seq_len or labels[b] != labels[a]:
            intervals.append((a, b, labels[a]))
            a = b
    key_ranges = tuple((a, b, name) for a, b, name in intervals if name)
    return {"regional": True, "row_ranges": row_ranges, "query_intervals": tuple(intervals),
            "key_ranges": key_ranges, "decoded_frames": cursor, "latent_t": latent_t,
            "text_span": text, "video_span": video}
