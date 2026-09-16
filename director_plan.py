"""LongMedia Director timeline model and compiler.

The Director is an authoring surface, not a sampler. It keeps human semantic time
in seconds and compiles a three-track MAIN / CAMERA / AUDIO timeline into the
existing LongMedia clip-plan contract. Native H3 ``17*k+5`` alignment remains
owned by LongMedia execution.

The v4 document intentionally stores only media references (ComfyUI input-folder
filename/subfolder records). Actual IMAGE/AUDIO/video-frame tensors are loaded at Director execution time and bundled with the compiled MultiClip/camera plan into one H3_LONGMEDIA_DIRECTOR contract for Setup.
"""
from __future__ import annotations

import json
import math
import uuid
from typing import Any, Callable

if __package__:
    from .director_timing import conditioning_controls, conditioning_regions
else:
    from director_timing import conditioning_controls, conditioning_regions

MIN_SHOTS = 1
MAX_SHOTS = 32
MAX_SUBJECTS = 16
MAX_CAMERA_BLOCKS = 64
MAX_AUDIO_BLOCKS = 64
MAX_EXTRA_TRACKS = 24
MAX_EXTRA_CLIPS = 256
EXTRA_TRACK_TYPES = {"prompt", "embedding", "character", "reference", "video", "audio"}

DEFAULT_CAMERA = {
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


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4()}"


def _finite_float(value: Any, fallback: float) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(fallback)
    return out if math.isfinite(out) else float(fallback)


def _media_record(value: Any, expected_kind: str | None = None) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    filename = str(value.get("filename") or "").strip()
    if not filename:
        return None
    kind = str(value.get("kind") or expected_kind or "image").strip().lower()
    if kind not in {"image", "video", "audio"}:
        kind = expected_kind or "image"
    out: dict[str, Any] = {
        "kind": kind,
        "filename": filename,
        "subfolder": str(value.get("subfolder") or "").strip("/\\"),
    }
    for key in ("width", "height"):
        try:
            n = int(value.get(key) or 0)
        except Exception:
            n = 0
        if n > 0:
            out[key] = n
    try:
        seconds = float(value.get("seconds") or 0.0)
    except Exception:
        seconds = 0.0
    if math.isfinite(seconds) and seconds > 0:
        out["seconds"] = seconds
    return out


def default_shot(index: int = 0, duration: float = 5.0) -> dict[str, Any]:
    return {
        "clip_id": _id("shot"),
        "name": f"Shot {index + 1}",
        "prompt": "",
        "duration": float(duration),
        "seed": None,
        "subjects": [],
        "media_subject_id": None,
        "cast_subject_id": None,
        "first_frame_anchor": False,
        "last_frame_anchor": False,
        "source_in": 0.0,
    }


def default_camera_block(start: float = 0.0, duration: float = 5.0) -> dict[str, Any]:
    return {
        "block_id": _id("camera"),
        "start": float(start),
        "duration": float(duration),
        "camera": dict(DEFAULT_CAMERA),
        "note": "",
    }


def default_document() -> dict[str, Any]:
    shots = [default_shot(0, 5.0), default_shot(1, 5.0), default_shot(2, 5.0)]
    return {
        "version": 9,
        "kind": "h3_longmedia_director",
        "project_id": _id("project"),
        "fps": 24.0,
        "audio_mode": "auto",
        "setup_h3_mode": "auto",
        "setup_timeline_mode": "auto",
        "segmented_count": 3,
        "segmented_duration": 5.0,
        "resolution_source": "base_layer",
        "resolution": "1920x1080",
        "megapixel": 0.8,
        "regeneration": {"mode": "none", "clip_id": None, "request_id": None},
        "subjects": [],
        "shots": shots,
        "camera_blocks": [
            default_camera_block(0.0, 5.0),
            default_camera_block(5.0, 5.0),
            default_camera_block(10.0, 5.0),
        ],
        "audio_blocks": [],
        "extra_tracks": [],
    }


def default_document_json() -> str:
    return json.dumps(default_document(), ensure_ascii=False)


def _normalize_subject(item: Any, index: int) -> dict[str, Any]:
    item = item if isinstance(item, dict) else {}
    kind = str(item.get("kind") or "Picture").strip().title()
    if kind not in {"Picture", "Video", "Audio"}:
        kind = "Picture"
    max_slot = 9 if kind == "Picture" else 3
    try:
        slot = int(item.get("slot", index + 1))
    except Exception:
        slot = index + 1
    slot = max(1, min(max_slot, slot))
    default_retention = "reference" if kind == "Audio" else "fully_preserved"
    media = _media_record(item.get("media"), kind.lower().replace("picture", "image"))
    return {
        "subject_id": str(item.get("subject_id") or item.get("id") or "").strip() or _id("subject"),
        "name": str(item.get("name") or f"Subject {index + 1}").strip()[:160],
        "kind": kind,
        "slot": slot,
        "description": str(item.get("description") or "").strip(),
        "retention": str(item.get("retention") or default_retention).strip(),
        "media": media,
    }


def _normalize_shot(item: Any, index: int, subject_ids: set[str]) -> dict[str, Any]:
    item = item if isinstance(item, dict) else {}
    duration = max(0.25, min(150.0, _finite_float(item.get("duration"), 5.0)))
    seed = item.get("seed")
    if seed in ("", None):
        seed = None
    else:
        try:
            seed = max(0, int(seed))
        except Exception:
            seed = None
    selected = item.get("subjects") if isinstance(item.get("subjects"), list) else []
    selected = [str(value) for value in selected if str(value) in subject_ids]
    media_subject_id = str(item.get("media_subject_id") or "").strip() or None
    if media_subject_id not in subject_ids:
        media_subject_id = None
    cast_subject_id = str(item.get("cast_subject_id") or "").strip() or None
    if cast_subject_id not in subject_ids:
        cast_subject_id = None
    return {
        "clip_id": str(item.get("clip_id") or item.get("id") or "").strip() or _id("shot"),
        "name": str(item.get("name") or item.get("clip_name") or f"Shot {index + 1}").strip()[:120],
        "prompt": str(item.get("prompt") or "").strip(),
        "duration": duration,
        "seed": seed,
        "subjects": selected,
        "media_subject_id": media_subject_id,
        "cast_subject_id": cast_subject_id,
        "first_frame_anchor": bool(item.get("first_frame_anchor", False)),
        "last_frame_anchor": bool(item.get("last_frame_anchor", False)),
        "source_in": max(0.0, _finite_float(item.get("source_in"), 0.0)),
    }


def _normalize_camera(item: Any) -> dict[str, Any]:
    item = item if isinstance(item, dict) else {}
    out = dict(DEFAULT_CAMERA)
    for key, fallback in DEFAULT_CAMERA.items():
        if key == "transition_to_next":
            out[key] = bool(item.get(key, fallback))
        else:
            out[key] = str(item.get(key) or fallback)
    return out


def _normalize_camera_block(item: Any, index: int) -> dict[str, Any]:
    item = item if isinstance(item, dict) else {}
    return {
        "block_id": str(item.get("block_id") or item.get("id") or "").strip() or _id("camera"),
        "start": max(0.0, _finite_float(item.get("start"), 0.0)),
        "duration": max(0.25, min(600.0, _finite_float(item.get("duration"), 5.0))),
        "camera": _normalize_camera(item.get("camera")),
        "note": str(item.get("note") or "").strip(),
    }


def _normalize_audio_block(item: Any, index: int, subject_ids: set[str]) -> dict[str, Any]:
    item = item if isinstance(item, dict) else {}
    kind = str(item.get("kind") or "prompt").strip().lower()
    if kind not in {"prompt", "reference"}:
        kind = "prompt"
    subject_id = str(item.get("subject_id") or "").strip() or None
    if subject_id not in subject_ids:
        subject_id = None
    role = str(item.get("role") or "diegetic").strip().lower()
    if role not in {"diegetic", "music", "reactive"}:
        role = "diegetic"
    return {
        "block_id": str(item.get("block_id") or item.get("id") or "").strip() or _id("audio"),
        "start": max(0.0, _finite_float(item.get("start"), 0.0)),
        "duration": max(0.25, min(600.0, _finite_float(item.get("duration"), 5.0))),
        "kind": kind,
        "role": role,
        "prompt": str(item.get("prompt") or "").strip(),
        "subject_id": subject_id,
        "source_in": max(0.0, _finite_float(item.get("source_in"), 0.0)),
    }


def _normalize_extra_clip(item: Any, index: int, subject_ids: set[str]) -> dict[str, Any]:
    item = item if isinstance(item, dict) else {}
    subject_id = str(item.get("subject_id") or "").strip() or None
    if subject_id not in subject_ids:
        subject_id = None
    embedding_name = str(item.get("embedding_name") or "").strip().replace("\\", "/")
    if embedding_name.lower().startswith("embedding:"):
        embedding_name = embedding_name.split(":", 1)[1].strip()
    if embedding_name.lower().endswith(".safetensors"):
        embedding_name = embedding_name[:-12]
    embedding_name = embedding_name.replace("\r", " ").replace("\n", " ").strip()[:260]
    return {
        "clip_id": str(item.get("clip_id") or item.get("id") or "").strip() or _id("layer"),
        "name": str(item.get("name") or f"Layer {index + 1}").strip()[:120],
        "start": max(0.0, _finite_float(item.get("start"), 0.0)),
        "duration": max(0.25, min(600.0, _finite_float(item.get("duration"), 5.0))),
        "prompt": str(item.get("prompt") or "").strip(),
        "embedding_name": embedding_name,
        "subject_id": subject_id,
        "source_in": max(0.0, _finite_float(item.get("source_in"), 0.0)),
    }


def _normalize_extra_track(item: Any, index: int, subject_ids: set[str]) -> dict[str, Any]:
    item = item if isinstance(item, dict) else {}
    track_type = str(item.get("type") or "prompt").strip().lower()
    if track_type not in EXTRA_TRACK_TYPES:
        track_type = "prompt"
    clips_raw = item.get("clips") if isinstance(item.get("clips"), list) else []
    clips = [_normalize_extra_clip(c, i, subject_ids) for i, c in enumerate(clips_raw[:MAX_EXTRA_CLIPS])]
    return {
        "track_id": str(item.get("track_id") or item.get("id") or "").strip() or _id("track"),
        "type": track_type,
        "name": str(item.get("name") or f"{track_type.title()} {index + 1}").strip()[:100],
        "enabled": item.get("enabled") is not False,
        "locked": item.get("locked") is True,
        "muted": item.get("muted") is True,
        "clips": clips,
    }


def _migrate_v1(data: dict[str, Any]) -> dict[str, Any]:
    """Convert the P1 card document into the v2 three-track document once."""
    if int(data.get("version") or 1) >= 2:
        return data
    shots = data.get("shots") if isinstance(data.get("shots"), list) else []
    cursor = 0.0
    camera_blocks: list[dict[str, Any]] = []
    audio_blocks: list[dict[str, Any]] = []
    extra_tracks = data.get("extra_tracks") if isinstance(data.get("extra_tracks"), list) else []
    migrated_shots: list[dict[str, Any]] = []
    for index, raw in enumerate(shots):
        raw = raw if isinstance(raw, dict) else {}
        duration = max(0.25, _finite_float(raw.get("duration"), 5.0))
        migrated_shots.append({
            "clip_id": raw.get("clip_id") or raw.get("id") or _id("shot"),
            "name": raw.get("name") or f"Shot {index + 1}",
            "prompt": raw.get("prompt") or "",
            "duration": duration,
            "seed": raw.get("seed"),
            "subjects": raw.get("subjects") if isinstance(raw.get("subjects"), list) else [],
            "source_in": max(0.0, _finite_float(raw.get("source_in"), 0.0)),
        })
        camera_blocks.append({
            "block_id": _id("camera"),
            "start": cursor,
            "duration": duration,
            "camera": raw.get("camera") if isinstance(raw.get("camera"), dict) else dict(DEFAULT_CAMERA),
            "note": "",
        })
        audio = raw.get("audio") if isinstance(raw.get("audio"), dict) else {}
        for role in ("diegetic", "music", "reactive"):
            text = str(audio.get(role) or "").strip()
            if text:
                audio_blocks.append({
                    "block_id": _id("audio"),
                    "start": cursor,
                    "duration": duration,
                    "kind": "prompt",
                    "role": role,
                    "prompt": text,
                    "subject_id": None,
                    "source_in": 0.0,
                })
        cursor += duration
    return {
        "version": 9,
        "kind": "h3_longmedia_director",
        "project_id": str(data.get("project_id") or "").strip(),
        "fps": data.get("fps", 24),
        "audio_mode": str(data.get("audio_mode") or "auto"),
        "setup_h3_mode": str(data.get("setup_h3_mode") or "auto"),
        "setup_timeline_mode": str(data.get("setup_timeline_mode") or "auto"),
        "segmented_count": max(1, min(64, int(_finite_float(data.get("segmented_count"), 3)))),
        "segmented_duration": max(0.25, min(150.0, _finite_float(data.get("segmented_duration"), 5.0))),
        "regeneration": data.get("regeneration") if isinstance(data.get("regeneration"), dict) else {"mode": "none", "clip_id": None, "request_id": None},
        "subjects": data.get("subjects") if isinstance(data.get("subjects"), list) else [],
        "shots": migrated_shots,
        "camera_blocks": camera_blocks,
        "audio_blocks": audio_blocks,
        "extra_tracks": extra_tracks,
    }


def _stable_project_id(data: dict[str, Any], shots: list[dict[str, Any]]) -> str:
    existing = str(data.get("project_id") or "").strip()
    if existing:
        return existing[:120]
    # Old Director documents predate project_id. Derive the fallback from the
    # stable first clip id using the same plain-string rule as the browser UI,
    # so the very first full render and a later Regenerate button address the
    # same persistent take store even before the workflow is re-saved.
    first = str((shots[0] if shots else {}).get("clip_id") or "director").strip()
    safe = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in first).strip('._-')
    return f"project-{(safe or 'director')[:100]}"


def _normalize_regeneration(value: Any, shots: list[dict[str, Any]]) -> dict[str, Any]:
    item = value if isinstance(value, dict) else {}
    mode = str(item.get("mode") or "none").strip().lower()
    if mode not in {"none", "clip", "from_here", "recast", "recast_range"}:
        mode = "none"
    valid_ids = {str(shot.get("clip_id") or "") for shot in shots}
    clip_id = str(item.get("clip_id") or "").strip() or None
    request_id = str(item.get("request_id") or "").strip() or None
    source_revision = str(item.get("source_revision") or "").strip() or None
    recast_subject_id = str(item.get("recast_subject_id") or "").strip() or None
    preserve_audio = bool(item.get("preserve_audio", mode in {"recast", "recast_range"}))
    recast_video_denoise = 1.0  # legacy compatibility field; native-reference Recast does not denoise source x0
    recast_reference_scale_raw = item.get("recast_reference_scale", "auto")
    recast_reference_scale = str(recast_reference_scale_raw or "auto").strip().lower()
    if recast_reference_scale not in {"auto", "1.0", "0.875", "0.75", "0.625", "0.5"}:
        recast_reference_scale = "auto"
    recast_clip_ids = [str(v).strip() for v in (item.get("recast_clip_ids") or []) if str(v).strip() in valid_ids]
    source_revisions_raw = item.get("source_revisions") if isinstance(item.get("source_revisions"), dict) else {}
    source_revisions = {str(k): str(v).strip() for k, v in source_revisions_raw.items() if str(k) in valid_ids and str(v).strip()}
    if mode != "none" and (clip_id not in valid_ids or request_id is None):
        mode, clip_id, request_id = "none", None, None
    if mode == "recast" and source_revision is None:
        mode, clip_id, request_id = "none", None, None
    if mode == "recast_range":
        recast_clip_ids = [cid for cid in recast_clip_ids if cid in valid_ids]
        if not recast_clip_ids:
            mode, clip_id, request_id = "none", None, None
        else:
            missing = [cid for cid in recast_clip_ids if cid not in source_revisions]
            if missing:
                mode, clip_id, request_id = "none", None, None
    if mode == "none":
        clip_id, request_id, source_revision, recast_subject_id = None, None, None, None
        preserve_audio = False
        recast_clip_ids = []
        source_revisions = {}
        recast_reference_scale = "auto"
    return {
        "mode": mode,
        "clip_id": clip_id,
        "request_id": request_id,
        "source_revision": source_revision,
        "recast_subject_id": recast_subject_id,
        "preserve_audio": preserve_audio,
        "recast_video_denoise": recast_video_denoise,
        "recast_reference_scale": recast_reference_scale,
        "recast_clip_ids": recast_clip_ids,
        "source_revisions": source_revisions,
    }


def normalize_document(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            data = json.loads(raw or "{}")
        except Exception:
            data = {}
    elif isinstance(raw, dict):
        data = dict(raw)
    else:
        data = {}
    data = _migrate_v1(data)

    subjects_raw = data.get("subjects") if isinstance(data.get("subjects"), list) else []
    subjects = [_normalize_subject(item, i) for i, item in enumerate(subjects_raw[:MAX_SUBJECTS])]
    subject_ids = {item["subject_id"] for item in subjects}

    shots_raw = data.get("shots") if isinstance(data.get("shots"), list) else []
    shots_raw = list(shots_raw[:MAX_SHOTS])
    if not shots_raw:
        shots_raw = [default_shot(0, 5.0)]
    shots = [_normalize_shot(item, i, subject_ids) for i, item in enumerate(shots_raw)]

    camera_raw = data.get("camera_blocks") if isinstance(data.get("camera_blocks"), list) else []
    camera_blocks = [_normalize_camera_block(item, i) for i, item in enumerate(camera_raw[:MAX_CAMERA_BLOCKS])]
    if not isinstance(data.get("camera_blocks"), list):
        cursor = 0.0
        for shot in shots:
            camera_blocks.append(default_camera_block(cursor, shot["duration"]))
            cursor += shot["duration"]

    audio_raw = data.get("audio_blocks") if isinstance(data.get("audio_blocks"), list) else []
    audio_blocks = [
        _normalize_audio_block(item, i, subject_ids)
        for i, item in enumerate(audio_raw[:MAX_AUDIO_BLOCKS])
    ]

    # v3 Director trim contract: timeline duration is playback duration at 1x.
    # Video/audio blocks carry a source_in cursor; expanding a media-backed block is
    # clamped to the source tail instead of silently time-stretching it. Images and
    # authored prompt/camera blocks have no finite source duration and remain freely sizable.
    by_id = {item["subject_id"]: item for item in subjects}
    for shot in shots:
        anchor_subject = by_id.get(str(shot.get("media_subject_id") or ""))
        anchor_media = anchor_subject.get("media") if isinstance(anchor_subject, dict) else None
        anchor_is_picture = bool(
            isinstance(anchor_subject, dict)
            and str(anchor_subject.get("kind") or "") == "Picture"
            and isinstance(anchor_media, dict)
        )
        if not anchor_is_picture:
            shot["first_frame_anchor"] = False
            shot["last_frame_anchor"] = False
        subject = by_id.get(str(shot.get("media_subject_id") or ""))
        media = subject.get("media") if isinstance(subject, dict) else None
        seconds = _finite_float(media.get("seconds"), 0.0) if isinstance(media, dict) else 0.0
        if seconds > 0.0 and str(media.get("kind") or "").lower() == "video":
            source_in = min(max(0.0, float(shot.get("source_in") or 0.0)), max(0.0, seconds - 0.25))
            shot["source_in"] = source_in
            shot["duration"] = max(0.25, min(float(shot["duration"]), max(0.25, seconds - source_in)))
        else:
            shot["source_in"] = 0.0

    for block in audio_blocks:
        subject = by_id.get(str(block.get("subject_id") or "")) if block.get("kind") == "reference" else None
        media = subject.get("media") if isinstance(subject, dict) else None
        seconds = _finite_float(media.get("seconds"), 0.0) if isinstance(media, dict) else 0.0
        if seconds > 0.0:
            source_in = min(max(0.0, float(block.get("source_in") or 0.0)), max(0.0, seconds - 0.25))
            block["source_in"] = source_in
            block["duration"] = max(0.25, min(float(block["duration"]), max(0.25, seconds - source_in)))
        else:
            block["source_in"] = 0.0

    audio_mode = str(data.get("audio_mode") or "auto").strip().lower()
    if audio_mode not in {"auto", "preserve", "generate", "reference_only", "preserve_reference", "lip_sync"}:
        audio_mode = "auto"

    project_id = _stable_project_id(data, shots)
    regeneration = _normalize_regeneration(data.get("regeneration"), shots)
    extra_raw = data.get("extra_tracks") if isinstance(data.get("extra_tracks"), list) else []
    extra_tracks = [_normalize_extra_track(item, i, subject_ids) for i, item in enumerate(extra_raw[:MAX_EXTRA_TRACKS])]

    setup_h3_mode = str(data.get("setup_h3_mode") or "auto").strip().lower()
    if setup_h3_mode not in {"auto", "t2va", "fl2va", "ref2va", "hybrid", "video_ref_edit"}:
        setup_h3_mode = "auto"
    setup_timeline_mode = str(data.get("setup_timeline_mode") or "auto").strip().lower()
    if setup_timeline_mode not in {"auto", "single", "segmented", "multiclip"}:
        setup_timeline_mode = "auto"
    segmented_count = max(1, min(64, int(_finite_float(data.get("segmented_count"), 3))))
    segmented_duration = max(0.25, min(150.0, _finite_float(data.get("segmented_duration"), 5.0)))
    if setup_timeline_mode == "segmented" and len(shots) == 1 and data.get("segmented_count") is None and data.get("segmented_duration") is None:
        total = max(0.25, float(shots[0].get("duration") or 5.0))
        segmented_count = max(1, min(64, int(round(total / 5.0))))
        segmented_duration = max(0.25, min(150.0, total / float(segmented_count)))
    resolution_source = str(data.get("resolution_source") or "base_layer").strip().lower()
    if resolution_source not in {"base_layer", "custom", "16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "2.39:1"}:
        resolution_source = "base_layer"
    resolution = str(data.get("resolution") or "1920x1080").strip().lower().replace("×", "x")
    if not resolution:
        resolution = "1920x1080"
    megapixel = max(0.1, min(8.0, _finite_float(data.get("megapixel"), 0.8)))

    return {
        "version": 9,
        "kind": "h3_longmedia_director",
        "project_id": project_id,
        "fps": max(1.0, min(120.0, _finite_float(data.get("fps"), 24.0))),
        "audio_mode": audio_mode,
        "setup_h3_mode": setup_h3_mode,
        "setup_timeline_mode": setup_timeline_mode,
        "segmented_count": segmented_count,
        "segmented_duration": segmented_duration,
        "resolution_source": resolution_source,
        "resolution": resolution,
        "megapixel": megapixel,
        "regeneration": regeneration,
        "subjects": subjects,
        "shots": shots,
        "camera_blocks": camera_blocks,
        "audio_blocks": audio_blocks,
        "extra_tracks": extra_tracks,
    }


def director_frame_anchor_contract(document: dict[str, Any]) -> dict[str, Any]:
    """Resolve explicit FIRST/LAST roles from MAIN shots into one runtime contract.

    Roles belong to semantic MAIN media, never to Picture slot numbers. FIRST and LAST
    are unique across the timeline. A single Picture may own both roles, which means
    the exact same source tensor is used at both H3 keyframe boundaries.
    """
    subjects = {
        str(item.get("subject_id") or ""): item
        for item in (document.get("subjects") or [])
        if isinstance(item, dict)
    }
    first: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    last: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for index, shot in enumerate(document.get("shots") or []):
        if not isinstance(shot, dict):
            continue
        subject = subjects.get(str(shot.get("media_subject_id") or ""))
        valid_picture = bool(
            subject
            and str(subject.get("kind") or "") == "Picture"
            and isinstance(subject.get("media"), dict)
        )
        if not valid_picture:
            continue
        if bool(shot.get("first_frame_anchor")):
            first.append((index, shot, subject))
        if bool(shot.get("last_frame_anchor")):
            last.append((index, shot, subject))
    if len(first) > 1:
        raise ValueError("Director allows exactly one FIRST frame anchor across MAIN.")
    if len(last) > 1:
        raise ValueError("Director allows exactly one LAST frame anchor across MAIN.")

    first_item = first[0] if first else None
    last_item = last[0] if last else None
    first_sid = str(first_item[2].get("subject_id") or "") if first_item else None
    last_sid = str(last_item[2].get("subject_id") or "") if last_item else None
    same_source = bool(first_sid and last_sid and first_sid == last_sid)
    anchor_subject_ids = []
    for sid in (first_sid, last_sid):
        if sid and sid not in anchor_subject_ids:
            anchor_subject_ids.append(sid)
    return {
        "enabled": bool(first_item or last_item),
        "first_clip_id": (str(first_item[1].get("clip_id") or "") if first_item else None),
        "first_subject_id": first_sid,
        "first_picture_slot": (int(first_item[2].get("slot") or 1) if first_item else None),
        "last_clip_id": (str(last_item[1].get("clip_id") or "") if last_item else None),
        "last_subject_id": last_sid,
        "last_picture_slot": (int(last_item[2].get("slot") or 1) if last_item else None),
        "same_source_loop": same_source,
        "anchor_subject_ids": anchor_subject_ids,
        "anchor_count": int(first_item is not None) + int(last_item is not None),
    }


def _active_picture_subject_ids(document: dict[str, Any]) -> set[str]:
    active: set[str] = set()
    for shot in document.get("shots") or []:
        if not isinstance(shot, dict):
            continue
        for sid in list(shot.get("subjects") or []) + [shot.get("media_subject_id"), shot.get("cast_subject_id")]:
            sid = str(sid or "").strip()
            if sid:
                active.add(sid)
    for track in document.get("extra_tracks") or []:
        if not isinstance(track, dict) or track.get("enabled") is False or track.get("muted") is True:
            continue
        for clip in track.get("clips") or []:
            sid = str((clip or {}).get("subject_id") or "").strip() if isinstance(clip, dict) else ""
            if sid:
                active.add(sid)
    return active


def director_picture_runtime_map(document: dict[str, Any]) -> dict[str, Any]:
    """Return contiguous native Picture ordinals after removing frame anchors.

    WHO & WHAT slot numbers are editor identities. Native H3 Picture ordinals are a
    packed runtime ABI and therefore must be compacted from only active references.
    """
    anchors = director_frame_anchor_contract(document)
    anchor_ids = set(anchors.get("anchor_subject_ids") or [])
    active_ids = _active_picture_subject_ids(document)
    pictures = [
        item for item in (document.get("subjects") or [])
        if isinstance(item, dict)
        and str(item.get("kind") or "") == "Picture"
        and isinstance(item.get("media"), dict)
        and str(item.get("subject_id") or "") in active_ids
        and str(item.get("subject_id") or "") not in anchor_ids
    ]
    pictures.sort(key=lambda item: (int(item.get("slot") or 1), str(item.get("subject_id") or "")))
    runtime_slots = {str(item.get("subject_id") or ""): index + 1 for index, item in enumerate(pictures)}
    return {
        "runtime_slots": runtime_slots,
        "reference_subject_ids": list(runtime_slots),
        "anchors": anchors,
    }


def subject_token(subject: dict[str, Any]) -> str:
    runtime_slot = subject.get("_runtime_slot")
    slot = int(runtime_slot) if runtime_slot is not None else int(subject["slot"])
    return f"<{subject['kind']} {slot}>"


def _subject_sentence(subject: dict[str, Any], anchor_roles: tuple[str, ...] = ()) -> str:
    roles = tuple(str(role) for role in anchor_roles if str(role))
    if roles:
        if roles == ("first", "last") or set(roles) == {"first", "last"}:
            lead = "The supplied opening and ending frame image"
        elif "first" in roles:
            lead = "The supplied opening frame image"
        else:
            lead = "The supplied ending frame image"
        name = str(subject.get("name") or "subject").strip()
        description = str(subject.get("description") or "").strip()
        retention = str(subject.get("retention") or "").strip()
        parts = [f"{lead} represents {name}."]
        if description:
            parts.append(description.rstrip(".") + ".")
        if retention:
            parts.append(f"Anchor relationship: {retention.replace('_', ' ')}.")
        return " ".join(parts)
    token = subject_token(subject)
    name = str(subject.get("name") or "subject").strip()
    description = str(subject.get("description") or "").strip()
    retention = str(subject.get("retention") or "").strip()
    parts = [f"{token} represents {name}."]
    if description:
        parts.append(description.rstrip(".") + ".")
    if retention:
        parts.append(f"Reference relationship: {retention.replace('_', ' ')}.")
    return " ".join(parts)


def _overlap(start_a: float, duration_a: float, start_b: float, duration_b: float) -> float:
    return max(0.0, min(start_a + duration_a, start_b + duration_b) - max(start_a, start_b))


def _camera_blocks_for_shot(document: dict[str, Any], start: float, duration: float) -> list[dict[str, Any]]:
    shot_end = float(start) + float(duration)
    matches: list[dict[str, Any]] = []
    for raw in document["camera_blocks"]:
        block = dict(raw)
        amount = _overlap(start, duration, float(block["start"]), float(block["duration"]))
        if amount <= 0.0:
            continue
        local_start = max(float(start), float(block["start"])) - float(start)
        local_end = min(shot_end, float(block["start"]) + float(block["duration"])) - float(start)
        if local_end <= local_start:
            continue
        block["_local_start_seconds"] = float(local_start)
        block["_local_end_seconds"] = float(local_end)
        block["_overlap_seconds"] = float(amount)
        matches.append(block)
    matches.sort(key=lambda item: float(item.get('_local_start_seconds', 0.0)))
    return matches


def _camera_for_shot(document: dict[str, Any], start: float, duration: float) -> dict[str, Any]:
    blocks = _camera_blocks_for_shot(document, start, duration)
    if not blocks:
        return {"camera": dict(DEFAULT_CAMERA), "note": "", "block_id": None}
    return max(blocks, key=lambda item: float(item.get('_overlap_seconds', 0.0)))


def _audio_for_shot(
    document: dict[str, Any],
    start: float,
    duration: float,
    subjects_by_id: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    ids: list[str] = []
    for block in document["audio_blocks"]:
        if _overlap(start, duration, float(block["start"]), float(block["duration"])) <= 0:
            continue
        role = str(block.get("role") or "diegetic")
        prefix = {
            "diegetic": "Diegetic audio direction",
            "music": "Music direction",
            "reactive": "Audio-reactive visual direction",
        }.get(role, "Audio direction")
        text = str(block.get("prompt") or "").strip()
        subject_id = block.get("subject_id")
        subject = subjects_by_id.get(str(subject_id)) if subject_id else None
        if subject is not None:
            ids.append(subject["subject_id"])
            token = subject_token(subject)
            if text:
                lines.append(f"{prefix}: {text.rstrip('.')} using {token}.")
            else:
                lines.append(f"{prefix}: use {token} as the active reference for this interval.")
        elif text:
            lines.append(f"{prefix}: {text.rstrip('.')}.")
    return lines, ids


def _extra_layers_for_shot(
    document: dict[str, Any],
    start: float,
    duration: float,
    subjects_by_id: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str], list[str], list[dict[str, Any]]]:
    """Compile semantic timeline layers active for one MAIN shot.

    MiniMax-H3 text embeddings are kept out of prose and returned separately.
    Their exact local timeline spans are also preserved so LongMedia can gate the
    native embedding tokens inside one H3 attention pass instead of splitting a
    MAIN shot into hidden MultiClip renders.
    """
    lines: list[str] = []
    ids: list[str] = []
    embeddings: list[str] = []
    embedding_spans: list[dict[str, Any]] = []
    prefixes = {
        "prompt": "Timeline prompt layer",
        "character": "Character layer",
        "reference": "Reference layer",
        "video": "Video influence layer",
        "audio": "Audio influence layer",
    }
    for track in document.get("extra_tracks") or []:
        if track.get("enabled") is False or track.get("muted") is True:
            continue
        track_type = str(track.get("type") or "prompt")
        for clip in track.get("clips") or []:
            if _overlap(start, duration, float(clip.get("start") or 0.0), float(clip.get("duration") or 0.0)) <= 0:
                continue
            if track_type == "embedding":
                name = str(clip.get("embedding_name") or "").strip().replace("\\", "/")
                if name.lower().startswith("embedding:"):
                    name = name.split(":", 1)[1].strip()
                if name.lower().endswith(".safetensors"):
                    name = name[:-12]
                name = name.replace("\r", " ").replace("\n", " ").strip()
                if name:
                    if name not in embeddings:
                        embeddings.append(name)
                    clip_start = float(clip.get("start") or 0.0)
                    clip_duration = max(0.0, float(clip.get("duration") or 0.0))
                    local_start = max(float(start), clip_start) - float(start)
                    local_end = min(float(start) + float(duration), clip_start + clip_duration) - float(start)
                    if local_end > local_start + 1e-9:
                        embedding_spans.append({
                            "name": name,
                            "start_seconds": max(0.0, float(local_start)),
                            "end_seconds": min(float(duration), float(local_end)),
                        })
                continue
            prompt = str(clip.get("prompt") or "").strip()
            sid = str(clip.get("subject_id") or "").strip()
            subject = subjects_by_id.get(sid) if sid else None
            pieces: list[str] = []
            if prompt:
                pieces.append(prompt.rstrip(".") + ".")
            if subject is not None:
                ids.append(subject["subject_id"])
                pieces.append(f"Use {subject_token(subject)} for this layer.")
            if pieces:
                label = str(track.get("name") or prefixes.get(track_type, "Timeline layer"))
                lines.append(f"{prefixes.get(track_type, 'Timeline layer')} [{label}]: {' '.join(pieces)}")
    return lines, ids, embeddings, embedding_spans


def _native_character_replace_contract(document: dict[str, Any]) -> dict[str, Any] | None:
    """Return the one-shot native H3 character-replacement contract, if unambiguous.

    Native ``video_ref_edit`` owns one source-video timeline.  Director therefore
    auto-arms it only when one MAIN shot is backed by real Video media and one
    enabled CHARACTER layer covers that entire shot with a real Picture subject.
    Partial CHARACTER spans remain ordinary semantic layers; selective interval
    replacement is the persistent-take Recast contract instead.
    """
    regeneration = document.get("regeneration") if isinstance(document.get("regeneration"), dict) else {}
    if str(regeneration.get("mode") or "none").strip().lower() != "none":
        # Any queued persistent-take operation owns execution for this run.  Do not
        # let first-generation native replacement steal Recast/Clip/From-Here.
        return None
    shots = document.get("shots") if isinstance(document.get("shots"), list) else []
    if len(shots) != 1:
        return None
    shot = shots[0] if isinstance(shots[0], dict) else {}
    subjects = {
        str(item.get("subject_id") or ""): item
        for item in (document.get("subjects") or [])
        if isinstance(item, dict)
    }
    source_id = str(shot.get("media_subject_id") or "").strip()
    source = subjects.get(source_id)
    if not source or str(source.get("kind") or "") != "Video" or not isinstance(source.get("media"), dict):
        return None

    start = 0.0
    duration = max(0.25, float(shot.get("duration") or 0.0))
    end = start + duration
    epsilon = max(1.0 / max(float(document.get("fps") or 24.0), 1.0), 1e-3)
    candidates: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    for track in document.get("extra_tracks") or []:
        if not isinstance(track, dict) or track.get("enabled") is False or str(track.get("type") or "") != "character":
            continue
        for layer in track.get("clips") or []:
            if not isinstance(layer, dict):
                continue
            layer_start = float(layer.get("start") or 0.0)
            layer_end = layer_start + max(0.0, float(layer.get("duration") or 0.0))
            if layer_start > start + epsilon or layer_end < end - epsilon:
                continue
            sid = str(layer.get("subject_id") or "").strip()
            character = subjects.get(sid)
            if not character or str(character.get("kind") or "") != "Picture" or not isinstance(character.get("media"), dict):
                continue
            candidates[sid] = (character, track, layer)

    if len(candidates) != 1:
        return None
    character_id, (character, track, layer) = next(iter(candidates.items()))
    return {
        "enabled": True,
        "mode": "video_ref_edit",
        "clip_id": str(shot.get("clip_id") or ""),
        "source_video_subject_id": source_id,
        "source_video_slot": int(source.get("slot") or 1),
        "character_subject_id": character_id,
        "character_picture_slot": int(character.get("slot") or 1),
        "character_track_id": str(track.get("track_id") or ""),
        "character_layer_id": str(layer.get("clip_id") or ""),
        "start_seconds": start,
        "end_seconds": end,
        "full_shot_coverage": True,
    }


def compile_director_plan(
    raw: Any,
    *,
    global_prompt: str = "",
    camera_compiler: Callable[[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None], str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    """Compile Director v1/v2 JSON into LongMedia's existing clip-plan contract."""
    document = normalize_document(raw)
    frame_anchors = director_frame_anchor_contract(document)
    picture_runtime = director_picture_runtime_map(document)
    runtime_picture_slots = dict(picture_runtime.get("runtime_slots") or {})
    for item in document["subjects"]:
        sid = str(item.get("subject_id") or "")
        if str(item.get("kind") or "") == "Picture" and sid in runtime_picture_slots:
            item["_runtime_slot"] = int(runtime_picture_slots[sid])
    subjects_by_id = {item["subject_id"]: item for item in document["subjects"]}
    shots = document["shots"]
    native_character_replace = _native_character_replace_contract(document)

    starts: list[float] = []
    cursor = 0.0
    for shot in shots:
        starts.append(cursor)
        cursor += float(shot["duration"])

    camera_states: list[dict[str, Any]] = []
    shot_camera_blocks: list[list[dict[str, Any]]] = []
    for shot, start in zip(shots, starts):
        blocks = _camera_blocks_for_shot(document, start, float(shot["duration"]))
        shot_camera_blocks.append(blocks)
        if blocks:
            camera = dict(blocks[0]["camera"])
        else:
            block = _camera_for_shot(document, start, float(shot["duration"]))
            camera = dict(block["camera"])
        camera_states.append(camera)

    output_clips: list[dict[str, Any]] = []
    camera_clips: list[dict[str, Any]] = []
    used_tokens: set[str] = set()

    for index, (shot, start) in enumerate(zip(shots, starts)):
        camera = dict(camera_states[index])
        camera["transition_to_next"] = index < len(shots) - 1
        previous_camera = dict(camera_states[index - 1]) if index > 0 else None
        next_camera = dict(camera_states[index + 1]) if index + 1 < len(shots) else None
        shot_blocks = shot_camera_blocks[index] if index < len(shot_camera_blocks) else []
        camera_block = _camera_for_shot(document, start, float(shot["duration"]))
        camera_instruction = ""

        selected_ids = list(shot["subjects"])
        if shot.get("media_subject_id") and shot["media_subject_id"] not in selected_ids:
            selected_ids.append(shot["media_subject_id"])
        cast_subject_id = str(shot.get("cast_subject_id") or "").strip() or None
        if cast_subject_id and cast_subject_id not in selected_ids:
            selected_ids.append(cast_subject_id)
        audio_lines, audio_subject_ids = _audio_for_shot(
            document, start, float(shot["duration"]), subjects_by_id
        )
        extra_lines, extra_subject_ids, shot_embeddings, shot_embedding_spans = _extra_layers_for_shot(
            document, start, float(shot["duration"]), subjects_by_id
        )
        for sid in [*audio_subject_ids, *extra_subject_ids]:
            if sid not in selected_ids:
                selected_ids.append(sid)

        native_replace_here = bool(
            native_character_replace
            and str(native_character_replace.get("clip_id") or "") == str(shot.get("clip_id") or "")
        )
        native_replace_direction = ""
        if native_replace_here:
            source_id = str(native_character_replace.get("source_video_subject_id") or "")
            character_id = str(native_character_replace.get("character_subject_id") or "")
            character_subject = subjects_by_id.get(character_id)
            character_token = subject_token(character_subject) if character_subject is not None else "<Picture 1>"
            # Direct native replacement has two visual authorities only: the source
            # performance video and the CHARACTER identity.  Other visual assets in
            # WHO & WHAT must not silently compete for identity during video_ref_edit.
            selected_ids = [
                sid for sid in selected_ids
                if sid in {source_id, character_id}
                or str((subjects_by_id.get(sid) or {}).get("kind") or "") == "Audio"
            ]
            for required_sid in (source_id, character_id):
                if required_sid and required_sid not in selected_ids:
                    selected_ids.append(required_sid)
            _excluded_visual_tokens = {
                subject_token(item)
                for sid, item in subjects_by_id.items()
                if sid not in {source_id, character_id}
                and str(item.get("kind") or "") in {"Picture", "Video"}
            }
            if _excluded_visual_tokens:
                extra_lines = [
                    line for line in extra_lines
                    if not any(token in line for token in _excluded_visual_tokens)
                ]
            native_replace_direction = (
                f"Native character replacement: <Video 1> provides the complete source performance, body motion, facial timing, camera motion, framing, blocking, environment, lighting, and scene continuity. "
                f"{character_token} provides the single principal visible character identity and appearance throughout the complete shot. "
                f"The resulting performance consistently carries {character_token}'s identity while retaining the source video's timing, staging, and cinematic continuity."
            )

        if native_replace_here:
            # CHARACTER owns first-generation native identity. A stale CAST/RECAST
            # dropdown value must not override the authored CHARACTER layer.
            cast_subject_id = None

        # Recast is an identity replacement contract, not an additional-reference
        # contract. Once a cast subject is authoritative, other visual Picture/Video
        # subjects would compete with it and can produce sequential identity swaps.
        # Keep audio references, prompts and camera guidance, but make the cast the
        # sole visual identity reference for this shot.
        if cast_subject_id:
            selected_ids = [
                sid for sid in selected_ids
                if sid == cast_subject_id
                or str((subjects_by_id.get(sid) or {}).get("kind") or "") == "Audio"
            ]
            if cast_subject_id not in selected_ids:
                selected_ids.insert(0, cast_subject_id)

        selected_subjects = [subjects_by_id[sid] for sid in selected_ids if sid in subjects_by_id]
        _first_anchor_sid = str(frame_anchors.get("first_subject_id") or "")
        _last_anchor_sid = str(frame_anchors.get("last_subject_id") or "")
        def _roles_for_subject(item: dict[str, Any]) -> tuple[str, ...]:
            sid = str(item.get("subject_id") or "")
            roles: list[str] = []
            if sid and sid == _first_anchor_sid:
                roles.append("first")
            if sid and sid == _last_anchor_sid:
                roles.append("last")
            return tuple(roles)
        subject_text = " ".join(_subject_sentence(item, _roles_for_subject(item)) for item in selected_subjects)
        for item in selected_subjects:
            if not _roles_for_subject(item):
                used_tokens.add(subject_token(item))
        if native_replace_here:
            _source_subject = subjects_by_id.get(str(native_character_replace.get("source_video_subject_id") or ""))
            if _source_subject is not None:
                _source_token = subject_token(_source_subject)
                if _source_token != "<Video 1>":
                    subject_text = subject_text.replace(_source_token, "<Video 1>")
                    used_tokens.discard(_source_token)
                    used_tokens.add("<Video 1>")

        cast_direction = ""
        cast_subject = subjects_by_id.get(cast_subject_id) if cast_subject_id else None
        if cast_subject is not None:
            cast_token = subject_token(cast_subject)
            cast_direction = (
                f"Casting direction: {cast_token} is the single principal visible character identity throughout this entire shot, from the first visible frame through the last. "
                f"Every visible face and body belonging to the principal performance consistently carries {cast_token}'s identity and appearance. "
                "The established environment, framing, lighting, blocking, body performance, speech timing, and scene continuity remain coherent around this recast performance."
            )

        # v0.6.20: temporal controls are factorized.  Camera blocks and partial
        # embeddings never clone the full scene prompt and never manufacture a
        # new camera presentation merely because an embedding starts/stops.
        camera_spans = [
            {"start_seconds": float(item["_local_start_seconds"]),
             "end_seconds": float(item["_local_end_seconds"]),
             "camera": dict(item.get("camera") or DEFAULT_CAMERA),
             "note": str(item.get("note") or ""), "block_id": item.get("block_id")}
            for item in shot_blocks
        ]
        duration = float(shot["duration"])
        full_camera = None
        if len(camera_spans) == 1:
            only = camera_spans[0]
            if float(only["start_seconds"]) <= 1e-6 and float(only["end_seconds"]) >= duration - 1e-6:
                full_camera = only
        full_embeddings = [
            item for item in shot_embedding_spans
            if float(item["start_seconds"]) <= 1e-6 and float(item["end_seconds"]) >= duration - 1e-6
        ]
        full_embedding_names = list(dict.fromkeys(str(item["name"]) for item in full_embeddings))
        full_embedding_name_set = set(full_embedding_names)
        partial_embeddings = [
            item for item in shot_embedding_spans
            if item not in full_embeddings and str(item["name"]) not in full_embedding_name_set
        ]
        temporal_camera_spans = [] if full_camera is not None else camera_spans
        controls = conditioning_controls(duration, temporal_camera_spans, partial_embeddings, camera_compiler)

        # A single full-shot camera remains native/global.  Full-shot embeddings
        # retain stock H3 semantics too.  Only genuinely time-varying controls
        # are appended as independent presentations in Setup.
        if full_camera is not None:
            card = dict(full_camera.get("camera") or DEFAULT_CAMERA)
            card["director_interval_seconds"] = duration
            card["director_camera_block_duration_seconds"] = duration
            card["director_trajectory_phase_start"] = 0.0
            card["director_trajectory_phase_end"] = 1.0
            camera_instruction = camera_compiler(card, None, None) if camera_compiler else ""
            note = str(full_camera.get("note") or "").strip()
            camera_instruction = " ".join(x for x in (camera_instruction, note) if x)
        else:
            camera_instruction = ""
        global_embedding_text = "\n".join("embedding:" + name for name in full_embedding_names)
        parts = [shot["prompt"], subject_text, " ".join(extra_lines), " ".join(audio_lines),
                 native_replace_direction, cast_direction,
                 camera_instruction, global_embedding_text]
        local_prompt = "\n\n".join(part.strip() for part in parts if str(part or "").strip())
        end = start + float(shot["duration"])
        output_clips.append({
            "clip_id": shot["clip_id"],
            "name": shot["name"],
            "base_prompt": shot["prompt"],
            "prompt": local_prompt,
            "duration": float(shot["duration"]),
            "seed": shot["seed"],
            "camera": camera,
            "camera_instruction": camera_instruction,
            "director_subject_ids": selected_ids,
            "director_cast_subject_id": cast_subject_id,
            "director_native_character_replace": bool(native_replace_here),
            "director_native_character_subject_id": (str(native_character_replace.get("character_subject_id") or "") if native_replace_here else None),
            "director_native_source_video_subject_id": (str(native_character_replace.get("source_video_subject_id") or "") if native_replace_here else None),
            "director_embeddings": list(shot_embeddings),
            "director_embedding_spans": [dict(item) for item in shot_embedding_spans],
            # Retained for project/backward metadata only; v0.6.20 execution uses
            # factorized controls below and never consumes combined regions.
            "director_conditioning_regions": [],
            "director_temporal_controls": [dict(item) for item in controls],
            "director_timing_version": 2,
            "director_camera_spans": camera_spans,
            "director_start_seconds": start,
            "director_end_seconds": end,
            "director_source_in_seconds": float(shot.get("source_in") or 0.0),
            "director_source_out_seconds": float(shot.get("source_in") or 0.0) + float(shot["duration"]),
            "director_first_frame_anchor": bool(shot.get("first_frame_anchor")),
            "director_last_frame_anchor": bool(shot.get("last_frame_anchor")),
        })
        camera_clips.append({"index": index + 1, **camera, "instruction": camera_instruction})

    clip_plan = {
        "version": 1,
        "kind": "h3_longmedia_clip_plan",
        "source": "MiniMax H3 LongMedia Director",
        "global_prompt": str(global_prompt or "").strip(),
        "clips": output_clips,
        "camera_plan": camera_clips,
        "director": {
            "version": 4,
            "semantic_timeline_seconds": True,
            "project_id": document["project_id"],
            "regeneration": dict(document["regeneration"]),
            "fps_display": document["fps"],
            "audio_mode": document["audio_mode"],
            "setup_h3_mode": document.get("setup_h3_mode", "auto"),
            "setup_timeline_mode": document.get("setup_timeline_mode", "auto"),
            "segmented_count": int(document.get("segmented_count", 3) or 3),
            "segmented_duration": float(document.get("segmented_duration", 5.0) or 5.0),
            "resolution_source": document.get("resolution_source", "base_layer"),
            "resolution": document.get("resolution", "1920x1080"),
            "megapixel": document.get("megapixel", 0.8),
            "subjects": document["subjects"],
            "camera_blocks": document["camera_blocks"],
            "audio_blocks": document["audio_blocks"],
            "extra_tracks": document.get("extra_tracks") or [],
            "used_reference_tokens": sorted(used_tokens),
            "native_character_replace": dict(native_character_replace) if native_character_replace else None,
            "frame_anchors": dict(frame_anchors),
            "picture_runtime_slots": dict(runtime_picture_slots),
        },
    }
    camera_plan = {
        "version": 1,
        "kind": "h3_longmedia_camera_plan",
        "source": "MiniMax H3 LongMedia Director",
        "clip_count": len(output_clips),
        "clips": camera_clips,
    }
    director_plan = {
        "version": 4,
        "kind": "h3_longmedia_director_plan",
        "source": "MiniMax H3 LongMedia Director",
        "project_id": document["project_id"],
        "regeneration": dict(document["regeneration"]),
        "global_prompt": str(global_prompt or "").strip(),
        "total_duration": cursor,
        "document": document,
        "clip_plan": clip_plan,
        "native_character_replace": dict(native_character_replace) if native_character_replace else None,
        "frame_anchors": dict(frame_anchors),
        "picture_runtime_slots": dict(runtime_picture_slots),
    }
    warnings: list[str] = []
    if document["audio_mode"] == "lip_sync":
        has_picture_1 = any(
            item.get("kind") == "Picture" and int(item.get("slot", 0) or 0) == 1 and item.get("media")
            for item in document["subjects"]
        )
        has_audio_1 = any(
            item.get("kind") == "Audio" and int(item.get("slot", 0) or 0) == 1 and item.get("media")
            for item in document["subjects"]
        )
        if not has_picture_1:
            warnings.append("lip_sync requires Picture 1 media (or an explicit Setup.image_1 override).")
        if not has_audio_1:
            warnings.append("lip_sync requires Audio 1 media (or an explicit Setup.audio_1 override).")

    report = json.dumps({
        "kind": director_plan["kind"],
        "shots": len(output_clips),
        "subjects": len(document["subjects"]),
        "camera_blocks": len(document["camera_blocks"]),
        "audio_blocks": len(document["audio_blocks"]),
        "requested_seconds": cursor,
        "semantic_timeline_seconds": True,
        "audio_mode": document["audio_mode"],
        "setup_h3_mode": document.get("setup_h3_mode", "auto"),
        "setup_timeline_mode": document.get("setup_timeline_mode", "auto"),
        "segmented_count": int(document.get("segmented_count", 3) or 3),
        "segmented_duration": float(document.get("segmented_duration", 5.0) or 5.0),
        "resolution_source": document.get("resolution_source", "base_layer"),
        "resolution": document.get("resolution", "1920x1080"),
        "megapixel": document.get("megapixel", 0.8),
        "project_id": document["project_id"],
        "regeneration": dict(document["regeneration"]),
        "native_character_replace": dict(native_character_replace) if native_character_replace else None,
        "frame_anchors": dict(frame_anchors),
        "picture_runtime_slots": dict(runtime_picture_slots),
        "warnings": warnings,
        "h3_frame_lattice_owned_by": "LongMedia execution planner",
        "used_reference_tokens": sorted(used_tokens),
        "trim_contract": "1x playback; edge trims change source_in/duration; no implicit time-stretch",
        "media_manifest": [
            {
                "subject_id": subject["subject_id"],
                "kind": subject["kind"],
                "slot": subject["slot"],
                "name": subject["name"],
                "media": subject.get("media"),
            }
            for subject in document["subjects"] if subject.get("media")
        ],
    }, ensure_ascii=False, indent=2)
    return clip_plan, camera_plan, director_plan, report
