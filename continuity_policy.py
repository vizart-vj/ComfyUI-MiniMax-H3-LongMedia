"""Pure prompt/reference policy for segmented MiniMax H3 generation.

This module deliberately has no ComfyUI or Torch imports so the temporal and
reference-indexing contracts can be regression-tested in a normal Python
process.  Tensor payload construction remains in :mod:`nodes`.
"""

from __future__ import annotations

import re
from typing import Any, Sequence


PICTURE_TAG_RE = re.compile(r"<Picture\s+(\d+)>", re.IGNORECASE)
SEGMENT_EVENT_RE = re.compile(
    r"^\s*(?P<sec>\d+(?:\.\d+)?)\s*(?::|sec\s*:|sec:|s\s*:|s:)\s*(?P<body>.+?)\s*$",
    re.IGNORECASE,
)
EXPLICIT_CONTINUATION_RE = re.compile(
    r"(?im)^(?=\s*continue\s+directly\s+from\s+the\s+preceding\s+video"
    r"(?:\s+scene|\s+segment)?\b)"
)

_RESTART_MARKERS = (
    "start from the supplied first frame composition",
    "start from the supplied first-frame composition",
    "start from supplied first frame composition",
    "start from the first frame composition",
    "begin from the supplied first frame",
    "begin from the first frame",
)

# Only durable appearance/style/camera constraints are allowed to cross from a
# global prompt synopsis into a continuation pass.  Narrative action is owned by
# timestamped events and must not be replayed at every segment boundary.
_PERSISTENT_MARKERS = (
    "preserve", "maintain", "keep consistent", "remain consistent",
    "identity", "appearance", "face", "facial", "hairstyle", "wardrobe",
    "proportions", "lighting", "visual style", "color palette", "camera style",
    "сохрани", "сохраняй", "сохранить", "идентич", "внешност", "лицо",
    "причес", "пропорц", "одежд", "освещен", "визуальн", "стил",
)

# continuity policy: narrative actions are never persistent header state.  A sentence
# may contain an identity/style word and still be an action (for example
# "Preserve the man while he walks toward camera").  These markers provide a
# conservative filter so action ownership stays with timestamped events/state.
_ACTION_MARKERS = (
    " walk", " walks", " walking", " run", " runs", " running",
    " approach", " approaches", " approaching", " meet", " meets", " meeting",
    " kiss", " kisses", " kissing", " hold", " holds", " holding",
    " take", " takes", " taking", " turn", " turns", " turning",
    " embrace", " embraces", " embracing", " sit", " sits", " sitting",
    " stand", " stands", " standing", " enter", " enters", " entering",
    " leave", " leaves", " leaving", " move", " moves", " moving",
    " идёт", " идет", " идут", " идти", " ходит", " бежит", " бегут",
    " встреч", " целу", " берёт", " берет", " держ", " поворач",
    " обнима", " садит", " вста", " входит", " выход", " движ",
)


def _is_action_sentence(sentence: str) -> bool:
    lowered = f" {str(sentence or '').lower()} "
    return any(marker in lowered for marker in _ACTION_MARKERS)


def _completed_state_lines(
    parsed_events: Sequence[tuple[int, float, str]],
    *,
    visible_start_seconds: float,
    max_events: int = 4,
) -> list[str]:
    """Summarize already-completed events as durable state, never replay commands.

    This stays intentionally language-agnostic: event bodies are preserved verbatim
    but framed as facts that are already true.  Keeping only the most recent events
    bounds prompt growth across 3-4+ passes while still carrying interaction/motion
    state forward.
    """

    completed = [
        (sec, body) for _line, sec, body in parsed_events
        if float(sec) < float(visible_start_seconds) - 1e-6
    ]
    if not completed:
        return []
    tail = completed[-max(1, int(max_events)):]
    lines = [
        "Established scene state from earlier timeline events (already completed; do not replay or restart them):"
    ]
    lines.extend(f"- By {_format_local_time(sec)} globally: {body}" for sec, body in tail)
    return lines


def _continuation_contract() -> str:
    return (
        "Continuation contract: continue from the exact final state of the preceding segment. "
        "Preserve every character currently present, their interaction state, body orientation, "
        "pose phase, signed direction of travel, camera velocity, camera side, framing mode, "
        "environment, lighting, wardrobe, and identity. Do not reverse locomotion, backpedal, "
        "pivot, mirror the gait, restart an earlier action, remove and later reintroduce an active "
        "character, or switch to an isolated hero close-up unless the current segment-local timeline "
        "explicitly requests that change. Treat the pass as the same uninterrupted shot: no cut, "
        "no insert shot, no alternate staging, no reset to the opening frame."
    )


def _audio_continuation_contract() -> str:
    return (
        "Audio continuity contract: continue the exact ongoing soundtrack from the preceding segment. "
        "Preserve ambient sound, environmental sound, footsteps and other foley, voices, music, room or "
        "outdoor acoustic space, loudness, and overall audio energy across the boundary. Do not mute, "
        "fade out, restart, replace, or drop the soundtrack at a segment boundary unless the current "
        "segment-local timeline explicitly requests an audio change. Hidden overlap is context only; "
        "the first audible moment after the boundary must sound like an immediate continuation of the "
        "preceding segment."
    )


def _format_local_time(seconds_value: float) -> str:
    seconds_value = max(0.0, float(seconds_value))
    rounded = int(round(seconds_value))
    if abs(seconds_value - rounded) < 1e-6:
        return f"{rounded:02d} sec"
    return f"{seconds_value:.2f} sec"


def explicit_prompt_sections(base_prompt: str) -> list[str]:
    """Split an author-provided prompt into explicitly local pass sections."""

    text = str(base_prompt or "").strip()
    if not text:
        return []
    starts = [match.start() for match in EXPLICIT_CONTINUATION_RE.finditer(text)]
    if not starts:
        return [text]
    chunks: list[str] = []
    first = text[: starts[0]].strip()
    if first:
        chunks.append(first)
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _sentence_chunks(lines: Sequence[str]) -> list[str]:
    chunks: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = re.split(r"(?<=[.!?])\s+", stripped)
        chunks.extend(part.strip() for part in parts if part.strip())
    return chunks


def _persistent_header(lines: Sequence[str]) -> tuple[list[str], int]:
    kept: list[str] = []
    dropped = 0
    for sentence in _sentence_chunks(lines):
        lowered = sentence.lower()
        if any(marker in lowered for marker in _RESTART_MARKERS):
            dropped += 1
            continue
        if _is_action_sentence(sentence):
            dropped += 1
            continue
        if any(marker in lowered for marker in _PERSISTENT_MARKERS):
            kept.append(sentence)
        else:
            dropped += 1
    return kept, dropped


def normalize_hybrid_picture_tags(
    prompt: str,
    *,
    anchor_roles: Sequence[str],
    reference_count: int,
) -> tuple[str, dict[str, Any]]:
    """Normalize intuitive input-slot Picture tags to native Ref2VA ordinals.

    Native H3 numbers only Ref2VA inputs.  A first/last-frame anchor is not a
    ``<Picture N>`` item.  Existing LongMedia prompts commonly number every
    connected ``image_N`` socket instead.  We only enable compatibility mapping
    when the prompt contains an ordinal that is impossible in native reference
    order but valid in input-slot order, making the detection unambiguous.
    """

    text = str(prompt or "")
    tags = [int(match.group(1)) for match in PICTURE_TAG_RE.finditer(text)]
    anchor_count = len(tuple(anchor_roles))
    ref_count = max(0, int(reference_count))
    total_inputs = anchor_count + ref_count
    impossible_native = any(index > ref_count for index in tags)
    valid_input_order = bool(tags) and all(1 <= index <= total_inputs for index in tags)
    use_input_compat = bool(anchor_count and impossible_native and valid_input_order)

    report: dict[str, Any] = {
        "mode": "input_socket_compat" if use_input_compat else "native_reference_order",
        "anchor_count": anchor_count,
        "reference_count": ref_count,
        "tags_seen": tags,
        "mapping": {},
        "invalid_tags": [],
    }

    if not use_input_compat:
        report["invalid_tags"] = sorted({index for index in tags if index > ref_count})
        return text, report

    roles = tuple(str(role) for role in anchor_roles)

    def replace_tag(match: re.Match[str]) -> str:
        source_index = int(match.group(1))
        if source_index <= anchor_count:
            replacement = f"the supplied {roles[source_index - 1]}"
        else:
            replacement = f"<Picture {source_index - anchor_count}>"
        report["mapping"][str(source_index)] = replacement
        return replacement

    normalized = PICTURE_TAG_RE.sub(replace_tag, text)
    return normalized, report


def build_segment_prompt(
    base_prompt: str,
    *,
    segment_index: int,
    segment_starts: Sequence[int],
    segment_lengths: Sequence[int],
    overlap_frames: int,
    passes: int,
    fps: float = 24.0,
) -> tuple[str, dict[str, Any]]:
    """Return one pass-local prompt with a globally consistent event timeline."""

    text = str(base_prompt or "").strip()
    index = int(segment_index)
    if not text:
        return text, {"segment_index": index, "empty": True}
    if index < 0 or index >= int(passes):
        raise IndexError(f"segment_index={index} is outside passes={int(passes)}")

    explicit = explicit_prompt_sections(text)
    if len(explicit) > 1 and index < len(explicit):
        return explicit[index], {
            "segment_index": index,
            "mode": "explicit_local_section",
            "section_index": index,
            "section_count": len(explicit),
        }

    context_start = int(segment_starts[index])
    local_overlap = max(0, int(overlap_frames)) if index > 0 else 0
    visible_start = context_start + local_overlap
    length = int(segment_lengths[index])
    visible_frames = max(1, length - local_overlap)
    visible_end = visible_start + visible_frames
    start_sec = float(visible_start) / float(fps)
    end_sec = float(visible_end) / float(fps)

    lines = text.splitlines()
    parsed_events: list[tuple[int, float, str]] = []
    for line_index, raw in enumerate(lines):
        match = SEGMENT_EVENT_RE.match(raw.strip())
        if match:
            parsed_events.append((line_index, float(match.group("sec")), match.group("body").strip()))

    report: dict[str, Any] = {
        "segment_index": index,
        "mode": "global_timeline" if parsed_events else "untimed_prompt",
        "context_start_frame": context_start,
        "visible_start_frame": visible_start,
        "visible_end_frame": visible_end,
        "visible_start_seconds": start_sec,
        "visible_end_seconds": end_sec,
        "events_total": len(parsed_events),
        "events_selected": 0,
        "events_dropped": 0,
        "header_sentences_dropped": 0,
    }

    if not parsed_events:
        if index == 0:
            return text, report
        header, dropped = _persistent_header(lines)
        report["header_sentences_dropped"] = dropped
        parts = [
            "Continue directly from the preceding video segment.",
            _continuation_contract(),
        ]
        if header:
            parts.append("\n".join(header))
        parts.append("Continue the current action naturally through this segment without a scene reset.")
        return "\n\n".join(part for part in parts if part.strip()), report

    first_event_line = parsed_events[0][0]
    last_event_line = parsed_events[-1][0]
    header_lines = lines[:first_event_line]
    footer_lines = lines[last_event_line + 1 :]
    between_notes = [
        raw for line_index, raw in enumerate(lines[first_event_line : last_event_line + 1], first_event_line)
        if raw.strip() and not SEGMENT_EVENT_RE.match(raw.strip())
    ]

    if index == 0:
        header = [line for line in header_lines if line.strip()]
    else:
        header, dropped = _persistent_header(header_lines)
        report["header_sentences_dropped"] = dropped

    selected: list[tuple[float, str]] = []
    for _line_index, global_sec, body in parsed_events:
        if global_sec + 1e-6 >= start_sec and global_sec < end_sec - 1e-6:
            selected.append((global_sec - start_sec, body))
    report["events_selected"] = len(selected)
    report["events_dropped"] = len(parsed_events) - len(selected)

    parts: list[str] = []
    if index > 0:
        parts.extend([
            "Continue directly from the preceding video segment.",
            _continuation_contract(),
        ])
        completed_state = _completed_state_lines(
            parsed_events, visible_start_seconds=start_sec, max_events=4,
        )
        if completed_state:
            parts.append("\n".join(completed_state))
            report["completed_state_events"] = len(completed_state) - 1
        else:
            report["completed_state_events"] = 0
    if header:
        parts.append("\n".join(header))
    if selected:
        event_lines = ["Segment-local timeline:"]
        event_lines.extend(f"{_format_local_time(sec)}: {body}" for sec, body in selected)
        parts.append("\n".join(event_lines))
    elif index > 0:
        parts.append("Continue the current action naturally through this segment without a scene reset.")
    persistent_tail = [line for line in (*between_notes, *footer_lines) if line.strip()]
    if persistent_tail:
        parts.append("\n".join(persistent_tail))
    return "\n\n".join(part for part in parts if part.strip()), report
