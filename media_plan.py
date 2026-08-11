"""Pure planning utilities for MiniMax H3 long-media orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch


FPS = 24
MIN_FRAMES = 5
FRAME_STRIDE = 17


@dataclass(frozen=True)
class LongMediaPlan:
    mode: str
    duration_basis: str
    total_duration: float
    output_frames: int
    segment_frames: int
    segment_lengths: tuple[int, ...]
    segment_starts: tuple[int, ...]
    overlap_frames: int
    step_frames: int
    passes: int
    generated_frames: int
    trim_frames: int
    resolution_mode: str
    video_strength: float
    audio_strength: float
    video_fps: float
    source_video: Any = None
    source_audio: Any = None
    # Optional conditioning-only audio. Kept separate from source_audio so a
    # driving/reference track does not get injected into the target audio latent.
    reference_audio: Any = None
    video_vae: Any = None
    audio_vae: Any = None
    final_audio_override: Any = None
    final_audio_track_count: int = 0
    first_frame_override: Any = None
    first_frame_mode: str = "latent_inject"
    first_frame_denoise: float = 0.25
    first_frame_blend_frames: int = 3


def _align_up_frames(frame_count: int) -> int:
    frame_count = max(MIN_FRAMES, int(frame_count))
    return frame_count + (MIN_FRAMES - frame_count) % FRAME_STRIDE


def _align_down_frames(frame_count: int) -> int:
    frame_count = max(MIN_FRAMES, int(frame_count))
    return frame_count - (frame_count - MIN_FRAMES) % FRAME_STRIDE


def collect_numbered_inputs(values: dict[str, Any], prefix: str, maximum: int):
    """Return connected dynamic media inputs in stable numeric order."""
    collected = []
    for index in range(1, maximum + 1):
        value = values.get(f"{prefix}{index}")
        if value is not None:
            collected.append(value)
    return collected


def slice_video_segment(
    video,
    start_frame: int,
    length_frames: int,
    source_fps: float,
):
    """Sample a source IMAGE batch onto the H3 24 fps segment timeline."""
    if video is None or video.shape[0] <= 0:
        raise ValueError("Video input contains no frames.")
    timeline = torch.arange(
        start_frame,
        start_frame + length_frames,
        device=video.device,
        dtype=torch.float64,
    )
    indices = torch.floor(timeline * float(source_fps) / FPS).long()
    indices = indices.clamp_(0, video.shape[0] - 1)
    return video[indices]


def slice_audio_segment(audio, start_frame: int, length_frames: int):
    """Crop one H3 segment from AUDIO and silence-pad beyond the source end."""
    waveform = audio["waveform"][:1]
    sample_rate = int(audio["sample_rate"])
    start_sample = math.floor(start_frame / FPS * sample_rate)
    target_samples = round(length_frames / FPS * sample_rate)
    available = waveform[..., start_sample : start_sample + target_samples]
    if available.shape[-1] < target_samples:
        available = torch.nn.functional.pad(
            available,
            (0, target_samples - available.shape[-1]),
        )
    return {"waveform": available, "sample_rate": sample_rate}


def _audio_duration(audio: Any) -> float:
    if not isinstance(audio, dict):
        return 0.0
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if waveform is None or not sample_rate or waveform.shape[-1] <= 0:
        return 0.0
    return float(waveform.shape[-1]) / float(sample_rate)


def _video_duration(video: Any, video_fps: float) -> float:
    if video is None or not hasattr(video, "shape") or not video.shape:
        return 0.0
    if video.shape[0] <= 0 or video_fps <= 0:
        return 0.0
    return float(video.shape[0]) / float(video_fps)


def _longest_media(items: Sequence[Any], duration_fn):
    measured = [(duration_fn(item), item) for item in items]
    measured = [entry for entry in measured if entry[0] > 0]
    return max(measured, key=lambda entry: entry[0]) if measured else (0.0, None)


def _plan_segments(output_frames: int, max_segment_frames: int, overlap_frames: int):
    first = _align_up_frames(min(output_frames, max_segment_frames))
    lengths = [first]
    starts = [0]
    covered = first
    while covered < output_frames:
        remaining = output_frames - covered
        requested = min(max_segment_frames, overlap_frames + remaining)
        length = _align_up_frames(requested)
        if length > max_segment_frames:
            length = max_segment_frames
        if length <= overlap_frames:
            raise ValueError("Continuation segment must be longer than its overlap.")
        starts.append(covered - overlap_frames)
        lengths.append(length)
        covered += length - overlap_frames
    return tuple(lengths), tuple(starts), covered


def build_media_plan(
    *,
    audios,
    videos,
    manual_duration: float,
    duration_source: str,
    segment_seconds: float,
    overlap_frames: int,
    video_fps: float,
    resolution_mode: str,
    video_strength: float,
    audio_strength: float,
) -> LongMediaPlan:
    if resolution_mode not in {"match", "max"}:
        raise ValueError("resolution_mode must be 'match' or 'max'.")
    if video_fps <= 0:
        raise ValueError("video_fps must be positive.")
    if manual_duration <= 0 or segment_seconds <= 0:
        raise ValueError("Durations must be positive.")

    first_audio = audios[0] if audios else None
    first_video = videos[0] if videos else None
    first_audio_duration = _audio_duration(first_audio) if first_audio is not None else 0.0
    first_video_duration = (
        _video_duration(first_video, video_fps) if first_video is not None else 0.0
    )
    longest_audio_duration, longest_audio = _longest_media(audios, _audio_duration)
    longest_video_duration, longest_video = _longest_media(
        videos,
        lambda video: _video_duration(video, video_fps),
    )
    if duration_source == "longest_input":
        audio_duration, source_audio = longest_audio_duration, longest_audio
        video_duration, source_video = longest_video_duration, longest_video
    else:
        audio_duration, source_audio = first_audio_duration, first_audio
        video_duration, source_video = first_video_duration, first_video

    if source_audio is None and source_video is None:
        mode = "t2v"
    elif source_video is None:
        mode = "audio_to_video"
    elif source_audio is None:
        mode = "video_to_video"
    else:
        mode = "video_audio_to_video"

    choices = {
        "manual": (float(manual_duration), "manual"),
        "audio": (audio_duration, "audio"),
        "video": (video_duration, "video"),
        "longest_input": (
            max(audio_duration, video_duration),
            "audio" if audio_duration >= video_duration else "video",
        ),
    }
    if duration_source == "auto":
        if audio_duration > 0:
            total_duration, duration_basis = audio_duration, "audio"
        elif video_duration > 0:
            total_duration, duration_basis = video_duration, "video"
        else:
            total_duration, duration_basis = float(manual_duration), "manual"
    elif duration_source in choices:
        total_duration, duration_basis = choices[duration_source]
        if total_duration <= 0:
            total_duration, duration_basis = float(manual_duration), "manual_fallback"
    else:
        raise ValueError(f"Unknown duration source: {duration_source}")

    output_frames = max(1, math.floor(total_duration * FPS))
    max_segment_frames = _align_up_frames(math.floor(segment_seconds * FPS))
    actual_overlap = _align_down_frames(overlap_frames)
    if actual_overlap >= max_segment_frames:
        actual_overlap = _align_down_frames(max_segment_frames - FRAME_STRIDE)
    if actual_overlap >= max_segment_frames:
        raise ValueError("overlap_frames must be shorter than a segment.")

    segment_lengths, segment_starts, generated_frames = _plan_segments(
        output_frames,
        max_segment_frames,
        actual_overlap,
    )
    return LongMediaPlan(
        mode=mode,
        duration_basis=duration_basis,
        total_duration=float(total_duration),
        output_frames=output_frames,
        segment_frames=segment_lengths[0],
        segment_lengths=segment_lengths,
        segment_starts=segment_starts,
        overlap_frames=actual_overlap,
        step_frames=segment_lengths[0] - actual_overlap,
        passes=len(segment_lengths),
        generated_frames=generated_frames,
        trim_frames=generated_frames - output_frames,
        resolution_mode=resolution_mode,
        video_strength=float(video_strength),
        audio_strength=float(audio_strength),
        video_fps=float(video_fps),
        source_video=source_video,
        source_audio=source_audio,
    )
