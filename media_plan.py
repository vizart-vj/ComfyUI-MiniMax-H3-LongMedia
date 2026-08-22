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
    audio_output_mode: str = "auto"
    # v0.3.104 lip-sync: native Ref2VA Audio1 + local native H3 Audio Guide.
    lip_sync_native_audio_guide: bool = False
    # v0.3.113: authoritative source audio is injected into the target AV latent
    # and frozen (noise mask 0) so the joint H3 transformer predicts video
    # against the exact speech timeline instead of treating Audio1 only as a ref.
    lip_sync_target_audio_locked: bool = False
    first_frame_override: Any = None
    first_frame_mode: str = "latent_inject"
    first_frame_denoise: float = 0.25
    first_frame_blend_frames: int = 3
    # True when image_1 was written into the target video latent before pass 0.
    # Kept separate from first_frame_override so latent mode does not retain a
    # full-resolution pixel tensor throughout long-media sampling.
    first_frame_latent_injected: bool = False
    # Per-pass text conditionings are pre-encoded inside Setup while the TE is
    # intentionally resident.  Never store CLIP/TE/model-patcher objects here.
    segment_positive_conditionings: Any = None
    segment_prompt_summaries: Any = None
    # v0.3.85 MultiClip: optional per-pass seeds; None = sampler base seed + clip index.
    segment_seeds: Any = None
    # V63 storyboard bridge: ready per-pass AV latents and decoded boundary index.
    storyboard_segment_avs: Any = None
    storyboard_bridge_frame: int = -1
    # v0.3.39 segmented mode: preserve the known-good hybrid sampling anchor
    # but hide only its literal decoded frame-0 from final output.
    suppress_visible_opening_anchor: bool = False
    regression_safe_segmented_conditioning: bool = False
    # v0.3.43: original still-image refs are used only on pass 0. Later passes
    # continue from generated AV state and must not carry full spatial image ref latents.
    decouple_original_image_refs_after_pass0: bool = False
    # v0.3.111 unified clip executor: legacy/single, fixed segmentation, or planned MultiClip.
    timeline_policy: str = "legacy"
    # v0.4.6: explicit workflow ownership. Never infer segmentation from passes.
    workflow_mode: str = "hybrid_auto"
    segmentation_active: bool = False
    # Production console guard. False enables full LongMedia diagnostics for profiling.
    release_guard: bool = True


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


def _plan_segments(output_frames: int, nominal_new_frames: int, overlap_frames: int):
    """Plan H3 passes where segment duration means NEW output timeline.

    ``nominal_new_frames`` is the user-facing segment duration converted to
    frames.  Overlap is additional continuation context and must not consume
    that duration.  This keeps e.g. 15 s / 5 s at exactly three passes instead
    of producing a fourth tail pass just because overlap reduced each pass's
    useful coverage.

    H3 pass lengths still have to satisfy the 5 + 17*n temporal geometry, so
    individual passes may be a few frames longer than the nominal timeline
    chunk.  The final stitched result is trimmed back to ``output_frames``.
    """
    nominal_new_frames = max(1, int(nominal_new_frames))
    pass_count = max(1, math.ceil(output_frames / nominal_new_frames))

    first_target = min(output_frames, nominal_new_frames)
    first = _align_up_frames(first_target)
    lengths = [first]
    starts = [0]
    covered = first

    for pass_index in range(1, pass_count):
        remaining = max(0, output_frames - covered)
        remaining_passes = pass_count - pass_index
        # Share the still-uncovered output across the remaining passes.  The
        # overlap is prepended as context and is not counted as new coverage.
        desired_new = max(1, math.ceil(remaining / remaining_passes))
        requested = overlap_frames + desired_new
        length = _align_up_frames(requested)
        if length <= overlap_frames:
            length = _align_up_frames(overlap_frames + 1)
        starts.append(max(0, covered - overlap_frames))
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
    nominal_new_frames = max(1, math.floor(segment_seconds * FPS))
    # H3 overlap itself must also land on the 5 + 17*n temporal grid.  Keep it
    # shorter than the nominal user-visible timeline chunk; continuation pass
    # length may exceed segment_seconds because overlap is extra context.
    actual_overlap = _align_down_frames(overlap_frames)
    if actual_overlap >= nominal_new_frames:
        actual_overlap = _align_down_frames(max(MIN_FRAMES, nominal_new_frames - FRAME_STRIDE))
    if actual_overlap >= nominal_new_frames:
        raise ValueError("overlap_frames must be shorter than the new timeline span of a segment.")

    segment_lengths, segment_starts, generated_frames = _plan_segments(
        output_frames,
        nominal_new_frames,
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
        step_frames=nominal_new_frames,
        passes=len(segment_lengths),
        generated_frames=generated_frames,
        trim_frames=generated_frames - output_frames,
        resolution_mode=resolution_mode,
        video_fps=float(video_fps),
        source_video=source_video,
        source_audio=source_audio,
    )
