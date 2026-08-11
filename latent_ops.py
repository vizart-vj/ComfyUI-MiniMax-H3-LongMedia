"""Pure tensor operations for MiniMax H3 joint audio/video latents.

This module deliberately has no ComfyUI imports so its contracts can be tested
outside a running ComfyUI process. Node wrappers provide NestedTensor creation.
"""

from __future__ import annotations

import torch

FPS = 24
AUDIO_LATENT_FPS = 40


def align_frame_count(length: int) -> int:
    """Snap a requested frame count up to the H3 ``17*k + 5`` grid."""
    frame_count = max(5, int(length))
    remainder = (frame_count - 5) % 17
    if remainder:
        frame_count += 17 - remainder
    return frame_count


def video_latent_t(frame_count: int) -> int:
    """Return H3 video-latent time for an aligned pixel frame count."""
    frame_count = align_frame_count(frame_count)
    return 2 + 5 * ((frame_count - 5) // 17)


def audio_latent_t(frame_count: int) -> int:
    """Return H3's 40 Hz audio-latent length for a 24 fps clip."""
    return round(align_frame_count(frame_count) / FPS * AUDIO_LATENT_FPS)


def temporal_shape(length: int) -> tuple[int, int, int]:
    frame_count = align_frame_count(length)
    return frame_count, video_latent_t(frame_count), audio_latent_t(frame_count)


def frame_count_from_video_t(latent_t: int) -> int:
    """Invert H3's video temporal grid; reject non-H3 latent lengths."""
    latent_t = int(latent_t)
    if latent_t < 2 or (latent_t - 2) % 5:
        raise ValueError(
            f"MiniMax H3 video latent time must be 5*k+2, got {latent_t}."
        )
    return 5 + 17 * ((latent_t - 2) // 5)


def _validate_video(video: torch.Tensor) -> None:
    if video.ndim != 5 or video.shape[1] != 24:
        raise ValueError(
            f"MiniMax H3 video must be [B, 24, T, H, W] (24 channels), got {tuple(video.shape)}."
        )
    frame_count_from_video_t(video.shape[2])


def _validate_audio(audio: torch.Tensor) -> None:
    if audio.ndim != 4 or audio.shape[1] != 32 or audio.shape[2] != 2:
        raise ValueError(
            f"MiniMax H3 audio must be [B, 32, 2, T], got {tuple(audio.shape)}."
        )


def unpack_av_samples(av_latent: dict):
    """Extract ``(video, audio)`` from a ComfyUI H3 AV LATENT dictionary."""
    try:
        samples = av_latent["samples"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Expected a LATENT dictionary with a 'samples' entry.") from exc
    if not getattr(samples, "is_nested", False):
        raise ValueError("Expected MiniMax H3 NestedTensor AV samples.")
    streams = list(samples.unbind())
    if len(streams) != 2:
        raise ValueError(f"Expected exactly 2 AV streams, got {len(streams)}.")
    video, audio = streams
    _validate_video(video)
    _validate_audio(audio)
    if video.shape[0] != audio.shape[0]:
        raise ValueError(
            f"Video/audio batch sizes must match, got {video.shape[0]} and {audio.shape[0]}."
        )
    return video, audio


def _default_mask(stream: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "video":
        return torch.ones(
            (stream.shape[0], 1, stream.shape[2], 1, 1),
            dtype=torch.float32,
            device=stream.device,
        )
    return torch.ones(
        (stream.shape[0], 1, 1, stream.shape[-1]),
        dtype=torch.float32,
        device=stream.device,
    )


def _require_av_sync(video: torch.Tensor, audio: torch.Tensor, label: str) -> None:
    expected_audio_t = audio_latent_t(frame_count_from_video_t(video.shape[2]))
    if audio.shape[-1] != expected_audio_t:
        raise ValueError(
            f"{label} video/audio are not synchronized: video T={video.shape[2]} "
            f"requires audio T={expected_audio_t}, got {audio.shape[-1]}."
        )


def pack_av_latents(video_latent: dict, audio_latent: dict, nested_factory) -> dict:
    """Pack standalone H3 stream LATENTs, preserving independent noise masks."""
    video = video_latent["samples"]
    audio = audio_latent["samples"]
    _validate_video(video)
    _validate_audio(audio)
    if video.shape[0] != audio.shape[0]:
        raise ValueError(
            f"Video/audio batch sizes must match, got {video.shape[0]} and {audio.shape[0]}."
        )
    if video.shape[0] != 1:
        raise ValueError(
            f"The native MiniMax H3 packed model supports batch size 1, got {video.shape[0]}."
        )
    _require_av_sync(video, audio, "Packed")

    output = {
        key: value
        for key, value in video_latent.items()
        if key not in {"samples", "noise_mask"}
    }
    for key, value in audio_latent.items():
        if key not in {"samples", "noise_mask"} and key not in output:
            output[key] = value
    output["samples"] = nested_factory((video, audio))

    video_mask = video_latent.get("noise_mask")
    audio_mask = audio_latent.get("noise_mask")
    if video_mask is not None or audio_mask is not None:
        if video_mask is None:
            video_mask = _default_mask(video, "video")
        if audio_mask is None:
            audio_mask = _default_mask(audio, "audio")
        output["noise_mask"] = nested_factory((video_mask, audio_mask))
    return output


def split_av_latent(av_latent: dict) -> tuple[dict, dict]:
    """Split an H3 AV LATENT without leaking a nested mask into either stream."""
    video, audio = unpack_av_samples(av_latent)
    metadata = {
        key: value
        for key, value in av_latent.items()
        if key not in {"samples", "noise_mask"}
    }
    video_latent = {**metadata, "samples": video}
    audio_latent = {**metadata, "samples": audio}

    masks = av_latent.get("noise_mask")
    if masks is not None:
        if not getattr(masks, "is_nested", False):
            raise ValueError("H3 AV noise_mask must be a NestedTensor pair.")
        mask_streams = list(masks.unbind())
        if len(mask_streams) != 2:
            raise ValueError(
                f"Expected exactly 2 AV noise masks, got {len(mask_streams)}."
            )
        video_latent["noise_mask"], audio_latent["noise_mask"] = mask_streams
    return video_latent, audio_latent


def _existing_masks(av_latent: dict, video: torch.Tensor, audio: torch.Tensor):
    masks = av_latent.get("noise_mask")
    if masks is None:
        return _default_mask(video, "video"), _default_mask(audio, "audio")
    if not getattr(masks, "is_nested", False):
        raise ValueError("H3 AV noise_mask must be a NestedTensor pair.")
    streams = list(masks.unbind())
    if len(streams) != 2:
        raise ValueError(f"Expected exactly 2 AV noise masks, got {len(streams)}.")
    return streams[0], streams[1]


def _constant_mask(stream: torch.Tensor, kind: str, value: float) -> torch.Tensor:
    return _default_mask(stream, kind) * float(value)


def set_stream_denoise(
    av_latent: dict,
    video_denoise: float,
    audio_denoise: float,
    merge_mode: str,
    nested_factory,
) -> dict:
    """Set independent stream masks (0=preserve, 1=fully denoise)."""
    video, audio = unpack_av_samples(av_latent)
    strengths = (float(video_denoise), float(audio_denoise))
    if any(value < 0.0 or value > 1.0 for value in strengths):
        raise ValueError("Denoise values must be in the [0, 1] range.")

    old_video, old_audio = _existing_masks(av_latent, video, audio)
    new_video = _constant_mask(video, "video", strengths[0])
    new_audio = _constant_mask(audio, "audio", strengths[1])

    output = av_latent.copy()
    output["noise_mask"] = nested_factory(
        (
            _merge_mask(old_video, new_video, merge_mode),
            _merge_mask(old_audio, new_audio, merge_mode),
        )
    )
    return output


def _stream_from_latent(latent: dict, kind: str) -> torch.Tensor:
    samples = latent["samples"]
    if getattr(samples, "is_nested", False):
        video, audio = unpack_av_samples(latent)
        return video if kind == "video" else audio
    if kind == "video":
        _validate_video(samples)
    elif kind == "audio":
        _validate_audio(samples)
    else:
        raise ValueError(f"Unknown stream kind: {kind}")
    return samples


def _aligned_slice(source_size: int, target_size: int, alignment: str):
    count = min(source_size, target_size)
    if alignment == "start":
        source_start = target_start = 0
    elif alignment == "end":
        source_start = source_size - count
        target_start = target_size - count
    elif alignment == "center":
        source_start = (source_size - count) // 2
        target_start = (target_size - count) // 2
    else:
        raise ValueError(f"Unknown alignment: {alignment}")
    return (
        slice(source_start, source_start + count),
        slice(target_start, target_start + count),
    )


def _crop_pad_stream(
    source: torch.Tensor,
    target: torch.Tensor,
    kind: str,
    alignment: str,
) -> torch.Tensor:
    if source.shape[:2] != target.shape[:2]:
        raise ValueError(
            f"crop_pad requires matching batch/channels, got {tuple(source.shape[:2])} "
            f"and {tuple(target.shape[:2])}."
        )
    output = torch.zeros_like(target)
    if kind == "video":
        source_t, target_t = _aligned_slice(source.shape[2], target.shape[2], alignment)
        source_h, target_h = _aligned_slice(source.shape[3], target.shape[3], "center")
        source_w, target_w = _aligned_slice(source.shape[4], target.shape[4], "center")
        output[:, :, target_t, target_h, target_w] = source[
            :, :, source_t, source_h, source_w
        ].to(output)
    else:
        source_t, target_t = _aligned_slice(source.shape[-1], target.shape[-1], alignment)
        output[..., target_t] = source[..., source_t].to(output)
    return output


def _fit_stream(
    source: torch.Tensor,
    target: torch.Tensor,
    kind: str,
    fit_mode: str,
    alignment: str,
) -> torch.Tensor:
    if fit_mode == "strict":
        if tuple(source.shape) != tuple(target.shape):
            raise ValueError(
                f"Strict {kind} replacement requires shape {tuple(target.shape)}, "
                f"got {tuple(source.shape)}."
            )
        return source
    if fit_mode == "crop_pad":
        return _crop_pad_stream(source, target, kind, alignment)
    raise ValueError(f"Unknown fit mode: {fit_mode}")


def replace_stream(
    av_latent: dict,
    replacement_latent: dict,
    kind: str,
    fit_mode: str,
    alignment: str,
    denoise: float,
    nested_factory,
) -> dict:
    """Replace one H3 stream while keeping the other stream and its mask."""
    if not 0.0 <= float(denoise) <= 1.0:
        raise ValueError("Denoise must be in the [0, 1] range.")

    video, audio = unpack_av_samples(av_latent)
    replacement = _stream_from_latent(replacement_latent, kind)
    target = video if kind == "video" else audio
    replacement = _fit_stream(replacement, target, kind, fit_mode, alignment)

    old_video_mask, old_audio_mask = _existing_masks(av_latent, video, audio)
    if kind == "video":
        video = replacement
        old_video_mask = _constant_mask(video, "video", denoise)
    elif kind == "audio":
        audio = replacement
        old_audio_mask = _constant_mask(audio, "audio", denoise)
    else:
        raise ValueError(f"Unknown stream kind: {kind}")

    output = av_latent.copy()
    output["samples"] = nested_factory((video, audio))
    output["noise_mask"] = nested_factory((old_video_mask, old_audio_mask))
    return output


def inject_leading_video_frame(
    av_latent: dict,
    frame_latent: dict,
    denoise: float,
    nested_factory,
) -> dict:
    """Overwrite only the first video-latent frame, with its own partial-denoise mask.

    Unlike ``replace_stream``, this touches a single leading frame instead of the
    whole stream — audio and the remaining video frames (and their existing masks)
    are left exactly as they were. ``denoise=0`` reproduces the old hard pixel-pin
    behavior at the latent level (frame is exact, but sampling still sees it as
    context); ``denoise>0`` lets the sampler soften the transition into it.
    """
    if not 0.0 <= float(denoise) <= 1.0:
        raise ValueError("Denoise must be in the [0, 1] range.")

    video, audio = unpack_av_samples(av_latent)
    frame = _stream_from_latent(frame_latent, "video")
    if frame.shape[2] != 1:
        frame = frame[:, :, :1]
    if frame.shape[:2] != video.shape[:2] or frame.shape[3:] != video.shape[3:]:
        raise ValueError(
            f"Leading frame shape {tuple(frame.shape)} does not match the "
            f"target video canvas {tuple(video.shape)}."
        )

    old_video_mask, old_audio_mask = _existing_masks(av_latent, video, audio)
    video = video.clone()
    video[:, :, :1] = frame.to(video)
    new_video_mask = old_video_mask.clone()
    new_video_mask[:, :, :1] = float(denoise)

    output = av_latent.copy()
    output["samples"] = nested_factory((video, audio))
    output["noise_mask"] = nested_factory((new_video_mask, old_audio_mask))
    return output


def _merge_mask(base: torch.Tensor, update: torch.Tensor, mode: str) -> torch.Tensor:
    update = update.to(base)
    if mode == "replace":
        return update
    if mode == "multiply":
        return base * update
    if mode == "minimum":
        return torch.minimum(base, update)
    if mode == "maximum":
        return torch.maximum(base, update)
    raise ValueError(f"Unknown mask merge mode: {mode}")


def _mask_to_video_grid(mask: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
    """Convert a ComfyUI MASK into [1,1,T,H,W] on the H3 video grid."""
    if mask.ndim == 2:
        mask_5d = mask[None, None, None]
    elif mask.ndim == 3:
        # ComfyUI MASK is normally [frames, H, W].
        mask_5d = mask[None, None]
    elif mask.ndim == 4 and mask.shape[0] == 1:
        # Explicit [B=1, frames, H, W].
        mask_5d = mask[:, None]
    elif mask.ndim == 4 and mask.shape[1] == 1:
        # Image-style [frames, 1, H, W].
        mask_5d = mask[:, 0][None, None]
    else:
        raise ValueError(
            "Video inpaint MASK must be [H,W], [frames,H,W], "
            "[1,frames,H,W], or [frames,1,H,W]."
        )
    mask_5d = mask_5d.to(device=video.device, dtype=torch.float32).clamp(0.0, 1.0)
    return torch.nn.functional.interpolate(
        mask_5d,
        size=(video.shape[2], video.shape[3], video.shape[4]),
        mode="trilinear",
        align_corners=False,
    ).to(dtype=video.dtype)


def apply_video_inpaint_mask(
    av_latent: dict,
    mask: torch.Tensor,
    denoise_inside: float,
    denoise_outside: float,
    audio_denoise: float,
    merge_mode: str,
    nested_factory,
) -> dict:
    """Apply a pixel/video mask to the H3 video denoise mask."""
    strengths = (
        float(denoise_inside),
        float(denoise_outside),
        float(audio_denoise),
    )
    if any(value < 0.0 or value > 1.0 for value in strengths):
        raise ValueError("Denoise values must be in the [0, 1] range.")

    video, audio = unpack_av_samples(av_latent)
    spatial_mask = _mask_to_video_grid(mask, video)
    video_update = strengths[1] + spatial_mask * (strengths[0] - strengths[1])
    audio_update = _constant_mask(audio, "audio", strengths[2])
    old_video, old_audio = _existing_masks(av_latent, video, audio)

    output = av_latent.copy()
    output["noise_mask"] = nested_factory(
        (
            _merge_mask(old_video, video_update, merge_mode),
            _merge_mask(old_audio, audio_update, merge_mode),
        )
    )
    return output


def merge_av_latents(
    target_av: dict,
    source_av: dict,
    video_mix: float,
    audio_mix: float,
    fit_mode: str,
    alignment: str,
    video_denoise: float,
    audio_denoise: float,
    nested_factory,
) -> dict:
    """Blend streams from source AV into target AV and assign output masks."""
    values = (
        float(video_mix),
        float(audio_mix),
        float(video_denoise),
        float(audio_denoise),
    )
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Mix and denoise values must be in the [0, 1] range.")

    target_video, target_audio = unpack_av_samples(target_av)
    source_video, source_audio = unpack_av_samples(source_av)
    _require_av_sync(target_video, target_audio, "Target")
    _require_av_sync(source_video, source_audio, "Source")
    source_video = _fit_stream(
        source_video, target_video, "video", fit_mode, alignment
    ).to(target_video)
    source_audio = _fit_stream(
        source_audio, target_audio, "audio", fit_mode, alignment
    ).to(target_audio)

    video = torch.lerp(target_video, source_video, values[0])
    audio = torch.lerp(target_audio, source_audio, values[1])
    video_mask = _constant_mask(video, "video", values[2])
    audio_mask = _constant_mask(audio, "audio", values[3])

    output = {
        key: value
        for key, value in target_av.items()
        if key not in {"samples", "noise_mask"}
    }
    output["samples"] = nested_factory((video, audio))
    output["noise_mask"] = nested_factory((video_mask, audio_mask))
    return output


def _aligned_overlap_frames(requested: int, maximum: int) -> int:
    requested = min(max(0, int(requested)), int(maximum))
    if requested < 5:
        raise ValueError(
            "Latent-space continuation overlap must be at least 5 frames to preserve "
            "the MiniMax H3 temporal grid. Concatenate decoded clips for zero overlap."
        )
    return 5 + 17 * ((requested - 5) // 17)


def prepare_continuation(
    source_av: dict,
    length: int,
    overlap_frames: int,
    video_context_denoise: float,
    audio_context_denoise: float,
    nested_factory,
):
    """Build a new AV latent with a synchronized source tail pinned at the start."""
    if not 0.0 <= float(video_context_denoise) <= 1.0:
        raise ValueError("Video context denoise must be in the [0, 1] range.")
    if not 0.0 <= float(audio_context_denoise) <= 1.0:
        raise ValueError("Audio context denoise must be in the [0, 1] range.")

    source_video, source_audio = unpack_av_samples(source_av)
    _require_av_sync(source_video, source_audio, "Source")
    source_frames = frame_count_from_video_t(source_video.shape[2])
    frame_count, target_video_t, target_audio_t = temporal_shape(length)
    overlap_frames = _aligned_overlap_frames(
        overlap_frames, min(source_frames, frame_count)
    )
    overlap_video_t = video_latent_t(overlap_frames) if overlap_frames else 0
    overlap_audio_t = (
        round(overlap_frames / FPS * AUDIO_LATENT_FPS) if overlap_frames else 0
    )
    if source_audio.shape[-1] < overlap_audio_t:
        raise ValueError(
            f"Source audio has {source_audio.shape[-1]} latent frames, but "
            f"the {overlap_frames}-frame overlap needs {overlap_audio_t}."
        )

    video = torch.zeros(
        (
            source_video.shape[0],
            source_video.shape[1],
            target_video_t,
            source_video.shape[3],
            source_video.shape[4],
        ),
        dtype=source_video.dtype,
        device=source_video.device,
    )
    audio = torch.zeros(
        (
            source_audio.shape[0],
            source_audio.shape[1],
            source_audio.shape[2],
            target_audio_t,
        ),
        dtype=source_audio.dtype,
        device=source_audio.device,
    )
    if overlap_video_t:
        video[:, :, :overlap_video_t] = source_video[:, :, -overlap_video_t:]
    if overlap_audio_t:
        audio[..., :overlap_audio_t] = source_audio[..., -overlap_audio_t:]

    video_mask = _constant_mask(video, "video", 1.0)
    audio_mask = _constant_mask(audio, "audio", 1.0)
    if overlap_video_t:
        # Hold the inherited overlap at one constant denoise value instead of
        # ramping toward 1.0. For H3 long-media generation this preserves
        # inter-segment continuity much better than a soft ramp.
        video_mask[:, :, :overlap_video_t] = float(video_context_denoise)
    if overlap_audio_t:
        audio_mask[..., :overlap_audio_t] = float(audio_context_denoise)

    output = {
        key: value
        for key, value in source_av.items()
        if key not in {"samples", "noise_mask"}
    }
    output["samples"] = nested_factory((video, audio))
    output["noise_mask"] = nested_factory((video_mask, audio_mask))
    return output, frame_count, overlap_frames


def feather_continuation_overlap(
    prepared_av: dict,
    overlap_frames: int,
    max_denoise: float,
    nested_factory,
) -> dict:
    """Let a continuation adapt inside its existing overlap without shifting time."""
    max_denoise = float(max_denoise)
    if not 0.0 <= max_denoise <= 1.0:
        raise ValueError("Overlap denoise must be in the [0, 1] range.")

    video, audio = unpack_av_samples(prepared_av)
    _require_av_sync(video, audio, "Prepared continuation")
    frame_count = frame_count_from_video_t(video.shape[2])
    aligned_overlap = _aligned_overlap_frames(overlap_frames, frame_count)
    overlap_video_t = video_latent_t(aligned_overlap)
    video_mask, audio_mask = _existing_masks(prepared_av, video, audio)
    feathered_mask = video_mask.clone()
    ramp = torch.linspace(
        0.0,
        max_denoise,
        overlap_video_t,
        dtype=feathered_mask.dtype,
        device=feathered_mask.device,
    )
    feathered_mask[:, :, :overlap_video_t] = ramp.view(
        1, 1, overlap_video_t, 1, 1
    )

    output = prepared_av.copy()
    output["noise_mask"] = nested_factory((feathered_mask, audio_mask))
    return output


def inject_continuation_sources(
    prepared_av: dict,
    source_video_latent: dict | None,
    source_audio_latent: dict | None,
    overlap_frames: int,
    video_denoise: float,
    audio_denoise: float,
    nested_factory,
) -> dict:
    """Inject source media after a frozen continuation overlap.

    The overlap remains the sampled tail from the preceding pass. New source
    video/audio only replaces the suffix, and each suffix receives its own
    denoise strength.
    """
    strengths = (float(video_denoise), float(audio_denoise))
    if any(value < 0.0 or value > 1.0 for value in strengths):
        raise ValueError("Denoise values must be in the [0, 1] range.")

    video, audio = unpack_av_samples(prepared_av)
    _require_av_sync(video, audio, "Prepared continuation")
    frame_count = frame_count_from_video_t(video.shape[2])
    aligned_overlap = _aligned_overlap_frames(overlap_frames, frame_count)
    overlap_video_t = video_latent_t(aligned_overlap)
    overlap_audio_t = round(aligned_overlap / FPS * AUDIO_LATENT_FPS)
    video_mask, audio_mask = _existing_masks(prepared_av, video, audio)

    result_video = video.clone() if source_video_latent is not None else video
    result_audio = audio.clone() if source_audio_latent is not None else audio
    result_video_mask = video_mask.clone()
    result_audio_mask = audio_mask.clone()

    if source_video_latent is not None:
        source_video = _stream_from_latent(source_video_latent, "video")
        source_video = _fit_stream(source_video, video, "video", "strict", "start")
        result_video[:, :, overlap_video_t:] = source_video[
            :, :, overlap_video_t:
        ].to(result_video)
        result_video_mask[:, :, overlap_video_t:] = strengths[0]

    if source_audio_latent is not None:
        source_audio = _stream_from_latent(source_audio_latent, "audio")
        source_audio = _fit_stream(source_audio, audio, "audio", "strict", "start")
        result_audio[..., overlap_audio_t:] = source_audio[..., overlap_audio_t:].to(
            result_audio
        )
        result_audio_mask[..., overlap_audio_t:] = strengths[1]

    output = prepared_av.copy()
    output["samples"] = nested_factory((result_video, result_audio))
    output["noise_mask"] = nested_factory(
        (result_video_mask, result_audio_mask)
    )
    return output


def stitch_continuation(
    previous_av: dict,
    continuation_av: dict,
    overlap_frames: int,
    nested_factory,
    blend_video_overlap: bool = False,
    offload_to_cpu: bool = False,
):
    """Join continuation on the existing AV overlap without shifting time.

    ``offload_to_cpu`` moves both operands to CPU before concatenating, and
    the result is therefore CPU-resident too. This matters for long multi-pass
    runs: this function is called once per pass with ``previous_av`` being the
    *entire* accumulated clip so far, so leaving that accumulator on the GPU
    means its footprint keeps growing across passes on top of whatever VRAM
    the next pass's sampling needs. Nothing downstream needs it on-device
    until final decode — each pass's own sampling only ever consumes the
    single most recent segment (bounded size), never this accumulator — so
    offloading it is free correctness-wise and just trades a small per-pass
    host<->device copy for materially lower peak VRAM.
    """
    previous_video, previous_audio = unpack_av_samples(previous_av)
    next_video, next_audio = unpack_av_samples(continuation_av)
    if offload_to_cpu:
        previous_video = previous_video.cpu()
        previous_audio = previous_audio.cpu()
        next_video = next_video.cpu()
        next_audio = next_audio.cpu()
    _require_av_sync(previous_video, previous_audio, "Previous")
    _require_av_sync(next_video, next_audio, "Continuation")
    if (
        previous_video.shape[:2] != next_video.shape[:2]
        or previous_video.shape[3:] != next_video.shape[3:]
    ):
        raise ValueError(
            "Continuation video batch/channels/spatial shape must match the previous video."
        )
    if previous_audio.shape[:3] != next_audio.shape[:3]:
        raise ValueError("Continuation audio batch/channels shape must match the previous audio.")

    previous_frames = frame_count_from_video_t(previous_video.shape[2])
    next_frames = frame_count_from_video_t(next_video.shape[2])
    overlap_frames = _aligned_overlap_frames(
        overlap_frames, min(previous_frames, next_frames)
    )
    overlap_video_t = video_latent_t(overlap_frames) if overlap_frames else 0
    overlap_audio_t = (
        round(overlap_frames / FPS * AUDIO_LATENT_FPS) if overlap_frames else 0
    )
    if next_audio.shape[-1] < overlap_audio_t:
        raise ValueError("Continuation audio is shorter than the requested overlap.")

    if blend_video_overlap:
        t = torch.linspace(
            0.0,
            1.0,
            overlap_video_t,
            dtype=previous_video.dtype,
            device=previous_video.device,
        )
        # Smoothstep: S(t) = 3t² − 2t³ — zero slope at both endpoints
        # eliminates the visible seam that linear lerp creates.
        blend = (t * t * (3.0 - 2.0 * t)).view(1, 1, overlap_video_t, 1, 1)
        blended_overlap = torch.lerp(
            previous_video[:, :, -overlap_video_t:],
            next_video[:, :, :overlap_video_t].to(previous_video),
            blend,
        )
        video = torch.cat(
            (
                previous_video[:, :, :-overlap_video_t],
                blended_overlap,
                next_video[:, :, overlap_video_t:].to(previous_video),
            ),
            dim=2,
        )
    else:
        video = torch.cat((previous_video, next_video[:, :, overlap_video_t:]), dim=2)
    raw_audio = torch.cat((previous_audio, next_audio[..., overlap_audio_t:]), dim=-1)
    total_frames = previous_frames + next_frames - overlap_frames
    expected_audio_t = round(total_frames / FPS * AUDIO_LATENT_FPS)
    if raw_audio.shape[-1] != expected_audio_t:
        audio_target = torch.zeros(
            (*raw_audio.shape[:-1], expected_audio_t),
            dtype=raw_audio.dtype,
            device=raw_audio.device,
        )
        audio = _crop_pad_stream(raw_audio, audio_target, "audio", "start")
    else:
        audio = raw_audio

    output = {
        key: value
        for key, value in previous_av.items()
        if key not in {"samples", "noise_mask"}
    }
    output["samples"] = nested_factory((video, audio))
    return output, total_frames


def describe_av(av_latent: dict) -> dict:
    """Return geometry, timing and synchronization facts for an H3 AV latent."""
    video, audio = unpack_av_samples(av_latent)
    frames = frame_count_from_video_t(video.shape[2])
    expected_audio_t = audio_latent_t(frames)
    return {
        "batch": video.shape[0],
        "width": video.shape[4] * 16,
        "height": video.shape[3] * 16,
        "frames": frames,
        "duration_seconds": frames / FPS,
        "video_latent_t": video.shape[2],
        "audio_latent_t": audio.shape[-1],
        "expected_audio_latent_t": expected_audio_t,
        "audio_duration_seconds": audio.shape[-1] / AUDIO_LATENT_FPS,
        "synchronized": audio.shape[-1] == expected_audio_t,
        "video_dtype": str(video.dtype).removeprefix("torch."),
        "audio_dtype": str(audio.dtype).removeprefix("torch."),
        "video_device": str(video.device),
        "audio_device": str(audio.device),
        "has_noise_mask": av_latent.get("noise_mask") is not None,
    }
