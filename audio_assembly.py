"""Audio layout normalization for Director final assembly.

This module is intentionally independent of ComfyUI so its tensor contract can be
unit-tested without importing the whole node package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class AudioConcatReport:
    input_shapes: tuple[tuple[int, ...], ...]
    target_channels: int
    mono_upmixes: int
    output_shape: tuple[int, ...]
    output_dtype: str
    output_device: str


def _canonical_audio_piece(waveform: torch.Tensor) -> torch.Tensor:
    """Return one ComfyUI timeline waveform as contiguous ``[1, C, L]``.

    The Director timeline is not a batched render.  Existing LongMedia semantics
    select the first audio batch, so this helper preserves that contract while
    making channel/time layout explicit before concatenation.
    """
    if not torch.is_tensor(waveform):
        raise TypeError(f"Director audio piece must be a torch.Tensor, got {type(waveform)!r}.")

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2:
        # Accept [C,L] and [L,C].  Audio channel counts are expected to be tiny.
        if int(waveform.shape[-1]) <= 8 and int(waveform.shape[0]) > int(waveform.shape[-1]):
            waveform = waveform.transpose(0, 1)
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 3:
        # Prefer [B,C,L].  A tiny final dimension with a large middle dimension is [B,L,C].
        if int(waveform.shape[-1]) <= 8 and int(waveform.shape[1]) > int(waveform.shape[-1]):
            waveform = waveform.movedim(-1, 1)
    else:
        raise ValueError(
            "Director audio piece must be [L], [C,L], [L,C], [B,C,L], or [B,L,C], "
            f"got {tuple(waveform.shape)}."
        )

    if int(waveform.shape[0]) < 1 or int(waveform.shape[1]) < 1 or int(waveform.shape[-1]) < 1:
        raise ValueError(f"Director audio piece has an empty dimension: {tuple(waveform.shape)}.")

    waveform = waveform[:1]
    channels = int(waveform.shape[1])
    if channels > 2:
        raise ValueError(
            "Director mixed MEDIA/GENERATED assembly currently supports mono/stereo audio only; "
            f"got {channels} channels with shape {tuple(waveform.shape)}."
        )
    return waveform.contiguous()


def concatenate_director_audio(waveforms: Sequence[torch.Tensor]) -> tuple[torch.Tensor, AudioConcatReport]:
    """Concatenate timeline audio after lossless mono/stereo layout normalization.

    H3 AudioVAE output is stereo while imported media may be mono.  Concatenating
    those tensors directly fails because ``torch.cat(..., dim=-1)`` requires every
    non-time dimension to match.  The only channel conversion performed here is
    mono -> stereo duplication when any timeline piece is stereo.  Existing stereo
    samples are not remixed or altered.
    """
    if not waveforms:
        raise ValueError("Director audio concatenation requires at least one waveform.")

    pieces = [_canonical_audio_piece(w) for w in waveforms]
    input_shapes = tuple(tuple(int(x) for x in w.shape) for w in pieces)
    target_channels = max(int(w.shape[1]) for w in pieces)

    dtype = pieces[0].dtype
    for piece in pieces[1:]:
        dtype = torch.promote_types(dtype, piece.dtype)

    # Final ComfyUI AUDIO should not retain an AudioVAE CUDA allocation merely to
    # concatenate PCM.  CPU transfer changes placement, not sample values.
    target_device = torch.device("cpu")
    normalized: list[torch.Tensor] = []
    mono_upmixes = 0
    for piece in pieces:
        current = piece.detach().to(device=target_device, dtype=dtype)
        channels = int(current.shape[1])
        if channels == target_channels:
            normalized.append(current.contiguous())
            continue
        if channels == 1 and target_channels == 2:
            # expand().contiguous() duplicates samples exactly; no averaging/resampling.
            current = current.expand(1, 2, -1).contiguous()
            mono_upmixes += 1
            normalized.append(current)
            continue
        raise RuntimeError(
            "Unsupported Director audio channel normalization: "
            f"{channels} -> {target_channels}."
        )

    output = torch.cat(normalized, dim=-1).contiguous()
    return output, AudioConcatReport(
        input_shapes=input_shapes,
        target_channels=target_channels,
        mono_upmixes=mono_upmixes,
        output_shape=tuple(int(x) for x in output.shape),
        output_dtype=str(output.dtype),
        output_device=str(output.device),
    )
