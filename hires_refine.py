"""Bounded temporal refine planning for MiniMax H3 latent hi-res.

This module is deliberately ComfyUI-free.  LongMedia's main MultiClip planner is
not involved: these windows exist only after a clean x0 has already been
produced (and optionally learned-upscaled) and bound the second/refine sampling
stage to legal H3 temporal geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch

FRAME_PATTERN: tuple[int, ...] = (1, 4, 4, 4, 4)
FPS = 24
AUDIO_LATENT_FPS = 40


@dataclass(frozen=True, slots=True)
class HiresRefineWindow:
    index: int
    token_start: int
    token_stop: int
    frame_start: int
    frame_stop: int
    overlap_tokens: int
    exact_prefix_tokens: int

    @property
    def token_count(self) -> int:
        return int(self.token_stop - self.token_start)

    @property
    def frame_count(self) -> int:
        return int(self.frame_stop - self.frame_start)

    @property
    def blend_tokens(self) -> int:
        return max(0, int(self.overlap_tokens - self.exact_prefix_tokens))


def frames_for_tokens(tokens: int) -> int:
    """Decode an H3-local temporal token count to visible frames."""
    tokens = int(tokens)
    if tokens < 0:
        raise ValueError("tokens must be non-negative")
    cycles, rem = divmod(tokens, len(FRAME_PATTERN))
    return cycles * sum(FRAME_PATTERN) + sum(FRAME_PATTERN[:rem])


def tokens_covering_frames(frames: int) -> int:
    """Smallest local temporal-token count whose decoded span covers frames."""
    target = max(0, int(frames))
    tokens = 0
    covered = 0
    while covered < target:
        covered += FRAME_PATTERN[tokens % len(FRAME_PATTERN)]
        tokens += 1
    return tokens


def validate_h3_latent_t(latent_t: int) -> int:
    latent_t = int(latent_t)
    if latent_t < 2 or (latent_t - 2) % 5:
        raise ValueError(f"MiniMax H3 latent time must be 5*k+2, got {latent_t}.")
    return latent_t


def legal_window_tokens_for_frames(frames: int) -> int:
    """Snap a requested window to the nearest legal 5*k+2 token geometry."""
    frames = max(5, int(frames))
    k = max(0, int(round((frames - 5) / 17.0)))
    return 5 * k + 2


def default_chunk_tokens(total_vram_bytes: int | None) -> int:
    """Conservative refine window size; 73f/22 tokens is the 16GB baseline."""
    gb = (float(total_vram_bytes) / float(1024**3)) if total_vram_bytes else 0.0
    if gb and gb <= 18.5:
        return 22   # 73 frames
    if gb and gb <= 24.5:
        return 32   # 107 frames
    if gb and gb <= 32.5:
        return 42   # 141 frames
    return 52       # 175 frames


def fallback_chunk_tokens(start_tokens: int) -> tuple[int, ...]:
    """Legal progressively smaller windows for OOM recovery.

    All values are 5*k+2.  With the default seven-token overlap every hop is a
    multiple of five, so every local window starts on the same H3 (1,4,4,4,4)
    phase as the global source timeline.
    """
    start = max(7, int(start_tokens))
    candidates = [v for v in (52, 42, 32, 22, 17, 12, 7) if v <= start]
    if start not in candidates and start >= 7 and (start - 2) % 5 == 0:
        candidates.insert(0, start)
    out: list[int] = []
    for value in candidates:
        if value not in out:
            out.append(value)
    return tuple(out)


def plan_refine_windows(
    latent_t: int,
    *,
    chunk_tokens: int,
    overlap_tokens: int = 7,
    exact_prefix_tokens: int = 2,
) -> tuple[HiresRefineWindow, ...]:
    """Plan phase-safe H3 refine windows over one already-generated latent.

    This is intentionally *not* LongMedia segmentation.  Window starts are kept
    at multiples of five temporal tokens.  The last window is allowed to be
    shorter instead of shifting backwards and changing the native temporal
    phase.
    """
    latent_t = validate_h3_latent_t(latent_t)
    chunk_tokens = int(chunk_tokens)
    overlap_tokens = int(overlap_tokens)
    exact_prefix_tokens = int(exact_prefix_tokens)
    if chunk_tokens < 2 or (chunk_tokens - 2) % 5:
        raise ValueError(f"refine chunk_tokens must be 5*k+2, got {chunk_tokens}")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be non-negative")
    if overlap_tokens and (overlap_tokens - 2) % 5:
        raise ValueError(f"refine overlap_tokens must be 0 or 5*k+2, got {overlap_tokens}")
    if chunk_tokens >= latent_t:
        return (HiresRefineWindow(
            index=0,
            token_start=0,
            token_stop=latent_t,
            frame_start=0,
            frame_stop=frames_for_tokens(latent_t),
            overlap_tokens=0,
            exact_prefix_tokens=0,
        ),)
    if overlap_tokens >= chunk_tokens:
        raise ValueError("refine overlap must be smaller than chunk")
    hop = chunk_tokens - overlap_tokens
    if hop <= 0 or hop % 5:
        raise ValueError(
            f"refine hop must preserve H3 five-token phase; chunk={chunk_tokens}, overlap={overlap_tokens}"
        )

    windows: list[HiresRefineWindow] = []
    start = 0
    index = 0
    previous_stop = 0
    while start < latent_t:
        stop = min(latent_t, start + chunk_tokens)
        count = stop - start
        # Because source T is 5*k+2 and every start is a multiple of five, the
        # final short window is also legal 5*k+2.
        validate_h3_latent_t(count)
        overlap = max(0, previous_stop - start) if windows else 0
        frame_start = frames_for_tokens(start)
        frame_stop = frame_start + frames_for_tokens(count)
        exact = min(max(0, exact_prefix_tokens), overlap)
        windows.append(HiresRefineWindow(
            index=index,
            token_start=start,
            token_stop=stop,
            frame_start=frame_start,
            frame_stop=frame_stop,
            overlap_tokens=overlap,
            exact_prefix_tokens=exact,
        ))
        if stop >= latent_t:
            break
        previous_stop = stop
        start += hop
        index += 1
    return tuple(windows)


def denoise_profile(window: HiresRefineWindow, *, dtype=torch.float32) -> torch.Tensor:
    """Per-token refine mask: inherited head, then a long C1 bridge into the new tail."""
    weights = torch.ones((window.token_count,), dtype=dtype)
    if window.overlap_tokens <= 0:
        return weights
    exact = min(window.exact_prefix_tokens, window.token_count)
    if exact:
        weights[:exact] = 0
    blend = min(window.blend_tokens, window.token_count - exact)
    if blend:
        # Endpoint-inclusive smoothstep is deliberate.  v0.6.24/25 sampled only
        # the interior of (0, 1), so the final overlap token was ~0.93 and the
        # very next token jumped to 1.0.  On a 22-frame H3 overlap that produced
        # a perfectly periodic visible kick at every window stop (73, 124, ...).
        # Include both endpoints inside the bridge: the first bridge token stays
        # on the inherited trajectory and the last reaches the new trajectory
        # exactly, giving a flat 0/1 plateau on each side of the handoff.
        if blend == 1:
            ramp = torch.ones((1,), dtype=torch.float32)
        else:
            u = torch.linspace(0.0, 1.0, blend, dtype=torch.float32)
            ramp = u * u * (3.0 - 2.0 * u)
        weights[exact:exact + blend] = ramp.to(dtype=dtype)
    return weights


def merge_profile(window: HiresRefineWindow, *, dtype=torch.float32) -> torch.Tensor:
    """Weights for writing a newly refined window into the accumulated x0."""
    weights = torch.ones((window.token_count,), dtype=dtype)
    if window.overlap_tokens <= 0:
        return weights
    exact = min(window.exact_prefix_tokens, window.token_count)
    if exact:
        weights[:exact] = 0
    blend = min(window.blend_tokens, window.token_count - exact)
    if blend:
        if blend == 1:
            ramp = torch.ones((1,), dtype=torch.float32)
        else:
            u = torch.linspace(0.0, 1.0, blend, dtype=torch.float32)
            ramp = u * u * (3.0 - 2.0 * u)
        weights[exact:exact + blend] = ramp.to(dtype=dtype)
    return weights


def audio_window_bounds(frame_start: int, frame_count: int, full_audio_t: int) -> tuple[int, int, int]:
    """Return source start/end and required local audio length for an H3 window."""
    frame_start = max(0, int(frame_start))
    frame_count = max(1, int(frame_count))
    full_audio_t = max(0, int(full_audio_t))
    start = int(round(frame_start / FPS * AUDIO_LATENT_FPS))
    required = int(round(frame_count / FPS * AUDIO_LATENT_FPS))
    end = min(full_audio_t, start + required)
    return start, end, required


def describe_windows(windows: Iterable[HiresRefineWindow]) -> list[dict[str, int]]:
    return [
        {
            "index": int(w.index),
            "token_start": int(w.token_start),
            "token_stop": int(w.token_stop),
            "frame_start": int(w.frame_start),
            "frame_stop": int(w.frame_stop),
            "overlap_tokens": int(w.overlap_tokens),
            "exact_prefix_tokens": int(w.exact_prefix_tokens),
        }
        for w in windows
    ]
