"""Unified immutable continuation contract for Director GENERATED and MEDIA parents.

The sampler consumes only :class:`ContinuationContext`; parent provenance is an
editor/runtime concern.  External MEDIA is reduced to a tail window before VAE
encoding.  Full source media is never routed through diffusion merely to append.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class ContinuationContext:
    video_latent_tail: torch.Tensor
    audio_latent_tail: torch.Tensor
    motion_context: Any
    source_fps: float
    source_resolution: tuple[int, int]
    boundary_frame: int
    overlap_frames: int
    source_revision: str | None
    source_fingerprint: str
    source_kind: str
    timing: dict[str, Any]

    def __post_init__(self) -> None:
        if self.source_kind not in {"generated_take", "external_video"}:
            raise ValueError(f"Unsupported continuation source_kind={self.source_kind!r}")
        if self.overlap_frames <= 0:
            raise ValueError("Continuation overlap_frames must be positive.")
        if self.video_latent_tail.ndim != 5:
            raise ValueError(f"video_latent_tail must be 5D, got {tuple(self.video_latent_tail.shape)}")
        if self.audio_latent_tail.ndim != 4:
            raise ValueError(f"audio_latent_tail must be 4D, got {tuple(self.audio_latent_tail.shape)}")



@dataclass(frozen=True, slots=True)
class DirectorRenderUnit:
    """One GENERATED editorial block that may enter H3 sampling.

    MEDIA blocks deliberately never become render units; they can only appear as
    continuation parents for the immediately following GENERATED unit.
    """

    editorial_index: int
    parent_index: int | None
    parent_kind: str | None


def build_director_render_units(base_kinds: Any) -> tuple[DirectorRenderUnit, ...]:
    kinds = tuple(str(v or "generated").strip().lower() for v in (base_kinds or ()))
    units: list[DirectorRenderUnit] = []
    for index, kind in enumerate(kinds):
        if kind == "media":
            continue
        if kind != "generated":
            raise ValueError(f"Unsupported Director BASE kind at {index}: {kind!r}")
        parent_index = index - 1 if index > 0 else None
        if parent_index is None:
            parent_kind = None
        else:
            parent_kind = "external_video" if kinds[parent_index] == "media" else "generated_take"
        units.append(DirectorRenderUnit(
            editorial_index=index, parent_index=parent_index, parent_kind=parent_kind,
        ))
    return tuple(units)

def canonical_fingerprint(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ExternalContinuationCache:
    """Small process-local cache for VAE-encoded external tails.

    Keys include source identity, source range, geometry and overlap.  Tensors are
    retained on CPU so repeated append/regeneration does not keep GPU residency.
    """

    def __init__(self) -> None:
        self._items: dict[str, ContinuationContext] = {}

    def get(self, key: str) -> ContinuationContext | None:
        return self._items.get(str(key))

    def put(self, key: str, value: ContinuationContext) -> None:
        self._items[str(key)] = ContinuationContext(
            video_latent_tail=value.video_latent_tail.detach().to("cpu").contiguous(),
            audio_latent_tail=value.audio_latent_tail.detach().to("cpu").contiguous(),
            motion_context=value.motion_context,
            source_fps=float(value.source_fps),
            source_resolution=tuple(value.source_resolution),
            boundary_frame=int(value.boundary_frame),
            overlap_frames=int(value.overlap_frames),
            source_revision=value.source_revision,
            source_fingerprint=str(value.source_fingerprint),
            source_kind=str(value.source_kind),
            timing=dict(value.timing),
        )


EXTERNAL_CONTINUATION_CACHE = ExternalContinuationCache()
