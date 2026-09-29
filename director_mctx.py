"""MCTX v1 sidecar helpers used by the LongMedia Director TAKE store."""
from __future__ import annotations

import json
import math
import os
import uuid
from typing import Any

import torch

try:
    from safetensors.torch import load_file as _safe_load_file
    from safetensors.torch import save_file as _safe_save_file
except Exception:  # pragma: no cover - ComfyUI ships safetensors
    _safe_load_file = None
    _safe_save_file = None


FORMAT = "mctx_v1"
SIDECAR_NAME = "motion.mctx.safetensors"


def sidecar_header(
    video: torch.Tensor,
    audio: torch.Tensor,
    *,
    project_id: str,
    clip_id: str,
    revision: str,
    metadata: dict[str, Any],
) -> dict[str, str]:
    """Return the open MCTX v1 string-only header for native H3 AV latents.

    Parent identity is retained in ``user_meta``. The Director does not currently
    expose the exact parent pin window and join coordinate at this storage boundary,
    so this writer intentionally leaves MCTX lineage pins empty instead of emitting
    guessed coordinates that another timeline could mistake for exact latent joins.
    """
    if video.ndim != 5:
        raise ValueError(f"MCTX video latent must be [B,C,T,H,W], got {tuple(video.shape)}")
    if audio.ndim != 4:
        raise ValueError(f"MCTX audio latent must be [B,C,2,T], got {tuple(audio.shape)}")

    latent_t = max(0, int(video.shape[2]))
    aligned_frames = max(5, 17 * max(0, (latent_t - 2) // 5) + 5)
    try:
        raw_frames = max(1, int(metadata.get("segment_length_frames") or aligned_frames))
    except (TypeError, ValueError, OverflowError):
        raw_frames = aligned_frames
    try:
        delivered_frames = max(1, int(metadata.get("delivered_frames") or raw_frames))
    except (TypeError, ValueError, OverflowError):
        delivered_frames = raw_frames
    try:
        fps = float(metadata.get("fps") or 24.0)
    except (TypeError, ValueError, OverflowError):
        fps = 24.0
    if not math.isfinite(fps) or fps <= 0:
        fps = 24.0

    user_meta = {
        key: metadata.get(key)
        for key in (
            "effective_seed", "prompt_summary", "parent_clip_id", "parent_revision",
            "continuation_parent_kind", "overlap_frames", "geometry_fingerprint",
            "semantic_fingerprint", "runtime_workflow",
        )
        if metadata.get(key) not in (None, "")
    }
    user_meta.update({"project_id": project_id, "clip_id": clip_id, "revision": revision})
    encoded_user_meta = json.dumps(user_meta, ensure_ascii=False, separators=(",", ":"), default=str)

    return {
        "format": FORMAT,
        "self_id": revision,
        "parent_id": "",
        "relation": "",
        "parent_join_frame": "0",
        "width": str(max(1, int(video.shape[-1]) * 16)),
        "height": str(max(1, int(video.shape[-2]) * 16)),
        "fps": f"{fps:.8g}",
        "raw_frames": str(raw_frames),
        "pinned_head_frames": "0",
        "pinned_tail_frames": "0",
        "delivered_frames": str(delivered_frames),
        "pins": "[]",
        "parent_grade": "",
        "user_meta": encoded_user_meta,
    }


def write_sidecar(
    path: str,
    *,
    video: torch.Tensor,
    audio: torch.Tensor,
    display_video: torch.Tensor | None,
    display_audio: torch.Tensor | None,
    header: dict[str, str],
) -> None:
    """Write one MCTX sidecar atomically, keeping display tensors only if needed."""
    if _safe_save_file is None:
        raise RuntimeError("safetensors is unavailable; Director MCTX TAKE storage cannot save sidecars.")
    if header.get("format") != FORMAT:
        raise ValueError(f"MCTX sidecar header must use {FORMAT}.")
    if (display_video is None) != (display_audio is None):
        raise ValueError("Display video and audio latents must be supplied together.")

    tensors: dict[str, torch.Tensor] = {
        "video": video.detach().to(device="cpu").contiguous(),
        "audio": audio.detach().to(device="cpu").contiguous(),
    }
    if display_video is not None and display_audio is not None:
        tensors["display_video"] = display_video.detach().to(device="cpu").contiguous()
        tensors["display_audio"] = display_audio.detach().to(device="cpu").contiguous()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.write.{uuid.uuid4().hex}.tmp"
    try:
        _safe_save_file(tensors, temporary, metadata=dict(header))
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def load_sidecar(path: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Read standard latents and display variants from a Director MCTX sidecar."""
    if _safe_load_file is None:
        raise RuntimeError("safetensors is unavailable; Director MCTX TAKE storage cannot load sidecars.")
    tensors = _safe_load_file(path, device="cpu")
    video = tensors.get("video")
    audio = tensors.get("audio")
    if not torch.is_tensor(video) or not torch.is_tensor(audio):
        raise ValueError(f"MCTX sidecar is missing standard video/audio tensors: {path}")
    if ("display_video" in tensors) != ("display_audio" in tensors):
        raise ValueError(f"MCTX display video/audio tensors must be stored as a pair: {path}")
    if "display_video" not in tensors:
        tensors["display_video"] = video
    if "display_audio" not in tensors:
        tensors["display_audio"] = audio
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as handle:
            header = dict(handle.metadata() or {})
    except Exception as exc:
        raise ValueError(f"Unable to read MCTX sidecar metadata: {path}") from exc
    if header.get("format") != FORMAT:
        raise ValueError(f"Unsupported MCTX format in {path}: {header.get('format')!r}")
    return tensors, header
