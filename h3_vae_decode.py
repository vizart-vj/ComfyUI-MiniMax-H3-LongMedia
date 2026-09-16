"""MiniMax H3 VideoVAE decode acceleration helpers.

The upstream H3 VAE owns semantic 256px spatial tiling internally and decodes
those tiles serially.  This module preserves the exact tile geometry, overlap,
blend order and temporal chunking while batching independent tiles along the
existing batch dimension to reduce Python/kernel-launch overhead.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Any, Iterator

import torch

_GIB = 1024 ** 3


def is_minimax_h3_video_vae(vae_wrapper: Any) -> bool:
    stage = getattr(vae_wrapper, "first_stage_model", None)
    if stage is None:
        return False
    return (
        stage.__class__.__name__ == "MiniMaxH3VideoVAE"
        and callable(getattr(stage, "split_tiles", None))
        and callable(getattr(stage, "_decode_pixels", None))
        and callable(getattr(stage, "blend", None))
        and int(getattr(stage, "vae_ratio", 0) or 0) > 0
    )


def choose_h3_tile_batch(requested: int, free_vram_bytes: int | None) -> int:
    """Choose tile parallelism.

    ``batch_size=1`` was historically unused by LongMedia Decode, so it now means
    Auto for H3 only. Explicit values >1 remain user-owned caps.
    """
    req = max(1, int(requested))
    if req > 1:
        return min(req, 16)
    free = max(0, int(free_vram_bytes or 0))
    if free >= 20 * _GIB:
        return 8
    if free >= 12 * _GIB:
        return 4
    if free >= 8 * _GIB:
        return 2
    return 1


def fallback_tile_batches(start: int) -> tuple[int, ...]:
    start = max(1, min(16, int(start)))
    out: list[int] = []
    value = start
    while value > 1:
        if value not in out:
            out.append(value)
        value = max(1, value // 2)
    if 1 not in out:
        out.append(1)
    return tuple(out)


def _batched_tiled_decode(stage: Any, z: torch.Tensor, *, tile_batch: int, stats: dict[str, int]) -> torch.Tensor:
    height = int(z.shape[-2]) * int(stage.vae_ratio)
    width = int(z.shape[-1]) * int(stage.vae_ratio)
    y_idx, y_len, y_overlap = stage.split_tiles(height)
    x_idx, x_len, x_overlap = stage.split_tiles(width)

    stats["rows"] = int(len(y_idx))
    stats["columns"] = int(len(x_idx))
    stats["temporal_calls"] = int(stats.get("temporal_calls", 0)) + 1
    stats["tiles"] = int(len(y_idx) * len(x_idx))
    stats["tiles_total"] = int(stats.get("tiles_total", 0)) + int(len(y_idx) * len(x_idx))
    call_start = int(stats.get("decode_calls_total", 0))

    canvas = None
    row_tails: list[torch.Tensor] = []
    out_y = 0
    source_batch = int(z.shape[0])
    batch_limit = max(1, int(tile_batch))

    for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
        zi, zl = int(i_pos) // int(stage.vae_ratio), int(i_len) // int(stage.vae_ratio)
        new_tails: list[torch.Tensor] = []
        left_tail = None
        out_x = 0

        j0 = 0
        while j0 < len(x_idx):
            group_specs = list(zip(x_idx[j0:j0 + batch_limit], x_len[j0:j0 + batch_limit]))
            latent_tiles = [
                z[..., zi:zi + zl, int(j_pos) // int(stage.vae_ratio):int(j_pos) // int(stage.vae_ratio) + int(j_len) // int(stage.vae_ratio)]
                for j_pos, j_len in group_specs
            ]
            shapes = {tuple(tile.shape) for tile in latent_tiles}
            if len(shapes) == 1 and len(latent_tiles) > 1:
                packed = torch.cat(latent_tiles, dim=0).contiguous()
                decoded_batch = stage._decode_pixels(packed)
                decoded_tiles = list(decoded_batch.split(source_batch, dim=0))
                stats["decode_calls_total"] = int(stats.get("decode_calls_total", 0)) + 1
                stats["max_tiles_per_call"] = max(stats["max_tiles_per_call"], len(decoded_tiles))
            else:
                packed = None
                decoded_batch = None
                decoded_tiles = []
                for latent_tile in latent_tiles:
                    decoded_tiles.append(stage._decode_pixels(latent_tile))
                    stats["decode_calls_total"] = int(stats.get("decode_calls_total", 0)) + 1
                    stats["max_tiles_per_call"] = max(stats["max_tiles_per_call"], 1)

            for offset, tile in enumerate(decoded_tiles):
                j = j0 + offset
                if i < len(y_idx) - 1:
                    new_tails.append(tile[..., -int(y_overlap[i]):, :].clone())
                next_left_tail = (
                    tile[..., :, -int(x_overlap[j]):].clone()
                    if j < len(x_idx) - 1 else None
                )
                if i > 0:
                    tile = stage.blend(row_tails[j], tile, int(y_overlap[i - 1]), dim=-2)
                if j > 0:
                    tile = stage.blend(left_tail, tile, int(x_overlap[j - 1]), dim=-1)
                left_tail = next_left_tail
                if i < len(y_idx) - 1:
                    tile = tile[..., :-int(y_overlap[i]), :]
                if j < len(x_idx) - 1:
                    tile = tile[..., :, :-int(x_overlap[j])]
                if canvas is None:
                    canvas = torch.empty(
                        *tile.shape[:-2], height, width,
                        dtype=tile.dtype, device=tile.device,
                    )
                canvas[..., out_y:out_y + int(tile.shape[-2]), out_x:out_x + int(tile.shape[-1])].copy_(tile)
                out_x += int(tile.shape[-1])

            del decoded_tiles
            if decoded_batch is not None:
                del decoded_batch
            if packed is not None:
                del packed
            del latent_tiles
            j0 += len(group_specs)

        row_tails = new_tails
        if canvas is None:
            raise RuntimeError("MiniMax H3 batched tiled decode produced no tiles.")
        # Same row height as upstream after overlap trimming.
        row_height = int(canvas.shape[-2]) if len(y_idx) == 1 else (
            int(y_len[i]) - (int(y_overlap[i]) if i < len(y_idx) - 1 else 0)
        )
        out_y += row_height

    if canvas is None:
        raise RuntimeError("MiniMax H3 batched tiled decode produced no output canvas.")
    stats["decode_calls"] = int(stats.get("decode_calls_total", 0)) - call_start
    return canvas


@contextmanager
def batched_h3_tiles(vae_wrapper: Any, tile_batch: int) -> Iterator[dict[str, int]]:
    """Temporarily batch the upstream H3 VAE's native spatial tiles."""
    stage = getattr(vae_wrapper, "first_stage_model", None)
    stats: dict[str, int] = {
        "tile_batch": max(1, int(tile_batch)),
        "rows": 0,
        "columns": 0,
        "tiles": 0,
        "tiles_total": 0,
        "decode_calls": 0,
        "decode_calls_total": 0,
        "temporal_calls": 0,
        "max_tiles_per_call": 0,
    }
    if stage is None or not is_minimax_h3_video_vae(vae_wrapper) or int(tile_batch) <= 1:
        yield stats
        return

    had_instance_attr = "tiled_decode" in getattr(stage, "__dict__", {})
    previous_instance_value = stage.__dict__.get("tiled_decode") if had_instance_attr else None

    def patched(self: Any, z: torch.Tensor) -> torch.Tensor:
        return _batched_tiled_decode(self, z, tile_batch=int(tile_batch), stats=stats)

    stage.tiled_decode = MethodType(patched, stage)
    try:
        yield stats
    finally:
        if had_instance_attr:
            stage.tiled_decode = previous_instance_value
        else:
            try:
                delattr(stage, "tiled_decode")
            except AttributeError:
                pass
