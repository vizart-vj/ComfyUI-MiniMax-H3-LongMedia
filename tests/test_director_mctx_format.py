from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from director_cache import DirectorClipCache
from director_mctx import FORMAT


def test_new_director_take_round_trips_as_mctx(tmp_path):
    cache = DirectorClipCache(str(tmp_path), "mctx-project")
    video = torch.arange(1 * 24 * 2 * 2 * 3, dtype=torch.float32).reshape(1, 24, 2, 2, 3)
    audio = torch.arange(1 * 32 * 2 * 2, dtype=torch.float32).reshape(1, 32, 2, 2)

    record = cache.save_take(
        clip_id="shot-1",
        clip_index=0,
        display_video=video,
        display_audio=audio,
        continuation_video=video,
        continuation_audio=audio,
        metadata={"fps": 24, "segment_length_frames": 5, "delivered_frames": 5},
    )

    assert record["storage_layout"] == "mctx_v1"
    tensor_path, metadata_path = cache._take_paths("shot-1", record["revision"])
    assert tensor_path.endswith("motion.mctx.safetensors")
    assert json.loads(Path(metadata_path).read_text(encoding="utf-8"))["storage_layout"] == "mctx_v1"
    loaded = cache.load_revision("shot-1", record["revision"])
    assert loaded is not None
    assert torch.equal(loaded["display_video"], video)
    assert torch.equal(loaded["display_audio"], audio)
    assert torch.equal(loaded["continuation_video"], video)
    assert torch.equal(loaded["continuation_audio"], audio)


def test_mctx_header_uses_standard_av_keys_and_records_identity(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    safetensors = pytest.importorskip("safetensors")
    from director_mctx import load_sidecar, sidecar_header, write_sidecar

    video = torch.zeros((1, 24, 2, 2, 3), dtype=torch.float16)
    audio = torch.zeros((1, 32, 2, 2), dtype=torch.float32)
    header = sidecar_header(
        video,
        audio,
        project_id="project-a",
        clip_id="clip-a",
        revision="rev-a",
        metadata={"fps": 24, "segment_length_frames": 5, "delivered_frames": 5},
    )
    path = str(tmp_path / "motion.mctx.safetensors")
    write_sidecar(path, video=video, audio=audio, display_video=None, display_audio=None, header=header)

    tensors, loaded_header = load_sidecar(path)
    assert loaded_header["format"] == FORMAT
    assert "video" in tensors and "audio" in tensors
    assert torch.equal(tensors["display_video"], video)
    assert torch.equal(tensors["display_audio"], audio)
    assert safetensors_torch.load_file(path, device="cpu").keys() == {"video", "audio"}
    assert safetensors.safe_open is not None
