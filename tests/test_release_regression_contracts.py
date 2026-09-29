"""Release-gate regressions for v0.6.51-v0.6.55 contracts."""

from __future__ import annotations

import ast
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
pkg = types.ModuleType("ComfyUI-MiniMax-H3-LongMedia")
pkg.__path__ = [str(ROOT)]
pkg.__package__ = "ComfyUI-MiniMax-H3-LongMedia"
for module_name in (
    "ComfyUI-MiniMax-H3-LongMedia",
    "ComfyUI-MiniMax-H3-LongMedia.__init__",
    "__init__",
):
    sys.modules.setdefault(module_name, pkg)

import torch

from audio_assembly import concatenate_director_audio


NODES = (ROOT / "nodes.py").read_text(encoding="utf-8")
CACHE = (ROOT / "director_cache.py").read_text(encoding="utf-8")
DIRECTOR = (ROOT / "director_plan.py").read_text(encoding="utf-8")


def _function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(item for item in ast.walk(tree) if isinstance(item, ast.FunctionDef) and item.name == name)
    return ast.get_source_segment(source, node) or ""


def test_generated_and_media_base_types_remain_distinct():
    assert '"base_kind"' in DIRECTOR
    assert '== "media"' in DIRECTOR
    assert "director_base_kinds" in NODES
    assert "generated" in NODES and "media" in NODES


def test_media_to_generated_uses_tail_only_native_keyframe_contract():
    body = _function_source(NODES, "_lm_external_continuation_context")
    assert "source_tail" in body
    assert "never VAE-encode the full clip" in body
    assert "native_minimax_keyframe" in body
    assert "vae_encoded_frames" in body


def test_generated_continuation_uses_cached_native_take_streams():
    assert "_lm_cached_take_to_latent" in NODES
    body = _function_source(NODES, "_lm_cached_take_to_latent")
    assert "prefix = 'display' if which == 'display' else 'continuation'" in body
    assert "take[f'{prefix}_video']" in body
    assert "take[f'{prefix}_audio']" in body
    assert "NestedTensor" in body
    assert '"continuation_video"' in CACHE
    assert '"continuation_audio"' in CACHE


def test_trim_aware_media_tail_contract_is_preserved():
    assert "director_source_in_seconds" in NODES
    assert "director_source_out_seconds" in NODES
    assert "external_tail_vae_encoded_frames" in NODES


def test_mixed_mono_stereo_assembly_duplicates_mono_exactly():
    mono = torch.tensor([[[1.0, -2.0, 3.5]]])
    stereo = torch.tensor([[[10.0, 11.0], [20.0, 21.0]]])
    output, report = concatenate_director_audio((mono, stereo))
    assert tuple(output.shape) == (1, 2, 5)
    torch.testing.assert_close(output[0, 0, :3], mono[0, 0])
    torch.testing.assert_close(output[0, 1, :3], mono[0, 0])
    torch.testing.assert_close(output[..., 3:], stereo)
    assert report.mono_upmixes == 1


def test_refiner_restores_exact_stage1_audio_and_ignores_refined_audio():
    assert "audio_passthrough=stage1_exact" in NODES
    assert "Intentionally ignore refined_audio" in NODES
    assert "Stage-1 audio owns" in NODES


def test_streamed_final_layer_keeps_current_pdd_signature():
    body = _function_source(NODES, "_streamed_forward")
    assert "sigma=None" in body
    assert "sample_sigmas=None" in body
    assert "shifts=None" in body
    assert "PDD" in body


def test_take_snapshot_storage_remains_whole_document_based():
    assert '"director_timeline_snapshot"' in CACHE
    assert "refmods" in DIRECTOR
