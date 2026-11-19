"""Regression tests for LongMedia integration with ComfyUI's H3 runtime."""
from __future__ import annotations

import ast
import sys
import unittest
from collections.abc import Mapping
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import director_plan  # noqa: E402
from h3_vae_decode import batched_h3_tiles  # noqa: E402


@lru_cache(maxsize=1)
def _nodes_tree() -> ast.Module:
    return ast.parse((ROOT / "nodes.py").read_text(encoding="utf-8"))


def _load_node_functions(scope: dict[str, Any], *names: str) -> None:
    """Exercise the production functions without initializing the ComfyUI server."""
    functions = [
        node for node in _nodes_tree().body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {function.name for function in functions} != set(names):
        raise AssertionError(f"Missing production functions: {names}")
    source = ast.Module(body=functions, type_ignores=[])
    exec(compile(source, str(ROOT / "nodes.py"), "exec"), scope)


class MiniMaxH3VideoVAE:
    """Small stand-in for the upstream VAE's frames-aware tile API."""

    vae_ratio = 1

    def split_tiles(self, length: int) -> tuple[list[int], list[int], list[int]]:
        return [0], [length], []

    def _decode_pixels(self, latent: torch.Tensor) -> torch.Tensor:
        return latent + 10

    def blend(
        self, previous: torch.Tensor, current: torch.Tensor, overlap: int, dim: int
    ) -> torch.Tensor:
        raise AssertionError("This stand-in has one tile")

    def tiled_decode(self, latent: torch.Tensor, frames: slice = slice(None)) -> torch.Tensor:
        return self._decode_pixels(latent)[:, :, frames]


class NativeH3VaeTilingTests(unittest.TestCase):
    def test_new_upstream_tiling_preserves_the_requested_temporal_slice(self) -> None:
        stage = MiniMaxH3VideoVAE()
        vae = SimpleNamespace(first_stage_model=stage)
        latent = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 2, 2)
        frames = slice(1, 3)
        expected = stage.tiled_decode(latent, frames)

        with batched_h3_tiles(vae, tile_batch=4):
            actual = stage.tiled_decode(latent, frames)

        torch.testing.assert_close(actual, expected)
        self.assertNotIn("tiled_decode", vars(stage))

    def test_old_upstream_tiling_still_uses_the_legacy_adapter(self) -> None:
        stage = MiniMaxH3VideoVAE()
        vae = SimpleNamespace(first_stage_model=stage)
        latent = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 2, 2)

        def legacy_tiled_decode(z: torch.Tensor) -> torch.Tensor:
            return stage._decode_pixels(z)

        stage.tiled_decode = legacy_tiled_decode
        with batched_h3_tiles(vae, tile_batch=4) as stats:
            actual = stage.tiled_decode(latent)

        torch.testing.assert_close(actual, latent + 10)
        self.assertEqual(stats["decode_calls_total"], 1)
        self.assertIs(stage.tiled_decode, legacy_tiled_decode)


class _SharedHead:
    policy = "stock"

    def forward(self, value: int) -> tuple[str, int]:
        return self.policy, value


def _streamed_forward(layer: _SharedHead, value: int) -> tuple[str, int]:
    return layer.policy, value * 2


class SharedHeadLifecycleTests(unittest.TestCase):
    def test_streaming_override_ends_before_a_stock_forward(self) -> None:
        from sampler_lifecycle import temporary_method_override

        layer = _SharedHead()

        with temporary_method_override(
            layer, "forward", _streamed_forward, attributes={"policy": "streaming"},
        ):
            self.assertEqual(layer.forward(7), ("streaming", 14))

        self.assertEqual(layer.forward(7), ("stock", 7))
        self.assertNotIn("forward", vars(layer))
        self.assertNotIn("policy", vars(layer))

    def test_exception_restores_preexisting_instance_override_and_attributes(self) -> None:
        from sampler_lifecycle import temporary_method_override

        layer = _SharedHead()
        previous_forward = lambda value: ("other adapter", value + 1)
        layer.forward = previous_forward
        layer.policy = "third party"

        with self.assertRaisesRegex(RuntimeError, "sampler interrupted"):
            with temporary_method_override(
                layer, "forward", _streamed_forward,
                attributes={"policy": "streaming", "workspace": object()},
            ):
                self.assertEqual(layer.forward(7), ("streaming", 14))
                raise RuntimeError("sampler interrupted")

        self.assertIs(layer.forward, previous_forward)
        self.assertEqual(layer.policy, "third party")
        self.assertNotIn("workspace", vars(layer))


class SamplerCleanupTests(unittest.TestCase):
    def test_later_cleanup_runs_and_a_render_error_survives(self) -> None:
        from sampler_lifecycle import SamplerCleanup

        cleanup_error = RuntimeError("hook restoration failed")
        released = []

        def fail() -> None:
            raise cleanup_error

        with self.assertRaisesRegex(ValueError, "render failed"):
            try:
                raise ValueError("render failed")
            finally:
                failure = sys.exc_info()[1]
                cleanup = SamplerCleanup()
                with self.assertLogs("sampler_lifecycle", level="ERROR"):
                    cleanup.attempt("hooks", fail)
                cleanup.attempt("guider", lambda: released.append("guider"))
                cleanup.finish(failure)

        self.assertEqual(released, ["guider"])

    def test_cleanup_error_reaches_caller_after_successful_render(self) -> None:
        from sampler_lifecycle import SamplerCleanup

        def fail() -> None:
            raise RuntimeError("pool shutdown failed")

        cleanup = SamplerCleanup()
        with self.assertLogs("sampler_lifecycle", level="ERROR"):
            cleanup.attempt("pool", fail)
        with self.assertRaisesRegex(RuntimeError, "pool shutdown failed"):
            cleanup.finish(None)


class _GridFinalHead:
    """CPU reference for the installed host's per-stream grid contract."""

    def __init__(self, heads: int = 1) -> None:
        self.pdd_steps = heads
        self.video_dim = 3
        self.audio_dim = 2
        self.video_out = self._linear(self.video_dim, heads)
        self.audio_out = self._linear(self.audio_dim, heads)

    @staticmethod
    def _linear(width: int, heads: int) -> torch.nn.Linear:
        layer = torch.nn.Linear(4, width * heads)
        with torch.no_grad():
            layer.weight.copy_(torch.arange(width * heads * 4).reshape(-1, 4) / 100)
            layer.bias.copy_(torch.arange(width * heads) / 100)
        layer.out_features = width
        return layer

    @staticmethod
    def adaln_proj(embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return embedding * 0.25, embedding * 0.5

    @staticmethod
    def norm(values: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(values, (values.shape[-1],))

    def forward(self, x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, grids):
        shift, scale = self.adaln_proj(t_emb)
        steps = self.pdd_steps
        if steps > 1:
            i = int((sample_sigmas - sigma).abs().argmin())
            endpoints = (sigma, sample_sigmas[min(i + 1, len(sample_sigmas) - 1)])
            start, stop = (min(range(len(grids[0])), key=lambda k: abs(grids[0][k] - s))
                           for s in endpoints)
            start = min(start, steps - 1)
            stop = max(stop, start + 1)
        result = []
        for head, (a, b, row), grid in zip(
            (self.video_out, self.audio_out), (video_seg, audio_seg), grids,
        ):
            hidden = self.norm(x[a:b]) * (1 + scale[row]) + shift[row]
            if steps == 1:
                result.append(head(hidden))
                continue
            weights = head.weight.reshape(steps, -1, head.weight.shape[-1])
            biases = head.bias.reshape(steps, -1)
            durations = [grid[k] - grid[k + 1] for k in range(start, stop)]
            normalized = [dt / sum(durations) for dt in durations]
            weight = weights[0]
            bias = biases[0]
            for k in range(max(1, start), stop):
                weight = weight + normalized[k - start] * weights[k]
                bias = bias + normalized[k - start] * biases[k]
            result.append(torch.nn.functional.linear(hidden, weight, bias))
        return result


class FinalHeadGridContractTests(unittest.TestCase):
    @staticmethod
    def _wrapped_forward(layer, state, *args):
        from sampler_lifecycle import temporary_method_override

        extension = ModuleType("comfy.patcher_extension")
        extension.WrappersMP = SimpleNamespace(DIFFUSION_MODEL="diffusion_model")
        comfy = ModuleType("comfy")
        comfy.__path__ = []
        ops = ModuleType("comfy.ops")

        @contextmanager
        def cast_weight(head: torch.nn.Linear, *unused, **options):
            yield head.weight, head.bias

        ops.CastBiasWeightContext = cast_weight
        comfy.ops = ops
        module_stubs = {"comfy": comfy, "comfy.ops": ops, "comfy.patcher_extension": extension}
        options = {}
        scope = {"torch": torch, "temporary_method_override": temporary_method_override,
                 "_lm_print": lambda *a, **kw: None}
        _load_node_functions(scope, "_install_h3_final_output_streaming")
        model = SimpleNamespace(final_layer=layer)
        patcher = SimpleNamespace(model=SimpleNamespace(diffusion_model=model))

        class Executor:
            class_obj = model

            def __call__(self, *a):
                return model.final_layer.forward(*a)

        with patch.dict(sys.modules, module_stubs):
            installed = scope["_install_h3_final_output_streaming"](patcher, options, state, 256)
            if not installed:
                raise AssertionError(state)
            wrapped, = options["wrappers"]["diffusion_model"]["LongMediaFinalHead"]
            return wrapped(Executor(), *args)

    def test_streaming_matches_native_two_stream_sigma_grids(self) -> None:
        layer = _GridFinalHead(heads=3)
        values = torch.arange(700 * 4, dtype=torch.float32).reshape(700, 4).sin()
        embedding = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10
        args = (
            values, embedding, (0, 580, torch.arange(580) % 2), (580, 700, 0),
            torch.tensor(0.8), torch.tensor([0.8, 0.0]),
            ([1.0, 0.8, 0.5, 0.0], [1.0, 0.65, 0.25, 0.0]),
        )
        expected = layer.forward(*args)
        state = {"v24_final_video_captured": True, "v24_final_audio_captured": True}

        actual = self._wrapped_forward(layer, state, *args)

        self.assertEqual(len(actual), 2)
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-6)
        self.assertEqual(state["final_output_max_chunks"], 3)
        self.assertNotIn("forward", vars(layer))

    def test_native_video_only_contract_needs_no_audio_segment(self) -> None:
        layer = _GridFinalHead()
        values = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        embedding = torch.zeros((1, 4))
        args = (values, embedding, (0, 4, 0), None, None, None, ([1.0, 0.0],))
        expected, = layer.forward(*args)

        actual, = self._wrapped_forward(layer, {"v24_final_video_captured": True}, *args)

        torch.testing.assert_close(actual, expected)
        self.assertNotIn("forward", vars(layer))


class DirectorLipSyncSourceTests(unittest.TestCase):
    def test_repeated_shot_references_keep_speech_after_a_silent_first_clip(self) -> None:
        sample_rate = 32
        samples_per_clip = 5 * sample_rate
        waveform = torch.tensor([0.0, 0.25, -0.25]).repeat_interleave(
            samples_per_clip,
        ).reshape(1, 1, -1)
        audio = {"waveform": waveform, "sample_rate": sample_rate}
        document = {
            "version": 10,
            "kind": "h3_longmedia_director",
            "project_id": "lip-sync-silent-opening",
            "audio_mode": "lip_sync",
            "setup_timeline_mode": "multiclip",
            "subjects": [{
                "subject_id": "voice",
                "kind": "Audio",
                "slot": 1,
                "media": {"kind": "audio", "filename": "voice.wav"},
            }],
            "shots": [
                {
                    "clip_id": f"shot-{i}",
                    "duration": 5,
                    "start": 5 * i,
                    "subjects": ["voice"],
                }
                for i in range(3)
            ],
        }

        def load_audio(node_name: str, **inputs: Any) -> tuple[dict[str, Any]]:
            self.assertEqual(node_name, "LoadAudio")
            self.assertEqual(inputs["audio"], "voice.wav")
            return (audio,)

        scope = {
            "Any": Any,
            "FPS": 24,
            "Mapping": Mapping,
            "torch": torch,
            "_director_normalize_document": director_plan.normalize_document,
            "_director_picture_runtime_map": director_plan.director_picture_runtime_map,
            "_director_frame_anchor_contract": director_plan.director_frame_anchor_contract,
            "_lm_director_media_path": lambda record: record["filename"],
            "_lm_call_core_node": load_audio,
        }
        _load_node_functions(
            scope, "_lm_build_director_media_bundle", "_lm_director_usage_ranges",
            "_lm_director_unique_trim", "_lm_trim_audio_seconds",
            "_lm_director_video_reference_scales",
            "_v104_slice_lipsync_guide_audio", "_fit_passthrough_audio_to_timeline",
            "_lm_materialize_comfy_audio",
        )

        soundtrack = scope["_lm_build_director_media_bundle"](document)["audios"][0]
        later_clip, start, end = scope["_v104_slice_lipsync_guide_audio"](soundtrack, 120, 120)
        passthrough, report = scope["_fit_passthrough_audio_to_timeline"](soundtrack, 15)

        self.assertEqual((start, end), (samples_per_clip, 2 * samples_per_clip))
        self.assertEqual(later_clip["sample_rate"], sample_rate)
        torch.testing.assert_close(later_clip["waveform"], waveform[..., start:end])
        self.assertEqual(report["action"], "exact")
        torch.testing.assert_close(passthrough["waveform"], waveform)


if __name__ == "__main__":
    unittest.main()
