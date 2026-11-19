"""Runtime H3 TAKE tensor validation used before two-sided bridge sampling."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from director_cache import validate_take_tensor_contract


class DirectorTakeTensorContractTests(unittest.TestCase):
    def setUp(self):
        self.continuation_shapes = ((1, 24, 7, 4, 6), (1, 32, 2, 38))
        self.display_shapes = ((1, 24, 7, 4, 6), (1, 32, 2, 38))
        pairs = {
            "continuation": self.continuation_shapes,
            "display": self.display_shapes,
        }
        self.take = {
            "metadata": {
                f"{name}_shapes": [list(shape) for shape in shapes]
                for name, shapes in pairs.items()
            },
            "continuation_video": torch.zeros(self.continuation_shapes[0], dtype=torch.float16),
            "continuation_audio": torch.zeros(self.continuation_shapes[1], dtype=torch.float32),
            "display_video": torch.ones(self.display_shapes[0], dtype=torch.float16),
            "display_audio": torch.ones(self.display_shapes[1], dtype=torch.float32),
        }

    def validate(self, take=None, *, continuation=None, display=None):
        return validate_take_tensor_contract(
            self.take if take is None else take,
            continuation_shapes=continuation or self.continuation_shapes,
            display_shapes=display or self.display_shapes,
            video_dtype=torch.float16,
            audio_dtype=torch.float32,
        )

    def test_accepts_cached_shapes_and_keeps_cpu_offload_valid(self):
        ok, report = self.validate()
        self.assertTrue(ok)
        self.assertEqual(report["reason"], "tensor_contract_match")
        self.assertEqual(self.take["continuation_video"].device.type, "cpu")

    def test_rejects_mismatch_between_loaded_tensor_and_saved_shape(self):
        corrupt = dict(self.take)
        corrupt["display_audio"] = torch.ones((1, 32, 2, 37), dtype=torch.float32)
        ok, report = self.validate(corrupt)
        self.assertFalse(ok)
        self.assertEqual(report["reason"], "display_audio_shape_mismatch")

    def test_rejects_cached_contract_that_differs_from_current_plan(self):
        wrong_expected = (self.continuation_shapes[0], (1, 32, 2, 39))
        ok, report = self.validate(continuation=wrong_expected)
        self.assertFalse(ok)
        self.assertEqual(report["reason"], "continuation_shapes_contract_mismatch")

    def test_rejects_dtype_mismatch_before_boundary_copy(self):
        corrupt = dict(self.take)
        corrupt["continuation_video"] = corrupt["continuation_video"].float()
        ok, report = self.validate(corrupt)
        self.assertFalse(ok)
        self.assertEqual(report["reason"], "continuation_video_dtype_mismatch")


if __name__ == "__main__":
    unittest.main()
