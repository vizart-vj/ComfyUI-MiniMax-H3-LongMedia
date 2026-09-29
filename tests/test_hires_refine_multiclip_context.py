"""Regression coverage for refined overlap context across MultiClip cuts."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from hires_refine import (  # noqa: E402
    freeze_video_prefix_mask,
    inherit_refined_video_prefix,
)


class TestMulticlipHiresBoundaryContext(unittest.TestCase):
    def test_current_clip_inherits_only_previous_refined_tail_as_hidden_prefix(self):
        previous = torch.arange(2 * 2 * 64 * 3 * 4, dtype=torch.float16).reshape(2, 2, 64, 3, 4)
        current = torch.full((2, 2, 512, 3, 4), -1.0, dtype=torch.float16)

        copied = inherit_refined_video_prefix(current, previous, overlap_tokens=3)

        self.assertEqual(copied, 3)
        self.assertTrue(torch.equal(current[:, :, :3], previous[:, :, -3:]))
        self.assertTrue(torch.equal(current[:, :, 3:], torch.full_like(current[:, :, 3:], -1.0)))

    def test_refine_mask_freezes_inherited_prefix_in_full_and_window_coordinates(self):
        full_mask = torch.ones((1, 1, 8, 2, 2), dtype=torch.float32)
        frozen_full = freeze_video_prefix_mask(full_mask, frozen_tokens=3)

        self.assertEqual(frozen_full, 3)
        self.assertEqual(int(full_mask[:, :, :3].sum()), 0)
        self.assertTrue(torch.all(full_mask[:, :, 3:] == 1.0))

        window_mask = torch.ones((1, 1, 4, 2, 2), dtype=torch.float32)
        frozen_window = freeze_video_prefix_mask(
            window_mask, frozen_tokens=3, global_start_token=2,
        )

        self.assertEqual(frozen_window, 1)
        self.assertEqual(int(window_mask[:, :, :1].sum()), 0)
        self.assertTrue(torch.all(window_mask[:, :, 1:] == 1.0))

    def test_overlap_copy_rejects_incompatible_spatial_geometry(self):
        previous = torch.zeros((1, 2, 5, 3, 4))
        current = torch.zeros((1, 2, 7, 4, 4))

        with self.assertRaisesRegex(ValueError, "geometry"):
            inherit_refined_video_prefix(current, previous, overlap_tokens=2)

    def test_overlap_copy_converts_only_to_the_target_dtype_and_device(self):
        previous = torch.full((1, 2, 4, 2, 2), 7.0, dtype=torch.float32)
        current = torch.zeros((1, 2, 6, 2, 2), dtype=torch.float16)

        copied = inherit_refined_video_prefix(current, previous, overlap_tokens=2)

        self.assertEqual(copied, 2)
        self.assertEqual(current.dtype, torch.float16)
        self.assertTrue(torch.all(current[:, :, :2] == 7.0))

    @unittest.skipUnless(torch.cuda.device_count() > 1, "requires two CUDA devices")
    def test_overlap_copy_handles_cross_gpu_storage(self):
        previous = torch.full((1, 2, 4, 2, 2), 5.0, device="cuda:0")
        current = torch.zeros((1, 2, 6, 2, 2), device="cuda:1")

        copied = inherit_refined_video_prefix(current, previous, overlap_tokens=2)

        self.assertEqual(copied, 2)
        self.assertEqual(current.device, torch.device("cuda:1"))
        self.assertTrue(torch.all(current[:, :, :2] == 5.0).item())

    def test_sampler_integrates_refined_overlap_only_for_multiclip_hires_refine(self):
        nodes_path = Path(_PROJECT_ROOT, "nodes.py")
        source = nodes_path.read_text(encoding="utf-8")

        self.assertIn("source=previous_refined_display_tail", source)
        self.assertIn("getattr(plan, 'mode', None) == 'multiclip'", source)
        self.assertIn("frozen_video_prefix_tokens=int(frozen_refined_prefix_tokens)", source)


if __name__ == "__main__":
    unittest.main()
