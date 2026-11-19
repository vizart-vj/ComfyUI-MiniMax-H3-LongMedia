"""Backward-compatible Director range request contract tests."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT.parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

_spec = importlib.util.spec_from_file_location("longmedia_director_plan", ROOT / "director_plan.py")
director_plan = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(director_plan)


class DirectorRangeContractTests(unittest.TestCase):
    @staticmethod
    def document(regeneration=None, *, base_kind="generated"):
        return {
            "version": 10,
            "kind": "h3_longmedia_director",
            "project_id": "range-test",
            "fps": 24,
            "subjects": [],
            "shots": [{
                "clip_id": "shot-a",
                "name": "A",
                "prompt": "preserve the scene",
                "start": 0,
                "duration": 5,
                "base_kind": base_kind,
            }],
            "camera_blocks": [],
            "audio_blocks": [],
            "extra_tracks": [],
            "regeneration": regeneration or {"mode": "none"},
        }

    def test_range_request_is_normalized_as_a_one_shot_local_frame_contract(self):
        raw = self.document({
            "mode": "range",
            "clip_id": "shot-a",
            "request_id": "req-1",
            "source_revision": "take-r3",
            "range": {"start_frame": 24, "end_frame": 72},
        })
        request = director_plan.normalize_document(raw)["regeneration"]
        self.assertEqual(request["mode"], "range")
        self.assertEqual(request["range"], {"start_frame": 24, "end_frame": 72})
        self.assertTrue(request["preserve_audio"])
        self.assertEqual(request["source_revision"], "take-r3")

    def test_range_rejects_missing_source_branch_instead_of_falling_back_to_full_run(self):
        raw = self.document({
            "mode": "range",
            "clip_id": "shot-a",
            "request_id": "req-1",
            "range": {"start_frame": 24, "end_frame": 72},
        })
        with self.assertRaisesRegex(ValueError, "source_revision"):
            director_plan.normalize_document(raw)

    def test_range_rejects_invalid_bounds_and_media_clip(self):
        invalid = self.document({
            "mode": "range",
            "clip_id": "shot-a",
            "request_id": "req-1",
            "source_revision": "take-r3",
            "range": {"start_frame": 72, "end_frame": 24},
        })
        with self.assertRaisesRegex(ValueError, "range"):
            director_plan.normalize_document(invalid)

        media = self.document({
            "mode": "range",
            "clip_id": "shot-a",
            "request_id": "req-1",
            "source_revision": "take-r3",
            "range": {"start_frame": 24, "end_frame": 72},
        }, base_kind="media")
        with self.assertRaisesRegex(ValueError, "GENERATED"):
            director_plan.normalize_document(media)

    def test_legacy_clip_request_defaults_are_unchanged(self):
        raw = self.document({"mode": "none"})
        normalized = director_plan.normalize_document(raw)
        self.assertEqual(normalized["regeneration"]["mode"], "none")
        self.assertNotIn("range", normalized["regeneration"])

    @staticmethod
    def bridge_document(regeneration):
        left_duration = 5.0
        bridge_duration = 119 / 24
        right_start = left_duration + bridge_duration
        raw = DirectorRangeContractTests.document(regeneration)
        raw["shots"] = [
            {"clip_id": "left", "name": "Left", "prompt": "left", "start": 0,
             "duration": left_duration, "base_kind": "generated"},
            {"clip_id": "bridge", "name": "Bridge", "prompt": "between", "start": left_duration,
             "duration": bridge_duration, "base_kind": "generated"},
            {"clip_id": "right", "name": "Right", "prompt": "right", "start": right_start,
             "duration": 5.0, "base_kind": "generated"},
        ]
        raw["regeneration"] = regeneration
        return raw

    def test_bridge_request_requires_current_immediate_neighbor_revisions(self):
        raw = self.bridge_document({
            "mode": "bridge", "clip_id": "bridge", "request_id": "bridge-run-1",
            "bridge": {
                "left_clip_id": "left", "right_clip_id": "right",
                "left_revision": "left-r3", "right_revision": "right-r8",
            },
        })
        request = director_plan.normalize_document(raw)["regeneration"]
        self.assertEqual(request["mode"], "bridge")
        self.assertEqual(request["bridge"]["left_revision"], "left-r3")
        self.assertEqual(request["bridge"]["right_revision"], "right-r8")

    def test_bridge_accepts_default_duration_after_six_decimal_timeline_serialization(self):
        raw = self.bridge_document({
            "mode": "bridge", "clip_id": "bridge", "request_id": "bridge-run-1",
            "bridge": {
                "left_clip_id": "left", "right_clip_id": "right",
                "left_revision": "left-r3", "right_revision": "right-r8",
            },
        })
        # The Director persists durations to six decimal places. The default
        # 119-frame bridge therefore arrives as 4.958333 seconds.
        raw["shots"][1]["duration"] = 4.958333
        raw["shots"][2]["start"] = 9.958333

        request = director_plan.normalize_document(raw)["regeneration"]

        self.assertEqual(request["mode"], "bridge")
        self.assertEqual(director_plan.director_bridge_duration_frames(4.958333), 119)
        self.assertIsNone(director_plan.director_bridge_duration_frames(4.5))

    def test_bridge_rejects_stale_sources_non_generated_neighbors_and_off_grid_duration(self):
        stale = self.bridge_document({
            "mode": "bridge", "clip_id": "bridge", "request_id": "bridge-run-1",
            "bridge": {
                "left_clip_id": "old-left", "right_clip_id": "right",
                "left_revision": "left-r3", "right_revision": "right-r8",
            },
        })
        with self.assertRaisesRegex(ValueError, "immediate MAIN neighbors"):
            director_plan.normalize_document(stale)

        media = self.bridge_document({
            "mode": "bridge", "clip_id": "bridge", "request_id": "bridge-run-1",
            "bridge": {
                "left_clip_id": "left", "right_clip_id": "right",
                "left_revision": "left-r3", "right_revision": "right-r8",
            },
        })
        media["shots"][2]["base_kind"] = "media"
        with self.assertRaisesRegex(ValueError, "GENERATED MAIN"):
            director_plan.normalize_document(media)

        off_grid = self.bridge_document({
            "mode": "bridge", "clip_id": "bridge", "request_id": "bridge-run-1",
            "bridge": {
                "left_clip_id": "left", "right_clip_id": "right",
                "left_revision": "left-r3", "right_revision": "right-r8",
            },
        })
        off_grid["shots"][1]["duration"] = 4.5
        off_grid["shots"][2]["start"] = 9.5
        with self.assertRaisesRegex(ValueError, "17-frame H3 output periods"):
            director_plan.normalize_document(off_grid)

        gapped = self.bridge_document({
            "mode": "bridge", "clip_id": "bridge", "request_id": "bridge-run-1",
            "bridge": {
                "left_clip_id": "left", "right_clip_id": "right",
                "left_revision": "left-r3", "right_revision": "right-r8",
            },
        })
        gapped["shots"][1]["start"] += 0.5
        gapped["shots"][2]["start"] += 0.5
        with self.assertRaisesRegex(ValueError, "touching GENERATED MAIN clips"):
            director_plan.normalize_document(gapped)


if __name__ == "__main__":
    unittest.main()
