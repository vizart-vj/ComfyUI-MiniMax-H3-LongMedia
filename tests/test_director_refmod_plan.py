import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import importlib.util

_spec = importlib.util.spec_from_file_location("longmedia_director_plan", ROOT / "director_plan.py")
director_plan = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(director_plan)


class DirectorRefModPlanTests(unittest.TestCase):
    def _document(self):
        return {
            "version": 10,
            "kind": "h3_longmedia_director",
            "project_id": "refmod-test",
            "fps": 24,
            "subjects": [],
            "shots": [
                {"clip_id": "a", "name": "A", "prompt": "first", "start": 0, "duration": 5},
                {"clip_id": "b", "name": "B", "prompt": "second", "start": 5, "duration": 5},
            ],
            "camera_blocks": [],
            "audio_blocks": [],
            "refmods": [
                {
                    "refmod_id": "joker",
                    "name": "Joker",
                    "path": "characters/joker.safetensors",
                    "kind": "image",
                    "concept_type": "character",
                    "description": "Reusable identity reference",
                }
            ],
            "extra_tracks": [
                {
                    "track_id": "r1",
                    "type": "refmod",
                    "name": "REFMOD Joker",
                    "enabled": True,
                    "locked": False,
                    "muted": False,
                    "clips": [
                        {
                            "clip_id": "rc1",
                            "name": "Joker",
                            "start": 3,
                            "duration": 4,
                            "refmod_id": "joker",
                            "strength": 1.0,
                        }
                    ],
                }
            ],
        }

    def test_normalization_preserves_refmod_library_and_track_binding(self):
        doc = director_plan.normalize_document(self._document())
        self.assertEqual(10, doc["version"])
        self.assertEqual("joker", doc["refmods"][0]["refmod_id"])
        track = doc["extra_tracks"][0]
        self.assertEqual("refmod", track["type"])
        self.assertEqual("joker", track["clips"][0]["refmod_id"])
        self.assertEqual(1.0, track["clips"][0]["strength"])

    def test_compile_localizes_refmod_intervals_per_main_clip(self):
        clip_plan, _camera, plan, _report = director_plan.compile_director_plan(
            json.dumps(self._document()), global_prompt="world"
        )
        clips = clip_plan["clips"]
        self.assertEqual(2, len(clips))
        self.assertEqual(
            [(3.0, 5.0)],
            [(x["start_seconds"], x["end_seconds"]) for x in clips[0]["director_refmod_spans"]],
        )
        self.assertEqual(
            [(0.0, 2.0)],
            [(x["start_seconds"], x["end_seconds"]) for x in clips[1]["director_refmod_spans"]],
        )
        self.assertEqual("joker", clips[0]["director_refmod_spans"][0]["refmod_id"])
        self.assertEqual("characters/joker.safetensors", clips[1]["director_refmod_spans"][0]["path"])
        self.assertEqual(plan["document"]["refmods"], clip_plan["director"]["refmods"])

    def test_invalid_refmod_binding_is_dropped_not_retargeted(self):
        doc = self._document()
        doc["extra_tracks"][0]["clips"][0]["refmod_id"] = "missing"
        normalized = director_plan.normalize_document(doc)
        self.assertIsNone(normalized["extra_tracks"][0]["clips"][0]["refmod_id"])
        clip_plan, *_ = director_plan.compile_director_plan(normalized)
        self.assertEqual([], clip_plan["clips"][0]["director_refmod_spans"])


if __name__ == "__main__":
    unittest.main()
