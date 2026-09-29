import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("longmedia_director_plan", ROOT / "director_plan.py")
director_plan = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(director_plan)


class DirectorClipActivationTests(unittest.TestCase):
    def test_enabled_run_scope_excludes_disabled_and_media_clips(self):
        self.assertEqual(
            director_plan.director_enabled_render_indices(
                (True, False, True, False),
                ("generated", "generated", "media", "generated"),
            ),
            (0,),
        )

    def test_enabled_run_scope_rejects_misaligned_contract(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            director_plan.director_enabled_render_indices((True,), ("generated", "media"))

    def test_partial_run_requires_current_geometry_for_all_generated_clips(self):
        self.assertEqual(
            director_plan.director_validate_enabled_run_scope(
                (True, False, True),
                ("generated", "generated", "media"),
                (True, True, True),
            ),
            (0,),
        )
        with self.assertRaisesRegex(RuntimeError, "outdated clip indices: 2"):
            director_plan.director_validate_enabled_run_scope(
                (True, False),
                ("generated", "generated"),
                (True, False),
            )

    def test_director_plan_preserves_clip_run_switch_and_legacy_default(self):
        raw = {
            "version": 10,
            "kind": "h3_longmedia_director",
            "project_id": "activation-test",
            "fps": 24,
            "subjects": [],
            "shots": [
                {"clip_id": "kept", "name": "Kept", "duration": 5, "run_enabled": False},
                {"clip_id": "legacy", "name": "Legacy", "duration": 5},
            ],
            "camera_blocks": [],
            "audio_blocks": [],
            "extra_tracks": [],
            "refmods": [],
        }
        clip_plan, _, _, _ = director_plan.compile_director_plan(raw)
        self.assertEqual(
            [clip["director_run_enabled"] for clip in clip_plan["clips"]],
            [False, True],
        )


if __name__ == "__main__":
    unittest.main()
