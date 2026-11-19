"""Pending Director renders stay outside the approved active branch."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from director_cache import DirectorClipCache


class DirectorPendingAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = DirectorClipCache(self.temp.name, "pending-bridge")
        self.video = torch.zeros((1, 24, 2, 2, 2), dtype=torch.float32)
        self.audio = torch.zeros((1, 32, 2, 2), dtype=torch.float32)

    def save(self, *, activate=True, metadata=None):
        return self.cache.save_take(
            clip_id="bridge",
            clip_index=1,
            display_video=self.video,
            display_audio=self.audio,
            continuation_video=self.video,
            continuation_audio=self.audio,
            metadata={"segment_length_frames": 5, **(metadata or {})},
            activate=activate,
        )

    def save_clip(self, clip_id, clip_index, *, activate=True, metadata=None):
        return self.cache.save_take(
            clip_id=clip_id,
            clip_index=clip_index,
            display_video=self.video,
            display_audio=self.audio,
            continuation_video=self.video,
            continuation_audio=self.audio,
            metadata={"segment_length_frames": 22, "overlap_frames": 5, **(metadata or {})},
            activate=activate,
        )

    def make_pending_bridge(self):
        left = self.save_clip("left", 0, metadata={"semantic_fingerprint": "left-old"})
        right = self.save_clip("right", 1, metadata={"semantic_fingerprint": "right-old"})
        candidate = self.save_clip("bridge", 1, activate=False, metadata={"request_mode": "bridge"})
        self.cache.set_pending_bridge_metadata("bridge", candidate["revision"], {
            "timeline_clip_order": ["left", "bridge", "right"],
            "generated_clip_ids": ["left", "right"],
            "left_clip_id": "left",
            "left_revision": left["revision"],
            "right_clip_id": "right",
            "right_revision": right["revision"],
            "overlap_frames": 5,
            "rebase_manifest": [
                {"clip_id": "left", "revision": left["revision"], "clip_index": 0,
                 "semantic_fingerprint": "left-new", "outgoing_contract_fingerprint": "left-out"},
                {"clip_id": "right", "revision": right["revision"], "clip_index": 2,
                 "semantic_fingerprint": "right-new", "outgoing_contract_fingerprint": "right-out"},
            ],
        }, merge=False)
        return left, right, candidate

    def test_pending_render_does_not_replace_active_render_or_editor_take(self):
        approved = self.save()
        pending = self.save(activate=False, metadata={"request_mode": "bridge"})

        self.assertTrue(pending["pending_acceptance"])
        self.assertEqual(pending["state"], "pending_acceptance")
        self.assertEqual(self.cache.active_metadata("bridge")["revision"], approved["revision"])
        self.assertEqual(self.cache.editor_active_revision("bridge"), approved["revision"])

    def test_cache_instances_for_one_project_share_a_reentrant_mutation_lock(self):
        same_project = DirectorClipCache(self.temp.name, "pending-bridge")
        other_project = DirectorClipCache(self.temp.name, "other-project")
        self.assertIs(self.cache._mutation_lock, same_project._mutation_lock)
        self.assertIsNot(self.cache._mutation_lock, other_project._mutation_lock)

    def test_pending_result_cannot_be_activated_through_ordinary_restore(self):
        pending = self.save(activate=False)
        with self.assertRaisesRegex(ValueError, "pending render result"):
            self.cache.activate_revision("bridge", pending["revision"])
        plan = self.cache.plan_restore_branch(["bridge"], "bridge", pending["revision"])
        self.assertFalse(plan["ok"])
        self.assertEqual(plan["reason"], "take_pending_acceptance")
        self.assertIsNone(self.cache.active_metadata("bridge"))

    def test_pending_result_cannot_be_duplicated_as_an_approved_take(self):
        pending = self.save(activate=False, metadata={"request_mode": "bridge"})
        with self.assertRaisesRegex(ValueError, "must be accepted or discarded"):
            self.cache.duplicate_revision("bridge", pending["revision"])
        self.assertIsNone(self.cache.active_metadata("bridge"))

    def test_deleting_last_approved_take_does_not_promote_pending_candidate(self):
        approved = self.save()
        pending = self.save(activate=False, metadata={"request_mode": "bridge"})
        self.cache.delete_revision("bridge", approved["revision"])
        self.assertIsNone(self.cache.active_metadata("bridge"))
        self.assertIsNone(self.cache.editor_active_revision("bridge"))
        self.assertTrue(self.cache.metadata_for_revision("bridge", pending["revision"])["pending_acceptance"])

    def test_pending_render_does_not_consume_an_active_empty_take_slot(self):
        draft = self.cache.create_draft_take(clip_id="bridge", clip_index=1)
        pending = self.save(activate=False)
        self.assertNotEqual(pending["revision"], draft["revision"])
        self.assertEqual(self.cache.editor_active_revision("bridge"), draft["revision"])
        self.assertIsNone(self.cache.active_metadata("bridge"))

    def test_accept_rebases_metadata_and_activates_only_after_boundary_proof(self):
        left, right, candidate = self.make_pending_bridge()

        result = self.cache.accept_pending_bridge(
            "bridge", candidate["revision"], clip_order=["left", "bridge", "right"],
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["left_boundary_proof"]["reason"], "boundary_hash_match")
        self.assertEqual(result["right_boundary_proof"]["reason"], "boundary_hash_match")
        self.assertEqual(self.cache.active_metadata("bridge")["revision"], candidate["revision"])
        self.assertFalse(self.cache.metadata_for_revision("bridge", candidate["revision"])["pending_acceptance"])
        self.assertEqual(self.cache.metadata_for_revision("left", left["revision"])["semantic_fingerprint"], "left-new")
        self.assertEqual(self.cache.metadata_for_revision("right", right["revision"])["clip_index"], 2)
        self.assertEqual(result["rebased_clips"], ["left", "right"])

    def test_failed_second_boundary_proof_rolls_back_rebased_metadata_and_pointers(self):
        left, right, candidate = self.make_pending_bridge()
        bad_candidate = self.cache.metadata_for_revision("bridge", candidate["revision"])
        bad_candidate["outgoing_boundary_hash"] = "deliberately-not-the-right-head"
        self.cache._write_take_metadata("bridge", candidate["revision"], bad_candidate)
        left_before = self.cache.metadata_for_revision("left", left["revision"])
        right_before = self.cache.metadata_for_revision("right", right["revision"])

        with self.assertRaisesRegex(RuntimeError, "Right bridge boundary proof failed"):
            self.cache.accept_pending_bridge(
                "bridge", candidate["revision"], clip_order=["left", "bridge", "right"],
            )

        self.assertEqual(self.cache.active_metadata("left")["revision"], left["revision"])
        self.assertEqual(self.cache.active_metadata("right")["revision"], right["revision"])
        self.assertIsNone(self.cache.active_metadata("bridge"))
        self.assertEqual(self.cache.metadata_for_revision("left", left["revision"]), left_before)
        self.assertEqual(self.cache.metadata_for_revision("right", right["revision"]), right_before)
        self.assertTrue(self.cache.metadata_for_revision("bridge", candidate["revision"])["pending_acceptance"])


if __name__ == "__main__":
    unittest.main()
