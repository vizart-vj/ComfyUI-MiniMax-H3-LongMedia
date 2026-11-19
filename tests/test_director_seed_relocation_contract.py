"""Prove that a bridge insertion can rebase indices without rerolling cached seeds."""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from director_cache import canonical_hash


def load_contract_functions():
    source = (ROOT / "nodes.py").read_text(encoding="utf-8")
    wanted = {
        "_v85_segment_seed",
        "_lm_director_render_clip_metadata",
        "_lm_director_semantic_payload",
        "_lm_director_position_insensitive_payload",
        "_lm_director_take_matches_after_relocation",
    }
    tree = ast.parse(source)
    module = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted],
        type_ignores=[],
    )
    namespace = {"_director_canonical_hash": canonical_hash}
    exec(compile(module, str(ROOT / "nodes.py"), "exec"), namespace)
    return namespace


class DirectorSeedRelocationContractTests(unittest.TestCase):
    @staticmethod
    def plan(*, start, seed=None):
        return SimpleNamespace(
            mode="multiclip",
            segment_seeds=(None, seed),
            director_clip_ids=("first", "suffix"),
            segment_prompt_summaries=("first prompt", "unchanged suffix prompt"),
            director_clip_metadata=(
                {"clip_id": "first", "duration": 5.0, "seed": None},
                {
                    "clip_id": "suffix", "duration": 5.0, "seed": seed,
                    "director_start_seconds": start,
                    "director_end_seconds": start + 5.0,
                    "camera": {"transition_to_next": False, "rig": "locked"},
                },
            ),
            director_reference_fingerprint="same-references",
            audio_output_mode="preserve",
            lip_sync_target_audio_locked=False,
        )

    def test_unseeded_suffix_keeps_its_approved_effective_seed_after_index_shift(self):
        functions = load_contract_functions()
        base_seed = 100
        old_plan = self.plan(start=5.0)
        old_payload = functions["_lm_director_semantic_payload"](old_plan, base_seed, 1)
        metadata = {
            "clip_id": "suffix",
            "clip_index": 1,
            "effective_seed": old_payload["effective_seed"],
            "prompt_summary": old_payload["prompt"],
            "clip_metadata": old_payload["clip_metadata"],
            "reference_fingerprint": old_payload["reference_fingerprint"],
            "semantic_fingerprint": canonical_hash(old_payload),
        }

        moved_plan = self.plan(start=5.0 + 119 / 24)
        # The suffix is now index 2 and its default derived seed would change.
        moved_plan.segment_seeds = (None, None, None)
        moved_plan.director_clip_ids = ("first", "bridge", "suffix")
        moved_plan.segment_prompt_summaries = (
            "first prompt", "bridge prompt", "unchanged suffix prompt",
        )
        moved_plan.director_clip_metadata = (
            {"clip_id": "first", "duration": 5.0, "seed": None},
            {"clip_id": "bridge", "duration": 119 / 24, "seed": None},
            {
                **moved_plan.director_clip_metadata[1],
                "director_start_seconds": 5.0 + 119 / 24,
                "director_end_seconds": 10.0 + 119 / 24,
            },
        )
        self.assertTrue(
            functions["_lm_director_take_matches_after_relocation"](
                metadata, moved_plan, base_seed, 2,
            )
        )
        self.assertFalse(
            functions["_lm_director_take_matches_after_relocation"](
                metadata, moved_plan, base_seed + 1, 2,
            )
        )

    def test_explicit_clip_seed_change_still_invalidates_the_cached_take(self):
        functions = load_contract_functions()
        old_plan = self.plan(start=5.0, seed=9001)
        old_payload = functions["_lm_director_semantic_payload"](old_plan, 100, 1)
        metadata = {
            "clip_id": "suffix", "clip_index": 1,
            "effective_seed": old_payload["effective_seed"],
            "prompt_summary": old_payload["prompt"],
            "clip_metadata": old_payload["clip_metadata"],
            "reference_fingerprint": old_payload["reference_fingerprint"],
            "semantic_fingerprint": canonical_hash(old_payload),
        }
        changed = self.plan(start=5.0 + 119 / 24, seed=9002)
        changed.segment_seeds = (None, None, 9002)
        changed.director_clip_ids = ("first", "bridge", "suffix")
        changed.segment_prompt_summaries = (
            "first prompt", "bridge prompt", "unchanged suffix prompt",
        )
        changed.director_clip_metadata = (
            {"clip_id": "first", "duration": 5.0, "seed": None},
            {"clip_id": "bridge", "duration": 119 / 24, "seed": None},
            changed.director_clip_metadata[1],
        )
        self.assertFalse(
            functions["_lm_director_take_matches_after_relocation"](
                metadata, changed, 100, 2,
            )
        )

    def test_exact_frozen_uint64_seed_preserves_a_cached_take_after_index_shift(self):
        functions = load_contract_functions()
        base_seed = (1 << 63) + 0x123456789
        old_plan = self.plan(start=5.0)
        old_payload = functions["_lm_director_semantic_payload"](old_plan, base_seed, 1)
        cached_seed = old_payload["effective_seed"]
        metadata = {
            "clip_id": "suffix", "clip_index": 1,
            "effective_seed": cached_seed,
            "prompt_summary": old_payload["prompt"],
            "clip_metadata": old_payload["clip_metadata"],
            "reference_fingerprint": old_payload["reference_fingerprint"],
            "semantic_fingerprint": canonical_hash(old_payload),
        }

        frozen = self.plan(start=5.0 + 119 / 24, seed=str(cached_seed))
        frozen.segment_seeds = (None, None, str(cached_seed))
        frozen.director_clip_ids = ("first", "bridge", "suffix")
        frozen.segment_prompt_summaries = (
            "first prompt", "bridge prompt", "unchanged suffix prompt",
        )
        frozen.director_clip_metadata = (
            {"clip_id": "first", "duration": 5.0, "seed": None},
            {"clip_id": "bridge", "duration": 119 / 24, "seed": None},
            {
                **frozen.director_clip_metadata[1],
                "seed": str(cached_seed),
                "director_start_seconds": 5.0 + 119 / 24,
                "director_end_seconds": 10.0 + 119 / 24,
            },
        )
        self.assertGreater(cached_seed, 2**53)
        self.assertTrue(
            functions["_lm_director_take_matches_after_relocation"](
                metadata, frozen, base_seed, 2,
            )
        )

        changed_seed = str(cached_seed + 1)
        frozen.segment_seeds = (None, None, changed_seed)
        frozen.director_clip_metadata[2]["seed"] = changed_seed
        self.assertFalse(
            functions["_lm_director_take_matches_after_relocation"](
                metadata, frozen, base_seed, 2,
            )
        )


class ExactSeedJsonContractTests(unittest.TestCase):
    def test_status_seed_text_preserves_unsigned_64_bit_decimal_exactly(self):
        source = (ROOT / "__init__.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_director_effective_seed_text"
        )
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "__init__.py", "exec"), namespace)
        expected = str((1 << 64) - 1)
        self.assertEqual(namespace["_director_effective_seed_text"]((1 << 64) - 1), expected)
        self.assertEqual(namespace["_director_effective_seed_text"]("18446744073709551615"), expected)
        self.assertIsNone(namespace["_director_effective_seed_text"]("not-a-seed"))


if __name__ == "__main__":
    unittest.main()
