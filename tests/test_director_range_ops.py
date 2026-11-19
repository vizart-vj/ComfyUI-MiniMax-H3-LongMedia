"""H3-native range mapping and source-lock regression tests."""

from __future__ import annotations

import unittest
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latent_ops import (
    apply_h3_range_source_lock,
    apply_h3_right_take_head_lock,
    h3_video_frame_spans,
    restore_h3_range_outside,
    restore_h3_right_take_head,
    snap_h3_video_range,
)


class DirectorRangeOpsTests(unittest.TestCase):
    def test_h3_video_frame_spans_follow_native_one_then_four_frame_steps(self):
        self.assertEqual(h3_video_frame_spans(5), ((0, 1), (1, 5)))
        self.assertEqual(h3_video_frame_spans(22)[-3:], ((13, 17), (17, 18), (18, 22)))
        self.assertEqual(h3_video_frame_spans(39)[-2:], ((34, 35), (35, 39)))

    def test_snap_range_expands_to_token_boundaries_and_accounts_for_hidden_prefix(self):
        # A 22-frame MultiClip preroll ends at a real H3 token boundary.  A visible
        # frame range of 1..3 expands to the native 22..26-frame token.
        self.assertEqual(snap_h3_video_range(1, 3, 39, frame_offset=22), (0, 4))

    def test_invalid_or_unmappable_range_fails_closed(self):
        cases = [
            (4, 4, 22, 0),
            (-1, 3, 22, 0),
            (3, 23, 22, 0),
            (0, 1, 22, 2),  # hidden prefix is not on the H3 token boundary
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                snap_h3_video_range(*args[:3], frame_offset=args[3])

    def test_source_lock_preserves_both_video_sides_and_full_audio_exactly(self):
        target_video = torch.full((1, 24, 12, 2, 2), 99.0, dtype=torch.float16)
        source_video = torch.arange(1 * 24 * 12 * 2 * 2, dtype=torch.float16).reshape(1, 24, 12, 2, 2)
        target_audio = torch.full((1, 32, 2, 65), 99.0, dtype=torch.float32)
        source_audio = torch.arange(1 * 32 * 2 * 65, dtype=torch.float32).reshape(1, 32, 2, 65)

        video, audio, video_mask, audio_mask, report = apply_h3_range_source_lock(
            target_video,
            target_audio,
            source_video,
            source_audio,
            start_frame=0,
            end_frame=4,
            frame_offset=22,
            audio_prefix_tokens=37,
            preserve_audio=True,
        )

        # The approved TAKE supplies x0 everywhere; the mask alone decides which
        # token is regenerated. Token 7 covers visible frames 0..4 after preroll.
        self.assertTrue(torch.equal(video, source_video))
        self.assertTrue(torch.equal(audio, source_audio))
        self.assertTrue(torch.equal(video_mask[:, :, :7], torch.zeros_like(video_mask[:, :, :7])))
        self.assertTrue(torch.equal(video_mask[:, :, 7:8], torch.ones_like(video_mask[:, :, 7:8])))
        self.assertTrue(torch.equal(video_mask[:, :, 8:], torch.zeros_like(video_mask[:, :, 8:])))
        self.assertEqual(int(torch.count_nonzero(audio_mask)), 0)
        self.assertEqual(report["snapped_range"], [0, 4])
        self.assertEqual(report["video_token_range"], [7, 8])
        self.assertEqual(report["audio_token_range"], [37, 44])

    def test_existing_locked_tokens_inside_edit_remain_locked(self):
        target_video = torch.full((1, 24, 12, 1, 1), 2.0)
        source_video = torch.full_like(target_video, 1.0)
        target_audio = torch.full((1, 32, 2, 65), 2.0)
        source_audio = torch.full_like(target_audio, 1.0)
        existing_video_mask = torch.ones((1, 1, 12, 1, 1))
        existing_audio_mask = torch.ones((1, 1, 1, 65))
        existing_video_mask[:, :, 7] = 0.0
        existing_audio_mask[..., 37:39] = 0.0

        video, audio, video_mask, audio_mask, _ = apply_h3_range_source_lock(
            target_video,
            target_audio,
            source_video,
            source_audio,
            start_frame=0,
            end_frame=4,
            frame_offset=22,
            audio_prefix_tokens=37,
            preserve_audio=False,
            existing_video_mask=existing_video_mask,
            existing_audio_mask=existing_audio_mask,
        )

        self.assertEqual(float(video_mask[0, 0, 7, 0, 0]), 0.0)
        self.assertTrue(torch.equal(video[:, :, 7], target_video[:, :, 7]))
        self.assertTrue(torch.equal(audio_mask[..., 37:39], torch.zeros_like(audio_mask[..., 37:39])))
        self.assertTrue(torch.equal(audio[..., 37:39], target_audio[..., 37:39]))
        self.assertTrue(torch.all(audio_mask[..., 39:44] == 1.0))

    def test_source_lock_rejects_changed_geometry_or_dtype(self):
        video = torch.zeros((1, 24, 12, 2, 2), dtype=torch.float16)
        audio = torch.zeros((1, 32, 2, 65), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "geometry"):
            apply_h3_range_source_lock(
                video, audio, video[:, :, :, :, :-1], audio,
                start_frame=0, end_frame=4, frame_offset=22,
                audio_prefix_tokens=37, preserve_audio=True,
            )
        with self.assertRaisesRegex(RuntimeError, "dtype"):
            apply_h3_range_source_lock(
                video, audio, video.float(), audio,
                start_frame=0, end_frame=4, frame_offset=22,
                audio_prefix_tokens=37, preserve_audio=True,
            )

    def test_bridge_pins_right_take_head_to_tail_and_freezes_that_mask(self):
        target_video = torch.full((1, 24, 22, 2, 2), 9.0, dtype=torch.float16)
        right_video = torch.arange(1 * 24 * 22 * 2 * 2, dtype=torch.float16).reshape(1, 24, 22, 2, 2)
        target_audio = torch.full((1, 32, 2, 122), 9.0, dtype=torch.float32)
        right_audio = torch.arange(1 * 32 * 2 * 122, dtype=torch.float32).reshape(1, 32, 2, 122)
        old_video_mask = torch.ones((1, 1, 22, 1, 1))
        old_audio_mask = torch.ones((1, 1, 1, 122))

        video, audio, video_mask, audio_mask, report = apply_h3_right_take_head_lock(
            target_video,
            target_audio,
            right_video,
            right_audio,
            overlap_frames=5,
            existing_video_mask=old_video_mask,
            existing_audio_mask=old_audio_mask,
        )

        self.assertTrue(torch.equal(video[:, :, -2:], right_video[:, :, :2]))
        self.assertTrue(torch.equal(audio[..., -8:], right_audio[..., :8]))
        self.assertTrue(torch.all(video_mask[:, :, -2:] == 0.0))
        self.assertTrue(torch.all(audio_mask[..., -8:] == 0.0))
        self.assertEqual(report["mode"], "right_take_head_to_bridge_tail")
        self.assertGreater(report["free_video_tokens"], 0)
        self.assertGreater(report["free_audio_tokens"], 0)

    def test_bridge_post_refine_restore_copies_only_right_overlap(self):
        rendered_video = torch.full((1, 24, 22, 2, 2), 7.0, dtype=torch.float16)
        right_video = torch.arange(1 * 24 * 22 * 2 * 2, dtype=torch.float16).reshape(1, 24, 22, 2, 2)
        rendered_audio = torch.full((1, 32, 2, 122), 7.0, dtype=torch.float32)
        right_audio = torch.arange(1 * 32 * 2 * 122, dtype=torch.float32).reshape(1, 32, 2, 122)

        video, audio, report = restore_h3_right_take_head(
            rendered_video,
            rendered_audio,
            right_video,
            right_audio,
            overlap_frames=5,
        )

        self.assertTrue(torch.equal(video[:, :, :-2], rendered_video[:, :, :-2]))
        self.assertTrue(torch.equal(audio[..., :-8], rendered_audio[..., :-8]))
        self.assertTrue(torch.equal(video[:, :, -2:], right_video[:, :, :2]))
        self.assertTrue(torch.equal(audio[..., -8:], right_audio[..., :8]))
        self.assertEqual(report["mode"], "post_refine_right_head_restore")

    def test_bridge_rejects_when_boundary_locks_leave_no_generated_middle(self):
        video = torch.zeros((1, 24, 12, 1, 1))  # 22f overlap maps to 7 tokens per edge.
        audio = torch.zeros((1, 32, 2, 65))
        with self.assertRaisesRegex(RuntimeError, "no free middle"):
            apply_h3_right_take_head_lock(
                video, audio, video, audio, overlap_frames=22,
            )

    def test_post_refine_restore_keeps_only_requested_video_range_and_audio_policy(self):
        source_video = torch.arange(1 * 24 * 12, dtype=torch.float32).reshape(1, 24, 12, 1, 1)
        rendered_video = torch.full_like(source_video, -1.0)
        source_audio = torch.arange(1 * 32 * 2 * 65, dtype=torch.float32).reshape(1, 32, 2, 65)
        rendered_audio = torch.full_like(source_audio, -2.0)

        video, audio, report = restore_h3_range_outside(
            rendered_video,
            rendered_audio,
            source_video,
            source_audio,
            start_frame=0,
            end_frame=4,
            frame_offset=22,
            audio_prefix_tokens=37,
            preserve_audio=False,
        )

        self.assertTrue(torch.equal(video[:, :, :7], source_video[:, :, :7]))
        self.assertTrue(torch.all(video[:, :, 7:8] == -1.0))
        self.assertTrue(torch.equal(video[:, :, 8:], source_video[:, :, 8:]))
        self.assertTrue(torch.equal(audio[..., :37], source_audio[..., :37]))
        self.assertTrue(torch.all(audio[..., 37:44] == -2.0))
        self.assertTrue(torch.equal(audio[..., 44:], source_audio[..., 44:]))
        self.assertEqual(report["video_token_range"], [7, 8])
        self.assertEqual(report["audio_token_range"], [37, 44])
        self.assertFalse(report["preserve_audio"])


if __name__ == "__main__":
    unittest.main()
