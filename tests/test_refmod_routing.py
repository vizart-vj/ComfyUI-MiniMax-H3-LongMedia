"""
TDD tests for PackedLayout-aware temporal RefMod routing.

Design under test (redesigned per architecture review):
  - Members come from ORDERED NATIVE REF BLOCK DESCRIPTORS, never inferred
    from adjacent ref segments.  Segments are signature-walked in packed
    order: text -> keyframes (cond/cond_audio) -> one consumed run of
    segments per descriptor (image => ref_img, audio => ref_audio when
    rt > 0 else NO segment, video/video_audio => optional ref_audio +
    ref_img as ONE member) -> audio -> video targets.
  - Visibility comes from AUTHORED start_frame/end_frame evaluated against
    target latent_t FRAME_PER_TOKEN cells (frames, not RoPE clocks).
  - Ref/text/keyframe rows as queries keep STOCK attention (empty key
    tuple); only target audio/video query intervals gate RefMod keys.
  - Route shape is built for nodes.py combined masking: exact key ranges,
    query intervals, per-key strengths, deterministic ValueError
    validation for every malformed input.

Run (from project root):
    python run_refmod_tests.py
    python tests/test_refmod_routing.py
    python -m pytest tests/test_refmod_routing.py -q --import-mode=importlib

The pytest path works because this module pre-registers inert stubs for the
project package (see _install_pytest_package_stub): pytest 9 treats the repo
root -- which carries a ComfyUI-runtime __init__.py with relative imports --
as a Package collector and imports that __init__ during test setup.
"""
from __future__ import annotations

import math
import sys
import os
import types
import unittest

# Ensure project root is on sys.path for refmod_routing import
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _install_pytest_package_stub() -> None:
    """Register inert package stubs so pytest can import the repo root.

    The project root's __init__.py performs relative imports that require the
    ComfyUI runtime.  pytest 9 models that root as a ``Package`` collector and
    imports its ``__init__`` when running test setup; the module names pytest
    may derive (the directory name or the root-level ``__init__``) are
    pre-seeded here, during collection, so ``Package.setup()`` resolves the
    stub from ``sys.modules`` instead of executing the real file.  Purely
    additive: other names keep their normal import semantics.
    """
    pkg = types.ModuleType("ComfyUI-MiniMax-H3-LongMedia")
    pkg.__path__ = [_PROJECT_ROOT]
    pkg.__package__ = "ComfyUI-MiniMax-H3-LongMedia"
    for name in (
        "ComfyUI-MiniMax-H3-LongMedia",
        "ComfyUI-MiniMax-H3-LongMedia.__init__",
        "__init__",
    ):
        sys.modules.setdefault(name, pkg)


_install_pytest_package_stub()

from refmod_routing import (
    RefModRange,
    RefModRoute,
    FakePackedLayout,
    group_refmod_segments,
    compute_temporal_gates,
    merge_attention_routes,
    resolve_refmod_spec_index,
    refmod_row_mask,
)


# ---------------------------------------------------------------------------
# Canonical test geometry (matches native PackedLayout arithmetic)
# ---------------------------------------------------------------------------

TEXT_LEN = 10
LATENT_T = 4  # target video latent cells
LATENT_H = 8
LATENT_W = 8
AUDIO_T = 5  # target audio latent frames

FRAME_PATTERN = (1, 4, 4, 4, 4)  # native FRAME_PER_TOKEN
FRAME_ROWS = (LATENT_H // 2) * (LATENT_W // 2)  # 16
AUDIO_ROWS = AUDIO_T * 2  # channel-major stereo
VIDEO_ROWS = LATENT_T * FRAME_ROWS  # 64

# Target timeline frames covered by the latent_t cells (FRAME_PER_TOKEN):
#   cell 0 -> [0, 1), cell 1 -> [1, 5), cell 2 -> [5, 9), cell 3 -> [9, 13)
CELL_FRAMES = [(0, 1), (1, 5), (5, 9), (9, 13)]
TOTAL_FRAMES = 13  # sum(FRAME_PATTERN[:LATENT_T])

REF_VIDEO_T = 3  # latent_t of a video ref (3 * 16 = 48 rows)
REF_AUDIO_T = 2  # ref audio latent frames (2 * 2 = 4 rows)

REF_KIND_SEGMENTS = ("ref_img", "ref_audio")


def _ref_segments(layout):
    """All ref segments of a layout in packed row order."""
    return sorted((a, b, k) for a, b, k in layout.segments
                  if k in REF_KIND_SEGMENTS)


def _interval_keys(route, start, stop):
    """Keys of the query interval exactly matching (start, stop)."""
    matches = [keys for s, e, keys in route.query_intervals
               if s == start and e == stop]
    assert len(matches) == 1, (
        f"expected exactly one query interval {start}:{stop}, "
        f"got {len(matches)}"
    )
    return matches[0]


def _video_cell_rows(layout):
    video_a, video_b = layout.video_span
    return [
        (video_a + t * FRAME_ROWS, video_a + (t + 1) * FRAME_ROWS)
        for t in range(LATENT_T)
    ]


def _key_range_triples(keys):
    """Drop strengths for comparisons: ((start, stop), ...)."""
    return tuple((k[0], k[1]) for k in keys)


# ---------------------------------------------------------------------------
# FakePackedLayout: deterministic native-segment scaffold
# ---------------------------------------------------------------------------

class TestFakePackedLayout(unittest.TestCase):
    def test_no_refs(self):
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T)
        kinds = [kind for _, _, kind in layout.segments]
        self.assertEqual(kinds, ["text", "audio", "video"])
        self.assertEqual(layout.seq_len, TEXT_LEN + AUDIO_ROWS + VIDEO_ROWS)

    def test_signature(self):
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T)
        self.assertEqual(layout.signature,
                         (TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T))

    def test_with_image_ref(self):
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  refs=[{"kind": "image", "latent_h": LATENT_H,
                                         "latent_w": LATENT_W}])
        kinds = [kind for _, _, kind in layout.segments]
        self.assertIn("ref_img", kinds)
        self.assertEqual(len(layout.ref_ranges), 1)

    def test_with_video_ref(self):
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  refs=[{"kind": "video_audio",
                                         "latent_t": REF_VIDEO_T,
                                         "ref_audio_t": REF_AUDIO_T,
                                         "latent_h": LATENT_H,
                                         "latent_w": LATENT_W}])
        kinds = [kind for _, _, kind in layout.segments]
        self.assertIn("ref_img", kinds)
        self.assertIn("ref_audio", kinds)
        self.assertEqual(len(layout.ref_ranges), 1)

    def test_with_audio_ref(self):
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  refs=[{"kind": "audio",
                                         "ref_audio_t": REF_AUDIO_T}])
        kinds = [kind for _, _, kind in layout.segments]
        self.assertIn("ref_audio", kinds)

    def test_seq_len_consistency(self):
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  refs=[{"kind": "video_audio",
                                         "latent_t": REF_VIDEO_T,
                                         "ref_audio_t": REF_AUDIO_T,
                                         "latent_h": LATENT_H,
                                         "latent_w": LATENT_W},
                                        {"kind": "image", "latent_h": LATENT_H,
                                         "latent_w": LATENT_W}])
        self.assertEqual(layout.segments[-1][1], layout.seq_len)

    def test_keyframes_pack_before_refs(self):
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  keyframes=[{"latent_t": 2}, {"audio_t": 3}],
                                  refs=[{"kind": "image", "latent_h": LATENT_H,
                                         "latent_w": LATENT_W}])
        kinds = [kind for _, _, kind in layout.segments]
        self.assertEqual(kinds[:3], ["text", "cond", "cond_audio"])
        self.assertEqual(kinds[3], "ref_img")
        # cond rows = 2 * 16, cond_audio rows = 3 * 2
        self.assertEqual(layout.segments[1], (10, 42, "cond"))
        self.assertEqual(layout.segments[2], (42, 48, "cond_audio"))
        self.assertEqual(layout.segments[3], (48, 64, "ref_img"))


# ---------------------------------------------------------------------------
# group_refmod_segments: signature-walk over ordered native descriptors
# ---------------------------------------------------------------------------

class TestGroupRefModSegments(unittest.TestCase):
    def test_no_refs_returns_empty(self):
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T)
        self.assertEqual(group_refmod_segments(layout), [])

    def test_image_member_exact_range(self):
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  refs=[{"kind": "image", "latent_h": LATENT_H,
                                         "latent_w": LATENT_W}])
        ranges = group_refmod_segments(layout)
        self.assertEqual(len(ranges), 1)
        r = ranges[0]
        self.assertEqual(r.member_index, 0)
        self.assertEqual(r.kind, "image")
        self.assertEqual((r.row_start, r.row_stop), (10, 10 + FRAME_ROWS))
        self.assertEqual(r.segment_groups,
                         ((10, 10 + FRAME_ROWS, "ref_img"),))
        self.assertEqual(r.row_start, layout.segments[0][1])  # right after text

    def test_video_audio_block_is_one_member(self):
        """video_audio => optional ref_audio + ref_img consumed as ONE member."""
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  refs=[{"kind": "video_audio",
                                         "latent_t": REF_VIDEO_T,
                                         "ref_audio_t": REF_AUDIO_T,
                                         "latent_h": LATENT_H,
                                         "latent_w": LATENT_W}])
        ranges = group_refmod_segments(layout)
        self.assertEqual(len(ranges), 1)
        r = ranges[0]
        self.assertEqual(r.kind, "video_audio")
        self.assertEqual(r.segment_groups,
                         ((10, 10 + REF_AUDIO_T * 2, "ref_audio"),
                          (10 + REF_AUDIO_T * 2,
                           10 + REF_AUDIO_T * 2 + REF_VIDEO_T * FRAME_ROWS,
                           "ref_img")))
        self.assertEqual((r.row_start, r.row_stop), (10, 62))

    def test_video_rt0_consumes_no_audio_segment(self):
        """rt=0 => the descriptor contributes no ref_audio segment."""
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  refs=[{"kind": "video", "latent_t": 2,
                                         "ref_audio_t": 0,
                                         "latent_h": LATENT_H,
                                         "latent_w": LATENT_W},
                                        {"kind": "audio",
                                         "ref_audio_t": REF_AUDIO_T}])
        ranges = group_refmod_segments(layout)
        self.assertEqual(len(ranges), 2)
        self.assertEqual(ranges[0].segment_groups,
                         ((10, 10 + 2 * FRAME_ROWS, "ref_img"),))
        # Audio member starts exactly where the video member ended: no
        # ref_audio segment was consumed for rt=0.
        self.assertEqual(ranges[1].row_start, ranges[0].row_stop)
        self.assertEqual(ranges[1].kind, "audio")

    def test_audio_rt0_member_is_segmentless_but_keeps_its_id(self):
        """An rt=0 audio block packs no rows yet keeps a stable member id."""
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  refs=[{"kind": "audio", "ref_audio_t": 0},
                                        {"kind": "image", "latent_h": LATENT_H,
                                         "latent_w": LATENT_W}])
        ranges = group_refmod_segments(layout)
        self.assertEqual([r.member_index for r in ranges], [0, 1])
        empty = ranges[0]
        self.assertEqual(empty.kind, "audio")
        self.assertEqual(empty.row_start, empty.row_stop)
        self.assertEqual(empty.segment_groups, ())
        # The image member still lands at the exact packed offset.
        self.assertEqual((ranges[1].row_start, ranges[1].row_stop),
                         (TEXT_LEN, TEXT_LEN + FRAME_ROWS))

    def test_multi_block_video_audio_audio_image_exact_ranges(self):
        """Multiple blocks incl. video+audio -> stable ids, exact ranges."""
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  refs=[
                                      {"kind": "video_audio",
                                       "latent_t": REF_VIDEO_T,
                                       "ref_audio_t": REF_AUDIO_T,
                                       "latent_h": LATENT_H,
                                       "latent_w": LATENT_W},
                                      {"kind": "audio",
                                       "ref_audio_t": REF_AUDIO_T},
                                      {"kind": "image", "latent_h": LATENT_H,
                                       "latent_w": LATENT_W},
                                  ])
        ranges = group_refmod_segments(layout)
        self.assertEqual([r.member_index for r in ranges], [0, 1, 2])
        self.assertEqual([r.kind for r in ranges],
                         ["video_audio", "audio", "image"])
        # Exact row ranges for the packed layout.
        self.assertEqual([(r.row_start, r.row_stop) for r in ranges],
                         [(10, 62), (62, 66), (66, 82)])
        # Segment groups tile the ref region exactly, in packed order.
        flat = [g for r in ranges for g in r.segment_groups]
        self.assertEqual(sorted(flat), _ref_segments(layout))
        for prev, nxt in zip(ranges, ranges[1:]):
            self.assertEqual(prev.row_stop, nxt.row_start)
        for r in ranges:
            self.assertEqual(r.row_start, r.segment_groups[0][0])
            self.assertEqual(r.row_stop, r.segment_groups[-1][1])
        # Deterministic across runs.
        again = group_refmod_segments(layout)
        self.assertEqual(
            [(r.member_index, r.kind, r.row_start, r.row_stop, r.segment_groups,
              r.start_frame, r.end_frame, r.strength) for r in ranges],
            [(r.member_index, r.kind, r.row_start, r.row_stop, r.segment_groups,
              r.start_frame, r.end_frame, r.strength) for r in again],
        )

    def test_keyframes_are_skipped_before_refs(self):
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  keyframes=[{"latent_t": 2}, {"audio_t": 3}],
                                  refs=[{"kind": "image", "latent_h": LATENT_H,
                                         "latent_w": LATENT_W}])
        ranges = group_refmod_segments(layout)
        self.assertEqual(len(ranges), 1)
        self.assertEqual((ranges[0].row_start, ranges[0].row_stop), (48, 64))

    def test_host_spelling_segment_kinds_tolerated(self):
        """ref_image / ref_video segment spellings normalize inside groups."""
        layout = FakePackedLayout.__new__(FakePackedLayout)
        layout.segments = [
            (0, 10, "text"),
            (10, 14, "ref_image"),   # image member, 4x4 latent -> 2*2 rows
            (14, 16, "ref_audio"),   # video_audio member, rt=1 -> 2 rows
            (16, 48, "ref_video"),   # video_audio visual, vt=2 -> 2*16 rows
            (48, 58, "audio"),
            (58, 122, "video"),
        ]
        layout.signature = (10, 4, 8, 8, 5)
        layout.seq_len = 122
        ranges = group_refmod_segments(layout, [
            {"kind": "image", "latent_h": 4, "latent_w": 4},
            {"kind": "video_audio", "latent_t": 2, "ref_audio_t": 1,
             "latent_h": 8, "latent_w": 8},
        ])
        self.assertEqual([r.kind for r in ranges], ["image", "video_audio"])
        self.assertEqual(ranges[0].segment_groups, ((10, 14, "ref_img"),))
        self.assertEqual(ranges[1].segment_groups,
                         ((14, 16, "ref_audio"), (16, 48, "ref_img")))

    def test_wrong_kind_order_raises(self):
        """A video descriptor expecting ref_audio first must not accept ref_img."""
        layout = FakePackedLayout.__new__(FakePackedLayout)
        layout.segments = [
            (0, 10, "text"),
            (10, 42, "ref_img"),     # wrong: descriptor expects ref_audio first
            (42, 52, "audio"),
            (52, 116, "video"),
        ]
        layout.signature = (10, 4, 8, 8, 5)
        layout.seq_len = 116
        with self.assertRaises(ValueError):
            group_refmod_segments(layout, [
                {"kind": "video_audio", "latent_t": 2, "ref_audio_t": 1,
                 "latent_h": 8, "latent_w": 8},
            ])

    def test_missing_descriptors_raises(self):
        """Members are never inferred from adjacent segments."""
        layout = FakePackedLayout.__new__(FakePackedLayout)
        layout.segments = [
            (0, 10, "text"),
            (10, 26, "ref_img"),
            (26, 36, "audio"),
            (36, 100, "video"),
        ]
        layout.signature = (10, 4, 8, 8, 5)
        layout.seq_len = 100
        with self.assertRaises(ValueError):
            group_refmod_segments(layout)  # no ref_blocks attr, no argument

    def test_unreferenced_extra_ref_segment_raises(self):
        """More ref segments than descriptors is malformed."""
        layout = FakePackedLayout.__new__(FakePackedLayout)
        layout.segments = [
            (0, 10, "text"),
            (10, 26, "ref_img"),
            (26, 36, "audio"),
            (36, 100, "video"),
        ]
        layout.signature = (10, 4, 8, 8, 5)
        layout.seq_len = 100
        with self.assertRaises(ValueError):
            group_refmod_segments(layout, [])  # descriptors say: no refs

    def test_audio_row_count_mismatch_raises(self):
        layout = FakePackedLayout.__new__(FakePackedLayout)
        layout.segments = [
            (0, 10, "text"),
            (10, 16, "ref_audio"),   # 6 rows != rt*2 == 4
            (16, 26, "audio"),
            (26, 90, "video"),
        ]
        layout.signature = (10, 4, 8, 8, 5)
        layout.seq_len = 90
        with self.assertRaises(ValueError):
            group_refmod_segments(layout, [{"kind": "audio", "ref_audio_t": 2}])

    def test_unknown_descriptor_kind_raises(self):
        layout = FakePackedLayout.__new__(FakePackedLayout)
        layout.segments = [
            (0, 10, "text"),
            (10, 20, "audio"),
            (20, 84, "video"),
        ]
        layout.signature = (10, 4, 8, 8, 5)
        layout.seq_len = 84
        with self.assertRaises(ValueError):
            group_refmod_segments(layout, [{"kind": "depth"}])

    def test_text_rows_must_match_signature(self):
        layout = FakePackedLayout.__new__(FakePackedLayout)
        layout.segments = [
            (0, 12, "text"),         # signature text_len is 10
            (12, 22, "audio"),
            (22, 86, "video"),
        ]
        layout.signature = (10, 4, 8, 8, 5)
        layout.seq_len = 86
        with self.assertRaises(ValueError):
            group_refmod_segments(layout, [])

    def test_target_row_counts_must_match_signature(self):
        layout = FakePackedLayout.__new__(FakePackedLayout)
        layout.segments = [
            (0, 10, "text"),
            (10, 16, "audio"),       # 6 rows != audio_t * 2 == 10
            (16, 80, "video"),       # 64 rows ok
        ]
        layout.signature = (10, 4, 8, 8, 5)
        layout.seq_len = 80
        with self.assertRaises(ValueError):
            group_refmod_segments(layout, [])
        layout.segments = [
            (0, 10, "text"),
            (10, 20, "audio"),
            (20, 36, "video"),       # 16 rows != 64
        ]
        layout.seq_len = 36
        with self.assertRaises(ValueError):
            group_refmod_segments(layout, [])

    def test_malformed_segment_shapes_raise(self):
        def bare(segments):
            layout = FakePackedLayout.__new__(FakePackedLayout)
            layout.segments = segments
            layout.signature = (10, 4, 8, 8, 5)
            layout.seq_len = 0
            return layout

        with self.assertRaises(ValueError):  # non-iterable
            group_refmod_segments(bare(42), [])
        with self.assertRaises(ValueError):  # wrong arity
            group_refmod_segments(bare([(0, 10)]), [])
        with self.assertRaises(ValueError):  # inverted range
            group_refmod_segments(
                bare([(0, 10, "text"), (15, 12, "ref_img"),
                      (15, 25, "audio"), (25, 89, "video")]), [])
        with self.assertRaises(ValueError):  # negative bounds
            group_refmod_segments(
                bare([(-3, 10, "text"), (10, 20, "audio"),
                      (20, 84, "video")]), [])
        with self.assertRaises(ValueError):  # empty segment list
            group_refmod_segments(bare([]), [])

    def test_missing_signature_raises(self):
        layout = FakePackedLayout.__new__(FakePackedLayout)
        layout.segments = [(0, 10, "text"), (10, 20, "audio"),
                           (20, 84, "video")]
        with self.assertRaises(ValueError):
            group_refmod_segments(layout, [])

    def test_authored_frame_validation(self):
        base = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                refs=[{"kind": "image", "latent_h": LATENT_H,
                                       "latent_w": LATENT_W}])
        bad_windows = [
            {"start_frame": 5, "end_frame": 3},   # inverted
            {"start_frame": -1, "end_frame": 4},  # negative start
            {"start_frame": "x", "end_frame": 4},  # non-numeric
            {"start_frame": 0, "end_frame": math.nan},  # non-finite
        ]
        for window in bad_windows:
            with self.subTest(window=window):
                blk = {"kind": "image", "latent_h": LATENT_H,
                       "latent_w": LATENT_W}
                blk.update(window)
                with self.assertRaises(ValueError):
                    group_refmod_segments(base, [blk])

    def test_strength_validation(self):
        base = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                refs=[{"kind": "image", "latent_h": LATENT_H,
                                       "latent_w": LATENT_W}])
        for strength in (-0.1, 1.5, math.nan, "strong"):
            with self.subTest(strength=strength):
                blk = {"kind": "image", "latent_h": LATENT_H,
                       "latent_w": LATENT_W, "strength": strength}
                with self.assertRaises(ValueError):
                    group_refmod_segments(base, [blk])

    def test_authored_window_and_strength_flow_into_members(self):
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  refs=[{"kind": "image", "latent_h": LATENT_H,
                                         "latent_w": LATENT_W,
                                         "start_frame": 2, "end_frame": 9,
                                         "strength": 0.5}])
        r = group_refmod_segments(layout)[0]
        self.assertEqual((r.start_frame, r.end_frame), (2.0, 9.0))
        self.assertEqual(r.strength, 0.5)

    def test_default_authored_window_is_full_span(self):
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  refs=[{"kind": "image", "latent_h": LATENT_H,
                                         "latent_w": LATENT_W}])
        r = group_refmod_segments(layout)[0]
        self.assertEqual(r.start_frame, 0.0)
        self.assertTrue(math.isinf(r.end_frame))
        self.assertEqual(r.strength, 1.0)


# ---------------------------------------------------------------------------
# compute_temporal_gates -> RefModRoute (target-only gating)
# ---------------------------------------------------------------------------

class TestRouteShape(unittest.TestCase):
    def test_returns_route_with_practical_shape(self):
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T)
        route = compute_temporal_gates(layout, frame_pattern=FRAME_PATTERN)
        self.assertIsInstance(route, RefModRoute)
        self.assertEqual(route.text_span, (0, TEXT_LEN))
        self.assertEqual(route.audio_span,
                         (TEXT_LEN, TEXT_LEN + AUDIO_ROWS))
        self.assertEqual(route.video_span,
                         (TEXT_LEN + AUDIO_ROWS,
                          TEXT_LEN + AUDIO_ROWS + VIDEO_ROWS))
        self.assertEqual(len(route.cells), LATENT_T + 1)  # video cells + audio

    def test_intervals_tile_the_entire_sequence(self):
        """nodes.py can build a combined mask by iterating intervals alone."""
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  keyframes=[{"latent_t": 2}, {"audio_t": 3}],
                                  refs=[{"kind": "image", "latent_h": LATENT_H,
                                         "latent_w": LATENT_W}])
        route = compute_temporal_gates(layout, frame_pattern=FRAME_PATTERN)
        intervals = [(s, e) for s, e, _k in route.query_intervals]
        self.assertEqual(intervals, sorted(intervals))
        expected = []
        cursor = 0
        for s, e in intervals:
            self.assertEqual(s, cursor, f"gap/overlap before row {s}")
            self.assertGreater(e, s)
            cursor = e
        self.assertEqual(cursor, layout.seq_len)

    def test_video_cells_use_frame_per_token_windows(self):
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T)
        route = compute_temporal_gates(layout, frame_pattern=FRAME_PATTERN)
        video_cells = [c for c in route.cells if c[4] == "video"]
        self.assertEqual([(c[2], c[3]) for c in video_cells], CELL_FRAMES)
        self.assertEqual([(c[0], c[1]) for c in video_cells],
                         _video_cell_rows(layout))
        audio_cells = [c for c in route.cells if c[4] == "audio"]
        self.assertEqual(len(audio_cells), 1)
        self.assertEqual((audio_cells[0][2], audio_cells[0][3]),
                         (0, TOTAL_FRAMES))

    def test_no_refs_yields_common_only_target_intervals(self):
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T)
        route = compute_temporal_gates(layout, frame_pattern=FRAME_PATTERN)
        for _s, _e, keys in route.query_intervals:
            self.assertEqual(keys, ())

    def test_keys_are_exact_ranges_with_member_and_strength(self):
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  refs=[{"kind": "image", "latent_h": LATENT_H,
                                         "latent_w": LATENT_W,
                                         "start_frame": 0,
                                         "end_frame": TOTAL_FRAMES,
                                         "strength": 0.5}])
        route = compute_temporal_gates(layout, frame_pattern=FRAME_PATTERN)
        rows = _video_cell_rows(layout)
        keys = _interval_keys(route, rows[0][0], rows[0][1])
        self.assertEqual(keys, ((10, 10 + FRAME_ROWS, 0, 0.5),))
        # Every key range belongs to a routed member and carries its strength.
        member = route.members[0]
        for s, e, m_index, strength in keys:
            self.assertEqual(m_index, 0)
            self.assertEqual(strength, member.strength)
            self.assertTrue(member.row_start <= s < e <= member.row_stop)

    def test_authored_window_gates_cells(self):
        """start_frame/end_frame decide which target cells see the ref."""
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  refs=[{"kind": "video", "latent_t": REF_VIDEO_T,
                                         "ref_audio_t": 0,
                                         "latent_h": LATENT_H,
                                         "latent_w": LATENT_W,
                                         "start_frame": 0, "end_frame": 5}])
        route = compute_temporal_gates(layout, frame_pattern=FRAME_PATTERN)
        ref_group = (10, 10 + REF_VIDEO_T * FRAME_ROWS)
        rows = _video_cell_rows(layout)
        # Cells [0,1) and [1,5) overlap [0,5); [5,9) and [9,13) do not.
        for tidx, expect_active in enumerate((True, True, False, False)):
            s, e = rows[tidx]
            keys = _interval_keys(route, s, e)
            self.assertEqual(_key_range_triples(keys),
                             ((ref_group[0], ref_group[1]),) if expect_active else (),
                             f"cell {tidx} (frames {CELL_FRAMES[tidx]})")
            mask = route.key_mask_for_cell(s, e)
            self.assertEqual(mask.get(ref_group[0], False), expect_active)

    def test_authored_window_outside_target_is_inactive_everywhere(self):
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  refs=[{"kind": "image", "latent_h": LATENT_H,
                                         "latent_w": LATENT_W,
                                         "start_frame": 100, "end_frame": 200}])
        route = compute_temporal_gates(layout, frame_pattern=FRAME_PATTERN)
        for s, e, keys in route.query_intervals:
            video_a, video_b = route.video_span
            audio_a, audio_b = route.audio_span
            if (video_a <= s and e <= video_b) or (audio_a <= s and e <= audio_b):
                self.assertEqual(keys, (),
                                 f"target interval {s}:{e} saw an inactive ref")

    def test_audio_target_queries_gate_active_rows_only(self):
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  refs=[
                                      {"kind": "audio", "ref_audio_t": REF_AUDIO_T,
                                       "start_frame": 0,
                                       "end_frame": TOTAL_FRAMES},
                                      {"kind": "audio", "ref_audio_t": REF_AUDIO_T,
                                       "start_frame": 100, "end_frame": 200},
                                  ])
        route = compute_temporal_gates(layout, frame_pattern=FRAME_PATTERN)
        audio_a, audio_b = route.audio_span
        keys = _interval_keys(route, audio_a, audio_b)
        active_group = (10, 10 + REF_AUDIO_T * 2)          # member 0
        inactive_group = (10 + REF_AUDIO_T * 2,
                          10 + 2 * REF_AUDIO_T * 2)        # member 1
        self.assertEqual(_key_range_triples(keys),
                         (active_group,))
        self.assertNotIn(inactive_group, _key_range_triples(keys))
        # And member 1's rows stay hidden in every per-cell mask.
        mask = route.key_mask_for_cell(audio_a, audio_b)
        self.assertTrue(mask.get(active_group[0], False))
        self.assertFalse(mask.get(inactive_group[0], False))

    def test_non_target_queries_keep_stock_attention(self):
        """Text / keyframe / ref rows never receive gated key tuples."""
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  keyframes=[{"latent_t": 2}, {"audio_t": 3}],
                                  refs=[{"kind": "video_audio",
                                         "latent_t": REF_VIDEO_T,
                                         "ref_audio_t": REF_AUDIO_T,
                                         "latent_h": LATENT_H,
                                         "latent_w": LATENT_W}])
        route = compute_temporal_gates(layout, frame_pattern=FRAME_PATTERN)
        stock_rows = set(range(0, route.video_span[0]))  # everything before targets
        checked = 0
        for s, e, keys in route.query_intervals:
            if s >= route.video_span[0]:
                continue  # target video cells gate; checked elsewhere
            if (route.audio_span[0] <= s and e <= route.audio_span[1]):
                continue  # target audio gates; checked elsewhere
            if e > s and all(r in stock_rows for r in range(s, e)):
                checked += 1
                self.assertEqual(keys, (),
                                 f"stock query {s}:{e} carries gated keys")
        # text + cond + cond_audio + 2 ref groups + audio target is excluded
        self.assertGreaterEqual(checked, 5)

    def test_route_is_deterministic(self):
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  keyframes=[{"latent_t": 1}],
                                  refs=[{"kind": "video_audio",
                                         "latent_t": REF_VIDEO_T,
                                         "ref_audio_t": REF_AUDIO_T,
                                         "latent_h": LATENT_H,
                                         "latent_w": LATENT_W,
                                         "start_frame": 0, "end_frame": 9,
                                         "strength": 0.75},
                                        {"kind": "image", "latent_h": LATENT_H,
                                         "latent_w": LATENT_W}])
        r1 = compute_temporal_gates(layout, frame_pattern=FRAME_PATTERN)
        r2 = compute_temporal_gates(layout, frame_pattern=FRAME_PATTERN)
        self.assertEqual(r1.query_intervals, r2.query_intervals)
        self.assertEqual(
            {k: dict(v) for k, v in r1.key_masks.items()},
            {k: dict(v) for k, v in r2.key_masks.items()},
        )
        self.assertEqual(r1.cells, r2.cells)

    def test_malformed_layouts_raise_deterministically(self):
        layout = FakePackedLayout.__new__(FakePackedLayout)
        layout.segments = []
        with self.assertRaises(ValueError):
            compute_temporal_gates(layout, frame_pattern=FRAME_PATTERN)
        layout.segments = 42
        with self.assertRaises(ValueError):
            compute_temporal_gates(layout, frame_pattern=FRAME_PATTERN)
        layout.segments = [(0, 10, "text"), (10, 20, "audio"),
                           (20, 84, "video")]
        with self.assertRaises(ValueError):  # no signature
            compute_temporal_gates(layout, frame_pattern=FRAME_PATTERN)

    def test_expected_target_frame_count_contract_rejects_clock_drift(self):
        refs = [{
            "kind": "image", "latent_h": LATENT_H, "latent_w": LATENT_W,
            "start_frame": 0, "end_frame": 13,
        }]
        layout = FakePackedLayout(
            TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T, refs=refs,
        )
        route = compute_temporal_gates(
            layout, refs, frame_pattern=FRAME_PATTERN, expected_frame_count=13,
        )
        self.assertTrue(route.cells)
        with self.assertRaisesRegex(ValueError, "authored target clock"):
            compute_temporal_gates(
                layout, refs, frame_pattern=FRAME_PATTERN, expected_frame_count=14,
            )


# ---------------------------------------------------------------------------
# refmod_row_mask: flat key-side visibility for combined masking
# ---------------------------------------------------------------------------

class TestRefModRowMask(unittest.TestCase):
    def _layout(self, refs):
        return FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                refs=refs)

    def test_mask_shape_and_text_without_refs(self):
        layout = self._layout(None)
        mask = refmod_row_mask(layout, frame_pattern=FRAME_PATTERN)
        self.assertEqual(len(mask), layout.seq_len)
        self.assertTrue(all(mask))

    def test_active_ref_rows_visible_inactive_hidden(self):
        layout = self._layout([
            {"kind": "video", "latent_t": REF_VIDEO_T, "ref_audio_t": 0,
             "latent_h": LATENT_H, "latent_w": LATENT_W,
             "start_frame": 0, "end_frame": 5},
            {"kind": "image", "latent_h": LATENT_H, "latent_w": LATENT_W,
             "start_frame": 100, "end_frame": 200},
        ])
        route = compute_temporal_gates(layout, frame_pattern=FRAME_PATTERN)
        # Union view (target_cell=None): member 0 overlaps target cells,
        # member 1 is authored outside the target timeline.
        mask = refmod_row_mask(layout, frame_pattern=FRAME_PATTERN)
        m0_start, m0_stop = route.members[0].row_start, route.members[0].row_stop
        m1_start, m1_stop = route.members[1].row_start, route.members[1].row_stop
        self.assertTrue(any(mask[m0_start:m0_stop]))
        self.assertFalse(any(mask[m1_start:m1_stop]))
        # Text rows keep stock visibility.
        self.assertTrue(all(mask[0:TEXT_LEN]))

    def test_mask_agrees_with_route_for_each_cell(self):
        layout = self._layout([
            {"kind": "video_audio", "latent_t": REF_VIDEO_T,
             "ref_audio_t": REF_AUDIO_T,
             "latent_h": LATENT_H, "latent_w": LATENT_W,
             "start_frame": 0, "end_frame": 5},
        ])
        route = compute_temporal_gates(layout, frame_pattern=FRAME_PATTERN)
        member = route.members[0]
        for cell in route.cells:
            s, e = cell[0], cell[1]
            mask = refmod_row_mask(layout, target_cell=(s, e),
                                   frame_pattern=FRAME_PATTERN)
            masked_active = any(mask[member.row_start:member.row_stop])
            cell_keys = route.key_mask_for_cell(s, e)
            route_active = any(v for k, v in cell_keys.items())
            self.assertEqual(masked_active, route_active,
                             f"mask/route disagree for cell {s}:{e}")

    def test_full_span_target_cell_is_a_union_query(self):
        layout = self._layout([
            {"kind": "image", "latent_h": LATENT_H, "latent_w": LATENT_W,
             "start_frame": 5, "end_frame": 9},
        ])
        mask = refmod_row_mask(layout, target_cell=layout.video_span,
                               frame_pattern=FRAME_PATTERN)
        ref_start = TEXT_LEN
        self.assertTrue(any(mask[ref_start:ref_start + FRAME_ROWS]))

    def test_target_cell_outside_targets_hides_refs(self):
        layout = self._layout([
            {"kind": "image", "latent_h": LATENT_H, "latent_w": LATENT_W},
        ])
        mask = refmod_row_mask(layout, target_cell=(0, TEXT_LEN),
                               frame_pattern=FRAME_PATTERN)
        ref_start = TEXT_LEN
        self.assertFalse(any(mask[ref_start:ref_start + FRAME_ROWS]))

    def test_inverted_target_cell_raises(self):
        layout = self._layout([
            {"kind": "image", "latent_h": LATENT_H, "latent_w": LATENT_W},
        ])
        with self.assertRaises(ValueError):
            refmod_row_mask(layout, target_cell=(20, 10),
                            frame_pattern=FRAME_PATTERN)

    def test_mask_is_deterministic(self):
        layout = self._layout([
            {"kind": "video_audio", "latent_t": REF_VIDEO_T,
             "ref_audio_t": REF_AUDIO_T,
             "latent_h": LATENT_H, "latent_w": LATENT_W},
            {"kind": "audio", "ref_audio_t": REF_AUDIO_T},
        ])
        self.assertEqual(refmod_row_mask(layout, frame_pattern=FRAME_PATTERN),
                         refmod_row_mask(layout, frame_pattern=FRAME_PATTERN))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):
    def test_single_image_ref_route_nonempty(self):
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  refs=[{"kind": "image", "latent_h": LATENT_H,
                                         "latent_w": LATENT_W}])
        route = compute_temporal_gates(layout, frame_pattern=FRAME_PATTERN)
        self.assertGreater(len(route.query_intervals), 0)

    def test_many_refs_stable_indexing(self):
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  refs=[
                                      {"kind": "image", "latent_h": LATENT_H,
                                       "latent_w": LATENT_W},
                                      {"kind": "image", "latent_h": LATENT_H,
                                       "latent_w": LATENT_W},
                                      {"kind": "audio",
                                       "ref_audio_t": REF_AUDIO_T},
                                  ])
        ranges = group_refmod_segments(layout)
        self.assertEqual([r.member_index for r in ranges], [0, 1, 2])
        self.assertEqual([r.kind for r in ranges], ["image", "image", "audio"])

    def test_default_full_span_ref_active_for_every_target(self):
        """Untimed refs (no authored frames) condition the whole timeline."""
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  refs=[{"kind": "image", "latent_h": LATENT_H,
                                         "latent_w": LATENT_W}])
        route = compute_temporal_gates(layout, frame_pattern=FRAME_PATTERN)
        expected = ((10, 10 + FRAME_ROWS, 0, 1.0),)
        for s, e, _f0, _f1, _modality in route.cells:
            self.assertEqual(_interval_keys(route, s, e), expected)
        mask = refmod_row_mask(layout, target_cell=layout.video_span,
                               frame_pattern=FRAME_PATTERN)
        self.assertTrue(any(mask[10:10 + FRAME_ROWS]))

    def test_video_audio_and_image_members_route_independently(self):
        """Each member's authored window is evaluated on its own."""
        layout = FakePackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                                  refs=[
                                      {"kind": "video_audio",
                                       "latent_t": REF_VIDEO_T,
                                       "ref_audio_t": REF_AUDIO_T,
                                       "latent_h": LATENT_H,
                                       "latent_w": LATENT_W,
                                       "start_frame": 0, "end_frame": 5},
                                      {"kind": "image", "latent_h": LATENT_H,
                                       "latent_w": LATENT_W,
                                       "start_frame": 9,
                                       "end_frame": TOTAL_FRAMES},
                                  ])
        route = compute_temporal_gates(layout, frame_pattern=FRAME_PATTERN)
        # member 0 (video_audio) rows: 10..62 as two groups; the image member
        # follows directly at 62..78 (no audio ref in this layout).
        video_groups = {(10, 14), (14, 62)}
        image_group = (62, 62 + FRAME_ROWS)
        rows = _video_cell_rows(layout)
        # cell0 frames [0,1): member 0 active ([0,5)), member 1 inactive
        keys0 = set(_key_range_triples(_interval_keys(route, rows[0][0], rows[0][1])))
        self.assertTrue(video_groups <= keys0)
        self.assertNotIn(image_group, keys0)
        # cell3 frames [9,13): member 0 window ended, member 1 active
        keys3 = set(_key_range_triples(_interval_keys(route, rows[3][0], rows[3][1])))
        self.assertFalse(video_groups & keys3)
        self.assertIn(image_group, keys3)


# ---------------------------------------------------------------------------
# Combined embedding + RefMod query partition
# ---------------------------------------------------------------------------

class TestMergeAttentionRoutes(unittest.TestCase):
    def _route(self):
        refs = [{
            "kind": "image", "latent_h": LATENT_H, "latent_w": LATENT_W,
            "start_frame": 1, "end_frame": 9, "strength": 0.5,
        }]
        layout = FakePackedLayout(
            TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T, refs=refs,
        )
        return layout, compute_temporal_gates(
            layout, refs, frame_pattern=FRAME_PATTERN,
        )

    def test_union_partition_covers_every_query_once(self):
        layout, ref_route = self._route()
        # One partial temporal-embedding key range is active only over the
        # middle two video cells. Its boundaries deliberately differ from the
        # RefMod target-cell boundaries.
        video_a, _video_b = layout.video_span
        embedding_key = (2, 4)
        intervals = merge_attention_routes(
            layout.seq_len,
            always_hidden=(embedding_key,),
            temporal_intervals=((video_a + 8, video_a + 40,
                                 ((embedding_key[0], embedding_key[1], 1.0),)),),
            refmod_route=ref_route,
        )
        cursor = 0
        for item in intervals:
            self.assertEqual(cursor, item[0])
            self.assertGreater(item[1], item[0])
            cursor = item[1]
        self.assertEqual(layout.seq_len, cursor)

    def test_target_queries_hide_inactive_refs_and_weight_active_refs(self):
        layout, ref_route = self._route()
        intervals = merge_attention_routes(
            layout.seq_len, always_hidden=(), temporal_intervals=(),
            refmod_route=ref_route,
        )
        ref_range = (TEXT_LEN, TEXT_LEN + FRAME_ROWS)
        video_cells = _video_cell_rows(layout)

        def owner(q_start):
            return next(item for item in intervals if item[0] <= q_start < item[1])

        # [0,1) is outside the authored [1,9) RefMod interval: hidden and not active.
        first = owner(video_cells[0][0])
        self.assertIn(ref_range, first[2])
        self.assertNotIn((ref_range[0], ref_range[1], 0.5), first[3])
        # [1,5) is active: the same rows are restored with strength 0.5.
        second = owner(video_cells[1][0])
        self.assertIn(ref_range, second[2])
        self.assertIn((ref_range[0], ref_range[1], 0.5), second[3])
        # Text/ref queries retain stock native RefMod visibility.
        text = owner(0)
        self.assertNotIn(ref_range, text[2])

    def test_temporal_and_refmod_keys_are_active_in_the_same_interval(self):
        layout, ref_route = self._route()
        video_a, _video_b = layout.video_span
        embedding_key = (2, 4)
        intervals = merge_attention_routes(
            layout.seq_len,
            always_hidden=(embedding_key,),
            temporal_intervals=((video_a, layout.seq_len,
                                 ((embedding_key[0], embedding_key[1], 0.75),)),),
            refmod_route=ref_route,
        )
        second_cell = _video_cell_rows(layout)[1][0]
        item = next(row for row in intervals if row[0] <= second_cell < row[1])
        self.assertIn(embedding_key, item[2])
        self.assertIn((embedding_key[0], embedding_key[1], 0.75), item[3])
        self.assertIn((TEXT_LEN, TEXT_LEN + FRAME_ROWS, 0.5), item[3])


class TestResolveRefModSpecIndex(unittest.TestCase):
    def test_identity_mapping_wins_over_stale_positional_offset(self):
        blocks = [
            {"kind": "image"},
            {
                "kind": "video",
                "longmedia_refmod_id": "hero",
                "longmedia_refmod_member_index": 0,
                "longmedia_refmod_appended_block_index": 0,
            },
        ]
        spec = {
            "ref_index": 0,
            "refmod_id": "hero",
            "member_index": 0,
            "appended_block_index": 0,
        }
        self.assertEqual(resolve_refmod_spec_index(blocks, spec), 1)

    def test_duplicate_identity_is_rejected(self):
        block = {
            "kind": "image",
            "longmedia_refmod_id": "hero",
            "longmedia_refmod_member_index": 0,
            "longmedia_refmod_appended_block_index": 0,
        }
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            resolve_refmod_spec_index([dict(block), dict(block)], {
                "refmod_id": "hero", "member_index": 0,
                "appended_block_index": 0,
            })

    def test_missing_identity_is_rejected_instead_of_gating_wrong_member(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            resolve_refmod_spec_index([{"kind": "image"}], {
                "refmod_id": "missing", "member_index": 0,
                "appended_block_index": 0,
            })

    def test_legacy_spec_can_use_validated_ref_index(self):
        self.assertEqual(
            resolve_refmod_spec_index([{"kind": "image"}], {"ref_index": 0}),
            0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
