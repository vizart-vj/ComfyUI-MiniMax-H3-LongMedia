#!/usr/bin/env python3
"""Regression tests for the v0.3.29 timeline/reference contract."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from continuity_policy import build_segment_prompt, normalize_hybrid_picture_tags

# media_plan's planner is pure, but the runtime module also contains optional
# tensor slicing helpers. Keep this regression runnable outside a ComfyUI/torch
# environment by providing only the import placeholder those unused helpers need.
if importlib.util.find_spec("torch") is None:
    sys.modules["torch"] = types.ModuleType("torch")
from media_plan import build_media_plan


PROMPT = """A man in a WWII-era German military uniform walks through the snow-covered forest. Preserve the man's face, hairstyle, age, proportions, and uniform identity from <Picture 2>. Девушка встречает мужчину в форме в холодном лесу. Preserve the woman's face, hairstyle, age, proportions, and uniform identity from <Picture 3>. Start from the supplied first frame composition and integrate the man naturally into that environment.
Сохрани естественное освещение на всех персонажах из <Picture 1>
00:A man in a WWII-era German military uniform walks through the snow-covered forest.
03 sec: девушка встречает мужчину в форме в холодном лесу.
07 sec: мужчина и девушка целуются страстно трогая друг друга.

Камера: сопровождает идущего мужчину, ручная тряска камеры, DV.

Музыка: тревожные нотки военного фильма
Звук: шаги мужчины по снегу"""


def scoped(prompt: str, index: int) -> tuple[str, dict]:
    return build_segment_prompt(
        prompt,
        segment_index=index,
        segment_starts=(0, 102),
        segment_lengths=(124, 141),
        overlap_frames=22,
        passes=2,
        fps=24.0,
    )


def test_supplied_workflow_case() -> None:
    plan = build_media_plan(
        audios=[],
        videos=[],
        manual_duration=10.0,
        duration_source="manual",
        segment_seconds=5.0,
        overlap_frames=22,
        video_fps=24.0,
        resolution_mode="match",
    )
    assert plan.output_frames == 240
    assert plan.segment_lengths == (124, 141)
    assert plan.segment_starts == (0, 102)
    assert plan.overlap_frames == 22

    normalized, mapping = normalize_hybrid_picture_tags(
        PROMPT,
        anchor_roles=("opening frame",),
        reference_count=2,
    )
    assert mapping["mode"] == "input_socket_compat"
    assert mapping["mapping"] == {
        "1": "the supplied opening frame",
        "2": "<Picture 1>",
        "3": "<Picture 2>",
    }
    assert "<Picture 3>" not in normalized
    assert "identity from <Picture 1>" in normalized
    assert "identity from <Picture 2>" in normalized
    assert "освещение на всех персонажах из the supplied opening frame" in normalized

    pass0, report0 = scoped(normalized, 0)
    assert report0["visible_start_frame"] == 0
    assert report0["visible_end_frame"] == 124
    assert report0["events_selected"] == 2
    assert "00 sec: A man" in pass0
    assert "03 sec: девушка" in pass0
    assert "целуются страстно" not in pass0

    pass1, report1 = scoped(normalized, 1)
    assert report1["visible_start_frame"] == 124
    assert report1["visible_end_frame"] == 243
    assert report1["events_selected"] == 1
    assert "1.83 sec: мужчина и девушка целуются" in pass1
    assert "00 sec: A man" not in pass1
    assert "03 sec: девушка" not in pass1
    assert "00 sec: A man" not in pass1
    assert "By 00 sec globally: A man" in pass1
    assert "Start from the supplied first frame" not in pass1
    assert "Preserve the man's face" in pass1
    assert "Preserve the woman's face" in pass1
    assert "Сохрани естественное освещение" in pass1
    assert "Камера:" in pass1


def test_native_tags_are_not_guessed() -> None:
    prompt = "Keep <Picture 1> and <Picture 2> identities."
    normalized, mapping = normalize_hybrid_picture_tags(
        prompt,
        anchor_roles=("opening frame",),
        reference_count=2,
    )
    assert normalized == prompt
    assert mapping["mode"] == "native_reference_order"
    assert mapping["invalid_tags"] == []


def test_explicit_local_sections_win() -> None:
    prompt = (
        "Opening local instructions.\n"
        "Continue directly from the preceding video segment. Second local instructions."
    )
    first, first_report = scoped(prompt, 0)
    second, second_report = scoped(prompt, 1)
    assert first == "Opening local instructions."
    assert second.startswith("Continue directly from the preceding video segment.")
    assert first_report["mode"] == "explicit_local_section"
    assert second_report["mode"] == "explicit_local_section"


def test_source_topology_invariants() -> None:
    source_path = ROOT / "nodes.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_v61_build_identity_sheet"
    ]
    assert calls == [], "v0.3.29 must never collapse distinct refs into one sheet"
    assert "[V329 STABLE NATIVE REFS]" in source
    assert "prompt=segment0_prompt" in source
    assert ("first_frame_latent_injected=first_frame_latent_injected" in source or "first_frame_latent_injected=(False if lip_sync_enabled else first_frame_latent_injected)" in source)


def test_native_ref_payload_is_stable() -> None:
    """Execute the two small backend helpers without importing ComfyUI/torch."""

    source_path = ROOT / "nodes.py"
    source_tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    wanted = {
        "_v329_terminal_keyframes_for_segment",
        "_v329_encode_continuation_native_refs",
    }
    definitions = [
        node for node in source_tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {"_conditioning_meta": lambda entry: entry[1]}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source_path), "exec"), namespace)

    helper_module = types.ModuleType("node_helpers")

    def conditioning_set_values(encoded, values):
        return [[entry[0], {**entry[1], **values}] for entry in encoded]

    helper_module.conditioning_set_values = conditioning_set_values
    sys.modules["node_helpers"] = helper_module

    ref_item_1, ref_item_2 = object(), object()
    ref_block_1, ref_block_2 = object(), object()
    calls = []

    class Clip:
        def tokenize(self, prompt, *, minimax_ref_items):
            calls.append((prompt, minimax_ref_items))
            return "tokens"

        def encode(self, tokens):
            assert tokens == "tokens"
            return [["embedding", {}]]

    source_terminal = {
        "resolved_frame_index": 123,
        "motion_context_index": 123,
        "latent": object(),
    }
    positive = [["source", {
        "minimax_refs": [ref_block_1, ref_block_2],
        "minimax_keyframes": [
            {"resolved_frame_index": 0, "latent": object()},
            source_terminal,
        ],
        "minimax_visual_cond_noise_aug": 0.25,
    }]]
    plan = SimpleNamespace(passes=2, segment_lengths=(124, 141))
    encoded = namespace["_v329_encode_continuation_native_refs"](
        Clip(),
        "pass 2",
        positive,
        plan,
        1,
        (ref_item_1, ref_item_2),
        (ref_block_1, ref_block_2),
    )
    assert calls == [("pass 2", [ref_item_1, ref_item_2])]
    metadata = encoded[0][1]
    assert metadata["minimax_refs"] == [ref_block_1, ref_block_2]
    assert metadata["minimax_keyframes"][0]["resolved_frame_index"] == 140
    assert metadata["minimax_keyframes"][0]["motion_context_index"] == 140
    assert metadata["minimax_frame_count"] == 141
    assert source_terminal["resolved_frame_index"] == 123




def test_v331_four_pass_state_carry() -> None:
    """Completed events become state facts on every continuation pass."""

    prompt = """Preserve the man's identity from <Picture 1>. Preserve the woman's identity from <Picture 2>.
00 sec: The man walks alone through the forest.
03 sec: The man meets the woman.
07 sec: They take each other by the hand.
10 sec: They walk together along the forest path.
14 sec: They continue walking together deeper into the forest.

Camera: continuous follow shot from behind and slightly from the side.
Style: realistic cinematic live-action, no cuts."""
    starts = (0, 102, 204, 306)
    lengths = (124, 124, 124, 124)
    outputs = []
    reports = []
    for idx in range(4):
        out, report = build_segment_prompt(
            prompt,
            segment_index=idx,
            segment_starts=starts,
            segment_lengths=lengths,
            overlap_frames=22,
            passes=4,
            fps=24.0,
        )
        outputs.append(out)
        reports.append(report)

    assert "Continuation contract:" not in outputs[0]
    for idx in (1, 2, 3):
        assert "Continuation contract:" in outputs[idx]
        assert "already completed; do not replay or restart them" in outputs[idx]
        assert "signed direction of travel" in outputs[idx]
        assert "isolated hero close-up" in outputs[idx]

    # Old events become facts, not local timestamp commands.
    assert "00 sec: The man walks alone" not in outputs[1]
    assert "By 00 sec globally: The man walks alone" in outputs[1]
    assert "By 07 sec globally: They take each other by the hand" in outputs[2]
    assert "By 10 sec globally: They walk together" in outputs[3]

    # Prompt growth is bounded to the most recent four completed events.
    assert reports[3]["completed_state_events"] <= 4


def test_v331_source_seam_invariants() -> None:
    source = (ROOT / "nodes.py").read_text(encoding="utf-8")
    assert "blend_video_overlap=False,  # continuity policy: hidden frozen overlap is context only; never re-blend it" in source
    assert "merged_keyframes.sort(" in source
    assert '"hidden_overlap": "exact_context_trim_no_blend"' in source



def test_v332_native_av_context_contract() -> None:
    source = (ROOT / "nodes.py").read_text(encoding="utf-8")
    assert "def _v0322_attach_native_av_context_ref" in source
    assert "'kind': 'video_audio'" in source
    assert "'audio_latent': audio_tail" in source
    assert "longmedia_audio_grid_offset" in source

    # Audio continuity is now owned by native paired AV state, not extra prompt pressure.
    prompt = "00 sec: A man walks.\n07 sec: He continues walking.\n"
    compiled, _ = build_segment_prompt(
        prompt, segment_index=1, passes=2, segment_starts=(0, 102),
        segment_lengths=(124, 124), overlap_frames=22, fps=24.0,
    )
    assert "audio continuity contract" not in compiled.lower()

if __name__ == "__main__":
    test_supplied_workflow_case()
    test_native_tags_are_not_guessed()
    test_explicit_local_sections_win()
    test_source_topology_invariants()
    test_native_ref_payload_is_stable()
    test_v331_four_pass_state_carry()
    test_v331_source_seam_invariants()
    test_v332_native_av_context_contract()
    print("CONTINUITY_POLICY_REGRESSION: PASS")
