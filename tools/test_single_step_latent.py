#!/usr/bin/env python3
"""Torch-free regression for the v0.3.30 one-step keyframe extractor."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "latent_ops.py"


class FakeTensor:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.ndim = len(shape)
        self.is_nested = False

    def __getitem__(self, key):
        assert len(key) == 3
        time_slice = key[2]
        assert isinstance(time_slice, slice)
        assert time_slice.start is None and time_slice.stop == 1
        return FakeTensor((self.shape[0], self.shape[1], 1, *self.shape[3:]))


def load_extractor():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_leading_video_step_from_latent"
    )
    namespace = {
        "torch": SimpleNamespace(Tensor=object),
        "unpack_av_samples": lambda _latent: (_ for _ in ()).throw(
            AssertionError("plain T=1 keyframe must not enter AV/full-video validation")
        ),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE_PATH), "exec"),
        namespace,
    )
    return namespace["_leading_video_step_from_latent"]


def main() -> None:
    extractor = load_extractor()

    single = FakeTensor((1, 24, 1, 48, 36))
    result = extractor({"samples": single})
    assert result.shape == (1, 24, 1, 48, 36)

    full = FakeTensor((1, 24, 7, 48, 36))
    result = extractor({"samples": full})
    assert result.shape == (1, 24, 1, 48, 36)

    try:
        extractor({"samples": FakeTensor((1, 16, 1, 48, 36))})
    except ValueError as exc:
        assert "[B, 24, T>=1, H, W]" in str(exc)
    else:
        raise AssertionError("wrong channel count must be rejected")

    try:
        extractor({"samples": object()})
    except ValueError as exc:
        assert "got object" in str(exc)
    else:
        raise AssertionError("non-tensor payload must be rejected cleanly")

    source = SOURCE_PATH.read_text(encoding="utf-8")
    inject_body = source.split("def inject_leading_video_frame", 1)[1].split(
        "\ndef ", 1
    )[0]
    assert "_leading_video_step_from_latent(frame_latent)" in inject_body
    assert "_stream_from_latent(frame_latent" not in inject_body
    print("SINGLE_STEP_LATENT_REGRESSION: PASS")


if __name__ == "__main__":
    main()
