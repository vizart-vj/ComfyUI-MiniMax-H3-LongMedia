#!/usr/bin/env python3
"""Static release audit for the three public LongMedia nodes.

Does not import ComfyUI. It verifies that INPUT_TYPES and the Python execution
method stay name-compatible, combo defaults belong to their enumerations, and
release-facing JS mentions every mode widget that must be sanitized before
ComfyUI prompt validation.
"""
from __future__ import annotations
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODES = ROOT / "nodes.py"
FACADE = ROOT / "web" / "node_facade.js"
DYNAMIC = ROOT / "web" / "long_media_dynamic_inputs.js"

PUBLIC = {
    "MiniMaxH3LongMediaPlanner": "build",
    "MiniMaxH3LatentLabLongMediaSetup": "setup",
    "MiniMaxH3LatentLabLongMediaSampler": "sample",
    "MiniMaxH3LatentLabLongMediaDecode": "decode",
}


def class_node(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def func_node(cls, name):
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def input_schema(cls):
    fn = func_node(cls, "INPUT_TYPES")
    ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
    return ast.literal_eval(ret.value)


def check_combo(name, spec):
    if not isinstance(spec, tuple) or not spec or not isinstance(spec[0], list):
        return
    values = spec[0]
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    default = opts.get("default", values[0] if values else None)
    assert default in values, f"{name}: default {default!r} not in {values!r}"


def main():
    tree = ast.parse(NODES.read_text(encoding="utf-8"))
    schemas = {}
    for cls_name, method_name in PUBLIC.items():
        cls = class_node(tree, cls_name)
        schema = input_schema(cls)
        schemas[cls_name] = schema
        inputs = list(schema.get("required", {})) + list(schema.get("optional", {}))
        method = func_node(cls, method_name)
        args = [a.arg for a in method.args.args if a.arg not in {"self", "cls"}]
        missing = [n for n in inputs if n not in args]
        compat_only = set()
        if cls_name == "MiniMaxH3LatentLabLongMediaSetup":
            compat_only = {
                "generation_mode",
                "first_frame_mode",
                "first_frame_denoise",
                "first_frame_blend_frames",
                "opening_frame",
            }
        extra = [n for n in args if n not in inputs and n not in compat_only]
        assert not missing, f"{cls_name}: schema inputs missing from {method_name}: {missing}"
        assert not extra, f"{cls_name}: {method_name} args absent from schema: {extra}"
        for group in ("required", "optional"):
            for name, spec in schema.get(group, {}).items():
                check_combo(f"{cls_name}.{name}", spec)

    planner = schemas["MiniMaxH3LongMediaPlanner"]
    setup = schemas["MiniMaxH3LatentLabLongMediaSetup"]
    sampler = schemas["MiniMaxH3LatentLabLongMediaSampler"]
    decode = schemas["MiniMaxH3LatentLabLongMediaDecode"]

    planner_names = set(planner.get("required", {})) | set(planner.get("optional", {}))
    setup_names = set(setup.get("required", {})) | set(setup.get("optional", {}))
    sampler_names = set(sampler.get("required", {}))
    decode_names = set(decode.get("required", {}))
    assert "clips_json" in planner_names
    assert "clip_plan" in setup_names
    assert {"workflow_mode", "conditioning_mode"} <= setup_names
    assert "sampler_mode" in sampler_names
    assert "video_vae" not in decode_names and "audio_vae" not in decode_names

    planner_js = (ROOT / "web" / "longmedia_planner.js").read_text(encoding="utf-8")
    assert "MiniMaxH3LongMediaPlanner" in planner_js and "+ Add Clip" in planner_js
    facade = FACADE.read_text(encoding="utf-8")
    dynamic = DYNAMIC.read_text(encoding="utf-8")
    for token in ["workflow_mode", "conditioning_mode", "sampler_mode", "lmSanitizeSetup", "lmSanitizeSampler"]:
        assert token in facade, f"node_facade.js missing {token}"
    for token in ["workflow_mode", "conditioning_mode"]:
        assert token in dynamic, f"long_media_dynamic_inputs.js missing {token} recovery"

    print("release_schema_audit: PASS")
    for cls_name, schema in schemas.items():
        print(f"  {cls_name}: {len(schema.get('required', {}))} required, {len(schema.get('optional', {}))} optional")


if __name__ == "__main__":
    main()
