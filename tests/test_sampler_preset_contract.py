"""Keep the frontend preset schema aligned with Sampler INPUT_TYPES."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOCKETS_AND_EXCLUDED_WIDGETS = {
    "initial_av",
    "long_media_plan",
    "guider",
    "sampler",
    "sigmas",
    "seed",
    "refine_add_noise",
    "refine_seed",
    "refine_steps",
}


def sampler_required_fields(source: str) -> set[str]:
    tree = ast.parse(source)
    for item in tree.body:
        if not isinstance(item, ast.ClassDef) or item.name != "MiniMaxH3LatentLabLongMediaSampler":
            continue
        for method in item.body:
            if not isinstance(method, ast.FunctionDef) or method.name != "INPUT_TYPES":
                continue
            for node in ast.walk(method):
                if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Dict):
                    continue
                for key, value in zip(node.value.keys, node.value.values):
                    if not isinstance(key, ast.Constant) or key.value != "required" or not isinstance(value, ast.Dict):
                        continue
                    return {
                        child.value
                        for child in value.keys
                        if isinstance(child, ast.Constant) and isinstance(child.value, str)
                    }
    raise AssertionError("Could not find Sampler INPUT_TYPES required fields")


class SamplerPresetContractTests(unittest.TestCase):
    def test_presets_cover_all_active_widgets_but_not_seed_or_sockets(self) -> None:
        python_fields = sampler_required_fields((ROOT / "nodes.py").read_text(encoding="utf-8"))
        source = (ROOT / "web" / "node_facade.js").read_text(encoding="utf-8")
        match = re.search(r"const LM_SAMPLER_PRESET_FIELDS = Object\.freeze\(\[(.*?)\]\);", source, re.S)
        self.assertIsNotNone(match, "Sampler preset field list must remain explicit")
        assert match is not None
        preset_list = re.findall(r"'([^']+)'", match.group(1))
        preset_fields = set(preset_list)
        self.assertEqual(len(preset_list), len(preset_fields), "Sampler preset field list must not contain duplicates")
        self.assertNotIn("seed", preset_fields, "seed remains independent for each render")
        self.assertEqual(preset_fields, python_fields - SOCKETS_AND_EXCLUDED_WIDGETS)
        self.assertEqual(len(preset_fields), 31)


if __name__ == "__main__":
    unittest.main(verbosity=2)
