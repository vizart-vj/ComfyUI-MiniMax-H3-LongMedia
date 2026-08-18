from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nodes = (ROOT / "nodes.py").read_text(encoding="utf-8")
facade = (ROOT / "web" / "node_facade.js").read_text(encoding="utf-8")
assert (ROOT / "VERSION").read_text().strip() == "0.4.11"
assert '__version__ = "0.4.11"' in (ROOT / "__init__.py").read_text(encoding="utf-8")
assert 'version = "0.4.11"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
assert "segmentation_active = workflow_mode in ('manual', 'segmented_continuation')" in nodes
assert "Unexpected extra LongMedia pass outside segmentation/MultiClip" in nodes
assert "[0.4.7 REFINER]" in nodes
assert "refine_noise = torch.zeros_like" in nodes
assert "same_model_lifecycle=True" in nodes
assert "[0.4.11 TE PINNED-MEMORY GATE]" in nodes
assert "_clone_model_options_safe" in nodes
assert "copy.deepcopy(getattr(guider, 'model_options'" not in nodes
assert 'lmSetWidgetVisible(segmentDuration, manual || segmented);' in facade
assert 'lmSetWidgetVisible(lmWidget(node, "overlap_frames"), manual || segmented);' in facade
print("RELEASE_0411_REGRESSION: PASS")
