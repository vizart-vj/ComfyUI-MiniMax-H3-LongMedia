from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
assert (ROOT / "VERSION").read_text().strip() == "0.4.30"
assert '__version__ = "0.4.30"' in (ROOT / "__init__.py").read_text(encoding="utf-8")
assert 'version = "0.4.30"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
nodes = (ROOT / "nodes.py").read_text(encoding="utf-8")
plan = (ROOT / "media_plan.py").read_text(encoding="utf-8")
assert "release_guard" not in nodes
assert "release_guard" not in plan
assert "_set_longmedia_release_guard" not in nodes
assert "single_native_continuous_video_vae_decode" in nodes
assert "frame_count_from_video_t(int(continuous_video.shape[2]))" in nodes
print("RELEASE_0430: PASS")
