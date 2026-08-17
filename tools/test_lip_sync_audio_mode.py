from pathlib import Path
root = Path(__file__).resolve().parents[1]
nodes = (root/'nodes.py').read_text(encoding='utf-8')
facade = (root/'web'/'node_facade.js').read_text(encoding='utf-8')
assert "lip_sync_enabled = audio_mode == 'lip_sync'" in nodes
assert "audio_mode='lip_sync' requires connected image_1 and audio_1" in nodes
assert "final_audio_override=audio_1" in nodes
assert "final_audio_track_count=1" in nodes
assert "lip_sync_native_audio_guide=True" in nodes
assert "positive = _v104_attach_native_lipsync_guide" in nodes
assert "encoded = _v104_attach_native_lipsync_guide" in nodes
assert "bool(kf.get('longmedia_lipsync_audio_guide'))" in nodes
assert "if bool(getattr(plan, 'lip_sync_native_audio_guide', False)):\n        audio_t = 0" in nodes
assert "generation_mode is legacy compatibility storage" in facade
assert 'lmSetWidgetVisible(lmWidget(node, "generation_mode"), false);' in facade
assert 'lmSetWidgetVisible(lmWidget(node, name), false);' in facade
assert 'lmSetWidgetVisible(segmentDuration, !multiclip);' in facade
assert 'const lipSync = audioMode === "lip_sync";' in facade
print('LIP_SYNC_AUDIO_MODE_REGRESSION: PASS')
