from pathlib import Path
s = (Path(__file__).resolve().parents[1] / "nodes.py").read_text()
assert "[0.3.108 AUTHORITATIVE LOCAL-0 LIP SYNC]" in s
assert "lip_sync_native_audio_guide=True" in s
assert "lip_sync_audio=(audio_1 if lip_sync_enabled else None), audio_vae=audio_vae" in s
assert "[0.3.108 VIDEO MOTION CONTEXT + SOURCE AUDIO CLOCK]" in s
assert "longmedia_native_motion_audio" in s
assert "bool(kf.get('longmedia_lipsync_audio_guide'))" in s
assert "if bool(getattr(plan, 'lip_sync_native_audio_guide', False)):\n        audio_t = 0" in s
assert "positive = _v104_attach_native_lipsync_guide" in s
assert "encoded = _v104_attach_native_lipsync_guide" in s
print("LIP_SYNC_AUTHORITATIVE_SOURCE_CLOCK_REGRESSION PASS")
