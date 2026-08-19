from pathlib import Path
text=(Path(__file__).resolve().parents[1]/"nodes.py").read_text(encoding="utf-8")
start=text.index("    def decode(self, final_av")
body=text[start:]
assert "single_native_continuous_video_vae_decode" in body
assert "rgb_seam_processing': False" in body
# Helpers may remain for backward/source compatibility, but the active decode path must not call them.
branch=body[body.index("        if use_per_clip_native_video_decode:"):body.index("        storyboard_duplicate_removed = False") ]
assert "_multiclip_temporal_seam_index(" not in branch
assert "_match_leading_frames_photometrically(" not in branch
assert "_suppress_boundary_photometric_outliers(" not in branch
print("MULTICLIP_RGB_SEAM_RETIRED_STATIC: PASS")
