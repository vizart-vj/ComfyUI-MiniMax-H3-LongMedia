from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
text = (root / 'nodes.py').read_text(encoding='utf-8')

checks = {
    'native_overlap_5': "multiclip_native_overlap = 5" in text,
    'single_decode_mode': "single_native_continuous_video_vae_decode" in text,
    'latent_strip': "contribution = clip_video[:, :, overlap_video_t:].contiguous()" in text,
    'one_cat': "continuous_video = torch.cat(video_parts, dim=2)" in text,
    'native_grid_validation': "frame_count_from_video_t(int(continuous_video.shape[2]))" in text,
    'no_rgb_seam_processing': "'rgb_seam_processing': False" in text,
    'no_hidden_preroll_runtime': "local_lengths[segment_index] = int(local_lengths[segment_index]) + int(hidden_overlap)" not in text,
}
failed = [k for k,v in checks.items() if not v]
if failed:
    raise SystemExit('MULTICLIP_NATIVE_CONTINUOUS_STATIC FAILED: ' + ', '.join(failed))

# Contract sanity for the common 124f H3 clip: 124f -> T37, native 5f overlap -> T2.
# 3 clips: 37 + (37-2) + (37-2) = T107 -> 362f = 124 + 119 + 119.
def t(frames):
    assert (frames - 5) % 17 == 0
    return 2 + 5 * ((frames - 5)//17)
def frames(tt):
    assert (tt - 2) % 5 == 0
    return 5 + 17 * ((tt - 2)//5)
assembled = t(124) + 2*(t(124)-t(5))
assert assembled == 107
assert frames(assembled) == 362
print('MULTICLIP_NATIVE_CONTINUOUS_STATIC: PASS')
