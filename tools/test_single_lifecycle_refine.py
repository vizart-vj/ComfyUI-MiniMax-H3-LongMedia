from pathlib import Path

src = (Path(__file__).resolve().parents[1] / 'nodes.py').read_text(encoding='utf-8')
start = src.index('class MiniMaxH3LatentLabLongMediaSampler:')
end = src.index('NODE_CLASS_MAPPINGS', start)
body = src[start:end]

assert '[0.3.58 SINGLE-LIFECYCLE REFINE]' in body
assert 'refine_sigmas' not in body
assert 'refine_noise' not in body
assert 'first_refine = graph.node' not in body
assert 'refined = graph.node' not in body
assert 'sigmas=main_sigmas' in body
assert 'main_sigmas = sigmas' in body
assert 'MiniMaxH3LatentLabProtectRefineAV' in body
print('SINGLE_LIFECYCLE_REFINE_REGRESSION: PASS')
