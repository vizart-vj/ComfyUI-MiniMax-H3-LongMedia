from pathlib import Path
body = Path(__file__).resolve().parents[1].joinpath('nodes.py').read_text(encoding='utf-8')
assert 'class _MiniMaxH3DirectEulerRefinerSampler' in body
assert 'k_sampling.sample_euler' in body
assert 'model_sampling.noise_scaling' in body  # documented as deliberately bypassed
assert 'sampler=first_refine_sampler.out(0)' in body
assert 'sampler=refine_sampler.out(0)' in body
assert 'graph.node("DisableNoise")' in body
print('REFINER_DIRECT_EULER_REGRESSION: PASS')
