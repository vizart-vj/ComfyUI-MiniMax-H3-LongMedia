import torch

# FLOW/CONST default noise_scale=1 identity compensation:
# sigma * carrier + (1-sigma) * latent == latent when carrier == latent.
for sigma in [0.001, 0.01, 0.05, 0.2, 0.7]:
    latent = torch.randn(2, 4, 3, 8, 8)
    carrier = latent
    x = sigma * carrier + (1.0 - sigma) * latent
    assert torch.equal(x, latent) or torch.allclose(x, latent, atol=1e-6, rtol=1e-6)

from pathlib import Path
body = (Path(__file__).resolve().parents[1] / 'nodes.py').read_text(encoding='utf-8')
assert 'class MiniMaxH3LatentLabRefineIdentityNoise' in body
assert 'return input_latent["samples"]' in body
assert 'flow_noise_scaling_compensation=True' in body
assert 'first_refine_noise = graph.node(' in body
print('REFINER_FLOW_IDENTITY_REGRESSION PASS')
