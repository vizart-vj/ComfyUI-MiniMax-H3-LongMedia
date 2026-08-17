from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
body = (ROOT / 'nodes.py').read_text(encoding='utf-8')
assert '[0.4.2 ADDITIVE FINAL-LATENT REFINE]' in body
assert 'MiniMaxH3LatentLabRefineFinalLatentInput' in body
assert 'final_latent=first_main_output' in body
assert 'final_latent=sampled_output' in body
assert 'latent_image=first_refine_input.out(0)' in body
assert 'latent_image=refine_input.out(0)' in body
assert 'MiniMaxH3LatentLabRefineIdentityNoise' in body
assert 'flow_noise_scaling_compensation=True' in body
assert 'sigmas=refine_sigmas' in body
assert '"advanced_refine_latent_source": "exact_base_sampler_output"' in body
assert '"advanced_refine_renoise": False' in body
assert 'full_base_schedule_plus_additive_low_noise_tail' in body
print('FINAL_LATENT_TAIL_REFINE_REGRESSION: PASS')
