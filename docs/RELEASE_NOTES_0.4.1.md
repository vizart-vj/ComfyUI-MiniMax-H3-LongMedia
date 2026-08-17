# ComfyUI-MiniMax-H3-LongMedia v0.4.1

This is a focused bugfix release for the LongMedia refiner introduced after v0.4.0.

## Refiner fix

The refiner now follows the same continuation semantics as a chained **KSampler (Advanced)** workflow. Refinement is no longer performed by taking a fully denoised latent and replaying the low-sigma tail.

When `refine_enabled=true`:

1. The connected SIGMAS schedule remains the authoritative full trajectory.
2. Stage 1 runs from the beginning to `total_steps - refine_steps` and stops before final denoise.
3. Stage 2 receives that in-progress AV latent, uses `add_noise=disable`, starts at the split step, and completes the same schedule to sigma zero.
4. The same effective seed is carried into both stages; stage 2 does not generate a new starting noise tensor.

Example: for a 12-step connected schedule and `refine_steps=3`, stage 1 performs the first 9 intervals and the refiner performs the final 3 intervals. The total trajectory remains 12 steps.

## Cleanup

Experimental refiner implementations used during debugging were removed from the release runtime. The stable v0.4.0 long-form architecture, continuity, lip-sync, OOM Governor V4 and streamed Sol paths remain unchanged.
