"""Sigma policy for LongMedia two-stage refiner inside one model lifecycle.

The connected SIGMAS schedule is split into a base stage and a low-noise refine
stage. Both stages are executed by the unified runtime while the same H3 model,
guider wrappers and sampler lifecycle stay resident.

For T total steps and R refine steps::

    main_sigmas   = sigmas[:T-R+1]
    refine_sigmas = sigmas[T-R:]

Stage 1 uses the normal random noise. Stage 2 uses zero noise with the same
effective seed and continues from the stage-1 solver state.
"""
from __future__ import annotations
import torch


def split_refine_sigmas(sigmas, refine_steps: int):
    src = sigmas.detach() if torch.is_tensor(sigmas) else torch.as_tensor(sigmas, dtype=torch.float32)
    src = src.flatten()
    if int(src.numel()) < 2:
        raise ValueError("Refine requires at least two sigma points.")
    total_steps = int(src.numel()) - 1
    requested = max(1, int(refine_steps))
    effective = min(requested, total_steps)
    switch_step = max(0, total_steps - effective)
    main = src[:switch_step + 1].clone()
    refine = src[switch_step:].clone()
    return main, refine, total_steps, switch_step, effective, requested
