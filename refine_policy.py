"""Sigma policy for LongMedia refiner using true KSampler Advanced split semantics.

The connected SIGMAS schedule is treated as the FULL schedule. When refinement
is enabled, the primary sampler stops early and returns the in-progress latent
with leftover noise. The refiner then continues the SAME schedule without adding
new noise, exactly like chaining two KSampler (Advanced) nodes.

For T total steps and R refine steps::

    main_sigmas   = sigmas[:T-R+1]   # first T-R intervals
    refine_start  = T-R              # continue from this global step
    full_sigmas   = sigmas           # refiner sees the full schedule

This means the refiner is not a tail replay on a finished x0. It is the second
stage of one continuous denoising trajectory.
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

    # First sampler stops early and returns the partially denoised state.
    main = src[:switch_step + 1].clone()
    # Refiner continues the full connected schedule from switch_step onward.
    refine = src.clone()

    return main, refine, total_steps, switch_step, effective, requested
