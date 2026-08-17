"""Pure sigma-schedule policy for LongMedia's two-stage refiner.

The refiner is a continuation of one diffusion trajectory, not an extra pass
appended after a complete denoise.  `total_steps` comes from the connected
SIGMAS tensor (N+1 sigma points for N denoise intervals).  The schedule is
split at `total_steps - refine_steps` and the boundary sigma is shared by the
two slices without duplicating a denoise interval.
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

    # Keep at least one interval in the high-noise/main stage.  A zero-step
    # main stage would never inject the requested initial noise before a
    # DisableNoise refiner, which is not a valid two-stage trajectory.
    effective = min(requested, max(1, total_steps - 1)) if total_steps > 1 else 0
    if effective <= 0:
        # Degenerate 1-step scheduler: keep the full schedule in main and make
        # refine a zero-interval boundary slice.  Normal H3 schedules are >1.
        switch_step = total_steps
        main = src.clone()
        refine = src[-1:].clone()
    else:
        switch_step = total_steps - effective
        main = src[: switch_step + 1].clone()
        refine = src[switch_step:].clone()

    return main, refine, total_steps, switch_step, effective, requested
