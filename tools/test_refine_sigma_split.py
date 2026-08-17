from pathlib import Path
import importlib.util
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('refine_policy', ROOT / 'refine_policy.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def check(total, refine):
    # Values themselves are irrelevant; indices make boundary/interval accounting explicit.
    sigmas = torch.arange(total + 1, 0, -1, dtype=torch.float32)
    main, low, total_steps, switch, effective, requested = mod.split_refine_sigmas(sigmas, refine)
    assert total_steps == total
    assert switch + effective == total
    assert len(main) - 1 == switch
    assert len(low) - 1 == effective
    assert main[-1].item() == low[0].item()  # shared boundary sigma only
    assert (len(main)-1) + (len(low)-1) == total
    return switch, effective

assert check(12, 3) == (9, 3)
assert check(8, 2) == (6, 2)
assert check(20, 5) == (15, 5)
assert check(12, 99) == (1, 11)  # retain one valid noisy main interval
print('REFINE_SIGMA_SPLIT_REGRESSION: PASS')
