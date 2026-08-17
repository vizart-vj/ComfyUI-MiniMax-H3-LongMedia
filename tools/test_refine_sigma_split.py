from pathlib import Path
import importlib.util
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('refine_policy', ROOT / 'refine_policy.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def check(total, refine):
    sigmas = torch.arange(total, -1, -1, dtype=torch.float32)
    main, low, base_steps, tail_start, effective, requested = mod.split_refine_sigmas(sigmas, refine)
    assert base_steps == total
    assert requested == max(1, refine)
    assert effective == min(max(1, refine), total)
    assert torch.equal(main, sigmas), (main, sigmas)
    assert torch.equal(low, sigmas[-(effective + 1):])
    assert tail_start == total - effective
    assert main.numel() - 1 == total
    assert low.numel() - 1 == effective
    assert total + effective == (main.numel() - 1) + (low.numel() - 1)

check(12, 3)
check(8, 2)
check(4, 99)
print('REFINE_SIGMA_SPLIT_REGRESSION: PASS')
