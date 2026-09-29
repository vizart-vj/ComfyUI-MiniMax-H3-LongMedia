"""Small helpers for ComfyUI's converted CFG conditioning contract."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def copy_condition_branches(original_conds: Mapping[str, Any] | None) -> dict[str, list[Any]]:
    """Copy CFG branches as ComfyUI lists, treating only optional ``None`` as empty.

    ComfyUI's ``preprocess_conds_hooks`` and ``CFGGuider.sample`` iterate every
    condition branch. A missing optional branch therefore has the canonical value
    ``[]``; silently dropping the required positive branch could produce an
    unconditioned sample, so it is rejected with a useful error instead.
    """
    if original_conds is None:
        raise ValueError("Required positive conditioning is missing; refusing to sample without its prompt.")
    if not isinstance(original_conds, Mapping):
        raise TypeError(
            "ComfyUI original conditioning must be a mapping of branch names to lists; "
            f"received {type(original_conds).__name__}."
        )
    if "positive" not in original_conds or original_conds["positive"] is None:
        raise ValueError("Required positive conditioning is missing; refusing to sample without its prompt.")

    copied: dict[str, list[Any]] = {}
    for raw_key, values in original_conds.items():
        key = str(raw_key)
        if values is None:
            if key == "positive":
                raise ValueError("Required positive conditioning is None; refusing to sample without its prompt.")
            copied[key] = []
            continue
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            raise TypeError(
                f"ComfyUI conditioning branch {key!r} must be a list or tuple, "
                f"received {type(values).__name__}."
            )
        copied[key] = [item.copy() if hasattr(item, "copy") else item for item in values]
    return copied
