from __future__ import annotations

import logging
import re
import threading
from typing import Any

_LOG = logging.getLogger(__name__)
_PATCH_LOCK = threading.Lock()

_MINIMAX_PREFIXES = (
    "diffusion_model.blocks.0.attn.qkv_proj.weight",
    "diffusion_model.blocks.0.mlp.fc1.weight",
    "diffusion_model.blocks.0.mlp.fc2.weight",
)

_ALIAS_PREFIXES = (
    "",
    "transformer.",
    "base_model.model.",
    "unet.base_model.model.",
)

_BLOCK_RE = re.compile(r"^diffusion_model\.blocks\.(\d+)\.")


def _is_minimax_h3_state_dict(sd_keys: set[str]) -> bool:
    return all(k in sd_keys for k in _MINIMAX_PREFIXES)


def _set_alias(key_map: dict[str, Any], alias: str, target: Any) -> None:
    if alias not in key_map:
        key_map[alias] = target


def _set_prefixed_aliases(key_map: dict[str, Any], alias: str, target: Any) -> None:
    for prefix in _ALIAS_PREFIXES:
        full_alias = f"{prefix}{alias}"
        _set_alias(key_map, full_alias, target)
        _set_alias(key_map, f"lycoris_{full_alias.replace('.', '_')}", target)
        _set_alias(key_map, f"lora_transformer_{full_alias.replace('.', '_')}", target)


def _collect_block_indices(sd_keys: set[str]) -> list[int]:
    indices: set[int] = set()
    for key in sd_keys:
        match = _BLOCK_RE.match(key)
        if match is not None:
            indices.add(int(match.group(1)))
    return sorted(indices)


def _maybe_add_qkv_aliases(key_map: dict[str, Any], target: str, block_idx: int, sd: dict[str, Any]) -> None:
    tensor = sd.get(target)
    shape = getattr(tensor, "shape", None)
    if shape is None or len(shape) < 2:
        return

    out_features = int(shape[0])
    if out_features <= 0 or out_features % 3 != 0:
        _LOG.warning(
            "[MiniMaxH3 LongMedia][LoRA Compat] qkv shape for block %s is not divisible by 3: %s",
            block_idx,
            tuple(shape),
        )
        return

    span = out_features // 3
    aliases = {
        "to_q": (0, 0, span),
        "to_k": (0, span, span),
        "to_v": (0, span * 2, span),
    }
    for attn_name in ("attn", "attn1"):
        for leaf, offset in aliases.items():
            base = f"transformer_blocks.{block_idx}.{attn_name}.{leaf}"
            _set_prefixed_aliases(key_map, base, (target, offset))
            _set_prefixed_aliases(key_map, base.replace(f".{leaf}", f".processor.{leaf}"), (target, offset))


def _add_minimax_h3_lora_aliases(model: Any, key_map: dict[str, Any]) -> dict[str, Any]:
    sd = model.state_dict()
    sd_keys = set(sd.keys())
    if not _is_minimax_h3_state_dict(sd_keys):
        return key_map

    block_indices = _collect_block_indices(sd_keys)

    # Generic aliases for native MiniMax names with transformer/base_model prefixes.
    for key in sd_keys:
        if not key.startswith("diffusion_model.") or not key.endswith(".weight"):
            continue
        inner = key[len("diffusion_model.") : -len(".weight")]
        _set_prefixed_aliases(key_map, inner, key)

    for block_idx in block_indices:
        block_prefix = f"diffusion_model.blocks.{block_idx}"

        qkv = f"{block_prefix}.attn.qkv_proj.weight"
        if qkv in sd_keys:
            _maybe_add_qkv_aliases(key_map, qkv, block_idx, sd)

        out_proj = f"{block_prefix}.attn.out_proj.weight"
        if out_proj in sd_keys:
            for attn_name in ("attn", "attn1"):
                _set_prefixed_aliases(key_map, f"transformer_blocks.{block_idx}.{attn_name}.to_out.0", out_proj)
                _set_prefixed_aliases(key_map, f"transformer_blocks.{block_idx}.{attn_name}.to_out", out_proj)

        fc1 = f"{block_prefix}.mlp.fc1.weight"
        if fc1 in sd_keys:
            _set_prefixed_aliases(key_map, f"transformer_blocks.{block_idx}.ff.net.0.proj", fc1)
            _set_prefixed_aliases(key_map, f"transformer_blocks.{block_idx}.ff.net.0", fc1)

        fc2 = f"{block_prefix}.mlp.fc2.weight"
        if fc2 in sd_keys:
            _set_prefixed_aliases(key_map, f"transformer_blocks.{block_idx}.ff.net.2", fc2)

        adaln = f"{block_prefix}.adaln_proj.linear.weight"
        if adaln in sd_keys:
            _set_prefixed_aliases(key_map, f"transformer_blocks.{block_idx}.adaln_proj.linear", adaln)

    final_prefix = "diffusion_model.final_layer"
    final_aliases = {
        f"{final_prefix}.adaln_proj.linear.weight": "proj_out.norm_out.linear",
        f"{final_prefix}.video_out.weight": "proj_out.video_out",
        f"{final_prefix}.audio_out.weight": "proj_out.audio_out",
    }
    for target, alias in final_aliases.items():
        if target in sd_keys:
            _set_prefixed_aliases(key_map, alias, target)

    _LOG.debug(
        "[MiniMaxH3 LongMedia][LoRA Compat] MiniMax H3 aliases active; blocks=%s",
        len(block_indices),
    )
    return key_map


def patch_comfy_lora() -> None:
    with _PATCH_LOCK:
        try:
            import comfy.lora as comfy_lora  # type: ignore
        except Exception:
            _LOG.debug("[MiniMaxH3 LongMedia][LoRA Compat] comfy.lora unavailable at import time")
            return

        current = getattr(comfy_lora, "model_lora_keys_unet", None)
        if current is None:
            return
        if getattr(current, "_minimax_h3_longmedia_patched", False):
            return

        original = current

        def wrapped(model: Any, key_map: dict[str, Any] = {}):
            result = original(model, key_map)
            try:
                return _add_minimax_h3_lora_aliases(model, result)
            except Exception:
                _LOG.exception("[MiniMaxH3 LongMedia][LoRA Compat] failed to extend MiniMax H3 LoRA key map")
                return result

        wrapped._minimax_h3_longmedia_patched = True  # type: ignore[attr-defined]
        wrapped._minimax_h3_longmedia_original = original  # type: ignore[attr-defined]
        comfy_lora.model_lora_keys_unet = wrapped
        _LOG.debug("[MiniMaxH3 LongMedia][LoRA Compat] patched comfy.lora.model_lora_keys_unet")


patch_comfy_lora()
