from __future__ import annotations

"""Compatibility for Kijai/FastVideo MiniMax-H3 VSA checkpoints.

This is deliberately separate from ``fasth3_vsa_compat``.  Kijai's
``minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot`` is a normal
Comfy-layout MiniMax-H3 transformer plus one learned ``to_gate_compress``
projection in every DiT attention block.  It does *not* use H3ddle's
input-major core, serialized seven-row AdaLN tables, or ``h3.fasth3.*`` package
metadata.

Current upstream ComfyUI PR #15958 creates these 50 Linear modules at model
construction time.  Until that lands in the user's ComfyUI, LongMedia creates
exactly those modules immediately before ``MiniMaxH3.load_model_weights`` and
lets Comfy's own mixed-precision/dynamic loader own the quantized weights.
"""

import torch

BLOCKS = 50
HIDDEN = 5376
HEAD_DIM = 128
HEADS = 56
INNER = HEADS * HEAD_DIM  # 7168
GATE_SUFFIX = "attn.to_gate_compress"
FASTH3_MARKER = "h3.fasth3.version"


def _shape(value):
    try:
        return tuple(int(x) for x in value.shape)
    except Exception:
        return None


def _gate_prefixes(sd, prefix=""):
    return [f"{prefix}blocks.{i}.{GATE_SUFFIX}" for i in range(BLOCKS)]


def classify_fastvideo_vsa(sd, prefix=""):
    """Return a strict Kijai/FastVideo VSA contract or ``None``.

    Detection is intentionally fail-closed and prefix-local.  A partial set of
    gates is treated as a malformed VSA checkpoint, not as stock H3.  H3ddle
    FastH3 packages are excluded explicitly so the two runtimes cannot overlap.
    """
    if f"{prefix}{FASTH3_MARKER}" in sd:
        return None

    gates = _gate_prefixes(sd, prefix)
    present = [f"{g}.weight" in sd for g in gates]
    if not any(present):
        return None
    if not all(present):
        missing = [i for i, ok in enumerate(present) if not ok]
        raise RuntimeError(
            "[LongMedia][FastVideo VSA PRECHECK] partial to_gate_compress checkpoint: "
            f"missing block(s) {missing[:8]}{'...' if len(missing) > 8 else ''}"
        )

    # This exact experimental Kijai file is INT8 ConvRot and carries the
    # companion scale/quant descriptors for every gate.  Requiring all three
    # prevents silently accepting a half-converted checkpoint.
    missing_companions = []
    for i, gate in enumerate(gates):
        for suffix in ("weight_scale", "comfy_quant"):
            if f"{gate}.{suffix}" not in sd:
                missing_companions.append(f"{i}:{suffix}")
    if missing_companions:
        raise RuntimeError(
            "[LongMedia][FastVideo VSA PRECHECK] gate quantization payload is incomplete: "
            + ", ".join(missing_companions[:12])
            + ("..." if len(missing_companions) > 12 else "")
        )

    bad = []
    for i, gate in enumerate(gates):
        got = _shape(sd[f"{gate}.weight"])
        if got != (INNER, HIDDEN):
            bad.append((i, got))
    if bad:
        detail = ", ".join(f"block{i}={shape}" for i, shape in bad[:6])
        raise RuntimeError(
            "[LongMedia][FastVideo VSA PRECHECK] to_gate_compress shape mismatch; "
            f"expected {(INNER, HIDDEN)} for all {BLOCKS} blocks, got {detail}"
        )

    return {
        "family": "kijai_fastvideo_vsa",
        "blocks": BLOCKS,
        "hidden": HIDDEN,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "inner": INNER,
        "tile_size": 64,
        "sparsity": 0.90,
        "topk_ratio": 0.10,
        "transformer_forwards": 4,
        "sigma_shift_video": 12.0,
        "sigma_shift_audio": 3.0,
        "task": "t2av",
    }


def _linear_dtype(reference):
    dtype = getattr(reference, "weight_comfy_model_dtype", None)
    if dtype is not None:
        return dtype
    weight = getattr(reference, "weight", None)
    if weight is not None:
        return getattr(weight, "dtype", None)
    return None


def _new_gate_like(attn):
    qkv = attn.qkv_proj
    hidden = int(getattr(qkv, "in_features", HIDDEN))
    heads = int(getattr(attn, "heads", HEADS))
    head_dim = int(getattr(attn, "head_dim", HEAD_DIM))
    inner = heads * head_dim
    if hidden != HIDDEN or inner != INNER:
        raise RuntimeError(
            "[LongMedia][FastVideo VSA PRECHECK] loaded MiniMax-H3 architecture does not match "
            f"the VSA student: hidden={hidden}, inner={inner}, expected={HIDDEN}/{INNER}"
        )

    linear_cls = type(qkv)
    dtype = _linear_dtype(qkv)
    try:
        gate = linear_cls(hidden, inner, bias=False, device=None, dtype=dtype)
    except TypeError:
        gate = linear_cls(hidden, inner, bias=False)
    gate._longmedia_fastvideo_vsa_owned = True
    return gate


def _clear_runtime(base_model):
    diffusion = getattr(base_model, "diffusion_model", None)
    if diffusion is None:
        return {"gates": 0, "contracts": 0}
    removed = 0
    for block in list(getattr(diffusion, "blocks", ())):
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        gate = getattr(attn, "to_gate_compress", None)
        if gate is not None and bool(getattr(gate, "_longmedia_fastvideo_vsa_owned", False)):
            try:
                delattr(attn, "to_gate_compress")
                removed += 1
            except Exception:
                pass
    cleared = 0
    for owner in (base_model, diffusion):
        if hasattr(owner, "_longmedia_fastvideo_vsa_contract"):
            try:
                delattr(owner, "_longmedia_fastvideo_vsa_contract")
                cleared += 1
            except Exception:
                pass
    return {"gates": removed, "contracts": cleared}


def _prepare_modules(base_model, contract):
    diffusion = getattr(base_model, "diffusion_model", None)
    if diffusion is None:
        raise RuntimeError("[LongMedia][FastVideo VSA PRECHECK] diffusion_model is unavailable")
    blocks = list(getattr(diffusion, "blocks", ()))
    if len(blocks) != BLOCKS:
        raise RuntimeError(
            f"[LongMedia][FastVideo VSA PRECHECK] expected {BLOCKS} H3 blocks, got {len(blocks)}"
        )

    created = 0
    native = 0
    for i, block in enumerate(blocks):
        attn = block.attn
        gate = getattr(attn, "to_gate_compress", None)
        if gate is None:
            gate = _new_gate_like(attn)
            attn.add_module("to_gate_compress", gate)
            created += 1
        else:
            native += 1
        got = (
            int(getattr(gate, "out_features", -1)),
            int(getattr(gate, "in_features", -1)),
        )
        if got != (INNER, HIDDEN):
            raise RuntimeError(
                f"[LongMedia][FastVideo VSA PRECHECK] gate module {i} logical shape={got}, "
                f"expected={(INNER, HIDDEN)}"
            )

    contract = dict(contract)
    contract.update({"created_gates": created, "native_gates": native})
    setattr(base_model, "_longmedia_fastvideo_vsa_contract", contract)
    setattr(diffusion, "_longmedia_fastvideo_vsa_contract", contract)
    return contract


def _postcheck(base_model, contract):
    diffusion = base_model.diffusion_model
    gates = [getattr(block.attn, "to_gate_compress", None) for block in diffusion.blocks]
    if len(gates) != BLOCKS or any(g is None for g in gates):
        raise RuntimeError("[LongMedia][FastVideo VSA POSTCHECK] one or more gate modules disappeared during load")
    for i, gate in enumerate(gates):
        logical = (int(gate.out_features), int(gate.in_features))
        if logical != (INNER, HIDDEN):
            raise RuntimeError(
                f"[LongMedia][FastVideo VSA POSTCHECK] gate {i} logical shape={logical}, expected={(INNER,HIDDEN)}"
            )
        weight = getattr(gate, "weight", None)
        if weight is not None:
            got = _shape(weight)
            if got is not None and got != (INNER, HIDDEN):
                raise RuntimeError(
                    f"[LongMedia][FastVideo VSA POSTCHECK] gate {i} loaded weight shape={got}, expected={(INNER,HIDDEN)}"
                )
    setattr(base_model, "_longmedia_fastvideo_vsa_contract", contract)
    setattr(diffusion, "_longmedia_fastvideo_vsa_contract", contract)


def install_fastvideo_vsa_loader_compat() -> bool:
    try:
        import comfy.model_base as model_base
    except Exception:
        return False

    cls = getattr(model_base, "MiniMaxH3", None)
    if cls is None:
        return False
    if getattr(cls, "_longmedia_fastvideo_vsa_loader_installed", False):
        return True

    original = cls.load_model_weights

    def wrapped(self, sd, unet_prefix="", assign=False):
        reset = _clear_runtime(self)
        try:
            contract = classify_fastvideo_vsa(sd, unet_prefix)
            if contract is None:
                if any(reset.values()):
                    print(
                        "[MiniMaxH3 LongMedia][FastVideo VSA RESET] non-VSA checkpoint selected; "
                        f"removed_owned_gates={reset['gates']} cleared_contracts={reset['contracts']}",
                        flush=True,
                    )
                return original(self, sd, unet_prefix=unet_prefix, assign=assign)

            contract = _prepare_modules(self, contract)
            print(
                "[MiniMaxH3 LongMedia][FastVideo VSA PRECHECK] PASS: "
                f"gates={BLOCKS}/50; created={contract['created_gates']} native={contract['native_gates']}; "
                f"shape={(INNER,HIDDEN)}; tile=64 sparsity=0.90; 4-step T2AV contract; "
                "Comfy mixed-precision/dynamic-VRAM loader remains owner",
                flush=True,
            )
            result = original(self, sd, unet_prefix=unet_prefix, assign=assign)
            _postcheck(self, contract)
            print(
                "[MiniMaxH3 LongMedia][FastVideo VSA POSTCHECK] PASS: 50/50 to_gate_compress layers loaded; "
                "no H3ddle remap/AdaLN substitution; runtime contract attached",
                flush=True,
            )
            return result
        except Exception:
            _clear_runtime(self)
            raise

    cls.load_model_weights = wrapped
    cls._longmedia_fastvideo_vsa_loader_installed = True
    cls._longmedia_fastvideo_vsa_original_load_model_weights = original
    return True


install_fastvideo_vsa_loader_compat()
