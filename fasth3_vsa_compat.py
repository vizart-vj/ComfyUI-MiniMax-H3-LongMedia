from __future__ import annotations

"""FastH3 VSA checkpoint compatibility for native ComfyUI MiniMax H3.

PulpCut/H3ddle FastH3 VSA packages differ from stock Comfy H3 in three ways
that matter at runtime:

* 200 DiT core projections and 50 learned VSA gates are stored input-major;
* the learned VSA gate modules do not exist in stock MiniMaxH3Model;
* FastH3's full-width AdaLN path is serialized as seven exact BF16 lookup rows
  for the four-call dual-clock schedule.  These lookup rows are part of the
  trained function and must not be replaced by the pruned template AdaLN.

This module patches two narrowly scoped Comfy H3 entry points.  Model detection
must be corrected *before* MiniMaxH3Model is constructed because H3ddle stores
QKV/MLP input-major and stock Comfy otherwise infers 14 heads / FFN 2688 from
the wrong matrix axis.  MiniMaxH3.load_model_weights is then patched to convert
only storage orientation (never dequantize model weights), install learned VSA
gates, keep exact AdaLN lookup tables on CPU, and leave Comfy/LongMedia dynamic
VRAM ownership intact.  Every published FastH3 input-major shape is validated
before model construction so unsupported packages fail closed rather than
creating a silently wrong network.
"""

import dataclasses
import re
import types
import weakref
from typing import Any

import torch

FASTH3_MARKER = "h3.fasth3.version"
INPUT_MAJOR_MARKER = "h3.transformer_input_major.version"
FASTH3_VSA_FORMAT = 2
FASTH3_STEPS = 4
FASTH3_ROWS = 7
FASTH3_TILE = 64
FASTH3_SPARSITY = 0.9
FASTH3_VIDEO_SHIFT = 12.0
FASTH3_AUDIO_SHIFT = 3.0
FASTH3_BLOCK_MODALITIES = 3
FASTH3_BLOCK_EXPAND = 6
FASTH3_FINAL_MODALITIES = 1
FASTH3_FINAL_EXPAND = 2

# Exact MiniMax H3 / H3ddle FastH3 Preview v1 structural contract.  H3ddle's
# converter transposes only these 200 core matrices plus 50 VSA gates.
FASTH3_HIDDEN = 5376
FASTH3_HEAD_DIM = 128
FASTH3_HEADS = 56
FASTH3_ATTN_INNER = FASTH3_HEADS * FASTH3_HEAD_DIM  # 7168
FASTH3_FFN = 14336
FASTH3_BLOCKS = 50

_FASTH3_INPUT_MAJOR_CORE_SHAPES = {
    "attn.qkv_proj.weight": (FASTH3_HIDDEN, FASTH3_ATTN_INNER * 3),
    "attn.out_proj.weight": (FASTH3_ATTN_INNER, FASTH3_HIDDEN),
    "mlp.fc1.weight": (FASTH3_HIDDEN, FASTH3_FFN * 2),
    "mlp.fc2.weight": (FASTH3_FFN, FASTH3_HIDDEN),
}
_FASTH3_INPUT_MAJOR_GATE_SHAPE = (FASTH3_HIDDEN, FASTH3_ATTN_INNER)

_GATE_RE = re.compile(r"^blocks\.(\d+)\.attn\.vsa_gate\.weight$")
_MARKER_RE = re.compile(
    r"^h3\.fasth3\.(?:version|steps|times|vsa\.(?:tile_size|sparsity)|"
    r"blocks\.\d+\.adaln|final\.adaln)$"
)


def _value_cpu(value: Any):
    if not torch.is_tensor(value):
        return value
    v = value.detach().cpu()
    if v.numel() == 1:
        return v.item()
    return v


def _required_fast_h3_keys(prefix: str) -> tuple[str, ...]:
    keys = [
        f"{prefix}{FASTH3_MARKER}",
        f"{prefix}{INPUT_MAJOR_MARKER}",
        f"{prefix}h3.fasth3.steps",
        f"{prefix}h3.fasth3.times",
        f"{prefix}h3.fasth3.vsa.tile_size",
        f"{prefix}h3.fasth3.vsa.sparsity",
        f"{prefix}h3.fasth3.final.adaln",
    ]
    for block in range(FASTH3_BLOCKS):
        keys.append(f"{prefix}h3.fasth3.blocks.{block}.adaln")
        keys.append(f"{prefix}blocks.{block}.attn.vsa_gate.weight")
    return tuple(keys)


def _strict_fast_h3_prefix(sd: dict[str, Any], key_prefix: str) -> str | None:
    """Classify only a complete FastH3 VSA package at this exact prefix.

    Ordinary H3 / FL2VA / Ref2VA checkpoints must be byte-for-byte invisible to
    the FastH3 runtime.  A lone input-major marker is not FastH3, and a FastH3
    marker without the full VSA/AdaLN payload is treated as a malformed FastH3
    package rather than silently falling through to stock H3.
    """
    marker = f"{key_prefix}{FASTH3_MARKER}"
    if marker not in sd:
        return None
    missing = [key for key in _required_fast_h3_keys(key_prefix) if key not in sd]
    if missing:
        raise RuntimeError(
            "[LongMedia][FastH3 DETECT PRECHECK] FastH3 marker is present but the "
            "package contract is incomplete; refusing to patch a partial checkpoint. "
            f"missing={missing[:12]} total_missing={len(missing)}"
        )
    return key_prefix


def _detection_fast_prefix(sd: dict[str, Any], key_prefix: str) -> str | None:
    return _strict_fast_h3_prefix(sd, key_prefix)


def _require_tensor_shape(sd: dict[str, Any], key: str, expected: tuple[int, ...]) -> None:
    value = sd.get(key)
    got = _shape_tuple(value)
    if got != expected:
        raise RuntimeError(
            "[LongMedia][FastH3 DETECT PRECHECK] input-major package shape mismatch: "
            f"{key}: checkpoint={got}, expected={expected}"
        )


def _validate_fast_h3_input_major_header(sd: dict[str, Any], prefix: str) -> dict[str, int]:
    """Validate the complete H3ddle FastH3 matrix contract before model creation.

    This deliberately uses checkpoint/header shapes rather than any Comfy module
    objects.  Stock Comfy's H3 detector reads axis 0 as output-major; on this
    package that would infer 14 heads and FFN 2688, which then poisons every
    Linear shape created afterwards.
    """
    if f"{prefix}{INPUT_MAJOR_MARKER}" not in sd:
        raise RuntimeError(
            "[LongMedia][FastH3 DETECT PRECHECK] FastH3 marker is present but "
            "h3.transformer_input_major.version is missing"
        )

    core = 0
    gates = 0
    for block in range(FASTH3_BLOCKS):
        base = f"{prefix}blocks.{block}."
        for suffix, expected in _FASTH3_INPUT_MAJOR_CORE_SHAPES.items():
            _require_tensor_shape(sd, base + suffix, expected)
            core += 1
        _require_tensor_shape(
            sd,
            f"{prefix}blocks.{block}.attn.vsa_gate.weight",
            _FASTH3_INPUT_MAJOR_GATE_SHAPE,
        )
        gates += 1

    if core != 200 or gates != 50:
        raise RuntimeError(
            f"[LongMedia][FastH3 DETECT PRECHECK] internal validation count error: "
            f"core={core}/200 gates={gates}/50"
        )
    return {"core": core, "gates": gates}


def _correct_fast_h3_detected_config(
    sd: dict[str, Any], key_prefix: str, config: dict[str, Any] | None
) -> dict[str, Any] | None:
    prefix = _detection_fast_prefix(sd, key_prefix)
    if prefix is None:
        return config

    receipt = _validate_fast_h3_input_major_header(sd, prefix)
    if not isinstance(config, dict) or config.get("image_model") != "minimax_h3":
        raise RuntimeError(
            "[LongMedia][FastH3 DETECT PRECHECK] Comfy did not identify the "
            f"input-major package as MiniMax H3: config={config!r}"
        )

    hidden = int(config.get("hidden_size", -1))
    head_dim = int(config.get("attention_head_dim", -1))
    if hidden != FASTH3_HIDDEN or head_dim != FASTH3_HEAD_DIM:
        raise RuntimeError(
            "[LongMedia][FastH3 DETECT PRECHECK] base H3 architecture mismatch: "
            f"hidden={hidden} head_dim={head_dim}, expected "
            f"{FASTH3_HIDDEN}/{FASTH3_HEAD_DIM}"
        )

    # The two fields below are precisely the ones stock Comfy infers from the
    # wrong axis of input-major QKV/fc1.  Correct them before supported_models
    # instantiates MiniMaxH3Model; all remaining detector fields come from
    # tensors that H3ddle leaves output-major.
    fixed = dict(config)
    fixed["num_attention_heads"] = FASTH3_HEADS
    fixed["ffn_hidden_size"] = FASTH3_FFN

    print(
        "[MiniMaxH3 LongMedia][FastH3 DETECT PRECHECK] PASS: "
        f"input-major core={receipt['core']}/200 gates={receipt['gates']}/50; "
        f"corrected Comfy architecture heads={FASTH3_HEADS}, "
        f"attn_inner={FASTH3_ATTN_INNER}, ffn={FASTH3_FFN}",
        flush=True,
    )
    return fixed


def install_fast_h3_model_detection_compat() -> bool:
    """Patch only MiniMax H3 config inference for marked input-major FastH3."""
    try:
        import comfy.model_detection as model_detection
    except Exception:
        return False

    if getattr(model_detection, "_longmedia_fasth3_detection_installed", False):
        return True
    original = model_detection.detect_unet_config

    def wrapped(state_dict, key_prefix, metadata=None):
        config = original(state_dict, key_prefix, metadata=metadata)
        return _correct_fast_h3_detected_config(state_dict, key_prefix, config)

    model_detection.detect_unet_config = wrapped
    model_detection._longmedia_fasth3_detection_installed = True
    model_detection._longmedia_fasth3_original_detect_unet_config = original
    return True


def _module_for_path(root, path: str):
    obj = root
    for part in path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def _shape_tuple(value) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(x) for x in shape)
    except Exception:
        return None


def _expected_weight_shape(module) -> tuple[int, int]:
    """Return logical Linear [out,in] shape without requiring materialized weight.

    Current Comfy MixedPrecisionOps.Linear deliberately has no ``.weight`` in
    __init__ while dynamic loading is preparing the model.  ``in_features`` /
    ``out_features`` and ``_orig_shape`` are the load-time ABI and therefore the
    authoritative source here.
    """
    in_features = getattr(module, "in_features", None)
    out_features = getattr(module, "out_features", None)
    if in_features is not None and out_features is not None:
        expected = (int(out_features), int(in_features))
        orig = getattr(module, "_orig_shape", None)
        if orig is not None:
            orig_shape = _shape_tuple(orig) if not isinstance(orig, (tuple, list)) else tuple(int(x) for x in orig)
            if orig_shape is not None and orig_shape != expected:
                raise RuntimeError(
                    "[LongMedia][FastH3 PRECHECK] Linear logical-shape contract changed: "
                    f"features imply {expected}, _orig_shape={orig_shape}"
                )
        return expected

    # Compatibility fallback for ordinary torch.nn.Linear-like modules.  This
    # path is intentionally secondary; dynamic Comfy linears can be weightless.
    weight = getattr(module, "weight", None)
    shape = _shape_tuple(weight)
    if shape is None or len(shape) != 2:
        raise RuntimeError(
            "[LongMedia][FastH3 PRECHECK] cannot determine Linear logical shape: "
            f"type={type(module).__name__}, in_features={in_features}, "
            f"out_features={out_features}, weight_shape={shape}"
        )
    return int(shape[0]), int(shape[1])


def _same_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Best-effort proof that a transpose is metadata-only, not a RAM copy."""
    try:
        return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    except Exception:
        try:
            return a.storage().data_ptr() == b.storage().data_ptr()
        except Exception:
            return False


def _transpose_quantized_zero_copy(value, expected: tuple[int, int], key: str):
    """Flip physical INT8 storage without materialising ~20 GiB of copies.

    H3ddle repacks only qdata input-major and intentionally leaves output-row
    scales/ConvRot metadata untouched.  If Comfy has already wrapped the tensor
    in a QuantizedTensor, rebuild the wrapper around a *view* of qdata.T and
    restore a non-transposed logical layout so the regular INT8 linear fast path
    remains eligible.  No qdata bytes are copied here.
    """
    qdata = getattr(value, "_qdata", None)
    params = getattr(value, "_params", None)
    layout = getattr(value, "_layout_cls", None)
    if not torch.is_tensor(qdata) or params is None or layout is None:
        return None
    if qdata.ndim != 2:
        raise RuntimeError(
            f"[LongMedia][FastH3 PRECHECK] {key}: quantized qdata must be 2D, got {tuple(qdata.shape)}"
        )
    qview = qdata.t()
    if not _same_storage(qdata, qview):
        raise RuntimeError(
            f"[LongMedia][FastH3 PRECHECK] {key}: quantized transpose unexpectedly allocated storage"
        )
    try:
        new_params = dataclasses.replace(params, orig_shape=tuple(expected), transposed=False)
        rebuilt = type(value)(qview, layout, new_params)
    except Exception as exc:
        raise RuntimeError(
            f"[LongMedia][FastH3 PRECHECK] {key}: could not rebuild quantized zero-copy transpose: {exc}"
        ) from exc
    if tuple(int(x) for x in rebuilt.shape) != tuple(expected):
        raise RuntimeError(
            f"[LongMedia][FastH3 PRECHECK] {key}: rebuilt quantized shape={tuple(rebuilt.shape)}, expected={expected}"
        )
    return rebuilt


def _convert_weight_to_expected(value: torch.Tensor, expected: tuple[int, int], key: str):
    if not torch.is_tensor(value) or value.ndim != 2:
        raise RuntimeError(
            f"[LongMedia][FastH3 PRECHECK] {key}: expected a 2D tensor, "
            f"got {type(value).__name__} shape={getattr(value, 'shape', None)}"
        )
    got = tuple(int(x) for x in value.shape)
    if got == expected:
        return value, False
    if got == expected[::-1]:
        # CRITICAL MEMORY CONTRACT: never call .contiguous() here.  The FastH3
        # core + VSA gates are ~19.74 GiB of INT8 data; eager materialisation of
        # all transposes at checkpoint load pushes a 64-GiB Windows host into
        # pagefile thrash and defeats Comfy Dynamic VRAM/AIMDO zero-copy loading.
        rebuilt = _transpose_quantized_zero_copy(value, expected, key)
        if rebuilt is not None:
            return rebuilt, True
        view = value.t()
        if not _same_storage(value, view):
            raise RuntimeError(
                f"[LongMedia][FastH3 PRECHECK] {key}: transpose unexpectedly allocated storage"
            )
        return view, True
    raise RuntimeError(
        "[LongMedia][FastH3 PRECHECK] tensor shape mismatch before model load: "
        f"{key}: checkpoint={got}, Comfy expects={expected}, "
        f"accepted input-major={expected[::-1]}"
    )


def _new_gate_like(attention):
    """Create a VSA gate with the exact same Linear implementation as QKV.

    Do not manufacture a dummy ``weight``: that would defeat Comfy's dynamic
    loader and create a large transient allocation on Windows.
    """
    qkv = attention.qkv_proj
    hidden = int(qkv.in_features)
    inner = int(attention.heads) * int(attention.head_dim)
    cls = type(qkv)

    kwargs: dict[str, Any] = {"bias": False}
    factory = getattr(qkv, "factory_kwargs", None)
    if isinstance(factory, dict):
        if factory.get("device", None) is not None:
            kwargs["device"] = factory["device"]
        if factory.get("dtype", None) is not None:
            kwargs["dtype"] = factory["dtype"]
    else:
        weight = getattr(qkv, "weight", None)
        if torch.is_tensor(weight):
            kwargs["device"] = weight.device
            if weight.dtype is not None:
                kwargs["dtype"] = weight.dtype

    try:
        gate = cls(hidden, inner, **kwargs)
    except TypeError:
        kwargs.pop("dtype", None)
        kwargs.pop("device", None)
        gate = cls(hidden, inner, **kwargs)

    got = _expected_weight_shape(gate)
    expected = (inner, hidden)
    if got != expected:
        raise RuntimeError(
            "[LongMedia][FastH3 PRECHECK] created VSA gate has wrong logical shape: "
            f"got={got}, expected={expected}, class={cls.__name__}"
        )
    return gate


def _validate_scale(sd: dict[str, Any], prefix: str, rel_weight_key: str, expected: tuple[int, int]):
    scale_key = prefix + rel_weight_key.removesuffix(".weight") + ".weight_scale"
    scale = sd.get(scale_key)
    if scale is None or not torch.is_tensor(scale):
        return
    n = int(scale.numel())
    if n not in (1, int(expected[0])):
        raise RuntimeError(
            f"[LongMedia][FastH3 PRECHECK] quant scale mismatch: {scale_key} "
            f"has {tuple(scale.shape)} ({n} values), expected scalar or {expected[0]} outputs"
        )


def _shift_sigma(base: torch.Tensor, shift: float) -> torch.Tensor:
    s = float(shift)
    return (s * base) / (1.0 + (s - 1.0) * base)


def _expected_fast_times() -> torch.Tensor:
    """Exact row order emitted by H3ddle convert-fasth3-package.py."""
    rows: list[float] = []
    for index in range(FASTH3_STEPS):
        base = torch.tensor(1.0 - index / FASTH3_STEPS, dtype=torch.float32)
        tv = float(1.0 - _shift_sigma(base, FASTH3_VIDEO_SHIFT))
        ta = float(1.0 - _shift_sigma(base, FASTH3_AUDIO_SHIFT))
        if abs(tv - ta) <= 1e-8:
            rows.append(tv)
        elif tv < ta:
            rows.extend((tv, ta))
        else:
            rows.extend((ta, tv))
    return torch.tensor(rows, dtype=torch.float32)


def _capture_fast_adaln_tables(
    sd: dict[str, Any], prefix: str, hidden: int
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor, dict[str, Any]]:
    times_key = prefix + "h3.fasth3.times"
    times = sd.get(times_key)
    if not torch.is_tensor(times):
        raise RuntimeError(f"[LongMedia][FastH3 PRECHECK] missing required lookup axis: {times_key}")
    times_cpu = times.detach().to(device="cpu", dtype=torch.float32).reshape(-1).contiguous()
    if tuple(times_cpu.shape) != (FASTH3_ROWS,):
        raise RuntimeError(
            f"[LongMedia][FastH3 PRECHECK] {times_key}: expected {FASTH3_ROWS} rows, "
            f"got {tuple(times_cpu.shape)}"
        )
    expected_times = _expected_fast_times()
    max_time_err = float((times_cpu - expected_times).abs().max().item())
    if max_time_err > 5e-5:
        raise RuntimeError(
            "[LongMedia][FastH3 PRECHECK] FastH3 AdaLN time table does not match the "
            f"published 4-call 12/3 schedule; max_abs_error={max_time_err:.7g}; "
            f"checkpoint={times_cpu.tolist()}, expected={expected_times.tolist()}"
        )

    block_width = FASTH3_BLOCK_MODALITIES * FASTH3_BLOCK_EXPAND * int(hidden)
    block_tables: list[torch.Tensor] = []
    for i in range(50):
        key = prefix + f"h3.fasth3.blocks.{i}.adaln"
        value = sd.get(key)
        if not torch.is_tensor(value):
            raise RuntimeError(f"[LongMedia][FastH3 PRECHECK] missing required AdaLN lookup: {key}")
        shape = tuple(int(x) for x in value.shape)
        expected = (FASTH3_ROWS, block_width)
        if shape != expected:
            raise RuntimeError(
                f"[LongMedia][FastH3 PRECHECK] {key}: shape={shape}, expected={expected}"
            )
        if value.dtype != torch.bfloat16:
            raise RuntimeError(
                f"[LongMedia][FastH3 PRECHECK] {key}: dtype={value.dtype}, expected torch.bfloat16"
            )
        # Keep the original CPU payload; do not widen/dequantize/copy to GPU.
        block_tables.append(value.detach().cpu())

    final_key = prefix + "h3.fasth3.final.adaln"
    final = sd.get(final_key)
    if not torch.is_tensor(final):
        raise RuntimeError(f"[LongMedia][FastH3 PRECHECK] missing required final AdaLN lookup: {final_key}")
    final_expected = (FASTH3_ROWS, FASTH3_FINAL_EXPAND * int(hidden))
    final_shape = tuple(int(x) for x in final.shape)
    if final_shape != final_expected:
        raise RuntimeError(
            f"[LongMedia][FastH3 PRECHECK] {final_key}: shape={final_shape}, expected={final_expected}"
        )
    if final.dtype != torch.bfloat16:
        raise RuntimeError(
            f"[LongMedia][FastH3 PRECHECK] {final_key}: dtype={final.dtype}, expected torch.bfloat16"
        )

    receipt = {
        "rows": FASTH3_ROWS,
        "times": [float(v) for v in times_cpu.tolist()],
        "max_time_error": max_time_err,
        "block_tables": 50,
        "block_shape": (FASTH3_ROWS, block_width),
        "final_shape": final_expected,
        "dtype": "torch.bfloat16",
    }
    return times_cpu, tuple(block_tables), final.detach().cpu(), receipt


def _curve_candidates(diffusion, times_cpu: torch.Tensor) -> torch.Tensor:
    table = getattr(diffusion, "adaln_t_table", None)
    if not torch.is_tensor(table):
        raise RuntimeError(
            "[LongMedia][FastH3 POSTCHECK] FastH3 package requires the pruned-curve "
            "MiniMaxH3 architecture with adaln_t_table, but the loaded Comfy model does not expose it"
        )
    curve = table.detach().to(device="cpu", dtype=torch.float32)
    if curve.ndim != 2 or int(curve.shape[0]) < 2:
        raise RuntimeError(
            f"[LongMedia][FastH3 POSTCHECK] invalid adaln_t_table shape={tuple(curve.shape)}"
        )
    pos = times_cpu.clamp(0.0, 1.0) * (int(curve.shape[0]) - 1)
    i0 = pos.floor().long().clamp(max=int(curve.shape[0]) - 2)
    frac = (pos - i0.to(pos.dtype)).unsqueeze(1)
    candidates = torch.lerp(curve.index_select(0, i0), curve.index_select(0, i0 + 1), frac)
    return candidates.contiguous()


def _resolve_fast_temb_rows(diffusion, t_emb: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t_emb) or t_emb.ndim != 2:
        raise RuntimeError(
            "[LongMedia][FastH3 AdaLN] expected t_emb [M,D], "
            f"got {type(t_emb).__name__} shape={getattr(t_emb, 'shape', None)}"
        )

    cache = getattr(diffusion, "_longmedia_fasth3_temb_cache", None)
    if isinstance(cache, dict):
        ref = cache.get("ref")
        try:
            if ref is not None and ref() is t_emb:
                return cache["indices"]
        except Exception:
            pass

    candidates = getattr(diffusion, "_longmedia_fasth3_curve_candidates_cpu", None)
    times = getattr(diffusion, "_longmedia_fasth3_times_cpu", None)
    if not torch.is_tensor(candidates) or not torch.is_tensor(times):
        raise RuntimeError("[LongMedia][FastH3 AdaLN] lookup candidates were not installed")
    if int(t_emb.shape[-1]) != int(candidates.shape[-1]):
        raise RuntimeError(
            "[LongMedia][FastH3 AdaLN] t_emb width mismatch: "
            f"runtime={int(t_emb.shape[-1])}, lookup_basis={int(candidates.shape[-1])}"
        )

    observed = t_emb.detach().to(device="cpu", dtype=torch.float32)
    distances = (observed[:, None, :] - candidates[None, :, :]).abs().amax(dim=-1)
    errors, indices = distances.min(dim=1)
    scale = candidates.index_select(0, indices).abs().amax(dim=1)
    tolerance = 5e-4 + 2e-4 * scale
    bad = errors > tolerance
    if bool(bad.any()):
        row = int(torch.nonzero(bad, as_tuple=False)[0].item())
        nearest = int(indices[row].item())
        raise RuntimeError(
            "[MiniMaxH3 LongMedia][FastH3 T2VA CONTRACT] runtime requested an AdaLN timestep "
            "outside FastH3 Preview v1's exact seven-row 4-call table. This checkpoint is T2VA-only; "
            "continuation/reference/conditioning timesteps are not silently approximated. "
            f"row={row}, nearest_fast_time={float(times[nearest]):.8f}, "
            f"embedding_max_error={float(errors[row]):.6g}, tolerance={float(tolerance[row]):.6g}"
        )

    indices = indices.to(dtype=torch.long, device="cpu")
    try:
        ref = weakref.ref(t_emb)
    except TypeError:
        ref = None
    diffusion._longmedia_fasth3_temb_cache = {"ref": ref, "indices": indices}
    return indices


def _fasth3_adaln_lookup_forward(self, t_emb):
    owner_ref = getattr(self, "_longmedia_fasth3_owner_ref", None)
    diffusion = owner_ref() if callable(owner_ref) else None
    if diffusion is None:
        raise RuntimeError("[LongMedia][FastH3 AdaLN] owning MiniMaxH3Model is unavailable")
    indices = _resolve_fast_temb_rows(diffusion, t_emb)
    table = getattr(self, "_longmedia_fasth3_lookup_table_cpu", None)
    if not torch.is_tensor(table):
        raise RuntimeError("[LongMedia][FastH3 AdaLN] lookup table is missing")
    selected = table.index_select(0, indices).to(device=t_emb.device, non_blocking=True)
    modalities = int(getattr(self, "_longmedia_fasth3_modalities"))
    expand = int(getattr(self, "_longmedia_fasth3_expand"))
    hidden = int(getattr(self, "_longmedia_fasth3_hidden"))
    expected_width = modalities * expand * hidden
    if int(selected.shape[-1]) != expected_width:
        raise RuntimeError(
            "[LongMedia][FastH3 AdaLN] selected lookup width changed unexpectedly: "
            f"got={int(selected.shape[-1])}, expected={expected_width}"
        )
    x = selected.reshape(int(selected.shape[0]) * modalities, expand * hidden)
    return x.chunk(expand, dim=-1)


def _install_adaln_lookup_runtime(diffusion, contract: dict[str, Any]) -> None:
    times_cpu = contract.get("_adaln_times_cpu")
    block_tables = contract.get("_adaln_block_tables_cpu")
    final_table = contract.get("_adaln_final_table_cpu")
    if not torch.is_tensor(times_cpu) or not isinstance(block_tables, tuple) or len(block_tables) != 50:
        raise RuntimeError("[LongMedia][FastH3 POSTCHECK] exact AdaLN payload is incomplete")
    if not torch.is_tensor(final_table):
        raise RuntimeError("[LongMedia][FastH3 POSTCHECK] final AdaLN payload is incomplete")

    candidates = _curve_candidates(diffusion, times_cpu)
    blocks = list(diffusion.blocks)
    hidden = int(blocks[0].attn.qkv_proj.in_features)

    # Verify the compact Comfy curve coordinate width before replacing forwards.
    block_linear = blocks[0].adaln_proj.linear
    block_linear_shape = _expected_weight_shape(block_linear)
    if block_linear_shape[1] != int(candidates.shape[1]):
        raise RuntimeError(
            "[LongMedia][FastH3 POSTCHECK] pruned AdaLN basis mismatch: "
            f"block input={block_linear_shape[1]}, curve width={int(candidates.shape[1])}"
        )
    if block_linear_shape[0] != FASTH3_BLOCK_MODALITIES * FASTH3_BLOCK_EXPAND * hidden:
        raise RuntimeError(
            "[LongMedia][FastH3 POSTCHECK] block AdaLN output ABI changed: "
            f"got={block_linear_shape[0]}, expected={FASTH3_BLOCK_MODALITIES * FASTH3_BLOCK_EXPAND * hidden}"
        )

    final_proj = diffusion.final_layer.adaln_proj
    final_shape = _expected_weight_shape(final_proj.linear)
    if final_shape[1] != int(candidates.shape[1]) or final_shape[0] != FASTH3_FINAL_EXPAND * hidden:
        raise RuntimeError(
            "[LongMedia][FastH3 POSTCHECK] final AdaLN ABI mismatch: "
            f"linear={final_shape}, expected=({FASTH3_FINAL_EXPAND * hidden}, {int(candidates.shape[1])})"
        )

    diffusion._longmedia_fasth3_times_cpu = times_cpu
    diffusion._longmedia_fasth3_curve_candidates_cpu = candidates
    diffusion._longmedia_fasth3_temb_cache = None

    def patch_proj(proj, table, modalities: int, expand: int):
        if getattr(proj, "_longmedia_fasth3_lookup_installed", False):
            return
        proj._longmedia_fasth3_original_forward = proj.forward
        proj._longmedia_fasth3_lookup_table_cpu = table
        proj._longmedia_fasth3_owner_ref = weakref.ref(diffusion)
        proj._longmedia_fasth3_modalities = int(modalities)
        proj._longmedia_fasth3_expand = int(expand)
        proj._longmedia_fasth3_hidden = int(hidden)
        proj.forward = types.MethodType(_fasth3_adaln_lookup_forward, proj)
        proj._longmedia_fasth3_lookup_installed = True

    for i, block in enumerate(blocks):
        patch_proj(block.adaln_proj, block_tables[i], FASTH3_BLOCK_MODALITIES, FASTH3_BLOCK_EXPAND)
    patch_proj(final_proj, final_table, FASTH3_FINAL_MODALITIES, FASTH3_FINAL_EXPAND)

    # Mathematical smoke check without allocating hidden activations: each table
    # must reshape/chunk exactly like stock AdalnProj.forward.
    sample = block_tables[0].index_select(0, torch.tensor([0, 1], dtype=torch.long))
    chunks = sample.reshape(2 * FASTH3_BLOCK_MODALITIES, FASTH3_BLOCK_EXPAND * hidden).chunk(
        FASTH3_BLOCK_EXPAND, dim=-1
    )
    if len(chunks) != FASTH3_BLOCK_EXPAND or any(tuple(c.shape) != (2 * FASTH3_BLOCK_MODALITIES, hidden) for c in chunks):
        raise RuntimeError("[LongMedia][FastH3 POSTCHECK] block AdaLN lookup reshape/chunk contract failed")
    fchunks = final_table[:2].reshape(2, FASTH3_FINAL_EXPAND * hidden).chunk(FASTH3_FINAL_EXPAND, dim=-1)
    if len(fchunks) != 2 or any(tuple(c.shape) != (2, hidden) for c in fchunks):
        raise RuntimeError("[LongMedia][FastH3 POSTCHECK] final AdaLN lookup reshape/chunk contract failed")

    contract["adaln_lookup_ready"] = True
    contract["adaln_curve_dim"] = int(candidates.shape[1])
    contract["adaln_rows"] = FASTH3_ROWS


def _clear_fast_h3_runtime_state(base_model) -> dict[str, int]:
    """Restore a MiniMaxH3 instance to a stock-H3 runtime before any reload.

    Comfy may reuse a MiniMaxH3 Python object when the user switches diffusion
    checkpoints.  FastH3 previously patched AdaLN forwards and attached VSA
    modules/contract attributes to that object; without an explicit reset, a
    later ordinary FL2VA checkpoint inherited the seven-row FastH3 AdaLN path.
    This reset is idempotent and removes only LongMedia-owned runtime state.
    """
    diffusion = getattr(base_model, "diffusion_model", None)
    if diffusion is None:
        return {"adaln": 0, "gates": 0, "contracts": 0}

    had_contract = bool(
        getattr(base_model, "_longmedia_fasth3_contract", None) is not None
        or getattr(diffusion, "_longmedia_fasth3_contract", None) is not None
    )
    restored = 0
    removed_gates = 0

    def restore_proj(proj):
        nonlocal restored
        if proj is None:
            return
        original = getattr(proj, "_longmedia_fasth3_original_forward", None)
        if original is not None:
            proj.forward = original
            restored += 1
        for name in (
            "_longmedia_fasth3_original_forward",
            "_longmedia_fasth3_lookup_table_cpu",
            "_longmedia_fasth3_owner_ref",
            "_longmedia_fasth3_modalities",
            "_longmedia_fasth3_expand",
            "_longmedia_fasth3_hidden",
            "_longmedia_fasth3_lookup_installed",
        ):
            try:
                delattr(proj, name)
            except AttributeError:
                pass

    blocks = list(getattr(diffusion, "blocks", ()) or ())
    for block in blocks:
        restore_proj(getattr(block, "adaln_proj", None))
        attn = getattr(block, "attn", None)
        gate = getattr(attn, "vsa_gate", None) if attn is not None else None
        if gate is not None and (had_contract or bool(getattr(gate, "_longmedia_fasth3_owned", False))):
            try:
                delattr(attn, "vsa_gate")
                removed_gates += 1
            except AttributeError:
                pass

    final_layer = getattr(diffusion, "final_layer", None)
    restore_proj(getattr(final_layer, "adaln_proj", None) if final_layer is not None else None)

    contracts = 0
    for owner in (base_model, diffusion):
        if hasattr(owner, "_longmedia_fasth3_contract"):
            try:
                delattr(owner, "_longmedia_fasth3_contract")
                contracts += 1
            except AttributeError:
                pass

    for name in (
        "_longmedia_fasth3_times_cpu",
        "_longmedia_fasth3_curve_candidates_cpu",
        "_longmedia_fasth3_temb_cache",
    ):
        try:
            delattr(diffusion, name)
        except AttributeError:
            pass

    return {"adaln": restored, "gates": removed_gates, "contracts": contracts}


def _prepare_fast_h3_checkpoint(base_model, sd: dict[str, Any], unet_prefix: str = "") -> dict[str, Any] | None:
    """Mutate raw state_dict into Comfy-compatible layout, fail closed on mismatch."""
    prefix = _strict_fast_h3_prefix(sd, unet_prefix)
    if prefix is None:
        return None

    diffusion = base_model.diffusion_model
    blocks = list(getattr(diffusion, "blocks", ()))
    if len(blocks) != FASTH3_BLOCKS:
        raise RuntimeError(
            f"[LongMedia][FastH3 PRECHECK] expected {FASTH3_BLOCKS} MiniMax H3 DiT blocks, got {len(blocks)}"
        )
    hidden = int(blocks[0].attn.qkv_proj.in_features)
    if hidden != 5376:
        raise RuntimeError(
            f"[LongMedia][FastH3 PRECHECK] target checkpoint is MiniMax H3 hidden=5376, loaded model reports {hidden}"
        )

    version_value = sd.get(prefix + FASTH3_MARKER)
    if not torch.is_tensor(version_value) or int(version_value.reshape(-1)[0].item()) != FASTH3_VSA_FORMAT:
        raise RuntimeError(
            "[LongMedia][FastH3 PRECHECK] this backend expects learned-VSA package format 2; "
            f"{prefix + FASTH3_MARKER}={_value_cpu(version_value)}"
        )

    # Capture exact FastH3 AdaLN function before consuming engine-only keys.
    times_cpu, block_tables, final_table, adaln_receipt = _capture_fast_adaln_tables(sd, prefix, hidden)

    raw_gate_indices = []
    for key in list(sd.keys()):
        if not key.startswith(prefix):
            continue
        rel = key[len(prefix):]
        m = _GATE_RE.match(rel)
        if m:
            raw_gate_indices.append(int(m.group(1)))
    raw_gate_indices = sorted(set(raw_gate_indices))
    if raw_gate_indices != list(range(50)):
        missing = sorted(set(range(50)) - set(raw_gate_indices))
        extra = sorted(set(raw_gate_indices) - set(range(50)))
        raise RuntimeError(
            "[LongMedia][FastH3 PRECHECK] VSA gate contract mismatch: "
            f"found={len(raw_gate_indices)}/50 missing={missing[:12]} extra={extra[:12]}"
        )

    # Add the learned gates before Comfy's load_state_dict so dynamic/quantized
    # ownership is native.  No dummy weights are allocated.
    for block in blocks:
        attn = block.attn
        if not hasattr(attn, "vsa_gate"):
            gate = _new_gate_like(attn)
            gate._longmedia_fasth3_owned = True
            attn.add_module("vsa_gate", gate)

    converted = 0
    checked = 0
    expected_core = set()
    for i in range(FASTH3_BLOCKS):
        expected_core.update({
            f"blocks.{i}.attn.qkv_proj.weight",
            f"blocks.{i}.attn.out_proj.weight",
            f"blocks.{i}.mlp.fc1.weight",
            f"blocks.{i}.mlp.fc2.weight",
        })

    for rel in sorted(expected_core):
        key = prefix + rel
        if key not in sd:
            raise RuntimeError(f"[LongMedia][FastH3 PRECHECK] missing required FastH3 core tensor: {key}")
        module = _module_for_path(diffusion, rel.removesuffix(".weight"))
        expected = _expected_weight_shape(module)
        sd[key], did = _convert_weight_to_expected(sd[key], expected, key)
        converted += int(did)
        checked += 1
        _validate_scale(sd, prefix, rel, expected)
    if checked != 200:
        raise RuntimeError(f"[LongMedia][FastH3 PRECHECK] internal core count error: {checked}/200")

    gate_converted = 0
    for i, block in enumerate(blocks):
        rel = f"blocks.{i}.attn.vsa_gate.weight"
        key = prefix + rel
        expected = _expected_weight_shape(block.attn.vsa_gate)
        sd[key], did = _convert_weight_to_expected(sd[key], expected, key)
        gate_converted += int(did)
        _validate_scale(sd, prefix, rel, expected)

    # Small scalar metadata receipt only. Exact AdaLN tensors stay in the private
    # CPU runtime contract below, not in the generic diagnostic dict.
    meta: dict[str, Any] = {}
    consumed: list[str] = []
    for key in list(sd.keys()):
        if not key.startswith(prefix):
            continue
        rel = key[len(prefix):]
        if rel == INPUT_MAJOR_MARKER or _MARKER_RE.match(rel):
            if rel == "h3.fasth3.times" or rel.endswith(".adaln"):
                consumed.append(key)
                continue
            meta[rel] = _value_cpu(sd[key])
            consumed.append(key)
    for key in consumed:
        sd.pop(key, None)

    steps = int(meta.get("h3.fasth3.steps", FASTH3_STEPS) or FASTH3_STEPS)
    if steps != FASTH3_STEPS:
        raise RuntimeError(
            f"[LongMedia][FastH3 PRECHECK] expected exactly {FASTH3_STEPS} calls, checkpoint says steps={steps}"
        )
    tile = int(meta.get("h3.fasth3.vsa.tile_size", FASTH3_TILE) or FASTH3_TILE)
    sparsity = float(meta.get("h3.fasth3.vsa.sparsity", FASTH3_SPARSITY) or FASTH3_SPARSITY)
    if tile != FASTH3_TILE or abs(sparsity - FASTH3_SPARSITY) > 1e-5:
        raise RuntimeError(
            f"[LongMedia][FastH3 PRECHECK] unsupported VSA contract: tile={tile}, sparsity={sparsity}; "
            f"expected tile={FASTH3_TILE}, sparsity={FASTH3_SPARSITY}"
        )

    contract = {
        "version": int(meta.get("h3.fasth3.version", FASTH3_VSA_FORMAT) or FASTH3_VSA_FORMAT),
        "input_major": meta.get(INPUT_MAJOR_MARKER, 1),
        "steps": steps,
        "tile_size": tile,
        "sparsity": sparsity,
        "topk_ratio": 1.0 - sparsity,
        "core_weights_checked": checked,
        "core_weights_transposed": converted,
        "gate_weights_checked": 50,
        "gate_weights_transposed": gate_converted,
        "adaln_receipt": adaln_receipt,
        "adaln_lookup_ready": False,
        "metadata": meta,
        "source_prefix": prefix,
        "task_family": "t2va_only",
        "sigma_shift_video": FASTH3_VIDEO_SHIFT,
        "sigma_shift_audio": FASTH3_AUDIO_SHIFT,
        # Intentionally CPU-only plain tensor attributes: they are not module
        # buffers and therefore do not inflate GPU residency/offload staging.
        "_adaln_times_cpu": times_cpu,
        "_adaln_block_tables_cpu": block_tables,
        "_adaln_final_table_cpu": final_table,
    }
    setattr(base_model, "_longmedia_fasth3_contract", contract)
    setattr(diffusion, "_longmedia_fasth3_contract", contract)

    print(
        "[MiniMaxH3 LongMedia][FastH3 PRECHECK] PASS: "
        f"core={checked}/200 transposed={converted}; VSA gates=50/50 transposed={gate_converted}; "
        f"AdaLN exact rows={FASTH3_ROWS} x (50+final); steps={steps}; tile={tile}; sparsity={sparsity:.2f}; "
        "zero-copy input-major remap; no model-weight dequantization; dynamic-VRAM ownership preserved",
        flush=True,
    )
    return contract


def install_fast_h3_loader_compat() -> bool:
    try:
        import comfy.model_base as model_base
    except Exception:
        return False

    cls = getattr(model_base, "MiniMaxH3", None)
    if cls is None:
        return False
    if getattr(cls, "_longmedia_fasth3_loader_installed", False):
        return True

    original = cls.load_model_weights

    def wrapped(self, sd, unet_prefix="", assign=False):
        # Always reset per-instance FastH3 mutations first.  This is essential
        # when Comfy reuses the same MiniMaxH3 object after the user switches
        # from FastH3 back to a stock FL2VA/Ref2VA/hybrid checkpoint.
        reset = _clear_fast_h3_runtime_state(self)
        try:
            fast_prefix = _strict_fast_h3_prefix(sd, unet_prefix)
            if fast_prefix is None:
                if any(reset.values()):
                    print(
                        "[MiniMaxH3 LongMedia][FastH3 RESET] stock H3 checkpoint selected; "
                        f"restored_adaln={reset['adaln']} removed_vsa_gates={reset['gates']} "
                        f"cleared_contracts={reset['contracts']}; FastH3 runtime disabled",
                        flush=True,
                    )
                return original(self, sd, unet_prefix=unet_prefix, assign=assign)

            _prepare_fast_h3_checkpoint(self, sd, unet_prefix=fast_prefix)
            result = original(self, sd, unet_prefix=unet_prefix, assign=assign)
            contract = getattr(self, "_longmedia_fasth3_contract", None)
            if contract is None:
                raise RuntimeError("[LongMedia][FastH3 POSTCHECK] runtime contract disappeared during load")
            diffusion = self.diffusion_model
            gates = [getattr(b.attn, "vsa_gate", None) for b in diffusion.blocks]
            if len(gates) != 50 or any(g is None for g in gates):
                raise RuntimeError("[LongMedia][FastH3 POSTCHECK] one or more VSA gate modules disappeared during load")
            # Dynamic-safe logical shape verification; still no .weight access.
            for i, gate in enumerate(gates):
                expected = (int(diffusion.blocks[i].attn.heads) * int(diffusion.blocks[i].attn.head_dim),
                            int(diffusion.blocks[i].attn.qkv_proj.in_features))
                got = _expected_weight_shape(gate)
                if got != expected:
                    raise RuntimeError(
                        f"[LongMedia][FastH3 POSTCHECK] VSA gate {i} logical shape={got}, expected={expected}"
                    )
            _install_adaln_lookup_runtime(diffusion, contract)
            print(
                "[MiniMaxH3 LongMedia][FastH3 POSTCHECK] PASS: 50 learned VSA gates loaded; "
                "exact 7-row FastH3 AdaLN lookup installed; weightless/dynamic Linear ABI supported; "
                "runtime contract attached",
                flush=True,
            )
            return result
        except Exception:
            # A failed FastH3 preload must not poison the model instance for the
            # next checkpoint selection.  Roll back every LongMedia-owned patch.
            _clear_fast_h3_runtime_state(self)
            raise

    cls.load_model_weights = wrapped
    cls._longmedia_fasth3_loader_installed = True
    cls._longmedia_fasth3_original_load_model_weights = original
    return True


# Install at custom-node import time, before a diffusion checkpoint can be detected or loaded.
install_fast_h3_model_detection_compat()
install_fast_h3_loader_compat()
