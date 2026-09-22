"""Windows compatibility for very large safetensors files.

The stock ``safetensors.safe_open(...).get_tensor()`` mmap path can terminate
long-running Windows ComfyUI processes with 0xC0000005 while opening very large
MiniMax H3 / Qwen checkpoints.  An earlier LongMedia workaround avoided that by
reading the whole file with ``pread``.  It prevented the crash, but it also made
all checkpoint tensors ordinary RAM allocations, which could consume tens of
GiB before Setup and changed ComfyUI low-VRAM residency/performance.

This compatibility layer keeps the checkpoint file-backed.  For large Windows
safetensors it prefers ComfyUI's own ``load_safetensors`` implementation backed
by ``comfy_aimdo.model_mmap.ModelMMAP`` even when global DynamicVRAM/AIMDO is
not enabled.  That path creates read-only tensor views plus TensorFileSlice
metadata instead of materializing the full checkpoint in host RAM.

If ModelMMAP is unavailable, the last-resort fallback uses safetensors' ``pread``
backend.  That fallback is intentionally noisy because it is RAM-eager; it is
safer than a native access violation but should not be the normal H3 path.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

_LOG = logging.getLogger(__name__)
_THRESHOLD_BYTES = 8 * 1024**3
_PATCH_ATTR = "_longmedia_windows_filebacked_patch_v3"


def _eligible(path: Any) -> bool:
    if os.name != "nt":
        return False
    try:
        filename = os.fspath(path)
    except TypeError:
        return False
    if not filename.lower().endswith((".safetensors", ".sft")):
        return False
    try:
        return os.path.getsize(filename) >= _THRESHOLD_BYTES
    except OSError:
        return False


def _metadata_quant_summary(metadata: Any) -> tuple[bool, int]:
    if not isinstance(metadata, dict):
        return False, 0
    raw = metadata.get("_quantization_metadata")
    if raw is None:
        return False, 0
    try:
        import json
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        layers = parsed.get("layers") if isinstance(parsed, dict) else None
        return True, len(layers) if isinstance(layers, dict) else 0
    except Exception:
        return True, -1


def _annotate(sd: dict[str, Any], filename: str) -> None:
    try:
        import comfy.storage as comfy_storage
        comfy_storage.annotate_state_dict(sd, filename)
    except Exception:
        pass


def _load_filebacked(filename: str, comfy_utils: Any):
    """Use ComfyUI's ModelMMAP/TensorFileSlice loader without enabling AIMDO.

    ``comfy.utils.load_safetensors`` itself is file-backed and does not depend on
    ``aimdo_enabled`` for correctness; the global flag normally only decides
    whether ``load_torch_file`` selects it.  Calling it directly lets legacy
    ModelPatcher users keep the safer ModelMMAP file mapping on Windows too.
    """
    loader = getattr(comfy_utils, "load_safetensors", None)
    if not callable(loader):
        raise RuntimeError("ComfyUI load_safetensors is unavailable")
    sd, metadata = loader(filename)
    if not isinstance(sd, dict):
        raise RuntimeError("ComfyUI load_safetensors returned an invalid state dict")
    _annotate(sd, filename)
    return sd, (dict(metadata) if isinstance(metadata, dict) else {})


def _read_metadata_header(filename: str) -> dict[str, Any]:
    """Read only the small safetensors JSON header; never mmap tensor payloads."""
    import json
    import struct
    try:
        with open(filename, "rb", buffering=0) as fh:
            raw = fh.read(8)
            if len(raw) != 8:
                return {}
            (header_len,) = struct.unpack("<Q", raw)
            if header_len <= 0 or header_len > 100_000_000:
                return {}
            payload = fh.read(int(header_len))
        header = json.loads(payload.decode("utf-8"))
        meta = header.get("__metadata__") if isinstance(header, dict) else None
        return dict(meta) if isinstance(meta, dict) else {}
    except Exception:
        return {}


def _load_raw_fallback(filename: str, device: Any, comfy_utils: Any):
    """Last-resort ordinary-I/O loader for old safetensors bindings.

    This path is intentionally RAM-eager and exists only so ModelMMAP failure
    does not fall back to the crashing native mmap implementation.
    """
    import json
    import math
    import struct
    import torch

    dtype_map = getattr(comfy_utils, "_TYPES", None)
    if not isinstance(dtype_map, dict):
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        for key, attr in (("F8_E4M3", "float8_e4m3fn"), ("F8_E5M2", "float8_e5m2")):
            value = getattr(torch, attr, None)
            if value is not None:
                dtype_map[key] = value

    with open(filename, "rb", buffering=0) as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError("incomplete safetensors header")
        (header_len,) = struct.unpack("<Q", raw)
        if header_len <= 0 or header_len > 100_000_000:
            raise ValueError("invalid safetensors header length")
        header_raw = fh.read(int(header_len))
        header = json.loads(header_raw.decode("utf-8"))
        data_base = 8 + int(header_len)
        sd = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = [int(v) for v in info["data_offsets"]]
            dtype = dtype_map[str(info["dtype"])]
            shape = tuple(int(v) for v in info["shape"])
            if end - start != int(math.prod(shape)) * int(dtype.itemsize):
                raise ValueError(f"invalid tensor range for {name}")
            if start == end:
                sd[name] = torch.empty(shape, dtype=dtype, device=device)
                continue
            fh.seek(data_base + start)
            payload = bytearray(end - start)
            view = memoryview(payload)
            done = 0
            while done < len(payload):
                n = fh.readinto(view[done:])
                if not n:
                    view.release()
                    raise ValueError(f"truncated tensor payload for {name}")
                done += int(n)
            view.release()
            tensor = torch.frombuffer(payload, dtype=dtype).view(shape)
            if getattr(device, "type", str(device)) != "cpu":
                tensor = tensor.to(device=device)
            sd[name] = tensor
    metadata = header.get("__metadata__") if isinstance(header, dict) else None
    return sd, (dict(metadata) if isinstance(metadata, dict) else {})


def _load_pread_fallback(filename: str, device: Any, comfy_utils: Any):
    """Crash-safe fallback when ModelMMAP cannot be created."""
    import safetensors

    kwargs = {"framework": "pt", "device": getattr(device, "type", str(device))}
    # Newer safetensors bindings expose backend="pread" on safe_open.  Prefer
    # it when present; older portable ComfyUI builds may not, in which case use
    # the explicit ordinary-I/O reader above rather than native mmap.
    try:
        ctx = safetensors.safe_open(filename, backend="pread", **kwargs)
    except TypeError:
        return _load_raw_fallback(filename, device, comfy_utils)

    with ctx as f:
        sd = {k: f.get_tensor(k) for k in f.keys()}
        metadata = dict(f.metadata() or {})
    return sd, metadata


def install_windows_large_safetensors_compat() -> bool:
    if os.name != "nt":
        return False
    try:
        import torch
        import comfy.utils as comfy_utils
        import comfy.memory_management as comfy_memory
    except Exception:
        return False

    current = getattr(comfy_utils, "load_torch_file", None)
    if not callable(current) or bool(getattr(current, _PATCH_ATTR, False)):
        return bool(getattr(current, _PATCH_ATTR, False))

    stock_load_torch_file = current
    announced: set[tuple[str, str]] = set()
    announce_lock = threading.Lock()

    def _announce(filename: str, backend: str, metadata: Any, extra: str = "") -> None:
        key = (filename, backend)
        with announce_lock:
            if key in announced:
                return
            announced.add(key)
        quant_present, quant_layers = _metadata_quant_summary(metadata)
        _LOG.info(
            "[MiniMaxH3 LongMedia] Windows large-safetensors %s load: %s "
            "(%.2f GiB); file_backed=%s; quant_metadata=%s; quant_layers=%s%s",
            backend,
            os.path.basename(filename),
            os.path.getsize(filename) / 1024**3,
            backend == "ModelMMAP",
            bool(quant_present),
            quant_layers,
            extra,
        )

    def patched_load_torch_file(
        ckpt: Any,
        safe_load: bool = False,
        device: Any = None,
        return_metadata: bool = False,
    ):
        if not _eligible(ckpt):
            return stock_load_torch_file(
                ckpt,
                safe_load=safe_load,
                device=device,
                return_metadata=return_metadata,
            )

        # If ComfyUI DynamicVRAM is active, stock load_torch_file already selects
        # load_safetensors()/ModelMMAP and owns the full lifecycle.  Do not wrap it.
        if bool(getattr(comfy_memory, "aimdo_enabled", False)):
            return stock_load_torch_file(
                ckpt,
                safe_load=safe_load,
                device=device,
                return_metadata=return_metadata,
            )

        filename = os.fspath(ckpt)
        target_device = torch.device("cpu") if device is None else torch.device(device)
        # The file-backed compatibility path is CPU-backed by design.  Unusual
        # explicit CUDA loads keep ComfyUI's stock behavior.
        if target_device.type != "cpu":
            return stock_load_torch_file(
                ckpt,
                safe_load=safe_load,
                device=device,
                return_metadata=return_metadata,
            )

        try:
            sd, metadata = _load_filebacked(filename, comfy_utils)
            _announce(
                filename,
                "ModelMMAP",
                metadata,
                "; core_dynamic_vram=False; TensorFileSlice=True",
            )
        except Exception as mmap_exc:
            _LOG.warning(
                "[MiniMaxH3 LongMedia] ModelMMAP compatibility failed for %s: %s: %s; "
                "falling back to RAM-eager pread",
                os.path.basename(filename),
                type(mmap_exc).__name__,
                mmap_exc,
            )
            sd, metadata = _load_pread_fallback(filename, target_device, comfy_utils)
            _annotate(sd, filename)
            _announce(
                filename,
                "pread-fallback",
                metadata,
                "; WARNING=RAM_EAGER",
            )

        return (sd, metadata) if return_metadata else sd

    setattr(patched_load_torch_file, _PATCH_ATTR, True)
    setattr(patched_load_torch_file, "_longmedia_stock_loader", stock_load_torch_file)
    comfy_utils.load_torch_file = patched_load_torch_file
    _LOG.info(
        "[MiniMaxH3 LongMedia] Windows large-safetensors compatibility enabled "
        "for files >= %.0f GiB (file-backed ModelMMAP preferred)",
        _THRESHOLD_BYTES / 1024**3,
    )
    return True


install_windows_large_safetensors_compat()
