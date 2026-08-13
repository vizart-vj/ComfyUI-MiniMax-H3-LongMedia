"""Development hot-reload bridge for LongMedia runtime code.

This module is intentionally tiny and stable.  ComfyUI keeps the originally
registered node classes alive, so those classes call :func:`dispatch_latest`
before executing the two runtime-critical entry points.  If package Python
files changed on disk, a fresh private copy of ``nodes.py`` is loaded and the
call is forwarded to the new implementation without restarting ComfyUI.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import sys
import threading
from pathlib import Path

_LOCK = threading.RLock()
_BASELINE = None
_ACTIVE_FINGERPRINT = None
_ACTIVE_MODULE = None
_RELOAD_COUNT = 0

# Support modules whose implementation can materially affect LongMedia runtime.
_SUPPORT_SUFFIXES = (
    ".latent_ops",
    ".media_plan",
    ".temporal_positioning",
    ".vram_tracker",
    ".sol_kernel.sm120",
    ".sol_kernel",
)


def _fingerprint(root: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    for path in sorted(root.rglob("*.py")):
        # Ignore generated caches and this stable bootstrap itself.  The bootstrap
        # must stay resident so it can keep dispatching after files are replaced.
        if "__pycache__" in path.parts or path.name == "dev_hot_reload.py":
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        rel = path.relative_to(root).as_posix().encode("utf-8", "replace")
        h.update(rel)
        h.update(str(st.st_mtime_ns).encode())
        h.update(str(st.st_size).encode())
    return h.hexdigest()


def _reload_support_modules(package: str) -> None:
    for suffix in _SUPPORT_SUFFIXES:
        name = f"{package}{suffix}" if package else suffix.lstrip(".")
        mod = sys.modules.get(name)
        if mod is None:
            continue
        try:
            importlib.reload(mod)
        except Exception as exc:
            print(f"[MiniMaxH3 LongMedia][DEV HOT RELOAD] support reload skipped {name}: {type(exc).__name__}: {exc}", flush=True)


def _load_nodes(nodes_file: str, package: str, fingerprint: str):
    global _RELOAD_COUNT
    _RELOAD_COUNT += 1
    _reload_support_modules(package)

    # A unique private module name avoids replacing the module object ComfyUI
    # originally registered.  Relative imports still resolve through __package__.
    hot_name = f"{package}._longmedia_hot_nodes_{_RELOAD_COUNT}" if package else f"_longmedia_hot_nodes_{_RELOAD_COUNT}"
    spec = importlib.util.spec_from_file_location(hot_name, nodes_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create module spec for {nodes_file}")
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = package
    # Prevent the freshly loaded methods from recursively dispatching themselves.
    mod._LONGMEDIA_HOT_RELOAD_BYPASS = True
    mod._LONGMEDIA_HOT_RELOAD_FINGERPRINT = fingerprint
    sys.modules[hot_name] = mod
    spec.loader.exec_module(mod)
    return mod


def dispatch_latest(nodes_file: str, package: str, class_name: str, method_name: str, instance, *args, **kwargs):
    """Return ``(dispatched, result)``.

    The first call after ComfyUI startup records the current source tree as the
    baseline and executes the already-loaded implementation.  Once any package
    ``.py`` file changes, all future calls are routed through the latest private
    module until another change is detected.
    """
    global _BASELINE, _ACTIVE_FINGERPRINT, _ACTIVE_MODULE

    if os.environ.get("LONGMEDIA_DEV_HOT_RELOAD", "1").strip().lower() in {"0", "false", "off", "no"}:
        return False, None

    root = Path(nodes_file).resolve().parent
    fp = _fingerprint(root)
    with _LOCK:
        if _BASELINE is None:
            _BASELINE = fp
            print(
                "[MiniMaxH3 LongMedia][DEV HOT RELOAD] armed; replace package .py files and queue again — ComfyUI restart not required",
                flush=True,
            )
            return False, None

        if fp != _BASELINE and fp != _ACTIVE_FINGERPRINT:
            try:
                mod = _load_nodes(nodes_file, package, fp)
            except Exception as exc:
                print(
                    f"[MiniMaxH3 LongMedia][DEV HOT RELOAD] reload FAILED; keeping previous runtime: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                return (_ACTIVE_MODULE is not None), (
                    getattr(getattr(_ACTIVE_MODULE, class_name), method_name)(instance, *args, **kwargs)
                    if _ACTIVE_MODULE is not None else None
                )
            _ACTIVE_MODULE = mod
            _ACTIVE_FINGERPRINT = fp
            print(
                f"[MiniMaxH3 LongMedia][DEV HOT RELOAD] loaded revision #{_RELOAD_COUNT} fingerprint={fp[:10]}",
                flush=True,
            )

        mod = _ACTIVE_MODULE

    if mod is None:
        return False, None
    cls = getattr(mod, class_name)
    method = getattr(cls, method_name)
    return True, method(instance, *args, **kwargs)
