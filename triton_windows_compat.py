from __future__ import annotations

import functools
import inspect
import os
import sys
from pathlib import Path
from typing import Any

_PATCH_MARKER = "_longmedia_windows_tcc_paths"


def _normalized(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.normpath(os.fspath(path)))


def _append_unique_paths(existing: list[str], candidates: list[Path], *, prepend: bool) -> list[str]:
    result = [os.fspath(value) for value in existing if value is not None]
    known = {_normalized(value) for value in result}
    additions: list[str] = []
    for path in candidates:
        normalized = _normalized(path)
        if not path.is_dir() or normalized in known:
            continue
        additions.append(os.fspath(path))
        known.add(normalized)
    if prepend:
        return additions + result
    return result + additions


def install_windows_triton_build_compat() -> bool:
    """Patch Triton's Windows JIT build inputs without modifying site-packages.

    Some Triton-Windows builds select the bundled TinyCC but omit TinyCC's own
    CRT/WinAPI headers from the generated command line.  The first Triton kernel
    then fails while compiling ``cuda_utils``/``__triton_launcher`` before any GPU
    kernel is launched.  Embedded Python installations can likewise expose their
    headers while omitting the sibling ``libs`` directory from ``library_dirs``.

    This shim only augments Triton's existing ``_build`` arguments.  Kernel code,
    compiler selection, CUDA selection, caching and launch semantics remain owned
    by Triton.
    """
    if sys.platform != "win32":
        return False

    try:
        import triton
        import triton.runtime.build as triton_build
    except Exception:
        return False

    original = getattr(triton_build, "_build", None)
    if not callable(original):
        return False
    if bool(getattr(original, _PATCH_MARKER, False)):
        return True

    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        return False
    if "include_dirs" not in signature.parameters or "library_dirs" not in signature.parameters:
        return False

    triton_root = Path(triton.__file__).resolve().parent
    tcc_include = triton_root / "runtime" / "tcc" / "include"
    tcc_candidates = [tcc_include / "winapi", tcc_include]

    # Embedded CPython puts Include/ and libs/ next to python.exe.  Triton already
    # discovers Include in the failing configurations we target; add libs as a
    # defensive companion only when a real Python import library exists there.
    python_lib_candidates: list[Path] = []
    py_lib_name = f"python{sys.version_info.major}{sys.version_info.minor}.lib"
    for root in (Path(sys.prefix), Path(sys.base_prefix)):
        candidate = root / "libs"
        if (candidate / py_lib_name).is_file():
            python_lib_candidates.append(candidate)

    # If the wheel does not contain TinyCC headers there is nothing useful for
    # this compatibility shim to do.  Do not mask another Triton installation.
    if not any(path.is_dir() for path in tcc_candidates) and not python_lib_candidates:
        return False

    @functools.wraps(original)
    def patched_build(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs)
        include_dirs = list(bound.arguments.get("include_dirs") or [])
        library_dirs = list(bound.arguments.get("library_dirs") or [])

        # Prepend TinyCC's own headers so C-runtime/WinAPI includes resolve before
        # the CUDA/Python headers that depend on them.
        include_dirs = _append_unique_paths(include_dirs, tcc_candidates, prepend=True)
        library_dirs = _append_unique_paths(library_dirs, python_lib_candidates, prepend=False)

        bound.arguments["include_dirs"] = include_dirs
        bound.arguments["library_dirs"] = library_dirs
        return original(*bound.args, **bound.kwargs)

    setattr(patched_build, _PATCH_MARKER, True)
    setattr(patched_build, "_longmedia_original_build", original)
    triton_build._build = patched_build
    return True
