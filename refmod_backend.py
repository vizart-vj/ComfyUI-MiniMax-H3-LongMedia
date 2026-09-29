"""
refmod_backend.py — RefMod file-format/backend core for LongMedia.

Stores and loads compressed reference latents for MiniMax H3 in a standalone
safetensors-based format.  Two modes:

  * FULL — the raw VAE-encoded latent is stored as-is (no compression).
  * COMPRESSED — the latent is average-pooled to a tiny grid, optionally
    refined via Adam/MSE to recover lost detail.

v4 standalone: one ref per .safetensors file, tensor key ``ref_0`` (canonical)
or ``latent`` (legacy compat), metadata under the ``refmod_meta`` header key.
v5 bundle: multiple refs in one file, tensors keyed ``ref_0 … ref_N``, header
metadata ``kind=bundle``, ``_format_version=5``.

MiniMaxH3Mod legacy mode names (``encode``/``training``/``pooled``/
``full``/``Full Reference``/``Compressed Reference``) are normalized to
FULL/COMPRESSED by :func:`normalize_mode`, so files written by
Luisacaotica/ComfyUI-MiniMaxH3Mod load unchanged (both their ``latent``
tensor key and their older ``_format_version`` headers are accepted).

Native block shapes:
  image — [1, 24, 1, H, W]        latent_h, latent_w
  video — [1, 24, T, H, W]        latent_t, latent_h, latent_w
  audio — [1, 32, 2, T]           ref_audio_t

Attribution: design and format compat derived from
Luisacaotica/ComfyUI-MiniMaxH3Mod (MIT License).
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

# ── constants ────────────────────────────────────────────────────────────

META_KEY = "refmod_meta"
_FORMAT_VERSION_STANDALONE = 4
_FORMAT_VERSION_BUNDLE = 5

_VALID_KINDS = ("image", "video", "audio")
_VALID_MODES = ("FULL", "COMPRESSED")

# MiniMaxH3Mod legacy mode names (and pre-rename UI labels) -> canonical mode.
# Reference files in the wild carry ``encode``/``training`` (current upstream)
# and ``full``/``pooled``/``Full Reference``/``Compressed Reference`` (older).
MODE_ALIASES = {
    "full": "FULL",
    "encode": "FULL",
    "full reference": "FULL",
    "compressed": "COMPRESSED",
    "training": "COMPRESSED",
    "pooled": "COMPRESSED",
    "compressed reference": "COMPRESSED",
}


def normalize_mode(mode) -> str:
    """Map any legacy/alternate mode name onto ``FULL`` / ``COMPRESSED``.

    Unknown strings are returned unchanged so the caller's validation still
    rejects them with a clear error.
    """
    if not isinstance(mode, str):
        return mode
    text = mode.strip()
    if text in _VALID_MODES:
        return text
    return MODE_ALIASES.get(text.lower(), text)


# ═══════════════════════════════════════════════════════════════════════════
# A.  Dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RefMod:
    """A compressed or full reference for MiniMax H3.

    Tensor shapes:
      image — [1, 24, 1, H, W]
      video — [1, 24, T, H, W]
      audio — [1, 32, 2, T]
    """

    name: str
    kind: str                    # "image" | "video" | "audio"
    latent: torch.Tensor
    mode: str = "COMPRESSED"     # "FULL" | "COMPRESSED"
    latent_t: int = 1
    latent_h: int = 4
    latent_w: int = 4
    source: str = ""             # "image" | "video" | "manual"
    source_shape: str = ""       # original latent dims as "TxHxW"
    pool: str = "4x4x1"         # pool_t x pool_h x pool_w
    optimize_steps: int = 0
    tags: List[str] = field(default_factory=list)
    description: str = ""
    concept_type: str = "generic"
    sample_rate: int = 32000
    config: Dict = field(default_factory=dict)
    path: str = ""               # path_no_ext this was loaded/saved from
    bundle_index: int = -1       # -1 for standalone; 0..N in bundles

    def __post_init__(self):
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"kind must be one of {_VALID_KINDS}, got {self.kind!r}")
        # Accept MiniMaxH3Mod legacy mode names (encode/training/pooled/...)
        self.mode = normalize_mode(self.mode)
        if self.mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {self.mode!r}")
        if self.kind == "audio":
            if self.latent.ndim != 4 or tuple(self.latent.shape[:3]) != (1, 32, 2) or self.latent.shape[-1] < 1:
                raise ValueError("Audio RefMod must be [1, 32, 2, T] with T >= 1.")
            self.latent_t = self.latent.shape[-1]
            self.latent_h = 0
            self.latent_w = 0
            return
        # image / video — exact visual contract: [1, 24, T, H, W]
        if self.latent.ndim != 5:
            raise ValueError(
                f"{self.kind} RefMod must be 5-D [1, 24, T, H, W], "
                f"got shape {tuple(self.latent.shape)}."
            )
        b, c, t, h, w = self.latent.shape
        if b != 1 or c != 24:
            raise ValueError(
                f"{self.kind} RefMod must be [1, 24, T, H, W] "
                f"(got [{b}, {c}, {t}, {h}, {w}])."
            )
        if t < 1 or h < 1 or w < 1:
            raise ValueError(
                f"{self.kind} RefMod needs positive T/H/W "
                f"(got [{b}, {c}, {t}, {h}, {w}])."
            )
        if self.kind == "image":
            self.latent_t = 1
        else:
            self.latent_t = t
        self.latent_h = h
        self.latent_w = w

    # ── serialization dict ────────────────────────────────────────────

    def metadata_dict(self) -> dict:
        """Metadata dict suitable for JSON-encoding into the safetensors header."""
        meta: Dict = {
            "name": self.name,
            "kind": self.kind,
            "mode": self.mode,
            "latent_t": self.latent_t,
            "latent_h": self.latent_h,
            "latent_w": self.latent_w,
            "source": self.source,
            "source_shape": self.source_shape,
            "pool": self.pool,
            "optimize_steps": self.optimize_steps,
            "tags": self.tags,
            "description": self.description,
            "concept_type": self.concept_type,
            "sample_rate": self.sample_rate,
            "_format_version": _FORMAT_VERSION_STANDALONE,
        }
        if self.config:
            meta["refmod_config"] = json.dumps(self.config)
        return meta


# ═══════════════════════════════════════════════════════════════════════════
# B.  Metadata normalization
# ═══════════════════════════════════════════════════════════════════════════

def normalize_metadata(raw: dict) -> dict:
    """Safely normalize a raw metadata dict to canonical form.

    - Fills missing keys with defaults.
    - Strips unknown keys.
    - Detects bundle vs standalone.
    """
    fmt = raw.get("_format_version", _FORMAT_VERSION_STANDALONE)
    kind = raw.get("kind", "image")
    is_bundle = (fmt == _FORMAT_VERSION_BUNDLE and kind == "bundle")

    out: Dict = {
        "_format_version": fmt,
        "kind": kind,
        "name": raw.get("name", "unnamed"),
        "mode": normalize_mode(raw.get("mode", "FULL")),
        "latent_t": int(raw.get("latent_t", 1)),
        "latent_h": int(raw.get("latent_h", 0)),
        "latent_w": int(raw.get("latent_w", 0)),
        "source": raw.get("source", ""),
        "source_shape": raw.get("source_shape", ""),
        "pool": raw.get("pool", ""),
        "optimize_steps": int(raw.get("optimize_steps", 0)),
        "tags": list(raw.get("tags", [])),
        "description": str(raw.get("description", "") or ""),
        "concept_type": str(raw.get("concept_type", "generic") or "generic"),
        "sample_rate": int(raw.get("sample_rate", 32000)),
        "is_bundle": is_bundle,
    }
    if is_bundle:
        out["members"] = raw.get("members", [])
    if "refmod_config" in raw:
        out["refmod_config"] = raw["refmod_config"]
    return out


# ═══════════════════════════════════════════════════════════════════════════
# C.  Latent utilities
# ═══════════════════════════════════════════════════════════════════════════

def _blur_latent(z: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """Heavy spatial low-pass for strength < 1.0 blending."""
    if z.dim() == 4:
        # audio: [1, 32, 2, T]
        b, c, stereo, t = z.shape
        if t <= 1:
            return z
        flat = z.reshape(b * c * stereo, 1, t).float()
        down = F.adaptive_avg_pool1d(flat, max(1, t // factor))
        return F.interpolate(down, size=t, mode="linear",
                             align_corners=False).reshape_as(z).to(z.dtype)
    if z.dim() != 5:
        raise ValueError(f"Unsupported latent shape: {tuple(z.shape)}")
    t, h, w = z.shape[2], z.shape[3], z.shape[4]
    sh, sw = max(1, h // factor), max(1, w // factor)
    down = F.adaptive_avg_pool3d(z.float(), (t, sh, sw))
    up = F.interpolate(down, size=(t, h, w), mode="trilinear",
                       align_corners=False)
    return up.to(z.dtype)


def aspect_grid(pool_h: int, pool_w: int, aspect: float) -> Tuple[int, int]:
    """Even pool grid dims that match ``aspect`` (h/w) within dial caps.

    Square sources keep the exact dial value; portrait/landscape sources
    are derived from the long edge and aspect ratio.
    """
    long_edge = max(pool_h, pool_w)
    if aspect >= 1.0:
        h, w = long_edge, long_edge / aspect
    else:
        w, h = long_edge, long_edge * aspect
    h = max(2, round(h / 2) * 2)
    w = max(2, round(w / 2) * 2)
    return int(h), int(w)


def pool_latent(z: torch.Tensor, latent_t: int, latent_h: int = 0,
                latent_w: int = 0) -> torch.Tensor:
    """Average-pool a latent to target dims.

    Visual latents are 5-D ``[1, 24, T, H, W]`` and pool across T/H/W —
    ``latent_h``/``latent_w`` must be even (DiT 2×2 patch).  Audio latents
    are 4-D ``[1, 32, 2, T]`` and pool along T only (``latent_h``/
    ``latent_w`` are ignored).  Returns the input unchanged if the dims
    already match.
    """
    if z.ndim == 4:
        # audio: only the time axis pools
        if latent_t < 1:
            raise ValueError(f"pool_t must be >= 1 (got {latent_t})")
        if z.shape[-1] == latent_t:
            return z
        b, c, stereo, t = z.shape
        flat = z.float().reshape(b * c * stereo, 1, t)
        pooled = F.adaptive_avg_pool1d(flat, latent_t)
        return pooled.reshape(b, c, stereo, latent_t).to(z.dtype)
    if z.ndim != 5:
        raise ValueError(
            f"pool_latent expects [1, 24, T, H, W] (visual) or "
            f"[1, 32, 2, T] (audio), got shape {tuple(z.shape)}."
        )
    if z.shape[2] == latent_t and z.shape[3] == latent_h and z.shape[4] == latent_w:
        return z
    if latent_t < 1 or latent_h < 1 or latent_w < 1:
        raise ValueError(
            f"pool targets must be >= 1 (got {latent_t}x{latent_h}x{latent_w})"
        )
    if latent_h % 2 != 0 or latent_w % 2 != 0:
        raise ValueError(f"pool_h/pool_w must be even (got {latent_h}x{latent_w})")
    pooled = F.adaptive_avg_pool3d(z.float(), (latent_t, latent_h, latent_w))
    return pooled.to(z.dtype)


def refine_latent(z_small: torch.Tensor, z_full: torch.Tensor,
                  steps: int = 150, lr: float = 0.02,
                  device: Optional[torch.device] = None) -> torch.Tensor:
    """Model-free Adam/MSE refinement of a compressed latent.

    Optimizes ``z_small`` so its upsampled reconstruction matches
    ``z_full`` — trilinearly for 5-D visual latents, linearly along T for
    4-D audio latents.  Returns refined tensor detached, same shape as
    ``z_small``.
    """
    if steps <= 0:
        return z_small
    if z_small.ndim not in (4, 5) or z_small.ndim != z_full.ndim:
        raise ValueError(
            f"refine_latent expects matching 4-D (audio) or 5-D (visual) "
            f"latents, got {tuple(z_small.shape)} and {tuple(z_full.shape)}."
        )
    device = device or z_full.device
    is_audio = z_small.ndim == 4
    with torch.inference_mode(False), torch.set_grad_enabled(True):
        target = z_full.clone().float().to(device)
        param = nn.Parameter(z_small.clone().float().to(device))
        opt = torch.optim.Adam([param], lr=lr)
        size = tuple(target.shape[1:] if is_audio else target.shape[2:])
        for _ in range(steps):
            opt.zero_grad()
            if is_audio:
                b, c, stereo, t = param.shape
                flat = param.reshape(b * c * stereo, 1, t)
                up = F.interpolate(flat, size=size[-1:], mode="linear",
                                   align_corners=False)
                up = up.reshape(b, c, stereo, size[-1])
            else:
                up = F.interpolate(param, size=size, mode="trilinear",
                                   align_corners=False)
            loss = F.mse_loss(up, target)
            loss.backward()
            opt.step()
        refined = param.detach().to(z_small.dtype)
    return refined


# ═══════════════════════════════════════════════════════════════════════════
# D.  High-level encode / compress
# ═══════════════════════════════════════════════════════════════════════════

def encode_full(latent: torch.Tensor, name: str = "full",
                kind: str = "image", **kwargs) -> RefMod:
    """FULL mode: store the raw VAE latent as-is."""
    return RefMod(name=name, kind=kind, latent=latent, mode="FULL", **kwargs)


def compress_refmod(latent: torch.Tensor, name: str = "compressed",
                    kind: str = "image", pool_t: int = 1,
                    pool_h: int = 4, pool_w: int = 4,
                    refine_steps: int = 0, refine_lr: float = 0.02,
                    **kwargs) -> RefMod:
    """COMPRESSED mode: pool + optional Adam/MSE refinement.

    Visual latents pool across T/H/W (images always collapse to a single
    latent frame); audio latents pool along T only — ``pool_h``/``pool_w``
    are ignored for audio, and the stored ``pool`` string becomes
    ``"{pool_t}x1x1"``.
    """
    if kind == "audio":
        if latent.ndim != 4:
            raise ValueError(
                f"Audio latent must be [1, 32, 2, T], got {tuple(latent.shape)}."
            )
        pooled = pool_latent(latent, pool_t)
        if refine_steps > 0:
            pooled = refine_latent(pooled, latent, steps=refine_steps,
                                   lr=refine_lr)
        return RefMod(name=name, kind=kind, latent=pooled, mode="COMPRESSED",
                      pool=f"{pool_t}x1x1", **kwargs)
    if kind == "image":
        pool_t = 1
    pooled = pool_latent(latent, pool_t, pool_h, pool_w)
    if refine_steps > 0:
        pooled = refine_latent(pooled, latent, steps=refine_steps,
                               lr=refine_lr)
    return RefMod(name=name, kind=kind, latent=pooled, mode="COMPRESSED",
                  pool=f"{pool_t}x{pool_h}x{pool_w}", **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# E.  Native block conversion
# ═══════════════════════════════════════════════════════════════════════════

_H3_SPATIAL_PATCH = 2   # comfy.ldm.minimax.model.patchify_video -> patch_size (1, 2, 2)


def fit_patch_geometry(latent: torch.Tensor) -> torch.Tensor:
    """Trim a visual latent onto the H3 DiT's 2x2 spatial patch grid.

    ``comfy.ldm.minimax.model.patchify_video`` rewrites ``[B, C, T, H, W]`` as
    ``(b, c, t, pt, h, ph, w, pw)`` with ``h = H // 2`` and ``w = W // 2``, then
    reshapes to exactly ``h*ph`` by ``w*pw`` spatial columns. An odd H or W hands
    one row/column to floor-division and the reshape dies inside the host forward
    with an opaque ``shape ... is invalid for input of size N``, nowhere near
    anything RefMod-shaped.

    RefMod sources are arbitrary user media, so their latents do not necessarily
    land on the patch grid. Trimming to the largest even H x W costs at most one
    latent row/column (8 px of source) and puts the packed row math, the native
    descriptor and the host patchify back in agreement.
    """
    if latent.ndim != 5:
        return latent
    h, w = int(latent.shape[-2]), int(latent.shape[-1])
    even_h = h - (h % _H3_SPATIAL_PATCH)
    even_w = w - (w % _H3_SPATIAL_PATCH)
    if even_h == h and even_w == w:
        return latent
    if even_h < _H3_SPATIAL_PATCH or even_w < _H3_SPATIAL_PATCH:
        raise ValueError(
            f"RefMod latent {h}x{w} is smaller than the H3 "
            f"{_H3_SPATIAL_PATCH}x{_H3_SPATIAL_PATCH} spatial patch grid."
        )
    return latent[..., :even_h, :even_w]


def convert_to_native_block(r: RefMod, strength: float = 1.0) -> Optional[Dict]:
    """Convert a RefMod to the native minimax_refs block dict.

    Returns None when ``strength <= 0`` (block should be dropped).

    Block shapes:
      image — {kind, latent_h, latent_w, latent:[1,24,1,H,W]}
      video — {kind, latent_t, latent_h, latent_w, ref_audio_t:0,
               audio_latent:None, latent:[1,24,T,H,W]}
      audio — {kind, ref_audio_t, audio_latent:[1,32,2,T]}

    Visual dims always describe the tensor actually handed to the host, after
    ``fit_patch_geometry`` — never the stored RefMod metadata, which can describe
    a latent that does not fit the DiT patch grid.
    """
    if strength <= 0.0:
        return None
    latent = r.latent
    if strength < 1.0:
        latent = strength * latent + (1.0 - strength) * _blur_latent(latent)

    if r.kind == "audio":
        return {
            "kind": "audio",
            "ref_audio_t": r.latent_t,
            "audio_latent": latent,
        }
    latent = fit_patch_geometry(latent)
    _b, _c, latent_t, latent_h, latent_w = latent.shape
    block: Dict = {
        "kind": r.kind,
        "latent_h": int(latent_h),
        "latent_w": int(latent_w),
        "latent": latent,
    }
    if r.kind == "video":
        block["latent_t"] = int(latent_t)
        block["ref_audio_t"] = 0
        block["audio_latent"] = None
    return block


# ═══════════════════════════════════════════════════════════════════════════
# F.  read_refmod_meta (header-only)
# ═══════════════════════════════════════════════════════════════════════════

def read_refmod_meta(path_no_ext: str) -> Optional[dict]:
    """Read metadata from a .safetensors header without loading tensors.

    Falls back to a legacy ``{path}.json`` sidecar if the header lacks
    ``refmod_meta``.
    """
    try:
        with safe_open(path_no_ext + ".safetensors", framework="pt") as f:
            meta = f.metadata()
        if meta:
            for key in (META_KEY, "audio_refmod_meta"):
                if key in meta:
                    return json.loads(meta[key])
    except Exception:
        pass
    jpath = path_no_ext + ".json"
    if os.path.isfile(jpath):
        try:
            with open(jpath) as fh:
                return json.load(fh)
        except Exception:
            return None
    return None


# ═══════════════════════════════════════════════════════════════════════════
# G.  Atomic v4 standalone save/load
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_tensor_key(r: RefMod) -> str:
    """Canonical tensor key for a standalone ref."""
    return "ref_0"


def save_refmod(r: RefMod, path_no_ext: str) -> str:
    """Atomically save a standalone RefMod to ``{path}.safetensors``.

    Uses a temp file + ``os.replace`` for crash safety.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path_no_ext)) or ".", exist_ok=True)
    meta = r.metadata_dict()
    key = _resolve_tensor_key(r)
    destination = path_no_ext + ".safetensors"
    fd, temporary = tempfile.mkstemp(
        prefix=".refmod-", suffix=".tmp",
        dir=os.path.dirname(os.path.abspath(destination)) or ".",
    )
    os.close(fd)
    try:
        save_file(
            {key: r.latent.detach().cpu().contiguous().clone()},
            temporary,
            metadata={META_KEY: json.dumps(meta)},
        )
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    r.path = path_no_ext
    return destination


def load_refmod(path_no_ext: str, device: str = "cpu") -> RefMod:
    """Load a standalone RefMod from ``{path}.safetensors``.

    Accepts both legacy ``latent`` tensor key and canonical ``ref_0``.
    """
    meta = read_refmod_meta(path_no_ext)
    if not isinstance(meta, dict):
        raise ValueError(
            f"{path_no_ext}.safetensors has no RefMod metadata "
            f"(header key '{META_KEY}' missing or invalid)."
        )
    if meta.get("kind") == "bundle":
        raise ValueError(
            "This is a RefMod bundle. Use load_bundle() instead."
        )
    nmeta = normalize_metadata(meta)
    if not meta.get("name"):
        # MiniMaxH3Mod files written without a name fall back to the file
        nmeta["name"] = os.path.basename(path_no_ext)
    # Load tensors — accept both legacy 'latent' and canonical 'ref_0'
    tensors = load_file(path_no_ext + ".safetensors", device=device)
    if "ref_0" in tensors:
        latent = tensors["ref_0"].clone()
    elif "latent" in tensors:
        latent = tensors["latent"].clone()
    else:
        raise ValueError(
            f"No recognized tensor key ('ref_0' or 'latent') in "
            f"{path_no_ext}.safetensors."
        )
    # Infer shape dims from tensor if metadata left them at defaults
    kind = nmeta["kind"]
    if kind == "audio":
        nmeta["latent_t"] = latent.shape[-1]
        nmeta["latent_h"] = 0
        nmeta["latent_w"] = 0
    elif latent.ndim == 5:
        nmeta["latent_t"] = latent.shape[2]
        nmeta["latent_h"] = latent.shape[3]
        nmeta["latent_w"] = latent.shape[4]

    return RefMod(
        name=nmeta["name"],
        kind=nmeta["kind"],
        latent=latent,
        mode=nmeta["mode"],
        latent_t=nmeta["latent_t"],
        latent_h=nmeta["latent_h"],
        latent_w=nmeta["latent_w"],
        source=nmeta["source"],
        source_shape=nmeta["source_shape"],
        pool=nmeta["pool"],
        optimize_steps=nmeta["optimize_steps"],
        tags=nmeta["tags"],
        description=nmeta["description"],
        concept_type=nmeta["concept_type"],
        sample_rate=nmeta["sample_rate"],
        config=json.loads(nmeta["refmod_config"]) if "refmod_config" in nmeta else {},
        path=path_no_ext,
    )


# ═══════════════════════════════════════════════════════════════════════════
# H.  v5 bundle save/load
# ═══════════════════════════════════════════════════════════════════════════

def _validate_bundle_refs(refs: List[RefMod]) -> None:
    if not refs:
        raise ValueError("Bundle must contain at least one reference.")
    if len(refs) > 256:
        raise ValueError("Bundle cannot contain more than 256 references.")


def save_bundle(refs: List[RefMod], name: str,
                path_no_ext: str) -> str:
    """Save multiple RefMods as a v5 bundle in one .safetensors file.

    Tensors keyed ``ref_0, ref_1, ...``; header metadata ``kind=bundle``.
    """
    _validate_bundle_refs(refs)
    metadata = {
        "_format_version": _FORMAT_VERSION_BUNDLE,
        "kind": "bundle",
        "name": name,
        "members": [r.metadata_dict() for r in refs],
    }
    tensors = {}
    for i, r in enumerate(refs):
        tensors[f"ref_{i}"] = r.latent.detach().cpu().contiguous().clone()

    directory = os.path.dirname(os.path.abspath(path_no_ext)) or "."
    os.makedirs(directory, exist_ok=True)
    destination = path_no_ext + ".safetensors"
    fd, temporary = tempfile.mkstemp(
        prefix=".refmod-", suffix=".tmp", dir=directory,
    )
    os.close(fd)
    try:
        save_file(tensors, temporary, metadata={META_KEY: json.dumps(metadata)})
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def load_bundle(path_no_ext: str, selection: str = "All",
                visual_strength: float = 1.0,
                audio_strength: float = 1.0) -> List[Tuple[RefMod, float]]:
    """Load a v5 bundle and return ``[(RefMod, strength), ...]``.

    ``selection``: ``"All"``, ``"Visual"``, or ``"Audio"``.
    """
    if selection not in ("All", "Visual", "Audio"):
        raise ValueError("selection must be 'All', 'Visual', or 'Audio'.")
    if not (0 <= visual_strength <= 1 and math.isfinite(visual_strength)):
        raise ValueError("visual_strength must be in [0, 1].")
    if not (0 <= audio_strength <= 1 and math.isfinite(audio_strength)):
        raise ValueError("audio_strength must be in [0, 1].")

    raw_meta = read_refmod_meta(path_no_ext)
    if not isinstance(raw_meta, dict):
        raise ValueError(f"{path_no_ext} has no bundle metadata.")
    nmeta = normalize_metadata(raw_meta)
    if not nmeta["is_bundle"]:
        raise ValueError(f"{path_no_ext} is not a v5 bundle.")

    members_meta = nmeta.get("members")
    if not isinstance(members_meta, list) or not members_meta or len(members_meta) > 256:
        raise ValueError("A RefMod bundle must contain 1-256 references.")
    for i, mm in enumerate(members_meta):
        if not isinstance(mm, dict) or mm.get("kind") not in _VALID_KINDS:
            raise ValueError(f"Invalid RefMod bundle member {i}.")
    loaded: List[Tuple[RefMod, float]] = []

    with safe_open(path_no_ext + ".safetensors", framework="pt",
                   device="cpu") as fh:
        for i, mm in enumerate(members_meta):
            is_audio = mm.get("kind") == "audio"
            strength = audio_strength if is_audio else visual_strength
            if strength <= 0:
                continue
            if selection == "Visual" and is_audio:
                continue
            if selection == "Audio" and not is_audio:
                continue

            key = f"ref_{i}"
            if key not in fh.keys():
                raise ValueError(f"Missing tensor '{key}' in bundle.")
            latent = fh.get_tensor(key).clone()

            # Validate shape
            if is_audio:
                valid = (latent.ndim == 4 and tuple(latent.shape[:3]) == (1, 32, 2)
                         and latent.shape[-1] > 0)
            else:
                valid = (latent.ndim == 5 and tuple(latent.shape[:2]) == (1, 24)
                         and all(n > 0 for n in latent.shape[2:])
                         and all(n % 2 == 0 for n in latent.shape[-2:]))
                if valid and mm.get("kind") == "image":
                    valid = latent.shape[2] == 1
            if not valid:
                raise ValueError(f"Invalid tensor layout for bundle member {i}.")

            nmm = normalize_metadata(mm)
            if not is_audio:
                nmm["latent_t"] = latent.shape[2]
                nmm["latent_h"] = latent.shape[3]
                nmm["latent_w"] = latent.shape[4]

            mod = RefMod(
                name=nmm["name"],
                kind=nmm["kind"],
                latent=latent,
                mode=nmm["mode"],
                latent_t=nmm["latent_t"],
                latent_h=nmm["latent_h"],
                latent_w=nmm["latent_w"],
                source=nmm["source"],
                source_shape=nmm["source_shape"],
                pool=nmm["pool"],
                optimize_steps=nmm["optimize_steps"],
                tags=nmm["tags"],
                description=nmm["description"],
                concept_type=nmm["concept_type"],
                sample_rate=nmm["sample_rate"],
                config=json.loads(nmm["refmod_config"]) if "refmod_config" in nmm else {},
                path=path_no_ext,
                bundle_index=i,
            )
            loaded.append((mod, strength))

    return loaded
