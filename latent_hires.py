"""Integrated MiniMax H3 learned latent upscaler.

Architecture and normalization statistics are derived from the Apache-2.0
LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler project.  This copy is kept
small and sampler-oriented: 3D model only, multiplier mode only, AV handling
is owned by LongMedia.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F
import folder_paths

_FOLDER: Final[str] = "latent_upscale_models"
if _FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(_FOLDER, os.path.join(folder_paths.models_dir, _FOLDER))

LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
]


def scan_models() -> list[str]:
    paths = folder_paths.get_folder_paths(_FOLDER)
    files: list[str] = []
    for root in paths:
        for ext in ("*.safetensors", "*.pth"):
            files.extend(glob.glob(os.path.join(root, ext)))
    names = sorted({os.path.basename(x) for x in files})
    return names


def _find_model(name: str) -> str:
    for root in folder_paths.get_folder_paths(_FOLDER):
        p = os.path.join(root, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Latent upscaler model not found: {name}")


def _norm(ch: int) -> nn.Module:
    return nn.GroupNorm(32, ch)


def _zero(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels: int, emb_channels: int, dropout: float = 0.0):
        super().__init__()
        self.in_layers = nn.Sequential(_norm(channels), nn.SiLU(), nn.Conv3d(channels, channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * channels))
        self.out_norm = _norm(channels)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(dropout), _zero(nn.Conv3d(channels, channels, 3, padding=1)))

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        e = self.emb_layers(emb).to(h.dtype)
        while e.ndim < h.ndim:
            e = e[..., None]
        scale, shift = e.chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        return x + self.out_layers(h)


class TemporalConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.norm = _norm(channels)
        self.dwconv = nn.Conv3d(channels, channels, (kernel_size, 1, 1), padding=(pad, 0, 0), groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, 1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm(x))
        return x + self.pwconv(self.dwconv(h))


class LatentResizer3D(nn.Module):
    def __init__(self, in_channels: int = 24, in_blocks: int = 12, out_blocks: int = 12,
                 channels: int = 512, dropout: float = 0.1, temporal_every: int = 2,
                 temporal_kernel: int = 5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        emb_dim = 64
        self.embed = nn.Sequential(nn.Linear(1, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            self.in_blocks.append(ResBlockEmb3D(channels, emb_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(TemporalConv(channels, temporal_kernel))
        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            self.out_blocks.append(ResBlockEmb3D(channels, emb_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(TemporalConv(channels, temporal_kernel))
        self.norm_out = _norm(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def _segment(self, x: torch.Tensor, scale: float, size: tuple[int, int, int]) -> torch.Tensor:
        emb = self.embed(torch.tensor([[scale - 1.0]], dtype=x.dtype, device=x.device))
        x = self.conv_in(x)
        for b in self.in_blocks:
            x = b(x, emb.expand(x.shape[0], -1)) if isinstance(b, ResBlockEmb3D) else b(x)
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        for b in self.out_blocks:
            x = b(x, emb.expand(x.shape[0], -1)) if isinstance(b, ResBlockEmb3D) else b(x)
        return self.conv_out(F.silu(self.norm_out(x)))

    def forward(self, x: torch.Tensor, scale: float, target_hw: tuple[int, int]) -> torch.Tensor:
        t = int(x.shape[2])
        target = (t, int(target_hw[0]), int(target_hw[1]))
        kernel = 5
        for b in self.in_blocks:
            if isinstance(b, TemporalConv):
                kernel = int(b.dwconv.weight.shape[2]); break
        overlap = kernel // 2
        chunk_t = 16
        if t <= chunk_t:
            return self._segment(x, scale, target)
        outs: list[torch.Tensor] = []
        start = 0
        while start < t:
            lo = max(0, start - overlap)
            hi = min(t, start + chunk_t + overlap)
            seg = self._segment(x[:, :, lo:hi], scale, (hi - lo, target[1], target[2]))
            s0 = start - lo
            s1 = s0 + min(chunk_t, t - start)
            outs.append(seg[:, :, s0:s1])
            start += chunk_t
        return torch.cat(outs, dim=2)




class _LegacyResBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.silu(x + r)


class _LegacySpatialResampler(nn.Module):
    def __init__(self, mid_channels: int, numerator: int, denominator: int, kernel_size: int = 5):
        super().__init__()
        self.numerator = int(numerator)
        self.denominator = int(denominator)
        self.conv = nn.Conv2d(mid_channels, (self.numerator ** 2) * mid_channels, 3, padding=1)
        # Keep the exact checkpoint key used by LTX-derived weights.
        k = torch.tensor([1., 4., 6., 4., 1.])
        k2 = (k[:, None] @ k[None, :])
        k2 = (k2 / k2.sum()).float()
        self.blur_down = nn.Module()
        self.blur_down.register_buffer('kernel', k2[None, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        y = self.conv(y)
        y = F.pixel_shuffle(y, self.numerator)
        if self.denominator > 1:
            ch = y.shape[1]
            weight = self.blur_down.kernel.to(device=y.device, dtype=y.dtype).expand(ch, 1, -1, -1)
            y = F.conv2d(y, weight, stride=self.denominator, padding=2, groups=ch)
        h2, w2 = y.shape[-2:]
        return y.reshape(b, t, c, h2, w2).permute(0, 2, 1, 3, 4).contiguous()


class LegacyLTXStyleLatentUpsampler(nn.Module):
    """Compatibility loader for the currently published LBH checkpoint family.

    Those checkpoints use the LTX-style key layout (initial_conv/res_blocks/
    upsampler/post_upsample_res_blocks/final_conv), not the newer emb-conditioned
    LatentResizer3D layout used by the companion node source.
    """
    def __init__(self, in_channels: int, mid_channels: int, num_blocks: int, numerator: int, denominator: int):
        super().__init__()
        self.native_scale = float(numerator) / float(denominator)
        self.initial_conv = nn.Conv3d(in_channels, mid_channels, 3, padding=1)
        self.initial_norm = nn.GroupNorm(32, mid_channels)
        self.res_blocks = nn.ModuleList([_LegacyResBlock3D(mid_channels) for _ in range(num_blocks)])
        self.upsampler = _LegacySpatialResampler(mid_channels, numerator, denominator)
        self.post_upsample_res_blocks = nn.ModuleList([_LegacyResBlock3D(mid_channels) for _ in range(num_blocks)])
        self.final_conv = nn.Conv3d(mid_channels, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.initial_norm(self.initial_conv(x)))
        for block in self.res_blocks:
            x = block(x)
        x = self.upsampler(x)
        for block in self.post_upsample_res_blocks:
            x = block(x)
        return self.final_conv(x)


_CACHE: dict[tuple[str, str], nn.Module] = {}


def _raw_state(path: str) -> dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path, device="cpu")
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    if any(str(k).startswith("upscaler.") for k in sd):
        sd = {str(k)[9:]: v for k, v in sd.items() if str(k).startswith("upscaler.")}
    return sd


def _detect(sd: dict[str, torch.Tensor]) -> dict[str, int | float]:
    conv = sd.get("conv_in.weight")
    channels = int(conv.shape[0]) if conv is not None else 512
    in_channels = int(conv.shape[1]) if conv is not None else 24
    in_ids = {int(m.group(1)) for k in sd for m in [re.match(r"in_blocks\.(\d+)\.in_layers\.", k)] if m}
    out_ids = {int(m.group(1)) for k in sd for m in [re.match(r"out_blocks\.(\d+)\.in_layers\.", k)] if m}
    tk = next((int(v.shape[2]) for k, v in sd.items() if k.endswith("dwconv.weight") and v.ndim == 5), 5)
    has_temporal = any(k.endswith("dwconv.weight") for k in sd)
    return {
        "in_channels": in_channels, "channels": channels,
        "in_blocks": len(in_ids) or 12, "out_blocks": len(out_ids) or 12,
        "temporal_every": 2 if has_temporal else 0, "temporal_kernel": tk,
    }


def _legacy_cfg(sd: dict[str, torch.Tensor]) -> dict[str, int | float]:
    iw = sd.get("initial_conv.weight")
    if iw is None or iw.ndim != 5:
        raise ValueError("LTX-style checkpoint is missing initial_conv.weight")
    in_channels = int(iw.shape[1])
    mid_channels = int(iw.shape[0])
    block_ids = {int(m.group(1)) for k in sd for m in [re.match(r"res_blocks\.(\d+)\.", k)] if m}
    num_blocks = (max(block_ids) + 1) if block_ids else 4
    uw = sd.get("upsampler.conv.weight")
    if uw is None:
        # Some non-rational x2 checkpoints use Sequential index 0.
        uw = sd.get("upsampler.0.weight")
    if uw is None or uw.ndim != 4:
        raise ValueError("LTX-style checkpoint is missing upsampler conv weights")
    ratio = int(uw.shape[0]) / float(mid_channels)
    numerator = int(round(ratio ** 0.5))
    if numerator * numerator * mid_channels != int(uw.shape[0]):
        raise ValueError(f"Unsupported upsampler channel ratio: {tuple(uw.shape)} vs mid={mid_channels}")
    # Published upscaler checkpoints use the LTX rational mapping. For upscale-only
    # checkpoints numerator uniquely identifies the native scale.
    denominator = {2: 1, 3: 2, 4: 1}.get(numerator)
    if denominator is None:
        raise ValueError(f"Unsupported rational numerator {numerator}")
    return {
        "in_channels": in_channels, "mid_channels": mid_channels, "num_blocks": num_blocks,
        "numerator": numerator, "denominator": denominator,
    }


def _get_model(name: str, precision: str) -> nn.Module:
    key = (name, precision)
    if key in _CACHE:
        return _CACHE[key]
    sd = _raw_state(_find_model(name))
    if "initial_conv.weight" in sd:
        cfg = _legacy_cfg(sd)
        model: nn.Module = LegacyLTXStyleLatentUpsampler(**cfg)
        # Sequential LTX checkpoints call this upsampler.0; normalize the one key
        # without touching rational checkpoints that already use upsampler.conv.
        if "upsampler.0.weight" in sd and "upsampler.conv.weight" not in sd:
            sd = dict(sd)
            sd["upsampler.conv.weight"] = sd.pop("upsampler.0.weight")
            if "upsampler.0.bias" in sd:
                sd["upsampler.conv.bias"] = sd.pop("upsampler.0.bias")
        arch = f"legacy_ltx_style native_scale={getattr(model, 'native_scale', 0):.3g}x"
    else:
        cfg = _detect(sd)
        model = LatentResizer3D(**cfg)
        arch = "embedding_3d continuous_scale"
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        sample = list(sd)[:8]
        raise RuntimeError(
            f"Unsupported latent upscaler checkpoint architecture for {name}. "
            f"Detected={arch}; first_keys={sample}. Original error: {e}"
        ) from e
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    model = model.eval().requires_grad_(False).to(dtype=dtype, device="cpu")
    setattr(model, "_longmedia_arch", arch)
    _CACHE[key] = model
    return model


def upscale_video(video: torch.Tensor, model_name: str, scale: float, precision: str,
                  device: torch.device, align_pixels: int = 32) -> torch.Tensor:
    if video.ndim != 5 or int(video.shape[1]) != 24:
        raise ValueError(f"Expected H3 video latent [B,24,T,H,W], got {tuple(video.shape)}")
    scale = float(max(1.0, min(4.0, scale)))
    if scale == 1.0:
        return video
    # H3 VAE spatial factor is 16. Align output pixel dimensions to 32 -> latent H/W even.
    h, w = int(video.shape[-2]), int(video.shape[-1])
    px_h = round((h * 16 * scale) / align_pixels) * align_pixels
    px_w = round((w * 16 * scale) / align_pixels) * align_pixels
    out_h = max(1, int(round(px_h / 16)))
    out_w = max(1, int(round(px_w / 16)))
    effective_scale = ((out_h / h) + (out_w / w)) * 0.5
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    model = _get_model(model_name, precision).to(device=device)
    mean = torch.tensor(LATENTS_MEAN, device=device, dtype=dtype).view(1, 24, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, device=device, dtype=dtype).view(1, 24, 1, 1, 1)
    src = video.to(device=device, dtype=dtype)
    with torch.inference_mode():
        src_n = (src - mean) / std
        if isinstance(model, LegacyLTXStyleLatentUpsampler):
            native = float(model.native_scale)
            if abs(effective_scale - native) > 0.075:
                raise ValueError(
                    f"Checkpoint {model_name} is a fixed {native:g}x LTX-style latent upscaler, "
                    f"but requested effective scale is {effective_scale:.3f}x. "
                    "Use the checkpoint's native scale for exact learned upscaling."
                )
            out = model(src_n)
            if tuple(out.shape[-2:]) != (out_h, out_w):
                # Alignment can move one latent cell; do not silently distort more than that.
                dh, dw = abs(int(out.shape[-2]) - out_h), abs(int(out.shape[-1]) - out_w)
                if dh <= 1 and dw <= 1:
                    out = F.interpolate(out, size=(out.shape[2], out_h, out_w), mode="trilinear", align_corners=False)
                else:
                    raise RuntimeError(f"Upscaler produced {tuple(out.shape[-2:])}, expected {(out_h, out_w)}")
        else:
            out = model(src_n, effective_scale, (out_h, out_w))
        out = out * std + mean
    # Keep checkpoint cached in CPU RAM; free its GPU residency before high-res H3 refine.
    model.to("cpu")
    return out.to(device=video.device, dtype=video.dtype)
