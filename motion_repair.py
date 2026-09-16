"""Latent-native selective temporal repair for MiniMax H3.

This module is intentionally ComfyUI-free.  It analyzes the final H3 video
latent for temporally overloaded regions, builds a locally dilated H3-valid
latent timeline, and supplies exact recovery indices so callers can re-sample
only those regions and restore untouched tokens bit-for-bit.

The implementation is independent/clean-room.  It uses MiniMax H3's public
(1, 4, 4, 4, 4) temporal token clock and does not copy code from third-party
motion-repair extensions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

try:
    from .latent_ops import FPS, AUDIO_LATENT_FPS, audio_latent_t, frame_count_from_video_t, video_latent_t
except ImportError:  # pragma: no cover - standalone audit execution
    from latent_ops import FPS, AUDIO_LATENT_FPS, audio_latent_t, frame_count_from_video_t, video_latent_t


FRAME_PATTERN = (1, 4, 4, 4, 4)


@dataclass(frozen=True)
class MotionRepairPreset:
    threshold_z: float
    grow_tokens: int
    max_hold: int
    max_dilation: float
    denoise_fraction: float
    max_steps: int
    d1_weight: float
    d3_weight: float
    texture_weight: float


PRESETS: dict[str, MotionRepairPreset] = {
    "auto": MotionRepairPreset(
        threshold_z=1.85,
        grow_tokens=1,
        max_hold=3,
        max_dilation=1.40,
        denoise_fraction=0.40,
        max_steps=5,
        d1_weight=0.20,
        d3_weight=0.80,
        texture_weight=0.00,
    ),
    "fluid": MotionRepairPreset(
        # Fluid is a SENSITIVITY preset, not a stronger rewrite.  Thin spray,
        # water, hair and cloth need an easier trigger but a gentler second pass.
        threshold_z=1.15,
        grow_tokens=1,
        max_hold=3,
        max_dilation=1.40,
        denoise_fraction=0.35,
        max_steps=4,
        d1_weight=0.25,
        d3_weight=0.40,
        texture_weight=0.35,
    ),
    "strong": MotionRepairPreset(
        threshold_z=0.80,
        grow_tokens=2,
        max_hold=4,
        max_dilation=1.70,
        denoise_fraction=0.50,
        max_steps=8,
        d1_weight=0.30,
        d3_weight=0.70,
        texture_weight=0.00,
    ),
}


def normalize_motion_repair_mode(mode: str | None) -> str:
    value = str(mode or "off").strip().lower()
    if value not in ("off", *PRESETS.keys()):
        return "off"
    return value


def preset_for(mode: str | None) -> MotionRepairPreset | None:
    return PRESETS.get(normalize_motion_repair_mode(mode))


def _token_spans(latent_t: int) -> tuple[list[tuple[int, int]], int]:
    if latent_t < 2 or (latent_t - 2) % 5:
        raise ValueError(f"MiniMax H3 video latent time must be 5*k+2, got {latent_t}.")
    spans: list[tuple[int, int]] = []
    frame = 0
    for index in range(int(latent_t)):
        width = int(FRAME_PATTERN[index % len(FRAME_PATTERN)])
        spans.append((frame, frame + width))  # end-exclusive
        frame += width
    expected = frame_count_from_video_t(int(latent_t))
    if frame != expected:
        raise RuntimeError(f"H3 token clock mismatch: latent_t={latent_t} maps {frame} frames, expected {expected}.")
    return spans, frame


def _frame_to_token(latent_t: int) -> torch.Tensor:
    spans, frame_count = _token_spans(latent_t)
    out = torch.empty((frame_count,), dtype=torch.long)
    for token_index, (a, b) in enumerate(spans):
        out[a:b] = int(token_index)
    return out


def _phase_equalize(profile: torch.Tensor) -> torch.Tensor:
    """Remove the repeating 1/4-frame token phase bias without amplifying zeros."""
    p = profile.detach().float().clone()
    if p.numel() < 3:
        return p
    global_mean = p.mean().clamp_min(1e-8)
    for phase in range(5):
        idx = torch.arange(phase, p.numel(), 5, device=p.device)
        if idx.numel() == 0:
            continue
        mean = p[idx].mean()
        if float(mean) > 1e-8:
            p[idx] *= global_mean / mean
    return p


def _robust_positive_z(profile: torch.Tensor) -> torch.Tensor:
    p = profile.detach().float()
    median = p.median()
    mad = (p - median).abs().median()
    # A small mean-derived floor avoids exploding an almost-static clip.
    scale = torch.maximum(mad * 1.4826, p.abs().mean() * 0.08).clamp_min(1e-6)
    return ((p - median) / scale).clamp_min_(0.0)


def _difference_profile(x: torch.Tensor, order: int) -> torch.Tensor:
    t = int(x.shape[2])
    if t <= order:
        return torch.zeros((t,), dtype=torch.float32, device=x.device)
    d = torch.diff(x, n=int(order), dim=2).abs().mean(dim=(0, 1, 3, 4))
    left = order // 2
    right = t - int(d.numel()) - left
    if int(d.numel()) == 0:
        return torch.zeros((t,), dtype=torch.float32, device=x.device)
    parts = []
    if left > 0:
        parts.append(d[:1].expand(left))
    parts.append(d)
    if right > 0:
        parts.append(d[-1:].expand(right))
    return torch.cat(parts, dim=0)


def _texture_motion_profile(x: torch.Tensor) -> torch.Tensor:
    """Temporal motion of spatial high frequencies; useful for spray/water/cloth."""
    b, c, t, h, w = x.shape
    if t < 2:
        return torch.zeros((t,), dtype=torch.float32, device=x.device)
    flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    low = F.avg_pool2d(flat, kernel_size=3, stride=1, padding=1)
    hp = (flat - low).reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    return _difference_profile(hp, 1)


def _grow_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    out = mask.clone().bool()
    for _ in range(max(0, int(radius))):
        left = F.pad(out[:-1], (1, 0), value=False)
        right = F.pad(out[1:], (0, 1), value=False)
        out |= left | right
    return out




def _macrocell_repair_weights(risk_mask: torch.Tensor, *, context_groups: int = 1,
                              context_weight: float = 0.25) -> torch.Tensor:
    """Expand token risk to atomic H3 17-frame macrocells.

    H3's temporal VAE clock is (1,4,4,4,4): five latent tokens form one
    17-pixel-frame macrocell whose first token is a singleton anchor.  Mixing
    an original anchor with regenerated neighbours produces a visible flash
    every 17 frames.  A macrocell must therefore be authored atomically.

    Core cells touched by the oracle get weight 1.0.  One neighbouring cell on
    either side is sampled gently to move the seam away from the overloaded
    motion while keeping the outer transition close to the original take.
    """
    risk = torch.as_tensor(risk_mask, dtype=torch.bool).cpu()
    t = int(risk.numel())
    out = torch.zeros((t,), dtype=torch.float32)
    if t == 0 or not bool(risk.any()):
        return out
    group_count = (t + 4) // 5
    core_groups = sorted({int(i) // 5 for i in torch.nonzero(risk, as_tuple=False).flatten().tolist()})
    ctx = max(0, int(context_groups))
    cw = max(0.0, min(1.0, float(context_weight)))
    for g in core_groups:
        a, b = 5 * g, min(t, 5 * g + 5)
        out[a:b] = 1.0
    if ctx > 0 and cw > 0.0:
        for g in core_groups:
            for q in range(max(0, g - ctx), min(group_count, g + ctx + 1)):
                if q in core_groups:
                    continue
                a, b = 5 * q, min(t, 5 * q + 5)
                out[a:b] = torch.maximum(out[a:b], torch.full((b - a,), cw))
    return out

def _contiguous_spans(mask: torch.Tensor, token_spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    last: int | None = None
    for i, active in enumerate(mask.detach().cpu().tolist()):
        if active and start is None:
            start = i
        if active:
            last = i
        elif start is not None and last is not None:
            spans.append((token_spans[start][0], token_spans[last][1]))
            start = last = None
    if start is not None and last is not None:
        spans.append((token_spans[start][0], token_spans[last][1]))
    return spans


def analyze_motion(video: torch.Tensor, mode: str, *, protected_prefix_frames: int = 0,
                   protect_last_token: bool = True) -> dict[str, Any]:
    """Return a robust per-token overload mask and integer local hold counts."""
    preset = preset_for(mode)
    if preset is None:
        return {"enabled": False, "mode": "off", "reason": "disabled"}
    if video.ndim != 5 or int(video.shape[1]) != 24:
        raise ValueError(f"Motion Repair expected H3 video [B,24,T,H,W], got {tuple(video.shape)}")

    latent_t = int(video.shape[2])
    token_spans, frame_count = _token_spans(latent_t)
    if latent_t < 7:
        return {
            "enabled": True,
            "mode": normalize_motion_repair_mode(mode),
            "applied": False,
            "reason": "clip_too_short",
            "source_latent_t": latent_t,
            "source_frames": frame_count,
        }

    # Analyze a bounded spatial proxy.  Motion evidence remains temporal, while
    # 32x32 keeps the oracle cheap even on high-resolution latents.
    x = video.detach().float()
    h, w = int(x.shape[-2]), int(x.shape[-1])
    if h > 32 or w > 32:
        proxy = x.permute(0, 2, 1, 3, 4).reshape(-1, x.shape[1], h, w)
        proxy = F.adaptive_avg_pool2d(proxy, (min(32, h), min(32, w)))
        x = proxy.reshape(video.shape[0], latent_t, video.shape[1], proxy.shape[-2], proxy.shape[-1]).permute(0, 2, 1, 3, 4)

    d1 = _phase_equalize(_difference_profile(x, 1))
    d3 = _phase_equalize(_difference_profile(x, 3))
    tex = _phase_equalize(_texture_motion_profile(x)) if preset.texture_weight > 0 else torch.zeros_like(d1)
    z1 = _robust_positive_z(d1)
    z3 = _robust_positive_z(d3)
    zt = _robust_positive_z(tex) if preset.texture_weight > 0 else torch.zeros_like(z1)
    score = preset.d1_weight * z1 + preset.d3_weight * z3 + preset.texture_weight * zt

    raw_mask = score >= float(preset.threshold_z)
    # A completely quiet clip should not acquire a repair just because MAD is tiny.
    activity = float(d1.mean().detach().cpu())
    peak = float(score.max().detach().cpu()) if score.numel() else 0.0
    if activity <= 1e-7 or peak < float(preset.threshold_z):
        raw_mask.zero_()

    mask = _grow_mask(raw_mask, preset.grow_tokens)
    protected_prefix = max(0, min(frame_count, int(protected_prefix_frames)))
    if protected_prefix > 0:
        for token_index, (_a, b) in enumerate(token_spans):
            if b <= protected_prefix:
                mask[token_index] = False
    # Keep full boundary macrocells stable, not only the literal first/last token.
    # Single repaired context tokens near the start/end can create flashes or glow
    # blooms because neighbouring temporal anchors stay original.
    boundary_groups = 2 if latent_t >= 27 else 1
    head_boundary_tokens = min(latent_t, max(1, boundary_groups * 5))
    last_macrocell_start = ((latent_t - 1) // 5) * 5 if latent_t > 0 else 0
    tail_boundary_start = max(0, last_macrocell_start - ((boundary_groups - 1) * 5))
    if mask.numel():
        mask[:head_boundary_tokens] = False
        if protect_last_token:
            mask[tail_boundary_start:] = False

    # v0.6.9-r3: never splice old/new tokens inside one 17-frame H3 macrocell.
    # The oracle mask still controls where time is dilated; repair_weight controls
    # which complete macrocells the second pass is allowed to author.
    repair_weight = _macrocell_repair_weights(mask, context_groups=1, context_weight=0.25)
    token_spans_local = token_spans
    # Preserve complete boundary macrocells. Repairing only the neighbour of a
    # boundary anchor is visually worse than leaving a little smear untouched.
    repair_weight[:head_boundary_tokens] = 0.0
    if protected_prefix > 0:
        for g0 in range(0, latent_t, 5):
            g1 = min(latent_t, g0 + 5)
            fa = token_spans_local[g0][0]
            fb = token_spans_local[g1 - 1][1]
            if fa < protected_prefix:
                repair_weight[g0:g1] = 0.0
    if protect_last_token and latent_t:
        repair_weight[tail_boundary_start:latent_t] = 0.0

    if not bool(mask.any()) or not bool((repair_weight > 0).any()):
        return {
            "enabled": True,
            "mode": normalize_motion_repair_mode(mode),
            "applied": False,
            "reason": "no_smear_risk_detected",
            "source_latent_t": latent_t,
            "source_frames": frame_count,
            "peak_score": peak,
            "threshold_z": float(preset.threshold_z),
            "mean_motion": activity,
            "risk_tokens": 0,
        }

    holds = torch.ones((latent_t,), dtype=torch.long)
    severity = score.detach().cpu() / max(float(preset.threshold_z), 1e-6)
    mask_cpu = mask.detach().cpu()
    for i in range(latent_t):
        if not bool(mask_cpu[i]):
            continue
        # 1x = unchanged, 2x/3x/4x = progressively more temporal room.
        hold = 2 + int(max(0.0, math.floor(float(severity[i]) - 1.6)))
        holds[i] = min(int(preset.max_hold), max(2, hold))

    # Enforce a strict PIXEL-time cost ceiling. H3 tokens have non-uniform
    # (1,4,4,4,4) frame spans, so a token-count cap is not a duration cap.
    # Snap the ceiling DOWN to the nearest legal 17*k+5 target, then peel
    # extra holds from the least severe repaired tokens first.
    raw_limit = max(frame_count, int(math.floor(frame_count * float(preset.max_dilation))))
    max_legal_frames = 5 + 17 * max(0, (raw_limit - 5) // 17)
    max_legal_frames = max(frame_count, max_legal_frames)
    widths = [int(b - a) for a, b in token_spans]

    def _expanded_frames() -> int:
        return sum(widths[i] * int(holds[i]) for i in range(latent_t))

    order = sorted(
        (i for i in range(latent_t) if int(holds[i]) > 1),
        key=lambda i: float(severity[i]),
    )
    while _expanded_frames() > max_legal_frames and order:
        changed = False
        for i in order:
            if int(holds[i]) > 1 and _expanded_frames() > max_legal_frames:
                holds[i] -= 1
                changed = True
        if not changed:
            break

    if _expanded_frames() <= frame_count:
        return {
            "enabled": True,
            "mode": normalize_motion_repair_mode(mode),
            "applied": False,
            "reason": "dilation_budget_no_legal_expansion",
            "source_latent_t": latent_t,
            "source_frames": frame_count,
            "peak_score": peak,
            "threshold_z": float(preset.threshold_z),
            "mean_motion": activity,
            "risk_tokens": int(mask_cpu.sum().item()),
            "risk_spans_frames": _contiguous_spans(mask_cpu, token_spans),
        }

    return {
        "enabled": True,
        "mode": normalize_motion_repair_mode(mode),
        "applied": True,
        "reason": "repair_requested",
        "source_latent_t": latent_t,
        "source_frames": frame_count,
        "score": score.detach().cpu(),
        "risk_mask": mask_cpu,
        "repair_weight": repair_weight,
        "repair_mask": (repair_weight > 0),
        "holds": holds,
        "risk_tokens": int(mask_cpu.sum().item()),
        "repair_tokens": int((repair_weight > 0).sum().item()),
        "macrocell_atomic": True,
        "risk_spans_frames": _contiguous_spans(mask_cpu, token_spans),
        "peak_score": peak,
        "threshold_z": float(preset.threshold_z),
        "mean_motion": activity,
        "denoise_fraction": float(preset.denoise_fraction),
        "max_steps": int(preset.max_steps),
        "max_dilation": float(preset.max_dilation),
    }


def _legal_frame_ceil(frame_count: int) -> int:
    value = max(5, int(frame_count))
    rem = (value - 5) % 17
    return value if rem == 0 else value + (17 - rem)


def build_dilated_latents(video: torch.Tensor, audio: torch.Tensor, analysis: dict[str, Any]) -> dict[str, Any]:
    """Create a locally time-stretched AV latent and exact inverse token map."""
    if not bool(analysis.get("applied")):
        raise ValueError("Motion Repair dilation requires an applied analysis result.")
    source_t = int(video.shape[2])
    token_spans, source_frames = _token_spans(source_t)
    holds_token = torch.as_tensor(analysis["holds"], dtype=torch.long).cpu()
    risk_token = torch.as_tensor(analysis["risk_mask"], dtype=torch.bool).cpu()
    repair_weight = torch.as_tensor(analysis.get("repair_weight", risk_token.float()), dtype=torch.float32).cpu()
    repair_token = repair_weight > 0
    if (int(holds_token.numel()) != source_t or int(risk_token.numel()) != source_t
            or int(repair_weight.numel()) != source_t):
        raise ValueError("Motion Repair analysis/token geometry mismatch.")

    frame_to_token = _frame_to_token(source_t)
    expanded_frame_map: list[int] = []
    for source_frame in range(source_frames):
        tok = int(frame_to_token[source_frame])
        repeat = max(1, int(holds_token[tok]))
        expanded_frame_map.extend([source_frame] * repeat)

    target_frames = _legal_frame_ceil(len(expanded_frame_map))
    if not expanded_frame_map:
        raise RuntimeError("Motion Repair produced an empty temporal map.")
    if target_frames > len(expanded_frame_map):
        expanded_frame_map.extend([expanded_frame_map[-1]] * (target_frames - len(expanded_frame_map)))

    target_t = int(video_latent_t(target_frames))
    target_spans, _ = _token_spans(target_t)
    source_token_for_target: list[int] = []
    target_repair_weight: list[float] = []
    target_video_tokens: list[torch.Tensor] = []
    for out_token, (a, _b) in enumerate(target_spans):
        source_frame = int(expanded_frame_map[min(a, len(expanded_frame_map) - 1)])
        src_token = int(frame_to_token[source_frame])
        source_token_for_target.append(src_token)
        target_repair_weight.append(float(repair_weight[src_token]))
        target_video_tokens.append(video[:, :, src_token:src_token + 1])
    expanded_video = torch.cat(target_video_tokens, dim=2).contiguous()

    # Locally stretch the audio latent by the same world-frame map.  Audio is
    # frozen during repair and restored exactly afterward; this stream exists
    # only to keep H3's joint AV timing context coherent during the repair pass.
    source_audio_t = int(audio.shape[-1])
    target_audio_t = int(audio_latent_t(target_frames))
    if source_audio_t <= 1:
        expanded_audio = audio[..., :1].expand(*audio.shape[:-1], target_audio_t).clone()
    else:
        audio_indices = []
        for out_tick in range(target_audio_t):
            out_frame = int(round((out_tick / max(1, target_audio_t - 1)) * max(0, target_frames - 1)))
            src_frame = int(expanded_frame_map[min(out_frame, len(expanded_frame_map) - 1)])
            src_tick = int(round((src_frame / max(1, source_frames - 1)) * max(0, source_audio_t - 1)))
            audio_indices.append(min(source_audio_t - 1, max(0, src_tick)))
        idx = torch.tensor(audio_indices, dtype=torch.long, device=audio.device)
        expanded_audio = audio.index_select(-1, idx).contiguous()

    target_repair_tensor = torch.tensor(target_repair_weight, dtype=torch.float32)
    video_mask = target_repair_tensor.to(device=video.device, dtype=torch.float32).view(1, 1, target_t, 1, 1)
    if int(video.shape[0]) > 1:
        video_mask = video_mask.expand(int(video.shape[0]), 1, target_t, 1, 1).clone()
    audio_mask = torch.zeros((audio.shape[0], 1, 1, target_audio_t), dtype=torch.float32, device=audio.device)

    # For each source token, select the center target token that represents it.
    recover_indices: list[int] = []
    for source_token in range(source_t):
        candidates = [q for q, src in enumerate(source_token_for_target) if int(src) == source_token]
        if candidates:
            recover_indices.append(candidates[len(candidates) // 2])
        else:
            recover_indices.append(min(range(target_t), key=lambda q: abs(int(source_token_for_target[q]) - source_token)))

    return {
        "video": expanded_video,
        "audio": expanded_audio,
        "video_mask": video_mask,
        "audio_mask": audio_mask,
        "source_token_for_target": tuple(int(v) for v in source_token_for_target),
        "recover_indices": tuple(int(v) for v in recover_indices),
        "source_risk_mask": risk_token,
        "source_repair_mask": repair_token,
        "source_repair_weight": repair_weight,
        "macrocell_atomic": True,
        "expanded_frame_to_source_frame": tuple(int(v) for v in expanded_frame_map),
        "source_frames": int(source_frames),
        "target_frames": int(target_frames),
        "source_latent_t": int(source_t),
        "target_latent_t": int(target_t),
        "source_audio_t": int(source_audio_t),
        "target_audio_t": int(target_audio_t),
        "dilation_ratio": float(target_frames) / float(max(1, source_frames)),
    }


def recover_repaired_video(original_video: torch.Tensor, repaired_expanded_video: torch.Tensor,
                           dilation: dict[str, Any]) -> torch.Tensor:
    """Recover original H3 T and restore non-repair tokens bit-for-bit."""
    source_t = int(original_video.shape[2])
    indices = list(dilation.get("recover_indices") or ())
    repair = torch.as_tensor(
        dilation.get("source_repair_mask", dilation.get("source_risk_mask")), dtype=torch.bool
    )
    if len(indices) != source_t or int(repair.numel()) != source_t:
        raise ValueError("Motion Repair recovery map does not match source latent time.")
    idx = torch.tensor(indices, dtype=torch.long, device=repaired_expanded_video.device)
    recovered = repaired_expanded_video.index_select(2, idx)
    if tuple(recovered.shape) != tuple(original_video.shape):
        raise RuntimeError(
            f"Motion Repair recovered video geometry mismatch: {tuple(recovered.shape)} vs {tuple(original_video.shape)}"
        )
    keep_original = (~repair).to(device=original_video.device).view(1, 1, source_t, 1, 1)
    if int(original_video.shape[0]) > 1:
        keep_original = keep_original.expand(int(original_video.shape[0]), 1, source_t, 1, 1)
    return torch.where(keep_original, original_video, recovered.to(original_video.device)).to(dtype=original_video.dtype)


def remap_frame_index(source_frame: int, expanded_frame_to_source_frame: tuple[int, ...]) -> int:
    """Map a source pixel-frame anchor onto the first matching expanded frame."""
    if not expanded_frame_to_source_frame:
        return max(0, int(source_frame))
    target = max(0, int(source_frame))
    for out_index, source in enumerate(expanded_frame_to_source_frame):
        if int(source) >= target:
            return int(out_index)
    return int(len(expanded_frame_to_source_frame) - 1)


def choose_repair_sigmas(full_sigmas: torch.Tensor, mode: str) -> tuple[torch.Tensor | None, dict[str, Any]]:
    preset = preset_for(mode)
    if preset is None or not torch.is_tensor(full_sigmas) or int(full_sigmas.numel()) < 2:
        return None, {"steps": 0, "start_index": None}
    steps = int(full_sigmas.numel()) - 1
    desired = max(2, int(math.ceil(steps * float(preset.denoise_fraction))))
    desired = min(steps, int(preset.max_steps), desired)
    if desired <= 0:
        return None, {"steps": 0, "start_index": None}
    start = max(0, steps - desired)
    tail = full_sigmas[start:].detach().clone()
    return tail, {
        "steps": int(tail.numel()) - 1,
        "start_index": int(start),
        "denoise_fraction": float(preset.denoise_fraction),
        "max_steps": int(preset.max_steps),
        "sigma_start": float(tail[0].detach().cpu()),
    }


def plan_repair_windows(analysis: dict[str, Any], *, context_tokens: int = 5,
                        max_windows: int = 3) -> list[tuple[int, int]]:
    """Plan phase-aligned local repair windows as [start_token, end_token).

    Window starts are H3 phase-0 token indices (multiple of five) and ends are
    phase-2 boundaries, so every local T remains 5*k+2 and can be sampled as an
    independent native H3 clip. Nearby risk runs are merged; if more than
    ``max_windows`` remain, the closest gaps are merged first.
    """
    risk = torch.as_tensor(analysis.get("risk_mask"), dtype=torch.bool).cpu()
    t = int(risk.numel())
    if t < 2 or (t - 2) % 5 or not bool(risk.any()):
        return []

    runs: list[tuple[int, int]] = []
    start = None
    for i, active in enumerate(risk.tolist()):
        if active and start is None:
            start = i
        if start is not None and (not active or i == t - 1):
            end = i if not active else i + 1
            runs.append((int(start), int(end)))
            start = None

    ctx = max(0, int(context_tokens))
    windows: list[list[int]] = []
    for a, b in runs:
        desired_start = max(0, a - ctx)
        start_tok = (desired_start // 5) * 5
        desired_end = min(t, b + ctx)
        if desired_end >= t:
            end_tok = t
        else:
            # Smallest global token boundary >= desired_end with phase index 2.
            end_tok = desired_end + ((2 - desired_end) % 5)
            end_tok = min(t, end_tok)
        if end_tok - start_tok < 7 and t >= 7:
            end_tok = min(t, start_tok + 7)
            if end_tok < t and end_tok % 5 != 2:
                end_tok = min(t, end_tok + ((2 - end_tok) % 5))
            if end_tok - start_tok < 7:
                start_tok = max(0, end_tok - 7)
                start_tok = (start_tok // 5) * 5
        if (end_tok - start_tok) >= 2 and (end_tok - start_tok - 2) % 5 == 0:
            windows.append([int(start_tok), int(end_tok)])

    # Merge overlap/touching windows first.
    merged: list[list[int]] = []
    for a, b in sorted(windows):
        if not merged or a > merged[-1][1]:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)

    # Bound the number of sampler calls by merging the closest phase-aligned gaps.
    limit = max(1, int(max_windows))
    while len(merged) > limit:
        gap_index = min(range(len(merged) - 1), key=lambda i: merged[i + 1][0] - merged[i][1])
        merged[gap_index][1] = merged[gap_index + 1][1]
        merged.pop(gap_index + 1)

    return [(int(a), int(b)) for a, b in merged]


def slice_analysis_for_window(analysis: dict[str, Any], start_token: int, end_token: int) -> dict[str, Any]:
    """Project a global oracle result into one phase-aligned local H3 window."""
    start = int(start_token)
    end = int(end_token)
    risk = torch.as_tensor(analysis.get("risk_mask"), dtype=torch.bool).cpu()[start:end].clone()
    repair_weight_all = analysis.get("repair_weight")
    repair_weight = (
        torch.as_tensor(repair_weight_all, dtype=torch.float32).cpu()[start:end].clone()
        if repair_weight_all is not None else risk.float()
    )
    holds = torch.as_tensor(analysis.get("holds"), dtype=torch.long).cpu()[start:end].clone()
    score_all = analysis.get("score")
    score = torch.as_tensor(score_all, dtype=torch.float32).cpu()[start:end].clone() if score_all is not None else torch.ones_like(holds, dtype=torch.float32)
    local_t = int(end - start)
    spans, local_frames = _token_spans(local_t)
    mode = normalize_motion_repair_mode(analysis.get("mode"))
    preset = preset_for(mode)
    if preset is None:
        raise ValueError("Cannot slice disabled Motion Repair analysis.")

    # Re-apply the pixel-time ceiling locally. A global budget can otherwise
    # concentrate too much dilation into one small window after cropping.
    widths = [int(b - a) for a, b in spans]
    threshold = max(float(analysis.get("threshold_z") or preset.threshold_z), 1e-6)
    severity = score / threshold
    raw_limit = max(local_frames, int(math.floor(local_frames * float(preset.max_dilation))))
    max_legal = 5 + 17 * max(0, (raw_limit - 5) // 17)
    max_legal = max(local_frames, max_legal)

    def expanded_frames() -> int:
        return sum(widths[i] * int(holds[i]) for i in range(local_t))

    order = sorted((i for i in range(local_t) if int(holds[i]) > 1), key=lambda i: float(severity[i]))
    while expanded_frames() > max_legal and order:
        changed = False
        for i in order:
            if int(holds[i]) > 1 and expanded_frames() > max_legal:
                holds[i] -= 1
                changed = True
        if not changed:
            break

    applied = bool(risk.any()) and bool((repair_weight > 0).any()) and expanded_frames() > local_frames
    return {
        "enabled": True,
        "mode": mode,
        "applied": applied,
        "reason": "window_repair_requested" if applied else "window_dilation_budget_no_legal_expansion",
        "source_latent_t": local_t,
        "source_frames": local_frames,
        "score": score,
        "risk_mask": risk,
        "repair_weight": repair_weight,
        "repair_mask": (repair_weight > 0),
        "holds": holds,
        "risk_tokens": int(risk.sum().item()),
        "repair_tokens": int((repair_weight > 0).sum().item()),
        "macrocell_atomic": True,
        "risk_spans_frames": _contiguous_spans(risk, spans),
        "peak_score": float(score.max().item()) if score.numel() else 0.0,
        "threshold_z": float(analysis.get("threshold_z") or preset.threshold_z),
        "mean_motion": float(analysis.get("mean_motion") or 0.0),
        "denoise_fraction": float(preset.denoise_fraction),
        "max_steps": int(preset.max_steps),
        "max_dilation": float(preset.max_dilation),
        "global_start_token": start,
        "global_end_token": end,
    }


def slice_audio_for_token_window(audio: torch.Tensor, source_video_t: int,
                                 start_token: int, end_token: int) -> tuple[torch.Tensor, int]:
    """Slice/pad the frozen audio latent for a phase-aligned video-token window."""
    global_spans, global_frames = _token_spans(int(source_video_t))
    start = int(start_token)
    end = int(end_token)
    if start < 0 or end > int(source_video_t) or end <= start:
        raise ValueError(f"Invalid Motion Repair token window {start}:{end} for T={source_video_t}.")
    source_frame_offset = int(global_spans[start][0])
    local_frames = frame_count_from_video_t(end - start)
    target_t = audio_latent_t(local_frames)
    source_audio_t = int(audio.shape[-1])
    if source_audio_t <= 0:
        raise ValueError("Motion Repair source audio latent is empty.")
    start_tick = int(round((source_frame_offset / max(1, global_frames)) * source_audio_t))
    end_tick = min(source_audio_t, start_tick + target_t)
    local = audio[..., start_tick:end_tick]
    if int(local.shape[-1]) < target_t:
        pad = target_t - int(local.shape[-1])
        tail = local[..., -1:] if int(local.shape[-1]) > 0 else audio[..., -1:]
        local = torch.cat([local, tail.expand(*tail.shape[:-1], pad)], dim=-1)
    elif int(local.shape[-1]) > target_t:
        local = local[..., :target_t]
    return local.contiguous(), source_frame_offset
