"""Sol-H3 acceleration helpers adapted to ComfyUI's native MiniMax-H3 runtime.

Current Comfy H3 already owns two of NVIDIA Sol-H3's lossless operator fusions:
QK RMSNorm + partial RoPE is fused through Comfy Kitchen, and SwiGLU is executed
through ``linear_input_act`` (including quantized INT8/W4A8 semantics).  This
module adds the missing trajectory-wide AdaLN precompute/cache while preserving
those native kernels.

The important lifecycle detail is *when* this runs: an APPLY_MODEL wrapper calls
``ensure_sol_h3_adaln_cache`` before MiniMax-H3 creates its per-block prefetch
queue.  Each large ``adaln_proj`` is evaluated for the complete sampler
trajectory while that projection is hot, then replaced by a parameter-free
lookup.  The original projection modules live outside the model tree during the
denoise loop and are restored after sampling / at hard memory isolation.
"""
from __future__ import annotations

import math
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn

_VISUAL_COND_TIMESTEP = 0.999
_AUDIO_COND_TIMESTEP = 1.0
_LOCK = threading.RLock()
_ACTIVE: dict[int, "SolH3AdalnController"] = {}


def _time_shift_sigma(sigma: float, from_shift: float, to_shift: float) -> float:
    sigma = max(float(sigma), 1.0e-6)
    base = sigma / (float(from_shift) + sigma * (1.0 - float(from_shift)))
    return float(to_shift) * base / (1.0 + (float(to_shift) - 1.0) * base)


def _clean_sigmas(values: Iterable[float]) -> tuple[float, ...]:
    out: list[float] = []
    for value in values:
        try:
            sigma = float(value)
        except Exception:
            continue
        if not math.isfinite(sigma) or sigma <= 0.0:
            continue
        if not any(abs(sigma - old) <= 1.0e-8 for old in out):
            out.append(sigma)
    return tuple(out)


def _mask_video_t_values(mask: Any, video: Any, sigma_v: float, t_pin_v: float) -> set[float]:
    if not torch.is_tensor(mask) or not torch.is_tensor(video) or mask.numel() == 0:
        return set()
    try:
        m = mask[0, 0].to(torch.float32)
        latent_t = int(video.shape[-3])
        lat_h = ((int(video.shape[-2]) + 1) // 2) * 2
        lat_w = ((int(video.shape[-1]) + 1) // 2) * 2
        if m.ndim != 3:
            return set()
        m = torch.nn.functional.pad(
            m,
            (0, max(0, lat_w - int(m.shape[-1])), 0, max(0, lat_h - int(m.shape[-2]))),
            mode="replicate",
        )
        m = m[:latent_t, :lat_h, :lat_w]
        m = m.reshape(latent_t, lat_h // 2, 2, lat_w // 2, 2).amax(dim=(2, 4)).reshape(-1)
        if bool((m >= 1.0 - 1.0e-3).all()):
            return set()
        rows_t = (1.0 - m * float(sigma_v)).clamp(max=float(t_pin_v))
        return {float(v) for v in rows_t.unique().detach().cpu().tolist()}
    except Exception:
        return set()


def _mask_audio_t_values(mask: Any, sigma_a: float, t_pin_a: float) -> set[float]:
    if not torch.is_tensor(mask) or mask.numel() == 0:
        return set()
    try:
        m = mask[0, 0].to(torch.float32).reshape(-1)
        if bool((m >= 1.0 - 1.0e-3).all()):
            return set()
        rows_t = (1.0 - m * float(sigma_a)).clamp(max=float(t_pin_a))
        return {float(v) for v in rows_t.unique().detach().cpu().tolist()}
    except Exception:
        return set()


def _step_t_values(
    sigmas: Iterable[float],
    *,
    sigma_shift_video: float,
    sigma_shift_audio: float,
    visual_cond_noise_aug: float,
    audio_cond_noise_aug: float,
    layout: Any = None,
    video: Any = None,
    denoise_mask: Any = None,
    audio_denoise_mask: Any = None,
) -> tuple[tuple[float, ...], ...]:
    """Reproduce Comfy H3's sorted ``unique_t`` set for each model forward."""
    kinds: set[str] = {"text", "video", "audio"}
    try:
        for _a, _b, kind in layout.segments:
            kinds.add(str(kind))
    except Exception:
        pass
    out: list[tuple[float, ...]] = []
    for sigma_v in _clean_sigmas(sigmas):
        sigma_a = _time_shift_sigma(sigma_v, sigma_shift_video, sigma_shift_audio)
        t_v = 1.0 - sigma_v
        t_a = 1.0 - sigma_a
        vis_aug = float(visual_cond_noise_aug)
        aud_aug = float(audio_cond_noise_aug)
        seg_t = {
            "text": t_v,
            "video": t_v,
            "audio": t_a,
            "cond": max(t_v, vis_aug),
            "ref_img": max(t_v, vis_aug),
            "cond_audio": max(t_a, aud_aug),
            "ref_audio": max(t_a, aud_aug),
        }
        values = {float(t_v), float(t_a)}
        values.update(float(seg_t[k]) for k in kinds if k in seg_t)
        values.update(
            _mask_video_t_values(
                denoise_mask, video, sigma_v, max(t_v, _VISUAL_COND_TIMESTEP)
            )
        )
        values.update(
            _mask_audio_t_values(
                audio_denoise_mask, sigma_a, max(t_a, _AUDIO_COND_TIMESTEP)
            )
        )
        signature = tuple(sorted(values))
        if not out or signature != out[-1]:
            out.append(signature)
    return tuple(out)


@dataclass(frozen=True)
class SolH3AdalnStats:
    installed: bool
    reason: str
    timestep_rows: int = 0
    schedule_points: int = 0
    blocks: int = 0
    cached_mb: float = 0.0
    original_parameter_mb: float = 0.0


class _AdalnLookup(nn.Module):
    """Parameter-free proxy; originals/tables remain outside the model tree."""

    def __init__(self, controller: "SolH3AdalnController", slot: tuple[str, int]):
        super().__init__()
        object.__setattr__(self, "_controller_ref", weakref.ref(controller))
        object.__setattr__(self, "_slot", slot)
        original = controller.original_for(slot)
        self.expand = int(getattr(original, "expand", 0) or 0)
        self.modalities = int(getattr(original, "modalities", 0) or 0)
        self.hidden = int(getattr(original, "hidden", 0) or 0)
        self.apply_silu = bool(getattr(original, "apply_silu", True))

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        controller = object.__getattribute__(self, "_controller_ref")()
        if controller is None:
            raise RuntimeError("Sol-H3 AdaLN controller was released while a lookup proxy was active")
        return controller.lookup(object.__getattribute__(self, "_slot"), t_emb)


class SolH3AdalnController:
    """Exact per-schedule AdaLN cache for one loaded Comfy H3 transformer."""

    def __init__(self, diffusion_model: Any):
        self.diffusion_model = diffusion_model
        self._originals: dict[tuple[str, int], nn.Module] = {}
        self._tables: dict[tuple[str, int], tuple[tuple[torch.Tensor, ...], ...]] = {}
        self._step_t_emb: tuple[torch.Tensor, ...] = ()
        self._last_mapping_key: tuple[int, tuple[int, ...], torch.dtype, torch.device] | None = None
        self._last_step_index: int | None = None
        self.installed = False
        self.restore_reason: str | None = None
        self.lookup_fallbacks = 0

    def original_for(self, slot: tuple[str, int]) -> nn.Module:
        return self._originals[slot]

    @staticmethod
    def _parameter_bytes(module: nn.Module) -> int:
        try:
            return sum(int(p.numel()) * int(p.element_size()) for p in module.parameters(recurse=True))
        except Exception:
            return 0

    @staticmethod
    def _values_bytes(values: tuple[torch.Tensor, ...]) -> int:
        return sum(int(x.numel()) * int(x.element_size()) for x in values)

    def _match_step(self, t_emb: torch.Tensor) -> int | None:
        key = (int(t_emb.data_ptr()), tuple(int(v) for v in t_emb.shape), t_emb.dtype, t_emb.device)
        if self._last_mapping_key == key:
            return self._last_step_index
        tolerance = 5.0e-4 if t_emb.dtype in (torch.float16, torch.bfloat16) else 2.0e-5
        best_index = None
        best_error = float("inf")
        for index, reference in enumerate(self._step_t_emb):
            if tuple(reference.shape) != tuple(t_emb.shape):
                continue
            rhs = reference
            if rhs.device != t_emb.device or rhs.dtype != t_emb.dtype:
                rhs = rhs.to(device=t_emb.device, dtype=t_emb.dtype)
            error = float((t_emb.float() - rhs.float()).abs().max().detach().cpu().item())
            if error < best_error:
                best_error = error
                best_index = index
        if best_index is None or best_error > tolerance:
            best_index = None
        self._last_mapping_key = key
        self._last_step_index = best_index
        return best_index

    def lookup(self, slot: tuple[str, int], t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        schedules = self._tables.get(slot)
        original = self._originals.get(slot)
        if schedules is None or original is None:
            raise RuntimeError(f"invalid Sol-H3 AdaLN cache slot {slot!r}")
        step_index = self._match_step(t_emb)
        if step_index is None or step_index >= len(schedules):
            # Unknown fractional-mask or alternate refine timetable: exact stock
            # fallback, never nearest-neighbor approximation.
            self.lookup_fallbacks += 1
            return original(t_emb)
        return schedules[step_index]

    @torch.inference_mode()
    def install(
        self,
        *,
        sample_sigmas: Iterable[float],
        context_dtype: torch.dtype,
        device: torch.device,
        visual_cond_noise_aug: float = _VISUAL_COND_TIMESTEP,
        audio_cond_noise_aug: float = _AUDIO_COND_TIMESTEP,
        layout: Any = None,
        video: Any = None,
        denoise_mask: Any = None,
        audio_denoise_mask: Any = None,
        max_timestep_rows: int = 128,
        max_cache_mb: int = 1536,
    ) -> SolH3AdalnStats:
        model = self.diffusion_model
        if self.installed:
            rows = sum(int(x.shape[0]) for x in self._step_t_emb)
            return SolH3AdalnStats(True, "already installed", rows, len(self._step_t_emb), len(getattr(model, "blocks", ())))
        if bool(getattr(model, "use_adaln_curves", False)):
            return SolH3AdalnStats(False, "Comfy H3 AdaLN curve checkpoint already uses compact curve-form projections")
        blocks = list(getattr(model, "blocks", ()) or ())
        final_layer = getattr(model, "final_layer", None)
        time_embedder = getattr(model, "time_embedder", None)
        if not blocks or final_layer is None or time_embedder is None:
            return SolH3AdalnStats(False, "loaded diffusion model does not expose stock MiniMax-H3 AdaLN structure")

        signatures = _step_t_values(
            sample_sigmas,
            sigma_shift_video=float(getattr(model, "sigma_shift_video", 12.0)),
            sigma_shift_audio=float(getattr(model, "sigma_shift_audio", 3.0)),
            visual_cond_noise_aug=float(visual_cond_noise_aug),
            audio_cond_noise_aug=float(audio_cond_noise_aug),
            layout=layout,
            video=video,
            denoise_mask=denoise_mask,
            audio_denoise_mask=audio_denoise_mask,
        )
        total_rows = sum(len(values) for values in signatures)
        if not signatures:
            return SolH3AdalnStats(False, "sampler sigma trajectory is empty")
        if total_rows > int(max_timestep_rows):
            return SolH3AdalnStats(False, f"trajectory has {total_rows} total timestep rows > cache limit {max_timestep_rows}")

        step_t_emb = tuple(
            time_embedder(torch.tensor(values, dtype=torch.float32, device=device)).to(context_dtype)
            for values in signatures
        )
        originals: dict[tuple[str, int], nn.Module] = {}
        for index, block in enumerate(blocks):
            projection = getattr(block, "adaln_proj", None)
            if not isinstance(projection, nn.Module):
                return SolH3AdalnStats(False, f"block {index} has no compatible adaln_proj")
            originals[("block", index)] = projection
        final_projection = getattr(final_layer, "adaln_proj", None)
        if not isinstance(final_projection, nn.Module):
            return SolH3AdalnStats(False, "final layer has no compatible adaln_proj")
        originals[("final", 0)] = final_projection

        original_bytes = sum(self._parameter_bytes(module) for module in originals.values())
        tables: dict[tuple[str, int], tuple[tuple[torch.Tensor, ...], ...]] = {}
        cached_bytes = sum(int(x.numel()) * int(x.element_size()) for x in step_t_emb)
        try:
            # Deliberately one GEMM per *actual step signature*, matching the
            # runtime M dimension.  This follows NVIDIA's bit-stability design:
            # batch-all-steps GEMMs can select a different GEMM implementation.
            for slot, projection in originals.items():
                schedule_values: list[tuple[torch.Tensor, ...]] = []
                for t_emb in step_t_emb:
                    values = tuple(part.detach() for part in projection(t_emb))
                    cached_bytes += self._values_bytes(values)
                    if cached_bytes > int(max_cache_mb) * 1024 * 1024:
                        raise RuntimeError(
                            f"AdaLN cache would exceed {max_cache_mb} MB ({cached_bytes / 1024**2:.1f} MB)"
                        )
                    schedule_values.append(values)
                tables[slot] = tuple(schedule_values)
        except Exception as exc:
            tables.clear()
            return SolH3AdalnStats(False, f"precompute failed: {type(exc).__name__}: {exc}")

        self._originals = originals
        self._tables = tables
        self._step_t_emb = tuple(x.detach() for x in step_t_emb)
        for index, block in enumerate(blocks):
            block.adaln_proj = _AdalnLookup(self, ("block", index))
        final_layer.adaln_proj = _AdalnLookup(self, ("final", 0))
        self.installed = True
        return SolH3AdalnStats(
            True,
            "trajectory-wide exact AdaLN cache installed",
            timestep_rows=total_rows,
            schedule_points=len(signatures),
            blocks=len(blocks),
            cached_mb=cached_bytes / 1024**2,
            original_parameter_mb=original_bytes / 1024**2,
        )

    def restore(self, reason: str = "cleanup") -> None:
        if not self.installed:
            return
        model = self.diffusion_model
        blocks = list(getattr(model, "blocks", ()) or ())
        for index, block in enumerate(blocks):
            original = self._originals.get(("block", index))
            if original is not None:
                block.adaln_proj = original
        final_layer = getattr(model, "final_layer", None)
        original_final = self._originals.get(("final", 0))
        if final_layer is not None and original_final is not None:
            final_layer.adaln_proj = original_final
        self.restore_reason = str(reason)
        self.installed = False
        self._tables.clear()
        self._step_t_emb = ()
        self._last_mapping_key = None
        self._last_step_index = None
        self._originals.clear()


def _diffusion_from_state(state: dict[str, Any]) -> Any | None:
    patcher = state.get("residency_model_patcher")
    if patcher is None:
        return None
    try:
        return patcher.get_model_object("diffusion_model")
    except Exception:
        pass
    try:
        return patcher.model.diffusion_model
    except Exception:
        return None


def ensure_sol_h3_adaln_cache(
    state: dict[str, Any],
    *,
    context: Any,
    payload: dict[str, Any] | None = None,
    video: Any = None,
    denoise_mask: Any = None,
    audio_denoise_mask: Any = None,
) -> SolH3AdalnStats:
    """Install once for a ``sol_h3`` LongMedia run; safe to call every forward."""
    if not isinstance(state, dict) or not bool(state.get("sol_h3_profile", False)):
        return SolH3AdalnStats(False, "Sol-H3 profile is inactive")
    if bool(state.get("fasth3_vsa_active", False)):
        return SolH3AdalnStats(False, "FastH3 checkpoint already owns its exact AdaLN lookup")
    sigmas = state.get("sol_h3_sample_sigmas") or ()
    if not sigmas:
        return SolH3AdalnStats(False, "sampler sigma trajectory was not published")
    diffusion = _diffusion_from_state(state)
    if diffusion is None:
        return SolH3AdalnStats(False, "active MiniMax-H3 diffusion model could not be resolved")

    # 0.6.22 consumer-GPU policy. NVIDIA's resident AdaLN precompute is a win
    # when the transformer can stay resident, but on a 16-GB DynamicVRAM run it
    # front-loads every large projection before block 0 and also leaves a GPU
    # cache competing with the multi-gigabyte H3 attention workspace. The user
    # trace showed exactly that failure mode (223 s first step followed by an
    # allocation OOM). Keep the exact stock AdaLN path on <=18.5-GiB dynamic
    # cards; FastH3 has its own trained lookup and is already excluded above.
    patcher = state.get("residency_model_patcher")
    try:
        dynamic = bool(patcher is not None and patcher.is_dynamic())
    except Exception:
        dynamic = False
    runtime_device = state.get("sol_h3_runtime_device")
    try:
        probe_device = runtime_device if isinstance(runtime_device, torch.device) else None
        if probe_device is None and torch.is_tensor(context):
            probe_device = context.device
        total_bytes = (
            int(torch.cuda.get_device_properties(probe_device).total_memory)
            if probe_device is not None and probe_device.type == "cuda" and torch.cuda.is_available()
            else 0
        )
    except Exception:
        total_bytes = 0
    if dynamic and total_bytes and total_bytes <= int(18.5 * 1024**3):
        state["sol_h3_low_vram_adaln_bypass"] = True
        state["sol_h3_device_total_mb"] = round(total_bytes / 1024**2, 1)
        return SolH3AdalnStats(
            False,
            "<=18.5 GiB DynamicVRAM: trajectory AdaLN precompute bypassed to preserve attention workspace and startup latency",
        )

    key = id(state)
    with _LOCK:
        controller = _ACTIVE.get(key)
        if controller is not None and controller.installed:
            rows = sum(int(x.shape[0]) for x in controller._step_t_emb)
            return SolH3AdalnStats(True, "already installed", rows, len(controller._step_t_emb), len(getattr(diffusion, "blocks", ())))
        if controller is None:
            controller = SolH3AdalnController(diffusion)
            _ACTIVE[key] = controller
    try:
        if torch.is_tensor(context):
            context_dtype = context.dtype
            device = context.device
        else:
            parameter = next(diffusion.parameters())
            context_dtype = parameter.dtype if parameter.dtype.is_floating_point else torch.bfloat16
            device = parameter.device
        runtime_device = state.get("sol_h3_runtime_device")
        if isinstance(runtime_device, torch.device):
            device = runtime_device
        payload = payload if isinstance(payload, dict) else {}
        stats = controller.install(
            sample_sigmas=sigmas,
            context_dtype=context_dtype,
            device=device,
            visual_cond_noise_aug=float(payload.get("visual_cond_noise_aug", _VISUAL_COND_TIMESTEP)),
            audio_cond_noise_aug=float(payload.get("audio_cond_noise_aug", _AUDIO_COND_TIMESTEP)),
            layout=payload.get("layout"),
            video=video,
            denoise_mask=denoise_mask,
            audio_denoise_mask=audio_denoise_mask,
        )
        if not stats.installed:
            with _LOCK:
                _ACTIVE.pop(key, None)
        return stats
    except Exception as exc:
        with _LOCK:
            _ACTIVE.pop(key, None)
        return SolH3AdalnStats(False, f"install exception: {type(exc).__name__}: {exc}")


def restore_sol_h3_adaln_for_state(state: dict[str, Any] | None, reason: str = "cleanup") -> bool:
    if not isinstance(state, dict):
        return False
    key = id(state)
    with _LOCK:
        controller = _ACTIVE.pop(key, None)
    if controller is None:
        return False
    controller.restore(reason)
    return True


def restore_all_sol_h3_adaln(reason: str = "memory isolation") -> int:
    with _LOCK:
        controllers = list(_ACTIVE.values())
        _ACTIVE.clear()
    restored = 0
    for controller in controllers:
        try:
            controller.restore(reason)
            restored += 1
        except Exception:
            pass
    return restored


def active_sol_h3_adaln_controllers() -> int:
    with _LOCK:
        return sum(1 for controller in _ACTIVE.values() if controller.installed)
