"""ComfyUI nodes for direct MiniMax H3 audio/video latent control."""

from __future__ import annotations
import copy
import builtins
import gc
import json
import math
import os
import re
import sys
import time
import uuid as _uuid_mod
import importlib.metadata as _importlib_metadata


def _pkg_version_tuple(name: str):
    try:
        raw = _importlib_metadata.version(name)
    except Exception:
        return None, None
    nums = []
    for part in raw.split('.'):
        digits = ''.join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        nums.append(int(digits))
    while len(nums) < 3:
        nums.append(0)
    return raw, tuple(nums[:3])

import torch
import torchaudio
import comfy.nested_tensor
import comfy.model_management

try:
    import comfy.model_prefetch as _comfy_model_prefetch
except Exception:
    _comfy_model_prefetch = None

try:
    import comfy_aimdo as _comfy_aimdo
except Exception:
    _comfy_aimdo = None

try:
    from .latent_ops import (
        AUDIO_LATENT_FPS, FPS, _fit_stream, _validate_audio, _validate_video,
        align_frame_count, apply_video_inpaint_mask, audio_latent_t,
        describe_av, frame_count_from_video_t, inject_leading_video_frame,
        merge_av_latents, pack_av_latents, prepare_continuation, replace_stream,
        set_stream_denoise, split_av_latent, stitch_continuation,
        unpack_av_samples, video_latent_t,
    )
except ImportError:
    from latent_ops import (
        AUDIO_LATENT_FPS, FPS, _fit_stream, _validate_audio, _validate_video,
        align_frame_count, apply_video_inpaint_mask, audio_latent_t,
        describe_av, frame_count_from_video_t, inject_leading_video_frame,
        merge_av_latents, pack_av_latents, prepare_continuation, replace_stream,
        set_stream_denoise, split_av_latent, stitch_continuation,
        unpack_av_samples, video_latent_t,
    )

from dataclasses import replace as _dc_replace

try:
    from .media_plan import (
        build_media_plan, LongMediaPlan, slice_video_segment,
        slice_audio_segment, collect_numbered_inputs,
    )
except ImportError:
    from media_plan import (
        build_media_plan, LongMediaPlan, slice_video_segment,
        slice_audio_segment, collect_numbered_inputs,
    )

try:
    from .temporal_positioning import (
        TEMPORAL_OFFSET_OPTION, temporal_offset_for_frame, h3_temporal_offset_wrapper,
    )
except ImportError:
    from temporal_positioning import (
        TEMPORAL_OFFSET_OPTION, temporal_offset_for_frame, h3_temporal_offset_wrapper,
    )

try:
    from .continuity_policy import (
        build_segment_prompt as _policy_build_segment_prompt,
        normalize_hybrid_picture_tags,
    )
except ImportError:
    from continuity_policy import (
        build_segment_prompt as _policy_build_segment_prompt,
        normalize_hybrid_picture_tags,
    )

try:
    from .refine_policy import split_refine_sigmas
except ImportError:
    from refine_policy import split_refine_sigmas

try:
    from comfy.patcher_extension import WrappersMP
except Exception:  # pragma: no cover - only available inside ComfyUI
    WrappersMP = None

CATEGORY = 'MiniMax H3/LongMedia'
CATEGORY_STREAMS = f'{CATEGORY}/Streams'
CATEGORY_CONTINUATION = f'{CATEGORY}/Continuation'
CATEGORY_LONGMEDIA = f'{CATEGORY}/LongMedia'
CATEGORY_UTIL = f'{CATEGORY}/Utility'
MAX_RESOLUTION = 16384
CANVAS_MULTIPLE = 32
NestedTensor = comfy.nested_tensor.NestedTensor

NativeReferenceToVideo = None


# Release console guard. True = quiet release console; False = full LongMedia diagnostics.
# The Long Media Setup node owns this switch so the choice is serialized in the workflow.
_LONGMEDIA_RELEASE_GUARD = True
_LONGMEDIA_VERBOSE = False
_LONGMEDIA_ALWAYS_CONSOLE = (
    "oom", "error", "failed", "failure", "fallback", "exception",
    "cleanup failed", "emergency trim", "storage guard] trim", "late guard] late guard trim",
)

def _set_longmedia_release_guard(enabled):
    global _LONGMEDIA_RELEASE_GUARD, _LONGMEDIA_VERBOSE
    _LONGMEDIA_RELEASE_GUARD = bool(enabled)
    _LONGMEDIA_VERBOSE = not _LONGMEDIA_RELEASE_GUARD
    # Always print the mode transition itself so users can verify the workflow state.
    builtins.print(
        f"[MiniMaxH3 LongMedia][0.3.87 RELEASE GUARD] "
        f"{'ON (release console)' if _LONGMEDIA_RELEASE_GUARD else 'OFF (full diagnostics)'}"
    )


def _lm_print(*args, **kwargs):
    if _LONGMEDIA_VERBOSE:
        return builtins.print(*args, **kwargs)
    text = " ".join(str(a) for a in args).lower()
    if any(marker in text for marker in _LONGMEDIA_ALWAYS_CONSOLE):
        return builtins.print(*args, **kwargs)
    return None



_RAM_FILECACHE_PREWARM_SEEN = set()

def _iter_quant_tensor_leaves(value, _seen=None):
    """Yield real tensor leaves from normal or Comfy QuantizedTensor values."""
    if _seen is None:
        _seen = set()
    oid = id(value)
    if oid in _seen:
        return
    _seen.add(oid)
    if torch.is_tensor(value):
        # QuantizedTensor is a Tensor subclass; expose its backing tensors too.
        try:
            names, _ctx = value.__tensor_flatten__()
        except Exception:
            names = None
        if names:
            for name in names:
                try:
                    child = getattr(value, name)
                except Exception:
                    continue
                yield from _iter_quant_tensor_leaves(child, _seen)
        else:
            yield value
        return
    if isinstance(value, dict):
        for child in value.values():
            yield from _iter_quant_tensor_leaves(child, _seen)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _iter_quant_tensor_leaves(child, _seen)


def _collect_h3_mmap_payloads(model_patcher):
    """Return unique AIMDO mmap payload memoryviews backing this model."""
    model = getattr(model_patcher, 'model', None)
    if model is None:
        return []
    try:
        sd = model.state_dict()
    except Exception:
        return []
    payloads = []
    seen_maps = set()
    for value in sd.values():
        for tensor in _iter_quant_tensor_leaves(value):
            try:
                storage = tensor.untyped_storage()
                refs = getattr(storage, '_comfy_tensor_mmap_refs', None)
            except Exception:
                refs = None
            if not refs or len(refs) < 2:
                continue
            mmap_obj, payload = refs[0], refs[1]
            key = id(mmap_obj)
            if key in seen_maps:
                continue
            seen_maps.add(key)
            try:
                nbytes = int(payload.nbytes)
            except Exception:
                try:
                    nbytes = len(payload)
                except Exception:
                    nbytes = 0
            if nbytes > 0:
                payloads.append((mmap_obj, payload, nbytes))
    return payloads


def _prewarm_h3_file_cache(model_patcher, model_size_bytes=0, min_ram_headroom_gb=10.0):
    """Warm AIMDO-backed checkpoint pages into the OS file cache without tensor copies.

    This intentionally changes neither model tensors nor VRAM residency.  On Windows
    it asks the memory manager to prefetch mapped checkpoint pages, then lightly
    touches one byte per 64 KiB region so the request is observable before sampling.
    The warm budget is capped by currently available RAM minus a hard headroom.
    """
    result = {
        'status': 'skipped', 'payloads': 0, 'payload_bytes': 0,
        'budget_bytes': 0, 'touched_bytes': 0, 'seconds': 0.0,
        'reason': None,
    }
    try:
        import psutil
        vm = psutil.virtual_memory()
        available = int(vm.available)
    except Exception as exc:
        result['reason'] = f'RAM query failed: {type(exc).__name__}'
        return result

    hard_headroom = int(float(min_ram_headroom_gb) * (1024 ** 3))
    budget = max(0, available - hard_headroom)
    if budget < 2 * (1024 ** 3):
        result['reason'] = f'available RAM headroom too small ({available/(1024**3):.1f}GB available)'
        return result

    payloads = _collect_h3_mmap_payloads(model_patcher)
    result['payloads'] = len(payloads)
    total_payload = sum(x[2] for x in payloads)
    result['payload_bytes'] = int(total_payload)
    if not payloads:
        result['reason'] = 'no AIMDO mmap payloads found on model tensors'
        return result

    # Never consume the user's entire free RAM.  The cap is deliberately smaller
    # than the mapped model on a 64 GB machine so Windows retains normal working
    # set/pagefile headroom while still caching a large fraction of a 32 GB H3.
    budget = min(budget, total_payload)
    if model_size_bytes:
        budget = min(budget, int(model_size_bytes))
    result['budget_bytes'] = int(budget)

    signature = tuple(sorted((id(mm), nbytes) for mm, _mv, nbytes in payloads))
    if signature in _RAM_FILECACHE_PREWARM_SEEN:
        result['status'] = 'already_warm'
        result['reason'] = 'same AIMDO mappings already prewarmed in this process'
        return result

    t0 = time.perf_counter()
    remaining = int(budget)
    touched = 0
    checksum = 0
    try:
        import numpy as np
        kernel32 = None
        process_handle = None
        range_type = None
        if os.name == 'nt':
            try:
                import ctypes
                from ctypes import wintypes
                class _WIN32_MEMORY_RANGE_ENTRY(ctypes.Structure):
                    _fields_ = [('VirtualAddress', ctypes.c_void_p), ('NumberOfBytes', ctypes.c_size_t)]
                kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
                prefetch = kernel32.PrefetchVirtualMemory
                prefetch.argtypes = [wintypes.HANDLE, ctypes.c_ulong, ctypes.POINTER(_WIN32_MEMORY_RANGE_ENTRY), ctypes.c_ulong]
                prefetch.restype = wintypes.BOOL
                process_handle = kernel32.GetCurrentProcess()
                range_type = _WIN32_MEMORY_RANGE_ENTRY
            except Exception:
                kernel32 = process_handle = range_type = None

        for _mmap_obj, payload, nbytes in payloads:
            if remaining <= 0:
                break
            take = min(int(nbytes), remaining)
            if take <= 0:
                continue
            arr = np.frombuffer(payload, dtype=np.uint8, count=take)
            if kernel32 is not None and process_handle is not None and range_type is not None:
                try:
                    entry = range_type(ctypes.c_void_p(int(arr.ctypes.data)), ctypes.c_size_t(int(take)))
                    kernel32.PrefetchVirtualMemory(process_handle, 1, ctypes.byref(entry), 0)
                except Exception:
                    pass
            # One touch per Windows allocation-granularity-sized region.  AIMDO's
            # mmap remains the owner; this creates no duplicate tensor allocation.
            stride = 64 * 1024
            for start in range(0, take, 512 * 1024 * 1024):
                end = min(take, start + 512 * 1024 * 1024)
                probe = arr[start:end:stride]
                if probe.size:
                    checksum ^= int(probe.sum(dtype=np.uint64)) & 0xFFFFFFFF
            touched += take
            remaining -= take
        _RAM_FILECACHE_PREWARM_SEEN.add(signature)
        result['status'] = 'warmed'
        result['touched_bytes'] = int(touched)
        result['checksum'] = int(checksum)
    except Exception as exc:
        result['status'] = 'failed'
        result['reason'] = f'{type(exc).__name__}: {exc}'
    result['seconds'] = float(time.perf_counter() - t0)
    return result

def _resolve_native_reference_to_video():
    """Lazy-load MiniMaxH3ReferenceToVideo from stock nodes."""
    try:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        return MiniMaxH3ReferenceToVideo
    except ImportError:
        return None


def _is_valid_frame_count(count: int) -> bool:
    return count >= 5 and (count - 5) % 17 == 0


def _fit_frames(frames: torch.Tensor, target_count: int, mode: str) -> torch.Tensor:
    count = frames.shape[0]
    if count == 0:
        raise ValueError('Video input contains no frames.')
    if mode == 'strict':
        if count != target_count:
            raise ValueError(
                f'Strict frame fit requires {target_count} frames, got {count}.'
            )
        return frames
    elif mode == 'crop_or_pad_last':
        if count >= target_count:
            return frames[:target_count]
        else:
            pad = frames[-1:].expand(target_count - count, *frames.shape[1:])
            return torch.cat((frames, pad), dim=0)
    elif mode == 'loop':
        indices = torch.arange(target_count, device=frames.device) % count
        return frames[indices]
    else:
        raise ValueError(f'Unknown frame fit mode: {mode}')


def _resize_frames(frames: torch.Tensor, width: int, height: int, resize_mode: str) -> torch.Tensor:
    frames = frames[..., :3]
    if frames.shape[2] == width and frames.shape[1] == height:
        return frames
    elif resize_mode == 'none':
        raise ValueError(
            f'Input is {frames.shape[2]}x{frames.shape[1]}, '
            f'target is {width}x{height}. Choose stretch or center_crop.'
        )
    else:
        crop = 'disabled' if resize_mode == 'stretch' else 'center'
        import comfy.utils
        samples = frames.movedim(-1, 1)
        samples = comfy.utils.common_upscale(samples, width, height, 'lanczos', crop)
        return samples.movedim(1, -1)


# H3 spatial patchification consumes VAE latents in 2x2 spatial patches.
# With a 16x VAE spatial reduction, pixel geometry therefore needs to stay on
# a 32px grid. More importantly, native Ref2VA `match` grows reference-token
# area with the target canvas; empirically that changes H3 behaviour sharply
# between the long-tested ~0.6 MP regime and ~1 MP. Keep generation resolution
# independent from reference-conditioning resolution: target may be any H3-safe
# 32px canvas, while still-image refs are capped to a stable token budget.
_H3_SAFE_REF_PIXELS = int(round(0.60 * 1024 * 1024))

def _h3_safe_dim(value: int, multiple: int = CANVAS_MULTIPLE) -> int:
    value = max(int(multiple), int(value))
    return max(int(multiple), int(round(value / float(multiple))) * int(multiple))

def _h3_safe_target_canvas(width: int, height: int):
    requested_w, requested_h = int(width), int(height)
    safe_w = _h3_safe_dim(requested_w)
    safe_h = _h3_safe_dim(requested_h)
    return safe_w, safe_h, {
        'requested_width': requested_w, 'requested_height': requested_h,
        'safe_width': safe_w, 'safe_height': safe_h,
        'changed': bool((safe_w, safe_h) != (requested_w, requested_h)),
        'pixel_multiple': int(CANVAS_MULTIPLE),
    }

def _h3_safe_reference_image(image: torch.Tensor, max_pixels: int = _H3_SAFE_REF_PIXELS):
    if image is None:
        return None, None
    h, w = int(image.shape[1]), int(image.shape[2])
    area = max(1, w * h)
    scale = min(1.0, math.sqrt(float(max_pixels) / float(area)))
    # Floor after scaling so rounding can never push the reference back over
    # the conditioning budget. A 32px grid guarantees even H/16 and W/16.
    tw = max(CANVAS_MULTIPLE, int(math.floor((w * scale) / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)
    th = max(CANVAS_MULTIPLE, int(math.floor((h * scale) / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)
    if tw == w and th == h:
        resized = image[..., :3]
    else:
        resized = _resize_frames(image, tw, th, 'stretch')
    return resized, {
        'source_width': w, 'source_height': h,
        'safe_width': tw, 'safe_height': th,
        'source_pixels': area, 'safe_pixels': tw * th,
        'pixel_budget': int(max_pixels),
        'latent_width': tw // 16, 'latent_height': th // 16,
        'patch_width': (tw // 16) // 2, 'patch_height': (th // 16) // 2,
    }

def _h3_safe_reference_images(images, max_pixels: int = _H3_SAFE_REF_PIXELS):
    prepared, reports = [], []
    for image in images or []:
        if image is None:
            continue
        safe, report = _h3_safe_reference_image(image, max_pixels=max_pixels)
        prepared.append(safe)
        reports.append(report)
    return prepared, reports


def _fit_waveform(waveform: torch.Tensor, samples: int, mode: str) -> torch.Tensor:
    current = waveform.shape[-1]
    if current == 0:
        raise ValueError('Audio input contains no samples.')
    if mode == 'strict':
        return waveform
    elif mode == 'crop_or_pad_silence':
        if current >= samples:
            return waveform[..., :samples]
        else:
            return torch.nn.functional.pad(waveform, (0, samples - current))
    elif mode == 'loop':
        repeats = math.ceil(samples / current)
        return (waveform.repeat(1, 1, repeats))[..., :samples]
    else:
        raise ValueError(f'Unknown audio fit mode: {mode}')


def _target_video_geometry(target_av):
    """Extract (video_tensor, frame_count, width, height) from a target AV latent."""
    video, _ = unpack_av_samples(target_av)
    frames = frame_count_from_video_t(video.shape[2])
    width = video.shape[4] * 16
    height = video.shape[3] * 16
    return video, frames, width, height


def _free_cuda_memory():
    """Force a Python GC pass and let the CUDA allocator release cached blocks.

    This doesn't unload anything by itself; it just gives back memory that
    Python/PyTorch would otherwise keep cached for reuse, which matters right
    after a large tensor (e.g. a stitched multi-pass accumulator) is offloaded
    or freed, so the freed VRAM is actually available to the next pass.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _release_model_memory_for_decode():
    """Create a hard memory boundary before final VAE decode.

    Sampling can leave the diffusion model and several transient CUDA pools
    resident.  A long video VAE decode should not compete with those weights.
    This helper is deliberately best-effort/fail-open: decode still proceeds if
    a particular ComfyUI build does not expose one of the cleanup helpers.
    """
    before = _cuda_memory_snapshot()
    errors = []
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as exc:
        errors.append(f'sync_before: {type(exc).__name__}: {exc}')
    try:
        comfy.model_management.unload_all_models()
    except Exception as exc:
        errors.append(f'unload_all_models: {type(exc).__name__}: {exc}')
    try:
        gc.collect()
        try:
            comfy.model_management.soft_empty_cache(force=True)
        except TypeError:
            comfy.model_management.soft_empty_cache()
    except Exception as exc:
        errors.append(f'soft_empty_cache: {type(exc).__name__}: {exc}')
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception as exc:
        errors.append(f'sync_after: {type(exc).__name__}: {exc}')
    after = _cuda_memory_snapshot()
    if before is not None or after is not None:
        bfree = _mb(before['driver_free']) if before else None
        afree = _mb(after['driver_free']) if after else None
        bres = _mb(before['reserved']) if before else None
        ares = _mb(after['reserved']) if after else None
        _lm_print(
            '[MiniMaxH3 LongMedia][V316 DECODE MEMORY BARRIER] '
            f'driver_free_mb {bfree}->{afree}, reserved_mb {bres}->{ares}'
            + (f', cleanup_errors={errors}' if errors else ''),
            flush=True,
        )
    return {'before': before, 'after': after, 'errors': errors}


def _decode_video_vae_safe(video_vae, video, enable_tiling, tile_size, temporal_size):
    """Decode H3 video latent without triggering ComfyUI's full-decode OOM path.

    When tiling is enabled, call the public VAE.decode_tiled API directly using
    the same unit conversion as ComfyUI's VAEDecodeTiled node.  This avoids the
    regular VAE.decode() attempt that may exhaust VRAM before fallback can run.
    """
    if not bool(enable_tiling):
        _lm_print('[MiniMaxH3 LongMedia][V316 DECODE] regular decode requested', flush=True)
        return video_vae.decode(video), {
            'strategy': 'regular',
            'tile_size_px': None,
            'temporal_size_frames': None,
        }

    tile_size_px = max(64, int(tile_size))
    spatial_overlap_px = min(64, max(16, tile_size_px // 4))
    if tile_size_px < spatial_overlap_px * 4:
        spatial_overlap_px = max(1, tile_size_px // 4)

    temporal_frames = max(8, int(temporal_size))
    temporal_overlap_frames = max(4, min(8, temporal_frames // 4))
    if temporal_frames < temporal_overlap_frames * 2:
        temporal_overlap_frames = max(1, temporal_frames // 2)

    compression = int(video_vae.spacial_compression_decode())
    tile_x = max(1, tile_size_px // compression)
    tile_y = max(1, tile_size_px // compression)
    overlap = max(0, spatial_overlap_px // compression)

    temporal_compression = video_vae.temporal_compression_decode()
    if temporal_compression is not None:
        temporal_compression = int(temporal_compression)
        tile_t = max(2, temporal_frames // temporal_compression)
        overlap_t = max(1, min(tile_t // 2, temporal_overlap_frames // temporal_compression))
    else:
        tile_t = None
        overlap_t = None

    _lm_print(
        '[MiniMaxH3 LongMedia][V316 DECODE] direct tiled decode: '
        f'latent_shape={tuple(video.shape)}, tile_px={tile_size_px}, overlap_px={spatial_overlap_px}, '
        f'tile_latent=({tile_x},{tile_y}), temporal_frames={temporal_frames}, '
        f'tile_t={tile_t}, overlap_t={overlap_t}',
        flush=True,
    )
    images = video_vae.decode_tiled(
        video,
        tile_x=tile_x,
        tile_y=tile_y,
        overlap=overlap,
        tile_t=tile_t,
        overlap_t=overlap_t,
    )
    return images, {
        'strategy': 'direct_tiled',
        'tile_size_px': tile_size_px,
        'spatial_overlap_px': spatial_overlap_px,
        'tile_x_latent': tile_x,
        'tile_y_latent': tile_y,
        'temporal_size_frames': temporal_frames,
        'temporal_overlap_frames': temporal_overlap_frames,
        'tile_t_latent': tile_t,
        'overlap_t_latent': overlap_t,
    }


def _cuda_memory_snapshot(device=None):
    """Best-effort CUDA allocator/driver snapshot; never raises after async OOM."""
    if not torch.cuda.is_available():
        return None
    try:
        if device is None:
            device = torch.cuda.current_device()
        free_driver, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        cached = max(0, reserved - allocated)
        return {
            'device': int(device) if isinstance(device, int) else device,
            'driver_free': int(free_driver),
            'total': int(total),
            'allocated': int(allocated),
            'reserved': int(reserved),
            'cached': int(cached),
        }
    except Exception:
        return None



def _mb(value):
    return round(float(value) / (1024.0 ** 2), 1)


def _soft_empty_cuda_cache():
    """Release unused CUDA allocator blocks without unloading active models."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        comfy.model_management.soft_empty_cache(force=True)
    except TypeError:
        # Compatibility with older ComfyUI builds whose helper had no force kwarg.
        comfy.model_management.soft_empty_cache()


def _aimdo_setup_boundary_reset(label: str):
    """Synchronize and clear transient AIMDO/Comfy loader state before Setup TE work.

    Mirrors the cleanup ComfyUI itself performs at execution boundaries on current
    DynamicVRAM builds: synchronize pending CUDA work, cleanup prefetch queues,
    reset temporary cast buffers, and reset VBAR watermark limits. This is
    scheduling/lifecycle only; it does not change model weights or math.
    """
    events = []
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            events.append('cuda_sync_pre')
        except Exception as exc:
            events.append(f'cuda_sync_pre:{type(exc).__name__}')

    if _comfy_model_prefetch is not None:
        fn = getattr(_comfy_model_prefetch, 'cleanup_prefetch_queues', None)
        if callable(fn):
            try:
                fn()
                events.append('prefetch_cleanup')
            except Exception as exc:
                events.append(f'prefetch_cleanup:{type(exc).__name__}')

    fn = getattr(comfy.model_management, 'reset_cast_buffers', None)
    if callable(fn):
        try:
            fn()
            events.append('cast_buffers_reset')
        except Exception as exc:
            events.append(f'cast_buffers_reset:{type(exc).__name__}')

    if _comfy_aimdo is not None:
        mv = getattr(_comfy_aimdo, 'model_vbar', None)
        fn = getattr(mv, 'vbars_reset_watermark_limits', None) if mv is not None else None
        if callable(fn):
            try:
                fn()
                events.append('vbar_watermarks_reset')
            except Exception as exc:
                events.append(f'vbar_watermarks_reset:{type(exc).__name__}')

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            events.append('cuda_sync_post')
        except Exception as exc:
            events.append(f'cuda_sync_post:{type(exc).__name__}')

    _lm_print(
        '[MiniMaxH3 LongMedia][0.3.82 AIMDO SETUP BOUNDARY] '
        f'{label}: ' + ','.join(events),
        flush=True,
    )
    return events


def _is_aimdo_transient_fault(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        'fault failed' in text
        or 'device not ready' in text
        or 'vram allocation failed (non oom)' in text
        or 'cumemsetaccess' in text
    )


def _setup_clip_encode_retry(fn, *, label: str):
    """Run one TE encode with one lifecycle-only retry for transient AIMDO VBAR races."""
    try:
        return fn()
    except RuntimeError as exc:
        if not _is_aimdo_transient_fault(exc):
            raise
        _lm_print(
            '[MiniMaxH3 LongMedia][0.3.82 AIMDO TE RETRY] '
            f'{label}: transient AIMDO fault ({type(exc).__name__}: {exc}); resetting loader state and retrying once',
            flush=True,
        )
        try:
            comfy.model_management.unload_all_models()
        except Exception:
            pass
        _aimdo_setup_boundary_reset(label + ':retry')
        try:
            _soft_empty_cuda_cache()
        except Exception:
            pass
        return fn()


def _setup_memory_isolation(label, unload_models=True):
    """Create a clean VRAM boundary between Setup's heavy model stages.

    LongMediaSetup may run after a previous H3 execution, so diffusion weights
    can still occupy most of VRAM when Qwen/CLIP starts encoding.  Explicitly
    unload registered models and flush only dead allocator cache before/after
    conditioning stages.  Returned counters are JSON-safe and used in the
    Setup report for diagnostics.
    """
    before = _cuda_memory_snapshot()
    unload_error = None
    # v0.3.82: native AIMDO uses asynchronous VBAR/prefetch state that can
    # outlive the previous heavy stage. Clear it at Setup boundaries before
    # asking Qwen/CLIP to fault a different model into the same GPU.
    _aimdo_setup_boundary_reset(str(label) + ':pre')
    if unload_models:
        try:
            comfy.model_management.unload_all_models()
        except Exception as exc:
            unload_error = f'{type(exc).__name__}: {exc}'
    _aimdo_setup_boundary_reset(str(label) + ':post_unload')
    try:
        _soft_empty_cuda_cache()
    except Exception as exc:
        if unload_error is None:
            unload_error = f'cache: {type(exc).__name__}: {exc}'
    after = _cuda_memory_snapshot()

    def compact(snap):
        if snap is None:
            return None
        return {
            'allocated_mb': _mb(snap['allocated']),
            'reserved_mb': _mb(snap['reserved']),
            'cached_mb': _mb(snap['cached']),
            'driver_free_mb': _mb(snap['driver_free']),
        }

    b = compact(before)
    a = compact(after)
    if a is not None:
        before_alloc = b['allocated_mb'] if b else 0.0
        before_free = b['driver_free_mb'] if b else 0.0
        _lm_print(
            '[MiniMaxH3 LongMedia] Setup memory isolation: '
            f'{label}, allocated {before_alloc:.1f} -> {a["allocated_mb"]:.1f} MB, '
            f'driver free {before_free:.1f} -> {a["driver_free_mb"]:.1f} MB',
            flush=True,
        )
    if unload_error:
        _lm_print(
            '[MiniMaxH3 LongMedia] Setup memory isolation warning: '
            f'{label}: {unload_error}',
            flush=True,
        )
    return {
        'stage': str(label),
        'before': b,
        'after': a,
        'unload_models': bool(unload_models),
        'warning': unload_error,
    }


def _memory_profile_output_path(kind: str) -> str:
    """Return a Comfy temp path for a CUDA allocator snapshot."""
    try:
        import folder_paths
        base = folder_paths.get_temp_directory()
    except Exception:
        base = os.path.abspath(os.path.join(os.getcwd(), 'temp'))
    directory = os.path.join(base, 'minimax_h3_latentlab_memory')
    os.makedirs(directory, exist_ok=True)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    token = _uuid_mod.uuid4().hex[:8]
    return os.path.join(directory, f'h3_{kind}_{stamp}_{token}.pickle')


def _cuda_allocator_backend():
    """Best-effort name of the active PyTorch CUDA allocator backend."""
    if not torch.cuda.is_available():
        return None
    getter = getattr(getattr(torch.cuda, 'memory', None), 'get_allocator_backend', None)
    if getter is None:
        return None
    try:
        return str(getter())
    except Exception:
        return None


def _start_cuda_memory_history(max_entries=20000):
    """Enable PyTorch CUDA allocator history with compatibility fallbacks.

    cudaMallocAsync currently cannot record allocator history. Detect it up front
    so diagnostic builds do not emit a RuntimeError on every run.
    """
    if not torch.cuda.is_available():
        return False, None
    backend = _cuda_allocator_backend()
    if backend and 'cudamallocasync' in backend.lower():
        return False, f'allocator backend {backend} does not support record_memory_history'
    recorder = getattr(getattr(torch.cuda, 'memory', None), '_record_memory_history', None)
    if recorder is None:
        return False, 'torch.cuda.memory._record_memory_history unavailable'
    try:
        recorder(enabled='all', context='all', stacks='python', max_entries=int(max_entries), clear_history=True)
        return True, None
    except TypeError:
        try:
            recorder(max_entries=int(max_entries))
            return True, None
        except Exception as exc:
            return False, f'{type(exc).__name__}: {exc}'
    except Exception as exc:
        return False, f'{type(exc).__name__}: {exc}'


def _stop_cuda_memory_history():
    recorder = getattr(getattr(torch.cuda, 'memory', None), '_record_memory_history', None)
    if recorder is None:
        return
    try:
        recorder(enabled=None)
    except TypeError:
        try:
            recorder(None)
        except Exception:
            pass
    except Exception:
        pass


def _dump_cuda_memory_snapshot(kind: str):
    """Dump a PyTorch allocator snapshot and return (path, error)."""
    dumper = getattr(getattr(torch.cuda, 'memory', None), '_dump_snapshot', None)
    if dumper is None:
        return None, 'torch.cuda.memory._dump_snapshot unavailable'
    path = _memory_profile_output_path(kind)
    try:
        dumper(path)
        return path, None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'



class _H3QueryChunkAttentionOverride:
    """Low-VRAM full-attention override that chunks only the query sequence.

    K/V remain complete for every chunk, so this is still global/full attention;
    only the query rows are evaluated in smaller batches to cap temporary
    attention workspace. H3 currently calls optimized_attention with
    skip_reshape=True and mask=None, which is the path optimized here.
    """

    def __init__(self, chunk_tokens=8192, state=None):
        self.chunk_tokens = max(256, int(chunk_tokens))
        self.state = state if state is not None else {}

    def __call__(self, func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        state = self.state
        # Keep the override deliberately narrow. Anything that is not H3's
        # unmasked pre-shaped attention path is delegated untouched.
        if (
            mask is not None
            or not skip_reshape
            or getattr(q, 'ndim', 0) != 4
            or int(q.shape[-2]) <= self.chunk_tokens
        ):
            return func(
                q, k, v, heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )

        seq = int(q.shape[-2])
        chunks = (seq + self.chunk_tokens - 1) // self.chunk_tokens
        state['calls'] = int(state.get('calls', 0)) + 1
        state['chunked_calls'] = int(state.get('chunked_calls', 0)) + 1
        state['max_sequence_tokens'] = max(int(state.get('max_sequence_tokens', 0)), seq)
        state['max_chunks_per_call'] = max(int(state.get('max_chunks_per_call', 0)), chunks)

        if not state.get('announced'):
            _lm_print(
                '[MiniMaxH3 LongMedia] Low-VRAM attention enabled: '
                f'query sequence {seq} tokens -> {chunks} chunks of <= {self.chunk_tokens}; K/V stay full',
                flush=True,
            )
            state['announced'] = True

        outputs = []
        for start in range(0, seq, self.chunk_tokens):
            end = min(seq, start + self.chunk_tokens)
            q_chunk = q[..., start:end, :]
            out = func(
                q_chunk, k, v, heads,
                mask=None,
                attn_precision=attn_precision,
                skip_reshape=True,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )
            outputs.append(out)

        # Core Comfy attention functions return [B, Nq, H*D] by default;
        # skip_output_reshape=True keeps [B, H, Nq, D].
        cat_dim = 2 if skip_output_reshape else 1
        return torch.cat(outputs, dim=cat_dim)


class MiniMaxH3LatentLabAttentionChunking:
    """Internal GUIDER wrapper enabling query-chunked full attention."""

    DESCRIPTION = 'Internal H3 low-VRAM query-chunk attention wrapper.'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'guider': ('GUIDER',),
                'memory_mode': (['normal', 'low_vram', 'ultra_low_vram'], {'default': 'normal'}),
                'requested_memory_mode': (['auto', 'normal', 'low_vram', 'ultra_low_vram'], {'default': 'normal'}),
                'chunk_tokens': ('INT', {'default': 8192, 'min': 256, 'max': 65536, 'step': 256}),
            }
        }

    RETURN_TYPES = ('GUIDER', 'H3_ATTENTION_CHUNK_STATE')
    RETURN_NAMES = ('guider', 'attention_chunk_state')
    FUNCTION = 'wrap'
    CATEGORY = CATEGORY_LONGMEDIA

    def wrap(self, guider, chunk_tokens=8192):
        wrapped = copy.copy(guider)
        try:
            wrapped.model_options = copy.deepcopy(getattr(guider, 'model_options', {}) or {})
        except Exception:
            wrapped.model_options = dict(getattr(guider, 'model_options', {}) or {})

        transformer_options = wrapped.model_options.setdefault('transformer_options', {})
        state = {
            'chunk_tokens': int(chunk_tokens),
            'calls': 0,
            'chunked_calls': 0,
            'max_sequence_tokens': 0,
            'max_chunks_per_call': 0,
            'announced': False,
            'step_boundary_forward_count': 0,
            'step_boundary_transitions': [],
            'step_boundary_forward_times': [],
        }
        existing = transformer_options.get('optimized_attention_override')
        if existing is not None:
            # Do not silently trample a user's/custom node's existing attention
            # override. The diagnostic state makes this visible in the report/log.
            state['enabled'] = False
            state['reason'] = 'existing optimized_attention_override present'
            _lm_print(
                '[MiniMaxH3 LongMedia] Low-VRAM attention NOT installed: '
                'another optimized_attention_override is already present.',
                flush=True,
            )
        else:
            transformer_options['optimized_attention_override'] = _H3QueryChunkAttentionOverride(
                chunk_tokens=int(chunk_tokens), state=state
            )
            state['enabled'] = True
            state['reason'] = 'query chunking active'
        return (wrapped, state)


class _H3BlockMemoryTracePatch:
    """Deep first-block tracer for the first H3 DiT forward only."""

    def __init__(self, index, state):
        self.index = int(index)
        self.state = state

    @staticmethod
    def _extract_block(original_block):
        # Stock H3 creates block_wrap as a closure over the actual DiTBlock.
        closure = getattr(original_block, '__closure__', None) or ()
        for cell in closure:
            try:
                obj = cell.cell_contents
            except Exception:
                continue
            if all(hasattr(obj, name) for name in ('adaln_proj', 'norm1', 'attn', 'norm2', 'mlp')):
                return obj
        return None

    @staticmethod
    def _mod_scale_shift(h, shift, scale, segments):
        for a, b, row in segments:
            h[a:b].mul_(1.0 + scale[row].to(h.dtype)).add_(shift[row].to(h.dtype))
        return h

    @staticmethod
    def _mod_gate(x, gate, other, segments):
        for a, b, row in segments:
            x[a:b].addcmul_(other[a:b], gate[row].to(x.dtype))
        return x

    def _inter_block_pressure_guard(self):
        """TEST: effective-headroom inter-block guard with hysteresis.

        Uses driver-free VRAM + allocator-reclaimable cache instead of driver
        free alone. This avoids destroying useful cache when DynamicVRAM/AIMDO
        already has several GB of reclaimable memory available.
        """
        state = self.state

        # Guarantee AUTO calibration after the first completed block.
        if self.index == 0 and not state.get('auto_vram_controller_done'):
            try:
                token_count = int(state.get('current_token_count', 0) or state.get('last_token_count', 0) or 0)
                self._auto_vram_controller_after_probe(token_count)
            except Exception as exc:
                state['auto_vram_controller_done'] = True
                state['auto_vram_controller_mode'] = 'SAFE'
                _lm_print(
                    f"[MiniMaxH3 LongMedia][AUTO VRAM] calibration failed; SAFE baseline retained: {exc!r}",
                    flush=True,
                )

        if not torch.cuda.is_available():
            return

        guard_mb = float(int(state.get('inter_block_vram_guard_mb', 0) or 0))
        emergency_mb = float(int(state.get('inter_block_guard_emergency_mb', 0) or 0))
        cooldown_blocks = int(state.get('inter_block_guard_cooldown_blocks', 0) or 0)
        emergency_cooldown_blocks = int(state.get('inter_block_guard_emergency_cooldown_blocks', 0) or 0)

        if guard_mb <= 0 and emergency_mb <= 0:
            return

        snap = _cuda_memory_snapshot()
        if not snap:
            return

        mb = 1024.0 ** 2
        free_mb = float(snap['driver_free']) / mb
        cached_mb = float(snap['cached']) / mb
        effective_mb = free_mb + cached_mb

        normal_hyst = float(state.get('inter_block_guard_hysteresis_mb', 1024.0) or 1024.0)
        emergency_hyst = float(state.get('inter_block_emergency_hysteresis_mb', 512.0) or 512.0)

        # Existing cooldown bookkeeping, but now only decremented when evaluated.
        cd = int(state.get('inter_block_guard_cooldown', 0) or 0)
        ecd = int(state.get('inter_block_guard_emergency_cooldown', 0) or 0)

        # Emergency should reflect *effective* pressure, not merely low driver free.
        emergency_trigger = emergency_mb > 0 and effective_mb < emergency_mb
        normal_trigger = guard_mb > 0 and effective_mb < guard_mb

        if not emergency_trigger and not normal_trigger:
            if cd > 0:
                state['inter_block_guard_cooldown'] = cd - 1
            if ecd > 0:
                state['inter_block_guard_emergency_cooldown'] = ecd - 1
            skips = int(state.get('inter_block_effective_skip_count', 0) or 0) + 1
            state['inter_block_effective_skip_count'] = skips
            if skips == 1 or skips % 25 == 0:
                _lm_print(
                    '[MiniMaxH3 LongMedia][VRAM GUARD] skip: '
                    f'block {self.index}, free={free_mb:.0f} + cached={cached_mb:.0f} '
                    f'= effective={effective_mb:.0f} MB, guard={guard_mb:.0f}, emergency={emergency_mb:.0f}',
                    flush=True,
                )
            return

        # Hysteresis band prevents trim/reload ping-pong close to threshold.
        if normal_trigger and effective_mb >= max(0.0, guard_mb - normal_hyst):
            state['inter_block_hysteresis_skip_count'] = int(
                state.get('inter_block_hysteresis_skip_count', 0) or 0
            ) + 1
            if cd > 0:
                state['inter_block_guard_cooldown'] = cd - 1
            return

        if emergency_trigger and effective_mb >= max(0.0, emergency_mb - emergency_hyst):
            state['inter_block_emergency_hyst_skip_count'] = int(
                state.get('inter_block_emergency_hyst_skip_count', 0) or 0
            ) + 1
            if ecd > 0:
                state['inter_block_guard_emergency_cooldown'] = ecd - 1
            return

        # Cooldowns suppress repeated trims unless pressure is materially worse.
        if emergency_trigger:
            hard_emergency = effective_mb < max(0.0, emergency_mb - 1024.0)
            if ecd > 0 and not hard_emergency:
                state['inter_block_emergency_cooldown_skip_count'] = int(
                    state.get('inter_block_emergency_cooldown_skip_count', 0) or 0
                ) + 1
                state['inter_block_guard_emergency_cooldown'] = ecd - 1
                return
        elif normal_trigger:
            hard_normal = effective_mb < max(0.0, guard_mb - 1536.0)
            if cd > 0 and not hard_normal:
                state['inter_block_cooldown_skip_count'] = int(
                    state.get('inter_block_cooldown_skip_count', 0) or 0
                ) + 1
                state['inter_block_guard_cooldown'] = cd - 1
                return

        # If cache is tiny, cleanup is unlikely to help; preserve cache and let
        # Sol adaptive retry / CUDA OOM handling be the final safety net.
        min_reclaim_mb = float(state.get('inter_block_min_reclaim_mb', 256.0) or 256.0)
        if cached_mb < min_reclaim_mb:
            state['inter_block_low_cache_skip_count'] = int(
                state.get('inter_block_low_cache_skip_count', 0) or 0
            ) + 1
            return

        try:
            gc.collect()
            comfy.model_management.soft_empty_cache()
        except Exception as exc:
            _lm_print(
                f"[MiniMaxH3 LongMedia][VRAM GUARD] cleanup failed at block {self.index}: {exc!r}",
                flush=True,
            )
            return

        after = _cuda_memory_snapshot()
        if emergency_trigger:
            state['inter_block_guard_emergency_cooldown'] = emergency_cooldown_blocks
            state['inter_block_emergency_trim_count'] = int(
                state.get('inter_block_emergency_trim_count', 0) or 0
            ) + 1
            label = 'EMERGENCY TRIM'
        else:
            state['inter_block_guard_cooldown'] = cooldown_blocks
            state['inter_block_normal_trim_count'] = int(
                state.get('inter_block_normal_trim_count', 0) or 0
            ) + 1
            label = 'NORMAL TRIM'

        if after:
            free_after = float(after['driver_free']) / mb
            cached_after = float(after['cached']) / mb
            _lm_print(
                f'[MiniMaxH3 LongMedia][VRAM GUARD] {label}: block {self.index}, '
                f'effective={effective_mb:.0f} MB, free {free_mb:.0f}->{free_after:.0f} MB, '
                f'cached {cached_mb:.0f}->{cached_after:.0f} MB',
                flush=True,
            )

    def _measure(self, name, fn, state, device):
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        before = _cuda_memory_snapshot()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        out = fn()
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        after = _cuda_memory_snapshot()
        peak_alloc = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        entry = {
            'stage': name,
            'allocated_before_mb': _mb(before['allocated']),
            'allocated_after_mb': _mb(after['allocated']),
            'reserved_before_mb': _mb(before['reserved']),
            'reserved_after_mb': _mb(after['reserved']),
            'driver_free_after_mb': _mb(after['driver_free']),
            'peak_allocated_mb': _mb(peak_alloc),
            'peak_reserved_mb': _mb(peak_reserved),
            'elapsed_ms': round(elapsed_ms, 1),
        }
        state['stages'].append(entry)
        state['highest_block_peak_allocated_mb'] = max(
            float(state.get('highest_block_peak_allocated_mb') or 0.0), float(entry['peak_allocated_mb']))
        state['highest_block_peak_reserved_mb'] = max(
            float(state.get('highest_block_peak_reserved_mb') or 0.0), float(entry['peak_reserved_mb']))
        if float(entry['peak_allocated_mb']) >= float(state.get('worst_stage_peak_allocated_mb') or -1.0):
            state['worst_stage'] = name
            state['worst_stage_peak_allocated_mb'] = float(entry['peak_allocated_mb'])
        _lm_print(
            '[MiniMaxH3 LongMedia] H3 block0 stage: '
            f"{name}, alloc {entry['allocated_before_mb']:.1f} -> {entry['allocated_after_mb']:.1f} MB, "
            f"peak {entry['peak_allocated_mb']:.1f} MB, reserved peak {entry['peak_reserved_mb']:.1f} MB, "
            f"free {entry['driver_free_after_mb']:.1f} MB, {entry['elapsed_ms']:.1f} ms",
            flush=True,
        )
        return out

    def __call__(self, args, extra_options):
        original_block = extra_options['original_block']
        state = self.state

        if self.index != 0 or state.get('first_forward_complete'):
            return original_block(args)
        if state.get('forward_count', 0) > 0:
            state['first_forward_complete'] = True
            return original_block(args)

        state['forward_count'] = 1
        state['first_forward_started'] = True
        state['first_forward_started_at'] = time.time()
        _lm_print('[MiniMaxH3 LongMedia] H3 block0 deep memory trace: first forward started', flush=True)

        if not torch.cuda.is_available():
            return original_block(args)

        block = self._extract_block(original_block)
        if block is None:
            state['fallback_reason'] = 'could not extract DiTBlock from original_block closure'
            _lm_print('[MiniMaxH3 LongMedia] H3 block0 deep trace fallback: DiTBlock closure not found', flush=True)
            return original_block(args)

        device = torch.cuda.current_device()
        x = args['img']
        t_emb = args['t_emb']
        mod_segments = args['mod_segments']
        rope_freqs = args['rope_freqs']
        transformer_options = args['transformer_options']

        try:
            vals = self._measure('adaln_proj', lambda: block.adaln_proj(t_emb), state, device)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = vals

            h = self._measure(
                'norm1_mod',
                lambda: self._mod_scale_shift(block.norm1(x), shift_msa, scale_msa, mod_segments),
                state, device,
            )
            attn_out = self._measure(
                'attention_full',
                lambda: block.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options),
                state, device,
            )
            x = self._measure(
                'attention_gate_residual',
                lambda: self._mod_gate(x, gate_msa, attn_out, mod_segments),
                state, device,
            )
            del attn_out

            h = self._measure(
                'norm2_mod',
                lambda: self._mod_scale_shift(block.norm2(x), shift_mlp, scale_mlp, mod_segments),
                state, device,
            )
            mlp_out = self._measure('mlp_full', lambda: block.mlp(h), state, device)
            x = self._measure(
                'mlp_gate_residual',
                lambda: self._mod_gate(x, gate_mlp, mlp_out, mod_segments),
                state, device,
            )
            del mlp_out, h

            # Preserve patches_replace contract.
            state['blocks'].append({
                'block': 0,
                'peak_allocated_mb': state.get('highest_block_peak_allocated_mb', 0.0),
                'peak_reserved_mb': state.get('highest_block_peak_reserved_mb', 0.0),
                'deep_trace': True,
            })
            _lm_print(
                '[MiniMaxH3 LongMedia] H3 block0 deep trace summary: '
                f"worst stage {state.get('worst_stage')}, "
                f"peak allocated {state.get('highest_block_peak_allocated_mb', 0.0):.1f} MB",
                flush=True,
            )
            return {'img': x}
        except Exception as exc:
            message = str(exc).lower()
            is_oom = isinstance(exc, getattr(torch, 'OutOfMemoryError', RuntimeError)) or 'out of memory' in message
            if is_oom:
                state['oom'] = True
                state['oom_block'] = 0
                state['oom_stage'] = state.get('stages', [])[-1]['stage'] if state.get('stages') else 'unknown'
                state['oom_message'] = str(exc)[:2000]
                _lm_print(
                    f"[MiniMaxH3 LongMedia] H3 BLOCK0 CUDA OOM near stage {state.get('oom_stage')}: {state['oom_message']}",
                    flush=True,
                )
            raise


class MiniMaxH3LatentLabBlockMemoryTracer:
    """Internal GUIDER wrapper that instruments H3 DiT blocks via patches_replace."""

    DESCRIPTION = 'Internal first-forward H3 transformer block VRAM tracer.'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'guider': ('GUIDER',),
                'max_blocks': ('INT', {'default': 128, 'min': 1, 'max': 256, 'step': 1}),
            }
        }

    RETURN_TYPES = ('GUIDER', 'H3_BLOCK_MEMORY_TRACE_STATE')
    RETURN_NAMES = ('guider', 'block_trace_state')
    FUNCTION = 'wrap'
    CATEGORY = CATEGORY_LONGMEDIA

    def wrap(self, guider, max_blocks=128):
        traced = copy.copy(guider)
        try:
            traced.model_options = copy.deepcopy(getattr(guider, 'model_options', {}) or {})
        except Exception:
            traced.model_options = dict(getattr(guider, 'model_options', {}) or {})

        transformer_options = traced.model_options.setdefault('transformer_options', {})
        patches_replace = transformer_options.setdefault('patches_replace', {})
        dit = patches_replace.setdefault('dit', {})
        state = {
            'allocator_backend': _cuda_allocator_backend(),
            'max_blocks': int(max_blocks),
            'forward_count': 0,
            'first_forward_started': False,
            'first_forward_complete': False,
            'first_forward_started_at': None,
            'pre_block0': None,
            'blocks': [],
            'stages': [],
            'worst_stage': None,
            'worst_stage_peak_allocated_mb': 0.0,
            'fallback_reason': None,
            'skipped_existing_patch_indices': [],
            'highest_block_peak_allocated_mb': 0.0,
            'highest_block_peak_reserved_mb': 0.0,
            'worst_block': None,
            'worst_block_peak_allocated_mb': 0.0,
            'oom': False,
            'oom_block': None,
            'oom_message': None,
            'oom_stats': None,
        }
        for i in range(int(max_blocks)):
            key = ('double_block', i)
            if key in dit:
                state['skipped_existing_patch_indices'].append(i)
                continue
            dit[key] = _H3BlockMemoryTracePatch(i, state)
        return (traced, state)


def _h3_sol_span_wrapper(executor, *args, **kwargs):
    """Publish H3's packed video span without altering APPLY_MODEL call semantics.

    ComfyUI APPLY_MODEL wrappers receive the full BaseModel.apply_model argument
    list: (x, t, c_concat, c_crossattn, control, transformer_options, **kwargs).
    MiniMax-specific payload data lives in kwargs.  Keep the positional layout
    untouched so this wrapper composes with other APPLY_MODEL wrappers.
    """
    call_args = list(args)
    options = call_args[5] if len(call_args) > 5 and isinstance(call_args[5], dict) else kwargs.get('transformer_options')
    options = options or {}
    payload = kwargs.get('minimax_payload')
    layout = payload.get('layout') if isinstance(payload, dict) else None
    if layout is not None and hasattr(layout, 'segments'):
        try:
            span = next(((int(a), int(b)) for a, b, kind in layout.segments if kind == 'video'), None)
        except Exception:
            span = None
        if span is not None:
            options = dict(options)
            options['latentlab_sol_h3_video_span'] = span
            if len(call_args) > 5:
                call_args[5] = options
            else:
                kwargs['transformer_options'] = options
    return executor(*call_args, **kwargs)






def _h3_segment_layout_guard_wrapper(executor, *args, **kwargs):
    """Repair MiniMax H3 text-tag length drift and validate PackedLayout cheaply.

    Some ComfyUI/H3 revisions carry presentation modality tags separately from
    the encoded context. LongMedia pre-encodes a different prompt per segment;
    after hybrid/reference metadata is reattached, a stale tag tensor can be a
    few rows shorter/longer than ``context.shape[1]``. Stock H3 then indexes the
    list up to the text segment length and raises an opaque ``IndexError``.

    The safe semantic for missing tail rows is the ordinary text modality (tag
    1); surplus rows are unreachable and are truncated. The payload is copied
    before modification so cached/shared segment conditioning remains immutable.
    """
    call_args = list(args)

    # DIFFUSION_MODEL currently receives (..., context, transformer_options,
    # minimax_payload=...), but avoid depending on a single ComfyUI revision.
    context = None
    if len(call_args) >= 3 and hasattr(call_args[2], 'shape'):
        context = call_args[2]
    if context is None:
        context = kwargs.get('context')

    text_len = None
    try:
        if context is not None and len(context.shape) >= 2:
            text_len = int(context.shape[1])
    except Exception:
        text_len = None

    payload = kwargs.get('minimax_payload')
    payload_location = ('kw', 'minimax_payload') if isinstance(payload, dict) else None
    if not isinstance(payload, dict):
        for idx, value in enumerate(call_args):
            if isinstance(value, dict) and any(
                key in value for key in ('text_token_tags', 'keyframes', 'refs', 'layout')
            ):
                payload = value
                payload_location = ('arg', idx)
                break

    if isinstance(payload, dict) and text_len is not None and text_len >= 0:
        tags = payload.get('text_token_tags')
        if tags is not None:
            try:
                flat = tags.reshape(-1) if hasattr(tags, 'reshape') else tags
                tag_len = int(flat.shape[0]) if hasattr(flat, 'shape') else len(flat)
                if tag_len != text_len:
                    new_payload = dict(payload)
                    if hasattr(flat, 'new_full'):
                        if tag_len < text_len:
                            pad = flat.new_full((text_len - tag_len,), 1)
                            import torch
                            fixed = torch.cat((flat, pad), dim=0)
                        else:
                            fixed = flat[:text_len].clone()
                    else:
                        fixed = list(flat[:text_len])
                        if len(fixed) < text_len:
                            fixed.extend([1] * (text_len - len(fixed)))
                    new_payload['text_token_tags'] = fixed
                    new_payload['longmedia_text_tag_alignment'] = {
                        'from': int(tag_len), 'to': int(text_len),
                    }
                    payload = new_payload
                    if payload_location and payload_location[0] == 'arg':
                        call_args[int(payload_location[1])] = new_payload
                    else:
                        kwargs['minimax_payload'] = new_payload
                    _lm_print(
                        '[MiniMaxH3 LongMedia][V315 SEGMENT TAG GUARD] '
                        f'text_token_tags {tag_len}->{text_len}; '
                        + ('padded missing tail as text(tag=1)' if tag_len < text_len else 'truncated unreachable tail'),
                        flush=True,
                    )
            except Exception as exc:
                _lm_print(
                    '[MiniMaxH3 LongMedia][V315 SEGMENT TAG GUARD] WARNING: '
                    f'could not inspect text_token_tags: {type(exc).__name__}: {exc}',
                    flush=True,
                )

        # Fail early with useful dimensions for a genuinely corrupt layout.
        layout = payload.get('layout')
        if layout is not None and hasattr(layout, 'segments'):
            try:
                segments = list(layout.segments)
                expected = 0
                for seg_idx, (a, b, kind) in enumerate(segments):
                    a, b = int(a), int(b)
                    if a != expected or b < a:
                        raise RuntimeError(
                            f'non-contiguous PackedLayout at segment {seg_idx} '
                            f'({kind}: {a}:{b}, expected start {expected})'
                        )
                    expected = b
                seq_len = int(getattr(layout, 'seq_len', expected))
                pos_len = int(layout.position_ids.shape[0]) if hasattr(layout, 'position_ids') else seq_len
                if expected != seq_len or pos_len != seq_len:
                    raise RuntimeError(
                        'PackedLayout size mismatch: '
                        f'segments_end={expected}, seq_len={seq_len}, position_ids={pos_len}, '
                        f'text_len={text_len}'
                    )
                if segments and str(segments[0][2]) == 'text':
                    packed_text_len = int(segments[0][1]) - int(segments[0][0])
                    if packed_text_len != text_len:
                        raise RuntimeError(
                            'PackedLayout/context text mismatch: '
                            f'layout_text={packed_text_len}, context_text={text_len}, '
                            f'tags={getattr(payload.get("text_token_tags"), "shape", None)}'
                        )
            except RuntimeError:
                raise
            except Exception as exc:
                _lm_print(
                    '[MiniMaxH3 LongMedia][V315 LAYOUT GUARD] WARNING: '
                    f'layout inspection skipped: {type(exc).__name__}: {exc}',
                    flush=True,
                )

    return executor(*call_args, **kwargs)



def _h3_runtime_prefetch_wrapper(executor, *args, _bound_residency_state=None, _bound_residency_patcher=None, **kwargs):
    """Disable Comfy dynamic-VBAR prefetch at the DIFFUSION_MODEL boundary.

    Comfy BaseModel._apply_model() overwrites prefetch_dynamic_vbars from
    current_patcher.is_dynamic() immediately before invoking diffusion_model.
    Therefore sampler-side transformer_options alone are insufficient. This
    wrapper runs after that overwrite and is the authoritative out-of-core gate.

    Do not assume a fixed positional index for transformer_options. WrapperExecutor
    call shapes can differ between ComfyUI revisions. Find the actual options dict
    by our LongMedia runtime marker / known transformer-option keys.
    """
    call_args = list(args)

    candidates = []
    for idx, value in enumerate(call_args):
        if isinstance(value, dict):
            candidates.append(('arg', idx, value))

    for key, value in kwargs.items():
        if isinstance(value, dict):
            candidates.append(('kw', key, value))

    transformer_options = None
    location = None

    # Strong marker first: survives BaseModel's transformer_options.copy().
    for kind, key, value in candidates:
        if (
            'latentlab_h3_runtime_backend' in value
            or 'latentlab_disable_dynamic_vbar_prefetch' in value
        ):
            transformer_options = value
            location = f'{kind}[{key}]'
            break

    # Fallback to recognizable transformer-options structure.
    if transformer_options is None:
        for kind, key, value in candidates:
            if any(
                marker in value
                for marker in (
                    'patches_replace',
                    'wrappers',
                    'prefetch_dynamic_vbars',
                    'sigmas',
                )
            ):
                transformer_options = value
                location = f'{kind}[{key}]'
                break

    if transformer_options is None:
        _lm_print(
            '[MiniMaxH3 LongMedia][INT8 PREFETCH V7] WARNING: '
            'DIFFUSION_MODEL wrapper could not locate transformer_options; '
            f'dict_candidates={[(k, x) for k, x, _ in candidates]}',
            flush=True,
        )
        return executor(*call_args, **kwargs)

    backend = str(
        transformer_options.get(
            'latentlab_h3_runtime_backend',
            transformer_options.get('model_runtime_backend', 'unknown'),
        )
    ).lower()
    disable_requested = bool(
        transformer_options.get(
            'latentlab_disable_dynamic_vbar_prefetch', False
        )
    )

    before = transformer_options.get('prefetch_dynamic_vbars', '<missing>')

    if disable_requested:
        transformer_options['prefetch_dynamic_vbars'] = False

        if not transformer_options.get(
            'latentlab_prefetch_disable_announced_v8', False
        ):
            _lm_print(
                '[MiniMaxH3 LongMedia][0.3.52 PREFETCH HARD-GATE] '
                f'located transformer_options at {location}; '
                f'prefetch_dynamic_vbars {before!r}->False AFTER BaseModel override; '
                f'backend={backend}; synchronous one-block demand loading active',
                flush=True,
            )
            transformer_options['latentlab_prefetch_disable_announced_v8'] = True

    # v0.3.63: reopen AIMDO VBAR residency at the authoritative diffusion-forward
    # boundary, not in the sampler callback.  The first forward is deliberately
    # left untouched (proven safe in v0.3.61); subsequent forwards may reset the
    # model VBAR watermark only when real CUDA driver-free memory is above the
    # mode-specific floor.  This changes residency policy only, never H3 math.
    residency_state = transformer_options.get('latentlab_h3_residency_state') or _bound_residency_state
    residency_patcher = transformer_options.get('latentlab_h3_residency_patcher') or _bound_residency_patcher
    if disable_requested and isinstance(residency_state, dict) and residency_patcher is not None:
        fwd = int(residency_state.get('vbar_forward_count', 0)) + 1
        residency_state['vbar_forward_count'] = fwd
        if fwd <= 3:
            _lm_print('[MiniMaxH3 LongMedia][0.3.64 VBAR BOUND] ' f'forward={fwd} state_bound={_bound_residency_state is not None} patcher_bound={_bound_residency_patcher is not None}', flush=True)
        mode = str(residency_state.get('memory_policy_mode', residency_state.get('memory_mode', 'normal')))
        floors = {
            'normal': 2304.0,
            'low_vram': 3072.0,
            'ultra_low_vram': 4096.0,
        }
        floor_mb = float(floors.get(mode, floors['low_vram']))
        # Never promote before the first complete H3 forward.  This preserves the
        # exact startup behaviour that made the 32.4 GB checkpoint stable on 16 GB.
        if fwd > 1 and residency_state.get('vbar_first_forward_complete', False):
            try:
                free_b, total_b = torch.cuda.mem_get_info(torch.cuda.current_device())
                free_mb = float(free_b) / (1024.0 ** 2)
                residency_state['vbar_forward_free_mb'] = free_mb
                vbar_get = getattr(residency_patcher, '_vbar_get', None)
                vbar = vbar_get(create=False) if callable(vbar_get) else None
                loaded_before = int(vbar.loaded_size()) if vbar is not None and hasattr(vbar, 'loaded_size') else 0
                residency_state['vbar_forward_loaded_before'] = loaded_before
                if vbar is not None and free_mb >= floor_mb:
                    # One prioritize per diffusion forward is enough. AIMDO will
                    # naturally lower the watermark again if this forward creates
                    # pressure, so we never pin or force-load weights ourselves.
                    vbar.prioritize()
                    residency_state['vbar_forward_promote_count'] = int(residency_state.get('vbar_forward_promote_count', 0)) + 1
                    residency_state['vbar_last_promote_free_mb'] = free_mb
                    _lm_print(
                        '[MiniMaxH3 LongMedia][0.3.64 VBAR FORWARD PROMOTE] '
                        f'forward={fwd} mode={mode} free={free_mb:.0f}MB '
                        f'floor={floor_mb:.0f}MB loaded_before={loaded_before/(1024.0**2):.0f}MB; '
                        'watermark reopened before H3 forward',
                        flush=True,
                    )
                else:
                    residency_state['vbar_forward_skip_pressure'] = int(residency_state.get('vbar_forward_skip_pressure', 0)) + 1
                    # Announce pressure skips sparsely so the console remains usable.
                    if fwd <= 3 or fwd % 4 == 0:
                        _lm_print(
                            '[MiniMaxH3 LongMedia][0.3.64 VBAR FORWARD HOLD] '
                            f'forward={fwd} mode={mode} free={free_mb:.0f}MB '
                            f'floor={floor_mb:.0f}MB loaded={loaded_before/(1024.0**2):.0f}MB',
                            flush=True,
                        )
            except Exception as exc:
                residency_state['vbar_forward_error'] = f'{type(exc).__name__}: {exc}'
                if not residency_state.get('vbar_forward_error_announced'):
                    residency_state['vbar_forward_error_announced'] = True
                    _lm_print(
                        '[MiniMaxH3 LongMedia][0.3.64 VBAR FORWARD] disabled: ' + residency_state['vbar_forward_error'],
                        flush=True,
                    )

    # Diagnostic invariant: when out-of-core streaming requested this MUST be
    # False immediately before MiniMaxH3Model._forward calls make_prefetch_queue().
    if disable_requested:
        effective = transformer_options.get('prefetch_dynamic_vbars', '<missing>')
        if not transformer_options.get('latentlab_prefetch_effective_announced_v8', False):
            _lm_print(
                '[MiniMaxH3 LongMedia][0.3.52 PREFETCH HARD-GATE CHECK] '
                f'effective prefetch_dynamic_vbars={effective!r} '
                f'at DIFFUSION_MODEL boundary ({location})',
                flush=True,
            )
            transformer_options['latentlab_prefetch_effective_announced_v8'] = True

    _h3_result = executor(*call_args, **kwargs)
    if disable_requested and isinstance(residency_state, dict):
        residency_state['vbar_first_forward_complete'] = True
    return _h3_result



def _detect_h3_model_runtime(model_patcher):
    """Best-effort, side-effect-free diffusion quantization/runtime detector.

    IMPORTANT: logical module.weight dtype is not sufficient for Comfy quantized
    models. NVFP4/INT8 may expose BF16 logical weights while the actual execution
    format is carried by QuantizedTensor/layout metadata.
    """
    profile = {
        'backend': 'unknown',
        'model_class': None,
        'diffusion_class': None,
        'weight_dtypes': {},
        'weight_classes': {},
        'layout_types': {},
        'quant_evidence': [],
        'sampled_modules': 0,
        'quantized_weight_count': 0,
        'error': None,
    }

    try:
        try:
            from comfy.quant_ops import QuantizedTensor
        except Exception:
            QuantizedTensor = ()

        base_model = getattr(model_patcher, 'model', None)
        diffusion = getattr(base_model, 'diffusion_model', None)
        if diffusion is None:
            diffusion = getattr(model_patcher, 'diffusion_model', None)

        if base_model is not None:
            profile['model_class'] = (
                f'{type(base_model).__module__}.{type(base_model).__name__}'
            )
        if diffusion is not None:
            profile['diffusion_class'] = (
                f'{type(diffusion).__module__}.{type(diffusion).__name__}'
            )

        root = diffusion if diffusion is not None else base_model
        if root is None:
            return profile

        dtype_counts = {}
        weight_class_counts = {}
        layout_counts = {}
        evidence = []
        evidence_seen = set()

        def _add_evidence(item):
            item = str(item)
            if item not in evidence_seen and len(evidence) < 40:
                evidence.append(item)
                evidence_seen.add(item)

        def _layout_name(value):
            if value is None:
                return None
            if isinstance(value, type):
                return f'{value.__module__}.{value.__name__}'
            cls = type(value)
            # For instances, include both concrete type and readable repr/name.
            name = getattr(value, '__name__', None)
            if name:
                return f'{cls.__module__}.{cls.__name__}:{name}'
            return f'{cls.__module__}.{cls.__name__}:{str(value)[:120]}'

        for idx, (name, module) in enumerate(root.named_modules()):
            if idx >= 3000:
                break
            profile['sampled_modules'] += 1

            mod_cls = f'{type(module).__module__}.{type(module).__name__}'
            mod_low = mod_cls.lower()

            # Module-level quant layout metadata used by mixed-precision ops.
            layout_value = None
            for attr in ('layout_type', 'weight_layout', 'quant_layout'):
                try:
                    candidate = getattr(module, attr, None)
                except Exception:
                    candidate = None
                if candidate is not None:
                    layout_value = candidate
                    lname = _layout_name(candidate)
                    layout_counts[lname] = int(layout_counts.get(lname, 0)) + 1
                    _add_evidence(f'{name or "<root>"}.{attr}={lname}')

            weight = None
            try:
                weight = getattr(module, 'weight', None)
            except Exception:
                weight = None

            if weight is not None:
                wcls = f'{type(weight).__module__}.{type(weight).__name__}'
                weight_class_counts[wcls] = int(weight_class_counts.get(wcls, 0)) + 1

                dtype = getattr(weight, 'dtype', None)
                if dtype is not None:
                    key = str(dtype)
                    dtype_counts[key] = int(dtype_counts.get(key, 0)) + 1

                # Direct QuantizedTensor detection.
                is_quantized_tensor = False
                try:
                    if QuantizedTensor:
                        is_quantized_tensor = isinstance(weight, QuantizedTensor)
                except Exception:
                    is_quantized_tensor = False

                weight_low = wcls.lower()
                if is_quantized_tensor or 'quantizedtensor' in weight_low:
                    profile['quantized_weight_count'] += 1
                    _add_evidence(
                        f'{name or "<root>"}.weight_class={wcls},dtype={dtype}'
                    )

                    # QuantizedTensor implementations may keep layout as an
                    # instance or a class under different attribute names.
                    for attr in (
                        'layout', 'layout_type', '_layout',
                        'tensor_layout', 'quant_layout',
                    ):
                        try:
                            qlayout = getattr(weight, attr, None)
                        except Exception:
                            qlayout = None
                        if qlayout is not None:
                            lname = _layout_name(qlayout)
                            layout_counts[lname] = int(layout_counts.get(lname, 0)) + 1
                            _add_evidence(
                                f'{name or "<root>"}.weight.{attr}={lname}'
                            )

                # Some quant tensors expose an internal data tensor with the real
                # storage dtype (uint8 for NVFP4, int8 for tensorwise INT8).
                for attr in ('qdata', 'data', '_data'):
                    try:
                        qdata = getattr(weight, attr, None)
                    except Exception:
                        qdata = None
                    qdtype = getattr(qdata, 'dtype', None)
                    if qdtype is not None and str(qdtype) in (
                        'torch.uint8', 'torch.int8',
                        'torch.float8_e4m3fn', 'torch.float8_e5m2',
                    ):
                        _add_evidence(
                            f'{name or "<root>"}.weight.{attr}.dtype={qdtype}'
                        )

            # Quantization scales on module are strong evidence even if weight
            # has already been exposed as logical BF16.
            present_scale_attrs = []
            for attr in (
                'weight_scale', 'weight_scale_2', 'input_scale',
                'scale', 'scales',
            ):
                try:
                    value = getattr(module, attr, None)
                except Exception:
                    value = None
                if value is not None:
                    present_scale_attrs.append(attr)
            if present_scale_attrs:
                _add_evidence(
                    f'{name or "<root>"}.scale_attrs={",".join(present_scale_attrs)}'
                )

            # Class-name evidence remains useful for custom loaders.
            if any(term in mod_low for term in (
                'nvfp4', 'int8', 'fp8', 'quant', 'gguf',
                'torchao', 'quanto', 'bitsandbytes',
            )):
                _add_evidence(f'{name or "<root>"}:module={mod_cls}')

        profile['weight_dtypes'] = dtype_counts
        profile['weight_classes'] = weight_class_counts
        profile['layout_types'] = layout_counts
        profile['quant_evidence'] = evidence

        searchable = ' '.join(
            [
                str(profile.get('model_class', '')),
                str(profile.get('diffusion_class', '')),
                *layout_counts.keys(),
                *weight_class_counts.keys(),
                *evidence,
            ]
        ).lower()

        # V28: classify the concrete quant variant independently from the broad
        # backend.  W4A8 checkpoints can also expose TensorWiseINT8Layout, so
        # backend='int8' alone is not sufficient to select a safe activation
        # policy.
        if any(marker in searchable for marker in (
            'asymw4a8int8layout', 'asymw4a8', 'asym_w4a8', 'w4a8',
            'w4a8int8', 'w4a8_mixed',
        )):
            profile['quant_variant'] = 'w4a8'
        elif any(marker in searchable for marker in (
            'tensorcoreconvrotw4a4layout', 'convrot_w4a4', 'w4a4',
        )):
            profile['quant_variant'] = 'convrot-w4a4'
        elif 'tensorwiseint8layout' in searchable or 'int8layout' in searchable:
            profile['quant_variant'] = 'tensorwise-int8'
        else:
            profile['quant_variant'] = None

        # Strongest evidence: named quant layout/class.
        if (
            'tensorcorenvfp4layout' in searchable
            or 'nvfp4layout' in searchable
            or 'nvfp4' in searchable
        ):
            backend = 'nvfp4'
        elif (
            'tensorwiseint8layout' in searchable
            or 'int8layout' in searchable
            or 'int8_tensorwise' in searchable
        ):
            backend = 'int8'
        elif (
            'tensorcoreconvrotw4a4layout' in searchable
            or 'convrot_w4a4' in searchable
        ):
            backend = 'int8-convrot-w4a4'
        elif (
            'tensorcorefp8' in searchable
            or 'float8' in searchable
            or 'fp8layout' in searchable
        ):
            backend = 'fp8'
        elif int(dtype_counts.get('torch.int8', 0)) > 0:
            backend = 'int8'
        elif any(x in searchable for x in (
            'gguf', 'quantizedtensor', 'quant', 'torchao',
            'quanto', 'bitsandbytes',
        )):
            backend = 'quantized-other'
        elif int(dtype_counts.get('torch.bfloat16', 0)) > 0:
            backend = 'bf16'
        elif int(dtype_counts.get('torch.float16', 0)) > 0:
            backend = 'fp16'
        elif int(dtype_counts.get('torch.float32', 0)) > 0:
            backend = 'fp32'
        else:
            backend = 'unknown'

        profile['backend'] = backend
        return profile

    except Exception as exc:
        profile['error'] = f'{type(exc).__name__}: {exc}'
        return profile



def _announce_h3_model_runtime(profile):
    evidence = list(profile.get('quant_evidence') or [])
    _lm_print(
        '[MiniMaxH3 LongMedia][MODEL RUNTIME V2] '
        f'backend={profile.get("backend", "unknown")}, '
        f'diffusion={profile.get("diffusion_class")}, '
        f'weight_dtypes={profile.get("weight_dtypes") or {}}, '
        f'quantized_weights={profile.get("quantized_weight_count", 0)}, '
        f'quant_variant={profile.get("quant_variant")}, '
        f'sampled_modules={profile.get("sampled_modules", 0)}',
        flush=True,
    )
    if profile.get('layout_types'):
        _lm_print(
            '[MiniMaxH3 LongMedia][MODEL RUNTIME LAYOUTS] '
            + str(profile['layout_types']),
            flush=True,
        )
    if profile.get('weight_classes'):
        # Only the most common classes are useful in console.
        classes = sorted(
            profile['weight_classes'].items(),
            key=lambda kv: (-int(kv[1]), kv[0]),
        )[:12]
        _lm_print(
            '[MiniMaxH3 LongMedia][MODEL RUNTIME WEIGHT CLASSES] '
            + str(dict(classes)),
            flush=True,
        )
    if evidence:
        _lm_print(
            '[MiniMaxH3 LongMedia][MODEL RUNTIME EVIDENCE V2] '
            + ' | '.join(evidence[:20]),
            flush=True,
        )
    if profile.get('error'):
        _lm_print(
            '[MiniMaxH3 LongMedia][MODEL RUNTIME WARNING] '
            + str(profile['error']),
            flush=True,
        )




def _h3_runtime_auto_policy(
    backend,
    *,
    quant_variant=None,
    chunk_tokens,
    sol_qkv_chunk_tokens,
    sol_out_proj_chunk_tokens,
    vram_activation_reserve_mb,
):
    """Return conservative backend-aware startup settings.

    NVFP4 is the proven reference path and is intentionally left untouched.
    INT8/BF16-class backends start with more activation headroom and smaller
    activation chunks; block0 AUTO VRAM still adapts guards after the first
    successful block.
    """
    backend = str(backend or 'unknown').lower()

    policy = {
        'backend': backend,
        'name': 'user-defaults',
        'chunk_tokens': int(chunk_tokens),
        'sol_qkv_chunk_tokens': int(sol_qkv_chunk_tokens),
        'sol_out_proj_chunk_tokens': int(sol_out_proj_chunk_tokens),
        'vram_activation_reserve_mb': int(vram_activation_reserve_mb),
    }

    if backend == 'nvfp4':
        policy['name'] = 'nvfp4-proven'
        return policy

    if backend in ('int8', 'int8-convrot-w4a4'):
        quant_variant = str(quant_variant or '').lower()
        policy['quant_variant'] = quant_variant or None
        if quant_variant == 'w4a8':
            # V34: W4A8 throughput pass. The CUDA W4A8 kernel dequantizes grouped
            # int4 weights as part of each Linear invocation, so long-media throughput
            # depends strongly on reducing projection-call count. Keep native kitchen
            # math, but permit larger streamed projection chunks when headroom allows.
            policy['name'] = 'w4a8-cuda-throughput'
            policy['vram_activation_reserve_mb'] = max(
                int(vram_activation_reserve_mb), 5120
            )
            policy['chunk_tokens'] = min(int(chunk_tokens), 8192)
        else:
            policy['name'] = 'int8-native-resident'
            policy['vram_activation_reserve_mb'] = int(vram_activation_reserve_mb)
            policy['chunk_tokens'] = min(int(chunk_tokens), 16384)
        if int(sol_qkv_chunk_tokens) > 0:
            policy['sol_qkv_chunk_tokens'] = min(
                int(sol_qkv_chunk_tokens), 16384 if quant_variant == 'w4a8' else 8192
            )
        if int(sol_out_proj_chunk_tokens) > 0:
            policy['sol_out_proj_chunk_tokens'] = min(
                int(sol_out_proj_chunk_tokens), 16384
            )
        return policy

    if backend in ('bf16', 'fp16', 'fp32'):
        policy['name'] = f'{backend}-conservative'
        policy['vram_activation_reserve_mb'] = max(
            int(vram_activation_reserve_mb), 6144
        )
        policy['chunk_tokens'] = min(int(chunk_tokens), 12288)
        if int(sol_qkv_chunk_tokens) > 0:
            policy['sol_qkv_chunk_tokens'] = min(
                int(sol_qkv_chunk_tokens), 8192
            )
        if int(sol_out_proj_chunk_tokens) > 0:
            policy['sol_out_proj_chunk_tokens'] = min(
                int(sol_out_proj_chunk_tokens), 12288
            )
        return policy

    if backend in ('fp8', 'quantized-other'):
        policy['name'] = f'{backend}-conservative'
        policy['vram_activation_reserve_mb'] = max(
            int(vram_activation_reserve_mb), 5120
        )
        policy['chunk_tokens'] = min(int(chunk_tokens), 16384)
        if int(sol_out_proj_chunk_tokens) > 0:
            policy['sol_out_proj_chunk_tokens'] = min(
                int(sol_out_proj_chunk_tokens), 16384
            )
        return policy

    return policy



def _auto_select_h3_attention_mode(token_count, state):
    """Choose full-token existing/Sage vs embedded Sol. No token compression."""
    s = int(token_count)
    try:
        total_gb = float(torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).total_memory) / (1024.0 ** 3)
    except Exception:
        total_gb = 16.0

    if total_gb <= 18.5:
        threshold = 120000
    elif total_gb <= 26.0:
        threshold = 180000
    elif total_gb <= 36.0:
        threshold = 260000
    else:
        threshold = 360000

    mode = 'existing' if s < threshold else 'sol'
    return mode, (
        f'{s} tokens, VRAM={total_gb:.1f} GB, '
        f'threshold={threshold} -> {mode}'
    )



def _sol_schedule_tau(transformer_options, state):
    """Geometry-aware Sol tau scheduler used by AUTO high-res mode.

    AUTO keeps the proven 1.70 -> 2.10 base schedule and adds a modest,
    bounded boost from packed token count.  This is a routing-policy change
    only: no H3 tokens are merged, pooled, or discarded.
    """
    mode = str(state.get('sol_mode', 'existing'))
    requested = str(state.get('requested_attention_mode', mode))

    sigma = None
    opts = transformer_options or {}
    for key in ('sigmas', 'sigma', 'timestep'):
        value = opts.get(key)
        if value is None:
            continue
        try:
            if torch.is_tensor(value):
                sigma = float(value.flatten()[0].detach().float().cpu().item())
            else:
                sigma = float(value)
            break
        except Exception:
            pass

    tau_start = float(state.get('sol_tau_start', 1.3))
    tau_end = float(state.get('sol_tau_end', 0.8))
    sigma_hi = float(state.get('sol_sigma_hi', 1.0))
    sigma_lo = float(state.get('sol_sigma_lo', 0.0))
    curve = str(state.get('sol_curve', 'linear'))

    token_count = int(state.get('current_token_count', 0) or 0)
    auto_speed = (
        requested == 'auto'
        and mode == 'sol'
        and token_count >= 120000
    )

    # V39: quantized backends use the SAME geometry-aware SOL routing policy as
    # the proven NVFP4 path.  V38 showed that the old numerical-error calibration
    # was the wrong control signal for SOL throughput: a 1% rel-RMS budget selected
    # tau=-1.50 and still routed ~94% exact.  Backend-specific tau overrides are
    # therefore removed; AUTO existing/SOL selection remains unchanged.
    if auto_speed and _v12_is_int8_family(state) and not state.get('v39_policy_parity_announced', False):
        _lm_print(
            '[MiniMaxH3 LongMedia][V39 SOL POLICY PARITY] '
            'INT8/W4A8 uses the same AUTO geometry-aware tau schedule as NVFP4; '
            'legacy V16/V37 quality-budget tau override disabled',
            flush=True,
        )
        state['v39_policy_parity_announced'] = True

    if auto_speed:
        # v0.3.24: keep NVFP4 on the proven quality-safe AUTO SOL schedule, but
        # bias INT8/ConvRot slightly denser (lower tau) to reduce early geometry
        # instability under the same seed/conditioning.  This is deliberately a
        # modest routing change, not a revival of the old aggressive quality-budget
        # calibration path from V16/V37.
        is_int8_family = _v12_is_int8_family(state)
        if is_int8_family:
            base_start = 1.00
            base_end = 1.45
            boost_cap = 0.08
            boost_scale = 80000.0
        else:
            base_start = 1.30
            base_end = 1.85
            boost_cap = 0.15
            boost_scale = 60000.0

        token_boost = 0.0
        if token_count > 150000:
            token_boost = min(
                boost_cap,
                max(
                    0.0,
                    (float(token_count) - 150000.0) / boost_scale * 0.12,
                ),
            )

        tau_start = base_start + token_boost
        tau_end = base_end + token_boost
        curve = 'linear'
        state['sol_geometry_tau_boost'] = float(token_boost)
        state['sol_geometry_tau_profile'] = (
            'int8_quality_safe' if is_int8_family else 'nvfp4_quality_safe'
        )

        announce_key = 'sol_speed_tau_announced_v324'
        if not state.get(announce_key):
            label = 'AUTO GEO TAU INT8 SAFE' if is_int8_family else 'AUTO GEO TAU'
            _lm_print(
                f'[MiniMaxH3 LongMedia][{label}] '
                f'base {base_start:.2f}->{base_end:.2f}, '
                f'tokens={token_count}, boost={token_boost:.3f} => '
                f'{tau_start:.2f}->{tau_end:.2f}',
                flush=True,
            )
            state[announce_key] = True
            state['sol_speed_tau_announced'] = True

    if mode == 'sol' and not auto_speed:
        return tau_start
    if mode not in ('scheduled_sol', 'sol'):
        return None
    if sigma is None:
        return tau_start

    denom = max(1.0e-8, sigma_hi - sigma_lo)
    progress = max(0.0, min(1.0, (sigma_hi - sigma) / denom))
    if curve == 'ease_in':
        progress *= progress
    elif curve == 'ease_out':
        progress = 1.0 - (1.0 - progress) * (1.0 - progress)
    elif curve == 'smoothstep':
        progress = progress * progress * (3.0 - 2.0 * progress)

    tau = tau_start + (tau_end - tau_start) * progress
    state['last_sol_tau'] = float(tau)
    return float(tau)




def _int8_sync_cast_stream(state, *, block_index=None):
    """Synchronize Comfy's cast/offload stream before Sol storage allocation.

    INT8 dynamic/on-demand casting may use a secondary CUDA stream.  A failure
    on that stream can otherwise surface later at an unrelated torch.empty(),
    making the Sol storage allocator look like the source of the OOM.
    """
    backend = str(state.get('model_runtime_backend', 'unknown')).lower()
    if backend not in ('int8', 'int8-convrot-w4a4'):
        return True

    try:
        stream = getattr(comfy.model_management, 'offload_stream', None)
        if stream is not None:
            stream.synchronize()
        return True
    except Exception as exc:
        _lm_print(
            '[MiniMaxH3 LongMedia][INT8 CAST SYNC] failed before Sol storage: '
            f'block={block_index}, {type(exc).__name__}: {exc}',
            flush=True,
        )
        raise



def _int8_prepare_block_linear(linear, probe_input):
    """Cast/stream one Comfy linear once, keep it valid until explicit release."""
    import comfy.ops
    weight, bias, offload_state = comfy.ops.cast_bias_weight(
        linear,
        probe_input,
        offloadable=True,
        compute_dtype=probe_input.dtype,
        want_requant=True,
    )
    return {
        'linear': linear,
        'weight': weight,
        'bias': bias,
        'offload_state': offload_state,
    }


def _int8_release_block_linear(handle):
    if not handle:
        return
    import comfy.ops
    comfy.ops.uncast_bias_weight(
        handle['linear'],
        handle['weight'],
        handle['bias'],
        handle['offload_state'],
    )


def _int8_cached_linear(handle, x, *, input_act=None):
    """Use a block-resident cast weight with stock Comfy quantized semantics.

    Important:
    - ordinary Linear must go through F.linear(x, QuantizedTensor, bias) so
      comfy-kitchen/QuantizedTensor dispatch can honor layout metadata.
    - only fused input activation mirrors comfy.ops.linear_input_act and calls
      ck.int8_linear directly when the weight is a non-transposed
      TensorWiseINT8Layout.
    """
    import comfy.quant_ops

    weight = handle['weight']
    bias = handle['bias']

    # Stock comfy Linear semantics: QuantizedTensor __torch_dispatch__ decides
    # the correct kernel/layout. Do NOT manually unpack qdata for qkv/fc1/out.
    if input_act is None:
        return torch.nn.functional.linear(x, weight, bias)

    # Mirror comfy.ops.linear_input_act exactly for the cached cast result.
    def _apply_input_act(value, act):
        if act == 'swiglu':
            gate, up = value.chunk(2, dim=-1)
            return torch.nn.functional.silu(gate).mul_(up)
        if act == 'gelu_tanh':
            return torch.nn.functional.gelu(value, approximate='tanh')
        raise ValueError(f'unsupported cached input_act={act!r}')

    QuantizedTensor = getattr(comfy.quant_ops, 'QuantizedTensor', ())
    is_quant = isinstance(weight, QuantizedTensor) if QuantizedTensor else False

    if (
        not is_quant
        or getattr(weight, '_layout_cls', None) != 'TensorWiseINT8Layout'
        or getattr(getattr(weight, '_params', None), 'transposed', False)
    ):
        return torch.nn.functional.linear(
            _apply_input_act(x, input_act),
            weight,
            bias,
        )

    qdata, scale = comfy.quant_ops.TensorWiseINT8Layout.get_plain_tensors(weight)
    return comfy.quant_ops.ck.int8_linear(
        x,
        qdata,
        scale,
        bias,
        x.dtype,
        convrot=getattr(weight._params, 'convrot', False),
        convrot_groupsize=getattr(weight._params, 'convrot_groupsize', 256),
        input_act=input_act,
    )


_V34_KERNEL_BACKEND_ANNOUNCED = set()


def _v33_kitchen_impl(func_name, kwargs):
    """Prefer comfy-kitchen's native CUDA implementation on NVIDIA.

    The registry normally prefers CUDA already, but this makes the long-media
    quant path explicit and logs the implementation actually selected.  If the
    CUDA implementation rejects a shape/version, fall back to normal registry
    dispatch without changing math.
    """
    from comfy_kitchen.registry import registry
    backend = None
    impl = None
    if torch.version.cuda is not None and torch.cuda.is_available():
        try:
            impl = registry.get_implementation(func_name, backend='cuda', kwargs=kwargs)
            backend = 'cuda'
        except Exception:
            impl = None
    if impl is None:
        try:
            backend = registry.get_capable_backend(func_name, kwargs=kwargs)
            impl = registry.get_implementation(func_name, backend=backend, kwargs=kwargs)
        except Exception:
            impl = registry.get_implementation(func_name, kwargs=kwargs)
            backend = getattr(impl, '__module__', 'auto')
    key = (func_name, str(backend))
    if key not in _V34_KERNEL_BACKEND_ANNOUNCED:
        _V34_KERNEL_BACKEND_ANNOUNCED.add(key)
        _lm_print(
            '[MiniMaxH3 LongMedia][V40 KITCHEN BACKEND] '
            f'{func_name} -> {backend} ({getattr(impl, "__module__", "?")}.{getattr(impl, "__name__", "?")})',
            flush=True,
        )
    return impl


def _v32_quant_linear_rows(handle, x, row_start, row_end):
    """Run a row slice of a prepared native Comfy quantized Linear.

    Q/K/V occupy independent contiguous output rows in H3 qkv_proj.  Calling
    the official comfy-kitchen kernel on only the rows required by the current
    streaming phase avoids computing throw-away Q during KV build and
    throw-away K/V during query replay.  No quantization math is reimplemented.
    """
    weight = handle['weight']
    bias = handle['bias']
    layout = getattr(weight, '_layout_cls', None)
    params = getattr(weight, '_params', None)
    if params is None or bool(getattr(params, 'transposed', False)):
        return None
    rs, re = int(row_start), int(row_end)
    if rs < 0 or re <= rs:
        return None
    b = None if bias is None else bias[rs:re]

    # Stock INT8 TensorWise / ConvRot path.
    if layout == 'TensorWiseINT8Layout':
        try:
            qdata, scale = comfy.quant_ops.TensorWiseINT8Layout.get_plain_tensors(weight)
            kwargs = {
                'x': x.contiguous(),
                'weight': qdata[rs:re].contiguous(),
                'weight_scale': scale[rs:re],
                'bias': b,
                'out_dtype': x.dtype,
                'convrot': getattr(params, 'convrot', False),
                'convrot_groupsize': getattr(params, 'convrot_groupsize', 256),
            }
            impl = _v33_kitchen_impl('int8_linear', kwargs)
            return impl(**kwargs)
        except Exception:
            return None

    # Stock grouped W4A8 path. The layout itself routes to this exact kitchen op;
    # slicing output rows is mathematically identical to slicing Linear outputs.
    if layout == 'AsymW4A8Int8Layout':
        try:
            from comfy_kitchen.tensor.w4a8_int8 import (
                AsymW4A8Int8Layout, w4a8_int8_linear,
            )
            qdata, s_rel, s_channel, correction, codebook = (
                AsymW4A8Int8Layout.get_plain_tensors(weight)
            )
            corr = None if correction is None else correction[:, rs:re]
            kwargs = {
                'x': x,
                'qdata': qdata[rs:re],
                's_rel': s_rel[rs:re],
                's_channel': s_channel[rs:re],
                'codebook': codebook,
                'correction': corr,
                'bias': b,
                'group_size': getattr(params, 'group_size', 16),
                'convrot_groupsize': getattr(params, 'convrot_groupsize', 256),
                'out_dtype': x.dtype,
            }
            impl = _v33_kitchen_impl('w4a8_int8_linear', kwargs)
            return impl(**kwargs)
        except Exception:
            return None
    return None


def _v12b_linear_ab_enabled(state, label):
    """Run each numeric A/B once, only on INT8/W4A8 block 0 / forward 1."""
    if not _v12_is_int8_family(state):
        return False
    if int(state.get('active_block_index', -1)) != 0:
        return False
    if int(state.get('v12_int8_sol_forward_generation', 0) or 0) != 1:
        return False
    done = state.setdefault('v12b_linear_ab_done', {})
    return not bool(done.get(str(label), False))


def _v12b_linear_ab_report(state, label, stock, cached):
    """Compare real stock/cached outputs without retaining GPU-sized tensors."""
    label = str(label)
    done = state.setdefault('v12b_linear_ab_done', {})
    try:
        stock32 = stock.detach().to(device='cpu', dtype=torch.float32)
        cached32 = cached.detach().to(device='cpu', dtype=torch.float32)
        if tuple(stock32.shape) != tuple(cached32.shape):
            _lm_print(
                '[MiniMaxH3 LongMedia][V12-B LINEAR A/B] '
                f'{label}: SHAPE MISMATCH stock={tuple(stock32.shape)} '
                f'cached={tuple(cached32.shape)}',
                flush=True,
            )
            done[label] = True
            return

        stock_flat = stock32.flatten()
        cached_flat = cached32.flatten()
        stock_finite = bool(torch.isfinite(stock_flat).all().item())
        cached_finite = bool(torch.isfinite(cached_flat).all().item())
        diff = cached_flat - stock_flat
        diff_finite = bool(torch.isfinite(diff).all().item())

        if stock_flat.numel() == 0:
            rel_rms = max_abs = mean_abs = 0.0
            cosine = 1.0
        else:
            eps = 1.0e-12
            rms_ref = torch.sqrt(torch.mean(stock_flat.square())).item()
            rms_diff = torch.sqrt(torch.mean(diff.square())).item()
            rel_rms = float(rms_diff / max(eps, rms_ref))
            max_abs = float(diff.abs().max().item())
            mean_abs = float(diff.abs().mean().item())
            denom = float(
                torch.linalg.vector_norm(stock_flat).item()
                * torch.linalg.vector_norm(cached_flat).item()
            )
            cosine = float(torch.dot(stock_flat, cached_flat).item() / max(eps, denom))

        verdict = (
            'MATCH'
            if stock_finite and cached_finite and diff_finite
            and rel_rms <= 1.0e-5 and cosine >= 0.99999
            else 'DIVERGED'
        )
        _lm_print(
            '[MiniMaxH3 LongMedia][V12-B LINEAR A/B] '
            f'{label}: {verdict}, shape={tuple(stock32.shape)}, '
            f'rel_rms={rel_rms:.8e}, mean_abs={mean_abs:.8e}, '
            f'max_abs={max_abs:.8e}, cosine={cosine:.10f}, '
            f'finite(stock/cached/diff)='
            f'{stock_finite}/{cached_finite}/{diff_finite}',
            flush=True,
        )
    except Exception as exc:
        _lm_print(
            '[MiniMaxH3 LongMedia][V12-B LINEAR A/B] '
            f'{label}: diagnostic failed: {type(exc).__name__}: {exc}',
            flush=True,
        )
    finally:
        done[label] = True


def _v19_selected_block(state, block_index=None):
    # V25 cleanup build: forensic V22 stage probes disabled.
    return False
    """V22: robust first/middle/last H3 selection for the first diagnostic forward.

    Targets are frozen when the patch set is installed, rather than recomputed
    from mutable runtime state while blocks are executing.  A separate forward
    latch is armed on block 0 and stays true until the last target exits.
    """
    if not _v12_is_int8_family(state):
        return False
    block_index = int(
        state.get('active_block_index', -1)
        if block_index is None else block_index
    )
    targets = state.get('v21_stage_ab_targets')
    if not targets:
        last = int(state.get('last_patched_block_index', -1) or -1)
        targets = (0,) if last < 0 else (0, last // 2, last)
    if block_index not in set(int(v) for v in targets):
        return False
    # Arm exactly once when the first INT8 generation reaches block 0.  Do not
    # depend on the generation counter afterwards; other helpers may mutate or
    # release forward-scoped state before the later target blocks run.
    if block_index == int(targets[0]):
        if not state.get('v21_stage_ab_completed', False) and not state.get('v21_stage_ab_armed', False):
            state['v21_stage_ab_armed'] = True
            state['v21_stage_ab_generation'] = int(state.get('v12_int8_sol_forward_generation', 0) or 0)
            _lm_print(
                '[MiniMaxH3 LongMedia][V22 STAGE A/B TARGET] '
                f'armed generation={state["v21_stage_ab_generation"]}, targets={list(targets)}',
                flush=True,
            )
    return bool(state.get('v21_stage_ab_armed', False))


def _v19_probe_offsets(token_count, chunk_hint=8192):
    """Four aligned probe regions matching the successful V17/V18 samples."""
    token_count = int(token_count)
    chunk = max(64, (int(chunk_hint or 8192) // 64) * 64)
    return sorted({
        0,
        min(chunk, max(0, token_count - 1)),
        (token_count // 2 // chunk) * chunk,
        ((token_count - 1) // chunk) * chunk,
    })


def _v19_report(state, stage, reference, candidate, *, offsets=None):
    """Report a compact combined A/B across all V19 token probes."""
    block_index = int(state.get('active_block_index', -1))
    key = f'{block_index}:{stage}'
    done = state.setdefault('v19_stage_ab_done', set())
    if key in done:
        return
    try:
        ref = reference.detach().to(device='cpu', dtype=torch.float32)
        got = candidate.detach().to(device='cpu', dtype=torch.float32)
        if tuple(ref.shape) != tuple(got.shape):
            _lm_print(
                '[MiniMaxH3 LongMedia][V22 MULTI-BLOCK STAGE A/B] '
                f'block={block_index}, stage={stage}, SHAPE-MISMATCH '
                f'reference={tuple(ref.shape)}, candidate={tuple(got.shape)}',
                flush=True,
            )
            return
        ref_flat = ref.flatten()
        got_flat = got.flatten()
        diff = got_flat - ref_flat
        finite = bool(
            torch.isfinite(ref_flat).all().item()
            and torch.isfinite(got_flat).all().item()
            and torch.isfinite(diff).all().item()
        )

        # V20: never let a numerically noisy FP32 cosine turn a bit-exact
        # comparison into DIVERGED.  V19 exposed this on the real workload:
        # rel_rms/mean_abs/max_abs were all exactly zero while FP32 dot/norm
        # produced cosine=0.99998 (and even >1.0 on another stage).
        exact = bool(torch.equal(ref_flat, got_flat))
        mismatch_count = int(torch.count_nonzero(diff).item())
        if exact:
            rel_rms = 0.0
            cosine = 1.0
            mean_abs = 0.0
            max_abs = 0.0
        else:
            eps = 1.0e-30
            ref64 = ref_flat.to(dtype=torch.float64)
            got64 = got_flat.to(dtype=torch.float64)
            diff64 = got64 - ref64
            rms_ref = float(torch.sqrt(torch.mean(ref64.square())).item())
            rms_diff = float(torch.sqrt(torch.mean(diff64.square())).item())
            rel_rms = rms_diff / max(eps, rms_ref)
            norm_ref = float(torch.linalg.vector_norm(ref64).item())
            norm_got = float(torch.linalg.vector_norm(got64).item())
            denom = norm_ref * norm_got
            if denom <= eps:
                cosine = 1.0 if rms_diff <= eps else 0.0
            else:
                cosine = float(torch.dot(ref64, got64).item() / denom)
                cosine = max(-1.0, min(1.0, cosine))
            mean_abs = float(diff64.abs().mean().item())
            max_abs = float(diff64.abs().max().item())
            del ref64, got64, diff64

        verdict = (
            'MATCH'
            if finite and (exact or (rel_rms <= 1.0e-5 and cosine >= 0.99999))
            else 'DIVERGED'
        )
        _lm_print(
            '[MiniMaxH3 LongMedia][V22 MULTI-BLOCK STAGE A/B] '
            f'block={block_index}, stage={stage}, verdict={verdict}, '
            f'offsets={list(offsets or [])}, rows={int(ref.shape[0])}, '
            f'exact={exact}, mismatches={mismatch_count}, '
            f'rel_rms={rel_rms:.8e}, cosine={cosine:.10f}, '
            f'mean_abs={mean_abs:.8e}, max_abs={max_abs:.8e}, finite={finite}',
            flush=True,
        )
    except Exception as exc:
        _lm_print(
            '[MiniMaxH3 LongMedia][V22 MULTI-BLOCK STAGE A/B] '
            f'block={block_index}, stage={stage}, diagnostic failed: '
            f'{type(exc).__name__}: {exc}',
            flush=True,
        )
    finally:
        done.add(key)


def _v13_exact_attention_from_compressed(q, storage, key_chunk=1024):
    """Exact softmax attention over the token-level compressed Sol K/V.

    Only a few query rows are used by the caller. Keys and values are
    reconstructed in bounded chunks, so this never materializes the full
    Q-by-K attention matrix.
    """
    q32 = q.detach().to(dtype=torch.float32)
    batch, queries, heads, head_dim = q32.shape
    tokens = int(storage['tokens'])
    if batch != int(storage['k8'].shape[0]):
        raise ValueError('V13 exact A/B batch geometry mismatch')

    running_max = torch.full(
        (batch, heads, queries),
        -float('inf'),
        device=q.device,
        dtype=torch.float32,
    )
    running_sum = torch.zeros_like(running_max)
    running_out = torch.zeros(
        (batch, heads, queries, head_dim),
        device=q.device,
        dtype=torch.float32,
    )
    scale = float(head_dim ** -0.5)

    for key_start in range(0, tokens, int(key_chunk)):
        key_end = min(tokens, key_start + int(key_chunk))
        block_indices = (
            torch.arange(key_start, key_end, device=q.device, dtype=torch.long)
            // 64
        )
        k = (
            storage['k8'][:, key_start:key_end].to(dtype=torch.float32)
            * storage['ks'][:, key_start:key_end].unsqueeze(-1)
            + storage['kc'][:, block_indices].to(dtype=torch.float32)
        ).to(dtype=torch.bfloat16).to(dtype=torch.float32)
        v = (
            storage['v8'][:, key_start:key_end].to(dtype=torch.float32)
            * storage['vs'][:, key_start:key_end].unsqueeze(-1)
        ).to(dtype=torch.bfloat16).to(dtype=torch.float32)

        scores = torch.einsum('bqhd,bkhd->bhqk', q32, k) * scale
        chunk_max = scores.amax(dim=-1)
        next_max = torch.maximum(running_max, chunk_max)
        previous_scale = torch.exp(running_max - next_max)
        probability = torch.exp(scores - next_max.unsqueeze(-1))
        running_out = (
            running_out * previous_scale.unsqueeze(-1)
            + torch.einsum('bhqk,bkhd->bhqd', probability, v)
        )
        running_sum = (
            running_sum * previous_scale
            + probability.sum(dim=-1)
        )
        running_max = next_max
        del block_indices, k, v, scores, chunk_max, next_max
        del previous_scale, probability

    return (
        running_out / running_sum.unsqueeze(-1)
    ).permute(0, 2, 1, 3).contiguous()


def _v15_attention_ab_report(label, exact, candidate, storage, tau):
    """Report one candidate against an already-computed exact reference."""
    exact_flat = exact.detach().to(device='cpu', dtype=torch.float32).flatten()
    candidate_flat = (
        candidate[:, :exact.shape[1]]
        .detach()
        .to(device='cpu', dtype=torch.float32)
        .flatten()
    )
    diff = candidate_flat - exact_flat
    exact_finite = bool(torch.isfinite(exact_flat).all().item())
    candidate_finite = bool(torch.isfinite(candidate_flat).all().item())
    diff_finite = bool(torch.isfinite(diff).all().item())
    eps = 1.0e-12
    rms_ref = float(torch.sqrt(torch.mean(exact_flat.square())).item())
    rms_diff = float(torch.sqrt(torch.mean(diff.square())).item())
    rel_rms = rms_diff / max(eps, rms_ref)
    mean_abs = float(diff.abs().mean().item())
    max_abs = float(diff.abs().max().item())
    denom = float(
        torch.linalg.vector_norm(exact_flat).item()
        * torch.linalg.vector_norm(candidate_flat).item()
    )
    cosine = float(torch.dot(exact_flat, candidate_flat).item() / max(eps, denom))
    _lm_print(
        '[MiniMaxH3 LongMedia][V15 TAU CALIBRATION] '
        f'{label}: block=0, forward=1, queries={int(exact.shape[1])}, '
        f'keys={int(storage["tokens"])}, tau={float(tau):.4f}, '
        f'rel_rms={rel_rms:.8e}, mean_abs={mean_abs:.8e}, '
        f'max_abs={max_abs:.8e}, cosine={cosine:.10f}, '
        f'finite(exact/candidate/diff)='
        f'{exact_finite}/{candidate_finite}/{diff_finite}',
        flush=True,
    )



def _v37_auto_calibrate_sol_tau(state, q, storage, sink_blocks, sink_q, q_offset):
    """V38: one-shot *full query-chunk* SOL speed/quality calibration.

    V37 measured quality on 64 rows, but its timing was also for those 64 rows and
    therefore did not predict the real 8192-row query replay cost.  V38 keeps the
    exact same SOL math and AUTO existing/SOL policy, but benchmarks each tau on
    the actual first query chunk after one warmup.  Quality is compared on a small
    prefix against compressed-exact to keep the reference cheap; runtime/exact
    routing are measured on the full chunk.  We print feasible points for several
    numerical budgets so one run exposes the whole speed/quality curve.
    """
    # V39: retained for forensic history only.  Do not alter runtime tau; the
    # scheduler above now owns quantized SOL policy exactly as it does for NVFP4.
    return None

    if not _v12_is_int8_family(state):
        return None
    if str(state.get('requested_attention_mode', '')).lower() != 'auto':
        return None
    if str(state.get('sol_mode', '')).lower() != 'sol':
        return None
    if state.get('v37_auto_quality_tau') is not None:
        return float(state['v37_auto_quality_tau'])
    if bool(state.get('v37_tau_autocal_running', False)):
        return None
    if int(state.get('active_block_index', -1)) != 0 or int(q_offset) != 0:
        return None

    from .sol_kernel import sol_attn_query_compressed_sm120
    state['v37_tau_autocal_running'] = True
    quality_budget = float(state.get('v37_tau_quality_budget', 0.0100))
    # Wide enough to expose the useful speed/quality knee, but not so many points
    # that calibration itself becomes a minutes-long benchmark.
    candidates = (2.00, 1.50, 1.00, 0.50, 0.00, -0.50, -1.00, -1.50, -2.00)
    try:
        full_rows = int(q.shape[1])
        compare_rows = min(8, full_rows)
        q_full = q
        exact = _v13_exact_attention_from_compressed(
            q_full[:, :compare_rows], storage, key_chunk=1024
        ).float()
        exact_rms = torch.sqrt(torch.mean(exact.square())).clamp_min(1.0e-12)

        # Warm up the real 8192-row kernel once so compilation/first-use cost does
        # not poison the first candidate's timing.
        _warm, _warm_telem = sol_attn_query_compressed_sm120(
            q_full, storage, q_offset=int(q_offset), tau=-2.0,
            sink_blocks=sink_blocks, sink_q=sink_q, telemetry=True,
        )
        torch.cuda.synchronize(q.device)
        del _warm, _warm_telem
        _lm_print(
            '[MiniMaxH3 LongMedia][V38 FULL-CHUNK SWEEP] '
            f'warmup complete; rows={full_rows}, quality_compare_rows={compare_rows}, '
            f'candidates={len(candidates)}',
            flush=True,
        )

        rows = []
        selected = None
        for candidate_tau in candidates:
            torch.cuda.synchronize(q.device)
            torch.cuda.reset_peak_memory_stats(q.device)
            t0 = time.perf_counter()
            candidate, telem = sol_attn_query_compressed_sm120(
                q_full, storage, q_offset=int(q_offset), tau=float(candidate_tau),
                sink_blocks=sink_blocks, sink_q=sink_q, telemetry=True,
            )
            torch.cuda.synchronize(q.device)
            elapsed = time.perf_counter() - t0
            cand_cmp = candidate[:, :compare_rows].float()
            diff = cand_cmp - exact
            rel_rms = float((torch.sqrt(torch.mean(diff.square())) / exact_rms).item())
            denom = (torch.linalg.vector_norm(exact) * torch.linalg.vector_norm(cand_cmp)).clamp_min(1.0e-12)
            cosine = float((torch.sum(exact * cand_cmp) / denom).item())
            exact_ratio = float(telem.get('exact_ratio', 1.0))
            peak_mb = float(torch.cuda.max_memory_allocated(q.device)) / (1024.0 * 1024.0)
            rows.append((float(candidate_tau), rel_rms, cosine, exact_ratio, elapsed, peak_mb))
            _lm_print(
                '[MiniMaxH3 LongMedia][V38 TAU FULL] '
                f'tau={candidate_tau:+.2f} rel_rms={rel_rms:.6f} cosine={cosine:.7f} '
                f'exact={exact_ratio*100.0:.2f}% kernel={elapsed:.4f}s '
                f'peak_alloc={peak_mb:.0f}MB',
                flush=True,
            )
            if selected is None and rel_rms <= quality_budget and torch.isfinite(diff).all():
                selected = (float(candidate_tau), rel_rms, cosine, exact_ratio, elapsed, peak_mb)
            del candidate, cand_cmp, diff

        # Report the fastest point satisfying several useful quality budgets.
        for budget in (0.01, 0.05, 0.10, 0.15, 0.20):
            feasible = [r for r in rows if r[1] <= budget]
            if feasible:
                best = min(feasible, key=lambda r: r[4])
                _lm_print(
                    '[MiniMaxH3 LongMedia][V38 BUDGET KNEE] '
                    f'budget={budget*100:.0f}% tau={best[0]:+.2f} '
                    f'rel_rms={best[1]:.6f} exact={best[3]*100.0:.2f}% '
                    f'kernel={best[4]:.4f}s',
                    flush=True,
                )
            else:
                _lm_print(
                    '[MiniMaxH3 LongMedia][V38 BUDGET KNEE] '
                    f'budget={budget*100:.0f}% no-candidate',
                    flush=True,
                )

        # Keep V37's conservative automatic behavior for generation.  V38 is a
        # forensic throughput sweep, not a silent quality-policy change.
        if selected is None:
            selected = min(rows, key=lambda r: r[1])
            reason = 'fallback-lowest-error'
        else:
            reason = 'quality-budget'
        tau_sel, err_sel, cos_sel, exact_sel, time_sel, peak_sel = selected
        state['v37_auto_quality_tau'] = float(tau_sel)
        state['v37_auto_quality_rel_rms'] = float(err_sel)
        state['v37_auto_quality_exact_ratio'] = float(exact_sel)
        state['last_sol_tau'] = float(tau_sel)
        _lm_print(
            '[MiniMaxH3 LongMedia][V38 TAU SELECT] '
            f'tau={tau_sel:+.2f} reason={reason} rel_rms={err_sel:.6f} '
            f'cosine={cos_sel:.7f} exact={exact_sel*100.0:.2f}% '
            f'full_chunk_kernel={time_sel:.4f}s budget={quality_budget:.4f}',
            flush=True,
        )
        del exact
        return float(tau_sel)
    except Exception as exc:
        _lm_print(
            '[MiniMaxH3 LongMedia][V38 TAU AUTOCAL] '
            f'failed: {type(exc).__name__}: {exc}; keeping provisional tau=-2.0',
            flush=True,
        )
        return None
    finally:
        state['v37_tau_autocal_running'] = False


def _v15_tau_calibration(
    state, q, storage, sol_output, tau, sink_blocks, sink_q
):
    """Calibrate routed Sol tau against the same exact reference."""
    if not _v12_is_int8_family(state):
        return
    if int(state.get('active_block_index', -1)) != 0:
        return
    if int(state.get('v12_int8_sol_forward_generation', 0) or 0) != 1:
        return
    if bool(state.get('v15_tau_calibration_done', False)):
        return

    # Mark first so a diagnostic failure cannot repeat for every query chunk.
    state['v15_tau_calibration_done'] = True
    query_count = min(4, int(q.shape[1]))
    try:
        exact = _v13_exact_attention_from_compressed(
            q[:, :query_count], storage, key_chunk=1024
        )
        _v15_attention_ab_report(
            'CURRENT-ROUTED', exact, sol_output, storage, tau
        )

        # A full 64-row query block is required because Sol makes routing
        # decisions per block. Sweep below the aggressive AUTO value to find
        # the quality/performance knee for this real prompt geometry.
        from .sol_kernel import sol_attn_query_compressed_sm120
        probe_query_rows = min(64, int(q.shape[1]))
        q_probe = q[:, :probe_query_rows]
        for candidate_tau in (1.30, 0.80, 0.00, -1.00, -2.00):
            torch.cuda.synchronize(q.device)
            started = time.perf_counter()
            candidate = sol_attn_query_compressed_sm120(
                q_probe,
                storage,
                q_offset=0,
                tau=float(candidate_tau),
                sink_blocks=sink_blocks,
                sink_q=(0, 0),
            )
            torch.cuda.synchronize(q.device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            _v15_attention_ab_report(
                f'TAU-SWEEP elapsed_ms={elapsed_ms:.1f}',
                exact, candidate, storage, candidate_tau,
            )
            del candidate

        # Forced-exact is the accuracy floor of the compressed Triton path.
        torch.cuda.synchronize(q.device)
        started = time.perf_counter()
        forced_exact = sol_attn_query_compressed_sm120(
            q_probe,
            storage,
            q_offset=0,
            tau=float(tau),
            sink_blocks=sink_blocks,
            sink_q=(0, 1),
        )
        torch.cuda.synchronize(q.device)
        forced_ms = (time.perf_counter() - started) * 1000.0
        _v15_attention_ab_report(
            f'FORCED-EXACT elapsed_ms={forced_ms:.1f}',
            exact, forced_exact, storage, tau,
        )
        _lm_print(
            '[MiniMaxH3 LongMedia][V15 STRIDES] '
            f'q={tuple(q.stride())}, routed={tuple(sol_output.stride())}, '
            f'forced={tuple(forced_exact.stride())}, '
            f'contiguous(routed/forced)='
            f'{sol_output.is_contiguous()}/{forced_exact.is_contiguous()}',
            flush=True,
        )
        del exact, forced_exact
    except Exception as exc:
        _lm_print(
            '[MiniMaxH3 LongMedia][V15 TAU CALIBRATION] '
            f'diagnostic failed: {type(exc).__name__}: {exc}',
            flush=True,
        )


def _v17_multi_offset_calibration(
    state, q, storage, sol_output, tau, sink_blocks,
    q_offset, sequence_tokens, query_chunk,
):
    """Measure routed Sol accuracy across conditioning/video/tail regions."""
    if not _v12_is_int8_family(state):
        return
    if int(state.get('active_block_index', -1)) != 0:
        return
    if int(state.get('v12_int8_sol_forward_generation', 0) or 0) != 1:
        return

    query_offset = int(q_offset)
    sequence_tokens = int(sequence_tokens)
    query_chunk = int(query_chunk)
    targets = {
        0,
        min(query_chunk, max(0, sequence_tokens - 1)),
        (sequence_tokens // 2 // query_chunk) * query_chunk,
        ((sequence_tokens - 1) // query_chunk) * query_chunk,
    }
    if query_offset not in targets:
        return
    done = state.setdefault('v17_calibrated_offsets', set())
    if query_offset in done:
        return
    done.add(query_offset)

    query_count = min(4, int(q.shape[1]))
    probe_rows = min(64, int(q.shape[1]))
    q_probe = q[:, :probe_rows]
    try:
        exact = _v13_exact_attention_from_compressed(
            q[:, :query_count], storage, key_chunk=1024
        )

        def _report(label, candidate, candidate_tau, elapsed_ms=None):
            exact_flat = (
                exact.detach().to(device='cpu', dtype=torch.float32).flatten()
            )
            candidate_flat = (
                candidate[:, :query_count]
                .detach()
                .to(device='cpu', dtype=torch.float32)
                .flatten()
            )
            diff = candidate_flat - exact_flat
            eps = 1.0e-12
            rms_ref = float(torch.sqrt(torch.mean(exact_flat.square())).item())
            rms_diff = float(torch.sqrt(torch.mean(diff.square())).item())
            rel_rms = rms_diff / max(eps, rms_ref)
            denom = float(
                torch.linalg.vector_norm(exact_flat).item()
                * torch.linalg.vector_norm(candidate_flat).item()
            )
            cosine = float(
                torch.dot(exact_flat, candidate_flat).item() / max(eps, denom)
            )
            timing = (
                ''
                if elapsed_ms is None
                else f', elapsed_ms={float(elapsed_ms):.1f}'
            )
            _lm_print(
                '[MiniMaxH3 LongMedia][V17 MULTI-OFFSET TAU] '
                f'offset={query_offset}, label={label}, '
                f'tau={float(candidate_tau):.2f}{timing}, '
                f'rel_rms={rel_rms:.8e}, cosine={cosine:.10f}, '
                f'max_abs={float(diff.abs().max().item()):.8e}, '
                f'finite={bool(torch.isfinite(diff).all().item())}',
                flush=True,
            )

        _report('CURRENT', sol_output, tau)

        from .sol_kernel import sol_attn_query_compressed_sm120
        for candidate_tau in (-3.0, -4.0):
            torch.cuda.synchronize(q.device)
            started = time.perf_counter()
            candidate = sol_attn_query_compressed_sm120(
                q_probe,
                storage,
                q_offset=query_offset,
                tau=candidate_tau,
                sink_blocks=sink_blocks,
                sink_q=(0, 0),
            )
            torch.cuda.synchronize(q.device)
            _report(
                'ROUTED', candidate, candidate_tau,
                (time.perf_counter() - started) * 1000.0,
            )
            del candidate

        global_query_block = query_offset // 64
        torch.cuda.synchronize(q.device)
        started = time.perf_counter()
        forced_exact = sol_attn_query_compressed_sm120(
            q_probe,
            storage,
            q_offset=query_offset,
            tau=float(tau),
            sink_blocks=sink_blocks,
            sink_q=(global_query_block, global_query_block + 1),
        )
        torch.cuda.synchronize(q.device)
        _report(
            'FORCED-EXACT', forced_exact, tau,
            (time.perf_counter() - started) * 1000.0,
        )
        del exact, forced_exact
    except Exception as exc:
        _lm_print(
            '[MiniMaxH3 LongMedia][V17 MULTI-OFFSET TAU] '
            f'offset={query_offset}, diagnostic failed: '
            f'{type(exc).__name__}: {exc}',
            flush=True,
        )



def _v12_is_int8_family(state):
    """True for TensorWise INT8 and Asym W4A8 INT8; never NVFP4."""
    backend = str(state.get('model_runtime_backend', 'unknown')).lower()
    if backend != 'int8':
        return False

    profile = state.get('model_runtime_profile') or {}
    evidence = ' '.join([
        *(str(value) for value in (profile.get('layout_types') or {}).keys()),
        *(str(value) for value in (profile.get('weight_classes') or {}).keys()),
        *(str(value) for value in (profile.get('quant_evidence') or [])),
    ]).lower()
    if 'tensorcorenvfp4' in evidence or 'nvfp4' in evidence:
        return False

    tensorwise_int8 = 'tensorwiseint8layout' in evidence
    asym_w4a8_int8 = any(marker in evidence for marker in (
        'asymw4a8int8layout', 'asymw4a8', 'asym_w4a8', 'w4a8',
    ))
    return tensorwise_int8 or asym_w4a8_int8


def _v12_begin_int8_sol_forward(state):
    """Start a fresh Sol-workspace generation for one INT8/W4A8 forward."""
    if not _v12_is_int8_family(state):
        return False
    state['int8_reusable_sol_storage'] = None
    state['int8_reusable_sol_storage_key'] = None
    generation = int(state.get('v12_int8_sol_forward_generation', 0) or 0) + 1
    state['v12_int8_sol_forward_generation'] = generation
    state['v12_int8_sol_forward_active'] = True
    _lm_print(
        '[MiniMaxH3 LongMedia][V12-A INT8 SOL FORWARD SCOPE] '
        f'begin forward={generation}; fresh workspace required; '
        'NVFP4 path unchanged',
        flush=True,
    )
    return True


def _v12_release_int8_sol_forward(state, *, block_index):
    """Drop all INT8/W4A8 Sol references after the final attention call."""
    if (
        not _v12_is_int8_family(state)
        or not state.get('v12_int8_sol_forward_active')
    ):
        return False
    had_storage = state.get('int8_reusable_sol_storage') is not None
    state['int8_reusable_sol_storage'] = None
    state['int8_reusable_sol_storage_key'] = None
    state['v12_int8_sol_forward_active'] = False
    state['v12_int8_sol_forward_release_count'] = int(
        state.get('v12_int8_sol_forward_release_count', 0) or 0
    ) + 1
    _lm_print(
        '[MiniMaxH3 LongMedia][V12-A INT8 SOL FORWARD SCOPE] '
        f'release forward={state.get("v12_int8_sol_forward_generation", 0)} '
        f'after block={int(block_index)}; storage_present={had_storage}; '
        'next denoise step cannot reuse K/V or kc statistics',
        flush=True,
    )
    return had_storage


def _int8_reusable_sol_storage(state, *, tokens, heads, head_dim, device, allocator):
    """V11 reuse, forward-scoped for positively identified INT8/W4A8."""
    backend = str(state.get('model_runtime_backend', 'unknown')).lower()
    if backend not in ('int8', 'int8-convrot-w4a4'):
        return allocator(1, tokens, heads, head_dim, device), False

    base_key = (
        int(tokens), int(heads), int(head_dim),
        str(device), str(getattr(device, 'index', None)),
    )
    int8_family = _v12_is_int8_family(state)
    if int8_family:
        key = (
            int(state.get('v12_int8_sol_forward_generation', 0) or 0),
            *base_key,
        )
    else:
        # Exact V11 key/behavior for non-INT8-family paths.
        key = base_key
    storage = state.get('int8_reusable_sol_storage')
    old_key = state.get('int8_reusable_sol_storage_key')

    if storage is None or old_key != key:
        storage = allocator(1, tokens, heads, head_dim, device)
        state['int8_reusable_sol_storage'] = storage
        state['int8_reusable_sol_storage_key'] = key
        approx = 0
        for value in storage.values():
            if torch.is_tensor(value):
                approx += value.numel() * value.element_size()
        if int8_family:
            _lm_print(
                '[MiniMaxH3 LongMedia][V12-A INT8 SOL FORWARD SCOPE] allocated: '
                f'forward={state.get("v12_int8_sol_forward_generation", 0)}, '
                f'tokens={tokens}, heads={heads}, head_dim={head_dim}, '
                f'workspace={approx / (1024**2):.1f} MB',
                flush=True,
            )
        else:
            _lm_print(
                '[MiniMaxH3 LongMedia][INT8 REUSABLE SOL] allocated once: '
                f'tokens={tokens}, heads={heads}, head_dim={head_dim}, '
                f'workspace={approx / (1024**2):.1f} MB',
                flush=True,
            )
        return storage, True

    return storage, True


def _int8_pre_sol_storage_guard(state, *, block_index=None, force=False):
    """INT8-only guard immediately before Sol compressed storage allocation.

    The cast/offload stream is synchronized first so async INT8 casting errors
    cannot masquerade as Sol torch.empty() failures.
    """
    backend = str(state.get('model_runtime_backend', 'unknown')).lower()
    if backend not in ('int8', 'int8-convrot-w4a4'):
        return False
    if state.get('int8_reusable_sol_storage') is not None:
        return False
    if not torch.cuda.is_available():
        return False

    _int8_sync_cast_stream(state, block_index=block_index)

    snap = _cuda_memory_snapshot()
    if not snap:
        return False

    mb = 1024.0 ** 2
    free_mb = float(snap['driver_free']) / mb
    cached_mb = float(snap['cached']) / mb

    floor_mb = float(state.get('int8_sol_storage_free_floor_mb', 3072) or 3072)
    emergency_mb = float(
        state.get('int8_sol_storage_emergency_free_mb', 2048) or 2048
    )
    min_cached_mb = float(state.get('int8_sol_storage_min_cached_mb', 1024) or 1024)
    cooldown = int(state.get('int8_sol_storage_guard_cooldown_blocks', 4) or 4)
    cooldown_left = int(state.get('int8_sol_storage_guard_cooldown_left', 0) or 0)

    # Healthy driver-visible memory: keep the allocator cache for speed.
    if not force and free_mb >= floor_mb:
        if cooldown_left > 0:
            state['int8_sol_storage_guard_cooldown_left'] = max(
                0, cooldown_left - 1
            )
        return False

    # Do not churn cache every block unless driver-free is actually dangerous.
    if (
        not force
        and cooldown_left > 0
        and free_mb >= emergency_mb
    ):
        state['int8_sol_storage_guard_cooldown_left'] = max(
            0, cooldown_left - 1
        )
        return False

    if cached_mb < min_cached_mb:
        return False

    before_free = free_mb
    before_cached = cached_mb

    try:
        gc.collect()
        comfy.model_management.soft_empty_cache()
    except Exception as exc:
        _lm_print(
            '[MiniMaxH3 LongMedia][INT8 SOL STORAGE GUARD] cleanup failed: '
            f'block={block_index}, {type(exc).__name__}: {exc}',
            flush=True,
        )
        return False

    state['int8_sol_storage_guard_cooldown_left'] = cooldown
    state['int8_sol_storage_trim_count'] = int(
        state.get('int8_sol_storage_trim_count', 0) or 0
    ) + 1

    after = _cuda_memory_snapshot()
    if after:
        free_after = float(after['driver_free']) / mb
        cached_after = float(after['cached']) / mb
        _lm_print(
            '[MiniMaxH3 LongMedia][INT8 SOL STORAGE GUARD] TRIM: '
            f'block={block_index}, driver_free {before_free:.0f}->{free_after:.0f} MB, '
            f'cached {before_cached:.0f}->{cached_after:.0f} MB, '
            f'floor={floor_mb:.0f}, emergency={emergency_mb:.0f} MB, '
            f'cooldown={cooldown}, force={bool(force)}',
            flush=True,
        )
    else:
        _lm_print(
            '[MiniMaxH3 LongMedia][INT8 SOL STORAGE GUARD] TRIM: '
            f'block={block_index}, pre_free={before_free:.0f} MB, '
            f'pre_cached={before_cached:.0f} MB, force={bool(force)}',
            flush=True,
        )
    return True




class _INT8SolStorageOOM(RuntimeError):
    """Terminal INT8 Sol-storage failure; never route to external attention."""
    pass


def _execute_h3_sol_attention(attn, x, rope_freqs, transformer_options, state, tau, measure=None):
    """Execute embedded H3 Sol.

    When sol_qkv_chunk_tokens > 0, use a two-pass streamed-Q path:
      1) project small token chunks and retain only full-sequence K/V;
      2) project Q chunks again, run rectangular Sol against full K/V, and
         overwrite the dead norm1 activation with attention output.

    This trades extra projection compute for much lower peak activation memory
    and is intended for very long single-pass sequences where the full fused
    BF16 QKV tensor itself no longer fits in VRAM.
    """
    from .sol_kernel import sol_attn_sm120, prepare_kv_sm120, sol_attn_query_sm120
    import comfy.quant_ops

    s = int(x.shape[0])
    state['last_sol_tau'] = float(tau)
    inner = int(attn.heads * attn.head_dim)
    meas = measure or (lambda _name, fn: fn())
    qkv_chunk_tokens = int(state.get('sol_qkv_chunk_tokens', 0) or 0)

    span = (transformer_options or {}).get('latentlab_sol_h3_video_span')
    sink_blocks = (0, 0)
    sink_q = (0, 0)
    if state.get('sol_sink_conditioning', 'exact_kv') != 'off' and span is not None:
        video_start, _video_stop = span
        if int(video_start) > 0:
            sink_blocks = (0, (int(video_start) + 63) // 64)
            if state.get('sol_sink_conditioning') == 'exact_kv_and_rows':
                sink_q = sink_blocks

    def _weights():
        qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
        return qw, kw

    # Existing zero-copy full-QKV path for normal sequence lengths / A-B tests.
    if qkv_chunk_tokens <= 0 or s <= qkv_chunk_tokens:
        qkv = meas('sol_qkv_proj', lambda: attn.qkv_proj(x))
        q, k, v = qkv.split(inner, dim=-1)
        q = q.view(1, s, attn.heads, attn.head_dim)
        k = k.view(1, s, attn.heads, attn.head_dim)
        v = v.view(1, s, attn.heads, attn.head_dim)

        def _rope():
            if rope_freqs is not None:
                qw, kw = _weights()
                rot = int(rope_freqs.shape[-3] * 2)
                if comfy.model_management.in_training:
                    return comfy.quant_ops.ck.rms_rope_split_half(
                        q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
                    )
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
                )
                return q, k
            return attn.q_norm(q), attn.k_norm(k)

        q2, k2 = meas('sol_rms_rope', _rope)
        out = meas(
            'sol_kernel',
            lambda: sol_attn_sm120(q2, k2, v, tau=float(tau), sink_blocks=sink_blocks, sink_q=sink_q),
        )
        del q2, k2, q, k, v, qkv, _rope
    else:
        # Long-sequence path: never retain full BF16 K/V.  Token-level K/V is
        # compressed to INT8 + per-token scales as each projection chunk is
        # produced, while Sol's 64-token routing summaries remain BF16.
        from .sol_kernel import (
            allocate_compressed_kv_sm120,
            append_compressed_kv_sm120,
            finalize_compressed_kv_sm120,
            sol_attn_query_compressed_sm120,
        )
        chunk = max(64, (qkv_chunk_tokens // 64) * 64)
        chunks = (s + chunk - 1) // chunk
        state['sol_qkv_streamed_calls'] = int(state.get('sol_qkv_streamed_calls', 0)) + 1
        state['sol_qkv_max_chunks'] = max(int(state.get('sol_qkv_max_chunks', 0)), chunks)
        if not state.get('sol_qkv_announced'):
            _lm_print(
                '[MiniMaxH3 LongMedia] Low-VRAM compressed streamed QKV enabled: '
                f'{s} tokens -> {chunks} query chunks of <= {chunk}; '
                'K/V=INT8+scale, Sol summaries=BF16, Q streamed',
                flush=True,
            )
            state['sol_qkv_announced'] = True

        H, D = int(attn.heads), int(attn.head_dim)
        _active_block = state.get('active_block_index', None)
        _int8_pre_sol_storage_guard(
            state, block_index=_active_block, force=False
        )

        _backend = str(state.get('model_runtime_backend', 'unknown')).lower()
        if _backend in ('int8', 'int8-convrot-w4a4'):
            storage, _reused = _int8_reusable_sol_storage(
                state,
                tokens=s,
                heads=H,
                head_dim=D,
                device=x.device,
                allocator=allocate_compressed_kv_sm120,
            )
        else:
            storage = meas(
                'sol_stream_kv_storage_alloc',
                lambda: allocate_compressed_kv_sm120(1, s, H, D, x.device),
            )

        # V31: keep stock Comfy quant math, but prepare each quantized projection
        # ONCE per H3 block and reuse it across all streamed token chunks.
        # This follows comfy.ops.cast_bias_weight(offloadable=True) contract:
        # cast once, use many times, uncast after the final use.
        # 0.3.112: streamed-Sol must resolve the memory mode locally.
        # _ultra_streaming used to be defined only inside the MLP path, which
        # made the long-sequence attention preflight route fail with NameError
        # before the first QKV chunk. Keep the semantic identical to the MLP
        # governor: ultra_low_vram disables cached INT8 projection residency.
        _ultra_streaming = str(state.get('memory_mode', 'normal')) == 'ultra_low_vram'
        _int8_backend = (
            _v12_is_int8_family(state)
            and not comfy.model_management.in_training
            and not _ultra_streaming
        )
        if _ultra_streaming and _v12_is_int8_family(state) and not state.get('v354_ultra_mlp_stream_announced'):
            _lm_print(
                '[MiniMaxH3 LongMedia][0.3.54 ULTRA MLP STREAM] '
                'cached fc1+fc2 residency disabled; stock Comfy MLP streams linear weights sequentially per token chunk',
                flush=True,
            )
            state['v354_ultra_mlp_stream_announced'] = True
        _qkv_handle = None
        _out_handle = None
        _v19_active = _v19_selected_block(state, _active_block)
        _v19_offsets = _v19_probe_offsets(s, chunk) if _v19_active else []
        _v19_out_inputs = []
        _v19_out_cached = []
        if _int8_backend:
            _probe = x[:4]
            _stock_qkv_probe = None
            if _v12b_linear_ab_enabled(state, 'qkv_proj'):
                _stock_qkv_probe = attn.qkv_proj(_probe).detach()
            _v19_qkv_input = None
            _v19_qkv_stock = None
            if _v19_active:
                _v19_qkv_input = torch.cat([
                    x[int(offset):min(s, int(offset) + 4)]
                    for offset in _v19_offsets
                ], dim=0).detach().clone()
                _v19_qkv_stock = attn.qkv_proj(_v19_qkv_input).detach()
            _qkv_handle = _int8_prepare_block_linear(attn.qkv_proj, _probe)
            if _v19_qkv_stock is not None:
                _v19_qkv_cached = _int8_cached_linear(
                    _qkv_handle, _v19_qkv_input
                )
                _v19_report(
                    state, 'QKV-PROJ', _v19_qkv_stock, _v19_qkv_cached,
                    offsets=_v19_offsets,
                )
                del _v19_qkv_input, _v19_qkv_stock, _v19_qkv_cached
            if _stock_qkv_probe is not None:
                _cached_qkv_probe = _int8_cached_linear(
                    _qkv_handle, _probe
                )
                _v12b_linear_ab_report(
                    state, 'qkv_proj', _stock_qkv_probe, _cached_qkv_probe
                )
                del _stock_qkv_probe, _cached_qkv_probe
            if not state.get('int8_semantic_dispatch_announced'):
                _lm_print(
                    '[MiniMaxH3 LongMedia][INT8 SEMANTIC DISPATCH] '
                    'ordinary qkv/fc1/out use F.linear + QuantizedTensor dispatch; '
                    'fc2 SwiGLU mirrors comfy.ops.linear_input_act',
                    flush=True,
                )
                state['int8_semantic_dispatch_announced'] = True

        qw = kw = None
        if rope_freqs is not None:
            qw, kw = _weights()
            rot = int(rope_freqs.shape[-3] * 2)

        # V18: V13-V17 compared Sol against an exact reference reconstructed
        # from the already-compressed INT8 K/V store.  That proved the routed
        # kernel, but could not measure information lost by K/V compression.
        # For block 0 / forward 1 only, retain four tiny Q probes and stream an
        # online exact-softmax reference over the original BF16 K/V chunks
        # before each chunk is compressed.  Full BF16 K/V is never retained.
        _v18 = None
        if (
            _v12_is_int8_family(state)
            and int(state.get('active_block_index', -1)) == 0
            and int(state.get('v12_int8_sol_forward_generation', 0) or 0) == 1
            and not bool(state.get('v18_bf16_kv_reference_done', False))
        ):
            _v18_targets = sorted({
                0,
                min(chunk, max(0, s - 1)),
                (s // 2 // chunk) * chunk,
                ((s - 1) // chunk) * chunk,
            })
            _v18 = {'targets': {}, 'started': time.perf_counter()}
            for _target in _v18_targets:
                _probe_end = min(s, int(_target) + 4)
                if _qkv_handle is not None:
                    _probe_qkv = _int8_cached_linear(
                        _qkv_handle, x[int(_target):_probe_end]
                    )
                else:
                    _probe_qkv = attn.qkv_proj(x[int(_target):_probe_end])
                _probe_q, _probe_k_dead, _probe_v_dead = _probe_qkv.split(
                    inner, dim=-1
                )
                _probe_n = _probe_end - int(_target)
                _probe_q = _probe_q.view(1, _probe_n, H, D)
                _probe_k_dead = _probe_k_dead.view(1, _probe_n, H, D)
                if rope_freqs is not None:
                    _probe_rf = rope_freqs[:, int(_target):_probe_end]
                    if comfy.model_management.in_training:
                        _probe_q, _probe_k_dead = (
                            comfy.quant_ops.ck.rms_rope_split_half(
                                _probe_q, _probe_k_dead, _probe_rf, qw, kw,
                                epsilon=attn.q_norm.eps, rot_dim=rot,
                            )
                        )
                    else:
                        comfy.quant_ops.ck.rms_rope_split_half_(
                            _probe_q, _probe_k_dead, _probe_rf, qw, kw,
                            epsilon=attn.q_norm.eps, rot_dim=rot,
                        )
                else:
                    _probe_q = attn.q_norm(_probe_q)
                _probe_q32 = _probe_q.detach().to(dtype=torch.float32).clone()
                _probe_queries = int(_probe_q32.shape[1])
                _running_max = torch.full(
                    (1, H, _probe_queries), -float('inf'),
                    device=x.device, dtype=torch.float32,
                )
                _v18['targets'][int(_target)] = {
                    'q': _probe_q32,
                    'running_max': _running_max,
                    'running_sum': torch.zeros_like(_running_max),
                    'running_out': torch.zeros(
                        (1, H, _probe_queries, D),
                        device=x.device, dtype=torch.float32,
                    ),
                }
                del _probe_qkv, _probe_q, _probe_k_dead, _probe_v_dead
                del _probe_q32, _running_max
            _lm_print(
                '[MiniMaxH3 LongMedia][V18 BF16-KV REFERENCE] '
                f'prepared offsets={_v18_targets}, queries_per_offset=4; '
                'streaming exact reference before K/V compression',
                flush=True,
            )

        def _v18_accumulate_original_bf16(k_chunk, v_chunk):
            if _v18 is None:
                return
            k32 = k_chunk.detach().to(dtype=torch.float32)
            v32 = v_chunk.detach().to(dtype=torch.float32)
            attention_scale = float(D ** -0.5)
            for _item in _v18['targets'].values():
                scores = torch.einsum(
                    'bqhd,bkhd->bhqk', _item['q'], k32
                ).mul_(attention_scale)
                chunk_max = scores.amax(dim=-1)
                next_max = torch.maximum(_item['running_max'], chunk_max)
                previous_scale = torch.exp(_item['running_max'] - next_max)
                probability = torch.exp(scores - next_max.unsqueeze(-1))
                _item['running_out'] = (
                    _item['running_out'] * previous_scale.unsqueeze(-1)
                    + torch.einsum('bhqk,bkhd->bhqd', probability, v32)
                )
                _item['running_sum'] = (
                    _item['running_sum'] * previous_scale
                    + probability.sum(dim=-1)
                )
                _item['running_max'] = next_max
                del scores, chunk_max, next_max, previous_scale, probability
            del k32, v32

        def _v18_finalize_original_bf16():
            if _v18 is None:
                return
            for _item in _v18['targets'].values():
                _item['reference'] = (
                    _item['running_out']
                    / _item['running_sum'].unsqueeze(-1)
                ).permute(0, 2, 1, 3).contiguous()
                del _item['running_max'], _item['running_sum']
                del _item['running_out']
            elapsed_ms = (time.perf_counter() - _v18['started']) * 1000.0
            _lm_print(
                '[MiniMaxH3 LongMedia][V18 BF16-KV REFERENCE] '
                f'original BF16 exact references ready, elapsed_ms={elapsed_ms:.1f}',
                flush=True,
            )

        def _build_compressed_kv():
            for start in range(0, s, chunk):
                end = min(s, start + chunk)
                _split_kv = None
                if _qkv_handle is not None:
                    _split_kv = _v32_quant_linear_rows(
                        _qkv_handle, x[start:end], inner, inner * 3
                    )
                if _split_kv is not None:
                    _k, _v = _split_kv.split(inner, dim=-1)
                    qkv_part = None
                else:
                    if _qkv_handle is not None:
                        qkv_part = _int8_cached_linear(_qkv_handle, x[start:end])
                    else:
                        qkv_part = attn.qkv_proj(x[start:end])
                    _q_dead, _k, _v = qkv_part.split(inner, dim=-1)
                n = end - start
                _k = _k.view(1, n, H, D)
                _v = _v.view(1, n, H, D)
                if rope_freqs is not None:
                    rf = rope_freqs[:, start:end]
                    # The public paired RMS+RoPE op is the only H3 path exposing
                    # rot_dim. Reuse K as the discarded Q operand so K remains
                    # bit-faithful while avoiding the expensive Q projection.
                    _q_dummy = _k.clone()
                    if comfy.model_management.in_training:
                        _q_dummy, _k = comfy.quant_ops.ck.rms_rope_split_half(
                            _q_dummy, _k, rf, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
                        )
                    else:
                        comfy.quant_ops.ck.rms_rope_split_half_(
                            _q_dummy, _k, rf, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
                        )
                    del _q_dummy
                else:
                    _k = attn.k_norm(_k)
                _v18_accumulate_original_bf16(_k, _v)
                append_compressed_kv_sm120(storage, _k, _v, start)
                if qkv_part is not None:
                    del qkv_part
                if '_q_dead' in locals():
                    del _q_dead
                del _split_kv, _k, _v
            return storage

        # V40 production baseline: diagnostic CUDA-sync profiling is disabled.
        # Execution math, chunking, SOL routing and quantized kernels are unchanged from V39.
        _v33_profile = False
        _v35_forensic = False
        _v35_chunks = []
        if _v35_forensic and not state.get('v35_forensic_announced'):
            _lm_print(
                '[MiniMaxH3 LongMedia][V35 FORENSIC] enabled for blocks 0..2; '
                'collecting AUTO decision, tau, per-query-chunk exact/approx routing, '
                'thresholds, threshold-kernel time, SOL-forward time, Q/out/MLP and VRAM',
                flush=True,
            )
            state['v35_forensic_announced'] = True
        if _v35_forensic:
            _lm_print(
                '[MiniMaxH3 LongMedia][V35 POLICY] '
                f'block={int(_active_block)} requested={state.get("requested_attention_mode")} '
                f'effective={state.get("sol_mode")} tokens={s} tau={float(tau):.4f} '
                f'tau_start={float(state.get("sol_tau_start", 0.0)):.4f} '
                f'tau_end={float(state.get("sol_tau_end", 0.0)):.4f} '
                f'curve={state.get("sol_curve")} sink_blocks={sink_blocks} sink_q={sink_q} '
                f'qkv_chunk={chunk} query_chunks={chunks} kv_blocks={int(storage.get("blocks", 0))}',
                flush=True,
            )
        if _v33_profile:
            torch.cuda.synchronize()
            _v33_kv_t0 = time.perf_counter()
        meas('sol_stream_kv_projection_compress', _build_compressed_kv)
        meas('sol_stream_kv_summaries', lambda: finalize_compressed_kv_sm120(storage))
        if _v33_profile:
            torch.cuda.synchronize()
            state['v33_last_kvpass_s'] = time.perf_counter() - _v33_kv_t0
        _v18_finalize_original_bf16()

        def _v18_report(offset, q, current):
            if _v18 is None or int(offset) not in _v18['targets']:
                return
            item = _v18['targets'][int(offset)]
            query_count = int(item['reference'].shape[1])
            q_replay = q[:, :query_count].detach().to(dtype=torch.float32)
            q_reference = item['q'][:, :query_count]
            q_delta = q_replay - q_reference
            q_ref_rms = float(
                torch.sqrt(torch.mean(q_reference.square())).item()
            )
            q_rel_rms = float(
                torch.sqrt(torch.mean(q_delta.square())).item()
            ) / max(1.0e-12, q_ref_rms)

            compressed_exact = _v13_exact_attention_from_compressed(
                q[:, :query_count], storage, key_chunk=1024
            )
            from .sol_kernel import sol_attn_query_compressed_sm120
            probe_rows = min(64, int(q.shape[1]))
            torch.cuda.synchronize(q.device)
            started = time.perf_counter()
            tau3 = sol_attn_query_compressed_sm120(
                q[:, :probe_rows], storage, q_offset=int(offset), tau=-3.0,
                sink_blocks=sink_blocks, sink_q=sink_q,
            )
            torch.cuda.synchronize(q.device)
            tau3_ms = (time.perf_counter() - started) * 1000.0

            reference = item['reference'].detach().to(
                device='cpu', dtype=torch.float32
            ).flatten()

            def _report(label, candidate, elapsed_ms=None):
                candidate_flat = (
                    candidate[:, :query_count]
                    .detach().to(device='cpu', dtype=torch.float32).flatten()
                )
                diff = candidate_flat - reference
                eps = 1.0e-12
                rms_ref = float(torch.sqrt(torch.mean(reference.square())).item())
                rms_diff = float(torch.sqrt(torch.mean(diff.square())).item())
                denom = float(
                    torch.linalg.vector_norm(reference).item()
                    * torch.linalg.vector_norm(candidate_flat).item()
                )
                cosine = float(
                    torch.dot(reference, candidate_flat).item()
                    / max(eps, denom)
                )
                timing = (
                    '' if elapsed_ms is None
                    else f', elapsed_ms={float(elapsed_ms):.1f}'
                )
                _lm_print(
                    '[MiniMaxH3 LongMedia][V18 BF16-KV REFERENCE] '
                    f'offset={int(offset)}, label={label}{timing}, '
                    f'rel_rms={rms_diff / max(eps, rms_ref):.8e}, '
                    f'cosine={cosine:.10f}, '
                    f'max_abs={float(diff.abs().max().item()):.8e}, '
                    f'q_replay_rel_rms={q_rel_rms:.8e}, '
                    f'finite={bool(torch.isfinite(diff).all().item())}',
                    flush=True,
                )

            _report('COMPRESSED-EXACT', compressed_exact)
            _report('CURRENT-TAU-2', current)
            _report('SOL-TAU-3', tau3, tau3_ms)
            del compressed_exact, tau3, reference, q_replay, q_reference, q_delta
            del item['q'], item['reference']

        # Important geometry fix: attention inner width is 7168 on H3 while
        # hidden width is 5376.  Never write raw attention output into x.
        # Instead each streamed Q chunk immediately runs out_proj and only the
        # projected [tokens, hidden] result overwrites the dead norm1 slice.
        def _stream_queries_and_project():
            nonlocal _out_handle, tau
            _v34_qproj_s = 0.0
            _v34_rope_s = 0.0
            _v34_sol_s = 0.0
            _v34_outproj_s = 0.0
            _v34_copy_s = 0.0
            for start in range(0, s, chunk):
                end = min(s, start + chunk)
                _split_q = None
                if _v33_profile:
                    torch.cuda.synchronize()
                    _v34_t0 = time.perf_counter()
                if _qkv_handle is not None:
                    _split_q = _v32_quant_linear_rows(
                        _qkv_handle, x[start:end], 0, inner
                    )
                if _split_q is not None:
                    _q = _split_q
                    qkv_part = None
                else:
                    if _qkv_handle is not None:
                        qkv_part = _int8_cached_linear(_qkv_handle, x[start:end])
                    else:
                        qkv_part = attn.qkv_proj(x[start:end])
                    _q, _k_dead, _v_dead = qkv_part.split(inner, dim=-1)
                if _v33_profile:
                    torch.cuda.synchronize()
                    _v34_qproj_s += time.perf_counter() - _v34_t0
                    _v34_t0 = time.perf_counter()
                n = end - start
                _q = _q.view(1, n, H, D)
                if rope_freqs is not None:
                    rf = rope_freqs[:, start:end]
                    # Keep exact H3 q RMS+RoPE semantics; paired public op carries
                    # rot_dim, while the K result is intentionally discarded.
                    _k_dummy = _q.clone()
                    if comfy.model_management.in_training:
                        _q, _k_dummy = comfy.quant_ops.ck.rms_rope_split_half(
                            _q, _k_dummy, rf, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
                        )
                    else:
                        comfy.quant_ops.ck.rms_rope_split_half_(
                            _q, _k_dummy, rf, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
                        )
                    del _k_dummy
                else:
                    _q = attn.q_norm(_q)
                if _v33_profile:
                    torch.cuda.synchronize()
                    _v34_rope_s += time.perf_counter() - _v34_t0
                    _v34_t0 = time.perf_counter()
                if start == 0:
                    _v37_tau = _v37_auto_calibrate_sol_tau(
                        state, _q, storage, sink_blocks, sink_q, start
                    )
                    if _v37_tau is not None:
                        tau = float(_v37_tau)
                        state['last_sol_tau'] = float(tau)
                _v35_ret = sol_attn_query_compressed_sm120(
                    _q, storage, q_offset=start, tau=float(tau),
                    sink_blocks=sink_blocks, sink_q=sink_q,
                    telemetry=_v35_forensic,
                )
                if _v35_forensic:
                    q_out, _v35_stat = _v35_ret
                    _v35_stat['start'] = int(start)
                    _v35_stat['end'] = int(end)
                    _v35_chunks.append(_v35_stat)
                    _lm_print(
                        '[MiniMaxH3 LongMedia][V35 SOL CHUNK] '
                        f'block={int(_active_block)} q={int(start)}:{int(end)} '
                        f'tau={float(tau):.3f} exact={_v35_stat["exact_ratio"]*100.0:.2f}% '
                        f'exact_range={_v35_stat["exact_ratio_min"]*100.0:.1f}-'
                        f'{_v35_stat["exact_ratio_max"]*100.0:.1f}% '
                        f'score_routes={_v35_stat["score_routed"]} '
                        f'local_forced={_v35_stat["local_forced"]} '
                        f'sink_forced={_v35_stat["sink_forced"]} '
                        f'q_sink_programs={_v35_stat["q_sink_programs"]} '
                        f'thr={_v35_stat["threshold_min"]:.3f}/'
                        f'{_v35_stat["threshold_mean"]:.3f}/'
                        f'{_v35_stat["threshold_max"]:.3f} '
                        f't_threshold={_v35_stat["threshold_s"]:.4f}s '
                        f't_forward={_v35_stat["forward_s"]:.4f}s',
                        flush=True,
                    )
                else:
                    q_out = _v35_ret
                if _v33_profile:
                    torch.cuda.synchronize()
                    _v34_sol_s += time.perf_counter() - _v34_t0
                    _v34_t0 = time.perf_counter()
                _v18_report(start, _q, q_out)
                if _int8_backend and _out_handle is None:
                    _out_probe = q_out.view(n, inner)[:4]
                    _stock_out_probe = None
                    if _v12b_linear_ab_enabled(state, 'out_proj'):
                        _stock_out_probe = attn.out_proj(_out_probe).detach()
                    _out_handle = _int8_prepare_block_linear(
                        attn.out_proj, _out_probe
                    )
                    if _stock_out_probe is not None:
                        _cached_out_probe = _int8_cached_linear(
                            _out_handle, _out_probe
                        )
                        _v12b_linear_ab_report(
                            state, 'out_proj',
                            _stock_out_probe, _cached_out_probe,
                        )
                        del _stock_out_probe, _cached_out_probe
                if _out_handle is not None:
                    projected = _int8_cached_linear(
                        _out_handle, q_out.view(n, inner)
                    )
                else:
                    projected = attn.out_proj(q_out.view(n, inner))
                if _v33_profile:
                    torch.cuda.synchronize()
                    _v34_outproj_s += time.perf_counter() - _v34_t0
                    _v34_t0 = time.perf_counter()
                if _v19_active and int(start) in _v19_offsets:
                    _v19_rows = min(4, n)
                    _v19_out_inputs.append(
                        q_out.view(n, inner)[:_v19_rows].detach().clone()
                    )
                    _v19_out_cached.append(
                        projected[:_v19_rows].detach().clone()
                    )
                x[start:end].copy_(projected)
                if _v33_profile:
                    torch.cuda.synchronize()
                    _v34_copy_s += time.perf_counter() - _v34_t0
                if qkv_part is not None:
                    del qkv_part
                if '_k_dead' in locals():
                    del _k_dead
                if '_v_dead' in locals():
                    del _v_dead
                del _split_q, _q, q_out, projected
            if _v35_forensic and _v35_chunks:
                _ex = [z['exact_ratio'] for z in _v35_chunks]
                _tf = [z['forward_s'] for z in _v35_chunks]
                _tt = [z['threshold_s'] for z in _v35_chunks]
                _exact = sum(z['exact'] for z in _v35_chunks)
                _approx = sum(z['approx'] for z in _v35_chunks)
                _den = max(1, _exact + _approx)
                _alloc = torch.cuda.memory_allocated(x.device) / (1024.0 ** 2)
                _reserved = torch.cuda.memory_reserved(x.device) / (1024.0 ** 2)
                _free, _total = torch.cuda.mem_get_info(x.device)
                _lm_print(
                    '[MiniMaxH3 LongMedia][V35 SOL SUMMARY] '
                    f'block={int(_active_block)} tau={float(tau):.3f} chunks={len(_v35_chunks)} '
                    f'exact_global={100.0*_exact/_den:.2f}% '
                    f'exact_chunk_min/avg/max={100.0*min(_ex):.2f}/'
                    f'{100.0*sum(_ex)/len(_ex):.2f}/{100.0*max(_ex):.2f}% '
                    f'forward_chunk_min/avg/max={min(_tf):.4f}/'
                    f'{sum(_tf)/len(_tf):.4f}/{max(_tf):.4f}s '
                    f'threshold_total={sum(_tt):.4f}s forward_total={sum(_tf):.4f}s '
                    f'alloc={_alloc:.0f}MB reserved={_reserved:.0f}MB driver_free={_free/(1024.0**2):.0f}MB',
                    flush=True,
                )
                state['v35_last_sol_summary'] = {
                    'block': int(_active_block), 'tau': float(tau),
                    'exact_global': float(_exact) / float(_den),
                    'threshold_total_s': float(sum(_tt)),
                    'forward_total_s': float(sum(_tf)),
                }
            if _v33_profile:
                state['v34_last_qproj_s'] = _v34_qproj_s
                state['v34_last_rope_s'] = _v34_rope_s
                state['v34_last_sol_s'] = _v34_sol_s
                state['v34_last_outproj_s'] = _v34_outproj_s
                state['v34_last_copy_s'] = _v34_copy_s
            return x

        try:
            if _v33_profile:
                torch.cuda.synchronize()
                _v33_query_t0 = time.perf_counter()
            result = meas('sol_stream_query_kernel_outproj', _stream_queries_and_project)
            if _v33_profile:
                torch.cuda.synchronize()
                state['v33_last_querypass_s'] = time.perf_counter() - _v33_query_t0
        finally:
            if _out_handle is not None:
                _int8_release_block_linear(_out_handle)
            if _qkv_handle is not None:
                _int8_release_block_linear(_qkv_handle)
            if _v18 is not None:
                _v18['targets'].clear()
                state['v18_bf16_kv_reference_done'] = True
        if _v19_active and _v19_out_inputs:
            _v19_out_input = torch.cat(_v19_out_inputs, dim=0)
            _v19_out_stock = attn.out_proj(_v19_out_input).detach()
            _v19_out_got = torch.cat(_v19_out_cached, dim=0)
            _v19_report(
                state, 'OUT-PROJ', _v19_out_stock, _v19_out_got,
                offsets=_v19_offsets,
            )
            del _v19_out_input, _v19_out_stock, _v19_out_got
            _v19_out_inputs.clear()
            _v19_out_cached.clear()
        if _backend not in ('int8', 'int8-convrot-w4a4'):
            del storage
        del qw, kw
        return result, sink_blocks

    def _out_proj():
        flat = out.view(s, inner)
        chunk_tokens = int(state.get('sol_out_proj_chunk_tokens', 24576))
        if chunk_tokens <= 0 or s <= chunk_tokens:
            return attn.out_proj(flat)

        chunks = (s + chunk_tokens - 1) // chunk_tokens
        state['sol_out_proj_chunked_calls'] = int(state.get('sol_out_proj_chunked_calls', 0)) + 1
        state['sol_out_proj_max_chunks'] = max(int(state.get('sol_out_proj_max_chunks', 0)), chunks)
        if not state.get('sol_out_proj_announced'):
            _lm_print(
                '[MiniMaxH3 LongMedia] Low-VRAM Sol out_proj enabled: '
                f'{s} tokens -> {chunks} chunks of <= {chunk_tokens}',
                flush=True,
            )
            state['sol_out_proj_announced'] = True

        # In streamed-QKV mode `out` reuses the dead norm1 activation `x`.
        # Token-wise out_proj can therefore overwrite each already-consumed
        # input slice in place, avoiding another full [S, hidden] allocation.
        reuse_input = out.data_ptr() == x.data_ptr()
        projected = flat if reuse_input else torch.empty_like(x)
        for start in range(0, s, chunk_tokens):
            end = min(s, start + chunk_tokens)
            part = attn.out_proj(flat[start:end])
            projected[start:end].copy_(part)
            del part
        return projected

    result = meas('sol_out_proj', _out_proj)
    return result, sink_blocks

def _sol_exception_is_oom(exc):
    msg = f"{type(exc).__name__}: {exc}".lower()
    return (
        'outofmemoryerror' in msg
        or 'out of memory' in msg
        or 'allocation on device' in msg
        or 'cuda oom' in msg
        or 'would exceed allowed memory' in msg
    )


def _sol_retry_chunk_schedule(current_chunk):
    cur = int(current_chunk or 0)
    if cur <= 0:
        return []
    ladder = []
    for candidate in (cur // 2, 4096, 2048, 1024):
        candidate = max(64, (int(candidate) // 64) * 64)
        if candidate > 0 and candidate < cur and candidate not in ladder:
            ladder.append(candidate)
    return ladder


def _run_h3_sol_attention(attn, x, rope_freqs, transformer_options, state, measure=None):
    """Embedded H3 Sol path with adaptive low-VRAM retries.

    OOM inside Sol should never fall through to the generic attention fallback,
    because that path can trigger catastrophic NVFP4 dequantization on large H3
    sequences. Instead we trim cache and retry the same embedded Sol path with a
    smaller streamed QKV chunk size.
    """
    token_count = int(x.shape[0])
    min_tokens = int(state.get('sol_min_tokens', 4096))
    if token_count < min_tokens:
        return attn(x, rope_freqs=rope_freqs, transformer_options=transformer_options)
    tau = _sol_schedule_tau(transformer_options, state)
    if tau is None:
        return attn(x, rope_freqs=rope_freqs, transformer_options=transformer_options)

    try:
        result, sink_blocks = _execute_h3_sol_attention(
            attn, x, rope_freqs, transformer_options, state, tau, measure=measure
        )
    except Exception as exc:
        if isinstance(exc, _INT8SolStorageOOM):
            _lm_print(
                '[MiniMaxH3 LongMedia][INT8 SOL STORAGE] terminal failure; '
                'external attention fallback is disabled for this large INT8 sequence',
                flush=True,
            )
            raise

        if _sol_exception_is_oom(exc):
            original_chunk = int(state.get('sol_qkv_chunk_tokens', 0) or 0)
            original_out_proj = int(state.get('sol_out_proj_chunk_tokens', 0) or 0)
            seen = state.setdefault('sol_oom_reasons', [])
            reason = f'{type(exc).__name__}: {exc}'
            if reason not in seen:
                seen.append(reason)
                _lm_print('[MiniMaxH3 LongMedia] Embedded Sol-Attn OOM: ' + reason, flush=True)
            retry_chunks = _sol_retry_chunk_schedule(original_chunk)
            last_exc = exc
            for retry_idx, retry_chunk in enumerate(retry_chunks, start=1):
                try:
                    state['sol_qkv_chunk_tokens'] = int(retry_chunk)
                    if original_out_proj > 0:
                        state['sol_out_proj_chunk_tokens'] = min(original_out_proj, max(retry_chunk * 3, 1024))
                    _lm_print(
                        '[MiniMaxH3 LongMedia] Embedded Sol-Attn retry '
                        f'#{retry_idx}: qkv chunk {original_chunk} -> {retry_chunk}, '
                        f'out_proj <= {int(state.get("sol_out_proj_chunk_tokens", 0) or 0)}',
                        flush=True,
                    )
                    gc.collect()
                    try:
                        import comfy.model_management as _mm
                        _mm.soft_empty_cache()
                    except Exception:
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    result, sink_blocks = _execute_h3_sol_attention(
                        attn, x, rope_freqs, transformer_options, state, tau, measure=measure
                    )
                    if retry_chunk != original_chunk:
                        _lm_print(
                            '[MiniMaxH3 LongMedia] Embedded Sol-Attn retry succeeded: '
                            f'using persistent qkv chunk {retry_chunk}',
                            flush=True,
                        )
                    break
                except Exception as retry_exc:
                    last_exc = retry_exc
                    if not _sol_exception_is_oom(retry_exc):
                        raise
                    _lm_print(
                        '[MiniMaxH3 LongMedia] Embedded Sol-Attn retry failed: '
                        f'qkv chunk {retry_chunk}: {type(retry_exc).__name__}: {retry_exc}',
                        flush=True,
                    )
                    gc.collect()
                    try:
                        import comfy.model_management as _mm
                        _mm.soft_empty_cache()
                    except Exception:
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
            else:
                state['sol_qkv_chunk_tokens'] = int(original_chunk)
                state['sol_out_proj_chunk_tokens'] = int(original_out_proj)
                raise last_exc
        else:
            _backend = str(state.get('model_runtime_backend', 'unknown')).lower()
            if _backend in ('int8', 'int8-convrot-w4a4') and token_count >= min_tokens:
                _lm_print(
                    '[MiniMaxH3 LongMedia][INT8 SOL] non-OOM Sol failure; '
                    'external Sage fallback disabled to avoid full-sequence INT8 QKV allocation: '
                    f'{type(exc).__name__}: {exc}',
                    flush=True,
                )
                raise

            reason = f'{type(exc).__name__}: {exc}'
            seen = state.setdefault('sol_fallbacks', [])
            if reason not in seen:
                seen.append(reason)
                _lm_print('[MiniMaxH3 LongMedia] Embedded Sol-Attn fallback: ' + reason, flush=True)

            try:
                del exc
            except Exception:
                pass
            gc.collect()
            try:
                import comfy.model_management as _mm
                _mm.soft_empty_cache()
            except Exception:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            return attn(x, rope_freqs=rope_freqs, transformer_options=transformer_options)

    state['sol_calls'] = int(state.get('sol_calls', 0)) + 1
    if not state.get('sol_announced'):
        _lm_print(
            '[MiniMaxH3 LongMedia] Embedded Sol-Attn active: '
            f'{token_count} tokens, tau={float(tau):.3f}, sink={sink_blocks}',
            flush=True,
        )
        state['sol_announced'] = True
    return result


class _H3MLPChunkPatch:
    """Exact low-VRAM H3 block replacement that chunks only the token-wise MLP.

    Attention is left untouched. The FFN/MLP is point-wise over the packed token
    axis, so splitting dim 0 and writing each result into a preallocated output
    tensor preserves the full-block result while avoiding a full-sequence
    expanded-hidden activation. Block 0 of the first forward also emits deep
    stage memory telemetry for A/B testing.
    """

    def __init__(self, index, state, chunk_tokens=8192):
        self.index = int(index)
        self.state = state
        self.chunk_tokens = max(256, int(chunk_tokens))

    @staticmethod
    def _extract_block(original_block):
        return _H3BlockMemoryTracePatch._extract_block(original_block)

    @staticmethod
    def _mod_scale_shift(h, shift, scale, segments):
        return _H3BlockMemoryTracePatch._mod_scale_shift(h, shift, scale, segments)

    @staticmethod
    def _mod_gate(x, gate, other, segments):
        return _H3BlockMemoryTracePatch._mod_gate(x, gate, other, segments)

    def _inter_block_pressure_guard(self):
        """TEST FIXED: effective-headroom inter-block guard.

        ACTIVE implementation for _H3MLPChunkPatch.
        Uses driver-free VRAM + reclaimable PyTorch cache instead of free VRAM
        alone, avoiding unnecessary DynamicVRAM/AIMDO cache destruction.
        """
        state = self.state

        # Guaranteed one-shot AUTO calibration after completed block 0.
        if self.index == 0 and not state.get('auto_vram_controller_done'):
            try:
                token_count = int(
                    state.get('current_token_count', 0)
                    or state.get('last_token_count', 0)
                    or 0
                )
                self._auto_vram_controller_after_probe(token_count)
            except Exception as exc:
                state['auto_vram_controller_done'] = True
                state['auto_vram_controller_mode'] = 'SAFE'
                _lm_print(
                    f"[MiniMaxH3 LongMedia][AUTO VRAM] calibration failed; "
                    f"SAFE baseline retained: {exc!r}",
                    flush=True,
                )

        if not torch.cuda.is_available():
            return

        guard_mb = float(int(state.get('inter_block_vram_guard_mb', 0) or 0))
        emergency_mb = float(int(state.get('inter_block_guard_emergency_mb', 0) or 0))
        cooldown_blocks = max(
            0, int(state.get('inter_block_guard_cooldown_blocks', 0) or 0)
        )
        emergency_cooldown_blocks = max(
            0, int(state.get('inter_block_guard_emergency_cooldown_blocks', 0) or 0)
        )

        if guard_mb <= 0 and emergency_mb <= 0:
            return

        guard_call = int(state.get('inter_block_guard_calls', 0)) + 1
        state['inter_block_guard_calls'] = guard_call

        snap = _cuda_memory_snapshot()
        if not snap:
            return

        mb = 1024.0 ** 2
        free_mb = float(snap['driver_free']) / mb
        cached_mb = float(snap['cached']) / mb
        effective_mb = free_mb + cached_mb

        normal_hyst = float(
            state.get('inter_block_guard_hysteresis_mb', 1024.0) or 1024.0
        )
        emergency_hyst = float(
            state.get('inter_block_emergency_hysteresis_mb', 512.0) or 512.0
        )
        min_reclaim_mb = float(
            state.get('inter_block_min_reclaim_mb', 256.0) or 256.0
        )

        normal_trigger = guard_mb > 0 and effective_mb < guard_mb
        emergency_trigger = emergency_mb > 0 and effective_mb < emergency_mb

        # Healthy effective headroom: preserve allocator cache.
        if not normal_trigger and not emergency_trigger:
            state['inter_block_effective_skip_count'] = int(
                state.get('inter_block_effective_skip_count', 0) or 0
            ) + 1
            skips = int(state['inter_block_effective_skip_count'])
            if skips <= 3 or skips % 25 == 0:
                _lm_print(
                    '[MiniMaxH3 LongMedia][VRAM GUARD] skip: '
                    f'block {self.index}, free={free_mb:.0f} + cached={cached_mb:.0f} '
                    f'= effective={effective_mb:.0f} MB, '
                    f'guard={guard_mb:.0f}, emergency={emergency_mb:.0f}',
                    flush=True,
                )
            return

        # Threshold hysteresis.
        if normal_trigger and effective_mb >= max(0.0, guard_mb - normal_hyst):
            state['inter_block_hysteresis_skip_count'] = int(
                state.get('inter_block_hysteresis_skip_count', 0) or 0
            ) + 1
            return

        if emergency_trigger and effective_mb >= max(
            0.0, emergency_mb - emergency_hyst
        ):
            state['inter_block_emergency_hyst_skip_count'] = int(
                state.get('inter_block_emergency_hyst_skip_count', 0) or 0
            ) + 1
            return

        # Re-use the existing monotonically increasing call counter for cooldown.
        last_trim_call = int(
            state.get('inter_block_last_trim_call', -1000000000)
        )
        blocks_since_trim = guard_call - last_trim_call - 1

        if emergency_trigger:
            last_emergency = int(
                state.get('inter_block_last_emergency_trim_call', -1000000000)
            )
            since_emergency = guard_call - last_emergency - 1
            hard_emergency = effective_mb < max(0.0, emergency_mb - 1024.0)
            if (
                emergency_cooldown_blocks > 0
                and since_emergency < emergency_cooldown_blocks
                and not hard_emergency
            ):
                state['inter_block_emergency_cooldown_skip_count'] = int(
                    state.get('inter_block_emergency_cooldown_skip_count', 0) or 0
                ) + 1
                return
        else:
            hard_normal = effective_mb < max(0.0, guard_mb - 1536.0)
            if (
                cooldown_blocks > 0
                and blocks_since_trim < cooldown_blocks
                and not hard_normal
            ):
                state['inter_block_cooldown_skip_count'] = int(
                    state.get('inter_block_cooldown_skip_count', 0) or 0
                ) + 1
                return

        # No useful cache to reclaim: don't force a pointless sync/trim.
        if cached_mb < min_reclaim_mb:
            state['inter_block_low_cache_skip_count'] = int(
                state.get('inter_block_low_cache_skip_count', 0) or 0
            ) + 1
            return

        before = snap
        try:
            _soft_empty_cuda_cache()
        except Exception as exc:
            _lm_print(
                f"[MiniMaxH3 LongMedia][VRAM GUARD] cleanup failed at "
                f"block {self.index}: {exc!r}",
                flush=True,
            )
            return

        after = _cuda_memory_snapshot()
        state['inter_block_last_trim_call'] = guard_call
        state['inter_block_trim_count'] = int(
            state.get('inter_block_trim_count', 0) or 0
        ) + 1

        if emergency_trigger:
            state['inter_block_last_emergency_trim_call'] = guard_call
            state['inter_block_emergency_trim_count'] = int(
                state.get('inter_block_emergency_trim_count', 0) or 0
            ) + 1
            label = 'EMERGENCY TRIM'
        else:
            state['inter_block_normal_trim_count'] = int(
                state.get('inter_block_normal_trim_count', 0) or 0
            ) + 1
            label = 'NORMAL TRIM'

        if after:
            free_after = float(after['driver_free']) / mb
            cached_after = float(after['cached']) / mb
            _lm_print(
                f'[MiniMaxH3 LongMedia][VRAM GUARD] {label}: '
                f'block {self.index}, effective={effective_mb:.0f} MB, '
                f'free {free_mb:.0f}->{free_after:.0f} MB, '
                f'cached {cached_mb:.0f}->{cached_after:.0f} MB',
                flush=True,
            )

    def _adaptive_memory_governor(self, phase='block_start'):
        """v0.3.60 adaptive residency governor shared by every memory mode.

        Memory modes no longer mean fixed chunk presets.  They select a safety
        envelope; runtime policy is then derived from real driver-free VRAM,
        observed packed-token count, model/VRAM oversubscription and host-RAM
        pressure.  The governor may change activation chunk size and barrier
        aggressiveness, but never swaps in custom transformer math.
        """
        state = self.state
        if not bool(state.get('adaptive_memory_governor_enabled', True)):
            return
        if not torch.cuda.is_available():
            return
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            free_mb = float(free_b) / (1024.0 ** 2)
            total_mb = float(total_b) / (1024.0 ** 2)
        except Exception as exc:
            if not state.get('adaptive_memory_probe_error_announced'):
                _lm_print('[MiniMaxH3 LongMedia][0.3.60 GOVERNOR V3] memory probe failed: '
                          f'{type(exc).__name__}: {exc}', flush=True)
                state['adaptive_memory_probe_error_announced'] = True
            return

        mode = str(state.get('memory_policy_mode', state.get('memory_mode', 'normal')))
        tokens = int(state.get('current_token_count', state.get('last_token_count', 0)) or 0)
        model_b = int(state.get('model_size_bytes', 0) or 0)
        gpu_b = int(state.get('gpu_size_bytes', 0) or 0)
        ratio = (float(model_b) / float(gpu_b)) if model_b and gpu_b else 0.0

        # v0.3.110 Governor V4: sequence geometry is part of the safety envelope.
        # V3 could call a 137k-token forward FAST_PLUS because mem_get_info() was
        # sampled before demand-loaded attention weights/QKV workspaces existed.
        # Large packed sequences therefore promote the *safety* mode even when
        # the model-size probe under-reports an AIMDO/dynamic checkpoint.
        geometry_mode = mode
        if total_mb <= 18.5 * 1024.0:
            if tokens >= 150000:
                geometry_mode = 'ultra_low_vram'
            elif tokens >= 90000 and mode == 'normal':
                geometry_mode = 'low_vram'
        elif total_mb <= 26.0 * 1024.0 and tokens >= 180000 and mode == 'normal':
            geometry_mode = 'low_vram'
        if geometry_mode != mode:
            if not state.get('v110_geometry_mode_announced'):
                _lm_print(
                    '[MiniMaxH3 LongMedia][0.3.110 GOVERNOR V4] '
                    f'geometry safety envelope {mode}->{geometry_mode}; '
                    f'tokens={tokens} VRAM={total_mb/1024.0:.1f}GB',
                    flush=True,
                )
                state['v110_geometry_mode_announced'] = True
            mode = geometry_mode

        # Token-dependent activation margin. V3 capped this too aggressively and
        # only approximated generic activations. V4 keeps a larger dynamic floor;
        # attention itself has a separate geometry preflight before QKV allocation.
        token_margin_mb = min(4096.0, max(512.0, float(tokens) * 0.0120)) if tokens else 1024.0
        base_floor = {
            'normal': 640.0,
            'low_vram': 1152.0,
            'ultra_low_vram': 1792.0,
        }.get(mode, 1536.0)
        if ratio >= 1.5:
            base_floor += 384.0
        if ratio >= 2.0:
            base_floor += 256.0
        hard_floor = min(total_mb * 0.34, base_floor + token_margin_mb)
        soft_floor = min(total_mb * 0.42, hard_floor + (768.0 if mode == 'normal' else 1024.0))

        # Chunking is activation-only and executes stock block.mlp() for every
        # chunk.  Larger chunks reduce repeated weight faults when real headroom
        # exists; demotion is immediate, promotion is one rung per block.
        ladders = {
            'normal': (2048, 4096, 8192, 16384, 32768),
            'low_vram': (1024, 2048, 4096, 8192, 16384),
            'ultra_low_vram': (256, 512, 1024, 2048, 4096),
        }
        ladder = ladders.get(mode, ladders['low_vram'])
        current = int(state.get('chunk_tokens', ladder[0]) or ladder[0])
        if free_mb <= hard_floor:
            target_idx = 0
            zone = 'HARD_SAFE'
            barrier = True
        elif free_mb <= soft_floor:
            target_idx = min(1, len(ladder)-1)
            zone = 'CAUTION'
            barrier = True
        elif free_mb <= soft_floor + 1536.0:
            target_idx = min(2, len(ladder)-1)
            zone = 'BALANCED'
            barrier = (mode == 'ultra_low_vram')
        elif free_mb <= soft_floor + 3072.0:
            target_idx = min(3, len(ladder)-1)
            zone = 'FAST'
            barrier = False
        else:
            target_idx = len(ladder)-1
            zone = 'FAST_PLUS'
            barrier = False

        # V4: driver-free VRAM before demand loading is not enough evidence for a
        # FAST zone on a huge sequence. Cap the performance rung by geometry.
        if total_mb <= 18.5 * 1024.0 and tokens >= 120000:
            target_idx = min(target_idx, 1)
            zone = 'CAUTION_GEOMETRY'
            barrier = True
        elif total_mb <= 18.5 * 1024.0 and tokens >= 90000:
            target_idx = min(target_idx, 2)
            zone = 'BALANCED_GEOMETRY'
            barrier = True

        ci = 0
        for i, v in enumerate(ladder):
            if v <= current:
                ci = i
        if target_idx > ci + 1:
            target_idx = ci + 1
        selected = int(ladder[target_idx])
        old_zone = str(state.get('adaptive_memory_zone', 'CALIBRATION_SAFE'))
        old_chunk = current
        old_barrier = bool(state.get('ultra_stage_barrier_required', True))
        state['chunk_tokens'] = selected
        state['ultra_stage_barrier_required'] = bool(barrier)
        state['adaptive_memory_zone'] = zone
        state['adaptive_memory_last_free_mb'] = round(free_mb, 1)
        state['adaptive_memory_total_mb'] = round(total_mb, 1)
        state['adaptive_memory_hard_floor_mb'] = round(hard_floor, 1)
        state['adaptive_memory_soft_floor_mb'] = round(soft_floor, 1)
        state['adaptive_memory_adjustments'] = int(state.get('adaptive_memory_adjustments', 0)) + 1
        if (old_zone, old_chunk, old_barrier) != (zone, selected, barrier):
            _lm_print(
                '[MiniMaxH3 LongMedia][0.3.60 GOVERNOR V3] '
                f'mode={mode} block={self.index:02d} free={free_mb:.0f}MB '
                f'hard={hard_floor:.0f}MB soft={soft_floor:.0f}MB tokens={tokens} '
                f'zone={zone}; MLP {old_chunk}->{selected}; barrier={barrier}',
                flush=True,
            )

    def _auto_vram_controller_after_probe(self, token_count):
        """TEST controller: tune runtime memory knobs once after block 0.

        User values are treated as the SAFE baseline.  The controller only
        becomes more aggressive when block 0 completed successfully and the
        post-block memory snapshot shows meaningful recoverable headroom.
        """
        state = self.state
        if self.index != 0 or state.get('auto_vram_controller_done'):
            return
        state['auto_vram_controller_done'] = True

        if not torch.cuda.is_available():
            state['auto_vram_controller_mode'] = 'SAFE'
            state['auto_vram_controller_reason'] = 'CUDA unavailable'
            return

        snap = _cuda_memory_snapshot()
        if not snap:
            state['auto_vram_controller_mode'] = 'SAFE'
            state['auto_vram_controller_reason'] = 'memory snapshot unavailable'
            return

        mb = 1024.0 ** 2
        total_mb = float(snap['total']) / mb
        free_mb = float(snap['driver_free']) / mb
        cached_mb = float(snap['cached']) / mb
        recoverable_mb = free_mb + cached_mb
        headroom_ratio = recoverable_mb / max(1.0, total_mb)

        before = {
            'mlp': int(state.get('chunk_tokens', self.chunk_tokens) or self.chunk_tokens),
            'qkv': int(state.get('sol_qkv_chunk_tokens', 0) or 0),
            'out': int(state.get('sol_out_proj_chunk_tokens', 0) or 0),
            'guard': int(state.get('inter_block_vram_guard_mb', 0) or 0),
            'cooldown': int(state.get('inter_block_guard_cooldown_blocks', 0) or 0),
            'late_start': int(state.get('late_block_guard_start', 40) or 40),
            'late_target': int(state.get('late_block_guard_target_mb', 0) or 0),
            'step_cleanup': int(state.get('step_boundary_cleanup_mb', 0) or 0),
        }

        _freeze_large_attention_chunks = int(token_count) >= 180000
        _backend = str(state.get('model_runtime_backend', 'unknown')).lower()

        # Conservative v1 thresholds.  Block 0 has already proven that the
        # sequence fits with the user's SAFE baseline.  We only relax memory
        # management when there is enough free + allocator-reclaimable memory.
        if recoverable_mb >= 6144.0 or headroom_ratio >= 0.38:
            mode = 'FAST'
            # TEST AUTO memory-only: preserve the user's proven chunk sizes.
            # Only tune memory-management policy.
            state['chunk_tokens'] = before['mlp']
            state['sol_qkv_chunk_tokens'] = before['qkv']
            state['sol_out_proj_chunk_tokens'] = before['out']
            state['inter_block_vram_guard_mb'] = min(before['guard'], 768) if before['guard'] > 0 else 0
            state['inter_block_guard_cooldown_blocks'] = max(before['cooldown'], 8)
            state['late_block_guard_start'] = max(before['late_start'], 46)
            state['late_block_guard_target_mb'] = min(before['late_target'], 4096) if before['late_target'] > 0 else 0
            state['step_boundary_cleanup_mb'] = min(before['step_cleanup'], 1024) if before['step_cleanup'] > 0 else 0
        elif recoverable_mb >= 3584.0 or headroom_ratio >= 0.24:
            mode = 'BALANCED'
            # TEST AUTO memory-only: preserve chunk sizes in BALANCED too.
            state['chunk_tokens'] = before['mlp']
            state['sol_qkv_chunk_tokens'] = before['qkv']
            state['sol_out_proj_chunk_tokens'] = before['out']
            state['inter_block_vram_guard_mb'] = min(before['guard'], 1024) if before['guard'] > 0 else 0
            state['inter_block_guard_cooldown_blocks'] = max(before['cooldown'], 6)
            state['late_block_guard_start'] = max(before['late_start'], 44)
            state['late_block_guard_target_mb'] = min(before['late_target'], 5120) if before['late_target'] > 0 else 0
            state['step_boundary_cleanup_mb'] = min(before['step_cleanup'], 1536) if before['step_cleanup'] > 0 else 0
        else:
            mode = 'SAFE'
            # Preserve every user-supplied SAFE value.
            state['chunk_tokens'] = before['mlp']
            state['sol_qkv_chunk_tokens'] = before['qkv']
            state['sol_out_proj_chunk_tokens'] = before['out']
            state['inter_block_vram_guard_mb'] = before['guard']
            state['inter_block_guard_cooldown_blocks'] = before['cooldown']
            state['late_block_guard_start'] = before['late_start']
            state['late_block_guard_target_mb'] = before['late_target']
            state['step_boundary_cleanup_mb'] = before['step_cleanup']

        # Backend safety caps.  NVFP4 is intentionally uncapped here because
        # its current settings are the measured reference baseline.
        if _backend in ('int8', 'int8-convrot-w4a4'):
            # V27: native QuantizedTensor residency is more valuable than allocator
            # cache reclamation. Disable routine inter/late-block trims; the existing
            # emergency guard plus SOL adaptive retry remain the OOM safety net.
            state['inter_block_vram_guard_mb'] = 0
            state['late_block_guard_target_mb'] = 0
            state['step_boundary_cleanup_mb'] = 0
            state['chunk_tokens'] = min(int(state.get('chunk_tokens', before['mlp'])), 16384)
            if int(state.get('sol_qkv_chunk_tokens', 0) or 0) > 0:
                _qv = str(state.get('runtime_quant_variant') or state.get('quant_variant') or '').lower()
                state['sol_qkv_chunk_tokens'] = min(
                    int(state['sol_qkv_chunk_tokens']), 16384 if _qv == 'w4a8' else 8192
                )
            if int(state.get('sol_out_proj_chunk_tokens', 0) or 0) > 0:
                state['sol_out_proj_chunk_tokens'] = min(
                    int(state['sol_out_proj_chunk_tokens']), 16384
                )
        elif _backend in ('bf16', 'fp16', 'fp32'):
            state['chunk_tokens'] = min(int(state.get('chunk_tokens', before['mlp'])), 12288)
            if int(state.get('sol_qkv_chunk_tokens', 0) or 0) > 0:
                state['sol_qkv_chunk_tokens'] = min(
                    int(state['sol_qkv_chunk_tokens']), 8192
                )
            if int(state.get('sol_out_proj_chunk_tokens', 0) or 0) > 0:
                state['sol_out_proj_chunk_tokens'] = min(
                    int(state['sol_out_proj_chunk_tokens']), 12288
                )
        elif _backend in ('fp8', 'quantized-other'):
            state['chunk_tokens'] = min(int(state.get('chunk_tokens', before['mlp'])), 16384)
            if int(state.get('sol_out_proj_chunk_tokens', 0) or 0) > 0:
                state['sol_out_proj_chunk_tokens'] = min(
                    int(state['sol_out_proj_chunk_tokens']), 16384
                )

        after = {
            'mlp': int(state.get('chunk_tokens', before['mlp'])),
            'qkv': int(state.get('sol_qkv_chunk_tokens', before['qkv'])),
            'out': int(state.get('sol_out_proj_chunk_tokens', before['out'])),
            'guard': int(state.get('inter_block_vram_guard_mb', before['guard'])),
            'cooldown': int(state.get('inter_block_guard_cooldown_blocks', before['cooldown'])),
            'late_start': int(state.get('late_block_guard_start', before['late_start'])),
            'late_target': int(state.get('late_block_guard_target_mb', before['late_target'])),
            'step_cleanup': int(state.get('step_boundary_cleanup_mb', before['step_cleanup'])),
        }

        state['auto_vram_controller_mode'] = mode
        state['auto_vram_controller_probe'] = {
            'tokens': int(token_count),
            'total_mb': round(total_mb, 1),
            'driver_free_mb': round(free_mb, 1),
            'cached_mb': round(cached_mb, 1),
            'recoverable_mb': round(recoverable_mb, 1),
            'headroom_ratio': round(headroom_ratio, 4),
        }
        state['auto_vram_controller_before'] = before
        state['auto_vram_controller_after'] = after

        _lm_print(
            '[MiniMaxH3 LongMedia][AUTO VRAM] block0 probe: '
            f'backend={_backend}, {int(token_count)} tokens, total={total_mb:.0f} MB, '
            f'free={free_mb:.0f} MB, cached={cached_mb:.0f} MB, '
            f'recoverable={recoverable_mb:.0f} MB ({headroom_ratio*100.0:.1f}%) -> {mode}',
            flush=True,
        )
        _lm_print(
            '[MiniMaxH3 LongMedia][AUTO VRAM] runtime tuning: '
            f"MLP {before['mlp']}->{after['mlp']}, "
            f"QKV {before['qkv']}->{after['qkv']}, "
            f"OUT {before['out']}->{after['out']}, "
            f"guard {before['guard']}->{after['guard']} MB, "
            f"cooldown {before['cooldown']}->{after['cooldown']}, "
            f"late {before['late_start']}->{after['late_start']} "
            f"target {before['late_target']}->{after['late_target']} MB, "
            f"step-cleanup {before['step_cleanup']}->{after['step_cleanup']} MB",
            flush=True,
        )

    def _int8_prefetch_guard(self):
        """V321 oversubscription-aware emergency safety net for native Comfy INT8.

        A model larger than physical VRAM normally runs through Comfy's dynamic
        weight streaming.  In that regime a low CUDA *driver_free* value is not,
        by itself, an emergency: PyTorch's allocator cache is reclaimable and is
        valuable for avoiding block-by-block allocation/transfer thrash.

        Only trim when BOTH physical free VRAM and effective reclaimable headroom
        are critically low.  A block cooldown prevents repeated cache destruction
        when the workload is hovering around the emergency boundary.
        """
        state = self.state
        backend = str(state.get('model_runtime_backend', 'unknown')).lower()
        if backend not in ('int8', 'int8-convrot-w4a4') or not torch.cuda.is_available():
            return

        cooldown_left = int(state.get('int8_residency_guard_cooldown_left', 0) or 0)
        if cooldown_left > 0:
            state['int8_residency_guard_cooldown_left'] = cooldown_left - 1
            return

        snap = _cuda_memory_snapshot()
        if not snap:
            return
        mb = 1024.0 ** 2
        free_mb = float(snap['driver_free']) / mb
        cached_mb = float(snap['cached']) / mb
        effective_mb = free_mb + cached_mb

        emergency_free_mb = float(state.get('int8_residency_emergency_free_mb', 384) or 384)
        emergency_effective_mb = float(
            state.get('int8_residency_emergency_effective_mb', 768) or 768
        )
        min_cached_mb = float(state.get('int8_residency_min_cached_mb', 256) or 256)

        # Fast path for healthy streaming.  Example from a 20 GB model on a 16 GB
        # GPU: free~120 MB + cached~2780 MB => ~2.9 GB effective headroom.  The
        # allocator cache must be preserved in that state.
        if free_mb >= emergency_free_mb or effective_mb >= emergency_effective_mb:
            state['int8_residency_last_effective_mb'] = float(effective_mb)
            return
        if cached_mb < min_cached_mb:
            # There is little useful cache to reclaim, so empty_cache would not
            # materially improve the situation and would only add synchronization.
            return

        before_free, before_cached, before_effective = free_mb, cached_mb, effective_mb
        try:
            gc.collect()
            comfy.model_management.soft_empty_cache()
        except Exception as exc:
            _lm_print('[MiniMaxH3 LongMedia][V321 INT8 RESIDENCY] emergency cleanup failed: '
                  f'block={self.index}, {type(exc).__name__}: {exc}', flush=True)
            return

        state['int8_residency_emergency_trim_count'] = int(
            state.get('int8_residency_emergency_trim_count', 0) or 0
        ) + 1
        state['int8_residency_guard_cooldown_left'] = int(
            state.get('int8_residency_guard_cooldown_blocks', 8) or 8
        )
        after = _cuda_memory_snapshot()
        if after:
            after_free = float(after['driver_free']) / mb
            after_cached = float(after['cached']) / mb
            after_effective = after_free + after_cached
            _lm_print('[MiniMaxH3 LongMedia][V321 INT8 RESIDENCY] EMERGENCY TRIM: '
                  f'block={self.index}, free {before_free:.0f}->{after_free:.0f} MB, '
                  f'cached {before_cached:.0f}->{after_cached:.0f} MB, '
                  f'effective {before_effective:.0f}->{after_effective:.0f} MB, '
                  f'cooldown={state["int8_residency_guard_cooldown_left"]} blocks', flush=True)

    def _late_block_hard_guard(self, phase):
        """TEST: pressure-aware late guard with hysteresis and cooldown.

        Avoids repeated soft_empty_cache() calls when effective headroom is
        oscillating just below the configured late target.
        """
        state = self.state
        start_block = int(state.get('late_block_guard_start', 40) or 40)
        if self.index < start_block or not torch.cuda.is_available():
            return

        target_mb = float(int(state.get('late_block_guard_target_mb', 0) or 0))
        min_cached_mb = float(int(state.get('late_block_guard_min_cached_mb', 0) or 0))
        if target_mb <= 0:
            return

        snap = _cuda_memory_snapshot()
        if not snap:
            return

        mb = 1024.0 ** 2
        free_before = float(snap['driver_free']) / mb
        cached_before = float(snap['cached']) / mb
        effective_before = free_before + cached_before

        auto_mode = str(state.get('auto_vram_controller_mode') or 'SAFE').upper()
        effective_target = target_mb
        if auto_mode == 'FAST':
            effective_target = min(effective_target, 3584.0)
        elif auto_mode == 'BALANCED':
            effective_target = min(effective_target, 4608.0)

        # Hysteresis band:
        # - above target: never trim
        # - inside lower band (target - 1024 MB): avoid repeated trims
        # - below hard threshold: cleanup is allowed immediately
        hysteresis_mb = float(state.get('late_guard_hysteresis_mb', 1024.0) or 1024.0)
        hard_threshold = max(0.0, effective_target - hysteresis_mb)

        # Cooldown is counted in late-guard phases (pre_attention/pre_ffn).
        # A trim starts a cooldown; only a hard pressure drop may override it.
        cooldown_phases = int(state.get('late_guard_cooldown_phases', 4) or 4)
        cooldown_left = int(state.get('late_guard_cooldown_left', 0) or 0)

        if effective_before >= effective_target:
            state['late_guard_skipped_count'] = int(state.get('late_guard_skipped_count', 0) or 0) + 1
            if cooldown_left > 0:
                state['late_guard_cooldown_left'] = max(0, cooldown_left - 1)
            skipped = int(state['late_guard_skipped_count'])
            if skipped == 1 or skipped % 20 == 0:
                _lm_print(
                    '[MiniMaxH3 LongMedia][LATE GUARD] late guard skip: '
                    f'block {self.index} {phase}, effective={effective_before:.0f} MB '
                    f'>= target={effective_target:.0f} MB',
                    flush=True,
                )
            return

        # Inside the hysteresis band, keep cache intact rather than repeatedly
        # reclaiming a few hundred MB and immediately forcing reload/allocation.
        if effective_before >= hard_threshold:
            state['late_guard_hysteresis_skip_count'] = int(
                state.get('late_guard_hysteresis_skip_count', 0) or 0
            ) + 1
            if cooldown_left > 0:
                state['late_guard_cooldown_left'] = max(0, cooldown_left - 1)
            hs = int(state['late_guard_hysteresis_skip_count'])
            if hs == 1 or hs % 20 == 0:
                _lm_print(
                    '[MiniMaxH3 LongMedia][LATE GUARD] hysteresis skip: '
                    f'block {self.index} {phase}, effective={effective_before:.0f} MB, '
                    f'band={hard_threshold:.0f}..{effective_target:.0f} MB',
                    flush=True,
                )
            return

        # Cooldown blocks repeated trims unless we are in genuinely hard pressure.
        hard_pressure = effective_before < max(0.0, effective_target - 2048.0)
        if cooldown_left > 0 and not hard_pressure:
            state['late_guard_cooldown_skip_count'] = int(
                state.get('late_guard_cooldown_skip_count', 0) or 0
            ) + 1
            state['late_guard_cooldown_left'] = max(0, cooldown_left - 1)
            cs = int(state['late_guard_cooldown_skip_count'])
            if cs == 1 or cs % 20 == 0:
                _lm_print(
                    '[MiniMaxH3 LongMedia][LATE GUARD] cooldown skip: '
                    f'block {self.index} {phase}, effective={effective_before:.0f} MB, '
                    f'cooldown_left={cooldown_left}',
                    flush=True,
                )
            return

        # Nothing meaningful to reclaim: let emergency guard / Sol OOM retry
        # handle the true low-memory event instead of forcing a useless trim.
        if cached_before < min_cached_mb:
            state['late_guard_low_cache_skip_count'] = int(
                state.get('late_guard_low_cache_skip_count', 0) or 0
            ) + 1
            if cooldown_left > 0:
                state['late_guard_cooldown_left'] = max(0, cooldown_left - 1)
            return

        try:
            gc.collect()
            comfy.model_management.soft_empty_cache()
        except Exception as exc:
            _lm_print(
                f"[MiniMaxH3 LongMedia][LATE GUARD] cleanup failed "
                f"at block {self.index} {phase}: {exc!r}",
                flush=True,
            )
            return

        state['late_guard_cooldown_left'] = cooldown_phases
        after = _cuda_memory_snapshot()
        state['late_guard_trim_count'] = int(state.get('late_guard_trim_count', 0) or 0) + 1

        if after:
            free_after = float(after['driver_free']) / mb
            cached_after = float(after['cached']) / mb
            _lm_print(
                '[MiniMaxH3 LongMedia][LATE GUARD] late guard TRIM: '
                f'block {self.index} {phase}, effective={effective_before:.0f} MB, '
                f'free {free_before:.0f}->{free_after:.0f} MB, '
                f'cached {cached_before:.0f}->{cached_after:.0f} MB, '
                f'cooldown={cooldown_phases}',
                flush=True,
            )

    def _trace_attention(self, attn, x, rope_freqs, transformer_options, measure):
        """Execute stock H3 Attention.forward in measured substages.

        This mirrors comfy's MiniMax H3 Attention implementation exactly:
        qkv projection -> fused RMSNorm/RoPE -> optimized_attention -> out projection.
        It is used only for block 0 of the first forward so normal execution is
        unaffected after the diagnostic sample.
        """
        from comfy.ldm.modules.attention import optimized_attention
        import comfy.quant_ops

        s = int(x.shape[0])
        inner = int(attn.heads * attn.head_dim)

        q, k, v = measure('attn_qkv_proj', lambda: attn.qkv_proj(x).split(inner, dim=-1))
        v = v.view(s, attn.heads, attn.head_dim)

        def _norm_rope():
            nonlocal q, k
            if rope_freqs is not None:
                qv = q.view(1, s, attn.heads, attn.head_dim)
                kv = k.view(1, s, attn.heads, attn.head_dim)
                qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
                kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
                rot = int(rope_freqs.shape[-3] * 2)
                if comfy.model_management.in_training:
                    qv, kv = comfy.quant_ops.ck.rms_rope_split_half(
                        qv, kv, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
                    )
                else:
                    comfy.quant_ops.ck.rms_rope_split_half_(
                        qv, kv, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
                    )
                return qv[0], kv[0]
            return (
                attn.q_norm(q.view(s, attn.heads, attn.head_dim)),
                attn.k_norm(k.view(s, attn.heads, attn.head_dim)),
            )

        q, k = measure('attn_rms_rope', _norm_rope)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        out = measure(
            'attn_kernel',
            lambda: optimized_attention(
                q, k, v, attn.heads, mask=None, skip_reshape=True,
                transformer_options=transformer_options,
            ),
        )
        # Drop Q/K/V before the output projection just as soon as the attention
        # kernel no longer needs them, so this diagnostic does not inflate the
        # projection peak by extending their lifetime.
        del q, k, v
        return measure('attn_out_proj', lambda: attn.out_proj(out.squeeze(0)))

    def _chunk_mlp(self, block, h):
        """Legacy exact token-chunked MLP path retained for compatibility."""
        token_count = int(h.shape[0])
        if token_count <= self.chunk_tokens:
            return block.mlp(h)

        state = self.state
        chunks = (token_count + self.chunk_tokens - 1) // self.chunk_tokens
        state['mlp_chunked_calls'] = int(state.get('mlp_chunked_calls', 0)) + 1
        state['max_sequence_tokens'] = max(int(state.get('max_sequence_tokens', 0)), token_count)
        state['max_chunks_per_mlp'] = max(int(state.get('max_chunks_per_mlp', 0)), chunks)
        if not state.get('announced'):
            _lm_print(
                '[MiniMaxH3 LongMedia] Low-VRAM MLP enabled: '
                f'sequence {token_count} tokens -> {chunks} chunks of <= {self.chunk_tokens}',
                flush=True,
            )
            state['announced'] = True

        state['mlp_inplace_reuse'] = True
        for start in range(0, token_count, self.chunk_tokens):
            end = min(token_count, start + self.chunk_tokens)
            chunk_out = block.mlp(h[start:end])
            h[start:end].copy_(chunk_out)
            del chunk_out
        return h

    def _chunk_norm2_mlp_gate_residual(self, block, x, shift, scale, gate, segments):
        """Stream the entire second half of an H3 block over token chunks.

        Stock H3 materializes ``norm2(x)`` plus AdaLN modulation as a full
        [tokens, hidden] tensor before entering the FFN.  At ~225k packed tokens
        that BF16 buffer is about 2.26 GiB.  This path keeps the whole sequence
        resident only once in ``x`` and performs, per token chunk:

            norm2 -> scale/shift modulation -> MLP -> gate -> residual add

        Every operation in this chain is token-local.  Later chunks therefore
        read untouched rows of ``x`` and are mathematically independent of rows
        already updated by earlier chunks.  No full-size norm2/modulated or MLP
        output tensor is ever created.
        """
        token_count = int(x.shape[0])
        self.state['current_token_count'] = token_count
        self.state['last_token_count'] = token_count
        # AUTO controller may raise/lower the runtime MLP chunk after
        # block-0 calibration.  self.chunk_tokens remains the user/manual fallback.
        runtime_chunk_tokens = int(self.state.get('chunk_tokens', self.chunk_tokens) or self.chunk_tokens)
        state = self.state
        _ultra_streaming = str(state.get('memory_mode', 'normal')) == 'ultra_low_vram'
        _chunk_floor = 64 if _ultra_streaming else 256
        chunk_tokens = min(max(_chunk_floor, runtime_chunk_tokens), max(1, token_count))
        if bool(state.get('auto_mlp_chunk_enabled')) and (not _ultra_streaming) and torch.cuda.is_available():
            # Estimate usable allocator headroom without forcing an empty_cache().
            # fc1 + SwiGLU temporary storage is roughly 80-90 KiB/token for H3;
            # 96 KiB/token deliberately includes allocator/alignment margin.
            try:
                free_b, _total_b = torch.cuda.mem_get_info(x.device)
                allocated_b = torch.cuda.memory_allocated(x.device)
                reserved_b = torch.cuda.memory_reserved(x.device)
                reclaimable_b = max(0, int(reserved_b) - int(allocated_b))
                effective_b = int(free_b) + reclaimable_b
                safety_b = int(state.get('auto_mlp_chunk_safety_mb', 640)) * 1024 * 1024
                per_token_b = max(1, int(state.get('auto_mlp_chunk_bytes_per_token', 96 * 1024)))
                usable_b = max(0, effective_b - safety_b)
                max_tokens_by_mem = max(1024, usable_b // per_token_b)
                ceiling = min(int(runtime_chunk_tokens), int(max_tokens_by_mem))
                ladder = (16384, 8192, 4096, 2048, 1024)
                selected = 1024
                for candidate in ladder:
                    if candidate <= ceiling:
                        selected = candidate
                        break
                chunk_tokens = min(max(1024, selected), max(1, token_count))
                previous = state.get('auto_mlp_chunk_last')
                if previous != int(chunk_tokens):
                    state['auto_mlp_chunk_last'] = int(chunk_tokens)
                    state['auto_mlp_chunk_changes'] = int(state.get('auto_mlp_chunk_changes', 0)) + 1
                    _lm_print(
                        '[MiniMaxH3 LongMedia][V30 AUTO MLP] '
                        f'block={self.index}, effective_headroom={effective_b/(1024*1024):.0f} MB, '
                        f'safety={safety_b/(1024*1024):.0f} MB, selected={chunk_tokens} tokens',
                        flush=True,
                    )
            except Exception as _auto_mlp_exc:
                if not state.get('auto_mlp_chunk_error_announced'):
                    _lm_print(
                        '[MiniMaxH3 LongMedia][V30 AUTO MLP] fallback to policy chunk: '
                        f'{type(_auto_mlp_exc).__name__}: {_auto_mlp_exc}',
                        flush=True,
                    )
                    state['auto_mlp_chunk_error_announced'] = True
        chunks = (token_count + chunk_tokens - 1) // chunk_tokens
        state['mlp_chunked_calls'] = int(state.get('mlp_chunked_calls', 0)) + 1
        state['mlp_fused_gate_residual_calls'] = int(state.get('mlp_fused_gate_residual_calls', 0)) + 1
        state['norm2_mlp_fused_calls'] = int(state.get('norm2_mlp_fused_calls', 0)) + 1
        state['max_sequence_tokens'] = max(int(state.get('max_sequence_tokens', 0)), token_count)
        state['max_chunks_per_mlp'] = max(int(state.get('max_chunks_per_mlp', 0)), chunks)
        state['mlp_fused_gate_residual'] = True
        state['norm2_mlp_fused_streaming'] = True
        if not state.get('announced'):
            _lm_print(
                '[MiniMaxH3 LongMedia] Low-VRAM fused norm2+MLP+gate+residual enabled: '
                f'sequence {token_count} tokens -> {chunks} chunks of <= {chunk_tokens}',
                flush=True,
            )
            state['announced'] = True

        # Tiny per-segment vectors are cast once and reused.  They are negligible
        # compared with even one token chunk and avoid repeated weight casts.
        rows = {}
        for _a, _b, row in segments:
            row = int(row)
            if row not in rows:
                rows[row] = (
                    shift[row].to(dtype=x.dtype, device=x.device),
                    scale[row].to(dtype=x.dtype, device=x.device),
                    gate[row].to(dtype=x.dtype, device=x.device),
                )

        # v0.3.60: block-resident native INT8 MLP weights.  v0.3.59 called
        # stock block.mlp() once per token chunk, which repeated cast/VBAR weight
        # preparation dozens of times per transformer block and roughly doubled
        # iteration time on the 32.4GB/16GB out-of-core case.  Prepare fc1/fc2
        # once per block and reuse them, but only after a strict stock-vs-cached
        # numerical parity probe on block 0.  If parity fails, the fast path is
        # disabled globally for the rest of the job.
        _int8_backend = (
            _v12_is_int8_family(state)
            and not comfy.model_management.in_training
            and state.get('int8_cached_mlp_parity', 'unknown') != 'failed'
        )
        state['stock_mlp_math'] = False
        _fc1_handle = _fc2_handle = None

        _v19_active = _v19_selected_block(state, self.index)
        _v19_offsets = (
            _v19_probe_offsets(token_count, 8192) if _v19_active else []
        )
        _v19_h_reference = None
        _v19_mlp_reference = None
        _v19_final_reference = None
        _v19_h_actual_parts = []
        _v19_mlp_actual_parts = []
        _v19_final_actual_parts = []
        if _v19_active:
            _v19_before_parts = []
            _v19_h_parts = []
            _v19_lengths = []
            for _offset in _v19_offsets:
                _probe_end = min(token_count, int(_offset) + 4)
                _before = x[int(_offset):_probe_end].detach().clone()
                _href = block.norm2(_before)
                for a, b, row in segments:
                    lo = max(int(_offset), int(a))
                    hi = min(_probe_end, int(b))
                    if lo >= hi:
                        continue
                    local_lo = lo - int(_offset)
                    local_hi = hi - int(_offset)
                    shift_row, scale_row, _gate_row = rows[int(row)]
                    _href[local_lo:local_hi].mul_(
                        1.0 + scale_row
                    ).add_(shift_row)
                _v19_before_parts.append(_before)
                _v19_h_parts.append(_href)
                _v19_lengths.append(int(_href.shape[0]))
            _v19_h_reference = torch.cat(_v19_h_parts, dim=0)
            _v19_mlp_reference = block.mlp(
                _v19_h_reference.clone()
            ).detach()
            _v19_final_parts = []
            _cursor = 0
            for _offset, _before, _length in zip(
                _v19_offsets, _v19_before_parts, _v19_lengths
            ):
                _expected = _before.clone()
                _mlp_piece = _v19_mlp_reference[_cursor:_cursor + _length]
                _probe_end = int(_offset) + _length
                for a, b, row in segments:
                    lo = max(int(_offset), int(a))
                    hi = min(_probe_end, int(b))
                    if lo >= hi:
                        continue
                    local_lo = lo - int(_offset)
                    local_hi = hi - int(_offset)
                    _shift_row, _scale_row, gate_row = rows[int(row)]
                    _expected[local_lo:local_hi].addcmul_(
                        _mlp_piece[local_lo:local_hi], gate_row
                    )
                _v19_final_parts.append(_expected)
                _cursor += _length
            _v19_final_reference = torch.cat(_v19_final_parts, dim=0)
            del _v19_before_parts, _v19_h_parts, _v19_final_parts

        try:
            for start in range(0, token_count, chunk_tokens):
                end = min(token_count, start + chunk_tokens)
                # norm2 produces only a chunk-sized temporary.
                h_chunk = block.norm2(x[start:end])

                # Apply the same packed-segment AdaLN modulation as the stock full
                # path, but only to intersections inside this chunk.
                for a, b, row in segments:
                    lo = max(start, int(a))
                    hi = min(end, int(b))
                    if lo >= hi:
                        continue
                    local_lo = lo - start
                    local_hi = hi - start
                    shift_row, scale_row, _gate_row = rows[int(row)]
                    h_chunk[local_lo:local_hi].mul_(1.0 + scale_row).add_(shift_row)

                if _v19_active:
                    for _offset in _v19_offsets:
                        if start <= int(_offset) < end:
                            _local = int(_offset) - start
                            _rows = min(4, end - int(_offset))
                            _v19_h_actual_parts.append(
                                h_chunk[_local:_local + _rows].detach().clone()
                            )

                if _int8_backend and _fc1_handle is None:
                    _mlp_probe = h_chunk[:4]
                    _need_parity = False
                    _stock_mlp_probe = None
                    if _need_parity:
                        # Pay one tiny stock call once per job, before enabling
                        # resident weights for every subsequent block/chunk.
                        _stock_mlp_probe = block.mlp(_mlp_probe).detach()

                    _fc1_handle = _int8_prepare_block_linear(
                        block.mlp.fc1, _mlp_probe
                    )
                    # fc2 must be prepared with its real post-SwiGLU input shape,
                    # not the hidden-size fc1 input used by older builds.
                    _prep_fc1 = torch.nn.functional.linear(
                        _mlp_probe, _fc1_handle['weight'], _fc1_handle['bias']
                    )
                    _prep_gate, _prep_up = _prep_fc1.chunk(2, dim=-1)
                    _fc2_probe = torch.nn.functional.silu(_prep_gate).mul_(_prep_up)
                    _fc2_handle = _int8_prepare_block_linear(
                        block.mlp.fc2, _fc2_probe
                    )
                    del _prep_gate, _prep_up, _prep_fc1, _fc2_probe

                    if _need_parity:
                        _cached_fc1_probe = _int8_cached_linear(
                            _fc1_handle, _mlp_probe
                        )
                        _cached_mlp_probe = _int8_cached_linear(
                            _fc2_handle, _cached_fc1_probe, input_act='swiglu'
                        )
                        try:
                            _s = _stock_mlp_probe.detach().to(device='cpu', dtype=torch.float32)
                            _c = _cached_mlp_probe.detach().to(device='cpu', dtype=torch.float32)
                            _d = _c - _s
                            _eps = 1.0e-12
                            _rms_ref = float(torch.sqrt(torch.mean(_s.square())).item()) if _s.numel() else 0.0
                            _rms_diff = float(torch.sqrt(torch.mean(_d.square())).item()) if _d.numel() else 0.0
                            _rel = _rms_diff / max(_eps, _rms_ref)
                            _den = float(torch.linalg.vector_norm(_s).item() * torch.linalg.vector_norm(_c).item()) if _s.numel() else 1.0
                            _cos = float(torch.dot(_s.flatten(), _c.flatten()).item() / max(_eps, _den)) if _s.numel() else 1.0
                            _finite = bool(torch.isfinite(_s).all() and torch.isfinite(_c).all() and torch.isfinite(_d).all())
                            _ok = _finite and _rel <= 5.0e-5 and _cos >= 0.99999
                        except Exception:
                            _ok, _rel, _cos = False, float('inf'), -1.0
                        state['int8_cached_mlp_parity'] = 'verified' if _ok else 'failed'
                        _lm_print(
                            '[MiniMaxH3 LongMedia][0.3.60 MLP PARITY] '
                            f"{'PASS' if _ok else 'FAIL'} rel_rms={_rel:.3e} cosine={_cos:.8f}; "
                            + ('block-resident fc1/fc2 enabled' if _ok else 'falling back to stock MLP'),
                            flush=True,
                        )
                        del _stock_mlp_probe, _cached_fc1_probe, _cached_mlp_probe
                        if not _ok:
                            _int8_release_block_linear(_fc2_handle)
                            _int8_release_block_linear(_fc1_handle)
                            _fc1_handle = _fc2_handle = None
                            _int8_backend = False
                            state['stock_mlp_math'] = True

                    if _fc1_handle is not None and not state.get('int8_block_mlp_weights_announced'):
                        _lm_print(
                            '[MiniMaxH3 LongMedia][0.3.61 BLOCK-RESIDENT SAFE MLP] '
                            'fc1+fc2 prepared once per H3 block and reused across all token chunks',
                            flush=True,
                        )
                        state['int8_block_mlp_weights_announced'] = True

                if _fc1_handle is not None and _fc2_handle is not None:
                    # v0.3.61: keep block-resident weights for throughput, but do
                    # NOT use the custom fused int8_linear(input_act='swiglu')
                    # shortcut.  Tiny 4-token probes could pass while a real
                    # thousands-token chunk produced catastrophic latent corruption.
                    # Use QuantizedTensor/F.linear dispatch for both projections and
                    # apply SwiGLU explicitly, matching stock H3 math ordering.
                    _ff = torch.nn.functional.linear(
                        h_chunk, _fc1_handle['weight'], _fc1_handle['bias']
                    )
                    _gate, _up = _ff.chunk(2, dim=-1)
                    _ff_act = torch.nn.functional.silu(_gate).mul_(_up)
                    chunk_out = torch.nn.functional.linear(
                        _ff_act, _fc2_handle['weight'], _fc2_handle['bias']
                    )
                    del _gate, _up, _ff_act, _ff
                else:
                    chunk_out = block.mlp(h_chunk)

                # v0.3.61 real-shape parity gate.  Validate the resident path on
                # the complete first runtime chunk, because INT8 kernels/layouts
                # can be shape-sensitive and the old 4-token probe was not
                # representative.  One duplicate stock MLP call is paid once.
                if (
                    self.index == 0
                    and start == 0
                    and _fc1_handle is not None
                    and _fc2_handle is not None
                    and state.get('int8_cached_mlp_parity', 'unknown') == 'unknown'
                ):
                    _s = _c = _d = None
                    _stock_chunk = None
                    try:
                        _stock_chunk = block.mlp(h_chunk.clone()).detach()
                        _s = _stock_chunk.to(device='cpu', dtype=torch.float32)
                        _c = chunk_out.detach().to(device='cpu', dtype=torch.float32)
                        _d = _c - _s
                        _eps = 1.0e-12
                        _rms_ref = float(torch.sqrt(torch.mean(_s.square())).item()) if _s.numel() else 0.0
                        _rms_diff = float(torch.sqrt(torch.mean(_d.square())).item()) if _d.numel() else 0.0
                        _rel = _rms_diff / max(_eps, _rms_ref)
                        _den = float(torch.linalg.vector_norm(_s).item() * torch.linalg.vector_norm(_c).item()) if _s.numel() else 1.0
                        _cos = float(torch.dot(_s.flatten(), _c.flatten()).item() / max(_eps, _den)) if _s.numel() else 1.0
                        _finite = bool(torch.isfinite(_s).all() and torch.isfinite(_c).all() and torch.isfinite(_d).all())
                        _ok = _finite and _rel <= 2.0e-4 and _cos >= 0.9999
                    except Exception:
                        _ok, _rel, _cos = False, float('inf'), -1.0
                        _stock_chunk = None
                    state['int8_cached_mlp_parity'] = 'verified' if _ok else 'failed'
                    _lm_print(
                        '[MiniMaxH3 LongMedia][0.3.61 REAL-CHUNK MLP PARITY] '
                        f"{'PASS' if _ok else 'FAIL'} rows={int(h_chunk.shape[0])} "
                        f"rel_rms={_rel:.3e} cosine={_cos:.8f}; "
                        + ('resident F.linear path enabled' if _ok else 'resident path DISABLED -> stock MLP'),
                        flush=True,
                    )
                    if not _ok:
                        if _stock_chunk is not None:
                            chunk_out = _stock_chunk.to(device=x.device, dtype=x.dtype)
                        _int8_release_block_linear(_fc2_handle)
                        _int8_release_block_linear(_fc1_handle)
                        _fc1_handle = _fc2_handle = None
                        _int8_backend = False
                        state['stock_mlp_math'] = True
                    for _tmp in (_s, _c, _d):
                        try:
                            del _tmp
                        except Exception:
                            pass
                    if _stock_chunk is not None:
                        del _stock_chunk

                del h_chunk

                if _v19_active:
                    for _offset in _v19_offsets:
                        if start <= int(_offset) < end:
                            _local = int(_offset) - start
                            _rows = min(4, end - int(_offset))
                            _v19_mlp_actual_parts.append(
                                chunk_out[_local:_local + _rows].detach().clone()
                            )

                # Consume the FFN result immediately into the corresponding residual
                # rows.  A chunk may cross modality/conditioning boundaries.
                for a, b, row in segments:
                    lo = max(start, int(a))
                    hi = min(end, int(b))
                    if lo >= hi:
                        continue
                    local_lo = lo - start
                    local_hi = hi - start
                    _shift_row, _scale_row, gate_row = rows[int(row)]
                    x[lo:hi].addcmul_(chunk_out[local_lo:local_hi], gate_row)
                if _v19_active:
                    for _offset in _v19_offsets:
                        if start <= int(_offset) < end:
                            _rows = min(4, end - int(_offset))
                            _v19_final_actual_parts.append(
                                x[int(_offset):int(_offset) + _rows]
                                .detach().clone()
                            )
                del chunk_out

        finally:
            if _fc2_handle is not None:
                _int8_release_block_linear(_fc2_handle)
            if _fc1_handle is not None:
                _int8_release_block_linear(_fc1_handle)

        if _v19_active:
            if _v19_h_actual_parts:
                _v19_report(
                    state, 'NORM2-ADALN', _v19_h_reference,
                    torch.cat(_v19_h_actual_parts, dim=0),
                    offsets=_v19_offsets,
                )
            if _v19_mlp_actual_parts:
                _v19_report(
                    state, 'MLP-FC1-FC2', _v19_mlp_reference,
                    torch.cat(_v19_mlp_actual_parts, dim=0),
                    offsets=_v19_offsets,
                )
            if _v19_final_actual_parts:
                _v19_report(
                    state, 'MLP-GATE-RESIDUAL', _v19_final_reference,
                    torch.cat(_v19_final_actual_parts, dim=0),
                    offsets=_v19_offsets,
                )
            _v19_h_actual_parts.clear()
            _v19_mlp_actual_parts.clear()
            _v19_final_actual_parts.clear()
            del _v19_h_reference, _v19_mlp_reference, _v19_final_reference

        rows.clear()
        return x

    def _measure(self, name, fn, state, device):
        # Only block 0 / first forward is synchronized and measured. Every other
        # block follows the exact same chunked execution without profiler stalls.
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        before = _cuda_memory_snapshot()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        out = fn()
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        after = _cuda_memory_snapshot()
        peak_alloc = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        entry = {
            'stage': name,
            'allocated_before_mb': _mb(before['allocated']),
            'allocated_after_mb': _mb(after['allocated']),
            'reserved_before_mb': _mb(before['reserved']),
            'reserved_after_mb': _mb(after['reserved']),
            'driver_free_after_mb': _mb(after['driver_free']),
            'peak_allocated_mb': _mb(peak_alloc),
            'peak_reserved_mb': _mb(peak_reserved),
            'elapsed_ms': round(elapsed_ms, 1),
        }
        state['stages'].append(entry)
        state['highest_block_peak_allocated_mb'] = max(
            float(state.get('highest_block_peak_allocated_mb') or 0.0), float(entry['peak_allocated_mb']))
        state['highest_block_peak_reserved_mb'] = max(
            float(state.get('highest_block_peak_reserved_mb') or 0.0), float(entry['peak_reserved_mb']))
        if float(entry['peak_allocated_mb']) >= float(state.get('worst_stage_peak_allocated_mb') or -1.0):
            state['worst_stage'] = name
            state['worst_stage_peak_allocated_mb'] = float(entry['peak_allocated_mb'])
        _lm_print(
            '[MiniMaxH3 LongMedia] H3 block0 stage: '
            f"{name}, alloc {entry['allocated_before_mb']:.1f} -> {entry['allocated_after_mb']:.1f} MB, "
            f"peak {entry['peak_allocated_mb']:.1f} MB, reserved peak {entry['peak_reserved_mb']:.1f} MB, "
            f"free {entry['driver_free_after_mb']:.1f} MB, {entry['elapsed_ms']:.1f} ms",
            flush=True,
        )
        return out

    def __call__(self, args, extra_options):
        original_block = extra_options['original_block']
        state = self.state
        if self.index == 0:
            # V12-A/B2 is gated inside this helper. INT8 and W4A8 participate;
            # NVFP4 and floating-point backends do not mutate V11 state.
            _v12_begin_int8_sol_forward(state)
        state['active_block_index'] = int(self.index)

        # V40 production baseline: keep the proven V39 execution path but remove
        # first-forward CUDA-synchronized per-block profiling overhead entirely.
        if self.index == 0:
            state['v30_block_metric_active'] = False
        _v30_metric = False
        if _v30_metric:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            _v30_block_t0 = time.perf_counter()
            _v30_attn_s = 0.0
            _v30_mlp_s = 0.0

        # AUTO attention selection happens before any QKV allocation:
        # `x` is assigned later in this wrapper, so use args['img'] directly.
        # This is still before any norm/attention/QKV allocation.
        _preblock_img = args['img']
        _preblock_tokens = int(_preblock_img.shape[0])
        state['current_token_count'] = _preblock_tokens

        _requested_mode = str(
            state.get('requested_attention_mode', state.get('sol_mode', 'existing'))
        )
        if _requested_mode == 'auto':
            _effective_mode, _auto_reason = _auto_select_h3_attention_mode(
                _preblock_tokens, state
            )
            state['sol_mode'] = _effective_mode
            state['auto_attention_selected_mode'] = _effective_mode
            state['auto_attention_reason'] = _auto_reason
            if not state.get('auto_attention_announced'):
                _lm_print(
                    '[MiniMaxH3 LongMedia][AUTO ATTENTION] '
                    f'{_auto_reason}; selected before QKV allocation; '
                    'full token semantics preserved',
                    flush=True,
                )
                state['auto_attention_announced'] = True
        else:
            state['sol_mode'] = _requested_mode

        # v0.3.110 OOM Governor V4: attention-workspace preflight.
        # KJ SageAttention materializes full-sequence quantization buffers (for
        # example V_fp8 ~= tokens * hidden_dim bytes) and cannot be rescued by
        # MLP chunking. On constrained GPUs, geometry that is known to exceed the
        # safe full-Sage envelope is routed to LongMedia's bounded QKV/Sol path
        # *before* block.attn allocates Q/K/V. This is a safety fallback, never an
        # after-OOM retry, so the CUDA allocator remains healthy.
        try:
            _v110_total_gb = float(torch.cuda.get_device_properties(
                torch.cuda.current_device()).total_memory) / (1024.0 ** 3)
        except Exception:
            _v110_total_gb = 0.0
        _v110_existing_unsafe = bool(
            state.get('sol_mode') == 'existing' and (
                (_v110_total_gb and _v110_total_gb <= 18.5 and _preblock_tokens >= 120000)
                or (_v110_total_gb and _v110_total_gb <= 26.0 and _preblock_tokens >= 180000)
            )
        )
        if _v110_existing_unsafe:
            state['sol_mode'] = 'sol'
            state['v110_attention_safety_fallback'] = True
            if not state.get('v110_attention_safety_announced'):
                _lm_print(
                    '[MiniMaxH3 LongMedia][0.3.110 ATTENTION PREFLIGHT] '
                    f'existing/Sage rejected before QKV allocation: tokens={_preblock_tokens}, '
                    f'VRAM={_v110_total_gb:.1f}GB; emergency route=SOL bounded-QKV; '
                    'reason=full-sequence Sage FP8 workspace unsafe',
                    flush=True,
                )
                state['v110_attention_safety_announced'] = True

        block = self._extract_block(original_block)
        if block is None:
            if self.index == 0 and not state.get('fallback_reason'):
                state['fallback_reason'] = 'could not extract DiTBlock from original_block closure'
                _lm_print('[MiniMaxH3 LongMedia] Low-VRAM MLP fallback: DiTBlock closure not found', flush=True)
            return original_block(args)

        x = args['img']
        t_emb = args['t_emb']
        mod_segments = args['mod_segments']
        rope_freqs = args['rope_freqs']
        transformer_options = args['transformer_options']

        # TEST build: block-0 step-boundary timing profiler disabled.
        # Removes profiling-only CUDA synchronization; SAFE guards are untouched.

        # TEST build: deep first-forward H3 stage profiler disabled.
        trace_this = False
        if trace_this:
            state['forward_count'] = 1
            state['first_forward_started'] = True
            state['first_forward_started_at'] = time.time()
            _lm_print('[MiniMaxH3 LongMedia] H3 block0 deep ATTENTION trace + MLP chunking: first forward started', flush=True)
            device = torch.cuda.current_device()
            measure = lambda name, fn: self._measure(name, fn, state, device)
        else:
            measure = lambda name, fn: fn()

        try:
            self._adaptive_memory_governor('block_start')
            vals = measure('adaln_proj', lambda: block.adaln_proj(t_emb))
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = vals

            _v19_block_active = _v19_selected_block(state, self.index)
            if _v19_block_active:
                _lm_print(
                    '[MiniMaxH3 LongMedia][V22 STAGE A/B ENTER] '
                    f'block={self.index}, generation={state.get("v21_stage_ab_generation", 0)}',
                    flush=True,
                )
            _v19_offsets = (
                _v19_probe_offsets(_preblock_tokens, 8192)
                if _v19_block_active else []
            )
            _v19_norm1_reference = None
            if _v19_block_active:
                _v19_norm1_parts = []
                for _offset in _v19_offsets:
                    _probe_end = min(_preblock_tokens, int(_offset) + 4)
                    _href = block.norm1(x[int(_offset):_probe_end])
                    for a, b, row in mod_segments:
                        lo = max(int(_offset), int(a))
                        hi = min(_probe_end, int(b))
                        if lo >= hi:
                            continue
                        local_lo = lo - int(_offset)
                        local_hi = hi - int(_offset)
                        _href[local_lo:local_hi].mul_(
                            1.0 + scale_msa[int(row)].to(_href.dtype)
                        ).add_(shift_msa[int(row)].to(_href.dtype))
                    _v19_norm1_parts.append(_href.detach())
                _v19_norm1_reference = torch.cat(_v19_norm1_parts, dim=0)
                del _v19_norm1_parts

            h = measure(
                'norm1_mod',
                lambda: self._mod_scale_shift(block.norm1(x), shift_msa, scale_msa, mod_segments),
            )
            if _v19_block_active:
                _v19_report(
                    state, 'NORM1-ADALN', _v19_norm1_reference,
                    torch.cat([
                        h[int(offset):min(_preblock_tokens, int(offset) + 4)]
                        for offset in _v19_offsets
                    ], dim=0),
                    offsets=_v19_offsets,
                )
                del _v19_norm1_reference
            self._late_block_hard_guard('pre_attention')
            if _v30_metric:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                _v30_attn_t0 = time.perf_counter()
            sol_mode = state.get('sol_mode', 'existing')
            if sol_mode != 'existing':
                attn_out = _run_h3_sol_attention(
                    block.attn, h, rope_freqs, transformer_options, state,
                    measure=measure if trace_this else None,
                )
            elif trace_this:
                attn_out = self._trace_attention(
                    block.attn, h, rope_freqs, transformer_options, measure
                )
            else:
                attn_out = block.attn(
                    h, rope_freqs=rope_freqs, transformer_options=transformer_options
                )
            if _v30_metric:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                _v30_attn_s = time.perf_counter() - _v30_attn_t0

            # The final block has consumed compressed K/V and kc statistics.
            # Release references before its FFN/output head so the next denoise
            # forward is forced to allocate a clean workspace.
            if self.index == int(state.get('last_patched_block_index', -1)):
                _v12_release_int8_sol_forward(
                    state, block_index=self.index
                )
            _v19_attn_reference = None
            if _v19_block_active:
                _v19_attn_parts = []
                for _offset in _v19_offsets:
                    _probe_end = min(_preblock_tokens, int(_offset) + 4)
                    _expected = x[int(_offset):_probe_end].detach().clone()
                    _other = attn_out[int(_offset):_probe_end]
                    for a, b, row in mod_segments:
                        lo = max(int(_offset), int(a))
                        hi = min(_probe_end, int(b))
                        if lo >= hi:
                            continue
                        local_lo = lo - int(_offset)
                        local_hi = hi - int(_offset)
                        _expected[local_lo:local_hi].addcmul_(
                            _other[local_lo:local_hi],
                            gate_msa[int(row)].to(_expected.dtype),
                        )
                    _v19_attn_parts.append(_expected)
                _v19_attn_reference = torch.cat(_v19_attn_parts, dim=0)
                del _v19_attn_parts
            x = measure(
                'attention_gate_residual',
                lambda: self._mod_gate(x, gate_msa, attn_out, mod_segments),
            )

            # v0.3.53 ultra-low-VRAM stage barrier.  Stock H3 immediately
            # replaces the full attention-normalized activation with the FFN
            # activation.  Our patched block used to keep the full `h` tensor
            # alive while entering chunked norm2/MLP, which raises the residency
            # peak exactly when the next block weights/cast buffers are needed.
            # On out-of-core 32+ GB models this can abort CUDA before the first
            # FFN kernel even launches.  Retire the attention stage explicitly.
            if bool(state.get('ultra_stage_barrier_required', False)):
                try:
                    torch.cuda.synchronize(device)
                except Exception:
                    pass
                try:
                    del h
                except Exception:
                    pass
                # attn_out is consumed by the residual above; free it before
                # Comfy has to materialize FFN weights.
                try:
                    del attn_out
                except Exception:
                    pass
                try:
                    import comfy.model_management as _mm
                    _mm.soft_empty_cache()
                except Exception:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                if not state.get('v353_stage_barrier_announced', False):
                    _lm_print(
                        '[MiniMaxH3 LongMedia][0.3.53 ULTRA STAGE BARRIER] '
                        'attention activation retired before FFN; CUDA synchronized; cache trimmed',
                        flush=True,
                    )
                    state['v353_stage_barrier_announced'] = True

            if _v19_block_active:
                _v19_report(
                    state, 'ATTENTION-GATE-RESIDUAL', _v19_attn_reference,
                    torch.cat([
                        x[int(offset):min(_preblock_tokens, int(offset) + 4)]
                        for offset in _v19_offsets
                    ], dim=0),
                    offsets=_v19_offsets,
                )
                del _v19_attn_reference
            try:
                del attn_out
            except Exception:
                pass
            # `h` is no longer needed after attention on any path. Releasing it
            # early also lowers normal/low-VRAM peaks without changing math.
            try:
                del h
            except Exception:
                pass

            self._late_block_hard_guard('pre_ffn')
            if _v30_metric:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                _v30_mlp_t0 = time.perf_counter()
            x = self._chunk_norm2_mlp_gate_residual(
                block, x, shift_mlp, scale_mlp, gate_mlp, mod_segments
            )
            if _v30_metric:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                _v30_mlp_s = time.perf_counter() - _v30_mlp_t0
            if _v19_block_active:
                _lm_print(
                    '[MiniMaxH3 LongMedia][V22 STAGE A/B EXIT] '
                    f'block={self.index}', flush=True,
                )
                _targets = tuple(int(v) for v in state.get('v21_stage_ab_targets', ()))
                if _targets and self.index == _targets[-1]:
                    state['v21_stage_ab_completed'] = True
                    state['v21_stage_ab_armed'] = False
                    _lm_print(
                        '[MiniMaxH3 LongMedia][V22 STAGE A/B COMPLETE] '
                        f'targets={list(_targets)}', flush=True,
                    )

            if trace_this:
                state['blocks'].append({
                    'block': 0,
                    'peak_allocated_mb': state.get('highest_block_peak_allocated_mb', 0.0),
                    'peak_reserved_mb': state.get('highest_block_peak_reserved_mb', 0.0),
                    'deep_trace': True,
                    'mlp_chunked': True,
                })
                _lm_print(
                    '[MiniMaxH3 LongMedia] H3 block0 attention+MLP trace summary: '
                    f"worst stage {state.get('worst_stage')}, "
                    f"peak allocated {state.get('highest_block_peak_allocated_mb', 0.0):.1f} MB",
                    flush=True,
                )
            # Drop modulation tables before asking the allocator to return dead
            # pages. They are block-local and no longer needed after the second
            # residual update.
            del vals, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
            self._inter_block_pressure_guard()

            # Leave real driver-free headroom for the next Comfy INT8 prefetch.
            self._int8_prefetch_guard()
            self._adaptive_memory_governor('block_end')

            if _v30_metric:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                _v30_total_s = time.perf_counter() - _v30_block_t0
                _v30_cum_s = time.perf_counter() - float(state.get('v30_block_metric_forward_t0', _v30_block_t0))
                try:
                    _free_b, _total_b = torch.cuda.mem_get_info()
                    _alloc_b = torch.cuda.memory_allocated()
                    _reserved_b = torch.cuda.memory_reserved()
                    _free_mb = _free_b / (1024 * 1024)
                    _alloc_mb = _alloc_b / (1024 * 1024)
                    _reserved_mb = _reserved_b / (1024 * 1024)
                except Exception:
                    _free_mb = _alloc_mb = _reserved_mb = float('nan')
                _mlp_chunk = int(state.get('auto_mlp_chunk_last') or state.get('chunk_tokens') or self.chunk_tokens)
                _other_s = max(0.0, _v30_total_s - _v30_attn_s - _v30_mlp_s)
                _lm_print(
                    '[MiniMaxH3 LongMedia][V35 BLOCK METRIC] '
                    f'block={self.index:02d}/49 total={_v30_total_s:.2f}s '
                    f'attn={_v30_attn_s:.2f}s kvpass={float(state.get("v33_last_kvpass_s", 0.0)):.2f}s '
                    f'querypass={float(state.get("v33_last_querypass_s", 0.0)):.2f}s '
                    f'qproj={float(state.get("v34_last_qproj_s", 0.0)):.2f}s rope={float(state.get("v34_last_rope_s", 0.0)):.2f}s '
                    f'sol={float(state.get("v34_last_sol_s", 0.0)):.2f}s outproj={float(state.get("v34_last_outproj_s", 0.0)):.2f}s copy={float(state.get("v34_last_copy_s", 0.0)):.2f}s '
                    f'mlp={_v30_mlp_s:.2f}s other={_other_s:.2f}s '
                    f'cum={_v30_cum_s:.2f}s qkv_chunk={int(state.get("sol_qkv_chunk_tokens", 0) or 0)} mlp_chunk={_mlp_chunk} '
                    f'alloc={_alloc_mb:.0f}MB reserved={_reserved_mb:.0f}MB driver_free={_free_mb:.0f}MB',
                    flush=True,
                )
                if self.index == int(state.get('last_patched_block_index', 49)):
                    state['v30_block_metric_active'] = False
                    _lm_print(
                        '[MiniMaxH3 LongMedia][V35 BLOCK METRIC] first-forward profiling COMPLETE; '
                        f'total_blocks_time={_v30_cum_s:.2f}s',
                        flush=True,
                    )
            return {'img': x}
        except Exception as exc:
            message = str(exc).lower()
            is_oom = isinstance(exc, getattr(torch, 'OutOfMemoryError', RuntimeError)) or 'out of memory' in message
            if is_oom:
                state['oom'] = True
                state['oom_block'] = self.index
                state['oom_stage'] = state.get('stages', [])[-1]['stage'] if state.get('stages') else 'unknown'
                state['oom_message'] = str(exc)[:2000]
                _lm_print(
                    f"[MiniMaxH3 LongMedia] H3 CUDA OOM in block {self.index} near {state.get('oom_stage')}: {state['oom_message']}",
                    flush=True,
                )
            raise



def _v24_final_report(state, stage, reference, candidate, *, stream, offsets):
    """Robust sampled A/B for the H3 final output layer (V24)."""
    key = f"{stream}:{stage}"
    done = state.setdefault('v24_final_ab_done', set())
    if key in done:
        return
    try:
        ref = reference.detach().to(device='cpu', dtype=torch.float32)
        got = candidate.detach().to(device='cpu', dtype=torch.float32)
        if tuple(ref.shape) != tuple(got.shape):
            _lm_print(
                '[MiniMaxH3 LongMedia][V24 FINAL-LAYER A/B] '
                f'stream={stream}, stage={stage}, verdict=SHAPE-MISMATCH, '
                f'reference={tuple(ref.shape)}, candidate={tuple(got.shape)}',
                flush=True,
            )
            return
        rf = ref.flatten()
        gf = got.flatten()
        diff = gf - rf
        finite = bool(torch.isfinite(rf).all().item() and torch.isfinite(gf).all().item() and torch.isfinite(diff).all().item())
        exact = bool(torch.equal(rf, gf))
        mismatches = int(torch.count_nonzero(diff).item())
        if exact:
            rel_rms = 0.0; cosine = 1.0; mean_abs = 0.0; max_abs = 0.0
        else:
            eps = 1.0e-30
            r64 = rf.to(torch.float64); g64 = gf.to(torch.float64); d64 = g64-r64
            rms_ref = float(torch.sqrt(torch.mean(r64.square())).item())
            rms_diff = float(torch.sqrt(torch.mean(d64.square())).item())
            rel_rms = rms_diff / max(eps, rms_ref)
            nr = float(torch.linalg.vector_norm(r64).item()); ng = float(torch.linalg.vector_norm(g64).item())
            denom = nr * ng
            cosine = 1.0 if denom <= eps and rms_diff <= eps else (0.0 if denom <= eps else float(torch.dot(r64,g64).item()/denom))
            cosine = max(-1.0, min(1.0, cosine))
            mean_abs = float(d64.abs().mean().item()); max_abs = float(d64.abs().max().item())
            del r64, g64, d64
        verdict = 'MATCH' if finite and (exact or (rel_rms <= 1.0e-5 and cosine >= 0.99999)) else 'DIVERGED'
        _lm_print(
            '[MiniMaxH3 LongMedia][V24 FINAL-LAYER A/B] '
            f'stream={stream}, stage={stage}, verdict={verdict}, offsets={list(offsets)}, rows={int(ref.shape[0])}, '
            f'exact={exact}, mismatches={mismatches}, rel_rms={rel_rms:.8e}, cosine={cosine:.10f}, '
            f'mean_abs={mean_abs:.8e}, max_abs={max_abs:.8e}, finite={finite}',
            flush=True,
        )
    except Exception as exc:
        _lm_print('[MiniMaxH3 LongMedia][V24 FINAL-LAYER A/B] '
              f'stream={stream}, stage={stage}, diagnostic failed: {type(exc).__name__}: {exc}', flush=True)
    finally:
        done.add(key)


def _v24_probe_local_offsets(n, max_rows=16):
    """Small deterministic row set spanning a stream without materializing full hidden tensors."""
    n = int(n)
    if n <= 0:
        return []
    anchors = [0, min(n-1, 1), n//4, n//2, (3*n)//4, max(0,n-2), n-1]
    # Fill to at most max_rows with evenly spaced rows.
    if n > 1:
        for i in range(max_rows):
            anchors.append((i*(n-1))//max(1,max_rows-1))
    return sorted(set(int(v) for v in anchors if 0 <= int(v) < n))[:max_rows]


def _install_h3_final_output_streaming(model_patcher, state, chunk_tokens=24576):
    """Patch MiniMax H3 FinalLayer so FP32 output-head inputs never exist full-size.

    Stock H3 FinalLayer normalizes/modulates the complete target video stream and
    then casts that [video_tokens, hidden] tensor to FP32 before video_out.  On
    long clips this FP32 island is several GiB (about 4.5 GiB at 225k x 5376).
    Norm/modulation and the output Linear are token-local, so process the target
    streams in token chunks and write only the small projected outputs into their
    final buffers.
    """
    try:
        base_model = getattr(model_patcher, 'model', None)
        diffusion = getattr(base_model, 'diffusion_model', None)
        final_layer = getattr(diffusion, 'final_layer', None)
        if final_layer is None:
            state['final_output_streaming_error'] = 'MiniMaxH3 final_layer not found'
            return False

        original = getattr(final_layer, '_latentlab_final_output_original_forward', None)
        if original is None:
            original = final_layer.forward
            final_layer._latentlab_final_output_original_forward = original

        # The model object is shared by ModelPatcher clones. Keep mutable runtime
        # settings on the module so a new prompt updates the already-installed wrapper.
        final_layer._latentlab_final_output_state = state
        final_layer._latentlab_final_output_chunk_tokens = max(256, int(chunk_tokens or 24576))

        if (
            getattr(final_layer, '_latentlab_final_output_streaming_installed', False)
            and not globals().get('_LONGMEDIA_HOT_RELOAD_BYPASS', False)
        ):
            return True

        import types

        def _streamed_forward(layer, x, t_emb, video_seg, audio_seg):
            st = getattr(layer, '_latentlab_final_output_state', {}) or {}
            chunk = max(256, int(getattr(layer, '_latentlab_final_output_chunk_tokens', 24576)))
            shift, scale = layer.adaln_proj(t_emb)
            va, vb, vrow = video_seg
            aa, ab, arow = audio_seg

            def _head_segment(start, stop, row, head, label):
                n = int(stop) - int(start)
                if n <= 0:
                    # Defensive fallback; target streams are expected non-empty.
                    return head((layer.norm(x[int(start):int(stop)]) * (1.0 + scale[int(row)]) + shift[int(row)]).to(torch.float32))

                # V24: sampled reference rows are evaluated with the stock mathematical
                # expression while the real generation still uses the streamed path.
                # This stays tiny (<=16 rows) and therefore does not recreate the
                # multi-GiB FP32 hidden tensor that final-output streaming was built to avoid.
                probe_local = _v24_probe_local_offsets(n) if not st.get(f'v24_final_{label}_captured', False) else []
                probe_abs = [int(start) + q for q in probe_local]
                candidate_hidden = {}
                candidate_output = {}

                out_features = getattr(head, 'out_features', None)
                if out_features is None:
                    w = getattr(head, 'weight', None)
                    out_features = int(w.shape[0]) if w is not None else None
                if out_features is None:
                    # Unknown custom Linear implementation: preserve correctness.
                    return head((layer.norm(x[int(start):int(stop)]) * (1.0 + scale[int(row)]) + shift[int(row)]).to(torch.float32))

                out = torch.empty((n, int(out_features)), device=x.device, dtype=torch.float32)
                chunks = (n + chunk - 1) // chunk
                for local in range(0, n, chunk):
                    end = min(n, local + chunk)
                    src = x[int(start) + local:int(start) + end]
                    h = layer.norm(src)
                    h.mul_(1.0 + scale[int(row)]).add_(shift[int(row)])
                    if probe_local:
                        for q_local, q_abs in zip(probe_local, probe_abs):
                            if local <= q_local < end:
                                candidate_hidden[q_abs] = h[q_local-local].detach().clone()
                    h32 = h.to(torch.float32)
                    del h
                    projected = head(h32)
                    if probe_local:
                        for q_local, q_abs in zip(probe_local, probe_abs):
                            if local <= q_local < end:
                                candidate_output[q_abs] = projected[q_local-local].detach().clone()
                    del h32
                    out[local:end].copy_(projected)
                    del projected
                if not st.get('final_output_streaming_announced'):
                    _lm_print(
                        '[MiniMaxH3 LongMedia] Final-output streaming enabled: '
                        f'{label} {n} tokens -> {chunks} chunks of <= {chunk}; FP32 hidden is chunk-local',
                        flush=True,
                    )
                    st['final_output_streaming_announced'] = True
                st['final_output_streaming_calls'] = int(st.get('final_output_streaming_calls', 0)) + 1
                st['final_output_max_tokens'] = max(int(st.get('final_output_max_tokens', 0)), n)
                st['final_output_max_chunks'] = max(int(st.get('final_output_max_chunks', 0)), chunks)

                if probe_local and candidate_hidden and candidate_output:
                    try:
                        # Gather all sampled rows in one tiny tensor.  Final norm and
                        # AdaLN are token-local; the output projection is checked with
                        # robust FP64 metrics because GEMM batching can change last-bit
                        # rounding even when the operation is mathematically equivalent.
                        idx = torch.tensor(probe_abs, device=x.device, dtype=torch.long)
                        ref_h = layer.norm(x.index_select(0, idx))
                        ref_h = ref_h * (1.0 + scale[int(row)]) + shift[int(row)]
                        got_h = torch.stack([candidate_hidden[q] for q in probe_abs], dim=0)
                        _v24_final_report(st, 'NORM-ADALN', ref_h, got_h, stream=label, offsets=probe_local)
                        ref_out = head(ref_h.to(torch.float32))
                        got_out = torch.stack([candidate_output[q] for q in probe_abs], dim=0)
                        _v24_final_report(st, 'OUTPUT-HEAD', ref_out, got_out, stream=label, offsets=probe_local)
                        st[f'v24_final_{label}_captured'] = True
                        del idx, ref_h, got_h, ref_out, got_out
                    except Exception as diag_exc:
                        _lm_print('[MiniMaxH3 LongMedia][V24 FINAL-LAYER A/B] '
                              f'stream={label}, diagnostic failed: {type(diag_exc).__name__}: {diag_exc}', flush=True)
                return out

            # TEST build: final-output CUDA timing profiler disabled; streaming unchanged.
            trace = False
            if trace:
                try:
                    torch.cuda.synchronize(torch.cuda.current_device())
                except Exception:
                    pass
                before = _cuda_memory_snapshot()
                torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
                started = time.perf_counter()

            v = _head_segment(va, vb, vrow, layer.video_out, 'video')
            a = _head_segment(aa, ab, arow, layer.audio_out, 'audio')

            if trace:
                try:
                    torch.cuda.synchronize(torch.cuda.current_device())
                except Exception:
                    pass
                after = _cuda_memory_snapshot()
                peak = int(torch.cuda.max_memory_allocated(torch.cuda.current_device()))
                elapsed = (time.perf_counter() - started) * 1000.0
                st['final_output_first_profile_complete'] = True
                st['final_output_peak_allocated_mb'] = _mb(peak)
                st['final_output_elapsed_ms'] = round(elapsed, 1)
                _lm_print(
                    '[MiniMaxH3 LongMedia] Final-output streaming profile: '
                    f'alloc {_mb(before["allocated"]):.1f} -> {_mb(after["allocated"]):.1f} MB, '
                    f'peak {_mb(peak):.1f} MB, reserved {_mb(after["reserved"]):.1f} MB, '
                    f'driver free {_mb(after["driver_free"]):.1f} MB, {elapsed:.1f} ms',
                    flush=True,
                )
            return v, a

        final_layer.forward = types.MethodType(_streamed_forward, final_layer)
        final_layer._latentlab_final_output_streaming_installed = True
        state['final_output_streaming_installed'] = True
        return True
    except Exception as exc:
        state['final_output_streaming_error'] = f'{type(exc).__name__}: {exc}'
        _lm_print('[MiniMaxH3 LongMedia] Final-output streaming fallback: ' + state['final_output_streaming_error'], flush=True)
        return False

class MiniMaxH3LatentLabMLPChunking:
    """Internal GUIDER wrapper enabling exact token-chunked H3 MLP execution."""

    DESCRIPTION = 'Internal H3 low-VRAM token-chunked MLP wrapper.'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'guider': ('GUIDER',),
                'chunk_tokens': ('INT', {'default': 8192, 'min': 256, 'max': 131072, 'step': 256}),
                'max_blocks': ('INT', {'default': 128, 'min': 1, 'max': 256, 'step': 1}),
                'sol_mode': (['auto', 'existing', 'sol', 'scheduled_sol'], {'default': 'existing'}),
                'sol_tau_start': ('FLOAT', {'default': 1.3, 'min': 0.0, 'max': 4.0, 'step': 0.05}),
                'sol_tau_end': ('FLOAT', {'default': 0.8, 'min': 0.0, 'max': 4.0, 'step': 0.05}),
                'sol_curve': (['linear', 'cosine', 'sqrt', 'smoothstep', 'exponential', 'step'], {'default': 'linear'}),
                'sol_min_tokens': ('INT', {'default': 4096, 'min': 256, 'max': 131072, 'step': 256}),
                'sol_dense_percent': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 0.9, 'step': 0.05}),
                'sol_sink_conditioning': (['exact_kv', 'exact_kv_and_rows', 'off'], {'default': 'exact_kv'}),
                'sol_qkv_chunk_tokens': ('INT', {'default': 8192, 'min': 0, 'max': 131072, 'step': 8192}),
                'sol_out_proj_chunk_tokens': ('INT', {'default': 24576, 'min': 0, 'max': 131072, 'step': 8192}),
                'vram_activation_reserve_mb': ('INT', {'default': 4096, 'min': 0, 'max': 12288, 'step': 512}),
                'inter_block_vram_guard_mb': ('INT', {'default': 2048, 'min': 0, 'max': 8192, 'step': 256}),
                'inter_block_guard_cooldown_blocks': ('INT', {'default': 4, 'min': 0, 'max': 32, 'step': 1}),
                'inter_block_guard_emergency_mb': ('INT', {'default': 512, 'min': 0, 'max': 4096, 'step': 256}),
                'inter_block_guard_emergency_cooldown_blocks': ('INT', {'default': 3, 'min': 0, 'max': 32, 'step': 1}),
                'late_block_guard_start': ('INT', {'default': 40, 'min': 0, 'max': 127, 'step': 1}),
                'late_block_guard_target_mb': ('INT', {'default': 6144, 'min': 0, 'max': 12288, 'step': 256}),
                'late_block_guard_min_cached_mb': ('INT', {'default': 512, 'min': 0, 'max': 4096, 'step': 256}),
                'step_boundary_cleanup_mb': ('INT', {'default': 2048, 'min': 0, 'max': 8192, 'step': 256}),
                'sol_sigma_hi': ('FLOAT', {'default': 1.0, 'min': -1000.0, 'max': 1000.0, 'step': 0.0001}),
                'sol_sigma_lo': ('FLOAT', {'default': 0.0, 'min': -1000.0, 'max': 1000.0, 'step': 0.0001}),
            }
        }

    RETURN_TYPES = ('GUIDER', 'H3_BLOCK_MEMORY_TRACE_STATE')
    RETURN_NAMES = ('guider', 'mlp_chunk_state')
    FUNCTION = 'wrap'
    CATEGORY = CATEGORY_LONGMEDIA

    def wrap(self, guider, memory_mode='normal', requested_memory_mode='normal', chunk_tokens=8192, max_blocks=128, sol_mode='existing', sol_tau_start=1.3, sol_tau_end=0.8, sol_curve='linear', sol_min_tokens=4096, sol_dense_percent=0.0, sol_sink_conditioning='exact_kv', sol_qkv_chunk_tokens=8192, sol_out_proj_chunk_tokens=24576, vram_activation_reserve_mb=4096, inter_block_vram_guard_mb=2048, inter_block_guard_cooldown_blocks=4, inter_block_guard_emergency_mb=512, inter_block_guard_emergency_cooldown_blocks=3, late_block_guard_start=40, late_block_guard_target_mb=6144, late_block_guard_min_cached_mb=512, step_boundary_cleanup_mb=2048, sol_sigma_hi=1.0, sol_sigma_lo=0.0):
        wrapped = copy.copy(guider)

        # Detect the actual diffusion execution backend before ComfyUI performs
        # memory planning.  This lets one workflow use backend-specific startup
        # headroom without altering the proven NVFP4 path.
        _runtime_patcher = getattr(guider, 'model_patcher', None)
        runtime_profile = _detect_h3_model_runtime(_runtime_patcher)
        _announce_h3_model_runtime(runtime_profile)

        runtime_policy = _h3_runtime_auto_policy(
            runtime_profile.get('backend', 'unknown'),
            quant_variant=runtime_profile.get('quant_variant'),
            chunk_tokens=chunk_tokens,
            sol_qkv_chunk_tokens=sol_qkv_chunk_tokens,
            sol_out_proj_chunk_tokens=sol_out_proj_chunk_tokens,
            vram_activation_reserve_mb=vram_activation_reserve_mb,
        )

        # Apply startup policy locally.  Public node inputs and workflow ABI stay
        # unchanged; the chosen values are stored in state for diagnostics.
        chunk_tokens = int(runtime_policy['chunk_tokens'])
        sol_qkv_chunk_tokens = int(runtime_policy['sol_qkv_chunk_tokens'])
        sol_out_proj_chunk_tokens = int(runtime_policy['sol_out_proj_chunk_tokens'])
        vram_activation_reserve_mb = int(runtime_policy['vram_activation_reserve_mb'])

        _lm_print(
            '[MiniMaxH3 LongMedia][MODEL POLICY] '
            f"backend={runtime_policy['backend']}, profile={runtime_policy['name']}, "
            f"quant_variant={runtime_profile.get('quant_variant')}, "
            f"reserve={vram_activation_reserve_mb} MB, "
            f"MLP={chunk_tokens}, QKV={sol_qkv_chunk_tokens}, "
            f"OUT={sol_out_proj_chunk_tokens}"
            + (
                ", native_weight_prefetch=ON, cache_trim=EMERGENCY_ONLY"
                if str(runtime_policy.get('backend', '')).lower()
                in ('int8', 'int8-convrot-w4a4')
                else ""
            ),
            flush=True,
        )

        # Ask ComfyUI's normal prepare_sampling/load_models_gpu path to reserve
        # additional activation headroom before it decides how many H3 weights
        # to keep resident on the GPU.  This is deliberately done by augmenting
        # ModelPatcher.memory_required(), rather than unloading weights from an
        # already-running forward.  ComfyUI can then use its native partial-load
        # / partial-unload machinery and keep the remainder on the offload device.
        reserve_bytes = max(0, int(vram_activation_reserve_mb)) * 1024 * 1024
        reserve_stats = {
            'requested_mb': int(vram_activation_reserve_mb),
            'memory_required_calls': 0,
            'last_base_required_mb': None,
            'last_total_required_mb': None,
            'patcher_cloned': False,
            'error': None,
        }
        if reserve_bytes > 0 and hasattr(guider, 'model_patcher'):
            try:
                reserve_patcher = guider.model_patcher.clone()
                base_memory_required = reserve_patcher.memory_required

                def _memory_required_with_activation_reserve(input_shape, _base=base_memory_required, _reserve=reserve_bytes, _stats=reserve_stats):
                    base = int(_base(input_shape))
                    total = base + int(_reserve)
                    _stats['memory_required_calls'] = int(_stats.get('memory_required_calls', 0)) + 1
                    _stats['last_base_required_mb'] = round(base / (1024 * 1024), 1)
                    _stats['last_total_required_mb'] = round(total / (1024 * 1024), 1)
                    if _stats['memory_required_calls'] == 1:
                        _lm_print(
                            '[MiniMaxH3 LongMedia] Activation VRAM reserve requested: '
                            f"base {base / (1024*1024):.1f} MB + reserve {_reserve / (1024*1024):.1f} MB "
                            f"= {total / (1024*1024):.1f} MB for ComfyUI memory planning",
                            flush=True,
                        )
                    return total

                # Instance attribute intentionally shadows the class method.
                # sampler_helpers.prepare_sampling() calls this before model load.
                reserve_patcher.memory_required = _memory_required_with_activation_reserve
                wrapped.model_patcher = reserve_patcher
                reserve_stats['patcher_cloned'] = True
            except Exception as exc:
                reserve_stats['error'] = f'{type(exc).__name__}: {exc}'
                _lm_print('[MiniMaxH3 LongMedia] Activation reserve fallback: ' + reserve_stats['error'], flush=True)
        try:
            wrapped.model_options = copy.deepcopy(getattr(guider, 'model_options', {}) or {})
        except Exception:
            wrapped.model_options = dict(getattr(guider, 'model_options', {}) or {})

        transformer_options = wrapped.model_options.setdefault('transformer_options', {})
        _runtime_backend = str(
            runtime_profile.get('backend', 'unknown')
        ).lower()
        transformer_options['latentlab_h3_runtime_backend'] = _runtime_backend
        transformer_options['model_runtime_backend'] = _runtime_backend
        # V27: native INT8 keeps Comfy's dynamic-VBAR prefetch enabled.
        # LongMedia owns activation memory, but upstream owns quantized-weight residency.
        _runtime_quant_variant = str(runtime_profile.get('quant_variant') or '').lower()

        # v0.3.25: native INT8 checkpoints can be larger than physical VRAM. On
        # 8-18 GB cards, speculative Comfy dynamic-VBAR prefetch may issue the next
        # 64 MB AIMDO device copy while the current block/activations already occupy
        # the remaining headroom, causing HostBuffer.read_file_slice -> CUDA OOM
        # before the first denoise step. Use demand loading on constrained cards.
        _device_vram_gb = None
        if torch.cuda.is_available():
            try:
                _device_vram_gb = float(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory) / (1024.0 ** 3)
            except Exception:
                _device_vram_gb = None
        _int8_low_vram_streaming = (
            _runtime_backend in ('int8', 'int8-convrot-w4a4')
            and _runtime_quant_variant != 'w4a8'
            and _device_vram_gb is not None
            and _device_vram_gb <= 18.5
        )
        _forced_streaming_mode = str(memory_mode) in ('low_vram', 'ultra_low_vram')
        _model_size_b = _h3_model_size_bytes_from_guider(guider) or 0
        _gpu_size_b = int((_device_vram_gb or 0.0) * (1024 ** 3))
        _out_of_core_streaming = bool(_model_size_b and _gpu_size_b and _model_size_b > int(_gpu_size_b * 1.05))

        # v0.3.75: for oversized AIMDO-backed H3 models on RAM-rich hosts, warm
        # checkpoint mmap pages into the OS file cache before the first denoise
        # forward.  This never creates a second tensor copy and never changes H3
        # math or VRAM policy; it only aims to replace repeated NVMe reads with
        # reclaimable filesystem-cache hits.
        if _out_of_core_streaming:
            _prewarm = _prewarm_h3_file_cache(
                getattr(guider, 'model_patcher', None),
                model_size_bytes=int(_model_size_b or 0),
                min_ram_headroom_gb=10.0,
            )
            builtins.print(
                '[MiniMaxH3 LongMedia][0.3.75 RAM FILE-CACHE PREWARM] '
                f"status={_prewarm.get('status')} payloads={_prewarm.get('payloads',0)} "
                f"payload={_prewarm.get('payload_bytes',0)/(1024**3):.1f}GB "
                f"budget={_prewarm.get('budget_bytes',0)/(1024**3):.1f}GB "
                f"touched={_prewarm.get('touched_bytes',0)/(1024**3):.1f}GB "
                f"time={_prewarm.get('seconds',0.0):.2f}s "
                f"reason={_prewarm.get('reason') or '-'}",
                flush=True,
            )
        # v0.3.77: recent Comfy/AIMDO (0.4.6+) contains threaded-loader and
        # DynamicVRAM fixes that did not exist when our 0.3.52 hard gate was
        # introduced.  For oversized *native* INT8, hand residency back to the
        # native DynamicVRAM loader as a clean A/B. W4A8 remains on the proven
        # guarded path. This changes scheduling only, never H3 math.
        _aimdo_raw, _aimdo_ver = _pkg_version_tuple('comfy-aimdo')
        _kitchen_raw, _kitchen_ver = _pkg_version_tuple('comfy-kitchen')
        _recent_aimdo = bool(_aimdo_ver is not None and _aimdo_ver >= (0, 4, 6))
        _native_aimdo_fastpath = bool(
            _recent_aimdo
            and _out_of_core_streaming
            and _runtime_backend in ('int8', 'int8-convrot-w4a4')
            and _runtime_quant_variant != 'w4a8'
        )
        _disable_dynamic_vbar_prefetch = (
            (
                _forced_streaming_mode or _out_of_core_streaming or (
                    _runtime_backend in ('int8', 'int8-convrot-w4a4')
                    and (_runtime_quant_variant == 'w4a8' or _int8_low_vram_streaming)
                )
            )
            and not _native_aimdo_fastpath
        )
        _lm_print(
            '[MiniMaxH3 LongMedia][0.3.77 NATIVE AIMDO FASTPATH] '
            f'aimdo={_aimdo_raw or "unknown"} kitchen={_kitchen_raw or "unknown"} '
            f'recent_aimdo={_recent_aimdo} native_int8_fastpath={_native_aimdo_fastpath} '
            f'out_of_core={_out_of_core_streaming} requested_mode={memory_mode}; '
            f'prefetch={"NATIVE" if _native_aimdo_fastpath else "GUARDED"}; H3 math=UNCHANGED',
            flush=True,
        )
        transformer_options['latentlab_disable_dynamic_vbar_prefetch'] = bool(_disable_dynamic_vbar_prefetch)
        if _runtime_backend in ('int8', 'int8-convrot-w4a4') or _forced_streaming_mode:
            transformer_options['prefetch_dynamic_vbars'] = not bool(_disable_dynamic_vbar_prefetch)
        if _forced_streaming_mode:
            _lm_print('[MiniMaxH3 LongMedia][0.3.52 OUT-OF-CORE] '
                f'memory_mode={memory_mode}: speculative prefetch disabled; demand residency + activation reserve active', flush=True)

        if _runtime_backend in ('int8', 'int8-convrot-w4a4'):
            if _runtime_quant_variant == 'w4a8':
                _residency_message = 'W4A8 detected: dynamic-VBAR prefetch DISABLED; AUTO MLP owns activation headroom'
            elif _native_aimdo_fastpath:
                _residency_message = (
                    f'native INT8 on {_device_vram_gb:.1f} GB GPU: recent AIMDO native DynamicVRAM/threaded prefetch ENABLED; '
                    'legacy LongMedia hard-gate bypassed for performance A/B'
                )
            elif _int8_low_vram_streaming:
                _residency_message = (
                    f'native INT8 on {_device_vram_gb:.1f} GB GPU: dynamic-VBAR prefetch DISABLED; '
                    'demand-loaded residency prevents speculative AIMDO copy OOM'
                )
            else:
                _residency_message = 'native Comfy dynamic-VBAR prefetch ENABLED; quantized-weight residency owned by Comfy'
            _lm_print(
                '[MiniMaxH3 LongMedia][V325 QUANT RESIDENCY] ' + _residency_message,
                flush=True,
            )
            if _runtime_quant_variant == 'w4a8':
                _lm_print(
                    '[MiniMaxH3 LongMedia][V30 W4A8 THROUGHPUT] AUTO MLP ceiling=8192; '
                    'target is fewer native quantized dispatches per H3 block',
                    flush=True,
                )
        if WrappersMP is not None:
            # Segment/presentation compatibility is independent of SOL and must
            # protect every H3 execution mode.
            wrappers = transformer_options.setdefault('wrappers', {})
            diffusion_model = wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})
            diffusion_model['MiniMaxH3LatentLabSegmentLayoutGuard'] = [
                _h3_segment_layout_guard_wrapper
            ]

            # The prefetch hard-gate is a memory policy, not an attention policy.
            # It must remain installed even when the user runs existing/Sage
            # attention; otherwise BaseModel._apply_model() re-enables dynamic
            # VBAR prefetch from current_patcher.is_dynamic() and ultra_low_vram
            # can still OOM before the first denoise step.
            # v0.3.64: defer runtime-prefetch wrapper attachment until the
            # residency state exists, so the wrapper can capture the real
            # ModelPatcher/state directly instead of relying on copied options.

            if str(sol_mode) in ('auto', 'sol', 'scheduled_sol'):
                apply_model = wrappers.setdefault(WrappersMP.APPLY_MODEL, {})
                apply_model['MiniMaxH3LatentLabSolSpan'] = [_h3_sol_span_wrapper]
        patches_replace = transformer_options.setdefault('patches_replace', {})
        dit = patches_replace.setdefault('dit', {})
        state = {
            'mode': 'token_chunked_mlp',
            'model_runtime_profile': runtime_profile,
            'model_runtime_backend': str(runtime_profile.get('backend', 'unknown')),
            'model_runtime_quant_variant': runtime_profile.get('quant_variant'),
            'model_runtime_policy': dict(runtime_policy),
            'memory_mode': str(memory_mode),
            'requested_memory_mode': str(requested_memory_mode),
            'adaptive_memory_governor_enabled': True,
            'adaptive_memory_zone': 'CALIBRATION_SAFE',
            'memory_policy_mode': str(memory_mode),
            'model_size_bytes': int(_model_size_b or 0),
            'gpu_size_bytes': int(_gpu_size_b or 0),
            'stock_transformer_math': True,
            'adaptive_memory_adjustments': 0,
            'ultra_stage_barrier_required': True,
            # V29 backend-aware throughput AUTO MLP controller. NVFP4 remains on the
            # proven fixed path; W4A8 adapts up to 8192 tokens from actual CUDA headroom.
            'auto_mlp_chunk_enabled': str(runtime_profile.get('quant_variant') or '').lower() == 'w4a8',
            'auto_mlp_chunk_last': None,
            'auto_mlp_chunk_changes': 0,
            'auto_mlp_chunk_safety_mb': 640,
            'auto_mlp_chunk_bytes_per_token': 96 * 1024,
            'int8_reusable_sol_storage': None,
            'int8_reusable_sol_storage_key': None,
            'v12_int8_sol_forward_generation': 0,
            'v12_int8_sol_forward_active': False,
            'v12_int8_sol_forward_release_count': 0,
            # V16 is a quality candidate, not a diagnostic build. Keep the
            # proven A/B helpers dormant so generation has no probe overhead.
            'v12b_linear_ab_done': {
                'qkv_proj': True,
                'out_proj': True,
                'fc1': True,
                'mlp_fc1_fc2': True,
            },
            'v13_sol_exact_ab_done': True,
            'v14_sol_exact_ab_done': True,
            'v15_tau_calibration_done': True,
            'v16_int8_quality_tau_announced': False,
            'v17_calibrated_offsets': set(),
            'v18_bf16_kv_reference_done': True,
            'v19_stage_ab_done': set(),
            'int8_block_mlp_weights_announced': False,
            'int8_cached_mlp_parity': 'unknown',
            'int8_cached_mlp_disabled_reason': None,
            'int8_semantic_dispatch_announced': False,
            # V321 native INT8: oversubscription-aware residency hysteresis.
            # Keep allocator cache during normal 20 GB-on-16 GB streaming; trim only
            # when physical free AND reclaimable effective headroom are both critical.
            'int8_residency_emergency_free_mb': 384,
            'int8_residency_emergency_effective_mb': 768,
            'int8_residency_min_cached_mb': 256,
            'int8_residency_guard_cooldown_blocks': 8,
            'int8_residency_guard_cooldown_left': 0,
            'int8_residency_last_effective_mb': 0.0,
            'int8_residency_emergency_trim_count': 0,
            'int8_sol_storage_free_floor_mb': 3072,
            'int8_sol_storage_emergency_free_mb': 2048,
            'int8_sol_storage_min_cached_mb': 1024,
            'int8_sol_storage_guard_cooldown_blocks': 4,
            'int8_sol_storage_guard_cooldown_left': 0,
            'int8_sol_storage_trim_count': 0,
            'requested_attention_mode': str(sol_mode),
            'sol_mode': str(sol_mode),
            'auto_attention_selected_mode': None,
            'auto_attention_reason': None,
            'auto_attention_announced': False,
            'last_sol_tau': 0.0,
            'active_block_index': -1,
            'sol_tau_start': float(sol_tau_start),
            'sol_tau_end': float(sol_tau_end),
            'sol_curve': str(sol_curve),
            'sol_min_tokens': int(sol_min_tokens),
            'sol_dense_percent': float(sol_dense_percent),
            'sol_sink_conditioning': str(sol_sink_conditioning),
            'sol_qkv_chunk_tokens': int(sol_qkv_chunk_tokens),
            'sol_out_proj_chunk_tokens': int(sol_out_proj_chunk_tokens),
            'vram_activation_reserve_mb': int(vram_activation_reserve_mb),
            'inter_block_vram_guard_mb': int(inter_block_vram_guard_mb),
            'inter_block_guard_cooldown_blocks': int(inter_block_guard_cooldown_blocks),
            'inter_block_guard_emergency_mb': int(inter_block_guard_emergency_mb),
            'inter_block_guard_emergency_cooldown_blocks': int(inter_block_guard_emergency_cooldown_blocks),
            'late_block_guard_start': int(late_block_guard_start),
            'late_block_guard_target_mb': int(late_block_guard_target_mb),
            'late_block_guard_min_cached_mb': int(late_block_guard_min_cached_mb),
            'step_boundary_cleanup_mb': int(step_boundary_cleanup_mb),
            'late_block_guard_trim_count': 0,
            'late_block_guard_reclaimed_mb': 0.0,
            'step_boundary_cleanup_count': 0,
            'step_boundary_cleanup_reclaimed_mb': 0.0,
            'inter_block_guard_calls': 0,
            'inter_block_last_trim_call': -1000000000,
            'inter_block_last_emergency_trim_call': -1000000000,
            'inter_block_cooldown_skip_count': 0,
            'inter_block_emergency_cooldown_skip_count': 0,
            'inter_block_emergency_trim_count': 0,
            'inter_block_trim_count': 0,
            'inter_block_reclaimed_mb': 0.0,
            'mlp_inplace_reuse': False,
            'activation_reserve': reserve_stats,
            'sol_out_proj_chunked_calls': 0,
            'sol_out_proj_max_chunks': 0,
            'sol_out_proj_announced': False,
            'sol_calls': 0,
            'sol_announced': False,
            'sol_fallbacks': [],
            'sol_sigma_hi': float(sol_sigma_hi),
            'sol_sigma_lo': float(sol_sigma_lo),
            'sol_geometry_tau_boost': 0.0,
            'chunk_tokens': int(chunk_tokens),
            # AUTO VRAM controller state.
            'auto_vram_controller_enabled': True,
            'auto_vram_controller_done': False,
            'auto_vram_controller_mode': None,
            'auto_vram_controller_probe': None,
            'auto_vram_controller_before': None,
            'auto_vram_controller_after': None,
            'inter_block_guard_hysteresis_mb': 1024,
            'inter_block_emergency_hysteresis_mb': 512,
            'inter_block_min_reclaim_mb': 256,
            'inter_block_effective_skip_count': 0,
            'inter_block_hysteresis_skip_count': 0,
            'inter_block_emergency_hyst_skip_count': 0,
            'inter_block_cooldown_skip_count': 0,
            'inter_block_emergency_cooldown_skip_count': 0,
            'inter_block_low_cache_skip_count': 0,
            'inter_block_normal_trim_count': 0,
            'inter_block_emergency_trim_count': 0,
            'late_guard_hysteresis_mb': 1024,
            'late_guard_cooldown_phases': 4,
            'late_guard_cooldown_left': 0,
            'late_guard_hysteresis_skip_count': 0,
            'late_guard_cooldown_skip_count': 0,
            'allocator_backend': _cuda_allocator_backend(),
            'max_blocks': int(max_blocks),
            'forward_count': 0,
            'first_forward_started': False,
            'first_forward_complete': False,
            'first_forward_started_at': None,
            'blocks': [],
            'stages': [],
            'worst_stage': None,
            'worst_stage_peak_allocated_mb': 0.0,
            'fallback_reason': None,
            'skipped_existing_patch_indices': [],
            'patched_block_indices': [],
            'last_patched_block_index': -1,
            'highest_block_peak_allocated_mb': 0.0,
            'highest_block_peak_reserved_mb': 0.0,
            'worst_block': 0,
            'worst_block_peak_allocated_mb': 0.0,
            'oom': False,
            'oom_block': None,
            'oom_message': None,
            'oom_stats': None,
            'mlp_chunked_calls': 0,
            'max_sequence_tokens': 0,
            'max_chunks_per_mlp': 0,
            'announced': False,
            'final_output_streaming_installed': False,
            'final_output_streaming_calls': 0,
            'final_output_streaming_announced': False,
            'final_output_first_profile_complete': False,
            'final_output_streaming_error': None,
            'v24_final_ab_done': set(),
            # V25 cleanup build: V24 final-layer forensic probes disabled.
            'v24_final_video_captured': True,
            'v24_final_audio_captured': True,
            'v25_native_quant_announced': False,
            # v0.3.62 AIMDO/VBAR residency governor. Keep a reference only in
            # runtime state; this never enters workflow serialization.
            'residency_model_patcher': getattr(wrapped, 'model_patcher', None),
            'vbar_promote_count': 0,
            'vbar_last_loaded_bytes': 0,
            'vbar_last_promote_free_mb': 0.0,
            'vbar_governor_skip_pressure': 0,
            'vbar_governor_skip_hysteresis': 0,
        }
        # v0.3.63 authoritative residency wiring. These are runtime-only object
        # references carried through shallow transformer_options copies.
        transformer_options['latentlab_h3_residency_state'] = state
        transformer_options['latentlab_h3_residency_patcher'] = getattr(wrapped, 'model_patcher', None)
        state['vbar_forward_count'] = 0
        state['vbar_first_forward_complete'] = False
        state['vbar_forward_promote_count'] = 0
        state['vbar_forward_skip_pressure'] = 0

        if WrappersMP is not None and _disable_dynamic_vbar_prefetch:
            import functools
            wrappers = transformer_options.setdefault('wrappers', {})
            diffusion_model = wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})
            _bound_runtime_wrapper = functools.partial(
                _h3_runtime_prefetch_wrapper,
                _bound_residency_state=state,
                _bound_residency_patcher=getattr(wrapped, 'model_patcher', None),
            )
            diffusion_model['MiniMaxH3LatentLabRuntimePrefetch'] = [_bound_runtime_wrapper]
            _lm_print(
                '[MiniMaxH3 LongMedia][0.3.64 VBAR BIND] runtime wrapper bound directly to active ModelPatcher/state',
                flush=True,
            )

        if _runtime_backend in ('int8', 'int8-convrot-w4a4') and not state.get('v28_native_quant_announced'):
            _lm_print(
                '[MiniMaxH3 LongMedia][V31 NATIVE QUANT] INT8/W4A8 math delegated to stock Comfy quantized modules; LongMedia owns memory/chunking/SOL only',
                flush=True,
            )
            state['v28_native_quant_announced'] = True

        for i in range(int(max_blocks)):
            key = ('double_block', i)
            if key in dit:
                state['skipped_existing_patch_indices'].append(i)
                continue
            dit[key] = _H3MLPChunkPatch(i, state, chunk_tokens=int(chunk_tokens))
            state['patched_block_indices'].append(i)
        if state['patched_block_indices']:
            state['last_patched_block_index'] = max(
                state['patched_block_indices']
            )
            # MiniMax H3 DiT has 50 transformer blocks, indexed 0..49.
            # max_blocks is only the patch-install scan ceiling (default 128) and
            # must not be used as the architectural block count for diagnostics.
            _v22_h3_last = 49
            state['v21_stage_ab_targets'] = (0, _v22_h3_last // 2, _v22_h3_last)
            state['v21_stage_ab_armed'] = False
            state['v21_stage_ab_completed'] = False
            state['v21_stage_ab_generation'] = 0
            _lm_print(
                '[MiniMaxH3 LongMedia][V22 STAGE A/B TARGET] '
                f'configured targets={list(state["v21_stage_ab_targets"])}',
                flush=True,
            )
        if state['skipped_existing_patch_indices']:
            _lm_print(
                '[MiniMaxH3 LongMedia] Low-VRAM MLP skipped existing DiT patches at indices: '
                + ','.join(map(str, state['skipped_existing_patch_indices'])),
                flush=True,
            )
        if hasattr(wrapped, 'model_patcher'):
            _install_h3_final_output_streaming(wrapped.model_patcher, state, chunk_tokens=int(chunk_tokens))
        return (wrapped, state)



def _h3_vbar_residency_step_governor(block_state, snapshot, step=None):
    """v0.3.62: safely reopen AIMDO's residency watermark between denoise steps.

    ModelVBAR intentionally lowers a watermark after VRAM pressure; once lowered,
    later weights stop faulting resident even if several GB become free.  At a
    completed step boundary there are no live per-block temporaries, so this is
    the safest point to let the *next* forward attempt more persistent residency.
    We do not pin or preload weights ourselves and never touch H3 math.
    """
    if not isinstance(block_state, dict) or not snapshot:
        return
    patcher = block_state.get('residency_model_patcher')
    if patcher is None:
        return
    mode = str(block_state.get('memory_policy_mode', block_state.get('memory_mode', 'normal')))
    policy = {
        'normal': (1536.0, 2816.0, 512.0),
        'low_vram': (2304.0, 3584.0, 768.0),
        'ultra_low_vram': (3072.0, 4608.0, 1024.0),
    }
    hard_mb, promote_mb, hyst_mb = policy.get(mode, policy['low_vram'])
    free_mb = float(snapshot.get('driver_free', 0)) / (1024.0 ** 2)
    # Never reopen the watermark near the hard floor. Promotion requires a
    # generous one-step envelope above it, with per-mode conservatism.
    if free_mb < promote_mb:
        block_state['vbar_governor_skip_pressure'] = int(block_state.get('vbar_governor_skip_pressure', 0)) + 1
        return
    try:
        vbar_get = getattr(patcher, '_vbar_get', None)
        if not callable(vbar_get):
            return
        vbar = vbar_get(create=False)
        if vbar is None:
            return
        loaded_before = int(vbar.loaded_size()) if hasattr(vbar, 'loaded_size') else 0
        last_loaded = int(block_state.get('vbar_last_loaded_bytes', 0) or 0)
        last_free = float(block_state.get('vbar_last_promote_free_mb', 0.0) or 0.0)
        # Avoid resetting the watermark every step after residency has converged.
        # Reopen only if there is materially more headroom or residency is still
        # growing/near-zero. This gives AIMDO hysteresis instead of oscillation.
        residency_stalled = loaded_before <= max(last_loaded + 64 * 1024 * 1024, 256 * 1024 * 1024)
        more_headroom = free_mb >= last_free + hyst_mb
        if block_state.get('vbar_promote_count', 0) > 0 and not residency_stalled and not more_headroom:
            block_state['vbar_governor_skip_hysteresis'] = int(block_state.get('vbar_governor_skip_hysteresis', 0)) + 1
            block_state['vbar_last_loaded_bytes'] = loaded_before
            return
        vbar.prioritize()
        block_state['vbar_promote_count'] = int(block_state.get('vbar_promote_count', 0)) + 1
        block_state['vbar_last_loaded_bytes'] = loaded_before
        block_state['vbar_last_promote_free_mb'] = free_mb
        block_state['vbar_hard_floor_mb'] = hard_mb
        block_state['vbar_promote_floor_mb'] = promote_mb
        _lm_print(
            '[MiniMaxH3 LongMedia][0.3.62 VBAR RESIDENCY] '
            f'step={step} mode={mode} free={free_mb:.0f}MB '
            f'loaded={loaded_before/(1024.0**2):.0f}MB; watermark reopened for next forward',
            flush=True,
        )
    except Exception as exc:
        block_state['vbar_governor_error'] = f'{type(exc).__name__}: {exc}'
        if not block_state.get('vbar_governor_error_announced'):
            block_state['vbar_governor_error_announced'] = True
            _lm_print('[MiniMaxH3 LongMedia][0.3.62 VBAR RESIDENCY] disabled: ' + block_state['vbar_governor_error'], flush=True)

class _FirstStepMemoryProfilerSampler:
    """Transparent SAMPLER proxy that profiles allocator activity from before step 1."""

    def __init__(self, inner_sampler, state):
        self.inner_sampler = inner_sampler
        self.state = state

    def __getattr__(self, name):
        return getattr(self.inner_sampler, name)

    def sample(self, model_wrap, sigmas, extra_args, callback, noise,
               latent_image=None, denoise_mask=None, disable_pbar=False):
        state = self.state
        if not torch.cuda.is_available():
            return self.inner_sampler.sample(
                model_wrap, sigmas, extra_args, callback, noise,
                latent_image, denoise_mask, disable_pbar,
            )

        device = torch.cuda.current_device()
        # TEST build: skip profiling-only sampler-entry sync and allocator history.
        before = _cuda_memory_snapshot()
        state['before_sampling'] = {k + '_mb': _mb(v) for k, v in before.items()} if before else None
        enabled, history_error = False, 'disabled in TEST remove-deep-profiling build'
        state['history_enabled'] = False
        state['history_error'] = history_error
        first_callback_seen = False

        def profiled_callback(*args, **kwargs):
            nonlocal first_callback_seen
            # TEST build: no profiling-only CUDA synchronization at step boundary.
            callback_arrival = time.perf_counter()
            snapshot = _cuda_memory_snapshot()
            result = callback(*args, **kwargs) if callback is not None else None
            block_state = state.get('block_trace_state')
            # Optional hard cleanup at a completed denoise-step boundary.  This
            # runs after the sampler callback has consumed the step result, so it
            # cannot invalidate live denoised tensors; it only returns dead CUDA
            # allocator pages before AIMDO prepares the next forward.
            if isinstance(block_state, dict):
                target_mb = max(0, int(block_state.get('step_boundary_cleanup_mb', 0) or 0))
                if target_mb > 0 and snapshot:
                    free_mb0 = snapshot['driver_free'] / (1024.0 ** 2)
                    cached_mb0 = snapshot['cached'] / (1024.0 ** 2)
                    if free_mb0 < target_mb and cached_mb0 >= 256.0:
                        _soft_empty_cuda_cache()
                        cleaned = _cuda_memory_snapshot()
                        if cleaned:
                            block_state['step_boundary_cleanup_count'] = int(block_state.get('step_boundary_cleanup_count', 0)) + 1
                            reclaimed = max(0, int(cleaned['driver_free']) - int(snapshot['driver_free']))
                            block_state['step_boundary_cleanup_reclaimed_mb'] = round(
                                float(block_state.get('step_boundary_cleanup_reclaimed_mb', 0.0)) + reclaimed / (1024.0 ** 2), 1
                            )
                            _lm_print(
                                '[MiniMaxH3 LongMedia] Step-boundary hard cleanup: '
                                f'free {free_mb0:.1f} -> {cleaned["driver_free"]/(1024.0**2):.1f} MB, '
                                f'cached {cached_mb0:.1f} -> {cleaned["cached"]/(1024.0**2):.1f} MB, target={target_mb}',
                                flush=True,
                            )
                            snapshot = cleaned
            step = None
            total_steps = None
            if len(args) >= 1:
                try:
                    step = int(args[0]) + 1
                except Exception:
                    pass
            if len(args) >= 4:
                try:
                    total_steps = int(args[3])
                except Exception:
                    pass
            # v0.3.63: VBAR promotion moved to the authoritative diffusion-forward
            # boundary. Keep callback profiling only; never mutate residency here.
            entry = {
                'step': step,
                'total_steps': total_steps,
                'allocated_mb': _mb(snapshot['allocated']) if snapshot else None,
                'reserved_mb': _mb(snapshot['reserved']) if snapshot else None,
                'cached_mb': _mb(snapshot['cached']) if snapshot else None,
                'driver_free_mb': _mb(snapshot['driver_free']) if snapshot else None,
                'peak_allocated_mb': _mb(torch.cuda.max_memory_allocated(device)),
                'peak_reserved_mb': _mb(torch.cuda.max_memory_reserved(device)),
            }
            if isinstance(block_state, dict) and block_state.get('blocks'):
                # Per-block tracing resets CUDA peak counters at every block boundary.
                # Replace the sampler-level peak with the maximum captured across blocks.
                entry['peak_allocated_mb'] = max(
                    float(entry['peak_allocated_mb']),
                    float(block_state.get('highest_block_peak_allocated_mb') or 0.0),
                )
                entry['peak_reserved_mb'] = max(
                    float(entry['peak_reserved_mb']),
                    float(block_state.get('highest_block_peak_reserved_mb') or 0.0),
                )
                entry['peak_source'] = 'max_of_h3_block_trace'
            else:
                entry['peak_source'] = 'torch_cuda_global'
            state['steps'].append(entry)
            if isinstance(block_state, dict):
                current = block_state.get('step_boundary_current_forward')
                if isinstance(current, dict):
                    forward_ms = max(0.0, (callback_arrival - float(current.get('block0_started_perf', callback_arrival))) * 1000.0)
                    forward_entry = {
                        'step': step,
                        'forward': current.get('forward'),
                        'block0_to_step_end_ms': round(forward_ms, 1),
                        'end_allocated_mb': entry['allocated_mb'],
                        'end_reserved_mb': entry['reserved_mb'],
                        'end_driver_free_mb': entry['driver_free_mb'],
                    }
                    block_state.setdefault('step_boundary_forward_times', []).append(forward_entry)
                    _lm_print(
                        '[MiniMaxH3 LongMedia] Step compute profile: '
                        f"step {step}, block0 -> callback {forward_entry['block0_to_step_end_ms']:.1f} ms; "
                        f"end alloc/res/free {entry['allocated_mb']:.1f}/{entry['reserved_mb']:.1f}/{entry['driver_free_mb']:.1f} MB",
                        flush=True,
                    )
                block_state['step_boundary_pending_callback'] = {
                    'step': step,
                    'time_perf': callback_arrival,
                    'allocated_mb': entry['allocated_mb'],
                    'reserved_mb': entry['reserved_mb'],
                    'driver_free_mb': entry['driver_free_mb'],
                }
            if not first_callback_seen:
                first_callback_seen = True
                path, error = _dump_cuda_memory_snapshot('after_step1') if enabled else (None, history_error)
                state['first_step_snapshot'] = path
                state['first_step_snapshot_error'] = error
                _lm_print(
                    '[MiniMaxH3 LongMedia] First-step memory profile: '
                    f"allocated {entry['allocated_mb']:.1f} MB, reserved {entry['reserved_mb']:.1f} MB, "
                    f"peak allocated {entry['peak_allocated_mb']:.1f} MB, peak reserved {entry['peak_reserved_mb']:.1f} MB, "
                    f"driver free {entry['driver_free_mb']:.1f} MB",
                    flush=True,
                )
                if isinstance(block_state, dict) and block_state.get('blocks'):
                    block_state['first_forward_complete'] = True
                    _lm_print(
                        '[MiniMaxH3 LongMedia] H3 block trace summary: '
                        f"{len(block_state['blocks'])} blocks, worst block {block_state.get('worst_block')}, "
                        f"peak allocated {block_state.get('highest_block_peak_allocated_mb', 0.0):.1f} MB, "
                        f"peak reserved {block_state.get('highest_block_peak_reserved_mb', 0.0):.1f} MB",
                        flush=True,
                    )
                if path:
                    _lm_print(f'[MiniMaxH3 LongMedia] First-step allocator snapshot: {path}', flush=True)
                elif error:
                    _lm_print(f'[MiniMaxH3 LongMedia] Snapshot unavailable: {error}', flush=True)
            return result

        try:
            output = self.inner_sampler.sample(
                model_wrap, sigmas, extra_args, profiled_callback, noise,
                latent_image, denoise_mask, disable_pbar,
            )
            state['completed'] = True
            return output
        except Exception as exc:
            message = str(exc).lower()
            is_oom = isinstance(exc, getattr(torch, 'OutOfMemoryError', RuntimeError)) or 'out of memory' in message
            if is_oom:
                state['oom'] = True
                state['oom_message'] = str(exc)[:2000]
                snapshot = _cuda_memory_snapshot()
                if snapshot:
                    state['oom_counters'] = {k + '_mb': _mb(v) for k, v in snapshot.items()}
                state['oom_peak_allocated_mb'] = _mb(torch.cuda.max_memory_allocated(device))
                state['oom_peak_reserved_mb'] = _mb(torch.cuda.max_memory_reserved(device))
                path, error = _dump_cuda_memory_snapshot('oom') if enabled else (None, history_error)
                state['oom_snapshot'] = path
                state['oom_snapshot_error'] = error
                _lm_print(
                    '[MiniMaxH3 LongMedia] CUDA OOM captured by first-step profiler. '
                    f"peak allocated {state['oom_peak_allocated_mb']:.1f} MB, "
                    f"peak reserved {state['oom_peak_reserved_mb']:.1f} MB",
                    flush=True,
                )
                if path:
                    _lm_print(f'[MiniMaxH3 LongMedia] OOM allocator snapshot: {path}', flush=True)
                elif error:
                    _lm_print(f'[MiniMaxH3 LongMedia] OOM snapshot unavailable: {error}', flush=True)
            raise
        finally:
            _stop_cuda_memory_history()


class MiniMaxH3LatentLabFirstStepMemoryProfiler:
    """Internal allocator profiler used to diagnose first-step H3 VRAM peaks."""

    DESCRIPTION = 'Internal first-step CUDA allocator profiler for Long Media sampling.'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'sampler': ('SAMPLER',),
                'max_history_entries': ('INT', {'default': 20000, 'min': 1000, 'max': 200000, 'step': 1000}),
            },
            'optional': {
                'block_trace_state': ('H3_BLOCK_MEMORY_TRACE_STATE',),
            },
        }

    RETURN_TYPES = ('SAMPLER', 'H3_MEMORY_PROFILE_STATE')
    RETURN_NAMES = ('sampler', 'profile_state')
    FUNCTION = 'wrap'
    CATEGORY = CATEGORY_LONGMEDIA

    def wrap(self, sampler, max_history_entries=20000, block_trace_state=None):
        state = {
            'max_history_entries': int(max_history_entries),
            'history_enabled': False,
            'history_error': None,
            'before_sampling': None,
            'steps': [],
            'first_step_snapshot': None,
            'first_step_snapshot_error': None,
            'oom': False,
            'oom_message': None,
            'oom_counters': None,
            'oom_snapshot': None,
            'oom_snapshot_error': None,
            'completed': False,
            'block_trace_state': block_trace_state,
        }
        return (_FirstStepMemoryProfilerSampler(sampler, state), state)


class _VRAMPressureGuardSampler:
    """Transparent SAMPLER proxy that flushes CUDA cache only under real pressure."""

    def __init__(self, inner_sampler, state):
        self.inner_sampler = inner_sampler
        self.state = state

    def __getattr__(self, name):
        return getattr(self.inner_sampler, name)

    def sample(self, model_wrap, sigmas, extra_args, callback, noise,
               latent_image=None, denoise_mask=None, disable_pbar=False):
        state = self.state

        def guarded_callback(*args, **kwargs):
            result = callback(*args, **kwargs) if callback is not None else None
            state['checks'] += 1
            if not torch.cuda.is_available() or state['flushes'] >= state['max_flushes']:
                return result

            snapshot = _cuda_memory_snapshot()
            if snapshot is None:
                return result

            if (
                snapshot['driver_free'] < state['free_threshold_bytes']
                and snapshot['cached'] > state['cache_threshold_bytes']
            ):
                step = None
                total_steps = None
                if len(args) >= 1:
                    try:
                        step = int(args[0]) + 1
                    except Exception:
                        step = None
                if len(args) >= 4:
                    try:
                        total_steps = int(args[3])
                    except Exception:
                        total_steps = None

                before = snapshot
                _soft_empty_cuda_cache()
                after = _cuda_memory_snapshot()
                state['flushes'] += 1
                event = {
                    'step': step,
                    'total_steps': total_steps,
                    'cached_before_mb': _mb(before['cached']),
                    'cached_after_mb': _mb(after['cached']),
                    'reserved_before_mb': _mb(before['reserved']),
                    'reserved_after_mb': _mb(after['reserved']),
                    'driver_free_before_mb': _mb(before['driver_free']),
                    'driver_free_after_mb': _mb(after['driver_free']),
                }
                state['events'].append(event)
                step_text = (
                    f"step {step}/{total_steps}"
                    if step is not None and total_steps is not None
                    else 'sampling step'
                )
                _lm_print(
                    '[MiniMaxH3 LongMedia] VRAM pressure guard: '
                    f"{step_text}, cached {_mb(before['cached']):.1f} -> {_mb(after['cached']):.1f} MB, "
                    f"driver free {_mb(before['driver_free']):.1f} -> {_mb(after['driver_free']):.1f} MB",
                    flush=True,
                )
            return result

        return self.inner_sampler.sample(
            model_wrap, sigmas, extra_args, guarded_callback, noise,
            latent_image, denoise_mask, disable_pbar,
        )


class MiniMaxH3LatentLabVRAMPressureGuard:
    """Internal SAMPLER wrapper for adaptive intra-sampling cache cleanup."""

    DESCRIPTION = 'Internal adaptive VRAM pressure guard for Long Media sampling.'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'sampler': ('SAMPLER',),
                'free_threshold_mb': ('INT', {'default': 768, 'min': 128, 'max': 8192, 'step': 64}),
                'cache_threshold_mb': ('INT', {'default': 4096, 'min': 256, 'max': 32768, 'step': 256}),
                'max_flushes': ('INT', {'default': 2, 'min': 0, 'max': 16, 'step': 1}),
            }
        }

    RETURN_TYPES = ('SAMPLER', 'H3_VRAM_GUARD_STATE')
    RETURN_NAMES = ('sampler', 'guard_state')
    FUNCTION = 'wrap'
    CATEGORY = CATEGORY_LONGMEDIA

    def wrap(self, sampler, free_threshold_mb=768, cache_threshold_mb=4096, max_flushes=2):
        state = {
            'checks': 0,
            'flushes': 0,
            'max_flushes': int(max_flushes),
            'free_threshold_mb': int(free_threshold_mb),
            'cache_threshold_mb': int(cache_threshold_mb),
            'free_threshold_bytes': int(free_threshold_mb) * 1024 * 1024,
            'cache_threshold_bytes': int(cache_threshold_mb) * 1024 * 1024,
            'events': [],
        }
        return (_VRAMPressureGuardSampler(sampler, state), state)


class MiniMaxH3LatentLabVRAMCacheCleanup:
    """Internal passthrough used after sampling to measure and release CUDA cache."""

    DESCRIPTION = 'Internal post-sampling CUDA cache cleanup and diagnostics.'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'latent': ('LATENT',),
                'sampler_report': ('STRING', {'default': '', 'multiline': True}),
            },
            'optional': {
                'vram_guard_state': ('H3_VRAM_GUARD_STATE',),
                'memory_profile_state': ('H3_MEMORY_PROFILE_STATE',),
                'block_trace_state': ('H3_BLOCK_MEMORY_TRACE_STATE',),
            },
        }

    RETURN_TYPES = ('LATENT', 'STRING')
    RETURN_NAMES = ('latent', 'report')
    FUNCTION = 'cleanup'
    CATEGORY = CATEGORY_LONGMEDIA

    def cleanup(self, latent, sampler_report, vram_guard_state=None, memory_profile_state=None, block_trace_state=None):
        if not torch.cuda.is_available():
            return (latent, sampler_report)

        # The dependency on ``latent`` guarantees the sampler has completed.
        # Synchronize once so the before/after counters describe a stable point.
        torch.cuda.synchronize()
        before = _cuda_memory_snapshot()
        _soft_empty_cuda_cache()
        torch.cuda.synchronize()
        after = _cuda_memory_snapshot()

        released_reserved = max(0, before['reserved'] - after['reserved'])
        released_cached = max(0, before['cached'] - after['cached'])
        cleanup_data = {
            'allocated_before_mb': _mb(before['allocated']),
            'allocated_after_mb': _mb(after['allocated']),
            'reserved_before_mb': _mb(before['reserved']),
            'reserved_after_mb': _mb(after['reserved']),
            'cached_before_mb': _mb(before['cached']),
            'cached_after_mb': _mb(after['cached']),
            'released_reserved_mb': _mb(released_reserved),
            'released_cached_mb': _mb(released_cached),
            'driver_free_before_mb': _mb(before['driver_free']),
            'driver_free_after_mb': _mb(after['driver_free']),
        }

        # TEST cleanup fix: removed invalid AUTO summary that referenced
        # an out-of-scope local `state`. Core post-sampling cleanup is unchanged.

        _lm_print(
            '[MiniMaxH3 LongMedia] Post-sampling VRAM cleanup: '
            f"cached {_mb(before['cached']):.1f} -> {_mb(after['cached']):.1f} MB, "
            f"reserved {_mb(before['reserved']):.1f} -> {_mb(after['reserved']):.1f} MB, "
            f"driver free {_mb(before['driver_free']):.1f} -> {_mb(after['driver_free']):.1f} MB",
            flush=True,
        )

        try:
            report_data = json.loads(sampler_report) if sampler_report else {}
            if not isinstance(report_data, dict):
                report_data = {'sampler_report': sampler_report}
        except Exception:
            report_data = {'sampler_report': sampler_report}
        report_data['post_sampling_vram_cleanup'] = cleanup_data
        if isinstance(memory_profile_state, dict):
            report_data['first_step_memory_profile'] = {
                'max_history_entries': memory_profile_state.get('max_history_entries'),
                'history_enabled': memory_profile_state.get('history_enabled'),
                'history_error': memory_profile_state.get('history_error'),
                'before_sampling': memory_profile_state.get('before_sampling'),
                'steps': list(memory_profile_state.get('steps', [])),
                'first_step_snapshot': memory_profile_state.get('first_step_snapshot'),
                'first_step_snapshot_error': memory_profile_state.get('first_step_snapshot_error'),
                'oom': memory_profile_state.get('oom', False),
                'oom_message': memory_profile_state.get('oom_message'),
                'oom_counters': memory_profile_state.get('oom_counters'),
                'oom_peak_allocated_mb': memory_profile_state.get('oom_peak_allocated_mb'),
                'oom_peak_reserved_mb': memory_profile_state.get('oom_peak_reserved_mb'),
                'oom_snapshot': memory_profile_state.get('oom_snapshot'),
                'oom_snapshot_error': memory_profile_state.get('oom_snapshot_error'),
                'completed': memory_profile_state.get('completed', False),
            }
        if isinstance(block_trace_state, dict):
            report_data['h3_block_memory_trace'] = {
                'allocator_backend': block_trace_state.get('allocator_backend'),
                'pre_block0': block_trace_state.get('pre_block0'),
                'block_count_traced': len(block_trace_state.get('blocks', [])),
                'blocks': list(block_trace_state.get('blocks', [])),
                'stages': list(block_trace_state.get('stages', [])),
                'worst_stage': block_trace_state.get('worst_stage'),
                'worst_stage_peak_allocated_mb': block_trace_state.get('worst_stage_peak_allocated_mb'),
                'fallback_reason': block_trace_state.get('fallback_reason'),
                'highest_block_peak_allocated_mb': block_trace_state.get('highest_block_peak_allocated_mb'),
                'highest_block_peak_reserved_mb': block_trace_state.get('highest_block_peak_reserved_mb'),
                'worst_block': block_trace_state.get('worst_block'),
                'worst_block_peak_allocated_mb': block_trace_state.get('worst_block_peak_allocated_mb'),
                'skipped_existing_patch_indices': list(block_trace_state.get('skipped_existing_patch_indices', [])),
                'oom': block_trace_state.get('oom', False),
                'oom_block': block_trace_state.get('oom_block'),
                'oom_message': block_trace_state.get('oom_message'),
                'oom_stats': block_trace_state.get('oom_stats'),
                'activation_reserve': block_trace_state.get('activation_reserve'),
                'vram_activation_reserve_mb': block_trace_state.get('vram_activation_reserve_mb'),
                'step_boundary_transitions': list(block_trace_state.get('step_boundary_transitions', [])),
                'step_boundary_forward_times': list(block_trace_state.get('step_boundary_forward_times', [])),
            }
        if isinstance(vram_guard_state, dict):
            report_data['intra_sampling_vram_guard'] = {
                'free_threshold_mb': vram_guard_state.get('free_threshold_mb'),
                'cache_threshold_mb': vram_guard_state.get('cache_threshold_mb'),
                'max_flushes': vram_guard_state.get('max_flushes'),
                'checks': vram_guard_state.get('checks', 0),
                'flushes': vram_guard_state.get('flushes', 0),
                'events': list(vram_guard_state.get('events', [])),
            }
        return (latent, json.dumps(report_data, indent=2))


def _match_frames_color_to_reference(images: torch.Tensor, reference_index: int, strength: float) -> torch.Tensor:
    """Nudge every frame's per-channel color statistics toward one reference frame.

    Blends each frame between its own colors (strength=0) and colors rescaled to
    match the reference frame's per-channel mean/std (strength=1). Useful when one
    frame is pinned exactly to a source/reference image but the rest of the clip has
    drifted slightly in color, which shows up as a visible jump at a loop seam.
    images is [T, H, W, C] in the 0..1 range. The reference frame itself is left
    untouched.
    """
    strength = float(strength)
    if strength <= 0.0 or images.shape[0] <= 1:
        return images
    strength = min(1.0, strength)
    reference = images[reference_index]
    ref_mean = reference.mean(dim=(0, 1), keepdim=True)
    ref_std = reference.std(dim=(0, 1), keepdim=True).clamp_min(1e-5)
    frame_mean = images.mean(dim=(1, 2), keepdim=True)
    frame_std = images.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
    normalized = (images - frame_mean) / frame_std * ref_std + ref_mean
    matched = torch.lerp(images, normalized.clamp(0.0, 1.0), strength)
    matched[reference_index] = images[reference_index]
    return matched


def _blend_leading_frames_to_reference(
    images: torch.Tensor, reference: torch.Tensor, n_frames: int
) -> torch.Tensor:
    """Cross-fade the first n_frames of a decoded clip toward a reference frame.

    Frame 0 is a full blend toward the reference (still not pixel-identical unless
    n_frames == 1 with an implicit weight of 1.0 handled by the caller), tapering
    linearly to 0 by frame n_frames-1. Cheaper than latent_inject — pure
    post-decode compositing, no extra sampling cost — but the reference frame
    itself is never pixel-perfect in the output. reference is [H, W, C] in 0..1.
    """
    n_frames = max(1, min(int(n_frames), images.shape[0]))
    weights = torch.linspace(1.0, 0.0, n_frames + 1, device=images.device)[:n_frames]
    reference = reference.to(images.dtype).to(images.device)
    for i in range(n_frames):
        images[i] = torch.lerp(images[i], reference, weights[i])
    return images


def _mix_audio_tracks(audio_list, total_duration=None):
    """Mix multiple audio dicts into one, padding channels and duration."""
    if not audio_list:
        return None
    if len(audio_list) == 1:
        return audio_list[0]
    sample_rate = audio_list[0]['sample_rate']
    max_channels = max(a['waveform'][:1].shape[1] for a in audio_list)
    if total_duration is not None:
        max_samples = round(total_duration * sample_rate)
    else:
        max_samples = max(a['waveform'][:1].shape[-1] for a in audio_list)
    mixed = torch.zeros(1, max_channels, max_samples)
    for audio in audio_list:
        wf = audio['waveform'][:1]
        if wf.shape[1] < max_channels:
            wf = wf.expand(1, max_channels, -1).clone()
        if wf.shape[-1] < max_samples:
            wf = torch.nn.functional.pad(wf, (0, max_samples - wf.shape[-1]))
        mixed = mixed + wf[:, :max_channels, :max_samples].to(mixed)
    return {'waveform': mixed, 'sample_rate': sample_rate}


def _normalize_decoded_audio(decoded, sample_rate, target_samples=None):
    """Normalize Audio VAE decode output to ComfyUI AUDIO: waveform [B,C,L].

    MiniMax H3 Audio VAE encode consumes [B,L,C], and some decode implementations
    return that same layout. Passing [B,L,C] straight to ComfyUI makes L look like
    the channel count (e.g. 165600 channels), which later explodes in ffmpeg.
    """
    if isinstance(decoded, dict):
        waveform = decoded.get('waveform')
        sample_rate = int(decoded.get('sample_rate', sample_rate))
        if waveform is None:
            raise ValueError('Audio VAE decode returned an AUDIO dict without waveform.')
    else:
        waveform = decoded

    if not torch.is_tensor(waveform):
        raise ValueError(f'Audio VAE decode returned unsupported type: {type(waveform)!r}.')

    if waveform.ndim == 2:
        # [L,C] or [C,L]
        if waveform.shape[-1] <= 8 and waveform.shape[0] > waveform.shape[-1]:
            waveform = waveform.transpose(0, 1)
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 3:
        # Prefer [B,C,L]. H3 is stereo, so a tiny last dimension strongly means [B,L,C].
        if waveform.shape[-1] <= 8 and waveform.shape[1] > waveform.shape[-1]:
            waveform = waveform.movedim(-1, 1)
    else:
        raise ValueError(
            f'Audio VAE decode must return [B,C,L] or [B,L,C], got {tuple(waveform.shape)}.'
        )

    if waveform.shape[0] != 1:
        waveform = waveform[:1]
    # H3 audio is stereo. Do not ever let a time axis leak into the channel axis.
    if waveform.shape[1] > 2:
        if waveform.shape[-1] <= 2:
            waveform = waveform.movedim(-1, 1)
        else:
            raise ValueError(
                f'Decoded H3 audio has {waveform.shape[1]} channels; expected mono/stereo. '
                f'Raw shape: {tuple(waveform.shape)}.'
            )
    if target_samples is not None:
        target_samples = max(1, int(target_samples))
        if waveform.shape[-1] > target_samples:
            waveform = waveform[..., :target_samples]
        elif waveform.shape[-1] < target_samples:
            waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.shape[-1]))

    return {'waveform': waveform.contiguous(), 'sample_rate': int(sample_rate)}


def _slice_source_audio_for_segment(source_audio, start_frame, length_frames):
    """Slice source audio using H3 audio-latent-aware sample counting."""
    waveform = source_audio['waveform'][:1]
    sample_rate = int(source_audio['sample_rate'])
    audio_t = audio_latent_t(length_frames)
    target_samples = round(audio_t / AUDIO_LATENT_FPS * sample_rate)
    start_sample = math.floor(start_frame / FPS * sample_rate)
    available = waveform[..., start_sample:start_sample + target_samples]
    if available.shape[-1] < target_samples:
        available = torch.nn.functional.pad(available, (0, target_samples - available.shape[-1]))
    return available, target_samples


def _segment_timeline_contract(plan, segment_index):
    """Return the canonical global/local timeline for one LongMedia pass.

    ``segment_starts`` is the global origin of the *context window*. Continuation
    passes contain ``overlap_frames`` of hidden preroll before their user-visible
    output. Any full local conditioning stream (audio/video reference) must start
    at context_start so local t=0 stays aligned with the inherited latent overlap.
    New visible media begins at visible_start/local_visible_offset.
    """
    idx = int(segment_index)
    context_start = int(plan.segment_starts[idx])
    overlap = int(getattr(plan, 'overlap_frames', 0) or 0) if idx > 0 else 0
    length = int(plan.segment_lengths[idx])
    visible_start = context_start + overlap
    visible_frames = max(1, length - overlap)
    return {
        'segment_index': idx,
        'context_start': context_start,
        'visible_start': visible_start,
        'local_visible_offset': overlap,
        'length_frames': length,
        'visible_frames': visible_frames,
        'visible_end': visible_start + visible_frames,
    }


def _visible_segment_start_frame(plan, segment_index):
    """Compatibility helper for code that only needs the visible global origin."""
    return int(_segment_timeline_contract(plan, segment_index)['visible_start'])


def _build_lipsync_prompt(prompt, plan, has_image, has_audio):
    """Build prompt for automatic lip sync mode."""
    parts = [prompt]
    if has_image:
        parts.append('<Picture 1>')
    if has_audio:
        parts.append('<Audio 1>')
    parts.append(
        'Focus on natural mouth movement and lip synchronization with the audio.'
    )
    if plan.passes > 1:
        parts.append(
            'Maintain a single continuous uninterrupted shot. '
            'No cuts. No scene reset. Consistent character and lighting throughout.'
        )
    return '\n'.join(parts)


_V57_SEGMENT_EVENT_RE = re.compile(
    r'^\s*(?P<sec>\d+(?:\.\d+)?)\s*(?::|sec\s*:|sec:|s\s*:|s:)\s*(?P<body>.+?)\s*$',
    re.IGNORECASE,
)


def _conditioning_meta(entry):
    """Return metadata dict for both canonical [tensor, meta] and legacy dict entries."""
    if isinstance(entry, dict):
        return entry
    if isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], dict):
        return entry[1]
    return None


def _v57_format_local_time(seconds_value):
    seconds_value = max(0.0, float(seconds_value))
    rounded = int(round(seconds_value))
    if abs(seconds_value - rounded) < 1e-6:
        return f'{rounded:02d} sec'
    return f'{seconds_value:.2f} sec'


def _v62_explicit_prompt_sections(base_prompt):
    """Split an author-written prompt into shot 0 + explicit continuation sections.

    A line beginning with ``Continue directly from the preceding video`` starts a
    new local-time section.  This lets users write ``00 sec`` / ``02 sec`` inside
    the continuation without those values being mistaken for global movie time.
    """
    text = str(base_prompt or '').strip()
    if not text:
        return []
    marker_re = re.compile(r'(?im)^(?=\s*continue\s+directly\s+from\s+the\s+preceding\s+video(?:\s+scene|\s+segment)?\b)')
    starts = [m.start() for m in marker_re.finditer(text)]
    if not starts:
        return [text]
    chunks = []
    first = text[:starts[0]].strip()
    if first:
        chunks.append(first)
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _v57_build_segment_prompt(base_prompt, plan, segment_index):
    """Create one pass-local prompt, including pass 0, on the global timeline."""
    selected, policy = _policy_build_segment_prompt(
        base_prompt,
        segment_index=int(segment_index),
        segment_starts=tuple(plan.segment_starts),
        segment_lengths=tuple(plan.segment_lengths),
        overlap_frames=int(getattr(plan, 'overlap_frames', 0) or 0),
        passes=int(getattr(plan, 'passes', len(plan.segment_lengths))),
        fps=float(FPS),
    )
    # v0.3.81 release polish: for the only supported release continuation
    # shape (exactly two passes), keep the established shot/identity contract
    # alive for the *whole* second pass instead of relying on a long AV ref
    # whose finite span can create a visible release point mid-segment.
    if (
        int(segment_index) == 1
        and int(getattr(plan, 'passes', len(plan.segment_lengths))) == 2
        and getattr(plan, 'mode', None) == 'segmented_continuation'
    ):
        continuity_lock = (
            "\n\nContinuity lock for this continuation: continue the exact same uninterrupted shot from the preceding generated video. "
            "Preserve the established camera direction, framing, subject order, relative scale, motion direction, clothing, face identity, "
            "and all visible accessories or facial coverings exactly as established. Do not replace a mask with glasses or introduce/remove accessories. "
            "Do not re-stage, re-frame, cut, reset, or start a new shot; only continue the ongoing action naturally."
        )
        selected = (str(selected).rstrip() + continuity_lock).strip()
        _lm_print(
            '[MiniMaxH3 LongMedia][0.3.81 TWO-PASS CONTINUITY LOCK] '
            'pass=1 full-segment shot/identity lock active; AV carry restored to overlap-sized baseline',
            flush=True,
        )
    if policy.get('mode') == 'explicit_local_section':
        _lm_print(
            f'[MiniMaxH3 LongMedia][V331 SEGMENT PROMPT] pass={int(segment_index)} '
            f'uses explicit local-time section {int(policy.get("section_index", 0))+1}/'
            f'{int(policy.get("section_count", 1))}',
            flush=True,
        )
    else:
        _lm_print(
            f'[MiniMaxH3 LongMedia][V331 SEGMENT PROMPT] pass={int(segment_index)} '
            f'visible={int(policy.get("visible_start_frame", 0))}..'
            f'{int(policy.get("visible_end_frame", 0))}f '
            f'events={int(policy.get("events_selected", 0))}/'
            f'{int(policy.get("events_total", 0))} '
            f'header_actions_dropped={int(policy.get("header_sentences_dropped", 0))}',
            flush=True,
        )
    return selected


def _v57_attach_minimax_metadata(encoded_positive, source_positive, plan, segment_index, *, drop_image_refs=False):
    """Attach H3 payload by REFERENCE, optionally stripping still-image refs after pass 0."""
    if not encoded_positive or not source_positive:
        return encoded_positive
    source_meta = _conditioning_meta(source_positive[0])
    if not source_meta:
        return encoded_positive

    is_hybrid = getattr(plan, 'mode', None) == 'hybrid'
    is_final = int(segment_index) == int(plan.passes) - 1
    for entry in encoded_positive:
        meta = _conditioning_meta(entry)
        if meta is None:
            continue
        # Keep all native MiniMax payload fields. Values intentionally stay
        # shared/read-only, except that later segmented passes may strip original
        # still-image refs to prevent literal source-frame re-entry.
        for key, value in source_meta.items():
            if not str(key).startswith('minimax_'):
                continue
            if drop_image_refs and key == 'minimax_refs':
                refs = [ref for ref in (value or [])
                        if not (isinstance(ref, dict) and str(ref.get('kind', '')).lower() == 'image')]
                if refs:
                    meta[key] = refs
                else:
                    meta.pop(key, None)
            else:
                meta[key] = value

        # Hybrid frame-0 is a global opening anchor, never a reset anchor for pass > 0.
        if is_hybrid and segment_index > 0:
            keyframes = source_meta.get('minimax_keyframes') or []
            kept = []
            if is_final:
                kept = _v329_terminal_keyframes_for_segment(
                    keyframes, plan, segment_index,
                )
            if kept:
                meta['minimax_keyframes'] = kept
            else:
                meta.pop('minimax_keyframes', None)
                meta.pop('minimax_frame_count', None)
    return encoded_positive


def _v329_terminal_keyframes_for_segment(keyframes, plan, segment_index):
    """Move a global terminal guide to the final pass's local last frame."""

    local_last = max(0, int(plan.segment_lengths[int(segment_index)]) - 1)
    result = []
    for keyframe in keyframes or []:
        if (
            float(keyframe.get('resolved_frame_index', 0)) <= 0.0
            or bool(keyframe.get('longmedia_startup_anchor'))
        ):
            continue
        item = dict(keyframe)
        item['resolved_frame_index'] = local_last
        if 'motion_context_index' in item:
            item['motion_context_index'] = local_last
        result.append(item)
    return result


def _v43_filter_continuation_ref_payload(ref_items, ref_blocks, *, drop_image_refs=False):
    """Filter pass>0 ref payloads so original still-image refs do not persist.

    H3 can occasionally literalize pass-0 image refs inside later passes. In
    segmented_continuation we therefore allow original image refs only on pass 0.
    Video/audio refs remain available because they represent temporal/audio
    context rather than static source images.
    """
    items = list(ref_items or [])
    blocks = list(ref_blocks or [])
    if not drop_image_refs:
        return items, blocks, 0

    filtered_items = []
    filtered_blocks = []
    dropped_images = 0
    for item, block in zip(items, blocks):
        kind = str(block.get('kind', '') or '').lower() if isinstance(block, dict) else ''
        if kind == 'image':
            dropped_images += 1
            continue
        filtered_items.append(item)
        filtered_blocks.append(block)
    return filtered_items, filtered_blocks, dropped_images


def _v43_strip_image_refs_from_conditioning_meta(encoded_positive, source_positive):
    """Copy MiniMax metadata except original still-image refs."""
    if not encoded_positive or not source_positive:
        return encoded_positive, 0
    source_meta = _conditioning_meta(source_positive[0])
    if not source_meta:
        return encoded_positive, 0
    dropped_images = 0
    for entry in encoded_positive:
        meta = _conditioning_meta(entry)
        if meta is None:
            continue
        for key, value in source_meta.items():
            if not str(key).startswith('minimax_'):
                continue
            if key == 'minimax_refs':
                refs = []
                for ref in (value or []):
                    if isinstance(ref, dict) and str(ref.get('kind', '')).lower() == 'image':
                        dropped_images += 1
                        continue
                    refs.append(ref)
                if refs:
                    meta[key] = refs
                else:
                    meta.pop(key, None)
            else:
                meta[key] = value
    return encoded_positive, dropped_images


def _v329_encode_continuation_native_refs(
    clip, prompt, positive, plan, segment_index, ref_items, ref_blocks,
):
    """Re-encode continuation text with the exact pass-0 Ref2VA presentation.

    Picture/video/audio ordering and every reference latent geometry stay
    unchanged across passes. This never exceeds pass-0 reference cost and avoids
    a conditioning-family discontinuity at the visible join.
    """
    import node_helpers

    items = list(ref_items or [])
    blocks = list(ref_blocks or [])
    if items:
        tokens = clip.tokenize(prompt, minimax_ref_items=items)
    else:
        tokens = clip.tokenize(prompt)
    scheduled = getattr(clip, 'encode_from_tokens_scheduled', None)
    encoded = scheduled(tokens) if callable(scheduled) else clip.encode(tokens)

    source_meta = _conditioning_meta(positive[0]) or {}
    values = {}
    if blocks:
        values['minimax_refs'] = blocks
    for key in ('minimax_visual_cond_noise_aug', 'minimax_audio_cond_noise_aug'):
        if key in source_meta:
            values[key] = source_meta[key]

    # Opening anchors are global and never reappear. A real terminal anchor is
    # retained only on the final pass, with the local pass frame count.
    if int(segment_index) == int(plan.passes) - 1:
        terminal = _v329_terminal_keyframes_for_segment(
            source_meta.get('minimax_keyframes') or [], plan, segment_index,
        )
        if terminal:
            values['minimax_keyframes'] = terminal
            values['minimax_frame_count'] = int(plan.segment_lengths[int(segment_index)])
    return node_helpers.conditioning_set_values(encoded, values)


def _v57_preencode_segment_conditionings(clip, base_prompt, positive, plan, v329_native_refs=None, lip_sync_audio=None, audio_vae=None):
    """Encode every pass in Setup and store Comfy's *converted* guider format.

    Raw CONDITIONING is ``[[cross_attn, metadata], ...]``. ``CFGGuider.original_conds``
    is deliberately a different representation: ``list[dict]`` produced by
    ``comfy.sampler_helpers.convert_cond``.  V57 accidentally stored raw CONDITIONING
    and later assigned it directly to ``original_conds['positive']``.  The first pass
    was unaffected because it used the externally-created guider, but pass 2 crashed
    when Comfy called ``kk.get(...)`` on the raw list entry.  Convert here while TE is
    already resident; no CLIP/TE/model object is retained in LongMediaPlan.
    """
    import comfy.sampler_helpers

    raw_result = [positive]
    prompts = [_v57_build_segment_prompt(base_prompt, plan, 0)]
    decouple_image_refs = bool(getattr(plan, 'decouple_original_image_refs_after_pass0', False))
    total_dropped_image_refs = 0
    for segment_index in range(1, int(getattr(plan, 'passes', 1))):
        segment_prompt = _v57_build_segment_prompt(base_prompt, plan, segment_index)
        drop_image_refs = bool(decouple_image_refs and int(segment_index) > 0)
        if v329_native_refs is not None:
            ref_items, ref_blocks = v329_native_refs
            identity_reanchor = bool(
                drop_image_refs
                and int(segment_index) == 1
                and int(getattr(plan, 'passes', 1)) == 2
                and getattr(plan, 'mode', None) == 'segmented_continuation'
            )
            if identity_reanchor:
                # v0.3.84: preserve the original still latents only as an
                # unlabelled visual prior.  Do NOT tokenize Picture items again:
                # Picture labels were the composition/literal-source re-entry path
                # fixed by v0.3.43.  Native Motion Context remains the temporal and
                # compositional authority for pass 1.
                non_image_items = [
                    item for item in (ref_items or [])
                    if not (isinstance(item, dict) and str(item.get('type', '')).lower() == 'image')
                ]
                identity_blocks = [dict(block) for block in (ref_blocks or [])]
                encoded = _encode_prompt(clip, segment_prompt)
                encoded = _v57_attach_minimax_metadata(
                    encoded, positive, plan, segment_index, drop_image_refs=True,
                )
                for entry in encoded:
                    meta = _conditioning_meta(entry)
                    if meta is None:
                        continue
                    existing = [dict(ref) for ref in (meta.get('minimax_refs', []) or [])]
                    # Keep non-image refs already carried by the normal decoupled
                    # path, then append still-image latents without tokenizer labels.
                    seen = set()
                    merged = []
                    for ref in existing + identity_blocks:
                        if not isinstance(ref, dict):
                            continue
                        kind = str(ref.get('kind', '')).lower()
                        if kind == 'image':
                            latent = ref.get('latent')
                            key = ('image', id(latent))
                        else:
                            key = (kind, id(ref.get('audio_latent')), id(ref.get('latent')))
                        if key in seen:
                            continue
                        seen.add(key)
                        merged.append(ref)
                    if merged:
                        meta['minimax_refs'] = merged
                    meta['longmedia_identity_reanchor'] = True
                dropped = sum(
                    1 for item in (ref_items or [])
                    if isinstance(item, dict) and str(item.get('type', '')).lower() == 'image'
                )
                total_dropped_image_refs += int(dropped)
                _lm_print(
                    '[MiniMaxH3 LongMedia][0.3.84 IDENTITY RE-ANCHOR] '
                    f'pass=1 retained {sum(1 for b in identity_blocks if str(b.get("kind", "")).lower() == "image")} '
                    'still-image latent blocks WITHOUT Picture tokenizer items; native motion context owns shot/motion',
                    flush=True,
                )
            else:
                ref_items, ref_blocks, dropped = _v43_filter_continuation_ref_payload(
                    ref_items, ref_blocks, drop_image_refs=drop_image_refs,
                )
                total_dropped_image_refs += int(dropped)
                if ref_items or ref_blocks:
                    encoded = _v329_encode_continuation_native_refs(
                        clip, segment_prompt, positive, plan, segment_index,
                        ref_items, ref_blocks,
                    )
                else:
                    encoded = _encode_prompt(clip, segment_prompt)
                    encoded = _v57_attach_minimax_metadata(encoded, positive, plan, segment_index, drop_image_refs=True)
        else:
            encoded = _encode_prompt(clip, segment_prompt)
            encoded = _v57_attach_minimax_metadata(
                encoded, positive, plan, segment_index, drop_image_refs=drop_image_refs,
            )
        if lip_sync_audio is not None:
            encoded = _v104_attach_native_lipsync_guide(
                encoded, audio_vae, lip_sync_audio, plan, segment_index,
            )
        raw_result.append(encoded)
        prompts.append(segment_prompt)

    converted_result = tuple(comfy.sampler_helpers.convert_cond(cond) for cond in raw_result)
    if v329_native_refs is not None and len(converted_result) > 1:
        ref_items, ref_blocks = v329_native_refs
        _lm_print(
            '[MiniMaxH3 LongMedia][V329 STABLE NATIVE REFS] '
            f'continuation preserves {len(ref_items)} tokenizer items and '
            f'{len(ref_blocks)} latent blocks in pass-0 order/geometry; no identity sheet',
            flush=True,
        )
    if decouple_image_refs and total_dropped_image_refs > 0:
        _lm_print(
            '[MiniMaxH3 LongMedia][0.3.43 REF DECOUPLING] '
            f'pass>0 removed {int(total_dropped_image_refs)} original still-image ref blocks; '
            'later passes keep generated AV context and any non-image refs only',
            flush=True,
        )
    _lm_print(
        f'[MiniMaxH3 LongMedia][V58 CONDITIONING FORMAT] pre-encoded {len(converted_result)} pass conditionings '
        'inside Setup and converted to CFGGuider list[dict] format; CLIP/TE is NOT stored in LongMediaPlan',
        flush=True,
    )
    return converted_result, tuple(prompts)


def _v60_context_step_offsets(latent_t):
    frame_per_token = (1, 4, 4, 4, 4)
    out, acc = [], 0
    for k in range(int(latent_t)):
        out.append(acc)
        acc += frame_per_token[k % 5]
    return out, acc


def _v60_attach_previous_head_guides(positive_list, previous_av, plan, segment_index):
    """Pin the previous latent tail onto the HEAD of the current H3 timeline.

    Unlike V59, this is NOT a Ref2VA video reference.  Each latent step is a
    never-denoised MiniMax keyframe guide at its real target-relative time.
    The carried span intentionally matches LongMedia's frozen overlap so the
    existing stitch removes exactly the repeated head after sampling.
    """
    if not positive_list or previous_av is None or int(segment_index) <= 0:
        return positive_list, 0
    # During GraphBuilder expansion ``previous_av`` is normally a graph-output proxy,
    # not the runtime LATENT dictionary.  Do not treat that proxy as an AV latent.
    # The actual continuation overlap is already copied/frozen at runtime by
    # MiniMaxH3LatentLabLongMediaNextSegment, so skipping the auxiliary V60 guides
    # here preserves real motion context instead of emitting a misleading error.
    if not isinstance(previous_av, dict) or 'samples' not in previous_av:
        _lm_print(
            '[MiniMaxH3 LongMedia][V319 MOTION CONTEXT] auxiliary V60 guides skipped '
            '(previous segment is a GraphBuilder proxy); frozen latent overlap remains active',
            flush=True,
        )
        return positive_list, 0
    try:
        prev_video, _prev_audio = unpack_av_samples(previous_av)
        try:
            from . import motion_context_layout_patch
        except Exception:
            import importlib
            motion_context_layout_patch = importlib.import_module(__package__ + '.motion_context_layout_patch')
        if not motion_context_layout_patch.apply_patch():
            raise RuntimeError('PackedLayout motion-context patch could not activate')
        overlap = int(getattr(plan, 'overlap_frames', 0) or 0)
        # Native H3 video-run grid.  Match (never exceed) the frozen overlap.
        run = next((g for g in (56, 39, 22, 5, 1) if g <= overlap), 0)
        if run <= 0:
            return positive_list, 0
        context_t = int(video_latent_t(run))
        context_t = min(context_t, int(prev_video.shape[2]))
        offsets, covered = _v60_context_step_offsets(context_t)
        if covered != run:
            raise RuntimeError(
                f'context latent grid mismatch: {context_t} steps cover {covered} frames, wanted {run}')
        tail = prev_video[:, :, -context_t:]
        keyframes = [
            {
                'resolved_frame_index': 0,  # stock-safe; PackedLayout patch uses marker below
                'motion_context_index': int(offset),
                'latent': tail[:, :, k:k + 1],
                'longmedia_motion_context': True,
            }
            for k, offset in enumerate(offsets)
        ]

        out = []
        attached = False
        frame_count = int(plan.segment_lengths[int(segment_index)])
        for entry in positive_list:
            if isinstance(entry, dict):
                meta = dict(entry)
                prior = list(meta.get('minimax_keyframes', []) or [])
                # Continuation head owns 0..run-1.  Preserve only anchors after it
                # (e.g. an explicit final-frame destination).
                kept = []
                for kf in prior:
                    pos = float(kf.get('motion_context_index', kf.get('resolved_frame_index', 0)))
                    if pos >= run:
                        kept.append(kf)
                merged_keyframes = kept + keyframes
                merged_keyframes.sort(
                    key=lambda kf: float(kf.get('motion_context_index', kf.get('resolved_frame_index', 0)))
                )
                meta['minimax_keyframes'] = merged_keyframes
                meta['minimax_frame_count'] = frame_count
                meta['longmedia_motion_context_frames'] = int(run)
                out.append(meta)
                attached = True
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], dict):
                new_entry = list(entry)
                meta = dict(entry[1])
                prior = list(meta.get('minimax_keyframes', []) or [])
                kept = []
                for kf in prior:
                    pos = float(kf.get('motion_context_index', kf.get('resolved_frame_index', 0)))
                    if pos >= run:
                        kept.append(kf)
                merged_keyframes = kept + keyframes
                merged_keyframes.sort(
                    key=lambda kf: float(kf.get('motion_context_index', kf.get('resolved_frame_index', 0)))
                )
                meta['minimax_keyframes'] = merged_keyframes
                meta['minimax_frame_count'] = frame_count
                meta['longmedia_motion_context_frames'] = int(run)
                new_entry[1] = meta
                out.append(new_entry)
                attached = True
            else:
                out.append(entry)
        if attached:
            _lm_print(
                f'[MiniMaxH3 LongMedia][V60 TRUE MOTION CONTEXT] previous tail pinned as '
                f'{len(keyframes)} timeline keyframe blocks covering {run} frames; '
                f'indices={offsets}; Ref2VA semantics NOT used; repeated head trimmed by overlap',
                flush=True,
            )
            return out, int(run)
    except Exception as exc:
        _lm_print(
            f'[MiniMaxH3 LongMedia][V60 TRUE MOTION CONTEXT] disabled: '
            f'{type(exc).__name__}: {exc}', flush=True,
        )
    return positive_list, 0






def _v83_native_guide_api_supported():
    """Return True when current ComfyUI supports arbitrary native H3 guide positions."""
    try:
        import inspect
        import comfy.ldm.minimax.model as minimax_model
        cls = getattr(minimax_model, 'PackedLayout', None)
        if cls is None:
            return False
        params = inspect.signature(cls.__init__).parameters
        # Current native API removed the legacy frame_count-only restriction and
        # accepts arbitrary resolved_frame_index guide rows directly.
        return 'frame_count' not in params
    except Exception:
        return False


def _v83_attach_native_motion_context(positive_list, previous_av, plan, segment_index):
    """Attach previous sampled AV tail as native H3 temporal keyframes.

    Release scope is deliberately narrow: only the first continuation of an
    exactly two-pass segmented generation.  The target latent remains fresh;
    the first 22 decoded frames are generated under never-denoised temporal
    guide rows and are trimmed by the existing stitch.
    """
    if (
        not positive_list or previous_av is None
        or int(segment_index) <= 0
        or (
            getattr(plan, 'mode', None) != 'multiclip'
            and not (
                int(segment_index) == 1
                and int(getattr(plan, 'passes', 0) or 0) == 2
                and getattr(plan, 'mode', None) == 'segmented_continuation'
            )
        )
        or not _v83_native_guide_api_supported()
    ):
        return positive_list, 0
    if not isinstance(previous_av, dict) or 'samples' not in previous_av:
        return positive_list, 0

    prev_video, prev_audio = unpack_av_samples(previous_av)
    overlap = int(getattr(plan, 'overlap_frames', 0) or 0)
    run = next((g for g in (56, 39, 22, 5) if g <= overlap), 0)
    if run <= 0:
        return positive_list, 0

    context_t = int(video_latent_t(run))
    if context_t > int(prev_video.shape[2]):
        return positive_list, 0
    offsets, covered = _v60_context_step_offsets(context_t)
    if int(covered) != int(run):
        raise RuntimeError(
            f'0.3.83 native motion context grid mismatch: {context_t} steps cover {covered}f, expected {run}f'
        )

    video_tail = prev_video[:, :, -context_t:]
    keyframes = [
        {
            'resolved_frame_index': int(offset),
            'latent': video_tail[:, :, k:k + 1].clone(),
            'longmedia_native_motion_context': True,
        }
        for k, offset in enumerate(offsets)
    ]

    # Carry audio from the same sampled AV latent and place it on the same
    # target timeline.  This mirrors H3's native guide coordinate system rather
    # than treating the tail as a generic reference-audio embedding.
    audio_t = min(int(audio_latent_t(run)), int(prev_audio.shape[-1]))
    # v0.3.108: source Audio1 is the authoritative lip-sync clock.  Keep the
    # previous VIDEO latent as Motion Context, but do not add a second sampled
    # audio clock when a native local-0 lip-sync guide is active.  Competing
    # sampled/source audio conditions weaken articulation and can drift.
    if bool(getattr(plan, 'lip_sync_native_audio_guide', False)):
        audio_t = 0
    if audio_t > 0:
        source_frames = frame_count_from_video_t(int(prev_video.shape[2]))
        frame_rescale = float(AUDIO_LATENT_FPS) / float(FPS)
        overhang = float(prev_audio.shape[-1]) - frame_rescale * float(source_frames)
        if not (0.0 <= overhang < 1.0):
            _lm_print(
                '[MiniMaxH3 LongMedia][0.3.105 AUDIO GRID] '
                f'unexpected previous AV audio grid: audio_t={int(prev_audio.shape[-1])} '
                f'video_frames={int(source_frames)} raw_overhang={overhang:.6f}; using 0.0',
                flush=True,
            )
            overhang = 0.0
        end_frame = float(run) + overhang / frame_rescale
        end_coord = round(frame_rescale * end_frame)
        end_frame = float(end_coord) / frame_rescale
        audio_start_frame = end_frame - float(audio_t) / frame_rescale
        keyframes.append({
            'resolved_frame_index': float(audio_start_frame),
            'audio_latent': prev_audio[..., -audio_t:].clone(),
            'longmedia_native_motion_audio': True,
        })

    out = []
    attached = False
    for entry in positive_list:
        if isinstance(entry, dict):
            meta = dict(entry)
            prior = [dict(kf) for kf in (meta.get('minimax_keyframes', []) or [])
                     if not bool(kf.get('longmedia_native_motion_context'))
                     and not bool(kf.get('longmedia_native_motion_audio'))]
            # The continuation guide owns the repeated 0..run-1 head. Keep only
            # explicit destination guides after that region.
            prior = [
                kf for kf in prior
                if bool(kf.get('longmedia_lipsync_audio_guide'))
                or float(kf.get('resolved_frame_index', 0.0)) >= float(run)
            ]
            meta['minimax_keyframes'] = prior + [dict(kf) for kf in keyframes]
            meta.pop('minimax_frame_count', None)  # native arbitrary-guide API
            meta['longmedia_native_motion_context_frames'] = int(run)
            out.append(meta)
            attached = True
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], dict):
            new_entry = list(entry)
            meta = dict(entry[1])
            prior = [dict(kf) for kf in (meta.get('minimax_keyframes', []) or [])
                     if not bool(kf.get('longmedia_native_motion_context'))
                     and not bool(kf.get('longmedia_native_motion_audio'))]
            prior = [
                kf for kf in prior
                if bool(kf.get('longmedia_lipsync_audio_guide'))
                or float(kf.get('resolved_frame_index', 0.0)) >= float(run)
            ]
            meta['minimax_keyframes'] = prior + [dict(kf) for kf in keyframes]
            meta.pop('minimax_frame_count', None)
            meta['longmedia_native_motion_context_frames'] = int(run)
            new_entry[1] = meta
            out.append(new_entry)
            attached = True
        else:
            out.append(entry)

    if attached:
        _lm_print(
            '[MiniMaxH3 LongMedia][0.3.108 VIDEO MOTION CONTEXT + SOURCE AUDIO CLOCK] '
            f'segment0->1 context={run}f video_steps={context_t} indices={offsets}; '
            f'audio_steps={audio_t}; source_audio_clock={bool(getattr(plan, 'lip_sync_native_audio_guide', False))}; target head is FRESH and generated under native minimax_keyframes; '
            'existing stitch trims the repeated guide span',
            flush=True,
        )
        return out, int(run)
    return positive_list, 0


def _v0322_audio_grid_offset(frame_count: int, actual_audio_t: int) -> float:
    """Signed H3 audio-grid phase at the end of a video frame span."""
    return float(actual_audio_t) - (float(frame_count) * float(AUDIO_LATENT_FPS) / float(FPS))


def _v80_native_av_context_frames(previous_frames, overlap_frames, segment_index, first_handoff_bridge=False):
    """Choose a native H3 AV-reference span for one continuation boundary.

    The first segmented handoff is special: pass 0 was image/startup-conditioned,
    while pass 1 is the first pass after still-image ref decoupling.  A context
    span limited to the visible frozen overlap (normally 22f) is often too short
    to preserve the established shot/composition through that conditioning-family
    transition.  Give *only* segment 1 a longer conditioning-only raw AV tail.

    Later continuation->continuation boundaries stay on the proven overlap-sized
    path so this does not accumulate context or change their already-good joins.
    """
    previous_frames = max(0, int(previous_frames))
    overlap_frames = max(0, int(overlap_frames))
    segment_index = int(segment_index)
    if bool(first_handoff_bridge) and segment_index == 1:
        cap = min(previous_frames, 56)
        candidates = (56, 39, 22, 5)
    else:
        cap = min(previous_frames, overlap_frames)
        candidates = (39, 22, 5)
    return next((frames for frames in candidates if frames <= cap), 0)


def _v0322_attach_native_av_context_ref(positive_list, previous_av, plan, segment_index):
    """Attach the previous raw AV tail as one native paired H3 context reference.

    This follows Continuum's latent-first handoff principle: the previous video and
    audio tails travel together in ``minimax_refs`` while the visible overlap remains
    frozen in the target latent and is trimmed after sampling.  Identity refs stay
    intact and this context ref is appended without adding tokenizer Picture tokens.
    """
    if not positive_list or previous_av is None or int(segment_index) <= 0:
        return positive_list, 0
    if not isinstance(previous_av, dict) or 'samples' not in previous_av:
        return positive_list, 0
    prev_video, prev_audio = unpack_av_samples(previous_av)
    previous_frames = frame_count_from_video_t(int(prev_video.shape[2]))
    overlap = int(getattr(plan, 'overlap_frames', 0) or 0)
    first_handoff_bridge = False  # v0.3.81 release polish: restore proven overlap-sized AV carry
    # v0.3.81: segment 0 -> 1 is the only boundary where conditioning changes
    # from startup/Picture-driven to still-ref-decoupled continuation.  Preserve
    # the exact 22f frozen head, but feed up to 56f of the already-generated raw
    # AV tail as conditioning-only history.  Later joins retain the 0.3.77 path.
    run = _v80_native_av_context_frames(
        previous_frames, overlap, segment_index,
        first_handoff_bridge=first_handoff_bridge,
    )
    if run <= 0:
        return positive_list, 0
    video_t = int(video_latent_t(run))
    audio_t = int(audio_latent_t(run))
    if int(prev_video.shape[2]) < video_t or int(prev_audio.shape[-1]) < audio_t:
        return positive_list, 0
    video_tail = prev_video[:, :, -video_t:].contiguous()
    audio_tail = prev_audio[..., -audio_t:].contiguous()
    grid_offset = _v0322_audio_grid_offset(previous_frames, int(prev_audio.shape[-1]))
    context_ref = {
        'kind': 'video_audio',
        'latent_t': int(video_tail.shape[2]),
        'latent_h': int(video_tail.shape[-2]),
        'latent_w': int(video_tail.shape[-1]),
        'ref_audio_t': int(audio_tail.shape[-1]),
        'latent': video_tail,
        'audio_latent': audio_tail,
        'longmedia_native_av_context': True,
        'longmedia_context_frames': int(run),
        'longmedia_audio_grid_offset': float(grid_offset),
        'longmedia_source_segment_index': int(segment_index) - 1,
        'longmedia_first_handoff_bridge': bool(first_handoff_bridge),
    }
    out = []
    attached = False
    for entry in positive_list:
        if isinstance(entry, dict):
            meta = dict(entry)
            refs = [dict(ref) for ref in (meta.get('minimax_refs', []) or [])
                    if not bool(ref.get('longmedia_native_av_context'))]
            refs.append(dict(context_ref))
            meta['minimax_refs'] = refs
            meta['longmedia_native_av_context_frames'] = int(run)
            out.append(meta)
            attached = True
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], dict):
            new_entry = list(entry)
            meta = dict(entry[1])
            refs = [dict(ref) for ref in (meta.get('minimax_refs', []) or [])
                    if not bool(ref.get('longmedia_native_av_context'))]
            refs.append(dict(context_ref))
            meta['minimax_refs'] = refs
            meta['longmedia_native_av_context_frames'] = int(run)
            new_entry[1] = meta
            out.append(new_entry)
            attached = True
        else:
            out.append(entry)
    if attached:
        _lm_print(
            '[MiniMaxH3 LongMedia][V0322 NATIVE AV CONTEXT] '
            f'segment={int(segment_index)} context={run}f video_t={video_t} '
            f'audio_t={audio_t} audio_grid_offset={grid_offset:+.6f}; '
            f'first_handoff_bridge={bool(first_handoff_bridge)}; '
            'paired raw AV tail appended to minimax_refs',
            flush=True,
        )
        if first_handoff_bridge:
            _lm_print(
                '[MiniMaxH3 LongMedia][0.3.81 FIRST HANDOFF BRIDGE] '
                f'segment0->1 uses {int(run)}f generated raw AV history while frozen overlap remains {int(overlap)}f; '
                'original still-image refs remain decoupled; pass>=2 unchanged from 0.3.77',
                flush=True,
            )
        return out, int(run)
    return positive_list, 0

def _clone_guider_with_segment_audio(guider, plan, segment_index, previous_av=None):
    """Clone guider cheaply, select pass conditioning, and add previous motion context."""
    shifted = copy.copy(guider)
    shifted.model_options = copy.deepcopy(getattr(guider, 'model_options', {}) or {})
    timeline = _segment_timeline_contract(plan, segment_index)
    start_frame = int(timeline['context_start'])
    visible_start_frame = int(timeline['visible_start'])

    # IMPORTANT V57: do not deepcopy the whole conditioning/media payload. Hybrid refs can
    # contain encoded image/video/audio latents. Only make a new dict shell and swap positive.
    shifted.original_conds = dict(getattr(guider, 'original_conds', {}) or {})
    segment_conds = getattr(plan, 'segment_positive_conditionings', None)
    if segment_conds and int(segment_index) < len(segment_conds):
        shifted.original_conds['positive'] = segment_conds[int(segment_index)]

    # paired AV continuation: carry the previous RAW video+audio tails together as one
    # native H3 context reference. Keep the existing ordered motion guides as a
    # complementary C1/pose-phase signal; the paired AV ref owns soundtrack phase.
    if getattr(plan, 'mode', None) != 'storyboard_bridge' and int(segment_index) > 0 and previous_av is not None:
        native_two_pass = (
            (
                getattr(plan, 'mode', None) == 'multiclip' and int(segment_index) > 0
            )
            or (
                int(segment_index) == 1
                and int(getattr(plan, 'passes', 0) or 0) == 2
                and getattr(plan, 'mode', None) == 'segmented_continuation'
            )
        ) and _v83_native_guide_api_supported()
        if native_two_pass:
            motion_positive, _motion_frames = _v83_attach_native_motion_context(
                shifted.original_conds.get('positive', []), previous_av, plan, segment_index,
            )
            shifted.original_conds['positive'] = motion_positive
        else:
            av_positive, _av_frames = _v0322_attach_native_av_context_ref(
                shifted.original_conds.get('positive', []), previous_av, plan, segment_index,
            )
            shifted.original_conds['positive'] = av_positive
            motion_positive, _motion_frames = _v60_attach_previous_head_guides(
                shifted.original_conds.get('positive', []), previous_av, plan, segment_index,
            )
            shifted.original_conds['positive'] = motion_positive

    transformer_options = shifted.model_options.setdefault('transformer_options', {})
    if getattr(plan, 'mode', None) != 'storyboard_bridge':
        transformer_options[TEMPORAL_OFFSET_OPTION] = temporal_offset_for_frame(start_frame)
    else:
        transformer_options.pop(TEMPORAL_OFFSET_OPTION, None)
    if WrappersMP is not None:
        wrappers = transformer_options.setdefault('wrappers', {})
        apply_model = wrappers.setdefault(WrappersMP.APPLY_MODEL, {})
        apply_model['MiniMaxH3LatentLabTemporalOffset'] = [h3_temporal_offset_wrapper]

    # Only source/reference streams that genuinely change per pass require a tiny mutable
    # metadata shell. Hybrid global refs are read-only and remain shared with zero copies.
    reference_audio = getattr(plan, 'reference_audio', None) or plan.source_audio
    positive_list = shifted.original_conds.get('positive', [])
    needs_mutable_refs = bool(reference_audio is not None or plan.source_video is not None)
    if needs_mutable_refs and positive_list:
        cloned_positive = []
        for entry in positive_list:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], dict):
                new_meta = dict(entry[1])
                if 'minimax_refs' in new_meta:
                    new_meta['minimax_refs'] = [dict(ref) for ref in new_meta.get('minimax_refs', [])]
                new_entry = list(entry)
                new_entry[1] = new_meta
                cloned_positive.append(new_entry)
            elif isinstance(entry, dict):
                new_entry = dict(entry)
                if 'minimax_refs' in new_entry:
                    new_entry['minimax_refs'] = [dict(ref) for ref in new_entry.get('minimax_refs', [])]
                cloned_positive.append(new_entry)
            else:
                cloned_positive.append(entry)
        shifted.original_conds['positive'] = cloned_positive
        positive_list = cloned_positive

    refs = []
    if positive_list:
        meta0 = _conditioning_meta(positive_list[0])
        if meta0:
            refs = meta0.get('minimax_refs', []) or []

    audio_timeline_logged = False
    for ref in refs:
        if ref.get('kind') == 'audio' and reference_audio is not None and plan.audio_vae is not None:
            length_frames = int(timeline['length_frames'])
            # V318 timeline contract: the H3 reference stream covers the whole local
            # pass, including hidden overlap. Starting it at visible_start would put
            # the music visible boundary at local t=0 while the target boundary is
            # local t=overlap, creating an overlap-sized phase error on pass 2+.
            audio_start_frame = int(timeline['context_start'])
            if (not audio_timeline_logged) and int(segment_index) > 0:
                _lm_print(
                    '[MiniMaxH3 LongMedia][V318 TIMELINE] '
                    f'segment={int(segment_index)} context_start={int(timeline["context_start"])}f '
                    f'visible_start={int(timeline["visible_start"])}f '
                    f'local_visible_offset={int(timeline["local_visible_offset"])}f '
                    f'audio_ref_window_start={audio_start_frame}f',
                    flush=True,
                )
                audio_timeline_logged = True
            available, _ = _slice_source_audio_for_segment(reference_audio, audio_start_frame, length_frames)
            waveform_for_encode = available.movedim(1, -1)
            audio_lat = plan.audio_vae.encode(waveform_for_encode)
            ref['ref_audio_t'] = audio_lat.shape[-1]
            ref['audio_latent'] = audio_lat
            ref['longmedia_context_start_frame'] = int(timeline['context_start'])
            ref['longmedia_visible_start_frame'] = int(timeline['visible_start'])
            ref['longmedia_local_visible_offset_frames'] = int(timeline['local_visible_offset'])
        elif ref.get('kind') == 'video' and plan.source_video is not None and plan.video_vae is not None:
            length_frames = int(timeline['length_frames'])
            source_frames = slice_video_segment(plan.source_video, start_frame, length_frames, plan.video_fps)
            ref['video_latent'] = plan.video_vae.encode(source_frames)
            ref['longmedia_context_start_frame'] = int(timeline['context_start'])
            ref['longmedia_visible_start_frame'] = int(timeline['visible_start'])
            ref['longmedia_local_visible_offset_frames'] = int(timeline['local_visible_offset'])
    return shifted

def _encode_prompt(clip, prompt):
    """Encode a text prompt into CONDITIONING across supported ComfyUI CLIP APIs."""
    tokens = clip.tokenize(prompt)

    # Current ComfyUI's canonical CONDITIONING path.  This keeps hook/LoRA
    # schedules attached to the conditioning and avoids relying on the old
    # MiniMax-era CLIP.encode(..., control=None) signature, which was removed.
    scheduled = getattr(clip, "encode_from_tokens_scheduled", None)
    if callable(scheduled):
        return scheduled(tokens)

    # Compatibility fallback for older ComfyUI builds that exposed the
    # MiniMax-specific control keyword on CLIP.encode().
    try:
        return clip.encode(tokens, control=None)
    except TypeError as exc:
        if "control" not in str(exc):
            raise

    # Last-resort compatibility for CLIP implementations where encode() takes
    # raw text rather than pre-tokenized input.
    return clip.encode(prompt)


class MiniMaxH3LatentLabRuntimeContinuationGuider:
    """Build continuation conditioning after the previous segment exists at runtime."""

    DESCRIPTION = 'Internal runtime continuation guider with real previous-segment motion context.'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'guider': ('GUIDER',),
                'long_media_plan': ('LONG_MEDIA_PLAN',),
                'previous_av': ('LATENT',),
                'segment_index': ('INT', {'default': 1, 'min': 1, 'max': 100, 'step': 1}),
            }
        }

    RETURN_TYPES = ('GUIDER', 'STRING')
    RETURN_NAMES = ('guider', 'report')
    FUNCTION = 'build'
    CATEGORY = CATEGORY_LONGMEDIA

    def build(self, guider, long_media_plan, previous_av, segment_index):
        seg_idx = int(segment_index)
        if not isinstance(previous_av, dict) or 'samples' not in previous_av:
            raise ValueError(
                "V320 runtime continuation handoff expected completed previous LATENT with 'samples'."
            )
        prev_video, _prev_audio = unpack_av_samples(previous_av)
        previous_frames = frame_count_from_video_t(int(prev_video.shape[2]))
        shifted = _clone_guider_with_segment_audio(
            guider, long_media_plan, seg_idx, previous_av=previous_av,
        )
        overlap = int(getattr(long_media_plan, 'overlap_frames', 0) or 0)
        run = next((g for g in (56, 39, 22, 5, 1) if g <= overlap), 0)
        _lm_print(
            '[MiniMaxH3 LongMedia][V320 RUNTIME MOTION HANDOFF] '
            f'segment={seg_idx} previous_frames={previous_frames}f overlap={overlap}f '
            f'guide_span={run}f runtime_previous_latent=yes',
            flush=True,
        )
        report = json.dumps({
            'segment_index': seg_idx,
            'previous_frames': int(previous_frames),
            'overlap_frames': overlap,
            'motion_guide_span_frames': int(run),
            'runtime_previous_latent': True,
        }, indent=2)
        return (shifted, report)


class MiniMaxH3LatentLabVideoEncode:
    DESCRIPTION = (
        'Encode IMAGE frames as a standalone MiniMax H3 video stream. '
        'Connect target_av to force exact H3 canvas and temporal shape '
        'before packing/replacement.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'vae': ('VAE',),
                'frames': ('IMAGE',),
                'frame_fit': (['strict', 'crop_or_pad_last', 'loop'],),
                'resize_mode': (['none', 'stretch', 'center_crop'],),
            },
            'optional': {
                'target_av': (
                    'LATENT',
                    {'tooltip': 'Optional H3 AV latent whose video shape is the target.'},
                ),
            },
        }

    RETURN_TYPES = ('LATENT', 'INT', 'INT', 'INT')
    RETURN_NAMES = ('video_latent', 'frames', 'width', 'height')
    FUNCTION = 'encode'
    CATEGORY = CATEGORY_STREAMS

    def encode(self, vae, frames, frame_fit, resize_mode, target_av=None):
        if target_av is not None:
            target_video, target_count, width, height = _target_video_geometry(target_av)
        else:
            target_video = None
            source_count = int(frames.shape[0])
            if frame_fit == 'strict':
                if not _is_valid_frame_count(source_count):
                    raise ValueError(
                        f'MiniMax H3 frame count must be 17*k+5, got {source_count}.'
                    )
                target_count = source_count
            else:
                target_count = align_frame_count(source_count)
            height = int(frames.shape[1])
            width = int(frames.shape[2])
            if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
                if resize_mode == 'none':
                    raise ValueError(
                        f'H3 canvas must be divisible by 32, got {width}x{height}.'
                    )
                width = max(CANVAS_MULTIPLE, round(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                height = max(CANVAS_MULTIPLE, round(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        fitted = _fit_frames(frames, target_count, frame_fit)
        fitted = _resize_frames(fitted, width, height, resize_mode)
        latent = vae.encode(fitted)
        _validate_video(latent)
        if target_video is not None and tuple(latent.shape) != tuple(target_video.shape):
            raise ValueError(
                f'Video VAE produced {tuple(latent.shape)}, '
                f'target AV requires {tuple(target_video.shape)}.'
            )
        return ({'samples': latent}, target_count, width, height)


class MiniMaxH3LatentLabAudioEncode:
    DESCRIPTION = (
        'Resample and encode AUDIO as a standalone MiniMax H3 audio stream. '
        'target_av makes the encoded stream exactly match the target duration.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'audio_vae': ('VAE',),
                'audio': ('AUDIO',),
                'fit_mode': (['strict', 'crop_or_pad_silence', 'loop'],),
            },
            'optional': {
                'target_av': (
                    'LATENT',
                    {'tooltip': 'Optional H3 AV latent whose audio shape is the target.'},
                ),
            },
        }

    RETURN_TYPES = ('LATENT', 'FLOAT', 'INT')
    RETURN_NAMES = ('audio_latent', 'duration_seconds', 'sample_rate')
    FUNCTION = 'encode'
    CATEGORY = CATEGORY_STREAMS

    def encode(self, audio_vae, audio, fit_mode, target_av=None):
        waveform = audio['waveform'][:1]
        source_rate = int(audio['sample_rate'])
        vae_rate = int(getattr(audio_vae, 'audio_sample_rate', 32000))
        if source_rate != vae_rate:
            waveform = torchaudio.functional.resample(waveform, source_rate, vae_rate)
        target_audio = None
        if target_av is not None:
            _, target_audio = unpack_av_samples(target_av)
            target_samples = round(target_audio.shape[-1] / AUDIO_LATENT_FPS * vae_rate)
            waveform = _fit_waveform(waveform, target_samples, fit_mode)
        latent = audio_vae.encode(waveform.movedim(1, -1))
        if target_audio is not None:
            if fit_mode == 'strict':
                latent = _fit_stream(latent, target_audio, 'audio', 'strict', 'start')
            else:
                latent = _fit_stream(latent, target_audio, 'audio', 'crop_pad', 'start')
        duration = latent.shape[-1] / AUDIO_LATENT_FPS
        return ({'samples': latent}, float(duration), vae_rate)


class MiniMaxH3LatentLabPackAV:
    DESCRIPTION = (
        'Pack standalone 24-channel video and 32-channel stereo audio latents '
        'into the NestedTensor format consumed by MiniMax H3. Durations must match.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'video_latent': ('LATENT',),
                'audio_latent': ('LATENT',),
            }
        }

    RETURN_TYPES = ('LATENT',)
    RETURN_NAMES = ('av_latent',)
    FUNCTION = 'pack'
    CATEGORY = CATEGORY_STREAMS

    def pack(self, video_latent, audio_latent):
        return (pack_av_latents(video_latent, audio_latent, NestedTensor),)


class MiniMaxH3LatentLabSplitAV:
    DESCRIPTION = (
        'Split a MiniMax H3 NestedTensor AV latent into editable '
        'video and audio LATENT streams.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'av_latent': ('LATENT',)}}

    RETURN_TYPES = ('LATENT', 'LATENT')
    RETURN_NAMES = ('video_latent', 'audio_latent')
    FUNCTION = 'split'
    CATEGORY = CATEGORY_STREAMS

    def split(self, av_latent):
        return split_av_latent(av_latent)


class MiniMaxH3LatentLabReplaceStream:
    """Replace one whole stream (video or audio) in an H3 AV latent.

    FIXED_KIND is None on the primary node (stream picked via widget) and
    'video'/'audio' on the two legacy subclasses kept below so old saved
    graphs (which have no 'stream' widget) keep loading and running.
    """

    FIXED_KIND = None
    DESCRIPTION = (
        'Replace the video or audio stream in an H3 AV latent. '
        'strict is lossless; crop_pad center-crops/pads spatially '
        'and uses alignment for time.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            'av_latent': ('LATENT',),
            'replacement': ('LATENT',),
            'fit_mode': (['strict', 'crop_pad'],),
            'alignment': (['start', 'end', 'center'],),
            'denoise': (
                'FLOAT',
                {
                    'default': 0.0,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'tooltip': '0 preserves the replacement exactly; 1 fully denoises it.',
                },
            ),
        }
        if cls.FIXED_KIND is None:
            required['stream'] = (['video', 'audio'],)
        return {'required': required}

    RETURN_TYPES = ('LATENT',)
    RETURN_NAMES = ('av_latent',)
    FUNCTION = 'replace'
    CATEGORY = CATEGORY_STREAMS

    def replace(self, av_latent, replacement, fit_mode, alignment, denoise, stream=None):
        kind = self.FIXED_KIND or stream
        return (
            replace_stream(av_latent, replacement, kind, fit_mode, alignment, denoise, NestedTensor),
        )


class MiniMaxH3LatentLabReplaceVideo(MiniMaxH3LatentLabReplaceStream):
    FIXED_KIND = 'video'
    DESCRIPTION = (
        'Deprecated — kept only so old saved graphs keep loading. '
        'Use "MiniMax H3 \u2022 Replace Stream" for new graphs (stream=video).'
    )


class MiniMaxH3LatentLabReplaceAudio(MiniMaxH3LatentLabReplaceStream):
    FIXED_KIND = 'audio'
    DESCRIPTION = (
        'Deprecated — kept only so old saved graphs keep loading. '
        'Use "MiniMax H3 \u2022 Replace Stream" for new graphs (stream=audio).'
    )


class MiniMaxH3LatentLabStreamDenoise:
    DESCRIPTION = (
        "Independent H3 stream control through ComfyUI's stock noise_mask path. "
        '0 preserves a stream; 1 fully regenerates it.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'av_latent': ('LATENT',),
                'video_denoise': (
                    'FLOAT',
                    {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'audio_denoise': (
                    'FLOAT',
                    {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'merge_mode': (['replace', 'multiply', 'minimum', 'maximum'],),
            }
        }

    RETURN_TYPES = ('LATENT',)
    RETURN_NAMES = ('av_latent',)
    FUNCTION = 'control'
    CATEGORY = CATEGORY_STREAMS

    def control(self, av_latent, video_denoise, audio_denoise, merge_mode):
        return (
            set_stream_denoise(
                av_latent, video_denoise, audio_denoise, merge_mode, NestedTensor
            ),
        )


class MiniMaxH3LatentLabLipSyncSetup:
    DESCRIPTION = (
        'Replace the native H3 audio stream and configure independent '
        'video/audio denoise controls for lip-sync-oriented sampling.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'av_latent': ('LATENT',),
                'audio_latent': ('LATENT',),
                'fit_mode': (['strict', 'crop_pad'],),
                'alignment': (['start', 'end', 'center'],),
                'video_denoise': (
                    'FLOAT',
                    {'default': 0.35, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'audio_denoise': (
                    'FLOAT',
                    {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
            }
        }

    RETURN_TYPES = ('LATENT',)
    RETURN_NAMES = ('av_latent',)
    FUNCTION = 'setup'
    CATEGORY = CATEGORY_UTIL

    def setup(self, av_latent, audio_latent, fit_mode, alignment, video_denoise, audio_denoise):
        replaced = replace_stream(
            av_latent, audio_latent, 'audio', fit_mode, alignment, audio_denoise, NestedTensor
        )
        return (
            set_stream_denoise(replaced, video_denoise, audio_denoise, 'replace', NestedTensor),
        )


class MiniMaxH3LatentLabVideoInpaint:
    DESCRIPTION = (
        'Map a ComfyUI MASK onto the H3 video latent grid. '
        'White uses denoise_inside, black uses denoise_outside; '
        'audio is controlled separately.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'av_latent': ('LATENT',),
                'mask': ('MASK',),
                'denoise_inside': (
                    'FLOAT',
                    {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'denoise_outside': (
                    'FLOAT',
                    {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'audio_denoise': (
                    'FLOAT',
                    {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'merge_mode': (['replace', 'multiply', 'minimum', 'maximum'],),
            }
        }

    RETURN_TYPES = ('LATENT',)
    RETURN_NAMES = ('av_latent',)
    FUNCTION = 'apply'
    CATEGORY = CATEGORY_STREAMS

    def apply(self, av_latent, mask, denoise_inside, denoise_outside, audio_denoise, merge_mode):
        return (
            apply_video_inpaint_mask(
                av_latent, mask, denoise_inside, denoise_outside, audio_denoise, merge_mode,
                NestedTensor,
            ),
        )


class MiniMaxH3LatentLabMergeAV:
    DESCRIPTION = (
        'Blend video and audio independently from a source H3 AV latent '
        'into a target H3 AV latent. The target defines output geometry.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'target_av': ('LATENT',),
                'source_av': ('LATENT',),
                'video_mix': (
                    'FLOAT',
                    {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'audio_mix': (
                    'FLOAT',
                    {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'fit_mode': (['strict', 'crop_pad'],),
                'alignment': (['start', 'end', 'center'],),
                'video_denoise': (
                    'FLOAT',
                    {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'audio_denoise': (
                    'FLOAT',
                    {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
            }
        }

    RETURN_TYPES = ('LATENT',)
    RETURN_NAMES = ('av_latent',)
    FUNCTION = 'merge'
    CATEGORY = CATEGORY_STREAMS

    def merge(self, target_av, source_av, video_mix, audio_mix, fit_mode, alignment, video_denoise, audio_denoise):
        return (
            merge_av_latents(
                target_av, source_av, video_mix, audio_mix, fit_mode, alignment,
                video_denoise, audio_denoise, NestedTensor,
            ),
        )


class MiniMaxH3LatentLabPrepareContinuation:
    DESCRIPTION = (
        'Create a new H3 AV latent whose opening is the synchronized tail '
        'of a previous result. Sample it, then use Stitch Continuation to '
        'remove overlap.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'source_av': ('LATENT',),
                'length': (
                    'INT',
                    {'default': 124, 'min': 5, 'max': 3600, 'step': 17},
                ),
                'overlap_frames': (
                    'INT',
                    {
                        'default': 22,
                        'min': 5,
                        'max': 3600,
                        'step': 17,
                        'tooltip': 'Snapped down to the 17*k+5 H3 grid.',
                    },
                ),
                'video_context_denoise': (
                    'FLOAT',
                    {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'audio_context_denoise': (
                    'FLOAT',
                    {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
            }
        }

    RETURN_TYPES = ('LATENT', 'INT', 'INT', 'FLOAT')
    RETURN_NAMES = ('continuation_av', 'frame_count', 'actual_overlap_frames', 'overlap_seconds')
    FUNCTION = 'prepare'
    CATEGORY = CATEGORY_CONTINUATION

    def prepare(self, source_av, length, overlap_frames, video_context_denoise, audio_context_denoise):
        output, frame_count, actual_overlap = prepare_continuation(
            source_av, length, overlap_frames, video_context_denoise, audio_context_denoise,
            NestedTensor,
        )
        return (output, frame_count, actual_overlap, actual_overlap / FPS)


class MiniMaxH3LatentLabStitchContinuation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'previous_av': ('LATENT',),
                'sampled_continuation_av': ('LATENT',),
                'overlap_frames': (
                    'INT',
                    {'default': 22, 'min': 5, 'max': 3600, 'step': 17},
                ),
            },
            'optional': {
                'blend_video_overlap': (
                    'BOOLEAN',
                    {
                        'default': False,
                        'tooltip': 'Smoothstep blend the video overlap seam.',
                    },
                ),
                'offload_to_cpu': (
                    'BOOLEAN',
                    {
                        'default': False,
                        'tooltip': (
                            'Move the stitched result to CPU RAM instead of leaving it '
                            'on the GPU. Safe to enable for long multi-pass runs: this '
                            'accumulator is never read back by the sampler, only by the '
                            'next stitch and the final decode, so keeping it resident on '
                            'the GPU across many passes just wastes VRAM.'
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ('LATENT', 'INT')
    RETURN_NAMES = ('stitched_av', 'total_frames')
    FUNCTION = 'stitch'
    CATEGORY = CATEGORY_CONTINUATION

    def stitch(self, previous_av, sampled_continuation_av, overlap_frames,
               blend_video_overlap=False, offload_to_cpu=False):
        prev_video, prev_audio = unpack_av_samples(previous_av)
        next_video, next_audio = unpack_av_samples(sampled_continuation_av)
        prev_frames = frame_count_from_video_t(prev_video.shape[2])
        next_frames = frame_count_from_video_t(next_video.shape[2])
        result = stitch_continuation(
            previous_av, sampled_continuation_av, overlap_frames, NestedTensor,
            blend_video_overlap, bool(offload_to_cpu),
        )
        stitched_av, reported_total_frames = result
        stitched_video, stitched_audio = unpack_av_samples(stitched_av)
        stitched_frames = frame_count_from_video_t(stitched_video.shape[2])
        expected_audio_t = audio_latent_t(stitched_frames)
        actual_overlap = prev_frames + next_frames - int(reported_total_frames)
        if stitched_frames != int(reported_total_frames):
            raise RuntimeError(
                '[V318 BOUNDARY AUDIT] stitched video frame mismatch: '
                f'actual={stitched_frames}, expected={int(reported_total_frames)}, '
                f'previous={prev_frames}, next={next_frames}, overlap={actual_overlap}'
            )
        if int(stitched_audio.shape[-1]) != int(expected_audio_t):
            raise RuntimeError(
                '[V318 BOUNDARY AUDIT] stitched AV sync mismatch: '
                f'audio_t={int(stitched_audio.shape[-1])}, expected_audio_t={int(expected_audio_t)}, '
                f'frames={stitched_frames}'
            )
        visible_seam_latent = 0
        _lm_print(
            '[MiniMaxH3 LongMedia][V327 PHASE-SAFE BOUNDARY AUDIT] PASS '
            f'previous={prev_frames}f next={next_frames}f overlap={actual_overlap}f '
            f'stitched={stitched_frames}f audio_t={int(stitched_audio.shape[-1])} '
            f'cross_time_blend_latent_t={visible_seam_latent}',
            flush=True,
        )
        if offload_to_cpu:
            _free_cuda_memory()
        return result


class MiniMaxH3LatentLabInfo:
    DESCRIPTION = 'Validate and inspect a MiniMax H3 NestedTensor AV latent.'
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'av_latent': ('LATENT',)}}

    RETURN_TYPES = ('STRING', 'INT', 'INT', 'INT', 'FLOAT', 'BOOLEAN')
    RETURN_NAMES = ('report', 'width', 'height', 'frames', 'duration_seconds', 'synchronized')
    FUNCTION = 'inspect'
    CATEGORY = CATEGORY_UTIL

    def inspect(self, av_latent):
        info = describe_av(av_latent)
        report = json.dumps(info, indent=2, ensure_ascii=False)
        return (
            report, info['width'], info['height'], info['frames'],
            info['duration_seconds'], info['synchronized'],
        )



# -----------------------------------------------------------------------------
# V43/V64: self-contained hybrid conditioning + ref-aware anchor placement
# -----------------------------------------------------------------------------

# Shared Motion Context / Contex Loop marker: target-timeline pixel frame an
# anchor belongs to. The marker-gated PackedLayout helper already ships with
# LongMedia and only moves conditioning rows carrying this key.
MC_ANCHOR_KEY = 'motion_context_index'

def _activate_longmedia_hybrid_support():
    """Activate LongMedia's self-contained keyframe+Ref2VA payload merge.

    This fixes the stock ``cond_video_latents`` last-writer-wins collision.
    Keyframe positions additionally need the marker-gated PackedLayout helper
    whenever refs are packed before the target timeline.
    If Contex Loop/Motion Context already owns the compatible payload merge,
    the shared marker makes this helper stand down cleanly.
    """
    try:
        from . import hybrid_payload_patch
    except Exception:
        import importlib
        pkg = __package__
        if not pkg:
            raise RuntimeError('LongMedia hybrid payload helper could not be imported')
        hybrid_payload_patch = importlib.import_module(pkg + '.hybrid_payload_patch')
    if not hybrid_payload_patch.apply_patch():
        raise RuntimeError(
            'LongMedia hybrid payload merge could not activate. Check the console '
            'for a MiniMaxH3.extra_conds patch collision.'
        )
    return hybrid_payload_patch


def _activate_longmedia_anchor_layout():
    """Keep hybrid first/last anchors on the ref-shifted target timeline.

    Stock H3 places keyframe conditioning rows relative to text_len, while the
    actual target video grid starts after all packed reference rows. Stock H3
    normally keeps FL2VA keyframes and Ref2VA references in separate workflows,
    so it never has to reconcile those origins. LongMedia hybrid/loop combines
    them, therefore refs can otherwise push both anchors into the past.

    The existing marker-gated ``motion_context_layout_patch`` builds marked
    keyframe rows legally, then translates only those rows to the true target
    origin. It is activated only when keyframes and refs coexist.
    """
    try:
        from . import motion_context_layout_patch
    except Exception:
        import importlib
        pkg = __package__
        if not pkg:
            raise RuntimeError('LongMedia anchor layout helper could not be imported')
        motion_context_layout_patch = importlib.import_module(
            pkg + '.motion_context_layout_patch'
        )
    if not motion_context_layout_patch.apply_patch():
        raise RuntimeError(
            'LongMedia keyframe anchors require the PackedLayout patch whenever '
            'references are connected. Check the console for a PackedLayout '
            'patch collision.'
        )
    return motion_context_layout_patch


def _hybrid_encode_ref_audio(audio_vae, audio):
    waveform = audio['waveform']
    sr = int(audio['sample_rate'])
    vae_sr = int(getattr(audio_vae, 'audio_sample_rate', 32000))
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    return z, int(z.shape[-1])


def _build_longmedia_hybrid_conditioning(
    clip, vae, audio_vae, prompt, width, height, length, resolution_mode,
    first_frame=None, last_frame=None, ref_images=None, ref_videos=None,
    ref_audios=None, first_latent_override=None, last_latent_override=None,
):
    """Build H3 keyframes + Ref2VA references in one conditioning payload.

    image_1/image_2 role assignment is handled by Setup; this helper receives
    only the already-separated anchors and reference lists.
    """
    payload_patch = _activate_longmedia_hybrid_support()
    try:
        from comfy_extras.nodes_minimax_h3 import (
            _empty_av_latent, _resize, adapt_canvas,
            REF_IMAGE_SHORT_EDGE, FPS as H3_FPS,
        )
        import node_helpers
    except Exception as exc:
        raise RuntimeError('Current ComfyUI MiniMax H3 helpers are unavailable: %s' % exc)

    latent, frame_count = _empty_av_latent(int(width), int(height), int(length))
    mc_key = getattr(payload_patch, 'LM_KEY', 'longmedia_hybrid_keyframe')
    native_guides = True

    keyframes = []
    keyframe_images = []

    def add_keyframe(frame_index, image, crop, latent_override=None):
        if image is None:
            return None
        img = _resize(image[:1], int(width), int(height), crop)
        resolved = int(frame_index) if native_guides else 0
        entry = {
            'resolved_frame_index': resolved,
            mc_key: True,
            # Target-timeline pixel frame for the anchor. Without refs this is
            # identical to stock placement; with refs the layout patch uses it
            # to compensate for the packed reference span.
            MC_ANCHOR_KEY: int(frame_index),
            'latent': latent_override if latent_override is not None else vae.encode(img),
        }
        keyframes.append(entry)
        keyframe_images.append(img)
        return entry

    first_keyframe = add_keyframe(0, first_frame, 'disabled', first_latent_override)
    add_keyframe(frame_count - 1, last_frame, 'center', last_latent_override)

    ref_items = []
    ref_blocks = []

    for img in ref_images or []:
        if img is None:
            continue
        h, w = int(img.shape[1]), int(img.shape[2])
        if resolution_mode == 'match':
            scale = min(1.0, math.sqrt((int(width) * int(height)) / float(w * h)))
        else:
            scale = min(1.0, float(REF_IMAGE_SHORT_EDGE) / float(min(w, h)))
        tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        resized = _resize(img[:1], tw, th, 'disabled')
        ref_items.append({'type': 'image', 'data': resized})
        ref_blocks.append({
            'kind': 'image', 'latent_h': th // 16, 'latent_w': tw // 16,
            'latent': vae.encode(resized),
        })

    for video_frames in ref_videos or []:
        if video_frames is None:
            continue
        vh, vw = int(video_frames.shape[1]), int(video_frames.shape[2])
        cw, ch = adapt_canvas(vw, vh)
        if vw * vh < cw * ch:
            cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        frames = _resize(video_frames, cw, ch, 'disabled')
        if frames.shape[0] > frame_count:
            frames = frames[:frame_count]
        n = int(frames.shape[0])
        if n < 5:
            raise ValueError('MiniMax H3 reference videos need at least 5 frames.')
        while n % 17 != 5:
            n -= 1
        frames = frames[:n]
        sample_step = max(1, int(H3_FPS) // 2)
        sample_idx = list(range(0, int(frames.shape[0]), sample_step))
        ref_items.append({
            'type': 'video', 'data': frames[sample_idx],
            'timestamps': [i / 2.0 for i in range(len(sample_idx))],
        })
        video_latent = vae.encode(frames)
        ref_blocks.append({
            'kind': 'video', 'latent_t': int(video_latent.shape[2]),
            'latent_h': ch // 16, 'latent_w': cw // 16,
            'ref_audio_t': 0, 'latent': video_latent, 'audio_latent': None,
        })

    for audio in ref_audios or []:
        if audio is None:
            continue
        audio_latent, ref_audio_t = _hybrid_encode_ref_audio(audio_vae, audio)
        ref_items.append({'type': 'audio'})
        ref_blocks.append({
            'kind': 'audio', 'ref_audio_t': ref_audio_t,
            'audio_latent': audio_latent,
        })

    # Refs occupy packed timeline rows before the target video. When first/last
    # anchors coexist with refs, stock keyframe coordinates are therefore early
    # by the total reference span. Activate the marker-gated layout adjustment
    # only for that combined case; graphs without refs remain completely stock.
    if keyframes and ref_blocks:
        _activate_longmedia_anchor_layout()

    # Same semantics as upstream minimax-h3-hybrid-cond: refs use the H3
    # minimax_ref_items tokenizer path; keyframes live in payload guides.
    if ref_items:
        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    else:
        tokens = clip.tokenize(prompt, images=keyframe_images)
    scheduled = getattr(clip, 'encode_from_tokens_scheduled', None)
    cond = _setup_clip_encode_retry(
        lambda: (scheduled(tokens) if callable(scheduled) else clip.encode(tokens)),
        label='hybrid_conditioning',
    )

    values = {}
    if keyframes:
        values['minimax_keyframes'] = keyframes
        values['minimax_frame_count'] = frame_count
    if ref_blocks:
        values['minimax_refs'] = ref_blocks
    if values:
        cond = node_helpers.conditioning_set_values(cond, values)
    return cond, latent, {
        'keyframes': len(keyframes),
        'image_refs': len(ref_images or []),
        'video_refs': len(ref_videos or []),
        'audio_refs': len(ref_audios or []),
        'native_guides': native_guides,
    }, {
        'ref_items': tuple(ref_items),
        'ref_blocks': tuple(ref_blocks),
        'first_keyframe_latent': (
            first_keyframe.get('latent') if first_keyframe is not None else None
        ),
    }



def _v111_build_fixed_clip_specs(total_duration, segment_seconds, overlap_frames, prompt):
    """Build fixed-duration clip specs for the unified clip executor.

    Every generated pass has identical H3-aligned length.  The final pass is
    intentionally generated at full length and trimmed after stitching; this
    keeps tensor geometry, Motion Context, lip-sync slicing and memory behavior
    identical across all passes.
    """
    output_frames = max(1, int(math.floor(float(total_duration) * float(FPS))))
    overlap = int(overlap_frames)
    # segment_seconds keeps its historical meaning: NEW visible timeline.  The
    # fixed clip itself also carries the hidden overlap context, so every pass
    # uses one equal full clip length while its visible stride stays close to
    # segment_seconds (modulo H3 temporal alignment).
    visible_frames = max(1, round(float(segment_seconds) * float(FPS)))
    clip_frames = int(align_frame_count(max(5, int(visible_frames) + overlap)))
    if clip_frames <= overlap:
        raise ValueError(
            f"Fixed segmentation clip length ({clip_frames}f) must be greater than overlap_frames={overlap}."
        )
    step = int(clip_frames - overlap)
    if output_frames <= clip_frames:
        passes = 1
    else:
        passes = 1 + int(math.ceil(float(output_frames - clip_frames) / float(step)))
    if passes > 64:
        raise ValueError(f"Fixed segmentation requires {passes} clips; maximum is 64. Increase segment duration.")
    duration = float(clip_frames) / float(FPS)
    specs = tuple({'prompt': str(prompt or ''), 'duration': duration, 'seed': None} for _ in range(passes))
    lengths = tuple(clip_frames for _ in range(passes))
    starts = tuple(int(i * step) for i in range(passes))
    generated = int(clip_frames + max(0, passes - 1) * step)
    return specs, lengths, starts, generated


def _v85_parse_multiclip_json(raw, fallback_prompt, fallback_duration):
    try:
        payload = json.loads(raw or '[]')
    except Exception as exc:
        raise ValueError(f"MultiClip JSON is invalid: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError('MultiClip JSON must be an array of clip objects.')
    if len(payload) < 2:
        raise ValueError('MultiClip requires at least 2 clips.')
    if len(payload) > 16:
        raise ValueError('MultiClip prototype supports at most 16 clips.')
    clips = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f'MultiClip clip {i+1} must be an object.')
        prompt = str(item.get('prompt') or fallback_prompt or '').strip()
        try:
            duration = float(item.get('duration', fallback_duration))
        except Exception:
            raise ValueError(f'MultiClip clip {i+1} has invalid duration.')
        if duration <= 0.0 or duration > 150.0:
            raise ValueError(f'MultiClip clip {i+1} duration must be >0 and <=150 seconds.')
        seed = item.get('seed', None)
        if seed is not None:
            try:
                seed = int(seed) & 0xFFFFFFFFFFFFFFFF
            except Exception:
                raise ValueError(f'MultiClip clip {i+1} has invalid seed.')
        clips.append({'prompt': prompt, 'duration': duration, 'seed': seed})
    return tuple(clips)


def _v85_multiclip_geometry(clips, overlap_frames):
    overlap = int(overlap_frames)
    lengths = tuple(int(align_frame_count(max(5, round(float(c['duration']) * FPS)))) for c in clips)
    if any(n <= overlap for n in lengths[1:]):
        raise ValueError(f'MultiClip every continuation clip must be longer than overlap_frames={overlap}.')
    starts = [0]
    visible_end = int(lengths[0])
    for n in lengths[1:]:
        starts.append(int(visible_end - overlap))
        visible_end += int(n - overlap)
    return lengths, tuple(starts), int(visible_end)



def _v104_slice_lipsync_guide_audio(source_audio, start_frame, length_frames):
    """Return the exact source waveform window for one local H3 pass.

    The continuation pass begins at the global context_start, not at its visible
    start.  This gives the native H3 Audio Guide the same hidden preroll as the
    video Motion Context while keeping the guide on local frame 0.
    """
    waveform = source_audio['waveform'][:1]
    sr = int(source_audio['sample_rate'])
    start_frame = int(start_frame)
    length_frames = int(length_frames)
    start_sample = int(round(float(start_frame) * float(sr) / float(FPS)))
    end_sample = int(round(float(start_frame + length_frames) * float(sr) / float(FPS)))
    expected = max(1, end_sample - start_sample)
    left_pad = max(0, -start_sample)
    src0 = max(0, start_sample)
    src1 = max(src0, end_sample)
    sliced = waveform[..., src0:src1]
    if left_pad:
        sliced = torch.nn.functional.pad(sliced, (left_pad, 0))
    if int(sliced.shape[-1]) < expected:
        sliced = torch.nn.functional.pad(sliced, (0, expected - int(sliced.shape[-1])))
    elif int(sliced.shape[-1]) > expected:
        sliced = sliced[..., :expected]
    return {'waveform': sliced.contiguous(), 'sample_rate': sr}, start_sample, end_sample


def _v113_lock_source_audio_in_target(target_av, audio_vae, source_audio, start_frame, length_frames):
    """Put the exact source-audio window into the target AV latent and freeze it.

    H3 is a joint audio/video transformer.  Ref2VA audio and AddGuide audio are
    conditioning references; they do not make the target audio stream authoritative.
    For actual speech-driven video we instead expose the source speech as the target
    audio tokens (noise mask 0) while leaving the target video fully denoised.
    """
    if target_av is None or audio_vae is None or source_audio is None:
        return target_av
    sliced, start_sample, end_sample = _v104_slice_lipsync_guide_audio(
        source_audio, int(start_frame), int(length_frames),
    )
    waveform = sliced['waveform']
    sr = int(sliced['sample_rate'])
    vae_sr = int(getattr(audio_vae, 'audio_sample_rate', 32000))
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    audio_latent = audio_vae.encode(waveform[:1].movedim(1, -1))
    locked = replace_stream(
        target_av, {'samples': audio_latent}, 'audio', 'crop_pad', 'start', 0.0, NestedTensor,
    )
    video, audio = unpack_av_samples(locked)
    locked = set_stream_denoise(locked, 1.0, 0.0, 'replace', NestedTensor)
    _lm_print(
        '[MiniMaxH3 LongMedia][0.3.113 LOCKED TARGET AUDIO] '
        f'global={int(start_frame)}f length={int(length_frames)}f '
        f'samples={start_sample}..{end_sample} target_audio_t={int(audio.shape[-1])}; '
        'video_denoise=1 audio_denoise=0',
        flush=True,
    )
    return locked


def _v104_attach_native_lipsync_guide(positive, audio_vae, source_audio, plan, segment_index):
    """Add a stock-H3-style audio keyframe at local frame 0 for one pass.

    Important: Audio 1 remains a normal Ref2VA <Audio 1> reference as well.
    The reference conveys speech/music identity/content; this guide supplies the
    pass-local time anchor.  No target-audio freezing or custom latent resampling.
    """
    if source_audio is None or audio_vae is None:
        return positive
    timeline = _segment_timeline_contract(plan, int(segment_index))
    start_frame = int(timeline['context_start'])
    length_frames = int(timeline['length_frames'])
    sliced, start_sample, end_sample = _v104_slice_lipsync_guide_audio(
        source_audio, start_frame, length_frames,
    )
    audio_latent, _ = _hybrid_encode_ref_audio(audio_vae, sliced)
    max_rt = int(audio_latent_t(length_frames))
    if int(audio_latent.shape[-1]) > max_rt:
        audio_latent = audio_latent[..., :max_rt].clone()

    def patch_meta(meta):
        meta = dict(meta)
        keyframes = [dict(kf) for kf in (meta.get('minimax_keyframes', []) or [])
                     if not bool(kf.get('longmedia_lipsync_audio_guide'))]
        keyframes.append({
            'resolved_frame_index': 0,
            'audio_latent': audio_latent,
            'longmedia_lipsync_audio_guide': True,
            'longmedia_audio_context_start': start_frame,
        })
        meta['minimax_keyframes'] = keyframes
        meta.pop('minimax_frame_count', None)
        return meta

    out=[]
    attached=False
    for entry in (positive or []):
        if isinstance(entry, dict):
            out.append(patch_meta(entry)); attached=True
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], dict):
            ne=list(entry); ne[1]=patch_meta(entry[1]); out.append(ne); attached=True
        else:
            out.append(entry)
    if not attached:
        raise RuntimeError('lip_sync: native H3 Audio Guide could not attach to conditioning metadata')
    _lm_print(
        '[MiniMaxH3 LongMedia][0.3.104 LIP SYNC GUIDE] '
        f'clip={int(segment_index)+1} local=0f global={start_frame}f '
        f'visible={int(timeline["visible_start"])}f length={length_frames}f '
        f'samples={start_sample}..{end_sample} latent_t={int(audio_latent.shape[-1])}; '
        'Audio1 remains native Ref2VA reference',
        flush=True,
    )
    return out




def _v107_attach_visible_lipsync_guide(positive, audio_vae, source_audio, plan, segment_index):
    """Anchor only the NEW visible source-audio span on continuation passes.

    Full Audio1 remains the untouched native Ref2VA reference on every clip.
    The hidden overlap is owned by Extender-style AV Motion Context (video+audio
    tail from the previous sampled AV latent).  This guide starts exactly at the
    first visible local frame, so the two temporal conditions do not overlap.
    """
    idx = int(segment_index)
    if idx <= 0 or source_audio is None or audio_vae is None:
        return positive
    timeline = _segment_timeline_contract(plan, idx)
    mark_in = int(timeline['visible_start'])
    visible_frames = int(timeline['visible_frames'])
    local_in = int(timeline['local_visible_offset'])
    waveform = source_audio['waveform'][:1]
    sr = int(source_audio['sample_rate'])
    start_sample = int(round(float(mark_in) * float(sr) / float(FPS)))
    end_sample = int(round(float(mark_in + visible_frames) * float(sr) / float(FPS)))
    expected = max(1, end_sample - start_sample)
    sliced = waveform[..., max(0, start_sample):max(0, end_sample)]
    if int(sliced.shape[-1]) < expected:
        sliced = torch.nn.functional.pad(sliced, (0, expected - int(sliced.shape[-1])))
    elif int(sliced.shape[-1]) > expected:
        sliced = sliced[..., :expected]
    audio = {'waveform': sliced.contiguous(), 'sample_rate': sr}
    audio_latent, encoded_t = _hybrid_encode_ref_audio(audio_vae, audio)

    # Match stock MiniMaxH3AddGuide: guide audio may occupy only the remaining
    # target-audio timeline after resolved_frame_index.
    target_audio_t = int(audio_latent_t(int(timeline['length_frames'])))
    max_rt = max(1, int(math.floor(float(target_audio_t) -
                                   (float(AUDIO_LATENT_FPS) / float(FPS)) * float(local_in))))
    if int(audio_latent.shape[-1]) > max_rt:
        audio_latent = audio_latent[..., :max_rt].clone()

    def patch(meta):
        meta = dict(meta)
        keyframes = [dict(kf) for kf in (meta.get('minimax_keyframes', []) or [])
                     if not bool(kf.get('longmedia_v107_visible_lipsync_guide'))]
        keyframes.append({
            'resolved_frame_index': int(local_in),
            'audio_latent': audio_latent,
            'longmedia_v107_visible_lipsync_guide': True,
            'longmedia_global_visible_start': int(mark_in),
        })
        meta['minimax_keyframes'] = keyframes
        meta.pop('minimax_frame_count', None)
        return meta

    out=[]; attached=False
    for entry in (positive or []):
        if isinstance(entry, dict):
            out.append(patch(entry)); attached=True
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], dict):
            ne=list(entry); ne[1]=patch(entry[1]); out.append(ne); attached=True
        else:
            out.append(entry)
    if not attached:
        raise RuntimeError('lip_sync: visible native H3 Audio Guide could not attach')
    _lm_print(
        '[MiniMaxH3 LongMedia][0.3.107 VISIBLE LIP SYNC GUIDE] '
        f'clip={idx+1} local_start={local_in}f global_start={mark_in}f '
        f'visible_frames={visible_frames}f samples={start_sample}..{end_sample} '
        f'encoded_ref_t={int(encoded_t)} guide_t={int(audio_latent.shape[-1])} '
        f'target_audio_t={target_audio_t}; hidden 0..{local_in-1}f remains AV Motion Context only',
        flush=True,
    )
    return out

def _v85_preencode_multiclip_conditionings(clip, positive, plan, prompts, v329_native_refs=None, lip_sync_audio=None, audio_vae=None):
    import comfy.sampler_helpers
    # pass 0 is attached once in Setup because the external guider samples that
    # exact CONDITIONING object. Continuation passes are attached below.
    raw = [positive]
    for idx in range(1, len(prompts)):
        text = str(prompts[idx])
        if v329_native_refs is not None:
            ref_items, ref_blocks = v329_native_refs
            encoded = _v329_encode_continuation_native_refs(
                clip, text, positive, plan, idx, ref_items, ref_blocks,
            )
        else:
            encoded = _encode_prompt(clip, text)
            encoded = _v57_attach_minimax_metadata(
                encoded, positive, plan, idx, drop_image_refs=False,
            )
        if lip_sync_audio is not None:
            encoded = _v104_attach_native_lipsync_guide(
                encoded, audio_vae, lip_sync_audio, plan, idx,
            )
        raw.append(encoded)
    converted = tuple(comfy.sampler_helpers.convert_cond(cond) for cond in raw)
    _lm_print(
        '[MiniMaxH3 LongMedia][0.3.108 MULTICLIP LIP SYNC CONDITIONING] '
        f'pre-encoded {len(converted)} clips; shared native refs preserved; lip_sync_guide={bool(lip_sync_audio is not None)}',
        flush=True,
    )
    return converted, tuple(str(x) for x in prompts)


def _v85_segment_seed(plan, base_seed, segment_index):
    idx = int(segment_index)
    seeds = getattr(plan, 'segment_seeds', None)
    if seeds and idx < len(seeds) and seeds[idx] is not None:
        return int(seeds[idx]) & 0xFFFFFFFFFFFFFFFF
    return (int(base_seed) + idx) & 0xFFFFFFFFFFFFFFFF



class MiniMaxH3LongMediaPlanner:
    """User-facing MultiClip script/timeline planner.

    The frontend renders clip cards; Python receives one stable JSON widget and
    emits an opaque H3_LONGMEDIA_CLIP_PLAN consumed by Long Media Setup.
    """

    DESCRIPTION = (
        'Build a MultiClip plan with one prompt, duration, and optional seed per clip. '
        'Connect clip_plan to Long Media Setup; the Setup automatically uses MultiClip mode.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'clips_json': ('STRING', {
                    'default': '[{"prompt":"","duration":7.5,"seed":null},{"prompt":"","duration":7.5,"seed":null}]',
                    'multiline': True,
                    'tooltip': 'Internal serialized clip-card state. The LongMedia Planner frontend hides this field and renders clip cards instead.',
                }),
            },
        }

    RETURN_TYPES = ('H3_LONGMEDIA_CLIP_PLAN', 'INT', 'FLOAT', 'STRING')
    RETURN_NAMES = ('clip_plan', 'clip_count', 'requested_seconds', 'report')
    FUNCTION = 'build'
    CATEGORY = CATEGORY_LONGMEDIA

    def build(self, clips_json):
        clips = _v85_parse_multiclip_json(clips_json, '', 7.5)
        normalized = []
        for idx, clip in enumerate(clips):
            normalized.append({
                'prompt': str(clip.get('prompt') or ''),
                'duration': float(clip.get('duration') or 7.5),
                'seed': None if clip.get('seed') is None else int(clip.get('seed')),
            })
        if len(normalized) < 2:
            raise ValueError('MiniMax H3 LongMedia Planner requires at least 2 clips.')
        requested = sum(float(c['duration']) for c in normalized)
        plan = {
            'version': 1,
            'kind': 'h3_longmedia_clip_plan',
            'source': 'MiniMax H3 LongMedia Planner',
            'clips': normalized,
        }
        report = json.dumps({
            'clip_count': len(normalized),
            'requested_seconds': requested,
            'clips': normalized,
        }, indent=2)
        return (plan, int(len(normalized)), float(requested), report)


class MiniMaxH3LatentLabLongMediaSetup:
    DESCRIPTION = (
        'Orchestrate long-media generation: build a multi-segment plan, '
        'encode source media, and set up references for NativeReferenceToVideo.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'clip': ('CLIP',),
                'vae': ('VAE',),
                'audio_vae': ('VAE',),
                'prompt': (
                    'STRING',
                    {
                        'default': '',
                        'multiline': True,
                        'tooltip': 'Prompt text. Can also be connected directly through the prompt input socket.',
                    },
                ),
                'width': (
                    'INT',
                    {'default': 512, 'min': 32, 'max': 8192, 'step': 32},
                ),
                'height': (
                    'INT',
                    {'default': 512, 'min': 32, 'max': 8192, 'step': 32},
                ),
                'manual_duration': (
                    'FLOAT',
                    {'default': 5.0, 'min': 0.1, 'max': 600.0, 'step': 0.1},
                ),
                'duration_source': (['auto', 'manual', 'audio', 'video', 'longest_input'],),
                'segment_seconds': (
                    'FLOAT',
                    {
                        'default': 8.0, 'min': 1.0, 'max': 60.0, 'step': 0.5,
                        'tooltip': 'New output timeline per segment. overlap_frames is added as continuation context and does not reduce this duration.',
                    },
                ),
                'overlap_frames': (
                    'INT',
                    {'default': 22, 'min': 5, 'max': 3600, 'step': 17},
                ),
                'resolution_mode': (['match', 'max'],),
                'reference_budget': (['low', 'medium', 'high', 'max'],),
                'video_fps': (
                    'FLOAT',
                    {'default': 24.0, 'min': 1.0, 'max': 120.0, 'step': 1.0},
                ),
                'video_mode': (['auto', 'preserve', 'transform'],),
                'audio_mode': (['auto', 'preserve', 'generate', 'reference_only', 'preserve_reference', 'lip_sync'], {'tooltip': 'auto: legacy behavior. preserve: restore original audio at output without using it as a reference when possible. generate: generate final H3 audio. reference_only: use input audio as H3 reference but output generated audio. preserve_reference: use input audio as H3 timing/rhythm reference and restore the untouched original track. lip_sync: audio_1 stays native <Audio 1> Ref2VA content conditioning and is also time-anchored per clip with the native H3 Audio Guide; the untouched audio_1 is restored at output.'}),
                'conditioning_mode': (
                    ['auto_refs', 'hybrid_first_frame', 'hybrid_first_last'],
                    {
                        'default': 'auto_refs',
                        'tooltip': (
                            'auto_refs keeps the original LongMedia behavior. '
                            'hybrid_first_frame: image_1 is the opening keyframe, image_2..image_9 become '
                            '<Picture 1>..<Picture 8> identity/style refs. '
                            'hybrid_first_last: image_1 is the first keyframe, image_2 is the last keyframe, '
                            'and image_3..image_9 become <Picture 1>..<Picture 7> refs. '
                            'Manual mode only. auto_refs sends all connected images as Picture refs. hybrid_first_frame uses image_1 as the opening keyframe. hybrid_first_last uses image_1/image_2 as first/last keyframes. video_1..3 and audio_1..3 remain native H3 refs.'
                        ),
                    },
                ),
            },
            'optional': {
                'clip_plan': ('H3_LONGMEDIA_CLIP_PLAN', {
                    'tooltip': 'Connect MiniMax H3 LongMedia Planner. It is authoritative only when workflow_mode=multiclip; all other workflows ignore the connected Planner.',
                }),
                'workflow_mode': (
                    ['hybrid_auto', 'segmented_continuation', 'multiclip', 'ref2va_full', 'loop', 'manual', 'video_ref_edit'],
                    {
                        'default': 'hybrid_auto',
                        'tooltip': (
                            'hybrid_auto: image_1 is first frame; if image_2 is connected it is last frame; '
                            'segmented_continuation: fixed-duration timeline policy using the exact same Ref2VA/Motion-Context clip executor as MultiClip; every generated clip has the same H3-aligned length and the final excess is trimmed. segment_duration controls fixed clip size; Planner is ignored. '
                            'remaining images are Picture refs. video_ref_edit: video_1 is the main motion/camera/composition '
                            'reference, image_1..9 are Picture refs for identity/style replacement, and audio_1 can carry the '
                            'paired source soundtrack. ref2va_full: all connected images are Picture refs with no first/last '
                            'anchors. loop: image_1 is reused as BOTH first and last frame for a seam-friendly viral loop; '
                            'image_2..9 are Picture refs. manual: exposes legacy conditioning and segmentation controls for '
                            'advanced diagnostics and A/B tests.'
                        ),
                    },
                ),
                'multiclip_json': ('STRING', {
                    'default': '[{"prompt":"","duration":7.5,"seed":null},{"prompt":"","duration":7.5,"seed":null}]',
                    'multiline': True,
                    'tooltip': 'MultiClip only. JSON array: [{prompt, duration, seed}, ...]. Blank prompt inherits main prompt; null seed uses sampler seed + clip index.',
                }),
                'image_1': ('IMAGE', {'tooltip': 'Native MiniMax H3 <Picture 1> reference.'}),
                'image_2': ('IMAGE', {'lazy': True, 'tooltip': 'Native MiniMax H3 <Picture 2> reference.'}),
                'image_3': ('IMAGE', {'lazy': True, 'tooltip': 'Native MiniMax H3 <Picture 3> reference.'}),
                'image_4': ('IMAGE', {'lazy': True, 'tooltip': 'Native MiniMax H3 <Picture 4> reference.'}),
                'image_5': ('IMAGE', {'lazy': True, 'tooltip': 'Native MiniMax H3 <Picture 5> reference.'}),
                'image_6': ('IMAGE', {'lazy': True, 'tooltip': 'Native MiniMax H3 <Picture 6> reference.'}),
                'image_7': ('IMAGE', {'lazy': True, 'tooltip': 'Native MiniMax H3 <Picture 7> reference.'}),
                'image_8': ('IMAGE', {'lazy': True, 'tooltip': 'Native MiniMax H3 <Picture 8> reference.'}),
                'image_9': ('IMAGE', {'lazy': True, 'tooltip': 'Native MiniMax H3 <Picture 9> reference.'}),
                'video_1': ('IMAGE', {'lazy': True, 'tooltip': 'Video frames only (IMAGE batch). video_ref_edit: primary motion/camera/composition source as <Video 1>. Other modes: regular <Video 1> reference. If the source video has audio, connect that extracted audio to audio_1.'}),
                'video_2': ('IMAGE', {'lazy': True, 'tooltip': 'Video frames only (IMAGE batch). Passed as <Video 2> reference. Pair with audio_2 when they come from the same source.'}),
                'video_3': ('IMAGE', {'lazy': True, 'tooltip': 'Video frames only (IMAGE batch). Passed as <Video 3> reference. Pair with audio_3 when they come from the same source.'}),
                'audio_1': ('AUDIO', {'tooltip': 'Native MiniMax H3 <Audio 1> reference. With audio_mode=lip_sync it remains <Audio 1> and also drives a native per-clip H3 Audio Guide; the untouched source is restored at output.'}),
                'audio_2': ('AUDIO', {'lazy': True, 'tooltip': 'Optional second audio reference. Passed as <Audio 2>; pair with video_2 by convention when they come from the same source.'}),
                'audio_3': ('AUDIO', {'lazy': True, 'tooltip': 'Optional third audio reference. Passed as <Audio 3>; pair with video_3 by convention when they come from the same source.'}),
            },
        }

    RETURN_TYPES = ('CONDITIONING', 'LATENT', 'LONG_MEDIA_PLAN', 'FLOAT', 'INT', 'STRING')
    RETURN_NAMES = ('positive', 'long_media_av', 'long_media_plan', 'duration_seconds', 'passes', 'report')
    FUNCTION = 'setup'
    CATEGORY = CATEGORY_LONGMEDIA

    @classmethod
    def check_lazy_status(cls, reference_budget='low', video_mode='auto', audio_mode='auto',
                          duration_source='auto', generation_mode='auto', **kwargs):
        # Only request inputs that are actually connected in the graph (present in kwargs).
        # Never request an unconnected input — ComfyUI crashes with NodeInputError.
        candidates = [
            'image_1', 'image_2', 'image_3', 'image_4', 'image_5', 'image_6', 'image_7', 'image_8', 'image_9',
            'video_1', 'video_2', 'video_3',
            'audio_1', 'audio_2', 'audio_3',
        ]
        return [name for name in candidates if name in kwargs]

    def setup(self, clip, vae, audio_vae, prompt, width, height, manual_duration,
              duration_source, segment_seconds, overlap_frames, resolution_mode,
              reference_budget, video_fps, video_mode, audio_mode,
              clip_plan=None, workflow_mode='hybrid_auto', generation_mode='auto', conditioning_mode='auto_refs',
              first_frame_mode='latent_inject',
              first_frame_denoise=0.25, first_frame_blend_frames=3,
              opening_frame=None, multiclip_json=None,
              image_1=None, image_2=None, image_3=None, image_4=None, image_5=None,
              image_6=None, image_7=None, image_8=None, image_9=None,
              video_1=None, video_2=None, video_3=None,
              audio_1=None, audio_2=None, audio_3=None):
        global NativeReferenceToVideo

        _set_longmedia_release_guard(True)
        setup_memory_events = []
        # Start Setup from a clean model residency state.  This is especially
        # important when re-running a workflow after H3 occupied most of VRAM.
        setup_memory_events.append(_setup_memory_isolation('setup_entry', unload_models=True))

        effective_prompt = prompt
        segment0_prompt = None
        hybrid_artifacts = None
        audio_mode = str(audio_mode or 'auto')
        # v0.3.95: lip-sync is an audio policy, not a separate generation workflow.
        # Migrate legacy workflows that still carry generation_mode=lip_sync.
        if str(generation_mode or 'auto') == 'lip_sync' and audio_mode != 'lip_sync':
            audio_mode = 'lip_sync'
        lip_sync_enabled = audio_mode == 'lip_sync'
        preserve_audio_output = audio_mode in ('preserve', 'preserve_reference')
        use_audio_as_reference = audio_mode != 'preserve'

        images = [v for v in [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9] if v is not None]
        segmented_opening_frame = opening_frame
        videos = [v for v in [video_1, video_2, video_3] if v is not None]
        audios = [a for a in [audio_1, audio_2, audio_3] if a is not None]
        if lip_sync_enabled:
            if image_1 is None or audio_1 is None:
                missing = []
                if image_1 is None:
                    missing.append('image_1')
                if audio_1 is None:
                    missing.append('audio_1')
                raise ValueError(
                    "audio_mode='lip_sync' requires connected image_1 and audio_1; missing: "
                    + ', '.join(missing)
                )
            # One authoritative speech driver. Extra audio sockets must not compete
            # with timing/lip conditioning or contaminate the clean output track.
            audios = [audio_1]

        # v0.3.49: connected INT nodes can bypass the UI's step=32 constraint.
        # Normalize every target canvas here so all downstream H3 paths see the
        # same patch-safe geometry. This is a no-op for existing 32px canvases.
        width, height, h3_target_geometry = _h3_safe_target_canvas(width, height)
        safe_images, h3_ref_geometry = _h3_safe_reference_images(images)
        safe_image_by_id = {id(src): safe for src, safe in zip(images, safe_images)}

        # v0.3.13: segmentation is available in every public workflow mode.
        # segment_seconds is the NEW visible timeline produced per pass; overlap_frames
        # is additional hidden continuation context and does not reduce that duration.
        workflow_mode = str(workflow_mode or 'hybrid_auto')
        external_clip_plan = clip_plan if isinstance(clip_plan, dict) else None
        if workflow_mode == 'multiclip' and external_clip_plan is not None:
            kind = str(external_clip_plan.get('kind') or '')
            version = int(external_clip_plan.get('version', 0) or 0)
            clips_payload = external_clip_plan.get('clips')
            if kind != 'h3_longmedia_clip_plan' or version != 1 or not isinstance(clips_payload, list):
                raise ValueError('Invalid H3_LONGMEDIA_CLIP_PLAN payload. Rebuild it with MiniMax H3 LongMedia Planner.')
            multiclip_json = json.dumps(clips_payload, ensure_ascii=False)
            _lm_print(
                f'[MiniMaxH3 LongMedia][0.3.91 PLANNER] workflow=multiclip; external clip_plan selected; clips={len(clips_payload)}',
                flush=True,
            )
        elif external_clip_plan is not None:
            _lm_print(
                f'[MiniMaxH3 LongMedia][0.3.91 PLANNER] clip_plan connected but workflow={workflow_mode}; planner is ignored',
                flush=True,
            )
        _lm_print(
            f'[MiniMaxH3 LongMedia][0.3.109 WORKFLOW OWNERSHIP] selected={workflow_mode}; '
            f'planner_connected={external_clip_plan is not None}; '
            f'planner_active={bool(workflow_mode == "multiclip" and external_clip_plan is not None)}',
            flush=True,
        )
        loop_last_override = None
        multiclip_clips = None
        if workflow_mode == 'hybrid_auto':
            if image_1 is not None and image_2 is not None:
                conditioning_mode = 'hybrid_first_last'
            elif image_1 is not None:
                conditioning_mode = 'hybrid_first_frame'
            else:
                conditioning_mode = 'auto_refs'
        elif workflow_mode == 'segmented_continuation':
            # v0.3.111: fixed segmentation is no longer a separate continuation
            # engine. It uses the exact MultiClip Ref2VA + Motion Context executor;
            # only the timeline planner differs (fixed equal-length clips).
            conditioning_mode = 'multiclip_ref2va'
        elif workflow_mode == 'multiclip':
            # 0.3.92: true Extender-style MultiClip semantics. MultiClip is not a
            # hybrid-first-frame workflow: every connected image remains a native
            # Ref2VA <Picture N> reference on every clip, and each clip starts from
            # a fresh AV latent. Native Motion Context is added only for clip 2+.
            conditioning_mode = 'multiclip_ref2va'
            multiclip_clips = _v85_parse_multiclip_json(multiclip_json, prompt, manual_duration)
            effective_prompt = multiclip_clips[0]['prompt']
        elif workflow_mode == 'video_ref_edit':
            if not videos:
                raise ValueError("workflow_mode='video_ref_edit' requires at least video_1. Connect the source clip frames to video_1.")
            conditioning_mode = 'auto_refs'
        elif workflow_mode == 'ref2va_full':
            conditioning_mode = 'auto_refs'
        elif workflow_mode == 'loop':
            if image_1 is None:
                raise ValueError("workflow_mode='loop' requires image_1. image_1 is copied internally into both first and last frame anchors.")
            conditioning_mode = 'hybrid_first_last'
            loop_last_override = image_1
        elif workflow_mode != 'manual':
            raise ValueError(f'Unknown workflow_mode={workflow_mode!r}')

        effective_segment_seconds = float(segment_seconds)
        effective_overlap_frames = int(overlap_frames)
        effective_manual_duration = float(manual_duration)
        effective_duration_source = duration_source
        if workflow_mode == 'multiclip':
            effective_manual_duration = float(multiclip_clips[0]['duration'])
            effective_segment_seconds = float(multiclip_clips[0]['duration'])
            effective_duration_source = 'manual'

        plan = build_media_plan(
            audios=audios,
            videos=videos,
            manual_duration=effective_manual_duration,
            duration_source=effective_duration_source,
            segment_seconds=effective_segment_seconds,
            overlap_frames=effective_overlap_frames,
            video_fps=float(video_fps),
            resolution_mode=resolution_mode,
        )

        if workflow_mode == 'segmented_continuation':
            multiclip_clips, fixed_lengths, fixed_starts, fixed_generated = _v111_build_fixed_clip_specs(
                float(plan.total_duration), float(effective_segment_seconds), int(plan.overlap_frames), effective_prompt,
            )
            plan = _dc_replace(
                plan,
                mode='multiclip', duration_basis=f'{plan.duration_basis}:fixed_segments',
                segment_frames=int(fixed_lengths[0]), segment_lengths=tuple(fixed_lengths),
                segment_starts=tuple(fixed_starts), overlap_frames=int(plan.overlap_frames),
                step_frames=max(1, int(fixed_lengths[0]) - int(plan.overlap_frames)),
                passes=len(fixed_lengths), generated_frames=int(fixed_generated),
                trim_frames=max(0, int(fixed_generated) - int(plan.output_frames)),
                segment_seeds=tuple(None for _ in fixed_lengths), timeline_policy='fixed',
            )
            _lm_print(
                '[MiniMaxH3 LongMedia][0.3.111 UNIFIED CLIP ENGINE] '
                f'workflow=segmented_continuation timeline=fixed clips={len(fixed_lengths)} '
                f'length={int(fixed_lengths[0])}f starts={list(fixed_starts)} overlap={int(plan.overlap_frames)}f '
                f'final={int(plan.output_frames)}f generated={int(fixed_generated)}f trim={int(plan.trim_frames)}f; '
                'executor=multiclip',
                flush=True,
            )

        if workflow_mode == 'multiclip':
            mc_lengths, mc_starts, mc_output_frames = _v85_multiclip_geometry(multiclip_clips, effective_overlap_frames)
            plan = _dc_replace(
                plan,
                mode='multiclip', duration_basis='multiclip',
                total_duration=float(mc_output_frames) / float(FPS), output_frames=int(mc_output_frames),
                segment_frames=int(mc_lengths[0]), segment_lengths=tuple(mc_lengths), segment_starts=tuple(mc_starts),
                overlap_frames=int(effective_overlap_frames), step_frames=max(1, int(mc_lengths[0]) - int(effective_overlap_frames)),
                passes=len(mc_lengths), generated_frames=int(mc_output_frames), trim_frames=0,
                segment_seeds=tuple(c['seed'] for c in multiclip_clips), timeline_policy='planned',
            )
            _lm_print(
                '[MiniMaxH3 LongMedia][0.3.87 MULTICLIP PLAN] '
                f'clips={len(mc_lengths)} lengths={list(mc_lengths)} starts={list(mc_starts)} '
                f'overlap={int(effective_overlap_frames)} final={int(mc_output_frames)}f',
                flush=True,
            )
        mode = plan.mode
        if workflow_mode != 'manual':
            _lm_print(
                f'[MiniMaxH3 LongMedia][0.3.13 MODE] workflow={workflow_mode} conditioning={conditioning_mode} '
                f'passes={int(plan.passes)} segment_duration={float(effective_segment_seconds):.3f}s '
                f'overlap={int(plan.overlap_frames)}f',
                flush=True,
            )
            if workflow_mode == 'segmented_continuation':
                _lm_print(
                    '[MiniMaxH3 LongMedia][0.3.111 FIXED TIMELINE] '
                    'fixed segmentation uses the shared MultiClip executor; only clip-boundary math differs',
                    flush=True,
                )
            if workflow_mode == 'video_ref_edit':
                _lm_print('[MiniMaxH3 LongMedia][0.3.0 VIDEO EDIT] video_1 drives motion/camera/staging; image_1..9 stay as <Picture N> replacement refs; audio_1 may be the paired source soundtrack', flush=True)

        # PR#3 compatibility fix: when image refs are connected together with
        # audio but there is no source video, keep the NativeReferenceToVideo
        # route instead of audio_to_video. The audio_to_video path ignores
        # image refs and creates a 1x1 spatial video latent.
        if (
            mode == 'audio_to_video'
            and conditioning_mode == 'auto_refs'
            and len(images) > 0
            and len(videos) == 0
        ):
            mode = 't2v'

        target_av = None

        if NativeReferenceToVideo is None:
            NativeReferenceToVideo = _resolve_native_reference_to_video()

        hybrid_info = None
        if lip_sync_enabled:
            # Orthogonal policy: keep the selected workflow/conditioning family,
            # but make audio_1 an explicit speech/lip driver for every local pass.
            effective_prompt = _build_lipsync_prompt(
                effective_prompt, plan, image_1 is not None, audio_1 is not None
            )
            generation_mode = 'auto'
            _lm_print(
                '[MiniMaxH3 LongMedia][0.3.95 LIP SYNC] audio_mode=lip_sync; '
                'workflow preserved; Audio 1 remains native Ref2VA content reference + native per-clip H3 Audio Guide timing',
                flush=True,
            )

        # Every generation route must encode pass 0 from the same global-to-local
        # timeline policy used by continuation passes.  Before v0.3.29 pass 0 saw
        # future events (for example a 07 sec kiss inside a 5 sec segment), while
        # pass 1 saw shifted local events and therefore replayed the action.
        segment0_prompt = _v57_build_segment_prompt(effective_prompt, plan, 0)

        if conditioning_mode == 'multiclip_ref2va':
            # Extender parity: all image/video/audio references are shared native
            # Ref2VA payload on every clip. No startup keyframe consumes image_1.
            multiclip_ref_images = [safe_image_by_id.get(id(v), v) for v in
                                    [image_1, image_2, image_3, image_4, image_5,
                                     image_6, image_7, image_8, image_9] if v is not None]
            multiclip_ref_videos = [v for v in [video_1, video_2, video_3] if v is not None]
            multiclip_ref_audios = ([v for v in [audio_1, audio_2, audio_3] if v is not None]
                                    if use_audio_as_reference else [])
            setup_memory_events.append(
                _setup_memory_isolation('before_multiclip_ref2va_conditioning', unload_models=True)
            )
            positive, target_av, hybrid_info, hybrid_artifacts = _build_longmedia_hybrid_conditioning(
                clip=clip, vae=vae, audio_vae=audio_vae,
                prompt=segment0_prompt, width=width, height=height,
                length=plan.segment_lengths[0], resolution_mode=resolution_mode,
                first_frame=None, last_frame=None,
                ref_images=multiclip_ref_images,
                ref_videos=multiclip_ref_videos,
                ref_audios=multiclip_ref_audios,
                first_latent_override=None, last_latent_override=None,
            )
            hybrid_info.update({
                'multiclip_ref2va': True,
                'shared_picture_refs': len(multiclip_ref_images),
                'continuation_reference_policy': 'extender_shared_ref2va_every_clip',
            })
            plan = _dc_replace(
                plan, mode='multiclip', source_video=None, source_audio=None,
                reference_audio=(audio_1 if (use_audio_as_reference and audio_1 is not None) else None),
                final_audio_override=(_mix_audio_tracks(audios) if (preserve_audio_output and audios) else None),
                final_audio_track_count=(len(audios) if preserve_audio_output else 0),
                first_frame_override=None, first_frame_latent_injected=False,
                audio_vae=audio_vae, video_vae=vae,
            )
            _lm_print(
                '[MiniMaxH3 LongMedia][0.3.92 EXTENDER REF2VA PARITY] '
                f'all {len(multiclip_ref_images)} image refs remain native Picture refs on every clip; '
                'no first-frame anchor; fresh AV target per clip + native Motion Context for clip 2+',
                flush=True,
            )
            setup_memory_events.append(
                _setup_memory_isolation('after_multiclip_ref2va_conditioning_release', unload_models=True)
            )

        elif conditioning_mode == 'storyboard_bridge':
            if image_1 is None or image_2 is None or image_3 is None:
                raise ValueError(
                    'storyboard_bridge V64 requires image_1=panel A, image_2=shared panel B, image_3=panel C.'
                )
            if int(plan.passes) != 2:
                raise ValueError(
                    f'storyboard_bridge V64 requires exactly 2 passes; current plan has {int(plan.passes)}. '
                    'Use manual_duration=10 and segment_seconds=5 for the first test.'
                )
            nominal = max(5, int(math.floor(float(segment_seconds) * FPS)))
            first_len = int(align_frame_count(nominal))
            remaining = max(5, int(plan.output_frames) - first_len + 1)
            second_len = int(align_frame_count(remaining))
            plan = _dc_replace(
                plan, mode='storyboard_bridge', overlap_frames=0,
                segment_lengths=(first_len, second_len), segment_starts=(0, nominal),
                segment_frames=first_len, generated_frames=first_len + second_len - 1,
                trim_frames=max(0, first_len + second_len - 1 - int(plan.output_frames)), passes=2,
                source_video=None, source_audio=None, reference_audio=None,
                final_audio_override=None, final_audio_track_count=0,
            )
            try:
                from comfy_extras.nodes_minimax_h3 import _resize as _h3_resize
            except Exception as exc:
                raise RuntimeError(f'Current ComfyUI MiniMax H3 resize helper unavailable: {exc}')

            # V64 true storyboard: every panel is fitted ONCE and encoded ONCE.
            # Panel B is therefore bit-identical on both sides of the join.
            panel_a = _h3_resize(image_1[:1], int(width), int(height), 'center')
            panel_b = _h3_resize(image_2[:1], int(width), int(height), 'center')
            panel_c = _h3_resize(image_3[:1], int(width), int(height), 'center')
            panel_a_lat = vae.encode(panel_a)
            panel_b_lat = vae.encode(panel_b)
            panel_c_lat = vae.encode(panel_c)

            # image_1..3 are storyboard panels, never Ref2VA images in this mode.
            storyboard_refs = [v for v in [image_4, image_5, image_6, image_7, image_8, image_9] if v is not None]
            storyboard_videos = [v for v in [video_1, video_2, video_3] if v is not None]
            storyboard_audios = [v for v in [audio_1, audio_2, audio_3] if v is not None]
            setup_memory_events.append(_setup_memory_isolation('before_storyboard_conditioning', unload_models=True))

            pass0_prompt = segment0_prompt
            positive, target_av, sb0, _sb0_artifacts = _build_longmedia_hybrid_conditioning(
                clip=clip, vae=vae, audio_vae=audio_vae, prompt=pass0_prompt,
                width=width, height=height, length=first_len, resolution_mode=resolution_mode,
                first_frame=panel_a, last_frame=panel_b, ref_images=storyboard_refs,
                ref_videos=storyboard_videos, ref_audios=storyboard_audios,
                first_latent_override=panel_a_lat, last_latent_override=panel_b_lat,
            )
            pass1_prompt = _v57_build_segment_prompt(effective_prompt, plan, 1)
            positive_1, target_av_1, sb1, _sb1_artifacts = _build_longmedia_hybrid_conditioning(
                clip=clip, vae=vae, audio_vae=audio_vae, prompt=pass1_prompt,
                width=width, height=height, length=second_len, resolution_mode=resolution_mode,
                first_frame=panel_b, last_frame=panel_c, ref_images=storyboard_refs,
                ref_videos=storyboard_videos, ref_audios=storyboard_audios,
                first_latent_override=panel_b_lat, last_latent_override=panel_c_lat,
            )
            import comfy.sampler_helpers
            segment_positive_conditionings = (
                comfy.sampler_helpers.convert_cond(positive),
                comfy.sampler_helpers.convert_cond(positive_1),
            )
            plan = _dc_replace(
                plan, video_vae=vae, audio_vae=audio_vae,
                segment_positive_conditionings=segment_positive_conditionings,
                segment_prompt_summaries=(pass0_prompt, pass1_prompt),
                storyboard_segment_avs=(target_av, target_av_1),
                storyboard_bridge_frame=first_len,
            )
            hybrid_info = {
                'storyboard_bridge': True, 'panels': 3,
                'pass0': sb0, 'pass1': sb1, 'refs': len(storyboard_refs),
            }
            _lm_print(
                f'[MiniMaxH3 LongMedia][V64 TRUE 3-PANEL STORYBOARD] A->B then B->C; '
                f'panel B latent reused exactly on both sides; lengths={first_len},{second_len}; '
                f'overlap=0; image refs start at image_4; refs={len(storyboard_refs)}',
                flush=True,
            )
            setup_memory_events.append(_setup_memory_isolation('after_storyboard_conditioning_release', unload_models=True))

        elif conditioning_mode in ('hybrid_first_frame', 'hybrid_first_last'):
            if workflow_mode == 'segmented_continuation' and segmented_opening_frame is not None:
                hybrid_first = segmented_opening_frame
                if conditioning_mode == 'hybrid_first_last':
                    hybrid_last = (loop_last_override if loop_last_override is not None else image_2)
                else:
                    hybrid_last = None
                loop_latent_override = None
                if conditioning_mode == 'hybrid_first_last' and image_2 is None and loop_last_override is None:
                    raise ValueError(
                        'hybrid_first_last requires image_2 as the final-frame keyframe.'
                    )
                # With explicit opening_frame, every image_N remains a native Picture ref.
                hybrid_ref_images = [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9]
            else:
                if image_1 is None:
                    raise ValueError(
                        'Hybrid conditioning requires image_1 as the first-frame keyframe.'
                    )
                hybrid_first = image_1
                hybrid_last = (loop_last_override if loop_last_override is not None else image_2) if conditioning_mode == 'hybrid_first_last' else None
                loop_latent_override = None
                if conditioning_mode == 'hybrid_first_last' and image_2 is None and loop_last_override is None:
                    raise ValueError(
                        'hybrid_first_last requires image_2 as the final-frame keyframe.'
                    )
                if workflow_mode == 'loop':
                    # Hybrid parity: exactly emulate manually wiring the same IMAGE object
                    # into image_1 (first frame) and image_2 (last frame) in hybrid_auto.
                    # image_2 itself is reserved/ignored in loop mode; refs begin at image_3.
                    hybrid_ref_images = [image_3, image_4, image_5, image_6, image_7, image_8, image_9]
                else:
                    hybrid_ref_images = (
                        [image_3, image_4, image_5, image_6, image_7, image_8, image_9]
                        if conditioning_mode == 'hybrid_first_last'
                        else [image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9]
                    )
            hybrid_ref_images = [safe_image_by_id.get(id(v), v) for v in hybrid_ref_images if v is not None]
            hybrid_ref_videos = [v for v in [video_1, video_2, video_3] if v is not None]
            hybrid_ref_audios = ([v for v in [audio_1, audio_2, audio_3] if v is not None] if use_audio_as_reference else [])

            # Keyframe anchors are target-timeline guides and are not numbered as
            # native <Picture N> references.  Detect the unambiguous legacy/input-
            # socket convention (present in the supplied workflow) and translate it
            # before any pass is tokenized.  Native-numbered prompts stay untouched.
            anchor_roles = ['opening frame']
            if conditioning_mode == 'hybrid_first_last':
                anchor_roles.append('ending frame')
            effective_prompt, picture_tag_policy = normalize_hybrid_picture_tags(
                effective_prompt,
                anchor_roles=tuple(anchor_roles),
                reference_count=len(hybrid_ref_images),
            )
            segment0_prompt = _v57_build_segment_prompt(effective_prompt, plan, 0)
            setup_memory_events.append(
                _setup_memory_isolation('before_hybrid_conditioning', unload_models=True)
            )
            positive, target_av, hybrid_info, hybrid_artifacts = _build_longmedia_hybrid_conditioning(
                clip=clip, vae=vae, audio_vae=audio_vae,
                prompt=segment0_prompt, width=width, height=height,
                length=plan.segment_lengths[0], resolution_mode=resolution_mode,
                first_frame=hybrid_first, last_frame=hybrid_last,
                ref_images=hybrid_ref_images, ref_videos=hybrid_ref_videos,
                ref_audios=hybrid_ref_audios,
                first_latent_override=loop_latent_override,
                last_latent_override=loop_latent_override,
            )
            first_keyframe_latent = hybrid_artifacts.get('first_keyframe_latent')
            first_frame_latent_injected = False
            if (not lip_sync_enabled) and first_frame_mode == 'latent_inject' and first_keyframe_latent is not None:
                # A native H3 keyframe is a conditioning guide, not a frozen target
                # latent.  Reuse the already-encoded anchor latent for one leading
                # target step so the opening composition cannot unpack as a collage
                # of the connected references before converging several frames later.
                target_av = inject_leading_video_frame(
                    target_av,
                    {'samples': first_keyframe_latent},
                    float(first_frame_denoise),
                    NestedTensor,
                )
                first_frame_latent_injected = True
            hybrid_info.update({
                'picture_tag_policy': picture_tag_policy,
                'continuation_reference_policy': 'native_order_geometry_stable',
                'first_frame_policy': (
                    'target_latent_inject' if first_frame_latent_injected
                    else str(first_frame_mode)
                ),
            })
            _lm_print(
                '[MiniMaxH3 LongMedia][V327 STARTUP CONTINUITY] '
                'pass0 uses only the native frame-0 anchor; repeated 5/22/39 anchors disabled',
                flush=True,
            )
            setup_memory_events.append(
                _setup_memory_isolation('after_hybrid_conditioning_release', unload_models=True)
            )
            if workflow_mode == 'loop':
                _lm_print('[MiniMaxH3 LongMedia][0.3.0 LOOP] hybrid parity: image_1 is internally used as both first+last frame; image_2 ignored; refs start at image_3', flush=True)
            # In hybrid mode connected video/audio sockets are conditioning references,
            # not source streams to inject into later long-media segments.
            plan = _dc_replace(
                plan, mode='hybrid', source_video=None, source_audio=None,
                reference_audio=(audio_1 if (use_audio_as_reference and audio_1 is not None) else None),
                final_audio_override=(_mix_audio_tracks(audios) if (preserve_audio_output and audios) else None),
                final_audio_track_count=(len(audios) if preserve_audio_output else 0),
                first_frame_override=(
                    hybrid_first if ((not lip_sync_enabled) and first_frame_mode in ('pixel_override', 'blend')) else None
                ),
                first_frame_mode=('disabled' if lip_sync_enabled else first_frame_mode),
                first_frame_denoise=(0.0 if lip_sync_enabled else float(first_frame_denoise)),
                first_frame_blend_frames=(0 if lip_sync_enabled else int(first_frame_blend_frames)),
                first_frame_latent_injected=(False if lip_sync_enabled else first_frame_latent_injected),
                audio_vae=audio_vae, video_vae=vae,
            )
        elif mode == 'automatic_lip_sync':
            ref_audio_waveform = audio_1['waveform'][:1]
            segment_samples = int(round(segment_seconds * audio_1['sample_rate']))
            if ref_audio_waveform.shape[-1] > segment_samples:
                ref_audio_waveform = ref_audio_waveform[..., :segment_samples]
            ref_audio = {'waveform': ref_audio_waveform, 'sample_rate': audio_1['sample_rate']}
            ref_images = {}
            ref_videos = {}
            ref_audios = {}
            for i, img in enumerate(safe_images):
                ref_images[f'ref_image_{i}'] = img
            for i, vid in enumerate(videos):
                ref_videos[f'ref_video_{i}'] = vid
            if ref_audio is not None:
                ref_audios['ref_audio_0'] = ref_audio
            setup_memory_events.append(_setup_memory_isolation('before_native_reference', unload_models=True))
            positive, target_av = NativeReferenceToVideo.execute(
                clip=clip, vae=vae, audio_vae=audio_vae,
                prompt=segment0_prompt, width=width, height=height,
                length=plan.segment_lengths[0], ref_image_size=('max' if images else resolution_mode),
                ref_images=ref_images, ref_videos=ref_videos, ref_audios=ref_audios,
            )
            setup_memory_events.append(_setup_memory_isolation('after_native_reference_release', unload_models=True))
            if first_frame_mode == 'latent_inject':
                # Write image_1 into the video latent's leading frame *before*
                # sampling, with a partial-denoise mask, so the sampler can
                # soften the transition instead of a hard post-decode pixel
                # splice. Falls back to a strict pixel splice at denoise=0.
                # Only the minimal 5-frame H3 unit is encoded here (not the
                # full segment) — inject_leading_video_frame only needs its
                # first latent frame, and encoding the whole segment length
                # just to discard the rest would waste VRAM for nothing.
                min_frames = align_frame_count(5)
                held = _fit_frames(image_1, min_frames, 'crop_or_pad_last')
                held = _resize_frames(held, width, height, 'center_crop')
                frame_latent = {'samples': vae.encode(held)}
                setup_memory_events.append(_setup_memory_isolation('after_first_frame_vae_release', unload_models=True))
                target_av = inject_leading_video_frame(
                    target_av, frame_latent, float(first_frame_denoise), NestedTensor,
                )
            plan = _dc_replace(
                plan, mode='automatic_lip_sync',
                source_audio=None,
                reference_audio=audio_1,
                final_audio_override=(audio_1 if audio_mode in ('auto', 'preserve', 'preserve_reference') else None),
                final_audio_track_count=max(1, len(audios)),
                first_frame_override=image_1, audio_vae=audio_vae, video_vae=vae,
                first_frame_mode=first_frame_mode,
                first_frame_denoise=float(first_frame_denoise),
                first_frame_blend_frames=int(first_frame_blend_frames),
                first_frame_latent_injected=(first_frame_mode == 'latent_inject'),
            )
            if len(audios) > 1 and audio_mode in ('auto', 'preserve', 'preserve_reference'):
                mixed = _mix_audio_tracks(audios)
                plan = _dc_replace(
                    plan, final_audio_override=mixed, final_audio_track_count=len(audios),
                )
        elif mode == 't2v':
            ref_images = {}
            ref_videos = {}
            ref_audios = {}
            for i, img in enumerate(safe_images):
                ref_images[f'ref_image_{i}'] = img
            for i, vid in enumerate(videos):
                ref_videos[f'ref_video_{i}'] = vid
            if use_audio_as_reference:
                for i, aud in enumerate(audios):
                    ref_audios[f'ref_audio_{i}'] = aud
            setup_memory_events.append(_setup_memory_isolation('before_native_reference', unload_models=True))
            positive, target_av = NativeReferenceToVideo.execute(
                clip=clip, vae=vae, audio_vae=audio_vae,
                width=width, height=height, length=plan.segment_lengths[0],
                prompt=segment0_prompt, ref_image_size=('max' if images else resolution_mode),
                ref_images=ref_images, ref_videos=ref_videos, ref_audios=ref_audios,
            )
            setup_memory_events.append(_setup_memory_isolation('after_native_reference_release', unload_models=True))
            if preserve_audio_output and audios:
                plan = _dc_replace(plan, final_audio_override=_mix_audio_tracks(audios), final_audio_track_count=len(audios))
        elif mode == 'audio_to_video':
            length_frames = plan.segment_lengths[0]
            start_frame = plan.segment_starts[0]
            available, _ = _slice_source_audio_for_segment(audio_1, start_frame, length_frames)
            waveform_for_encode = available.movedim(1, -1)
            frozen_audio_latent = audio_vae.encode(waveform_for_encode)
            video_t = video_latent_t(length_frames)
            audio_t = audio_latent_t(length_frames)
            audio_lat = frozen_audio_latent
            if audio_lat.shape[-1] != audio_t:
                audio_lat = torch.zeros(
                    (1, 32, 2, audio_t),
                    dtype=frozen_audio_latent.dtype,
                    device=frozen_audio_latent.device,
                )
                copy_len = min(audio_t, frozen_audio_latent.shape[-1])
                audio_lat[..., :copy_len] = frozen_audio_latent[..., :copy_len]
            video_lat = torch.zeros((1, 24, video_t, 1, 1), dtype=frozen_audio_latent.dtype)
            av_samples = NestedTensor((video_lat, audio_lat))
            video_mask = torch.ones((1, 1, video_t, 1, 1), dtype=torch.float32)
            audio_denoise = (
                1.0 if audio_mode in ('generate', 'reference_only') else 0.0
            )
            audio_mask = torch.full(
                (1, 1, 1, audio_lat.shape[-1]), audio_denoise, dtype=torch.float32
            )
            mask_samples = NestedTensor((video_mask, audio_mask))
            target_av = {'samples': av_samples, 'noise_mask': mask_samples}
            setup_memory_events.append(_setup_memory_isolation('before_clip_encode', unload_models=True))
            positive = _encode_prompt(clip, segment0_prompt)
            setup_memory_events.append(_setup_memory_isolation('after_clip_release', unload_models=True))
            mixed_audio = _mix_audio_tracks(audios) if audios else None
            plan = _dc_replace(
                plan, mode='audio_to_video',
                source_audio=audio_1 if audio_mode in ('generate', 'reference_only', 'preserve_reference') else None,
                final_audio_override=(mixed_audio if audio_mode in ('auto', 'preserve', 'preserve_reference') else None),
                final_audio_track_count=len(audios),
                audio_vae=audio_vae, video_vae=vae,
            )
        elif mode in ('video_to_video', 'video_audio_to_video'):
            length_frames = plan.segment_lengths[0]
            start_frame = plan.segment_starts[0]
            source_frames = slice_video_segment(
                video_1, start_frame, length_frames, video_fps,
            )
            import comfy.utils
            samples = source_frames.movedim(-1, 1)
            samples = comfy.utils.common_upscale(samples, width, height, 'lanczos', 'disabled')
            source_frames = samples.movedim(1, -1)
            video_lat = vae.encode(source_frames)
            audio_lat = None
            if audio_1 is not None:
                available, _ = _slice_source_audio_for_segment(audio_1, start_frame, length_frames)
                waveform_for_encode = available.movedim(1, -1)
                audio_lat = audio_vae.encode(waveform_for_encode)
            if audio_lat is None:
                a_t = audio_latent_t(length_frames)
                audio_lat = torch.zeros((1, 32, 2, a_t), dtype=video_lat.dtype)
            av_samples = NestedTensor((video_lat, audio_lat))
            video_mask = torch.ones((1, 1, video_lat.shape[2], 1, 1), dtype=torch.float32)
            audio_denoise = (
                1.0 if audio_mode in ('generate', 'reference_only') else 0.0
            )
            audio_mask = torch.full(
                (1, 1, 1, audio_lat.shape[-1]), audio_denoise, dtype=torch.float32
            )
            mask_samples = NestedTensor((video_mask, audio_mask))
            target_av = {'samples': av_samples, 'noise_mask': mask_samples}
            setup_memory_events.append(_setup_memory_isolation('before_clip_encode', unload_models=True))
            positive = _encode_prompt(clip, segment0_prompt)
            setup_memory_events.append(_setup_memory_isolation('after_clip_release', unload_models=True))
            mixed_audio = _mix_audio_tracks(audios) if audios else None
            plan = _dc_replace(
                plan,
                source_audio=(audio_1 if (audio_1 is not None and use_audio_as_reference) else None),
                source_video=video_1 if video_1 is not None else None,
                final_audio_override=(mixed_audio if audio_mode in ('auto', 'preserve', 'preserve_reference') else None),
                final_audio_track_count=len(audios),
                audio_vae=audio_vae,
                video_vae=vae,
            )
        else:
            ref_images = {}
            ref_videos = {}
            ref_audios = {}
            for i, img in enumerate(safe_images):
                ref_images[f'ref_image_{i}'] = img
            for i, vid in enumerate(videos):
                ref_videos[f'ref_video_{i}'] = vid
            if use_audio_as_reference:
                for i, aud in enumerate(audios):
                    ref_audios[f'ref_audio_{i}'] = aud
            setup_memory_events.append(_setup_memory_isolation('before_native_reference', unload_models=True))
            positive, target_av = NativeReferenceToVideo.execute(
                clip=clip, vae=vae, audio_vae=audio_vae,
                width=width, height=height, length=plan.segment_lengths[0],
                prompt=segment0_prompt, ref_image_size=('max' if images else resolution_mode),
                ref_images=ref_images, ref_videos=ref_videos, ref_audios=ref_audios,
            )
            setup_memory_events.append(_setup_memory_isolation('after_native_reference_release', unload_models=True))
            if preserve_audio_output and audios:
                plan = _dc_replace(plan, final_audio_override=_mix_audio_tracks(audios), final_audio_track_count=len(audios))

        # Normalize passthrough semantics across *all* workflow branches. Historically
        # auto preserved an attached source soundtrack in lip-sync / A2V / V2V paths,
        # but native Ref2VA/T2V branches forgot to populate final_audio_override. That
        # allowed Turbo-distilled model audio with incompatible latent geometry to reach
        # AudioVAE.decode(). If an input soundtrack exists, auto/preserve/preserve_reference
        # always retain the untouched waveform for final output. generate/reference_only
        # are the only modes that intentionally decode model-generated audio.
        passthrough_audio_mode = audio_mode in ('auto', 'preserve', 'preserve_reference')
        if passthrough_audio_mode and audios and getattr(plan, 'final_audio_override', None) is None:
            plan = _dc_replace(
                plan,
                final_audio_override=_mix_audio_tracks(audios),
                final_audio_track_count=len(audios),
            )
        if lip_sync_enabled:
            # v0.3.104: preserve current UI/output semantics.  Audio 1 stays a
            # native Ref2VA content reference and is additionally anchored to the
            # local H3 timeline via minimax_keyframes.  Final mux restores the
            # pristine input track; sampled H3 audio is not the timing authority.
            plan = _dc_replace(
                plan,
                source_audio=audio_1,
                final_audio_override=audio_1,
                final_audio_track_count=1,
                lip_sync_native_audio_guide=True,
                lip_sync_target_audio_locked=True,
            )
            _lm_print(
                '[MiniMaxH3 LongMedia][0.3.108 AUTHORITATIVE LOCAL-0 LIP SYNC] '
                'Audio1=full native Ref2VA reference + native local-0 timing guide on every clip; continuation keeps VIDEO Motion Context while preserving the source-audio guide through hidden overlap; original Audio1 restored at output',
                flush=True,
            )
            # v0.3.113: Ref2VA refs/guide remain useful semantic conditioning, but
            # exact timing authority now comes from the frozen target audio stream.
            target_av = _v113_lock_source_audio_in_target(
                target_av, audio_vae, audio_1,
                int(plan.segment_starts[0]), int(plan.segment_lengths[0]),
            )

        # Persist the requested audio output policy in the plan. Decode must not infer
        # preserve semantics from the shape/content of the sampled audio stream: Turbo
        # LoRAs may leave a stream that is invalid for the stock Audio VAE decoder.
        plan = _dc_replace(
            plan,
            audio_output_mode=audio_mode,
            suppress_visible_opening_anchor=False,
            regression_safe_segmented_conditioning=(workflow_mode in ('segmented_continuation', 'multiclip')),
            decouple_original_image_refs_after_pass0=False,
            release_guard=True,
        )

        # V57: build every per-pass TEXT conditioning now, while TE is intentionally available.
        # The plan receives only ready CONDITIONING tensors; never CLIP/TE/model-patcher objects.
        plan = _dc_replace(plan, video_vae=vae, audio_vae=audio_vae)
        # v0.3.108: pass 0 is sampled from the Setup output CONDITIONING itself,
        # so attach the authoritative native source-audio clock directly here.
        if lip_sync_enabled:
            positive = _v104_attach_native_lipsync_guide(
                positive, audio_vae, audio_1, plan, 0,
            )
        v329_native_refs = None
        if (
            conditioning_mode in ('hybrid_first_frame', 'hybrid_first_last', 'multiclip_ref2va')
            and hybrid_artifacts is not None
            and hybrid_artifacts.get('ref_items')
        ):
            # Keep tokenizer presentation AND latent block geometry identical for
            # every pass.  Combining distinct people into one sheet changed the
            # reference count and target packed-layout origin at the visible join.
            v329_native_refs = (
                hybrid_artifacts['ref_items'],
                hybrid_artifacts['ref_blocks'],
            )

        if conditioning_mode != 'storyboard_bridge':
            if getattr(plan, 'mode', None) == 'multiclip':
                mc_prompts = tuple(c['prompt'] for c in multiclip_clips)
                segment_positive_conditionings, segment_prompt_summaries = _v85_preencode_multiclip_conditionings(
                    clip, positive, plan, mc_prompts, v329_native_refs=v329_native_refs,
                    lip_sync_audio=(audio_1 if lip_sync_enabled else None), audio_vae=audio_vae,
                )
            else:
                segment_positive_conditionings, segment_prompt_summaries = _v57_preencode_segment_conditionings(
                    clip, effective_prompt, positive, plan, v329_native_refs=v329_native_refs,
                    lip_sync_audio=(audio_1 if lip_sync_enabled else None), audio_vae=audio_vae,
                )
            plan = _dc_replace(
                plan,
                segment_positive_conditionings=segment_positive_conditionings,
                segment_prompt_summaries=segment_prompt_summaries,
            )
        setup_memory_events.append(_setup_memory_isolation('setup_exit_release', unload_models=True))

        report = json.dumps({
            'mode': plan.mode,
            'passes': plan.passes,
            'segmentation_active': bool(plan.passes > 1),
            'manual_duration_seconds': float(plan.total_duration),
            'segment_seconds_requested': float(segment_seconds),
            'multiclip_enabled': bool(workflow_mode == 'multiclip'),
            'clip_engine_enabled': bool(getattr(plan, 'mode', None) == 'multiclip'),
            'timeline_policy': getattr(plan, 'timeline_policy', 'legacy'),
            'selected_workflow_mode': workflow_mode,
            'multiclip_plan_source': ('external_planner' if (workflow_mode == 'multiclip' and external_clip_plan is not None) else ('setup_editor' if workflow_mode == 'multiclip' else ('fixed_segment_math' if workflow_mode == 'segmented_continuation' else None))),
            'release_guard': True,
            'multiclip_clip_durations': ([float(c['duration']) for c in multiclip_clips] if multiclip_clips else None),
            'multiclip_segment_seeds': ([c['seed'] for c in multiclip_clips] if multiclip_clips else None),
            'segment_seconds_semantics': 'new_output_timeline_plus_extra_overlap_context',
            'segment_lengths_frames': list(plan.segment_lengths),
            'segment_starts_frames': list(plan.segment_starts),
            'overlap_frames': plan.overlap_frames,
            'segment_conditioning_policy': 'preencoded_in_setup_no_clip_in_plan',
            'segment_conditionings_preencoded': int(len(getattr(plan, 'segment_positive_conditionings', ()) or ())),
            'decouple_original_image_refs_after_pass0': bool(getattr(plan, 'decouple_original_image_refs_after_pass0', False)),
            'h3_target_geometry': h3_target_geometry,
            'h3_reference_geometry': h3_ref_geometry,
            'h3_reference_pixel_budget': int(_H3_SAFE_REF_PIXELS),
            'h3_reference_resolution_policy': 'independent_safe_0.60MP_max_no_upscale',
            'output_frames': int(plan.output_frames),
            'trim_frames': int(plan.trim_frames),
            'final_audio_tracks': plan.final_audio_track_count,
            'audio_mode': audio_mode,
            'audio_reference_enabled': bool(use_audio_as_reference),
            'lip_sync_enabled': bool(lip_sync_enabled),
            'audio_output_bypass': bool(getattr(plan, 'final_audio_override', None) is not None),
            'workflow_mode': workflow_mode,
            'conditioning_mode': conditioning_mode,
            'startup_anchor_frames': [],
            'first_frame_mode': first_frame_mode,
            'first_frame_latent_injected': bool(
                getattr(plan, 'first_frame_latent_injected', False)
            ),
            'continuation_reference_policy': (
                'native_order_geometry_stable' if v329_native_refs is not None
                else 'native_metadata_passthrough'
            ),
            'hybrid': hybrid_info,
            'setup_memory_isolation': setup_memory_events,
        }, indent=2)
        return (positive, target_av, plan, plan.total_duration, plan.passes, report)


class MiniMaxH3LatentLabLongMediaNextSegment:
    DESCRIPTION = (
        'Prepare the next segment AV latent by injecting the frozen overlap '
        'from the previous result and source media after the overlap.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'long_media_plan': ('LONG_MEDIA_PLAN',),
                'previous_av': ('LATENT',),
                'segment_index': (
                    'INT',
                    {'default': 1, 'min': 1, 'max': 100, 'step': 1},
                ),
                'video_context_denoise': (
                    'FLOAT',
                    {
                        'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                        'tooltip': '0 preserves the inherited overlap exactly; 1 fully denoises it.',
                    },
                ),
                'audio_context_denoise': (
                    'FLOAT',
                    {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
            }
        }

    RETURN_TYPES = ('LATENT', 'STRING')
    RETURN_NAMES = ('continuation_av', 'report')
    FUNCTION = 'prepare'
    CATEGORY = CATEGORY_LONGMEDIA

    def prepare(self, long_media_plan, previous_av, segment_index,
                video_context_denoise=0.0, audio_context_denoise=0.0):
        plan = long_media_plan
        seg_idx = int(segment_index)
        timeline = _segment_timeline_contract(plan, seg_idx)
        start_frame = int(timeline['context_start'])
        visible_start_frame = int(timeline['visible_start'])
        length_frames = int(timeline['length_frames'])
        overlap = int(timeline['local_visible_offset'])

        prev_video, prev_audio = unpack_av_samples(previous_av)
        target_video_t = video_latent_t(length_frames)
        target_audio_t = audio_latent_t(length_frames)
        overlap_video_t = video_latent_t(overlap) if overlap else 0
        overlap_audio_t = round(overlap / FPS * AUDIO_LATENT_FPS) if overlap else 0
        native_motion_head = bool(
            (
                getattr(plan, 'mode', None) == 'multiclip' and seg_idx > 0
            )
            or (
                seg_idx == 1
                and int(getattr(plan, 'passes', 0) or 0) == 2
                and getattr(plan, 'mode', None) == 'segmented_continuation'
            )
        ) and _v83_native_guide_api_supported()

        video = torch.zeros(
            (prev_video.shape[0], prev_video.shape[1], target_video_t,
             prev_video.shape[3], prev_video.shape[4]),
            dtype=prev_video.dtype, device=prev_video.device,
        )
        audio = torch.zeros(
            (prev_audio.shape[0], prev_audio.shape[1], prev_audio.shape[2], target_audio_t),
            dtype=prev_audio.dtype, device=prev_audio.device,
        )
        if overlap_video_t and not native_motion_head:
            video[:, :, :overlap_video_t] = prev_video[:, :, -overlap_video_t:]
        if overlap_audio_t and not native_motion_head:
            audio[..., :overlap_audio_t] = prev_audio[..., -overlap_audio_t:]

        video_overlap_policy = ('native_motion_context_fresh_head' if native_motion_head else 'zero_fill')
        latent_value_transform = 'none'

        if plan.source_video is not None and plan.video_vae is not None:
            source_frames = slice_video_segment(
                plan.source_video, start_frame, length_frames, plan.video_fps,
            )
            target_av_for_encode = {'samples': NestedTensor((video, audio))}
            encoded_result = MiniMaxH3LatentLabVideoEncode().encode(
                plan.video_vae, source_frames, 'strict', 'none', target_av_for_encode,
            )
            source_video_latent = encoded_result[0]['samples']
            if overlap_video_t:
                video[:, :, overlap_video_t:] = source_video_latent[:, :, overlap_video_t:].to(video)
            else:
                video = source_video_latent.to(video)
            video_overlap_policy = 'exact_frozen'

        if plan.source_audio is not None and plan.audio_vae is not None:
            available, _ = _slice_source_audio_for_segment(
                plan.source_audio, start_frame, length_frames
            )
            waveform_for_encode = available.movedim(1, -1)
            source_audio_latent = plan.audio_vae.encode(waveform_for_encode)
            if bool(getattr(plan, 'lip_sync_target_audio_locked', False)):
                # Source speech is the authoritative local clock.  Use the exact
                # global slice across the whole clip, including hidden overlap; do
                # not inherit sampled/generated audio from the previous clip.
                audio = _fit_stream(source_audio_latent, audio, 'audio', 'crop_pad', 'start')
            elif overlap_audio_t:
                audio[..., overlap_audio_t:] = source_audio_latent[
                    ..., :audio.shape[-1] - overlap_audio_t
                ].to(audio)
            else:
                audio = source_audio_latent.to(audio)

        video_mask = torch.ones((1, 1, target_video_t, 1, 1), dtype=torch.float32)
        audio_mask = torch.ones((1, 1, 1, target_audio_t), dtype=torch.float32)
        if bool(getattr(plan, 'lip_sync_target_audio_locked', False)):
            audio_mask.zero_()
        if overlap_video_t and not native_motion_head:
            # Legacy continuation: inherited overlap is frozen/partially denoised.
            video_mask[:, :, :overlap_video_t] = float(video_context_denoise)
        if overlap_audio_t and not native_motion_head and not bool(getattr(plan, 'lip_sync_target_audio_locked', False)):
            audio_mask[..., :overlap_audio_t] = float(audio_context_denoise)
        if native_motion_head:
            _lm_print(
                '[MiniMaxH3 LongMedia][0.3.87 FRESH CONTINUATION HEAD] '
                f'segment=1 overlap={overlap}f target head is zero-init/full-denoise; '
                'continuity is owned by native minimax_keyframes, not latent copying',
                flush=True,
            )

        av_samples = NestedTensor((video, audio))
        mask_samples = NestedTensor((video_mask, audio_mask))
        output = {k: v for k, v in previous_av.items() if k not in ('noise_mask', 'samples')}
        output['samples'] = av_samples
        output['noise_mask'] = mask_samples

        _lm_print(
            '[MiniMaxH3 LongMedia][V318 TIMELINE] '
            f'segment={seg_idx} context_start={start_frame}f visible_start={visible_start_frame}f '
            f'local_visible_offset={overlap}f visible_frames={int(timeline["visible_frames"])}f '
            f'source_video_window_start={start_frame if plan.source_video is not None else "none"} '
            f'source_audio_window_start={start_frame if plan.source_audio is not None else "none"} '
            f'locked_target_audio={bool(getattr(plan, "lip_sync_target_audio_locked", False))}',
            flush=True,
        )
        report = json.dumps({
            'segment_index': seg_idx,
            'context_start_frame': start_frame,
            'visible_start_frame': visible_start_frame,
            'local_visible_offset_frames': overlap,
            'visible_frames': int(timeline['visible_frames']),
            'visible_end_frame': int(timeline['visible_end']),
            'source_video_window_start_frame': (start_frame if plan.source_video is not None else None),
            'source_audio_window_start_frame': (start_frame if plan.source_audio is not None else None),
            'length_frames': length_frames,
            'overlap_frames': overlap,
            'video_overlap_policy': video_overlap_policy,
            'overlap_mask_policy': ('native_motion_context_full_denoise' if native_motion_head else 'constant_overlap_denoise'),
            'native_motion_context_head': bool(native_motion_head),
            'latent_value_transform': latent_value_transform,
            'video_context_denoise': float(video_context_denoise),
            'audio_context_denoise': float(audio_context_denoise),
        }, indent=2)
        return (output, report)


class MiniMaxH3LatentLabRefineSigmas:
    """Split the connected schedule into a true KSampler Advanced two-stage trajectory."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'sigmas': ('SIGMAS',),
                'refine_steps': ('INT', {'default': 2, 'min': 1, 'max': 1000, 'step': 1}),
            }
        }

    RETURN_TYPES = ('SIGMAS', 'SIGMAS', 'INT', 'INT', 'INT', 'STRING')
    RETURN_NAMES = ('main_sigmas', 'refine_sigmas', 'total_steps', 'switch_step', 'refine_steps_effective', 'report')
    FUNCTION = 'build'
    CATEGORY = CATEGORY_LONGMEDIA

    def build(self, sigmas, refine_steps=2):
        main, refine, total_steps, switch_step, effective, requested = split_refine_sigmas(
            sigmas, refine_steps
        )
        report = json.dumps({
            'total_steps': total_steps,
            'main_steps': int(max(0, switch_step)),
            'refine_steps_requested': requested,
            'refine_steps_effective': effective,
            'switch_step': switch_step,
            'main_sigma_points': int(main.numel()),
            'refine_sigma_points': int(refine.numel()),
            'main_sigma_start': float(main[0].detach().float().cpu()),
            'main_sigma_end': float(main[-1].detach().float().cpu()),
            'refine_sigma_start': float(refine[0].detach().float().cpu()),
            'refine_sigma_end': float(refine[-1].detach().float().cpu()),
            'scheduler_source': 'connected_sigmas_true_advanced_split',
            'main_return_with_leftover_noise': True,
            'refine_add_noise': False,
            'intervals_total': total_steps,
            'intervals_main': int(max(0, switch_step)),
            'intervals_refine': effective,
            'intervals_total_model_evaluations': total_steps,
        })
        return (main, refine, total_steps, switch_step, effective, report)





class MiniMaxH3LatentLabStockAdvancedRefiner:
    """Run refiner through the stock ComfyUI KSampler Advanced path.

    This node intentionally returns the stock solver state from
    ``comfy.sample.sample()``. It does not substitute preview x0, does not freeze
    streams, and does not reconstruct NestedTensor outputs. The caller is
    expected to pass an in-progress latent from stage 1 and a full connected
    sigma schedule together with ``start_at_step``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'guider': ('GUIDER',),
                'final_av': ('LATENT',),
                'sigmas': ('SIGMAS',),
                'start_at_step': ('INT', {'default': 0, 'min': 0, 'max': 100000, 'step': 1}),
                'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ('LATENT', 'STRING')
    RETURN_NAMES = ('refined_av', 'report')
    FUNCTION = 'refine'
    CATEGORY = 'MiniMax H3/LongMedia/Internal'

    @staticmethod
    def _restore_raw_conditioning(converted):
        raw = []
        for item in list(converted or []):
            if not isinstance(item, dict):
                raise RuntimeError('Stock refiner expected CFGGuider converted conditioning dictionaries.')
            meta = item.copy()
            cross_attn = meta.pop('cross_attn', None)
            meta.pop('uuid', None)
            raw.append([cross_attn, meta])
        return raw

    def refine(self, guider, final_av, sigmas, start_at_step=0, seed=0):
        import comfy.sample

        if final_av is None or not isinstance(final_av, dict) or 'samples' not in final_av:
            raise RuntimeError('Stock Advanced refiner requires the stage-1 H3 AV LATENT.')
        model = getattr(guider, 'model_patcher', None)
        if model is None:
            raise RuntimeError('Stock Advanced refiner could not resolve model_patcher from GUIDER.')
        original_conds = getattr(guider, 'original_conds', None) or {}
        positive = self._restore_raw_conditioning(original_conds.get('positive', []))
        negative = self._restore_raw_conditioning(original_conds.get('negative', []))
        cfg = float(getattr(guider, 'cfg', 1.0))

        latent_image = final_av['samples']
        latent_image = comfy.sample.fix_empty_latent_channels(
            model,
            latent_image,
            final_av.get('downscale_ratio_spacial', None),
            final_av.get('downscale_ratio_temporal', None),
        )
        noise = comfy.sample.prepare_empty_noise(latent_image)
        noise_mask = final_av.get('noise_mask', None)
        total_steps = max(0, int(sigmas.numel()) - 1) if torch.is_tensor(sigmas) else max(0, len(sigmas) - 1)
        start_at_step = max(0, min(int(start_at_step), total_steps))

        _lm_print(
            '[MiniMaxH3 LongMedia][0.4.1 STOCK ADVANCED REFINER] '
            f'sampler=euler; scheduler=simple; total_steps={int(total_steps)}; '
            f'add_noise=disable; start_at_step={int(start_at_step)}; end_at_step=10000; '
            'return_with_leftover_noise=disable; output=stock_solver_state',
            flush=True,
        )

        samples = comfy.sample.sample(
            model,
            noise,
            int(total_steps),
            cfg,
            'euler',
            'simple',
            positive,
            negative,
            latent_image,
            denoise=1.0,
            disable_noise=True,
            start_step=int(start_at_step),
            last_step=10000,
            force_full_denoise=True,
            noise_mask=noise_mask,
            sigmas=sigmas,
            callback=None,
            disable_pbar=False,
            seed=int(seed),
        )

        out = final_av.copy()
        out.pop('downscale_ratio_spacial', None)
        out.pop('downscale_ratio_temporal', None)
        out['samples'] = samples
        report = json.dumps({
            'mode': 'stock_ksampler_advanced_true_split',
            'sampler': 'euler',
            'scheduler': 'simple',
            'steps': int(total_steps),
            'start_at_step': int(start_at_step),
            'end_at_step': 10000,
            'add_noise': False,
            'return_with_leftover_noise': False,
            'force_full_denoise': True,
            'sigma_source': 'connected_full_schedule',
            'output': 'stock_solver_state',
            'x0_substitution': False,
            'stream_freeze': False,
        })
        return (out, report)


class MiniMaxH3LatentLabProtectRefineAV:
    """Keep refine video-only and restore the frozen continuation head."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'base_av': ('LATENT',),
                'refined_av': ('LATENT',),
                'overlap_frames': ('INT', {'default': 0, 'min': 0, 'max': 10000, 'step': 1}),
            }
        }

    RETURN_TYPES = ('LATENT', 'STRING')
    RETURN_NAMES = ('av', 'report')
    FUNCTION = 'protect'
    CATEGORY = CATEGORY_LONGMEDIA

    def protect(self, base_av, refined_av, overlap_frames=0):
        base_video, base_audio = unpack_av_samples(base_av)
        refined_video, refined_audio = unpack_av_samples(refined_av)
        if tuple(base_video.shape) != tuple(refined_video.shape):
            raise ValueError(
                f'Refine changed video latent geometry: {tuple(base_video.shape)} -> {tuple(refined_video.shape)}'
            )
        if tuple(base_audio.shape) != tuple(refined_audio.shape):
            raise ValueError(
                f'Refine changed audio latent geometry: {tuple(base_audio.shape)} -> {tuple(refined_audio.shape)}'
            )
        out_video = refined_video.clone()
        out_audio = refined_audio.clone()
        protected_video_t = 0
        protected_audio_t = 0
        overlap_frames = max(0, int(overlap_frames))
        if overlap_frames > 0:
            try:
                protected_video_t = min(int(video_latent_t(overlap_frames)), int(out_video.shape[2]))
            except Exception:
                protected_video_t = 0
            try:
                protected_audio_t = min(int(audio_latent_t(overlap_frames)), int(out_audio.shape[-1]))
            except Exception:
                protected_audio_t = 0
            if protected_video_t > 0:
                out_video[:, :, :protected_video_t] = base_video[:, :, :protected_video_t]
            if protected_audio_t > 0:
                out_audio[..., :protected_audio_t] = base_audio[..., :protected_audio_t]

        out = dict(refined_av)
        out['samples'] = NestedTensor((out_video, out_audio))
        # Preserve the original context mask metadata. The low-noise stage is a
        # continuation of the same schedule, so audio is refined too; only the
        # exact frozen overlap is restored from the pre-sampling segment input.
        if 'noise_mask' in base_av:
            out['noise_mask'] = base_av['noise_mask']
        else:
            out.pop('noise_mask', None)
        report = json.dumps({
            'audio_restored_from_main_pass': False,
            'audio_refined_as_same_trajectory': True,
            'protected_overlap_frames': overlap_frames,
            'protected_video_latent_steps': protected_video_t,
            'protected_audio_latent_steps': protected_audio_t,
        })
        return (out, report)


def _h3_model_size_bytes_from_guider(guider):
    """Best-effort model storage size for sampler-local residency policy."""
    patcher = getattr(guider, 'model_patcher', None)
    if patcher is None:
        return None
    for name in ('model_size', 'loaded_size'):
        fn = getattr(patcher, name, None)
        if callable(fn):
            try:
                value = int(fn())
                if value > 0:
                    return value
            except Exception:
                pass
    return None



class MiniMaxH3LatentLabUltraPinnedMemoryGate:
    """Temporarily mirror ComfyUI --disable-pinned-memory for ultra H3 sampling."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'guider': ('GUIDER',), 'enable': ('BOOLEAN', {'default': True})}}
    RETURN_TYPES = ('GUIDER', 'BOOLEAN')
    RETURN_NAMES = ('guider', 'previous_disable_pinned_memory')
    FUNCTION = 'apply'
    CATEGORY = CATEGORY_LONGMEDIA

    def apply(self, guider, enable=True):
        previous = False
        if bool(enable):
            try:
                from comfy.cli_args import args as _args
                previous = bool(getattr(_args, 'disable_pinned_memory', False))
                _args.disable_pinned_memory = True
                patcher = getattr(guider, 'model_patcher', None)
                if patcher is not None and hasattr(patcher, 'unpin_all_weights'):
                    try:
                        patcher.unpin_all_weights()
                    except Exception as exc:
                        _lm_print('[MiniMaxH3 LongMedia][0.3.59 PINNED-MEMORY GATE] unpin warning: '
                                  f'{type(exc).__name__}: {exc}', flush=True)
                try:
                    import comfy.model_management as _mm
                    if hasattr(_mm, 'soft_empty_cache'):
                        _mm.soft_empty_cache()
                except Exception:
                    pass
                _lm_print(
                    '[MiniMaxH3 LongMedia][0.3.59 PINNED-MEMORY GATE] '
                    f'disable_pinned_memory {previous}->True for ultra_low_vram H3 sampling; '
                    'existing model pins released before first weight fault',
                    flush=True,
                )
            except Exception as exc:
                _lm_print('[MiniMaxH3 LongMedia][0.3.59 PINNED-MEMORY GATE] unavailable: '
                          f'{type(exc).__name__}: {exc}', flush=True)
        return (guider, previous)


class MiniMaxH3LatentLabUltraPinnedMemoryRestore:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'final_av': ('LATENT',),
            'previous_disable_pinned_memory': ('BOOLEAN', {'default': False}),
            'restore': ('BOOLEAN', {'default': True}),
        }}
    RETURN_TYPES = ('LATENT',)
    RETURN_NAMES = ('final_av',)
    FUNCTION = 'apply'
    CATEGORY = CATEGORY_LONGMEDIA

    def apply(self, final_av, previous_disable_pinned_memory=False, restore=True):
        if bool(restore):
            try:
                from comfy.cli_args import args as _args
                _args.disable_pinned_memory = bool(previous_disable_pinned_memory)
                _lm_print(
                    '[MiniMaxH3 LongMedia][0.3.59 PINNED-MEMORY RESTORE] '
                    f'disable_pinned_memory restored to {bool(previous_disable_pinned_memory)}',
                    flush=True,
                )
            except Exception as exc:
                _lm_print('[MiniMaxH3 LongMedia][0.3.59 PINNED-MEMORY RESTORE] warning: '
                          f'{type(exc).__name__}: {exc}', flush=True)
        return (final_av,)


def _resolve_h3_memory_mode(guider, requested):
    requested = str(requested or 'auto').lower()
    if requested not in ('auto', 'normal', 'low_vram', 'ultra_low_vram'):
        requested = 'auto'
    try:
        gpu_bytes = int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory) if torch.cuda.is_available() else 0
    except Exception:
        gpu_bytes = 0
    model_bytes = _h3_model_size_bytes_from_guider(guider)
    if requested != 'auto':
        effective, reason = requested, 'forced by sampler memory_mode'
    else:
        ratio = (float(model_bytes) / float(gpu_bytes)) if (model_bytes and gpu_bytes) else None
        if ratio is not None and ratio >= 1.75:
            effective, reason = 'ultra_low_vram', f'model/VRAM ratio={ratio:.2f} >= 1.75'
        elif ratio is not None and ratio >= 1.10:
            effective, reason = 'low_vram', f'model/VRAM ratio={ratio:.2f} >= 1.10'
        elif gpu_bytes and gpu_bytes <= int(18.5 * 1024**3) and model_bytes and model_bytes >= int(24 * 1024**3):
            effective, reason = 'ultra_low_vram', 'large model on <=18.5GB GPU'
        else:
            effective, reason = 'normal', 'model fits normal residency policy'
    return {'requested': requested, 'effective': effective, 'reason': reason, 'model_bytes': model_bytes, 'gpu_bytes': gpu_bytes}


class MiniMaxH3LatentLabLongMediaSampler:
    DESCRIPTION = 'Expand a long-media plan into a sequential multi-pass sampler graph. When manual_duration exceeds segment_seconds, segments are sampled one after another, context is carried across segment boundaries, and the result is stitched into one final AV latent.'
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'initial_av': ('LATENT',),
                'long_media_plan': ('LONG_MEDIA_PLAN',),
                'guider': ('GUIDER',),
                'sampler': ('SAMPLER',),
                'sigmas': ('SIGMAS',),
                'seed': (
                    'INT',
                    {'default': 0, 'min': 0, 'max': 18446744073709551615},
                ),
                'video_context_denoise': (
                    'FLOAT',
                    {
                        'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                        'tooltip': '0 preserves each inherited overlap exactly; 1 fully denoises it.',
                    },
                ),
                'audio_context_denoise': (
                    'FLOAT',
                    {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'offload_completed_segments': (
                    'BOOLEAN',
                    {
                        'default': True,
                        'tooltip': (
                            'Move each pass\'s stitched result to CPU RAM once it has '
                            'been folded in, instead of leaving the whole growing clip '
                            'resident on the GPU for the rest of the run. Only the '
                            'accumulator moves — the small per-pass sampling context '
                            'stays on the GPU as before — so this has no effect on '
                            'output, only on peak VRAM during long multi-pass runs. '
                            'Turn off only to restore the previous (all-GPU) behavior.'
                        ),
                    },
                ),
                'mlp_chunk_tokens': (
                    'INT',
                    {
                        'default': 8192, 'min': 0, 'max': 131072, 'step': 512,
                        'tooltip': (
                            'Token chunk size for the low-VRAM H3 MLP path. Manual mode uses 512-token increments so low-VRAM users can select 4096/3072/2048/1536/1024/512. '
                            '8192 is the current safe default. Larger values are faster '
                            'but use more VRAM. Set 0 to effectively disable MLP '
                            'chunking for A/B testing.'
                        ),
                    },
                ),
                'attention_mode': (
                    ['auto', 'existing', 'sol', 'scheduled_sol'],
                    {'default': 'auto', 'tooltip': 'auto selects existing/Sage for smaller sequences and embedded Sol for large sequences without changing H3 tokens. existing forces current Sage/Comfy attention. sol/scheduled_sol force the embedded Apache-2.0 SM120 Sol path.'},
                ),
                'sol_tau_start': ('FLOAT', {'default': 1.30, 'min': 0.0, 'max': 4.0, 'step': 0.05}),
                'sol_tau_end': ('FLOAT', {'default': 0.80, 'min': 0.0, 'max': 4.0, 'step': 0.05}),
                'sol_curve': (['linear', 'cosine', 'sqrt', 'smoothstep', 'exponential', 'step'], {'default': 'linear'}),
                'sol_min_tokens': ('INT', {'default': 4096, 'min': 256, 'max': 131072, 'step': 256}),
                'sol_dense_percent': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 0.9, 'step': 0.05}),
                'sol_sink_conditioning': (['exact_kv', 'exact_kv_and_rows', 'off'], {'default': 'exact_kv'}),
                'sol_qkv_chunk_tokens': (
                    'INT',
                    {
                        'default': 8192, 'min': 0, 'max': 131072, 'step': 512,
                        'tooltip': (
                            'Stream H3 QKV projection in token chunks. In streamed mode token-level '
                            'K/V are retained as INT8+scale while Sol block summaries stay BF16; Q is '
                            'reprojected and consumed chunk-by-chunk. This targets very long single-pass '
                            'clips on limited VRAM. Manual mode uses 512-token increments so 4096/3072/2048/1536/1024/512 are selectable. 0 restores the full fused-QKV path.'
                        ),
                    },
                ),
                'sol_out_proj_chunk_tokens': (
                    'INT',
                    {
                        'default': 24576, 'min': 0, 'max': 131072, 'step': 512,
                        'tooltip': (
                            'Token chunk size for the embedded Sol output projection. '
                            'Smaller values reduce peak VRAM; larger values are faster. Manual mode uses 512-token increments for fine low-VRAM tuning. '
                            '0 disables out_proj chunking.'
                        ),
                    },
                ),
                'vram_activation_reserve_mb': (
                    'INT',
                    {
                        'default': 4096, 'min': 0, 'max': 12288, 'step': 256,
                        'tooltip': (
                            'Extra VRAM headroom requested from ComfyUI before model loading. '
                            'ComfyUI will keep fewer H3 weights resident and offload more to RAM, '
                            'leaving this space for long-sequence activations. 0 disables the extra reserve.'
                        ),
                    },
                ),
                'inter_block_vram_guard_mb': (
                    'INT',
                    {
                        'default': 2048, 'min': 0, 'max': 8192, 'step': 128,
                        'tooltip': (
                            'Minimum driver-free VRAM target between H3 transformer blocks. '
                            'When free VRAM falls below this value and PyTorch is holding >=256 MB '
                            'of dead reserved cache, LongMedia returns that cache to the driver. '
                            '0 disables inter-block trimming.'
                        ),
                    },
                ),
                'inter_block_guard_cooldown_blocks': (
                    'INT',
                    {
                        'default': 4, 'min': 0, 'max': 32, 'step': 1,
                        'tooltip': (
                            'Completed H3 blocks to wait between normal cache trims. '
                            'Emergency pressure bypasses this cooldown. 0 restores the 0.2.36 behavior.'
                        ),
                    },
                ),
                'inter_block_guard_emergency_mb': (
                    'INT',
                    {
                        'default': 512, 'min': 0, 'max': 4096, 'step': 128,
                        'tooltip': (
                            'Emergency driver-free VRAM threshold. Below this value the emergency guard '
                            'may trim even while the normal guard is cooling down. 0 disables emergency mode.'
                        ),
                    },
                ),
                'inter_block_guard_emergency_cooldown_blocks': (
                    'INT',
                    {
                        'default': 3, 'min': 0, 'max': 32, 'step': 1,
                        'tooltip': (
                            'Minimum completed H3 blocks between EMERGENCY cache trims. '
                            'This prevents Dynamic VRAM/AIMDO free==0 states from causing a trim storm. '
                            '0 restores the 0.2.37 immediate-emergency behavior.'
                        ),
                    },
                ),
                'late_block_guard_start': (
                    'INT',
                    {
                        'default': 40, 'min': 0, 'max': 127, 'step': 1,
                        'tooltip': 'First H3 transformer block where the late hard guard is allowed to run. 40 targets only the tail of the 50-block H3 stack.',
                    },
                ),
                'late_block_guard_target_mb': (
                    'INT',
                    {
                        'default': 6144, 'min': 0, 'max': 12288, 'step': 256,
                        'tooltip': 'Driver-free VRAM target before attention/FFN in late H3 blocks. 0 disables the late-block hard guard.',
                    },
                ),
                'late_block_guard_min_cached_mb': (
                    'INT',
                    {
                        'default': 512, 'min': 0, 'max': 4096, 'step': 128,
                        'tooltip': 'Minimum reclaimable PyTorch CUDA cache required before a late-block hard trim is attempted.',
                    },
                ),
                'step_boundary_cleanup_mb': (
                    'INT',
                    {
                        'default': 2048, 'min': 0, 'max': 8192, 'step': 128,
                        'tooltip': 'Minimum driver-free VRAM target after each completed denoise step. Dead allocator cache is returned before the next H3 forward. 0 disables.',
                    },
                ),
                'refine_enabled': (
                    'BOOLEAN',
                    {'default': False, 'tooltip': 'Run the full connected SIGMAS schedule, then add extra low-noise refine steps using the exact tail sigmas from the base schedule.'},
                ),
                'refine_add_noise': (
                    'BOOLEAN',
                    {'default': False, 'tooltip': 'Legacy compatibility input. Ignored: a true refine stage always continues with no fresh noise.'},
                ),
                'refine_seed': (
                    'INT',
                    {'default': 0, 'min': 0, 'max': 18446744073709551615, 'tooltip': 'Legacy compatibility input. Ignored: refine continues the same trajectory and does not generate fresh noise.'},
                ),
                'refine_steps': (
                    'INT',
                    {
                        'default': 2, 'min': 1, 'max': 1000, 'step': 1,
                        'tooltip': (
                            'How many extra low-noise steps to run after the complete base sampler. '
                            'Example: steps=12, refine_steps=3 -> stage1 runs 9 steps, stage2 runs the final 3 steps of the same 12-step schedule.'
                        ),
                    },
                ),
                'memory_mode': (
                    ['auto', 'normal', 'low_vram', 'ultra_low_vram'],
                    {'default': 'auto', 'tooltip': 'Sampler-local residency policy. auto selects from model-size/VRAM ratio; low_vram and ultra_low_vram work without ComfyUI launch flags.'},
                ),
                'sampler_mode': (
                    ['auto', 'manual'],
                    {
                        'default': 'auto',
                        'tooltip': 'auto uses the validated 0.4.0 attention/VRAM policy. manual exposes all low-level tuning widgets.',
                    },
                ),
            }
        }

    RETURN_TYPES = ('LATENT', 'INT', 'INT', 'INT', 'STRING')
    RETURN_NAMES = ('final_av', 'total_frames', 'trim_frames', 'passes', 'report')
    FUNCTION = 'sample'
    CATEGORY = CATEGORY_LONGMEDIA

    def sample(self, initial_av, long_media_plan, guider, sampler, sigmas, seed,
               video_context_denoise=0.0, audio_context_denoise=0.0,
               offload_completed_segments=True, mlp_chunk_tokens=8192,
               attention_mode='auto', sol_tau_start=1.3, sol_tau_end=0.8,
               sol_curve='linear', sol_min_tokens=4096, sol_dense_percent=0.0,
               sol_sink_conditioning='exact_kv', sol_qkv_chunk_tokens=8192, sol_out_proj_chunk_tokens=24576,
               vram_activation_reserve_mb=4096, inter_block_vram_guard_mb=2048,
               inter_block_guard_cooldown_blocks=4, inter_block_guard_emergency_mb=512, inter_block_guard_emergency_cooldown_blocks=3,
               late_block_guard_start=40, late_block_guard_target_mb=6144, late_block_guard_min_cached_mb=512,
               step_boundary_cleanup_mb=2048,
               refine_enabled=False, refine_add_noise=False, refine_seed=0,
               refine_steps=2,
               memory_mode='auto',
               sampler_mode='auto'):
        from comfy_execution.graph_utils import GraphBuilder

        plan = long_media_plan
        graph = GraphBuilder()
        sampler_mode = str(sampler_mode or 'auto')
        requested_memory_mode = str(memory_mode or 'auto')
        memory_profile = _resolve_h3_memory_mode(guider, requested_memory_mode)
        memory_mode = str(memory_profile['effective'])
        # v0.3.60: every mode uses the same adaptive governor.  Modes differ
        # only by safety envelope; no mode is allowed to blindly run into OOM.
        # Fixed 8GB ultra reserves left several GB of a 16GB card idle, so reserve
        # is now a smaller planning margin and runtime driver-free floors own safety.
        _ms, _gs = memory_profile.get('model_bytes'), memory_profile.get('gpu_bytes')
        _ratio = (float(_ms) / float(_gs)) if (_ms and _gs) else 0.0
        try:
            import psutil as _psutil
            _vm = _psutil.virtual_memory()
            _ram_avail_gb = float(_vm.available) / (1024.0 ** 3)
        except Exception:
            _ram_avail_gb = 0.0
        offload_completed_segments = True if memory_mode in ('low_vram','ultra_low_vram') or _ratio > 1.0 else bool(offload_completed_segments)
        if memory_mode == 'normal':
            vram_activation_reserve_mb = max(256, min(int(vram_activation_reserve_mb), 768))
            inter_block_vram_guard_mb = max(768, min(int(inter_block_vram_guard_mb), 1280))
            late_block_guard_target_mb = max(1792, min(int(late_block_guard_target_mb), 3072))
            step_boundary_cleanup_mb = max(1024, min(int(step_boundary_cleanup_mb), 1792))
            mlp_chunk_tokens = min(max(int(mlp_chunk_tokens), 512), 8192)
        elif memory_mode == 'low_vram':
            vram_activation_reserve_mb = max(768, min(int(vram_activation_reserve_mb), 1536))
            inter_block_vram_guard_mb = max(1280, min(int(inter_block_vram_guard_mb), 2048))
            late_block_guard_target_mb = max(2560, min(int(late_block_guard_target_mb), 4096))
            step_boundary_cleanup_mb = max(1536, min(int(step_boundary_cleanup_mb), 2560))
            mlp_chunk_tokens = min(max(int(mlp_chunk_tokens), 256), 4096)
        else:  # ultra_low_vram
            vram_activation_reserve_mb = max(1792, min(int(vram_activation_reserve_mb), 2816))
            inter_block_vram_guard_mb = max(2048, min(int(inter_block_vram_guard_mb), 3072))
            late_block_guard_target_mb = max(3328, min(int(late_block_guard_target_mb), 4608))
            step_boundary_cleanup_mb = max(2048, min(int(step_boundary_cleanup_mb), 3072))
            inter_block_guard_cooldown_blocks = min(max(int(inter_block_guard_cooldown_blocks), 1), 3)
            mlp_chunk_tokens = min(max(int(mlp_chunk_tokens), 128), 2048)

        _lm_print('[MiniMaxH3 LongMedia][0.3.60 MEMORY POLICY V3] '
            f"requested={memory_profile['requested']} effective={memory_mode}; model={(float(_ms)/(1024**3)) if _ms else 0.0:.1f}GB GPU={(float(_gs)/(1024**3)) if _gs else 0.0:.1f}GB; "
            f"reason={memory_profile['reason']}; MLP={int(mlp_chunk_tokens)} QKV={int(sol_qkv_chunk_tokens)} OUT={int(sol_out_proj_chunk_tokens)} reserve={int(vram_activation_reserve_mb)}MB", flush=True)
        requested_attention_mode = str(attention_mode or 'auto')
        effective_attention_mode = requested_attention_mode
        if int(getattr(plan, 'passes', 1)) > 1 and requested_attention_mode == 'auto':
            # AUTO previously made an independent token-threshold decision inside
            # every pass. H3 alignment and per-pass conditioning change sequence
            # length, so one movie could silently use existing/Sage in pass 0 and
            # approximate Sol in pass 1. V14-V17 established that this operator
            # switch is not quality-neutral on INT8/W4A8. A segmented job therefore
            # keeps one exact attention family unless the user explicitly forces Sol.
            effective_attention_mode = 'existing'
            _lm_print(
                '[MiniMaxH3 LongMedia][V327 ATTENTION CONTINUITY LOCK] '
                'segmented AUTO -> existing for every pass; force sol/scheduled_sol explicitly to override',
                flush=True,
            )
        attention_mode = effective_attention_mode
        if sampler_mode == 'auto':
            # v0.3.22 A/B override mode: INPUT_TYPES defaults remain the validated
            # production AUTO policy, but explicit widget edits are honored. This
            # lets AUTO routing be compared against forced existing/SOL without
            # switching to Manual and changing any other sampler state.
            _lm_print(
                '[MiniMaxH3 LongMedia][V322 AUTO OVERRIDES] production defaults active; '
                f'attention_mode={requested_attention_mode}->{effective_attention_mode}, '
                f'tau={float(sol_tau_start):.3f}->{float(sol_tau_end):.3f}, '
                f'mlp_chunk={int(mlp_chunk_tokens)}',
                flush=True,
            )
        requested_mlp_chunk_tokens = int(mlp_chunk_tokens)
        effective_mlp_chunk_tokens = requested_mlp_chunk_tokens if requested_mlp_chunk_tokens > 0 else (1 << 30)
        mlp_chunking_enabled = requested_mlp_chunk_tokens > 0
        try:
            _sig = sigmas.detach().float().cpu() if torch.is_tensor(sigmas) else torch.as_tensor(sigmas, dtype=torch.float32)
            sol_sigma_hi = float(_sig[0]) if _sig.numel() else 1.0
            _nonzero = _sig[_sig > 0]
            sol_sigma_lo = float(_nonzero[-1]) if _nonzero.numel() else 0.0
        except Exception:
            sol_sigma_hi, sol_sigma_lo = 1.0, 0.0
        mlp_chunker = graph.node(
            "MiniMaxH3LatentLabMLPChunking",
            guider=guider,
            chunk_tokens=effective_mlp_chunk_tokens,
            max_blocks=128,
            sol_mode=str(attention_mode),
            sol_tau_start=float(sol_tau_start),
            sol_tau_end=float(sol_tau_end),
            sol_curve=str(sol_curve),
            sol_min_tokens=int(sol_min_tokens),
            sol_dense_percent=float(sol_dense_percent),
            sol_sink_conditioning=str(sol_sink_conditioning),
            sol_qkv_chunk_tokens=int(sol_qkv_chunk_tokens),
            sol_out_proj_chunk_tokens=int(sol_out_proj_chunk_tokens),
            vram_activation_reserve_mb=int(vram_activation_reserve_mb),
            inter_block_vram_guard_mb=int(inter_block_vram_guard_mb),
            inter_block_guard_cooldown_blocks=int(inter_block_guard_cooldown_blocks),
            inter_block_guard_emergency_mb=int(inter_block_guard_emergency_mb),
            inter_block_guard_emergency_cooldown_blocks=int(inter_block_guard_emergency_cooldown_blocks),
            late_block_guard_start=int(late_block_guard_start),
            late_block_guard_target_mb=int(late_block_guard_target_mb),
            late_block_guard_min_cached_mb=int(late_block_guard_min_cached_mb),
            step_boundary_cleanup_mb=int(step_boundary_cleanup_mb),
            sol_sigma_hi=float(sol_sigma_hi),
            sol_sigma_lo=float(sol_sigma_lo),
            memory_mode=str(memory_mode),
            requested_memory_mode=str(requested_memory_mode),
        )
        traced_guider = mlp_chunker.out(0)
        ultra_pin_previous = None
        _out_of_core = bool(_ms and _gs and float(_ms) > float(_gs) * 1.05)
        if _out_of_core:
            ultra_pin_gate = graph.node(
                "MiniMaxH3LatentLabUltraPinnedMemoryGate",
                guider=traced_guider,
                enable=True,
            )
            traced_guider = ultra_pin_gate.out(0)
            ultra_pin_previous = ultra_pin_gate.out(1)
        block_trace_state = mlp_chunker.out(1)
        memory_profiler = graph.node(
            "MiniMaxH3LatentLabFirstStepMemoryProfiler",
            sampler=sampler,
            max_history_entries=20000,
            block_trace_state=block_trace_state,
        )
        profiled_sampler = memory_profiler.out(0)
        memory_profile_state = memory_profiler.out(1)
        # 0.2.16 diagnostic build deliberately disables intra-step cache flushing:
        # we want an undistorted first-step allocator trace, including OOM events.
        guard_state = None
        # 0.4.1 refine policy: when enabled, one connected SIGMAS trajectory is
        # split into two KSampler-Advanced-style stages. Stage 1 stops before the
        # final R intervals and returns the in-progress latent; stage 2 continues
        # the same schedule with add_noise disabled and performs the final denoise.
        main_sigmas = sigmas
        refine_sigmas = None
        refine_steps_effective = 0
        refine_tail_start = 0
        base_steps = max(0, int(sigmas.numel()) - 1) if torch.is_tensor(sigmas) else max(0, len(sigmas) - 1)
        if bool(refine_enabled):
            main_sigmas, refine_sigmas, base_steps, refine_tail_start, refine_steps_effective, _ = split_refine_sigmas(
                sigmas, int(refine_steps)
            )
            _lm_print(
                '[MiniMaxH3 LongMedia][0.4.2 ADDITIVE FINAL-LATENT REFINE] '
                f'base_steps={int(base_steps)} + refine_steps={int(refine_steps_effective)}; '
                f'refine_tail_start={int(refine_tail_start)}; main sampler keeps full SIGMAS; '
                'refiner calls stock comfy.sample.sample KSampler Advanced path; H3 audio is frozen with denoise=0; refiner AV output is passed through directly without post-refine repack',
                flush=True,
            )

        first_effective_seed = _v85_segment_seed(plan, seed, 0) if getattr(plan, 'mode', None) == 'multiclip' else int(seed)
        first_noise = graph.node("RandomNoise", noise_seed=int(first_effective_seed))
        _lm_print(
            (f'[MiniMaxH3 LongMedia][0.3.87 MULTICLIP SEED] clip=1 seed={int(first_effective_seed) & 0xFFFFFFFFFFFFFFFF}'
             if getattr(plan, 'mode', None) == 'multiclip'
             else f'[MiniMaxH3 LongMedia][V62 SAME SEED] pass=0 seed={int(seed) & 0xFFFFFFFFFFFFFFFF}'),
            flush=True,
        )
        first_sample = graph.node(
            "SamplerCustomAdvanced",
            noise=first_noise.out(0),
            guider=traced_guider,
            sampler=profiled_sampler,
            sigmas=main_sigmas,
            latent_image=initial_av,
        )
        first_main_output = first_sample.out(0)
        if bool(refine_enabled) and int(refine_steps_effective) > 0:
            first_refine = graph.node(
                "MiniMaxH3LatentLabStockAdvancedRefiner",
                guider=traced_guider,
                final_av=first_main_output,
                sigmas=refine_sigmas,
                start_at_step=int(refine_tail_start),
                seed=int(first_effective_seed),
            )
            first_main_output = first_refine.out(0)
            _lm_print(
                '[MiniMaxH3 LongMedia][0.4.1 TRUE ADVANCED REFINER] '
                'stage2_output=stock_solver_state; x0_substitution=False; stream_freeze=False',
                flush=True,
            )
        # Pass 0 has no frozen continuation head. The optional second-stage refine
        # has completed before this latent becomes continuation state.
        previous_segment = first_main_output
        stitched = previous_segment

        for segment_index in range(1, plan.passes):
            if getattr(plan, 'mode', None) == 'storyboard_bridge':
                storyboard_avs = getattr(plan, 'storyboard_segment_avs', None)
                if not storyboard_avs or segment_index >= len(storyboard_avs):
                    raise RuntimeError('V64 storyboard pass latent is missing from LongMediaPlan')
                prepared_av = storyboard_avs[segment_index]
            else:
                prepared = graph.node(
                    "MiniMaxH3LatentLabLongMediaNextSegment",
                    long_media_plan=plan,
                    previous_av=previous_segment,
                    segment_index=segment_index,
                    video_context_denoise=float(video_context_denoise),
                    audio_context_denoise=float(audio_context_denoise),
                )
                prepared_av = prepared.out(0)
            # V62: one long shot owns one stochastic trajectory.  Continuation
            # passes reuse the exact same base seed; motion/keyframe context, not
            # a fresh random seed, is responsible for advancing the scene.
            segment_effective_seed = (
                _v85_segment_seed(plan, seed, segment_index)
                if getattr(plan, 'mode', None) == 'multiclip'
                else int(seed) & 0xFFFFFFFFFFFFFFFF
            )
            noise = graph.node(
                "RandomNoise",
                noise_seed=int(segment_effective_seed),
            )
            _lm_print(
                (f'[MiniMaxH3 LongMedia][0.3.87 MULTICLIP SEED] clip={segment_index + 1} seed={int(segment_effective_seed)}'
                 if getattr(plan, 'mode', None) == 'multiclip'
                 else f'[MiniMaxH3 LongMedia][V62 SAME SEED] pass={segment_index} seed={int(seed) & 0xFFFFFFFFFFFFFFFF} (same as pass 0)'),
                flush=True,
            )
            if getattr(plan, 'mode', None) == 'storyboard_bridge':
                segment_guider = _clone_guider_with_segment_audio(
                    guider, plan, segment_index, previous_av=None,
                )
            else:
                runtime_guider = graph.node(
                    "MiniMaxH3LatentLabRuntimeContinuationGuider",
                    guider=guider,
                    long_media_plan=plan,
                    previous_av=previous_segment,
                    segment_index=segment_index,
                )
                segment_guider = runtime_guider.out(0)
            segment_mlp_chunker = graph.node(
                "MiniMaxH3LatentLabMLPChunking",
                guider=segment_guider,
                chunk_tokens=effective_mlp_chunk_tokens,
                max_blocks=128,
                sol_mode=str(attention_mode),
                sol_tau_start=float(sol_tau_start),
                sol_tau_end=float(sol_tau_end),
                sol_curve=str(sol_curve),
                sol_min_tokens=int(sol_min_tokens),
                sol_dense_percent=float(sol_dense_percent),
                sol_sink_conditioning=str(sol_sink_conditioning),
                sol_qkv_chunk_tokens=int(sol_qkv_chunk_tokens),
                sol_out_proj_chunk_tokens=int(sol_out_proj_chunk_tokens),
                vram_activation_reserve_mb=int(vram_activation_reserve_mb),
                inter_block_vram_guard_mb=int(inter_block_vram_guard_mb),
                inter_block_guard_cooldown_blocks=int(inter_block_guard_cooldown_blocks),
                inter_block_guard_emergency_mb=int(inter_block_guard_emergency_mb),
                inter_block_guard_emergency_cooldown_blocks=int(inter_block_guard_emergency_cooldown_blocks),
                late_block_guard_start=int(late_block_guard_start),
                late_block_guard_target_mb=int(late_block_guard_target_mb),
                late_block_guard_min_cached_mb=int(late_block_guard_min_cached_mb),
                step_boundary_cleanup_mb=int(step_boundary_cleanup_mb),
                sol_sigma_hi=float(sol_sigma_hi),
                sol_sigma_lo=float(sol_sigma_lo),
                memory_mode=str(memory_mode),
                requested_memory_mode=str(requested_memory_mode),
            )
            sampled = graph.node(
                "SamplerCustomAdvanced",
                noise=noise.out(0),
                guider=segment_mlp_chunker.out(0),
                sampler=profiled_sampler,
                sigmas=main_sigmas,
                latent_image=prepared_av,
            )
            sampled_output = sampled.out(0)
            if bool(refine_enabled) and int(refine_steps_effective) > 0:
                refined = graph.node(
                    "MiniMaxH3LatentLabStockAdvancedRefiner",
                    guider=segment_mlp_chunker.out(0),
                    final_av=sampled_output,
                    sigmas=refine_sigmas,
                    start_at_step=int(refine_tail_start),
                    seed=int(segment_effective_seed),
                )
                sampled_output = refined.out(0)
                _lm_print(
                    f'[MiniMaxH3 LongMedia][0.4.1 TRUE ADVANCED REFINER] clip={int(segment_index)+1}; '
                    'stage2_output=stock_solver_state; x0_substitution=False; stream_freeze=False',
                    flush=True,
                )
            if bool(refine_enabled) and getattr(plan, 'mode', None) != 'storyboard_bridge' and int(plan.overlap_frames) > 0:
                # Preserve the exact frozen AV continuation head after both the
                # both stages of the continuous refine trajectory.
                protected = graph.node(
                    "MiniMaxH3LatentLabProtectRefineAV",
                    base_av=prepared_av,
                    refined_av=sampled_output,
                    overlap_frames=int(plan.overlap_frames),
                )
                sampled_output = protected.out(0)
            joined = graph.node(
                "MiniMaxH3LatentLabStitchContinuation",
                previous_av=stitched,
                sampled_continuation_av=sampled_output,
                overlap_frames=(0 if getattr(plan, 'mode', None) == 'storyboard_bridge' else plan.overlap_frames),
                blend_video_overlap=False,  # continuity policy: hidden frozen overlap is context only; never re-blend it
                offload_to_cpu=bool(offload_completed_segments),
            )
            previous_segment = sampled_output
            stitched = joined.out(0)

        try:
            _profile_video, _profile_audio = unpack_av_samples(initial_av)
            input_geometry = {
                "video_shape": list(_profile_video.shape),
                "audio_shape": list(_profile_audio.shape),
                "video_dtype": str(_profile_video.dtype),
                "audio_dtype": str(_profile_audio.dtype),
                "latent_payload_mb": _mb(
                    _profile_video.numel() * _profile_video.element_size()
                    + _profile_audio.numel() * _profile_audio.element_size()
                ),
            }
        except Exception as _profile_exc:
            input_geometry = {"error": f"{type(_profile_exc).__name__}: {_profile_exc}"}

        report = json.dumps({
            "passes": plan.passes,
            "segmentation_active": bool(plan.passes > 1),
            "segment_lengths_frames": list(plan.segment_lengths),
            "segment_starts_frames": list(plan.segment_starts),
            "overlap_frames": int(plan.overlap_frames),
            "sequential_context_carry": bool(plan.passes > 1),
            "advanced_refine_enabled": bool(refine_enabled),
            "advanced_refine_add_noise": False,
            "advanced_refine_legacy_add_noise_input_ignored": bool(refine_add_noise),
            "advanced_refine_seed": None,
            "advanced_refine_legacy_seed_input_ignored": int(refine_seed) & 0xFFFFFFFFFFFFFFFF,
            "advanced_refine_steps_requested": int(refine_steps),
            "advanced_refine_steps_effective": int(refine_steps_effective),
            "advanced_refine_base_steps": int(base_steps),
            "advanced_refine_total_model_steps": int(base_steps),
            "advanced_refine_tail_start": int(refine_tail_start),
            "advanced_refine_interval_policy": "single_connected_schedule_split_between_main_and_refiner",
            "advanced_refine_latent_source": "stage1_in_progress_solver_state",
            "advanced_refine_latent_clone": False,
            "advanced_refine_latent_rebuild": False,
            "advanced_refine_renoise": False,
            "advanced_refine_scheduler_source": "true_connected_two_stage_schedule",
            "advanced_refine_audio_policy": "joint_AV_continues_through_both_sampling_stages",
            "advanced_refine_overlap_policy": "restore_exact_pre_sampling_frozen_av_head",
            "hybrid_keyframe_scope": (
                "first_only_pass0_last_only_final"
                if getattr(plan, 'mode', None) == 'hybrid' and plan.passes > 1
                else "unchanged"
            ),
            "continuation_driver": ("V64_storyboard_A_to_B_to_C_exact_shared_bridge" if getattr(plan, 'mode', None) == 'storyboard_bridge' else "V331_stateful_frozen_overlap_motion_context"),
            "motion_context": {
                "enabled": bool(plan.passes > 1),
                "prototype_frames": 56,
                "source": "previous_generated_h3_video_latent_tail",
                "vae_roundtrip": False,
                "te_roundtrip": False,
            },
            "segment_prompting": "V331_iterative_completed_event_state_plus_local_timeline_native_refs",
            "conditioning_payload_copy": "shared_read_only_media_metadata_no_deepcopy",
            "stitched_single_output": True,
            "stitch_policy": {
                "hidden_overlap": "exact_context_trim_no_blend",
                "cross_time_visible_latent_blend": False,
                "first_visible_continuation_step_preserved": True,
            },
            "memory_mode": {
                "requested": str(memory_profile.get('requested')),
                "effective": str(memory_profile.get('effective')),
                "reason": str(memory_profile.get('reason')),
                "model_gb": (round(float(memory_profile.get('model_bytes')) / (1024**3), 3) if memory_profile.get('model_bytes') else None),
                "gpu_gb": (round(float(memory_profile.get('gpu_bytes')) / (1024**3), 3) if memory_profile.get('gpu_bytes') else None),
            },
            "low_vram_mlp": {
                "mode": "token_chunk_exact",
                "enabled": bool(mlp_chunking_enabled),
                "chunk_tokens_requested": int(requested_mlp_chunk_tokens),
                "chunk_tokens_effective": int(effective_mlp_chunk_tokens),
                "attention_unchanged": str(attention_mode) in ("auto", "existing"),
            },
            "attention": {
                "mode": str(effective_attention_mode),
                "requested_mode": str(requested_attention_mode),
                "continuity_locked": bool(
                    int(getattr(plan, 'passes', 1)) > 1
                    and requested_attention_mode == 'auto'
                ),
                "embedded_sol": str(attention_mode) in ("sol", "scheduled_sol"),
                "sol_tau_start": float(sol_tau_start),
                "sol_tau_end": float(sol_tau_end),
                "sol_curve": str(sol_curve),
                "sol_min_tokens": int(sol_min_tokens),
                "sol_dense_percent": float(sol_dense_percent),
                "sol_sink_conditioning": str(sol_sink_conditioning),
                "sol_qkv_chunk_tokens": int(sol_qkv_chunk_tokens),
                "sol_qkv_streaming_enabled": int(sol_qkv_chunk_tokens) > 0,
                "sol_out_proj_chunk_tokens": int(sol_out_proj_chunk_tokens),
                "sol_out_proj_chunking_enabled": int(sol_out_proj_chunk_tokens) > 0,
                "vram_activation_reserve_mb": int(vram_activation_reserve_mb),
                "inter_block_vram_guard_mb": int(inter_block_vram_guard_mb),
                "inter_block_guard_cooldown_blocks": int(inter_block_guard_cooldown_blocks),
                "inter_block_guard_emergency_mb": int(inter_block_guard_emergency_mb),
                "inter_block_guard_emergency_cooldown_blocks": int(inter_block_guard_emergency_cooldown_blocks),
                "late_block_guard_start": int(late_block_guard_start),
                "late_block_guard_target_mb": int(late_block_guard_target_mb),
                "late_block_guard_min_cached_mb": int(late_block_guard_min_cached_mb),
                "step_boundary_cleanup_mb": int(step_boundary_cleanup_mb),
                "mlp_inplace_reuse": True,
                "implementation": "LongMedia embedded SM120 BF16 Sol-Attn (Apache-2.0 adapted)",
            },
            "input_geometry": input_geometry,
            "total_frames": plan.output_frames,
            "trim_frames": plan.trim_frames,
            "audio_reference_timeline": (
                "cropped_per_pass" if plan.mode == "automatic_lip_sync" else "full"
            ),
            "video_context_denoise": float(video_context_denoise),
            "audio_context_denoise": float(audio_context_denoise),
            "offload_completed_segments": bool(offload_completed_segments),
        }, indent=2)

        # Always run once after the final sampling pass, including the common
        # one-segment case where manual_duration == segment duration. This only
        # releases unused allocator cache; it deliberately does not unload H3.
        cleanup = graph.node(
            "MiniMaxH3LatentLabVRAMCacheCleanup",
            latent=stitched,
            sampler_report=report,
            memory_profile_state=memory_profile_state,
            block_trace_state=block_trace_state,
        )
        final_latent = cleanup.out(0)
        if ultra_pin_previous is not None:
            ultra_pin_restore = graph.node(
                "MiniMaxH3LatentLabUltraPinnedMemoryRestore",
                final_av=final_latent,
                previous_disable_pinned_memory=ultra_pin_previous,
                restore=True,
            )
            final_latent = ultra_pin_restore.out(0)
        return {
            "result": (final_latent, plan.output_frames, plan.trim_frames, plan.passes, cleanup.out(1)),
            "expand": graph.finalize(),
        }


class MiniMaxH3LatentLabLongMediaDecode:
    DESCRIPTION = (
        'Decode the final stitched H3 AV latent back to pixel frames and audio. '
        'This is the single combined result after any multi-pass long-media segmentation.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'final_av': ('LATENT',),
                'long_media_plan': ('LONG_MEDIA_PLAN',),
                'enable_tiling': ('BOOLEAN', {'default': True}),
                'tile_size': (
                    'INT',
                    {'default': 256, 'min': 32, 'max': 2048, 'step': 32},
                ),
                'width': (
                    'INT',
                    {'default': 512, 'min': 32, 'max': 8192, 'step': 32},
                ),
                'temporal_size': (
                    'INT',
                    {'default': 32, 'min': 1, 'max': 256, 'step': 1},
                ),
                'batch_size': (
                    'INT',
                    {'default': 1, 'min': 1, 'max': 16, 'step': 1},
                ),
                'color_match_strength': (
                    'FLOAT',
                    {
                        'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                        'tooltip': (
                            '0 disables. >0 nudges every frame\'s color statistics '
                            'toward frame 0, useful when frame 0 is pinned to a '
                            'reference and the rest of the clip drifts in color '
                            '(visible as a jump at a loop seam).'
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ('IMAGE', 'AUDIO', 'FLOAT', 'STRING')
    RETURN_NAMES = ('images', 'audio', 'duration_seconds', 'report')
    FUNCTION = 'decode'
    CATEGORY = CATEGORY_LONGMEDIA

    def decode(self, final_av, long_media_plan, enable_tiling, tile_size, width, temporal_size,
               batch_size, color_match_strength=0.0):
        plan = long_media_plan
        decode_barrier = _release_model_memory_for_decode()
        video, audio = unpack_av_samples(final_av)
        video_vae = plan.video_vae
        audio_vae = plan.audio_vae
        images, video_decode_info = _decode_video_vae_safe(
            video_vae, video, enable_tiling, tile_size, temporal_size,
        )
        output_frames = plan.output_frames
        if images.dim() == 5:
            images = images[0]
        storyboard_duplicate_removed = False
        if getattr(plan, 'mode', None) == 'storyboard_bridge':
            bridge = int(getattr(plan, 'storyboard_bridge_frame', -1))
            if 0 < bridge < int(images.shape[0]):
                images = torch.cat((images[:bridge], images[bridge + 1:]), dim=0)
                storyboard_duplicate_removed = True
        trimmed = 0
        if images.shape[0] > output_frames:
            trimmed = images.shape[0] - output_frames
            images = images[:output_frames]
        opening_anchor_suppressed = False
        opening_anchor_suppressed_frames = 0
        if bool(getattr(plan, 'suppress_visible_opening_anchor', False)) and int(images.shape[0]) > 1:
            # Keep the proven sampling-time H3 opening anchor exactly as in the
            # known-good 0.3.32/0.3.39 path, but hide the first anchor-biased visible
            # decode frames at output time only. In practice frame 1 can still carry a
            # weak residual of the startup anchor, so for segmented_continuation we
            # promote the first clearly-generated frame (frame 2 when present) into the
            # first visible slots. This leaves all latents, continuation guides,
            # segment conditioning, and later frames untouched.
            images = images.clone()
            if int(images.shape[0]) > 2:
                replacement = images[2]
                images[0] = replacement
                images[1] = replacement
                opening_anchor_suppressed_frames = 2
            else:
                images[0] = images[1]
                opening_anchor_suppressed_frames = 1
            opening_anchor_suppressed = True
        def _compact_decode_snap(snap):
            if snap is None:
                return None
            return {
                'driver_free_mb': _mb(snap['driver_free']),
                'allocated_mb': _mb(snap['allocated']),
                'reserved_mb': _mb(snap['reserved']),
                'cached_mb': _mb(snap['cached']),
            }

        report_data = {
            'model_memory_released_before_decode': True,
            'decode_memory_barrier_before': _compact_decode_snap(decode_barrier.get('before')),
            'decode_memory_barrier_after': _compact_decode_snap(decode_barrier.get('after')),
            'decode_memory_barrier_errors': decode_barrier.get('errors', []),
            'decode_uses_plan_vaes': True,
            'video_decode': video_decode_info,
            'trimmed_video_frames': trimmed,
            'storyboard_duplicate_boundary_frame_removed': storyboard_duplicate_removed,
            'segmented_visible_opening_anchor_suppressed': opening_anchor_suppressed,
            'segmented_visible_opening_anchor_suppressed_frames': opening_anchor_suppressed_frames,
        }
        audio_output_mode = str(getattr(plan, 'audio_output_mode', 'auto') or 'auto')
        passthrough_audio_mode = audio_output_mode in ('auto', 'preserve', 'preserve_reference', 'lip_sync')
        preserve_audio_bypass = audio_output_mode in ('preserve', 'preserve_reference', 'lip_sync')

        # Preserve means literal bypass: never send the sampled/model audio stream back
        # through the H3 Audio VAE. This is intentionally checked before every generated
        # audio decode path, because distilled/Turbo LoRAs can alter the sampled audio
        # stream geometry and make it invalid for the VAE normalizer.
        if preserve_audio_bypass and plan.final_audio_override is None:
            raise RuntimeError(
                f"audio_mode={audio_output_mode!r} requires a connected source audio track, "
                "but LongMediaPlan has no final_audio_override. Connect audio_1 (or another "
                "audio input) or switch audio_mode to generate/reference_only."
            )

        # Pixel override/blend is an output policy, not a lip-sync-only feature.
        # Apply it for Manual hybrid first-frame workflows as well. latent_inject
        # is already baked into the sampled target and needs no post-decode edit.
        first_frame_mode = getattr(plan, 'first_frame_mode', 'latent_inject')
        first_frame_override = getattr(plan, 'first_frame_override', None)
        first_frame_latent_injected = bool(
            getattr(plan, 'first_frame_latent_injected', False)
        )
        if first_frame_override is not None and first_frame_mode in ('pixel_override', 'blend'):
            first_frame = first_frame_override
            if first_frame.dim() == 4:
                first_frame = first_frame[0]
            target_h, target_w = images.shape[1], images.shape[2]
            if first_frame.shape[0] != target_h or first_frame.shape[1] != target_w:
                import comfy.utils
                ff = first_frame.unsqueeze(0).movedim(-1, 0)
                ff = comfy.utils.common_upscale(ff, target_w, target_h, 'lanczos', 'disabled')
                first_frame = ff.squeeze(0).movedim(0, -1)
            first_frame = first_frame.to(images.dtype)
            if first_frame_mode == 'pixel_override':
                images[0] = first_frame
            else:  # blend
                images = _blend_leading_frames_to_reference(
                    images, first_frame, plan.first_frame_blend_frames,
                )
        if first_frame_override is not None or first_frame_latent_injected:
            report_data['first_frame_mode'] = first_frame_mode
            report_data['first_frame_restored'] = True
            report_data['first_frame_latent_injected'] = first_frame_latent_injected

        if plan.mode == 'automatic_lip_sync':
            if (passthrough_audio_mode and plan.final_audio_override is not None) or preserve_audio_bypass:
                audio = plan.final_audio_override
                report_data['original_audio_restored'] = True
                report_data['generated_audio_decoded'] = False
                report_data['audio_output_mode'] = audio_output_mode
                report_data['audio_vae_bypassed'] = True
            elif audio_vae is not None:
                if not hasattr(audio, 'shape') or audio.ndim != 4 or int(audio.shape[1]) != 32 or int(audio.shape[2]) != 2:
                    shape = tuple(audio.shape) if hasattr(audio, 'shape') else type(audio).__name__
                    raise RuntimeError(
                        'LongMedia received an invalid generated H3 audio latent for AudioVAE decode: '
                        f'shape={shape}, audio_mode={audio_output_mode!r}. '
                        'Use preserve_reference/preserve (or auto with attached audio) to bypass AudioVAE.'
                    )
                sr = int(getattr(audio_vae, 'audio_sample_rate', 32000))
                decoded_audio = audio_vae.decode(audio)
                audio = _normalize_decoded_audio(
                    decoded_audio, sr, round(plan.total_duration * sr)
                )
                report_data['original_audio_restored'] = False
                report_data['generated_audio_decoded'] = True
        else:
            # Restore original audio only when requested; otherwise decode model audio.
            if (passthrough_audio_mode and plan.final_audio_override is not None) or preserve_audio_bypass:
                audio = plan.final_audio_override
                report_data['generated_audio_decoded'] = False
                report_data['audio_output_mode'] = audio_output_mode
                report_data['audio_vae_bypassed'] = True
            elif audio_vae is not None and hasattr(audio, 'shape') and audio.ndim == 4:
                # MiniMax H3 AudioVAE expects latent layout [B, 32, 2, T]. A Turbo LoRA
                # or wrong routing can leave a packed/non-audio tensor here. Never hand
                # such a tensor to the VAE normalizer, which otherwise fails with an
                # opaque 19296-vs-128 broadcast error.
                if int(audio.shape[1]) != 32 or int(audio.shape[2]) != 2:
                    raise RuntimeError(
                        'LongMedia received an invalid generated H3 audio latent for AudioVAE decode: '
                        f'shape={tuple(audio.shape)}, audio_mode={audio_output_mode!r}. '
                        'For an attached source soundtrack use audio_mode=preserve_reference/preserve '
                        '(or auto, which now preserves attached audio). Use generate/reference_only only '
                        'when model-generated audio is intended.'
                    )
                sr = int(getattr(audio_vae, 'audio_sample_rate', 32000))
                decoded_audio = audio_vae.decode(audio)
                audio = _normalize_decoded_audio(
                    decoded_audio, sr, round(plan.total_duration * sr)
                )
                if getattr(plan, 'mode', None) == 'storyboard_bridge' and storyboard_duplicate_removed:
                    wave = audio['waveform']
                    cut_at = int(round((int(getattr(plan, 'storyboard_bridge_frame', 0)) / FPS) * sr))
                    one = int(round(sr / FPS))
                    if 0 <= cut_at and cut_at + one <= int(wave.shape[-1]):
                        wave = torch.cat((wave[..., :cut_at], wave[..., cut_at + one:]), dim=-1)
                        target = int(round(plan.total_duration * sr))
                        wave = torch.nn.functional.pad(wave, (0, max(0, target - int(wave.shape[-1]))))[..., :target]
                        audio = {'waveform': wave, 'sample_rate': sr}
                        report_data['storyboard_duplicate_audio_frame_removed'] = True
                report_data['generated_audio_decoded'] = True
            report_data['original_audio_restored'] = bool((passthrough_audio_mode and plan.final_audio_override is not None) or preserve_audio_bypass)
            report_data['first_frame_restored'] = False
        if color_match_strength > 0.0:
            images = _match_frames_color_to_reference(images, 0, color_match_strength)
        report_data['color_match_strength'] = float(color_match_strength)
        report = json.dumps(report_data, indent=2)
        return (images, audio, plan.total_duration, report)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3LatentLabUltraPinnedMemoryGate": MiniMaxH3LatentLabUltraPinnedMemoryGate,
    "MiniMaxH3LatentLabUltraPinnedMemoryRestore": MiniMaxH3LatentLabUltraPinnedMemoryRestore,
    'MiniMaxH3LatentLabVideoEncode': MiniMaxH3LatentLabVideoEncode,
    'MiniMaxH3LatentLabAudioEncode': MiniMaxH3LatentLabAudioEncode,
    'MiniMaxH3LatentLabPackAV': MiniMaxH3LatentLabPackAV,
    'MiniMaxH3LatentLabSplitAV': MiniMaxH3LatentLabSplitAV,
    'MiniMaxH3LatentLabReplaceStream': MiniMaxH3LatentLabReplaceStream,
    'MiniMaxH3LatentLabReplaceVideo': MiniMaxH3LatentLabReplaceVideo,  # deprecated alias
    'MiniMaxH3LatentLabReplaceAudio': MiniMaxH3LatentLabReplaceAudio,  # deprecated alias
    'MiniMaxH3LatentLabStreamDenoise': MiniMaxH3LatentLabStreamDenoise,
    'MiniMaxH3LatentLabLipSyncSetup': MiniMaxH3LatentLabLipSyncSetup,
    'MiniMaxH3LatentLabVideoInpaint': MiniMaxH3LatentLabVideoInpaint,
    'MiniMaxH3LatentLabMergeAV': MiniMaxH3LatentLabMergeAV,
    'MiniMaxH3LatentLabPrepareContinuation': MiniMaxH3LatentLabPrepareContinuation,
    'MiniMaxH3LatentLabStitchContinuation': MiniMaxH3LatentLabStitchContinuation,
    'MiniMaxH3LatentLabInfo': MiniMaxH3LatentLabInfo,
    'MiniMaxH3LongMediaPlanner': MiniMaxH3LongMediaPlanner,
    'MiniMaxH3LatentLabLongMediaSetup': MiniMaxH3LatentLabLongMediaSetup,
    'MiniMaxH3LatentLabLongMediaNextSegment': MiniMaxH3LatentLabLongMediaNextSegment,
    'MiniMaxH3LatentLabRuntimeContinuationGuider': MiniMaxH3LatentLabRuntimeContinuationGuider,
    'MiniMaxH3LatentLabRefineSigmas': MiniMaxH3LatentLabRefineSigmas,
    'MiniMaxH3LatentLabStockAdvancedRefiner': MiniMaxH3LatentLabStockAdvancedRefiner,
    'MiniMaxH3LatentLabProtectRefineAV': MiniMaxH3LatentLabProtectRefineAV,
    'MiniMaxH3LatentLabLongMediaSampler': MiniMaxH3LatentLabLongMediaSampler,
    'MiniMaxH3LatentLabLongMediaDecode': MiniMaxH3LatentLabLongMediaDecode,
    'MiniMaxH3LatentLabAttentionChunking': MiniMaxH3LatentLabAttentionChunking,
    'MiniMaxH3LatentLabBlockMemoryTracer': MiniMaxH3LatentLabBlockMemoryTracer,
    'MiniMaxH3LatentLabMLPChunking': MiniMaxH3LatentLabMLPChunking,
    'MiniMaxH3LatentLabFirstStepMemoryProfiler': MiniMaxH3LatentLabFirstStepMemoryProfiler,
    'MiniMaxH3LatentLabVRAMPressureGuard': MiniMaxH3LatentLabVRAMPressureGuard,
    'MiniMaxH3LatentLabVRAMCacheCleanup': MiniMaxH3LatentLabVRAMCacheCleanup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'MiniMaxH3LatentLabVideoEncode': 'MiniMax H3 \u2022 Encode Video Stream',
    'MiniMaxH3LatentLabAudioEncode': 'MiniMax H3 \u2022 Encode Audio Stream',
    'MiniMaxH3LatentLabPackAV': 'MiniMax H3 \u2022 Pack AV Streams',
    'MiniMaxH3LatentLabSplitAV': 'MiniMax H3 \u2022 Split AV Streams',
    'MiniMaxH3LatentLabReplaceStream': 'MiniMax H3 \u2022 Replace Stream',
    'MiniMaxH3LatentLabReplaceVideo': 'MiniMax H3 \u2022 Replace Video Stream (deprecated)',
    'MiniMaxH3LatentLabReplaceAudio': 'MiniMax H3 \u2022 Replace Audio Stream (deprecated)',
    'MiniMaxH3LatentLabStreamDenoise': 'MiniMax H3 \u2022 Stream Denoise Controls',
    'MiniMaxH3LatentLabLipSyncSetup': 'MiniMax H3 \u2022 LipSync Latent Setup',
    'MiniMaxH3LatentLabVideoInpaint': 'MiniMax H3 \u2022 Video Inpaint',
    'MiniMaxH3LatentLabMergeAV': 'MiniMax H3 \u2022 Merge AV Latents',
    'MiniMaxH3LatentLabPrepareContinuation': 'MiniMax H3 \u2022 Prepare Continuation',
    'MiniMaxH3LatentLabStitchContinuation': 'MiniMax H3 \u2022 Stitch Continuation',
    'MiniMaxH3LatentLabInfo': 'MiniMax H3 \u2022 AV Latent Info',
    'MiniMaxH3LongMediaPlanner': 'MiniMax H3 LongMedia Planner',
    'MiniMaxH3LatentLabLongMediaSetup': 'MiniMax H3 \u2022 Long Media Setup',
    'MiniMaxH3LatentLabLongMediaNextSegment': 'MiniMax H3 \u2022 Long Media Next Segment',
    'MiniMaxH3LatentLabRuntimeContinuationGuider': 'MiniMax H3 \u2022 Runtime Continuation Guider',
    'MiniMaxH3LatentLabLongMediaSampler': 'MiniMax H3 \u2022 Long Media Sampler',
    'MiniMaxH3LatentLabLongMediaDecode': 'MiniMax H3 \u2022 Long Media Decode',
    'MiniMaxH3LatentLabAttentionChunking': 'MiniMax H3 \u2022 Low-VRAM Attention Chunking (internal)',
    'MiniMaxH3LatentLabMLPChunking': 'MiniMax H3 \u2022 Low-VRAM MLP Chunking (internal)',
    'MiniMaxH3LatentLabFirstStepMemoryProfiler': 'MiniMax H3 \u2022 First-Step Memory Profiler (internal)',
    'MiniMaxH3LatentLabVRAMPressureGuard': 'MiniMax H3 \u2022 VRAM Pressure Guard (internal)',
    'MiniMaxH3LatentLabVRAMCacheCleanup': 'MiniMax H3 \u2022 VRAM Cache Cleanup (internal)',
}

replace = _dc_replace
