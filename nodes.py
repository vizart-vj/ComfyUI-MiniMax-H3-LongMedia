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
import torch
import torchaudio
import comfy.nested_tensor
import comfy.model_management

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


# Release console policy: keep normal runs quiet. Set MINIMAX_H3_LONGMEDIA_VERBOSE=1
# to restore development telemetry. Errors, OOM/fallbacks and actual cleanup/trim
# events remain visible because they can require user action.
_LONGMEDIA_VERBOSE = os.environ.get("MINIMAX_H3_LONGMEDIA_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}
_LONGMEDIA_ALWAYS_CONSOLE = (
    "oom", "error", "failed", "failure", "fallback", "exception",
    "cleanup failed", "emergency trim", "storage guard] trim", "late guard] late guard trim",
)

def _lm_print(*args, **kwargs):
    if _LONGMEDIA_VERBOSE:
        return builtins.print(*args, **kwargs)
    text = " ".join(str(a) for a in args).lower()
    if any(marker in text for marker in _LONGMEDIA_ALWAYS_CONSOLE):
        return builtins.print(*args, **kwargs)
    return None


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
    """Release GPU model memory before VAE decode to prevent OOM."""
    _free_cuda_memory()


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
    if unload_models:
        try:
            comfy.model_management.unload_all_models()
        except Exception as exc:
            unload_error = f'{type(exc).__name__}: {exc}'
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





def _h3_runtime_prefetch_wrapper(executor, *args, **kwargs):
    """Disable Comfy dynamic-VBAR prefetch for INT8 at DIFFUSION_MODEL boundary.

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

    if backend in ('int8', 'int8-convrot-w4a4') and disable_requested:
        transformer_options['prefetch_dynamic_vbars'] = False

        if not transformer_options.get(
            'latentlab_int8_prefetch_disable_announced_v7', False
        ):
            _lm_print(
                '[MiniMaxH3 LongMedia][INT8 PREFETCH V7] '
                f'located transformer_options at {location}; '
                f'prefetch_dynamic_vbars {before!r}->False; '
                f'backend={backend}; DynamicVRAM remains enabled',
                flush=True,
            )
            transformer_options[
                'latentlab_int8_prefetch_disable_announced_v7'
            ] = True

    # Diagnostic invariant: for INT8 this MUST be False immediately before
    # MiniMaxH3Model._forward calls make_prefetch_queue().
    if backend in ('int8', 'int8-convrot-w4a4'):
        effective = transformer_options.get(
            'prefetch_dynamic_vbars', '<missing>'
        )
        if not transformer_options.get(
            'latentlab_int8_prefetch_effective_announced_v7', False
        ):
            _lm_print(
                '[MiniMaxH3 LongMedia][INT8 PREFETCH V7 CHECK] '
                f'effective prefetch_dynamic_vbars={effective!r} '
                f'at DIFFUSION_MODEL boundary ({location})',
                flush=True,
            )
            transformer_options[
                'latentlab_int8_prefetch_effective_announced_v7'
            ] = True

    return executor(*call_args, **kwargs)



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
        # 0.3.0 quality-safe AUTO SOL policy.
        # Historical manual/default SOL started at 1.30, while the old AUTO
        # geometry policy used 1.70->2.10 plus as much as +0.30 for long
        # sequences.  That upper tail can become overly sparse and visibly
        # distort motion/temporal detail.  Pull the operating window down and
        # cap geometry pressure so AUTO never exceeds ~2.0.
        base_start = 1.30
        base_end = 1.85
        token_boost = 0.0
        if token_count > 150000:
            token_boost = min(
                0.15,
                max(
                    0.0,
                    (float(token_count) - 150000.0) / 60000.0 * 0.12,
                ),
            )

        tau_start = base_start + token_boost
        tau_end = base_end + token_boost
        curve = 'linear'
        state['sol_geometry_tau_boost'] = float(token_boost)

        if not state.get('sol_speed_tau_announced'):
            _lm_print(
                '[MiniMaxH3 LongMedia][AUTO GEO TAU] '
                f'base {base_start:.2f}->{base_end:.2f}, '
                f'tokens={token_count}, boost={token_boost:.3f} => '
                f'{tau_start:.2f}->{tau_end:.2f}',
                flush=True,
            )
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
        _int8_backend = (
            _v12_is_int8_family(state)
            and not comfy.model_management.in_training
        )
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
        """V27 emergency-only safety net for native Comfy INT8 residency.

        Routine soft_empty_cache() calls were catastrophic for TensorWiseINT8Layout:
        they discarded useful allocator/model residency every few blocks and forced
        repeated transfer/prepare work.  Comfy owns quantized-weight prefetch/residency;
        LongMedia trims only when *real driver-free* VRAM is critically low.
        """
        state = self.state
        backend = str(state.get('model_runtime_backend', 'unknown')).lower()
        if backend not in ('int8', 'int8-convrot-w4a4') or not torch.cuda.is_available():
            return
        snap = _cuda_memory_snapshot()
        if not snap:
            return
        mb = 1024.0 ** 2
        free_mb = float(snap['driver_free']) / mb
        cached_mb = float(snap['cached']) / mb
        emergency_mb = float(state.get('int8_residency_emergency_free_mb', 512) or 512)
        min_cached_mb = float(state.get('int8_residency_min_cached_mb', 512) or 512)
        # Preserve cache/resident quantized weights unless we are genuinely near OOM.
        if free_mb >= emergency_mb or cached_mb < min_cached_mb:
            return
        before_free, before_cached = free_mb, cached_mb
        try:
            gc.collect()
            comfy.model_management.soft_empty_cache()
        except Exception as exc:
            _lm_print('[MiniMaxH3 LongMedia][V27 INT8 RESIDENCY] emergency cleanup failed: '
                  f'block={self.index}, {type(exc).__name__}: {exc}', flush=True)
            return
        state['int8_residency_emergency_trim_count'] = int(
            state.get('int8_residency_emergency_trim_count', 0) or 0
        ) + 1
        after = _cuda_memory_snapshot()
        if after:
            _lm_print('[MiniMaxH3 LongMedia][V27 INT8 RESIDENCY] EMERGENCY TRIM: '
                  f'block={self.index}, driver_free {before_free:.0f}->{float(after["driver_free"])/mb:.0f} MB, '
                  f'cached {before_cached:.0f}->{float(after["cached"])/mb:.0f} MB', flush=True)

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
        chunk_tokens = min(max(256, runtime_chunk_tokens), max(1, token_count))
        if bool(state.get('auto_mlp_chunk_enabled')) and torch.cuda.is_available():
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

        # V31: native Comfy quant kernels, block-local prepared weights.
        # Avoid repeating cast/VBAR preparation for every MLP token chunk.
        _int8_backend = (
            _v12_is_int8_family(state)
            and not comfy.model_management.in_training
        )
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
                    _stock_fc1_probe = None
                    _stock_mlp_probe = None
                    if _v12b_linear_ab_enabled(state, 'fc1'):
                        _stock_fc1_probe = block.mlp.fc1(
                            _mlp_probe
                        ).detach()
                    if _v12b_linear_ab_enabled(state, 'mlp_fc1_fc2'):
                        _stock_mlp_probe = block.mlp(_mlp_probe).detach()

                    _fc1_handle = _int8_prepare_block_linear(
                        block.mlp.fc1, _mlp_probe
                    )
                    _fc2_handle = _int8_prepare_block_linear(
                        block.mlp.fc2, _mlp_probe
                    )

                    _cached_fc1_probe = None
                    if _stock_fc1_probe is not None or _stock_mlp_probe is not None:
                        _cached_fc1_probe = _int8_cached_linear(
                            _fc1_handle, _mlp_probe
                        )
                    if _stock_fc1_probe is not None:
                        _v12b_linear_ab_report(
                            state, 'fc1',
                            _stock_fc1_probe, _cached_fc1_probe,
                        )
                        del _stock_fc1_probe
                    if _stock_mlp_probe is not None:
                        _cached_mlp_probe = _int8_cached_linear(
                            _fc2_handle, _cached_fc1_probe, input_act='swiglu'
                        )
                        _v12b_linear_ab_report(
                            state, 'mlp_fc1_fc2',
                            _stock_mlp_probe, _cached_mlp_probe,
                        )
                        del _stock_mlp_probe, _cached_mlp_probe
                    if _cached_fc1_probe is not None:
                        del _cached_fc1_probe

                    if not state.get('int8_block_mlp_weights_announced'):
                        _lm_print(
                            '[MiniMaxH3 LongMedia][INT8 BLOCK WEIGHTS] '
                            'MLP fc1+fc2 cast once per H3 block and reused across all chunks',
                            flush=True,
                        )
                        state['int8_block_mlp_weights_announced'] = True

                if _fc1_handle is not None and _fc2_handle is not None:
                    _ff = _int8_cached_linear(_fc1_handle, h_chunk)
                    chunk_out = _int8_cached_linear(
                        _fc2_handle, _ff, input_act='swiglu'
                    )
                    del _ff
                else:
                    chunk_out = block.mlp(h_chunk)
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
            del attn_out

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

    def wrap(self, guider, chunk_tokens=8192, max_blocks=128, sol_mode='existing', sol_tau_start=1.3, sol_tau_end=0.8, sol_curve='linear', sol_min_tokens=4096, sol_dense_percent=0.0, sol_sink_conditioning='exact_kv', sol_qkv_chunk_tokens=8192, sol_out_proj_chunk_tokens=24576, vram_activation_reserve_mb=4096, inter_block_vram_guard_mb=2048, inter_block_guard_cooldown_blocks=4, inter_block_guard_emergency_mb=512, inter_block_guard_emergency_cooldown_blocks=3, late_block_guard_start=40, late_block_guard_target_mb=6144, late_block_guard_min_cached_mb=512, step_boundary_cleanup_mb=2048, sol_sigma_hi=1.0, sol_sigma_lo=0.0):
        if not globals().get('_LONGMEDIA_HOT_RELOAD_BYPASS', False):
            try:
                from .dev_hot_reload import dispatch_latest
                _did, _result = dispatch_latest(
                    __file__, __package__, self.__class__.__name__, 'wrap', self,
                    guider, chunk_tokens, max_blocks, sol_mode, sol_tau_start, sol_tau_end,
                    sol_curve, sol_min_tokens, sol_dense_percent, sol_sink_conditioning,
                    sol_qkv_chunk_tokens, sol_out_proj_chunk_tokens, vram_activation_reserve_mb,
                    inter_block_vram_guard_mb, inter_block_guard_cooldown_blocks,
                    inter_block_guard_emergency_mb, inter_block_guard_emergency_cooldown_blocks,
                    late_block_guard_start, late_block_guard_target_mb, late_block_guard_min_cached_mb,
                    step_boundary_cleanup_mb, sol_sigma_hi, sol_sigma_lo,
                )
                if _did:
                    return _result
            except Exception as _hot_exc:
                _lm_print(f'[MiniMaxH3 LongMedia][DEV HOT RELOAD] dispatcher fallback: {type(_hot_exc).__name__}: {_hot_exc}', flush=True)
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
        transformer_options['latentlab_disable_dynamic_vbar_prefetch'] = False
        _runtime_quant_variant = str(runtime_profile.get('quant_variant') or '').lower()
        if _runtime_backend in ('int8', 'int8-convrot-w4a4'):
            # W4A8 needs activation headroom more than aggressive weight prefetch;
            # ConvRot/TensorWise INT8 keeps the V27 residency experiment.
            transformer_options['prefetch_dynamic_vbars'] = (_runtime_quant_variant != 'w4a8')

        if _runtime_backend in ('int8', 'int8-convrot-w4a4'):
            _lm_print(
                '[MiniMaxH3 LongMedia][V29 QUANT RESIDENCY] '
                + (
                    'W4A8 detected: dynamic-VBAR prefetch DISABLED; AUTO MLP owns activation headroom'
                    if _runtime_quant_variant == 'w4a8'
                    else 'native Comfy dynamic-VBAR prefetch ENABLED; quantized-weight residency owned by Comfy'
                ),
                flush=True,
            )
            if _runtime_quant_variant == 'w4a8':
                _lm_print(
                    '[MiniMaxH3 LongMedia][V30 W4A8 THROUGHPUT] AUTO MLP ceiling=8192; '
                    'target is fewer native quantized dispatches per H3 block',
                    flush=True,
                )
        if str(sol_mode) in ('auto', 'sol', 'scheduled_sol') and WrappersMP is not None:
            wrappers = transformer_options.setdefault('wrappers', {})
            apply_model = wrappers.setdefault(WrappersMP.APPLY_MODEL, {})
            apply_model['MiniMaxH3LatentLabSolSpan'] = [_h3_sol_span_wrapper]

            diffusion_model = wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})
            diffusion_model['MiniMaxH3LatentLabRuntimePrefetch'] = [
                _h3_runtime_prefetch_wrapper
            ]
        patches_replace = transformer_options.setdefault('patches_replace', {})
        dit = patches_replace.setdefault('dit', {})
        state = {
            'mode': 'token_chunked_mlp',
            'model_runtime_profile': runtime_profile,
            'model_runtime_backend': str(runtime_profile.get('backend', 'unknown')),
            'model_runtime_quant_variant': runtime_profile.get('quant_variant'),
            'model_runtime_policy': dict(runtime_policy),
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
            'int8_semantic_dispatch_announced': False,
            # V27 native INT8: preserve residency/cache; only trim at a real emergency.
            'int8_residency_emergency_free_mb': 512,
            'int8_residency_min_cached_mb': 512,
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
        }
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
    """Create per-pass text without leaking hidden overlap into visible timing."""
    base_prompt = str(base_prompt or '').strip()
    if not base_prompt:
        return base_prompt

    # V62: if the author explicitly supplied one continuation section per pass,
    # trust those sections and keep their timestamps LOCAL to that pass.
    explicit = _v62_explicit_prompt_sections(base_prompt)
    if len(explicit) > 1 and int(segment_index) < len(explicit):
        selected = explicit[int(segment_index)]
        _lm_print(
            f'[MiniMaxH3 LongMedia][V62 SEGMENT PROMPT] pass={int(segment_index)} '
            f'uses explicit local-time section {int(segment_index)+1}/{len(explicit)}; '
            f'hidden overlap does not shift its 00 sec origin',
            flush=True,
        )
        return selected

    if segment_index <= 0:
        return base_prompt

    # segment_starts is the CONTEXT-window origin.  The user-visible pass starts
    # after its hidden overlap, so prompt events must be filtered against that
    # visible origin or every continuation fires overlap/FPS seconds too early.
    context_start = int(plan.segment_starts[segment_index])
    visible_start = context_start + int(getattr(plan, 'overlap_frames', 0) or 0)
    start_sec = float(visible_start) / float(FPS)
    visible_frames = max(1, int(plan.segment_lengths[segment_index]) - int(getattr(plan, 'overlap_frames', 0) or 0))
    end_sec = float(visible_start + visible_frames) / float(FPS)
    _lm_print(
        f'[MiniMaxH3 LongMedia][V62 TIMELINE] pass={int(segment_index)} '
        f'context_start={context_start}f, visible_start={visible_start}f, '
        f'hidden_preroll={int(getattr(plan, "overlap_frames", 0) or 0)}f; '
        f'prompt origin corrected by +{float(getattr(plan, "overlap_frames", 0) or 0)/float(FPS):.3f}s',
        flush=True,
    )
    restart_markers = (
        'start from the supplied first frame composition',
        'start from the supplied first-frame composition',
        'start from supplied first frame composition',
        'start from the first frame composition',
    )
    body_lines = []
    events = []
    for raw in base_prompt.splitlines():
        stripped = raw.strip()
        event = _V57_SEGMENT_EVENT_RE.match(stripped)
        if event:
            global_sec = float(event.group('sec'))
            if global_sec + 1e-6 >= start_sec and global_sec < end_sec - 1e-6:
                events.append((global_sec - start_sec, event.group('body').strip()))
            continue
        if any(m in stripped.lower() for m in restart_markers):
            continue
        body_lines.append(raw)

    parts = [
        'Continue directly from the preceding video segment. Do not restart the scene, '
        'camera, shot composition, character placement, or action. Preserve the exact '
        'identities, wardrobe, environment, lighting, camera trajectory, and motion continuity.',
        'Treat this pass as an immediate continuation of the same uninterrupted shot. '
        'No new introduction, no reset to the initial frame, and no alternate staging.',
    ]
    cleaned = '\n'.join(body_lines).strip()
    if cleaned:
        parts.append(cleaned)
    if events:
        lines = ['Segment-local timeline:']
        lines.extend(f'{_v57_format_local_time(sec)}: {body}' for sec, body in events)
        parts.append('\n'.join(lines))
    else:
        parts.append('Continue the current action naturally through this segment without a scene reset.')
    return '\n\n'.join(parts)


def _v57_attach_minimax_metadata(encoded_positive, source_positive, plan, segment_index):
    """Attach H3 payload by REFERENCE, never deepcopy media latents just to change text."""
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
        # Keep all native MiniMax payload fields. Values intentionally stay shared/read-only.
        for key, value in source_meta.items():
            if str(key).startswith('minimax_'):
                meta[key] = value

        # Hybrid frame-0 is a global opening anchor, never a reset anchor for pass > 0.
        if is_hybrid and segment_index > 0:
            keyframes = source_meta.get('minimax_keyframes') or []
            kept = []
            if is_final:
                kept = [
                    kf for kf in keyframes
                    if float(kf.get('resolved_frame_index', 0)) > 0.0
                ]
            if kept:
                meta['minimax_keyframes'] = kept
            else:
                meta.pop('minimax_keyframes', None)
                meta.pop('minimax_frame_count', None)
    return encoded_positive


def _v61_build_identity_sheet(ref_images, tile_height=512):
    """Pack multiple hybrid image refs into one continuation-only visual asset."""
    import torch.nn.functional as F

    images = [img for img in (ref_images or []) if img is not None]
    if len(images) < 2:
        return None
    tiles = []
    th = max(128, int(tile_height))
    for image in images:
        x = image[:1][..., :3]
        h, w = int(x.shape[1]), int(x.shape[2])
        tw = max(64, int(round(w * (th / float(max(1, h))))))
        y = F.interpolate(x.movedim(-1, 1), size=(th, tw), mode='bilinear', align_corners=False)
        tiles.append(y.movedim(1, -1))
    return torch.cat(tiles, dim=2).contiguous()


def _v61_prepare_identity_ref(vae, identity_sheet, width, height, resolution_mode):
    """Create one stock-compatible H3 image-ref block from the combined sheet."""
    if identity_sheet is None:
        return None, None
    from comfy_extras.nodes_minimax_h3 import _resize, REF_IMAGE_SHORT_EDGE

    h, w = int(identity_sheet.shape[1]), int(identity_sheet.shape[2])
    if resolution_mode == 'match':
        scale = min(1.0, math.sqrt((int(width) * int(height)) / float(max(1, w * h))))
    else:
        scale = min(1.0, float(REF_IMAGE_SHORT_EDGE) / float(max(1, min(w, h))))
    tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    resized = _resize(identity_sheet, tw, th, 'disabled')
    block = {
        'kind': 'image', 'latent_h': th // 16, 'latent_w': tw // 16,
        'latent': vae.encode(resized),
        'longmedia_identity_sheet': True,
    }
    return {'type': 'image', 'data': resized}, block


def _v61_identity_sheet_prompt(prompt):
    """Remap continuation Picture tags to the single combined identity sheet."""
    text = str(prompt or '')
    text = re.sub(r'<Picture\s+\d+>', '<Picture 1>', text, flags=re.IGNORECASE)
    prefix = (
        '<Picture 1> is a combined identity/reference sheet containing the recurring subjects. '
        'Use it only to preserve their distinct identities, faces, wardrobe and appearance; '
        'do not treat the separate subjects inside the sheet as separate shots or scene changes.'
    )
    return prefix + '\n\n' + text


def _v61_encode_continuation_identity_sheet(clip, prompt, positive, plan, segment_index,
                                              ref_item, ref_block):
    """Encode continuation text against one combined image reference, not N image refs."""
    import node_helpers

    prompt = _v61_identity_sheet_prompt(prompt)
    tokens = clip.tokenize(prompt, minimax_ref_items=[ref_item])
    scheduled = getattr(clip, 'encode_from_tokens_scheduled', None)
    encoded = scheduled(tokens) if callable(scheduled) else clip.encode(tokens)

    source_meta = _conditioning_meta(positive[0]) or {}
    values = {'minimax_refs': [ref_block]}
    for key in ('minimax_visual_cond_noise_aug', 'minimax_audio_cond_noise_aug'):
        if key in source_meta:
            values[key] = source_meta[key]

    is_final = int(segment_index) == int(plan.passes) - 1
    if is_final:
        last = [
            kf for kf in (source_meta.get('minimax_keyframes') or [])
            if float(kf.get('resolved_frame_index', 0)) > 0.0
        ]
        if last:
            values['minimax_keyframes'] = last
            values['minimax_frame_count'] = int(plan.segment_lengths[int(segment_index)])
    encoded = node_helpers.conditioning_set_values(encoded, values)
    return encoded, prompt


def _v57_preencode_segment_conditionings(clip, base_prompt, positive, plan, v61_identity_ref=None):
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
    prompts = [str(base_prompt or '')]
    for segment_index in range(1, int(getattr(plan, 'passes', 1))):
        segment_prompt = _v57_build_segment_prompt(base_prompt, plan, segment_index)
        if v61_identity_ref is not None:
            ref_item, ref_block = v61_identity_ref
            encoded, segment_prompt = _v61_encode_continuation_identity_sheet(
                clip, segment_prompt, positive, plan, segment_index, ref_item, ref_block
            )
        else:
            encoded = _encode_prompt(clip, segment_prompt)
            encoded = _v57_attach_minimax_metadata(encoded, positive, plan, segment_index)
        raw_result.append(encoded)
        prompts.append(segment_prompt)

    converted_result = tuple(comfy.sampler_helpers.convert_cond(cond) for cond in raw_result)
    if v61_identity_ref is not None and len(converted_result) > 1:
        _lm_print(
            f'[MiniMaxH3 LongMedia][V61 IDENTITY SHEET] continuation passes use ONE combined image ref; '
            f'Picture tags remapped to <Picture 1>; pass0 native refs unchanged',
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
    try:
        try:
            from . import motion_context_layout_patch
        except Exception:
            import importlib
            motion_context_layout_patch = importlib.import_module(__package__ + '.motion_context_layout_patch')
        if not motion_context_layout_patch.apply_patch():
            raise RuntimeError('PackedLayout motion-context patch could not activate')

        prev_video, _prev_audio = unpack_av_samples(previous_av)
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
                meta['minimax_keyframes'] = kept + keyframes
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
                meta['minimax_keyframes'] = kept + keyframes
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


def _clone_guider_with_segment_audio(guider, plan, segment_index, previous_av=None):
    """Clone guider cheaply, select pass conditioning, and add previous motion context."""
    shifted = copy.copy(guider)
    shifted.model_options = copy.deepcopy(getattr(guider, 'model_options', {}) or {})
    start_frame = int(plan.segment_starts[segment_index])

    # IMPORTANT V57: do not deepcopy the whole conditioning/media payload. Hybrid refs can
    # contain encoded image/video/audio latents. Only make a new dict shell and swap positive.
    shifted.original_conds = dict(getattr(guider, 'original_conds', {}) or {})
    segment_conds = getattr(plan, 'segment_positive_conditionings', None)
    if segment_conds and int(segment_index) < len(segment_conds):
        shifted.original_conds['positive'] = segment_conds[int(segment_index)]

    # V60: previous motion is target-timeline conditioning, not a Ref2VA clip.
    # Pin the same span as the frozen overlap onto the head of this target.
    if getattr(plan, 'mode', None) != 'storyboard_bridge' and int(segment_index) > 0 and previous_av is not None:
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
    for ref in refs:
        if ref.get('kind') == 'audio' and reference_audio is not None and plan.audio_vae is not None:
            length_frames = plan.segment_lengths[segment_index]
            available, _ = _slice_source_audio_for_segment(reference_audio, start_frame, length_frames)
            waveform_for_encode = available.movedim(1, -1)
            audio_lat = plan.audio_vae.encode(waveform_for_encode)
            ref['ref_audio_t'] = audio_lat.shape[-1]
            ref['audio_latent'] = audio_lat
        elif ref.get('kind') == 'video' and plan.source_video is not None and plan.video_vae is not None:
            length_frames = plan.segment_lengths[segment_index]
            source_frames = slice_video_segment(plan.source_video, start_frame, length_frames, plan.video_fps)
            ref['video_latent'] = plan.video_vae.encode(source_frames)
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
        result = stitch_continuation(
            previous_av, sampled_continuation_av, overlap_frames, NestedTensor,
            blend_video_overlap, bool(offload_to_cpu),
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
# V43: self-contained hybrid conditioning bridge inside Long Media Setup
# -----------------------------------------------------------------------------

def _activate_longmedia_hybrid_support():
    """Activate LongMedia's self-contained keyframe+Ref2VA payload merge.

    First/last keyframes use positions already supported by stock H3, so the
    standalone hybrid path does not require Contex Loop's PackedLayout patch.
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
            return
        img = _resize(image[:1], int(width), int(height), crop)
        resolved = int(frame_index) if native_guides else 0
        keyframes.append({
            'resolved_frame_index': resolved,
            mc_key: True,
            'latent': latent_override if latent_override is not None else vae.encode(img),
        })
        keyframe_images.append(img)

    add_keyframe(0, first_frame, 'disabled', first_latent_override)
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

    # Same semantics as upstream minimax-h3-hybrid-cond: refs use the H3
    # minimax_ref_items tokenizer path; keyframes live in payload guides.
    if ref_items:
        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    else:
        tokens = clip.tokenize(prompt, images=keyframe_images)
    scheduled = getattr(clip, 'encode_from_tokens_scheduled', None)
    cond = scheduled(tokens) if callable(scheduled) else clip.encode(tokens)

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
    }


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
                'audio_mode': (['auto', 'preserve', 'generate', 'reference_only', 'preserve_reference'], {'tooltip': 'auto: legacy behavior. preserve: restore original audio at output without using it as a reference when possible. generate: generate final H3 audio. reference_only: use input audio as H3 reference but output generated audio. preserve_reference: use input audio as H3 timing/rhythm/lip-sync reference, discard generated H3 audio, and restore the untouched original track at output.'}),
                'video_strength': (
                    'FLOAT',
                    {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'audio_strength': (
                    'FLOAT',
                    {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
                ),
                'generation_mode': (
                    ['auto', 'lip_sync'],
                    {
                        'default': 'auto',
                        'tooltip': (
                            'auto infers T2V / image-ref / video-ref / audio-ref from '
                            'what is connected. lip_sync pins image_1 as the identity '
                            'anchor and audio_1 as the driving track — explicit opt-in, '
                            'requires both image_1 and audio_1 connected.'
                        ),
                    },
                ),
                'first_frame_mode': (
                    ['latent_inject', 'pixel_override', 'blend'],
                    {
                        'default': 'latent_inject',
                        'tooltip': (
                            'lip_sync only. latent_inject: image_1 is written into the '
                            'video latent before sampling with first_frame_denoise, so '
                            'the model can smooth the transition into it (recommended). '
                            'pixel_override: old behavior — image_1 pixels hard-replace '
                            'frame 0 after decode, exact but can show a visible seam. '
                            'blend: cheap post-decode cross-fade of the first '
                            'first_frame_blend_frames toward image_1, no extra sampling.'
                        ),
                    },
                ),
                'first_frame_denoise': (
                    'FLOAT',
                    {
                        'default': 0.25, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                        'tooltip': (
                            'latent_inject only. 0 = pixel-identical to image_1 but may '
                            'still show a seam; higher lets the sampler soften the '
                            'transition at some cost to exact identity.'
                        ),
                    },
                ),
                'first_frame_blend_frames': (
                    'INT',
                    {
                        'default': 3, 'min': 1, 'max': 17, 'step': 1,
                        'tooltip': 'blend only. How many leading frames taper toward image_1.',
                    },
                ),
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
                'workflow_mode': (
                    ['hybrid_auto', 'ref2va_full', 'loop', 'manual', 'video_ref_edit'],
                    {
                        'default': 'hybrid_auto',
                        'tooltip': (
                            'hybrid_auto: image_1 is first frame; if image_2 is connected it is last frame; '
                            'remaining images are Picture refs. video_ref_edit: video_1 is the main motion/camera/composition '
                            'reference, image_1..9 are Picture refs for identity/style replacement, and audio_1 can carry the '
                            'paired source soundtrack. ref2va_full: all connected images are Picture refs with no first/last '
                            'anchors. loop: image_1 is reused as BOTH first and last frame for a seam-friendly viral loop; '
                            'image_2..9 are Picture refs. manual: exposes legacy conditioning and segmentation controls for '
                            'development/A-B tests.'
                        ),
                    },
                ),
                'image_1': ('IMAGE', {'lazy': True, 'tooltip': 'hybrid_auto: first frame. video_ref_edit/ref2va_full: regular <Picture 1> identity/style ref. loop: reused as both first and last frame. manual: role follows conditioning_mode.'}),
                'image_2': ('IMAGE', {'lazy': True, 'tooltip': 'hybrid_auto: last frame when connected. video_ref_edit/ref2va_full: regular <Picture 2> identity/style ref. loop: first Picture ref.'}),
                'image_3': ('IMAGE', {'lazy': True, 'tooltip': 'Picture reference in hybrid_auto/video_ref_edit/loop/ref2va_full; manual role follows conditioning_mode.'}),
                'image_4': ('IMAGE', {'lazy': True, 'tooltip': 'Image reference. Continues the <Picture N> sequence in hybrid modes.'}),
                'image_5': ('IMAGE', {'lazy': True, 'tooltip': 'Image reference. Continues the <Picture N> sequence in hybrid modes.'}),
                'image_6': ('IMAGE', {'lazy': True, 'tooltip': 'Image reference. Continues the <Picture N> sequence in hybrid modes.'}),
                'image_7': ('IMAGE', {'lazy': True, 'tooltip': 'Image reference. Continues the <Picture N> sequence in hybrid modes.'}),
                'image_8': ('IMAGE', {'lazy': True, 'tooltip': 'Image reference. Continues the <Picture N> sequence in hybrid modes.'}),
                'image_9': ('IMAGE', {'lazy': True, 'tooltip': 'Image reference. Continues the <Picture N> sequence in hybrid modes.'}),
                'video_1': ('IMAGE', {'lazy': True, 'tooltip': 'Video frames only (IMAGE batch). video_ref_edit: primary motion/camera/composition source as <Video 1>. Other modes: regular <Video 1> reference. If the source video has audio, connect that extracted audio to audio_1.'}),
                'video_2': ('IMAGE', {'lazy': True, 'tooltip': 'Video frames only (IMAGE batch). Passed as <Video 2> reference. Pair with audio_2 when they come from the same source.'}),
                'video_3': ('IMAGE', {'lazy': True, 'tooltip': 'Video frames only (IMAGE batch). Passed as <Video 3> reference. Pair with audio_3 when they come from the same source.'}),
                'audio_1': ('AUDIO', {'lazy': True, 'tooltip': 'Audio reference / source audio. Passed as <Audio 1>. In video_ref_edit, pair this with video_1 when it is the soundtrack extracted from the same source clip.'}),
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
              reference_budget, video_fps, video_mode, audio_mode, video_strength,
              audio_strength, workflow_mode='hybrid_auto', generation_mode='auto', conditioning_mode='auto_refs',
              first_frame_mode='latent_inject',
              first_frame_denoise=0.25, first_frame_blend_frames=3,
              image_1=None, image_2=None, image_3=None, image_4=None, image_5=None,
              image_6=None, image_7=None, image_8=None, image_9=None,
              video_1=None, video_2=None, video_3=None,
              audio_1=None, audio_2=None, audio_3=None):
        global NativeReferenceToVideo

        setup_memory_events = []
        # Start Setup from a clean model residency state.  This is especially
        # important when re-running a workflow after H3 occupied most of VRAM.
        setup_memory_events.append(_setup_memory_isolation('setup_entry', unload_models=True))

        effective_prompt = prompt
        audio_mode = str(audio_mode or 'auto')
        preserve_audio_output = audio_mode in ('preserve', 'preserve_reference')
        use_audio_as_reference = audio_mode != 'preserve'

        images = [v for v in [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9] if v is not None]
        videos = [v for v in [video_1, video_2, video_3] if v is not None]
        audios = [a for a in [audio_1, audio_2, audio_3] if a is not None]

        # 0.3.0 public facade: validated single-pass H3 modes by default.
        # Legacy segmentation/continuation machinery remains available only in Manual.
        workflow_mode = str(workflow_mode or 'hybrid_auto')
        loop_last_override = None
        if workflow_mode == 'hybrid_auto':
            if image_1 is not None and image_2 is not None:
                conditioning_mode = 'hybrid_first_last'
            elif image_1 is not None:
                conditioning_mode = 'hybrid_first_frame'
            else:
                conditioning_mode = 'auto_refs'
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

        effective_segment_seconds = float(segment_seconds) if workflow_mode == 'manual' else 600.0
        effective_overlap_frames = int(overlap_frames) if workflow_mode == 'manual' else 0

        plan = build_media_plan(
            audios=audios,
            videos=videos,
            manual_duration=float(manual_duration),
            duration_source=duration_source,
            segment_seconds=effective_segment_seconds,
            overlap_frames=effective_overlap_frames,
            video_fps=float(video_fps),
            resolution_mode=resolution_mode,
            video_strength=float(video_strength),
            audio_strength=float(audio_strength),
        )

        mode = plan.mode
        if workflow_mode != 'manual':
            _lm_print(f'[MiniMaxH3 LongMedia][0.3.0 MODE] workflow={workflow_mode} conditioning={conditioning_mode} passes={int(plan.passes)} segmentation=off', flush=True)
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
        if conditioning_mode != 'auto_refs' and generation_mode == 'lip_sync':
            raise ValueError(
                "conditioning_mode=hybrid_* cannot be combined with generation_mode='lip_sync'. "
                "Use generation_mode='auto' for hybrid H3 conditioning."
            )

        if generation_mode == 'lip_sync':
            if image_1 is None or audio_1 is None:
                raise ValueError(
                    "generation_mode='lip_sync' requires both image_1 and audio_1 "
                    'to be connected.'
                )
            mode = 'automatic_lip_sync'

        if conditioning_mode == 'storyboard_bridge':
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

            pass0_prompt = _v57_build_segment_prompt(effective_prompt, plan, 0)
            positive, target_av, sb0 = _build_longmedia_hybrid_conditioning(
                clip=clip, vae=vae, audio_vae=audio_vae, prompt=pass0_prompt,
                width=width, height=height, length=first_len, resolution_mode=resolution_mode,
                first_frame=panel_a, last_frame=panel_b, ref_images=storyboard_refs,
                ref_videos=storyboard_videos, ref_audios=storyboard_audios,
                first_latent_override=panel_a_lat, last_latent_override=panel_b_lat,
            )
            pass1_prompt = _v57_build_segment_prompt(effective_prompt, plan, 1)
            positive_1, target_av_1, sb1 = _build_longmedia_hybrid_conditioning(
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
            hybrid_ref_images = [v for v in hybrid_ref_images if v is not None]
            hybrid_ref_videos = [v for v in [video_1, video_2, video_3] if v is not None]
            hybrid_ref_audios = ([v for v in [audio_1, audio_2, audio_3] if v is not None] if use_audio_as_reference else [])
            setup_memory_events.append(
                _setup_memory_isolation('before_hybrid_conditioning', unload_models=True)
            )
            positive, target_av, hybrid_info = _build_longmedia_hybrid_conditioning(
                clip=clip, vae=vae, audio_vae=audio_vae,
                prompt=effective_prompt, width=width, height=height,
                length=plan.segment_lengths[0], resolution_mode=resolution_mode,
                first_frame=hybrid_first, last_frame=hybrid_last,
                ref_images=hybrid_ref_images, ref_videos=hybrid_ref_videos,
                ref_audios=hybrid_ref_audios,
                first_latent_override=loop_latent_override,
                last_latent_override=loop_latent_override,
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
                final_audio_track_count=(len(audios) if preserve_audio_output else 0), first_frame_override=None,
                audio_vae=audio_vae, video_vae=vae,
            )
        elif mode == 'automatic_lip_sync':
            effective_prompt = _build_lipsync_prompt(
                effective_prompt, plan, image_1 is not None, audio_1 is not None
            )
            ref_audio_waveform = audio_1['waveform'][:1]
            segment_samples = int(round(segment_seconds * audio_1['sample_rate']))
            if ref_audio_waveform.shape[-1] > segment_samples:
                ref_audio_waveform = ref_audio_waveform[..., :segment_samples]
            ref_audio = {'waveform': ref_audio_waveform, 'sample_rate': audio_1['sample_rate']}
            ref_images = {}
            ref_videos = {}
            ref_audios = {}
            for i, img in enumerate(images):
                ref_images[f'ref_image_{i}'] = img
            for i, vid in enumerate(videos):
                ref_videos[f'ref_video_{i}'] = vid
            if ref_audio is not None:
                ref_audios['ref_audio_0'] = ref_audio
            setup_memory_events.append(_setup_memory_isolation('before_native_reference', unload_models=True))
            positive, target_av = NativeReferenceToVideo.execute(
                clip=clip, vae=vae, audio_vae=audio_vae,
                prompt=effective_prompt, width=width, height=height,
                length=plan.segment_lengths[0], ref_image_size=resolution_mode,
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
            for i, img in enumerate(images):
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
                prompt=effective_prompt, ref_image_size=resolution_mode,
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
                0.0 if audio_mode in ('preserve', 'preserve_reference') else
                1.0 if audio_mode in ('generate', 'reference_only') else
                float(audio_strength)
            )
            audio_mask = torch.full(
                (1, 1, 1, audio_lat.shape[-1]), audio_denoise, dtype=torch.float32
            )
            mask_samples = NestedTensor((video_mask, audio_mask))
            target_av = {'samples': av_samples, 'noise_mask': mask_samples}
            setup_memory_events.append(_setup_memory_isolation('before_clip_encode', unload_models=True))
            positive = _encode_prompt(clip, effective_prompt)
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
                0.0 if audio_mode in ('preserve', 'preserve_reference') else
                1.0 if audio_mode in ('generate', 'reference_only') else
                float(audio_strength)
            )
            audio_mask = torch.full(
                (1, 1, 1, audio_lat.shape[-1]), audio_denoise, dtype=torch.float32
            )
            mask_samples = NestedTensor((video_mask, audio_mask))
            target_av = {'samples': av_samples, 'noise_mask': mask_samples}
            setup_memory_events.append(_setup_memory_isolation('before_clip_encode', unload_models=True))
            positive = _encode_prompt(clip, effective_prompt)
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
            for i, img in enumerate(images):
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
                prompt=effective_prompt, ref_image_size=resolution_mode,
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

        # Persist the requested audio output policy in the plan. Decode must not infer
        # preserve semantics from the shape/content of the sampled audio stream: Turbo
        # LoRAs may leave a stream that is invalid for the stock Audio VAE decoder.
        plan = _dc_replace(plan, audio_output_mode=audio_mode)

        # V57: build every per-pass TEXT conditioning now, while TE is intentionally available.
        # The plan receives only ready CONDITIONING tensors; never CLIP/TE/model-patcher objects.
        plan = _dc_replace(plan, video_vae=vae, audio_vae=audio_vae)
        v61_identity_ref = None
        if conditioning_mode in ('hybrid_first_frame', 'hybrid_first_last') and int(plan.passes) > 1:
            # V61 prototype only collapses independent IMAGE refs. Mixed video/audio refs keep
            # V60 semantics so we do not silently renumber unrelated media types.
            if not hybrid_ref_videos and not hybrid_ref_audios and len(hybrid_ref_images) >= 2:
                identity_sheet = _v61_build_identity_sheet(hybrid_ref_images)
                ref_item, ref_block = _v61_prepare_identity_ref(
                    vae, identity_sheet, width, height, resolution_mode
                )
                if ref_item is not None and ref_block is not None:
                    v61_identity_ref = (ref_item, ref_block)
                    _lm_print(
                        f'[MiniMaxH3 LongMedia][V61 IDENTITY SHEET] built continuation-only sheet from '
                        f'{len(hybrid_ref_images)} image refs at {int(ref_item["data"].shape[2])}x{int(ref_item["data"].shape[1])}; '
                        'original individual refs remain pass0-only',
                        flush=True,
                    )

        if conditioning_mode != 'storyboard_bridge':
            segment_positive_conditionings, segment_prompt_summaries = _v57_preencode_segment_conditionings(
                clip, effective_prompt, positive, plan, v61_identity_ref=v61_identity_ref,
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
            'segment_seconds_semantics': 'new_output_timeline_plus_extra_overlap_context',
            'segment_lengths_frames': list(plan.segment_lengths),
            'segment_starts_frames': list(plan.segment_starts),
            'overlap_frames': plan.overlap_frames,
            'segment_conditioning_policy': 'preencoded_in_setup_no_clip_in_plan',
            'segment_conditionings_preencoded': int(len(getattr(plan, 'segment_positive_conditionings', ()) or ())),
            'output_frames': int(plan.output_frames),
            'trim_frames': int(plan.trim_frames),
            'final_audio_tracks': plan.final_audio_track_count,
            'audio_mode': audio_mode,
            'audio_reference_enabled': bool(use_audio_as_reference or generation_mode == 'lip_sync'),
            'audio_output_bypass': bool(getattr(plan, 'final_audio_override', None) is not None),
            'workflow_mode': workflow_mode,
            'conditioning_mode': conditioning_mode,
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
        start_frame = plan.segment_starts[seg_idx]
        length_frames = plan.segment_lengths[seg_idx]
        overlap = plan.overlap_frames

        prev_video, prev_audio = unpack_av_samples(previous_av)
        target_video_t = video_latent_t(length_frames)
        target_audio_t = audio_latent_t(length_frames)
        overlap_video_t = video_latent_t(overlap) if overlap else 0
        overlap_audio_t = round(overlap / FPS * AUDIO_LATENT_FPS) if overlap else 0

        video = torch.zeros(
            (prev_video.shape[0], prev_video.shape[1], target_video_t,
             prev_video.shape[3], prev_video.shape[4]),
            dtype=prev_video.dtype, device=prev_video.device,
        )
        audio = torch.zeros(
            (prev_audio.shape[0], prev_audio.shape[1], prev_audio.shape[2], target_audio_t),
            dtype=prev_audio.dtype, device=prev_audio.device,
        )
        if overlap_video_t:
            video[:, :, :overlap_video_t] = prev_video[:, :, -overlap_video_t:]
        if overlap_audio_t:
            audio[..., :overlap_audio_t] = prev_audio[..., -overlap_audio_t:]

        video_overlap_policy = 'zero_fill'
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
            if overlap_audio_t:
                audio[..., overlap_audio_t:] = source_audio_latent[
                    ..., :audio.shape[-1] - overlap_audio_t
                ].to(audio)
            else:
                audio = source_audio_latent.to(audio)

        video_mask = torch.ones((1, 1, target_video_t, 1, 1), dtype=torch.float32)
        audio_mask = torch.ones((1, 1, 1, target_audio_t), dtype=torch.float32)
        if overlap_video_t:
            # Keep the inherited overlap under a constant denoise budget instead of
            # ramping to 1.0 across the overlap. This preserves continuity much
            # more faithfully across long-media passes.
            video_mask[:, :, :overlap_video_t] = float(video_context_denoise)
        if overlap_audio_t:
            audio_mask[..., :overlap_audio_t] = float(audio_context_denoise)

        av_samples = NestedTensor((video, audio))
        mask_samples = NestedTensor((video_mask, audio_mask))
        output = {k: v for k, v in previous_av.items() if k not in ('noise_mask', 'samples')}
        output['samples'] = av_samples
        output['noise_mask'] = mask_samples

        report = json.dumps({
            'segment_index': seg_idx,
            'start_frame': start_frame,
            'length_frames': length_frames,
            'overlap_frames': overlap,
            'video_overlap_policy': video_overlap_policy,
            'overlap_mask_policy': 'constant_overlap_denoise',
            'latent_value_transform': latent_value_transform,
            'video_context_denoise': float(video_context_denoise),
            'audio_context_denoise': float(audio_context_denoise),
        }, indent=2)
        return (output, report)


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
                'sampler_mode': (
                    ['auto', 'manual'],
                    {
                        'default': 'auto',
                        'tooltip': 'auto uses the validated 0.3.0 attention/VRAM policy. manual exposes all low-level tuning widgets.',
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
               step_boundary_cleanup_mb=2048, sampler_mode='auto'):
        if not globals().get('_LONGMEDIA_HOT_RELOAD_BYPASS', False):
            try:
                from .dev_hot_reload import dispatch_latest
                _did, _result = dispatch_latest(
                    __file__, __package__, self.__class__.__name__, 'sample', self,
                    initial_av, long_media_plan, guider, sampler, sigmas, seed,
                    video_context_denoise, audio_context_denoise, offload_completed_segments,
                    mlp_chunk_tokens, attention_mode, sol_tau_start, sol_tau_end, sol_curve,
                    sol_min_tokens, sol_dense_percent, sol_sink_conditioning, sol_qkv_chunk_tokens,
                    sol_out_proj_chunk_tokens, vram_activation_reserve_mb, inter_block_vram_guard_mb,
                    inter_block_guard_cooldown_blocks, inter_block_guard_emergency_mb,
                    inter_block_guard_emergency_cooldown_blocks, late_block_guard_start,
                    late_block_guard_target_mb, late_block_guard_min_cached_mb, step_boundary_cleanup_mb,
                    sampler_mode,
                )
                if _did:
                    return _result
            except Exception as _hot_exc:
                _lm_print(f'[MiniMaxH3 LongMedia][DEV HOT RELOAD] dispatcher fallback: {type(_hot_exc).__name__}: {_hot_exc}', flush=True)
        from comfy_execution.graph_utils import GraphBuilder

        plan = long_media_plan
        graph = GraphBuilder()
        sampler_mode = str(sampler_mode or 'auto')
        if sampler_mode == 'auto':
            # Frozen production policy from the validated V39/V40 branch.
            video_context_denoise = 0.0
            audio_context_denoise = 0.0
            offload_completed_segments = True
            mlp_chunk_tokens = 8192
            attention_mode = 'auto'
            sol_tau_start = 1.3
            sol_tau_end = 0.8
            sol_curve = 'linear'
            sol_min_tokens = 4096
            sol_dense_percent = 0.0
            sol_sink_conditioning = 'exact_kv'
            sol_qkv_chunk_tokens = 8192
            sol_out_proj_chunk_tokens = 24576
            vram_activation_reserve_mb = 4096
            inter_block_vram_guard_mb = 2048
            inter_block_guard_cooldown_blocks = 4
            inter_block_guard_emergency_mb = 512
            inter_block_guard_emergency_cooldown_blocks = 3
            late_block_guard_start = 40
            late_block_guard_target_mb = 6144
            late_block_guard_min_cached_mb = 512
            step_boundary_cleanup_mb = 2048
            _lm_print('[MiniMaxH3 LongMedia][0.3.0 SAMPLER] auto production policy active', flush=True)
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
        )
        traced_guider = mlp_chunker.out(0)
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
        first_noise = graph.node("RandomNoise", noise_seed=int(seed))
        _lm_print(
            f'[MiniMaxH3 LongMedia][V62 SAME SEED] pass=0 seed={int(seed) & 0xFFFFFFFFFFFFFFFF}',
            flush=True,
        )
        first_sample = graph.node(
            "SamplerCustomAdvanced",
            noise=first_noise.out(0),
            guider=traced_guider,
            sampler=profiled_sampler,
            sigmas=sigmas,
            latent_image=initial_av,
        )
        previous_segment = first_sample.out(0)
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
            noise = graph.node(
                "RandomNoise",
                noise_seed=int(seed) & 0xFFFFFFFFFFFFFFFF,
            )
            _lm_print(
                f'[MiniMaxH3 LongMedia][V62 SAME SEED] pass={segment_index} '
                f'seed={int(seed) & 0xFFFFFFFFFFFFFFFF} (same as pass 0)',
                flush=True,
            )
            segment_mlp_chunker = graph.node(
                "MiniMaxH3LatentLabMLPChunking",
                guider=_clone_guider_with_segment_audio(
                    guider, plan, segment_index, previous_av=(None if getattr(plan, 'mode', None) == 'storyboard_bridge' else previous_segment),
                ),
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
            )
            sampled = graph.node(
                "SamplerCustomAdvanced",
                noise=noise.out(0),
                guider=segment_mlp_chunker.out(0),
                sampler=profiled_sampler,
                sigmas=sigmas,
                latent_image=prepared_av,
            )
            joined = graph.node(
                "MiniMaxH3LatentLabStitchContinuation",
                previous_av=stitched,
                sampled_continuation_av=sampled.out(0),
                overlap_frames=(0 if getattr(plan, 'mode', None) == 'storyboard_bridge' else plan.overlap_frames),
                blend_video_overlap=False,
                offload_to_cpu=bool(offload_completed_segments),
            )
            previous_segment = sampled.out(0)
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
            "hybrid_keyframe_scope": (
                "first_only_pass0_last_only_final"
                if getattr(plan, 'mode', None) == 'hybrid' and plan.passes > 1
                else "unchanged"
            ),
            "continuation_driver": ("V64_storyboard_A_to_B_to_C_exact_shared_bridge" if getattr(plan, 'mode', None) == 'storyboard_bridge' else "V61_frozen_overlap_plus_true_motion_context_plus_single_identity_sheet"),
            "motion_context": {
                "enabled": bool(plan.passes > 1),
                "prototype_frames": 56,
                "source": "previous_generated_h3_video_latent_tail",
                "vae_roundtrip": False,
                "te_roundtrip": False,
            },
            "segment_prompting": "V61_preencoded_continuation_single_identity_sheet_no_TE_in_sampler",
            "conditioning_payload_copy": "shared_read_only_media_metadata_no_deepcopy",
            "stitched_single_output": True,
            "low_vram_mlp": {
                "mode": "token_chunk_exact",
                "enabled": bool(mlp_chunking_enabled),
                "chunk_tokens_requested": int(requested_mlp_chunk_tokens),
                "chunk_tokens_effective": int(effective_mlp_chunk_tokens),
                "attention_unchanged": str(attention_mode) in ("auto", "existing"),
            },
            "attention": {
                "mode": str(attention_mode),
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
        return {
            "result": (cleanup.out(0), plan.output_frames, plan.trim_frames, plan.passes, cleanup.out(1)),
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
        _release_model_memory_for_decode()
        video, audio = unpack_av_samples(final_av)
        video_vae = plan.video_vae
        audio_vae = plan.audio_vae
        images = video_vae.decode(video)
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
        report_data = {
            'model_memory_released_before_decode': True,
            'decode_uses_plan_vaes': True,
            'trimmed_video_frames': trimmed,
            'storyboard_duplicate_boundary_frame_removed': storyboard_duplicate_removed,
        }
        audio_output_mode = str(getattr(plan, 'audio_output_mode', 'auto') or 'auto')
        passthrough_audio_mode = audio_output_mode in ('auto', 'preserve', 'preserve_reference')
        preserve_audio_bypass = audio_output_mode in ('preserve', 'preserve_reference')

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

        if plan.mode == 'automatic_lip_sync':
            first_frame_mode = getattr(plan, 'first_frame_mode', 'pixel_override')
            if plan.first_frame_override is not None and first_frame_mode in ('pixel_override', 'blend'):
                first_frame = plan.first_frame_override
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
                report_data['first_frame_mode'] = first_frame_mode
            elif plan.first_frame_override is not None:
                # latent_inject already baked the reference into the sampled
                # latent before this decode ran — nothing left to do here.
                report_data['first_frame_mode'] = first_frame_mode
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
            report_data['first_frame_restored'] = plan.first_frame_override is not None
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
    'MiniMaxH3LatentLabLongMediaSetup': MiniMaxH3LatentLabLongMediaSetup,
    'MiniMaxH3LatentLabLongMediaNextSegment': MiniMaxH3LatentLabLongMediaNextSegment,
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
    'MiniMaxH3LatentLabLongMediaSetup': 'MiniMax H3 \u2022 Long Media Setup',
    'MiniMaxH3LatentLabLongMediaNextSegment': 'MiniMax H3 \u2022 Long Media Next Segment',
    'MiniMaxH3LatentLabLongMediaSampler': 'MiniMax H3 \u2022 Long Media Sampler',
    'MiniMaxH3LatentLabLongMediaDecode': 'MiniMax H3 \u2022 Long Media Decode',
    'MiniMaxH3LatentLabAttentionChunking': 'MiniMax H3 \u2022 Low-VRAM Attention Chunking (internal)',
    'MiniMaxH3LatentLabMLPChunking': 'MiniMax H3 \u2022 Low-VRAM MLP Chunking (internal)',
    'MiniMaxH3LatentLabFirstStepMemoryProfiler': 'MiniMax H3 \u2022 First-Step Memory Profiler (internal)',
    'MiniMaxH3LatentLabVRAMPressureGuard': 'MiniMax H3 \u2022 VRAM Pressure Guard (internal)',
    'MiniMaxH3LatentLabVRAMCacheCleanup': 'MiniMax H3 \u2022 VRAM Cache Cleanup (internal)',
}

replace = _dc_replace
