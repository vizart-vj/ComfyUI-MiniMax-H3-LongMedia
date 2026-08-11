"""ComfyUI nodes for direct MiniMax H3 audio/video latent control."""

from __future__ import annotations
import copy
import gc
import json
import math
import os
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


def _cuda_memory_snapshot():
    """Return allocator/driver memory counters in bytes for diagnostics."""
    if not torch.cuda.is_available():
        return None
    device = torch.cuda.current_device()
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    free_driver, total = torch.cuda.mem_get_info(device)
    return {
        'allocated': allocated,
        'reserved': reserved,
        'cached': max(0, reserved - allocated),
        'driver_free': int(free_driver),
        'total': int(total),
    }


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
        print(
            '[MiniMaxH3 LongMedia] Setup memory isolation: '
            f'{label}, allocated {before_alloc:.1f} -> {a["allocated_mb"]:.1f} MB, '
            f'driver free {before_free:.1f} -> {a["driver_free_mb"]:.1f} MB',
            flush=True,
        )
    if unload_error:
        print(
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
            print(
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
            print(
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
        state = self.state
        target_mb = int(state.get('inter_block_vram_guard_mb', 0) or 0)
        if target_mb <= 0 or not torch.cuda.is_available():
            return
        before = _cuda_memory_snapshot()
        if not before:
            return
        free_mb = before['driver_free'] / (1024.0 ** 2)
        cached_mb = before['cached'] / (1024.0 ** 2)
        # Avoid allocator churn unless the driver is genuinely pressured and
        # there is meaningful dead cache to return. Active tensors/weights are
        # untouched; this only releases allocator-reserved unused pages.
        if free_mb >= target_mb or cached_mb < 256.0:
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        _soft_empty_cuda_cache()
        after = _cuda_memory_snapshot()
        if not after:
            return
        state['inter_block_trim_count'] = int(state.get('inter_block_trim_count', 0)) + 1
        reclaimed = max(0, int(after['driver_free']) - int(before['driver_free']))
        state['inter_block_reclaimed_mb'] = round(
            float(state.get('inter_block_reclaimed_mb', 0.0)) + reclaimed / (1024.0 ** 2), 1
        )
        # Keep logs useful without printing 50 lines per forward.
        count = int(state['inter_block_trim_count'])
        if count <= 5 or count % 10 == 0:
            print(
                '[MiniMaxH3 LongMedia] Inter-block VRAM guard: '
                f'block {self.index}, free {free_mb:.1f} -> {after["driver_free"]/(1024.0**2):.1f} MB, '
                f'cached {cached_mb:.1f} -> {after["cached"]/(1024.0**2):.1f} MB',
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
        print(
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
        print('[MiniMaxH3 LongMedia] H3 block0 deep memory trace: first forward started', flush=True)

        if not torch.cuda.is_available():
            return original_block(args)

        block = self._extract_block(original_block)
        if block is None:
            state['fallback_reason'] = 'could not extract DiTBlock from original_block closure'
            print('[MiniMaxH3 LongMedia] H3 block0 deep trace fallback: DiTBlock closure not found', flush=True)
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
            print(
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
                print(
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


def _sol_schedule_tau(transformer_options, state):
    mode = state.get('sol_mode', 'existing')
    if mode != 'scheduled_sol':
        return float(state.get('sol_tau_start', 1.3))
    sigmas = (transformer_options or {}).get('sigmas')
    sigma = None
    if torch.is_tensor(sigmas) and sigmas.numel():
        try:
            sigma = float(sigmas.flatten()[0])
        except Exception:
            sigma = None
    hi = state.get('sol_sigma_hi')
    lo = state.get('sol_sigma_lo')
    if sigma is None or hi is None or lo is None or abs(float(hi)-float(lo)) < 1e-8:
        progress = 0.0
    else:
        progress = min(max((float(hi)-sigma) / max(float(hi)-float(lo), 1e-8), 0.0), 1.0)
    dense_percent = float(state.get('sol_dense_percent', 0.0) or 0.0)
    if dense_percent > 0.0 and progress < dense_percent:
        return None
    curve = state.get('sol_curve', 'linear')
    f = progress
    if curve == 'cosine':
        w = 0.5 - 0.5 * math.cos(math.pi * f)
    elif curve == 'sqrt':
        w = math.sqrt(f)
    elif curve == 'smoothstep':
        w = f * f * (3.0 - 2.0 * f)
    elif curve == 'exponential':
        w = math.expm1(3.0 * f) / math.expm1(3.0)
    elif curve == 'step':
        w = 1.0 if f >= 0.5 else 0.0
    else:
        w = f
    start = float(state.get('sol_tau_start', 1.3))
    end = float(state.get('sol_tau_end', 0.8))
    return start + (end - start) * w


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
            print(
                '[MiniMaxH3 LongMedia] Low-VRAM compressed streamed QKV enabled: '
                f'{s} tokens -> {chunks} query chunks of <= {chunk}; '
                'K/V=INT8+scale, Sol summaries=BF16, Q streamed',
                flush=True,
            )
            state['sol_qkv_announced'] = True

        H, D = int(attn.heads), int(attn.head_dim)
        storage = meas(
            'sol_stream_kv_storage_alloc',
            lambda: allocate_compressed_kv_sm120(1, s, H, D, x.device),
        )

        qw = kw = None
        if rope_freqs is not None:
            qw, kw = _weights()
            rot = int(rope_freqs.shape[-3] * 2)

        def _build_compressed_kv():
            for start in range(0, s, chunk):
                end = min(s, start + chunk)
                qkv_part = attn.qkv_proj(x[start:end])
                _q, _k, _v = qkv_part.split(inner, dim=-1)
                n = end - start
                _q = _q.view(1, n, H, D)
                _k = _k.view(1, n, H, D)
                _v = _v.view(1, n, H, D)
                if rope_freqs is not None:
                    rf = rope_freqs[:, start:end]
                    if comfy.model_management.in_training:
                        _q, _k = comfy.quant_ops.ck.rms_rope_split_half(
                            _q, _k, rf, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
                        )
                    else:
                        comfy.quant_ops.ck.rms_rope_split_half_(
                            _q, _k, rf, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
                        )
                else:
                    _k = attn.k_norm(_k)
                append_compressed_kv_sm120(storage, _k, _v, start)
                del qkv_part, _q, _k, _v
            return storage

        meas('sol_stream_kv_projection_compress', _build_compressed_kv)
        meas('sol_stream_kv_summaries', lambda: finalize_compressed_kv_sm120(storage))

        # Important geometry fix: attention inner width is 7168 on H3 while
        # hidden width is 5376.  Never write raw attention output into x.
        # Instead each streamed Q chunk immediately runs out_proj and only the
        # projected [tokens, hidden] result overwrites the dead norm1 slice.
        def _stream_queries_and_project():
            for start in range(0, s, chunk):
                end = min(s, start + chunk)
                qkv_part = attn.qkv_proj(x[start:end])
                _q, _k_dead, _v_dead = qkv_part.split(inner, dim=-1)
                n = end - start
                _q = _q.view(1, n, H, D)
                _k_dead = _k_dead.view(1, n, H, D)
                if rope_freqs is not None:
                    rf = rope_freqs[:, start:end]
                    if comfy.model_management.in_training:
                        _q, _k_dead = comfy.quant_ops.ck.rms_rope_split_half(
                            _q, _k_dead, rf, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
                        )
                    else:
                        comfy.quant_ops.ck.rms_rope_split_half_(
                            _q, _k_dead, rf, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
                        )
                else:
                    _q = attn.q_norm(_q)
                q_out = sol_attn_query_compressed_sm120(
                    _q, storage, q_offset=start, tau=float(tau),
                    sink_blocks=sink_blocks, sink_q=sink_q,
                )
                projected = attn.out_proj(q_out.view(n, inner))
                x[start:end].copy_(projected)
                del qkv_part, _q, _k_dead, _v_dead, q_out, projected
            return x

        result = meas('sol_stream_query_kernel_outproj', _stream_queries_and_project)
        del storage, qw, kw
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
            print(
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
        if _sol_exception_is_oom(exc):
            original_chunk = int(state.get('sol_qkv_chunk_tokens', 0) or 0)
            original_out_proj = int(state.get('sol_out_proj_chunk_tokens', 0) or 0)
            seen = state.setdefault('sol_oom_reasons', [])
            reason = f'{type(exc).__name__}: {exc}'
            if reason not in seen:
                seen.append(reason)
                print('[MiniMaxH3 LongMedia] Embedded Sol-Attn OOM: ' + reason, flush=True)
            retry_chunks = _sol_retry_chunk_schedule(original_chunk)
            last_exc = exc
            for retry_idx, retry_chunk in enumerate(retry_chunks, start=1):
                try:
                    state['sol_qkv_chunk_tokens'] = int(retry_chunk)
                    if original_out_proj > 0:
                        state['sol_out_proj_chunk_tokens'] = min(original_out_proj, max(retry_chunk * 3, 1024))
                    print(
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
                        print(
                            '[MiniMaxH3 LongMedia] Embedded Sol-Attn retry succeeded: '
                            f'using persistent qkv chunk {retry_chunk}',
                            flush=True,
                        )
                    break
                except Exception as retry_exc:
                    last_exc = retry_exc
                    if not _sol_exception_is_oom(retry_exc):
                        raise
                    print(
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
            reason = f'{type(exc).__name__}: {exc}'
            seen = state.setdefault('sol_fallbacks', [])
            if reason not in seen:
                seen.append(reason)
                print('[MiniMaxH3 LongMedia] Embedded Sol-Attn fallback: ' + reason, flush=True)

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
        print(
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
        state = self.state
        target_mb = int(state.get('inter_block_vram_guard_mb', 0) or 0)
        if target_mb <= 0 or not torch.cuda.is_available():
            return

        # Block indices restart on every denoising step, so cooldown uses a
        # monotonically increasing guard-call counter instead of self.index.
        guard_call = int(state.get('inter_block_guard_calls', 0)) + 1
        state['inter_block_guard_calls'] = guard_call

        before = _cuda_memory_snapshot()
        if not before:
            return
        free_mb = before['driver_free'] / (1024.0 ** 2)
        cached_mb = before['cached'] / (1024.0 ** 2)
        if free_mb >= target_mb or cached_mb < 256.0:
            return

        cooldown = max(0, int(state.get('inter_block_guard_cooldown_blocks', 0) or 0))
        emergency_mb = max(0, int(state.get('inter_block_guard_emergency_mb', 0) or 0))
        emergency_cooldown = max(0, int(state.get('inter_block_guard_emergency_cooldown_blocks', 0) or 0))
        emergency = emergency_mb > 0 and free_mb < emergency_mb
        last_trim_call = int(state.get('inter_block_last_trim_call', -1000000000))
        blocks_since_trim = guard_call - last_trim_call - 1

        if emergency:
            last_emergency_trim_call = int(state.get('inter_block_last_emergency_trim_call', -1000000000))
            blocks_since_emergency = guard_call - last_emergency_trim_call - 1
            if emergency_cooldown > 0 and blocks_since_emergency < emergency_cooldown:
                state['inter_block_emergency_cooldown_skip_count'] = int(
                    state.get('inter_block_emergency_cooldown_skip_count', 0)
                ) + 1
                skip_count = int(state['inter_block_emergency_cooldown_skip_count'])
                if skip_count <= 3 or skip_count % 25 == 0:
                    print(
                        f'[MiniMaxH3 LongMedia] Inter-block VRAM guard skipped by emergency cooldown: '
                        f'block {self.index}, free {free_mb:.1f} MB, cached {cached_mb:.1f} MB, '
                        f'emergency_cooldown={emergency_cooldown}, blocks_since_emergency={blocks_since_emergency}',
                        flush=True,
                    )
                return
        elif cooldown > 0 and blocks_since_trim < cooldown:
            state['inter_block_cooldown_skip_count'] = int(state.get('inter_block_cooldown_skip_count', 0)) + 1
            return

        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        _soft_empty_cuda_cache()
        after = _cuda_memory_snapshot()
        if not after:
            return

        state['inter_block_last_trim_call'] = guard_call
        state['inter_block_trim_count'] = int(state.get('inter_block_trim_count', 0)) + 1
        if emergency:
            state['inter_block_last_emergency_trim_call'] = guard_call
            state['inter_block_emergency_trim_count'] = int(state.get('inter_block_emergency_trim_count', 0)) + 1
        reclaimed = max(0, int(after['driver_free']) - int(before['driver_free']))
        state['inter_block_reclaimed_mb'] = round(
            float(state.get('inter_block_reclaimed_mb', 0.0)) + reclaimed / (1024.0 ** 2), 1
        )
        count = int(state['inter_block_trim_count'])
        if count <= 5 or count % 10 == 0 or emergency:
            kind = 'EMERGENCY' if emergency else 'adaptive'
            print(
                f'[MiniMaxH3 LongMedia] Inter-block VRAM guard ({kind}): '
                f'block {self.index}, free {free_mb:.1f} -> {after["driver_free"]/(1024.0**2):.1f} MB, '
                f'cached {cached_mb:.1f} -> {after["cached"]/(1024.0**2):.1f} MB, cooldown={cooldown}, emergency_cooldown={emergency_cooldown}',
                flush=True,
            )

    def _late_block_hard_guard(self, phase):
        """Targeted allocator trim for the tail of very long H3 forwards.

        The normal inter-block guard runs only after a block completes.  On very
        long sequences Dynamic VRAM/AIMDO can repopulate several GiB of allocator
        cache while entering the next late transformer block, leaving too little
        contiguous driver headroom for the next large attention/weight allocation.
        This guard runs only in the configured tail blocks and only at two safe
        points where dead temporaries may be returned without unloading active
        model weights: immediately before attention and before the streamed FFN.
        """
        state = self.state
        start_block = max(0, int(state.get('late_block_guard_start', 0) or 0))
        target_mb = max(0, int(state.get('late_block_guard_target_mb', 0) or 0))
        min_cached_mb = max(0, int(state.get('late_block_guard_min_cached_mb', 512) or 0))
        if target_mb <= 0 or self.index < start_block or not torch.cuda.is_available():
            return
        before = _cuda_memory_snapshot()
        if not before:
            return
        free_mb = before['driver_free'] / (1024.0 ** 2)
        cached_mb = before['cached'] / (1024.0 ** 2)
        if free_mb >= target_mb or cached_mb < min_cached_mb:
            return
        try:
            torch.cuda.synchronize(torch.cuda.current_device())
        except Exception:
            pass
        _soft_empty_cuda_cache()
        after = _cuda_memory_snapshot()
        if not after:
            return
        state['late_block_guard_trim_count'] = int(state.get('late_block_guard_trim_count', 0)) + 1
        reclaimed = max(0, int(after['driver_free']) - int(before['driver_free']))
        state['late_block_guard_reclaimed_mb'] = round(
            float(state.get('late_block_guard_reclaimed_mb', 0.0)) + reclaimed / (1024.0 ** 2), 1
        )
        print(
            '[MiniMaxH3 LongMedia] Late-block hard guard: '
            f'block {self.index} {phase}, free {free_mb:.1f} -> {after["driver_free"]/(1024.0**2):.1f} MB, '
            f'cached {cached_mb:.1f} -> {after["cached"]/(1024.0**2):.1f} MB, target={target_mb}',
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
            print(
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
        chunk_tokens = min(max(256, int(self.chunk_tokens)), max(1, token_count))
        chunks = (token_count + chunk_tokens - 1) // chunk_tokens
        state = self.state
        state['mlp_chunked_calls'] = int(state.get('mlp_chunked_calls', 0)) + 1
        state['mlp_fused_gate_residual_calls'] = int(state.get('mlp_fused_gate_residual_calls', 0)) + 1
        state['norm2_mlp_fused_calls'] = int(state.get('norm2_mlp_fused_calls', 0)) + 1
        state['max_sequence_tokens'] = max(int(state.get('max_sequence_tokens', 0)), token_count)
        state['max_chunks_per_mlp'] = max(int(state.get('max_chunks_per_mlp', 0)), chunks)
        state['mlp_fused_gate_residual'] = True
        state['norm2_mlp_fused_streaming'] = True
        if not state.get('announced'):
            print(
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

            chunk_out = block.mlp(h_chunk)
            del h_chunk

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
            del chunk_out

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
        print(
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
        block = self._extract_block(original_block)
        if block is None:
            if self.index == 0 and not state.get('fallback_reason'):
                state['fallback_reason'] = 'could not extract DiTBlock from original_block closure'
                print('[MiniMaxH3 LongMedia] Low-VRAM MLP fallback: DiTBlock closure not found', flush=True)
            return original_block(args)

        x = args['img']
        t_emb = args['t_emb']
        mod_segments = args['mod_segments']
        rope_freqs = args['rope_freqs']
        transformer_options = args['transformer_options']

        # Step-boundary profiler: block 0 is the first reliable point inside the
        # H3 transformer after sampler/model/AIMDO preparation for a denoise step.
        # The sampler callback stores the synchronized end-of-step timestamp in
        # this shared state; the next block-0 entry measures the transition gap.
        if self.index == 0 and torch.cuda.is_available():
            try:
                torch.cuda.synchronize(torch.cuda.current_device())
            except Exception:
                pass
            now = time.perf_counter()
            snap = _cuda_memory_snapshot()
            forward_no = int(state.get('step_boundary_forward_count', 0)) + 1
            state['step_boundary_forward_count'] = forward_no
            pending = state.pop('step_boundary_pending_callback', None)
            if isinstance(pending, dict):
                transition_ms = max(0.0, (now - float(pending.get('time_perf', now))) * 1000.0)
                entry = {
                    'from_step': pending.get('step'),
                    'to_forward': forward_no,
                    'transition_ms': round(transition_ms, 1),
                    'callback_allocated_mb': pending.get('allocated_mb'),
                    'callback_reserved_mb': pending.get('reserved_mb'),
                    'callback_driver_free_mb': pending.get('driver_free_mb'),
                    'block0_allocated_mb': _mb(snap['allocated']) if snap else None,
                    'block0_reserved_mb': _mb(snap['reserved']) if snap else None,
                    'block0_driver_free_mb': _mb(snap['driver_free']) if snap else None,
                }
                state.setdefault('step_boundary_transitions', []).append(entry)
                print(
                    '[MiniMaxH3 LongMedia] Step-boundary profile: '
                    f"step {entry['from_step']} -> forward {forward_no}, {entry['transition_ms']:.1f} ms; "
                    f"end alloc/res/free {entry['callback_allocated_mb']:.1f}/{entry['callback_reserved_mb']:.1f}/{entry['callback_driver_free_mb']:.1f} MB -> "
                    f"block0 {entry['block0_allocated_mb']:.1f}/{entry['block0_reserved_mb']:.1f}/{entry['block0_driver_free_mb']:.1f} MB",
                    flush=True,
                )
            state['step_boundary_current_forward'] = {
                'forward': forward_no,
                'block0_started_perf': now,
                'block0_allocated_mb': _mb(snap['allocated']) if snap else None,
                'block0_reserved_mb': _mb(snap['reserved']) if snap else None,
                'block0_driver_free_mb': _mb(snap['driver_free']) if snap else None,
            }

        trace_this = (
            self.index == 0
            and not state.get('first_forward_complete')
            and int(state.get('forward_count', 0)) == 0
            and torch.cuda.is_available()
        )
        if trace_this:
            state['forward_count'] = 1
            state['first_forward_started'] = True
            state['first_forward_started_at'] = time.time()
            print('[MiniMaxH3 LongMedia] H3 block0 deep ATTENTION trace + MLP chunking: first forward started', flush=True)
            device = torch.cuda.current_device()
            measure = lambda name, fn: self._measure(name, fn, state, device)
        else:
            measure = lambda name, fn: fn()

        try:
            vals = measure('adaln_proj', lambda: block.adaln_proj(t_emb))
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = vals

            h = measure(
                'norm1_mod',
                lambda: self._mod_scale_shift(block.norm1(x), shift_msa, scale_msa, mod_segments),
            )
            self._late_block_hard_guard('pre_attention')
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
            x = measure(
                'attention_gate_residual',
                lambda: self._mod_gate(x, gate_msa, attn_out, mod_segments),
            )
            del attn_out

            self._late_block_hard_guard('pre_ffn')
            x = measure(
                'norm2_mlp_chunked_fused_residual',
                lambda: self._chunk_norm2_mlp_gate_residual(
                    block, x, shift_mlp, scale_mlp, gate_mlp, mod_segments
                ),
            )

            if trace_this:
                state['blocks'].append({
                    'block': 0,
                    'peak_allocated_mb': state.get('highest_block_peak_allocated_mb', 0.0),
                    'peak_reserved_mb': state.get('highest_block_peak_reserved_mb', 0.0),
                    'deep_trace': True,
                    'mlp_chunked': True,
                })
                print(
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
            return {'img': x}
        except Exception as exc:
            message = str(exc).lower()
            is_oom = isinstance(exc, getattr(torch, 'OutOfMemoryError', RuntimeError)) or 'out of memory' in message
            if is_oom:
                state['oom'] = True
                state['oom_block'] = self.index
                state['oom_stage'] = state.get('stages', [])[-1]['stage'] if state.get('stages') else 'unknown'
                state['oom_message'] = str(exc)[:2000]
                print(
                    f"[MiniMaxH3 LongMedia] H3 CUDA OOM in block {self.index} near {state.get('oom_stage')}: {state['oom_message']}",
                    flush=True,
                )
            raise



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

        if getattr(final_layer, '_latentlab_final_output_streaming_installed', False):
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
                    h32 = h.to(torch.float32)
                    del h
                    projected = head(h32)
                    del h32
                    out[local:end].copy_(projected)
                    del projected
                if not st.get('final_output_streaming_announced'):
                    print(
                        '[MiniMaxH3 LongMedia] Final-output streaming enabled: '
                        f'{label} {n} tokens -> {chunks} chunks of <= {chunk}; FP32 hidden is chunk-local',
                        flush=True,
                    )
                    st['final_output_streaming_announced'] = True
                st['final_output_streaming_calls'] = int(st.get('final_output_streaming_calls', 0)) + 1
                st['final_output_max_tokens'] = max(int(st.get('final_output_max_tokens', 0)), n)
                st['final_output_max_chunks'] = max(int(st.get('final_output_max_chunks', 0)), chunks)
                return out

            trace = not st.get('final_output_first_profile_complete') and torch.cuda.is_available()
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
                print(
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
        print('[MiniMaxH3 LongMedia] Final-output streaming fallback: ' + state['final_output_streaming_error'], flush=True)
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
                'sol_mode': (['existing', 'sol', 'scheduled_sol'], {'default': 'existing'}),
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
        wrapped = copy.copy(guider)

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
                        print(
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
                print('[MiniMaxH3 LongMedia] Activation reserve fallback: ' + reserve_stats['error'], flush=True)
        try:
            wrapped.model_options = copy.deepcopy(getattr(guider, 'model_options', {}) or {})
        except Exception:
            wrapped.model_options = dict(getattr(guider, 'model_options', {}) or {})

        transformer_options = wrapped.model_options.setdefault('transformer_options', {})
        if str(sol_mode) != 'existing' and WrappersMP is not None:
            wrappers = transformer_options.setdefault('wrappers', {})
            apply_model = wrappers.setdefault(WrappersMP.APPLY_MODEL, {})
            apply_model['MiniMaxH3LatentLabSolSpan'] = [_h3_sol_span_wrapper]
        patches_replace = transformer_options.setdefault('patches_replace', {})
        dit = patches_replace.setdefault('dit', {})
        state = {
            'mode': 'token_chunked_mlp',
            'sol_mode': str(sol_mode),
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
            'chunk_tokens': int(chunk_tokens),
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
        }
        for i in range(int(max_blocks)):
            key = ('double_block', i)
            if key in dit:
                state['skipped_existing_patch_indices'].append(i)
                continue
            dit[key] = _H3MLPChunkPatch(i, state, chunk_tokens=int(chunk_tokens))
        if state['skipped_existing_patch_indices']:
            print(
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
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        torch.cuda.reset_peak_memory_stats(device)
        before = _cuda_memory_snapshot()
        state['before_sampling'] = {k + '_mb': _mb(v) for k, v in before.items()} if before else None
        enabled, history_error = _start_cuda_memory_history(state['max_history_entries'])
        state['history_enabled'] = bool(enabled)
        state['history_error'] = history_error
        first_callback_seen = False

        def profiled_callback(*args, **kwargs):
            nonlocal first_callback_seen
            # Synchronize exactly once at each denoise-step boundary so the
            # wall-clock transition measurement is not distorted by queued CUDA work.
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
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
                            print(
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
                    print(
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
                print(
                    '[MiniMaxH3 LongMedia] First-step memory profile: '
                    f"allocated {entry['allocated_mb']:.1f} MB, reserved {entry['reserved_mb']:.1f} MB, "
                    f"peak allocated {entry['peak_allocated_mb']:.1f} MB, peak reserved {entry['peak_reserved_mb']:.1f} MB, "
                    f"driver free {entry['driver_free_mb']:.1f} MB",
                    flush=True,
                )
                if isinstance(block_state, dict) and block_state.get('blocks'):
                    block_state['first_forward_complete'] = True
                    print(
                        '[MiniMaxH3 LongMedia] H3 block trace summary: '
                        f"{len(block_state['blocks'])} blocks, worst block {block_state.get('worst_block')}, "
                        f"peak allocated {block_state.get('highest_block_peak_allocated_mb', 0.0):.1f} MB, "
                        f"peak reserved {block_state.get('highest_block_peak_reserved_mb', 0.0):.1f} MB",
                        flush=True,
                    )
                if path:
                    print(f'[MiniMaxH3 LongMedia] First-step allocator snapshot: {path}', flush=True)
                elif error:
                    print(f'[MiniMaxH3 LongMedia] Snapshot unavailable: {error}', flush=True)
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
                print(
                    '[MiniMaxH3 LongMedia] CUDA OOM captured by first-step profiler. '
                    f"peak allocated {state['oom_peak_allocated_mb']:.1f} MB, "
                    f"peak reserved {state['oom_peak_reserved_mb']:.1f} MB",
                    flush=True,
                )
                if path:
                    print(f'[MiniMaxH3 LongMedia] OOM allocator snapshot: {path}', flush=True)
                elif error:
                    print(f'[MiniMaxH3 LongMedia] OOM snapshot unavailable: {error}', flush=True)
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
                print(
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

        print(
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


def _clone_guider_with_segment_audio(guider, plan, segment_index):
    """Clone a guider with per-pass references and a global temporal offset.

    Later long-media passes should look like a single continuous timeline to H3,
    not like fresh 0-based clips. We therefore keep reference media cropped to
    the current pass *and* attach a pass-global temporal offset that shifts the
    model's temporal RoPE/layout coordinates forward by the segment start.
    """
    shifted = copy.copy(guider)
    shifted.model_options = copy.deepcopy(getattr(guider, 'model_options', {}) or {})
    start_frame = int(plan.segment_starts[segment_index])
    shifted.original_conds = copy.deepcopy(guider.original_conds)

    transformer_options = shifted.model_options.setdefault('transformer_options', {})
    transformer_options[TEMPORAL_OFFSET_OPTION] = temporal_offset_for_frame(start_frame)
    if WrappersMP is not None:
        wrappers = transformer_options.setdefault('wrappers', {})
        apply_model = wrappers.setdefault(WrappersMP.APPLY_MODEL, {})
        apply_model['MiniMaxH3LatentLabTemporalOffset'] = [h3_temporal_offset_wrapper]

    reference_audio = getattr(plan, 'reference_audio', None) or plan.source_audio
    for ref in shifted.original_conds['positive'][0].get('minimax_refs', []):
        if ref['kind'] == 'audio' and reference_audio is not None and plan.audio_vae is not None:
            length_frames = plan.segment_lengths[segment_index]
            available, _ = _slice_source_audio_for_segment(
                reference_audio, start_frame, length_frames
            )
            waveform_for_encode = available.movedim(1, -1)
            audio_lat = plan.audio_vae.encode(waveform_for_encode)
            ref['ref_audio_t'] = audio_lat.shape[-1]
            ref['audio_latent'] = audio_lat
        elif ref['kind'] == 'video' and plan.source_video is not None and plan.video_vae is not None:
            length_frames = plan.segment_lengths[segment_index]
            source_frames = slice_video_segment(
                plan.source_video, start_frame, length_frames, plan.video_fps,
            )
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
                        'tooltip': 'Internal prompt. Ignored when prompt_input is connected.',
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
                    {'default': 8.0, 'min': 1.0, 'max': 60.0, 'step': 0.5},
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
                'audio_mode': (['auto', 'preserve', 'generate', 'reference_only'],),
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
            },
            'optional': {
                'prompt_input': ('STRING', {'forceInput': True}),
                'image_1': ('IMAGE', {'lazy': True}),
                'image_2': ('IMAGE', {'lazy': True}),
                'image_3': ('IMAGE', {'lazy': True}),
                'image_4': ('IMAGE', {'lazy': True}),
                'image_5': ('IMAGE', {'lazy': True}),
                'image_6': ('IMAGE', {'lazy': True}),
                'image_7': ('IMAGE', {'lazy': True}),
                'image_8': ('IMAGE', {'lazy': True}),
                'image_9': ('IMAGE', {'lazy': True}),
                'video_1': ('IMAGE', {'lazy': True, 'tooltip': 'Video frames only (IMAGE batch). If the source video has audio, connect its extracted audio separately to audio_1.'}),
                'video_2': ('IMAGE', {'lazy': True, 'tooltip': 'Video frames only (IMAGE batch). Connect matching extracted audio to audio_2 when needed.'}),
                'video_3': ('IMAGE', {'lazy': True, 'tooltip': 'Video frames only (IMAGE batch). Connect matching extracted audio to audio_3 when needed.'}),
                'audio_1': ('AUDIO', {'lazy': True, 'tooltip': 'Audio reference / source audio. For V2V with video_1, connect the audio extracted from that same video here.'}),
                'audio_2': ('AUDIO', {'lazy': True, 'tooltip': 'Optional second audio reference; pair with video_2 by convention when they come from the same source.'}),
                'audio_3': ('AUDIO', {'lazy': True, 'tooltip': 'Optional third audio reference; pair with video_3 by convention when they come from the same source.'}),
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
              audio_strength, generation_mode='auto', first_frame_mode='latent_inject',
              first_frame_denoise=0.25, first_frame_blend_frames=3, prompt_input=None,
              image_1=None, image_2=None, image_3=None, image_4=None, image_5=None,
              image_6=None, image_7=None, image_8=None, image_9=None,
              video_1=None, video_2=None, video_3=None,
              audio_1=None, audio_2=None, audio_3=None):
        global NativeReferenceToVideo

        setup_memory_events = []
        # Start Setup from a clean model residency state.  This is especially
        # important when re-running a workflow after H3 occupied most of VRAM.
        setup_memory_events.append(_setup_memory_isolation('setup_entry', unload_models=True))

        effective_prompt = prompt_input if prompt_input else prompt

        images = [v for v in [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9] if v is not None]
        videos = [v for v in [video_1, video_2, video_3] if v is not None]
        audios = [a for a in [audio_1, audio_2, audio_3] if a is not None]

        plan = build_media_plan(
            audios=audios,
            videos=videos,
            manual_duration=float(manual_duration),
            duration_source=duration_source,
            segment_seconds=float(segment_seconds),
            overlap_frames=int(overlap_frames),
            video_fps=float(video_fps),
            resolution_mode=resolution_mode,
            video_strength=float(video_strength),
            audio_strength=float(audio_strength),
        )

        mode = plan.mode
        target_av = None

        if NativeReferenceToVideo is None:
            NativeReferenceToVideo = _resolve_native_reference_to_video()

        if generation_mode == 'lip_sync':
            if image_1 is None or audio_1 is None:
                raise ValueError(
                    "generation_mode='lip_sync' requires both image_1 and audio_1 "
                    'to be connected.'
                )
            mode = 'automatic_lip_sync'

        if mode == 'automatic_lip_sync':
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
                final_audio_override=(audio_1 if audio_mode in ('auto', 'preserve') else None),
                final_audio_track_count=max(1, len(audios)),
                first_frame_override=image_1, audio_vae=audio_vae, video_vae=vae,
                first_frame_mode=first_frame_mode,
                first_frame_denoise=float(first_frame_denoise),
                first_frame_blend_frames=int(first_frame_blend_frames),
            )
            if len(audios) > 1 and audio_mode in ('auto', 'preserve'):
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
                audio_lat = torch.zeros((1, 32, 2, audio_t), dtype=frozen_audio_latent.dtype)
                audio_lat[..., :frozen_audio_latent.shape[-1]] = frozen_audio_latent
            video_lat = torch.zeros((1, 24, video_t, 1, 1), dtype=frozen_audio_latent.dtype)
            av_samples = NestedTensor((video_lat, audio_lat))
            video_mask = torch.ones((1, 1, video_t, 1, 1), dtype=torch.float32)
            audio_denoise = (
                0.0 if audio_mode == 'preserve' else
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
                source_audio=audio_1 if audio_mode in ('generate', 'reference_only') else None,
                final_audio_override=(mixed_audio if audio_mode in ('auto', 'preserve') else None),
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
                0.0 if audio_mode == 'preserve' else
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
                source_audio=audio_1 if audio_1 else None,
                source_video=video_1 if video_1 else None,
                final_audio_override=(mixed_audio if audio_mode in ('auto', 'preserve') else None),
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

        # Ensure plan always carries VAE references for decode
        plan = _dc_replace(plan, video_vae=vae, audio_vae=audio_vae)
        setup_memory_events.append(_setup_memory_isolation('setup_exit_release', unload_models=True))

        report = json.dumps({
            'mode': plan.mode,
            'passes': plan.passes,
            'segment_lengths': list(plan.segment_lengths),
            'overlap_frames': plan.overlap_frames,
            'final_audio_tracks': plan.final_audio_track_count,
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
    DESCRIPTION = 'Expand a long-media plan into a ComfyUI sub-graph of sampler nodes.'
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
                        'default': 8192, 'min': 0, 'max': 131072, 'step': 8192,
                        'tooltip': (
                            'Token chunk size for the low-VRAM H3 MLP path. '
                            '8192 is the current safe default. Larger values are faster '
                            'but use more VRAM. Set 0 to effectively disable MLP '
                            'chunking for A/B testing.'
                        ),
                    },
                ),
                'attention_mode': (
                    ['existing', 'sol', 'scheduled_sol'],
                    {'default': 'existing', 'tooltip': 'existing keeps your current Sage/Comfy attention. sol/scheduled_sol use the embedded Apache-2.0 SM120 Sol path.'},
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
                        'default': 8192, 'min': 0, 'max': 131072, 'step': 8192,
                        'tooltip': (
                            'Stream H3 QKV projection in token chunks. In streamed mode token-level '
                            'K/V are retained as INT8+scale while Sol block summaries stay BF16; Q is '
                            'reprojected and consumed chunk-by-chunk. This targets very long single-pass '
                            'clips on limited VRAM. 0 restores the full fused-QKV path.'
                        ),
                    },
                ),
                'sol_out_proj_chunk_tokens': (
                    'INT',
                    {
                        'default': 24576, 'min': 0, 'max': 131072, 'step': 8192,
                        'tooltip': (
                            'Token chunk size for the embedded Sol output projection. '
                            'Smaller values reduce peak VRAM; larger values are faster. '
                            '0 disables out_proj chunking.'
                        ),
                    },
                ),
                'vram_activation_reserve_mb': (
                    'INT',
                    {
                        'default': 4096, 'min': 0, 'max': 12288, 'step': 512,
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
                        'default': 2048, 'min': 0, 'max': 8192, 'step': 256,
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
                        'default': 512, 'min': 0, 'max': 4096, 'step': 256,
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
                        'default': 512, 'min': 0, 'max': 4096, 'step': 256,
                        'tooltip': 'Minimum reclaimable PyTorch CUDA cache required before a late-block hard trim is attempted.',
                    },
                ),
                'step_boundary_cleanup_mb': (
                    'INT',
                    {
                        'default': 2048, 'min': 0, 'max': 8192, 'step': 256,
                        'tooltip': 'Minimum driver-free VRAM target after each completed denoise step. Dead allocator cache is returned before the next H3 forward. 0 disables.',
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
               attention_mode='existing', sol_tau_start=1.3, sol_tau_end=0.8,
               sol_curve='linear', sol_min_tokens=4096, sol_dense_percent=0.0,
               sol_sink_conditioning='exact_kv', sol_qkv_chunk_tokens=8192, sol_out_proj_chunk_tokens=24576,
               vram_activation_reserve_mb=4096, inter_block_vram_guard_mb=2048,
               inter_block_guard_cooldown_blocks=4, inter_block_guard_emergency_mb=512, inter_block_guard_emergency_cooldown_blocks=3,
               late_block_guard_start=40, late_block_guard_target_mb=6144, late_block_guard_min_cached_mb=512,
               step_boundary_cleanup_mb=2048):
        from comfy_execution.graph_utils import GraphBuilder

        plan = long_media_plan
        graph = GraphBuilder()
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
            prepared = graph.node(
                "MiniMaxH3LatentLabLongMediaNextSegment",
                long_media_plan=plan,
                previous_av=previous_segment,
                segment_index=segment_index,
                video_context_denoise=float(video_context_denoise),
                audio_context_denoise=float(audio_context_denoise),
            )
            noise = graph.node(
                "RandomNoise",
                noise_seed=(int(seed) + segment_index) & 0xFFFFFFFFFFFFFFFF,
            )
            segment_mlp_chunker = graph.node(
                "MiniMaxH3LatentLabMLPChunking",
                guider=_clone_guider_with_segment_audio(guider, plan, segment_index),
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
                latent_image=prepared.out(0),
            )
            joined = graph.node(
                "MiniMaxH3LatentLabStitchContinuation",
                previous_av=stitched,
                sampled_continuation_av=sampled.out(0),
                overlap_frames=plan.overlap_frames,
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
            "low_vram_mlp": {
                "mode": "token_chunk_exact",
                "enabled": bool(mlp_chunking_enabled),
                "chunk_tokens_requested": int(requested_mlp_chunk_tokens),
                "chunk_tokens_effective": int(effective_mlp_chunk_tokens),
                "attention_unchanged": str(attention_mode) == "existing",
            },
            "attention": {
                "mode": str(attention_mode),
                "embedded_sol": str(attention_mode) != "existing",
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
        'Decode the final stitched H3 AV latent back to pixel frames and audio.'
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
        trimmed = 0
        if images.shape[0] > output_frames:
            trimmed = images.shape[0] - output_frames
            images = images[:output_frames]
        report_data = {
            'model_memory_released_before_decode': True,
            'trimmed_video_frames': trimmed,
        }
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
            if plan.final_audio_override is not None:
                audio = plan.final_audio_override
                report_data['original_audio_restored'] = True
                report_data['generated_audio_decoded'] = False
            elif audio_vae is not None:
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
            if plan.final_audio_override is not None:
                audio = plan.final_audio_override
                report_data['generated_audio_decoded'] = False
            elif audio_vae is not None and hasattr(audio, 'shape') and audio.ndim == 4:
                sr = int(getattr(audio_vae, 'audio_sample_rate', 32000))
                decoded_audio = audio_vae.decode(audio)
                audio = _normalize_decoded_audio(
                    decoded_audio, sr, round(plan.total_duration * sr)
                )
                report_data['generated_audio_decoded'] = True
            report_data['original_audio_restored'] = plan.final_audio_override is not None
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
