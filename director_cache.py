"""Persistent TAKE library and runtime clip cache for LongMedia Director.

User-visible revisions are TAKE-centric under ``output/longmedia_director`` and
own the Director snapshot, per-clip latent data and previews.  Runtime clip
pointers used by selective MultiClip regeneration are isolated under
``_runtime``; legacy clip-centric data is migrated best-effort for compatibility.

This module deliberately has no ComfyUI imports so its filesystem and tensor
round-trip behavior can be regression-tested in isolation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
import uuid
from typing import Any

import torch

try:
    from safetensors.torch import load_file as _safe_load_file
    from safetensors.torch import save_file as _safe_save_file
except Exception:  # pragma: no cover - ComfyUI ships safetensors
    _safe_load_file = None
    _safe_save_file = None


CACHE_VERSION = 2
BOUNDARY_HASH_VERSION = 2
SEAM_LINEAGE_VERSION = 1
H3_FPS = 24
H3_AUDIO_LATENT_FPS = 40


def _safe_name(value: Any, fallback: str = "item") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._-")
    return (text or fallback)[:120]


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.write.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _read_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _same_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a is b:
        return True
    try:
        return (
            a.device == b.device
            and a.dtype == b.dtype
            and tuple(a.shape) == tuple(b.shape)
            and a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
            and a.storage_offset() == b.storage_offset()
        )
    except Exception:
        return False


def _cpu_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise TypeError(f"Director cache expected torch.Tensor, got {type(tensor).__name__}")
    return tensor.detach().to(device="cpu").contiguous()


def _tensor_tail_hash(*tensors: torch.Tensor, tokens: int = 8) -> str:
    h = hashlib.sha256()
    for tensor in tensors:
        if not torch.is_tensor(tensor):
            continue
        t = tensor.detach().to(device="cpu").contiguous()
        if t.dim() >= 3 and int(t.shape[2]) > tokens:
            t = t[:, :, -tokens:].contiguous()
        h.update(str(tuple(t.shape)).encode("ascii"))
        h.update(str(t.dtype).encode("ascii"))
        try:
            payload = t.numpy().tobytes(order="C")
        except TypeError:
            payload = t.float().numpy().tobytes(order="C")
        h.update(payload)
    return h.hexdigest()


def _align_video_frames(frame_count: int) -> int:
    frame_count = max(5, int(frame_count))
    remainder = (frame_count - 5) % 17
    if remainder:
        frame_count += 17 - remainder
    return frame_count


def _video_latent_t(frame_count: int) -> int:
    frame_count = _align_video_frames(frame_count)
    return 2 + 5 * ((frame_count - 5) // 17)


def _boundary_token_counts(overlap_frames: int, video: torch.Tensor, audio: torch.Tensor) -> tuple[int, int]:
    overlap = max(0, int(overlap_frames or 0))
    if overlap <= 0:
        return 0, 0
    if video.dim() < 3 or audio.dim() < 1:
        raise ValueError(
            f"Director boundary tensors have invalid ranks: video={tuple(video.shape)}, audio={tuple(audio.shape)}"
        )
    video_tokens = min(_video_latent_t(overlap), int(video.shape[2]))
    audio_tokens = min(
        int(round(overlap / H3_FPS * H3_AUDIO_LATENT_FPS)),
        int(audio.shape[-1]),
    )
    return max(0, video_tokens), max(0, audio_tokens)


def _hash_tensor_region(hasher: "hashlib._Hash", label: str, tensor: torch.Tensor) -> None:
    t = tensor.detach().to(device="cpu").contiguous()
    hasher.update(label.encode("ascii"))
    hasher.update(str(tuple(t.shape)).encode("ascii"))
    hasher.update(str(t.dtype).encode("ascii"))
    try:
        payload = t.numpy().tobytes(order="C")
    except TypeError:
        payload = t.float().numpy().tobytes(order="C")
    hasher.update(payload)


def _av_boundary_hash(
    video: torch.Tensor,
    audio: torch.Tensor,
    overlap_frames: int,
    *,
    edge: str,
) -> str:
    """Hash exactly the H3 continuation seam, with stream-specific time axes.

    v1 used a generic fixed eight-token slice on dim=2 for every stream. That was
    incorrect for H3 audio (time is the last axis) and could include unlocked video
    middle tokens. A reroll could therefore look incompatible even when its actual
    locked seam was bit-identical.
    """
    edge = str(edge).strip().lower()
    if edge not in {"head", "tail"}:
        raise ValueError(f"Unsupported Director boundary edge: {edge!r}")
    ov_v, ov_a = _boundary_token_counts(overlap_frames, video, audio)
    h = hashlib.sha256()
    # Head and tail hashes intentionally share one namespace so the outgoing hash
    # of clip N can be compared directly with the incoming hash of clip N+1.
    h.update(f"director-boundary-v{BOUNDARY_HASH_VERSION}".encode("ascii"))
    h.update(f":overlap_frames={int(overlap_frames or 0)}".encode("ascii"))
    if ov_v:
        region_v = video[:, :, :ov_v] if edge == "head" else video[:, :, -ov_v:]
        _hash_tensor_region(h, "video", region_v)
    else:
        h.update(b"video:none")
    if ov_a:
        region_a = audio[..., :ov_a] if edge == "head" else audio[..., -ov_a:]
        _hash_tensor_region(h, "audio", region_a)
    else:
        h.update(b"audio:none")
    return h.hexdigest()


def _boundary_tensors_equal(
    left_video: torch.Tensor,
    left_audio: torch.Tensor,
    right_video: torch.Tensor,
    right_audio: torch.Tensor,
    overlap_frames: int,
) -> tuple[bool, dict[str, Any]]:
    """Compare the actual continuation seam: left tail against right head."""
    if tuple(left_video.shape[:2]) != tuple(right_video.shape[:2]) or tuple(left_video.shape[3:]) != tuple(right_video.shape[3:]):
        return False, {"reason": "video_geometry_mismatch"}
    if tuple(left_audio.shape[:-1]) != tuple(right_audio.shape[:-1]):
        return False, {"reason": "audio_geometry_mismatch"}
    ov_v_left, ov_a_left = _boundary_token_counts(overlap_frames, left_video, left_audio)
    ov_v_right, ov_a_right = _boundary_token_counts(overlap_frames, right_video, right_audio)
    ov_v = min(ov_v_left, ov_v_right)
    ov_a = min(ov_a_left, ov_a_right)
    if int(overlap_frames or 0) > 0 and ov_v <= 0:
        return False, {"reason": "missing_video_boundary_tokens"}
    video_ok = True if ov_v <= 0 else torch.equal(
        left_video[:, :, -ov_v:].cpu(), right_video[:, :, :ov_v].cpu()
    )
    audio_ok = True if ov_a <= 0 else torch.equal(
        left_audio[..., -ov_a:].cpu(), right_audio[..., :ov_a].cpu()
    )
    return bool(video_ok and audio_ok), {
        "reason": "exact_boundary_match" if video_ok and audio_ok else "exact_boundary_mismatch",
        "video_tokens": int(ov_v),
        "audio_tokens": int(ov_a),
        "video_ok": bool(video_ok),
        "audio_ok": bool(audio_ok),
    }


class DirectorClipCache:
    """Immutable take store plus active pointers for one Director project."""

    def __init__(self, output_root: str, project_id: Any):
        root = os.path.realpath(os.path.abspath(output_root))
        project = _safe_name(project_id, "director")
        self.output_root = root
        self.project_id = project
        # 0.6.40 take-centric storage. User-visible TAKE folders live directly
        # under longmedia_director/<collection>/<take-folder>. Runtime pointers are
        # hidden under _runtime and no longer own take data.
        self.library_root = os.path.realpath(os.path.join(root, "longmedia_director"))
        self.default_collection = "Unsorted"
        self.runtime_root = os.path.join(self.library_root, "_runtime", project)
        self.project_root = self.runtime_root
        self.legacy_project_root = os.path.realpath(os.path.join(self.library_root, project))
        if os.path.commonpath([root, self.library_root]) != root:
            raise ValueError("Director cache path escapes the ComfyUI output directory.")
        self.clips_root = os.path.join(self.runtime_root, "clips")
        self.requests_root = os.path.join(self.runtime_root, "requests")
        self.editor_active_root = os.path.join(self.runtime_root, "editor_active")
        self._take_dir_cache: dict[str, str] = {}
        os.makedirs(os.path.join(self.library_root, self.default_collection), exist_ok=True)
        self._migrate_legacy_layout_best_effort()

    def _clip_dir(self, clip_id: Any) -> str:
        return os.path.join(self.clips_root, _safe_name(clip_id, "clip"))

    def _legacy_clip_dir(self, clip_id: Any) -> str:
        return os.path.join(self.legacy_project_root, "clips", _safe_name(clip_id, "clip"))

    def _active_path(self, clip_id: Any) -> str:
        return os.path.join(self._clip_dir(clip_id), "active.json")

    def _legacy_active_path(self, clip_id: Any) -> str:
        return os.path.join(self._legacy_clip_dir(clip_id), "active.json")

    def _editor_active_path(self, clip_id: Any) -> str:
        return os.path.join(self.editor_active_root, f"{_safe_name(clip_id, 'clip')}.json")

    def _set_editor_active(self, clip_id: Any, revision: Any | None) -> None:
        path = self._editor_active_path(clip_id)
        revision_s = _safe_name(revision, "")
        if not revision_s:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            return
        _atomic_json(path, {
            "version": CACHE_VERSION, "project_id": self.project_id,
            "clip_id": str(clip_id), "revision": revision_s, "updated_at_unix": time.time(),
        })

    def editor_active_revision(self, clip_id: Any) -> str | None:
        pointer = _read_json(self._editor_active_path(clip_id))
        revision = str((pointer or {}).get("revision") or "").strip()
        if revision:
            return revision
        active = self.active_metadata(clip_id)
        return str((active or {}).get("revision") or "").strip() or None

    @staticmethod
    def _take_folder_basename(label: Any, revision: str) -> str:
        name = _safe_name(label, "Take")
        return f"{name}__{_safe_name(revision, 'take')[:8]}"

    def _collection_dir(self, collection: Any | None) -> str:
        name = _safe_name(collection, self.default_collection)
        if name == "_runtime":
            name = self.default_collection
        path = os.path.realpath(os.path.join(self.library_root, name))
        if os.path.commonpath([self.library_root, path]) != self.library_root:
            raise ValueError("Director TAKE collection escapes the library root.")
        return path

    def _workspace_manifest(self, take_dir: str) -> dict[str, Any] | None:
        return _read_json(os.path.join(take_dir, "take.json"))

    def _scan_take_workspaces(self) -> list[tuple[str, dict[str, Any]]]:
        out: list[tuple[str, dict[str, Any]]] = []
        if not os.path.isdir(self.library_root):
            return out
        for collection in os.listdir(self.library_root):
            if collection == "_runtime":
                continue
            cdir = os.path.join(self.library_root, collection)
            if not os.path.isdir(cdir):
                continue
            for child in os.listdir(cdir):
                tdir = os.path.join(cdir, child)
                if not os.path.isdir(tdir):
                    continue
                record = self._workspace_manifest(tdir)
                if not isinstance(record, dict):
                    continue
                if str(record.get("project_id") or "") != self.project_id:
                    continue
                revision = str(record.get("revision") or "").strip()
                if revision:
                    self._take_dir_cache[revision] = tdir
                out.append((tdir, record))
        return out

    def _take_workspace_dir(self, revision: Any, *, create: bool = False, label: Any | None = None, collection: Any | None = None) -> str | None:
        revision_s = _safe_name(revision, "")
        if not revision_s:
            return None
        cached = self._take_dir_cache.get(revision_s)
        if cached and os.path.isdir(cached):
            return cached
        for tdir, record in self._scan_take_workspaces():
            if str(record.get("revision") or "") == revision_s:
                self._take_dir_cache[revision_s] = tdir
                return tdir
        if not create:
            return None
        cdir = self._collection_dir(collection)
        os.makedirs(cdir, exist_ok=True)
        chosen_label = str(label or "").strip() or self._next_take_name(os.path.basename(cdir))
        base = self._take_folder_basename(chosen_label, revision_s)
        tdir = os.path.join(cdir, base)
        suffix = 2
        while os.path.exists(tdir):
            tdir = os.path.join(cdir, f"{base}_{suffix}")
            suffix += 1
        os.makedirs(os.path.join(tdir, "state"), exist_ok=True)
        os.makedirs(os.path.join(tdir, "clips"), exist_ok=True)
        os.makedirs(os.path.join(tdir, "previews"), exist_ok=True)
        self._take_dir_cache[revision_s] = tdir
        return tdir

    def _next_take_name(self, collection: Any | None = None) -> str:
        cdir = self._collection_dir(collection)
        count = 0
        if os.path.isdir(cdir):
            for child in os.listdir(cdir):
                if os.path.isfile(os.path.join(cdir, child, "take.json")):
                    count += 1
        return f"Take {count + 1}"

    def _take_paths(self, clip_id: Any, revision: str) -> tuple[str, str]:
        revision_s = _safe_name(revision, "revision")
        tdir = self._take_workspace_dir(revision_s, create=False)
        if tdir:
            clip_dir = os.path.join(tdir, "clips", _safe_name(clip_id, "clip"))
            return os.path.join(clip_dir, "latent.safetensors"), os.path.join(tdir, "take.json")
        # Legacy compatibility: old revisions remain readable until migrated.
        take_dir = os.path.join(self._legacy_clip_dir(clip_id), "takes")
        return os.path.join(take_dir, f"{revision_s}.safetensors"), os.path.join(take_dir, f"{revision_s}.json")

    def _new_take_paths(self, clip_id: Any, revision: str, *, label: Any | None = None, collection: Any | None = None) -> tuple[str, str]:
        tdir = self._take_workspace_dir(revision, create=True, label=label, collection=collection)
        assert tdir is not None
        clip_dir = os.path.join(tdir, "clips", _safe_name(clip_id, "clip"))
        os.makedirs(clip_dir, exist_ok=True)
        return os.path.join(clip_dir, "latent.safetensors"), os.path.join(tdir, "take.json")

    def _write_snapshot_sidecars(self, meta_path: str, record: dict[str, Any]) -> None:
        tdir = os.path.dirname(meta_path)
        state_dir = os.path.join(tdir, "state")
        os.makedirs(state_dir, exist_ok=True)
        snapshot = record.get("director_timeline_snapshot")
        if isinstance(snapshot, dict):
            _atomic_json(os.path.join(state_dir, "director.json"), snapshot)
        prompt = str(record.get("director_global_prompt_snapshot") or "")
        try:
            with open(os.path.join(state_dir, "global_prompt.txt"), "w", encoding="utf-8") as handle:
                handle.write(prompt)
        except OSError:
            pass

    def _migrate_legacy_layout_best_effort(self) -> None:
        legacy_clips = os.path.join(self.legacy_project_root, "clips")
        if not os.path.isdir(legacy_clips):
            return
        for clip_name in list(os.listdir(legacy_clips)):
            legacy_clip = os.path.join(legacy_clips, clip_name)
            if not os.path.isdir(legacy_clip):
                continue
            old_active = _read_json(os.path.join(legacy_clip, "active.json"))
            new_active_path = os.path.join(self.clips_root, clip_name, "active.json")
            if old_active and not os.path.isfile(new_active_path):
                try:
                    _atomic_json(new_active_path, old_active)
                except Exception:
                    pass
            if old_active and os.path.isfile(new_active_path):
                try:
                    os.unlink(os.path.join(legacy_clip, "active.json"))
                except OSError:
                    pass
            old_takes = os.path.join(legacy_clip, "takes")
            if not os.path.isdir(old_takes):
                continue
            for filename in list(os.listdir(old_takes)):
                if not filename.endswith(".json"):
                    continue
                old_meta = os.path.join(old_takes, filename)
                record = _read_json(old_meta)
                if not record or str(record.get("project_id") or self.project_id) != self.project_id:
                    continue
                revision = str(record.get("revision") or filename[:-5]).strip()
                if not revision or self._take_workspace_dir(revision, create=False):
                    continue
                label = str(record.get("label") or record.get("take_name") or "").strip() or self._next_take_name(self.default_collection)
                _tensor_old = os.path.join(old_takes, f"{revision}.safetensors")
                tensor_new, meta_new = self._new_take_paths(clip_name, revision, label=label, collection=self.default_collection)
                record["take_name"] = label
                record["label"] = label
                record["library_folder"] = self.default_collection
                record["storage_layout"] = "take_centric_v1"
                if os.path.isfile(_tensor_old):
                    try:
                        shutil.move(_tensor_old, tensor_new)
                    except Exception:
                        try:
                            shutil.copy2(_tensor_old, tensor_new)
                        except Exception:
                            pass
                take_dir = os.path.dirname(meta_new)
                preview_dir = os.path.join(take_dir, "previews")
                os.makedirs(preview_dir, exist_ok=True)
                old_preview_dir = os.path.join(legacy_clip, "previews")
                preview_map = {
                    "preview_file": (f"{revision}.png", "poster.png"),
                    "preview_sprite_file": (f"{revision}.sprite.jpg", "sprite.jpg"),
                    "preview_audio_file": (f"{revision}.monitor.wav", "audio.wav"),
                }
                for field, (old_name, new_name) in preview_map.items():
                    old_path = os.path.join(old_preview_dir, old_name)
                    new_path = os.path.join(preview_dir, new_name)
                    if os.path.isfile(old_path):
                        try:
                            shutil.move(old_path, new_path)
                        except Exception:
                            try:
                                shutil.copy2(old_path, new_path)
                            except Exception:
                                continue
                        record[field] = os.path.relpath(new_path, self.output_root)
                _atomic_json(meta_new, record)
                self._write_snapshot_sidecars(meta_new, record)
                try:
                    os.unlink(old_meta)
                except OSError:
                    pass
            for cleanup in (old_takes, os.path.join(legacy_clip, "previews"), legacy_clip):
                try:
                    os.rmdir(cleanup)
                except OSError:
                    pass
        legacy_requests = os.path.join(self.legacy_project_root, "requests")
        if os.path.isdir(legacy_requests):
            os.makedirs(self.requests_root, exist_ok=True)
            for name in os.listdir(legacy_requests):
                src = os.path.join(legacy_requests, name); dst = os.path.join(self.requests_root, name)
                if os.path.isfile(src) and not os.path.exists(dst):
                    try:
                        shutil.move(src, dst)
                    except OSError:
                        pass
            try:
                os.rmdir(legacy_requests)
            except OSError:
                pass
        for cleanup in (legacy_clips, self.legacy_project_root):
            try:
                os.rmdir(cleanup)
            except OSError:
                pass

    def _take_metadata(self, clip_id: Any, revision: Any) -> dict[str, Any] | None:
        revision = _safe_name(revision, "")
        if not revision:
            return None
        _tensor_path, meta_path = self._take_paths(clip_id, revision)
        metadata = _read_json(meta_path)
        return dict(metadata) if isinstance(metadata, dict) else None

    def _write_take_metadata(self, clip_id: Any, revision: Any, metadata: dict[str, Any]) -> None:
        revision = _safe_name(revision, "")
        if not revision:
            raise ValueError("Director take revision is required to update metadata.")
        _tensor_path, meta_path = self._take_paths(clip_id, revision)
        _atomic_json(meta_path, dict(metadata))

    @staticmethod
    def _new_seam_id() -> str:
        return uuid.uuid4().hex

    def _bind_seam_lineage(
        self,
        left_clip_id: Any,
        left_revision: Any,
        right_clip_id: Any,
        right_revision: Any,
        *,
        seam_id: str | None = None,
        reason: str | None = None,
    ) -> str:
        """Bind two take boundaries to one persistent editorial seam identity.

        A Director seam is a lineage/state contract, not a floating-point equality
        contract.  The active chain and explicit parent_revision edges are stronger
        evidence than byte hashes because clip-only rerolls intentionally preserve an
        approved boundary while replacing the clip middle.
        """
        left_meta = self._take_metadata(left_clip_id, left_revision)
        right_meta = self._take_metadata(right_clip_id, right_revision)
        if not left_meta or not right_meta:
            raise FileNotFoundError("Director seam lineage cannot bind missing take metadata.")

        chosen = str(seam_id or "").strip()
        if not chosen:
            left_existing = str(left_meta.get("outgoing_seam_id") or "").strip()
            right_existing = str(right_meta.get("incoming_seam_id") or "").strip()
            if left_existing and right_existing and left_existing == right_existing:
                chosen = left_existing
            elif left_existing:
                chosen = left_existing
            elif right_existing:
                chosen = right_existing
            else:
                chosen = self._new_seam_id()

        left_meta["seam_lineage_version"] = SEAM_LINEAGE_VERSION
        left_meta["outgoing_seam_id"] = chosen
        right_meta["seam_lineage_version"] = SEAM_LINEAGE_VERSION
        right_meta["incoming_seam_id"] = chosen
        if reason:
            left_meta["outgoing_seam_lineage_reason"] = str(reason)
            right_meta["incoming_seam_lineage_reason"] = str(reason)
        self._write_take_metadata(left_clip_id, left_revision, left_meta)
        self._write_take_metadata(right_clip_id, right_revision, right_meta)
        return chosen

    def ensure_active_chain_lineage(self, clip_order: list[Any] | tuple[Any, ...]) -> dict[str, Any]:
        """Repair missing lineage only when compatibility is actually provable.

        v0.5.56 treated the set of ``active.json`` pointers as authoritative and, when
        two existing seam ids disagreed, minted a brand-new id and rewrote both takes.
        That silently collapsed two different historical branches into one.  Status
        refresh is not allowed to invent provenance.  We now bind only when an exact
        parent edge or the stored boundary tensors/hashes prove the seam.
        """
        order = [str(v) for v in clip_order if str(v)]
        bound: list[dict[str, Any]] = []
        conflicts: list[dict[str, Any]] = []
        for index in range(max(0, len(order) - 1)):
            left_id = order[index]
            right_id = order[index + 1]
            left = self.active_metadata(left_id)
            right = self.active_metadata(right_id)
            if not isinstance(left, dict) or not isinstance(right, dict):
                continue
            left_revision = str(left.get("revision") or "").strip()
            right_revision = str(right.get("revision") or "").strip()
            if not left_revision or not right_revision:
                continue
            left_seam = str(left.get("outgoing_seam_id") or "").strip()
            right_seam = str(right.get("incoming_seam_id") or "").strip()
            if left_seam and right_seam and left_seam == right_seam:
                continue

            right_parent = str(right.get("parent_revision") or "").strip()
            proof = None
            if right_parent and right_parent == left_revision:
                proof = "parent_revision_lineage"
            else:
                # Missing lineage on legacy takes may still be repaired from the real
                # locked AV seam.  Existing contradictory lineage is never overwritten
                # merely because these revisions happen to be active at the same time.
                if left_seam and right_seam and left_seam != right_seam:
                    conflicts.append({
                        "left_clip_id": left_id,
                        "left_revision": left_revision,
                        "right_clip_id": right_id,
                        "right_revision": right_revision,
                        "reason": "existing_seam_lineage_conflict",
                    })
                    continue
                compatible, report = self.boundary_compatible(
                    left_id, left_revision, right_id, right_revision
                )
                if compatible:
                    proof = str(report.get("reason") or "boundary_compatible")
                else:
                    conflicts.append({
                        "left_clip_id": left_id,
                        "left_revision": left_revision,
                        "right_clip_id": right_id,
                        "right_revision": right_revision,
                        "reason": str(report.get("reason") or "boundary_incompatible"),
                    })
                    continue

            chosen = left_seam or right_seam or self._new_seam_id()
            chosen = self._bind_seam_lineage(
                left_id, left_revision, right_id, right_revision,
                seam_id=chosen,
                reason=f"active_chain_proven:{proof}",
            )
            bound.append({
                "left_clip_id": left_id,
                "left_revision": left_revision,
                "right_clip_id": right_id,
                "right_revision": right_revision,
                "seam_id": chosen,
                "proof": proof,
            })
        return {
            "version": SEAM_LINEAGE_VERSION,
            "active_seams_bound": len(bound),
            "bound": bound,
            "conflicts": conflicts,
        }

    def active_metadata(self, clip_id: Any) -> dict[str, Any] | None:
        pointer = _read_json(self._active_path(clip_id)) or _read_json(self._legacy_active_path(clip_id))
        if not pointer:
            return None
        revision = str(pointer.get("revision") or "").strip()
        if not revision:
            return None
        _tensor_path, meta_path = self._take_paths(clip_id, revision)
        metadata = _read_json(meta_path)
        if not metadata:
            return None
        metadata = dict(metadata)
        metadata["active"] = True
        return metadata

    def load_revision(self, clip_id: Any, revision: Any) -> dict[str, Any] | None:
        if _safe_load_file is None:
            raise RuntimeError("safetensors is unavailable; Director selective regeneration cannot load takes.")
        revision = _safe_name(revision, "")
        if not revision:
            return None
        tensor_path, meta_path = self._take_paths(clip_id, revision)
        metadata = _read_json(meta_path)
        if not metadata or not os.path.isfile(tensor_path):
            return None
        tensors = _safe_load_file(tensor_path, device="cpu")
        required = {"display_video", "display_audio"}
        if not required <= set(tensors):
            return None
        continuation_is_display = bool(metadata.get("continuation_is_display", False))
        if continuation_is_display:
            cont_video = tensors["display_video"]
            cont_audio = tensors["display_audio"]
        else:
            if "continuation_video" not in tensors or "continuation_audio" not in tensors:
                return None
            cont_video = tensors["continuation_video"]
            cont_audio = tensors["continuation_audio"]
        return {
            "metadata": dict(metadata),
            "display_video": tensors["display_video"],
            "display_audio": tensors["display_audio"],
            "continuation_video": cont_video,
            "continuation_audio": cont_audio,
        }

    def load_active(self, clip_id: Any) -> dict[str, Any] | None:
        metadata = self.active_metadata(clip_id)
        if not metadata:
            return None
        take = self.load_revision(clip_id, metadata.get("revision"))
        if take is not None:
            take["metadata"]["active"] = True
        return take

    def _upgrade_boundary_metadata(self, clip_id: Any, take: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(take.get("metadata") or {})
        revision = str(metadata.get("revision") or "").strip()
        if not revision:
            return metadata
        overlap_frames = int(metadata.get("overlap_frames") or 0)
        video = take.get("continuation_video")
        audio = take.get("continuation_audio")
        if not torch.is_tensor(video) or not torch.is_tensor(audio):
            return metadata
        metadata["boundary_hash_version"] = BOUNDARY_HASH_VERSION
        metadata["incoming_boundary_hash"] = _av_boundary_hash(
            video, audio, overlap_frames, edge="head"
        )
        metadata["outgoing_boundary_hash"] = _av_boundary_hash(
            video, audio, overlap_frames, edge="tail"
        )
        _tensor_path, meta_path = self._take_paths(clip_id, revision)
        _atomic_json(meta_path, metadata)
        take["metadata"] = dict(metadata)
        return metadata

    def seam_compatible(
        self,
        left_clip_id: Any,
        left_revision: Any,
        right_clip_id: Any,
        right_revision: Any,
    ) -> tuple[bool, dict[str, Any]]:
        """Validate one Director seam using editorial lineage first.

        ``parent_revision`` is an exact provenance edge: the right take was generated
        from the left take.  Persistent seam ids then carry that compatibility across
        clip-only rerolls that preserve both boundaries while changing only the middle.
        Tensor comparison remains a legacy fallback/diagnostic, never the primary
        source of truth for already-established Director branches.
        """
        left_meta = self._take_metadata(left_clip_id, left_revision)
        right_meta = self._take_metadata(right_clip_id, right_revision)
        if not left_meta or not right_meta:
            return False, {"reason": "missing_take_metadata"}

        left_revision_s = str(left_meta.get("revision") or left_revision or "").strip()
        right_parent = str(right_meta.get("parent_revision") or "").strip()
        left_seam = str(left_meta.get("outgoing_seam_id") or "").strip()
        right_seam = str(right_meta.get("incoming_seam_id") or "").strip()

        # Strongest legacy migration proof: this exact right take was produced from
        # this exact left take. Bind the seam id even if old numeric hashes disagree.
        if left_revision_s and right_parent == left_revision_s:
            seam_id = left_seam or right_seam or self._new_seam_id()
            seam_id = self._bind_seam_lineage(
                left_clip_id, left_revision, right_clip_id, right_revision,
                seam_id=seam_id,
                reason="parent_revision_lineage",
            )
            return True, {
                "reason": "parent_revision_lineage",
                "seam_lineage_version": SEAM_LINEAGE_VERSION,
                "seam_id": seam_id,
            }

        if left_seam and right_seam:
            return left_seam == right_seam, {
                "reason": "seam_lineage_match" if left_seam == right_seam else "seam_lineage_mismatch",
                "seam_lineage_version": SEAM_LINEAGE_VERSION,
                "left_seam_id": left_seam,
                "right_seam_id": right_seam,
            }

        # Legacy take with incomplete lineage: try the old exact boundary check once.
        # If it succeeds, promote that fact into a persistent seam id so future
        # restores survive restart without re-reading large safetensors.
        compatible, report = self.boundary_compatible(
            left_clip_id, left_revision, right_clip_id, right_revision
        )
        report = dict(report)
        report["lineage_fallback"] = True
        if compatible:
            seam_id = left_seam or right_seam or self._new_seam_id()
            seam_id = self._bind_seam_lineage(
                left_clip_id, left_revision, right_clip_id, right_revision,
                seam_id=seam_id,
                reason="legacy_tensor_revalidation",
            )
            report.update({
                "reason": "legacy_tensor_revalidation",
                "seam_lineage_version": SEAM_LINEAGE_VERSION,
                "seam_id": seam_id,
            })
        return compatible, report

    def plan_restore_branch(
        self,
        clip_order: list[Any] | tuple[Any, ...],
        clip_id: Any,
        revision: Any,
    ) -> dict[str, Any]:
        """Plan restoration of a take plus the cached downstream branch it belongs to.

        Restoring a historical middle take is not the same operation as splicing that
        take into whatever suffix happens to be active now.  If the current next clip
        belongs to another branch, walk the immutable take graph (``parent_revision``
        first, persistent seam lineage second) and recover a compatible cached suffix.
        No active pointer is changed by this planner.
        """
        order = [str(v) for v in clip_order if str(v)]
        clip_id_s = str(clip_id or "").strip()
        revision_s = _safe_name(revision, "")
        if not clip_id_s or not revision_s:
            return {"ok": False, "reason": "missing_clip_or_revision"}
        try:
            index = order.index(clip_id_s)
        except ValueError:
            return {"ok": False, "reason": "clip_not_in_timeline"}

        candidate = self._take_metadata(clip_id_s, revision_s)
        if not isinstance(candidate, dict):
            return {"ok": False, "reason": "take_revision_not_found"}

        compatibility: dict[str, Any] = {}
        if index > 0:
            prev_clip_id = order[index - 1]
            prev_meta = self.active_metadata(prev_clip_id)
            prev_revision = str((prev_meta or {}).get("revision") or "").strip()
            if not prev_revision:
                return {
                    "ok": False,
                    "reason": "active_previous_take_missing",
                    "failed_clip_id": prev_clip_id,
                }
            incoming_ok, incoming_report = self.seam_compatible(
                prev_clip_id, prev_revision, clip_id_s, revision_s
            )
            compatibility["incoming"] = incoming_report
            if not incoming_ok:
                return {
                    "ok": False,
                    "reason": "incoming_branch_mismatch",
                    "compatibility": compatibility,
                    "failed_clip_id": clip_id_s,
                }

        # DFS is deliberately over cached metadata, not tensors.  Exact tensor/hash
        # comparison is still available inside seam_compatible for old lineage only.
        # Takes are few in normal editorial use, and memoization prevents branch blowup.
        memo: dict[tuple[int, str], tuple[list[dict[str, Any]], list[dict[str, Any]]] | None] = {}

        def _candidate_revisions(right_clip_id: str, left_revision: str, left_clip_id: str) -> list[dict[str, Any]]:
            takes = [
                item for item in self.list_take_metadata(right_clip_id)
                if not bool(item.get("draft")) and str(item.get("tensor_file") or "")
            ]
            active = self.active_metadata(right_clip_id)
            active_revision = str((active or {}).get("revision") or "").strip()
            left_meta = self._take_metadata(left_clip_id, left_revision) or {}
            left_seam = str(left_meta.get("outgoing_seam_id") or "").strip()

            def _score(item: dict[str, Any]) -> tuple[int, int, int, float]:
                rev = str(item.get("revision") or "").strip()
                parent_match = int(str(item.get("parent_revision") or "").strip() == left_revision)
                seam_match = int(bool(left_seam) and str(item.get("incoming_seam_id") or "").strip() == left_seam)
                active_match = int(bool(active_revision) and rev == active_revision)
                return (active_match, parent_match, seam_match, float(item.get("created_at_unix") or 0.0))

            return sorted(takes, key=_score, reverse=True)

        def _walk(pos: int, left_clip_id: str, left_revision: str):
            if pos >= len(order):
                return [], []
            key = (pos, left_revision)
            if key in memo:
                return memo[key]
            right_clip_id = order[pos]
            attempts: list[dict[str, Any]] = []
            for right_meta in _candidate_revisions(right_clip_id, left_revision, left_clip_id):
                right_revision = str(right_meta.get("revision") or "").strip()
                if not right_revision:
                    continue
                ok, report = self.seam_compatible(
                    left_clip_id, left_revision, right_clip_id, right_revision
                )
                attempts.append({
                    "clip_id": right_clip_id,
                    "revision": right_revision,
                    "compatible": bool(ok),
                    "reason": str(report.get("reason") or "unknown"),
                })
                if not ok:
                    continue
                tail = _walk(pos + 1, right_clip_id, right_revision)
                if tail is None:
                    continue
                tail_path, tail_reports = tail
                result = (
                    [{"clip_id": right_clip_id, "revision": right_revision}] + tail_path,
                    [{
                        "left_clip_id": left_clip_id,
                        "left_revision": left_revision,
                        "right_clip_id": right_clip_id,
                        "right_revision": right_revision,
                        "report": report,
                    }] + tail_reports,
                )
                memo[key] = result
                return result
            memo[key] = None
            return None

        suffix = _walk(index + 1, clip_id_s, revision_s)
        if suffix is None:
            return {
                "ok": False,
                "reason": "no_compatible_cached_suffix",
                "compatibility": compatibility,
                "failed_clip_id": order[index + 1] if index + 1 < len(order) else None,
            }
        suffix_path, transition_reports = suffix
        compatibility["suffix"] = transition_reports
        return {
            "ok": True,
            "reason": "cached_branch_found",
            "clip_id": clip_id_s,
            "revision": revision_s,
            "path": [{"clip_id": clip_id_s, "revision": revision_s}] + suffix_path,
            "compatibility": compatibility,
        }

    def restore_take_branch(
        self,
        clip_order: list[Any] | tuple[Any, ...],
        clip_id: Any,
        revision: Any,
    ) -> dict[str, Any]:
        """Atomically-ish activate a historical take and its compatible suffix.

        Active pointers are tiny JSON files while take tensors are immutable.  Plan the
        complete path before touching pointers and roll back every changed pointer if a
        filesystem write fails, so a partial branch activation cannot leak into the UI.
        """
        plan = self.plan_restore_branch(clip_order, clip_id, revision)
        if not bool(plan.get("ok")):
            return plan

        path = list(plan.get("path") or [])
        previous: dict[str, str | None] = {}
        changed: list[dict[str, Any]] = []
        try:
            for item in path:
                current_clip_id = str(item.get("clip_id") or "")
                current_revision = str(item.get("revision") or "")
                active = self.active_metadata(current_clip_id)
                old_revision = str((active or {}).get("revision") or "").strip() or None
                previous[current_clip_id] = old_revision
                if old_revision == current_revision:
                    changed.append({
                        "clip_id": current_clip_id,
                        "revision": current_revision,
                        "changed": False,
                    })
                    continue
                self.activate_revision(current_clip_id, current_revision)
                changed.append({
                    "clip_id": current_clip_id,
                    "revision": current_revision,
                    "changed": True,
                    "previous_revision": old_revision,
                })
        except Exception:
            for item in reversed(changed):
                if not bool(item.get("changed")):
                    continue
                current_clip_id = str(item.get("clip_id") or "")
                old_revision = previous.get(current_clip_id)
                try:
                    if old_revision:
                        self.activate_revision(current_clip_id, old_revision)
                    else:
                        os.unlink(self._active_path(current_clip_id))
                except FileNotFoundError:
                    pass
                except Exception:
                    pass
            raise

        lineage = self.ensure_active_chain_lineage(clip_order)
        selected = self.active_metadata(clip_id)
        return {
            **plan,
            "ok": True,
            "reason": "branch_restored",
            "take": selected,
            "activated": changed,
            "changed_clips": [item["clip_id"] for item in changed if bool(item.get("changed"))],
            "lineage": lineage,
        }

    def boundary_compatible(
        self,
        left_clip_id: Any,
        left_revision: Any,
        right_clip_id: Any,
        right_revision: Any,
    ) -> tuple[bool, dict[str, Any]]:
        """Validate one active Director seam, repairing v1 metadata on demand.

        v2 metadata can answer from compact hashes. Legacy takes are loaded once and
        compared on the actual H3 continuation seam, then upgraded in place. This is
        what keeps take restoration valid across a ComfyUI restart and across the
        v0.5.54 boundary-hash bug.
        """
        left_meta_path = self._take_paths(left_clip_id, left_revision)[1]
        right_meta_path = self._take_paths(right_clip_id, right_revision)[1]
        left_meta = _read_json(left_meta_path)
        right_meta = _read_json(right_meta_path)
        if not left_meta or not right_meta:
            return False, {"reason": "missing_take_metadata"}

        left_overlap = int(left_meta.get("overlap_frames") or 0)
        right_overlap = int(right_meta.get("overlap_frames") or 0)
        if left_overlap != right_overlap:
            return False, {
                "reason": "overlap_contract_changed",
                "left_overlap_frames": left_overlap,
                "right_overlap_frames": right_overlap,
            }
        overlap_frames = right_overlap
        if overlap_frames <= 0:
            return True, {"reason": "no_overlap_contract", "overlap_frames": 0}

        left_v2 = int(left_meta.get("boundary_hash_version") or 0) >= BOUNDARY_HASH_VERSION
        right_v2 = int(right_meta.get("boundary_hash_version") or 0) >= BOUNDARY_HASH_VERSION
        left_hash = str(left_meta.get("outgoing_boundary_hash") or "")
        right_hash = str(right_meta.get("incoming_boundary_hash") or "")
        if left_v2 and right_v2 and left_hash and right_hash:
            return left_hash == right_hash, {
                "reason": "boundary_hash_match" if left_hash == right_hash else "boundary_hash_mismatch",
                "boundary_hash_version": BOUNDARY_HASH_VERSION,
                "overlap_frames": overlap_frames,
            }

        left_take = self.load_revision(left_clip_id, left_revision)
        right_take = self.load_revision(right_clip_id, right_revision)
        if left_take is None or right_take is None:
            return False, {"reason": "missing_take_tensor"}
        compatible, report = _boundary_tensors_equal(
            left_take["continuation_video"], left_take["continuation_audio"],
            right_take["continuation_video"], right_take["continuation_audio"],
            overlap_frames,
        )
        # Upgrade old metadata regardless of the result; future checks then use the
        # correct stream-aware seam hashes without reloading large safetensors.
        self._upgrade_boundary_metadata(left_clip_id, left_take)
        self._upgrade_boundary_metadata(right_clip_id, right_take)
        report = dict(report)
        report.update({
            "legacy_revalidated": True,
            "boundary_hash_version": BOUNDARY_HASH_VERSION,
            "overlap_frames": overlap_frames,
        })
        return compatible, report

    def save_take(
        self,
        *,
        clip_id: Any,
        clip_index: int,
        display_video: torch.Tensor,
        display_audio: torch.Tensor,
        continuation_video: torch.Tensor,
        continuation_audio: torch.Tensor,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if _safe_save_file is None:
            raise RuntimeError("safetensors is unavailable; Director selective regeneration cannot save takes.")
        metadata_in = dict(metadata or {})
        # CREATE TAKE owns the next render. If the editor-active TAKE is an empty
        # draft, render directly into that workspace instead of spawning a second
        # unrelated revision/card.
        editor_revision = str(self.editor_active_revision(clip_id) or "").strip()
        editor_meta = self._take_metadata(clip_id, editor_revision) if editor_revision else None
        fill_active_draft = bool(editor_meta and (editor_meta.get("draft") or str(editor_meta.get("state") or "") == "draft"))
        revision = editor_revision if fill_active_draft else uuid.uuid4().hex
        requested_label = str(metadata_in.get("label") or metadata_in.get("take_name") or (editor_meta or {}).get("take_name") or (editor_meta or {}).get("label") or "").strip() or None
        requested_folder = str(metadata_in.get("library_folder") or (editor_meta or {}).get("library_folder") or self.default_collection).strip() or self.default_collection
        resolved_label = requested_label or self._next_take_name(requested_folder)
        tensor_path, meta_path = self._new_take_paths(clip_id, revision, label=resolved_label, collection=requested_folder)
        os.makedirs(os.path.dirname(tensor_path), exist_ok=True)

        continuation_is_display = (
            _same_storage(display_video, continuation_video)
            and _same_storage(display_audio, continuation_audio)
        )
        tensors: dict[str, torch.Tensor] = {
            "display_video": _cpu_contiguous(display_video),
            "display_audio": _cpu_contiguous(display_audio),
        }
        if not continuation_is_display:
            tensors["continuation_video"] = _cpu_contiguous(continuation_video)
            tensors["continuation_audio"] = _cpu_contiguous(continuation_audio)

        temporary = f"{tensor_path}.write.{uuid.uuid4().hex}.tmp"
        try:
            _safe_save_file(tensors, temporary)
            os.replace(temporary, tensor_path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

        metadata_in.pop("draft", None)
        metadata_in.pop("state", None)
        preserve_seams_from_revision = str(
            metadata_in.pop("_preserve_seams_from_revision", "") or ""
        ).strip()
        has_next_clip = bool(metadata_in.pop("_has_next_clip", False))

        overlap_frames = int(metadata_in.get("overlap_frames") or 0)
        incoming_boundary_hash = _av_boundary_hash(
            continuation_video, continuation_audio, overlap_frames, edge="head"
        )
        outgoing_boundary_hash = _av_boundary_hash(
            continuation_video, continuation_audio, overlap_frames, edge="tail"
        )
        parent_boundary_hash = None
        parent_clip_id = str(metadata_in.get("parent_clip_id") or "").strip()
        if parent_clip_id:
            parent_meta = self.active_metadata(parent_clip_id)
            if isinstance(parent_meta, dict):
                parent_boundary_hash = str(parent_meta.get("outgoing_boundary_hash") or "") or None

        incoming_seam_id = None
        outgoing_seam_id = None
        seam_source_revision = None
        if preserve_seams_from_revision:
            source_meta = self._take_metadata(clip_id, preserve_seams_from_revision)
            if isinstance(source_meta, dict):
                incoming_seam_id = str(source_meta.get("incoming_seam_id") or "").strip() or None
                outgoing_seam_id = str(source_meta.get("outgoing_seam_id") or "").strip() or None
                seam_source_revision = preserve_seams_from_revision

        if incoming_seam_id is None and parent_clip_id:
            parent_meta = self.active_metadata(parent_clip_id)
            if isinstance(parent_meta, dict):
                incoming_seam_id = str(parent_meta.get("outgoing_seam_id") or "").strip() or None

        if outgoing_seam_id is None and has_next_clip:
            outgoing_seam_id = self._new_seam_id()

        record = {
            **metadata_in,
            # Cache-owned identity/compatibility fields are authoritative and cannot
            # be shadowed by caller metadata. This keeps the restore contract stable
            # across DTO/call-site changes.
            "version": CACHE_VERSION,
            "kind": "h3_longmedia_director_take",
            "project_id": self.project_id,
            "clip_id": str(clip_id),
            "clip_index": int(clip_index),
            "revision": revision,
            "created_at_unix": time.time(),
            "take_name": resolved_label,
            "label": resolved_label,
            "library_folder": _safe_name(requested_folder, self.default_collection),
            "storage_layout": "take_centric_v1",
            "tensor_file": os.path.relpath(tensor_path, self.output_root),
            "continuation_is_display": bool(continuation_is_display),
            "display_shapes": [list(display_video.shape), list(display_audio.shape)],
            "continuation_shapes": [list(continuation_video.shape), list(continuation_audio.shape)],
            "boundary_hash_version": BOUNDARY_HASH_VERSION,
            "incoming_boundary_hash": incoming_boundary_hash,
            "outgoing_boundary_hash": outgoing_boundary_hash,
            "parent_boundary_hash": parent_boundary_hash,
            "seam_lineage_version": SEAM_LINEAGE_VERSION,
            "incoming_seam_id": incoming_seam_id,
            "outgoing_seam_id": outgoing_seam_id,
            "seam_source_revision": seam_source_revision,
        }
        _atomic_json(meta_path, record)
        self._write_snapshot_sidecars(meta_path, record)
        _atomic_json(self._active_path(clip_id), {
            "version": CACHE_VERSION,
            "clip_id": str(clip_id),
            "revision": revision,
            "metadata_file": os.path.relpath(meta_path, self.output_root),
        })
        self._set_editor_active(clip_id, revision)
        return record


    def create_draft_take(
        self,
        *,
        clip_id: Any,
        clip_index: int,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an editorial TAKE slot without rendering or copying latent tensors.

        A draft is intentionally metadata-only. It can carry a complete Director
        timeline snapshot and later be restored as a settings preset, while not
        becoming the active rendered branch until an actual render exists.
        """
        revision = uuid.uuid4().hex
        metadata_in = dict(metadata or {})
        requested_label = str(metadata_in.get("label") or metadata_in.get("take_name") or "").strip() or None
        requested_folder = str(metadata_in.get("library_folder") or self.default_collection).strip() or self.default_collection
        resolved_label = requested_label or self._next_take_name(requested_folder)
        _tensor_path, meta_path = self._new_take_paths(clip_id, revision, label=resolved_label, collection=requested_folder)
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        for key in (
            "version", "kind", "project_id", "clip_id", "clip_index",
            "revision", "created_at_unix", "tensor_file", "active",
        ):
            metadata_in.pop(key, None)
        record = {
            **metadata_in,
            "version": CACHE_VERSION,
            "kind": "h3_longmedia_director_take_draft",
            "state": "draft",
            "draft": True,
            "project_id": self.project_id,
            "clip_id": str(clip_id),
            "clip_index": int(clip_index),
            "revision": revision,
            "created_at_unix": time.time(),
            "take_name": resolved_label,
            "label": resolved_label,
            "library_folder": _safe_name(requested_folder, self.default_collection),
            "storage_layout": "take_centric_v1",
            "tensor_file": None,
        }
        _atomic_json(meta_path, record)
        self._write_snapshot_sidecars(meta_path, record)
        self._set_editor_active(clip_id, revision)
        return record

    def duplicate_revision(self, clip_id: Any, revision: Any) -> dict[str, Any]:
        """Duplicate a TAKE. Drafts duplicate metadata only; rendered takes keep AV."""
        source_meta = self._take_metadata(clip_id, revision)
        if not isinstance(source_meta, dict):
            raise FileNotFoundError(f"Director take {revision!r} was not found.")
        source_revision = str(source_meta.get("revision") or revision or "").strip()
        if bool(source_meta.get("draft")) or str(source_meta.get("state") or "") == "draft":
            metadata = dict(source_meta)
            metadata["duplicated_from_revision"] = source_revision
            source_name = str(metadata.get("take_name") or metadata.get("label") or "Take").strip() or "Take"
            metadata["take_name"] = f"{source_name} Copy"
            metadata["label"] = metadata["take_name"]
            metadata.pop("preview_file", None)
            metadata.pop("preview_sprite_file", None)
            metadata.pop("preview_audio_file", None)
            return self.create_draft_take(
                clip_id=clip_id,
                clip_index=int(metadata.get("clip_index") or 0),
                metadata=metadata,
            )

        source = self.load_revision(clip_id, revision)
        if source is None:
            raise FileNotFoundError(f"Director take {revision!r} was not found.")
        metadata = dict(source.get("metadata") or {})
        metadata["duplicated_from_revision"] = source_revision
        source_name = str(metadata.get("take_name") or metadata.get("label") or "Take").strip() or "Take"
        metadata["take_name"] = f"{source_name} Copy"
        metadata["label"] = metadata["take_name"]
        metadata["_preserve_seams_from_revision"] = source_revision
        metadata["_has_next_clip"] = bool(metadata.get("outgoing_seam_id"))
        return self.save_take(
            clip_id=clip_id,
            clip_index=int(metadata.get("clip_index") or 0),
            display_video=source["display_video"],
            display_audio=source["display_audio"],
            continuation_video=source["continuation_video"],
            continuation_audio=source["continuation_audio"],
            metadata=metadata,
        )

    def delete_revision(self, clip_id: Any, revision: Any) -> dict[str, Any]:
        """Delete one TAKE entry, including metadata-only drafts."""
        revision_s = _safe_name(revision, "")
        if not revision_s:
            raise ValueError("Director take revision is required.")
        tensor_path, meta_path = self._take_paths(clip_id, revision_s)
        metadata = _read_json(meta_path)
        if not metadata:
            raise FileNotFoundError(f"Director take {revision_s!r} was not found.")
        is_draft = bool(metadata.get("draft")) or str(metadata.get("state") or "") == "draft"
        active = self.active_metadata(clip_id)
        was_active = (not is_draft) and str((active or {}).get("revision") or "") == revision_s
        workspace_dir = self._take_workspace_dir(revision_s, create=False)
        if workspace_dir and os.path.isfile(os.path.join(workspace_dir, "take.json")):
            try:
                shutil.rmtree(workspace_dir)
            except FileNotFoundError:
                pass
            self._take_dir_cache.pop(revision_s, None)
        else:
            paths = [meta_path]
            if not is_draft:
                paths.insert(0, tensor_path)
            for path in paths:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
        activated_revision = None
        if was_active:
            remaining = [
                item for item in self.list_take_metadata(clip_id)
                if not bool(item.get("draft")) and str(item.get("tensor_file") or "")
            ]
            if remaining:
                activated_revision = str(remaining[0].get("revision") or "") or None
                if activated_revision:
                    self.activate_revision(clip_id, activated_revision)
            else:
                try:
                    os.unlink(self._active_path(clip_id))
                except FileNotFoundError:
                    pass
        editor_was_active = self.editor_active_revision(clip_id) == revision_s
        if editor_was_active:
            remaining_editor = self.list_take_metadata(clip_id)
            fallback_editor = str((remaining_editor[0] if remaining_editor else {}).get("revision") or "").strip()
            self._set_editor_active(clip_id, fallback_editor or None)
        return {
            "ok": True,
            "clip_id": str(clip_id),
            "deleted_revision": revision_s,
            "deleted_draft": bool(is_draft),
            "was_active": bool(was_active),
            "editor_was_active": bool(editor_was_active),
            "active_revision": activated_revision,
            "editor_active_revision": self.editor_active_revision(clip_id),
        }

    def list_library_folders(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        os.makedirs(self._collection_dir(self.default_collection), exist_ok=True)
        for name in sorted(os.listdir(self.library_root), key=str.lower):
            if name == "_runtime":
                continue
            path = os.path.join(self.library_root, name)
            if not os.path.isdir(path):
                continue
            # Old project-id cache roots are implementation leftovers, not user folders.
            if name == self.project_id and os.path.isdir(os.path.join(path, "clips")):
                continue
            take_count = sum(1 for child in os.listdir(path) if os.path.isfile(os.path.join(path, child, "take.json")))
            out.append({"name": name, "take_count": take_count, "default": name == self.default_collection})
        return out

    def create_library_folder(self, name: Any) -> dict[str, Any]:
        clean = _safe_name(name, "")
        if not clean or clean == "_runtime":
            raise ValueError("Project folder name is empty or reserved.")
        path = self._collection_dir(clean)
        if os.path.exists(path):
            raise FileExistsError(f"Project folder {clean!r} already exists.")
        os.makedirs(path, exist_ok=False)
        return {"ok": True, "name": clean}

    def _rename_workspace_dir(self, revision: str, desired_label: str) -> tuple[str, bool]:
        tdir = self._take_workspace_dir(revision, create=False)
        if not tdir:
            raise FileNotFoundError(f"Director take {revision!r} was not found.")
        parent = os.path.dirname(tdir)
        base = self._take_folder_basename(desired_label, revision)
        target = os.path.join(parent, base)
        if os.path.realpath(target) == os.path.realpath(tdir):
            return tdir, False
        suffix = 2
        while os.path.exists(target):
            target = os.path.join(parent, f"{base}_{suffix}")
            suffix += 1
        # Windows may temporarily hold preview/safetensors handles. Retry gently; if
        # still busy, keep the metadata rename and mark the folder rename pending.
        last_exc = None
        for _ in range(4):
            try:
                os.replace(tdir, target)
                self._take_dir_cache[revision] = target
                return target, False
            except (PermissionError, OSError) as exc:
                last_exc = exc
                import gc
                gc.collect()
                time.sleep(0.08)
        return tdir, True

    def _refresh_workspace_record_paths(self, record: dict[str, Any], take_dir: str, clip_id: Any) -> dict[str, Any]:
        record = dict(record)
        latent = os.path.join(take_dir, "clips", _safe_name(clip_id, "clip"), "latent.safetensors")
        record["tensor_file"] = os.path.relpath(latent, self.output_root) if os.path.isfile(latent) else None
        preview_map = {
            "preview_file": os.path.join(take_dir, "previews", "poster.png"),
            "preview_sprite_file": os.path.join(take_dir, "previews", "sprite.jpg"),
            "preview_audio_file": os.path.join(take_dir, "previews", "audio.wav"),
        }
        for field, path in preview_map.items():
            if os.path.isfile(path):
                record[field] = os.path.relpath(path, self.output_root)
            else:
                record.pop(field, None)
        record["workspace_folder"] = os.path.basename(take_dir)
        record["library_folder"] = os.path.basename(os.path.dirname(take_dir))
        return record

    def rename_take(self, clip_id: Any, revision: Any, name: Any) -> dict[str, Any]:
        revision_s = _safe_name(revision, "")
        desired = str(name or "").strip()
        if not revision_s or not desired:
            raise ValueError("Take revision and non-empty name are required.")
        metadata = self._take_metadata(clip_id, revision_s)
        if not metadata:
            raise FileNotFoundError(f"Director take {revision_s!r} was not found.")
        metadata["take_name"] = desired[:120]
        metadata["label"] = desired[:120]
        tdir = self._take_workspace_dir(revision_s, create=False)
        pending = False
        if tdir:
            _atomic_json(os.path.join(tdir, "take.json"), metadata)
            new_dir, pending = self._rename_workspace_dir(revision_s, desired)
            metadata = self._refresh_workspace_record_paths(metadata, new_dir, clip_id)
            metadata["folder_rename_pending"] = bool(pending)
            _atomic_json(os.path.join(new_dir, "take.json"), metadata)
            self._write_snapshot_sidecars(os.path.join(new_dir, "take.json"), metadata)
        else:
            self._write_take_metadata(clip_id, revision_s, metadata)
        return {"ok": True, "take": metadata, "folder_rename_pending": bool(pending)}

    def move_take_to_folder(self, clip_id: Any, revision: Any, folder: Any) -> dict[str, Any]:
        revision_s = _safe_name(revision, "")
        if not revision_s:
            raise ValueError("Take revision is required.")
        tdir = self._take_workspace_dir(revision_s, create=False)
        if not tdir:
            raise FileNotFoundError(f"Director take {revision_s!r} was not found or is still legacy-only.")
        dest_parent = self._collection_dir(folder)
        if not os.path.isdir(dest_parent):
            raise FileNotFoundError(f"Project folder {_safe_name(folder, '')!r} does not exist.")
        target = os.path.join(dest_parent, os.path.basename(tdir))
        if os.path.realpath(target) != os.path.realpath(tdir):
            base = target
            n = 2
            while os.path.exists(target):
                target = f"{base}_{n}"
                n += 1
            os.replace(tdir, target)
            self._take_dir_cache[revision_s] = target
        metadata = self._workspace_manifest(target) or {}
        metadata = self._refresh_workspace_record_paths(metadata, target, clip_id)
        _atomic_json(os.path.join(target, "take.json"), metadata)
        self._write_snapshot_sidecars(os.path.join(target, "take.json"), metadata)
        return {"ok": True, "take": metadata, "library_folder": metadata["library_folder"]}

    def request_completed(self, request_id: Any) -> dict[str, Any] | None:
        request = _safe_name(request_id, "")
        if not request:
            return None
        return _read_json(os.path.join(self.requests_root, f"{request}.json"))

    def mark_request_completed(self, request_id: Any, payload: dict[str, Any]) -> None:
        request = _safe_name(request_id, "")
        if not request:
            return
        record = {
            "version": CACHE_VERSION,
            "request_id": str(request_id),
            "project_id": self.project_id,
            "completed_at_unix": time.time(),
            **dict(payload or {}),
        }
        _atomic_json(os.path.join(self.requests_root, f"{request}.json"), record)

    def activate_revision(self, clip_id: Any, revision: Any) -> dict[str, Any]:
        revision = _safe_name(revision, "")
        if not revision:
            raise ValueError("Director take revision is required.")
        _tensor_path, meta_path = self._take_paths(clip_id, revision)
        metadata = _read_json(meta_path)
        if not metadata:
            raise FileNotFoundError(f"Director take {revision!r} was not found.")
        tensor_path, _ = self._take_paths(clip_id, revision)
        if not os.path.isfile(tensor_path):
            raise FileNotFoundError(f"Director take tensor {revision!r} was not found.")
        _atomic_json(self._active_path(clip_id), {
            "version": CACHE_VERSION,
            "clip_id": str(clip_id),
            "revision": revision,
            "metadata_file": os.path.relpath(meta_path, self.output_root),
        })
        self._set_editor_active(clip_id, revision)
        result = dict(metadata)
        result["active"] = True
        return result

    @staticmethod
    def _preview_pil_frame(image: torch.Tensor, max_width: int | None = None):
        """Convert one ComfyUI IMAGE frame to a small RGB PIL image.

        Resize happens on the tensor's current device before the CPU transfer so a
        1080p Director preview never causes a full-resolution GPU->CPU copy per
        thumbnail.  The function accepts both HWC and CHW layouts.
        """
        from PIL import Image

        if not torch.is_tensor(image):
            raise TypeError(f"Director preview expected torch.Tensor, got {type(image).__name__}")
        frame = image.detach()
        if frame.dim() == 4 and int(frame.shape[0]) == 1:
            frame = frame[0]
        if frame.dim() != 3:
            raise ValueError(f"Director preview expected a 3D frame, got shape={tuple(frame.shape)}")
        if int(frame.shape[-1]) in (1, 3, 4):
            chw = frame.permute(2, 0, 1).unsqueeze(0)
        elif int(frame.shape[0]) in (1, 3, 4):
            chw = frame.unsqueeze(0)
        else:
            raise ValueError(f"Director preview cannot infer channels from shape={tuple(frame.shape)}")
        chw = chw.float().clamp(0.0, 1.0)
        height, width = int(chw.shape[-2]), int(chw.shape[-1])
        target_width = width
        if max_width is not None:
            target_width = max(1, min(width, int(max_width)))
        if target_width != width:
            import torch.nn.functional as F
            target_height = max(1, int(round(height * target_width / max(1, width))))
            chw = F.interpolate(chw, size=(target_height, target_width), mode="bilinear", align_corners=False)
        hwc = (chw[0].permute(1, 2, 0) * 255.0 + 0.5).to(torch.uint8).cpu().contiguous().numpy()
        if hwc.shape[-1] == 1:
            hwc = hwc[..., 0]
        return Image.fromarray(hwc).convert("RGB")

    def _update_preview_metadata(self, clip_id: Any, metadata: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
        revision = str(metadata.get("revision") or "").strip()
        if not revision:
            raise ValueError("Director active take is missing its revision.")
        _tensor_path, meta_path = self._take_paths(clip_id, revision)
        record = _read_json(meta_path) or dict(metadata)
        record.update(fields)
        record["preview_generated_at_unix"] = time.time()
        record.pop("preview_error", None)
        _atomic_json(meta_path, record)
        return record

    def attach_active_preview(self, clip_id: Any, image: torch.Tensor) -> str | None:
        metadata = self.active_metadata(clip_id)
        if not metadata or not torch.is_tensor(image):
            return None
        frame = self._preview_pil_frame(image, max_width=512)
        revision = str(metadata.get("revision") or "")
        take_dir = self._take_workspace_dir(revision, create=False)
        preview_dir = os.path.join(take_dir, "previews") if take_dir else os.path.join(self._legacy_clip_dir(clip_id), "previews")
        os.makedirs(preview_dir, exist_ok=True)
        preview_path = os.path.join(preview_dir, "poster.png") if take_dir else os.path.join(preview_dir, f"{revision}.png")
        frame.save(preview_path, format="PNG", optimize=True)
        record = self._update_preview_metadata(clip_id, metadata, {
            "preview_file": os.path.relpath(preview_path, self.output_root),
        })
        return str(record.get("preview_file") or "") or None

    def attach_active_preview_sprite(
        self,
        clip_id: Any,
        frames: torch.Tensor,
        *,
        source_fps: float,
        duration_seconds: float | None = None,
        sample_fps: float = 12.0,
        max_samples: int = 120,
        cell_width: int = 640,
        columns: int = 8,
        preview_fingerprint: str | None = None,
    ) -> dict[str, Any] | None:
        """Attach a scrub/play preview sprite to the active immutable take.

        Only a small set of downscaled frames is persisted.  This gives the web
        Director real timeline scrubbing without encoding another full video or
        keeping decoded RGB tensors alive after the Decode node returns.
        """
        metadata = self.active_metadata(clip_id)
        if not metadata or not torch.is_tensor(frames):
            return None
        if frames.dim() == 5 and int(frames.shape[0]) == 1:
            frames = frames[0]
        if frames.dim() != 4 or int(frames.shape[0]) <= 0:
            raise ValueError(f"Director preview sprite expected T-frame IMAGE tensor, got shape={tuple(frames.shape)}")

        frame_count = int(frames.shape[0])
        fps = max(float(source_fps), 1e-6)
        duration = float(duration_seconds) if duration_seconds is not None else frame_count / fps
        duration = max(duration, 1.0 / fps)
        requested = max(2, int(math.ceil(duration * max(float(sample_fps), 0.25))) + 1)
        sample_count = min(frame_count, max(2, min(int(max_samples), requested)))
        if sample_count >= frame_count:
            indices = list(range(frame_count))
        else:
            indices = [int(round(i * (frame_count - 1) / max(1, sample_count - 1))) for i in range(sample_count)]
            indices = list(dict.fromkeys(indices))

        thumbs = [self._preview_pil_frame(frames[index], max_width=cell_width) for index in indices]
        if not thumbs:
            return None
        cell_w, cell_h = thumbs[0].size
        cols = max(1, min(int(columns), len(thumbs)))
        rows = int(math.ceil(len(thumbs) / cols))
        from PIL import Image
        sprite = Image.new("RGB", (cell_w * cols, cell_h * rows), (0, 0, 0))
        for ordinal, thumb in enumerate(thumbs):
            if thumb.size != (cell_w, cell_h):
                thumb = thumb.resize((cell_w, cell_h), getattr(getattr(Image, "Resampling", Image), "BILINEAR"))
            sprite.paste(thumb, ((ordinal % cols) * cell_w, (ordinal // cols) * cell_h))

        revision = str(metadata.get("revision") or "")
        take_dir = self._take_workspace_dir(revision, create=False)
        preview_dir = os.path.join(take_dir, "previews") if take_dir else os.path.join(self._legacy_clip_dir(clip_id), "previews")
        os.makedirs(preview_dir, exist_ok=True)
        sprite_path = os.path.join(preview_dir, "sprite.jpg") if take_dir else os.path.join(preview_dir, f"{revision}.sprite.jpg")
        preview_path = os.path.join(preview_dir, "poster.png") if take_dir else os.path.join(preview_dir, f"{revision}.png")
        sprite.save(sprite_path, format="JPEG", quality=90, optimize=True)
        midpoint = min(range(len(indices)), key=lambda i: abs(indices[i] - (frame_count - 1) * 0.5))
        thumbs[midpoint].save(preview_path, format="PNG", optimize=True)

        fields = {
            "preview_file": os.path.relpath(preview_path, self.output_root),
            "preview_sprite_file": os.path.relpath(sprite_path, self.output_root),
            "preview_sprite_cols": cols,
            "preview_sprite_rows": rows,
            "preview_sprite_cell_width": cell_w,
            "preview_sprite_cell_height": cell_h,
            "preview_sprite_indices": indices,
            "preview_sprite_source_fps": fps,
            "preview_duration": duration,
            "preview_source_frame_count": frame_count,
            "preview_fingerprint": str(preview_fingerprint or "") or None,
        }
        record = self._update_preview_metadata(clip_id, metadata, fields)
        return {key: record.get(key) for key in fields}

    def attach_active_preview_audio(
        self,
        clip_id: Any,
        audio: dict[str, Any],
        *,
        preview_fingerprint: str | None = None,
    ) -> dict[str, Any] | None:
        """Persist a lightweight WAV monitor track for the active immutable take.

        This is generated from the already-final decoded/passthrough waveform. It
        never invokes AudioVAE and exists only so Director Play can monitor the
        rendered take; scrubbing remains silent in the browser.
        """
        metadata = self.active_metadata(clip_id)
        if not metadata or not isinstance(audio, dict):
            return None
        waveform = audio.get('waveform')
        if not torch.is_tensor(waveform) or waveform.numel() <= 0:
            return None
        sample_rate = max(1, int(audio.get('sample_rate') or 32000))
        revision = str(metadata.get('revision') or '')
        if not revision:
            return None
        take_dir = self._take_workspace_dir(revision, create=False)
        preview_dir = os.path.join(take_dir, 'previews') if take_dir else os.path.join(self._legacy_clip_dir(clip_id), 'previews')
        os.makedirs(preview_dir, exist_ok=True)
        audio_path = os.path.join(preview_dir, 'audio.wav') if take_dir else os.path.join(preview_dir, f'{revision}.monitor.wav')
        rel = os.path.relpath(audio_path, self.output_root)
        fingerprint = str(preview_fingerprint or '') or None
        if (
            os.path.isfile(audio_path)
            and str(metadata.get('preview_audio_fingerprint') or '') == str(fingerprint or '')
        ):
            return {
                'preview_audio_file': rel,
                'preview_audio_sample_rate': int(metadata.get('preview_audio_sample_rate') or sample_rate),
                'preview_audio_duration': float(metadata.get('preview_audio_duration') or 0.0),
                'preview_audio_reused': True,
            }

        wave_tensor = waveform.detach()
        if wave_tensor.ndim == 3:
            wave_tensor = wave_tensor[0]
        elif wave_tensor.ndim == 1:
            wave_tensor = wave_tensor.unsqueeze(0)
        if wave_tensor.ndim != 2:
            raise ValueError(f'Director monitor audio expected [B,C,T]/[C,T], got shape={tuple(waveform.shape)}')
        channels = int(wave_tensor.shape[0])
        if channels <= 0 or int(wave_tensor.shape[-1]) <= 0:
            return None
        # PCM16 keeps browser compatibility high and the monitor file compact.
        pcm = (wave_tensor.float().clamp(-1.0, 1.0).transpose(0, 1) * 32767.0).round().to(torch.int16).cpu().contiguous().numpy()
        import wave as wave_module
        with wave_module.open(audio_path, 'wb') as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes(order='C'))
        duration = float(int(wave_tensor.shape[-1]) / sample_rate)
        fields = {
            'preview_audio_file': rel,
            'preview_audio_sample_rate': sample_rate,
            'preview_audio_duration': duration,
            'preview_audio_fingerprint': fingerprint,
        }
        record = self._update_preview_metadata(clip_id, metadata, fields)
        return {key: record.get(key) for key in fields}

    def list_take_metadata(self, clip_id: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        editor_active = self.editor_active_revision(clip_id)
        for tdir, raw in self._scan_take_workspaces():
            record = dict(raw)
            if str(record.get("clip_id") or "") != str(clip_id):
                continue
            revision = str(record.get("revision") or "").strip()
            if not revision or revision in seen:
                continue
            seen.add(revision)
            record["take_name"] = str(record.get("take_name") or record.get("label") or "Take")
            record["label"] = record["take_name"]
            record["library_folder"] = os.path.basename(os.path.dirname(tdir))
            record["workspace_folder"] = os.path.basename(tdir)
            record["active_editor"] = bool(editor_active and revision == editor_active)
            out.append(record)
        # Read-only legacy fallback for anything that could not be migrated.
        legacy_take_dir = os.path.join(self._legacy_clip_dir(clip_id), "takes")
        if os.path.isdir(legacy_take_dir):
            for name in os.listdir(legacy_take_dir):
                if not name.endswith(".json"):
                    continue
                record = _read_json(os.path.join(legacy_take_dir, name))
                revision = str((record or {}).get("revision") or "").strip()
                if record and revision and revision not in seen:
                    record = dict(record)
                    record.setdefault("take_name", record.get("label") or f"Legacy {revision[:8]}")
                    record.setdefault("library_folder", "Legacy")
                    record["active_editor"] = bool(editor_active and revision == editor_active)
                    out.append(record)
        out.sort(key=lambda item: float(item.get("created_at_unix") or 0.0), reverse=True)
        return out
