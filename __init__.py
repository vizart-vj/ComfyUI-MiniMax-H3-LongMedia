__version__ = "0.6.60"

from . import windows_safetensors_compat as _windows_safetensors_compat  # noqa: F401
from . import lora_compat as _lora_compat  # noqa: F401
from . import fasth3_vsa_compat as _fasth3_vsa_compat  # noqa: F401
from . import fastvideo_vsa_compat as _fastvideo_vsa_compat  # noqa: F401
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"


def _install_director_take_routes():
    try:
        from aiohttp import web
        import asyncio
        import json
        import os
        import re
        import shutil
        import uuid
        import folder_paths
        from server import PromptServer
        from .director_cache import DirectorClipCache
        from .refmod_backend import (
            RefMod, load_bundle, load_refmod, normalize_metadata,
            read_refmod_meta, save_bundle, save_refmod,
        )
    except Exception:
        return

    routes = PromptServer.instance.routes

    def _refmod_safe(value, fallback='refmod'):
        text = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value or '').strip()).strip('._-')
        return (text or fallback)[:120]

    def _refmod_root(_project_id=None):
        # RefMods are shared by every Director project; folders live directly here.
        root = os.path.realpath(os.path.join(folder_paths.get_output_directory(), 'longmedia_refmods'))
        os.makedirs(root, exist_ok=True)
        return root, root

    def _refmod_migrate_legacy(root):
        # One-time move from longmedia_refmods/project-<uuid>/... into the shared
        # catalog. Preserve user folders; fold the former Unsorted bucket into root.
        legacy_name = re.compile(r'^project-(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{32})$', re.I)
        for dirname in os.listdir(root):
            project_dir = os.path.join(root, dirname)
            if not legacy_name.fullmatch(dirname) or not os.path.isdir(project_dir):
                continue
            for current, dirs, names in os.walk(project_dir, topdown=False):
                for name in names:
                    if not name.lower().endswith('.safetensors'):
                        continue
                    source_path = os.path.join(current, name)
                    relative = os.path.relpath(current, project_dir)
                    folder = '' if relative == '.' or relative.lower() == 'unsorted' else relative
                    target_dir = os.path.realpath(os.path.join(root, folder))
                    if os.path.commonpath([root, target_dir]) != root:
                        continue
                    os.makedirs(target_dir, exist_ok=True)
                    target = os.path.join(target_dir, name)
                    if os.path.exists(target):
                        target = os.path.join(target_dir, f'{os.path.splitext(name)[0]}__migrated_{uuid.uuid4().hex[:8]}.safetensors')
                    shutil.move(source_path, target)
                try:
                    os.rmdir(current)
                except OSError:
                    pass
            try:
                os.rmdir(project_dir)
            except OSError:
                pass

    # Only these are determined by writing the artifact. Every other field of a
    # RefMod record (mode, concept_type, description, resolution_short_edge,
    # max_tokens, grid_long_edge, clip_frames, voice_seconds,
    # refinement_steps, strength, source_*) is
    # authored in the Director document and is never reconstructed from disk.
    _REFMOD_SERVER_FIELDS = ('path', 'folder', 'state', 'kind', 'format_version', 'members')

    def _refmod_authored_record(record: dict, stored: dict, refmod_id: str) -> dict:
        """Authored settings with only the artifact-owned fields overlaid."""
        merged = {**record, **{key: stored[key] for key in _REFMOD_SERVER_FIELDS}}
        merged['refmod_id'] = refmod_id
        merged['strength'] = max(0.0, min(1.0, float(record.get('strength', 1.0) or 0.0)))
        return merged

    def _refmod_record(path, *, refmod_id=None, project_root=None):
        meta = read_refmod_meta(os.path.splitext(path)[0]) or {}
        normalized = normalize_metadata(meta)
        base = os.path.splitext(os.path.basename(path))[0]
        root = os.path.realpath(folder_paths.get_output_directory())
        real = os.path.realpath(path)
        stored = os.path.relpath(real, root).replace('\\', '/') if os.path.commonpath([root, real]) == root else real
        return {
            'refmod_id': str(refmod_id or normalized.get('refmod_id') or base),
            'name': str(normalized.get('name') or base),
            'kind': str(normalized.get('kind') or 'image'),
            'mode': str(normalized.get('mode') or 'FULL'),
            'concept_type': str(normalized.get('concept_type') or 'generic'),
            'description': str(normalized.get('description') or ''),
            'source': str(normalized.get('source') or ''),
            'path': stored,
            "folder": (os.path.relpath(os.path.dirname(real), project_root).replace('\\', '/') if project_root and os.path.commonpath([os.path.realpath(project_root), real]) == os.path.realpath(project_root) and os.path.relpath(os.path.dirname(real), project_root) != '.' else 'Unsorted'),
            'strength': 1.0,
            'format_version': int(normalized.get('_format_version') or 4),
            'members': normalized.get('members') if isinstance(normalized.get('members'), list) else [],
            'state': 'ready',
        }

    @routes.post('/longmedia/refmods/upload')
    async def _refmod_upload(request):
        try:
            reader = await request.multipart()
            part = await reader.next()
            while part is not None and part.name != 'file':
                part = await reader.next()
            filename = os.path.basename(str(part.filename or '')) if part else ''
            if not part or not filename.lower().endswith('.safetensors'):
                return web.json_response({'error': 'a .safetensors file is required'}, status=400)
            name = f"{uuid.uuid4().hex}_{_refmod_safe(filename, 'refmod.safetensors')}"
            if not name.lower().endswith('.safetensors'):
                name += '.safetensors'
            path = os.path.join(folder_paths.get_input_directory(), name)
            with open(path, 'wb') as stream:
                while chunk := await part.read_chunk(1024 * 1024):
                    stream.write(chunk)
            return web.json_response({'name': name, 'subfolder': ''})
        except Exception as exc:
            return web.json_response({'error': f'upload failed: {type(exc).__name__}: {exc}'}, status=400)

    @routes.post('/longmedia/refmods/source_upload')
    async def _refmod_source_upload(request):
        """Store one raw media source in ComfyUI input for RefMod creation."""
        allowed = {
            '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tif', '.tiff',
            '.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v',
            '.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac',
        }
        path = None
        try:
            if request.content_length is not None and request.content_length > 2 * 1024 * 1024 * 1024:
                return web.json_response({'error': 'source exceeds the 2 GiB upload limit'}, status=413)
            reader = await request.multipart()
            part = await reader.next()
            while part is not None and part.name != 'file':
                part = await reader.next()
            filename = os.path.basename(str(part.filename or '')) if part else ''
            extension = os.path.splitext(filename)[1].lower()
            if not part or extension not in allowed:
                return web.json_response({'error': 'supported image, video or audio file required'}, status=400)
            name = f"lmd_refmod_{uuid.uuid4().hex}_{_refmod_safe(filename, 'source' + extension)}"
            if not name.lower().endswith(extension):
                name += extension
            path = os.path.realpath(os.path.join(folder_paths.get_input_directory(), name))
            input_root = os.path.realpath(folder_paths.get_input_directory())
            if os.path.commonpath([input_root, path]) != input_root:
                return web.json_response({'error': 'invalid upload path'}, status=400)
            total = 0
            with open(path, 'wb') as stream:
                while chunk := await part.read_chunk(1024 * 1024):
                    total += len(chunk)
                    if total > 2 * 1024 * 1024 * 1024:
                        stream.close()
                        os.remove(path)
                        path = None
                        return web.json_response({'error': 'source exceeds the 2 GiB upload limit'}, status=413)
                    stream.write(chunk)
            if total <= 0:
                os.remove(path)
                path = None
                return web.json_response({'error': 'source file is empty'}, status=400)
            return web.json_response({'name': name, 'subfolder': '', 'bytes': total})
        except Exception as exc:
            if path and os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            return web.json_response({'error': f'upload failed: {type(exc).__name__}: {exc}'}, status=400)

    @routes.post('/longmedia/refmods/list')
    async def _refmod_list(request):
        body = await request.json()
        _, project = _refmod_root(body.get('project_id'))
        _refmod_migrate_legacy(project)
        records = []
        folders = ['Unsorted']
        for current, dirs, names in os.walk(project):
            dirs[:] = sorted(dirs, key=str.lower)
            relative = os.path.relpath(current, project).replace('\\', '/')
            if relative != '.':
                folders.append(relative)
            for name in sorted(names, key=str.lower):
                if not name.lower().endswith('.safetensors'):
                    continue
                try:
                    records.append(_refmod_record(os.path.join(current, name), project_root=project))
                except Exception:
                    continue
        return web.json_response({'refmods': records, 'folders': sorted(set(folders), key=str.lower)})

    @routes.post('/longmedia/refmods/folder')
    async def _refmod_folder(request):
        body = await request.json()
        name = _refmod_safe(body.get('name'), '')
        if not name:
            return web.json_response({'error': 'folder name is required'}, status=400)
        _, project = _refmod_root(body.get('project_id'))
        path = os.path.realpath(os.path.join(project, name))
        if os.path.commonpath([project, path]) != project:
            return web.json_response({'error': 'invalid folder name'}, status=400)
        os.makedirs(path, exist_ok=True)
        return web.json_response({'ok': True, 'name': name})

    @routes.post('/longmedia/refmods/from_take')
    async def _refmod_from_take(request):
        body = await request.json()
        project_id = str(body.get('project_id') or '').strip()
        clip_id = str(body.get('clip_id') or '').strip()
        revision = str(body.get('revision') or '').strip()
        if not project_id or not clip_id or not revision:
            return web.json_response({'error': 'project_id, clip_id and revision are required'}, status=400)
        cache = DirectorClipCache(folder_paths.get_output_directory(), project_id)
        take = cache.load_revision(clip_id, revision)
        if not isinstance(take, dict):
            return web.json_response({'error': 'take revision not found'}, status=404)
        video = take.get('display_video')
        audio = take.get('display_audio')
        if video is None:
            return web.json_response({'error': 'take has no native display latent'}, status=409)
        take_meta = dict(take.get('metadata') or {})
        refmod_id = uuid.uuid4().hex
        label = str(take_meta.get('take_name') or take_meta.get('label') or f'TAKE {clip_id}')
        refs = [RefMod(
            name=label, kind='video' if int(video.shape[2]) > 1 else 'image',
            latent=video.detach().to(device='cpu').contiguous(), mode='FULL',
            source=f'take:{project_id}:{clip_id}:{revision}', concept_type='motion',
            description='Native LongMedia TAKE visual latent; no decode/re-encode.',
        )]
        if audio is not None and int(audio.numel()) > 0 and int(audio.shape[-1]) > 0:
            refs.append(RefMod(
                name=f'{label} Audio Reference', kind='audio',
                latent=audio.detach().to(device='cpu').contiguous(), mode='FULL',
                source=f'take:{project_id}:{clip_id}:{revision}', concept_type='audio_reference',
                description='Native LongMedia TAKE audio latent; reference conditioning, not voice cloning.',
            ))
        _, project = _refmod_root(project_id)
        stem = os.path.join(project, f'{_refmod_safe(label)}__{_refmod_safe(refmod_id)}')
        path = save_bundle(refs, label, stem) if len(refs) > 1 else save_refmod(refs[0], stem)
        return web.json_response({'ok': True, 'refmod': _refmod_record(path, refmod_id=refmod_id, project_root=project)})

    @routes.post('/longmedia/refmods/save')
    async def _refmod_save(request):
        body = await request.json()
        project_id = str(body.get('project_id') or '').strip()
        record = body.get('refmod') if isinstance(body.get('refmod'), dict) else {}
        if not project_id or not record:
            return web.json_response({'error': 'project_id and refmod are required'}, status=400)
        refmod_id = _refmod_safe(record.get('refmod_id'), uuid.uuid4().hex)
        _, project = _refmod_root(project_id)
        source = str(record.get('source') or '').strip()
        source_path = os.path.realpath(os.path.expanduser(source)) if source else ''
        if source and not os.path.isabs(source_path):
            input_candidate = os.path.realpath(os.path.join(folder_paths.get_input_directory(), source.replace('\\\\', '/').removeprefix('input/')))
            if os.path.commonpath([os.path.realpath(folder_paths.get_input_directory()), input_candidate]) == os.path.realpath(folder_paths.get_input_directory()):
                source_path = input_candidate
        if source_path and os.path.isfile(source_path) and source_path.lower().endswith('.safetensors'):
            # Import a compatible v4/v5 artifact into the managed library without
            # changing tensors or metadata; validation is header-first then full load.
            meta = read_refmod_meta(os.path.splitext(source_path)[0])
            if not isinstance(meta, dict):
                return web.json_response({'error': 'source is not a readable RefMod safetensors file'}, status=400)
            try:
                if str(meta.get('kind') or '').lower() == 'bundle' or int(meta.get('_format_version') or 0) >= 5:
                    load_bundle(os.path.splitext(source_path)[0])
                else:
                    load_refmod(os.path.splitext(source_path)[0])
            except Exception as exc:
                return web.json_response({'error': f'incompatible RefMod: {type(exc).__name__}: {exc}'}, status=400)
            folder = _refmod_safe(record.get('folder'), 'Unsorted')
            target_dir = project if folder.lower() == 'unsorted' else os.path.realpath(os.path.join(project, folder))
            if os.path.commonpath([project, target_dir]) != project:
                return web.json_response({'error': 'invalid library folder'}, status=400)
            os.makedirs(target_dir, exist_ok=True)
            target = os.path.join(target_dir, f'{_refmod_safe(record.get("name"), refmod_id)}__{refmod_id}.safetensors')
            tmp = target + f'.{uuid.uuid4().hex}.tmp'
            shutil.copyfile(source_path, tmp)
            os.replace(tmp, target)
            stored = _refmod_record(target, refmod_id=refmod_id, project_root=project)
            return web.json_response({'ok': True, 'refmod': _refmod_authored_record(record, stored, refmod_id)})
        # Pixel/audio sources require the active H3 VAE objects and are intentionally
        # encoded by Long Media Setup on its next execution, never by a hidden model.
        pending = dict(record)
        pending['refmod_id'] = refmod_id
        folder = _refmod_safe(record.get('folder'), 'Unsorted')
        target_dir = project if folder.lower() == 'unsorted' else os.path.realpath(os.path.join(project, folder))
        if os.path.commonpath([project, target_dir]) != project:
            return web.json_response({'error': 'invalid library folder'}, status=400)
        os.makedirs(target_dir, exist_ok=True)
        pending['folder'] = folder
        pending['path'] = os.path.relpath(
            os.path.join(target_dir, f'{_refmod_safe(record.get("name"), refmod_id)}__{refmod_id}.safetensors'),
            folder_paths.get_output_directory(),
        ).replace('\\', '/')
        pending['state'] = 'pending_runtime_encode'
        return web.json_response({'ok': True, 'pending': True, 'refmod': pending})

    @routes.post('/longmedia/refmods/delete')
    async def _refmod_delete(request):
        body = await request.json()
        refmod_id = _refmod_safe(body.get('refmod_id'), '')
        _, project = _refmod_root(body.get('project_id'))
        if not refmod_id:
            return web.json_response({'error': 'refmod_id is required'}, status=400)
        removed = []
        for current, _, names in os.walk(project):
            for name in names:
                stem = os.path.splitext(name)[0]
                if refmod_id not in (stem, stem.rsplit('__', 1)[-1]):
                    continue
                path = os.path.realpath(os.path.join(current, name))
                if os.path.commonpath([project, path]) == project and os.path.isfile(path):
                    os.remove(path)
                    removed.append(os.path.relpath(path, project).replace('\\', '/'))
        return web.json_response({'ok': True, 'removed': removed})

    @routes.get('/longmedia/refmods/vaes')
    async def _refmod_vaes(_request):
        """Which MiniMax H3 VAEs Director ENCODE would use. Nothing is loaded."""
        try:
            from .nodes import _lm_refmod_vae_catalog
            return web.json_response({'ok': True, **_lm_refmod_vae_catalog()})
        except Exception as exc:
            return web.json_response({'error': f'{type(exc).__name__}: {exc}'}, status=400)

    @routes.post('/longmedia/refmods/encode')
    async def _refmod_encode(request):
        """Director ENCODE: VAE-only RefMod preparation.

        Loads the MiniMax H3 VideoVAE/AudioVAE and writes the latent artifact.
        Text encoder, DiT, sampler and Long Media Setup are never touched here.
        """
        body = await request.json()
        project_id = str(body.get('project_id') or '').strip()
        record = body.get('refmod') if isinstance(body.get('refmod'), dict) else {}
        director_json = body.get('director_json')
        if not project_id or not record or director_json in (None, ''):
            return web.json_response(
                {'error': 'project_id, refmod and director_json are required'}, status=400,
            )

        def _encode_blocking():
            from .nodes import _lm_refmod_encode_one
            _, project = _refmod_root(project_id)
            refmod_id = str(record.get('refmod_id') or '').strip() or uuid.uuid4().hex
            label = str(record.get('name') or refmod_id)
            result = _lm_refmod_encode_one(
                {**record, 'refmod_id': refmod_id}, director_json, project_id,
                video_vae_name=body.get('video_vae_name'),
                audio_vae_name=body.get('audio_vae_name'),
                target_stem=os.path.join(project, f'{_refmod_safe(label)}__{_refmod_safe(refmod_id)}'),
            )
            stored = _refmod_record(result['path'], refmod_id=refmod_id, project_root=project)
            result['refmod'] = _refmod_authored_record(record, stored, refmod_id)
            result['refmod']['state'] = 'ready'
            result['state'] = 'ready'
            return result

        try:
            result = await asyncio.to_thread(_encode_blocking)
        except Exception as exc:
            return web.json_response({'error': f'{type(exc).__name__}: {exc}'}, status=400)
        return web.json_response({'ok': True, **result})

    @routes.post('/longmedia/refmods/encode_sources')
    async def _refmod_encode_sources(request):
        """Create one RefMod bundle from independently uploaded image/video/audio files."""
        body = await request.json()
        project_id = str(body.get('project_id') or '').strip()
        record = body.get('refmod') if isinstance(body.get('refmod'), dict) else {}
        sources = body.get('sources') if isinstance(body.get('sources'), list) else []
        if not project_id or not record or not sources:
            return web.json_response({'error': 'project_id, RefMod settings and at least one source are required'}, status=400)
        selected = [item for item in sources if isinstance(item, dict) and item.get('selected', True)]
        if not selected:
            return web.json_response({'error': 'select at least one source'}, status=400)
        if len(selected) > 128:
            return web.json_response({'error': 'a RefMod can contain at most 128 sources'}, status=400)
        if any(str(item.get('kind') or '').lower() not in {'image', 'video', 'audio'} for item in selected):
            return web.json_response({'error': 'sources must be image, video or audio'}, status=400)

        def _encode_blocking():
            from .nodes import _lm_load_h3_vae, _lm_refmod_encode_uploaded_sources
            _, project = _refmod_root(project_id)
            refmod_id = _refmod_safe(record.get('refmod_id'), uuid.uuid4().hex)
            label = str(record.get('name') or refmod_id)
            required_kinds = {
                'audio' if str(source.get('kind') or '').lower() == 'audio' else 'video'
                for source in selected
            }
            video_vae, audio_vae, used = _lm_load_h3_vae(
                body.get('video_vae_name'), body.get('audio_vae_name'),
                required_kinds=required_kinds,
            )
            result = _lm_refmod_encode_uploaded_sources(
                {**record, 'refmod_id': refmod_id}, selected, video_vae, audio_vae,
                target_stem=os.path.join(project, f'{_refmod_safe(label)}__{refmod_id}'),
            )
            result['vaes'] = used
            stored = _refmod_record(result['path'], refmod_id=refmod_id, project_root=project)
            result['refmod'] = _refmod_authored_record(record, stored, refmod_id)
            result['refmod']['state'] = 'ready'
            result['state'] = 'ready'
            return result

        try:
            result = await asyncio.to_thread(_encode_blocking)
        except Exception as exc:
            return web.json_response({'error': f'{type(exc).__name__}: {exc}'}, status=400)
        return web.json_response({'ok': True, **result})

    @routes.get('/longmedia/director/takes')
    async def _director_takes(request):
        project_id = str(request.query.get('project_id') or '').strip()
        clip_id = str(request.query.get('clip_id') or '').strip()
        if not project_id or not clip_id:
            return web.json_response({'error': 'project_id and clip_id are required'}, status=400)
        cache = DirectorClipCache(folder_paths.get_output_directory(), project_id)
        active = cache.active_metadata(clip_id)
        takes = cache.list_take_metadata(clip_id)
        active_revision = str((active or {}).get('revision') or '')
        editor_active_revision = str(cache.editor_active_revision(clip_id) or '')
        for take in takes:
            take['active'] = str(take.get('revision') or '') == editor_active_revision
            take['render_active'] = bool(active_revision and str(take.get('revision') or '') == active_revision)
        return web.json_response({'project_id': project_id, 'clip_id': clip_id, 'active_revision': active_revision or None, 'editor_active_revision': editor_active_revision or None, 'takes': takes, 'folders': cache.list_library_folders()})

    @routes.post('/longmedia/director/activate_take')
    async def _director_activate_take(request):
        body = await request.json()
        project_id = str(body.get('project_id') or '').strip()
        clip_id = str(body.get('clip_id') or '').strip()
        revision = str(body.get('revision') or '').strip()
        order = [str(v) for v in (body.get('clip_order') or []) if str(v)]
        restore_timeline = bool(body.get('restore_timeline', True))
        if not project_id or not clip_id or not revision:
            return web.json_response({'error': 'project_id, clip_id and revision are required'}, status=400)
        cache = DirectorClipCache(folder_paths.get_output_directory(), project_id)
        candidate = next((x for x in cache.list_take_metadata(clip_id) if str(x.get('revision') or '') == revision), None)
        if not candidate:
            return web.json_response({'error': 'take revision not found'}, status=404)
        is_draft = bool(candidate.get('draft')) or str(candidate.get('state') or '') == 'draft'
        if is_draft:
            if not restore_timeline:
                return web.json_response({'error': 'draft TAKE has no rendered AV to activate'}, status=409)
            cache._set_editor_active(clip_id, revision)
            candidate['active'] = True
            return web.json_response({
                'ok': True, 'take': candidate, 'draft': True, 'activated': [],
                'changed_clips': [], 'compatibility': {}, 'restore_reason': 'draft_settings_snapshot',
            })
        if restore_timeline:
            snapshot = candidate.get('director_timeline_snapshot')
            if isinstance(snapshot, dict):
                snapshot_shots = snapshot.get('shots') if isinstance(snapshot.get('shots'), list) else []
                snapshot_order = [
                    str(item.get('clip_id') or '').strip()
                    for item in snapshot_shots if isinstance(item, dict) and str(item.get('clip_id') or '').strip()
                ]
                if clip_id in snapshot_order:
                    order = snapshot_order

        # v0.5.57: a historical take owns a branch, not an obligation to splice
        # into the currently active suffix. Recover the compatible cached suffix
        # first, then move active pointers as one editorial operation.
        if order:
            cache.ensure_active_chain_lineage(order)
            try:
                result = cache.restore_take_branch(
                    order, clip_id, revision, allow_partial=restore_timeline
                )
            except Exception as exc:
                return web.json_response({
                    'error': f'branch restore failed: {type(exc).__name__}: {exc}',
                }, status=500)
            if not bool(result.get('ok')):
                reason = str(result.get('reason') or 'branch_restore_unavailable')
                messages = {
                    'active_previous_take_missing': 'the active previous clip take is missing',
                    'incoming_branch_mismatch': 'this take belongs to a different upstream branch',
                    'no_compatible_cached_suffix': 'no compatible cached downstream branch exists for this take',
                    'clip_not_in_timeline': 'the clip is no longer present in the current timeline',
                    'take_revision_not_found': 'take revision not found',
                }
                return web.json_response({
                    'error': messages.get(reason, reason),
                    'reason': reason,
                    'compatibility': result.get('compatibility') or {},
                    'failed_clip_id': result.get('failed_clip_id'),
                }, status=409)
            return web.json_response({
                'ok': True,
                'take': result.get('take'),
                'activated': result.get('activated') or [],
                'changed_clips': result.get('changed_clips') or [],
                'partial_restore': bool(result.get('partial')),
                'invalidated_clips': result.get('invalidated_clips') or [],
                'failed_clip_id': result.get('failed_clip_id'),
                'compatibility': result.get('compatibility') or {},
                'restore_reason': result.get('reason'),
            })

        # Legacy/API fallback when no timeline order is supplied.
        record = cache.activate_revision(clip_id, revision)
        return web.json_response({'ok': True, 'take': record, 'activated': [{'clip_id': clip_id, 'revision': revision, 'changed': True}]})


    @routes.post('/longmedia/director/create_take')
    async def _director_create_take(request):
        body = await request.json()
        project_id = str(body.get('project_id') or '').strip()
        clip_id = str(body.get('clip_id') or '').strip()
        clip_index = int(body.get('clip_index') or 0)
        snapshot = body.get('director_timeline_snapshot')
        shot_snapshot = body.get('shot_snapshot')
        if not project_id or not clip_id:
            return web.json_response({'error': 'project_id and clip_id are required'}, status=400)
        cache = DirectorClipCache(folder_paths.get_output_directory(), project_id)
        metadata = {
            'director_timeline_snapshot': snapshot if isinstance(snapshot, dict) else None,
            'director_global_prompt_snapshot': str(body.get('director_global_prompt_snapshot') or (snapshot.get('_global_prompt') if isinstance(snapshot, dict) else '') or ''),
            'shot_snapshot': shot_snapshot if isinstance(shot_snapshot, dict) else None,
            'effective_seed': body.get('effective_seed'),
            'label': str(body.get('label') or '').strip() or None,
            'take_name': str(body.get('take_name') or body.get('label') or '').strip() or None,
            'library_folder': str(body.get('library_folder') or 'Unsorted').strip() or 'Unsorted',
        }
        try:
            record = cache.create_draft_take(clip_id=clip_id, clip_index=clip_index, metadata=metadata)
        except Exception as exc:
            return web.json_response({'error': f'{type(exc).__name__}: {exc}'}, status=500)
        return web.json_response({'ok': True, 'take': record})


    @routes.post('/longmedia/director/duplicate_take')
    async def _director_duplicate_take(request):
        body = await request.json()
        project_id = str(body.get('project_id') or '').strip()
        clip_id = str(body.get('clip_id') or '').strip()
        revision = str(body.get('revision') or '').strip()
        if not project_id or not clip_id or not revision:
            return web.json_response({'error': 'project_id, clip_id and revision are required'}, status=400)
        cache = DirectorClipCache(folder_paths.get_output_directory(), project_id)
        try:
            record = cache.duplicate_revision(clip_id, revision)
        except FileNotFoundError as exc:
            return web.json_response({'error': str(exc)}, status=404)
        except Exception as exc:
            return web.json_response({'error': f'{type(exc).__name__}: {exc}'}, status=500)
        return web.json_response({'ok': True, 'take': record})

    @routes.post('/longmedia/director/delete_take')
    async def _director_delete_take(request):
        body = await request.json()
        project_id = str(body.get('project_id') or '').strip()
        clip_id = str(body.get('clip_id') or '').strip()
        revision = str(body.get('revision') or '').strip()
        if not project_id or not clip_id or not revision:
            return web.json_response({'error': 'project_id, clip_id and revision are required'}, status=400)
        cache = DirectorClipCache(folder_paths.get_output_directory(), project_id)
        try:
            result = cache.delete_revision(clip_id, revision)
        except FileNotFoundError as exc:
            return web.json_response({'error': str(exc)}, status=404)
        except Exception as exc:
            return web.json_response({'error': f'{type(exc).__name__}: {exc}'}, status=500)
        return web.json_response(result)

    @routes.post('/longmedia/director/rename_take')
    async def _director_rename_take(request):
        body = await request.json()
        project_id = str(body.get('project_id') or '').strip()
        clip_id = str(body.get('clip_id') or '').strip()
        revision = str(body.get('revision') or '').strip()
        name = str(body.get('name') or '').strip()
        if not project_id or not clip_id or not revision or not name:
            return web.json_response({'error': 'project_id, clip_id, revision and name are required'}, status=400)
        cache = DirectorClipCache(folder_paths.get_output_directory(), project_id)
        try:
            result = cache.rename_take(clip_id, revision, name)
        except FileNotFoundError as exc:
            return web.json_response({'error': str(exc)}, status=404)
        except Exception as exc:
            return web.json_response({'error': f'{type(exc).__name__}: {exc}'}, status=500)
        return web.json_response(result)

    @routes.get('/longmedia/director/take_folders')
    async def _director_take_folders(request):
        project_id = str(request.query.get('project_id') or '').strip()
        if not project_id:
            return web.json_response({'error': 'project_id is required'}, status=400)
        cache = DirectorClipCache(folder_paths.get_output_directory(), project_id)
        return web.json_response({'project_id': project_id, 'folders': cache.list_library_folders()})

    @routes.post('/longmedia/director/create_take_folder')
    async def _director_create_take_folder(request):
        body = await request.json()
        project_id = str(body.get('project_id') or '').strip()
        name = str(body.get('name') or '').strip()
        if not project_id or not name:
            return web.json_response({'error': 'project_id and name are required'}, status=400)
        cache = DirectorClipCache(folder_paths.get_output_directory(), project_id)
        try:
            result = cache.create_library_folder(name)
        except FileExistsError as exc:
            return web.json_response({'error': str(exc)}, status=409)
        except Exception as exc:
            return web.json_response({'error': f'{type(exc).__name__}: {exc}'}, status=500)
        return web.json_response(result)

    @routes.post('/longmedia/director/move_take')
    async def _director_move_take(request):
        body = await request.json()
        project_id = str(body.get('project_id') or '').strip()
        clip_id = str(body.get('clip_id') or '').strip()
        revision = str(body.get('revision') or '').strip()
        folder = str(body.get('folder') or '').strip()
        if not project_id or not clip_id or not revision or not folder:
            return web.json_response({'error': 'project_id, clip_id, revision and folder are required'}, status=400)
        cache = DirectorClipCache(folder_paths.get_output_directory(), project_id)
        try:
            result = cache.move_take_to_folder(clip_id, revision, folder)
        except FileNotFoundError as exc:
            return web.json_response({'error': str(exc)}, status=404)
        except Exception as exc:
            return web.json_response({'error': f'{type(exc).__name__}: {exc}'}, status=500)
        return web.json_response(result)

    @routes.get('/longmedia/embeddings')
    async def _longmedia_embeddings(request):
        """Return the live ComfyUI embedding catalog for Director timeline layers."""
        try:
            from safetensors import safe_open

            names = []
            for value in folder_paths.get_filename_list('embeddings'):
                stored = str(value or '').replace('\\', '/').strip()
                if not stored or not stored.lower().endswith('.safetensors'):
                    continue
                # Fast path for the official Comfy-Org naming convention. Custom
                # DiffSynth exports are accepted too after a header-only shape check.
                compatible = stored.lower().split('/')[-1].startswith('minimaxh3_')
                if not compatible:
                    try:
                        path = folder_paths.get_full_path_or_raise('embeddings', value)
                        with safe_open(path, framework='pt', device='cpu') as handle:
                            for key in handle.keys():
                                shape = tuple(int(v) for v in handle.get_slice(key).get_shape())
                                if len(shape) == 2 and shape[-1] == 5120:
                                    compatible = True
                                    break
                    except Exception:
                        compatible = False
                if not compatible:
                    continue
                name = stored[:-12]
                if name and name not in names:
                    names.append(name)
            names.sort(key=lambda x: (0 if x.lower().startswith('minimaxh3_') else 1, x.lower()))
            return web.json_response({'embeddings': names})
        except Exception as exc:
            return web.json_response({'error': f'{type(exc).__name__}: {exc}', 'embeddings': []}, status=500)

    @routes.get('/longmedia/director/status')
    async def _director_status(request):
        project_id = str(request.query.get('project_id') or '').strip()
        clip_ids = [v for v in str(request.query.get('clip_ids') or '').split(',') if v]
        if not project_id:
            return web.json_response({'error': 'project_id is required'}, status=400)
        cache = DirectorClipCache(folder_paths.get_output_directory(), project_id)
        if clip_ids:
            cache.ensure_active_chain_lineage(clip_ids)
        clips = {}
        for clip_id in clip_ids:
            active = cache.active_metadata(clip_id)
            takes = cache.list_take_metadata(clip_id)
            active_revision = str((active or {}).get('revision') or '')
            editor_active_revision = str(cache.editor_active_revision(clip_id) or '')
            for take in takes:
                take['active'] = bool(editor_active_revision and str(take.get('revision') or '') == editor_active_revision)
                take['render_active'] = bool(active_revision and str(take.get('revision') or '') == active_revision)
            clips[clip_id] = {
                'active': active,
                'active_revision': active_revision or None,
                'editor_active_revision': editor_active_revision or None,
                'take_count': len(takes),
                'takes': takes,
            }
        known_clips = set(clip_ids)
        all_takes = []
        gallery_fields = {
            'clip_id', 'revision', 'take_name', 'label', 'library_folder',
            'workspace_folder', 'created_at_unix', 'effective_seed', 'draft',
            'state', 'preview_file', 'preview_generated_at_unix', 'active',
            'render_active',
        }
        for take in cache.list_all_take_metadata():
            # Current clips already include full manifests above. Keep the project-wide
            # supplement limited to other clips, so snapshots are not duplicated.
            if str(take.get('clip_id') or '') in known_clips:
                continue
            all_takes.append({key: value for key, value in take.items() if key in gallery_fields})
        return web.json_response({
            'project_id': project_id,
            'clips': clips,
            'all_takes': all_takes,
            'folders': cache.list_library_folders(),
        })


_install_director_take_routes()

__all__ = ["__version__", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
