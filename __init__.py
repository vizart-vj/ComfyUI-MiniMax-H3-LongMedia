__version__ = "0.6.41"

from . import lora_compat as _lora_compat  # noqa: F401
from . import fasth3_vsa_compat as _fasth3_vsa_compat  # noqa: F401
from . import fastvideo_vsa_compat as _fastvideo_vsa_compat  # noqa: F401
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"


def _install_director_take_routes():
    try:
        from aiohttp import web
        import folder_paths
        from server import PromptServer
        from .director_cache import DirectorClipCache
    except Exception:
        return

    routes = PromptServer.instance.routes

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
                result = cache.restore_take_branch(order, clip_id, revision)
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
        return web.json_response({'project_id': project_id, 'clips': clips, 'folders': cache.list_library_folders()})


_install_director_take_routes()

__all__ = ["__version__", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
