"""Regression tests for approved-prefix + new-suffix Director execution."""
from __future__ import annotations

import ast
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from director_plan import director_validate_enabled_run_scope


class EnabledAppendTests(unittest.TestCase):
    def test_new_enabled_suffix_does_not_require_previous_take(self):
        self.assertEqual(
            director_validate_enabled_run_scope(
                (False, True, True), ('generated',) * 3, (True, False, False)
            ), (1, 2),
        )

    def test_disabled_missing_take_is_rejected(self):
        with self.assertRaises(RuntimeError):
            director_validate_enabled_run_scope(
                (False, True), ('generated',) * 2, (False, False)
            )

    def test_media_is_not_a_generated_cache_dependency(self):
        self.assertEqual(
            director_validate_enabled_run_scope(
                (False, True), ('media', 'generated'), (False, False)
            ), (1,),
        )


class SamplerAppendTests(unittest.TestCase):
    """Execute the production Sampler cache preflight, with real on-disk TAKEs.

    The neural sampler/VAEs are deliberately outside this fast offline seam.
    AST loading avoids booting ComfyUI or importing unrelated GPU accelerators.
    """
    def setUp(self):
        import tempfile
        import torch
        from director_cache import DirectorClipCache, canonical_hash
        from director_plan import director_enabled_render_indices
        from continuation_context import build_director_render_units
        from latent_ops import unpack_av_samples, video_latent_t, audio_latent_t
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = self.temp.name
        self.torch = torch
        self.cache = DirectorClipCache(self.output, 'append-test')
        source = (ROOT / 'nodes.py').read_text(encoding='utf-8')
        names = {
            '_v85_segment_seed', '_lm_director_clip_geometry_fingerprint',
            '_lm_director_outgoing_contract_fingerprint',
            '_lm_director_clip_semantic_fingerprint', '_lm_director_take_matches',
            '_lm_director_take_shape_compatibility',
            '_lm_director_legacy_single_take_matches_prefix',
            '_lm_director_auto_incremental_plan', '_lm_director_is_media',
            '_lm_director_render_clip_metadata', '_lm_director_semantic_payload',
            '_lm_director_legacy_semantic_matches',
        }
        self.env = {
            'torch': torch,
            '_director_canonical_hash': canonical_hash,
            '_DirectorClipCache': DirectorClipCache,
            '_director_enabled_render_indices': director_enabled_render_indices,
            '_director_validate_enabled_run_scope': director_validate_enabled_run_scope,
            'build_director_render_units': build_director_render_units,
            'unpack_av_samples': unpack_av_samples,
            'video_latent_t': video_latent_t, 'audio_latent_t': audio_latent_t,
            '_lm_print': lambda *args, **kwargs: None,
        }
        module = ast.Module(body=[item for item in ast.parse(source).body
                                  if isinstance(item, ast.FunctionDef) and item.name in names], type_ignores=[])
        exec(compile(module, str(ROOT / 'nodes.py'), 'exec'), self.env)
        import textwrap
        start = source.index('        # Director persistent per-clip selective cache.')
        end = source.index('        # A repeated already-completed request', start)
        self.preflight = compile(textwrap.dedent(source[start:end]), 'Sampler.production.preflight', 'exec')
        self.video = torch.arange(1 * 24 * 7 * 2 * 3, dtype=torch.float32).reshape(1, 24, 7, 2, 3)
        self.audio = torch.arange(1 * 32 * 2 * 37, dtype=torch.float32).reshape(1, 32, 2, 37)
        self.hires = dict(latent_hires_enabled=False, latent_hires_model='',
                          latent_hires_scale=1.0, latent_hires_precision='', latent_hires_align=0)

    def plan(self, enabled=(False, True, True), mode='none', target=None):
        count = len(enabled)
        return SimpleNamespace(
            mode='multiclip', workflow_mode='multiclip', overlap_frames=5,
            segment_lengths=(22,) * count, director_project_id='append-test',
            director_clip_ids=tuple(f'clip-{i}' for i in range(count)),
            director_clip_enabled=enabled, director_base_kinds=('generated',) * count,
            director_clip_metadata=tuple(dict(clip_id=f'clip-{i}', name=f'Clip {i}',
                duration=22 / 24, run_enabled=enabled[i], camera={'transition_to_next': i + 1 < count})
                for i in range(count)),
            segment_prompt_summaries=tuple(f'prompt {i}' for i in range(count)),
            director_reference_fingerprint='same-refs', segment_seeds=None,
            director_regeneration={'mode': mode, 'clip_id': target},
        )

    def save(self, plan, index=0, legacy_single=False, legacy_semantic=False):
        clip_metadata = copy.deepcopy(plan.director_clip_metadata[index])
        metadata = {
            'clip_metadata': clip_metadata, 'overlap_frames': 5,
            'segment_length_frames': 22, 'prompt_summary': plan.segment_prompt_summaries[index],
            'effective_seed': 100 + index, 'reference_fingerprint': 'same-refs',
            'geometry_fingerprint': self.env['_lm_director_clip_geometry_fingerprint'](
                plan, self.video, self.audio, index, **self.hires),
            'semantic_fingerprint': self.env['_lm_director_clip_semantic_fingerprint'](plan, 100, index),
            'outgoing_contract_fingerprint': self.env['_lm_director_outgoing_contract_fingerprint'](plan, index),
            '_has_next_clip': index + 1 < len(plan.director_clip_ids),
        }
        if legacy_single:
            metadata.update(take_scope='director_timeline', geometry_fingerprint=None, semantic_fingerprint=None)
        if legacy_semantic:
            metadata['semantic_fingerprint'] = self.env['_director_canonical_hash']({
                'clip_id': plan.director_clip_ids[index],
                'prompt': plan.segment_prompt_summaries[index],
                'effective_seed': 100 + index, 'clip_metadata': clip_metadata,
                'reference_fingerprint': 'same-refs',
                'audio_output_mode': 'auto', 'lip_sync_locked': False,
            })
        return self.cache.save_take(clip_id=plan.director_clip_ids[index], clip_index=index,
            display_video=self.video, display_audio=self.audio,
            continuation_video=self.video, continuation_audio=self.audio, metadata=metadata)

    def run_preflight(self, plan, seed=100):
        from types import ModuleType
        folder_paths = ModuleType('folder_paths')
        folder_paths.get_output_directory = lambda: self.output
        old = sys.modules.get('folder_paths')
        sys.modules['folder_paths'] = folder_paths
        class Streams:
            is_nested = True
            def unbind(inner):
                return self.video, self.audio
        env = dict(self.env, plan=plan, passes=len(plan.director_clip_ids), seed=seed,
                   workflow_name='multiclip', initial_av={'samples': Streams()},
                   external_refine_global_mode=False, **self.hires)
        try:
            exec(self.preflight, env)
        finally:
            if old is None:
                sys.modules.pop('folder_paths', None)
            else:
                sys.modules['folder_paths'] = old
        return env

    def test_disabled_single_take_stays_frozen_when_next_queue_randomizes_seed(self):
        original = self.plan((True,))
        saved = self.save(original, legacy_single=True)
        appended = self.plan((False, True))
        result = self.run_preflight(appended, seed=200)
        self.assertEqual(result['director_render_indices'], {1})
        self.assertEqual(set(result['director_cached_takes']), {0})
        self.assertEqual(self.cache.active_metadata('clip-0')['revision'], saved['revision'])
        # Freezing must not falsely certify old tensors as generated at seed 200.
        reenabled = self.run_preflight(self.plan((True, True)), seed=200)
        self.assertEqual(reenabled['director_render_indices'], {0, 1})

    def test_consecutive_new_targets_lock_only_the_frozen_right_neighbor(self):
        plan = self.plan((False, True, True, False))
        for index in (0, 2, 3):
            self.save(plan, index=index)
        result = self.run_preflight(plan)
        self.assertEqual(result['director_render_indices'], {1, 2})
        self.assertEqual(result['director_clip_only_targets'], {2})
        self.assertEqual(set(result['director_cached_takes']), {0, 2, 3})

    def test_disabled_prefix_changed_duration_is_rejected_without_resampling(self):
        original = self.plan((True,))
        self.save(original, legacy_single=True)
        changed = self.plan((False, True))
        changed.segment_lengths = (39, 22)
        with self.assertRaisesRegex(RuntimeError, 'geometry-compatible'):
            self.run_preflight(changed)

    def test_legacy_hash_migration_does_not_accept_a_changed_prompt(self):
        original = self.plan((True, True))
        self.save(original, legacy_semantic=True)
        changed = self.plan((True, True, True))
        changed.segment_prompt_summaries = ('a different scene', 'prompt 1', 'prompt 2')
        result = self.run_preflight(changed)
        self.assertEqual(result['director_render_indices'], {0, 1, 2})
        self.assertNotIn(0, result['director_cache_report'].get('legacy_semantic_revalidated_indices', []))

    def test_cache_initialization_failure_does_not_start_full_render(self):
        def broken_cache(*args):
            raise OSError('cache storage unavailable')
        self.env['_DirectorClipCache'] = broken_cache
        with self.assertRaisesRegex(RuntimeError, 'refusing an implicit full render'):
            self.run_preflight(self.plan((True, True), mode='clip', target='clip-1'))

    def test_current_complete_timeline_replays_without_diffusion_units(self):
        plan = self.plan((True, True))
        for i in range(2):
            self.save(plan, index=i)
        result = self.run_preflight(plan)
        self.assertEqual(result['director_render_indices'], set())
        self.assertEqual(set(result['director_cached_takes']), {0, 1})
        self.assertEqual(result['director_cache_report']['effective_mode'], 'cached_replay')

    def test_missing_cached_dependency_never_expands_requested_clip_run(self):
        plan = self.plan((True, True), mode='clip', target='clip-1')
        self.save(plan)
        # Metadata validated, then a file vanished before materialization.
        self.cache.load_active = lambda clip_id: None
        self.env['_DirectorClipCache'] = lambda *args: self.cache
        with self.assertRaisesRegex(RuntimeError, 'disappeared|missing|cache'):
            self.run_preflight(plan)

    def test_setup_prepares_conditioning_for_cache_dependent_expansion_without_refmods(self):
        import textwrap
        source = (ROOT / 'nodes.py').read_text(encoding='utf-8')
        start = source.index('        _director_required_conditioning_indices = None')
        end = source.index('        # Director RefMods are independent', start)
        env = dict(plan=self.plan((True, True, True)),
            multiclip_clips=[{'clip_id': f'clip-{i}'} for i in range(3)],
            director_cfg={}, director_regeneration_cfg={'mode': 'clip', 'clip_id': 'clip-2'})
        exec(compile(textwrap.dedent(source[start:end]), 'Setup.production.conditioning-scope', 'exec'), env)
        self.assertEqual(env['_director_required_conditioning_indices'], {0, 1, 2})

    def test_v063_multiclip_take_append_reuses_both_approved_clips(self):
        original = self.plan((True, True))
        saved = [self.save(original, index=i, legacy_semantic=True) for i in range(2)]
        appended = self.plan((True, True, True))
        result = self.run_preflight(appended)
        self.assertEqual(result['director_render_indices'], {2})
        self.assertEqual(set(result['director_cached_takes']), {0, 1})
        for i, record in enumerate(saved):
            self.assertEqual(self.cache.active_metadata(f'clip-{i}')['revision'], record['revision'])

    def test_production_single_take_append_reuses_approved_prefix(self):
        original = self.plan((True,))
        saved = self.save(original, legacy_single=True)
        appended = self.plan((True, True))
        result = self.run_preflight(appended)
        self.assertEqual(result['director_render_indices'], {1})
        self.assertEqual(set(result['director_cached_takes']), {0})
        self.assertEqual(self.cache.active_metadata('clip-0')['revision'], saved['revision'])

    def test_execution_switch_does_not_change_clip_semantics(self):
        original = self.plan((True, True))
        switched = self.plan((False, True))
        fingerprint = self.env['_lm_director_clip_semantic_fingerprint']
        self.assertEqual(fingerprint(original, 100, 0), fingerprint(switched, 100, 0))

    def test_production_partial_run_samples_only_new_suffix(self):
        plan = self.plan()
        saved = self.save(plan)
        result = self.run_preflight(plan)
        self.assertEqual(result['director_render_indices'], {1, 2})
        self.assertEqual(result['director_clip_only_targets'], set())
        self.assertEqual(set(result['director_cached_takes']), {0})
        self.assertTrue(self.torch.equal(result['director_cached_takes'][0]['display_video'], self.video))
        self.assertEqual(self.cache.active_metadata('clip-0')['revision'], saved['revision'])


if __name__ == '__main__':
    unittest.main()
