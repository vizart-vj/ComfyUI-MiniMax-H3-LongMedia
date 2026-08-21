from pathlib import Path

root = Path(__file__).resolve().parents[1]
text = (root / 'nodes.py').read_text(encoding='utf-8')

assert 'def _run_stock_sample_with_reused_lifecycle' in text
assert 'return guider.sample(' in text, 'Unified runtime must enter stock CFGGuider.sample()'
assert 'types.MethodType(_outer_sample_reuse, guider)' in text
assert 'def _run_inner_sample_with_outer_wrappers' not in text, 'Old manual OUTER_SAMPLE shim must be removed'
assert 'original_outer_sample_wrappers' not in text, 'Old wrapper snapshot path must be removed'
assert 'cfg_guider_sample=True; outer_sample_lifecycle=reused' in text
assert 'prepare_sampling(self.model_patcher' not in text[text.index('def _outer_sample_reuse'):text.index('guider.outer_sample = types.MethodType')], 'Reuse terminal must not reopen prepare_sampling'
assert '.pre_run()' not in text[text.index('def _outer_sample_reuse'):text.index('guider.outer_sample = types.MethodType')], 'Reuse terminal must not rerun pre_run'
assert '.cleanup()' not in text[text.index('def _outer_sample_reuse'):text.index('guider.outer_sample = types.MethodType')], 'Reuse terminal must not cleanup model per pass'
print('PASS: stock CFGGuider.sample extension contract preserved with lifecycle-reuse outer_sample')
