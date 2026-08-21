from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'nodes.py').read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)
WANTED = {
    '_v043_parse_multiclip_prompt_text',
    '_v043_import_multiclip_prompts',
    '_v043_join_global_local_prompt',
}
selected = []
for node in TREE.body:
    if isinstance(node, ast.Assign):
        names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if '_V043_MULTICLIP_SECTION_RE' in names:
            selected.append(node)
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in WANTED:
        selected.append(node)
module = ast.Module(body=selected, type_ignores=[])
ns = {'re': re}
exec(compile(ast.fix_missing_locations(module), str(ROOT / 'nodes.py'), 'exec'), ns)
parse = ns['_v043_parse_multiclip_prompt_text']
imp = ns['_v043_import_multiclip_prompts']
join = ns['_v043_join_global_local_prompt']

assert parse('clip_1:\nA\n\nclip_2:\nB') == ('A', 'B')
assert parse('### shot-1:\nA\nshot 2:\nB\nshot_3:\nC') == ('A', 'B', 'C')
assert parse('one ordinary prompt only') == tuple()
try:
    parse('clip_1:\nA\nclip_3:\nC')
except ValueError as exc:
    assert 'missing' in str(exc).lower()
else:
    raise AssertionError('Non-contiguous sections must be rejected')

clips = (
    {'prompt': 'old 1', 'duration': 5.0, 'seed': 111},
    {'prompt': 'old 2', 'duration': 8.0, 'seed': None},
)
updated, imported = imp(clips, 'clip_1:\nnew 1\nclip_2:\nnew 2\nclip_3:\nnew 3')
assert imported is True
assert [c['prompt'] for c in updated] == ['new 1', 'new 2', 'new 3']
assert updated[0]['duration'] == 5.0 and updated[0]['seed'] == 111
assert updated[1]['duration'] == 8.0 and updated[1]['seed'] is None
assert updated[2]['duration'] == 8.0 and updated[2]['seed'] is None
assert join('GLOBAL', 'LOCAL') == 'GLOBAL\n\nLOCAL'
assert join('GLOBAL', '') == 'GLOBAL'
assert join('', 'LOCAL') == 'LOCAL'

# 0.4.30 ownership: prompt authoring lives in Planner, not Setup.
planner_pos = SOURCE.index('class MiniMaxH3LongMediaPlanner:')
setup_pos = SOURCE.index('class MiniMaxH3LatentLabLongMediaSetup:')
planner_source = SOURCE[planner_pos:setup_pos]
setup_head = SOURCE[setup_pos:SOURCE.index('    def setup(', setup_pos)]
for token in ("'global_prompt': ('STRING'", "'multiclip_prompt': ('STRING'", "'multiclip_auto_import': ('BOOLEAN'"):
    assert token in planner_source, token
    assert token not in setup_head, token
assert "send_sync('minimax_h3_planner_prompt_import'" in planner_source
assert "'global_prompt': str(global_prompt or '').strip()" in planner_source

ui = (ROOT / 'web' / 'longmedia_planner.js').read_text(encoding='utf-8')
for token in ('Global Prompt', 'Multiple Clips Prompt', 'Auto Import Prompt', 'Import Prompt', 'minimax_h3_planner_prompt_import'):
    assert token in ui, token
assert 'parseStructuredPrompt' in ui
assert 'importPrompts' in ui
print('PASS: 0.4.30 Planner-owned structured prompt import')

# 0.4.30 hotfix: manual Import must be independent from Auto Import and must have
# a one-shot backend path for dynamic connected STRING outputs.
js = (ROOT / 'web' / 'longmedia_planner.js').read_text(encoding='utf-8')
nodes_text = (ROOT / 'nodes.py').read_text(encoding='utf-8')
assert 'auto.value = true' not in js, 'Import Prompt must never enable Auto Import'
assert 'multiclip_import_request' in js
assert 'connectedTextValue' in js
assert 'import queued — run workflow once' in js
assert "'multiclip_import_request': ('BOOLEAN'" in nodes_text
assert 'manual_import = bool(multiclip_import_request)' in nodes_text

assert 'multiclip_last_import_source' in js
assert 'markImportedSource' in js
assert "'multiclip_last_import_source': ('STRING'" in nodes_text
assert "source_changed = import_source != str(multiclip_last_import_source or '')" in nodes_text
