from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nodes = (ROOT / 'nodes.py').read_text(encoding='utf-8')
plan = (ROOT / 'media_plan.py').read_text(encoding='utf-8')
assert 'release_guard' not in nodes
assert 'release_guard' not in plan
assert '_set_longmedia_release_guard' not in nodes
assert nodes.count('builtins.print(') == 1, 'release runtime contains direct console prints outside _lm_print'
for name in ('hybrid_payload_patch.py', 'lora_compat.py', 'motion_context_layout_patch.py'):
    text = (ROOT / name).read_text(encoding='utf-8')
    assert '_LOG.info(' not in text, f'{name}: routine INFO logging remains'
print('RELEASE_CONSOLE_STATIC: PASS')
