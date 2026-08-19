import ast
from pathlib import Path

src = Path(__file__).resolve().parents[1] / 'nodes.py'
mod = ast.parse(src.read_text(encoding='utf-8'))
fn = next(n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == '_v80_native_av_context_frames')
mini = ast.Module(body=[fn], type_ignores=[])
ns = {}
exec(compile(mini, str(src), 'exec'), ns)
f = ns['_v80_native_av_context_frames']

assert f(120, 22, 1, True) == 56
assert f(50, 22, 1, True) == 39
assert f(30, 22, 1, True) == 22
assert f(120, 22, 2, True) == 22
assert f(120, 22, 1, False) == 22
assert f(20, 22, 1, True) == 5
print('FIRST_HANDOFF_BRIDGE_REGRESSION: PASS')
