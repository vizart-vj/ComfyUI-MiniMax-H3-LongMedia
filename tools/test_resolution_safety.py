import ast, math
from pathlib import Path

src = (Path(__file__).resolve().parents[1] / "nodes.py").read_text()
tree = ast.parse(src)
keep = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(getattr(t, "id", "") in {"CANVAS_MULTIPLE", "_H3_SAFE_REF_PIXELS"} for t in node.targets):
        keep.append(node)
    if isinstance(node, ast.FunctionDef) and node.name in {"_h3_safe_dim", "_h3_safe_target_canvas"}:
        keep.append(node)
mod = ast.Module(body=keep, type_ignores=[])
ns={"math": math}
exec(compile(mod, "nodes.py", "exec"), ns)
assert ns["_h3_safe_target_canvas"](1344,768)[:2] == (1344,768)
assert ns["_h3_safe_target_canvas"](1365,768)[:2] == (1376,768)
assert ns["_h3_safe_target_canvas"](1056,592)[:2] == (1056,576)
for w,h in [(1376,768),(1056,576),(640,960)]:
    assert w % 32 == 0 and h % 32 == 0
    assert (w//16) % 2 == 0 and (h//16) % 2 == 0
print("RESOLUTION_SAFETY_REGRESSION: PASS")
