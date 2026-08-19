from pathlib import Path

src = (Path(__file__).resolve().parents[1] / 'nodes.py').read_text(encoding='utf-8')
checks = [
    'def _row_mod_params(row, start, stop, local=0, end=None):',
    "'[0.4.20 FINAL-HEAD MOD-ROW COMPAT] unsupported modulation-row layout: '",
    'scale.index_select(0, ids)',
    'shift.index_select(0, ids)',
    'sc, sh = _row_mod_params(row, start, stop, local, end)',
]
for needle in checks:
    assert needle in src, f'missing mod-row compatibility marker: {needle}'
for bad in ('scale[int(row)]', 'shift[int(row)]', 'scale_msa[int(row)]', 'gate_msa[int(row)]'):
    assert bad not in src, f'unsafe scalar row coercion remains: {bad}'
print('MOD_ROW_COMPAT_STATIC: PASS')
