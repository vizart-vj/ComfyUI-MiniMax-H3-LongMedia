# 0.4.40 Release Audit

## Release checks

- Package metadata synchronized across `VERSION`, `__init__.py`, `pyproject.toml`, README and release notes.
- `segmented_continuation` is excluded from MultiClip native per-clip VideoVAE decode on both sampler and decoder sides.
- MultiClip final-latent continuation and 22-frame native hidden context are preserved.
- Legacy sectioned-UI positional serialization repair remains runtime-safe and silent in production mode.
- Routine LongMedia diagnostics are suppressed by default; actionable failures remain visible.
- Full diagnostics are available when `release_guard = false`.
- Python runtime sources compile successfully.
- Frontend JavaScript passes syntax validation.
- Release ZIP integrity is verified and bytecode/cache files are excluded.

## Packaging

The distributable contains runtime code, documentation, web extensions, workflow examples, license and third-party notices. Development-only tests and CI files are excluded from the release asset.
