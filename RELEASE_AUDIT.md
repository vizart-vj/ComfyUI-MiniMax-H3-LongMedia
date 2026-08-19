# 0.4.2 Release Audit

Release base: validated public `0.4.11` runtime plus the current MultiClip/native-decode and compatibility fixes.

## Release checks

- package metadata synchronized to `0.4.2`;
- Long Media Setup schema audited against Python call signatures;
- `release_guard` removed from runtime plan/report and no longer exists as a workflow policy;
- routine LongMedia console diagnostics suppressed in release mode;
- MultiClip native continuous latent assembly validated on the H3 temporal grid;
- modulation-row compatibility checked for chunked MLP and final output heads;
- continuity, first-handoff, lip-sync, refiner, resolution and frontend regression tests run;
- Python sources compile cleanly;
- release ZIP integrity verified.

## Packaging

The distributable contains release runtime code, documentation, web extensions, tests, workflow examples and third-party notices. Python bytecode/cache files are excluded.
