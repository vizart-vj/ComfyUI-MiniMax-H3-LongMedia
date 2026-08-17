# 0.4.0 Release Audit

## Scope

Baseline: `0.3.115-dev-planner-bidirectional-resize`, promoted to stable `0.4.0`.

## Release cleanup

- Removed all per-build `BUILD_NOTES_v0.3.x*` files from the distributable tree.
- Removed the pre-release hot-reload module and its runtime dispatch hooks.
- Removed the unused development-only `vram_tracker.py` helper.
- Removed version-named root regression scripts; maintained release regression coverage under `tools/` and release audit commands.
- Promoted `VERSION`, package `__version__`, and `pyproject.toml` to `0.4.0`.
- Replaced stale historical README material with release-facing documentation.

## Public architecture reviewed

- Setup/Planner ownership: Planner is authoritative only in `multiclip`.
- Fixed segmentation and MultiClip share the unified clip executor.
- Obsolete manual reference-strength controls are absent from the public and backend schema.
- Lip-sync uses the source `audio_1` policy and no longer exposes a user strength control.
- Release guard changes logging only; it does not disable memory/OOM safety.
- LongMediaPlan does not retain text-encoder/model-patcher objects after Setup.

## Memory safety reviewed

- Auto memory profile resolves from model/GPU ratio.
- Geometry-aware Governor V4 prevents very long sequences from being classified from free-VRAM alone.
- Unsafe full-sequence Sage attention can be rejected before QKV allocation.
- Bounded streamed Sol QKV path is available for long constrained sequences.
- Optimized MLP path retains a numerical parity gate and stock fallback.

## UI reviewed

- Planner workflow ownership is explicit.
- Planner vertical resize is bidirectional after manual expansion.
- Removed strength controls do not remain in frontend widget lists/defaults.

## Release verification

The final build is expected to pass:

```text
python -m py_compile *.py sol_kernel/*.py
python tools/release_schema_audit.py
python tools/test_continuity_policy.py
python tools/test_refine_sigma_split.py
python tools/test_single_lifecycle_refine.py
python tools/test_single_step_latent.py
python tools/test_first_handoff_bridge.py
python tools/test_lip_sync_audio_mode.py
python tools/test_lip_sync_extender_av_context.py
python tools/test_resolution_safety.py
node tools/test_frontend_ui.mjs
```

The audit is static/regression validation. Full GPU validation still depends on the installed ComfyUI, H3 checkpoint, quantization backend and custom acceleration patches.

## Final verification result

Release candidate verification completed successfully before packaging:

```text
py_compile: PASS
release_schema_audit: PASS
continuity_policy: PASS
refine_sigma_split: PASS
single_lifecycle_refine: PASS
single_step_latent: PASS
first_handoff_bridge: PASS
lip_sync_audio_mode: PASS
lip_sync_authoritative_source_clock: PASS
resolution_safety: PASS
frontend_ui: PASS
oom_governor_static: PASS
unified_clip_engine_static: PASS
streamed_sol_scope_static: PASS
locked_target_audio_static: PASS
relative_import_audit: PASS
artifact_hygiene: PASS
```

Release schema at packaging time:

```text
Planner: 1 required, 0 optional
Setup: 16 required, 19 optional
Sampler: 34 required, 0 optional
Decode: 8 required, 0 optional
```

No Python bytecode caches are included in the release package.
