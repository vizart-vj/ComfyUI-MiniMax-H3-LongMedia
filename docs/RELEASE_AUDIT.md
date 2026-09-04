# 0.5.40 Release Audit

## Baseline

Public comparison baseline: GitHub release/tag `v0.4.40`.

0.5.40 consolidates the later development work while preserving legacy workflow loading.

## Documentation Audit

Current user-facing guides were checked for stale production instructions.

Updated:

- root `README.md`;
- `docs/README.md`;
- `docs/ARCHITECTURE.md`;
- `docs/SAMPLER_OPTIMIZATION.md`;
- `docs/PROMPTING_MULTICLIP.md`;
- `docs/PROMPTING_SEGMENTATION.md`;
- `workflows/README.md`.

Added:

- `docs/MODES_GUIDE.md`;
- `docs/TWO_PASS_LATENT_HIRES_REFINER_GUIDE.md`;
- `docs/RELEASE_NOTES_0.5.40.md`;
- `docs/GITHUB_RELEASE_0.5.40.md`.

Historical 0.4.30/0.4.40 release notes remain intentionally unchanged as historical records.

## Contract Checks

- `duration_source` remains independent from `audio_mode` and H3 reference conditioning.
- `video_ref_edit + auto` defaults timeline ownership to Video1.
- preserve-style source AV and arbitrary `lip_sync` redub remain different contracts.
- Audio2/Audio3 remain prompt-addressable references and are not mixed into `video_ref_edit` passthrough output; Audio1 is the sole source/output soundtrack authority.
- Planner owns `clip_plan` only for `timeline_mode=multiclip`.
- Cameras keep stable Planner `clip_id` ownership.
- Latent Hi-Res uses denoised x0 and preserves audio latent.
- Hi-Res refiner is an independent same-seed fresh-noise H3 pass.
- Refiner-only mode remains a continuous zero-noise low-sigma tail.
- seed-only reruns keep the sampler execution memory boundary.
- constrained native INT8 keeps guarded one-block residency and RAM-pressure-aware pinning.

## Package Identity

- GitHub owner/repository: `vizart-vj/ComfyUI-MiniMax-H3-LongMedia`.
- Comfy Registry `PublisherId`: `noise`.
- Release version is synchronized across `VERSION`, `__init__.py`, and `pyproject.toml`.

## Workflow Hygiene

The Latent Upscale / Detailer example is sanitized before release:

- personal/input image names replaced;
- personal/input audio names replaced;
- source video name replaced;
- saved input/output video preview state removed;
- local Windows output paths removed;
- stale embedded 0.4.x Markdown notes replaced with current documentation text.

## Verification

Release packaging must pass:

- Python compile / AST validation;
- JavaScript syntax validation;
- JSON parse for all workflows;
- documentation link validation;
- stale-current-doc terminology scan;
- media/path sanitization scan on release workflows;
- archive root check;
- version/PublisherId/repository identity check;
- no `__pycache__` / `.pyc` in the archive.
