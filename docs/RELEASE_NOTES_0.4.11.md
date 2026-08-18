# ComfyUI-MiniMax-H3-LongMedia 0.4.11

0.4.11 is the consolidated public release after 0.4.1. It keeps the validated 0.4.0/0.4.1 feature surface and folds in the runtime fixes developed afterward.

## Runtime fixes

- Unified H3 model lifecycle for sequential LongMedia clip execution.
- Safe ComfyUI-native cloning of `model_options`; no Python deep-copy of AIMDO/CUDA storage.
- Strict segmentation ownership: only `manual` and `segmented_continuation` can use `segment_duration`, overlap, hidden continuation segments, or segmentation stitching.
- MultiClip remains sequential multi-clip generation but is explicitly not segmentation.
- Real two-stage refiner restored inside the already-open H3 lifecycle.
- AIMDO TE/reference pinned-memory gate restored before native reference conditioning.

## Refiner contract

For a connected schedule of `N` intervals and `R` refine steps:

- base stage executes the first `N-R` intervals;
- refine stage executes the final `R` intervals;
- refine adds no fresh noise;
- seed, sampler, guider wrappers and model lifecycle remain the same.

## Cleanup

Removed obsolete experimental refiner tests and unreleased intermediate release-note files. Package metadata is synchronized to `0.4.11`.
