# Two-Stage Sampling, Latent Hi-Res and Refiner

LongMedia supports a two-stage H3 workflow inside the **same model lifecycle**. It can either refine the low-noise tail at the original resolution or upscale the denoised video latent and run an independent high-resolution H3 pass.

The important controls are in **MiniMax H3 • Long Media Sampler**:

```text
latent_hires_enabled
latent_hires_model
latent_hires_scale
latent_hires_precision
latent_hires_align

refine_enabled
refine_steps
```

## Recommended topology

```text
Long Media Setup
      │
      ├── positive ──> Basic Guider
      │
      ├── initial_av
      │
      └── long_media_plan
                │
                ▼
        Long Media Sampler
                │
                ▼
        Long Media Decode
```

You do **not** need to load a second MiniMax H3 model for the high-resolution pass.

---

## Mode A — Refiner Only

Settings:

```text
latent_hires_enabled = false
refine_enabled       = true
```

LongMedia splits the connected SIGMAS schedule into two continuous parts.

For `T` denoise steps and `R` refine steps:

```text
stage 1 = the early/base part of the connected schedule
stage 2 = the final R-step low-sigma tail
```

Stage 1 starts from normal random noise.

Stage 2:

- continues from the Stage-1 solver state;
- uses zero added noise;
- keeps the same effective seed;
- stays at the same latent resolution.

This is a **continuous low-noise refinement**, not a second independent generation.

---

## Mode B — Latent Hi-Res Only

Settings:

```text
latent_hires_enabled = true
refine_enabled       = false
```

LongMedia completes the low-resolution H3 pass, takes the **denoised x0**, and sends only the video latent through the learned MiniMax H3 latent upscaler.

The audio latent is preserved exactly.

```text
low-resolution H3 x0
        │
        ├── video latent ──> learned latent upscale
        │
        └── audio latent ──> preserved
        │
        ▼
high-resolution AV latent
```

No second H3 denoise pass is performed.

This is useful when the learned upscaler alone provides enough spatial detail.

---

## Mode C — Latent Hi-Res + Refiner

Settings:

```text
latent_hires_enabled = true
refine_enabled       = true
```

This is the full two-stage high-resolution path.

### Stage 1 — Low-resolution generation

H3 generates the base movie at the original latent resolution.

LongMedia takes the callback's **denoised x0** rather than the noisy in-progress solver state.

For segmented or MultiClip generation, clean low-resolution x0 is also the continuation source between units.

### Learned latent upscale

Only the video latent is spatially upscaled.

```text
video x0 ──> learned MiniMax H3 latent upscaler
audio x0 ──> unchanged
```

`latent_hires_scale` is continuous. A value such as `1.2` means a 1.2× spatial latent upscale; `2.0` means 2×.

The selected model is loaded from:

```text
ComfyUI/models/latent_upscale_models/
```

### Conditioning rebuild

Native H3 VIDEO keyframes are tied to the target spatial grid. After the latent resolution changes, low-resolution target-grid video keyframes cannot be reused directly.

LongMedia therefore rebuilds the runtime conditioning for the new H/W:

- incompatible low-resolution VIDEO keyframes are removed;
- Ref2VA Picture/Video references remain available;
- audio-only guides remain available;
- audio reference timing remains preserved.

### Stage 2 — Independent high-resolution H3 pass

The high-resolution pass is intentionally **not** the low-sigma continuation used by Refiner Only.

It starts from:

- the learned-upscaled denoised x0;
- fresh noise generated from the **same effective seed**;
- an independent scheduler subset selected from the original connected SIGMAS curve.

For a typical 8-step scheduler and `refine_steps=3`, LongMedia selects every-other scheduler points after sigma-max, for example:

```text
original sigma indices: 0 1 2 3 4 5 6 7 8
hi-res pass:              1   3   5       0
```

The exact selected indices are reported in the console as:

```text
[TWO-PASS HIRES]
[HIRES SECOND PASS]
```

This gives H3 enough noise authority to re-author high-frequency structure at the new spatial resolution instead of merely polishing an interpolated latent.

---

## Why x0 Matters

The learned upscaler expects a clean denoised latent.

LongMedia therefore refuses to feed the partial noisy solver state into the learned upscaler. The Stage-1 callback x0 is first converted through the normal H3 latent-output contract and only then upscaled.

This avoids carrying solver noise into spatial detail reconstruction.

---

## Seed Behavior

The integrated high-resolution second pass uses the **same effective seed** as Stage 1.

That does not make Stage 2 identical to Stage 1: the latent resolution, noise tensor, conditioning geometry, and sigma subset are different.

The shared seed keeps the two stages deterministically related.

---

## Suggested Starting Point

A practical general-purpose setup:

```text
BasicScheduler steps       = 8

latent_hires_enabled       = true
latent_hires_scale         = 1.2–2.0
latent_hires_precision     = bf16

refine_enabled             = true
refine_steps               = 2–3

sampler_mode               = auto
memory_mode                = auto
attention_mode             = auto
```

For 16 GB-class GPUs, start with a modest upscale such as `1.2–1.5` before attempting a large 2× pass. High-resolution H3 attention and QKV workspaces grow quickly with spatial token count.

LongMedia's guarded Dynamic VRAM paths remain active, but a smaller target is still faster than proving that the allocator has philosophical depth.

---

## The Optional Explicit Second Sampler in the Example Workflow

The included workflow:

```text
workflows/MiniMax-H3-LongMedia-LatentUpscale-Detailer.json
```

also contains an **explicit second Long Media Sampler node** connected after the first sampler. It is shipped **bypassed by default**.

That node is an advanced chained-sampler option:

```text
Sampler #1
   │
   ▼
Sampler #2 (bypassed by default)
   │
   ▼
Decode
```

Do not confuse this with the integrated `latent_hires_enabled + refine_enabled` path.

- **Integrated two-stage mode** runs the learned upscale/refiner inside one Long Media Sampler and one H3 model lifecycle.
- **Explicit chained Sampler #2** is a separate sampling invocation with its own SIGMAS input and should be enabled only when you deliberately want another diffusion stage.

Avoid enabling a chained Sampler #2 merely because the integrated refiner is enabled. That would add another denoise trajectory rather than "finishing" the existing one.

---

## FastVideo VSA 4-Step Checkpoints

For Kijai FastVideo VSA, the distilled base model owns exactly four transformer calls for **Sampler #1**.

LongMedia does not split those four calls into an inferred 2+2 or similar schedule.

When a workflow-owned refiner is allowed, the second-stage step count comes from `refine_steps` and its sigma selection comes from the workflow-connected SIGMAS schedule.

H3ddle FastH3 Preview v1 has a stricter four-call contract and fails closed when extra latent-hires/refine denoise calls would be off-distribution.

---

## Troubleshooting

### Latent Hi-Res enabled but no upscaler selected

Select a model from `models/latent_upscale_models`.

### High-resolution pass OOMs while the low-resolution pass succeeds

The spatial token count has increased. Keep:

```text
memory_mode = auto
```

and reduce `latent_hires_scale` first. On current native INT8 / Comfy Kitchen paths, LongMedia can also stream impossible full-QKV workloads rather than substituting approximate attention.

### H3 layout mismatch after upscale

Current LongMedia rebuilds the target layout and drops only incompatible target-grid VIDEO keyframes. Ref2VA references and audio guides remain intact.

### Re-running with another seed OOMs before step 1

Current LongMedia creates a sampler execution memory boundary before each run, including seed-only reruns that bypass cached Setup execution.
