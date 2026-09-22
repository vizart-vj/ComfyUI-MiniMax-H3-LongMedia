# ComfyUI-MiniMax-H3-LongMedia

Production-oriented ComfyUI nodes для **MiniMax H3**: long-form video/audio generation, native reference editing, Director timeline authoring, MultiClip planning, camera direction, segmentation, lip-sync/redubbing, latent hi-res refine, reconstruction и adaptive low-VRAM execution.

![screenshot](ex.png)

**Текущий релиз: 0.6.54.**

## Что изменилось после публичного v0.6.50

- **Настоящая selective regeneration:** clip-only rerender семплирует только выбранный GENERATED clip, когда cached seam contracts остаются валидными; From Here пересчитывает только зависимый suffix.
- **Immutable MEDIA blocks:** импортированное видео может лежать прямо на BASE timeline и не участвовать в diffusion.
- **Native continuation:** GENERATED parent использует сохранённый TAKE latent state; для MEDIA кодируется только нужный tail и прикрепляется как native frame-0 H3 keyframe.
- **O(1)-style append:** добавление следующего клипа использует уже сохранённый TAKE/cache state и не пересобирает весь timeline.
- **Quantized AUTO routing:** INT8/W4A8/NVFP4 policy использует реальный packed storage и activation headroom вместо BF16-equivalent logical size.
- **Windows/AIMDO:** большие checkpoints остаются file-backed через ComfyUI `ModelMMAP` / `TensorFileSlice`; eager full-model RAM load не используется.
- **Mixed AV assembly:** исходный MEDIA остаётся вне VideoVAE; mono/stereo timeline audio собирается по явному channel contract.

См. [Release Notes 0.6.54](docs/RELEASE_NOTES_0.6.54_RU.md).

## Документация

- [Русский индекс документации](docs/README_RU.md)
- [English documentation index](docs/README_EN.md)
- [LongMedia Director — полное руководство](docs/DIRECTOR_GUIDE_RU.md)
- [Режимы LongMedia](docs/MODES_GUIDE_RU.md)
- [Режимы аудио и `video_ref_edit`](docs/AUDIO_MODES_GUIDE_RU.md)
- [Integrated Refine и Latent Hi-Res](docs/TWO_PASS_LATENT_HIRES_REFINER_GUIDE_RU.md)
- [Sampler, VRAM и производительность](docs/SAMPLER_OPTIMIZATION_RU.md)

## Основные ноды

- **MiniMax H3 • Long Media Setup**
- **MiniMax H3 • Long Media Planner**
- **MiniMax H3 • LongMedia Director**
- **MiniMax H3 • Long Media Cameras**
- **MiniMax H3 • Long Media Sampler**
- **MiniMax H3 • Long Media Decode**
- **MiniMax H3 • Long Media Video Reconstructor**

Legacy `MiniMaxH3LatentLab...` class identifiers остаются зарегистрированными для совместимости старых workflow.

## Установка

Установите в:

```text
ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-LongMedia
```

или через Comfy Registry.

Идентичность package:

```text
GitHub:             vizart-vj/ComfyUI-MiniMax-H3-LongMedia
Comfy PublisherId:  noise
```

После установки/обновления перезапустите ComfyUI.

## Текущая semantic model Setup

Новые workflow используют независимые controls:

```text
control_mode
h3_mode
timeline_mode
duration_source
audio_mode
motion_repair
```

H3 conditioning families:

```text
t2va
fl2va
ref2va
hybrid
video_ref_edit
```

Timeline modes:

```text
single
segmented
multiclip
```

Duration ownership:

```text
auto
video
audio
manual
longest_input
```

См. [Режимы LongMedia](docs/MODES_GUIDE_RU.md).

## LongMedia Director 0.6.54

Director — unified authoring surface для MAIN timing, prompts, WHO & WHAT media, ролей REF/FIRST/LAST, cameras, temporal embeddings, audio policy, resolution policy, Program Monitor и TAKE management.

```text
LongMedia Director
        ↓ director
Long Media Setup  (control_mode=director)
```

Текущее хранилище TAKE — TAKE-centric. CREATE создаёт активный пустой workspace; следующий успешный render заполняет тот же TAKE. Restore атомарно возвращает полный saved Director state.

Начиная с 0.6.54 продолжение формально разделяет `GENERATED` и `MEDIA`: `GENERATED → GENERATED` использует сохранённый native TAKE latent, а `MEDIA → GENERATED` строит свежий H3 target и закрепляет только VAE-encoded tail внешнего видео как native frame-0 `minimax_keyframes` guide. Полный импортированный ролик не становится diffusion target и не проходит full-video VAE roundtrip.

См. [полное руководство Director](docs/DIRECTOR_GUIDE_RU.md).

## Integrated Refine / Latent Hi-Res

Одна Long Media Sampler содержит MAIN stage и optional internal Stage 2, включаемый `refine_enabled`. При включённом Refine `windowed_refine` можно выключить, чтобы принудительно выполнить единый full-latent Stage 2 pass для максимальной temporal/lighting continuity.

```text
Stage 1 MAIN x0
    ↓
(optional) learned video-latent upscale
    ↓
Stage 2 Refine using Refine Sigmas
    ↓
final video + exact Stage-1 audio
```

Длинный/high-resolution Stage 2 может работать bounded temporal windows при включённом `windowed_refine`. Refine video-authoritative: Stage-1 audio сохраняется точно и не пересэмпливается как audio refiner.

См. [Integrated Refine и Latent Hi-Res](docs/TWO_PASS_LATENT_HIRES_REFINER_GUIDE_RU.md).

## Sampler / memory

Рекомендуемый старт:

```text
sampler_mode   = auto
memory_mode    = auto
attention_mode = auto
```

Оставляйте ComfyUI Dynamic VRAM включённым. Runtime включает очистку transient CUDA references между повторными Queue, packed-storage-aware quantized AUTO routing, guarded native INT8 residency, exact Existing/Kitchen fallback paths, file-backed Windows loading больших safetensors и безопасный offload Latent Hi-Res model. Явные manual Sampler значения сохраняются в NORMAL/AUTO-normal профилях.

См. [Sampler, VRAM и производительность](docs/SAMPLER_OPTIMIZATION_RU.md).
