# Примеры workflow

В релизе оставлены ровно три поддерживаемых workflow.

## MiniMax-H3-LongMedia-Director-Unified.json

Основной workflow **LongMedia Director**. Director владеет MAIN / CAMERA / AUDIO timeline, WHO & WHAT references, ролями FIRST/LAST/REF, resolution policy, Program Monitor и TAKE workspace.

```text
MiniMax H3 • LongMedia Director
        ↓ director
MiniMax H3 • Long Media Setup
                 ↓
Long Media Sampler
                 ↓
Long Media Decode
```

См. `docs/DIRECTOR_GUIDE_RU.md` и `docs/DIRECTOR_REGENERATION_GUIDE_RU.md`.

## MiniMax-H3-LongMedia-LatentUpscale-Detailer.json

Production graph для learned Latent Hi-Res и Refine. Integrated Refine работает как внутренний Stage 2 той же Long Media Sampler.

При включённом Refine появляется **windowed refine**:

- ON — LongMedia может разбивать длинный/high-resolution Stage 2 на bounded temporal windows.
- OFF — Stage 2 рефайнит полный latent одним монолитным проходом. Это даёт максимальную temporal/lighting continuity, но может потребовать заметно больше VRAM.

См. `docs/TWO_PASS_LATENT_HIRES_REFINER_GUIDE_RU.md`.

## MiniMax-H3-LongMedia-SAFE-1080p-15s.json

Консервативный production-пример 1080p / 15 секунд.

## Рекомендуемые runtime defaults

```text
sampler_mode   = auto
memory_mode    = auto
attention_mode = auto
```

ComfyUI Dynamic VRAM оставляйте включённым.

## Media placeholders

Release workflow не содержат пользовательских media. Loader selections используют neutral `REPLACE_ME_*`, а saved local preview/output metadata удалены. После загрузки выберите собственные image/video/audio files.

## Dependencies

Обязательно:

- **ComfyUI-MiniMax-H3-LongMedia**

В сохранённых примерах могут встречаться optional utility nodes из ComfyUI-KJNodes, ComfyUI-VideoHelperSuite или WAS Node Suite. Это зависимости конкретного workflow, а не runtime dependencies LongMedia.
