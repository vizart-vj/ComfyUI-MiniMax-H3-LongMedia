# LongMedia Director — Selective Regeneration и TAKE Workflow

Руководство описывает текущую модель regeneration в Director 0.6.54.

## TAKE — сохранённая ревизия

Хранилище Director остаётся TAKE-centric. TAKE содержит полный authored Director state плюс совместимые clip-level cache/continuation данные. `CREATE TAKE` создаёт активную пустую ревизию; следующий успешный render заполняет именно её. Restore атомарно возвращает MAIN, prompts, WHO & WHAT media, FIRST/LAST/REF roles, CAMERA/AUDIO/EMBEDDING tracks, resolution policy и режимы Director.

## BASE types участвуют по-разному

```text
GENERATED = H3 render unit + TAKE/cache + native continuation state
MEDIA     = immutable external video; никогда не H3 render unit
```

MEDIA может быть continuation parent для следующего GENERATED clip, но никогда не семплируется само.

## Правила Selective MultiClip

### Regenerate Clip

Обычная generated chain:

```text
Clip1 -> Clip2 -> Clip3
Regenerate Clip3

Clip1 = TAKE HIT
Clip2 = TAKE HIT
Clip3 = SAMPLE
```

Если incoming/outgoing seam contracts валидны, clip-only regeneration семплирует ровно один target. Если безопасный two-sided lock невозможен, backend расширяет dependency range вместо использования несовместимого cached suffix.

MEDIA chain:

```text
Video MEDIA -> Clip1 -> Clip2
Regenerate Clip2

Video = untouched
Clip1 = TAKE HIT
Clip2 = SAMPLE
```

Regenerate Clip1 оставляет Video untouched и семплирует только Clip1. Clip2 становится stale, но автоматически не запускается.

### Regenerate From Here

Сохраняет валидированный prefix и семплирует выбранный GENERATED clip плюс все зависимые GENERATED clips справа. MEDIA остаются immutable и никогда не попадают в `sampled_indices`.

## Добавление нового clip

Append после cached GENERATED parent использует сохранённый native TAKE tail/motion/audio continuation и семплирует только новый clip. Append после MEDIA создаёт fresh child target, кодирует только external overlap tail и прикрепляет его как native frame-0 H3 keyframe.

Главный runtime invariant:

```text
one new GENERATED clip = one H3 sampling target
```

## External MEDIA cache

Cache key включает source identity, editorial trim range, geometry и overlap. Tail latents хранятся на CPU. Повторное использование той же неизменённой MEDIA boundary поэтому не требует повторного VideoVAE/audio-tail encode.

Авторитетен именно видимый MEDIA range: trimmed/reused source продолжается от конца BASE block, а не от физического конца исходного файла.

## Final assembly

- GENERATED runs декодируются через H3 VideoVAE.
- MEDIA RGB/audio вставляется в финальный timeline напрямую.
- Hidden overlap для MEDIA continuation вырезается из видимого child output.
- Исходный MEDIA audio не регенерируется.
- Mixed mono/stereo pieces нормализуются к `[1,C,L]`; mono точно дублируется, когда нужен stereo output.

## Refine / Latent Hi-Res

Selective regeneration не меняет Refine contract. Stage 2 остаётся video-only; Stage-1 audio сохраняется как exact passthrough.

См. также:

- [Director — полное руководство](DIRECTOR_GUIDE_RU.md)
- [Архитектура](ARCHITECTURE_RU.md)
- [Sampler, VRAM и производительность](SAMPLER_OPTIMIZATION_RU.md)
