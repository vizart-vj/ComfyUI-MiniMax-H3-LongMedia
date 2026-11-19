# LongMedia Director — Selective Regeneration и TAKE Workflow

Руководство описывает модель regeneration в Director версии LongMedia 0.6.69.

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

### Перегенерация CUT-диапазона

Range меняет интервал внутри одного GENERATED MAIN-клипа. Нужны режим MultiClip в Director, активный совместимый TAKE клипа и выключенный Loop Closure. Длительность и геометрия должны совпадать с сохранённым TAKE.

1. Добавьте хотя бы один CUT-маркер внутри клипа. CUT отмечает границу интервала и не разрезает клип, как Knife.
2. Откройте контекстное меню клипа внутри интервала между двумя маркерами либо между маркером и краем клипа. Выберите **Regenerate … · keep source audio** для сохранения аудио или **Regenerate … · regenerate audio** для его перегенерации.
3. Проверьте окно подтверждения: запрошенный интервал расширяется до границ сетки H3, и фактический диапазон показывается до запуска. Подтверждение ставит задачу в очередь.

Источником служит латент активного TAKE; заблокированная часть латента вне диапазона сохраняется. В режиме сохранения аудио сохраняется аудиолатент всего клипа. При несовместимом TAKE, сетке или необходимых границах операция останавливается и не превращается молча в полный рендер.

### Двусторонний Bridge

Bridge создаёт GENERATED-переход между двумя соседними GENERATED-клипами. Нужны MultiClip, выключенный Loop Closure, совместимые активные TAKE обоих соседей и остальных GENERATED-клипов, которые нужно воспроизвести, а также соприкасающиеся границы MAIN. MEDIA в этом режиме не может служить одним из двух native-родителей моста.

1. Выберите границу двух соприкасающихся клипов и в контекстном меню MAIN нажмите **Insert bridge at selected / nearest boundary**. Появится новый клип; последующие MAIN-клипы сдвинутся вправо. При включённом Ripple сдвинутся и последующие блоки камер, аудио и других слоёв. Блок, пересекающий границу вставки, сначала нужно переместить или разделить.
2. В контекстном меню нового клипа выберите **Render two-sided bridge from neighboring Takes**. Если длительность не совпадает с сеткой H3, меню предложит доступные ближайшие меньшую и большую длительности; изменение также сдвигает последующие MAIN-клипы.
3. Проверьте ожидающий TAKE моста в Program Monitor. **ACCEPT** повторно проверит исходные ревизии, таймлайн, Global Prompt и обе AV-границы перед активацией. **DISCARD** удалит кандидат без его активации.

Контракты Range и Bridge покрыты регрессионными тестами; полный сквозной GPU-прогон всех вариантов этих операций для данного релиза не подтверждён.

## Partial RUN: frozen TAKE + новый хвост (0.6.64)

Отключите участие готовых GENERATED клипов в RUN, оставьте новые клипы включёнными. Старый TAKE нужен только для replay отключённого клипа и для target перед отключённым GENERATED соседом. Для новых соседних targets и нового хвоста старые TAKE не нужны. Отключённый готовый single-shot клип остаётся frozen при переходе на MultiClip даже с новым seed; его duration/geometry должны совпадать с TAKE.

При всех включённых клипах ordinary RUN остаётся semantic incremental; изменение seed/prompt/refs может инвалидировать префикс. Для automatic append сохраняйте фиксированный seed. Сбой cache read/initialization теперь останавливает запуск, а не включает неявный full render.

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
