# Integrated Refine и Latent Hi-Res

Начиная с **0.6.24**, `Refine` в LongMedia является независимым post-x0 этапом внутри той же ноды **MiniMax H3 • Long Media Sampler**. MAIN sampler не сокращается, не делится и не заменяется.

## Входы Sampler

```text
sigmas          -> MAIN generation schedule
refine_sigmas   -> REFINE schedule (отдельный SIGMAS socket)

latent_hires_enabled
latent_hires_model
latent_hires_scale
latent_hires_precision
latent_hires_align

refine_enabled
windowed_refine
```

`refine_steps`, `refine_add_noise`, `refine_seed` сохраняются только как legacy serialized fields для старых workflow. Они скрыты и игнорируются. Число refine steps равно `len(refine_sigmas) - 1`.

## Контракт исполнения

```text
MAIN sigmas
    ↓
complete MAIN x0
    ↓
(optional) learned H3 latent upscale — только video
    ↓
post-x0 Refine using refine_sigmas
    ↓
exact Stage-1 audio restore
```

Включение Refine не меняет MAIN sigma schedule. Для MultiClip это принципиально: существующий segment/continuation planner остаётся авторитетным и генерирует MAIN так же, как при выключенном Refine.

## Внутренний Sampler #2

Логически в одной Long Media Sampler node находятся два последовательных stage:

```text
Stage 1 = MAIN sampler
Stage 2 = Refine sampler
```

`refine_enabled` действует как bypass/unbypass Stage 2. Stage 2 получает результат Stage 1 и использует только `refine_sigmas`.

На коротком/достаточно маленьком latent Stage 2 работает монолитно. При **windowed refine = ON** длинный/high-resolution latent может переходить в bounded temporal windows для экономии VRAM. При **windowed refine = OFF** Stage 2 работает только монолитно и рефайнит полный latent одним проходом.

`windowed_refine` показывается только при включённом Refine и по умолчанию включён ради memory-safe поведения. OFF — continuity-first режим для сцен, где между соседними refine windows заметны скачки света, framing или положения изображения. Exact-only режим намеренно не откатывается обратно на windows после OOM.

## Temporal-windowed Refine

При **windowed refine = ON** длинный/high-resolution x0 не принуждается к одному гигантскому H3 refine pass. Video refine разбивается на окна, согласованные с нативной temporal lattice H3 `(1,4,4,4,4)`.

Для 16 GB-класса стартовый безопасный размер соответствует примерно:

```text
chunk ≈ 73 decoded frames = 22 H3 temporal tokens
```

Planner сохраняет правильную H3 temporal phase. Если окно всё ещё вызывает OOM, LongMedia уменьшает окно до меньшего legal H3 размера вместо изменения MAIN generation или attention math.

## Continuity между окнами

Window split — это implementation detail Refine, **не** LongMedia segmentation и не новый Director clip.

Окна используют контекст соседних областей и пишут только предназначенный им refined core. Цель — не смешивать две независимые абсолютные геометрии через latent crossfade и одновременно не обеднять полезный H3-refine до одного high-frequency residual.

Глобальная temporal position исходного сегмента сохраняется для каждого окна.

## Director / embeddings

Camera и embedding ranges проецируются из глобальных часов segment в локальные часы refine window. Text-row ownership не меняется.

Поэтому начало нового refine window само по себе не должно перезапускать Director camera или embedding range.

## Audio / lip-sync

Refine является **video-authoritative**:

```text
video: refine with refine_sigmas
audio: Stage-1 exact passthrough
```

Stage 2 не является audio refiner:

- audio noise = 0;
- audio denoise mask = 0;
- refined audio output Stage 2 не становится новым авторитетным soundtrack;
- финальный audio latent восстанавливается точно из Stage 1.

При Latent Hi-Res callback/x0 используется для video path; он не должен подменять финальное audio Stage 1.

Для lip-sync forced target audio остаётся тем же авторитетным clock. Рекомендуемое значение MiniMax-H3: `audio_strength=1.0`.

## Latent Hi-Res

При `latent_hires_enabled=true` LongMedia берёт полный denoised MAIN x0, пропускает **только 24-channel video latent** через learned H3 latent upscaler и затем запускает Refine на новом H/W.

```text
low-res clean video x0
    ↓
learned latent upscaler
    ↓
high-res video x0
    ↓
H3 Refine (windowed или full-latent, задаётся windowed refine)
```

Audio через latent upscaler не проходит.

Cached upscaler возвращается на CPU через `try/finally`, включая error/OOM/cancel paths, чтобы повторные Queue не удерживали лишний VRAM.

## Topology workflow

Нужна только одна Long Media Sampler node:

```text
Long Media Setup
       ↓
Long Media Sampler
   MAIN Sigmas ────────┐
   Refine Sigmas ──────┤
                       ↓
                 final AV latent
                       ↓
                 Long Media Decode
```

Старый chained second-Long-Media-Sampler handoff сохраняется только ради backward compatibility и специализированных reconstruction workflows. Для обычного Latent Hi-Res/Refine рекомендуется внутренний Stage 2 одной Sampler node.
