# Long Media MultiClip — краткое руководство

**MultiClip** создаёт одну длинную sequence из нескольких независимо управляемых clips.

У каждого Planner clip есть:

- Name;
- Prompt;
- Duration;
- optional Seed.

LongMedia управляет temporal continuation и handoff между соседними clips.

## Базовое подключение

```text
Long Media Planner
        ↓ clip_plan
Long Media Setup
```

С отдельным camera control:

```text
Long Media Planner
        ↓ clip_plan
Long Media Cameras
        ↓ clip_plan
Long Media Setup
```

В Setup выберите:

```text
Timeline Mode: multiclip
```

Подключённый Planner становится авторитетным только при `timeline=multiclip`.

## Global Prompt

Используйте **Global Prompt** для свойств, которые должны оставаться постоянными во всей sequence:

- characters и identity;
- clothing и persistent props;
- persistent environment;
- visual style;
- lighting logic;
- atmosphere;
- другие global continuity constraints.

Clip cards затем описывают локальные изменения timeline вместо повторения полного world description.

## Clip cards

Prompt каждого clip должен описывать, что происходит на соответствующем участке timeline.

```text
Clip 1
Duration: 6s
The woman walks slowly through the ruined city.

Clip 2
Duration: 5s
She continues walking and gradually raises her right hand.

Clip 3
Duration: 7s
Her raised hand reaches her face as she looks toward the burning buildings.
```

Длительность каждого clip может отличаться.

Практическая отправная точка:

```text
4–8 s   — высокий контроль
7–12 s  — универсальный диапазон
12+ s   — меньше boundaries, но выше вероятность visual drift
```

## Seed

Пустой clip Seed означает:

```text
auto
```

LongMedia выводит per-clip seed из sampler base seed и identity/order clip.

Фиксированный clip seed полезен для воспроизводимых A/B тестов.

## Reorder clips

Planner cards можно переставлять drag-and-drop.

Card сохраняет:

- prompt;
- duration;
- seed;
- name;
- `clip_id`.

Если Cameras подключён с **Auto Sync Planner**, соответствующая camera card следует за тем же `clip_id`.

## Import нескольких prompts

Planner поддерживает structured prompt import:

```text
clip_1:
The woman enters the hall.

clip_2:
She approaches the central altar.

clip_3:
She slowly raises both hands.
```

Поддерживается alias `shot_N:`:

```text
shot_1:
...

shot_2:
...
```

Используйте **Import Prompt**, чтобы преобразовать structured text в обычные editable cards.

**Auto Import Prompt** удобен, если structured text приходит из другой node или внешнего LLM.

## Presets

Clip cards поддерживают reusable presets, включая JSON import/export. Они полезны для повторяющихся action structures, transition patterns и production templates.

## Audio

Audio behavior управляется `audio_mode` в Setup. Для source-video editing см. [Режимы аудио и video_ref_edit](AUDIO_MODES_GUIDE_RU.md).
