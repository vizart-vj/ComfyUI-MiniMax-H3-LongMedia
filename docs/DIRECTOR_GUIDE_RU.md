# LongMedia Director 0.6.50 — полное руководство

LongMedia Director — таймлайн-ориентированная среда авторинга для **MiniMax H3 • LongMedia**. В одном документе Director объединяет длительности шотов, семантические референсы, якоря FIRST/LAST, камеры, временные H3-эмбеддинги, политику аудио, предпросмотр и систему TAKE.

## Рекомендуемое подключение

```text
MiniMax H3 • LongMedia Director.director
                    ↓
MiniMax H3 • Long Media Setup.director
```

В Setup установите `control_mode = director`. После этого Director владеет авторингом и таймлайном, а Setup/Sampler продолжают отвечать за runtime, память, совместимость модели и декодирование.

## Чем управляет Director

Director владеет:

- таймлайном MAIN shot/clip и авторской длительностью;
- глобальным и поклиповыми промптами;
- семантическими Picture / Video / Audio референсами;
- ролями FIRST и LAST;
- CAMERA-треками и временными H3 EMBEDDING-треками;
- политикой аудио Director;
- aspect/resolution/megapixel политикой Director;
- TAKE-снапшотами, состоянием просмотра и метаданными selective regeneration.

Setup продолжает управлять runtime-параметрами, которые Director явно не переопределяет: совместимостью модели, reference budget, loop closure, motion repair и downstream sampling/memory поведением.

Явные media-сокеты Setup остаются рабочими override. Например, `Setup.image_1` заменяет только Director `<Picture 1>`, сохраняя остальной документ Director.

## Режимы таймлайна

**Один MAIN shot остаётся одним H3 sampler pass.** Временные CAMERA/EMBEDDING controls не создают скрытых дополнительных generation passes.

Верхний селектор `TIMELINE` выбирает модель исполнения Director.

### `single`

Один MAIN shot = один H3 generation pass. Это основной вариант для непрерывного шота или одного source-video edit.

### `multiclip`

Границы MAIN становятся границами клипов. У каждого клипа свой prompt/seed/timing, а LongMedia переносит нативный continuation context между соседними клипами.

Director показывает перетаскиваемые clip markers. `+ CLIP` делит клип в позиции playhead, `− CLIP` удаляет выбранную/ближайшую границу и объединяет соседние клипы.

### `segmented`

Одна семантическая сцена разбивается на внутренние фиксированные execution-сегменты ради VRAM/стабильности. `SEG` / `SEG LEN` задают политику сегментации. Это не storyboard cut: сцена должна оставаться непрерывной.

## MAIN, Ripple и snapping

MAIN-клипы можно выбирать, перемещать, переставлять, разделять и менять по длительности. **Magnet** включает snapping к релевантным границам и playhead.

`Ripple` определяет, следуют ли CAMERA/AUDIO/прочие слои за правками MAIN:

- Ripple ON — зависимые слои двигаются вместе с MAIN;
- Ripple OFF — сохраняют абсолютное авторское время.

## Семантический H3 routing

Director отделяет **семантическую роль** от физического media slot.

### REF

REF — нативный семантический reference. Он остаётся референсом и не превращается скрыто в FIRST/LAST.

В prompt используются нативные H3-теги:

```text
<Picture 1>
<Video 1>
<Audio 1>
```

### FIRST

FIRST — нативный opening-frame anchor.

### LAST

LAST — нативный closing-frame anchor.

FIRST и LAST — глобально эксклюзивные роли. Один и тот же source можно назначить одновременно FIRST и LAST для same-source loop anchor. В этом случае LongMedia использует одну согласованную геометрическую трансформацию обоих endpoint, чтобы FIRST/LAST не получили различный crop.

## Автоматический выбор conditioning family

Director выводит безопасную H3-схему из семантических ролей:

```text
только текст                 -> t2va
REF media                    -> ref2va
только FIRST и/или LAST      -> fl2va
FIRST/LAST + REF media       -> hybrid
```

Если к FL2VA-шоту добавлен реальный REF, runtime переводится в Hybrid вместо попытки протащить несовместимый reference через чистый FL2VA.

### Редактирование source video

Обычный Video с ролью REF остаётся нативным Ref2VA reference conditioning. Source-performance / replacement editing использует отдельный нативный character/source-edit path, который строит `video_ref_edit` вокруг исходного видео и, при необходимости, source/dub audio. Подробнее: [Audio Modes и `video_ref_edit`](AUDIO_MODES_GUIDE_RU.md).

## Управление разрешением

Director может полностью владеть геометрией результата независимо от `width` / `height` Setup.

Верхние параметры:

- `RES SRC` — Base layer, 16:9, 9:16, 1:1, 4:3, 3:4, 21:9, 2.39:1 или Custom;
- `RES` — явный W×H источник aspect при Custom;
- `MP` — целевой бюджет мегапикселей.

`Base layer` получает aspect из первого визуального MAIN-источника; при его отсутствии используется 16:9. Итоговые размеры привязываются к безопасной для H3 кратности.

При `control_mode=director` виджеты width/height в Setup остаются видимыми и подключаемыми ради совместимости workflow, но геометрия Director является авторитетной.

## Редакторы промптов

Director содержит global prompt и отдельные prompt editors для MAIN. Их высота меняется пользователем и сохраняется в UI state ноды.

Для временных CAMERA/EMBEDDING controls общий scene/subject/reference prompt кодируется один раз на весь MAIN. Частичные CAMERA и EMBEDDING слои являются факторизованными controls, а не повторными копиями полного scene prompt.

## Камеры

CAMERA blocks — недиегетический слой кинематографии. Они должны описывать framing/movement, а не дублировать содержимое сцены.

Важные правила тайминга:

- blocks используют авторские half-open интервалы;
- gap означает отсутствие camera instruction;
- при overlap выигрывает блок с более поздним start; при равенстве — более поздняя запись документа;
- соседние camera blocks получают короткий centered handoff вместо скрытого hard cut;
- одна камера на весь шот может оставаться на нативном/global conditioning path.

Camera control H3 — это learned language conditioning, а не математически калиброванные 3D camera extrinsics.

См. [Long Media Cameras](CAMERAS_GUIDE_RU.md).

## H3 embedding tracks

Нажмите `✦`, чтобы добавить EMBEDDING track и выбрать установленный H3 embedding. Переместите/обрежьте block для выбора временного диапазона; `↔` растягивает его на весь timeline.

Пересекающиеся embedding layers комбинируются. Disabled/muted слои неактивны. Camera и embedding boundaries независимы: embedding transition не перезапускает camera presentation, а camera boundary не клонирует полный scene prompt.

## AUDIO selector

Верхний `AUDIO` selector использует стандартный LongMedia audio contract:

- `auto` — автоматическая политика LongMedia;
- `lip_sync` — Audio 1 является авторитетным таймингом речи/вокала и финальной восстановленной waveform;
- `generate` — финальное audio генерирует H3;
- `reference_only` — входное audio участвует в H3 conditioning, финальное audio генерируется;
- `preserve_reference` — input audio является conditioning/timing reference, Audio 1 восстанавливается на выходе;
- `preserve` — Audio 1 сохраняется на выходе без использования как обычного H3 reference.

Для нативного lip-sync используйте `audio_strength = 1.0`. Audio 1 является авторитетным target clock; дополнительные audio refs могут использоваться для музыки, ритма или ambience, если выбранный mode это допускает.

См. [Audio Modes](AUDIO_MODES_GUIDE_RU.md).

## Program Monitor

Декодированные preview TAKE отображаются в Program Monitor Director. Доступны play/pause, timeline seek, переходы по границам клипов, loop и draggable playhead.

Переключатели качества `25 / 50 / 75 / FULL` меняют только browser-side качество предпросмотра и не создают несколько encoded media files.

## Система TAKE

TAKE — это полный снапшот состояния редактора Director, а не только thumbnail.

В TAKE сохраняется всё необходимое для возврата к авторской версии:

- MAIN timeline;
- global/per-shot prompts;
- WHO & WHAT media;
- назначения FIRST/LAST/REF;
- CAMERA / AUDIO / EMBEDDING и дополнительные tracks;
- H3/timeline/audio/resolution controls;
- seed и render metadata;
- cached latent/preview data, если они существуют.

### CREATE TAKE

`CREATE TAKE` создаёт новый пустой TAKE и сразу делает его активным editor workspace. Следующий успешный render заполняет **тот же TAKE**, а не создаёт скрыто второй.

### Restore

Клик/Restore TAKE атомарно заменяет документ Director снапшотом выбранного TAKE. Поэтому Restore намеренно пустого TAKE возвращает редактор в пустое состояние.

### Duplicate / Delete / Rename

TAKE можно дублировать, удалять и переименовывать. Rename меняет label в UI и соответствующую папку TAKE на диске, когда это возможно.

### Папки проектов

Библиотека TAKE начинается с `Unsorted`. Пользовательские папки можно создавать вручную и перемещать между ними TAKE cards.

Текущее хранилище TAKE-centric:

```text
longmedia_director/
├─ Unsorted/
│  └─ Take_Name__<id>/
│     ├─ take.json
│     ├─ state/
│     │  ├─ director.json
│     │  └─ global_prompt.txt
│     ├─ clips/
│     │  └─ <clip_id>/latent.safetensors
│     └─ previews/
│        ├─ poster.png
│        ├─ sprite.jpg
│        └─ audio.wav
├─ <user project folders>/
└─ _runtime/
```

Фактический набор файлов зависит от того, является ли TAKE пустым, отрендеренным, закешированным и декодированным.

## Layout TAKE gallery

TAKE panel можно растягивать по горизонтали. Scrollbar расположен слева, чтобы правый край оставался чистым resize handle. Изменение ширины TAKE panel одновременно расширяет/сужает саму Director node. Карточки автоматически раскладываются в несколько колонок.

Timeline zoom, ширина TAKE, высота inspector, размеры prompt editors и видимость TAKE сериализуются в Director UI state и переживают обычные rerender/workflow save.

## Selective regeneration

Director умеет сохранять уже утверждённую работу и пересчитывать только необходимый dependency range.

Основные действия:

- **Regenerate Clip** — пересчитать выбранный clip, когда оба seam constraints можно безопасно сохранить;
- **Reroll Seed** — изменить seed выбранного clip и использовать selective path;
- **Regenerate From Here** — сохранить валидированный prefix и пересчитать выбранный clip плюс зависимый suffix;
- **Recast / source replacement tools** — перестроить необходимую native-reference branch, сохраняя незатронутое состояние timeline там, где это допускает dependency contract.

Если exact seam locking небезопасен, LongMedia расширяет invalidated region вместо попытки использовать несовместимый cached suffix.

См. [Director Selective Regeneration](DIRECTOR_REGENERATION_GUIDE_RU.md).

## Взаимодействие с Refine / Latent Hi-Res

Тайминг Director остаётся глобальным даже при windowed post-x0 Refine в Sampler. Camera/embedding ranges проецируются в каждое refine window по исходным часам сегмента; refine windows не становятся новыми Director clips.

Текущий Refine video-authoritative: audio Stage 1 сохраняется точно, а video latent может проходить upscale/refine. См. [Integrated Refine and Latent Hi-Res](TWO_PASS_LATENT_HIRES_REFINER_GUIDE_RU.md).

## Рекомендуемый workflow

1. Подключите Director к Setup и установите `control_mode=director`.
2. Выберите `TIMELINE` и resolution policy.
3. Добавьте MAIN media и prompts.
4. Явно назначьте семантические роли REF, FIRST, LAST.
5. Добавляйте CAMERA / EMBEDDING / AUDIO layers только там, где они нужны.
6. Выберите Director `AUDIO` policy.
7. Перед альтернативным экспериментом создайте TAKE, если нужен отдельный branch.
8. Рендерьте через обычный LongMedia Sampler/Decode path.
9. Проверяйте результат в Program Monitor и используйте selective regeneration вместо повторного рендера уже утверждённых областей.

## Диагностика

### Reference ведёт себя как frame anchor
Проверьте semantic role. REF, FIRST и LAST — разные контракты.

### FIRST/LAST loop медленно перекадрируется
Используйте один и тот же media source для обеих ролей. Same-source path использует согласованную endpoint geometry.

### Restore возвращает пустой editor
Проверьте, выбран ли намеренно пустой TAKE или rendered TAKE. Rendered TAKE должен содержать финализированный snapshot/state.

### TAKE panel не расширяется
Тяните правую границу TAKE pane. Scrollbar специально перенесён влево; ширина node следует за resize.

### Director resolution не совпадает с Setup width/height
При `control_mode=director` геометрия Director является авторитетной по дизайну.

### Camera changes выглядят как cuts
Не помещайте scene-change language в CAMERA blocks. Scene/action continuity относится к MAIN prompts; CAMERA управляет framing и movement.

## Связанная документация

- [Operating Modes](MODES_GUIDE_RU.md)
- [Audio Modes и video_ref_edit](AUDIO_MODES_GUIDE_RU.md)
- [Director Selective Regeneration](DIRECTOR_REGENERATION_GUIDE_RU.md)
- [Cameras Guide](CAMERAS_GUIDE_RU.md)
- [Integrated Refine and Latent Hi-Res](TWO_PASS_LATENT_HIRES_REFINER_GUIDE_RU.md)
- [Sampler / VRAM / Performance](SAMPLER_OPTIMIZATION_RU.md)
