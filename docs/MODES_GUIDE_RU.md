# Режимы LongMedia

LongMedia намеренно разделяет **как H3 получает conditioning**, **как строится timeline**, **кто задаёт длительность** и **что происходит с аудио**. Эти controls независимы.

## 1. Control Mode

### `auto`
Рекомендуется для обычной работы. LongMedia выводит legacy internal workflow contract из семантических controls ниже.

### `manual`
Открывает advanced/legacy controls для диагностики, A/B тестов и намеренно нестандартного conditioning.

Для production начинайте с `control_mode=auto`.

---

## 2. H3 Conditioning Mode

### `t2va`
Чистая text-to-video/audio генерация.

Используйте, когда шот должен строиться только из prompt без Picture/Video reference conditioning.

### `fl2va`
Нативный MiniMax H3 first/last-frame conditioning.

- `image_1` обязателен как first frame;
- `image_2` можно использовать как last frame;
- дополнительные Picture/Video/reference-audio inputs не входят в эту conditioning family;
- LongMedia сохраняет нативный H3 keyframe path.

### `ref2va`
Нативный MiniMax H3 reference-to-video/audio conditioning.

Подключённые Picture, Video и Audio подаются как native references и доступны в prompt как `<Picture N>`, `<Video N>`, `<Audio N>`.

### `hybrid`
LongMedia workflow с opening keyframe и дополнительными references.

- `image_1` — opening anchor;
- `image_2` может быть final anchor, если активная policy это допускает;
- дополнительные references могут управлять identity, style, environment или motion.

Режим открывает `first_frame_mode` для advanced opening-frame behavior.

### `video_ref_edit`
Использует `video_1` как главный source-performance / motion / camera / composition reference, а Picture references задают замену identity или style changes.

Типичный character replacement:

```text
video_1  = source performance
image_1  = replacement identity
audio_1  = source soundtrack or new dub
```

`video_1` — только IMAGE batch кадров. Soundtrack через него не передаётся; аудио нужно извлечь/загрузить отдельно.

---

## 3. Timeline Mode

### `single`
Один H3 target timeline.

Лучший выбор для обычного T2VA, FL2VA, Ref2VA, Hybrid и single-source `video_ref_edit`.

### `segmented`
LongMedia делит один непрерывный фильм на внутренние сегменты фиксированной длительности.

Используйте, когда:

- один semantic prompt продолжается через весь ролик;
- удобны одинаковые segment sizes;
- segmentation нужна прежде всего для VRAM или стабильности длинного прогона.

`segment_duration` — количество новой видимой timeline на segment. `transition_frames` — скрытый continuation context.

### `multiclip`
**Long Media Planner** владеет clip prompts, durations, names и optional seeds.

Рекомендуемая схема с Cameras:

```text
Long Media Planner
        ↓ clip_plan
Long Media Cameras
        ↓ clip_plan
Long Media Setup
```

`clip_plan` авторитетен только при `timeline_mode=multiclip` и игнорируется в `single`/`segmented`.

---

## 4. Duration Source

`duration_source` всегда видим. Он управляет **только длиной timeline** и не меняет semantic role аудио и не удаляет references из H3 conditioning.

### `auto`
Mode-aware default.

Для `video_ref_edit` `auto` использует длительность `video_1`.

### `video`
Явно использовать длительность `video_1`.

### `audio`
Использовать длительность `audio_1`.

Если Audio1 длиннее Video1 в `video_ref_edit`, target может продолжиться после конца source video. Если Audio1 короче — target сокращается до audio timeline.

### `manual`
Использовать `manual_duration`.

Подходит, когда output намеренно должен быть короче или длиннее всех подключённых sources.

### `longest_input`
Использовать самый длинный подключённый Video/Audio source.

```text
video_1 = 6 s
audio_1 = 11 s
duration_source = longest_input
→ target = 11 s
```

MultiClip durations остаются Planner-owned. Reconstruction workflows остаются source-plan-owned.

---

## 5. Audio Mode

### `auto`
Гибкий default.

В `video_ref_edit`:

- с `audio_1`: Audio1 становится source soundtrack/performance clock и сохраняется на выходе;
- без `audio_1`: H3 может генерировать audio.

### `preserve`
Сохраняет подключённый source soundtrack на выходе.

Для `video_ref_edit` подключайте `audio_1`; Video1 содержит только frames.

### `generate`
Использовать финальное H3-generated audio.

### `reference_only`
Использовать подключённый звук как H3 semantic/reference conditioning, но финальный звук оставить generated.

### `preserve_reference`
Использовать подключённый звук как H3 reference/timing context и восстановить source waveform на выходе.

### `lip_sync`
Audio1 становится авторитетным performance clock и финальным soundtrack.

Для `video_ref_edit` это также **redub mode**: Audio1 может содержать совершенно другую речь/вокал по сравнению с source Video1. Video1 остаётся visual/motion reference, а Audio1 управляет новой articulation.

Текущий LongMedia lip-sync требует `image_1 + audio_1`. Для H3 lip-sync используйте `audio_strength=1.0`.

См. [Режимы аудио и video_ref_edit](AUDIO_MODES_GUIDE_RU.md).

---

## 6. Несколько Audio References

`audio_1`, `audio_2`, `audio_3` могут выполнять разные задачи. В `video_ref_edit` Audio1 владеет source/final passthrough soundtrack; Audio2/Audio3 остаются conditioning references и не микшируются в preserved track.

```text
Audio 1 = dialogue / lip-sync
Audio 2 = percussion reference
Audio 3 = bass reference
```

Prompt может обращаться к ним независимо:

```text
<Audio 2> defines the percussion timing.
Street lights pulse with the percussion accents.
Architectural glitch patterns react to the strongest drum hits.
The road illumination moves in smooth waves following the bass rhythm from <Audio 3>.
```

LongMedia не делает автоматический stem separation. Если несколько музыкальных элементов находятся в одном mixed track, H3 интерпретирует их мультимодально из этой дорожки.

---

## 7. Loop Closure

Loop Closure независим от H3 и timeline modes.

Включайте, когда хвост ролика должен вернуться к opening macro-state. LongMedia выполняет closure в latent/H3 space, а не RGB crossfade.

Основные controls:

- `loop_closure_enabled`
- `loop_closure_frames`
- `loop_closure_strength`

---

## Типовые рецепты

### Text-to-video/audio

```text
control_mode    = auto
h3_mode         = t2va
timeline_mode   = single
duration_source = manual
audio_mode      = generate
```

### Нативный first/last frame

```text
control_mode  = auto
h3_mode       = fl2va
timeline_mode = single
image_1       = first frame
image_2       = optional last frame
```

### Reference-driven generation

```text
control_mode  = auto
h3_mode       = ref2va
timeline_mode = single
```

### Character replacement с оригинальным soundtrack

```text
control_mode    = auto
h3_mode         = video_ref_edit
timeline_mode   = single
duration_source = auto
audio_mode      = preserve

video_1 = source video frames
image_1 = replacement character
audio_1 = source soundtrack
```

### Character replacement с новым dub и continuation

```text
control_mode    = auto
h3_mode         = video_ref_edit
timeline_mode   = single
duration_source = audio
audio_mode      = lip_sync

video_1 = source performance
image_1 = replacement character
audio_1 = new longer dialogue
```

Если Audio1 длиннее Video1, H3 может продолжить сцену за пределы source clip, следуя новому audio clock.

### Длинная непрерывная сцена

```text
control_mode  = auto
h3_mode       = hybrid or ref2va
timeline_mode = segmented
```

### Directed storyboard

```text
control_mode  = auto
h3_mode       = ref2va
timeline_mode = multiclip
Planner → Cameras → Setup
```

## Совместимость

Старые workflow могут содержать `workflow_mode` вроде `hybrid_auto`, `ref2va_full`, `segmented_continuation`, `multiclip`. LongMedia сохраняет эти serialized values для backward compatibility и мигрирует их в semantic controls. Новые workflow должны использовать `control_mode`, `h3_mode`, `timeline_mode`.
