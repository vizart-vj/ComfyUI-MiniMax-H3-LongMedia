# Режимы аудио и `video_ref_edit`

Это руководство объясняет, когда аудиовход обязателен и что происходит с финальной звуковой дорожкой.

## Важно: `video_1` никогда не содержит аудио

Сокет `video_1` получает **IMAGE batch только с видеокадрами**.

Даже если эти кадры загружены из видеофайла со звуковой дорожкой, звук не передаётся через соединение `video_1`.

Если нужен оригинальный звук source video, извлеките/загрузите его отдельно и подключите к `audio_1` (или к другому audio input, если того требует сценарий).

Рекомендуемый source-video workflow:

```text
Source movie
├── video frames ──> video_1
└── extracted audio ──> audio_1
```

## Требования к аудио в `video_ref_edit`

`video_ref_edit` всегда требует `video_1`. Необходимость аудио зависит от `audio_mode`.

## Синхронизация source performance в `video_ref_edit`

Когда новый персонаж генерируется по Picture references, простого возврата оригинальной дорожки только на финальном mux недостаточно для сохранения синхронизации речи/вокала. Поэтому LongMedia рассматривает подключённый парный source soundtrack как часть source-performance contract.

Для `video_ref_edit`:

```text
audio_mode = auto + audio_1 connected
audio_mode = preserve
audio_mode = preserve_reference
```

`audio_1` кодируется в target AV timeline и фиксируется как авторитетный audio clock, пока video stream регенерируется. Новый персонаж генерируется относительно точного тайминга исходной performance, а неизменённая source waveform всё равно восстанавливается на финальном выходе.

Это сохраняет синхронизацию:

- артикуляции рта;
- речи и вокала;
- ритма дыхания;
- тайминга выражений лица;
- движения головы/тела, связанного с исходным soundtrack.

`reference_only` оставляет подключённый звук отдельным H3 audio reference, а финальный soundtrack генерирует H3. `generate` также оставляет финальный звук за H3. В `video_ref_edit + lip_sync` `audio_1` намеренно **не считается** оригинальным soundtrack Video1: это независимый авторитетный dub/timing source, поэтому совершенно новая речь или вокал могут управлять заменённым персонажем.

| `audio_mode` | Нужен ли `audio_1`? | Поведение финального soundtrack |
| --- | --- | --- |
| `auto` | **Нет** | Подключённый `audio_1` сохраняется/восстанавливается. Если Audio1 отключён, используется H3-generated audio; Audio2/Audio3 могут оставаться conditioning references. |
| `preserve` | **Да**, если нужно сохранить source audio | Восстанавливает неизменённый подключённый звук. В `video_ref_edit` Audio1 также нативно объединяется с Video1 как soundtrack и фиксируется на target AV clock. |
| `generate` | Нет | Для финального результата используется звук, сгенерированный H3. |
| `reference_only` | Только если нужен audio reference | Подключённый звук участвует как H3 reference, но финальная дорожка генерируется моделью. |
| `preserve_reference` | **Да** | Использует подключённый звук как H3 reference и восстанавливает неизменённую source track на выходе. |
| `lip_sync` | **Да: `audio_1`** | `audio_1` — авторитетный timing/content source для нативного H3 lip-sync; неизменённая дорожка восстанавливается на выходе. Текущий lip-sync также требует `image_1`. |

## `auto`

`auto` — гибкий режим по умолчанию.

### С подключённым `audio_1`

```text
video_1  = source frames
audio_1  = source soundtrack
audio_mode = auto
```

LongMedia сохраняет подключённое аудио и возвращает его как финальную дорожку. В `video_ref_edit` `audio_1` дополнительно фиксируется внутри target AV stream как авторитетный source-performance clock, поэтому facial/mouth motion нового персонажа генерируется относительно исходного тайминга soundtrack.

### Без `audio_1`

```text
video_1  = source frames
audio_1  = disconnected
audio_mode = auto
```

Это допустимо. LongMedia позволяет H3 сгенерировать output audio и декодирует созданный audio stream.

Следовательно, **`audio_1` необязателен для `video_ref_edit + auto`**.

Audio2/Audio3 не становятся автоматически passthrough soundtrack, если Audio1 отключён; они остаются prompt-conditioning references.

## `preserve`

Используйте `preserve`, когда исходная звуковая дорожка должна остаться неизменной.

```text
video_1  = source frames
audio_1  = extracted original soundtrack
audio_mode = preserve
```

Source audio восстанавливается на выходе вместо sampled H3 audio stream. В `video_ref_edit` `preserve` также делает этот source track авторитетным target-audio timing stream, чтобы сохранить синхронизацию mouth/facial performance при генерации новой identity.

Preserve-style режиму нужен реально подключённый source soundtrack. Если его нет, LongMedia не может восстановить звук, который не поступал в workflow.

Поэтому **подключайте `audio_1` для `video_ref_edit + preserve`**.

## `preserve_reference`

Используйте этот режим, когда source audio должно участвовать в H3 conditioning и одновременно точная исходная waveform должна быть восстановлена на выходе. В `video_ref_edit` он также включает тот же source-performance timing lock, что `auto + audio_1` и `preserve`.

```text
video_1  = source frames
audio_1  = source soundtrack / reference
audio_mode = preserve_reference
```

Для предполагаемого контракта режим требует подключённого source audio.

## `generate`

Используйте, когда H3 должен создать новую звуковую дорожку.

```text
video_1  = source frames
audio_mode = generate
```

Входной soundtrack не требуется для финального audio contract.

## `reference_only`

Используйте, когда аудио подаётся только как conditioning reference, а финальный soundtrack всё равно генерирует H3.

```text
audio_1 = reference audio
audio_mode = reference_only
```

Подключение имеет смысл, когда audio reference действительно нужен. Финальный звук остаётся generated, а не source-audio passthrough.

## `lip_sync`

Для нативного LongMedia H3 lip-sync:

```text
image_1 = visual subject / opening image
audio_1 = authoritative speech or singing performance
audio_mode = lip_sync
```

`audio_1` остаётся нативным H3 audio conditioning, управляет таймингом per-clip H3 Audio Guide и восстанавливается неизменённым на финальном выходе.

Для текущего LongMedia lip-sync обязательны и `image_1`, и `audio_1`. `audio_2` и `audio_3` могут использоваться как дополнительные prompt-addressable H3 references; они не заменяют Audio1 как lip-sync clock или финальный passthrough track.

Для lip-sync используйте `audio_strength = 1.0`.

## Быстрая таблица выбора

Если у source movie важен оригинальный звук:

```text
Нужно автоматическое поведение?          -> auto + connect audio_1
Нужно гарантированно сохранить звук?     -> preserve + connect audio_1
Нужен ref + неизменённый output?         -> preserve_reference + connect audio_1
Нужен lip-sync по этой дорожке?          -> lip_sync + connect image_1 + audio_1
Нужен новый H3 soundtrack?               -> generate
Нужен звук только как H3 reference?      -> reference_only + connect audio_1
```

Если полезного source soundtrack нет:

```text
auto     -> audio_1 можно не подключать; звук генерирует H3
generate -> audio_1 можно не подключать; звук генерирует H3
```

## `video_ref_edit`: парная Source AV Performance

Для character replacement/editing `video_1` и его оригинальный soundtrack следует рассматривать как одну source performance.

При `audio_mode = auto`, `preserve` или `preserve_reference` и подключённом `audio_1` LongMedia использует два взаимодополняющих механизма:

1. `video_1 + audio_1` отправляются MiniMax H3 как единый нативный paired `video_audio` reference block. Это сохраняет связь facial/body performance источника с его soundtrack.
2. `audio_1` также записывается в target audio stream и фиксируется как generation clock. Неизменённая source waveform возвращается на финальном выходе.

Для `video_ref_edit` `duration_source = auto` следует таймлайну `video_1`. Это предотвращает обрезание последнего визуального фрагмента из-за чуть более короткой encoded audio/container duration.

Рекомендуемая конфигурация:

```text
h3_mode:    video_ref_edit
video_1:    source performance frames
image_1:    replacement character / identity reference
audio_1:    soundtrack extracted from video_1
audio_mode: preserve   (or auto / preserve_reference)
```

`video_1` остаётся IMAGE batch и сам по себе не содержит аудио. Извлечённый soundtrack подключайте отдельно к `audio_1`.

## `duration_source`: владелец таймлайна независим

`duration_source` управляет **только длиной target timeline**. Он не решает, доступно ли подключённое аудио H3, и не заменяет `audio_mode`.

В `video_ref_edit`:

| `duration_source` | Target duration | Типичный сценарий |
| --- | --- | --- |
| `auto` | длительность `video_1` | Безопасный source-edit default. |
| `video` | длительность `video_1` | Явно сохранить горизонт исходного видео. |
| `audio` | длительность `audio_1` | Redub на более короткую/длинную дорожку; длинное audio может продолжить сцену за Video1. |
| `manual` | `manual_duration` | Принудительно задать любую target duration. |
| `longest_input` | Самый длинный подключённый video/audio input | Автоматически следовать самому длинному source/reference. |

Пример:

```text
video_1 = 6 s
audio_1 = 11 s
h3_mode = video_ref_edit
audio_mode = lip_sync
duration_source = audio
```

Target ≈11 секунд. Video1 задаёт первые ≈6 секунд scene/camera/performance reference, затем H3 продолжает сцену, а Audio1 остаётся авторитетным redub clock.

```text
video_1 = 10 s
audio_1 = 6 s
duration_source = audio
```

Target использует первые ≈6 секунд Video1 и заканчивается на audio-owned horizon.

```text
video_1 = 6 s
manual_duration = 8 s
duration_source = manual
```

Target ≈8 секунд независимо от длины input media.

Смена `duration_source` никогда не удаляет `<Audio N>` references из prompt conditioning.

Passthrough audio подгоняется к выбранному timeline только на финальном выходе: более короткий target обрезает waveform на target boundary, более длинный сохраняет всю source waveform и добавляет тишину после неё. Samples внутри сохранённого диапазона не resample и не retime.

## Произвольный redub в `video_ref_edit`

Для новой речи/вокала, **отличающихся от оригинальной performance source video**:

```text
h3_mode = video_ref_edit
video_1 = source scene / movement / camera
image_1 = replacement character
audio_1 = NEW speech or singing
audio_mode = lip_sync
duration_source = video | audio | manual | longest_input
```

Audio1 рассматривается как standalone авторитетный target-performance clock. Он не объявляется ложно оригинальным soundtrack, парным с Video1.

Используйте `duration_source=audio`, если новая performance должна определять длину output. Используйте `video`, если edit должен остаться внутри длительности source video.

## Несколько audio references

`video_ref_edit` может использовать несколько подключённых audio inputs. Нативные prompt tags доступны как `<Audio 1>`, `<Audio 2>`, `<Audio 3>` в соответствии с порядком подключённых references.

Роли могут различаться:

```text
Audio 1 = original soundtrack or authoritative dub
Audio 2 = percussion / rhythm reference
Audio 3 = bass / music / ambience reference
```

`audio_mode` управляет final-audio/timing semantics и не удаляет Audio2/Audio3 из prompt conditioning. В `video_ref_edit` **только Audio1** является source/final passthrough soundtrack authority для `auto`, `preserve`, `preserve_reference` и `lip_sync`. Audio2/Audio3 — дополнительные semantic/music references и не смешиваются в preserved output track.

### Audio-reactive prompting

Музыкальную структуру можно адресовать прямо в prompt. Для одного mixed soundtrack:

```text
<Video 1> defines the original street, motion and camera path.
<Picture 1> defines the replacement character.
<Audio 1> defines the musical rhythm and temporal structure.

Street lights illuminate rhythmically with the percussion transients from <Audio 1>.
Architectural glitch patterns pulse on the strongest drum hits.
The road surface, reflections and flowing light respond to the low-frequency bass rhythm.
Each bass accent sends a smooth wave of illumination forward along the street.
```

Для отдельных references/stems:

```text
<Audio 2> defines the percussion timing.
<Audio 3> defines the bass rhythm.
Street lights pulse with <Audio 2>.
Glitch patterns travel across the buildings on the strongest <Audio 2> hits.
The road and reflections move in smooth low-frequency waves following <Audio 3>.
```

Это H3 semantic AV-conditioning instructions. LongMedia сам не выполняет DSP stem separation; если drums и bass находятся в одном mixed soundtrack, явно опишите нужный компонент соответствующего `<Audio N>` в prompt.

## Краткий контракт

Удобно считать controls тремя независимыми измерениями:

```text
duration_source -> кто определяет target length
audio_mode      -> какое audio авторитетно / сохраняется / генерируется
prompt          -> как каждый <Audio N> влияет на visuals и performance
```

Такое разделение позволяет выполнять source character replacement, точное сохранение source performance, произвольный redub, продолжение source video, trimming и audio-reactive visual edits без введения отдельных workflow modes.
