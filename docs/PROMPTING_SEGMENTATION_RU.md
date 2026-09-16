# Правила prompting для Fixed Segmentation

Используйте fixed segmentation при:

```text
timeline_mode = segmented
```

Segmentation — внутренний execution strategy для одного непрерывного semantic movie. Это не storyboard scheduler.

## Mental Model

LongMedia создаёт внутренние units фиксированной длительности из `segment_duration` и переносит H3 continuation context между ними.

Prompt должен описывать **финальный непрерывный фильм**, а не скрытую segmentation.

Хорошо:

```text
The woman walks steadily through the corridor while the surrounding lights gradually become warmer.
Her pace and direction remain consistent as the environment develops around her.
```

Не пишите инструкции вроде “segment 1 starts” или “each segment resets”. Модель не должна разыгрывать внутреннюю бухгалтерию LongMedia.

## Segment Duration

`segment_duration` — количество новой видимой timeline, которое создаёт каждый fixed unit.

`transition_frames` — скрытый continuation context и не вычитается из видимой длительности.

Более короткие segments обычно:

- уменьшают peak sequence geometry;
- улучшают control на constrained GPU;
- создают больше continuation boundaries.

Более длинные segments обычно:

- уменьшают число boundaries;
- увеличивают packed sequence/workspace size.

## Непрерывные действия

Используйте temporal continuation language:

```text
continues walking
keeps singing
gradually turns
maintains the same direction
the illumination develops steadily
```

Описывайте начало действия только если оно действительно начинается в этой точке финального ролика.

## References

Conditioning family по-прежнему задаётся `h3_mode`.

Примеры:

- `hybrid + segmented` — continuous movie с opening keyframe;
- `ref2va + segmented` — continuous movie, управляемый references;
- manual mode — advanced custom diagnostics.

`video_ref_edit` обычно использует один source timeline; для reconstruction/edit plans с сегментированными source windows используйте Video Reconstructor contract.

## Audio

При `audio_mode=lip_sync` Audio1 владеет performance timing. LongMedia сам режет/выравнивает активный source timeline; не делите phonemes вручную по segment boundaries.

Для preserve-style modes помните: Video IMAGE input не содержит soundtrack. Source audio подключается отдельно.

## Когда нужен MultiClip

Выбирайте MultiClip, когда отдельным секциям нужны собственные:

- prompt;
- duration;
- seed;
- camera card;
- явная storyboard identity.

Выбирайте Segmented, когда ролик по сути является одним continuous prompt, а segmentation нужна прежде всего как execution/VRAM решение.
