# LongMedia Director — Selective Regeneration и TAKE Workflow

Руководство описывает текущую модель regeneration в Director 0.6.41.

## TAKE — верхнеуровневая сохранённая ревизия

Хранилище Director теперь TAKE-centric. Пользовательские ревизии принадлежат TAKE, а не отдельным clip-cache папкам.

Концептуально:

```text
longmedia_director/
├── Unsorted/
│   └── <take>/
│       ├── take.json
│       ├── state/
│       │   ├── director.json
│       │   └── global_prompt.txt
│       ├── clips/
│       │   └── <clip_id>/latent.safetensors
│       └── previews/
└── _runtime/
```

Пользовательские project folders находятся рядом с `Unsorted`. `_runtime` хранит служебные runtime pointers и не является коллекцией TAKE.

## CREATE TAKE

CREATE создаёт полный пустой TAKE workspace и сразу делает его активной редакцией. Следующий успешный render заполняет именно этот TAKE; завершение render не создаёт дополнительную соседнюю ревизию.

## Restore

Выбор/Restore TAKE заменяет Director document сохранённым snapshot целиком, а не переключает только preview branch. Возвращаются timeline media, WHO & WHAT media, prompts, GLOBAL PROMPT, роли FIRST/LAST/REF, camera/audio/extra tracks, resolution policy и режимы Director.

Browser media cache инвалидируется, чтобы UI показывал именно восстановленную ревизию, а не старый preview state.

## Duplicate, rename и folders

- Duplicate создаёт новый TAKE из полного snapshot исходной ревизии.
- Rename меняет и видимое имя TAKE, и имя его директории.
- TAKE cards можно переносить между пользовательскими project folders.
- `Unsorted` используется по умолчанию, если отдельная папка проекта не выбрана.

## Selective MultiClip regeneration

В MultiClip LongMedia может повторно использовать approved cached clip states и перегенерировать только invalidated range. Dependency checks консервативны: если incoming/outgoing continuation несовместим, invalidated range расширяется, а небезопасный cached suffix не используется как будто он валиден.

При активном Latent Hi-Res display state и continuation state могут иметь разную геометрию, поэтому runtime сохраняет данные и для видимого результата, и для handoff следующему clip.

## Single и segmented timelines

В single/segmented активный TAKE финализируется из final stitched AV. Это не меняет ownership MultiClip selective cache: пользовательский TAKE остаётся верхнеуровневой ревизией, а runtime clip pointers остаются служебной деталью внутри `_runtime`.

## Refine / Latent Hi-Res

Selective regeneration не меняет контракт Refine. Stage 2 — video refiner; аудио Stage 1 сохраняется точно. Большие high-resolution Stage 2 passes могут выполняться bounded temporal windows.

См. также:

- [Director — полное руководство](DIRECTOR_GUIDE_RU.md)
- [Integrated Refine и Latent Hi-Res](TWO_PASS_LATENT_HIRES_REFINER_GUIDE_RU.md)
- [Режимы работы](MODES_GUIDE_RU.md)
