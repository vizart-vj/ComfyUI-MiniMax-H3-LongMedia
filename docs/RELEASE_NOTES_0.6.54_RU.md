# ComfyUI-MiniMax-H3-LongMedia 0.6.54

Публичный релиз построен поверх GitHub baseline `v0.6.50` и объединяет проверенные runtime-изменения, сделанные после него.

## Selective regeneration теперь действительно selective

- `Regenerate Clip` может семплировать только выбранный GENERATED clip, пока approved prefix/suffix TAKE остаются cache hits.
- `Regenerate From Here` семплирует только выбранный clip и зависимый GENERATED suffix.
- Добавление нового clip использует continuation предыдущего TAKE и не пересэмплирует approved history.
- Cached clips пропускают повторный text/reference conditioning при selective Setup preparation.

## Native continuation для generated и импортированного видео

Director BASE blocks формально разделены на `GENERATED` и immutable `MEDIA`.

```text
GENERATED -> GENERATED
    saved native TAKE continuation -> one fresh child sample

MEDIA -> GENERATED
    external tail only -> VideoVAE tail encode -> native frame-0 H3 guide -> one fresh child sample
```

Импортированный MEDIA никогда не становится H3 render unit, не копируется в target x0 и не проходит full-video VAE/diffusion roundtrip только ради продолжения. Trimmed/reused media продолжается от конца видимого BASE range.

## Mixed final assembly и audio

- H3 VideoVAE декодирует только GENERATED runs.
- Исходный MEDIA RGB/audio вставляется в финальный timeline напрямую.
- Hidden overlap удаляется из видимого generated child.
- External tail continuation кэшируется на CPU.
- Per-block audio continuation поддерживает `AUTO`, `CONTINUE`, `FRESH`.
- Mono/stereo pieces собираются по явному `[1,C,L]` contract; mono точно дублируется, когда нужен stereo.

## Quantized runtime и Windows memory

- AUTO policy маршрутизирует INT8/W4A8/NVFP4 по реальному packed storage + activation headroom, а не по BF16-equivalent logical size.
- Явные manual MLP/VRAM controls остаются авторитетными в NORMAL/AUTO-normal.
- Большие Windows safetensors остаются file-backed через ComfyUI `ModelMMAP` / `TensorFileSlice`; eager whole-model RAM materialization не используется.
- Setup conditioning ограничен нужными clips и освобождает TE/transient refs до diffusion sampling без aggressive Windows working-set trimming.
- Admission pinned transfers опирается на реальный projected host-memory headroom.

## Сохранённые контракты

Релиз не меняет H3 quality math, не внедряет SPEED/progressive-resolution и сохраняет Refiner AV contract: Stage 2 остаётся video-only, exact Stage-1 audio восстанавливается. Все существующие Director NLE/TAKE возможности сохраняются.
