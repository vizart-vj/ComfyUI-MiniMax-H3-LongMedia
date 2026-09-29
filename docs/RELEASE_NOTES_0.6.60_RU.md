# ComfyUI-MiniMax-H3-LongMedia 0.6.60

**Версия пакета остаётся 0.6.60.** Этот релиз собирает последнюю линию Director и runtime fixes в одном ZIP и приводит документацию в соответствие с параметрами фактического билда.

## Главное

- **Полный справочник параметров на русском и английском.** Описаны INPUTS и controls Setup, Director, RefMods, Planner/Cameras, Video Reconstructor, Sampler и Decode; отдельно отмечены условия видимости, скрытые служебные поля и legacy-поля, которые больше не действуют.
- **RefMod Sources builder.** Можно добавлять до 128 изображений, видео и аудио, вставлять картинку из буфера обмена и видеть оценку visual tokens для источника и набора. Параметры short-edge resolution, visual token cap, grid, clip frames, voice seconds и Compressed refinement доходят до нужных этапов VAE-кодирования.
- **RefMod не подменяет собой style transfer.** Concept Type добавляет общую положительную prompt-подсказку; Description остаётся описанием библиотеки. Strength управляет силой native reference на интервале timeline. Эти поля не меняют смысл латента и сами по себе не гарантируют перенос стиля.
- **RefMod attention path.** Полноинтервальный native RefMod с Strength=1 сохраняет выбранный H3 attention backend, включая zero-copy SLA, когда merged mask является identity. Сохранённый ComfyUI SLA override больше не теряется при клонировании guider options.
- **Character RefMod в MultiClip.** На target-video queries поддерживается авторитет идентичности через границы клипов, не меняя маршрутизацию аудио и других RefMod-концептов.
- **Latent Hi-Res + Refine на стыках.** Следующий клип получает refined overlap предыдущего и фиксирует этот контекст на Stage 2; Stage 1 continuation и точный passthrough аудио сохранены.
- **Director timeline.** Gap-кнопки существуют только в реальных промежутках и показывают их длительность; Fit вписывает таймлайн при любом horizontal scroll. Cut marker живёт внутри своего клипа; Cut-to-cursor отличается от Knife split; trim handles показывают source time. Добавлены расширенное меню сортировки Takes и сохранение позиции галереи при выборе тейка.
- **Память предпросмотра.** Director освобождает отделённые декодеры filmstrip при обновлении, выгружает старые TAKE thumbnails, использует ленивую загрузку галереи и ограниченный LRU-кэш изображений. Это касается preview/media cache Director; VRAM генерации регулирует Sampler.
- **Сохранение motion_repair.** Выбор больше не сбрасывается после сохранения и повторной загрузки workflow.
- **Пресеты LongMedia Sampler.** В верхней части ноды можно создавать, перезаписывать и удалять собственные пресеты. Они сохраняются в workflow; строка статуса показывает, когда настройки отличаются от выбранного пресета.
- **Обычный MEDIA-only рендер.** BASE timeline, состоящий только из immutable MEDIA клипов, обходит подготовку H3-модели и напрямую использует media assembly.

## Быстрая проверка важных параметров

- Для обычного запуска Sampler оставляйте sampler_mode=auto, memory_mode=auto, attention_mode=auto.
- Для отдельного RefMod нажмите ENCODE после настройки источников и полей; подготовка VAE-артефакта не запускает DiT/Sampler.
- Refinement steps относятся только к RefMod COMPRESSED. Refine steps самого Sampler берутся из отдельного Refine Sigmas; старое поле refine_steps оставлено скрытым для workflow-совместимости.
- Director RefMod работает только внутри интервала RefMod clip. Для сквозной референции используйте растяжение на весь timeline.
- motion_repair — необязательный постпроход: off сохраняет прежний путь, auto — рекомендуемая отправная точка при проблемном быстром движении.

## Документация

- [Полный справочник параметров — RU](PARAMETER_REFERENCE_RU.md)
- [Complete Parameter Reference — EN](PARAMETER_REFERENCE_EN.md)
- [Director — руководство](DIRECTOR_GUIDE_RU.md)
- [Sampler, VRAM и производительность](SAMPLER_OPTIMIZATION_RU.md)

Архив содержит package VERSION 0.6.60; изменения здесь документируют текущий билд и не меняют номер версии.
