# ComfyUI-MiniMax-H3-LongMedia 0.6.40

0.6.40 — консолидированный production-релиз ветки Director 0.6.

## Director

- Director работает с `t2va`, `fl2va`, `ref2va`, `hybrid` и нативным `video_ref_edit` routing.
- MAIN timing, CAMERA/AUDIO/embedding tracks, Ripple editing, MultiClip markers, segmented controls, политика resolution/aspect и Program Monitor объединены в одном интерфейсе.
- FIRST, LAST и REF остаются разными семантическими ролями. Один и тот же source может использоваться как FIRST+LAST для loop/endpoint anchor.
- Хранилище TAKE теперь TAKE-centric. CREATE создаёт активный пустой TAKE; следующий успешный render заполняет именно его. Restore атомарно возвращает полный сохранённый Director state.
- Ширина TAKE gallery, zoom timeline, высота inspector и размеры редактируемых text areas сохраняются вместе с workflow.

## Refine и Latent Hi-Res

- Одна Long Media Sampler содержит MAIN Stage 1 и optional internal Stage 2, включаемый `refine_enabled`.
- Stage 2 использует отдельный вход Refine Sigmas.
- Длинный/high-resolution refine может работать bounded temporal windows с overlap-save core ownership, чтобы соседние окна не записывали один seam дважды.
- Новый свитч **windowed refine** появляется только при включённом Refine. ON сохраняет memory-bounded adaptive path; OFF принудительно рефайнит полный latent одним Stage 2 pass и после OOM не откатывается обратно на windows.
- Refine является video-authoritative: аудио Stage 1 сохраняется точно и никогда не денойзится заново Stage 2.

## Segmented и chained execution

- Segmented continuation использует точный latent tail предыдущего сегмента как frozen target prefix и детерминированный seed offset для каждого сегмента. Это устраняет короткий повторяющийся motion-cycle от дублированного motion context/noise phase.
- Chained Sampler motion-context layout поддерживает смешанные visual и audio conditioning segments.

## Память и runtime

- Repeat-Queue cleanup разрывает только LongMedia-owned transient CUDA references, которые могли пережить предыдущий запуск через cached guider state.
- FastH3/VSA execution geometry cache теперь ограничен одним execution.
- Residency Latent Hi-Res защищена `try/finally`: cached upscaler возвращается на CPU после success, error, OOM или cancel.
- Существующая политика ComfyUI Dynamic VRAM сохранена; релиз не добавляет безусловную выгрузку основной модели после каждого sampling.

## VAE и совместимость

- MiniMax H3 VAE decode умеет пакетировать нативные spatial tiles с безопасным OOM fallback.
- Legacy class identifiers `MiniMaxH3LatentLab...` остаются зарегистрированными для совместимости старых workflow.

## Документация

В release archive остаётся только актуальная двуязычная пользовательская документация (`*_EN.md` и `*_RU.md`). Исторические release notes по каждой промежуточной сборке и development audit scripts из пользовательского архива удалены; история разработки сохранена в `CHANGELOG.md`.
