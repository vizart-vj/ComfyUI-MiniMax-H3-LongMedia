# Проверка LongMedia 0.6.64

> Исторический отчёт проверки 0.6.64. Для текущего состояния релиза см. [Release Notes 0.6.69](RELEASE_NOTES_0.6.69_RU.md).

## База и рабочая копия

- База: архив v0.6.63, SHA-256 `15d8b60aab83cf48ceb37ed96911693dd68cf20f36085777232800e045723c2f`.
- Установленная копия: `D:/ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-LongMedia`, VERSION 0.6.63. До исправления `nodes.py`, `director_plan.py`, `director_cache.py`, `web/longmedia_director.js` совпадали с архивной v0.6.63 побайтно по SHA-256.
- Изменения: отдельная рабочая копия `E:/AI/__WORK_NODES__/LongMedia/working_0.6.64_incremental_append/ComfyUI-MiniMax-H3-LongMedia`, Git tag `baseline-v0.6.63`.
- Установленный runtime, существующий `src`, модели, пользовательские workflows/TAKE автоматически не менялись. ComfyUI не запускался и не перезапускался.

## RCA по живому логу и host core

В `comfyui_8199.log:486–517` воспроизведено исключение `Partial Director RUN requires geometry-compatible active TAKEs for every GENERATED clip; missing or outdated clip indices: 1, 2`. Host `execution.py` вызывает функцию custom node; сбой происходит в Director preflight до H3 forward, не в устройстве AV-контейнера.

Проверен `comfy/ldm/minimax/model.py:_forward`: отдельные VIDEO/AUDIO streams, batch 1, native PackedLayout по text/video/audio geometry. Исправление меняет только выбор render/replay/boundary dependencies и semantic validation. Оно сохраняет нативные VIDEO rank 5, AUDIO rank 4, structural NestedTensor, tensor dtype/device handling и VAE/continuation machinery.

RED→GREEN проверки отдельно поймали и подтвердили исправления:

1. Новый enabled suffix ошибочно требовал старые TAKE в validator.
2. Production Sampler materialization повторно требовал TAKE нового target.
3. SINGLE→MultiClip append инвалидировал первый клип по compiler-owned outgoing flag.
4. `run_enabled` менял semantic fingerprint.
5. Старые MultiClip хеши требовали безопасной совместимости после изменения semantic key.
6. Исчезнувший cached dependency молча расширял clip RUN до полного пересчёта.
7. Setup без RefMods не готовил conditioning для расширенного диапазона.
8. Disabled single-shot prefix со старым seed нельзя было продолжить при randomize seed; проверено также, что его старые tensors не сертифицируются с новым seed.

## Фактические результаты

Среда тестов: portable Python 3.13.14, PyTorch 2.13.0+cu130, safetensors 0.8.0. pytest установлен отдельно в scratch; пакетные зависимости активного ComfyUI не изменялись.

- Новые focused tests: **15 passed, 0 failed**. Количество и отсутствие ошибок проверены программно из JUnit XML полного запуска.
- Полный набор v0.6.64: **288 passed, 7 failed, 8 subtests passed**, 73.10 s.
- Неизменённая архивная v0.6.63: **272 passed, 7 failed, 1 skipped, 8 subtests passed**, 39.83 s. Различие в skipped count не используется как доказательство нового исправления.
- Семь падающих test IDs одинаковы у baseline и исправленной версии. Новых падений в выполненном полном наборе не обнаружено; полный набор НЕ объявлен полностью зелёным.
- Python syntax: **46 файлов**, PASS.
- `node --check`: **7 JS файлов**, PASS.
- JSON parse: **3 workflow JSON**, PASS.
- `test_sampler_preset_state.js`: PASS (вывод `Sampler preset state tests passed.`).
- `test_motion_repair_widget_persistence.js`: exit 0.

### Семь унаследованных падений

- `test_conditioning_contract.py::ConditioningContractTests::test_required_positive_branch_cannot_be_silently_dropped` — старый regex ожидает `None`, актуальный exception сообщает `missing`.
- `test_director_refmod_contract.py::TestRefmodSourceComposerRefinementSteps::test_js_normalize_refmod_covers_required_keys` — отсутствует `REQUIRED_KEYS` у test class.
- `test_director_refmod_contract.py::TestRefmodSourceComposerRefinementSteps::test_dead_refmod_fields_stay_removed` — устаревшая проверка substring `resolution` конфликтует с `resolution_short_edge`.
- `test_director_refmod_contract.py::TestRefmodFormFields::test_all_form_fields_present` — expectation старого поля `Spatial Grid`.
- `test_director_refmod_contract.py::TestRefmodPanelSections::test_panel_is_mounted_in_body_grid` — expectation старого UI mount path.
- `test_director_refmod_contract.py::TestRefmodPanelSections::test_panel_is_collapsible_and_persisted` — expectation старого toolbar path.
- `test_refmod_integration_contract.py::test_inactive_picture_is_available_to_refmod_without_becoming_normal_image_input` — AST harness не определяет `Any` при standalone exec.

Эти проверки/несвязанный RefMod UI не правились и не скрывались xfail/skip.

## Сценарии, подтверждённые offline

| Сценарий | Production preflight |
|---|---|
| Cached Clip 1, включены новые Clip 2–3 | render `{1,2}`, replay `{0}`, historical target locks не требуются |
| Старый completed SINGLE Clip 1 + новый Clip 2 | render `{1}`, первый TAKE revision сохранён |
| Старые MultiClip TAKE 1–2 из v0.6.63 + новый Clip 3 | render `{2}`, replay `{0,1}` |
| Disabled SINGLE prefix + новый queue seed | render `{1}`; старый TAKE остаётся frozen |
| Последовательные enabled targets перед frozen right neighbor | lock только последнего enabled target, не между ними |
| Все текущие TAKE совместимы | render `{}`, cached replay |
| Disabled prefix с изменённой длительностью | явный geometry error; не implicit resampling |
| Изменён реальный prompt при all-enabled RUN | legacy hash не мигрируется, зависимый хвост инвалидируется |
| Cache initialization/read failure | явный отказ, не implicit full render |

`render_indices` — нулевые индексы. Тесты используют real on-disk safetensors и AV tensors и исполняют production preflight, но не нейронный diffusion sampler/VAEs. **GPU end-to-end генерация видео новой версией не выполнялась.** Повторный final decode/export cached timeline не равен повторному diffusion.

## Воспроизведение focused tests

Из исходников пакета, Python-окружением с torch/safetensors:

```text
python tests/test_incremental_append.py -v
```

Полный набор запускался через `pytest.main` с `--import-mode=importlib`. Изолированный запуск одного файла через pytest может попытаться загрузить package `__init__.py` вне ComfyUI; direct unittest-команда выше не требует host startup.
