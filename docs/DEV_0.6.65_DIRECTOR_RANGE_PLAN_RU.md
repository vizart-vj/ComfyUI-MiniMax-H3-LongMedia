# LongMedia Director 0.6.65-DEV — диапазонная генерация и мосты

> Исторический план разработки 0.6.65-DEV, сохранённый для контекста. Текущие действия и ограничения описаны в [руководстве 0.6.69](DIRECTOR_REGENERATION_GUIDE_RU.md); незакрытые пункты ниже отражают состояние на момент этого плана.

## Цель

Развить Director как недеструктивный монтажный и генерационный граф: продолжать клип, перерендеривать один клип, заменять диапазон между маркерами и генерировать мост между двумя клипами, сохраняя утверждённый контент вне цели и его аудио.

## Карта возможностей и порядок

| Область | Ответственность | Зависимости |
|---|---|---|
| `range-contract` | Проверяемая операция генерации: тип, целевой клип/диапазон, границы, версии исходных TAKE, политика аудио | — |
| `masked-range-render` | Точные H3 AV noise masks, чтение исходного TAKE, splice результата и безопасный cache lineage | `range-contract` |
| `timeline-range-ui` | Маркеры CUT, выбор диапазона, понятная подсветка цели и preview до принятия | `range-contract` |
| `two-sided-bridge` | Контекст от двух независимых родителей, generation gap/replace между клипами | `range-contract`, `masked-range-render` |
| `portable-lineage` | Переносимый MCTX pin/parent contract и повторное использование результатов | `two-sided-bridge` |

Порядок: контракт → диапазон внутри одного GENERATED MAIN → мост между GENERATED clips → MEDIA pins → preview/accept и переносимая lineage → унификация extend/prepend/refine.

## Сверка с OBVPM

Сверка выполнена по [upstream README](https://github.com/chanon/comfyui-obvpm-timeline) и его [H3 node reference](https://github.com/chanon/comfyui-obvpm-timeline/blob/main/docs/h3-nodes.md). Переносим архитектурные принципы, не чужую реализацию:

- Утвержденный клип хранит native H3 AV latents и lineage в sidecar. Если sidecar валиден, продолжение не должно делать pixel → VAE → latent круг.
- Extend / prepend / bridge описываются pin-ами: источник, исходный диапазон, сторона целевого клипа и точный frame-grid. Bridge имеет независимые левый и правый pins; свободная середина остается единственной областью генерации.
- Если native motion context отсутствует, pixel/VAE handoff — явный путь с пониженной точностью, а не тихая подмена native latents.
- Новый take сначала проверяется в контексте соседей и только потом принимается в монтаж. Preview измеряет стык; отклонение не должно переписывать исходные takes.
- Монтаж и генерация — отдельные операции. Assembly использует lineage/cut positions, а общий refine обрабатывает соседние клипы в одном временном контексте с conditioning-владельцем для каждого окна.

В DEV реализованы native H3 range edit и мост только между уже утвержденными GENERATED takes. Результат bridge остаётся отдельным pending TAKE с превью, пока пользователь явно не нажмёт Accept; принятие повторно проверяет все frozen source revisions и обе AV-границы, затем транзакционно ребейзит metadata и переключает активную ветку. MEDIA pins, pixel-grade fallback и единый windowed refine всего монтажа пока не реализованы.

## Контракт операции

Старые Director JSON остаются валидны. Одноразовый запрос хранится в `regeneration` и не подменяет исходные assets:

```json
{
  "mode": "range",
  "clip_id": "main-clip-id",
  "request_id": "request-id",
  "range": {"start_frame": 24, "end_frame": 72},
  "preserve_audio": true
}
```

`start_frame` включён, `end_frame` исключён; координаты локальны относительно видимого MAIN-клипа и задаются на фиксированной сетке H3 24 fps. Планировщик проверяет диапазон и H3 temporal grid, затем переводит его в индексы латентных шагов. `CUT` остаётся недеструктивной меткой. Knife остаётся физическим разрезом.

Только `mode=range` запускает masked edit. Обычный RUN, `clip`, `from_here`, Recast и workflow без Director сохраняют текущие контракты. Отсутствующий или несовместимый исходный TAKE приводит к понятной ошибке, а не к тихому full render.

## Модель H3 и аудио

Для masked edit каждый AV stream получает свою маску: 0 означает сохранить токен, 1 — генерировать. VIDEO маска адресуется по native H3 latent steps; границы диапазона должны совпадать с допустимыми шагами. AUDIO использует свою 40 Hz временную сетку. `preserve_audio=true` сохраняет audio latent клипа целиком; иначе аудио семплируется только в целевом диапазоне.

Диапазонная операция использует активный TAKE как x0 и источник точного восстановления за пределами диапазона. Входные и выходные формы, dtype, устройство, hidden overlap, resolution и Hi-Res profile проверяются до sampling. После Hi-Res outside-range VIDEO/AUDIO восстанавливаются из соответствующих веток исходного TAKE, чтобы refinement не переписал утверждённую часть.

## Границы реализации

- Реализованный вертикальный срез: диапазон внутри одного GENERATED MAIN-клипа с совместимым активным TAKE, MultiClip cache и точным сохранением контента за границами.
- Реализуемый в этом DEV-срезе: вставленный bridge между двумя соприкасающимися GENERATED MAIN-клипами. Рендерится только bridge; оба края блокируются native AV TAKE-ами; остальные активные clips проверяются и повторно используются без sampling.
- Bridge требует непрерывного MultiClip, валидных активных соседних revisions, неизменных prompt/reference/audio/render semantics, совпадающих tensor shapes/dtypes и доступного свободного H3 temporal middle. Нарушение любого условия останавливает RUN до sampling.
- Далее: MEDIA pins, сохранение встроенного аудио для импортированного источника, переносимая MCTX lineage и refine assembled timeline окнами.
- Для артефактов 0.6.65-DEV использовать отдельное имя `ComfyUI-MiniMax-H3-LongMedia-0.6.65-DEV-director-range-bridge.zip`; release 0.6.64 и его исходники не изменять.

## Текущее состояние

- Реализован и покрыт CPU regression-тестами контракт H3 frame-span, snapping, AV masks, аудио-политики и точного post-refine splice.
- Setup валидирует одноразовый range request; backend требует активный source revision, неизменённую геометрию, текущие соседние клипы и доказанные граничные швы. Неявного full render при ошибке нет.
- CUT context menu предлагает интервал между соседними маркерами/краем клипа, показывает подтверждение с фактическим native H3 диапазоном и выбором аудио-политики.
- Sampled clip единственный в `render_indices`; исходный revision и range report сохраняются в новой TAKE, старые seam IDs переносятся после точного восстановления обеих сторон.
- Insert Bridge доступен с MultiClip toolbar и BASE context menu; он вставляет отдельный generated clip на H3 17-frame сетке, сохраняет длительность соседей, сдвигает downstream MAIN и учитывает Ripple для остальных слоёв.
- Bridge preflight проверяет активные revisions, формы continuation/display tensors и dtype всех active generated takes, clip-owned semantics и H3 форму target. До изменения MAIN индексов неявные seed действующих generated takes фиксируются в JSON точной десятичной строкой; проверка принимает только seed, равный сохранённому effective seed. Семплируется только новый bridge; post-refine и Hi-Res восстанавливают оба AV края.
- Bridge сначала сохраняется как pending TAKE: active/editor pointers и approved source manifests остаются нетронутыми. После decode его preview sprite/audio прикрепляются к кандидату, а обе границы повторно проверяются по native AV hash.
- Accept — отдельная транзакция под общим reentrant lock проекта: проверяет точный порядок клипов, frozen revisions и набор generated источников; метаданные ребейзятся без перечитывания/копирования continuation tensors, обе границы доказываются заново, seam lineage связывается и только после этого кандидат становится active. RUN/Restore/удаление для того же проекта не могут вклиниться в commit. Ошибка откатывает metadata и все render/editor pointers. Pending TAKE нельзя активировать/дублировать через обычные операции и он исключён из поиска cached branches.
- Status API передаёт дополнительное `effective_seed_text`, чтобы значения uint64 не округлялись при JSON → JavaScript переходе. Обычное числовое поле оставлено для совместимости.
- Проверен контракт pending save/restore guard, атомарное acceptance с точными hash boundaries и rollback при провале второй границы. JavaScript syntax, seed precision, Python compile и diff checks проходят. Полный запуск на ComfyUI с настоящей моделью/проектом ещё не выполнен; MEDIA pins, MCTX lineage, общий extend/prepend/refine и DEV архив остаются следующими этапами.

## Критерии приёмки

1. Внутриклипный range RUN семплирует только выбранный GENERATED-клип и не меняет кадры/аудио латентно вне диапазона.
2. Границы режутся по native H3 grid; фактическая подсвеченная/показанная область совпадает с переданной моделью.
3. `preserve_audio=true` сохраняет исходный audio latent побитово; при выключенном флаге его меняют только аудио-шаги диапазона.
4. Изменение prompt/RefMod, формы TAKE, geometry или H3 grid не может незаметно превратить range edit в перегенерацию всего проекта.
5. Повторный обычный RUN читает принятый активный TAKE; старый TAKE можно восстановить через существующую историю.
6. Мост A→B между GENERATED takes семплирует только вставленный clip, сохраняет оба исходника, exact AV overlap и длительность timeline.
7. Вставка bridge не меняет фактический seed уже подтверждённых suffix takes, даже если исходный seed был неявным или превышает `2^53`.
8. Обычный single, MultiClip, MEDIA continuation, Recast, RefMod, Hi-Res и отключённые клипы сохраняют прежнее поведение, если range/bridge operation не активна.

## Рабочие задачи

- [x] Добавить чистые helpers для H3 frame-step mapping, range masks и range splice; покрыть граничные случаи regression-тестами.
- [x] Расширить Director regeneration DTO с backward-compatible валидацией range и audio policy.
- [x] Добавить действие UI для диапазона между CUT-маркерами, подтверждение фактических snapped boundaries и сохранение request_id.
- [x] Подключить active continuation/display TAKE к маскам; жёстко валидировать geometry, shapes, dtype/device, cache и Hi-Res.
- [x] Сохранять source revision и range contract в новой TAKE; переносить старые seam IDs после splice.
- [x] Реализовать двухсторонний native bridge между GENERATED clips: UI insertion, pin validation, selective sampling, exact post-refine restore и verified seam lineage.
- [x] Фиксировать неявные effective seed перед bridge insertion без потери точности uint64; валидировать actual loaded TAKE tensor shape/dtype до sampling.
- [ ] Реализовать двусторонний bridge к MEDIA с явным pixel-grade VAE fallback и сохранением/выбором embedded audio.
- [x] Добавить pending-result TAKE с отдельным preview sprite/audio и явными Accept/Discard; принятие проверяет обе границы и атомарно обновляет lineage, а генерация не меняет активную ветку до принятия.
- [ ] Реализовать переносимую MCTX lineage с pin-ами, исходными frame ranges и parent revisions.
- [ ] Добавить общий windowed refine / seam repair для assembled timeline, с корректным conditioning для каждого окна.
- [ ] Собрать отдельный DEV zip; production custom_nodes не перезаписывать.
