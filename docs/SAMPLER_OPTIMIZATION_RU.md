# Sampler, VRAM и производительность

Рекомендации для текущей ветки LongMedia 0.6.41.

## Production default

Начинайте с:

```text
sampler_mode   = auto
memory_mode    = auto
attention_mode = auto
```

Оставляйте ComfyUI Dynamic VRAM включённым.

Не запускайте production workflow с:

```text
--disable-dynamic-vram
```

LongMedia рассчитывает на coordinated dynamic residency большой модели.

## Memory Modes

### `auto`
Рекомендуется. Выбирает профиль по model size, quantization/backend, VRAM и packed sequence geometry.

### `normal`
Используйте только если model + activation workspace уверенно помещаются.

### `low_vram`
Более жёсткие activation/residency limits и агрессивнее chunking.

### `ultra_low_vram`
Последний вариант для очень ограниченного GPU или экстремальной sequence geometry. Ожидайте дополнительный transfer overhead.

## Attention Modes

### `auto`
Рекомендуется. LongMedia выбирает совместимый exact/bounded path из активного backend и geometry.

### `existing`
Сохраняет установленную/выбранную ComfyUI attention family.

С текущим Comfy Kitchen INT8 LongMedia умеет перейти на **exact query-streaming**, когда full fused QKV сам по себе не помещается на 16 GB-class GPU. Это сохраняет контракт Comfy Kitchen, а не подменяет его Sol.

### `sol`
Использует embedded H3 Sol path с bounded long-sequence execution.

### `scheduled_sol`
Использует Sol с явным sigma/tau schedule.

## Native INT8 на 16 GB-классе

LongMedia рассматривает native INT8 H3 на GPU <=18.5 GB как constrained-residency case.

В этом режиме:

- speculative `prefetch_dynamic_vbars` отключён;
- blocks fault-on-demand;
- следующий block не резервирует competing VBAR/cast destination, пока activations текущего ещё живы;
- full-model RAM pinning отклоняется, если съедает слишком много физической памяти или оставляет мало headroom.

Это защищает и VRAM, и system RAM в Windows/portable ComfyUI.

## Повторные Queue

Изменение только seed может заставить ComfyUI повторно использовать cached Setup output/guider state.

Поэтому LongMedia создаёт execution memory boundary при каждом входе в Sampler до `prepare_sampling()`:

- синхронизирует активную CUDA работу;
- чистит stale prefetch queues;
- reset cast buffers;
- reset AIMDO/VBAR watermark state;
- снимает предыдущую зарегистрированную model residency;
- освобождает dead allocator cache;
- очищает transient CUDA references, которые могли пережить предыдущий Queue через cached runtime state;
- очищает persistent FastH3/VSA CUDA geometry cache между executions.

Latent Hi-Res upscaler возвращается на CPU через `try/finally`, включая OOM/exception/cancel paths.

Второй/третий одинаковый Queue должен начинаться из тех же residency assumptions, что первый, а baseline allocated VRAM не должен монотонно расти из-за наших transient caches.

## Manual VRAM controls

Sampler сохраняет детальные controls для точечного tuning:

```text
mlp_chunk_tokens
sol_qkv_chunk_tokens
sol_out_proj_chunk_tokens
vram_activation_reserve_mb
inter_block_vram_guard_mb
inter_block_guard_cooldown_blocks
inter_block_guard_emergency_mb
late_block_guard_start
late_block_guard_target_mb
step_boundary_cleanup_mb
```

Не уменьшайте все chunks заранее. Начните с Auto и меняйте по одному pressure point.

## Latent Hi-Res

Spatial latent upscale быстро увеличивает token count.

Для 16 GB-class GPU:

- начинайте с `latent_hires_scale=1.2–1.5`;
- оставляйте `memory_mode=auto`;
- держите reference payload разумным;
- двигайтесь к 2× только после стабильного меньшего high-resolution pass.

См. [Integrated Refine и Latent Hi-Res](TWO_PASS_LATENT_HIRES_REFINER_GUIDE_RU.md).

## Reference Budget

Большие Picture, Video и Audio references добавляют packed conditioning rows.

Для длинных или memory-constrained runs начните с:

```text
reference_budget = low
```

Повышайте только когда дополнительная identity/style/reference fidelity действительно нужна.

## Segmentation против одного гигантского pass

Для одной непрерывной сцены короткие fixed segments часто дают лучший throughput/stability, чем огромная packed sequence с extreme streaming.

Отправные точки:

- 16 GB: 5–10 s в зависимости от references/edit complexity;
- 12 GB: 5–8 s с low reference budget;
- 8 GB: 4–6 s, ожидайте transfer-bound execution.

Это рекомендации, а не model limits.

## Windows Triton / RTX 50-Series

LongMedia содержит process-local Triton TinyCC compatibility bootstrap для embedded Windows Python, где bundled TCC не находит WinAPI headers.

`site-packages` постоянно не патчится.

## Preview compatibility

Сломанный внешний latent-preview hook не должен убивать успешный H3 denoise. LongMedia изолирует VideoHelperSuite animated-preview exceptions только когда traceback реально приходит из VHS preview code; посторонние callback errors остаются fatal.

## Диагностика OOM

Полезно фиксировать:

```text
GPU total/free
PyTorch allocated/reserved
AIMDO/VBAR residency
pinned RAM
active attention backend
packed sequence geometry
failure before block 0 / QKV / MLP / final output
```

Ошибка внутри `prefetch_queue_pop` отличается от fused-QKV allocation failure и должна рассматриваться как transport/residency issue, а не лечиться сменой attention math.
