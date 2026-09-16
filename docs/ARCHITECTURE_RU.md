# Архитектура LongMedia

Документ описывает текущую runtime-архитектуру LongMedia. Имена параметров и внутренних идентификаторов оставлены на английском, чтобы они совпадали с UI и кодом.

## Семантический контракт Setup

Публичный Setup больше не определяется одним монолитным `workflow_mode`.

Новые workflow используют независимые controls:

```text
control_mode
h3_mode
timeline_mode
duration_source
audio_mode
```

`workflow_mode` сохраняется внутри только ради совместимости со старыми workflow.

См. [Режимы LongMedia](MODES_GUIDE_RU.md).

## Conditioning families

`h3_mode` выбирает семью H3 conditioning:

- `t2va` — чистая text-to-video/audio генерация;
- `fl2va` — нативные first/last frame anchors;
- `ref2va` — нативные Picture/Video/Audio references;
- `hybrid` — frame anchors плюс LongMedia reference behavior;
- `video_ref_edit` — редактирование/замена на основе source video.

Conditioning family не зависит от способа построения таймлайна.

## Timeline engine

`timeline_mode` определяет владельца таймлайна:

```text
single
segmented
multiclip
```

### Single

Один target AV latent и один логический movie timeline.

### Segmented

LongMedia создаёт внутренние сегменты фиксированной длительности и переносит native H3 continuation context между ними. Режим предназначен для одного непрерывного semantic prompt, а не для storyboard cuts.

### MultiClip

Planner владеет clip prompts, durations, names и опциональными per-clip seeds. Cameras добавляет недиегетическую кинематографию, сохраняя идентичность клипа через стабильный `clip_id`.

Planner авторитетен только при `timeline_mode=multiclip`.

## Общий AV latent

MiniMax H3 использует nested AV latent:

```text
video: [B, 24, T, H, W]
audio: [B, 32, 2, T40]
```

LongMedia сохраняет нативную MiniMax temporal lattice и проверяет согласованность AV duration на границах сборки.

## References и source editing

Picture, Video и Audio — отдельные модальности.

ComfyUI video IMAGE batch никогда не содержит soundtrack. Звук source video подключается отдельно.

Для `video_ref_edit` LongMedia может построить нативный paired Video1+Audio1 source-performance reference и одновременно использовать Audio1 как авторитетный timing/output source в preserve-style режимах. `lip_sync` рассматривает Audio1 как независимый dub, поэтому новая речь не обязана совпадать с оригинальным soundtrack Video1.

Audio2/Audio3 остаются независимыми prompt-addressable references.

## Слой Cameras

Long Media Cameras — отдельный недиегетический direction layer.

Рекомендуемое владение в MultiClip:

```text
Planner scene/action prompts
        │
        ▼
Cameras cinematography compiler
        │
        ▼
Setup conditioning
```

При наличии camera guidance camera directives удаляются из Planner text до добавления compiled camera instructions. Это предотвращает конфликт двух подсистем за framing и motion.

## Director temporal controls

Director хранит общий scene/subject/reference presentation на весь MAIN shot. Частичные CAMERA и EMBEDDING blocks добавляются как отдельные временные controls и не создают новые H3 passes сами по себе.

Camera и embedding boundaries независимы. Глобальный authored clock остаётся 24 fps; локальные sampler/refine windows получают проекцию глобального времени, а не собственный новый таймлайн.

## Integrated Refine / Latent Hi-Res

Текущий Sampler содержит внутренний Stage 2, управляемый `refine_enabled`.

Контракт:

1. MAIN Sampler получает полный Stage-1 x0;
2. при Latent Hi-Res learned upscaler изменяет только video latent;
3. Stage-1 audio сохраняется как авторитетный exact passthrough;
4. Refine использует отдельный `refine_sigmas` schedule;
5. длинные/high-res video latents обрабатываются temporal windows;
6. финальное audio восстанавливается точно из Stage 1.

Refine не должен быть audio refiner.

См. [Integrated Refine и Latent Hi-Res](TWO_PASS_LATENT_HIRES_REFINER_GUIDE_RU.md).

## Dynamic VRAM и exact attention

LongMedia координирует lifetime activations с текущими ComfyUI Dynamic VRAM/AIMDO.

Ключевые механизмы:

- streamed/chunked transformer MLP и output projections;
- embedded Sol paths для bounded long-sequence execution;
- exact Comfy Kitchen EXISTING query streaming, когда full fused QKV не помещается;
- inter-block и step-boundary VRAM guards;
- guarded native INT8 residency на constrained GPU;
- sampler-entry memory isolation для cache-driven reruns;
- очистка transient CUDA references между повторными Queue;
- RAM-pressure-aware pinned-host-memory policy.

На constrained native INT8 системах speculative dynamic-VBAR prefetch ограничивается, чтобы следующий block не резервировал competing transfer destination, пока текущие activations ещё живы.

## Совместимость моделей

LongMedia содержит изолированные compatibility paths для:

- stock MiniMax H3;
- поддерживаемых native INT8/W4A8 ComfyUI weights;
- H3ddle/PulpCut FastH3 VSA;
- Kijai FastVideo VSA.

FastH3/FastVideo adaptations определяются структурно и fail closed, если trained contract не выполняется. Runtime state сбрасывается при возврате к обычным H3 checkpoints.

## Reconstruction

Video Reconstructor строит source-video edit plans поверх той же H3/Ref2VA основы. Более поздние reconstruction revisions добавляют detail-recovery passes, сохраняя low-frequency source geometry и AV timing contracts.

## Loop Closure

Loop Closure независим от timeline/conditioning selection. Он регенерирует/притягивает хвост к opening macro-state в latent/H3 space, а не делает RGB crossfade.

## Совместимость

Legacy class identifiers вида `MiniMaxH3LatentLab...` намеренно остаются зарегистрированными, чтобы старые ComfyUI workflow продолжали открываться.

Legacy `workflow_mode` мигрируется в semantic Setup controls на границах load/runtime.
