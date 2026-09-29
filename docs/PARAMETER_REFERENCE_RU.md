# Справочник параметров LongMedia 0.6.60

Справочник сверён с Python INPUT_TYPES и интерфейсом из ZIP-билда 0.6.60. «Внутреннее» означает сохранённое поле workflow, которое не нужно менять вручную. Поля показываются по условиям режима; скрытое поле может оставаться в старом workflow для совместимости.

## Setup

Узел MiniMax H3 • Long Media Setup соединяет H3-модель, CLIP и обе VAE. При Control = Director авторский таймлайн/промпт/аудио принадлежат Director; явные media-сокеты Setup по-прежнему работают как overrides. Manual показывает диагностические и низкоуровневые параметры.

| Параметр | Значение по умолчанию / допустимые значения | Что делает |
|---|---|---|
| clip, vae, audio_vae | CLIP, VAE, VAE; обязательные сокеты | H3 text encoder и video/audio VAE для conditioning, кодирования и декодирования. |
| control_mode | auto; director; manual | Auto использует селекторы Setup. Director передаёт владение планом, conditioning и аудиополитикой подключённому Director. Manual раскрывает legacy и diagnostic controls. |
| h3_mode | director: auto/t2va/fl2va/ref2va/hybrid/video_ref_edit; Setup manual: t2va/fl2va/ref2va/hybrid/video_ref_edit | Выбирает семейство H3 conditioning, независимо от single/segmented/multiclip. Director Auto выводит режим из ролей FIRST/LAST и референсов. |
| timeline_mode | auto; single; segmented; multiclip | Организация исполнения: один проход, фиксированные continuation-сегменты или авторские клипы Planner/Director. Не выбирает H3 conditioning. |
| prompt / global_prompt | строка, по умолчанию пустая | Общая текстовая инструкция Setup. В Director общая сцена задаётся в его Global Prompt. |
| width, height | 512 × 512; каждый 32–8192, шаг 32 | Геометрия целевого кадра в Setup. В Director итоговую геометрию владеет RES SRC + MP; Setup-размеры там не являются источником canvas. |
| resolution_mode | match; max | Политика целевой геометрии относительно подключённых референсов. |
| reference_budget | low; medium; high; max | Бюджет нативных референсов H3. Это не сила отдельного RefMod. |
| video_fps | 24; 1–120, шаг 1 | Целевая частота кадров. |
| duration_source | auto; manual; audio; video; longest_input | Источник длины цели. Manual использует manual_duration; audio/video — Audio 1/Video 1; longest_input — самый длинный подключённый audio/video. В video_ref_edit Auto следует Video 1. Planner и Reconstructor владеют собственной длительностью. |
| manual_duration | 10 s; 0.1–600 s, шаг 0.1 | Длина при duration_source=manual. |
| segment_seconds | 5 s; 1–60 s, шаг 0.5 | Видимая новая длительность каждого fixed сегмента; continuation-контекст задаётся отдельно и не вычитается из этого времени. Показывается для segmented и Manual. Reconstructor использует собственное одноимённое поле. |
| transition_frames | 22; 5–3600, шаг 17 | Длина контекста/перекрытия между сегментами MultiClip/Segmented. H3-последовательность значений начинается 5, 22, 39… Поле overlap_frames скрыто и оставлено для совместимости. |
| audio_mode | auto; preserve; generate; reference_only; preserve_reference; lip_sync | Auto выбирает политику. Preserve копирует исходный звук в результат, не делая его H3 audio reference. Generate генерирует звук. Reference only использует его только как условие. Preserve reference использует Audio 1 как условие и возвращает оригинальную дорожку. Lip sync фиксирует Audio 1 как целевой speech/music clock и восстанавливает исходную волну. В Director поле владеет Director. |
| video_mode | auto; preserve; transform | Только Manual либо video_ref_edit. Определяет, сохранять ли входное видео или преобразовывать его. |
| motion_repair | off; auto; fluid; strong; default off | Дополнительная temporal-починка быстрого движения после H3. Auto находит перегруженные интервалы; Fluid чувствительнее к тонким хаотичным структурам; Strong применяет более широкую локальную коррекцию. Аудио во время repair заморожено и затем восстанавливается точно. Настройка сохраняется между загрузками workflow. |
| loop_closure_enabled | false | Выполняет дополнительный tail-проход для бесшовного контекста с начальным кадром. |
| loop_closure_frames | 57; 2–720 | Примерная область конца, участвующая в loop closure; число приводится к допустимому H3 frame count. Показывается при включённой функции или в Manual. |
| loop_closure_strength | 0.65; 0–1, шаг 0.05 | Насколько структура конца притягивается к макроконтексту начала; мелкие детали остаются свободными. |
| conditioning_mode | auto_refs; hybrid_first_frame; hybrid_first_last; multiclip_ref2va | Legacy conditioning override, виден только в Manual. Auto refs отправляет изображения как Picture refs; Hybrid first frame использует Image 1 как первый кадр; Hybrid first/last назначает Image 1 и 2 якорями; multiclip_ref2va сохраняет legacy маршрут MultiClip. |
| first_frame_mode | native_keyframe; latent_inject; pixel_override; blend; default latent_inject | Политика первого кадра Hybrid. FL2VA принудительно использует native_keyframe. Остальные доступны по правилам режима. |
| first_frame_denoise | 0.25; 0–1 | Denoise для latent_inject. |
| first_frame_blend_frames | 3; 1–17 | Длина opening blend при first_frame_mode=blend. |
| image_1…image_9 | IMAGE sockets | Нативные Picture references; значение конкретного слота зависит от H3/conditioning режима и Director override. |
| video_1…video_3 | IMAGE batches | Кадры видео, не AUDIO. В video_ref_edit Video 1 — основной источник движения/камеры/композиции. Звук видео при необходимости подключается отдельно в audio_1. |
| audio_1…audio_3 | AUDIO sockets | Нативные аудио-референсы. В lip_sync Audio 1 становится целевой дорожкой и исключается из обычных Ref2VA аудио-референсов. |
| release_guard | true | В Manual отключает рутинные диагностические сообщения; выключите для профилирования и A/B. |
| clip_plan | H3_LONGMEDIA_CLIP_PLAN, optional | Planner/Cameras. Авторитетен для MultiClip, если Director не подключён. |
| director | H3_LONGMEDIA_DIRECTOR, optional | Полный Director-контракт: timeline, камеры, media. Прямые Setup image/video/audio sockets переопределяют соответствующие Director media slots. |
| reconstruction | H3_LONGMEDIA_RECONSTRUCTION, optional | Подключённый Reconstructor сам владеет входным видео/аудио, FPS, окнами и силой реконструкции. |

Внутренние/legacy поля всех узлов: Director director_json; Cameras cameras_json и sync_request; Planner clips_json, multiclip_import_request и multiclip_last_import_source; Setup workflow_mode, generation_mode, overlap_frames и multiclip_json; служебный unique_id ComfyUI. Это сериализованное состояние/совместимость, не пользовательские переключатели. Видимые семантические переключатели Setup — control_mode, h3_mode, timeline_mode, duration_source и transition_frames.

### Входные ссылки и режимы

| Режим / поле | Роль |
|---|---|
| t2va | Текст/аудио → видео без первого/последнего кадра как обязательных якорей. |
| fl2va | Нативные FIRST/LAST image anchors; без LongMedia latent injection. |
| ref2va | Изображения остаются Picture references. |
| hybrid | Разрешает политику инъекции первого кадра. |
| video_ref_edit | Video 1 задаёт движение, камеру и композицию; Picture refs задают заменяемую идентичность/детали. |
| segmented | Одна сцена с фиксированными continuation-сегментами, а не storyboard cuts. |
| multiclip | Разные клипы/промпты/seed с продолжением между ними. |

## Director

### Верхняя панель

| Параметр | Значения / диапазон | Значение |
|---|---|---|
| H3 | auto, t2va, fl2va, ref2va, hybrid, video_ref_edit | Ручной выбор conditioning family либо безопасная маршрутизация по якорям и ссылкам. Auto выбирает FL2VA для чистых FIRST/LAST anchors; Hybrid для anchors + REF; иначе Ref2VA/T2VA по активным ссылкам. |
| TIMELINE | auto, single, segmented, multiclip | Auto — выбор движка Director; single — один MAIN pass; segmented — один MAIN, разделённый на фиксированные continuation units; multiclip — редактируемая цепочка MAIN клипов. |
| SEG | 1–64, default 3 | Число сегментов. Только при timeline=segmented. |
| SEG LEN | 0.25–150 s, default 5, шаг 0.25 | Видимая длина каждого сегмента. Внутренний context overlap задаётся отдельно. |
| AUDIO | auto, lip_sync, generate, reference_only, preserve_reference, preserve | Политика звука Director; значения имеют те же смыслы, что и Setup audio_mode. |
| FPS | 24; 1–120 | Глобальная частота кадров Director; служит authored timebase для клипов, камеры, аудио, cut markers и RefMod intervals. |
| RES SRC | Base layer, 16:9, 9:16, 1:1, 4:3, 3:4, 21:9, 2.39:1, Custom | Откуда берётся aspect ratio результата. Base layer следует первой MAIN visual media; Custom использует W×H только как пропорцию. |
| RES (Custom) | W×H, каждая сторона 2–5 цифр | Пользовательский источник пропорции. |
| MP | 0.1–8 MP, default 0.8, шаг 0.05 | Общий pixel budget; целевая H3 геометрия выравнивается по 32 px. |
| LEN | 0.25–150 s | Показывается для одного MAIN вне Segmented; задаёт его длительность. |
| Global Prompt | текст + scene guidance | Постоянные указания на всю сцену. Scene guidance содержит Style, Atmosphere, Palette, Lighting, Texture, Era: встроенные пресеты и пользовательские. Эти селекторы добавляют положительное текстовое руководство, а не числовой attention weight. |
| Magnet / Ripple | включено по умолчанию | Magnet привязывает перемещения/края к релевантным границам. Ripple задаёт, следует ли остальным слоям за изменением MAIN-времени. |
| Render | PREVIEW / FINAL | Preview рендерит 50% ширины и высоты (25% числа пикселей); число шагов Sampler не меняется. Final использует заданный MP. |
| Preview monitor | 25%, 50%, 75%, Full | Разрешение локального Program Monitor; не меняет queued output. Playback включает переход к началу/концу, предыдущему/следующему кадру и play/pause. |
| Timeline Zoom / Fit / Layer Height | элементы интерфейса | Zoom меняет горизонтальный масштаб; Fit показывает весь timeline независимо от scroll; Layer Height меняет высоту дорожек. Это только вид редактора. |

### MAIN clip inspector

MAIN-поле: Name; Duration (0.25–150 s, default 5); Seed (пусто = авто); для media — Source In и вычисляемый Source Out; BASE TYPE GENERATED (default) или MEDIA; AUDIO CONTINUATION AUTO (default)/CONTINUE/FRESH; FIRST/LAST (оба выключены по умолчанию); Prompt Priority (default Layer influence; варианты none/motion_first/balanced/free); Clip Prompt; Cast/Recast Identity; ссылки на доступные источники. BASE VIDEO SCALE показывается только если у BASE есть Video; default 100%, активный Video layer владеет собственной шкалой. FIRST/LAST требуют Picture media. Cut markers живут внутри выбранного MAIN clip, остаются на своей позиции при trim и удаляются, когда clip укорочен до области перед маркером.

### CAMERA, AUDIO и дополнительные слои

| Область | Параметры |
|---|---|
| CAMERA block | Start; Duration (0.25–600 s); Shot Size; Movement; Speed; Rig; Camera Body; Lens; Stabilization; Transition Type; Space Relation; Entity Continuity; Camera Note. Значения списков перечислены ниже. Default: start 0/5/10 s, duration 5 s, Medium Shot, Locked-Off / Static, Static speed, Tripod / Locked Head, Cinematic Neutral, Auto / Native Lens, Rig Native, Continuous / Same Shot, Same Space, Lock Population / Layout. |
| AUDIO block | Kind: prompt/reference; Start; Duration (0.25–600 s); при media — Source In; Role: diegetic/music/reactive; текст prompt либо назначенный audio source. |
| Слой | До 24 пользовательских слоёв типов Prompt, Embedding, Character, Reference, Video, Audio Ref и RefMod; до 32 MAIN shots, 64 CAMERA blocks, 64 AUDIO blocks и 256 clips на дополнительный track. Новый слой: Enabled=true, Locked=false, Muted=false; Layer name и, для Prompt/Character/Reference/Video, Prompt Priority (none по умолчанию). CORE CAMERA/AUDIO дорожки можно убрать и добавить обратно. |
| Clip слоя | Name; Start (неотрицательное время); Duration (0.25–600 s). Контент зависит от типа: prompt-текст, embedding из ComfyUI/models/embeddings, Picture/Video/Audio source, Character source, либо RefMod + Strength (0–1). Видео-клип дополнительно может иметь Source In и Video Reference Scale (0.25–1). |
| TAKE | Хранит версию Director timeline и состояние выбранного клипа для безопасной selective regeneration; действия Clip Only/From Here/Reroll/Recast доступны на GENERATED клипах. |

Camera enum values:

| Поле | Допустимые значения |
|---|---|
| Shot Size | Extreme Wide Shot; Wide Shot; Full Shot; Cowboy Shot; Medium Full Shot; Medium Shot; Medium Close-Up; Close-Up; Extreme Close-Up; Macro / Detail; Over-the-Shoulder; Two-Shot; POV Framing. |
| Movement | Locked-Off / Static; Push-In; Pull-Out; Track Forward; Track Backward; Track Left; Track Right; Pan Left; Pan Right; Tilt Up; Tilt Down; Crane Up; Crane Down; Pedestal Up; Pedestal Down; Arc Left; Arc Right; Orbit Clockwise; Orbit Counterclockwise; Full 360 Orbit Clockwise; Full 360 Orbit Counterclockwise; Half Orbit Clockwise; Half Orbit Counterclockwise; Spiral In Clockwise; Spiral In Counterclockwise; Spiral Out Clockwise; Spiral Out Counterclockwise; Diagonal Forward Left; Diagonal Forward Right; Diagonal Backward Left; Diagonal Backward Right; Rise + Push-In; Descend + Push-In; Rise + Pull-Out; Descend + Pull-Out. |
| Speed | Static; Ultra Slow; Slow; Controlled; Medium; Fast; Aggressive; Variable / Ramping. |
| Rig | Tripod / Locked Head; Fluid Head Tripod; Dolly / Track; Slider; Jib / Crane; Technocrane; Steadicam; 3-Axis Gimbal; Shoulder Rig; Handheld; Vehicle Mount; Cable Cam; Robot Arm · Bolt; Robot Arm · KUKA; Drone · Heavy-Lift Cinema; Drone · DJI Inspire 3; Drone · DJI Mavic 3 Cine; Drone · DJI Air 3S; Drone · DJI Mini 4 Pro; FPV · DJI Avata 2; FPV · Cinewhoop; FPV · Racing; Bodycam Mount; Helmet / Head Mount; Static Security Mount. |
| Camera Body | Cinematic Neutral; ARRI Alexa 35; ARRI Alexa Mini LF; Sony VENICE 2; RED V-RAPTOR XL; RED KOMODO-X; Blackmagic URSA Cine 12K; Sony FX3; Sony FX6; Canon C400; Canon EOS R5 C; Canon EOS 5D Mark II; Nikon D850; Sony DCR-VX1000; Canon XL1; Panasonic DVX100; VHS Camcorder; VHS-C Camcorder; Sony Hi8 Handycam; Super 8 Camera; Aaton XTR 16mm; Arricam LT 35mm; IMAX 65mm; Smartphone · Snapshot; Smartphone · Cinematic; Action Camera; Broadcast ENG; CCTV Sensor; Webcam. |
| Lens | Auto / Native Lens; Ultra-Wide 10mm; Ultra-Wide 12mm; Ultra-Wide 14mm; Wide 18mm; Wide 21mm; Wide 24mm; Wide 28mm; Natural 35mm; Natural 40mm; Standard 50mm; Portrait 65mm; Portrait 85mm; Telephoto 100mm; Telephoto 135mm; Long Telephoto 200mm; Long Telephoto 300mm; Macro 60mm; Macro 100mm; Anamorphic 28mm; Anamorphic 35mm; Anamorphic 50mm; Anamorphic 75mm; Vintage Spherical · Wide; Vintage Spherical · Normal; Vintage Spherical · Portrait; Probe Lens; Tilt-Shift; Fisheye; Smartphone Ultra-Wide; Smartphone Wide; Smartphone Tele. |
| Stabilization | Rig Native; Hard Locked; Fluid Controlled; Gyro Stabilized; Gimbal Smooth; Steadicam Organic; Handheld Controlled; Handheld Raw; FPV Stabilized; FPV Raw. |
| Transition Type | Continuous / Same Shot; Threshold Entry; Occluded Hidden Cut; Hard Cut. |
| Space Relation | Same Space; Adjacent Space; Different Space. |
| Entity Continuity | Lock Population / Layout; Preserve Main Subjects; Allow Background Evolution. |

## RefMods

RefMod — подготовленный VAE native reference. ENCODE запускает H3 VideoVAE/AudioVAE и создаёт артефакт; на рендере RefMod назначается клипу слоя REFMOD с временным интервалом. Это не learned text embedding. Важно: Concept Type добавляет общую positive prompt подсказку; Description — метаданные каталога. Ни одно из полей само по себе не заставляет модель переносить стиль или менять смысл латента.

| Поле | Default / значения | Где применяется |
|---|---|---|
| Name | Untitled RefMod; до 160 знаков | Имя элемента библиотеки. |
| Folder | Unsorted; до 120 знаков | Папка общего каталога. |
| Mode | FULL; COMPRESSED | Полный native latent или сжатое представление. |
| Concept Type | scenery, character, object, style, motion, audio_reference, abstract, generic, custom | Общее prompt руководство; не меняет VAE extraction. |
| Description | пусто; до 2000 знаков | Только метаданные библиотеки. |
| Sources | до 128 на RefMod; максимум 256 RefMods в документе | Можно добавить изображения, видео, аудио и использовать Paste Image. У каждого источника своя галочка включения, удаление и Crop to Fit (aspect/source crop). |
| Resolution · short edge | 1024; 256–2048, шаг 64 | Только уменьшение перед VideoVAE; aspect ratio сохраняется. Влияет на пространственный размер visual latent. |
| Visual max tokens / source | 5120; 0 = без лимита | Лимит visual tokens после encode/compression. Если один кадр превышает лимит — нужно уменьшить Resolution или Grid; если превышает только временная сумма, убираются near-duplicate frames, затем равномерно сокращается время. Spatial latent не уменьшается этим лимитом. |
| Grid long edge | 16; 2–64, шаг 2 | Только COMPRESSED. Длинная сторона spatial grid; короткая сторона вычисляется по пропорции источника. |
| Refinement steps | 0; 0–2000, шаг 10 | Только COMPRESSED. Дополнительное latent reconstruction refinement после spatial pooling. Это не сила стиля/идентичности. |
| Clip frames | 16; минимум 1 | Только видео. FULL выбирает до N исходных кадров, округляя выборку к сетке H3 4k+1. COMPRESSED хранит до N latent frames после encode. |
| Voice seconds | 30 s; 0.025–600 s, шаг 0.025 | Только аудио; обрезает начало исходного audio перед AudioVAE. |
| Strength | 1.00; 0–1, шаг 0.01 | Сила RefMod на REFMOD timeline layer. 1.0 — точный native reference; меньшее значение меняет attention bias/weight, но не качество извлечения. |
| Start / Duration | Start от 0; Duration 0.25–600 s | Интервал timeline, в течение которого RefMod виден генерации. Кнопка ↔ растягивает его на весь Director timeline. Несколько REFMOD слоёв могут складываться. |

## Planner и Cameras

Planner: Global Prompt — общее для всех клипов; Multiple Clips Prompt — импортируемый текст со структурой clip_1:/clip_2: или shot_1:/shot_2:; Auto Import — false по умолчанию, при включении импортирует источник только после его изменения; карточки клипов содержат локальный Prompt, Duration (default 7.5 s, 0.25–150 s), Seed (null = Sampler seed + индекс клипа). Для исполнения нужно минимум два клипа. clips_json и import bookkeeping — внутренние.

Cameras: Auto Sync Planner включён по умолчанию; синхронизирует число camera cards с Planner clips. На карточке настраиваются десять полей CAMERA из таблицы выше и заметка. cameras_json/sync_request — внутренние поля; optional clip_plan передаёт Planner output дальше в Setup.

## Video Reconstructor

| Параметр | Default / допустимые значения | Значение |
|---|---|---|
| source_video | IMAGE batch, required | Полный исходник, кадры выбираются лениво внутри локальных окон. |
| source_fps | 24; 1–120, шаг 0.001 | Частота кадров исходника. |
| profile | conservative; balanced; neural_remaster | Профиль реконструкции. |
| source_fit | center_crop; stretch; strict | Crop по центру с сохранением геометрии; растяжение с возможным искажением; exact match размера. |
| reconstruction_strength | 0.55; 0.05–1, шаг 0.01 | Власть исходного видео как native Ref2VA reference. Detail Recovery регулируется отдельно. |
| detail_recovery | true | Отдельный dual-candidate detail pass после глобального прохода; низкая частота геометрии, движения и звук сохраняются. |
| detail_strength | 0.35; 0–1, шаг 0.01 | Сила ограниченного multi-band переноса деталей; ориентир 0.35–0.55. |
| detail_steps | 3; 1–8 | Оценки модели отдельного detail pass; больше шагов дороже и может выдумывать текстуру. |
| segment_seconds | 5 s; 1–30, шаг 0.5 | Локальное окно реконструкции. VRAM определяется этим окном, не общей длиной исходника. |
| source_audio | AUDIO, optional | Исходная дорожка сохраняется нетронутой и не генерируется заново. |
| overlap_frames | 22; 5–360, шаг 17 | Скрытый continuation-контекст соседних окон. |

## Sampler

Требует initial_av, long_media_plan, guider, sampler, sigmas. Sampler Mode=auto и Memory Mode=auto рекомендованы. Режим UI: Basic показывает обычные пользовательские параметры; Advanced добавляет attention/chunk tuning; Debug раскрывает VRAM guards и внутренние knobs. Legacy refine_add_noise/refine_seed/refine_steps сериализуются ради старых workflow, но не участвуют в актуальной логике.

В верхней части ноды находятся пресеты Sampler. `New` создаёт пресет из текущих значений, `Overwrite` записывает текущие значения в выбранный пресет, `Delete` удаляет его. Список и выбранный пресет хранятся в workflow внутри этой ноды. Пресет сохраняет 31 активный параметр виджета; seed остаётся независимым параметром запуска, входы-сокеты и скрытые legacy-поля не входят. Строка состояния показывает `Modified` и количество отличающихся настроек, если значения изменили после выбора пресета.

| Параметр | Default / диапазон | Значение |
|---|---|---|
| seed | 0; 0…2^64−1 | Начальный seed Sampler. |
| sampler_mode | auto; manual | Auto выбирает проверенную runtime policy. Manual даёт доступ ко всем tuning controls. |
| memory_mode | auto; normal; low_vram; ultra_low_vram | Политика residency/chunking для текущего узла. Auto учитывает quantized physical storage и запас VRAM для активаций. |
| offload_completed_segments | true | После каждого прохода накопленный результат переносится в CPU RAM; снижает пиковый VRAM в длинных прогонах, не меняя результат. |
| video_context_denoise | 0; 0–1 | 0 точно сохраняет унаследованный video overlap; 1 полностью его денойзит. |
| audio_context_denoise | 0; 0–1 | Denoise для контекста аудио на стыках. |
| mlp_chunk_tokens | 24576; 0–131072, шаг 512 | Chunk MLP. Меньше — ниже VRAM/скорость; 0 отключает LongMedia MLP chunking. |
| attention_mode | auto; existing; sol_h3; sol; scheduled_sol | Выбор attention backend/profile. Auto выбирает совместимый маршрут; existing уважает ComfyUI backend; остальные включают встроенные Sol варианты при поддерживаемом железе. |
| sol_tau_start / sol_tau_end | 1.3 / 0.8; каждый 0–4, шаг 0.05 | Начало и конец порога адаптивной разреженности Sol. |
| sol_curve | linear; cosine; sqrt; smoothstep; exponential; step | Форма изменения порога между start/end. |
| sol_min_tokens | 4096; 256–131072, шаг 256 | Минимальная геометрия токенов, начиная с которой применяется Sol policy. |
| sol_dense_percent | 0; 0–0.9, шаг 0.05 | Доля dense вычислений в Sol-профиле. |
| sol_sink_conditioning | exact_kv; exact_kv_and_rows; off | Обработка conditioning sink в Sol маршруте. |
| sol_qkv_chunk_tokens | 8192; 0–131072, шаг 512 | QKV projection chunk. 0 возвращает полный fused-QKV маршрут. |
| sol_out_proj_chunk_tokens | 24576; 0–131072, шаг 512 | Chunk выходной projection; 0 отключает эту разбивку. |
| vram_activation_reserve_mb | 2048; 0–12288 MB, шаг 256 | Запрашиваемый ComfyUI запас для activation workspace до загрузки модели; 0 отключает дополнительный reserve. |
| inter_block_vram_guard_mb | 2048; 0–8192 MB, шаг 128 | Цель свободной VRAM между H3-блоками; 0 отключает обычный межблочный cache trim. |
| inter_block_guard_cooldown_blocks | 4; 0–32 | Число блоков между обычными trim. |
| inter_block_guard_emergency_mb | 512; 0–4096 MB, шаг 128 | Аварийный порог свободной VRAM, обходящий обычный cooldown; 0 отключает аварийный режим. |
| inter_block_guard_emergency_cooldown_blocks | 3; 0–32 | Минимальная пауза между аварийными trim. |
| late_block_guard_start | 40; 0–127 | Номер первого Transformer block, с которого разрешён поздний жёсткий guard. |
| late_block_guard_target_mb | 4096; 0–12288 MB, шаг 256 | Свободная VRAM, к которой стремится late-block guard; 0 отключает. |
| late_block_guard_min_cached_mb | 512; 0–4096 MB, шаг 128 | Минимум освобождаемого PyTorch cache, необходимый для late trim. |
| step_boundary_cleanup_mb | 1024; 0–8192 MB, шаг 128 | Цель свободной VRAM после каждого denoise step; 0 отключает cleanup. |
| latent_hires_enabled | false | Learned spatial latent upscaler перед Stage-2 Refine; audio сохраняется точно. |
| latent_hires_model | первый доступный из models/latent_upscale_models | Выбранная модель latent upscaler. |
| latent_hires_scale | 2.0; 1–4×, шаг 0.1 | Пространственный масштаб латента. |
| latent_hires_precision | fp16; bf16; fp32 | Точность инференса upscaler; fp16 — практичный default. |
| latent_hires_align | 32; 16–256 px, шаг 16 | Выравнивание выходной геометрии; 32 — рекомендация upstream для избежания edge/light-band артефактов. |
| refine_enabled | false | Включает Sampler Stage 2. Требуются отдельные Refine Sigmas. |
| windowed_refine | true | Разрешает временные окна Refine для длинных клипов; выключение принудительно использует единый проход и может потребовать значительно больше VRAM. |
| refine_sigmas | SIGMAS, optional socket | Независимое расписание Stage 2; необходимо при включённом Refine. |
| refine_add_noise, refine_seed, refine_steps | legacy | Старые workflow-поля, скрыты и игнорируются. Количество Refine шагов задаётся Refine Sigmas. |

VRAM guard fields доступны в Debug. Необязательно менять их для обычного запуска: auto-профиль предпочтителен, а нулевые значения отдельных guard/chunk параметров являются диагностическими/A-B настройками.

## Decode

| Параметр | Default / диапазон | Значение |
|---|---|---|
| final_av | LATENT, required | Итоговая аудио-видео латентная последовательность. |
| long_media_plan | LONG_MEDIA_PLAN, required | План декодирования, пришедший от Setup. |
| enable_tiling | true | Разрешает пространственную VAE tile decode. |
| tile_size | 256; 32–2048 px, шаг 32 | Размер пространственного VAE tile. |
| width | 512; 32–8192 px, шаг 32 | Целевая ширина декодирования; обычно задаётся планом Setup. |
| temporal_size | 32; 1–256 frames | Размер временного окна декодирования. |
| batch_size | 1; 1–16 | H3 VAE tile parallelism. 1 = Auto; при OOM batch автоматически уменьшается. |
| color_match_strength | 0; 0–1, шаг 0.01 | 0 отключает. Выше нуля подтягивает цветовые статистики кадров к кадру 0; помогает уменьшить drift на loop/seam. |

## Версия источника и замечания

Эта редакция относится к пакету VERSION 0.6.60 из ZIP, а не к соседней рабочей копии или установленной директории custom_nodes. Динамические диапазоны и условия видимости сверены с INPUT_TYPES, node_facade.js и longmedia_director.js внутри этого билда.
