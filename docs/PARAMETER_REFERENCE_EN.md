# LongMedia 0.6.60 Parameter Reference

This reference is checked against Python INPUT_TYPES and the frontend shipped in the 0.6.60 ZIP build. “Internal” means a serialized workflow field that users should not edit manually. Controls are conditional on mode; a hidden value may remain in an older workflow for compatibility.

## Setup

MiniMax H3 • Long Media Setup connects the H3 model, CLIP and both VAEs. With Control = Director, the authored timeline, prompt and audio policy belong to Director; explicit Setup media sockets remain overrides. Manual exposes diagnostic and low-level controls.

| Parameter | Default / allowed values | Purpose |
|---|---|---|
| clip, vae, audio_vae | CLIP, VAE, VAE; required sockets | H3 text encoder and video/audio VAEs for conditioning, encoding and decoding. |
| control_mode | auto; director; manual | Auto uses Setup selectors. Director delegates plan, conditioning and audio semantics to a connected Director. Manual reveals legacy and diagnostic controls. |
| h3_mode | Director: auto/t2va/fl2va/ref2va/hybrid/video_ref_edit; Setup Manual: t2va/fl2va/ref2va/hybrid/video_ref_edit | Selects the H3 conditioning family independently of single/segmented/multiclip. Director Auto derives the mode from FIRST/LAST roles and references. |
| timeline_mode | auto; single; segmented; multiclip | Execution timeline: one pass, fixed continuation segments, or authored Planner/Director clips. Does not select H3 conditioning. |
| prompt / global_prompt | empty string by default | Shared text instructions from Setup. In Director mode, scene-wide instructions live in the Director Global Prompt. |
| width, height | 512 × 512; each 32–8192, step 32 | Setup target canvas geometry. In Director mode, RES SRC + MP own final geometry; Setup dimensions are not the canvas source. |
| resolution_mode | match; max | Target-geometry policy relative to connected references. |
| reference_budget | low; medium; high; max | Budget for native H3 references. This is not an individual RefMod strength. |
| video_fps | 24; 1–120, step 1 | Target frame rate. |
| duration_source | auto; manual; audio; video; longest_input | Chooses target length. Manual uses manual_duration; audio/video use Audio 1/Video 1; longest_input follows the longest connected audio/video. In video_ref_edit, Auto follows Video 1. Planner and Reconstructor own their own timeline lengths. |
| manual_duration | 10 s; 0.1–600 s, step 0.1 | Length used with duration_source=manual. |
| segment_seconds | 5 s; 1–60 s, step 0.5 | Visible new duration of each fixed segment; continuation context is separate and is not subtracted from this time. Shown for segmented and Manual. Reconstructor has a separate field with the same name. |
| transition_frames | 22; 5–3600, step 17 | Context/overlap length between Segmented/MultiClip units. H3 values start 5, 22, 39… The legacy overlap_frames field is hidden and retained for compatibility. |
| audio_mode | auto; preserve; generate; reference_only; preserve_reference; lip_sync | Auto chooses policy. Preserve copies source audio to output without using it as an H3 audio reference. Generate synthesizes audio. Reference only conditions H3. Preserve reference conditions on and restores Audio 1. Lip sync makes Audio 1 the target speech/music clock and restores its original waveform. Director owns this field in Director mode. |
| video_mode | auto; preserve; transform | Manual or video_ref_edit only. Chooses whether input video is preserved or transformed. |
| motion_repair | off; auto; fluid; strong; default off | Optional temporal fast-motion repair after H3. Auto finds overloaded spans; Fluid is more sensitive to fine chaotic structures; Strong applies wider local repair. Audio is frozen during repair and restored exactly afterward. This value persists when workflows are reloaded. |
| loop_closure_enabled | false | Adds a tail pass guided by the opening-frame context for a seamless loop. |
| loop_closure_frames | 57; 2–720 | Approximate ending region used by loop closure; the value snaps to a legal H3 frame count. Shown when enabled or in Manual. |
| loop_closure_strength | 0.65; 0–1, step 0.05 | How strongly the ending structure is guided toward the opening macro-context; fine detail stays free. |
| conditioning_mode | auto_refs; hybrid_first_frame; hybrid_first_last; multiclip_ref2va | Legacy conditioning override, visible only in Manual. Auto refs sends images as Picture refs; Hybrid first frame uses Image 1 as the opening frame; Hybrid first/last assigns Image 1 and 2 as anchors; multiclip_ref2va preserves the legacy MultiClip route. |
| first_frame_mode | native_keyframe; latent_inject; pixel_override; blend; default latent_inject | Hybrid opening-frame policy. FL2VA forces native_keyframe. Other choices are shown according to mode. |
| first_frame_denoise | 0.25; 0–1 | Denoise amount for latent_inject. |
| first_frame_blend_frames | 3; 1–17 | Opening blend duration when first_frame_mode=blend. |
| image_1…image_9 | IMAGE sockets | Native Picture references. A slot’s role depends on the H3/conditioning mode and may be overridden by matching Director media. |
| video_1…video_3 | IMAGE batches | Video frames, not AUDIO. In video_ref_edit, Video 1 is the primary motion/camera/composition source. Connect extracted source audio separately to audio_1 when needed. |
| audio_1…audio_3 | AUDIO sockets | Native audio references. In lip_sync, Audio 1 becomes the target track and is excluded from ordinary Ref2VA audio references. |
| release_guard | true | In Manual, suppresses routine diagnostic logs. Turn it off for profiling and A/B runs. |
| clip_plan | H3_LONGMEDIA_CLIP_PLAN, optional | Planner/Cameras output. Authoritative for MultiClip only when Director is not connected. |
| director | H3_LONGMEDIA_DIRECTOR, optional | Complete Director contract: timeline, cameras and media. Direct Setup image/video/audio sockets override matching Director media slots. |
| reconstruction | H3_LONGMEDIA_RECONSTRUCTION, optional | A connected Reconstructor owns source video/audio, FPS, windows and reconstruction strength. |

Internal/legacy fields across nodes: Director director_json; Cameras cameras_json and sync_request; Planner clips_json, multiclip_import_request and multiclip_last_import_source; Setup workflow_mode, generation_mode, overlap_frames and multiclip_json; ComfyUI's service unique_id. These are serialized state/compatibility fields, not user-facing selectors. The visible semantic Setup controls are control_mode, h3_mode, timeline_mode, duration_source and transition_frames.

### Conditioning and timeline modes

| Mode | Role |
|---|---|
| t2va | Text/audio-to-video without required first/last image anchors. |
| fl2va | Native FIRST/LAST image anchors, without LongMedia latent injection. |
| ref2va | Images stay as Picture references. |
| hybrid | Enables LongMedia opening-frame injection policies. |
| video_ref_edit | Video 1 supplies motion, camera and composition; Picture refs supply replacement identity/details. |
| segmented | One continuous scene executed as fixed continuation segments, not storyboard cuts. |
| multiclip | Separate authored clips/prompts/seeds with continuation between clips. |

## Director

### Header controls

| Parameter | Values / range | Meaning |
|---|---|---|
| H3 | auto, t2va, fl2va, ref2va, hybrid, video_ref_edit | Manual conditioning-family selection or safe routing from anchors/references. Auto chooses FL2VA for pure FIRST/LAST anchors, Hybrid for anchors + REF, otherwise Ref2VA/T2VA from active references. |
| TIMELINE | auto, single, segmented, multiclip | Auto selects Director execution; single is one MAIN pass; segmented splits one MAIN shot into fixed continuation units; multiclip is an editable chain of MAIN clips. |
| SEG | 1–64, default 3 | Segment count, only in timeline=segmented. |
| SEG LEN | 0.25–150 s, default 5, step 0.25 | Visible duration per segment. Hidden transition/context overlap is controlled separately. |
| AUDIO | auto, lip_sync, generate, reference_only, preserve_reference, preserve | Director audio policy; values have the same meanings as Setup audio_mode. |
| FPS | 24; 1–120 | Director frame rate and authored timebase for clips, camera, audio, cut markers and RefMod intervals. |
| RES SRC | Base layer, 16:9, 9:16, 1:1, 4:3, 3:4, 21:9, 2.39:1, Custom | Aspect-ratio source. Base layer follows the first MAIN visual media; Custom uses W×H only as a ratio. |
| RES (Custom) | W×H, each side 2–5 digits | User-defined aspect-ratio source. |
| MP | 0.1–8 MP, default 0.8, step 0.05 | Total pixel budget; final H3 geometry is aligned to 32 px. |
| LEN | 0.25–150 s | Shown for one MAIN outside Segmented; sets its duration. |
| Global Prompt | text + scene guidance | Persistent instructions for the scene. Scene guidance has Style, Atmosphere, Palette, Lighting, Texture and Era categories, with built-in and custom presets. These add positive text guidance, not numeric attention weights. |
| Magnet / Ripple | enabled by default | Magnet snaps moves/edges to relevant boundaries. Ripple controls whether other layers follow MAIN timing edits. |
| Render | PREVIEW / FINAL | Preview renders at 50% width and height (25% pixel count); Sampler steps do not change. Final uses the authored MP target. |
| Preview monitor | 25%, 50%, 75%, Full | Local Program Monitor resolution; does not change queued output. Playback includes start/end, previous/next frame and play/pause. |
| Timeline Zoom / Fit / Layer Height | editor controls | Zoom changes horizontal scale; Fit shows the full timeline regardless of current scroll; Layer Height changes track-row height. These affect only the editor view. |

### MAIN clip inspector

MAIN controls: Name; Duration (0.25–150 s, default 5); Seed (blank = auto); Source In and derived Source Out for media; BASE TYPE GENERATED (default) or MEDIA; AUDIO CONTINUATION AUTO (default)/CONTINUE/FRESH; FIRST/LAST (both off by default); Prompt Priority (default Layer influence; choices none/motion_first/balanced/free); Clip Prompt; Cast/Recast Identity; and links to available sources. BASE VIDEO SCALE appears only when BASE owns a Video; it defaults to 100%, while an active Video layer owns its own scale. FIRST/LAST require Picture media. Cut markers belong to their MAIN clip, stay fixed when that clip is trimmed, and are removed if the clip is shortened to before the marker.

### CAMERA, AUDIO and extra layers

| Area | Parameters |
|---|---|
| CAMERA block | Start; Duration (0.25–600 s); Shot Size; Movement; Speed; Rig; Camera Body; Lens; Stabilization; Transition Type; Space Relation; Entity Continuity; Camera Note. Enum choices are listed below. Defaults: starts at 0/5/10 s, duration 5 s, Medium Shot, Locked-Off / Static, Static speed, Tripod / Locked Head, Cinematic Neutral, Auto / Native Lens, Rig Native, Continuous / Same Shot, Same Space, Lock Population / Layout. |
| AUDIO block | Kind: prompt/reference; Start; Duration (0.25–600 s); Source In for media; Role: diegetic/music/reactive; prompt text or assigned audio source. |
| Layer | Up to 24 user layers of types Prompt, Embedding, Character, Reference, Video, Audio Ref or RefMod; up to 32 MAIN shots, 64 CAMERA blocks, 64 AUDIO blocks and 256 clips per extra track. New layer defaults: Enabled=true, Locked=false, Muted=false; Prompt Priority defaults to none on Prompt/Character/Reference/Video layers. CORE CAMERA/AUDIO tracks can be removed and restored. |
| Layer clip | Name; non-negative Start; Duration (0.25–600 s). Content depends on type: prompt text, embedding from ComfyUI/models/embeddings, Picture/Video/Audio source, Character source, or RefMod + Strength (0–1). A video clip can also have Source In and Video Reference Scale (0.25–1). |
| TAKE | Stores a Director timeline version and selected-clip state for safe selective regeneration. Clip Only/From Here/Reroll/Recast actions apply to GENERATED clips. |

Camera enum values:

| Field | Choices |
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

A RefMod is a prepared native VAE reference. ENCODE runs the H3 VideoVAE/AudioVAE and creates the artifact; at render time it is assigned to a REFMOD timeline layer with a time interval. It is not a learned text embedding. Important: Concept Type adds broad positive prompt guidance; Description is library metadata. Neither field alone forces style transfer or changes the semantic content of the latent.

| Field | Default / range | Where it applies |
|---|---|---|
| Name | Untitled RefMod; up to 160 characters | Library name. |
| Folder | Unsorted; up to 120 characters | Shared-catalog folder. |
| Mode | FULL; COMPRESSED | Full native latent or compressed representation. |
| Concept Type | scenery, character, object, style, motion, audio_reference, abstract, generic, custom | Broad prompt guidance; does not change VAE extraction. |
| Description | empty; up to 2000 characters | Library metadata only. |
| Sources | up to 128 per RefMod; up to 256 RefMods per document | Add images, videos, audio, or paste an image. Each source has an enable checkbox, remove action and Crop to Fit/source crop. |
| Resolution · short edge | 1024; 256–2048, step 64 | Downscale-only before VideoVAE; keeps aspect ratio. Affects visual latent spatial size. |
| Visual max tokens / source | 5120; 0 = unlimited | Visual-token cap after encode/compression. If one frame alone exceeds the cap, lower Resolution or Grid. If only total time exceeds it, near-duplicate frames are removed and the temporal span is uniformly reduced. Spatial latent size is not reduced by this cap. |
| Grid long edge | 16; 2–64, step 2 | COMPRESSED only. Long side of the spatial grid; short side follows source aspect ratio. |
| Refinement steps | 0; 0–2000, step 10 | COMPRESSED only. Additional latent reconstruction after spatial pooling. Not style or identity strength. |
| Clip frames | 16; minimum 1 | Video only. FULL selects up to N source frames, rounded to H3's 4k+1 sampling grid. COMPRESSED stores up to N latent frames after encoding. |
| Voice seconds | 30 s; 0.025–600 s, step 0.025 | Audio only; trims the beginning of the source before AudioVAE. |
| Strength | 1.00; 0–1, step 0.01 | RefMod strength on its timeline layer. 1.0 is the exact native reference; lower values alter attention bias/weight, not extraction quality. |
| Start / Duration | Start ≥0; Duration 0.25–600 s | Timeline interval during which the RefMod is visible to generation. ↔ stretches it across the complete Director timeline. Multiple REFMOD layers can stack. |

## Planner and Cameras

Planner: Global Prompt is shared by every clip. Multiple Clips Prompt is an import source formatted as clip_1:/clip_2: or shot_1:/shot_2:. Auto Import is false by default; when enabled, it imports only after the source changes. Clip cards contain local Prompt, Duration (default 7.5 s, 0.25–150 s), and Seed (null = Sampler seed + clip index). At least two clips are required for execution. clips_json and import bookkeeping are internal.

Cameras: Auto Sync Planner is enabled by default and keeps camera-card count aligned with Planner clips. Each card exposes the ten CAMERA fields above plus a note. cameras_json/sync_request are internal; optional clip_plan passes Planner output downstream to Setup.

## Video Reconstructor

| Parameter | Default / allowed values | Purpose |
|---|---|---|
| source_video | IMAGE batch, required | Full source; frames are selected lazily within local windows. |
| source_fps | 24; 1–120, step 0.001 | Source frame rate. |
| profile | conservative; balanced; neural_remaster | Reconstruction profile. |
| source_fit | center_crop; stretch; strict | Center crop preserving geometry; stretch which can distort aspect; or require an exact size match. |
| reconstruction_strength | 0.55; 0.05–1, step 0.01 | Authority of the source video as a native Ref2VA reference. Detail Recovery is controlled separately. |
| detail_recovery | true | Separate dual-candidate detail pass after the global pass; low-frequency geometry, motion and audio remain locked. |
| detail_strength | 0.35; 0–1, step 0.01 | Strength of bounded multi-band detail transfer; suggested starting range 0.35–0.55. |
| detail_steps | 3; 1–8 | Model evaluations in the separate detail pass; more steps cost more and may invent texture. |
| segment_seconds | 5 s; 1–30, step 0.5 | Local reconstruction window. VRAM follows this window, not full source duration. |
| source_audio | AUDIO, optional | Original soundtrack is preserved unchanged and is not regenerated. |
| overlap_frames | 22; 5–360, step 17 | Hidden continuation context between adjacent windows. |

## Sampler

Requires initial_av, long_media_plan, guider, sampler and sigmas. Sampler Mode=auto and Memory Mode=auto are recommended. Basic UI shows common controls; Advanced adds attention/chunk tuning; Debug reveals VRAM guards and internal knobs. Legacy refine_add_noise/refine_seed/refine_steps are serialized for old workflows but do not affect current behavior.

Sampler presets appear at the top of the node. `New` captures the current values, `Overwrite` saves the current values into the selected preset, and `Delete` removes it. The preset list and selection are stored in the workflow on that node. A preset covers 31 active widget settings; seed remains independent for each run, and input sockets and hidden legacy fields are excluded. The status line shows `Modified` and the number of changed settings when values differ from the selected preset.

| Parameter | Default / range | Purpose |
|---|---|---|
| seed | 0; 0…2^64−1 | Sampler seed. |
| sampler_mode | auto; manual | Auto uses the validated runtime policy. Manual exposes all tuning controls. |
| memory_mode | auto; normal; low_vram; ultra_low_vram | Node-local residency/chunking policy. Auto accounts for quantized physical storage and activation headroom. |
| offload_completed_segments | true | Moves the accumulated result to CPU RAM after each pass to reduce peak VRAM in long runs without changing output. |
| video_context_denoise | 0; 0–1 | 0 preserves inherited video overlap exactly; 1 fully denoises it. |
| audio_context_denoise | 0; 0–1 | Denoise for audio context at boundaries. |
| mlp_chunk_tokens | 24576; 0–131072, step 512 | MLP token chunking. Smaller lowers VRAM and can lower speed; 0 disables LongMedia MLP chunking. |
| attention_mode | auto; existing; sol_h3; sol; scheduled_sol | Attention backend/profile. Auto selects a compatible route; existing respects the ComfyUI backend; other values enable embedded Sol variants on supported hardware. |
| sol_tau_start / sol_tau_end | 1.3 / 0.8; each 0–4, step 0.05 | Start and end thresholds for adaptive Sol sparsity. |
| sol_curve | linear; cosine; sqrt; smoothstep; exponential; step | Shape of threshold progression between start and end. |
| sol_min_tokens | 4096; 256–131072, step 256 | Minimum token geometry at which Sol policy applies. |
| sol_dense_percent | 0; 0–0.9, step 0.05 | Dense-computation fraction in Sol mode. |
| sol_sink_conditioning | exact_kv; exact_kv_and_rows; off | Conditioning-sink handling in the Sol route. |
| sol_qkv_chunk_tokens | 8192; 0–131072, step 512 | QKV projection chunk size. 0 restores the full fused-QKV route. |
| sol_out_proj_chunk_tokens | 24576; 0–131072, step 512 | Output-projection chunk size; 0 disables this chunking. |
| vram_activation_reserve_mb | 2048; 0–12288 MB, step 256 | Extra headroom requested from ComfyUI before model loading for activation workspace; 0 disables the extra reserve. |
| inter_block_vram_guard_mb | 2048; 0–8192 MB, step 128 | Free-VRAM target between H3 blocks; 0 disables normal inter-block cache trimming. |
| inter_block_guard_cooldown_blocks | 4; 0–32 | Blocks between normal trims. |
| inter_block_guard_emergency_mb | 512; 0–4096 MB, step 128 | Emergency free-VRAM threshold that bypasses normal cooldown; 0 disables emergency behavior. |
| inter_block_guard_emergency_cooldown_blocks | 3; 0–32 | Minimum wait between emergency trims. |
| late_block_guard_start | 40; 0–127 | First Transformer block where the late hard guard may run. |
| late_block_guard_target_mb | 4096; 0–12288 MB, step 256 | Free-VRAM target for the late-block guard; 0 disables it. |
| late_block_guard_min_cached_mb | 512; 0–4096 MB, step 128 | Minimum reclaimable PyTorch cache before a late trim is attempted. |
| step_boundary_cleanup_mb | 1024; 0–8192 MB, step 128 | Free-VRAM target after each denoise step; 0 disables cleanup. |
| latent_hires_enabled | false | Learned spatial latent upscaler before Stage-2 Refine; audio remains exact. |
| latent_hires_model | first available under models/latent_upscale_models | Selected latent upscaler. |
| latent_hires_scale | 2.0; 1–4×, step 0.1 | Spatial latent scale. |
| latent_hires_precision | fp16; bf16; fp32 | Upscaler inference precision; fp16 is the practical default. |
| latent_hires_align | 32; 16–256 px, step 16 | Output alignment; 32 is the upstream recommendation to avoid edge/light-band artifacts. |
| refine_enabled | false | Enables Sampler Stage 2. A separate Refine Sigmas input is required. |
| windowed_refine | true | Allows temporal Refine windows for long clips; OFF forces one full pass and can require much more VRAM. |
| refine_sigmas | SIGMAS, optional socket | Independent Stage-2 schedule; required when Refine is enabled. |
| refine_add_noise, refine_seed, refine_steps | legacy | Old workflow fields, hidden and ignored. Refine step count comes from Refine Sigmas. |

VRAM guard controls are in Debug. They do not usually need tuning for standard renders: Auto is preferred, and zero values for individual guards/chunk controls are diagnostic/A-B options.

## Decode

| Parameter | Default / range | Purpose |
|---|---|---|
| final_av | LATENT, required | Final audio/video latent sequence. |
| long_media_plan | LONG_MEDIA_PLAN, required | Decode plan from Setup. |
| enable_tiling | true | Enables spatial VAE tile decode. |
| tile_size | 256; 32–2048 px, step 32 | Spatial VAE tile size. |
| width | 512; 32–8192 px, step 32 | Target decode width; normally supplied by Setup's plan. |
| temporal_size | 32; 1–256 frames | Temporal decode window size. |
| batch_size | 1; 1–16 | H3 VAE tile parallelism. 1 = Auto; on OOM the batch is reduced automatically. |
| color_match_strength | 0; 0–1, step 0.01 | 0 disables. Higher values pull frame color statistics toward frame 0 to reduce drift at loop/seam boundaries. |

## Source build and scope

This edition describes VERSION 0.6.60 from the ZIP build, not a neighboring working copy or installed custom_nodes folder. Dynamic ranges and visibility rules were checked against INPUT_TYPES, node_facade.js and longmedia_director.js in that build.
