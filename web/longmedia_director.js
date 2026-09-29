/**
 * LongMedia Director v4.11.0
 *
 * A unified editorial surface with a large Program Monitor, centered transport, MAIN /
 * CAMERA / AUDIO tracks, media thumbnails, waveforms, persistent inspector sizing,
 * WHO & WHAT assets and contextual tooltips. The raw JSON/global widgets remain the serialized source of truth.
 *
 * Inspired by the interaction model of imbutus/ComfyUI-MiniMaxDirector (MIT), while
 * compiling into LongMedia's single H3_LONGMEDIA_DIRECTOR contract for Setup.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const DIRECTOR_CLASS = "MiniMaxH3LongMediaDirector";
const EDITOR_MIN_HEIGHT = 760;
const EDITOR_MAX_HEIGHT = 2400;
const TIMELINE_PANEL_HEIGHT = 244;
const PROGRAM_MONITOR_HEIGHT = 430;
const NODE_HEIGHT = 1240;
const NODE_MIN_HEIGHT = 780;
const NODE_MAX_HEIGHT = 2600;
const NODE_MIN_WIDTH = 1120;
const NODE_DEFAULT_WIDTH = 1380;
const MAX_SUBJECTS = 16;
const MAX_SHOTS = 32;
const MAX_CAMERA = 64;
const MAX_AUDIO = 64;
const MAX_EXTRA_TRACKS = 24;
const MAX_EXTRA_CLIPS = 256;
const EXTRA_TRACK_TYPES = ["prompt","embedding","character","reference","video","audio","refmod"];
const EXTRA_TRACK_META = {prompt:{label:"PROMPT",color:"#d9ad66"},embedding:{label:"EMBEDDING",color:"#c99bf0"},character:{label:"CHARACTER",color:"#e58cab"},reference:{label:"REFERENCE",color:"#8fc8e5"},video:{label:"VIDEO",color:"#78b9e8"},audio:{label:"AUDIO",color:"#83d39f"},refmod:{label:"REFMOD",color:"#e8a952"}};
const AUDIO_MODES = ["auto","lip_sync","generate","reference_only","preserve_reference","preserve"];
const DIRECTOR_H3_MODES = ["auto","t2va","fl2va","ref2va","hybrid","video_ref_edit"];
const DIRECTOR_TIMELINE_MODES = ["auto","single","segmented","multiclip"];
const PROMPT_GUIDANCE_OPTIONS = [
  {value:"none",label:"No added priority"},
  {value:"motion_first",label:"Motion first"},
  {value:"balanced",label:"Balanced"},
  {value:"free",label:"Free adaptation"},
];
const CLIP_PROMPT_GUIDANCE_OPTIONS = [
  {value:"inherit",label:"Layer influence"},
  ...PROMPT_GUIDANCE_OPTIONS,
];
const DIRECTOR_RESOLUTION_SOURCES = [
  {value:"base_layer",label:"Base layer"},
  {value:"16:9",label:"16:9"},{value:"9:16",label:"9:16"},{value:"1:1",label:"1:1"},
  {value:"4:3",label:"4:3"},{value:"3:4",label:"3:4"},{value:"21:9",label:"21:9"},
  {value:"2.39:1",label:"2.39:1"},{value:"custom",label:"Custom"},
];
const PREVIEW_QUALITY_MODES = ["25","50","75","Full"];
const AUDIO_MODE_HELP = {
  auto: "Automatic LongMedia policy chooses the final-audio path from the connected references.",
  lip_sync: "Audio 1 is the authoritative mouth/timing clock and the original Audio 1 waveform is restored at output.",
  generate: "H3 generates the final soundtrack; connected audio may still guide the model.",
  reference_only: "Connected audio conditions H3 but is not copied to the final soundtrack.",
  preserve_reference: "Audio 1 conditions H3 and the original Audio 1 waveform is restored at output.",
  preserve: "Audio 1 is copied to the final output but is not used as an H3 audio reference.",
};

const CAMERA = {
  shot_size: ["Extreme Wide Shot","Wide Shot","Full Shot","Cowboy Shot","Medium Full Shot","Medium Shot","Medium Close-Up","Close-Up","Extreme Close-Up","Macro / Detail","Over-the-Shoulder","Two-Shot","POV Framing"],
  rig: ["Tripod / Locked Head","Fluid Head Tripod","Dolly / Track","Slider","Jib / Crane","Technocrane","Steadicam","3-Axis Gimbal","Shoulder Rig","Handheld","Vehicle Mount","Cable Cam","Robot Arm · Bolt","Robot Arm · KUKA","Drone · Heavy-Lift Cinema","Drone · DJI Inspire 3","Drone · DJI Mavic 3 Cine","Drone · DJI Air 3S","Drone · DJI Mini 4 Pro","FPV · DJI Avata 2","FPV · Cinewhoop","FPV · Racing","Bodycam Mount","Helmet / Head Mount","Static Security Mount"],
  camera_body: ["Cinematic Neutral","ARRI Alexa 35","ARRI Alexa Mini LF","Sony VENICE 2","RED V-RAPTOR XL","RED KOMODO-X","Blackmagic URSA Cine 12K","Sony FX3","Sony FX6","Canon C400","Canon EOS R5 C","Canon EOS 5D Mark II","Nikon D850","Sony DCR-VX1000","Canon XL1","Panasonic DVX100","VHS Camcorder","VHS-C Camcorder","Sony Hi8 Handycam","Super 8 Camera","Aaton XTR 16mm","Arricam LT 35mm","IMAX 65mm","Smartphone · Snapshot","Smartphone · Cinematic","Action Camera","Broadcast ENG","CCTV Sensor","Webcam"],
  lens: ["Auto / Native Lens","Ultra-Wide 10mm","Ultra-Wide 12mm","Ultra-Wide 14mm","Wide 18mm","Wide 21mm","Wide 24mm","Wide 28mm","Natural 35mm","Natural 40mm","Standard 50mm","Portrait 65mm","Portrait 85mm","Telephoto 100mm","Telephoto 135mm","Long Telephoto 200mm","Long Telephoto 300mm","Macro 60mm","Macro 100mm","Anamorphic 28mm","Anamorphic 35mm","Anamorphic 50mm","Anamorphic 75mm","Vintage Spherical · Wide","Vintage Spherical · Normal","Vintage Spherical · Portrait","Probe Lens","Tilt-Shift","Fisheye","Smartphone Ultra-Wide","Smartphone Wide","Smartphone Tele"],
  stabilization: ["Rig Native","Hard Locked","Fluid Controlled","Gyro Stabilized","Gimbal Smooth","Steadicam Organic","Handheld Controlled","Handheld Raw","FPV Stabilized","FPV Raw"],
  movement: ["Locked-Off / Static","Push-In","Pull-Out","Track Forward","Track Backward","Track Left","Track Right","Pan Left","Pan Right","Tilt Up","Tilt Down","Crane Up","Crane Down","Pedestal Up","Pedestal Down","Arc Left","Arc Right","Orbit Clockwise","Orbit Counterclockwise","Full 360 Orbit Clockwise","Full 360 Orbit Counterclockwise","Half Orbit Clockwise","Half Orbit Counterclockwise","Spiral In Clockwise","Spiral In Counterclockwise","Spiral Out Clockwise","Spiral Out Counterclockwise","Diagonal Forward Left","Diagonal Forward Right","Diagonal Backward Left","Diagonal Backward Right","Rise + Push-In","Descend + Push-In","Rise + Pull-Out","Descend + Pull-Out"],
  speed: ["Static","Ultra Slow","Slow","Controlled","Medium","Fast","Aggressive","Variable / Ramping"],
  transition_type: ["Continuous / Same Shot","Threshold Entry","Occluded Hidden Cut","Hard Cut"],
  space_relation: ["Same Space","Adjacent Space","Different Space"],
  entity_continuity: ["Lock Population / Layout","Preserve Main Subjects","Allow Background Evolution"],
};
const DEFAULT_CAMERA = {
  shot_size: "Medium Shot", rig: "Tripod / Locked Head", camera_body: "Cinematic Neutral",
  lens: "Auto / Native Lens", stabilization: "Rig Native", movement: "Locked-Off / Static",
  speed: "Static", transition_type: "Continuous / Same Shot", space_relation: "Same Space",
  entity_continuity: "Lock Population / Layout", transition_to_next: false,
};

const REFMOD_MODES = ["FULL","COMPRESSED"];
const REFMOD_CONCEPT_TYPES = ["scenery","character","object","style","motion","audio_reference","abstract","generic","custom"];
const REFMOD_MAX = 256;
const SCENE_GUIDANCE_CATEGORIES = [
  {id:"style",label:"VISUAL STYLE"},{id:"atmosphere",label:"ATMOSPHERE"},
  {id:"palette",label:"COLOR PALETTE"},{id:"lighting",label:"LIGHTING"},
  {id:"texture",label:"TEXTURE"},{id:"era",label:"ERA / PERIOD"},
];
const SCENE_GUIDANCE_BUILTINS = {
  style:[
    ["cinematic_realism","Cinematic realism","Cinematic photorealism with natural material detail and coherent filmic rendering."],
    ["analog_film","Analog film","Analog 35mm film character with restrained grain, soft halation and organic tonal response."],
    ["pixel_sort_glitch","Pixel-sorting glitch","A consistent digital pixel-sorting aesthetic: long vertical blue data streaks distort the subject while the scene remains clearly readable."],
    ["graphic_noir","Graphic noir","High-contrast graphic noir with deliberate silhouettes, crisp edges and controlled detail."],
    ["dreamlike","Dreamlike","Dreamlike visual language with softened transitions, luminous edges and fluid, coherent forms."],
    ["documentary","Documentary realism","Observational documentary imagery with natural imperfections, available-light character and unembellished detail."],
    ["anime_cel","Hand-painted animation","Expressive hand-painted animation with clean silhouettes, intentional linework and carefully controlled cel shading."],
    ["watercolor","Watercolor illustration","Watercolor forms with translucent pigment layers, soft color blooms and visible paper-like edges."],
    ["oil_painting","Oil painting","Painterly oil-on-canvas rendering with layered brushwork, rich pigments and tactile highlights."],
    ["retro_futurism","Retro futurism","Retro-futurist visual design with bold geometric forms, polished materials and optimistic vintage technology."],
    ["stop_motion","Stop-motion miniature","Tactile stop-motion craft with miniature set detail, shaped materials and a subtly handmade finish."],
    ["editorial_fashion","Fashion editorial","High-end fashion editorial imagery with graphic composition, refined styling and deliberate visual polish."],
    ["cyberpunk","Cyberpunk","Dense cyberpunk visual language with luminous signage, layered urban detail and saturated electronic color."],
    ["ink_woodcut","Ink and woodcut","Graphic ink illustration with carved-looking marks, bold shapes and expressive high-contrast texture."],
    ["architectural_minimal","Architectural minimalism","Minimal visual language built from precise geometry, restrained materials and generous negative space."],
  ],
  atmosphere:[
    ["quiet_tension","Quiet tension","A restrained, watchful tension hangs over the scene; the mood stays controlled and immersive."],
    ["melancholic","Melancholic","A reflective, subdued melancholy shapes the emotional tone of the scene."],
    ["eerie","Eerie","An uncanny, mysterious atmosphere with a calm sense of unease."],
    ["warm_hopeful","Warm and hopeful","An intimate, warm and quietly hopeful emotional atmosphere."],
    ["serene","Serene","A peaceful, unhurried atmosphere with a sense of stillness and emotional ease."],
    ["claustrophobic","Claustrophobic","A close, compressed atmosphere with limited breathing room and mounting pressure."],
    ["ominous","Ominous","A foreboding mood with restrained unease and a sense that something is approaching."],
    ["wonder","Wonder and discovery","An open, curious atmosphere shaped by discovery, scale and quiet amazement."],
    ["romantic","Romantic","An intimate, tender atmosphere with gentle emotional warmth and closeness."],
    ["urgent","Urgent","A focused, high-stakes atmosphere with forward pressure and immediate purpose."],
    ["playful","Playful","A lively, mischievous mood with lightness and spontaneous energy."],
    ["lonely","Lonely","A quiet sense of isolation, with the subject emotionally distinct from the surrounding space."],
    ["oppressive","Oppressive","A heavy, oppressive mood with dense stillness and sustained emotional pressure."],
    ["nostalgic","Nostalgic","A warm, reflective mood colored by memory and a gentle sense of time passed."],
    ["ethereal","Ethereal","A weightless, otherworldly atmosphere with delicate light and a sense of suspended time."],
  ],
  palette:[
    ["cyan_steel","Cyan steel","A unified palette of deep navy, cold cyan and steel blue with restrained neutral highlights."],
    ["amber_teal","Amber and teal","A cinematic balance of warm amber highlights and cool teal shadows."],
    ["muted_earth","Muted earth","Muted ochre, clay, olive and charcoal tones with low saturation."],
    ["monochrome","Monochrome","A disciplined monochrome palette with clear tonal separation and subtle gradation."],
    ["pastel","Soft pastel","Soft pastel hues with gentle saturation and smooth tonal transitions."],
    ["crimson_charcoal","Crimson and charcoal","Deep charcoal and near-black tones with controlled crimson accents."],
    ["violet_indigo","Violet and indigo","Layered violet, indigo and midnight blue with cool luminous accents."],
    ["jade_ivory","Jade and ivory","Jade green, warm ivory and dark mineral tones in a balanced, elegant palette."],
    ["sunset_coral","Sunset coral","Coral, soft peach and dusky rose balanced by warm violet shadows."],
    ["forest_moss","Forest moss","Moss green, pine, damp earth and muted natural highlights."],
    ["desert_sand","Desert sand","Sun-bleached sand, ochre, dusty terracotta and pale atmospheric neutrals."],
    ["ice_blue","Ice blue","Glacial blue, pale cyan and cool silver with crisp dark-blue separation."],
    ["warm_monochrome","Warm monochrome","A unified range of warm brown, sepia and cream with clear tonal depth."],
    ["primary_pop","Primary color pop","Confident red, yellow and blue accents against a clean, restrained neutral base."],
    ["faded_print","Faded print","A gently faded print palette with softened blacks, aged cream and subdued color."],
  ],
  lighting:[
    ["soft_overcast","Soft overcast","Soft overcast illumination with broad, gentle shadows and even highlights."],
    ["neon_night","Neon night","Night lighting shaped by saturated cyan and blue practical lights with deep shadow detail."],
    ["golden_hour","Golden hour","Low warm sunlight, long soft-edged shadows and natural golden highlights."],
    ["hard_chiaroscuro","Chiaroscuro","Directional low-key lighting with sculpted highlights and deep but legible shadows."],
    ["moonlight","Moonlight","Cool moonlit illumination with soft blue highlights and readable night shadows."],
    ["fluorescent","Fluorescent interior","Cool overhead fluorescent light with a subtle green-cyan cast and practical interior falloff."],
    ["backlit_silhouette","Strong backlight","A clear backlight outlines the subject and separates the silhouette from the environment."],
    ["volumetric","Volumetric light","Visible shafts of directional light reveal atmospheric depth and suspended particles."],
    ["candlelight","Candlelight","Small warm practical sources create intimate amber pools and softly moving shadows."],
    ["softbox","Soft studio light","Large diffused studio sources create smooth highlights, controlled shadows and polished detail."],
    ["harsh_noon","Hard midday sun","Direct overhead sunlight creates crisp shadows, bright highlights and a strong sense of place."],
    ["tungsten_practical","Warm practical lamps","Warm tungsten lamps motivate the scene lighting with localized amber highlights."],
    ["blue_hour","Blue hour","Cool twilight ambience with a blue-violet sky glow and balanced practical lights."],
    ["rain_reflections","Rain-lit reflections","Wet surfaces catch and reflect colored practical lights, extending them across the scene."],
    ["window_light","Diffused window light","Soft directional daylight enters from a nearby window with gentle falloff across the space."],
  ],
  texture:[
    ["fine_grain","Fine film grain","Fine, even film grain across the image, preserving facial and material detail."],
    ["clean_digital","Clean digital","A clean, precise digital image with smooth gradients and crisp material detail."],
    ["analog_wear","Analog wear","Subtle analog wear with fine grain, slight gate weave and naturally imperfect edges."],
    ["data_streaks","Data streaks","Distinct vertical data streaks and pixel-sorted trails integrated into the image texture."],
    ["halftone","Halftone print","Fine halftone dots and printed ink texture, with clean shapes that remain readable."],
    ["paper_fiber","Paper fiber","Visible natural paper fibers and soft surface tooth beneath the image."],
    ["brushstroke","Visible brushwork","Expressive layered brush marks remain visible in painted surfaces and transitions."],
    ["matte_surface","Matte surface","Soft matte material response with restrained reflections and tactile surface detail."],
    ["polished_glass","Polished glass","Clean glassy highlights, subtle refraction and controlled reflective surfaces."],
    ["weathered_metal","Weathered metal","Worn metal with fine scratches, softened edges and varied patina."],
    ["soft_bloom","Soft halation","Gentle highlight bloom and halation around bright practical lights, preserving edge detail."],
    ["chromatic_fringe","Chromatic fringe","Subtle color fringing at high-contrast edges, integrated as a restrained optical artifact."],
    ["scanlines","Display scanlines","Fine display scanlines and electronic image texture remain visible across the frame."],
    ["dust_specks","Dust and scratches","Fine dust specks and light scratches evoke an aged film element without obscuring the image."],
    ["crisp_edges","Crisp digital edges","Precise, clean edges with detailed surfaces and no added surface grain."],
  ],
  era:[
    ["contemporary","Contemporary","Contemporary production design, wardrobe and environmental details."],
    ["1980s","1980s","1980s-inspired production design, wardrobe, materials and visual references."],
    ["near_future","Near future","Near-future design language grounded in plausible materials and practical technology."],
    ["timeless","Timeless","A timeless setting without strong period-specific visual cues."],
    ["1920s","1920s","1920s-inspired architecture, clothing, objects and period design details."],
    ["1940s_noir","1940s noir","1940s-inspired noir styling, tailored silhouettes, period props and urban design."],
    ["1950s","1950s","1950s-inspired interiors, vehicles, clothing and mid-century design details."],
    ["1960s","1960s","1960s-inspired design, fashion, graphic forms and period technology."],
    ["1970s","1970s","1970s-inspired colors, clothing, interiors and analog-era objects."],
    ["1990s","1990s","1990s-inspired streetwear, consumer technology, vehicles and everyday environments."],
    ["early_2000s","Early 2000s","Early-2000s-inspired clothing, devices, materials and popular visual culture."],
    ["art_deco","Art Deco period","Art Deco geometry, stepped forms, polished ornament and period architectural details."],
    ["medieval","Medieval","Medieval-inspired architecture, garments, tools and handcrafted materials."],
    ["retro_future","Retro-future period","A retro-futurist period setting with optimistic speculative technology and vintage industrial design."],
    ["far_future","Far future","Far-future production design with unfamiliar yet coherent materials, objects and architecture."],
  ],
};
function normalizeSceneGuidanceItems(raw,{library=false}={}){const result=[],seen=new Set();for(const item of Array.isArray(raw)?raw.slice(0,library?128:SCENE_GUIDANCE_CATEGORIES.length):[]){if(!item||typeof item!=="object")continue;const category=String(item.category||"").toLowerCase(),name=String(item.name||"").trim().slice(0,100),prompt=String(item.prompt||"").trim().slice(0,2000),presetId=String(item.id||"").trim().slice(0,160);if(!SCENE_GUIDANCE_CATEGORIES.some(x=>x.id===category)||!name||!prompt)continue;const key=library?(presetId||`custom-${result.length+1}`):category;if(seen.has(key))continue;seen.add(key);result.push({id:presetId||key,category,name,prompt});}return result;}

function defaultRefmod(){return {refmod_id:id("refmod"),name:"Untitled RefMod",folder:"Unsorted",source:"",sources:[],source_subject_id:null,source_clip_id:null,source_take_revision:null,path:"",state:"draft",mode:"FULL",concept_type:"scenery",description:"",resolution_short_edge:1024,max_tokens:5120,grid_long_edge:16,clip_frames:16,voice_seconds:30,refinement_steps:0,spatial_grid:"16x16",temporal_frames:16,strength:1.0};}
function normalizeRefmod(r){r=r&&typeof r==="object"?r:{};let mode=String(r.mode||"FULL").toUpperCase();if(!REFMOD_MODES.includes(mode))mode="FULL";let concept=String(r.concept_type||"scenery").toLowerCase();if(!REFMOD_CONCEPT_TYPES.includes(concept))concept="custom";const legacyGrid=String(r.spatial_grid||"16x16").match(/\d+/g)||[];const gridLongEdge=Math.max(2,Math.min(64,Math.round(finite(r.grid_long_edge,Math.max(Number(legacyGrid[0])||16,Number(legacyGrid[1])||16))/2)*2));const clipFrames=Math.max(1,Math.min(2147483647,Math.trunc(finite(r.clip_frames,r.temporal_frames??16))));const resolution=Math.max(256,Math.min(2048,Math.round(finite(r.resolution_short_edge??r.resolution,1024)/64)*64));return {refmod_id:String(r.refmod_id||r.id||"").trim()||id("refmod"),name:String(r.name||"Untitled RefMod").slice(0,160),folder:String(r.folder||"Unsorted").slice(0,120),source:String(r.source||"").slice(0,500),sources:Array.isArray(r.sources)?r.sources.slice(0,128).map(s=>({name:String(s?.name||"Source").slice(0,160),filename:String(s?.filename||"").slice(0,300),kind:String(s?.kind||"image").slice(0,20),crop:s?.crop&&typeof s.crop==="object"?{x:clamp(s.crop.x,0,1,0),y:clamp(s.crop.y,0,1,0),width:clamp(s.crop.width,.01,1,1),height:clamp(s.crop.height,.01,1,1)}:null})):[],source_subject_id:String(r.source_subject_id||"").trim()||null,source_clip_id:String(r.source_clip_id||"").trim()||null,source_take_revision:String(r.source_take_revision||"").trim()||null,path:String(r.path||"").slice(0,1000),state:String(r.state||"draft").slice(0,80),kind:String(r.kind||"image").slice(0,40),format_version:Math.max(0,Math.trunc(finite(r.format_version,0))),members:Array.isArray(r.members)?r.members:[],mode,concept_type:concept,description:String(r.description||"").slice(0,2000),resolution_short_edge:resolution,max_tokens:Math.max(0,Math.min(2147483647,Math.trunc(finite(r.max_tokens,5120)))),grid_long_edge:gridLongEdge,clip_frames:clipFrames,voice_seconds:Math.max(.025,Math.min(600,finite(r.voice_seconds,30))),refinement_steps:Math.max(0,Math.min(2000,Math.trunc(finite(r.refinement_steps,0)))),spatial_grid:`${gridLongEdge}x${gridLongEdge}`,temporal_frames:clipFrames,strength:clamp(r.strength,0,1,1.0)};}
// Fields the backend actually determines by writing the artifact. Everything
// else (mode, concept, description, grid, temporal frames, refinement,
// strength) is authored document state and must never be
// re-defaulted from a disk projection — that is what reset the RefMod inspector
// after ENCODE/SAVE.
const REFMOD_SERVER_FIELDS=["refmod_id","path","folder","state","kind","format_version","members"];
function applyRefmodServerState(local,returned){
  const merged={...(local&&typeof local==="object"?local:{})};
  if(returned&&typeof returned==="object")for(const key of REFMOD_SERVER_FIELDS)if(returned[key]!==undefined)merged[key]=returned[key];
  return normalizeRefmod(merged);
}

const styles = {
  bg: "#111318", panel: "#171a20", panel2: "#1d2128", border: "#303640", text: "#e7ebf0",
  dim: "#9199a5", main: "#315f78", camera: "#6b4d91", audio: "#39724f", accent: "#7ba6ff",
};

function cls(node) { return node?.comfyClass ?? node?.ComfyClass ?? node?.constructor?.comfyClass ?? node?.constructor?.ComfyClass ?? null; }
function isDirector(node) { return cls(node) === DIRECTOR_CLASS; }

function reconcileDirectorOutputs(node) {
  if (!node?.outputs) return false;
  const wanted = ["director", "report"];
  const keep = new Set();
  for (const name of wanted) {
    const index = node.outputs.findIndex((slot) => slot?.name === name);
    if (index >= 0) keep.add(index);
  }
  let changed = false;
  for (let i = node.outputs.length - 1; i >= 0; i -= 1) {
    const name = String(node.outputs[i]?.name || "");
    if (wanted.includes(name) && keep.has(i)) continue;
    // 0.5.41-0.5.46 exposed intermediate Director implementation outputs.
    // Saved LiteGraph nodes can retain those slots even after Python's RETURN_TYPES
    // changed. Remove them here so the graph presents the current single-wire contract.
    try {
      if (typeof node.removeOutput === "function") node.removeOutput(i);
      else {
        try { node.disconnectOutput?.(i); } catch (_) {}
        node.outputs.splice(i, 1);
      }
    } catch (_) {
      try { node.disconnectOutput?.(i); } catch (_) {}
      node.outputs.splice(i, 1);
    }
    changed = true;
  }
  if (changed) node.graph?.setDirtyCanvas?.(true, true);
  return changed;
}
function widget(node, name) { return node.widgets?.find((w) => w?.name === name); }
function id(prefix) { try { return `${prefix}-${crypto.randomUUID()}`; } catch (_) { return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`; } }
function safeProjectPart(value){const text=String(value||"director").replace(/[^A-Za-z0-9._-]+/g,"_").replace(/^[._-]+|[._-]+$/g,"");return (text||"director").slice(0,100);}
function stableProjectId(value,shots){const existing=String(value||"").trim();if(existing)return existing.slice(0,120);return `project-${safeProjectPart(shots?.[0]?.clip_id||"director")}`;}
function normalizeRegeneration(value,shots){const r=value&&typeof value==="object"?value:{};let mode=String(r.mode||"none").toLowerCase();if(!["none","clip","from_here","recast","recast_range"].includes(mode))mode="none";const ids=new Set((shots||[]).map(x=>String(x.clip_id||"")));let clip_id=String(r.clip_id||"").trim()||null,request_id=String(r.request_id||"").trim()||null,source_revision=String(r.source_revision||"").trim()||null,recast_subject_id=String(r.recast_subject_id||"").trim()||null;let preserve_audio=Boolean(r.preserve_audio??(["recast","recast_range"].includes(mode))),recast_video_denoise=1.0;let recast_reference_scale=String(r.recast_reference_scale||"auto").toLowerCase();if(!["auto","1.0","0.875","0.75","0.625","0.5"].includes(recast_reference_scale))recast_reference_scale="auto";let recast_clip_ids=(Array.isArray(r.recast_clip_ids)?r.recast_clip_ids:[]).map(String).filter(x=>ids.has(x)),source_revisions={};if(r.source_revisions&&typeof r.source_revisions==="object")for(const [k,v] of Object.entries(r.source_revisions)){const key=String(k),val=String(v||"").trim();if(ids.has(key)&&val)source_revisions[key]=val;}if(mode!=="none"&&(!ids.has(clip_id)||!request_id)){mode="none";clip_id=null;request_id=null;}if(mode==="recast"&&!source_revision){mode="none";clip_id=null;request_id=null;}if(mode==="recast_range"&&(!recast_clip_ids.length||recast_clip_ids.some(id=>!source_revisions[id]))){mode="none";clip_id=null;request_id=null;}if(mode==="none"){clip_id=null;request_id=null;source_revision=null;recast_subject_id=null;preserve_audio=false;recast_reference_scale="auto";recast_clip_ids=[];source_revisions={};}return {mode,clip_id,request_id,source_revision,recast_subject_id,preserve_audio,recast_video_denoise,recast_reference_scale,recast_clip_ids,source_revisions};}
function randomSeed(){try{const x=new Uint32Array(2);crypto.getRandomValues(x);return Number((BigInt(x[0])<<21n) ^ BigInt(x[1])) % Number.MAX_SAFE_INTEGER;}catch(_){return Math.floor(Math.random()*0x7fffffff);}}
function finite(v, fallback) { const n = Number(v); return Number.isFinite(n) ? n : fallback; }
function clamp(v, lo, hi, fallback) { return Math.max(lo, Math.min(hi, finite(v, fallback))); }
function directorUIState(node){
  node.properties=node.properties||{};
  const key="__lmd_ui_state_v1";
  const raw=node.properties[key];
  if(!raw||typeof raw!=="object"||Array.isArray(raw))node.properties[key]={};
  return node.properties[key];
}
function loadDirectorUIState(node){
  const s=directorUIState(node);
  const takeSortModes=["newest","oldest","name_az","name_za","current_clip","render_active"];
  if(!takeSortModes.includes(node.__lmdTakeSort))node.__lmdTakeSort=takeSortModes.includes(s.take_sort)?s.take_sort:"newest";
  if(!Number.isFinite(node.__lmdPps))node.__lmdPps=clamp(s.timeline_zoom,6,120,54);
  if(!Number.isFinite(node.__lmdTakesWidth))node.__lmdTakesWidth=clamp(s.takes_width,190,900,260);
  if(!Number.isFinite(node.__lmdInspectorHeight))node.__lmdInspectorHeight=clamp(s.inspector_height,260,900,420);
  if(!Number.isFinite(node.__lmdTimelineHeight))node.__lmdTimelineHeight=clamp(s.timeline_height,240,1100,320);
  if(!Number.isFinite(node.__lmdTrackHeight))node.__lmdTrackHeight=clamp(s.track_height,44,120,72);
  if(!node.__lmdTrackHeights||typeof node.__lmdTrackHeights!=="object"||Array.isArray(node.__lmdTrackHeights))node.__lmdTrackHeights={...(s.track_heights||{})};
  if(node.__lmdShowTakes===undefined)node.__lmdShowTakes=s.show_takes===undefined?true:Boolean(s.show_takes);
  if(node.__lmdShowRefmods===undefined)node.__lmdShowRefmods=s.show_refmods===undefined?true:Boolean(s.show_refmods);
  if(!["takes","refmods"].includes(node.__lmdLeftTab))node.__lmdLeftTab=["takes","refmods"].includes(s.left_tab)?s.left_tab:"takes";
  if(!["media","clip","sources"].includes(node.__lmdRightTab))node.__lmdRightTab=["media","clip","sources"].includes(s.right_tab)?s.right_tab:"media";
  if(!["clip","global","scene"].includes(node.__lmdBottomTab))node.__lmdBottomTab=["clip","global","scene"].includes(s.bottom_tab)?s.bottom_tab:"clip";
  if(!Number.isFinite(node.__lmdLeftWidth))node.__lmdLeftWidth=clamp(s.left_width,180,420,220);
  if(!Number.isFinite(node.__lmdRightWidth))node.__lmdRightWidth=clamp(s.right_width,220,500,280);
  if(!node.__lmdTextareaHeights||typeof node.__lmdTextareaHeights!=="object")node.__lmdTextareaHeights={...(s.textarea_heights||{})};
  if(!node.__lmdCoreTrackNames||typeof node.__lmdCoreTrackNames!=="object")node.__lmdCoreTrackNames={...(s.core_track_names||{})};
  if(!node.__lmdCoreTrackLocks||typeof node.__lmdCoreTrackLocks!=="object")node.__lmdCoreTrackLocks={...(s.core_track_locks||{})};
}
function saveDirectorUIPref(node,key,value){
  const s=directorUIState(node);s[key]=value;
  try{node.graph?.setDirtyCanvas?.(true,true);}catch(_){}
}
function timelineTrackHeight(node,key){
  const fallback=clamp(node.__lmdTrackHeight,44,120,72),value=node.__lmdTrackHeights?.[String(key||"")];
  return clamp(value,36,240,fallback);
}
function setTimelineTrackHeight(node,key,height,{persist=true}={}){
  const k=String(key||"");if(!k)return;
  node.__lmdTrackHeights=node.__lmdTrackHeights||{};
  node.__lmdTrackHeights[k]=Math.round(clamp(height,36,240,timelineTrackHeight(node,k)));
  if(persist)saveDirectorUIPref(node,"track_heights",{...node.__lmdTrackHeights});
}
function installTimelineTrackHeightGrip(node,key,label,row){
  if(!label||!row)return;
  const k=String(key||"");
  const apply=h=>{
    const height=Math.round(clamp(h,36,240,timelineTrackHeight(node,k)));
    label.style.height=`${height}px`;row.style.height=`${height}px`;
    for(const block of row.querySelectorAll('[data-lmd-timeline-block="1"]'))block.style.height=`${Math.max(30,height-12)}px`;
    for(const marker of row.querySelectorAll('[data-lmd-boundary="1"]'))marker.style.height=`${Math.max(32,height-6)}px`;
    return height;
  };
  const begin=e=>{
    if(e.button!==0)return;e.preventDefault();e.stopPropagation();
    const y0=e.clientY,h0=timelineTrackHeight(node,k);let latest=h0;
    const move=ev=>{latest=apply(h0+(ev.clientY-y0));};
    const up=()=>{window.removeEventListener("pointermove",move,true);window.removeEventListener("pointerup",up,true);window.removeEventListener("pointercancel",up,true);setTimelineTrackHeight(node,k,latest);render(node);};
    window.addEventListener("pointermove",move,true);window.addEventListener("pointerup",up,true);window.addEventListener("pointercancel",up,true);
  };
  const addGrip=host=>{
    host.style.position="relative";
    const grip=el("div",null,{position:"absolute",left:"0",right:"0",bottom:"0",height:"8px",zIndex:"80",cursor:"ns-resize",background:"transparent",pointerEvents:"auto"});
    const line=el("div",null,{position:"absolute",left:"0",right:"0",bottom:"0",height:"1px",background:styles.border,boxShadow:"0 -2px 0 rgba(123,166,255,0)"});
    grip.onmouseenter=()=>{line.style.background="#6f7f94";line.style.boxShadow="0 -2px 0 rgba(123,166,255,.16)";};
    grip.onmouseleave=()=>{line.style.background=styles.border;line.style.boxShadow="0 -2px 0 rgba(123,166,255,0)";};
    grip.onpointerdown=begin;grip.onclick=e=>{e.preventDefault();e.stopPropagation();};grip.oncontextmenu=e=>{e.preventDefault();e.stopPropagation();};attachDirectorTooltip(grip,"Drag this divider to resize only this timeline layer. The height is saved separately for each layer.");
    grip.append(line);host.append(grip);
  };
  addGrip(label);addGrip(row);
}
function coreTrackDefaultName(doc,kind){
  if(kind==="main")return String(doc?.setup_timeline_mode)==="multiclip"?"V1 · BASE / CLIPS":"V1 · BASE / MAIN";
  if(kind==="camera")return "C1 · CAMERA";
  if(kind==="audio")return "A1 · AUDIO";
  return String(kind||"LAYER").toUpperCase();
}
function coreTrackName(node,doc,kind){
  const value=String(node.__lmdCoreTrackNames?.[kind]||"").trim();
  return value||coreTrackDefaultName(doc,kind);
}
function renameCoreTrack(node,doc,kind){
  const current=coreTrackName(node,doc,kind),name=window.prompt("Rename layer",current);
  if(name==null)return;
  const clean=String(name||"").trim().slice(0,100);
  node.__lmdCoreTrackNames=node.__lmdCoreTrackNames||{};
  if(!clean||clean===coreTrackDefaultName(doc,kind))delete node.__lmdCoreTrackNames[kind];
  else node.__lmdCoreTrackNames[kind]=clean;
  saveDirectorUIPref(node,"core_track_names",{...node.__lmdCoreTrackNames});
  render(node);
}
function coreTrackLocked(node,kind){return Boolean(node.__lmdCoreTrackLocks?.[kind]);}
function toggleCoreTrackLock(node,kind){
  node.__lmdCoreTrackLocks=node.__lmdCoreTrackLocks||{};
  node.__lmdCoreTrackLocks[kind]=!Boolean(node.__lmdCoreTrackLocks[kind]);
  saveDirectorUIPref(node,"core_track_locks",{...node.__lmdCoreTrackLocks});
  render(node);
}
let __lmdMediaViewEpoch = 0;
function mediaPath(record) { const sub = String(record?.subfolder || "").replace(/^\/+|\/+$/g, ""); return sub ? `${sub}/${record.filename}` : String(record?.filename || ""); }
function mediaURL(record) {
  if (!record?.filename) return "";
  const base = api.apiURL(`/view?filename=${encodeURIComponent(record.filename)}&type=input&subfolder=${encodeURIComponent(record.subfolder || "")}`);
  return `${base}${base.includes("?")?"&":"?"}lmdv=${__lmdMediaViewEpoch}`;
}
function releaseImageResource(image){if(!image)return;image.onload=null;image.onerror=null;try{image.removeAttribute("src");}catch(_){} }
function releaseTakeGalleryImages(node){for(const image of node.__lmdTakeGalleryImages||[])releaseImageResource(image);node.__lmdTakeGalleryImages=[];}
function releaseTimelineVideoElements(root,node){for(const video of root?.querySelectorAll?.("video")||[]){if(video===node.__lmdSourceVideo)continue;video.onloadedmetadata=null;video.onerror=null;try{video.pause();video.removeAttribute("src");video.load();}catch(_){}}}
function invalidateDirectorMediaViews(node){
  __lmdMediaViewEpoch += 1;
  WAVE_CACHE.clear();
  releaseTakeGalleryImages(node);
  try{for(const entry of node.__lmdPreviewImages?.values?.()||[])releaseImageResource(entry?.image);node.__lmdPreviewImages?.clear?.();}catch(_){}
  try{node.__lmdSourceVideo?.pause?.();}catch(_){}
  if(node.__lmdSourceVideo){try{node.__lmdSourceVideo.removeAttribute("src");node.__lmdSourceVideo.dataset.lmdSrc="";node.__lmdSourceVideo.load?.();}catch(_){}}
  for(const entry of node.__lmdAudioPlayers?.values?.()||[]){try{entry.el?.pause?.();entry.el?.removeAttribute?.("src");entry.el?.load?.();}catch(_){}}
  try{node.__lmdAudioPlayers?.clear?.();}catch(_){}
}

function defaultShot(i=0, duration=5, start=null) { return { clip_id:id("shot"), name:`Shot ${i+1}`, prompt:"", prompt_guidance_mode:"inherit", motion_prompt_source:null, start:start==null?null:Math.max(0,finite(start,0)), duration, cut_marks:[], seed:null, subjects:[], media_subject_id:null, cast_subject_id:null, first_frame_anchor:false, last_frame_anchor:false, source_in:0, video_reference_scale:1, base_kind:"generated", run_enabled:true, audio_continuation:"AUTO" }; }
function defaultCameraBlock(start=0,duration=5,scene_id=null,scene_offset=0) { return { block_id:id("camera"), start, duration, scene_id, scene_offset, camera:{...DEFAULT_CAMERA}, camera_preset_id:null, note:"" }; }
function defaultDoc() {
  const shots=[defaultShot(0),defaultShot(1),defaultShot(2)];
  return { version:10, kind:"h3_longmedia_director", project_id:id("project"), fps:24, audio_mode:"auto", setup_h3_mode:"auto", setup_timeline_mode:"auto", render_mode:"final", segmented_count:3, segmented_duration:5, ripple_tracks:true, snapping:true, base_audio_muted:false, audio_track_muted:false, camera_track_enabled:true, audio_track_enabled:true,
    resolution_source:"base_layer", resolution:"1920x1080", megapixel:0.8,
    regeneration:{mode:"none",clip_id:null,request_id:null}, subjects:[], shots, merged_groups:[],
    camera_blocks:[defaultCameraBlock(0,5),defaultCameraBlock(5,5),defaultCameraBlock(10,5)], camera_preset_library:[], audio_blocks:[], extra_tracks:[], refmods:[], scene_guidance:[], scene_guidance_library:[] };
}
function normalizeMedia(m, kindHint="image") {
  if (!m || typeof m !== "object" || !String(m.filename||"").trim()) return null;
  let kind=String(m.kind||kindHint).toLowerCase(); if (!["image","video","audio"].includes(kind)) kind=kindHint;
  const out={kind, filename:String(m.filename), subfolder:String(m.subfolder||"")};
  if (finite(m.width,0)>0) out.width=Math.round(m.width); if (finite(m.height,0)>0) out.height=Math.round(m.height);
  if (finite(m.seconds,0)>0) out.seconds=Number(m.seconds); return out;
}
function normalizeSubject(s,i) {
  s=s&&typeof s==="object"?s:{}; let kind=String(s.kind||"Picture"); if (!["Picture","Video","Audio"].includes(kind)) kind="Picture";
  const max=kind==="Picture"?9:3; return { subject_id:String(s.subject_id||s.id||"").trim()||id("subject"), name:String(s.name||`Subject ${i+1}`).slice(0,160),
    kind, slot:Math.max(1,Math.min(max,Math.trunc(finite(s.slot,i+1)))), description:String(s.description||""),
    retention:String(s.retention||(kind==="Audio"?"reference":"fully_preserved")), media:normalizeMedia(s.media, kind==="Picture"?"image":kind.toLowerCase()) };
}
function normalizeShot(s,i,ids) { s=s&&typeof s==="object"?s:{}; let seed=s.seed; seed=(seed===""||seed==null)?null:Math.max(0,Math.trunc(finite(seed,0)));
  const subs=Array.isArray(s.subjects)?s.subjects.map(String).filter(x=>ids.has(x)):[]; const mid=String(s.media_subject_id||"").trim();
  const cast=String(s.cast_subject_id||"").trim();
  const duration=clamp(s.duration,.25,150,5),cutMarks=[...new Set((Array.isArray(s.cut_marks)?s.cut_marks:[]).map(value=>Math.round(finite(value,-1000)*1e6)/1e6).filter(value=>value>=0&&value<=duration))].sort((a,b)=>a-b).slice(0,128);
  return { clip_id:String(s.clip_id||s.id||"").trim()||id("shot"), name:String(s.name||`Shot ${i+1}`).slice(0,120), prompt:String(s.prompt||""), prompt_guidance_mode:["inherit",...PROMPT_GUIDANCE_OPTIONS.map(x=>x.value)].includes(String(s.prompt_guidance_mode||"inherit"))?String(s.prompt_guidance_mode||"inherit"):"inherit", motion_prompt_source:String(s.motion_prompt_source||"").trim()||null,
    start:(s.start==null||s.start==="")?null:Math.max(0,finite(s.start,0)), duration, cut_marks:cutMarks, seed, subjects:subs, media_subject_id:ids.has(mid)?mid:null, cast_subject_id:ids.has(cast)?cast:null,
    first_frame_anchor:Boolean(s.first_frame_anchor), last_frame_anchor:Boolean(s.last_frame_anchor), source_in:Math.max(0,finite(s.source_in,0)), video_reference_scale:clamp(s.video_reference_scale,.25,1,1),
    base_kind:String(s.base_kind||"generated").toLowerCase()==="media"?"media":"generated", run_enabled:s.run_enabled!==false, audio_continuation:["AUTO","CONTINUE","FRESH"].includes(String(s.audio_continuation||"AUTO").toUpperCase())?String(s.audio_continuation||"AUTO").toUpperCase():"AUTO" };
}
function normalizeCameraBlock(b,i) { b=b&&typeof b==="object"?b:{}; const c={...DEFAULT_CAMERA,...(b.camera&&typeof b.camera==="object"?b.camera:{})};
  return {block_id:String(b.block_id||b.id||"").trim()||id("camera"), start:Math.max(0,finite(b.start,0)), duration:clamp(b.duration,.25,600,5), scene_id:String(b.scene_id||"").trim()||null, scene_offset:finite(b.scene_offset,0), camera:c, camera_preset_id:String(b.camera_preset_id||"").trim()||null, note:String(b.note||"")}; }
function normalizeCameraPresetLibrary(raw){const out=[],seenIds=new Set(),seenNames=new Set();for(const item of (Array.isArray(raw)?raw:[]).slice(0,128)){if(!item||typeof item!=="object")continue;const presetId=String(item.id||item.preset_id||"").trim()||id("camera-preset"),name=String(item.name||"").trim().slice(0,100),nameKey=name.toLowerCase();if(!name||seenIds.has(presetId)||seenNames.has(nameKey))continue;seenIds.add(presetId);seenNames.add(nameKey);out.push({id:presetId.slice(0,160),name,camera:cameraPresetSettings(item.camera)});}return out;}
function cameraPresetSettings(camera){const source=camera&&typeof camera==="object"?camera:{};return Object.fromEntries(Object.keys(DEFAULT_CAMERA).map(key=>[key,key==="transition_to_next"?Boolean(source[key]??DEFAULT_CAMERA[key]):String(source[key]||DEFAULT_CAMERA[key])]));}
function cameraPresetMatches(left,right){const a=cameraPresetSettings(left),b=cameraPresetSettings(right);return Object.keys(DEFAULT_CAMERA).every(key=>a[key]===b[key]);}
function normalizeAudioBlock(b,i,ids) { b=b&&typeof b==="object"?b:{}; let kind=String(b.kind||"prompt"); if (!["prompt","reference"].includes(kind)) kind="prompt";
  let role=String(b.role||"diegetic"); if (!["diegetic","music","reactive"].includes(role)) role="diegetic"; const sid=String(b.subject_id||"").trim();
  return {block_id:String(b.block_id||b.id||"").trim()||id("audio"), start:Math.max(0,finite(b.start,0)), duration:clamp(b.duration,.25,600,5), scene_id:String(b.scene_id||"").trim()||null, scene_offset:finite(b.scene_offset,0), kind, role, prompt:String(b.prompt||""), subject_id:ids.has(sid)?sid:null, source_in:Math.max(0,finite(b.source_in,0))}; }
function defaultExtraTrack(type="prompt",name="") {
  type=EXTRA_TRACK_TYPES.includes(type)?type:"prompt";
  return {track_id:id("track"),type,name:name||EXTRA_TRACK_META[type].label+" "+(Date.now()%1000),enabled:true,locked:false,muted:false,prompt_guidance_mode:"none",clips:[]};
}
function defaultExtraClip(track,start=0,duration=5) {
  return {clip_id:id("layer"),name:(EXTRA_TRACK_META[track.type]?.label||"Layer")+" clip",start:Math.max(0,start),duration:clamp(duration,.25,600,5),prompt:"",embedding_name:"",subject_id:null,source_in:0,video_reference_scale:1};
}
function normalizeExtraClip(c,i,ids) {
  c=c&&typeof c==="object"?c:{};
  const sid=String(c.subject_id||"").trim();
  let emb=String(c.embedding_name||"").trim().replaceAll("\\","/");
  if(emb.toLowerCase().startsWith("embedding:"))emb=emb.slice(10).trim();
  if(emb.toLowerCase().endsWith(".safetensors"))emb=emb.slice(0,-12);
  const rid=String(c.refmod_id||"").trim()||null;
  return {clip_id:String(c.clip_id||c.id||"").trim()||id("layer"),name:String(c.name||"Layer "+(i+1)).slice(0,120),start:Math.max(0,finite(c.start,0)),duration:clamp(c.duration,.25,600,5),prompt:String(c.prompt||""),embedding_name:emb,subject_id:ids.has(sid)?sid:null,source_in:Math.max(0,finite(c.source_in,0)),refmod_id:rid,strength:clamp(c.strength,0,1,1.0),video_reference_scale:clamp(c.video_reference_scale,.25,1,1)};
}
function normalizeExtraTrack(t,i,ids) {
  t=t&&typeof t==="object"?t:{};
  let type=String(t.type||"prompt").toLowerCase();
  if(!EXTRA_TRACK_TYPES.includes(type))type="prompt";
  const rawClips=(Array.isArray(t.clips)?t.clips:[]).slice(0,MAX_EXTRA_CLIPS);
  const rawMode=String(t.prompt_guidance_mode||"").trim();
  const legacyMode=rawClips.map(c=>String(c?.prompt_guidance_mode||"")).find(mode=>PROMPT_GUIDANCE_OPTIONS.some(option=>option.value===mode&&mode!=="none"))||"none";
  const promptMode=PROMPT_GUIDANCE_OPTIONS.some(option=>option.value===rawMode)&&rawMode!="none"?rawMode:legacyMode;
  return {track_id:String(t.track_id||t.id||"").trim()||id("track"),type,name:String(t.name||EXTRA_TRACK_META[type].label+" "+(i+1)).slice(0,100),enabled:t.enabled!==false,locked:t.locked===true,muted:t.muted===true,prompt_guidance_mode:promptMode,clips:rawClips.map((c,j)=>normalizeExtraClip(c,j,ids))};
}
function normalizeMainShotStarts(shots){let cursor=0;for(const shot of shots||[]){const explicit=shot.start!=null&&Number.isFinite(Number(shot.start));shot.start=Math.max(0,explicit?Number(shot.start):cursor);shot.start=Math.round(shot.start*1e6)/1e6;cursor=Math.max(cursor,shot.start+finite(shot.duration,0));}shots.sort((a,b)=>finite(a.start,0)-finite(b.start,0));return shots;}

function migrateV1(d) {
  if (finite(d?.version,1)>=2) return d; const shots=Array.isArray(d?.shots)?d.shots:[]; let cursor=0; const cams=[], aud=[];
  const out=shots.map((s,i)=>{ const duration=clamp(s?.duration,.25,150,5); cams.push({block_id:id("camera"),start:cursor,duration,camera:{...DEFAULT_CAMERA,...(s?.camera||{})},note:""});
    const a=s?.audio||{}; for (const role of ["diegetic","music","reactive"]) if (String(a[role]||"").trim()) aud.push({block_id:id("audio"),start:cursor,duration,kind:"prompt",role,prompt:String(a[role]),subject_id:null,source_in:0});
    const start=cursor;cursor+=duration; return {clip_id:s?.clip_id||id("shot"),name:s?.name||`Shot ${i+1}`,prompt:s?.prompt||"",start,duration,seed:s?.seed??null,subjects:Array.isArray(s?.subjects)?s.subjects:[],media_subject_id:null,source_in:Math.max(0,finite(s?.source_in,0))}; });
  return {version:10,kind:"h3_longmedia_director",project_id:String(d?.project_id||""),fps:d?.fps||24,audio_mode:AUDIO_MODES.includes(String(d?.audio_mode||"auto"))?String(d.audio_mode||"auto"):"auto",setup_h3_mode:DIRECTOR_H3_MODES.includes(String(d?.setup_h3_mode||"auto"))?String(d.setup_h3_mode||"auto"):"auto",setup_timeline_mode:DIRECTOR_TIMELINE_MODES.includes(String(d?.setup_timeline_mode||"auto"))?String(d.setup_timeline_mode||"auto"):"auto",segmented_count:Math.max(1,Math.min(64,Math.trunc(finite(d?.segmented_count,3)))),segmented_duration:clamp(d?.segmented_duration,.25,150,5),regeneration:d?.regeneration,subjects:Array.isArray(d?.subjects)?d.subjects:[],shots:out,camera_blocks:cams,audio_blocks:aud};
}
function normalizeMergedGroups(rawGroups,shots){
  const indexByClip=new Map((shots||[]).map((shot,index)=>[String(shot.clip_id),index])),occupied=new Set(),out=[];
  for(const raw of Array.isArray(rawGroups)?rawGroups.slice(0,MAX_SHOTS):[]){
    if(!raw||typeof raw!=="object")continue;
    const ids=[...new Set((Array.isArray(raw.clip_ids)?raw.clip_ids:[]).map(String).filter(id=>indexByClip.has(id)))].sort((a,b)=>indexByClip.get(a)-indexByClip.get(b));
    if(ids.length<2||ids.some(id=>occupied.has(id)))continue;
    const first=indexByClip.get(ids[0]);
    if(ids.some((id,offset)=>indexByClip.get(id)!==first+offset))continue;
    for(const id of ids)occupied.add(id);
    out.push({group_id:String(raw.group_id||raw.id||"").trim()||id("merge"),name:String(raw.name||"Merged clip").trim().slice(0,120)||"Merged clip",clip_ids:ids});
  }
  return out;
}
function mergedGroupById(doc,groupId){return (doc?.merged_groups||[]).find(group=>String(group.group_id)===String(groupId))||null;}
function mergedGroupForClip(doc,clipId){return (doc?.merged_groups||[]).find(group=>(group.clip_ids||[]).some(value=>String(value)===String(clipId)))||null;}
function mergedGroupIndices(doc,group){const lookup=new Map((doc?.shots||[]).map((shot,index)=>[String(shot.clip_id),index]));return (group?.clip_ids||[]).map(clipId=>lookup.get(String(clipId))).filter(Number.isInteger);}
function normalizeDoc(d) {
  d=migrateV1(d&&typeof d==="object"?d:{}); const subjects=(Array.isArray(d.subjects)?d.subjects:[]).slice(0,MAX_SUBJECTS).map(normalizeSubject); const ids=new Set(subjects.map(x=>x.subject_id));
  let shots=(Array.isArray(d.shots)?d.shots:[]).slice(0,MAX_SHOTS).map((x,i)=>normalizeShot(x,i,ids)); if (!shots.length) shots=[defaultShot(0)]; normalizeMainShotStarts(shots);
  const subjectMap=new Map(subjects.map(s=>[s.subject_id,s]));
  for(const shot of shots){const anchorSubject=subjectMap.get(shot.media_subject_id);const valid=Boolean(anchorSubject?.kind==="Picture"&&anchorSubject?.media);if(!valid){shot.first_frame_anchor=false;shot.last_frame_anchor=false;}}
  // JSON edits/old experimental builds may contain duplicate role flags. Keep the
  // timeline deterministic: earliest FIRST and latest LAST win, matching UI intent.
  let seenFirst=false;for(const shot of shots){if(shot.first_frame_anchor){if(seenFirst)shot.first_frame_anchor=false;else seenFirst=true;}}
  let seenLast=false;for(let i=shots.length-1;i>=0;i--){if(shots[i].last_frame_anchor){if(seenLast)shots[i].last_frame_anchor=false;else seenLast=true;}}
  let cameras=(Array.isArray(d.camera_blocks)?d.camera_blocks:[]).slice(0,MAX_CAMERA).map(normalizeCameraBlock); if (!Array.isArray(d.camera_blocks)) { let c=0; cameras=shots.map(s=>{const b=defaultCameraBlock(c,s.duration);c+=s.duration;return b;}); }
  const audio=(Array.isArray(d.audio_blocks)?d.audio_blocks:[]).slice(0,MAX_AUDIO).map((x,i)=>normalizeAudioBlock(x,i,ids));
  const audioMode=AUDIO_MODES.includes(String(d.audio_mode||"auto"))?String(d.audio_mode||"auto"):"auto";
  const setupH3Mode=DIRECTOR_H3_MODES.includes(String(d.setup_h3_mode||"auto"))?String(d.setup_h3_mode||"auto"):"auto";
  const setupTimelineMode=DIRECTOR_TIMELINE_MODES.includes(String(d.setup_timeline_mode||"auto"))?String(d.setup_timeline_mode||"auto"):"auto";
  const segmentedCount=Math.max(1,Math.min(64,Math.trunc(finite(d.segmented_count,3))));
  const segmentedDuration=clamp(d.segmented_duration,.25,150,5);
  const resolutionSource=DIRECTOR_RESOLUTION_SOURCES.some(x=>x.value===String(d.resolution_source||"base_layer"))?String(d.resolution_source||"base_layer"):"base_layer";
  let resolution=String(d.resolution||"1920x1080").trim().replaceAll("×","x").replace(/\s+/g,"");if(!/^\d{2,5}x\d{2,5}$/i.test(resolution))resolution="1920x1080";
  const megapixel=clamp(d.megapixel,.1,8,.8);
  const renderMode=["preview","final"].includes(String(d.render_mode||"final").toLowerCase())?String(d.render_mode||"final").toLowerCase():"final";
  const project_id=stableProjectId(d.project_id,shots);
  const regeneration=normalizeRegeneration(d.regeneration,shots);
  const extra=(Array.isArray(d.extra_tracks)?d.extra_tracks:[]).slice(0,MAX_EXTRA_TRACKS).map((x,i)=>normalizeExtraTrack(x,i,ids));
  const refmods=(Array.isArray(d.refmods)?d.refmods:[]).slice(0,REFMOD_MAX).map((x)=>normalizeRefmod(x));
  const cameraPresetLibrary=normalizeCameraPresetLibrary(d.camera_preset_library);
  const sceneGuidance=normalizeSceneGuidanceItems(d.scene_guidance);
  const sceneGuidanceLibrary=normalizeSceneGuidanceItems(d.scene_guidance_library,{library:true});
  const doc=normalizeTrimRanges({version:10,kind:"h3_longmedia_director",project_id,fps:clamp(d.fps,1,120,24),audio_mode:audioMode,setup_h3_mode:setupH3Mode,setup_timeline_mode:setupTimelineMode,render_mode:renderMode,segmented_count:segmentedCount,segmented_duration:segmentedDuration,resolution_source:resolutionSource,resolution,megapixel,ripple_tracks:d.ripple_tracks!==false,snapping:d.snapping!==false,base_audio_muted:d.base_audio_muted===true,audio_track_muted:d.audio_track_muted===true,camera_track_enabled:d.camera_track_enabled!==false,audio_track_enabled:d.audio_track_enabled!==false,regeneration,subjects,shots,camera_blocks:cameras,camera_preset_library:cameraPresetLibrary,audio_blocks:audio,extra_tracks:extra,refmods,merged_groups:normalizeMergedGroups(d.merged_groups,shots),scene_guidance:sceneGuidance,scene_guidance_library:sceneGuidanceLibrary});
  if(setupTimelineMode==="segmented"&&shots.length===1&&d.segmented_count==null&&d.segmented_duration==null){const total=Math.max(.25,shots[0].duration),count=Math.max(1,Math.min(64,Math.round(total/5)));doc.segmented_count=count;doc.segmented_duration=Math.round(total/count*1000)/1000;}
  syncSegmentedTimeline(doc);bindLegacyBlocksToScenes(doc); reflowAnchoredBlocks(doc); return doc;
}
function timedMediaForShot(doc,shot){const s=subjectById(doc,shot?.media_subject_id);const m=s?.media;return m&&m.kind==="video"&&finite(m.seconds,0)>0?m:null;}
function timedMediaForAudio(doc,block){if(block?.kind!=="reference")return null;const s=subjectById(doc,block?.subject_id);const m=s?.media;return m&&finite(m.seconds,0)>0?m:null;}
function timedMediaForExtra(doc,clip){const s=subjectById(doc,clip?.subject_id);const m=s?.media;return m&&["video","audio"].includes(m.kind)&&finite(m.seconds,0)>0?m:null;}
function normalizeTrimRanges(doc){
  for(const shot of doc.shots||[]){const m=timedMediaForShot(doc,shot);if(m){shot.source_in=clamp(shot.source_in,0,Math.max(0,m.seconds-.25),0);shot.duration=clamp(shot.duration,.25,Math.max(.25,m.seconds-shot.source_in),shot.duration);}else shot.source_in=0;}
  for(const b of doc.audio_blocks||[]){const m=timedMediaForAudio(doc,b);if(m){b.source_in=clamp(b.source_in,0,Math.max(0,m.seconds-.25),0);b.duration=clamp(b.duration,.25,Math.max(.25,m.seconds-b.source_in),b.duration);}else b.source_in=0;}
  return doc;
}
function sceneInfoAt(doc,time){for(let i=0;i<doc.shots.length;i++){const shot=doc.shots[i],start=shotStart(doc,i),end=start+finite(shot.duration,0);if(time>=start&&(time<end||(i===doc.shots.length-1&&time<=end)))return {shot,index:i,start,end};}return null;}
function bindLegacyBlocksToScenes(doc){for(const list of [doc.camera_blocks||[],doc.audio_blocks||[]])for(const b of list){if(b.scene_id&&doc.shots.some(s=>s.clip_id===b.scene_id))continue;const hit=sceneInfoAt(doc,Math.max(0,finite(b.start,0)));if(hit){b.scene_id=hit.shot.clip_id;b.scene_offset=finite(b.start,0)-hit.start;}}}
function setBlockStart(doc,block,value){
  block.start=Math.max(0,finite(value,block.start));
  const hit=sceneInfoAt(doc,block.start);
  block.scene_id=hit?.shot?.clip_id||null;
  block.scene_offset=hit?block.start-hit.start:0;
}
function reflowAnchoredBlocks(doc){
  const starts=new Map();
  for(let i=0;i<doc.shots.length;i++){const shot=doc.shots[i];starts.set(shot.clip_id,shotStart(doc,i));}
  for(const list of [doc.camera_blocks||[],doc.audio_blocks||[]])for(const block of list){
    if(block.scene_id&&starts.has(block.scene_id)){
      if(doc.ripple_tracks===false)block.scene_offset=block.start-starts.get(block.scene_id);
      else block.start=Math.max(0,starts.get(block.scene_id)+finite(block.scene_offset,0));
    }
  }
  return doc;
}
function blocksForScene(doc,sceneId){return {camera:(doc.camera_blocks||[]).filter(b=>b.scene_id===sceneId),audio:(doc.audio_blocks||[]).filter(b=>b.scene_id===sceneId)};}
function snapshotDoc(doc){return JSON.stringify(normalizeDoc(doc));}
function pushHistory(node,previous){if(node.__lmdHistoryApplying)return;node.__lmdUndo=node.__lmdUndo||[];node.__lmdRedo=[];if(previous&&node.__lmdUndo[node.__lmdUndo.length-1]!==previous){node.__lmdUndo.push(previous);if(node.__lmdUndo.length>80)node.__lmdUndo.shift();}}
function applyHistory(node,serialized,destination){if(!serialized)return;const current=String(widget(node,"director_json")?.value||"");destination.push(current);node.__lmdHistoryApplying=true;try{const doc=normalizeDoc(JSON.parse(serialized));const w=widget(node,"director_json");if(w){w.value=JSON.stringify(doc);try{w.callback?.(w.value);}catch(_){}}node.__lmdDoc=doc;node.__lmdRegenPreview=null;render(node);}finally{node.__lmdHistoryApplying=false;}}
function undo(node){node.__lmdUndo=node.__lmdUndo||[];node.__lmdRedo=node.__lmdRedo||[];applyHistory(node,node.__lmdUndo.pop(),node.__lmdRedo);}
function redo(node){node.__lmdUndo=node.__lmdUndo||[];node.__lmdRedo=node.__lmdRedo||[];applyHistory(node,node.__lmdRedo.pop(),node.__lmdUndo);}
function parseDoc(node) { try { return normalizeDoc(JSON.parse(String(widget(node,"director_json")?.value||"{}"))); } catch (_) { return defaultDoc(); } }
function commit(node, doc, rerender=true) { reflowAnchoredBlocks(doc); const w=widget(node,"director_json"); const previous=String(w?.value||""); doc=normalizeDoc(doc); pushHistory(node,previous); if (w) { w.value=JSON.stringify(doc); try{w.callback?.(w.value);}catch(_){} } node.__lmdDoc=doc; if (rerender) render(node); node.setDirtyCanvas?.(true,true); app.canvas?.setDirty?.(true,true); }
function totalDuration(doc) { let end=0;for(let i=0;i<(doc.shots||[]).length;i++)end=Math.max(end,shotStart(doc,i)+finite(doc.shots[i]?.duration,0));return end; }
function timelineExtent(doc){let end=totalDuration(doc);for(const b of doc.camera_blocks||[])end=Math.max(end,finite(b.start,0)+finite(b.duration,0));for(const b of doc.audio_blocks||[])end=Math.max(end,finite(b.start,0)+finite(b.duration,0));for(const t of doc.extra_tracks||[])for(const c of t.clips||[])end=Math.max(end,finite(c.start,0)+finite(c.duration,0));return end;}
function shotStart(doc,index) { const shot=doc?.shots?.[index];if(shot&&shot.start!=null&&Number.isFinite(Number(shot.start)))return Math.max(0,Number(shot.start));let x=0;for(let i=0;i<index;i++)x=Math.max(x,shotStart(doc,i)+Number(doc.shots[i]?.duration||0));return x; }
function subjectById(doc,sid) { return doc.subjects.find(s=>s.subject_id===sid)||null; }
function token(s) { return s?`<${s.kind} ${s.slot}>`:""; }
function frameAnchorRoles(doc,sid){const roles=[];for(const shot of doc.shots||[]){if(String(shot.media_subject_id||"")!==String(sid||""))continue;if(shot.first_frame_anchor&&!roles.includes("FIRST"))roles.push("FIRST");if(shot.last_frame_anchor&&!roles.includes("LAST"))roles.push("LAST");}return roles;}
function runtimePictureSlots(doc){const active=new Set();for(const shot of doc.shots||[]){for(const sid of [shot.media_subject_id,shot.cast_subject_id,...(shot.subjects||[])])if(sid)active.add(String(sid));}for(const tr of doc.extra_tracks||[]){if(tr.enabled===false||tr.muted===true)continue;for(const c of tr.clips||[])if(c.subject_id)active.add(String(c.subject_id));}const anchors=new Set();for(const shot of doc.shots||[])if((shot.first_frame_anchor||shot.last_frame_anchor)&&shot.media_subject_id)anchors.add(String(shot.media_subject_id));const pics=(doc.subjects||[]).filter(s=>s.kind==="Picture"&&s.media&&active.has(String(s.subject_id))&&!anchors.has(String(s.subject_id))).sort((a,b)=>(finite(a.slot,1)-finite(b.slot,1))||String(a.subject_id).localeCompare(String(b.subject_id)));const out=new Map();pics.forEach((s,i)=>out.set(String(s.subject_id),i+1));return out;}
function runtimeToken(doc,s){if(!s)return "";if(s.kind!=="Picture")return token(s);const roles=frameAnchorRoles(doc,s.subject_id);if(roles.length)return roles.join("+");const slot=runtimePictureSlots(doc).get(String(s.subject_id));return slot?`<Picture ${slot}>`:`Picture asset #${s.slot} · inactive`;}
function nextSlot(doc,kind) { const max=kind==="Picture"?9:3; const used=new Set(doc.subjects.filter(s=>s.kind===kind).map(s=>s.slot)); for(let i=1;i<=max;i++) if(!used.has(i)) return i; return null; }

function el(tag, text="", css={}) { const n=document.createElement(tag); if (text!==null) n.textContent=text; Object.assign(n.style,css); return n; }
const BUTTON_TOOLTIPS = new Map([
  ["↶","Undo the last Director edit (Ctrl/Cmd+Z)."],["↷","Redo the last undone Director edit (Ctrl/Cmd+Y)."],
  ["Ripple ✓","Ripple is enabled: CAMERA and AUDIO blocks stay bound to their MAIN scene when the timeline changes."],["Ripple off","Ripple is disabled: secondary tracks keep absolute timing when MAIN changes."],
  ["+ Video Prompt","Add a new MAIN prompt clip after the current timeline."],["+ Sound Prompt","Add a generated/reactive AUDIO direction block for the selected scene."],["+ Camera Prompt","Add a CAMERA direction block for the selected scene."],
  ["+ Image","Import an image into WHO & WHAT."],["+ Picture","Import an image into WHO & WHAT."],["+ Audio","Import an audio reference into WHO & WHAT."],["+ Video","Import a video reference into WHO & WHAT."],
  ["refresh","Refresh cached takes, active revision and decoded preview metadata."],["Restore take","Restore this cached take and automatically recover its compatible cached downstream branch. The current suffix is not forced onto a historical take."],
  ["⟳ Regenerate Clip","Queue seam-safe regeneration of the selected clip."],["🎲 Reroll Clip","Queue a new-seed regeneration of the selected clip."],["↻ Regenerate From Here","Queue regeneration of the selected clip and all dependent clips after it."],
  ["Duplicate","Duplicate the selected MAIN clip and its scene-bound secondary blocks when Ripple is enabled."],["Delete","Delete the selected MAIN clip."],
  ["Delete camera block","Remove this CAMERA block."],["Delete audio block","Remove this AUDIO block."],
]);
function buttonTooltip(text){
  const raw=String(text??"").trim();
  if(BUTTON_TOOLTIPS.has(raw))return BUTTON_TOOLTIPS.get(raw);
  if(raw.startsWith("▶ Run "))return "Queue the regeneration plan shown above.";
  if(raw.includes("<Picture")||raw.includes("<Video")||raw.includes("<Audio"))return "Attach or detach this reference from the selected MAIN clip.";
  if(raw==="×")return "Remove this media asset and detach its references from the Director.";
  if(raw.includes("LOOP"))return "Toggle cyclic timeline playback.";
  return raw?`${raw} — Director action.`:"Director action.";
}
let __lmdTooltip=null,__lmdTooltipTimer=0;
function hideDirectorTooltip(){clearTimeout(__lmdTooltipTimer);__lmdTooltipTimer=0;if(__lmdTooltip){__lmdTooltip.remove();__lmdTooltip=null;}}
function attachDirectorTooltip(target,text){
  const message=String(text||"").trim();if(!target||!message)return;target.setAttribute("aria-label",message);target.dataset.lmdTooltip=message;
  target.addEventListener("mouseenter",()=>{clearTimeout(__lmdTooltipTimer);__lmdTooltipTimer=window.setTimeout(()=>{hideDirectorTooltip();const current=String(target.dataset.lmdTooltip||message);const tip=el("div",current,{position:"fixed",zIndex:"2147483000",maxWidth:"330px",padding:"7px 9px",background:"rgba(10,13,17,.97)",border:"1px solid #46505d",borderRadius:"6px",boxShadow:"0 8px 24px rgba(0,0,0,.38)",color:"#e8edf3",font:"10px/1.35 system-ui",pointerEvents:"none",whiteSpace:"normal"});document.body.append(tip);const r=target.getBoundingClientRect(),tr=tip.getBoundingClientRect();let left=r.left+(r.width-tr.width)/2,top=r.bottom+7;left=Math.max(8,Math.min(window.innerWidth-tr.width-8,left));if(top+tr.height>window.innerHeight-8)top=Math.max(8,r.top-tr.height-7);tip.style.left=`${left}px`;tip.style.top=`${top}px`;__lmdTooltip=tip;},260);});
  target.addEventListener("mouseleave",hideDirectorTooltip);target.addEventListener("pointerdown",hideDirectorTooltip);
}
function button(text, click, active=false, title="") {
  const b=el("button",text,{color:styles.text,borderRadius:"5px",padding:"5px 9px",font:"600 10.5px system-ui",cursor:"pointer",whiteSpace:"nowrap",transition:"background .12s ease,border-color .12s ease,box-shadow .12s ease"});
  b.type="button";b.dataset.lmdActive=active?"1":"0";const tooltip=title||buttonTooltip(text);attachDirectorTooltip(b,tooltip);b.onpointerdown=e=>e.stopPropagation();
  b.__lmdPaint=(hover=false)=>{const a=b.dataset.lmdActive==="1";b.style.background=hover?(a?"#416b9a":"#303640"):(a?"#365b86":"#252a32");b.style.border=`1px solid ${hover?(a?"#78a9df":"#586270"):(a?"#5f91cb":styles.border)}`;b.style.boxShadow=a?"0 0 0 1px rgba(123,166,255,.12)":"none";};
  b.onmouseenter=()=>b.__lmdPaint(true);b.onmouseleave=()=>b.__lmdPaint(false);b.__lmdPaint(false);
  b.onclick=e=>{e.stopPropagation();click?.(e);};return b;
}
function iconButton(symbol,title,click,active=false,size=30){const b=button(symbol,click,active,title);Object.assign(b.style,{width:`${size}px`,minWidth:`${size}px`,height:"28px",padding:"3px",display:"inline-grid",placeItems:"center",font:"700 13px/1 system-ui"});return b;}
function input(value,onchange,type="text") { const n=el("input",null,{width:"100%",boxSizing:"border-box",background:"#0f1115",border:`1px solid ${styles.border}`,color:styles.text,borderRadius:"5px",padding:"6px 7px",font:"11px system-ui",outline:"none"}); n.type=type; n.value=value??""; n.onpointerdown=e=>e.stopPropagation(); n.onchange=()=>onchange?.(n.value); return n; }
function textarea(value,oninput,rows=3) { const n=el("textarea",null,{width:"100%",boxSizing:"border-box",resize:"vertical",minHeight:`${rows*24}px`,background:"#0f1115",border:`1px solid ${styles.border}`,color:styles.text,borderRadius:"5px",padding:"7px",font:"11px/1.45 system-ui",outline:"none"}); n.value=value||""; n.onpointerdown=e=>e.stopPropagation(); n.oninput=()=>oninput?.(n.value); return n; }
function persistentTextarea(node,key,value,oninput,rows=3){
  loadDirectorUIState(node);
  node.__lmdTextareaHeights=node.__lmdTextareaHeights||{};
  const minH=rows*24,saved=Math.max(minH,finite(node.__lmdTextareaHeights[key],minH));
  const wrap=el("div",null,{position:"relative",width:"100%",boxSizing:"border-box",paddingBottom:"7px"});
  const n=textarea(value,oninput,rows);Object.assign(n.style,{height:`${saved}px`,minHeight:`${minH}px`,resize:"none",display:"block",overflowY:"auto"});
  const grip=el("div",null,{position:"absolute",left:"0",right:"0",bottom:"0",height:"7px",cursor:"ns-resize",borderRadius:"0 0 5px 5px",background:"linear-gradient(180deg,transparent,rgba(123,166,255,.16))"});
  grip.append(el("div",null,{position:"absolute",left:"50%",top:"3px",width:"32px",height:"2px",transform:"translateX(-50%)",borderRadius:"2px",background:"#566171"}));
  attachDirectorTooltip(grip,"Drag to resize this prompt editor. Its height stays fixed until you resize it and is shared by all clips of this prompt type.");
  grip.onpointerdown=e=>{e.preventDefault();e.stopPropagation();const y0=e.clientY,h0=n.getBoundingClientRect().height;let latest=h0;const move=ev=>{latest=Math.max(minH,Math.min(640,h0+(ev.clientY-y0)));n.style.height=`${Math.round(latest)}px`;};const up=()=>{window.removeEventListener("pointermove",move,true);window.removeEventListener("pointerup",up,true);window.removeEventListener("pointercancel",up,true);node.__lmdTextareaHeights[key]=Math.round(latest);saveDirectorUIPref(node,"textarea_heights",{...node.__lmdTextareaHeights});};window.addEventListener("pointermove",move,true);window.addEventListener("pointerup",up,true);window.addEventListener("pointercancel",up,true);};
  wrap.append(n,grip);wrap.__lmdTextarea=n;return wrap;
}
function select(values,value,onchange) { const n=el("select",null,{width:"100%",background:"#0f1115",border:`1px solid ${styles.border}`,color:styles.text,borderRadius:"4px",padding:"5px",font:"11px system-ui"}); for(const v of values){const o=document.createElement("option");o.value=v;o.textContent=v;o.selected=v===value;n.append(o);} n.onpointerdown=e=>e.stopPropagation(); n.onchange=()=>onchange?.(n.value); return n; }
function selectOptions(options,value,onchange){const n=el("select",null,{width:"100%",background:"#0f1115",border:`1px solid ${styles.border}`,color:styles.text,borderRadius:"4px",padding:"5px",font:"11px system-ui"});for(const item of options){const o=document.createElement("option");o.value=String(item.value??"");o.textContent=String(item.label??item.value??"");o.selected=String(o.value)===String(value??"");n.append(o);}n.onpointerdown=e=>e.stopPropagation();n.onchange=()=>onchange?.(n.value);return n;}
function renderSceneGuidanceBuilder(node,doc,host){
  const grid=el("div",null,{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"5px",marginTop:"7px"});
  for(const category of SCENE_GUIDANCE_CATEGORIES){
    const selected=(doc.scene_guidance||[]).find(item=>item.category===category.id),custom=(doc.scene_guidance_library||[]).filter(item=>item.category===category.id);
    const options=[{value:"",label:"— none —"},...(SCENE_GUIDANCE_BUILTINS[category.id]||[]).map(([key,name,prompt])=>({value:`builtin:${category.id}:${key}`,label:name,item:{id:`builtin:${category.id}:${key}`,category:category.id,name,prompt}})),...custom.map(item=>({value:item.id,label:`★ ${item.name}`,item}))];
    const control=selectOptions(options,selected?.id||"",value=>{doc.scene_guidance=(doc.scene_guidance||[]).filter(item=>item.category!==category.id);const choice=options.find(item=>item.value===value)?.item;if(choice)doc.scene_guidance.push({...choice});commit(node,doc);});
    grid.append(field(category.label,control));
  }
  host.append(el("div","SCENE-WIDE DIRECTION",{font:"800 9px system-ui",letterSpacing:".06em",color:"#c0c8d3",marginTop:"8px"}),grid);
  const customBox=el("div",null,{display:"grid",gridTemplateColumns:".7fr 1fr",gap:"5px",marginTop:"6px",paddingTop:"6px",borderTop:`1px solid ${styles.border}`});
  const categoryControl=selectOptions(SCENE_GUIDANCE_CATEGORIES.map(item=>({value:item.id,label:item.label})),node.__lmdSceneCustomCategory||"style",value=>{node.__lmdSceneCustomCategory=value;});
  const nameControl=input(node.__lmdSceneCustomName||"",()=>{},"text");
  nameControl.placeholder="Preset name";nameControl.value=node.__lmdSceneCustomName||"";nameControl.oninput=()=>{node.__lmdSceneCustomName=nameControl.value;};
  const direction=document.createElement("textarea");Object.assign(direction.style,{width:"100%",minHeight:"42px",boxSizing:"border-box",resize:"vertical",background:"#0f1115",border:`1px solid ${styles.border}`,color:styles.text,borderRadius:"5px",padding:"6px 7px",font:"10px system-ui"});direction.placeholder="Describe the global scene direction in positive language";direction.value=node.__lmdSceneCustomPrompt||"";direction.onpointerdown=e=>e.stopPropagation();direction.oninput=()=>{node.__lmdSceneCustomPrompt=direction.value;};
  customBox.append(field("CATEGORY",categoryControl),field("SAVE YOUR OWN PRESET",nameControl));host.append(customBox,direction);
  const actions=el("div",null,{display:"flex",gap:"5px",marginTop:"5px"});
  actions.append(button("SAVE CUSTOM",()=>{const category=SCENE_GUIDANCE_CATEGORIES.find(item=>item.id===String(node.__lmdSceneCustomCategory||"style"))?.id||"style",name=String(node.__lmdSceneCustomName||"").trim(),prompt=String(node.__lmdSceneCustomPrompt||"").trim();if(!name||!prompt){alert("Enter a preset name and scene direction first.");return;}const preset={id:id("look"),category,name:name.slice(0,100),prompt:prompt.slice(0,2000)};doc.scene_guidance_library=[...(doc.scene_guidance_library||[]),preset].slice(-128);doc.scene_guidance=(doc.scene_guidance||[]).filter(item=>item.category!==category);doc.scene_guidance.push({...preset});node.__lmdSceneCustomName="";node.__lmdSceneCustomPrompt="";commit(node,doc);},false,"Save this direction in the workflow and apply it globally"));
  actions.append(button("DELETE SELECTED CUSTOM",()=>{const selected=(doc.scene_guidance||[]).find(item=>(doc.scene_guidance_library||[]).some(saved=>saved.id===item.id));if(!selected)return;doc.scene_guidance_library=(doc.scene_guidance_library||[]).filter(item=>item.id!==selected.id);doc.scene_guidance=doc.scene_guidance.filter(item=>item.id!==selected.id);commit(node,doc);},false,"Remove the currently selected saved custom preset"));host.append(actions);
  const applied=(doc.scene_guidance||[]).length;host.append(el("div",`${applied} direction(s) apply across the scene; custom presets are saved in this workflow.`,{font:"9px/1.35 system-ui",color:styles.dim,marginTop:"4px"}));
}
function checkbox(label,checked,onchange,{disabled=false,title=""}={}){const wrap=el("label",null,{display:"inline-flex",alignItems:"center",gap:"6px",minHeight:"28px",padding:"4px 8px",border:`1px solid ${checked?styles.accent:styles.border}`,borderRadius:"5px",background:checked?"rgba(123,166,255,.13)":"#11151a",color:disabled?styles.dim:styles.text,font:"700 10px system-ui",cursor:disabled?"not-allowed":"pointer",userSelect:"none"});const box=document.createElement("input");box.type="checkbox";box.checked=Boolean(checked);box.disabled=Boolean(disabled);box.onpointerdown=e=>e.stopPropagation();box.onchange=()=>onchange?.(box.checked);wrap.append(box,document.createTextNode(label));if(title)wrap.title=title;return wrap;}
function outputPreviewURL(path,version=null){if(!path)return "";const clean=String(path).replace(/\\/g,"/");const at=clean.lastIndexOf("/");const sub=at>=0?clean.slice(0,at):"";const file=at>=0?clean.slice(at+1):clean;const v=version?`&v=${encodeURIComponent(String(version))}`:"";return api.apiURL(`/view?filename=${encodeURIComponent(file)}&type=output&subfolder=${encodeURIComponent(sub)}${v}`);}
function formatTime(seconds){const t=Math.max(0,finite(seconds,0));const m=Math.floor(t/60),sec=t-m*60;return `${String(m).padStart(2,"0")}:${sec.toFixed(2).padStart(5,"0")}`;}
function playheadHit(doc,time){const total=totalDuration(doc);if(total<=0)return null;return sceneInfoAt(doc,Math.min(Math.max(0,finite(time,0)),Math.max(0,total-1e-6)))||sceneInfoAt(doc,0);}
function previewImageEntry(node,src){if(!src)return null;node.__lmdPreviewImages=node.__lmdPreviewImages||new Map();let entry=node.__lmdPreviewImages.get(src);if(entry){node.__lmdPreviewImages.delete(src);node.__lmdPreviewImages.set(src,entry);return entry;}const image=new Image();entry={image,loaded:false,error:false};image.onload=()=>{entry.loaded=true;entry.error=false;updatePlayheadUI(node,node.__lmdDoc??parseDoc(node));};image.onerror=()=>{entry.error=true;entry.loaded=false;};node.__lmdPreviewImages.set(src,entry);image.src=src;while(node.__lmdPreviewImages.size>8){const oldest=node.__lmdPreviewImages.keys().next().value,stale=node.__lmdPreviewImages.get(oldest);node.__lmdPreviewImages.delete(oldest);releaseImageResource(stale?.image);}return entry;}
const LIVE_PREVIEW_SAMPLERS=new Set(["MiniMaxH3LatentLabLongMediaSampler","MiniMaxH3LatentLabUnifiedRuntimeSampler"]);
function downstreamSampler(node){const q=[node],seen=new Set();for(let depth=0;q.length&&depth<6;depth++){const next=[];for(const cur of q){if(!cur||seen.has(cur.id))continue;seen.add(cur.id);if(cur!==node&&LIVE_PREVIEW_SAMPLERS.has(String(cls(cur)||"")))return cur;for(const out of cur.outputs||[])for(const linkId of out?.links||[]){const link=cur.graph?.links?.[linkId],target=link?cur.graph?.getNodeById?.(link.target_id):null;if(target)next.push(target);}}q.splice(0,q.length,...next);}return null;}
function clearLiveLatentPreview(node){node.__lmdLatentPreviewLive=false;if(node.__lmdLatentPreviewUrl){try{URL.revokeObjectURL(node.__lmdLatentPreviewUrl);}catch(_){}node.__lmdLatentPreviewUrl=null;}node.__lmdLatentPreviewImage=null;if(node.__lmdPreviewCanvas?.isConnected)updatePlayheadUI(node,node.__lmdDoc??parseDoc(node));}
function drawLiveLatentPreview(node){const canvas=node.__lmdPreviewCanvas,img=node.__lmdLatentPreviewImage;if(!node.__lmdLatentPreviewEnabled||!node.__lmdLatentPreviewLive||!canvas?.isConnected||!img?.complete||!img.naturalWidth)return false;node.__lmdSourceVideo?.pause?.();if(node.__lmdSourceVideo)node.__lmdSourceVideo.style.display="none";canvas.style.display="block";drawPreviewImage(canvas,img);if(node.__lmdPreviewTitle?.isConnected)node.__lmdPreviewTitle.textContent="LATENT PREVIEW · LIVE";if(node.__lmdPreviewDetail?.isConnected)node.__lmdPreviewDetail.textContent="Sampler latent preview mirrored here · final rendered take returns automatically after Decode";return true;}
function acceptLiveLatentPreview(node,blob){if(!node.__lmdLatentPreviewEnabled||!(blob instanceof Blob))return;node.__lmdLatentPreviewLive=true;const old=node.__lmdLatentPreviewUrl,url=URL.createObjectURL(blob),img=new Image();node.__lmdLatentPreviewUrl=url;node.__lmdLatentPreviewImage=img;img.onload=()=>{if(node.__lmdLatentPreviewUrl===url)drawLiveLatentPreview(node);if(old)try{URL.revokeObjectURL(old);}catch(_){}};img.onerror=()=>{if(old)try{URL.revokeObjectURL(old);}catch(_){}};img.src=url;}
function previewQualityScale(node){const q=String(node.__lmdPreviewQuality||"Full");return q==="25"?.25:q==="50"?.5:q==="75"?.75:1;}
function resizePreviewCanvas(node,stage,{redraw=true}={}){const canvas=node.__lmdPreviewCanvas;if(!canvas?.isConnected||!stage?.isConnected)return;const rect=stage.getBoundingClientRect();if(rect.width<2||rect.height<2)return;const scale=previewQualityScale(node),dpr=Math.max(1,Math.min(2,Number(window.devicePixelRatio)||1));const w=Math.max(160,Math.round(rect.width*scale*dpr)),h=Math.max(90,Math.round(rect.height*scale*dpr));if(canvas.width===w&&canvas.height===h)return;canvas.width=w;canvas.height=h;const ctx=canvas.getContext("2d");ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality=scale>=.75?"high":scale>=.5?"medium":"low";if(redraw)requestAnimationFrame(()=>updatePlayheadUI(node,node.__lmdDoc??parseDoc(node)));}
function drawPreviewMessage(canvas,message){if(!canvas)return;const ctx=canvas.getContext("2d");ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle="#090b0e";ctx.fillRect(0,0,canvas.width,canvas.height);ctx.fillStyle="#9199a5";ctx.font="18px system-ui";ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText(message,canvas.width/2,canvas.height/2);}
function drawPreviewImage(canvas,image,sx=0,sy=0,sw=image?.naturalWidth||0,sh=image?.naturalHeight||0){if(!canvas||!image||!sw||!sh)return;const ctx=canvas.getContext("2d"),dw=canvas.width,dh=canvas.height;ctx.fillStyle="#090b0e";ctx.fillRect(0,0,dw,dh);const scale=Math.min(dw/sw,dh/sh),w=sw*scale,h=sh*scale,dx=(dw-w)/2,dy=(dh-h)/2;ctx.drawImage(image,sx,sy,sw,sh,dx,dy,w,h);}
function setButtonActive(buttonNode,active){if(!buttonNode)return;buttonNode.dataset.lmdActive=active?"1":"0";buttonNode.__lmdPaint?.(false);}
function syncTransportButtons(node){if(node.__lmdPlayButton){node.__lmdPlayButton.textContent=node.__lmdPlaying?"❚❚":"▶";const tip=node.__lmdPlaying?"Pause playback (Space)":"Play from the current playhead (Space)";node.__lmdPlayButton.dataset.lmdTooltip=tip;node.__lmdPlayButton.setAttribute("aria-label",tip);}setButtonActive(node.__lmdLoopButton,Boolean(node.__lmdLoop));if(node.__lmdLoopButton)node.__lmdLoopButton.textContent="↻";}
function selectedVideoSourceAt(node,doc,time){const sel=selection(node);if(sel.track==="main"){const index=doc.shots.findIndex(s=>s.clip_id===sel.id);if(index>=0){const shot=doc.shots[index],start=shotStart(doc,index),m=timedMediaForShot(doc,shot);if(m&&time>=start&&time<=start+shot.duration)return {record:m,sourceTime:finite(shot.source_in,0)+Math.max(0,time-start),label:shot.name,forced:false};}}if(String(sel.track||"").startsWith("extra:")){const tr=doc.extra_tracks.find(t=>t.track_id===String(sel.track).slice(6)),c=tr?.clips?.find(x=>x.clip_id===sel.id),m=timedMediaForExtra(doc,c);if(c&&m?.kind==="video"&&time>=c.start&&time<=c.start+c.duration)return {record:m,sourceTime:finite(c.source_in,0)+Math.max(0,time-c.start),label:c.name,forced:true};}return null;}
function sourceVideoAt(node,doc,time){const selected=selectedVideoSourceAt(node,doc,time);if(selected)return selected;const hit=playheadHit(doc,time);if(hit){const m=timedMediaForShot(doc,hit.shot);if(m)return {record:m,sourceTime:finite(hit.shot.source_in,0)+Math.max(0,time-hit.start),label:hit.shot.name,forced:false};}for(let i=(doc.extra_tracks||[]).length-1;i>=0;i--){const tr=doc.extra_tracks[i];if(tr.enabled===false||!["video"].includes(tr.type))continue;const c=(tr.clips||[]).find(x=>time>=x.start&&time<=x.start+x.duration),m=timedMediaForExtra(doc,c);if(c&&m?.kind==="video")return {record:m,sourceTime:finite(c.source_in,0)+Math.max(0,time-c.start),label:c.name,forced:false};}return null;}
function collectAudioSources(node,doc,time){const out=[];const hit=playheadHit(doc,time);if(hit){const status=node.__lmdTakeStatus?.clips?.[hit.shot.clip_id];const active=selectedTakeRecord(node,hit.shot,status)||status?.active;if(active?.preview_audio_file){const local=Math.max(0,time-hit.start);return [{key:`take:${hit.shot.clip_id}:${active.revision||"active"}`,url:outputPreviewURL(active.preview_audio_file),kind:"audio",sourceTime:local}];}}if(hit&&!doc.base_audio_muted){const m=timedMediaForShot(doc,hit.shot);if(m?.kind==="video")out.push({key:`base:${hit.shot.clip_id}`,record:m,sourceTime:finite(hit.shot.source_in,0)+Math.max(0,time-hit.start)});}if(!doc.audio_track_muted){for(const b of doc.audio_blocks||[]){if(b.kind!=="reference"||time<b.start||time>b.start+b.duration)continue;const m=timedMediaForAudio(doc,b);if(m)out.push({key:`audio:${b.block_id}`,record:m,sourceTime:finite(b.source_in,0)+Math.max(0,time-b.start)});}}for(const tr of doc.extra_tracks||[]){if(tr.enabled===false||tr.muted===true||!["audio","video","reference","character"].includes(tr.type))continue;for(const c of tr.clips||[]){if(time<c.start||time>c.start+c.duration)continue;const m=timedMediaForExtra(doc,c);if(m)out.push({key:`extra:${tr.track_id}:${c.clip_id}`,record:m,sourceTime:finite(c.source_in,0)+Math.max(0,time-c.start)});}}return out;}
function seekMediaElement(media,target,playing){const setTime=()=>{const dur=finite(media.duration,0),t=Math.max(0,dur>0?Math.min(Math.max(0,dur-.01),target):target);if(!playing||Math.abs(finite(media.currentTime,0)-t)>.16){try{media.currentTime=t;}catch(_){}}};if(media.readyState>=1)setTime();else media.addEventListener("loadedmetadata",setTime,{once:true});if(playing){if(media.paused)media.play().catch(()=>{});}else media.pause();}
function syncPlaybackAudio(node,doc,time){node.__lmdAudioPlayers=node.__lmdAudioPlayers||new Map();const desired=new Set();for(const src of collectAudioSources(node,doc,time)){desired.add(src.key);let entry=node.__lmdAudioPlayers.get(src.key),url=src.url||mediaURL(src.record);if(!entry||entry.url!==url){if(entry){entry.el?.pause?.();entry.el?.removeAttribute?.("src");entry.el?.load?.();}const kind=src.kind||src.record?.kind||"audio",media=document.createElement(kind==="video"?"video":"audio");media.preload="auto";media.src=url;media.volume=1;media.muted=false;media.playsInline=true;entry={el:media,url};node.__lmdAudioPlayers.set(src.key,entry);}seekMediaElement(entry.el,src.sourceTime,Boolean(node.__lmdPlaying));}for(const [key,entry] of node.__lmdAudioPlayers){if(!desired.has(key)){try{entry.el?.pause?.();entry.el?.removeAttribute?.("src");entry.el?.load?.();}catch(_){}node.__lmdAudioPlayers.delete(key);}}}
function pausePlaybackMedia(node){for(const entry of node.__lmdAudioPlayers?.values?.()||[])entry.el?.pause?.();node.__lmdSourceVideo?.pause?.();}
function syncSourceVideo(node,source,show){const v=node.__lmdSourceVideo,canvas=node.__lmdPreviewCanvas;if(!v)return;if(!show||!source?.record){v.style.display="none";v.pause();if(canvas)canvas.style.display="block";return;}const url=mediaURL(source.record);if(v.dataset.lmdSrc!==url){v.pause();v.src=url;v.dataset.lmdSrc=url;v.load();}v.style.display="block";if(canvas)canvas.style.display="none";seekMediaElement(v,source.sourceTime,Boolean(node.__lmdPlaying));}
function updatePlayheadUI(node,doc){if(!doc)return;const total=Math.max(0,totalDuration(doc)),t=Math.max(0,Math.min(total,finite(node.__lmdPlayhead,0)));node.__lmdPlayhead=t;const x=t*clamp(node.__lmdPps,6,120,54);for(const line of node.__lmdPlayheadLines||[])if(line?.isConnected)line.style.left=`${x}px`;if(node.__lmdPlayheadHandle?.isConnected)node.__lmdPlayheadHandle.style.left=`${x-6}px`;if(node.__lmdTimeDisplay?.isConnected)node.__lmdTimeDisplay.textContent=`${formatTime(t)} / ${formatTime(total)}`;
  const hit=playheadHit(doc,t),activeClipId=hit?.shot?.clip_id||null;if(activeClipId!==node.__lmdTimelineActiveClipId){const previous=node.__lmdMainTimelineBlocks?.get(node.__lmdTimelineActiveClipId);if(previous)previous.style.outline="";const current=node.__lmdMainTimelineBlocks?.get(activeClipId);if(current)current.style.outline="2px solid #54c7ff";node.__lmdTimelineActiveClipId=activeClipId;}if(drawLiveLatentPreview(node))return;const status=hit?node.__lmdTakeStatus?.clips?.[hit.shot.clip_id]:null,selectedTake=hit?selectedTakeRecord(node,hit.shot,status):null,active=selectedTake||status?.active,canvas=node.__lmdPreviewCanvas,source=sourceVideoAt(node,doc,t),hasDecoded=Boolean(active&&(active.preview_sprite_file||active.preview_file)),showSource=Boolean(source&&(source.forced||!hasDecoded));
  if(node.__lmdPreviewTitle?.isConnected)node.__lmdPreviewTitle.textContent=source&&showSource?`SOURCE · ${source.label||"Video"}`:hit?`PLAYHEAD ${t.toFixed(2)}s · ${hit.shot.name}`:`PLAYHEAD ${t.toFixed(2)}s`;
  if(node.__lmdPreviewDetail?.isConnected)node.__lmdPreviewDetail.textContent=showSource?`Original media · ${source.sourceTime.toFixed(2)}s · audio monitoring follows track mute states`:active?`${Boolean(active?.draft)||String(active?.state||"")==="draft"?"Selected empty take":"Selected take"} · seed ${active.effective_seed??"?"} · ${status.take_count||1} take(s)`:(hit?"This point has no cached rendered take yet.":"Outside MAIN timeline.");
  syncSourceVideo(node,source,showSource);syncPlaybackAudio(node,doc,t);if(showSource||!canvas?.isConnected)return;if(!active){drawPreviewMessage(canvas,"No decoded take preview yet");return;}
  const spriteSrc=outputPreviewURL(active.preview_sprite_file,active.preview_generated_at_unix);const indices=Array.isArray(active.preview_sprite_indices)?active.preview_sprite_indices:[];if(spriteSrc&&indices.length){const entry=previewImageEntry(node,spriteSrc);if(entry?.loaded){const local=Math.max(0,t-hit.start),fps=Math.max(.001,finite(active.preview_sprite_source_fps,doc.fps||24)),target=local*fps;let ordinal=0,best=Infinity;for(let i=0;i<indices.length;i++){const d=Math.abs(finite(indices[i],0)-target);if(d<best){best=d;ordinal=i;}}const cols=Math.max(1,Math.trunc(finite(active.preview_sprite_cols,1))),cw=Math.max(1,Math.trunc(finite(active.preview_sprite_cell_width,entry.image.naturalWidth))),ch=Math.max(1,Math.trunc(finite(active.preview_sprite_cell_height,entry.image.naturalHeight)));drawPreviewImage(canvas,entry.image,(ordinal%cols)*cw,Math.floor(ordinal/cols)*ch,cw,ch);return;}if(entry?.error){drawPreviewMessage(canvas,"Preview unavailable");return;}drawPreviewMessage(canvas,"Loading preview…");return;}
  const stillSrc=outputPreviewURL(active.preview_file,active.preview_generated_at_unix);if(stillSrc){const entry=previewImageEntry(node,stillSrc);if(entry?.loaded){drawPreviewImage(canvas,entry.image);return;}if(entry?.error){drawPreviewMessage(canvas,"Preview unavailable");return;}drawPreviewMessage(canvas,"Loading preview…");return;}drawPreviewMessage(canvas,"No decoded take preview yet");}
function setPlayheadTime(node,doc,time){node.__lmdPlayhead=Math.max(0,Math.min(totalDuration(doc),finite(time,0)));updatePlayheadUI(node,doc);}
function stopPlayback(node){node.__lmdPlaying=false;if(node.__lmdPlaybackRAF){cancelAnimationFrame(node.__lmdPlaybackRAF);node.__lmdPlaybackRAF=0;}pausePlaybackMedia(node);syncTransportButtons(node);}
function startPlayback(node,doc){const total=totalDuration(doc);if(total<=0)return;if(node.__lmdPlaying){stopPlayback(node);return;}if(finite(node.__lmdPlayhead,0)>=total-.001)setPlayheadTime(node,doc,0);node.__lmdPlaying=true;node.__lmdPlaybackBaseClock=performance.now();node.__lmdPlaybackBaseHead=finite(node.__lmdPlayhead,0);syncTransportButtons(node);updatePlayheadUI(node,doc);const tick=(now)=>{if(!node.__lmdPlaying)return;const currentDoc=node.__lmdDoc??doc,currentTotal=totalDuration(currentDoc);let t=node.__lmdPlaybackBaseHead+(now-node.__lmdPlaybackBaseClock)/1000;if(t>=currentTotal){if(node.__lmdLoop&&currentTotal>0){t=t%currentTotal;node.__lmdPlaybackBaseHead=t;node.__lmdPlaybackBaseClock=now;}else{setPlayheadTime(node,currentDoc,currentTotal);stopPlayback(node);return;}}setPlayheadTime(node,currentDoc,t);node.__lmdPlaybackRAF=requestAnimationFrame(tick);};node.__lmdPlaybackRAF=requestAnimationFrame(tick);}
function previousClipTime(doc,time){const hit=playheadHit(doc,time);if(!hit)return 0;if(time>hit.start+.05)return hit.start;return hit.index>0?shotStart(doc,hit.index-1):0;}
function nextClipTime(doc,time){const hit=playheadHit(doc,time);if(!hit)return 0;return hit.index+1<doc.shots.length?shotStart(doc,hit.index+1):totalDuration(doc);}
function renderTransport(node,doc,host){
  const row=el("div",null,{display:"grid",gridTemplateColumns:"auto 1fr auto",alignItems:"center",gap:"10px",minHeight:"42px",boxSizing:"border-box",padding:"6px 10px",marginTop:"8px",background:"linear-gradient(180deg,#181c22,#14171c)",border:`1px solid ${styles.border}`,borderRadius:"7px",boxShadow:"0 5px 18px rgba(0,0,0,.18)"});
  const zoom=el("div",null,{gridColumn:"1",display:"flex",alignItems:"center",justifyContent:"flex-start",gap:"4px",minWidth:"0"});
  zoom.append(el("span","⌕",{font:"12px system-ui",color:styles.dim}),button("−",()=>setTimelineZoom(node,doc,finite(node.__lmdPps,54)/1.25),false,"Zoom out timeline"));
  const slider=input(node.__lmdPps||54,v=>setTimelineZoom(node,doc,v),"range");slider.min="6";slider.max="120";slider.step="2";slider.style.width="62px";slider.oninput=()=>setTimelineZoom(node,doc,slider.value);node.__lmdZoomSlider=slider;
  zoom.append(slider,button("+",()=>setTimelineZoom(node,doc,finite(node.__lmdPps,54)*1.25),false,"Zoom in timeline"),button("FIT",()=>fitTimelineZoom(node,doc),false,"Fit the complete timeline in view"));
  zoom.append(el("span","↕",{font:"12px system-ui",color:styles.dim,marginLeft:"3px"}));
  const applyAllTrackHeights=value=>{node.__lmdTrackHeight=clamp(value,44,120,72);node.__lmdTrackHeights={};saveDirectorUIPref(node,"track_height",node.__lmdTrackHeight);saveDirectorUIPref(node,"track_heights",{});};
  const heightSlider=input(node.__lmdTrackHeight||72,v=>{applyAllTrackHeights(v);render(node);},"range");heightSlider.min="44";heightSlider.max="120";heightSlider.step="4";heightSlider.style.width="58px";heightSlider.title="Track and layer height";heightSlider.oninput=()=>applyAllTrackHeights(heightSlider.value);heightSlider.onchange=()=>render(node);node.__lmdTrackHeightSlider=heightSlider;zoom.append(heightSlider);
  const controls=el("div",null,{gridColumn:"2",display:"flex",alignItems:"center",justifyContent:"center",gap:"5px"});const mk=(symbol,title,fn,active=false,size=34)=>iconButton(symbol,title,fn,active,size);
  controls.append(mk("|◀","Go to timeline start (Home)",()=>{stopPlayback(node);setPlayheadTime(node,doc,0);}),mk("◀│","Go to previous clip boundary",()=>{stopPlayback(node);setPlayheadTime(node,doc,previousClipTime(doc,finite(node.__lmdPlayhead,0)));}),mk("←","Rewind one second (Left Arrow)",()=>{stopPlayback(node);setPlayheadTime(node,doc,finite(node.__lmdPlayhead,0)-1);}));
  const play=mk(node.__lmdPlaying?"❚❚":"▶",node.__lmdPlaying?"Pause playback (Space)":"Play from current playhead (Space)",()=>startPlayback(node,doc),node.__lmdPlaying,44);play.style.fontSize="13px";node.__lmdPlayButton=play;controls.append(play,mk("→","Advance one second (Right Arrow)",()=>{stopPlayback(node);setPlayheadTime(node,doc,finite(node.__lmdPlayhead,0)+1);}),mk("│▶","Go to next clip boundary",()=>{stopPlayback(node);setPlayheadTime(node,doc,nextClipTime(doc,finite(node.__lmdPlayhead,0)));}),mk("▶|","Go to timeline end (End)",()=>{stopPlayback(node);setPlayheadTime(node,doc,totalDuration(doc));}));
  const loop=mk("↻","Toggle cyclic timeline playback",()=>{node.__lmdLoop=!node.__lmdLoop;syncTransportButtons(node);},node.__lmdLoop);node.__lmdLoopButton=loop;controls.append(loop);
  const time=el("div",`${formatTime(finite(node.__lmdPlayhead,0))} / ${formatTime(totalDuration(doc))}`,{gridColumn:"3",justifySelf:"end",minWidth:"124px",textAlign:"right",font:"700 10px ui-monospace, SFMono-Regular, Consolas, monospace",color:"#ffcf67",letterSpacing:".02em"});node.__lmdTimeDisplay=time;row.append(zoom,controls,time);host.append(row);syncTransportButtons(node);
}
function field(label,control,hint="") { const w=el("label",null,{display:"block",font:"10px system-ui",color:styles.dim}); w.append(el("div",label,{marginBottom:"3px",fontWeight:"650",color:"#b7c0cc"}),control); if(hint) w.append(el("div",hint,{fontSize:"9px",opacity:".7",marginTop:"2px"})); return w; }
function grid(cols="1fr 1fr",gap="7px"){return el("div",null,{display:"grid",gridTemplateColumns:cols,gap});}
function videoReferenceScaleControl(value,onchange){const wrap=el("div",null,{display:"flex",alignItems:"center",gap:"8px",minHeight:"28px"}),slider=el("input",null,{flex:"1 1 auto",minWidth:"70px",accentColor:styles.accent,margin:"0"}),readout=el("span",`${Math.round(clamp(value,.25,1,1)*100)}%`,{minWidth:"36px",textAlign:"right",font:"700 11px ui-monospace,monospace",color:styles.text});slider.type="range";slider.min="25";slider.max="100";slider.step="5";slider.value=String(Math.round(clamp(value,.25,1,1)*100));slider.onpointerdown=e=>e.stopPropagation();slider.oninput=()=>{const percent=Math.max(25,Math.min(100,Number(slider.value)||100));readout.textContent=`${percent}%`;};slider.onchange=()=>onchange?.((Number(slider.value)||100)/100);wrap.append(slider,readout);return wrap;}

function hideWidget(w) { if(!w)return; if(!w.__lmdCaptured){w.__lmdCaptured=true;w.__lmdCompute=w.computeSize;w.__lmdDraw=w.draw;} w.hidden=true;w.options={...(w.options||{}),hidden:true};w.computeSize=()=>[0,-4];w.draw=()=>{}; }

async function pick(kind) { return new Promise(resolve=>{ const i=document.createElement("input");i.type="file";i.accept=kind==="refmod"?".safetensors":kind==="image"?"image/*":kind==="video"?"video/*":"audio/*";i.onchange=()=>resolve(i.files?.[0]||null);i.click(); }); }

const DROP_MIME = "application/x-longmedia-director-subject";
const EXT_KIND = new Map([
  ["png","image"],["jpg","image"],["jpeg","image"],["webp","image"],["gif","image"],["bmp","image"],["tif","image"],["tiff","image"],
  ["mp4","video"],["mov","video"],["mkv","video"],["webm","video"],["avi","video"],["m4v","video"],["mts","video"],["m2ts","video"],
  ["wav","audio"],["mp3","audio"],["flac","audio"],["m4a","audio"],["aac","audio"],["ogg","audio"],["opus","audio"],["aiff","audio"],["aif","audio"],
]);
function fileKind(file) {
  const type=String(file?.type||"").toLowerCase();
  for(const kind of ["image","video","audio"]) if(type.startsWith(`${kind}/`)) return kind;
  const name=String(file?.name||""); const dot=name.lastIndexOf(".");
  if(dot>=0&&name.slice(dot+1).toLowerCase()==="safetensors")return "refmod";
  return dot>=0 ? (EXT_KIND.get(name.slice(dot+1).toLowerCase())||null) : null;
}
function allDroppedFiles(event){return Array.from(event?.dataTransfer?.files||[]);}
function droppedFiles(event){return allDroppedFiles(event).filter(file=>fileKind(file));}
function dropSeconds(event,row,pps){const rect=row.getBoundingClientRect();return Math.max(0,(event.clientX-rect.left)/Math.max(1,pps));}
function shotAtTime(doc,time){let cursor=0;for(let i=0;i<doc.shots.length;i++){const shot=doc.shots[i],end=cursor+shot.duration;if(time>=cursor&&time<end)return {shot,index:i,start:cursor,end};cursor=end;}return null;}
function showDropHint(host,text){
  host.style.boxShadow=`inset 0 0 0 2px ${styles.accent}`;
  if(host.__lmdDropHint){host.__lmdDropHint.textContent=text;return;}
  const hint=el("div",text,{position:"absolute",inset:"4px",zIndex:"30",display:"grid",placeItems:"center",background:"rgba(22,28,38,.72)",border:`1px dashed ${styles.accent}`,borderRadius:"4px",color:"#dce8ff",font:"700 10px system-ui",letterSpacing:".02em",pointerEvents:"none"});
  host.__lmdDropHint=hint;host.append(hint);
}
function clearDropHint(host){host.style.boxShadow="";host.__lmdDropHint?.remove();host.__lmdDropHint=null;}
function subjectDropId(event){try{return String(event?.dataTransfer?.getData(DROP_MIME)||"").trim();}catch(_){return "";}}
function dropHasPayload(event){const types=Array.from(event?.dataTransfer?.types||[]);return allDroppedFiles(event).length>0||types.includes("Files")||types.includes(DROP_MIME)||Boolean(subjectDropId(event));}
function attachVisualSubject(node,doc,subject,time){
  let hit=shotAtTime(doc,time),created=false;
  if(!hit){
    if(doc.shots.length>=MAX_SHOTS){alert("LongMedia Director: maximum shot count reached.");return;}
    const duration=subject.kind==="Video"&&subject.media?.seconds?clamp(subject.media.seconds,.25,15,5):5;
    const shot=defaultShot(doc.shots.length,duration);doc.shots.push(shot);hit={shot,index:doc.shots.length-1,start:totalDuration(doc)-duration,end:totalDuration(doc)};created=true;
  }
  const shot=hit.shot;
  // A video dropped onto a newly-created or deliberately empty MAIN BASE block is
  // timeline media, not a generated-video reference. Promote it immediately so
  // MEDIA -> GENERATED continuation never falls back to sampling the imported prefix.
  // Non-empty GENERATED shots keep their historical behavior: a dropped Video can
  // still be used as a visual reference unless the user explicitly changes BASE TYPE.
  const emptyBase=!shot.media_subject_id&&!(shot.subjects||[]).length&&!String(shot.prompt||"").trim();
  shot.media_subject_id=subject.subject_id;if(!shot.subjects.includes(subject.subject_id))shot.subjects.push(subject.subject_id);
  if(subject.kind==="Video"&&(created||emptyBase||String(shot.base_kind||"generated")==="media")){
    shot.base_kind="media";shot.source_in=0;shot.audio_continuation=shot.audio_continuation||"AUTO";
    console.info(`[MiniMaxH3 LongMedia][MEDIA DROP] clip=${shot.clip_id} promoted_to=MEDIA source=${subject.subject_id}`);
  }
  node.__lmdSelection={track:"main",id:shot.clip_id};
}
function attachAudioSubject(node,doc,subject,time){
  const duration=clamp(subject.media?.seconds, .25, 600, 5);
  const hit=sceneInfoAt(doc,Math.max(0,time));const block={block_id:id("audio"),start:Math.max(0,time),duration,scene_id:hit?.shot?.clip_id||null,scene_offset:hit?Math.max(0,time)-hit.start:0,kind:"reference",role:"diegetic",prompt:"",subject_id:subject.subject_id,source_in:0};
  doc.audio_blocks.push(block);node.__lmdSelection={track:"audio",id:block.block_id};
}
async function importDroppedFile(node,doc,file,target,time=0){
  const kind=fileKind(file);if(!kind)return null;
  const effectiveTarget=target==="smart"?(kind==="audio"?"audio":"main"):target;
  const record=await upload(kind,file);
  // A video dropped onto AUDIO means "use this file's soundtrack". It is represented
  // as an <Audio N> subject whose media record still says video, so runtime can extract
  // the real soundtrack without also turning the video into a visual reference.
  const semanticKind=(effectiveTarget==="audio"&&kind==="video")?"audio":kind;
  const subject=await addMediaSubject(node,doc,semanticKind,record,file.name);if(!subject)return null;
  if(effectiveTarget==="assets") return subject;
  if(effectiveTarget==="audio" || kind==="audio") attachAudioSubject(node,doc,subject,time);
  else attachVisualSubject(node,doc,subject,time);
  return subject;
}
async function handleTimelineDrop(node,doc,event,target,row,pps,targetTrack=null){
  event.preventDefault();event.stopPropagation();clearDropHint(row);
  const time=snapTime(node,doc,dropSeconds(event,row,pps),pps,{includePlayhead:true});
  const files=droppedFiles(event),refmodFiles=files.filter(file=>fileKind(file)==="refmod");
  if(refmodFiles.length){try{for(const file of refmodFiles)await importRefmodFile(node,doc,file,{folder:node.__lmdRefmodFolder||"Unsorted",track:targetTrack,time});return;}catch(err){alert(`Import RefMod failed: ${err?.message||err}`);return;}}
  const existing=subjectDropId(event);
  if(existing){const subject=subjectById(doc,existing);if(subject){
      const effectiveTarget=target==="smart"?(subject.kind==="Audio"?"audio":"main"):target;
      if(effectiveTarget==="audio"){
        if(subject.kind==="Audio") attachAudioSubject(node,doc,subject,time);
        else if(subject.kind==="Video") {
          const slot=nextSlot(doc,"Audio");
          if(slot==null){alert("No free Audio slots remain.");return;}
          const audioAlias={subject_id:id("subject"),name:`${subject.name} · soundtrack`,kind:"Audio",slot,description:`Soundtrack extracted from ${token(subject)}.`,retention:"reference",media:subject.media?{...subject.media}:null};
          doc.subjects.push(audioAlias);attachAudioSubject(node,doc,audioAlias,time);
        } else attachVisualSubject(node,doc,subject,time);
      } else if(subject.kind==="Audio") attachAudioSubject(node,doc,subject,time);
      else attachVisualSubject(node,doc,subject,time);
      commit(node,doc);return;
  }}
  const mediaFiles=files.filter(file=>fileKind(file)!=="refmod");if(!mediaFiles.length){if(allDroppedFiles(event).length)alert("LongMedia Director: unsupported file type.");return;}
  try{for(const file of mediaFiles)await importDroppedFile(node,doc,file,target,time);commit(node,doc);}catch(err){console.error("LongMedia Director drop failed",err);alert(`LongMedia Director: ${err?.message||err}`);}
}
function installTimelineDrop(row,node,doc,target,pps,targetTrack=null){
  const hint=targetTrack?`DROP MEDIA ON ${String(targetTrack.name||targetTrack.type||"LAYER").toUpperCase()}`:target==="audio"?"DROP AUDIO / VIDEO SOUNDTRACK HERE":target==="smart"?"DROP MEDIA · ROUTES TO MAIN / AUDIO":"DROP MEDIA OR REFMOD HERE";
  row.ondragenter=e=>{if(!dropHasPayload(e))return;e.preventDefault();e.stopPropagation();showDropHint(row,hint);};
  row.ondragover=e=>{if(!dropHasPayload(e))return;e.preventDefault();e.stopPropagation();if(e.dataTransfer)e.dataTransfer.dropEffect="copy";showDropHint(row,hint);};
  row.ondragleave=e=>{if(e.relatedTarget&&row.contains(e.relatedTarget))return;clearDropHint(row);};
  row.ondrop=e=>{if(!targetTrack)return handleTimelineDrop(node,doc,e,target,row,pps,targetTrack);e.preventDefault();e.stopPropagation();clearDropHint(row);void handleDirectorLayerFileDrop(node,doc,e,row,targetTrack).catch(err=>{console.error("LongMedia Director layer drop failed",err);alert(`LongMedia Director: ${err?.message||err}`);});};
}
function installAssetDrop(host,node,doc){
  host.ondragenter=e=>{if(!allDroppedFiles(e).length&&!Array.from(e?.dataTransfer?.types||[]).includes("Files"))return;e.preventDefault();e.stopPropagation();showDropHint(host,"DROP MEDIA TO WHO & WHAT");};
  host.ondragover=e=>{if(!allDroppedFiles(e).length&&!Array.from(e?.dataTransfer?.types||[]).includes("Files"))return;e.preventDefault();e.stopPropagation();if(e.dataTransfer)e.dataTransfer.dropEffect="copy";showDropHint(host,"DROP MEDIA TO WHO & WHAT");};
  host.ondragleave=e=>{if(e.relatedTarget&&host.contains(e.relatedTarget))return;clearDropHint(host);};
  host.ondrop=async e=>{e.preventDefault();e.stopPropagation();clearDropHint(host);const files=droppedFiles(e);if(!files.length){if(allDroppedFiles(e).length)alert("LongMedia Director: unsupported file type.");return;}try{for(const file of files)await importDroppedFile(node,doc,file,"assets",0);commit(node,doc);}catch(err){console.error("LongMedia Director asset drop failed",err);alert(`LongMedia Director: ${err?.message||err}`);}};
}
async function handleDirectorLayerFileDrop(node,doc,event,row,track){
  if(!track)throw new Error("The target Director layer is no longer available.");
  if(track.locked)throw new Error("Unlock the layer before dropping media on it.");
  const files=droppedFiles(event),time=snapTime(node,doc,dropSeconds(event,row,finite(node.__lmdPps,54)),finite(node.__lmdPps,54),{includePlayhead:true});
  if(!files.length){if(allDroppedFiles(event).length)throw new Error("Unsupported file type.");return;}
  let selectedClip=null;
  for(const file of files){
    const kind=fileKind(file);
    if(kind==="refmod"){await importRefmodFile(node,doc,file,{folder:node.__lmdRefmodFolder||"Unsorted",track,time});continue;}
    const allowed=track.type==="character"?["image"]:track.type==="reference"?["image","video"]:track.type==="video"?["video"]:track.type==="audio"?["audio","video"]:[];
    if(!allowed.includes(kind))throw new Error(`This ${track.type} layer accepts ${allowed.length?allowed.join(" or "):"text only"} media.`);
    const semanticKind=track.type==="audio"&&kind==="video"?"audio":kind;
    const record=await upload(kind,file),activeClip=(track.clips||[]).find(item=>time>=finite(item.start,0)&&time<finite(item.start,0)+finite(item.duration,0));
    const clip=activeClip||defaultExtraClip(track,time,clamp(record.seconds,.25,600,5));
    if(!activeClip)track.clips.push(clip);
    const subject=await addMediaSubject(node,doc,semanticKind,record,file.name);
    if(subject){clip.subject_id=subject.subject_id;if(!clip.name||/^Layer clip$/i.test(clip.name))clip.name=file.name.slice(0,120);}
    selectedClip=clip;
  }
  if(selectedClip){node.__lmdSelection={track:`extra:${track.track_id}`,id:selectedClip.clip_id};commit(node,doc);}
}
async function durationOf(record) { const src=mediaURL(record); if(!src) return null; return new Promise(resolve=>{ const p=document.createElement(record.kind==="audio"?"audio":"video");p.preload="metadata";p.onloadedmetadata=()=>resolve(Number.isFinite(p.duration)&&p.duration>0?p.duration:null);p.onerror=()=>resolve(null);p.src=src; }); }
async function imageSize(record) { const src=mediaURL(record); if(!src) return null; return new Promise(resolve=>{const im=new Image();im.onload=()=>resolve({width:im.naturalWidth,height:im.naturalHeight});im.onerror=()=>resolve(null);im.src=src;}); }
async function upload(kind,file) { if(kind==="refmod"){const body=new FormData();body.append("file",file,file.name);const r=await api.fetchApi("/longmedia/refmods/upload",{method:"POST",body});const x=await r.json();if(!r.ok)throw new Error(x.error||`upload failed: ${r.status}`);return {kind,filename:x.name||file.name,subfolder:x.subfolder||""};}const body=new FormData();body.append("image",file);body.append("type","input");body.append("overwrite","false"); const r=await api.fetchApi("/upload/image",{method:"POST",body});if(!r.ok)throw new Error(`upload failed: ${r.status}`);const x=await r.json();const rec={kind,filename:x.name||file.name,subfolder:x.subfolder||""}; if(kind==="image"){const s=await imageSize(rec);if(s)Object.assign(rec,s);} else {const sec=await durationOf(rec);if(sec)rec.seconds=sec;} return rec; }
const WAVE_CACHE=new Map();
async function peaks(src){if(WAVE_CACHE.has(src))return WAVE_CACHE.get(src);const work=(async()=>{try{const bytes=await(await fetch(src)).arrayBuffer();const C=window.AudioContext||window.webkitAudioContext;const ctx=new C();try{const a=await ctx.decodeAudioData(bytes);const d=a.getChannelData(0),b=1200,step=Math.max(1,Math.floor(d.length/b)),p=new Float32Array(b);for(let j=0;j<b;j++){let m=0;for(let k=0;k<step;k++)m=Math.max(m,Math.abs(d[j*step+k]||0));p[j]=m;}return p;}finally{ctx.close();}}catch(_){return null;}})();WAVE_CACHE.set(src,work);return work;}
async function drawWave(canvas,record,sourceIn=0,sourceDuration=null){
  const src=mediaURL(record),p=await peaks(src);if(!p||!canvas.isConnected)return;
  const w=Math.max(1,canvas.clientWidth||180),h=Math.max(1,canvas.clientHeight||42);canvas.width=w;canvas.height=h;
  const total=Math.max(.0001,finite(record?.seconds,0)||1);const start=Math.max(0,Math.min(total,finite(sourceIn,0)));
  const duration=sourceDuration==null?Math.max(.0001,total-start):Math.max(.0001,finite(sourceDuration,total-start));
  const end=Math.min(total,start+duration);const from=Math.max(0,Math.min(p.length-1,Math.floor((start/total)*p.length)));
  const to=Math.max(from+1,Math.min(p.length,Math.ceil((end/total)*p.length)));
  const c=canvas.getContext("2d");c.clearRect(0,0,w,h);c.fillStyle="#79c995";const span=Math.max(1,to-from),step=span/w;
  for(let x=0;x<w;x++){let m=0;const a=from+Math.floor(x*step),b=Math.max(a+1,from+Math.floor((x+1)*step));for(let i=a;i<Math.min(to,b);i++)m=Math.max(m,p[i]||0);const bar=Math.max(1,m*h);c.fillRect(x,(h-bar)/2,1,bar);}
}
function decorateVideoStrip(host,record,sourceIn=0,sourceDuration=null,opacity=.65){
  const src=mediaURL(record);if(!src)return;const total=Math.max(.01,finite(record?.seconds,0)||.01);const start=Math.max(0,Math.min(total-.01,finite(sourceIn,0)));
  const dur=Math.max(.01,Math.min(sourceDuration==null?total-start:finite(sourceDuration,total-start),total-start));const count=4;
  for(let i=0;i<count;i++){const v=document.createElement("video");Object.assign(v.style,{position:"absolute",left:`${i*25}%`,top:"0",width:"25.2%",height:"100%",objectFit:"cover",opacity:String(opacity),pointerEvents:"none",borderRight:i<count-1?"1px solid rgba(255,255,255,.09)":"none"});v.src=src;v.muted=true;v.preload="metadata";v.onloadedmetadata=()=>{try{const t=start+dur*((i+.5)/count);v.currentTime=Math.max(.01,Math.min(v.duration||total,t));}catch(_){}};host.prepend(v);}
}
function decorateMedia(host,record,{audioVisual=false,sourceIn=0,sourceDuration=null}={}) {
  if(!record)return; const src=mediaURL(record); if(!src)return; host.style.overflow="hidden";
  if(record.kind==="image"){host.style.backgroundImage=`linear-gradient(90deg,rgba(0,0,0,.12),rgba(0,0,0,.38)),url("${src}")`;host.style.backgroundSize="auto 100%";host.style.backgroundRepeat="repeat-x";host.style.backgroundPosition="center";}
  else if(record.kind==="video"){decorateVideoStrip(host,record,sourceIn,sourceDuration,audioVisual?.22:.62);if(audioVisual){const c=el("canvas",null,{position:"absolute",inset:"0",width:"100%",height:"100%",pointerEvents:"none",zIndex:"2"});host.prepend(c);drawWave(c,record,sourceIn,sourceDuration);}}
  else if(record.kind==="audio"){const c=el("canvas",null,{position:"absolute",inset:"0",width:"100%",height:"100%",pointerEvents:"none"});host.prepend(c);drawWave(c,record,sourceIn,sourceDuration);}
}
function mediaSourceCaption(record,sourceIn,duration){if(!record||finite(record.seconds,0)<=0)return "";const a=Math.max(0,finite(sourceIn,0)),b=Math.min(finite(record.seconds,0),a+finite(duration,0));return `src ${a.toFixed(2)}–${b.toFixed(2)}s`; }
function timelineSnapPoints(doc,excludeIds=[]){const exclude=new Set((excludeIds||[]).map(String)),points=[0,totalDuration(doc),timelineExtent(doc)];for(let i=0;i<(doc.shots||[]).length;i++){const shot=doc.shots[i],start=shotStart(doc,i),end=start+finite(shot.duration,0);if(!exclude.has(String(shot.clip_id)))points.push(start,end);for(const mark of shot.cut_marks||[])points.push(start+finite(mark,0));}for(const b of doc.camera_blocks||[])if(!exclude.has(String(b.block_id)))points.push(finite(b.start,0),finite(b.start,0)+finite(b.duration,0));for(const b of doc.audio_blocks||[])if(!exclude.has(String(b.block_id)))points.push(finite(b.start,0),finite(b.start,0)+finite(b.duration,0));for(const tr of doc.extra_tracks||[])for(const c of tr.clips||[])if(!exclude.has(String(c.clip_id)))points.push(finite(c.start,0),finite(c.start,0)+finite(c.duration,0));return points.filter(Number.isFinite);}
function addCutMarkAtTime(node,doc,time){const t=Math.max(0,finite(time,node.__lmdPlayhead||0)),hit=sceneInfoAt(doc,t);if(!hit||t<hit.start+.25||t>hit.end-.25){alert("Place the playhead at least 0.25s inside a MAIN clip to add a CUT marker.");return false;}const local=Math.round((t-hit.start)*1e6)/1e6,fps=Math.max(1,finite(doc.fps,24)),tolerance=Math.max(1/fps,.04);hit.shot.cut_marks=Array.isArray(hit.shot.cut_marks)?hit.shot.cut_marks:[];if(hit.shot.cut_marks.some(mark=>Math.abs(finite(mark,0)-local)<=tolerance)){alert("A CUT marker already exists this close to the playhead.");return false;}if(hit.shot.cut_marks.length>=128){alert("This clip already has the maximum of 128 CUT markers.");return false;}hit.shot.cut_marks.push(local);hit.shot.cut_marks.sort((a,b)=>a-b);node.__lmdPlayhead=t;node.__lmdSelection={track:"main",id:hit.shot.clip_id};commit(node,doc);return true;}
function removeCutMarkAtTime(node,doc,shot,time){const index=doc.shots.indexOf(shot);if(index<0)return false;const start=shotStart(doc,index),local=finite(time,start)-start,marks=shot.cut_marks||[];let best=-1,distance=Infinity;for(let i=0;i<marks.length;i++){const d=Math.abs(finite(marks[i],0)-local);if(d<distance){best=i;distance=d;}}if(best<0||distance>.35)return false;marks.splice(best,1);node.__lmdPlayhead=start+local;commit(node,doc);return true;}
function syncCutMarkDisplay(block,doc,shot,index,pps){
  const start=shotStart(doc,index),duration=Math.max(.25,finite(shot.duration,.25));
  for(const marker of block.querySelectorAll("[data-lmd-cut-mark-time]")){
    const absolute=finite(marker.dataset.lmdCutMarkTime,NaN),local=absolute-start;
    marker.style.left=`${local*pps}px`;
    marker.style.display=Number.isFinite(local)&&local>=0&&local<=duration?"":"none";
  }
}
function installCutMarks(block,row,node,doc,shot,index,pps){
  const duration=Math.max(.25,finite(shot.duration,.25)),start=shotStart(doc,index),scale=Math.max(.01,pps*pointerScale(row));
  for(const markValue of shot.cut_marks||[]){const mark=finite(markValue,0);if(mark<0||mark>duration)continue;const marker=el("div",null,{position:"absolute",left:`${mark*pps}px`,top:"0",bottom:"0",width:"12px",transform:"translateX(-50%)",zIndex:"28",cursor:"ew-resize",pointerEvents:"auto"});marker.dataset.lmdCutMark="1";marker.dataset.lmdCutMarkTime=String(start+mark);marker.title=`CUT marker · ${(start+mark).toFixed(3)}s. Drag inside this clip; clip edges snap to it. This is not a Knife split.`;marker.append(el("div",null,{position:"absolute",left:"5px",top:"0",bottom:"0",width:"2px",background:"#f3bf68",boxShadow:"0 0 5px rgba(243,191,104,.7)",pointerEvents:"none"}),el("div","CUT",{position:"absolute",left:"50%",top:"2px",transform:"translateX(-50%)",padding:"1px 3px",borderRadius:"3px",background:"rgba(71,49,16,.95)",color:"#ffe2a5",font:"800 7px/11px system-ui",letterSpacing:".04em",pointerEvents:"none"}));marker.onclick=e=>{e.preventDefault();e.stopPropagation();node.__lmdPlayhead=start+mark;setSelection(node,"main",shot.clip_id);};marker.oncontextmenu=e=>{const end=start+duration;showTimelineContextMenu(node,doc,e,{hit:{kind:"main",item:shot,index,start,end,list:doc.shots,idKey:"clip_id"},trackKey:"main",time:start+mark});};marker.onpointerdown=e=>{if(e.button!==0)return;e.preventDefault();e.stopPropagation();node.__lmdSelection={track:"main",id:shot.clip_id};const x0=e.clientX,rect=row.getBoundingClientRect();let next=mark,moved=false;const move=ev=>{if(Math.abs(ev.clientX-x0)>2)moved=true;const raw=Math.max(start+.25,Math.min(start+duration-.25,(ev.clientX-rect.left)/scale)),fps=Math.max(1,finite(doc.fps,24)),frame=1/fps,absolute=Math.max(start+.25,Math.min(start+duration-.25,ev.altKey?raw:Math.round(raw/frame)*frame));next=absolute-start;marker.style.left=`${next*pps}px`;marker.title=`CUT marker · ${absolute.toFixed(3)}s. Release to save; clip edges snap to it.`;};const up=()=>{window.removeEventListener("pointermove",move,true);window.removeEventListener("pointerup",up,true);window.removeEventListener("pointercancel",up,true);if(moved){shot.cut_marks=(shot.cut_marks||[]).filter(value=>Math.abs(finite(value,0)-mark)>.0005);shot.cut_marks.push(Math.round(next*1e6)/1e6);shot.cut_marks.sort((a,b)=>a-b);node.__lmdPlayhead=start+next;commit(node,doc);}else{marker.style.left=`${mark*pps}px`;setSelection(node,"main",shot.clip_id);}};window.addEventListener("pointermove",move,true);window.addEventListener("pointerup",up,true);window.addEventListener("pointercancel",up,true);};block.append(marker);}
}
function snapTime(node,doc,time,pps,{excludeIds=[],includePlayhead=false,bypass=false}={}){let raw=Math.max(0,finite(time,0));if(doc?.snapping===false||bypass)return raw;const fps=Math.max(1,finite(doc?.fps,24)),frame=1/fps;raw=Math.round(raw/frame)*frame;const points=timelineSnapPoints(doc,excludeIds);if(includePlayhead&&Number.isFinite(node?.__lmdPlayhead))points.push(node.__lmdPlayhead);const tol=Math.max(frame*.55,6/Math.max(28,finite(pps,54)));let best=raw,dist=Infinity;for(const p of points){const d=Math.abs(p-raw);if(d<dist){dist=d;best=p;}}return Math.round((dist<=tol?best:raw)*1e6)/1e6;}
function snapClipStart(node,doc,start,duration,pps,excludeIds=[]){if(doc?.snapping===false)return Math.max(0,start);const a=snapTime(node,doc,start,pps,{excludeIds,includePlayhead:true});const endCandidate=snapTime(node,doc,start+duration,pps,{excludeIds,includePlayhead:true})-duration;return Math.max(0,Math.abs(a-start)<=Math.abs(endCandidate-start)?a:endCandidate);}
function trimHandle(side,enabled=true){const h=el("div",null,{position:"absolute",top:"0",bottom:"0",width:"12px",[side]:"-1px",zIndex:"25",cursor:enabled?"ew-resize":"not-allowed",opacity:enabled?"1":".35",pointerEvents:"auto"});const bar=el("div",null,{position:"absolute",top:"7px",bottom:"7px",[side]:"3px",width:"2px",background:"rgba(245,248,255,.92)",boxShadow:"0 0 3px rgba(0,0,0,.9)"});const hook1=el("div",null,{position:"absolute",top:"7px",[side]:"3px",width:"6px",height:"2px",background:"rgba(245,248,255,.92)"});const hook2=el("div",null,{position:"absolute",bottom:"7px",[side]:"3px",width:"6px",height:"2px",background:"rgba(245,248,255,.92)"});h.dataset.lmdTrim="1";h.append(bar,hook1,hook2);return h;}
function markTrimmedEdge(handle,side,changed,title){handle.title=title;if(!changed)return;const arrow=el("span",side==="left"?"↦":"↤",{position:"absolute",top:"50%",[side]:"-2px",transform:"translateY(-50%)",padding:"0 2px",borderRadius:"3px",background:"rgba(13,17,23,.9)",color:"#fff",font:"900 12px/16px system-ui",textShadow:"0 1px 3px #000",pointerEvents:"none"});handle.append(arrow);}
function pointerScale(row){const rect=row.getBoundingClientRect();const css=Math.max(1,row.offsetWidth||rect.width||1);return Math.max(.01,rect.width/css);}
function beginTrim(event,node,doc,row,pps,apply){event.preventDefault();event.stopPropagation();const x0=event.clientX,screenPerSecond=Math.max(.01,pps*pointerScale(row));let raf=0,last=0;const move=e=>{last=(e.clientX-x0)/screenPerSecond;if(raf)return;raf=requestAnimationFrame(()=>{raf=0;apply(last,false);updateTimelineGeometry(node,doc);});};const up=e=>{window.removeEventListener("pointermove",move,true);window.removeEventListener("pointerup",up,true);window.removeEventListener("pointercancel",up,true);if(raf){cancelAnimationFrame(raf);raf=0;}last=(e.clientX-x0)/screenPerSecond;apply(last,true);commit(node,doc);};window.addEventListener("pointermove",move,true);window.addEventListener("pointerup",up,true);window.addEventListener("pointercancel",up,true);}
function maxBlockDuration(doc,track,block){if(track==="main"){const m=timedMediaForShot(doc,block);return m?Math.max(.25,m.seconds-finite(block.source_in,0)):150;}if(track==="audio"){const m=timedMediaForAudio(doc,block);return m?Math.max(.25,m.seconds-finite(block.source_in,0)):600;}return 600;}
function installMainTrim(block,row,node,doc,shot,index,pps){
  const left=trimHandle("left",index>0),right=trimHandle("right",true);block.append(left,right);const start0=shotStart(doc,index),d0=finite(shot.duration,0),media=timedMediaForShot(doc,shot),in0=finite(shot.source_in,0),id0=shot.clip_id,cutMarks0=(shot.cut_marks||[]).map(mark=>finite(mark,0));if(media){markTrimmedEdge(left,"left",in0>.001,`Source cut-in ${in0.toFixed(2)}s · original media ${finite(media.seconds,0).toFixed(2)}s`);markTrimmedEdge(right,"right",in0+d0<finite(media.seconds,0)-.01,`Source cut-out ${(in0+d0).toFixed(2)}s · original media ${finite(media.seconds,0).toFixed(2)}s`);}
  if(index>0)left.onpointerdown=e=>{const prev=doc.shots[index-1],prevEnd=shotStart(doc,index-1)+finite(prev.duration,0);beginTrim(e,node,doc,row,pps,(raw)=>{let target=snapTime(node,doc,start0+raw,pps,{excludeIds:[id0],includePlayhead:true}),delta=target-start0;const minDelta=Math.max(prevEnd-start0,media?-in0:-Infinity),maxDelta=d0-.25;delta=Math.max(minDelta,Math.min(maxDelta,delta));shot.start=start0+delta;shot.duration=d0-delta;shot.cut_marks=cutMarks0.map(mark=>Math.round((mark-delta)*1e6)/1e6);syncCutMarkDisplay(block,doc,shot,index,pps);if(media)shot.source_in=in0+delta;});};
  else left.title="The first MAIN clip begins at 0s; trim its right edge to create a gap after it.";
  right.onpointerdown=e=>{const mediaMax=maxBlockDuration(doc,"main",shot),nextStart=index+1<doc.shots.length?shotStart(doc,index+1):Infinity,max=Math.max(.25,Math.min(mediaMax,nextStart-start0)),end0=start0+d0;beginTrim(e,node,doc,row,pps,(raw)=>{const snapped=snapTime(node,doc,end0+raw,pps,{excludeIds:[id0],includePlayhead:true});shot.duration=Math.max(.25,Math.min(max,snapped-start0));shot.cut_marks=cutMarks0.slice();syncCutMarkDisplay(block,doc,shot,index,pps);});};
}
function installMainExtendZone(row,node,doc,shot,index,pps,height){
  if(String(shot.base_kind||"generated")==="media"||coreTrackLocked(node,"main"))return null;
  const start=shotStart(doc,index),end=start+finite(shot.duration,0),nextStart=index+1<doc.shots.length?shotStart(doc,index+1):Infinity;
  // Extend is an insert into empty timeline space. A touching/overlapping next clip
  // owns the area after this shot; never draw an affordance over that clip.
  const room=Number.isFinite(nextStart)?nextStart-end:Math.max(0,150-end);
  if(room<.25)return null;
  const zoneSeconds=Math.min(4,room);
  const zone=el("div",null,{position:"absolute",left:`${end*pps}px`,top:"4px",width:`${zoneSeconds*pps}px`,height:`${Math.max(30,height-10)}px`,boxSizing:"border-box",zIndex:"20",border:"1px dashed #b790ff",borderLeft:"2px solid #b790ff",borderRadius:"0 5px 5px 0",background:"repeating-linear-gradient(135deg,rgba(154,112,224,.22),rgba(154,112,224,.22) 8px,rgba(154,112,224,.08) 8px,rgba(154,112,224,.08) 16px)",color:"#e0d1ff",cursor:"ew-resize",overflow:"hidden",userSelect:"none"});
  zone.append(el("div","DRAG TO EXTEND · RE-RENDER",{position:"sticky",left:"4px",top:"5px",width:"max-content",maxWidth:`${zoneSeconds*pps-8}px`,padding:"5px 6px",font:"800 8px system-ui",letterSpacing:".04em",textShadow:"0 1px 2px #000",pointerEvents:"none"}));
  zone.title="Drag right to increase this BASE clip duration. The next render regenerates this clip and rebuilds dependent clips.";
  zone.onpointerdown=e=>{
    if(e.button!==0)return;e.preventDefault();e.stopPropagation();setSelection(node,"main",shot.clip_id);
    const x0=e.clientX,d0=finite(shot.duration,0),scale=pointerScale(row),maxExtra=zoneSeconds,preview=el("div",null,{position:"absolute",left:`${end*pps}px`,top:"2px",width:"0px",height:`${Math.max(32,height-6)}px`,zIndex:"45",boxSizing:"border-box",border:"1px solid #d2b8ff",background:"rgba(146,101,220,.32)",pointerEvents:"none"});row.append(preview);
    let moved=false;
    const move=ev=>{const delta=(ev.clientX-x0)/(Math.max(1,pps)*scale);if(delta>.015)moved=true;const next=Math.max(d0,Math.min(d0+maxExtra,d0+delta)),shift=next-d0;shot.duration=next;preview.style.width=`${Math.max(0,shift)*pps}px`;zone.style.left=`${(start+next)*pps}px`;updateTimelineGeometry(node,doc);};
    const finish=()=>{window.removeEventListener("pointermove",move,true);window.removeEventListener("pointerup",finish,true);window.removeEventListener("pointercancel",finish,true);preview.remove();if(moved){node.__lmdPlayhead=start+shot.duration;commit(node,doc);}else{shot.duration=d0;updateTimelineGeometry(node,doc);setSelection(node,"main",shot.clip_id);render(node);}};
    window.addEventListener("pointermove",move,true);window.addEventListener("pointerup",finish,true);window.addEventListener("pointercancel",finish,true);
  };
  return zone;
}
function installMainReorder(block,row,node,doc,shot,index,pps){
  block.addEventListener("pointerdown",e=>{
    if(e.button!==0||e.target?.closest?.('[data-lmd-trim="1"]'))return;
    const x0=e.clientX,start0=shotStart(doc,index),screenPerSecond=Math.max(.01,pps*pointerScale(row)),suffix=doc.shots.slice(index),suffixIds=suffix.map(s=>s.clip_id),suffixStarts=suffix.map((s,j)=>shotStart(doc,index+j));let moved=false,lastX=x0,lastAlt=false;
    const move=ev=>{lastX=ev.clientX;lastAlt=Boolean(ev.altKey);const dx=ev.clientX-x0;if(Math.abs(dx)>4)moved=true;if(!moved)return;block.style.transform=`translateX(${dx}px)`;block.style.opacity=".78";block.style.zIndex="40";};
    const up=()=>{window.removeEventListener("pointermove",move,true);window.removeEventListener("pointerup",up,true);window.removeEventListener("pointercancel",up,true);block.style.transform="";block.style.opacity="";block.style.zIndex="";if(!moved)return;const raw=Math.max(0,start0+(lastX-x0)/screenPerSecond),prevEnd=index>0?shotStart(doc,index-1)+finite(doc.shots[index-1].duration,0):0,target=index===0?0:Math.max(prevEnd,lastAlt?raw:snapTime(node,doc,raw,pps,{excludeIds:suffixIds,includePlayhead:true})),delta=target-start0;if(Math.abs(delta)<1e-6)return;for(let j=0;j<suffix.length;j++)suffix[j].start=Math.max(0,suffixStarts[j]+delta);node.__lmdSelection={track:"main",id:shot.clip_id};node.__lmdSuppressMainClickUntil=performance.now()+220;commit(node,doc);};
    window.addEventListener("pointermove",move,true);window.addEventListener("pointerup",up,true);window.addEventListener("pointercancel",up,true);
  });
}
function installMulticlipBoundary(marker,row,node,doc,index,pps){
  const leftShot=doc.shots[index-1],rightShot=doc.shots[index];if(!leftShot||!rightShot)return;
  marker.onpointerdown=e=>{e.preventDefault();e.stopPropagation();setMulticlipBoundarySelection(node,doc,index);const boundary0=shotStart(doc,index),l0=leftShot.duration,r0=rightShot.duration,in0=finite(rightShot.source_in,0),rightMedia=timedMediaForShot(doc,rightShot),leftMax=maxBlockDuration(doc,"main",leftShot);beginTrim(e,node,doc,row,pps,(raw)=>{let delta=snapTime(node,doc,boundary0+raw,pps,{excludeIds:doc.shots.map(x=>x.clip_id),includePlayhead:true})-boundary0;const minDelta=Math.max(-(l0-.25),rightMedia?-in0:-Infinity),maxDelta=Math.min(r0-.25,leftMax-l0);delta=Math.max(minDelta,Math.min(maxDelta,delta));leftShot.duration=l0+delta;rightShot.duration=r0-delta;if(rightMedia)rightShot.source_in=in0+delta;});};
}
function addMulticlipAtPlayhead(node,doc){
  const t=finite(node.__lmdPlayhead,0),hit=sceneInfoAt(doc,t);if(!hit||t<=hit.start+.249||t>=hit.end-.249){alert("Place the playhead at least 0.25s inside a clip to add a cut.");return;}const split=splitMainAtTime(node,doc,t,{selectRight:true,splitBoundTracks:true});if(!split)return;commit(node,doc);const rightIndex=doc.shots.findIndex(s=>s.clip_id===split.right.clip_id);if(rightIndex>0){setMulticlipBoundarySelection(node,doc,rightIndex);render(node);}
}
function deleteMulticlipBoundary(node,doc){
  if(doc.shots.length<2)return;let idx=selectedMulticlipBoundaryIndex(node,doc);if(idx<1){const t=finite(node.__lmdPlayhead,0);let best=1,dist=Infinity;for(let i=1;i<doc.shots.length;i++){const d=Math.abs(shotStart(doc,i)-t);if(d<dist){dist=d;best=i;}}idx=best;}const left=doc.shots[idx-1],right=doc.shots[idx];if(!left||!right)return;const leftStart=shotStart(doc,idx-1),rightStart=shotStart(doc,idx);left.duration=Math.max(.25,(rightStart+finite(right.duration,0))-leftStart);if(!String(left.prompt||"").trim())left.prompt=String(right.prompt||"");else if(String(right.prompt||"").trim()&&String(right.prompt||"").trim()!==String(left.prompt||"").trim())left.prompt=`${String(left.prompt).trim()}\n\n${String(right.prompt).trim()}`;left.subjects=[...new Set([...(left.subjects||[]),...(right.subjects||[])])];if(!left.media_subject_id)left.media_subject_id=right.media_subject_id||null;if(left.media_subject_id&&right.media_subject_id&&left.media_subject_id!==right.media_subject_id&&!left.subjects.includes(right.media_subject_id))left.subjects.push(right.media_subject_id);left.last_frame_anchor=Boolean(right.last_frame_anchor||left.last_frame_anchor);if(!left.cast_subject_id)left.cast_subject_id=right.cast_subject_id||null;for(const list of [doc.camera_blocks,doc.audio_blocks])for(const b of list){if(b.scene_id===right.clip_id){if(doc.ripple_tracks!==false){b.scene_id=left.clip_id;b.scene_offset=finite(b.start,0)-leftStart;}else b.scene_id=null;}}doc.shots.splice(idx,1);node.__lmdSelection={track:"main",id:left.clip_id};node.__lmdMulticlipBoundary=null;doc.regeneration={mode:"none",clip_id:null,request_id:null};commit(node,doc);
}
function installFreeTrim(block,row,node,doc,item,track,pps){const left=trimHandle("left",true),right=trimHandle("right",true);block.append(left,right);const s0=item.start,d0=item.duration,in0=finite(item.source_in,0),media=track==="audio"?timedMediaForAudio(doc,item):track==="extra"?timedMediaForExtra(doc,item):null,id0=item.block_id||item.clip_id;if(media){markTrimmedEdge(left,"left",in0>.001,`Source cut-in ${in0.toFixed(2)}s · original media ${finite(media.seconds,0).toFixed(2)}s`);markTrimmedEdge(right,"right",in0+d0<finite(media.seconds,0)-.01,`Source cut-out ${(in0+d0).toFixed(2)}s · original media ${finite(media.seconds,0).toFixed(2)}s`);}left.onpointerdown=e=>beginTrim(e,node,doc,row,pps,(raw)=>{let target=snapTime(node,doc,s0+raw,pps,{excludeIds:[id0],includePlayhead:true}),delta=target-s0;const minDelta=Math.max(-s0,media?-in0:-Infinity),maxDelta=d0-.25;delta=Math.max(minDelta,Math.min(maxDelta,delta));item.start=s0+delta;item.duration=d0-delta;if(item.scene_id){const idx=doc.shots.findIndex(s=>s.clip_id===item.scene_id);if(idx>=0)item.scene_offset=item.start-shotStart(doc,idx);}if(media)item.source_in=in0+delta;});right.onpointerdown=e=>{const max=maxBlockDuration(doc,track,item),end0=s0+d0;beginTrim(e,node,doc,row,pps,(raw)=>{const target=snapTime(node,doc,end0+raw,pps,{excludeIds:[id0],includePlayhead:true});item.duration=Math.max(.25,Math.min(max,target-s0));});};}
function installFreeMove(block,row,node,doc,item,track,pps,selectionTrack=track){block.addEventListener("pointerdown",e=>{if(e.button!==0||e.target?.closest?.('[data-lmd-trim="1"]'))return;e.stopPropagation();node.__lmdSelection={track:selectionTrack,id:item.block_id||item.clip_id};const x0=e.clientX,s0=finite(item.start,0),screenPerSecond=Math.max(.01,pps*pointerScale(row)),itemId=item.block_id||item.clip_id;let moved=false,last=s0,raf=0;const move=ev=>{const dx=ev.clientX-x0;if(Math.abs(dx)>3)moved=true;if(!moved)return;const rawStart=Math.max(0,s0+dx/screenPerSecond);last=ev.altKey?rawStart:snapClipStart(node,doc,rawStart,finite(item.duration,0),pps,[itemId]);if(raf)return;raf=requestAnimationFrame(()=>{raf=0;if(track==="extra")item.start=last;else setBlockStart(doc,item,last);updateTimelineGeometry(node,doc);});};const up=()=>{window.removeEventListener("pointermove",move,true);window.removeEventListener("pointerup",up,true);window.removeEventListener("pointercancel",up,true);if(raf){cancelAnimationFrame(raf);raf=0;}if(moved){if(track==="extra")item.start=last;else setBlockStart(doc,item,last);commit(node,doc);}};window.addEventListener("pointermove",move,true);window.addEventListener("pointerup",up,true);window.addEventListener("pointercancel",up,true);});}
function selection(node){return node.__lmdSelection||{track:"main",id:null};}
function setSelection(node,track,id){node.__lmdSelection={track,id};node.__lmdMainMultiSelection=null;node.__lmdMainSelectionAnchor=track==="main"?id:null;render(node);}
function selectMainTimelineClip(node,doc,clipId,event){
  const index=doc.shots.findIndex(shot=>String(shot.clip_id)===String(clipId));if(index<0)return;
  const multi=Boolean(event?.shiftKey||event?.ctrlKey||event?.metaKey),current=Array.isArray(node.__lmdMainMultiSelection)?node.__lmdMainMultiSelection:[];
  if(event?.shiftKey){const anchorId=String(node.__lmdMainSelectionAnchor||selection(node).id||clipId),anchor=doc.shots.findIndex(shot=>String(shot.clip_id)===anchorId),first=Math.min(index,anchor<0?index:anchor),last=Math.max(index,anchor<0?index:anchor);node.__lmdMainMultiSelection=doc.shots.slice(first,last+1).map(shot=>shot.clip_id);}
  else if(event?.ctrlKey||event?.metaKey){const next=new Set(current);if(next.has(clipId))next.delete(clipId);else next.add(clipId);if(!next.size)next.add(clipId);node.__lmdMainMultiSelection=[...next];node.__lmdMainSelectionAnchor=clipId;}
  else{node.__lmdMainMultiSelection=null;node.__lmdMainSelectionAnchor=clipId;}
  node.__lmdSelection={track:"main",id:clipId};render(node);
}
function selectedShot(node,doc){const s=selection(node);return s.track==="main"?doc.shots.find(x=>x.clip_id===s.id):null;}
function directorMainTerm(doc){return String(doc?.setup_timeline_mode||"auto")==="multiclip"?"clip":"shot";}
function directorMainTermUpper(doc){return directorMainTerm(doc).toUpperCase();}
function displayMainName(doc,shot,index){const raw=String(shot?.name||"").trim();if(directorMainTerm(doc)==="clip"&&/^Shot\b/i.test(raw))return raw.replace(/^Shot/i,"Clip");return raw||`${directorMainTermUpper(doc)} ${index+1}`;}
function syncSegmentedTimeline(doc){
  if(String(doc?.setup_timeline_mode||"")!=="segmented"||!Array.isArray(doc?.shots)||doc.shots.length!==1)return;
  doc.shots[0].start=0;
  let count=Math.max(1,Math.min(64,Math.trunc(finite(doc.segmented_count,3))));
  let duration=clamp(doc.segmented_duration,.25,150,5);
  if(count*duration>150)duration=Math.max(.25,150/count);
  doc.segmented_count=count;doc.segmented_duration=Math.round(duration*1000)/1000;doc.shots[0].duration=Math.round(count*duration*1000)/1000;
}
function selectedMulticlipBoundaryIndex(node,doc){const sel=node.__lmdMulticlipBoundary;if(!sel)return -1;for(let i=1;i<(doc.shots||[]).length;i++)if(doc.shots[i-1].clip_id===sel.left&&doc.shots[i].clip_id===sel.right)return i;return -1;}
function setMulticlipBoundarySelection(node,doc,index){if(index>0&&index<(doc.shots||[]).length)node.__lmdMulticlipBoundary={left:doc.shots[index-1].clip_id,right:doc.shots[index].clip_id};else node.__lmdMulticlipBoundary=null;}
function setMainClipRunEnabled(node,doc,shot,enabled){if(!shot||String(shot.base_kind||"generated")==="media")return;shot.run_enabled=Boolean(enabled);commit(node,doc);}
function mergedGroupRunEnabled(doc,group){const members=(group?.clip_ids||[]).map(clipId=>doc.shots.find(shot=>String(shot.clip_id)===String(clipId))).filter(Boolean);return members.length>0&&members.every(shot=>shot.run_enabled!==false);}
function setMergedGroupRunEnabled(node,doc,group,enabled){for(const clipId of group?.clip_ids||[]){const shot=doc.shots.find(item=>String(item.clip_id)===String(clipId));if(shot&&String(shot.base_kind||"generated")!=="media")shot.run_enabled=Boolean(enabled);}commit(node,doc);}
function selectedMainClipIds(node,doc,clickedClipId){const ids=Array.isArray(node.__lmdMainMultiSelection)?node.__lmdMainMultiSelection:[];return ids.includes(clickedClipId)?ids:[clickedClipId];}
function mergeSelectionCheck(doc,clipIds){
  if(String(doc.setup_timeline_mode)!=="multiclip")return {ok:false,reason:"Merge is available in MultiClip timeline mode."};
  const unique=[...new Set((clipIds||[]).map(String))],indices=unique.map(clipId=>doc.shots.findIndex(shot=>String(shot.clip_id)===clipId)).sort((a,b)=>a-b);
  if(indices.length<2||indices.some((index,offset)=>index<0||(offset>0&&index!==indices[offset-1]+1)))return {ok:false,reason:"Select at least two neighboring MAIN clips."};
  const shots=indices.map(index=>doc.shots[index]);
  for(let offset=1;offset<indices.length;offset++){const previousEnd=shotStart(doc,indices[offset-1])+finite(shots[offset-1]?.duration,0),nextStart=shotStart(doc,indices[offset]);if(Math.abs(nextStart-previousEnd)>.02)return {ok:false,reason:"Close gaps between the selected clips before merging them."};}
  if(shots.some(shot=>String(shot.base_kind||"generated")==="media"))return {ok:false,reason:"External MEDIA clips cannot be merged into a generated TAKE group."};
  if(shots.some(shot=>mergedGroupForClip(doc,shot.clip_id)))return {ok:false,reason:"Restore the existing merged clip boundaries before merging this range."};
  return {ok:true,indices,shots};
}
async function mergeSelectedMainClips(node,doc,clipIds){
  const check=mergeSelectionCheck(doc,clipIds);if(!check.ok){alert(check.reason);return;}
  await refreshTakeStatus(node,doc,{renderAfter:false});
  const withoutTake=check.shots.filter(shot=>!node.__lmdTakeStatus?.clips?.[shot.clip_id]?.active);
  if(withoutTake.length){alert(`Each selected clip needs an active rendered TAKE before merging. Missing: ${withoutTake.map(shot=>shot.name||shot.clip_id).join(", ")}.`);return;}
  const first=check.indices[0],last=check.indices.at(-1),group={group_id:id("merge"),name:`Merged Clip ${first+1}–${last+1}`,clip_ids:check.shots.map(shot=>shot.clip_id)};
  doc.merged_groups.push(group);node.__lmdMainMultiSelection=null;node.__lmdSelection={track:"main_group",id:group.group_id};node.__lmdPlayhead=shotStart(doc,first);doc.regeneration={mode:"none",clip_id:null,request_id:null};commit(node,doc);
}
function restoreMergedGroupBoundaries(node,doc,group){doc.merged_groups=doc.merged_groups.filter(item=>item.group_id!==group.group_id);node.__lmdMainMultiSelection=[...(group.clip_ids||[])];node.__lmdSelection={track:"main",id:group.clip_ids?.[0]||doc.shots[0]?.clip_id};commit(node,doc);}

async function refreshEmbeddingCatalog(node,{rerender=true}={}){
  try{const r=await api.fetchApi('/longmedia/embeddings');if(!r.ok)throw new Error(`HTTP ${r.status}`);const data=await r.json();node.__lmdEmbeddings=Array.isArray(data?.embeddings)?data.embeddings.map(String).filter(Boolean):[];node.__lmdEmbeddingCatalogError="";}
  catch(err){node.__lmdEmbeddings=[];node.__lmdEmbeddingCatalogError=String(err?.message||err||"Embedding catalog unavailable");}
  if(rerender)render(node);
}
function embeddingCatalog(node,current=""){const values=[...(node.__lmdEmbeddings||[])];const cur=String(current||"").trim();if(cur&&!values.includes(cur))values.unshift(cur);return values;}
async function refreshTakeStatus(node,doc,{renderAfter=true}={}){if(!doc?.project_id)return false;try{const ids=doc.shots.map(s=>s.clip_id).join(",");const r=await api.fetchApi(`/longmedia/director/status?project_id=${encodeURIComponent(doc.project_id)}&clip_ids=${encodeURIComponent(ids)}`);if(!r.ok)return false;node.__lmdTakeStatus=await r.json();if(renderAfter)render(node);return true;}catch(err){console.debug("Director take status unavailable",err);return false;}}
function connectedSetup(node){for(const linkId of (node.outputs?.[0]?.links||[])){const link=node.graph?.links?.[linkId];const target=link?node.graph?.getNodeById?.(link.target_id):null;if(target&&String(cls(target))==="MiniMaxH3LatentLabLongMediaSetup")return target;}return null;}
async function regenerationPlan(node,doc,shot,mode){await refreshTakeStatus(node,doc);const index=doc.shots.indexOf(shot),all=doc.shots.map(s=>s.clip_id);let indices=[];let reason="";const setup=connectedSetup(node);const loop=Boolean(widget(setup,"loop_closure_enabled")?.value);if(loop){indices=all.map((_,i)=>i);reason="Loop Closure couples the end of the sequence back to the beginning, so the complete chain must be rebuilt.";}else if(mode==="from_here"){indices=all.map((_,i)=>i).filter(i=>i>=index);reason="Regenerate From Here intentionally rebuilds the selected clip and every dependent clip after it.";}else{const status=node.__lmdTakeStatus?.clips||{};const active=status[shot.clip_id]?.active;const prev=index>0?status[all[index-1]]?.active:null;const next=index+1<all.length?status[all[index+1]]?.active:null;const savedDuration=finite(active?.clip_metadata?.duration,NaN);const durationChanged=Number.isFinite(savedDuration)&&Math.abs(savedDuration-shot.duration)>.001;if(!active){indices=all.map((_,i)=>i).filter(i=>i>=index);reason="The selected clip has no approved cached take to provide the outgoing boundary lock.";}else if(index>0&&!prev){indices=all.map((_,i)=>i).filter(i=>i>=index);reason="The previous clip has no cached continuation checkpoint, so the incoming seam cannot be locked.";}else if(index+1<all.length&&!next){indices=all.map((_,i)=>i).filter(i=>i>=index);reason="The next clip has no approved cached take, so a safe clip-only replacement cannot preserve the existing suffix.";}else if(durationChanged){indices=all.map((_,i)=>i).filter(i=>i>=index);reason="The selected clip duration changed since its cached take; temporal geometry moved, so the dependent suffix must be rebuilt.";}else{indices=[index];reason="Compatible cached neighbors are available; two-sided AV boundary locking can keep both seams fixed.";}}
  return {mode,clip_id:shot.clip_id,indices,reason};}
async function queueRegeneration(node,doc,shot,mode,{reroll=false,extra={}}={}){
  if(!shot)return;
  const existing=node.__lmdLastRegen;
  if(existing?.state==="queueing")return;
  if(reroll)shot.seed=randomSeed();
  const request_id=id("regen");
  node.__lmdRegenPreview=null;
  doc.regeneration={mode,clip_id:shot.clip_id,request_id,...extra};
  node.__lmdLastRegen={state:"queueing",request_id,clip_id:shot.clip_id,mode,seed:shot.seed??null};
  commit(node,doc);
  try{
    // Yield one microtask after committing director_json so ComfyUI graphToPrompt sees
    // the exact one-shot regeneration request in the workflow snapshot it queues.
    await Promise.resolve();
    const queued=await app.queuePrompt(0);
    if(queued===false){
      node.__lmdLastRegen={...node.__lmdLastRegen,state:"ready",message:"Queue rejected; press Queue manually."};
      render(node);
      return;
    }
    node.__lmdLastRegen={...node.__lmdLastRegen,state:"queued"};
    render(node);
  }catch(err){
    console.error("LongMedia Director selective regeneration queue failed",err);
    node.__lmdLastRegen={...node.__lmdLastRegen,state:"ready",message:`Queue failed: ${err?.message||err}. Press Queue manually.`};
    render(node);
  }
}

async function addMediaSubject(node,doc,kind,record,name="") {
  const capital=kind==="image"?"Picture":kind[0].toUpperCase()+kind.slice(1); const slot=nextSlot(doc,capital); if(slot==null){alert(`No free ${capital} slots remain.`);return null;}
  const subject={subject_id:id("subject"),name:name||record.filename,kind:capital,slot,description:"",retention:capital==="Audio"?"reference":"fully_preserved",media:record};doc.subjects.push(subject);commit(node,doc,false);return subject;
}


function selectedTimelineItem(node,doc){const sel=selection(node);if(sel.track==="main"){const index=doc.shots.findIndex(x=>x.clip_id===sel.id);if(index<0)return null;const item=doc.shots[index],start=shotStart(doc,index);return {kind:"main",item,index,start,end:start+item.duration,list:doc.shots,idKey:"clip_id"};}if(sel.track==="camera"){const index=doc.camera_blocks.findIndex(x=>x.block_id===sel.id);if(index<0)return null;const item=doc.camera_blocks[index];return {kind:"camera",item,index,start:item.start,end:item.start+item.duration,list:doc.camera_blocks,idKey:"block_id"};}if(sel.track==="audio"){const index=doc.audio_blocks.findIndex(x=>x.block_id===sel.id);if(index<0)return null;const item=doc.audio_blocks[index];return {kind:"audio",item,index,start:item.start,end:item.start+item.duration,list:doc.audio_blocks,idKey:"block_id"};}if(String(sel.track||"").startsWith("extra:")){const track=doc.extra_tracks.find(t=>t.track_id===String(sel.track).slice(6)),index=track?.clips?.findIndex(x=>x.clip_id===sel.id)??-1;if(!track||index<0)return null;const item=track.clips[index];return {kind:"extra",track,item,index,start:item.start,end:item.start+item.duration,list:track.clips,idKey:"clip_id"};}return null;}

function timelineSelectionHit(node,doc){return selectedTimelineItem(node,doc);}
function timelineItemName(hit){if(!hit)return "clip";if(hit.kind==="main")return String(hit.item?.name||"MAIN clip");if(hit.kind==="camera")return String(hit.item?.camera?.movement||"Camera block");if(hit.kind==="audio")return String(hit.item?.prompt||"Audio block");return String(hit.item?.name||"Layer clip");}
function renameTimelineHit(node,doc,hit){
  if(!hit)return;
  if(hit.kind==="camera"){node.__lmdSelection={track:"camera",id:hit.item.block_id};render(node);return;}
  const current=timelineItemName(hit),name=window.prompt("Rename clip",current);if(name==null)return;const clean=String(name||"").trim().slice(0,120);if(!clean)return;
  if(hit.kind==="audio"&&hit.item.kind!=="reference")hit.item.prompt=clean;else hit.item.name=clean;
  commit(node,doc);
}
function duplicateTimelineHit(node,doc,hit){
  if(!hit)return;
  if(hit.kind==="main"){duplicateMainShot(node,doc,hit.item);return;}
  const copy=JSON.parse(JSON.stringify(hit.item));
  if(hit.kind==="camera")copy.block_id=id("camera");else if(hit.kind==="audio")copy.block_id=id("audio");else copy.clip_id=id("layer");
  const nextStart=Math.max(0,finite(hit.end,finite(hit.start,0)+finite(hit.item.duration,0)));
  copy.start=snapClipStart(node,doc,nextStart,finite(copy.duration,0),clamp(node.__lmdPps,6,120,54),[hit.item[hit.idKey]]);
  if(hit.kind==="camera"||hit.kind==="audio")setBlockStart(doc,copy,copy.start);
  hit.list.splice(hit.index+1,0,copy);
  node.__lmdSelection={track:hit.kind==="extra"?`extra:${hit.track.track_id}`:hit.kind,id:copy[hit.idKey]};
  commit(node,doc);
}
function deleteTimelineHit(node,doc,hit){
  if(!hit)return;
  if(hit.kind==="main"){deleteMainShot(node,doc,hit.item);return;}
  hit.list.splice(hit.index,1);
  node.__lmdSelection=hit.kind==="extra"?{track:`layer:${hit.track.track_id}`,id:hit.track.track_id}:{track:`layer:${hit.kind}`,id:hit.kind};
  commit(node,doc);
}
function copyTimelineHit(node,hit){
  if(!hit)return false;
  node.__lmdTimelineClipboard={kind:hit.kind,track_id:hit.track?.track_id||null,track_type:hit.track?.type||null,payload:JSON.parse(JSON.stringify(hit.item))};
  return true;
}
function cutTimelineHit(node,doc,hit){if(!copyTimelineHit(node,hit))return;deleteTimelineHit(node,doc,hit);}
function pasteTimelineClipboard(node,doc,time,targetTrack=null){
  const clip=node.__lmdTimelineClipboard;if(!clip?.payload)return false;const payload=JSON.parse(JSON.stringify(clip.payload)),t=Math.max(0,finite(time,node.__lmdPlayhead||0));
  if(clip.kind==="main"){
    if(doc.shots.length>=MAX_SHOTS){alert(`Maximum ${MAX_SHOTS} MAIN clips.`);return false;}
    payload.clip_id=id("shot");payload.name=`${String(payload.name||directorMainTermUpper(doc))} copy`.slice(0,120);payload.first_frame_anchor=false;payload.last_frame_anchor=false;payload.start=t;
    let insertAt=0;while(insertAt<doc.shots.length&&shotStart(doc,insertAt)<t)insertAt++;const needEnd=t+finite(payload.duration,0),next=doc.shots[insertAt];if(next&&shotStart(doc,insertAt)<needEnd){const shift=needEnd-shotStart(doc,insertAt);for(let i=insertAt;i<doc.shots.length;i++)doc.shots[i].start=shotStart(doc,i)+shift;}doc.shots.splice(insertAt,0,payload);node.__lmdSelection={track:"main",id:payload.clip_id};commit(node,doc);return true;
  }
  if(clip.kind==="camera"){
    payload.block_id=id("camera");payload.start=t;setBlockStart(doc,payload,t);doc.camera_blocks.push(payload);node.__lmdSelection={track:"camera",id:payload.block_id};commit(node,doc);return true;
  }
  if(clip.kind==="audio"){
    payload.block_id=id("audio");payload.start=t;setBlockStart(doc,payload,t);doc.audio_blocks.push(payload);node.__lmdSelection={track:"audio",id:payload.block_id};commit(node,doc);return true;
  }
  if(clip.kind==="extra"){
    let track=null;if(String(targetTrack||"").startsWith("extra:"))track=(doc.extra_tracks||[]).find(x=>x.track_id===String(targetTrack).slice(6));if(track&&clip.track_type&&track.type!==clip.track_type)track=null;if(!track)track=(doc.extra_tracks||[]).find(x=>x.track_id===clip.track_id);if(!track){alert("The source layer no longer exists.");return false;}if((track.clips||[]).length>=MAX_EXTRA_CLIPS){alert(`Maximum ${MAX_EXTRA_CLIPS} clips per layer.`);return false;}payload.clip_id=id("layer");payload.start=t;track.clips.push(payload);node.__lmdSelection={track:`extra:${track.track_id}`,id:payload.clip_id};commit(node,doc);return true;
  }
  return false;
}
function knifeTimelineHitAt(node,doc,hit,time){
  if(!hit)return false;const t=Math.max(0,finite(time,node.__lmdPlayhead||0));let second=null;
  if(hit.kind==="main"){if(coreTrackLocked(node,"main"))return false;second=splitMainAtTime(node,doc,t,{selectRight:true,splitBoundTracks:false});}
  else if(hit.kind==="camera"||hit.kind==="audio"){if(coreTrackLocked(node,hit.kind))return false;second=splitFreeItemAt(node,doc,hit.item,hit.kind,t,null);if(second)node.__lmdSelection={track:hit.kind,id:second.block_id};}
  else if(hit.kind==="extra"){if(hit.track?.locked)return false;second=splitFreeItemAt(node,doc,hit.item,"extra",t,hit.track);if(second)node.__lmdSelection={track:`extra:${hit.track.track_id}`,id:second.clip_id};}
  if(!second)return false;node.__lmdPlayhead=t;commit(node,doc);return true;
}
function nearestCutMark(shot,start,time){const target=finite(time,start)-start;let best=null,distance=Infinity;for(const value of shot?.cut_marks||[]){const mark=finite(value,-1),d=Math.abs(mark-target);if(d<distance){best=mark;distance=d;}}return distance<=.35?best:null;}
function timelineTrackItems(doc,trackKey){if(trackKey==="main")return (doc.shots||[]).map((item,index)=>({item,index,start:shotStart(doc,index),end:shotStart(doc,index)+finite(item.duration,0)}));if(trackKey==="camera")return (doc.camera_blocks||[]).map((item,index)=>({item,index,start:finite(item.start,0),end:finite(item.start,0)+finite(item.duration,0)}));if(trackKey==="audio")return (doc.audio_blocks||[]).map((item,index)=>({item,index,start:finite(item.start,0),end:finite(item.start,0)+finite(item.duration,0)}));if(String(trackKey||"").startsWith("extra:")){const track=(doc.extra_tracks||[]).find(t=>t.track_id===String(trackKey).slice(6));return (track?.clips||[]).map((item,index)=>({item,index,start:finite(item.start,0),end:finite(item.start,0)+finite(item.duration,0),track}));}return [];}
function timelineGapAt(doc,trackKey,time){const items=timelineTrackItems(doc,trackKey).sort((a,b)=>a.start-b.start),t=Math.max(0,finite(time,0));if(!items.length)return null;const first=items[0];if(first.start>.000001&&t<=first.start+.000001)return {start:0,end:first.start,duration:first.start,left:null,right:first};let left=first,coveredEnd=first.end;for(let i=1;i<items.length;i++){const right=items[i];if(right.start>coveredEnd+.000001&&t>=coveredEnd-.000001&&t<=right.start+.000001)return {start:coveredEnd,end:right.start,duration:right.start-coveredEnd,left,right};if(right.end>coveredEnd){coveredEnd=right.end;left=right;}}return null;}
function cutTimelineSideAt(node,doc,hit,time,side){if(!hit||!(side==="left"||side==="right"))return false;const start=finite(hit.start,0),end=finite(hit.end,start+finite(hit.item?.duration,0)),at=Math.max(start+.25,Math.min(end-.25,finite(time,start)));if(at<=start+.001||at>=end-.001)return false;const item=hit.item,removed=side==="left"?at-start:end-at;if(hit.kind==="main"){if(String(item.base_kind||"generated")!=="media"||!timedMediaForShot(doc,item))return false;if(side==="left"){item.source_in=finite(item.source_in,0)+removed;item.start=at;item.duration=end-at;}else item.duration=at-start;}else{let media=null;if(hit.kind==="audio")media=timedMediaForAudio(doc,item);else if(hit.kind==="extra")media=timedMediaForExtra(doc,item);if(side==="left"){if(media)item.source_in=finite(item.source_in,0)+removed;item.start=at;item.duration=end-at;if(hit.kind==="camera"||hit.kind==="audio")setBlockStart(doc,item,at);}else item.duration=at-start;}node.__lmdPlayhead=at;commit(node,doc);return true;}
function fillTimelineGap(node,doc,trackKey,gap,choice={}){if(!gap||gap.duration<.001)return false;const start=Math.max(0,gap.start),duration=Math.max(.25,gap.duration),subject=choice.subject_id?subjectById(doc,choice.subject_id):null;if(trackKey==="main"){if(coreTrackLocked(node,"main"))return false;if(doc.shots.length>=MAX_SHOTS)return alert(`Maximum ${MAX_SHOTS} MAIN clips.`);const insertAt=Math.max(0,Math.min(doc.shots.length,(gap.right?.index??doc.shots.length))),shot=defaultShot(insertAt,duration,start);shot.name=String(doc.setup_timeline_mode)==="multiclip"?`Clip ${insertAt+1}`:`Shot ${insertAt+1}`;if(subject?.kind==="Picture"){shot.media_subject_id=subject.subject_id;shot.subjects=[subject.subject_id];shot.name=`${shot.name} · ${subject.name}`.slice(0,120);}else if(subject?.kind==="Video"&&choice.kind==="media"){shot.media_subject_id=subject.subject_id;shot.subjects=[subject.subject_id];shot.base_kind="media";shot.name=`Video · ${subject.name}`.slice(0,120);}doc.shots.splice(insertAt,0,shot);node.__lmdSelection={track:"main",id:shot.clip_id};commit(node,doc);return true;}if(trackKey==="camera"){if(coreTrackLocked(node,"camera"))return false;const block=defaultCameraBlock(start,duration);setBlockStart(doc,block,start);doc.camera_blocks.push(block);node.__lmdSelection={track:"camera",id:block.block_id};commit(node,doc);return true;}if(trackKey==="audio"){if(coreTrackLocked(node,"audio"))return false;const reference=choice.kind==="reference"&&subject?.kind==="Audio",block={block_id:id("audio"),start,duration,scene_id:null,scene_offset:0,kind:reference?"reference":"prompt",role:"diegetic",prompt:"",subject_id:reference?subject.subject_id:null,source_in:0};setBlockStart(doc,block,start);doc.audio_blocks.push(block);node.__lmdSelection={track:"audio",id:block.block_id};commit(node,doc);return true;}if(String(trackKey||"").startsWith("extra:")){const track=(doc.extra_tracks||[]).find(t=>t.track_id===String(trackKey).slice(6));if(!track||track.locked)return false;if((track.clips||[]).length>=MAX_EXTRA_CLIPS)return alert(`Maximum ${MAX_EXTRA_CLIPS} clips per layer.`);const clip=defaultExtraClip(track,start,duration);if(subject&&subject.kind===(track.type==="video"?"Video":track.type==="audio"?"Audio":"Picture"))clip.subject_id=subject.subject_id;if(choice.refmod_id&&track.type==="refmod")clip.refmod_id=String(choice.refmod_id);track.clips.push(clip);node.__lmdSelection={track:`extra:${track.track_id}`,id:clip.clip_id};commit(node,doc);return true;}return false;}
function addTimelineGapChoices(add,node,doc,trackKey,gap){const locked=trackKey==="main"||trackKey==="camera"||trackKey==="audio"?coreTrackLocked(node,trackKey):String(trackKey||"").startsWith("extra:")?Boolean(doc.extra_tracks.find(t=>t.track_id===String(trackKey).slice(6))?.locked):true,disabled=locked||gap.duration<.25,create=(choice={})=>fillTimelineGap(node,doc,trackKey,gap,choice);if(trackKey==="main"){add("Generated MAIN clip · prompt",()=>create(),{disabled:disabled||doc.shots.length>=MAX_SHOTS});for(const subject of doc.subjects||[]){if(subject.kind==="Picture"&&subject.media)add(`Generated with Picture · ${subject.name}`,()=>create({subject_id:subject.subject_id}),{disabled:disabled||doc.shots.length>=MAX_SHOTS});else if(subject.kind==="Video"&&subject.media)add(`Use Video as MAIN media · ${subject.name}`,()=>create({kind:"media",subject_id:subject.subject_id}),{disabled:disabled||doc.shots.length>=MAX_SHOTS});}return;}if(trackKey==="camera"){add("Camera direction block",()=>create(),{disabled});return;}if(trackKey==="audio"){add("Sound prompt clip",()=>create(),{disabled});for(const subject of doc.subjects||[])if(subject.kind==="Audio"&&subject.media)add(`Audio reference · ${subject.name}`,()=>create({kind:"reference",subject_id:subject.subject_id}),{disabled});return;}const track=doc.extra_tracks.find(t=>t.track_id===String(trackKey||"").slice(6));if(!track)return;const labels={prompt:"Prompt clip",embedding:"Embedding clip",character:"Character reference clip",reference:"Reference clip",video:"Video reference clip",audio:"Audio clip",refmod:"RefMod clip"};add(labels[track.type]||"Layer clip",()=>create(),{disabled:(track.locked||gap.duration<.25)||(track.clips||[]).length>=MAX_EXTRA_CLIPS});const wantedKind=track.type==="video"?"Video":track.type==="audio"?"Audio":["reference","character"].includes(track.type)?"Picture":null;if(wantedKind)for(const subject of doc.subjects||[])if(subject.kind===wantedKind&&subject.media)add(`${labels[track.type]} · ${subject.name}`,()=>create({subject_id:subject.subject_id}),{disabled:track.locked||gap.duration<.25||(track.clips||[]).length>=MAX_EXTRA_CLIPS});if(track.type==="refmod")for(const refmod of doc.refmods||[])add(`RefMod clip · ${refmod.name}`,()=>create({refmod_id:refmod.refmod_id}),{disabled:track.locked||gap.duration<.25||(track.clips||[]).length>=MAX_EXTRA_CLIPS});}
function timelineGaps(doc,trackKey){const items=timelineTrackItems(doc,trackKey).sort((a,b)=>a.start-b.start),gaps=[];if(!items.length)return gaps;if(items[0].start>.001)gaps.push({start:0,end:items[0].start,duration:items[0].start,left:null,right:items[0]});let previous=items[0];for(let i=1;i<items.length;i++){const next=items[i];if(next.start>previous.end+.001)gaps.push({start:previous.end,end:next.start,duration:next.start-previous.end,left:previous,right:next});if(next.end>previous.end)previous=next;}return gaps;}
function appendTimelineGapControl(row,node,doc,trackKey,gap,pps,height){if(!gap||gap.duration<.001)return;const center=(gap.start+gap.end)/2,slot=el("div",null,{position:"absolute",left:`${center*pps}px`,top:`${Math.max(4,Math.round(height/2)-11)}px`,transform:"translateX(-50%)",display:"flex",alignItems:"center",gap:"4px",zIndex:"38",pointerEvents:"none",whiteSpace:"nowrap"}),plus=button("+",e=>showTimelineContextMenu(node,doc,e,{trackKey,time:center,gap}),false,"Create a clip for this layer in the empty interval");plus.disabled=gap.duration<.25;Object.assign(plus.style,{width:"22px",minWidth:"22px",height:"21px",padding:"0",font:"800 12px system-ui",pointerEvents:"auto",zIndex:"1"});const badge=el("span",`${gap.duration.toFixed(2)}s`,{padding:"3px 5px",border:`1px solid ${styles.border}`,borderRadius:"4px",background:"rgba(15,18,23,.94)",color:"#c7d0dc",font:"700 8px ui-monospace,monospace",pointerEvents:"auto"});badge.title=`Empty time ${gap.start.toFixed(3)}–${gap.end.toFixed(3)} seconds`;slot.append(plus,badge);row.append(slot);}
function closeMainGap(node,doc,gap){const first=Number(gap?.right?.index),delta=finite(gap?.duration,0);if(!Number.isInteger(first)||first<1||delta<=.001)return false;for(let i=first;i<doc.shots.length;i++)doc.shots[i].start=Math.max(0,shotStart(doc,i)-delta);node.__lmdSelection={track:"main",id:doc.shots[first]?.clip_id||doc.shots[first-1]?.clip_id};node.__lmdMulticlipBoundary=null;commit(node,doc);return true;}
function closeTimelineContextMenu(node){try{node.__lmdTimelineContextMenu?.remove?.();}catch(_){}node.__lmdTimelineContextMenu=null;}
function timelineMenuItem(label,action,{disabled=false,danger=false,shortcut=""}={}){
  const row=el("div",null,{display:"grid",gridTemplateColumns:"minmax(0,1fr) auto",gap:"18px",alignItems:"center",padding:"7px 10px",borderRadius:"4px",font:"10px system-ui",color:disabled?"#626a75":danger?"#ff9898":"#e2e7ee",cursor:disabled?"default":"pointer",whiteSpace:"nowrap",userSelect:"none"});row.append(el("span",label),el("span",shortcut,{font:"8px system-ui",color:"#737d8a"}));if(!disabled){row.onmouseenter=()=>row.style.background="rgba(123,166,255,.15)";row.onmouseleave=()=>row.style.background="transparent";row.onclick=e=>{e.preventDefault();e.stopPropagation();action?.();};}return row;
}
function timelineMenuSep(){return el("div",null,{height:"1px",background:styles.border,margin:"4px 5px"});}
function showTimelineContextMenu(node,doc,event,{hit=null,trackKey=null,time=null,layerTrack=null,gap=null}={}){
  event?.preventDefault?.();event?.stopPropagation?.();closeTimelineContextMenu(node);const at=Math.max(0,finite(time,node.__lmdPlayhead||0));
  const menu=el("div",null,{position:"fixed",left:`${event?.clientX||0}px`,top:`${event?.clientY||0}px`,zIndex:"100000",minWidth:"210px",maxWidth:"320px",padding:"6px",background:"#0f1217",border:`1px solid ${styles.border}`,borderRadius:"7px",boxShadow:"0 14px 38px rgba(0,0,0,.55)",font:"10px system-ui"});node.__lmdTimelineContextMenu=menu;
  const add=(label,fn,opts={})=>{const item=timelineMenuItem(label,()=>{closeTimelineContextMenu(node);fn?.();},opts);menu.append(item);};
  if(gap){
    add(`Create ${trackKey==="main"?"MAIN clip":trackKey==="camera"?"camera block":trackKey==="audio"?"audio clip":"layer clip"} · ${gap.duration.toFixed(2)}s`,()=>{}, {disabled:true});menu.append(timelineMenuSep());addTimelineGapChoices(add,node,doc,trackKey,gap);if(trackKey==="main"&&Number(gap.right?.index)>0){menu.append(timelineMenuSep());add(`Close Gap · ${gap.duration.toFixed(2)}s`,()=>closeMainGap(node,doc,gap));}
  }else if(layerTrack){
    if(layerTrack.kind==="extra"){
      const tr=layerTrack.track;node.__lmdSelection={track:`layer:${tr.track_id}`,id:tr.track_id};add("Rename layer",()=>renameExtraTrack(node,doc,tr));add("Duplicate layer",()=>duplicateExtraTrack(node,doc,tr));add("Add clip at playhead",()=>addClipToExtraTrack(node,doc,tr));menu.append(timelineMenuSep());add(tr.enabled===false?"Enable layer":"Disable layer",()=>{tr.enabled=tr.enabled===false;commit(node,doc);});add(tr.locked?"Unlock layer":"Lock layer",()=>{tr.locked=!tr.locked;commit(node,doc);});if(["audio","video","reference","character"].includes(tr.type))add(tr.muted?"Unmute monitor":"Mute monitor",()=>{tr.muted=!tr.muted;commit(node,doc);});menu.append(timelineMenuSep());add("Delete layer",()=>{if(window.confirm(`Delete layer “${tr.name}” and all ${(tr.clips||[]).length} clip(s)?`))deleteExtraTrack(node,doc,tr);},{danger:true});
    }else{
      const kind=layerTrack.kind;node.__lmdSelection={track:`layer:${kind}`,id:kind};add("Rename layer",()=>renameCoreTrack(node,doc,kind));add(coreTrackLocked(node,kind)?"Unlock layer":"Lock layer",()=>toggleCoreTrackLock(node,kind));if(kind==="main")add(doc.base_audio_muted?"Unmute embedded audio":"Mute embedded audio",()=>{doc.base_audio_muted=!doc.base_audio_muted;commit(node,doc);});if(kind==="audio")add(doc.audio_track_muted?"Unmute monitor":"Mute monitor",()=>{doc.audio_track_muted=!doc.audio_track_muted;commit(node,doc);});menu.append(timelineMenuSep());if(kind==="camera")add("Add camera block at playhead",()=>toolbarAction(node,doc,"camera_prompt"));if(kind==="audio")add("Add sound block at playhead",()=>toolbarAction(node,doc,"sound_prompt"));if(kind==="main"&&String(doc.setup_timeline_mode)==="multiclip")add("Add MultiClip boundary at playhead",()=>addMulticlipAtPlayhead(node,doc));menu.append(timelineMenuSep());if(kind==="camera")add("Remove CAMERA layer",()=>{if(window.confirm(`Delete all ${doc.camera_blocks.length} CAMERA block(s)?`)){doc.camera_blocks=[];doc.camera_track_enabled=false;node.__lmdSelection={track:"main",id:doc.shots[0]?.clip_id};commit(node,doc);}},{danger:true});else if(kind==="audio")add("Remove AUDIO layer",()=>{if(window.confirm(`Delete all ${doc.audio_blocks.length} AUDIO block(s)?`)){doc.audio_blocks=[];doc.audio_track_enabled=false;node.__lmdSelection={track:"main",id:doc.shots[0]?.clip_id};commit(node,doc);}},{danger:true});else add("BASE layer is required",()=>{}, {disabled:true});
    }
  }else if(hit){
    if(hit.kind==="main_group"){
      const group=mergedGroupById(doc,hit.group.group_id)||hit.group,enabled=mergedGroupRunEnabled(doc,group);
      add(enabled?"Disable merged clip for RUN":"Enable merged clip for RUN",()=>setMergedGroupRunEnabled(node,doc,group,!enabled));
      menu.append(timelineMenuSep());add("Restore original clip boundaries",()=>restoreMergedGroupBoundaries(node,doc,group));
    }else{
    add("Duplicate",()=>duplicateTimelineHit(node,doc,hit),{shortcut:"Ctrl+D"});add("Copy",()=>copyTimelineHit(node,hit),{shortcut:"Ctrl+C"});add("Cut to clipboard",()=>cutTimelineHit(node,doc,hit),{shortcut:"Ctrl+X"});add("Paste at cursor",()=>pasteTimelineClipboard(node,doc,at,trackKey),{disabled:!node.__lmdTimelineClipboard,shortcut:"Ctrl+V"});menu.append(timelineMenuSep());if(hit.kind==="main"){const mark=nearestCutMark(hit.item,hit.start,at),inside=at>finite(hit.start,0)+.249&&at<finite(hit.end,0)-.249;add("Add CUT marker at playhead",()=>addCutMarkAtTime(node,doc,at),{disabled:!inside||coreTrackLocked(node,"main")});if(mark!=null)add("Remove CUT marker",()=>removeCutMarkAtTime(node,doc,hit.item,at),{danger:true});menu.append(timelineMenuSep());const clipIds=selectedMainClipIds(node,doc,hit.item.clip_id),runTargets=clipIds.map(clipId=>doc.shots.find(shot=>shot.clip_id===clipId)).filter(shot=>shot&&String(shot.base_kind||"generated")!=="media");if(runTargets.length){const enabled=runTargets.every(shot=>shot.run_enabled!==false);add(`${enabled?"Disable":"Enable"} ${runTargets.length===1?"clip":"selected clips"} for RUN`,()=>{for(const shot of runTargets)shot.run_enabled=!enabled;commit(node,doc);});}if(clipIds.length>1){const mergeCheck=mergeSelectionCheck(doc,clipIds);add("Merge selected clips into one",()=>mergeSelectedMainClips(node,doc,clipIds),{disabled:!mergeCheck.ok,title:mergeCheck.ok?"Group these adjacent generated clips as one reversible MAIN clip.":mergeCheck.reason});}menu.append(timelineMenuSep());}const canTrim=hit.kind!=="main"||(String(hit.item?.base_kind||"generated")==="media"&&Boolean(timedMediaForShot(doc,hit.item)));add("Trim left · keep right side",()=>cutTimelineSideAt(node,doc,hit,at,"left"),{disabled:!canTrim||at<=finite(hit.start,0)+.25||at>=finite(hit.end,0)-.25});add("Trim right · keep left side",()=>cutTimelineSideAt(node,doc,hit,at,"right"),{disabled:!canTrim||at<=finite(hit.start,0)+.25||at>=finite(hit.end,0)-.25});menu.append(timelineMenuSep());add("Knife clip here",()=>knifeTimelineHitAt(node,doc,hit,at),{shortcut:"B"});add("Knife all layers here",()=>{node.__lmdPlayhead=at;splitAllAtPlayhead(node,doc);});if(hit.kind==="main"||hit.kind==="extra"||(hit.kind==="audio"&&hit.item.kind!=="reference"))add("Rename",()=>renameTimelineHit(node,doc,hit));menu.append(timelineMenuSep());add("Delete",()=>deleteTimelineHit(node,doc,hit),{danger:true,shortcut:"Del"});
    }
  }else{
    const activeGap=trackKey&&trackKey!=="timeline_blank"?timelineGapAt(doc,trackKey,at):null;if(activeGap&&activeGap.duration>.001){add(`Create in Gap · ${activeGap.duration.toFixed(2)}s`,()=>{}, {disabled:true});menu.append(timelineMenuSep());addTimelineGapChoices(add,node,doc,trackKey,activeGap);if(trackKey==="main"&&Number(activeGap.right?.index)>0){menu.append(timelineMenuSep());add(`Close Gap · ${activeGap.duration.toFixed(2)}s`,()=>closeMainGap(node,doc,activeGap));}menu.append(timelineMenuSep());}
    add("Paste at cursor",()=>pasteTimelineClipboard(node,doc,at,trackKey),{disabled:!node.__lmdTimelineClipboard,shortcut:"Ctrl+V"});if(trackKey==="main")add("Add CUT marker at playhead",()=>addCutMarkAtTime(node,doc,at),{disabled:!sceneInfoAt(doc,at)||coreTrackLocked(node,"main")});add("Knife all layers here",()=>{node.__lmdPlayhead=at;splitAllAtPlayhead(node,doc);},{shortcut:"B"});menu.append(timelineMenuSep());
    if(trackKey==="camera")add("Add camera block here",()=>{node.__lmdPlayhead=at;toolbarAction(node,doc,"camera_prompt");});
    else if(trackKey==="audio")add("Add sound block here",()=>{node.__lmdPlayhead=at;toolbarAction(node,doc,"sound_prompt");});
    else if(String(trackKey||"").startsWith("extra:")){const tr=(doc.extra_tracks||[]).find(x=>x.track_id===String(trackKey).slice(6));if(tr)add("Add clip here",()=>{node.__lmdPlayhead=at;addClipToExtraTrack(node,doc,tr);});}
    else if(trackKey==="main"&&String(doc.setup_timeline_mode)==="multiclip")add("Add MultiClip boundary here",()=>{node.__lmdPlayhead=at;addMulticlipAtPlayhead(node,doc);});
    else if(!trackKey||trackKey==="timeline_blank"){
      add("Add camera block here",()=>{node.__lmdPlayhead=at;toolbarAction(node,doc,"camera_prompt");});
      add("Add sound block here",()=>{node.__lmdPlayhead=at;toolbarAction(node,doc,"sound_prompt");});
      if(String(doc.setup_timeline_mode)==="multiclip")add("Add MultiClip boundary here",()=>{node.__lmdPlayhead=at;addMulticlipAtPlayhead(node,doc);});
      menu.append(timelineMenuSep());
      const createLayer=(type)=>{node.__lmdPlayhead=at;addExtraTrack(node,doc,type);};
      add("New Prompt layer",()=>createLayer("prompt"));
      add("New Embedding layer",()=>createLayer("embedding"));
      add("New Character layer",()=>createLayer("character"));
      add("New Reference layer",()=>createLayer("reference"));
      add("New Video layer",()=>createLayer("video"));
      add("New Audio layer",()=>createLayer("audio"));
      add("New RefMod layer",()=>createLayer("refmod"));
    }
  }
  document.body.append(menu);const rect=menu.getBoundingClientRect(),left=Math.max(6,Math.min(window.innerWidth-rect.width-6,event?.clientX||0)),top=Math.max(6,Math.min(window.innerHeight-rect.height-6,event?.clientY||0));menu.style.left=`${left}px`;menu.style.top=`${top}px`;
  const outside=e=>{if(menu.contains(e.target))return;closeTimelineContextMenu(node);document.removeEventListener("pointerdown",outside,true);};setTimeout(()=>document.addEventListener("pointerdown",outside,true),0);menu.oncontextmenu=e=>{e.preventDefault();e.stopPropagation();};
}

function splitSceneBoundTracksForMain(doc,oldClipId,newClipId,splitTime){
  if(doc.ripple_tracks===false)return;
  for(const [kind,list] of [["camera",doc.camera_blocks||[]],["audio",doc.audio_blocks||[]]]){
    const additions=[];for(const b of list){if(b.scene_id!==oldClipId)continue;const start=finite(b.start,0),end=start+finite(b.duration,0);if(start>=splitTime-.000001){b.scene_id=newClipId;b.scene_offset=Math.max(0,start-splitTime);continue;}if(end>splitTime+.000001){const tail=JSON.parse(JSON.stringify(b));if(kind==="camera")tail.block_id=id("camera");else tail.block_id=id("audio");tail.start=splitTime;tail.duration=end-splitTime;tail.scene_id=newClipId;tail.scene_offset=0;if(kind==="audio"&&timedMediaForAudio(doc,b))tail.source_in=finite(b.source_in,0)+(splitTime-start);b.duration=Math.max(.25,splitTime-start);additions.push(tail);}}list.push(...additions);
  }
}
function splitMainAtTime(node,doc,t,{selectRight=true,splitBoundTracks=true}={}){
  const hit=sceneInfoAt(doc,t);if(!hit||t<=hit.start+.249||t>=hit.end-.249)return null;
  const item=hit.shot,left=t-hit.start,right=hit.end-t,second=JSON.parse(JSON.stringify(item));
  second.clip_id=id("shot");const base=String(item.name||directorMainTermUpper(doc)).replace(/ [AB]$/,'');item.name=`${base} A`;second.name=`${base} B`;
  const oldMarks=(item.cut_marks||[]).map(value=>finite(value,-1000));item.cut_marks=oldMarks.filter(mark=>mark<=left+.001);second.cut_marks=oldMarks.filter(mark=>mark>left+.001).map(mark=>Math.round((mark-left)*1e6)/1e6);
  if(timedMediaForShot(doc,item))second.source_in=finite(item.source_in,0)+left;
  const carriedLast=Boolean(item.last_frame_anchor);second.first_frame_anchor=false;second.last_frame_anchor=carriedLast;item.last_frame_anchor=false;item.duration=left;second.start=t;second.duration=right;
  doc.shots.splice(hit.index+1,0,second);if(splitBoundTracks)splitSceneBoundTracksForMain(doc,item.clip_id,second.clip_id,t);
  doc.regeneration={mode:"none",clip_id:null,request_id:null};node.__lmdMulticlipBoundary=null;if(selectRight)node.__lmdSelection={track:"main",id:second.clip_id};
  return {left:item,right:second,index:hit.index};
}
function splitFreeItemAt(node,doc,item,kind,t,track=null){
  const start=finite(item.start,0),end=start+finite(item.duration,0);if(t<=start+.249||t>=end-.249)return null;
  const left=t-start,right=end-t,second=JSON.parse(JSON.stringify(item));
  if(kind==="camera")second.block_id=id("camera");else if(kind==="audio")second.block_id=id("audio");else second.clip_id=id("layer");
  item.duration=left;second.start=t;second.duration=right;
  const media=kind==="audio"?timedMediaForAudio(doc,item):kind==="extra"?timedMediaForExtra(doc,item):null;if(media)second.source_in=finite(item.source_in,0)+left;
  if(second.scene_id){const idx=doc.shots.findIndex(s=>s.clip_id===second.scene_id);if(idx>=0)second.scene_offset=second.start-shotStart(doc,idx);}
  const list=kind==="camera"?doc.camera_blocks:kind==="audio"?doc.audio_blocks:track?.clips;if(!list)return null;const index=list.indexOf(item);if(index<0)return null;list.splice(index+1,0,second);return second;
}
function splitSelectedAtPlayhead(node,doc){
  const hit=selectedTimelineItem(node,doc);if(!hit){alert("Select a timeline clip first.");return;}const t=finite(node.__lmdPlayhead,0);if(t<=hit.start+.249||t>=hit.end-.249){alert("Place the playhead at least 0.25s inside the selected clip.");return;}
  if(hit.kind==="main"){splitMainAtTime(node,doc,t,{selectRight:true,splitBoundTracks:true});}
  else {const second=splitFreeItemAt(node,doc,hit.item,hit.kind,t,hit.track);if(second)node.__lmdSelection={track:hit.kind==="extra"?`extra:${hit.track.track_id}`:hit.kind,id:second[hit.idKey]};}
  commit(node,doc);
}
function splitAllAtPlayhead(node,doc){
  const t=finite(node.__lmdPlayhead,0);let cuts=0;const beforeSel=selection(node),selectedRight={track:null,id:null};
  const mainHit=sceneInfoAt(doc,t);let mainSplit=null;if(!coreTrackLocked(node,"main")&&mainHit&&t>mainHit.start+.249&&t<mainHit.end-.249){const selectedMain=beforeSel.track==="main"&&beforeSel.id===mainHit.shot.clip_id;mainSplit=splitMainAtTime(node,doc,t,{selectRight:false,splitBoundTracks:true});if(mainSplit){cuts++;if(selectedMain){selectedRight.track="main";selectedRight.id=mainSplit.right.clip_id;}}}
  for(const kind of ["camera","audio"]){if(coreTrackLocked(node,kind))continue;const list=kind==="camera"?doc.camera_blocks:doc.audio_blocks;for(const item of [...list]){const iid=item.block_id,wasSelected=beforeSel.track===kind&&beforeSel.id===iid;const second=splitFreeItemAt(node,doc,item,kind,t,null);if(second){cuts++;if(wasSelected){selectedRight.track=kind;selectedRight.id=second.block_id;}}}}
  for(const tr of doc.extra_tracks||[]){if(tr.locked)continue;for(const clip of [...(tr.clips||[])]){const wasSelected=beforeSel.track===`extra:${tr.track_id}`&&beforeSel.id===clip.clip_id;const second=splitFreeItemAt(node,doc,clip,"extra",t,tr);if(second){cuts++;if(wasSelected){selectedRight.track=`extra:${tr.track_id}`;selectedRight.id=second.clip_id;}}}}
  if(!cuts){alert("Place the playhead at least 0.25s inside one or more unlocked timeline clips.");return;}
  if(selectedRight.id)node.__lmdSelection=selectedRight;else if(beforeSel?.id)node.__lmdSelection=beforeSel;
  commit(node,doc);
}
function addExtraTrack(node,doc,type){if((doc.extra_tracks||[]).length>=MAX_EXTRA_TRACKS)return alert(`Maximum ${MAX_EXTRA_TRACKS} extra tracks.`);const tr=defaultExtraTrack(type);doc.extra_tracks.push(tr);const at=Math.max(0,finite(node.__lmdPlayhead,0)),hit=sceneInfoAt(doc,at),c=defaultExtraClip(tr,at,hit?.shot?.duration??5);tr.clips.push(c);node.__lmdSelection={track:`extra:${tr.track_id}`,id:c.clip_id};commit(node,doc);}
function deleteExtraTrack(node,doc,track){doc.extra_tracks=(doc.extra_tracks||[]).filter(t=>t.track_id!==track.track_id);node.__lmdSelection={track:"main",id:doc.shots[0]?.clip_id};commit(node,doc);}
function renameExtraTrack(node,doc,track){const current=String(track?.name||EXTRA_TRACK_META[track?.type]?.label||"Layer");const name=window.prompt("Rename layer",current);if(name==null||!String(name).trim()||String(name).trim()===current)return;track.name=String(name).trim().slice(0,100);commit(node,doc);}
function duplicateExtraTrack(node,doc,track){if((doc.extra_tracks||[]).length>=MAX_EXTRA_TRACKS)return alert(`Maximum ${MAX_EXTRA_TRACKS} extra tracks.`);const index=doc.extra_tracks.indexOf(track);if(index<0)return;const copy=JSON.parse(JSON.stringify(track));copy.track_id=id("track");copy.name=`${track.name} copy`.slice(0,100);copy.clips=(copy.clips||[]).map(c=>({...c,clip_id:id("layer")}));doc.extra_tracks.splice(index+1,0,copy);node.__lmdSelection={track:`layer:${copy.track_id}`,id:copy.track_id};commit(node,doc);}
function addClipToExtraTrack(node,doc,track){if(track.locked)return alert("Unlock the layer before adding a clip.");if((track.clips||[]).length>=MAX_EXTRA_CLIPS)return alert(`Maximum ${MAX_EXTRA_CLIPS} clips per layer.`);const hit=sceneInfoAt(doc,finite(node.__lmdPlayhead,0));const c=defaultExtraClip(track,finite(node.__lmdPlayhead,hit?.start??0),hit?.shot?.duration??5);track.clips.push(c);node.__lmdSelection={track:`extra:${track.track_id}`,id:c.clip_id};commit(node,doc);}

async function toolbarAction(node,doc,action){
  const total=totalDuration(doc); const sel=selection(node); const shot=selectedShot(node,doc); const selectedStart=shot?shotStart(doc,doc.shots.indexOf(shot)):total; const selectedDuration=shot?shot.duration:5;
  try {
    if(action==="split") { splitAllAtPlayhead(node,doc); }
    else if(action==="cut_mark") { addCutMarkAtTime(node,doc,finite(node.__lmdPlayhead,0)); }
    else if(action.startsWith("track_")) { addExtraTrack(node,doc,action.slice(6)); }
    else if(action==="video_prompt") { const s=defaultShot(doc.shots.length,5,totalDuration(doc));if(String(doc.setup_timeline_mode)==="multiclip")s.name=`Clip ${doc.shots.length+1}`; doc.shots.push(s); setSelection(node,"main",s.clip_id); commit(node,doc); }
    else if(action==="camera_prompt") { doc.camera_track_enabled=true;const b=defaultCameraBlock(selectedStart,selectedDuration,shot?.clip_id||null,0); doc.camera_blocks.push(b); setSelection(node,"camera",b.block_id); commit(node,doc); }
    else if(action==="sound_prompt") { doc.audio_track_enabled=true;const b={block_id:id("audio"),start:selectedStart,duration:selectedDuration,scene_id:shot?.clip_id||null,scene_offset:0,kind:"prompt",role:"diegetic",prompt:"",subject_id:null,source_in:0};doc.audio_blocks.push(b);setSelection(node,"audio",b.block_id);commit(node,doc); }
    else if(["image","video","audio"].includes(action)) { const f=await pick(action);if(!f)return;const rec=await upload(action,f);const sub=await addMediaSubject(node,doc,action,rec,f.name);if(!sub)return;
      if(action==="image"||action==="video"){const dur=action==="video"&&rec.seconds?Math.max(.25,Math.min(150,rec.seconds)):5;const s=defaultShot(doc.shots.length,dur);s.media_subject_id=sub.subject_id;s.subjects=[sub.subject_id];if(action==="video")s.base_kind="media";s.name=action==="video"?`Video · ${f.name}`:`Image · ${f.name}`;const pristine=doc.shots.length<=3&&doc.shots.every(x=>!x.prompt&&!x.media_subject_id&&!(x.subjects||[]).length);if(action==="video"&&pristine){doc.shots=[s];doc.camera_blocks=[defaultCameraBlock(0,dur,s.clip_id,0)];doc.audio_blocks=[];node.__lmdPlayhead=0;}else{const startAt=totalDuration(doc);doc.shots.push(s);node.__lmdPlayhead=startAt;}node.__lmdSelection={track:"main",id:s.clip_id};} else {doc.audio_track_enabled=true;const b={block_id:id("audio"),start:selectedStart,duration:Math.max(.25,Math.min(rec.seconds||selectedDuration,600)),kind:"reference",role:"diegetic",prompt:"",subject_id:sub.subject_id,source_in:0};doc.audio_blocks.push(b);setSelection(node,"audio",b.block_id);} commit(node,doc); }
  } catch(err) { console.error("LongMedia Director media action failed",err); alert(`LongMedia Director: ${err?.message||err}`); }
}

function ruler(total,pps,width){const r=el("div",null,{position:"relative",height:"24px",borderBottom:`1px solid ${styles.border}`,background:"#12151a"});const step=total>45?5:total>20?2:1;for(let t=0;t<=Math.ceil(total);t+=step){const x=t*pps;const line=el("div",null,{position:"absolute",left:`${x}px`,top:"10px",height:"14px",borderLeft:"1px solid #404650"});const label=el("span",`${t}s`,{position:"absolute",left:`${x+3}px`,top:"1px",font:"9px system-ui",color:"#8f98a4"});r.append(line,label);}r.style.width=`${width}px`;return r;}
function trackRow(width,height=72){return el("div",null,{position:"relative",height:`${height}px`,width:`${width}px`,background:"#0e1014",borderBottom:`1px solid ${styles.border}`});}
function blockBase(left,width,color,selected,height=72){const b=el("div",null,{position:"absolute",left:`${left}px`,top:"5px",width:`${Math.max(28,width-2)}px`,height:`${Math.max(30,height-12)}px`,boxSizing:"border-box",border:`2px solid ${selected?styles.accent:"rgba(255,255,255,.13)"}`,borderRadius:"4px",background:color,color:"#fff",cursor:"pointer",overflow:"hidden",boxShadow:selected?"0 0 0 1px #0b0d10, 0 0 10px rgba(123,166,255,.35)":"none"});b.dataset.lmdTimelineBlock="1";return b;}
function blockLabel(block,title,sub){const shade=el("div",null,{position:"absolute",inset:"0",background:"linear-gradient(90deg,rgba(0,0,0,.48),rgba(0,0,0,.08))",pointerEvents:"none"});const txt=el("div",null,{position:"relative",zIndex:"3",padding:"5px 6px",textShadow:"0 1px 2px #000",pointerEvents:"none"});const titleNode=el("div",title,{font:"600 10px system-ui",whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}),subNode=el("div",sub,{font:"9px system-ui",opacity:".82",whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis",marginTop:"3px"});txt.append(titleNode,subNode);block.append(shade,txt);return {title:titleNode,sub:subNode};}
function appendRunToggle(block,enabled,selected,onToggle){
  const control=document.createElement("button");control.type="button";control.textContent=enabled?"RUN":"OFF";control.title=enabled?"This clip is regenerated when you press RUN. Click to skip it and replay its active TAKE.":"This clip reuses its active TAKE when you press RUN. Click to include it in the next render.";
  Object.assign(control.style,{position:"absolute",right:"4px",top:"4px",zIndex:"18",height:"19px",minWidth:"34px",padding:"0 5px",border:`1px solid ${enabled?"#78c99a":"#e6a35d"}`,borderRadius:"4px",background:enabled?"rgba(22,74,48,.94)":"rgba(93,52,20,.96)",color:enabled?"#d9ffe6":"#ffe2bd",font:"800 8px system-ui",cursor:"pointer",opacity:selected?"1":"0",pointerEvents:selected?"auto":"none",transition:"opacity .12s ease"});
  control.onpointerdown=event=>{event.preventDefault();event.stopPropagation();};control.onclick=event=>{event.preventDefault();event.stopPropagation();onToggle();};
  block.addEventListener("mouseenter",()=>{control.style.opacity="1";control.style.pointerEvents="auto";});block.addEventListener("mouseleave",()=>{if(!selected){control.style.opacity="0";control.style.pointerEvents="none";}});block.append(control);
}
function renderMergedGroupBlock(node,doc,row,group,selected,pps,height){
  const indices=mergedGroupIndices(doc,group);if(indices.length<2)return;
  const first=indices[0],last=indices.at(-1),members=indices.map(index=>doc.shots[index]);
  const start=shotStart(doc,first),end=shotStart(doc,last)+finite(members.at(-1)?.duration,0),duration=Math.max(.25,end-start);
  for(const member of members){const child=node.__lmdMainTimelineBlocks.get(member.clip_id);if(child)child.style.display="none";}
  const isSelected=selected.track==="main_group"&&selected.id===group.group_id,allEnabled=mergedGroupRunEnabled(doc,group),anyEnabled=members.some(member=>member.run_enabled!==false);
  const block=blockBase(start*pps,duration*pps,styles.main,isSelected,height);block.dataset.lmdMergedGroup=group.group_id;
  if(!allEnabled){block.style.opacity=".72";block.append(el("div",anyEnabled?"PARTIAL · OFF":"SKIP ON RUN",{position:"absolute",left:"5px",bottom:"4px",zIndex:"8",padding:"2px 4px",borderRadius:"3px",background:"rgba(77,45,18,.95)",color:"#ffe2bd",font:"800 7px system-ui",pointerEvents:"none"}));}
  const takeCount=members.reduce((sum,member)=>sum+Math.max(0,Number(node.__lmdTakeStatus?.clips?.[member.clip_id]?.take_count)||0),0);
  if(takeCount)block.append(el("div",`${takeCount} takes`,{position:"absolute",right:"7px",bottom:"4px",zIndex:"6",background:"rgba(0,0,0,.72)",border:"1px solid rgba(255,255,255,.25)",borderRadius:"3px",padding:"2px 4px",font:"8px system-ui",pointerEvents:"none"}));
  const labels=blockLabel(block,`${group.name} · ${members.length} clips`,`${start.toFixed(2)}–${end.toFixed(2)}s · approved TAKEs grouped`);
  node.__lmdTimelineGeometryNodes.push({kind:"main_group",item:group,index:first,block,sub:labels.sub});
  block.onclick=event=>{event.stopPropagation();node.__lmdSelection={track:"main_group",id:group.group_id};node.__lmdMainMultiSelection=null;render(node);};
  block.oncontextmenu=event=>{const rect=row.getBoundingClientRect(),scale=pointerScale(row),at=Math.max(0,Math.min(timelineExtent(doc),(event.clientX-rect.left)/(pps*scale)));showTimelineContextMenu(node,doc,event,{hit:{kind:"main_group",group,index:first,start,end},trackKey:"main",time:at});};
  appendRunToggle(block,allEnabled,isSelected,()=>setMergedGroupRunEnabled(node,doc,group,!allEnabled));
  if(!coreTrackLocked(node,"main"))installMainReorder(block,row,node,doc,members[0],first,pps);
  row.append(block);
}
function renderMergedGroups(node,doc,row,selected,pps,height){for(const group of doc.merged_groups||[])renderMergedGroupBlock(node,doc,row,group,selected,pps,height);}

function updateTimelineGeometry(node,doc){
  const pps=clamp(node.__lmdPps,6,120,54);
  for(const entry of node.__lmdTimelineGeometryNodes||[]){
    const {kind,item,index,block,sub}=entry;let start=0,duration=0;
    if(kind==="main"){start=shotStart(doc,index);duration=finite(item.duration,0);}
    else if(kind==="main_group"){const group=mergedGroupById(doc,item.group_id)||item,indices=mergedGroupIndices(doc,group);if(indices.length){start=shotStart(doc,indices[0]);duration=shotStart(doc,indices.at(-1))+finite(doc.shots[indices.at(-1)]?.duration,0)-start;}}
    else{start=finite(item.start,0);duration=finite(item.duration,0);}
    block.style.left=`${start*pps}px`;block.style.width=`${Math.max(28,duration*pps-2)}px`;
    if(sub)sub.textContent=`${start.toFixed(2)}–${(start+duration).toFixed(2)}s`;
  }
  const width=Math.max(860,Math.ceil(timelineExtent(doc)*pps)+40);
  for(const row of node.__lmdTimelineRows||[])row.style.width=`${width}px`;
  for(const line of node.__lmdPlayheadLines||[])if(line?.isConnected)line.style.left=`${finite(node.__lmdPlayhead,0)*pps}px`;
  if(node.__lmdPlayheadHandle?.isConnected)node.__lmdPlayheadHandle.style.left=`${finite(node.__lmdPlayhead,0)*pps-6}px`;
}

function setTimelineZoom(node,doc,value,anchorX=null){const oldPps=clamp(node.__lmdPps,6,120,54),scroll=node.__lmdTimelineScrollElement,host=node.__lmdTimelineHost;if(!host?.isConnected)return;const anchor=anchorX==null?(scroll?scroll.clientWidth/2:0):anchorX,anchorTime=scroll?(scroll.scrollLeft+anchor)/oldPps:0;node.__lmdPps=clamp(value,6,120,54);saveDirectorUIPref(node,"timeline_zoom",node.__lmdPps);renderTimeline(node,doc,host);const next=node.__lmdTimelineScrollElement;if(next){next.scrollLeft=Math.max(0,anchorTime*node.__lmdPps-anchor);node.__lmdScrollLeft=next.scrollLeft;}if(node.__lmdZoomSlider)node.__lmdZoomSlider.value=String(node.__lmdPps);}
function fitTimelineZoom(node,doc){const scroll=node.__lmdTimelineScrollElement;if(!scroll)return;node.__lmdScrollLeft=0;scroll.scrollLeft=0;const extent=Math.max(.25,timelineExtent(doc)),usable=Math.max(1,scroll.clientWidth-24);setTimelineZoom(node,doc,usable/extent,0);const next=node.__lmdTimelineScrollElement;if(next){next.scrollLeft=0;node.__lmdScrollLeft=0;}}
function insertBridgeAtBoundary(node,doc){if(doc.shots.length<2||doc.shots.length>=MAX_SHOTS){alert("A bridge needs two adjacent MAIN clips and available clip capacity.");return;}let index=selectedMulticlipBoundaryIndex(node,doc);if(index<1){const t=finite(node.__lmdPlayhead,0);let best=1,distance=Infinity;for(let i=1;i<doc.shots.length;i++){const d=Math.abs(shotStart(doc,i)-t);if(d<distance){distance=d;best=i;}}index=best;}const at=shotStart(doc,index),duration=2.5,followingStarts=doc.shots.slice(index).map((_,offset)=>shotStart(doc,index+offset));for(let offset=0;offset<followingStarts.length;offset++)doc.shots[index+offset].start=followingStarts[offset]+duration;const bridge=defaultShot(index,duration,at);bridge.name=`Bridge ${index}`;bridge.prompt="";doc.shots.splice(index,0,bridge);node.__lmdSelection={track:"main",id:bridge.clip_id};setMulticlipBoundarySelection(node,doc,index+1);node.__lmdPlayhead=at;commit(node,doc);}function renderTimeline(node,doc,host){
  host.replaceChildren();node.__lmdTimelineHost=host;
  const selected=selection(node),selectedMain=selected.track==="main"?doc.shots.find(s=>s.clip_id===selected.id):null,selectedMainIndex=selectedMain?doc.shots.indexOf(selectedMain):-1,extensionEnd=selectedMain&&String(selectedMain.base_kind||"generated")!=="media"&&!coreTrackLocked(node,"main")?shotStart(doc,selectedMainIndex)+selectedMain.duration+4:0,total=Math.max(.25,timelineExtent(doc),extensionEnd),pps=clamp(node.__lmdPps,6,120,54),defaultTrackHeight=clamp(node.__lmdTrackHeight,44,120,72);node.__lmdPps=pps;node.__lmdTrackHeight=defaultTrackHeight;if(!Number.isFinite(node.__lmdPlayhead))node.__lmdPlayhead=0;
  const width=Math.max(860,Math.ceil(total*pps)+40),extras=(doc.extra_tracks||[]);
  const frame=el("div",null,{display:"grid",gridTemplateColumns:"190px minmax(0,1fr)",height:"100%",minHeight:"0",border:`1px solid ${styles.border}`,borderRadius:"7px",overflow:"hidden",background:styles.bg});
  const labels=el("div",null,{height:"100%",minHeight:"0",overflowX:"hidden",overflowY:"auto",scrollbarWidth:"none",background:"#14171c",borderRight:`1px solid ${styles.border}`});
  const labelBoxes=new Map();const addLabel=(text,color="#aab3bf",height=defaultTrackHeight,sub="",actions=[],opts={})=>{const active=Boolean(opts.active);const box=el("div",null,{position:"relative",height:`${height}px`,boxSizing:"border-box",padding:height===24?"6px 7px":"12px 6px 0",font:"700 9px system-ui",color,borderBottom:`1px solid ${styles.border}`,whiteSpace:"nowrap",overflow:"hidden",background:active?"rgba(123,166,255,.12)":"transparent",boxShadow:active?"inset 3px 0 0 #7ba6ff":"none",cursor:opts.onClick?"pointer":"default"});if(opts.onClick)box.onclick=e=>{if(e.target?.closest?.("button"))return;e.stopPropagation();opts.onClick();};if(opts.onContext)box.oncontextmenu=e=>opts.onContext(e);if(height===24){box.append(el("div",text,{font:"800 9px system-ui"}));}else{const row=el("div",null,{display:"flex",alignItems:"center",gap:"3px"});row.append(el("div",text,{font:"800 9px system-ui",overflow:"hidden",textOverflow:"ellipsis",flex:"1 1 auto"}));for(const a of actions){const b=iconButton(a.symbol,a.title,e=>{a.click(e);},a.active,23);Object.assign(b.style,{height:"21px",fontSize:"10px",borderRadius:"4px",padding:"0"});row.append(b);}box.append(row);if(sub)box.append(el("div",sub,{font:"7px system-ui",opacity:".55",marginTop:"3px",overflow:"hidden",textOverflow:"ellipsis"}));}labels.append(box);return box;};
  addLabel("TIME",styles.dim,24);
  const coreLabel=(kind,color,sub,actions=[])=>{const box=addLabel(coreTrackName(node,doc,kind),color,timelineTrackHeight(node,kind),sub,actions,{active:selected.track===`layer:${kind}`,onClick:()=>setSelection(node,`layer:${kind}`,kind),onContext:e=>showTimelineContextMenu(node,doc,e,{layerTrack:{kind}})});labelBoxes.set(kind,box);return box;};
  coreLabel("main","#8fc8e5",String(doc.setup_timeline_mode)==="multiclip"?"markers + AI takes":"cuts + AI takes",[{symbol:"✎",title:"Rename BASE layer",click:()=>renameCoreTrack(node,doc,"main")},{symbol:coreTrackLocked(node,"main")?"▣":"□",title:coreTrackLocked(node,"main")?"Unlock BASE layer editing":"Lock BASE layer editing",active:coreTrackLocked(node,"main"),click:()=>toggleCoreTrackLock(node,"main")},{symbol:doc.base_audio_muted?"🔇":"🔊",title:doc.base_audio_muted?"Unmute embedded audio from BASE video":"Mute embedded audio from BASE video",active:doc.base_audio_muted,click:()=>{doc.base_audio_muted=!doc.base_audio_muted;commit(node,doc);}}]);
  if(doc.camera_track_enabled!==false)coreLabel("camera","#c1a1e8","camera direction",[{symbol:"✎",title:"Rename CAMERA layer",click:()=>renameCoreTrack(node,doc,"camera")},{symbol:coreTrackLocked(node,"camera")?"▣":"□",title:coreTrackLocked(node,"camera")?"Unlock CAMERA layer editing":"Lock CAMERA layer editing",active:coreTrackLocked(node,"camera"),click:()=>toggleCoreTrackLock(node,"camera")}]);
  if(doc.audio_track_enabled!==false)coreLabel("audio","#8bd5a5","monitor mix",[{symbol:"✎",title:"Rename AUDIO layer",click:()=>renameCoreTrack(node,doc,"audio")},{symbol:coreTrackLocked(node,"audio")?"▣":"□",title:coreTrackLocked(node,"audio")?"Unlock AUDIO layer editing":"Lock AUDIO layer editing",active:coreTrackLocked(node,"audio"),click:()=>toggleCoreTrackLock(node,"audio")},{symbol:doc.audio_track_muted?"🔇":"🔊",title:doc.audio_track_muted?"Unmute A1 monitor playback":"Mute A1 monitor playback",active:doc.audio_track_muted,click:()=>{doc.audio_track_muted=!doc.audio_track_muted;commit(node,doc);}}]);
  for(const tr of extras){const active=selected.track===`layer:${tr.track_id}`||selected.track===`extra:${tr.track_id}`,key=`extra:${tr.track_id}`;const actions=[{symbol:"✎",title:"Rename layer",click:()=>renameExtraTrack(node,doc,tr)},{symbol:"⧉",title:"Duplicate layer",click:()=>duplicateExtraTrack(node,doc,tr)},{symbol:"×",title:"Delete layer",click:()=>{if(window.confirm(`Delete layer “${tr.name}” and all ${(tr.clips||[]).length} clip(s)?`))deleteExtraTrack(node,doc,tr);}}];const box=addLabel(`${tr.enabled===false?"○":"●"} ${tr.name}`,EXTRA_TRACK_META[tr.type]?.color||"#bbb",timelineTrackHeight(node,key),EXTRA_TRACK_META[tr.type]?.label||tr.type,actions,{active,onClick:()=>setSelection(node,`layer:${tr.track_id}`,tr.track_id),onContext:e=>showTimelineContextMenu(node,doc,e,{layerTrack:{kind:"extra",track:tr}})});labelBoxes.set(key,box);}
  labels.append(el("div",null,{height:"108px",minHeight:"108px",borderTop:`1px solid ${styles.border}`,boxSizing:"border-box"}));
  const scroll=el("div",null,{height:"100%",minHeight:"0",overflowX:"auto",overflowY:"auto",scrollBehavior:"auto"});node.__lmdTimelineScrollElement=scroll;scroll.dataset.lmdTimelineScroll="1";scroll.scrollLeft=finite(node.__lmdScrollLeft,0);scroll.scrollTop=finite(node.__lmdScrollTop,0);let syncing=false;const syncLabels=()=>{if(syncing)return;syncing=true;labels.scrollTop=scroll.scrollTop;syncing=false;};const syncTimeline=()=>{if(syncing)return;syncing=true;scroll.scrollTop=labels.scrollTop;syncing=false;};scroll.onscroll=()=>{node.__lmdScrollLeft=scroll.scrollLeft;node.__lmdScrollTop=scroll.scrollTop;syncLabels();};labels.onscroll=syncTimeline;scroll.addEventListener("wheel",e=>{e.preventDefault();const rect=scroll.getBoundingClientRect(),anchor=Math.max(0,Math.min(scroll.clientWidth,e.clientX-rect.left)),factor=Math.exp(-e.deltaY*.0015);setTimelineZoom(node,doc,pps*factor,anchor);},{passive:false});
  scroll.oncontextmenu=e=>{if(e.target!==scroll)return;const rect=scroll.getBoundingClientRect(),scale=pointerScale(scroll),at=Math.max(0,Math.min(total,(e.clientX-rect.left+scroll.scrollLeft)/(pps*scale)));showTimelineContextMenu(node,doc,e,{trackKey:"timeline_blank",time:at});};
  node.__lmdTimelineGeometryNodes=[];node.__lmdBoundaryNodes=[];node.__lmdMainTimelineBlocks=new Map();node.__lmdTimelineRows=[];const inner=el("div",null,{width:`${width}px`}),timeRuler=ruler(total,pps,width);timeRuler.oncontextmenu=e=>{const rect=timeRuler.getBoundingClientRect(),scale=pointerScale(timeRuler),at=Math.max(0,Math.min(total,(e.clientX-rect.left)/(pps*scale)));const st=selection(node).track,trackKey=st==="main"||st==="camera"||st==="audio"||String(st||"").startsWith("extra:")?st:"main";showTimelineContextMenu(node,doc,e,{trackKey,time:at});};inner.append(timeRuler);const mainHeight=timelineTrackHeight(node,"main"),cameraHeight=doc.camera_track_enabled===false?0:timelineTrackHeight(node,"camera"),audioHeight=doc.audio_track_enabled===false?0:timelineTrackHeight(node,"audio"),main=trackRow(width,mainHeight),cam=trackRow(width,cameraHeight),aud=trackRow(width,audioHeight),rows=[main,...(doc.camera_track_enabled===false?[]:[cam]),...(doc.audio_track_enabled===false?[]:[aud])];node.__lmdTimelineRows=rows;installTimelineTrackHeightGrip(node,"main",labelBoxes.get("main"),main);if(doc.camera_track_enabled!==false)installTimelineTrackHeightGrip(node,"camera",labelBoxes.get("camera"),cam);if(doc.audio_track_enabled!==false)installTimelineTrackHeightGrip(node,"audio",labelBoxes.get("audio"),aud);
  doc.shots.forEach((shot,i)=>{const start=shotStart(doc,i),end=start+shot.duration,planned=(node.__lmdRegenPreview?.indices||[]).includes(i),b=blockBase(start*pps,shot.duration*pps,styles.main,selected.track==="main"&&(selected.id===shot.clip_id||(node.__lmdMainMultiSelection||[]).includes(shot.clip_id)),mainHeight);if(planned){b.style.outline="3px solid #ffb24a";b.style.outlineOffset="-3px";}const ts=node.__lmdTakeStatus?.clips?.[shot.clip_id];if(ts?.take_count)b.append(el("div",`${ts.take_count} take${ts.take_count===1?"":"s"}`,{position:"absolute",right:"16px",bottom:"4px",zIndex:"6",background:"rgba(0,0,0,.72)",border:"1px solid rgba(255,255,255,.25)",borderRadius:"3px",padding:"2px 4px",font:"8px system-ui",pointerEvents:"none"}));const mediaSub=subjectById(doc,shot.media_subject_id);if(mediaSub?.media)decorateMedia(b,mediaSub.media,{sourceIn:shot.source_in||0,sourceDuration:shot.duration});const src=mediaSourceCaption(mediaSub?.media,shot.source_in||0,shot.duration),isMedia=String(shot.base_kind||"generated")==="media",pending=!isMedia&&doc.regeneration?.clip_id===shot.clip_id?(doc.regeneration.mode==="from_here"?" ↻":" ⟳"):"",activeDuration=finite(ts?.active?.clip_metadata?.duration,NaN),extensionLabel=!isMedia&&Number.isFinite(activeDuration)&&shot.duration>activeDuration+.001?` · EXTEND +${(shot.duration-activeDuration).toFixed(2)}s · RE-RENDER`:"",anchorBadges=`${shot.first_frame_anchor?" · FIRST":""}${shot.last_frame_anchor?" · LAST":""}`;if(extensionLabel)b.append(el("div",extensionLabel.trim(),{position:"absolute",left:"6px",right:"6px",bottom:"5px",zIndex:"8",padding:"3px 5px",border:"1px dashed #d2b8ff",borderRadius:"3px",background:"rgba(86,54,134,.82)",font:"800 8px system-ui",color:"#f0e6ff",pointerEvents:"none",whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}));const labelNodes=blockLabel(b,`${isMedia?"VIDEO · ":"TAKE · "}${displayMainName(doc,shot,i)}${pending}${extensionLabel}${anchorBadges}`,`${start.toFixed(2)}–${end.toFixed(2)}s${src?` · ${src}`:""}`);node.__lmdTimelineGeometryNodes.push({kind:"main",item:shot,index:i,block:b,sub:labelNodes.sub});node.__lmdMainTimelineBlocks.set(shot.clip_id,b);b.onclick=e=>{e.stopPropagation();if(performance.now()<finite(node.__lmdSuppressMainClickUntil,0))return;selectMainTimelineClip(node,doc,shot.clip_id,e);};b.oncontextmenu=e=>{const rect=main.getBoundingClientRect(),scale=pointerScale(main),at=Math.max(0,Math.min(total,(e.clientX-rect.left)/(pps*scale)));showTimelineContextMenu(node,doc,e,{hit:{kind:"main",item:shot,index:i,start,end,list:doc.shots,idKey:"clip_id"},trackKey:"main",time:at});};if(!coreTrackLocked(node,"main")){installMainTrim(b,main,node,doc,shot,i,pps);installMainReorder(b,main,node,doc,shot,i,pps);}if(!isMedia){const active=shot.run_enabled!==false;if(!active)b.style.opacity=".68";appendRunToggle(b,active,selected.track==="main"&&(selected.id===shot.clip_id||(node.__lmdMainMultiSelection||[]).includes(shot.clip_id)),()=>setMainClipRunEnabled(node,doc,shot,!active));}main.append(b);installCutMarks(b,main,node,doc,shot,i,pps);});renderMergedGroups(node,doc,main,selected,pps,mainHeight);
  const firstMainStart=doc.shots.length?shotStart(doc,0):0;if(firstMainStart>.001&&!coreTrackLocked(node,"main")){const gap={start:0,end:firstMainStart,duration:firstMainStart,left:null,right:{item:doc.shots[0],index:0,start:firstMainStart,end:firstMainStart+finite(doc.shots[0]?.duration,0)}},center=firstMainStart/2,slot=el("div",null,{position:"absolute",left:`${center*pps}px`,top:`${Math.max(5,Math.round(mainHeight/2)-12)}px`,transform:"translateX(-50%)",display:"flex",alignItems:"center",gap:"4px",zIndex:"38",pointerEvents:"none",whiteSpace:"nowrap"}),plus=button("+",e=>showTimelineContextMenu(node,doc,e,{trackKey:"main",time:center,gap}),false,"Create a MAIN clip in the empty time before the first clip");plus.disabled=gap.duration<.25;Object.assign(plus.style,{width:"22px",minWidth:"22px",height:"21px",padding:"0",font:"800 12px system-ui",pointerEvents:"auto",zIndex:"1"});const badge=el("span",`${gap.duration.toFixed(2)}s`,{padding:"3px 5px",border:`1px solid ${styles.border}`,borderRadius:"4px",background:"rgba(15,18,23,.94)",color:"#c7d0dc",font:"700 8px ui-monospace,monospace",pointerEvents:"auto"});badge.title=`Empty time ${gap.start.toFixed(3)}–${gap.end.toFixed(3)} seconds`;slot.append(plus,badge);main.append(slot);}
  if(doc.shots.length>1){const selectedBoundary=selectedMulticlipBoundaryIndex(node,doc);for(let i=1;i<doc.shots.length;i++){const leftShot=doc.shots[i-1],rightShot=doc.shots[i];const leftGroup=mergedGroupForClip(doc,leftShot.clip_id);if(leftGroup&&leftGroup===mergedGroupForClip(doc,rightShot.clip_id))continue;const leftEnd=shotStart(doc,i-1)+finite(leftShot.duration,0),rightStart=shotStart(doc,i),gap={start:leftEnd,end:rightStart,duration:rightStart-leftEnd,left:{item:leftShot,index:i-1,start:shotStart(doc,i-1),end:leftEnd},right:{item:rightShot,index:i,start:rightStart,end:rightStart+finite(rightShot.duration,0)}};if(gap.duration>.001){const center=(gap.start+gap.end)/2,active=i===selectedBoundary,slot=el("div",null,{position:"absolute",left:`${center*pps}px`,top:`${Math.max(5,Math.round(mainHeight/2)-12)}px`,transform:"translateX(-50%)",display:"flex",alignItems:"center",gap:"4px",zIndex:"38",pointerEvents:"none",whiteSpace:"nowrap"});const plus=button("+",e=>showTimelineContextMenu(node,doc,e,{trackKey:"main",time:center,gap}),false,"Create a MAIN clip in this gap");plus.disabled=gap.duration<.25;const cross=button("×",()=>closeMainGap(node,doc,gap),false,"Close this gap and ripple later MAIN clips left");for(const control of [plus,cross]){Object.assign(control.style,{width:"22px",minWidth:"22px",height:"21px",padding:"0",font:"800 12px system-ui",pointerEvents:"auto",zIndex:"1"});control.onpointerdown=e=>e.stopPropagation();}const badge=el("span",`${gap.duration.toFixed(2)}s`,{padding:"3px 5px",border:`1px solid ${active?"#9677cf":"#535d6b"}`,borderRadius:"4px",background:"rgba(15,18,23,.94)",color:active?"#d9c7ff":"#c7d0dc",font:"700 8px ui-monospace,monospace",pointerEvents:"auto"});badge.title=`Gap ${gap.start.toFixed(3)}–${gap.end.toFixed(3)} seconds`;slot.append(plus,badge,cross);main.append(slot);continue;}const x=rightStart*pps,active=i===selectedBoundary,marker=el("div",null,{position:"absolute",left:`${x-6}px`,top:"-2px",width:"12px",height:`${Math.max(32,mainHeight-6)}px`,zIndex:"35",cursor:"ew-resize",pointerEvents:"auto"});marker.dataset.lmdBoundary="1";marker.title="Clip boundary · drag to adjust the adjacent clip durations";marker.append(el("div",null,{position:"absolute",left:"5px",top:"0",bottom:"0",width:active?"2px":"1px",background:active?"#ffcf67":"#f1f4f8",boxShadow:active?"0 0 5px rgba(255,207,103,.85)":"0 0 3px rgba(0,0,0,.8)"}),el("div",null,{position:"absolute",left:"2px",top:"0",width:"0",height:"0",borderLeft:"4px solid transparent",borderRight:"4px solid transparent",borderTop:`7px solid ${active?"#ffcf67":"#f1f4f8"}`}));marker.onclick=e=>{e.preventDefault();e.stopPropagation();setMulticlipBoundarySelection(node,doc,i);render(node);};installMulticlipBoundary(marker,main,node,doc,i,pps);node.__lmdBoundaryNodes.push({marker,index:i});main.append(marker);}}
  if(selectedMain){const zone=installMainExtendZone(main,node,doc,selectedMain,selectedMainIndex,pps,mainHeight);if(zone)main.append(zone);}
  doc.camera_blocks.forEach((c,i)=>{const start=finite(c.start,0),end=start+finite(c.duration,0),b=blockBase(start*pps,c.duration*pps,styles.camera,selected.track==="camera"&&selected.id===c.block_id,cameraHeight);const labelNodes=blockLabel(b,c.camera?.movement||"Camera",`${start.toFixed(2)}–${end.toFixed(2)}s`);node.__lmdTimelineGeometryNodes.push({kind:"camera",item:c,index:i,block:b,sub:labelNodes.sub});b.onclick=e=>{e.stopPropagation();setSelection(node,"camera",c.block_id);};b.oncontextmenu=e=>{const rect=cam.getBoundingClientRect(),scale=pointerScale(cam),at=Math.max(0,Math.min(total,(e.clientX-rect.left)/(pps*scale)));showTimelineContextMenu(node,doc,e,{hit:{kind:"camera",item:c,index:i,start,end,list:doc.camera_blocks,idKey:"block_id"},trackKey:"camera",time:at});};if(!coreTrackLocked(node,"camera")){installFreeTrim(b,cam,node,doc,c,"camera",pps);installFreeMove(b,cam,node,doc,c,"camera",pps);}cam.append(b);});
  doc.audio_blocks.forEach((a,i)=>{const start=finite(a.start,0),end=start+finite(a.duration,0),b=blockBase(start*pps,a.duration*pps,styles.audio,selected.track==="audio"&&selected.id===a.block_id,audioHeight),sub=subjectById(doc,a.subject_id);if(sub?.media)decorateMedia(b,sub.media,{audioVisual:true,sourceIn:a.source_in||0,sourceDuration:a.duration});const labelNodes=blockLabel(b,a.kind==="reference"?(sub?.name||"Audio reference"):(a.prompt||"Sound prompt"),`${a.role} · ${start.toFixed(2)}–${end.toFixed(2)}s`);node.__lmdTimelineGeometryNodes.push({kind:"audio",item:a,index:i,block:b,sub:labelNodes.sub});b.onclick=e=>{e.stopPropagation();setSelection(node,"audio",a.block_id);};b.oncontextmenu=e=>{const rect=aud.getBoundingClientRect(),scale=pointerScale(aud),at=Math.max(0,Math.min(total,(e.clientX-rect.left)/(pps*scale)));showTimelineContextMenu(node,doc,e,{hit:{kind:"audio",item:a,index:i,start,end,list:doc.audio_blocks,idKey:"block_id"},trackKey:"audio",time:at});};if(!coreTrackLocked(node,"audio")){installFreeTrim(b,aud,node,doc,a,"audio",pps);installFreeMove(b,aud,node,doc,a,"audio",pps);}aud.append(b);});if(doc.camera_track_enabled!==false&&!coreTrackLocked(node,"camera"))for(const gap of timelineGaps(doc,"camera"))appendTimelineGapControl(cam,node,doc,"camera",gap,pps,cameraHeight);if(doc.audio_track_enabled!==false&&!coreTrackLocked(node,"audio"))for(const gap of timelineGaps(doc,"audio"))appendTimelineGapControl(aud,node,doc,"audio",gap,pps,audioHeight);cam.style.display=doc.camera_track_enabled===false?"none":"block";aud.style.display=doc.audio_track_enabled===false?"none":"block";if(labelBoxes.get("camera"))labelBoxes.get("camera").style.display=doc.camera_track_enabled===false?"none":"block";if(labelBoxes.get("audio"))labelBoxes.get("audio").style.display=doc.audio_track_enabled===false?"none":"block";inner.append(main,cam,aud);
  for(const tr of extras){const key=`extra:${tr.track_id}`,height=timelineTrackHeight(node,key),row=trackRow(width,height);rows.push(row);installTimelineTrackHeightGrip(node,key,labelBoxes.get(key),row);row.style.opacity=tr.enabled===false?".42":"1";for(const [ci,c] of (tr.clips||[]).entries()){const start=finite(c.start,0),end=start+finite(c.duration,0),meta=EXTRA_TRACK_META[tr.type]||{},b=blockBase(start*pps,c.duration*pps,meta.color||"#555",selected.track===`extra:${tr.track_id}`&&selected.id===c.clip_id,height),sub=subjectById(doc,c.subject_id);if(sub?.media)decorateMedia(b,sub.media,{audioVisual:tr.type==="audio",sourceIn:c.source_in||0,sourceDuration:c.duration});const rmRec=tr.type==="refmod"?(doc.refmods||[]).find(r=>r.refmod_id===c.refmod_id)||null:null;
const labelNodes=blockLabel(b,tr.type==="embedding"?(c.embedding_name||c.name||meta.label):rmRec?rmRec.name:(c.name||meta.label),`${start.toFixed(2)}–${end.toFixed(2)}s${rmRec?` · ${rmRec.mode} · str ${clamp(c.strength,0,1,1).toFixed(2)}`:tr.type==="embedding"&&c.embedding_name?` · ${c.embedding_name}`:c.prompt?" · prompt":""}`);node.__lmdTimelineGeometryNodes.push({kind:"extra",item:c,index:ci,block:b,sub:labelNodes.sub});b.onclick=e=>{e.stopPropagation();setSelection(node,`extra:${tr.track_id}`,c.clip_id);};b.oncontextmenu=e=>{const rect=row.getBoundingClientRect(),scale=pointerScale(row),at=Math.max(0,Math.min(total,(e.clientX-rect.left)/(pps*scale)));showTimelineContextMenu(node,doc,e,{hit:{kind:"extra",track:tr,item:c,index:ci,start,end,list:tr.clips,idKey:"clip_id"},trackKey:`extra:${tr.track_id}`,time:at});};if(!tr.locked){installFreeTrim(b,row,node,doc,c,"extra",pps);installFreeMove(b,row,node,doc,c,"extra",pps,`extra:${tr.track_id}`);}row.append(b);}if(!tr.locked)for(const gap of timelineGaps(doc,key))appendTimelineGapControl(row,node,doc,key,gap,pps,height);row.addEventListener("dblclick",e=>{if(tr.locked)return;const rect=row.getBoundingClientRect(),scale=pointerScale(row),time=snapTime(node,doc,Math.max(0,(e.clientX-rect.left)/(pps*scale)),pps,{includePlayhead:true}),c=defaultExtraClip(tr,time,5);tr.clips.push(c);node.__lmdSelection={track:`extra:${tr.track_id}`,id:c.clip_id};commit(node,doc);});row.oncontextmenu=e=>{if(e.target!==row)return;const rect=row.getBoundingClientRect(),scale=pointerScale(row),at=Math.max(0,Math.min(total,(e.clientX-rect.left)/(pps*scale)));showTimelineContextMenu(node,doc,e,{trackKey:`extra:${tr.track_id}`,time:at});};if(["refmod","character","reference","video","audio"].includes(tr.type))installTimelineDrop(row,node,doc,"layer",pps,tr);inner.append(row);}
  const blankLane=el("div",null,{height:"108px",minHeight:"108px",position:"relative",background:"#0f1217",borderTop:`1px solid ${styles.border}`,boxSizing:"border-box",cursor:"default"});blankLane.dataset.lmdTimelineBlank="1";const blankHint=el("div","CREATE LAYER",{position:"absolute",left:"12px",top:"9px",font:"700 8px system-ui",letterSpacing:".08em",color:"#4f5966",pointerEvents:"none",userSelect:"none"});blankLane.append(blankHint);const layerButtons=el("div",null,{position:"absolute",left:"12px",right:"12px",top:"30px",display:"flex",alignItems:"center",alignContent:"flex-start",gap:"5px",flexWrap:"wrap"});const addLayerButton=(label,title,create)=>{const control=button(label,create,false,title);control.style.height="25px";control.style.padding="3px 7px";layerButtons.append(control);};if(doc.camera_track_enabled===false)addLayerButton("+ CAMERA LAYER","Add the core CAMERA direction layer",()=>{doc.camera_track_enabled=true;doc.camera_blocks=[];node.__lmdSelection={track:"layer:camera",id:"camera"};commit(node,doc);});if(doc.audio_track_enabled===false)addLayerButton("+ AUDIO LAYER","Add the core AUDIO mix layer",()=>{doc.audio_track_enabled=true;doc.audio_blocks=[];node.__lmdSelection={track:"layer:audio",id:"audio"};commit(node,doc);});const extraLayerNames={prompt:"+ PROMPT LAYER",embedding:"+ EMBEDDING LAYER",character:"+ CHARACTER LAYER",reference:"+ REFERENCE LAYER",video:"+ VIDEO LAYER",audio:"+ AUDIO REF LAYER",refmod:"+ REFMOD LAYER"};for(const type of EXTRA_TRACK_TYPES){const control=button(extraLayerNames[type]||`+ ${String(type).toUpperCase()} LAYER`,()=>addExtraTrack(node,doc,type),false,`Create a new ${EXTRA_TRACK_META[type]?.label||type} layer with its first clip`);control.style.height="25px";control.style.padding="3px 7px";control.disabled=(doc.extra_tracks||[]).length>=MAX_EXTRA_TRACKS;layerButtons.append(control);}blankLane.append(layerButtons);
  const blankTime=e=>{const rect=blankLane.getBoundingClientRect(),scale=pointerScale(blankLane);return Math.max(0,Math.min(total,(e.clientX-rect.left)/(pps*scale)));};blankLane.oncontextmenu=e=>showTimelineContextMenu(node,doc,e,{trackKey:"timeline_blank",time:blankTime(e)});blankLane.onclick=e=>{if(e.defaultPrevented)return;stopPlayback(node);setPlayheadTime(node,doc,snapTime(node,doc,blankTime(e),pps,{bypass:e.altKey}));};inner.append(blankLane);
  const timeForRow=(row,clientX)=>{const rect=row.getBoundingClientRect(),scale=pointerScale(row);return Math.max(0,Math.min(total,(clientX-rect.left)/(pps*scale)));};
  main.oncontextmenu=e=>{if(e.target!==main)return;showTimelineContextMenu(node,doc,e,{trackKey:"main",time:timeForRow(main,e.clientX)});};cam.oncontextmenu=e=>{if(e.target!==cam)return;showTimelineContextMenu(node,doc,e,{trackKey:"camera",time:timeForRow(cam,e.clientX)});};aud.oncontextmenu=e=>{if(e.target!==aud)return;showTimelineContextMenu(node,doc,e,{trackKey:"audio",time:timeForRow(aud,e.clientX)});};
  const timeFromClient=clientX=>timeForRow(main,clientX),setFromEvent=e=>{if(e.defaultPrevented)return;stopPlayback(node);setPlayheadTime(node,doc,snapTime(node,doc,timeFromClient(e.clientX),pps,{bypass:e.altKey}));};for(const row of rows)row.addEventListener("click",setFromEvent);
  const x=Math.max(0,Math.min(total,node.__lmdPlayhead))*pps;node.__lmdPlayheadLines=[];for(const row of rows){const line=el("div",null,{position:"absolute",left:`${x}px`,top:"0",bottom:"0",width:"1px",background:"#ffcf67",zIndex:"24",pointerEvents:"none",boxShadow:"0 0 4px rgba(255,207,103,.8)"});row.append(line);node.__lmdPlayheadLines.push(line);}const handle=el("div",null,{position:"absolute",left:`${x-6}px`,top:"0",width:"12px",height:"24px",zIndex:"32",cursor:"ew-resize",pointerEvents:"auto"});attachDirectorTooltip(handle,"Drag playhead. With Magnet enabled it snaps to frames and clip edges.");handle.append(el("div",null,{position:"absolute",left:"5px",top:"7px",bottom:"0",width:"2px",background:"#ffcf67"}),el("div",null,{position:"absolute",left:"2px",top:"2px",width:"0",height:"0",borderLeft:"4px solid transparent",borderRight:"4px solid transparent",borderTop:"6px solid #ffcf67"}));timeRuler.append(handle);node.__lmdPlayheadHandle=handle;
  const beginScrub=e=>{e.preventDefault();e.stopPropagation();stopPlayback(node);const move=ev=>setPlayheadTime(node,node.__lmdDoc??doc,snapTime(node,node.__lmdDoc??doc,timeFromClient(ev.clientX),pps)),up=()=>{window.removeEventListener("pointermove",move,true);window.removeEventListener("pointerup",up,true);};window.addEventListener("pointermove",move,true);window.addEventListener("pointerup",up,true);move(e);};timeRuler.addEventListener("pointerdown",beginScrub);handle.addEventListener("pointerdown",beginScrub);installTimelineDrop(main,node,doc,"main",pps);installTimelineDrop(cam,node,doc,"smart",pps);installTimelineDrop(aud,node,doc,"audio",pps);scroll.append(inner);frame.append(labels,scroll);host.append(frame);requestAnimationFrame(()=>{labels.scrollTop=scroll.scrollTop;});updatePlayheadUI(node,doc);
}
function duplicateMainShot(node,doc,shot){const index=doc.shots.indexOf(shot);if(index<0||doc.shots.length>=MAX_SHOTS)return;const copy=JSON.parse(JSON.stringify(shot));const oldId=shot.clip_id,insertStart=shotStart(doc,index)+finite(shot.duration,0),shift=finite(shot.duration,0);copy.clip_id=id("shot");copy.name=`${shot.name} copy`;copy.start=insertStart;for(let i=index+1;i<doc.shots.length;i++)doc.shots[i].start=shotStart(doc,i)+shift;doc.shots.splice(index+1,0,copy);if(doc.ripple_tracks!==false){const bound=blocksForScene(doc,oldId);for(const c of bound.camera){const x=JSON.parse(JSON.stringify(c));x.block_id=id("camera");x.scene_id=copy.clip_id;doc.camera_blocks.push(x);}for(const a of bound.audio){const x=JSON.parse(JSON.stringify(a));x.block_id=id("audio");x.scene_id=copy.clip_id;doc.audio_blocks.push(x);}}doc.regeneration={mode:"none",clip_id:null,request_id:null};node.__lmdSelection={track:"main",id:copy.clip_id};commit(node,doc);}
function deleteMainShot(node,doc,shot){const index=doc.shots.indexOf(shot);if(index<0||doc.shots.length<=1)return;const dead=shot.clip_id;doc.shots.splice(index,1);if(doc.ripple_tracks!==false){doc.camera_blocks=doc.camera_blocks.filter(b=>b.scene_id!==dead);doc.audio_blocks=doc.audio_blocks.filter(b=>b.scene_id!==dead);}else{for(const list of [doc.camera_blocks,doc.audio_blocks])for(const b of list)if(b.scene_id===dead)b.scene_id=null;}doc.regeneration={mode:"none",clip_id:null,request_id:null};node.__lmdSelection={track:"main",id:doc.shots[Math.max(0,index-1)]?.clip_id};commit(node,doc);}
function selectedTakeRevision(node,shot,status){node.__lmdSelectedTakeByClip=node.__lmdSelectedTakeByClip||{};const explicit=String(node.__lmdSelectedTakeByClip[shot.clip_id]||"");if(explicit&&(status?.takes||[]).some(t=>String(t.revision||"")===explicit))return explicit;return String(status?.editor_active_revision||status?.active_revision||status?.active?.revision||status?.takes?.[0]?.revision||"");}
function setSelectedTakeRevision(node,shot,revision){node.__lmdSelectedTakeByClip=node.__lmdSelectedTakeByClip||{};node.__lmdSelectedTakeByClip[shot.clip_id]=String(revision||"");render(node);}
function selectedTakeRecord(node,shot,status){const rev=selectedTakeRevision(node,shot,status);return (status?.takes||[]).find(t=>String(t.revision||"")===String(rev||""))||null;}
function takeMultiSelection(node,shot){node.__lmdTakeMultiSelectedByClip=node.__lmdTakeMultiSelectedByClip||{};let value=node.__lmdTakeMultiSelectedByClip[shot.clip_id];if(!(value instanceof Set)){value=new Set(Array.isArray(value)?value.map(String):[]);node.__lmdTakeMultiSelectedByClip[shot.clip_id]=value;}return value;}
function pruneTakeMultiSelection(node,shot,takes){const set=takeMultiSelection(node,shot),valid=new Set((takes||[]).map(t=>String(t.revision||"")));for(const rev of [...set])if(!valid.has(rev))set.delete(rev);return set;}
function toggleTakeMultiSelection(node,shot,revision,checked){const set=takeMultiSelection(node,shot),rev=String(revision||"");if(!rev)return;if(checked===undefined){if(set.has(rev))set.delete(rev);else set.add(rev);}else if(checked)set.add(rev);else set.delete(rev);render(node);}
async function createTake(node,doc,shot){try{const emptyDoc=buildEmptyTakeWorkingDoc(doc,shot.clip_id);const emptyShot=(emptyDoc.shots||[]).find(s=>s.clip_id===shot.clip_id)||emptyDoc.shots?.[0]||defaultShot(0);const libraryFolder=String(node.__lmdTakeFolderCurrent||"Unsorted");const r=await api.fetchApi('/longmedia/director/create_take',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project_id:doc.project_id,clip_id:shot.clip_id,clip_index:Math.max(0,doc.shots.indexOf(shot)),director_timeline_snapshot:emptyDoc,director_global_prompt_snapshot:"",shot_snapshot:JSON.parse(JSON.stringify(emptyShot)),effective_seed:emptyShot.seed??null,library_folder:libraryFolder})});const data=await r.json();if(!r.ok)throw new Error(data.error||`HTTP ${r.status}`);node.__lmdSelectedTakeByClip=node.__lmdSelectedTakeByClip||{};node.__lmdSelectedTakeByClip[shot.clip_id]=String(data.take?.revision||"");const gp=widget(node,"global_prompt");if(gp){gp.value="";try{gp.callback?.(gp.value);}catch(_){}}clearLiveLatentPreview(node);invalidateDirectorMediaViews(node);node.__lmdRegenPreview=null;commit(node,emptyDoc);if(shot?.clip_id&&node.__lmdDoc?.shots?.some?.(s=>s.clip_id===shot.clip_id))node.__lmdSelection={track:"main",id:shot.clip_id};await refreshTakeStatus(node,node.__lmdDoc||emptyDoc);render(node);}catch(err){alert(`Create take failed: ${err?.message||err}`);}}
async function duplicateTake(node,doc,shot){const liveDoc=node.__lmdDoc||doc,status=node.__lmdTakeStatus?.clips?.[shot.clip_id],revision=selectedTakeRevision(node,shot,status);if(!revision)return alert("No take selected to duplicate.");try{const r=await api.fetchApi('/longmedia/director/duplicate_take',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project_id:liveDoc.project_id,clip_id:shot.clip_id,revision})});const data=await r.json();if(!r.ok)throw new Error(data.error||`HTTP ${r.status}`);const newRevision=String(data.take?.revision||"");node.__lmdSelectedTakeByClip=node.__lmdSelectedTakeByClip||{};node.__lmdSelectedTakeByClip[shot.clip_id]=newRevision;if(data.take?.director_timeline_snapshot)applyTakeTimelineSnapshot(node,liveDoc,data.take);await refreshTakeStatus(node,node.__lmdDoc||liveDoc);}catch(err){alert(`Duplicate take failed: ${err?.message||err}`);}}
async function renameTake(node,doc,shot,take){const current=String(take?.take_name||take?.label||"Take");const name=window.prompt("Rename TAKE",current);if(name==null||!String(name).trim()||String(name).trim()===current)return;try{const r=await api.fetchApi('/longmedia/director/rename_take',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project_id:doc.project_id,clip_id:shot.clip_id,revision:String(take.revision||""),name:String(name).trim()})});const data=await r.json();if(!r.ok)throw new Error(data.error||`HTTP ${r.status}`);if(data.folder_rename_pending)console.warn("LongMedia Director: TAKE label renamed; folder rename is pending until Windows releases the open file handle.");await refreshTakeStatus(node,node.__lmdDoc||doc);}catch(err){alert(`Rename take failed: ${err?.message||err}`);}}
async function createTakeFolder(node,doc){const name=window.prompt("New TAKE project folder name");if(name==null||!String(name).trim())return;try{const r=await api.fetchApi('/longmedia/director/create_take_folder',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project_id:doc.project_id,name:String(name).trim()})});const data=await r.json();if(!r.ok)throw new Error(data.error||`HTTP ${r.status}`);node.__lmdTakeFolderCurrent=String(data.name||name).trim();await refreshTakeStatus(node,node.__lmdDoc||doc);}catch(err){alert(`Create TAKE folder failed: ${err?.message||err}`);}}
async function moveTakeRevisionsToFolder(node,doc,shot,revisions,folder){
  const target=String(folder||"Unsorted"),unique=[...new Set((revisions||[]).map(String).filter(Boolean))];if(!unique.length)return alert("Select one or more TAKEs first.");
  const errors=[];let moved=0;for(const revision of unique){try{const r=await api.fetchApi('/longmedia/director/move_take',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project_id:doc.project_id,clip_id:shot.clip_id,revision,folder:target})});const data=await r.json();if(!r.ok)throw new Error(data.error||`HTTP ${r.status}`);moved++;}catch(err){errors.push(`${revision.slice(0,8)}: ${err?.message||err}`);}}
  if(moved)node.__lmdTakeFolderCurrent=target;await refreshTakeStatus(node,node.__lmdDoc||doc);if(errors.length)alert(`Some TAKEs could not be moved:
${errors.join("\n")}`);
}
async function moveTakeToFolder(node,doc,shot,take,folder){const revision=String(take?.revision||"");if(!revision)return;return moveTakeRevisionsToFolder(node,doc,shot,[revision],folder);}
async function deleteTakeRevisions(node,doc,shot,revisions){const unique=[...new Set((revisions||[]).map(String).filter(Boolean))];if(!unique.length)return;const status=node.__lmdTakeStatus?.clips?.[shot.clip_id]||{},takes=status.takes||[],names=unique.map(rev=>{const t=takes.find(x=>String(x.revision||"")===rev);return String(t?.take_name||t?.label||rev.slice(0,8));});const preview=names.slice(0,5).join("\n• "),more=names.length>5?`\n… +${names.length-5} more`:"";if(!window.confirm(`Delete ${unique.length} TAKE${unique.length===1?"":"s"}?\n\n• ${preview}${more}\n\nRendered output media files are not deleted. This action cannot be undone.`))return;const errors=[];for(const revision of unique){try{const r=await api.fetchApi('/longmedia/director/delete_take',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project_id:doc.project_id,clip_id:shot.clip_id,revision})});const data=await r.json();if(!r.ok)throw new Error(data.error||`HTTP ${r.status}`);}catch(err){errors.push(`${revision.slice(0,8)}: ${err?.message||err}`);}}
  const multi=takeMultiSelection(node,shot);for(const revision of unique)multi.delete(revision);if(node.__lmdSelectedTakeByClip&&unique.includes(String(node.__lmdSelectedTakeByClip[shot.clip_id]||"")))delete node.__lmdSelectedTakeByClip[shot.clip_id];clearLiveLatentPreview(node);await refreshTakeStatus(node,node.__lmdDoc||doc);
  const liveDoc=node.__lmdDoc||doc;if(!(liveDoc.shots||[]).some(s=>String(s.clip_id)===String(shot.clip_id))){await refreshTakeStatus(node,liveDoc);if(errors.length)alert(`Some TAKEs could not be deleted:\n${errors.join("\n")}`);return;}const liveShot=(liveDoc.shots||[]).find(s=>s.clip_id===shot.clip_id)||liveDoc.shots?.[0],nextStatus=node.__lmdTakeStatus?.clips?.[shot.clip_id],fallbackRevision=String(nextStatus?.editor_active_revision||nextStatus?.active_revision||nextStatus?.active?.revision||nextStatus?.takes?.[0]?.revision||"");if(fallbackRevision&&liveShot){node.__lmdSelectedTakeByClip=node.__lmdSelectedTakeByClip||{};node.__lmdSelectedTakeByClip[shot.clip_id]=fallbackRevision;const fallback=(nextStatus?.takes||[]).find(t=>String(t.revision||"")===fallbackRevision);if(fallback?.director_timeline_snapshot)applyTakeTimelineSnapshot(node,liveDoc,fallback);else render(node);}else{const emptyDoc=buildEmptyTakeWorkingDoc(liveDoc,shot.clip_id),gp=widget(node,"global_prompt");if(gp){gp.value="";try{gp.callback?.(gp.value);}catch(_){}}invalidateDirectorMediaViews(node);commit(node,emptyDoc);}await refreshTakeStatus(node,node.__lmdDoc||liveDoc);if(errors.length)alert(`Some TAKEs could not be deleted:\n${errors.join("\n")}`);
}
async function deleteTake(node,doc,shot){const status=node.__lmdTakeStatus?.clips?.[shot.clip_id],revision=selectedTakeRevision(node,shot,status);if(!revision)return alert("No take selected to delete.");return deleteTakeRevisions(node,doc,shot,[revision]);}
async function deleteSelectedTakes(node,doc,shot){const revisions=[...takeMultiSelection(node,shot)];if(!revisions.length)return alert("Select TAKE checkboxes first.");return deleteTakeRevisions(node,doc,shot,revisions);}
function renderShotInspector(node,doc,shot,host){const index=doc.shots.indexOf(shot),start=shotStart(doc,index),media=timedMediaForShot(doc,shot),mainTerm=directorMainTermUpper(doc),mainName=displayMainName(doc,shot,index);host.append(el("div",`MAIN · ${mainTerm} ${index+1} · ${mainName}`,{font:"700 11px system-ui",color:"#9bd4f1",marginBottom:"6px"}));const canSelectForRun=String(doc.setup_timeline_mode)==="multiclip";host.append(checkbox(shot.run_enabled===false?"Reuse active TAKE on RUN":"Regenerate this clip on RUN",shot.run_enabled!==false,enabled=>setMainClipRunEnabled(node,doc,shot,enabled),{disabled:!canSelectForRun&&shot.run_enabled!==false,title:canSelectForRun?"Disabled clips stay on the timeline and reuse their last active TAKE; enabled clips are the only generated clips sampled by ordinary RUN.":"Clip activation requires MultiClip mode. Switch this clip back on here before leaving MultiClip mode."}));
  const g=grid(media?"1.1fr .48fr .48fr .48fr":"1.2fr .6fr .55fr");g.append(field(`${mainTerm} NAME`,input(shot.name,v=>{shot.name=v;commit(node,doc,false);})),field("Duration (s)",input(shot.duration,v=>{shot.duration=clamp(v,.25,150,shot.duration);commit(node,doc);},"number"),`timeline ${start.toFixed(2)}–${(start+shot.duration).toFixed(2)}s`));if(media){g.append(field("Source In (s)",input(shot.source_in||0,v=>{shot.source_in=Math.max(0,finite(v,shot.source_in||0));commit(node,doc);},"number"),`source ${finite(media.seconds,0).toFixed(2)}s`));g.append(field("Source Out",el("div",`${(finite(shot.source_in,0)+shot.duration).toFixed(3)} s`,{height:"28px",boxSizing:"border-box",padding:"6px",background:"#0f1115",border:`1px solid ${styles.border}`,borderRadius:"4px",color:styles.text,font:"11px system-ui"}),"derived from trim"));}else g.append(field("Seed",input(shot.seed??"",v=>{shot.seed=v===""?null:Math.max(0,Math.trunc(finite(v,0)));commit(node,doc,false);},"number"),"blank = auto"));host.append(g);if(media){const seedRow=grid(".5fr 1.5fr");seedRow.append(field("Seed",input(shot.seed??"",v=>{shot.seed=v===""?null:Math.max(0,Math.trunc(finite(v,0)));commit(node,doc,false);},"number"),"blank = auto"),field("Trim",el("div","Drag the white [ ] handles on the clip. Playback stays 1×; edges change Source In / Duration and the filmstrip follows the visible source range.",{font:"10px/1.35 system-ui",color:styles.dim,padding:"5px 0"})));host.append(seedRow);}const kindRow=grid("1fr 1fr");kindRow.append(field("BASE TYPE",selectOptions([{value:"generated",label:"GENERATED · H3 TAKE"},{value:"media",label:"MEDIA · immutable video"}],shot.base_kind||"generated",v=>{if(v==="media"){const ms=subjectById(doc,shot.media_subject_id);if(ms?.kind!=="Video"||!ms?.media){alert("MEDIA requires a Video assigned to this BASE block.");return;}}shot.base_kind=v;commit(node,doc);}),"MEDIA is never diffusion-sampled. Its tail is encoded only when a following GENERATED clip needs continuation."),field("AUDIO CONTINUATION",select(["AUTO","CONTINUE","FRESH"],shot.audio_continuation||"AUTO",v=>{shot.audio_continuation=v;commit(node,doc);}),"AUTO continues across adjacent BASE blocks; FRESH resets the generated audio context."));host.append(kindRow);
  const baseSource=subjectById(doc,shot.media_subject_id);
  const baseOwnsVideo=Boolean(baseSource?.kind==="Video"&&baseSource?.media)||(shot.subjects||[]).some(sid=>{const ref=subjectById(doc,sid);return Boolean(ref?.kind==="Video"&&ref?.media&&Number(ref.slot)===1);});
  const legacyVideoScale=!baseOwnsVideo&&clamp(shot.video_reference_scale,.25,1,1)<.999;
  if(baseOwnsVideo||legacyVideoScale)host.append(field(legacyVideoScale?"EXTERNAL VIDEO 1 SCALE":"BASE VIDEO SCALE",videoReferenceScaleControl(shot.video_reference_scale,v=>{shot.video_reference_scale=v;commit(node,doc);}),legacyVideoScale?"Saved scale for an externally connected Video 1. Set 100% to remove this legacy override.":"Scales the BASE clip's video reference before H3 conditioning. An active VIDEO layer on the same slot sets its own scale. Output canvas and trim stay unchanged."));
  const anchorSubject=subjectById(doc,shot.media_subject_id),canAnchor=Boolean(anchorSubject?.kind==="Picture"&&anchorSubject?.media);const anchorBox=el("div",null,{marginTop:"8px",padding:"7px",border:`1px solid ${styles.border}`,borderRadius:"5px",background:"#11151a"});anchorBox.append(el("div","FRAME ANCHORS",{font:"800 9px system-ui",letterSpacing:".07em",color:"#c7d0dc",marginBottom:"6px"}));const anchorRow=el("div",null,{display:"flex",alignItems:"center",gap:"6px",flexWrap:"wrap"});anchorRow.append(checkbox("FIRST",shot.first_frame_anchor,enabled=>{if(enabled){for(const other of doc.shots)if(other!==shot)other.first_frame_anchor=false;}shot.first_frame_anchor=enabled;commit(node,doc);},{disabled:!canAnchor,title:canAnchor?"Use this MAIN Picture as the native opening-frame keyframe.":"FIRST/LAST require this MAIN shot to be backed by Picture media."}),checkbox("LAST",shot.last_frame_anchor,enabled=>{if(enabled){for(const other of doc.shots)if(other!==shot)other.last_frame_anchor=false;}shot.last_frame_anchor=enabled;commit(node,doc);},{disabled:!canAnchor,title:canAnchor?"Use this MAIN Picture as the native ending-frame keyframe.":"FIRST/LAST require this MAIN shot to be backed by Picture media."}));const loopActive=Boolean(shot.first_frame_anchor&&shot.last_frame_anchor);const anchorInfo=el("span",loopActive?"LOOP · same source is bound to both H3 boundaries":canAnchor?"Roles are unique across MAIN. FIRST + LAST on this shot creates an exact same-source loop anchor.":"Drop/assign a Picture to this MAIN shot to enable frame anchoring.",{font:"9px/1.3 system-ui",color:loopActive?"#9fe0b7":styles.dim,marginLeft:"4px",flex:"1 1 260px",minWidth:"180px"});anchorRow.append(anchorInfo);const shotActions=el("div",null,{display:"flex",alignItems:"center",gap:"5px",marginLeft:"auto",flex:"0 0 auto"});if(String(shot.base_kind||"generated")!=="media")shotActions.append(iconButton("⟳","Regenerate this clip now. Compatible cached seams are reused; the backend expands the dependency range only when required for safety.",()=>queueRegeneration(node,doc,shot,"clip")),iconButton("⚄","Reroll this clip now with a new seed using the same selective MultiClip dependency rules.",()=>queueRegeneration(node,doc,shot,"clip",{reroll:true})),iconButton("↻","Regenerate this clip and every dependent GENERATED clip after it now, reusing immutable MEDIA and validated TAKEs.",()=>queueRegeneration(node,doc,shot,"from_here")));shotActions.append(iconButton("⧉","Duplicate selected MAIN clip",()=>duplicateMainShot(node,doc,shot)),iconButton("⌫","Delete selected MAIN clip",()=>deleteMainShot(node,doc,shot)));anchorRow.append(shotActions);anchorBox.append(anchorRow);host.append(anchorBox);host.append(field("PROMPT PRIORITY",selectOptions(CLIP_PROMPT_GUIDANCE_OPTIONS,shot.prompt_guidance_mode,v=>{shot.prompt_guidance_mode=v;commit(node,doc);}),"Layer influence uses active prompt, character, reference, and video layers. An explicit mode overrides those layer settings for this MAIN clip. This adds prompt text; it is not a numeric attention weight."));host.append(field(String(doc.setup_timeline_mode)==="multiclip"?"CLIP PROMPT":"SEGMENT PROMPT",persistentTextarea(node,"segment:prompt",shot.prompt,v=>{shot.prompt=v;commit(node,doc,false);},4)));const castOptions=[{value:"",label:"— original cast —"},...doc.subjects.filter(s=>s.media&&["Picture","Video"].includes(s.kind)).map(s=>({value:s.subject_id,label:`${s.name} · ${runtimeToken(doc,s)}`}))];host.append(field("CAST / RECAST IDENTITY",selectOptions(castOptions,shot.cast_subject_id||"",v=>{shot.cast_subject_id=v||null;commit(node,doc);}),"Recast uses this identity. The 🎭 button on a take preserves that take's generated audio and regenerates the visual actor."));
  const refs=el("div",null,{display:"flex",gap:"5px",flexWrap:"wrap",marginTop:"6px"});refs.append(el("span","REFERENCES",{font:"700 9px system-ui",color:styles.dim,paddingTop:"5px"}));for(const s of doc.subjects){const active=shot.subjects.includes(s.subject_id);refs.append(button(`${active?"✓ ":""}${s.name} ${runtimeToken(doc,s)}`,()=>{shot.subjects=active?shot.subjects.filter(x=>x!==s.subject_id):[...shot.subjects,s.subject_id];commit(node,doc);},active));}host.append(refs);
  const regenInfo=el("div",null,{marginTop:"7px",padding:"6px 7px",border:`1px solid ${styles.border}`,borderRadius:"4px",background:"#12161c",font:"9px/1.35 system-ui",color:styles.dim});
  const pending=doc.regeneration?.clip_id===shot.clip_id?doc.regeneration:null;
  const last=node.__lmdLastRegen?.clip_id===shot.clip_id?node.__lmdLastRegen:null;
  if(pending)regenInfo.textContent=pending.mode==="from_here"?"Queued: regenerate this clip and every clip after it. Cached prefix is reused.":pending.mode==="recast"?"Queued: RECAST. The active Character/Reference layer span is authoritative: every overlapping MAIN shot is recast to the same identity while each cached generated audio performance is frozen exactly.":"Queued: regenerate only this clip. Incoming and outgoing continuation boundaries are locked when compatible cached takes exist.";
  else if(last?.state==="done")regenInfo.textContent=`Last regeneration: ${last.mode||"completed"}${last.reason?` · ${last.reason}`:""}.`;
  else if(last?.message)regenInfo.textContent=last.message;
  else regenInfo.textContent="Selective regeneration is one-click. Clip Only keeps both seams compatible when possible; the backend expands the invalidated range when required. From Here rebuilds the dependent suffix.";
  host.append(regenInfo);
}
function renderCameraInspector(node,doc,b,host){
  host.append(el("div","CAMERA",{font:"700 11px system-ui",color:"#ceb2ee",marginBottom:"6px"}));
  const timing=grid("1fr 1fr");
  timing.append(field("Start (s)",input(b.start,v=>{setBlockStart(doc,b,v);commit(node,doc);},"number")),field("Duration (s)",input(b.duration,v=>{b.duration=clamp(v,.25,600,b.duration);commit(node,doc);},"number")));
  host.append(timing);
  const library=doc.camera_preset_library||[],selected=library.find(item=>item.id===b.camera_preset_id)||null;
  node.__lmdCameraPresetNames=node.__lmdCameraPresetNames||{};
  const presetOptions=[{value:"",label:library.length?"— choose shared camera preset —":"— no saved camera presets —"},...library.map(item=>({value:item.id,label:item.name}))];
  const presetSelect=selectOptions(presetOptions,selected?.id||"",value=>{
    const preset=library.find(item=>item.id===value);
    if(!preset){b.camera_preset_id=null;delete node.__lmdCameraPresetNames[b.block_id];commit(node,doc);return;}
    b.camera=cameraPresetSettings(preset.camera);b.camera_preset_id=preset.id;node.__lmdCameraPresetNames[b.block_id]=preset.name;commit(node,doc);
  });
  const presetName=input(node.__lmdCameraPresetNames[b.block_id]??selected?.name??"",()=>{});
  presetName.placeholder="Camera preset name";presetName.oninput=()=>{node.__lmdCameraPresetNames[b.block_id]=presetName.value;};
  host.append(el("div","SHARED CAMERA PRESETS",{font:"800 9px system-ui",letterSpacing:".06em",color:"#c0c8d3",marginTop:"8px"}),
    field("Preset available to every Camera clip",presetSelect,"Stored in this Director document and kept when you add another Camera clip."),
    field("Preset name",presetName));
  const actions=el("div",null,{display:"flex",flexWrap:"wrap",gap:"5px",marginTop:"5px"});
  const saveAs=button("SAVE AS",()=>{
    const name=String(node.__lmdCameraPresetNames[b.block_id]||"").trim().slice(0,100);
    if(!name){alert("Enter a camera preset name first.");return;}
    if(library.length>=128){alert("The shared camera preset library is full (128 presets).");return;}
    if(library.some(item=>item.name.toLowerCase()===name.toLowerCase())){alert("A camera preset with this name already exists. Select it and use OVERWRITE to edit it.");return;}
    const preset={id:id("camera-preset"),name,camera:cameraPresetSettings(b.camera)};
    doc.camera_preset_library=[...library,preset];b.camera_preset_id=preset.id;node.__lmdCameraPresetNames[b.block_id]=name;commit(node,doc);
  },false,"Save this clip's camera settings as a shared preset");
  const overwrite=button("OVERWRITE",()=>{
    const index=library.findIndex(item=>item.id===b.camera_preset_id);if(index<0)return;
    const name=String(node.__lmdCameraPresetNames[b.block_id]||"").trim().slice(0,100)||library[index].name;
    if(library.some((item,i)=>i!==index&&item.name.toLowerCase()===name.toLowerCase())){alert("Another camera preset already uses this name.");return;}
    const updated=[...library];updated[index]={...updated[index],name,camera:cameraPresetSettings(b.camera)};doc.camera_preset_library=updated;node.__lmdCameraPresetNames[b.block_id]=name;commit(node,doc);
  },false,"Update the selected shared preset with this clip's current camera settings and name");
  overwrite.disabled=!selected;overwrite.style.opacity=selected?"1":".5";
  const remove=button("DELETE",()=>{
    if(!selected)return;
    doc.camera_preset_library=library.filter(item=>item.id!==selected.id);
    for(const block of doc.camera_blocks||[])if(block.camera_preset_id===selected.id)block.camera_preset_id=null;
    for(const block of doc.camera_blocks||[])delete node.__lmdCameraPresetNames[block.block_id];commit(node,doc);
  },false,"Delete this shared preset from the Director document");
  remove.disabled=!selected;remove.style.opacity=selected?"1":".5";actions.append(saveAs,overwrite,remove);
  const unchanged=selected&&selected.name===String(node.__lmdCameraPresetNames[b.block_id]??selected.name).trim()&&cameraPresetMatches(selected.camera,b.camera);
  host.append(actions,el("div",selected?(unchanged?"Selected preset · unchanged":"Selected preset · edited; OVERWRITE to update it"):(library.length?"Preset changes here stay local until saved as a shared preset.":"No shared presets yet. Save one here to use it in any Camera clip."),{font:"9px/1.35 system-ui",color:unchanged?"#8ccfa3":styles.dim,marginTop:"4px"}));
  const g=grid("1fr 1fr 1fr");
  for(const key of ["shot_size","movement","speed","rig","camera_body","lens","stabilization","transition_type","space_relation","entity_continuity"])g.append(field(key.replaceAll("_"," ").toUpperCase(),select(CAMERA[key]||[b.camera[key]],b.camera[key],v=>{b.camera[key]=v;if(key==="movement"){if(v==="Locked-Off / Static")b.camera.speed="Static";else if(b.camera.speed==="Static")b.camera.speed="Controlled";}commit(node,doc);}))) ;
  host.append(g,field("Camera note",persistentTextarea(node,`camera:${b.block_id}:note`,b.note,v=>{b.note=v;commit(node,doc,false);},2)),iconButton("⌫","Delete this camera block",()=>{doc.camera_blocks=doc.camera_blocks.filter(x=>x.block_id!==b.block_id);node.__lmdSelection={track:"main",id:doc.shots[0]?.clip_id};commit(node,doc);}));
}
function renderAudioInspector(node,doc,b,host){const media=timedMediaForAudio(doc,b);host.append(el("div","AUDIO",{font:"700 11px system-ui",color:"#95dbaa",marginBottom:"6px"}));const timing=grid(media?".55fr .55fr .55fr .75fr":".7fr .7fr .8fr");timing.append(field("Start (s)",input(b.start,v=>{setBlockStart(doc,b,v);commit(node,doc);},"number")),field("Duration (s)",input(b.duration,v=>{b.duration=clamp(v,.25,600,b.duration);commit(node,doc);},"number")));if(media)timing.append(field("Source In (s)",input(b.source_in||0,v=>{b.source_in=Math.max(0,finite(v,b.source_in||0));commit(node,doc);},"number"),`source ${finite(media.seconds,0).toFixed(2)}s`));timing.append(field("Role",select(["diegetic","music","reactive"],b.role,v=>{b.role=v;commit(node,doc);})));host.append(timing);
  const audioSubjects=doc.subjects.filter(s=>s.kind==="Audio"); if(b.kind==="reference")host.append(field("Reference",selectOptions([{value:"",label:"— choose audio reference —"},...audioSubjects.map(s=>({value:s.subject_id,label:`${s.media?.filename||s.name} · ${runtimeToken(doc,s)}`}))],b.subject_id||"",v=>{b.subject_id=v||null;b.source_in=0;commit(node,doc);}),"Choose by file/name and H3 token; internal subject IDs stay hidden."));if(media)host.append(el("div",`Visible source: ${finite(b.source_in,0).toFixed(2)}–${(finite(b.source_in,0)+b.duration).toFixed(2)}s. Drag either white bracket on the AUDIO block; waveform redraws to the remaining source range.`,{font:"9px/1.35 system-ui",color:styles.dim,margin:"4px 0 6px"}));host.append(field(b.role==="reactive"?"Audio-reactive visual direction":"Audio prompt / direction",persistentTextarea(node,`audio:${b.block_id}:prompt`,b.prompt,v=>{b.prompt=v;commit(node,doc,false);},3)),iconButton("⌫","Delete this audio block",()=>{doc.audio_blocks=doc.audio_blocks.filter(x=>x.block_id!==b.block_id);node.__lmdSelection={track:"main",id:doc.shots[0]?.clip_id};commit(node,doc);}));}
function buildEmptyTakeWorkingDoc(sourceDoc, selectedClipId=""){
  const base=normalizeDoc(JSON.parse(JSON.stringify(sourceDoc||defaultDoc())));
  const out=normalizeDoc(JSON.parse(JSON.stringify(base)));
  out.subjects=[];
  out.audio_blocks=[];
  out.extra_tracks=[];
  out.regeneration={mode:"none",clip_id:null,request_id:null,source_revision:null,recast_subject_id:null,preserve_audio:false,recast_video_denoise:1.0,recast_reference_scale:"auto",recast_clip_ids:[],source_revisions:{}};
  const selected=(out.shots||[]).find(shot=>String(shot.clip_id||"")===String(selectedClipId||""))||out.shots?.[0]||defaultShot(0);
  out.shots=[({
    ...selected,
    start:0,
    name:String(selected?.name||"Shot 1"),
    prompt:"",
    seed:null,
    subjects:[],
    media_subject_id:null,
    cast_subject_id:null,
    first_frame_anchor:false,
    last_frame_anchor:false,
    source_in:0,
    cut_marks:[],
    base_kind:"generated",
    audio_continuation:"AUTO",
  })];
  out.camera_blocks=out.camera_track_enabled===false?[]:[defaultCameraBlock(0,selected.duration,selected.clip_id,0)];
  if(selectedClipId&&out.shots.some(s=>s.clip_id===selectedClipId))out._selection_clip_id=selectedClipId;
  return normalizeDoc(out);
}

function applyTakeTimelineSnapshot(node,_doc,take,{renderAfter=true}={}){
  const raw=take?.director_timeline_snapshot;
  if(!raw||typeof raw!=="object"||Array.isArray(raw))return false;
  const globalPromptSnapshot=String(take?.director_global_prompt_snapshot??raw?._global_prompt??raw?.global_prompt??"");
  let restored;
  try{restored=normalizeDoc(JSON.parse(JSON.stringify(raw)));}
  catch(err){console.error("Director take timeline snapshot is invalid",err);return false;}
  restored.regeneration={mode:"none",clip_id:null,request_id:null,source_revision:null,recast_subject_id:null,preserve_audio:false,recast_video_denoise:1.0,recast_reference_scale:"auto",recast_clip_ids:[],source_revisions:{}};

  // TAKE restore is an editor-state replacement, not a patch onto the currently
  // rendered document. Async sidebar callbacks may hold an older doc object, so
  // mutating that object can leave Timeline / WHO & WHAT / prompts on different
  // revisions. Replace the canonical Director document atomically instead.
  const previous=String(widget(node,"director_json")?.value||"");
  pushHistory(node,previous);
  const w=widget(node,"director_json");
  if(w){w.value=JSON.stringify(restored);try{w.callback?.(w.value);}catch(_){}}
  const gp=widget(node,"global_prompt");
  if(gp){gp.value=globalPromptSnapshot;try{gp.callback?.(gp.value);}catch(_){}}

  stopPlayback(node);
  clearLiveLatentPreview(node);
  invalidateDirectorMediaViews(node);
  node.__lmdDoc=restored;
  node.__lmdRegenPreview=null;
  node.__lmdMulticlipBoundary=null;
  node.__lmdInspectorScrollTop=0;
  const selected=String(take?.clip_id||"");
  if(selected&&restored.shots.some(s=>s.clip_id===selected))node.__lmdSelection={track:"main",id:selected};
  else node.__lmdSelection={track:"main",id:restored.shots[0]?.clip_id||null};
  node.__lmdPlayhead=Math.max(0,Math.min(finite(node.__lmdPlayhead,0),totalDuration(restored)));
  node.setDirtyCanvas?.(true,true);
  app.canvas?.setDirty?.(true,true);
  if(renderAfter)render(node);
  return true;
}
async function restoreTake(node,doc,shot,revision,{applyTimeline=true,takeHint=null}={}){try{const liveDoc=node.__lmdDoc||doc;const liveShot=(liveDoc?.shots||[]).find(s=>s.clip_id===shot.clip_id)||shot;const snapshotOrder=Array.isArray(takeHint?.director_timeline_snapshot?.shots)?takeHint.director_timeline_snapshot.shots.map(s=>String(s?.clip_id||"")).filter(Boolean):[];const clipOrder=(liveDoc.shots||[]).some(s=>String(s.clip_id)===String(liveShot.clip_id))?(liveDoc.shots||[]).map(s=>s.clip_id):(snapshotOrder.length?snapshotOrder:[liveShot.clip_id]);const r=await api.fetchApi('/longmedia/director/activate_take',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project_id:liveDoc.project_id,clip_id:liveShot.clip_id,revision,clip_order:clipOrder,restore_timeline:applyTimeline})});const data=await r.json();if(!r.ok){const details=Object.entries(data.compatibility||{}).map(([side,v])=>`${side}: ${v?.reason||'unknown'}`).join('\n');alert(`Cannot restore this take safely: ${data.reasons?.join('; ' )||data.error||r.status}${details?`\n\n${details}`:''}`);return false;}node.__lmdSelectedTakeByClip=node.__lmdSelectedTakeByClip||{};node.__lmdSelectedTakeByClip[liveShot.clip_id]=String(revision||"");const snapshotTake=(data.take?.director_timeline_snapshot?data.take:(takeHint?.director_timeline_snapshot?takeHint:data.take));if(applyTimeline&&!applyTakeTimelineSnapshot(node,liveDoc,snapshotTake,{renderAfter:false})){alert("This cached take predates Director timeline snapshots. Its rendered branch was restored, but the historical timeline metadata was never stored and cannot be reconstructed exactly.");}else if(!applyTimeline){clearLiveLatentPreview(node);invalidateDirectorMediaViews(node);}await refreshTakeStatus(node,node.__lmdDoc||liveDoc,{renderAfter:false});render(node);if(data.partial_restore){const invalidated=Array.isArray(data.invalidated_clips)?data.invalidated_clips.length:0;alert(`TAKE restored. No compatible cached continuation was available${data.failed_clip_id?` at ${data.failed_clip_id}`:""}. ${invalidated} downstream clip(s) were marked for regeneration.`);}return true;}catch(err){alert(`Director take restore failed: ${err?.message||err}`);return false;}}
function recastTargetForShot(doc,shot){const index=doc.shots.indexOf(shot),start=shotStart(doc,index),end=start+shot.duration;for(const type of ["character","reference"]){for(const tr of doc.extra_tracks||[]){if(tr.enabled===false||tr.type!==type)continue;for(const c of tr.clips||[]){if(c.start<end&&c.start+c.duration>start&&c.subject_id){const sub=subjectById(doc,c.subject_id);if(sub?.media&&["Picture","Video"].includes(sub.kind))return {subject:sub,start:Math.max(0,finite(c.start,0)),end:Math.max(0,finite(c.start,0)+finite(c.duration,0)),track:tr,clip:c};}}}}if(shot.cast_subject_id){const sub=subjectById(doc,shot.cast_subject_id);if(sub)return {subject:sub,start,end,track:null,clip:null};}const shotRefs=(shot.subjects||[]).map(id=>subjectById(doc,id)).filter(s=>s?.media&&["Picture","Video"].includes(s.kind));if(shotRefs.length===1)return {subject:shotRefs[0],start,end,track:null,clip:null};const all=(doc.subjects||[]).filter(s=>s?.media&&["Picture","Video"].includes(s.kind));return all.length===1?{subject:all[0],start,end,track:null,clip:null}:null;}
function recastCandidateForShot(doc,shot){return recastTargetForShot(doc,shot)?.subject||null;}
async function recastTake(node,doc,shot,take){const targetInfo=recastTargetForShot(doc,shot),target=targetInfo?.subject;if(!target){alert("Add a Character or Reference layer with the replacement character over this shot, then press Recast again.");return;}const ok=await restoreTake(node,doc,shot,take.revision,{applyTimeline:false});if(!ok)return;await refreshTakeStatus(node,doc);const spanStart=targetInfo.clip?targetInfo.start:shotStart(doc,doc.shots.indexOf(shot)),spanEnd=targetInfo.clip?targetInfo.end:spanStart+shot.duration;const targetShots=[];let cursor=0;for(const s of doc.shots){const a=cursor,b=cursor+s.duration;if(a<spanEnd&&b>spanStart)targetShots.push(s);cursor=b;}if(!targetShots.length)targetShots.push(shot);const revisions={};for(const s of targetShots){let active=node.__lmdTakeStatus?.clips?.[s.clip_id]?.active;if(s.clip_id===shot.clip_id)active={...(active||{}),revision:String(take.revision||active?.revision||"")};const rev=String(active?.revision||"").trim();if(!rev){alert(`Recast span needs a cached active take for ${s.name}. Render/restore that shot first.`);return;}revisions[s.clip_id]=rev;s.cast_subject_id=target.subject_id;}if(Number.isFinite(Number(take.effective_seed)))shot.seed=Math.max(0,Math.trunc(Number(take.effective_seed)));const rangeMode=targetShots.length>1?"recast_range":"recast";const extra=rangeMode==="recast_range"?{recast_subject_id:target.subject_id,preserve_audio:true,recast_video_denoise:1.0,recast_reference_scale:"auto",recast_clip_ids:targetShots.map(s=>s.clip_id),source_revisions:revisions}:{source_revision:String(take.revision||""),recast_subject_id:target.subject_id,preserve_audio:true,recast_video_denoise:1.0,recast_reference_scale:"auto"};await queueRegeneration(node,doc,shot,rangeMode,{extra});}
function takeShotForSidebar(node,doc){const s=selection(node);if(s.track==="main"){const hit=doc.shots.find(x=>x.clip_id===s.id);if(hit)return hit;}return playheadHit(doc,finite(node.__lmdPlayhead,0))?.shot||doc.shots[0]||null;}
const TAKE_SORT_OPTIONS=[
  {value:"newest",label:"Newest first"},
  {value:"oldest",label:"Oldest first"},
  {value:"name_az",label:"Name · A–Z"},
  {value:"name_za",label:"Name · Z–A"},
  {value:"current_clip",label:"Current clip first"},
  {value:"render_active",label:"Render active first"},
];
function sortTakeRecords(records,mode,currentClipId){
  const collator=new Intl.Collator(undefined,{numeric:true,sensitivity:"base"});
  const byName=(a,b)=>collator.compare(String(a.take_name||a.label||""),String(b.take_name||b.label||""));
  const newest=(a,b)=>finite(b.created_at_unix,0)-finite(a.created_at_unix,0);
  return [...records].sort((a,b)=>{
    let result=0;
    if(mode==="oldest")result=finite(a.created_at_unix,0)-finite(b.created_at_unix,0)||byName(a,b);
    else if(mode==="name_az")result=byName(a,b)||newest(a,b);
    else if(mode==="name_za")result=byName(b,a)||newest(a,b);
    else if(mode==="current_clip")result=(Number(String(b.clip_id||"")===currentClipId)-Number(String(a.clip_id||"")===currentClipId))||newest(a,b);
    else if(mode==="render_active")result=(Number(Boolean(b.render_active))-Number(Boolean(a.render_active)))||(Number(Boolean(b.active))-Number(Boolean(a.active)))||newest(a,b);
    else result=newest(a,b)||byName(a,b);
    return result||String(a.revision||"").localeCompare(String(b.revision||""));
  });
}
function renderTakeSidebar(node,doc,host){
  releaseTakeGalleryImages(node);const shot=takeShotForSidebar(node,doc);host.replaceChildren();if(!shot){host.append(el("div","NO MAIN SHOT",{font:"800 9px system-ui",color:styles.dim}));return;}
  const status=node.__lmdTakeStatus?.clips?.[shot.clip_id]||{},clipTakes=status.takes||[],takes=[],seenTakes=new Set();for(const owner of doc.shots||[])for(const record of node.__lmdTakeStatus?.clips?.[owner.clip_id]?.takes||[]){const revision=String(record.revision||""),key=`${owner.clip_id}:${revision}`;if(!revision||seenTakes.has(key))continue;seenTakes.add(key);takes.push({...record,clip_id:String(record.clip_id||owner.clip_id)});}for(const record of node.__lmdTakeStatus?.all_takes||[]){const revision=String(record.revision||""),clipId=String(record.clip_id||""),key=`${clipId}:${revision}`;if(!revision||!clipId||seenTakes.has(key))continue;seenTakes.add(key);takes.push(record);}const selectedRevision=selectedTakeRevision(node,shot,status),multi=pruneTakeMultiSelection(node,shot,clipTakes);
  const folderRecords=[...(node.__lmdTakeStatus?.folders||[{name:"Unsorted",default:true}])].map(f=>({name:String(f?.name||""),take_count:Number(f?.take_count||0),default:Boolean(f?.default)})).filter(f=>f.name);if(!folderRecords.some(f=>f.name==="Unsorted"))folderRecords.unshift({name:"Unsorted",take_count:0,default:true});const folders=folderRecords.map(f=>f.name);if(!folders.includes(String(node.__lmdTakeFolderCurrent||"")))node.__lmdTakeFolderCurrent="Unsorted";
  const head=el("div",null,{position:"sticky",top:"0",zIndex:"25",background:"#171b21",padding:"8px 0 9px",borderBottom:`1px solid ${styles.border}`,boxShadow:"0 8px 12px rgba(23,27,33,.98)"});const titleRow=el("div",null,{display:"flex",alignItems:"center",justifyContent:"space-between",gap:"6px"});titleRow.append(el("div","TAKES",{font:"900 11px system-ui",letterSpacing:".08em",color:"#c7d0dc"}));const headerActions=el("div",null,{display:"flex",alignItems:"center",gap:"4px"}),sortWrap=el("div",null,{position:"relative"}),sortMode=node.__lmdTakeSort||"newest",sortLabel=TAKE_SORT_OPTIONS.find(option=>option.value===sortMode)?.label||"Newest first",sortButton=iconButton("⇅",`Sort TAKEs · ${sortLabel}`,()=>{node.__lmdTakeSortMenuOpen=!node.__lmdTakeSortMenuOpen;render(node);},Boolean(node.__lmdTakeSortMenuOpen),30);sortButton.setAttribute("aria-haspopup","menu");sortButton.setAttribute("aria-expanded",String(Boolean(node.__lmdTakeSortMenuOpen)));sortWrap.append(sortButton);if(node.__lmdTakeSortMenuOpen){const menu=el("div",null,{position:"absolute",right:"0",top:"calc(100% + 4px)",zIndex:"55",minWidth:"172px",padding:"5px",background:"#101319",border:`1px solid ${styles.border}`,borderRadius:"6px",boxShadow:"0 10px 24px rgba(0,0,0,.48)"});menu.setAttribute("role","menu");menu.setAttribute("aria-label","Sort TAKEs");for(const option of TAKE_SORT_OPTIONS){const selected=sortMode===option.value,item=button(`${selected?"✓ ":""}${option.label}`,()=>{node.__lmdTakeSort=option.value;node.__lmdTakeSortMenuOpen=false;saveDirectorUIPref(node,"take_sort",option.value);render(node);},selected,`Sort TAKEs: ${option.label}`);Object.assign(item.style,{display:"block",width:"100%",textAlign:"left",marginBottom:"3px"});item.setAttribute("role","menuitemradio");item.setAttribute("aria-checked",String(selected));menu.append(item);}menu.onkeydown=e=>{if(e.key==="Escape"){e.preventDefault();node.__lmdTakeSortMenuOpen=false;render(node);}};sortWrap.append(menu);const dismiss=e=>{if(sortWrap.contains(e.target))return;setTimeout(()=>{if(node.__lmdTakeSortDismiss!==dismiss)return;window.removeEventListener("pointerdown",dismiss,true);node.__lmdTakeSortDismiss=null;node.__lmdTakeSortMenuOpen=false;render(node);},0);};node.__lmdTakeSortDismiss=dismiss;window.addEventListener("pointerdown",dismiss,true);}headerActions.append(sortWrap,iconButton("+F","Create project folder inside longmedia_director",()=>createTakeFolder(node,doc),false,30));titleRow.append(headerActions);head.append(titleRow);head.append(el("div",`${displayMainName(doc,shot,doc.shots.indexOf(shot))} · ${clipTakes.length} for clip · ${takes.length} in project · ${multi.size} selected`,{font:"8px system-ui",color:multi.size?"#ffcf67":styles.dim,marginTop:"2px"}));
  const manage=el("div",null,{display:"grid",gridTemplateColumns:"repeat(3,minmax(0,1fr))",gap:"5px",marginTop:"7px",position:"relative"});const allSelected=clipTakes.length>0&&multi.size===clipTakes.length;const moveRevisions=multi.size?[...multi]:(selectedRevision?[selectedRevision]:[]);const moveWrap=el("div",null,{position:"relative",minWidth:"0"});const moveButton=button("MOVE TO",()=>{node.__lmdTakeMoveMenuOpen=!node.__lmdTakeMoveMenuOpen;render(node);},Boolean(node.__lmdTakeMoveMenuOpen),"Move checked TAKEs — or the active TAKE when none are checked — into an existing folder");Object.assign(moveButton.style,{width:"100%",height:"100%"});moveWrap.append(moveButton);if(node.__lmdTakeMoveMenuOpen){const menu=el("div",null,{position:"absolute",left:"0",right:"0",top:"calc(100% + 3px)",zIndex:"45",maxHeight:"240px",overflowY:"auto",padding:"5px",background:"#101319",border:`1px solid ${styles.border}`,borderRadius:"6px",boxShadow:"0 12px 30px rgba(0,0,0,.5)"});menu.append(el("div",`${moveRevisions.length||0} TAKE${moveRevisions.length===1?"":"S"} →`,{font:"800 8px system-ui",letterSpacing:".07em",color:styles.dim,padding:"3px 5px 5px"}));for(const rec of folderRecords){const folder=rec.name,row=el("div",`📁 ${folder}`,{padding:"6px 7px",borderRadius:"4px",font:"700 9px system-ui",color:"#dce2ea",cursor:moveRevisions.length?"pointer":"default",opacity:moveRevisions.length?"1":".45",whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"});if(moveRevisions.length){row.onmouseenter=()=>row.style.background="rgba(123,166,255,.15)";row.onmouseleave=()=>row.style.background="transparent";row.onclick=e=>{e.preventDefault();e.stopPropagation();node.__lmdTakeMoveMenuOpen=false;moveTakeRevisionsToFolder(node,node.__lmdDoc||doc,shot,moveRevisions,folder);};}menu.append(row);}moveWrap.append(menu);}manage.append(button("CREATE",()=>createTake(node,doc,shot)),button("DUPLICATE",()=>duplicateTake(node,doc,shot),Boolean(selectedRevision)),moveWrap,button(allSelected?"CLEAR":"ALL",()=>{if(allSelected)multi.clear();else for(const t of clipTakes)multi.add(String(t.revision||""));render(node);},multi.size>0,allSelected?"Clear TAKE multi-selection":"Select all TAKEs for this MAIN clip"),button(`DELETE ${multi.size||"SELECTED"}`,()=>deleteSelectedTakes(node,doc,shot),multi.size>0,"Delete all checked TAKEs in one batch"),button("REFRESH",()=>refreshTakeStatus(node,doc)));head.append(manage);
  const chooser=el("details",null,{position:"relative",marginTop:"7px"});const summary=el("summary",`📁 ${String(node.__lmdTakeFolderCurrent||"Unsorted")}`,{listStyle:"none",cursor:"pointer",padding:"6px 8px",border:`1px solid ${styles.border}`,borderRadius:"5px",background:"#11151a",font:"800 9px system-ui",color:"#d5dbe4",userSelect:"none"});chooser.append(summary);const tree=el("div",null,{position:"absolute",left:"0",right:"0",top:"calc(100% + 3px)",zIndex:"30",maxHeight:"260px",overflowY:"auto",padding:"5px",border:`1px solid ${styles.border}`,borderRadius:"6px",background:"#101319",boxShadow:"0 10px 28px rgba(0,0,0,.45)"});tree.append(el("div","TAKE LIBRARY",{font:"800 8px system-ui",color:styles.dim,padding:"3px 5px 5px",letterSpacing:".07em"}));for(const rec of folderRecords){const folder=rec.name,row=el("div",null,{display:"grid",gridTemplateColumns:"16px minmax(0,1fr) auto",alignItems:"center",gap:"4px",padding:"3px 4px 3px 8px",borderRadius:"4px",background:String(node.__lmdTakeFolderCurrent)===folder?"rgba(123,166,255,.14)":"transparent",cursor:"pointer"});row.append(el("span","└",{color:"#566171",font:"9px ui-monospace"}),el("span",folder,{overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",font:"700 9px system-ui"}),el("span",String(rec.take_count??0),{font:"8px system-ui",color:styles.dim}));row.onclick=()=>{node.__lmdTakeFolderCurrent=folder;render(node);};row.ondragover=e=>{e.preventDefault();if(e.dataTransfer)e.dataTransfer.dropEffect="move";};row.ondrop=e=>{e.preventDefault();e.stopPropagation();const rev=e.dataTransfer?.getData("application/x-longmedia-take"),take=takes.find(t=>String(t.revision||"")===String(rev||""));if(take){const owner=doc.shots.find(s=>String(s.clip_id)===String(take.clip_id))||{clip_id:String(take.clip_id),name:String(take.take_name||take.label||"Take")};moveTakeToFolder(node,node.__lmdDoc||doc,owner,take,folder);}};tree.append(row);}chooser.append(tree);head.append(chooser);host.append(head);
  const current=String(node.__lmdTakeFolderCurrent||"Unsorted"),visible=sortTakeRecords(takes.filter(t=>String(t.library_folder||"Unsorted")===current),sortMode,String(shot.clip_id||""));if(!visible.length){host.append(el("div",`No takes in ${current}. CREATE adds an active empty TAKE here.`,{font:"9px/1.4 system-ui",color:styles.dim,padding:"10px 2px"}));return;}
  const list=el("div",null,{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(150px,1fr))",gap:"8px",paddingBottom:"8px",alignItems:"start"});for(const t of visible){const ownerShot=doc.shots.find(s=>String(s.clip_id)===String(t.clip_id))||{clip_id:String(t.clip_id),name:String(t.take_name||t.label||"Take")},ownerStatus=node.__lmdTakeStatus?.clips?.[ownerShot.clip_id]||{},ownerSelectedRevision=selectedTakeRevision(node,ownerShot,ownerStatus),ownerMulti=takeMultiSelection(node,ownerShot),revision=String(t.revision||""),selected=revision===ownerSelectedRevision||(!doc.shots.some(s=>String(s.clip_id)===String(t.clip_id))&&Boolean(t.active)),checked=ownerMulti.has(revision),draft=Boolean(t.draft)||String(t.state||"")==="draft",name=String(t.take_name||t.label||`Take ${revision.slice(0,8)}`);const card=el("div",null,{background:"#101319",border:`1px solid ${checked?"#7ba6ff":selected?"#ffcf67":styles.border}`,borderRadius:"7px",padding:"6px",boxShadow:checked?"0 0 0 1px rgba(123,166,255,.28)":selected?"0 0 0 1px rgba(255,207,103,.32)":"none",cursor:"pointer",minWidth:"0"});card.draggable=true;card.ondragstart=e=>{e.stopPropagation();e.dataTransfer?.setData("application/x-longmedia-take",revision);if(e.dataTransfer)e.dataTransfer.effectAllowed="move";};card.onclick=e=>{if(e.ctrlKey||e.metaKey||e.shiftKey){e.preventDefault();if(ownerShot===shot)toggleTakeMultiSelection(node,ownerShot,revision);return;}if(ownerShot!==shot)node.__lmdSelection={track:"main",id:ownerShot.clip_id};restoreTake(node,node.__lmdDoc||doc,ownerShot,revision,{applyTimeline:true,takeHint:t});};
    const src=outputPreviewURL(t.preview_file,t.preview_generated_at_unix);if(src){const img=el("img",null,{width:"100%",height:"104px",objectFit:"contain",objectPosition:"center center",borderRadius:"4px",background:"#08090b",display:"block"});img.loading="lazy";img.decoding="async";img.src=src;node.__lmdTakeGalleryImages.push(img);card.append(img);}else{const emptyPreview=el("div",draft?"EMPTY TAKE":"NO PREVIEW",{height:"104px",display:"grid",placeItems:"center",background:"#0a0c10",color:draft?"#ffcf67":styles.dim,font:"800 9px system-ui",letterSpacing:draft?".08em":"normal",borderRadius:"4px"});if(!draft)emptyPreview.title="No decoded preview was saved for this TAKE. This does not by itself mean its cached latent is missing.";card.append(emptyPreview);}
    const nameRow=el("div",null,{display:"flex",alignItems:"center",gap:"4px",marginTop:"6px"});const cb=document.createElement("input");cb.type="checkbox";cb.checked=checked;cb.title="Add this TAKE to multi-selection";Object.assign(cb.style,{margin:"0 2px 0 0",cursor:"pointer",accentColor:"#7ba6ff"});cb.onclick=e=>e.stopPropagation();cb.disabled=ownerShot!==shot;cb.onchange=e=>{e.stopPropagation();toggleTakeMultiSelection(node,ownerShot,revision,cb.checked);};nameRow.append(cb,el("div",`${selected?"● ":""}${name}`,{font:"800 10px system-ui",color:selected?"#fff0bd":"#e5e9ef",whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis",flex:"1 1 auto"}));const rn=iconButton("✎","Rename TAKE and its folder",()=>renameTake(node,doc,shot,t),false,28);rn.onclick=e=>{e.stopPropagation();renameTake(node,node.__lmdDoc||doc,ownerShot,t);};nameRow.append(rn);card.append(nameRow);
    card.append(el("div",new Date((t.created_at_unix||0)*1000).toLocaleString(),{font:"8px system-ui",color:"#aab3bf",marginTop:"1px",whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}));const state=draft?"empty":t.render_active?"render active":"cached";card.append(el("div",`${state} · seed ${t.effective_seed??"—"}`,{font:"7px system-ui",color:styles.dim,marginTop:"3px",textTransform:"uppercase",letterSpacing:".04em"}));const actions=el("div",null,{display:"flex",gap:"4px",marginTop:"5px"});const restore=iconButton("↩",draft?"Restore this empty TAKE settings snapshot":"Restore this cached take and its authored Director timeline metadata",()=>restoreTake(node,doc,ownerShot,t.revision),false,30);restore.onclick=e=>{e.stopPropagation();if(ownerShot!==shot)node.__lmdSelection={track:"main",id:ownerShot.clip_id};restoreTake(node,node.__lmdDoc||doc,ownerShot,t.revision,{applyTimeline:true,takeHint:t});};actions.append(restore);if(!draft&&doc.shots.some(s=>String(s.clip_id)===String(t.clip_id))){const recast=iconButton("🎭","Recast from this rendered take",()=>recastTake(node,doc,ownerShot,t),true,30);recast.onclick=e=>{e.stopPropagation();recastTake(node,doc,ownerShot,t);};actions.append(recast);}card.append(actions);list.append(card);}
  host.append(list);
}
function renderTakeGallery(node,doc,host){return renderTakeSidebar(node,doc,host);}
function renderProgramMonitor(node,doc,host){
  if(!Number.isFinite(node.__lmdPlayhead))node.__lmdPlayhead=0;if(!PREVIEW_QUALITY_MODES.includes(String(node.__lmdPreviewQuality||"")))node.__lmdPreviewQuality="Full";node.__lmdSourceVideo?.pause?.();
  const card=el("section",null,{height:`${PROGRAM_MONITOR_HEIGHT}px`,boxSizing:"border-box",background:"linear-gradient(180deg,#161a20 0%,#111419 100%)",border:`1px solid ${styles.border}`,borderRadius:"8px",padding:"9px 10px 8px",boxShadow:"0 10px 28px rgba(0,0,0,.22)",overflow:"hidden"});
  const header=el("div",null,{display:"flex",alignItems:"center",justifyContent:"space-between",gap:"10px",height:"22px",marginBottom:"6px"});const monitorLeft=el("div",null,{display:"flex",alignItems:"center",gap:"6px"});monitorLeft.append(el("div","PROGRAM MONITOR",{font:"800 9px system-ui",letterSpacing:".11em",color:"#aeb8c6"}));if(node.__lmdLatentPreviewEnabled===undefined)node.__lmdLatentPreviewEnabled=false;const latentBtn=iconButton("◫","Mirror the live sampler latent preview in Program Monitor during rendering. After Decode the monitor automatically returns to rendered-take playback.",()=>{node.__lmdLatentPreviewEnabled=!node.__lmdLatentPreviewEnabled;if(!node.__lmdLatentPreviewEnabled)clearLiveLatentPreview(node);render(node);},Boolean(node.__lmdLatentPreviewEnabled),28);monitorLeft.append(latentBtn);header.append(monitorLeft);const title=el("div","",{font:"750 10px system-ui",color:"#ffcf67",whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis",textAlign:"right"});node.__lmdPreviewTitle=title;header.append(title);card.append(header);
  const stageRow=el("div",null,{width:"min(100%, 690px)",height:"300px",margin:"0 auto",display:"flex",alignItems:"stretch",justifyContent:"center",gap:"7px"});
  const stage=el("div",null,{flex:"1 1 auto",minWidth:"0",height:"300px",position:"relative",background:"#080a0d",border:"1px solid #262c35",borderRadius:"7px",overflow:"hidden",boxShadow:"inset 0 0 0 1px rgba(255,255,255,.015),0 8px 20px rgba(0,0,0,.28)"});
  const sourceVideo=node.__lmdSourceVideo||document.createElement("video");Object.assign(sourceVideo.style,{position:"absolute",inset:"0",width:"100%",height:"100%",objectFit:"contain",objectPosition:"center center",display:"none",background:"#080a0d"});sourceVideo.preload="metadata";sourceVideo.muted=true;sourceVideo.playsInline=true;node.__lmdSourceVideo=sourceVideo;
  const canvas=node.__lmdPreviewCanvas||el("canvas",null,{position:"absolute",inset:"0",width:"100%",height:"100%",display:"block",background:"#080a0d"});node.__lmdPreviewCanvas=canvas;stage.append(sourceVideo,canvas);
  const quality=el("div",null,{flex:"0 0 42px",display:"flex",flexDirection:"column",alignItems:"stretch",justifyContent:"center",gap:"5px"});quality.append(el("div","VIEW",{font:"800 7px system-ui",letterSpacing:".08em",color:styles.dim,textAlign:"center",marginBottom:"1px"}));for(const q of PREVIEW_QUALITY_MODES){const active=String(node.__lmdPreviewQuality)===q;const b=button(q==="Full"?"FULL":`${q}%`,()=>{node.__lmdPreviewQuality=q;resizePreviewCanvas(node,stage);render(node);},active,`Program Monitor display quality ${q}${q==="Full"?"":"%"}. Browser-side only: no extra preview files are created.`);Object.assign(b.style,{padding:"4px 2px",font:"800 8px system-ui",width:"42px",height:"28px"});quality.append(b);}stageRow.append(stage,quality);card.append(stageRow);
  renderTransport(node,doc,card);const meta=el("div",null,{height:"24px",display:"flex",alignItems:"center",justifyContent:"center",padding:"0 6px",boxSizing:"border-box",borderTop:`1px solid ${styles.border}`,marginTop:"3px"});const detail=el("div","",{font:"9px/1.3 system-ui",color:styles.dim,textAlign:"center",whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis",maxWidth:"90%"});node.__lmdPreviewDetail=detail;meta.append(detail);card.append(meta);host.append(card);
  requestAnimationFrame(()=>resizePreviewCanvas(node,stage));if(typeof ResizeObserver!=="undefined"){const ro=new ResizeObserver(()=>resizePreviewCanvas(node,stage));ro.observe(stage);node.__lmdResizeObservers=node.__lmdResizeObservers||[];node.__lmdResizeObservers.push(ro);}updatePlayheadUI(node,doc);
}
function renderCoreLayerInspector(node,doc,kind,host){
  const color=kind==="main"?"#8fc8e5":kind==="camera"?"#c1a1e8":"#8bd5a5",count=kind==="main"?(doc.shots||[]).length:kind==="camera"?(doc.camera_blocks||[]).length:(doc.audio_blocks||[]).length;
  host.append(el("div",`LAYER · ${coreTrackName(node,doc,kind)}`,{font:"800 11px system-ui",color,marginBottom:"7px"}));
  const info=grid("1fr 120px 90px");info.append(field("Layer name",input(coreTrackName(node,doc,kind),v=>{node.__lmdCoreTrackNames=node.__lmdCoreTrackNames||{};const clean=String(v||"").trim().slice(0,100);if(!clean||clean===coreTrackDefaultName(doc,kind))delete node.__lmdCoreTrackNames[kind];else node.__lmdCoreTrackNames[kind]=clean;saveDirectorUIPref(node,"core_track_names",{...node.__lmdCoreTrackNames});render(node);},"text")),field("Type",el("div",kind==="main"?"SYSTEM · BASE":kind==="camera"?"SYSTEM · CAMERA":"SYSTEM · AUDIO",{height:"28px",boxSizing:"border-box",padding:"6px",background:"#0f1115",border:`1px solid ${styles.border}`,borderRadius:"4px",font:"700 9px system-ui",color})),field(kind==="main"?"Clips":"Blocks",el("div",String(count),{height:"28px",boxSizing:"border-box",padding:"6px",background:"#0f1115",border:`1px solid ${styles.border}`,borderRadius:"4px",font:"700 9px system-ui"})));host.append(info);
  const state=el("div",null,{display:"flex",gap:"5px",marginTop:"8px",flexWrap:"wrap"});state.append(iconButton(coreTrackLocked(node,kind)?"▣":"□",coreTrackLocked(node,kind)?"Unlock layer editing":"Lock layer editing",()=>toggleCoreTrackLock(node,kind),coreTrackLocked(node,kind)));if(kind==="main")state.append(iconButton(doc.base_audio_muted?"🔇":"🔊",doc.base_audio_muted?"Unmute BASE embedded audio":"Mute BASE embedded audio",()=>{doc.base_audio_muted=!doc.base_audio_muted;commit(node,doc);},doc.base_audio_muted));if(kind==="audio")state.append(iconButton(doc.audio_track_muted?"🔇":"🔊",doc.audio_track_muted?"Unmute AUDIO monitor":"Mute AUDIO monitor",()=>{doc.audio_track_muted=!doc.audio_track_muted;commit(node,doc);},doc.audio_track_muted));host.append(state);
  const ops=el("div",null,{display:"flex",gap:"6px",marginTop:"10px",flexWrap:"wrap"});ops.append(button("RENAME",()=>renameCoreTrack(node,doc,kind)));if(kind==="camera")ops.append(button("+ BLOCK",()=>toolbarAction(node,doc,"camera_prompt")));if(kind==="audio")ops.append(button("+ BLOCK",()=>toolbarAction(node,doc,"sound_prompt")));if(kind==="main"&&String(doc.setup_timeline_mode)==="multiclip")ops.append(button("+ CLIP",()=>addMulticlipAtPlayhead(node,doc)));if(kind==="camera")ops.append(button("REMOVE LAYER",()=>{if(window.confirm(`Remove CAMERA layer and all ${doc.camera_blocks.length} block(s)?`)){doc.camera_blocks=[];doc.camera_track_enabled=false;node.__lmdSelection={track:"main",id:doc.shots[0]?.clip_id};commit(node,doc);}},false,"Remove all CAMERA blocks from this system layer"));if(kind==="audio")ops.append(button("REMOVE LAYER",()=>{if(window.confirm(`Remove AUDIO layer and all ${doc.audio_blocks.length} block(s)?`)){doc.audio_blocks=[];doc.audio_track_enabled=false;node.__lmdSelection={track:"main",id:doc.shots[0]?.clip_id};commit(node,doc);}},false,"Remove all AUDIO blocks from this system layer"));host.append(ops);
  host.append(el("div","This is a fixed Director system layer. It can be selected, renamed and locked; custom layers below support full duplicate/delete operations.",{font:"9px/1.35 system-ui",color:styles.dim,marginTop:"8px"}));
}
function renderLayerInspector(node,doc,track,host){const meta=EXTRA_TRACK_META[track.type]||{};host.append(el("div",`LAYER · ${meta.label||track.type}`,{font:"800 11px system-ui",color:meta.color||"#ddd",marginBottom:"7px"}));const info=grid("1fr 120px 90px");info.append(field("Layer name",input(track.name,v=>{track.name=String(v||"").slice(0,100);commit(node,doc,false);})),field("Type",el("div",String(meta.label||track.type).toUpperCase(),{height:"28px",boxSizing:"border-box",padding:"6px",background:"#0f1115",border:`1px solid ${styles.border}`,borderRadius:"4px",font:"700 9px system-ui",color:meta.color||styles.text})),field("Clips",el("div",String((track.clips||[]).length),{height:"28px",boxSizing:"border-box",padding:"6px",background:"#0f1115",border:`1px solid ${styles.border}`,borderRadius:"4px",font:"700 9px system-ui"})));host.append(info);if(["prompt","character","reference","video"].includes(track.type))host.append(field("PROMPT PRIORITY",selectOptions(PROMPT_GUIDANCE_OPTIONS,track.prompt_guidance_mode,v=>{track.prompt_guidance_mode=v;commit(node,doc);}),"Applies during clips on this layer. A MAIN clip with its own mode overrides this setting while it overlaps; this adds text guidance, not numeric attention."));
  const state=el("div",null,{display:"flex",gap:"5px",marginTop:"8px",flexWrap:"wrap"});state.append(iconButton(track.enabled===false?"○":"●",track.enabled===false?"Enable layer":"Disable layer",()=>{track.enabled=track.enabled===false;commit(node,doc);},track.enabled!==false),iconButton(track.locked?"▣":"□",track.locked?"Unlock layer":"Lock layer edits",()=>{track.locked=!track.locked;commit(node,doc);},track.locked));if(["audio","video","reference","character"].includes(track.type))state.append(iconButton(track.muted?"🔇":"🔊",track.muted?"Unmute layer monitor":"Mute layer monitor",()=>{track.muted=!track.muted;commit(node,doc);},track.muted));host.append(state);
  const ops=el("div",null,{display:"flex",gap:"6px",marginTop:"10px",flexWrap:"wrap"});ops.append(button("RENAME",()=>renameExtraTrack(node,doc,track)),button("DUPLICATE",()=>duplicateExtraTrack(node,doc,track)),button("+ CLIP",()=>addClipToExtraTrack(node,doc,track),false,"Add a clip to this layer at the playhead"),button("DELETE LAYER",()=>{if(window.confirm(`Delete layer “${track.name}” and all ${(track.clips||[]).length} clip(s)?`))deleteExtraTrack(node,doc,track);},false,"Delete this complete layer"));host.append(ops,el("div","Select the layer name at the left of the timeline to edit layer-level properties. Select a clip to edit its timing/content.",{font:"9px/1.35 system-ui",color:styles.dim,marginTop:"8px"}));}
function renderExtraInspector(node,doc,track,clip,host){const meta=EXTRA_TRACK_META[track.type]||{};host.append(el("div",`${meta.label||track.type} · ${track.name}`,{font:"800 11px system-ui",color:meta.color||"#ddd",marginBottom:"7px"}));const header=el("div",null,{display:"grid",gridTemplateColumns:"1fr 90px 90px",gap:"6px"});header.append(field("Clip name",input(clip.name,v=>{clip.name=v;commit(node,doc,false);})),field("Start",input(clip.start,v=>{clip.start=Math.max(0,finite(v,clip.start));commit(node,doc);},"number")),field("Duration",input(clip.duration,v=>{clip.duration=clamp(v,.25,600,clip.duration);commit(node,doc);},"number")));host.append(header);if(track.type==="refmod"){const refmods=doc.refmods||[];const rmOpts=[{value:"",label:"— choose RefMod —"},...refmods.map(r=>({value:r.refmod_id,label:`${r.name} (${r.mode})`}))];const rmRow=grid("1fr 1fr auto");rmRow.append(field("RefMod",selectOptions(rmOpts,clip.refmod_id||"",v=>{clip.refmod_id=v||null;commit(node,doc);})),field("Strength",input(clip.strength??1,v=>{clip.strength=clamp(v,0,1,1);commit(node,doc);},"number"),"0.0–1.0 blend weight"),iconButton("↔","Stretch this RefMod across the complete Director timeline",()=>{clip.start=0;clip.duration=Math.max(.25,totalDuration(doc));commit(node,doc);},false,34));host.append(rmRow);host.append(el("div","Reaches the render as an unlabelled native H3 reference, and only during this layer's interval. Use ↔ to cover the whole timeline (typical for a style reference). Several REFMOD layers stack in layer order. 1.0 = exact native reference.",{marginTop:"6px",padding:"6px 7px",border:`1px solid ${styles.border}`,borderRadius:"4px",background:"#11151a",color:styles.dim,font:"9px/1.35 system-ui"}));}
  if(track.type==="embedding"){
    if(node.__lmdEmbeddings===undefined&&!node.__lmdEmbeddingCatalogLoading){node.__lmdEmbeddingCatalogLoading=true;refreshEmbeddingCatalog(node).finally(()=>{node.__lmdEmbeddingCatalogLoading=false;});}
    const catalog=embeddingCatalog(node,clip.embedding_name);
    const options=[{value:"",label:catalog.length?"— choose H3 embedding —":"— no embeddings found —"},...catalog.map(v=>({value:v,label:v}))];
    const row=grid("1fr auto auto");
    row.append(field("H3 TEXT EMBEDDING",selectOptions(options,clip.embedding_name||"",v=>{clip.embedding_name=String(v||"");if(clip.embedding_name&&(/^EMBEDDING clip$/i.test(clip.name)||/^Layer \d+$/i.test(clip.name)))clip.name=clip.embedding_name.split("/").pop();commit(node,doc);}),node.__lmdEmbeddingCatalogError?`catalog: ${node.__lmdEmbeddingCatalogError}`:"ComfyUI/models/embeddings"),iconButton("↻","Refresh ComfyUI embedding catalog",()=>refreshEmbeddingCatalog(node),false,34),iconButton("↔","Stretch this embedding layer across the complete Director timeline",()=>{clip.start=0;clip.duration=Math.max(.25,totalDuration(doc));commit(node,doc);},false,34));
    host.append(row,el("div","Applies during this layer's timeline interval. Overlapping embeddings combine. One MAIN remains one sampler pass; no hidden split is created. Timing follows the 24 fps output and H3 temporal grid.",{marginTop:"6px",padding:"6px 7px",border:`1px solid ${styles.border}`,borderRadius:"4px",background:"#11151a",color:styles.dim,font:"9px/1.35 system-ui"}));
  } else if(["prompt","character","reference","video"].includes(track.type)){host.append(field(track.type==="character"?"Character direction":track.type==="reference"?"Reference direction":"Layer prompt",persistentTextarea(node,`layer:${track.type}:prompt`,clip.prompt,v=>{clip.prompt=v;commit(node,doc,false);},4)));}
  if(["character","reference","video","audio"].includes(track.type)){const kinds=track.type==="audio"?["Audio","Video"]:track.type==="video"?["Video"]:["Picture","Video"];const opts=[{value:"",label:"— choose media / subject —"},...doc.subjects.filter(s=>kinds.includes(s.kind)).map(s=>({value:s.subject_id,label:`${s.name} · ${runtimeToken(doc,s)}`}))];host.append(field("Media / Subject",selectOptions(opts,clip.subject_id||"",v=>{clip.subject_id=v||null;commit(node,doc);})));
  }
  if(track.type==="video"&&subjectById(doc,clip.subject_id)?.kind==="Video"&&subjectById(doc,clip.subject_id)?.media)host.append(field("VIDEO LAYER SCALE",videoReferenceScaleControl(clip.video_reference_scale,v=>{clip.video_reference_scale=v;commit(node,doc);}),"Scales this VIDEO clip’s source before H3 conditioning. Shared preprocessing uses the highest required scale for each video slot; output resolution and trim stay unchanged."));
  if(track.type==="character"){const shot=doc.shots.length===1?doc.shots[0]:null,source=shot?subjectById(doc,shot.media_subject_id):null,character=subjectById(doc,clip.subject_id),eps=1/Math.max(1,finite(doc.fps,24)),full=Boolean(shot&&clip.start<=eps&&(clip.start+clip.duration)>=shot.duration-eps),native=Boolean(track.enabled!==false&&shot&&source?.kind==="Video"&&source?.media&&character?.kind==="Picture"&&character?.media&&full);const info=el("div",native?"NATIVE CHARACTER REPLACE · Render uses BASE video as <Video 1>, this Picture as the replacement identity, and the BASE soundtrack as Audio 1.":"Native replace arms automatically for one MAIN shot when a Picture CHARACTER layer covers the complete BASE video shot.",{marginTop:"7px",padding:"6px 7px",border:`1px solid ${native?"#4c795d":styles.border}`,borderRadius:"4px",background:native?"rgba(45,92,62,.18)":"#11151a",color:native?"#9fe0b7":styles.dim,font:"9px/1.35 system-ui"});host.append(info);}
  const controls=el("div",null,{display:"flex",gap:"5px",marginTop:"8px",flexWrap:"wrap"});controls.append(iconButton(track.enabled===false?"○":"●",track.enabled===false?"Enable this track":"Disable this track",()=>{track.enabled=track.enabled===false;commit(node,doc);},track.enabled!==false),iconButton(track.locked?"▣":"□",track.locked?"Unlock this track":"Lock this track",()=>{track.locked=!track.locked;commit(node,doc);},track.locked));if(["audio","video","reference","character"].includes(track.type))controls.append(iconButton(track.muted?"🔇":"🔊",track.muted?"Unmute this track for monitor playback":"Mute this track for monitor playback",()=>{track.muted=!track.muted;commit(node,doc);},track.muted));controls.append(iconButton("⌫","Delete selected clip",()=>{track.clips=track.clips.filter(c=>c.clip_id!==clip.clip_id);node.__lmdSelection={track:"main",id:doc.shots[0]?.clip_id};commit(node,doc);}),iconButton("🗑","Delete entire track",()=>deleteExtraTrack(node,doc,track)));host.append(controls);
}
function renderMergedGroupInspector(node,doc,group,host){
  const indices=mergedGroupIndices(doc,group),members=indices.map(index=>doc.shots[index]);if(members.length<2)return;
  const first=indices[0],last=indices.at(-1),start=shotStart(doc,first),end=shotStart(doc,last)+finite(members.at(-1)?.duration,0),allEnabled=mergedGroupRunEnabled(doc,group),anyEnabled=members.some(shot=>shot.run_enabled!==false),mixed=anyEnabled&&!allEnabled;
  host.append(el("div",`MERGED MAIN · ${members.length} clips`,{font:"800 11px system-ui",color:"#9bd4f1",marginBottom:"6px"}),el("div",group.name,{font:"700 10px system-ui",color:styles.text,marginBottom:"4px"}),el("div",`${start.toFixed(2)}–${end.toFixed(2)}s · ${members.map(shot=>shot.name||"Clip").join(" + ")}`,{font:"9px/1.35 system-ui",color:styles.dim,marginBottom:"7px"}));
  host.append(checkbox(allEnabled?"Regenerate merged clip on RUN":"Reuse merged clip TAKEs on RUN",allEnabled,enabled=>setMergedGroupRunEnabled(node,doc,group,enabled),{title:"This reversible group displays as one MAIN clip. Its approved component TAKEs remain intact and are reused when disabled."}));
  const memberSwitches=el("div",null,{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(180px,1fr))",gap:"3px",marginTop:"5px"});
  members.forEach((member,memberIndex)=>memberSwitches.append(checkbox(`${indices[memberIndex]+1}. ${member.name||"Clip"} · RUN`,member.run_enabled!==false,enabled=>setMainClipRunEnabled(node,doc,member,enabled),{title:"Choose whether this component clip is sampled by ordinary RUN."})));
  host.append(el("div","COMPONENT CLIP SWITCHES",{font:"800 9px system-ui",letterSpacing:".06em",color:styles.dim,marginTop:"6px",marginBottom:"3px"}),memberSwitches);
  if(mixed)host.append(el("div","Some component clips are disabled. Turning this group on/off applies to every component; restoring boundaries brings back their individual switches.",{font:"9px/1.35 system-ui",color:"#eab779",marginTop:"5px"}));
  host.append(el("div","Merge keeps the rendered TAKEs and does not run diffusion. Undo restores the previous clips and switches.",{font:"9px/1.35 system-ui",color:styles.dim,marginTop:"6px"}),button("Restore original clip boundaries",()=>restoreMergedGroupBoundaries(node,doc,group),false,"Split this reversible merged clip back into its original MAIN clips."));
}
function renderInspector(node,doc,host){const s=selection(node);const panel=el("div",null,{background:styles.panel,border:`1px solid ${styles.border}`,borderRadius:"5px",padding:"8px",minHeight:"0",height:"max-content",alignSelf:"start"});if(s.track==="main"){const b=doc.shots.find(x=>x.clip_id===s.id)||doc.shots[0];if(b){node.__lmdSelection={track:"main",id:b.clip_id};renderShotInspector(node,doc,b,panel);}}else if(s.track==="main_group"){const group=mergedGroupById(doc,s.id);if(group)renderMergedGroupInspector(node,doc,group,panel);}else if(s.track==="camera"){const b=doc.camera_blocks.find(x=>x.block_id===s.id);if(b)renderCameraInspector(node,doc,b,panel);}else if(s.track==="audio"){const b=doc.audio_blocks.find(x=>x.block_id===s.id);if(b)renderAudioInspector(node,doc,b,panel);}else if(String(s.track||"").startsWith("layer:")){const tid=String(s.track).slice(6);if(["main","camera","audio"].includes(tid))renderCoreLayerInspector(node,doc,tid,panel);else{const tr=(doc.extra_tracks||[]).find(t=>t.track_id===tid);if(tr)renderLayerInspector(node,doc,tr,panel);}}else if(String(s.track||"").startsWith("extra:")){const tid=String(s.track).slice(6),tr=(doc.extra_tracks||[]).find(t=>t.track_id===tid),c=tr?.clips?.find(x=>x.clip_id===s.id);if(tr&&c)renderExtraInspector(node,doc,tr,c,panel);}host.append(panel);}

function mediaCardPreview(node,doc,subject){
  const p=el("div",null,{position:"relative",height:"76px",borderRadius:"4px",background:"#0c0e12",border:`1px solid ${styles.border}`,overflow:"hidden",marginBottom:"6px",cursor:"grab"});
  if(subject.media)decorateMedia(p,subject.media,{audioVisual:true});else p.append(el("div","DROP MEDIA HERE",{padding:"29px 8px",textAlign:"center",font:"9px system-ui",color:styles.dim}));
  p.draggable=true;p.title="Drag this asset onto MAIN/AUDIO, or drop a replacement file here";
  p.ondragstart=e=>{e.stopPropagation();try{e.dataTransfer.setData(DROP_MIME,subject.subject_id);e.dataTransfer.effectAllowed="copy";}catch(_){}};
  p.ondragenter=e=>{if(!droppedFiles(e).length)return;e.preventDefault();e.stopPropagation();showDropHint(p,"REPLACE MEDIA");};
  p.ondragover=e=>{if(!droppedFiles(e).length)return;e.preventDefault();e.stopPropagation();if(e.dataTransfer)e.dataTransfer.dropEffect="copy";showDropHint(p,"REPLACE MEDIA");};
  p.ondragleave=e=>{if(e.relatedTarget&&p.contains(e.relatedTarget))return;clearDropHint(p);};
  p.ondrop=async e=>{e.preventDefault();e.stopPropagation();clearDropHint(p);const file=droppedFiles(e)[0];if(!file)return;const kind=fileKind(file);const allowed=subject.kind==="Picture"?["image"]:subject.kind==="Video"?["video"]:["audio","video"];if(!allowed.includes(kind)){alert(`Drop ${subject.kind==="Picture"?"an image":subject.kind==="Video"?"a video":"audio or a video soundtrack"} on this ${subject.kind} card.`);return;}try{subject.media=await upload(kind,file);if(!subject.name||/^Subject \d+$/.test(subject.name))subject.name=file.name;commit(node,doc);}catch(err){alert(`LongMedia Director: ${err?.message||err}`);}};
  return p;
}
function renderAssets(node,doc,host){host.append(el("div","WHO & WHAT",{font:"800 11px system-ui",letterSpacing:".06em",marginBottom:"6px"}),el("div","Drop image / video / audio files here. Then drag any thumbnail straight onto MAIN or AUDIO. Media is part of the single Director contract sent to Setup through the director socket.",{font:"9px/1.35 system-ui",color:styles.dim,marginBottom:"7px"}));
  installAssetDrop(host,node,doc);
  const tools=el("div",null,{display:"flex",gap:"4px",flexWrap:"wrap",marginBottom:"8px"});for(const [label,k,title] of [["▧","image","Import picture"],["▶","video","Import video"],["♪","audio","Import audio"]])tools.append(iconButton(label,title,async()=>{const f=await pick(k);if(!f)return;try{const r=await upload(k,f);await addMediaSubject(node,doc,k,r,f.name);commit(node,doc);}catch(e){alert(e?.message||e);}}));host.append(tools);
  const list=el("div",null,{display:"grid",gap:"7px"});doc.subjects.forEach((s,i)=>{const card=el("div",null,{background:styles.panel2,border:`1px solid ${styles.border}`,borderRadius:"5px",padding:"7px"});card.append(mediaCardPreview(node,doc,s));const head=el("div",null,{display:"flex",justifyContent:"space-between",gap:"5px",marginBottom:"5px"});head.append(el("span",`${runtimeToken(doc,s)} · ${s.kind}`,{font:"700 9px system-ui",color:s.kind==="Audio"?"#95dbaa":s.kind==="Video"?"#f0b477":"#9bd4f1"}),button("×",()=>{const sid=s.subject_id;doc.subjects.splice(i,1);for(const shot of doc.shots){shot.subjects=shot.subjects.filter(x=>x!==sid);if(shot.media_subject_id===sid)shot.media_subject_id=null;}for(const a of doc.audio_blocks)if(a.subject_id===sid)a.subject_id=null;commit(node,doc);}));card.append(head,field("Name",input(s.name,v=>{s.name=v;commit(node,doc,false);})),field("Description",textarea(s.description,v=>{s.description=v;commit(node,doc,false);},2)),field("Retention",select(s.kind==="Audio"?["fully_copy","partially_copy","reference","weak_reference"]:["fully_preserved","partially_preserved","attribute_transfer","weak_reference"],s.retention,v=>{s.retention=v;commit(node,doc,false);})));if(s.media)card.append(el("div",mediaPath(s.media),{font:"8px system-ui",color:styles.dim,marginTop:"4px",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}));list.append(card);});host.append(list);}

function directorResolutionPreview(doc){
  const source=String(doc?.resolution_source||"base_layer");let rw=16,rh=9,basis="16:9";
  if(source==="custom"){const m=String(doc?.resolution||"1920x1080").match(/^(\d+)x(\d+)$/i);if(m){rw=Math.max(1,+m[1]);rh=Math.max(1,+m[2]);basis=`${rw}:${rh}`;}}
  else if(source==="base_layer"){const byId=new Map((doc?.subjects||[]).map(s=>[s.subject_id,s]));let media=null;for(const shot of doc?.shots||[]){const m=byId.get(shot.media_subject_id)?.media;if(m?.width>0&&m?.height>0){media=m;break;}}if(!media){media=(doc?.subjects||[]).map(s=>s.media).find(m=>m?.width>0&&m?.height>0)||null;}if(media){rw=+media.width;rh=+media.height;basis=`${rw}:${rh}`;}else basis="fallback 16:9";}
  else if(source==="2.39:1"){rw=239;rh=100;basis=source;}else{const m=source.match(/^([\d.]+):([\d.]+)$/);if(m){rw=+m[1];rh=+m[2];basis=source;}}
  const mp=clamp(doc?.megapixel,.1,8,.8),ratio=Math.max(.0001,rw/rh),pixels=mp*1e6;let w=Math.sqrt(pixels*ratio),h=pixels/w;w=Math.max(32,Math.round(w/32)*32);h=Math.max(32,Math.round(h/32)*32);return {width:w,height:h,basis,actual:(w*h/1e6)};
}

// ── RefMod backend routes ────────────────────────────────────────────────
async function refmodList(node,doc){
  // A failed listing must never look like an empty library: SYNC reconciles the
  // document against this result and would otherwise delete live entries (or
  // silently no-op) whenever the backend errors out on a given install.
  let r;
  try{r=await api.fetchApi('/longmedia/refmods/list',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project_id:doc.project_id})});}
  catch(err){throw new Error(`RefMod library request failed: ${err?.message||err}`);}
  const raw=await r.text();
  let d={};
  try{d=raw?JSON.parse(raw):{};}catch(_){throw new Error(`RefMod library returned a non-JSON response (HTTP ${r.status}).`);}
  if(!r.ok)throw new Error(d.error||`RefMod library request failed (HTTP ${r.status}).`);
  const folders=Array.isArray(d.folders)?d.folders:["Unsorted"];
  node.__lmdRefmodFolders=folders;
  return {refmods:Array.isArray(d.refmods)?d.refmods.slice(0,REFMOD_MAX):[],folders};
}
function reconcileRefmodLibrary(local,remote){
  // Library listing is authoritative for anything saved to disk. Unsaved drafts
  // (no `path`) survive so a half-finished CREATE is never destroyed.
  const current=Array.isArray(local)?local:[];
  const library=Array.isArray(remote)?remote:[];
  const remoteIds=new Set(library.map(r=>String(r.refmod_id||"").trim()).filter(Boolean));
  const isSaved=r=>Boolean(String(r?.path||"").trim());
  const keptDrafts=current.filter(r=>!isSaved(r));
  const removed=current.filter(r=>isSaved(r)&&!remoteIds.has(String(r.refmod_id||"").trim()));
  const refmods=[];
  let added=0,updated=0;
  for(const r of library){
    const rid=String(r.refmod_id||"").trim();
    const existing=current.find(x=>String(x.refmod_id||"").trim()===rid);
    if(existing){Object.assign(existing,{path:r.path,folder:r.folder,state:r.state,kind:r.kind,format_version:r.format_version,members:r.members});refmods.push(existing);updated++;}
    else{refmods.push(normalizeRefmod(r));added++;}
  }
  for(const draft of keptDrafts){
    const did=String(draft.refmod_id||"").trim();
    if(!refmods.some(x=>String(x.refmod_id||"").trim()===did))refmods.push(draft);
  }
  return {refmods,added,updated,removed,drafts:keptDrafts,removedIds:removed.map(r=>String(r.refmod_id||"").trim())};
}
function refmodTimelineReferrers(doc,ids){
  const gone=new Set((ids||[]).map(String));
  let count=0;
  for(const track of doc?.extra_tracks||[])for(const clip of track.clips||[])if(gone.has(String(clip?.refmod_id||"")))count++;
  return count;
}
async function refmodSave(node,doc,record){const r=await api.fetchApi('/longmedia/refmods/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project_id:doc.project_id,refmod:record})});const d=await r.json();if(!r.ok)throw new Error(d.error||`HTTP ${r.status}`);return d.refmod||record;}
async function refmodEncode(node,doc,record){const r=await api.fetchApi('/longmedia/refmods/encode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project_id:doc.project_id,refmod:record,director_json:String(widget(node,"director_json")?.value||"{}")})});const d=await r.json();if(!r.ok)throw new Error(d.error||`HTTP ${r.status}`);return d;}
async function importRefmodFile(node,doc,file,{folder="Unsorted",track=null,time=null}={}){
  if(track?.locked)throw new Error("Unlock the RefMod layer before dropping a file on it.");
  const uploaded=await upload("refmod",file);
  const source=`input/${uploaded.subfolder?`${uploaded.subfolder}/`:""}${uploaded.filename}`;
  const authored={...defaultRefmod(),name:file.name.replace(/\.safetensors$/i,""),source,folder,state:"ready",strength:1};
  const saved=applyRefmodServerState(authored,await refmodSave(node,doc,authored));
  doc.refmods=[...(doc.refmods||[]).filter(x=>x.refmod_id!==saved.refmod_id),saved];
  let target=track;
  if(!target){target=(doc.extra_tracks||[]).find(t=>t.type==="refmod"&&!t.locked);if(!target){target=defaultExtraTrack("refmod");doc.extra_tracks.push(target);}}
  if(target.locked)throw new Error("Unlock the RefMod layer before dropping a file on it.");
  const at=Math.max(0,finite(time,node.__lmdPlayhead||0));
  const clip=defaultExtraClip(target,at,Math.max(.25,sceneInfoAt(doc,at)?.shot?.duration||5));
  clip.refmod_id=saved.refmod_id;clip.strength=1;target.clips.push(clip);
  node.__lmdSelection={track:`extra:${target.track_id}`,id:clip.clip_id};node.__lmdRefmodSelected=saved.refmod_id;
  commit(node,doc);
}
async function createRefmodFolder(node,doc){const raw=window.prompt("New RefMod folder name");if(raw==null||!raw.trim())return;const name=raw.trim().replace(/[\\\\/]+/g,"-").slice(0,80);try{const r=await api.fetchApi('/longmedia/refmods/folder',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project_id:doc.project_id,name})});const d=await r.json();if(!r.ok)throw new Error(d.error||`HTTP ${r.status}`);node.__lmdRefmodFolder=d.name;node.__lmdRefmodFolders=[...new Set([...(node.__lmdRefmodFolders||["Unsorted"]),d.name])];render(node);}catch(err){alert(`Create folder failed: ${err?.message||err}`);}}
async function refmodDelete(node,doc,refmodId){const r=await api.fetchApi('/longmedia/refmods/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project_id:doc.project_id,refmod_id:refmodId})});if(!r.ok){const d=await r.json();throw new Error(d.error||`HTTP ${r.status}`);}return true;}
async function refmodFromTake(node,doc,shot,take){const r=await api.fetchApi('/longmedia/refmods/from_take',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project_id:doc.project_id,clip_id:shot.clip_id,revision:String(take.revision||"")})});const d=await r.json();if(!r.ok)throw new Error(d.error||`HTTP ${r.status}`);return d.refmod||null;}
// A library RefMod conditions nothing until a clip on the REFMOD layer points at
// it. This is the bridge from "encoded" to "affects the render".
function placeRefmodOnTimeline(node,doc,refmod){
  if(!refmod||!refmod.refmod_id)throw new Error("Select a RefMod first.");
  let target=(doc.extra_tracks||[]).find(t=>t.type==="refmod"&&!t.locked);
  if(!target){
    const locked=(doc.extra_tracks||[]).find(t=>t.type==="refmod"&&t.locked);
    if(locked)throw new Error("Unlock the REFMOD layer before placing on it.");
    target=defaultExtraTrack("refmod");
    doc.extra_tracks.push(target);
  }
  const at=Math.max(0,finite(node.__lmdPlayhead,0));
  const clip=defaultExtraClip(target,at,Math.max(.25,sceneInfoAt(doc,at)?.shot?.duration||5));
  clip.name=String(refmod.name||clip.name).slice(0,120);
  clip.refmod_id=refmod.refmod_id;
  clip.strength=clamp(refmod.strength,0,1,1);
  target.clips.push(clip);
  node.__lmdSelection={track:`extra:${target.track_id}`,id:clip.clip_id};
  node.__lmdRefmodSelected=refmod.refmod_id;
  return clip;
}

function refmodSourceKind(file){const ext=String(file?.name||"").split(".").pop().toLowerCase();if(String(file?.type||"").startsWith("video/")||["mp4","mov","mkv","webm","avi","m4v"].includes(ext))return "video";if(String(file?.type||"").startsWith("audio/")||["wav","mp3","flac","m4a","ogg","aac"].includes(ext))return "audio";return "image";}
function refmodCropForAspect(source,ratio,offsetX=.5,offsetY=.5){const w=Math.max(1,finite(source.width,1)),h=Math.max(1,finite(source.height,1)),current=w/h;if(!Number.isFinite(ratio)||ratio<=0||Math.abs(current-ratio)<.005)return null;let cw=1,ch=1;if(current>ratio)cw=ratio/current;else ch=current/ratio;return {x:(1-cw)*clamp(offsetX,0,1,.5),y:(1-ch)*clamp(offsetY,0,1,.5),width:cw,height:ch,ratio};}
function refmodEstimatedTokens(source,record){if(!source||!source.width||!source.height||source.kind==="audio")return 0;const crop=source.crop||null,w=Math.max(1,Math.round(source.width*(crop?.width||1))),h=Math.max(1,Math.round(source.height*(crop?.height||1))),scale=Math.min(1,finite(record.resolution_short_edge,1024)/Math.min(w,h)),tw=Math.max(32,Math.round(w*scale/32)*32),th=Math.max(32,Math.round(h*scale/32)*32),lh=Math.max(1,Math.floor(th/16)),lw=Math.max(1,Math.floor(tw/16));let t=1;if(source.kind==="video"){const raw=Math.max(1,Math.min(Math.trunc(finite(source.frameEstimate,record.clip_frames)),Math.trunc(finite(record.clip_frames,16))));const frames=raw<=1?1:Math.max(1,Math.floor((raw-1)/4)*4+1);t=Math.max(1,Math.floor((frames-1)/4)+1);}let per=(lh>>1)*(lw>>1);if(String(record.mode)==="COMPRESSED"){const edge=Math.max(2,Math.trunc(finite(record.grid_long_edge,16))),aspect=h/w;let ph,pw;if(aspect>=1){ph=edge;pw=Math.max(2,Math.round((edge/aspect)/2)*2);}else{pw=edge;ph=Math.max(2,Math.round((edge*aspect)/2)*2);}ph=Math.min(ph,lh-(lh%2));pw=Math.min(pw,lw-(lw%2));per=Math.max(1,ph>>1)*Math.max(1,pw>>1);t=source.kind==="video"?Math.min(t,Math.max(1,Math.trunc(finite(record.clip_frames,16)))):1;}let tokens=t*per;const cap=Math.max(0,Math.trunc(finite(record.max_tokens,5120)));if(cap>0&&per<=cap)tokens=Math.min(tokens,Math.floor(cap/per)*per);return tokens;}
async function refmodResponseJson(response,context){const raw=await response.text();let data;try{data=raw?JSON.parse(raw):{};}catch(_){const excerpt=String(raw||"<empty response>").replace(/\s+/g," ").trim().slice(0,240);const hint=response.status===404?"The route is not active in this ComfyUI process; restart ComfyUI, then refresh the browser.":excerpt;throw new Error(`${context} returned HTTP ${response.status} ${response.statusText||""}: ${hint}`.trim());}if(!response.ok){if(response.status===404)throw new Error(`${context}: HTTP 404. The route is not active in this ComfyUI process; restart ComfyUI, then refresh the browser.`);throw new Error(data.error||`${context} failed with HTTP ${response.status}`);}return data;}
async function uploadRefmodSource(file){const body=new FormData();body.append("file",file,file.name);const response=await api.fetchApi("/longmedia/refmods/source_upload",{method:"POST",body});return refmodResponseJson(response,`Upload ${file.name}`);}
function closeRefmodCreator(node){const state=node?.__lmdRefmodComposer;if(!state)return;window.removeEventListener("paste",state.onPaste,true);for(const source of state.sources||[]){if(source._probe){source._probe.onloadedmetadata=null;source._probe.onseeked=null;source._probe.onerror=null;source._probe.removeAttribute("src");source._probe.load();source._probe=null;}if(source.url)URL.revokeObjectURL(source.url);}state.overlay?.remove();node.__lmdRefmodComposer=null;}
function openRefmodCreator(node,doc){if(node.__lmdRefmodComposer){node.__lmdRefmodComposer.overlay?.focus();return;}const record={...defaultRefmod(),name:"New RefMod",folder:node.__lmdRefmodFolder||"Unsorted"},state={record,sources:[],busy:false,overlay:null};node.__lmdRefmodComposer=state;
  const overlay=el("div",null,{position:"fixed",inset:"0",zIndex:"2147482500",display:"grid",placeItems:"center",padding:"18px",background:"rgba(4,6,9,.78)"});overlay.tabIndex=-1;state.overlay=overlay;
  const dialog=el("section",null,{width:"min(940px,96vw)",maxHeight:"92vh",display:"flex",flexDirection:"column",background:"#171b21",border:`1px solid ${styles.border}`,borderRadius:"9px",boxShadow:"0 18px 70px rgba(0,0,0,.62)",color:styles.text,font:"11px system-ui",overflow:"hidden"});
  const header=el("div",null,{display:"flex",alignItems:"center",justifyContent:"space-between",gap:"12px",padding:"12px 14px",borderBottom:`1px solid ${styles.border}`});header.append(el("div","CREATE REFMOD · SOURCES",{font:"850 12px system-ui",letterSpacing:".06em",color:"#e8a952"}));header.append(iconButton("×","Close RefMod source editor",()=>closeRefmodCreator(node),false,32));
  const body=el("div",null,{padding:"12px 14px",overflowY:"auto",display:"grid",gap:"10px"});
  const addInput=el("input",null,{display:"none"});addInput.type="file";addInput.multiple=true;addInput.accept="image/*,video/*,audio/*";
  const addFiles=files=>{for(const file of Array.from(files||[])){if(state.sources.length>=128){alert("A RefMod can contain at most 128 sources.");break;}const kind=refmodSourceKind(file),mime=String(file?.type||"");if(!file||(!mime.startsWith("image/")&&!mime.startsWith("video/")&&!mime.startsWith("audio/")&&!/\.(png|jpe?g|webp|bmp|gif|tiff?)$/i.test(file.name||"")&&!/\.(mp4|mov|mkv|webm|avi|m4v|wav|mp3|flac|m4a|ogg|aac)$/i.test(file.name||"")))continue;const source={key:id("source"),file,name:file.name,kind,selected:true,url:URL.createObjectURL(file),width:0,height:0,duration:0,frames:0,crop:null};state.sources.push(source);if(kind==="image"){const image=new Image();image.onload=()=>{source.width=image.naturalWidth;source.height=image.naturalHeight;refresh();};image.src=source.url;}else if(kind==="video"){const video=document.createElement("video");source._probe=video;video.preload="metadata";video.muted=true;video.onloadedmetadata=()=>{source.width=video.videoWidth;source.height=video.videoHeight;source.duration=Number.isFinite(video.duration)?video.duration:0;source.frameEstimate=Math.max(1,Math.round(source.duration*24));source.frames=source.frameEstimate;refresh();try{video.currentTime=Math.min(.1,Math.max(0,source.duration-.01));}catch(_){}};video.onseeked=()=>{if(source.poster||!video.videoWidth||!video.videoHeight)return;try{const canvas=document.createElement("canvas");canvas.width=video.videoWidth;canvas.height=video.videoHeight;canvas.getContext("2d")?.drawImage(video,0,0);source.poster=canvas.toDataURL("image/jpeg",.78);refresh();}catch(_){}};video.onerror=()=>{source.metadataError="Could not read video metadata";refresh();};video.src=source.url;}else{const audio=document.createElement("audio");source._probe=audio;audio.preload="metadata";audio.onloadedmetadata=()=>{source.duration=Number.isFinite(audio.duration)?audio.duration:0;refresh();};audio.onerror=()=>{source.metadataError="Could not read audio metadata";refresh();};audio.src=source.url;}}refresh();};
  const topActions=el("div",null,{display:"flex",alignItems:"center",gap:"7px",flexWrap:"wrap"});const pickButton=button("+ ADD MEDIA",()=>addInput.click(),true,"Add images, videos or audio files from disk.");const pasteButton=button("PASTE IMAGE",()=>{navigator.clipboard?.read?.().then(async items=>{for(const item of items){const type=item.types.find(t=>t.startsWith("image/"));if(type){const blob=await item.getType(type);addFiles([new File([blob],`Clipboard ${Date.now()}.${type.split("/")[1]==="jpeg"?"jpg":type.split("/")[1]}`,{type})]);return;}}}).catch(()=>{});},false,"Paste an image from the clipboard (Ctrl/Cmd+V also works while this window is open).");topActions.append(pickButton,pasteButton,el("span","Drop files here · up to 128 sources",{fontSize:"9px",color:styles.dim}));
  const nameField=input(record.name,value=>{record.name=value;}),modeField=selectOptions([{value:"FULL",label:"Full Reference"},{value:"COMPRESSED",label:"Compressed Reference"}],record.mode,value=>{record.mode=value;refresh();});const conceptField=selectOptions(REFMOD_CONCEPT_TYPES.map(value=>({value,label:value.replaceAll("_"," ")})),record.concept_type,value=>{record.concept_type=value;});
  const resolutionField=input(record.resolution_short_edge,value=>{record.resolution_short_edge=Math.max(256,Math.min(2048,Math.round(finite(value,1024)/64)*64));refresh();},"number");resolutionField.min="256";resolutionField.max="2048";resolutionField.step="64";
  const tokensField=input(record.max_tokens,value=>{record.max_tokens=Math.max(0,Math.trunc(finite(value,5120)));refresh();},"number");tokensField.min="0";tokensField.step="256";
  const framesField=input(record.clip_frames,value=>{record.clip_frames=Math.max(1,Math.trunc(finite(value,16)));refresh();},"number");framesField.min="1";framesField.step="1";
  const gridField=input(record.grid_long_edge,value=>{record.grid_long_edge=Math.max(2,Math.min(64,Math.round(finite(value,16)/2)*2));refresh();},"number");gridField.min="2";gridField.max="64";gridField.step="2";const refinementField=input(record.refinement_steps,value=>{record.refinement_steps=Math.max(0,Math.min(2000,Math.trunc(finite(value,0))));},"number");refinementField.min="0";refinementField.max="2000";refinementField.step="10";
  const settings=el("div",null,{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(155px,1fr))",gap:"8px"});settings.append(field("RefMod name",nameField),field("Mode",modeField),field("Concept type",conceptField),field("Resolution · short edge",resolutionField),field("Visual max tokens / source",tokensField),field("Video frames",framesField),field("Compressed grid edge",gridField),field("Refinement steps",refinementField,"Compressed mode only. 0 disables latent reconstruction refinement; 150 is the upstream experimental style starting point."));
  const list=el("div",null,{display:"grid",gap:"7px",maxHeight:"46vh",overflowY:"auto",paddingRight:"3px"});
  const total=el("div","0 selected · estimated visual tokens: 0",{font:"700 10px system-ui",color:"#e8a952",fontVariantNumeric:"tabular-nums"});
  const footer=el("div",null,{display:"flex",justifyContent:"space-between",alignItems:"center",gap:"10px",padding:"10px 14px",borderTop:`1px solid ${styles.border}`});const status=el("div","Images use exact pixel geometry; video token counts are estimates until VAE encoding.",{font:"9px/1.35 system-ui",color:styles.dim});const actions=el("div",null,{display:"flex",gap:"7px"});const cancel=button("CANCEL",()=>closeRefmodCreator(node));const create=button("ENCODE & CREATE",()=>state.completed?closeRefmodCreator(node):encodeSources(),true,"Encode selected sources with the chosen H3 VAEs and save them as one RefMod bundle.");actions.append(cancel,create);footer.append(status,actions);
  const refresh=()=>{list.replaceChildren();refinementField.disabled=record.mode!=="COMPRESSED";const selected=state.sources.filter(s=>s.selected),sum=selected.reduce((n,s)=>n+(Number.isFinite(s.actualTokens)?s.actualTokens:refmodEstimatedTokens(s,record)),0);total.textContent=`${selected.length} selected / ${state.sources.length} sources · ${state.completed?"encoded":"estimated"} visual tokens: ${sum.toLocaleString()}`;
    if(!state.sources.length){const empty=el("div","Drop images, videos or audio here, or use Add Media. Paste copied images with Ctrl/Cmd+V.",{minHeight:"115px",display:"grid",placeItems:"center",textAlign:"center",padding:"20px",border:`1px dashed #4a5361`,borderRadius:"6px",color:styles.dim,font:"10px/1.5 system-ui"});list.append(empty);}
    for(const source of state.sources){const row=el("div",null,{display:"grid",gridTemplateColumns:"22px 158px minmax(0,1fr) auto 30px",gap:"9px",alignItems:"center",padding:"8px",background:"#1c2027",border:`1px solid ${styles.border}`,borderRadius:"6px"});const check=el("input",null,{width:"15px",height:"15px",accentColor:"#4c9cff"});check.type="checkbox";check.checked=source.selected;check.onchange=()=>{source.selected=check.checked;refresh();};
      let preview;if(source.kind==="image"||source.kind==="video"&&source.poster){preview=el("img");preview.src=source.kind==="image"?source.url:source.poster;}else{preview=el("div",source.kind==="video"?"▶ VIDEO":source.kind==="audio"?"♫ AUDIO":"MEDIA",{display:"grid",placeItems:"center",color:"#aab4c1",font:"700 13px system-ui"});}Object.assign(preview.style,{width:"158px",height:"91px",objectFit:"contain",background:"#0e1014",borderRadius:"4px"});
      const info=el("div",null,{minWidth:"0",display:"grid",gap:"4px"});info.append(el("div",source.name,{font:"700 10px system-ui",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",color:styles.text}));let detail=source.metadataError|| (source.kind==="audio"?`Audio · ${source.duration?`${source.duration.toFixed(2)} s`:"reading metadata…"}`:`${source.kind.toUpperCase()} · ${source.width&&source.height?`${source.width} × ${source.height}`:"reading metadata…"}${source.kind==="video"?` · ${source.frames||"…"} source frames · ${(source.duration||0).toFixed(2)} s`:" · 1 frame"}`);info.append(el("div",detail,{font:"9px system-ui",color:source.metadataError?"#e58f8f":styles.dim}));const estimate=refmodEstimatedTokens(source,record);info.append(el("div",source.kind==="audio"?"Audio tokens are independent of visual budget":`${source.actualTokens!=null?"Encoded":"Estimated"} ${Number(source.actualTokens??estimate).toLocaleString()} visual tokens`,{font:"9px system-ui",color:"#c8a96a"}));
      const cropWrap=el("div",null,{display:"grid",gap:"4px",justifyItems:"end"});if(source.kind!=="audio"){const cropBtn=button(source.crop?"CROP SET":"CROP TO FIT…",()=>{source.showCrop=!source.showCrop;refresh();},Boolean(source.crop));cropWrap.append(cropBtn);if(source.showCrop){const cropBox=el("div",null,{display:"grid",gap:"4px",width:"210px",padding:"6px",background:"#11151a",border:`1px solid ${styles.border}`,borderRadius:"5px"});const ratioOptions=[{value:"original",label:"Keep original framing"},{value:"match",label:"Match first selected source"},{value:"1.7777778",label:"Crop to 16:9"},{value:"0.5625",label:"Crop to 9:16"},{value:"1",label:"Crop to 1:1"}];const ratioValue=source.crop?.ratioMode||source.cropRatioMode||"original";const ratioControl=selectOptions(ratioOptions,ratioValue,value=>{source.cropRatioMode=value;if(value==="original")source.crop=null;else{const first=state.sources.find(item=>item!==source&&item.selected&&item.width&&item.height);const ratio=value==="match"&&first?first.width/first.height:Number(value);source.crop=refmodCropForAspect(source,ratio,.5,.5);if(source.crop)source.crop.ratioMode=value;}refresh();});cropBox.append(field("Aspect",ratioControl));if(source.crop){const moveAxis=(axis,value)=>{const crop=source.crop||{},width=finite(crop.width,1),height=finite(crop.height,1),rangeX=Math.max(0,1-width),rangeY=Math.max(0,1-height);crop[axis]=Math.max(0,(axis==="x"?rangeX:rangeY)*finite(value,50)/100);source.crop=crop;};for(const [axis,label] of [["x","Horizontal position"],["y","Vertical position"]]){const range=input("",null,"range");range.min="0";range.max="100";range.value="50";range.oninput=()=>moveAxis(axis,range.value);range.onchange=refresh;cropBox.append(field(label,range));}}cropWrap.append(cropBox);}}
      const remove=iconButton("×","Remove this source from the RefMod draft",()=>{if(source._probe){source._probe.onloadedmetadata=null;source._probe.onseeked=null;source._probe.onerror=null;source._probe.removeAttribute("src");source._probe.load();source._probe=null;}URL.revokeObjectURL(source.url);state.sources=state.sources.filter(s=>s!==source);refresh();},false,30);row.append(check,preview,info,cropWrap,remove);list.append(row);}
    create.disabled=state.busy||(!state.completed&&!selected.length);create.textContent=state.completed?"DONE":"ENCODE & CREATE";create.style.opacity=create.disabled?".55":"1";};
  const onPaste=e=>{if(!node.__lmdRefmodComposer)return;const files=[];for(const item of e.clipboardData?.items||[])if(item.type.startsWith("image/")){const blob=item.getAsFile();if(blob)files.push(new File([blob],`Clipboard ${Date.now()}.${item.type.split("/")[1]==="jpeg"?"jpg":item.type.split("/")[1]}`,{type:item.type}));}if(files.length){e.preventDefault();e.stopPropagation();addFiles(files);}};state.onPaste=onPaste;window.addEventListener("paste",onPaste,true);
  const encodeSources=async()=>{const selected=state.sources.filter(s=>s.selected);if(!selected.length||state.completed)return;state.busy=true;create.disabled=true;status.textContent="Uploading sources and encoding H3 VAE latents…";try{const uploaded=[];for(const source of selected){const result=await uploadRefmodSource(source.file);source.uploadedName=result.name;uploaded.push({...source,filename:result.name,selected:true});}record.folder=node.__lmdRefmodFolder||record.folder||"Unsorted";record.sources=uploaded.map(s=>({source_id:s.key,name:s.name,filename:s.filename,kind:s.kind,crop:s.crop?{x:s.crop.x,y:s.crop.y,width:s.crop.width,height:s.crop.height}:null}));const response=await api.fetchApi("/longmedia/refmods/encode_sources",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({project_id:doc.project_id,refmod:record,sources:uploaded.map(s=>({source_id:s.key,filename:s.filename,name:s.name,kind:s.kind,crop:s.crop,selected:true})),video_vae_name:node.__lmdVideoVaeName||null,audio_vae_name:node.__lmdAudioVaeName||null})}),data=await refmodResponseJson(response,"RefMod encoding");const saved=normalizeRefmod(applyRefmodServerState(record,data.refmod||{}));const liveDoc=node.__lmdDoc||parseDoc(node);liveDoc.refmods=[...(liveDoc.refmods||[]).filter(item=>item.refmod_id!==saved.refmod_id),saved];node.__lmdRefmodSelected=saved.refmod_id;node.__lmdRefmodFolder=saved.folder||"Unsorted";commit(node,liveDoc);render(node);for(const member of data.source_members||[]){const source=state.sources.find(item=>item.key===member.source_id);if(source)source.actualTokens=(source.actualTokens||0)+Number(member.tokens||0);}state.completed=true;status.textContent=`Saved to RefMods · ${data.members||0} member(s). Counts below are exact VAE-encoded visual tokens.`;cancel.textContent="CLOSE";refresh();}catch(error){state.busy=false;console.error("RefMod source encode failed",error);status.textContent=`Encode failed: ${error?.message||error}`;create.disabled=false;alert(`Create RefMod failed: ${error?.message||error}`);}};
  addInput.onchange=()=>{addFiles(addInput.files);addInput.value="";};const dropZone=el("div",null,{display:"grid",gap:"9px"});dropZone.append(topActions,settings,total,list,addInput);dropZone.ondragover=e=>{e.preventDefault();e.stopPropagation();dropZone.style.outline="2px solid #4b83bb";};dropZone.ondragleave=e=>{e.stopPropagation();dropZone.style.outline="none";};dropZone.ondrop=e=>{e.preventDefault();e.stopPropagation();dropZone.style.outline="none";addFiles(e.dataTransfer?.files);};
  body.append(dropZone);dialog.append(header,body,footer);overlay.append(dialog);overlay.ondragover=e=>{if(Array.from(e.dataTransfer?.types||[]).includes("Files")){e.preventDefault();e.stopPropagation();}};overlay.ondrop=e=>{if(e.dataTransfer?.files?.length){e.preventDefault();e.stopPropagation();addFiles(e.dataTransfer.files);}};overlay.onpointerdown=e=>{if(e.target===overlay)closeRefmodCreator(node);};overlay.onkeydown=e=>{if(e.key==="Escape"){e.preventDefault();closeRefmodCreator(node);}};document.body.append(overlay);refresh();overlay.focus();}

// ── RefMod inspector / panel ─────────────────────────────────────────────
function refmodSourceOptions(node,doc){
  const out=[{value:"",label:"Select source…"}],characterIds=new Set(),referenceIds=new Set();
  for(const tr of doc.extra_tracks||[]){if(tr.type!=="character"&&tr.type!=="reference")continue;for(const c of tr.clips||[]){if(!c.subject_id)continue;(tr.type==="character"?characterIds:referenceIds).add(c.subject_id);}}
  for(const s of doc.subjects||[]){let role=s.kind==="Picture"?"Image":String(s.kind||"Reference");if(characterIds.has(s.subject_id))role="Character";else if(referenceIds.has(s.subject_id))role="Reference";out.push({value:`subject:${s.subject_id}`,label:`${role} · ${s.name||runtimeToken(doc,s)}`});}
  const shot=takeShotForSidebar(node,doc);if(shot)out.push({value:`clip:${shot.clip_id}`,label:`Selected timeline clip · ${shot.name||shot.clip_id}`});
  const status=node.__lmdTakeStatus?.clips?.[shot?.clip_id]||{},revision=shot?selectedTakeRevision(node,shot,status):null;if(shot&&revision)out.push({value:`take:${shot.clip_id}:${revision}`,label:`Selected TAKE · ${revision.slice(0,8)}`});
  if(!out.some(x=>x.value===String(node?.__lmdRefmodSource||""))&&node?.__lmdRefmodSource)out.push({value:String(node.__lmdRefmodSource),label:String(node.__lmdRefmodSource)});return out;
}
function renderRefmodInspector(node,doc,refmod,host){
  const details=el("details",null,{marginTop:"7px",borderTop:`1px solid ${styles.border}`,paddingTop:"7px"});
  details.open=node.__lmdRefmodOpen===refmod.refmod_id;
  details.ontoggle=()=>{if(details.open){node.__lmdRefmodOpen=refmod.refmod_id;node.__lmdRefmodSelected=refmod.refmod_id;}};
  const summary=el("summary","REFMOD · INSPECTOR",{listStyle:"none",cursor:"pointer",font:"800 10px system-ui",color:"#e8a952",padding:"2px 0 7px"});details.append(summary);
  const body=el("div",null,{display:"grid",gridTemplateColumns:"minmax(0,1fr) minmax(0,1fr)",gap:"7px",alignItems:"start"});
  const sourceOptions=refmodSourceOptions(node,doc);if(refmod.source&&!sourceOptions.some(x=>x.value===refmod.source))sourceOptions.push({value:refmod.source,label:refmod.source});
  const sourceSelect=selectOptions(sourceOptions,refmod.source,v=>{refmod.source=v;refmod.source_subject_id=v.startsWith("subject:")?v.slice(8):null;refmod.source_clip_id=v.startsWith("clip:")?v.slice(5):(v.startsWith("take:")?v.split(":")[1]:null);refmod.source_take_revision=v.startsWith("take:")?v.split(":").slice(2).join(":"):null;refmod.state="draft";commit(node,doc);});
  body.append(field("Name",input(refmod.name,v=>{refmod.name=v;commit(node,doc,false);})),field("Source",sourceSelect));
  body.append(field("Mode",selectOptions([{value:"FULL",label:"Full Reference"},{value:"COMPRESSED",label:"Compressed Reference"}],refmod.mode,v=>{refmod.mode=v;commit(node,doc);})),el("div","H3 EXTRACTION SETTINGS",{gridColumn:"1 / -1",marginTop:"3px",paddingTop:"6px",borderTop:`1px solid ${styles.border}`,font:"800 9px system-ui",letterSpacing:".06em",color:"#e8a952"}));
  const numberField=(label,value,min,max,step,onChange,title)=>{const control=input(value,v=>onChange(v),"number");control.min=String(min);control.max=String(max);control.step=String(step);return field(label,control,title);};
  body.append(numberField("Resolution (short edge)",refmod.resolution_short_edge,256,2048,64,v=>{refmod.resolution_short_edge=Math.max(256,Math.min(2048,Math.round(finite(v,1024)/64)*64));commit(node,doc);},"Downscale-only cap applied before VideoVAE encoding in Full and Compressed modes. Keeps source aspect ratio."),numberField("Visual max tokens",refmod.max_tokens,0,2147483647,256,v=>{refmod.max_tokens=Math.max(0,Math.trunc(finite(v,5120)));commit(node,doc);},"Visual token budget after encoding/compression. 0 disables the cap. Overflow drops/resamples temporal frames; spatial size is preserved."));
  body.append(numberField("Grid (long edge)",refmod.grid_long_edge,2,64,2,v=>{refmod.grid_long_edge=Math.max(2,Math.min(64,Math.round(finite(v,16)/2)*2));refmod.spatial_grid=`${refmod.grid_long_edge}x${refmod.grid_long_edge}`;commit(node,doc);},"Compressed mode only. The other grid edge is derived from the source aspect ratio; larger grids preserve more spatial detail."),numberField("Refinement steps",refmod.refinement_steps,0,2000,10,v=>{refmod.refinement_steps=Math.max(0,Math.min(2000,Math.trunc(finite(v,0))));commit(node,doc);},"Compressed mode only. Latent reconstruction refinement; 0 means average pooling only. Not identity strength."));
  body.append(numberField("Clip frames",refmod.clip_frames,1,2147483647,1,v=>{refmod.clip_frames=Math.max(1,Math.min(2147483647,Math.trunc(finite(v,16))));refmod.temporal_frames=refmod.clip_frames;commit(node,doc);},"Video only. Full mode samples up to this many source frames across the clip before encoding (rounded down to H3's 4k+1 frame grid); Compressed mode stores up to this many latent frames after encoding."),numberField("Voice seconds",refmod.voice_seconds,.025,600,.025,v=>{refmod.voice_seconds=Math.max(.025,Math.min(600,finite(v,30)));commit(node,doc);},"Audio only. Trims the source audio from its beginning before AudioVAE encoding."));
  body.append(el("div","LIBRARY & TIMELINE",{gridColumn:"1 / -1",marginTop:"3px",paddingTop:"6px",borderTop:`1px solid ${styles.border}`,font:"800 9px system-ui",letterSpacing:".06em",color:"#e8a952"}));
  body.append(field("Concept Type",select(REFMOD_CONCEPT_TYPES,refmod.concept_type,v=>{refmod.concept_type=v;commit(node,doc);}),"Director adds a broad prompt hint for this type; Character also balances identity-reference attention against longer visual continuation references on each active clip. It does not change the extracted latent."),field("Description",persistentTextarea(node,`refmod:${refmod.refmod_id}:desc`,refmod.description,v=>{refmod.description=v;commit(node,doc,false);},3)));
  const strengthValue=el("span",clamp(refmod.strength,0,1,1).toFixed(2),{font:"700 10px system-ui",fontVariantNumeric:"tabular-nums",color:"#e8a952"});
  const strengthTitle=el("div",null,{display:"flex",justifyContent:"space-between",alignItems:"center",gap:"8px",marginBottom:"3px"});strengthTitle.append(el("span","Strength",{fontWeight:"650",color:"#b7c0cc"}),strengthValue);
  const range=input("",null,"range");range.min="0";range.max="1";range.step="0.01";range.value=String(clamp(refmod.strength,0,1,1));Object.assign(range.style,{width:"100%",height:"22px",padding:"0",margin:"0",accentColor:"#e8a952",cursor:"pointer"});
  const strength=el("div",null,{gridColumn:"1 / -1",minWidth:"0",padding:"2px 0"});strength.append(strengthTitle,range,el("div","1.0 = exact native reference",{fontSize:"9px",opacity:".7",marginTop:"2px"}));
  range.oninput=()=>{refmod.strength=clamp(range.value,0,1,1);strengthValue.textContent=refmod.strength.toFixed(2);};
  range.onchange=()=>{refmod.strength=clamp(range.value,0,1,1);commit(node,doc);};body.append(strength);
  body.append(el("div","Resolution and Clip frames affect visual VAE input. Max tokens caps visual tokens after encoding; Grid and Refinement steps only affect Compressed mode. Voice seconds trims audio before its VAE. Concept Type adds broad prompt guidance; Description remains library metadata. Strength controls timeline attention, not extraction quality.",{gridColumn:"1 / -1",marginTop:"2px",padding:"6px 7px",border:`1px solid ${styles.border}`,borderRadius:"4px",background:"#11151a",color:styles.dim,font:"9px/1.35 system-ui"}));
  const save=button("SAVE",async()=>{try{Object.assign(refmod,applyRefmodServerState(refmod,await refmodSave(node,doc,refmod)));commit(node,doc);render(node);}catch(err){alert(`Save RefMod failed: ${err?.message||err}`);}},false,"Save settings or import the source with the current H3 VAEs");Object.assign(save.style,{gridColumn:"1 / -1",width:"100%"});body.append(save);
const place=button("PLACE ON LAYER",()=>{try{placeRefmodOnTimeline(node,doc,refmod);commit(node,doc);}catch(err){alert(`Place failed: ${err?.message||err}`);}},false,"Put this RefMod on the REFMOD layer at the playhead — it conditions the render only during that interval. The clip inspector opens with Start / Duration / Strength; use ↔ there to stretch it across the whole timeline.");Object.assign(place.style,{gridColumn:"1 / -1",width:"100%"});body.append(place);
  const encoded=Boolean(refmod.path)&&String(refmod.state||"")!=="pending_runtime_encode";
  const savedSources=Array.isArray(refmod.sources)?refmod.sources.filter(s=>s&&String(s.filename||"").trim()&&["image","video","audio"].includes(String(s.kind||"").toLowerCase())):[];
  const refmodSource=String(refmod.source||"").trim();
  const hasDirectorSource=/^(subject|clip|take):/.test(refmodSource)||Boolean(refmod.source_subject_id||refmod.source_clip_id);
  const canReencodeSource=savedSources.length>0||hasDirectorSource;
  const importedArtifactSource=/^input[\\/].+\.safetensors$/i.test(refmodSource);
  const noSourceMessage=importedArtifactSource?"This imported .safetensors is already encoded and has no raw media source. Use CREATE FROM SOURCES to rebuild it from images, video or audio.":"Re-encoding needs attached source media. Add original images, video or audio with CREATE FROM SOURCES, or attach a Director media/TAKE source.";
  const encodeTitle=canReencodeSource?"ENCODE — VAE-only RefMod preparation. Reuses the saved raw sources when available; no text encoder, DiT, sampler or Long Media Setup.":noSourceMessage;
  const encode=button("ENCODE",async()=>{if(node.__lmdRefmodBusy)return;node.__lmdRefmodBusy=true;try{const res=await refmodEncode(node,doc,refmod);if(res.refmod)Object.assign(refmod,applyRefmodServerState(refmod,res.refmod));else Object.assign(refmod,{path:res.path||refmod.path,state:"ready"});commit(node,doc);render(node);}catch(err){alert(`ENCODE failed: ${err?.message||err}`);}finally{node.__lmdRefmodBusy=false;}},false,encodeTitle);encode.disabled=!canReencodeSource;Object.assign(encode.style,{gridColumn:"1 / -1",width:"100%",opacity:!canReencodeSource?"0.48":encoded?"0.65":"1",cursor:canReencodeSource?"pointer":"not-allowed"});body.append(encode);
  const statusText=encoded?(canReencodeSource?"READY":"READY · encoded artifact; raw source unavailable"):(canReencodeSource?"NOT ENCODED · press ENCODE":"NO SOURCE MEDIA · attach a source or use CREATE FROM SOURCES");
  const state=el("div",`${statusText}${refmod.path?` · ${refmod.path}`:""}`,{gridColumn:"1 / -1",font:"8px/1.35 system-ui",color:encoded?"#8fd3a8":"#e8a952",overflowWrap:"anywhere"});body.append(state);
  if(!canReencodeSource)body.append(el("div",noSourceMessage,{gridColumn:"1 / -1",font:"8px/1.35 system-ui",color:styles.dim}));
  details.append(body);host.append(details);
  const del=iconButton("🗑","Delete RefMod from library and document",async()=>{if(!window.confirm(`Delete RefMod "${refmod.name}"?`))return;try{await refmodDelete(node,doc,refmod.refmod_id);}catch(err){return alert(`Delete failed: ${err?.message||err}`);}doc.refmods=(doc.refmods||[]).filter(r=>r.refmod_id!==refmod.refmod_id);node.__lmdRefmodSelected=null;node.__lmdRefmodOpen=null;commit(node,doc);});del.style.marginTop="6px";host.append(del);
}
function renderRefmodPanel(node,doc,host){
  const refmods=doc.refmods||[];
  const selectedId=node.__lmdRefmodSelected||null;
  const activeFolder=node.__lmdRefmodFolder||"Unsorted",visibleRefmods=refmods.filter(r=>String(r.folder||"Unsorted")===activeFolder);
  const selectedRefmod=refmods.find(r=>r.refmod_id===selectedId&&String(r.folder||"Unsorted")===activeFolder)||null;
  const head=el("div",null,{position:"sticky",top:"0",zIndex:"25",background:"#171b21",padding:"8px 0 9px",borderBottom:`1px solid ${styles.border}`,boxShadow:"0 8px 12px rgba(23,27,33,.98)"});
  const titleRow=el("div",null,{display:"flex",alignItems:"center",justifyContent:"space-between",gap:"6px"});
  titleRow.append(el("div","REFMODS",{font:"900 11px system-ui",letterSpacing:".08em",color:"#e8a952"}));
  head.append(titleRow);
  head.append(el("div",`${visibleRefmods.length} item${visibleRefmods.length===1?"":"s"} · ${refmods.length} total`,{font:"8px system-ui",color:styles.dim,marginTop:"2px"}));
  if(node.__lmdRefmodSyncStatus)head.append(el("div",String(node.__lmdRefmodSyncStatus),{font:"7px/1.35 system-ui",color:styles.dim,marginTop:"2px",overflowWrap:"anywhere"}));
  const toolbar=el("div",null,{display:"flex",gap:"5px",marginTop:"7px",flexWrap:"wrap"});
  toolbar.append(button("CREATE FROM SOURCES",()=>openRefmodCreator(node,doc),true,"Open a source editor for multiple images/videos, audio and clipboard paste; encode the selected sources as one RefMod bundle."));
  const folderSelect=selectOptions((node.__lmdRefmodFolders||["Unsorted"]).map(f=>({value:f,label:`📁 ${f}`})),node.__lmdRefmodFolder||"Unsorted",v=>{node.__lmdRefmodFolder=v;render(node);});Object.assign(folderSelect.style,{flex:"1 1 100%",minWidth:"0"});
  toolbar.append(folderSelect,button("+ FOLDER",()=>createRefmodFolder(node,doc),false,"Create a RefMod library folder"),button("IMPORT .SAFETENSORS",async()=>{const file=await pick("refmod");if(!file)return;try{await importRefmodFile(node,doc,file,{folder:node.__lmdRefmodFolder||"Unsorted"});}catch(err){alert(`Import RefMod failed: ${err?.message||err}`);}},false,"Import a compatible RefMod file and add it to the selected folder and timeline"),button("CREATE",()=>{const rm=defaultRefmod();rm.folder=node.__lmdRefmodFolder||"Unsorted";doc.refmods=[...(doc.refmods||[]),rm];node.__lmdRefmodSelected=rm.refmod_id;commit(node,doc);},false,"Create a new blank RefMod in the library"),button("FROM TAKE",async()=>{const shot=takeShotForSidebar(node,doc);const status=node.__lmdTakeStatus?.clips?.[shot?.clip_id]||{};const revision=selectedTakeRevision(node,shot,status);if(!revision)return alert("Select a TAKE first.");const take=(status.takes||[]).find(t=>String(t.revision||"")===revision);if(!take)return alert("Take not found.");try{const rm=await refmodFromTake(node,doc,shot,take);if(rm){rm.folder=node.__lmdRefmodFolder||"Unsorted";doc.refmods=[...(doc.refmods||[]),normalizeRefmod(rm)];node.__lmdRefmodSelected=rm.refmod_id;commit(node,doc);}}catch(err){alert(`From TAKE failed: ${err?.message||err}`);}},false,"Derive a new RefMod snapshot from the active TAKE"),button("SYNC",async()=>{
    // Reconcile, do not union: entries whose .safetensors vanished from the
    // shared library must leave the document, or the panel can never empty.
    try{
      const {refmods:remote,folders}=await refmodList(node,doc);
      const result=reconcileRefmodLibrary(doc.refmods||[],remote);
      const dangling=result.removedIds.length?refmodTimelineReferrers(doc,result.removedIds):0;
      doc.refmods=result.refmods;
      const known=node.__lmdRefmodFolders||folders;
      node.__lmdRefmodFolder=known.includes(node.__lmdRefmodFolder)?node.__lmdRefmodFolder:(known.includes("Unsorted")?"Unsorted":(known[0]||"Unsorted"));
      if(node.__lmdRefmodSelected&&!doc.refmods.some(r=>String(r.refmod_id||"")===String(node.__lmdRefmodSelected||""))){node.__lmdRefmodSelected=null;node.__lmdRefmodOpen=null;}
      commit(node,doc);
      node.__lmdRefmodSyncStatus=`SYNC ${new Date().toLocaleTimeString()} · library ${remote.length} · +${result.added} ~${result.updated} −${result.removed.length}${result.drafts.length?` · ${result.drafts.length} unsaved draft(s) kept`:""}${dangling?` · ${dangling} timeline clip(s) reference a removed RefMod`:""}`;
      render(node);
    }catch(err){
      alert(`SYNC failed: ${err?.message||err}\nNothing was changed — the RefMod list is untouched.`);
    }
  },false,"Reconcile this doc with the RefMod library on disk: adds new, refreshes existing and removes entries whose .safetensors no longer exists. Unsaved drafts are kept; Ctrl+Z restores the previous list."));
  head.append(toolbar);
  host.append(head);
  const lib=el("div",null,{marginTop:"7px"});
  lib.append(el("div","LIBRARY",{font:"800 8px system-ui",letterSpacing:".07em",color:styles.dim,padding:"5px 0 4px"}));
  if(!visibleRefmods.length){lib.append(el("div",`No RefMods in ${activeFolder}. Drop a .safetensors file on the timeline or click IMPORT.`,{font:"9px/1.4 system-ui",color:styles.dim,padding:"10px 2px"}));}
  else{for(const rm of visibleRefmods){const active=rm.refmod_id===selectedId;const card=el("div",null,{background:"#101319",border:`1px solid ${active?"#e8a952":styles.border}`,borderRadius:"6px",padding:"6px",cursor:"pointer",marginBottom:"5px",boxShadow:active?"0 0 0 1px rgba(232,169,82,.28)":"none"});
    card.append(el("div",rm.name,{font:"800 10px system-ui",color:active?"#fff0bd":"#e5e9ef",whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}));
    card.append(el("div",`${rm.mode} · ${rm.concept_type} · str ${clamp(rm.strength,0,1,1).toFixed(2)}${String(rm.path||"").trim()?"":" · DRAFT"}`,{font:"7px system-ui",color:String(rm.path||"").trim()?styles.dim:"#e8a952",marginTop:"2px",textTransform:"uppercase",letterSpacing:".03em"}));
    card.onclick=()=>{if(node.__lmdRefmodSelected===rm.refmod_id){node.__lmdRefmodSelected=null;node.__lmdRefmodOpen=null;}else{node.__lmdRefmodSelected=rm.refmod_id;node.__lmdRefmodOpen=rm.refmod_id;}render(node);};
    card.ondragover=e=>{if(droppedFiles(e).some(f=>fileKind(f)==="refmod")){e.preventDefault();e.stopPropagation();card.style.outline="2px solid #e8a952";}};card.ondragleave=()=>card.style.outline="";
    card.ondrop=async e=>{const file=droppedFiles(e).find(f=>fileKind(f)==="refmod");if(!file)return;e.preventDefault();e.stopPropagation();try{const uploaded=await upload("refmod",file),source=`input/${uploaded.subfolder?`${uploaded.subfolder}/`:""}${uploaded.filename}`,authored={...rm,source,state:"ready"},saved=applyRefmodServerState(authored,await refmodSave(node,doc,authored));Object.assign(rm,saved);node.__lmdRefmodSelected=rm.refmod_id;node.__lmdRefmodOpen=rm.refmod_id;commit(node,doc);render(node);}catch(err){alert(`Replace RefMod failed: ${err?.message||err}`);}};
    lib.append(card);}}
  host.append(lib);
  if(selectedRefmod){const insp=el("div",null,{marginTop:"7px",paddingTop:"7px",borderTop:`1px solid ${styles.border}`});renderRefmodInspector(node,doc,selectedRefmod,insp);host.append(insp);}
}
function renderRefmodSidebar(node,doc,host){return renderRefmodPanel(node,doc,host);}

function renderDirectorClipSummary(node,doc,host){
  const selected=selectedTimelineItem(node,doc),sel=selection(node);let title="No timeline item selected",kind="Timeline";if(selected){kind=String(selected.kind||"Timeline").toUpperCase();title=String(selected.item?.name||selected.item?.clip_name||selected.track?.name||kind);}
  const card=el("section",null,{padding:"10px",background:styles.panel,border:`1px solid ${styles.border}`,borderRadius:"6px"});card.append(el("div","SELECTED CLIP",{font:"900 9px system-ui",letterSpacing:".08em",color:"#c7d0dc"}),el("div",title,{font:"800 13px system-ui",marginTop:"7px",overflowWrap:"anywhere"}));
  if(selected){card.append(el("div",`${kind} · ${finite(selected.start,0).toFixed(2)}–${finite(selected.end,0).toFixed(2)} s · ${Math.max(0,finite(selected.end,0)-finite(selected.start,0)).toFixed(2)} s`,{font:"9px system-ui",color:styles.dim,marginTop:"5px"}));if(selected.kind==="main"){const shot=selected.item,subject=subjectById(doc,shot?.media_subject_id);card.append(el("div",`Base type · ${shot?.base_type||"generated"}`,{font:"9px system-ui",color:styles.dim,marginTop:"8px"}));if(subject)card.append(el("div",`Source · ${subject.name||subject.kind}`,{font:"9px system-ui",color:styles.dim,marginTop:"4px",overflowWrap:"anywhere"}));if(shot?.prompt)card.append(el("div",String(shot.prompt),{font:"9px/1.45 system-ui",color:"#c7d0dc",marginTop:"8px",whiteSpace:"pre-wrap",overflowWrap:"anywhere"}));}}else card.append(el("div",`Selected layer: ${String(sel.track||"main")}`,{font:"9px system-ui",color:styles.dim,marginTop:"5px"}));
  const jump=button("OPEN CLIP SETTINGS",()=>{node.__lmdBottomTab="clip";saveDirectorUIPref(node,"bottom_tab","clip");render(node);});Object.assign(jump.style,{width:"100%",marginTop:"10px"});card.append(jump);host.append(card);
}
function renderDirectorSources(node,doc,host){
  host.append(el("div","SOURCE MEDIA",{font:"900 10px system-ui",letterSpacing:".07em",color:"#c7d0dc",marginBottom:"5px"}),el("div","Media attached to this Director project and referenced by its timeline.",{font:"9px/1.4 system-ui",color:styles.dim,marginBottom:"9px"}));if(!(doc.subjects||[]).length){host.append(el("div","No source media yet. Import files from the Media tab.",{font:"9px/1.4 system-ui",color:styles.dim,padding:"12px 2px"}));return;}
  const shotUses=new Map();for(const shot of doc.shots||[])for(const id of [shot.media_subject_id,...(shot.subjects||[])].filter(Boolean))shotUses.set(id,(shotUses.get(id)||0)+1);const rows=el("div",null,{display:"grid",gap:"6px"});for(const subject of doc.subjects||[]){const media=subject.media||{},kind=String(subject.kind||"Media"),uses=shotUses.get(subject.subject_id)||0,row=el("div",null,{padding:"8px",background:styles.panel,border:`1px solid ${styles.border}`,borderRadius:"5px",minWidth:"0"});row.append(el("div",`${kind.toUpperCase()} · ${subject.name||"Untitled source"}`,{font:"800 9px system-ui",color:kind==="Audio"?"#95dbaa":kind==="Video"?"#f0b477":"#9bd4f1",overflowWrap:"anywhere"}));const detail=[media.width&&media.height?`${media.width}×${media.height}`:"",media.seconds?`${Number(media.seconds).toFixed(2)} s`:"",media.fps?`${media.fps} fps`:"",`${uses} timeline reference${uses===1?"":"s"}`].filter(Boolean).join(" · ");row.append(el("div",detail||mediaPath(media)||"No imported file",{font:"8px/1.4 system-ui",color:styles.dim,marginTop:"4px",overflowWrap:"anywhere"}));if(mediaPath(media))row.append(el("div",mediaPath(media),{font:"7px/1.35 system-ui",color:styles.dim,marginTop:"4px",overflowWrap:"anywhere"}));rows.append(row);}host.append(rows);
}
function render(node){if(!isDirector(node))return;closeTimelineContextMenu(node);if(node.__lmdTakeSortDismiss){window.removeEventListener("pointerdown",node.__lmdTakeSortDismiss,true);node.__lmdTakeSortDismiss=null;}loadDirectorUIState(node);for(const ro of node.__lmdResizeObservers||[])try{ro.disconnect();}catch(_){}node.__lmdResizeObservers=[];const editor=ensureEditor(node);const root=editor?.element||node.__lmdRoot;if(!root)return;const doc=node.__lmdDoc??parseDoc(node);node.__lmdDoc=doc;
  if(node.__lmdInspectorHost?.isConnected)node.__lmdInspectorScrollTop=node.__lmdInspectorHost.scrollTop;if(node.__lmdLeftContent?.isConnected&&!node.__lmdLeftContent.__lmdRestoringScroll)node.__lmdLeftScrollTop=node.__lmdLeftContent.scrollTop;if(node.__lmdRightContent?.isConnected)node.__lmdRightScrollTop=node.__lmdRightContent.scrollTop;if(node.__lmdBottomContent?.isConnected)node.__lmdBottomScrollTop=node.__lmdBottomContent.scrollTop;for(const ro of node.__lmdResizeObservers||[])try{ro.disconnect();}catch(_){}node.__lmdResizeObservers=[];releaseTakeGalleryImages(node);releaseTimelineVideoElements(root,node);root.replaceChildren();
  const firstShot=(doc.shots||[]).find(s=>s.first_frame_anchor),lastShot=[...(doc.shots||[])].reverse().find(s=>s.last_frame_anchor),loop=Boolean(firstShot&&lastShot&&firstShot.clip_id===lastShot.clip_id&&firstShot.media_subject_id===lastShot.media_subject_id);const anchorStatus=loop?"LOOP":firstShot&&lastShot?"FIRST + LAST":firstShot?"FIRST":lastShot?"LAST":"NONE";
  const header=el("div",null,{display:"flex",alignItems:"center",gap:"8px",height:"40px",boxSizing:"border-box",padding:"5px 8px",borderBottom:`1px solid ${styles.border}`,background:"linear-gradient(180deg,#1b2027,#16191f)",boxShadow:"0 2px 8px rgba(0,0,0,.16)",whiteSpace:"nowrap",overflow:"hidden"});
  header.append(el("div","LongMedia Director",{font:"800 12px system-ui",flex:"0 0 auto"}));
  const statusField=el("div",null,{display:"inline-flex",alignItems:"center",gap:"6px",height:"28px",boxSizing:"border-box",padding:"0 8px",border:`1px solid ${loop?"#4c795d":styles.border}`,borderRadius:"5px",background:loop?"rgba(45,92,62,.18)":"#101319",flex:"0 0 auto"});statusField.append(el("span","STATUS",{font:"800 8px system-ui",letterSpacing:".06em",color:styles.dim}),el("span",`${doc.shots.length} ${directorMainTerm(doc)}${doc.shots.length===1?"":"s"} · ${totalDuration(doc).toFixed(2)}s · ${anchorStatus}`,{font:"700 9px system-ui",color:loop?"#9fe0b7":"#c7d0dc"}));header.append(statusField);
  const addHeaderMode=(label,control,help)=>{const group=el("div",null,{display:"inline-flex",alignItems:"center",gap:"4px",flex:"0 0 auto"});group.append(el("span",label,{font:"800 8px system-ui",color:styles.dim}),control);if(help)control.title=help;header.append(group);};
  const h3Select=select(DIRECTOR_H3_MODES,doc.setup_h3_mode||"auto",v=>{doc.setup_h3_mode=v;commit(node,doc);});h3Select.style.width="118px";addHeaderMode("H3",h3Select,"Director-owned H3 family. auto selects FL2VA for pure FIRST/LAST anchors, Hybrid for anchors + REF, otherwise Ref2VA/T2VA from active references.");
  const tlSelect=select(DIRECTOR_TIMELINE_MODES,doc.setup_timeline_mode||"auto",v=>{const prev=String(doc.setup_timeline_mode||"auto");if(v==="segmented"&&doc.shots.length!==1){alert("Segmented mode owns one MAIN shot and divides it into fixed continuation segments. Keep/collapse MAIN to one shot first.");render(node);return;}doc.setup_timeline_mode=v;if(v==="multiclip"){for(let i=0;i<doc.shots.length;i++)if(/^Shot\b/i.test(String(doc.shots[i].name||"")))doc.shots[i].name=String(doc.shots[i].name).replace(/^Shot/i,"Clip");}if(v==="segmented"){if(prev!=="segmented"){const total=Math.max(.25,totalDuration(doc)),count=Math.max(1,Math.min(64,Math.round(total/5)));doc.segmented_count=count;doc.segmented_duration=Math.round((total/count)*1000)/1000;}syncSegmentedTimeline(doc);}if(v!=="multiclip")node.__lmdMulticlipBoundary=null;commit(node,doc);});tlSelect.style.width="108px";addHeaderMode("TIMELINE",tlSelect,"single = one H3 pass; segmented = fixed continuation segments; multiclip = authored clip markers/prompts.");
  if(String(doc.setup_timeline_mode)==="segmented"){const segCount=input(doc.segmented_count??3,v=>{doc.segmented_count=Math.max(1,Math.min(64,Math.trunc(finite(v,doc.segmented_count||3))));syncSegmentedTimeline(doc);commit(node,doc);},"number");segCount.min="1";segCount.max="64";segCount.step="1";segCount.style.width="54px";addHeaderMode("SEG",segCount,"Number of fixed continuation segments. Director owns this value in segmented mode.");const segLen=input(doc.segmented_duration??5,v=>{doc.segmented_duration=clamp(v,.25,150,doc.segmented_duration||5);syncSegmentedTimeline(doc);commit(node,doc);},"number");segLen.min="0.25";segLen.max="150";segLen.step="0.25";segLen.style.width="64px";addHeaderMode("SEG LEN",segLen,"Visible duration of each fixed segment in seconds. Hidden transition/context overlap remains controlled separately.");}
  const audioSelect=select(AUDIO_MODES,doc.audio_mode||"auto",v=>{doc.audio_mode=v;commit(node,doc);});audioSelect.style.width="112px";addHeaderMode("AUDIO",audioSelect,"Director audio behavior. lip_sync makes Audio 1 the authoritative mouth/timing clock and restores Audio 1 at output.");
  const resolutionSource=selectOptions(DIRECTOR_RESOLUTION_SOURCES,doc.resolution_source||"base_layer",v=>{doc.resolution_source=v;commit(node,doc);});resolutionSource.style.width="102px";addHeaderMode("RES SRC",resolutionSource,"Output aspect source. Base layer follows the first MAIN visual media aspect; presets use their ratio; Custom uses the W×H field as an aspect source. Setup width/height are ignored in Director mode.");
  if(String(doc.resolution_source||"base_layer")==="custom"){const res=input(doc.resolution||"1920x1080",v=>{const cleaned=String(v||"").trim().replaceAll("×","x").replace(/\s+/g,"");if(/^\d{2,5}x\d{2,5}$/i.test(cleaned)){doc.resolution=cleaned;commit(node,doc);}},"text");res.style.width="92px";addHeaderMode("RES",res,"Custom W×H aspect source. Megapixel controls final total pixel budget; H3 output snaps to 32px geometry.");}
  const mp=input(doc.megapixel??0.8,v=>{doc.megapixel=clamp(v,.1,8,.8);commit(node,doc);},"number");mp.min="0.1";mp.max="8";mp.step="0.05";mp.style.width="58px";const rp=directorResolutionPreview(doc);addHeaderMode("MP",mp,`Director-owned total pixel budget. Current target ≈ ${rp.width}×${rp.height} (${rp.actual.toFixed(3)} MP), aspect source: ${rp.basis}.`);
  if((doc.shots||[]).length===1&&String(doc.setup_timeline_mode)!=="segmented"){const len=input(doc.shots[0]?.duration||5,v=>{if(doc.shots?.[0]){doc.shots[0].duration=clamp(v,.25,150,doc.shots[0].duration||5);commit(node,doc);}},"number");len.style.width="58px";addHeaderMode("LEN",len,"Single-shot Director duration in seconds.");}
  root.append(header);
  const toolbar=el("div",null,{display:"flex",alignItems:"center",justifyContent:"center",gap:"5px",height:"38px",boxSizing:"border-box",padding:"4px 8px",borderBottom:`1px solid ${styles.border}`,background:"linear-gradient(180deg,#171b21,#14171c)",overflow:"hidden"});
  toolbar.append(button("PREVIEW",()=>{doc.render_mode="preview";commit(node,doc);},doc.render_mode==="preview","Render at 50% width and height (25% pixel budget); sampler steps stay the same."),button("FINAL",()=>{doc.render_mode="final";commit(node,doc);},doc.render_mode!=="preview","Render at the authored megapixel target."),iconButton("↶","Undo Director edit (Ctrl/Cmd+Z)",()=>undo(node)),iconButton("↷","Redo Director edit (Ctrl/Cmd+Y)",()=>redo(node)),iconButton("⇆",doc.ripple_tracks!==false?"Ripple ON · CAMERA/AUDIO follow their MAIN scene when clips are reordered or durations change":"Ripple OFF · CAMERA/AUDIO keep absolute timeline positions",()=>{doc.ripple_tracks=doc.ripple_tracks===false;commit(node,doc);},doc.ripple_tracks!==false),iconButton("🧲","Toggle Magnet snapping for playhead, trims and movable clips. Hold Alt while dragging to bypass snapping temporarily.",()=>{doc.snapping=doc.snapping===false;commit(node,doc);},doc.snapping!==false));
  const divider=()=>el("span","",{width:"1px",height:"20px",background:styles.border,margin:"0 4px",flex:"0 0 auto"});toolbar.append(divider());
  for(const [symbol,action,title] of [["CUT","cut_mark","Add a non-destructive CUT marker inside the MAIN clip. Clip edges snap to markers; this does not split or render."],["B","split","Knife: physically split all unlocked clips at the playhead (B). CUT markers are separate."],["▣","video_prompt",String(doc.setup_timeline_mode)==="multiclip"?"Add MAIN clip":"Add MAIN shot"],["◉","camera_prompt","Add camera block"],["♪","sound_prompt","Add sound prompt"]])toolbar.append(iconButton(symbol,title,()=>toolbarAction(node,doc,action)));
  for(const [symbol,action,title] of [["T","track_prompt","Add prompt layer"],["✦","track_embedding","Add MiniMax H3 embedding layer"],["♙","track_character","Add character layer"],["◆","track_reference","Add reference layer"],["▤","track_video","Add video layer"],["♫","track_audio","Add audio layer"],["⬡","track_refmod","Add RefMod layer"]])toolbar.append(iconButton(symbol,title,()=>toolbarAction(node,doc,action)));
  root.append(toolbar);
  const leftWidth=clamp(node.__lmdLeftWidth,180,420,220),rightWidth=clamp(node.__lmdRightWidth,220,500,280);node.__lmdLeftWidth=leftWidth;node.__lmdRightWidth=rightWidth;
  const body=el("div",null,{display:"grid",gridTemplateColumns:`${leftWidth}px minmax(0,1fr) ${rightWidth}px`,gap:"8px",height:"calc(100% - 78px)",minHeight:"0",boxSizing:"border-box",padding:"8px",background:"linear-gradient(180deg,#111318 0%,#0f1115 100%)"});
  const sidePanel=()=>el("aside",null,{display:"flex",flexDirection:"column",minWidth:"0",minHeight:"0",overflow:"hidden",background:"linear-gradient(180deg,#171b21,#13161b)",border:`1px solid ${styles.border}`,borderRadius:"7px",boxShadow:"0 8px 22px rgba(0,0,0,.16)"});
  const tabBar=()=>el("div",null,{display:"flex",gap:"4px",flex:"0 0 auto",padding:"6px",borderBottom:`1px solid ${styles.border}`,background:"#15191f"});
  const tabButton=(label,active,onClick)=>{const b=button(label,onClick,active);Object.assign(b.style,{flex:"1 1 0",minWidth:"0",height:"31px",padding:"0 8px",fontSize:"10px",fontWeight:"800",borderRadius:"4px"});return b;};
  const leftPanel=sidePanel(),leftTabs=tabBar(),leftContent=el("div",null,{flex:"1 1 auto",minHeight:"0",overflowY:"auto",overflowX:"hidden",padding:"0 9px 9px"});node.__lmdLeftContent=leftContent;leftContent.__lmdRestoringScroll=true;leftContent.addEventListener("scroll",()=>{if(!leftContent.__lmdRestoringScroll)node.__lmdLeftScrollTop=leftContent.scrollTop;},{passive:true});
  leftTabs.append(tabButton("Takes",node.__lmdLeftTab==="takes",()=>{node.__lmdLeftTab="takes";saveDirectorUIPref(node,"left_tab","takes");render(node);}),tabButton("RefMods",node.__lmdLeftTab==="refmods",()=>{node.__lmdLeftTab="refmods";saveDirectorUIPref(node,"left_tab","refmods");render(node);}));leftPanel.append(leftTabs,leftContent);
  if(node.__lmdLeftTab==="refmods")renderRefmodPanel(node,doc,leftContent);else renderTakeSidebar(node,doc,leftContent);leftContent.scrollTop=node.__lmdLeftScrollTop||0;
  const timelineHeight=clamp(node.__lmdTimelineHeight,240,1100,320);node.__lmdTimelineHeight=timelineHeight;
  const center=el("main",null,{display:"grid",gridTemplateRows:`${PROGRAM_MONITOR_HEIGHT}px ${timelineHeight}px minmax(320px,1fr)`,gap:"8px",minWidth:"0",minHeight:"0",overflowY:"auto",overflowX:"hidden",paddingRight:"2px"});
  const monitorBox=el("div",null,{minWidth:"0"});renderProgramMonitor(node,doc,monitorBox);center.append(monitorBox);
  const timelineBox=el("div",null,{position:"relative",overflow:"hidden",height:`${timelineHeight}px`,minHeight:"240px",maxHeight:"1100px",paddingBottom:"9px",boxSizing:"border-box",background:styles.panel,border:`1px solid ${styles.border}`,borderRadius:"7px",boxShadow:"0 7px 20px rgba(0,0,0,.15)"});const timelineSurface=el("div",null,{height:"100%",minHeight:"0"});renderTimeline(node,doc,timelineSurface);
  const timelineGrip=el("div",null,{position:"absolute",left:"0",right:"0",bottom:"0",height:"10px",cursor:"ns-resize",touchAction:"none",userSelect:"none",zIndex:"40",background:"linear-gradient(180deg,transparent,rgba(123,166,255,.14))"});timelineGrip.append(el("div",null,{position:"absolute",left:"50%",bottom:"2px",width:"46px",height:"3px",transform:"translateX(-50%)",borderRadius:"3px",background:"#566171"}));attachDirectorTooltip(timelineGrip,"Drag to resize the timeline viewport. Use the ↕ slider beside timeline zoom and Fit to change every track/layer height.");timelineGrip.onpointerdown=e=>{if(e.button!==0)return;e.preventDefault();e.stopPropagation();const pointerId=e.pointerId,startY=e.clientY,startHeight=clamp(node.__lmdTimelineHeight,240,1100,timelineHeight);try{timelineGrip.setPointerCapture(pointerId);}catch(_){}const move=ev=>{if(ev.pointerId!==pointerId)return;ev.preventDefault();ev.stopPropagation();const h=clamp(startHeight+(ev.clientY-startY),240,1100,startHeight);node.__lmdTimelineHeight=h;timelineBox.style.height=`${h}px`;center.style.gridTemplateRows=`${PROGRAM_MONITOR_HEIGHT}px ${h}px minmax(320px,1fr)`;};const up=ev=>{if(ev.pointerId!==pointerId)return;ev.preventDefault();ev.stopPropagation();saveDirectorUIPref(node,"timeline_height",Math.round(node.__lmdTimelineHeight));window.removeEventListener("pointermove",move,true);window.removeEventListener("pointerup",up,true);window.removeEventListener("pointercancel",up,true);try{timelineGrip.releasePointerCapture(pointerId);}catch(_){}};window.addEventListener("pointermove",move,true);window.addEventListener("pointerup",up,true);window.addEventListener("pointercancel",up,true);};timelineGrip.onmousedown=e=>{e.preventDefault();e.stopPropagation();};timelineBox.append(timelineSurface,timelineGrip);center.append(timelineBox);
  const settings=el("section",null,{display:"flex",flexDirection:"column",minWidth:"0",minHeight:"320px",overflow:"hidden",background:"linear-gradient(180deg,#171b21,#13161b)",border:`1px solid ${styles.border}`,borderRadius:"7px",boxShadow:"0 7px 20px rgba(0,0,0,.14)"});const bottomTabs=tabBar(),bottomContent=el("div",null,{flex:"1 1 auto",minHeight:"0",overflowY:"auto",overflowX:"hidden",padding:"8px"});node.__lmdBottomContent=bottomContent;
  for(const [key,label] of [["clip","Clip Settings"],["global","Global Prompt"],["scene","Scene Direction"]])bottomTabs.append(tabButton(label,node.__lmdBottomTab===key,()=>{node.__lmdBottomTab=key;saveDirectorUIPref(node,"bottom_tab",key);render(node);}));
  if(node.__lmdBottomTab==="global"){const gp=widget(node,"global_prompt"),editor=persistentTextarea(node,"global:prompt",gp?.value||"",v=>{if(gp){gp.value=v;try{gp.callback?.(v);}catch(_){}}},5);editor.style.maxWidth="100%";bottomContent.append(editor);}
  else if(node.__lmdBottomTab==="scene"){bottomContent.append(el("div","Scene-wide style, atmosphere, palette and visual direction are shared across the complete timeline.",{font:"9px/1.4 system-ui",color:styles.dim,marginBottom:"7px"}));renderSceneGuidanceBuilder(node,doc,bottomContent);}
  else{const inspector=el("div",null,{minWidth:"0",overflow:"visible",background:"linear-gradient(180deg,#15191f,#12151a)",border:`1px solid ${styles.border}`,borderRadius:"6px",padding:"8px",boxSizing:"border-box"});node.__lmdInspectorHost=inspector;renderInspector(node,doc,inspector);bottomContent.append(inspector);}
  bottomContent.scrollTop=node.__lmdBottomScrollTop||0;settings.append(bottomTabs,bottomContent);center.append(settings);
  const rightPanel=sidePanel(),rightTabs=tabBar(),rightContent=el("div",null,{flex:"1 1 auto",minHeight:"0",overflowY:"auto",overflowX:"hidden",padding:"8px"});node.__lmdRightContent=rightContent;
  rightTabs.append(tabButton("Media",node.__lmdRightTab==="media",()=>{node.__lmdRightTab="media";saveDirectorUIPref(node,"right_tab","media");render(node);}),tabButton("Clip",node.__lmdRightTab==="clip",()=>{node.__lmdRightTab="clip";saveDirectorUIPref(node,"right_tab","clip");render(node);}),tabButton("Sources",node.__lmdRightTab==="sources",()=>{node.__lmdRightTab="sources";saveDirectorUIPref(node,"right_tab","sources");render(node);}));
  if(node.__lmdRightTab==="media")renderAssets(node,doc,rightContent);else if(node.__lmdRightTab==="clip")renderDirectorClipSummary(node,doc,rightContent);else renderDirectorSources(node,doc,rightContent);rightContent.scrollTop=node.__lmdRightScrollTop||0;rightPanel.append(rightTabs,rightContent);
  const installResize=edge=>{const grip=el("div",null,{position:"absolute",top:"0",bottom:"0",[edge]:"-5px",width:"10px",zIndex:"40",cursor:"ew-resize",background:"transparent"});grip.onpointerdown=e=>{if(e.button!==0)return;e.preventDefault();e.stopPropagation();const x0=e.clientX,left0=node.__lmdLeftWidth,right0=node.__lmdRightWidth;const move=ev=>{const dx=ev.clientX-x0;if(edge==="right"){node.__lmdLeftWidth=clamp(left0+dx,180,420,left0);body.style.gridTemplateColumns=`${node.__lmdLeftWidth}px minmax(0,1fr) ${node.__lmdRightWidth}px`;}else{node.__lmdRightWidth=clamp(right0-dx,220,500,right0);body.style.gridTemplateColumns=`${node.__lmdLeftWidth}px minmax(0,1fr) ${node.__lmdRightWidth}px`;}};const up=()=>{saveDirectorUIPref(node,"left_width",Math.round(node.__lmdLeftWidth));saveDirectorUIPref(node,"right_width",Math.round(node.__lmdRightWidth));window.removeEventListener("pointermove",move,true);window.removeEventListener("pointerup",up,true);window.removeEventListener("pointercancel",up,true);};window.addEventListener("pointermove",move,true);window.addEventListener("pointerup",up,true);window.addEventListener("pointercancel",up,true);};return grip;};leftPanel.style.position="relative";rightPanel.style.position="relative";leftPanel.append(installResize("right"));rightPanel.append(installResize("left"));
  body.append(leftPanel,center,rightPanel);root.append(body);
  // These panels are built while detached. Assigning scrollTop before appending
  // them is clamped to zero because the browser has not computed their viewport
  // yet. Restore after layout so selecting a TAKE does not jump the gallery.
  const restorePanelScroll=(element,top)=>{
    const target=finite(top,0);let restored=false;
    const apply=()=>{if(restored||!element.isConnected)return;if(element.clientHeight<=0)return;element.scrollTop=target;element.__lmdRestoringScroll=false;restored=true;observer.disconnect();};
    const observer=new ResizeObserver(apply);observer.observe(element);observer.observe(root);node.__lmdResizeObservers.push(observer);
    requestAnimationFrame(()=>requestAnimationFrame(apply));
  };
  restorePanelScroll(leftContent,node.__lmdLeftScrollTop);restorePanelScroll(rightContent,node.__lmdRightScrollTop);restorePanelScroll(bottomContent,node.__lmdBottomScrollTop);
}

function ensureEditor(node){
  if(node.__lmdEditor){if((node.widgets||[]).includes(node.__lmdEditor))return node.__lmdEditor;node.__lmdEditor=null;node.__lmdRoot=null;}
  if(typeof node.addDOMWidget!=="function")return null;
  // The Director surface is presentation-only. Never keep a second, detached
  // director_surface widget around: it would ship serialized decorative state
  // back into the workflow JSON on queue.
  for(const w of [...(node.widgets||[])]){
    if(w&&w.name==="director_surface"&&w!==node.__lmdEditor){
      const i=node.widgets.indexOf(w);if(i>=0)node.widgets.splice(i,1);
      try{w.element?.remove?.();}catch(_){}
    }
  }
  hideWidget(widget(node,"director_json"));hideWidget(widget(node,"global_prompt"));
  // Height ownership is deliberately one-way: the ComfyUI node allocates the DOM row,
  // while the editor fills it. The widget's minimum/maximum are constants and never read
  // node.size, which avoids the old node-height -> widget-height -> node-height feedback loop.
  const root=el("div",null,{width:"100%",height:"100%",minHeight:`${EDITOR_MIN_HEIGHT}px`,boxSizing:"border-box",background:styles.bg,color:styles.text,font:"11px system-ui",overflow:"hidden",border:`1px solid ${styles.border}`,borderRadius:"6px"});root.tabIndex=0;root.addEventListener("keydown",e=>{const tag=String(e.target?.tagName||"").toLowerCase(),typing=["input","textarea","select"].includes(tag),key=String(e.key||"").toLowerCase(),doc=node.__lmdDoc??parseDoc(node);if((e.ctrlKey||e.metaKey)&&key==="z"){e.preventDefault();e.shiftKey?redo(node):undo(node);}else if((e.ctrlKey||e.metaKey)&&key==="y"){e.preventDefault();redo(node);}else if(!typing&&(e.ctrlKey||e.metaKey)&&key==="c"){const hit=timelineSelectionHit(node,doc);if(hit){e.preventDefault();copyTimelineHit(node,hit);}}else if(!typing&&(e.ctrlKey||e.metaKey)&&key==="x"){const hit=timelineSelectionHit(node,doc);if(hit){e.preventDefault();cutTimelineHit(node,doc,hit);}}else if(!typing&&(e.ctrlKey||e.metaKey)&&key==="v"){if(node.__lmdTimelineClipboard){e.preventDefault();pasteTimelineClipboard(node,doc,node.__lmdPlayhead,selection(node).track);}}else if(!typing&&(e.ctrlKey||e.metaKey)&&key==="d"){const hit=timelineSelectionHit(node,doc);if(hit){e.preventDefault();duplicateTimelineHit(node,doc,hit);}}else if(!typing&&(e.key==="Delete"||e.key==="Backspace")){const hit=timelineSelectionHit(node,doc);if(hit){e.preventDefault();deleteTimelineHit(node,doc,hit);}}else if(!typing&&e.code==="Space"){e.preventDefault();startPlayback(node,doc);}else if(!typing&&e.key==="ArrowLeft"){e.preventDefault();stopPlayback(node);setPlayheadTime(node,doc,finite(node.__lmdPlayhead,0)-1);}else if(!typing&&e.key==="ArrowRight"){e.preventDefault();stopPlayback(node);setPlayheadTime(node,doc,finite(node.__lmdPlayhead,0)+1);}else if(!typing&&e.key==="Home"){e.preventDefault();stopPlayback(node);setPlayheadTime(node,doc,0);}else if(!typing&&e.key==="End"){e.preventDefault();stopPlayback(node);setPlayheadTime(node,doc,totalDuration(doc));}else if(!typing&&key==="b"){e.preventDefault();splitAllAtPlayhead(node,doc);}});root.oncontextmenu=e=>{if(e.target===root){e.preventDefault();e.stopPropagation();}};
  const ed=node.addDOMWidget("director_surface","longmedia_director",root,{serialize:false,hideOnZoom:false,getMinHeight:()=>EDITOR_MIN_HEIGHT,getMaxHeight:()=>EDITOR_MAX_HEIGHT,getValue:()=>undefined,setValue:()=>{}});ed.serialize=false;ed.serializeValue=()=>undefined;node.__lmdRoot=root;node.__lmdEditor=ed;
  const prev=node.onConfigure;node.onConfigure=function(){const r=prev?.apply(this,arguments);requestAnimationFrame(()=>{this.__lmdDoc=parseDoc(this);clampNode(this);render(this);});return r;};
  const resize=node.onResize;node.onResize=function(){const r=resize?.apply(this,arguments);if(Number(this.size?.[1])>NODE_MAX_HEIGHT)this.size[1]=NODE_MAX_HEIGHT;if(Number(this.size?.[1])<NODE_MIN_HEIGHT)this.size[1]=NODE_MIN_HEIGHT;if(Number(this.size?.[0])<NODE_MIN_WIDTH)this.size[0]=NODE_MIN_WIDTH;this.graph?.setDirtyCanvas?.(true,true);return r;};
  const removed=node.onRemoved;node.onRemoved=function(){const r=removed?.apply(this,arguments);try{stopPlayback(this);}catch(_){}try{for(const ro of this.__lmdResizeObservers||[])ro.disconnect();}catch(_){}this.__lmdResizeObservers=[];try{hideDirectorTooltip();closeTimelineContextMenu(this);invalidateDirectorMediaViews(this);}catch(_){}this.__lmdEditor=null;this.__lmdRoot=null;return r;};
  requestAnimationFrame(()=>clampNode(node));return ed;}
function clampNode(node){if(!node?.size)return;const w=Math.max(NODE_MIN_WIDTH,Number(node.size[0])||NODE_DEFAULT_WIDTH);let h=Number(node.size[1]);if(!Number.isFinite(h)||h<100||h>6000)h=NODE_HEIGHT;h=Math.max(NODE_MIN_HEIGHT,Math.min(NODE_MAX_HEIGHT,h));if(w!==Number(node.size[0])||h!==Number(node.size[1])){try{node.setSize([w,h]);}catch(_){node.size[0]=w;node.size[1]=h;}}}
function refresh(node){if(!isDirector(node))return;loadDirectorUIState(node);reconcileDirectorOutputs(node);hideWidget(widget(node,"director_json"));hideWidget(widget(node,"global_prompt"));node.__lmdDoc=parseDoc(node);if(!node.__lmdSelection?.id)node.__lmdSelection={track:"main",id:node.__lmdDoc.shots[0]?.clip_id};ensureEditor(node);render(node);if(!node.__lmdTakeStatus)setTimeout(()=>refreshTakeStatus(node,node.__lmdDoc),20);}

app.registerExtension({name:"MiniMaxH3.LongMediaDirector.v4_11_0",async beforeRegisterNodeDef(nodeType,nodeData){const c=nodeType?.comfyClass??nodeType?.ComfyClass??nodeData?.name;if(c!==DIRECTOR_CLASS)return;nodeType.category="MiniMax H3/Long Media";if(nodeData){nodeData.hidden=false;nodeData.category="MiniMax H3/Long Media";}},async nodeCreated(node){if(isDirector(node))setTimeout(()=>refresh(node),0);},async afterConfigureGraph(){for(const node of app.graph?._nodes??[])if(isDirector(node))refresh(node);}});

api.addEventListener("minimax_h3_director_preview_ready", (event) => {
  const detail=event?.detail??event,nodeId=String(detail?.node_id??"");if(!nodeId)return;const node=app.graph?._nodes?.find((n)=>String(n?.id)===nodeId&&isDirector(n));if(!node)return;clearLiveLatentPreview(node);setTimeout(()=>refreshTakeStatus(node,node.__lmdDoc??parseDoc(node)),30);
});

function routeLatentPreviewEvent(event){const detail=event?.detail??event,blob=detail?.blob instanceof Blob?detail.blob:(detail instanceof Blob?detail:null);if(!blob)return;const displayId=String(detail?.displayNodeId??detail?.node_id??app.runningNodeId??"");for(const node of app.graph?._nodes??[]){if(!isDirector(node)||!node.__lmdLatentPreviewEnabled)continue;const sampler=downstreamSampler(node);if(!sampler)continue;if(displayId&&String(sampler.id)!==displayId)continue;acceptLiveLatentPreview(node,blob);}}
api.addEventListener("b_preview_with_metadata",routeLatentPreviewEvent);
api.addEventListener("b_preview",routeLatentPreviewEvent);
for(const eventName of ["execution_error","execution_interrupted"]){api.addEventListener(eventName,()=>{for(const node of app.graph?._nodes??[])if(isDirector(node))clearLiveLatentPreview(node);});}

api.addEventListener("minimax_h3_director_regeneration_done", (event) => {
  const detail=event?.detail??event;
  const nodeId=String(detail?.node_id??"");
  const requestId=String(detail?.request_id??"");
  if(!nodeId||!requestId)return;
  const node=app.graph?._nodes?.find((n)=>String(n?.id)===nodeId&&isDirector(n));
  if(!node)return;
  const doc=parseDoc(node);
  if(String(doc.regeneration?.request_id||"")===requestId){
    doc.regeneration={mode:"none",clip_id:null,request_id:null};
    const w=widget(node,"director_json");
    if(w){w.value=JSON.stringify(normalizeDoc(doc));try{w.callback?.(w.value);}catch(_){}}
    node.__lmdDoc=normalizeDoc(doc);
  }
  node.__lmdLastRegen={
    state:"done",request_id:requestId,clip_id:String(detail?.target_clip_id||""),
    mode:String(detail?.mode||"completed"),reason:String(detail?.reason||""),
    rendered_indices:Array.isArray(detail?.rendered_indices)?detail.rendered_indices:[],
  };
  render(node);
  setTimeout(()=>refreshTakeStatus(node,node.__lmdDoc),50);
  node.setDirtyCanvas?.(true,true);
  app.canvas?.setDirty?.(true,true);
});
