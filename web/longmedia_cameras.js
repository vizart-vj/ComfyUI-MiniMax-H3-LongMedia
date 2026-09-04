import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CAMERA_CLASS = "MiniMaxH3LongMediaCameras";
const PLANNER_CLASS = "MiniMaxH3LongMediaPlanner";
const MIN_CLIPS = 2;
const MAX_CLIPS = 16;
const PLANNER_CHANGED_EVENT = "minimax-h3-longmedia-planner-changed";

const SHOT_SIZES = [
    "Extreme Wide Shot", "Wide Shot", "Full Shot", "Cowboy Shot",
    "Medium Full Shot", "Medium Shot", "Medium Close-Up", "Close-Up",
    "Extreme Close-Up", "Macro / Detail", "Over-the-Shoulder", "Two-Shot", "POV Framing",
];

const RIGS = [
    "Tripod / Locked Head", "Fluid Head Tripod", "Dolly / Track", "Slider",
    "Jib / Crane", "Technocrane", "Steadicam", "3-Axis Gimbal", "Shoulder Rig",
    "Handheld", "Vehicle Mount", "Cable Cam", "Robot Arm · Bolt", "Robot Arm · KUKA",
    "Drone · Heavy-Lift Cinema", "Drone · DJI Inspire 3", "Drone · DJI Mavic 3 Cine",
    "Drone · DJI Air 3S", "Drone · DJI Mini 4 Pro", "FPV · DJI Avata 2",
    "FPV · Cinewhoop", "FPV · Racing", "Bodycam Mount", "Helmet / Head Mount",
    "Static Security Mount",
];

const CAMERA_BODIES = [
    "Cinematic Neutral", "ARRI Alexa 35", "ARRI Alexa Mini LF", "Sony VENICE 2",
    "RED V-RAPTOR XL", "RED KOMODO-X", "Blackmagic URSA Cine 12K",
    "Sony FX3", "Sony FX6", "Canon C400", "Canon EOS R5 C",
    "Canon EOS 5D Mark II", "Nikon D850", "Sony DCR-VX1000", "Canon XL1",
    "Panasonic DVX100", "VHS Camcorder", "VHS-C Camcorder", "Sony Hi8 Handycam",
    "Super 8 Camera", "Aaton XTR 16mm", "Arricam LT 35mm", "IMAX 65mm",
    "Smartphone · Snapshot", "Smartphone · Cinematic", "Action Camera",
    "Broadcast ENG", "CCTV Sensor", "Webcam",
];

const LENSES = [
    "Auto / Native Lens",
    "Ultra-Wide 10mm", "Ultra-Wide 12mm", "Ultra-Wide 14mm",
    "Wide 18mm", "Wide 21mm", "Wide 24mm", "Wide 28mm",
    "Natural 35mm", "Natural 40mm", "Standard 50mm",
    "Portrait 65mm", "Portrait 85mm", "Telephoto 100mm", "Telephoto 135mm",
    "Long Telephoto 200mm", "Long Telephoto 300mm",
    "Macro 60mm", "Macro 100mm",
    "Anamorphic 28mm", "Anamorphic 35mm", "Anamorphic 50mm", "Anamorphic 75mm",
    "Vintage Spherical · Wide", "Vintage Spherical · Normal", "Vintage Spherical · Portrait",
    "Probe Lens", "Tilt-Shift", "Fisheye",
    "Smartphone Ultra-Wide", "Smartphone Wide", "Smartphone Tele",
];

const STABILIZATION = [
    "Rig Native", "Hard Locked", "Fluid Controlled", "Gyro Stabilized", "Gimbal Smooth",
    "Steadicam Organic", "Handheld Controlled", "Handheld Raw", "FPV Stabilized", "FPV Raw",
];

const MOVEMENTS = [
    "Locked-Off / Static",
    "Push-In", "Pull-Out",
    "Track Forward", "Track Backward", "Track Left", "Track Right",
    "Pan Left", "Pan Right", "Tilt Up", "Tilt Down",
    "Crane Up", "Crane Down", "Pedestal Up", "Pedestal Down",
    "Arc Left", "Arc Right",
    "Orbit Clockwise", "Orbit Counterclockwise",
    "Full 360 Orbit Clockwise", "Full 360 Orbit Counterclockwise",
    "Half Orbit Clockwise", "Half Orbit Counterclockwise",
    "Spiral In Clockwise", "Spiral In Counterclockwise",
    "Spiral Out Clockwise", "Spiral Out Counterclockwise",
    "Diagonal Forward Left", "Diagonal Forward Right",
    "Diagonal Backward Left", "Diagonal Backward Right",
    "Rise + Push-In", "Descend + Push-In",
    "Rise + Pull-Out", "Descend + Pull-Out",
];

const SPEEDS = ["Static", "Ultra Slow", "Slow", "Controlled", "Medium", "Fast", "Aggressive", "Variable / Ramping"];

const TRANSITION_TYPES = ["Continuous / Same Shot", "Threshold Entry", "Occluded Hidden Cut", "Hard Cut"];
const SPACE_RELATIONS = ["Same Space", "Adjacent Space", "Different Space"];
const ENTITY_CONTINUITY = ["Lock Population / Layout", "Preserve Main Subjects", "Allow Background Evolution"];


const CAMERA_PRESETS = {
    "Continuous Push-In": {
        description: "wide → medium → close, one continuous approach",
        apply(cards) {
            const shots = ["Wide Shot", "Full Shot", "Medium Shot", "Medium Close-Up"];
            cards.forEach((c, i) => {
                const t = cards.length <= 1 ? 0 : i / (cards.length - 1);
                c.shot_size = shots[Math.min(shots.length - 1, Math.round(t * (shots.length - 1)))];
                c.rig = "3-Axis Gimbal";
                c.lens = "Standard 50mm";
                c.stabilization = "Gimbal Smooth";
                c.movement = "Track Forward";
                c.speed = i === cards.length - 1 ? "Ultra Slow" : "Slow";
                c.transition_type = "Continuous / Same Shot";
                c.space_relation = "Same Space";
                c.entity_continuity = "Lock Population / Layout";
                c.transition_to_next = i < cards.length - 1;
            });
        },
    },
    "Reveal Pull-Back": {
        description: "close → wide reveal, continuous retreat",
        apply(cards) {
            const shots = ["Medium Close-Up", "Medium Shot", "Full Shot", "Wide Shot", "Extreme Wide Shot"];
            cards.forEach((c, i) => {
                const t = cards.length <= 1 ? 0 : i / (cards.length - 1);
                c.shot_size = shots[Math.min(shots.length - 1, Math.round(t * (shots.length - 1)))];
                c.rig = "3-Axis Gimbal";
                c.lens = "Standard 50mm";
                c.stabilization = "Gimbal Smooth";
                c.movement = i >= Math.ceil(cards.length * 0.65) ? "Rise + Pull-Out" : "Track Backward";
                c.speed = "Slow";
                c.transition_type = "Continuous / Same Shot";
                c.space_relation = "Same Space";
                c.entity_continuity = "Lock Population / Layout";
                c.transition_to_next = i < cards.length - 1;
            });
        },
    },
    "Ritual Orbit": {
        description: "arc into a sustained clockwise orbit",
        apply(cards) {
            const shots = ["Full Shot", "Medium Full Shot", "Medium Shot", "Medium Close-Up"];
            cards.forEach((c, i) => {
                const t = cards.length <= 1 ? 0 : i / (cards.length - 1);
                c.shot_size = shots[Math.min(shots.length - 1, Math.round(t * (shots.length - 1)))];
                c.rig = "3-Axis Gimbal";
                c.lens = "Standard 50mm";
                c.stabilization = "Gimbal Smooth";
                c.movement = i === 0 ? "Arc Right" : "Orbit Clockwise";
                c.speed = i === 0 ? "Slow" : "Ultra Slow";
                c.transition_type = "Continuous / Same Shot";
                c.space_relation = "Same Space";
                c.entity_continuity = "Lock Population / Layout";
                c.transition_to_next = i < cards.length - 1;
            });
        },
    },
    "Lateral Reveal": {
        description: "side travel that bends naturally into an arc",
        apply(cards) {
            const shots = ["Wide Shot", "Full Shot", "Medium Full Shot", "Medium Shot"];
            cards.forEach((c, i) => {
                const t = cards.length <= 1 ? 0 : i / (cards.length - 1);
                c.shot_size = shots[Math.min(shots.length - 1, Math.round(t * (shots.length - 1)))];
                c.rig = "Dolly / Track";
                c.lens = "Natural 40mm";
                c.stabilization = "Fluid Controlled";
                c.movement = i < Math.ceil(cards.length * 0.6) ? "Track Right" : "Arc Right";
                c.speed = "Slow";
                c.transition_type = "Continuous / Same Shot";
                c.space_relation = "Same Space";
                c.entity_continuity = "Lock Population / Layout";
                c.transition_to_next = i < cards.length - 1;
            });
        },
    },
    "Descent Into Scene": {
        description: "high/wide descent converted into forward travel",
        apply(cards) {
            const shots = ["Extreme Wide Shot", "Wide Shot", "Full Shot", "Medium Full Shot"];
            cards.forEach((c, i) => {
                const t = cards.length <= 1 ? 0 : i / (cards.length - 1);
                c.shot_size = shots[Math.min(shots.length - 1, Math.round(t * (shots.length - 1)))];
                c.rig = "Jib / Crane";
                c.lens = "Natural 35mm";
                c.stabilization = "Fluid Controlled";
                if (t < 0.34) c.movement = "Crane Down";
                else if (t < 0.67) c.movement = "Descend + Push-In";
                else c.movement = "Track Forward";
                c.speed = "Slow";
                c.transition_type = "Continuous / Same Shot";
                c.space_relation = "Same Space";
                c.entity_continuity = "Lock Population / Layout";
                c.transition_to_next = i < cards.length - 1;
            });
        },
    },
    "Slow Cinematic Drift": {
        description: "maximum seam safety with ultra-slow continuous travel",
        apply(cards) {
            const shots = ["Full Shot", "Medium Full Shot", "Medium Shot"];
            cards.forEach((c, i) => {
                const t = cards.length <= 1 ? 0 : i / (cards.length - 1);
                c.shot_size = shots[Math.min(shots.length - 1, Math.round(t * (shots.length - 1)))];
                c.rig = "3-Axis Gimbal";
                c.lens = "Standard 50mm";
                c.stabilization = "Gimbal Smooth";
                c.movement = "Track Forward";
                c.speed = "Ultra Slow";
                c.transition_type = "Continuous / Same Shot";
                c.space_relation = "Same Space";
                c.entity_continuity = "Lock Population / Layout";
                c.transition_to_next = i < cards.length - 1;
            });
        },
    },
    "Static Tension → Push": {
        description: "locked opening, then controlled push without direction reversal",
        apply(cards) {
            cards.forEach((c, i) => {
                c.shot_size = i === 0 ? "Wide Shot" : (i === cards.length - 1 ? "Medium Close-Up" : "Medium Shot");
                c.rig = i === 0 ? "Tripod / Locked Head" : "3-Axis Gimbal";
                c.lens = "Standard 50mm";
                c.stabilization = i === 0 ? "Hard Locked" : "Gimbal Smooth";
                c.movement = i === 0 ? "Locked-Off / Static" : "Track Forward";
                c.speed = i === 0 ? "Static" : "Ultra Slow";
                c.transition_type = "Continuous / Same Shot";
                c.space_relation = "Same Space";
                c.entity_continuity = "Lock Population / Layout";
                c.transition_to_next = i < cards.length - 1;
            });
        },
    },

    "Approach → Threshold → Interior": {
        description: "continuous exterior approach, visible threshold crossing, then interior reveal",
        apply(cards) {
            cards.forEach((c, i) => {
                const t = cards.length <= 1 ? 0 : i / (cards.length - 1);
                c.rig = "3-Axis Gimbal";
                c.lens = "Standard 50mm";
                c.stabilization = "Gimbal Smooth";
                c.entity_continuity = "Lock Population / Layout";
                c.transition_to_next = i < cards.length - 1;

                if (i === 0) {
                    c.shot_size = "Wide Shot";
                    c.movement = "Track Forward";
                    c.speed = "Slow";
                    c.transition_type = "Continuous / Same Shot";
                    c.space_relation = "Same Space";
                } else if (i === cards.length - 1) {
                    c.shot_size = "Medium Shot";
                    c.movement = "Track Forward";
                    c.speed = "Ultra Slow";
                    c.transition_type = "Continuous / Same Shot";
                    c.space_relation = "Same Space";
                } else if (i === cards.length - 2) {
                    c.shot_size = "Full Shot";
                    c.movement = "Track Forward";
                    c.speed = "Ultra Slow";
                    c.transition_type = "Threshold Entry";
                    c.space_relation = "Adjacent Space";
                } else {
                    c.shot_size = "Full Shot";
                    c.movement = "Track Forward";
                    c.speed = "Slow";
                    c.transition_type = "Continuous / Same Shot";
                    c.space_relation = "Same Space";
                }
            });
        },
    },
};

const PRESET_NAMES = ["Custom", ...Object.keys(CAMERA_PRESETS)];

function className(node) {
    return node?.comfyClass ?? node?.ComfyClass ?? node?.constructor?.comfyClass ?? node?.constructor?.ComfyClass ?? null;
}
function isCamera(node) { return className(node) === CAMERA_CLASS; }
function widget(node, name) { return node.widgets?.find((w) => w?.name === name); }

function inputConnected(node, name) {
    const input = node.inputs?.find((i) => i?.name === name);
    if (!input) return false;
    if (input.link != null) return true;
    return Array.isArray(input.links) && input.links.some((id) => id != null);
}

function linkedOrigin(node, name) {
    const input = node.inputs?.find((i) => i?.name === name);
    const linkId = input?.link ?? (Array.isArray(input?.links) ? input.links.find((id) => id != null) : null);
    if (linkId == null) return null;
    const graph = node.graph ?? app.graph;
    const links = graph?.links;
    const link = links instanceof Map ? links.get(linkId) : links?.[linkId];
    if (!link) return null;
    const originId = link.origin_id ?? link.originId;
    return graph?.getNodeById?.(originId) ?? graph?._nodes?.find((n) => String(n?.id) === String(originId)) ?? null;
}

function makeClipId() {
    try { if (globalThis.crypto?.randomUUID) return `clip-${globalThis.crypto.randomUUID()}`; } catch (_) {}
    return `clip-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function moveItem(items, from, to) {
    from = Number(from); to = Number(to);
    if (!Array.isArray(items) || from === to || from < 0 || to < 0 || from >= items.length || to >= items.length) return false;
    const [item] = items.splice(from, 1);
    items.splice(to, 0, item);
    return true;
}

function plannerClipInfo(node) {
    const origin = linkedOrigin(node, "clip_plan");
    if (!origin || className(origin) !== PLANNER_CLASS) return null;
    const storage = widget(origin, "clips_json");
    try {
        const clips = JSON.parse(String(storage?.value ?? "[]"));
        if (!Array.isArray(clips) || clips.length < 1) return null;
        let changed = false;
        const normalized = clips.slice(0, MAX_CLIPS).map((c) => {
            c = c && typeof c === "object" ? { ...c } : {};
            if (!String(c.clip_id ?? "").trim()) { c.clip_id = makeClipId(); changed = true; }
            if (c.name == null) { c.name = ""; changed = true; }
            return c;
        });
        if (changed && storage) {
            const value = JSON.stringify(normalized);
            storage.value = value;
            try { storage.callback?.(value); } catch (_) {}
            origin.__lmPlannerClips = normalized;
        }
        return normalized.map((c) => ({
            clip_id: String(c.clip_id),
            name: String(c.name ?? ""),
            duration: Number(c.duration) || 7.5,
            seed: c.seed ?? null,
        }));
    } catch (_) {
        return null;
    }
}

function reorderLinkedPlanner(node, from, to) {
    const origin = linkedOrigin(node, "clip_plan");
    if (!origin || className(origin) !== PLANNER_CLASS) return false;
    const storage = widget(origin, "clips_json");
    if (!storage) return false;
    try {
        const clips = JSON.parse(String(storage.value ?? "[]"));
        if (!Array.isArray(clips)) return false;
        clips.forEach((c) => { if (c && typeof c === "object" && !String(c.clip_id ?? "").trim()) c.clip_id = makeClipId(); });
        if (!moveItem(clips, from, to)) return false;
        const value = JSON.stringify(clips);
        storage.value = value;
        try { storage.callback?.(value); } catch (_) {}
        origin.__lmPlannerClips = clips;
        origin.__lmPlannerRaw = value;
        origin.__lmPlannerEditor?.__lmPlannerRender?.();
        try { window.dispatchEvent(new CustomEvent(PLANNER_CHANGED_EVENT, { detail: { node_id: String(origin.id ?? "") } })); } catch (_) {}
        origin.graph?.setDirtyCanvas?.(true, true);
        app.canvas?.setDirty?.(true, true);
        return true;
    } catch (_) { return false; }
}

function defaultCard() {
    return {
        clip_id: makeClipId(),
        clip_name: "",
        shot_size: "Medium Shot",
        rig: "Tripod / Locked Head",
        camera_body: "Cinematic Neutral",
        lens: "Auto / Native Lens",
        stabilization: "Rig Native",
        movement: "Locked-Off / Static",
        speed: "Static",
        transition_type: "Continuous / Same Shot",
        space_relation: "Same Space",
        entity_continuity: "Lock Population / Layout",
        transition_to_next: false,
    };
}

function parse(node) {
    const storage = widget(node, "cameras_json");
    let raw = [];
    try { raw = JSON.parse(String(storage?.value ?? "[]")); } catch (_) {}
    if (!Array.isArray(raw)) raw = [];
    const cards = raw.slice(0, MAX_CLIPS).map((item) => {
        const src = (item && typeof item === "object") ? item : {};
        const card = { ...defaultCard(), ...src };
        card.clip_id = String(src.clip_id ?? "").trim() || makeClipId();
        card.clip_name = String(src.clip_name ?? src.name ?? "").trim().slice(0, 120);
        if (!src.camera_body && src.camera_profile) card.camera_body = src.camera_profile;

        const legacyDroneRig = {
            "Drone · DJI Inspire 3": "Drone · DJI Inspire 3",
            "Drone · DJI Mavic 3 Cine": "Drone · DJI Mavic 3 Cine",
            "Drone · DJI Air 3S": "Drone · DJI Air 3S",
            "Drone · DJI Mini 4 Pro": "Drone · DJI Mini 4 Pro",
            "FPV Drone · DJI Avata 2": "FPV · DJI Avata 2",
            "FPV Drone · Racing": "FPV · Racing",
            "FPV Drone · Cinewhoop": "FPV · Cinewhoop",
            "Heavy-Lift Cinema Drone": "Drone · Heavy-Lift Cinema",
        };
        if (!src.rig && legacyDroneRig[card.camera_body]) {
            card.rig = legacyDroneRig[card.camera_body];
            card.camera_body = "Cinematic Neutral";
        }
        return card;
    });
    while (cards.length < MIN_CLIPS) cards.push({ ...(cards[cards.length - 1] ?? defaultCard()), clip_id: makeClipId() });
    return cards;
}

function commit(node, cards) {
    const storage = widget(node, "cameras_json");
    if (!storage) return;
    const normalized = cards.slice(0, MAX_CLIPS).map((c) => ({
        clip_id: String(c?.clip_id ?? "").trim() || makeClipId(),
        clip_name: String(c?.clip_name ?? "").trim().slice(0, 120),
        shot_size: SHOT_SIZES.includes(c?.shot_size) ? c.shot_size : "Medium Shot",
        rig: RIGS.includes(c?.rig) ? c.rig : "Tripod / Locked Head",
        camera_body: CAMERA_BODIES.includes(c?.camera_body) ? c.camera_body : "Cinematic Neutral",
        lens: LENSES.includes(c?.lens) ? c.lens : "Auto / Native Lens",
        stabilization: STABILIZATION.includes(c?.stabilization) ? c.stabilization : "Rig Native",
        movement: MOVEMENTS.includes(c?.movement) ? c.movement : "Locked-Off / Static",
        speed: SPEEDS.includes(c?.speed) ? c.speed : "Static",
        transition_type: TRANSITION_TYPES.includes(c?.transition_type) ? c.transition_type : "Continuous / Same Shot",
        space_relation: SPACE_RELATIONS.includes(c?.space_relation) ? c.space_relation : "Same Space",
        entity_continuity: ENTITY_CONTINUITY.includes(c?.entity_continuity) ? c.entity_continuity : "Lock Population / Layout",
        transition_to_next: Boolean(c?.transition_to_next),
    }));
    while (normalized.length < MIN_CLIPS) normalized.push({ ...(normalized[normalized.length - 1] ?? defaultCard()), clip_id: makeClipId() });
    const value = JSON.stringify(normalized);
    if (storage.value !== value) {
        storage.value = value;
        try { storage.callback?.(value); } catch (_) {}
    }
    node.__lmCameraCards = normalized;
    node.__lmCameraRaw = value;
    node.graph?.setDirtyCanvas?.(true, true);
    app.canvas?.setDirty?.(true, true);
}

function syncToCount(node, target) {
    target = Math.max(MIN_CLIPS, Math.min(MAX_CLIPS, Number(target) || MIN_CLIPS));
    const cards = node.__lmCameraCards ?? parse(node);
    while (cards.length < target) cards.push({ ...(cards[cards.length - 1] ?? defaultCard()), clip_id: makeClipId() });
    if (cards.length > target) cards.splice(target);
    commit(node, cards);
    node.__lmCameraEditor?.__lmCameraRender?.();
}

function syncToPlannerOrder(node, info = null) {
    info = info ?? plannerClipInfo(node);
    if (!Array.isArray(info) || !info.length) return false;
    const current = node.__lmCameraCards ?? parse(node);
    const unused = new Set(current.map((_, i) => i));
    const byId = new Map();
    current.forEach((card, i) => { if (String(card?.clip_id ?? "").trim()) byId.set(String(card.clip_id), i); });
    const next = info.map((clip, position) => {
        let idx = byId.get(String(clip.clip_id));
        if (idx == null || !unused.has(idx)) {
            idx = unused.has(position) ? position : [...unused][0];
        }
        let card;
        if (idx != null && unused.has(idx)) {
            unused.delete(idx);
            card = { ...current[idx] };
        } else {
            card = defaultCard();
        }
        card.clip_id = String(clip.clip_id);
        card.clip_name = String(clip.name ?? "");
        return card;
    });
    commit(node, next);
    return true;
}

function syncFromPlanner(node) {
    return syncToPlannerOrder(node, plannerClipInfo(node));
}

const HIDDEN_SIZE = () => [0, -4];
const HIDDEN_DRAW = () => {};
function setVisible(w, visible) {
    if (!w) return;
    if (!w.__lmCameraCaptured) {
        w.__lmCameraCaptured = true;
        w.__lmCameraOwnCompute = Object.prototype.hasOwnProperty.call(w, "computeSize");
        w.__lmCameraOwnDraw = Object.prototype.hasOwnProperty.call(w, "draw");
        w.__lmCameraCompute = w.computeSize;
        w.__lmCameraDraw = w.draw;
    }
    const hidden = !visible;
    w.options = { ...(w.options ?? {}), hidden };
    w.hidden = hidden;
    if (visible) {
        if (w.__lmCameraOwnCompute) w.computeSize = w.__lmCameraCompute; else { try { delete w.computeSize; } catch (_) {} }
        if (w.__lmCameraOwnDraw) w.draw = w.__lmCameraDraw; else { try { delete w.draw; } catch (_) {} }
    } else {
        w.computeSize = HIDDEN_SIZE;
        w.draw = HIDDEN_DRAW;
    }
}

function styleSelect(el) {
    Object.assign(el.style, {
        boxSizing: "border-box", width: "100%", color: "var(--input-text, #eee)",
        background: "var(--comfy-input-bg, #171717)", border: "1px solid #555",
        borderRadius: "6px", padding: "6px 7px", font: "12px sans-serif", outline: "none",
    });
    return el;
}

function labeledSelect(labelText, options, value, onChange) {
    const wrap = document.createElement("div");
    const label = document.createElement("div");
    label.textContent = labelText;
    Object.assign(label.style, { fontSize: "10px", marginBottom: "3px", opacity: "0.82" });
    const select = styleSelect(document.createElement("select"));
    for (const option of options) {
        const el = document.createElement("option");
        el.value = option; el.textContent = option; select.append(el);
    }
    select.value = options.includes(value) ? value : options[0];
    select.addEventListener("change", () => onChange(select.value));
    wrap.append(label, select);
    return wrap;
}

function transitionToggle(value, disabled, onChange) {
    const wrap = document.createElement("label");
    Object.assign(wrap.style, {
        gridColumn: "1 / -1",
        display: "flex",
        alignItems: "center",
        gap: "8px",
        borderTop: "1px solid rgba(255,255,255,0.10)",
        paddingTop: "8px",
        marginTop: "1px",
        fontSize: "11px",
        cursor: disabled ? "default" : "pointer",
        opacity: disabled ? "0.45" : "0.95",
        userSelect: "none",
    });
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(value) && !disabled;
    input.disabled = Boolean(disabled);
    input.addEventListener("change", () => onChange(Boolean(input.checked)));
    const text = document.createElement("span");
    text.textContent = disabled ? "Transition Contract · final clip" : "Transition to Next Clip";
    const hint = document.createElement("span");
    hint.textContent = disabled ? "" : "camera + scene + population continuity";
    Object.assign(hint.style, { opacity: "0.58", marginLeft: "auto", fontSize: "10px" });
    wrap.append(input, text, hint);
    return wrap;
}

function ensureEditor(node) {
    if (node.__lmCameraEditor) return node.__lmCameraEditor;
    if (typeof node.addDOMWidget !== "function") return null;

    const root = document.createElement("div");
    Object.assign(root.style, {
        width: "100%", height: "100%", minWidth: "0", minHeight: "0",
        boxSizing: "border-box", padding: "4px 2px 6px", fontFamily: "sans-serif",
        color: "#ddd", display: "flex", flexDirection: "column", overflow: "hidden", contain: "layout paint style",
    });

    const toolbar = document.createElement("div");
    Object.assign(toolbar.style, {
        display: "flex", flex: "0 0 auto", gap: "6px", alignItems: "center", flexWrap: "wrap",
        marginBottom: "8px", minWidth: "0",
    });
    const sync = document.createElement("button"); sync.textContent = "Sync Planner";
    const add = document.createElement("button"); add.textContent = "+ Add Clip";
    const remove = document.createElement("button"); remove.textContent = "− Remove Last";
    const preset = document.createElement("select");
    const applyPreset = document.createElement("button"); applyPreset.textContent = "Apply Motion Preset";
    for (const name of PRESET_NAMES) {
        const option = document.createElement("option"); option.value = name; option.textContent = name; preset.append(option);
    }
    preset.value = "Custom";
    styleSelect(preset);
    Object.assign(preset.style, { width: "170px", padding: "3px 6px", fontSize: "11px" });
    const count = document.createElement("span");
    const status = document.createElement("span");
    for (const b of [sync, add, remove, applyPreset]) Object.assign(b.style, {
        color: "#eee", background: "#202020", border: "1px solid #777",
        borderRadius: "4px", padding: "3px 7px", cursor: "pointer", fontSize: "11px",
    });
    Object.assign(count.style, { fontSize: "11px", opacity: "0.8" });
    Object.assign(status.style, { fontSize: "11px", opacity: "0.72", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" });
    toolbar.append(sync, add, remove, preset, applyPreset, count, status);

    const viewport = document.createElement("div");
    Object.assign(viewport.style, {
        flex: "1 1 auto", minWidth: "0", minHeight: "0", width: "100%",
        boxSizing: "border-box", overflowX: "auto", overflowY: "auto",
        scrollbarGutter: "stable", overscrollBehavior: "contain",
    });
    const cards = document.createElement("div");
    Object.assign(cards.style, {
        display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(360px, 1fr))",
        gridAutoRows: "max-content", alignContent: "start", gap: "10px",
        width: "100%", minWidth: "730px", boxSizing: "border-box", padding: "0 2px 2px 0", contain: "layout paint style",
    });
    viewport.append(cards); root.append(toolbar, viewport);

    function render() {
        let plannerInfo = plannerClipInfo(node);
        const plannerConnected = inputConnected(node, "clip_plan");
        const autoSync = Boolean(widget(node, "auto_sync_planner")?.value);
        if (plannerInfo && autoSync) {
            syncToPlannerOrder(node, plannerInfo);
            plannerInfo = plannerClipInfo(node);
        }

        const cameraCards = node.__lmCameraCards ?? parse(node);
        node.__lmCameraCards = cameraCards;
        count.textContent = `${cameraCards.length} clips`;
        sync.disabled = !plannerConnected;
        add.disabled = plannerConnected || cameraCards.length >= MAX_CLIPS;
        remove.disabled = plannerConnected || cameraCards.length <= MIN_CLIPS;
        status.textContent = plannerConnected ? (autoSync ? "order + names inherited from Planner" : "Planner connected · manual camera order") : "standalone";

        cards.replaceChildren();
        cameraCards.forEach((cardData, index) => {
            const card = document.createElement("div");
            Object.assign(card.style, {
                border: `1px solid ${index === 0 ? "#d9a400" : "#397db0"}`,
                borderRadius: "9px", padding: "9px", background: "rgba(10,10,10,0.45)", minWidth: "0", transition: "opacity 100ms ease, outline 100ms ease", contain: "layout paint style",
            });
            const info = plannerInfo?.[index];
            const header = document.createElement("div");
            header.draggable = true;
            Object.assign(header.style, { display: "grid", gridTemplateColumns: "auto auto minmax(80px, 1fr)", gap: "7px", alignItems: "center", marginBottom: "8px", cursor: "grab", userSelect: "none" });
            const grip = document.createElement("span"); grip.textContent = "⠿"; grip.title = plannerConnected && autoSync ? "Drag to reorder this clip in Planner and Cameras" : "Drag camera card to reorder"; Object.assign(grip.style, { opacity: "0.72", fontSize: "17px", lineHeight: "1" });
            const ordinal = document.createElement("span"); ordinal.textContent = `CLIP ${index + 1}`; Object.assign(ordinal.style, { fontWeight: "700", fontSize: "13px", whiteSpace: "nowrap" });
            const titleText = document.createElement("div");
            const displayName = String(info?.name ?? cardData.clip_name ?? "").trim();
            titleText.textContent = `${displayName || "Untitled"}${info ? `  ·  ${Number(info.duration).toFixed(1)}s` : ""}`;
            Object.assign(titleText.style, { fontSize: "11px", opacity: displayName ? "0.92" : "0.55", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" });
            header.append(grip, ordinal, titleText);

            if (!plannerConnected) {
                const standaloneName = styleSelect(document.createElement("input")); standaloneName.type = "text"; standaloneName.placeholder = "Clip name"; standaloneName.value = cardData.clip_name ?? ""; standaloneName.draggable = false;
                Object.assign(standaloneName.style, { gridColumn: "1 / -1", padding: "4px 7px", fontSize: "11px" });
                standaloneName.addEventListener("dragstart", (ev) => ev.stopPropagation());
                standaloneName.addEventListener("change", () => { cardData.clip_name = standaloneName.value.trim().slice(0, 120); commit(node, cameraCards); render(); });
                header.append(standaloneName);
            }

            const grid = document.createElement("div");
            Object.assign(grid.style, { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" });
            grid.append(
                labeledSelect("Shot Size / Крупность", SHOT_SIZES, cardData.shot_size, (v) => { preset.value = "Custom"; cardData.shot_size = v; commit(node, cameraCards); }),
                labeledSelect("Rig / Support", RIGS, cardData.rig, (v) => { preset.value = "Custom"; cardData.rig = v; commit(node, cameraCards); }),
                labeledSelect("Camera Body", CAMERA_BODIES, cardData.camera_body, (v) => { preset.value = "Custom"; cardData.camera_body = v; commit(node, cameraCards); }),
                labeledSelect("Lens / Объектив", LENSES, cardData.lens, (v) => { preset.value = "Custom"; cardData.lens = v; commit(node, cameraCards); }),
                labeledSelect("Stabilization", STABILIZATION, cardData.stabilization, (v) => { preset.value = "Custom"; cardData.stabilization = v; commit(node, cameraCards); }),
                labeledSelect("Movement Path", MOVEMENTS, cardData.movement, (v) => {
                    preset.value = "Custom";
                    cardData.movement = v;
                    if (v === "Locked-Off / Static") cardData.speed = "Static";
                    commit(node, cameraCards); render();
                }),
                labeledSelect("Movement Intensity", SPEEDS, cardData.speed, (v) => { preset.value = "Custom"; cardData.speed = v; commit(node, cameraCards); }),
                labeledSelect("Transition Type", TRANSITION_TYPES, cardData.transition_type, (v) => {
                    preset.value = "Custom";
                    cardData.transition_type = v;
                    if (v === "Threshold Entry") cardData.space_relation = "Adjacent Space";
                    if (v === "Hard Cut") cardData.space_relation = "Different Space";
                    commit(node, cameraCards); render();
                }),
                labeledSelect("Space Relation", SPACE_RELATIONS, cardData.space_relation, (v) => { preset.value = "Custom"; cardData.space_relation = v; commit(node, cameraCards); }),
                labeledSelect("Entity Continuity", ENTITY_CONTINUITY, cardData.entity_continuity, (v) => { preset.value = "Custom"; cardData.entity_continuity = v; commit(node, cameraCards); }),
                transitionToggle(cardData.transition_to_next, index === cameraCards.length - 1, (v) => { cardData.transition_to_next = v; commit(node, cameraCards); render(); }),
            );

            header.addEventListener("dragstart", (ev) => {
                node.__lmCameraDragIndex = index;
                card.style.opacity = "0.58";
                try { ev.dataTransfer.effectAllowed = "move"; ev.dataTransfer.setData("text/plain", String(index)); } catch (_) {}
            });
            header.addEventListener("dragend", () => { node.__lmCameraDragIndex = null; card.style.opacity = "1"; for (const c of cards.children) c.style.outline = ""; });
            card.addEventListener("dragover", (ev) => { if (node.__lmCameraDragIndex == null) return; ev.preventDefault(); try { ev.dataTransfer.dropEffect = "move"; } catch (_) {} card.style.outline = "2px solid rgba(217,164,0,0.75)"; });
            card.addEventListener("dragleave", () => { card.style.outline = ""; });
            card.addEventListener("drop", (ev) => {
                ev.preventDefault(); ev.stopPropagation(); card.style.outline = "";
                const from = Number(node.__lmCameraDragIndex), to = index;
                node.__lmCameraDragIndex = null;
                if (from === to) return;
                if (plannerConnected && autoSync) {
                    if (!reorderLinkedPlanner(node, from, to)) { status.textContent = "Planner reorder failed"; return; }
                    plannerInfo = plannerClipInfo(node);
                    syncToPlannerOrder(node, plannerInfo);
                    status.textContent = `moved Planner clip ${from + 1} → ${to + 1}`;
                    render();
                    return;
                }
                if (moveItem(cameraCards, from, to)) {
                    commit(node, cameraCards);
                    status.textContent = `moved camera ${from + 1} → ${to + 1}`;
                    render();
                }
            });

            card.append(header, grid); cards.append(card);
        });

        requestAnimationFrame(() => {
            node.setDirtyCanvas?.(true, true);
            app.canvas?.setDirty?.(true, true);
        });
    }

    applyPreset.addEventListener("click", (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        const name = preset.value;
        const def = CAMERA_PRESETS[name];
        if (!def) { status.textContent = "select a motion preset"; return; }
        const current = node.__lmCameraCards ?? parse(node);
        def.apply(current);
        commit(node, current);
        status.textContent = `${name}: ${def.description}`;
        render();
    });

    sync.addEventListener("click", (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        if (syncFromPlanner(node)) { status.textContent = "synced by clip id"; render(); return; }
        if (inputConnected(node, "clip_plan")) {
            const request = widget(node, "sync_request");
            if (request) {
                request.value = true;
                try { request.callback?.(true); } catch (_) {}
                status.textContent = "sync queued — run workflow once";
            }
        } else status.textContent = "connect LongMedia Planner to sync";
    });

    add.addEventListener("click", (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        if (inputConnected(node, "clip_plan")) return;
        const current = node.__lmCameraCards ?? parse(node);
        if (current.length >= MAX_CLIPS) return;
        current.push({ ...(current[current.length - 1] ?? defaultCard()), clip_id: makeClipId(), clip_name: "" });
        commit(node, current); render();
    });
    remove.addEventListener("click", (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        if (inputConnected(node, "clip_plan")) return;
        const current = node.__lmCameraCards ?? parse(node);
        if (current.length <= MIN_CLIPS) return;
        current.pop(); commit(node, current); render();
    });

    const editor = node.addDOMWidget("camera_editor", "camera_editor", root, {
        serialize: false, hideOnZoom: true, getValue: () => null, setValue: () => {},
    });
    node.__lmCameraViewportHeight = Number(node.__lmCameraViewportHeight) || 300;
    editor.computeSize = function(width) {
        const w = Math.max(180, Number(width) || Number(node.size?.[0]) || 700);
        const h = Math.max(120, Number(node.__lmCameraViewportHeight) || 300);
        return [w, h];
    };
    editor.__lmCameraRender = render;
    node.__lmCameraEditor = editor;
    node.__lmCameraCards = parse(node);
    if (Boolean(widget(node, "auto_sync_planner")?.value) && plannerClipInfo(node)) syncToPlannerOrder(node);
    else commit(node, node.__lmCameraCards); // migrate stable ids and preserve all camera fields
    setVisible(widget(node, "cameras_json"), false);
    setVisible(widget(node, "sync_request"), false);

    if (!Number.isFinite(node.__lmCameraChromeHeight)) {
        try {
            const total = node.computeSize?.();
            const totalH = Number(total?.[1]);
            if (Number.isFinite(totalH)) node.__lmCameraChromeHeight = Math.max(40, totalH - node.__lmCameraViewportHeight);
        } catch (_) {}
        if (!Number.isFinite(node.__lmCameraChromeHeight)) node.__lmCameraChromeHeight = 72;
    }
    if (!node.__lmCameraComputeSizeHooked && typeof node.computeSize === "function") {
        node.__lmCameraComputeSizeHooked = true;
        const previousComputeSize = node.computeSize;
        node.computeSize = function(...args) {
            const saved = this.__lmCameraViewportHeight;
            try { this.__lmCameraViewportHeight = 120; return previousComputeSize.apply(this, args); }
            finally { this.__lmCameraViewportHeight = saved; }
        };
    }
    if (!node.__lmCameraResizeHooked) {
        node.__lmCameraResizeHooked = true;
        const previousOnResize = node.onResize;
        node.onResize = function(size) {
            try { previousOnResize?.apply(this, arguments); } catch (_) {}
            const h = Number(size?.[1]), chrome = Number(this.__lmCameraChromeHeight);
            if (Number.isFinite(h) && Number.isFinite(chrome)) this.__lmCameraViewportHeight = Math.max(120, h - chrome);
            try { this.setDirtyCanvas?.(true, true); } catch (_) {}
        };
    }
    render();
    return editor;
}

function refresh(node) {
    if (!isCamera(node)) return;
    setVisible(widget(node, "cameras_json"), false);
    setVisible(widget(node, "sync_request"), false);
    const auto = widget(node, "auto_sync_planner");
    if (auto) { auto.label = "Auto Sync Planner"; auto.localized_name = "Auto Sync Planner"; }
    const raw = String(widget(node, "cameras_json")?.value ?? "");
    if (raw !== node.__lmCameraRaw) {
        node.__lmCameraRaw = raw;
        node.__lmCameraCards = parse(node);
    }
    const editor = ensureEditor(node);
    if (Boolean(widget(node, "auto_sync_planner")?.value)) syncFromPlanner(node);
    editor?.__lmCameraRender?.();
}

app.registerExtension({
    name: "MiniMaxH3.LongMediaCameras.v2",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const cls = nodeType?.comfyClass ?? nodeType?.ComfyClass ?? nodeData?.name;
        if (cls !== CAMERA_CLASS) return;
        nodeType.category = "MiniMax H3/Long Media";
        if (nodeData) { nodeData.hidden = false; nodeData.category = "MiniMax H3/Long Media"; }
    },
    async nodeCreated(node) { if (isCamera(node)) setTimeout(() => refresh(node), 0); },
    async afterConfigureGraph() {
        for (const node of app.graph?._nodes ?? []) if (isCamera(node)) refresh(node);
    },
});

window.addEventListener(PLANNER_CHANGED_EVENT, (event) => {
    const plannerId = String(event?.detail?.node_id ?? "");
    for (const node of app.graph?._nodes ?? []) {
        if (!isCamera(node) || !Boolean(widget(node, "auto_sync_planner")?.value)) continue;
        const origin = linkedOrigin(node, "clip_plan");
        if (!origin || className(origin) !== PLANNER_CLASS) continue;
        if (plannerId && String(origin.id ?? "") !== plannerId) continue;
        syncToPlannerOrder(node);
        node.__lmCameraEditor?.__lmCameraRender?.();
    }
});

api.addEventListener("minimax_h3_cameras_sync", (event) => {
    const detail = event?.detail ?? event;
    const nodeId = String(detail?.node_id ?? "");
    if (!nodeId) return;
    const node = app.graph?._nodes?.find((n) => String(n?.id) === nodeId && isCamera(n));
    if (!node) return;
    if (detail?.clear_request) {
        const request = widget(node, "sync_request");
        if (request) { request.value = false; try { request.callback?.(false); } catch (_) {} }
    }
    if (Number.isFinite(Number(detail?.clip_count))) {
        syncToCount(node, Number(detail.clip_count));
        refresh(node);
    }
});
