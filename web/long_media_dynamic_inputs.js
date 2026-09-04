import { app } from "../../scripts/app.js";

const NODE_CLASS = "MiniMaxH3LatentLabLongMediaSetup";

const FAMILIES = [
    { prefix: "image_", type: "IMAGE", max: 9 },
    { prefix: "video_", type: "IMAGE", max: 3 },
    { prefix: "audio_", type: "AUDIO", max: 3 },
];

// This extension intentionally does NOT hide, disable, redraw, resize individual
// widgets, or replace widget callbacks. Python owns widget schema + serialization.
// JS owns only dynamic media sockets and recovery of clearly invalid saved values.

let configuringGraph = false;
const pending = new WeakSet();

function className(node) {
    return node?.comfyClass
        ?? node?.ComfyClass
        ?? node?.constructor?.comfyClass
        ?? node?.constructor?.ComfyClass
        ?? null;
}

function isSetup(node) {
    return className(node) === NODE_CLASS;
}

function suffix(name, prefix) {
    const text = String(name ?? "");
    if (!text.startsWith(prefix)) return null;
    const tail = text.slice(prefix.length);
    if (!/^\d+$/.test(tail)) return null;
    return Number.parseInt(tail, 10);
}

function isConnected(input) {
    if (!input) return false;
    if (input.link != null) return true;
    return Array.isArray(input.links) && input.links.some((id) => id != null);
}

function familyInputs(node, family) {
    return (node.inputs ?? [])
        .filter((input) => suffix(input?.name, family.prefix) != null)
        .sort((a, b) => suffix(a.name, family.prefix) - suffix(b.name, family.prefix));
}

function updateLinkSlots(node) {
    const links = app.graph?.links;
    if (!links) return;
    (node.inputs ?? []).forEach((input, slot) => {
        const ids = Array.isArray(input?.links) ? input.links : [input?.link];
        for (const id of ids) {
            if (id == null) continue;
            const link = links[id];
            if (link) link.target_slot = slot;
        }
    });
}

function insertFamilyInput(node, family, index) {
    const name = `${family.prefix}${index}`;
    if ((node.inputs ?? []).some((input) => input?.name === name)) return false;

    // LiteGraph has no stable cross-version insertInput API. addInput is stable;
    // move the newly-created slot only when needed and then repair target_slot.
    node.addInput(name, family.type);
    const inputs = node.inputs ?? [];
    const createdIndex = inputs.findIndex((input) => input?.name === name);
    if (createdIndex < 0) return true;

    const familyOrder = FAMILIES.findIndex((item) => item.prefix === family.prefix);
    let insertAt = inputs.length - 1;

    // Keep image_* before video_* before audio_* without touching static sockets.
    for (let i = 0; i < inputs.length; i += 1) {
        const input = inputs[i];
        const laterFamily = FAMILIES.findIndex((item) => suffix(input?.name, item.prefix) != null);
        if (laterFamily > familyOrder) {
            insertAt = i;
            break;
        }
    }

    if (createdIndex !== insertAt) {
        const [created] = inputs.splice(createdIndex, 1);
        // Removing an element before the desired position shifts the index by one.
        const adjusted = createdIndex < insertAt ? insertAt - 1 : insertAt;
        inputs.splice(adjusted, 0, created);
    }
    updateLinkSlots(node);
    return true;
}

function normalizeFamily(node, family) {
    let inputs = familyInputs(node, family);
    const highestConnected = inputs.reduce((highest, input) => {
        if (!isConnected(input)) return highest;
        return Math.max(highest, suffix(input.name, family.prefix) ?? 0);
    }, 0);

    // Always expose #1. Once #N is connected, expose #N+1 up to the family max.
    const desired = Math.min(family.max, Math.max(1, highestConnected + 1));
    let changed = false;

    for (let i = 1; i <= desired; i += 1) {
        changed = insertFamilyInput(node, family, i) || changed;
    }

    // Re-read after additions and remove ONLY unconnected tail sockets.
    inputs = familyInputs(node, family);
    for (let i = inputs.length - 1; i >= 0; i -= 1) {
        const input = inputs[i];
        const n = suffix(input.name, family.prefix);
        if (n == null || n <= desired || isConnected(input)) continue;
        const slot = (node.inputs ?? []).indexOf(input);
        if (slot >= 0) {
            node.removeInput(slot);
            changed = true;
        }
    }

    if (changed) updateLinkSlots(node);
    return changed;
}

function widget(node, name) {
    return (node.widgets ?? []).find((item) => item?.name === name);
}

function setInputDisplay(input, label) {
    if (!input) return;
    if (input.__lmOrigName === undefined) input.__lmOrigName = input.name;
    if (input.__lmOrigLabel === undefined) input.__lmOrigLabel = input.label ?? input.name;
    const display = label ?? input.__lmOrigLabel ?? input.__lmOrigName ?? input.name;
    input.label = display;
    input.localized_name = display;
}

function refreshSocketLabels(node) {
    const get = (name) => (node.inputs ?? []).find((input) => input?.name === name);
    const pictures = Array.from({ length: 9 }, (_, i) => get(`image_${i + 1}`));
    const videos = Array.from({ length: 3 }, (_, i) => get(`video_${i + 1}`));
    const audios = Array.from({ length: 3 }, (_, i) => get(`audio_${i + 1}`));

    const control = widget(node, 'control_mode')?.value ?? 'auto';
    const h3Mode = widget(node, 'h3_mode')?.value ?? 'hybrid';
    const timeline = widget(node, 'timeline_mode')?.value ?? 'single';

    pictures.forEach((input, idx) => setInputDisplay(input, `image_${idx + 1} • picture_${idx + 1}`));
    videos.forEach((input, idx) => setInputDisplay(input, `video_${idx + 1}`));
    audios.forEach((input, idx) => setInputDisplay(input, `audio_${idx + 1}`));

    if (control !== 'manual') {
        if (h3Mode === 'fl2va') {
            setInputDisplay(pictures[0], 'image_1 • first_frame');
            setInputDisplay(pictures[1], 'image_2 • last_frame (optional)');
            for (let i = 2; i < pictures.length; i += 1) setInputDisplay(pictures[i], `image_${i + 1} • inactive in FL2VA`);
        } else if (h3Mode === 'hybrid') {
            setInputDisplay(pictures[0], 'image_1 • first_frame');
            setInputDisplay(pictures[1], 'image_2 • last_frame / optional');
            for (let i = 2; i < pictures.length; i += 1) setInputDisplay(pictures[i], `image_${i + 1} • picture_${i - 1}`);
        } else if (h3Mode === 't2va') {
            pictures.forEach((input, idx) => setInputDisplay(input, `image_${idx + 1} • inactive in T2VA`));
            videos.forEach((input, idx) => setInputDisplay(input, `video_${idx + 1} • inactive in T2VA`));
        } else if (h3Mode === 'video_ref_edit') {
            setInputDisplay(videos[0], 'video_1 • source motion/camera');
        }
    }

    const clipPlan = get('clip_plan');
    if (clipPlan) setInputDisplay(clipPlan, timeline === 'multiclip' ? 'clip_plan • active' : 'clip_plan • ignored');

    const audioMode = widget(node, 'audio_mode')?.value ?? 'auto';
    if (audioMode === 'lip_sync') {
        setInputDisplay(audios[0], 'audio_1 • lip_sync');
    } else if (h3Mode === 'video_ref_edit' && ['auto', 'preserve', 'preserve_reference'].includes(audioMode)) {
        setInputDisplay(audios[0], 'audio_1 • source soundtrack + sync');
    }
}

const COMBOS = {
    duration_source: { values: ["auto", "manual", "audio", "video", "longest_input"], fallback: "auto" },
    resolution_mode: { values: ["match", "max"], fallback: "match" },
    reference_budget: { values: ["low", "medium", "high", "max"], fallback: "low" },
    video_mode: { values: ["auto", "preserve", "transform"], fallback: "auto" },
    audio_mode: { values: ["auto", "preserve", "generate", "reference_only", "preserve_reference", "lip_sync"], fallback: "auto" },
    generation_mode: { values: ["auto", "lip_sync"], fallback: "auto" },
    first_frame_mode: { values: ["native_keyframe", "latent_inject", "pixel_override", "blend"], fallback: "latent_inject" },
    conditioning_mode: { values: ["auto_refs", "hybrid_first_frame", "hybrid_first_last", "multiclip_ref2va"], fallback: "auto_refs" },
    workflow_mode: { values: ["hybrid_auto", "segmented_continuation", "multiclip", "reconstruct", "ref2va_full", "loop", "manual", "video_ref_edit"], fallback: "hybrid_auto" },
    control_mode: { values: ["auto", "manual"], fallback: "auto" },
    h3_mode: { values: ["t2va", "fl2va", "ref2va", "hybrid", "video_ref_edit"], fallback: "hybrid" },
    timeline_mode: { values: ["single", "segmented", "multiclip"], fallback: "single" },
};

const NUMBERS = {
    width: 512,
    height: 512,
    manual_duration: 5.0,
    segment_seconds: 8.0,
    overlap_frames: 22,
    transition_frames: 22,
    loop_closure_frames: 57,
    loop_closure_strength: 0.65,
    video_fps: 24.0,
    first_frame_denoise: 0.25,
    first_frame_blend_frames: 3,
};

function repairCorruptWidgetValues(node) {
    // Migrate the one-release 0.5.23 semantic sentinels before generic combo
    // repair. This must live here as well as in node_facade.js because frontend
    // extension callback ordering is not guaranteed.
    const legacyWorkflow = String(widget(node, "workflow_mode")?.value ?? "hybrid_auto");
    const semanticMap = {
        hybrid_auto: ["auto", "hybrid", "single"],
        segmented_continuation: ["auto", "ref2va", "segmented"],
        multiclip: ["auto", "ref2va", "multiclip"],
        ref2va_full: ["auto", "ref2va", "single"],
        video_ref_edit: ["auto", "video_ref_edit", "single"],
        manual: ["manual", "hybrid", "segmented"],
        loop: ["auto", "hybrid", "single"],
        reconstruct: ["auto", "video_ref_edit", "segmented"],
    };
    const migrated = semanticMap[legacyWorkflow] ?? semanticMap.hybrid_auto;
    const control = widget(node, "control_mode");
    const h3 = widget(node, "h3_mode");
    const timeline = widget(node, "timeline_mode");
    if (control?.value === "legacy") control.value = migrated[0];
    if (h3?.value === "legacy") h3.value = migrated[1];
    if (timeline?.value === "legacy") timeline.value = migrated[2];

    // Recovery only: valid values are never touched. This repairs workflows saved
    // while an older JS build had shifted widgets_values (e.g. generation_mode=0).
    for (const [name, spec] of Object.entries(COMBOS)) {
        const w = widget(node, name);
        if (!w) continue;
        if (spec.values.includes(w.value)) continue;
        if (Number.isInteger(w.value) && w.value >= 0 && w.value < spec.values.length) {
            w.value = spec.values[w.value];
        } else {
            w.value = spec.fallback;
        }
    }
    for (const [name, fallback] of Object.entries(NUMBERS)) {
        const w = widget(node, name);
        if (!w) continue;
        if (typeof w.value !== "number" || !Number.isFinite(w.value)) w.value = fallback;
    }
}

function fitNode(node) {
    const computed = node.computeSize?.();
    if (!computed || !Array.isArray(computed)) return;
    const width = Math.max(node.size?.[0] ?? 0, computed[0], 420);
    // Exact computed height intentionally allows recovery from giant persisted sizes.
    const height = computed[1];
    if (Number.isFinite(width) && Number.isFinite(height) && height > 0) {
        node.setSize?.([width, height]);
    }
}

function syncNode(node, { repairWidgets = false, fit = true } = {}) {
    if (!isSetup(node)) return;
    if (repairWidgets) repairCorruptWidgetValues(node);

    let changed = false;
    for (const family of FAMILIES) changed = normalizeFamily(node, family) || changed;
    refreshSocketLabels(node);
    if (fit || changed) fitNode(node);
    app.graph?.setDirtyCanvas?.(true, true);
}

function scheduleSync(node) {
    if (!isSetup(node) || configuringGraph || pending.has(node)) return;
    pending.add(node);
    setTimeout(() => {
        pending.delete(node);
        if (!configuringGraph && node.graph) syncNode(node, { repairWidgets: false, fit: true });
    }, 0);
}

app.registerExtension({
    name: "MiniMaxH3LatentLab.LongMediaDynamicInputs.v4",

    async beforeConfigureGraph() {
        configuringGraph = true;
    },

    async afterConfigureGraph() {
        configuringGraph = false;
        for (const node of app.graph?._nodes ?? []) {
            if (isSetup(node)) syncNode(node, { repairWidgets: true, fit: true });
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const cls = nodeType?.comfyClass ?? nodeType?.ComfyClass ?? nodeData?.name;
        if (cls !== NODE_CLASS) return;

        const original = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = original?.apply(this, arguments);
            scheduleSync(this);
            return result;
        };
    },

    async nodeCreated(node) {
        if (!isSetup(node) || configuringGraph) return;
        // New node only. Workflow-loaded nodes are normalized in afterConfigureGraph,
        // after saved links and widget values have been restored.
        scheduleSync(node);
        // node_facade.js owns workflow/generation/conditioning callbacks and resizing.
        // Dynamic inputs react only to connection changes, preventing two async
        // callback chains from racing over the same Setup node size/labels.
    },
});
