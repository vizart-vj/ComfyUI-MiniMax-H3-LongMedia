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

const COMBOS = {
    duration_source: { values: ["auto", "manual", "audio", "video", "longest_input"], fallback: "auto" },
    resolution_mode: { values: ["match", "max"], fallback: "match" },
    reference_budget: { values: ["low", "medium", "high", "max"], fallback: "low" },
    video_mode: { values: ["auto", "preserve", "transform"], fallback: "auto" },
    audio_mode: { values: ["auto", "preserve", "generate", "reference_only"], fallback: "auto" },
    generation_mode: { values: ["auto", "lip_sync"], fallback: "auto" },
    first_frame_mode: { values: ["latent_inject", "pixel_override", "blend"], fallback: "latent_inject" },
};

const NUMBERS = {
    width: 512,
    height: 512,
    manual_duration: 5.0,
    segment_seconds: 8.0,
    overlap_frames: 22,
    video_fps: 24.0,
    video_strength: 0.5,
    audio_strength: 0.0,
    first_frame_denoise: 0.25,
    first_frame_blend_frames: 3,
};

function repairCorruptWidgetValues(node) {
    // Recovery only: valid values are never touched. This repairs workflows saved
    // while an older JS build had shifted widgets_values (e.g. generation_mode=0).
    for (const [name, spec] of Object.entries(COMBOS)) {
        const w = widget(node, name);
        if (!w) continue;
        if (!spec.values.includes(w.value)) w.value = spec.fallback;
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
    name: "MiniMaxH3LatentLab.LongMediaDynamicInputs.v3",

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
    },
});
