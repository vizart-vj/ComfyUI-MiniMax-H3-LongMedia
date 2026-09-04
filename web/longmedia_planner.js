import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PLANNER_CLASS = "MiniMaxH3LongMediaPlanner";
const MIN_CLIPS = 2;
const MAX_CLIPS = 16;
const CLIP_PRESET_STORAGE_KEY = "MiniMaxH3.LongMedia.clipPresets.v1";
const PLANNER_CHANGED_EVENT = "minimax-h3-longmedia-planner-changed";

function className(node) {
    return node?.comfyClass ?? node?.ComfyClass ?? node?.constructor?.comfyClass ?? node?.constructor?.ComfyClass ?? null;
}

function isPlanner(node) { return className(node) === PLANNER_CLASS; }
function widget(node, name) { return node.widgets?.find((w) => w?.name === name); }

function inputConnected(node, name) {
    const input = node.inputs?.find((i) => i?.name === name);
    if (!input) return false;
    if (input.link != null) return true;
    return Array.isArray(input.links) && input.links.some((id) => id != null);
}

function makeClipId() {
    try { if (globalThis.crypto?.randomUUID) return `clip-${globalThis.crypto.randomUUID()}`; } catch (_) {}
    return `clip-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function normalizeClip(item, fallbackDuration = 7.5) {
    item = item && typeof item === "object" ? item : {};
    let duration = Number(item.duration);
    if (!Number.isFinite(duration)) duration = Number(fallbackDuration) || 7.5;
    duration = Math.max(0.25, Math.min(150, duration));
    let seed = item.seed;
    if (seed === "" || seed === undefined) seed = null;
    if (seed !== null) {
        const n = Number(seed);
        seed = Number.isFinite(n) ? Math.max(0, Math.trunc(n)) : null;
    }
    return {
        clip_id: String(item.clip_id ?? item.id ?? "").trim() || makeClipId(),
        name: String(item.name ?? item.clip_name ?? "").trim().slice(0, 120),
        prompt: String(item.prompt ?? ""),
        duration,
        seed,
    };
}

function newClip(duration = 7.5, source = null) {
    const base = normalizeClip(source ?? {}, duration);
    base.clip_id = makeClipId();
    return base;
}

function formatTimelineTime(seconds) {
    const totalMs = Math.max(0, Math.round((Number(seconds) || 0) * 1000));
    const ms = totalMs % 1000;
    const totalSec = Math.floor(totalMs / 1000);
    const sec = totalSec % 60;
    const totalMin = Math.floor(totalSec / 60);
    const min = totalMin % 60;
    const hour = Math.floor(totalMin / 60);
    const core = hour > 0
        ? `${String(hour).padStart(2, "0")}:${String(min).padStart(2, "0")}:${String(sec).padStart(2, "0")}`
        : `${String(min).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
    return `${core}.${String(ms).padStart(3, "0")}`;
}

function moveItem(items, from, to) {
    from = Number(from); to = Number(to);
    if (!Array.isArray(items) || from === to || from < 0 || to < 0 || from >= items.length || to >= items.length) return false;
    const [item] = items.splice(from, 1);
    items.splice(to, 0, item);
    return true;
}

function notifyPlannerChanged(node) {
    try {
        window.dispatchEvent(new CustomEvent(PLANNER_CHANGED_EVENT, { detail: { node_id: String(node?.id ?? "") } }));
    } catch (_) {}
}

function presetClipPayload(clip) {
    const c = normalizeClip(clip ?? {});
    return { name: c.name, prompt: c.prompt, duration: c.duration, seed: c.seed };
}

function readClipPresets() {
    try {
        const raw = JSON.parse(localStorage.getItem(CLIP_PRESET_STORAGE_KEY) || "{}");
        const entries = Array.isArray(raw) ? raw : raw?.presets;
        if (!Array.isArray(entries)) return [];
        return entries.map((entry) => {
            const presetName = String(entry?.preset_name ?? entry?.name ?? "").trim().slice(0, 160);
            const clip = presetClipPayload(entry?.clip ?? entry ?? {});
            return presetName ? { preset_name: presetName, clip } : null;
        }).filter(Boolean).slice(0, 512);
    } catch (_) { return []; }
}

function writeClipPresets(presets) {
    const payload = {
        version: 1,
        kind: "h3_longmedia_clip_presets",
        exported_by: "ComfyUI-MiniMax-H3-LongMedia",
        presets: (Array.isArray(presets) ? presets : []).slice(0, 512),
    };
    try { localStorage.setItem(CLIP_PRESET_STORAGE_KEY, JSON.stringify(payload)); return true; }
    catch (_) { return false; }
}

function upsertClipPreset(presetName, clip) {
    presetName = String(presetName ?? "").trim().slice(0, 160);
    if (!presetName) return false;
    const presets = readClipPresets();
    const value = { preset_name: presetName, clip: presetClipPayload(clip) };
    const idx = presets.findIndex((p) => p.preset_name.toLowerCase() === presetName.toLowerCase());
    if (idx >= 0) presets[idx] = value; else presets.push(value);
    presets.sort((a, b) => a.preset_name.localeCompare(b.preset_name));
    return writeClipPresets(presets);
}

function deleteClipPreset(presetName) {
    const key = String(presetName ?? "").trim().toLowerCase();
    const presets = readClipPresets();
    const next = presets.filter((p) => p.preset_name.toLowerCase() !== key);
    if (next.length === presets.length) return false;
    return writeClipPresets(next);
}

function exportClipPresets() {
    const payload = {
        version: 1,
        kind: "h3_longmedia_clip_presets",
        exported_at: new Date().toISOString(),
        presets: readClipPresets(),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "MiniMax-H3-LongMedia-clip-presets.json";
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    return payload.presets.length;
}

async function importClipPresets(file) {
    const text = await file.text();
    const raw = JSON.parse(text);
    const entries = Array.isArray(raw) ? raw : raw?.presets;
    if (!Array.isArray(entries)) throw new Error("preset file has no presets array");
    const merged = readClipPresets();
    let imported = 0;
    for (const entry of entries) {
        const presetName = String(entry?.preset_name ?? entry?.name ?? "").trim().slice(0, 160);
        if (!presetName) continue;
        const value = { preset_name: presetName, clip: presetClipPayload(entry?.clip ?? entry ?? {}) };
        const idx = merged.findIndex((p) => p.preset_name.toLowerCase() === presetName.toLowerCase());
        if (idx >= 0) merged[idx] = value; else merged.push(value);
        imported += 1;
    }
    merged.sort((a, b) => a.preset_name.localeCompare(b.preset_name));
    if (!writeClipPresets(merged)) throw new Error("browser preset storage is unavailable");
    return imported;
}

// Best-effort immediate import for ordinary text/primitive source nodes.
// Dynamic outputs (LLM/API/etc.) do not necessarily exist in the browser; those
// fall back to a one-shot backend import request on the next execution.
function connectedTextValue(node, name) {
    const input = node.inputs?.find((i) => i?.name === name);
    const linkId = input?.link ?? (Array.isArray(input?.links) ? input.links.find((id) => id != null) : null);
    if (linkId == null) return null;
    const graph = node.graph ?? app.graph;
    const links = graph?.links;
    const link = links instanceof Map ? links.get(linkId) : links?.[linkId];
    if (!link) return null;
    const originId = link.origin_id ?? link.originId;
    const originSlot = Number(link.origin_slot ?? link.originSlot ?? 0);
    const origin = graph?.getNodeById?.(originId) ?? graph?._nodes?.find((n) => String(n?.id) === String(originId));
    if (!origin) return null;

    try {
        const direct = origin.getOutputData?.(originSlot);
        if (typeof direct === "string" && direct.trim()) return direct;
    } catch (_) {}
    const out = origin.outputs?.[originSlot];
    for (const candidate of [out?._data, out?.value]) {
        if (typeof candidate === "string" && candidate.trim()) return candidate;
    }

    const stringWidgets = (origin.widgets ?? []).filter((w) => typeof w?.value === "string");
    if (!stringWidgets.length) return null;
    const outputName = String(out?.name ?? "").toLowerCase();
    const preferred = stringWidgets.find((w) => {
        const n = String(w?.name ?? w?.label ?? "").toLowerCase();
        return outputName && n === outputName;
    }) ?? stringWidgets.find((w) => {
        const n = String(w?.name ?? w?.label ?? "").toLowerCase();
        return /(?:text|prompt|string|value)/.test(n);
    }) ?? (stringWidgets.length === 1 ? stringWidgets[0] : null);
    return typeof preferred?.value === "string" ? preferred.value : null;
}

function _lmStripPromptFence(raw) {
    let text = String(raw ?? "").replace(/\r\n?/g, "\n").trim();
    const fenced = text.match(/^```(?:json|yaml|yml|text|markdown|md)?\s*\n([\s\S]*?)\n```\s*$/i);
    return fenced ? fenced[1].trim() : text;
}

function _lmValidateIndexedSections(sections) {
    if (!(sections instanceof Map) || sections.size < MIN_CLIPS) {
        return { ok: false, error: "need at least clip_1 and clip_2" };
    }
    const max = Math.max(...sections.keys());
    for (let i = 1; i <= max; i++) {
        if (!sections.has(i)) return { ok: false, error: `missing clip_${i}` };
    }
    return {
        ok: true,
        prompts: Array.from({ length: max }, (_, i) => String(sections.get(i + 1) ?? "").trim()),
    };
}

function _lmPromptBody(lines) {
    let body = Array.isArray(lines) ? [...lines] : String(lines ?? "").split("\n");
    while (body.length && !String(body[0]).trim()) body.shift();
    while (body.length && !String(body[body.length - 1]).trim()) body.pop();
    if (!body.length) return "";

    // LLMs often emit:
    // clip_1:
    //   duration: 5
    //   prompt: |
    //     actual prompt...
    //   seed: auto
    //
    // Import Prompt owns only the prompt text, so wrappers are accepted but
    // duration/seed remain untouched in the existing Planner cards.
    const promptIdx = body.findIndex((line) => /^\s*(?:[-*+]\s*)?prompt\s*:\s*(?:[|>][-+]?)?\s*(.*)$/i.test(String(line)));
    if (promptIdx >= 0) {
        const m = String(body[promptIdx]).match(/^\s*(?:[-*+]\s*)?prompt\s*:\s*(?:[|>][-+]?)?\s*(.*)$/i);
        const inline = String(m?.[1] ?? "").trim();
        let selected = body.slice(promptIdx + 1);
        if (inline) selected.unshift(inline);
        // Remove only trailing card metadata, never arbitrary prompt lines.
        while (selected.length && /^\s*(?:duration|seed)\s*:\s*.*$/i.test(String(selected[selected.length - 1]))) {
            selected.pop();
        }
        body = selected;
    } else {
        // If there is no prompt: wrapper, tolerate leading/trailing card metadata.
        while (body.length && /^\s*(?:duration|seed)\s*:\s*.*$/i.test(String(body[0]))) body.shift();
        while (body.length && /^\s*(?:duration|seed)\s*:\s*.*$/i.test(String(body[body.length - 1]))) body.pop();
    }

    // Remove common YAML block indentation while preserving intentional internal layout.
    const nonEmpty = body.filter((line) => String(line).trim());
    if (nonEmpty.length) {
        const indents = nonEmpty.map((line) => (String(line).match(/^\s*/) ?? [""])[0].length);
        const minIndent = Math.min(...indents);
        if (minIndent > 0) body = body.map((line) => String(line).slice(minIndent));
    }
    return body.join("\n").trim();
}

function _lmTryJsonPrompt(text) {
    let value;
    try { value = JSON.parse(text); } catch (_) { return null; }

    let entries = null;
    if (Array.isArray(value)) {
        entries = value;
    } else if (value && typeof value === "object" && Array.isArray(value.clips)) {
        entries = value.clips;
    }

    if (entries) {
        const prompts = entries.slice(0, MAX_CLIPS).map((item) => {
            if (typeof item === "string") return item.trim();
            if (item && typeof item === "object") return String(item.prompt ?? item.text ?? item.description ?? "").trim();
            return "";
        });
        if (prompts.length >= MIN_CLIPS) return { ok: true, prompts };
    }

    if (value && typeof value === "object" && !Array.isArray(value)) {
        const sections = new Map();
        for (const [key, item] of Object.entries(value)) {
            const m = String(key).match(/^(?:clip|shot)[ _-]*(\d{1,2})$/i);
            if (!m) continue;
            const idx = Number(m[1]);
            if (idx < 1 || idx > MAX_CLIPS || sections.has(idx)) return null;
            const prompt = (item && typeof item === "object")
                ? String(item.prompt ?? item.text ?? item.description ?? "")
                : String(item ?? "");
            sections.set(idx, prompt.trim());
        }
        if (sections.size >= MIN_CLIPS) return _lmValidateIndexedSections(sections);
    }
    return null;
}

function _lmTryXmlPrompt(text) {
    const sections = new Map();
    const re = /<(?:clip|shot)[ _-]*(\d{1,2})\b[^>]*>([\s\S]*?)<\/(?:clip|shot)[ _-]*\1\s*>/gi;
    for (const match of text.matchAll(re)) {
        const idx = Number(match[1]);
        if (idx < 1 || idx > MAX_CLIPS || sections.has(idx)) return null;
        sections.set(idx, _lmPromptBody(String(match[2] ?? "").split("\n")));
    }
    return sections.size >= MIN_CLIPS ? _lmValidateIndexedSections(sections) : null;
}

function _lmHeader(line) {
    // Normalize harmless Markdown decoration only. Keep "_" because clip_1 uses it.
    let s = String(line ?? "").replace(/^\s{0,3}#{1,6}\s*/, "").trim();
    s = s.replace(/^\s*(?:[-+]\s+)(?=(?:\*\*)?\[?(?:clip|shot)\b)/i, "");
    s = s.replace(/\*\*/g, "").replace(/`/g, "").trim();

    // Accepted examples:
    // clip_1:
    // Clip 1: text
    // SHOT-2 - text
    // [clip_3]: text
    // Clip 4 (5s): text
    // clip_5 = text
    const m = s.match(/^\[?(?:clip|shot)[ _-]*(\d{1,2})\]?(?:\s*\([^)]*\))?\s*(?:(?::|=|[-–—])\s*)?(.*)$/i);
    if (!m) return null;
    const idx = Number(m[1]);
    if (!Number.isInteger(idx) || idx < 1 || idx > MAX_CLIPS) return { error: `clip index ${idx} is outside 1..${MAX_CLIPS}` };
    return { idx, inline: String(m[2] ?? "").trim() };
}

function _lmTryHeaderSections(text) {
    const sections = new Map();
    let current = null;
    let body = [];

    const flush = () => {
        if (current == null) return true;
        if (sections.has(current)) return false;
        sections.set(current, _lmPromptBody(body));
        body = [];
        return true;
    };

    for (const line of text.split("\n")) {
        const header = _lmHeader(line);
        if (header?.error) return { ok: false, error: header.error };
        if (header) {
            if (!flush()) return { ok: false, error: `duplicate clip_${current}` };
            current = header.idx;
            body = header.inline ? [header.inline] : [];
        } else if (current != null) {
            body.push(line);
        }
    }
    if (!flush()) return { ok: false, error: `duplicate clip_${current}` };
    return sections.size >= MIN_CLIPS ? _lmValidateIndexedSections(sections) : null;
}

function _lmTryNumberedList(text) {
    const nonEmpty = text.split("\n").filter((line) => String(line).trim());
    if (nonEmpty.length < MIN_CLIPS) return null;
    const sections = new Map();
    for (const line of nonEmpty) {
        const m = String(line).match(/^\s*(\d{1,2})\s*[.)]\s+(.+?)\s*$/);
        if (!m) return null; // deliberately strict fallback to avoid parsing prose lists
        const idx = Number(m[1]);
        if (idx < 1 || idx > MAX_CLIPS || sections.has(idx)) return null;
        sections.set(idx, String(m[2]).trim());
    }
    return _lmValidateIndexedSections(sections);
}

function parseStructuredPrompt(raw) {
    const text = _lmStripPromptFence(raw);
    if (!text) return { ok: false, error: "prompt is empty" };

    // Ordered from least ambiguous to most permissive.
    for (const parser of [_lmTryJsonPrompt, _lmTryXmlPrompt, _lmTryHeaderSections, _lmTryNumberedList]) {
        const parsed = parser(text);
        if (parsed?.ok) return parsed;
        if (parsed && parsed.ok === false) return parsed;
    }
    return {
        ok: false,
        error: "no clip sections found — use clip_1/clip_2, JSON clips, XML clips, or a numbered list",
    };
}

function importPrompts(node, prompts) {
    if (!Array.isArray(prompts) || prompts.length < MIN_CLIPS) return false;
    const clips = node.__lmPlannerClips ?? parse(node);
    const fallbackDuration = Number(clips[clips.length - 1]?.duration) || 7.5;
    while (clips.length < Math.min(prompts.length, MAX_CLIPS)) clips.push(newClip(fallbackDuration));
    prompts.slice(0, MAX_CLIPS).forEach((p, i) => { clips[i].prompt = String(p ?? ""); });
    commit(node, clips);
    node.__lmPlannerEditor?.__lmPlannerRender?.();
    return true;
}

function markImportedSource(node, text) {
    const state = widget(node, "multiclip_last_import_source");
    if (!state) return;
    const value = String(text ?? "");
    if (state.value !== value) {
        state.value = value;
        try { state.callback?.(value); } catch (_) {}
    }
}

function importFromWidget(node, automatic = false) {
    const source = widget(node, "multiclip_prompt");
    const sourceText = String(source?.value ?? "");
    const parsed = parseStructuredPrompt(sourceText);
    if (!parsed.ok) return parsed;
    importPrompts(node, parsed.prompts);
    markImportedSource(node, sourceText);
    return { ok: true, count: parsed.prompts.length };
}

const HIDDEN_SIZE = () => [0, -4];
const HIDDEN_DRAW = () => {};
function setVisible(w, visible) {
    if (!w) return;
    if (!w.__lmPlannerCaptured) {
        w.__lmPlannerCaptured = true;
        w.__lmPlannerOwnCompute = Object.prototype.hasOwnProperty.call(w, "computeSize");
        w.__lmPlannerOwnDraw = Object.prototype.hasOwnProperty.call(w, "draw");
        w.__lmPlannerCompute = w.computeSize;
        w.__lmPlannerDraw = w.draw;
    }
    const hidden = !visible;
    w.options = { ...(w.options ?? {}), hidden };
    w.hidden = hidden;
    if (visible) {
        if (w.__lmPlannerOwnCompute) w.computeSize = w.__lmPlannerCompute; else { try { delete w.computeSize; } catch (_) {} }
        if (w.__lmPlannerOwnDraw) w.draw = w.__lmPlannerDraw; else { try { delete w.draw; } catch (_) {} }
    } else {
        w.computeSize = HIDDEN_SIZE;
        w.draw = HIDDEN_DRAW;
    }
}

function fallbackClips() {
    return [newClip(7.5), newClip(7.5)];
}

function parse(node) {
    const storage = widget(node, "clips_json");
    try {
        const raw = JSON.parse(String(storage?.value ?? "[]"));
        if (!Array.isArray(raw)) return fallbackClips();
        const clips = raw.slice(0, MAX_CLIPS).map((item) => normalizeClip(item));
        while (clips.length < MIN_CLIPS) clips.push(newClip(7.5));
        return clips;
    } catch (_) {
        return fallbackClips();
    }
}

function commit(node, clips, notify = true) {
    const storage = widget(node, "clips_json");
    if (!storage) return;
    const normalized = clips.slice(0, MAX_CLIPS).map((clip) => normalizeClip(clip));
    while (normalized.length < MIN_CLIPS) normalized.push(newClip(7.5));
    const value = JSON.stringify(normalized);
    if (storage.value !== value) {
        storage.value = value;
        try { storage.callback?.(value); } catch (_) {}
    }
    node.__lmPlannerClips = normalized;
    node.__lmPlannerRaw = value;
    if (notify) notifyPlannerChanged(node);
    node.graph?.setDirtyCanvas?.(true, true);
    app.canvas?.setDirty?.(true, true);
}

function styleInput(el) {
    Object.assign(el.style, {
        boxSizing: "border-box", width: "100%", color: "var(--input-text, #eee)",
        background: "var(--comfy-input-bg, #171717)", border: "1px solid #555",
        borderRadius: "6px", padding: "6px 8px", font: "12px sans-serif", outline: "none",
    });
    return el;
}

function ensureEditor(node) {
    if (node.__lmPlannerEditor) return node.__lmPlannerEditor;
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
    const importButton = document.createElement("button"); importButton.textContent = "Import Prompt";
    const add = document.createElement("button"); add.textContent = "+ Add Clip";
    const remove = document.createElement("button"); remove.textContent = "− Remove Last";
    const presetSelect = styleInput(document.createElement("select"));
    Object.assign(presetSelect.style, { width: "180px", padding: "3px 6px", fontSize: "11px" });
    const addPresetClip = document.createElement("button"); addPresetClip.textContent = "+ From Preset";
    const exportPresets = document.createElement("button"); exportPresets.textContent = "Export Presets";
    const importPresets = document.createElement("button"); importPresets.textContent = "Import Presets";
    const deletePreset = document.createElement("button"); deletePreset.textContent = "Delete Preset";
    const presetFile = document.createElement("input"); presetFile.type = "file"; presetFile.accept = ".json,application/json"; presetFile.style.display = "none";
    const count = document.createElement("span");
    const status = document.createElement("span");
    for (const b of [importButton, add, remove, addPresetClip, exportPresets, importPresets, deletePreset]) Object.assign(b.style, { color: "#eee", background: "#202020", border: "1px solid #777", borderRadius: "4px", padding: "3px 7px", cursor: "pointer", fontSize: "11px" });
    Object.assign(count.style, { fontSize: "11px", opacity: "0.8" });
    Object.assign(status.style, { fontSize: "11px", opacity: "0.75", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", minWidth: "120px" });
    toolbar.append(importButton, add, remove, presetSelect, addPresetClip, exportPresets, importPresets, deletePreset, count, status, presetFile);

    const viewport = document.createElement("div");
    Object.assign(viewport.style, {
        flex: "1 1 auto", minWidth: "0", minHeight: "0", width: "100%",
        boxSizing: "border-box", overflowX: "auto", overflowY: "auto",
        scrollbarGutter: "stable", overscrollBehavior: "contain",
    });
    const cards = document.createElement("div");
    Object.assign(cards.style, {
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))",
        gridAutoRows: "max-content", alignContent: "start", gap: "10px",
        width: "100%", minWidth: "610px", minHeight: "min-content",
        boxSizing: "border-box", padding: "0 2px 2px 0", contain: "layout paint style",
    });
    viewport.append(cards);
    root.append(toolbar, viewport);

    function refreshPresetSelect(keep = null) {
        const current = keep ?? presetSelect.value;
        const presets = readClipPresets();
        presetSelect.replaceChildren();
        const empty = document.createElement("option"); empty.value = ""; empty.textContent = presets.length ? "Select clip preset…" : "No clip presets"; presetSelect.append(empty);
        for (const p of presets) {
            const option = document.createElement("option"); option.value = p.preset_name; option.textContent = p.preset_name; presetSelect.append(option);
        }
        presetSelect.value = presets.some((p) => p.preset_name === current) ? current : "";
        deletePreset.disabled = !presetSelect.value;
        addPresetClip.disabled = !presetSelect.value;
    }
    presetSelect.addEventListener("change", () => { deletePreset.disabled = !presetSelect.value; addPresetClip.disabled = !presetSelect.value; });

    function render() {
        const clips = node.__lmPlannerClips ?? parse(node);
        node.__lmPlannerClips = clips;
        count.textContent = `${clips.length} clips`;
        add.disabled = clips.length >= MAX_CLIPS;
        remove.disabled = clips.length <= MIN_CLIPS;
        refreshPresetSelect();
        cards.replaceChildren();

        let cursor = 0;
        const timeline = clips.map((clip) => {
            const start = cursor;
            const end = start + (Number(clip.duration) || 0);
            cursor = end;
            return { start, end };
        });

        clips.forEach((clip, index) => {
            const card = document.createElement("div");
            card.dataset.lmCardIndex = String(index);
            Object.assign(card.style, { border: `1px solid ${index === 0 ? "#d9a400" : "#397db0"}`, borderRadius: "9px", padding: "9px", background: "rgba(10,10,10,0.45)", minWidth: "0", transition: "opacity 100ms ease, outline 100ms ease", contain: "layout paint style" });

            const header = document.createElement("div");
            header.draggable = true;
            Object.assign(header.style, { display: "grid", gridTemplateColumns: "auto auto minmax(90px, 1fr) auto", gap: "7px", alignItems: "center", marginBottom: "5px", cursor: "grab", userSelect: "none" });
            const grip = document.createElement("span"); grip.textContent = "⠿"; grip.title = "Drag clip to reorder"; Object.assign(grip.style, { opacity: "0.72", fontSize: "17px", lineHeight: "1" });
            const ordinal = document.createElement("span"); ordinal.textContent = `CLIP ${index + 1}`; Object.assign(ordinal.style, { fontWeight: "700", fontSize: "13px", whiteSpace: "nowrap" });
            const name = styleInput(document.createElement("input")); name.type = "text"; name.placeholder = "Clip name"; name.value = clip.name ?? ""; Object.assign(name.style, { padding: "4px 7px", fontSize: "11px" });
            name.draggable = false;
            name.addEventListener("dragstart", (ev) => ev.stopPropagation());
            name.addEventListener("change", () => { clip.name = name.value.trim().slice(0, 120); name.value = clip.name; commit(node, clips); render(); });
            const savePreset = document.createElement("button"); savePreset.textContent = "Save Preset";
            Object.assign(savePreset.style, { color: "#eee", background: "#202020", border: "1px solid #666", borderRadius: "4px", padding: "3px 6px", cursor: "pointer", fontSize: "10px" });
            header.append(grip, ordinal, name, savePreset);

            const tl = timeline[index];
            const timing = document.createElement("div");
            timing.textContent = `${formatTimelineTime(tl.start)}  →  ${formatTimelineTime(tl.end)}   ·   ${Number(clip.duration).toFixed(2)}s`;
            Object.assign(timing.style, { fontSize: "10px", opacity: "0.66", margin: "0 0 7px 25px", fontVariantNumeric: "tabular-nums" });

            const promptLabel = document.createElement("div"); promptLabel.textContent = "Prompt";
            Object.assign(promptLabel.style, { fontSize: "11px", marginBottom: "3px", opacity: "0.9" });
            const prompt = styleInput(document.createElement("textarea")); prompt.rows = 8; prompt.value = clip.prompt ?? ""; prompt.style.resize = "vertical";
            prompt.addEventListener("input", () => { clip.prompt = prompt.value; commit(node, clips, false); });

            const row = document.createElement("div");
            Object.assign(row.style, { display: "grid", gridTemplateColumns: "1fr 96px", gap: "8px", marginTop: "7px" });
            const seedWrap = document.createElement("div");
            const seedLabel = document.createElement("div"); seedLabel.textContent = "Seed"; Object.assign(seedLabel.style, { fontSize: "11px", marginBottom: "3px", opacity: "0.9" });
            const seed = styleInput(document.createElement("input")); seed.type = "number"; seed.step = "1"; seed.min = "0"; seed.placeholder = "auto"; seed.value = clip.seed == null ? "" : String(clip.seed);
            seed.addEventListener("change", () => { clip.seed = seed.value.trim() === "" ? null : Math.max(0, Math.trunc(Number(seed.value) || 0)); commit(node, clips, false); });
            seedWrap.append(seedLabel, seed);
            const durWrap = document.createElement("div");
            const durLabel = document.createElement("div"); durLabel.textContent = "Duration s"; Object.assign(durLabel.style, { fontSize: "11px", marginBottom: "3px", opacity: "0.9" });
            const duration = styleInput(document.createElement("input")); duration.type = "number"; duration.step = "0.1"; duration.min = "0.25"; duration.max = "150"; duration.value = String(clip.duration ?? 7.5);
            duration.addEventListener("change", () => { clip.duration = Math.max(0.25, Math.min(150, Number(duration.value) || 7.5)); duration.value = String(clip.duration); commit(node, clips); render(); });
            durWrap.append(durLabel, duration);
            row.append(seedWrap, durWrap);

            const presetRow = document.createElement("div"); Object.assign(presetRow.style, { display: "flex", gap: "6px", marginTop: "7px", alignItems: "center" });
            const applyPreset = document.createElement("button"); applyPreset.textContent = "Apply Selected Preset";
            Object.assign(applyPreset.style, { color: "#eee", background: "#202020", border: "1px solid #666", borderRadius: "4px", padding: "3px 6px", cursor: "pointer", fontSize: "10px" });
            const clipIdHint = document.createElement("span"); clipIdHint.textContent = "drag title to reorder"; Object.assign(clipIdHint.style, { marginLeft: "auto", fontSize: "9px", opacity: "0.45" });
            presetRow.append(applyPreset, clipIdHint);

            function startDrag(ev) {
                if (ev.target === name || name.contains?.(ev.target)) { ev.preventDefault(); return; }
                node.__lmPlannerDragIndex = index;
                card.style.opacity = "0.58";
                try { ev.dataTransfer.effectAllowed = "move"; ev.dataTransfer.setData("text/plain", String(index)); } catch (_) {}
            }
            header.addEventListener("dragstart", startDrag);
            header.addEventListener("dragend", () => { node.__lmPlannerDragIndex = null; card.style.opacity = "1"; for (const c of cards.children) c.style.outline = ""; });
            card.addEventListener("dragover", (ev) => { if (node.__lmPlannerDragIndex == null) return; ev.preventDefault(); try { ev.dataTransfer.dropEffect = "move"; } catch (_) {} card.style.outline = "2px solid rgba(217,164,0,0.75)"; });
            card.addEventListener("dragleave", () => { card.style.outline = ""; });
            card.addEventListener("drop", (ev) => {
                ev.preventDefault(); ev.stopPropagation(); card.style.outline = "";
                const from = Number(node.__lmPlannerDragIndex);
                const to = index;
                node.__lmPlannerDragIndex = null;
                if (moveItem(clips, from, to)) {
                    commit(node, clips);
                    status.textContent = `moved clip ${from + 1} → ${to + 1}`;
                    render();
                }
            });

            savePreset.addEventListener("click", (ev) => {
                ev.preventDefault(); ev.stopPropagation();
                const proposed = window.prompt("Preset name", clip.name || `Clip ${index + 1}`);
                if (proposed == null) return;
                const presetName = String(proposed).trim();
                if (!presetName) { status.textContent = "preset name is empty"; return; }
                const exists = readClipPresets().some((p) => p.preset_name.toLowerCase() === presetName.toLowerCase());
                if (exists && !window.confirm(`Replace preset “${presetName}”?`)) return;
                if (upsertClipPreset(presetName, clip)) {
                    status.textContent = `saved preset: ${presetName}`;
                    refreshPresetSelect(presetName);
                } else status.textContent = "failed to save preset";
            });
            applyPreset.addEventListener("click", (ev) => {
                ev.preventDefault(); ev.stopPropagation();
                const presetName = presetSelect.value;
                const selected = readClipPresets().find((p) => p.preset_name === presetName);
                if (!selected) { status.textContent = "select a clip preset"; return; }
                const keepId = clip.clip_id;
                Object.assign(clip, presetClipPayload(selected.clip), { clip_id: keepId });
                commit(node, clips);
                status.textContent = `applied preset: ${presetName}`;
                render();
            });

            card.append(header, timing, promptLabel, prompt, row, presetRow);
            cards.append(card);
        });
        requestAnimationFrame(() => {
            node.setDirtyCanvas?.(true, true);
            app.canvas?.setDirty?.(true, true);
        });
    }

    importButton.addEventListener("click", (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        if (inputConnected(node, "multiclip_prompt")) {
            const linkedText = connectedTextValue(node, "multiclip_prompt");
            if (typeof linkedText === "string" && linkedText.trim()) {
                const parsed = parseStructuredPrompt(linkedText);
                if (parsed.ok) {
                    importPrompts(node, parsed.prompts);
                    markImportedSource(node, linkedText);
                    status.textContent = `imported ${parsed.prompts.length} clips`;
                    render();
                    return;
                }
                status.textContent = parsed.error;
                return;
            }
            const request = widget(node, "multiclip_import_request");
            if (request) {
                request.value = true;
                try { request.callback?.(true); } catch (_) {}
                status.textContent = "import queued — run workflow once";
            } else status.textContent = "connected prompt is dynamic; run workflow with Auto Import";
            node.setDirtyCanvas?.(true, true);
            return;
        }
        const result = importFromWidget(node, false);
        status.textContent = result.ok ? `imported ${result.count} clips` : result.error;
        if (result.ok) render();
    });

    add.addEventListener("click", (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        const clips = node.__lmPlannerClips ?? parse(node);
        if (clips.length >= MAX_CLIPS) return;
        const prev = clips[clips.length - 1] ?? { duration: 7.5 };
        clips.push(newClip(Number(prev.duration) || 7.5));
        commit(node, clips); render();
    });
    remove.addEventListener("click", (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        const clips = node.__lmPlannerClips ?? parse(node);
        if (clips.length <= MIN_CLIPS) return;
        clips.pop(); commit(node, clips); render();
    });
    addPresetClip.addEventListener("click", (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        const selected = readClipPresets().find((p) => p.preset_name === presetSelect.value);
        if (!selected) { status.textContent = "select a clip preset"; return; }
        const clips = node.__lmPlannerClips ?? parse(node);
        if (clips.length >= MAX_CLIPS) { status.textContent = `maximum ${MAX_CLIPS} clips`; return; }
        clips.push(newClip(selected.clip.duration, selected.clip));
        commit(node, clips); status.textContent = `added preset clip: ${selected.preset_name}`; render();
    });
    deletePreset.addEventListener("click", (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        const name = presetSelect.value;
        if (!name) return;
        if (!window.confirm(`Delete preset “${name}”?`)) return;
        if (deleteClipPreset(name)) { status.textContent = `deleted preset: ${name}`; refreshPresetSelect(); }
    });
    exportPresets.addEventListener("click", (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        try { const n = exportClipPresets(); status.textContent = `exported ${n} presets`; }
        catch (err) { status.textContent = `export failed: ${err?.message ?? err}`; }
    });
    importPresets.addEventListener("click", (ev) => { ev.preventDefault(); ev.stopPropagation(); presetFile.value = ""; presetFile.click(); });
    presetFile.addEventListener("change", async () => {
        const file = presetFile.files?.[0];
        if (!file) return;
        try { const n = await importClipPresets(file); status.textContent = `imported ${n} presets`; refreshPresetSelect(); }
        catch (err) { status.textContent = `preset import failed: ${err?.message ?? err}`; }
    });

    const editor = node.addDOMWidget("clip_editor", "clip_editor", root, { serialize: false, hideOnZoom: true, getValue: () => null, setValue: () => {} });
    node.__lmPlannerViewportHeight = Number(node.__lmPlannerViewportHeight) || 340;
    editor.computeSize = function(width) {
        const w = Math.max(180, Number(width) || Number(node.size?.[0]) || 650);
        const h = Math.max(120, Number(node.__lmPlannerViewportHeight) || 340);
        return [w, h];
    };
    editor.__lmPlannerRender = render;
    node.__lmPlannerEditor = editor;
    node.__lmPlannerClips = parse(node);
    commit(node, node.__lmPlannerClips); // one-time migration adds stable clip_id/name fields to old workflows
    setVisible(widget(node, "clips_json"), false);

    if (!Number.isFinite(node.__lmPlannerChromeHeight)) {
        try {
            const total = node.computeSize?.();
            const totalH = Number(total?.[1]);
            if (Number.isFinite(totalH)) node.__lmPlannerChromeHeight = Math.max(40, totalH - node.__lmPlannerViewportHeight);
        } catch (_) {}
        if (!Number.isFinite(node.__lmPlannerChromeHeight)) node.__lmPlannerChromeHeight = 82;
    }
    if (!node.__lmPlannerComputeSizeHooked && typeof node.computeSize === "function") {
        node.__lmPlannerComputeSizeHooked = true;
        const previousComputeSize = node.computeSize;
        node.__lmPlannerPreviousComputeSize = previousComputeSize;
        node.computeSize = function(...args) {
            const savedViewportHeight = this.__lmPlannerViewportHeight;
            try { this.__lmPlannerViewportHeight = 120; return previousComputeSize.apply(this, args); }
            finally { this.__lmPlannerViewportHeight = savedViewportHeight; }
        };
    }
    if (!node.__lmPlannerResizeHooked) {
        node.__lmPlannerResizeHooked = true;
        const previousOnResize = node.onResize;
        node.onResize = function(size) {
            try { previousOnResize?.apply(this, arguments); } catch (_) {}
            const requestedH = Number(size?.[1]);
            const chromeH = Number(this.__lmPlannerChromeHeight);
            if (Number.isFinite(requestedH) && Number.isFinite(chromeH)) this.__lmPlannerViewportHeight = Math.max(120, requestedH - chromeH);
            try { this.setDirtyCanvas?.(true, true); } catch (_) {}
            try { app.canvas?.setDirty?.(true, true); } catch (_) {}
        };
    }
    requestAnimationFrame(() => {
        const w = Number(node.size?.[0]);
        const h = Number(node.size?.[1]);
        if (Number.isFinite(h) && h > 1600) {
            node.__lmPlannerViewportHeight = 340;
            try { node.setSize?.([Number.isFinite(w) ? Math.max(420, w) : 650, 440]); } catch (_) {}
        }
        try { node.setDirtyCanvas?.(true, true); } catch (_) {}
        try { app.canvas?.setDirty?.(true, true); } catch (_) {}
    });

    refreshPresetSelect();
    render();
    return editor;
}

function refresh(node) {
    if (!isPlanner(node)) return;
    setVisible(widget(node, "clips_json"), false);
    const gp = widget(node, "global_prompt");
    if (gp) { gp.label = "Global Prompt"; gp.localized_name = "Global Prompt"; }
    const mp = widget(node, "multiclip_prompt");
    if (mp) { mp.label = "Multiple Clips Prompt"; mp.localized_name = "Multiple Clips Prompt"; }
    const ai = widget(node, "multiclip_auto_import");
    if (ai) { ai.label = "Auto Import Prompt"; ai.localized_name = "Auto Import Prompt"; }
    setVisible(widget(node, "multiclip_import_request"), false);
    setVisible(widget(node, "multiclip_last_import_source"), false);
    const raw = String(widget(node, "clips_json")?.value ?? "");
    if (raw !== node.__lmPlannerRaw) {
        node.__lmPlannerRaw = raw;
        node.__lmPlannerClips = parse(node);
    }
    const editor = ensureEditor(node);
    editor?.__lmPlannerRender?.();
}

function wireAutoImport(node) {
    if (!isPlanner(node) || node.__lmPlannerAutoImportWired) return;
    node.__lmPlannerAutoImportWired = true;
    for (const name of ["multiclip_prompt", "multiclip_auto_import"]) {
        const w = widget(node, name);
        if (!w) continue;
        const prior = w.callback;
        w.callback = function (...args) {
            const out = prior?.apply(this, args);
            const auto = Boolean(widget(node, "multiclip_auto_import")?.value);
            if (auto && !inputConnected(node, "multiclip_prompt")) {
                const result = importFromWidget(node, true);
                if (result.ok) refresh(node);
            }
            return out;
        };
    }
}

app.registerExtension({
    name: "MiniMaxH3.LongMediaPlanner.v4",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const cls = nodeType?.comfyClass ?? nodeType?.ComfyClass ?? nodeData?.name;
        if (cls !== PLANNER_CLASS) return;
        nodeType.category = "MiniMax H3/Long Media";
        if (nodeData) { nodeData.hidden = false; nodeData.category = "MiniMax H3/Long Media"; }
    },
    async nodeCreated(node) { if (isPlanner(node)) setTimeout(() => { wireAutoImport(node); refresh(node); }, 0); },
    async afterConfigureGraph() { for (const node of app.graph?._nodes ?? []) if (isPlanner(node)) { wireAutoImport(node); refresh(node); } },
});

api.addEventListener("minimax_h3_planner_prompt_import", (event) => {
    const detail = event?.detail ?? event;
    const nodeId = String(detail?.node_id ?? "");
    if (!nodeId) return;
    const node = app.graph?._nodes?.find((n) => String(n?.id) === nodeId && isPlanner(n));
    if (!node) return;

    if (detail?.clear_request) {
        const request = widget(node, "multiclip_import_request");
        if (request) { request.value = false; try { request.callback?.(false); } catch (_) {} }
    }
    const prompts = detail?.prompts;
    if (Array.isArray(prompts) && prompts.length >= MIN_CLIPS) {
        if (importPrompts(node, prompts)) {
            if (typeof detail?.source_text === "string") markImportedSource(node, detail.source_text);
            refresh(node);
        }
    }
});
