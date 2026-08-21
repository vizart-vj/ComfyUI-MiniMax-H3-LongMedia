import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PLANNER_CLASS = "MiniMaxH3LongMediaPlanner";
const MIN_CLIPS = 2;
const MAX_CLIPS = 16;

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

function parseStructuredPrompt(raw) {
    const text = String(raw ?? "").replace(/\r\n?/g, "\n");
    const re = /^\s{0,3}(?:#{1,6}\s*)?(?:clip|shot)[ _-]*(\d{1,2})\s*:?\s*$/gim;
    const matches = [...text.matchAll(re)];
    if (matches.length < 2) return { ok: false, error: "need at least clip_1 and clip_2" };
    const sections = new Map();
    for (let i = 0; i < matches.length; i++) {
        const idx = Number(matches[i][1]);
        if (!Number.isInteger(idx) || idx < 1 || idx > MAX_CLIPS) return { ok: false, error: `clip index ${idx} is outside 1..${MAX_CLIPS}` };
        if (sections.has(idx)) return { ok: false, error: `duplicate clip_${idx}` };
        const start = matches[i].index + matches[i][0].length;
        const end = i + 1 < matches.length ? matches[i + 1].index : text.length;
        sections.set(idx, text.slice(start, end).trim());
    }
    const max = Math.max(...sections.keys());
    for (let i = 1; i <= max; i++) if (!sections.has(i)) return { ok: false, error: `missing clip_${i}` };
    return { ok: true, prompts: Array.from({ length: max }, (_, i) => sections.get(i + 1) ?? "") };
}

function importPrompts(node, prompts) {
    if (!Array.isArray(prompts) || prompts.length < MIN_CLIPS) return false;
    const clips = node.__lmPlannerClips ?? parse(node);
    const fallbackDuration = Number(clips[clips.length - 1]?.duration) || 7.5;
    while (clips.length < Math.min(prompts.length, MAX_CLIPS)) clips.push({ prompt: "", duration: fallbackDuration, seed: null });
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
    return [
        { prompt: "", duration: 7.5, seed: null },
        { prompt: "", duration: 7.5, seed: null },
    ];
}

function parse(node) {
    const storage = widget(node, "clips_json");
    try {
        const raw = JSON.parse(String(storage?.value ?? "[]"));
        if (!Array.isArray(raw)) return fallbackClips();
        const clips = raw.slice(0, MAX_CLIPS).map((item) => {
            item = item && typeof item === "object" ? item : {};
            let duration = Number(item.duration);
            if (!Number.isFinite(duration)) duration = 7.5;
            duration = Math.max(0.25, Math.min(150, duration));
            let seed = item.seed;
            if (seed === "" || seed === undefined) seed = null;
            if (seed !== null) {
                const n = Number(seed);
                seed = Number.isFinite(n) ? Math.max(0, Math.trunc(n)) : null;
            }
            return { prompt: String(item.prompt ?? ""), duration, seed };
        });
        while (clips.length < MIN_CLIPS) clips.push({ prompt: "", duration: 7.5, seed: null });
        return clips;
    } catch (_) {
        return fallbackClips();
    }
}

function commit(node, clips) {
    const storage = widget(node, "clips_json");
    if (!storage) return;
    const normalized = clips.slice(0, MAX_CLIPS).map((clip) => ({
        prompt: String(clip.prompt ?? ""),
        duration: Math.max(0.25, Math.min(150, Number(clip.duration) || 7.5)),
        seed: clip.seed == null || clip.seed === "" ? null : Math.max(0, Math.trunc(Number(clip.seed) || 0)),
    }));
    while (normalized.length < MIN_CLIPS) normalized.push({ prompt: "", duration: 7.5, seed: null });
    const value = JSON.stringify(normalized);
    if (storage.value !== value) {
        storage.value = value;
        try { storage.callback?.(value); } catch (_) {}
    }
    node.__lmPlannerClips = normalized;
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
        color: "#ddd", display: "flex", flexDirection: "column", overflow: "hidden",
    });
    const toolbar = document.createElement("div");
    Object.assign(toolbar.style, {
        display: "flex", flex: "0 0 auto", gap: "6px", alignItems: "center",
        marginBottom: "8px", minWidth: "0",
    });
    const importButton = document.createElement("button"); importButton.textContent = "Import Prompt";
    const add = document.createElement("button"); add.textContent = "+ Add Clip";
    const remove = document.createElement("button"); remove.textContent = "− Remove Last";
    const count = document.createElement("span");
    const status = document.createElement("span");
    for (const b of [importButton, add, remove]) Object.assign(b.style, { color: "#eee", background: "#202020", border: "1px solid #777", borderRadius: "4px", padding: "3px 7px", cursor: "pointer", fontSize: "11px" });
    Object.assign(count.style, { fontSize: "11px", opacity: "0.8" });
    Object.assign(status.style, { fontSize: "11px", opacity: "0.75", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" });
    toolbar.append(importButton, add, remove, count, status);

    // 0.3.93: one bounded viewport owns all clip cards.  The node itself may be
    // freely resized; content never forces the node back to its natural size.
    // The inner grid deliberately keeps a two-tile minimum width, so shrinking
    // the node produces a horizontal scrollbar instead of collapsing cards.
    const viewport = document.createElement("div");
    Object.assign(viewport.style, {
        flex: "1 1 auto", minWidth: "0", minHeight: "0", width: "100%",
        boxSizing: "border-box", overflowX: "auto", overflowY: "auto",
        scrollbarGutter: "stable", overscrollBehavior: "contain",
    });
    const cards = document.createElement("div");
    Object.assign(cards.style, {
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
        gridAutoRows: "max-content", alignContent: "start", gap: "10px",
        width: "100%", minWidth: "570px", minHeight: "min-content",
        boxSizing: "border-box", padding: "0 2px 2px 0",
    });
    viewport.append(cards);
    root.append(toolbar, viewport);

    function render() {
        const clips = node.__lmPlannerClips ?? parse(node);
        node.__lmPlannerClips = clips;
        count.textContent = `${clips.length} clips`;
        add.disabled = clips.length >= MAX_CLIPS;
        remove.disabled = clips.length <= MIN_CLIPS;
        cards.replaceChildren();
        clips.forEach((clip, index) => {
            const card = document.createElement("div");
            Object.assign(card.style, { border: `1px solid ${index === 0 ? "#d9a400" : "#397db0"}`, borderRadius: "9px", padding: "9px", background: "rgba(10,10,10,0.45)", minWidth: "0" });
            const title = document.createElement("div"); title.textContent = `CLIP ${index + 1}`;
            Object.assign(title.style, { fontWeight: "700", fontSize: "14px", marginBottom: "7px" });
            const promptLabel = document.createElement("div"); promptLabel.textContent = "Prompt";
            Object.assign(promptLabel.style, { fontSize: "11px", marginBottom: "3px", opacity: "0.9" });
            const prompt = styleInput(document.createElement("textarea")); prompt.rows = 8; prompt.value = clip.prompt ?? ""; prompt.style.resize = "vertical";
            prompt.addEventListener("input", () => { clip.prompt = prompt.value; commit(node, clips); });

            const row = document.createElement("div");
            Object.assign(row.style, { display: "grid", gridTemplateColumns: "1fr 96px", gap: "8px", marginTop: "7px" });
            const seedWrap = document.createElement("div");
            const seedLabel = document.createElement("div"); seedLabel.textContent = "Seed"; Object.assign(seedLabel.style, { fontSize: "11px", marginBottom: "3px", opacity: "0.9" });
            const seed = styleInput(document.createElement("input")); seed.type = "number"; seed.step = "1"; seed.min = "0"; seed.placeholder = "auto"; seed.value = clip.seed == null ? "" : String(clip.seed);
            seed.addEventListener("change", () => { clip.seed = seed.value.trim() === "" ? null : Math.max(0, Math.trunc(Number(seed.value) || 0)); commit(node, clips); });
            seedWrap.append(seedLabel, seed);
            const durWrap = document.createElement("div");
            const durLabel = document.createElement("div"); durLabel.textContent = "Duration s"; Object.assign(durLabel.style, { fontSize: "11px", marginBottom: "3px", opacity: "0.9" });
            const duration = styleInput(document.createElement("input")); duration.type = "number"; duration.step = "0.1"; duration.min = "0.25"; duration.max = "150"; duration.value = String(clip.duration ?? 7.5);
            duration.addEventListener("change", () => { clip.duration = Math.max(0.25, Math.min(150, Number(duration.value) || 7.5)); duration.value = String(clip.duration); commit(node, clips); });
            durWrap.append(durLabel, duration);
            row.append(seedWrap, durWrap);
            card.append(title, promptLabel, prompt, row);
            cards.append(card);
        });
        requestAnimationFrame(() => {
            // Never resize the node to fit its children.  Resize belongs to the
            // user; the viewport scrollbars absorb any overflow.
            node.setDirtyCanvas?.(true, true);
            app.canvas?.setDirty?.(true, true);
        });
    }

    importButton.addEventListener("click", (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        if (inputConnected(node, "multiclip_prompt")) {
            // Never change Auto Import from the manual button. First try to read
            // an ordinary connected text node directly so the cards update now.
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

            // A truly dynamic connected output only exists on the backend during
            // execution. Arm an independent one-shot request; do NOT touch Auto Import.
            const request = widget(node, "multiclip_import_request");
            if (request) {
                request.value = true;
                try { request.callback?.(true); } catch (_) {}
                status.textContent = "import queued — run workflow once";
            } else {
                status.textContent = "connected prompt is dynamic; run workflow with Auto Import";
            }
            node.setDirtyCanvas?.(true, true);
            return;
        }
        const result = importFromWidget(node, false);
        status.textContent = result.ok ? `imported ${result.count} clips` : result.error;
        if (result.ok) render();
    });

    add.addEventListener("click", (ev) => { ev.preventDefault(); ev.stopPropagation(); const clips = node.__lmPlannerClips ?? parse(node); if (clips.length >= MAX_CLIPS) return; const prev = clips[clips.length - 1] ?? { duration: 7.5 }; clips.push({ prompt: "", duration: Number(prev.duration) || 7.5, seed: null }); commit(node, clips); render(); });
    remove.addEventListener("click", (ev) => { ev.preventDefault(); ev.stopPropagation(); const clips = node.__lmPlannerClips ?? parse(node); if (clips.length <= MIN_CLIPS) return; clips.pop(); commit(node, clips); render(); });

    const editor = node.addDOMWidget("clip_editor", "clip_editor", root, { serialize: false, hideOnZoom: false, getValue: () => null, setValue: () => {} });

    // 0.3.94: IMPORTANT — widget height must never be derived from node.size.
    // ComfyUI calculates node.size from widget computeSize(), so doing that creates
    // a positive feedback loop (node -> widget -> node -> ...), producing the giant
    // vertical nodes seen in 0.3.93.  Keep an independent viewport-height target.
    node.__lmPlannerViewportHeight = Number(node.__lmPlannerViewportHeight) || 340;
    editor.computeSize = function(width) {
        const w = Math.max(180, Number(width) || Number(node.size?.[0]) || 650);
        const h = Math.max(120, Number(node.__lmPlannerViewportHeight) || 340);
        return [w, h];
    };
    editor.__lmPlannerRender = render;
    node.__lmPlannerEditor = editor;
    node.__lmPlannerClips = parse(node);
    setVisible(widget(node, "clips_json"), false);

    // Measure non-editor node chrome once while the editor has a known independent
    // height.  Subsequent user resizes only change the editor viewport to consume
    // the requested remaining height; this keeps node.computeSize() stable.
    if (!Number.isFinite(node.__lmPlannerChromeHeight)) {
        try {
            const total = node.computeSize?.();
            const totalH = Number(total?.[1]);
            if (Number.isFinite(totalH)) {
                node.__lmPlannerChromeHeight = Math.max(40, totalH - node.__lmPlannerViewportHeight);
            }
        } catch (_) {}
        if (!Number.isFinite(node.__lmPlannerChromeHeight)) node.__lmPlannerChromeHeight = 82;
    }

    // 0.3.115: decouple LiteGraph's *minimum* node size from the current
    // user-selected editor viewport height.  LiteGraph calls node.computeSize()
    // before accepting a resize.  If computeSize() sees the expanded editor
    // height, that height becomes a one-way minimum and the node can grow but
    // can no longer shrink.  During minimum-size calculation, temporarily ask
    // the editor for its true minimum viewport height; onResize still updates
    // __lmPlannerViewportHeight to the user's requested runtime height.
    if (!node.__lmPlannerComputeSizeHooked && typeof node.computeSize === "function") {
        node.__lmPlannerComputeSizeHooked = true;
        const previousComputeSize = node.computeSize;
        node.__lmPlannerPreviousComputeSize = previousComputeSize;
        node.computeSize = function(...args) {
            const savedViewportHeight = this.__lmPlannerViewportHeight;
            try {
                this.__lmPlannerViewportHeight = 120;
                return previousComputeSize.apply(this, args);
            } finally {
                this.__lmPlannerViewportHeight = savedViewportHeight;
            }
        };
    }

    if (!node.__lmPlannerResizeHooked) {
        node.__lmPlannerResizeHooked = true;
        const previousOnResize = node.onResize;
        node.onResize = function(size) {
            try { previousOnResize?.apply(this, arguments); } catch (_) {}
            const requestedH = Number(size?.[1]);
            const chromeH = Number(this.__lmPlannerChromeHeight);
            if (Number.isFinite(requestedH) && Number.isFinite(chromeH)) {
                this.__lmPlannerViewportHeight = Math.max(120, requestedH - chromeH);
            }
            try { this.setDirtyCanvas?.(true, true); } catch (_) {}
            try { app.canvas?.setDirty?.(true, true); } catch (_) {}
        };
    }

    // Recover workflows saved by 0.3.93 after the recursive sizing bug expanded
    // the Planner to thousands of pixels.  Normal user-sized nodes are untouched.
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
    name: "MiniMaxH3.LongMediaPlanner.v3",
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
