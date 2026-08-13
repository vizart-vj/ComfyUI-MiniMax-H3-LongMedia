import { app } from "../../scripts/app.js";

const PUBLIC = new Set([
    "MiniMaxH3LatentLabLongMediaSetup",
    "MiniMaxH3LatentLabLongMediaSampler",
    "MiniMaxH3LatentLabLongMediaDecode",
]);
const CATEGORY = "MiniMax H3/Long Media";

app.registerExtension({
    name: "MiniMaxH3LatentLab.NodeFacade.v2",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const cls = nodeType?.comfyClass ?? nodeType?.ComfyClass ?? nodeData?.name;
        if (!cls?.startsWith?.("MiniMaxH3LatentLab")) return;

        if (PUBLIC.has(cls)) {
            nodeType.skip_list = false;
            nodeType.category = CATEGORY;
            if (nodeData) {
                nodeData.hidden = false;
                nodeData.category = CATEGORY;
            }
            return;
        }

        // Keep internal classes registered for GraphBuilder/legacy workflows.
        nodeType.skip_list = true;
        if (nodeData) nodeData.hidden = true;
    },
});

// 0.3.0 release facade.
// IMPORTANT: Python owns the complete backend schema and serialized widget order.
// JS only hides/shows controls and repairs invalid values. Never reorder widgets.
function lmSetWidgetVisible(widget, visible) {
    if (!widget) return;
    if (widget.__lmOrigType === undefined) widget.__lmOrigType = widget.type;
    if (widget.__lmOrigComputeSize === undefined) widget.__lmOrigComputeSize = widget.computeSize;
    widget.hidden = !visible;
    if (visible) {
        widget.type = widget.__lmOrigType;
        widget.computeSize = widget.__lmOrigComputeSize;
    } else {
        widget.type = "converted-widget";
        widget.computeSize = () => [0, -4];
    }
    if (widget.linkedWidgets) for (const linked of widget.linkedWidgets) lmSetWidgetVisible(linked, visible);
}

function lmWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function lmSetInputDisplay(input, label) {
    if (!input) return;
    if (input.__lmOrigName === undefined) input.__lmOrigName = input.name;
    if (input.__lmOrigLabel === undefined) input.__lmOrigLabel = input.label ?? input.name;
    const display = label ?? input.__lmOrigLabel ?? input.__lmOrigName ?? input.name;
    // Keep real slot name intact for backend serialization; only override UI labels.
    input.label = display;
    input.localized_name = display;
}

function lmFindInput(node, name) {
    return node.inputs?.find((i) => i?.name === name);
}

function lmRefreshSetupInputLabels(node) {
    const mode = lmWidget(node, 'workflow_mode')?.value ?? 'hybrid_auto';
    const pictures = [];
    for (let i = 1; i <= 9; i += 1) pictures.push(lmFindInput(node, `image_${i}`));
    const videos = [];
    for (let i = 1; i <= 3; i += 1) videos.push(lmFindInput(node, `video_${i}`));
    const audios = [];
    for (let i = 1; i <= 3; i += 1) audios.push(lmFindInput(node, `audio_${i}`));

    // Reset all dynamic sockets to their baseline labels first.
    pictures.forEach((input, idx) => lmSetInputDisplay(input, `image_${idx + 1}`));
    videos.forEach((input, idx) => lmSetInputDisplay(input, `video_${idx + 1}`));
    audios.forEach((input, idx) => lmSetInputDisplay(input, `audio_${idx + 1}`));

    if (mode === 'hybrid_auto') {
        lmSetInputDisplay(pictures[0], 'image_1 • first_frame');
        lmSetInputDisplay(pictures[1], 'image_2 • last_frame');
        for (let i = 2; i < pictures.length; i += 1) lmSetInputDisplay(pictures[i], `image_${i + 1} • picture_${i - 1}`);
    } else if (mode === 'video_ref_edit') {
        lmSetInputDisplay(videos[0], 'video_1 • source_video');
        lmSetInputDisplay(audios[0], 'audio_1 • source_audio');
        for (let i = 0; i < pictures.length; i += 1) lmSetInputDisplay(pictures[i], `image_${i + 1} • picture_${i + 1}`);
        for (let i = 1; i < videos.length; i += 1) lmSetInputDisplay(videos[i], `video_${i + 1} • extra_video_${i + 1}`);
        for (let i = 1; i < audios.length; i += 1) lmSetInputDisplay(audios[i], `audio_${i + 1} • extra_audio_${i + 1}`);
    } else if (mode === 'ref2va_full') {
        for (let i = 0; i < pictures.length; i += 1) lmSetInputDisplay(pictures[i], `image_${i + 1} • picture_${i + 1}`);
    } else if (mode === 'loop') {
        lmSetInputDisplay(pictures[0], 'image_1 • first+last_frame');
        lmSetInputDisplay(pictures[1], 'image_2 • reserved/ignored');
        for (let i = 2; i < pictures.length; i += 1) lmSetInputDisplay(pictures[i], `image_${i + 1} • picture_${i - 1}`);
    } else if (mode === 'manual') {
        // Manual stays closest to raw backend naming.
    }

    const generationMode = lmWidget(node, 'generation_mode')?.value ?? 'auto';
    if (generationMode === 'lip_sync') {
        lmSetInputDisplay(pictures[0], 'image_1 • lip_sync_identity');
        lmSetInputDisplay(audios[0], 'audio_1 • lip_sync_driver');
    }
}

function lmSetCombo(node, name, values, fallback) {
    const w = lmWidget(node, name);
    if (!w) return;
    if (values.includes(w.value)) return;
    // Some broken 0.3.0 workflows persisted a combo index rather than its value.
    if (Number.isInteger(w.value) && w.value >= 0 && w.value < values.length) {
        w.value = values[w.value];
        return;
    }
    w.value = fallback;
}

function lmSetNumber(node, name, fallback, min, max, integer = false) {
    const w = lmWidget(node, name);
    if (!w) return;
    let value = w.value;
    if (typeof value !== "number" || !Number.isFinite(value)) value = fallback;
    if (integer) value = Math.round(value);
    if (min != null) value = Math.max(min, value);
    if (max != null) value = Math.min(max, value);
    w.value = value;
}

function lmSetBoolean(node, name, fallback) {
    const w = lmWidget(node, name);
    if (!w) return;
    if (typeof w.value !== "boolean") w.value = fallback;
}

function lmSanitizeSetup(node) {
    // Base release widgets: repair only invalid/corrupt values, never overwrite a
    // valid user choice. This covers workflows saved by the early broken 0.3.0.
    lmSetCombo(node, "duration_source", ["auto", "manual", "audio", "video", "longest_input"], "auto");
    lmSetCombo(node, "resolution_mode", ["match", "max"], "match");
    lmSetCombo(node, "reference_budget", ["low", "medium", "high", "max"], "low");
    lmSetCombo(node, "video_mode", ["auto", "preserve", "transform"], "auto");
    lmSetCombo(node, "audio_mode", ["auto", "preserve", "generate", "reference_only", "preserve_reference"], "auto");
    lmSetNumber(node, "width", 512, 32, 8192, true);
    lmSetNumber(node, "height", 512, 32, 8192, true);
    lmSetNumber(node, "manual_duration", 5.0, 0.1, 600.0, false);
    lmSetNumber(node, "video_fps", 24.0, 1.0, 120.0, false);
    lmSetNumber(node, "video_strength", 0.5, 0.0, 1.0, false);
    lmSetNumber(node, "audio_strength", 0.0, 0.0, 1.0, false);

    const workflowValues = ["hybrid_auto", "ref2va_full", "loop", "manual", "video_ref_edit"];
    lmSetCombo(node, "workflow_mode", workflowValues, "hybrid_auto");
    const mode = lmWidget(node, "workflow_mode")?.value ?? "hybrid_auto";

    // These legacy widgets are always submitted by ComfyUI even while hidden,
    // so they MUST contain values accepted by the Python INPUT_TYPES validator.
    lmSetCombo(node, "generation_mode", ["auto", "lip_sync"], "auto");
    lmSetCombo(node, "first_frame_mode", ["latent_inject", "pixel_override", "blend"], "latent_inject");
    lmSetCombo(node, "conditioning_mode", ["auto_refs", "hybrid_first_frame", "hybrid_first_last"], "auto_refs");
    lmSetNumber(node, "first_frame_denoise", 0.25, 0.0, 1.0, false);
    lmSetNumber(node, "first_frame_blend_frames", 3, 1, 17, true);
    lmSetNumber(node, "segment_seconds", 8.0, 1.0, 60.0, false);
    lmSetNumber(node, "overlap_frames", 22, 5, 3600, true);

    // Public modes infer all of these in Python. Force safe validator values so
    // a stale workflow cannot fail before setup() gets a chance to normalize it.
    if (mode !== "manual") {
        const f = lmWidget(node, "first_frame_mode"); if (f) f.value = "latent_inject";
        const d = lmWidget(node, "first_frame_denoise"); if (d) d.value = 0.25;
        const b = lmWidget(node, "first_frame_blend_frames"); if (b) b.value = 3;
        const c = lmWidget(node, "conditioning_mode"); if (c) c.value = "auto_refs";
        const s = lmWidget(node, "segment_seconds"); if (s) s.value = 8.0;
        const o = lmWidget(node, "overlap_frames"); if (o) o.value = 22;
    }
}

function lmSanitizeSampler(node) {
    lmSetNumber(node, "seed", 0, 0, 18446744073709551615, true);
    lmSetCombo(node, "sampler_mode", ["auto", "manual"], "auto");
    const mode = lmWidget(node, "sampler_mode")?.value ?? "auto";
    const specs = [
        ["video_context_denoise", 0.0, 0.0, 1.0, false],
        ["audio_context_denoise", 0.0, 0.0, 1.0, false],
        ["mlp_chunk_tokens", 8192, 0, 131072, true],
        ["sol_tau_start", 1.3, 0.0, 4.0, false],
        ["sol_tau_end", 0.8, 0.0, 4.0, false],
        ["sol_min_tokens", 4096, 256, 131072, true],
        ["sol_dense_percent", 0.0, 0.0, 0.9, false],
        ["sol_qkv_chunk_tokens", 8192, 0, 131072, true],
        ["sol_out_proj_chunk_tokens", 24576, 0, 131072, true],
        ["vram_activation_reserve_mb", 4096, 0, 12288, true],
        ["inter_block_vram_guard_mb", 2048, 0, 8192, true],
        ["inter_block_guard_cooldown_blocks", 4, 0, 32, true],
        ["inter_block_guard_emergency_mb", 512, 0, 4096, true],
        ["inter_block_guard_emergency_cooldown_blocks", 3, 0, 32, true],
        ["late_block_guard_start", 40, 0, 127, true],
        ["late_block_guard_target_mb", 6144, 0, 12288, true],
        ["late_block_guard_min_cached_mb", 512, 0, 4096, true],
        ["step_boundary_cleanup_mb", 2048, 0, 8192, true],
    ];
    lmSetCombo(node, "attention_mode", ["auto", "existing", "sol", "scheduled_sol"], "auto");
    lmSetCombo(node, "sol_curve", ["linear", "cosine", "sqrt", "smoothstep", "exponential", "step"], "linear");
    lmSetCombo(node, "sol_sink_conditioning", ["exact_kv", "exact_kv_and_rows", "off"], "exact_kv");
    lmSetBoolean(node, "offload_completed_segments", true);
    for (const [name, fallback, min, max, integer] of specs) lmSetNumber(node, name, fallback, min, max, integer);

    if (mode === "auto") {
        const defaults = {
            video_context_denoise: 0.0, audio_context_denoise: 0.0,
            offload_completed_segments: true, mlp_chunk_tokens: 8192,
            attention_mode: "auto", sol_tau_start: 1.3, sol_tau_end: 0.8,
            sol_curve: "linear", sol_min_tokens: 4096, sol_dense_percent: 0.0,
            sol_sink_conditioning: "exact_kv", sol_qkv_chunk_tokens: 8192,
            sol_out_proj_chunk_tokens: 24576, vram_activation_reserve_mb: 4096,
            inter_block_vram_guard_mb: 2048, inter_block_guard_cooldown_blocks: 4,
            inter_block_guard_emergency_mb: 512,
            inter_block_guard_emergency_cooldown_blocks: 3,
            late_block_guard_start: 40, late_block_guard_target_mb: 6144,
            late_block_guard_min_cached_mb: 512, step_boundary_cleanup_mb: 2048,
        };
        for (const [name, value] of Object.entries(defaults)) {
            const w = lmWidget(node, name); if (w) w.value = value;
        }
    }
}

function lmRefreshSetup(node) {
    lmSanitizeSetup(node);
    const mode = lmWidget(node, "workflow_mode")?.value ?? "hybrid_auto";
    const generationMode = lmWidget(node, "generation_mode")?.value ?? "auto";
    const manual = mode === "manual";
    const lipSync = generationMode === "lip_sync";
    for (const name of [
        "segment_seconds", "overlap_frames", "conditioning_mode"
    ]) lmSetWidgetVisible(lmWidget(node, name), manual);
    // generation_mode is a public production control, not a legacy/manual-only widget.
    lmSetWidgetVisible(lmWidget(node, "generation_mode"), true);
    // Lip-sync-specific first-frame controls become visible immediately when selected.
    for (const name of ["first_frame_mode", "first_frame_denoise", "first_frame_blend_frames"])
        lmSetWidgetVisible(lmWidget(node, name), manual || lipSync);
    lmRefreshSetupInputLabels(node);
    node.setSize?.([node.size[0], node.computeSize?.()[1] ?? node.size[1]]);
    node.setDirtyCanvas?.(true, true);
}

function lmRefreshSampler(node) {
    lmSanitizeSampler(node);
    const mode = lmWidget(node, "sampler_mode")?.value ?? "auto";
    const manual = mode === "manual";
    const alwaysVisible = new Set(["sampler_mode", "seed"]);
    const manualOnly = new Set([
        "video_context_denoise", "audio_context_denoise", "offload_completed_segments", "mlp_chunk_tokens",
        "attention_mode", "sol_tau_start", "sol_tau_end", "sol_curve", "sol_min_tokens",
        "sol_dense_percent", "sol_sink_conditioning", "sol_qkv_chunk_tokens", "sol_out_proj_chunk_tokens",
        "vram_activation_reserve_mb", "inter_block_vram_guard_mb", "inter_block_guard_cooldown_blocks",
        "inter_block_guard_emergency_mb", "inter_block_guard_emergency_cooldown_blocks", "late_block_guard_start",
        "late_block_guard_target_mb", "late_block_guard_min_cached_mb", "step_boundary_cleanup_mb",
    ]);
    for (const w of node.widgets ?? []) {
        if (alwaysVisible.has(w.name)) {
            lmSetWidgetVisible(w, true);
            continue;
        }
        if (manualOnly.has(w.name)) {
            lmSetWidgetVisible(w, manual);
            continue;
        }
        // Fallback: keep unknown widgets visible in manual mode, hidden in auto.
        lmSetWidgetVisible(w, manual);
    }
    node.setSize?.([Math.max(node.size?.[0] ?? 0, 420), node.computeSize?.()[1] ?? node.size[1]]);
    node.setDirtyCanvas?.(true, true);
}

function lmInstallModeWatcher(node, modeName, refresh) {
    if (!node) return;
    const key = `__lmModeWatcher_${modeName}`;
    if (node[key]) return;
    node[key] = true;
    const lastKey = `${key}_last`;

    // Draw callbacks are useful but not reliable enough on every ComfyUI frontend:
    // extensions can replace onDrawForeground after nodeCreated, and collapsed/offscreen
    // nodes may not be redrawn immediately after a combo changes. Keep the draw watcher
    // as a fast path and add a tiny value-only polling watcher as the durable fallback.
    const priorDraw = node.onDrawForeground;
    node.onDrawForeground = function (...args) {
        const result = priorDraw?.apply(this, args);
        const current = lmWidget(this, modeName)?.value;
        if (current !== this[lastKey]) {
            this[lastKey] = current;
            refresh(this);
        }
        return result;
    };

    const timerKey = `${key}_timer`;
    node[timerKey] = setInterval(() => {
        if (!node.graph) return;
        const current = lmWidget(node, modeName)?.value;
        if (current !== node[lastKey]) {
            node[lastKey] = current;
            refresh(node);
        }
    }, 100);

    const priorRemoved = node.onRemoved;
    node.onRemoved = function (...args) {
        const timer = this[timerKey];
        if (timer != null) {
            clearInterval(timer);
            this[timerKey] = null;
        }
        return priorRemoved?.apply(this, args);
    };
}

function lmWireModeCallback(node, modeName, refresh) {
    const w = lmWidget(node, modeName);
    if (w && !w.__lmModeCallbackWrapped) {
        w.__lmModeCallbackWrapped = true;
        const cb = w.callback;
        w.callback = function (...args) {
            const r = cb?.apply(this, args);
            requestAnimationFrame(() => refresh(node));
            setTimeout(() => refresh(node), 0);
            setTimeout(() => refresh(node), 50);
            return r;
        };
    }
    lmInstallModeWatcher(node, modeName, refresh);
    requestAnimationFrame(() => refresh(node));
    setTimeout(() => refresh(node), 0);
    setTimeout(() => refresh(node), 50);
}

app.registerExtension({
    name: "MiniMaxH3LatentLab.ReleaseFacade030",
    async nodeCreated(node) {
        const cls = node?.comfyClass ?? node?.constructor?.comfyClass;
        if (cls === "MiniMaxH3LatentLabLongMediaSetup") {
            requestAnimationFrame(() => {
                lmPruneLegacyPromptInput(node);
                setTimeout(() => lmPruneLegacyPromptInput(node), 0);
            });
            lmWireModeCallback(node, "workflow_mode", lmRefreshSetup);
            lmWireModeCallback(node, "generation_mode", lmRefreshSetup);
            setTimeout(() => lmRefreshSetup(node), 100);
        } else if (cls === "MiniMaxH3LatentLabLongMediaSampler") {
            lmWireModeCallback(node, "sampler_mode", lmRefreshSampler);
            // Some frontend builds install/replace combo callbacks after nodeCreated.
            // The durable watcher above catches that; these immediate passes make the
            // initial state correct without requiring a page reload.
            queueMicrotask(() => lmRefreshSampler(node));
            requestAnimationFrame(() => lmRefreshSampler(node));
            setTimeout(() => lmRefreshSampler(node), 100);
        }
    },
    async afterConfigureGraph() {
        for (const node of app.graph?._nodes ?? []) {
            const cls = node?.comfyClass ?? node?.constructor?.comfyClass;
            if (cls === "MiniMaxH3LatentLabLongMediaSetup") {
                lmPruneLegacyPromptInput(node);
                lmRefreshSetup(node);
            }
            if (cls === "MiniMaxH3LatentLabLongMediaSampler") lmRefreshSampler(node);
        }
    },
});



// 0.3.0 Setup prompt socket cleanup.
// Older workflows serialized a duplicate forceInput socket named prompt_input.
// The public node now has one prompt field only; ComfyUI can expose/connect the
// widget-backed prompt input directly. Remove the stale duplicate socket while
// preserving the real `prompt` widget/input.
function lmPruneLegacyPromptInput(node) {
    if (!node?.inputs) return false;
    let changed = false;
    for (;;) {
        const index = node.inputs?.findIndex?.((input) => input?.name === "prompt_input") ?? -1;
        if (index < 0) break;
        try {
            node.removeInput?.(index);
        } catch (e) {
            const input = node.inputs[index];
            if (input?.link != null) {
                try { node.disconnectInput?.(index); } catch (_) {}
            }
            node.inputs.splice(index, 1);
        }
        changed = true;
    }
    if (changed) {
        node.setSize?.([node.size[0], node.computeSize?.()[1] ?? node.size[1]]);
        node.setDirtyCanvas?.(true, true);
    }
    return changed;
}

// 0.3.0 Decode socket cleanup.
// Older workflows serialized video_vae/audio_vae as explicit Decode inputs.
// Decode now owns no VAE sockets: both VAEs travel inside LONG_MEDIA_PLAN.
// LiteGraph can preserve removed sockets from a saved workflow, so prune them
// after node creation/configuration as a one-way compatibility migration.
function lmPruneLegacyDecodeVaeInputs(node) {
    if (!node?.inputs) return false;
    let changed = false;
    for (const name of ["audio_vae", "video_vae"]) {
        for (;;) {
            const index = node.inputs?.findIndex?.((input) => input?.name === name) ?? -1;
            if (index < 0) break;
            try {
                node.removeInput?.(index);
            } catch (e) {
                // Fallback for frontends where removeInput is unavailable during
                // early configure: disconnect then splice the stale socket.
                const input = node.inputs[index];
                if (input?.link != null) {
                    try { node.disconnectInput?.(index); } catch (_) {}
                }
                node.inputs.splice(index, 1);
            }
            changed = true;
        }
    }
    if (changed) {
        console.warn("[MiniMaxH3 LongMedia 0.3.0] removed legacy Decode video_vae/audio_vae sockets; VAEs now come from LONG_MEDIA_PLAN");
        node.setSize?.([node.size[0], node.computeSize?.()[1] ?? node.size[1]]);
        node.setDirtyCanvas?.(true, true);
    }
    return changed;
}

app.registerExtension({
    name: "MiniMaxH3LatentLab.DecodeVaeSocketMigration030",
    async nodeCreated(node) {
        const cls = node?.comfyClass ?? node?.constructor?.comfyClass;
        if (cls !== "MiniMaxH3LatentLabLongMediaDecode") return;
        // Run more than once because some frontend versions restore serialized
        // inputs after nodeCreated but before the next animation frame.
        requestAnimationFrame(() => {
            lmPruneLegacyDecodeVaeInputs(node);
            setTimeout(() => lmPruneLegacyDecodeVaeInputs(node), 0);
        });
    },
});
