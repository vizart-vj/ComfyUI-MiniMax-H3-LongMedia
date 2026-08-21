import { app } from "../../scripts/app.js";

const PUBLIC = new Set([
    "MiniMaxH3LatentLabLongMediaSetup",
    "MiniMaxH3LongMediaPlanner",
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
const LM_COLLAPSED_WIDGET_SIZE = () => [0, -4];
const LM_HIDDEN_WIDGET_DRAW = () => {};

function lmRestoreWidgetProperty(widget, name, hadOwnProperty, originalValue) {
    if (hadOwnProperty) {
        widget[name] = originalValue;
        return;
    }
    // Deleting restores a renderer/prototype implementation when the widget did
    // not originally own the property. Assigning undefined would shadow it.
    try {
        delete widget[name];
    } catch (_) {
        widget[name] = originalValue;
    }
}

function lmSetWidgetVisible(widget, visible, visited = new WeakSet()) {
    if (!widget || visited.has(widget)) return false;
    visited.add(widget);

    const nextVisible = Boolean(visible);
    const nextHidden = !nextVisible;

    // v0.3.28: use a dedicated capture sentinel. Testing the stored value for
    // undefined is incorrect because ordinary Comfy widgets often have no custom
    // computeSize. A second hide pass then captured our [0,-4] collapse function
    // as the "original", making the control stay zero-height after Manual reveal.
    if (!widget.__lmPresentationCapturedV328) {
        widget.__lmPresentationCapturedV328 = true;
        widget.__lmHadOwnComputeSizeV328 = Object.prototype.hasOwnProperty.call(widget, "computeSize");
        widget.__lmHadOwnDrawV328 = Object.prototype.hasOwnProperty.call(widget, "draw");
        widget.__lmOrigComputeSizeV328 = widget.computeSize;
        widget.__lmOrigDrawV328 = widget.draw;
    }

    // Nodes 2.0 / Vue renderer tracks widget visibility through options.hidden.
    // Do NOT change widget.type to "converted-widget": changing the widget type at
    // runtime can leave the Vue widget registry/render key stale until a page reload.
    // Replace the options object (rather than mutating it in place) so reactive
    // frontends can observe the visibility change immediately.
    if (widget.__lmVisibleV328 !== nextVisible || widget.options?.hidden !== nextHidden) {
        widget.options = { ...(widget.options ?? {}), hidden: nextHidden };
    }
    widget.hidden = nextHidden; // compatibility marker used by some legacy extensions
    widget.__lmVisibleV328 = nextVisible;

    if (nextVisible) {
        lmRestoreWidgetProperty(
            widget, "computeSize", widget.__lmHadOwnComputeSizeV328,
            widget.__lmOrigComputeSizeV328,
        );
        lmRestoreWidgetProperty(
            widget, "draw", widget.__lmHadOwnDrawV328,
            widget.__lmOrigDrawV328,
        );
    } else {
        // Legacy LiteGraph fallback: collapse the widget without changing its type.
        widget.computeSize = LM_COLLAPSED_WIDGET_SIZE;
        widget.draw = LM_HIDDEN_WIDGET_DRAW;
    }

    if (widget.linkedWidgets) {
        for (const linked of widget.linkedWidgets) lmSetWidgetVisible(linked, nextVisible, visited);
    }
    return true;
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

function lmInputConnected(node, name) {
    const input = lmFindInput(node, name);
    if (!input) return false;
    if (input.link != null) return true;
    return Array.isArray(input.links) && input.links.some((id) => id != null);
}

function lmWireSetupConnectionRefresh(node) {
    if (!node || node.__lmSetupConnectionRefreshV389) return;
    node.__lmSetupConnectionRefreshV389 = true;
    const original = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
        const result = original?.apply(this, args);
        // LiteGraph calls this before/after link bookkeeping depending on renderer.
        // Defer refreshes so clip_plan.link/links reflects the final connection state.
        queueMicrotask(() => lmRefreshSetup(node));
        requestAnimationFrame(() => lmRefreshSetup(node));
        setTimeout(() => lmRefreshSetup(node), 0);
        setTimeout(() => lmRefreshSetup(node), 50);
        return result;
    };
}

function lmRefreshSetupInputLabels(node) {
    const pictures = [];
    for (let i = 1; i <= 9; i += 1) pictures.push(lmFindInput(node, `image_${i}`));
    const videos = [];
    for (let i = 1; i <= 3; i += 1) videos.push(lmFindInput(node, `video_${i}`));
    const audios = [];
    for (let i = 1; i <= 3; i += 1) audios.push(lmFindInput(node, `audio_${i}`));

    // v0.3.96: native refs keep stable meanings. Workflow/audio policies only annotate.
    pictures.forEach((input, idx) => lmSetInputDisplay(input, `image_${idx + 1} • picture_${idx + 1}`));
    videos.forEach((input, idx) => lmSetInputDisplay(input, `video_${idx + 1}`));
    audios.forEach((input, idx) => lmSetInputDisplay(input, `audio_${idx + 1}`));

    const audioMode = lmWidget(node, 'audio_mode')?.value ?? 'auto';
    if (audioMode === 'lip_sync') {
        lmSetInputDisplay(audios[0], 'audio_1 • lip_sync');
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
    lmSetCombo(node, "audio_mode", ["auto", "preserve", "generate", "reference_only", "preserve_reference", "lip_sync"], "auto");
    lmSetNumber(node, "width", 512, 32, 8192, true);
    lmSetNumber(node, "height", 512, 32, 8192, true);
    lmSetNumber(node, "manual_duration", 5.0, 0.1, 600.0, false);
    lmSetNumber(node, "video_fps", 24.0, 1.0, 120.0, false);

    const workflowValues = ["hybrid_auto", "segmented_continuation", "multiclip", "ref2va_full", "loop", "manual", "video_ref_edit"];
    lmSetCombo(node, "workflow_mode", workflowValues, "hybrid_auto");
    // These legacy widgets are always submitted by ComfyUI even while hidden,
    // so they MUST contain values accepted by the Python INPUT_TYPES validator.
    lmSetCombo(node, "generation_mode", ["auto", "lip_sync"], "auto");
    lmSetCombo(node, "first_frame_mode", ["latent_inject", "pixel_override", "blend"], "latent_inject");
    lmSetCombo(node, "conditioning_mode", ["auto_refs", "hybrid_first_frame", "hybrid_first_last"], "auto_refs");
    lmSetNumber(node, "first_frame_denoise", 0.25, 0.0, 1.0, false);
    lmSetNumber(node, "first_frame_blend_frames", 3, 1, 17, true);
    lmSetNumber(node, "segment_seconds", 8.0, 1.0, 60.0, false);
    lmSetNumber(node, "overlap_frames", 22, 5, 3600, true);

    // v0.3.28: valid Manual values are preserved while public modes hide them.
    // Python already derives the effective public-mode policy. Rewriting these
    // widgets on every refresh made Manual -> public -> Manual destructive.
}

function lmSanitizeSampler(node) {
    lmSetNumber(node, "seed", 0, 0, 18446744073709551615, true);
    lmSetCombo(node, "memory_mode", ["auto", "normal", "low_vram", "ultra_low_vram"], "auto");
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

    // v0.3.22: AUTO widgets stay user-editable. The schema defaults are the
    // production AUTO defaults; changing a value acts as an explicit A/B override.
    // Do not rewrite widget values on refresh/mode changes.

}



// MultiClip clip editing is owned exclusively by MiniMax H3 Long Media Planner.
// Setup keeps multiclip_json only as hidden legacy/backend compatibility storage.

function lmRefreshSetup(node) {
    lmSanitizeSetup(node);
    const mode = lmWidget(node, "workflow_mode")?.value ?? "hybrid_auto";
    const audioMode = lmWidget(node, "audio_mode")?.value ?? "auto";
    const manual = mode === "manual";
    const segmented = mode === "segmented_continuation";
    const externalPlanner = lmInputConnected(node, "clip_plan");
    const multiclip = mode === "multiclip";
    const lipSync = audioMode === "lip_sync";
    // Global duration controls are meaningless in MultiClip: each card owns its duration.
    const segmentDuration = lmWidget(node, "segment_seconds");
    if (segmentDuration) {
        segmentDuration.label = "segment_duration";
        segmentDuration.localized_name = "segment_duration";
    }
    lmSetWidgetVisible(lmWidget(node, "manual_duration"), !multiclip);
    lmSetWidgetVisible(lmWidget(node, "duration_source"), !multiclip);
    lmSetWidgetVisible(segmentDuration, manual || segmented);
    // Planner is the only MultiClip editor. Setup retains this serialized field
    // solely for legacy workflow/backend compatibility and never renders clip cards.
    lmSetWidgetVisible(lmWidget(node, "multiclip_json"), false);
    // Segmentation controls exist only in Manual and Segmented Continuation.
    lmSetWidgetVisible(lmWidget(node, "overlap_frames"), manual || segmented);
    lmSetWidgetVisible(lmWidget(node, "conditioning_mode"), manual);
    // v0.3.95: generation_mode is legacy compatibility storage; lip-sync lives in audio_mode.
    lmSetWidgetVisible(lmWidget(node, "generation_mode"), false);
    // Planner supplies clip data only; Setup always owns workflow selection.
    lmSetWidgetVisible(lmWidget(node, "workflow_mode"), true);
    // Lip-sync-specific first-frame controls become visible immediately when selected.
    for (const name of ["first_frame_mode", "first_frame_denoise", "first_frame_blend_frames"])
        lmSetWidgetVisible(lmWidget(node, name), false);
    lmRefreshSetupInputLabels(node);
    const computed = node.computeSize?.();
    // Preserve the exact user-selected width across workflow_mode changes.
    // Only the height is recomputed when MultiClip widgets appear/disappear.
    const width = Number(node.size?.[0]);
    const height = computed?.[1] ?? node.size?.[1];
    if (Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0) {
        node.setSize?.([width, height]);
    }
    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
    app.canvas?.setDirty?.(true, true);
}

function lmRefreshSampler(node) {
    lmSanitizeSampler(node);
    // v0.3.22 A/B controls: sampler_mode=auto keeps every tuning widget visible.
    // AUTO values begin at the production defaults, but user edits are explicit
    // overrides and must survive refresh/mode changes.
    for (const w of node.widgets ?? []) lmSetWidgetVisible(w, true);
    // v0.3.50 true-refine split: these serialized legacy controls stay in the
    // backend schema for old workflow compatibility but have no valid role in a
    // continuous two-stage diffusion trajectory. Keep them hidden in the UI.
    lmSetWidgetVisible(lmWidget(node, "refine_add_noise"), false);
    lmSetWidgetVisible(lmWidget(node, "refine_seed"), false);
    const refineEnabled = Boolean(lmWidget(node, "refine_enabled")?.value);
    lmSetWidgetVisible(lmWidget(node, "refine_steps"), refineEnabled);
    node.setSize?.([Math.max(node.size?.[0] ?? 0, 420), node.computeSize?.()[1] ?? node.size[1]]);
    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
    app.canvas?.setDirty?.(true, true);
}

let lmConfiguringGraph = false;

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
            if (!lmConfiguringGraph) refresh(this);
        }
        return result;
    };

    const timerKey = `${key}_timer`;
    node[timerKey] = setInterval(() => {
        if (lmConfiguringGraph || !node.graph) return;
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

function lmWireModeCallback(node, modeName, refresh, scheduleInitial = true) {
    const w = lmWidget(node, modeName);
    if (w && !w.__lmModeCallbackWrapped) {
        w.__lmModeCallbackWrapped = true;
        const cb = w.callback;
        w.callback = function (...args) {
            const r = cb?.apply(this, args);
            // Saved widget values may invoke callbacks while LiteGraph is still
            // restoring the node. Defer all presentation work to afterConfigureGraph
            // so half-restored values cannot produce a partial Manual expansion.
            if (!lmConfiguringGraph) {
                requestAnimationFrame(() => refresh(node));
                setTimeout(() => refresh(node), 0);
                setTimeout(() => refresh(node), 50);
            }
            return r;
        };
    }
    lmInstallModeWatcher(node, modeName, refresh);
    if (!scheduleInitial) return;
    requestAnimationFrame(() => refresh(node));
    setTimeout(() => refresh(node), 0);
    setTimeout(() => refresh(node), 50);
}

app.registerExtension({
    name: "MiniMaxH3LatentLab.ReleaseFacade030",
    async beforeConfigureGraph() {
        lmConfiguringGraph = true;
    },
    async nodeCreated(node) {
        const cls = node?.comfyClass ?? node?.constructor?.comfyClass;
        if (cls === "MiniMaxH3LatentLabLongMediaSetup") {
            requestAnimationFrame(() => {
                lmPruneLegacyPromptInput(node);
                setTimeout(() => lmPruneLegacyPromptInput(node), 0);
            });
            lmWireSetupConnectionRefresh(node);
            lmWireModeCallback(node, "workflow_mode", lmRefreshSetup, !lmConfiguringGraph);
            lmWireModeCallback(node, "audio_mode", lmRefreshSetup, !lmConfiguringGraph);
            lmWireModeCallback(node, "generation_mode", lmRefreshSetup, !lmConfiguringGraph);
            lmWireModeCallback(node, "conditioning_mode", lmRefreshSetup, !lmConfiguringGraph);
            if (!lmConfiguringGraph) setTimeout(() => lmRefreshSetup(node), 100);
        } else if (cls === "MiniMaxH3LatentLabLongMediaSampler") {
            lmWireModeCallback(node, "sampler_mode", lmRefreshSampler, !lmConfiguringGraph);
            lmWireModeCallback(node, "refine_enabled", lmRefreshSampler, !lmConfiguringGraph);
            // Some frontend builds install/replace combo callbacks after nodeCreated.
            // The durable watcher above catches that; these immediate passes make the
            // initial state correct without requiring a page reload.
            if (!lmConfiguringGraph) {
                queueMicrotask(() => lmRefreshSampler(node));
                requestAnimationFrame(() => lmRefreshSampler(node));
                setTimeout(() => lmRefreshSampler(node), 100);
            }
        }
    },
    async afterConfigureGraph() {
        lmConfiguringGraph = false;
        for (const node of app.graph?._nodes ?? []) {
            const cls = node?.comfyClass ?? node?.constructor?.comfyClass;
            if (cls === "MiniMaxH3LatentLabLongMediaSetup") {
                lmPruneLegacyPromptInput(node);
                lmWireSetupConnectionRefresh(node);
                lmWireModeCallback(node, "workflow_mode", lmRefreshSetup, false);
                lmWireModeCallback(node, "audio_mode", lmRefreshSetup, false);
                lmWireModeCallback(node, "generation_mode", lmRefreshSetup, false);
                lmWireModeCallback(node, "conditioning_mode", lmRefreshSetup, false);
                lmRefreshSetup(node);
            }
            if (cls === "MiniMaxH3LatentLabLongMediaSampler") {
                lmWireModeCallback(node, "sampler_mode", lmRefreshSampler, false);
                lmWireModeCallback(node, "refine_enabled", lmRefreshSampler, false);
                lmRefreshSampler(node);
            }
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
