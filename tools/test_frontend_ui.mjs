import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

function loadFrontend(path, exportedNames, appOverrides = {}) {
    const extensions = [];
    const app = {
        registerExtension(extension) { extensions.push(extension); },
        graph: { links: {}, _nodes: [], setDirtyCanvas() {} },
        canvas: { setDirty() {} },
        ...appOverrides,
    };
    let source = fs.readFileSync(path, "utf8")
        .replace(/^import \{ app \} from .*?;\s*/m, "");
    source += `\n;globalThis.__lmExports = { ${exportedNames.join(", ")} };`;
    const timers = [];
    const context = {
        app,
        console,
        WeakSet,
        requestAnimationFrame: (callback) => { callback(); return 1; },
        queueMicrotask: (callback) => callback(),
        setTimeout: (callback) => { callback(); return 1; },
        setInterval: (callback) => { timers.push(callback); return timers.length; },
        clearInterval() {},
    };
    vm.createContext(context);
    vm.runInContext(source, context, { filename: path });
    return { app, context, extensions, exports: context.__lmExports, timers };
}

function makeSetupNode(values) {
    const node = {
        comfyClass: "MiniMaxH3LatentLabLongMediaSetup",
        widgets: Object.entries(values).map(([name, value]) => ({ name, value, options: {} })),
        inputs: [],
        size: [450, 500],
        graph: { setDirtyCanvas() {} },
        computeSize() {
            const visible = this.widgets.filter((widget) => !widget.options?.hidden).length;
            return [450, 60 + visible * 24];
        },
        setSize(size) { this.size = size; },
        setDirtyCanvas() {},
        addInput(name, type) { this.inputs.push({ name, type, link: null }); },
        removeInput(index) { this.inputs.splice(index, 1); },
    };
    return node;
}

const setupValues = {
    workflow_mode: "manual",
    duration_source: "manual",
    resolution_mode: "match",
    reference_budget: "low",
    video_mode: "auto",
    audio_mode: "generate",
    width: 768,
    height: 768,
    manual_duration: 10,
    video_fps: 24,
    generation_mode: "auto",
    first_frame_mode: "blend",
    first_frame_denoise: 0.77,
    first_frame_blend_frames: 9,
    conditioning_mode: "hybrid_first_last",
    segment_seconds: 5,
    overlap_frames: 22,
};

{
    const loaded = loadFrontend(
        "web/node_facade.js",
        ["lmSetWidgetVisible", "lmRefreshSetup"],
    );
    const { lmSetWidgetVisible, lmRefreshSetup } = loaded.exports;

    // Regression: repeated hidden refreshes must never capture the collapse
    // function as the original computeSize implementation.
    const widget = { name: "overlap_frames", value: 22, options: {} };
    lmSetWidgetVisible(widget, false);
    const collapsed = widget.computeSize;
    lmSetWidgetVisible(widget, false);
    lmSetWidgetVisible(widget, true);
    assert.equal(widget.options.hidden, false);
    assert.notEqual(widget.computeSize, collapsed);
    assert.equal(Object.hasOwn(widget, "computeSize"), false);
    assert.equal(widget.__lmOrigComputeSizeV328, undefined);

    // linkedWidgets can be cyclic in third-party widget implementations.
    const linkedA = { options: {} };
    const linkedB = { options: {} };
    linkedA.linkedWidgets = [linkedB];
    linkedB.linkedWidgets = [linkedA];
    lmSetWidgetVisible(linkedA, false);
    lmSetWidgetVisible(linkedA, true);
    assert.equal(linkedA.options.hidden, false);
    assert.equal(linkedB.options.hidden, false);

    // Manual settings and exact expanded height must survive a mode round trip.
    const node = makeSetupNode(setupValues);
    node.addInput("image_1", "IMAGE");
    node.addInput("image_2", "IMAGE");
    node.addInput("image_3", "IMAGE");
    const get = (name) => node.widgets.find((widgetItem) => widgetItem.name === name);
    lmRefreshSetup(node);
    const firstManualHeight = node.size[1];
    get("workflow_mode").value = "hybrid_auto";
    lmRefreshSetup(node);
    const publicHeight = node.size[1];
    get("workflow_mode").value = "manual";
    lmRefreshSetup(node);
    assert.equal(get("first_frame_mode").value, "blend");
    assert.equal(get("first_frame_denoise").value, 0.77);
    assert.equal(get("first_frame_blend_frames").value, 9);
    assert.equal(get("conditioning_mode").value, "hybrid_first_last");
    for (const name of ["overlap_frames", "conditioning_mode"]) {
        assert.equal(get(name).options.hidden, false, `${name} should be visible in Manual`);
        assert.notDeepEqual(get(name).computeSize?.(), [0, -4]);
    }
    for (const name of ["first_frame_mode", "first_frame_denoise", "first_frame_blend_frames"]) {
        assert.equal(get(name).options.hidden, true, `${name} is legacy-hidden in v0.3.95`);
    }
    assert.ok(firstManualHeight > publicHeight);
    assert.equal(node.size[1], firstManualHeight);
    assert.equal(node.inputs[0].label, "image_1 • picture_1");
    assert.equal(node.inputs[1].label, "image_2 • picture_2");
    assert.equal(node.inputs[2].label, "image_3 • picture_3");

    // Changing the expanded Manual conditioning policy must relabel native
    // Picture ordinals immediately and must not alter the saved setting.
    get("conditioning_mode").value = "hybrid_first_frame";
    lmRefreshSetup(node);
    assert.equal(node.inputs[0].label, "image_1 • picture_1");
    assert.equal(node.inputs[1].label, "image_2 • picture_2");
    assert.equal(node.inputs[2].label, "image_3 • picture_3");
    get("conditioning_mode").value = "auto_refs";
    lmRefreshSetup(node);
    assert.equal(node.inputs[0].label, "image_1 • picture_1");
    assert.equal(node.inputs[1].label, "image_2 • picture_2");

    // v0.3.95: lip-sync is always available through audio_mode and never
    // exposes the retired first-frame tuning widgets.
    get("workflow_mode").value = "hybrid_auto";
    get("audio_mode").value = "lip_sync";
    lmRefreshSetup(node);
    assert.equal(get("generation_mode").options.hidden, true);
    assert.equal(get("first_frame_mode").options.hidden, true);
    assert.equal(get("first_frame_denoise").options.hidden, true);
    assert.equal(get("first_frame_blend_frames").options.hidden, true);

    // Widget callbacks fired during graph restoration must not mutate the
    // partially configured node. afterConfigureGraph performs the final pass.
    const releaseExtension = loaded.extensions.find(
        (item) => item.name === "MiniMaxH3LatentLab.ReleaseFacade030",
    );
    assert.ok(releaseExtension);
    const restored = makeSetupNode({ ...setupValues, workflow_mode: "hybrid_auto" });
    loaded.app.graph._nodes = [restored];
    await releaseExtension.beforeConfigureGraph();
    await releaseExtension.nodeCreated(restored);
    const restoredWorkflow = restored.widgets.find((item) => item.name === "workflow_mode");
    restoredWorkflow.callback?.(restoredWorkflow.value);
    assert.equal(restored.size[1], 500);
    assert.equal(
        restored.widgets.find((item) => item.name === "overlap_frames").options.hidden,
        undefined,
    );
    await releaseExtension.afterConfigureGraph();
    assert.equal(
        restored.widgets.find((item) => item.name === "overlap_frames").options.hidden,
        true,
    );
    assert.ok(restored.size[1] < 500);
    assert.equal(
        restored.widgets.find((item) => item.name === "conditioning_mode").__lmModeCallbackWrapped,
        true,
    );
}

{
    const loaded = loadFrontend(
        "web/long_media_dynamic_inputs.js",
        ["syncNode", "refreshSocketLabels"],
    );
    const { syncNode, refreshSocketLabels } = loaded.exports;
    const node = makeSetupNode({
        workflow_mode: "hybrid_auto",
        audio_mode: "auto",
        generation_mode: "auto",
        conditioning_mode: "auto_refs",
    });
    node.inputs.push(
        { name: "clip", type: "CLIP", link: null },
        { name: "vae", type: "VAE", link: null },
        { name: "audio_vae", type: "VAE", link: null },
    );
    syncNode(node, { repairWidgets: false, fit: true });
    assert.deepEqual(
        node.inputs.map((input) => input.name),
        ["clip", "vae", "audio_vae", "image_1", "video_1", "audio_1"],
    );

    const image1 = node.inputs.find((input) => input.name === "image_1");
    image1.link = 101;
    loaded.app.graph.links[101] = { target_slot: -1 };
    syncNode(node, { repairWidgets: false, fit: true });
    assert.ok(node.inputs.some((input) => input.name === "image_2"));
    assert.equal(loaded.app.graph.links[101].target_slot, node.inputs.indexOf(image1));

    const image2 = node.inputs.find((input) => input.name === "image_2");
    image2.link = 102;
    loaded.app.graph.links[102] = { target_slot: -1 };
    syncNode(node, { repairWidgets: false, fit: true });
    assert.ok(node.inputs.some((input) => input.name === "image_3"));
    image2.link = null;
    syncNode(node, { repairWidgets: false, fit: true });
    assert.equal(node.inputs.some((input) => input.name === "image_3"), false);

    // A connection-triggered dynamic sync must retain lip-sync labels.
    node.widgets.find((widgetItem) => widgetItem.name === "audio_mode").value = "lip_sync";
    refreshSocketLabels(node);
    assert.equal(image1.label, "image_1 • picture_1");
    assert.equal(node.inputs.find((input) => input.name === "audio_1").label, "audio_1 • lip_sync");

    node.widgets.find((widgetItem) => widgetItem.name === "audio_mode").value = "auto";
    node.widgets.find((widgetItem) => widgetItem.name === "workflow_mode").value = "manual";
    node.widgets.find((widgetItem) => widgetItem.name === "conditioning_mode").value = "hybrid_first_frame";
    refreshSocketLabels(node);
    assert.equal(image1.label, "image_1 • picture_1");
    assert.equal(image2.label, "image_2 • picture_2");
    node.widgets.find((widgetItem) => widgetItem.name === "conditioning_mode").value = "hybrid_first_last";
    refreshSocketLabels(node);
    assert.equal(image2.label, "image_2 • picture_2");

    // Dynamic-input extension must not wrap workflow callbacks anymore.
    const workflow = node.widgets.find((widgetItem) => widgetItem.name === "workflow_mode");
    const originalCallback = () => "original";
    workflow.callback = originalCallback;
    const extension = loaded.extensions.find(
        (item) => item.name === "MiniMaxH3LatentLab.LongMediaDynamicInputs.v4",
    );
    assert.ok(extension);
    extension.nodeCreated(node);
    assert.equal(workflow.callback, originalCallback);
    assert.equal(workflow.__lmSocketLabelWrapped, undefined);
}

console.log("FRONTEND_UI_REGRESSION: PASS");
