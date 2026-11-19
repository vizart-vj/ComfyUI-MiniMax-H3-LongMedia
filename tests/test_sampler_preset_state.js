"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

const sourcePath = path.join(__dirname, "..", "web", "node_facade.js");
const source = fs.readFileSync(sourcePath, "utf8");
const context = {
    app: { registerExtension() {} },
    console,
    setTimeout,
    clearTimeout,
    requestAnimationFrame: (callback) => callback(),
};
vm.runInNewContext(
    `${source.replace(/^import\s+\{\s*app\s*\}\s+from\s+["'][^"']+["'];\s*/m, "")}\nglobalThis.__testHooks = {
        fields: LM_SAMPLER_PRESET_FIELDS,
        snapshot: lmSamplerPresetSnapshot,
        create: lmCreateSamplerPreset,
        overwrite: lmOverwriteSamplerPreset,
        remove: lmDeleteSamplerPreset,
        modifiedCount: lmSamplerPresetModifiedCount,
        updateStatus: lmUpdateSamplerPresetStatus,
        watchChanges: lmInstallSamplerPresetChangeWatcher,
        installUi: lmInstallSamplerPresets,
        installPersistence: lmInstallNamedWidgetPersistence,
        refreshUi: lmRefreshSampler,
    };`,
    context,
    { filename: sourcePath },
);

const hooks = context.__testHooks;
assert.ok(!hooks.fields.includes("seed"), "presets leave the per-run sampler seed untouched");
assert.ok(hooks.fields.includes("vram_activation_reserve_mb"), "presets capture advanced memory settings");
assert.ok(hooks.fields.includes("windowed_refine"), "presets capture Refine settings");
assert.ok(!hooks.fields.includes("refine_steps"), "ignored legacy controls are excluded");

const values = {
    seed: 17,
    sampler_mode: "manual",
    memory_mode: "low_vram",
    latent_hires_enabled: true,
    latent_hires_scale: 1.5,
    refine_enabled: true,
    windowed_refine: false,
    vram_activation_reserve_mb: 1024,
};
const node = {
    widgets: Object.entries(values).map(([name, value]) => ({ name, value })),
    properties: {},
};

const created = hooks.create(node, "Low VRAM · Refine");
assert.ok(created, "a named preset is created");
assert.equal(node.properties.lmSamplerPresetSelectedV060, created.id);
assert.equal(hooks.modifiedCount(node, created), 0);
node.__lmSamplerPresetUiV060 = {
    select: { value: "", options: {} },
    overwrite: {},
    remove: {},
};
hooks.watchChanges(node);
hooks.updateStatus(node);
assert.match(node.__lmSamplerPresetUiV060.select.label, /Saved/);

node.widgets.find((widget) => widget.name === "memory_mode").value = "normal";
node.widgets.find((widget) => widget.name === "seed").value = 18;
assert.equal(hooks.modifiedCount(node, created), 1, "status detects setting changes but ignores seed");
node.widgets.find((widget) => widget.name === "memory_mode").callback();
assert.match(node.__lmSamplerPresetUiV060.select.label, /Modified \(1 setting\)/);

const overwritten = hooks.overwrite(node, created.id);
assert.equal(overwritten.name, "Low VRAM · Refine");
assert.equal(hooks.modifiedCount(node, overwritten), 0, "overwrite updates the baseline");
assert.ok(!Object.prototype.hasOwnProperty.call(overwritten.values, "seed"));

const duplicate = hooks.create(node, "low vram · refine");
assert.equal(duplicate, null, "preset names are unique without case sensitivity");

assert.equal(hooks.remove(node, created.id), true);
assert.equal(node.properties.lmSamplerPresetsV060.length, 0);
assert.equal(node.properties.lmSamplerPresetSelectedV060, "");

const persistentNode = {
    widgets: [{ name: "seed", value: 7 }],
    properties: { lmSamplerPresetsV060: [{ id: "preset-1", name: "Fast", values: { seed: 7 } }] },
    serialize() {
        return { widgets_values: this.widgets.map((widget) => widget.value), properties: { ...this.properties } };
    },
};
hooks.installPersistence(persistentNode);
const serialized = persistentNode.serialize();
assert.equal(serialized.properties.lmSamplerPresetsV060[0].name, "Fast", "presets travel with the saved workflow");

context.prompt = () => "Portable preset";
context.confirm = () => true;
const uiNode = {
    widgets: Object.entries(values).map(([name, value]) => ({ name, value })),
    properties: {},
    addWidget(type, name, value, callback, options) {
        const widget = { type, name, value, callback, options };
        this.widgets.push(widget);
        return widget;
    },
    addDOMWidget() {
        assert.fail("Sampler presets must stay inside the canvas layout");
    },
};
hooks.installUi(uiNode);
assert.equal(uiNode.widgets[0].name, "__lm_sampler_presets_v060");
assert.deepEqual(uiNode.widgets.slice(1, 4).map((widget) => widget.type), ["button", "button", "button"]);
assert.deepEqual(uiNode.widgets.slice(1, 4).map((widget) => widget.label), ["New preset", "Overwrite preset", "Delete preset"]);
const ui = uiNode.__lmSamplerPresetUiV060;
assert.equal(ui.select.type, "combo");
assert.equal(ui.overwrite.disabled, true);
assert.equal(ui.remove.disabled, true);
ui.overwrite.callback();
ui.remove.callback();
assert.equal(uiNode.properties.lmSamplerPresetsV060.length, 0);
ui.create.callback();
assert.equal(uiNode.properties.lmSamplerPresetsV060.length, 1, "New creates a workflow-persisted preset");
assert.match(ui.select.label, /Saved/);
assert.equal(ui.overwrite.disabled, false);
assert.equal(ui.remove.disabled, false);
uiNode.widgets.find((widget) => widget.name === "memory_mode").value = "normal";
uiNode.widgets.find((widget) => widget.name === "memory_mode").callback();
assert.match(ui.select.label, /Modified/);
const seedBeforeApply = uiNode.widgets.find((widget) => widget.name === "seed").value;
ui.select.callback("Portable preset");
assert.equal(uiNode.widgets.find((widget) => widget.name === "memory_mode").value, "low_vram");
assert.equal(uiNode.widgets.find((widget) => widget.name === "seed").value, seedBeforeApply);
uiNode.widgets.find((widget) => widget.name === "memory_mode").value = "normal";
ui.overwrite.callback();
assert.match(ui.select.label, /Saved/);
context.confirm = () => false;
ui.remove.callback();
assert.equal(uiNode.properties.lmSamplerPresetsV060.length, 1, "cancelling keeps the preset");
context.confirm = () => true;
ui.remove.callback();
assert.equal(uiNode.properties.lmSamplerPresetsV060.length, 0);
assert.equal(ui.select.label, "Preset");
assert.equal(ui.overwrite.disabled, true);
assert.equal(ui.remove.disabled, true);

uiNode.properties.lmSamplerPresetsV060 = [{ id: "persist-me", name: "Persistent", values: { memory_mode: "auto" } }];
uiNode.widgets.push({ name: "__lm_group_test", __lmGroupHeader: true, value: undefined });
uiNode.serialize = function () {
    return {
        widgets_values: this.widgets.map((widget) => widget.value),
        properties: { ...this.properties },
    };
};
hooks.installPersistence(uiNode);
const uiSaved = uiNode.serialize();
assert.equal(uiSaved.widgets_values.length, Object.keys(values).length, "presentation-only preset controls do not shift positional widget values");
assert.equal(uiSaved.properties.lmSamplerPresetsV060[0].name, "Persistent", "preset definitions are saved alongside widget values");

const samplerWidgets = Object.entries(values).map(([name, value]) => ({ name, value }));
const sampler = {
    ...uiNode,
    properties: { lmSamplerUiLevel: "advanced" },
    get widgets() { return samplerWidgets; },
    set widgets(next) {
        // Current ComfyUI exposes one mutation view for the life of a node.
        samplerWidgets.splice(0, samplerWidgets.length, ...next);
    },
    serialize() {
        const serializable = this.widgets.filter((widget) => widget.serialize !== false);
        return {
            widgets_values: serializable.map((widget) => widget.value),
            widgets_values_named: Object.fromEntries(serializable.map((widget) => [widget.name, widget.value])),
            properties: { ...this.properties },
        };
    },
};
for (const key of Object.keys(sampler)) {
    if (key.startsWith("__lm")) delete sampler[key];
}
hooks.refreshUi(sampler);
const view = sampler.widgets[0];
assert.equal(view.name, "__lm_sampler_ui_level", "View is the first row in Sampler");
assert.equal(view.type, "combo", "View uses the same layout as model widgets");
assert.equal(view.value, "Advanced", "View reflects the saved workflow property");
const beforeSave = [...sampler.widgets];
assert.equal(sampler.serialize().widgets_values.length, Object.keys(values).length, "UI controls are excluded even on frontends without group headers");
assert.deepEqual(sampler.widgets, beforeSave, "saving cannot remove widgets from ComfyUI's live mutation view");
assert.equal(sampler.properties.__lm_named_widget_state_v040, undefined, "saving leaves live node properties alone");
view.callback("Debug");
assert.equal(sampler.properties.lmSamplerUiLevel, "debug");
sampler.properties = {};
sampler.widgets.push(sampler.widgets.shift());
hooks.refreshUi(sampler);
assert.equal(sampler.widgets[0], view, "an existing View is re-anchored after reset");
assert.equal(view.value, "Basic");
sampler.widgets = Object.entries(values).map(([name, value]) => ({ name, value }));
hooks.refreshUi(sampler);
assert.equal(sampler.widgets[0].name, "__lm_sampler_ui_level", "replaced widgets restore View once");
assert.equal(sampler.widgets[0].value, "Basic");
assert.equal(sampler.widgets.filter((widget) => widget.name.startsWith("__lm_")).length, 5);
assert.equal(sampler.serialize().widgets_values[0], values.seed, "backend input order survives rebuilding the canvas rows");

const legacy = {
    widgets: [{name: "first", value: 17}, {name: "second", value: 23}],
    serialize() {
        const widgets_values = [];
        this.widgets.forEach((widget, index) => {
            if (widget.serialize !== false) widgets_values[index] = widget.value;
        });
        return {widgets_values};
    },
};
hooks.installPersistence(legacy);
legacy.widgets.reverse();
legacy.widgets.unshift({name: "__lm_header", serialize: false, __lmGroupHeader: true});
assert.deepEqual(Array.from(legacy.serialize().widgets_values), [17, 23], "legacy sparse serialization retains backend order");

console.log("Sampler preset state tests passed.");
