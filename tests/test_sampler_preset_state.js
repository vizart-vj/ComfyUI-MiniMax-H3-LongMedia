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
        installSafeSerialize: lmInstallPresentationSafeSerialize,
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
    select: { value: "" },
    status: { textContent: "", style: {} },
    overwrite: { disabled: false, style: {} },
    remove: { disabled: false, style: {} },
};
hooks.watchChanges(node);
hooks.updateStatus(node);
assert.match(node.__lmSamplerPresetUiV060.status.textContent, /Saved/);

node.widgets.find((widget) => widget.name === "memory_mode").value = "normal";
node.widgets.find((widget) => widget.name === "seed").value = 18;
assert.equal(hooks.modifiedCount(node, created), 1, "status detects setting changes but ignores seed");
node.widgets.find((widget) => widget.name === "memory_mode").callback();
assert.match(node.__lmSamplerPresetUiV060.status.textContent, /Modified \(1 setting\)/);

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

function makeElement() {
    return {
        style: {}, attributes: {}, children: [], value: "", textContent: "", disabled: false,
        append(...items) { this.children.push(...items); },
        replaceChildren(...items) { this.children = [...items]; },
        setAttribute(name, value) { this.attributes[name] = value; },
    };
}
context.document = { createElement: () => makeElement() };
context.prompt = () => "Portable preset";
context.confirm = () => true;
const uiNode = {
    widgets: Object.entries(values).map(([name, value]) => ({ name, value })),
    properties: {},
    addDOMWidget(name, type, element, options) {
        const widget = { name, type };
        if (name === "__lm_sampler_presets_v060") {
            this.presetWidgetElement = element;
            this.presetWidgetOptions = options;
        }
        this.widgets.push(widget);
        return widget;
    },
};
hooks.installUi(uiNode);
assert.equal(uiNode.widgets[0].name, "__lm_sampler_presets_v060", "preset controls are first in widget order, below node inputs");
assert.equal(uiNode.presetWidgetOptions.getMinHeight(), 58, "preset widget reserves space for its status and separator");
assert.equal(uiNode.presetWidgetOptions.getMaxHeight(), 58);
assert.match(uiNode.presetWidgetElement.style.cssText, /border-bottom:1px solid/, "preset panel has a visible divider below the status");
const ui = uiNode.__lmSamplerPresetUiV060;
assert.match(ui.status.textContent, /No preset selected/);
ui.create.onclick({ stopPropagation() {} });
assert.equal(uiNode.properties.lmSamplerPresetsV060.length, 1, "New creates a workflow-persisted preset");
assert.match(ui.status.textContent, /Portable preset · Saved/);
uiNode.widgets.find((widget) => widget.name === "memory_mode").value = "normal";
uiNode.widgets.find((widget) => widget.name === "memory_mode").callback();
assert.match(ui.status.textContent, /Modified \(1 setting\)/, "widget changes update the visible status");
const seedBeforeApply = uiNode.widgets.find((widget) => widget.name === "seed").value;
ui.select.value = uiNode.properties.lmSamplerPresetSelectedV060;
ui.select.onchange();
assert.equal(uiNode.widgets.find((widget) => widget.name === "memory_mode").value, "low_vram", "selecting a preset applies its settings");
assert.equal(uiNode.widgets.find((widget) => widget.name === "seed").value, seedBeforeApply, "applying a preset never changes seed");
assert.match(ui.status.textContent, /Portable preset · Saved/);
uiNode.widgets.find((widget) => widget.name === "memory_mode").value = "normal";
uiNode.widgets.find((widget) => widget.name === "memory_mode").callback();
ui.overwrite.onclick({ stopPropagation() {} });
assert.match(ui.status.textContent, /Portable preset · Saved/, "Overwrite saves the current settings");
ui.remove.onclick({ stopPropagation() {} });
assert.equal(uiNode.properties.lmSamplerPresetsV060.length, 0, "Delete removes the selected preset");
assert.match(ui.status.textContent, /No preset selected/);

uiNode.properties.lmSamplerPresetsV060 = [{ id: "persist-me", name: "Persistent", values: { memory_mode: "auto" } }];
uiNode.widgets.push({ name: "__lm_group_test", __lmGroupHeader: true, value: undefined });
uiNode.serialize = function () {
    return {
        widgets_values: this.widgets.map((widget) => widget.value),
        properties: { ...this.properties },
    };
};
hooks.installPersistence(uiNode);
hooks.installSafeSerialize(uiNode);
const uiSaved = uiNode.serialize();
assert.equal(uiSaved.widgets_values.length, Object.keys(values).length, "presentation-only preset controls do not shift positional widget values");
assert.equal(uiSaved.properties.lmSamplerPresetsV060[0].name, "Persistent", "preset definitions are saved alongside widget values");

console.log("Sampler preset state tests passed.");
