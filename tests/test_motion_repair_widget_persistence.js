"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const sourcePath = require("node:path").join(__dirname, "..", "web", "node_facade.js");
const source = fs.readFileSync(sourcePath, "utf8");
const context = {
    app: { registerExtension() {} },
    console,
    setTimeout,
    clearTimeout,
    requestAnimationFrame: (callback) => callback(),
};
vm.runInNewContext(
    `${source.replace(/^import\s+\{\s*app\s*\}\s+from\s+["'][^"']+["'];\s*/m, "")}\nglobalThis.__testHooks = { install: lmInstallNamedWidgetPersistence, move: lmMoveWidgetBefore, restore: lmRestoreNamedWidgetState };`,
    context,
    { filename: sourcePath },
);

const names = [
    "loop_closure_strength",
    "resolution_mode",
    "reference_budget",
    "video_fps",
    "video_mode",
    "audio_mode",
    "conditioning_mode",
    "motion_repair",
    "release_guard",
];
const originalValues = [0.65, "match", "low", 24, "auto", "auto", "auto_refs", "strong", true];
const widgets = names.map((name, index) => ({ name, value: originalValues[index] }));
const node = {
    widgets,
    properties: {},
    serialize() {
        return {
            widgets_values: this.widgets.map((widget) =>
                typeof widget.serializeValue === "function" ? widget.serializeValue() : widget.value,
            ),
            properties: {},
        };
    },
};

context.__testHooks.install(node);
context.__testHooks.move(node, "motion_repair", "resolution_mode");
node.widgets.push({ name: "__lm_setup_group_motion_repair", value: undefined, __lmGroupHeader: true });

assert.deepEqual(node.widgets.filter((widget) => !widget.__lmGroupHeader).map((widget) => widget.name), [
    "loop_closure_strength",
    "motion_repair",
    "resolution_mode",
    "reference_budget",
    "video_fps",
    "video_mode",
    "audio_mode",
    "conditioning_mode",
    "release_guard",
]);

for (const mode of ["off", "auto", "fluid", "strong"]) {
    node.widgets.find((widget) => widget.name === "motion_repair").value = mode;
    const saved = node.serialize();
    const expected = [...originalValues];
    expected[names.indexOf("motion_repair")] = mode;
    assert.equal(
        JSON.stringify(saved.widgets_values),
        JSON.stringify(expected),
        `serialized positional widgets_values must keep '${mode}' in the Motion Repair schema slot`,
    );

    const restored = {
        widgets: names.map((name) => ({ name, value: null })),
        properties: saved.properties,
    };
    for (let index = 0; index < names.length; index += 1) restored.widgets[index].value = saved.widgets_values[index];
    context.__testHooks.restore(restored);
    assert.equal(
        restored.widgets[names.indexOf("motion_repair")].value,
        mode,
        `Motion Repair '${mode}' must survive workflow serialization and page reload`,
    );
}
