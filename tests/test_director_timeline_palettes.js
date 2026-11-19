"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "web", "longmedia_director.js"),
  "utf8",
);
const start = source.indexOf("const TIMELINE_COLOR_PALETTES = {");
assert.notEqual(start, -1, "timeline color palettes are declared");
const objectStart = source.indexOf("{", start);
let depth = 0;
let objectEnd = -1;
for (let index = objectStart; index < source.length; index += 1) {
  if (source[index] === "{") depth += 1;
  if (source[index] === "}") depth -= 1;
  if (depth === 0) {
    objectEnd = index + 1;
    break;
  }
}
assert.notEqual(objectEnd, -1, "timeline color palette object is balanced");
const palettes = vm.runInNewContext(`(${source.slice(objectStart, objectEnd)})`);

assert.ok(Object.keys(palettes).length >= 5, "at least five color schemes are available");
for (const [key, palette] of Object.entries(palettes)) {
  assert.ok(palette.label, `${key} has a user-facing label`);
  assert.ok(palette.selection, `${key} defines a selected-clip outline`);
  assert.ok(palette.main && palette.camera && palette.audio, `${key} colors each core timeline layer`);
  assert.deepEqual(
    Object.keys(palette.tracks).sort(),
    ["audio", "character", "embedding", "prompt", "reference", "refmod", "video"],
    `${key} colors every supported extra layer`,
  );
}

assert.match(source, /saveDirectorUIPref\(node,"timeline_palette",value\)/);
assert.match(source, /const palette=timelineColorPalette\(node\)/);
assert.match(source, /selected\?palette\.selection:palette\.idleBorder/);
