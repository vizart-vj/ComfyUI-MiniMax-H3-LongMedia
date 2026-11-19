"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "..", "web", "longmedia_director.js"), "utf8");
const start = source.indexOf("function refresh(node)");
const end = source.indexOf('api.addEventListener("minimax_h3_director_preview_ready"', start);
assert.ok(start >= 0 && end > start);

const node = { graph: {}, __lmdTakeStatus: {}, __lmdDoc: null };
const frames = [];
let renders = 0;
let extension;
const context = {
  DIRECTOR_CLASS: "MiniMaxH3LongMediaDirector",
  app: {
    graph: { _nodes: [node] },
    registerExtension(value) { extension = value; },
  },
  isDirector: (value) => value === node,
  loadDirectorUIState() {},
  reconcileDirectorOutputs() {},
  hideWidget() {},
  widget() {},
  ensureEditor() {},
  clampNode() {},
  parseDoc() { return { shots: [{ clip_id: "clip" }] }; },
  render() { renders++; },
  requestAnimationFrame(callback) { frames.push(callback); return frames.length; },
};
vm.runInNewContext(source.slice(start, end), context);
extension.nodeCreated(node);
extension.afterConfigureGraph();
extension.afterConfigureGraph();
assert.equal(renders, 0, "graph hooks defer the expensive rebuild");
assert.equal(frames.length, 1, "one frame is scheduled per Director");
frames.shift()();
assert.equal(renders, 1, "one Director render per load transaction");
extension.afterConfigureGraph();
node.graph = null;
frames.shift()();
assert.equal(renders, 1, "detached nodes skip pending renders");

assert.match(source, /addDOMWidget\("director_surface"[^\n]*hideOnZoom:true/);
