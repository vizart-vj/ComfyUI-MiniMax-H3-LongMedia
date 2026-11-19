"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const sourcePath = path.join(__dirname, "..", "web", "director_canvas_input.js");
const source = fs.readFileSync(sourcePath, "utf8");

class FakeHTMLElement {
  constructor() {
    this.listeners = new Map();
    this.captured = new Set();
  }
  addEventListener(name, handler) {
    const listeners = this.listeners.get(name) || [];
    listeners.push(handler);
    this.listeners.set(name, listeners);
  }
  dispatch(name, event) {
    for (const handler of this.listeners.get(name) || []) handler(event);
  }
  setPointerCapture(pointerId) { this.captured.add(pointerId); }
  releasePointerCapture(pointerId) { this.captured.delete(pointerId); }
}

class FakeWheelEvent {
  constructor(type, init) { this.type = type; Object.assign(this, init); }
}

const canvasEvents = [];
const frames = new Map();
let nextFrame = 0;
function runFrame() {
  const callbacks = [...frames.values()];
  frames.clear();
  for (const callback of callbacks) callback();
}
let dirtyCalls = 0;
const comfyCanvas = {
  canvas: { dispatchEvent(event) { canvasEvents.push(event); } },
  ds: { offset: [2, 4], scale: 2 },
  setDirty() { dirtyCalls += 1; },
};
let registeredExtension;
const context = {
  app: {
    canvas: comfyCanvas,
    graph: { _nodes: [] },
    registerExtension(extension) { registeredExtension = extension; },
  },
  HTMLElement: FakeHTMLElement,
  WheelEvent: FakeWheelEvent,
  window: { screenX: 0, screenY: 0 },
  console: { info() {} },
  setInterval,
  clearInterval,
  requestAnimationFrame(callback) { frames.set(++nextFrame, callback); return nextFrame; },
  cancelAnimationFrame(id) { frames.delete(id); },
};

vm.runInNewContext(
  source.replace(/^import\s+\{\s*app\s*\}\s+from\s+["'][^"']+["'];\s*/m, ""),
  context,
  { filename: sourcePath },
);

assert.ok(registeredExtension, "the canvas input extension registers with ComfyUI");
const root = new FakeHTMLElement();
let removedCalled = false;
const node = {
  id: 17,
  comfyClass: "MiniMaxH3LongMediaDirector",
  __lmdRoot: root,
  onRemoved() { removedCalled = true; },
};
registeredExtension.nodeCreated(node);

let prevented = false;
let stopped = false;
root.dispatch("wheel", {
  ctrlKey: true,
  clientX: 230,
  clientY: 140,
  deltaX: 0,
  deltaY: -120,
  deltaZ: 0,
  deltaMode: 0,
  preventDefault() { prevented = true; },
  stopImmediatePropagation() { stopped = true; },
});
assert.equal(prevented, true, "Ctrl+wheel suppresses browser zoom over the Director");
assert.equal(stopped, true, "Director widgets do not also consume the forwarded zoom gesture");
assert.equal(canvasEvents.length, 1);
assert.equal(canvasEvents[0].clientX, 230);
assert.equal(canvasEvents[0].clientY, 140);
assert.equal(canvasEvents[0].deltaY, -120);
assert.equal(canvasEvents[0].ctrlKey, true);

root.dispatch("pointerdown", {
  button: 1,
  pointerId: 4,
  clientX: 20,
  clientY: 30,
  preventDefault() {},
  stopImmediatePropagation() {},
});
const sendMove = (clientX, clientY) => root.dispatch("pointermove", {
  pointerId: 4, clientX, clientY, preventDefault() {}, stopImmediatePropagation() {},
});
for (let index = 1; index <= 60; index++) sendMove(20 + index / 6, 30 - index / 15);
assert.equal(frames.size, 1, "a burst schedules one viewport update");
assert.equal(dirtyCalls, 0, "pointer events do not invalidate the graph before the frame");
runFrame();
assert.deepEqual(comfyCanvas.ds.offset, [7, 2], "MMB pan is scale-compensated in graph coordinates");
assert.equal(dirtyCalls, 1, "the burst requests one redraw");
sendMove(34, 28);
root.dispatch("pointerup", {
  pointerId: 4,
  preventDefault() {},
  stopImmediatePropagation() {},
});
assert.equal(root.captured.has(4), false, "MMB capture is released at the end of the gesture");
assert.equal(frames.size, 0, "release flushes the tail delta and cancels the scheduled update");
root.dispatch("pointermove", {
  pointerId: 4,
  clientX: 40,
  clientY: 26,
  preventDefault() {},
  stopImmediatePropagation() {},
});
assert.deepEqual(comfyCanvas.ds.offset, [9, 3], "release preserves the final movement; later moves are ignored");

node.onRemoved();
assert.equal(removedCalled, true, "Director cleanup preserves its original onRemoved handler");
