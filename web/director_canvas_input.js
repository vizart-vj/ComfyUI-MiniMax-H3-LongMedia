import { app } from "../../scripts/app.js";

const DIRECTOR_CLASS = "MiniMaxH3LongMediaDirector";
const PATCH_NAME = "MiniMaxH3.LongMediaDirector.CanvasInput";

function isDirector(node) {
  const className = node?.comfyClass
    ?? node?.ComfyClass
    ?? node?.constructor?.comfyClass
    ?? node?.constructor?.ComfyClass;
  return className === DIRECTOR_CLASS;
}

function getCanvas() {
  const canvas = app?.canvas;
  return canvas?.canvas && canvas?.ds?.offset ? canvas : null;
}

function forwardCanvasZoom(event) {
  const canvas = getCanvas();
  if (!canvas) return false;

  const forwarded = new WheelEvent("wheel", {
    bubbles: true,
    cancelable: true,
    clientX: event.clientX,
    clientY: event.clientY,
    screenX: window.screenX + event.clientX,
    screenY: window.screenY + event.clientY,
    deltaX: event.deltaX,
    deltaY: event.deltaY,
    deltaZ: event.deltaZ,
    deltaMode: event.deltaMode,
    ctrlKey: true,
    shiftKey: event.shiftKey,
    altKey: event.altKey,
    metaKey: event.metaKey,
    button: 0,
    buttons: 0,
  });
  canvas.canvas.dispatchEvent(forwarded);
  return true;
}

function panCanvas(dx, dy) {
  const canvas = getCanvas();
  if (!canvas) return;

  const scale = Number(canvas.ds.scale);
  if (!Number.isFinite(scale) || scale <= 0) return;

  canvas.ds.offset[0] += dx / scale;
  canvas.ds.offset[1] += dy / scale;
  canvas.setDirty?.(true, true);
}

function install(node) {
  if (!isDirector(node) || node.__lmdCanvasInputInstalled) return;

  const tryInstall = () => {
    if (!isDirector(node) || node.__lmdCanvasInputInstalled) return true;
    const root = node.__lmdRoot;
    if (!(root instanceof HTMLElement)) return false;

    node.__lmdCanvasInputInstalled = true;
    let middleDragging = false;
    let pointerId = null;
    let lastX = 0;
    let lastY = 0;
    let frame = null;
    let pendingX = 0;
    let pendingY = 0;
    const flushPan = () => {
      frame = null;
      if (pendingX || pendingY) panCanvas(pendingX, pendingY);
      pendingX = pendingY = 0;
    };

    root.addEventListener("wheel", (event) => {
      if (!event.ctrlKey || !forwardCanvasZoom(event)) return;
      event.preventDefault();
      event.stopImmediatePropagation();
    }, {capture: true, passive: false});

    root.addEventListener("pointerdown", (event) => {
      if (event.button !== 1 || middleDragging) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      middleDragging = true;
      pointerId = event.pointerId;
      lastX = event.clientX;
      lastY = event.clientY;
      try { root.setPointerCapture(event.pointerId); } catch (_) {}
    }, {capture: true});

    root.addEventListener("pointermove", (event) => {
      if (!middleDragging || event.pointerId !== pointerId) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      pendingX += event.clientX - lastX;
      pendingY += event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      if (frame === null && (pendingX || pendingY)) frame = requestAnimationFrame(flushPan);
    }, {capture: true});

    const finishMiddleDrag = (event) => {
      if (!middleDragging || (event.pointerId !== undefined && event.pointerId !== pointerId)) return;
      event.preventDefault?.();
      event.stopImmediatePropagation?.();
      middleDragging = false;
      if (frame !== null) cancelAnimationFrame(frame);
      flushPan();
      try { root.releasePointerCapture(pointerId); } catch (_) {}
      pointerId = null;
    };
    root.addEventListener("pointerup", finishMiddleDrag, {capture: true});
    root.addEventListener("pointercancel", finishMiddleDrag, {capture: true});
    root.addEventListener("lostpointercapture", finishMiddleDrag, {capture: true});

    const removed = node.onRemoved;
    node.onRemoved = function (...args) {
      middleDragging = false;
      if (frame !== null) cancelAnimationFrame(frame);
      frame = null;
      pendingX = pendingY = 0;
      try { if (pointerId !== null) root.releasePointerCapture(pointerId); } catch (_) {}
      pointerId = null;
      return removed?.apply(this, args);
    };

    console.info(`[${PATCH_NAME}] installed`, node.id);
    return true;
  };

  if (tryInstall()) return;
  if (node.__lmdCanvasInputTimer != null) return;
  let attempts = 0;
  const removed = node.onRemoved;
  const stopPolling = () => {
    clearInterval(node.__lmdCanvasInputTimer);
    node.__lmdCanvasInputTimer = null;
  };
  node.onRemoved = function (...args) {
    stopPolling();
    return removed?.apply(this, args);
  };
  node.__lmdCanvasInputTimer = setInterval(() => {
    attempts += 1;
    if (tryInstall() || attempts >= 40) stopPolling();
  }, 250);
}

app.registerExtension({
  name: PATCH_NAME,
  async nodeCreated(node) { install(node); },
  async afterConfigureGraph() {
    for (const node of app.graph?._nodes ?? []) install(node);
  },
});
