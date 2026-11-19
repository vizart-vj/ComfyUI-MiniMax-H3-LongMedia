"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(
    path.join(__dirname, "..", "web", "longmedia_director.js"),
    "utf8",
);
const start = source.indexOf("function bridgeDurationFrameCount(durationSeconds)");
const end = source.indexOf("async function queueBridgeRegeneration", start + 1);
assert.ok(start >= 0 && end > start, "bridge duration and resize helpers are standalone functions");

const context = {
    finite(value, fallback) {
        const number = Number(value);
        return Number.isFinite(number) ? number : fallback;
    },
    shotStart(doc, index) {
        const shot = doc.shots[index];
        if (shot && shot.start != null && Number.isFinite(Number(shot.start))) return Math.max(0, Number(shot.start));
        let start = 0;
        for (let i = 0; i < index; i++) start = Math.max(start, this.shotStart(doc, i) + Number(doc.shots[i]?.duration || 0));
        return start;
    },
};
vm.runInNewContext(
    `${source.slice(start, end)}\nglobalThis.bridgeDurationFrameCount = bridgeDurationFrameCount;\nglobalThis.bridgeDurationChoices = bridgeDurationChoices;\nglobalThis.bridgeMenuStatus = bridgeMenuStatus;\nglobalThis.bridgeDurationResizePlan = bridgeDurationResizePlan;\nglobalThis.applyBridgeDurationResize = applyBridgeDurationResize;`,
    context,
);

assert.equal(context.bridgeDurationFrameCount(4.958333), 119);
assert.equal(context.bridgeDurationFrameCount(7.083333), 170);
assert.equal(context.bridgeDurationFrameCount(4.5), null);
assert.equal(context.bridgeDurationFrameCount(6.9), null);
assert.equal(context.bridgeDurationFrameCount(Number.NaN), null);
const durationChoices = context.bridgeDurationChoices(6.9);
assert.equal(durationChoices.shorter.frames, 153);
assert.equal(durationChoices.shorter.durationSeconds, 6.375);
assert.equal(durationChoices.longer.frames, 170);
assert.equal(durationChoices.longer.durationSeconds, 170 / 24);
assert.equal(context.bridgeMenuStatus(4.958333).enabled, true);
assert.equal(context.bridgeMenuStatus(6.9).enabled, false);
assert.match(context.bridgeMenuStatus(6.9).title, /165\.6 frames/);
assert.match(context.bridgeMenuStatus(6.9).title, /6\.375000 s \(153 frames\)/);
assert.match(context.bridgeMenuStatus(6.9).title, /7\.083333 s \(170 frames\)/);
assert.ok(source.includes("const bridgeStatus=bridgeMenuStatus(hit.item.duration);"));
assert.ok(source.includes("{disabled:!bridgeStatus.enabled,title:bridgeStatus.title}"));
assert.ok(source.includes('"Shorten current clip to "+shorterLabel'));
assert.ok(source.includes('"Extend current clip to "+longerLabel'));

function bridgeDocument() {
    const left = { clip_id: "left", start: 0, duration: 7.6, base_kind: "generated" };
    const bridge = { clip_id: "bridge", start: 7.6, duration: 6.9, base_kind: "generated" };
    const right = { clip_id: "right", start: 14.5, duration: 5, base_kind: "generated" };
    const laterLayer = { start: 15, duration: 0.5 };
    return {
        setup_timeline_mode: "multiclip",
        ripple_tracks: true,
        shots: [left, bridge, right],
        camera_blocks: [],
        audio_blocks: [],
        extra_tracks: [{ clips: [laterLayer] }],
        target: bridge,
        laterLayer,
    };
}

const shorter = bridgeDocument();
const shortenPlan = context.bridgeDurationResizePlan(shorter, shorter.target, 153);
assert.equal(shortenPlan.ok, true);
assert.ok(Math.abs(shortenPlan.delta - (6.375 - 6.9)) < 1e-12);
assert.equal(context.applyBridgeDurationResize(shorter, shorter.target, 153).ok, true);
assert.equal(shorter.target.duration, 6.375);
assert.equal(shorter.shots[2].start, 13.975);
assert.equal(shorter.laterLayer.start, 14.475);

const longer = bridgeDocument();
const lengthenPlan = context.bridgeDurationResizePlan(longer, longer.target, 170);
assert.equal(lengthenPlan.ok, true);
assert.equal(context.applyBridgeDurationResize(longer, longer.target, 170).ok, true);
assert.equal(longer.target.duration, 170 / 24);
assert.equal(longer.shots[2].start, 14.5 + (170 / 24 - 6.9));
assert.equal(longer.laterLayer.start, 15 + (170 / 24 - 6.9));

const gap = bridgeDocument();
gap.shots[2].start = 14.75;
const gapResult = context.applyBridgeDurationResize(gap, gap.target, 153);
assert.equal(gapResult.ok, false);
assert.equal(gap.target.duration, 6.9);
assert.equal(gap.shots[2].start, 14.75);

const crossing = bridgeDocument();
crossing.laterLayer.start = 14.4;
crossing.laterLayer.duration = 0.2;
const crossingResult = context.applyBridgeDurationResize(crossing, crossing.target, 153);
assert.equal(crossingResult.ok, false);
assert.equal(crossing.target.duration, 6.9);
assert.equal(crossing.shots[2].start, 14.5);

const noRipple = bridgeDocument();
noRipple.ripple_tracks = false;
context.applyBridgeDurationResize(noRipple, noRipple.target, 153);
assert.equal(noRipple.shots[2].start, 13.975);
assert.equal(noRipple.laterLayer.start, 15);
