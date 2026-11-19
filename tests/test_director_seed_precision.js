"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const sourcePath = path.join(__dirname, "..", "web", "longmedia_director.js");
const source = fs.readFileSync(sourcePath, "utf8");
const start = source.indexOf("function normalizeShot(s,i,ids)");
const end = source.indexOf("\nfunction ", start + 1);
assert.ok(start >= 0 && end > start, "normalizeShot is present as a standalone function");

const context = {
    PROMPT_GUIDANCE_OPTIONS: [],
    clamp(value, minimum, maximum, fallback) {
        const number = Number(value);
        return Number.isFinite(number) ? Math.max(minimum, Math.min(maximum, number)) : fallback;
    },
    finite(value, fallback) {
        const number = Number(value);
        return Number.isFinite(number) ? number : fallback;
    },
    id: () => "generated-id",
    normalizeMedia: (value) => value ?? null,
};
vm.runInNewContext(`${source.slice(start, end)}\nglobalThis.normalizeShot = normalizeShot;`, context);

const maxUint64 = "18446744073709551615";
const huge = context.normalizeShot({ clip_id: "huge", seed: maxUint64 }, 0, new Set());
assert.equal(huge.seed, maxUint64, "uint64 seed text remains exact across JS document normalization");

const ordinary = context.normalizeShot({ clip_id: "ordinary", seed: "12345" }, 0, new Set());
assert.equal(ordinary.seed, 12345, "safe decimal seeds keep the existing numeric representation");

const empty = context.normalizeShot({ clip_id: "empty", seed: "" }, 0, new Set());
assert.equal(empty.seed, null, "empty seed remains index-derived");
