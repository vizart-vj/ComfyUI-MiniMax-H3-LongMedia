"""VRAM tracker for MiniMax H3 LongMedia multi-pass generation.

Logs per-pass VRAM usage and writes a JSON timeline for visualization.
Drop into workflow between LongMediaSetup and LongMediaSampler nodes.

Usage:
  1. Connect VRAMTracker.start → LongMediaSetup
  2. Connect LongMediaSampler → VRAMTracker.end
  3. Read output report or check web/ memory_timeline.json
"""

import json
import os
import time
from dataclasses import dataclass, field

import torch

CATEGORY = "MiniMax H3/LongMedia/Debug"

@dataclass
class _PassRecord:
    index: int
    timestamp: float
    allocated_mb: float
    reserved_mb: float
    peak_mb: float
    free_mb: float
    label: str = ""


@dataclass
class _Timeline:
    records: list = field(default_factory=list)
    start_time: float = 0.0
    peak_allocated_mb: float = 0.0
    peak_reserved_mb: float = 0.0
    peak_free_mb: float = 0.0

    def snapshot(self, label=""):
        if not torch.cuda.is_available():
            return
        device = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
        res = torch.cuda.memory_reserved(device) / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        free = (torch.cuda.get_device_properties(device).total_memory - res) / (1024 ** 2)
        self.peak_allocated_mb = max(self.peak_allocated_mb, alloc)
        self.peak_reserved_mb = max(self.peak_reserved_mb, res)
        self.peak_free_mb = free
        rec = _PassRecord(
            index=len(self.records),
            timestamp=time.time() - self.start_time,
            allocated_mb=round(alloc, 1),
            reserved_mb=round(res, 1),
            peak_mb=round(peak, 1),
            free_mb=round(free, 1),
            label=label,
        )
        self.records.append(rec)

    def to_json(self):
        return {
            "total_passes": len(self.records),
            "peak_allocated_mb": round(self.peak_allocated_mb, 1),
            "peak_reserved_mb": round(self.peak_reserved_mb, 1),
            "min_free_mb": round(min(r.free_mb for r in self.records), 1) if self.records else 0,
            "records": [
                {
                    "pass": r.index,
                    "time_s": round(r.timestamp, 2),
                    "allocated_mb": r.allocated_mb,
                    "reserved_mb": r.reserved_mb,
                    "peak_mb": r.peak_mb,
                    "free_mb": r.free_mb,
                    "label": r.label,
                }
                for r in self.records
            ],
        }

    def summary(self):
        if not self.records:
            return "No passes recorded."
        lines = [
            f"=== VRAM Timeline ({len(self.records)} passes) ===",
            f"Peak allocated: {self.peak_allocated_mb:.0f} MB",
            f"Peak reserved:  {self.peak_reserved_mb:.0f} MB",
            f"Min free:       {min(r.free_mb for r in self.records):.0f} MB",
            "",
            f"{'Pass':<6} {'Time':>6} {'Alloc':>8} {'Reserv':>8} {'Peak':>8} {'Free':>8}  Label",
            "-" * 72,
        ]
        for r in self.records:
            lines.append(
                f"{r.index:<6} {r.timestamp:>5.1f}s {r.allocated_mb:>7.0f}M {r.reserved_mb:>7.0f}M "
                f"{r.peak_mb:>7.0f}M {r.free_mb:>7.0f}M  {r.label}"
            )
        return "\n".join(lines)


_timeline = None


class VRAMTrackerStart:
    DESCRIPTION = "Start VRAM tracking. Connect output to LongMediaSetup."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "start"
    CATEGORY = CATEGORY

    def start(self, seed):
        global _timeline
        _timeline = _Timeline(start_time=time.time())
        torch.cuda.reset_peak_memory_stats()
        _timeline.snapshot("start")
        print(f"[VRAMTracker] Started tracking. Initial: {_timeline.records[-1].allocated_mb:.0f} MB allocated", flush=True)
        return (seed,)


class VRAMTrackerEnd:
    DESCRIPTION = "End VRAM tracking. Report + timeline JSON."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "final_av": ("LATENT",),
            },
            "optional": {
                "report": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING", "STRING")
    RETURN_NAMES = ("final_av", "report", "json_path")
    FUNCTION = "end"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def end(self, final_av, report=""):
        global _timeline
        if _timeline is None:
            _timeline = _Timeline(start_time=time.time())
        _timeline.snapshot("end")
        summary = _timeline.summary()
        print(f"\n{summary}\n", flush=True)

        # Save JSON timeline
        project_root = os.path.dirname(os.path.dirname(__file__))
        out_dir = os.path.join(os.path.dirname(project_root), "memory_timeline")
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(out_dir, f"timeline_{ts}.json")
        with open(json_path, "w") as f:
            json.dump(_timeline.to_json(), f, indent=2)
        print(f"[VRAMTracker] Timeline saved: {json_path}", flush=True)

        full_report = f"{report}\n\n{summary}" if report else summary
        _timeline = None
        return {"ui": {"text": [summary]}, "result": (final_av, full_report, json_path)}


class VRAMTrackerSnapshot:
    DESCRIPTION = "Take a VRAM snapshot at any point. Use between nodes to pinpoint usage."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": ("*"),
            },
            "optional": {
                "label": ("STRING", {"default": "snapshot", "multiline": False}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("data",)
    FUNCTION = "snapshot"
    CATEGORY = CATEGORY

    def snapshot(self, data, label="snapshot"):
        global _timeline
        if _timeline is None:
            _timeline = _Timeline(start_time=time.time())
        _timeline.snapshot(label)
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"[VRAMTracker] {label}: {alloc:.0f} MB allocated", flush=True)
        return (data,)


NODE_CLASS_MAPPINGS = {
    "VRAMTrackerStart": VRAMTrackerStart,
    "VRAMTrackerEnd": VRAMTrackerEnd,
    "VRAMTrackerSnapshot": VRAMTrackerSnapshot,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRAMTrackerStart": "⚡ VRAM Tracker Start",
    "VRAMTrackerEnd": "⚡ VRAM Tracker End",
    "VRAMTrackerSnapshot": "⚡ VRAM Snapshot",
}
