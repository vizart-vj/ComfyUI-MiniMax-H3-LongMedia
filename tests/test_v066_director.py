"""Director route regressions for cache responsiveness and bridge acceptance."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from aiohttp import web


ROOT = Path(__file__).resolve().parents[1]
PROJECT = "v066-director"


def load_director_routes(output_root):
    """Import actual route registration without starting a ComfyUI host."""
    package = "_longmedia_v066_route_tests"
    routes = web.RouteTableDef()
    folder_paths = ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: output_root
    server = ModuleType("server")
    server.PromptServer = SimpleNamespace(instance=SimpleNamespace(routes=routes))
    stubs = {"folder_paths": folder_paths, "server": server}
    for name in ("windows_safetensors_compat", "lora_compat", "fasth3_vsa_compat",
                 "fastvideo_vsa_compat", "nodes"):
        module = ModuleType(f"{package}.{name}")
        if name == "nodes":
            module.NODE_CLASS_MAPPINGS = {}
            module.NODE_DISPLAY_NAME_MAPPINGS = {}
        stubs[module.__name__] = module
    spec = importlib.util.spec_from_file_location(
        package, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**stubs, package: module}):
        spec.loader.exec_module(module)
        cache_type = sys.modules[f"{package}.director_cache"].DirectorClipCache
    return {route.path: route.handler for route in routes}, cache_type


class DirectorRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.routes, self.cache_type = load_director_routes(temp.name)
        self.cache = self.cache_type(temp.name, PROJECT)
        self.video = torch.zeros((1, 24, 2, 2, 2))
        self.audio = torch.zeros((1, 32, 2, 2))

    def save_clip(self, clip_id, index, *, activate=True, metadata=None):
        return self.cache.save_take(
            clip_id=clip_id,
            clip_index=index,
            display_video=self.video,
            display_audio=self.audio,
            continuation_video=self.video,
            continuation_audio=self.audio,
            metadata={"segment_length_frames": 22, "overlap_frames": 5, **(metadata or {})},
            activate=activate,
        )

    async def status(self, *clip_ids):
        handler = self.routes["/longmedia/director/status"]
        return await handler(SimpleNamespace(
            query={"project_id": PROJECT, "clip_ids": ",".join(clip_ids)},
        ))

    async def test_take_publishes_only_after_staging_and_does_not_hold_the_project_lock(self):
        write_sidecar = self.cache.save_take.__globals__["_write_mctx_sidecar"]
        writing = threading.Event()
        resume = threading.Event()

        def slow_write(*args, **kwargs):
            writing.set()
            if not resume.wait(10):
                raise TimeoutError("test did not release the staged write")
            return write_sidecar(*args, **kwargs)

        with patch.dict(self.cache.save_take.__globals__, {"_write_mctx_sidecar": slow_write}):
            save = asyncio.create_task(asyncio.to_thread(self.save_clip, "clip", 0))
            try:
                self.assertTrue(await asyncio.to_thread(writing.wait, 5))
                unlocked = self.cache._mutation_lock.acquire(blocking=False)
                if unlocked:
                    self.cache._mutation_lock.release()
                self.assertTrue(unlocked, "large tensor writes must run outside the project lock")
                self.assertIsNone(self.cache.active_metadata("clip"))
            finally:
                resume.set()
                record = await save
        self.assertEqual(self.cache.active_metadata("clip")["revision"], record["revision"])
        self.assertIsNotNone(self.cache.load_revision("clip", record["revision"]))

    async def accept_bridge(self, revision, timeline, global_prompt):
        handler = self.routes["/longmedia/director/accept_bridge"]

        async def request_body():
            return {
                "project_id": PROJECT,
                "clip_id": "bridge",
                "revision": revision,
                "clip_order": ["left", "bridge", "right"],
                "director_timeline": timeline,
                "global_prompt": global_prompt,
            }

        return await handler(SimpleNamespace(json=request_body))

    def make_pending_bridge(self, *, include_snapshot=True):
        timeline = {
            "project_id": PROJECT,
            "shots": [
                {"clip_id": clip_id, "duration": 1.0, "start": float(index), "prompt": clip_id}
                for index, clip_id in enumerate(("left", "bridge", "right"))
            ],
            "regeneration": {"mode": "none", "clip_id": None, "request_id": None},
        }
        left = self.save_clip("left", 0)
        right = self.save_clip("right", 2)
        metadata = {"request_mode": "bridge"}
        if include_snapshot:
            metadata.update(
                director_timeline_snapshot=timeline,
                director_global_prompt_snapshot="night exterior",
            )
        pending = self.save_clip("bridge", 1, activate=False, metadata=metadata)
        self.cache.set_pending_bridge_metadata("bridge", pending["revision"], {
            "timeline_clip_order": ["left", "bridge", "right"],
            "generated_clip_ids": ["left", "right"],
            "left_clip_id": "left",
            "left_revision": left["revision"],
            "right_clip_id": "right",
            "right_revision": right["revision"],
            "overlap_frames": 5,
            "rebase_manifest": [
                {
                    "clip_id": take["clip_id"],
                    "revision": take["revision"],
                    "clip_index": index,
                    "semantic_fingerprint": f"rebased-{take['clip_id']}",
                }
                for take, index in ((left, 0), (right, 2))
            ],
        }, merge=False)
        return pending["revision"], timeline

    async def test_status_does_not_rewrite_active_seam_metadata(self):
        left = self.save_clip("left", 0)
        right = self.save_clip("right", 1, metadata={"parent_revision": left["revision"]})
        before = self.cache.metadata_for_revision("right", right["revision"])

        response = await self.status("left", "right")

        self.assertEqual(response.status, 200)
        self.assertEqual(self.cache.metadata_for_revision("right", right["revision"]), before)

    async def test_status_scans_current_and_gallery_takes_once(self):
        left = self.save_clip("left", 0)
        right = self.save_clip("right", 1)
        older = self.save_clip("removed-from-timeline", 2)
        scan = self.cache_type._scan_take_workspaces
        scanned = []

        def count_scan(cache):
            scanned.append(cache.project_id)
            return scan(cache)

        with patch.object(self.cache_type, "_scan_take_workspaces", count_scan):
            response = await self.status("left", "right")

        self.assertEqual(response.status, 200)
        status = json.loads(response.body)
        self.assertEqual(status["clips"]["left"]["active_revision"], left["revision"])
        self.assertEqual(status["clips"]["right"]["active_revision"], right["revision"])
        self.assertEqual([take["revision"] for take in status["all_takes"]], [older["revision"]])
        self.assertEqual(scanned, [PROJECT])

    async def test_status_leaves_event_loop_responsive_while_a_take_is_saving(self):
        locked = threading.Event()
        release = threading.Event()
        heartbeat_before_timeout = []

        def simulated_save():
            with self.cache._mutation_lock:
                locked.set()
                heartbeat_before_timeout.append(release.wait(1.0))

        worker = threading.Thread(target=simulated_save, daemon=True)
        worker.start()
        try:
            self.assertTrue(locked.wait(1.0))
            status = asyncio.create_task(self.status("left"))
            # Queue behind the handler. It must run even while the save owns the lock.
            asyncio.get_running_loop().call_soon(release.set)
            response = await asyncio.wait_for(status, timeout=2.0)
        finally:
            release.set()
            worker.join(timeout=2.0)

        self.assertEqual(response.status, 200)
        self.assertEqual(heartbeat_before_timeout, [True])

    async def test_bridge_cannot_accept_a_changed_prompt_with_identical_clip_order(self):
        revision, timeline = self.make_pending_bridge()
        edited = deepcopy(timeline)
        edited["shots"][1]["prompt"] = "a different bridge"

        response = await self.accept_bridge(revision, edited, "night exterior")

        self.assertEqual(response.status, 409)
        self.assertIsNone(self.cache.active_metadata("bridge"))
        self.assertTrue(self.cache.metadata_for_revision("bridge", revision)["pending_acceptance"])

    async def test_bridge_cannot_accept_a_changed_global_prompt(self):
        revision, timeline = self.make_pending_bridge()

        response = await self.accept_bridge(revision, timeline, "day interior")

        self.assertEqual(response.status, 409)
        self.assertIsNone(self.cache.active_metadata("bridge"))

    async def test_bridge_accepts_the_authored_timeline_with_a_new_ui_request_state(self):
        revision, timeline = self.make_pending_bridge()
        current = deepcopy(timeline)
        current["regeneration"] = {"mode": "bridge", "clip_id": "bridge", "request_id": "ui"}

        response = await self.accept_bridge(revision, current, "night exterior")

        self.assertEqual(response.status, 200, response.body)
        self.assertEqual(self.cache.active_metadata("bridge")["revision"], revision)

    async def test_bridge_requires_a_persisted_render_timeline(self):
        revision, timeline = self.make_pending_bridge(include_snapshot=False)

        response = await self.accept_bridge(revision, timeline, "night exterior")

        self.assertEqual(response.status, 409)
        self.assertIsNone(self.cache.active_metadata("bridge"))


if __name__ == "__main__":
    unittest.main()
