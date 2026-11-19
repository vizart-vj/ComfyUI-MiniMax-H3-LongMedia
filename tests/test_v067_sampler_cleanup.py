"""Reproduce the real MultiClip cleanup boundary after stock CFGGuider.sample."""

from __future__ import annotations

import ast
import logging
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sampler_lifecycle import SamplerCleanup, restore_attributes, snapshot_attributes


class PreparedModelCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT / "nodes.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        runtime = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3LatentLabUnifiedRuntimeSampler"
        )
        run = next(node for node in runtime.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        cleanup = next(
            node.finalbody for node in run.body
            if isinstance(node, ast.Try) and any(
                isinstance(child, ast.Constant) and child.value == "additional model cleanup"
                for item in node.finalbody for child in ast.walk(item)
            )
        )
        # The real host deletes its transient attribute after each pass. Apply
        # that behavior before executing the production terminal cleanup.
        terminal = ast.Try(
            body=ast.parse("finish_pass(guider)").body,
            handlers=[], orelse=[], finalbody=cleanup,
        )
        cls.cleanup_code = compile(ast.fix_missing_locations(
            ast.Module(body=[terminal], type_ignores=[]),
        ), str(source), "exec")

    def run_cleanup(self, finish_pass):
        original_options = {}
        guider = SimpleNamespace(model_options=original_options)
        original_state = snapshot_attributes(guider, "model_options", "conds")
        prepared = {"positive": [{"control": object()}], "negative": []}
        loaded = [object()]
        guider.conds = prepared
        guider.model_options = {"runtime": True}
        cleanup_models = Mock()
        patcher = Mock(hook_mode="runtime")
        scope = {
            "sys": sys, "time": time, "SamplerCleanup": SamplerCleanup,
            "_LONGMEDIA_LOG": logging.getLogger(__name__),
            "comfy": SimpleNamespace(sampler_helpers=SimpleNamespace(cleanup_models=cleanup_models)),
            "device": "cpu", "runtime_thread_pool": None, "prepared_patchers": [],
            "guider": guider, "loaded_models": loaded, "prepared_conds": prepared,
            "original_guider_state": original_state, "original_model_options": original_options,
            "original_hook_mode": "before", "model_patcher": patcher,
            "restore_attributes": restore_attributes, "cleanup_started": True,
            "_release_longmedia_transient_cuda_refs": Mock(), "_soft_empty_cuda_cache": Mock(),
            "finish_pass": finish_pass,
        }
        exec(self.cleanup_code, scope)
        cleanup_models.assert_called_once()
        actual_conds, actual_models = cleanup_models.call_args.args
        self.assertIs(actual_conds, prepared)
        self.assertIs(actual_models, loaded)
        self.assertFalse(hasattr(guider, "conds"))
        self.assertIs(guider.model_options, original_options)
        self.assertEqual(patcher.hook_mode, "before")

    def test_cleanup_when_stock_guider_deleted_scratch_conditions(self):
        def stock_sample_exit(guider):
            del guider.conds

        self.run_cleanup(stock_sample_exit)

    def test_cleanup_does_not_use_conditions_of_last_pass(self):
        def alternate_sample_exit(guider):
            guider.conds = {"positive": [{"control": object()}]}

        self.run_cleanup(alternate_sample_exit)


if __name__ == "__main__":
    unittest.main()
