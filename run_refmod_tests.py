#!/usr/bin/env python
"""Run refmod_routing tests without pytest collection issues."""
import importlib.util
import os
import sys
import types

# Stub the parent package before any imports
_pkg = types.ModuleType("ComfyUI-MiniMax-H3-LongMedia")
_pkg.__path__ = []
sys.modules["ComfyUI-MiniMax-H3-LongMedia"] = _pkg
sys.modules["ComfyUI-MiniMax-H3-LongMedia.__init__"] = _pkg

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Load the implementation module
impl_spec = importlib.util.spec_from_file_location(
    "refmod_routing", os.path.join(ROOT, "refmod_routing.py"))
impl = importlib.util.module_from_spec(impl_spec)
sys.modules["refmod_routing"] = impl
impl_spec.loader.exec_module(impl)

# Load the test module (depends on refmod_routing)
test_spec = importlib.util.spec_from_file_location(
    "test_refmod_routing", os.path.join(ROOT, "tests", "test_refmod_routing.py"))
test_mod = importlib.util.module_from_spec(test_spec)
sys.modules["test_refmod_routing"] = test_mod
test_spec.loader.exec_module(test_mod)

# Collect all Test* classes
import unittest
loader = unittest.TestLoader()
suite = unittest.TestSuite()
for name in dir(test_mod):
    obj = getattr(test_mod, name)
    if isinstance(obj, type) and name.startswith("Test"):
        suite.addTests(loader.loadTestsFromTestCase(obj))

runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
