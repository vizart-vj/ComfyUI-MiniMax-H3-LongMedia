import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("longmedia_conditioning_contract", ROOT / "conditioning_contract.py")
conditioning_contract = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(conditioning_contract)


class ConditioningContractTests(unittest.TestCase):
    def test_missing_optional_branch_becomes_empty_comfy_condition_list(self):
        copied = conditioning_contract.copy_condition_branches({"positive": [{"text": "scene"}], "negative": None})
        self.assertEqual(copied, {"positive": [{"text": "scene"}], "negative": []})

    def test_required_positive_branch_cannot_be_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, "positive conditioning.*None"):
            conditioning_contract.copy_condition_branches({"positive": None, "negative": []})

    def test_copied_condition_metadata_isolated_from_source(self):
        source = {"positive": [{"strength": 0.5}]}
        copied = conditioning_contract.copy_condition_branches(source)
        copied["positive"][0]["strength"] = 1.0
        self.assertEqual(source["positive"][0]["strength"], 0.5)

    def test_invalid_branch_shape_names_the_condition_key(self):
        with self.assertRaisesRegex(TypeError, "'negative'.*list or tuple"):
            conditioning_contract.copy_condition_branches({"positive": [], "negative": "invalid"})


if __name__ == "__main__":
    unittest.main()
