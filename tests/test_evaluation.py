import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("evaluation", Path(__file__).resolve().parents[1] / "experiments/evaluate.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class EvaluationTests(unittest.TestCase):
    def test_one_label_cannot_match_twice(self):
        result = evaluation.match_changes([9, 11], [10], 1)
        self.assertEqual(result["matches"], 1)
        self.assertAlmostEqual(result["f1"], 2 / 3)

    def test_inclusive_tolerance_and_empty_inputs(self):
        self.assertEqual(evaluation.match_changes([8, 22], [10, 20], 2)["f1"], 1)
        self.assertEqual(evaluation.match_changes([], [], 0)["f1"], 1)
        self.assertEqual(evaluation.match_changes([], [1], 0)["f1"], 0)
        with self.assertRaises(ValueError):
            evaluation.match_changes([], [], -1)
