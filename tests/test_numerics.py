import json
from pathlib import Path
import unittest
import numpy as np
from detection.pipeline import compute_statistics
from detection.utils import SummaryAdapter
from detection.core import run_embedding_pipeline
from time_series_preprocessing import create_context_windows

ROOT = Path(__file__).resolve().parents[1]


class NumericalTests(unittest.TestCase):
    def test_reference_outputs(self):
        config = json.loads((ROOT / "configs/example.json").read_text())
        config = config["cpd"]
        with np.load(ROOT / "tests/fixtures/reference.npz") as reference:
            embedded = run_embedding_pipeline(reference["series"], SummaryAdapter(), window_size=32)
            denoised, geometry, statistics, indices = compute_statistics(embedded.embeddings, embedded.ends - 1, config)
            np.testing.assert_array_equal(indices, reference["source_indices"])
            np.testing.assert_allclose(embedded.embeddings, reference["embeddings"], rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(denoised, reference["denoised"], rtol=1e-6, atol=1e-7)
            for name, values in statistics.items():
                np.testing.assert_allclose(values, reference[name], rtol=1e-6, atol=1e-7)

    def test_window_alignment(self):
        windows = create_context_windows(np.arange(12), 4, stride=3)
        np.testing.assert_array_equal(windows.starts, [0, 3, 6])
        np.testing.assert_array_equal(windows.ends, [4, 7, 10])
        np.testing.assert_array_equal(windows.values[:, 0, -1], [3, 6, 9])

    def test_invalid_series(self):
        for values in [[], [1, np.nan], [1, np.inf]]:
            with self.assertRaises(ValueError):
                create_context_windows(values, 2)


if __name__ == "__main__":
    unittest.main()
