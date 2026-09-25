import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tsfm_cpd.utils import generate_window_sizes
from tsfm_cpd.detection import detect


class PipelineTests(unittest.TestCase):
    def test_window_bounds(self):
        for T in range(97, 3000):
            windows = generate_window_sizes(T)
            self.assertEqual(len(set(windows)), 7)
            self.assertEqual(windows, sorted(windows))
            self.assertGreaterEqual(min(windows), 15)
            self.assertLessEqual(max(windows) ** 3, T ** 2)
            self.assertLessEqual(2 * max(windows), T)
            if int(T ** (2 / 3)) >= 45:
                self.assertTrue(all(w % 5 == 0 for w in windows))

    def test_short_and_random_fallback(self):
        for T in [1, 30, 60, 96]:
            with self.assertRaises(ValueError):
                generate_window_sizes(T)
        self.assertEqual(generate_window_sizes(125), generate_window_sizes(125))
        self.assertTrue({15, 20, 25}.issubset(generate_window_sizes(125)))
        self.assertTrue(any(w % 5 for w in generate_window_sizes(125)))
        self.assertEqual(generate_window_sizes(1000), [15, 30, 45, 55, 70, 85, 100])

    def test_scan_thresholds_and_alignment(self):
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs' / 'example.json').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'synthetic'
            directory.mkdir()
            config['output_dir'] = temporary
            config['models'] = [{'key': 'synthetic'}]
            config['scan']['n_boot'] = 19
            rng = np.random.default_rng(7)
            values = np.r_[rng.normal(0, .1, 150), rng.normal(4, .1, 150)]
            np.savez(directory / 'statistics.npz', shifted=values, constant=np.ones(300),
                     source_indices=np.arange(300) * 2 + 128)
            detect(config)
            report = json.loads((directory / 'detections.json').read_text())
            for metric in report['statistics'].values():
                previous = set(range(300))
                for threshold in ['0.25', '0.4', '0.5', '0.6']:
                    detected = metric['detections'][threshold]
                    cps = set(detected['statistic_indices'])
                    self.assertTrue(cps <= previous)
                    self.assertEqual(detected['source_indices'],
                                     [cp * 2 + 128 for cp in detected['statistic_indices']])
                    previous = cps
            self.assertEqual(report['statistics']['constant']['detections']['0.25']['source_indices'], [])


if __name__ == '__main__':
    unittest.main()
