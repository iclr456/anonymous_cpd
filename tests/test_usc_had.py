import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from detection.utils import load_config, load_series


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_usc_had', ROOT / 'examples/prepare_usc_had.py')
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class UscHadTests(unittest.TestCase):
    def test_split_matches_main_pipeline(self):
        train, test = example.split_series(reversed(example.SERIES))
        self.assertEqual(train, ['USC-HAD-TS1', 'USC-HAD-TS5'])
        self.assertEqual(test, ['USC-HAD-TS2', 'USC-HAD-TS3', 'USC-HAD-TS4', 'USC-HAD-TS6'])
        self.assertFalse(set(train) & set(test))

    def test_bundled_data_and_configs_agree(self):
        folder = ROOT / 'examples/usc_had'
        manifest = json.loads((folder / 'split.json').read_text())
        for partition in ('train', 'test'):
            config = load_config(folder / f'{partition}.json')
            self.assertEqual([job['id'] for job in config['jobs']], manifest[partition])
            for job in config['jobs']:
                values = load_series(job['input'])
                self.assertEqual(values.shape, (manifest['series'][job['id']]['samples'], 3))
                labels = json.loads((folder / 'labels' / f"{job['id']}.json").read_text())['change_points']
                self.assertEqual(len(labels), 5)
                self.assertTrue(all(0 < point < len(values) for point in labels))
                self.assertEqual(labels, sorted(set(labels)))

    def test_preparation_keeps_labels_out_of_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'csv'
            source.mkdir()
            for name in example.SERIES:
                (source / f'{name}.csv').write_text(
                    'timestamp,acc_x,acc_y,acc_z,changepoint\n'
                    '0,1,2,3,0\n1,4,5,6,1\n', encoding='utf-8')
            output = Path(temporary) / 'prepared'
            example.prepare(source, output)
            with np.load(output / 'data/USC-HAD-TS1.npz') as archive:
                np.testing.assert_array_equal(archive['values'], [[1, 2, 3], [4, 5, 6]])
            self.assertEqual(json.loads((output / 'labels/USC-HAD-TS1.json').read_text()),
                             {'change_points': [1]})
            with self.assertRaises(ValueError):
                example.prepare(source, output)


if __name__ == '__main__':
    unittest.main()
