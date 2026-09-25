import copy
import json
from pathlib import Path
import tempfile
import unittest

from detection.utils import load_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_all_supplied_configs_use_jobs(self):
        for path in (ROOT / 'configs').glob('*.json'):
            config = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(config['schema_version'], 1)
            self.assertTrue(config['jobs'])
            self.assertNotIn('models', config)
        foundation = load_config(ROOT / 'configs/foundation.json')
        self.assertEqual(len(foundation['jobs']), 13)
        timesfm = next(job for job in foundation['jobs'] if job['model']['backend'] == 'timesfm1')
        self.assertEqual(Path(timesfm['model']['timesfm1_source']), ROOT / 'models/timesfm1')

    def test_overrides_do_not_modify_other_jobs(self):
        config = load_config(ROOT / 'configs/unified_example.json')
        self.assertTrue(config['jobs'][0]['cpd']['tv']['drop_edge_statistics'])
        self.assertFalse(config['jobs'][1]['cpd']['tv']['drop_edge_statistics'])
        self.assertTrue(config['cpd']['tv']['drop_edge_statistics'])

    def test_invalid_schema_and_duplicate_ids(self):
        valid = load_config(ROOT / 'configs/example.json')
        invalid_cases = [{'models': []}, copy.deepcopy(valid), copy.deepcopy(valid)]
        invalid_cases[1]['jobs'].append(copy.deepcopy(valid['jobs'][0]))
        invalid_cases[2]['jobs'][0]['embedding']['stride'] = 0
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'config.json'
            for config in invalid_cases:
                path.write_text(json.dumps(config), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_config(path)


if __name__ == '__main__':
    unittest.main()
