import os
import subprocess
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from detection.runner import extract, run_stage
from detection.utils import load_config
from detection.pipeline import compute_statistics as compute
from detection.utils import SummaryAdapter
from detection.core import run_embedding_pipeline

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT/'configs/unified_example.json').read_text())['cpd']

class UnifiedTests(unittest.TestCase):
    def test_time_series_job_uses_context_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.npy'
            values = np.random.default_rng(1).normal(size=(400, 2))
            np.save(path, values)
            job = {'modality': 'time_series', 'input': {'path': str(path)},
                   'model': {'backend': 'summary', 'batch_size': 32},
                   'embedding': {'context_length': 16, 'stride': 2}}
            embeddings, indices = extract(job, Path(tmp))
            expected = run_embedding_pipeline(values, SummaryAdapter(), window_size=16, stride=2)
            np.testing.assert_array_equal(indices, expected.ends - 1)
            np.testing.assert_array_equal(embeddings, expected.embeddings)

    def test_vision_mapping_and_invalid_mapping(self):
        values = np.random.default_rng(3).normal(size=(100, 4))
        config = copy.deepcopy(CONFIG)
        config['tv']['drop_edge_statistics'] = False
        indices = np.arange(100)*3
        *_, mapped = compute(values, indices, config)
        np.testing.assert_array_equal(mapped, indices[1:])
        with self.assertRaises(ValueError):
            compute(values, np.zeros(100, dtype=int), config)

    def test_staged_run_and_input_change_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = np.random.default_rng(4).normal(size=(100, 4))
            np.savez(root/'input.npz', embeddings=values, source_indices=np.arange(100))
            config = {'schema_version':1, 'output_dir':'out', 'cpd':copy.deepcopy(CONFIG),
                'jobs':[{'id':'clip', 'modality':'embeddings','input':{'path':'input.npz'}}]}
            config['cpd']['scan']['n_boot']=3
            (root/'config.json').write_text(json.dumps(config))
            resolved=load_config(root/'config.json')
            run_stage(resolved, 'extract')
            if os.environ.get('CPD_SCAN_PYTHON'):
                subprocess.run([os.environ['CPD_SCAN_PYTHON'], '-m', 'detection',
                                '--config', str(root/'config.json'), '--stage', 'scan'], check=True)
            else:
                run_stage(resolved, 'scan')
            report=json.loads((root/'out/clip/detections.json').read_text())
            self.assertEqual(len(report['statistics']), 4)
            for stat in report['statistics'].values():
                self.assertEqual(len(stat['detections']), 4)
            np.savez(root/'input.npz', embeddings=values+1, source_indices=np.arange(100))
            with self.assertRaisesRegex(ValueError, 'changed'):
                run_stage(resolved, 'scan')

if __name__ == '__main__':
    unittest.main()

