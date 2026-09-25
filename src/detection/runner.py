"""Frozen experiment configuration, modality adapters, and shared CPD stages."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from detection.core import run_embedding_pipeline
from .utils import SummaryAdapter, load_series, load_config
from .utils import json_save
from .pipeline import compute_statistics


def input_digest(path):
    path = Path(path)
    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    for file in files:
        digest.update((file.relative_to(path).as_posix() if path.is_dir() else file.name).encode())
        with file.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def extract(job, destination):
    modality, source = job['modality'], job['input']
    if modality == 'embeddings':
        with np.load(source['path'], allow_pickle=False) as data:
            values = data['embeddings'].copy()
            # Explicit mapping required: never guess frame base or window delay.
            indices = data['source_indices'].copy()
        return values, indices
    model = job['model']
    if modality == 'time_series':
        values = load_series(source)
        embedding = job['embedding']
        if model['backend'] == 'summary':
            adapter = SummaryAdapter()
        else:
            if not Path(model['path']).is_dir():
                raise FileNotFoundError(model['path'])
            from detection.foundation_adapters import load_foundation_adapter
            import torch
            torch.manual_seed(job['cpd']['windows']['seed'])
            adapter = load_foundation_adapter(model['backend'], Path(model['path']),
                embedding['device'], timesfm1_source=Path(model.get('timesfm1_source', '.')))
        result = run_embedding_pipeline(values, adapter,
            window_size=embedding['context_length'], stride=embedding['stride'],
            batch_size=model['batch_size'])
        return result.embeddings, result.ends - 1
    from .vision_adapters import embed
    from video_preprocessing import preprocess
    from .defaults import vision_config
    cfg = vision_config(job)
    frames = Path(source['path'])
    if not frames.is_dir():
        frames = destination / 'frames'
        preprocess(source['path'], frames, cfg['preprocessing'])
    embed(frames, destination / 'vision_embeddings.npz', model['name'], cfg)
    with np.load(destination / 'vision_embeddings.npz', allow_pickle=False) as data:
        return data['embeddings'].copy(), data['frame_numbers'].copy() - 1


def run_stage(config, stage):
    """Run one extraction or SCAN stage for every resolved modality job."""
    if stage not in ('extract', 'scan'):
        raise ValueError('stage must be extract or scan')
    root = Path(config['output_dir'])
    identity = {'config': config, 'input_sha256': {
        j['id']: input_digest(j['input']['path']) for j in config['jobs']}}
    marker = root / 'run_identity.json'
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8')) != identity:
            raise ValueError('Configuration or inputs changed; use a new output directory')
    elif root.exists() and any(root.iterdir()):
        raise ValueError('Output directory must be empty for a new run')
    json_save(marker, identity)
    versions = {}
    for name in ('numpy', 'scipy', 'scikit-learn', 'scan-py', 'torch', 'torchvision'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    json_save(root / ('environment_' + stage + '.json'), {'python': sys.version, 'packages': versions})
    for job in config['jobs']:
        directory = root / job['id']
        directory.mkdir(parents=True, exist_ok=True)
        if stage == 'extract':
            values, indices = extract(job, directory)
            denoised, geometry, statistics, mapping = compute_statistics(values, indices, job['cpd'])
            np.savez_compressed(directory / 'embeddings.npz', embeddings=values, source_indices=indices)
            np.savez_compressed(directory / 'tv_embeddings.npz', embeddings=denoised)
            np.savez_compressed(directory / 'covariance.npz', **geometry)
            np.savez_compressed(directory / 'statistics.npz', **statistics, source_indices=mapping)
        else:
            from detection.detection import detect_statistics
            with np.load(directory / 'statistics.npz', allow_pickle=False) as data:
                statistics = {k: data[k] for k in data.files if k != 'source_indices'}
                report = detect_statistics(statistics, data['source_indices'], job['cpd'])
            json_save(directory / 'detections.json', {
                'schema_version': 1, 'job': job['id'], 'modality': job['modality'],
                'model': job.get('model', {}).get('key', job.get('model', {}).get('name', 'precomputed')),
                'index_convention': 'zero-based source row/frame; right embedding anchor',
                'statistics': report})
        print(f"{job['id']}: {stage} complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=('all', 'extract', 'scan'), default='all')
    args = parser.parse_args()
    # These adapters consume already-downloaded assets.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    config = load_config(args.config)
    if args.stage == 'all':
        for stage, key in (('extract', 'foundation_python'), ('scan', 'scan_python')):
            executable = config.get('runtime', {}).get(key) or sys.executable
            subprocess.run([executable, '-m', 'detection', '--config', str(Path(args.config).resolve()), '--stage', stage], check=True)
    else:
        run_stage(config, args.stage)
