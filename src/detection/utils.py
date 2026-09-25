"""Configuration, input loading, artifact persistence, summary embeddings, and SCAN utilities."""
import copy
import json
import math
import random
from pathlib import Path
import numpy as np
from time_series_preprocessing import validate_series


class SummaryAdapter:
    """Window summaries for the example only; not a foundation-model substitute."""

    def embed(self, batch):
        features = np.stack([
            batch.mean(axis=-1), batch.std(axis=-1),
            batch.min(axis=-1), batch.max(axis=-1),
        ], axis=-1)
        return features.mean(axis=1).astype(np.float32)


def load_series(settings):
    path = Path(settings["path"])
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            values = archive[settings["npz_key"]]
    else:
        values = np.loadtxt(path, delimiter=settings.get("delimiter", ","),
                            skiprows=settings.get("skiprows", 0),
                            usecols=settings.get("columns"), ndmin=2)
    return validate_series(values)


def generate_window_sizes(T: int, min_window: int = 15, n_windows: int = 7,
                          seed: int = 0) -> list[int]:
    """Spread multiples of five across [min_window, floor(T**(2/3))].

    T is the number of statistic observations sent to SCAN. Its two-sided
    constraint also requires window <= T//2. When there are too few multiples
    of five, retain all of them and sample remaining integers without replacement.
    Reject impossible requests instead of duplicating ensemble members.
    """
    for name, value in (("T", T), ("min_window", min_window), ("n_windows", n_windows)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    # Correct floating-point rounding at perfect cubes using exact integers.
    upper = int(T ** (2 / 3))
    while (upper + 1) ** 3 <= T ** 2:
        upper += 1
    while upper ** 3 > T ** 2:
        upper -= 1
    upper = min(upper, T // 2)
    if upper - min_window + 1 < n_windows:
        raise ValueError(f"T={T}: cannot select {n_windows} distinct windows in [{min_window}, {upper}]")
    preferred = list(range(math.ceil(min_window / 5) * 5, upper + 1, 5))
    if len(preferred) >= n_windows:
        if n_windows == 1:
            return [preferred[0]]
        return [preferred[round(i * (len(preferred) - 1) / (n_windows - 1))]
                for i in range(n_windows)]
    candidates = [w for w in range(min_window, upper + 1) if w % 5]
    return sorted(preferred + random.Random(seed).sample(candidates, n_windows - len(preferred)))


def format_scan_result(result, length, windows, thresholds):
    """Serialize native SCAN scores and apply inclusive voting thresholds."""
    if result is None:
        scores, votes = {}, {}
        per_window = {str(window): [] for window in windows}
        status = "constant statistic; no change points"
    else:
        scores = {int(k): float(v) for k, v in result.scores.items()}
        votes = {int(k): int(v) for k, v in result.votes.items()}
        per_window = {str(w): [int(cp) for cp in r.change_points]
                      for w, r in result.window_results.items()}
        status = "ok"
    detections = {}
    for threshold in thresholds:
        points = sorted(cp for cp, score in scores.items() if score >= threshold)
        if any(cp < 0 or cp >= length for cp in points):
            raise ValueError("SCAN returned an out-of-range split index")
        detections[str(threshold)] = {"statistic_indices": points}
    return dict(T=length, windows=windows, status=status, scores=scores,
                votes=votes, per_window=per_window, detections=detections)


def simple_name(value):
    if not isinstance(value, str) or not value or value in ('.', '..') or any(c in value for c in '/\\:'):
        raise ValueError('Job IDs must be simple directory names')
    return value


def load_config(path):
    """Load schema-version-1 jobs, merge CPD overrides, and resolve local paths."""
    path = Path(path).resolve()
    c = json.loads(path.read_text(encoding='utf-8'))
    if c.get('schema_version') != 1 or not c.get('jobs'):
        raise ValueError('schema_version=1 and nonempty jobs required')
    def resolve(value):
        return str((path.parent / Path(value).expanduser()).resolve())
    c['output_dir'] = resolve(c['output_dir'])
    for key, value in c.get('runtime', {}).items():
        if value:
            c['runtime'][key] = resolve(value)
    ids = []
    for job in c['jobs']:
        ids.append(simple_name(job['id']))
        if job['modality'] not in ('time_series', 'vision', 'embeddings'):
            raise ValueError('Unknown modality')
        job['input']['path'] = resolve(job['input']['path'])
        if job['modality'] == 'time_series':
            embedding = job['embedding']
            for key in ('context_length', 'stride'):
                value = embedding[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f'{key} must be a positive integer')
            if job['model']['batch_size'] < 1:
                raise ValueError('batch_size must be positive')
        for key in ('path', 'repository', 'checkpoint', 'timesfm1_source'):
            if job.get('model', {}).get(key):
                job['model'][key] = resolve(job['model'][key])
        settings = copy.deepcopy(c['cpd'])
        for key, value in job.get('cpd', {}).items():
            if isinstance(value, dict) and isinstance(settings.get(key), dict):
                settings[key].update(value)
            else:
                settings[key] = value
        job['cpd'] = settings
        tv, scan = settings['tv'], settings['scan']
        if not np.isfinite(tv['weight']) or tv['weight'] < 0 or tv['max_iterations'] < 1 or tv['tolerance'] <= 0:
            raise ValueError('Invalid TV settings')
        votes = scan['vote_thresholds']
        if not votes or any(not 0 <= v <= 1 for v in votes) or len(set(votes)) != len(votes):
            raise ValueError('Invalid or duplicate voting thresholds')
        if scan['n_boot'] < 1 or not 0 < scan['alpha'] < 1:
            raise ValueError('Invalid SCAN bootstrap count or alpha')
        source, output = Path(job['input']['path']), Path(c['output_dir'])
        if not source.exists():
            raise FileNotFoundError(source)
        if output == source or source in output.parents or output in source.parents:
            raise ValueError('Inputs and output must be separate paths')
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate job IDs')
    return c


def json_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


