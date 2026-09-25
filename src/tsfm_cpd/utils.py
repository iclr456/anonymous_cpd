"""Configuration, input loading, artifact persistence, summary embeddings, and SCAN utilities."""
import hashlib
import importlib.metadata
import json
import math
import random
from pathlib import Path
import platform
import sys
import numpy as np
from .core import save_embedding_result, validate_series


def save_statistics(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_embedding_result(result.embeddings, directory / "embeddings.npz")
    np.savez_compressed(directory / "tv_embeddings.npz", embeddings=result.denoised)
    np.savez_compressed(directory / "statistics.npz", **result.statistics,
                        source_indices=result.source_indices)
    np.savez_compressed(directory / "covariance.npz", **result.geometry)


def save_manifest(config, stage):
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "scan-py",
                 "chronos-forecasting", "momentfm", "uni2ts", "timesfm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    checksum = hashlib.sha256()
    with Path(config["input"]["path"]).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    manifest = dict(python=sys.version, platform=platform.system(), packages=versions,
                    input_sha256=checksum.hexdigest(), config=config)
    (output / f"manifest_{stage}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


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


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))

    def resolve(value):
        candidate = Path(value).expanduser()
        return str((path.parent / candidate).resolve())

    config["input"]["path"] = resolve(config["input"]["path"])
    config["output_dir"] = resolve(config["output_dir"])
    if config.get("timesfm1_source"):
        config["timesfm1_source"] = resolve(config["timesfm1_source"])
    keys = [model["key"] for model in config["models"]]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Provide at least one model, with unique model keys")
    for model in config["models"]:
        key = model["key"]
        if not key or key in {".", ".."} or any(c in key for c in '/\\:'):
            raise ValueError("Model keys must be simple directory names")
        if "path" in model:
            model["path"] = resolve(model["path"])
    for key, value in config.get("runtime", {}).items():
        if value:
            config["runtime"][key] = resolve(value)
    if config["embedding"]["context_length"] < 1 or config["embedding"]["stride"] < 1:
        raise ValueError("Context length and stride must be positive")
    if not config["scan"]["vote_thresholds"] or any(
        not 0 <= threshold <= 1 for threshold in config["scan"]["vote_thresholds"]
    ):
        raise ValueError("Voting thresholds must lie in [0, 1]")
    return config
