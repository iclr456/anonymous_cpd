"""SCAN scoring, thresholding, and original-series index alignment."""
import json
from pathlib import Path

import numpy as np
import scan

from .utils import format_scan_result, generate_window_sizes


def detect_changes(projected_series, config):
    """Detect changes in one projected scalar series using SCAN.

    Args:
        projected_series: Finite one-dimensional values, with one value per
            comparison of consecutive context-window embeddings.
        config: Settings containing ``windows`` for generate_window_sizes()
            and ``scan`` for SCAN options and inclusive voting thresholds.

    Returns:
        A report with detector window sizes, scores, votes, and detections at
        each threshold. ``statistic_indices`` are zero-based positions in the
        projected series. Context-window endpoint mapping is applied separately
        by detect_statistics(). Constant series produce empty detections.

    The series is standardized before SCAN. Detector window sizes are measured
    in projected-series observations, independently of the embedding context
    window length.
    """
    values = np.asarray(projected_series, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Projected series must be a finite one-dimensional array")
    windows = generate_window_sizes(len(values), **config["windows"])
    settings = dict(config["scan"])
    thresholds = settings.pop("vote_thresholds")
    if not thresholds or any(not 0 <= threshold <= 1 for threshold in thresholds):
        raise ValueError("Voting thresholds must lie in [0, 1]")

    result = None
    std = values.std()
    if std > 1e-12:
        result = scan.scan_cpd(
            (values - values.mean()) / std,
            window_sizes=windows, vote_threshold=0.0, return_all=True, **settings,
        )
    return format_scan_result(result, len(values), windows, thresholds)


def detect_statistics(statistics, source_indices, config):
    """Detect each statistic and map changes to context-window endpoints.

    Args:
        statistics: Mapping from statistic names to projected scalar series.
        source_indices: Context-window endpoint indices in the original input
            series, aligned one-to-one with each statistic. Each index is the
            zero-based final observation of the right-hand context window in
            the compared pair, rather than the ordinal number of that window.
            Any edge trimming must already be reflected in this mapping.
        config: Detector window settings and SCAN options passed to
            detect_changes().

    Returns:
        A report keyed by statistic name. Each threshold includes both
        ``statistic_indices`` and their mapped context-window endpoints under
        ``source_indices``. Mapping preserves endpoint alignment; it does not
        compensate for detection delay caused by the context window.
    """
    indices = np.asarray(source_indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Source indices must be a one-dimensional integer array")
    if np.any(indices < 0) or np.any(np.diff(indices) <= 0):
        raise ValueError("Source indices must be nonnegative and strictly increasing")

    report = {}
    for name, values in statistics.items():
        if np.asarray(values).shape != indices.shape:
            raise ValueError(f"{name}: statistic must be aligned with source indices")
        report[name] = detect_changes(values, config)
        for detection in report[name]["detections"].values():
            detection["source_indices"] = [
                int(indices[cp]) for cp in detection["statistic_indices"]
            ]
    return report


def detect(config):
    """Run detection on saved statistics for every configured model.

    Args:
        config: Resolved configuration containing ``output_dir``, ``models``,
            detector window settings, and SCAN options.

    Loads statistics.npz from each model's output directory, including its
    context-window endpoint mapping stored as ``source_indices``. Calls
    detect_statistics() and writes the report to detections.json in that
    directory, replacing any existing report. Returns None.
    """
    for model in config["models"]:
        directory = Path(config["output_dir"]) / model["key"]
        with np.load(directory / "statistics.npz", allow_pickle=False) as archive:
            statistics = {key: archive[key] for key in archive.files if key != "source_indices"}
            results = detect_statistics(statistics, archive["source_indices"], config)
        report = {
            "model": model["key"],
            "index_convention": "zero-based source row; right embedding endpoint",
            "statistics": results,
        }
        (directory / "detections.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"SCAN complete: {model['key']}", flush=True)
