"""Shared numerical processing for time-series and vision embeddings."""
import numpy as np
from .tv import tv_denoise_embeddings
from .utils import generate_window_sizes


def compute_statistics(values, indices, config):
    """Compute shared CPD statistics from embeddings and their source anchors.

    Returns denoised embeddings, covariance geometry, named statistics, and
    the aligned anchors of the right-hand embeddings after optional trimming.
    This numerical function is independent of modality, files, and SCAN.
    """
    from detection.measures import fit_ledoit_wolf, consecutive_statistics

    values, indices = np.asarray(values), np.asarray(indices)
    if values.ndim != 2 or len(values) < 3 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError('Expected at least three finite embedding vectors')
    if indices.shape != (len(values),) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('One integer source index per embedding is required')
    if np.any(indices < 0) or np.any(np.diff(indices) <= 0):
        raise ValueError('Source indices must be nonnegative and strictly increasing')
    tv = config['tv']
    selection = slice(1, -1) if tv['drop_edge_statistics'] else slice(None)
    mapping = indices[1:][selection]
    generate_window_sizes(len(mapping), **config['windows'])
    geometry = fit_ledoit_wolf(values)
    denoised = tv_denoise_embeddings(values, tv['weight'],
        max_iterations=tv['max_iterations'], tolerance=tv['tolerance'])
    statistics = consecutive_statistics(denoised, precision=geometry['precision'], location=geometry['location'])
    statistics = {name: v[selection] for name, v in statistics.items()}
    if any(not np.isfinite(v).all() for v in statistics.values()):
        raise ValueError('Undefined statistic (e.g. zero-norm embedding); no implicit imputation')
    return denoised, geometry, statistics, mapping


