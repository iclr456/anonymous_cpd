"""Temporal total-variation denoising for embedding trajectories."""

from __future__ import annotations

import numpy as np


def tv_denoise_embeddings(
    embeddings: np.ndarray,
    weight: float,
    *,
    max_iterations: int = 200,
    tolerance: float = 1e-5,
) -> np.ndarray:
    """Independently TV-denoise each embedding coordinate over window time.

    Solves ``0.5 ||X-H||_F^2 + weight * sum_t,j |X[t+1,j]-X[t,j]|``
    with a vectorized one-dimensional Chambolle projection algorithm.
    """
    H = np.asarray(embeddings, dtype=np.float64)
    if H.ndim != 2 or H.shape[0] < 2:
        raise ValueError("embeddings must have shape (num_windows, d), num_windows >= 2")
    if not np.isfinite(H).all():
        raise ValueError("embeddings must be finite")
    if weight < 0:
        raise ValueError("weight must be non-negative")
    if weight == 0:
        return H.astype(np.float32, copy=True)
    if max_iterations <= 0 or tolerance <= 0:
        raise ValueError("max_iterations and tolerance must be positive")

    dual = np.zeros_like(H)
    step = 0.25
    for iteration in range(max_iterations):
        divergence = -dual.copy()
        divergence[1:] += dual[:-1]
        estimate = H + divergence

        gradient = np.zeros_like(H)
        gradient[:-1] = np.diff(estimate, axis=0)
        updated = (dual - step * gradient) / (1.0 + (step / weight) * np.abs(gradient))

        relative_change = np.linalg.norm(updated - dual) / max(
            np.linalg.norm(dual), 1.0
        )
        dual = updated
        if iteration >= 5 and relative_change < tolerance:
            break

    divergence = -dual.copy()
    divergence[1:] += dual[:-1]
    return (H + divergence).astype(np.float32)
