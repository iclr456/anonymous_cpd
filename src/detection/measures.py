"""The four consecutive-embedding statistics used by the change-point pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


METRIC_NAMES = (
    "scaled_dot_product",
    "cosine_similarity",
    "covariance_adjusted_scaled_dot_product",
    "covariance_adjusted_cosine_similarity",
)


def fit_ledoit_wolf(embeddings: np.ndarray) -> dict[str, np.ndarray | float | str]:
    """Fit Ledoit-Wolf once on raw embeddings and retain covariance geometry."""
    H = _validate_embeddings(embeddings, minimum_rows=3)
    estimator = LedoitWolf(store_precision=True, assume_centered=False).fit(H)
    return {
        "precision": estimator.precision_,
        "covariance": estimator.covariance_,
        "location": estimator.location_,
        "shrinkage": float(estimator.shrinkage_),
        "estimator": "LedoitWolf",
    }


def consecutive_statistics(
    embeddings: np.ndarray,
    *,
    precision: np.ndarray,
    location: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the change-point pipeline's four statistics for consecutive windows."""
    H = _validate_embeddings(embeddings, minimum_rows=2)
    P = np.asarray(precision, dtype=np.float64)
    mu = np.asarray(location, dtype=np.float64)
    d = H.shape[1]
    if P.shape != (d, d):
        raise ValueError(f"precision must have shape {(d, d)}, got {P.shape}")
    if mu.shape != (d,):
        raise ValueError(f"location must have shape {(d,)}, got {mu.shape}")

    left, right = H[:-1], H[1:]
    products = np.einsum("ij,ij->i", left, right)
    ordinary_denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = np.divide(
        products,
        ordinary_denominator,
        out=np.full(len(products), np.nan),
        where=ordinary_denominator > 0,
    )

    centered_left, centered_right = left - mu, right - mu
    adjusted_cross = np.einsum(
        "ij,jk,ik->i", centered_left, P, centered_right, optimize=True
    )
    left_norm = np.sqrt(np.maximum(np.einsum(
        "ij,jk,ik->i", centered_left, P, centered_left, optimize=True
    ), 0.0))
    right_norm = np.sqrt(np.maximum(np.einsum(
        "ij,jk,ik->i", centered_right, P, centered_right, optimize=True
    ), 0.0))
    adjusted_denominator = left_norm * right_norm

    return {
        "scaled_dot_product": (products / np.sqrt(d)).astype(np.float32),
        "cosine_similarity": cosine.astype(np.float32),
        "covariance_adjusted_scaled_dot_product": (
            adjusted_cross / np.sqrt(d)
        ).astype(np.float32),
        "covariance_adjusted_cosine_similarity": np.divide(
            adjusted_cross,
            adjusted_denominator,
            out=np.full(len(adjusted_cross), np.nan),
            where=adjusted_denominator > 0,
        ).astype(np.float32),
    }


def _validate_embeddings(embeddings: np.ndarray, minimum_rows: int) -> np.ndarray:
    H = np.asarray(embeddings, dtype=np.float64)
    if H.ndim != 2 or H.shape[0] < minimum_rows or H.shape[1] < 1:
        raise ValueError(f"embeddings must have shape (T, d), T >= {minimum_rows}")
    if not np.isfinite(H).all():
        raise ValueError("embeddings must be finite")
    return H
