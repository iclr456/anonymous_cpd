"""Core, model-independent TSFM embedding pipeline.

Input series use the repository convention ``(time, channels)``. Model batches
use the common TSFM convention ``(batch, channels, time)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class WindowedSeries:
    """Context windows and their half-open locations in the source series."""

    values: np.ndarray  # (n_windows, n_channels, window_size)
    starts: np.ndarray  # inclusive
    ends: np.ndarray  # exclusive
    source_length: int
    window_size: int
    stride: int


@dataclass(frozen=True)
class EmbeddingResult:
    """Embeddings aligned with context-window locations."""

    embeddings: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    source_length: int
    window_size: int
    stride: int


@runtime_checkable
class EmbeddingAdapter(Protocol):
    """Minimal interface implemented by each TSFM-specific adapter."""

    def embed(self, batch: np.ndarray) -> np.ndarray:
        """Embed a float32 batch shaped (batch, channels, time)."""


def validate_series(series: np.ndarray) -> np.ndarray:
    """Return a finite float32 ``(time, channels)`` array."""
    values = np.asarray(series, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"Expected (time, channels), got shape {values.shape}")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("The series must contain at least one row and channel")
    if not np.isfinite(values).all():
        raise ValueError("The series contains NaN or infinite values; impute first")
    return np.ascontiguousarray(values)


def create_context_windows(
    series: np.ndarray,
    window_size: int,
    *,
    stride: int = 1,
) -> WindowedSeries:
    """Create full overlapping windows without padding.

    A source ``series[start:end]`` becomes one model input shaped
    ``(channels, window_size)``. The final partial window is intentionally
    omitted so every TSFM receives the same amount of context.
    """
    values = validate_series(series)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if window_size > len(values):
        raise ValueError(
            f"window_size={window_size} exceeds series length={len(values)}"
        )

    starts = np.arange(0, len(values) - window_size + 1, stride, dtype=np.int64)
    ends = starts + window_size
    windows = np.stack([values[start:end].T for start, end in zip(starts, ends)])
    return WindowedSeries(
        values=np.ascontiguousarray(windows, dtype=np.float32),
        starts=starts,
        ends=ends,
        source_length=len(values),
        window_size=window_size,
        stride=stride,
    )


def extract_embeddings(
    windows: WindowedSeries,
    adapter: EmbeddingAdapter,
    *,
    batch_size: int = 32,
) -> EmbeddingResult:
    """Embed all windows in deterministic batches and concatenate outputs."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    chunks: list[np.ndarray] = []
    expected_tail: tuple[int, ...] | None = None
    for offset in range(0, len(windows.values), batch_size):
        batch = windows.values[offset : offset + batch_size]
        output = _to_numpy(adapter.embed(batch))
        if output.ndim < 2:
            raise ValueError(
                f"Adapter output must be at least 2D (batch, features...), got {output.shape}"
            )
        if output.shape[0] != batch.shape[0]:
            raise ValueError(
                f"Adapter returned {output.shape[0]} embeddings for batch size {batch.shape[0]}"
            )
        if expected_tail is None:
            expected_tail = output.shape[1:]
        elif output.shape[1:] != expected_tail:
            raise ValueError(
                f"Inconsistent embedding shapes: expected (*, {expected_tail}), got {output.shape}"
            )
        chunks.append(np.asarray(output, dtype=np.float32))

    return EmbeddingResult(
        embeddings=np.concatenate(chunks, axis=0),
        starts=windows.starts.copy(),
        ends=windows.ends.copy(),
        source_length=windows.source_length,
        window_size=windows.window_size,
        stride=windows.stride,
    )


def save_embedding_result(result: EmbeddingResult, output_path: str | Path) -> Path:
    """Save embeddings and alignment metadata as one compressed NPZ file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=result.embeddings,
        starts=result.starts,
        ends=result.ends,
        source_length=np.int64(result.source_length),
        window_size=np.int64(result.window_size),
        stride=np.int64(result.stride),
    )
    return path


def run_embedding_pipeline(
    series: np.ndarray,
    adapter: EmbeddingAdapter,
    *,
    window_size: int,
    stride: int = 1,
    batch_size: int = 32,
    output_path: str | Path | None = None,
) -> EmbeddingResult:
    """Window a multivariate series, embed it, and optionally save the result."""
    windows = create_context_windows(series, window_size, stride=stride)
    result = extract_embeddings(windows, adapter, batch_size=batch_size)
    if output_path is not None:
        save_embedding_result(result, output_path)
    return result


def _to_numpy(value) -> np.ndarray:
    """Convert NumPy arrays and CPU/GPU tensor-like outputs to NumPy."""
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)
