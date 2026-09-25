"""Create complete sliding context windows from numeric time-series inputs."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
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


def main():
    """Load numeric input and save context windows and endpoint indices."""
    from detection.utils import load_series
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--context-length', type=int, default=128)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--delimiter', default=',')
    parser.add_argument('--skiprows', type=int, default=0)
    parser.add_argument('--columns', type=int, nargs='+')
    parser.add_argument('--npz-key', default='values')
    args = parser.parse_args()
    values = load_series(dict(path=args.input, delimiter=args.delimiter,
        skiprows=args.skiprows, columns=args.columns, npz_key=args.npz_key))
    windows = create_context_windows(values, args.context_length, stride=args.stride)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, windows=windows.values, starts=windows.starts,
        ends=windows.ends, source_indices=windows.ends - 1,
        source_length=windows.source_length, window_size=windows.window_size, stride=windows.stride)


if __name__ == '__main__':
    main()
