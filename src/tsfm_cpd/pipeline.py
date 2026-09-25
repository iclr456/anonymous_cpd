from dataclasses import dataclass
import numpy as np

from .core import EmbeddingResult, run_embedding_pipeline
from .measures import consecutive_statistics, fit_ledoit_wolf
from .tv import tv_denoise_embeddings
from .utils import generate_window_sizes


@dataclass
class StatisticResult:
    embeddings: EmbeddingResult
    denoised: np.ndarray
    statistics: dict[str, np.ndarray]
    source_indices: np.ndarray
    geometry: dict


def compute_statistics(series, adapter, config, *, batch_size=32):
    """Fit geometry before TV and retain right-window endpoint alignment."""
    embedding = config["embedding"]
    tv = config["tv"]
    count = (len(series) - embedding["context_length"]) // embedding["stride"] + 1
    length = count - 1 - (2 if tv["drop_edge_statistics"] else 0)
    generate_window_sizes(length, **config["windows"])
    result = run_embedding_pipeline(
        series, adapter, window_size=embedding["context_length"],
        stride=embedding["stride"], batch_size=batch_size,
    )
    geometry = fit_ledoit_wolf(result.embeddings)
    denoised = tv_denoise_embeddings(
        result.embeddings, tv["weight"], max_iterations=tv["max_iterations"],
        tolerance=tv["tolerance"],
    )
    statistics = consecutive_statistics(
        denoised, precision=geometry["precision"], location=geometry["location"],
    )
    selection = slice(1, -1) if tv["drop_edge_statistics"] else slice(None)
    statistics = {name: values[selection] for name, values in statistics.items()}
    indices = (result.ends[1:] - 1)[selection]
    return StatisticResult(result, denoised, statistics, indices, geometry)


def detect_changes(series, adapter, config, *, batch_size=32):
    """Return SCAN results for all four statistics; does not accept labels."""
    from .detection import detect_statistics

    result = compute_statistics(series, adapter, config, batch_size=batch_size)
    return detect_statistics(result.statistics, result.source_indices, config)
