"""Change-point detection from windowed time-series embeddings."""


def compute_statistics(*args, **kwargs):
    from .pipeline import compute_statistics as compute
    return compute(*args, **kwargs)


def detect_changes(*args, **kwargs):
    from .pipeline import detect_changes as detect
    return detect(*args, **kwargs)
