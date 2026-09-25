from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
from scipy.spatial.distance import cdist, pdist


class _CERInputError(Exception):
    """Internal validation failure caught by the public CER methods."""


@dataclass
class CER:
    """
    This class takes latent representations from any foundation model along with the change-points 
    and returns the Change Encoding Ratio of the foundation model. 
    """
    """
    Args:
        latent_representations: Target latent_representations shaped (time, features). A one-dimensional
            input is treated as a single feature.
        change_points: Integer boundary positions in the target series.
            A boundary marks the start of the post-change segment.
        background_latent_representations: Reference latent_representations with the same feature
            dimension as the target, used to estimate the background baseline.
        n_side: Number of clean latent_representations on each side of a boundary.
            Must be at least 2. Defaults to 128.
        reference_step: Spacing between candidate background boundaries.
            Defaults to 20.
        bandwidth_sample_size: Maximum number of pooled target and background
            latent_representations sampled to estimate the kernel bandwidth.
            Defaults to 3000.
        seed: Random seed for bandwidth sampling. Defaults to 0.
        epsilon: Positive numerical stabilizer used in ratios and logarithms.
            Defaults to 1e-12.
        sigma: Positive Gaussian-kernel bandwidth. If None, estimated as the
            median positive pairwise distance among sampled latent_representations.
        starts: Inclusive source start index for each target embedding.
        ends: Inclusive source end index for each target embedding.
            Provide both starts and ends, or neither. If omitted, each
            embedding is treated as a point at its zero-based row index.
        background_starts: Inclusive source starts for background latent_representations.
        background_ends: Inclusive source ends for background latent_representations.
            Provide both background coordinate arrays, or neither.

    """

    latent_representations: np.ndarray = field(repr=False)
    change_points: Sequence[int]
    background_latent_representations: np.ndarray = field(repr=False)
    n_side: int = 128
    reference_step: int = 20
    bandwidth_sample_size: int = 3000
    seed: int = 0
    epsilon: float = 1e-12
    sigma: float | None = None
    starts: np.ndarray | None = field(default=None, repr=False)
    ends: np.ndarray | None = field(default=None, repr=False)
    background_starts: np.ndarray | None = field(default=None, repr=False)
    background_ends: np.ndarray | None = field(default=None, repr=False)

    error: str | None = field(default=None, init=False)

    def __post_init__(self):
        try:
            self._initialize()
        except (_CERInputError, ValueError, TypeError, IndexError, OverflowError, FloatingPointError) as exc:
            self.error = str(exc)

    def _initialize(self):
        """Normalize valid inputs; public initialization catches failures."""
        for name in ("latent_representations", "background_latent_representations"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.ndim == 1:
                values = values[:, None]
            if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all():
                raise _CERInputError(f"{name} must be a nonempty finite (time, features) array.")
            setattr(self, name, values.copy())
        if self.latent_representations.shape[1] != self.background_latent_representations.shape[1]:
            raise _CERInputError("Target and background must have the same feature dimension.")
        for name, minimum in (("n_side", 2), ("reference_step", 1), ("bandwidth_sample_size", 2)):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise _CERInputError(f"{name} must be an integer >= {minimum}.")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise _CERInputError("epsilon must be finite and positive.")
        if self.sigma is not None and (not np.isfinite(self.sigma) or self.sigma <= 0):
            raise _CERInputError("sigma must be finite and positive.")
        self.starts, self.ends = self._coordinates(self.latent_representations, self.starts, self.ends)
        self.background_starts, self.background_ends = self._coordinates(
            self.background_latent_representations, self.background_starts, self.background_ends
        )
        cps = np.asarray(self.change_points)
        if cps.ndim != 1 or (cps.size and cps.dtype.kind not in "iu"):
            raise _CERInputError("change_points must be a sequence of integer indices.")
        if np.any(cps <= 0) or np.any(cps > self.ends[-1]):
            raise _CERInputError("Change points must be positive and within the target series.")
        self.change_points = np.unique(cps).astype(int)

    @staticmethod
    def _coordinates(z, starts, ends):
        if starts is None and ends is None:
            return np.arange(len(z)), np.arange(len(z))
        if starts is None or ends is None:
            raise _CERInputError("Provide both starts and ends, or neither.")
        starts, ends = np.asarray(starts), np.asarray(ends)
        for values in (starts, ends):
            if values.shape != (len(z),) or values.dtype.kind not in "iu":
                raise _CERInputError("Coordinates must contain one integer per embedding.")
            if np.any(values < 0) or np.any(values[1:] < values[:-1]):
                raise _CERInputError("Coordinates must be nonnegative and sorted.")
        if np.any(starts > ends):
            raise _CERInputError("Each start must be <= its inclusive end.")
        return starts.copy(), ends.copy()

    def _blocks(self, z, starts, ends, boundary):
        pre = np.flatnonzero(ends < boundary)
        post = np.flatnonzero(starts >= boundary)
        if min(len(pre), len(post)) < self.n_side:
            return None
        return z[pre[-self.n_side:]], z[post[:self.n_side]]

    def _snr(self, left, right, sigma):
        def kernel(x, y):
            return np.exp(-cdist(x, y, "sqeuclidean") / (2.0 * sigma**2))

        kxx, kyy, kxy = kernel(left, left), kernel(right, right), kernel(left, right)
        signal = max(float(kxx.mean() + kyy.mean() - 2 * kxy.mean()), 0.0)
        var_left = max(float(np.diag(kxx).mean() - kxx.mean()), 0.0)
        var_right = max(float(np.diag(kyy).mean() - kyy.mean()), 0.0)
        noise = 0.5 * (var_left + var_right)
        return dict(mmd2=signal, var_left=var_left, var_right=var_right,
                    within_variance=noise, ce_snr=signal / (noise + self.epsilon))

    def evaluate(self) -> list[dict]:
        """Return boundary results, or an explicit error row if evaluation fails."""
        try:
            if self.error is not None:
                raise _CERInputError(self.error)
            return self._evaluate()
        except (_CERInputError, ValueError, TypeError, IndexError, OverflowError, FloatingPointError) as exc:
            return [{"change_point": None, "status": "error", "error": str(exc),
                     "cer": float("nan")}]

    def _evaluate(self) -> list[dict]:
        """Compute one result per sorted, unique change point."""
        if not len(self.change_points):
            return []
        sigma = self.sigma
        if sigma is None:
            pool = np.vstack((self.latent_representations, self.background_latent_representations))
            rng = np.random.default_rng(self.seed)
            sample = pool[rng.choice(len(pool), min(self.bandwidth_sample_size, len(pool)), replace=False)]
            distances = pdist(sample, "euclidean")
            distances = distances[distances > 0]
            if not len(distances):
                raise _CERInputError("Cannot estimate bandwidth: sampled latent_representations are identical; supply sigma.")
            sigma = float(np.median(distances))

        reference = []
        for boundary in range(0, int(self.background_ends[-1]) + 2, self.reference_step):
            blocks = self._blocks(self.background_latent_representations, self.background_starts,
                                  self.background_ends, boundary)
            if blocks is not None:
                reference.append(self._snr(*blocks, sigma)["ce_snr"])
        if not reference:
            raise _CERInputError("No valid reference boundaries; provide more background data or reduce n_side/reference_step.")
        baseline = float(np.median(reference))

        results = []
        for cp in self.change_points:
            row = dict(change_point=int(cp), sigma=sigma, background_median=baseline,
                       n_reference=len(reference))
            blocks = self._blocks(self.latent_representations, self.starts, self.ends, cp)
            if blocks is None:
                row.update(status="skipped: insufficient clean latent_representations", cer=float("nan"))
            else:
                diagnostics = self._snr(*blocks, sigma)
                ratio = diagnostics["ce_snr"] / (baseline + self.epsilon)
                row.update(diagnostics, status="ok", cer=ratio,
                           log2_cer=float(np.log2(ratio + self.epsilon)),
                           preserved=bool(diagnostics["ce_snr"] > baseline))
            results.append(row)
        return results

    def score(self) -> float:
        """Mean CER of valid annotated boundaries, or NaN if none are valid."""
        scores = [row["cer"] for row in self.evaluate() if row["status"] == "ok"]
        return float(np.mean(scores)) if scores else float("nan")

    def change_preserving_rate(self) -> float:
        """Fraction of evaluable change points with CER > 1 (NaN if none are evaluable)."""
        flags = [row["preserved"] for row in self.evaluate() if row["status"] == "ok"]
        return float(np.mean(flags)) if flags else float("nan")
