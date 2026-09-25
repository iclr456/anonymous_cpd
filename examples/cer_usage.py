"""End-to-end Change Encoding Ratio (CER) example.

1. Build a target series with known change points and a change-free
   background series from the same domain.
2. Load a foundation-model adapter (Chronos-T5-Tiny by default).
3. Slide context windows over both series and embed every window.
4. Pass the latent representations, their source coordinates, and the
   change points to ``CER``.

Run from the package root:

    python examples/cer_usage.py
    python examples/cer_usage.py --backend moment --model AutonLab/MOMENT-1-small
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from change_encoding_ratio import CER
from detection.core import run_embedding_pipeline
from detection.foundation_adapters import load_foundation_adapter


def make_series(seed: int = 0):
    """Return (target, change_points, background) as (time, channels) arrays."""
    rng = np.random.default_rng(seed)
    t = np.arange(1200)
    seasonal = np.sin(2 * np.pi * t / 50)

    # Segment 1: baseline. Segment 2: mean shift. Segment 3: variance + period change.
    target = np.concatenate([
        seasonal[:400] + 0.3 * rng.standard_normal(400),
        seasonal[400:800] + 2.0 + 0.3 * rng.standard_normal(400),
        np.sin(2 * np.pi * t[800:] / 20) + 1.0 * rng.standard_normal(400),
    ])
    change_points = [400, 800]  # index of the first post-change sample

    # Background: same generator as segment 1, but no changes at all.
    background = np.sin(2 * np.pi * np.arange(1000) / 50) + 0.3 * rng.standard_normal(1000)
    return target[:, None], change_points, background[:, None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="chronos")
    parser.add_argument("--model", default="amazon/chronos-t5-tiny",
                        help="Local model directory or Hugging Face id.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--n-side", type=int, default=128)
    args = parser.parse_args()

    target, change_points, background = make_series()

    # 1. Adapter: wraps the model and maps (batch, channels, time) -> (batch, features).
    adapter = load_foundation_adapter(args.backend, args.model, args.device)

    # 2. Project both series into the model's latent space, one vector per window.
    target_emb = run_embedding_pipeline(target, adapter, window_size=args.context_length, stride=1)
    background_emb = run_embedding_pipeline(background, adapter, window_size=args.context_length, stride=1)
    print(f"target latents:     {target_emb.embeddings.shape}")
    print(f"background latents: {background_emb.embeddings.shape}")

    # 3. CER. Window ends from the pipeline are exclusive; CER expects inclusive ends,
    #    so it can pick only windows that lie entirely before / after each change point.
    cer = CER(
        latent_representations=target_emb.embeddings,
        change_points=change_points,
        background_latent_representations=background_emb.embeddings,
        starts=target_emb.starts,
        ends=target_emb.ends - 1,
        background_starts=background_emb.starts,
        background_ends=background_emb.ends - 1,
        n_side=args.n_side,
    )
    if cer.error:
        raise SystemExit(f"CER input error: {cer.error}")

    rows = cer.evaluate()
    print(f"\nsigma={rows[0].get('sigma', float('nan')):.4g}  "
          f"background median CE-SNR={rows[0].get('background_median', float('nan')):.4g}  "
          f"({rows[0].get('n_reference', 0)} reference boundaries)\n")
    print(f"{'change_point':>12} {'MMD^2':>10} {'CE-SNR':>10} {'CER':>10} {'log2 CER':>9}  preserved")
    for row in rows:
        if row["status"] != "ok":
            print(f"{row['change_point']!s:>12}  {row['status']}")
            continue
        print(f"{row['change_point']:>12} {row['mmd2']:>10.4g} {row['ce_snr']:>10.4g} "
              f"{row['cer']:>10.3f} {row['log2_cer']:>9.2f}  {row['preserved']}")

    print(f"\nmean CER:               {cer.score():.3f}")
    print(f"change-preserving rate: {cer.change_preserving_rate():.2f}")


if __name__ == "__main__":
    main()
