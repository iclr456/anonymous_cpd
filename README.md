# Time-series change-point detection using foundation models

This repository contains code for time-series change-point detection using foundation models. The offline detector turns windowed embeddings into scalar statistics and applies a SCAN window ensemble. Vision foundation models for generic event boundary detection are also listed below; vision adapters are not included in the current package.

The proposed framework consists of the following

## Quick start

First, download the model checkpoints you want to use from Hugging Face. To test a single model, download only that model and keep only its entry in the `models` list in your configs file. Set its `path` to the downloaded model directory.

**Time-series foundation models**

| Model family | Variant / parameters | Hugging Face |
|---|---|---|
| Chronos-T5 | Tiny, 8M | [amazon/chronos-t5-tiny](https://huggingface.co/amazon/chronos-t5-tiny) |
| Chronos-T5 | Large, 710M | [amazon/chronos-t5-large](https://huggingface.co/amazon/chronos-t5-large) |
| Chronos-Bolt | Tiny, 9M | [amazon/chronos-bolt-tiny](https://huggingface.co/amazon/chronos-bolt-tiny) |
| Chronos-Bolt | Base, 205M | [amazon/chronos-bolt-base](https://huggingface.co/amazon/chronos-bolt-base) |
| Chronos-2 | 120M | [amazon/chronos-2](https://huggingface.co/amazon/chronos-2) |
| Moirai | Small | [Salesforce/moirai-1.0-R-small](https://huggingface.co/Salesforce/moirai-1.0-R-small) |
| Moirai | Large | [Salesforce/moirai-1.0-R-large](https://huggingface.co/Salesforce/moirai-1.0-R-large) |
| MOMENT | Small | [AutonLab/MOMENT-1-small](https://huggingface.co/AutonLab/MOMENT-1-small) |
| MOMENT | Large | [AutonLab/MOMENT-1-large](https://huggingface.co/AutonLab/MOMENT-1-large) |
| TimesFM | 1.0, 200M | [google/timesfm-1.0-200m-pytorch](https://huggingface.co/google/timesfm-1.0-200m-pytorch) |
| TimesFM | 2.5, 200M | [google/timesfm-2.5-200m-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch) |
| Moirai-MoE | Small | [Salesforce/moirai-moe-1.0-R-small](https://huggingface.co/Salesforce/moirai-moe-1.0-R-small) |


**Vision foundation models**

| Model | Notation | Hugging Face |
|---|---|---|
| DINO ViT-S/16 | DINO-S/16 | [facebook/dino-vits16](https://huggingface.co/facebook/dino-vits16) |
| DINO ViT-B/16 | DINO-B/16 | [facebook/dino-vitb16](https://huggingface.co/facebook/dino-vitb16) |
| DINOv2 ViT-S/14 | DINOv2-S/14 | [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small) |
| DINOv2 ViT-B/14 | DINOv2-B/14 | [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base) |

These vision checkpoints are references for event boundary detection; the CLI below currently supports time-series inputs and adapters only.

**Install and run**

Next, install the dependencies using the `requirements.txt` file. Run the following command from the repository root:

```sh
python -m pip install -r requirements.txt
```

Then install the package and run the small synthetic example:

```sh
python -m pip install -e ".[scan]"
python -m tsfm_cpd --config configs/example.json
python experiments/evaluate.py --predictions results/example/synthetic/detections.json --labels examples/labels.json --tolerance 24 --output results/example/metrics.csv
```

The example uses a deterministic window-summary adapter. It checks the complete processing path without downloading weights; its results are not foundation-model benchmark results. The synthetic signal contains 600 observations with changes at zero-based indices 200 and 400. SCAN uses only 19 bootstrap samples in this example to keep it quick.

For the Python API, see `examples/minimal.py`. `detect_changes(series, adapter, config)` returns a dictionary keyed by statistic, including detections at each configured voting threshold. It does not read labels or tune against them.

## Foundation models

`configs/foundation.json` lists the 13 model variants supported by the existing adapters. Put weights in the configured model folders, or change their paths. Paths are relative to the JSON configuration file. Install the dependencies required by the selected backend; the files in `environments/` record the previous environments, including a CUDA-specific PyTorch build. These are reference snapshots, not portable lockfiles or a promise of a successful fresh installation on every platform.

Backends are Chronos, MOMENT, Moirai, Moirai-MoE, TimesFM 1.0, and TimesFM 2.5. Unrelated model libraries are imported only when needed. TimesFM 1.0 decoder code and its license are retained under `vendor/`.

To use separate foundation and SCAN environments, install this package in both environments and set `runtime.foundation_python` and `runtime.scan_python` in a local copy of the configuration. Empty strings mean the current interpreter. Alternatively run stages explicitly:

```sh
python -m tsfm_cpd --config configs/foundation.json --stage extract
python -m tsfm_cpd --config configs/foundation.json --stage scan
```

## Method and conventions

1. Load finite numeric values shaped `(time, channels)`. CSV column selection must exclude timestamps and labels. There is no implicit imputation or input normalization.
2. Extract complete context windows, embed them, and pool channels according to the model adapter.
3. Fit centered Ledoit-Wolf covariance geometry on the raw embeddings.
4. Apply TV denoising separately to every embedding coordinate.
5. Compute consecutive scaled dot product, cosine similarity, covariance-adjusted scaled dot product, and covariance-adjusted cosine similarity. Optionally trim the first and last statistic.
6. Standardize each statistic and run SCAN over the selected detector windows. Constant statistics produce no detections; undefined statistics raise errors.
7. Apply inclusive voting thresholds to native SCAN scores, preserving its grouping.
8. Map indices to the final observation of the right-hand embedding window.

Embedding context length and detector window sizes are different settings. The default context length is 128. Seven detector windows are selected deterministically subject to SCAN bounds; short inputs that cannot support these windows fail explicitly. There is no voting across models or statistics.

Indices are zero-based. Endpoint alignment does not correct context-window detection delay. Covariance fitting, TV smoothing, and statistic standardization use the full series: this implementation is offline, not causal.

## Repository layout

- `src/tsfm_cpd/`: numerical routines, adapters, configuration, artifacts, and CLI.
- `configs/`: portable example and foundation-model configurations.
- `experiments/evaluate.py`: evaluation of saved predictions; labels never enter detection.
- `paper/plot_metrics.py`: plots saved measured scores without changing them.
- `examples/`: a synthetic signal, its labels, and the Python API example.
- `tests/`: numerical reference, alignment, window-selection, SCAN, and matching tests.

Each model produces `embeddings.npz`, `tv_embeddings.npz`, `covariance.npz`, `statistics.npz`, and `detections.json`. Stage manifests record resolved settings, the input SHA-256 checksum, Python, and relevant package versions. Output files at the chosen location are overwritten on rerun; use a different output directory to retain a run. Record the exact downloaded weight revision separately in the model configuration when preparing an experiment.

## Evaluation

The provided evaluator uses one-to-one maximum-cardinality matching within an inclusive tolerance. Multiple predictions cannot receive credit for the same label. Duplicate indices are collapsed. Both-empty inputs score 1; an empty prediction or label set paired with a nonempty set scores 0. This is an explicit new evaluation utility; equivalence to every historical benchmark evaluator has not been established.

Thresholds are reported separately, with no automatic selection of the best test score. Label-based oracle searches and dataset-specific paper experiments have not been migrated. Add their frozen configurations and clearly documented selection protocols before claiming paper reproduction.

Optional figure generation:

```sh
python -m pip install -e ".[plots]"
python paper/plot_metrics.py results/example/metrics.csv results/example/metrics.png
```

## Tests

In an environment with numerical and SCAN dependencies:

```sh
python -m unittest discover -s tests -v
```

With separate environments, run `test_numerics.py` in the foundation environment and `test_detection.py` in the SCAN environment, using unittest's `-p` option. `test_evaluation.py` requires only the standard library.

The numerical fixture compares raw embeddings, TV output, four statistics, and index alignment with outputs generated by the previous implementation on a fixed synthetic input. It does not validate every foundation model. A full 13-model run and a clean dependency installation have not been performed for this release.

## Anonymous submission

This folder is a local release candidate, not a published repository. Start a fresh repository from its source files; do not copy the parent repository history, environments, model weights, caches, local settings, or generated results. Runtime manifests contain resolved local paths and must be reviewed before sharing. Keep third-party notices intact. Choose an appropriate license for your own code before publication; this export does not assign one on your behalf.

The reconstructed and deliberately modified plots from the working project are not included. Paper evidence must come from measured experiment outputs.
