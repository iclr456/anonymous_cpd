# Change-point detection using foundation models

One offline pipeline for time-series and video embeddings: context windows,
foundation-model embeddings, covariance fitting, TV denoising, four scalar
statistics, and SCAN change-point detection.

## Quick start

Install the dependencies using `requirements.txt` for your foundation environment,
then install this package. The requirements file is an environment snapshot with
a CUDA-specific PyTorch build; install only the backends needed for your models.

```sh
python -m pip install -r requirements.txt
python -m pip install -e ".[scan]"
python -m detection --config configs/example.json
python -m detection.evaluate --predictions results/example_jobs/synthetic/detections.json --labels examples/labels.json --tolerance 24 --output results/example_jobs/metrics.csv
```

The example uses deterministic window summaries and requires no model weights.
It checks the processing path, not foundation-model accuracy. The 600-observation
signal changes at zero-based indices 200 and 400; SCAN uses 19 bootstrap samples.
Use a fresh output directory when changing settings or inputs.

For an example with both time-series data and saved vision-shaped embeddings:

```sh
python examples/make_inputs.py
python -m detection --config configs/unified_example.json
```

Download only the models you need and keep their entries in the configuration's
`jobs` list. Set each job's `model.path` to its downloaded model directory.

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


TimesFM 1.0 requires the PyTorch `torch_model.ckpt` checkpoint linked above.
`configs/foundation.json` contains 13 time-series jobs, including Moirai Base and
Moirai-MoE Base. It does not include Moirai Large by default.

Vision jobs require local DINO/DINOv2 source checkouts and plain state-dictionary
checkpoints, or a local ResNet-50 checkpoint. The current vision loader does not
load Hugging Face model directories directly. Install `.[scan,vision]` for video
and image dependencies. No downloads are initiated by the runner.

## USC-HAD dataset example

[examples/usc_had](examples/usc_had/README.md) includes the six processed USC-HAD
accelerometer series, separate labels, and train/test configurations. It preserves
the main pipeline's seeded, whole-series 30/70 split: TS1 and TS5 for training/tuning;
TS2, TS3, TS4, and TS6 held out. Rounding gives 2/6 and 4/6 series (33.3%/66.7%).

```sh
python -m detection --config examples/usc_had/train.json
python -m detection --config examples/usc_had/test.json
```

The example uses summary embeddings without downloaded weights. See its README
for foundation-model configurations, regeneration, and evaluation commands.

## One runner and configuration format

Every configuration uses `schema_version: 1`, shared `cpd` settings, and a `jobs`
list. Each job has a unique `id`, `modality`, and `input`, plus model and embedding
settings when extracting new embeddings. Supported modalities are `time_series`,
`vision`, and `embeddings`. Per-job `cpd` overrides merge one section at a time.
All paths are resolved relative to the configuration file.

- `configs/example.json`: summary embeddings of the synthetic time series.
- `configs/unified_example.json`: time-series and saved vision embeddings.
- `configs/foundation.json`: 13 local time-series foundation-model jobs.
- `configs/frozen.json`: template for local time-series and vision experiments.

Run any of these with the same command:

```sh
python -m detection --config configs/foundation.json
```

The installed `cpd` command invokes this same runner. The old `models`-based
configuration, `detection.cli` command, and array-based package-level API have
been replaced. Existing custom configurations must use the `jobs` format shown
in the supplied examples. For the Python API, see `examples/minimal.py`:
`load_config(path)` followed by `run_stage(config, "extract")` and
`run_stage(config, "scan")`.

For separate environments, install this package in both and set
`runtime.foundation_python` and `runtime.scan_python`. Empty strings use the
current interpreter. Or run each stage explicitly in its corresponding environment:

```sh
python -m detection --config configs/foundation.json --stage extract
python -m detection --config configs/foundation.json --stage scan
```

## Inputs and preprocessing

```text
src/
  time_series_preprocessing.py
  video_preprocessing.py
  detection/
```

Time-series inputs are finite numeric CSV, NPY, or NPZ arrays shaped
`(time, channels)`. Select columns to exclude timestamps and labels. There is no
implicit imputation. Complete context windows are unpadded, and incomplete tails
are omitted. Their anchors are the final observations in the original series.

Video inputs are videos or naturally ordered frame directories. OpenCV decodes
videos to 224x224 RGB frames. The image encoders use ImageNet normalization.
Video context windows are centered, with repeated boundary frames; the default
is a five-frame mean of embeddings. Anchors are zero-based frame positions.

Saved embedding inputs are NPZ archives containing finite `(time, features)`
`embeddings` and strictly increasing integer `source_indices`, one per embedding.
These embeddings must already be pooled. Existing GEBD archives can use their
`frame_numbers - 1` as source anchors.

The preprocessing scripts can also export windows independently:

```sh
python src/time_series_preprocessing.py --input examples/synthetic.csv --skiprows 1 --context-length 128 --output results/time_windows.npz
python src/video_preprocessing.py --input path/to/video.mp4 --context-length 5 --output results/video_frames
```

Time-series exports contain windows, starts, and exclusive ends. Video exports
contain decoded PNGs, timestamps, and `context_windows.npz` with frame-index
windows and center anchors. These are inspection/export outputs, not embedding
inputs. The runner uses the same preprocessing functions internally.

## Method and outputs

The shared numerical function in `detection/pipeline.py` fits centered Ledoit-Wolf
geometry on raw embeddings, denoises each coordinate with TV, and computes four
consecutive statistics: scaled dot product, cosine similarity, and their
covariance-adjusted counterparts. Optional edge trimming removes the first and
last statistics and their anchors together.

`detection/detection.py` standardizes each projected series and calls SCAN with
windows generated by `utils.generate_window_sizes()`. Voting thresholds are
inclusive and preserve native SCAN grouping. Constant series produce no changes;
undefined statistics fail explicitly. There is no cross-model/statistic voting.

Embedding context length and SCAN detector window sizes are separate settings.
All indices are zero-based. A comparison is anchored at its right-hand embedding:
the context-window endpoint for time series or center frame for vision.
This mapping does not correct context-window delay. Full-series covariance
fitting, TV smoothing, and standardization make this an offline method.

`output_dir/<job-id>/` contains `embeddings.npz`, `tv_embeddings.npz`,
`covariance.npz`, `statistics.npz`, and `detections.json`. The output root records
resolved settings, input content hashes, and stage environment versions. Changed
inputs or settings require a fresh output directory. Partial artifacts are not
completion markers, and concurrent writers are unsupported. Record exact model
weight revisions separately.

## Evaluation and tests

Evaluation is separate from detection and does not use labels for parameter
selection. It uses one-to-one maximum-cardinality matching within an inclusive
absolute tolerance. Duplicates are collapsed; both-empty inputs score 1 and
one-empty inputs score 0. Thresholds are reported separately. This evaluator
does not replace vision best-annotator/relative-tolerance benchmark protocols.

```sh
python -m unittest discover -s tests -v
```

With separate environments, run numerical and preprocessing tests in the
foundation environment and `test_detection.py` in the SCAN environment using
unittest's `-p` option. Set `CPD_SCAN_PYTHON` to the SCAN interpreter when running
`test_unified.py` in the foundation environment. Evaluation tests require only
the standard library. The numerical fixture checks embeddings, TV output, four
statistics, and index alignment against saved reference outputs.

Optional plotting:

```sh
python -m pip install -e ".[plots]"
python paper/plot_metrics.py results/example_jobs/metrics.csv results/example_jobs/metrics.png
```

Actual foundation-model inference and video decoding need local weights and the
corresponding dependencies; synthetic tests do not establish model accuracy.
Vision now shares the time-series TV solver and SCAN semantics, which differ from
historical vision experiments. Revalidate frozen parameters before reproduction
claims. No label-based oracle selection is performed.

## Provenance and anonymous submission

TimesFM 1.0 source and its license remain in `models/timesfm1/`. Preserve third-party
notices. For an installed wheel, retain the TimesFM source directory separately and set
`model.timesfm1_source` accordingly. No license for user-authored code is assigned
by this export. Review runtime manifests for local paths before sharing. Exclude
weights, environments, caches, generated results, and parent repository history
from an anonymous release. Paper evidence must come from measured outputs.
