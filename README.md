# Change-point detection using foundation models

One offline pipeline for time-series and video embeddings: context windows,
foundation-model embeddings, covariance fitting, TV denoising, four scalar
statistics, and SCAN change-point detection.

<table>
  <tr>
    <td width="50%">
      <a href="resources/timeseries_pair%20%288%29.pdf">
        <img src="resources/timeseries_pair.png" alt="Time-series change-point detection figure" width="100%">
      </a>
    </td>
    <td width="50%">
      <a href="resources/video_pair%20%283%29.pdf">
        <img src="resources/video_pair.png" alt="Video event-boundary detection figure" width="100%">
      </a>
    </td>
  </tr>
</table>

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

1. **Update the configuration file.**

   Specify the foundation models you want to run and set the parameters for the
   SCAN univariate change-point detector for video and time-series data
   accordingly. Parameters can be selected by evaluating a parameter grid on a
   validation set before running the pipeline with the selected values.

   - `input.path`: path to your dataset.
   - `model.path`: local directory containing the downloaded model.
   - `embedding.context_length` and `embedding.stride`: context-window settings.
   - `embedding.device`: `cpu` or `cuda`.
   - `cpd.tv.weight`: TV smoothing strength.
   - `cpd.windows`: detector window settings.
   - `cpd.scan`: significance level, bootstrap count, and voting thresholds.
   - `output_dir`: folder for the results.

2. **Run the pipeline.**

   From the repository root:

   ```sh
   python -m detection --config configs/foundation.json
   ```

   The pipeline creates context windows, extracts embeddings, computes the
   projected statistics, and detects changes using SCAN within a unified framework.

3. **View the results.**

   Results are saved in the `output_dir` folder, including embeddings, denoised
   embeddings, the covariance matrix, projected univariate series in
   `statistics.npz`, and detected change points in `detections.json`.
