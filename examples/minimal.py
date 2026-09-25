"""Run the Python API in an environment with both numerical and SCAN dependencies."""
from pathlib import Path
from tsfm_cpd import detect_changes
from tsfm_cpd.utils import SummaryAdapter, load_config, load_series

config = load_config(Path(__file__).resolve().parents[1] / "configs" / "example.json")
result = detect_changes(load_series(config["input"]), SummaryAdapter(), config)
for statistic, report in result.items():
    print(statistic, report["detections"]["0.5"]["source_indices"])
