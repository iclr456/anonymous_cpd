"""Evaluate saved detections using one-to-one matching within a tolerance.

Example: python -m detection.evaluate --predictions outputs/example/series_summary/detections.json --labels examples/unified_labels.json --tolerance 24 --output results/metrics.csv
"""
import argparse
import csv
import json
from pathlib import Path


def match_changes(predicted, actual, tolerance):
    if tolerance < 0:
        raise ValueError("Tolerance must be nonnegative")
    predicted, actual = sorted(set(predicted)), sorted(set(actual))
    i = j = matches = 0
    while i < len(predicted) and j < len(actual):
        if predicted[i] < actual[j] - tolerance:
            i += 1
        elif actual[j] < predicted[i] - tolerance:
            j += 1
        else:
            matches += 1
            i += 1
            j += 1
    precision = matches / len(predicted) if predicted else float(not actual)
    recall = matches / len(actual) if actual else float(not predicted)
    f1 = 2 * matches / (len(predicted) + len(actual)) if predicted or actual else 1.0
    return dict(matches=matches, n_predicted=len(predicted), n_actual=len(actual),
                precision=precision, recall=recall, f1=f1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--tolerance", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.predictions.read_text())
    actual = json.loads(args.labels.read_text())["change_points"]
    rows = []
    for statistic, result in report["statistics"].items():
        for threshold, detection in result["detections"].items():
            rows.append(dict(model=report["model"], statistic=statistic,
                             threshold=threshold, tolerance=args.tolerance,
                             **match_changes(detection["source_indices"], actual, args.tolerance)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
