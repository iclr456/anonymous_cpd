"""Plot a metrics CSV produced by experiments/evaluate.py without altering scores."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = sorted({(row["model"], row["statistic"], row["tolerance"]) for row in rows})
    for model, statistic, tolerance in groups:
        selected = sorted((row for row in rows if (row["model"], row["statistic"], row["tolerance"]) == (model, statistic, tolerance)), key=lambda row: float(row["threshold"]))
        ax.plot([float(row["threshold"]) for row in selected],
                [float(row["f1"]) for row in selected], marker="o",
                label=f"{model}, {statistic}, tolerance {tolerance}")
    ax.set(xlabel="Voting threshold", ylabel="F1", ylim=(0, 1.02))
    ax.legend(fontsize=6)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
