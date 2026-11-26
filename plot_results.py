"""Draw the synthetic reliability and verifier-cost results from aggregates."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    colors = ("#006b67", "#a45a22")
    figure, (quality, cost) = plt.subplots(1, 2, figsize=(10.4, 4.6), layout="constrained")
    labels = ("Single", "Four roles", "Quote check", "+ A.X", "+ Phi")
    policies = ("single", "vote", "citation", "same_model", "other_model")
    for index, order in enumerate(("original", "reversed")):
        aggregate = load(args.results / f"summary-{order}.json")
        failures = [aggregate["policies"][policy]["false_pass"] for policy in policies]
        positions = [position + (index - 0.5) * 0.35 for position in range(len(policies))]
        bars = quality.bar(positions, failures, width=0.35, color=colors[index], label=order.title())
        quality.bar_label(bars, padding=2, fontsize=8)
    quality.set_title("False passes at 25% human review", loc="left", fontweight="bold")
    quality.set_ylabel("False passes per 120 misleading titles")
    quality.set_xticks(range(len(policies)), labels)
    quality.set_ylim(0, 24)
    quality.legend(frameon=False)

    for index, name in enumerate(("same", "other")):
        aggregate = load(args.results / f"runtime-{name}.json")
        original = [row for row in aggregate["results"] if row["scheduling"] == "original"]
        cost.plot([row["batch_ms_p95"] for row in original], [row["title_per_sec"] for row in original],
                  marker="o", color=colors[index], label=("A.X", "Phi")[index])
    cost.set_title("Verifier throughput and batch latency", loc="left", fontweight="bold")
    cost.set_xlabel("p95 batch latency (ms)")
    cost.set_ylabel("Titles per second")
    cost.legend(frameon=False)
    for axis in (quality, cost):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#e3e7ea", linewidth=0.7)
        axis.set_axisbelow(True)
    figure.savefig(args.out, dpi=180, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
