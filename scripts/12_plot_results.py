import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

root = Path(__file__).resolve().parents[1] / "outputs"


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def retrieval():
    data = rows(root / "tables" / "e1_retrieval.csv")
    names = ["BM25", "Dense BGE", "Hybrid RRF", "Hybrid + rerank"]
    colors = ["#405463", "#168274", "#3872B1", "#C26638"]
    fig, (hit, latency) = plt.subplots(1, 2, figsize=(10.4, 3.9), layout="constrained")
    for name, color, row in zip(names, colors, data, strict=True):
        hit.plot([4, 8, 16], [float(row[f"recall@{k}"]) for k in (4, 8, 16)],
                 marker="o", color=color, label=name)
    hit.set(xlabel="Retrieved abstracts (k)", ylabel="Claims with a gold hit",
            xticks=[4, 8, 16], ylim=(0.84, 1))
    hit.yaxis.set_major_formatter(PercentFormatter(1))
    hit.legend(loc="lower right", frameon=False, fontsize=8)
    hit.grid(axis="y", alpha=.22)
    for index, (color, row) in enumerate(zip(colors, data, strict=True)):
        p50 = float(row["median_ms@8"])
        p95 = float(row["p95_ms@8"])
        latency.plot([p50, p95], [index, index], color=color, linewidth=2)
        latency.scatter(p50, index, color=color, marker="o")
        latency.scatter(p95, index, color=color, marker="|", s=85)
    latency.set(xscale="log", xlabel="Query latency at k = 8 (ms, log scale)",
                yticks=range(len(names)), yticklabels=names,
                title="Dot = median, tick = 95th percentile")
    latency.invert_yaxis()
    latency.grid(axis="x", alpha=.22)
    fig.suptitle("SciFact development retrieval · 188 evidence-bearing claims", fontsize=14)
    fig.savefig(root / "figures" / "e1_retrieval.png", dpi=200)
    plt.close(fig)


def gate():
    points_path = root / "figures" / "risk_coverage_c5.csv"
    if not points_path.exists():
        return
    points = [r for r in rows(points_path) if r["seed"] == "0"]
    decisions = [r for r in rows(root / "tables" / "e5_conformal.csv") if r["seed"] == "0"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    ax.plot([float(r["coverage"]) for r in points], [float(r["risk"]) for r in points],
            color="#3872B1", label="C4 confidence sweep")
    for r in decisions:
        coverage, risk = float(r["coverage"]), float(r["selective_error"])
        if coverage > 0 and math.isfinite(risk):
            ax.scatter(coverage, risk, color="#C26638", clip_on=False, zorder=3,
                       label=f"C5 α={float(r['alpha']):.2f}")
        elif coverage == 0:
            ax.scatter([], [], color="#C26638", marker="x",
                       label=f"C5 α={float(r['alpha']):.2f} · no answers")
    top = min(1, max(0.5, max(float(r["risk"]) for r in points) + 0.05))
    ax.set(xlim=(0, 1), ylim=(0, top), xlabel="Fraction answered",
           ylabel="Error among answers", title="Held-out SciFact dev · seed 0")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False)
    ax.grid(alpha=.22)
    fig.savefig(root / "figures" / "e5_risk_coverage.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    retrieval()
    gate()
