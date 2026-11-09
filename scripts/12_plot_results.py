import csv
import json
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


def selection():
    files = [root / "tables" / f"e2_{name}.json" for name in ("selection_c2", "empty_rule")]
    if not all(path.is_file() for path in files):
        return
    study, rule = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    results = study["evaluation"]
    reports = [results["c1_reference"], results["original"]]
    labels = ["C1: abstract excerpts", "C2: original selection"]
    colors = ["#405463", "#C26638"]
    if study["chosen_policy"] != "original":
        policy = study["chosen_policy"]
        reports.append(results[policy])
        labels.append("C2: " + {"no_threshold": "remove score cutoff", "wide8": "wider selection",
                                "neighbors8": "include neighboring sentences"}[policy])
        colors.append("#3872B1")
    reports.append(rule["evaluation"]["empty_rule"])
    labels.append("C2: empty context → NEI")
    colors.append("#168274")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True, layout="constrained")
    for ax, metric, title in zip(axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro F1"), strict=True):
        values = [report[metric] for report in reports]
        bars = ax.barh(labels, values, color=colors, height=.52)
        ax.bar_label(bars, labels=[f"{value:.1%}" if metric == "accuracy" else f"{value:.3f}"
                                   for value in values], padding=5)
        ax.set(xlim=(0, 1), title=title)
        ax.grid(axis="x", alpha=.22)
    axes[0].invert_yaxis()
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("Sentence selection and the empty-context rule · 150 held-out SciFact claims")
    fig.savefig(root / "figures" / "e2_selection_followup.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    retrieval()
    gate()
    selection()
