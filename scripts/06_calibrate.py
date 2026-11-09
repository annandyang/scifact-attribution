"""Calibrate C5 from cached C4 predictions and keep the dev evaluation half held out."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scifact_attr import ABSTAIN, LABELS
from scifact_attr.config import REPO_ROOT, load_base
from scifact_attr.gate.conformal import (
    calibrate, coverage_and_error, gate_prediction, gold_score, label_scores,
    partition_claim_ids,
)
from scifact_attr.gate.risk_coverage import risk_coverage_curve, summarise_gate
from scifact_attr.utils.io import load_jsonl_as_list, write_jsonl


def load_c4(path):
    if not path.exists():
        raise SystemExit(f"Missing {path.name}. Run condition c4 on this split first.")
    rows = load_jsonl_as_list(path)
    if not rows or any(r.get("condition") != "c4" or r.get("status") == "dry_run"
                       or r.get("mode") == "smoke" for r in rows):
        raise SystemExit(f"{path.name} must contain real C4 predictions.")
    keys = {(str(r["claim_id"]), int(r["seed"])) for r in rows}
    if len(keys) != len(rows):
        raise SystemExit(f"Duplicate claim/seed pairs in {path.name}.")
    return rows


def gate_fields(row, alpha, threshold):
    predicted, label_set = gate_prediction(row, threshold)
    return {
        "pred_label": predicted,
        "prediction_set": label_set,
        "gate_threshold": threshold,
        "gate_alpha": alpha,
        "gate_decision": "ABSTAIN" if predicted == ABSTAIN else "answer",
    }


def c5_row(row, alpha, threshold, partition, options=None):
    base_label = row["pred_label"]
    scores = label_scores(row)
    score = scores.get(base_label) if any(scores.values()) else None
    result = {
        **row,
        "condition": "c5",
        "base_pred_label": base_label,
        "conformal_score": score,
        "gate_partition": partition,
        **gate_fields(row, alpha, threshold),
    }
    if options:
        result["gate_options"] = {f"{float(value):.2f}": gate_fields(row, float(value), limit)
                                  for value, limit in options.items()}
    return result


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_seed(seed, calibration, evaluation, alphas):
    scores = [gold_score(row) for row in calibration]
    gold = [row["gold_label"] for row in evaluation]
    outcomes = []
    for alpha in alphas:
        info = calibrate(scores, alpha)
        results = [gate_prediction(row, info["threshold"]) for row in evaluation]
        labels = [label for label, _ in results]
        answer_stats = coverage_and_error(labels, gold, [label != ABSTAIN for label in labels])
        set_coverage = sum(label in label_set for label, (_, label_set) in zip(gold, results)) / len(gold)
        outcomes.append({
            "seed": seed,
            **info,
            "n_evaluation": len(gold),
            "nominal_set_coverage": 1.0 - alpha,
            "empirical_set_coverage": set_coverage,
            "coverage": answer_stats["coverage"],
            "selective_error": answer_stats["selective_error"],
        })

    pred = [row["pred_label"] for row in evaluation]
    confidence = []
    for row, label in zip(evaluation, pred):
        scores = label_scores(row)
        if label in LABELS and row.get("status") == "ok" and any(scores.values()):
            confidence.append(1.0 - scores[label])
        else:
            confidence.append(0.0)
    correct = [p == g for p, g in zip(pred, gold)]
    coverage, risk = risk_coverage_curve(confidence, correct)
    points = [{"coverage": c, "risk": r, "seed": seed} for c, r in zip(coverage, risk)]
    fixed = coverage_and_error(pred, gold, [conf >= 0.70 for conf in confidence])
    summary = summarise_gate(confidence, correct)
    return outcomes, points, fixed, summary


def split_dev(rows, path):
    official = load_jsonl_as_list(path)
    gold = {str(row["claim_id"]): row["label"] for row in official}
    calibration_ids, evaluation_ids = partition_claim_ids(gold)
    if any(gold.get(str(row["claim_id"])) != row.get("gold_label")
           or row.get("gold_label") not in LABELS for row in rows):
        raise SystemExit("C4 dev rows must match the processed SciFact dev gold labels.")
    return calibration_ids, evaluation_ids


def number(value):
    return f"{value:.3f}" if value == value else "n/a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    args = parser.parse_args()
    cfg = load_base()
    paths = {name: REPO_ROOT / path for name, path in cfg["paths"].items()}
    predictions = paths["predictions"]
    source = predictions / f"c4_{args.split}.jsonl"
    dev_source = predictions / "c4_dev.jsonl"
    output = predictions / f"c5_{args.split}.jsonl"
    rows = load_c4(source)
    gate_path = predictions.parent / "gate" / "conformal_thresholds.json"
    alphas = [float(a) for a in cfg["gate"]["alphas"]]
    alpha = float(cfg["gate"].get("apply_alpha", alphas[0]))
    if alpha not in alphas:
        raise SystemExit("gate.apply_alpha must be one of gate.alphas.")
    seeds = sorted({int(row["seed"]) for row in rows})
    fingerprint = sha256(dev_source.read_bytes()).hexdigest() if dev_source.exists() else None

    if args.split == "test":
        if not gate_path.exists():
            raise SystemExit("Run scripts/06_calibrate.py --split dev first.")
        frozen = json.loads(gate_path.read_text(encoding="utf-8"))
        if not fingerprint or frozen.get("c4_dev_sha256") != fingerprint:
            raise SystemExit("C4 dev predictions changed. Recalibrate on dev first.")
        thresholds = frozen["thresholds_by_seed"]
        if any(str(seed) not in thresholds or str(alpha) not in thresholds[str(seed)]
               for seed in seeds):
            raise SystemExit("Test predictions need matching calibrated dev seeds and alpha.")
        output_rows = [
            c5_row(row, alpha, thresholds[str(row["seed"])][str(alpha)], "prediction")
            for row in rows
        ]
        for row in output_rows:
            row["gold_label"] = None
        write_jsonl(output, output_rows)
        print(f"Wrote {len(rows)} unlabeled predictions to {output}")
        return 0

    cal_ids, eval_ids = split_dev(rows, paths["processed"] / "claims_dev.jsonl")
    thresholds, all_outcomes, all_points, c5_rows = {}, [], [], []
    lines = [
        "# E5 conformal gate on held-out SciFact dev",
        "",
        "Calibration and evaluation use disjoint halves of the official dev split.",
        "The label set has marginal coverage at least 1 minus alpha under exchangeability.",
        "The answer rate and error among answered claims are empirical.",
        "",
        "| seed | rule | calibration n | evaluation n | nominal set coverage | "
        "empirical set coverage | answer rate | selective error |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for seed in seeds:
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        cal = [row for row in seed_rows if str(row["claim_id"]) in cal_ids]
        eval_rows = [row for row in seed_rows if str(row["claim_id"]) in eval_ids]
        if {str(row["claim_id"]) for row in cal} != cal_ids or {
            str(row["claim_id"]) for row in eval_rows
        } != eval_ids:
            raise SystemExit(f"Seed {seed} needs every C4 dev prediction before calibration.")
        if not any(any(label_scores(row).values()) for row in cal):
            raise SystemExit(f"Seed {seed} has no calibration probabilities.")
        outcomes, points, fixed, summary = evaluate_seed(seed, cal, eval_rows, alphas)
        thresholds[str(seed)] = {str(result["alpha"]): result["threshold"] for result in outcomes}
        threshold = thresholds[str(seed)][str(alpha)]
        c5_rows.extend(
            c5_row(row, alpha, threshold,
                   "calibration" if str(row["claim_id"]) in cal_ids else "evaluation",
                   options=thresholds[str(seed)])
            for row in seed_rows
        )
        all_outcomes.extend(outcomes)
        all_points.extend(points)
        for result in outcomes:
            lines.append(
                f"| {seed} | conformal, alpha={result['alpha']:.2f} | {len(cal)} | {len(eval_rows)} | "
                f"{number(result['nominal_set_coverage'])} | {number(result['empirical_set_coverage'])} | "
                f"{number(result['coverage'])} | {number(result['selective_error'])} |"
            )
        lines.append(
            f"| {seed} | fixed confidence >= 0.70 | - | {len(eval_rows)} | - | - | "
            f"{number(fixed['coverage'])} | {number(fixed['selective_error'])} |"
        )
        lines.append(
            f"| {seed} | confidence sweep AURC {number(summary['aurc'])} | - | {len(eval_rows)} | "
            "- | - | - | - |"
        )

    frozen = {
        "method": "split_conformal_singleton",
        "calibration_split": "dev",
        "evaluation_split": "dev",
        "c4_dev_sha256": fingerprint,
        "calibration_claim_ids": sorted(cal_ids),
        "evaluation_claim_ids": sorted(eval_ids),
        "thresholds_by_seed": thresholds,
    }
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    write_jsonl(output, c5_rows)
    write_csv(paths["tables"] / "e5_conformal.csv", all_outcomes)
    write_csv(paths["figures"] / "risk_coverage_c5.csv", all_points)
    report = "\n".join(lines) + "\n"
    (paths["tables"] / "e5_gate_compare.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {len(c5_rows)} C5 rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
