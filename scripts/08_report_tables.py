"""Report the ladder on shared, held-out SciFact dev claims."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scifact_attr import ABSTAIN, LABELS
from scifact_attr.config import REPO_ROOT, load_base
from scifact_attr.eval.bootstrap import bootstrap_ci, mcnemar, paired_delta_ci
from scifact_attr.eval.metrics import expected_calibration_error, per_class_f1, summarise
from scifact_attr.gate.conformal import partition_claim_ids
from scifact_attr.reader.parse import citation_metrics
from scifact_attr.utils.io import get_logger, load_jsonl_as_list


LOG = get_logger("tables")
ORDER = ["c0", "c1", "c2", "c3", "c4", "c5", "c3_prime"]
NAMES = {
    "c0": "closed book",
    "c1": "+ retrieval",
    "c2": "+ evidence selection",
    "c3": "+ LoRA",
    "c4": "+ CoT",
    "c5": "+ conformal gate",
    "c3_prime": "adapter, cue removed",
}


def load_condition_rows(path, evaluation_ids):
    groups = {}
    for row in load_jsonl_as_list(path):
        claim_id = str(row["claim_id"])
        if claim_id not in evaluation_ids or row.get("gold_label") not in LABELS:
            continue
        if row.get("status") == "dry_run" or row.get("mode") == "smoke":
            continue
        if row.get("condition") == "c5" and row.get("gate_partition") != "evaluation":
            continue
        seed_rows = groups.setdefault(int(row["seed"]), {})
        if claim_id in seed_rows:
            raise SystemExit(f"Duplicate claim {claim_id} for seed {row['seed']} in {path.name}")
        seed_rows[claim_id] = row
    return groups


def evidence_metrics(condition, rows):
    annotated = [r for r in rows if r.get("gold_sentence_ids")]
    results = {"n_gold_evidence": len(annotated)}
    if not annotated:
        return results

    if condition != "c0":
        hits = 0
        for row in annotated:
            gold_docs = {sid.split("#", 1)[0] for sid in row["gold_sentence_ids"]}
            retrieved = {str(doc) for doc in row.get("retrieved_ids", [])[:8]}
            hits += bool(gold_docs & retrieved)
        results["retrieval_hit_at_8"] = hits / len(annotated)
    if condition in ("c2", "c3", "c4", "c5", "c3_prime"):
        recalls = [
            citation_metrics(row.get("selected_ids", [])[:8], row["gold_sentence_ids"])["recall"]
            for row in annotated
        ]
        results["evidence_recall_at_8"] = sum(recalls) / len(recalls)
    if condition in ("c4", "c5"):
        scores = [
            citation_metrics(row.get("cited_ids", []), row["gold_sentence_ids"])
            for row in annotated
        ]
        for metric in ("precision", "recall", "f1"):
            results[f"citation_{metric}"] = sum(score[metric] for score in scores) / len(scores)
    return results


def summarise_run(condition, seed, rows, resamples):
    gold = [r["gold_label"] for r in rows]
    pred = [r["pred_label"] for r in rows]
    metrics = summarise(gold, pred)
    class_f1 = per_class_f1(gold, pred)
    latencies = [r["latency_ms"] for r in rows if isinstance(r.get("latency_ms"), (int, float))]
    peaks = [r["peak_mem_mb"] for r in rows if isinstance(r.get("peak_mem_mb"), (int, float))]
    low, high = bootstrap_ci([p == g for p, g in zip(pred, gold)], n_resamples=resamples)
    confidence, correct = [], []
    for row in rows:
        probability = row.get("pred_prob")
        if (row.get("status") == "ok" and row["pred_label"] in LABELS
                and isinstance(probability, (float, int)) and isfinite(probability)):
            confidence.append(probability)
            correct.append(row["gold_label"] == row["pred_label"])
    modes = sorted({str(row.get("retrieval_mode") or "unspecified") for row in rows})
    metrics.update({
        "condition": condition,
        "seed": seed,
        "name": NAMES.get(condition, condition),
        "accuracy_ci_low": low,
        "accuracy_ci_high": high,
        "ece": expected_calibration_error(confidence, correct),
        "n_ece": len(confidence),
        "retrieval_mode": ", ".join(modes),
        "n_invalid": sum(p not in LABELS and p != ABSTAIN for p in pred),
        "n_unparsed_or_flag": sum(r.get("status") not in ("ok", None) for r in rows),
        "support_f1": class_f1["SUPPORT"],
        "refute_f1": class_f1["REFUTE"],
        "median_latency_ms": median(latencies) if latencies else None,
        "peak_mem_mb": max(peaks) if peaks else None,
    })
    metrics.update(evidence_metrics(condition, rows))
    return metrics


def compare_runs(loaded, shared, resamples):
    rows = []
    pairs = list(zip(ORDER[:4], ORDER[1:5])) + [
        ("c3", name) for name in ("c3_prime", "rank8", "rank32", "bf16_r16")
    ]
    for before, after in pairs:
        if before not in loaded or after not in loaded:
            continue
        before_runs, after_runs = loaded[before], loaded[after]
        for seed, later in sorted(after_runs.items()):
            before_seed = seed if seed in before_runs else None
            if before_seed is None and before in ORDER[:3] and len(before_runs) == 1:
                before_seed = next(iter(before_runs))
            if before_seed is None:
                continue
            earlier = before_runs[before_seed]
            gold = [later[cid]["gold_label"] for cid in shared]
            first = [earlier[cid]["pred_label"] for cid in shared]
            second = [later[cid]["pred_label"] for cid in shared]
            delta = paired_delta_ci(first, second, gold, n_resamples=resamples)
            test = mcnemar(first, second, gold)
            verdict = "inconclusive"
            if delta["ci_low"] > 0:
                verdict = "improvement"
            elif delta["ci_high"] < 0:
                verdict = "regression"
            rows.append({
                "comparison": f"{after} - {before}",
                "baseline_seed": before_seed,
                "seed": seed,
                **delta,
                "delta_accuracy": delta["delta"],
                "p_value": test["p_value"],
                "discordant": test["n_discordant"],
                "verdict": verdict,
            })
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return f"{value:.3f}" if value is not None and isfinite(value) else "n/a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--conditions", default=",".join(ORDER))
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--tag", choices=("ablation",), help="keep separate tables for E7")
    args = parser.parse_args()
    requested = {name.strip() for name in args.conditions.split(",")}
    order = ("rank8", "c3", "rank32", "bf16_r16") if args.tag else ORDER
    if requested - set(order) or args.resamples < 1:
        parser.error("Use known conditions and a positive resample count.")
    cfg = load_base()
    predictions = REPO_ROOT / cfg["paths"]["predictions"]
    tables = REPO_ROOT / cfg["paths"]["tables"]
    tables.mkdir(parents=True, exist_ok=True)

    if args.split == "test":
        print("Official SciFact test is unlabeled. Use the per-claim test predictions and report metrics on dev.")
        return 0

    official = load_jsonl_as_list(REPO_ROOT / cfg["paths"]["processed"] / "claims_dev.jsonl")
    gold_by_id = {str(row["claim_id"]): row for row in official}
    _, evaluation_ids = partition_claim_ids(gold_by_id)
    loaded = {}
    for condition in (name for name in order if name in requested):
        path = predictions / f"{condition}_dev.jsonl"
        if not path.exists():
            LOG.warning("Skipping %s, no cached predictions", condition)
            continue
        if condition == "c5":
            frozen_path = predictions.parent / "gate" / "conformal_thresholds.json"
            c4_path = predictions / "c4_dev.jsonl"
            if not (frozen_path.exists() and c4_path.exists()):
                LOG.warning("Skipping c5, frozen calibration or C4 source is missing")
                continue
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            if frozen.get("c4_dev_sha256") != sha256(c4_path.read_bytes()).hexdigest():
                LOG.warning("Skipping c5, C4 changed since calibration. Rerun script 06.")
                continue
        groups = load_condition_rows(path, evaluation_ids)
        if not groups:
            LOG.warning("Skipping %s, no real held-out dev predictions", condition)
            continue
        loaded[condition] = groups

    all_runs = [run for group in loaded.values() for run in group.values()]
    if not all_runs:
        raise SystemExit("No real held-out dev predictions. Run scripts/04_run_condition.py first.")
    shared = sorted(set.intersection(*(set(run) for run in all_runs)))
    if not shared:
        raise SystemExit("The requested condition/seed runs have no shared held-out claims.")
    for run in all_runs:
        for cid in shared:
            gold_row = gold_by_id[cid]
            if run[cid]["gold_label"] != gold_row["label"]:
                raise SystemExit(f"Gold label mismatch for claim {cid}. Regenerate cached predictions.")
            run[cid]["gold_sentence_ids"] = gold_row.get("gold_sentence_ids", [])

    rows = [
        summarise_run(condition, seed, [run[cid] for cid in shared], args.resamples)
        for condition, runs in loaded.items()
        for seed, run in sorted(runs.items())
    ]
    if args.tag == "ablation":
        if set(loaded) != requested:
            raise SystemExit("Run every requested adapter condition before generating the ablation table.")
        if len({frozenset(runs) for runs in loaded.values()}) != 1:
            raise SystemExit("Adapter comparisons require the same seeds for every rank.")
        adapter_names = {"rank8": "lora_r16_r8_qlora", "c3": "lora_r16",
                         "rank32": "lora_r16_r32_qlora", "bf16_r16": "lora_r16_r16_bf16_lora"}
        for row in rows:
            adapter = adapter_names[row["condition"]]
            path = predictions.parent / "adapters" / f"{adapter}_s{row['seed']}" / "train_meta.json"
            meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            row["training_seconds"] = meta.get("metrics", {}).get("train_runtime")
            row["training_peak_gib"] = meta.get("peak_mem_gb")
    comparisons = compare_runs(loaded, shared, args.resamples)
    suffix = f"_{args.tag}" if args.tag else ""
    out = tables / f"table_ladder_dev{suffix}.csv"
    write_csv(out, rows)
    if comparisons:
        write_csv(tables / f"table_deltas_dev{suffix}.csv", comparisons)
    else:
        (tables / f"table_deltas_dev{suffix}.csv").unlink(missing_ok=True)

    lines = [
        "# Ladder results on held-out SciFact dev",
        "",
        f"Shared evaluation claims: {len(shared)} / {len(evaluation_ids)}.",
        "Official test predictions are unlabeled. Each row reports one run seed.",
        "Accuracy and F1 count abstentions and invalid outputs as missed labels. Selective accuracy uses valid answers.",
        "ECE uses valid answered rows with finite probabilities. Evidence metrics use annotated claims and the union of gold rationale sentences.",
        "Retrieval hit@8 is the fraction of annotated claims with any gold abstract in the top 8.",
        "C5 citations are inherited from C4 and remain visible when the gate abstains.",
        "Use E5 for the effect of C5 on answer rate and error among answers.",
        "",
        "| condition | seed | accuracy | 95% CI | macro-F1 | NEI F1 | answer rate | selective accuracy | ECE | hit@8 | selected evidence recall | citation F1 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        cells = [
            row["condition"], str(row["seed"]), fmt(row["accuracy"]),
            f"[{fmt(row['accuracy_ci_low'])}, {fmt(row['accuracy_ci_high'])}]",
            *(fmt(row.get(key)) for key in (
                "macro_f1", "nei_f1", "coverage", "selective_accuracy", "ece",
                "retrieval_hit_at_8", "evidence_recall_at_8", "citation_f1",
            )),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    if comparisons:
        lines += [
            "", "## Paired accuracy differences", "",
            "Paired bootstrap intervals resample claims. McNemar p-values are unadjusted.",
            "", "| comparison | baseline seed | seed | delta | 95% CI | McNemar p | verdict |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in comparisons:
            lines.append(
                f"| {row['comparison']} | {row['baseline_seed']} | {row['seed']} | {row['delta']:+.3f} | "
                f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] | {row['p_value']:.3f} | {row['verdict']} |"
            )
    content = "\n".join(lines) + "\n"
    (tables / f"table_ladder_dev{suffix}.md").write_text(content, encoding="utf-8")
    print(content)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
