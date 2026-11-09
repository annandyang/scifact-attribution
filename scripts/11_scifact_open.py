"""Prepare SciFact-Open and evaluate frozen SciFact predictions across corpora."""

import argparse
import csv
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scifact_attr.config import REPO_ROOT, load_base
from scifact_attr.eval.bootstrap import paired_delta_ci
from scifact_attr.eval.metrics import summarise
from scifact_attr.gate.conformal import gate_prediction
from scifact_attr.open_data import ARCHIVE_URL, SCOPES, prepare_open
from scifact_attr.utils.io import load_jsonl_as_list, write_jsonl

CONDITIONS = ("c0", "c1", "c2", "c3", "c4", "c5")


def read_predictions(path, ids, expected_condition):
    metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if metadata.get("dry_run") or metadata.get("smoke_adapter"):
        raise ValueError(f"{path.name} contains smoke outputs")
    grouped = {}
    for row in load_jsonl_as_list(path):
        claim_id = row["claim_id"]
        if row["condition"] != expected_condition or row["status"] == "dry_run":
            raise ValueError(f"{path.name} must contain real {expected_condition} predictions")
        if claim_id not in ids:
            continue
        seed_rows = grouped.setdefault(row["seed"], {})
        if claim_id in seed_rows:
            raise ValueError(f"Duplicate seed and claim in {path.name}")
        seed_rows[claim_id] = row
    if not grouped or any(set(seed_rows) != ids for seed_rows in grouped.values()):
        raise ValueError(f"{path.name} must cover every Open claim for each seed")
    return grouped, metadata


def gated(rows, alpha, threshold):
    result = {}
    for claim_id, row in rows.items():
        label, prediction_set = gate_prediction(row, threshold)
        result[claim_id] = {
            **row, "condition": "c5", "base_pred_label": row["pred_label"],
            "pred_label": label, "prediction_set": prediction_set, "gate_threshold": threshold,
            "gate_alpha": alpha, "gate_decision": "ABSTAIN" if label == "ABSTAIN" else "answer",
        }
    return result


def score_rows(rows, claims, k, condition):
    labeled = [claim_id for claim_id, claim in claims.items() if claim["label"] is not None]
    gold = [claims[claim_id]["label"] for claim_id in labeled]
    predicted = [rows[claim_id]["pred_label"] for claim_id in labeled]
    metrics = summarise(gold, predicted)
    eligible = [cid for cid, claim in claims.items() if claim["gold_doc_ids"]]
    hits = [
        cid for cid in eligible
        if set(claims[cid]["gold_doc_ids"]) & set(rows[cid]["retrieved_ids"][:k])
    ]
    labeled_hits = [cid for cid in hits if claims[cid]["label"] is not None]
    has_retrieval = condition != "c0"
    metrics[f"hit@{k}"] = len(hits) / len(eligible) if eligible and has_retrieval else None
    metrics["n_labeled_retrieval_hits"] = len(labeled_hits) if has_retrieval else None
    metrics["accuracy_given_hit"] = mean(
        rows[cid]["pred_label"] == claims[cid]["label"] for cid in labeled_hits
    ) if labeled_hits else None
    metrics["set_coverage"] = mean(
        claims[cid]["label"] in rows[cid]["prediction_set"] for cid in labeled
    ) if condition == "c5" else None
    return metrics, predicted, gold


def evaluate(cfg, paths, scope, conditions, resamples):
    prediction_dir = paths["predictions"]
    open_dir = prediction_dir / "open" / scope
    dataset_dir = paths["processed"] / "open" / scope
    claims = {row["claim_id"]: row for row in load_jsonl_as_list(dataset_dir / "claims_open.jsonl")}
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    k = int(cfg["retrieval"]["top_k_abstracts"])
    alpha = float(cfg["gate"]["apply_alpha"])
    thresholds = {}
    if "c5" in conditions:
        gate_path = prediction_dir.parent / "gate" / "conformal_thresholds.json"
        frozen = json.loads(gate_path.read_text(encoding="utf-8"))
        actual = hashlib.sha256((prediction_dir / "c4_dev.jsonl").read_bytes()).hexdigest()
        if actual != frozen["c4_dev_sha256"]:
            raise ValueError("C4 dev predictions changed since gate calibration")
        thresholds = frozen["thresholds_by_seed"]

    rows = []
    c5_open_rows = []
    for condition in conditions:
        base = "c4" if condition == "c5" else condition
        small, small_meta = read_predictions(prediction_dir / f"{base}_test.jsonl", set(claims), base)
        large, large_meta = read_predictions(open_dir / f"{base}_open.jsonl", set(claims), base)
        if small.keys() != large.keys() or small_meta["retriever"] != large_meta["retriever"]:
            raise ValueError(f"{base}: both runs need the same seeds and retrieval mode")
        if large_meta.get("open_scope") != scope:
            raise ValueError(f"{base}: predictions were made for a different SciFact-Open scope")

        for seed in sorted(small):
            small_rows, open_rows = small[seed], large[seed]
            if condition == "c5":
                threshold = thresholds.get(str(seed), {}).get(str(alpha))
                if threshold is None:
                    raise ValueError(f"Missing frozen threshold for seed {seed}, alpha {alpha}")
                small_rows = gated(small_rows, alpha, threshold)
                open_rows = gated(open_rows, alpha, threshold)
                c5_open_rows.extend(open_rows.values())
            before, before_pred, gold = score_rows(small_rows, claims, k, condition)
            after, after_pred, _ = score_rows(open_rows, claims, k, condition)
            delta = paired_delta_ci(before_pred, after_pred, gold, n_resamples=resamples)
            rows.append({
                "scope": scope, "n_open_docs": manifest["n_docs"], "condition": condition,
                "seed": seed, "retriever": small_meta["retriever"],
                "open_dense_index": large_meta.get("open_dense_index"),
                **{f"small_{name}": value for name, value in before.items()},
                **{f"open_{name}": value for name, value in after.items()},
                "accuracy_delta": delta["delta"], "delta_ci_low": delta["ci_low"],
                "delta_ci_high": delta["ci_high"],
            })
    out = paths["tables"] / f"e6_transfer_{scope}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if c5_open_rows:
        write_jsonl(open_dir / "c5_open.jsonl", c5_open_rows)
    print(f"Scored {manifest['n_claims']} matched Open claims against the same pooled annotations.")
    print(f"Excluded {manifest['n_ambiguous_claims']} conflicting claims from label metrics.")
    print(f"Wrote {out}")


def compare_indexes(cfg, paths, device):
    from scifact_attr.retrieval.dense import Encoder, FlatIndex, IVFIndex, recall_against_exact

    indexes = paths["indexes"] / "open" / "full"
    claims = load_jsonl_as_list(paths["processed"] / "open" / "full" / "claims_open.jsonl")
    exact = FlatIndex.load(indexes / "dense_open")
    approx = IVFIndex.load(indexes / "ivf_open")
    dense = cfg["retrieval"]["dense"]
    encoder = Encoder(dense["model"], device=device, revision=dense.get("revision"))
    queries = encoder.encode_queries([row["claim"] for row in claims])
    k = int(cfg["retrieval"]["top_k_abstracts"])
    stats = {"n_queries": len(claims), "k": k, "nprobe": approx.index.nprobe,
             "approximate_overlap_with_exact": recall_against_exact(approx, exact, queries, k)}
    out = paths["tables"] / "e6_index_overlap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("prepare", help="prepare the official archive")
    prepare.add_argument("--archive", type=Path, required=True, help=ARCHIVE_URL)
    prepare.add_argument("--scope", default="full", choices=SCOPES)
    score = actions.add_parser("score", help="compare completed original-test and Open runs")
    score.add_argument("--scope", default="full", choices=SCOPES)
    score.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    score.add_argument("--resamples", type=int, default=10000)
    check = actions.add_parser("index-check", help="measure full IVF-PQ search overlap with exact")
    check.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()

    cfg = load_base()
    paths = {key: REPO_ROOT / value for key, value in cfg["paths"].items()}
    try:
        if args.action == "prepare":
            manifest = prepare_open(args.archive, paths["processed"] / "open" / args.scope,
                                    args.scope, paths["processed"] / "claims_test.jsonl")
            print(json.dumps(manifest, indent=2))
        elif args.action == "index-check":
            compare_indexes(cfg, paths, args.device)
        else:
            if args.resamples < 1:
                parser.error("--resamples must be positive")
            evaluate(cfg, paths, args.scope, args.conditions, args.resamples)
    except (OSError, ValueError, tarfile.TarError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
