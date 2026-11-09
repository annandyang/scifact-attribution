"""Compare fixed-retrieval sentence policies, choosing on the calibration half."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scifact_attr.config import adapter_dir, load_condition
from scifact_attr.eval.bootstrap import paired_delta_ci
from scifact_attr.eval.metrics import accuracy, macro_f1, per_class_f1
from scifact_attr.gate.conformal import partition_claim_ids
from scifact_attr.reader.infer import Reader
from scifact_attr.reader.prompts import build_prompt
from scifact_attr.selection.selector import CrossEncoderSelector, select
from scifact_attr.utils.io import load_jsonl_as_list, set_seed, write_jsonl


POLICIES = ("original", "no_threshold", "wide8", "neighbors8")


def sentence_id(row):
    return f"{row['doc_id']}#{row['sent_idx']}"


def with_neighbors(anchors, candidates, limit=8):
    ids = dict.fromkeys(anchors)
    for key in anchors:
        anchor = candidates[key]
        doc, pos = anchor["doc_id"], anchor["sent_idx"]
        for neighbor in (f"{doc}#{pos - 1}", f"{doc}#{pos + 1}"):
            if neighbor in candidates and len(ids) < limit:
                ids[neighbor] = None
    return [candidates[key] for key in ids]


def evidence_coverage(claims, selected):
    annotated = partial = complete = 0
    for claim, ids in zip(claims, selected, strict=True):
        groups = [set(group) for group in claim["gold_evidence_sets"] if group]
        if not groups:
            continue
        selected_ids = set(ids)
        annotated += 1
        partial += any(group & selected_ids for group in groups)
        complete += any(group <= selected_ids for group in groups)
    return annotated, partial, complete


def load_predictions(rows, claims, condition, seed):
    predictions = {}
    for row in rows:
        if row["seed"] != seed:
            continue
        key = int(row["claim_id"])
        if key in predictions:
            raise ValueError(f"Duplicate claim {key} in {condition} seed {seed}")
        predictions[key] = row
    if set(predictions) != {int(claim["claim_id"]) for claim in claims}:
        raise ValueError(f"{condition} seed {seed} does not cover the full dev split")
    for claim in claims:
        row = predictions[int(claim["claim_id"])]
        if row.get("prediction_source") or (
            row["condition"], row["status"], row["mode"], row["claim"], row["gold_label"]
        ) != (
            condition, "ok", "experiment", claim["claim"], claim["label"]
        ):
            raise ValueError(f"Invalid {condition} cache row for claim {claim['claim_id']}")
    return predictions


def require_same_retrieval(key, a, b):
    if a["retrieved_ids"] != b["retrieved_ids"] or a.get("retrieval_scores") != b.get("retrieval_scores"):
        raise ValueError(f"Retrieval differs for claim {key}")


def candidates_for(policies, claim, original, by_doc, selector, cfg):
    sentences = [row for doc in original["retrieved_ids"] for row in by_doc[int(doc)]]
    by_id = {sentence_id(row): row for row in sentences}
    anchors = original["selected_ids"]
    contexts = {"original": [by_id[key] for key in anchors]}
    if "neighbors8" in policies:
        contexts["neighbors8"] = with_neighbors(anchors, by_id)
    scored_policies = set(policies) & {"no_threshold", "wide8"}
    if scored_policies:
        limit = cfg["corpus"]["max_sentences"]
        scored = (sentences if "wide8" in policies
                  else [row for row in sentences if row["sent_idx"] < limit])
        scores = selector.score(claim["claim"], [row["text"] for row in scored])
        if "wide8" in policies:
            contexts["wide8"] = select(scores, scored, top_n=8, max_per_doc=4,
                                       min_score=cfg["selection"]["min_score"])
        if "no_threshold" in policies:
            pairs = [(score, row) for score, row in zip(scores, scored, strict=True)
                     if row["sent_idx"] < limit]
            contexts["no_threshold"] = select([s for s, _ in pairs], [r for _, r in pairs],
                                               top_n=cfg["selection"]["top_sentences"],
                                               max_per_doc=cfg["selection"]["max_per_abstract"])
    return contexts


def run_part(claims, policies, original, by_doc, selector, reader, cfg, partition):
    results = {policy: [] for policy in policies}
    for i, claim in enumerate(claims, 1):
        key = int(claim["claim_id"])
        baseline = original[key]
        base_prompt = cfg["prompt"]
        labels = ({(tuple(baseline["selected_ids"]), base_prompt): baseline["pred_label"]}
                  if reader is None else {})
        contexts = candidates_for(policies, claim, baseline, by_doc, selector, cfg)
        for policy in policies:
            context = contexts["original" if policy in ("empty_rule", "clarify") else policy]
            ids = tuple(map(sentence_id, context))
            template = "grounded_with_definitions" if policy == "clarify" else base_prompt
            key_and_prompt = ids, template
            if key_and_prompt not in labels:
                prompt = build_prompt(template, claim["claim"], sentences=context)
                scores = reader.score_labels(prompt)
                predicted = max(scores, key=scores.get)
                labels[key_and_prompt] = "NEI" if predicted == "NOT_ENOUGH_INFO" else predicted
            results[policy].append({
                "partition": partition, "policy": policy, "claim_id": key,
                "gold_label": claim["label"],
                "pred_label": "NEI" if policy == "empty_rule" and not ids else labels[key_and_prompt],
                "retrieved_ids": baseline["retrieved_ids"], "selected_ids": list(ids),
                "context_words": sum(len(row["text"].split()) for row in context),
            })
        if i % 25 == 0:
            print(f"{partition}: {i}/{len(claims)}", flush=True)
    return results


def describe(claims, rows, baseline):
    gold = [row["gold_label"] for row in rows]
    predicted = [row["pred_label"] for row in rows]
    base_labels = [row["pred_label"] for row in baseline]
    annotated, partial, complete = evidence_coverage(claims, [row["selected_ids"] for row in rows])
    return {
        "n": len(rows), "accuracy": accuracy(gold, predicted), "macro_f1": macro_f1(gold, predicted),
        "f1_by_label": per_class_f1(gold, predicted),
        "paired_accuracy_delta": paired_delta_ci(base_labels, predicted, gold),
        "annotated": annotated, "any_evidence": partial, "complete_evidence": complete,
        "mean_context_words": round(sum(row["context_words"] for row in rows) / len(rows), 1),
        "empty_contexts": sum(not row["selected_ids"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("c2", "c3"), default="c2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--processed", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    study_type = parser.add_mutually_exclusive_group()
    study_type.add_argument("--empty-rule-only", action="store_true",
                            help="replay C2 with the empty-context rule using only cached predictions")
    study_type.add_argument("--clarify-evidence", action="store_true",
                            help="try explicit label definitions on the same C3 evidence")
    args = parser.parse_args()
    if args.seed < 0 or (args.condition == "c2" and (args.seed != 0 or args.adapter)):
        parser.error("C2 uses seed 0 and no adapter; C3 requires a nonnegative seed.")
    if args.empty_rule_only and args.condition != "c2":
        parser.error("--empty-rule-only is a C2 comparison.")
    if args.clarify_evidence and args.condition != "c3":
        parser.error("--clarify-evidence is a C3 comparison.")
    cfg = load_condition(args.condition)
    processed = args.processed or cfg["paths"]["processed"]
    predictions = args.predictions or cfg["paths"]["predictions"]
    sources = [predictions / f"{condition}_dev.jsonl" for condition in ("c1", args.condition)]
    protected = {p.resolve() for path in sources for p in (path, path.with_suffix(".meta.json"))}
    if any(args.out.with_suffix(suffix).resolve() in protected for suffix in (".json", ".jsonl")):
        parser.error("--out must be different from all input prediction and metadata files.")
    claims = load_jsonl_as_list(processed / "claims_dev.jsonl")
    ids = [int(row["claim_id"]) for row in claims]
    calibration_ids, evaluation_ids = partition_claim_ids(ids)
    if len(ids) != 300 or len(set(ids)) != 300:
        raise ValueError("Expected all 300 unique SciFact dev claims")
    calibration = [row for row in claims if str(row["claim_id"]) in calibration_ids]
    evaluation = [row for row in claims if str(row["claim_id"]) in evaluation_ids]
    original = load_predictions(load_jsonl_as_list(predictions / f"{args.condition}_dev.jsonl"),
                                claims, args.condition, args.seed)
    c1 = load_predictions(load_jsonl_as_list(predictions / "c1_dev.jsonl"), claims, "c1", 0)
    by_doc = {}
    for row in load_jsonl_as_list(processed / "sentences.jsonl"):
        by_doc.setdefault(int(row["doc_id"]), []).append(row)
    for rows in by_doc.values():
        rows.sort(key=lambda row: row["sent_idx"])
    for key in ids:
        require_same_retrieval(key, c1[key], original[key])
    reranker = cfg["retrieval"]["rerank"]
    policies = POLICIES
    run_mode = "model_inference"
    if args.empty_rule_only:
        policies, run_mode = ("original", "empty_rule"), "cached_rule_replay"
    elif args.clarify_evidence:
        policies, run_mode = ("original", "clarify"), "reader_prompt_check"
    reader = selector = None
    adapter_hash = None
    if not args.empty_rule_only:
        set_seed(cfg["seed"] + args.seed)
        adapter = (args.adapter or adapter_dir(cfg, args.seed)) if args.condition == "c3" else None
        if adapter is not None and not (adapter / "adapter_config.json").is_file():
            parser.error(f"Adapter not found: {adapter}")
        if adapter is not None:
            with (adapter / "adapter_model.safetensors").open("rb") as weights:
                adapter_hash = file_digest(weights, "sha256").hexdigest()
        reader = Reader(cfg["reader"]["model"], quant=cfg["reader"]["quant"], adapter_path=adapter,
                        revision=cfg["reader"]["revision"])
        if not args.clarify_evidence:
            selector = CrossEncoderSelector(reranker["model"], batch_size=16, revision=reranker["revision"])

    cal = run_part(calibration, policies, original, by_doc, selector, reader, cfg, "calibration")
    cal_metrics = {policy: describe(calibration, rows, cal["original"]) for policy, rows in cal.items()}
    winner = max(policies, key=lambda policy: cal_metrics[policy]["macro_f1"])
    print(f"Selected on calibration macro-F1: {winner}", flush=True)
    eval_policies = tuple(dict.fromkeys(("original", winner)))
    evaluated = run_part(evaluation, eval_policies, original, by_doc, selector, reader, cfg, "evaluation")
    eval_metrics = {policy: describe(evaluation, rows, evaluated["original"])
                    for policy, rows in evaluated.items()}
    for group, rows, metrics in ((calibration, cal, cal_metrics), (evaluation, evaluated, eval_metrics)):
        gold = [row["label"] for row in group]
        labels = [c1[int(row["claim_id"])]["pred_label"] for row in group]
        metrics["c1_reference"] = {"n": len(gold), "accuracy": accuracy(gold, labels),
                                   "macro_f1": macro_f1(gold, labels)}
        metrics["baseline_cache_agreement"] = sum(
            row["pred_label"] == original[row["claim_id"]]["pred_label"] for row in rows["original"]
        )
    report = {"condition": args.condition, "seed": args.seed, "chosen_policy": winner,
              "run_mode": run_mode,
              "reader_revision": cfg["reader"]["revision"], "selector_revision": reranker["revision"],
              "adapter_sha256": adapter_hash,
              "calibration": cal_metrics, "evaluation": eval_metrics}
    write_jsonl(args.out.with_suffix(".jsonl"), [
        row for results in (cal, evaluated) for rows in results.values() for row in rows
    ])
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
