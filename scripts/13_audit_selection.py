"""Compare C1 and C2 using saved SciFact dev predictions on CPU."""

from __future__ import annotations

import argparse
from collections import Counter
from math import isclose
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scifact_attr import LABELS
from scifact_attr.config import REPO_ROOT, load_base
from scifact_attr.gate.conformal import partition_claim_ids
from scifact_attr.system import limited_abstract
from scifact_attr.utils.io import read_jsonl


def load_predictions(path, condition, claim_ids):
    predictions = {}
    for row in read_jsonl(path):
        if row["seed"] != 0:
            continue
        claim_id = str(row["claim_id"])
        if claim_id in predictions:
            raise ValueError(f"{path.name}: duplicate prediction for claim {claim_id}.")
        if row["condition"] != condition or row["split"] != "dev" or row["status"] != "ok":
            raise ValueError(f"{path.name}: expected a completed dev prediction for {claim_id}.")
        if row.get("prediction_source") == "empty_evidence_rule":
            raise ValueError(f"{path.name}: claim {claim_id} uses empty_evidence_rule. Use the original baseline.")
        probabilities = {label: row[f"p_{label.lower()}"] for label in LABELS}
        winner = max(probabilities, key=probabilities.get)
        if (any(not 0 <= value <= 1 for value in probabilities.values())
                or not isclose(sum(probabilities.values()), 1.0)
                or row["pred_label"] != winner
                or not isclose(row["pred_prob"], probabilities[winner])):
            raise ValueError(f"{path.name}: inconsistent probabilities for claim {claim_id}.")
        predictions[claim_id] = row
    if predictions.keys() != claim_ids:
        raise ValueError(f"{path.name}: predictions do not cover the same claims as claims_dev.jsonl.")
    return predictions


def compare_claim(claim, c1, c2, abstracts, c1_abstracts, selection):
    claim_id = claim["claim_id"]
    gold_label = claim["label"]
    for row in (c1, c2):
        if (row["gold_label"] != gold_label or row["gold_doc_ids"] != claim["gold_doc_ids"]
                or row["gold_sentence_ids"] != claim["gold_sentence_ids"]):
            raise ValueError(f"Gold annotation mismatch for claim {claim_id}.")
    if any(c1[key] != c2[key] for key in ("retrieved_ids", "retrieval_scores", "retrieval_mode")):
        raise ValueError(f"Retrieval mismatch for claim {claim_id}.")
    retrieved = set(c1["retrieved_ids"])
    selected = set(c2["selected_ids"])
    selected_docs = Counter(int(sid.split("#")[0]) for sid in selected)
    if (len(selected) != len(c2["selected_ids"]) or len(selected) > selection["top_sentences"]
            or max(selected_docs.values(), default=0) > selection["max_per_abstract"]):
        raise ValueError(f"Selection budget or duplicate sentence violation for claim {claim_id}.")
    for sid in selected:
        doc_id, sent_idx = map(int, sid.split("#"))
        if doc_id not in retrieved or not 0 <= sent_idx < len(abstracts.get(doc_id, ())):
            raise ValueError(f"Selected sentence {sid} is invalid for claim {claim_id}.")

    gold = set(claim["gold_sentence_ids"])
    alternatives = [set(group) for group in claim["gold_evidence_sets"] if group]
    c1_gold = set()
    for sid in gold:
        doc_id, sent_idx = map(int, sid.split("#"))
        if doc_id in retrieved and sent_idx < len(c1_abstracts[doc_id]):
            if c1_abstracts[doc_id][sent_idx] == abstracts[doc_id][sent_idx]:
                c1_gold.add(sid)
    return {
        "label": gold_label,
        "c1_prediction": c1["pred_label"],
        "c2_prediction": c2["pred_label"],
        "c1_correct": c1["pred_label"] == gold_label,
        "c2_correct": c2["pred_label"] == gold_label,
        "annotated": bool(gold),
        "gold_doc": bool(retrieved.intersection(claim["gold_doc_ids"])),
        "c1_gold": bool(c1_gold),
        "c2_gold": bool(selected.intersection(gold)),
        "c1_complete": any(group <= c1_gold for group in alternatives),
        "c2_complete": any(group <= selected for group in alternatives),
        "empty": not selected,
    }


def render_partition(name, rows):
    annotated = [row for row in rows if row["annotated"]]
    lost = [row for row in rows if row["c1_correct"] and not row["c2_correct"]]
    gained = [row for row in rows if not row["c1_correct"] and row["c2_correct"]]
    lost_annotated = [row for row in lost if row["annotated"]]
    empty = [row for row in rows if row["empty"]]
    empty_predictions = Counter(row["c2_prediction"] for row in empty)

    def count(key, group=rows):
        return sum(row[key] for row in group)

    lines = [
        f"## {name}", "",
        "| Metric | Count |", "|---|---:|",
        f"| C1 correct | {count('c1_correct')}/{len(rows)} |",
        f"| C2 correct | {count('c2_correct')}/{len(rows)} |",
        f"| Correct in C1 only | {len(lost)} |",
        f"| Correct in C2 only | {len(gained)} |",
        f"| Gold document retrieved | {count('gold_doc', annotated)}/{len(annotated)} |",
        f"| C1 contains a complete gold sentence | {count('c1_gold', annotated)}/{len(annotated)} |",
        f"| C2 contains a complete gold sentence | {count('c2_gold', annotated)}/{len(annotated)} |",
        f"| C1 contains at least one complete alternative evidence set | {count('c1_complete', annotated)}/{len(annotated)} |",
        f"| C2 contains at least one complete alternative evidence set | {count('c2_complete', annotated)}/{len(annotated)} |",
        f"| Gold document retrieved but no gold sentence selected | {sum(row['gold_doc'] and not row['c2_gold'] for row in annotated)} |",
        f"| Correct in C1 only, with gold evidence annotated | {len(lost_annotated)} |",
        f"| Of those, C2 contains no gold sentence | {len(lost_annotated) - count('c2_gold', lost_annotated)} |",
        f"| Of those, C1 has a complete alternative set and C2 contains no gold sentence | {sum(row['c1_complete'] and not row['c2_gold'] for row in lost_annotated)} |",
        f"| Of those, C2 contains at least one complete alternative set | {count('c2_complete', lost_annotated)} |",
        f"| Of those, C2 predicts NEI | {sum(row['c2_prediction'] == 'NEI' for row in lost_annotated)} |",
        f"| Empty C2 context | {len(empty)}/{len(rows)} |",
        f"| Empty C2 context with a correct verdict | {count('c2_correct', empty)}/{len(empty)} |",
        "", "Empty C2 context predictions were " + ", ".join(
            f"{empty_predictions[label]} {label}" for label in LABELS
        ) + ".", "",
        "| Label | Gold count | C1 predicted | C2 predicted | C1 correct | C2 correct | C1 only | C2 only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        group = [row for row in rows if row["label"] == label]
        predicted_c1 = sum(row["c1_prediction"] == label for row in rows)
        predicted_c2 = sum(row["c2_prediction"] == label for row in rows)
        lost_count = sum(row["label"] == label for row in lost)
        gained_count = sum(row["label"] == label for row in gained)
        lines.append(f"| {label} | {len(group)} | {predicted_c1} | {predicted_c2} | {count('c1_correct', group)} | "
                     f"{count('c2_correct', group)} | {lost_count} | {gained_count} |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=REPO_ROOT,
                        help="project folder containing the run config, processed data, and predictions")
    parser.add_argument("--output", type=Path, help="write Markdown to this file instead of stdout")
    args = parser.parse_args()
    cfg = load_base(args.run_root.resolve() / "configs" / "base.yaml")
    processed = args.run_root / cfg["paths"]["processed"]
    predictions = args.run_root / cfg["paths"]["predictions"]
    claims = {str(row["claim_id"]): row for row in read_jsonl(processed / "claims_dev.jsonl")}
    abstracts = {row["doc_id"]: row["sentences"] for row in read_jsonl(processed / "corpus.jsonl")}
    limit = cfg["corpus"]
    c1_abstracts = {doc_id: limited_abstract(sentences, limit["max_sentences"], limit["max_tokens"])
                    for doc_id, sentences in abstracts.items()}
    c1, c2 = (load_predictions(predictions / f"{condition}_dev.jsonl", condition, claims.keys())
              for condition in ("c1", "c2"))
    calibration, evaluation = partition_claim_ids(claims)
    report = [
        "# E2 evidence selection audit", "",
        "This audit pairs seed 0 C1 and C2 predictions for the same SciFact dev claims. "
        "It checks that retrieved documents and scores match and that selected sentence IDs, "
        "selection budgets, and label probabilities are consistent.", "",
        f"C1 supplies abstracts capped at {limit['max_sentences']} sentences and "
        f"{limit['max_tokens']} words each. A truncated sentence counts only if its full text "
        "reaches the prompt. C2 supplies up to "
        f"{cfg['selection']['top_sentences']} sentences with up to "
        f"{cfg['selection']['max_per_abstract']} from one abstract. "
        "A complete alternative set requires every sentence from at least one annotated rationale.", "",
    ]
    for name, ids in (("Calibration half", calibration), ("Held-out evaluation half", evaluation)):
        rows = [compare_claim(claims[cid], c1[cid], c2[cid], abstracts, c1_abstracts, cfg["selection"])
                for cid in ids]
        report.extend(render_partition(name, rows) + [""])
    report.extend(["Reproduce after running C1 and C2 on dev with `python scripts/13_audit_selection.py`.", ""])
    markdown = "\n".join(report)
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")


if __name__ == "__main__":
    main()
