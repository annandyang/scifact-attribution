"""Export held-out C4 traces for manual error coding and count observable flags."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scifact_attr import ABSTAIN, LABELS
from scifact_attr.config import REPO_ROOT, load_base
from scifact_attr.gate.conformal import partition_claim_ids
from scifact_attr.utils.io import load_jsonl_as_list


POLARITY = {"SUPPORT", "REFUTE"}


def load_rows(path, seed, ids):
    rows = {}
    for row in load_jsonl_as_list(path):
        claim_id = str(row["claim_id"])
        if claim_id not in ids or int(row["seed"]) != seed or row.get("status") == "dry_run":
            continue
        if claim_id in rows:
            raise SystemExit(f"Duplicate claim {claim_id} in {path.name} for seed {seed}.")
        rows[claim_id] = row
    return rows


def load_gate_rows(predictions, source, seed, ids):
    gate_file = predictions / "c5_dev.jsonl"
    frozen_file = predictions.parent / "gate" / "conformal_thresholds.json"
    if not gate_file.exists() or not frozen_file.exists():
        return {}
    frozen = json.loads(frozen_file.read_text(encoding="utf-8"))
    if frozen.get("c4_dev_sha256") != sha256(source.read_bytes()).hexdigest():
        return {}
    return {
        cid: row for cid, row in load_rows(gate_file, seed, ids).items()
        if row.get("gate_partition") == "evaluation"
    }


def sentence_text(sentence_ids, corpus):
    lines = []
    for sid in sentence_ids:
        doc_id, _, index = sid.partition("#")
        doc = corpus.get(doc_id, {})
        sentences = doc.get("sentences", [])
        text = sentences[int(index)] if index.isdigit() and int(index) < len(sentences) else ""
        lines.append(f"[{sid}] {text or '(sentence text unavailable)'}")
    return "\n".join(lines)


def review_row(claim, c4, c5, corpus, second_coder):
    gold = claim["label"]
    if c4.get("gold_label") != gold:
        raise SystemExit(f"Gold label mismatch for claim {claim['claim_id']}.")
    retrieved = [str(doc_id) for doc_id in c4.get("retrieved_ids", [])]
    selected = c4.get("selected_ids", [])
    cited = c4.get("cited_ids", [])
    gold_sentences = claim.get("gold_sentence_ids", [])
    gold_docs = {str(doc_id) for doc_id in claim.get("gold_doc_ids", [])}
    doc_hit = bool(gold_docs.intersection(retrieved))
    sentence_hit = bool(set(gold_sentences).intersection(selected))
    predicted = c4["pred_label"]
    return {
        "claim_id": claim["claim_id"],
        "claim": claim["claim"],
        "gold_label": gold,
        "c4_pred_label": predicted,
        "c4_status": c4.get("status", ""),
        "c5_pred_label": c5["pred_label"] if c5 else "",
        "c5_gate_decision": c5.get("gate_decision", "") if c5 else "",
        "retrieval_miss": bool(gold_docs) and not doc_hit,
        "selection_miss": doc_hit and bool(gold_sentences) and not sentence_hit,
        "polarity_error_with_gold_selected": (
            sentence_hit and gold in POLARITY and predicted in POLARITY and predicted != gold
        ),
        "nei_disagreement": predicted == "NEI" and gold in POLARITY,
        "gate_withheld_correct_c4": (predicted == gold and c5["pred_label"] == ABSTAIN) if c5 else "",
        "invalid_label": predicted not in LABELS,
        "retrieved_titles": "\n".join(
            f"[{doc_id}] {corpus.get(doc_id, {}).get('title', '')}" for doc_id in retrieved
        ),
        "selected_sentences": sentence_text(selected, corpus),
        "cited_sentences": sentence_text(cited, corpus),
        "gold_sentences": sentence_text(gold_sentences, corpus),
        "rationale": c4.get("rationale", ""),
        "raw_output": c4.get("raw_output", ""),
        "second_coder_sample": second_coder,
        "manual_category": "",
        "reviewer": "",
        "review_notes": "",
        "second_coder_category": "",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    paths = {key: REPO_ROOT / value for key, value in load_base()["paths"].items()}
    source = paths["predictions"] / "c4_dev.jsonl"
    if not source.exists():
        raise SystemExit("Run C4 on SciFact dev before exporting E8 review data.")
    official = load_jsonl_as_list(paths["processed"] / "claims_dev.jsonl")
    gold_by_id = {str(row["claim_id"]): row for row in official}
    _, evaluation_ids = partition_claim_ids(gold_by_id)
    c4 = load_rows(source, args.seed, evaluation_ids)
    if not c4:
        raise SystemExit(f"No real held-out C4 predictions for seed {args.seed}.")
    c5 = load_gate_rows(paths["predictions"], source, args.seed, set(c4))
    corpus = {
        str(doc["doc_id"]): doc for doc in load_jsonl_as_list(paths["processed"] / "corpus.jsonl")
    }
    sample = set(sorted(c4, key=lambda cid: sha256(f"scifact-e8:{cid}".encode()).digest())[:60])
    rows = [
        review_row(gold_by_id[cid], c4[cid], c5.get(cid), corpus, cid in sample)
        for cid in sorted(c4)
    ]
    tables = paths["tables"]
    tables.mkdir(parents=True, exist_ok=True)
    review_file = tables / f"e8_review_seed{args.seed}.csv"
    if review_file.exists():
        with review_file.open(encoding="utf-8-sig", newline="") as fh:
            existing = list(csv.DictReader(fh))
        manual_fields = ("manual_category", "reviewer", "review_notes", "second_coder_category")
        if any(row.get(field) for row in existing for field in manual_fields):
            raise SystemExit(f"Preserve the annotated file before regenerating {review_file.name}.")
    with review_file.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    flags = (
        ("retrieval_miss", "Annotated evidence document absent from retrieval"),
        ("selection_miss", "Gold document retrieved, no annotated sentence selected"),
        ("polarity_error_with_gold_selected", "Wrong polarity with an annotated sentence selected"),
        ("nei_disagreement", "NEI emitted for an annotated SUPPORT or REFUTE claim"),
        ("gate_withheld_correct_c4", "C5 withheld an otherwise correct C4 label"),
        ("invalid_label", "No valid C4 label"),
    )
    lines = [
        "# E8 error review on held-out SciFact dev", "",
        f"C4 seed {args.seed}, {len(rows)} of {len(evaluation_ids)} held-out claims.",
        f"Matched C5 predictions available for {len(c5)} reviewed claims.", "",
        "| Observable flag | Flagged claims |", "|---|---|",
    ]
    for key, label in flags:
        count = str(sum(row[key] is True for row in rows))
        if key == "gate_withheld_correct_c4" and not c5:
            count = "unavailable"
        lines.append(f"| {label} | {count} |")
    lines += [
        "",
        "Flags may overlap. Citation and evidence flags use the released annotations.",
        "Review the actual sentences before coding causes. A model can emit NEI for several reasons.",
        "Use the manual category field to record retrieval miss, selection error, contradiction miss, "
        "shortcut-driven NEI, over-refusal, reasoning drift, other, or no error.",
        f"The second-coder sample marks {len(sample)} claims. Record that coding independently before comparing agreement.",
    ]
    (tables / f"e8_breakdown_seed{args.seed}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote review traces for {len(rows)} claims to {review_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
