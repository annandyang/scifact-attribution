"""Count the NEI cue and compare matching C3/C3-prime predictions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scifact_attr.audit.shortcut import (compare_c3_and_control, count_empty_evidence_nei,
                                        nei_rate_by_retrieval_strength)
from scifact_attr.config import REPO_ROOT, load_base
from scifact_attr.gate.conformal import partition_claim_ids
from scifact_attr.utils.io import load_jsonl_as_list


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=["dev"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    paths = {key: REPO_ROOT / path for key, path in load_base()["paths"].items()}
    train = load_jsonl_as_list(paths["processed"] / "claims_train.jsonl")
    dev = load_jsonl_as_list(paths["processed"] / "claims_dev.jsonl")
    _, eval_ids = partition_claim_ids(str(c["claim_id"]) for c in dev)
    count = count_empty_evidence_nei(train)
    lines = [
        "# E3: Empty-evidence shortcut audit", "",
        f"Prediction comparison: held-out dev, seed {args.seed}.", "",
        f"- NEI with empty gold evidence: {count['nei_empty_evidence']:.0f}/{count['nei_total']:.0f}.",
        f"- SUPPORT/REFUTE with empty gold evidence: {count['non_nei_empty_evidence']:.0f}/"
        f"{count['non_nei_total']:.0f}.", "",
    ]

    def rows(condition):
        path = paths["predictions"] / f"{condition}_dev.jsonl"
        if not path.exists():
            return []
        return [row for row in load_jsonl_as_list(path)
                if str(row["claim_id"]) in eval_ids and row.get("seed") == args.seed
                and row["status"] != "dry_run"]

    c3 = rows("c3")
    ctrl = rows("c3_prime")
    if c3:
        probe = nei_rate_by_retrieval_strength(c3)
        lines.extend(["| Top retrieval score range | n | NEI rate |",
                      "|---|---:|---:|"])
        for b in probe["buckets"]:
            lines.append(f"| {b['score_low']:.5f} to {b['score_high']:.5f} | "
                         f"{b['n']:.0f} | {b['nei_rate']:.3f} |")
        lines += ["", f"Rate span: {probe['nei_rate_span']:.3f}. "
                  "This association does not establish that the reader used the evidence.", ""]
    result = compare_c3_and_control(c3, ctrl)
    if result["n"]:
        lines.extend([f"{result['n']:.0f} matched claims.", "",
                      "| Metric | C3 | C3-prime | C3 minus C3-prime |",
                      "|---|---:|---:|---:|"])
        for metric in ("macro_f1", "nei_f1", "support_f1", "refute_f1"):
            a = result[f"c3_{metric}"]
            b = result[f"control_{metric}"]
            lines.append(f"| {metric} | {a:.3f} | {b:.3f} | {a-b:+.3f} |")
        lines += ["", "C3-prime receives BM25 distractors for NEI training rows. "
                  "Their irrelevance is an assumption, not a human judgment."]
    else:
        lines.append("C3/C3-prime comparison pending: no matched real prediction rows.")
    out = paths["tables"] / "e3_shortcut.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
