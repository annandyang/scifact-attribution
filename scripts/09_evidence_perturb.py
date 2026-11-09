"""Compare matched evidence perturbations and label-first decisions on held-out dev."""

from __future__ import annotations

import argparse
import csv
import gc
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scifact_attr import LABELS
from scifact_attr.config import adapter_dir, load_condition
from scifact_attr.gate.conformal import partition_claim_ids
from scifact_attr.reader.infer import Reader, labels_to_predictions
from scifact_attr.reader.parse import parse_evidence_assessment
from scifact_attr.reader.prompts import build_prompt
from scifact_attr.selection.selector import sentence_alias_map
from scifact_attr.system import generate_assessments, reasoned_verdict
from scifact_attr.utils.io import load_jsonl_as_list, set_seed, write_jsonl


def matched_pair(selected, cited_ids, distractors, rng, intervention="replace"):
    cited = [sid for sid in selected if sid in cited_ids]
    uncited = [sid for sid in selected if sid not in cited_ids]
    count = len(cited)
    if not count or len(uncited) < count:
        return None
    matched_uncited = set(rng.sample(uncited, count))

    if intervention == "drop":
        return ([sid for sid in selected if sid not in cited],
                [sid for sid in selected if sid not in matched_uncited])
    if len(distractors) < count:
        return None

    replacement = rng.sample(distractors, count)

    def replace(ids):
        texts = iter(replacement)
        return [next(texts) if sid in ids else sid for sid in selected]

    return replace(set(cited)), replace(matched_uncited)


def prepare_case(row, claim, sentence_lookup, cfg, intervention):
    ids = row.get("selected_ids") or []
    sentences = [sentence_lookup[sid] for sid in ids]
    prompt_name = cfg["prompt"]
    prompt = build_prompt(prompt_name, claim["claim"], sentences=sentences)
    result = {
        "claim_id": str(row["claim_id"]),
        "claim": claim["claim"],
        "seed": row["seed"],
        "gold_label": row["gold_label"],
        "base_label": row["pred_label"],
        "base_status": row["status"],
        "base_selected_ids": ids,
        "base_cited_ids": row.get("cited_ids") or [],
    }
    jobs = [("label_first", prompt, sentences, result)]

    if row["pred_label"] not in LABELS or row["status"] != "ok":
        return result, jobs
    cited = set(result["base_cited_ids"]) & set(ids)
    if not cited or len(ids) < 2 * len(cited):
        return result, jobs

    rng = random.Random(f"{row['seed']}:{row['claim_id']}")
    distractors = []
    if intervention == "replace":
        excluded_docs = set(row["retrieved_ids"]) | set(claim["gold_doc_ids"])
        selected_text = {s["text"].strip().casefold() for s in sentences}
        distractors = [sid for sid, sentence in sentence_lookup.items()
                       if sentence["doc_id"] not in excluded_docs
                       and sentence["text"].strip().casefold() not in selected_text]
    pair = matched_pair(ids, cited, distractors, rng, intervention)
    if pair is None:
        return result, jobs

    for name, changed_ids in zip(("cited", "uncited"), pair):
        changed = [sentence_lookup[sid] for sid in changed_ids]
        changed_prompt = build_prompt(prompt_name, claim["claim"], sentences=changed)
        result[f"{name}_selected_ids"] = changed_ids
        jobs.append((name, changed_prompt, changed, result))
    return result, jobs


def run_jobs(reader, jobs, gen, seed):
    batch_size = int(gen["batch_size"])
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start:start + batch_size]
        for name, _, sentences, result in batch:
            if name == "label_first":
                prompt = build_prompt("grounded_label_only", result["claim"], sentences=sentences)
                scores = reader.score_labels(prompt)
                result["label_first"] = labels_to_predictions(scores)["pred_label"]

        outputs = generate_assessments(reader, [(prompt, sents) for _, prompt, sents, _ in batch], gen)
        for (name, _, sentences, result), raw in zip(batch, outputs, strict=True):
            if name == "label_first":
                cited, rationale, status = parse_evidence_assessment(raw, sentence_alias_map(sentences))
            else:
                label, _, cited, rationale, status = reasoned_verdict(result["claim"], sentences, raw, reader)
                result[f"{name}_label"] = label
            result.update({f"{name}_assessment_ids": cited, f"{name}_assessment_status": status,
                           f"{name}_assessment_text": rationale})
        print(f"E4 seed {seed}: {min(start + batch_size, len(jobs))}/{len(jobs)} variants", flush=True)


def summarise(rows, seed, intervention):
    matched = [row for row in rows if "cited_label" in row]
    joint = [row for row in matched
             if row["cited_label"] in LABELS and row["uncited_label"] in LABELS]
    first_assessed = [row for row in rows if row["base_selected_ids"]]

    def fraction(count, total):
        return count / total if total else None

    def assessment_rates(name, group):
        return {
            f"{name}_assessment_ok_rate": fraction(
                sum(row[f"{name}_assessment_status"] == "ok" for row in group), len(group)),
            f"{name}_assessment_citation_rate": fraction(
                sum(bool(row[f"{name}_assessment_ids"]) for row in group), len(group)),
        }

    cited_flip = fraction(sum(row["cited_label"] != row["base_label"] for row in joint), len(joint))
    control_flip = fraction(sum(row["uncited_label"] != row["base_label"] for row in joint), len(joint))
    base_acc = sum(row["base_label"] == row["gold_label"] for row in rows) / len(rows)
    first_acc = sum(row["label_first"] == row["gold_label"] for row in rows) / len(rows)
    return {
        "seed": seed,
        "intervention": intervention,
        "n_held_out": len(rows),
        "n_matched": len(matched),
        "n_comparable_labels": len(joint),
        "cited_flip_rate": cited_flip,
        "uncited_flip_rate": control_flip,
        "flip_rate_delta": cited_flip - control_flip if joint else None,
        "c4_accuracy": base_acc,
        "label_first_accuracy": first_acc,
        "label_first_accuracy_delta": first_acc - base_acc,
        "label_first_n_assessed": len(first_assessed),
        **assessment_rates("label_first", first_assessed),
        **assessment_rates("cited", matched),
        **assessment_rates("uncited", matched),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervention", choices=("replace", "drop"), default="replace")
    parser.add_argument("--seed", type=int, help="Use one seed. The default uses all cached seeds.")
    parser.add_argument("--limit", type=int, default=0, help="Write an explicitly limited smoke run.")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit cannot be negative")

    cfg = load_condition("c4")
    paths = cfg["paths"]
    c4_path = paths["predictions"] / "c4_dev.jsonl"
    if not c4_path.is_file():
        raise SystemExit("Run C4 on dev with script 04 before running E4.")
    cached = load_jsonl_as_list(c4_path)
    official = {str(row["claim_id"]): row
                for row in load_jsonl_as_list(paths["processed"] / "claims_dev.jsonl")}
    _, held_out = partition_claim_ids(official)
    sentence_lookup = {row["sid"]: row
                       for row in load_jsonl_as_list(paths["processed"] / "sentences.jsonl")}
    rows = {}
    seen = set()
    for row in cached:
        cid, seed = str(row["claim_id"]), int(row["seed"])
        if (cid, seed) in seen:
            raise SystemExit(f"Duplicate C4 prediction for claim {cid}, seed {seed}.")
        seen.add((cid, seed))
        if (row.get("condition") != "c4" or row.get("split") != "dev"
                or row.get("status") == "dry_run" or cid not in official
                or row.get("gold_label") != official[cid]["label"]):
            raise SystemExit("E4 needs real C4 dev predictions with matching SciFact labels.")
        if cid in held_out and (args.seed is None or seed == args.seed):
            missing = set(row.get("selected_ids") or []) - sentence_lookup.keys()
            if missing:
                raise SystemExit(f"C4 cache uses missing sentence IDs for claim {cid}: {sorted(missing)}")
            rows.setdefault(seed, []).append(row)
    if not rows:
        raise SystemExit("No matching held-out C4 rows.")
    for seed, group in rows.items():
        if {str(row["claim_id"]) for row in group} != held_out and not args.limit:
            raise SystemExit(f"C4 seed {seed} lacks the full held-out split. Use --limit for a smoke run.")
        path = adapter_dir(cfg, seed)
        if not (path / "adapter_config.json").is_file():
            raise SystemExit(f"Missing C4 adapter for seed {seed}: {path}")

    records, summaries = [], []
    for seed, group in sorted(rows.items()):
        set_seed(int(cfg["seed"]) + seed)
        reader = Reader(cfg["reader"]["model"], quant=cfg["reader"]["quant"],
                        adapter_path=str(adapter_dir(cfg, seed)),
                        revision=cfg["reader"].get("revision"))
        group = sorted(group, key=lambda row: int(row["claim_id"]))
        if args.limit:
            group = group[:args.limit]
        seed_rows, jobs = [], []
        for row in group:
            cid = str(row["claim_id"])
            result, case_jobs = prepare_case(row, official[cid], sentence_lookup, cfg, args.intervention)
            seed_rows.append(result)
            jobs.extend(case_jobs)
        run_jobs(reader, jobs, cfg["reader"]["gen"], seed)
        records.extend(seed_rows)
        summaries.append(summarise(seed_rows, seed, args.intervention))
        torch = reader.torch
        del reader
        gc.collect()
        torch.cuda.empty_cache()

    smoke = Path("smoke") if args.limit else Path(".")
    raw_path = paths["predictions"] / smoke / f"c4_perturb_dev_{args.intervention}.jsonl"
    table_path = paths["tables"] / smoke / f"e4_evidence_perturb_{args.intervention}.csv"
    write_jsonl(raw_path, records)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote {len(records)} rows to {raw_path}")
    print(f"Summary: {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
