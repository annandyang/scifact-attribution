"""Run the frozen C3 reader with matched BM25 rankings on small and full corpora."""

import argparse
from hashlib import file_digest
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scifact_attr.config import adapter_dir, load_condition
from scifact_attr.eval.bootstrap import paired_delta_ci
from scifact_attr.eval.metrics import accuracy, macro_f1
from scifact_attr.open_data import DiskSentences
from scifact_attr.reader.infer import Reader
from scifact_attr.selection.selector import CrossEncoderSelector
from scifact_attr.system import RunContext, run_condition
from scifact_attr.utils.io import load_jsonl_as_list, read_jsonl, set_seed, write_jsonl


class FrozenBM25:
    name = "bm25"

    def __init__(self, claims, rows, k):
        expected = {str(row["claim_id"]): row["claim"] for row in claims}
        found = set()
        self.by_query = {}
        for row in rows:
            key = str(row["claim_id"])
            if key in found or expected.get(key) != row["claim"]:
                raise ValueError(f"Unknown, duplicate or mismatched ranking for claim {key}")
            ranking = row["results"]
            ids = [item["doc_id"] for item in ranking]
            scores = [item["score"] for item in ranking]
            if (len(ranking) > k or len(set(ids)) != len(ids)
                    or any(type(doc_id) is not int for doc_id in ids)
                    or any(not math.isfinite(score) or score <= 0 for score in scores)
                    or scores != sorted(scores, reverse=True)):
                raise ValueError(f"Invalid BM25 ranking for claim {key}")
            if row["claim"] in self.by_query and self.by_query[row["claim"]] != ranking:
                raise ValueError(f"Identical query text has different results for claim {key}")
            self.by_query[row["claim"]] = ranking
            found.add(key)
        if found != set(expected):
            raise ValueError("Rankings do not cover every matched Open claim")

    def search(self, query, top_k):
        return self.by_query[query][:top_k]


def build_context(claims, rankings, corpus_path, cfg, selector):
    k = cfg["retrieval"]["top_k_abstracts"]
    retriever = FrozenBM25(claims, load_jsonl_as_list(rankings), k)
    needed = {item["doc_id"] for hits in retriever.by_query.values() for item in hits}
    corpus = {}
    for doc in read_jsonl(corpus_path):
        doc_id = int(doc["doc_id"])
        if doc_id in needed:
            corpus[doc_id] = doc
        if len(corpus) == len(needed):
            break
    if needed - corpus.keys():
        raise ValueError(f"Some ranked documents do not exist in {corpus_path}")
    sentences = DiskSentences(corpus, cfg["corpus"]["max_sentences"])
    return RunContext(corpus, sentences, retriever, selector, cfg)


def summarize(claims, rows):
    annotated = hits = 0
    gold, predicted = [], []
    for claim, row in zip(claims, rows, strict=True):
        if claim["claim_id"] != row["claim_id"]:
            raise ValueError("Prediction order does not match the benchmark")
        if claim["gold_doc_ids"]:
            annotated += 1
            hits += bool(set(claim["gold_doc_ids"]) & set(row["retrieved_ids"]))
        if claim["label"] is not None:
            gold.append(claim["label"])
            predicted.append(row["pred_label"])
    return {"labeled_claims": len(gold), "accuracy": accuracy(gold, predicted),
            "macro_f1": macro_f1(gold, predicted), "hit_at_8": hits / annotated,
            "retrieval_hits": hits, "evidence_claims": annotated}, gold, predicted


def digest(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("small-corpus", "full-corpus", "open-claims", "rankings"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.out and not args.validate_only:
        parser.error("--out is required for inference")
    claims = load_jsonl_as_list(args.open_claims)
    if len(claims) != 279 or len({str(row["claim_id"]) for row in claims}) != 279:
        raise ValueError("Expected all 279 unique SciFact-Open claims")

    cfg = load_condition("c3")
    contexts = {}
    rankings = {scope: args.rankings / f"bm25_{scope}.jsonl" for scope in ("small", "full")}
    for scope, corpus in (("small", args.small_corpus), ("full", args.full_corpus)):
        contexts[scope] = build_context(claims, rankings[scope], corpus, cfg, None)
        print(f"{scope}: checked 279 rankings and {len(contexts[scope].corpus_by_id)} documents",
              flush=True)
    if args.validate_only:
        return

    set_seed(cfg["seed"])
    reader_cfg = cfg["reader"]
    reranker_cfg = cfg["retrieval"]["rerank"]
    adapter = args.adapter or adapter_dir(cfg, 0)
    adapter_hash = digest(adapter / "adapter_model.safetensors")
    reader = Reader(reader_cfg["model"], quant=reader_cfg["quant"], adapter_path=adapter,
                    revision=reader_cfg["revision"])
    selector = CrossEncoderSelector(reranker_cfg["model"], batch_size=16,
                                    revision=reranker_cfg["revision"])
    result, trace, labels = {}, [], {}
    for scope, ctx in contexts.items():
        print(f"Scoring C3 on {scope} BM25 results", flush=True)
        ctx.selector = selector
        rows = run_condition(claims, ctx, reader)
        result[scope], gold, labels[scope] = summarize(claims, rows)
        trace.extend({"scope": scope, **row} for row in rows)
    report = {
        "condition": "c3", "seed": 0, "retriever": "bm25",
        "reader_revision": reader_cfg["revision"], "selector_revision": reranker_cfg["revision"],
        "adapter_sha256": adapter_hash, "claims_sha256": digest(args.open_claims),
        "ranking_sha256": {scope: digest(path) for scope, path in rankings.items()},
        **result, "paired_full_minus_small": paired_delta_ci(labels["small"], labels["full"], gold),
    }
    write_jsonl(args.out.with_suffix(".jsonl"), trace)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
