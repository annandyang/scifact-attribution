"""Compare BM25, BGE dense retrieval, RRF and reranking on labeled SciFact claims."""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from statistics import median, quantiles

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scifact_attr.config import REPO_ROOT, load_base
from scifact_attr.retrieval.bm25 import BM25Index
from scifact_attr.retrieval.evaluate import evaluate_rankings
from scifact_attr.system import BM25Retriever, DenseRetriever, HybridRetriever, RerankedRetriever
from scifact_attr.utils.io import get_logger, load_jsonl_as_list

LOG = get_logger("retrieval-study")
CUTOFFS = (4, 8, 16, 100)
ANCHORS = {
    "bm25": {"full_recall@100": 0.883, "ndcg@10": 0.662},
    "dense": {"ndcg@10": 0.537},
    "hybrid_rrf": {"full_recall@100": 0.923},
    "hybrid_rrf+rerank": {"ndcg@10": 0.659},
}


def load_retrievers(cfg, indexes, processed, device, with_rerank, bm25_only):
    bm25_path = indexes / "bm25_scifact.pkl"
    if not bm25_path.exists():
        raise SystemExit("BM25 index missing; run scripts/02_build_index.py --retriever bm25")
    sparse = BM25Retriever(BM25Index.load(bm25_path))
    retrievers = [sparse]
    if bm25_only:
        return retrievers

    dense_prefix = indexes / "dense_scifact"
    if not dense_prefix.with_suffix(".npy").exists():
        if with_rerank:
            raise SystemExit("Dense index missing; run scripts/02_build_index.py --retriever dense")
        LOG.warning("No dense index found. Running the BM25 baseline.")
        return retrievers

    from scifact_attr.retrieval.dense import Encoder, FlatIndex

    settings = cfg["retrieval"]
    encoder = Encoder(settings["dense"]["model"], device=device,
                      revision=settings["dense"].get("revision"))
    dense = DenseRetriever(FlatIndex.load(dense_prefix), encoder)
    hybrid = HybridRetriever(sparse, dense, k=int(settings["fuse"]["k"]),
                             candidate_depth=int(settings["fuse"]["top_k"]))
    retrievers.extend([dense, hybrid])
    if with_rerank:
        from scifact_attr.selection.selector import CrossEncoderSelector

        corpus = {int(doc["doc_id"]): doc for doc in load_jsonl_as_list(processed / "corpus.jsonl")}
        reranker = CrossEncoderSelector(settings["rerank"]["model"], device=device,
                                        revision=settings["rerank"].get("revision"))
        retrievers.append(RerankedRetriever(hybrid, reranker, corpus,
                                            top_n=int(settings["rerank"]["top_n"])))
    return retrievers


def measure(retriever, claims, gold, sample_ids):
    timings = {k: [] for k in CUTOFFS}
    rankings = {}
    retriever.search(claims[0]["claim"], CUTOFFS[-1])
    for i, claim in enumerate(claims, start=1):
        claim_id, text = claim["claim_id"], claim["claim"]
        started = time.perf_counter()
        ranking = retriever.search(text, CUTOFFS[-1])
        elapsed = 1000 * (time.perf_counter() - started)
        rankings[claim_id] = [str(result["doc_id"]) for result in ranking]
        if claim_id in sample_ids:
            timings[CUTOFFS[-1]].append(elapsed)
            for k in CUTOFFS[:-1]:
                started = time.perf_counter()
                retriever.search(text, k)
                timings[k].append(1000 * (time.perf_counter() - started))
        if i % 50 == 0:
            LOG.info("%s: %d/%d claims", retriever.name, i, len(claims))

    metrics = evaluate_rankings(rankings, gold, cutoffs=CUTOFFS)
    metrics["configuration"] = retriever.name
    for k, samples in timings.items():
        metrics[f"median_ms@{k}"] = median(samples)
        metrics[f"p95_ms@{k}"] = quantiles(samples, n=100, method="inclusive")[94] \
            if len(samples) > 1 else samples[0]
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timing-samples", type=int, default=20)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    methods = parser.add_mutually_exclusive_group()
    methods.add_argument("--with-rerank", action="store_true")
    methods.add_argument("--bm25-only", action="store_true")
    args = parser.parse_args()
    if args.limit < 0 or args.timing_samples < 1:
        parser.error("--limit must be non-negative and --timing-samples must be positive")

    cfg = load_base(args.config)
    indexes = REPO_ROOT / cfg["paths"]["indexes"]
    processed = REPO_ROOT / cfg["paths"]["processed"]
    tables = REPO_ROOT / cfg["paths"]["tables"]
    if args.bm25_only or args.limit:
        tables = tables / "smoke"
    claims = load_jsonl_as_list(processed / f"claims_{args.split}.jsonl")
    if args.limit:
        claims = claims[:args.limit]
    gold = {c["claim_id"]: c["gold_doc_ids"] for c in claims if c["gold_doc_ids"]}
    if not gold:
        raise SystemExit(f"No public gold evidence in the selected {args.split} claims.")
    rng = random.Random(int(cfg["seed"]))
    sample = rng.sample(claims, min(args.timing_samples, len(claims)))
    sample_ids = {claim["claim_id"] for claim in sample}
    LOG.info("%d claims (%d with gold documents); timing sample: %d", len(claims), len(gold),
             len(sample))

    retrievers = load_retrievers(cfg, indexes, processed, args.device, args.with_rerank,
                                args.bm25_only)
    rows = []
    for retriever in retrievers:
        row = measure(retriever, claims, gold, sample_ids)
        rows.append(row)
        print(f"{retriever.name:<22} " + "  ".join(
            f"{key}={row[key]:.3f}" for key in ("recall@4", "recall@8", "recall@16", "ndcg@10")),
            flush=True)

    tables.mkdir(parents=True, exist_ok=True)
    csv_path = tables / "e1_retrieval.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["configuration"] + [
            key for key in rows[0] if key != "configuration"
        ])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# E1 retrieval study", "",
        f"SciFact {args.split}: {len(claims)} claims, {len(gold)} with annotated evidence documents.",
        "Recall@k is the fraction of evidence-bearing claims with at least one gold document",
        "in the top k. Full recall@k averages the fraction of gold documents retrieved per claim.",
        "nDCG uses binary document relevance.", "",
        f"Latency uses {len(sample)} seeded, single-claim queries per cutoff after warmup,",
        "including query encoding and ranking. See the CSV for median and p95 in milliseconds.", "",
        "## Published context", "",
        "[Chen, Sun & Brown (2023)](https://doi.org/10.69987/JACS.2023.30102) report BEIR",
        "SciFact test qrels (300 queries and 339 relevant pairs), BM25 k1=1.2/b=0.75,",
        "a SciFact-trained mean-embedding encoder and a logistic regression reranker.",
        "This study uses official SciFact annotations, BM25 k1=0.9/b=0.4 by default,",
        "a pretrained BGE encoder and a pretrained BGE cross-encoder. These protocol",
        "differences prevent interpreting the gaps as reproduction errors.", "",
        "| configuration | metric | measured here | published |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        for metric, published in ANCHORS[row["configuration"]].items():
            lines.append(f"| {row['configuration']} | {metric} | {row[metric]:.3f} | {published:.3f} |")
    (tables / "e1_anchor_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
