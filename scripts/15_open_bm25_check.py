"""Measure BM25 retrieval on fixed SciFact-Open claims with bounded memory."""

import argparse
from array import array
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scifact_attr.config import load_base
from scifact_attr.retrieval.bm25 import BM25Index, tokenize
from scifact_attr.utils.io import read_jsonl, write_jsonl


class FrozenQueryBM25(BM25Index):
    def __init__(self, corpus, claims, directory, k1, b):
        super().__init__(k1=k1, b=b)
        terms = sorted({term for row in claims for term in tokenize(row["claim"])})
        self.term_paths = {term: directory / f"{i}.bin" for i, term in enumerate(terms)}
        lengths = array("f")
        postings = defaultdict(lambda: array("I"))
        for index, doc in enumerate(read_jsonl(corpus)):
            tokens = tokenize(f"{doc['title']}. {doc['text']}")
            self.doc_ids.append(str(doc["doc_id"]))
            lengths.append(len(tokens))
            for term, frequency in Counter(term for term in tokens if term in self.term_paths).items():
                postings[term].extend((index, frequency))
            if (index + 1) % 8192 == 0:
                self.flush_postings(postings)
        self.flush_postings(postings)
        if not lengths:
            raise ValueError("The corpus is empty.")
        self.doc_len = np.frombuffer(lengths, dtype=np.dtype("f"))
        self.avgdl = float(self.doc_len.mean()) or 1.0

    def flush_postings(self, postings):
        for term, posting in postings.items():
            with self.term_paths[term].open("ab") as file:
                posting.tofile(file)
        postings.clear()

    def get_scores(self, query):
        n_docs = len(self.doc_ids)
        scores = np.zeros(n_docs, dtype=np.float32)
        for term in tokenize(query):
            path = self.term_paths[term]
            if not path.exists():
                continue
            posting = np.frombuffer(path.read_bytes(), dtype=np.dtype("I")).reshape(-1, 2)
            doc_ids, frequencies = posting.T
            df = len(posting)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            lengths = self.doc_len[doc_ids].astype(np.float64)
            denominator = frequencies + self.k1 * (1.0 - self.b + self.b * lengths / self.avgdl)
            scores[doc_ids] = scores[doc_ids] + idf * frequencies * (self.k1 + 1.0) / denominator
        return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True, choices=("small", "candidates", "full"))
    parser.add_argument("--claims", required=True, type=Path, help="prepared claims_open.jsonl")
    parser.add_argument("--corpus", required=True, type=Path, help="prepared corpus.jsonl")
    parser.add_argument("--reference-index", type=Path, help="compare every score with a local BM25 pickle")
    parser.add_argument("--rankings-dir", type=Path, help="also save local top-k rankings for all claims")
    parser.add_argument("--out", required=True, type=Path, help="aggregate JSON output")
    args = parser.parse_args()

    all_claims = list(read_jsonl(args.claims))
    evidence_claims = [row for row in all_claims if row["gold_doc_ids"]]
    if not evidence_claims:
        parser.error("--claims must contain evidence-bearing claims")
    claims = all_claims if args.rankings_dir else evidence_claims
    config = load_base()["retrieval"]
    k1, b = config["bm25"]["k1"], config["bm25"]["b"]
    top_k = config["top_k_abstracts"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    hits, recalls, rankings = [], [], []
    reference = BM25Index.load(args.reference_index) if args.reference_index else None
    with TemporaryDirectory(dir=args.out.parent) as temp:
        index = FrozenQueryBM25(args.corpus, claims, Path(temp), k1, b)
        if reference is not None and (reference.k1, reference.b, reference.doc_ids) != (k1, b, index.doc_ids):
            parser.error("reference index must use the same BM25 parameters and corpus order")
        for row in claims:
            query = row["claim"]
            if reference is not None and not np.array_equal(index.get_scores(query), reference.get_scores(query)):
                parser.error(f"reference BM25 scores differ for claim {row['claim_id']}")
            results = index.search(query, top_k=top_k)
            if args.rankings_dir:
                rankings.append({
                    "claim_id": row["claim_id"],
                    "claim": query,
                    "results": [{"doc_id": int(doc_id), "score": score} for doc_id, score in results],
                })
            gold = set(map(str, row["gold_doc_ids"]))
            if gold:
                overlap = gold.intersection(doc_id for doc_id, _ in results)
                hits.append(bool(overlap))
                recalls.append(len(overlap) / len(gold))
        posting_bytes = sum(path.stat().st_size for path in Path(temp).iterdir())

    if args.rankings_dir:
        write_jsonl(args.rankings_dir / f"bm25_{args.scope}.jsonl", rankings)
    result = {
        "scope": args.scope,
        "n_docs": len(index.doc_ids),
        "n_evidence_claims": len(evidence_claims),
        "n_ranked_claims": len(rankings),
        "reference_score_parity_verified": reference is not None,
        "bm25": {"k1": k1, "b": b},
        "top_k": top_k,
        "hits": sum(hits),
        "hit_rate": mean(hits),
        "macro_recall": mean(recalls),
        "temporary_postings_bytes": posting_bytes,
        "wall_seconds": round(perf_counter() - started, 2),
    }
    output = json.dumps(result, indent=2)
    args.out.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
