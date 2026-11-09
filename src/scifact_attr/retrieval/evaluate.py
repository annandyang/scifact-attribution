"""Retrieval metrics: recall@k, evidence recall, nDCG@k.

Definitions are pinned here once so that the E1 table cannot change definition between
runs. The published anchors use abstract-level relevance with binary gains; that is what
these functions compute.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def recall_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    """1.0 if at least one gold document is in the top k, else 0.0 (per-claim hit rate)."""
    gold = {str(g) for g in gold_ids}
    if not gold:
        return float("nan")
    return 1.0 if any(str(d) in gold for d in list(ranked_ids)[:k]) else 0.0


def full_recall_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    """Fraction of gold documents recovered in the top k."""
    gold = {str(g) for g in gold_ids}
    if not gold:
        return float("nan")
    hits = sum(1 for d in list(ranked_ids)[:k] if str(d) in gold)
    return hits / len(gold)


def ndcg_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    """Binary-gain nDCG@k: gain 1 for a gold document, 0 otherwise."""
    gold = {str(g) for g in gold_ids}
    if not gold:
        return float("nan")
    dcg = 0.0
    for i, doc_id in enumerate(list(ranked_ids)[:k]):
        if str(doc_id) in gold:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def mean_metric(values: Sequence[float]) -> float:
    """Mean ignoring NaN, so that claims without gold evidence do not drag the average."""
    arr = np.asarray([v for v in values if not (isinstance(v, float) and math.isnan(v))],
                     dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def evaluate_rankings(rankings: Dict[str, List[str]],
                      gold: Dict[str, List[str]],
                      cutoffs: Sequence[int] = (4, 8, 16, 100),
                      ndcg_cutoff: int = 10) -> Dict[str, float]:
    """Aggregate over claims. `rankings` and `gold` are keyed by claim id."""
    out: Dict[str, float] = {}
    claim_ids = [cid for cid in rankings if cid in gold and gold[cid]]
    if not claim_ids:
        return {"n_claims": 0}
    for k in cutoffs:
        out[f"recall@{k}"] = mean_metric([recall_at_k(rankings[c], gold[c], k) for c in claim_ids])
        out[f"full_recall@{k}"] = mean_metric(
            [full_recall_at_k(rankings[c], gold[c], k) for c in claim_ids])
    out[f"ndcg@{ndcg_cutoff}"] = mean_metric(
        [ndcg_at_k(rankings[c], gold[c], ndcg_cutoff) for c in claim_ids])
    out["n_claims"] = float(len(claim_ids))
    return out
