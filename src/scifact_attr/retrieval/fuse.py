"""Reciprocal rank fusion.

RRF needs no score normalisation, which is why it is used here: BM25 scores and cosine
similarities live on different scales, and any normalisation choice would be one more
untraceable decision between C1 and the retrieval study.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

RankedList = Sequence[Tuple[str, float]]  # (doc_id, score), ordered best first


def rrf(ranked_lists: Sequence[RankedList], k: int = 60,
        weights: Optional[Sequence[float]] = None,
        top_k: Optional[int] = None) -> List[Tuple[str, float]]:
    """Fuse ranked lists: score(d) = sum_i w_i / (k + rank_i(d)), rank starting at 1.

    Each list may be a list of doc ids or of (doc_id, score) pairs; only the order is used,
    which is the property that makes RRF immune to score-scale differences.
    """
    if not ranked_lists:
        return []
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match the number of ranked lists")

    fused: Dict[str, float] = {}
    for weight, ranked in zip(weights, ranked_lists):
        for rank, item in enumerate(ranked, start=1):
            doc_id = item[0] if isinstance(item, (tuple, list)) else item
            fused[str(doc_id)] = fused.get(str(doc_id), 0.0) + weight / (k + rank)

    # Tie-break on doc_id so that the fused order is deterministic across runs and platforms.
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered if top_k is None else ordered[:top_k]


def weighted_score_fusion(ranked_lists: Sequence[RankedList],
                          weights: Optional[Sequence[float]] = None,
                          top_k: Optional[int] = None) -> List[Tuple[str, float]]:
    """Alternative fusion: min-max normalise each list, then sum. Kept for the E1 comparison."""
    if not ranked_lists:
        return []
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    fused: Dict[str, float] = {}
    for weight, ranked in zip(weights, ranked_lists):
        if not ranked:
            continue
        scores = [s for _, s in ranked]
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0
        for doc_id, score in ranked:
            fused[doc_id] = fused.get(doc_id, 0.0) + weight * (score - lo) / span

    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered if top_k is None else ordered[:top_k]
