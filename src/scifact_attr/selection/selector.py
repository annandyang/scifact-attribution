"""Rank and select evidence sentences."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

_TOKEN = re.compile(r"[a-z0-9]+")

# Words that appear in nearly every abstract and therefore carry no selection signal.
STOPWORDS = {
    "the", "of", "and", "in", "to", "a", "is", "are", "was", "were", "we", "our", "this",
    "that", "these", "those", "for", "with", "as", "by", "on", "from", "at", "be", "been",
    "it", "its", "their", "they", "has", "have", "had", "not", "but", "or", "than", "which",
    "can", "may", "also", "such", "more", "most", "between", "after", "before", "during",
}


def _tokens(text: str, drop_stopwords: bool = True) -> List[str]:
    raw = _TOKEN.findall(str(text).lower())
    return [t for t in raw if not (drop_stopwords and t in STOPWORDS)]


class OverlapSelector:
    """IDF-free lexical overlap. Weaker than a cross-encoder, and labelled as such."""

    def score(self, claim: str, sentences: Sequence[str]) -> List[float]:
        claim_tokens = set(_tokens(claim))
        scores = []
        for sent in sentences:
            sent_tokens = set(_tokens(sent))
            if not sent_tokens or not claim_tokens:
                scores.append(0.0)
                continue
            overlap = len(claim_tokens & sent_tokens)
            scores.append(overlap / math.sqrt(len(claim_tokens) * len(sent_tokens)))
        return scores


class CrossEncoderSelector:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: Optional[str] = None,
                 batch_size: int = 32, apply_sigmoid: bool = True, revision: Optional[str] = None):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise ImportError("sentence-transformers is required for the cross-encoder selector") from exc
        self.model = CrossEncoder(model_name, device=device, revision=revision, max_length=512)
        self.batch_size = batch_size
        self.apply_sigmoid = apply_sigmoid

    def score(self, claim: str, sentences: Sequence[str]) -> List[float]:
        if not sentences:
            return []
        pairs = [(claim, s) for s in sentences]
        from torch.nn import Identity

        logits = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False,
                                    activation_fn=Identity())
        values = [float(x) for x in logits]
        if self.apply_sigmoid:
            values = [1.0 / (1.0 + math.exp(-max(-80.0, v))) for v in values]
        return values


def select(scores: Sequence[float], candidates: Sequence[Dict[str, Any]],
           top_n: int = 4, max_per_doc: int = 2,
           min_score: float = 0.0) -> List[Dict[str, Any]]:
    """Pick the best sentences under a per-abstract cap.

    The cap matters: without it, four sentences from one abstract can crowd out the single
    sentence in another abstract that actually contradicts the claim, and C2 would then
    lose to C1 for a reason unrelated to selection quality.
    """
    if len(scores) != len(candidates):
        raise ValueError("scores and candidates must have the same length")
    scored = [(float(s), i, c) for i, (s, c) in enumerate(zip(scores, candidates))]
    scored.sort(key=lambda t: (-t[0], t[1]))

    per_doc: Dict[str, int] = {}
    chosen: List[Dict[str, Any]] = []
    for score, _, cand in scored:
        if score < min_score:
            continue
        doc_key = str(cand.get("doc_id"))
        if per_doc.get(doc_key, 0) >= max_per_doc:
            continue
        per_doc[doc_key] = per_doc.get(doc_key, 0) + 1
        item = dict(cand)
        item["selector_score"] = score
        chosen.append(item)
        if len(chosen) >= top_n:
            break
    return chosen


def select_for_claim(claim: str, candidates: Sequence[Dict[str, Any]],
                     selector: Any, top_n: int = 4, max_per_doc: int = 2,
                     min_score: float = 0.0) -> List[Dict[str, Any]]:
    texts = [c["text"] for c in candidates]
    scores = selector.score(claim, texts)
    return select(scores, candidates, top_n=top_n, max_per_doc=max_per_doc, min_score=min_score)


def sentence_alias_map(selected: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """E1..En -> sentence id, matching render_sentences()."""
    return {f"E{i}": f"{s['doc_id']}#{s.get('sent_idx')}" for i, s in enumerate(selected, start=1)}
