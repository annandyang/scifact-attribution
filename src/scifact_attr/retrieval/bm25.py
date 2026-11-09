"""BM25 with the BEIR/Lucene parameterisation (k1 = 0.9, b = 0.4).

Implemented in-repo rather than imported so that the exact scoring formula is visible and
auditable: the whole point of the retrieval stage in this project is that a number can be
traced to a decision, and an opaque library default is not a decision.
"""

from __future__ import annotations

import math
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens. No stemming and no stop-word removal.

    Stemming would change the recall curve in a way that is hard to attribute, and the
    published anchors were produced with a lightly preprocessed BM25.
    """
    return _TOKEN.findall(str(text).lower())


class BM25Index:
    """Sparse index over documents given as (id, text) pairs."""

    def __init__(self, k1: float = 0.9, b: float = 0.4):
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_len = np.zeros(0, dtype=np.float32)
        self.avgdl = 0.0
        self.idf: Dict[str, float] = {}
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        self._built = False

    # -- construction ---------------------------------------------------------------
    def fit(self, docs: Sequence[Tuple[str, str]]) -> "BM25Index":
        self.doc_ids = []
        self.postings = defaultdict(list)
        doc_len = []

        for internal_idx, (doc_id, text) in enumerate(docs):
            tokens = tokenize(text)
            self.doc_ids.append(str(doc_id))
            doc_len.append(len(tokens))
            term_freq: Dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1
            for token, freq in term_freq.items():
                self.postings[token].append((internal_idx, freq))

        n_docs = len(self.doc_ids)
        self.doc_len = np.asarray(doc_len, dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if n_docs else 0.0

        # Standard BM25 idf, always positive: log(1 + (N - df + 0.5) / (df + 0.5)).
        self.idf = {}
        for token, posting in self.postings.items():
            df = len(posting)
            self.idf[token] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        self._built = True
        return self

    # -- scoring --------------------------------------------------------------------
    def get_scores(self, query: str) -> np.ndarray:
        if not self._built:
            raise RuntimeError("call fit() before get_scores()")
        scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        k1, b, avgdl = self.k1, self.b, (self.avgdl or 1.0)
        for token in tokenize(query):
            idf = self.idf.get(token)
            if idf is None:
                continue
            for doc_idx, tf in self.postings[token]:
                dl = float(self.doc_len[doc_idx])
                denom = tf + k1 * (1.0 - b + b * dl / avgdl)
                scores[doc_idx] += idf * tf * (k1 + 1.0) / denom
        return scores

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        scores = self.get_scores(query)
        if not len(scores):
            return []
        k = min(top_k, len(scores))
        # argpartition then sort the candidate slice: O(k) instead of O(n log n) per query
        if k < len(scores):
            idx = np.argpartition(-scores, k - 1)[:k]
        else:
            idx = np.arange(len(scores))
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        return [(self.doc_ids[i], float(scores[i])) for i in idx if scores[i] > 0.0]

    # -- persistence -----------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"k1": self.k1, "b": self.b, "doc_ids": self.doc_ids,
                         "doc_len": self.doc_len, "avgdl": self.avgdl,
                         "idf": self.idf, "postings": dict(self.postings),
                         "built": self._built}, fh)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        index = cls(k1=state["k1"], b=state["b"])
        index.doc_ids = state["doc_ids"]
        index.doc_len = state["doc_len"]
        index.avgdl = state["avgdl"]
        index.idf = state["idf"]
        index.postings = defaultdict(list, state["postings"])
        index._built = state["built"]
        return index


def build(docs: Sequence[Tuple[str, str]], k1: float = 0.9, b: float = 0.4) -> BM25Index:
    return BM25Index(k1=k1, b=b).fit(docs)


class FrozenBM25:
    """Exact BM25 rankings saved for a fixed set of benchmark queries."""

    name = "bm25"

    def __init__(self, claims, rows, depth):
        expected = {str(row["claim_id"]): row["claim"] for row in claims}
        found = set()
        self.depth = depth
        self.by_query = {}
        for row in rows:
            key = str(row["claim_id"])
            query = row["claim"]
            if key in found or expected.get(key) != query:
                raise ValueError(f"Unknown, duplicate or mismatched ranking for claim {key}")
            ranking = row["results"]
            ids = [item["doc_id"] for item in ranking]
            scores = [item["score"] for item in ranking]
            if (len(ranking) > depth or len(set(ids)) != len(ids)
                    or any(type(doc_id) is not int for doc_id in ids)
                    or any(not math.isfinite(score) or score <= 0 for score in scores)
                    or scores != sorted(scores, reverse=True)):
                raise ValueError(f"Invalid BM25 ranking for claim {key}")
            if query in self.by_query and self.by_query[query] != ranking:
                raise ValueError(f"Identical query text has different results for claim {key}")
            self.by_query[query] = ranking
            found.add(key)
        if found != set(expected):
            raise ValueError("Rankings do not cover every matched claim")

    def search(self, query, top_k):
        if not 0 <= top_k <= self.depth:
            raise ValueError(f"Requested top {top_k} exceeds frozen ranking depth {self.depth}")
        return self.by_query[query][:top_k]
