"""Exact dense retrieval for SciFact and an optional FAISS index for larger corpora."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..utils.io import get_logger

LOG = get_logger(__name__)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Encoder:
    """Thin wrapper over sentence-transformers, with a clear failure message."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: Optional[str] = None,
                 batch_size: int = 64, revision: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "sentence-transformers is required for dense retrieval. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        self.model = SentenceTransformer(model_name, device=device, revision=revision)
        self.batch_size = batch_size
        self.model_name = model_name

    def encode(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 512,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        name = self.model_name.lower()
        if "bge-" in name and "-en" in name:
            texts = [BGE_QUERY_PREFIX + text for text in texts]
        return self.encode(texts)


class FlatIndex:
    """Exact cosine search over normalised vectors."""

    def __init__(self, vectors: np.ndarray, ids: Sequence[str]):
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2-D")
        if len(vectors) != len(ids):
            raise ValueError("vectors and ids must have the same length")
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.ids = [str(i) for i in ids]

    def search(self, query_vector: np.ndarray, top_k: int = 100) -> List[Tuple[str, float]]:
        if top_k <= 0 or not self.ids:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        scores = self.vectors @ q                      # cosine, vectors are normalised
        k = min(top_k, len(scores))
        if k < len(scores):
            idx = np.argpartition(-scores, k - 1)[:k]
        else:
            idx = np.arange(len(scores))
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        return [(self.ids[i], float(scores[i])) for i in idx]

    def save(self, prefix: Path) -> None:
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        np.save(prefix.with_suffix(".npy"), self.vectors)
        with open(prefix.with_suffix(".ids.json"), "w", encoding="utf-8") as fh:
            json.dump(self.ids, fh)

    @classmethod
    def load(cls, prefix: Path) -> "FlatIndex":
        prefix = Path(prefix)
        vectors = np.load(prefix.with_suffix(".npy"), mmap_mode="r")
        with open(prefix.with_suffix(".ids.json"), "r", encoding="utf-8") as fh:
            ids = json.load(fh)
        return cls(vectors, ids)


class IVFIndex:
    """FAISS IVF-PQ. Only for the open-domain corpus; approximation is measured."""

    def __init__(self, dim: int, nlist: int = 4096, m: int = 48, nbits: int = 8):
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover
            raise ImportError("faiss is required for the open-domain index") from exc
        self.faiss = faiss
        quantizer = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits,
                                      faiss.METRIC_INNER_PRODUCT)
        self.ids: List[str] = []

    def train(self, vectors: np.ndarray, ids: Sequence[str]) -> None:
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        self.index.train(vectors)
        self.index.add(vectors)
        self.ids = [str(i) for i in ids]

    def fit_large(self, vectors: np.ndarray, ids: Sequence[str], seed: int, nprobe: int) -> None:
        sample_size = min(len(vectors), max(160000, self.index.nlist * 39))
        rng = np.random.default_rng(seed)
        sample = np.sort(rng.choice(len(vectors), sample_size, replace=False))
        self.index.train(vectors[sample])
        for start in range(0, len(vectors), 4096):
            self.index.add(vectors[start:start + 4096])
        self.ids = list(ids)
        self.index.nprobe = min(nprobe, self.index.nlist)

    def save(self, prefix: Path) -> None:
        self.faiss.write_index(self.index, str(prefix.with_suffix(".faiss")))
        prefix.with_suffix(".ids.json").write_text(json.dumps(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, prefix: Path) -> "IVFIndex":
        import faiss

        index = cls.__new__(cls)
        index.faiss = faiss
        index.index = faiss.read_index(str(prefix.with_suffix(".faiss")))
        index.ids = json.loads(prefix.with_suffix(".ids.json").read_text(encoding="utf-8"))
        if index.index.ntotal != len(index.ids):
            raise ValueError("FAISS vector count differs from the stored document IDs")
        return index

    def search(self, query_vector: np.ndarray, top_k: int = 100) -> List[Tuple[str, float]]:
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, idx = self.index.search(q, min(top_k, len(self.ids)))
        return [(self.ids[i], float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]


def build_flat(corpus: Sequence[dict], encoder: Encoder,
               text_field: str = "text") -> FlatIndex:
    texts = []
    for doc in corpus:
        title = (doc.get("title") or "").strip()
        body = (doc.get(text_field) or "").strip()
        texts.append(f"{title}. {body}".strip() if title else body)
    vectors = encoder.encode(texts)
    return FlatIndex(vectors, [str(d["doc_id"]) for d in corpus])


def recall_against_exact(approx: IVFIndex, exact: FlatIndex, queries: Sequence[np.ndarray],
                         k: int = 100) -> float:
    """Overlap of the approximate top-k with the exact top-k, pooled over queries."""
    overlaps, totals = 0, 0
    for q in queries:
        a = {i for i, _ in approx.search(q, k)}
        e = {i for i, _ in exact.search(q, k)}
        overlaps += len(a & e)
        totals += len(e)
    return overlaps / totals if totals else float("nan")
