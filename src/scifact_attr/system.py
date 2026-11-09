"""Connect retrieval, evidence selection, and the reader."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .gate.conformal import nonconformity
from .reader.parse import citation_metrics, parse_evidence_assessment
from .reader.prompts import build_prompt
from .retrieval.fuse import rrf
from .selection.selector import sentence_alias_map
from .utils.io import get_logger

LOG = get_logger(__name__)


@dataclass
class RunContext:
    """Everything a run needs that is not a model: corpus, sentences and the retriever."""
    corpus_by_id: Dict[int, Dict[str, Any]]
    sentences_by_doc: Dict[int, List[Dict[str, Any]]]
    retriever: Any
    selector: Any
    config: Dict[str, Any]
    extra: Dict[str, Any] = field(default_factory=dict)


class BM25Retriever:
    def __init__(self, index):
        self.index = index
        self.name = "bm25"

    def search(self, claim: str, top_k: int) -> List[Dict[str, Any]]:
        return [{"doc_id": d, "score": s} for d, s in self.index.search(claim, top_k=top_k)]


class DenseRetriever:
    def __init__(self, index, encoder):
        self.index = index
        self.encoder = encoder
        self.name = "dense"

    def search(self, claim: str, top_k: int) -> List[Dict[str, Any]]:
        vector = self.encoder.encode_queries([claim])[0]
        return [{"doc_id": d, "score": s} for d, s in self.index.search(vector, top_k=top_k)]


class HybridRetriever:
    """RRF over BM25 and dense. Weights default to equal, which is the published setup."""

    def __init__(self, sparse, dense, k: int = 60, weights: Optional[Sequence[float]] = None,
                 candidate_depth: int = 100):
        self.sparse = sparse
        self.dense = dense
        self.k = k
        self.weights = list(weights) if weights else [1.0, 1.0]
        self.candidate_depth = candidate_depth
        self.name = "hybrid_rrf"

    def search(self, claim: str, top_k: int) -> List[Dict[str, Any]]:
        depth = max(top_k, self.candidate_depth)
        sparse = self.sparse.search(claim, depth)
        dense = self.dense.search(claim, depth)
        fused = rrf([[(r["doc_id"], r["score"]) for r in sparse],
                     [(r["doc_id"], r["score"]) for r in dense]],
                    k=self.k, weights=self.weights, top_k=top_k)
        return [{"doc_id": d, "score": s} for d, s in fused]


class RerankedRetriever:
    def __init__(self, base, reranker, corpus_by_id: Dict[int, Dict[str, Any]], top_n: int = 50):
        self.base = base
        self.reranker = reranker
        self.corpus_by_id = corpus_by_id
        self.top_n = top_n
        self.name = f"{getattr(base, 'name', 'base')}+rerank"

    def search(self, claim: str, top_k: int) -> List[Dict[str, Any]]:
        base_ranking = self.base.search(claim, max(top_k, self.top_n))
        candidates = base_ranking[:self.top_n]
        if not candidates:
            return base_ranking[:top_k]
        texts = []
        for cand in candidates:
            doc = self.corpus_by_id[int(cand["doc_id"])]
            texts.append(" ".join((doc.get("title", ""), doc["text"])).strip())
        scores = self.reranker.score(claim, texts)
        ranked = sorted(zip(candidates, scores), key=lambda pair: -pair[1])
        prefix = [{"doc_id": row["doc_id"], "score": float(score)} for row, score in ranked]
        return (prefix + base_ranking[self.top_n:])[:top_k]


def _candidates_for(top_docs: Sequence[Dict[str, Any]],
                    sentences_by_doc: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for doc in top_docs:
        for sent in sentences_by_doc.get(int(doc["doc_id"]), []):
            out.append({"doc_id": int(doc["doc_id"]), "sent_idx": sent["sent_idx"],
                        "text": sent["text"], "retrieval_score": doc["score"]})
    return out


def prepare_claim(claim, ctx):
    cfg = ctx.config
    components = cfg.get("components", {})
    started = time.perf_counter()

    retrieved: List[Dict[str, Any]] = []
    abstracts: List[Dict[str, Any]] = []
    selected: List[Dict[str, Any]] = []
    prompt = ""
    top_score = 0.0

    if components.get("retrieval"):
        retrieved = ctx.retriever.search(claim["claim"], int(cfg["retrieval"]["top_k_abstracts"]))
        abstracts = [ctx.corpus_by_id[int(r["doc_id"])] for r in retrieved
                     if int(r["doc_id"]) in ctx.corpus_by_id]
        top_score = float(retrieved[0]["score"]) if retrieved else 0.0

    if components.get("retrieval") and components.get("selection"):
        candidates = _candidates_for(retrieved, ctx.sentences_by_doc)
        sel_cfg = cfg.get("selection", {})
        if candidates:
            scores = ctx.selector.score(claim["claim"], [c["text"] for c in candidates])
            from .selection.selector import select
            selected = select(scores, candidates,
                              top_n=int(sel_cfg.get("top_sentences", 4)),
                              max_per_doc=int(sel_cfg.get("max_per_abstract", 2)),
                              min_score=float(sel_cfg.get("min_score", 0.0)))

    template = cfg.get("prompt", "closed_book")
    if components.get("retrieval") and components.get("selection"):
        prompt = build_prompt(template, claim["claim"], sentences=selected)
    elif components.get("retrieval"):
        max_sentences = int(cfg["corpus"]["max_sentences"])
        max_words = int(cfg["corpus"]["max_tokens"])
        prompt = build_prompt(template, claim["claim"], abstracts=[
            {**doc, "sentences": limited_abstract(doc["sentences"], max_sentences, max_words)}
            for doc in abstracts
        ])
    else:
        prompt = build_prompt("closed_book", claim["claim"])

    return prompt, retrieved, selected, top_score, time.perf_counter() - started


def run_claim(claim: Dict[str, Any], ctx: RunContext, reader: Any, seed: int = 0,
              prepared=None, generated=None, generation_time: float = 0.0) -> Dict[str, Any]:
    cfg = ctx.config
    components = cfg.get("components", {})
    prompt, retrieved, selected, top_score, preparation_time = (
        prepared if prepared is not None else prepare_claim(claim, ctx)
    )
    started = time.perf_counter()
    probs: Dict[str, float] = {}
    pred_label = ""
    pred_prob = float("nan")
    rationale = ""
    cited: List[str] = []
    status = "ok"
    raw = ""

    if reader is None:
        # dry run: exercises the pipeline without a model (used by --dry-run and by tests)
        pred_label = "NEI"
        pred_prob = 1.0 / 3.0
        probs = {"SUPPORT": 1 / 3, "REFUTE": 1 / 3, "NEI": 1 / 3}
        status = "dry_run"
    elif components.get("reasoning"):
        raw = generated if generated is not None else generate_assessments(
            reader, [(prompt, selected)], cfg["reader"]["gen"]
        )[0]
        pred_label, probs, cited, rationale, status = reasoned_verdict(
            claim["claim"], selected, raw, reader
        )
        pred_prob = float(probs[pred_label])
    else:
        scores = reader.score_labels(prompt)
        mapped = {("NEI" if k == "NOT_ENOUGH_INFO" else k): v for k, v in scores.items()}
        probs = mapped
        pred_label = max(mapped, key=mapped.get)
        pred_prob = float(mapped[pred_label])

    gold_sent_ids = claim.get("gold_sentence_ids", [])
    cite = citation_metrics(cited, gold_sent_ids) if cited else {"precision": 0.0, "recall": 0.0,
                                                                 "f1": 0.0, "n_cited": 0.0}

    return {
        "claim_id": claim["claim_id"],
        "claim": claim["claim"],
        "split": claim.get("split"),
        "condition": cfg["condition"],
        "seed": seed,
        "pred_label": pred_label,
        "gold_label": claim.get("label"),
        "gold_doc_ids": claim.get("gold_doc_ids", []),
        "gold_sentence_ids": gold_sent_ids,
        "pred_prob": pred_prob,
        "p_support": probs.get("SUPPORT", float("nan")),
        "p_refute": probs.get("REFUTE", float("nan")),
        "p_nei": probs.get("NEI", float("nan")),
        "conformal_score": nonconformity(pred_prob) if pred_prob == pred_prob else float("nan"),
        "gate_decision": "",
        "retrieved_ids": [int(r["doc_id"]) for r in retrieved],
        "retrieval_mode": getattr(ctx.retriever, "name", "none"),
        "retrieval_scores": [float(r["score"]) for r in retrieved],
        "top_score": top_score,
        "selected_ids": [f"{s['doc_id']}#{s['sent_idx']}" for s in selected],
        "cited_ids": cited,
        "citation_precision": cite["precision"],
        "citation_recall": cite["recall"],
        "citation_f1": cite["f1"],
        "rationale": rationale,
        "raw_output": raw,
        "status": status,
        "latency_ms": int((preparation_time + generation_time + time.perf_counter() - started) * 1000),
        "peak_mem_mb": 0,
    }


def limited_abstract(sentences, max_sentences, max_words):
    result = []
    for sentence in sentences[:max_sentences]:
        words = sentence.split()
        if not max_words:
            break
        result.append(sentence if len(words) <= max_words else " ".join(words[:max_words]))
        max_words -= min(len(words), max_words)
    return result


def generate_assessments(reader, contexts, gen_cfg):
    positions = [i for i, (_, selected) in enumerate(contexts) if selected]
    results = [""] * len(contexts)
    if positions:
        raw = reader.generate([contexts[i][0] for i in positions],
                              max_new_tokens=int(gen_cfg["max_new_tokens"]),
                              batch_size=int(gen_cfg["batch_size"]), use_adapter=False)
        for position, explanation in zip(positions, raw, strict=True):
            results[position] = explanation
    return results


def reasoned_verdict(claim, selected, raw, reader):
    cited, rationale, status = parse_evidence_assessment(raw, sentence_alias_map(selected))
    prompt = build_prompt("grounded_label_only", claim, sentences=selected,
                          note=rationale if status == "ok" else "")
    scores = reader.score_labels(prompt)
    probs = {("NEI" if label == "NOT_ENOUGH_INFO" else label): prob
             for label, prob in scores.items()}
    return max(probs, key=probs.get), probs, cited, rationale, status


def run_condition(claims: Sequence[Dict[str, Any]], ctx: RunContext, reader: Any,
                  seeds: Sequence[int] = (0,), progress: bool = True) -> List[Dict[str, Any]]:
    """Run every claim for every seed. Rows are returned, not written; the caller persists."""
    rows: List[Dict[str, Any]] = []
    total = len(claims) * len(seeds)
    done = 0
    reasoning = reader is not None and ctx.config.get("components", {}).get("reasoning")
    gen_cfg = ctx.config["reader"]["gen"] if reasoning else {}
    batch_size = int(gen_cfg.get("batch_size", 1))
    for seed in seeds:
        for start in range(0, len(claims), batch_size):
            batch = claims[start:start + batch_size]
            if reasoning:
                prepared = [prepare_claim(claim, ctx) for claim in batch]
                tick = time.perf_counter()
                outputs = generate_assessments(reader, [(item[0], item[2]) for item in prepared], gen_cfg)
                per_claim = (time.perf_counter() - tick) / len(batch)
                for claim, item, raw in zip(batch, prepared, outputs, strict=True):
                    rows.append(run_claim(claim, ctx, reader, seed, item, raw, per_claim))
            else:
                rows.extend(run_claim(claim, ctx, reader, seed=seed) for claim in batch)
            previous = done
            done += len(batch)
            if progress and (done // 25 > previous // 25 or done == total):
                LOG.info("%s: %d/%d claims (seed %s)", ctx.config["condition"], done, total, seed)
    return rows
