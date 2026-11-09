"""Count the training cue and compare held-out predictions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..eval.metrics import macro_f1, per_class_f1


def count_empty_evidence_nei(train_claims: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Measurement 1: the artefact, counted on the split in use."""
    nei = [c for c in train_claims if c.get("label") == "NEI"]
    non_nei = [c for c in train_claims if c.get("label") != "NEI"]
    empty_nei = sum(1 for c in nei if not c.get("gold_sentences"))
    empty_non_nei = sum(1 for c in non_nei if not c.get("gold_sentences"))
    return {
        "nei_total": float(len(nei)),
        "nei_empty_evidence": float(empty_nei),
        "nei_empty_fraction": float(empty_nei / len(nei)) if nei else float("nan"),
        "non_nei_total": float(len(non_nei)),
        "non_nei_empty_evidence": float(empty_non_nei),
        "cue_is_diagnostic": float(empty_nei / len(nei)) if nei else 0.0,
    }


def nei_rate_by_retrieval_strength(rows: Sequence[Dict[str, Any]],
                                   n_buckets: int = 4) -> Dict[str, Any]:
    """Measurement 2: does the NEI rate follow retrieval strength or the training cue?

    rows need `pred_label` and `top_score` (best retrieval score for the claim). Claims are
    bucketed by top_score; within each bucket we report the NEI rate and the fraction of
    claims where retrieval found nothing above a usable threshold.

    Interpretation is stated in advance, so the result cannot be read whichever way it
    lands: if NEI rate is flat across buckets, the label is not being driven by evidence;
    if it rises as retrieval weakens, the model is using retrieval, which is the behaviour
    we want and which makes the shortcut claim weaker than expected.
    """
    if not rows:
        return {"n": 0}
    scores = np.asarray([float(r.get("top_score", 0.0)) for r in rows], dtype=np.float64)
    labels = [r.get("pred_label") for r in rows]
    edges = np.quantile(scores, np.linspace(0, 1, n_buckets + 1))
    edges[-1] = edges[-1] + 1e-9
    buckets: List[Dict[str, float]] = []
    for b in range(n_buckets):
        mask = (scores >= edges[b]) & (scores < edges[b + 1])
        n = int(mask.sum())
        if n == 0:
            continue
        nei = sum(1 for i in np.where(mask)[0] if labels[i] == "NEI")
        buckets.append({
            "bucket": float(b),
            "score_low": float(edges[b]),
            "score_high": float(edges[b + 1]),
            "n": float(n),
            "nei_rate": float(nei / n),
        })
    rate_span = max(b["nei_rate"] for b in buckets) - min(b["nei_rate"] for b in buckets)
    return {"n": float(len(rows)), "buckets": buckets, "nei_rate_span": float(rate_span)}


def compare_c3_and_control(c3_rows: Sequence[Dict[str, Any]],
                           control_rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Measurement 3: C3 vs C3' on the same claims, with NEI F1 separated from macro-F1."""
    ids_c3 = {r["claim_id"]: r for r in c3_rows if r.get("gold_label")}
    ids_ctrl = {r["claim_id"]: r for r in control_rows if r.get("gold_label")}
    shared = sorted(set(ids_c3) & set(ids_ctrl))
    if not shared:
        return {"n": 0.0}

    gold = [ids_c3[i]["gold_label"] for i in shared]
    pred_c3 = [ids_c3[i].get("pred_label", "") for i in shared]
    pred_ctrl = [ids_ctrl[i].get("pred_label", "") for i in shared]

    def _block(pred: Sequence[str]) -> Dict[str, float]:
        f1 = per_class_f1(gold, pred)
        return {"macro_f1": macro_f1(gold, pred), "nei_f1": f1["NEI"],
                "support_f1": f1["SUPPORT"], "refute_f1": f1["REFUTE"]}

    c3 = _block(pred_c3)
    ctrl = _block(pred_ctrl)
    return {
        "n": float(len(shared)),
        "c3_macro_f1": c3["macro_f1"], "control_macro_f1": ctrl["macro_f1"],
        "macro_f1_inflation": c3["macro_f1"] - ctrl["macro_f1"],
        "c3_nei_f1": c3["nei_f1"], "control_nei_f1": ctrl["nei_f1"],
        "nei_f1_inflation": c3["nei_f1"] - ctrl["nei_f1"],
        "c3_support_f1": c3["support_f1"], "control_support_f1": ctrl["support_f1"],
        "c3_refute_f1": c3["refute_f1"], "control_refute_f1": ctrl["refute_f1"],
    }


def neutralise_nei_cue(train_claims: Sequence[Dict[str, Any]],
                       retrieval_fn,
                       selector_fn,
                       fallback_sentences: Optional[Sequence[Dict[str, Any]]] = None
                       ) -> List[Dict[str, Any]]:
    """Build the training set for C3'.

    Every NEI example whose evidence is empty is given evidence that retrieval returns but
    that the claim does not match, so the model sees NEI with evidence present. The label
    is unchanged; only the cue is removed. Non-NEI examples are untouched.
    """
    out: List[Dict[str, Any]] = []
    for claim in train_claims:
        item = dict(claim)
        if claim.get("label") == "NEI" and not claim.get("gold_sentences"):
            selected = selector_fn(claim["claim"], retrieval_fn(claim["claim"]))
            if selected:
                item["gold_sentences"] = [0] * len(selected)     # placeholder indices
                item["gold_doc_ids"] = [int(s["doc_id"]) for s in selected]
                item["neutralised_evidence"] = [s["text"] for s in selected]
                item["neutralised"] = True
            elif fallback_sentences:
                item["neutralised_evidence"] = [s["text"] for s in fallback_sentences[:4]]
                item["neutralised"] = True
        out.append(item)
    return out


def summarise(audit_counts: Dict[str, float], probe: Dict[str, Any],
              comparison: Dict[str, float]) -> str:
    """One sentence for the report's abstract, generated from the numbers."""
    if not comparison.get("n"):
        return "The audit ran but produced no comparable claims."
    inflation = comparison["macro_f1_inflation"]
    direction = "inflates" if inflation > 0.02 else ("deflates" if inflation < -0.02 else "does not change")
    return (
        f"The empty-evidence cue covered {audit_counts.get('nei_empty_fraction', float('nan')):.1%} "
        f"of NEI training claims and {direction} macro-F1 by {abs(inflation):.3f} "
        f"(NEI F1 by {abs(comparison['nei_f1_inflation']):.3f})."
    )
