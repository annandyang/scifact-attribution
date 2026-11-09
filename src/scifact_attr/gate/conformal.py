"""Split conformal label sets and the singleton rule for abstention."""

from __future__ import annotations

from decimal import Decimal, ROUND_CEILING
from hashlib import sha256
from math import isfinite
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .. import ABSTAIN, LABELS

PROBABILITY_FIELDS = dict(zip(LABELS, ("p_support", "p_refute", "p_nei")))


def partition_claim_ids(claim_ids: Iterable[str]) -> tuple[set[str], set[str]]:
    """Freeze the split by ID, independently of input order and labels."""
    ordered = sorted({str(cid) for cid in claim_ids},
                     key=lambda cid: sha256(f"scifact-dev-v1:{cid}".encode()).digest())
    halfway = len(ordered) // 2
    return set(ordered[:halfway]), set(ordered[halfway:])


def conformal_rank(n: int, alpha: float) -> int:
    if n <= 0:
        raise ValueError("calibration set is empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    rank = (n + 1) * (1 - Decimal(str(alpha)))
    return int(rank.to_integral_value(rounding=ROUND_CEILING))


def conformal_quantile_level(n: int, alpha: float) -> float:
    return min(1.0, conformal_rank(n, alpha) / n)


def conformal_threshold(calibration_scores: Sequence[float], alpha: float) -> float:
    """Nonconformity threshold at level alpha. Higher score means less confident."""
    scores = sorted(calibration_scores)
    rank = conformal_rank(len(scores), alpha)
    if any(not 0.0 <= s <= 1.0 for s in scores):
        raise ValueError("calibration scores must be in [0, 1]")
    return 1.0 if rank > len(scores) else float(scores[rank - 1])


def nonconformity(probability: float) -> float:
    return 1.0 - max(0.0, min(1.0, float(probability)))


def label_scores(row: dict) -> dict[str, float]:
    values = {label: row.get(field) for label, field in PROBABILITY_FIELDS.items()}
    if any(not isinstance(p, (int, float)) or not isfinite(p) or not 0 <= p <= 1
           for p in values.values()):
        # Missing probabilities yield a full set and always abstain.
        return dict.fromkeys(LABELS, 0.0)
    if abs(sum(values.values()) - 1.0) > 1e-4:
        raise ValueError("label probabilities must sum to 1")
    return {label: nonconformity(p) for label, p in values.items()}


def gold_score(row: dict) -> float:
    return label_scores(row)[row["gold_label"]]


def gate_prediction(row: dict, threshold: float) -> tuple[str, list[str]]:
    included = [label for label, score in label_scores(row).items() if score <= threshold]
    base_label = row.get("pred_label")
    if row.get("status") == "ok" and included == [base_label]:
        return base_label, included
    return ABSTAIN, included


def decide(scores: Sequence[float], threshold: float) -> List[str]:
    """answer / abstain for each claim, given its nonconformity score."""
    return ["ABSTAIN" if s > threshold else "answer" for s in scores]


def apply_to_predictions(predicted_labels: Sequence[str],
                         predicted_probs: Sequence[float],
                         threshold: float) -> List[str]:
    """Return labels with ABSTAIN substituted where the gate fires."""
    out: List[str] = []
    for label, prob in zip(predicted_labels, predicted_probs):
        out.append(ABSTAIN if nonconformity(prob) > threshold else label)
    return out


def coverage_and_error(predicted: Sequence[str], gold: Sequence[str],
                       answered: Sequence[bool]) -> Dict[str, float]:
    """Empirical coverage and the error rate among retained predictions."""
    answered_arr = np.asarray(list(answered), dtype=bool)
    n = answered_arr.size
    if n == 0:
        return {"coverage": float("nan"), "selective_error": float("nan"), "n": 0.0}
    kept = [(p, g) for p, g, a in zip(predicted, gold, answered_arr) if a]
    coverage = answered_arr.sum() / n
    if not kept:
        return {"coverage": float(coverage), "selective_error": float("nan"), "n": float(n)}
    errors = sum(1 for p, g in kept if p != g) / len(kept)
    return {"coverage": float(coverage), "selective_error": float(errors), "n": float(n)}


def calibrate(calibration_scores: Sequence[float], alpha: float) -> Dict[str, float]:
    """One call that returns everything the report needs for a single alpha."""
    scores = np.asarray(list(calibration_scores), dtype=np.float64)
    return {
        "alpha": float(alpha),
        "n_calibration": float(scores.size),
        "quantile_level": conformal_quantile_level(scores.size, alpha),
        "threshold": conformal_threshold(scores, alpha),
        "rank": conformal_rank(scores.size, alpha),
    }
