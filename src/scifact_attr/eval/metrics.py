"""Label-level metrics.

Every function takes plain Python or numpy inputs and returns plain floats, so the metric
layer can be unit-tested without a model, a GPU or a dataset. That matters because a
metric that is wrong is indistinguishable from a result that is interesting.

class indices: SUPPORT = 0, REFUTE = 1, NEI = 2, ABSTAIN = 3 (never a gold class).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .. import ABSTAIN, LABELS

CLASS_INDEX = {"SUPPORT": 0, "REFUTE": 1, "NEI": 2, ABSTAIN: 3}


def to_index(labels: Iterable[str]) -> np.ndarray:
    return np.asarray([CLASS_INDEX[str(l)] for l in labels], dtype=np.int64)


def accuracy(gold: Sequence[str], pred: Sequence[str]) -> float:
    gold_arr = np.asarray(list(gold))
    pred_arr = np.asarray(list(pred))
    if gold_arr.size != pred_arr.size:
        raise ValueError("gold and pred must have the same length")
    return float((gold_arr == pred_arr).mean()) if gold_arr.size else float("nan")


def confusion(gold: Sequence[str], pred: Sequence[str],
              labels: Sequence[str] = LABELS) -> np.ndarray:
    """Rows are gold classes, columns are predictions."""
    idx = {l: i for i, l in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for g, p in zip(gold, pred):
        if p in idx and g in idx:
            matrix[idx[g], idx[p]] += 1
    return matrix


def per_class_f1(gold: Sequence[str], pred: Sequence[str],
                 labels: Sequence[str] = LABELS) -> Dict[str, float]:
    """F1 per class. An absent class scores 0.0 rather than being skipped."""
    matrix = confusion(gold, pred, labels)
    out: Dict[str, float] = {}
    for i, label in enumerate(labels):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = sum(g == label for g in gold) - tp
        denom = 2 * tp + fp + fn
        out[label] = float(2 * tp / denom) if denom else 0.0
    return out


def macro_f1(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] = LABELS) -> float:
    scores = per_class_f1(gold, pred, labels)
    return float(np.mean([scores[l] for l in labels]))


def selective_accuracy(gold: Sequence[str], pred: Sequence[str]) -> Tuple[float, float]:
    """Accuracy over answered claims, and the coverage that produced it."""
    pairs = [(g, p) for g, p in zip(gold, pred) if p in LABELS]
    if not pairs:
        return float("nan"), 0.0
    correct = sum(1 for g, p in pairs if g == p)
    return correct / len(pairs), len(pairs) / len(list(gold))


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool],
                               n_bins: int = 10) -> float:
    """Equal-width binning: weighted mean of |mean confidence - mean accuracy| per bin."""
    conf = np.clip(np.asarray(confidences, dtype=np.float64), 0.0, 1.0)
    acc = np.asarray(correct, dtype=np.float64)
    if conf.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-hand bin inclusion for the top edge; a confidence of exactly 1.0 must land somewhere
    bins = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        total += abs(conf[mask].mean() - acc[mask].mean()) * mask.mean()
    return float(total)


def brier_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    conf = np.asarray(confidences, dtype=np.float64)
    acc = np.asarray(correct, dtype=np.float64)
    return float(np.mean((conf - acc) ** 2)) if conf.size else float("nan")


def summarise(gold: Sequence[str], pred: Sequence[str],
              confidences: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """The row that appears in every condition table."""
    out = {
        "n": float(len(gold)),
        "accuracy": accuracy(gold, pred),
        "macro_f1": macro_f1(gold, pred),
        "nei_f1": per_class_f1(gold, pred)["NEI"],
    }
    sel_acc, coverage = selective_accuracy(gold, pred)
    out["coverage"] = coverage
    out["selective_accuracy"] = sel_acc
    if confidences is not None:
        out["ece"] = expected_calibration_error(
            confidences, [g == p for g, p in zip(gold, pred)])
    return out
