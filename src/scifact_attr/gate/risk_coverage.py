"""Risk–coverage curve and its area.

A single operating point hides the trade-off that the gate actually makes. The curve sorts
claims by confidence, answers the most confident prefix, and plots error among answered
claims against the fraction answered. Two gates can land on the same coverage with very
different risk, and the curve is how that becomes visible.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def risk_coverage_curve(confidences: Sequence[float], correct: Sequence[bool]
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (coverage, risk) arrays, both length n, starting from the most confident claim.

    coverage[i] = (i + 1) / n, risk[i] = 1 - accuracy over the first i + 1 claims.
    """
    conf = np.asarray(list(confidences), dtype=np.float64)
    acc = np.asarray(list(correct), dtype=np.float64)
    if conf.size != acc.size:
        raise ValueError("confidences and correct must have the same length")
    if conf.size == 0:
        return np.zeros(0), np.zeros(0)

    order = np.argsort(-conf, kind="stable")
    acc_sorted = acc[order]
    n = acc_sorted.size
    coverage = np.arange(1, n + 1, dtype=np.float64) / n
    risk = 1.0 - np.cumsum(acc_sorted) / np.arange(1, n + 1, dtype=np.float64)
    return coverage, risk


def aurc(coverage: np.ndarray, risk: np.ndarray) -> float:
    """Area under the risk–coverage curve, trapezoidal. Lower is better.

    Computed by hand rather than via np.trapz/np.trapezoid, whose names differ between
    numpy 1.x and 2.x; a project that silently changes metric because of a numpy upgrade is
    not reproducible.
    """
    if coverage.size < 2:
        return float("nan")
    return float(np.sum(0.5 * (risk[1:] + risk[:-1]) * np.diff(coverage)))


def coverage_at_risk(risk: np.ndarray, coverage: np.ndarray, target_risk: float) -> float:
    """Highest coverage whose risk stays at or below the target."""
    valid = np.where(risk <= target_risk)[0]
    return float(coverage[valid[-1]]) if valid.size else 0.0


def summarise_gate(confidences: Sequence[float], correct: Sequence[bool],
                   target_risk: float = 0.10) -> Dict[str, float]:
    coverage, risk = risk_coverage_curve(confidences, correct)
    return {
        "aurc": aurc(coverage, risk),
        "coverage_at_target_risk": coverage_at_risk(risk, coverage, target_risk),
        "min_risk": float(risk.min()) if risk.size else float("nan"),
    }
