"""Bootstrap confidence intervals and McNemar's test.

Both exist because the test split has 300 claims. A single accuracy estimate there carries
an interval of roughly plus or minus six points, so a two-point difference between two
conditions is not a difference. Reporting intervals is what stops that from being
accidentally claimed.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np


def bootstrap_ci(values: Sequence[float], statistic: Callable[[np.ndarray], np.ndarray] = np.mean,
                 n_resamples: int = 10000, confidence: float = 0.95,
                 seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap over claims (the sampling unit is a claim, not a token)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = arr.size
    indices = rng.integers(0, n, size=(n_resamples, n))
    resampled = arr[indices]
    try:
        stats = np.asarray(statistic(resampled, axis=1), dtype=np.float64)
    except TypeError:
        # a statistic that does not accept an axis: apply it row by row
        stats = np.asarray([statistic(row) for row in resampled], dtype=np.float64)
    alpha = (1.0 - confidence) / 2.0
    return (float(np.percentile(stats, 100 * alpha)),
            float(np.percentile(stats, 100 * (1 - alpha))))


DEFAULT_RESAMPLES = 10000


def paired_delta_ci(pred_a: Sequence[str], pred_b: Sequence[str], gold: Sequence[str],
                    n_resamples: int = DEFAULT_RESAMPLES, confidence: float = 0.95,
                    seed: int = 0) -> Dict[str, float]:
    """Interval for acc(b) - acc(a) with the same claims resampled in both conditions.

    Resampling claims (not conditions independently) preserves the pairing, which is the
    only way the interval reflects the difference rather than the two level estimates.
    """
    gold_arr = np.asarray(list(gold))
    a = (np.asarray(list(pred_a)) == gold_arr).astype(np.float64)
    b = (np.asarray(list(pred_b)) == gold_arr).astype(np.float64)
    delta = b - a

    rng = np.random.default_rng(seed)
    n = delta.size
    indices = rng.integers(0, n, size=(n_resamples, n))
    boot = delta[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "delta": float(delta.mean()),
        "ci_low": float(np.percentile(boot, 100 * alpha)),
        "ci_high": float(np.percentile(boot, 100 * (1 - alpha))),
        "n": float(n),
    }


def mcnemar(pred_a: Sequence[str], pred_b: Sequence[str], gold: Sequence[str]
            ) -> Dict[str, float]:
    """Exact McNemar test on discordant pairs (binomial, two-sided).

    Uses the exact test rather than the chi-square approximation because the discordant
    count on 300 claims is frequently small, where the approximation is unreliable.
    """
    gold_arr = np.asarray(list(gold))
    a_right = np.asarray(list(pred_a)) == gold_arr
    b_right = np.asarray(list(pred_b)) == gold_arr

    only_a = int(np.sum(a_right & ~b_right))   # a correct, b wrong
    only_b = int(np.sum(~a_right & b_right))   # b correct, a wrong
    n_discordant = only_a + only_b

    if n_discordant == 0:
        return {"b": float(only_a), "c": float(only_b), "p_value": 1.0, "n_discordant": 0.0}

    from scipy.stats import binomtest

    result = binomtest(min(only_a, only_b), n_discordant, 0.5, alternative="two-sided")
    return {"b": float(only_a), "c": float(only_b),
            "p_value": float(result.pvalue), "n_discordant": float(n_discordant)}
