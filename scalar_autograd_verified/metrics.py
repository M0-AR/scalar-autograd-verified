"""Evaluation metrics: accuracy, baselines, AUC, and a binomial z-test."""

from __future__ import annotations

import math

__all__ = ["accuracy", "majority_baseline", "roc_auc", "binomial_z_test"]


def accuracy(predictions, targets) -> float:
    """Predictions are in (-1, 1); targets are exactly +-1."""
    hits = sum(1 for p, t in zip(predictions, targets) if (p > 0) == (t > 0))
    return hits / len(targets)


def majority_baseline(targets) -> float:
    n_pos = sum(1 for t in targets if t > 0)
    return max(n_pos, len(targets) - n_pos) / len(targets)


def roc_auc(scores, targets) -> float:
    """Rank-based (Mann-Whitney) AUC. Ties contribute 0.5."""
    pos = [s for s, t in zip(scores, targets) if t > 0]
    neg = [s for s, t in zip(scores, targets) if t <= 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def binomial_z_test(n_hits: int, n: int, p0: float = 0.5):
    """Two-sided normal-approximation test that hit rate equals ``p0``.

    Returns (z, p_value). Uses the exact binomial standard error
    sqrt(p0 (1-p0) / n); adequate for n >= ~30.
    """
    if n == 0:
        return float("nan"), float("nan")
    phat = n_hits / n
    se = math.sqrt(p0 * (1 - p0) / n)
    if se == 0:
        return float("nan"), float("nan")
    z = (phat - p0) / se
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return z, p_value
