"""Multiple-hypothesis corrections."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def benjamini_hochberg(pvals: Sequence[float], q: float = 0.10) -> list[bool]:
    """Benjamini-Hochberg step-up procedure controlling the false discovery rate at ``q``.

    Returns a boolean mask (same order as ``pvals``) of rejected nulls. NaN
    p-values are never rejected but do count towards the number of tests ``m``.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return []
    p_sorted = np.where(np.isnan(p), np.inf, p)
    order = np.argsort(p_sorted, kind="stable")
    ranked = p_sorted[order]
    passed = ranked <= q * np.arange(1, m + 1) / m
    if not passed.any():
        return [False] * m
    k = int(np.max(np.nonzero(passed)[0]))
    reject = np.zeros(m, dtype=bool)
    reject[order[: k + 1]] = True
    return [bool(v) for v in reject]
