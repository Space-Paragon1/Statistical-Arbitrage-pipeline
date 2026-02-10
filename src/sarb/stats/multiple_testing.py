from __future__ import annotations
import numpy as np

def benjamini_hochberg(pvals: list[float], q: float = 0.10) -> list[bool]:
    """
    Benjamini–Hochberg procedure (FDR control).
    Returns a boolean list indicating which hypotheses are rejected.
    """
    p = np.array(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return []

    order = np.argsort(p)
    ranked = p[order]

    thresh = q * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh

    if not passed.any():
        return [False] * m

    k = np.max(np.where(passed)[0])  # last index that passes
    cutoff = ranked[k]

    return [bool(pi <= cutoff) for pi in p]
