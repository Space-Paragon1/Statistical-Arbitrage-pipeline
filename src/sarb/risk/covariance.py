"""Covariance estimation and correlation-aware allocation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ledoit_wolf(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Ledoit & Wolf (2004) optimal linear shrinkage towards a scaled identity.

    ``Sigma = s * mu * I + (1 - s) * S`` where ``S`` is the (1/n) sample covariance,
    ``mu = tr(S)/p`` and the intensity ``s`` minimises expected Frobenius loss:

        delta^2 = ||S - mu I||^2 / p
        beta^2  = min(delta^2, (1/n^2) * sum_k ||x_k x_k' - S||^2 / p)
        s       = beta^2 / delta^2

    Matches ``sklearn.covariance.LedoitWolf`` (verified in the test-suite) without
    requiring scikit-learn. Returns ``(Sigma, s)``.
    """
    cols = returns.columns
    X = returns.to_numpy(dtype=float)
    n, p = X.shape
    if n < 2 or p < 1:
        return pd.DataFrame(np.eye(p), index=cols, columns=cols), 1.0
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / n
    mu = np.trace(S) / p
    delta2 = np.sum((S - mu * np.eye(p)) ** 2) / p
    if delta2 <= 0:
        return pd.DataFrame(S, index=cols, columns=cols), 0.0
    X2 = Xc**2
    # sum_k ||x_k x_k' - S||_F^2 = sum_k ||x_k||^4 - n ||S||_F^2
    beta2_bar = (np.sum(X2.T @ X2) / n - np.sum(S**2)) / (n * p)
    beta2 = min(max(beta2_bar, 0.0), delta2)
    shrink = beta2 / delta2
    sigma = shrink * mu * np.eye(p) + (1.0 - shrink) * S
    return pd.DataFrame(sigma, index=cols, columns=cols), float(shrink)


def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrunk covariance matrix (see :func:`ledoit_wolf`)."""
    return ledoit_wolf(returns)[0]


def _cap_and_redistribute(w: np.ndarray, cap: float, iters: int = 100) -> np.ndarray:
    """Project long-only weights summing to 1 onto ``w_i <= cap`` by water-filling."""
    w = w.copy()
    for _ in range(iters):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over & (w < cap)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum() if w[free].sum() > 0 else excess / free.sum()
    return w


def correlation_aware_weights(
    pair_returns: pd.DataFrame,
    max_weight: float = 0.5,
) -> pd.Series:
    """Long-only minimum-variance allocation on the Ledoit-Wolf covariance.

    Weights sum to 1 and respect ``max_weight`` (when feasible). Highly correlated
    pairs share risk budget instead of each receiving a full slot, which is what
    makes this *correlation*-aware rather than plain inverse-volatility.

    ``pair_returns`` must be **historical** returns (estimation window), never the
    returns of the period being traded.
    """
    cols = pair_returns.columns
    k = len(cols)
    if k == 0:
        return pd.Series(dtype=float)
    if k == 1:
        return pd.Series(1.0, index=cols)
    cov = ledoit_wolf_shrinkage(pair_returns.fillna(0.0)).to_numpy()
    cov = cov + np.eye(k) * 1e-12 * max(np.trace(cov) / k, 1e-12)

    # Long-only min variance via iterative active-set: drop assets with w < 0.
    active = np.ones(k, dtype=bool)
    w = np.zeros(k)
    for _ in range(k):
        sub = cov[np.ix_(active, active)]
        raw = np.linalg.solve(sub, np.ones(active.sum()))
        w_sub = raw / raw.sum()
        if (w_sub >= 0).all():
            w[:] = 0.0
            w[active] = w_sub
            break
        idx = np.flatnonzero(active)
        active[idx[w_sub < 0]] = False
    else:  # pragma: no cover - defensive
        w = np.full(k, 1.0 / k)

    w = _cap_and_redistribute(w, max_weight) if max_weight * k >= 1.0 else np.full(k, 1.0 / k)
    return pd.Series(w, index=cols)
