from __future__ import annotations
import numpy as np
import pandas as pd


def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Ledoit-Wolf shrinkage estimator for covariance matrix.
    Shrinks sample covariance toward scaled identity (constant correlation model).
    """
    X = returns.values
    n, p = X.shape
    if n < 2 or p < 2:
        return pd.DataFrame(np.eye(p), index=returns.columns, columns=returns.columns)

    S = np.cov(X, rowvar=False, ddof=1)

    # Shrinkage target: scaled identity
    mu = np.trace(S) / p
    F = mu * np.eye(p)

    # Compute optimal shrinkage intensity (simplified Ledoit-Wolf)
    # Use a fixed reasonable shrinkage intensity for robustness
    gamma_hat = np.linalg.norm(S - F, "fro") ** 2
    if gamma_hat == 0:
        alpha = 1.0
    else:
        # Approximate shrinkage: scales with 1/n for large samples
        alpha = min(max(p / (n * gamma_hat / mu**2), 0.0), 1.0)

    Sigma = alpha * F + (1.0 - alpha) * S
    return pd.DataFrame(Sigma, index=returns.columns, columns=returns.columns)


def correlation_aware_weights(
    pair_returns: pd.DataFrame,
    target_vol: float = 0.008,
    max_weight: float = 0.5,
) -> pd.Series:
    """
    Compute portfolio weights using inverse-vol adjusted by correlation.
    Uses Ledoit-Wolf covariance for stability.
    """
    if pair_returns.shape[1] < 2:
        return pd.Series(1.0, index=pair_returns.columns)

    cov = ledoit_wolf_shrinkage(pair_returns)
    diag = np.diag(cov.values)
    diag = np.maximum(diag, 1e-12)
    inv_vol = 1.0 / np.sqrt(diag)
    w = inv_vol / inv_vol.sum()

    # Scale to target vol
    port_vol = float(np.sqrt(w @ cov.values @ w))
    if port_vol > 1e-12:
        w = w * (target_vol / port_vol)

    # Cap individual weights
    w = np.minimum(w, max_weight)
    total = w.sum()
    if total > 0:
        w = w / total

    return pd.Series(w, index=pair_returns.columns)
