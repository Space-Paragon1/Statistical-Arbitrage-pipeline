"""Bootstrap confidence intervals for strategy returns.

Daily strategy returns are autocorrelated and volatility-clustered (positions
persist for weeks), so the i.i.d. bootstrap understates uncertainty. The default
here is the stationary block bootstrap of Politis & Romano (1994), which resamples
blocks of geometrically distributed length and preserves short-range dependence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def stationary_bootstrap_indices(
    n: int, n_boot: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """``(n_boot, n)`` index matrix for the stationary bootstrap (circular wrap)."""
    if mean_block <= 1:
        return rng.integers(0, n, size=(n_boot, n))
    p = 1.0 / mean_block
    starts = rng.integers(0, n, size=(n_boot, n))
    new_block = rng.random((n_boot, n)) < p
    new_block[:, 0] = True
    # Position within the current block: reset to 0 at each new block start.
    pos = np.arange(n)[None, :].repeat(n_boot, axis=0)
    last_start = np.where(new_block, pos, 0)
    np.maximum.accumulate(last_start, axis=1, out=last_start)
    offset = pos - last_start
    block_start = np.take_along_axis(starts, last_start, axis=1)
    return (block_start + offset) % n


def _resample_stat(
    r: np.ndarray,
    stat_fn,
    n_boot: int,
    seed: int,
    method: str,
    block_size: float | None,
    chunk: int = 1000,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(r)
    if method == "iid":
        mean_block = 1.0
    elif method == "stationary":
        mean_block = block_size if block_size is not None else max(1.0, n ** (1.0 / 3.0))
    else:
        raise ValueError(f"Unknown bootstrap method {method!r}; use 'stationary' or 'iid'.")
    out = []
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        idx = stationary_bootstrap_indices(n, b, mean_block, rng)
        out.append(stat_fn(r[idx]))
    return np.concatenate(out)


def _sharpe_rows(samples: np.ndarray, ann_factor: int) -> np.ndarray:
    mu = samples.mean(axis=1)
    sd = samples.std(axis=1, ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sh = np.where(sd > 0, mu / sd * np.sqrt(ann_factor), 0.0)
    return sh


def bootstrap_mean_ci(
    daily_returns: pd.Series,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
    method: str = "stationary",
    block_size: float | None = None,
) -> dict[str, float]:
    """Percentile CI for the mean daily return."""
    r = daily_returns.dropna().to_numpy(float)
    if len(r) == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    means = _resample_stat(r, lambda s: s.mean(axis=1), n_boot, seed, method, block_size)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(r.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def bootstrap_sharpe_ci(
    daily_returns: pd.Series,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
    ann_factor: int = 252,
    method: str = "stationary",
    block_size: float | None = None,
) -> dict[str, float]:
    """Percentile CI for the annualised Sharpe ratio."""
    r = daily_returns.dropna().to_numpy(float)
    if len(r) == 0:
        return {"sharpe": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    sharpes = _resample_stat(
        r, lambda s: _sharpe_rows(s, ann_factor), n_boot, seed, method, block_size
    )
    lo, hi = np.quantile(sharpes, [alpha / 2, 1 - alpha / 2])
    base = float(_sharpe_rows(r[None, :], ann_factor)[0])
    return {"sharpe": base, "ci_low": float(lo), "ci_high": float(hi)}
