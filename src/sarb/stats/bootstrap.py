from __future__ import annotations
import numpy as np
import pandas as pd

def bootstrap_mean_ci(daily_returns: pd.Series, n_boot: int = 5000, alpha: float = 0.05, seed: int = 42):
    r = daily_returns.dropna().values
    if len(r) == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        samp = rng.choice(r, size=len(r), replace=True)
        means.append(samp.mean())
    means = np.array(means)
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return {"mean": float(r.mean()), "ci_low": float(lo), "ci_high": float(hi)}

def bootstrap_sharpe_ci(daily_returns: pd.Series, n_boot: int = 5000, alpha: float = 0.05, seed: int = 42, ann_factor: int = 252):
    r = daily_returns.dropna().values
    if len(r) == 0:
        return {"sharpe": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    sharpes = []
    for _ in range(n_boot):
        samp = rng.choice(r, size=len(r), replace=True)
        mu = samp.mean()
        sd = samp.std(ddof=0)
        sh = 0.0 if sd == 0 else (mu / sd) * np.sqrt(ann_factor)
        sharpes.append(sh)
    sharpes = np.array(sharpes)
    lo = np.quantile(sharpes, alpha / 2)
    hi = np.quantile(sharpes, 1 - alpha / 2)
    base_sd = r.std(ddof=0)
    base_sh = 0.0 if base_sd == 0 else (r.mean() / base_sd) * np.sqrt(ann_factor)
    return {"sharpe": float(base_sh), "ci_low": float(lo), "ci_high": float(hi)}
