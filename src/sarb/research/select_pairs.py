from __future__ import annotations
from dataclasses import dataclass
import itertools
import numpy as np
import pandas as pd

from sarb.features.spread import fit_hedge_ratio, compute_spread, rolling_zscore
from sarb.stats.cointegration import engle_granger_adf_pvalue, estimate_half_life
from sarb.strategy.pairs import generate_spread_positions
from sarb.backtest.engine import backtest_pairs
from sarb.metrics.performance import sharpe
from sarb.stats.multiple_testing import benjamini_hochberg


@dataclass
class PairResult:
    y: str
    x: str
    beta: float
    alpha: float
    corr: float
    adf_p: float
    half_life: float
    val_sharpe: float


def _pair_corr(train_px: pd.DataFrame, y: str, x: str) -> float:
    r = train_px[[y, x]].pct_change().dropna()
    if len(r) < 50:
        return float("nan")
    return float(r[y].corr(r[x]))


def evaluate_pair_on_val(
    prices: pd.DataFrame,
    y: str,
    x: str,
    train_idx: pd.Index,
    val_idx: pd.Index,
    lookback_z: int,
    entry_z: float,
    exit_z: float,
    fee_bps: float,
    slippage_bps: float,
    leverage: float = 1.0,
) -> PairResult | None:
    train_px = prices.loc[train_idx, [y, x]].dropna()
    val_px = prices.loc[val_idx, [y, x]].dropna()

    if len(train_px) < 300 or len(val_px) < 100:
        return None

    # Fit hedge ratio on TRAIN only
    alpha, beta = fit_hedge_ratio(train_px[y], train_px[x])

    # Spread diagnostics on TRAIN only
    spread_train = compute_spread(train_px[y], train_px[x], alpha, beta)
    adf_p = engle_granger_adf_pvalue(spread_train)
    hl = estimate_half_life(spread_train)
    corr = _pair_corr(train_px, y, x)

    # Build z-score on TRAIN+VAL (allowed), but execution is lookahead-safe via shift in positions
    tv_idx = train_idx.union(val_idx)
    px_tv = prices.loc[tv_idx, [y, x]].dropna()
    spread_tv = compute_spread(px_tv[y], px_tv[x], alpha, beta)
    z_tv = rolling_zscore(spread_tv, lookback_z)
    pos_tv = generate_spread_positions(z_tv, entry_z, exit_z)

    bt_tv = backtest_pairs(
        prices=px_tv,
        y=y,
        x=x,
        alpha=alpha,
        beta=beta,
        spread_pos=pos_tv,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        leverage=leverage,
    )

    bt_val = bt_tv.loc[val_idx.intersection(bt_tv.index)]
    if len(bt_val) < 50:
        return None

    val_sh = sharpe(bt_val["ret_net"])

    return PairResult(
        y=y, x=x, beta=float(beta), alpha=float(alpha),
        corr=corr, adf_p=float(adf_p), half_life=float(hl),
        val_sharpe=float(val_sh),
    )


def scan_pairs(
    prices: pd.DataFrame,
    tickers: list[str],
    train_idx: pd.Index,
    val_idx: pd.Index,
    lookback_z: int,
    entry_z: float,
    exit_z: float,
    fee_bps: float,
    slippage_bps: float,
    leverage: float = 1.0,
    corr_threshold: float = 0.6,
    max_pairs: int = 300,
    fdr_q: float = 0.10,
    top_k: int = 10,
    prefilter_method: str = "correlation",
) -> list[PairResult]:
    """
    Pipeline:
    1) Prefilter pairs (correlation or ML clustering) on TRAIN returns
    2) For candidates, compute ADF p-value (TRAIN) and val Sharpe (VAL)
    3) Apply BH-FDR on ADF p-values
    4) Rank by validation Sharpe and return top_k

    prefilter_method: "correlation" (default) or "ml" (OPTICS clustering)
    """
    train_px = prices.loc[train_idx, tickers].dropna(axis=1, how="any")
    tickers_ok = list(train_px.columns)

    if prefilter_method == "ml":
        from sarb.research.ml_select import ml_prefilter_pairs
        ml_pairs = ml_prefilter_pairs(
            prices=prices, tickers=tickers_ok, train_idx=train_idx,
            method="optics", max_pairs=max_pairs,
        )
        candidates = [(y, x, 0.0) for y, x in ml_pairs]
    else:
        # Prefilter by correlation on TRAIN
        candidates = []
        for y, x in itertools.combinations(tickers_ok, 2):
            c = _pair_corr(train_px, y, x)
            if np.isfinite(c) and abs(c) >= corr_threshold:
                candidates.append((y, x, c))

        # keep most correlated first, cap
        candidates.sort(key=lambda t: abs(t[2]), reverse=True)
        candidates = candidates[:max_pairs]

    results: list[PairResult] = []
    for y, x, _c in candidates:
        res = evaluate_pair_on_val(
            prices=prices, y=y, x=x,
            train_idx=train_idx, val_idx=val_idx,
            lookback_z=lookback_z, entry_z=entry_z, exit_z=exit_z,
            fee_bps=fee_bps, slippage_bps=slippage_bps,
            leverage=leverage,
        )
        if res is not None and np.isfinite(res.adf_p):
            results.append(res)

    if not results:
        return []

    # Multiple testing control on ADF p-values
    pvals = [r.adf_p for r in results]
    keep_mask = benjamini_hochberg(pvals, q=fdr_q)

    filtered = [r for r, keep in zip(results, keep_mask) if keep]

    # If FDR keeps none, fall back to strongest stationarity evidence (smallest p-values)
    if not filtered:
        filtered = sorted(results, key=lambda r: r.adf_p)[:top_k]

    # Rank by validation Sharpe (research-style selection)
    filtered.sort(key=lambda r: r.val_sharpe, reverse=True)
    return filtered[:top_k]
