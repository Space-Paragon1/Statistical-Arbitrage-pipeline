"""Universe scanning: prefilter -> cointegration test -> FDR control -> rank.

Selection protocol (no test-period data is touched):

1. Prefilter candidate pairs on TRAIN returns (correlation or OPTICS clustering).
2. Engle-Granger cointegration test on TRAIN prices (MacKinnon critical values).
3. Benjamini-Hochberg FDR control across *all* tested pairs.
4. Survivors are ranked by VALIDATION Sharpe (hedge ratio frozen from TRAIN).
5. Each survivor's validation Sharpe is deflated for the number of pairs tried.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sarb.backtest.engine import backtest_pairs
from sarb.features.spread import compute_spread, rolling_zscore
from sarb.metrics.performance import sharpe
from sarb.stats.cointegration import engle_granger_test, estimate_half_life
from sarb.stats.multiple_testing import benjamini_hochberg
from sarb.stats.sharpe_inference import deflated_sharpe_ratio, per_period_sharpe
from sarb.strategy.pairs import generate_spread_positions

MIN_TRAIN_OBS = 300
MIN_VAL_OBS = 100


@dataclass
class PairResult:
    y: str
    x: str
    beta: float
    alpha: float
    corr: float
    adf_p: float  # Engle-Granger cointegration p-value on TRAIN (MacKinnon)
    half_life: float
    val_sharpe: float
    eg_stat: float = float("nan")
    val_dsr: float = float("nan")  # deflated Sharpe of the validation run
    fdr_pass: bool = False


@dataclass
class ScanReport:
    """Everything a scan did, so selection is auditable after the fact."""

    selected: list[PairResult]
    tested: list[PairResult]
    n_candidates: int
    n_fdr_pass: int


def _pair_corr(train_px: pd.DataFrame, y: str, x: str) -> float:
    r = train_px[[y, x]].pct_change().dropna()
    if len(r) < 50:
        return float("nan")
    return float(r[y].corr(r[x]))


def _val_returns(
    prices: pd.DataFrame,
    y: str,
    x: str,
    train_idx: pd.Index,
    val_idx: pd.Index,
    alpha: float,
    beta: float,
    lookback_z: int,
    entry_z: float,
    exit_z: float,
    cost_bps: float,
    leverage: float,
) -> pd.Series:
    """Net returns over VAL with parameters frozen from TRAIN.

    Z-scores are built over TRAIN+VAL so the rolling window is warm at the start
    of VAL; positions are lagged one bar, so nothing from the future leaks in.
    """
    tv_idx = train_idx.union(val_idx)
    px_tv = prices.loc[tv_idx, [y, x]].dropna()
    spread_tv = compute_spread(px_tv[y], px_tv[x], alpha, beta)
    z_tv = rolling_zscore(spread_tv, lookback_z)
    pos_tv = generate_spread_positions(z_tv, entry_z, exit_z)
    bt = backtest_pairs(px_tv, y, x, alpha, beta, pos_tv, cost_bps, 0.0, leverage)
    return bt["ret_net"].loc[val_idx.intersection(bt.index)]


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
    corr: float | None = None,
) -> PairResult | None:
    """Cointegration diagnostics on TRAIN and a frozen-parameter backtest on VAL."""
    out = _evaluate(
        prices,
        y,
        x,
        train_idx,
        val_idx,
        lookback_z,
        entry_z,
        exit_z,
        fee_bps + slippage_bps,
        leverage,
        corr,
    )
    return None if out is None else out[0]


def _evaluate(
    prices: pd.DataFrame,
    y: str,
    x: str,
    train_idx: pd.Index,
    val_idx: pd.Index,
    lookback_z: int,
    entry_z: float,
    exit_z: float,
    cost_bps: float,
    leverage: float,
    corr: float | None,
) -> tuple[PairResult, pd.Series] | None:
    train_px = prices.loc[train_idx, [y, x]].dropna()
    val_px = prices.loc[val_idx, [y, x]].dropna()
    if len(train_px) < MIN_TRAIN_OBS or len(val_px) < MIN_VAL_OBS:
        return None

    eg = engle_granger_test(train_px[y], train_px[x])
    if not np.isfinite(eg.beta):
        return None
    spread_train = compute_spread(train_px[y], train_px[x], eg.alpha, eg.beta)
    hl = estimate_half_life(spread_train)

    r_val = _val_returns(
        prices,
        y,
        x,
        train_idx,
        val_idx,
        eg.alpha,
        eg.beta,
        lookback_z,
        entry_z,
        exit_z,
        cost_bps,
        leverage,
    )
    if len(r_val) < 50:
        return None

    res = PairResult(
        y=y,
        x=x,
        beta=eg.beta,
        alpha=eg.alpha,
        corr=_pair_corr(train_px, y, x) if corr is None else corr,
        adf_p=eg.pvalue,
        half_life=hl,
        val_sharpe=sharpe(r_val),
        eg_stat=eg.stat,
    )
    return res, r_val


def candidate_pairs(
    prices: pd.DataFrame,
    tickers: list[str],
    train_idx: pd.Index,
    corr_threshold: float,
    max_pairs: int,
    prefilter_method: str = "correlation",
) -> list[tuple[str, str, float]]:
    train_px = prices.loc[train_idx, tickers].dropna(axis=1, how="any")
    tickers_ok = list(train_px.columns)

    if prefilter_method == "ml":
        from sarb.research.ml_select import ml_prefilter_pairs

        ml_pairs = ml_prefilter_pairs(
            prices=prices,
            tickers=tickers_ok,
            train_idx=train_idx,
            method="optics",
            max_pairs=max_pairs,
        )
        return [(y, x, _pair_corr(train_px, y, x)) for y, x in ml_pairs]
    if prefilter_method != "correlation":
        raise ValueError(f"Unknown prefilter_method {prefilter_method!r}.")

    corr = train_px.pct_change().dropna().corr().to_numpy()
    cands = []
    for (i, y), (j, x) in itertools.combinations(enumerate(tickers_ok), 2):
        c = float(corr[i, j])
        if np.isfinite(c) and abs(c) >= corr_threshold:
            cands.append((y, x, c))
    cands.sort(key=lambda t: abs(t[2]), reverse=True)
    return cands[:max_pairs]


def scan_pairs_report(
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
    fallback_to_lowest_p: bool = False,
    max_half_life: float | None = None,
) -> ScanReport:
    """Run the full selection protocol and return an auditable :class:`ScanReport`.

    ``fallback_to_lowest_p``: if no pair survives FDR control, the strict default
    is to select nothing (and therefore not trade). Setting this to True restores
    the old behaviour of trading the lowest p-values anyway, which voids the FDR
    guarantee; it is kept only for comparison.
    """
    cands = candidate_pairs(prices, tickers, train_idx, corr_threshold, max_pairs, prefilter_method)

    tested: list[PairResult] = []
    val_rets: dict[tuple[str, str], pd.Series] = {}
    for y, x, c in cands:
        out = _evaluate(
            prices,
            y,
            x,
            train_idx,
            val_idx,
            lookback_z,
            entry_z,
            exit_z,
            fee_bps + slippage_bps,
            leverage,
            c,
        )
        if out is not None and np.isfinite(out[0].adf_p):
            tested.append(out[0])
            val_rets[(y, x)] = out[1]

    if not tested:
        return ScanReport([], [], len(cands), 0)

    mask = benjamini_hochberg([r.adf_p for r in tested], q=fdr_q)
    for r, keep in zip(tested, mask):
        r.fdr_pass = keep

    pool = [r for r in tested if r.fdr_pass]
    n_pass = len(pool)
    if max_half_life is not None:
        pool = [r for r in pool if np.isfinite(r.half_life) and r.half_life <= max_half_life]
    if not pool and fallback_to_lowest_p:
        pool = sorted(tested, key=lambda r: r.adf_p)[:top_k]

    _attach_dsr(tested, pool, val_rets)

    pool.sort(key=lambda r: r.val_sharpe, reverse=True)
    return ScanReport(pool[:top_k], tested, len(cands), n_pass)


def _attach_dsr(
    tested: list[PairResult],
    pool: list[PairResult],
    val_rets: dict[tuple[str, str], pd.Series],
) -> None:
    """Deflate each pooled pair's validation Sharpe by the number of pairs tested.

    The null distribution of the *best* Sharpe among ``len(tested)`` trials uses
    the cross-sectional variance of all trial Sharpes (Bailey & Lopez de Prado).
    """
    sr_trials = np.array([per_period_sharpe(val_rets[(r.y, r.x)]) for r in tested])
    sr_trials = sr_trials[np.isfinite(sr_trials)]
    sr_var = float(np.var(sr_trials, ddof=1)) if len(sr_trials) > 1 else 0.0
    for r in pool:
        r.val_dsr = deflated_sharpe_ratio(val_rets[(r.y, r.x)], len(tested), sr_var)


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
    fallback_to_lowest_p: bool = False,
    max_half_life: float | None = None,
) -> list[PairResult]:
    """Selected pairs only. See :func:`scan_pairs_report` for the full audit trail."""
    return scan_pairs_report(
        prices,
        tickers,
        train_idx,
        val_idx,
        lookback_z,
        entry_z,
        exit_z,
        fee_bps,
        slippage_bps,
        leverage,
        corr_threshold,
        max_pairs,
        fdr_q,
        top_k,
        prefilter_method,
        fallback_to_lowest_p,
        max_half_life,
    ).selected
