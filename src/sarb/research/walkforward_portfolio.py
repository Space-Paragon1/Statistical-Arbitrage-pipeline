"""Quarterly walk-forward multi-pair portfolio.

For every quarter ``(train, val, trade)``:

1. **Select** pairs with :func:`scan_pairs_report` on TRAIN/VAL only.
2. **Fit** each pair's hedge ratio on TRAIN; build z-scores over TRAIN+VAL+TRADE
   (causal rolling windows, positions lagged one bar).
3. **Size** each pair to a volatility target estimated on TRAIN+VAL returns.
4. **Allocate** across pairs (equal weight or Ledoit-Wolf minimum variance), again
   estimated on TRAIN+VAL only.
5. **Risk**: per-pair caps, a daily gross-exposure cap and a drawdown breaker that
   flattens the book for one quarter when the *current* drawdown breaches the limit.

Pair weights are summed into a dated **asset-level** weight matrix and the whole
history is simulated once with :func:`backtest_weights`. Costs are therefore
charged on actual trades, including positions opened or closed at quarter
boundaries and netted across pairs that share a ticker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd

from sarb.backtest.engine import backtest_weights, pair_weights
from sarb.features.spread import compute_spread, fit_hedge_ratio, rolling_zscore
from sarb.portfolio.vol_target import vol_target_scale
from sarb.research.select_pairs import PairResult, scan_pairs_report
from sarb.risk.covariance import correlation_aware_weights
from sarb.risk.limits import (
    RiskLimits,
    apply_position_limits,
    cap_gross_exposure,
    check_drawdown_breaker,
)
from sarb.strategy.pairs import generate_spread_positions


def _fit_hedge(y: pd.Series, x: pd.Series, method: str = "ols") -> tuple[float, float]:
    if method == "kalman":
        from sarb.features.kalman import fit_hedge_ratio_kalman

        return fit_hedge_ratio_kalman(y, x)
    return fit_hedge_ratio(y, x)


@dataclass
class WFConfig:
    # Selection
    top_k: int = 5
    corr_threshold: float = 0.6
    max_pairs: int = 300
    fdr_q: float = 0.10
    prefilter_method: str = "correlation"
    fallback_to_lowest_p: bool = False  # True voids FDR control; kept for comparison
    max_half_life: float | None = None

    # Signal
    z_lookback: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float | None = None
    max_holding: int | None = None
    hedge_method: str = "ols"  # "ols" or "kalman"

    # Costs
    fee_bps: float = 1.0
    slippage_bps: float = 0.5
    short_borrow_cost_bps: float = 0.0  # annualised
    leverage: float = 1.0  # gross per pair before vol scaling

    # Sizing
    use_vol_targeting: bool = True
    target_daily_vol: float = 0.008
    max_pair_scale: float = 3.0

    # Allocation & risk
    use_correlation_weights: bool = False
    risk_limits: RiskLimits | None = None


class PortfolioBacktest(NamedTuple):
    returns: pd.DataFrame  # engine output: ret_net, equity, turnover, costs, exposures...
    meta: pd.DataFrame  # one row per quarter: what was selected and why
    weights: pd.DataFrame  # asset-level weights actually held


def _pair_book(
    prices: pd.DataFrame,
    r: PairResult,
    train_idx: pd.Index,
    sig_idx: pd.Index,
    cfg: WFConfig,
) -> pd.DataFrame | None:
    """Unit-leverage asset weights for one pair over ``sig_idx`` (TRAIN+VAL+TRADE)."""
    px_train = prices.loc[train_idx, [r.y, r.x]].dropna()
    if len(px_train) < 200:
        return None
    alpha, beta = _fit_hedge(px_train[r.y], px_train[r.x], cfg.hedge_method)
    px = prices.loc[sig_idx, [r.y, r.x]].dropna()
    z = rolling_zscore(compute_spread(px[r.y], px[r.x], alpha, beta), cfg.z_lookback)
    pos = generate_spread_positions(z, cfg.entry_z, cfg.exit_z, cfg.stop_z, cfg.max_holding)
    W = pair_weights(px[r.y], px[r.x], pos, beta, cfg.leverage)
    W.columns = [r.y, r.x]
    return W


def walkforward_quarterly_portfolio(
    prices: pd.DataFrame,
    tickers: list[str],
    windows: list[tuple[pd.Index, pd.Index, pd.Index]],
    cfg: WFConfig,
) -> PortfolioBacktest:
    px_all = prices[tickers].sort_index()
    idx_all = pd.DatetimeIndex(px_all.index)
    rets = px_all.pct_change().fillna(0.0)
    cost_bps = cfg.fee_bps + cfg.slippage_bps
    limits = cfg.risk_limits

    W = pd.DataFrame(0.0, index=idx_all, columns=tickers)
    meta_rows: list[dict[str, object]] = []
    hwm_anchor: pd.Timestamp | None = None  # breaker re-arms from here

    for train_idx, val_idx, trade_idx in windows:
        t0 = trade_idx[0]
        row: dict[str, object] = {"quarter_start": t0}

        report = scan_pairs_report(
            prices=px_all,
            tickers=tickers,
            train_idx=train_idx,
            val_idx=val_idx,
            lookback_z=cfg.z_lookback,
            entry_z=cfg.entry_z,
            exit_z=cfg.exit_z,
            fee_bps=cfg.fee_bps,
            slippage_bps=cfg.slippage_bps,
            leverage=cfg.leverage,
            corr_threshold=cfg.corr_threshold,
            max_pairs=cfg.max_pairs,
            fdr_q=cfg.fdr_q,
            top_k=cfg.top_k,
            prefilter_method=cfg.prefilter_method,
            fallback_to_lowest_p=cfg.fallback_to_lowest_p,
            max_half_life=cfg.max_half_life,
        )
        row.update(
            n_candidates=report.n_candidates,
            n_tested=len(report.tested),
            n_fdr_pass=report.n_fdr_pass,
        )

        # Drawdown breaker on realised equity *before* this quarter.
        breaker = False
        if limits is not None:
            past = W.loc[W.index < t0]
            if hwm_anchor is not None:
                past = past.loc[past.index >= hwm_anchor]
            if len(past) > 10:
                eq = backtest_weights(
                    past, rets.loc[past.index], cost_bps, cfg.short_borrow_cost_bps
                )["equity"]
                breaker = check_drawdown_breaker(eq, limits)
        row["breaker"] = breaker
        if breaker:
            hwm_anchor = t0  # flat for one quarter, then resume with a fresh high-water mark

        hist_idx = train_idx.union(val_idx)
        sig_idx = hist_idx.union(trade_idx)
        books: dict[str, pd.DataFrame] = {}
        hist_rets: dict[str, pd.Series] = {}
        scales: dict[str, float] = {}
        for r in report.selected if not breaker else []:
            book = _pair_book(px_all, r, train_idx, sig_idx, cfg)
            if book is None:
                continue
            name = f"{r.y}/{r.x}"
            h_idx = hist_idx.intersection(book.index)
            h = backtest_weights(book.loc[h_idx], rets.loc[h_idx, list(book.columns)], cost_bps)[
                "ret_net"
            ]
            scale = (
                vol_target_scale(h, cfg.target_daily_vol, cfg.max_pair_scale)
                if cfg.use_vol_targeting
                else 1.0
            )
            books[name], hist_rets[name], scales[name] = book, h * scale, scale

        if not books:
            row.update(n_pairs=0, pairs="", scales="", alloc="")
            meta_rows.append(row)
            continue

        names = list(books)
        if cfg.use_correlation_weights and len(names) >= 2:
            max_w = limits.max_position_size if limits is not None else 0.5
            alloc = correlation_aware_weights(pd.DataFrame(hist_rets), max_weight=max_w)
        else:
            alloc = pd.Series(1.0 / len(names), index=names)
        if limits is not None:
            alloc = apply_position_limits(alloc, limits)

        trade_rows = W.index.isin(trade_idx)
        for name in names:
            book = books[name].reindex(trade_idx).fillna(0.0)
            W.loc[trade_rows, list(book.columns)] += book.to_numpy() * scales[name] * alloc[name]
        if limits is not None:
            W.loc[trade_rows] = cap_gross_exposure(W.loc[trade_rows], limits.max_gross_exposure)

        row.update(
            n_pairs=len(names),
            pairs=", ".join(names),
            scales=", ".join(f"{scales[n]:.2f}" for n in names),
            alloc=", ".join(f"{alloc[n]:.2f}" for n in names),
        )
        meta_rows.append(row)

    meta = pd.DataFrame(meta_rows)
    if not windows:
        empty = backtest_weights(W.iloc[:0], rets.iloc[:0])
        return PortfolioBacktest(empty, meta, W.iloc[:0])

    # Report only the walk-forward period: zero-padding the warm-up would dilute
    # CAGR and Sharpe.
    start, end = windows[0][2][0], windows[-1][2][-1]
    W_live = W.loc[start:end]
    bt = backtest_weights(W_live, rets.loc[W_live.index], cost_bps, cfg.short_borrow_cost_bps)
    bt["n_assets"] = (W_live.abs() > 0).sum(axis=1).astype(int)
    return PortfolioBacktest(bt, meta, W_live)
