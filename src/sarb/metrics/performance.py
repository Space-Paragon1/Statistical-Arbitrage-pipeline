"""Performance and trade statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe(daily_returns: pd.Series, ann_factor: int = TRADING_DAYS) -> float:
    r = daily_returns.dropna()
    sd = r.std(ddof=0)
    if len(r) == 0 or sd == 0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(ann_factor))


def sortino(daily_returns: pd.Series, ann_factor: int = TRADING_DAYS) -> float:
    """Sortino ratio with a zero target (downside deviation over all observations)."""
    r = daily_returns.dropna()
    downside = np.sqrt((np.minimum(r, 0.0) ** 2).mean()) if len(r) else 0.0
    if downside == 0:
        return 0.0
    return float(r.mean() / downside * np.sqrt(ann_factor))


def annualized_vol(daily_returns: pd.Series, ann_factor: int = TRADING_DAYS) -> float:
    r = daily_returns.dropna()
    return float(r.std(ddof=0) * np.sqrt(ann_factor)) if len(r) else 0.0


def drawdown_series(equity: pd.Series) -> pd.Series:
    eq = equity.dropna()
    return eq / eq.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    dd = drawdown_series(equity)
    return float(dd.min()) if len(dd) else 0.0


def max_drawdown_duration(equity: pd.Series) -> int:
    """Longest stretch (in bars) spent below a previous equity high."""
    underwater = (drawdown_series(equity) < 0).to_numpy()
    longest = run = 0
    for flag in underwater:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return int(longest)


def cagr(equity: pd.Series, ann_factor: int = TRADING_DAYS) -> float:
    eq = equity.dropna()
    if len(eq) < 2 or eq.iloc[0] <= 0:
        return 0.0
    total = eq.iloc[-1] / eq.iloc[0]
    years = len(eq) / ann_factor
    if total <= 0:
        return -1.0
    return float(total ** (1 / years) - 1) if years > 0 else 0.0


def calmar(daily_returns: pd.Series, ann_factor: int = TRADING_DAYS) -> float:
    eq = (1.0 + daily_returns.fillna(0.0)).cumprod()
    mdd = max_drawdown(eq)
    return float(cagr(eq, ann_factor) / abs(mdd)) if mdd < 0 else 0.0


def trade_statistics(position: pd.Series, daily_returns: pd.Series) -> dict[str, float]:
    """Round-trip statistics from a held-position series and its daily P&L.

    A trade is a maximal run of bars with the same non-zero position.
    """
    pos = position.fillna(0.0).to_numpy()
    ret = daily_returns.reindex(position.index).fillna(0.0).to_numpy()
    trades: list[tuple[int, float]] = []  # (length, compounded return)
    i, n = 0, len(pos)
    while i < n:
        if pos[i] == 0:
            i += 1
            continue
        j = i
        while j < n and pos[j] == pos[i]:
            j += 1
        trades.append((j - i, float(np.prod(1.0 + ret[i:j]) - 1.0)))
        i = j
    if not trades:
        return {
            "n_trades": 0,
            "hit_rate": float("nan"),
            "avg_holding_days": float("nan"),
            "avg_trade_return": float("nan"),
        }
    lengths = np.array([t[0] for t in trades], dtype=float)
    rets = np.array([t[1] for t in trades])
    return {
        "n_trades": len(trades),
        "hit_rate": float((rets > 0).mean()),
        "avg_holding_days": float(lengths.mean()),
        "avg_trade_return": float(rets.mean()),
    }


def performance_summary(
    bt: pd.DataFrame,
    ret_col: str = "ret_net",
    ann_factor: int = TRADING_DAYS,
) -> dict[str, float]:
    """One-stop summary of a backtest frame produced by this package."""
    r = bt[ret_col].fillna(0.0)
    eq = (1.0 + r).cumprod()
    out: dict[str, float] = {
        "sharpe": sharpe(r, ann_factor),
        "sortino": sortino(r, ann_factor),
        "cagr": cagr(eq, ann_factor),
        "ann_vol": annualized_vol(r, ann_factor),
        "max_drawdown": max_drawdown(eq),
        "max_dd_days": float(max_drawdown_duration(eq)),
        "calmar": calmar(r, ann_factor),
        "total_return": float(eq.iloc[-1] - 1.0) if len(eq) else 0.0,
    }
    if "gross_exposure" in bt:
        out["time_in_market"] = float((bt["gross_exposure"] > 0).mean())
        out["avg_gross_exposure"] = float(bt["gross_exposure"].mean())
    if "turnover" in bt:
        out["ann_turnover"] = float(bt["turnover"].mean() * ann_factor)
    if "costs" in bt and "ret_gross" in bt:
        gross_pnl = bt["ret_gross"].sum()
        out["cost_drag_ann"] = float(bt["costs"].mean() * ann_factor)
        out["gross_sharpe"] = sharpe(bt["ret_gross"], ann_factor)
        out["costs_pct_of_gross"] = (
            float(bt["costs"].sum() / gross_pnl) if gross_pnl > 0 else float("nan")
        )
    if "pos" in bt:
        out.update(trade_statistics(bt["pos"], r))
    return out
