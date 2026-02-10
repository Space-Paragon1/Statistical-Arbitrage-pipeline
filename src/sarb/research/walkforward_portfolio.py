from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sarb.research.select_pairs import scan_pairs
from sarb.features.spread import fit_hedge_ratio, compute_spread, rolling_zscore
from sarb.strategy.pairs import generate_spread_positions
from sarb.backtest.engine import backtest_pairs
from sarb.portfolio.vol_target import vol_target_scale


@dataclass
class WFConfig:
    top_k: int = 5
    corr_threshold: float = 0.6
    max_pairs: int = 300
    fdr_q: float = 0.10

    z_lookback: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5

    fee_bps: float = 1.0
    slippage_bps: float = 0.5
    leverage: float = 1.0  # per pair (portfolio weights scale overall)

        # Vol targeting (new)
    use_vol_targeting: bool = True
    target_daily_vol: float = 0.008   # ~0.8% daily vol (reasonable baseline)
    max_pair_scale: float = 3.0       # cap leverage multiplier


def trade_one_pair_window(
    prices: pd.DataFrame,
    y: str,
    x: str,
    train_idx: pd.Index,
    all_idx_for_signals: pd.Index,
    trade_idx: pd.Index,
    cfg: WFConfig,
) -> pd.Series:
    """
    Returns daily net returns on trade window for one pair, trained only on train_idx.
    Signals computed on all_idx_for_signals (train+val+trade) but execution is lookahead-safe via shift.
    """
    px_train = prices.loc[train_idx, [y, x]].dropna()
    if len(px_train) < 200:
        return pd.Series(index=trade_idx, data=0.0)

    alpha, beta = fit_hedge_ratio(px_train[y], px_train[x])

    px_sig = prices.loc[all_idx_for_signals, [y, x]].dropna()
    spread = compute_spread(px_sig[y], px_sig[x], alpha, beta)
    z = rolling_zscore(spread, cfg.z_lookback)
    pos = generate_spread_positions(z, cfg.entry_z, cfg.exit_z)

    bt = backtest_pairs(
        prices=px_sig,
        y=y,
        x=x,
        alpha=alpha,
        beta=beta,
        spread_pos=pos,
        fee_bps=cfg.fee_bps,
        slippage_bps=cfg.slippage_bps,
        leverage=cfg.leverage,
    )

    # return only trade window
    out = bt.loc[trade_idx.intersection(bt.index), "ret_net"].copy()
    return out.reindex(trade_idx).fillna(0.0)

def pair_returns_on_window(
    prices: pd.DataFrame,
    y: str,
    x: str,
    train_idx: pd.Index,
    window_idx: pd.Index,
    cfg: WFConfig,
) -> pd.Series:
    """
    Compute pair net returns on a historical window (e.g., train+val),
    using alpha/beta fitted on TRAIN only. This is what we use to estimate vol.
    """
    px_train = prices.loc[train_idx, [y, x]].dropna()
    if len(px_train) < 200:
        return pd.Series(index=window_idx, data=0.0)

    alpha, beta = fit_hedge_ratio(px_train[y], px_train[x])

    px_win = prices.loc[window_idx, [y, x]].dropna()
    spread = compute_spread(px_win[y], px_win[x], alpha, beta)
    z = rolling_zscore(spread, cfg.z_lookback)
    pos = generate_spread_positions(z, cfg.entry_z, cfg.exit_z)

    bt = backtest_pairs(
        prices=px_win,
        y=y,
        x=x,
        alpha=alpha,
        beta=beta,
        spread_pos=pos,
        fee_bps=cfg.fee_bps,
        slippage_bps=cfg.slippage_bps,
        leverage=cfg.leverage,
    )

    r = bt["ret_net"].reindex(window_idx).fillna(0.0)
    return r

def walkforward_quarterly_portfolio(
    prices: pd.DataFrame,
    tickers: list[str],
    windows: list[tuple[pd.Index, pd.Index, pd.Index]],
    cfg: WFConfig,
) -> pd.DataFrame:
    """
    For each quarter:
      - select pairs using train+val (selection uses train for corr/ADF; val for Sharpe ranking)
      - trade selected pairs in next quarter
      - combine returns equally across pairs (simple, defensible baseline)
    """
    idx_all = pd.DatetimeIndex(prices.index).sort_values()
    portfolio_ret = pd.Series(index=idx_all, data=0.0)
    meta_rows = []

    for k, (train_idx, val_idx, trade_idx) in enumerate(windows):
        # selection uses train & val only
        selected = scan_pairs(
            prices=prices,
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
        )

        if not selected:
            meta_rows.append(
                {"quarter_start": trade_idx[0], "n_pairs": 0, "pairs": ""}
            )
            continue

        # signals can use train+val+trade (still time-safe due to shift),
        # but parameters (alpha/beta) are train-only
        all_sig_idx = train_idx.union(val_idx).union(trade_idx)

        pair_rets = []
        pair_names = []
        pair_scales = []

        hist_idx = train_idx.union(val_idx)  # past only, used for vol estimate
        all_sig_idx = hist_idx.union(trade_idx)

        for r in selected:
            name = f"{r.y}/{r.x}"

            # 1) Estimate scale from TRAIN+VAL returns (alpha/beta trained on TRAIN only)
            if cfg.use_vol_targeting:
                hist_r = pair_returns_on_window(
                    prices=prices,
                    y=r.y, x=r.x,
                    train_idx=train_idx,
                    window_idx=hist_idx,
                    cfg=cfg,
                )
                scale = vol_target_scale(
                    hist_r,
                    target_daily_vol=cfg.target_daily_vol,
                    max_scale=cfg.max_pair_scale,
                )
            else:
                scale = 1.0

            # 2) Trade in next quarter, then scale returns
            pr_trade = trade_one_pair_window(
                prices=prices,
                y=r.y,
                x=r.x,
                train_idx=train_idx,
                all_idx_for_signals=all_sig_idx,
                trade_idx=trade_idx,
                cfg=cfg,
            )

            pair_rets.append(pr_trade * scale)
            pair_names.append(name)
            pair_scales.append(scale)

        # 3) Equal-weight across (already vol-normalized) returns
        R = pd.concat(pair_rets, axis=1).fillna(0.0)
        port_q = R.mean(axis=1)
        portfolio_ret.loc[trade_idx] = port_q.values

        meta_rows.append(
            {
                "quarter_start": trade_idx[0],
                "n_pairs": len(pair_names),
                "pairs": ", ".join(pair_names),
                "scales": ", ".join([f"{s:.2f}" for s in pair_scales]),
            }
        )


    out = pd.DataFrame({"ret_net": portfolio_ret})
    out["equity"] = (1.0 + out["ret_net"]).cumprod()
    meta = pd.DataFrame(meta_rows)
    return out, meta
