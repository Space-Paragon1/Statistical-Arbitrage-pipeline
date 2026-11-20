"""Regenerate the comparison figures embedded in the README.

    python scripts/make_readme_figures.py            # writes docs/figures/*.png

Uses the cached Yahoo download (data/cache/) after the first run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from sarb.config import DEFAULT_UNIVERSE
from sarb.data.ingest import load_yfinance_prices
from sarb.features.spread import rolling_hedge_ratio
from sarb.metrics.performance import performance_summary
from sarb.research.walkforward_portfolio import WFConfig, walkforward_quarterly_portfolio
from sarb.split.rebalance import rolling_windows_by_quarter
from sarb.viz.charts import BENCH, LOSS, STRATEGY, _style, save_figure

OUT = Path("docs/figures")
START, END = "2015-01-01", "2025-01-01"


def hedge_units_figure(prices: pd.DataFrame) -> None:
    """Share ratio vs dollar ratio: the gap is the size of the old engine's hedging error."""
    _, beta = rolling_hedge_ratio(prices["KO"], prices["PEP"], 504)
    dollar = beta * prices["PEP"] / prices["KO"]
    fig = Figure(figsize=(10, 4))
    ax = fig.subplots()
    ax.plot(
        beta.index,
        beta.to_numpy(),
        color=BENCH,
        linewidth=1.2,
        label="β from the price regression (shares of PEP per share of KO)",
    )
    ax.plot(
        dollar.index,
        dollar.to_numpy(),
        color=STRATEGY,
        linewidth=1.4,
        label="β · P_PEP / P_KO (dollars of PEP per dollar of KO)",
    )
    ax.fill_between(
        beta.index,
        beta.to_numpy(),
        dollar.to_numpy(),
        color=LOSS,
        alpha=0.12,
        label="Unhedged exposure in v0.1",
    )
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)
    _style(ax, "KO/PEP hedge ratio: share units vs dollar units", "Hedge ratio")
    fig.tight_layout()
    save_figure(fig, OUT / "hedge_ratio_units.png")


def fdr_discipline_figure(prices: pd.DataFrame) -> None:
    """Strict FDR selection vs. trading the lowest p-values when nothing is significant."""
    tickers = list(prices.columns)
    windows = rolling_windows_by_quarter(prices, train_days=504, val_days=126)
    runs = {
        "Strict BH-FDR (default)": WFConfig(use_vol_targeting=True),
        "Fallback: trade lowest p-values anyway": WFConfig(
            use_vol_targeting=True, fallback_to_lowest_p=True
        ),
    }
    fig = Figure(figsize=(10, 4.5))
    ax = fig.subplots()
    colors = [STRATEGY, LOSS]
    for (label, cfg), color in zip(runs.items(), colors):
        res = walkforward_quarterly_portfolio(prices, tickers, windows, cfg)
        r = res.returns["ret_net"]
        s = performance_summary(res.returns)
        eq = (1 + r).cumprod()
        ax.plot(
            eq.index,
            eq.to_numpy(),
            color=color,
            linewidth=1.4,
            label=f"{label}  (Sharpe {s['sharpe']:.2f}, max DD {s['max_drawdown']:.0%}, "
            f"in market {s['time_in_market']:.0%})",
        )
        print(f"{label}: {s}")
    ax.axhline(1.0, color="black", linewidth=0.5)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    _style(
        ax, "Walk-forward portfolio, 16-ticker universe, 2017-2024 (vol-targeted)", "Growth of $1"
    )
    fig.tight_layout()
    save_figure(fig, OUT / "fdr_discipline.png")


def main() -> None:
    prices = load_yfinance_prices(list(DEFAULT_UNIVERSE), START, END).dropna(axis=1)
    hedge_units_figure(prices)
    fdr_discipline_figure(prices)
    print(f"Figures written to {OUT}/")


if __name__ == "__main__":
    main()
