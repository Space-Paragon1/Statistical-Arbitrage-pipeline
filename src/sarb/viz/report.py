from __future__ import annotations
from pathlib import Path
import pandas as pd

from sarb.viz.charts import (
    plot_equity_curve,
    plot_drawdown,
    plot_spread,
    plot_rolling_beta,
    save_figure,
)


def save_backtest_report(
    bt: pd.DataFrame,
    spread: pd.Series,
    z: pd.Series,
    entry_z: float,
    exit_z: float,
    output_dir: str | Path = "reports/figures",
    prefix: str = "pair",
) -> list[str]:
    """Generate and save standard backtest plots. Returns list of saved file paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    # Equity curve
    fig = plot_equity_curve(bt["equity"], title=f"{prefix} — Equity Curve")
    p = str(out / f"{prefix}_equity.png")
    save_figure(fig, p)
    saved.append(p)

    # Drawdown
    fig = plot_drawdown(bt["equity"], title=f"{prefix} — Drawdown")
    p = str(out / f"{prefix}_drawdown.png")
    save_figure(fig, p)
    saved.append(p)

    # Spread & z-score
    fig = plot_spread(spread, z, entry_z, exit_z, title=f"{prefix} — Spread & Z-Score")
    p = str(out / f"{prefix}_spread_zscore.png")
    save_figure(fig, p)
    saved.append(p)

    # Rolling beta (if available)
    if "beta" in bt.columns:
        fig = plot_rolling_beta(bt["beta"], title=f"{prefix} — Rolling Hedge Ratio")
        p = str(out / f"{prefix}_rolling_beta.png")
        save_figure(fig, p)
        saved.append(p)

    return saved
