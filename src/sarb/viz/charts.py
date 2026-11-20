"""Charts built on matplotlib's object-oriented API.

Figures are created with :class:`matplotlib.figure.Figure` directly instead of
``pyplot``: no global state, no forced backend, nothing to close, and they still
render inline in Jupyter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

STRATEGY = "#1f4e79"
BENCH = "#9a9a9a"
LOSS = "#b03a2e"
GAIN = "#1e8449"


def _style(ax, title: str | None = None, ylabel: str | None = None) -> None:
    if title:
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_equity_curve(
    equity: pd.Series,
    title: str = "Equity Curve",
    benchmark: pd.Series | None = None,
    label: str = "Strategy",
    benchmark_label: str = "Benchmark",
) -> Figure:
    fig = Figure(figsize=(10, 5))
    ax = fig.subplots()
    ax.plot(equity.index, equity.to_numpy(), label=label, linewidth=1.3, color=STRATEGY)
    if benchmark is not None:
        ax.plot(
            benchmark.index, benchmark.to_numpy(), label=benchmark_label, linewidth=1.0, color=BENCH
        )
        ax.legend(frameon=False)
    _style(ax, title, "Equity")
    fig.tight_layout()
    return fig


def plot_drawdown(equity: pd.Series, title: str = "Drawdown") -> Figure:
    dd = equity / equity.cummax() - 1.0
    fig = Figure(figsize=(10, 4))
    ax = fig.subplots()
    ax.fill_between(dd.index, dd.to_numpy(), 0, alpha=0.45, color=LOSS, linewidth=0)
    _style(ax, title, "Drawdown")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.tight_layout()
    return fig


def plot_spread(
    spread: pd.Series,
    z_score: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    title: str = "Spread & Z-Score",
) -> Figure:
    fig = Figure(figsize=(10, 7))
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    ax1.plot(spread.index, spread.to_numpy(), linewidth=0.8, color=STRATEGY)
    _style(ax1, title, "Spread")
    _zscore_axes(ax2, z_score, entry_z, exit_z)
    ax2.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    return fig


def _zscore_axes(ax, z: pd.Series, entry_z: float, exit_z: float) -> None:
    ax.plot(z.index, z.to_numpy(), linewidth=0.8, color=STRATEGY, label="z")
    ax.axhline(entry_z, color=LOSS, linestyle="--", alpha=0.7, label=f"±{entry_z} entry")
    ax.axhline(-entry_z, color=LOSS, linestyle="--", alpha=0.7)
    ax.axhline(exit_z, color=BENCH, linestyle=":", alpha=0.8, label=f"±{exit_z} exit")
    ax.axhline(-exit_z, color=BENCH, linestyle=":", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    _style(ax, None, "Z-score")


def plot_zscore(
    z: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    positions: pd.Series | None = None,
    title: str = "Z-Score",
) -> Figure:
    fig = Figure(figsize=(10, 4))
    ax = fig.subplots()
    _zscore_axes(ax, z, entry_z, exit_z)
    if positions is not None:
        lo, hi = np.nanmin(z.to_numpy()), np.nanmax(z.to_numpy())
        pos = positions.reindex(z.index).fillna(0.0)
        ax.fill_between(
            z.index, lo, hi, where=list(pos > 0), alpha=0.1, color=GAIN, label="long spread"
        )
        ax.fill_between(
            z.index, lo, hi, where=list(pos < 0), alpha=0.1, color=LOSS, label="short spread"
        )
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_rolling_beta(beta_series: pd.Series, title: str = "Rolling Hedge Ratio") -> Figure:
    fig = Figure(figsize=(10, 4))
    ax = fig.subplots()
    ax.plot(beta_series.index, beta_series.to_numpy(), linewidth=1.0, color=STRATEGY)
    _style(ax, title, "Beta (shares of x per share of y)")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(returns: pd.DataFrame, title: str = "Return Correlation") -> Figure:
    corr = returns.corr()
    cv = corr.to_numpy()
    fig = Figure(figsize=(8, 6))
    ax = fig.subplots()
    im = ax.imshow(cv, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8)
    tickers = list(corr.columns)
    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_yticklabels(tickers)
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            ax.text(j, i, f"{cv[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_tearsheet(
    returns: pd.Series,
    title: str = "Strategy",
    benchmark: dict[str, pd.Series] | None = None,
    rolling_window: int = 126,
) -> Figure:
    """Equity, drawdown and rolling Sharpe on one page.

    ``benchmark`` maps label -> daily return series overlaid on the equity panel.
    The equity axis switches to log scale only when growth spans more than 2.5x.
    Rolling Sharpe is blanked where the strategy was flat for most of the window,
    since a handful of non-zero days produces meaningless spikes.
    """
    r = returns.fillna(0.0)
    eq = (1.0 + r).cumprod()
    dd = eq / eq.cummax() - 1.0
    roll = r.rolling(rolling_window)
    active = (r != 0).rolling(rolling_window).mean() >= 0.5
    roll_sr = (roll.mean() / roll.std(ddof=0) * np.sqrt(252)).where(active)

    fig = Figure(figsize=(11, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1, 1], hspace=0.12)
    ax_eq = fig.add_subplot(gs[0])
    ax_dd = fig.add_subplot(gs[1], sharex=ax_eq)
    ax_sr = fig.add_subplot(gs[2], sharex=ax_eq)

    curves = [eq]
    ax_eq.plot(eq.index, eq.to_numpy(), color=STRATEGY, linewidth=1.4, label=title)
    palette = [BENCH, "#c9a227", "#7d3c98"]
    for i, (label, br) in enumerate((benchmark or {}).items()):
        beq = (1.0 + br.reindex(eq.index).fillna(0.0)).cumprod()
        curves.append(beq)
        ax_eq.plot(
            beq.index, beq.to_numpy(), color=palette[i % len(palette)], linewidth=1.0, label=label
        )
    lo = min(float(c.min()) for c in curves)
    hi = max(float(c.max()) for c in curves)
    log_scale = lo > 0 and hi / lo > 2.5
    if log_scale:
        ax_eq.set_yscale("log")
    ax_eq.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
    ax_eq.yaxis.set_minor_formatter(lambda v, _: "")
    ax_eq.axhline(1.0, color="black", linewidth=0.5, alpha=0.6)
    if benchmark:
        ax_eq.legend(frameon=False, loc="upper left")
    _style(ax_eq, title, "Growth of $1" + (" (log)" if log_scale else ""))

    ax_dd.fill_between(dd.index, dd.to_numpy(), 0, color=LOSS, alpha=0.45, linewidth=0)
    ax_dd.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _style(ax_dd, None, "Drawdown")

    ax_sr.plot(roll_sr.index, roll_sr.to_numpy(), color=STRATEGY, linewidth=0.9)
    ax_sr.axhline(0, color="black", linewidth=0.6)
    _style(ax_sr, None, f"Rolling {rolling_window}d Sharpe")

    for ax in (ax_eq, ax_dd):
        for lbl in ax.get_xticklabels():
            lbl.set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.06)
    return fig


def plot_heatmap(
    table: pd.DataFrame,
    title: str,
    fmt: str = "{:.2f}",
    cmap: str = "RdBu",
    center: float = 0.0,
    highlight: tuple[Any, Any] | None = None,
) -> Figure:
    """Annotated heatmap of a 2-D table (rows x columns), diverging around ``center``."""
    vals = table.to_numpy(dtype=float)
    span = float(np.nanmax(np.abs(vals - center))) or 1.0
    fig = Figure(figsize=(1.1 * table.shape[1] + 2.5, 0.6 * table.shape[0] + 1.8))
    ax = fig.subplots()
    im = ax.imshow(vals, cmap=cmap, vmin=center - span, vmax=center + span, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels([str(c) for c in table.columns])
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels([str(i) for i in table.index])
    ax.set_xlabel(str(table.columns.name or ""))
    ax.set_ylabel(str(table.index.name or ""))
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                dark = abs(v - center) > 0.6 * span
                ax.text(
                    j,
                    i,
                    fmt.format(v),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if dark else "black",
                )
    if highlight is not None:
        i = list(table.index).index(highlight[0])
        j = list(table.columns).index(highlight[1])
        ax.add_patch(_rect(j - 0.5, i - 0.5))
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def _rect(x: float, y: float):
    from matplotlib.patches import Rectangle

    return Rectangle((x, y), 1, 1, fill=False, edgecolor="black", linewidth=2.0)


def save_figure(fig: Figure, path: str | Path, dpi: int = 150) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
