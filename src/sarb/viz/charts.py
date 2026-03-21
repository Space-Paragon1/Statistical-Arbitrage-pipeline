from __future__ import annotations
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_equity_curve(
    equity: pd.Series,
    title: str = "Equity Curve",
    benchmark: pd.Series | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity.index, equity.values, label="Strategy", linewidth=1.2)
    if benchmark is not None:
        ax.plot(benchmark.index, benchmark.values, label="Benchmark", linewidth=1.0, alpha=0.7)
        ax.legend()
    ax.set_title(title)
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_drawdown(equity: pd.Series, title: str = "Drawdown") -> Figure:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(dd.index, dd.values, 0, alpha=0.5, color="red")
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_spread(
    spread: pd.Series,
    z_score: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    title: str = "Spread & Z-Score",
) -> Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(spread.index, spread.values, linewidth=0.8)
    ax1.set_title(title)
    ax1.set_ylabel("Spread")
    ax1.grid(True, alpha=0.3)

    ax2.plot(z_score.index, z_score.values, linewidth=0.8)
    ax2.axhline(entry_z, color="red", linestyle="--", alpha=0.6, label=f"+{entry_z}")
    ax2.axhline(-entry_z, color="green", linestyle="--", alpha=0.6, label=f"-{entry_z}")
    ax2.axhline(exit_z, color="gray", linestyle=":", alpha=0.5)
    ax2.axhline(-exit_z, color="gray", linestyle=":", alpha=0.5)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Z-Score")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_zscore(
    z: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    positions: pd.Series | None = None,
    title: str = "Z-Score",
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(z.index, z.values, linewidth=0.8, label="Z-Score")
    ax.axhline(entry_z, color="red", linestyle="--", alpha=0.6)
    ax.axhline(-entry_z, color="green", linestyle="--", alpha=0.6)
    ax.axhline(exit_z, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(-exit_z, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.5)

    if positions is not None:
        long_mask = positions > 0
        short_mask = positions < 0
        ax.fill_between(z.index, z.min(), z.max(), where=long_mask, alpha=0.1, color="green")
        ax.fill_between(z.index, z.min(), z.max(), where=short_mask, alpha=0.1, color="red")

    ax.set_title(title)
    ax.set_ylabel("Z-Score")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_rolling_beta(beta_series: pd.Series, title: str = "Rolling Hedge Ratio") -> Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(beta_series.index, beta_series.values, linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel("Beta (Hedge Ratio)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(returns: pd.DataFrame, title: str = "Return Correlation") -> Figure:
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8)

    tickers = list(corr.columns)
    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_yticklabels(tickers)

    for i in range(len(tickers)):
        for j in range(len(tickers)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title(title)
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: str, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
