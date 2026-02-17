from __future__ import annotations
import pandas as pd
import numpy as np
from matplotlib.figure import Figure

from sarb.viz.charts import (
    plot_equity_curve,
    plot_drawdown,
    plot_spread,
    plot_zscore,
    plot_rolling_beta,
    plot_correlation_heatmap,
    save_figure,
)


def test_plot_equity_curve(synthetic_prices):
    eq = (1 + synthetic_prices["Y"].pct_change().fillna(0)).cumprod()
    fig = plot_equity_curve(eq)
    assert isinstance(fig, Figure)


def test_plot_equity_curve_with_benchmark(synthetic_prices):
    eq = (1 + synthetic_prices["Y"].pct_change().fillna(0)).cumprod()
    bm = (1 + synthetic_prices["X"].pct_change().fillna(0)).cumprod()
    fig = plot_equity_curve(eq, benchmark=bm)
    assert isinstance(fig, Figure)


def test_plot_drawdown(synthetic_prices):
    eq = (1 + synthetic_prices["Y"].pct_change().fillna(0)).cumprod()
    fig = plot_drawdown(eq)
    assert isinstance(fig, Figure)


def test_plot_spread(synthetic_prices):
    spread = synthetic_prices["Y"] - 1.2 * synthetic_prices["X"]
    z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
    fig = plot_spread(spread, z)
    assert isinstance(fig, Figure)


def test_plot_zscore(synthetic_prices):
    spread = synthetic_prices["Y"] - 1.2 * synthetic_prices["X"]
    z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
    fig = plot_zscore(z)
    assert isinstance(fig, Figure)


def test_plot_rolling_beta():
    rng = np.random.default_rng(42)
    beta = pd.Series(1.2 + np.cumsum(rng.normal(0, 0.01, 200)),
                     index=pd.bdate_range("2020-01-01", periods=200))
    fig = plot_rolling_beta(beta)
    assert isinstance(fig, Figure)


def test_plot_correlation_heatmap(synthetic_returns):
    fig = plot_correlation_heatmap(synthetic_returns)
    assert isinstance(fig, Figure)


def test_save_figure(tmp_path, synthetic_prices):
    eq = (1 + synthetic_prices["Y"].pct_change().fillna(0)).cumprod()
    fig = plot_equity_curve(eq)
    path = str(tmp_path / "test.png")
    save_figure(fig, path)
    assert (tmp_path / "test.png").exists()
