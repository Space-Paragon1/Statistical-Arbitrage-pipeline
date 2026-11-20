from __future__ import annotations

import pandas as pd

from sarb.data.ingest import load_csv_prices


def test_load_csv_prices(tmp_path):
    # Write two temporary CSV files
    dates = pd.bdate_range("2020-01-01", periods=50)
    for tkr, base in [("A", 100), ("B", 50)]:
        df = pd.DataFrame({"Date": dates, "Adj Close": [base + i for i in range(50)]})
        df.to_csv(tmp_path / f"{tkr}.csv", index=False)

    paths = {"A": str(tmp_path / "A.csv"), "B": str(tmp_path / "B.csv")}
    px = load_csv_prices(paths, date_col="Date", price_col="Adj Close")

    assert isinstance(px, pd.DataFrame)
    assert list(px.columns) == ["A", "B"]
    assert len(px) == 50
    assert px.isna().sum().sum() == 0


def test_yfinance_loader_caches_and_validates(tmp_path, monkeypatch):
    import sys
    import types

    import pytest

    from sarb.data.ingest import load_yfinance_prices

    dates = pd.bdate_range("2020-01-01", periods=5)
    calls = []

    def fake_download(tickers, **kw):
        calls.append(tickers)
        cols = pd.MultiIndex.from_product([["Adj Close", "Close"], tickers])
        data = [[10.0 + i] * len(cols) for i in range(5)]
        df = pd.DataFrame(data, index=dates, columns=cols)
        df.iloc[2, 0] = float("nan")  # short gap -> forward-filled
        return df

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=fake_download))
    px = load_yfinance_prices(["A", "B"], "2020-01-01", "2020-01-10", cache_dir=tmp_path)
    assert list(px.columns) == ["A", "B"]
    assert px.isna().sum().sum() == 0
    assert px.loc[dates[2], "A"] == px.loc[dates[1], "A"]

    again = load_yfinance_prices(["A", "B"], "2020-01-01", "2020-01-10", cache_dir=tmp_path)
    assert len(calls) == 1, "second call should hit the cache"
    pd.testing.assert_frame_equal(px, again, check_freq=False)

    monkeypatch.setitem(
        sys.modules, "yfinance", types.SimpleNamespace(download=lambda *a, **k: pd.DataFrame())
    )
    with pytest.raises(RuntimeError, match="no data"):
        load_yfinance_prices(["Z"], "2020-01-01", "2020-01-10", cache_dir=None)
