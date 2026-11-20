"""Price data loading (Yahoo Finance or local CSV) with an on-disk cache."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = Path("data/cache")


def _tidy(px: pd.DataFrame, ffill_limit: int | None) -> pd.DataFrame:
    px = px.sort_index()
    px.index = pd.to_datetime(px.index)
    px.index.name = "Date"
    px.columns.name = None
    px = px.dropna(how="all")
    return px.ffill(limit=ffill_limit)


def load_yfinance_prices(
    tickers: list[str],
    start: str,
    end: str,
    field: str = "Adj Close",
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    ffill_limit: int | None = 5,
) -> pd.DataFrame:
    """Daily prices for ``tickers`` (columns in the order given).

    Downloads are cached as CSV under ``cache_dir`` keyed by tickers/dates/field so
    re-runs are fast and reproducible; pass ``cache_dir=None`` to disable or
    ``refresh=True`` to re-download. Short gaps are forward-filled (at most
    ``ffill_limit`` days). Tickers with a shorter history keep leading NaNs instead
    of silently truncating every other ticker's history; drop them explicitly with
    ``prices.dropna(axis=1)`` if you need a rectangular panel.
    """
    tickers = list(dict.fromkeys(tickers))
    cache_path: Path | None = None
    if cache_dir is not None:
        key = hashlib.sha1(f"{sorted(tickers)}|{start}|{end}|{field}".encode()).hexdigest()[:12]
        cache_path = Path(cache_dir) / f"yf_{key}.csv"
        if cache_path.exists() and not refresh:
            px = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return _tidy(px[tickers], ffill_limit)

    import yfinance as yf

    df = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {tickers} ({start}..{end}).")
    if isinstance(df.columns, pd.MultiIndex):
        px = df[field].copy()
    else:  # single ticker, flat columns
        px = df[[field]].copy()
        px.columns = tickers
    missing = [t for t in tickers if t not in px.columns or px[t].isna().all()]
    if missing:
        raise RuntimeError(f"No '{field}' data returned for {missing}.")
    px = _tidy(px[tickers], ffill_limit)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        px.to_csv(cache_path)
    return px


def load_csv_prices(
    path_by_ticker: dict[str, str],
    date_col: str = "Date",
    price_col: str = "Adj Close",
) -> pd.DataFrame:
    """One CSV per ticker (e.g. Yahoo exports) -> aligned price panel without NaNs."""
    frames = []
    for tkr, path in path_by_ticker.items():
        df = pd.read_csv(path)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).set_index(date_col)
        frames.append(df[[price_col]].rename(columns={price_col: tkr}))
    px = pd.concat(frames, axis=1).sort_index()
    return px.ffill().dropna()


def load_price_panel_csv(path: str | Path) -> pd.DataFrame:
    """A single wide CSV (``Date`` index, one column per ticker)."""
    return _tidy(pd.read_csv(path, index_col=0, parse_dates=True), ffill_limit=None)
