from __future__ import annotations
import pandas as pd
import yfinance as yf

def load_yfinance_prices(tickers: list[str], start: str, end: str, field: str = "Adj Close") -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        px = df[field].copy()
    else:
        # single ticker case
        px = df[[field]].copy()
        px.columns = tickers
    px = px.dropna(how="all")
    px = px.ffill().dropna()
    px.index = pd.to_datetime(px.index)
    return px

def load_csv_prices(path_by_ticker: dict[str, str], date_col: str = "Date", price_col: str = "Adj Close") -> pd.DataFrame:
    frames = []
    for tkr, path in path_by_ticker.items():
        df = pd.read_csv(path)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).set_index(date_col)
        frames.append(df[[price_col]].rename(columns={price_col: tkr}))
    px = pd.concat(frames, axis=1).sort_index()
    px = px.ffill().dropna()
    return px
