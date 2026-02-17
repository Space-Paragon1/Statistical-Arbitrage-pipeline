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
