"""End-to-end CLI smoke tests on a frozen CSV (no network)."""

from __future__ import annotations

import pytest

from sarb.cli import main


@pytest.fixture
def prices_csv(tmp_path, universe):
    path = tmp_path / "prices.csv"
    universe.to_csv(path)
    return str(path)


def _data(prices_csv):
    return ["--prices-csv", prices_csv, "--start", "2016-01-01", "--end", "2020-12-31"]


def test_cli_backtest(prices_csv, tmp_path, capsys):
    out = tmp_path / "figs"
    rc = main(
        [
            "backtest",
            "--y",
            "B",
            "--x",
            "A",
            "--train-lookback",
            "250",
            "--n-boot",
            "200",
            "--out",
            str(out),
            *_data(prices_csv),
        ]
    )
    assert rc == 0
    text = capsys.readouterr().out
    assert "Engle-Granger p-value" in text and "sharpe" in text
    assert (out / "B_A_tearsheet.png").exists()


def test_cli_scan(prices_csv, capsys):
    rc = main(
        [
            "scan",
            "--tickers",
            "A",
            "B",
            "C",
            "D",
            "E",
            "--corr-threshold",
            "0",
            "--fdr-q",
            "0.2",
            "--n-boot",
            "200",
            *_data(prices_csv),
        ]
    )
    assert rc == 0
    assert "pass BH-FDR" in capsys.readouterr().out


def test_cli_portfolio(prices_csv, tmp_path, capsys):
    out = tmp_path / "pf"
    rc = main(
        [
            "portfolio",
            "--tickers",
            "A",
            "B",
            "C",
            "D",
            "E",
            "--corr-threshold",
            "0",
            "--train-days",
            "300",
            "--val-days",
            "100",
            "--modes",
            "equal_weight",
            "--out",
            str(out),
            *_data(prices_csv),
        ]
    )
    assert rc == 0
    assert "Walk-forward portfolio" in capsys.readouterr().out
    assert (out / "portfolio_summary.csv").exists()


def test_cli_paper(prices_csv, capsys):
    rc = main(["paper", "--pairs", "B/A", *_data(prices_csv)])
    assert rc == 0
    assert "Paper trading" in capsys.readouterr().out


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "sarb" in capsys.readouterr().out


def test_cli_sweep(prices_csv, tmp_path, capsys):
    out = tmp_path / "sw"
    rc = main(
        [
            "sweep",
            "--y",
            "B",
            "--x",
            "A",
            "--train-lookback",
            "250",
            "--entry-grid",
            "1.5",
            "2.0",
            "--lookback-grid",
            "20",
            "40",
            "--out",
            str(out),
            *_data(prices_csv),
        ]
    )
    assert rc == 0
    text = capsys.readouterr().out
    assert "Deflated Sharpe" in text
    assert (out / "B_A_sweep.png").exists()
