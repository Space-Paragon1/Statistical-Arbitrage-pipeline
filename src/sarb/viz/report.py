"""Standard figure set for a single-pair backtest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sarb.viz.charts import (
    plot_rolling_beta,
    plot_spread,
    plot_tearsheet,
    plot_zscore,
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
    """Save tearsheet, spread/z-score and hedge-ratio plots. Returns the file paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    label = prefix.replace("_", "/")

    def _save(fig, name: str) -> None:
        path = out / f"{prefix}_{name}.png"
        save_figure(fig, path)
        saved.append(str(path))

    _save(plot_tearsheet(bt["ret_net"], title=label), "tearsheet")
    _save(
        plot_spread(spread, z, entry_z, exit_z, title=f"{label} spread & z-score"), "spread_zscore"
    )
    if "pos" in bt:
        _save(
            plot_zscore(
                z, entry_z, exit_z, positions=bt["pos"], title=f"{label} z-score and positions"
            ),
            "positions",
        )
    if "beta" in bt:
        _save(plot_rolling_beta(bt["beta"], title=f"{label} rolling hedge ratio"), "rolling_beta")
    return saved
