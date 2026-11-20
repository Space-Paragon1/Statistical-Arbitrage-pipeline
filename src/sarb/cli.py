"""Command-line interface: ``sarb {backtest,scan,portfolio,sweep,paper}``.

Every command accepts ``--prices-csv`` (a wide Date x ticker CSV) so results can be
reproduced offline from a frozen dataset instead of a live Yahoo download.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from sarb import __version__
from sarb.config import DEFAULT_UNIVERSE, PairConfig

PCT_KEYS = {
    "cagr",
    "ann_vol",
    "max_drawdown",
    "total_return",
    "time_in_market",
    "hit_rate",
    "cost_drag_ann",
    "avg_trade_return",
    "costs_pct_of_gross",
}


# --------------------------------------------------------------------------- helpers
def _fmt(key: str, v: object) -> str:
    if isinstance(v, float):
        if v != v:  # NaN
            return "n/a"
        return f"{v:.2%}" if key in PCT_KEYS else f"{v:.2f}"
    return str(v)


def _print_table(title: str, rows: dict[str, dict[str, float]]) -> None:
    df = pd.DataFrame(rows)
    print(f"\n=== {title} ===")
    print(df.apply(lambda col: [_fmt(k, v) for k, v in col.items()]).to_string())


def _load(tickers: Sequence[str], args: argparse.Namespace) -> pd.DataFrame:
    if args.prices_csv:
        from sarb.data.ingest import load_price_panel_csv

        px = load_price_panel_csv(args.prices_csv)
        missing = [t for t in tickers if t not in px.columns]
        if missing:
            raise SystemExit(f"{args.prices_csv} has no columns for {missing}")
        px = px[list(tickers)].loc[args.start : args.end]
    else:
        from sarb.data.ingest import load_yfinance_prices

        px = load_yfinance_prices(
            list(tickers),
            args.start,
            args.end,
            cache_dir=None if args.no_cache else args.cache_dir,
            refresh=args.refresh,
        )
    full = px.dropna(axis=1, how="any")
    dropped = sorted(set(px.columns) - set(full.columns))
    if dropped:
        print(f"[data] dropping tickers without full history: {dropped}", file=sys.stderr)
    return full


def _add_data_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("data")
    g.add_argument("--start", default=PairConfig.start)
    g.add_argument("--end", default=PairConfig.end)
    g.add_argument("--prices-csv", help="wide CSV (Date index, one column per ticker)")
    g.add_argument("--cache-dir", default="data/cache")
    g.add_argument("--no-cache", action="store_true")
    g.add_argument("--refresh", action="store_true", help="re-download even if cached")


def _add_signal_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("signal")
    g.add_argument("--z-lookback", type=int, default=PairConfig.lookback_z)
    g.add_argument("--entry-z", type=float, default=PairConfig.entry_z)
    g.add_argument("--exit-z", type=float, default=PairConfig.exit_z)
    g.add_argument("--stop-z", type=float, default=None)
    g.add_argument("--max-holding", type=int, default=None)
    g = p.add_argument_group("costs")
    g.add_argument("--fee-bps", type=float, default=PairConfig.fee_bps)
    g.add_argument("--slippage-bps", type=float, default=PairConfig.slippage_bps)
    g.add_argument("--borrow-bps", type=float, default=0.0, help="annualised short borrow")


# --------------------------------------------------------------------------- backtest
def cmd_backtest(args: argparse.Namespace) -> int:
    from sarb.backtest.walkforward import walkforward_pairs_backtest
    from sarb.features.spread import compute_spread
    from sarb.metrics.performance import performance_summary
    from sarb.split.time_split import time_train_test_split
    from sarb.stats.bootstrap import bootstrap_sharpe_ci
    from sarb.stats.cointegration import engle_granger_test, estimate_half_life
    from sarb.stats.sharpe_inference import probabilistic_sharpe_ratio

    y, x = args.y, args.x
    prices = _load([y, x], args)
    train_px, test_px = time_train_test_split(prices, 1.0 - args.test_frac)

    eg = engle_granger_test(train_px[y], train_px[x])
    hl = estimate_half_life(compute_spread(train_px[y], train_px[x], eg.alpha, eg.beta))
    print(
        f"=== {y}/{x} diagnostics (in-sample, {train_px.index[0].date()} .. "
        f"{train_px.index[-1].date()}) ==="
    )
    print(f"Engle-Granger p-value: {eg.pvalue:.4f}   (t = {eg.stat:.2f})")
    print(
        f"Hedge ratio: {y} = {eg.alpha:.3f} + {eg.beta:.3f} x {x}   "
        f"(dollar ratio ~ {eg.beta * train_px[x].iloc[-1] / train_px[y].iloc[-1]:.2f})"
    )
    print(f"Half-life: {hl:.1f} days")

    bt = walkforward_pairs_backtest(
        prices,
        y,
        x,
        train_lookback=args.train_lookback,
        z_lookback=args.z_lookback,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        hedge_method=args.hedge,
        stop_z=args.stop_z,
        max_holding=args.max_holding,
        short_borrow_cost_bps=args.borrow_bps,
    )
    bt_test = bt.loc[test_px.index]
    summ = performance_summary(bt_test)
    ci = bootstrap_sharpe_ci(bt_test["ret_net"], n_boot=args.n_boot)
    summ["sharpe_ci_low"], summ["sharpe_ci_high"] = ci["ci_low"], ci["ci_high"]
    summ["psr"] = probabilistic_sharpe_ratio(bt_test["ret_net"])
    _print_table(
        f"{y}/{x} walk-forward, out-of-sample {test_px.index[0].date()} .. "
        f"{test_px.index[-1].date()} ({args.hedge})",
        {"OOS": summ},
    )

    if args.out:
        from sarb.viz.report import save_backtest_report

        saved = save_backtest_report(
            bt_test,
            bt_test["spread"],
            bt_test["z"],
            args.entry_z,
            args.exit_z,
            output_dir=args.out,
            prefix=f"{y}_{x}",
        )
        print(f"\nFigures: {', '.join(saved)}")
    return 0


# --------------------------------------------------------------------------- scan
def cmd_scan(args: argparse.Namespace) -> int:
    from sarb.backtest.engine import backtest_pairs
    from sarb.features.spread import compute_spread, rolling_zscore
    from sarb.metrics.performance import max_drawdown, sharpe
    from sarb.research.select_pairs import scan_pairs_report
    from sarb.split.time_split import time_train_val_test_split
    from sarb.stats.bootstrap import bootstrap_sharpe_ci
    from sarb.strategy.pairs import generate_spread_positions

    prices = _load(args.tickers, args)
    tickers = list(prices.columns)
    train, val, test = time_train_val_test_split(prices, train_frac=0.6, val_frac=0.2)

    rep = scan_pairs_report(
        prices,
        tickers,
        train.index,
        val.index,
        args.z_lookback,
        args.entry_z,
        args.exit_z,
        args.fee_bps,
        args.slippage_bps,
        corr_threshold=args.corr_threshold,
        max_pairs=args.max_pairs,
        fdr_q=args.fdr_q,
        top_k=args.top_k,
        prefilter_method=args.prefilter,
        fallback_to_lowest_p=args.fallback,
    )
    print(
        f"Candidates after prefilter: {rep.n_candidates}   tested: {len(rep.tested)}   "
        f"pass BH-FDR(q={args.fdr_q}): {rep.n_fdr_pass}"
    )
    if not rep.selected:
        print(
            "No pair survives FDR control -> nothing to trade. "
            "(--fallback trades the lowest p-values anyway, voiding FDR control.)"
        )
        return 0

    rows = []
    cost = args.fee_bps + args.slippage_bps
    for r in rep.selected:
        spread = compute_spread(prices[r.y], prices[r.x], r.alpha, r.beta)
        pos = generate_spread_positions(
            rolling_zscore(spread, args.z_lookback), args.entry_z, args.exit_z
        )
        bt = backtest_pairs(prices[[r.y, r.x]], r.y, r.x, r.alpha, r.beta, pos, cost, 0.0)
        oos = bt.loc[test.index, "ret_net"]
        ci = bootstrap_sharpe_ci(oos, n_boot=args.n_boot)
        rows.append(
            {
                "pair": f"{r.y}/{r.x}",
                "EG p": r.adf_p,
                "half-life": r.half_life,
                "val SR": r.val_sharpe,
                "val DSR": r.val_dsr,
                "test SR": sharpe(oos),
                "test SR 95% CI": f"[{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]",
                "test MaxDD": max_drawdown((1 + oos).cumprod()),
            }
        )
    df = pd.DataFrame(rows).set_index("pair")
    print("\n=== Selected on TRAIN/VAL, evaluated on untouched TEST ===")
    print(df.to_string(float_format=lambda v: f"{v:.3f}"))
    return 0


# --------------------------------------------------------------------------- portfolio
def cmd_portfolio(args: argparse.Namespace) -> int:
    from sarb.metrics.performance import performance_summary
    from sarb.research.walkforward_portfolio import WFConfig, walkforward_quarterly_portfolio
    from sarb.risk.limits import RiskLimits
    from sarb.split.rebalance import rolling_windows_by_quarter

    prices = _load(args.tickers, args)
    tickers = list(prices.columns)
    windows = rolling_windows_by_quarter(prices, train_days=args.train_days, val_days=args.val_days)
    common = dict(
        top_k=args.top_k,
        corr_threshold=args.corr_threshold,
        fdr_q=args.fdr_q,
        z_lookback=args.z_lookback,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        stop_z=args.stop_z,
        max_holding=args.max_holding,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        short_borrow_cost_bps=args.borrow_bps,
        fallback_to_lowest_p=args.fallback,
        hedge_method=args.hedge,
    )
    configs = {
        "equal_weight": WFConfig(**common, use_vol_targeting=False),
        "vol_target": WFConfig(**common, use_vol_targeting=True, target_daily_vol=args.target_vol),
        "risk_managed": WFConfig(
            **common,
            use_vol_targeting=True,
            target_daily_vol=args.target_vol,
            use_correlation_weights=True,
            risk_limits=RiskLimits(
                max_position_size=0.4, max_gross_exposure=2.0, max_drawdown_pct=0.15
            ),
        ),
    }
    results, summaries = {}, {}
    for name in args.modes:
        res = walkforward_quarterly_portfolio(prices, tickers, windows, configs[name])
        results[name] = res
        summaries[name] = performance_summary(res.returns)

    first = results[args.modes[0]]
    span = f"{first.returns.index[0].date()} .. {first.returns.index[-1].date()}"
    _print_table(f"Walk-forward portfolio, {len(windows)} quarters, {span}", summaries)
    meta = first.meta
    significant = int((meta["n_fdr_pass"] > 0).sum())
    traded = int((meta["n_pairs"] > 0).sum())
    print(
        f"\nQuarters with >=1 FDR-significant pair: {significant}/{len(meta)}   "
        f"quarters traded: {traded}/{len(meta)}"
    )
    print(meta.tail(args.show_quarters).to_string(index=False))

    if args.out:
        from sarb.viz.charts import plot_tearsheet, save_figure

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        # No equity benchmark: the book is market-neutral, so the right comparison is cash.
        for name, res in results.items():
            fig = plot_tearsheet(res.returns["ret_net"], title=f"Portfolio: {name}")
            save_figure(fig, out / f"portfolio_{name}.png")
            res.meta.to_csv(out / f"portfolio_{name}_quarters.csv", index=False)
        pd.DataFrame(summaries).to_csv(out / "portfolio_summary.csv")
        print(f"\nFigures and tables written to {out}/")
    return 0


# --------------------------------------------------------------------------- sweep
def cmd_sweep(args: argparse.Namespace) -> int:
    from sarb.research.sweep import parameter_sweep
    from sarb.split.time_split import time_train_test_split

    y, x = args.y, args.x
    prices = _load([y, x], args)
    _, test_px = time_train_test_split(prices, 1.0 - args.test_frac)
    res = parameter_sweep(
        prices,
        y,
        x,
        args.entry_grid,
        args.lookback_grid,
        test_px.index,
        exit_z=args.exit_z,
        train_lookback=args.train_lookback,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        hedge_method=args.hedge,
    )
    print(
        f"=== {y}/{x}: out-of-sample Sharpe over {res.n_trials} configurations "
        f"({test_px.index[0].date()} .. {test_px.index[-1].date()}) ==="
    )
    print(res.sharpe.to_string(float_format=lambda v: f"{v:.2f}"))
    lb, ez = res.best
    print(f"\nBest cell: z_lookback={lb}, entry_z={ez}  Sharpe {res.best_sharpe:.2f}")
    print(
        f"Best Sharpe expected from luck alone across {res.n_trials} tries: "
        f"{res.expected_max_sharpe_ann:.2f}"
    )
    print(
        f"PSR (ignores the search): {res.best_psr:.2f}   "
        f"Deflated Sharpe (accounts for it): {res.best_dsr:.2f}"
    )
    if args.out:
        from sarb.viz.charts import plot_heatmap, save_figure

        fig = plot_heatmap(
            res.sharpe,
            title=f"{y}/{x} OOS Sharpe by parameters  (best DSR = {res.best_dsr:.2f})",
            highlight=res.best,
        )
        path = Path(args.out) / f"{y}_{x}_sweep.png"
        save_figure(fig, path)
        print(f"\nFigure: {path}")
    return 0


# --------------------------------------------------------------------------- paper
def cmd_paper(args: argparse.Namespace) -> int:
    from sarb.live.paper_broker import PaperBroker
    from sarb.live.runner import LiveConfig, LiveState, run_live_step
    from sarb.metrics.performance import performance_summary

    pairs = [tuple(p.split("/")) for p in args.pairs]
    tickers = list(dict.fromkeys(t for p in pairs for t in p))
    prices = _load(tickers, args)
    broker = PaperBroker(
        initial_capital=args.capital, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps
    )
    cfg = LiveConfig(
        pairs=pairs,
        notional_per_pair=args.notional,
        lookback_z=args.z_lookback,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        stop_z=args.stop_z,
        max_holding=args.max_holding,
    )
    state = LiveState()
    start = cfg.train_lookback + cfg.lookback_z
    values = {}
    for i in range(start, len(prices)):
        hist = prices.iloc[: i + 1]
        broker.set_current_prices({t: float(hist[t].iloc[-1]) for t in tickers})
        run_live_step(hist, cfg, broker, state)
        values[prices.index[i]] = broker.get_account_value()

    equity = pd.Series(values)
    summ = performance_summary(pd.DataFrame({"ret_net": equity.pct_change().fillna(0.0)}))
    summ["final_value"] = equity.iloc[-1]
    summ["fills"] = float(len(broker.get_fill_history()))
    _print_table(
        f"Paper trading {', '.join(args.pairs)} "
        f"({equity.index[0].date()} .. {equity.index[-1].date()})",
        {"paper": summ},
    )
    return 0


# --------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sarb", description="Statistical arbitrage research CLI")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("backtest", help="walk-forward backtest of one pair")
    b.add_argument("--y", default=PairConfig.y_ticker)
    b.add_argument("--x", default=PairConfig.x_ticker)
    b.add_argument("--hedge", choices=["ols", "kalman"], default="ols")
    b.add_argument("--train-lookback", type=int, default=PairConfig.train_lookback)
    b.add_argument("--test-frac", type=float, default=1.0 - PairConfig.train_frac)
    b.add_argument("--n-boot", type=int, default=2000)
    b.add_argument("--out", default="reports/figures", help="figure directory ('' to skip)")
    _add_data_args(b)
    _add_signal_args(b)
    b.set_defaults(func=cmd_backtest)

    s = sub.add_parser("scan", help="scan a universe with FDR-controlled selection")
    s.add_argument("--tickers", nargs="+", default=list(DEFAULT_UNIVERSE))
    s.add_argument("--corr-threshold", type=float, default=0.6)
    s.add_argument("--max-pairs", type=int, default=300)
    s.add_argument("--fdr-q", type=float, default=0.10)
    s.add_argument("--top-k", type=int, default=5)
    s.add_argument("--prefilter", choices=["correlation", "ml"], default="correlation")
    s.add_argument(
        "--fallback",
        action="store_true",
        help="trade lowest p-values if nothing passes FDR (voids FDR control)",
    )
    s.add_argument("--n-boot", type=int, default=2000)
    _add_data_args(s)
    _add_signal_args(s)
    s.set_defaults(func=cmd_scan)

    w = sub.add_parser("portfolio", help="quarterly walk-forward multi-pair portfolio")
    w.add_argument("--tickers", nargs="+", default=list(DEFAULT_UNIVERSE))
    w.add_argument(
        "--modes",
        nargs="+",
        default=["equal_weight", "vol_target", "risk_managed"],
        choices=["equal_weight", "vol_target", "risk_managed"],
    )
    w.add_argument("--hedge", choices=["ols", "kalman"], default="ols")
    w.add_argument("--train-days", type=int, default=504)
    w.add_argument("--val-days", type=int, default=126)
    w.add_argument("--corr-threshold", type=float, default=0.6)
    w.add_argument("--fdr-q", type=float, default=0.10)
    w.add_argument("--top-k", type=int, default=5)
    w.add_argument("--target-vol", type=float, default=0.008, help="daily vol target per pair")
    w.add_argument(
        "--fallback",
        action="store_true",
        help="trade lowest p-values if nothing passes FDR (voids FDR control)",
    )
    w.add_argument("--show-quarters", type=int, default=8)
    w.add_argument("--out", default="reports/figures", help="output directory ('' to skip)")
    _add_data_args(w)
    _add_signal_args(w)
    w.set_defaults(func=cmd_portfolio)

    sw = sub.add_parser("sweep", help="parameter grid with a deflated-Sharpe haircut")
    sw.add_argument("--y", default=PairConfig.y_ticker)
    sw.add_argument("--x", default=PairConfig.x_ticker)
    sw.add_argument("--hedge", choices=["ols", "kalman"], default="ols")
    sw.add_argument("--train-lookback", type=int, default=PairConfig.train_lookback)
    sw.add_argument("--test-frac", type=float, default=1.0 - PairConfig.train_frac)
    sw.add_argument("--entry-grid", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0])
    sw.add_argument("--lookback-grid", type=int, nargs="+", default=[20, 40, 60, 90, 120])
    sw.add_argument("--out", default="reports/figures", help="figure directory ('' to skip)")
    _add_data_args(sw)
    _add_signal_args(sw)
    sw.set_defaults(func=cmd_sweep)

    pp = sub.add_parser("paper", help="replay history through the live runner + paper broker")
    pp.add_argument("--pairs", nargs="+", default=["KO/PEP", "XOM/CVX"])
    pp.add_argument("--capital", type=float, default=100_000.0)
    pp.add_argument("--notional", type=float, default=20_000.0, help="gross notional per pair")
    _add_data_args(pp)
    _add_signal_args(pp)
    pp.set_defaults(func=cmd_paper, start="2020-01-01")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
