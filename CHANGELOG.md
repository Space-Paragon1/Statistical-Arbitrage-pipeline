# Changelog

## 0.2.0: methodology audit

A line-by-line audit of v0.1 found several issues that changed results, not just code quality. Each is listed with its impact and the test that now guards it.

### Correctness fixes

| # | Issue in v0.1 | Impact | Fix | Guarded by |
|---|---|---|---|---|
| 1 | **Hedge-ratio units.** β came from a *price-level* regression (a share ratio) but was used as a *dollar* weight (`w_x = −β`). | For KO/PEP the share ratio is ~0.3 and the dollar ratio ~0.8, so roughly half the gross was an unintended net-long KO bet. The backtest did not trade the spread its z-score described. Same bug in `live/runner.py`. | `pair_weights` converts β to dollar weights with prices; the live runner sizes shares with ratio β. | `test_pair_pnl_equals_spread_pnl`, `test_target_shares_use_share_hedge_ratio` |
| 2 | **Lookahead in risk-managed allocation.** `correlation_aware_weights` was fit on the returns of the quarter being traded. | Portfolio weights used future returns. | Allocation estimated on TRAIN+VAL returns only. | `test_portfolio_walkforward_has_no_lookahead` |
| 3 | **Cointegration p-values.** Plain ADF on in-sample OLS residuals. | Anti-conservative: OLS picks the β that makes residuals look most stationary. KO/PEP over 2018–2025 shows naive p = 0.028 but Engle-Granger p = 0.091, which is *not* significant. | `engle_granger_test` uses MacKinnon cointegration critical values. | `test_engle_granger_is_more_conservative_than_naive_adf`, `test_engle_granger_rejects_independent_random_walks` |
| 4 | **FDR fallback.** If no pair passed Benjamini-Hochberg, the lowest p-values were traded anyway. | FDR control had no effect on what got traded. In practice ~26 of 28 quarters were fallback trades. | Strict by default; `fallback_to_lowest_p=True` / `--fallback` kept only for comparison. | `test_scan_report_strict_fdr` |
| 5 | **Drawdown breaker never re-armed.** It checked the worst-ever drawdown. | After one trip, trading was halted for the rest of the backtest. | Checks *current* drawdown; flattens for one quarter, then resumes from a fresh high-water mark. | `test_drawdown_breaker_rearms` |
| 6 | **Costs at quarter boundaries.** Each pair was backtested separately and sliced to the trade window. | Positions inherited from the validation period, or open at quarter end, were never charged entry/exit costs; no netting across pairs. | Pair books are summed into one asset-level weight matrix and simulated once. | `test_costs_charged_at_quarter_boundaries` |
| 7 | **Zero-padded warm-up.** The portfolio return series included ~2.5 years of zeros before the first trade window. | Diluted CAGR and Sharpe. | Results span the walk-forward period only. | `test_portfolio_runs_and_reports` |
| 8 | **"Ledoit-Wolf" was not Ledoit-Wolf.** It used an ad-hoc shrinkage intensity. | Mislabelled estimator. | Exact LW (2004) formula. | `test_ledoit_wolf_matches_sklearn` (1e-10) |
| 9 | **"Correlation-aware" weights ignored correlation.** Inverse-vol weights were scaled to a vol target, then renormalized (undoing the target). | Equivalent to capped inverse-vol. | Long-only minimum variance on the LW covariance. | `test_correlation_aware_weights_split_correlated_risk` |
| 10 | **Walk-forward path inconsistency.** Each day the state machine was re-run over all history with that day's β, so the "held" position came from a re-written past. Also O(n²). | Positions did not correspond to any path a live trader could follow; 2.2 s per 10-year backtest. | One persistent state machine; O(n) rolling moments give an identical z-score. **≈275× faster** (8 ms). | `test_time_varying_zscore_matches_full_recompute`, `test_live_signal_matches_backtest` |
| 11 | **Exit rule.** Exit required `abs(z) ≤ exit_z`. | A long entered at z = −3 stayed open if z jumped straight to +2.5. | One-sided exit: longs close once `z ≥ −exit_z`. | `test_overshoot_through_band_exits` |
| 12 | **IID bootstrap** on autocorrelated strategy returns. | Confidence intervals too narrow. | Stationary block bootstrap (default). | `test_block_bootstrap_is_wider_for_autocorrelated_returns` |
| 13 | **ML prefilter scaling.** `StandardScaler` was applied to `returns.T`, standardizing each *date* across tickers. | Distances were not correlation distances. | Per-ticker standardization over time (distance² = 2(1 − ρ)). | `test_features_encode_correlation_distance` |
| 14 | **Data loader** dropped every row where *any* ticker was missing. | One late-listing ticker silently truncated the whole panel. | Bounded forward-fill; leading NaNs kept; callers drop explicitly. | `test_yfinance_loader_caches_and_validates` |

### Before / after (same data, 2015–2024)

| | v0.1 | v0.2 |
|---|---|---|
| KO/PEP walk-forward OOS Sharpe (2022–2024) | −0.60 | −0.49, 95% CI [−1.61, 0.75] |
| Portfolio Sharpe: equal / vol-target / risk-managed | 0.02 / −0.05 / −0.39 *(fallback trades, zero-padded)* | −0.20 / −0.24 / −0.46 *(strict FDR, 6% in market)* |
| Same, with the v0.1 fallback behaviour | — | −0.50 / −0.44 / −0.43 |
| Single-pair walk-forward runtime | 2.2 s | 8 ms |
| Tests | 59 (30 s) | 113 (~15 s, 93% branch coverage), including lookahead-invariance and live-parity tests |

### New

* `sarb` CLI (`backtest`, `scan`, `portfolio`, `sweep`, `paper`) with `--prices-csv` for offline reproduction.
* Deflated / Probabilistic Sharpe Ratio; `sarb sweep` reports the DSR of the best parameter cell.
* Stop-loss (`stop_z`) and time stop (`max_holding`) with re-entry lockout.
* Kalman innovation variance and z-score; prior/posterior timing documented.
* Performance summary: Sortino, Calmar, drawdown duration, turnover, cost drag, trade statistics.
* Live runner carries per-pair state and nets orders across pairs sharing a ticker.
* On-disk price cache; tearsheet and heatmap figures; object-oriented matplotlib (no global backend switch).
* Packaging (`[build-system]`, typed, console script), GitHub Actions CI (3.10–3.13), ruff, mypy, pre-commit, Makefile.
* `bridgewater_submission.html`: results table, technical claims and charts corrected to match v0.2 (Engle-Granger p-values, corrected hedge units, no partial-fill or expanding-window claims).

### Breaking changes

* `walkforward_quarterly_portfolio` returns `PortfolioBacktest(returns, meta, weights)` instead of a 2-tuple.
* `correlation_aware_weights(returns, max_weight)` no longer takes `target_vol`.
* `RiskLimits.short_borrow_cost_bps` moved to `WFConfig.short_borrow_cost_bps` (it is a cost, not a limit).
* `scan_pairs` no longer falls back to the lowest p-values unless `fallback_to_lowest_p=True`.

## 0.1.0

Initial release.
