# -*- coding: utf-8 -*-
"""
Alpha Analytics - end-to-end analysis driver.

Quantic MSBA Capstone. Author: Yahia Aktham.

Runs the full pipeline and writes every figure quoted in the written report to
``results/``, so that no number in the report is hand-transcribed:

    python run_analysis.py --panel <full panel csv> [--ledger <brokerage csv>]

Outputs
-------
results/capm_estimates.csv        per-asset beta, standard error, t, R-squared
results/optimal_weights.csv       in-sample MSR and GMV weights
results/mrp_sensitivity.csv       optimiser output across market-risk-premium assumptions
results/backtest_summary.csv      walk-forward out-of-sample performance by strategy
results/backtest_returns.csv      daily out-of-sample return series by strategy
results/sharpe_tests.csv          Jobson-Korkie pairwise Sharpe tests
results/risk_metrics.csv          multivariate Monte Carlo VaR / CVaR
results/report_figures.md         every headline number, formatted for the report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import portfolio_engine as pe
import ledger_reconstruction as lr

# --- modelling assumptions, stated in one place ---------------------------------------
BASE_MRP = 0.045          # forward-looking equity risk premium, base case
MRP_GRID = (0.030, 0.045, 0.060, 0.1261)   # last value = the realised premium, for contrast
WEIGHT_CAP = 0.10         # per-asset cap (Jagannathan and Ma 2003)
BETA_SHRINK = 0.50        # Vasicek weight on the prior beta of 1.0
ESTIMATION_WINDOW = 252   # trailing days used at each rebalance
TRANSACTION_COST = 0.0010 # 10 bps one-way on traded notional
NORMALISED_CAPITAL = 100_000.0
CASH_TICKER = "BOXX"      # box-spread ETF, T-bill-like cash proxy -- see
                           # portfolio_engine.cash_aware_covariance / cash_aware_expected_returns

STRATEGIES = ("MVO_naive", "MSR_capm", "MSR_capm_bayes_stein",
              "MSR_capm_vasicek", "GMV_capped", "GMV_unconstrained", "EW_1overN")


def in_sample(panel: pd.DataFrame, out: Path, start: str | None = None) -> dict:
    """Headline estimation: CAPM diagnostics, covariance conditioning, optimiser output.

    ``start`` trades window length against universe breadth. Starting later admits assets
    with shorter listing histories (notably the cash proxy BOXX) at the cost of
    observations; the chosen default keeps N=46 of 47 assets with T>600 daily observations,
    so N/T stays well below 1 and the sample covariance matrix is non-singular before
    shrinkage.
    """
    if start:
        panel = panel.loc[start:]
    returns, market, rf_daily = pe.split_returns(panel)
    all_assets = list(returns.columns)
    balanced = [c for c in returns.columns if returns[c].iloc[1:].notna().all()]
    insufficient_history = sorted(set(all_assets) - set(balanced) - {CASH_TICKER})

    # --- BOXX cash-proxy handling -----------------------------------------------------
    # A bounded cash treatment was attempted (joint covariance with correlation-shrinkage
    # -toward-identity for the cash row, own-mean expected return, no beta-shrink-to-1) and
    # produces a symmetric, PSD covariance matrix with BOXX's modelled volatility close to
    # its measured volatility. It was NOT used here to ADMIT BOXX into the optimized weight
    # vectors: the required regression test (see test_portfolio_engine.py) fixes the
    # contract that BOXX must be absent from every optimized weights vector and present
    # only as a passive benchmark series, and simple exclusion is also the safer, auditable
    # choice for a headline result built on an unreconciled ledger fallback (see module
    # docstring / write_report_figures). BOXX is therefore excluded from the optimized
    # universe BY DESIGN (not for insufficient history), while its raw measured
    # beta/return/vol are still estimated and reported below for transparency.
    excluded_by_design = [CASH_TICKER] if CASH_TICKER in balanced else []
    opt_assets = [c for c in balanced if c != CASH_TICKER]
    fit_universe = balanced  # CAPM fit over balanced+cash so BOXX's raw beta is estimated too
    returns_fit = returns[fit_universe]
    fit = pe.estimate_capm(returns_fit, market, rf_daily)
    fit.table.to_csv(out / "capm_estimates.csv", float_format="%.6f")

    returns = returns[opt_assets]
    order = list(returns.columns)
    sigma, shrinkage, cond_sample, cond_shrunk = pe.shrunk_covariance(returns)
    rf = fit.risk_free_annual
    n_obs = len(returns.dropna())
    solver_failures: list[dict] = []

    cash_diag = {"cash_proxy": CASH_TICKER, "fallback_used": "simple_exclusion",
                "reason_for_fallback": "keeps BOXX out of optimized weight vectors by "
                                       "design, per the universe contract enforced by the "
                                       "regression tests; see comment above in_sample()"}
    if CASH_TICKER in fit.table.index:
        row = fit.table.loc[CASH_TICKER]
        cash_diag.update({
            "cash_beta_raw": float(row["beta"]), "cash_beta_t": float(row["t_beta"]),
            "cash_return_raw_hist_mean": float(row["realised_cagr"]),
            "cash_vol_raw": float(row["ann_vol"]),
            "cash_beta_modeled": None, "cash_return_modeled": None, "cash_vol_modeled": None,
            "corr_shrink_intensity_if_included": pe.CASH_CORR_SHRINK,
            "note": "modeled_* left null: BOXX excluded from optimization, not assigned a "
                    "model weight/risk contribution",
        })

    def _solve_msr(mu_used, label, extra):
        try:
            return pe.max_sharpe_weights(mu_used, sigma, rf, WEIGHT_CAP)
        except (pe.InfeasibleWeightCapError, pe.OptimizationFailedError) as exc:
            solver_failures.append({"stage": "in_sample_mrp_grid", "strategy": label,
                                    "reason": str(exc), **extra})
            return None

    rows, weights_out = [], {}
    for mrp in MRP_GRID:
        for label, betas in (("raw", fit.betas.reindex(order)),
                             ("vasicek", pe.shrink_betas_vasicek(fit.betas.reindex(order), BETA_SHRINK))):
            mu = pe.capm_expected_returns(betas, rf, mrp).to_numpy()
            for shrink_mu in (False, True):
                if shrink_mu:
                    mu_used, intensity = pe.shrink_mean_bayes_stein(mu, sigma, n_obs)
                else:
                    mu_used, intensity = mu, 0.0
                w = _solve_msr(mu_used, label, {"mrp": mrp, "bayes_stein": shrink_mu})
                if w is None:
                    continue
                ret, vol, sharpe = pe.portfolio_moments(w, mu_used, sigma, rf)
                rows.append({"mrp": mrp, "betas": label, "bayes_stein": shrink_mu,
                             "bs_intensity": intensity, "exp_return": ret,
                             "volatility": vol, "sharpe": sharpe,
                             "max_weight": float(w.max()),
                             "n_positions": int((w > 0.005).sum())})
                if mrp == BASE_MRP and label == "vasicek" and not shrink_mu:
                    weights_out["MSR"] = pd.Series(w, index=order)
    pd.DataFrame(rows).to_csv(out / "mrp_sensitivity.csv", index=False, float_format="%.6f")

    if "MSR" not in weights_out:
        raise pe.OptimizationFailedError(
            "base-case MSR solve failed; see solver_failures", None)
    try:
        gmv_w = pe.min_variance_weights(sigma, WEIGHT_CAP)
    except (pe.InfeasibleWeightCapError, pe.OptimizationFailedError) as exc:
        solver_failures.append({"stage": "in_sample_gmv", "strategy": "GMV", "reason": str(exc)})
        raise
    weights_out["GMV"] = pd.Series(gmv_w, index=order)
    weights_df = pd.DataFrame(weights_out)
    weights_df.index.name = "Ticker"
    weights_df.to_csv(out / "optimal_weights.csv", float_format="%.6f")

    betas_v = pe.shrink_betas_vasicek(fit.betas.reindex(order), BETA_SHRINK)
    mu_base = pe.capm_expected_returns(betas_v, rf, BASE_MRP).to_numpy()
    msr_moments = pe.portfolio_moments(weights_out["MSR"].to_numpy(), mu_base, sigma, rf)
    gmv_moments = pe.portfolio_moments(weights_out["GMV"].to_numpy(), mu_base, sigma, rf)

    paths = pe.simulate_gbm_paths(weights_out["MSR"].to_numpy(), mu_base, sigma,
                                  capital=NORMALISED_CAPITAL)
    var95, cvar95 = pe.value_at_risk(NORMALISED_CAPITAL, paths, 0.95)
    terminal = paths[-1]
    pd.DataFrame([{
        "capital": NORMALISED_CAPITAL, "confidence": 0.95,
        "var_dollar": var95, "cvar_dollar": cvar95,
        "prob_capital_preserved": float((terminal > NORMALISED_CAPITAL).mean()),
        "median_terminal": float(np.median(terminal)),
        "p05_terminal": float(np.percentile(terminal, 5)),
        "p95_terminal": float(np.percentile(terminal, 95)),
    }]).to_csv(out / "risk_metrics.csv", index=False, float_format="%.2f")

    return {
        "window": (panel.index.min().date().isoformat(), panel.index.max().date().isoformat()),
        "n_obs": n_obs, "n_assets": len(returns.columns),
        "n_over_t": len(returns.columns) / n_obs,
        "excluded": {"insufficient_history": insufficient_history,
                    "excluded_by_design": excluded_by_design},
        "risk_free": rf, "market_cagr": fit.market_return_cagr,
        "market_vol": float(market.std() * np.sqrt(pe.TRADING_DAYS)),
        "realised_mrp": fit.market_return_cagr - rf,
        "lw_shrinkage": shrinkage, "cond_sample": cond_sample, "cond_shrunk": cond_shrunk,
        "beta_min": float(fit.betas.min()), "beta_max": float(fit.betas.max()),
        "beta_median_abs_t": float(fit.table.t_beta.abs().median()),
        "beta_share_significant": float((fit.table.t_beta.abs() > 1.96).mean()),
        "beta_median_r2": float(fit.table.r_squared.median()),
        "msr": msr_moments, "gmv": gmv_moments,
        "var95": var95, "cvar95": cvar95,
        "prob_preserved": float((terminal > NORMALISED_CAPITAL).mean()),
        "cash_diagnostics": cash_diag,
        "solver_failures": solver_failures,
    }


def _drift_holding_period(forward: pd.DataFrame, target_weights: np.ndarray
                          ) -> tuple[pd.Series, np.ndarray]:
    """Simulate true buy-and-hold weight drift within one monthly holding period.

    Weights start at ``target_weights`` (the post-rebalance allocation) and are NOT
    rebalanced back to target during the period -- each asset's weight drifts with its own
    cumulative return, as it would in a real unmanaged account. Same-close decision/execution
    convention as the rest of this module: the rebalance is assumed to trade at the same
    close used to compute the first forward-period return, i.e. no separate execution-lag
    day is modelled.

    Returns (daily portfolio return series, end-of-period drifted weights).
    """
    w = np.asarray(target_weights, dtype=float).copy()
    port_returns = []
    for _, row in forward.iterrows():
        r = np.nan_to_num(row.to_numpy(), nan=0.0)
        port_returns.append(float(w @ r))
        grown = w * (1.0 + r)
        total = grown.sum()
        if total > 0:
            w = grown / total
        # else: degenerate (portfolio value hit zero); hold weights unchanged rather than
        # divide by zero -- should not occur with real return data.
    return pd.Series(port_returns, index=forward.index), w


def walk_forward(panel: pd.DataFrame, actual: pd.Series | None, out: Path) -> dict:
    """Rolling-window out-of-sample validation with monthly rebalancing.

    Within each holding month, weights drift with realised asset returns rather than being
    rebalanced to target every day (see :func:`_drift_holding_period`); turnover at the next
    rebalance is the L1 distance between the new target weights and the DRIFTED end-of-period
    holdings from the previous period, over the union of the two periods' eligible assets --
    not just the previous period's target weights, which understates turnover whenever prices
    moved between rebalances.
    """
    returns, market, rf_daily = pe.split_returns(panel)
    month_ends = returns.index.to_series().groupby([returns.index.year, returns.index.month]).last()
    rebalances = [d for d in month_ends if d >= returns.index[0] + pd.Timedelta(days=400)]

    series = {s: [] for s in STRATEGIES}
    turnover = {s: [] for s in STRATEGIES}
    n_pos = {s: [] for s in STRATEGIES}
    max_w = {s: [] for s in STRATEGIES}
    previous = {s: None for s in STRATEGIES}  # (drifted end-of-period weights, eligible assets)
    universe_sizes = []
    solver_failures: list[dict] = []

    for i, date in enumerate(rebalances[:-1]):
        nxt = rebalances[i + 1]
        history = returns.loc[:date].tail(ESTIMATION_WINDOW)
        # BOXX (cash proxy) is excluded from the OPTIMIZED universe by design here too --
        # see the fallback note in in_sample() -- but stays eligible as a passive benchmark
        # column (combined["BOXX_cash"] below) with its own raw returns, untouched by the
        # risky-asset estimation pipeline.
        eligible_all = [c for c in history.columns if history[c].notna().all()]
        eligible = [c for c in eligible_all if c != CASH_TICKER]
        if len(eligible) < 10:
            continue
        universe_sizes.append(len(eligible))
        hist = history[eligible]
        rf_ann = float((rf_daily.loc[hist.index] * pe.TRADING_DAYS).mean())

        fit = pe.estimate_capm(hist, market.loc[hist.index], rf_daily.loc[hist.index], eligible)
        betas = fit.betas.reindex(eligible)
        sigma_lw, _, _, _ = pe.shrunk_covariance(hist)
        sigma_raw = np.cov(hist.dropna().to_numpy(), rowvar=False) * pe.TRADING_DAYS
        mu_sample = ((1.0 + hist).prod() ** (pe.TRADING_DAYS / len(hist)) - 1.0).to_numpy()
        mu_capm = pe.capm_expected_returns(betas, rf_ann, BASE_MRP).to_numpy()
        mu_vas = pe.capm_expected_returns(pe.shrink_betas_vasicek(betas, BETA_SHRINK),
                                          rf_ann, BASE_MRP).to_numpy()
        mu_bs, _ = pe.shrink_mean_bayes_stein(mu_capm, sigma_lw, len(hist))

        def _solve(fn, *args, strategy=None, **kwargs):
            try:
                return fn(*args, **kwargs)
            except (pe.InfeasibleWeightCapError, pe.OptimizationFailedError) as exc:
                solver_failures.append({"stage": "walk_forward", "strategy": strategy,
                                        "date": date.date().isoformat(), "reason": str(exc)})
                return None

        chosen_raw = {
            "MVO_naive": _solve(pe.max_sharpe_weights, mu_sample, sigma_raw, rf_ann, 1.0,
                                strategy="MVO_naive"),
            "MSR_capm": _solve(pe.max_sharpe_weights, mu_capm, sigma_lw, rf_ann, WEIGHT_CAP,
                               strategy="MSR_capm"),
            "MSR_capm_bayes_stein": _solve(pe.max_sharpe_weights, mu_bs, sigma_lw, rf_ann, WEIGHT_CAP,
                                           strategy="MSR_capm_bayes_stein"),
            "MSR_capm_vasicek": _solve(pe.max_sharpe_weights, mu_vas, sigma_lw, rf_ann, WEIGHT_CAP,
                                       strategy="MSR_capm_vasicek"),
            "GMV_capped": _solve(pe.min_variance_weights, sigma_lw, WEIGHT_CAP, strategy="GMV_capped"),
            "GMV_unconstrained": _solve(pe.min_variance_weights, sigma_lw, 1.0,
                                        strategy="GMV_unconstrained"),
            "EW_1overN": np.full(len(eligible), 1.0 / len(eligible)),
        }
        forward = returns.loc[(returns.index > date) & (returns.index <= nxt), eligible]
        forward = forward.fillna(0.0)
        for name, w in chosen_raw.items():
            if w is None:
                # Solver failed for this strategy this period: log it (already appended
                # above) and carry the previous drifted weights forward unchanged rather
                # than fabricating a new optimized portfolio. If there is no previous
                # period (first rebalance), fall back explicitly to equal weight and label
                # it as such in the failure log.
                if previous[name] is not None:
                    prev_w, prev_assets = previous[name]
                    w = pd.Series(prev_w, index=prev_assets).reindex(eligible).fillna(0.0).to_numpy()
                    total = w.sum()
                    w = w / total if total > 0 else np.full(len(eligible), 1.0 / len(eligible))
                else:
                    w = np.full(len(eligible), 1.0 / len(eligible))
                    solver_failures.append({"stage": "walk_forward", "strategy": name,
                                            "date": date.date().isoformat(),
                                            "reason": "no prior weights to carry forward; "
                                                      "used equal-weight fallback (logged, not silent)"})
            gross, end_w = _drift_holding_period(forward, w)
            if previous[name] is not None:
                prev_end_w, prev_assets = previous[name]
                old_series = pd.Series(prev_end_w, index=prev_assets)
                new_series = pd.Series(w, index=eligible)
                union_idx = old_series.index.union(new_series.index)
                old_aligned = old_series.reindex(union_idx).fillna(0.0).to_numpy()
                new_aligned = new_series.reindex(union_idx).fillna(0.0).to_numpy()
                traded = float(np.abs(new_aligned - old_aligned).sum() / 2.0)  # one-way notional
                turnover[name].append(traded)
                if len(gross):
                    gross.iloc[0] -= 2.0 * traded * TRANSACTION_COST  # cost on buy+sell notional
            series[name].append(gross)
            n_pos[name].append(int((w > 0.005).sum()))
            max_w[name].append(float(w.max()))
            previous[name] = (end_w, eligible)

    combined = {s: pd.concat(v) for s, v in series.items()}
    index = combined["EW_1overN"].index
    rf_slice = rf_daily.loc[index]
    combined["SPY_passive"] = market.loc[index]
    combined["BOXX_cash"] = returns["BOXX"].loc[index] if "BOXX" in returns else pd.Series(dtype=float)
    if actual is not None:
        combined["ACTUAL_discretionary"] = actual.reindex(index).fillna(0.0)

    pd.DataFrame(combined).to_csv(out / "backtest_returns.csv", float_format="%.8f")

    rows = []
    for name, s in combined.items():
        if s.dropna().empty:
            continue
        p = pe.performance(s, rf_slice)
        rows.append({"strategy": name, "ann_return": p.ann_return, "ann_vol": p.ann_vol,
                     "sharpe": p.sharpe, "max_drawdown": p.max_drawdown,
                     "total_return": p.total_return, "n_days": p.n_days,
                     "turnover_per_month": float(np.mean(turnover[name])) if turnover.get(name) else np.nan,
                     "avg_positions": float(np.mean(n_pos[name])) if n_pos.get(name) else np.nan,
                     "avg_max_weight": float(np.mean(max_w[name])) if max_w.get(name) else np.nan})
    summary = pd.DataFrame(rows).set_index("strategy")
    summary.to_csv(out / "backtest_summary.csv", float_format="%.6f")

    pairs = [("EW_1overN", "GMV_capped"), ("EW_1overN", "MSR_capm_vasicek"),
             ("MSR_capm_vasicek", "MVO_naive"), ("GMV_capped", "SPY_passive")]
    if "ACTUAL_discretionary" in combined:
        pairs += [("GMV_capped", "ACTUAL_discretionary"), ("EW_1overN", "ACTUAL_discretionary")]
    tests = []
    for a, b in pairs:
        if a in combined and b in combined and not combined[b].dropna().empty:
            s1, s2, z, p = pe.sharpe_difference_test(combined[a], combined[b], rf_slice)
            tests.append({"strategy_a": a, "sharpe_a": s1, "strategy_b": b, "sharpe_b": s2,
                          "z": z, "p_value": p, "significant_5pct": p < 0.05})
    pd.DataFrame(tests).to_csv(out / "sharpe_tests.csv", index=False, float_format="%.6f")

    return {"oos_start": index.min().date().isoformat(), "oos_end": index.max().date().isoformat(),
            "oos_days": len(index), "n_rebalances": len(universe_sizes),
            "universe_min": min(universe_sizes), "universe_max": max(universe_sizes),
            "rf_oos": float((rf_slice * pe.TRADING_DAYS).mean()),
            "summary": summary, "tests": pd.DataFrame(tests),
            "solver_failures": solver_failures,
            "actual_discretionary_included": actual is not None}


def write_environment(out: Path) -> None:
    """Record the exact environment that produced these results.

    Pinning package versions is not sufficient for bit-exact reproduction of the
    unregularised mean-variance baseline: that solve is ill-conditioned and its result
    depends on the platform's BLAS/LAPACK kernel selection, so the same OpenBLAS build on
    Windows and on Linux returns different values for that one row. Recording the backend
    is what lets a reader tell whether a discrepancy is expected.
    """
    import platform
    import sys

    lines = ["# Environment used to generate these results", "",
             "Pinning `requirements.txt` is necessary but not sufficient to reproduce every",
             "figure exactly. See report Section 18.4: the unregularised mean-variance",
             "baseline depends on the platform's BLAS/LAPACK kernel selection. Every",
             "regularised strategy and every Monte Carlo risk metric is platform-independent.",
             "", "## Platform", "",
             f"- Operating system: {platform.platform()}",
             f"- Machine: {platform.machine()}",
             f"- Python: {sys.version.split()[0]}", "", "## Packages", ""]
    for module in ("numpy", "pandas", "scipy", "sklearn", "matplotlib"):
        try:
            lines.append(f"- {module}: {__import__(module).__version__}")
        except Exception:
            lines.append(f"- {module}: not importable")

    lines += ["", "## Linear algebra backend", ""]
    try:
        config = np.show_config(mode="dicts") or {}
        deps = config.get("Build Dependencies", {})
        for key in ("blas", "lapack"):
            info = deps.get(key, {})
            lines.append(f"- {key}: {info.get('name', 'unknown')} "
                         f"{info.get('version', '')}".rstrip())
    except Exception as exc:
        lines.append(f"- numpy.show_config unavailable: {exc}")

    (out / "environment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_submission_metrics(out: Path, ins: dict, oos: dict, ledger,
                             include_legacy_actual_account: bool) -> None:
    """Structured, machine-readable version of the headline figures for the report sync."""
    summary = oos["summary"].reset_index().to_dict(orient="records")
    payload = {
        "asset_universe": {
            "n_optimized_assets": ins["n_assets"],
            "excluded_insufficient_history": ins["excluded"]["insufficient_history"],
            "excluded_by_design": ins["excluded"]["excluded_by_design"],
        },
        "sample_window": {"in_sample": ins["window"],
                          "out_of_sample": (oos["oos_start"], oos["oos_end"])},
        "cash_proxy_boxx": ins.get("cash_diagnostics", {}),
        "strategies": summary,
        "sharpe_tests": oos["tests"].to_dict(orient="records"),
        "solver_failures": {"in_sample": ins.get("solver_failures", []),
                            "walk_forward": oos.get("solver_failures", [])},
        "actual_discretionary_account": {
            "included_in_headline_results": include_legacy_actual_account,
            "reason_if_excluded": None if include_legacy_actual_account else
                "no independent broker-reported time-weighted-return statement exists; "
                "only the raw transaction ledger is available, so this whole-account "
                "performance/concentration comparison is kept out of headline/main-run "
                "results. Ledger parsing "
                "(counts, fees, corporate actions) still ran and is reported separately "
                "as a data-engineering capability, not a performance claim.",
            "ledger_parsed": ledger is not None,
        },
    }
    (out / "submission_metrics.json").write_text(json.dumps(payload, indent=2, default=str),
                                                  encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    ap.add_argument("--panel", default=str(here.parent / "Sources" / "price_panel_v2_adjusted_2022-2026.csv"))
    ap.add_argument("--ledger", default=None, help="brokerage CSV export (kept out of this repo)")
    ap.add_argument("--insample-start", default="2024-01-26",
                    help="start date for the headline estimation window (see in_sample docstring)")
    ap.add_argument("--out", default=str(here / "results"))
    ap.add_argument("--include-legacy-actual-account", action="store_true", default=False,
                    help="Include the unreconciled ledger-derived ACTUAL_discretionary "
                         "return series (and its whole-account concentration figures) in "
                         "the main walk-forward output and Sharpe tests. OFF by default: "
                         "there is no independent broker-reported time-weighted-return "
                         "statement to reconcile the ledger reconstruction against, only "
                         "the raw transaction ledger, so this performance/concentration "
                         "comparison is kept out of headline/main-run results by default. "
                         "Ledger PARSING still runs "
                         "unconditionally and is reported as a data-engineering capability "
                         "(transaction counts, fees, corporate actions) -- see "
                         "'Transaction ledger' in report_figures.md -- it is only the "
                         "derived PERFORMANCE claim that is gated by this flag.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    panel = pe.load_panel(args.panel)
    returns, _, _ = pe.split_returns(panel)

    ledger_summary = None
    actual_returns = None
    if args.ledger and Path(args.ledger).exists():
        tx = lr.read_ledger(args.ledger)
        ledger_summary = lr.summarise(tx)
        ledger_summary.corporate_actions.to_csv(out / "corporate_actions.csv", index=False)
        # Holdings/weights/time-weighted-return reconstruction is diagnostic-only unless
        # explicitly opted into: it is an unreconciled performance claim (no independent
        # broker time-weighted-return statement exists to check it against).
        holdings = lr.build_holdings(tx, panel.index, list(returns.columns))
        weights = lr.portfolio_weights(holdings, panel)
        diagnostic_returns = lr.time_weighted_returns(weights, returns)
        weights.iloc[[-1]].T.rename(columns={weights.index[-1]: "weight"}) \
            .sort_values("weight", ascending=False).to_csv(
                out / "diagnostic_actual_portfolio_weights.csv", float_format="%.6f")
        if args.include_legacy_actual_account:
            actual_returns = diagnostic_returns
            # Also write under the legacy filename write_report_figures looks for, so the
            # concentration section only appears in the report when explicitly requested.
            weights.iloc[[-1]].T.rename(columns={weights.index[-1]: "weight"}) \
                .sort_values("weight", ascending=False).to_csv(
                    out / "actual_portfolio_weights.csv", float_format="%.6f")

    ins = in_sample(panel, out, args.insample_start)
    oos = walk_forward(panel, actual_returns, out)

    write_report_figures(out, ins, oos, ledger_summary,
                         include_legacy_actual_account=args.include_legacy_actual_account)
    write_environment(out)
    write_submission_metrics(out, ins, oos, ledger_summary, args.include_legacy_actual_account)
    print(f"wrote analysis outputs to {out}")


def write_report_figures(out: Path, ins: dict, oos: dict, ledger, *,
                         include_legacy_actual_account: bool = False) -> None:
    """Emit every headline figure the report quotes, so nothing is hand-transcribed."""
    s = oos["summary"]
    excl = ins["excluded"]
    cash = ins.get("cash_diagnostics", {})
    lines = [
        "# Report figures (generated - do not hand-edit)", "",
        f"Estimation window {ins['window'][0]} to {ins['window'][1]}.", "",
        "## Data and estimation",
        f"- Daily observations T = {ins['n_obs']}; assets N = {ins['n_assets']}; "
        f"N/T = {ins['n_over_t']:.3f}",
        f"- Excluded for insufficient history: {', '.join(excl['insufficient_history']) or 'none'}",
        f"- Excluded by design (not a history gap): {', '.join(excl['excluded_by_design']) or 'none'}",
        f"- Risk-free rate (13-week T-bill, mean) = {ins['risk_free']*100:.2f}%",
        f"- Market proxy realised CAGR = {ins['market_cagr']*100:.2f}%, "
        f"volatility = {ins['market_vol']*100:.2f}%",
        f"- Realised market risk premium over the window = {ins['realised_mrp']*100:.2f}% "
        f"(NOT used as the forward estimate; base case is {BASE_MRP*100:.1f}%)",
        "",
        "## Beta estimation quality",
        f"- Beta range {ins['beta_min']:.2f} to {ins['beta_max']:.2f}",
        f"- Median |t| = {ins['beta_median_abs_t']:.1f}; "
        f"{ins['beta_share_significant']*100:.0f}% significant at 5%",
        f"- Median R-squared = {ins['beta_median_r2']:.3f}",
        "",
        "## Covariance conditioning",
        f"- Ledoit-Wolf shrinkage intensity = {ins['lw_shrinkage']:.4f}",
        f"- Condition number: {ins['cond_sample']:.3e} (sample) to {ins['cond_shrunk']:.2f} (shrunk)",
        "",
        "## In-sample optimiser output (base case)",
        f"- MSR: expected return {ins['msr'][0]*100:.2f}%, volatility {ins['msr'][1]*100:.2f}%, "
        f"Sharpe {ins['msr'][2]:.2f}",
        f"- GMV: expected return {ins['gmv'][0]*100:.2f}%, volatility {ins['gmv'][1]*100:.2f}%, "
        f"Sharpe {ins['gmv'][2]:.2f}",
        "",
        "## BOXX cash-proxy treatment",
        f"- BOXX excluded from optimized universe by design (fallback: {cash.get('fallback_used', 'n/a')})",
        f"- Raw measured beta = {cash.get('cash_beta_raw', float('nan')):.3f} "
        f"(t = {cash.get('cash_beta_t', float('nan')):.2f}); "
        f"raw historical mean return = {cash.get('cash_return_raw_hist_mean', float('nan'))*100:.2f}%; "
        f"raw volatility = {cash.get('cash_vol_raw', float('nan'))*100:.2f}%",
        f"- Modeled beta/return/vol: n/a (not assigned a model weight or risk contribution "
        f"once excluded); correlation-shrink intensity that WOULD apply if BOXX were "
        f"admitted to a joint covariance = {cash.get('corr_shrink_intensity_if_included', float('nan'))}",
        "",
        "## Solver failures (logged explicitly, never silently substituted)",
        (f"- {len(ins.get('solver_failures', [])) + len(oos.get('solver_failures', []))} "
         f"total ({len(ins.get('solver_failures', []))} in-sample, "
         f"{len(oos.get('solver_failures', []))} walk-forward)"
         if (ins.get("solver_failures") or oos.get("solver_failures"))
         else "- none"),
        "",
        "## Multivariate Monte Carlo (10,000 paths, 252 days, $100,000)",
        f"- 95% VaR = ${ins['var95']:,.0f}  (negative denotes a gain at the loss quantile)",
        f"- 95% CVaR = ${ins['cvar95']:,.0f}",
        f"- P(terminal value > initial capital) = {ins['prob_preserved']*100:.1f}%",
        "",
        "## Walk-forward out-of-sample results",
        f"OOS {oos['oos_start']} to {oos['oos_end']}, {oos['oos_days']} trading days, "
        f"{oos['n_rebalances']} monthly rebalances, universe "
        f"{oos['universe_min']}-{oos['universe_max']} assets, "
        f"risk-free {oos['rf_oos']*100:.2f}%, transaction cost "
        f"{TRANSACTION_COST*1e4:.0f} bps one-way.", "",
        "| Strategy | Ann. return | Ann. vol | Sharpe | Max DD | Turnover/mo | Positions | Max weight |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in s.iterrows():
        def fmt(v, pct=True, dp=2):
            if pd.isna(v):
                return "-"
            return f"{v*100:.{dp}f}%" if pct else f"{v:.{dp}f}"
        lines.append(f"| {name} | {fmt(r.ann_return)} | {fmt(r.ann_vol)} | "
                     f"{fmt(r.sharpe, pct=False)} | {fmt(r.max_drawdown)} | "
                     f"{fmt(r.turnover_per_month)} | {fmt(r.avg_positions, pct=False, dp=1)} | "
                     f"{fmt(r.avg_max_weight)} |")
    lines += ["", "## Sharpe-difference tests (Jobson-Korkie, Memmel correction)", "",
              "| A | Sharpe A | B | Sharpe B | z | p | Significant |", "|---|---|---|---|---|---|---|"]
    for _, r in oos["tests"].iterrows():
        lines.append(f"| {r.strategy_a} | {r.sharpe_a:.2f} | {r.strategy_b} | {r.sharpe_b:.2f} | "
                     f"{r.z:+.2f} | {r.p_value:.3f} | {'yes' if r.significant_5pct else 'no'} |")
    concentration = out / "actual_portfolio_weights.csv"
    if include_legacy_actual_account and concentration.exists():
        w = pd.read_csv(concentration, index_col=0).iloc[:, 0]
        tech = {"MU", "AMD", "TSM", "MXL", "CRDO", "ICHR", "TTMI", "RXT", "LITE",
                "ASX", "QQQ", "SNDK", "PENG"}
        top = w.sort_values(ascending=False)
        lines += ["", "## Actual discretionary portfolio concentration "
                      "(LEGACY, opted-in via --include-legacy-actual-account)",
                  "**Unreconciled**: no independent broker time-weighted-return statement "
                  "exists to check this against, only the raw ledger. Not a headline result.",
                  f"- Positions above 0.5% of market value: {int((w > 0.005).sum())}",
                  f"- Largest single position: {w.max()*100:.1f}% ({top.index[0]})",
                  f"- Top five positions: {top.head(5).sum()*100:.1f}%",
                  f"- Herfindahl-Hirschman index: {(w**2).sum():.4f}",
                  f"- High-beta technology and semiconductor share: "
                  f"{w[[t for t in w.index if t in tech]].sum()*100:.1f}%"]
    elif ledger is not None:
        lines += ["", "## Actual discretionary portfolio / ledger performance",
                  "- DISABLED from the headline run: there is no independent broker-reported "
                  "time-weighted-return statement, only the raw transaction ledger, so the "
                  "whole-account performance and concentration comparison is excluded from "
                  "main results by default. Ledger parsing "
                  "itself still ran (see Transaction ledger section below) as a demonstrated "
                  "data-engineering capability. Re-run with "
                  "--include-legacy-actual-account to reinstate the diagnostic-only figures."]
    if ledger is not None:
        lines += ["", "## Transaction ledger (real brokerage export)",
                  f"- {ledger.n_rows} transactions, {ledger.n_symbols} distinct symbols, "
                  f"{ledger.first_date.date()} to {ledger.last_date.date()}",
                  f"- {ledger.n_buys} buys, {ledger.n_sells} sells",
                  f"- Commissions charged on {ledger.n_commissioned_trades} transactions "
                  f"(buys, sells and one short sale): "
                  f"${ledger.commission_min:.2f} to ${ledger.commission_max:.2f}, "
                  f"total ${ledger.commission_total:.2f}",
                  f"- Symbols requiring inferred starting balances: "
                  f"{len(ledger.inferred_starting_balances)}",
                  f"- Corporate actions in ledger: {len(ledger.corporate_actions)} "
                  f"(see corporate_actions.csv)"]
    (out / "report_figures.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
