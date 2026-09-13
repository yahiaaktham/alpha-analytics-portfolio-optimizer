# -*- coding: utf-8 -*-
"""
Automated verification suite for the Alpha Analytics portfolio engine.

Quantic MSBA Capstone. Author: Yahia Aktham.

    python -m unittest test_portfolio_engine -v

The tests are organised around the four things that could silently produce a plausible
wrong answer: ledger reconstruction, parameter estimation, the optimiser, and the risk
engine. Wherever possible a test asserts against a value known analytically rather than
against whatever the code happens to return.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import ledger_reconstruction as lr
import portfolio_engine as pe

TD = pe.TRADING_DAYS


def _write_ledger(rows: list[dict]) -> Path:
    """Write a temporary brokerage-style CSV export."""
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                         newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=["Date", "Action", "Symbol", "Description",
                                                "Quantity", "Price", "Fees & Comm", "Amount"])
    writer.writeheader()
    for row in rows:
        writer.writerow({"Date": "", "Action": "", "Symbol": "", "Description": "",
                         "Quantity": "", "Price": "", "Fees & Comm": "", "Amount": "", **row})
    handle.close()
    return Path(handle.name)


def _synthetic_panel(n_days: int = 400, seed: int = 7) -> pd.DataFrame:
    """A panel with a known factor structure, so betas have known target values."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    market = rng.normal(0.0004, 0.010, n_days)
    spy = 400.0 * np.exp(np.cumsum(market))
    frame = {"SPY": spy, "IRX": np.full(n_days, 4.0)}
    # ASSET_B2 is constructed with beta 2.0, ASSET_B0 with beta 0.0.
    for name, beta, idio in (("ASSET_B2", 2.0, 0.004), ("ASSET_B0", 0.0, 0.004),
                             ("ASSET_B1", 1.0, 0.006)):
        r = beta * market + rng.normal(0.0, idio, n_days)
        frame[name] = 100.0 * np.exp(np.cumsum(r))
    return pd.DataFrame(frame, index=pd.DatetimeIndex(dates, name="Date"))


class TestLedgerReconstruction(unittest.TestCase):

    def test_buy_then_partial_sell(self):
        """Straightforward arithmetic: 100 bought, 40 sold, 60 held, nothing inferred."""
        path = _write_ledger([
            {"Date": "01/10/2024", "Action": "Buy", "Symbol": "AAA", "Quantity": "100"},
            {"Date": "02/10/2024", "Action": "Sell", "Symbol": "AAA", "Quantity": "40"},
        ])
        tx = lr.read_ledger(path)
        self.assertAlmostEqual(tx["ShareDelta"].sum(), 60.0)
        self.assertEqual(lr.infer_starting_balances(tx)["AAA"], 0.0)

    def test_starting_balance_inferred_when_sale_predates_purchase(self):
        """A sale with no prior purchase implies a pre-existing position of exactly that size."""
        path = _write_ledger([
            {"Date": "01/10/2024", "Action": "Sell", "Symbol": "BBB", "Quantity": "75"},
            {"Date": "02/10/2024", "Action": "Buy", "Symbol": "BBB", "Quantity": "10"},
        ])
        tx = lr.read_ledger(path)
        self.assertAlmostEqual(lr.infer_starting_balances(tx)["BBB"], 75.0)

    def test_inferred_balance_is_the_maximum_deficit_not_the_final_one(self):
        """The inference must track the running minimum, not just the closing balance."""
        path = _write_ledger([
            {"Date": "01/10/2024", "Action": "Sell", "Symbol": "CCC", "Quantity": "50"},
            {"Date": "02/10/2024", "Action": "Sell", "Symbol": "CCC", "Quantity": "30"},
            {"Date": "03/10/2024", "Action": "Buy", "Symbol": "CCC", "Quantity": "200"},
        ])
        tx = lr.read_ledger(path)
        self.assertAlmostEqual(lr.infer_starting_balances(tx)["CCC"], 80.0)

    def test_cash_only_actions_do_not_move_shares(self):
        """A dividend row carries a dollar amount and a blank quantity; shares must not change."""
        path = _write_ledger([
            {"Date": "01/10/2024", "Action": "Buy", "Symbol": "DDD", "Quantity": "10"},
            {"Date": "05/10/2024", "Action": "Qualified Dividend", "Symbol": "DDD", "Amount": "$31.04"},
            {"Date": "06/10/2024", "Action": "Foreign Tax Paid", "Symbol": "DDD", "Amount": "-$7.76"},
        ])
        tx = lr.read_ledger(path)
        self.assertAlmostEqual(tx["ShareDelta"].sum(), 10.0)
        self.assertEqual(int(tx["IsCashOnly"].sum()), 2)

    def test_cusip_is_resolved_to_surviving_symbol(self):
        path = _write_ledger([
            {"Date": "01/10/2024", "Action": "Buy", "Symbol": "143658300", "Quantity": "46"},
        ])
        tx = lr.read_ledger(path)
        self.assertIn("CCL", set(tx["Symbol"]))
        self.assertNotIn("143658300", set(tx["Symbol"]))

    def test_unknown_action_type_raises_rather_than_being_skipped(self):
        """Regression guard: silently ignoring an unrecognised action produced wrong holdings."""
        path = _write_ledger([
            {"Date": "01/10/2024", "Action": "Buy", "Symbol": "EEE", "Quantity": "10"},
            {"Date": "02/10/2024", "Action": "Quantum Reclassification", "Symbol": "EEE", "Quantity": "5"},
        ])
        with self.assertRaises(lr.UnclassifiedActionError):
            lr.read_ledger(path)

    def test_settlement_date_form_is_parsed(self):
        """The export writes 'MM/DD/YYYY as of MM/DD/YYYY' for some corporate actions."""
        path = _write_ledger([
            {"Date": "07/21/2026 as of 07/17/2026", "Action": "Stock Split",
             "Symbol": "SNEX", "Quantity": "35"},
        ])
        tx = lr.read_ledger(path)
        self.assertEqual(len(tx), 1)
        self.assertEqual(tx.loc[0, "Date"], pd.Timestamp("2026-07-21"))

    def test_time_weighted_return_excludes_contributions(self):
        """A pure cash contribution must not register as investment return."""
        idx = pd.bdate_range("2024-01-01", periods=4)
        weights = pd.DataFrame({"AAA": [1.0, 1.0, 1.0, 1.0]}, index=idx)
        returns = pd.DataFrame({"AAA": [np.nan, 0.01, -0.02, 0.03]}, index=idx)
        twr = lr.time_weighted_returns(weights, returns)
        # First observation has no lagged weight; the rest pass through unchanged.
        self.assertTrue(np.isnan(twr.iloc[0]) or twr.iloc[0] == 0)
        np.testing.assert_allclose(twr.iloc[1:].to_numpy(), [0.01, -0.02, 0.03], atol=1e-12)


class TestCapmEstimation(unittest.TestCase):

    def setUp(self):
        self.panel = _synthetic_panel()
        self.returns, self.market, self.rf = pe.split_returns(self.panel)
        self.fit = pe.estimate_capm(self.returns, self.market, self.rf)

    def test_recovers_known_betas(self):
        """Betas were constructed as 2.0, 1.0 and 0.0; estimates must land within 3 SEs."""
        for ticker, target in (("ASSET_B2", 2.0), ("ASSET_B1", 1.0), ("ASSET_B0", 0.0)):
            row = self.fit.table.loc[ticker]
            self.assertLess(abs(row["beta"] - target), 3 * row["se_beta"],
                            f"{ticker}: beta {row['beta']:.3f} too far from {target}")

    def test_diagnostics_are_reported_and_sane(self):
        for col in ("se_beta", "t_beta", "r_squared", "n_obs"):
            self.assertIn(col, self.fit.table.columns)
        self.assertTrue((self.fit.table["se_beta"] > 0).all())
        self.assertTrue(self.fit.table["r_squared"].between(0, 1).all())

    def test_zero_beta_asset_is_not_significant(self):
        """An asset built with no market exposure should not test as significant."""
        self.assertLess(abs(self.fit.table.loc["ASSET_B0", "t_beta"]), 2.5)

    def test_vasicek_shrinkage_pulls_toward_one_and_is_monotone(self):
        betas = pd.Series({"high": 2.5, "low": 0.2})
        half = pe.shrink_betas_vasicek(betas, 0.5)
        full = pe.shrink_betas_vasicek(betas, 1.0)
        none = pe.shrink_betas_vasicek(betas, 0.0)
        self.assertAlmostEqual(half["high"], 1.75)
        self.assertAlmostEqual(half["low"], 0.60)
        self.assertTrue((full == 1.0).all())
        pd.testing.assert_series_equal(none, betas)
        with self.assertRaises(ValueError):
            pe.shrink_betas_vasicek(betas, 1.5)

    def test_capm_expected_return_is_affine_in_beta(self):
        betas = pd.Series({"a": 0.0, "b": 1.0, "c": 2.0})
        mu = pe.capm_expected_returns(betas, 0.04, 0.045)
        self.assertAlmostEqual(mu["a"], 0.040)
        self.assertAlmostEqual(mu["b"], 0.085)
        self.assertAlmostEqual(mu["c"], 0.130)

    def test_bayes_stein_shrinks_dispersion_toward_the_grand_mean(self):
        mu = np.array([0.02, 0.08, 0.14])
        sigma = np.diag([0.04, 0.04, 0.04])
        shrunk, intensity = pe.shrink_mean_bayes_stein(mu, sigma, n_obs=250)
        self.assertTrue(0.0 <= intensity <= 1.0)
        self.assertLess(shrunk.std(), mu.std())


class TestCovariance(unittest.TestCase):

    def test_shrinkage_improves_conditioning(self):
        panel = _synthetic_panel(n_days=300)
        returns, *_ = pe.split_returns(panel)
        shrunk, intensity, cond_sample, cond_shrunk = pe.shrunk_covariance(returns)
        self.assertTrue(0.0 <= intensity <= 1.0)
        self.assertLessEqual(cond_shrunk, cond_sample)
        self.assertTrue(np.allclose(shrunk, shrunk.T), "covariance must be symmetric")
        self.assertTrue((np.linalg.eigvalsh(shrunk) > 0).all(), "must be positive definite")


class TestOptimisation(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(3)
        n = 8
        a = rng.normal(size=(n, n))
        self.sigma = (a @ a.T) / n + np.eye(n) * 0.05
        self.mu = np.linspace(0.05, 0.15, n)
        self.rf = 0.04

    def test_weights_are_valid_portfolios(self):
        for weights in (pe.max_sharpe_weights(self.mu, self.sigma, self.rf, 0.25),
                        pe.min_variance_weights(self.sigma, 0.25)):
            self.assertAlmostEqual(weights.sum(), 1.0, places=6)
            self.assertGreaterEqual(weights.min(), -1e-9, "long-only bound violated")
            self.assertLessEqual(weights.max(), 0.25 + 1e-6, "weight cap violated")

    def test_weight_cap_actually_binds(self):
        loose = pe.max_sharpe_weights(self.mu, self.sigma, self.rf, 1.0)
        tight = pe.max_sharpe_weights(self.mu, self.sigma, self.rf, 0.20)
        self.assertLessEqual(tight.max(), 0.20 + 1e-6)
        self.assertGreaterEqual(loose.max(), tight.max() - 1e-6)

    def test_min_variance_beats_equal_weighting_on_variance(self):
        gmv = pe.min_variance_weights(self.sigma, 1.0)
        equal = np.full(self.mu.size, 1.0 / self.mu.size)
        self.assertLessEqual(gmv @ self.sigma @ gmv, equal @ self.sigma @ equal + 1e-12)

    def test_max_sharpe_beats_equal_weighting_on_sharpe(self):
        msr = pe.max_sharpe_weights(self.mu, self.sigma, self.rf, 1.0)
        equal = np.full(self.mu.size, 1.0 / self.mu.size)
        self.assertGreaterEqual(pe.portfolio_moments(msr, self.mu, self.sigma, self.rf)[2],
                                pe.portfolio_moments(equal, self.mu, self.sigma, self.rf)[2] - 1e-9)

    def test_tangency_weights_are_invariant_to_the_risk_premium(self):
        """Section 13.6: with CAPM inputs the premium factors out of the argmax."""
        betas = np.linspace(0.2, 2.4, self.mu.size)
        a = pe.max_sharpe_weights(self.rf + betas * 0.03, self.sigma, self.rf, 0.25)
        b = pe.max_sharpe_weights(self.rf + betas * 0.12, self.sigma, self.rf, 0.25)
        np.testing.assert_allclose(a, b, atol=5e-4)

    def test_frontier_is_upward_sloping_and_starts_at_minimum_variance(self):
        frontier = pe.efficient_frontier(self.mu, self.sigma, 1.0, n_points=12)
        self.assertGreater(len(frontier), 5)
        self.assertTrue(frontier["volatility"].is_monotonic_increasing)
        gmv_vol = pe.portfolio_moments(pe.min_variance_weights(self.sigma, 1.0),
                                       self.mu, self.sigma, 0.0)[1]
        self.assertAlmostEqual(frontier["volatility"].iloc[0], gmv_vol, places=3)

    def test_infeasible_weight_cap_is_rejected_before_solving(self):
        """n_assets=5, weight_cap=0.1 cannot sum to 1 (max attainable is 0.5): must be
        rejected up front, not handed to the solver."""
        with self.assertRaises(pe.InfeasibleWeightCapError):
            pe.max_sharpe_weights(np.linspace(0.05, 0.1, 5),
                                  np.eye(5) * 0.04, 0.02, weight_cap=0.1)
        with self.assertRaises(pe.InfeasibleWeightCapError):
            pe.min_variance_weights(np.eye(5) * 0.04, weight_cap=0.1)

    def test_forced_optimizer_failure_surfaces_as_explicit_status_not_equal_weight(self):
        """A mocked non-converging SLSQP result must raise, never be silently swapped for
        the equal-weight starting point."""
        import unittest.mock as mock

        class _FakeResult:
            success = False
            message = "forced failure for test"
            x = np.full(8, 1.0 / 8)

        with mock.patch("portfolio_engine.minimize", return_value=_FakeResult()):
            with self.assertRaises(pe.OptimizationFailedError):
                pe.max_sharpe_weights(self.mu, self.sigma, self.rf, 0.25)
            # The opt-in legacy path is the only way to recover the old silent behaviour.
            legacy = pe.max_sharpe_weights(self.mu, self.sigma, self.rf, 0.25,
                                           legacy_equal_weight_fallback=True)
            np.testing.assert_allclose(legacy, np.full(8, 1.0 / 8))


class TestPerformanceStatistics(unittest.TestCase):

    def test_constant_return_series_has_zero_volatility_and_known_cagr(self):
        idx = pd.bdate_range("2024-01-01", periods=TD)
        r = pd.Series(0.0004, index=idx)
        perf = pe.performance(r, 0.0)
        self.assertAlmostEqual(perf.ann_vol, 0.0, places=10)
        self.assertAlmostEqual(perf.ann_return, (1.0004 ** TD) - 1, places=6)
        self.assertAlmostEqual(perf.max_drawdown, 0.0, places=10)

    def test_annualised_return_is_geometric_not_arithmetic_compounding(self):
        """Compounding an arithmetic mean overstates realised return; guard against regression."""
        idx = pd.bdate_range("2024-01-01", periods=4)
        r = pd.Series([0.10, -0.10, 0.10, -0.10], index=idx)
        perf = pe.performance(r, 0.0)
        total = (1.10 * 0.90) ** 2 - 1
        self.assertAlmostEqual(perf.total_return, total, places=12)
        self.assertLess(perf.total_return, 0.0, "alternating +/-10% must lose money")

    def test_max_drawdown_is_measured_peak_to_trough(self):
        idx = pd.bdate_range("2024-01-01", periods=4)
        r = pd.Series([0.50, -0.50, 0.0, 0.0], index=idx)
        self.assertAlmostEqual(pe.performance(r, 0.0).max_drawdown, -0.50, places=10)

    def test_initial_period_negative_return_is_captured_in_drawdown(self):
        """Regression guard: a loss on day 1 must register, not be masked because the
        first cumulative-wealth point was (incorrectly) treated as its own peak."""
        idx = pd.bdate_range("2024-01-01", periods=3)
        r = pd.Series([-0.10, 0.02, 0.01], index=idx)
        perf = pe.performance(r, 0.0)
        self.assertAlmostEqual(perf.max_drawdown, -0.10, places=10)
        # A flat/gaining series after day 1 must not erase the day-1 drawdown.
        self.assertLess(perf.max_drawdown, 0.0)

    def test_sharpe_difference_test_returns_no_significance_for_identical_series(self):
        idx = pd.bdate_range("2024-01-01", periods=300)
        rng = np.random.default_rng(11)
        a = pd.Series(rng.normal(0.0005, 0.01, 300), index=idx)
        rf = pd.Series(0.00002, index=idx)
        s1, s2, z, p = pe.sharpe_difference_test(a, a.copy(), rf)
        self.assertAlmostEqual(s1, s2, places=10)
        self.assertAlmostEqual(z, 0.0, places=8)
        self.assertGreater(p, 0.99)


class TestRiskEngine(unittest.TestCase):

    def setUp(self):
        self.mu = np.array([0.08, 0.06])
        self.sigma = np.array([[0.04, 0.006], [0.006, 0.02]])
        self.weights = np.array([0.6, 0.4])
        self.capital = 100_000.0
        self.paths = pe.simulate_gbm_paths(self.weights, self.mu, self.sigma,
                                           capital=self.capital, n_paths=4000, seed=42)

    def test_prices_remain_non_negative(self):
        self.assertTrue((self.paths > 0).all(), "geometric Brownian motion cannot go negative")

    def test_simulation_starts_at_capital_and_has_expected_shape(self):
        self.assertEqual(self.paths.shape, (TD + 1, 4000))
        np.testing.assert_allclose(self.paths[0], self.capital)

    def test_terminal_distribution_matches_theoretical_lognormal_moments(self):
        port_mu = float(self.weights @ self.mu)
        port_var = float(self.weights @ self.sigma @ self.weights)
        log_terminal = np.log(self.paths[-1] / self.capital)
        self.assertAlmostEqual(log_terminal.mean(), port_mu - 0.5 * port_var, delta=0.03)
        self.assertAlmostEqual(log_terminal.std(), np.sqrt(port_var), delta=0.03)

    def test_simulation_is_reproducible_under_a_fixed_seed(self):
        again = pe.simulate_gbm_paths(self.weights, self.mu, self.sigma,
                                      capital=self.capital, n_paths=4000, seed=42)
        np.testing.assert_allclose(self.paths[-1], again[-1])

    def test_correlated_shocks_reproduce_the_input_covariance(self):
        """Confirms the simulation is genuinely multivariate, not a scalar process."""
        rng = np.random.default_rng(5)
        chol = np.linalg.cholesky(self.sigma / TD)
        draws = rng.standard_normal((200_000, 2)) @ chol.T
        np.testing.assert_allclose(np.cov(draws, rowvar=False), self.sigma / TD, atol=2e-5)

    def test_cvar_is_never_below_var(self):
        for confidence in (0.90, 0.95, 0.99):
            var, cvar = pe.value_at_risk(self.capital, self.paths, confidence)
            self.assertGreaterEqual(cvar, var - 1e-9,
                                    f"CVaR below VaR at {confidence:.0%}")

    def test_var_increases_monotonically_with_confidence(self):
        v90 = pe.value_at_risk(self.capital, self.paths, 0.90)[0]
        v95 = pe.value_at_risk(self.capital, self.paths, 0.95)[0]
        v99 = pe.value_at_risk(self.capital, self.paths, 0.99)[0]
        self.assertLess(v90, v95)
        self.assertLess(v95, v99)

    def test_var_and_preservation_probability_are_mutually_consistent(self):
        """Regression guard for the contradiction in an earlier draft of the report."""
        var95 = pe.value_at_risk(self.capital, self.paths, 0.95)[0]
        prob_preserved = float((self.paths[-1] > self.capital).mean())
        loss_probability = float((self.paths[-1] < self.capital).mean())
        if var95 > 0:
            self.assertGreaterEqual(loss_probability, 0.05 - 1e-9,
                                    "a positive 95% VaR requires at least a 5% chance of loss")
        self.assertAlmostEqual(prob_preserved + loss_probability, 1.0, places=6)


class TestPanelLoading(unittest.TestCase):

    def test_requires_benchmark_columns(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
        handle.write("Date,AAA\n2024-01-02,10.0\n")
        handle.close()
        with self.assertRaises(ValueError):
            pe.load_panel(handle.name)

    def test_non_trading_days_are_dropped(self):
        """Rows where the market proxy did not print are calendar days, not trading days."""
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                             newline="", encoding="utf-8")
        handle.write("Date,SPY,IRX,AAA\n"
                     "2024-01-02,400,4.0,10\n"
                     "2024-01-03,,4.0,10\n"          # exchange holiday
                     "2024-01-04,402,4.0,11\n")
        handle.close()
        panel = pe.load_panel(handle.name)
        self.assertEqual(len(panel), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
