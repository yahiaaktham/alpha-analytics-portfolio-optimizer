# -*- coding: utf-8 -*-
"""
Regression tests for run_analysis.py's walk-forward backtest mechanics.

Quantic MSBA Capstone. Author: Yahia Aktham.

Covers the monthly holding-drift fix (constant-weight-for-the-month was overstating
returns and understating turnover), the transaction-cost convention, and BOXX's exclusion
from the optimized universe while remaining available as a passive benchmark series.

    python -m unittest test_run_analysis -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import portfolio_engine as pe
import run_analysis as ra


class TestHoldingDrift(unittest.TestCase):

    def test_unchanged_weights_and_round_trip_prices_give_zero_gross_return(self):
        """A: 100 -> 200 -> 100 (round trip). B: flat at 100. Starting 50/50, held (not
        rebalanced) for the period. Gross return must be ~0%, not the +12.5% the old
        constant-weight-for-the-month calculation produced."""
        idx = pd.bdate_range("2024-01-02", periods=2)
        forward = pd.DataFrame({"A": [1.0, -0.5], "B": [0.0, 0.0]}, index=idx)
        target = np.array([0.5, 0.5])
        gross, end_w = ra._drift_holding_period(forward, target)
        total_return = float((1.0 + gross).prod() - 1.0)
        self.assertAlmostEqual(total_return, 0.0, places=10)
        # sanity: the old constant-weight-for-the-month bug would have produced +12.5% here
        self.assertGreater(abs(total_return - 0.125), 0.1)

    def test_unequal_drift_with_unchanged_target_still_produces_positive_turnover(self):
        """A: 100 -> 150 -> 150 (+50% then flat). B: flat. Starting 50/50 drifts to 60/40
        by period end even though the NEXT target is again 50/50: turnover at the
        rebalance must be strictly positive, not zero."""
        idx = pd.bdate_range("2024-01-02", periods=2)
        forward = pd.DataFrame({"A": [0.5, 0.0], "B": [0.0, 0.0]}, index=idx)
        target = np.array([0.5, 0.5])
        _, end_w = ra._drift_holding_period(forward, target)
        np.testing.assert_allclose(end_w, [0.6, 0.4], atol=1e-9)
        new_target = np.array([0.5, 0.5])  # unchanged
        turnover = float(np.abs(new_target - end_w).sum() / 2.0)
        self.assertGreater(turnover, 0.0)
        self.assertAlmostEqual(turnover, 0.1, places=9)

    def test_transaction_cost_equals_declared_rate_times_total_buy_sell_notional(self):
        """Cost charged must equal TRANSACTION_COST * (total one-way traded notional * 2),
        i.e. the declared one-way rate applied to total buy+sell notional -- not to the
        old (wrong) basis of comparing against the prior TARGET instead of drifted
        holdings."""
        old_end_weights = pd.Series({"A": 0.6, "B": 0.4})
        new_target = pd.Series({"A": 0.5, "B": 0.5})
        union_idx = old_end_weights.index.union(new_target.index)
        old_aligned = old_end_weights.reindex(union_idx).fillna(0.0).to_numpy()
        new_aligned = new_target.reindex(union_idx).fillna(0.0).to_numpy()
        traded_one_way = float(np.abs(new_aligned - old_aligned).sum() / 2.0)
        cost = 2.0 * traded_one_way * ra.TRANSACTION_COST
        total_buy_sell_notional = float(np.abs(new_aligned - old_aligned).sum())
        self.assertAlmostEqual(cost, ra.TRANSACTION_COST * total_buy_sell_notional, places=12)
        self.assertAlmostEqual(traded_one_way, 0.1, places=9)
        self.assertAlmostEqual(cost, 0.0010 * 0.2, places=9)

    def test_turnover_uses_union_of_old_and_new_eligible_assets(self):
        """If the eligible universe changes between rebalances, turnover must be computed
        over the UNION of old and new assets, not just the assets in the new period (which
        would silently ignore a full exit from a dropped asset)."""
        old_end_weights = pd.Series({"A": 0.5, "C": 0.5})   # C drops out of eligibility
        new_target = pd.Series({"A": 0.5, "B": 0.5})        # B newly eligible
        union_idx = old_end_weights.index.union(new_target.index)
        self.assertEqual(set(union_idx), {"A", "B", "C"})
        old_aligned = old_end_weights.reindex(union_idx).fillna(0.0).to_numpy()
        new_aligned = new_target.reindex(union_idx).fillna(0.0).to_numpy()
        traded = float(np.abs(new_aligned - old_aligned).sum() / 2.0)
        # A unchanged (0 contribution), C fully sold (0.5), B fully bought (0.5) -> one-way = 0.5
        self.assertAlmostEqual(traded, 0.5, places=9)


def _synthetic_full_panel(n_days: int = 560, n_risky: int = 12, seed: int = 21) -> pd.DataFrame:
    """A panel large enough to exercise in_sample + walk_forward end to end, including a
    BOXX column built to look like a genuine near-riskless cash proxy (tiny vol, ~zero
    correlation to the market), the way the real ETF behaves in the source data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    market = rng.normal(0.0004, 0.010, n_days)
    spy = 400.0 * np.exp(np.cumsum(market))
    frame = {"SPY": spy, "IRX": np.full(n_days, 4.0)}
    for i in range(n_risky):
        beta = 0.5 + i * 0.15
        r = beta * market + rng.normal(0.0, 0.006, n_days)
        frame[f"AST{i:02d}"] = 100.0 * np.exp(np.cumsum(r))
    cash_r = rng.normal(0.00018, 0.00023, n_days)  # matches measured BOXX daily stats
    frame["BOXX"] = 100.0 * np.exp(np.cumsum(cash_r))
    return pd.DataFrame(frame, index=pd.DatetimeIndex(dates, name="Date"))


class TestBoxxExcludedFromOptimizedUniverse(unittest.TestCase):

    def setUp(self):
        self.panel = _synthetic_full_panel()
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_boxx_absent_from_optimized_weights_present_in_benchmark_series(self):
        ins = ra.in_sample(self.panel, self.out, start=None)
        oos = ra.walk_forward(self.panel, None, self.out)

        weights = pd.read_csv(self.out / "optimal_weights.csv", index_col=0)
        self.assertNotIn("BOXX", weights.index,
                         "BOXX must not appear in any optimized weights vector")

        backtest = pd.read_csv(self.out / "backtest_returns.csv", index_col=0)
        self.assertIn("BOXX_cash", backtest.columns,
                      "BOXX must remain available as a passive benchmark return series")
        self.assertTrue(backtest["BOXX_cash"].notna().any())

        self.assertIn("BOXX", ins["excluded"]["excluded_by_design"])
        self.assertNotIn("BOXX", ins["excluded"]["insufficient_history"])
        self.assertIn("cash_beta_raw", ins["cash_diagnostics"])
        self.assertEqual(oos["actual_discretionary_included"], False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
