# Alpha Analytics — Personal Portfolio Optimizer

Quantic MSBA Capstone project. Author: **Yahia Aktham**.

A prescriptive decision support system for personal portfolio allocation. It reconstructs a
real brokerage transaction ledger into a daily holdings series, estimates asset risk and
CAPM equilibrium expected returns with full regression diagnostics, solves a constrained
mean-variance problem with shrinkage regularisation, and — critically — validates the result
**out of sample**, with true monthly holding-drift between rebalances, before recommending
anything.

## What this project actually concludes

Worth stating up front, because it is not the usual claim:

- **Unregularised textbook mean-variance optimisation fails.** Out of sample it produced
  43.7% annualised volatility, a 35.6% maximum drawdown and 31.8% monthly turnover — the
  worst risk-adjusted result of every strategy tested.
- **The standard remedies work, as separate alternatives, not a monotonic ladder.** Capped
  minimum variance reaches 13.4% volatility and a 13.0% drawdown; the beta-shrunk CAPM
  tangency portfolio reaches 15.3% volatility and a 15.8% drawdown. Only one comparison
  survives significance testing: capped minimum variance versus passive SPY
  (Jobson-Korkie/Memmel z = +2.50, p = 0.013). The tangency portfolio's improvement over the
  unregularised baseline is not statistically significant (p = 0.069).
- **No comparison against the author's actual discretionary account is claimed.** There is
  no independent broker-reported time-weighted-return statement, only the raw transaction
  ledger, so any whole-account performance or concentration claim is withdrawn from the
  headline results. Ledger reconstruction still stands as a demonstrated data-engineering
  result (1,126 transactions, 197 symbols, $355.89 in commissions), not a performance claim.
- **Optimisation does not reliably beat naive 1/N on risk-adjusted return** (p = 0.895 vs
  capped minimum variance, p = 0.791 vs the tangency portfolio), reproducing DeMiguel,
  Garlappi and Uppal (2009).
- **The cash-equivalent asset (BOXX) is excluded from optimisation by design.** Its measured
  beta is statistically indistinguishable from zero (t = 1.63), so it does not receive the
  same equity-beta shrinkage or covariance fitting as the risky assets. It remains in the
  results only as a passive benchmark (4.7% return, 0.4% volatility).

So the value proposition is **risk control plus an illustrative cost-avoidance comparison,
not return enhancement or a proven cash saving**. Absolute return levels in the backtest are
inflated by survivorship bias — the asset universe is the author's *current* holdings,
selected after four years of trading — so only the relative comparisons between strategies
are interpretable, and even those comparisons are not protected from that selection.

## Modules

| File | Purpose |
|---|---|
| `run_analysis.py` | Entry point. Runs the full pipeline and writes every reported figure to the chosen results directory (`--results`, e.g. `results_v10/`), including `report_figures.md`. The unreconciled actual-account comparison is off by default; pass `--include-legacy-actual-account` to reinstate it as a diagnostic. |
| `portfolio_engine.py` | CAPM estimation with standard errors, t-statistics and R²; Vasicek beta shrinkage (risky assets only); Bayes-Stein mean shrinkage; Ledoit-Wolf covariance with conditioning diagnostics; constrained SLSQP tangency and minimum-variance optimisers, with an explicit feasibility check and `OptimizationFailedError`/`InfeasibleWeightCapError` on solver failure rather than a silent equal-weight fallback; efficient frontier; performance statistics (drawdown includes starting wealth) under one consistent convention; Jobson-Korkie Sharpe-difference test; multivariate Cholesky Monte Carlo; VaR and CVaR. |
| `ledger_reconstruction.py` | Brokerage CSV parser classifying **all 28** action types (raises on an unrecognised one rather than skipping it); starting-balance inference; CUSIP merger resolution; daily holdings and weights; time-weighted returns. |
| `make_figures.py` | Regenerates every figure used in the written report from the chosen results directory, reading the optimized asset list from `optimal_weights.csv` so BOXX-excluded charts stay consistent with the pipeline. |
| `test_portfolio_engine.py` | `unittest` suite covering ledger, estimation, optimiser and risk engine, including regression tests for solver-failure transparency, infeasible-cap rejection, and initial-period drawdown. |
| `test_run_analysis.py` | Regression tests for the walk-forward backtest: monthly holding-drift, turnover-after-unequal-drift over the union of eligible assets, transaction-cost convention, and BOXX exclusion end-to-end. |
| `portfolio_dashboard.py` | Streamlit dashboard: allocation and frontier, downside risk, estimates and data. Uses the same BOXX-excluded universe as the pipeline. |

## Running

```bash
pip install -r requirements.txt

# Full analysis into a fresh results directory. --ledger is optional; without it the
# ledger-derived diagnostics are skipped entirely. The actual-account performance
# comparison stays off unless --include-legacy-actual-account is also passed.
python run_analysis.py --panel <price panel csv> --ledger <brokerage export csv> --out results_v10

python make_figures.py --results results_v10
python -m unittest test_portfolio_engine test_run_analysis -v
streamlit run portfolio_dashboard.py
```

### Tested environment

The committed test suite (43 tests, synthetic data only — no brokerage or price inputs
required) passes with `python -m unittest test_portfolio_engine test_run_analysis -v` and
with pytest 9.1.1 on Python 3.14.3 / Linux, using the pinned versions in
`requirements.txt`. The environment used to generate the private reported results is
recorded in `environment.md`; see report Section 18.4 for the one platform-sensitive row
(the unregularised MVO baseline) and why every regularised figure is platform-independent.

### Input data format

`run_analysis.py` expects a wide daily price panel:

| Date | SPY | IRX | TICKER_1 | … |
|---|---|---|---|---|
| 2024-01-26 | 487.41 | 5.24 | 12.85 | … |

- `SPY` — market proxy, split- and dividend-adjusted closes
- `IRX` — 13-week Treasury bill yield, annualised percent (**not** a 10-year yield: a
  10-year yield embeds term premium and duration risk and is the wrong risk-free rate for a
  one-year horizon)
- remaining columns — adjusted closes, one per asset

Rows where `SPY` is missing are dropped, restricting the panel to genuine exchange trading
days rather than all weekdays.

The brokerage ledger is the raw broker CSV export with columns `Date`, `Action`, `Symbol`,
`Description`, `Quantity`, `Price`, `Fees & Comm`, `Amount`.

## Data is deliberately not included

**The price panel and the brokerage transaction ledger are excluded from this repository**
because they derive from personal financial holdings. `.gitignore` blocks them explicitly.
This repository reproduces the **code and the method**, not the private inputs. Supply your
own panel in the format above to run it. The public tests run entirely on synthetic data
and are the default validation path; no empirical result files are committed.

## AI assistance disclosure

Generative AI was used in this project's development, and is disclosed fully in the written
Capstone report (Section 3.3): Google Colab was the development and execution environment;
Gemini was used inside Colab for code drafting and debugging, and separately for
report-section drafting; Claude Code was used for line-by-line audit of the report and code
and for the rebuilt econometric engine. These were tools, not collaborators: all analytical
decisions, all numbers and all conclusions are the author's. An AI-generated audit of an
earlier draft returned confidently stated findings that were false on direct inspection —
one reason the final pipeline raises on unrecognized inputs instead of skipping them, and
every reported figure is regenerated deterministically by the committed code and its tests.

## Key design decisions

Each of these was a correction to something that was wrong in an earlier version of this
project, and each is documented in the module docstrings:

1. **Estimate on daily, not monthly, returns.** Monthly sampling over a two-year window gives
   T ≈ 18 observations for N = 45 assets — a rank-17 singular covariance matrix. Daily
   sampling gives T ≈ 632 and N/T ≈ 0.071 for the optimized universe (N = 46 including BOXX
   for the broader beta-diagnostic table).
2. **Report uncertainty on every estimate.** A beta without a standard error is a number, not
   an estimate. Median |t| is 10.6 and 96% of betas are significant at 5%.
3. **Use a forward-looking risk premium, not a realised one.** The realised market excess
   return over the estimation window was 17.1%; the model uses 4.5% with sensitivity across
   3–6%. Note that with CAPM inputs the premium **factors out of the max-Sharpe objective**,
   so it scales expected return and Sharpe but cannot move an allocation.
4. **Shrink both Σ and μ, but only for the risky assets.** Ledoit-Wolf on the covariance
   (scaled-identity target — this is Ledoit and Wolf 2004, *not* the constant-correlation
   estimator of 2003), Vasicek on the betas, Bayes-Stein available on the mean vector. BOXX
   does not inherit the same equity-beta shrinkage prior.
5. **Cap per-asset weights.** Binding constraints act as implicit covariance shrinkage
   (Jagannathan and Ma 2003).
6. **One performance convention everywhere.** Geometric CAGR, √252-scaled volatility, and
   Sharpe as √252 × mean/std of daily excess returns. Compounding an arithmetic mean
   overstates realised return and is avoided. Drawdown includes starting wealth so a
   first-period loss is not missed.
7. **Monte Carlo is multivariate.** Correlated shocks from the Cholesky factor of the
   covariance matrix, not a scalar portfolio-level process. It is a one-year buy-and-hold
   scenario, not a simulation of the monthly-rebalanced policy.
8. **True holding-drift between rebalances.** Weights drift daily with asset returns within
   a holding period rather than being rebalanced to target every day; turnover is measured
   against the drifted end-of-period holdings over the union of old and new eligible assets,
   not the prior target weights.
9. **Exclude the cash-equivalent asset from optimisation.** BOXX's measured beta is
   statistically indistinguishable from zero, so it is kept out of the optimized universe by
   design and reported only as a passive benchmark, rather than forcing it through the same
   equity-oriented shrinkage as the risky assets.
10. **Surface solver failures explicitly.** A failed optimisation raises rather than
    silently returning equal weights; failures are logged by strategy and date.
11. **No claim without an out-of-sample test and a significance test.**

## References

Artzner et al. (1999); Barney (1991); Blume (1971); DeMiguel, Garlappi and Uppal (2009);
Dimson, Marsh and Staunton (2002); Jagannathan and Ma (2003); Jobson and Korkie (1981);
Jorion (1986); Ledoit and Wolf (2003, 2004); Lintner (1965); Marchenko and Pastur (1967);
Markowitz (1952); Memmel (2003); Michaud (1989); Mossin (1966); Sharpe (1964, 1994);
Vasicek (1973). Full citations in the written Capstone report.
