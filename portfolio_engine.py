# -*- coding: utf-8 -*-
"""
Alpha Analytics - Personal Portfolio Optimizer: econometric engine.

Quantic MSBA Capstone. Author: Yahia Aktham.

Design notes (these are the decisions that matter, and why):

* Covariance and betas are estimated on DAILY returns, not monthly. With N=46 assets a
  monthly sample over a two-year window yields T~18 observations, which makes the sample
  covariance matrix singular (rank <= T-1). Daily sampling gives T~630, so N/T ~ 0.07 and
  the matrix is well conditioned before any shrinkage is applied.
* The risk-free rate is the 13-week Treasury bill (^IRX), not the 10-year note. A 10-year
  yield embeds term premium and duration risk and is not the correct rate for a
  short-horizon Sharpe ratio.
* Expected returns are NOT sample means. Mean-variance optimisation is far more sensitive
  to errors in mu than in Sigma, so mu comes from CAPM with a forward-looking market risk
  premium, and the betas feeding it are themselves shrunk toward 1.0 (Vasicek 1973,
  Blume 1971). Bayes-Stein shrinkage of mu toward the minimum-variance grand mean
  (Jorion 1986) is available as an alternative.
* Sigma is regularised with Ledoit-Wolf shrinkage. NOTE: scikit-learn's LedoitWolf shrinks
  toward a SCALED IDENTITY target, tr(S)/n * I -- not toward a constant-correlation target.
  The constant-correlation estimator is Ledoit and Wolf (2003); the scaled-identity /
  well-conditioned estimator is Ledoit and Wolf (2004), which is what is used here.
* A per-asset weight cap is applied. Binding weight constraints act as an implicit form of
  shrinkage on the covariance matrix and reliably improve out-of-sample behaviour
  (Jagannathan and Ma 2003).
* Performance statistics use ONE consistent convention throughout:
      annualised return  = geometric CAGR, (1 + total)**(252/n) - 1
      annualised vol     = std(daily) * sqrt(252)
      Sharpe             = sqrt(252) * mean(daily excess) / std(daily excess)
  Compounding an arithmetic daily mean ((1+mean)**252 - 1) overstates realised return and
  is deliberately avoided.
* Monte Carlo is MULTIVARIATE: correlated shocks are generated from the Cholesky factor of
  the covariance matrix, so per-asset paths and portfolio paths are jointly consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252
BENCHMARK_COLS = ("SPY", "IRX")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_panel(path: str | Path) -> pd.DataFrame:
    """Load a wide daily price panel: Date index, SPY, IRX, then one column per asset.

    Rows where the market proxy is missing are dropped, which restricts the panel to
    genuine exchange trading days rather than all weekdays.
    """
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    missing = [c for c in BENCHMARK_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"panel is missing required benchmark columns: {missing}")
    return df[df["SPY"].notna()]


def split_returns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (asset daily returns, market daily returns, daily risk-free rate)."""
    assets = [c for c in df.columns if c not in BENCHMARK_COLS]
    asset_returns = df[assets].pct_change()
    market_returns = df["SPY"].pct_change()
    # ^IRX quotes an annualised discount rate in percent.
    risk_free_daily = (df["IRX"] / 100.0) / TRADING_DAYS
    return asset_returns, market_returns, risk_free_daily


def eligible_assets(returns: pd.DataFrame, min_obs: int) -> list[str]:
    """Assets with a complete return history over the supplied window."""
    return [c for c in returns.columns
            if returns[c].notna().sum() >= min_obs and returns[c].tail(min_obs).notna().all()]


# ---------------------------------------------------------------------------
# CAPM estimation with regression diagnostics
# ---------------------------------------------------------------------------

@dataclass
class CapmFit:
    """Per-asset CAPM regression results, including the diagnostics MVO papers omit."""
    table: pd.DataFrame
    market_return_cagr: float
    risk_free_annual: float

    @property
    def betas(self) -> pd.Series:
        return self.table["beta"]


def estimate_capm(asset_returns: pd.DataFrame,
                  market_returns: pd.Series,
                  risk_free_daily: pd.Series,
                  assets: list[str] | None = None) -> CapmFit:
    """OLS of daily excess asset returns on daily excess market returns.

    Reports beta, its standard error, t-statistic and R-squared. A beta without a standard
    error is not an estimate, it is a number; the t-statistics are what justify (or refuse)
    any economic interpretation of an individual beta.
    """
    assets = assets or list(asset_returns.columns)
    market_excess = market_returns - risk_free_daily
    rows = []
    for asset in assets:
        excess = asset_returns[asset] - risk_free_daily
        mask = excess.notna() & market_excess.notna()
        y = excess[mask].to_numpy()
        x = market_excess[mask].to_numpy()
        n = y.size
        if n < 30:
            continue
        design = np.column_stack([np.ones(n), x])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coef
        sigma2 = resid @ resid / (n - 2)
        cov_beta = sigma2 * np.linalg.inv(design.T @ design)
        se = np.sqrt(np.diag(cov_beta))
        ss_tot = ((y - y.mean()) ** 2).sum()
        daily = asset_returns[asset].dropna()
        rows.append({
            "Ticker": asset,
            "n_obs": n,
            "alpha_daily": coef[0],
            "beta": coef[1],
            "se_beta": se[1],
            "t_beta": coef[1] / se[1] if se[1] > 0 else np.nan,
            "r_squared": 1.0 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan,
            "ann_vol": daily.std() * np.sqrt(TRADING_DAYS),
            "realised_cagr": (1.0 + daily).prod() ** (TRADING_DAYS / len(daily)) - 1.0,
        })
    table = pd.DataFrame(rows).set_index("Ticker")
    market = market_returns.dropna()
    return CapmFit(
        table=table,
        market_return_cagr=(1.0 + market).prod() ** (TRADING_DAYS / len(market)) - 1.0,
        risk_free_annual=float((risk_free_daily * TRADING_DAYS).mean()),
    )


def shrink_betas_vasicek(betas: pd.Series, weight: float = 0.5) -> pd.Series:
    """Shrink betas toward the market beta of 1.0 (Vasicek 1973, Blume 1971).

    Cross-sectional beta dispersion is inflated by estimation error, and shrinking toward
    1.0 is the standard correction. ``weight`` is the weight placed on the prior.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must lie in [0, 1]")
    return (1.0 - weight) * betas + weight * 1.0


def capm_expected_returns(betas: pd.Series, risk_free: float, market_risk_premium: float) -> pd.Series:
    """E(R_i) = r_f + beta_i * MRP, annualised."""
    return risk_free + betas * market_risk_premium


def shrink_mean_bayes_stein(mu: np.ndarray, sigma: np.ndarray, n_obs: int) -> tuple[np.ndarray, float]:
    """Bayes-Stein shrinkage of expected returns toward the minimum-variance grand mean.

    Jorion (1986). Returns the shrunk mean vector and the shrinkage intensity actually used.
    """
    n = mu.size
    ones = np.ones(n)
    precision = np.linalg.pinv(sigma)
    grand_mean = float(ones @ precision @ mu / (ones @ precision @ ones))
    deviation = mu - grand_mean * ones
    quad = float(deviation @ precision @ deviation)
    lam = (n + 2) / quad if quad > 0 else 0.0
    intensity = min(max(lam / (n_obs + lam), 0.0), 1.0)
    return (1.0 - intensity) * mu + intensity * grand_mean * ones, intensity


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------

def shrunk_covariance(returns: pd.DataFrame) -> tuple[np.ndarray, float, float, float]:
    """Annualised Ledoit-Wolf covariance (scaled-identity target).

    Returns (annualised shrunk covariance, shrinkage intensity, cond(sample), cond(shrunk)).
    The condition numbers are the evidence that shrinkage is necessary rather than cosmetic.
    """
    clean = returns.dropna()
    estimator = LedoitWolf().fit(clean)
    shrunk = estimator.covariance_ * TRADING_DAYS
    sample = np.cov(clean.to_numpy(), rowvar=False) * TRADING_DAYS
    return shrunk, float(estimator.shrinkage_), float(np.linalg.cond(sample)), float(np.linalg.cond(shrunk))


# ---------------------------------------------------------------------------
# Cash-proxy (BOXX) aware covariance and expected returns
# ---------------------------------------------------------------------------
#
# BOXX is a box-spread ETF: a T-bill-like cash proxy, not an equity. Feeding it through the
# same pipeline built for risky assets is wrong in three separate ways, all fixed here:
#   1. Vasicek beta shrinkage pulls EVERY beta toward the market beta of 1.0. That prior is
#      correct for an equity of unknown beta; it is wrong for an asset that is, by
#      construction, supposed to have (near) zero market exposure.
#   2. CAPM expected return is r_f + beta * MRP. For an asset with ~zero beta this collapses
#      to ~r_f, which happens to be approximately right for BOXX for the wrong reason -- the
#      declared rule here is to use BOXX's OWN historical mean return instead, so the number
#      is not an artefact of a beta estimate landing near zero.
#   3. Ledoit-Wolf's scaled-identity shrinkage target, tr(S)/n * I, is calibrated for a
#      homogeneous universe of risky assets with broadly similar variance. Mixed in with a
#      near-riskless asset whose variance is ~2 orders of magnitude smaller, that target
#      inflates BOXX's modelled variance towards the risky-asset average. The correction
#      here: shrink the CORRELATION matrix toward identity (not the covariance matrix
#      toward a scaled identity), then rescale by each asset's OWN measured standard
#      deviation, so BOXX's tiny measured volatility survives shrinkage intact.

CASH_CORR_SHRINK = 0.20  # disclosed shrinkage intensity for the correlation-based cash
                         # treatment; chosen as a moderate value in the 0.1-0.3 range,
                         # not fit to the data.
CASH_BETA_SIGNIFICANCE_T = 1.96  # two-sided 5% threshold for "beta indistinguishable from 0"


def cash_aware_covariance(returns: pd.DataFrame, cash_ticker: str,
                          corr_shrink: float = CASH_CORR_SHRINK) -> tuple[np.ndarray, dict]:
    """Annualised covariance for a universe that may include a cash-like proxy.

    If ``cash_ticker`` is not a column, this is exactly :func:`shrunk_covariance`. If it is
    present: the risky-risky block comes from Ledoit-Wolf on the risky assets only (as
    elsewhere in this module); the cash row/column and the cash variance come from the
    full-universe correlation matrix shrunk toward identity at ``corr_shrink``, rescaled by
    each asset's own historical standard deviation. The result is checked for symmetry and
    positive semi-definiteness (clipping negative eigenvalues to zero if needed) before it
    is returned, and the diagnostics needed to report raw vs modelled cash risk are
    returned alongside it.
    """
    order = list(returns.columns)
    if cash_ticker not in order:
        sigma, shrinkage, cond_sample, cond_shrunk = shrunk_covariance(returns)
        return sigma, {"cash_proxy": None, "lw_shrinkage_risky": shrinkage,
                       "cond_sample": cond_sample, "cond_shrunk": cond_shrunk}

    risky = [c for c in order if c != cash_ticker]
    clean = returns.dropna()
    sigma_risky, shrinkage, cond_sample, cond_shrunk = shrunk_covariance(clean[risky])

    n = len(order)
    corr_full = clean[order].corr().to_numpy()
    corr_shrunk = (1.0 - corr_shrink) * corr_full + corr_shrink * np.eye(n)
    std_full = (clean[order].std() * np.sqrt(TRADING_DAYS)).to_numpy()
    cov_from_corr = corr_shrunk * np.outer(std_full, std_full)

    cash_idx = order.index(cash_ticker)
    risky_idx = [order.index(c) for c in risky]
    sigma = np.zeros((n, n))
    for a, ia in enumerate(risky_idx):
        for b, ib in enumerate(risky_idx):
            sigma[ia, ib] = sigma_risky[a, b]
    for ia in risky_idx:
        sigma[ia, cash_idx] = cov_from_corr[ia, cash_idx]
        sigma[cash_idx, ia] = cov_from_corr[cash_idx, ia]
    sigma[cash_idx, cash_idx] = cov_from_corr[cash_idx, cash_idx]
    sigma = (sigma + sigma.T) / 2.0

    eigvals = np.linalg.eigvalsh(sigma)
    if (eigvals < -1e-8).any():
        eigvecs = np.linalg.eigh(sigma)[1]
        sigma = eigvecs @ np.diag(np.clip(eigvals, 0.0, None)) @ eigvecs.T
        sigma = (sigma + sigma.T) / 2.0
        eigvals = np.linalg.eigvalsh(sigma)
    is_symmetric = bool(np.allclose(sigma, sigma.T, atol=1e-10))
    is_psd = bool((eigvals > -1e-8).all())
    assert is_symmetric, "cash-aware covariance matrix is not symmetric"
    assert is_psd, "cash-aware covariance matrix is not positive semi-definite"

    cash_vol_raw = float(clean[cash_ticker].std() * np.sqrt(TRADING_DAYS))
    cash_vol_modeled = float(np.sqrt(sigma[cash_idx, cash_idx]))
    diagnostics = {
        "cash_proxy": cash_ticker, "lw_shrinkage_risky": shrinkage,
        "cond_sample_risky": cond_sample, "cond_shrunk_risky": cond_shrunk,
        "corr_shrink_intensity": corr_shrink,
        "cash_vol_raw": cash_vol_raw, "cash_vol_modeled": cash_vol_modeled,
        "cov_is_symmetric": is_symmetric, "cov_is_psd": is_psd,
    }
    return sigma, diagnostics


def cash_aware_expected_returns(fit: CapmFit, order: list[str], cash_ticker: str,
                                risk_free_annual: float, market_risk_premium: float,
                                beta_shrink: float = 0.5,
                                significance_t: float = CASH_BETA_SIGNIFICANCE_T
                                ) -> tuple[np.ndarray, dict]:
    """Expected returns for a universe that may include a cash-like proxy.

    Risky assets: Vasicek-shrunk beta, CAPM expected return (unchanged from elsewhere).
    Cash proxy: beta is NOT shrunk toward 1.0. Its own measured beta is used unless that
    beta's t-statistic is below ``significance_t`` (statistically indistinguishable from
    zero), in which case an explicit near-zero-beta assumption (0.0) is used instead --
    BOXX is a T-bill-like box-spread ETF, not an equity, so "shrink toward the market" is
    the wrong prior for it. Its expected return is its OWN historical mean return, not a
    CAPM-derived figure -- a declared modelling rule, not a claim that BOXX guarantees that
    yield going forward.
    """
    betas = fit.betas.reindex(order).copy()
    if cash_ticker not in order:
        betas = shrink_betas_vasicek(betas, beta_shrink)
        mu = capm_expected_returns(betas, risk_free_annual, market_risk_premium)
        return mu.reindex(order).to_numpy(), {"cash_proxy": None}

    risky = [c for c in order if c != cash_ticker]
    betas.loc[risky] = shrink_betas_vasicek(betas.loc[risky], beta_shrink)
    cash_row = fit.table.loc[cash_ticker]
    cash_t = float(cash_row["t_beta"]) if np.isfinite(cash_row["t_beta"]) else np.nan
    near_zero_used = bool(not np.isfinite(cash_t) or abs(cash_t) < significance_t)
    cash_beta_modeled = 0.0 if near_zero_used else float(cash_row["beta"])
    betas.loc[cash_ticker] = cash_beta_modeled

    mu = capm_expected_returns(betas, risk_free_annual, market_risk_premium)
    cash_return_raw = float(cash_row["realised_cagr"])
    mu.loc[cash_ticker] = cash_return_raw  # declared rule: own historical mean, not CAPM

    diagnostics = {
        "cash_proxy": cash_ticker,
        "cash_beta_raw": float(cash_row["beta"]), "cash_beta_t": cash_t,
        "cash_beta_modeled": cash_beta_modeled,
        "cash_near_zero_beta_assumption_used": near_zero_used,
        "cash_return_raw_hist_mean": cash_return_raw, "cash_return_modeled": cash_return_raw,
    }
    return mu.reindex(order).to_numpy(), diagnostics


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

class InfeasibleWeightCapError(ValueError):
    """Raised when a fully-invested long-only portfolio cannot satisfy the weight cap.

    A necessary condition for feasibility of ``sum(w) == 1`` with ``0 <= w_i <= weight_cap``
    is ``n_assets * weight_cap >= 1``. This is checked BEFORE calling the solver: SLSQP can
    return ``success=True`` on an infeasible problem by converging to the least-infeasible
    point it can find, which would silently produce a portfolio that does not sum to 1.
    """


class OptimizationFailedError(RuntimeError):
    """Raised when SLSQP does not converge (``result.success is False``).

    Carries the raw ``scipy.optimize.OptimizeResult`` as ``.result`` so the caller can log
    the failure reason (``result.message``) rather than silently substituting a different
    portfolio for the one that was asked for.
    """

    def __init__(self, message: str, result) -> None:
        super().__init__(message)
        self.result = result


def _solve(objective, n_assets: int, weight_cap: float, *,
          legacy_equal_weight_fallback: bool = False) -> np.ndarray:
    """Solve the long-only, fully-invested weight-capped problem.

    ``legacy_equal_weight_fallback`` reproduces the old (buggy) behaviour of silently
    returning the equal-weight starting point when SLSQP fails to converge. It defaults to
    False and must be opted into explicitly and by name; the default path raises
    :class:`OptimizationFailedError` instead, so a failed solve can never be mistaken for a
    valid optimized portfolio.
    """
    tol = 1e-9
    if n_assets * weight_cap < 1.0 - tol:
        raise InfeasibleWeightCapError(
            f"infeasible: n_assets * weight_cap = {n_assets} * {weight_cap} = "
            f"{n_assets * weight_cap:.4f} < 1.0; no long-only portfolio with this cap can "
            f"sum to 1."
        )
    constraints = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = tuple((0.0, weight_cap) for _ in range(n_assets))
    start = np.full(n_assets, 1.0 / n_assets)
    result = minimize(objective, start, method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"maxiter": 800, "ftol": 1e-12})
    if not result.success:
        if legacy_equal_weight_fallback:
            return start
        raise OptimizationFailedError(
            f"SLSQP failed to converge ({result.message}); refusing to silently return "
            f"an equal-weight portfolio in its place.", result)
    return result.x


def max_sharpe_weights(mu: np.ndarray, sigma: np.ndarray, risk_free: float,
                       weight_cap: float = 1.0, *,
                       legacy_equal_weight_fallback: bool = False) -> np.ndarray:
    """Long-only, fully invested tangency portfolio.

    Solved with SLSQP. This is a LOCAL gradient method: with a binding weight cap the
    solution is not guaranteed to be the global optimum, and no such claim is made.
    """
    return _solve(lambda w: -(w @ mu - risk_free) / np.sqrt(w @ sigma @ w), mu.size, weight_cap,
                 legacy_equal_weight_fallback=legacy_equal_weight_fallback)


def min_variance_weights(sigma: np.ndarray, weight_cap: float = 1.0, *,
                         legacy_equal_weight_fallback: bool = False) -> np.ndarray:
    """Long-only, fully invested global minimum-variance portfolio."""
    return _solve(lambda w: np.sqrt(w @ sigma @ w), sigma.shape[0], weight_cap,
                 legacy_equal_weight_fallback=legacy_equal_weight_fallback)


def portfolio_moments(weights: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                      risk_free: float) -> tuple[float, float, float]:
    """(expected return, volatility, Sharpe ratio) - all annualised, ex ante."""
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ sigma @ weights))
    return ret, vol, (ret - risk_free) / vol


def efficient_frontier(mu: np.ndarray, sigma: np.ndarray, weight_cap: float = 1.0,
                       n_points: int = 40) -> pd.DataFrame:
    """Minimum-variance frontier traced across attainable target returns."""
    lo = float(portfolio_moments(min_variance_weights(sigma, weight_cap), mu, sigma, 0.0)[0])
    hi = float(mu.max()) if weight_cap >= 1.0 else float(np.sort(mu)[::-1][:int(np.ceil(1 / weight_cap))].mean())
    points = []
    for target in np.linspace(lo, hi, n_points):
        constraints = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},
                       {"type": "eq", "fun": lambda w, t=target: w @ mu - t})
        bounds = tuple((0.0, weight_cap) for _ in range(mu.size))
        res = minimize(lambda w: np.sqrt(w @ sigma @ w), np.full(mu.size, 1.0 / mu.size),
                       method="SLSQP", bounds=bounds, constraints=constraints,
                       options={"maxiter": 600, "ftol": 1e-11})
        if res.success:
            points.append({"target_return": target, "volatility": float(np.sqrt(res.x @ sigma @ res.x))})
    return pd.DataFrame(points)


# ---------------------------------------------------------------------------
# Performance statistics - one convention, used everywhere
# ---------------------------------------------------------------------------

@dataclass
class Performance:
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    total_return: float
    n_days: int


def performance(daily_returns: pd.Series, risk_free_daily: pd.Series | float) -> Performance:
    """Realised performance under the conventions documented in the module docstring."""
    r = daily_returns.dropna()
    n = len(r)
    if isinstance(risk_free_daily, pd.Series):
        rf = risk_free_daily.reindex(r.index).ffill().fillna(0.0)
    else:
        rf = pd.Series(float(risk_free_daily), index=r.index)
    excess = r - rf
    total = float((1.0 + r).prod() - 1.0)
    cumulative = (1.0 + r).cumprod()
    # Prepend starting wealth of 1.0 before taking the running max: without this a loss on
    # the very first observation is invisible, because that first cumulative value would
    # also be its own running maximum (drawdown 0 by construction) instead of a decline
    # from the 1.0 starting point.
    wealth = np.insert(cumulative.to_numpy(), 0, 1.0)
    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / running_max - 1.0
    return Performance(
        ann_return=(1.0 + total) ** (TRADING_DAYS / n) - 1.0,
        ann_vol=float(r.std() * np.sqrt(TRADING_DAYS)),
        sharpe=float(np.sqrt(TRADING_DAYS) * excess.mean() / excess.std()) if excess.std() > 0 else np.nan,
        max_drawdown=float(drawdown.min()),
        total_return=total,
        n_days=n,
    )


def sharpe_difference_test(a: pd.Series, b: pd.Series,
                           risk_free_daily: pd.Series) -> tuple[float, float, float, float]:
    """Jobson-Korkie test of equal Sharpe ratios with the Memmel (2003) correction.

    Returns (sharpe_a, sharpe_b, z statistic, two-sided p-value). Reporting this matters:
    two strategies can differ by a wide margin in point estimate and still be statistically
    indistinguishable over a few hundred observations.
    """
    from math import erfc, sqrt

    joined = pd.concat([a.dropna(), b.dropna()], axis=1, join="inner").dropna()
    rf = risk_free_daily.reindex(joined.index).ffill().fillna(0.0)
    e1 = joined.iloc[:, 0] - rf
    e2 = joined.iloc[:, 1] - rf
    n = len(joined)
    rho = float(np.corrcoef(e1, e2)[0, 1])
    s1, s2 = e1.mean() / e1.std(), e2.mean() / e2.std()
    theta = (1.0 / n) * (2 - 2 * rho + 0.5 * (s1 ** 2 + s2 ** 2 - 2 * s1 * s2 * rho ** 2))
    annualised = (float(s1 * np.sqrt(TRADING_DAYS)), float(s2 * np.sqrt(TRADING_DAYS)))
    # Degenerate case: perfectly correlated series with equal Sharpe ratios drive theta to
    # zero, so the ratio is 0/0. Two identical return streams are by definition not
    # significantly different, so report that rather than propagating a NaN.
    if theta <= 0 or not np.isfinite(theta):
        return (*annualised, 0.0, 1.0)
    z = float((s1 - s2) / np.sqrt(theta))
    return (*annualised, z, float(erfc(abs(z) / sqrt(2))))


# ---------------------------------------------------------------------------
# Multivariate Monte Carlo
# ---------------------------------------------------------------------------

def simulate_gbm_paths(weights: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                       capital: float = 100_000.0, horizon_days: int = TRADING_DAYS,
                       n_paths: int = 10_000, seed: int = 42) -> np.ndarray:
    """Multivariate geometric Brownian motion for the whole asset vector.

    Correlated daily shocks are drawn as L @ z with L the Cholesky factor of the daily
    covariance matrix, so cross-asset dependence is preserved. Returns portfolio values of
    shape (horizon_days + 1, n_paths). A jitter is added to the diagonal only if required to
    make the factorisation succeed.
    """
    rng = np.random.default_rng(seed)
    n = mu.size
    daily_cov = sigma / TRADING_DAYS
    daily_mu = mu / TRADING_DAYS
    jitter = 0.0
    while True:
        try:
            chol = np.linalg.cholesky(daily_cov + jitter * np.eye(n))
            break
        except np.linalg.LinAlgError:
            jitter = max(jitter * 10, 1e-12)
            if jitter > 1e-4:
                raise
    drift = (daily_mu - 0.5 * np.diag(daily_cov))
    log_prices = np.zeros((horizon_days + 1, n_paths, n))
    for t in range(1, horizon_days + 1):
        shocks = rng.standard_normal((n_paths, n)) @ chol.T
        log_prices[t] = log_prices[t - 1] + drift + shocks
    asset_rel = np.exp(log_prices)                      # relative price, starts at 1.0
    return capital * (asset_rel @ weights)


def value_at_risk(capital: float, paths: np.ndarray,
                  confidence: float = 0.95) -> tuple[float, float]:
    """Dollar VaR and CVaR of terminal loss at the given confidence level.

    Positive numbers denote losses. A negative VaR is a legitimate outcome: it means the
    loss quantile is a gain, which is what a high-drift simulation should produce.
    """
    losses = capital - paths[-1]
    var = float(np.percentile(losses, confidence * 100))
    tail = losses[losses >= var]
    return var, float(tail.mean())
