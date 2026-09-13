# -*- coding: utf-8 -*-
"""
Alpha Analytics - Personal Portfolio Optimizer: decision dashboard.

Quantic MSBA Capstone. Author: Yahia Aktham.

    streamlit run portfolio_dashboard.py

A presentation layer over ``portfolio_engine``. Every number shown here is produced by the
same functions ``run_analysis.py`` calls, so the dashboard cannot display a figure the
pipeline would not produce.

The price panel is not committed to this repository (it derives from personal brokerage
holdings), so the panel is supplied at runtime: either place it at the default path, or
upload it in the sidebar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import portfolio_engine as pe  # noqa: E402

DEFAULT_PANEL = SCRIPT_DIR.parent / "Sources" / "price_panel_v2_adjusted_2022-2026.csv"
BASE_MRP, WEIGHT_CAP, BETA_SHRINK = 0.045, 0.10, 0.50
CASH_TICKER = "BOXX"  # box-spread ETF, cash proxy - excluded from the optimized universe by
                       # design, same rule as run_analysis.py/CASH_TICKER. Kept here so this
                       # dashboard's allocation/optimization panels match the pipeline's
                       # universe contract instead of re-admitting BOXX into the risky-asset universe.

st.set_page_config(page_title="Alpha Analytics — Portfolio Optimizer",
                   page_icon="📈", layout="wide")


# --------------------------------------------------------------------------- data

@st.cache_data(show_spinner=False)
def load(path_or_buffer, start: str) -> pd.DataFrame:
    panel = pe.load_panel(path_or_buffer)
    return panel.loc[start:]


@st.cache_data(show_spinner=False)
def estimate(panel: pd.DataFrame, mrp: float, beta_shrink: float):
    returns, market, rf_daily = pe.split_returns(panel)
    keep = [c for c in returns.columns
            if returns[c].iloc[1:].notna().all() and c != CASH_TICKER]
    returns = returns[keep]
    fit = pe.estimate_capm(returns, market, rf_daily)
    sigma, shrinkage, cond_sample, cond_shrunk = pe.shrunk_covariance(returns)
    betas = pe.shrink_betas_vasicek(fit.betas, beta_shrink)
    mu = pe.capm_expected_returns(betas, fit.risk_free_annual, mrp)
    return returns, fit, sigma, mu, shrinkage, cond_sample, cond_shrunk


# --------------------------------------------------------------------------- sidebar

st.sidebar.title("Alpha Analytics")
st.sidebar.caption("Personal Portfolio Optimizer")

uploaded = st.sidebar.file_uploader("Price panel CSV", type="csv",
                                    help="Columns: Date, SPY, IRX, then one column per asset.")
source = uploaded if uploaded is not None else (DEFAULT_PANEL if DEFAULT_PANEL.exists() else None)

if source is None:
    st.title("Personal Portfolio Optimizer")
    st.warning(
        "No price panel found. The daily price panel is excluded from the public repository "
        "because it derives from personal brokerage holdings.\n\n"
        f"Either place a panel at `{DEFAULT_PANEL}` or upload one in the sidebar. "
        "Required columns: `Date`, `SPY`, `IRX`, then one column per asset ticker."
    )
    st.stop()

start = st.sidebar.text_input("Estimation start date", value="2024-01-26")
mrp = st.sidebar.slider("Forward market risk premium", 0.00, 0.15, BASE_MRP, 0.005,
                        format="%.3f",
                        help="Note: with CAPM inputs the tangency weights are invariant to "
                             "this assumption. It scales expected return and Sharpe only.")
cap = st.sidebar.slider("Per-asset weight cap", 0.02, 1.00, WEIGHT_CAP, 0.01)
beta_shrink = st.sidebar.slider("Vasicek beta shrinkage toward 1.0", 0.0, 1.0, BETA_SHRINK, 0.05)
capital = st.sidebar.number_input("Capital base ($)", 1_000, 10_000_000, 100_000, 1_000)

try:
    panel = load(source, start)
    returns, fit, sigma, mu, shrinkage, cond_sample, cond_shrunk = estimate(panel, mrp, beta_shrink)
except Exception as exc:  # surfaced rather than swallowed: a bad panel must not fail silently
    st.error(f"Could not process the panel: {exc}")
    st.stop()

assets = list(returns.columns)
mu_vec = mu.to_numpy()
rf = fit.risk_free_annual

st.title("Personal Portfolio Optimizer")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Assets (N)", len(assets))
c2.metric("Observations (T)", len(returns.dropna()))
c3.metric("N / T", f"{len(assets)/len(returns.dropna()):.3f}")
c4.metric("Risk-free (13-wk T-bill)", f"{rf*100:.2f}%")
c5.metric("cond(Σ) after shrinkage", f"{cond_shrunk:.0f}",
          delta=f"from {cond_sample:,.0f}", delta_color="inverse")

tab_alloc, tab_risk, tab_data = st.tabs(
    ["Allocation & frontier", "Downside risk", "Estimates & data"])


# --------------------------------------------------------------------------- allocation

with tab_alloc:
    w_msr = pe.max_sharpe_weights(mu_vec, sigma, rf, cap)
    w_gmv = pe.min_variance_weights(sigma, cap)
    w_eq = np.full(len(assets), 1.0 / len(assets))

    choice = st.radio("Starting allocation", ["Tangency (max Sharpe)", "Minimum variance",
                                              "Equal weight (1/N)"], horizontal=True)
    base = {"Tangency (max Sharpe)": w_msr, "Minimum variance": w_gmv,
            "Equal weight (1/N)": w_eq}[choice]

    st.caption("Adjust any weight below; the portfolio is renormalised to sum to 100%.")
    held = pd.Series(base, index=assets)
    top = held[held > 0.0005].sort_values(ascending=False)
    edited = {}
    cols = st.columns(4)
    for i, (ticker, weight) in enumerate(top.items()):
        edited[ticker] = cols[i % 4].number_input(ticker, 0.0, 100.0, float(weight * 100),
                                                 0.25, format="%.2f", key=f"w_{ticker}")
    custom = pd.Series(0.0, index=assets)
    for ticker, value in edited.items():
        custom[ticker] = value / 100.0
    total = custom.sum()
    if total > 0:
        custom = custom / total

    rows = []
    for label, weights in (("Your allocation", custom.to_numpy()),
                           ("Tangency (max Sharpe)", w_msr),
                           ("Minimum variance", w_gmv),
                           ("Equal weight (1/N)", w_eq)):
        ret, vol, sharpe = pe.portfolio_moments(weights, mu_vec, sigma, rf)
        rows.append({"Portfolio": label, "Expected return": f"{ret*100:.2f}%",
                     "Volatility": f"{vol*100:.2f}%", "Sharpe (ex ante)": f"{sharpe:.2f}",
                     "Largest weight": f"{weights.max()*100:.1f}%",
                     "Positions": int((weights > 0.005).sum())})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    frontier = pe.efficient_frontier(mu_vec, sigma, cap, n_points=30)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.scatter(np.sqrt(np.diag(sigma)), mu_vec, s=14, color="#8a93a0", alpha=0.7,
               label="individual assets")
    ax.plot(frontier["volatility"], frontier["target_return"], lw=2, color="#2d6a9f",
            label="efficient frontier")
    for weights, label, colour, marker in ((custom.to_numpy(), "Your allocation", "#7a5c9e", "o"),
                                           (w_msr, "Tangency", "#b4442e", "*"),
                                           (w_gmv, "Min variance", "#2f7d5b", "D"),
                                           (w_eq, "1/N", "#1b2a41", "s")):
        ret, vol, _ = pe.portfolio_moments(weights, mu_vec, sigma, rf)
        ax.scatter([vol], [ret], s=170 if marker == "*" else 80, marker=marker,
                   color=colour, edgecolor="white", zorder=5, label=label)
    x = np.linspace(0, float(frontier["volatility"].max()) * 1.1, 30)
    ax.plot(x, rf + pe.portfolio_moments(w_msr, mu_vec, sigma, rf)[2] * x, ls="--", lw=1,
            color="#1b2a41", label="capital allocation line")
    upper = float(np.percentile(np.sqrt(np.diag(sigma)), 94)) * 1.05
    ax.set_xlim(0, upper)
    ax.set_xlabel("annualised volatility")
    ax.set_ylabel("expected annual return")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(alpha=0.25)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.info(
        "The tangency weights do not change when you move the risk-premium slider. That is "
        "not a bug: with CAPM expected returns the premium factors out of the max-Sharpe "
        "objective, so it scales expected return and the Sharpe ratio but cannot move an "
        "allocation."
    )


# --------------------------------------------------------------------------- risk

with tab_risk:
    st.subheader("Multivariate Monte Carlo")
    left, right = st.columns([1, 2])
    with left:
        n_paths = st.select_slider("Simulated paths", [1_000, 5_000, 10_000, 25_000], 10_000)
        horizon = st.slider("Horizon (trading days)", 21, 504, 252, 21)
        confidence = st.select_slider("Confidence level", [0.90, 0.95, 0.99], 0.95)
        which = st.radio("Portfolio", ["Tangency", "Minimum variance", "Equal weight"])
    weights = {"Tangency": pe.max_sharpe_weights(mu_vec, sigma, rf, cap),
               "Minimum variance": pe.min_variance_weights(sigma, cap),
               "Equal weight": np.full(len(assets), 1.0 / len(assets))}[which]

    paths = pe.simulate_gbm_paths(weights, mu_vec, sigma, capital=float(capital),
                                  horizon_days=int(horizon), n_paths=int(n_paths))
    var, cvar = pe.value_at_risk(float(capital), paths, float(confidence))
    terminal = paths[-1]

    with right:
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{confidence:.0%} VaR", f"${var:,.0f}",
                  help="Quantile of the loss distribution, not a maximum loss. "
                       "A negative value means the loss quantile is a gain.")
        m2.metric(f"{confidence:.0%} CVaR", f"${cvar:,.0f}",
                  help="Average loss across the tail beyond VaR. A coherent risk measure.")
        m3.metric("P(above initial capital)", f"{(terminal > capital).mean()*100:.1f}%")
        st.dataframe(pd.DataFrame([{
            "5th percentile": f"${np.percentile(terminal, 5):,.0f}",
            "Median": f"${np.median(terminal):,.0f}",
            "95th percentile": f"${np.percentile(terminal, 95):,.0f}",
        }]), hide_index=True, use_container_width=True)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    step = max(1, paths.shape[1] // 250)
    a1.plot(paths[:, ::step], lw=0.4, alpha=0.25, color="#2d6a9f")
    a1.axhline(capital, color="#b4442e", ls="--", lw=1)
    a1.set_title("Simulated portfolio paths", fontsize=10)
    a1.set_xlabel("trading day")
    a2.hist(terminal, bins=70, color="#2d6a9f", alpha=0.85)
    a2.axvline(capital, color="#b4442e", ls="--", lw=1.2, label="initial capital")
    a2.axvline(capital - var, color="#1b2a41", ls=":", lw=1.4, label=f"{confidence:.0%} VaR")
    a2.set_title("Terminal value distribution", fontsize=10)
    a2.legend(fontsize=8, frameon=False)
    for ax in (a1, a2):
        ax.grid(alpha=0.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "Shocks are drawn as L·z with L the Cholesky factor of the daily covariance matrix, "
        "so cross-asset dependence is preserved. Innovations are Gaussian, so tail estimates "
        "are likely optimistic relative to observed equity return distributions."
    )


# --------------------------------------------------------------------------- data

with tab_data:
    st.subheader("CAPM estimates")
    table = fit.table.copy()
    table["significant at 5%"] = table["t_beta"].abs() > 1.96
    st.dataframe(
        table[["n_obs", "beta", "se_beta", "t_beta", "r_squared", "ann_vol",
               "realised_cagr", "significant at 5%"]]
        .rename(columns={"n_obs": "obs", "se_beta": "std. error", "t_beta": "t",
                         "r_squared": "R²", "ann_vol": "ann. volatility",
                         "realised_cagr": "realised CAGR"})
        .sort_values("beta", ascending=False)
        .style.format({"beta": "{:.3f}", "std. error": "{:.3f}", "t": "{:.2f}",
                       "R²": "{:.3f}", "ann. volatility": "{:.1%}", "realised CAGR": "{:.1%}"}),
        use_container_width=True, height=420)

    n_sig = int((table["t_beta"].abs() > 1.96).sum())
    st.caption(f"{n_sig} of {len(table)} betas are significant at the 5% level; "
               f"median R² is {table['r_squared'].median():.3f}. Ledoit-Wolf shrinkage "
               f"intensity δ = {shrinkage:.4f}.")

    st.subheader("Aligned price panel")
    st.dataframe(panel.tail(250), use_container_width=True, height=320)
