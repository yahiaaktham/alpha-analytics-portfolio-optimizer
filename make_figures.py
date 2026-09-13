# -*- coding: utf-8 -*-
"""
Figure generation for the written report.

Quantic MSBA Capstone. Author: Yahia Aktham.

Every chart is rendered from the analysis outputs in ``results/`` so that no figure can
drift from the numbers quoted in the text. Run ``run_analysis.py`` first.

    python make_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

import portfolio_engine as pe

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "font.size": 9,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})
INK, ACCENT, WARN, MUTED = "#1b2a41", "#2d6a9f", "#b4442e", "#8a93a0"
BASE_MRP, WEIGHT_CAP, BETA_SHRINK = 0.045, 0.10, 0.50


def fig_beta_diagnostics(capm: pd.DataFrame, out: Path) -> None:
    """Betas with 95% confidence intervals - the diagnostic the original report omitted."""
    d = capm.sort_values("beta")
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    y = np.arange(len(d))
    ci = 1.96 * d["se_beta"].to_numpy()
    sig = d["t_beta"].abs() > 1.96
    ax.errorbar(d["beta"], y, xerr=ci, fmt="o", ms=3.2, lw=0.9, capsize=1.8,
                color=ACCENT, ecolor=MUTED, zorder=3)
    ax.scatter(d.loc[~sig, "beta"], y[~sig.to_numpy()], s=34, color=WARN, zorder=4,
               label="not significant at 5%")
    ax.axvline(1.0, color=INK, lw=1.0, ls="--", zorder=2)
    ax.text(1.02, len(d) - 1.5, "market beta = 1.0", fontsize=7.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(d.index, fontsize=6.6)
    ax.set_xlabel("OLS beta on daily excess returns (95% confidence interval)")
    ax.set_title("Every beta is an estimate: CAPM betas with confidence intervals\n"
                 f"N = {len(d)} assets (45 optimized + BOXX, benchmark-only); "
                 f"T = {int(d['n_obs'].max())} daily observations; "
                 f"{100*sig.mean():.0f}% significant at 5%; median R² = {d['r_squared'].median():.3f}")
    if "BOXX" in d.index:
        yb = int(np.flatnonzero(d.index == "BOXX")[0])
        for lbl in ax.get_yticklabels():
            if lbl.get_text() == "BOXX":
                lbl.set_fontweight("bold")
        ax.annotate("BOXX: cash proxy, excluded from the\noptimized universe by design; shown only\n"
                    "as a benchmark-only diagnostic here",
                    (d.loc["BOXX", "beta"], yb), textcoords="offset points", xytext=(70, 55),
                    fontsize=6.8, color=WARN, fontweight="bold", ha="left",
                    arrowprops=dict(arrowstyle="-", color=WARN, lw=0.7, shrinkA=0, shrinkB=6))
    ax.legend(loc="lower right", fontsize=7.5)
    fig.savefig(out / "fig_beta_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def _optimized_universe(results: Path) -> list[str]:
    """The final optimized asset list (45 assets, BOXX excluded by design).

    Every frontier/covariance/correlation figure must use exactly this list rather than
    independently re-deriving "assets with complete history", which silently pulls BOXX
    (and any other cash-proxy/benchmark-only column) back into figures that are supposed
    to describe the optimized risky-asset universe. This is the single source of truth:
    the row index of the final allocation table itself.
    """
    return pd.read_csv(results / "optimal_weights.csv", index_col=0).index.tolist()


def fig_frontier(panel_path: Path, out: Path, start: str, results: Path) -> None:
    """Efficient frontier from the real optimiser, with the capital allocation line."""
    panel = pe.load_panel(panel_path).loc[start:]
    returns, market, rf_daily = pe.split_returns(panel)
    returns = returns[_optimized_universe(results)]
    fit = pe.estimate_capm(returns, market, rf_daily)
    sigma, *_ = pe.shrunk_covariance(returns)
    rf = fit.risk_free_annual
    betas = pe.shrink_betas_vasicek(fit.betas, BETA_SHRINK)
    mu = pe.capm_expected_returns(betas, rf, BASE_MRP).to_numpy()

    frontier = pe.efficient_frontier(mu, sigma, WEIGHT_CAP, n_points=45)
    w_msr = pe.max_sharpe_weights(mu, sigma, rf, WEIGHT_CAP)
    w_gmv = pe.min_variance_weights(sigma, WEIGHT_CAP)
    msr = pe.portfolio_moments(w_msr, mu, sigma, rf)
    gmv = pe.portfolio_moments(w_gmv, mu, sigma, rf)
    ew = pe.portfolio_moments(np.full(mu.size, 1 / mu.size), mu, sigma, rf)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.scatter(np.sqrt(np.diag(sigma)), mu, s=16, color=MUTED, alpha=0.75,
               label="individual assets", zorder=2)
    ax.plot(frontier["volatility"], frontier["target_return"], lw=2.0, color=ACCENT,
            label=f"efficient frontier ({int(WEIGHT_CAP*100)}% weight cap)", zorder=3)
    x_cal = np.linspace(0, max(frontier["volatility"].max(), msr[1]) * 1.12, 50)
    ax.plot(x_cal, rf + msr[2] * x_cal, lw=1.1, ls="--", color=INK,
            label=f"capital allocation line (Sharpe {msr[2]:.2f})", zorder=3)
    # Offsets are hand-placed: MSR and GMV sit almost on top of each other under a low
    # forward premium, so an automatic annotation would overlap illegibly.
    for (v, r, lbl, col, off) in [(gmv[1], gmv[0], "Min variance", "#2f7d5b", (-40, -46)),
                                  (ew[1], ew[0], "Naive 1/N", INK, (16, -22)),
                                  (msr[1], msr[0], "Max Sharpe (tangency)", WARN, (34, 30))]:
        is_tangency = lbl.startswith("Max Sharpe")
        ax.scatter([v], [r], s=200 if is_tangency else 90, marker="*" if is_tangency else "D",
                   color=col, zorder=6, edgecolor="white", linewidth=0.8)
        ax.annotate(f"{lbl}\n{r*100:.1f}% ret / {v*100:.1f}% vol", (v, r),
                    textcoords="offset points", xytext=off, fontsize=7.4, color=col,
                    fontweight="bold", zorder=7,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.6,
                                    shrinkA=0, shrinkB=4))
    ax.scatter([0], [rf], s=45, color="#2f7d5b", zorder=5)
    ax.annotate(f"risk-free {rf*100:.2f}%", (0, rf), textcoords="offset points",
                xytext=(6, -12), fontsize=7.2, color="#2f7d5b")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0)); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("annualised volatility"); ax.set_ylabel("expected annual return\n(CAPM, forward MRP 4.5%)")
    ax.set_title("Ex ante efficient frontier under a forward-looking risk premium\n"
                 f"N = {mu.size} assets, T = {len(returns.dropna())} daily observations")
    # Clip the axis to the informative region; a couple of micro-cap outliers run past 120%
    # volatility and would otherwise compress everything that matters.
    asset_vol = np.sqrt(np.diag(sigma))
    x_max = float(np.percentile(asset_vol, 94)) * 1.05
    n_clipped = int((asset_vol > x_max).sum())
    ax.set_xlim(0, x_max)
    if n_clipped:
        ax.text(0.985, 0.03, f"{n_clipped} high-volatility assets fall outside the plotted range",
                transform=ax.transAxes, ha="right", fontsize=6.6, color=MUTED, style="italic")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.savefig(out / "fig_efficient_frontier.png", bbox_inches="tight")
    plt.close(fig)


def fig_oos_growth(bt: pd.DataFrame, out: Path) -> None:
    """Cumulative out-of-sample growth: the chart that carries the argument."""
    show = {"MVO_naive": ("Textbook MVO (no regularisation)", WARN, 1.9, "-"),
            "EW_1overN": ("Naive 1/N", INK, 1.5, "-"),
            "MSR_capm_vasicek": ("Regularised max-Sharpe", ACCENT, 2.1, "-"),
            "GMV_capped": ("Minimum variance (10% cap)", "#2f7d5b", 1.8, "-"),
            "ACTUAL_discretionary": ("Actual discretionary portfolio", "#7a5c9e", 1.5, "--"),
            "SPY_passive": ("SPY (passive)", MUTED, 1.3, ":")}
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.2), height_ratios=[2.1, 1],
                                  sharex=True)
    for col, (lbl, colour, lw, ls) in show.items():
        if col not in bt:
            continue
        growth = (1 + bt[col].fillna(0)).cumprod()
        ax.plot(growth.index, growth, label=lbl, color=colour, lw=lw, ls=ls)
        dd = growth / growth.cummax() - 1
        ax2.plot(dd.index, dd, color=colour, lw=lw * 0.8, ls=ls)
    ax.set_yscale("log")
    # Matplotlib's default log minor ticks render as "2 x 10^0", which mixes notations with
    # the major-tick "1.0x" labels. Set explicit ticks and format both levels.
    growth_max = max((1 + bt[c].fillna(0)).cumprod().max()
                     for c in show if c in bt)
    ticks = [t for t in (1, 1.5, 2, 3, 4, 5, 6, 8, 10) if t <= growth_max * 1.15]
    ax.set_yticks(ticks)
    ax.set_yticks([], minor=True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.set_ylabel("cumulative growth of $1 (log scale)")
    ax.set_title("Walk-forward out-of-sample performance, monthly rebalancing\n"
                 f"{bt.index.min().date()} to {bt.index.max().date()}, net of 10 bps one-way costs")
    ax.legend(loc="upper left", fontsize=7.6, ncol=2)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_ylabel("drawdown"); ax2.set_xlabel("")
    ax2.set_title("Drawdown: where regularisation earns its keep", fontsize=9.5)
    fig.savefig(out / "fig_oos_performance.png", bbox_inches="tight")
    plt.close(fig)


def fig_risk_ladder(summary: pd.DataFrame, out: Path) -> None:
    """Volatility and drawdown by strategy - the risk-reduction result."""
    order = ["MVO_naive", "EW_1overN", "ACTUAL_discretionary", "MSR_capm",
             "MSR_capm_bayes_stein", "MSR_capm_vasicek", "GMV_capped", "SPY_passive"]
    nice = {"MVO_naive": "Textbook MVO", "EW_1overN": "Naive 1/N",
            "ACTUAL_discretionary": "Actual portfolio", "MSR_capm": "MSR: CAPM+LW+cap",
            "MSR_capm_bayes_stein": "  +Bayes-Stein", "MSR_capm_vasicek": "  +Vasicek beta",
            "GMV_capped": "Min variance", "SPY_passive": "SPY passive"}
    d = summary.reindex([o for o in order if o in summary.index])
    y = np.arange(len(d))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.5), sharey=True)
    cols = [WARN if i in ("MVO_naive",) else ACCENT if i.startswith(("MSR", "GMV")) else MUTED
            for i in d.index]
    a1.barh(y, d["ann_vol"], color=cols, height=0.68)
    a2.barh(y, -d["max_drawdown"], color=cols, height=0.68)
    for ax, series, title in ((a1, d["ann_vol"], "Annualised volatility"),
                              (a2, -d["max_drawdown"], "Maximum drawdown")):
        ax.set_yticks(y); ax.set_yticklabels([nice[i] for i in d.index], fontsize=7.6)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0)); ax.set_title(title, fontsize=9.5)
        ax.invert_yaxis()
        for yi, v in zip(y, series):
            ax.text(v + 0.008, yi, f"{v*100:.1f}%", va="center", fontsize=7.0)
        ax.set_xlim(0, series.max() * 1.22)
    fig.suptitle("Out-of-sample risk: regularised optimisation roughly halves both measures\n"
                 "versus naive equal-weighting (1/N); it cuts both by roughly two-thirds versus textbook MVO",
                 fontsize=9.7, fontweight="bold")
    fig.savefig(out / "fig_risk_reduction.png", bbox_inches="tight")
    plt.close(fig)


def fig_mrp_sensitivity(sens: pd.DataFrame, out: Path) -> None:
    """Why the market risk premium assumption is the whole ball game."""
    d = sens[(sens.betas == "vasicek") & (~sens.bayes_stein)].sort_values("mrp")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    a1.plot(d.mrp, d.exp_return, "o-", color=ACCENT, lw=1.8, ms=5)
    a2.plot(d.mrp, d.sharpe, "o-", color=ACCENT, lw=1.8, ms=5)
    for ax, series, lbl in ((a1, d.exp_return, "MSR expected annual return"),
                            (a2, d.sharpe, "MSR ex ante Sharpe ratio")):
        ax.axvline(0.045, color="#2f7d5b", lw=1.1, ls="--")
        ax.axvline(0.1261, color=WARN, lw=1.1, ls="--")
        ax.set_xlabel("assumed market risk premium"); ax.set_title(lbl, fontsize=9.5)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(bottom=0)
    a1.yaxis.set_major_formatter(PercentFormatter(1.0))
    a1.text(0.047, a1.get_ylim()[1]*0.30, "base case\n4.5%", fontsize=7.2, color="#2f7d5b")
    a1.text(0.104, a1.get_ylim()[1]*0.72, "realised\n2-yr excess\n12.61%", fontsize=7.2, color=WARN)
    fig.suptitle("The headline return is an assumption, not a finding",
                 fontsize=10.5, fontweight="bold")
    fig.savefig(out / "fig_mrp_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def fig_correlation(panel_path: Path, out: Path, start: str, results: Path) -> None:
    """Sample vs Ledoit-Wolf correlation, side by side, with the condition numbers."""
    panel = pe.load_panel(panel_path).loc[start:]
    returns, *_ = pe.split_returns(panel)
    returns = returns[_optimized_universe(results)].dropna()
    sample = np.corrcoef(returns.to_numpy(), rowvar=False)
    sigma, shrink, cond_s, cond_l = pe.shrunk_covariance(returns)
    sd = np.sqrt(np.diag(sigma))
    shrunk = sigma / np.outer(sd, sd)
    labels = list(returns.columns)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    for ax, mat, title in ((axes[0], sample, f"Sample correlation\ncond(Σ) = {cond_s:.2e}"),
                           (axes[1], shrunk, f"Ledoit-Wolf shrunk (δ = {shrink:.3f})\n"
                                             f"cond(Σ) = {cond_l:.1f}")):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.6, vmax=1.0)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=4.6)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=4.6)
        ax.set_title(title, fontsize=9.5); ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.72)
    fig.suptitle("Shrinkage compresses the noise-dominated eigenvalue spectrum",
                 fontsize=10.5, fontweight="bold")
    fig.savefig(out / "fig_correlation_shrinkage.png", bbox_inches="tight")
    plt.close(fig)


def fig_concentration(out: Path, results: Path) -> None:
    """Actual portfolio concentration against the optimiser's recommendation."""
    actual_path = results / "actual_portfolio_weights.csv"
    if not actual_path.exists():
        return
    actual = pd.read_csv(actual_path, index_col=0).iloc[:, 0].sort_values(ascending=False)
    rec = pd.read_csv(results / "optimal_weights.csv", index_col=0)["GMV"]

    TECH = {"MU", "AMD", "TSM", "MXL", "CRDO", "ICHR", "TTMI", "RXT", "LITE",
            "ASX", "QQQ", "SNDK", "PENG"}
    # Thresholds must match report section 12.1: positions counted above 0.5% of market
    # value, tech share summed over the whole portfolio rather than the plotted subset.
    MATERIAL = 0.005
    n_positions = int((actual > MATERIAL).sum())
    tech_share = actual[[t for t in actual.index if t in TECH]].sum()
    top = actual[actual > MATERIAL].head(15)
    colours = [WARN if t in TECH else MUTED for t in top.index]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    y = np.arange(len(top))
    a1.barh(y, top.to_numpy(), color=colours, height=0.7)
    a1.set_yticks(y); a1.set_yticklabels(top.index, fontsize=7.6)
    a1.invert_yaxis(); a1.xaxis.set_major_formatter(PercentFormatter(1.0))
    a1.set_title(f"Actual portfolio: 15 largest of {n_positions} positions\n"
                 f"{tech_share*100:.1f}% in high-beta tech and semiconductors (red)",
                 fontsize=9.5)
    for yi, v in zip(y, top.to_numpy()):
        a1.text(v + 0.002, yi, f"{v*100:.1f}%", va="center", fontsize=6.8)
    a1.set_xlim(0, top.max() * 1.20)

    rec_top = rec[rec > 0.005].sort_values(ascending=False).head(15)
    y2 = np.arange(len(rec_top))
    a2.barh(y2, rec_top.to_numpy(), color=ACCENT, height=0.7)
    a2.set_yticks(y2); a2.set_yticklabels(rec_top.index, fontsize=7.6)
    a2.invert_yaxis(); a2.xaxis.set_major_formatter(PercentFormatter(1.0))
    a2.set_title("Recommended (minimum variance, 10% cap)\n"
                 "defensive and index-anchored", fontsize=9.5)
    for yi, v in zip(y2, rec_top.to_numpy()):
        a2.text(v + 0.002, yi, f"{v*100:.1f}%", va="center", fontsize=6.8)
    a2.set_xlim(0, max(rec_top.max() * 1.20, 0.12))

    fig.suptitle("Concentration is what the optimiser actually changes",
                 fontsize=10.5, fontweight="bold")
    fig.savefig(out / "fig_concentration.png", bbox_inches="tight")
    plt.close(fig)


def fig_architecture(out: Path) -> None:
    """System architecture and information flow (referenced by the POC section)."""
    stages = [
        ("1. Ingest", "brokerage ledger CSV\nvendor price panel\n13-wk T-bill (^IRX)"),
        ("2. Reconstruct", "classify 27 action types\ninfer starting balances\nresolve CUSIP mergers"),
        ("3. Align", "trading-day calendar\nforward-fill (LOCF)\nverify split adjustment"),
        ("4. Estimate", "CAPM betas + SE/t/R²\nVasicek beta shrinkage\nLedoit-Wolf covariance"),
        ("5. Optimise", "SLSQP max-Sharpe / GMV\nlong-only, 10% weight cap\nefficient frontier"),
        ("6. Validate", "walk-forward backtest\nJobson-Korkie tests\nmultivariate MC VaR/CVaR"),
        ("7. Decide", "Streamlit dashboard\nrebalancing instructions\nrisk report"),
    ]
    # Text is wrapped to a character budget matched to the box width. Sizing the boxes in
    # data units while leaving the font in points lets the labels overflow, which is what an
    # earlier version of this figure did.
    import textwrap
    WRAP = 20
    wrapped = [(t, [ln for item in b.split(chr(10))
                    for ln in textwrap.wrap(item, WRAP) or [""]]) for t, b in stages]
    max_lines = max(len(lines) for _, lines in wrapped)

    fig, ax = plt.subplots(figsize=(12.4, 2.9))
    ax.set_axis_off()
    w, gap = 1.0, 0.30
    for i, (title, lines) in enumerate(wrapped):
        x = i * (w + gap)
        ax.add_patch(plt.Rectangle((x, 0), w, 1.0, facecolor="#eef3f8",
                                   edgecolor=ACCENT, linewidth=1.1, zorder=2))
        ax.text(x + w / 2, 0.88, title, ha="center", va="center", fontsize=8.4,
                fontweight="bold", color=INK, zorder=4)
        ax.plot([x + 0.1, x + w - 0.1], [0.76, 0.76], color=ACCENT, lw=0.7, zorder=3)
        block = chr(10).join(lines)
        ax.text(x + w / 2, 0.40, block, ha="center", va="center", fontsize=6.6,
                color=INK, zorder=4, linespacing=1.55)
        if i < len(wrapped) - 1:
            # Arrow sits wholly inside the gap, clear of both boxes.
            ax.annotate("", xy=(x + w + gap - 0.04, 0.5), xytext=(x + w + 0.04, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.3,
                                        shrinkA=0, shrinkB=0), zorder=3)
    ax.set_xlim(-0.06, len(wrapped) * (w + gap) - gap + 0.06)
    ax.set_ylim(-0.06, 1.22)
    ax.set_title("End-to-end analytics workflow: raw broker records to rebalancing decision",
                 fontsize=10.5, fontweight="bold")
    fig.savefig(out / "fig_architecture.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(here / "results"))
    ap.add_argument("--panel", default=str(here.parent / "Sources" / "price_panel_v2_adjusted_2022-2026.csv"))
    ap.add_argument("--insample-start", default="2024-01-26")
    args = ap.parse_args()
    res = Path(args.results)
    out = res / "figures"
    out.mkdir(parents=True, exist_ok=True)

    capm = pd.read_csv(res / "capm_estimates.csv", index_col=0)
    bt = pd.read_csv(res / "backtest_returns.csv", index_col=0, parse_dates=True)
    summary = pd.read_csv(res / "backtest_summary.csv", index_col=0)
    sens = pd.read_csv(res / "mrp_sensitivity.csv")

    fig_beta_diagnostics(capm, out)
    fig_frontier(Path(args.panel), out, args.insample_start, res)
    fig_oos_growth(bt, out)
    fig_risk_ladder(summary, out)
    fig_mrp_sensitivity(sens, out)
    fig_correlation(Path(args.panel), out, args.insample_start, res)
    fig_concentration(out, res)
    fig_architecture(out)
    print(f"wrote {len(list(out.glob('*.png')))} figures to {out}")


if __name__ == "__main__":
    main()
