# -*- coding: utf-8 -*-
"""
Chronological reconstruction of a retail brokerage transaction ledger.

Quantic MSBA Capstone. Author: Yahia Aktham.

The problem this solves: a brokerage transaction export records what was *traded*, not what
was *held*. Positions established before the export window are invisible, so naively
accumulating buys and sells drives some balances negative. This module infers the minimum
starting position consistent with a non-negative holding path, then rebuilds a daily
holdings and weights matrix.

Two things are deliberately explicit here:

* Every action type in the source export is classified. Silently ignoring an unrecognised
  action type is how a reconstruction ends up quietly wrong -- ``UNCLASSIFIED_ACTIONS``
  is raised rather than skipped.
* Share-affecting actions are separated from cash-only actions. A "Qualified Dividend" row
  carries a dollar amount and an empty quantity; treating it as a share addition is a
  silent no-op that hides a modelling error.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Actions that increase the share count.
SHARE_INCREASING = frozenset({
    "Buy", "Reinvest Shares", "Reinvest Dividend", "Ret Cap Reinvest",
    "Journaled Shares", "Journal", "Internal Transfer",
    "Stock Split", "Stock Split Adj", "Stock Merger",
})

# Actions that decrease the share count.
SHARE_DECREASING = frozenset({"Sell", "Sell Short"})

# Cash-only actions: they move money, never shares. Their Quantity field is blank.
CASH_ONLY = frozenset({
    "Qualified Dividend", "Qual Div Reinvest", "Cash Dividend", "Non-Qualified Div",
    "Special Qual Div", "Special Dividend", "NRA Tax Adj", "Foreign Tax Paid",
    "Foreign Tax Reclaim", "ADR Mgmt Fee", "Bond Interest", "Credit Interest",
    "Margin Interest", "Interest Adj", "Wire Received", "Cash In Lieu",
})

# CUSIP identifiers appearing in place of a ticker, resolved to the surviving symbol.
CUSIP_RESOLUTION = {"143658300": "CCL"}   # Carnival Corp mandatory merger

_DATE_RE = re.compile(r"(\d{2}/\d{2}/\d{4})")


class UnclassifiedActionError(ValueError):
    """Raised when the export contains an action type this module does not know."""


def _parse_amount(raw: str | None) -> float:
    text = (raw or "").strip().replace("$", "").replace(",", "")
    if not text:
        return 0.0
    negative = text.startswith("-")
    try:
        value = float(text.lstrip("-"))
    except ValueError:
        return 0.0
    return -value if negative else value


def _parse_date(raw: str | None):
    """Handle both 'MM/DD/YYYY' and 'MM/DD/YYYY as of MM/DD/YYYY' settlement forms."""
    match = _DATE_RE.match((raw or "").strip())
    return datetime.strptime(match.group(1), "%m/%d/%Y") if match else None


@dataclass
class LedgerSummary:
    """Descriptive facts about the ledger, used directly in the report's EDA section."""
    n_rows: int
    n_symbols: int
    first_date: datetime
    last_date: datetime
    n_buys: int
    n_sells: int
    commission_min: float
    commission_max: float
    commission_total: float
    n_commissioned_trades: int
    corporate_actions: pd.DataFrame
    inferred_starting_balances: dict[str, float]


def read_ledger(path: str | Path) -> pd.DataFrame:
    """Read the brokerage CSV export into a normalised transaction frame."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path.name} contains no data rows")

    known = SHARE_INCREASING | SHARE_DECREASING | CASH_ONLY
    unknown = {r["Action"].strip() for r in rows} - known - {""}
    if unknown:
        raise UnclassifiedActionError(
            f"unclassified action types in {path.name}: {sorted(unknown)}. "
            "Classify them explicitly rather than letting them fall through."
        )

    records = []
    for row in rows:
        date = _parse_date(row.get("Date"))
        action = row["Action"].strip()
        symbol = row["Symbol"].strip()
        if date is None or not symbol:
            continue
        symbol = CUSIP_RESOLUTION.get(symbol, symbol)
        quantity = _parse_amount(row.get("Quantity"))
        if action in SHARE_INCREASING:
            delta = quantity
        elif action in SHARE_DECREASING:
            delta = -quantity
        else:
            delta = 0.0
        records.append({
            "Date": date, "Action": action, "Symbol": symbol,
            "Quantity": quantity, "Price": _parse_amount(row.get("Price")),
            "Fees": _parse_amount(row.get("Fees & Comm")),
            "Amount": _parse_amount(row.get("Amount")),
            "ShareDelta": delta,
            "IsCashOnly": action in CASH_ONLY,
        })
    return pd.DataFrame(records).sort_values("Date").reset_index(drop=True)


def infer_starting_balances(transactions: pd.DataFrame) -> dict[str, float]:
    """Minimum starting position per symbol that keeps the holding path non-negative.

    B_{i,0} = max(0, -min_t cumulative_delta_{i,t})
    """
    balances: dict[str, float] = {}
    for symbol, group in transactions.groupby("Symbol"):
        running_min = group["ShareDelta"].cumsum().min()
        balances[symbol] = float(max(0.0, -running_min))
    return balances


def build_holdings(transactions: pd.DataFrame, price_index: pd.DatetimeIndex,
                   assets: list[str]) -> pd.DataFrame:
    """Daily share-count matrix over ``price_index`` for ``assets``."""
    starting = infer_starting_balances(transactions)
    holdings = pd.DataFrame(0.0, index=price_index, columns=assets)
    for symbol, shares in starting.items():
        if symbol in holdings.columns and shares > 0:
            holdings[symbol] = shares
    relevant = transactions[transactions["Symbol"].isin(assets) & (transactions["ShareDelta"] != 0.0)]
    for symbol, group in relevant.groupby("Symbol"):
        deltas = group.groupby("Date")["ShareDelta"].sum().sort_index()
        # cumulative delta applied from each transaction date forward
        step = pd.Series(0.0, index=price_index)
        for date, delta in deltas.items():
            step.loc[price_index >= date] += delta
        holdings[symbol] = holdings[symbol] + step
    return holdings.clip(lower=0.0)


def portfolio_weights(holdings: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Daily market-value weights of the reconstructed portfolio."""
    values = (holdings * prices[holdings.columns]).fillna(0.0)
    total = values.sum(axis=1).replace(0.0, np.nan)
    return values.div(total, axis=0).fillna(0.0)


def time_weighted_returns(weights: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.Series:
    """Time-weighted portfolio return: sum_i w_{i,t-1} * r_{i,t}.

    This isolates the *allocation* decision from cash flows. Differencing total portfolio
    value would conflate investment performance with deposits and withdrawals -- for a
    portfolio funded by ongoing contributions that produces a wildly overstated return.
    """
    aligned = asset_returns[weights.columns].reindex(weights.index)
    lagged = weights.shift(1)
    return (lagged * aligned).sum(axis=1, min_count=1)


def summarise(transactions: pd.DataFrame) -> LedgerSummary:
    """Descriptive statistics and the corporate-action inventory."""
    fees = transactions.loc[transactions["Fees"] > 0, "Fees"]
    corporate = transactions[transactions["Action"].isin(
        {"Stock Split", "Stock Split Adj", "Stock Merger", "Cash In Lieu"}
    )][["Date", "Action", "Symbol", "Quantity", "Price", "Amount"]]
    return LedgerSummary(
        n_rows=len(transactions),
        n_symbols=int(transactions["Symbol"].nunique()),
        first_date=transactions["Date"].min(),
        last_date=transactions["Date"].max(),
        n_buys=int((transactions["Action"] == "Buy").sum()),
        n_sells=int((transactions["Action"] == "Sell").sum()),
        commission_min=float(fees.min()) if len(fees) else 0.0,
        commission_max=float(fees.max()) if len(fees) else 0.0,
        commission_total=float(fees.sum()),
        n_commissioned_trades=int(len(fees)),
        corporate_actions=corporate.reset_index(drop=True),
        inferred_starting_balances={k: v for k, v in
                                    infer_starting_balances(transactions).items() if v > 0},
    )
