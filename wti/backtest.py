"""A toy backtest: a sanity check on the signal, not a trading system.

No slippage model, no contract rolls, no risk management, no position sizing. A flat
cost is charged whenever the position changes and that is the whole of the friction
model. If a signal cannot clear this bar it certainly cannot clear a real one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wti.config import Horizon

# Periods per year, used only to annualise the Sharpe ratio.
BARS_PER_YEAR = {"daily": 252, "hourly": 24 * 5 * 52}


def build_positions(horizon: Horizon, pred: pd.Series) -> pd.DataFrame:
    """Turn predictions into -1 / 0 / +1 positions, one column per rule.

    Classification gets two rules: always in the market, and a "confident only" variant
    that stands aside unless the model is more than 5 points away from a coin flip.
    Regression just follows the sign of the predicted move.
    """
    pos = pd.DataFrame(index=pred.index)
    if horizon.is_classification:
        pos["always in"] = np.where(pred >= 0.5, 1, -1)
        pos["confident only"] = np.select([pred >= 0.55, pred <= 0.45], [1, -1], default=0)
    else:
        pos["always in"] = np.where(pred >= 0, 1, -1)
    return pos


def run(horizon: Horizon, pred: pd.Series, ret: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every position rule against realised returns.

    Returns ``(pnl, summary)``: per-bar P&L for each rule alongside buy-and-hold, and a
    summary table with total return, volatility, annualised Sharpe and time in market.
    """
    ret = ret.reindex(pred.index)
    positions = build_positions(horizon, pred)

    pnl = pd.DataFrame({"buy & hold": ret}, index=pred.index)
    time_in_market = {"buy & hold": 1.0}
    for rule in positions.columns:
        held = positions[rule]
        flips = held.diff().abs().fillna(0)
        pnl[rule] = held * ret - flips * horizon.cost_bps / 1e4
        time_in_market[rule] = float((held != 0).mean())

    periods = BARS_PER_YEAR[horizon.name]
    summary = pd.DataFrame({
        "total_return": pnl.sum(),
        "per_bar_std": pnl.std(),
        "time_in_market": pd.Series(time_in_market),
    })
    summary["sharpe_annualised"] = (
        summary["total_return"] / len(pnl) / summary["per_bar_std"] * np.sqrt(periods)
    )
    return pnl, summary[["total_return", "per_bar_std", "sharpe_annualised", "time_in_market"]]


def accuracy_by_confidence(proba: pd.Series, y_true: pd.Series) -> pd.DataFrame:
    """Does accuracy rise as the model moves away from 0.5?

    If the probabilities carry information it should, and a "trade only when confident"
    rule has something to stand on. A flat or non-monotonic table means the confidence
    is noise, whatever the headline accuracy says.
    """
    correct = ((proba >= 0.5).astype(int) == y_true.reindex(proba.index)).astype(int)
    confidence = (proba - 0.5).abs()
    bucket = pd.cut(confidence, bins=[0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.5],
                    labels=["<2%", "2-4%", "4-6%", "6-8%", "8-10%", ">10%"])
    return (pd.DataFrame({"correct": correct, "bucket": bucket})
            .groupby("bucket", observed=True)
            .agg(n=("correct", "size"), accuracy=("correct", "mean"))
            .round(3))


def rolling_win_rate(pred: pd.Series, y_true: pd.Series, price: pd.Series, window: int) -> float:
    """Share of rolling windows in which the model's RMSE beat the random walk's.

    A single whole-sample number can hide an edge that lived in one regime and died. 50%
    is a coin flip.
    """
    y_true = y_true.reindex(pred.index)
    close = price.reindex(pred.index)
    err_model = ((y_true - pred) * close) ** 2
    err_rw = (y_true * close) ** 2
    won = np.sqrt(err_model.rolling(window).mean()) < np.sqrt(err_rw.rolling(window).mean())
    return float(won.dropna().mean())
