"""Feature engineering and label construction.

The one rule every function here obeys: a feature at timestamp ``t`` may only use
information from bars ``<= t``. Break it and the backtest will look wonderful and the
live forecast will be worthless. ``tests/test_features.py`` checks this mechanically by
recomputing the features on a truncated series and comparing the last row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wti.config import TARGET_TICKER, Horizon
from wti.data import Panel


@dataclass
class Dataset:
    """A modelling-ready table plus the names of the columns that matter."""

    frame: pd.DataFrame       # features + labels, fully dropna'd
    features: list[str]       # feature column names, in model order
    target: str               # column the model is trained to predict
    ret: str                  # realised return of the next bar, for the backtest
    price: str                # close at the moment of the forecast

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[self.features]

    @property
    def y(self) -> pd.Series:
        return self.frame[self.target]

    def __len__(self) -> int:
        return len(self.frame)


def rsi(price: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI: 0-100, above 70 = ran up a lot, below 30 = sold off a lot."""
    change = price.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


# --------------------------------------------------------------------------- daily

def build_daily_features(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Fourteen features, each with a one-line plain-English meaning.

    Deliberately small: daily crude returns are mostly noise, and a wide feature set
    mainly buys more ways to fit that noise.
    """
    p = close[TARGET_TICKER]
    vol = volume[TARGET_TICKER]
    F = pd.DataFrame(index=close.index)

    # WTI itself
    F["move_1d"] = p.pct_change(1)                              # today's % move
    F["move_5d"] = p.pct_change(5)                              # last week
    F["move_21d"] = p.pct_change(21)                            # last month
    F["volatility_21d"] = p.pct_change().rolling(21).std()      # how choppy the month was
    F["vs_50d_avg"] = p / p.rolling(50).mean() - 1              # distance from the 50-day average
    F["rsi_14"] = rsi(p)                                        # overbought / oversold
    F["volume_vs_avg"] = vol / vol.rolling(21).mean()           # 1 = a normal day

    # related markets
    F["brent_move_1d"] = close["brent"].pct_change()
    F["gasoline_move_1d"] = close["gasoline"].pct_change()
    F["sp500_move_1d"] = close["sp500"].pct_change()
    F["dollar_move_1d"] = close["dollar"].pct_change()
    F["vix_level"] = close["vix"]
    F["wti_vs_brent"] = p / close["brent"] - 1                  # the benchmark spread

    # calendar
    F["day_of_week"] = close.index.dayofweek
    return F


def build_daily_labels(close: pd.DataFrame) -> pd.DataFrame:
    """Tomorrow's close and tomorrow's % change.

    We predict the *change*, not the price. A model that predicts the price scores an
    R-squared near 0.99 by copying today's close, which tells you nothing.
    """
    out = pd.DataFrame({"close": close[TARGET_TICKER]})
    out["next_close"] = out["close"].shift(-1)
    out["next_change"] = out["next_close"] / out["close"] - 1

    # 2020-04-20 settled at -37.63 $/bbl. A % change through a negative price is
    # meaningless, so that day and its neighbours lose their labels.
    bad = out["close"] <= 0
    out.loc[bad | bad.shift(1, fill_value=False) | bad.shift(-1, fill_value=False),
            ["next_close", "next_change"]] = np.nan
    return out


# -------------------------------------------------------------------------- hourly

def build_hourly_features(panel: Panel, target: str = TARGET_TICKER) -> pd.DataFrame:
    """Fifty-odd features for the next-hour direction problem.

    Groups: own returns and lags, volatility and mean reversion, bar anatomy, volume,
    cross-asset moves and spreads, calendar.
    """
    close, open_, high, low, volume = panel.close, panel.open, panel.high, panel.low, panel.volume
    c, o, h, l = close[target], open_[target], high[target], low[target]
    lr = np.log(c).diff()
    F = pd.DataFrame(index=c.index)

    # --- own returns
    for k in [1, 2, 3, 6, 12, 24, 48]:
        F[f"ret_{k}"] = np.log(c / c.shift(k))
    for k in [1, 2, 3, 4, 5]:
        F[f"ret1_lag{k}"] = lr.shift(k)

    # --- volatility / mean reversion
    for w in [6, 24, 72]:
        F[f"vol_{w}"] = lr.rolling(w).std()
    F["vol_ratio_6_72"] = F["vol_6"] / F["vol_72"]
    for w in [24, 72, 168]:
        m, s = c.rolling(w).mean(), c.rolling(w).std()
        F[f"z_{w}"] = (c - m) / s
    F["rsi_14"] = rsi(c)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    F["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c

    # --- bar anatomy
    F["body"] = (c - o) / o
    F["range"] = (h - l) / c
    F["upper_wick"] = (h - np.maximum(o, c)) / c
    F["lower_wick"] = (np.minimum(o, c) - l) / c
    F["close_pos_in_bar"] = ((c - l) / (h - l)).replace([np.inf, -np.inf], np.nan).fillna(0.5)
    lo24, hi24 = l.rolling(24).min(), h.rolling(24).max()
    F["close_pos_24"] = ((c - lo24) / (hi24 - lo24)).replace([np.inf, -np.inf], np.nan).fillna(0.5)

    # --- volume (log1p keeps zero-volume bars in the sample instead of poisoning rolling windows)
    lv = np.log1p(volume[target])
    F["log_vol"] = lv
    F["vol_surprise_24"] = ((lv - lv.rolling(24, min_periods=12).mean())
                            / lv.rolling(24, min_periods=12).std()).fillna(0)
    F["zero_volume"] = (volume[target] == 0).astype(int)

    # --- cross-asset (last known value on the WTI calendar)
    oc = close.ffill()
    for name in ["brent", "gasoline", "heating_oil", "sp500", "vix"]:
        s = oc[name]
        F[f"{name}_ret_1"] = np.log(s / s.shift(1))
        F[f"{name}_ret_6"] = np.log(s / s.shift(6))
    F["vix_level"] = oc["vix"]
    F["vix_z_72"] = (oc["vix"] - oc["vix"].rolling(72).mean()) / oc["vix"].rolling(72).std()

    spread = np.log(c / oc["brent"])
    F["wti_brent_spread"] = spread
    F["wti_brent_spread_chg_6"] = spread.diff(6)
    F["wti_brent_spread_z_168"] = (spread - spread.rolling(168).mean()) / spread.rolling(168).std()
    F["wti_minus_brent_ret_1"] = F["ret_1"] - F["brent_ret_1"]
    F["gasoline_crack_chg_6"] = np.log(oc["gasoline"] / c).diff(6)
    F["heating_crack_chg_6"] = np.log(oc["heating_oil"] / c).diff(6)

    # --- calendar
    idx = c.index
    F["hour"] = idx.hour
    F["dow"] = idx.dayofweek
    F["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    F["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    F["us_pit_session"] = ((idx.hour >= 9) & (idx.hour < 15)).astype(int)
    F["gap_prev_h"] = (idx.to_series().diff().dt.total_seconds() / 3600).values
    F["after_gap"] = (F["gap_prev_h"] > 1).astype(int)
    return F


def build_hourly_labels(panel: Panel, target: str = TARGET_TICKER) -> pd.DataFrame:
    """Next-hour direction, with the bar-spacing caveat carried along.

    Hourly bars are not contiguous: a daily maintenance break around 17:00 ET, plus
    weekends and holidays. "The next hour" only means something when the next bar really
    is one hour later, so ``gap_next_h`` is kept and used to filter in
    :func:`build_dataset`.
    """
    out = pd.DataFrame({"close": panel.close[target]})
    ts = out.index.to_series()
    out["gap_next_h"] = (-ts.diff(-1)).dt.total_seconds() / 3600
    out["next_ret"] = np.log(out["close"].shift(-1) / out["close"])
    out["target"] = (out["next_ret"] > 0).astype(int)
    return out


# ------------------------------------------------------------------------ assembly

def build_dataset(horizon: Horizon, panel: Panel) -> Dataset:
    """Features + labels, filtered down to the rows a model can actually learn from."""
    if horizon.is_classification:
        labels = build_hourly_labels(panel)
        X = build_hourly_features(panel)
        frame = X.join(labels[["close", "next_ret", "target", "gap_next_h"]])
        usable = (
            (frame["gap_next_h"] == 1)          # the next bar really is one hour later
            & frame["next_ret"].notna()
            & (frame["next_ret"] != 0)          # an unchanged close has no direction
        )
        frame = frame[usable]
        target, ret = "target", "next_ret"
    else:
        labels = build_daily_labels(panel.close)
        X = build_daily_features(panel.close, panel.volume)
        frame = X.join(labels)
        target, ret = "next_change", "next_change"

    features = list(X.columns)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    return Dataset(frame=frame, features=features, target=target, ret=ret, price="close")


def latest_features(horizon: Horizon, panel: Panel, feature_names: list[str]) -> pd.DataFrame:
    """The single most recent fully-formed feature row, for live scoring."""
    X = build_hourly_features(panel) if horizon.is_classification else \
        build_daily_features(panel.close, panel.volume)
    X = X[feature_names].replace([np.inf, -np.inf], np.nan).dropna()
    if X.empty:
        raise RuntimeError(
            "no complete feature row: the download is too short for the rolling windows"
        )
    return X.iloc[[-1]]
