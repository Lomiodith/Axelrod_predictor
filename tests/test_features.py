"""Tests for the finished code.

These already pass. They are here to stay passing: if you change a feature later, this
suite tells you whether you broke the no-look-ahead guarantee. The leakage tests are the
ones that matter -- a feature that peeks at the future is invisible in the metrics
(everything just looks better) and fatal in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wti.data import Panel, to_panel
from wti.features import (build_dataset, build_daily_features, build_daily_labels,
                          build_hourly_features, build_hourly_labels, latest_features, rsi)

from conftest import synthetic_raw


def truncate(panel: Panel, n: int) -> Panel:
    """The same panel as if the data had stopped after ``n`` bars."""
    return Panel(*(getattr(panel, f).iloc[:n] for f in ("close", "open", "high", "low", "volume")))


# ------------------------------------------------------------------------- rsi

def test_rsi_stays_in_range(daily_panel):
    values = rsi(daily_panel.close["wti"]).dropna()
    assert len(values) > 0
    assert values.between(0, 100).all()


def test_rsi_is_causal(daily_panel):
    price = daily_panel.close["wti"]
    full = rsi(price)
    partial = rsi(price.iloc[:300])
    pd.testing.assert_series_equal(full.iloc[:300], partial)


# ------------------------------------------------------------------- no look-ahead

@pytest.mark.parametrize("cut", [200, 350, 500])
def test_daily_features_have_no_lookahead(daily_panel, cut):
    """Row ``t`` must not change when the data after ``t`` is deleted."""
    full = build_daily_features(daily_panel.close, daily_panel.volume)
    short_panel = truncate(daily_panel, cut)
    partial = build_daily_features(short_panel.close, short_panel.volume)
    pd.testing.assert_frame_equal(full.iloc[:cut], partial)


@pytest.mark.parametrize("cut", [300, 500])
def test_hourly_features_have_no_lookahead(hourly_panel, cut):
    full = build_hourly_features(hourly_panel)
    partial = build_hourly_features(truncate(hourly_panel, cut))
    pd.testing.assert_frame_equal(full.iloc[:cut], partial)


# ----------------------------------------------------------------------- labels

def test_daily_label_is_tomorrow(daily_panel):
    labels = build_daily_labels(daily_panel.close)
    close = daily_panel.close["wti"]
    assert labels["next_close"].iloc[0] == pytest.approx(close.iloc[1])
    assert labels["next_close"].iloc[-1] != labels["next_close"].iloc[-1]  # NaN at the end
    expected = close.iloc[1] / close.iloc[0] - 1
    assert labels["next_change"].iloc[0] == pytest.approx(expected)


def test_negative_price_episode_is_dropped(daily_panel):
    """A % change through a negative price is meaningless: those rows lose their label."""
    close = daily_panel.close.copy()
    close.iloc[100, close.columns.get_loc("wti")] = -37.63

    labels = build_daily_labels(close)
    assert labels["next_change"].iloc[99:102].isna().all()
    assert labels["next_change"].iloc[95:99].notna().all()


def test_hourly_label_is_direction(hourly_panel):
    labels = build_hourly_labels(hourly_panel)
    close = hourly_panel.close["wti"]
    went_up = close.iloc[1] > close.iloc[0]
    assert bool(labels["target"].iloc[0]) == bool(went_up)


def test_hourly_gap_filtering(hourly_horizon, hourly_panel):
    """Bars whose successor is not exactly one hour later cannot carry a label."""
    labels = build_hourly_labels(hourly_panel)
    assert (labels["gap_next_h"].dropna() == 1).all()

    # Punch a hole in the series and the bar before it must drop out of the dataset.
    holed = Panel(*(getattr(hourly_panel, f).drop(hourly_panel.index[400])
                    for f in ("close", "open", "high", "low", "volume")))
    dataset = build_dataset(hourly_horizon, holed)
    assert hourly_panel.index[399] not in dataset.frame.index


# ---------------------------------------------------------------------- datasets

def test_daily_dataset_shape(daily_horizon, daily_panel):
    dataset = build_dataset(daily_horizon, daily_panel)
    assert len(dataset.features) == 14
    assert dataset.target == "next_change"
    assert not dataset.frame.isna().any().any()
    assert np.isfinite(dataset.X.to_numpy()).all()
    assert dataset.frame.index.is_monotonic_increasing


def test_hourly_dataset_shape(hourly_horizon, hourly_panel):
    dataset = build_dataset(hourly_horizon, hourly_panel)
    assert len(dataset.features) > 40
    assert dataset.target == "target"
    assert set(dataset.y.unique()) <= {0, 1}
    assert not dataset.frame.isna().any().any()
    assert (dataset.frame["next_ret"] != 0).all()


def test_dataset_drops_the_final_unlabelled_bar(daily_horizon, daily_panel):
    dataset = build_dataset(daily_horizon, daily_panel)
    assert dataset.frame.index.max() < daily_panel.index.max()


# --------------------------------------------------------------- panel and live

def test_to_panel_aligns_on_wti(daily_horizon):
    raw = synthetic_raw(daily_horizon, n=300)
    raw.loc[raw.index[50], ("Close", "CL=F")] = np.nan     # WTI did not trade
    panel = to_panel(raw, daily_horizon)
    assert raw.index[50] not in panel.index
    assert list(panel.close.columns) == list(daily_horizon.tickers.values())


def test_latest_features_is_one_complete_row(daily_horizon, daily_panel):
    dataset = build_dataset(daily_horizon, daily_panel)
    row = latest_features(daily_horizon, daily_panel, dataset.features)
    assert row.shape == (1, 14)
    assert row.notna().all().all()
    # The live row is the last bar, which the dataset cannot use: it has no label yet.
    assert row.index[0] == daily_panel.index.max()
