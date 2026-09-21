"""Shared fixtures.

Every test runs on synthetic data. No network, no dependency on whatever happens to be
sitting in ``data/*.pkl``, so the suite is fast and gives the same answer on any machine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wti.config import DAILY, HOURLY, Horizon
from wti.data import FIELDS, Panel, to_panel


def synthetic_raw(horizon: Horizon, n: int = 600, seed: int = 0) -> pd.DataFrame:
    """A yfinance-shaped download: MultiIndex columns of ``(field, ticker)``."""
    rng = np.random.default_rng(seed)
    if horizon.interval == "1d":
        index = pd.date_range("2020-01-02", periods=n, freq="B")
    else:
        index = pd.date_range("2024-01-02 09:00", periods=n, freq="h", tz="America/New_York")

    columns: dict[tuple[str, str], pd.Series] = {}
    for symbol in horizon.tickers:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        spread = np.abs(rng.normal(0, 0.004, n)) * close
        open_ = close * (1 + rng.normal(0, 0.003, n))
        columns[("Close", symbol)] = pd.Series(close, index=index)
        columns[("Open", symbol)] = pd.Series(open_, index=index)
        columns[("High", symbol)] = pd.Series(np.maximum(open_, close) + spread, index=index)
        columns[("Low", symbol)] = pd.Series(np.minimum(open_, close) - spread, index=index)
        columns[("Volume", symbol)] = pd.Series(rng.integers(1_000, 100_000, n), index=index)

    raw = pd.DataFrame(columns)
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)
    assert set(raw.columns.get_level_values(0)) == set(FIELDS)
    return raw


@pytest.fixture
def daily_panel() -> Panel:
    return to_panel(synthetic_raw(DAILY, n=600), DAILY)


@pytest.fixture
def hourly_panel() -> Panel:
    return to_panel(synthetic_raw(HOURLY, n=700), HOURLY)


@pytest.fixture
def daily_horizon() -> Horizon:
    return DAILY


@pytest.fixture
def hourly_horizon() -> Horizon:
    return HOURLY
