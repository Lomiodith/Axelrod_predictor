"""Static configuration: what we download, where it lands, how we split and score.

Nothing here does work. It is the single place to change a ticker, a cache path or a
cross-validation setting without hunting through the rest of the package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

# joblib probes physical cores with `wmic`, which Windows 11 no longer ships; this skips
# the probe and the noisy warning that comes with it. Must happen before sklearn imports.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max((os.cpu_count() or 2) // 2, 1)))

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class Horizon:
    """Everything that distinguishes one forecasting problem from the other."""

    name: str
    task: str                     # "classification" | "regression"
    interval: str                 # yfinance interval string
    tickers: Mapping[str, str]    # yfinance symbol -> friendly column name
    cache_name: str
    artifact_name: str
    period: str | None            # yfinance `period` for the training download
    start: str | None             # ...or `start`, for series that go back further
    live_period: str              # shorter download used when scoring the latest bar
    ffill_panel: bool             # forward-fill the whole panel on load?
    test_size: float              # share of the sample held out, taken from the end
    cv_splits: int
    cv_gap: int                   # bars left between each CV train and validation window
    selection_metric: str         # CV column used to pick the winner (higher is better)
    cost_bps: float               # friction charged per position change in the backtest

    @property
    def cache_path(self) -> Path:
        return DATA_DIR / self.cache_name

    @property
    def artifact_path(self) -> Path:
        return MODEL_DIR / self.artifact_name

    @property
    def is_classification(self) -> bool:
        return self.task == "classification"


DAILY = Horizon(
    name="daily",
    task="regression",
    interval="1d",
    tickers={
        "CL=F": "wti",          # the thing we forecast
        "BZ=F": "brent",        # the other big crude benchmark
        "RB=F": "gasoline",     # what refineries sell
        "ES=F": "sp500",        # risk appetite
        "^VIX": "vix",          # fear gauge
        "DX-Y.NYB": "dollar",   # oil is priced in dollars
    },
    cache_name="daily_raw.pkl",
    artifact_name="wti_next_day.joblib",
    period=None,
    start="2010-01-01",
    live_period="1y",
    ffill_panel=True,
    test_size=0.2,
    cv_splits=5,
    cv_gap=5,
    selection_metric="beats_rw_by",
    cost_bps=2.0,
)

HOURLY = Horizon(
    name="hourly",
    task="classification",
    interval="1h",
    tickers={
        "CL=F": "wti",
        "BZ=F": "brent",
        "RB=F": "gasoline",
        "HO=F": "heating_oil",
        "ES=F": "sp500",
        "^VIX": "vix",
    },
    cache_name="hourly_raw.pkl",
    artifact_name="wti_next_hour.joblib",
    period="730d",              # Yahoo caps hourly history at roughly two years
    start=None,
    live_period="60d",
    ffill_panel=False,          # the hourly feature builder forward-fills where it needs to
    test_size=0.2,
    cv_splits=5,
    cv_gap=48,                  # longest rolling window is 168 bars; 48 kills most overlap
    selection_metric="roc_auc",
    cost_bps=1.0,
)

HORIZONS: dict[str, Horizon] = {h.name: h for h in (DAILY, HOURLY)}

TARGET_TICKER = "wti"
