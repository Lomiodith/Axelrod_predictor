"""Download, cache and reshape Yahoo Finance data into a tidy panel.

yfinance hands back a DataFrame with a two-level column index ``(field, ticker)``.
Everything downstream wants one frame per field with friendly column names, so that
reshaping happens here once and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf

from wti.config import DATA_DIR, TARGET_TICKER, Horizon

FIELDS = ("Close", "Open", "High", "Low", "Volume")


@dataclass(frozen=True)
class Panel:
    """One frame per OHLCV field, columns renamed to ``wti``, ``brent``, ... .

    Every frame shares the same index: the timestamps on which WTI actually printed a
    bar. That is the calendar we forecast on; other markets are aligned onto it.
    """

    close: pd.DataFrame
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame

    @property
    def index(self) -> pd.Index:
        return self.close.index

    def __len__(self) -> int:
        return len(self.close)


def download(horizon: Horizon, period: str | None = None) -> pd.DataFrame:
    """Fetch raw multi-index data from Yahoo. No caching, no reshaping."""
    kwargs: dict[str, object] = {
        "interval": horizon.interval,
        "auto_adjust": False,
        "progress": False,
        # Serial download: parallel threads race on yfinance's empty SQLite cache in a
        # fresh container ("database is locked"). A few seconds slower, same data.
        "threads": False,
    }
    if period is not None:
        kwargs["period"] = period
    elif horizon.period is not None:
        kwargs["period"] = horizon.period
    else:
        kwargs["start"] = horizon.start
    return yf.download(list(horizon.tickers), **kwargs)


def load_raw(horizon: Horizon, force: bool = False) -> pd.DataFrame:
    """Return the raw download, from ``data/*.pkl`` when available.

    The cache is what makes the rest of the package reproducible offline. Pass
    ``force=True`` (``--refresh`` on the command line) to re-download and overwrite it.
    """
    cache: Path = horizon.cache_path
    if cache.exists() and not force:
        print(f"Loaded cached {horizon.interval} data from {cache}")
        return pd.read_pickle(cache)

    raw = download(horizon)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_pickle(cache)
    print(f"Downloaded and cached to {cache}")
    return raw


def to_panel(raw: pd.DataFrame, horizon: Horizon) -> Panel:
    """Reshape a raw download into a :class:`Panel` aligned on the WTI calendar."""
    tickers = dict(horizon.tickers)
    frames: dict[str, pd.DataFrame] = {}
    for field in FIELDS:
        df = raw[field].copy()
        # Restrict to the tickers this horizon asked for: a cache may hold extras.
        df = df[[t for t in tickers if t in df.columns]]
        df.columns = [tickers[t] for t in df.columns]
        frames[field] = df

    # Keep only timestamps on which WTI traded, then align the rest onto that calendar.
    mask = frames["Close"][TARGET_TICKER].notna()
    frames = {k: v[mask] for k, v in frames.items()}
    if horizon.ffill_panel:
        frames = {k: v.ffill() for k, v in frames.items()}

    return Panel(
        close=frames["Close"], open=frames["Open"], high=frames["High"],
        low=frames["Low"], volume=frames["Volume"],
    )


def load_panel(horizon: Horizon, force: bool = False) -> Panel:
    """Convenience: :func:`load_raw` followed by :func:`to_panel`."""
    return to_panel(load_raw(horizon, force=force), horizon)


def fresh_panel(horizon: Horizon, period: str | None = None) -> Panel:
    """A small, always-current panel for scoring the latest bar (never cached)."""
    return to_panel(download(horizon, period=period or horizon.live_period), horizon)


def coverage(panel: Panel) -> pd.DataFrame:
    """Per-market data-quality summary. Worth a look after any refresh."""
    return pd.DataFrame({
        "bars": panel.close.count(),
        "missing_share": panel.close.isna().mean().round(3),
        "zero_volume_share": panel.volume.eq(0).mean().round(3),
    })
