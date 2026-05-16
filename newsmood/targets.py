"""Price targets for news-driven prediction.

Provides:

- :func:`get_ohlcv`: cached daily OHLCV via ``yfinance``.
- :func:`forward_returns`: log forward returns over one or more horizons.
- :func:`align_news_to_session`: map a news timestamp to its "anchor session" —
  the first market close strictly after the news. This is the leakage-safe
  anchor at which we *know* the news, so any return computed *forward* from
  that close is a valid label.

Session-alignment rules (US equities)
-------------------------------------
The market close is 16:00 America/New_York (US Eastern Time). Given a news
timestamp ``t``:

- Convert ``t`` to US/Eastern.
- If ``t.date()`` is itself a trading day **and** ``t.time() < 16:00 ET``,
  the anchor session is ``t.date()``: the news is in-session and we use that
  day's close (still in the future relative to ``t``).
- Otherwise, the anchor session is the **next** trading day strictly after
  ``t.date()``.

The forward-h return at anchor session ``s`` is
``log(close[s + h_sessions] / close[s])`` — strictly forward-looking from ``s``.

Examples
--------
>>> import pandas as pd
>>> sessions = pd.to_datetime(['2025-03-12', '2025-03-13', '2025-03-14', '2025-03-17']).date
>>> from newsmood.targets import align_news_to_session
>>> # Wed 09:30 ET — before close on a trading day -> same-day session
>>> align_news_to_session('2025-03-12T09:30:00-04:00', sessions)
datetime.date(2025, 3, 12)
>>> # Wed 17:00 ET — after close -> Thursday session
>>> align_news_to_session('2025-03-12T17:00:00-04:00', sessions)
datetime.date(2025, 3, 13)
>>> # Saturday news -> Monday session
>>> align_news_to_session('2025-03-15T10:00:00-04:00', sessions)
datetime.date(2025, 3, 17)
"""

import logging
import os
import pathlib
from bisect import bisect_right
from datetime import date, datetime, time
from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd


DEFAULT_OHLCV_CACHE = os.environ.get(
    "NEWSMOOD_OHLCV_CACHE",
    str(pathlib.Path("~/.config/newsmood/ohlcv").expanduser()),
)

DEFAULT_MARKET_TZ = "America/New_York"
DEFAULT_CLOSE_TIME = time(16, 0)
DEFAULT_TICKERS = ("SPY",)
DEFAULT_HORIZONS = (1, 5, 20)

_logger = logging.getLogger(__name__)


# -- Session alignment -------------------------------------------------------


def _coerce_ts(ts: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tz is None:
        out = out.tz_localize("UTC")
    return out


def align_news_to_session(
    news_ts: Union[str, datetime, pd.Timestamp],
    sessions: Iterable[date],
    *,
    market_tz: str = DEFAULT_MARKET_TZ,
    close_time: time = DEFAULT_CLOSE_TIME,
) -> Optional[date]:
    """Return the anchor-session date for a single news timestamp.

    ``sessions`` is an iterable of trading-day ``date`` objects in ascending order.
    Returns ``None`` if no session lies on or after the news timestamp.

    See module docstring for the alignment rule.
    """
    sessions_list = sorted(sessions)
    if not sessions_list:
        return None

    ts = _coerce_ts(news_ts).tz_convert(market_tz)
    d = ts.date()
    t = ts.time()

    sessions_set = set(sessions_list)
    if t < close_time and d in sessions_set:
        return d

    # Next session strictly after d.
    i = bisect_right(sessions_list, d)
    if i >= len(sessions_list):
        return None
    return sessions_list[i]


def align_news_series_to_sessions(
    news_ts: pd.Series,
    sessions: Iterable[date],
    *,
    market_tz: str = DEFAULT_MARKET_TZ,
    close_time: time = DEFAULT_CLOSE_TIME,
) -> pd.Series:
    """Vectorized variant of :func:`align_news_to_session` over a Series.

    Returns a Series of ``date`` objects (or NaT-equivalent: ``None``) aligned
    with ``news_ts.index``.
    """
    return news_ts.map(
        lambda x: align_news_to_session(
            x, sessions, market_tz=market_tz, close_time=close_time
        )
    )


# -- OHLCV pull + cache ------------------------------------------------------


def _cache_path(ticker: str, root: str) -> pathlib.Path:
    return pathlib.Path(root) / f"{ticker.upper()}.parquet"


def _ensure_cache_dir(root: str) -> None:
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)


def get_ohlcv(
    ticker: str,
    *,
    start: Union[str, date, datetime] = "2020-01-01",
    end: Optional[Union[str, date, datetime]] = None,
    cache_root: str = DEFAULT_OHLCV_CACHE,
    force_refresh: bool = False,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Return a daily OHLCV DataFrame for ``ticker``, with a parquet cache.

    The cache is keyed by ticker and stores the full available history; on
    subsequent calls a slice is returned without hitting the network unless
    ``force_refresh=True`` or the requested ``end`` extends beyond cached data.

    Returns
    -------
    DataFrame with columns ``[Open, High, Low, Close, Volume]`` and a
    ``DatetimeIndex`` of session dates (timezone-naive, US/Eastern in spirit).
    """
    _ensure_cache_dir(cache_root)
    cache_file = _cache_path(ticker, cache_root)
    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp.utcnow()

    cached: Optional[pd.DataFrame] = None
    if cache_file.exists() and not force_refresh:
        try:
            cached = pd.read_parquet(cache_file)
        except Exception as e:
            _logger.warning("Failed to read OHLCV cache for %s: %s", ticker, e)
            cached = None

    need_fetch = (
        force_refresh
        or cached is None
        or cached.empty
        or cached.index.max() < (end_ts - pd.Timedelta(days=2))
    )

    if need_fetch:
        import yfinance as yf

        fetch_start = pd.Timestamp(start)
        if cached is not None and not cached.empty:
            # Refetch only the tail
            fetch_start = max(fetch_start, cached.index.max() - pd.Timedelta(days=7))
        new = yf.download(
            ticker,
            start=fetch_start.strftime("%Y-%m-%d"),
            end=end_ts.strftime("%Y-%m-%d") if end is not None else None,
            auto_adjust=auto_adjust,
            progress=False,
            threads=False,
        )
        if isinstance(new.columns, pd.MultiIndex):
            new.columns = new.columns.get_level_values(0)
        if not new.empty:
            new.index = pd.to_datetime(new.index).tz_localize(None)
            if cached is not None and not cached.empty:
                full = pd.concat([cached, new])
                full = full[~full.index.duplicated(keep="last")].sort_index()
            else:
                full = new
            try:
                full.to_parquet(cache_file)
            except Exception as e:
                _logger.warning("Failed to write OHLCV cache for %s: %s", ticker, e)
            cached = full

    if cached is None or cached.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    mask = pd.Series(True, index=cached.index)
    if start is not None:
        mask &= cached.index >= pd.Timestamp(start)
    if end is not None:
        mask &= cached.index <= end_ts
    return cached.loc[mask].copy()


def forward_returns(
    prices: pd.DataFrame,
    *,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    price_col: str = "Close",
) -> pd.DataFrame:
    """Compute forward log returns over the given session-count horizons.

    The return at session ``s`` for horizon ``h`` is
    ``log(close[s + h] / close[s])``. Sessions near the tail with insufficient
    forward data are ``NaN``.

    Returns a DataFrame indexed like ``prices``, with one column per horizon
    named ``ret_<h>d``.
    """
    if prices.empty or price_col not in prices.columns:
        return pd.DataFrame(index=prices.index)
    close = prices[price_col].astype(float)
    out = pd.DataFrame(index=prices.index)
    log_close = np.log(close)
    for h in horizons:
        out[f"ret_{h}d"] = log_close.shift(-int(h)) - log_close
    return out


# -- High-level: news → target frame ----------------------------------------


def attach_targets(
    news_df: pd.DataFrame,
    *,
    tickers: Iterable[str] = DEFAULT_TICKERS,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    start: Optional[Union[str, date, datetime]] = None,
    end: Optional[Union[str, date, datetime]] = None,
    cache_root: str = DEFAULT_OHLCV_CACHE,
    force_refresh: bool = False,
    ts_col: str = "ts",
    market_tz: str = DEFAULT_MARKET_TZ,
    close_time: time = DEFAULT_CLOSE_TIME,
) -> pd.DataFrame:
    """Attach forward-return columns to ``news_df`` via session alignment.

    For each ticker ``T`` and horizon ``h``, a column ``{T}_ret_{h}d`` is added
    holding the forward log return at the anchor session of each news row.
    A ``session`` column is also added (the anchor session date) for the
    first ticker — sessions for all tickers should agree within US equities.

    Parameters
    ----------
    news_df
        DataFrame with at least a tz-aware ``ts`` column (UTC or any tz).
    tickers
        Universe to compute returns for. Default: ``("SPY",)``.
    horizons
        Session-count horizons.
    start, end
        Optional bounds for fetching OHLCV. Defaults: span of ``news_df.ts``.

    Returns
    -------
    DataFrame with the original news rows plus added columns. Rows with no
    valid anchor session (e.g. ``ts`` past the available OHLCV history) are
    kept with NaN target columns — callers should drop NaNs before fitting.
    """
    if news_df.empty:
        return news_df.copy()

    if start is None:
        start = (news_df[ts_col].min() - pd.Timedelta(days=10)).date()
    if end is None:
        end = (news_df[ts_col].max() + pd.Timedelta(days=30)).date()

    tickers = tuple(tickers)
    horizons = tuple(int(h) for h in horizons)

    # Get all prices first
    prices_by_ticker: dict[str, pd.DataFrame] = {}
    rets_by_ticker: dict[str, pd.DataFrame] = {}
    for t in tickers:
        p = get_ohlcv(
            t,
            start=start,
            end=end,
            cache_root=cache_root,
            force_refresh=force_refresh,
        )
        if p.empty:
            _logger.warning("No price data for %s", t)
            continue
        prices_by_ticker[t] = p
        rets_by_ticker[t] = forward_returns(p, horizons=horizons)

    if not prices_by_ticker:
        out = news_df.copy()
        out["session"] = pd.NaT
        for t in tickers:
            for h in horizons:
                out[f"{t}_ret_{h}d"] = np.nan
        return out

    primary = next(iter(prices_by_ticker))
    sessions = [d.date() if hasattr(d, "date") else d for d in prices_by_ticker[primary].index]
    aligned_sessions = align_news_series_to_sessions(
        news_df[ts_col], sessions, market_tz=market_tz, close_time=close_time
    )

    out = news_df.copy()
    out["session"] = aligned_sessions
    session_idx = pd.to_datetime(aligned_sessions)
    for t in tickers:
        if t not in rets_by_ticker:
            for h in horizons:
                out[f"{t}_ret_{h}d"] = np.nan
            continue
        rets = rets_by_ticker[t]
        for h in horizons:
            col = f"ret_{h}d"
            if col in rets.columns:
                out[f"{t}_ret_{h}d"] = session_idx.map(rets[col].to_dict())
            else:
                out[f"{t}_ret_{h}d"] = np.nan
    return out


__all__ = [
    "DEFAULT_OHLCV_CACHE",
    "DEFAULT_MARKET_TZ",
    "DEFAULT_CLOSE_TIME",
    "DEFAULT_TICKERS",
    "DEFAULT_HORIZONS",
    "align_news_to_session",
    "align_news_series_to_sessions",
    "get_ohlcv",
    "forward_returns",
    "attach_targets",
]
