# newsmood.targets

Price targets for news-driven prediction.

Provides:

- [`get_ohlcv()`](#newsmood.targets.get_ohlcv): cached daily OHLCV via `yfinance`.
- [`forward_returns()`](#newsmood.targets.forward_returns): log forward returns over one or more horizons.
- [`align_news_to_session()`](#newsmood.targets.align_news_to_session): map a news timestamp to its “anchor session” —
  the first market close strictly after the news. This is the leakage-safe
  anchor at which we *know* the news, so any return computed *forward* from
  that close is a valid label.

## Session-alignment rules (US equities)

The market close is 16:00 America/New_York (US Eastern Time). Given a news
timestamp `t`:

- Convert `t` to US/Eastern.
- If `t.date()` is itself a trading day **and** `t.time() < 16:00 ET`,
  the anchor session is `t.date()`: the news is in-session and we use that
  day’s close (still in the future relative to `t`).
- Otherwise, the anchor session is the **next** trading day strictly after
  `t.date()`.

The forward-h return at anchor session `s` is
`log(close[s + h_sessions] / close[s])` — strictly forward-looking from `s`.

### Examples

```pycon
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
```

### Functions

| [`align_news_to_session`](#newsmood.targets.align_news_to_session)(news_ts, sessions, \*)       | Return the anchor-session date for a single news timestamp.                                                   |
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| [`align_news_series_to_sessions`](#newsmood.targets.align_news_series_to_sessions)(news_ts, ...)        | Vectorized variant of [`align_news_to_session()`](#newsmood.targets.align_news_to_session) over a Series. |
| [`get_ohlcv`](#newsmood.targets.get_ohlcv)(ticker, \*[, start, end, ...])           | Return a daily OHLCV DataFrame for `ticker`, with a parquet cache.                                            |
| [`forward_returns`](#newsmood.targets.forward_returns)(prices, \*[, horizons, price_col]) | Compute forward log returns over the given session-count horizons.                                            |
| [`attach_targets`](#newsmood.targets.attach_targets)(news_df, \*[, tickers, ...])        | Attach forward-return columns to `news_df` via session alignment.                                             |

### newsmood.targets.align_news_series_to_sessions(news_ts, sessions, , market_tz='America/New_York', close_time=datetime.time(16, 0))

Vectorized variant of [`align_news_to_session()`](#newsmood.targets.align_news_to_session) over a Series.

Returns a Series of `date` objects (or NaT-equivalent: `None`) aligned
with `news_ts.index`.

* **Return type:**
  `Series`

### newsmood.targets.align_news_to_session(news_ts, sessions, , market_tz='America/New_York', close_time=datetime.time(16, 0))

Return the anchor-session date for a single news timestamp.

`sessions` is an iterable of trading-day `date` objects in ascending order.
Returns `None` if no session lies on or after the news timestamp.

See module docstring for the alignment rule.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`date`](https://docs.python.org/3/library/datetime.html#datetime.date)]

### newsmood.targets.attach_targets(news_df, , tickers=('SPY',), horizons=(1, 5, 20), start=None, end=None, cache_root='/home/runner/.config/newsmood/ohlcv', force_refresh=False, ts_col='ts', market_tz='America/New_York', close_time=datetime.time(16, 0))

Attach forward-return columns to `news_df` via session alignment.

For each ticker `T` and horizon `h`, a column `{T}_ret_{h}d` is added
holding the forward log return at the anchor session of each news row.
A `session` column is also added (the anchor session date) for the
first ticker — sessions for all tickers should agree within US equities.

* **Parameters:**
  * **news_df** (`DataFrame`) – DataFrame with at least a tz-aware `ts` column (UTC or any tz).
  * **tickers** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Universe to compute returns for. Default: `("SPY",)`.
  * **horizons** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Session-count horizons.
  * **start** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`date`](https://docs.python.org/3/library/datetime.html#datetime.date), [`datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – Optional bounds for fetching OHLCV. Defaults: span of `news_df.ts`.
  * **end** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`date`](https://docs.python.org/3/library/datetime.html#datetime.date), [`datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – Optional bounds for fetching OHLCV. Defaults: span of `news_df.ts`.
* **Return type:**
  `DataFrame`
* **Returns:**
  DataFrame with the original news rows plus added columns. Rows with no
  valid anchor session (e.g. `ts` past the available OHLCV history) are
  kept with NaN target columns — callers should drop NaNs before fitting.

### newsmood.targets.forward_returns(prices, , horizons=(1, 5, 20), price_col='Close')

Compute forward log returns over the given session-count horizons.

The return at session `s` for horizon `h` is
`log(close[s + h] / close[s])`. Sessions near the tail with insufficient
forward data are `NaN`.

Returns a DataFrame indexed like `prices`, with one column per horizon
named `ret_<h>d`.

* **Return type:**
  `DataFrame`

### newsmood.targets.get_ohlcv(ticker, , start='2020-01-01', end=None, cache_root='/home/runner/.config/newsmood/ohlcv', force_refresh=False, auto_adjust=True)

Return a daily OHLCV DataFrame for `ticker`, with a parquet cache.

The cache is keyed by ticker and stores the full available history; on
subsequent calls a slice is returned without hitting the network unless
`force_refresh=True` or the requested `end` extends beyond cached data.

* **Return type:**
  `DataFrame`
* **Returns:**
  DataFrame with columns `[Open, High, Low, Close, Volume]` and a
  `DatetimeIndex` of session dates (timezone-naive, US/Eastern in spirit).
