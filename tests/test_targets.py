"""Tests for newsmood.targets — focused on session alignment and forward returns.

OHLCV fetching is not tested here (would require network).
"""

from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from newsmood.targets import (
    align_news_series_to_sessions,
    align_news_to_session,
    attach_targets,
    forward_returns,
)


# ---------------------------------------------------------------------------
# align_news_to_session
# ---------------------------------------------------------------------------


SESSIONS = [
    date(2025, 3, 10),  # Mon
    date(2025, 3, 11),  # Tue
    date(2025, 3, 12),  # Wed
    date(2025, 3, 13),  # Thu
    date(2025, 3, 14),  # Fri
    # Weekend off
    date(2025, 3, 17),  # Mon
    date(2025, 3, 18),
]


class TestAlignToSession:
    def test_intraday_before_close_same_day(self):
        # Wed 09:30 ET — same-session
        assert align_news_to_session(
            "2025-03-12T09:30:00-04:00", SESSIONS
        ) == date(2025, 3, 12)

    def test_intraday_at_close_next_day(self):
        # 16:00 ET exactly — not strictly before close -> next session
        assert align_news_to_session(
            "2025-03-12T16:00:00-04:00", SESSIONS
        ) == date(2025, 3, 13)

    def test_after_close_next_session(self):
        # Wed 17:00 ET -> Thu session
        assert align_news_to_session(
            "2025-03-12T17:00:00-04:00", SESSIONS
        ) == date(2025, 3, 13)

    def test_weekend_news_next_monday(self):
        # Saturday 10:00 ET -> Monday session
        assert align_news_to_session(
            "2025-03-15T10:00:00-04:00", SESSIONS
        ) == date(2025, 3, 17)

    def test_sunday_news_next_monday(self):
        assert align_news_to_session(
            "2025-03-16T20:00:00-04:00", SESSIONS
        ) == date(2025, 3, 17)

    def test_utc_input_converts(self):
        # 2025-03-12T20:30 UTC = 16:30 ET (after close) -> next session
        assert align_news_to_session(
            "2025-03-12T20:30:00Z", SESSIONS
        ) == date(2025, 3, 13)

    def test_naive_ts_assumed_utc(self):
        ts = pd.Timestamp("2025-03-12T13:30:00")  # naive — utc -> 09:30 ET
        assert align_news_to_session(ts, SESSIONS) == date(2025, 3, 12)

    def test_no_future_session_returns_none(self):
        assert align_news_to_session(
            "2030-01-01T12:00:00Z", SESSIONS
        ) is None

    def test_empty_sessions_returns_none(self):
        assert align_news_to_session("2025-03-12T12:00:00Z", []) is None


class TestAlignSeries:
    def test_aligns_each_row(self):
        s = pd.Series(
            [
                pd.Timestamp("2025-03-12T09:30:00-04:00"),
                pd.Timestamp("2025-03-12T17:00:00-04:00"),
                pd.Timestamp("2025-03-15T10:00:00-04:00"),
            ]
        )
        out = align_news_series_to_sessions(s, SESSIONS)
        assert list(out) == [date(2025, 3, 12), date(2025, 3, 13), date(2025, 3, 17)]


# ---------------------------------------------------------------------------
# forward_returns
# ---------------------------------------------------------------------------


def _make_prices(close_vals):
    idx = pd.date_range("2025-03-10", periods=len(close_vals), freq="B")
    return pd.DataFrame({"Close": close_vals}, index=idx)


class TestForwardReturns:
    def test_1d_horizon(self):
        # Close doubling each day → ret = log(2)
        prices = _make_prices([1.0, 2.0, 4.0])
        rets = forward_returns(prices, horizons=[1])
        assert rets["ret_1d"].iloc[0] == pytest.approx(np.log(2.0))
        assert rets["ret_1d"].iloc[1] == pytest.approx(np.log(2.0))
        # Last row has NaN (no forward data)
        assert np.isnan(rets["ret_1d"].iloc[-1])

    def test_multi_horizon(self):
        prices = _make_prices([1.0, 2.0, 4.0, 8.0, 16.0])
        rets = forward_returns(prices, horizons=[1, 2])
        # 1d at index 0: log(2/1)
        assert rets["ret_1d"].iloc[0] == pytest.approx(np.log(2.0))
        # 2d at index 0: log(4/1)
        assert rets["ret_2d"].iloc[0] == pytest.approx(np.log(4.0))

    def test_empty_prices(self):
        rets = forward_returns(pd.DataFrame(columns=["Close"]), horizons=[1])
        assert len(rets) == 0


# ---------------------------------------------------------------------------
# attach_targets (offline — monkeypatch get_ohlcv)
# ---------------------------------------------------------------------------


def _fake_prices():
    idx = pd.to_datetime(
        ["2025-03-10", "2025-03-11", "2025-03-12", "2025-03-13", "2025-03-14"]
    )
    # Geometric: each session up by ~1%
    close = np.exp(np.arange(len(idx)) * 0.01)
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 0},
        index=idx,
    )


def test_attach_targets_offline(monkeypatch):
    import newsmood.targets as m

    monkeypatch.setattr(m, "get_ohlcv", lambda *a, **k: _fake_prices())

    news = pd.DataFrame(
        [
            {"doc_id": "a", "ts": pd.Timestamp("2025-03-11T09:30:00-04:00")},  # Tue intraday
            {"doc_id": "b", "ts": pd.Timestamp("2025-03-12T17:00:00-04:00")},  # Wed after close
        ]
    )
    out = attach_targets(news, tickers=("SPY",), horizons=(1,))
    assert "session" in out.columns
    assert out["session"].tolist() == [date(2025, 3, 11), date(2025, 3, 13)]
    # 1-day forward return is log(1.01) ~ 0.00995 at every session in our synthetic series
    assert out["SPY_ret_1d"].iloc[0] == pytest.approx(0.01, rel=1e-2)
    assert out["SPY_ret_1d"].iloc[1] == pytest.approx(0.01, rel=1e-2)


def test_attach_targets_empty():
    out = attach_targets(pd.DataFrame(columns=["doc_id", "ts"]), tickers=("SPY",))
    assert out.empty
