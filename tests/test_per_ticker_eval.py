"""Tests for newsmood.per_ticker_eval (offline-only — monkeypatches get_ohlcv)."""

from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd
import pytest

from newsmood.embed import deterministic_dummy_embedder
from newsmood.per_ticker_eval import (
    attach_targets_per_ticker,
    per_ticker_evaluate,
    per_ticker_panel,
)


# ---------------------------------------------------------------------------
# per_ticker_panel
# ---------------------------------------------------------------------------


def _news_with_tickers(n_per_day: int = 5, days: int = 30):
    """Synthetic news where each row mentions AAPL or MSFT."""
    e = deterministic_dummy_embedder(dim=16)
    rows = []
    base = date(2025, 1, 1)
    for d_off in range(days):
        sess = pd.Timestamp(base) + pd.Timedelta(days=d_off)
        for i in range(n_per_day):
            text = f"Apple {i}" if i % 2 == 0 else f"Microsoft {i}"
            rows.append({
                "doc_id": f"{d_off}-{i}",
                "ts": sess,
                "session": sess.date(),
                "vector": e(text),
                "text_to_embed": text,
                "query": "test",
            })
    return pd.DataFrame(rows)


def test_per_ticker_panel_min_articles_filter():
    df = _news_with_tickers(n_per_day=5, days=10)  # 50 articles total
    # Only AAPL & MSFT: high threshold should drop both
    panel = per_ticker_panel(df, min_articles_per_ticker=1000)
    assert panel.empty


def test_per_ticker_panel_normal():
    df = _news_with_tickers(n_per_day=10, days=20)  # 200 articles
    panel = per_ticker_panel(df, min_articles_per_ticker=5)
    assert not panel.empty
    assert panel.index.names == ["ticker", "session"]
    tickers = set(panel.index.get_level_values("ticker"))
    assert tickers.issubset({"AAPL", "MSFT"})


def test_per_ticker_panel_with_seed_vecs():
    df = _news_with_tickers(n_per_day=10, days=20)
    e = deterministic_dummy_embedder(dim=16)
    seeds = {"x": e("foo")}
    panel = per_ticker_panel(df, seed_vecs=seeds, min_articles_per_ticker=5)
    assert "cos_x" in panel.columns


def test_per_ticker_panel_requires_session():
    df = pd.DataFrame({"doc_id": ["a"], "vector": [[0.1]], "text_to_embed": ["AAPL"]})
    with pytest.raises(ValueError, match="session"):
        per_ticker_panel(df)


def test_per_ticker_panel_empty():
    df = pd.DataFrame(columns=["doc_id", "ts", "session", "vector", "text_to_embed"])
    panel = per_ticker_panel(df)
    assert panel.empty


# ---------------------------------------------------------------------------
# attach_targets_per_ticker  (offline: monkeypatch get_ohlcv)
# ---------------------------------------------------------------------------


def _fake_prices_factory(start_close: float = 100.0, n: int = 60):
    """Return a callable that produces a synthetic OHLCV per ticker."""
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    closes = np.exp(np.arange(n) * 0.001) * start_close
    prices = pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 0},
        index=idx,
    )
    return prices


def test_attach_targets_per_ticker_offline(monkeypatch):
    import newsmood.targets as targets_mod

    # Build a synthetic panel
    sessions = pd.date_range("2025-01-01", periods=20, freq="B").date.tolist()
    panel = pd.DataFrame(
        {
            "news_count": np.ones(40, dtype=int),
            "cos_x": np.random.default_rng(0).normal(size=40),
        },
        index=pd.MultiIndex.from_product([["AAPL", "MSFT"], sessions], names=["ticker", "session"]),
    )
    monkeypatch.setattr(targets_mod, "get_ohlcv", lambda *a, **k: _fake_prices_factory())

    out = attach_targets_per_ticker(panel, horizons=(1,))
    assert "ret_1d" in out.columns
    # Most rows should have valid returns
    assert out["ret_1d"].notna().mean() > 0.7


def test_attach_targets_per_ticker_empty():
    out = attach_targets_per_ticker(pd.DataFrame())
    assert out.empty


# ---------------------------------------------------------------------------
# per_ticker_evaluate
# ---------------------------------------------------------------------------


def _eval_panel(n_sessions: int = 120, seed: int = 0):
    """A two-ticker MultiIndex panel with synthetic signal."""
    rng = np.random.default_rng(seed)
    sessions = pd.date_range("2025-01-01", periods=n_sessions, freq="B").date.tolist()
    rows = []
    for ticker in ("AAPL", "MSFT"):
        f = rng.normal(size=n_sessions)
        target = (0.5 if ticker == "AAPL" else -0.3) * f + rng.normal(scale=0.5, size=n_sessions)
        for s, fv, t in zip(sessions, f, target):
            rows.append({"ticker": ticker, "session": s, "cos_x": fv, "news_count": 1, "ret_1d": t})
    df = pd.DataFrame(rows).set_index(["ticker", "session"])
    return df


def test_per_ticker_evaluate_returns_per_ticker_rows():
    panel = _eval_panel(n_sessions=120, seed=42)
    out = per_ticker_evaluate(panel, target="ret_1d", feature_cols=["cos_x", "news_count"],
                              n_splits=3, min_train=40, embargo=1)
    assert set(out.index) == {"AAPL", "MSFT"}
    for col in ("mean_ic", "bt_sharpe", "bt_final_equity", "bt_hit_rate"):
        assert col in out.columns


def test_per_ticker_evaluate_recovers_signal():
    """AAPL has a positive signal; should have higher mean IC than MSFT (negative)."""
    panel = _eval_panel(n_sessions=200, seed=1)
    out = per_ticker_evaluate(panel, target="ret_1d", feature_cols=["cos_x", "news_count"],
                              n_splits=4, min_train=50, embargo=1)
    assert out.loc["AAPL", "mean_ic"] > out.loc["MSFT", "mean_ic"]


def test_per_ticker_evaluate_empty():
    out = per_ticker_evaluate(pd.DataFrame())
    assert out.empty


def test_per_ticker_evaluate_requires_multiindex():
    bad = pd.DataFrame({"x": [1, 2], "ret_1d": [0.1, 0.2]})
    with pytest.raises(ValueError, match="MultiIndex"):
        per_ticker_evaluate(bad)
