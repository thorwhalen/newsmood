"""Tests for newsmood.tickers."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from newsmood.embed import deterministic_dummy_embedder
from newsmood.tickers import (
    DEFAULT_UNIVERSE,
    TickerMatcher,
    attach_tickers,
    explode_to_ticker_rows,
    match_tickers,
    per_ticker_session_features,
)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def test_match_by_ticker():
    assert match_tickers("AAPL beat estimates", universe=DEFAULT_UNIVERSE) == ["AAPL"]


def test_match_by_dollar_prefix():
    assert match_tickers("$AAPL up 2%", universe=DEFAULT_UNIVERSE) == ["AAPL"]


def test_match_by_name():
    assert match_tickers(
        "Apple Inc. announced new chips", universe=DEFAULT_UNIVERSE
    ) == ["AAPL"]


def test_match_case_insensitive_name():
    assert match_tickers("apple is a fruit", universe={"AAPL": ["Apple"]}) == ["AAPL"]


def test_multiple_matches():
    out = match_tickers(
        "AAPL and MSFT both rallied; TSLA dipped", universe=DEFAULT_UNIVERSE
    )
    assert set(out) == {"AAPL", "MSFT", "TSLA"}


def test_no_false_positive_substring():
    # 'APPLES' contains 'APPL' but should not match AAPL by name nor by ticker token
    assert match_tickers("APPLES on sale today", universe={"AAPL": ["Apple"]}) == []


def test_no_false_positive_ticker_substring():
    # 'KOOL' contains 'KO' but should not match
    assert match_tickers(
        "KOOL beverage Co", universe={"KO": ["Coca-Cola"]}
    ) == []


def test_ticker_at_string_boundary():
    assert "AAPL" in match_tickers("AAPL", universe={"AAPL": ["Apple"]})


def test_name_with_optional_suffix():
    matcher = TickerMatcher({"WMT": ["Walmart"]})
    assert matcher("Walmart Inc. reported strong sales") == ["WMT"]
    assert matcher("Walmart announced") == ["WMT"]
    assert matcher("Walmart Corporation") == ["WMT"]


def test_alphabet_google_alias():
    out = match_tickers("Alphabet reported.", universe=DEFAULT_UNIVERSE)
    assert "GOOGL" in out
    out2 = match_tickers("Google search update", universe=DEFAULT_UNIVERSE)
    assert "GOOGL" in out2


def test_empty_text():
    assert match_tickers("", universe=DEFAULT_UNIVERSE) == []


# ---------------------------------------------------------------------------
# attach_tickers
# ---------------------------------------------------------------------------


def test_attach_tickers_adds_column():
    df = pd.DataFrame(
        {"doc_id": ["a", "b"], "text_to_embed": ["AAPL beat", "no mentions"]}
    )
    out = attach_tickers(df)
    assert "tickers" in out.columns
    assert out["tickers"].iloc[0] == ["AAPL"]
    assert out["tickers"].iloc[1] == []


def test_attach_tickers_does_not_mutate():
    df = pd.DataFrame(
        {"doc_id": ["a"], "text_to_embed": ["AAPL beat"]}
    )
    attach_tickers(df)
    assert "tickers" not in df.columns


# ---------------------------------------------------------------------------
# explode_to_ticker_rows
# ---------------------------------------------------------------------------


def test_explode_creates_row_per_ticker():
    df = pd.DataFrame(
        [
            {"doc_id": "a", "tickers": ["AAPL", "MSFT"]},
            {"doc_id": "b", "tickers": ["NVDA"]},
            {"doc_id": "c", "tickers": []},
        ]
    )
    out = explode_to_ticker_rows(df)
    assert len(out) == 3
    assert sorted(out["ticker"]) == ["AAPL", "MSFT", "NVDA"]


# ---------------------------------------------------------------------------
# per_ticker_session_features
# ---------------------------------------------------------------------------


def _toy_per_ticker_panel():
    e = deterministic_dummy_embedder(dim=8)
    rows = [
        # Mon: 2 AAPL, 1 MSFT
        {"doc_id": "a1", "ticker": "AAPL", "session": date(2025, 3, 10),
         "vector": e("apple x"), "query": "Earnings_Miss"},
        {"doc_id": "a2", "ticker": "AAPL", "session": date(2025, 3, 10),
         "vector": e("apple y"), "query": "Profit_Warning"},
        {"doc_id": "m1", "ticker": "MSFT", "session": date(2025, 3, 10),
         "vector": e("microsoft x"), "query": "Acquisition"},
        # Tue: 1 AAPL, 1 NVDA
        {"doc_id": "a3", "ticker": "AAPL", "session": date(2025, 3, 11),
         "vector": e("apple z"), "query": "Upgrade"},
        {"doc_id": "n1", "ticker": "NVDA", "session": date(2025, 3, 11),
         "vector": e("nvidia x"), "query": "Earnings_Miss"},
    ]
    return pd.DataFrame(rows)


def test_per_ticker_features_panel_shape():
    feats = per_ticker_session_features(_toy_per_ticker_panel())
    assert feats.index.names == ["ticker", "session"]
    assert ("AAPL", date(2025, 3, 10)) in feats.index
    assert feats.loc[("AAPL", date(2025, 3, 10)), "news_count"] == 2
    assert feats.loc[("MSFT", date(2025, 3, 10)), "news_count"] == 1
    assert feats.loc[("AAPL", date(2025, 3, 11)), "news_count"] == 1


def test_per_ticker_features_with_seeds():
    e = deterministic_dummy_embedder(dim=8)
    seeds = {"x": e("seed1"), "y": e("seed2")}
    feats = per_ticker_session_features(_toy_per_ticker_panel(), seed_vecs=seeds)
    assert "cos_x" in feats.columns
    assert "cos_y" in feats.columns


def test_per_ticker_features_query_counts():
    feats = per_ticker_session_features(_toy_per_ticker_panel())
    # AAPL on Mon has both Earnings_Miss and Profit_Warning
    assert feats.loc[("AAPL", date(2025, 3, 10)), "q_Earnings_Miss"] == 1
    assert feats.loc[("AAPL", date(2025, 3, 10)), "q_Profit_Warning"] == 1
    assert feats.loc[("AAPL", date(2025, 3, 10)), "q_Acquisition"] == 0
