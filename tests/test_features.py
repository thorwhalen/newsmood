"""Tests for newsmood.features."""

from datetime import date
import math

import numpy as np
import pandas as pd
import pytest

from newsmood.embed import deterministic_dummy_embedder
from newsmood.features import (
    DEFAULT_SEED_PROMPTS,
    build_seed_embeddings,
    cosine_to_seeds,
    panel_with_targets,
    session_features,
)
from newsmood.ingest import make_doc_id


# ---------------------------------------------------------------------------
# cosine_to_seeds
# ---------------------------------------------------------------------------


def test_cosine_to_seeds_identity():
    mean = np.array([[1.0, 0.0]])
    seeds = np.array([[1.0, 0.0], [0.0, 1.0]])
    cos = cosine_to_seeds(mean, seeds)
    assert cos.shape == (1, 2)
    assert cos[0, 0] == pytest.approx(1.0)
    assert cos[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_cosine_to_seeds_handles_zero_vec():
    mean = np.array([[0.0, 0.0]])
    seeds = np.array([[1.0, 0.0]])
    # Should be 0 (not NaN)
    cos = cosine_to_seeds(mean, seeds)
    assert math.isfinite(cos[0, 0])


# ---------------------------------------------------------------------------
# build_seed_embeddings
# ---------------------------------------------------------------------------


def test_build_seed_embeddings_uses_shared_cache():
    e = deterministic_dummy_embedder(dim=8)

    def batch(texts):
        return [e(t) for t in texts]

    cache: dict[str, list[float]] = {}
    seeds = {"a": "first seed", "b": "second seed"}
    out = build_seed_embeddings(seeds, cache=cache, embedder=batch)
    assert set(out.keys()) == {"a", "b"}
    # Cache holds entries keyed by make_doc_id of seed texts
    assert make_doc_id("first seed") in cache
    assert make_doc_id("second seed") in cache
    # Vectors are deterministic
    assert out["a"] == e("first seed")


# ---------------------------------------------------------------------------
# session_features
# ---------------------------------------------------------------------------


def _toy_news_with_vectors():
    e = deterministic_dummy_embedder(dim=8)
    rows = [
        # Session 1: two earnings-miss-ish articles
        {
            "doc_id": "a",
            "session": date(2025, 3, 13),
            "query": "Earnings_Miss",
            "text_to_embed": "Earnings miss for Acme",
            "vector": e("Earnings miss for Acme"),
        },
        {
            "doc_id": "b",
            "session": date(2025, 3, 13),
            "query": "Profit_Warning",
            "text_to_embed": "Profit warning issued",
            "vector": e("Profit warning issued"),
        },
        # Session 2: one acquisition
        {
            "doc_id": "c",
            "session": date(2025, 3, 14),
            "query": "Acquisition",
            "text_to_embed": "Big acquisition announced",
            "vector": e("Big acquisition announced"),
        },
    ]
    return pd.DataFrame(rows)


def test_session_features_basic_counts():
    feats = session_features(_toy_news_with_vectors())
    assert list(feats.index) == [date(2025, 3, 13), date(2025, 3, 14)]
    assert feats.loc[date(2025, 3, 13), "news_count"] == 2
    assert feats.loc[date(2025, 3, 14), "news_count"] == 1


def test_session_features_query_counts():
    feats = session_features(_toy_news_with_vectors())
    assert feats.loc[date(2025, 3, 13), "q_Earnings_Miss"] == 1
    assert feats.loc[date(2025, 3, 13), "q_Profit_Warning"] == 1
    assert feats.loc[date(2025, 3, 13), "q_Acquisition"] == 0
    assert feats.loc[date(2025, 3, 14), "q_Acquisition"] == 1


def test_session_features_with_seeds():
    e = deterministic_dummy_embedder(dim=8)
    seed_vecs = {"x": e("foo"), "y": e("bar")}
    feats = session_features(_toy_news_with_vectors(), seed_vecs=seed_vecs)
    for col in ("cos_x", "cos_y"):
        assert col in feats.columns
        # cosines in [-1, 1]
        assert ((feats[col] >= -1.0001) & (feats[col] <= 1.0001)).all()


def test_session_features_include_mean_components():
    feats = session_features(
        _toy_news_with_vectors(), include_mean_vector_components=True
    )
    # Default dummy embedder dim = 8 above
    for i in range(8):
        assert f"mean_vec_{i}" in feats.columns


def test_session_features_drops_no_session_or_no_vector():
    df = _toy_news_with_vectors()
    df.loc[len(df)] = {
        "doc_id": "missing_session",
        "session": None,
        "query": "X",
        "text_to_embed": "...",
        "vector": [0.0] * 8,
    }
    df.loc[len(df)] = {
        "doc_id": "missing_vec",
        "session": date(2025, 3, 13),
        "query": "X",
        "text_to_embed": "...",
        "vector": None,
    }
    feats = session_features(df)
    # Two valid sessions; the new doc with missing session is dropped
    assert list(feats.index) == [date(2025, 3, 13), date(2025, 3, 14)]
    # Session 13 count unchanged (the missing_vec row was filtered)
    assert feats.loc[date(2025, 3, 13), "news_count"] == 2


def test_session_features_empty():
    feats = session_features(pd.DataFrame())
    assert feats.empty


# ---------------------------------------------------------------------------
# panel_with_targets
# ---------------------------------------------------------------------------


def test_panel_joins_targets():
    df = _toy_news_with_vectors()
    df["SPY_ret_1d"] = [0.01, 0.01, -0.02]
    df["SPY_ret_5d"] = [0.05, 0.05, -0.10]
    panel = panel_with_targets(df)
    assert "SPY_ret_1d" in panel.columns
    assert "SPY_ret_5d" in panel.columns
    assert panel.loc[date(2025, 3, 13), "SPY_ret_1d"] == 0.01
    assert panel.loc[date(2025, 3, 14), "SPY_ret_5d"] == -0.10


def test_panel_explicit_target_cols():
    df = _toy_news_with_vectors()
    df["my_target"] = [1.0, 1.0, 2.0]
    df["other"] = ["x", "y", "z"]
    panel = panel_with_targets(df, target_cols=["my_target"])
    assert "my_target" in panel.columns
    assert "other" not in panel.columns
