"""Tests for newsmood.clusters."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from newsmood.clusters import (
    attach_cluster_id,
    fit_news_clusters,
    session_cluster_counts,
)


def _vec(theta: float, dim: int = 4) -> list[float]:
    """A direction-on-circle 4D vector."""
    base = np.array([np.cos(theta), np.sin(theta), 0.0, 0.0])
    return base.tolist()


def _two_cluster_df(n: int = 20):
    """Two well-separated synthetic clusters in 4D."""
    rng = np.random.default_rng(0)
    cluster_a = [_vec(0.0 + 0.01 * rng.normal()) for _ in range(n // 2)]
    cluster_b = [_vec(np.pi + 0.01 * rng.normal()) for _ in range(n // 2)]
    rows = []
    for i, v in enumerate(cluster_a + cluster_b):
        rows.append({"doc_id": f"d{i}", "vector": v, "session": date(2025, 3, 10 + (i % 3))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# fit_news_clusters
# ---------------------------------------------------------------------------


def test_fit_returns_callable_model():
    df = _two_cluster_df()
    model = fit_news_clusters(df, n_clusters=2, random_state=0)
    # Callable on new vectors
    assert callable(model)
    out = model([_vec(0.0), _vec(np.pi)])
    assert len(out) == 2


def test_fit_separates_clusters():
    df = _two_cluster_df()
    model = fit_news_clusters(df, n_clusters=2, random_state=0)
    preds = model([_vec(0.0), _vec(np.pi)])
    # The two well-separated points get different clusters
    assert preds[0] != preds[1]


def test_fit_empty_raises():
    df = pd.DataFrame({"doc_id": ["a"], "vector": [None]})
    with pytest.raises(ValueError):
        fit_news_clusters(df, n_clusters=2)


def test_fit_with_sampling():
    df = _two_cluster_df(n=20)
    model = fit_news_clusters(df, n_clusters=2, sample_size=10, random_state=0)
    assert model.n_clusters == 2


# ---------------------------------------------------------------------------
# attach_cluster_id
# ---------------------------------------------------------------------------


def test_attach_cluster_id_basic():
    df = _two_cluster_df()
    model = fit_news_clusters(df, n_clusters=2, random_state=0)
    out = attach_cluster_id(df, model)
    assert "cluster" in out.columns
    assert set(out["cluster"].unique()).issubset({0, 1})
    # Half should be in each cluster
    counts = out["cluster"].value_counts().sort_index()
    assert len(counts) == 2


def test_attach_cluster_id_handles_missing_vec():
    df = _two_cluster_df()
    df.loc[len(df)] = {"doc_id": "missing", "vector": None, "session": date(2025, 3, 12)}
    model = fit_news_clusters(df.dropna(subset=["vector"]), n_clusters=2, random_state=0)
    out = attach_cluster_id(df, model)
    # Missing-vector row gets -1
    assert out.loc[out["doc_id"] == "missing", "cluster"].iloc[0] == -1


# ---------------------------------------------------------------------------
# session_cluster_counts
# ---------------------------------------------------------------------------


def test_session_cluster_counts_zero_pads_columns():
    df = pd.DataFrame(
        [
            {"session": date(2025, 3, 10), "cluster": 0},
            {"session": date(2025, 3, 10), "cluster": 0},
            {"session": date(2025, 3, 10), "cluster": 1},
            {"session": date(2025, 3, 11), "cluster": 0},
        ]
    )
    out = session_cluster_counts(df, n_clusters=4)
    assert list(out.columns) == ["clust_0", "clust_1", "clust_2", "clust_3"]
    assert out.loc[date(2025, 3, 10), "clust_0"] == 2
    assert out.loc[date(2025, 3, 10), "clust_1"] == 1
    assert out.loc[date(2025, 3, 10), "clust_2"] == 0


def test_session_cluster_counts_excludes_missing():
    df = pd.DataFrame(
        [
            {"session": date(2025, 3, 10), "cluster": 0},
            {"session": date(2025, 3, 10), "cluster": -1},
            {"session": None, "cluster": 0},
        ]
    )
    out = session_cluster_counts(df, n_clusters=2)
    assert out.shape == (1, 2)
    assert out.loc[date(2025, 3, 10), "clust_0"] == 1


def test_session_cluster_counts_empty():
    assert session_cluster_counts(pd.DataFrame(), n_clusters=3).empty
