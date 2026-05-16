"""Tests for newsmood.embed."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from newsmood.embed import (
    deterministic_dummy_embedder,
    embed_dataframe,
    embed_doc_ids,
    make_news_collection,
    news_pipeline,
    populate_collection,
)


# -- deterministic embedder --------------------------------------------------


def test_dummy_embedder_deterministic():
    e = deterministic_dummy_embedder(dim=8)
    assert e("hello") == e("hello")
    assert len(e("hello")) == 8


def test_dummy_embedder_different_texts_different_vecs():
    e = deterministic_dummy_embedder(dim=8)
    assert e("a") != e("b")


# -- embed_doc_ids ----------------------------------------------------------


def test_embed_doc_ids_caches_misses():
    cache: dict[str, list[float]] = {}
    e = deterministic_dummy_embedder(dim=4)

    def batch_embedder(texts):
        return [e(t) for t in texts]

    out = embed_doc_ids(
        [("id1", "hello"), ("id2", "world")], cache=cache, embedder=batch_embedder
    )
    assert set(out.keys()) == {"id1", "id2"}
    assert len(cache) == 2
    # Re-run should hit cache
    calls = []

    def counting(texts):
        calls.append(len(texts))
        return [e(t) for t in texts]

    out2 = embed_doc_ids(
        [("id1", "hello"), ("id2", "world")], cache=cache, embedder=counting
    )
    assert calls == []
    assert out == out2


def test_embed_doc_ids_partial_hits():
    e = deterministic_dummy_embedder(dim=4)
    cache: dict[str, list[float]] = {"id1": e("seed")}

    def batch_embedder(texts):
        return [e(t) for t in texts]

    out = embed_doc_ids(
        [("id1", "hello"), ("id2", "world")], cache=cache, embedder=batch_embedder
    )
    assert out["id1"] == e("seed")
    assert out["id2"] == e("world")
    assert set(cache.keys()) == {"id1", "id2"}


def test_embed_doc_ids_dedupes_input():
    e = deterministic_dummy_embedder(dim=4)
    cache: dict[str, list[float]] = {}

    seen_batches = []

    def batch_embedder(texts):
        seen_batches.append(list(texts))
        return [e(t) for t in texts]

    out = embed_doc_ids(
        [("id1", "hello"), ("id1", "hello"), ("id1", "hello")],
        cache=cache,
        embedder=batch_embedder,
    )
    assert set(out.keys()) == {"id1"}
    # Only one batch of one text
    assert seen_batches == [["hello"]]


def test_embed_doc_ids_empty():
    assert embed_doc_ids([], cache={}) == {}


def test_embed_doc_ids_embedder_mismatch_raises():
    cache: dict[str, list[float]] = {}

    def bad_embedder(texts):
        return [[0.0]]  # returns 1 vec for 2 texts

    with pytest.raises(RuntimeError, match="Embedder yielded"):
        embed_doc_ids(
            [("id1", "a"), ("id2", "b")], cache=cache, embedder=bad_embedder
        )


# -- embed_dataframe --------------------------------------------------------


def _toy_df():
    return pd.DataFrame(
        [
            {
                "doc_id": "d1",
                "ts": datetime(2025, 3, 13, 9, tzinfo=timezone.utc),
                "text_to_embed": "Earnings miss",
                "source": "newsdata",
                "query": "Earnings_Miss",
            },
            {
                "doc_id": "d2",
                "ts": datetime(2025, 3, 13, 15, tzinfo=timezone.utc),
                "text_to_embed": "Profit warning",
                "source": "newsdata",
                "query": "Profit_Warning",
            },
        ]
    )


def test_embed_dataframe_adds_vector_column():
    e = deterministic_dummy_embedder(dim=4)

    def batch(texts):
        return [e(t) for t in texts]

    cache: dict[str, list[float]] = {}
    out = embed_dataframe(_toy_df(), cache=cache, embedder=batch)
    assert "vector" in out.columns
    assert out["vector"].iloc[0] == e("Earnings miss")
    assert out["vector"].iloc[1] == e("Profit warning")
    # Cache populated
    assert set(cache.keys()) == {"d1", "d2"}


def test_embed_dataframe_does_not_mutate_input():
    e = deterministic_dummy_embedder(dim=4)
    df = _toy_df()

    def batch(texts):
        return [e(t) for t in texts]

    embed_dataframe(df, cache={}, embedder=batch)
    assert "vector" not in df.columns


def test_embed_dataframe_empty():
    out = embed_dataframe(pd.DataFrame(columns=["doc_id", "text_to_embed"]), cache={})
    assert "vector" in out.columns


# -- collection population ---------------------------------------------------


def test_populate_collection_writes_rows():
    e = deterministic_dummy_embedder(dim=4)

    def batch(texts):
        return [e(t) for t in texts]

    df = embed_dataframe(_toy_df(), cache={}, embedder=batch)
    col = make_news_collection("test_news", backend="memory", embedder=e)
    n = populate_collection(df, collection=col)
    assert n == 2
    assert len(col) == 2
    ids_in_window = [d.id for d in col.query_window()]
    assert ids_in_window == ["d1", "d2"]


def test_populate_collection_metadata_present():
    e = deterministic_dummy_embedder(dim=4)

    def batch(texts):
        return [e(t) for t in texts]

    df = embed_dataframe(_toy_df(), cache={}, embedder=batch)
    col = make_news_collection("test_news_meta", backend="memory", embedder=e)
    populate_collection(df, collection=col)
    d1 = col["d1"]
    assert d1.metadata["source"] == "newsdata"
    assert d1.metadata["query"] == "Earnings_Miss"
    # ts is ISO-stamped by TimeIndexedCollection
    assert d1.metadata["ts"].startswith("2025-03-13T09:00:00")


def test_populate_collection_skips_missing_text_or_ts():
    e = deterministic_dummy_embedder(dim=4)
    df = pd.DataFrame(
        [
            {
                "doc_id": "ok",
                "ts": datetime(2025, 3, 13, 9, tzinfo=timezone.utc),
                "text_to_embed": "good",
                "vector": e("good"),
            },
            {"doc_id": "no_ts", "ts": None, "text_to_embed": "x", "vector": e("x")},
            {
                "doc_id": "no_text",
                "ts": datetime(2025, 3, 13, 9, tzinfo=timezone.utc),
                "text_to_embed": "",
                "vector": e("x"),
            },
        ]
    )
    col = make_news_collection("test_skip", backend="memory", embedder=e)
    n = populate_collection(df, collection=col)
    assert n == 1
    assert list(col) == ["ok"]


def test_news_pipeline_end_to_end():
    e = deterministic_dummy_embedder(dim=4)

    def batch(texts):
        return [e(t) for t in texts]

    col = news_pipeline(
        _toy_df(),
        cache={},
        collection_name="pipe_test",
        embedder=batch,
        instance_embedder=e,
    )
    assert len(col) == 2
    # Daily window count
    out = [v for _, _, v in col.window_iter("1d")]
    assert out == [2]
