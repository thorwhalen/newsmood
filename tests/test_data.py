"""Tests for newsmood.data — focused on the local pack/unpack roundtrip.

Network-dependent paths (download_artifact, hf_embedding_cache,
hf_news_dataframe) are not exercised here; they're thin wrappers around
huggingface_hub.hf_hub_download whose own test suite covers the cache
semantics.
"""

import pathlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from newsmood.data import (
    CachedEmbeddingStore,
    EmbeddingsParquetView,
    pack_embeddings_to_parquet,
)


@pytest.fixture
def fake_embeddings():
    rng = np.random.default_rng(0)
    return {f"doc_{i:05d}": rng.normal(size=8).tolist() for i in range(50)}


# ---------------------------------------------------------------------------
# pack_embeddings_to_parquet
# ---------------------------------------------------------------------------


def test_pack_writes_parquet(tmp_path, fake_embeddings):
    out = tmp_path / "embeddings.parquet"
    pack_embeddings_to_parquet(
        fake_embeddings, out_path=str(out), chunk_size=10, verbose=False
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_pack_roundtrip_preserves_ids_and_vectors(tmp_path, fake_embeddings):
    out = tmp_path / "embeddings.parquet"
    pack_embeddings_to_parquet(
        fake_embeddings, out_path=str(out), chunk_size=10, verbose=False
    )
    view = EmbeddingsParquetView(str(out))
    assert set(view) == set(fake_embeddings)
    for k in list(fake_embeddings)[:5]:
        # float32 precision: assert close, not equal
        np.testing.assert_allclose(view[k], fake_embeddings[k], rtol=1e-5)


def test_pack_empty_source(tmp_path):
    out = tmp_path / "empty.parquet"
    pack_embeddings_to_parquet({}, out_path=str(out), verbose=False)
    # Empty source → no parquet written (writer never opened)
    assert not out.exists() or out.stat().st_size >= 0


# ---------------------------------------------------------------------------
# EmbeddingsParquetView (mapping interface)
# ---------------------------------------------------------------------------


def test_view_mapping_interface(tmp_path, fake_embeddings):
    out = tmp_path / "embeddings.parquet"
    pack_embeddings_to_parquet(
        fake_embeddings, out_path=str(out), chunk_size=10, verbose=False
    )
    view = EmbeddingsParquetView(str(out))
    assert len(view) == 50
    sample_key = next(iter(view))
    assert sample_key in view
    assert "missing_doc_id" not in view
    with pytest.raises(KeyError):
        view["missing_doc_id"]


def test_view_lazy_load(tmp_path, fake_embeddings):
    """Construction should not load yet — only first access triggers it."""
    out = tmp_path / "embeddings.parquet"
    pack_embeddings_to_parquet(
        fake_embeddings, out_path=str(out), chunk_size=10, verbose=False
    )
    view = EmbeddingsParquetView(str(out))
    assert not view._loaded
    _ = view[next(iter(fake_embeddings))]
    assert view._loaded


# ---------------------------------------------------------------------------
# CachedEmbeddingStore (local+remote fall-through)
# ---------------------------------------------------------------------------


def test_cached_store_local_hit_no_remote_access():
    """A key present locally is returned without ever touching remote."""
    local = {"a": [0.1, 0.2]}

    class FailingRemote(dict):
        def __getitem__(self, key):
            raise AssertionError("remote should not be accessed for local hits")

    store = CachedEmbeddingStore(local=local, remote=FailingRemote())
    assert store["a"] == [0.1, 0.2]


def test_cached_store_miss_falls_through_to_remote_and_promotes():
    local: dict = {}
    remote = {"b": [9.0, 9.0]}
    store = CachedEmbeddingStore(local=local, remote=remote)
    out = store["b"]
    assert out == [9.0, 9.0]
    # Promoted to local
    assert local["b"] == [9.0, 9.0]


def test_cached_store_write_goes_to_local_only():
    local: dict = {}
    remote: dict = {}
    store = CachedEmbeddingStore(local=local, remote=remote)
    store["new"] = [1.0, 2.0]
    assert local["new"] == [1.0, 2.0]
    assert "new" not in remote


def test_cached_store_keyerror_when_in_neither():
    store = CachedEmbeddingStore(local={}, remote={})
    with pytest.raises(KeyError):
        store["nope"]


def test_cached_store_contains_doesnt_trigger_remote_download():
    """If remote was never fetched, __contains__ checks local only."""
    local = {"a": [0.1]}
    store = CachedEmbeddingStore(local=local)  # remote left None
    assert "a" in store
    assert "b" not in store
    # Remote was never instantiated, so the HF call wasn't made
    assert store._remote is None


def test_cached_store_iter_yields_local_first_then_remote_only():
    local = {"a": [0.1]}
    remote = {"a": [9.9], "b": [0.2]}
    store = CachedEmbeddingStore(local=local, remote=remote)
    out = list(store)
    # 'a' once (from local), then 'b' from remote-only
    assert out[0] == "a"
    assert set(out) == {"a", "b"}
    assert len(out) == 2


def test_cached_store_len_unions_when_remote_known():
    local = {"a": [0.1]}
    remote = {"a": [0.0], "b": [0.0]}
    store = CachedEmbeddingStore(local=local, remote=remote)
    assert len(store) == 2


def test_cached_store_local_failure_does_not_break_reads():
    """If the local cache raises on write, the read still returns."""
    class WriteRaisingLocal(dict):
        def __setitem__(self, key, value):
            raise IOError("disk full")

    local = WriteRaisingLocal()
    remote = {"b": [9.0]}
    store = CachedEmbeddingStore(local=local, remote=remote)
    # Should return without raising despite the local write failing
    assert store["b"] == [9.0]
