"""Embed canonical-DataFrame rows and load them into a vd collection.

This module is deliberately small: it composes three primitives that live in
other ecosystem packages —

- :func:`aix.embeddings.batched_embeddings` for batched API calls.
- :class:`vd.TimeIndexedCollection` for time-windowed vector retrieval.
- ``dol``-style ``MutableMapping`` stores for persistent caching keyed by
  :func:`newsmood.ingest.make_doc_id`.

The persistent cache is the SSOT for embeddings; the vd collection is a view
that can be rebuilt at any time from ``cache + canonical DataFrame``.

Typical flow
------------
.. code-block:: python

    from newsmood.ingest import ingest_searches, DEFAULT_SEARCHES_ROOT
    from newsmood.embed import default_embedding_cache, embed_dataframe, populate_collection, make_news_collection
    import dol

    store = dol.JsonFiles(DEFAULT_SEARCHES_ROOT)
    df = ingest_searches(store, source='newsdata', since='2025-03-13', until='2025-03-14')

    cache = default_embedding_cache()
    df = embed_dataframe(df, cache=cache)              # adds 'vector' column

    col = make_news_collection('newsdata')             # in-memory vd collection
    populate_collection(df, collection=col)            # writes Documents

    # Then:
    list(col.window_iter('1d', reducer=len))
"""

import os
import pathlib
from collections.abc import Iterable, MutableMapping
from typing import Any, Callable, Optional, Sequence

import pandas as pd


DEFAULT_CACHE_ROOT = os.environ.get(
    "NEWSMOOD_EMBEDDING_CACHE",
    str(pathlib.Path("~/.config/newsmood/embeddings").expanduser()),
)


def default_embedding_cache(
    root: str = DEFAULT_CACHE_ROOT,
) -> MutableMapping[str, list[float]]:
    """Return a pickle-coded dol Files store, creating the root if missing.

    Keys are document ids; values are vectors (``list[float]``). Each entry
    is stored in its own file, named ``<doc_id>``.
    """
    from dol import Files
    from dol.kv_codecs import ValueCodecs

    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    return ValueCodecs.pickle()(Files(root))


def deterministic_dummy_embedder(*, dim: int = 16) -> Callable[[str], list[float]]:
    """Return a deterministic toy embedder for tests / cost-free dry runs.

    >>> emb = deterministic_dummy_embedder(dim=4)
    >>> v1 = emb('hello')
    >>> v2 = emb('hello')
    >>> v1 == v2
    True
    >>> len(v1)
    4
    """
    import hashlib

    def _embed(text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Repeat-extend then truncate, normalize to roughly [-1, 1].
        raw = (h * ((dim // len(h)) + 1))[:dim]
        return [b / 128.0 - 1.0 for b in raw]

    _embed.dim = dim  # type: ignore[attr-defined]
    return _embed


# -- Core embedding routine --------------------------------------------------


def embed_doc_ids(
    pairs: Iterable[tuple[str, str]],
    *,
    cache: MutableMapping[str, list[float]],
    model: Optional[str] = None,
    batch_size: int = 512,
    embedder: Optional[Callable[[list[str]], Iterable[Sequence[float]]]] = None,
    on_batch: Optional[Callable[[int, int], None]] = None,
) -> dict[str, list[float]]:
    """Embed ``(doc_id, text)`` pairs, hitting the cache by ``doc_id``.

    Misses are batched and embedded via ``embedder``. By default
    ``aix.batched_embeddings`` is used. Each new vector is written back to the
    cache.

    Parameters
    ----------
    pairs
        Iterable of ``(doc_id, text)`` tuples. Duplicate ``doc_id``s are OK;
        only the first text is embedded.
    cache
        ``MutableMapping[str, list[float]]`` keyed by ``doc_id``.
    model
        Embedding model name. Ignored when ``embedder`` is given.
    batch_size
        Batch size when calling the default embedder.
    embedder
        Optional callable ``Iterable[str] -> Iterable[Vector]``. Useful for
        tests (see :func:`deterministic_dummy_embedder`).
    on_batch
        Optional progress callback ``(batch_index, batch_size) -> None``.

    Returns
    -------
    dict[doc_id, vector]
        Vectors for every input ``doc_id``.
    """
    pairs_list = list(pairs)
    if not pairs_list:
        return {}

    result: dict[str, list[float]] = {}
    seen_ids: set[str] = set()
    miss_ids: list[str] = []
    miss_texts: list[str] = []

    for did, text in pairs_list:
        if did in seen_ids:
            continue
        seen_ids.add(did)
        try:
            vec = cache[did]
        except KeyError:
            miss_ids.append(did)
            miss_texts.append(text)
        else:
            result[did] = vec

    if miss_texts:
        # Stream writes to the cache as each vector arrives — so a crash
        # mid-run doesn't lose all API spend, and a separate watcher process
        # can observe progress by counting cache entries.
        if embedder is None:
            from aix.embeddings import batched_embeddings

            vec_iter = batched_embeddings(
                miss_texts,
                model=model,
                batch_size=batch_size,
                on_batch=on_batch,
            )
        else:
            def _iter_via_embedder():
                for i, batch_start in enumerate(range(0, len(miss_texts), batch_size)):
                    batch = miss_texts[batch_start : batch_start + batch_size]
                    if on_batch is not None:
                        on_batch(i, len(batch))
                    yield from embedder(batch)

            vec_iter = _iter_via_embedder()

        n_written = 0
        for did, vec in zip(miss_ids, vec_iter):
            v = list(vec)
            cache[did] = v
            result[did] = v
            n_written += 1

        if n_written != len(miss_ids):
            raise RuntimeError(
                f"Embedder yielded {n_written} vectors for {len(miss_ids)} texts"
            )

    return result


def embed_dataframe(
    df: pd.DataFrame,
    *,
    cache: MutableMapping[str, list[float]],
    model: Optional[str] = None,
    batch_size: int = 512,
    embedder: Optional[Callable[[list[str]], Iterable[Sequence[float]]]] = None,
    on_batch: Optional[Callable[[int, int], None]] = None,
    text_col: str = "text_to_embed",
    id_col: str = "doc_id",
    vector_col: str = "vector",
) -> pd.DataFrame:
    """Add a ``vector`` column to ``df`` by embedding ``text_col`` via cache.

    A defensive copy is made; the input ``df`` is not mutated.
    """
    if df.empty:
        out = df.copy()
        out[vector_col] = pd.Series(dtype=object)
        return out

    pairs = list(zip(df[id_col].astype(str), df[text_col].astype(str)))
    id_to_vec = embed_doc_ids(
        pairs,
        cache=cache,
        model=model,
        batch_size=batch_size,
        embedder=embedder,
        on_batch=on_batch,
    )
    out = df.copy()
    out[vector_col] = out[id_col].map(id_to_vec)
    return out


# -- vd collection helpers ---------------------------------------------------


def make_news_collection(
    name: str = "news",
    *,
    backend: str = "memory",
    embedder: Optional[Callable[[str], list[float]]] = None,
    ts_field: str = "ts",
    **backend_kwargs,
):
    """Create (or reopen) a :class:`vd.TimeIndexedCollection` for news.

    For memory and chroma backends, idempotently obtains the underlying
    collection — ``get`` if it exists, else ``create``.
    """
    import vd

    if embedder is None:
        embedder = deterministic_dummy_embedder()  # safe default; no API cost

    client = vd.connect(backend, embedding_model=embedder, **backend_kwargs)
    try:
        base = client.get_collection(name)
    except KeyError:
        base = client.create_collection(name)
    return vd.TimeIndexedCollection(base, ts_field=ts_field)


def populate_collection(
    df: pd.DataFrame,
    *,
    collection,
    vector_col: str = "vector",
    id_col: str = "doc_id",
    text_col: str = "text_to_embed",
    ts_col: str = "ts",
    metadata_fields: Optional[Sequence[str]] = None,
) -> int:
    """Write ``df`` rows into a ``TimeIndexedCollection``. Returns rows written.

    The ``ts`` column is stamped into metadata under the collection's
    ``ts_field``; additional metadata fields are copied from ``df``.
    """
    from vd import Document

    if df.empty:
        return 0

    ts_field = getattr(collection, "_ts_field", "ts")
    default_meta_fields = ("source", "query", "source_id", "category", "country", "language", "link")
    meta_fields = tuple(metadata_fields) if metadata_fields is not None else default_meta_fields

    written = 0
    for row in df.itertuples(index=False):
        rowd = row._asdict()
        text = rowd.get(text_col)
        if not text:
            continue
        ts = rowd.get(ts_col)
        if ts is None or pd.isna(ts):
            continue
        # Serialize ts → ISO string (matches what TimeIndexedCollection does anyway).
        if hasattr(ts, "isoformat"):
            ts_val: Any = ts.isoformat()
        else:
            ts_val = str(ts)
        metadata: dict[str, Any] = {ts_field: ts_val}
        for f in meta_fields:
            if f in rowd:
                v = rowd[f]
                if isinstance(v, float) and v != v:  # NaN
                    continue
                metadata[f] = v
        vec = rowd.get(vector_col)
        if vec is not None and not isinstance(vec, list):
            try:
                vec = list(vec)
            except TypeError:
                vec = None
        doc = Document(
            id=str(rowd[id_col]),
            text=str(text),
            vector=vec,
            metadata=metadata,
        )
        collection[doc.id] = doc
        written += 1
    return written


# -- Convenience pipeline ----------------------------------------------------


def news_pipeline(
    df: pd.DataFrame,
    *,
    cache: Optional[MutableMapping[str, list[float]]] = None,
    collection_name: str = "news",
    backend: str = "memory",
    model: Optional[str] = None,
    embedder: Optional[Callable[[list[str]], Iterable[Sequence[float]]]] = None,
    instance_embedder: Optional[Callable[[str], list[float]]] = None,
    batch_size: int = 512,
    on_batch: Optional[Callable[[int, int], None]] = None,
):
    """Single-call: embed a DataFrame and load it into a vd collection.

    Returns the populated :class:`vd.TimeIndexedCollection`. Uses
    ``deterministic_dummy_embedder`` as the collection's at-query embedding
    function unless ``instance_embedder`` is given — keep that in mind when
    calling :meth:`search`/``search_window`` later.
    """
    if cache is None:
        cache = default_embedding_cache()
    df_embedded = embed_dataframe(
        df,
        cache=cache,
        model=model,
        batch_size=batch_size,
        embedder=embedder,
        on_batch=on_batch,
    )
    col = make_news_collection(
        collection_name, backend=backend, embedder=instance_embedder
    )
    populate_collection(df_embedded, collection=col)
    return col


__all__ = [
    "DEFAULT_CACHE_ROOT",
    "default_embedding_cache",
    "deterministic_dummy_embedder",
    "embed_doc_ids",
    "embed_dataframe",
    "make_news_collection",
    "populate_collection",
    "news_pipeline",
]
