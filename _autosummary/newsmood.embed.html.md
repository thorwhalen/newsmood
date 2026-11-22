# newsmood.embed

Embed canonical-DataFrame rows and load them into a vd collection.

This module is deliberately small: it composes three primitives that live in
other ecosystem packages —

- `aix.embeddings.batched_embeddings()` for batched API calls.
- `vd.TimeIndexedCollection` for time-windowed vector retrieval.
- `dol`-style `MutableMapping` stores for persistent caching keyed by
  [`newsmood.ingest.make_doc_id()`](newsmood.ingest.html.md#newsmood.ingest.make_doc_id).

The persistent cache is the SSOT for embeddings; the vd collection is a view
that can be rebuilt at any time from `cache + canonical DataFrame`.

## Typical flow

```python
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
```

### Functions

| [`default_embedding_cache`](#newsmood.embed.default_embedding_cache)([root])                | Return a pickle-coded dol Files store, creating the root if missing.   |
|-------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| [`deterministic_dummy_embedder`](#newsmood.embed.deterministic_dummy_embedder)(\*[, dim])        | Return a deterministic toy embedder for tests / cost-free dry runs.    |
| [`embed_doc_ids`](#newsmood.embed.embed_doc_ids)(pairs, \*, cache[, model, ...])  | Embed `(doc_id, text)` pairs, hitting the cache by `doc_id`.           |
| [`embed_dataframe`](#newsmood.embed.embed_dataframe)(df, \*, cache[, model, ...])   | Add a `vector` column to `df` by embedding `text_col` via cache.       |
| [`make_news_collection`](#newsmood.embed.make_news_collection)([name, backend, ...])     | Create (or reopen) a `vd.TimeIndexedCollection` for news.              |
| [`populate_collection`](#newsmood.embed.populate_collection)(df, \*, collection[, ...]) | Write `df` rows into a `TimeIndexedCollection`.                        |
| [`news_pipeline`](#newsmood.embed.news_pipeline)(df, \*[, cache, ...])            | Single-call: embed a DataFrame and load it into a vd collection.       |

### newsmood.embed.default_embedding_cache(root='/home/runner/.config/newsmood/embeddings')

Return a pickle-coded dol Files store, creating the root if missing.

Keys are document ids; values are vectors (`list[float]`). Each entry
is stored in its own file, named `<doc_id>`.

* **Return type:**
  [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]

### newsmood.embed.deterministic_dummy_embedder(, dim=16)

Return a deterministic toy embedder for tests / cost-free dry runs.

```pycon
>>> emb = deterministic_dummy_embedder(dim=4)
>>> v1 = emb('hello')
>>> v2 = emb('hello')
>>> v1 == v2
True
>>> len(v1)
4
```

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]

### newsmood.embed.embed_dataframe(df, , cache, model=None, batch_size=512, embedder=None, on_batch=None, text_col='text_to_embed', id_col='doc_id', vector_col='vector')

Add a `vector` column to `df` by embedding `text_col` via cache.

A defensive copy is made; the input `df` is not mutated.

* **Return type:**
  `DataFrame`

### newsmood.embed.embed_doc_ids(pairs, , cache, model=None, batch_size=512, embedder=None, on_batch=None)

Embed `(doc_id, text)` pairs, hitting the cache by `doc_id`.

Misses are batched and embedded via `embedder`. By default
`aix.batched_embeddings` is used. Each new vector is written back to the
cache.

* **Parameters:**
  * **pairs** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Iterable of `(doc_id, text)` tuples. Duplicate 

    ```
    ``
    ```

    doc_id\`\`s are OK;
    only the first text is embedded.
  * **cache** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – `MutableMapping[str, list[float]]` keyed by `doc_id`.
  * **model** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Embedding model name. Ignored when `embedder` is given.
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Batch size when calling the default embedder.
  * **embedder** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]]]) – Optional callable `Iterable[str] -> Iterable[Vector]`. Useful for
    tests (see [`deterministic_dummy_embedder()`](#newsmood.embed.deterministic_dummy_embedder)).
  * **on_batch** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`None`](https://docs.python.org/3/builtins/constants.html#None)]]) – Optional progress callback `(batch_index, batch_size) -> None`.
* **Returns:**
  Vectors for every input `doc_id`.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]

### newsmood.embed.make_news_collection(name='news', , backend='memory', embedder=None, ts_field='ts', \*\*backend_kwargs)

Create (or reopen) a `vd.TimeIndexedCollection` for news.

For memory and chroma backends, idempotently obtains the underlying
collection — `get` if it exists, else `create`.

### newsmood.embed.news_pipeline(df, , cache=None, collection_name='news', backend='memory', model=None, embedder=None, instance_embedder=None, batch_size=512, on_batch=None)

Single-call: embed a DataFrame and load it into a vd collection.

Returns the populated `vd.TimeIndexedCollection`. Uses
`deterministic_dummy_embedder` as the collection’s at-query embedding
function unless `instance_embedder` is given — keep that in mind when
calling `search()`/`search_window` later.

### newsmood.embed.populate_collection(df, , collection, vector_col='vector', id_col='doc_id', text_col='text_to_embed', ts_col='ts', metadata_fields=None)

Write `df` rows into a `TimeIndexedCollection`. Returns rows written.

The `ts` column is stamped into metadata under the collection’s
`ts_field`; additional metadata fields are copied from `df`.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)
