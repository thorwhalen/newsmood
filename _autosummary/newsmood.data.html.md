# newsmood.data

Pack local artifacts for publishing and download-on-miss from HF.

Two halves:

**Publishing (export side):** repackages the per-vector pickle cache, the
raw JSON snapshot tree, and the ingested canonical DataFrame into single
parquet files suitable for upload to a Hugging Face Dataset.

**Consuming (import side):** a thin read-through wrapper that downloads a
parquet shard from HF on first access, caches it under
`~/.cache/huggingface/hub/`, and exposes a `MutableMapping` view so
existing newsmood code paths (which expect a doc_id → vector mapping)
work without modification.

The default published dataset is `thorwhalen/newsmood-data` on the
Hugging Face Hub. The files served from there:

- `embeddings.parquet`: doc_id (str), vector (list[float32 × 1536]).
  ~1.5 GB compressed for 360k articles.
- `news.parquet`: full canonical ingest output (CANONICAL_COLUMNS).
  ~50 MB.
- `raw/searches.tar.gz`: optional original JSON snapshots, gzipped.
  ~500 MB.
- `ohlcv/<TICKER>.parquet`: per-ticker price files (small).

All inputs default to the local conventions
(`~/.config/newsmood/embeddings/` etc.) so the export functions are
idempotent and re-runnable.

### Functions

| [`cached_embedding_store`](#newsmood.data.cached_embedding_store)(\*[, local, eager_remote])   | Build a [`CachedEmbeddingStore`](#newsmood.data.CachedEmbeddingStore) over the default local cache.   |
|------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| [`download_artifact`](#newsmood.data.download_artifact)(filename, \*[, repo_id, ...])     | Download an HF dataset artifact (cached).                                                                     |
| [`hf_embedding_cache`](#newsmood.data.hf_embedding_cache)(\*[, repo_id, filename, ...])    | Return an [`EmbeddingsParquetView`](#newsmood.data.EmbeddingsParquetView) backed by an HF parquet.     |
| [`hf_news_dataframe`](#newsmood.data.hf_news_dataframe)(\*[, repo_id, filename, ...])     | Load the canonical news DataFrame from an HF dataset.                                                         |
| [`pack_all_to_staging`](#newsmood.data.pack_all_to_staging)(\*[, staging_dir, ...])         | Build every artifact under one staging directory and return paths.                                            |
| [`pack_embeddings_to_parquet`](#newsmood.data.pack_embeddings_to_parquet)([source, dtype, ...])    | Pack a doc_id → vector mapping into a single parquet file.                                                    |
| [`pack_news_to_parquet`](#newsmood.data.pack_news_to_parquet)(\*, out_path[, source, ...])   | Run the full ingest and write the canonical DataFrame as parquet.                                             |
| [`pack_ohlcv_to_dir`](#newsmood.data.pack_ohlcv_to_dir)(\*[, ohlcv_root, verbose])        | Copy existing per-ticker OHLCV parquet files to a staging directory.                                          |
| [`pack_raw_searches_to_tarball`](#newsmood.data.pack_raw_searches_to_tarball)(\*[, ...])             | Tar up the raw JSON snapshot tree (for full reproducibility).                                                 |
| [`upload_staging_to_hf`](#newsmood.data.upload_staging_to_hf)(\*[, staging_dir, ...])        | Push a staging directory to an HF Dataset.                                                                    |

### Classes

| [`CachedEmbeddingStore`](#newsmood.data.CachedEmbeddingStore)([local, remote])   | Local-first, HF-backed embedding cache with write-through semantics.   |
|------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| [`EmbeddingsParquetView`](#newsmood.data.EmbeddingsParquetView)(path)             | Read-only `MutableMapping`-style view over an embeddings parquet.      |

### *class* newsmood.data.CachedEmbeddingStore(local=None, remote=None)

Bases: [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)

Local-first, HF-backed embedding cache with write-through semantics.

The use case: scripts that already expect a `MutableMapping[str, list[float]]`
(e.g. `newsmood.embed.embed_doc_ids`) can transparently consume the
published HF dataset *without re-embedding*. The fall-through order is:

1. **Local cache** (`~/.config/newsmood/embeddings/`). If the key is
   already there, return it.
2. **HF parquet view** (lazy-downloaded on first miss). If the key is
   there, copy it to the local cache and return.
3. **Caller’s embedder** (via `newsmood.embed.embed_doc_ids`), if a
   genuinely new doc_id is requested. Cached locally.

The local store remains the SSOT for newly-computed vectors; the HF view
is read-only.

Writes (`store[doc_id] = vector`) always go to the local cache so any
fresh embeddings persist there for the next run.

### *class* newsmood.data.EmbeddingsParquetView(path)

Bases: [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

Read-only `MutableMapping`-style view over an embeddings parquet.

Loads the entire (doc_id, vector) table into a dict on construction
(~1.5 GB RAM for 360k × 1536 float32 vectors). For lower-memory access,
use the underlying parquet directly.

```pycon
>>> view = EmbeddingsParquetView('embeddings.parquet')
>>> vec = view['<some_doc_id>']
```

### newsmood.data.cached_embedding_store(, local=None, eager_remote=False)

Build a [`CachedEmbeddingStore`](#newsmood.data.CachedEmbeddingStore) over the default local cache.

* **Parameters:**
  * **local** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]]) – Pre-built local cache; defaults to
    [`newsmood.embed.default_embedding_cache()`](newsmood.embed.html.md#newsmood.embed.default_embedding_cache).
  * **eager_remote** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, fetch the HF parquet view immediately (one ~1 GB download
    on first call to a fresh machine). If False (default), defer until a
    miss occurs.
* **Return type:**
  [`CachedEmbeddingStore`](#newsmood.data.CachedEmbeddingStore)

### newsmood.data.download_artifact(filename, , repo_id='thorwhalen/newsmood-data', repo_type='dataset', revision=None)

Download an HF dataset artifact (cached). Returns the local path.

Wraps `huggingface_hub.hf_hub_download()`; subsequent calls hit the
HF cache under `~/.cache/huggingface/hub/`.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### newsmood.data.hf_embedding_cache(, repo_id='thorwhalen/newsmood-data', filename='embeddings.parquet', revision=None)

Return an [`EmbeddingsParquetView`](#newsmood.data.EmbeddingsParquetView) backed by an HF parquet.

Downloads the parquet to the local HF cache on first call; subsequent
calls reuse the cached file.

* **Return type:**
  [`EmbeddingsParquetView`](#newsmood.data.EmbeddingsParquetView)

### newsmood.data.hf_news_dataframe(, repo_id='thorwhalen/newsmood-data', filename='news.parquet', revision=None)

Load the canonical news DataFrame from an HF dataset.

* **Return type:**
  `DataFrame`

### newsmood.data.pack_all_to_staging(, staging_dir='/home/runner/.cache/newsmood/staging', include_raw=True, include_ohlcv=True, verbose=True)

Build every artifact under one staging directory and return paths.

Caller then uploads `staging_dir` to the HF Hub via
[`upload_staging_to_hf()`](#newsmood.data.upload_staging_to_hf).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### newsmood.data.pack_embeddings_to_parquet(source=None, , out_path, dtype='float32', chunk_size=25000, verbose=True)

Pack a doc_id → vector mapping into a single parquet file.

* **Parameters:**
  * **source** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]]) – Any mapping from doc_id (str) to vector (list[float]). Defaults to
    [`newsmood.embed.default_embedding_cache()`](newsmood.embed.html.md#newsmood.embed.default_embedding_cache).
  * **out_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Destination parquet path.
  * **dtype** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Numeric dtype for vectors. `"float32"` halves the size vs
    `float64` at negligible cosine-similarity drift.
  * **chunk_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Rows per write batch (memory bound). Default 25k.
  * **written.** (*Returns the absolute path*)
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### newsmood.data.pack_news_to_parquet(, out_path, source=None, searches_root=None, since=None, until=None, verbose=True)

Run the full ingest and write the canonical DataFrame as parquet.

See [`newsmood.ingest.ingest_searches()`](newsmood.ingest.html.md#newsmood.ingest.ingest_searches) for parameters.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### newsmood.data.pack_ohlcv_to_dir(, ohlcv_root=None, out_dir, verbose=True)

Copy existing per-ticker OHLCV parquet files to a staging directory.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### newsmood.data.pack_raw_searches_to_tarball(, searches_root=None, out_path, compression='gz', verbose=True)

Tar up the raw JSON snapshot tree (for full reproducibility).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### newsmood.data.upload_staging_to_hf(, staging_dir='/home/runner/.cache/newsmood/staging', repo_id='thorwhalen/newsmood-data', commit_message='Update artifacts', private=False, token=None, verbose=True)

Push a staging directory to an HF Dataset.

Requires a write-scoped HF token. Resolution order:

1. Explicit `token=` arg.
2. `HF_WRITE_TOKEN` env var.
3. `HF_TOKEN` env var (only works if it happens to have write scope).

Returns the dataset URL.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
