"""Pack local artifacts for publishing and download-on-miss from HF.

Two halves:

**Publishing (export side):** repackages the per-vector pickle cache, the
raw JSON snapshot tree, and the ingested canonical DataFrame into single
parquet files suitable for upload to a Hugging Face Dataset.

**Consuming (import side):** a thin read-through wrapper that downloads a
parquet shard from HF on first access, caches it under
``~/.cache/huggingface/hub/``, and exposes a `MutableMapping` view so
existing newsmood code paths (which expect a doc_id → vector mapping)
work without modification.

The default published dataset is ``thorwhalen/newsmood-data`` on the
Hugging Face Hub. The files served from there:

- ``embeddings.parquet``: doc_id (str), vector (list[float32 × 1536]).
  ~1.5 GB compressed for 360k articles.
- ``news.parquet``: full canonical ingest output (CANONICAL_COLUMNS).
  ~50 MB.
- ``raw/searches.tar.gz``: optional original JSON snapshots, gzipped.
  ~500 MB.
- ``ohlcv/<TICKER>.parquet``: per-ticker price files (small).

All inputs default to the local conventions
(``~/.config/newsmood/embeddings/`` etc.) so the export functions are
idempotent and re-runnable.
"""

from __future__ import annotations

import os
import pathlib
import pickle
import tarfile
from collections.abc import Iterator, Mapping, MutableMapping
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_HF_REPO_ID = os.environ.get("NEWSMOOD_HF_REPO", "thorwhalen/newsmood-data")
DEFAULT_STAGING_DIR = os.environ.get(
    "NEWSMOOD_STAGING_DIR",
    str(pathlib.Path("~/.cache/newsmood/staging").expanduser()),
)


def _hf_write_token() -> Optional[str]:
    """Return the token to use for HF *write* operations.

    Convention: ``HF_TOKEN`` is read-only (least-privilege default that
    ``huggingface_hub`` will pick up automatically); ``HF_WRITE_TOKEN`` is
    the elevated token used for ``create_repo`` / ``upload_*``. Falls back
    to ``HF_TOKEN`` if ``HF_WRITE_TOKEN`` isn't set.
    """
    return os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")


# ---------------------------------------------------------------------------
# Export side
# ---------------------------------------------------------------------------


def pack_embeddings_to_parquet(
    source: Optional[Mapping[str, list[float]]] = None,
    *,
    out_path: str,
    dtype: str = "float32",
    chunk_size: int = 25_000,
    verbose: bool = True,
) -> str:
    """Pack a doc_id → vector mapping into a single parquet file.

    Parameters
    ----------
    source
        Any mapping from doc_id (str) to vector (list[float]). Defaults to
        :func:`newsmood.embed.default_embedding_cache`.
    out_path
        Destination parquet path.
    dtype
        Numeric dtype for vectors. ``"float32"`` halves the size vs
        ``float64`` at negligible cosine-similarity drift.
    chunk_size
        Rows per write batch (memory bound). Default 25k.

    Returns the absolute path written.
    """
    if source is None:
        from newsmood.embed import default_embedding_cache

        source = default_embedding_cache()

    import pyarrow as pa
    import pyarrow.parquet as pq

    out = pathlib.Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    writer: Optional[pq.ParquetWriter] = None
    n_written = 0
    keys_iter = iter(source)
    while True:
        chunk_ids: list[str] = []
        chunk_vecs: list[list[float]] = []
        for _ in range(chunk_size):
            try:
                k = next(keys_iter)
            except StopIteration:
                break
            try:
                v = source[k]
            except (KeyError, EOFError, pickle.UnpicklingError):
                continue
            chunk_ids.append(k)
            chunk_vecs.append([float(x) for x in v])
        if not chunk_ids:
            break
        arr = np.asarray(chunk_vecs, dtype=dtype)
        table = pa.table(
            {
                "doc_id": pa.array(chunk_ids, type=pa.string()),
                "vector": pa.FixedSizeListArray.from_arrays(
                    pa.array(arr.ravel(), type=pa.float32() if dtype == "float32" else pa.float64()),
                    arr.shape[1],
                ),
            }
        )
        if writer is None:
            writer = pq.ParquetWriter(out, table.schema, compression="zstd")
        writer.write_table(table)
        n_written += len(chunk_ids)
        if verbose:
            print(f"  packed {n_written:,} embeddings", flush=True)
    if writer is not None:
        writer.close()
    if verbose:
        size_mb = out.stat().st_size / 1e6
        print(f"wrote {out} ({n_written:,} rows, {size_mb:.1f} MB)", flush=True)
    return str(out)


def pack_news_to_parquet(
    *,
    out_path: str,
    source: Optional[str] = None,
    searches_root: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """Run the full ingest and write the canonical DataFrame as parquet.

    See :func:`newsmood.ingest.ingest_searches` for parameters.
    """
    from newsmood.ingest import DEFAULT_SEARCHES_ROOT, ingest_searches
    import dol

    root = searches_root or DEFAULT_SEARCHES_ROOT
    store = dol.JsonFiles(root)
    if verbose:
        print(f"ingesting from {root}...", flush=True)
    df = ingest_searches(store, source=source, since=since, until=until)
    out = pathlib.Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression="zstd")
    if verbose:
        size_mb = out.stat().st_size / 1e6
        print(f"wrote {out} ({len(df):,} rows, {size_mb:.1f} MB)", flush=True)
    return str(out)


def pack_raw_searches_to_tarball(
    *,
    searches_root: Optional[str] = None,
    out_path: str,
    compression: str = "gz",
    verbose: bool = True,
) -> str:
    """Tar up the raw JSON snapshot tree (for full reproducibility)."""
    from newsmood.ingest import DEFAULT_SEARCHES_ROOT

    root = pathlib.Path(searches_root or DEFAULT_SEARCHES_ROOT).expanduser()
    out = pathlib.Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    mode = f"w:{compression}" if compression else "w"
    if verbose:
        print(f"tarring {root} → {out} (compression={compression})...", flush=True)
    with tarfile.open(out, mode) as tf:
        tf.add(root, arcname="searches")
    if verbose:
        size_mb = out.stat().st_size / 1e6
        print(f"wrote {out} ({size_mb:.1f} MB)", flush=True)
    return str(out)


def pack_ohlcv_to_dir(
    *,
    ohlcv_root: Optional[str] = None,
    out_dir: str,
    verbose: bool = True,
) -> str:
    """Copy existing per-ticker OHLCV parquet files to a staging directory."""
    import shutil

    from newsmood.targets import DEFAULT_OHLCV_CACHE

    src = pathlib.Path(ohlcv_root or DEFAULT_OHLCV_CACHE).expanduser()
    dst = pathlib.Path(out_dir).expanduser()
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    total_bytes = 0
    for p in src.glob("*.parquet"):
        target = dst / p.name
        shutil.copy2(p, target)
        n += 1
        total_bytes += target.stat().st_size
    if verbose:
        print(f"copied {n} ohlcv files ({total_bytes/1e6:.2f} MB) → {dst}", flush=True)
    return str(dst)


# ---------------------------------------------------------------------------
# Import side: download-on-miss from HF
# ---------------------------------------------------------------------------


def download_artifact(
    filename: str,
    *,
    repo_id: str = DEFAULT_HF_REPO_ID,
    repo_type: str = "dataset",
    revision: Optional[str] = None,
) -> str:
    """Download an HF dataset artifact (cached). Returns the local path.

    Wraps :func:`huggingface_hub.hf_hub_download`; subsequent calls hit the
    HF cache under ``~/.cache/huggingface/hub/``.
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
    )


class EmbeddingsParquetView(Mapping):
    """Read-only ``MutableMapping``-style view over an embeddings parquet.

    Loads the entire (doc_id, vector) table into a dict on construction
    (~1.5 GB RAM for 360k × 1536 float32 vectors). For lower-memory access,
    use the underlying parquet directly.

    >>> view = EmbeddingsParquetView('embeddings.parquet')  # doctest: +SKIP
    >>> vec = view['<some_doc_id>']                          # doctest: +SKIP
    """

    def __init__(self, path: str):
        self._path = path
        self._lookup: dict[str, np.ndarray] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import pyarrow.parquet as pq

        table = pq.read_table(self._path)
        ids = table.column("doc_id").to_pylist()
        vecs = table.column("vector").to_numpy(zero_copy_only=False)
        # vecs is an object array of lists for FixedSizeList; coerce
        self._lookup = {i: np.asarray(v, dtype=np.float32) for i, v in zip(ids, vecs)}
        self._loaded = True

    def __getitem__(self, key: str) -> list[float]:
        self._ensure_loaded()
        return list(self._lookup[key])

    def __iter__(self) -> Iterator[str]:
        self._ensure_loaded()
        return iter(self._lookup)

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._lookup)

    def __contains__(self, key) -> bool:
        self._ensure_loaded()
        return key in self._lookup


class CachedEmbeddingStore(MutableMapping):
    """Local-first, HF-backed embedding cache with write-through semantics.

    The use case: scripts that already expect a ``MutableMapping[str, list[float]]``
    (e.g. ``newsmood.embed.embed_doc_ids``) can transparently consume the
    published HF dataset *without re-embedding*. The fall-through order is:

    1. **Local cache** (``~/.config/newsmood/embeddings/``). If the key is
       already there, return it.
    2. **HF parquet view** (lazy-downloaded on first miss). If the key is
       there, copy it to the local cache and return.
    3. **Caller's embedder** (via ``newsmood.embed.embed_doc_ids``), if a
       genuinely new doc_id is requested. Cached locally.

    The local store remains the SSOT for newly-computed vectors; the HF view
    is read-only.

    Writes (``store[doc_id] = vector``) always go to the local cache so any
    fresh embeddings persist there for the next run.
    """

    def __init__(
        self,
        local: Optional[MutableMapping[str, list[float]]] = None,
        remote: Optional[Mapping[str, list[float]]] = None,
    ):
        if local is None:
            from newsmood.embed import default_embedding_cache

            local = default_embedding_cache()
        self._local = local
        self._remote = remote  # Lazily fetched on first miss if None.

    def _ensure_remote(self) -> Mapping[str, list[float]]:
        if self._remote is None:
            self._remote = hf_embedding_cache()
        return self._remote

    def __getitem__(self, key: str) -> list[float]:
        try:
            return self._local[key]
        except KeyError:
            pass
        remote = self._ensure_remote()
        vec = remote[key]  # raises KeyError if absent
        # Promote to local so subsequent reads hit the fast path.
        v = list(vec)
        try:
            self._local[key] = v
        except Exception:
            pass  # Local cache write failures shouldn't break reads.
        return v

    def __setitem__(self, key: str, value: list[float]) -> None:
        self._local[key] = list(value)

    def __delitem__(self, key: str) -> None:
        del self._local[key]

    def __iter__(self) -> Iterator[str]:
        seen: set[str] = set()
        for k in self._local:
            seen.add(k)
            yield k
        # Iterating remote is expensive (downloads + reads 1 GB parquet); only do
        # it if iteration is actually consumed beyond local — keep order: local,
        # then remote-only.
        if self._remote is not None:
            for k in self._remote:
                if k not in seen:
                    yield k

    def __len__(self) -> int:
        if self._remote is None:
            return len(self._local)
        return len(set(self._local).union(self._remote))

    def __contains__(self, key) -> bool:
        if key in self._local:
            return True
        if self._remote is not None:
            return key in self._remote
        # Don't trigger remote download just to check membership.
        return False


def cached_embedding_store(
    *,
    local: Optional[MutableMapping[str, list[float]]] = None,
    eager_remote: bool = False,
) -> CachedEmbeddingStore:
    """Build a :class:`CachedEmbeddingStore` over the default local cache.

    Parameters
    ----------
    local
        Pre-built local cache; defaults to
        :func:`newsmood.embed.default_embedding_cache`.
    eager_remote
        If True, fetch the HF parquet view immediately (one ~1 GB download
        on first call to a fresh machine). If False (default), defer until a
        miss occurs.
    """
    remote = hf_embedding_cache() if eager_remote else None
    return CachedEmbeddingStore(local=local, remote=remote)


def hf_embedding_cache(
    *,
    repo_id: str = DEFAULT_HF_REPO_ID,
    filename: str = "embeddings.parquet",
    revision: Optional[str] = None,
) -> EmbeddingsParquetView:
    """Return an :class:`EmbeddingsParquetView` backed by an HF parquet.

    Downloads the parquet to the local HF cache on first call; subsequent
    calls reuse the cached file.
    """
    path = download_artifact(filename, repo_id=repo_id, revision=revision)
    return EmbeddingsParquetView(path)


def hf_news_dataframe(
    *,
    repo_id: str = DEFAULT_HF_REPO_ID,
    filename: str = "news.parquet",
    revision: Optional[str] = None,
) -> pd.DataFrame:
    """Load the canonical news DataFrame from an HF dataset."""
    path = download_artifact(filename, repo_id=repo_id, revision=revision)
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# One-shot full publish
# ---------------------------------------------------------------------------


def pack_all_to_staging(
    *,
    staging_dir: str = DEFAULT_STAGING_DIR,
    include_raw: bool = True,
    include_ohlcv: bool = True,
    verbose: bool = True,
) -> dict[str, str]:
    """Build every artifact under one staging directory and return paths.

    Caller then uploads ``staging_dir`` to the HF Hub via
    :func:`upload_staging_to_hf`.
    """
    staging = pathlib.Path(staging_dir).expanduser()
    staging.mkdir(parents=True, exist_ok=True)

    out: dict[str, str] = {}
    out["embeddings"] = pack_embeddings_to_parquet(
        out_path=str(staging / "embeddings.parquet"), verbose=verbose
    )
    out["news"] = pack_news_to_parquet(
        out_path=str(staging / "news.parquet"), verbose=verbose
    )
    if include_raw:
        out["raw"] = pack_raw_searches_to_tarball(
            out_path=str(staging / "raw_searches.tar.gz"), verbose=verbose
        )
    if include_ohlcv:
        out["ohlcv"] = pack_ohlcv_to_dir(
            out_dir=str(staging / "ohlcv"), verbose=verbose
        )
    return out


def upload_staging_to_hf(
    *,
    staging_dir: str = DEFAULT_STAGING_DIR,
    repo_id: str = DEFAULT_HF_REPO_ID,
    commit_message: str = "Update artifacts",
    private: bool = False,
    token: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """Push a staging directory to an HF Dataset.

    Requires a write-scoped HF token. Resolution order:
    1. Explicit ``token=`` arg.
    2. ``HF_WRITE_TOKEN`` env var.
    3. ``HF_TOKEN`` env var (only works if it happens to have write scope).

    Returns the dataset URL.
    """
    from huggingface_hub import HfApi, create_repo

    tok = token or _hf_write_token()
    if tok is None:
        raise RuntimeError(
            "No HF token available. Set HF_WRITE_TOKEN (preferred) or HF_TOKEN."
        )
    api = HfApi(token=tok)
    info = create_repo(
        repo_id, repo_type="dataset", exist_ok=True, private=private, token=tok
    )
    if verbose:
        print(f"Repo ready: {info.url}", flush=True)
    staging = pathlib.Path(staging_dir).expanduser()
    api.upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        token=tok,
    )
    url = f"https://huggingface.co/datasets/{repo_id}"
    if verbose:
        print(f"Browse: {url}", flush=True)
    return url


__all__ = [
    "DEFAULT_HF_REPO_ID",
    "DEFAULT_STAGING_DIR",
    "CachedEmbeddingStore",
    "EmbeddingsParquetView",
    "cached_embedding_store",
    "download_artifact",
    "hf_embedding_cache",
    "hf_news_dataframe",
    "pack_all_to_staging",
    "pack_embeddings_to_parquet",
    "pack_news_to_parquet",
    "pack_ohlcv_to_dir",
    "pack_raw_searches_to_tarball",
    "upload_staging_to_hf",
]
