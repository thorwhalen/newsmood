# newsmood

News-embedding trading-signal research substrate. Ingests financial news,
embeds it with OpenAI `text-embedding-3-small`, computes per-session
features against price targets, and runs walk-forward evaluations + backtests.

To install: `pip install newsmood`

## Published data

Companion dataset on Hugging Face — embeddings, canonical news DataFrame,
raw snapshots, and OHLCV all in one place:

**https://huggingface.co/datasets/thorwhalen/newsmood-data**

| File | Size | Download URL |
|---|---:|---|
| `embeddings.parquet` | ~1.0 GB | https://huggingface.co/datasets/thorwhalen/newsmood-data/resolve/main/embeddings.parquet |
| `news.parquet` | ~213 MB | https://huggingface.co/datasets/thorwhalen/newsmood-data/resolve/main/news.parquet |
| `raw_searches.tar.gz` | ~329 MB | https://huggingface.co/datasets/thorwhalen/newsmood-data/resolve/main/raw_searches.tar.gz |
| `ohlcv/<TICKER>.parquet` | ~20 KB each | https://huggingface.co/datasets/thorwhalen/newsmood-data/tree/main/ohlcv |

### Quick start (no local data required)

```python
from newsmood.data import cached_embedding_store, hf_news_dataframe

# Local-first, HF-fallback embedding cache.
# First miss downloads embeddings.parquet to ~/.cache/huggingface/hub/ and
# promotes hits to ~/.config/newsmood/embeddings/.
store = cached_embedding_store()
vec = store["a1b2c3d4e5f6abcd"]  # transparent fetch + local cache

# Canonical news DataFrame (one-shot read).
df = hf_news_dataframe()  # 359,948 rows
```

## Phase 1 findings

See [misc/docs/PHASE_1_FINDINGS.md](misc/docs/PHASE_1_FINDINGS.md) for the
full writeup of data, pipeline, models, results, and analysis of why the
first baseline did not produce a tradable edge.

## Modules

| Module | Purpose |
|---|---|
| `newsmood.ingest` | Canonical DataFrame from raw JSON snapshots |
| `newsmood.embed` | Cache-aware embedder built on `aix.batched_embeddings` |
| `newsmood.targets` | Cached yfinance OHLCV + forward returns + session alignment |
| `newsmood.features` | Per-session aggregation (mean-embedding, seed cosines, query counts) |
| `newsmood.tickers` | Article→ticker regex matcher; per-`(ticker, session)` features |
| `newsmood.clusters` | K-Means codebook (via `imbed.fit_kmeans`) + cluster-bucket counts |
| `newsmood.models` | Walk-forward CV with embargo, Ridge baseline, IC / Sharpe metrics |
| `newsmood.backtest` | Position sizing, Sharpe / Sortino / max-DD / turnover |
| `newsmood.select` | Variance / correlation filters, MI selection, alpha sweep |
| `newsmood.per_ticker_eval` | End-to-end per-ticker walk-forward + backtest |
| `newsmood.infer` | Train + persist + predict + write to mall for `hedger` consumption |
| `newsmood.data` | HF Dataset publishing + read-through cache |

## Token convention for publishing

For consuming the HF dataset, a read-only `HF_TOKEN` is sufficient (often
no token is needed at all for public datasets).

For *publishing* updates back to the dataset, the code follows a
least-privilege convention:

- `HF_TOKEN` — read-only, picked up automatically by `huggingface_hub` for
  downloads.
- `HF_WRITE_TOKEN` — elevated, used explicitly by
  `newsmood.data.upload_staging_to_hf` (and any future write helpers).

The resolution order for writes is: explicit `token=` kwarg →
`HF_WRITE_TOKEN` → `HF_TOKEN` (fallback).
