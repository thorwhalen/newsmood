# newsmood Phase 1 — Findings

**Status:** Pipeline complete, no demonstrated edge.
**Date:** 2026-05-16

This document captures the first end-to-end build of `newsmood`'s news-embedding
trading-signal pipeline and the empirical results of evaluating it on ~17 months
of financial news.

## 1. Goal

> Use semantic vector embeddings of financial news headlines to predict next-day
> equity returns, and assess whether the resulting signal has economic value.

The pipeline was designed as a **substrate** for further experiments (data
sources, feature recipes, model classes) rather than as a single bet. Most of
the engineering is reusable for any variation; the question being answered in
this phase is the simplest one: *does the most obvious recipe work?*

## 2. Data

### 2.1 News corpus

| Source | Format | Period | Articles |
|---|---|---|---|
| `newsdata` | JSON (list of articles per file, hourly snapshot) | 2025-01 → 2026-05 | ~360,000 |
| `yahoo_finance` | JSON (Yahoo `/search` payload per query) | 2025-02 → 2026-05 | ~27,000 |
| `yahoo_finance_headlines` | JSON (bare list of headline strings) | 2025-02 → 2026-05 | ~17,000 |

Raw files live under `/Users/thorwhalen/Dropbox/_odata/finance/mood/news/searches/`.
Total ≈ 52,000 JSON files spanning 16 months. The `newsdata` source's
`content` field is paywalled (returns `"ONLY AVAILABLE IN PAID PLANS"`), so the
usable text is **title + description** only — a typical headline carries
~30–40 tokens.

Each file is keyed by query category (`Earnings_Miss`, `Profit_Warning`,
`Acquisition`, `Bankruptcy`, `Tariffs`, etc.). The query name is preserved
as a feature for downstream models.

### 2.2 Canonicalization

`newsmood.ingest.ingest_searches` reads the three sources and produces a
single canonical DataFrame with columns:

```
doc_id | ts | source | query | title | description | text_to_embed | link |
source_id | category | country | language | key
```

- `doc_id` = first 16 hex chars of `sha1(canonical(text_to_embed))` —
  enables cross-source deduplication and stable cache keys.
- `text_to_embed` = `title + ". " + description` (description used when
  non-paywalled).
- `ts` is converted to tz-aware UTC datetime.
- Paywall placeholders and empty texts are dropped.

After dedup: **~404k unique articles** across all sources.

### 2.3 Embeddings

All embeddings produced via `aix.embeddings.batched_embeddings` →
`text-embedding-3-small` (OpenAI), 1536-dimensional vectors.

- **Coverage:** 359,962 vectors cached (the full `newsdata` source over
  2025-01-01 → 2026-06-01, plus a small calibration set from other dates).
- **Throughput:** ~40–70 vec/s sustained (rate-limited by OpenAI tier).
- **Total wall time:** ~90 minutes.
- **Total spend:** ~$0.20 USD (text-embedding-3-small is $0.02/1M tokens).
- **Local cache:** `~/.config/newsmood/embeddings/` — one pickle file per
  `doc_id`, ~5 KB each, ~1.8 GB total. Re-runs are cache-hit free.

### 2.4 Price data

`newsmood.targets.get_ohlcv` fetches daily OHLCV from `yfinance`, caches as
one parquet per ticker under `~/.config/newsmood/ohlcv/`.

Forward log returns at h ∈ {1, 5, 20} sessions are computed via
`forward_returns(prices, horizons=...)`.

### 2.5 Session alignment (leakage-critical)

A news timestamp `t` is mapped to its **anchor session** — the first market
close strictly after `t` (in US/Eastern). Rule:

- If `t.date()` is a trading day and `t.time() < 16:00 ET`, anchor = `t.date()`.
- Else, anchor = next trading day after `t.date()`.

Forward returns are computed strictly *after* the anchor close, so labels never
peek at price information that wasn't already in the past at the moment we
"observed" the news.

## 3. Models

### 3.1 Pipeline overview

```
Raw JSON           → ingest (canonical DataFrame)
                     ↓
Text               → text-embedding-3-small (1536-d)
                     ↓
Cluster codebook   → K-Means (k=32) on cosine-normalized vectors
                     ↓
Per-session        → mean-embedding, seed cosines (13 event prompts),
features             query counts, cluster-bucket counts
                     ↓
Targets            → SPY close-to-close 1d return (anchor-aligned)
                     ↓
Model              → Ridge(α) over standardized features
                     ↓
Evaluation         → walk-forward CV (5 folds, embargo=5 sessions)
                     ↓
Backtest           → tanh sizing, 1bp transaction cost
```

### 3.2 Feature set (90 columns)

| Group | Count | Description |
|---|---:|---|
| `news_count` | 1 | Articles anchored to the session |
| `q_<topic>` | ~45 | Article count per query category |
| `cos_<prompt>` | 13 | Cosine similarity of session-mean embedding to each of 13 hand-crafted "event seed" prompts (earnings_miss, profit_warning, tariffs, rate_hike, recession, …) |
| `clust_<i>` | 32 | Article count per K-Means cluster |

### 3.3 Modeling choices

- **Ridge regression** with `StandardScaler` over all features.
- **Walk-forward CV**, 5 expanding-window folds, min_train=30 sessions,
  embargo=5 sessions between train and test (≥ the forward horizon, so the
  forward-return overlap can't leak across the boundary).
- **Alpha sweep**: {0.1, 1, 10, 100, 1000}.
- **Per-fold MI feature selection** (top-K mutual information) — tested as
  a leakage-free alternative to global feature selection; in practice it did
  not improve OOS performance.

### 3.4 Backtest convention

For each session `t`:
- `position_t = tanh(y_pred_t / 0.01)` — a +1% predicted return → ≈ 0.76
  position; capped at ±1 NAV.
- `pnl_t = position_t * y_true_t − 1bp * |position_t − position_{t-1}|`.
- Equity curve = cumulative product of `1 + pnl`.

Metrics reported: annualized Sharpe, Sortino, max drawdown, win rate, hit
rate (sign match), turnover, final equity, total cost.

## 4. Results

### 4.1 SPY single-target — alpha sweep, 90 features

| α | mean IC | mean rank IC | sign acc | mean Sharpe |
|---:|---:|---:|---:|---:|
| 0.1 | -0.087 | -0.092 | 0.500 | +0.46 |
| 1.0 | **-0.116** | -0.113 | 0.464 | -0.57 |
| 10.0 | -0.134 | -0.110 | 0.495 | +0.51 |
| 100.0 | -0.109 | -0.068 | 0.495 | +0.05 |
| 1000.0 | -0.069 | -0.032 | 0.532 | +1.26 |

ICs are consistently negative. Heavy regularization pulls IC toward zero but
doesn't recover any signal.

### 4.2 SPY single-target — per-fold MI top-20 (leakage-free)

| α | mean IC | mean rank IC | sign acc |
|---:|---:|---:|---:|
| 0.1 | -0.095 | -0.062 | 0.499 |
| 1.0 | -0.101 | -0.073 | 0.508 |
| 10.0 | -0.100 | -0.088 | 0.526 |
| 100.0 | -0.085 | -0.090 | 0.504 |

Per-fold feature selection does **not** rescue the signal — confirming the
negative result is robust to feature-count overfitting concerns.

### 4.3 SPY OOS backtest (α=100, tanh, 1bp)

| Metric | Value |
|---|---:|
| n periods | 222 |
| ann. return | -1.24 % |
| ann. vol | 2.44 % |
| **Sharpe** | **-0.51** |
| Sortino | -0.68 |
| max drawdown | -4.33 % |
| win rate | 47.3 % |
| hit rate | 50.4 % |
| turnover | 19.1 % |
| final equity | 0.989 |

### 4.4 Per-ticker walk-forward (13 names, 4 folds, α=10, 1bp)

| Ticker | n sessions | mean IC | mean rank IC | sign acc | Sharpe | max DD | final equity |
|---|---:|---:|---:|---:|---:|---:|---:|
| **NVDA** | 281 | **+0.069** | +0.058 | 0.552 | **+1.67** | -14.3 % | **1.49** |
| COIN | 244 | +0.035 | -0.014 | 0.453 | +0.15 | -41.0 % | 0.92 |
| GOOGL | 279 | +0.031 | +0.093 | 0.534 | +1.39 | -16.4 % | 1.33 |
| META | 279 | +0.023 | +0.083 | 0.583 | +0.90 | -15.6 % | 1.20 |
| NFLX | 232 | +0.005 | -0.077 | 0.445 | -1.40 | -24.4 % | 0.78 |
| MSTR | 198 | -0.016 | +0.028 | 0.543 | +0.97 | -29.9 % | 1.30 |
| BAC | 260 | -0.024 | +0.004 | 0.531 | -0.11 | -8.3 % | 0.98 |
| JPM | 280 | -0.026 | -0.028 | 0.514 | -0.07 | -10.9 % | 0.98 |
| TSLA | 280 | -0.027 | -0.006 | 0.492 | -1.12 | -44.3 % | 0.59 |
| MSFT | 280 | -0.030 | +0.074 | 0.520 | +0.48 | -11.5 % | 1.06 |
| AMD | 251 | -0.058 | -0.088 | 0.466 | -0.35 | -60.2 % | 0.73 |
| AMZN | 281 | -0.062 | -0.056 | 0.476 | -0.98 | -25.1 % | 0.79 |
| **AAPL** | 280 | **-0.066** | -0.078 | 0.468 | -0.97 | -22.1 % | **0.84** |

**Aggregate:**
- 5/13 positive mean IC; 6/13 positive Sharpe.
- Mean IC across tickers: **-0.011**.
- **Equal-weight final equity: 1.0000** — winners exactly cancel losers.

## 5. Analysis — why not better?

### 5.1 The aggregate signal is genuinely absent

The mean IC across 13 tickers (-0.011) and the equal-weight final equity
(1.0000) are about as clean a "no signal" result as one gets. This is not a
statistical-power problem alone: it's a recipe problem.

### 5.2 The per-ticker spread is mostly noise

5/13 positive (38 %) is only marginally different from coin-flip. With 13
tests at a notional 5 % significance, **we'd expect ~0.65 false positives**
by chance. The strongest result (NVDA Sharpe +1.67) over ~280 sessions could
plausibly be the lucky draw. Without a held-out validation window or a
formal multiple-testing correction, **none of the individual ticker results
should be treated as tradable signals.**

### 5.3 Where the recipe is weak

Several plausible culprits, in roughly decreasing-importance order:

1. **Text quality.** `newsdata` gives titles + descriptions only — typically
   30–40 tokens. The semantic content of an event ("Apple Q3 EPS missed by
   $0.05") is usually buried in the article body, which we don't have.
2. **Timestamp granularity.** All targets are end-of-day close-to-close.
   Most published news-alpha literature finds the bulk of price impact
   within minutes-to-hours of the news, not in the next-day close.
3. **Generic feed.** newsdata is a general macro/world-news feed. The
   density of *equity-actionable* news is low; most articles are political,
   social, or about non-tradable entities.
4. **Bag-of-features model.** Ridge over `(news_count, cosines,
   query_counts, cluster_counts)` is a flat-features linear model. It can't
   represent "earnings miss → 5-day drift" or "tariff news on tariff-exposed
   names → directional move." Event detection + name matching is missing
   from the model.
5. **No event filtering.** Trading every session whether or not anything
   material happened means our signal-to-noise is dominated by the (very
   common) zero-information sessions.
6. **Aggregating to SPY.** A piece of news about Apple isn't directly about
   the S&P 500. The SPY model averages firm-specific signal across hundreds
   of articles, much of which is irrelevant to the index.

### 5.4 What the experiment *does* rule out

- The most-obvious recipe (generic feed × session-mean embedding × Ridge ×
  next-day SPY return) does not produce a tradable edge after even
  minimal (1 bp) transaction costs.

### 5.5 What it does *not* rule out

- Better text → maybe better signal (SEC filings, earnings transcripts,
  FOMC text, curated business-news feeds).
- Shorter horizons (intraday, around event timestamps).
- Event-class filtering + per-event-class models.
- Sparser trading: only trade when news volume / cluster-bucket activity
  exceeds a threshold.
- Cross-sectional ranking within a clean universe (e.g., long top-decile
  predicted return, short bottom-decile, across S&P 100 names with > N
  daily article mentions).

## 6. Engineering deliverables

The 17-month build produced reusable infrastructure across five packages:

| Package | What was added | Tests |
|---|---|---:|
| `newsmood` | The whole pipeline: `ingest`, `embed`, `targets`, `features`, `models`, `tickers`, `clusters`, `infer`, `backtest`, `select`, `per_ticker_eval` | 167 |
| `vd` | `TimeIndexedCollection` — backend-agnostic time-windowed wrapper over any `Collection` | 64 |
| `aix` | `batched_embeddings`, `cached_embeddings`, `truncate_segment`, `iter_batches`, `text_cache_key` | 35 |
| `imbed` | `fit_kmeans` — returns a model with `.predict()` for new vectors, missing from the existing one-shot `kmeans_clusterer` | (covered indirectly via `newsmood/clusters` tests) |
| `hedger` | `strategies/news_embed.py` — thin strategy plugin that reads scores from `mall["features:news_embed_v1"]` | 13 |

**Total:** 279 passing tests across the four packages with directly-added tests.

## 7. Recommended next experiments

In order of (estimated marginal value) / (estimated effort):

1. **Cross-sectional long/short on high-news-volume names.** Use the existing
   `per_ticker_evaluate` outputs to rank tickers; long the top quintile,
   short the bottom; backtest as a portfolio. ~1 day, all infrastructure
   exists.
2. **Event-class filtering.** Only emit signals on sessions where one of the
   `cos_*` seed cosines exceeds a calibration-window threshold. Inverts the
   "trade every day" assumption and may sharpen IC dramatically. ~1 day.
3. **Intraday horizons.** Backfill 1-minute or 5-minute OHLCV for the top
   ticker universe; recompute targets as `close(news_ts + 1h) /
   close(news_ts)`. Substantial data work (~1 week) but is where most of
   the news-alpha literature finds edge.
4. **Better text.** Add `SEC EDGAR 8-K filings` as a second source (free,
   public, far more equity-relevant than generic newswire). Reuse the same
   embedding + features pipeline. ~2 days.
5. **Event-class model.** Train a small classifier per seed prompt
   (`is_earnings_miss`, `is_acquisition`, …) and use predicted-probability
   spikes as the trading signal. ~3 days.
6. **Per-fold cross-validated alpha + feature selection** baked into a
   `tune_walk_forward` helper. ~1 day; will surface whether the per-ticker
   wins generalize at all.

## 8. Reproducibility notes

- Embeddings cache: `~/.config/newsmood/embeddings/` (360k pickle files,
  ~1.4 GB). Recreating from scratch costs ~$0.20 + ~90 minutes wall.
- OHLCV cache: `~/.config/newsmood/ohlcv/` (one parquet per ticker).
- Raw news: `~/Dropbox/_odata/finance/mood/news/searches/`.
- Mall: `~/.config/newsmood/mall/` (per-(symbol, session) prediction store).
- Eval scripts: `misc/full_evaluation.py` and
  `misc/evaluate_real_embeddings.py`.
- Test suites pass on Python 3.12, with `pandas`, `numpy`, `scikit-learn`,
  `pyarrow`, `yfinance`, `litellm`, `dol`, `vd`, `imbed`, `aix`.

### Published artifacts (Hugging Face Dataset)

All four artifacts are mirrored at
**[`thorwhalen/newsmood-data`](https://huggingface.co/datasets/thorwhalen/newsmood-data)**
so any fresh checkout can hydrate without re-running ingest or paying for
embeddings:

| File | Size | URL |
|---|---:|---|
| `embeddings.parquet` | ~1.0 GB | https://huggingface.co/datasets/thorwhalen/newsmood-data/resolve/main/embeddings.parquet |
| `news.parquet` | ~213 MB | https://huggingface.co/datasets/thorwhalen/newsmood-data/resolve/main/news.parquet |
| `raw_searches.tar.gz` | ~329 MB | https://huggingface.co/datasets/thorwhalen/newsmood-data/resolve/main/raw_searches.tar.gz |
| `ohlcv/*.parquet` | ~280 KB total | https://huggingface.co/datasets/thorwhalen/newsmood-data/tree/main/ohlcv |

Programmatic access:

```python
from newsmood.data import cached_embedding_store, hf_news_dataframe
store = cached_embedding_store()   # local first, HF on miss; promotes hits to local
df = hf_news_dataframe()           # canonical news, one-shot cached download
```

Or directly via `huggingface_hub.hf_hub_download` — see
[`newsmood/data.py`](../../newsmood/data.py) for the low-level helpers.
