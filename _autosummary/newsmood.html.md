# newsmood

newsmood — news-embedding trading signals.

The pipeline is staged across modules:

- [`newsmood.ingest`](newsmood.ingest.html.md#module-newsmood.ingest) — read raw news from the local searches store and
  return a canonical pandas DataFrame.
- [`newsmood.embed`](newsmood.embed.html.md#module-newsmood.embed) (planned) — embed text into a
  `vd.TimeIndexedCollection` with content-hash caching.
- [`newsmood.targets`](newsmood.targets.html.md#module-newsmood.targets) (planned) — OHLCV + forward returns + session
  alignment (strict, leakage-safe).
- [`newsmood.features`](newsmood.features.html.md#module-newsmood.features) (planned) — per-(target, day) features:
  window mean embeddings, seed-prompt cosines, news volume.
- [`newsmood.models`](newsmood.models.html.md#module-newsmood.models) (planned) — walk-forward baselines (Ridge, GBM)
  with IC/rank-IC reporting.

### Functions

| [`ingest_searches`](#newsmood.ingest_searches)(store, \*[, source, since, ...])   | Read a searches store and return a canonical `DataFrame`.                                                                                                                                     |
|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`iter_canonical_rows`](#newsmood.iter_canonical_rows)(store, \*[, source, ...])      | Iterate canonical rows from a `dol.JsonFiles`-style store.                                                                                                                                    |
| [`make_doc_id`](#newsmood.make_doc_id)(text, \*[, hash_len])                  | SHA-1-based, lowercase-normalized content hash.                                                                                                                                               |
| [`make_text_to_embed`](#newsmood.make_text_to_embed)(title, description)             | Build the canonical text used for embedding.                                                                                                                                                  |
| [`parse_key`](#newsmood.parse_key)(key)                                     | Extract `date`, `file_ts`, and `query` from a store key.                                                                                                                                      |
| [`default_embedding_cache`](#newsmood.default_embedding_cache)([root])                    | Return a pickle-coded dol Files store, creating the root if missing.                                                                                                                          |
| [`deterministic_dummy_embedder`](#newsmood.deterministic_dummy_embedder)(\*[, dim])            | Return a deterministic toy embedder for tests / cost-free dry runs.                                                                                                                           |
| [`embed_dataframe`](#newsmood.embed_dataframe)(df, \*, cache[, model, ...])       | Add a `vector` column to `df` by embedding `text_col` via cache.                                                                                                                              |
| [`embed_doc_ids`](#newsmood.embed_doc_ids)(pairs, \*, cache[, model, ...])      | Embed `(doc_id, text)` pairs, hitting the cache by `doc_id`.                                                                                                                                  |
| [`make_news_collection`](#newsmood.make_news_collection)([name, backend, ...])         | Create (or reopen) a `vd.TimeIndexedCollection` for news.                                                                                                                                     |
| [`news_pipeline`](#newsmood.news_pipeline)(df, \*[, cache, ...])                | Single-call: embed a DataFrame and load it into a vd collection.                                                                                                                              |
| [`populate_collection`](#newsmood.populate_collection)(df, \*, collection[, ...])     | Write `df` rows into a `TimeIndexedCollection`.                                                                                                                                               |
| [`align_news_series_to_sessions`](#newsmood.align_news_series_to_sessions)(news_ts, ...)        | Vectorized variant of [`align_news_to_session()`](#newsmood.align_news_to_session) over a Series.                                                                                 |
| [`align_news_to_session`](#newsmood.align_news_to_session)(news_ts, sessions, \*)       | Return the anchor-session date for a single news timestamp.                                                                                                                                   |
| [`attach_targets`](#newsmood.attach_targets)(news_df, \*[, tickers, ...])        | Attach forward-return columns to `news_df` via session alignment.                                                                                                                             |
| [`forward_returns`](#newsmood.forward_returns)(prices, \*[, horizons, price_col]) | Compute forward log returns over the given session-count horizons.                                                                                                                            |
| [`get_ohlcv`](#newsmood.get_ohlcv)(ticker, \*[, start, end, ...])           | Return a daily OHLCV DataFrame for `ticker`, with a parquet cache.                                                                                                                            |
| [`build_seed_embeddings`](#newsmood.build_seed_embeddings)(seeds, \*, cache[, ...])     | Embed each seed prompt (via the shared cache) — returns `{name: vector}`.                                                                                                                     |
| [`panel_with_targets`](#newsmood.panel_with_targets)(news_df, \*[, seed_vecs, ...])  | Join per-session features with per-session targets.                                                                                                                                           |
| [`session_features`](#newsmood.session_features)(news_df, \*[, seed_vecs, ...])    | Group `news_df` by session, return one row per session.                                                                                                                                       |
| [`evaluate_walk_forward`](#newsmood.evaluate_walk_forward)(panel, \*, target[, ...])    | Walk-forward Ridge baseline over a session-level panel.                                                                                                                                       |
| [`fit_predict_ridge`](#newsmood.fit_predict_ridge)(X_train, y_train, X_test, \*)    | Standard-scale + Ridge fit, return test predictions.                                                                                                                                          |
| [`information_coefficient`](#newsmood.information_coefficient)(y_true, y_pred)            | Pearson IC.                                                                                                                                                                                   |
| [`long_short_sharpe`](#newsmood.long_short_sharpe)(y_true, y_pred, \*[, cost_bps])  | Sharpe of a sign-based long-short portfolio.                                                                                                                                                  |
| [`rank_information_coefficient`](#newsmood.rank_information_coefficient)(y_true, y_pred)       | Spearman IC.                                                                                                                                                                                  |
| [`sign_accuracy`](#newsmood.sign_accuracy)(y_true, y_pred)                      | Fraction of predictions whose sign matches the true sign.                                                                                                                                     |
| [`walk_forward_splits`](#newsmood.walk_forward_splits)(n, \*[, n_splits, ...])        | Yield `(train_idx, test_idx)` pairs for expanding-window CV.                                                                                                                                  |
| [`attach_tickers`](#newsmood.attach_tickers)(df, \*[, universe, text_col, ...])  | Return a copy of `df` with an added `tickers` column (list[str]).                                                                                                                             |
| [`explode_to_ticker_rows`](#newsmood.explode_to_ticker_rows)(df, \*[, ticker_col])       | Explode a `tickers` list column to one row per (article, ticker).                                                                                                                             |
| [`match_tickers`](#newsmood.match_tickers)(text, \*, universe)                  | One-shot helper; prefer [`TickerMatcher`](#newsmood.TickerMatcher) for many calls.                                                                                        |
| [`per_ticker_session_features`](#newsmood.per_ticker_session_features)(article_panel, \*)     | Session-by-ticker features.                                                                                                                                                                   |
| [`attach_cluster_id`](#newsmood.attach_cluster_id)(df, model, \*[, vec_col, ...])   | Return a copy of `df` with cluster IDs in `out_col`.                                                                                                                                          |
| [`fit_news_clusters`](#newsmood.fit_news_clusters)(df, \*[, n_clusters, ...])       | Fit a K-Means codebook on `df[vec_col]`.                                                                                                                                                      |
| [`session_cluster_counts`](#newsmood.session_cluster_counts)(df, \*, n_clusters[, ...])  | Per-session counts of articles in each cluster (zero-padded across all clusters).                                                                                                             |
| [`load_model`](#newsmood.load_model)(model_path)                             | Load an InferenceModel produced by [`save_model()`](#newsmood.save_model).                                                                                             |
| [`open_mall`](#newsmood.open_mall)([root])                                  | Open a persistent mall — a dol-backed JSON store at `root`.                                                                                                                                   |
| [`predict_scores`](#newsmood.predict_scores)(model, feature_df)                  | Apply model to feature_df; return Series indexed like feature_df.                                                                                                                             |
| [`run_inference`](#newsmood.run_inference)(feature_df, \*, model, mall[, ...])  | End-to-end: predict on `feature_df` and write to `mall`.                                                                                                                                      |
| [`save_model`](#newsmood.save_model)(model, model_path)                      | Persist an InferenceModel to disk (pickle).                                                                                                                                                   |
| [`train_and_save`](#newsmood.train_and_save)(panel, \*, target[, ...])           | Fit on the full panel (no CV; CV is for evaluation) and persist.                                                                                                                              |
| [`write_predictions_to_mall`](#newsmood.write_predictions_to_mall)(predictions, \*, mall)   | Write per-(ticker, session) scores into `mall[feature_key]`.                                                                                                                                  |
| [`annualized_sharpe`](#newsmood.annualized_sharpe)(pnl, \*[, periods_per_year])     | Sharpe ratio = mean / std × √(periods_per_year).                                                                                                                                              |
| [`annualized_sortino`](#newsmood.annualized_sortino)(pnl, \*[, periods_per_year])    | Sortino = mean / downside-std × √(periods_per_year).                                                                                                                                          |
| [`backtest`](#newsmood.backtest)(y_true, y_pred, \*[, sizing, ...])        | Run a single-asset, single-signal backtest.                                                                                                                                                   |
| [`backtest_walk_forward`](#newsmood.backtest_walk_forward)(...[, y_true_col, ...])      | Backtest from the long-form predictions DataFrame produced by [`newsmood.models.evaluate_walk_forward()`](newsmood.models.html.md#newsmood.models.evaluate_walk_forward). |
| [`hit_rate_sign`](#newsmood.hit_rate_sign)(y_true, y_pred)                      | Fraction of rows where `sign(y_pred) == sign(y_true)` (zeros skipped).                                                                                                                        |
| [`max_drawdown`](#newsmood.max_drawdown)(equity)                               | Largest peak-to-trough drawdown as a fraction (negative number).                                                                                                                              |
| [`turnover`](#newsmood.turnover)(positions)                                | Average <br/><br/>```<br/>|Δposition|<br/>```<br/><br/> per period — proxy for trading activity.                                                                                              |
| [`win_rate`](#newsmood.win_rate)(pnl)                                      | Fraction of strictly-positive P&L periods (zeros excluded).                                                                                                                                   |
| [`drop_high_correlation`](#newsmood.drop_high_correlation)(panel, \*, target[, ...])    | Drop one of each pair of features with `|corr| > threshold`.                                                                                                                                  |
| [`drop_low_variance`](#newsmood.drop_low_variance)(panel, \*[, target, ...])        | Drop features whose stdev is below `min_std`.                                                                                                                                                 |
| [`make_per_fold_mi_selector`](#newsmood.make_per_fold_mi_selector)([k, random_state])       | Return a closure `(train_df, target) -> [columns]` usable as `evaluate_walk_forward(..., feature_selector=...)`.                                                                              |
| [`sweep_alpha`](#newsmood.sweep_alpha)(panel, \*, target[, alphas, ...])      | Walk-forward Ridge sweep over a grid of `alpha` values.                                                                                                                                       |
| [`top_k_by_mutual_information`](#newsmood.top_k_by_mutual_information)(panel, \*, target)     | Keep the `k` features with highest mutual information against the target.                                                                                                                     |
| [`attach_targets_per_ticker`](#newsmood.attach_targets_per_ticker)(feature_panel, \*)       | Attach per-ticker forward returns to a `(ticker, session)`-indexed panel.                                                                                                                     |
| [`per_ticker_evaluate`](#newsmood.per_ticker_evaluate)(panel, \*[, target, ...])      | Walk-forward evaluate the model **per ticker**.                                                                                                                                               |
| [`per_ticker_panel`](#newsmood.per_ticker_panel)(news_df, \*[, universe, ...])     | Build a per-`(ticker, session)` feature panel from raw news.                                                                                                                                  |
| [`cached_embedding_store`](#newsmood.cached_embedding_store)(\*[, local, eager_remote])  | Build a [`CachedEmbeddingStore`](#newsmood.CachedEmbeddingStore) over the default local cache.                                                                                   |
| [`download_artifact`](#newsmood.download_artifact)(filename, \*[, repo_id, ...])    | Download an HF dataset artifact (cached).                                                                                                                                                     |
| [`hf_embedding_cache`](#newsmood.hf_embedding_cache)(\*[, repo_id, filename, ...])   | Return an [`EmbeddingsParquetView`](#newsmood.EmbeddingsParquetView) backed by an HF parquet.                                                                                     |
| [`hf_news_dataframe`](#newsmood.hf_news_dataframe)(\*[, repo_id, filename, ...])    | Load the canonical news DataFrame from an HF dataset.                                                                                                                                         |
| [`pack_all_to_staging`](#newsmood.pack_all_to_staging)(\*[, staging_dir, ...])        | Build every artifact under one staging directory and return paths.                                                                                                                            |
| [`pack_embeddings_to_parquet`](#newsmood.pack_embeddings_to_parquet)([source, dtype, ...])   | Pack a doc_id → vector mapping into a single parquet file.                                                                                                                                    |
| [`pack_news_to_parquet`](#newsmood.pack_news_to_parquet)(\*, out_path[, source, ...])  | Run the full ingest and write the canonical DataFrame as parquet.                                                                                                                             |
| [`pack_ohlcv_to_dir`](#newsmood.pack_ohlcv_to_dir)(\*[, ohlcv_root, verbose])       | Copy existing per-ticker OHLCV parquet files to a staging directory.                                                                                                                          |
| [`pack_raw_searches_to_tarball`](#newsmood.pack_raw_searches_to_tarball)(\*[, ...])            | Tar up the raw JSON snapshot tree (for full reproducibility).                                                                                                                                 |
| [`upload_staging_to_hf`](#newsmood.upload_staging_to_hf)(\*[, staging_dir, ...])       | Push a staging directory to an HF Dataset.                                                                                                                                                    |

### Classes

| [`WalkForwardResult`](#newsmood.WalkForwardResult)(target, per_fold, predictions)   | Output container for [`evaluate_walk_forward()`](#newsmood.evaluate_walk_forward).   |
|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| [`TickerMatcher`](#newsmood.TickerMatcher)(universe)                            | Pre-compiled matcher over a universe.                                                            |
| [`InferenceModel`](#newsmood.InferenceModel)(pipeline, feature_cols, target)     | A fitted Ridge pipeline plus the schema needed to apply it.                                      |
| [`BacktestResult`](#newsmood.BacktestResult)(equity, pnl, positions, cost)       |                                                                                                  |
| [`CachedEmbeddingStore`](#newsmood.CachedEmbeddingStore)([local, remote])              | Local-first, HF-backed embedding cache with write-through semantics.                             |
| [`EmbeddingsParquetView`](#newsmood.EmbeddingsParquetView)(path)                        | Read-only `MutableMapping`-style view over an embeddings parquet.                                |

### *class* newsmood.BacktestResult(equity, pnl, positions, cost, stats=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

### *class* newsmood.CachedEmbeddingStore(local=None, remote=None)

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

### *class* newsmood.EmbeddingsParquetView(path)

Bases: [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

Read-only `MutableMapping`-style view over an embeddings parquet.

Loads the entire (doc_id, vector) table into a dict on construction
(~1.5 GB RAM for 360k × 1536 float32 vectors). For lower-memory access,
use the underlying parquet directly.

```pycon
>>> view = EmbeddingsParquetView('embeddings.parquet')
>>> vec = view['<some_doc_id>']
```

### *class* newsmood.InferenceModel(pipeline, feature_cols, target, version='v1', metadata=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A fitted Ridge pipeline plus the schema needed to apply it.

#### pipeline

A fitted sklearn `Pipeline` (typically `StandardScaler → Ridge`).

* **Type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

#### feature_cols

Column names — both required and order-sensitive.

* **Type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### target

Target column the model was trained against (for provenance).

* **Type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### version

A free-form version string (e.g. `"v1"`, `"2026-05-15"`).

* **Type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### metadata

Anything else worth keeping: training row count, IC summary, etc.

* **Type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

#### predict(X)

Predict on a feature frame; missing feature columns raise.

* **Return type:**
  `ndarray`

### *class* newsmood.TickerMatcher(universe)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Pre-compiled matcher over a universe. `__call__` returns matched tickers.

```pycon
>>> matcher = TickerMatcher({'AAPL': ['Apple'], 'MSFT': ['Microsoft']})
>>> matcher('Apple beat estimates while Microsoft slipped.')
['AAPL', 'MSFT']
>>> matcher('$AAPL is up 2%')
['AAPL']
>>> matcher('No mentions here.')
[]
```

### *class* newsmood.WalkForwardResult(target, per_fold, predictions, feature_names=list())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Output container for [`evaluate_walk_forward()`](#newsmood.evaluate_walk_forward).

#### summary()

Aggregate metrics across folds (mean).

* **Return type:**
  `Series`

### newsmood.align_news_series_to_sessions(news_ts, sessions, , market_tz='America/New_York', close_time=datetime.time(16, 0))

Vectorized variant of [`align_news_to_session()`](#newsmood.align_news_to_session) over a Series.

Returns a Series of `date` objects (or NaT-equivalent: `None`) aligned
with `news_ts.index`.

* **Return type:**
  `Series`

### newsmood.align_news_to_session(news_ts, sessions, , market_tz='America/New_York', close_time=datetime.time(16, 0))

Return the anchor-session date for a single news timestamp.

`sessions` is an iterable of trading-day `date` objects in ascending order.
Returns `None` if no session lies on or after the news timestamp.

See module docstring for the alignment rule.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`date`](https://docs.python.org/3/library/datetime.html#datetime.date)]

### newsmood.annualized_sharpe(pnl, , periods_per_year=252)

Sharpe ratio = mean / std × √(periods_per_year). NaN if too few samples.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### newsmood.annualized_sortino(pnl, , periods_per_year=252)

Sortino = mean / downside-std × √(periods_per_year).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### newsmood.attach_cluster_id(df, model, , vec_col='vector', out_col='cluster')

Return a copy of `df` with cluster IDs in `out_col`.

Rows whose vector is missing get `-1`.

* **Return type:**
  `DataFrame`

### newsmood.attach_targets(news_df, , tickers=('SPY',), horizons=(1, 5, 20), start=None, end=None, cache_root='/home/runner/.config/newsmood/ohlcv', force_refresh=False, ts_col='ts', market_tz='America/New_York', close_time=datetime.time(16, 0))

Attach forward-return columns to `news_df` via session alignment.

For each ticker `T` and horizon `h`, a column `{T}_ret_{h}d` is added
holding the forward log return at the anchor session of each news row.
A `session` column is also added (the anchor session date) for the
first ticker — sessions for all tickers should agree within US equities.

* **Parameters:**
  * **news_df** (`DataFrame`) – DataFrame with at least a tz-aware `ts` column (UTC or any tz).
  * **tickers** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Universe to compute returns for. Default: `("SPY",)`.
  * **horizons** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Session-count horizons.
  * **start** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`date`](https://docs.python.org/3/library/datetime.html#datetime.date), [`datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – Optional bounds for fetching OHLCV. Defaults: span of `news_df.ts`.
  * **end** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`date`](https://docs.python.org/3/library/datetime.html#datetime.date), [`datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – Optional bounds for fetching OHLCV. Defaults: span of `news_df.ts`.
* **Return type:**
  `DataFrame`
* **Returns:**
  DataFrame with the original news rows plus added columns. Rows with no
  valid anchor session (e.g. `ts` past the available OHLCV history) are
  kept with NaN target columns — callers should drop NaNs before fitting.

### newsmood.attach_targets_per_ticker(feature_panel, , horizons=(1, 5), cache_root=None, ts_col='session')

Attach per-ticker forward returns to a `(ticker, session)`-indexed panel.

For each unique ticker in `feature_panel.index.get_level_values('ticker')`,
pulls its OHLCV and computes forward returns at the requested horizons,
then joins them onto the panel as columns `ret_<h>d`.

* **Return type:**
  `DataFrame`

### newsmood.attach_tickers(df, , universe={'AAPL': ['Apple'], 'ABBV': ['AbbVie'], 'ADBE': ['Adobe'], 'AMD': ['Advanced Micro Devices', 'AMD'], 'AMZN': ['Amazon'], 'AVGO': ['Broadcom'], 'BA': ['Boeing'], 'BAC': ['Bank of America'], 'BRK-B': ['Berkshire Hathaway', 'Berkshire'], 'C': ['Citigroup', 'Citi'], 'CAT': ['Caterpillar'], 'COIN': ['Coinbase'], 'COST': ['Costco'], 'CRM': ['Salesforce'], 'CVX': ['Chevron'], 'DIS': ['Disney', 'Walt Disney'], 'GE': ['GE Aerospace', 'General Electric'], 'GOOGL': ['Alphabet', 'Google'], 'GS': ['Goldman Sachs'], 'HD': ['Home Depot'], 'INTC': ['Intel'], 'IWM': ['Russell 2000 ETF', 'iShares Russell 2000'], 'JNJ': ['Johnson & Johnson'], 'JPM': ['JPMorgan', 'JP Morgan', 'JPMorgan Chase'], 'KO': ['Coca-Cola', 'Coca Cola'], 'LLY': ['Eli Lilly', 'Lilly'], 'MA': ['Mastercard'], 'MCD': ["McDonald's", 'McDonalds'], 'META': ['Meta Platforms', 'Meta', 'Facebook'], 'MRK': ['Merck'], 'MS': ['Morgan Stanley'], 'MSFT': ['Microsoft'], 'MSTR': ['MicroStrategy', 'Strategy'], 'NFLX': ['Netflix'], 'NKE': ['Nike'], 'NVDA': ['NVIDIA', 'Nvidia'], 'ORCL': ['Oracle'], 'PEP': ['PepsiCo', 'Pepsi'], 'PFE': ['Pfizer'], 'PG': ['Procter & Gamble', 'P&G'], 'QCOM': ['Qualcomm'], 'QQQ': ['Invesco QQQ', 'Nasdaq 100 ETF'], 'SBUX': ['Starbucks'], 'SPY': ['S&P 500 ETF', 'SPDR S&P 500'], 'T': ['AT&T'], 'TGT': ['Target Corp'], 'TMUS': ['T-Mobile'], 'TSLA': ['Tesla'], 'UNH': ['UnitedHealth', 'United Health'], 'V': ['Visa'], 'VZ': ['Verizon'], 'WFC': ['Wells Fargo'], 'WMT': ['Walmart', 'Wal-Mart'], 'XOM': ['ExxonMobil', 'Exxon']}, text_col='text_to_embed', out_col='tickers')

Return a copy of `df` with an added `tickers` column (list[str]).

Articles with no matches get an empty list.

* **Return type:**
  `DataFrame`

### newsmood.backtest(y_true, y_pred, , sizing='tanh', sizing_kwargs=None, cost_bps=1.0, periods_per_year=252)

Run a single-asset, single-signal backtest.

* **Parameters:**
  * **y_true** (`Series`) – Aligned Series of realized forward returns and model predictions
    (same index ordering).
  * **y_pred** (`Series`) – Aligned Series of realized forward returns and model predictions
    (same index ordering).
  * **sizing** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of a sizing function in `SIZING_FUNCS` (`"sign"`,
    `"tanh"`, `"decile"`) or a custom callable
    `(pd.Series) -> pd.Series`.
  * **sizing_kwargs** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Forwarded to the sizing function (e.g. `{"scale": 0.005}` for
    `tanh`).
  * **cost_bps** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Per-period transaction cost charged on 

    ```
    |Δposition|
    ```

     in basis points
    of NAV. `1.0` = 1bp = 0.01%. Default 1 bp is realistic for SPY-
    like ETFs at retail brokers.
  * **periods_per_year** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – For annualization.
* **Return type:**
  [`BacktestResult`](#newsmood.BacktestResult)

### newsmood.backtest_walk_forward(walk_forward_predictions, , y_true_col='y_true', y_pred_col='y_pred', sizing='tanh', sizing_kwargs=None, cost_bps=1.0, periods_per_year=252)

Backtest from the long-form predictions DataFrame produced by
[`newsmood.models.evaluate_walk_forward()`](newsmood.models.html.md#newsmood.models.evaluate_walk_forward).

The DataFrame must have a session-date index and the columns
`y_true`/`y_pred`. Concatenates all folds; cost is recomputed on the
full series (so fold boundaries pay turnover only if positions actually
flip there).

* **Return type:**
  [`BacktestResult`](#newsmood.BacktestResult)

### newsmood.build_seed_embeddings(seeds, , cache, model=None, embedder=None)

Embed each seed prompt (via the shared cache) — returns `{name: vector}`.

Uses [`newsmood.embed.embed_doc_ids()`](newsmood.embed.html.md#newsmood.embed.embed_doc_ids) so the cache is shared with the
news embeddings.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]

### newsmood.cached_embedding_store(, local=None, eager_remote=False)

Build a [`CachedEmbeddingStore`](#newsmood.CachedEmbeddingStore) over the default local cache.

* **Parameters:**
  * **local** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]]) – Pre-built local cache; defaults to
    [`newsmood.embed.default_embedding_cache()`](newsmood.embed.html.md#newsmood.embed.default_embedding_cache).
  * **eager_remote** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, fetch the HF parquet view immediately (one ~1 GB download
    on first call to a fresh machine). If False (default), defer until a
    miss occurs.
* **Return type:**
  [`CachedEmbeddingStore`](newsmood.data.html.md#newsmood.data.CachedEmbeddingStore)

### newsmood.default_embedding_cache(root='/home/runner/.config/newsmood/embeddings')

Return a pickle-coded dol Files store, creating the root if missing.

Keys are document ids; values are vectors (`list[float]`). Each entry
is stored in its own file, named `<doc_id>`.

* **Return type:**
  [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]

### newsmood.deterministic_dummy_embedder(, dim=16)

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

### newsmood.download_artifact(filename, , repo_id='thorwhalen/newsmood-data', repo_type='dataset', revision=None)

Download an HF dataset artifact (cached). Returns the local path.

Wraps `huggingface_hub.hf_hub_download()`; subsequent calls hit the
HF cache under `~/.cache/huggingface/hub/`.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### newsmood.drop_high_correlation(panel, , target, threshold=0.95, feature_cols=None)

Drop one of each pair of features with `|corr| > threshold`.

The dropped column is the one with the **smaller** absolute Pearson
correlation to the target, so we preserve the more target-relevant of
each redundant pair.

* **Return type:**
  `DataFrame`

### newsmood.drop_low_variance(panel, , target=None, min_std=1e-06, feature_cols=None)

Drop features whose stdev is below `min_std`.

The target column is preserved. If `feature_cols` is None, every column
other than `target` is treated as a feature.

* **Return type:**
  `DataFrame`

### newsmood.embed_dataframe(df, , cache, model=None, batch_size=512, embedder=None, on_batch=None, text_col='text_to_embed', id_col='doc_id', vector_col='vector')

Add a `vector` column to `df` by embedding `text_col` via cache.

A defensive copy is made; the input `df` is not mutated.

* **Return type:**
  `DataFrame`

### newsmood.embed_doc_ids(pairs, , cache, model=None, batch_size=512, embedder=None, on_batch=None)

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
    tests (see [`deterministic_dummy_embedder()`](#newsmood.deterministic_dummy_embedder)).
  * **on_batch** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`None`](https://docs.python.org/3/builtins/constants.html#None)]]) – Optional progress callback `(batch_index, batch_size) -> None`.
* **Returns:**
  Vectors for every input `doc_id`.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]

### newsmood.evaluate_walk_forward(panel, , target, feature_cols=None, n_splits=5, min_train=30, embargo=1, alpha=1.0, cost_bps=0.0, feature_selector=None)

Walk-forward Ridge baseline over a session-level panel.

The panel must be sorted by session ascending. Rows with NaN target or
feature values are dropped before training.

* **Parameters:**
  **feature_selector** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`DataFrame`, [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]]) – Optional callable `(train_df, target_name) -> list[selected_features]`.
  Called *inside each fold* on the **train slice only** — this avoids the
  look-ahead leakage that occurs when feature selection is run globally
  before splitting. The selector chooses among `feature_cols`.
* **Return type:**
  [`WalkForwardResult`](newsmood.models.html.md#newsmood.models.WalkForwardResult)

### newsmood.explode_to_ticker_rows(df, , ticker_col='tickers')

Explode a `tickers` list column to one row per (article, ticker).

Articles with zero matches are dropped. The new column is named `ticker`
(singular) for clarity at the panel level.

* **Return type:**
  `DataFrame`

### newsmood.fit_news_clusters(df, , n_clusters=64, vec_col='vector', sample_size=None, random_state=None, normalize=True, \*\*kmeans_kwargs)

Fit a K-Means codebook on `df[vec_col]`.

* **Parameters:**
  * **df** (`DataFrame`) – Must have `vec_col` of list-like vectors (rows with `None` are
    skipped).
  * **n_clusters** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Codebook size. 32-128 is typical; 64 is a reasonable default for
    ~hundreds-of-thousands of embeddings.
  * **sample_size** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – If set, randomly sample this many rows for fitting (full corpus is
    rarely needed; sampling speeds up KMeans dramatically).
  * **random_state** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Reproducibility seed.
  * **normalize** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – L2-normalize before fitting so Euclidean distance approximates cosine
    — appropriate for semantic embeddings.
  * **\*\*kmeans_kwargs** – Forwarded to `sklearn.cluster.KMeans`.
* **Returns:**
  Fitted model (callable on new vectors).

### newsmood.fit_predict_ridge(X_train, y_train, X_test, , alpha=1.0)

Standard-scale + Ridge fit, return test predictions.

* **Return type:**
  `ndarray`

### newsmood.forward_returns(prices, , horizons=(1, 5, 20), price_col='Close')

Compute forward log returns over the given session-count horizons.

The return at session `s` for horizon `h` is
`log(close[s + h] / close[s])`. Sessions near the tail with insufficient
forward data are `NaN`.

Returns a DataFrame indexed like `prices`, with one column per horizon
named `ret_<h>d`.

* **Return type:**
  `DataFrame`

### newsmood.get_ohlcv(ticker, , start='2020-01-01', end=None, cache_root='/home/runner/.config/newsmood/ohlcv', force_refresh=False, auto_adjust=True)

Return a daily OHLCV DataFrame for `ticker`, with a parquet cache.

The cache is keyed by ticker and stores the full available history; on
subsequent calls a slice is returned without hitting the network unless
`force_refresh=True` or the requested `end` extends beyond cached data.

* **Return type:**
  `DataFrame`
* **Returns:**
  DataFrame with columns `[Open, High, Low, Close, Volume]` and a
  `DatetimeIndex` of session dates (timezone-naive, US/Eastern in spirit).

### newsmood.hf_embedding_cache(, repo_id='thorwhalen/newsmood-data', filename='embeddings.parquet', revision=None)

Return an [`EmbeddingsParquetView`](#newsmood.EmbeddingsParquetView) backed by an HF parquet.

Downloads the parquet to the local HF cache on first call; subsequent
calls reuse the cached file.

* **Return type:**
  [`EmbeddingsParquetView`](newsmood.data.html.md#newsmood.data.EmbeddingsParquetView)

### newsmood.hf_news_dataframe(, repo_id='thorwhalen/newsmood-data', filename='news.parquet', revision=None)

Load the canonical news DataFrame from an HF dataset.

* **Return type:**
  `DataFrame`

### newsmood.hit_rate_sign(y_true, y_pred)

Fraction of rows where `sign(y_pred) == sign(y_true)` (zeros skipped).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### newsmood.information_coefficient(y_true, y_pred)

Pearson IC. `NaN`-tolerant; returns `nan` if insufficient data.

```pycon
>>> round(information_coefficient([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 6)
1.0
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### newsmood.ingest_searches(store, , source=None, since=None, until=None, dedupe=True)

Read a searches store and return a canonical `DataFrame`.

* **Parameters:**
  * **store** – Typically `dol.JsonFiles(DEFAULT_SEARCHES_ROOT)`.
  * **source** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional source filter; one of `SUPPORTED_SOURCES`.
  * **since** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional date bounds (`"YYYY-MM-DD"`), inclusive/exclusive.
  * **until** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional date bounds (`"YYYY-MM-DD"`), inclusive/exclusive.
  * **dedupe** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True (default), drop duplicate rows sharing the same `doc_id`
    keeping the earliest `ts`.
* **Returns:**
  Columns: `CANONICAL_COLUMNS`. `ts` is a tz-aware datetime
  (UTC). Rows with no usable text or no `ts` are dropped.
* **Return type:**
  `DataFrame`

### newsmood.iter_canonical_rows(store, , source=None, since=None, until=None)

Iterate canonical rows from a `dol.JsonFiles`-style store.

* **Parameters:**
  * **store** – A `MutableMapping` whose keys look like
    `"<source>/<YYYY-MM-DD>/<filename>"` and whose values are parsed JSON.
  * **source** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – If given, restrict to one source (must be a key of `PARSERS`).
  * **since** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional `"YYYY-MM-DD"` date filters (inclusive `since`, exclusive
    `until`), evaluated against the file date — fast key-level pruning.
  * **until** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional `"YYYY-MM-DD"` date filters (inclusive `since`, exclusive
    `until`), evaluated against the file date — fast key-level pruning.
* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### newsmood.load_model(model_path)

Load an InferenceModel produced by [`save_model()`](#newsmood.save_model).

* **Return type:**
  [`InferenceModel`](newsmood.infer.html.md#newsmood.infer.InferenceModel)

### newsmood.long_short_sharpe(y_true, y_pred, , cost_bps=0.0)

Sharpe of a sign-based long-short portfolio.

For each session, go long if `y_pred > 0`, short if `y_pred < 0`, scale
by 1; realized return is `sign(y_pred) * y_true`. Optional per-trade
transaction cost in basis points is subtracted when the sign flips.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### newsmood.make_doc_id(text, , hash_len=16)

SHA-1-based, lowercase-normalized content hash.

```pycon
>>> make_doc_id('Hello, World!') == make_doc_id('hello, world!')
True
>>> len(make_doc_id('x'))
16
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### newsmood.make_news_collection(name='news', , backend='memory', embedder=None, ts_field='ts', \*\*backend_kwargs)

Create (or reopen) a `vd.TimeIndexedCollection` for news.

For memory and chroma backends, idempotently obtains the underlying
collection — `get` if it exists, else `create`.

### newsmood.make_per_fold_mi_selector(k=20, , random_state=0)

Return a closure `(train_df, target) -> [columns]` usable as
`evaluate_walk_forward(..., feature_selector=...)`.

Per-fold MI selection — runs only on the **train slice**, so OOS folds
can’t leak through feature choice. This is the leakage-free counterpart
to [`top_k_by_mutual_information()`](#newsmood.top_k_by_mutual_information) (which is global).

### newsmood.make_text_to_embed(title, description)

Build the canonical text used for embedding.

Concatenates title + “. “ + description, normalizes whitespace, strips
paywall placeholders.

```pycon
>>> make_text_to_embed('Earnings miss', 'Company X reported weak Q3.')
'Earnings miss. Company X reported weak Q3.'
>>> make_text_to_embed('Just title', None)
'Just title'
>>> make_text_to_embed(None, 'Just desc')
'Just desc'
>>> make_text_to_embed('t', 'ONLY AVAILABLE IN PAID PLANS')
't'
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### newsmood.match_tickers(text, , universe)

One-shot helper; prefer [`TickerMatcher`](#newsmood.TickerMatcher) for many calls.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### newsmood.max_drawdown(equity)

Largest peak-to-trough drawdown as a fraction (negative number).

```pycon
>>> import pandas as pd
>>> e = pd.Series([1.0, 1.5, 1.2, 0.9, 1.3])
>>> round(max_drawdown(e), 4)
-0.4
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### newsmood.news_pipeline(df, , cache=None, collection_name='news', backend='memory', model=None, embedder=None, instance_embedder=None, batch_size=512, on_batch=None)

Single-call: embed a DataFrame and load it into a vd collection.

Returns the populated `vd.TimeIndexedCollection`. Uses
`deterministic_dummy_embedder` as the collection’s at-query embedding
function unless `instance_embedder` is given — keep that in mind when
calling `search()`/`search_window` later.

### newsmood.open_mall(root=None)

Open a persistent mall — a dol-backed JSON store at `root`.

Each top-level key (e.g. `features:news_embed_v1`) maps to a JSON-blob
file at `root`. Nested entries are JSON-coded `[ticker, session-iso] -> float`
pairs.

For a quick in-process mall, just pass a `dict()` everywhere — this
helper is for daemon-style daily inference jobs that need to persist.

* **Return type:**
  [`MutableMapping`](https://docs.python.org/3/library/typing.html#typing.MutableMapping)

### newsmood.pack_all_to_staging(, staging_dir='/home/runner/.cache/newsmood/staging', include_raw=True, include_ohlcv=True, verbose=True)

Build every artifact under one staging directory and return paths.

Caller then uploads `staging_dir` to the HF Hub via
[`upload_staging_to_hf()`](#newsmood.upload_staging_to_hf).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### newsmood.pack_embeddings_to_parquet(source=None, , out_path, dtype='float32', chunk_size=25000, verbose=True)

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

### newsmood.pack_news_to_parquet(, out_path, source=None, searches_root=None, since=None, until=None, verbose=True)

Run the full ingest and write the canonical DataFrame as parquet.

See [`newsmood.ingest.ingest_searches()`](newsmood.ingest.html.md#newsmood.ingest.ingest_searches) for parameters.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### newsmood.pack_ohlcv_to_dir(, ohlcv_root=None, out_dir, verbose=True)

Copy existing per-ticker OHLCV parquet files to a staging directory.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### newsmood.pack_raw_searches_to_tarball(, searches_root=None, out_path, compression='gz', verbose=True)

Tar up the raw JSON snapshot tree (for full reproducibility).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### newsmood.panel_with_targets(news_df, , seed_vecs=None, target_cols=None, session_col='session', vec_col='vector', query_col='query', include_mean_vector_components=False, top_queries=None)

Join per-session features with per-session targets.

Targets are computed as the **first** observation of each target column
per session (returns at a session are session-level, so any row in the
group has the same value).

* **Return type:**
  `DataFrame`

### newsmood.parse_key(key)

Extract `date`, `file_ts`, and `query` from a store key.

```pycon
>>> parse_key('2025-03-13/2025-03-13--18-00-04__Profit_Warning.json')['query']
'Profit_Warning'
>>> parse_key('2025-03-13/2025-03-13--18-00-04__.json')['query']
''
>>> ts = parse_key('2025-03-13/2025-03-13--18-00-04__Profit_Warning.json')['file_ts']
>>> ts.isoformat()
'2025-03-13T18:00:04+00:00'
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

### newsmood.per_ticker_evaluate(panel, , target='ret_1d', feature_cols=None, n_splits=5, min_train=30, embargo=1, alpha=1.0, cost_bps=1.0)

Walk-forward evaluate the model **per ticker**.

Returns a DataFrame indexed by ticker with per-ticker summary metrics
(mean IC, rank IC, sign accuracy, Sharpe, OOS Sharpe-after-cost) plus
sample counts.

* **Return type:**
  `DataFrame`

### newsmood.per_ticker_panel(news_df, , universe=None, seed_vecs=None, min_articles_per_ticker=50, text_col='text_to_embed')

Build a per-`(ticker, session)` feature panel from raw news.

* **Parameters:**
  * **news_df** (`DataFrame`) – Must include `doc_id`, `ts`, `vector`, `text_to_embed` (or
    whatever `text_col` is), and `session` (if not, call
    [`newsmood.targets.attach_targets()`](newsmood.targets.html.md#newsmood.targets.attach_targets) first).
  * **universe** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]]) – Ticker → name-variants mapping. Default: `DEFAULT_UNIVERSE`.
  * **seed_vecs** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]]) – Pre-computed seed embeddings (use
    [`newsmood.features.build_seed_embeddings()`](newsmood.features.html.md#newsmood.features.build_seed_embeddings)).
  * **min_articles_per_ticker** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Tickers with fewer than this many article matches across the whole
    period are dropped — too sparse to model.
  * **text_col** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Column used for ticker matching (regex over text).
* **Return type:**
  `DataFrame`
* **Returns:**
  DataFrame indexed by `(ticker, session)` with feature columns.

### newsmood.per_ticker_session_features(article_panel, , seed_vecs=None, session_col='session', ticker_col='ticker', vec_col='vector', query_col='query', top_queries=None, include_mean_vector_components=False)

Session-by-ticker features.

`article_panel` is an exploded DataFrame (one row per article-ticker
pair) with `session`, `ticker`, and `vector` columns. Aggregation
is identical to [`newsmood.features.session_features()`](newsmood.features.html.md#newsmood.features.session_features) but grouped by
`(ticker, session)`.

* **Return type:**
  `DataFrame`

### newsmood.populate_collection(df, , collection, vector_col='vector', id_col='doc_id', text_col='text_to_embed', ts_col='ts', metadata_fields=None)

Write `df` rows into a `TimeIndexedCollection`. Returns rows written.

The `ts` column is stamped into metadata under the collection’s
`ts_field`; additional metadata fields are copied from `df`.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### newsmood.predict_scores(model, feature_df)

Apply model to feature_df; return Series indexed like feature_df.

* **Return type:**
  `Series`

### newsmood.rank_information_coefficient(y_true, y_pred)

Spearman IC. `NaN`-tolerant.

```pycon
>>> round(rank_information_coefficient([1, 2, 3], [3, 2, 1]), 6)
-1.0
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### newsmood.run_inference(feature_df, , model, mall, ticker=None, feature_key='features:news_embed_v1')

End-to-end: predict on `feature_df` and write to `mall`.

Returns the number of mall entries written.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### newsmood.save_model(model, model_path)

Persist an InferenceModel to disk (pickle).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### newsmood.session_cluster_counts(df, , n_clusters, session_col='session', cluster_col='cluster', column_prefix='clust_')

Per-session counts of articles in each cluster (zero-padded across all clusters).

Returns a DataFrame indexed by session with `n_clusters` columns:
`{column_prefix}0` … `{column_prefix}{n_clusters-1}`.

* **Return type:**
  `DataFrame`

### newsmood.session_features(news_df, , seed_vecs=None, session_col='session', vec_col='vector', query_col='query', include_mean_vector_components=False, top_queries=None)

Group `news_df` by session, return one row per session.

* **Parameters:**
  * **news_df** (`DataFrame`) – Must include `session_col`, `vec_col` (list-like vectors), and
    optionally `query_col`.
  * **seed_vecs** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]]) – Mapping `{seed_name: vector}`. If given, a `cos_{seed_name}`
    column is added per seed (cosine of the session’s mean embedding to
    the seed).
  * **include_mean_vector_components** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, also include `mean_vec_0 ... mean_vec_{d-1}` columns
    (mean embedding broadcast as scalar features). Off by default — the
    cosines are usually more interpretable and parsimonious.
  * **top_queries** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – If given, only count articles in this set of queries (else: counts
    every query observed).
* **Return type:**
  `DataFrame`
* **Returns:**
  DataFrame indexed by session (dates), one row per session.

### newsmood.sign_accuracy(y_true, y_pred)

Fraction of predictions whose sign matches the true sign. Zero-true rows skipped.

```pycon
>>> sign_accuracy([1, -1, 1], [0.5, -0.2, 0.3])
1.0
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### newsmood.sweep_alpha(panel, , target, alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0), feature_cols=None, n_splits=5, min_train=30, embargo=5)

Walk-forward Ridge sweep over a grid of `alpha` values.

Returns a DataFrame indexed by alpha with mean and std of IC, rank-IC,
sign accuracy, and Sharpe across folds.

* **Return type:**
  `DataFrame`

### newsmood.top_k_by_mutual_information(panel, , target, k=20, feature_cols=None, random_state=0)

Keep the `k` features with highest mutual information against the target.

Uses `sklearn.feature_selection.mutual_info_regression`. Robust to
monotonic non-linearities; ties broken by original column order.

* **Return type:**
  `DataFrame`

### newsmood.train_and_save(panel, , target, feature_cols=None, alpha=1.0, model_path=None, version='v1', metadata=None)

Fit on the full panel (no CV; CV is for evaluation) and persist.

The CV pipeline lives in [`newsmood.models.evaluate_walk_forward()`](newsmood.models.html.md#newsmood.models.evaluate_walk_forward) —
that’s where you decide *whether* the model has signal. Once you have
chosen `alpha` / feature set there, run *this* function on the same
panel to produce the operational artifact.

* **Return type:**
  [`InferenceModel`](newsmood.infer.html.md#newsmood.infer.InferenceModel)

### newsmood.turnover(positions)

Average 

```
|Δposition|
```

 per period — proxy for trading activity.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### newsmood.upload_staging_to_hf(, staging_dir='/home/runner/.cache/newsmood/staging', repo_id='thorwhalen/newsmood-data', commit_message='Update artifacts', private=False, token=None, verbose=True)

Push a staging directory to an HF Dataset.

Requires a write-scoped HF token. Resolution order:

1. Explicit `token=` arg.
2. `HF_WRITE_TOKEN` env var.
3. `HF_TOKEN` env var (only works if it happens to have write scope).

Returns the dataset URL.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### newsmood.walk_forward_splits(n, , n_splits=5, min_train=30, embargo=1, test_size=None)

Yield `(train_idx, test_idx)` pairs for expanding-window CV.

Each test fold begins `embargo` indices *after* the last train index, so
the forward-return horizon — which makes label `r_t` peek at price
`t + h` — cannot leak across the train/test boundary as long as
`embargo >= h`.

* **Parameters:**
  * **n** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Total number of (already chronologically-ordered) rows.
  * **n_splits** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – How many folds to produce.
  * **min_train** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Minimum number of rows in the first training fold.
  * **embargo** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Gap (in rows / sessions) between train and test.
  * **test_size** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Size of each test fold. Default: `(n - min_train) // n_splits`.
  * **list****(****walk_forward_splits****(****20** ( *>>> splits =*)
  * **n_splits=3**
  * **min_train=8**
  * **embargo=1**
  * **test_size=3****)****)**
  * **len****(****splits****)** ( *>>>*)
  * **3**
  * **splits****[****0****]****[****1****]****.tolist****(****)** ( *>>>*)
  * **[****9**
  * **10**
  * **11****]**
* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/typing.html#typing.Iterator)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, `ndarray`]]

### newsmood.win_rate(pnl)

Fraction of strictly-positive P&L periods (zeros excluded).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### newsmood.write_predictions_to_mall(predictions, , mall, ticker=None, feature_key='features:news_embed_v1')

Write per-(ticker, session) scores into `mall[feature_key]`.

Two index shapes are supported:

- Single Index of session dates: `ticker` must be provided; every entry
  is keyed `(ticker, session)`.
- MultiIndex of `(ticker, session)`: `ticker` is ignored.

`mall[feature_key]` is created (as a plain `dict`) if absent, or
updated in place if present.

Returns the number of entries written.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### Modules

| [`clusters`](newsmood.clusters.html.md#module-newsmood.clusters)               | Cluster-bucket features.                                              |
|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| [`data`](newsmood.data.html.md#module-newsmood.data)                       | Pack local artifacts for publishing and download-on-miss from HF.     |
| [`embed`](newsmood.embed.html.md#module-newsmood.embed)                     | Embed canonical-DataFrame rows and load them into a vd collection.    |
| [`features`](newsmood.features.html.md#module-newsmood.features)               | Per-session features for news-driven prediction.                      |
| [`infer`](newsmood.infer.html.md#module-newsmood.infer)                     | Train, persist, and apply a news-embedding signal model.              |
| [`ingest`](newsmood.ingest.html.md#module-newsmood.ingest)                   | Ingest news from the local searches store into a canonical DataFrame. |
| [`models`](newsmood.models.html.md#module-newsmood.models)                   | Baseline walk-forward modeling on the news-feature panel.             |
| [`per_ticker_eval`](newsmood.per_ticker_eval.html.md#module-newsmood.per_ticker_eval) | End-to-end per-ticker evaluation.                                     |
| [`select`](newsmood.select.html.md#module-newsmood.select)                   | Feature selection + regularization tuning for the news panel.         |
| [`targets`](newsmood.targets.html.md#module-newsmood.targets)                 | Price targets for news-driven prediction.                             |
| [`tickers`](newsmood.tickers.html.md#module-newsmood.tickers)                 | Per-ticker article matching: lightweight regex-based NER.             |
