# newsmood.per_ticker_eval

End-to-end per-ticker evaluation.

Given a news DataFrame with embeddings, this module:

1. Attaches a `tickers` list column (via [`newsmood.tickers.attach_tickers()`](newsmood.tickers.html.md#newsmood.tickers.attach_tickers)).
2. Explodes to one row per `(article, ticker)`.
3. Computes per-`(ticker, session)` features (counts, seed-cosines, query
   counts) via [`newsmood.tickers.per_ticker_session_features()`](newsmood.tickers.html.md#newsmood.tickers.per_ticker_session_features).
4. Fetches OHLCV + forward returns for the ticker universe.
5. Joins targets to the panel and walk-forward-evaluates **per ticker** —
   returning a DataFrame summarising IC / hit-rate / Sharpe per name.

This is intentionally one orchestrator function: `per_ticker_evaluate`.
Use it for quick comparisons across the universe; for finer control wire
the building-blocks yourself.

### Functions

| [`per_ticker_panel`](#newsmood.per_ticker_eval.per_ticker_panel)(news_df, \*[, universe, ...])   | Build a per-`(ticker, session)` feature panel from raw news.              |
|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [`attach_targets_per_ticker`](#newsmood.per_ticker_eval.attach_targets_per_ticker)(feature_panel, \*)     | Attach per-ticker forward returns to a `(ticker, session)`-indexed panel. |
| [`per_ticker_evaluate`](#newsmood.per_ticker_eval.per_ticker_evaluate)(panel, \*[, target, ...])    | Walk-forward evaluate the model **per ticker**.                           |

### newsmood.per_ticker_eval.attach_targets_per_ticker(feature_panel, , horizons=(1, 5), cache_root=None, ts_col='session')

Attach per-ticker forward returns to a `(ticker, session)`-indexed panel.

For each unique ticker in `feature_panel.index.get_level_values('ticker')`,
pulls its OHLCV and computes forward returns at the requested horizons,
then joins them onto the panel as columns `ret_<h>d`.

* **Return type:**
  `DataFrame`

### newsmood.per_ticker_eval.per_ticker_evaluate(panel, , target='ret_1d', feature_cols=None, n_splits=5, min_train=30, embargo=1, alpha=1.0, cost_bps=1.0)

Walk-forward evaluate the model **per ticker**.

Returns a DataFrame indexed by ticker with per-ticker summary metrics
(mean IC, rank IC, sign accuracy, Sharpe, OOS Sharpe-after-cost) plus
sample counts.

* **Return type:**
  `DataFrame`

### newsmood.per_ticker_eval.per_ticker_panel(news_df, , universe=None, seed_vecs=None, min_articles_per_ticker=50, text_col='text_to_embed')

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
