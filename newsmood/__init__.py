"""newsmood — news-embedding trading signals.

The pipeline is staged across modules:

- :mod:`newsmood.ingest` — read raw news from the local searches store and
  return a canonical pandas DataFrame.
- :mod:`newsmood.embed` (planned) — embed text into a
  :class:`vd.TimeIndexedCollection` with content-hash caching.
- :mod:`newsmood.targets` (planned) — OHLCV + forward returns + session
  alignment (strict, leakage-safe).
- :mod:`newsmood.features` (planned) — per-(target, day) features:
  window mean embeddings, seed-prompt cosines, news volume.
- :mod:`newsmood.models` (planned) — walk-forward baselines (Ridge, GBM)
  with IC/rank-IC reporting.
"""

from newsmood.ingest import (
    DEFAULT_SEARCHES_ROOT,
    SUPPORTED_SOURCES,
    CANONICAL_COLUMNS,
    ingest_searches,
    iter_canonical_rows,
    make_doc_id,
    make_text_to_embed,
    parse_key,
)
from newsmood.embed import (
    DEFAULT_CACHE_ROOT,
    default_embedding_cache,
    deterministic_dummy_embedder,
    embed_dataframe,
    embed_doc_ids,
    make_news_collection,
    news_pipeline,
    populate_collection,
)
from newsmood.targets import (
    DEFAULT_HORIZONS,
    DEFAULT_OHLCV_CACHE,
    DEFAULT_TICKERS,
    align_news_series_to_sessions,
    align_news_to_session,
    attach_targets,
    forward_returns,
    get_ohlcv,
)
from newsmood.features import (
    DEFAULT_SEED_PROMPTS,
    build_seed_embeddings,
    panel_with_targets,
    session_features,
)
from newsmood.models import (
    WalkForwardResult,
    evaluate_walk_forward,
    fit_predict_ridge,
    information_coefficient,
    long_short_sharpe,
    rank_information_coefficient,
    sign_accuracy,
    walk_forward_splits,
)
from newsmood.tickers import (
    DEFAULT_UNIVERSE,
    TickerMatcher,
    attach_tickers,
    explode_to_ticker_rows,
    match_tickers,
    per_ticker_session_features,
)
from newsmood.clusters import (
    attach_cluster_id,
    fit_news_clusters,
    session_cluster_counts,
)
from newsmood.infer import (
    DEFAULT_FEATURE_KEY,
    DEFAULT_MODEL_ROOT,
    InferenceModel,
    load_model,
    open_mall,
    predict_scores,
    run_inference,
    save_model,
    train_and_save,
    write_predictions_to_mall,
)
from newsmood.backtest import (
    BacktestResult,
    annualized_sharpe,
    annualized_sortino,
    backtest,
    backtest_walk_forward,
    hit_rate_sign,
    max_drawdown,
    turnover,
    win_rate,
)
from newsmood.select import (
    drop_high_correlation,
    drop_low_variance,
    make_per_fold_mi_selector,
    sweep_alpha,
    top_k_by_mutual_information,
)
from newsmood.per_ticker_eval import (
    attach_targets_per_ticker,
    per_ticker_evaluate,
    per_ticker_panel,
)
from newsmood.data import (
    DEFAULT_HF_REPO_ID,
    DEFAULT_STAGING_DIR,
    CachedEmbeddingStore,
    EmbeddingsParquetView,
    cached_embedding_store,
    download_artifact,
    hf_embedding_cache,
    hf_news_dataframe,
    pack_all_to_staging,
    pack_embeddings_to_parquet,
    pack_news_to_parquet,
    pack_ohlcv_to_dir,
    pack_raw_searches_to_tarball,
)

__all__ = [
    # ingest
    "DEFAULT_SEARCHES_ROOT",
    "SUPPORTED_SOURCES",
    "CANONICAL_COLUMNS",
    "ingest_searches",
    "iter_canonical_rows",
    "make_doc_id",
    "make_text_to_embed",
    "parse_key",
    # embed
    "DEFAULT_CACHE_ROOT",
    "default_embedding_cache",
    "deterministic_dummy_embedder",
    "embed_dataframe",
    "embed_doc_ids",
    "make_news_collection",
    "news_pipeline",
    "populate_collection",
    # targets
    "DEFAULT_HORIZONS",
    "DEFAULT_OHLCV_CACHE",
    "DEFAULT_TICKERS",
    "align_news_series_to_sessions",
    "align_news_to_session",
    "attach_targets",
    "forward_returns",
    "get_ohlcv",
    # features
    "DEFAULT_SEED_PROMPTS",
    "build_seed_embeddings",
    "panel_with_targets",
    "session_features",
    # models
    "WalkForwardResult",
    "evaluate_walk_forward",
    "fit_predict_ridge",
    "information_coefficient",
    "long_short_sharpe",
    "rank_information_coefficient",
    "sign_accuracy",
    "walk_forward_splits",
    # tickers
    "DEFAULT_UNIVERSE",
    "TickerMatcher",
    "attach_tickers",
    "explode_to_ticker_rows",
    "match_tickers",
    "per_ticker_session_features",
    # clusters
    "attach_cluster_id",
    "fit_news_clusters",
    "session_cluster_counts",
    # infer
    "DEFAULT_FEATURE_KEY",
    "DEFAULT_MODEL_ROOT",
    "InferenceModel",
    "load_model",
    "open_mall",
    "predict_scores",
    "run_inference",
    "save_model",
    "train_and_save",
    "write_predictions_to_mall",
    # backtest
    "BacktestResult",
    "annualized_sharpe",
    "annualized_sortino",
    "backtest",
    "backtest_walk_forward",
    "hit_rate_sign",
    "max_drawdown",
    "turnover",
    "win_rate",
    # select
    "drop_high_correlation",
    "drop_low_variance",
    "make_per_fold_mi_selector",
    "sweep_alpha",
    "top_k_by_mutual_information",
    # per-ticker
    "attach_targets_per_ticker",
    "per_ticker_evaluate",
    "per_ticker_panel",
    # data publishing/consumption
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
]
