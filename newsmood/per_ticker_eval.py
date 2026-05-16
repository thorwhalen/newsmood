"""End-to-end per-ticker evaluation.

Given a news DataFrame with embeddings, this module:

1. Attaches a ``tickers`` list column (via :func:`newsmood.tickers.attach_tickers`).
2. Explodes to one row per ``(article, ticker)``.
3. Computes per-``(ticker, session)`` features (counts, seed-cosines, query
   counts) via :func:`newsmood.tickers.per_ticker_session_features`.
4. Fetches OHLCV + forward returns for the ticker universe.
5. Joins targets to the panel and walk-forward-evaluates **per ticker** —
   returning a DataFrame summarising IC / hit-rate / Sharpe per name.

This is intentionally one orchestrator function: ``per_ticker_evaluate``.
Use it for quick comparisons across the universe; for finer control wire
the building-blocks yourself.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def per_ticker_panel(
    news_df: pd.DataFrame,
    *,
    universe: Optional[Mapping[str, Iterable[str]]] = None,
    seed_vecs: Optional[Mapping[str, list[float]]] = None,
    min_articles_per_ticker: int = 50,
    text_col: str = "text_to_embed",
) -> pd.DataFrame:
    """Build a per-``(ticker, session)`` feature panel from raw news.

    Parameters
    ----------
    news_df
        Must include ``doc_id``, ``ts``, ``vector``, ``text_to_embed`` (or
        whatever ``text_col`` is), and ``session`` (if not, call
        :func:`newsmood.targets.attach_targets` first).
    universe
        Ticker → name-variants mapping. Default: ``DEFAULT_UNIVERSE``.
    seed_vecs
        Pre-computed seed embeddings (use
        :func:`newsmood.features.build_seed_embeddings`).
    min_articles_per_ticker
        Tickers with fewer than this many article matches across the whole
        period are dropped — too sparse to model.
    text_col
        Column used for ticker matching (regex over text).

    Returns
    -------
    DataFrame indexed by ``(ticker, session)`` with feature columns.
    """
    from newsmood.tickers import (
        DEFAULT_UNIVERSE,
        attach_tickers,
        explode_to_ticker_rows,
        per_ticker_session_features,
    )

    if "session" not in news_df.columns:
        raise ValueError("news_df must include a 'session' column; call attach_targets first")
    if "vector" not in news_df.columns:
        raise ValueError("news_df must include a 'vector' column")

    if universe is None:
        universe = DEFAULT_UNIVERSE

    df_t = attach_tickers(news_df, universe=universe, text_col=text_col)
    exploded = explode_to_ticker_rows(df_t)
    if exploded.empty:
        return pd.DataFrame()

    # Drop sparsely-mentioned tickers
    counts = exploded["ticker"].value_counts()
    keep_tickers = set(counts[counts >= min_articles_per_ticker].index)
    exploded = exploded[exploded["ticker"].isin(keep_tickers)].copy()
    if exploded.empty:
        return pd.DataFrame()

    panel = per_ticker_session_features(
        exploded, seed_vecs=seed_vecs, session_col="session", ticker_col="ticker"
    )
    return panel


def attach_targets_per_ticker(
    feature_panel: pd.DataFrame,
    *,
    horizons: Sequence[int] = (1, 5),
    cache_root: Optional[str] = None,
    ts_col: str = "session",
) -> pd.DataFrame:
    """Attach per-ticker forward returns to a ``(ticker, session)``-indexed panel.

    For each unique ticker in ``feature_panel.index.get_level_values('ticker')``,
    pulls its OHLCV and computes forward returns at the requested horizons,
    then joins them onto the panel as columns ``ret_<h>d``.
    """
    from newsmood.targets import (
        DEFAULT_OHLCV_CACHE,
        forward_returns,
        get_ohlcv,
    )

    if feature_panel.empty:
        return feature_panel.copy()

    if cache_root is None:
        cache_root = DEFAULT_OHLCV_CACHE

    panel = feature_panel.copy()
    tickers = sorted(set(panel.index.get_level_values("ticker")))
    sessions = sorted(set(panel.index.get_level_values("session")))
    if not sessions:
        return panel
    start = min(sessions)
    end = max(sessions) + pd.Timedelta(days=max(horizons) + 5)

    rets_by_ticker: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            prices = get_ohlcv(
                t, start=start, end=end.date() if hasattr(end, "date") else end,
                cache_root=cache_root,
            )
        except Exception:
            continue
        if prices.empty:
            continue
        rets = forward_returns(prices, horizons=horizons)
        rets.index = pd.to_datetime(rets.index).date  # convert to date
        rets_by_ticker[t] = rets

    for h in horizons:
        col = f"ret_{h}d"
        new_col_values = []
        for (tkr, sess) in panel.index:
            rets = rets_by_ticker.get(tkr)
            if rets is None or sess not in rets.index:
                new_col_values.append(np.nan)
                continue
            new_col_values.append(float(rets.loc[sess, col]) if col in rets.columns else np.nan)
        panel[col] = new_col_values
    return panel


def per_ticker_evaluate(
    panel: pd.DataFrame,
    *,
    target: str = "ret_1d",
    feature_cols: Optional[Iterable[str]] = None,
    n_splits: int = 5,
    min_train: int = 30,
    embargo: int = 1,
    alpha: float = 1.0,
    cost_bps: float = 1.0,
) -> pd.DataFrame:
    """Walk-forward evaluate the model **per ticker**.

    Returns a DataFrame indexed by ticker with per-ticker summary metrics
    (mean IC, rank IC, sign accuracy, Sharpe, OOS Sharpe-after-cost) plus
    sample counts.
    """
    from newsmood.backtest import backtest_walk_forward
    from newsmood.models import evaluate_walk_forward

    if panel.empty:
        return pd.DataFrame()
    if "ticker" not in panel.index.names:
        raise ValueError("panel must have a MultiIndex ('ticker', 'session')")

    rows: list[dict] = []
    for ticker, sub in panel.groupby(level="ticker"):
        sub = sub.copy()
        sub.index = sub.index.droplevel("ticker")  # index becomes plain session
        res = evaluate_walk_forward(
            sub,
            target=target,
            feature_cols=feature_cols,
            n_splits=n_splits,
            min_train=min_train,
            embargo=embargo,
            alpha=alpha,
        )
        if res.per_fold.empty:
            rows.append({"ticker": ticker, "n_sessions": len(sub), "n_folds": 0})
            continue
        bt = backtest_walk_forward(res.predictions, sizing="tanh", cost_bps=cost_bps)
        rows.append(
            {
                "ticker": ticker,
                "n_sessions": len(sub),
                "n_folds": int(len(res.per_fold)),
                "mean_ic": float(res.per_fold["ic"].mean()),
                "mean_rank_ic": float(res.per_fold["rank_ic"].mean()),
                "mean_sign_acc": float(res.per_fold["sign_acc"].mean()),
                "bt_sharpe": float(bt.stats["sharpe"]),
                "bt_max_dd": float(bt.stats["max_drawdown"]),
                "bt_final_equity": float(bt.stats["final_equity"]),
                "bt_hit_rate": float(bt.stats["hit_rate"]),
                "bt_turnover": float(bt.stats["turnover"]),
            }
        )
    return pd.DataFrame(rows).set_index("ticker").sort_values("mean_ic", ascending=False)


__all__ = [
    "per_ticker_panel",
    "attach_targets_per_ticker",
    "per_ticker_evaluate",
]
